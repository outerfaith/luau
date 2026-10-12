// This file is part of the Luau programming language and is licensed under MIT License; see LICENSE.txt for details
#include "JitContext.h"

#include "BytecodeAnalysis.h"
#include "BytecodeTranslation.h"
#include "VmTypes.h"

#include "Luau/Common.h"
#include "Luau/JitCompiler.h"

#include "lapi.h"
#include "lmem.h"
#include "lstate.h"

#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/TargetParser/Host.h>

LUAU_FASTFLAGVARIABLE(DebugLuauJitDumpIr)
LUAU_FASTFLAGVARIABLE(DebugLuauJitCounters)
LUAU_FASTFLAGVARIABLE(DebugLuauJitPerfMap)
LUAU_FASTFLAGVARIABLE(DebugLuauJitDumpObjects)
LUAU_FASTINTVARIABLE(LuauJitExitRecompileThreshold, 64)
LUAU_FASTINTVARIABLE(LuauJitTierUpThreshold, 10000)
LUAU_FASTINTVARIABLE(LuauJitAggressiveThreshold, 100000)

namespace Luau::Jit
{

// The interpreter starts a natively compiled function by executing this instruction, which hands control to 'enter'
static constexpr Instruction kCodeEntryInsn = LOP_NATIVECALL;

// how many times the optimized tier is compiled again with the types its exits saw before it stops speculating
static constexpr uint32_t kMaxRespeculations = 4;

JitContext::JitContext(std::unique_ptr<llvm::orc::LLJIT> jit)
    : m_jit(std::move(jit))
{
}

JitContext::~JitContext()
{
    if (m_countersUsed)
        m_counters.print();
}

static void on_close(lua_State* L);

[[nodiscard]] static JitContext* get_jit_context(const lua_State* L) noexcept
{
    return L && L->global->ecb.close == on_close ? static_cast<JitContext*>(L->global->ecb.context) : nullptr;
}

[[nodiscard]] static Proto* get_proto(lua_State* L, const int idx) noexcept
{
    LUAU_ASSERT(lua_isLfunction(L, idx));
    const TValue* func = luaA_toobject(L, idx);

    return clvalue(func)->l.p;
}

static int on_enter(lua_State* L, Proto* proto)
{
    if (JitContext* ctx = get_jit_context(L))
        return ctx->on_enter(L, proto);

    return 1;
}

static void on_hot(lua_State* L, Proto* proto)
{
    if (JitContext* ctx = get_jit_context(L))
        ctx->on_hot(L, proto);
}

static void on_close(lua_State* L)
{
    const JitContext* ctx = get_jit_context(L);
    L->global->ecb = lua_ExecutionCallbacks{};
    delete ctx;
}

static void on_destroy(lua_State* L, Proto* proto)
{
    if (JitContext* ctx = get_jit_context(L))
        ctx->on_destroy(proto);
}

static void on_disable(lua_State* L, Proto* proto)
{
    if (JitContext* ctx = get_jit_context(L))
        ctx->on_disable(L, proto);
}

std::string to_string(const JITCompilationResult& result)
{
    switch (result)
    {
    case JITCompilationResult::Success:
        return "Success";
    case JITCompilationResult::NotInitialized:
        return "NotInitialized";
    case JITCompilationResult::NotFunction:
        return "NotFunction";
    case JITCompilationResult::NothingToCompile:
        return "NothingToCompile";
    case JITCompilationResult::TranslationFailed:
        return "TranslationFailed";
    case JITCompilationResult::VerificationFailed:
        return "VerificationFailed";
    case JITCompilationResult::CodeGenFailed:
        return "CodeGenFailed";
    case JITCompilationResult::Count:
        return "Count";
    }

    LUAU_ASSERT(!"Unknown compilation result");
    LUAU_UNREACHABLE();
}

bool is_supported()
{
    if (llvm::InitializeNativeTarget() || llvm::InitializeNativeTargetAsmPrinter())
        return false;

    const llvm::Triple triple(llvm::sys::getDefaultTargetTriple());
    std::string error;
    return llvm::TargetRegistry::lookupTarget(triple, error) != nullptr;
}

void create(lua_State* L)
{
    LUAU_ASSERT(L);
    LUAU_ASSERT(!L->global->ecb.context);

    auto jit = llvm::orc::LLJITBuilder().create();
    if (!jit)
    {
        llvm::logAllUnhandledErrors(jit.takeError(), llvm::WithColor(llvm::errs(), llvm::HighlightColor::Error), "could not create the jit: ");
        return;
    }

    if (FFlag::DebugLuauJitPerfMap && !enable_perf_map(**jit))
        fprintf(stderr, "luaujit: perf map is only supported when linking through JITLink\n");

    if (FFlag::DebugLuauJitDumpObjects && !enable_object_dump(**jit))
        fprintf(stderr, "luaujit: could not create the object dump directory\n");

    auto* context = new JitContext(std::move(*jit));

    lua_ExecutionCallbacks* cb = &L->global->ecb;
    cb->context = context;
    cb->enter = on_enter;
    cb->hot = on_hot;
    cb->close = on_close;
    cb->destroy = on_destroy;
    cb->disable = on_disable;
}

JITCompilationResult compile(lua_State* L, const int idx)
{
    JitContext* context = get_jit_context(L);
    if (!context)
        return JITCompilationResult::NotInitialized;
    if (!lua_isLfunction(L, idx))
        return JITCompilationResult::NotFunction;

    return context->compile(get_proto(L, idx));
}

bool get_proto_stats(lua_State* L, const int idx, ProtoStats& stats)
{
    LUAU_ASSERT(L);

    JitContext* context = get_jit_context(L);

    if (!context)
        return false;
    if (!lua_isLfunction(L, idx))
        return false;

    Proto* proto = get_proto(L, idx);

    stats = ProtoStats{};
    stats.hits = proto->hits;

    if (const NativeProto* nproto = context->get_native_proto(proto))
    {
        stats.calls = nproto->stats.calls;
        stats.tier = nproto->stats.tier;
        stats.exits = nproto->exits;
        stats.speculative = !(nproto->flags & NP_FLAG_NO_SPECULATION);
    }

    return true;
}

int JitContext::on_enter(lua_State* L, Proto* proto)
{
    LUAU_ASSERT(L);
    LUAU_ASSERT(proto);
    auto* nproto = static_cast<NativeProto*>(proto->execdata);
    LUAU_ASSERT(nproto == get_native_proto(proto));

    if (!nproto || nproto->flags & NP_FLAG_INVALIDATED)
        return 1;

    nproto->stats.calls++;

    LUAU_ASSERT(nproto->entry);
    LUAU_ASSERT(L->ci->savedpc >= proto->code && L->ci->savedpc < proto->code + proto->sizecode);

    const int entrypc = static_cast<int>(L->ci->savedpc - proto->code);

    // Generated code returns 1 when it wants the interpreter to continue the frame from ci->savedpc, and 0 when it has
    // finished the whole frame; both are exactly what the interpreter expects back from the 'enter' callback
    return reinterpret_cast<NativeEntry>(nproto->entry)(L, entrypc);
}

void JitContext::on_hot(lua_State* L, Proto* proto)
{
    LUAU_ASSERT(L);
    LUAU_ASSERT(proto);

    compile(proto);
}

void JitContext::on_destroy(Proto* proto) noexcept
{
    LUAU_ASSERT(proto);

    proto->execdata = nullptr;
    proto->exectarget = 0;
    proto->codeentry = proto->code;

    m_nativeProtos.erase(proto);
}

void JitContext::on_disable(lua_State* L, Proto* proto)
{
    LUAU_ASSERT(L);
    LUAU_ASSERT(proto);
    if (proto->codeentry == proto->code)
        return;

    proto->codeentry = proto->code;
    proto->exectarget = 0;

    if (NativeProto* nproto = get_native_proto(proto))
        nproto->flags |= NP_FLAG_INVALIDATED;

    // Frames of this proto that are already on a call stack were entered natively and would be resumed natively, so the
    // flag has to be cleared from every one of them
    luaM_visitgco(
        L,
        proto,
        [](void* context, lua_Page* page, GCObject* gco)
        {
            Proto* proto = static_cast<Proto*>(context);

            if (gco->gch.tt != LUA_TTHREAD)
                return false;

            lua_State* th = gco2th(gco);

            for (CallInfo* ci = th->ci; ci > th->base_ci; ci--)
            {
                if (isLua(ci))
                {
                    if (ci->p == proto)
                        ci->flags &= ~LUA_CALLINFO_NATIVE;
                }
            }

            return false;
        }
    );
}

void handle_exit_threshold(lua_State* L, Proto* proto)
{
    if (JitContext* ctx = get_jit_context(L))
        ctx->on_exit_threshold(proto);
}

void handle_tier_up(lua_State* L, NativeProto* caller)
{
    if (JitContext* ctx = get_jit_context(L))
        ctx->tier_up(caller);
}

// entries and back edges until code at 'tier' is compiled at the next one, zero when it isn't
static int32_t tier_up_budget(const ExecutionTier tier)
{
    switch (tier)
    {
    case ExecutionTier::Baseline:
        return FInt::LuauJitTierUpThreshold;
    case ExecutionTier::Optimized:
        // only baseline code records feedback, so there is no tier above it without one
        return FInt::LuauJitTierUpThreshold > 0 ? FInt::LuauJitAggressiveThreshold : 0;
    default:
        return 0;
    }
}

static llvm::OptimizationLevel get_optimization_level(const ExecutionTier tier)
{
    switch (tier)
    {
    case ExecutionTier::Cold:
    case ExecutionTier::Baseline:
        // the mem2reg and SROA passes that turn the register allocas back into SSA values
        // only run from O1 upwards, and buildPerModuleDefaultPipeline does not accept O0 at all
        return llvm::OptimizationLevel::O1;
    case ExecutionTier::Optimized:
        return llvm::OptimizationLevel::O2;
    case ExecutionTier::Aggressive:
        return llvm::OptimizationLevel::O3;
    case ExecutionTier::Count:
        break;
    }

    LUAU_ASSERT(!"Unknown execution tier");
    LUAU_UNREACHABLE();
}

static void optimize_module(llvm::Module& module, ExecutionTier tier)
{
    llvm::PassBuilder passBuilder;

    llvm::LoopAnalysisManager loopAnalyses;
    llvm::FunctionAnalysisManager functionAnalyses;
    llvm::CGSCCAnalysisManager cgsccAnalyses;
    llvm::ModuleAnalysisManager moduleAnalyses;

    passBuilder.registerModuleAnalyses(moduleAnalyses);
    passBuilder.registerCGSCCAnalyses(cgsccAnalyses);
    passBuilder.registerFunctionAnalyses(functionAnalyses);
    passBuilder.registerLoopAnalyses(loopAnalyses);
    passBuilder.crossRegisterProxies(loopAnalyses, functionAnalyses, cgsccAnalyses, moduleAnalyses);

    llvm::ModulePassManager passes = passBuilder.buildPerModuleDefaultPipeline(get_optimization_level(tier));
    passes.run(module, moduleAnalyses);
}

JITCompilationResult JitContext::compile(Proto* proto)
{
    LUAU_ASSERT(proto);

    // A Proto is only put in the map once it has code, so being in there at all means there is nothing left to do: either
    // the code is still installed, or a breakpoint has taken it away for good
    if (get_native_proto(proto))
        return JITCompilationResult::NothingToCompile;

    JITCompilationResult result = JITCompilationResult::Success;
    std::unique_ptr<NativeProto> nproto = compile_native(proto, ExecutionTier::Baseline, /* speculate */ true, nullptr, result);

    // Nothing is handed over to the interpreter, and nothing is remembered about the Proto, until the code is ready; a
    // failed attempt leaves the Proto exactly as it was and can be retried
    if (!nproto)
        return result;

    bind_native_proto(proto, std::move(nproto));

    return JITCompilationResult::Success;
}

void JitContext::tier_up(NativeProto* caller)
{
    LUAU_ASSERT(caller && caller->proto);

    Proto* proto = caller->proto;

    // frames of replaced code ask too, before they switch over
    if (get_native_proto(proto) != caller || caller->flags & NP_FLAG_INVALIDATED)
        return;

    LUAU_ASSERT(caller->stats.tier == ExecutionTier::Baseline || caller->stats.tier == ExecutionTier::Optimized);
    const ExecutionTier next = caller->stats.tier == ExecutionTier::Baseline ? ExecutionTier::Optimized : ExecutionTier::Aggressive;

    // the baseline only gives up on annotations, which feedback overrides; above it, giving up is for good
    const bool speculate = caller->stats.tier == ExecutionTier::Baseline || !(caller->flags & NP_FLAG_NO_SPECULATION);

    JITCompilationResult result = JITCompilationResult::Success;

    if (std::unique_ptr<NativeProto> nproto = compile_native(proto, next, speculate, caller, result))
        replace_native_proto(proto, std::move(nproto));
    else
        caller->tierUpBudget = INT32_MAX; // the code stays, and doesn't ask again for a long time
}

void JitContext::on_exit_threshold(Proto* proto)
{
    LUAU_ASSERT(proto);

    const NativeProto* current = get_native_proto(proto);
    if (!current || current->flags & (NP_FLAG_INVALIDATED | NP_FLAG_NO_SPECULATION))
        return;

    // the exits recorded the tags that failed the guards, so the tiers above the baseline can speculate again without
    // those guards; the baseline speculates on annotations, which exits don't change, so it stops
    const bool respeculate = current->stats.tier != ExecutionTier::Baseline && current->respeculations < kMaxRespeculations;

    JITCompilationResult result = JITCompilationResult::Success;
    std::unique_ptr<NativeProto> nproto = compile_native(proto, current->stats.tier, respeculate, current, result);

    // the speculative code keeps working, it just keeps leaving
    if (!nproto)
        return;

    if (respeculate)
        nproto->respeculations++;

    replace_native_proto(proto, std::move(nproto));
}

std::unique_ptr<NativeProto> JitContext::compile_native(
    Proto* proto,
    const ExecutionTier tier,
    const bool speculate,
    const NativeProto* previous,
    JITCompilationResult& result
)
{
    LUAU_ASSERT(proto);
    LUAU_ASSERT(tier == ExecutionTier::Baseline || (previous && previous->feedback));

    auto owned = std::make_unique<NativeProto>();
    owned->proto = proto;

    if (previous)
    {
        owned->stats = previous->stats;
        owned->feedback = previous->feedback;
        owned->respeculations = previous->respeculations;
    }
    else if (FInt::LuauJitTierUpThreshold > 0)
    {
        owned->feedback = std::make_shared<TypeFeedback>(proto->sizecode, 0);
    }

    owned->stats.tier = tier;
    owned->tierUpBudget = tier_up_budget(tier);

    if (!speculate)
        owned->flags |= NP_FLAG_NO_SPECULATION;

    NativeProto* nproto = owned.get();

    BytecodeFunction function;
    analyze_bytecode(function, proto, m_hostHooks, tier == ExecutionTier::Baseline ? nullptr : nproto->feedback.get());

    std::string name = "luau_jit_" + std::to_string(m_nextFunctionId++);

    // the name is what profilers show for the function
    name += "_";
    name += proto->debugname ? getstr(proto->debugname) : "anon";
    name += "_" + std::to_string(proto->linedefined);

    auto context = std::make_unique<llvm::LLVMContext>();

    result = JITCompilationResult::Success;

    // The types and the translation both hold on to the context, so they have to be gone before it is handed to ORC
    {
        auto module = std::make_unique<llvm::Module>(name, *context);
        module->setDataLayout(m_jit->getDataLayout());
        module->setTargetTriple(m_jit->getTargetTriple());

        VmTypes types(*context);
        types.verify_layout(module->getDataLayout());

        JitCounters* counters = nullptr;

        if (FFlag::DebugLuauJitCounters)
        {
            counters = &m_counters;
            m_countersUsed = true;
        }

        TranslationUnit unit(*nproto, function, m_hostHooks, types, *module, name, speculate, counters);

        if (!unit.translate())
            result = JITCompilationResult::TranslationFailed;
        else if (llvm::verifyFunction(*unit.native_function(), &llvm::errs()))
            result = JITCompilationResult::VerificationFailed;

        if (result == JITCompilationResult::Success)
        {
            if (FFlag::DebugLuauJitDumpIr)
            {
                llvm::errs() << "; luaujit: " << name << " before optimization\n";
                module->print(llvm::errs(), nullptr);
            }

            optimize_module(*module, nproto->stats.tier);

            if (FFlag::DebugLuauJitDumpIr)
            {
                llvm::errs() << "; luaujit: " << name << " after optimization\n";
                module->print(llvm::errs(), nullptr);
            }

            llvm::orc::ThreadSafeModule tsm(std::move(module), llvm::orc::ThreadSafeContext(std::move(context)));

            if (llvm::Error error = m_jit->addIRModule(std::move(tsm)))
            {
                llvm::logAllUnhandledErrors(
                    std::move(error), llvm::WithColor(llvm::errs(), llvm::HighlightColor::Error), "could not add a module to the jit: "
                );
                result = JITCompilationResult::CodeGenFailed;
            }
        }
    }

    if (result != JITCompilationResult::Success)
        return nullptr;

    auto entry = m_jit->lookup(name);

    if (!entry)
    {
        llvm::logAllUnhandledErrors(
            entry.takeError(), llvm::WithColor(llvm::errs(), llvm::HighlightColor::Error), "could not resolve a compiled function: "
        );
        result = JITCompilationResult::CodeGenFailed;
        return nullptr;
    }

    nproto->entry = entry->getValue();

    return owned;
}

NativeProto* JitContext::get_native_proto(Proto* proto)
{
    LUAU_ASSERT(proto);

    const auto it = m_nativeProtos.find(proto);
    if (it == m_nativeProtos.end())
        return nullptr;

    return it->second.get();
}

void JitContext::bind_native_proto(Proto* proto, std::unique_ptr<NativeProto> nproto)
{
    LUAU_ASSERT(proto);
    LUAU_ASSERT(nproto && nproto->entry);
    LUAU_ASSERT(m_nativeProtos.find(proto) == m_nativeProtos.end());

    NativeProto* tracked = m_nativeProtos.emplace(proto, std::move(nproto)).first->second.get();

    proto->execdata = tracked;
    proto->exectarget = tracked->entry;
    proto->codeentry = &kCodeEntryInsn;
}

void JitContext::replace_native_proto(Proto* proto, std::unique_ptr<NativeProto> nproto)
{
    LUAU_ASSERT(proto);
    LUAU_ASSERT(nproto && nproto->entry);

    auto it = m_nativeProtos.find(proto);
    LUAU_ASSERT(it != m_nativeProtos.end());

    m_retiredProtos.push_back(std::move(it->second));
    it->second = std::move(nproto);

    // frames that are already running the old code are resumed through exectarget, which picks up the new code at
    // the same entry pcs
    proto->execdata = it->second.get();
    proto->exectarget = it->second->entry;
}

} // namespace Luau::Jit
