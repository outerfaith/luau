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

LUAU_FASTFLAGVARIABLE(LuauJitDumpIr)

LUAU_FASTFLAG(LuauCIProto)

namespace Luau::Jit
{

// The interpreter starts a natively compiled function by executing this instruction, which hands control to 'enter'
static constexpr Instruction kCodeEntryInsn = LOP_NATIVECALL;

JitContext::JitContext(std::unique_ptr<llvm::orc::LLJIT> jit)
    : m_jit(std::move(jit))
{
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
                    Proto* p = FFlag::LuauCIProto ? ci->p : clvalue(ci->func)->l.p;

                    if (p == proto)
                        ci->flags &= ~LUA_CALLINFO_NATIVE;
                }
            }

            return false;
        }
    );
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

    // Nothing is handed over to the interpreter, and nothing is remembered about the Proto, until the code is ready; a
    // failed attempt leaves the Proto exactly as it was and can be retried
    auto owned = std::make_unique<NativeProto>();
    owned->proto = proto;
    owned->stats.tier = ExecutionTier::Baseline;

    NativeProto* nproto = owned.get();

    BytecodeFunction function;
    analyze_bytecode(function, proto, m_hostHooks);

    std::string name = "luau_jit_" + std::to_string(m_nextFunctionId++);

    auto context = std::make_unique<llvm::LLVMContext>();

    JITCompilationResult result = JITCompilationResult::Success;

    // The types and the translation both hold on to the context, so they have to be gone before it is handed to ORC
    {
        auto module = std::make_unique<llvm::Module>(name, *context);
        module->setDataLayout(m_jit->getDataLayout());
        module->setTargetTriple(m_jit->getTargetTriple());

        VmTypes types(*context);
        types.verify_layout(module->getDataLayout());

        TranslationUnit unit(*nproto, function, m_hostHooks, types, *module, name);

        if (!unit.translate())
            result = JITCompilationResult::TranslationFailed;
        else if (llvm::verifyFunction(*unit.native_function(), &llvm::errs()))
            result = JITCompilationResult::VerificationFailed;

        if (result == JITCompilationResult::Success)
        {
            if (FFlag::LuauJitDumpIr)
            {
                llvm::errs() << "; luaujit: " << name << " before optimization\n";
                module->print(llvm::errs(), nullptr);
            }

            optimize_module(*module, nproto->stats.tier);

            if (FFlag::LuauJitDumpIr)
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

    if (result == JITCompilationResult::Success)
    {
        auto entry = m_jit->lookup(name);

        if (!entry)
        {
            llvm::logAllUnhandledErrors(
                entry.takeError(), llvm::WithColor(llvm::errs(), llvm::HighlightColor::Error), "could not resolve a compiled function: "
            );
            result = JITCompilationResult::CodeGenFailed;
        }
        else
        {
            nproto->entry = entry->getValue();
        }
    }

    if (result != JITCompilationResult::Success)
        return result;

    bind_native_proto(proto, std::move(owned));

    return JITCompilationResult::Success;
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

} // namespace Luau::Jit
