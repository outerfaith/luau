// This file is part of the Luau programming language and is licensed under MIT License; see LICENSE.txt for details
#include "BytecodeTranslation.h"

#include "BuiltinTranslation.h"

#include "BytecodeAnalysis.h"
#include "BytecodeVisitUseDef.h"
#include "JitContext.h"
#include "VmHelpers.h"

#include "Luau/BytecodeUtils.h"
#include "Luau/Common.h"

#include "lgc.h"
#include "lstate.h"
#include "ltable.h"
#include "lvm.h"

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/CFG.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/MDBuilder.h>
#include <llvm/IR/Module.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <optional>
#include <string.h>

LUAU_FASTFLAG(LuauBackedgeHeapCheck)
LUAU_FASTINT(LuauJitExitRecompileThreshold)
LUAU_FASTINT(LuauNativeHitsThreshold)

namespace Luau::Jit
{

// TKey keeps its type tag in a 4-bit field of the word that follows 'extra'
static constexpr size_t kNodeKeyTagOffset = offsetof(LuaNode, key) + offsetof(TKey, extra) + sizeof(int) * LUA_EXTRA_SIZE;
static_assert(sizeof(TKey) == offsetof(TKey, extra) + sizeof(int) * LUA_EXTRA_SIZE + sizeof(unsigned), "unexpected TKey layout");

TranslationUnit::TranslationUnit(
    NativeProto& nproto,
    const BytecodeFunction& function,
    const HostHooks& hostHooks,
    const VmTypes& types,
    llvm::Module& module,
    const llvm::StringRef name,
    const bool speculate,
    JitCounters* counters
)
    : m_nproto(nproto)
    , m_bytecode(function)
    , m_hostHooks(hostHooks)
    , m_types(types)
    , m_module(module)
    , m_counters(counters)
    , m_speculate(speculate)
    , m_profiling(nproto.feedback && nproto.stats.tier == ExecutionTier::Baseline)
    , m_tiersUp(nproto.tierUpBudget > 0)
    , m_builder(module.getContext())
{
    LUAU_ASSERT(nproto.proto == function.proto);

    m_function = llvm::Function::Create(types.nativeEntryTy, llvm::Function::ExternalLinkage, name, module);

    // VM errors are C++ exceptions, and they unwind through generated frames
    m_function->setUWTableKind(llvm::UWTableKind::Async);

    m_state = m_function->getArg(0);
    m_state->setName("L");

    m_entrypc = m_function->getArg(1);
    m_entrypc->setName("entrypc");
}

bool TranslationUnit::translate()
{
    const Proto* proto = m_bytecode.proto;
    LUAU_ASSERT(proto);

    if (m_bytecode.bcBlocks.empty())
        return false;

    m_nproto.deopts.clear();

    m_blocks.clear();
    m_blocks.resize(proto->sizecode, nullptr);
    m_genericBlocks.clear();
    m_genericBlocks.resize(proto->sizecode, nullptr);

    m_entryPoints.clear();
    m_exitBlocks.clear();
    m_slowBlocks.clear();
    m_loopPads.clear();

    m_loopHeaders.assign(proto->sizecode, false);
    for (int pcpos = 0; pcpos < proto->sizecode;)
    {
        const Instruction insn = proto->code[pcpos];

        if (const int target = getJumpTarget(insn, pcpos); target >= 0 && target <= pcpos)
            m_loopHeaders[target] = true;

        pcpos += getOpLength(LuauOpcode(LUAU_INSN_OP(insn)));
    }

    compute_dirty_regs();

    llvm::LLVMContext& context = m_module.getContext();

    m_entryBlock = llvm::BasicBlock::Create(context, "entry", m_function);
    m_bailBlock = llvm::BasicBlock::Create(context, "bail", m_function);
    m_yieldBlock = llvm::BasicBlock::Create(context, "yield", m_function);

    emit_prologue();
    find_integer_loops();

    // an entry pc without a case continues in the interpreter
    m_builder.SetInsertPoint(m_bailBlock);
    if (m_counters)
        emit_counter(&m_counters->interpreterResumes);
    m_builder.CreateRet(m_builder.getInt32(1));

    // the interpreter exits luau_execute when it gets 0 back
    m_builder.SetInsertPoint(m_yieldBlock);
    m_builder.CreateRet(m_builder.getInt32(0));

    // only the places the interpreter resumes a frame at are entry points; any other entry pc returns to the
    // interpreter, and every extra entry would make loop headers merge in registers of unknown type
    mark_entry_point(0);

    for (uint32_t blockIdx = 0; blockIdx < static_cast<uint32_t>(m_bytecode.bcBlocks.size()); blockIdx++)
        translate_block(blockIdx);

    // blocks can be referenced as branch targets and entry points after their bytecode block stopped being
    // translated early, those continue in the interpreter
    for (int pcpos = 0; pcpos < proto->sizecode; pcpos++)
    {
        for (llvm::BasicBlock* block : {m_blocks[pcpos], m_genericBlocks[pcpos]})
        {
            if (block && block->empty())
            {
                m_builder.SetInsertPoint(block);
                m_builder.CreateBr(vm_exit(pcpos));
            }
        }
    }

    emit_dispatch();

    std::sort(
        m_nproto.deopts.begin(),
        m_nproto.deopts.end(),
        [](const DeoptPoint& lhs, const DeoptPoint& rhs)
        {
            return lhs.pc < rhs.pc;
        }
    );

    return true;
}

llvm::Value* TranslationUnit::load_base(const llvm::Twine& name)
{
    LUAU_ASSERT(m_baseSlot);

    return m_builder.CreateLoad(m_types.ptrTy, m_baseSlot, name);
}

llvm::Value* TranslationUnit::stack_slot(const int reg, const llvm::Twine& name)
{
    return m_types.stack_slot(m_builder, load_base(), reg, name);
}

llvm::Value* TranslationUnit::reg_slot(const int reg) const
{
    LUAU_ASSERT(reg >= 0 && static_cast<size_t>(reg) < m_regSlots.size());

    return m_regSlots[reg];
}

llvm::Value* TranslationUnit::load_reg(const int reg, const llvm::Twine& name)
{
    return m_builder.CreateLoad(m_types.tvalueTy, reg_slot(reg), name);
}

void TranslationUnit::store_reg(const int reg, llvm::Value* value)
{
    LUAU_ASSERT(value->getType() == m_types.tvalueTy);

    m_builder.CreateStore(value, reg_slot(reg));
}

llvm::Value* TranslationUnit::load_tag(const int reg)
{
    return m_builder.CreateLoad(m_types.i32Ty, m_builder.CreateStructGEP(m_types.tvalueTy, reg_slot(reg), VmTypes::kTValueTt), "tt");
}

llvm::Value* TranslationUnit::load_number(const int reg)
{
    // only FORNLOOP writes the index register of a loop, and it keeps the counter in step
    for (const IntegerLoop& loop : m_integerLoops)
    {
        if (loop.reg == reg && m_pcpos > loop.prepPc && m_pcpos < loop.loopPc)
            return m_builder.CreateSIToFP(m_builder.CreateLoad(m_types.i32Ty, loop.counter), m_types.doubleTy, "n");
    }

    return m_builder.CreateLoad(m_types.doubleTy, reg_slot(reg), "n");
}

llvm::Value* TranslationUnit::load_pointer(const int reg)
{
    return m_builder.CreateLoad(m_types.ptrTy, reg_slot(reg), "gc");
}

llvm::Value* TranslationUnit::load_boolean(const int reg)
{
    return m_builder.CreateLoad(m_types.i32Ty, reg_slot(reg), "b");
}

// Vectors are accessed in the pieces that TValue copies use, the first two components in the value and the rest in
// the extra field. An access that straddles them keeps the register out of SSA form, and then a load of the whole
// vector right after the stores of the pieces stalls on store forwarding
llvm::Value* TranslationUnit::load_vector(const int reg)
{
    llvm::Value* slot = reg_slot(reg);

    llvm::Type* pairTy = llvm::FixedVectorType::get(m_types.floatTy, 2);
    llvm::Value* xy = m_builder.CreateAlignedLoad(pairTy, slot, llvm::Align(8), "xy");
    llvm::Value* v = m_builder.CreateShuffleVector(xy, llvm::ArrayRef<int>{0, 1, -1, -1}.take_front(LUA_VECTOR_SIZE));

    for (int i = 2; i < LUA_VECTOR_SIZE; i++)
    {
        llvm::Value* component = m_builder.CreateConstGEP1_64(m_types.i8Ty, slot, sizeof(double) + sizeof(float) * (i - 2));
        v = m_builder.CreateInsertElement(v, m_builder.CreateAlignedLoad(m_types.floatTy, component, llvm::Align(4)), uint64_t(i));
    }

    return v;
}

void TranslationUnit::store_tag(const int reg, const int tag)
{
    m_builder.CreateStore(m_builder.getInt32(tag), m_builder.CreateStructGEP(m_types.tvalueTy, reg_slot(reg), VmTypes::kTValueTt));
}

// The interpreter leaves the parts of a TValue that its tag doesn't use as they were, but nothing ever reads them.
// Zeroing them instead means the old contents don't have to be kept alive until the register is spilled
void TranslationUnit::clear_extra(const int reg)
{
    m_builder.CreateStore(
        llvm::Constant::getNullValue(m_types.tvalueTy->getElementType(VmTypes::kTValueExtra)),
        m_builder.CreateStructGEP(m_types.tvalueTy, reg_slot(reg), VmTypes::kTValueExtra)
    );
}

void TranslationUnit::store_nil(const int reg)
{
    m_builder.CreateStore(m_builder.getInt64(0), reg_slot(reg));
    clear_extra(reg);
    store_tag(reg, LUA_TNIL);
}

void TranslationUnit::store_object(const int reg, llvm::Value* object, const int tag)
{
    LUAU_ASSERT(object->getType() == m_types.ptrTy);

    m_builder.CreateStore(object, reg_slot(reg));
    clear_extra(reg);
    store_tag(reg, tag);
}

void TranslationUnit::store_lightuserdata(const int reg, llvm::Value* pointer, const int utag)
{
    LUAU_ASSERT(pointer->getType() == m_types.ptrTy);

    m_builder.CreateStore(pointer, reg_slot(reg));
    clear_extra(reg);
    store_field(m_builder.getInt32(utag), reg_slot(reg), offsetof(TValue, extra));
    store_tag(reg, LUA_TLIGHTUSERDATA);
}

void TranslationUnit::store_number(const int reg, llvm::Value* value)
{
    LUAU_ASSERT(value->getType() == m_types.doubleTy);

    m_builder.CreateStore(value, reg_slot(reg));
    clear_extra(reg);
    store_tag(reg, LUA_TNUMBER);
}

void TranslationUnit::store_vector(const int reg, llvm::Value* value)
{
    LUAU_ASSERT(value->getType() == m_types.vectorTy);

    llvm::Value* slot = reg_slot(reg);

    m_builder.CreateAlignedStore(m_builder.CreateShuffleVector(value, llvm::ArrayRef<int>{0, 1}), slot, llvm::Align(8));

    for (int i = 2; i < LUA_VECTOR_SIZE; i++)
    {
        llvm::Value* component = m_builder.CreateConstGEP1_64(m_types.i8Ty, slot, sizeof(double) + sizeof(float) * (i - 2));
        m_builder.CreateAlignedStore(m_builder.CreateExtractElement(value, uint64_t(i)), component, llvm::Align(4));
    }

    store_tag(reg, LUA_TVECTOR);
}

void TranslationUnit::store_boolean(const int reg, llvm::Value* value)
{
    LUAU_ASSERT(value->getType() == m_types.i32Ty);

    // the boolean is the low half of the value, the rest is cleared like the extra field
    m_builder.CreateStore(m_builder.CreateZExt(value, m_types.i64Ty), reg_slot(reg));
    clear_extra(reg);
    store_tag(reg, LUA_TBOOLEAN);
}

llvm::Value* TranslationUnit::is_falsy(const int reg)
{
    llvm::Value* tag = load_tag(reg);

    llvm::Value* isNil = m_builder.CreateICmpEQ(tag, m_builder.getInt32(LUA_TNIL));
    llvm::Value* isFalse = m_builder.CreateAnd(
        m_builder.CreateICmpEQ(tag, m_builder.getInt32(LUA_TBOOLEAN)), m_builder.CreateICmpEQ(load_boolean(reg), m_builder.getInt32(0))
    );

    return m_builder.CreateOr(isNil, isFalse, "falsy");
}

llvm::Constant* TranslationUnit::constant(const int kidx) const
{
    const Proto* proto = m_bytecode.proto;
    LUAU_ASSERT(kidx >= 0 && kidx < proto->sizek);

    // the constant table never changes after load, and it keeps every object it references alive for as long as the
    // generated code exists
    const TValue& value = proto->k[kidx];

    uint64_t bits = 0;
    memcpy(&bits, &value.value, sizeof(bits));

    std::vector<llvm::Constant*> extra;
    for (int i = 0; i < LUA_EXTRA_SIZE; i++)
        extra.push_back(llvm::ConstantInt::get(m_types.i32Ty, static_cast<uint32_t>(value.extra[i])));

    auto* extraTy = llvm::cast<llvm::ArrayType>(m_types.tvalueTy->getElementType(VmTypes::kTValueExtra));

    return llvm::ConstantStruct::get(
        m_types.tvalueTy,
        {llvm::ConstantInt::get(m_types.i64Ty, bits), llvm::ConstantArray::get(extraTy, extra), llvm::ConstantInt::get(m_types.i32Ty, value.tt)}
    );
}

llvm::Constant* TranslationUnit::const_ptr(const void* value) const
{
    llvm::LLVMContext& context = m_module.getContext();
    llvm::Type* intptrTy = m_module.getDataLayout().getIntPtrType(context);

    return llvm::ConstantExpr::getIntToPtr(llvm::ConstantInt::get(intptrTy, reinterpret_cast<uintptr_t>(value)), m_types.ptrTy);
}

llvm::Constant* TranslationUnit::insn_ptr(const int pcpos) const
{
    LUAU_ASSERT(pcpos >= 0 && pcpos <= m_bytecode.proto->sizecode);

    return const_ptr(&m_bytecode.proto->code[pcpos]);
}

llvm::Value* TranslationUnit::load_field(llvm::Type* type, llvm::Value* object, const size_t offset, const llvm::Twine& name)
{
    return m_builder.CreateLoad(type, m_types.field_ptr(m_builder, object, offset), name);
}

void TranslationUnit::store_field(llvm::Value* value, llvm::Value* object, const size_t offset)
{
    m_builder.CreateStore(value, m_types.field_ptr(m_builder, object, offset));
}

// Two scalar TBAA types under one root: accesses tagged with different ones never alias, untagged ones alias anything
static void create_tbaa(llvm::LLVMContext& context, llvm::MDNode*& header, llvm::MDNode*& buffer)
{
    if (header)
        return;

    llvm::MDBuilder md(context);
    llvm::MDNode* root = md.createTBAARoot("luau jit");
    llvm::MDNode* headerType = md.createTBAAScalarTypeNode("vm header", root);
    llvm::MDNode* bufferType = md.createTBAAScalarTypeNode("buffer data", root);

    header = md.createTBAAStructTagNode(headerType, headerType, 0);
    buffer = md.createTBAAStructTagNode(bufferType, bufferType, 0);
}

llvm::Value* TranslationUnit::load_header(llvm::Type* type, llvm::Value* object, const size_t offset, const llvm::Twine& name)
{
    create_tbaa(m_module.getContext(), m_tbaaHeader, m_tbaaBuffer);

    llvm::LoadInst* load = m_builder.CreateLoad(type, m_types.field_ptr(m_builder, object, offset), name);
    load->setMetadata(llvm::LLVMContext::MD_tbaa, m_tbaaHeader);
    return load;
}

void TranslationUnit::mark_buffer_access(llvm::Instruction* access)
{
    create_tbaa(m_module.getContext(), m_tbaaHeader, m_tbaaBuffer);

    access->setMetadata(llvm::LLVMContext::MD_tbaa, m_tbaaBuffer);
}

llvm::Value* TranslationUnit::is_safe_env()
{
    // the environment can stop being safe through getfenv/setfenv after the code was generated
    llvm::Value* env = load_header(m_types.ptrTy, m_closure, offsetof(Closure, env), "env");
    llvm::Value* safeenv = load_header(m_types.i8Ty, env, offsetof(LuaTable, safeenv), "safeenv");

    return m_builder.CreateICmpNE(safeenv, m_builder.getInt8(0));
}

llvm::Value* TranslationUnit::call_helper(const void* helper, const int pcpos, const bool returnsInt)
{
    llvm::Type* result = returnsInt ? static_cast<llvm::Type*>(m_types.i32Ty) : m_types.voidTy;

    if (m_counters)
        emit_counter(&m_counters->helperCalls[LUAU_INSN_OP(m_bytecode.proto->code[pcpos])]);

    return call_native(helper, result, {m_state, insn_ptr(pcpos)});
}

llvm::Value* TranslationUnit::call_native(const void* function, llvm::Type* result, const llvm::ArrayRef<llvm::Value*> args)
{
    std::vector<llvm::Type*> params;
    for (llvm::Value* arg : args)
        params.push_back(arg->getType());

    llvm::FunctionType* type = llvm::FunctionType::get(result, params, /* isVarArg */ false);

    return m_builder.CreateCall(type, const_ptr(function), args);
}

void TranslationUnit::spill(const int pcpos)
{
    LUAU_ASSERT(pcpos >= 0 && static_cast<size_t>(pcpos) < m_dirtyIn.size());

    const std::bitset<256>& dirty = m_dirtyIn[pcpos];

    llvm::Value* base = load_base();

    for (int reg = 0; reg < m_bytecode.proto->maxstacksize; reg++)
    {
        if (dirty.test(reg))
            m_types.copy_tvalue(m_builder, m_types.stack_slot(m_builder, base, reg), reg_slot(reg));
    }
}

static bool is_fastcall(const LuauOpcode op)
{
    return op == LOP_FASTCALL || op == LOP_FASTCALL1 || op == LOP_FASTCALL2 || op == LOP_FASTCALL2K || op == LOP_FASTCALL3;
}

// Instructions that call into the VM on every path, which leaves all registers in sync with the stack afterwards
static bool always_syncs(const LuauOpcode op)
{
    switch (op)
    {
    case LOP_SETGLOBAL:
    case LOP_CLOSEUPVALS:
    case LOP_NEWCLOSURE:
    case LOP_CALL:
    case LOP_CALLFB:
    case LOP_CONCAT:
    case LOP_SETLIST:
    case LOP_FORGPREP:
    case LOP_GETVARARGS:
    case LOP_PREPVARARGS:
    case LOP_DUPCLOSURE:
    case LOP_FASTCALL:
    case LOP_FASTCALL1:
    case LOP_FASTCALL2:
    case LOP_FASTCALL2K:
    case LOP_FASTCALL3:
    case LOP_FASTPCALL:
    case LOP_GETUDATAKS:
    case LOP_SETUDATAKS:
    case LOP_NAMECALLUDATA:
    case LOP_NEWCLASS:
    case LOP_NEWCLASSMEMBER:
    case LOP_CONSTRUCT:
    case LOP_FINCONSTRUCT:
        return true;
    default:
        return false;
    }
}

namespace
{

struct DirtyVisitor
{
    std::bitset<256>& dirty;
    int maxstacksize;

    void def(const int reg)
    {
        dirty.set(reg);
    }

    void def_range(const int start, const int count)
    {
        const int end = count == -1 ? maxstacksize : start + count;

        for (int reg = start; reg < end; reg++)
            dirty.set(reg);
    }

    void use(int) {}

    void use_range(int, int) {}

    void capture(int) {}
};

} // namespace

void TranslationUnit::compute_dirty_regs()
{
    Proto* proto = m_bytecode.proto;
    const CfgInfo& cfg = m_bytecode.cfg;
    const size_t blockCount = m_bytecode.bcBlocks.size();

    m_dirtyIn.assign(proto->sizecode, {});

    m_liveIn = compute_pc_liveness(m_bytecode).in;
    const std::vector<std::bitset<256>>& liveIn = m_liveIn;
    const std::bitset<256> captured = cfg.captured.regs;

    std::vector<std::bitset<256>> blockIn(blockCount);
    std::vector<std::bitset<256>> blockOut(blockCount);

    auto transfer = [&](const uint32_t blockIdx, std::bitset<256> dirty, const bool record)
    {
        const auto& [startpc, finishpc] = m_bytecode.bcBlocks[blockIdx];

        // An inline builtin skips over its paired CALL without syncing anything but the result. The skip isn't a CFG
        // edge, so the set it carries is merged in at the target, or at the end of the block when the target starts the
        // next one
        std::vector<std::pair<int, std::bitset<256>>> skips;

        for (int pcpos = startpc; pcpos <= finishpc;)
        {
            for (auto it = skips.begin(); it != skips.end();)
            {
                if (it->first == pcpos)
                {
                    dirty |= it->second;
                    it = skips.erase(it);
                }
                else
                {
                    ++it;
                }
            }

            // nothing reads a dead register again, but within a block its value still reaches the stack before every
            // call, which is what makes 'x = nil; collectgarbage()' release the old value. Like CodeGen, the value is
            // only dropped at the end of the block, and captured registers never are, since closures read them
            if (pcpos == startpc)
                dirty &= liveIn[pcpos] | captured;

            if (record)
                m_dirtyIn[pcpos] = dirty;

            const Instruction* pc = &proto->code[pcpos];
            const auto op = static_cast<LuauOpcode>(LUAU_INSN_OP(*pc));

            int count = 0;

            if (const int result = is_fastcall(op) ? inline_builtin_results(proto, pcpos, count) : -1; result >= 0)
            {
                skips.emplace_back(getJumpTarget(*pc, pcpos), dirty);

                for (int reg = result; reg < result + count; reg++)
                    skips.back().second.set(reg);
            }

            if (always_syncs(op))
            {
                dirty.reset();
            }
            else
            {
                DirtyVisitor visitor{dirty, proto->maxstacksize};
                visit_bytecode_reg_defs_uses(visitor, proto, pcpos);
            }

            pcpos += getOpLength(op);
        }

        for (const auto& [target, skipped] : skips)
            dirty |= skipped;

        return dirty;
    };

    // a may-analysis, so the sets only ever grow and the iteration ends once nothing changes
    bool changed = true;

    while (changed)
    {
        changed = false;

        for (uint32_t blockIdx = 0; blockIdx < blockCount; blockIdx++)
        {
            std::bitset<256> in;
            for (uint32_t pred : predecessors(cfg, blockIdx))
                in |= blockOut[pred];

            std::bitset<256> out = transfer(blockIdx, in, false);

            if (in != blockIn[blockIdx] || out != blockOut[blockIdx])
            {
                blockIn[blockIdx] = in;
                blockOut[blockIdx] = out;
                changed = true;
            }
        }
    }

    for (uint32_t blockIdx = 0; blockIdx < blockCount; blockIdx++)
        transfer(blockIdx, blockIn[blockIdx], true);
}

// Helpers only write the registers that their instruction defines and the ones closures have captured, except for the
// instructions that place a frame inside of this one, which leave nothing as it was
void TranslationUnit::reload(const int pcpos)
{
    Proto* proto = m_bytecode.proto;
    const auto op = static_cast<LuauOpcode>(LUAU_INSN_OP(proto->code[pcpos]));

    std::bitset<256> regs = m_bytecode.cfg.captured.regs;

    if (always_syncs(op))
    {
        regs.set();
    }
    else
    {
        DirtyVisitor visitor{regs, proto->maxstacksize};
        visit_bytecode_reg_defs_uses(visitor, proto, pcpos);
    }

    reload_regs(regs);
}

void TranslationUnit::mark_slow_path()
{
    m_slowBlocks.insert(current_block());
}

bool TranslationUnit::holds_new_table(const int reg, const int pcpos) const
{
    Proto* proto = m_bytecode.proto;
    bool fresh = false;

    for (int pos = m_blockStart; pos >= 0 && pos < pcpos;)
    {
        const auto op = static_cast<LuauOpcode>(LUAU_INSN_OP(proto->code[pos]));

        std::bitset<256> defs;
        DirtyVisitor visitor{defs, proto->maxstacksize};
        visit_bytecode_reg_defs_uses(visitor, proto, pos);

        if (defs.test(reg))
            fresh = op == LOP_NEWTABLE || op == LOP_DUPTABLE;

        pos += getOpLength(op);
    }

    return fresh;
}

void TranslationUnit::reload_regs(const std::bitset<256>& regs)
{
    mark_slow_path();

    // the stack might have been reallocated even when no register changed
    llvm::Value* base = m_types.load_state_base(m_builder, m_state);
    m_builder.CreateStore(base, m_baseSlot);

    for (int reg = 0; reg < m_bytecode.proto->maxstacksize; reg++)
    {
        if (regs.test(reg))
            m_types.copy_tvalue(m_builder, reg_slot(reg), m_types.stack_slot(m_builder, base, reg));
    }
}

void TranslationUnit::emit_safepoint(const int pcpos, const bool backedge)
{
    // before the interrupt, which the new code runs at the same pc
    if (backedge && m_tiersUp)
        emit_tier_up(pcpos);

    auto* global = llvm::cast<llvm::LoadInst>(load_field(m_types.ptrTy, m_state, offsetof(lua_State, global), "global"));
    global->setMetadata(llvm::LLVMContext::MD_invariant_load, llvm::MDNode::get(m_module.getContext(), {}));

    // the hook can be installed from outside of the VM at any point, so the read must not be hoisted out of loops
    llvm::LoadInst* interrupt = m_builder.CreateLoad(
        m_types.ptrTy, m_types.field_ptr(m_builder, global, offsetof(global_State, cb) + offsetof(lua_Callbacks, interrupt)), "interrupt"
    );
    interrupt->setVolatile(true);

    llvm::Value* cond = m_builder.CreateIsNotNull(interrupt);

    if (backedge && FFlag::LuauBackedgeHeapCheck)
    {
        llvm::Type* sizeTy = m_module.getDataLayout().getIntPtrType(m_module.getContext());

        llvm::Value* totalbytes = load_field(sizeTy, global, offsetof(global_State, totalbytes));
        llvm::Value* threshold = load_field(sizeTy, global, offsetof(global_State, GCthreshold));

        cond = m_builder.CreateOr(cond, m_builder.CreateICmpUGE(totalbytes, threshold));
    }

    llvm::BasicBlock* slow = new_block("safepoint");
    llvm::BasicBlock* resume = new_block("safepoint.done");

    m_builder.CreateCondBr(cond, slow, resume, unlikely());

    m_builder.SetInsertPoint(slow);
    spill(pcpos);

    const void* helper = backedge ? reinterpret_cast<const void*>(&execute_safepoint) : reinterpret_cast<const void*>(&execute_interrupt);
    llvm::Value* stop = call_helper(helper, pcpos, true);

    llvm::BasicBlock* reloadBlock = new_block("safepoint.reload");
    m_builder.CreateCondBr(m_builder.CreateICmpNE(stop, m_builder.getInt32(0)), yield_exit(), reloadBlock);

    m_builder.SetInsertPoint(reloadBlock);
    reload_regs(m_bytecode.cfg.captured.regs);
    m_builder.CreateBr(resume);

    m_builder.SetInsertPoint(resume);

    // after a yield or a break, the thread resumes by running this instruction again
    mark_entry_point(pcpos);
}

void TranslationUnit::emit_gc_check(const int pcpos, const int reg)
{
    auto* global = llvm::cast<llvm::LoadInst>(load_field(m_types.ptrTy, m_state, offsetof(lua_State, global), "global"));
    global->setMetadata(llvm::LLVMContext::MD_invariant_load, llvm::MDNode::get(m_module.getContext(), {}));

    llvm::Type* sizeTy = m_module.getDataLayout().getIntPtrType(m_module.getContext());
    llvm::Value* totalbytes = load_field(sizeTy, global, offsetof(global_State, totalbytes));
    llvm::Value* threshold = load_field(sizeTy, global, offsetof(global_State, GCthreshold));

    llvm::BasicBlock* slow = new_block("gc");
    llvm::BasicBlock* done = new_block("gc.done");

    m_builder.CreateCondBr(m_builder.CreateICmpUGE(totalbytes, threshold), slow, done, unlikely());

    // the collector marks what is on the stack, so the new object has to be there too
    m_builder.SetInsertPoint(slow);
    spill(pcpos);
    llvm::Value* base = load_base();
    m_types.copy_tvalue(m_builder, m_types.stack_slot(m_builder, base, reg), reg_slot(reg));

    call_helper(reinterpret_cast<const void*>(&execute_gc_check), pcpos, false);

    // a step can shrink the stack, but it leaves every value as it was
    m_builder.CreateStore(m_types.load_state_base(m_builder, m_state), m_baseSlot);
    m_builder.CreateBr(done);

    m_builder.SetInsertPoint(done);
}

void TranslationUnit::emit_enter_native(llvm::Value* target)
{
    if (m_counters)
        emit_counter(&m_counters->nativeTransfers);

    llvm::CallInst* call = m_builder.CreateCall(m_types.nativeEntryTy, target, {m_state, m_builder.getInt32(0)});
    call->setTailCallKind(llvm::CallInst::TCK_MustTail);
    m_builder.CreateRet(call);
}

void TranslationUnit::emit_resume_ci()
{
    llvm::Value* ci = m_types.load_state_ci(m_builder, m_state);
    llvm::Value* flags = load_field(m_types.i32Ty, ci, offsetof(CallInfo, flags), "flags");

    llvm::BasicBlock* checkTarget = new_block("resume.native");
    llvm::BasicBlock* enter = new_block("resume.enter");

    llvm::Value* isNative = m_builder.CreateICmpNE(m_builder.CreateAnd(flags, LUA_CALLINFO_NATIVE), m_builder.getInt32(0));
    m_builder.CreateCondBr(isNative, checkTarget, m_bailBlock);

    m_builder.SetInsertPoint(checkTarget);
    llvm::Value* proto = load_field(m_types.ptrTy, ci, offsetof(CallInfo, p), "p");
    llvm::Value* target = load_field(m_types.ptrTy, proto, offsetof(Proto, exectarget), "exectarget");

    // a breakpoint takes the code away by clearing exectarget, and then the frame continues in the interpreter
    m_builder.CreateCondBr(m_builder.CreateIsNotNull(target), enter, m_bailBlock);

    m_builder.SetInsertPoint(enter);
    llvm::Value* savedpc = load_field(m_types.ptrTy, ci, offsetof(CallInfo, savedpc), "savedpc");
    llvm::Value* code = load_field(m_types.ptrTy, proto, offsetof(Proto, code), "code");
    llvm::Value* entrypc = m_builder.CreateTrunc(m_builder.CreatePtrDiff(m_types.i32Ty, savedpc, code), m_types.i32Ty, "entrypc");

    if (m_counters)
        emit_counter(&m_counters->nativeTransfers);

    // the tail call keeps the C stack flat no matter how deep the Lua call stack gets
    llvm::CallInst* call = m_builder.CreateCall(m_types.nativeEntryTy, target, {m_state, entrypc});
    call->setTailCallKind(llvm::CallInst::TCK_MustTail);
    m_builder.CreateRet(call);
}

void TranslationUnit::emit_table_barrier(llvm::Value* table, const int valueReg)
{
    // luaC_barriert
    emit_barrier(table, valueReg, reinterpret_cast<const void*>(&luaC_barriertable));
}

void TranslationUnit::emit_barrier(llvm::Value* object, const int valueReg, const void* barrier)
{
    llvm::BasicBlock* checkObject = new_block("barrier.object");
    llvm::BasicBlock* checkValue = new_block("barrier.value");
    llvm::BasicBlock* slow = new_block("barrier");
    llvm::BasicBlock* done = new_block("barrier.done");

    llvm::Value* collectable = m_builder.CreateICmpSGE(load_tag(valueReg), m_builder.getInt32(LUA_TSTRING));
    m_builder.CreateCondBr(collectable, checkObject, done);

    m_builder.SetInsertPoint(checkObject);
    llvm::Value* objectMarked = load_field(m_types.i8Ty, object, offsetof(GCheader, marked));
    llvm::Value* isBlack = m_builder.CreateICmpNE(m_builder.CreateAnd(objectMarked, bitmask(BLACKBIT)), m_builder.getInt8(0));
    m_builder.CreateCondBr(isBlack, checkValue, done);

    m_builder.SetInsertPoint(checkValue);
    llvm::Value* value = load_pointer(valueReg);
    llvm::Value* valueMarked = load_field(m_types.i8Ty, value, offsetof(GCheader, marked));
    llvm::Value* isWhite = m_builder.CreateICmpNE(m_builder.CreateAnd(valueMarked, WHITEBITS), m_builder.getInt8(0));
    m_builder.CreateCondBr(isWhite, slow, done, unlikely());

    // the barrier never collects garbage or reads the stack, so nothing has to be spilled for it
    m_builder.SetInsertPoint(slow);
    call_native(barrier, m_types.voidTy, {m_state, object, value});
    m_builder.CreateBr(done);

    m_builder.SetInsertPoint(done);
}

llvm::MDNode* TranslationUnit::likely()
{
    if (!m_likely)
        m_likely = llvm::MDBuilder(m_module.getContext()).createLikelyBranchWeights();

    return m_likely;
}

llvm::MDNode* TranslationUnit::unlikely()
{
    if (!m_unlikely)
        m_unlikely = llvm::MDBuilder(m_module.getContext()).createUnlikelyBranchWeights();

    return m_unlikely;
}

llvm::BasicBlock* TranslationUnit::new_block(const llvm::Twine& name)
{
    return llvm::BasicBlock::Create(m_module.getContext(), name, m_function);
}

llvm::BasicBlock* TranslationUnit::block_at(const int pcpos)
{
    LUAU_ASSERT(pcpos >= 0 && static_cast<size_t>(pcpos) < m_blocks.size());

    const bool slow = m_generic || in_slow_path();

    if (pcpos > m_blockStart && pcpos <= m_blockFinish && slow)
        return generic_block_at(pcpos);

    // the back edge of the fast copy carries what the loop knows about its registers into the next iteration
    if (m_loopHeaders[pcpos] && !(m_pcpos >= pcpos && !slow))
    {
        if (llvm::BasicBlock* pad = loop_pad(pcpos))
            return pad;
    }

    return fast_block_at(pcpos);
}

llvm::BasicBlock* TranslationUnit::fast_block_at(const int pcpos)
{
    if (!m_blocks[pcpos])
        m_blocks[pcpos] = llvm::BasicBlock::Create(m_module.getContext(), "bc_" + llvm::Twine(pcpos), m_function);

    return m_blocks[pcpos];
}

llvm::BasicBlock* TranslationUnit::generic_block_at(const int pcpos)
{
    if (!m_genericBlocks[pcpos])
        m_genericBlocks[pcpos] = llvm::BasicBlock::Create(m_module.getContext(), "bc_" + llvm::Twine(pcpos) + ".generic", m_function);

    return m_genericBlocks[pcpos];
}

bool TranslationUnit::in_slow_path() const
{
    // the blocks after a helper call are usually a short chain, like the one past a check for a yield
    llvm::BasicBlock* block = current_block();

    for (int depth = 0; block && depth < 4; depth++, block = block->getSinglePredecessor())
    {
        if (m_slowBlocks.contains(block))
            return true;
    }

    return false;
}

void TranslationUnit::mark_entry_point(int pcpos)
{
    if (std::find(m_entryPoints.begin(), m_entryPoints.end(), pcpos) == m_entryPoints.end())
        m_entryPoints.push_back(pcpos);
}

llvm::BasicBlock* TranslationUnit::vm_exit(const int pcpos)
{
    auto [it, inserted] = m_exitBlocks.try_emplace(pcpos, nullptr);

    if (inserted)
    {
        it->second = llvm::BasicBlock::Create(m_module.getContext(), "exit_" + llvm::Twine(pcpos), m_function);

        llvm::IRBuilderBase::InsertPointGuard guard(m_builder);
        emit_vm_exit(pcpos, it->second);
    }

    return it->second;
}

llvm::BasicBlock* TranslationUnit::yield_exit()
{
    return m_yieldBlock;
}

void TranslationUnit::emit_prologue()
{
    const Proto* proto = m_bytecode.proto;

    m_builder.SetInsertPoint(m_entryBlock);
    m_baseSlot = m_builder.CreateAlloca(m_types.ptrTy, nullptr, "base.slot");

    m_regSlots.clear();
    m_regSlots.reserve(proto->maxstacksize);

    for (int i = 0; i < proto->maxstacksize; i++)
        m_regSlots.push_back(m_builder.CreateAlloca(m_types.tvalueTy, nullptr, "r" + llvm::Twine(i)));

    llvm::Value* base = m_types.load_state_base(m_builder, m_state);
    m_builder.CreateStore(base, m_baseSlot);

    // the closure of a frame never changes, even when the stack that holds it is reallocated
    llvm::Value* ci = m_types.load_state_ci(m_builder, m_state);
    llvm::Value* func = load_field(m_types.ptrTy, ci, offsetof(CallInfo, func), "func");
    m_closure = load_field(m_types.ptrTy, func, offsetof(TValue, value), "cl");
}

void TranslationUnit::emit_dispatch()
{
    LUAU_ASSERT(m_entryBlock && !m_entryBlock->hasTerminator());
    LUAU_ASSERT(!m_entryPoints.empty());

    m_builder.SetInsertPoint(m_entryBlock);

    llvm::SwitchInst* dispatch = m_builder.CreateSwitch(m_entrypc, m_bailBlock, static_cast<unsigned>(m_entryPoints.size()));

    // entering in the middle of a bytecode block goes to its generic copy, which keeps the fast copy a straight line
    for (const int pcpos : m_entryPoints)
    {
        llvm::BasicBlock* block = m_genericBlocks[pcpos] ? m_genericBlocks[pcpos] : m_blocks[pcpos];
        LUAU_ASSERT(block);

        if (block == m_blocks[pcpos] && m_loopHeaders[pcpos])
        {
            if (llvm::BasicBlock* pad = loop_pad(pcpos))
                block = pad;
        }

        llvm::BasicBlock* enter = new_block("enter." + llvm::Twine(pcpos));
        dispatch->addCase(m_builder.getInt32(pcpos), enter);

        // the other registers are dead, and they are written before anything can spill them: a register only stays
        // dirty across a block start while it's live, and a live one that isn't loaded here is written on the way
        m_builder.SetInsertPoint(enter);
        llvm::Value* base = load_base();
        const std::bitset<256> regs = m_liveIn[pcpos] | m_bytecode.cfg.captured.regs | m_dirtyIn[pcpos];

        for (int reg = 0; reg < m_bytecode.proto->maxstacksize; reg++)
        {
            if (regs.test(reg))
                m_types.copy_tvalue(m_builder, m_regSlots[reg], m_types.stack_slot(m_builder, base, reg));
        }

        emit_integer_loop_entry(pcpos);

        if (pcpos == 0 && m_tiersUp)
            emit_tier_up(pcpos);

        m_builder.CreateBr(block);
        m_builder.SetInsertPoint(m_entryBlock);
    }
}

// A slow path continues in a generic copy of the rest of its bytecode block instead of merging back into the code it
// left. The fast copy then has a single way into each instruction, so everything known about the registers stays
// known until the end of the block, like the linear blocks of CodeGen
void TranslationUnit::translate_block(const uint32_t blockIdx)
{
    const Proto* proto = m_bytecode.proto;

    const auto& [startpc, finishpc] = m_bytecode.bcBlocks[blockIdx];
    LUAU_ASSERT(startpc >= 0 && finishpc >= startpc && finishpc < proto->sizecode);

    m_blockStart = startpc;
    m_blockFinish = finishpc;

    translate_range(startpc, finishpc, /* generic */ false);

    for (const int pcpos : m_entryPoints)
    {
        if (pcpos > startpc && pcpos <= finishpc)
            generic_block_at(pcpos);
    }

    for (int pcpos = startpc + 1; pcpos <= finishpc; pcpos++)
    {
        if (m_genericBlocks[pcpos])
        {
            translate_range(pcpos, finishpc, /* generic */ true);
            break;
        }
    }

    m_blockStart = -1;
    m_blockFinish = -1;
}

void TranslationUnit::translate_range(const int startpc, const int finishpc, const bool generic)
{
    const Proto* proto = m_bytecode.proto;

    m_generic = generic;
    m_builder.ClearInsertionPoint();

    int pcpos = startpc;
    while (pcpos <= finishpc)
    {
        // falling through out of a slow path goes on in the generic copy
        if (llvm::BasicBlock* previous = current_block(); previous && !previous->hasTerminator())
            m_builder.CreateBr(block_at(pcpos));

        const Instruction* pc = &proto->code[pcpos];
        const auto op = static_cast<LuauOpcode>(LUAU_INSN_OP(*pc));

        // only the instructions before can branch into the fast copy, so an instruction that none of them reach there
        // can be left out; a later one can still be the target of a fastcall that skips its CALL
        if (!generic && pcpos != startpc && (!m_blocks[pcpos] || llvm::pred_empty(m_blocks[pcpos])))
        {
            m_builder.ClearInsertionPoint();
            pcpos += getOpLength(op);
            continue;
        }

        llvm::BasicBlock* target = generic ? generic_block_at(pcpos) : fast_block_at(pcpos);
        m_builder.SetInsertPoint(target);

        if (m_profiling)
            emit_type_feedback(pcpos);

        m_pcpos = pcpos;
        const bool translated = translate_inst(op, pc, pcpos);
        m_pcpos = -1;

        if (!translated)
        {
            m_builder.CreateBr(vm_exit(pcpos));
            break;
        }

        pcpos += getOpLength(op);
    }

    if (llvm::BasicBlock* previous = current_block(); previous && !previous->hasTerminator())
    {
        LUAU_ASSERT(pcpos < proto->sizecode);

        m_builder.CreateBr(block_at(pcpos));
    }

    m_generic = false;
}

bool TranslationUnit::translate_inst(const LuauOpcode op, const Instruction* pc, const int pcpos)
{
    LUAU_ASSERT(op < LOP__COUNT);
    LUAU_ASSERT(pc == &m_bytecode.proto->code[pcpos]);

    switch (op)
    {
    case LOP_NOP:
    // captures are consumed by the NEWCLOSURE and DUPCLOSURE helpers
    case LOP_CAPTURE:
        return true;
    case LOP_BREAK:
        // breakpoints take native code away, the interpreter executes the original instruction
        return false;
    case LOP_LOADNIL:
        lower_inst_loadnil(*this, pc, pcpos);
        return true;
    case LOP_LOADB:
        lower_inst_loadb(*this, pc, pcpos);
        return true;
    case LOP_LOADN:
        lower_inst_loadn(*this, pc, pcpos);
        return true;
    case LOP_LOADK:
    case LOP_LOADKX:
        lower_inst_loadk(*this, pc, pcpos);
        return true;
    case LOP_MOVE:
        lower_inst_move(*this, pc, pcpos);
        return true;
    case LOP_GETGLOBAL:
        lower_inst_getglobal(*this, pc, pcpos);
        return true;
    case LOP_SETGLOBAL:
        lower_inst_helper(*this, reinterpret_cast<const void*>(&execute_setglobal), pcpos);
        return true;
    case LOP_GETUPVAL:
        lower_inst_getupval(*this, pc, pcpos);
        return true;
    case LOP_SETUPVAL:
        lower_inst_setupval(*this, pc, pcpos);
        return true;
    case LOP_CLOSEUPVALS:
        lower_inst_helper(*this, reinterpret_cast<const void*>(&execute_closeupvals), pcpos);
        return true;
    case LOP_GETIMPORT:
        lower_inst_getimport(*this, pc, pcpos);
        return true;
    case LOP_GETTABLE:
        lower_inst_gettable(*this, pc, pcpos);
        return true;
    case LOP_SETTABLE:
        lower_inst_settable(*this, pc, pcpos);
        return true;
    case LOP_GETTABLEKS:
        lower_inst_gettableks(*this, pc, pcpos);
        return true;
    case LOP_SETTABLEKS:
        lower_inst_settableks(*this, pc, pcpos);
        return true;
    case LOP_GETTABLEN:
        lower_inst_gettablen(*this, pc, pcpos);
        return true;
    case LOP_SETTABLEN:
        lower_inst_settablen(*this, pc, pcpos);
        return true;
    case LOP_NEWCLOSURE:
        lower_inst_helper(*this, reinterpret_cast<const void*>(&execute_newclosure), pcpos);
        return true;
    case LOP_NAMECALL:
        lower_inst_namecall(*this, pc, pcpos);
        return true;
    case LOP_CALL:
    case LOP_CALLFB:
        lower_inst_call(*this, pc, pcpos);
        return true;
    case LOP_RETURN:
        lower_inst_return(*this, pc, pcpos);
        return true;
    case LOP_JUMP:
    case LOP_JUMPBACK:
    case LOP_JUMPX:
        lower_inst_jump(*this, pc, pcpos);
        return true;
    case LOP_JUMPIF:
    case LOP_JUMPIFNOT:
        lower_inst_jumpif(*this, pc, pcpos);
        return true;
    case LOP_JUMPIFEQ:
    case LOP_JUMPIFNOTEQ:
        lower_inst_jumpifeq(*this, pc, pcpos);
        return true;
    case LOP_JUMPIFLE:
    case LOP_JUMPIFLT:
    case LOP_JUMPIFNOTLE:
    case LOP_JUMPIFNOTLT:
        lower_inst_jumpifcmp(*this, pc, pcpos);
        return true;
    case LOP_JUMPXEQKNIL:
    case LOP_JUMPXEQKB:
    case LOP_JUMPXEQKN:
    case LOP_JUMPXEQKS:
        lower_inst_jumpxeqk(*this, pc, pcpos);
        return true;
    case LOP_ADD:
    case LOP_SUB:
    case LOP_MUL:
    case LOP_DIV:
    case LOP_IDIV:
    case LOP_MOD:
    case LOP_POW:
    case LOP_ADDK:
    case LOP_SUBK:
    case LOP_MULK:
    case LOP_DIVK:
    case LOP_IDIVK:
    case LOP_MODK:
    case LOP_POWK:
    case LOP_SUBRK:
    case LOP_DIVRK:
    case LOP_MINUS:
        lower_inst_binary(*this, pc, pcpos);
        return true;
    case LOP_AND:
    case LOP_OR:
    case LOP_ANDK:
    case LOP_ORK:
        lower_inst_logical(*this, pc, pcpos);
        return true;
    case LOP_CONCAT:
        lower_inst_helper(*this, reinterpret_cast<const void*>(&execute_concat), pcpos);
        return true;
    case LOP_NOT:
        lower_inst_not(*this, pc, pcpos);
        return true;
    case LOP_LENGTH:
        lower_inst_length(*this, pc, pcpos);
        return true;
    case LOP_NEWTABLE:
    case LOP_DUPTABLE:
        lower_inst_newtable(*this, pc, pcpos);
        return true;
    case LOP_SETLIST:
        lower_inst_helper(*this, reinterpret_cast<const void*>(&execute_setlist), pcpos);
        return true;
    case LOP_FORNPREP:
        lower_inst_fornprep(*this, pc, pcpos);
        return true;
    case LOP_FORNLOOP:
        lower_inst_fornloop(*this, pc, pcpos);
        return true;
    case LOP_FORGPREP:
    case LOP_FORGPREP_NEXT:
    case LOP_FORGPREP_INEXT:
        lower_inst_forgprep(*this, pc, pcpos);
        return true;
    case LOP_FORGLOOP:
        lower_inst_forgloop(*this, pc, pcpos);
        return true;
    case LOP_GETVARARGS:
        lower_inst_helper(*this, reinterpret_cast<const void*>(&execute_getvarargs), pcpos);
        return true;
    case LOP_PREPVARARGS:
        lower_inst_helper(*this, reinterpret_cast<const void*>(&execute_prepvarargs), pcpos);
        return true;
    case LOP_DUPCLOSURE:
        lower_inst_helper(*this, reinterpret_cast<const void*>(&execute_dupclosure), pcpos);
        return true;
    case LOP_FASTCALL:
    case LOP_FASTCALL1:
    case LOP_FASTCALL2:
    case LOP_FASTCALL2K:
    case LOP_FASTCALL3:
        lower_inst_fastcall(*this, pc, pcpos);
        return true;
    case LOP_FASTPCALL:
        lower_inst_fastpcall(*this, pc, pcpos);
        return true;
    case LOP_COVERAGE:
        lower_inst_coverage(*this, pc, pcpos);
        return true;
    case LOP_GETUDATAKS:
        lower_inst_helper(*this, reinterpret_cast<const void*>(&execute_getudataks), pcpos);
        return true;
    case LOP_SETUDATAKS:
        lower_inst_helper(*this, reinterpret_cast<const void*>(&execute_setudataks), pcpos);
        return true;
    case LOP_NAMECALLUDATA:
        lower_inst_namecalludata(*this, pc, pcpos);
        return true;
    case LOP_NEWCLASS:
        lower_inst_helper(*this, reinterpret_cast<const void*>(&execute_newclass), pcpos);
        return true;
    case LOP_NEWCLASSMEMBER:
        lower_inst_helper(*this, reinterpret_cast<const void*>(&execute_newclassmember), pcpos);
        return true;
    case LOP_CONSTRUCT:
        lower_inst_helper(*this, reinterpret_cast<const void*>(&execute_construct), pcpos);
        return true;
    case LOP_FINCONSTRUCT:
        lower_inst_finconstruct(*this, pc, pcpos);
        return true;
    case LOP_CMPPROTO:
        lower_inst_cmpproto(*this, pc, pcpos);
        return true;
    case LOP_NATIVECALL:
    case LOP__COUNT:
        break;
    }

    LUAU_ASSERT(!"Opcode is never part of Proto::code");
    return false;
}

void TranslationUnit::emit_vm_exit(const int pcpos, llvm::BasicBlock* block)
{
    const Proto* proto = m_bytecode.proto;

    m_builder.SetInsertPoint(block);

    // the tags that failed a guard tell the next compilation not to speculate on them
    if (m_nproto.feedback)
        emit_type_feedback(pcpos);

    DeoptPoint deopt = build_deopt_point(pcpos);

    llvm::Value* base = load_base();

    for (const DeoptValue& value : deopt.values)
    {
        LUAU_ASSERT(value.kind == DeoptValue::Kind::Register);

        m_types.copy_tvalue(m_builder, m_types.stack_slot(m_builder, base, value.reg), reg_slot(value.reg));
    }

    if (m_counters)
        emit_counter(&m_counters->vmExits[LUAU_INSN_OP(proto->code[pcpos])]);

    if (m_speculate)
    {
        // a function that keeps failing its guards gets recompiled without them
        llvm::Value* address = const_ptr(&m_nproto.exits);
        llvm::Value* exits = m_builder.CreateAdd(m_builder.CreateLoad(m_types.i32Ty, address), m_builder.getInt32(1));
        m_builder.CreateStore(exits, address);

        llvm::BasicBlock* recompile = new_block("exit.recompile");
        llvm::BasicBlock* leave = new_block("exit.leave");

        m_builder.CreateCondBr(m_builder.CreateICmpEQ(exits, m_builder.getInt32(FInt::LuauJitExitRecompileThreshold)), recompile, leave, unlikely());

        m_builder.SetInsertPoint(recompile);
        call_native(reinterpret_cast<const void*>(&handle_exit_threshold), m_types.voidTy, {m_state, const_ptr(proto)});
        m_builder.CreateBr(leave);

        m_builder.SetInsertPoint(leave);
    }

    llvm::Value* ci = m_types.load_state_ci(m_builder, m_state);
    m_types.store_ci_savedpc(m_builder, ci, const_ptr(&proto->code[pcpos]));
    m_builder.CreateRet(m_builder.getInt32(1));

    m_nproto.deopts.push_back(std::move(deopt));
}

void TranslationUnit::emit_counter(uint64_t* counter)
{
    llvm::Value* address = const_ptr(counter);
    llvm::Value* value = m_builder.CreateLoad(m_types.i64Ty, address);
    m_builder.CreateStore(m_builder.CreateAdd(value, m_builder.getInt64(1)), address);
}

void TranslationUnit::emit_type_feedback(const int pcpos)
{
    const std::array<int, 2> operands = feedback_operands(&m_bytecode.proto->code[pcpos]);

    llvm::Value* seen = nullptr;

    for (int slot = 0; slot < 2; slot++)
    {
        if (operands[slot] < 0)
            continue;

        llvm::Value* bit = m_builder.CreateShl(m_builder.getInt32(1u << (16 * slot)), load_tag(operands[slot]));
        seen = seen ? m_builder.CreateOr(seen, bit) : bit;
    }

    if (!seen)
        return;

    llvm::Value* address = const_ptr(&(*m_nproto.feedback)[pcpos]);
    m_builder.CreateStore(m_builder.CreateOr(m_builder.CreateLoad(m_types.i32Ty, address), seen), address);
}

void TranslationUnit::emit_operand_feedback(const int pcpos, const int slot)
{
    if (!m_nproto.feedback)
        return;

    const int reg = feedback_operands(&m_bytecode.proto->code[pcpos])[slot];
    LUAU_ASSERT(reg >= 0);

    llvm::Value* bit = m_builder.CreateShl(m_builder.getInt32(1u << (16 * slot)), load_tag(reg));
    llvm::Value* address = const_ptr(&(*m_nproto.feedback)[pcpos]);
    m_builder.CreateStore(m_builder.CreateOr(m_builder.CreateLoad(m_types.i32Ty, address), bit), address);
}

// The operands, in the slots of feedback_operands, that the translation of the instruction guards to be numbers with
// an exit; -1 for the others
static std::array<int, 2> exiting_number_operands(const TranslationUnit& unit, const Instruction* pc, const BytecodeTypes& types)
{
    const Proto* proto = unit.bytecode().proto;
    const std::array<int, 2> operands = feedback_operands(pc);

    auto guarded = [&](const int slot, const uint8_t type)
    {
        return unit.speculates_on(type, LBC_TYPE_NUMBER) ? operands[slot] : -1;
    };

    switch (LUAU_INSN_OP(*pc))
    {
    case LOP_ADD:
    case LOP_SUB:
    case LOP_MUL:
    case LOP_DIV:
    case LOP_IDIV:
    case LOP_MOD:
    case LOP_POW:
        // vector arithmetic picks its own guards
        if (types.a == LBC_TYPE_VECTOR || types.b == LBC_TYPE_VECTOR)
            return {-1, -1};
        return {guarded(0, types.a), guarded(1, types.b)};
    case LOP_ADDK:
    case LOP_SUBK:
    case LOP_MULK:
    case LOP_DIVK:
    case LOP_IDIVK:
    case LOP_MODK:
    case LOP_POWK:
        // a constant that isn't a number leaves the whole instruction to the helper
        if (types.a == LBC_TYPE_VECTOR || !ttisnumber(&proto->k[LUAU_INSN_C(*pc)]))
            return {-1, -1};
        return {guarded(0, types.a), -1};
    case LOP_SUBRK:
    case LOP_DIVRK:
        if (types.b == LBC_TYPE_VECTOR || !ttisnumber(&proto->k[LUAU_INSN_B(*pc)]))
            return {-1, -1};
        return {-1, guarded(1, types.b)};
    case LOP_MINUS:
        if (types.a == LBC_TYPE_VECTOR)
            return {-1, -1};
        return {guarded(0, types.a), -1};
    case LOP_JUMPIFLE:
    case LOP_JUMPIFLT:
    case LOP_JUMPIFNOTLE:
    case LOP_JUMPIFNOTLT:
        return {guarded(0, types.a), guarded(1, types.b)};
    case LOP_JUMPIFEQ:
    case LOP_JUMPIFNOTEQ:
        // equality only speculates when both sides are numbers
        if (guarded(0, types.a) < 0 || guarded(1, types.b) < 0)
            return {-1, -1};
        return {guarded(0, types.a), guarded(1, types.b)};
    default:
        return {-1, -1};
    }
}

// Exiting at the header instead of at the guard is always right: nothing of the iteration has happened yet
llvm::BasicBlock* TranslationUnit::loop_pad(const int header)
{
    if (auto it = m_loopPads.find(header); it != m_loopPads.end())
        return it->second;

    struct Guard
    {
        int reg;
        int pcpos;
        int slot;
    };

    std::vector<Guard> guards;

    const Proto* proto = m_bytecode.proto;
    const uint32_t blockIdx = m_bytecode.cfg.blockAtPc[header];

    if (m_speculate && blockIdx != kNoAssociatedBlock)
    {
        // every instruction of the first block runs in every iteration, as far as one gets without leaving the loop;
        // captured registers can change under a call in between
        std::bitset<256> defined = m_bytecode.cfg.captured.regs;

        for (int pcpos = header; pcpos <= m_bytecode.bcBlocks[blockIdx].finishpc;)
        {
            const Instruction* pc = &proto->code[pcpos];
            const std::array<int, 2> operands = exiting_number_operands(*this, pc, m_bytecode.bcTypes[pcpos]);

            for (int slot = 0; slot < 2; slot++)
            {
                const int reg = operands[slot];
                const bool seen = std::any_of(
                    guards.begin(),
                    guards.end(),
                    [&](const Guard& guard)
                    {
                        return guard.reg == reg;
                    }
                );

                if (reg >= 0 && !defined.test(reg) && !seen)
                    guards.push_back({reg, pcpos, slot});
            }

            DirtyVisitor visitor{defined, proto->maxstacksize};
            visit_bytecode_reg_defs_uses(visitor, m_bytecode.proto, pcpos);

            pcpos += getOpLength(LuauOpcode(LUAU_INSN_OP(*pc)));
        }
    }

    llvm::BasicBlock* pad = nullptr;

    if (!guards.empty())
    {
        llvm::IRBuilderBase::InsertPointGuard insertPoint(m_builder);

        pad = new_block("loop.pad." + llvm::Twine(header));
        m_builder.SetInsertPoint(pad);

        for (const Guard& guard : guards)
        {
            llvm::BasicBlock* next = new_block("loop.pad.next");
            llvm::BasicBlock* fail = new_block("loop.pad.fail");
            m_builder.CreateCondBr(m_builder.CreateICmpEQ(load_tag(guard.reg), m_builder.getInt32(LUA_TNUMBER)), next, fail, likely());

            // the next compilation learns about the operand of the guard that would have failed
            m_builder.SetInsertPoint(fail);
            emit_operand_feedback(guard.pcpos, guard.slot);
            m_builder.CreateBr(vm_exit(header));

            // a constant tag on every way into the loop lets the loop fold its own guard
            m_builder.SetInsertPoint(next);
            store_tag(guard.reg, LUA_TNUMBER);
        }

        m_builder.CreateBr(fast_block_at(header));
    }

    m_loopPads[header] = pad;
    return pad;
}

// every version of the code records these, so a recompilation after exits sees what made the code exit
void TranslationUnit::emit_feedback_bits(const int pcpos, llvm::Value* bits)
{
    if (!m_nproto.feedback)
        return;

    LUAU_ASSERT(feedback_operands(&m_bytecode.proto->code[pcpos]) == (std::array<int, 2>{-1, -1}));

    llvm::Value* address = const_ptr(&(*m_nproto.feedback)[pcpos]);
    m_builder.CreateStore(m_builder.CreateOr(m_builder.CreateLoad(m_types.i32Ty, address), bits), address);
}

uint32_t TranslationUnit::feedback_at(const int pcpos) const
{
    return m_nproto.feedback ? (*m_nproto.feedback)[pcpos] : 0;
}

void TranslationUnit::emit_tier_up(const int pcpos)
{
    llvm::Value* address = const_ptr(&m_nproto.tierUpBudget);
    llvm::Value* budget = m_builder.CreateSub(m_builder.CreateLoad(m_types.i32Ty, address), m_builder.getInt32(1));
    m_builder.CreateStore(budget, address);

    llvm::BasicBlock* tierUp = new_block("tierup");
    llvm::BasicBlock* transfer = new_block("tierup.switch");
    llvm::BasicBlock* resume = new_block("tierup.done");

    // the budget stays below zero once the code is replaced, so every frame still running it switches at its next back
    // edge; a failed compilation resets it
    m_builder.CreateCondBr(m_builder.CreateICmpSLE(budget, m_builder.getInt32(0)), tierUp, resume, unlikely());

    // compiling doesn't touch the Lua stack, so nothing has to be spilled for it
    m_builder.SetInsertPoint(tierUp);
    call_native(reinterpret_cast<const void*>(&handle_tier_up), m_types.voidTy, {m_state, const_ptr(&m_nproto)});

    llvm::Value* proto = const_ptr(m_bytecode.proto);
    llvm::Value* execdata = load_field(m_types.ptrTy, proto, offsetof(Proto, execdata), "execdata");
    llvm::Value* target = load_field(m_types.ptrTy, proto, offsetof(Proto, exectarget), "exectarget");

    // a breakpoint clears exectarget, and then this code keeps running the frame
    llvm::Value* replaced = m_builder.CreateAnd(m_builder.CreateICmpNE(execdata, const_ptr(&m_nproto)), m_builder.CreateIsNotNull(target));
    m_builder.CreateCondBr(replaced, transfer, resume);

    // the pc is an entry point of every version of the code: the start of the function, or the safepoint of a back edge
    m_builder.SetInsertPoint(transfer);
    spill(pcpos);

    // where the interpreter continues when the new code bails out
    m_types.store_ci_savedpc(m_builder, m_types.load_state_ci(m_builder, m_state), const_ptr(&m_bytecode.proto->code[pcpos]));

    if (m_counters)
        emit_counter(&m_counters->nativeTransfers);

    llvm::CallInst* call = m_builder.CreateCall(m_types.nativeEntryTy, target, {m_state, m_builder.getInt32(pcpos)});
    call->setTailCallKind(llvm::CallInst::TCK_MustTail);
    m_builder.CreateRet(call);

    m_builder.SetInsertPoint(resume);
}

DeoptPoint TranslationUnit::build_deopt_point(const int pcpos) const
{
    const Proto* proto = m_bytecode.proto;

    // registers that aren't dirty already hold their value on the stack
    const std::bitset<256>& dirty = m_dirtyIn[pcpos];

    DeoptPoint result;
    result.pc = static_cast<uint32_t>(pcpos);

    for (int i = 0; i < proto->maxstacksize; i++)
    {
        if (!dirty.test(i))
            continue;

        DeoptValue value;
        value.kind = DeoptValue::Kind::Register;
        value.reg = static_cast<uint8_t>(i);

        result.values.push_back(value);
    }

    return result;
}

static int next_pc(const Instruction* pc, const int pcpos)
{
    return pcpos + getOpLength(static_cast<LuauOpcode>(LUAU_INSN_OP(*pc)));
}

static llvm::Value* tag_is(TranslationUnit& unit, const int reg, const int tag)
{
    return unit.builder().CreateICmpEQ(unit.load_tag(reg), unit.builder().getInt32(tag));
}

// Fast path in the current block, then the whole instruction through 'helper' when 'fast' doesn't hold
template<typename F>
static void lower_with_fallback(TranslationUnit& unit, llvm::Value* fast, const void* helper, const int pcpos, const int nextpc, F&& lowerFast)
{
    llvm::IRBuilder<>& build = unit.builder();

    llvm::BasicBlock* fastBlock = unit.new_block("fast");
    llvm::BasicBlock* slowBlock = unit.new_block("slow");
    llvm::BasicBlock* next = unit.block_at(nextpc);

    build.CreateCondBr(fast, fastBlock, slowBlock, unit.likely());

    build.SetInsertPoint(fastBlock);
    lowerFast();
    if (!unit.current_block()->hasTerminator())
        build.CreateBr(next);

    build.SetInsertPoint(slowBlock);
    unit.spill(pcpos);
    unit.call_helper(helper, pcpos, false);
    unit.reload(pcpos);
    build.CreateBr(next);
}

static bool speculates_on_tag(const TranslationUnit& unit, const uint8_t type, const int tag)
{
    return (tag == LUA_TNUMBER && unit.speculates_on(type, LBC_TYPE_NUMBER)) || (tag == LUA_TVECTOR && unit.speculates_on(type, LBC_TYPE_VECTOR));
}

// An operand that the bytecode types say has its tag leaves generated code when it doesn't, which keeps the slow path
// from merging back into the code that follows
void check_operand_tags(
    TranslationUnit& unit,
    const int pcpos,
    const llvm::ArrayRef<TypedOperand> operands,
    llvm::BasicBlock* fast,
    llvm::BasicBlock* slow
)
{
    llvm::IRBuilder<>& build = unit.builder();

    std::vector<TypedOperand> checks;
    for (const TypedOperand& operand : operands)
    {
        if (operand.reg < 0)
            continue;

        bool duplicate = false;
        for (const TypedOperand& check : checks)
            duplicate |= check.reg == operand.reg;

        if (!duplicate)
            checks.push_back(operand);
    }

    if (checks.empty())
    {
        build.CreateBr(fast);
        return;
    }

    for (size_t i = 0; i < checks.size(); i++)
    {
        llvm::BasicBlock* next = i + 1 == checks.size() ? fast : unit.new_block("tag.check");
        llvm::BasicBlock* failure = speculates_on_tag(unit, checks[i].type, checks[i].tag) ? unit.vm_exit(pcpos) : slow;

        build.CreateCondBr(tag_is(unit, checks[i].reg, checks[i].tag), next, failure, unit.likely());
        build.SetInsertPoint(next);
    }
}

void lower_inst_loadnil(TranslationUnit& unit, const Instruction* pc, int pcpos)
{
    unit.store_nil(LUAU_INSN_A(*pc));
}

void lower_inst_loadb(TranslationUnit& unit, const Instruction* pc, int pcpos)
{
    const int ra = LUAU_INSN_A(*pc);

    unit.store_boolean(ra, unit.builder().getInt32(LUAU_INSN_B(*pc)));

    if (const int skip = LUAU_INSN_C(*pc); skip != 0)
        unit.builder().CreateBr(unit.block_at(pcpos + 1 + skip));
}

void lower_inst_loadn(TranslationUnit& unit, const Instruction* pc, int pcpos)
{
    unit.store_number(LUAU_INSN_A(*pc), llvm::ConstantFP::get(unit.types().doubleTy, double(LUAU_INSN_D(*pc))));
}

void lower_inst_loadk(TranslationUnit& unit, const Instruction* pc, int pcpos)
{
    const int kidx = LUAU_INSN_OP(*pc) == LOP_LOADKX ? static_cast<int>(pc[1]) : LUAU_INSN_D(*pc);

    unit.store_reg(LUAU_INSN_A(*pc), unit.constant(kidx));
}

void lower_inst_move(TranslationUnit& unit, const Instruction* pc, int pcpos)
{
    unit.store_reg(LUAU_INSN_A(*pc), unit.load_reg(LUAU_INSN_B(*pc)));
}

// whether the node has the string 'key' with a value that isn't nil
static llvm::Value* node_key_is(TranslationUnit& unit, llvm::Value* node, const TString* key)
{
    llvm::IRBuilder<>& build = unit.builder();
    const VmTypes& types = unit.types();

    llvm::Value* keyTag = build.CreateAnd(unit.load_field(types.i32Ty, node, kNodeKeyTagOffset), 0xf);
    llvm::Value* keyValue = unit.load_field(types.ptrTy, node, offsetof(LuaNode, key) + offsetof(TKey, value));

    return build.CreateAnd(build.CreateICmpEQ(keyTag, build.getInt32(LUA_TSTRING)), build.CreateICmpEQ(keyValue, unit.const_ptr(key)));
}

static llvm::Value* node_value_is_nil(TranslationUnit& unit, llvm::Value* node)
{
    llvm::Value* valueTag = unit.load_field(unit.types().i32Ty, node, offsetof(LuaNode, val) + offsetof(TValue, tt));

    return unit.builder().CreateICmpEQ(valueTag, unit.builder().getInt32(LUA_TNIL));
}

static llvm::Value* node_holds(TranslationUnit& unit, llvm::Value* node, const TString* key)
{
    return unit.builder().CreateAnd(node_key_is(unit, node, key), unit.builder().CreateNot(node_value_is_nil(unit, node)), "hit");
}

// The node that the cached slot in the instruction's C points to, like the interpreter's predictive lookups
static llvm::Value* cached_slot_node(TranslationUnit& unit, llvm::Value* table, const int pcpos)
{
    llvm::IRBuilder<>& build = unit.builder();
    const VmTypes& types = unit.types();

    // C gets patched by the helpers whenever the cached slot misses
    llvm::Value* insn = build.CreateLoad(types.i32Ty, unit.insn_ptr(pcpos), "insn");
    llvm::Value* cachedSlot = build.CreateLShr(insn, 24);

    llvm::Value* nodemask = build.CreateZExt(unit.load_field(types.i8Ty, table, offsetof(LuaTable, nodemask8)), types.i32Ty);
    llvm::Value* slot = build.CreateZExt(build.CreateAnd(cachedSlot, nodemask), types.i64Ty);

    llvm::Value* nodes = unit.load_field(types.ptrTy, table, offsetof(LuaTable, node), "node");
    return build.CreateInBoundsGEP(types.i8Ty, nodes, build.CreateMul(slot, build.getInt64(sizeof(LuaNode))), "n");
}

// Looks up 'key' in the cached slot; returns the condition for a hit and the address of the node's value
static std::pair<llvm::Value*, llvm::Value*> lookup_cached_slot(TranslationUnit& unit, llvm::Value* table, const int pcpos, const TString* key)
{
    llvm::Value* node = cached_slot_node(unit, table, pcpos);

    return {node_holds(unit, node, key), unit.types().field_ptr(unit.builder(), node, offsetof(LuaNode, val))};
}

static const TString* string_constant(TranslationUnit& unit, const int kidx)
{
    const TValue& value = unit.bytecode().proto->k[kidx];
    LUAU_ASSERT(ttisstring(&value));

    return tsvalue(&value);
}

void lower_inst_getglobal(TranslationUnit& unit, const Instruction* pc, int pcpos)
{
    llvm::IRBuilder<>& build = unit.builder();
    const VmTypes& types = unit.types();

    const int ra = LUAU_INSN_A(*pc);

    llvm::Value* env = unit.load_field(types.ptrTy, unit.closure(), offsetof(Closure, env), "env");
    auto [hit, value] = lookup_cached_slot(unit, env, pcpos, string_constant(unit, static_cast<int>(pc[1])));

    lower_with_fallback(
        unit,
        hit,
        reinterpret_cast<const void*>(&execute_getglobal),
        pcpos,
        next_pc(pc, pcpos),
        [&, value = value]
        {
            unit.store_reg(ra, build.CreateLoad(types.tvalueTy, value));
        }
    );
}

void lower_inst_getupval(TranslationUnit& unit, const Instruction* pc, int pcpos)
{
    llvm::IRBuilder<>& build = unit.builder();
    const VmTypes& types = unit.types();

    const int ra = LUAU_INSN_A(*pc);
    const int up = LUAU_INSN_B(*pc);

    llvm::Value* ur = types.field_ptr(build, unit.closure(), offsetof(Closure, l.uprefs) + sizeof(TValue) * up, "ur");

    // TValue* v = ttisupval(ur) ? upvalue(ur)->v : ur
    llvm::Value* isUpval = build.CreateICmpEQ(unit.load_field(types.i32Ty, ur, offsetof(TValue, tt)), build.getInt32(LUA_TUPVAL));

    llvm::BasicBlock* open = unit.new_block("upval.ref");
    llvm::BasicBlock* join = unit.new_block("upval.load");
    llvm::BasicBlock* origin = unit.current_block();

    build.CreateCondBr(isUpval, open, join);

    build.SetInsertPoint(open);
    llvm::Value* upval = unit.load_field(types.ptrTy, ur, offsetof(TValue, value));
    llvm::Value* v = unit.load_field(types.ptrTy, upval, offsetof(UpVal, v), "v");
    build.CreateBr(join);

    build.SetInsertPoint(join);
    llvm::PHINode* source = build.CreatePHI(types.ptrTy, 2);
    source->addIncoming(ur, origin);
    source->addIncoming(v, open);

    unit.store_reg(ra, build.CreateLoad(types.tvalueTy, source));
}

void lower_inst_setupval(TranslationUnit& unit, const Instruction* pc, int pcpos)
{
    llvm::IRBuilder<>& build = unit.builder();
    const VmTypes& types = unit.types();

    const int ra = LUAU_INSN_A(*pc);
    const int up = LUAU_INSN_B(*pc);

    // the compiler only emits SETUPVAL for upvalues captured by reference, and those always hold an UpVal; an open one
    // points into the frame of an enclosing function, which never is this one
    llvm::Value* ur = types.field_ptr(build, unit.closure(), offsetof(Closure, l.uprefs) + sizeof(TValue) * up, "ur");
    llvm::Value* upval = unit.load_field(types.ptrTy, ur, offsetof(TValue, value), "uv");
    llvm::Value* v = unit.load_field(types.ptrTy, upval, offsetof(UpVal, v), "v");

    types.copy_tvalue(build, v, unit.reg_slot(ra));

    // luaC_barrier
    unit.emit_barrier(upval, ra, reinterpret_cast<const void*>(&luaC_barrierf));
}

void lower_inst_getimport(TranslationUnit& unit, const Instruction* pc, int pcpos)
{
    llvm::IRBuilder<>& build = unit.builder();
    const VmTypes& types = unit.types();

    const int ra = LUAU_INSN_A(*pc);
    const int kidx = LUAU_INSN_D(*pc);

    const void* helper = reinterpret_cast<const void*>(&execute_getimport);

    // imports are resolved once at load time, a failed resolution is retried on every execution
    if (ttisnil(&unit.bytecode().proto->k[kidx]))
    {
        lower_inst_helper(unit, helper, pcpos);
        return;
    }

    lower_with_fallback(
        unit,
        unit.is_safe_env(),
        helper,
        pcpos,
        next_pc(pc, pcpos),
        [&]
        {
            unit.store_reg(ra, unit.constant(kidx));
        }
    );
}

// The component that lvmexecute.cpp reads inline for a vector index, or -1
static int vector_component(const TValue& key)
{
    const TString* name = tsvalue(&key);
    int ic = (getstr(name)[0] | ' ') - 'x';

#if LUA_VECTOR_SIZE == 4
    // 'w' is before 'x' in ascii
    if (ic == -1)
        ic = 3;
#endif

    return name->len == 1 && unsigned(ic) < LUA_VECTOR_SIZE ? ic : -1;
}

void lower_inst_gettableks(TranslationUnit& unit, const Instruction* pc, int pcpos)
{
    llvm::IRBuilder<>& build = unit.builder();
    const VmTypes& types = unit.types();

    const int ra = LUAU_INSN_A(*pc);
    const int rb = LUAU_INSN_B(*pc);
    const int nextpc = next_pc(pc, pcpos);

    const void* helper = reinterpret_cast<const void*>(&execute_gettableks);

    const BytecodeTypes& bcTypes = unit.bytecode().bcTypes[pcpos];

    if (const int component = bcTypes.a == LBC_TYPE_VECTOR ? vector_component(unit.bytecode().proto->k[pc[1]]) : -1; component >= 0)
    {
        llvm::BasicBlock* fast = unit.new_block("vector");
        llvm::BasicBlock* slow = unit.new_block("slow");

        check_operand_tags(unit, pcpos, {{rb, bcTypes.a, LUA_TVECTOR}}, fast, slow);

        build.SetInsertPoint(fast);
        llvm::Value* value = build.CreateExtractElement(unit.load_vector(rb), uint64_t(component));
        unit.store_number(ra, build.CreateFPExt(value, types.doubleTy));
        build.CreateBr(unit.block_at(nextpc));

        build.SetInsertPoint(slow);
        unit.spill(pcpos);
        unit.call_helper(helper, pcpos, false);
        unit.reload(pcpos);
        build.CreateBr(unit.block_at(nextpc));
        return;
    }

    llvm::BasicBlock* isTable = unit.new_block("table");
    llvm::BasicBlock* slow = unit.new_block("slow");

    build.CreateCondBr(tag_is(unit, rb, LUA_TTABLE), isTable, slow, unit.likely());

    build.SetInsertPoint(isTable);
    auto [hit, value] = lookup_cached_slot(unit, unit.load_pointer(rb), pcpos, string_constant(unit, static_cast<int>(pc[1])));

    llvm::BasicBlock* fast = unit.new_block("fast");
    build.CreateCondBr(hit, fast, slow, unit.likely());

    build.SetInsertPoint(fast);
    unit.store_reg(ra, build.CreateLoad(types.tvalueTy, value));
    build.CreateBr(unit.block_at(nextpc));

    build.SetInsertPoint(slow);
    unit.spill(pcpos);
    unit.call_helper(helper, pcpos, false);
    unit.reload(pcpos);
    build.CreateBr(unit.block_at(nextpc));
}

// A method of a string is in the cached slot of the __index table of the string metatable
static void lower_namecall_string(TranslationUnit& unit, const Instruction* pc, int pcpos)
{
    llvm::IRBuilder<>& build = unit.builder();
    const VmTypes& types = unit.types();

    const int ra = LUAU_INSN_A(*pc);
    const int rb = LUAU_INSN_B(*pc);
    const int nextpc = next_pc(pc, pcpos);

    llvm::BasicBlock* isString = unit.new_block("namecall.string");
    llvm::BasicBlock* indexTable = unit.new_block("namecall.indextable");
    llvm::BasicBlock* found = unit.new_block("namecall.found");
    llvm::BasicBlock* slow = unit.new_block("namecall.slow");

    build.CreateCondBr(tag_is(unit, rb, LUA_TSTRING), isString, slow, unit.likely());

    build.SetInsertPoint(isString);
    llvm::Value* global = unit.load_field(types.ptrTy, unit.state(), offsetof(lua_State, global), "global");
    llvm::Value* metatable = unit.load_field(types.ptrTy, global, offsetof(global_State, mt) + LUA_TSTRING * sizeof(LuaTable*), "mt");
    llvm::Value* index = unit.call_native(reinterpret_cast<const void*>(&get_namecall_index), types.ptrTy, {unit.state(), metatable});
    build.CreateCondBr(build.CreateIsNotNull(index), indexTable, slow, unit.likely());

    build.SetInsertPoint(indexTable);
    auto [hit, value] = lookup_cached_slot(unit, index, pcpos, string_constant(unit, static_cast<int>(pc[1])));
    build.CreateCondBr(hit, found, slow, unit.likely());

    // rb can alias ra or ra + 1
    build.SetInsertPoint(found);
    llvm::Value* method = build.CreateLoad(types.tvalueTy, value);
    llvm::Value* self = unit.load_reg(rb);
    unit.store_reg(ra + 1, self);
    unit.store_reg(ra, method);
    build.CreateBr(unit.block_at(nextpc));

    build.SetInsertPoint(slow);
    unit.spill(pcpos);
    unit.call_helper(reinterpret_cast<const void*>(&execute_namecall), pcpos, false);
    unit.reload(pcpos);
    build.CreateBr(unit.block_at(nextpc));
}

// The table part of LOP_NAMECALL: the method is in the table (usually in the main position of its key), or the key is
// absent and the method is in the cached slot of the __index table of the metatable
void lower_inst_namecall(TranslationUnit& unit, const Instruction* pc, int pcpos)
{
    llvm::IRBuilder<>& build = unit.builder();
    const VmTypes& types = unit.types();

    // the receivers of a site are almost always of one type, so only the one the types say gets a fast path
    if (unit.bytecode().bcTypes[pcpos].a == LBC_TYPE_STRING)
        return lower_namecall_string(unit, pc, pcpos);

    const int ra = LUAU_INSN_A(*pc);
    const int rb = LUAU_INSN_B(*pc);
    const int nextpc = next_pc(pc, pcpos);
    const TString* key = string_constant(unit, static_cast<int>(pc[1]));

    llvm::BasicBlock* isTable = unit.new_block("namecall.table");
    llvm::BasicBlock* inIndex = unit.new_block("namecall.index");
    llvm::BasicBlock* hasIndex = unit.new_block("namecall.hasindex");
    llvm::BasicBlock* indexTable = unit.new_block("namecall.indextable");
    llvm::BasicBlock* found = unit.new_block("namecall.found");
    llvm::BasicBlock* slow = unit.new_block("namecall.slow");

    build.CreateCondBr(tag_is(unit, rb, LUA_TTABLE), isTable, slow, unit.likely());

    build.SetInsertPoint(isTable);
    llvm::Value* table = unit.load_pointer(rb);
    llvm::Value* lsizenode = build.CreateZExt(unit.load_field(types.i8Ty, table, offsetof(LuaTable, lsizenode)), types.i32Ty);
    llvm::Value* mask = build.CreateSub(build.CreateShl(build.getInt32(1), lsizenode), build.getInt32(1));
    llvm::Value* slot = build.CreateZExt(build.CreateAnd(build.getInt32(key->hash), mask), types.i64Ty);
    llvm::Value* nodes = unit.load_field(types.ptrTy, table, offsetof(LuaTable, node), "node");
    llvm::Value* node = build.CreateInBoundsGEP(types.i8Ty, nodes, build.CreateMul(slot, build.getInt64(sizeof(LuaNode))), "n");
    llvm::BasicBlock* own = unit.current_block();
    llvm::BasicBlock* chained = unit.new_block("namecall.chained");
    llvm::BasicBlock* absent = unit.new_block("namecall.absent");
    build.CreateCondBr(node_holds(unit, node, key), found, inIndex);

    // a key that isn't in its main position can only be further down the chain from there; small objects with a
    // metatable often have one, since their few fields share the main positions with the method names
    build.SetInsertPoint(inIndex);
    llvm::Value* noNext = build.CreateICmpULT(unit.load_field(types.i32Ty, node, kNodeKeyTagOffset), build.getInt32(0x10));
    build.CreateCondBr(noNext, absent, chained);

    build.SetInsertPoint(chained);
    llvm::Value* chainValue = unit.call_native(reinterpret_cast<const void*>(&luaH_getstr), types.ptrTy, {table, unit.const_ptr(key)});
    llvm::Value* chainNil = build.CreateICmpEQ(unit.load_field(types.i32Ty, chainValue, offsetof(TValue, tt)), build.getInt32(LUA_TNIL));
    llvm::BasicBlock* chainHit = unit.current_block();
    build.CreateCondBr(chainNil, absent, found);

    build.SetInsertPoint(absent);
    llvm::Value* metatable = unit.load_field(types.ptrTy, table, offsetof(LuaTable, metatable), "mt");
    build.CreateCondBr(build.CreateIsNotNull(metatable), hasIndex, slow);

    build.SetInsertPoint(hasIndex);
    llvm::Value* index = unit.call_native(reinterpret_cast<const void*>(&get_fast_tm), types.ptrTy, {unit.state(), metatable, build.getInt32(TM_INDEX)});
    llvm::Value* indexIsTable = build.CreateAnd(
        build.CreateIsNotNull(index),
        build.CreateICmpEQ(unit.load_field(types.i32Ty, index, offsetof(TValue, tt)), build.getInt32(LUA_TTABLE))
    );
    build.CreateCondBr(indexIsTable, indexTable, slow);

    build.SetInsertPoint(indexTable);
    auto [hit, indexValue] = lookup_cached_slot(unit, unit.load_field(types.ptrTy, index, offsetof(TValue, value)), pcpos, key);
    llvm::BasicBlock* inherited = unit.current_block();
    build.CreateCondBr(hit, found, slow);

    // rb can alias ra or ra + 1
    build.SetInsertPoint(found);
    llvm::PHINode* value = build.CreatePHI(types.ptrTy, 3, "method");
    value->addIncoming(types.field_ptr(build, node, offsetof(LuaNode, val)), own);
    value->addIncoming(chainValue, chainHit);
    value->addIncoming(indexValue, inherited);
    llvm::Value* method = build.CreateLoad(types.tvalueTy, value);
    llvm::Value* self = unit.load_reg(rb);
    unit.store_reg(ra + 1, self);
    unit.store_reg(ra, method);
    build.CreateBr(unit.block_at(nextpc));

    build.SetInsertPoint(slow);
    unit.spill(pcpos);
    unit.call_helper(reinterpret_cast<const void*>(&execute_namecall), pcpos, false);
    unit.reload(pcpos);
    build.CreateBr(unit.block_at(nextpc));
}

void lower_inst_settableks(TranslationUnit& unit, const Instruction* pc, int pcpos)
{
    llvm::IRBuilder<>& build = unit.builder();
    const VmTypes& types = unit.types();

    const int ra = LUAU_INSN_A(*pc);
    const int rb = LUAU_INSN_B(*pc);
    const int nextpc = next_pc(pc, pcpos);

    const void* helper = reinterpret_cast<const void*>(&execute_settableks);

    llvm::BasicBlock* isTable = unit.new_block("table");
    llvm::BasicBlock* slow = unit.new_block("slow");

    build.CreateCondBr(tag_is(unit, rb, LUA_TTABLE), isTable, slow, unit.likely());

    build.SetInsertPoint(isTable);
    llvm::Value* table = unit.load_pointer(rb);
    llvm::Value* node = cached_slot_node(unit, table, pcpos);
    llvm::Value* readonly = unit.load_field(types.i8Ty, table, offsetof(LuaTable, readonly));
    llvm::Value* keyHit = build.CreateAnd(
        node_key_is(unit, node, string_constant(unit, static_cast<int>(pc[1]))), build.CreateICmpEQ(readonly, build.getInt8(0))
    );
    llvm::Value* wasNil = node_value_is_nil(unit, node);
    llvm::Value* valueAddr = types.field_ptr(build, node, offsetof(LuaNode, val));

    llvm::BasicBlock* fast = unit.new_block("fast");
    // even a cold extra path costs registers across a loop, so only the tables a function is filling in get one
    const bool fills = unit.holds_new_table(rb, pcpos);
    llvm::BasicBlock* checkFill = fills ? unit.new_block("fill.check") : slow;

    build.CreateCondBr(build.CreateAnd(keyHit, build.CreateNot(wasNil)), fast, checkFill, unit.likely());

    build.SetInsertPoint(fast);
    build.CreateStore(unit.load_reg(ra), valueAddr);
    unit.emit_table_barrier(table, ra);
    build.CreateBr(unit.block_at(nextpc));

    if (fills)
    {
        llvm::BasicBlock* fill = unit.new_block("fill");

        // a key with a nil value is absent as far as __newindex goes, which only a table without a metatable can ignore;
        // this is what fills in the keys of a table from DUPTABLE
        build.SetInsertPoint(checkFill);
        llvm::Value* metatable = unit.load_field(types.ptrTy, table, offsetof(LuaTable, metatable));
        build.CreateCondBr(build.CreateAnd(keyHit, build.CreateIsNull(metatable)), fill, slow, unit.likely());

        // like luaH_setstr: the table may be a metatable, and the key may have been one of its absent metamethods
        build.SetInsertPoint(fill);
        build.CreateStore(unit.load_reg(ra), valueAddr);
        unit.store_field(build.getInt8(0), table, offsetof(LuaTable, tmcache));
        unit.emit_table_barrier(table, ra);
        unit.mark_slow_path();
        build.CreateBr(unit.block_at(nextpc));
    }

    build.SetInsertPoint(slow);
    unit.spill(pcpos);
    unit.call_helper(helper, pcpos, false);
    unit.reload(pcpos);
    build.CreateBr(unit.block_at(nextpc));
}

// NEWTABLE and DUPTABLE call the allocator directly, and only the GC step that may follow needs the stack in sync
void lower_inst_newtable(TranslationUnit& unit, const Instruction* pc, int pcpos)
{
    llvm::IRBuilder<>& build = unit.builder();
    const VmTypes& types = unit.types();

    const auto op = static_cast<LuauOpcode>(LUAU_INSN_OP(*pc));
    const int ra = LUAU_INSN_A(*pc);

    // allocation fails with a memory error, which reports the line of savedpc like the interpreter's VM_PROTECT_PC
    types.store_ci_savedpc(build, types.load_state_ci(build, unit.state()), unit.insn_ptr(next_pc(pc, pcpos)));

    llvm::Value* table = nullptr;

    if (op == LOP_NEWTABLE)
    {
        const int b = LUAU_INSN_B(*pc);
        llvm::Value* narray = build.getInt32(static_cast<int>(pc[1]));
        llvm::Value* nhash = build.getInt32(b == 0 ? 0 : 1 << (b - 1));

        table = unit.call_native(reinterpret_cast<const void*>(&luaH_new), types.ptrTy, {unit.state(), narray, nhash});
    }
    else
    {
        const TValue& tmpl = unit.bytecode().proto->k[LUAU_INSN_D(*pc)];
        LUAU_ASSERT(ttistable(&tmpl));

        table = unit.call_native(reinterpret_cast<const void*>(&luaH_clone), types.ptrTy, {unit.state(), unit.const_ptr(hvalue(&tmpl))});
    }

    unit.store_object(ra, table, LUA_TTABLE);
    unit.emit_gc_check(pcpos, ra);
}

// Array part access for GETTABLE/SETTABLE/GETTABLEN/SETTABLEN; 'index' is zero-based
template<typename F>
static void lower_array_access(
    TranslationUnit& unit,
    llvm::Value* isCandidate,
    const int rb,
    llvm::Value* index,
    bool store,
    const void* helper,
    const int pcpos,
    const int nextpc,
    F&& access
)
{
    llvm::IRBuilder<>& build = unit.builder();
    const VmTypes& types = unit.types();

    llvm::BasicBlock* candidate = unit.new_block("array.check");
    llvm::BasicBlock* slow = unit.new_block("slow");

    build.CreateCondBr(isCandidate, candidate, slow, unit.likely());

    build.SetInsertPoint(candidate);
    llvm::Value* table = unit.load_pointer(rb);
    llvm::Value* sizearray = unit.load_field(types.i32Ty, table, offsetof(LuaTable, sizearray), "sizearray");

    llvm::Value* metatable = unit.load_field(types.ptrTy, table, offsetof(LuaTable, metatable), "metatable");
    llvm::Value* array = unit.load_field(types.ptrTy, table, offsetof(LuaTable, array), "array");
    llvm::Value* element = build.CreateInBoundsGEP(types.tvalueTy, array, build.CreateZExt(index, types.i64Ty), "element");

    llvm::Value* inArray = build.CreateICmpULT(index, sizearray);

    if (store)
        inArray = build.CreateAnd(inArray, build.CreateICmpEQ(unit.load_field(types.i8Ty, table, offsetof(LuaTable, readonly)), build.getInt8(0)));

    // __index and __newindex only apply to a nil value, so a table with a metatable can stay on the fast path too; that
    // costs a second way into it, which slows down every loop over plain arrays, so only sites that saw one get it
    const bool withMetatable = (unit.feedback_at(pcpos) & kFeedbackArrayMetatable) != 0;

    llvm::BasicBlock* miss = unit.new_block("array.miss");
    llvm::BasicBlock* fast = unit.new_block("array");
    build.CreateCondBr(build.CreateAnd(inArray, build.CreateIsNull(metatable)), fast, miss, unit.likely());

    build.SetInsertPoint(miss);

    if (withMetatable)
    {
        llvm::BasicBlock* inBounds = unit.new_block("array.meta");
        build.CreateCondBr(inArray, inBounds, slow);

        build.SetInsertPoint(inBounds);
        llvm::Value* notNil = build.CreateICmpNE(unit.load_field(types.i32Ty, element, offsetof(TValue, tt)), build.getInt32(LUA_TNIL));
        build.CreateCondBr(notNil, fast, slow);
    }
    else
    {
        unit.emit_feedback_bits(
            pcpos, build.CreateSelect(build.CreateIsNotNull(metatable), build.getInt32(kFeedbackArrayMetatable), build.getInt32(0))
        );
        build.CreateBr(slow);
    }

    build.SetInsertPoint(fast);
    access(table, element);
    build.CreateBr(unit.block_at(nextpc));

    build.SetInsertPoint(slow);
    unit.spill(pcpos);
    unit.call_helper(helper, pcpos, false);
    unit.reload(pcpos);
    build.CreateBr(unit.block_at(nextpc));
}

// Zero-based array index for a number key, or an index that's always out of bounds when the key isn't an integer
static llvm::Value* array_index(TranslationUnit& unit, const int reg)
{
    llvm::IRBuilder<>& build = unit.builder();
    const VmTypes& types = unit.types();

    llvm::Value* key = unit.load_number(reg);

    // whatever an out of range or NaN key converts to, it doesn't convert back to the key
    llvm::Value* index = build.CreateFreeze(build.CreateFPToSI(key, types.i32Ty));
    llvm::Value* exact = build.CreateFCmpOEQ(build.CreateSIToFP(index, types.doubleTy), key);

    return build.CreateSelect(exact, build.CreateSub(index, build.getInt32(1)), build.getInt32(-1), "index");
}

void lower_inst_gettable(TranslationUnit& unit, const Instruction* pc, int pcpos)
{
    llvm::IRBuilder<>& build = unit.builder();
    const VmTypes& types = unit.types();

    const int ra = LUAU_INSN_A(*pc);
    const int rb = LUAU_INSN_B(*pc);
    const int rc = LUAU_INSN_C(*pc);

    llvm::BasicBlock* checkKey = unit.new_block("key");
    llvm::BasicBlock* candidate = unit.new_block("candidate");
    llvm::BasicBlock* slow = unit.new_block("slow");

    // the key is only read as a number once its tag says it is one
    build.CreateCondBr(tag_is(unit, rb, LUA_TTABLE), checkKey, slow, unit.likely());
    build.SetInsertPoint(checkKey);
    build.CreateCondBr(tag_is(unit, rc, LUA_TNUMBER), candidate, slow, unit.likely());

    build.SetInsertPoint(slow);
    const int nextpc = next_pc(pc, pcpos);
    unit.spill(pcpos);
    unit.call_helper(reinterpret_cast<const void*>(&execute_gettable), pcpos, false);
    unit.reload(pcpos);
    build.CreateBr(unit.block_at(nextpc));

    build.SetInsertPoint(candidate);
    lower_array_access(
        unit,
        build.getTrue(),
        rb,
        array_index(unit, rc),
        false,
        reinterpret_cast<const void*>(&execute_gettable),
        pcpos,
        nextpc,
        [&](llvm::Value* table, llvm::Value* element)
        {
            unit.store_reg(ra, build.CreateLoad(types.tvalueTy, element));
        }
    );
}

void lower_inst_settable(TranslationUnit& unit, const Instruction* pc, int pcpos)
{
    llvm::IRBuilder<>& build = unit.builder();

    const int ra = LUAU_INSN_A(*pc);
    const int rb = LUAU_INSN_B(*pc);
    const int rc = LUAU_INSN_C(*pc);

    llvm::BasicBlock* checkKey = unit.new_block("key");
    llvm::BasicBlock* candidate = unit.new_block("candidate");
    llvm::BasicBlock* slow = unit.new_block("slow");

    build.CreateCondBr(tag_is(unit, rb, LUA_TTABLE), checkKey, slow, unit.likely());
    build.SetInsertPoint(checkKey);
    build.CreateCondBr(tag_is(unit, rc, LUA_TNUMBER), candidate, slow, unit.likely());

    build.SetInsertPoint(slow);
    const int nextpc = next_pc(pc, pcpos);
    unit.spill(pcpos);
    unit.call_helper(reinterpret_cast<const void*>(&execute_settable), pcpos, false);
    unit.reload(pcpos);
    build.CreateBr(unit.block_at(nextpc));

    build.SetInsertPoint(candidate);
    lower_array_access(
        unit,
        build.getTrue(),
        rb,
        array_index(unit, rc),
        true,
        reinterpret_cast<const void*>(&execute_settable),
        pcpos,
        nextpc,
        [&](llvm::Value* table, llvm::Value* element)
        {
            build.CreateStore(unit.load_reg(ra), element);
            unit.emit_table_barrier(table, ra);
        }
    );
}

void lower_inst_gettablen(TranslationUnit& unit, const Instruction* pc, int pcpos)
{
    llvm::IRBuilder<>& build = unit.builder();
    const VmTypes& types = unit.types();

    const int ra = LUAU_INSN_A(*pc);
    const int rb = LUAU_INSN_B(*pc);

    lower_array_access(
        unit,
        tag_is(unit, rb, LUA_TTABLE),
        rb,
        build.getInt32(LUAU_INSN_C(*pc)),
        false,
        reinterpret_cast<const void*>(&execute_gettablen),
        pcpos,
        next_pc(pc, pcpos),
        [&](llvm::Value* table, llvm::Value* element)
        {
            unit.store_reg(ra, build.CreateLoad(types.tvalueTy, element));
        }
    );
}

void lower_inst_settablen(TranslationUnit& unit, const Instruction* pc, int pcpos)
{
    llvm::IRBuilder<>& build = unit.builder();

    const int ra = LUAU_INSN_A(*pc);
    const int rb = LUAU_INSN_B(*pc);

    lower_array_access(
        unit,
        tag_is(unit, rb, LUA_TTABLE),
        rb,
        build.getInt32(LUAU_INSN_C(*pc)),
        true,
        reinterpret_cast<const void*>(&execute_settablen),
        pcpos,
        next_pc(pc, pcpos),
        [&](llvm::Value* table, llvm::Value* element)
        {
            build.CreateStore(unit.load_reg(ra), element);
            unit.emit_table_barrier(table, ra);
        }
    );
}

// The result copy of execute_return and luau_poscall: 'count' values at 'vals' go to 'res', truncated or padded with
// nil to 'nresults' unless that's LUA_MULTRET; returns the end of the results
static llvm::Value* emit_move_results(TranslationUnit& unit, llvm::Value* res, llvm::Value* vals, llvm::Value* count, llvm::Value* nresults)
{
    llvm::IRBuilder<>& build = unit.builder();
    const VmTypes& types = unit.types();

    llvm::Value* multret = build.CreateICmpEQ(nresults, build.getInt64(LUA_MULTRET));
    llvm::Value* wanted = build.CreateSelect(multret, count, nresults, "wanted");
    llvm::Value* copied = build.CreateSelect(build.CreateICmpSLT(count, wanted), count, wanted, "copied");

    // res is below vals, so copying forwards never overwrites a value that is still to be copied
    llvm::BasicBlock* entry = unit.current_block();
    llvm::BasicBlock* copy = unit.new_block("results.copy");
    llvm::BasicBlock* pad = unit.new_block("results.pad");
    llvm::BasicBlock* nil = unit.new_block("results.nil");
    llvm::BasicBlock* done = unit.new_block("results.done");

    build.CreateCondBr(build.CreateICmpSGT(copied, build.getInt64(0)), copy, pad);

    build.SetInsertPoint(copy);
    llvm::PHINode* i = build.CreatePHI(types.i64Ty, 2, "i");
    i->addIncoming(build.getInt64(0), entry);
    types.copy_tvalue(build, build.CreateGEP(types.tvalueTy, res, i), build.CreateGEP(types.tvalueTy, vals, i));
    llvm::Value* nextI = build.CreateAdd(i, build.getInt64(1));
    i->addIncoming(nextI, copy);
    build.CreateCondBr(build.CreateICmpSLT(nextI, copied), copy, pad);

    build.SetInsertPoint(pad);
    build.CreateCondBr(build.CreateICmpSLT(copied, wanted), nil, done, unit.unlikely());

    build.SetInsertPoint(nil);
    llvm::PHINode* j = build.CreatePHI(types.i64Ty, 2, "j");
    j->addIncoming(copied, pad);
    llvm::Value* slot = build.CreateGEP(types.tvalueTy, res, j);
    build.CreateStore(build.getInt32(LUA_TNIL), build.CreateStructGEP(types.tvalueTy, slot, VmTypes::kTValueTt));
    llvm::Value* nextJ = build.CreateAdd(j, build.getInt64(1));
    j->addIncoming(nextJ, nil);
    build.CreateCondBr(build.CreateICmpSLT(nextJ, wanted), nil, done);

    build.SetInsertPoint(done);
    return build.CreateGEP(types.tvalueTy, res, wanted, "resend");
}

struct CallSetup
{
    llvm::Value* cl = nullptr;
    llvm::Value* func = nullptr;
    llvm::Value* argtop = nullptr;
    // of the callee, in TValues
    llvm::Value* stacksize = nullptr;
    llvm::Value* ci = nullptr;
};

// The C function part of LOP_CALL in lvmexecute.cpp, once the stack and the call info array have room
static void lower_c_call(TranslationUnit& unit, const Instruction* pc, const int pcpos, const CallSetup& setup, llvm::BasicBlock* done)
{
    llvm::IRBuilder<>& build = unit.builder();
    const VmTypes& types = unit.types();

    const int nresults = LUAU_INSN_C(*pc) - 1;
    llvm::Value* state = unit.state();

    unit.store_field(unit.insn_ptr(next_pc(pc, pcpos)), setup.ci, offsetof(CallInfo, savedpc));

    llvm::Value* base = build.CreateConstGEP1_64(types.i8Ty, setup.func, sizeof(TValue), "newbase");
    llvm::Value* top = build.CreateGEP(types.i8Ty, setup.argtop, build.CreateMul(setup.stacksize, build.getInt64(sizeof(TValue))), "newtop");

    llvm::Value* newci = build.CreateConstGEP1_64(types.i8Ty, setup.ci, sizeof(CallInfo), "newci");
    unit.store_field(newci, state, offsetof(lua_State, ci));
    unit.store_field(setup.func, newci, offsetof(CallInfo, func));
    unit.store_field(unit.load_field(types.ptrTy, setup.cl, offsetof(Closure, l.p), "p"), newci, offsetof(CallInfo, p));
    unit.store_field(base, newci, offsetof(CallInfo, base));
    unit.store_field(top, newci, offsetof(CallInfo, top));
    unit.store_field(llvm::ConstantPointerNull::get(types.ptrTy), newci, offsetof(CallInfo, savedpc));
    unit.store_field(build.getInt32(nresults), newci, offsetof(CallInfo, nresults));
    unit.store_field(build.getInt32(0), newci, offsetof(CallInfo, flags));
    unit.store_field(base, state, offsetof(lua_State, base));
    unit.store_field(setup.argtop, state, offsetof(lua_State, top));

    llvm::FunctionType* cfunctionTy = llvm::FunctionType::get(types.i32Ty, {types.ptrTy}, false);
    llvm::Value* f = unit.load_field(types.ptrTy, setup.cl, offsetof(Closure, c.f), "f");
    llvm::Value* n = build.CreateCall(cfunctionTy, f, {state}, "n");

    llvm::BasicBlock* returned = unit.new_block("ccall.returned");
    build.CreateCondBr(build.CreateICmpSLT(n, build.getInt32(0)), unit.yield_exit(), returned, unit.unlikely());

    // the function can have reallocated the stack, and it's the call info at L->ci that gets popped
    build.SetInsertPoint(returned);
    llvm::Value* ci = types.load_state_ci(build, state);
    llvm::Value* cip = build.CreateConstGEP1_64(types.i8Ty, ci, -int64_t(sizeof(CallInfo)), "cip");
    llvm::Value* res = unit.load_field(types.ptrTy, ci, offsetof(CallInfo, func), "res");
    llvm::Value* count = build.CreateSExt(n, types.i64Ty);
    llvm::Value* vals = build.CreateGEP(types.tvalueTy, unit.load_field(types.ptrTy, state, offsetof(lua_State, top), "top"), build.CreateNeg(count));
    llvm::Value* resEnd = emit_move_results(unit, res, vals, count, build.getInt64(nresults));

    unit.store_field(cip, state, offsetof(lua_State, ci));
    unit.store_field(unit.load_field(types.ptrTy, cip, offsetof(CallInfo, base), "base"), state, offsetof(lua_State, base));
    llvm::Value* callerTop = nresults == LUA_MULTRET ? resEnd : unit.load_field(types.ptrTy, cip, offsetof(CallInfo, top), "top");
    unit.store_field(callerTop, state, offsetof(lua_State, top));

    build.CreateBr(done);
}

// The LOP_CALL of lvmexecute.cpp and the Lua part of luau_precall, for a callee that has generated code which is
// entered directly, or a C function that is called directly. Everything is checked before the first store, so that
// 'generic' can still do the whole call
static void lower_native_call(TranslationUnit& unit, const Instruction* pc, const int pcpos, llvm::BasicBlock* generic, llvm::BasicBlock* done)
{
    llvm::IRBuilder<>& build = unit.builder();
    const VmTypes& types = unit.types();

    const int ra = LUAU_INSN_A(*pc);
    const int nparams = LUAU_INSN_B(*pc) - 1;
    const int nresults = LUAU_INSN_C(*pc) - 1;

    auto require = [&](llvm::Value* cond, const char* name)
    {
        llvm::BasicBlock* next = unit.new_block(name);
        build.CreateCondBr(cond, next, generic, unit.likely());
        build.SetInsertPoint(next);
    };

    if (LUAU_INSN_OP(*pc) == LOP_CALLFB)
    {
        // the slot is sealed by the first call when nothing collects call targets, and collecting is left to the helper
        llvm::Value* slot = build.CreateLoad(types.i32Ty, unit.insn_ptr(pcpos + 1), "fbslot");
        require(build.CreateICmpEQ(slot, build.getInt32(LUAU_INSN_FBSLOT_SEALED)), "call.sealed");
    }

    require(tag_is(unit, ra, LUA_TFUNCTION), "call.function");

    llvm::Value* cl = unit.load_pointer(ra);
    llvm::Value* state = unit.state();
    llvm::Value* func = unit.stack_slot(ra, "func");

    // a variadic argument list ends at L->top, where the instruction that produced its tail left it
    llvm::Value* argtop = nullptr;
    llvm::Value* argcount = nullptr;

    if (nparams == LUA_MULTRET)
    {
        argtop = unit.load_field(types.ptrTy, state, offsetof(lua_State, top), "argtop");
        llvm::Value* args = build.CreatePtrDiff(types.tvalueTy, argtop, unit.stack_slot(ra + 1));
        argcount = build.CreateTrunc(args, types.i32Ty, "nparams");
    }
    else
    {
        argtop = unit.stack_slot(ra + 1 + nparams, "argtop");
        argcount = build.getInt32(nparams);
    }
    llvm::Value* stacksize = build.CreateZExt(unit.load_field(types.i8Ty, cl, offsetof(Closure, stacksize), "stacksize"), types.i64Ty);

    // stacklimitreached with L->top at argtop, which leaves growing the stack to luaD_checkstackfornewci
    llvm::Value* stackLast = unit.load_field(types.ptrTy, state, offsetof(lua_State, stack_last), "stack_last");
    llvm::Value* room = build.CreatePtrDiff(types.i8Ty, stackLast, argtop);
    require(build.CreateICmpSGT(room, build.CreateMul(stacksize, build.getInt64(sizeof(TValue)))), "call.stack");

    // incr_ci, which leaves growing the array to luaD_growCI
    llvm::Value* ci = unit.load_field(types.ptrTy, state, offsetof(lua_State, ci), "ci");
    llvm::Value* endCi = unit.load_field(types.ptrTy, state, offsetof(lua_State, end_ci), "end_ci");
    require(build.CreateICmpNE(ci, endCi), "call.ci");

    llvm::BasicBlock* cCall = unit.new_block("call.c");
    llvm::BasicBlock* luaCall = unit.new_block("call.lua");
    build.CreateCondBr(build.CreateICmpNE(unit.load_field(types.i8Ty, cl, offsetof(Closure, isC), "isC"), build.getInt8(0)), cCall, luaCall);

    build.SetInsertPoint(cCall);
    lower_c_call(unit, pc, pcpos, {cl, func, argtop, stacksize, ci}, done);

    build.SetInsertPoint(luaCall);
    llvm::Value* proto = unit.load_field(types.ptrTy, cl, offsetof(Closure, l.p), "p");
    llvm::Value* target = unit.load_field(types.ptrTy, proto, offsetof(Proto, exectarget), "exectarget");
    llvm::Value* execdata = unit.load_field(types.ptrTy, proto, offsetof(Proto, execdata), "execdata");
    require(build.CreateAnd(build.CreateIsNotNull(target), build.CreateIsNotNull(execdata)), "call.native");

    // the call that reaches the threshold runs the hot callback, which is the helper's job
    llvm::Value* hits = build.CreateAdd(unit.load_field(types.i32Ty, proto, offsetof(Proto, hits), "hits"), build.getInt32(1));
    llvm::Value* threshold = build.CreateLoad(types.i32Ty, unit.const_ptr(&FInt::LuauNativeHitsThreshold.value), "threshold");
    require(build.CreateICmpNE(hits, threshold), "call.enter");

    unit.store_field(hits, proto, offsetof(Proto, hits));
    unit.store_field(unit.insn_ptr(next_pc(pc, pcpos)), ci, offsetof(CallInfo, savedpc));

    llvm::Value* base = build.CreateConstGEP1_64(types.i8Ty, func, sizeof(TValue), "newbase");
    llvm::Value* top = build.CreateGEP(types.i8Ty, argtop, build.CreateMul(stacksize, build.getInt64(sizeof(TValue))), "newtop");

    llvm::Value* newci = build.CreateConstGEP1_64(types.i8Ty, ci, sizeof(CallInfo), "newci");
    unit.store_field(newci, state, offsetof(lua_State, ci));
    unit.store_field(func, newci, offsetof(CallInfo, func));
    unit.store_field(proto, newci, offsetof(CallInfo, p));
    unit.store_field(base, newci, offsetof(CallInfo, base));
    unit.store_field(top, newci, offsetof(CallInfo, top));
    unit.store_field(unit.load_field(types.ptrTy, proto, offsetof(Proto, code), "code"), newci, offsetof(CallInfo, savedpc));
    unit.store_field(build.getInt32(nresults), newci, offsetof(CallInfo, nresults));
    unit.store_field(build.getInt32(LUA_CALLINFO_NATIVE), newci, offsetof(CallInfo, flags));
    unit.store_field(base, state, offsetof(lua_State, base));

    // fill unused parameters with nil
    llvm::Value* numparams = build.CreateZExt(unit.load_field(types.i8Ty, proto, offsetof(Proto, numparams), "numparams"), types.i32Ty);
    llvm::BasicBlock* fill = unit.new_block("call.fill");
    llvm::BasicBlock* filled = unit.new_block("call.filled");
    llvm::BasicBlock* entry = unit.current_block();

    build.CreateCondBr(build.CreateICmpSGT(numparams, argcount), fill, filled, unit.unlikely());

    build.SetInsertPoint(fill);
    llvm::PHINode* arg = build.CreatePHI(types.i32Ty, 2, "arg");
    arg->addIncoming(argcount, entry);
    llvm::Value* slot = build.CreateGEP(types.tvalueTy, base, build.CreateZExt(arg, types.i64Ty));
    build.CreateStore(build.getInt32(LUA_TNIL), build.CreateStructGEP(types.tvalueTy, slot, VmTypes::kTValueTt));
    llvm::Value* nextArg = build.CreateAdd(arg, build.getInt32(1));
    arg->addIncoming(nextArg, fill);
    build.CreateCondBr(build.CreateICmpSLT(nextArg, numparams), fill, filled);

    build.SetInsertPoint(filled);
    llvm::Value* isVararg = build.CreateICmpNE(unit.load_field(types.i8Ty, proto, offsetof(Proto, is_vararg), "is_vararg"), build.getInt8(0));
    llvm::Value* filledTop = build.CreateGEP(
        types.tvalueTy,
        base,
        build.CreateZExt(build.CreateSelect(build.CreateICmpSGT(numparams, argcount), numparams, argcount), types.i64Ty)
    );
    unit.store_field(build.CreateSelect(isVararg, filledTop, top), state, offsetof(lua_State, top));

    unit.emit_enter_native(target);
}

void lower_inst_call(TranslationUnit& unit, const Instruction* pc, int pcpos)
{
    llvm::IRBuilder<>& build = unit.builder();

    const int nextpc = next_pc(pc, pcpos);

    unit.emit_safepoint(pcpos, false);

    unit.spill(pcpos);

    llvm::BasicBlock* generic = unit.new_block("call.generic");
    llvm::BasicBlock* done = unit.new_block("call.done");
    lower_native_call(unit, pc, pcpos, generic, done);

    build.SetInsertPoint(generic);
    llvm::Value* result = unit.call_helper(reinterpret_cast<const void*>(&execute_call), pcpos, true);

    llvm::BasicBlock* enter = unit.new_block("call.enter");

    llvm::SwitchInst* dispatch = build.CreateSwitch(result, done, 2);
    dispatch->addCase(build.getInt32(kCallEnter), enter);
    dispatch->addCase(build.getInt32(kCallYield), unit.yield_exit());

    build.SetInsertPoint(enter);
    unit.emit_resume_ci();

    build.SetInsertPoint(done);
    unit.reload(pcpos);
    build.CreateBr(unit.block_at(nextpc));

    // a Lua callee returns, and a yielded C function resumes, into the instruction that follows the call
    unit.mark_entry_point(nextpc);
}

static void emit_return_to_caller(TranslationUnit& unit, llvm::Value* ci)
{
    llvm::IRBuilder<>& build = unit.builder();
    const VmTypes& types = unit.types();

    llvm::Value* flags = unit.load_field(types.i32Ty, ci, offsetof(CallInfo, flags), "flags");
    llvm::Value* exits = build.CreateICmpNE(build.CreateAnd(flags, build.getInt32(LUA_CALLINFO_RETURN)), build.getInt32(0));

    llvm::BasicBlock* caller = unit.new_block("return.caller");
    build.CreateCondBr(exits, unit.yield_exit(), caller);

    build.SetInsertPoint(caller);
    unit.emit_resume_ci();
}

// execute_return for a variadic sequence, which is already on the stack and ends at L->top
static void lower_variadic_return(TranslationUnit& unit, const Instruction* pc, const int pcpos)
{
    llvm::IRBuilder<>& build = unit.builder();
    const VmTypes& types = unit.types();

    const int ra = LUAU_INSN_A(*pc);

    unit.spill(pcpos);

    llvm::Value* state = unit.state();
    llvm::Value* ci = types.load_state_ci(build, state);
    llvm::Value* nresults = build.CreateSExt(unit.load_field(types.i32Ty, ci, offsetof(CallInfo, nresults), "nresults"), types.i64Ty);
    llvm::Value* multret = build.CreateICmpEQ(nresults, build.getInt64(LUA_MULTRET));

    llvm::Value* res = unit.load_field(types.ptrTy, ci, offsetof(CallInfo, func), "res");
    llvm::Value* vals = unit.stack_slot(ra, "vals");
    llvm::Value* top = unit.load_field(types.ptrTy, state, offsetof(lua_State, top), "top");
    llvm::Value* count = build.CreatePtrDiff(types.tvalueTy, top, vals, "count");
    llvm::Value* resEnd = emit_move_results(unit, res, vals, count, nresults);

    llvm::Value* cip = build.CreateConstGEP1_64(types.i8Ty, ci, -int64_t(sizeof(CallInfo)), "cip");
    unit.store_field(cip, state, offsetof(lua_State, ci));
    unit.store_field(unit.load_field(types.ptrTy, cip, offsetof(CallInfo, base), "base"), state, offsetof(lua_State, base));

    unit.store_field(build.CreateSelect(multret, resEnd, unit.load_field(types.ptrTy, cip, offsetof(CallInfo, top), "top")), state, offsetof(lua_State, top));

    emit_return_to_caller(unit, ci);
}

// The LOP_RETURN of lvmexecute.cpp when the caller wants at most the values that are returned, or all of them. The
// values go straight from the registers to the caller's frame: nothing else in this frame is observable once it's
// popped, because the collector never marks the stack above top and the compiler closes upvalues before returning
static void lower_native_return(TranslationUnit& unit, const Instruction* pc, const int pcpos, llvm::BasicBlock* generic)
{
    llvm::IRBuilder<>& build = unit.builder();
    const VmTypes& types = unit.types();

    const int ra = LUAU_INSN_A(*pc);
    const int b = LUAU_INSN_B(*pc) - 1;

    LUAU_ASSERT(b != LUA_MULTRET);

    llvm::Value* state = unit.state();
    llvm::Value* ci = types.load_state_ci(build, state);
    llvm::Value* nresults = unit.load_field(types.i32Ty, ci, offsetof(CallInfo, nresults), "nresults");
    llvm::Value* multret = build.CreateICmpEQ(nresults, build.getInt32(LUA_MULTRET));

    // values the caller doesn't want land in the slots the callee frame held, which nothing reads anymore
    llvm::BasicBlock* fast = unit.new_block("return.fast");
    build.CreateCondBr(build.CreateICmpSLE(nresults, build.getInt32(b)), fast, generic, unit.likely());

    build.SetInsertPoint(fast);
    llvm::Value* res = unit.load_field(types.ptrTy, ci, offsetof(CallInfo, func), "res");

    for (int i = 0; i < b; i++)
        types.copy_tvalue(build, build.CreateConstGEP1_64(types.tvalueTy, res, i), unit.reg_slot(ra + i));

    llvm::Value* cip = build.CreateConstGEP1_64(types.i8Ty, ci, -int64_t(sizeof(CallInfo)), "cip");
    unit.store_field(cip, state, offsetof(lua_State, ci));
    unit.store_field(unit.load_field(types.ptrTy, cip, offsetof(CallInfo, base), "base"), state, offsetof(lua_State, base));

    llvm::Value* resEnd = build.CreateConstGEP1_64(types.tvalueTy, res, b);
    llvm::Value* top = build.CreateSelect(multret, resEnd, unit.load_field(types.ptrTy, cip, offsetof(CallInfo, top), "top"));
    unit.store_field(top, state, offsetof(lua_State, top));

    emit_return_to_caller(unit, ci);
}

void lower_inst_return(TranslationUnit& unit, const Instruction* pc, int pcpos)
{
    llvm::IRBuilder<>& build = unit.builder();

    unit.emit_safepoint(pcpos, false);

    if (LUAU_INSN_B(*pc) - 1 == LUA_MULTRET)
    {
        lower_variadic_return(unit, pc, pcpos);
        return;
    }

    llvm::BasicBlock* generic = unit.new_block("return.generic");
    lower_native_return(unit, pc, pcpos, generic);

    build.SetInsertPoint(generic);
    unit.spill(pcpos);
    llvm::Value* resume = unit.call_helper(reinterpret_cast<const void*>(&execute_return), pcpos, true);

    llvm::BasicBlock* caller = unit.new_block("return.caller");
    build.CreateCondBr(build.CreateICmpEQ(resume, build.getInt32(0)), unit.yield_exit(), caller);

    build.SetInsertPoint(caller);
    unit.emit_resume_ci();
}

void lower_inst_jump(TranslationUnit& unit, const Instruction* pc, int pcpos)
{
    const auto op = static_cast<LuauOpcode>(LUAU_INSN_OP(*pc));

    if (op == LOP_JUMPBACK || op == LOP_JUMPX)
        unit.emit_safepoint(pcpos, true);

    unit.builder().CreateBr(unit.block_at(getJumpTarget(*pc, pcpos)));
}

void lower_inst_jumpif(TranslationUnit& unit, const Instruction* pc, int pcpos)
{
    llvm::Value* falsy = unit.is_falsy(LUAU_INSN_A(*pc));

    llvm::BasicBlock* target = unit.block_at(getJumpTarget(*pc, pcpos));
    llvm::BasicBlock* next = unit.block_at(next_pc(pc, pcpos));

    if (LUAU_INSN_OP(*pc) == LOP_JUMPIF)
        unit.builder().CreateCondBr(falsy, next, target);
    else
        unit.builder().CreateCondBr(falsy, target, next);
}

// Branches on 'cond', which is the relation the instruction tests before its NOT variant negates it
static void branch_on_relation(TranslationUnit& unit, llvm::Value* cond, const bool negated, const int target, const int next)
{
    if (negated)
        unit.builder().CreateCondBr(cond, unit.block_at(next), unit.block_at(target));
    else
        unit.builder().CreateCondBr(cond, unit.block_at(target), unit.block_at(next));
}

void lower_inst_jumpifeq(TranslationUnit& unit, const Instruction* pc, int pcpos)
{
    llvm::IRBuilder<>& build = unit.builder();
    const VmTypes& types = unit.types();

    const int ra = LUAU_INSN_A(*pc);
    const int rb = static_cast<int>(pc[1]);

    const bool negated = LUAU_INSN_OP(*pc) == LOP_JUMPIFNOTEQ;
    const int target = getJumpTarget(*pc, pcpos);
    const int next = next_pc(pc, pcpos);

    llvm::BasicBlock* number = unit.new_block("eq.number");
    llvm::BasicBlock* slow = unit.new_block("eq.slow");

    const BytecodeTypes& bcTypes = unit.bytecode().bcTypes[pcpos];

    if (unit.speculates_on(bcTypes.a, LBC_TYPE_NUMBER) && unit.speculates_on(bcTypes.b, LBC_TYPE_NUMBER))
    {
        check_operand_tags(unit, pcpos, {{ra, bcTypes.a}, {rb, bcTypes.b}}, number, slow);

        build.SetInsertPoint(number);
        branch_on_relation(unit, build.CreateFCmpOEQ(unit.load_number(ra), unit.load_number(rb)), negated, target, next);

        // every operand check exits, so this is unreachable
        build.SetInsertPoint(slow);
        build.CreateUnreachable();
        return;
    }

    llvm::Value* tagA = unit.load_tag(ra);
    llvm::Value* tagB = unit.load_tag(rb);

    llvm::BasicBlock* sameType = unit.new_block("eq.same");
    llvm::BasicBlock* different = unit.new_block("eq.different");

    build.CreateCondBr(build.CreateICmpEQ(tagA, tagB), sameType, different);

    // values of different types are never equal, and no metamethod gets involved
    build.SetInsertPoint(different);
    branch_on_relation(unit, build.getFalse(), negated, target, next);

    llvm::BasicBlock* other = unit.new_block("eq.other");

    build.SetInsertPoint(sameType);
    build.CreateCondBr(build.CreateICmpEQ(tagA, build.getInt32(LUA_TNUMBER)), number, other, unit.likely());

    build.SetInsertPoint(number);
    branch_on_relation(unit, build.CreateFCmpOEQ(unit.load_number(ra), unit.load_number(rb)), negated, target, next);

    // luaV_equalval, where tables, userdata and objects can have an __eq that runs even for the same one
    llvm::BasicBlock* nil = unit.new_block("eq.nil");
    llvm::BasicBlock* boolean = unit.new_block("eq.boolean");
    llvm::BasicBlock* light = unit.new_block("eq.lightuserdata");
    llvm::BasicBlock* identity = unit.new_block("eq.identity");
    llvm::BasicBlock* table = unit.new_block("eq.table");

    build.SetInsertPoint(other);
    llvm::SwitchInst* dispatch = build.CreateSwitch(tagA, slow, 12);
    dispatch->addCase(build.getInt32(LUA_TNIL), nil);
    dispatch->addCase(build.getInt32(LUA_TBOOLEAN), boolean);
    dispatch->addCase(build.getInt32(LUA_TLIGHTUSERDATA), light);

    for (const int tag : {LUA_TINTEGER, LUA_TSTRING, LUA_TFUNCTION, LUA_TTHREAD, LUA_TBUFFER, LUA_TCLASS})
        dispatch->addCase(build.getInt32(tag), identity);

    dispatch->addCase(build.getInt32(LUA_TTABLE), table);

#if LUA_VECTOR_DOUBLE == 0
    llvm::BasicBlock* vector = unit.new_block("eq.vector");
    dispatch->addCase(build.getInt32(LUA_TVECTOR), vector);

    build.SetInsertPoint(vector);
    branch_on_relation(unit, build.CreateAndReduce(build.CreateFCmpOEQ(unit.load_vector(ra), unit.load_vector(rb))), negated, target, next);
#endif

    build.SetInsertPoint(nil);
    branch_on_relation(unit, build.getTrue(), negated, target, next);

    // only the low half of the value holds the boolean
    build.SetInsertPoint(boolean);
    branch_on_relation(unit, build.CreateICmpEQ(unit.load_boolean(ra), unit.load_boolean(rb)), negated, target, next);

    build.SetInsertPoint(light);
    llvm::Value* sameTag = build.CreateICmpEQ(
        unit.load_field(types.i32Ty, unit.reg_slot(ra), offsetof(TValue, extra)), unit.load_field(types.i32Ty, unit.reg_slot(rb), offsetof(TValue, extra))
    );
    branch_on_relation(unit, build.CreateAnd(build.CreateICmpEQ(unit.load_pointer(ra), unit.load_pointer(rb)), sameTag), negated, target, next);

    build.SetInsertPoint(identity);
    llvm::Value* bitsA = unit.load_field(types.i64Ty, unit.reg_slot(ra), offsetof(TValue, value));
    llvm::Value* bitsB = unit.load_field(types.i64Ty, unit.reg_slot(rb), offsetof(TValue, value));
    branch_on_relation(unit, build.CreateICmpEQ(bitsA, bitsB), negated, target, next);

    // get_compTM finds no __eq when the first table has no metatable
    llvm::BasicBlock* noMetatable = unit.new_block("eq.table.nomt");
    build.SetInsertPoint(table);
    llvm::Value* metatable = unit.load_field(types.ptrTy, unit.load_pointer(ra), offsetof(LuaTable, metatable));
    build.CreateCondBr(build.CreateIsNull(metatable), noMetatable, slow, unit.likely());

    build.SetInsertPoint(noMetatable);
    branch_on_relation(unit, build.CreateICmpEQ(unit.load_pointer(ra), unit.load_pointer(rb)), negated, target, next);

    build.SetInsertPoint(slow);
    unit.spill(pcpos);
    llvm::Value* equal = unit.call_helper(reinterpret_cast<const void*>(&execute_equal), pcpos, true);
    unit.reload(pcpos);
    branch_on_relation(unit, build.CreateICmpNE(equal, build.getInt32(0)), negated, target, next);
}

void lower_inst_jumpifcmp(TranslationUnit& unit, const Instruction* pc, int pcpos)
{
    llvm::IRBuilder<>& build = unit.builder();

    const auto op = static_cast<LuauOpcode>(LUAU_INSN_OP(*pc));
    const int ra = LUAU_INSN_A(*pc);
    const int rb = static_cast<int>(pc[1]);

    const bool lessEqual = op == LOP_JUMPIFLE || op == LOP_JUMPIFNOTLE;
    const bool negated = op == LOP_JUMPIFNOTLE || op == LOP_JUMPIFNOTLT;
    const int target = getJumpTarget(*pc, pcpos);
    const int next = next_pc(pc, pcpos);

    llvm::BasicBlock* number = unit.new_block("cmp.number");
    llvm::BasicBlock* other = unit.new_block("cmp.other");
    llvm::BasicBlock* strings = unit.new_block("cmp.string");
    llvm::BasicBlock* slow = unit.new_block("cmp.slow");

    const BytecodeTypes& bcTypes = unit.bytecode().bcTypes[pcpos];
    check_operand_tags(unit, pcpos, {{ra, bcTypes.a}, {rb, bcTypes.b}}, number, other);

    // ordered comparisons are false for NaN, the NOT variants then take the jump just like the interpreter does
    build.SetInsertPoint(number);
    llvm::Value* a = unit.load_number(ra);
    llvm::Value* b = unit.load_number(rb);
    branch_on_relation(unit, lessEqual ? build.CreateFCmpOLE(a, b) : build.CreateFCmpOLT(a, b), negated, target, next);

    build.SetInsertPoint(other);
    llvm::Value* isString = build.CreateICmpEQ(unit.load_tag(ra), build.getInt32(LUA_TSTRING));
    build.CreateCondBr(build.CreateAnd(isString, build.CreateICmpEQ(unit.load_tag(rb), build.getInt32(LUA_TSTRING))), strings, slow);

    // luaV_strcmp neither allocates nor reads the stack, so nothing has to be spilled for it
    build.SetInsertPoint(strings);
    llvm::Value* order = unit.call_native(
        reinterpret_cast<const void*>(&luaV_strcmp), unit.types().i32Ty, {unit.load_pointer(ra), unit.load_pointer(rb)}
    );
    llvm::Value* holds = lessEqual ? build.CreateICmpSLE(order, build.getInt32(0)) : build.CreateICmpSLT(order, build.getInt32(0));
    branch_on_relation(unit, holds, negated, target, next);

    build.SetInsertPoint(slow);
    unit.spill(pcpos);
    const void* helper = lessEqual ? reinterpret_cast<const void*>(&execute_lessequal) : reinterpret_cast<const void*>(&execute_lessthan);
    llvm::Value* result = unit.call_helper(helper, pcpos, true);
    unit.reload(pcpos);
    branch_on_relation(unit, build.CreateICmpNE(result, build.getInt32(0)), negated, target, next);
}

void lower_inst_jumpxeqk(TranslationUnit& unit, const Instruction* pc, int pcpos)
{
    llvm::IRBuilder<>& build = unit.builder();

    const auto op = static_cast<LuauOpcode>(LUAU_INSN_OP(*pc));
    const int ra = LUAU_INSN_A(*pc);
    const uint32_t aux = pc[1];

    const bool negated = LUAU_INSN_AUX_NOT(aux) != 0;
    const int target = getJumpTarget(*pc, pcpos);
    const int next = next_pc(pc, pcpos);

    llvm::Value* tag = unit.load_tag(ra);

    if (op == LOP_JUMPXEQKNIL)
    {
        branch_on_relation(unit, build.CreateICmpEQ(tag, build.getInt32(LUA_TNIL)), negated, target, next);
        return;
    }

    // the value is only inspected once its type matches, so the payload of other types is never misread
    llvm::BasicBlock* sameType = unit.new_block("eqk.same");
    llvm::BasicBlock* different = unit.new_block("eqk.different");

    int expectedTag = LUA_TNIL;

    switch (op)
    {
    case LOP_JUMPXEQKB:
        expectedTag = LUA_TBOOLEAN;
        break;
    case LOP_JUMPXEQKN:
        expectedTag = LUA_TNUMBER;
        break;
    case LOP_JUMPXEQKS:
        expectedTag = LUA_TSTRING;
        break;
    default:
        LUAU_UNREACHABLE();
    }

    build.CreateCondBr(build.CreateICmpEQ(tag, build.getInt32(expectedTag)), sameType, different);

    build.SetInsertPoint(different);
    branch_on_relation(unit, build.getFalse(), negated, target, next);

    build.SetInsertPoint(sameType);

    llvm::Value* equal = nullptr;

    switch (op)
    {
    case LOP_JUMPXEQKB:
        equal = build.CreateICmpEQ(unit.load_boolean(ra), build.getInt32(LUAU_INSN_AUX_KB(aux)));
        break;
    case LOP_JUMPXEQKN:
    {
        const TValue& kv = unit.bytecode().proto->k[LUAU_INSN_AUX_KV(aux)];
        LUAU_ASSERT(ttisnumber(&kv));

        equal = build.CreateFCmpOEQ(unit.load_number(ra), llvm::ConstantFP::get(unit.types().doubleTy, nvalue(&kv)));
        break;
    }
    case LOP_JUMPXEQKS:
        equal = build.CreateICmpEQ(unit.load_pointer(ra), unit.const_ptr(string_constant(unit, LUAU_INSN_AUX_KV(aux))));
        break;
    default:
        LUAU_UNREACHABLE();
    }

    branch_on_relation(unit, equal, negated, target, next);
}

static llvm::Value* number_floor(TranslationUnit& unit, llvm::Value* value)
{
    return unit.builder().CreateUnaryIntrinsic(llvm::Intrinsic::floor, value);
}

static llvm::Value* number_arith(TranslationUnit& unit, const LuauOpcode op, llvm::Value* a, llvm::Value* b, const TValue* kb)
{
    llvm::IRBuilder<>& build = unit.builder();

    switch (op)
    {
    case LOP_ADD:
    case LOP_ADDK:
        return build.CreateFAdd(a, b);
    case LOP_SUB:
    case LOP_SUBK:
    case LOP_SUBRK:
        return build.CreateFSub(a, b);
    case LOP_MUL:
    case LOP_MULK:
        return build.CreateFMul(a, b);
    case LOP_DIV:
    case LOP_DIVK:
    case LOP_DIVRK:
        return build.CreateFDiv(a, b);
    case LOP_IDIV:
    case LOP_IDIVK:
        // luai_numidiv
        return number_floor(unit, build.CreateFDiv(a, b));
    case LOP_MOD:
    case LOP_MODK:
        // luai_nummod
        return build.CreateFSub(a, build.CreateFMul(number_floor(unit, build.CreateFDiv(a, b)), b));
    case LOP_POW:
        return build.CreateBinaryIntrinsic(llvm::Intrinsic::pow, a, b);
    case LOP_POWK:
    {
        // the interpreter specializes ^2, ^0.5 and ^3 because pow is very slow
        const double k = nvalue(kb);

        if (k == 2.0)
            return build.CreateFMul(a, a);
        if (k == 0.5)
            return build.CreateUnaryIntrinsic(llvm::Intrinsic::sqrt, a);
        if (k == 3.0)
            return build.CreateFMul(build.CreateFMul(a, a), a);

        return build.CreateBinaryIntrinsic(llvm::Intrinsic::pow, a, b);
    }
    case LOP_MINUS:
        return build.CreateFNeg(a);
    default:
        LUAU_ASSERT(!"Unsupported arithmetic opcode");
        LUAU_UNREACHABLE();
    }
}

enum class VectorForm : uint8_t
{
    VectorVector,
    VectorNumber,
    NumberVector,
};

// the operand combinations that lvmexecute.cpp handles inline for vectors
static bool has_vector_form(const LuauOpcode op, const VectorForm form)
{
    switch (op)
    {
    case LOP_ADD:
    case LOP_SUB:
    case LOP_MINUS:
        return form == VectorForm::VectorVector;
    case LOP_MUL:
    case LOP_DIV:
        return true;
    case LOP_IDIV:
    case LOP_MULK:
    case LOP_DIVK:
    case LOP_IDIVK:
        return form == VectorForm::VectorNumber;
    case LOP_DIVRK:
        return form == VectorForm::NumberVector;
    default:
        return false;
    }
}

static bool may_hold(const TValue* k, const uint8_t type, const int tag)
{
    if (k)
        return tag == LUA_TNUMBER;

    return type == LBC_TYPE_ANY || type == (tag == LUA_TNUMBER ? LBC_TYPE_NUMBER : LBC_TYPE_VECTOR);
}

static llvm::Value* vector_operand(TranslationUnit& unit, const int reg, const TValue* k, const int tag)
{
    llvm::IRBuilder<>& build = unit.builder();
    const VmTypes& types = unit.types();

    if (tag == LUA_TVECTOR)
        return unit.load_vector(reg);

    llvm::Value* number = k ? llvm::ConstantFP::get(types.doubleTy, nvalue(k)) : unit.load_number(reg);

    return build.CreateVectorSplat(LUA_VECTOR_SIZE, build.CreateFPTrunc(number, types.floatTy));
}

static llvm::Value* vector_arith(TranslationUnit& unit, const LuauOpcode op, llvm::Value* a, llvm::Value* b)
{
    llvm::IRBuilder<>& build = unit.builder();

    switch (op)
    {
    case LOP_ADD:
        return build.CreateFAdd(a, b);
    case LOP_SUB:
        return build.CreateFSub(a, b);
    case LOP_MUL:
    case LOP_MULK:
        return build.CreateFMul(a, b);
    case LOP_DIV:
    case LOP_DIVK:
    case LOP_DIVRK:
        return build.CreateFDiv(a, b);
    case LOP_IDIV:
    case LOP_IDIVK:
        return number_floor(unit, build.CreateFDiv(a, b));
    case LOP_MINUS:
        return build.CreateFNeg(a);
    default:
        LUAU_ASSERT(!"Unsupported vector opcode");
        LUAU_UNREACHABLE();
    }
}

// Lowers arithmetic that the bytecode types say involves a vector; returns false when no vector form applies, which
// leaves the instruction to the number path
static bool lower_vector_arith(
    TranslationUnit& unit,
    const int pcpos,
    const LuauOpcode op,
    const int ra,
    const int regB,
    const int regC,
    const TValue* kb,
    const TValue* kc,
    const uint8_t typeB,
    const uint8_t typeC,
    const int nextpc
)
{
    llvm::IRBuilder<>& build = unit.builder();

    if (typeB != LBC_TYPE_VECTOR && typeC != LBC_TYPE_VECTOR)
        return false;

    std::vector<VectorForm> forms;

    for (VectorForm form : {VectorForm::VectorVector, VectorForm::VectorNumber, VectorForm::NumberVector})
    {
        const int tagB = form == VectorForm::NumberVector ? LUA_TNUMBER : LUA_TVECTOR;
        const int tagC = form == VectorForm::VectorNumber ? LUA_TNUMBER : LUA_TVECTOR;

        if (has_vector_form(op, form) && may_hold(kb, typeB, tagB) && may_hold(kc, typeC, tagC))
            forms.push_back(form);
    }

    if (forms.empty())
        return false;

    llvm::BasicBlock* slow = unit.new_block("slow");

    for (size_t i = 0; i < forms.size(); i++)
    {
        const bool last = i + 1 == forms.size();
        const int tagB = forms[i] == VectorForm::NumberVector ? LUA_TNUMBER : LUA_TVECTOR;
        const int tagC = forms[i] == VectorForm::VectorNumber ? LUA_TNUMBER : LUA_TVECTOR;

        llvm::BasicBlock* fast = unit.new_block("vector");
        llvm::BasicBlock* failure = last ? slow : unit.new_block("vector.next");

        // an operand can only be speculated on in the last form, the earlier ones fall through to the next
        const uint8_t checkB = last ? typeB : uint8_t(LBC_TYPE_ANY);
        const uint8_t checkC = last ? typeC : uint8_t(LBC_TYPE_ANY);
        check_operand_tags(unit, pcpos, {{regB, checkB, tagB}, {regC, checkC, tagC}}, fast, failure);

        build.SetInsertPoint(fast);
        llvm::Value* a = vector_operand(unit, regB, kb, tagB);
        llvm::Value* b = vector_operand(unit, regC, kc, tagC);
        unit.store_vector(ra, vector_arith(unit, op, a, b));
        build.CreateBr(unit.block_at(nextpc));

        build.SetInsertPoint(failure);
    }

    unit.spill(pcpos);
    unit.call_helper(reinterpret_cast<const void*>(&execute_arith), pcpos, false);
    unit.reload(pcpos);
    build.CreateBr(unit.block_at(nextpc));

    return true;
}

void lower_inst_binary(TranslationUnit& unit, const Instruction* pc, int pcpos)
{
    llvm::IRBuilder<>& build = unit.builder();
    const VmTypes& types = unit.types();
    const Proto* proto = unit.bytecode().proto;

    const auto op = static_cast<LuauOpcode>(LUAU_INSN_OP(*pc));
    const int ra = LUAU_INSN_A(*pc);

    // operands are a register or a constant, and only registers need a type check
    int regB = -1;
    int regC = -1;
    const TValue* kb = nullptr;
    const TValue* kc = nullptr;

    switch (op)
    {
    case LOP_ADDK:
    case LOP_SUBK:
    case LOP_MULK:
    case LOP_DIVK:
    case LOP_IDIVK:
    case LOP_MODK:
    case LOP_POWK:
        regB = LUAU_INSN_B(*pc);
        kc = &proto->k[LUAU_INSN_C(*pc)];
        break;
    case LOP_SUBRK:
    case LOP_DIVRK:
        kb = &proto->k[LUAU_INSN_B(*pc)];
        regC = LUAU_INSN_C(*pc);
        break;
    case LOP_MINUS:
        regB = LUAU_INSN_B(*pc);
        regC = regB;
        break;
    default:
        regB = LUAU_INSN_B(*pc);
        regC = LUAU_INSN_C(*pc);
        break;
    }

    // a constant that isn't a number sends every execution down the helper, which raises the same error as the interpreter
    if ((kb && !ttisnumber(kb)) || (kc && !ttisnumber(kc)))
    {
        lower_inst_helper(unit, reinterpret_cast<const void*>(&execute_arith), pcpos);
        return;
    }

    const BytecodeTypes& bcTypes = unit.bytecode().bcTypes[pcpos];
    const int nextpc = next_pc(pc, pcpos);

    const uint8_t typeC = op == LOP_MINUS ? bcTypes.a : bcTypes.b;

    if (lower_vector_arith(unit, pcpos, op, ra, regB, regC, kb, kc, bcTypes.a, typeC, nextpc))
        return;

    llvm::BasicBlock* fast = unit.new_block("fast");
    llvm::BasicBlock* slow = unit.new_block("slow");

    check_operand_tags(unit, pcpos, {{regB, bcTypes.a}, {regC, bcTypes.b}}, fast, slow);

    build.SetInsertPoint(fast);
    llvm::Value* a = kb ? llvm::ConstantFP::get(types.doubleTy, nvalue(kb)) : unit.load_number(regB);
    llvm::Value* b = kc ? llvm::ConstantFP::get(types.doubleTy, nvalue(kc)) : unit.load_number(regC);
    unit.store_number(ra, number_arith(unit, op, a, b, kc));
    build.CreateBr(unit.block_at(nextpc));

    build.SetInsertPoint(slow);
    unit.spill(pcpos);
    unit.call_helper(reinterpret_cast<const void*>(&execute_arith), pcpos, false);
    unit.reload(pcpos);
    build.CreateBr(unit.block_at(nextpc));
}

void lower_inst_logical(TranslationUnit& unit, const Instruction* pc, int pcpos)
{
    llvm::IRBuilder<>& build = unit.builder();

    const auto op = static_cast<LuauOpcode>(LUAU_INSN_OP(*pc));
    const int ra = LUAU_INSN_A(*pc);
    const int rb = LUAU_INSN_B(*pc);

    const bool isAnd = op == LOP_AND || op == LOP_ANDK;
    const bool isConstant = op == LOP_ANDK || op == LOP_ORK;

    llvm::Value* falsy = unit.is_falsy(rb);
    llvm::Value* left = unit.load_reg(rb);
    llvm::Value* right = isConstant ? unit.constant(LUAU_INSN_C(*pc)) : unit.load_reg(LUAU_INSN_C(*pc));

    // and: l_isfalse(rb) ? rb : rc, or: l_isfalse(rb) ? rc : rb
    unit.store_reg(ra, isAnd ? build.CreateSelect(falsy, left, right) : build.CreateSelect(falsy, right, left));
}

void lower_inst_not(TranslationUnit& unit, const Instruction* pc, int pcpos)
{
    llvm::Value* falsy = unit.is_falsy(LUAU_INSN_B(*pc));

    unit.store_boolean(LUAU_INSN_A(*pc), unit.builder().CreateZExt(falsy, unit.types().i32Ty));
}

void lower_inst_length(TranslationUnit& unit, const Instruction* pc, int pcpos)
{
    llvm::IRBuilder<>& build = unit.builder();
    const VmTypes& types = unit.types();

    const int ra = LUAU_INSN_A(*pc);
    const int rb = LUAU_INSN_B(*pc);
    const int nextpc = next_pc(pc, pcpos);

    llvm::BasicBlock* isTable = unit.new_block("len.table");
    llvm::BasicBlock* notTable = unit.new_block("len.other");
    llvm::BasicBlock* table = unit.new_block("len.array");
    llvm::BasicBlock* string = unit.new_block("len.string");
    llvm::BasicBlock* slow = unit.new_block("slow");

    llvm::Value* tag = unit.load_tag(rb);
    build.CreateCondBr(build.CreateICmpEQ(tag, build.getInt32(LUA_TTABLE)), isTable, notTable);

    build.SetInsertPoint(isTable);
    llvm::Value* h = unit.load_pointer(rb);
    llvm::Value* metatable = unit.load_field(types.ptrTy, h, offsetof(LuaTable, metatable));
    build.CreateCondBr(build.CreateIsNull(metatable), table, slow, unit.likely());

    // luaH_getn neither allocates nor reads the stack
    build.SetInsertPoint(table);
    llvm::Value* length = unit.call_native(reinterpret_cast<const void*>(&luaH_getn), types.i32Ty, {h});
    unit.store_number(ra, build.CreateSIToFP(length, types.doubleTy));
    build.CreateBr(unit.block_at(nextpc));

    build.SetInsertPoint(notTable);
    build.CreateCondBr(build.CreateICmpEQ(tag, build.getInt32(LUA_TSTRING)), string, slow, unit.likely());

    build.SetInsertPoint(string);
    llvm::Value* len = unit.load_field(types.i32Ty, unit.load_pointer(rb), offsetof(TString, len));
    unit.store_number(ra, build.CreateUIToFP(len, types.doubleTy));
    build.CreateBr(unit.block_at(nextpc));

    build.SetInsertPoint(slow);
    unit.spill(pcpos);
    unit.call_helper(reinterpret_cast<const void*>(&execute_length), pcpos, false);
    unit.reload(pcpos);
    build.CreateBr(unit.block_at(nextpc));
}

namespace
{

struct RegDefVisitor
{
    int reg;
    bool defined = false;

    void def(const int r)
    {
        defined |= r == reg;
    }

    void def_range(const int start, const int count)
    {
        defined |= reg >= start && (count == -1 || reg < start + count);
    }

    void use(int) {}

    void use_range(int, int) {}

    void capture(int) {}
};

} // namespace

// The step of the numeric loop that the FORNPREP at 'prepPc' starts, when its bytecode block ends up with a number
// constant in the step register. Nothing writes that register while the loop runs, so the step stays that constant
static std::optional<double> loop_step_constant(TranslationUnit& unit, const int prepPc)
{
    const BytecodeFunction& function = unit.bytecode();
    Proto* proto = function.proto;
    LUAU_ASSERT(LUAU_INSN_OP(proto->code[prepPc]) == LOP_FORNPREP);

    const uint32_t blockIdx = function.cfg.blockAtPc[prepPc];
    if (blockIdx == kNoAssociatedBlock)
        return std::nullopt;

    const int step = LUAU_INSN_A(proto->code[prepPc]) + 1;
    std::optional<double> value;

    for (int pcpos = function.bcBlocks[blockIdx].startpc; pcpos < prepPc;)
    {
        const Instruction* pc = &proto->code[pcpos];
        const auto op = static_cast<LuauOpcode>(LUAU_INSN_OP(*pc));

        RegDefVisitor visitor{step};
        visit_bytecode_reg_defs_uses(visitor, proto, pcpos);

        if (visitor.defined)
        {
            value.reset();

            if (op == LOP_LOADN)
                value = double(LUAU_INSN_D(*pc));
            else if (op == LOP_LOADK && ttisnumber(&proto->k[LUAU_INSN_D(*pc)]))
                value = nvalue(&proto->k[LUAU_INSN_D(*pc)]);
        }

        pcpos += getOpLength(op);
    }

    return value;
}

// step > 0 ? idx <= limit : limit <= idx, the condition FORNPREP and FORNLOOP share
static llvm::Value* forn_condition(TranslationUnit& unit, llvm::Value* limit, llvm::Value* step, llvm::Value* idx)
{
    llvm::IRBuilder<>& build = unit.builder();

    // a constant step is the common case, and folding the select keeps the other comparison out of the loop
    if (auto* constant = llvm::dyn_cast<llvm::ConstantFP>(step))
        return constant->getValueAPF().convertToDouble() > 0 ? build.CreateFCmpOLE(idx, limit) : build.CreateFCmpOLE(limit, idx);

    llvm::Value* positive = build.CreateFCmpOGT(step, llvm::ConstantFP::get(unit.types().doubleTy, 0.0));

    return build.CreateSelect(positive, build.CreateFCmpOLE(idx, limit), build.CreateFCmpOLE(limit, idx));
}

// the largest step an integer loop takes, which keeps the bounds it checks for far from the int32 limits
static constexpr int kMaxIntegerLoopStep = 1 << 16;

static int integer_step(const std::optional<double> step)
{
    return step && *step == std::trunc(*step) && *step != 0 && std::fabs(*step) <= kMaxIntegerLoopStep ? int(*step) : 0;
}

// The last value an integer counter can take: step > 0 ? idx <= limit : limit <= idx holds for an integer idx exactly
// when it holds for the floor (or ceiling) of the limit
static llvm::Value* integer_bound(TranslationUnit& unit, llvm::Value* limit, const int step)
{
    llvm::IRBuilder<>& build = unit.builder();

    return build.CreateUnaryIntrinsic(step > 0 ? llvm::Intrinsic::floor : llvm::Intrinsic::ceil, limit);
}

// The counter for an index, and whether an int32 counter can follow the loop from there: the index is an integer, and
// stepping past the bound stays in range
static std::pair<llvm::Value*, llvm::Value*> integer_counter(TranslationUnit& unit, llvm::Value* index, llvm::Value* limit, const int step)
{
    llvm::IRBuilder<>& build = unit.builder();
    const VmTypes& types = unit.types();

    llvm::Value* counter = build.CreateFreeze(build.CreateFPToSI(index, types.i32Ty), "counter");
    llvm::Value* exact = build.CreateFCmpOEQ(build.CreateSIToFP(counter, types.doubleTy), index);

    auto constant = [&](const double value)
    {
        return llvm::ConstantFP::get(types.doubleTy, value);
    };

    llvm::Value* bound = integer_bound(unit, limit, step);
    const double low = double(INT32_MIN) + (step < 0 ? -step : 0);
    const double high = double(INT32_MAX) - (step > 0 ? step : 0);
    llvm::Value* inRange = build.CreateAnd(build.CreateFCmpOGE(bound, constant(low)), build.CreateFCmpOLE(bound, constant(high)));

    return {counter, build.CreateAnd(exact, inRange)};
}

void TranslationUnit::find_integer_loops()
{
    m_integerLoops.clear();

    // the baseline only speculates on annotations, and a loop counter isn't one
    if (!m_speculate || m_profiling || !m_nproto.feedback)
        return;

    const Proto* proto = m_bytecode.proto;

    for (int pcpos = 0; pcpos < proto->sizecode;)
    {
        const Instruction* pc = &proto->code[pcpos];
        const auto op = static_cast<LuauOpcode>(LUAU_INSN_OP(*pc));

        if (op == LOP_FORNLOOP)
        {
            const int reg = LUAU_INSN_A(*pc) + 2;
            const int loopStart = getJumpTarget(*pc, pcpos);
            const int prepPc = loopStart - 1;
            const Instruction* prep = &proto->code[prepPc];

            const bool candidate = prepPc >= 0 && LUAU_INSN_OP(*prep) == LOP_FORNPREP && int(LUAU_INSN_A(*prep)) + 2 == reg &&
                                   (feedback_at(prepPc) & kFeedbackLoopNotInteger) == 0;
            const int step = candidate ? integer_step(loop_step_constant(*this, prepPc)) : 0;

            // the compiler copies a loop variable that the body assigns to, but nothing relies on that
            bool written = false;
            for (int body = loopStart; step != 0 && body < pcpos; body += getOpLength(LuauOpcode(LUAU_INSN_OP(proto->code[body]))))
            {
                RegDefVisitor visitor{reg};
                visit_bytecode_reg_defs_uses(visitor, m_bytecode.proto, body);
                written |= visitor.defined;
            }

            if (step != 0 && !written)
                m_integerLoops.push_back({prepPc, pcpos, reg, step, m_builder.CreateAlloca(m_types.i32Ty, nullptr, "counter")});
        }

        pcpos += getOpLength(op);
    }
}

const IntegerLoop* TranslationUnit::integer_loop(const int pcpos) const
{
    for (const IntegerLoop& loop : m_integerLoops)
    {
        if (loop.prepPc == pcpos || loop.loopPc == pcpos)
            return &loop;
    }

    return nullptr;
}

// A frame of an older version of the code, or of the interpreter, can be in a loop that a counter can't follow; that
// frame stays in the interpreter
void TranslationUnit::emit_integer_loop_entry(const int pcpos)
{
    for (const IntegerLoop& loop : m_integerLoops)
    {
        if (pcpos <= loop.prepPc || pcpos > loop.loopPc)
            continue;

        llvm::Value* index = m_builder.CreateLoad(m_types.doubleTy, reg_slot(loop.reg));
        llvm::Value* limit = m_builder.CreateLoad(m_types.doubleTy, reg_slot(loop.reg - 2));
        auto [counter, fits] = integer_counter(*this, index, limit, loop.step);

        llvm::BasicBlock* follow = new_block("enter.counter");
        llvm::BasicBlock* stay = new_block("enter.nocounter");
        m_builder.CreateCondBr(fits, follow, stay, likely());

        m_builder.SetInsertPoint(stay);
        emit_feedback_bits(loop.prepPc, m_builder.getInt32(kFeedbackLoopNotInteger));
        m_builder.CreateBr(m_bailBlock);

        m_builder.SetInsertPoint(follow);
        m_builder.CreateStore(counter, loop.counter);
    }
}

void lower_inst_fornprep(TranslationUnit& unit, const Instruction* pc, int pcpos)
{
    llvm::IRBuilder<>& build = unit.builder();
    const VmTypes& types = unit.types();

    const int ra = LUAU_INSN_A(*pc);
    const int target = getJumpTarget(*pc, pcpos);
    const int next = next_pc(pc, pcpos);

    llvm::Value* numbers =
        build.CreateAnd(build.CreateAnd(tag_is(unit, ra, LUA_TNUMBER), tag_is(unit, ra + 1, LUA_TNUMBER)), tag_is(unit, ra + 2, LUA_TNUMBER));

    llvm::BasicBlock* fast = unit.new_block("forn.fast");

    if (const IntegerLoop* loop = unit.integer_loop(pcpos))
    {
        llvm::BasicBlock* check = unit.new_block("forn.counter");
        llvm::BasicBlock* start = unit.new_block("forn.start");
        llvm::BasicBlock* notInteger = unit.new_block("forn.notinteger");

        build.CreateCondBr(numbers, fast, notInteger, unit.likely());

        build.SetInsertPoint(fast);
        llvm::Value* limit = unit.load_number(ra);
        llvm::Value* index = unit.load_number(ra + 2);
        build.CreateCondBr(forn_condition(unit, limit, unit.load_number(ra + 1), index), check, unit.block_at(target));

        build.SetInsertPoint(check);
        auto [counter, fits] = integer_counter(unit, index, limit, loop->step);
        build.CreateCondBr(fits, start, notInteger, unit.likely());

        build.SetInsertPoint(start);
        build.CreateStore(counter, loop->counter);
        build.CreateBr(unit.block_at(next));

        // the next compilation counts this loop in numbers
        build.SetInsertPoint(notInteger);
        unit.emit_feedback_bits(pcpos, build.getInt32(kFeedbackLoopNotInteger));
        build.CreateBr(unit.vm_exit(pcpos));
        return;
    }

    llvm::BasicBlock* slow = unit.new_block("forn.slow");

    build.CreateCondBr(numbers, fast, slow, unit.likely());

    build.SetInsertPoint(fast);
    llvm::Value* limit = unit.load_number(ra);
    llvm::Value* index = unit.load_number(ra + 2);
    llvm::Value* enter = forn_condition(unit, limit, unit.load_number(ra + 1), index);

    if (const int step = unit.profiling() ? integer_step(loop_step_constant(unit, pcpos)) : 0; step != 0)
    {
        llvm::Value* fits = integer_counter(unit, index, limit, step).second;
        llvm::Value* bits =
            build.CreateSelect(build.CreateAnd(enter, build.CreateNot(fits)), build.getInt32(kFeedbackLoopNotInteger), build.getInt32(0));
        unit.emit_feedback_bits(pcpos, bits);
    }

    build.CreateCondBr(enter, unit.block_at(next), unit.block_at(target));

    // non-number loop parameters get converted in place or raise an error
    build.SetInsertPoint(slow);
    unit.spill(pcpos);
    llvm::Value* skip = unit.call_helper(reinterpret_cast<const void*>(&execute_fornprep), pcpos, true);
    unit.reload(pcpos);
    build.CreateCondBr(build.CreateICmpNE(skip, build.getInt32(0)), unit.block_at(target), unit.block_at(next));
}

void lower_inst_fornloop(TranslationUnit& unit, const Instruction* pc, int pcpos)
{
    llvm::IRBuilder<>& build = unit.builder();

    const int ra = LUAU_INSN_A(*pc);

    unit.emit_safepoint(pcpos, true);

    // FORNPREP and the entry points have made sure the counter can step past the bound without overflowing
    if (const IntegerLoop* loop = unit.integer_loop(pcpos))
    {
        const VmTypes& types = unit.types();

        llvm::Value* counter = build.CreateNSWAdd(build.CreateLoad(types.i32Ty, loop->counter), build.getInt32(loop->step), "counter");
        build.CreateStore(counter, loop->counter);
        unit.store_number(ra + 2, build.CreateSIToFP(counter, types.doubleTy));

        llvm::Value* bound = build.CreateFreeze(build.CreateFPToSI(integer_bound(unit, unit.load_number(ra), loop->step), types.i32Ty));
        llvm::Value* inLoop = loop->step > 0 ? build.CreateICmpSLE(counter, bound) : build.CreateICmpSGE(counter, bound);

        build.CreateCondBr(inLoop, unit.block_at(getJumpTarget(*pc, pcpos)), unit.block_at(next_pc(pc, pcpos)));
        return;
    }

    // FORNPREP has made sure all three are numbers
    llvm::Value* limit = unit.load_number(ra);
    llvm::Value* step = nullptr;

    const int loopStart = getJumpTarget(*pc, pcpos);
    const Instruction* prep = &unit.bytecode().proto->code[loopStart - 1];

    if (LUAU_INSN_OP(*prep) == LOP_FORNPREP && LUAU_INSN_A(*prep) == ra)
    {
        if (std::optional<double> constant = loop_step_constant(unit, loopStart - 1))
            step = llvm::ConstantFP::get(unit.types().doubleTy, *constant);
    }

    if (!step)
        step = unit.load_number(ra + 1);
    llvm::Value* idx = build.CreateFAdd(unit.load_number(ra + 2), step);

    unit.store_number(ra + 2, idx);

    build.CreateCondBr(forn_condition(unit, limit, step, idx), unit.block_at(getJumpTarget(*pc, pcpos)), unit.block_at(next_pc(pc, pcpos)));
}

void lower_inst_forgprep(TranslationUnit& unit, const Instruction* pc, int pcpos)
{
    llvm::IRBuilder<>& build = unit.builder();
    const VmTypes& types = unit.types();

    const auto op = static_cast<LuauOpcode>(LUAU_INSN_OP(*pc));
    const int ra = LUAU_INSN_A(*pc);
    const int target = getJumpTarget(*pc, pcpos);
    const void* helper = reinterpret_cast<const void*>(&execute_forgprep);

    if (op == LOP_FORGPREP)
    {
        unit.spill(pcpos);
        unit.call_helper(helper, pcpos, false);
        unit.reload(pcpos);
        build.CreateBr(unit.block_at(target));
        return;
    }

    // pairs and ipairs of a table set up builtin iteration, as long as nothing could have replaced them
    llvm::Value* safeenv = unit.is_safe_env();
    llvm::Value* state = op == LOP_FORGPREP_NEXT
                             ? tag_is(unit, ra + 2, LUA_TNIL)
                             : build.CreateAnd(
                                   tag_is(unit, ra + 2, LUA_TNUMBER),
                                   build.CreateFCmpOEQ(unit.load_number(ra + 2), llvm::ConstantFP::get(types.doubleTy, 0.0))
                               );

    lower_with_fallback(
        unit,
        build.CreateAnd(build.CreateAnd(safeenv, tag_is(unit, ra + 1, LUA_TTABLE)), state),
        helper,
        pcpos,
        target,
        [&]
        {
            unit.store_nil(ra);
            unit.store_lightuserdata(ra + 2, llvm::ConstantPointerNull::get(types.ptrTy), LU_TAG_ITERATOR);
        }
    );
}

// The builtin table iteration of execute_forgloop
static void lower_table_forgloop(TranslationUnit& unit, const Instruction* pc)
{
    llvm::IRBuilder<>& build = unit.builder();
    const VmTypes& types = unit.types();

    const int pcpos = static_cast<int>(pc - unit.bytecode().proto->code);
    const int ra = LUAU_INSN_A(*pc);
    const int aux = static_cast<int>(pc[1]);
    llvm::BasicBlock* loop = unit.block_at(getJumpTarget(*pc, pcpos));
    llvm::BasicBlock* exit = unit.block_at(next_pc(pc, pcpos));

    llvm::Value* h = unit.load_pointer(ra + 1);
    llvm::Value* index = build.CreateTrunc(unit.load_field(types.i64Ty, unit.reg_slot(ra + 2), offsetof(TValue, value)), types.i32Ty, "index");
    llvm::Value* sizearray = unit.load_field(types.i32Ty, h, offsetof(LuaTable, sizearray), "sizearray");
    llvm::Value* array = unit.load_field(types.ptrTy, h, offsetof(LuaTable, array), "array");

    for (int i = 2; i < aux; i++)
        unit.store_nil(ra + 3 + i);

    auto is_nil = [&](llvm::Value* value)
    {
        return build.CreateICmpEQ(unit.load_field(types.i32Ty, value, offsetof(TValue, tt), "tt"), build.getInt32(LUA_TNIL));
    };

    // the value always goes to ra + 4, which isn't a loop variable of a loop with only one, and then only the stack has it
    auto store_iteration = [&](llvm::Value* idx, llvm::Value* value)
    {
        llvm::Value* next = build.CreateAdd(idx, build.getInt32(1), "next");
        unit.store_lightuserdata(ra + 2, build.CreateIntToPtr(build.CreateZExt(next, types.i64Ty), types.ptrTy), LU_TAG_ITERATOR);
        types.copy_tvalue(build, (aux & 0xff) >= 2 ? unit.reg_slot(ra + 4) : unit.stack_slot(ra + 4), value);
        return next;
    };

    llvm::BasicBlock* entry = unit.current_block();
    llvm::BasicBlock* arrayFound = unit.new_block("forg.array.found");

    build.SetInsertPoint(arrayFound);
    llvm::PHINode* arrayIndex = build.CreatePHI(types.i32Ty, 2, "found");
    llvm::Value* element = build.CreateGEP(types.tvalueTy, array, build.CreateZExt(arrayIndex, types.i64Ty), "e");
    unit.store_number(ra + 3, build.CreateSIToFP(store_iteration(arrayIndex, element), types.doubleTy));
    build.CreateBr(loop);

    build.SetInsertPoint(entry);

    if (aux < 0)
    {
        // ipairs stops at the first nil, and it never gets to the hash part
        llvm::BasicBlock* check = unit.new_block("forg.check");
        build.CreateCondBr(build.CreateICmpULT(index, sizearray), check, exit, unit.likely());

        build.SetInsertPoint(check);
        build.CreateCondBr(is_nil(build.CreateGEP(types.tvalueTy, array, build.CreateZExt(index, types.i64Ty))), exit, arrayFound, unit.unlikely());
        arrayIndex->addIncoming(index, check);
        return;
    }

    llvm::BasicBlock* arrayHead = unit.new_block("forg.array");
    llvm::BasicBlock* arrayBody = unit.new_block("forg.array.body");
    llvm::BasicBlock* nodeEntry = unit.new_block("forg.node.entry");
    llvm::BasicBlock* nodeHead = unit.new_block("forg.node");
    llvm::BasicBlock* nodeBody = unit.new_block("forg.node.body");
    llvm::BasicBlock* nodeFound = unit.new_block("forg.node.found");
    build.CreateBr(arrayHead);

    build.SetInsertPoint(arrayHead);
    llvm::PHINode* idx = build.CreatePHI(types.i32Ty, 2, "idx");
    idx->addIncoming(index, entry);
    build.CreateCondBr(build.CreateICmpULT(idx, sizearray), arrayBody, nodeEntry, unit.likely());

    build.SetInsertPoint(arrayBody);
    idx->addIncoming(build.CreateAdd(idx, build.getInt32(1)), arrayBody);
    build.CreateCondBr(is_nil(build.CreateGEP(types.tvalueTy, array, build.CreateZExt(idx, types.i64Ty))), arrayHead, arrayFound);
    arrayIndex->addIncoming(idx, arrayBody);

    build.SetInsertPoint(nodeEntry);
    llvm::Value* lsizenode = build.CreateZExt(unit.load_field(types.i8Ty, h, offsetof(LuaTable, lsizenode)), types.i32Ty);
    llvm::Value* sizenode = build.CreateShl(build.getInt32(1), lsizenode, "sizenode");
    llvm::Value* nodes = unit.load_field(types.ptrTy, h, offsetof(LuaTable, node), "node");
    build.CreateBr(nodeHead);

    build.SetInsertPoint(nodeHead);
    llvm::PHINode* nidx = build.CreatePHI(types.i32Ty, 2, "nidx");
    nidx->addIncoming(idx, nodeEntry);
    llvm::Value* slot = build.CreateSub(nidx, sizearray);
    build.CreateCondBr(build.CreateICmpULT(slot, sizenode), nodeBody, exit);

    build.SetInsertPoint(nodeBody);
    llvm::Value* node = build.CreateGEP(types.i8Ty, nodes, build.CreateMul(build.CreateZExt(slot, types.i64Ty), build.getInt64(sizeof(LuaNode))), "n");
    nidx->addIncoming(build.CreateAdd(nidx, build.getInt32(1)), nodeBody);
    build.CreateCondBr(is_nil(types.field_ptr(build, node, offsetof(LuaNode, val))), nodeHead, nodeFound);

    // getnodekey, where the key tag shares its word with the chain
    build.SetInsertPoint(nodeFound);
    store_iteration(nidx, types.field_ptr(build, node, offsetof(LuaNode, val)));
    types.copy_tvalue(build, unit.reg_slot(ra + 3), types.field_ptr(build, node, offsetof(LuaNode, key)));
    llvm::Value* keyTag = build.CreateAnd(unit.load_field(types.i32Ty, node, kNodeKeyTagOffset), 0xf);
    unit.store_field(keyTag, unit.reg_slot(ra + 3), offsetof(TValue, tt));
    build.CreateBr(loop);
}

void lower_inst_forgloop(TranslationUnit& unit, const Instruction* pc, int pcpos)
{
    llvm::IRBuilder<>& build = unit.builder();

    const int ra = LUAU_INSN_A(*pc);
    const int target = getJumpTarget(*pc, pcpos);
    const int next = next_pc(pc, pcpos);

    unit.emit_safepoint(pcpos, true);

    // FORGPREP* only leave nil in ra when ra + 1 is the table and ra + 2 the iteration index
    llvm::BasicBlock* table = unit.new_block("forg.table");
    llvm::BasicBlock* generic = unit.new_block("forg.generic");
    build.CreateCondBr(build.CreateAnd(tag_is(unit, ra, LUA_TNIL), tag_is(unit, ra + 1, LUA_TTABLE)), table, generic, unit.likely());

    build.SetInsertPoint(table);
    lower_table_forgloop(unit, pc);

    build.SetInsertPoint(generic);
    unit.spill(pcpos);
    llvm::Value* result = unit.call_helper(reinterpret_cast<const void*>(&execute_forgloop), pcpos, true);

    llvm::BasicBlock* exit = unit.new_block("forg.exit");
    llvm::BasicBlock* loop = unit.new_block("forg.loop");

    llvm::SwitchInst* dispatch = build.CreateSwitch(result, exit, 2);
    dispatch->addCase(build.getInt32(kForgloopContinue), loop);
    // luau_finishop completes the instruction on resume and continues at either of the targets, which start blocks
    dispatch->addCase(build.getInt32(kForgloopYield), unit.yield_exit());

    build.SetInsertPoint(loop);
    unit.reload(pcpos);
    build.CreateBr(unit.block_at(target));

    build.SetInsertPoint(exit);
    unit.reload(pcpos);
    build.CreateBr(unit.block_at(next));

    // luau_finishop resumes a yielded iterator call at either target
    unit.mark_entry_point(target);
    unit.mark_entry_point(next);
}

void lower_inst_fastcall(TranslationUnit& unit, const Instruction* pc, int pcpos)
{
    llvm::IRBuilder<>& build = unit.builder();

    const int target = getJumpTarget(*pc, pcpos);
    const int next = next_pc(pc, pcpos);

    llvm::BasicBlock* generic = unit.new_block("fastcall.generic");

    if (!lower_builtin(unit, pc, pcpos, generic))
        build.CreateBr(generic);

    build.SetInsertPoint(generic);
    unit.spill(pcpos);
    llvm::Value* handled = unit.call_helper(reinterpret_cast<const void*>(&execute_fastcall), pcpos, true);
    unit.reload(pcpos);

    // the skip target isn't a block boundary, it's the instruction right past the paired CALL
    build.CreateCondBr(build.CreateICmpNE(handled, build.getInt32(0)), unit.block_at(target), unit.block_at(next));
}

void lower_inst_fastpcall(TranslationUnit& unit, const Instruction* pc, int pcpos)
{
    llvm::IRBuilder<>& build = unit.builder();

    const int target = getJumpTarget(*pc, pcpos);
    const int next = next_pc(pc, pcpos);

    unit.spill(pcpos);
    llvm::Value* result = unit.call_helper(reinterpret_cast<const void*>(&execute_fastpcall), pcpos, true);

    llvm::BasicBlock* fallback = unit.new_block("fastpcall.fallback");
    llvm::BasicBlock* enter = unit.new_block("fastpcall.enter");
    llvm::BasicBlock* done = unit.new_block("fastpcall.done");

    llvm::SwitchInst* dispatch = build.CreateSwitch(result, fallback, 3);
    dispatch->addCase(build.getInt32(kFastpcallEnter), enter);
    dispatch->addCase(build.getInt32(kFastpcallDone), done);
    dispatch->addCase(build.getInt32(kFastpcallYield), unit.yield_exit());

    build.SetInsertPoint(fallback);
    unit.reload(pcpos);
    build.CreateBr(unit.block_at(next));

    build.SetInsertPoint(enter);
    unit.emit_resume_ci();

    build.SetInsertPoint(done);
    unit.reload(pcpos);
    build.CreateBr(unit.block_at(target));

    // the protected call completes into the instruction past the paired CALL; an interrupt yield runs this one again
    unit.mark_entry_point(target);
    unit.mark_entry_point(pcpos);
}

void lower_inst_namecalludata(TranslationUnit& unit, const Instruction* pc, int pcpos)
{
    llvm::IRBuilder<>& build = unit.builder();

    const int next = next_pc(pc, pcpos);
    const int afterCall = next + getOpLength(static_cast<LuauOpcode>(LUAU_INSN_OP(pc[2])));

    unit.spill(pcpos);
    llvm::Value* result = unit.call_helper(reinterpret_cast<const void*>(&execute_namecalludata), pcpos, true);

    llvm::BasicBlock* fallback = unit.new_block("namecall.fallback");
    llvm::BasicBlock* done = unit.new_block("namecall.done");

    llvm::SwitchInst* dispatch = build.CreateSwitch(result, fallback, 2);
    dispatch->addCase(build.getInt32(kNamecalludataDone), done);
    dispatch->addCase(build.getInt32(kNamecalludataYield), unit.yield_exit());

    build.SetInsertPoint(fallback);
    unit.reload(pcpos);
    build.CreateBr(unit.block_at(next));

    build.SetInsertPoint(done);
    unit.reload(pcpos);
    build.CreateBr(unit.block_at(afterCall));

    unit.mark_entry_point(afterCall);
}

void lower_inst_finconstruct(TranslationUnit& unit, const Instruction* pc, int pcpos)
{
    llvm::IRBuilder<>& build = unit.builder();

    const int target = getJumpTarget(*pc, pcpos);
    const int next = next_pc(pc, pcpos);

    unit.spill(pcpos);
    llvm::Value* handled = unit.call_helper(reinterpret_cast<const void*>(&execute_finconstruct), pcpos, true);
    unit.reload(pcpos);

    build.CreateCondBr(build.CreateICmpNE(handled, build.getInt32(0)), unit.block_at(target), unit.block_at(next));
}

void lower_inst_cmpproto(TranslationUnit& unit, const Instruction* pc, int pcpos)
{
    llvm::IRBuilder<>& build = unit.builder();
    const VmTypes& types = unit.types();

    const int ra = LUAU_INSN_A(*pc);
    const uint32_t funid = pc[1];

    llvm::BasicBlock* target = unit.block_at(getJumpTarget(*pc, pcpos));
    llvm::BasicBlock* next = unit.block_at(next_pc(pc, pcpos));

    llvm::BasicBlock* function = unit.new_block("cmpproto.function");
    llvm::BasicBlock* lua = unit.new_block("cmpproto.lua");

    build.CreateCondBr(tag_is(unit, ra, LUA_TFUNCTION), function, target);

    build.SetInsertPoint(function);
    llvm::Value* cl = unit.load_pointer(ra);
    llvm::Value* isC = unit.load_field(types.i8Ty, cl, offsetof(Closure, isC));
    build.CreateCondBr(build.CreateICmpNE(isC, build.getInt8(0)), target, lua);

    build.SetInsertPoint(lua);
    llvm::Value* proto = unit.load_field(types.ptrTy, cl, offsetof(Closure, l.p));
    llvm::Value* protoFunid = unit.load_field(types.i32Ty, proto, offsetof(Proto, funid));
    build.CreateCondBr(build.CreateICmpEQ(protoFunid, build.getInt32(funid)), next, target);
}

void lower_inst_coverage(TranslationUnit& unit, const Instruction* pc, int pcpos)
{
    llvm::IRBuilder<>& build = unit.builder();

    // hits are a saturating counter in E, patched into the instruction like the interpreter does
    llvm::Value* address = unit.insn_ptr(pcpos);
    llvm::Value* insn = build.CreateLoad(unit.types().i32Ty, address);
    llvm::Value* hits = build.CreateLShr(insn, 8);

    llvm::Value* saturated = build.CreateICmpUGE(hits, build.getInt32((1 << 23) - 1));
    llvm::Value* updated = build.CreateSelect(saturated, hits, build.CreateAdd(hits, build.getInt32(1)));

    build.CreateStore(build.CreateOr(build.CreateShl(updated, 8), build.CreateAnd(insn, 0xff)), address);
}

void lower_inst_helper(TranslationUnit& unit, const void* helper, int pcpos)
{
    const Instruction* pc = &unit.bytecode().proto->code[pcpos];

    unit.spill(pcpos);
    unit.call_helper(helper, pcpos, false);
    unit.reload(pcpos);
}

} // namespace Luau::Jit
