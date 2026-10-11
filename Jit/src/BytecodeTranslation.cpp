// This file is part of the Luau programming language and is licensed under MIT License; see LICENSE.txt for details
#include "BytecodeTranslation.h"

#include "BytecodeAnalysis.h"
#include "BytecodeVisitUseDef.h"
#include "VmHelpers.h"

#include "Luau/BytecodeUtils.h"
#include "Luau/Common.h"

#include "lgc.h"
#include "lstate.h"
#include "ltable.h"

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/MDBuilder.h>
#include <llvm/IR/Module.h>

#include <algorithm>
#include <string.h>

LUAU_FASTFLAG(LuauBackedgeHeapCheck)

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
    const llvm::StringRef name
)
    : m_nproto(nproto)
    , m_bytecode(function)
    , m_hostHooks(hostHooks)
    , m_types(types)
    , m_module(module)
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

    m_entryPoints.clear();
    m_exitBlocks.clear();

    compute_dirty_regs();

    llvm::LLVMContext& context = m_module.getContext();

    m_entryBlock = llvm::BasicBlock::Create(context, "entry", m_function);
    m_bailBlock = llvm::BasicBlock::Create(context, "bail", m_function);
    m_yieldBlock = llvm::BasicBlock::Create(context, "yield", m_function);

    emit_prologue();

    // an entry pc without a case continues in the interpreter
    m_builder.SetInsertPoint(m_bailBlock);
    m_builder.CreateRet(m_builder.getInt32(1));

    // the interpreter exits luau_execute when it gets 0 back
    m_builder.SetInsertPoint(m_yieldBlock);
    m_builder.CreateRet(m_builder.getInt32(0));

    for (uint32_t blockIdx = 0; blockIdx < static_cast<uint32_t>(m_bytecode.bcBlocks.size()); blockIdx++)
        translate_block(blockIdx);

    // blocks can be referenced as branch targets and entry points after their bytecode block stopped being
    // translated early, those continue in the interpreter
    for (int pcpos = 0; pcpos < proto->sizecode; pcpos++)
    {
        if (llvm::BasicBlock* block = m_blocks[pcpos]; block && block->empty())
        {
            m_builder.SetInsertPoint(block);
            m_builder.CreateBr(vm_exit(pcpos));
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

void TranslationUnit::store_tag(const int reg, const int tag)
{
    m_builder.CreateStore(m_builder.getInt32(tag), m_builder.CreateStructGEP(m_types.tvalueTy, reg_slot(reg), VmTypes::kTValueTt));
}

void TranslationUnit::store_number(const int reg, llvm::Value* value)
{
    LUAU_ASSERT(value->getType() == m_types.doubleTy);

    m_builder.CreateStore(value, reg_slot(reg));
    store_tag(reg, LUA_TNUMBER);
}

void TranslationUnit::store_boolean(const int reg, llvm::Value* value)
{
    LUAU_ASSERT(value->getType() == m_types.i32Ty);

    m_builder.CreateStore(value, reg_slot(reg));
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

llvm::Value* TranslationUnit::call_helper(const void* helper, const int pcpos, const bool returnsInt)
{
    llvm::Type* result = returnsInt ? static_cast<llvm::Type*>(m_types.i32Ty) : m_types.voidTy;

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

void TranslationUnit::reload()
{
    llvm::Value* base = m_types.load_state_base(m_builder, m_state);
    m_builder.CreateStore(base, m_baseSlot);

    for (int reg = 0; reg < m_bytecode.proto->maxstacksize; reg++)
        m_types.copy_tvalue(m_builder, reg_slot(reg), m_types.stack_slot(m_builder, base, reg));
}

// Instructions that call into the VM on every path, which leaves all registers in sync with the stack afterwards
static bool always_syncs(const LuauOpcode op)
{
    switch (op)
    {
    case LOP_SETGLOBAL:
    case LOP_SETUPVAL:
    case LOP_CLOSEUPVALS:
    case LOP_NEWCLOSURE:
    case LOP_NAMECALL:
    case LOP_CALL:
    case LOP_CALLFB:
    case LOP_CONCAT:
    case LOP_NEWTABLE:
    case LOP_DUPTABLE:
    case LOP_SETLIST:
    case LOP_FORGPREP:
    case LOP_FORGPREP_NEXT:
    case LOP_FORGPREP_INEXT:
    case LOP_FORGLOOP:
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

    std::vector<std::bitset<256>> blockIn(blockCount);
    std::vector<std::bitset<256>> blockOut(blockCount);

    auto transfer = [&](const uint32_t blockIdx, std::bitset<256> dirty, const bool record)
    {
        const auto& [startpc, finishpc] = m_bytecode.bcBlocks[blockIdx];

        for (int pcpos = startpc; pcpos <= finishpc;)
        {
            if (record)
                m_dirtyIn[pcpos] = dirty;

            const auto op = static_cast<LuauOpcode>(LUAU_INSN_OP(proto->code[pcpos]));

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

void TranslationUnit::emit_safepoint(const int pcpos, const bool backedge)
{
    llvm::Value* global = load_field(m_types.ptrTy, m_state, offsetof(lua_State, global), "global");

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

    m_builder.CreateCondBr(cond, slow, resume, llvm::MDBuilder(m_module.getContext()).createUnlikelyBranchWeights());

    m_builder.SetInsertPoint(slow);
    spill(pcpos);

    const void* helper = backedge ? reinterpret_cast<const void*>(&execute_safepoint) : reinterpret_cast<const void*>(&execute_interrupt);
    llvm::Value* stop = call_helper(helper, pcpos, true);

    llvm::BasicBlock* reloadBlock = new_block("safepoint.reload");
    m_builder.CreateCondBr(m_builder.CreateICmpNE(stop, m_builder.getInt32(0)), yield_exit(), reloadBlock);

    m_builder.SetInsertPoint(reloadBlock);
    reload();
    m_builder.CreateBr(resume);

    m_builder.SetInsertPoint(resume);

    // after a yield or a break, the thread resumes by running this instruction again
    mark_entry_point(pcpos);
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

    // the tail call keeps the C stack flat no matter how deep the Lua call stack gets
    llvm::CallInst* call = m_builder.CreateCall(m_types.nativeEntryTy, target, {m_state, entrypc});
    call->setTailCallKind(llvm::CallInst::TCK_MustTail);
    m_builder.CreateRet(call);
}

void TranslationUnit::emit_table_barrier(llvm::Value* table, const int valueReg)
{
    // luaC_barriert
    llvm::BasicBlock* checkTable = new_block("barrier.table");
    llvm::BasicBlock* checkValue = new_block("barrier.value");
    llvm::BasicBlock* slow = new_block("barrier");
    llvm::BasicBlock* done = new_block("barrier.done");

    llvm::Value* collectable = m_builder.CreateICmpSGE(load_tag(valueReg), m_builder.getInt32(LUA_TSTRING));
    m_builder.CreateCondBr(collectable, checkTable, done);

    m_builder.SetInsertPoint(checkTable);
    llvm::Value* tableMarked = load_field(m_types.i8Ty, table, offsetof(GCheader, marked));
    llvm::Value* isBlack = m_builder.CreateICmpNE(m_builder.CreateAnd(tableMarked, bitmask(BLACKBIT)), m_builder.getInt8(0));
    m_builder.CreateCondBr(isBlack, checkValue, done);

    m_builder.SetInsertPoint(checkValue);
    llvm::Value* object = load_pointer(valueReg);
    llvm::Value* valueMarked = load_field(m_types.i8Ty, object, offsetof(GCheader, marked));
    llvm::Value* isWhite = m_builder.CreateICmpNE(m_builder.CreateAnd(valueMarked, WHITEBITS), m_builder.getInt8(0));
    m_builder.CreateCondBr(isWhite, slow, done, llvm::MDBuilder(m_module.getContext()).createUnlikelyBranchWeights());

    // the barrier never collects garbage or reads the stack, so nothing has to be spilled for it
    m_builder.SetInsertPoint(slow);
    call_native(reinterpret_cast<const void*>(&luaC_barriertable), m_types.voidTy, {m_state, table, object});
    m_builder.CreateBr(done);

    m_builder.SetInsertPoint(done);
}

llvm::BasicBlock* TranslationUnit::new_block(const llvm::Twine& name)
{
    return llvm::BasicBlock::Create(m_module.getContext(), name, m_function);
}

llvm::BasicBlock* TranslationUnit::block_at(const int pcpos)
{
    LUAU_ASSERT(pcpos >= 0 && static_cast<size_t>(pcpos) < m_blocks.size());

    if (!m_blocks[pcpos])
        m_blocks[pcpos] = llvm::BasicBlock::Create(m_module.getContext(), "bc_" + llvm::Twine(pcpos), m_function);

    return m_blocks[pcpos];
}

void TranslationUnit::mark_entry_point(int pcpos)
{
    llvm::BasicBlock* block = block_at(pcpos);

    for (const auto& [fst, snd] : m_entryPoints)
    {
        if (fst == pcpos)
            return;
    }

    m_entryPoints.push_back({pcpos, block});
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

    for (int i = 0; i < proto->maxstacksize; i++)
        m_types.copy_tvalue(m_builder, m_regSlots[i], m_types.stack_slot(m_builder, base, i));
}

void TranslationUnit::emit_dispatch()
{
    LUAU_ASSERT(m_entryBlock && !m_entryBlock->hasTerminator());
    LUAU_ASSERT(!m_entryPoints.empty());

    m_builder.SetInsertPoint(m_entryBlock);

    llvm::SwitchInst* dispatch = m_builder.CreateSwitch(m_entrypc, m_bailBlock, static_cast<unsigned>(m_entryPoints.size()));

    for (const auto& [fst, snd] : m_entryPoints)
        dispatch->addCase(m_builder.getInt32(fst), snd);
}

void TranslationUnit::translate_block(const uint32_t blockIdx)
{
    const Proto* proto = m_bytecode.proto;

    const auto& [startpc, finishpc] = m_bytecode.bcBlocks[blockIdx];
    LUAU_ASSERT(startpc >= 0 && finishpc >= startpc && finishpc < proto->sizecode);

    mark_entry_point(startpc);

    m_builder.ClearInsertionPoint();

    int pcpos = startpc;
    while (pcpos <= finishpc)
    {
        llvm::BasicBlock* target = block_at(pcpos);

        if (llvm::BasicBlock* previous = current_block(); previous && !previous->hasTerminator())
            m_builder.CreateBr(target);

        m_builder.SetInsertPoint(target);

        const Instruction* pc = &proto->code[pcpos];
        const auto op = static_cast<LuauOpcode>(LUAU_INSN_OP(*pc));

        if (!translate_inst(op, pc, pcpos))
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
        lower_inst_helper(*this, reinterpret_cast<const void*>(&execute_setupval), pcpos);
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
        lower_inst_helper(*this, reinterpret_cast<const void*>(&execute_namecall), pcpos);
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
        lower_inst_helper(*this, reinterpret_cast<const void*>(&execute_newtable), pcpos);
        return true;
    case LOP_DUPTABLE:
        lower_inst_helper(*this, reinterpret_cast<const void*>(&execute_duptable), pcpos);
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

    DeoptPoint deopt = build_deopt_point(pcpos);

    llvm::Value* base = load_base();

    for (const DeoptValue& value : deopt.values)
    {
        LUAU_ASSERT(value.kind == DeoptValue::Kind::Register);

        m_types.copy_tvalue(m_builder, m_types.stack_slot(m_builder, base, value.reg), reg_slot(value.reg));
    }

    llvm::Value* ci = m_types.load_state_ci(m_builder, m_state);
    m_types.store_ci_savedpc(m_builder, ci, const_ptr(&proto->code[pcpos]));
    m_builder.CreateRet(m_builder.getInt32(1));

    m_nproto.deopts.push_back(std::move(deopt));
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

    build.CreateCondBr(fast, fastBlock, slowBlock);

    build.SetInsertPoint(fastBlock);
    lowerFast();
    if (!unit.current_block()->hasTerminator())
        build.CreateBr(next);

    build.SetInsertPoint(slowBlock);
    unit.spill(pcpos);
    unit.call_helper(helper, pcpos, false);
    unit.reload();
    build.CreateBr(next);
}

void lower_inst_loadnil(TranslationUnit& unit, const Instruction* pc, int pcpos)
{
    unit.store_tag(LUAU_INSN_A(*pc), LUA_TNIL);
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

// Looks up 'key' in the node that the cached slot in the instruction's C points to, like the interpreter's predictive
// lookups; returns the condition for a hit and the address of the node's value
static std::pair<llvm::Value*, llvm::Value*> lookup_cached_slot(TranslationUnit& unit, llvm::Value* table, const int pcpos, const TString* key)
{
    llvm::IRBuilder<>& build = unit.builder();
    const VmTypes& types = unit.types();

    // C gets patched by the helpers whenever the cached slot misses
    llvm::Value* insn = build.CreateLoad(types.i32Ty, unit.insn_ptr(pcpos), "insn");
    llvm::Value* cachedSlot = build.CreateLShr(insn, 24);

    llvm::Value* nodemask = build.CreateZExt(unit.load_field(types.i8Ty, table, offsetof(LuaTable, nodemask8)), types.i32Ty);
    llvm::Value* slot = build.CreateZExt(build.CreateAnd(cachedSlot, nodemask), types.i64Ty);

    llvm::Value* nodes = unit.load_field(types.ptrTy, table, offsetof(LuaTable, node), "node");
    llvm::Value* node = build.CreateInBoundsGEP(types.i8Ty, nodes, build.CreateMul(slot, build.getInt64(sizeof(LuaNode))), "n");

    llvm::Value* keyTag = build.CreateAnd(unit.load_field(types.i32Ty, node, kNodeKeyTagOffset), 0xf);
    llvm::Value* keyValue = unit.load_field(types.ptrTy, node, offsetof(LuaNode, key) + offsetof(TKey, value));
    llvm::Value* valueTag = unit.load_field(types.i32Ty, node, offsetof(LuaNode, val) + offsetof(TValue, tt));

    llvm::Value* hit = build.CreateAnd(
        build.CreateAnd(build.CreateICmpEQ(keyTag, build.getInt32(LUA_TSTRING)), build.CreateICmpEQ(keyValue, unit.const_ptr(key))),
        build.CreateICmpNE(valueTag, build.getInt32(LUA_TNIL)),
        "hit"
    );

    return {hit, types.field_ptr(build, node, offsetof(LuaNode, val))};
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

    // the environment can stop being safe through getfenv/setfenv after the code was generated
    llvm::Value* env = unit.load_field(types.ptrTy, unit.closure(), offsetof(Closure, env), "env");
    llvm::Value* safeenv = unit.load_field(types.i8Ty, env, offsetof(LuaTable, safeenv), "safeenv");

    lower_with_fallback(
        unit,
        build.CreateICmpNE(safeenv, build.getInt8(0)),
        helper,
        pcpos,
        next_pc(pc, pcpos),
        [&]
        {
            unit.store_reg(ra, unit.constant(kidx));
        }
    );
}

void lower_inst_gettableks(TranslationUnit& unit, const Instruction* pc, int pcpos)
{
    llvm::IRBuilder<>& build = unit.builder();
    const VmTypes& types = unit.types();

    const int ra = LUAU_INSN_A(*pc);
    const int rb = LUAU_INSN_B(*pc);
    const int nextpc = next_pc(pc, pcpos);

    const void* helper = reinterpret_cast<const void*>(&execute_gettableks);

    llvm::BasicBlock* isTable = unit.new_block("table");
    llvm::BasicBlock* slow = unit.new_block("slow");

    build.CreateCondBr(tag_is(unit, rb, LUA_TTABLE), isTable, slow);

    build.SetInsertPoint(isTable);
    auto [hit, value] = lookup_cached_slot(unit, unit.load_pointer(rb), pcpos, string_constant(unit, static_cast<int>(pc[1])));

    llvm::BasicBlock* fast = unit.new_block("fast");
    build.CreateCondBr(hit, fast, slow);

    build.SetInsertPoint(fast);
    unit.store_reg(ra, build.CreateLoad(types.tvalueTy, value));
    build.CreateBr(unit.block_at(nextpc));

    build.SetInsertPoint(slow);
    unit.spill(pcpos);
    unit.call_helper(helper, pcpos, false);
    unit.reload();
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

    build.CreateCondBr(tag_is(unit, rb, LUA_TTABLE), isTable, slow);

    build.SetInsertPoint(isTable);
    llvm::Value* table = unit.load_pointer(rb);
    auto [hit, value] = lookup_cached_slot(unit, table, pcpos, string_constant(unit, static_cast<int>(pc[1])));
    llvm::Value* readonly = unit.load_field(types.i8Ty, table, offsetof(LuaTable, readonly));

    llvm::BasicBlock* fast = unit.new_block("fast");
    build.CreateCondBr(build.CreateAnd(hit, build.CreateICmpEQ(readonly, build.getInt8(0))), fast, slow);

    build.SetInsertPoint(fast);
    build.CreateStore(unit.load_reg(ra), value);
    unit.emit_table_barrier(table, ra);
    build.CreateBr(unit.block_at(nextpc));

    build.SetInsertPoint(slow);
    unit.spill(pcpos);
    unit.call_helper(helper, pcpos, false);
    unit.reload();
    build.CreateBr(unit.block_at(nextpc));
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

    build.CreateCondBr(isCandidate, candidate, slow);

    build.SetInsertPoint(candidate);
    llvm::Value* table = unit.load_pointer(rb);
    llvm::Value* sizearray = unit.load_field(types.i32Ty, table, offsetof(LuaTable, sizearray), "sizearray");
    llvm::Value* metatable = unit.load_field(types.ptrTy, table, offsetof(LuaTable, metatable), "metatable");

    llvm::Value* fastCond = build.CreateAnd(build.CreateICmpULT(index, sizearray), build.CreateIsNull(metatable));

    if (store)
        fastCond = build.CreateAnd(fastCond, build.CreateICmpEQ(unit.load_field(types.i8Ty, table, offsetof(LuaTable, readonly)), build.getInt8(0)));

    llvm::BasicBlock* fast = unit.new_block("array");
    build.CreateCondBr(fastCond, fast, slow);

    build.SetInsertPoint(fast);
    llvm::Value* array = unit.load_field(types.ptrTy, table, offsetof(LuaTable, array), "array");
    llvm::Value* element = build.CreateInBoundsGEP(types.tvalueTy, array, build.CreateZExt(index, types.i64Ty), "element");
    access(table, element);
    build.CreateBr(unit.block_at(nextpc));

    build.SetInsertPoint(slow);
    unit.spill(pcpos);
    unit.call_helper(helper, pcpos, false);
    unit.reload();
    build.CreateBr(unit.block_at(nextpc));
}

// Zero-based array index for a number key, or an index that's always out of bounds when the key isn't an integer
static llvm::Value* array_index(TranslationUnit& unit, const int reg)
{
    llvm::IRBuilder<>& build = unit.builder();
    const VmTypes& types = unit.types();

    llvm::Value* key = unit.load_number(reg);

    // saturating conversion: unlike fptosi, it is defined for every input
    llvm::Value* index = build.CreateIntrinsic(llvm::Intrinsic::fptosi_sat, {types.i32Ty, types.doubleTy}, {key});
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
    build.CreateCondBr(tag_is(unit, rb, LUA_TTABLE), checkKey, slow);
    build.SetInsertPoint(checkKey);
    build.CreateCondBr(tag_is(unit, rc, LUA_TNUMBER), candidate, slow);

    build.SetInsertPoint(slow);
    const int nextpc = next_pc(pc, pcpos);
    unit.spill(pcpos);
    unit.call_helper(reinterpret_cast<const void*>(&execute_gettable), pcpos, false);
    unit.reload();
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

    build.CreateCondBr(tag_is(unit, rb, LUA_TTABLE), checkKey, slow);
    build.SetInsertPoint(checkKey);
    build.CreateCondBr(tag_is(unit, rc, LUA_TNUMBER), candidate, slow);

    build.SetInsertPoint(slow);
    const int nextpc = next_pc(pc, pcpos);
    unit.spill(pcpos);
    unit.call_helper(reinterpret_cast<const void*>(&execute_settable), pcpos, false);
    unit.reload();
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

void lower_inst_call(TranslationUnit& unit, const Instruction* pc, int pcpos)
{
    llvm::IRBuilder<>& build = unit.builder();

    const int nextpc = next_pc(pc, pcpos);

    unit.emit_safepoint(pcpos, false);

    unit.spill(pcpos);
    llvm::Value* result = unit.call_helper(reinterpret_cast<const void*>(&execute_call), pcpos, true);

    llvm::BasicBlock* done = unit.new_block("call.done");
    llvm::BasicBlock* enter = unit.new_block("call.enter");

    llvm::SwitchInst* dispatch = build.CreateSwitch(result, done, 2);
    dispatch->addCase(build.getInt32(kCallEnter), enter);
    dispatch->addCase(build.getInt32(kCallYield), unit.yield_exit());

    build.SetInsertPoint(enter);
    unit.emit_resume_ci();

    build.SetInsertPoint(done);
    unit.reload();
    build.CreateBr(unit.block_at(nextpc));

    // a Lua callee returns, and a yielded C function resumes, into the instruction that follows the call
    unit.mark_entry_point(nextpc);
}

void lower_inst_return(TranslationUnit& unit, const Instruction* pc, int pcpos)
{
    llvm::IRBuilder<>& build = unit.builder();

    unit.emit_safepoint(pcpos, false);

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

    const int ra = LUAU_INSN_A(*pc);
    const int rb = static_cast<int>(pc[1]);

    const bool negated = LUAU_INSN_OP(*pc) == LOP_JUMPIFNOTEQ;
    const int target = getJumpTarget(*pc, pcpos);
    const int next = next_pc(pc, pcpos);

    llvm::Value* tagA = unit.load_tag(ra);
    llvm::Value* tagB = unit.load_tag(rb);

    llvm::BasicBlock* sameType = unit.new_block("eq.same");
    llvm::BasicBlock* number = unit.new_block("eq.number");
    llvm::BasicBlock* slow = unit.new_block("eq.slow");
    llvm::BasicBlock* different = unit.new_block("eq.different");

    build.CreateCondBr(build.CreateICmpEQ(tagA, tagB), sameType, different);

    // values of different types are never equal, and no metamethod gets involved
    build.SetInsertPoint(different);
    branch_on_relation(unit, build.getFalse(), negated, target, next);

    build.SetInsertPoint(sameType);
    build.CreateCondBr(build.CreateICmpEQ(tagA, build.getInt32(LUA_TNUMBER)), number, slow);

    build.SetInsertPoint(number);
    branch_on_relation(unit, build.CreateFCmpOEQ(unit.load_number(ra), unit.load_number(rb)), negated, target, next);

    build.SetInsertPoint(slow);
    unit.spill(pcpos);
    llvm::Value* equal = unit.call_helper(reinterpret_cast<const void*>(&execute_equal), pcpos, true);
    unit.reload();
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
    llvm::BasicBlock* slow = unit.new_block("cmp.slow");

    build.CreateCondBr(build.CreateAnd(tag_is(unit, ra, LUA_TNUMBER), tag_is(unit, rb, LUA_TNUMBER)), number, slow);

    // ordered comparisons are false for NaN, the NOT variants then take the jump just like the interpreter does
    build.SetInsertPoint(number);
    llvm::Value* a = unit.load_number(ra);
    llvm::Value* b = unit.load_number(rb);
    branch_on_relation(unit, lessEqual ? build.CreateFCmpOLE(a, b) : build.CreateFCmpOLT(a, b), negated, target, next);

    build.SetInsertPoint(slow);
    unit.spill(pcpos);
    const void* helper = lessEqual ? reinterpret_cast<const void*>(&execute_lessequal) : reinterpret_cast<const void*>(&execute_lessthan);
    llvm::Value* holds = unit.call_helper(helper, pcpos, true);
    unit.reload();
    branch_on_relation(unit, build.CreateICmpNE(holds, build.getInt32(0)), negated, target, next);
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

    llvm::Value* fast = build.getTrue();

    if (regB >= 0)
        fast = build.CreateAnd(fast, tag_is(unit, regB, LUA_TNUMBER));
    if (regC >= 0 && regC != regB)
        fast = build.CreateAnd(fast, tag_is(unit, regC, LUA_TNUMBER));

    lower_with_fallback(
        unit,
        fast,
        reinterpret_cast<const void*>(&execute_arith),
        pcpos,
        next_pc(pc, pcpos),
        [&]
        {
            llvm::Value* a = kb ? llvm::ConstantFP::get(types.doubleTy, nvalue(kb)) : unit.load_number(regB);
            llvm::Value* b = kc ? llvm::ConstantFP::get(types.doubleTy, nvalue(kc)) : unit.load_number(regC);

            unit.store_number(ra, number_arith(unit, op, a, b, kc));
        }
    );
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
    build.CreateCondBr(build.CreateIsNull(metatable), table, slow);

    // luaH_getn neither allocates nor reads the stack
    build.SetInsertPoint(table);
    llvm::Value* length = unit.call_native(reinterpret_cast<const void*>(&luaH_getn), types.i32Ty, {h});
    unit.store_number(ra, build.CreateSIToFP(length, types.doubleTy));
    build.CreateBr(unit.block_at(nextpc));

    build.SetInsertPoint(notTable);
    build.CreateCondBr(build.CreateICmpEQ(tag, build.getInt32(LUA_TSTRING)), string, slow);

    build.SetInsertPoint(string);
    llvm::Value* len = unit.load_field(types.i32Ty, unit.load_pointer(rb), offsetof(TString, len));
    unit.store_number(ra, build.CreateUIToFP(len, types.doubleTy));
    build.CreateBr(unit.block_at(nextpc));

    build.SetInsertPoint(slow);
    unit.spill(pcpos);
    unit.call_helper(reinterpret_cast<const void*>(&execute_length), pcpos, false);
    unit.reload();
    build.CreateBr(unit.block_at(nextpc));
}

// step > 0 ? idx <= limit : limit <= idx, the condition FORNPREP and FORNLOOP share
static llvm::Value* forn_condition(TranslationUnit& unit, llvm::Value* limit, llvm::Value* step, llvm::Value* idx)
{
    llvm::IRBuilder<>& build = unit.builder();

    llvm::Value* positive = build.CreateFCmpOGT(step, llvm::ConstantFP::get(unit.types().doubleTy, 0.0));

    return build.CreateSelect(positive, build.CreateFCmpOLE(idx, limit), build.CreateFCmpOLE(limit, idx));
}

void lower_inst_fornprep(TranslationUnit& unit, const Instruction* pc, int pcpos)
{
    llvm::IRBuilder<>& build = unit.builder();

    const int ra = LUAU_INSN_A(*pc);
    const int target = getJumpTarget(*pc, pcpos);
    const int next = next_pc(pc, pcpos);

    llvm::Value* numbers =
        build.CreateAnd(build.CreateAnd(tag_is(unit, ra, LUA_TNUMBER), tag_is(unit, ra + 1, LUA_TNUMBER)), tag_is(unit, ra + 2, LUA_TNUMBER));

    llvm::BasicBlock* fast = unit.new_block("forn.fast");
    llvm::BasicBlock* slow = unit.new_block("forn.slow");

    build.CreateCondBr(numbers, fast, slow);

    build.SetInsertPoint(fast);
    llvm::Value* enter = forn_condition(unit, unit.load_number(ra), unit.load_number(ra + 1), unit.load_number(ra + 2));
    build.CreateCondBr(enter, unit.block_at(next), unit.block_at(target));

    // non-number loop parameters get converted in place or raise an error
    build.SetInsertPoint(slow);
    unit.spill(pcpos);
    llvm::Value* skip = unit.call_helper(reinterpret_cast<const void*>(&execute_fornprep), pcpos, true);
    unit.reload();
    build.CreateCondBr(build.CreateICmpNE(skip, build.getInt32(0)), unit.block_at(target), unit.block_at(next));
}

void lower_inst_fornloop(TranslationUnit& unit, const Instruction* pc, int pcpos)
{
    llvm::IRBuilder<>& build = unit.builder();

    const int ra = LUAU_INSN_A(*pc);

    unit.emit_safepoint(pcpos, true);

    // FORNPREP has made sure all three are numbers
    llvm::Value* limit = unit.load_number(ra);
    llvm::Value* step = unit.load_number(ra + 1);
    llvm::Value* idx = build.CreateFAdd(unit.load_number(ra + 2), step);

    unit.store_number(ra + 2, idx);

    build.CreateCondBr(forn_condition(unit, limit, step, idx), unit.block_at(getJumpTarget(*pc, pcpos)), unit.block_at(next_pc(pc, pcpos)));
}

void lower_inst_forgprep(TranslationUnit& unit, const Instruction* pc, int pcpos)
{
    const int target = getJumpTarget(*pc, pcpos);

    unit.spill(pcpos);
    unit.call_helper(reinterpret_cast<const void*>(&execute_forgprep), pcpos, false);
    unit.reload();
    unit.builder().CreateBr(unit.block_at(target));
}

void lower_inst_forgloop(TranslationUnit& unit, const Instruction* pc, int pcpos)
{
    llvm::IRBuilder<>& build = unit.builder();

    const int target = getJumpTarget(*pc, pcpos);
    const int next = next_pc(pc, pcpos);

    unit.emit_safepoint(pcpos, true);

    unit.spill(pcpos);
    llvm::Value* result = unit.call_helper(reinterpret_cast<const void*>(&execute_forgloop), pcpos, true);

    llvm::BasicBlock* exit = unit.new_block("forg.exit");
    llvm::BasicBlock* loop = unit.new_block("forg.loop");

    llvm::SwitchInst* dispatch = build.CreateSwitch(result, exit, 2);
    dispatch->addCase(build.getInt32(kForgloopContinue), loop);
    // luau_finishop completes the instruction on resume and continues at either of the targets, which start blocks
    dispatch->addCase(build.getInt32(kForgloopYield), unit.yield_exit());

    build.SetInsertPoint(loop);
    unit.reload();
    build.CreateBr(unit.block_at(target));

    build.SetInsertPoint(exit);
    unit.reload();
    build.CreateBr(unit.block_at(next));
}

void lower_inst_fastcall(TranslationUnit& unit, const Instruction* pc, int pcpos)
{
    llvm::IRBuilder<>& build = unit.builder();

    const int target = getJumpTarget(*pc, pcpos);
    const int next = next_pc(pc, pcpos);

    unit.spill(pcpos);
    llvm::Value* handled = unit.call_helper(reinterpret_cast<const void*>(&execute_fastcall), pcpos, true);
    unit.reload();

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
    unit.reload();
    build.CreateBr(unit.block_at(next));

    build.SetInsertPoint(enter);
    unit.emit_resume_ci();

    build.SetInsertPoint(done);
    unit.reload();
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
    unit.reload();
    build.CreateBr(unit.block_at(next));

    build.SetInsertPoint(done);
    unit.reload();
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
    unit.reload();

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
    unit.reload();
}

} // namespace Luau::Jit
