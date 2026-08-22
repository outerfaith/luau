// This file is part of the Luau programming language and is licensed under MIT License; see LICENSE.txt for details
#include "BytecodeTranslation.h"

#include "BytecodeAnalysis.h"

#include "Luau/BytecodeUtils.h"
#include "Luau/Common.h"

#include "lstate.h"

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>

#include <algorithm>

namespace Luau::Jit
{

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

    llvm::LLVMContext& context = m_module.getContext();

    m_entryBlock = llvm::BasicBlock::Create(context, "entry", m_function);
    m_bailBlock = llvm::BasicBlock::Create(context, "bail", m_function);

    emit_prologue();

    m_builder.SetInsertPoint(m_bailBlock);
    m_builder.CreateRet(m_builder.getInt32(1));

    for (uint32_t blockIdx = 0; blockIdx < static_cast<uint32_t>(m_bytecode.bcBlocks.size()); blockIdx++)
        translate_block(blockIdx);

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

    for (int i = 0; i < proto->maxstacksize; i++)
        m_types.copy_tvalue(m_builder, m_regSlots[i], m_types.stack_slot(m_builder, base, i));
}

void TranslationUnit::emit_dispatch()
{
    LUAU_ASSERT(m_entryBlock && !m_entryBlock->getTerminator());
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

        if (llvm::BasicBlock* previous = current_block(); previous && !previous->getTerminator())
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

    if (llvm::BasicBlock* previous = current_block(); previous && !previous->getTerminator())
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
        return true;
    case LOP_LOADNIL:
    {
        const int A = LUAU_INSN_A(*pc);

        llvm::Value* nil = llvm::ConstantStruct::get(
            m_types.tvalueTy,
            {llvm::ConstantInt::get(m_types.i64Ty, 0),
             llvm::ConstantAggregateZero::get(m_types.tvalueTy->getElementType(VmTypes::kTValueExtra)),
             llvm::ConstantInt::get(m_types.i32Ty, LUA_TNIL)}
        );

        store_reg(A, nil);
        return true;
    }
    case LOP_BREAK:
    default:
        // Lowering of the individual instructions is not implemented yet
        return false;
    }
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

    RegisterSet live = compute_live_in_at(m_bytecode, pcpos);
    live.regs |= m_bytecode.cfg.captured.regs;

    DeoptPoint result;
    result.pc = static_cast<uint32_t>(pcpos);

    for (int i = 0; i < proto->maxstacksize; i++)
    {
        if (!live.regs.test(i))
            continue;

        DeoptValue value;
        value.kind = DeoptValue::Kind::Register;
        value.reg = static_cast<uint8_t>(i);

        result.values.push_back(value);
    }

    return result;
}

llvm::Constant* TranslationUnit::const_ptr(const void* value) const
{
    llvm::LLVMContext& context = m_module.getContext();
    llvm::Type* intptrTy = m_module.getDataLayout().getIntPtrType(context);

    return llvm::ConstantExpr::getIntToPtr(llvm::ConstantInt::get(intptrTy, reinterpret_cast<uintptr_t>(value)), m_types.ptrTy);
}

} // namespace Luau::Jit
