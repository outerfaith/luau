// This file is part of the Luau programming language and is licensed under MIT License; see LICENSE.txt for details
#pragma once

#include "HostHooks.h"
#include "NativeProto.h"
#include "VmTypes.h"

#include "Luau/Bytecode.h"

#include "lobject.h"
#include "ltm.h"

#include <llvm/IR/IRBuilder.h>

#include <unordered_map>
#include <utility>
#include <vector>

namespace Luau::Jit
{

struct BytecodeFunction;

class TranslationUnit final
{
public:
    TranslationUnit(
        NativeProto& nproto,
        const BytecodeFunction& function,
        const HostHooks& hostHooks,
        const VmTypes& types,
        llvm::Module& module,
        llvm::StringRef name
    );

    bool translate();

    llvm::Function* native_function() const
    {
        return m_function;
    }

    const BytecodeFunction& bytecode() const
    {
        return m_bytecode;
    }

    const HostHooks& host_hooks() const
    {
        return m_hostHooks;
    }

    const VmTypes& types() const
    {
        return m_types;
    }

    llvm::IRBuilder<>& builder()
    {
        return m_builder;
    }

    llvm::BasicBlock* current_block() const
    {
        return m_builder.GetInsertBlock();
    }

    llvm::Value* state() const
    {
        return m_state;
    }

    llvm::Value* load_base(const llvm::Twine& name = "base");
    llvm::Value* stack_slot(int reg, const llvm::Twine& name = "");
    llvm::Value* reg_slot(int reg) const;

    llvm::Value* load_reg(int reg, const llvm::Twine& name = "");
    void store_reg(int reg, llvm::Value* value);
    llvm::BasicBlock* block_at(int pcpos);

    void mark_entry_point(int pcpos);

    llvm::BasicBlock* vm_exit(int pcpos);

private:
    void emit_prologue();
    void emit_dispatch();

    void translate_block(uint32_t blockIdx);
    bool translate_inst(LuauOpcode op, const Instruction* pc, int pcpos);

    void emit_vm_exit(int pcpos, llvm::BasicBlock* block);

    DeoptPoint build_deopt_point(int pcpos) const;
    llvm::Constant* const_ptr(const void* value) const;

    NativeProto& m_nproto;
    const BytecodeFunction& m_bytecode;

    const HostHooks& m_hostHooks;

    const VmTypes& m_types;

    llvm::Module& m_module;
    llvm::IRBuilder<> m_builder;
    llvm::Function* m_function = nullptr;
    llvm::BasicBlock* m_entryBlock = nullptr;
    llvm::BasicBlock* m_bailBlock = nullptr;

    llvm::Value* m_state = nullptr;
    llvm::Value* m_entrypc = nullptr;
    llvm::Value* m_baseSlot = nullptr;

    std::vector<llvm::Value*> m_regSlots;
    std::vector<llvm::BasicBlock*> m_blocks;
    std::vector<std::pair<int, llvm::BasicBlock*>> m_entryPoints;
    std::unordered_map<int, llvm::BasicBlock*> m_exitBlocks;
};

void lower_inst_loadnil(TranslationUnit& unit, const Instruction* pc);
void lower_inst_loadb(TranslationUnit& unit, const Instruction* pc, int pcpos);
void lower_inst_loadn(TranslationUnit& unit, const Instruction* pc);
void lower_inst_loadk(TranslationUnit& unit, const Instruction* pc);
void lower_inst_move(TranslationUnit& unit, const Instruction* pc);
void lower_inst_binary(TranslationUnit& unit, const Instruction* pc, int pcpos, TMS tm);

} // namespace Luau::Jit
