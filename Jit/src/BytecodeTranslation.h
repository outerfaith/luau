// This file is part of the Luau programming language and is licensed under MIT License; see LICENSE.txt for details
#pragma once

#include "BytecodeCfg.h"
#include "HostHooks.h"
#include "NativeProto.h"
#include "VmTypes.h"

#include "Luau/Bytecode.h"

#include "lobject.h"

#include <llvm/IR/IRBuilder.h>

#include <bitset>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Luau::Jit
{

struct BytecodeFunction;

// VM registers live in allocas while generated code runs, and the VM stack is only brought up to date around calls
// into the VM. Before anything that can run Lua code, collect garbage or inspect the frame, every register that may
// have been written since the last call is spilled to the stack, and afterwards all of them are reloaded. This keeps
// the stack exactly as the interpreter would have it, which weak tables and finalizers can observe, and it is GC safe
// because the collector leaves every stack slot either marked or nil. Reloads of registers nobody reads again are
// dead loads that the optimizer removes
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

    llvm::Value* closure() const
    {
        return m_closure;
    }

    llvm::Value* load_base(const llvm::Twine& name = "base");
    llvm::Value* stack_slot(int reg, const llvm::Twine& name = "");
    llvm::Value* reg_slot(int reg) const;

    llvm::Value* load_reg(int reg, const llvm::Twine& name = "");
    void store_reg(int reg, llvm::Value* value);

    llvm::Value* load_tag(int reg);
    llvm::Value* load_number(int reg);
    llvm::Value* load_pointer(int reg);
    llvm::Value* load_boolean(int reg);
    void store_tag(int reg, int tag);
    void store_number(int reg, llvm::Value* value);
    void store_boolean(int reg, llvm::Value* value);

    // l_isfalse
    llvm::Value* is_falsy(int reg);

    llvm::Constant* constant(int kidx) const;
    llvm::Constant* const_ptr(const void* value) const;
    llvm::Constant* insn_ptr(int pcpos) const;

    llvm::Value* load_field(llvm::Type* type, llvm::Value* object, size_t offset, const llvm::Twine& name = "");
    void store_field(llvm::Value* value, llvm::Value* object, size_t offset);

    // Calls a VmHelpers.h function with (L, &code[pcpos]); spilling and reloading is up to the caller
    llvm::Value* call_helper(const void* helper, int pcpos, bool returnsInt);
    llvm::Value* call_native(const void* function, llvm::Type* result, llvm::ArrayRef<llvm::Value*> args);

    void spill(int pcpos);
    void reload();

    // Runs the interrupt (and the GC step on back edges) the interpreter runs at the start of the instruction
    void emit_safepoint(int pcpos, bool backedge);
    // Continues in the frame at L->ci, with a tail call when it has native code; terminates the current block
    void emit_resume_ci();
    void emit_table_barrier(llvm::Value* table, int valueReg);

    llvm::BasicBlock* new_block(const llvm::Twine& name);
    llvm::BasicBlock* block_at(int pcpos);

    void mark_entry_point(int pcpos);

    llvm::BasicBlock* vm_exit(int pcpos);
    llvm::BasicBlock* yield_exit();

private:
    void emit_prologue();
    void emit_dispatch();

    void translate_block(uint32_t blockIdx);
    bool translate_inst(LuauOpcode op, const Instruction* pc, int pcpos);

    void emit_vm_exit(int pcpos, llvm::BasicBlock* block);

    void compute_dirty_regs();
    DeoptPoint build_deopt_point(int pcpos) const;

    NativeProto& m_nproto;
    const BytecodeFunction& m_bytecode;

    const HostHooks& m_hostHooks;

    const VmTypes& m_types;

    llvm::Module& m_module;
    llvm::IRBuilder<> m_builder;
    llvm::Function* m_function = nullptr;
    llvm::BasicBlock* m_entryBlock = nullptr;
    llvm::BasicBlock* m_bailBlock = nullptr;
    llvm::BasicBlock* m_yieldBlock = nullptr;

    llvm::Value* m_state = nullptr;
    llvm::Value* m_entrypc = nullptr;
    llvm::Value* m_baseSlot = nullptr;
    llvm::Value* m_closure = nullptr;

    std::vector<llvm::Value*> m_regSlots;
    std::vector<llvm::BasicBlock*> m_blocks;
    std::vector<std::pair<int, llvm::BasicBlock*>> m_entryPoints;
    std::unordered_map<int, llvm::BasicBlock*> m_exitBlocks;
    // registers that may differ from their stack slot when the instruction starts
    std::vector<std::bitset<256>> m_dirtyIn;
};

// Each lowering leaves the builder in a block that falls through to the next instruction, or terminates it
void lower_inst_loadnil(TranslationUnit& unit, const Instruction* pc, int pcpos);
void lower_inst_loadb(TranslationUnit& unit, const Instruction* pc, int pcpos);
void lower_inst_loadn(TranslationUnit& unit, const Instruction* pc, int pcpos);
void lower_inst_loadk(TranslationUnit& unit, const Instruction* pc, int pcpos);
void lower_inst_move(TranslationUnit& unit, const Instruction* pc, int pcpos);
void lower_inst_getglobal(TranslationUnit& unit, const Instruction* pc, int pcpos);
void lower_inst_getupval(TranslationUnit& unit, const Instruction* pc, int pcpos);
void lower_inst_getimport(TranslationUnit& unit, const Instruction* pc, int pcpos);
void lower_inst_gettableks(TranslationUnit& unit, const Instruction* pc, int pcpos);
void lower_inst_settableks(TranslationUnit& unit, const Instruction* pc, int pcpos);
void lower_inst_gettable(TranslationUnit& unit, const Instruction* pc, int pcpos);
void lower_inst_settable(TranslationUnit& unit, const Instruction* pc, int pcpos);
void lower_inst_gettablen(TranslationUnit& unit, const Instruction* pc, int pcpos);
void lower_inst_settablen(TranslationUnit& unit, const Instruction* pc, int pcpos);
void lower_inst_call(TranslationUnit& unit, const Instruction* pc, int pcpos);
void lower_inst_return(TranslationUnit& unit, const Instruction* pc, int pcpos);
void lower_inst_jump(TranslationUnit& unit, const Instruction* pc, int pcpos);
void lower_inst_jumpif(TranslationUnit& unit, const Instruction* pc, int pcpos);
void lower_inst_jumpifeq(TranslationUnit& unit, const Instruction* pc, int pcpos);
void lower_inst_jumpifcmp(TranslationUnit& unit, const Instruction* pc, int pcpos);
void lower_inst_jumpxeqk(TranslationUnit& unit, const Instruction* pc, int pcpos);
void lower_inst_binary(TranslationUnit& unit, const Instruction* pc, int pcpos);
void lower_inst_logical(TranslationUnit& unit, const Instruction* pc, int pcpos);
void lower_inst_not(TranslationUnit& unit, const Instruction* pc, int pcpos);
void lower_inst_length(TranslationUnit& unit, const Instruction* pc, int pcpos);
void lower_inst_fornprep(TranslationUnit& unit, const Instruction* pc, int pcpos);
void lower_inst_fornloop(TranslationUnit& unit, const Instruction* pc, int pcpos);
void lower_inst_forgprep(TranslationUnit& unit, const Instruction* pc, int pcpos);
void lower_inst_forgloop(TranslationUnit& unit, const Instruction* pc, int pcpos);
void lower_inst_fastcall(TranslationUnit& unit, const Instruction* pc, int pcpos);
void lower_inst_fastpcall(TranslationUnit& unit, const Instruction* pc, int pcpos);
void lower_inst_namecalludata(TranslationUnit& unit, const Instruction* pc, int pcpos);
void lower_inst_finconstruct(TranslationUnit& unit, const Instruction* pc, int pcpos);
void lower_inst_cmpproto(TranslationUnit& unit, const Instruction* pc, int pcpos);
void lower_inst_coverage(TranslationUnit& unit, const Instruction* pc, int pcpos);
// Instructions without a fast path: the whole instruction is a helper call
void lower_inst_helper(TranslationUnit& unit, const void* helper, int pcpos);

} // namespace Luau::Jit
