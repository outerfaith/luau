// This file is part of the Luau programming language and is licensed under MIT License; see LICENSE.txt for details
#pragma once

#include "BytecodeCfg.h"
#include "HostHooks.h"
#include "JitProfiling.h"
#include "NativeProto.h"
#include "VmTypes.h"

#include "Luau/Bytecode.h"

#include "lobject.h"

#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/IR/IRBuilder.h>

#include <bitset>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Luau::Jit
{

struct BytecodeFunction;

// A numeric loop with a constant integer step whose index the speculating upper tiers count in an int32 alongside the
// index register: FORNPREP checks that the start and limit allow it, and the register holds the counter converted to a
// number whenever anything else can see it, so the loop body indexes arrays without converting back and forth
struct IntegerLoop
{
    int prepPc = -1;
    int loopPc = -1;
    // the index register, A + 2 of FORNPREP and FORNLOOP
    int reg = -1;
    int step = 0;
    llvm::AllocaInst* counter = nullptr;
};

// VM registers live in allocas while generated code runs, and the VM stack is only brought up to date around calls
// into the VM. Before anything that can run Lua code, collect garbage or inspect the frame, every register that may
// have been written since the last call is spilled to the stack, and afterwards the ones the call might have changed
// are reloaded. Within a bytecode block this keeps the stack as the interpreter would have it. A register that is dead
// when a block starts isn't spilled anymore though, so like with CodeGen it can keep an older value, which only lets an
// object that weak tables observe live a little longer; it is GC safe because the collector has been marking that
// value all along and leaves every stack slot either marked or nil. Reloads of registers nobody reads again are dead
// loads that the optimizer removes
class TranslationUnit final
{
public:
    TranslationUnit(
        NativeProto& nproto,
        const BytecodeFunction& function,
        const HostHooks& hostHooks,
        const VmTypes& types,
        llvm::Module& module,
        llvm::StringRef name,
        bool speculate,
        JitCounters* counters
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
    llvm::Value* load_vector(int reg);
    void store_tag(int reg, int tag);
    void store_nil(int reg);
    void store_object(int reg, llvm::Value* object, int tag);
    void store_lightuserdata(int reg, llvm::Value* pointer, int utag);
    void store_number(int reg, llvm::Value* value);
    void store_boolean(int reg, llvm::Value* value);
    void store_vector(int reg, llvm::Value* value);

    // l_isfalse
    llvm::Value* is_falsy(int reg);

    llvm::Constant* constant(int kidx) const;
    llvm::Constant* const_ptr(const void* value) const;
    llvm::Constant* insn_ptr(int pcpos) const;

    llvm::Value* load_field(llvm::Type* type, llvm::Value* object, size_t offset, const llvm::Twine& name = "");
    void store_field(llvm::Value* value, llvm::Value* object, size_t offset);
    // A load of a VM object field that nothing but VM functions writes, which tells LLVM that stores of generated code
    // to buffer data leave it alone; every untagged access still aliases it
    llvm::Value* load_header(llvm::Type* type, llvm::Value* object, size_t offset, const llvm::Twine& name = "");
    // tags a load or store of buffer data
    void mark_buffer_access(llvm::Instruction* access);
    // whether the closure's environment is still safe, for the fast paths of builtins and imports
    llvm::Value* is_safe_env();

    // Calls a VmHelpers.h function with (L, &code[pcpos]); spilling and reloading is up to the caller
    llvm::Value* call_helper(const void* helper, int pcpos, bool returnsInt);
    llvm::Value* call_native(const void* function, llvm::Type* result, llvm::ArrayRef<llvm::Value*> args);

    void spill(int pcpos);
    // brings back the registers that the helper for the instruction at 'pcpos' might have changed
    void reload(int pcpos);

    // Runs the interrupt (and the GC step on back edges) the interpreter runs at the start of the instruction
    void emit_safepoint(int pcpos, bool backedge);
    // luaC_checkGC after the instruction allocated the object in 'reg', which so far is only in its register slot
    void emit_gc_check(int pcpos, int reg);
    // Continues in the frame at L->ci, with a tail call when it has native code; terminates the current block
    void emit_resume_ci();
    // tail calls generated code at its first instruction, for a frame that was just set up
    void emit_enter_native(llvm::Value* target);
    void emit_table_barrier(llvm::Value* table, int valueReg);
    // luaC_barrier with the given slow path, (lua_State*, GCObject* object, GCObject* value)
    void emit_barrier(llvm::Value* object, int valueReg, const void* barrier);

    // whether a check that 'type' (from the bytecode types) is 'expected' may leave generated code when it fails
    bool speculates_on(uint8_t type, uint8_t expected) const
    {
        return m_speculate && type == expected;
    }

    bool profiling() const
    {
        return m_profiling;
    }

    // branch weights for a condition that is true on the fast path, and for one that is true on a cold path
    llvm::MDNode* likely();
    llvm::MDNode* unlikely();

    llvm::BasicBlock* new_block(const llvm::Twine& name);
    llvm::BasicBlock* block_at(int pcpos);
    // the rest of the bytecode block continues in its generic copy from here, like it does after a helper call, so a
    // rare path doesn't merge into the fast copy
    void mark_slow_path();
    // whether reg holds a table that NEWTABLE or DUPTABLE created earlier in the bytecode block of pcpos
    bool holds_new_table(int reg, int pcpos) const;

    void mark_entry_point(int pcpos);

    llvm::BasicBlock* vm_exit(int pcpos);
    llvm::BasicBlock* yield_exit();

    // ors 'bits' into the feedback of the instruction
    void emit_feedback_bits(int pcpos, llvm::Value* bits);
    // the integer loop that the FORNPREP or FORNLOOP at pcpos belongs to, or null
    const IntegerLoop* integer_loop(int pcpos) const;
    // the feedback that the code this compilation replaces gathered for the instruction
    uint32_t feedback_at(int pcpos) const;

private:
    void emit_prologue();
    void emit_dispatch();

    void translate_block(uint32_t blockIdx);
    void translate_range(int startpc, int finishpc, bool generic);

    llvm::BasicBlock* fast_block_at(int pcpos);
    llvm::BasicBlock* generic_block_at(int pcpos);
    // whether the code being emitted comes after a helper call, rather than out of a fast path
    bool in_slow_path() const;
    bool translate_inst(LuauOpcode op, const Instruction* pc, int pcpos);

    void reload_regs(const std::bitset<256>& regs);
    void clear_extra(int reg);

    void emit_vm_exit(int pcpos, llvm::BasicBlock* block);
    void emit_counter(uint64_t* counter);
    // records the tags of the operands that feedback_operands lists for the instruction
    void emit_type_feedback(int pcpos);
    void emit_operand_feedback(int pcpos, int slot);
    // every way into the loop at 'header' but the back edge of its fast copy goes through this block: it checks the tags
    // that the first block of the loop guards with exits, so that the loop knows them from its first instruction on
    llvm::BasicBlock* loop_pad(int header);
    // counts down the tier-up budget at the start of the function or at a back edge, and switches the frame over to the
    // replacement code at the same pc once there is one
    void emit_tier_up(int pcpos);

    void compute_dirty_regs();
    void find_integer_loops();
    // rebuilds the counters of the integer loops around an entry point from their index registers
    void emit_integer_loop_entry(int pcpos);
    DeoptPoint build_deopt_point(int pcpos) const;

    NativeProto& m_nproto;
    const BytecodeFunction& m_bytecode;

    const HostHooks& m_hostHooks;

    const VmTypes& m_types;

    llvm::Module& m_module;
    JitCounters* m_counters = nullptr;
    bool m_speculate = false;
    // baseline code records operand types for the tiers above it
    bool m_profiling = false;
    // code below the top tier counts down to the next one
    bool m_tiersUp = false;
    llvm::IRBuilder<> m_builder;
    llvm::Function* m_function = nullptr;
    llvm::BasicBlock* m_entryBlock = nullptr;
    llvm::BasicBlock* m_bailBlock = nullptr;
    llvm::BasicBlock* m_yieldBlock = nullptr;

    llvm::Value* m_state = nullptr;
    llvm::Value* m_entrypc = nullptr;
    llvm::Value* m_baseSlot = nullptr;
    llvm::Value* m_closure = nullptr;
    llvm::MDNode* m_likely = nullptr;
    llvm::MDNode* m_unlikely = nullptr;
    llvm::MDNode* m_tbaaHeader = nullptr;
    llvm::MDNode* m_tbaaBuffer = nullptr;

    std::vector<llvm::Value*> m_regSlots;
    std::vector<llvm::BasicBlock*> m_blocks;
    std::vector<llvm::BasicBlock*> m_genericBlocks;
    std::vector<int> m_entryPoints;
    llvm::SmallPtrSet<llvm::BasicBlock*, 32> m_slowBlocks;

    // the bytecode block being translated, and whether this is its generic copy
    int m_blockStart = -1;
    int m_blockFinish = -1;
    bool m_generic = false;
    std::unordered_map<int, llvm::BasicBlock*> m_exitBlocks;
    // registers that may differ from their stack slot when the instruction starts
    std::vector<std::bitset<256>> m_dirtyIn;
    std::vector<std::bitset<256>> m_liveIn;

    std::vector<IntegerLoop> m_integerLoops;
    // the targets of backward jumps, and their pads (null when the loop guards nothing)
    std::vector<bool> m_loopHeaders;
    std::unordered_map<int, llvm::BasicBlock*> m_loopPads;
    // the instruction being translated, -1 outside of one
    int m_pcpos = -1;
};

struct TypedOperand
{
    int reg = -1;
    uint8_t type = LBC_TYPE_ANY;
    int tag = LUA_TNUMBER;
};

// Branches to 'fast' when every operand holds its tag, and to 'slow' (or a VM exit, when the bytecode types are
// speculated on) when one doesn't; negative registers are skipped
void check_operand_tags(TranslationUnit& unit, int pcpos, llvm::ArrayRef<TypedOperand> operands, llvm::BasicBlock* fast, llvm::BasicBlock* slow);

// Each lowering leaves the builder in a block that falls through to the next instruction, or terminates it
void lower_inst_loadnil(TranslationUnit& unit, const Instruction* pc, int pcpos);
void lower_inst_loadb(TranslationUnit& unit, const Instruction* pc, int pcpos);
void lower_inst_loadn(TranslationUnit& unit, const Instruction* pc, int pcpos);
void lower_inst_loadk(TranslationUnit& unit, const Instruction* pc, int pcpos);
void lower_inst_move(TranslationUnit& unit, const Instruction* pc, int pcpos);
void lower_inst_getglobal(TranslationUnit& unit, const Instruction* pc, int pcpos);
void lower_inst_getupval(TranslationUnit& unit, const Instruction* pc, int pcpos);
void lower_inst_setupval(TranslationUnit& unit, const Instruction* pc, int pcpos);
void lower_inst_getimport(TranslationUnit& unit, const Instruction* pc, int pcpos);
void lower_inst_gettableks(TranslationUnit& unit, const Instruction* pc, int pcpos);
void lower_inst_namecall(TranslationUnit& unit, const Instruction* pc, int pcpos);
void lower_inst_settableks(TranslationUnit& unit, const Instruction* pc, int pcpos);
void lower_inst_gettable(TranslationUnit& unit, const Instruction* pc, int pcpos);
void lower_inst_settable(TranslationUnit& unit, const Instruction* pc, int pcpos);
void lower_inst_gettablen(TranslationUnit& unit, const Instruction* pc, int pcpos);
void lower_inst_settablen(TranslationUnit& unit, const Instruction* pc, int pcpos);
void lower_inst_newtable(TranslationUnit& unit, const Instruction* pc, int pcpos);
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
