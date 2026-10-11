// This file is part of the Luau programming language and is licensed under MIT License; see LICENSE.txt for details
#pragma once

#include "lobject.h"

// Generated code calls these to run the parts of an instruction that it doesn't lower inline
//
// Every helper takes the instruction it executes and reproduces the interpreter's behavior for it exactly, fast paths
// included, so generated code is free to call one without trying its own fast path first. Registers are read from
// and written to the VM stack, which means that the caller has to spill the registers the instruction reads and
// reload the ones it needs afterwards, and that 'L->base' may change across any call
namespace Luau::Jit
{

// What generated code does after a call-like instruction
enum CallResult : int
{
    // The call has completed and its results are in place
    kCallDone = 0,
    // A Lua frame has been set up in L->ci and has to be entered
    kCallEnter = 1,
    // A C function yielded; luau_execute has to exit
    kCallYield = 2,
};

enum FastpcallResult : int
{
    kFastpcallFallback = 0,
    kFastpcallEnter = 1,
    kFastpcallDone = 2,
    kFastpcallYield = 3,
};

enum NamecalludataResult : int
{
    // The userdata has no direct namecall handler; NAMECALL has been executed and the paired CALL runs next
    kNamecalludataFallback = 0,
    // The handler performed the paired CALL as well
    kNamecalludataDone = 1,
    kNamecalludataYield = 2,
};

enum ForgloopResult : int
{
    kForgloopExit = 0,
    kForgloopContinue = 1,
    kForgloopYield = 2,
};

// Return 1 if the thread was put into an error, yield or break state and luau_execute has to exit; the safepoint
// variant also runs the GC step back edges take
int execute_interrupt(lua_State* L, const Instruction* pc);
int execute_safepoint(lua_State* L, const Instruction* pc);

void execute_getglobal(lua_State* L, const Instruction* pc);
void execute_setglobal(lua_State* L, const Instruction* pc);
void execute_setupval(lua_State* L, const Instruction* pc);
void execute_closeupvals(lua_State* L, const Instruction* pc);
void execute_getimport(lua_State* L, const Instruction* pc);
void execute_gettableks(lua_State* L, const Instruction* pc);
void execute_settableks(lua_State* L, const Instruction* pc);
void execute_gettable(lua_State* L, const Instruction* pc);
void execute_settable(lua_State* L, const Instruction* pc);
void execute_gettablen(lua_State* L, const Instruction* pc);
void execute_settablen(lua_State* L, const Instruction* pc);
void execute_newclosure(lua_State* L, const Instruction* pc);
void execute_namecall(lua_State* L, const Instruction* pc);

// CALL and CALLFB
int execute_call(lua_State* L, const Instruction* pc);
// Returns 0 when the frame was the last one of this luau_execute invocation, and 1 when L->ci has to be resumed
int execute_return(lua_State* L, const Instruction* pc);

// Return 1 when the relation holds; the NOT variants of the jumps negate the result
int execute_equal(lua_State* L, const Instruction* pc);
int execute_lessequal(lua_State* L, const Instruction* pc);
int execute_lessthan(lua_State* L, const Instruction* pc);

// All binary arithmetic opcodes, their constant variants, and MINUS
void execute_arith(lua_State* L, const Instruction* pc);
void execute_concat(lua_State* L, const Instruction* pc);
void execute_length(lua_State* L, const Instruction* pc);
void execute_newtable(lua_State* L, const Instruction* pc);
void execute_duptable(lua_State* L, const Instruction* pc);
void execute_setlist(lua_State* L, const Instruction* pc);

// Returns 1 when the loop body has to be skipped
int execute_fornprep(lua_State* L, const Instruction* pc);
// FORGPREP, FORGPREP_NEXT and FORGPREP_INEXT
void execute_forgprep(lua_State* L, const Instruction* pc);
int execute_forgloop(lua_State* L, const Instruction* pc);

void execute_getvarargs(lua_State* L, const Instruction* pc);
void execute_prepvarargs(lua_State* L, const Instruction* pc);
void execute_dupclosure(lua_State* L, const Instruction* pc);

// All FASTCALL variants; returns 1 when the builtin handled the call and the paired CALL has to be skipped
int execute_fastcall(lua_State* L, const Instruction* pc);
int execute_fastpcall(lua_State* L, const Instruction* pc);

void execute_getudataks(lua_State* L, const Instruction* pc);
void execute_setudataks(lua_State* L, const Instruction* pc);
int execute_namecalludata(lua_State* L, const Instruction* pc);

void execute_newclass(lua_State* L, const Instruction* pc);
void execute_newclassmember(lua_State* L, const Instruction* pc);
void execute_construct(lua_State* L, const Instruction* pc);
// Returns 1 when the construction succeeded and the paired CALL has to be skipped
int execute_finconstruct(lua_State* L, const Instruction* pc);
// Returns 1 when the jump is taken
int execute_cmpproto(lua_State* L, const Instruction* pc);

} // namespace Luau::Jit
