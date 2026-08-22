// This file is part of the Luau programming language and is licensed under MIT License; see LICENSE.txt for details
#pragma once

#include "Luau/Bytecode.h"
#include "Luau/Common.h"

#include "lobject.h"

namespace Luau::Jit
{

// Reports the VM register definitions and uses of the single bytecode instruction at 'pcpos'
//
// 'visitor' has to provide the following methods:
//     void def(int reg)                     -- 'reg' is written by this instruction
//     void use(int reg)                     -- 'reg' is read by this instruction
//     void def_range(int start, int count)  -- 'count' registers starting at 'start' are written; a 'count' of -1
//                                              stands for a variadic sequence that starts at 'start'
//     void use_range(int start, int count)  -- same as above, but read
//     void capture(int reg)                 -- 'reg' is captured by reference into a closure upvalue, which means it
//                                              can be written to at any point while that closure is alive
//
// For correct analysis, all uses of an instruction have to be reported before any of its definitions
//
// Control flow is not modeled here, the register effects of every path through an instruction are merged together
// An instruction that writes a register on some of its paths only is reported as both a use and a def, so that the
// register is never considered dead in front of it
template<typename T>
void visit_bytecode_reg_defs_uses(T& visitor, Proto* proto, const int pcpos)
{
    LUAU_ASSERT(proto);
    LUAU_ASSERT(pcpos >= 0 && pcpos < proto->sizecode);

    const Instruction* pc = &proto->code[pcpos];

    const auto op = static_cast<LuauOpcode>(LUAU_INSN_OP(*pc));

    int ra = LUAU_INSN_A(*pc);
    int rb = LUAU_INSN_B(*pc);
    int rc = LUAU_INSN_C(*pc);

    switch (op)
    {
    case LOP_NOP:
    case LOP_BREAK:
    case LOP_COVERAGE:
    case LOP_JUMP:
    case LOP_JUMPBACK:
    case LOP_JUMPX:
        break;
    case LOP_PREPVARARGS:
        // Adjusts the frame of the function that is being entered, without touching any of its registers
        break;
    case LOP_CLOSEUPVALS:
        // Closing an upvalue should be counted as a register use (it copies the fresh register value)
        // But we lack the required information about the specific set of registers that are affected
        // Because we don't plan to optimize captured registers atm, we skip full dataflow analysis for them right now
        break;

    case LOP_LOADNIL:
    case LOP_LOADB:
    case LOP_LOADN:
    case LOP_LOADK:
    case LOP_LOADKX:
    case LOP_GETGLOBAL:
    case LOP_GETUPVAL:
    case LOP_GETIMPORT:
    case LOP_NEWTABLE:
    case LOP_DUPTABLE:
    case LOP_NEWCLOSURE:
    case LOP_DUPCLOSURE:
        visitor.def(ra);
        break;
    case LOP_SETGLOBAL:
    case LOP_SETUPVAL:
        visitor.use(ra);
        break;

        // A <- B
    case LOP_MOVE:
    case LOP_NOT:
    case LOP_MINUS:
    case LOP_LENGTH:
    case LOP_GETTABLEKS:
    case LOP_GETTABLEN:
    case LOP_GETUDATAKS:
    case LOP_ADDK:
    case LOP_SUBK:
    case LOP_MULK:
    case LOP_DIVK:
    case LOP_IDIVK:
    case LOP_MODK:
    case LOP_POWK:
    case LOP_ANDK:
    case LOP_ORK:
        visitor.use(rb);

        visitor.def(ra);
        break;
        // A <- B, C
    case LOP_GETTABLE:
    case LOP_ADD:
    case LOP_SUB:
    case LOP_MUL:
    case LOP_DIV:
    case LOP_IDIV:
    case LOP_MOD:
    case LOP_POW:
    case LOP_AND:
    case LOP_OR:
        visitor.use(rb);
        visitor.use(rc);

        visitor.def(ra);
        break;
        // A <- C
    case LOP_SUBRK:
    case LOP_DIVRK:
        visitor.use(rc);

        visitor.def(ra);
        break;

    case LOP_SETTABLEKS:
    case LOP_SETTABLEN:
    case LOP_SETUDATAKS:
        visitor.use(ra);
        visitor.use(rb);
        break;
    case LOP_SETTABLE:
        visitor.use(ra);
        visitor.use(rb);
        visitor.use(rc);
        break;

    case LOP_JUMPIF:
    case LOP_JUMPIFNOT:
    case LOP_JUMPXEQKNIL:
    case LOP_JUMPXEQKB:
    case LOP_JUMPXEQKN:
    case LOP_JUMPXEQKS:
    case LOP_CMPPROTO:
        visitor.use(ra);
        break;
    case LOP_JUMPIFEQ:
    case LOP_JUMPIFLE:
    case LOP_JUMPIFLT:
    case LOP_JUMPIFNOTEQ:
    case LOP_JUMPIFNOTLE:
    case LOP_JUMPIFNOTLT:
        visitor.use(ra);
        visitor.use(static_cast<int>(pc[1])); // AUX holds the second source register
        break;

    case LOP_CONCAT:
        // The concatenation is performed in place over the source range and its result is then copied out of the
        // bottom of that range, so every register of the range has to be treated as written
        visitor.use_range(rb, rc - rb + 1);

        visitor.def_range(rb, rc - rb + 1);
        visitor.def(ra);
        break;
    case LOP_SETLIST:
        visitor.use(ra);
        visitor.use_range(rb, rc - 1);
        break;
    case LOP_GETVARARGS:
        visitor.def_range(ra, rb - 1);
        break;

    case LOP_NAMECALL:
    case LOP_NAMECALLUDATA:
        visitor.use(rb);

        visitor.def_range(ra, 2);
        break;
    case LOP_CALL:
    case LOP_CALLFB:
        visitor.use(ra);
        visitor.use_range(ra + 1, rb - 1);

        visitor.def_range(ra, rc - 1);
        break;
    case LOP_RETURN:
        visitor.use_range(ra, rb - 1);
        break;

        // Fastcalls read the argument registers and write the result registers of the CALL instruction they are
        // paired with; that CALL is always part of the same basic block, so only the arguments that are spelled out
        // in the fastcall itself have to be reported here
    case LOP_FASTCALL:
        break;
    case LOP_FASTCALL1:
    case LOP_FASTCALL2K:
        visitor.use(rb);
        break;
    case LOP_FASTCALL2:
        visitor.use(rb);
        visitor.use(static_cast<int>(pc[1] & 0xff));
        break;
    case LOP_FASTCALL3:
        visitor.use(rb);
        visitor.use(static_cast<int>(pc[1] & 0xff));
        visitor.use(static_cast<int>(pc[1] >> 8 & 0xff));
        break;

        // Numeric for loops own a block of 3 registers, [limit, step, index]
    case LOP_FORNPREP:
        // Non-number loop parameters are coerced in place, so all three registers can be redefined
        visitor.use_range(ra, 3);

        visitor.def_range(ra, 3);
        break;
    case LOP_FORNLOOP:
        visitor.use_range(ra, 3);

        visitor.def(ra + 2);
        break;

        // Generic for loops own a block of 3 registers, [generator, state, index], followed by the loop variables
    case LOP_FORGPREP:
        // This instruction doesn't always redefine Rn, Rn+1, Rn+2, so we have to mark it as implicit use
        visitor.use_range(ra, 3);

        visitor.def_range(ra, 3);
        break;
    case LOP_FORGPREP_NEXT:
    case LOP_FORGPREP_INEXT:
        // Same as above, the fast path rewrites the generator and the index in place
        visitor.use_range(ra, 3);

        visitor.def(ra);
        visitor.def(ra + 2);
        break;
    case LOP_FORGLOOP:
        visitor.use_range(ra, 3);

        visitor.def(ra + 2);
        visitor.def_range(ra + 3, static_cast<int>(static_cast<uint8_t>(pc[1]))); // AUX holds the loop variable count, ignore ipairs marker bit
        break;

    case LOP_CAPTURE:
        // A is the capture type, and B is a register for value and reference captures, an upvalue index otherwise
        switch (static_cast<LuauCaptureType>(ra))
        {
        case LCT_VAL:
            visitor.use(rb);
            break;
        case LCT_REF:
            visitor.use(rb);
            visitor.capture(rb);
            break;
        case LCT_UPVAL:
            break;
        default:
            LUAU_ASSERT(!"Unknown capture type");
            LUAU_UNREACHABLE();
        }
        break;

    case LOP_NEWCLASS:
        if (rb != 0xff) // B is the superclass register, 0xff when the class doesn't extend anything
            visitor.use(rb);

        visitor.def(ra);
        break;
    case LOP_NEWCLASSMEMBER:
        visitor.use(ra);
        visitor.use(rc);
        break;

    case LOP_NATIVECALL:
        LUAU_ASSERT(!"NATIVECALL is a pseudo-opcode and is never part of Proto::code");
        LUAU_UNREACHABLE();

    default:
        // All instructions that reference registers have to be handled explicitly
        LUAU_ASSERT(!"Unknown opcode");
        LUAU_UNREACHABLE();
    }
}

} // namespace Luau::Jit
