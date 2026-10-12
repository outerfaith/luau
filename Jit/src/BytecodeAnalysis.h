// This file is part of the Luau programming language and is licensed under MIT License; see LICENSE.txt for details
#pragma once

#include "BytecodeCfg.h"
#include "HostHooks.h"

#include "Luau/Bytecode.h"

#include <array>
#include <vector>

struct Proto;

namespace Luau::Jit
{

struct BytecodeBlock
{
    int startpc = -1;
    int finishpc = -1;
};

struct BytecodeTypes
{
    uint8_t result = LBC_TYPE_ANY;
    uint8_t a = LBC_TYPE_ANY;
    uint8_t b = LBC_TYPE_ANY;
    uint8_t c = LBC_TYPE_ANY;
};

struct BytecodeRegTypeInfo
{
    uint8_t type = LBC_TYPE_ANY;
    uint8_t reg = 0; // Register slot where variable is stored
    int startpc = 0; // First point where variable is alive (could be before variable has been assigned a value)
    int endpc = 0;   // First point where variable is dead
};

struct BytecodeTypeInfo
{
    std::vector<uint8_t> argumentTypes;
    std::vector<BytecodeRegTypeInfo> regTypes;
    std::vector<uint8_t> upvalueTypes;

    std::vector<uint32_t> regTypeOffsets;
};

struct BytecodeFunction
{
    Proto* proto = nullptr;

    std::vector<uint8_t> blockStarts;
    std::vector<BytecodeBlock> bcBlocks;

    CfgInfo cfg;

    std::vector<BytecodeTypes> bcTypes;
    BytecodeTypeInfo bcTypeInfo;
    BytecodeTypeInfo bcOriginalTypeInfo;
};

// Operand tags that generated code saw, per instruction: bit 'tag' of the low half for the first register that
// feedback_operands lists, of the high half for the second
using TypeFeedback = std::vector<uint32_t>;

// The registers whose tags are recorded for the optimized tier, in the order of the bytecode types 'a' and 'b' of the
// instruction; -1 when there is no such register
std::array<int, 2> feedback_operands(const uint32_t* pc);

// Instructions that record no operands keep flags of their own in their feedback instead
// GETTABLE, SETTABLE, GETTABLEN, SETTABLEN: a table with a metatable left the fast path of an array part access
constexpr uint32_t kFeedbackArrayMetatable = 1;
// FORNPREP: a loop started that an int32 counter can't follow
constexpr uint32_t kFeedbackLoopNotInteger = 1;

bool is_custom_userdata_bytecode_type(uint8_t ty);

void load_bytecode_type_info(BytecodeFunction& function);

void find_bytecode_block_starts(BytecodeFunction& function);

void build_bytecode_blocks(BytecodeFunction& function);

uint8_t get_reg_tag(std::array<uint8_t, 256>& regTags, BytecodeTypeInfo& bcTypeInfo, uint8_t reg, int pc);

// With feedback, the operands it covers get the type they always had at run time, or none when they had several, over
// the declared one
void analyze_bytecode_types(BytecodeFunction& function, const HostHooks& hostHooks, const TypeFeedback* feedback = nullptr);
void analyze_bytecode(BytecodeFunction& function, Proto* proto, const HostHooks& hostHooks, const TypeFeedback* feedback = nullptr);

} // namespace Luau::Jit
