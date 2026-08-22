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

bool is_custom_userdata_bytecode_type(uint8_t ty);

void load_bytecode_type_info(BytecodeFunction& function);

void find_bytecode_block_starts(BytecodeFunction& function);

void build_bytecode_blocks(BytecodeFunction& function);

uint8_t get_reg_tag(std::array<uint8_t, 256>& regTags, BytecodeTypeInfo& bcTypeInfo, uint8_t reg, int pc);

void analyze_bytecode_types(BytecodeFunction& function, const HostHooks& hostHooks);
void analyze_bytecode(BytecodeFunction& function, Proto* proto, const HostHooks& hostHooks);

} // namespace Luau::Jit
