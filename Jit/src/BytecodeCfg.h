// This file is part of the Luau programming language and is licensed under MIT License; see LICENSE.txt for details
#pragma once

#include "Luau/Common.h"

#include <bitset>
#include <vector>

#include <stdint.h>

namespace Luau::Jit
{

struct BytecodeFunction;

struct RegisterSet
{
    std::bitset<256> regs;

    bool varargSeq = false;
    uint8_t varargStart = 0;
};

void require_variadic_sequence(RegisterSet& sourceRs, const RegisterSet& defRs, uint8_t varargStart);

inline constexpr uint32_t kNoAssociatedBlock = ~0u;

struct CfgInfo
{
    std::vector<uint32_t> predecessors;
    std::vector<uint32_t> predecessorsOffsets;

    std::vector<uint32_t> successors;
    std::vector<uint32_t> successorsOffsets;

    std::vector<RegisterSet> in;
    std::vector<RegisterSet> def;
    std::vector<RegisterSet> out;
    RegisterSet captured;
    RegisterSet written;
    std::vector<uint32_t> blockAtPc;
};

struct BlockIteratorWrapper
{
    const uint32_t* itBegin = nullptr;
    const uint32_t* itEnd = nullptr;

    bool empty() const
    {
        return itBegin == itEnd;
    }

    size_t size() const
    {
        return static_cast<size_t>(itEnd - itBegin);
    }

    const uint32_t* begin() const
    {
        return itBegin;
    }

    const uint32_t* end() const
    {
        return itEnd;
    }

    uint32_t operator[](const size_t pos) const
    {
        LUAU_ASSERT(pos < static_cast<size_t>(itEnd - itBegin));
        return itBegin[pos];
    }
};

BlockIteratorWrapper predecessors(const CfgInfo& cfg, uint32_t blockIdx);
BlockIteratorWrapper successors(const CfgInfo& cfg, uint32_t blockIdx);

void compute_cfg_block_at_pc(BytecodeFunction& function);
void compute_cfg_block_edges(BytecodeFunction& function);
void compute_cfg_live_in_out_reg_sets(BytecodeFunction& function);
void compute_cfg_info(BytecodeFunction& function);
RegisterSet compute_live_in_at(const BytecodeFunction& function, int pcpos);

} // namespace Luau::Jit
