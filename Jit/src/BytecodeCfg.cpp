// This file is part of the Luau programming language and is licensed under MIT License; see LICENSE.txt for details
#include "BytecodeCfg.h"

#include "BytecodeAnalysis.h"
#include "BytecodeVisitUseDef.h"

#include "Luau/Bytecode.h"
#include "Luau/BytecodeUtils.h"
#include "Luau/Common.h"

#include "lobject.h"

namespace Luau::Jit
{

void require_variadic_sequence(RegisterSet& sourceRs, const RegisterSet& defRs, uint8_t varargStart)
{
    if (!defRs.varargSeq)
    {
        // Peel away registers from variadic sequence that we define
        while (defRs.regs.test(varargStart))
            varargStart++;

        LUAU_ASSERT(!sourceRs.varargSeq || sourceRs.varargStart == varargStart);

        sourceRs.varargSeq = true;
        sourceRs.varargStart = varargStart;
    }
    else
    {
        // Variadic use sequence might include registers before def sequence
        for (int i = varargStart; i < defRs.varargStart; i++)
        {
            if (!defRs.regs.test(i))
                sourceRs.regs.set(i);
        }
    }
}

BlockIteratorWrapper predecessors(const CfgInfo& cfg, const uint32_t blockIdx)
{
    LUAU_ASSERT(blockIdx < cfg.predecessorsOffsets.size());

    const uint32_t start = cfg.predecessorsOffsets[blockIdx];
    const uint32_t end =
        blockIdx + 1 < cfg.predecessorsOffsets.size() ? cfg.predecessorsOffsets[blockIdx + 1] : static_cast<uint32_t>(cfg.predecessors.size());

    return BlockIteratorWrapper{cfg.predecessors.data() + start, cfg.predecessors.data() + end};
}

BlockIteratorWrapper successors(const CfgInfo& cfg, const uint32_t blockIdx)
{
    LUAU_ASSERT(blockIdx < cfg.successorsOffsets.size());

    const uint32_t start = cfg.successorsOffsets[blockIdx];
    const uint32_t end =
        blockIdx + 1 < cfg.successorsOffsets.size() ? cfg.successorsOffsets[blockIdx + 1] : static_cast<uint32_t>(cfg.successors.size());

    return BlockIteratorWrapper{cfg.successors.data() + start, cfg.successors.data() + end};
}

void compute_cfg_block_at_pc(BytecodeFunction& function)
{
    const Proto* proto = function.proto;
    LUAU_ASSERT(proto);

    CfgInfo& info = function.cfg;

    info.blockAtPc.clear();
    info.blockAtPc.resize(proto->sizecode, kNoAssociatedBlock);

    for (size_t blockIdx = 0; blockIdx < function.bcBlocks.size(); blockIdx++)
    {
        const auto& [startpc, finishpc] = function.bcBlocks[blockIdx];

        LUAU_ASSERT(startpc >= 0 && startpc < proto->sizecode);
        LUAU_ASSERT(finishpc >= startpc && finishpc < proto->sizecode);

        for (int i = startpc; i <= finishpc;)
        {
            const int length = getOpLength(static_cast<LuauOpcode>(LUAU_INSN_OP(proto->code[i])));

            // Auxiliary words of an instruction are attributed to the block of that instruction
            for (int k = 0; k < length; k++)
            {
                LUAU_ASSERT(i + k < proto->sizecode);
                LUAU_ASSERT(info.blockAtPc[i + k] == kNoAssociatedBlock);

                info.blockAtPc[i + k] = static_cast<uint32_t>(blockIdx);
            }

            i += length;
        }
    }
}

static int get_block_successors(const BytecodeFunction& function, const uint32_t blockIdx, uint32_t targets[2])
{
    const Proto* proto = function.proto;
    const CfgInfo& info = function.cfg;
    const auto& [startpc, finishpc] = function.bcBlocks[blockIdx];

    const Instruction* pc = &proto->code[finishpc];
    const auto op = static_cast<LuauOpcode>(LUAU_INSN_OP(*pc));

    int count = 0;

    auto addTarget = [&](const int target)
    {
        LUAU_ASSERT(target >= 0 && target < proto->sizecode);

        const uint32_t targetIdx = info.blockAtPc[target];
        LUAU_ASSERT(targetIdx != kNoAssociatedBlock);

        // Control can only enter a block through its first instruction
        LUAU_ASSERT(function.bcBlocks[targetIdx].startpc == target);

        targets[count++] = targetIdx;
    };

    // Fastcall skip targets are not block boundaries, so they never form an edge; the same condition is used by
    // build_bytecode_blocks to decide that a fastcall does not terminate its block
    //
    // When degenerate bytecode does place a block boundary right after a fastcall, the skip target ends up in the
    // middle of an unrelated block and cannot be branched to; falling through to the paired CALL is always valid,
    // because that is the path the interpreter itself takes when the builtin doesn't handle the call
    if (const int target = getJumpTarget(*pc, static_cast<uint32_t>(finishpc)); target >= 0 && !isFastCall(op))
        addTarget(target);

    // 'C' of LOADB is an unconditional skip over the following instruction, so the fallthrough edge doesn't exist

    if (const bool skipsNextInst = isSkipC(op) && LUAU_INSN_C(*pc) != 0; isFallthrough(op) && !skipsNextInst)
        addTarget(finishpc + getOpLength(op));

    LUAU_ASSERT(count <= 2);
    return count;
}

void compute_cfg_block_edges(BytecodeFunction& function)
{
    LUAU_ASSERT(function.proto);

    CfgInfo& info = function.cfg;

    const size_t blockCount = function.bcBlocks.size();

    info.predecessors.clear();
    info.predecessorsOffsets.clear();
    info.successors.clear();
    info.successorsOffsets.clear();

    info.successors.reserve(blockCount * 2);
    info.successorsOffsets.reserve(blockCount);

    std::vector<uint32_t> predecessorCounts(blockCount, 0);

    for (uint32_t blockIdx = 0; blockIdx < blockCount; blockIdx++)
    {
        info.successorsOffsets.push_back(static_cast<uint32_t>(info.successors.size()));

        uint32_t targets[2];
        const int count = get_block_successors(function, blockIdx, targets);

        for (int i = 0; i < count; i++)
        {
            info.successors.push_back(targets[i]);
            predecessorCounts[targets[i]]++;
        }
    }

    info.predecessorsOffsets.reserve(blockCount);
    info.predecessors.resize(info.successors.size());

    uint32_t edgeCount = 0;

    for (uint32_t blockIdx = 0; blockIdx < blockCount; blockIdx++)
    {
        info.predecessorsOffsets.push_back(edgeCount);
        edgeCount += predecessorCounts[blockIdx];
    }

    LUAU_ASSERT(edgeCount == static_cast<uint32_t>(info.predecessors.size()));

    // We use a trick here, where the starting offset of the predecessor list is used as the position where to write the
    // next predecessor; the values are adjusted back in a separate loop later
    for (uint32_t blockIdx = 0; blockIdx < blockCount; blockIdx++)
    {
        for (const uint32_t succIdx : successors(info, blockIdx))
            info.predecessors[info.predecessorsOffsets[succIdx]++] = blockIdx;
    }

    for (uint32_t blockIdx = 0; blockIdx < blockCount; blockIdx++)
        info.predecessorsOffsets[blockIdx] -= predecessorCounts[blockIdx];
}

struct RangeRegLiveInComputation
{
    RangeRegLiveInComputation(RegisterSet& defRs, std::bitset<256>& capturedRegs)
        : defRs(defRs)
        , capturedRegs(capturedRegs)
    {
    }

    RegisterSet& defRs;
    std::bitset<256>& capturedRegs;

    RegisterSet inRs;

    void def(const int reg)
    {
        defRs.regs.set(reg, true);
    }

    void use(const int reg)
    {
        if (!defRs.regs.test(reg))
            inRs.regs.set(reg, true);
    }

    void def_varargs(const uint8_t varargStart) const
    {
        defRs.varargSeq = true;
        defRs.varargStart = varargStart;
    }

    void use_varargs(const uint8_t varargStart)
    {
        require_variadic_sequence(inRs, defRs, varargStart);

        // Variadic sequence has been consumed
        defRs.varargSeq = false;
        defRs.varargStart = 0;
    }

    void def_range(const int start, const int count)
    {
        if (count == -1)
        {
            def_varargs(static_cast<uint8_t>(start));
        }
        else
        {
            for (int i = start; i < start + count; i++)
                defRs.regs.set(i, true);
        }
    }

    void use_range(int start, int count)
    {
        if (count == -1)
        {
            use_varargs(static_cast<uint8_t>(start));
        }
        else
        {
            for (int i = start; i < start + count; i++)
            {
                if (!defRs.regs.test(i))
                    inRs.regs.set(i, true);
            }
        }
    }

    void capture(int reg)
    {
        capturedRegs.set(reg, true);
    }
};

static RegisterSet compute_range_live_in_reg_set(
    const BytecodeFunction& function,
    const int startpc,
    const int finishpc,
    RegisterSet& defRs,
    std::bitset<256>& capturedRegs
)
{
    Proto* proto = function.proto;

    RangeRegLiveInComputation visitor(defRs, capturedRegs);

    for (int i = startpc; i <= finishpc;)
    {
        visit_bytecode_reg_defs_uses(visitor, proto, i);

        i += getOpLength(static_cast<LuauOpcode>(LUAU_INSN_OP(proto->code[i])));
    }

    return visitor.inRs;
}

void compute_cfg_live_in_out_reg_sets(BytecodeFunction& function)
{
    const Proto* proto = function.proto;
    LUAU_ASSERT(proto);

    CfgInfo& info = function.cfg;

    const size_t blockCount = function.bcBlocks.size();

    info.def.clear();
    info.out.clear();

    info.in.resize(blockCount);
    info.def.resize(blockCount);
    info.out.resize(blockCount);

    std::bitset<256> capturedRegs;
    for (size_t blockIdx = 0; blockIdx < blockCount; blockIdx++)
    {
        const auto& [startpc, finishpc] = function.bcBlocks[blockIdx];

        info.in[blockIdx] = compute_range_live_in_reg_set(function, startpc, finishpc, info.def[blockIdx], capturedRegs);
    }

    info.captured.regs = capturedRegs;

    std::vector<uint32_t> worklist;

    std::vector<uint8_t> inWorklist;
    inWorklist.resize(blockCount, false);

    for (uint32_t blockIdx = 0; blockIdx < blockCount; blockIdx++)
    {
        worklist.push_back(blockIdx);
        inWorklist[blockIdx] = true;
    }

    while (!worklist.empty())
    {
        const uint32_t blockIdx = worklist.back();
        worklist.pop_back();
        inWorklist[blockIdx] = false;

        RegisterSet& inRs = info.in[blockIdx];
        RegisterSet& defRs = info.def[blockIdx];
        auto& [regs, varargSeq, varargStart] = info.out[blockIdx];

        for (const uint32_t succIdx : successors(info, blockIdx))
        {
            const RegisterSet& succRs = info.in[succIdx];

            regs |= succRs.regs;

            if (succRs.varargSeq)
            {
                LUAU_ASSERT(!varargSeq || varargStart == succRs.varargStart);

                varargSeq = true;
                varargStart = succRs.varargStart;
            }
        }

        RegisterSet oldInRs = inRs;

        inRs.regs |= regs & ~defRs.regs;

        if (varargSeq)
            require_variadic_sequence(inRs, defRs, varargStart);
        if (inRs.regs != oldInRs.regs || inRs.varargSeq != oldInRs.varargSeq)
        {
            for (uint32_t predIdx : predecessors(info, blockIdx))
            {
                if (!inWorklist[predIdx])
                {
                    worklist.push_back(predIdx);
                    inWorklist[predIdx] = true;
                }
            }
        }
    }

    info.written.regs.reset();
    info.written.varargSeq = false;
    info.written.varargStart = 0;

    for (size_t blockIdx = 0; blockIdx < blockCount; blockIdx++)
    {
        const auto& [regs, varargSeq, varargStart] = info.def[blockIdx];

        info.written.regs |= regs;

        if (varargSeq)
        {
            if (!info.written.varargSeq || varargStart < info.written.varargStart)
                info.written.varargStart = varargStart;

            info.written.varargSeq = true;
        }
    }

    const RegisterSet& entryIn = info.in[0];

    LUAU_ASSERT(!entryIn.varargSeq);

    for (size_t i = 0; i < entryIn.regs.size(); i++)
        LUAU_ASSERT(!entryIn.regs.test(i) || i < static_cast<size_t>(proto->numparams));
}

void compute_cfg_info(BytecodeFunction& function)
{
    compute_cfg_block_at_pc(function);
    compute_cfg_block_edges(function);
    compute_cfg_live_in_out_reg_sets(function);
}

RegisterSet compute_live_in_at(const BytecodeFunction& function, const int pcpos)
{
    const CfgInfo& info = function.cfg;

    LUAU_ASSERT(pcpos >= 0 && static_cast<size_t>(pcpos) < info.blockAtPc.size());

    const uint32_t blockIdx = info.blockAtPc[pcpos];
    LUAU_ASSERT(blockIdx != kNoAssociatedBlock);

    const auto& [startpc, finishpc] = function.bcBlocks[blockIdx];
    LUAU_ASSERT(pcpos >= startpc && pcpos <= finishpc);

    RegisterSet defRs;
    std::bitset<256> capturedRegs;

    RegisterSet liveRs = compute_range_live_in_reg_set(function, pcpos, finishpc, defRs, capturedRegs);

    const auto& [regs, varargSeq, varargStart] = info.out[blockIdx];

    liveRs.regs |= regs & ~defRs.regs;

    if (varargSeq)
        require_variadic_sequence(liveRs, defRs, varargStart);

    return liveRs;
}

} // namespace Luau::Jit
