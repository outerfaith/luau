// This file is part of the Luau programming language and is licensed under MIT License; see LICENSE.txt for details
#pragma once

#include "Luau/Bytecode.h"

#include <stdint.h>

namespace llvm::orc
{
class LLJIT;
}

namespace Luau::Jit
{

// Event counts that generated code bumps when it was compiled with DebugLuauJitCounters; they show which instructions
// still go through helpers or leave generated code, which is where the time goes that no fast path covers
struct JitCounters
{
    uint64_t helperCalls[LOP__COUNT] = {};
    uint64_t vmExits[LOP__COUNT] = {};
    // generated code returned to the interpreter to continue a frame that has no generated code (or a bad entry pc)
    uint64_t interpreterResumes = 0;
    // a call or a return continued in generated code without going through the interpreter
    uint64_t nativeTransfers = 0;

    void print() const;
};

// Writes /tmp/perf-<pid>.map entries for every function the JIT links, so that perf can name samples in generated
// code; returns false when the JIT doesn't link through JITLink
bool enable_perf_map(llvm::orc::LLJIT& jit);

// Writes every object the JIT links to /tmp/luaujit-<pid>/<function>.o, for objdump; returns false when the directory
// can't be created
bool enable_object_dump(llvm::orc::LLJIT& jit);

} // namespace Luau::Jit
