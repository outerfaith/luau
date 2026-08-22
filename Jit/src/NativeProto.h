// This file is part of the Luau programming language and is licensed under MIT License; see LICENSE.txt for details
#pragma once

#include "Luau/JitCompiler.h"

#include <cstdint>
#include <vector>

// The generated code must not run anymore, and the Proto must not be compiled again; this is what a breakpoint does
#define NP_FLAG_INVALIDATED (1u << 0)

struct Proto;
struct lua_State;

namespace Luau::Jit
{

using NativeEntry = int (*)(lua_State* L, int entrypc);

struct DeoptValue
{
    enum class Kind : uint8_t
    {
        Register = 0,
        Constant = 1,
        None = 2,
    };

    uint64_t value = 0;
    Kind kind = Kind::None;
    uint8_t reg = 0;
};

struct DeoptPoint
{
    uint32_t pc = 0;
    std::vector<DeoptValue> values;
};

struct NativeProto
{
    Proto* proto = nullptr;

    ProtoStats stats;
    uint64_t entry = 0;
    std::vector<DeoptPoint> deopts;
    uint32_t flags = 0;
};

} // namespace Luau::Jit