// This file is part of the Luau programming language and is licensed under MIT License; see LICENSE.txt for details
#pragma once

#include <cstdint>
#include <string>

struct lua_State;

namespace Luau::Jit
{

enum class JITCompilationResult : uint8_t
{
    Success = 0,
    // The JIT engine was never created for this lua_State
    NotInitialized = 1,
    // The value at the given index is not a Lua function
    NotFunction = 2,
    // The function already has generated code, or a breakpoint has permanently taken its generated code away
    NothingToCompile = 3,
    // The bytecode of the function could not be translated
    TranslationFailed = 4,
    // The generated LLVM IR is malformed; this is always a bug in the translation
    VerificationFailed = 5,
    // LLVM failed to add the module to the JIT or to resolve the generated function
    CodeGenFailed = 6,

    Count = 7
};


enum class ExecutionTier : uint8_t
{
    // Interpreted; no code has been generated for the function yet
    Cold = 0,
    // The function was called often enough for the interpreter to hand it over and has been compiled once
    Baseline = 1,
    // Not yet implemented
    Optimized = 2,
    Aggressive = 3,

    Count = 4
};

struct ProtoStats
{
    // How many times the interpreter has started a frame for this function
    uint64_t hits = 0;
    // How many times the interpreter has handed control to generated code
    uint64_t calls = 0;
    ExecutionTier tier = ExecutionTier::Cold;
    // How many times generated code left for the interpreter in the middle of the function
    uint64_t exits = 0;
    // Whether the generated code relies on bytecode types, which a function stops doing once it exits too often
    bool speculative = false;
};

std::string to_string(const JITCompilationResult& result);

bool is_supported();

void create(lua_State* L);

// Compiles a function immediately without waiting for it to become hot
JITCompilationResult compile(lua_State* L, int idx);

bool get_proto_stats(lua_State* L, int idx, ProtoStats& stats);

} // namespace Luau::Jit