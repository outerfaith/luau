// This file is part of the Luau programming language and is licensed under MIT License; see LICENSE.txt for details
#include "Luau/Compiler.h"
#include "Luau/JitCompiler.h"

#include "BytecodeAnalysis.h"
#include "BytecodeCfg.h"
#include "HostHooks.h"

#include "lua.h"
#include "lualib.h"

#include "lapi.h"
#include "lobject.h"
#include "lstate.h"

#include "ScopedFlags.h"
#include "doctest.h"

#include <algorithm>
#include <memory>
#include <string>

LUAU_FASTINT(LuauNativeHitsThreshold)

using namespace Luau;
using namespace Luau::Jit;

namespace
{

// The suite is inert on a host that LLVM cannot generate code for, so this is only queried once
bool jit_supported()
{
    static const bool supported = is_supported();
    return supported;
}

struct JitFixture
{
    std::unique_ptr<lua_State, void (*)(lua_State*)> state;
    lua_State* L = nullptr;

    JitFixture()
        : state(luaL_newstate(), lua_close)
        , L(state.get())
    {
    }

    // Compiles 'source' as a chunk and leaves the resulting closure on top of the stack
    void load(const char* source, int typeInfoLevel = 0)
    {
        CompileOptions options;
        options.typeInfoLevel = typeInfoLevel;

        std::string bytecode = Luau::compile(source, options);

        REQUIRE(luau_load(L, "=JitTest", bytecode.data(), bytecode.size(), 0) == 0);
        REQUIRE(lua_isLfunction(L, -1));
    }

    Proto* proto_at(int idx)
    {
        REQUIRE(lua_isLfunction(L, idx));

        return clvalue(luaA_toobject(L, idx))->l.p;
    }
};

} // namespace

TEST_SUITE_BEGIN("Jit");

TEST_CASE("CompilationResultIsPrintable")
{
    // The host reports why compilation was refused, so every enumerator has to have a name
    for (uint8_t i = 0; i < uint8_t(JITCompilationResult::Count); i++)
        CHECK(!to_string(JITCompilationResult(i)).empty());
}

TEST_CASE_FIXTURE(JitFixture, "CompileWithoutEngine")
{
    load("return 1");

    // Nothing is wired into the lua_State until 'create' is called, and this is checked before the value is even looked at
    CHECK(compile(L, -1) == JITCompilationResult::NotInitialized);
}

TEST_CASE_FIXTURE(JitFixture, "CompileNonFunction")
{
    if (!jit_supported())
        return;

    create(L);

    lua_pushnumber(L, 1);
    CHECK(compile(L, -1) == JITCompilationResult::NotFunction);

    // A C function has no bytecode to translate either
    lua_pushcfunction(
        L,
        [](lua_State*)
        {
            return 0;
        },
        "cfunc"
    );
    CHECK(compile(L, -1) == JITCompilationResult::NotFunction);
}

TEST_CASE_FIXTURE(JitFixture, "CompileMainChunk")
{
    if (!jit_supported())
        return;

    create(L);
    load("local sum = 0 for i = 1, 10 do sum = sum + i end return sum");

    Proto* proto = proto_at(-1);

    REQUIRE(compile(L, -1) == JITCompilationResult::Success);

    // The interpreter only treats a frame as native when both of these are set, and it only reaches the 'enter' callback
    // through the entry instruction that 'codeentry' now points at
    CHECK(proto->execdata != nullptr);
    CHECK(proto->exectarget != 0);
    CHECK(proto->codeentry != proto->code);

    // The code is already installed, so there is nothing left for a second request to do
    CHECK(compile(L, -1) == JITCompilationResult::NothingToCompile);

    ProtoStats stats;
    REQUIRE(get_proto_stats(L, -1, stats));
    CHECK(stats.hits == 0);
    CHECK(stats.calls == 0);
    CHECK(stats.tier == ExecutionTier::Baseline);

    lua_pushvalue(L, -1);
    REQUIRE(lua_pcall(L, 0, 1, 0) == 0);
    CHECK(lua_tonumber(L, -1) == 55.0);
    lua_pop(L, 1);

    // No instruction is translated yet, so the whole body still runs in the interpreter, but the frame has to have been
    // entered natively on the way in
    REQUIRE(get_proto_stats(L, -1, stats));
    CHECK(stats.hits == 1);
    CHECK(stats.calls == 1);
}

TEST_CASE_FIXTURE(JitFixture, "NativeFrameIsResumedAfterNestedCall")
{
    if (!jit_supported())
        return;

    create(L);

    load(R"(
        local function inner(x)
            return x * 2
        end

        local function outer(n)
            local total = 0
            for i = 1, n do
                total = total + inner(i)
            end
            return total
        end

        return outer
    )");

    REQUIRE(lua_pcall(L, 0, 1, 0) == 0);
    REQUIRE(compile(L, -1) == JITCompilationResult::Success);

    lua_pushvalue(L, -1);
    lua_pushnumber(L, 4);
    REQUIRE(lua_pcall(L, 1, 1, 0) == 0);
    CHECK(lua_tonumber(L, -1) == 20.0);
    lua_pop(L, 1);

    // The frame is entered once for the call itself, and once more every time 'inner' returns into it; that resume path
    // is the one that has to keep working while nothing is translated
    ProtoStats stats;
    REQUIRE(get_proto_stats(L, -1, stats));
    CHECK(stats.hits == 1);
    CHECK(stats.calls == 5);
}

TEST_CASE_FIXTURE(JitFixture, "HotFunctionIsCompiledWithoutBeingAsked")
{
    if (!jit_supported())
        return;

    ScopedFastInt threshold(FInt::LuauNativeHitsThreshold, 4);

    create(L);

    load(R"(
        local function work(n)
            return n + 1
        end

        for i = 1, 10 do
            work(i)
        end

        return work
    )");

    REQUIRE(lua_pcall(L, 0, 1, 0) == 0);

    Proto* proto = proto_at(-1);

    // Nothing asked for this function to be compiled; the interpreter handed it over on the call that reached the
    // threshold, and the code was installed in time for that very call
    CHECK(proto->execdata != nullptr);
    CHECK(proto->exectarget != 0);
    CHECK(proto->codeentry != proto->code);

    ProtoStats stats;
    REQUIRE(get_proto_stats(L, -1, stats));
    CHECK(stats.hits == 10);
    CHECK(stats.tier == ExecutionTier::Baseline);

    // The first three calls were interpreted, and every call from the threshold onwards entered natively
    CHECK(stats.calls == 7);

    // There is nothing left for an explicit request to do
    CHECK(compile(L, -1) == JITCompilationResult::NothingToCompile);
}

TEST_CASE_FIXTURE(JitFixture, "ColdFunctionIsLeftToTheInterpreter")
{
    if (!jit_supported())
        return;

    ScopedFastInt threshold(FInt::LuauNativeHitsThreshold, 1000);

    create(L);

    load(R"(
        local function work(n)
            return n + 1
        end

        for i = 1, 10 do
            work(i)
        end

        return work
    )");

    REQUIRE(lua_pcall(L, 0, 1, 0) == 0);

    Proto* proto = proto_at(-1);

    // A function nobody calls often enough is never even shown to the engine
    CHECK(proto->execdata == nullptr);
    CHECK(proto->exectarget == 0);
    CHECK(proto->codeentry == proto->code);

    // The call count is still reported, because it is what the decision is made on
    ProtoStats stats;
    REQUIRE(get_proto_stats(L, -1, stats));
    CHECK(stats.hits == 10);
    CHECK(stats.calls == 0);
    CHECK(stats.tier == ExecutionTier::Cold);
}

TEST_CASE_FIXTURE(JitFixture, "BreakpointTakesNativeCodeAway")
{
    if (!jit_supported())
        return;

    create(L);
    load("local x = 1\nlocal y = x + 1\nreturn y");

    Proto* proto = proto_at(-1);

    REQUIRE(compile(L, -1) == JITCompilationResult::Success);
    REQUIRE(proto->codeentry != proto->code);

    // Generated code has no notion of LOP_BREAK, so the debugger sends the Proto back to the interpreter
    REQUIRE(lua_breakpoint(L, -1, 1, true) != -1);

    CHECK(proto->codeentry == proto->code);
    CHECK(proto->exectarget == 0);

    // 'execdata' has to stay, because the interpreter asserts on it for any frame that is already marked as native
    CHECK(proto->execdata != nullptr);

    // The interpreter never asks for the code back, so the Proto is never compiled again
    CHECK(compile(L, -1) == JITCompilationResult::NothingToCompile);

    lua_pushvalue(L, -1);
    REQUIRE(lua_pcall(L, 0, 1, 0) == 0);
    CHECK(lua_tonumber(L, -1) == 2.0);
}

TEST_CASE_FIXTURE(JitFixture, "GeneratedCodeIsReleasedWithTheProto")
{
    if (!jit_supported())
        return;

    create(L);

    load("return 21 * 2");
    REQUIRE(compile(L, -1) == JITCompilationResult::Success);
    lua_pop(L, 1);

    // Dropping the last reference to the closure destroys its Proto, which is what takes the entry back out of the engine
    lua_gc(L, LUA_GCCOLLECT, 0);

    load("return 1 + 1");
    REQUIRE(compile(L, -1) == JITCompilationResult::Success);

    lua_pushvalue(L, -1);
    REQUIRE(lua_pcall(L, 0, 1, 0) == 0);
    CHECK(lua_tonumber(L, -1) == 2.0);
}

TEST_CASE_FIXTURE(JitFixture, "AnalysisWithoutTypeInfo")
{
    load("local function f(a) return a + 1 end return f");

    Proto* main = proto_at(-1);
    REQUIRE(main->sizep == 1);

    Proto* proto = main->p[0];

    // Type information is optional, and the analysis has to make up defaults for a Proto that has none
    REQUIRE(proto->typeinfo == nullptr);

    BytecodeFunction function;
    HostHooks hooks;
    analyze_bytecode(function, proto, hooks);

    CHECK(function.bcTypeInfo.argumentTypes.size() == size_t(proto->numparams));
    CHECK(function.bcTypeInfo.upvalueTypes.size() == size_t(proto->nups));
    CHECK(function.bcTypes.size() == size_t(proto->sizecode));
    CHECK(!function.bcBlocks.empty());
}

TEST_CASE_FIXTURE(JitFixture, "AnalysisRecoversBlocksAndEdges")
{
    load(
        R"(
        local function pick(a: number, b: number): number
            if a > b then
                return a
            end

            local total = 0
            for i = 1, 4 do
                total += b
            end
            return total
        end

        return pick
    )",
        1
    );

    Proto* main = proto_at(-1);
    REQUIRE(main->sizep == 1);

    Proto* proto = main->p[0];
    REQUIRE(proto->typeinfo != nullptr);
    REQUIRE(proto->numparams == 2);

    BytecodeFunction function;
    HostHooks hooks;
    analyze_bytecode(function, proto, hooks);

    REQUIRE(function.proto == proto);
    REQUIRE(function.bcBlocks.size() > 1);
    CHECK(function.bcTypes.size() == size_t(proto->sizecode));
    CHECK(function.cfg.blockAtPc.size() == size_t(proto->sizecode));

    // The annotated parameters have to survive into the analysis
    REQUIRE(function.bcTypeInfo.argumentTypes.size() == 2);
    CHECK(function.bcTypeInfo.argumentTypes[0] == LBC_TYPE_NUMBER);
    CHECK(function.bcTypeInfo.argumentTypes[1] == LBC_TYPE_NUMBER);

    // Blocks tile the whole instruction stream in program order
    CHECK(function.bcBlocks.front().startpc == 0);

    for (size_t i = 0; i < function.bcBlocks.size(); i++)
    {
        const BytecodeBlock& block = function.bcBlocks[i];

        CHECK(block.startpc >= 0);
        CHECK(block.finishpc >= block.startpc);
        CHECK(block.finishpc < proto->sizecode);

        if (i + 1 < function.bcBlocks.size())
            CHECK(block.finishpc < function.bcBlocks[i + 1].startpc);
    }

    // Every word of the instruction stream, auxiliary words included, belongs to the block that covers it
    for (int pcpos = 0; pcpos < proto->sizecode; pcpos++)
    {
        uint32_t blockIdx = function.cfg.blockAtPc[pcpos];
        REQUIRE(blockIdx != kNoAssociatedBlock);
        REQUIRE(blockIdx < function.bcBlocks.size());

        const BytecodeBlock& block = function.bcBlocks[blockIdx];
        CHECK(pcpos >= block.startpc);
        CHECK(pcpos <= block.finishpc + 1);
    }

    // Predecessor and successor lists have to agree with each other
    for (uint32_t blockIdx = 0; blockIdx < uint32_t(function.bcBlocks.size()); blockIdx++)
    {
        for (uint32_t succIdx : successors(function.cfg, blockIdx))
        {
            BlockIteratorWrapper preds = predecessors(function.cfg, succIdx);
            CHECK(std::find(preds.begin(), preds.end(), blockIdx) != preds.end());
        }

        for (uint32_t predIdx : predecessors(function.cfg, blockIdx))
        {
            BlockIteratorWrapper succs = successors(function.cfg, predIdx);
            CHECK(std::find(succs.begin(), succs.end(), blockIdx) != succs.end());
        }
    }

    // The loop back edge has to be there, so at least one block is its own or a later block's predecessor
    bool hasBackEdge = false;

    for (uint32_t blockIdx = 0; blockIdx < uint32_t(function.bcBlocks.size()); blockIdx++)
    {
        for (uint32_t succIdx : successors(function.cfg, blockIdx))
            hasBackEdge |= succIdx <= blockIdx;
    }

    CHECK(hasBackEdge);

    // Both parameters are read before anything writes them, and nothing else can be live where the function starts
    RegisterSet entry = compute_live_in_at(function, 0);
    CHECK(entry.regs.test(0));
    CHECK(entry.regs.test(1));
    CHECK(entry.regs.count() == size_t(proto->numparams));
    CHECK(!entry.varargSeq);
}

TEST_SUITE_END();
