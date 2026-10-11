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

    // Runs 'source' with every function compiled on its first call, and returns its results joined by tostring
    std::string run(const char* source, bool useJit)
    {
        ScopedFastInt threshold(FInt::LuauNativeHitsThreshold, 1);

        lua_settop(L, 0);

        if (useJit)
            create(L);

        load(source);

        if (useJit)
            REQUIRE(compile(L, -1) == JITCompilationResult::Success);

        if (lua_pcall(L, 0, LUA_MULTRET, 0) != 0)
            return std::string("error: ") + lua_tostring(L, -1);

        std::string result;

        for (int i = 1; i <= lua_gettop(L); i++)
        {
            if (i > 1)
                result += ",";

            result += luaL_tolstring(L, i, nullptr);
            lua_pop(L, 1);
        }

        return result;
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

    // The loop has no calls in it, so the whole body runs natively after the one entry
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

    // 'inner' is interpreted, so the frame is entered once for the call itself, and once more every time 'inner' returns
    // into it
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

TEST_CASE("GeneratedCodeMatchesInterpreter")
{
    if (!jit_supported())
        return;

    // Each snippet goes through a different part of the instruction surface; the interpreter is the reference
    const char* snippets[] = {
        // arithmetic, including the paths that fall back to metamethods and errors
        "local a, b = 7, 2 return a + b, a - b, a * b, a / b, a // b, a % b, a ^ b, -a, a % -b, -a // b, 2 ^ 0.5, 3 ^ 3",
        "local t = setmetatable({}, {__add = function(a, b) return 42 end, __unm = function() return 'neg' end}) return t + 1, 1 + t, -t",
        "local s = '10' return s + 1, s * '2', #s, #'abc' .. 'def'",
        "local x = nil return pcall(function() return x + 1 end)",
        "local nan = 0/0 return nan == nan, nan ~= nan, nan < 1, not (nan < 1), 1 <= 2, 'a' < 'b', 'b' <= 'a'",
        // comparisons and logic
        "local function f(a, b) return a == b, a ~= b, a and b, a or b, not a end return f(1, 1), f(nil, false), f('x', 'y'), f(false, 0)",
        "local t = {} local mt = {__eq = function() return true end} local a, b = setmetatable({}, mt), setmetatable({}, mt) return a == b, a == t, rawequal(a, b)",
        // loops
        "local s = 0 for i = 1, 100 do s += i end for i = 10, 1, -2 do s += i end for i = 1, 0 do s = -1 end return s",
        "local s = 0 for i = 1.5, 3.5, 0.5 do s += i end return s",
        "local t = {} for i = 1, 10 do t[i] = i * i end local s = 0 for _, v in ipairs(t) do s += v end for k, v in pairs({a = 1, b = 2}) do s += v end return s, #t",
        "local s = '' for k, v in next, {10, 20, 30} do s ..= k .. '=' .. v .. ';' end return s",
        "local function gen(n) local i = 0 return function() i += 1 if i <= n then return i end end end local s = 0 for v in gen(5) do s += v end return s",
        "local i, s = 0, 0 while i < 10 do i += 1 if i % 2 == 0 then continue end s += i end repeat s -= 1 until s < 20 return i, s",
        // tables
        "local t = {1, 2, 3, x = 'a', ['y z'] = 'b'} t[4] = 4 t.x = 'c' t[1.5] = 'f' t[-1] = 'n' return t[1], t[4], t.x, t['y z'], t[1.5], t[-1], t[100], #t",
        "local t = table.create(4, 0) for i = 1, 4 do t[i] = t[i] + i end return table.concat(t, ',')",
        "local proxy = setmetatable({}, {__index = function(_, k) return k .. '!' end, __newindex = function(t, k, v) rawset(t, k, v * 2) end}) proxy.a = 5 return proxy.a, proxy.b",
        "local t = table.freeze({1}) return pcall(function() t[1] = 2 end)",
        "local t = {} t.x = 1 t.y = 2 t.x = nil return t.x, t.y, next(t)",
        // calls, varargs, multiple returns, recursion, closures
        "local function f(...) return select('#', ...), ... end return f(1, nil, 3)",
        "local function f(...) local a, b = ... return {...}, a, b end local t, a, b = f(4, 5, 6) return #t, a, b",
        "local function fib(n) if n < 2 then return n end return fib(n - 1) + fib(n - 2) end return fib(20)",
        "local function make() local n = 0 return function() n += 1 return n end, function() return n end end local inc, get = make() inc() inc() return get()",
        "local fs = {} for i = 1, 3 do fs[i] = function() return i end end return fs[1](), fs[2](), fs[3]()",
        "local function f() return 1, 2, 3 end local function g(...) return ... end return g(f()), (f())",
        "local obj = {v = 3} function obj:get(k) return self.v * k end return obj:get(2), string.format('%d-%s', 5, 'x'), ('abc'):upper()",
        "local cl = setmetatable({}, {__call = function(self, a) return a * 2 end}) return cl(21)",
        // builtins that the compiler turns into fastcalls
        "return math.floor(3.7), math.abs(-2), math.max(1, 5, 3), math.sqrt(16), bit32.band(12, 10), type(nil), typeof({}), string.byte('A')",
        "local v = vector.create(1, 2, 3) return v.x + v.y + v.z, vector.magnitude(vector.create(3, 4, 0))",
        // errors and protected calls
        "local ok, err = pcall(error, 'boom') return ok, err",
        "local ok, err = pcall(function() local t = nil return t.x end) return ok, string.find(err, 'attempt to index') ~= nil",
        "local ok, err = xpcall(function() error({code = 7}) end, function(e) return e.code end) return ok, err",
        "local depth = 0 local function rec() depth += 1 rec() end local ok = pcall(rec) return ok, depth > 100",
        // coroutines, including yields across generated frames
        "local co = coroutine.create(function(a) local b = coroutine.yield(a + 1) local c = coroutine.yield(b * 2) return a + b + c end) "
        "local _, x = coroutine.resume(co, 1) local _, y = coroutine.resume(co, 10) local _, z = coroutine.resume(co, 100) return x, y, z, coroutine.status(co)",
        "local gen = coroutine.wrap(function() for i = 1, 3 do coroutine.yield(i) end end) return gen(), gen(), gen()",
        "local co = coroutine.wrap(function() for k, v in pairs({1, 2}) do coroutine.yield(v) end return 'done' end) return co(), co(), co()",
        "local co = coroutine.wrap(function() local ok, v = pcall(function() return coroutine.yield('in') + 1 end) return ok, v end) return co(), co(5)",
        // strings
        "local s = 'a' for i = 1, 5 do s = s .. i end return s, #s, s:sub(2, 3), tostring(12.5) .. 'x'",
        "local t = {} for w in string.gmatch('one two three', '%a+') do t[#t + 1] = w:len() end return table.concat(t, '/')",
        // globals and environment
        "x_global = 5 local function f() x_global += 1 return x_global end f() return f(), rawget(_G, 'x_global')",
        "return getfenv ~= nil, type(math), math.pi > 3",
    };

    for (const char* snippet : snippets)
    {
        INFO(snippet);

        JitFixture interpreted;
        JitFixture jit;

        std::string expected = interpreted.run(snippet, false);
        std::string actual = jit.run(snippet, true);

        CHECK(expected == actual);
    }
}

TEST_CASE_FIXTURE(JitFixture, "GeneratedCodeHandlesInterrupts")
{
    if (!jit_supported())
        return;

    create(L);

    // the interrupt runs on back edges and calls, an error raised from it has to unwind through generated frames
    static int interrupts = 0;
    interrupts = 0;

    lua_callbacks(L)->interrupt = [](lua_State* L, int gc)
    {
        if (gc >= 0)
            return;

        if (++interrupts == 1000)
            luaL_error(L, "stopped");
    };

    load("local i = 0 while true do i += 1 end");
    REQUIRE(compile(L, -1) == JITCompilationResult::Success);

    REQUIRE(lua_pcall(L, 0, 0, 0) != 0);
    CHECK(std::string(lua_tostring(L, -1)).find("stopped") != std::string::npos);
    CHECK(interrupts == 1000);
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
