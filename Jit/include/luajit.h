// This file is part of the Luau programming language and is licensed under MIT License; see LICENSE.txt for details
#pragma once

#ifndef LUAJIT_API
#define LUAJIT_API extern
#endif

typedef struct lua_State lua_State;

// returns 1 if the JIT engine is supported, 0 otherwise
LUAJIT_API int luau_jit_supported();

// creates an instance of the JIT engine. you must check that this feature is supported using luau_jit_supported().
LUAJIT_API void luau_jit_create(lua_State* L);

// compiles target function and all inner functions, even if they are cold
LUAJIT_API void luau_jit_compile_cold(lua_State* L, int idx);
