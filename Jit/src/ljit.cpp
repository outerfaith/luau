// This file is part of the Luau programming language and is licensed under MIT License; see LICENSE.txt for details
#include "luajit.h"

#include "Luau/JitCompiler.h"

int luau_jit_supported()
{
    return Luau::Jit::is_supported();
}

void luau_jit_create(lua_State* L)
{
    Luau::Jit::create(L);
}

void luau_jit_compile_cold(lua_State* L, const int idx)
{
    Luau::Jit::compile(L, idx);
}