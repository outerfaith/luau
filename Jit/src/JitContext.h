// This file is part of the Luau programming language and is licensed under MIT License; see LICENSE.txt for details
#pragma once

#include "HostHooks.h"
#include "NativeProto.h"

#include "Luau/JitCompiler.h"

#include <llvm/ExecutionEngine/Orc/LLJIT.h>

#include <memory>
#include <unordered_map>

struct lua_State;
struct Proto;

namespace Luau::Jit
{

class JitContext final
{
public:
    explicit JitContext(std::unique_ptr<llvm::orc::LLJIT> jit);

    int on_enter(lua_State* L, Proto* proto);
    void on_hot(lua_State* L, Proto* proto);
    void on_destroy(Proto* proto) noexcept;
    void on_disable(lua_State* L, Proto* proto);

    JITCompilationResult compile(Proto* proto);
    NativeProto* get_native_proto(Proto* proto);

private:
    std::unique_ptr<llvm::orc::LLJIT> m_jit;

    std::unordered_map<Proto*, std::unique_ptr<NativeProto>> m_nativeProtos;
    HostHooks m_hostHooks;
    uint64_t m_nextFunctionId = 0;

    void bind_native_proto(Proto* proto, std::unique_ptr<NativeProto> nproto);
};

} // namespace Luau::Jit
