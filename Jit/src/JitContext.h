// This file is part of the Luau programming language and is licensed under MIT License; see LICENSE.txt for details
#pragma once

#include "HostHooks.h"
#include "JitProfiling.h"
#include "NativeProto.h"

#include "Luau/JitCompiler.h"

#include <llvm/ExecutionEngine/Orc/LLJIT.h>

#include <memory>
#include <unordered_map>
#include <vector>

struct lua_State;
struct Proto;

namespace Luau::Jit
{

class JitContext final
{
public:
    explicit JitContext(std::unique_ptr<llvm::orc::LLJIT> jit);
    ~JitContext();

    int on_enter(lua_State* L, Proto* proto);
    void on_hot(lua_State* L, Proto* proto);
    void on_destroy(Proto* proto) noexcept;
    void on_disable(lua_State* L, Proto* proto);

    JITCompilationResult compile(Proto* proto);
    void recompile_without_speculation(Proto* proto);
    NativeProto* get_native_proto(Proto* proto);

private:
    std::unique_ptr<llvm::orc::LLJIT> m_jit;

    std::unordered_map<Proto*, std::unique_ptr<NativeProto>> m_nativeProtos;
    HostHooks m_hostHooks;
    uint64_t m_nextFunctionId = 0;
    JitCounters m_counters;
    bool m_countersUsed = false;

    std::unique_ptr<NativeProto> compile_native(Proto* proto, bool speculate, JITCompilationResult& result);
    void bind_native_proto(Proto* proto, std::unique_ptr<NativeProto> nproto);

    // code that was replaced can still be running in a frame below a helper call, so it is kept alive
    std::vector<std::unique_ptr<NativeProto>> m_retiredProtos;
};

// Called by generated code when its exit count reaches LuauJitExitRecompileThreshold
void handle_exit_threshold(lua_State* L, Proto* proto);

} // namespace Luau::Jit
