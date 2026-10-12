// This file is part of the Luau programming language and is licensed under MIT License; see LICENSE.txt for details
#include "JitProfiling.h"

#include "Luau/BytecodeDump.h"

#include <llvm/ExecutionEngine/JITLink/JITLink.h>
#include <llvm/ExecutionEngine/Orc/DebugUtils.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h>
#include <llvm/ExecutionEngine/Orc/ObjectTransformLayer.h>
#include <llvm/Support/FileSystem.h>

#include <algorithm>
#include <mutex>
#include <stdio.h>
#include <unistd.h>
#include <string>
#include <utility>
#include <vector>

namespace Luau::Jit
{

void JitCounters::print() const
{
    uint64_t helperTotal = 0;
    uint64_t exitTotal = 0;

    for (int op = 0; op < LOP__COUNT; op++)
    {
        helperTotal += helperCalls[op];
        exitTotal += vmExits[op];
    }

    fprintf(
        stderr,
        "luaujit counters: %llu helper calls, %llu vm exits, %llu interpreter resumes, %llu native transfers\n",
        (unsigned long long)helperTotal,
        (unsigned long long)exitTotal,
        (unsigned long long)interpreterResumes,
        (unsigned long long)nativeTransfers
    );

    auto printTable = [](const char* title, const uint64_t* counts, uint64_t total)
    {
        std::vector<std::pair<uint64_t, int>> sorted;

        for (int op = 0; op < LOP__COUNT; op++)
        {
            if (counts[op])
                sorted.push_back({counts[op], op});
        }

        std::sort(sorted.rbegin(), sorted.rend());

        for (const auto& [count, op] : sorted)
            fprintf(
                stderr,
                "  %-8s %-16s %14llu  %5.1f%%\n",
                title,
                Bytecode::getLuauOpcodeName(LuauOpcode(op)),
                (unsigned long long)count,
                100.0 * count / total
            );
    };

    printTable("helper", helperCalls, helperTotal);
    printTable("exit", vmExits, exitTotal);
}

namespace
{

class PerfMapPlugin final : public llvm::orc::ObjectLinkingLayer::Plugin
{
public:
    explicit PerfMapPlugin(FILE* file)
        : m_file(file)
    {
    }

    ~PerfMapPlugin() override
    {
        fclose(m_file);
    }

    void modifyPassConfig(
        llvm::orc::MaterializationResponsibility& mr,
        llvm::jitlink::LinkGraph& graph,
        llvm::jitlink::PassConfiguration& config
    ) override
    {
        // addresses are final once the graph has been allocated
        config.PostAllocationPasses.push_back(
            [this](llvm::jitlink::LinkGraph& graph)
            {
                std::lock_guard<std::mutex> lock(m_mutex);

                for (llvm::jitlink::Symbol* symbol : graph.defined_symbols())
                {
                    if (!symbol->isCallable() || !symbol->hasName() || symbol->getSize() == 0)
                        continue;

                    llvm::StringRef name = *symbol->getName();
                    fprintf(
                        m_file,
                        "%llx %llx %.*s\n",
                        (unsigned long long)symbol->getAddress().getValue(),
                        (unsigned long long)symbol->getSize(),
                        int(name.size()),
                        name.data()
                    );
                }

                fflush(m_file);
                return llvm::Error::success();
            }
        );
    }

    llvm::Error notifyFailed(llvm::orc::MaterializationResponsibility&) override
    {
        return llvm::Error::success();
    }

    llvm::Error notifyRemovingResources(llvm::orc::JITDylib&, llvm::orc::ResourceKey) override
    {
        return llvm::Error::success();
    }

    void notifyTransferringResources(llvm::orc::JITDylib&, llvm::orc::ResourceKey, llvm::orc::ResourceKey) override {}

private:
    FILE* m_file;
    std::mutex m_mutex;
};

} // namespace

bool enable_perf_map(llvm::orc::LLJIT& jit)
{
    auto* layer = llvm::dyn_cast<llvm::orc::ObjectLinkingLayer>(&jit.getObjLinkingLayer());
    if (!layer)
        return false;

    char path[64];
    snprintf(path, sizeof(path), "/tmp/perf-%d.map", int(getpid()));

    FILE* file = fopen(path, "w");
    if (!file)
        return false;

    layer->addPlugin(std::make_shared<PerfMapPlugin>(file));
    return true;
}

bool enable_object_dump(llvm::orc::LLJIT& jit)
{
    const std::string dir = "/tmp/luaujit-" + std::to_string(getpid());

    if (llvm::sys::fs::create_directories(dir))
        return false;

    jit.getObjTransformLayer().setTransform(llvm::orc::DumpObjects(dir));
    return true;
}

} // namespace Luau::Jit
