// This file is part of the Luau programming language and is licensed under MIT License; see LICENSE.txt for details
#pragma once

#include "Luau/Bytecode.h"

#include "lobject.h"

namespace llvm
{
class BasicBlock;
}

namespace Luau::Jit
{

class TranslationUnit;

// The first of the 'count' registers that the inline path of the fastcall at 'pcpos' writes, or -1 when the builtin
// call is left to the generic fastcall path
int inline_builtin_results(const Proto* proto, int pcpos, int& count);

// Lowers the builtin call of the fastcall at 'pcpos' inline: on success it skips past the paired CALL like the
// interpreter does, and everything else branches to 'fallback'. Returns false, without emitting anything, when the
// call has no inline form
bool lower_builtin(TranslationUnit& unit, const Instruction* pc, int pcpos, llvm::BasicBlock* fallback);

} // namespace Luau::Jit
