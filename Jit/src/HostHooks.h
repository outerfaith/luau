// This file is part of the Luau programming language and is licensed under MIT License; see LICENSE.txt for details
#pragma once

#include <stddef.h>
#include <stdint.h>

namespace Luau::Jit
{

class TranslationUnit;

struct HostOperand
{
    enum class Kind : uint8_t
    {
        Register = 0,
        Constant = 1,
    };

    Kind kind = Kind::Register;
    int index = 0;
};

enum class HostMetamethod
{
    Add,
    Sub,
    Mul,
    Div,
    Idiv,
    Mod,
    Pow,
    Minus,
    Equal,
    LessThan,
    LessEqual,
    Length,
    Concat,
};

using HostVectorOperationBytecodeType = uint8_t (*)(const char* member, size_t memberLength);
using HostVectorAccessHandler = bool (*)(TranslationUnit& unit, const char* member, size_t memberLength, int resultReg, int sourceReg, int pcpos);
using HostVectorNamecallHandler =
    bool (*)(TranslationUnit& unit, const char* member, size_t memberLength, int argResReg, int sourceReg, int params, int results, int pcpos);
using HostUserdataOperationBytecodeType = uint8_t (*)(uint8_t type, const char* member, size_t memberLength);
using HostUserdataMetamethodBytecodeType = uint8_t (*)(uint8_t lhsTy, uint8_t rhsTy, HostMetamethod method);
using HostUserdataAccessHandler =
    bool (*)(TranslationUnit& unit, uint8_t type, const char* member, size_t memberLength, int resultReg, int sourceReg, int pcpos);
using HostUserdataMetamethodHandler =
    bool (*)(TranslationUnit& unit, uint8_t lhsTy, uint8_t rhsTy, int resultReg, HostOperand lhs, HostOperand rhs, HostMetamethod method, int pcpos);
using HostUserdataNamecallHandler = bool (*)(
    TranslationUnit& unit,
    uint8_t type,
    const char* member,
    size_t memberLength,
    int argResReg,
    int sourceReg,
    int params,
    int results,
    int pcpos
);

struct HostHooks
{
    // Suggest result type of vector field access
    HostVectorOperationBytecodeType vectorAccessBytecodeType = nullptr;
    // Suggest result type of vector function namecall
    HostVectorOperationBytecodeType vectorNamecallBytecodeType = nullptr;
    // Handle vector value field access
    // 'sourceReg' is guaranteed to be a vector
    // Guards should take a VM exit to 'pcpos'
    HostVectorAccessHandler vectorAccess = nullptr;
    // Handle namecall performed on a vector value
    // 'sourceReg' (self argument) is guaranteed to be a vector
    // All other arguments can be of any type
    // Guards should take a VM exit to 'pcpos'
    HostVectorNamecallHandler vectorNamecall = nullptr;
    // Suggest result type of userdata field access
    HostUserdataOperationBytecodeType userdataAccessBytecodeType = nullptr;
    // Suggest result type of metamethod call
    HostUserdataMetamethodBytecodeType userdataMetamethodBytecodeType = nullptr;
    // Suggest result type of userdata namecall
    HostUserdataOperationBytecodeType userdataNamecallBytecodeType = nullptr;
    // Handle userdata value field access
    // 'sourceReg' is guaranteed to be a userdata, but tag has to be checked
    // Write to 'resultReg' might invalidate 'sourceReg'
    // Guards should take a VM exit to 'pcpos'
    HostUserdataAccessHandler userdataAccess = nullptr;
    // Handle metamethod operation on a userdata value
    // 'lhs' and 'rhs' operands can be VM registers or constants
    // Operand types have to be checked and userdata operand tags have to be checked
    // Write to 'resultReg' might invalidate source operands
    // Guards should take a VM exit to 'pcpos'
    HostUserdataMetamethodHandler userdataMetamethod = nullptr;
    // Handle namecall performed on a userdata value
    // 'sourceReg' (self argument) is guaranteed to be a userdata, but tag has to be checked
    // All other arguments can be of any type
    // Guards should take a VM exit to 'pcpos'
    HostUserdataNamecallHandler userdataNamecall = nullptr;
};

} // namespace Luau::Jit
