// This file is part of the Luau programming language and is licensed under MIT License; see LICENSE.txt for details
#include "VmTypes.h"

#include "Luau/Common.h"

#include "lobject.h"
#include "lstate.h"

#include <llvm/IR/DataLayout.h>

namespace Luau::Jit
{

VmTypes::VmTypes(llvm::LLVMContext& context)
{
    voidTy = llvm::Type::getVoidTy(context);
    i1Ty = llvm::Type::getInt1Ty(context);
    i8Ty = llvm::Type::getInt8Ty(context);
    i16Ty = llvm::Type::getInt16Ty(context);
    i32Ty = llvm::Type::getInt32Ty(context);
    i64Ty = llvm::Type::getInt64Ty(context);
    floatTy = llvm::Type::getFloatTy(context);
    doubleTy = llvm::Type::getDoubleTy(context);
    vectorTy = llvm::FixedVectorType::get(floatTy, LUA_VECTOR_SIZE);
    ptrTy = llvm::PointerType::getUnqual(context);

    // The 'value' union is modeled as an opaque 64-bit blob; accessors reinterpret it as needed
    // Structure types are uniqued by name inside a context, so this resolves to the same type on every construction
    tvalueTy = llvm::StructType::getTypeByName(context, "TValue");

    if (!tvalueTy)
    {
        tvalueTy = llvm::StructType::create(context, "TValue");
        tvalueTy->setBody({i64Ty, llvm::ArrayType::get(i32Ty, LUA_EXTRA_SIZE), i32Ty});
    }

    nativeEntryTy = llvm::FunctionType::get(i32Ty, {ptrTy, i32Ty}, /* isVarArg */ false);
}

llvm::Value* VmTypes::field_ptr(llvm::IRBuilder<>& builder, llvm::Value* base, const size_t offset, const llvm::Twine& name) const
{
    if (offset == 0)
        return base;

    return builder.CreateConstInBoundsGEP1_64(i8Ty, base, offset, name);
}

llvm::Value* VmTypes::stack_slot(llvm::IRBuilder<>& builder, llvm::Value* base, const int reg, const llvm::Twine& name) const
{
    LUAU_ASSERT(reg >= 0);

    if (reg == 0)
        return base;

    return builder.CreateConstInBoundsGEP1_32(tvalueTy, base, static_cast<unsigned>(reg), name);
}

llvm::Value* VmTypes::load_state_base(llvm::IRBuilder<>& builder, llvm::Value* state, const llvm::Twine& name) const
{
    return builder.CreateAlignedLoad(ptrTy, field_ptr(builder, state, offsetof(lua_State, base)), llvm::Align(alignof(StkId)), name);
}

llvm::Value* VmTypes::load_state_ci(llvm::IRBuilder<>& builder, llvm::Value* state, const llvm::Twine& name) const
{
    return builder.CreateAlignedLoad(ptrTy, field_ptr(builder, state, offsetof(lua_State, ci)), llvm::Align(alignof(CallInfo*)), name);
}

void VmTypes::store_ci_savedpc(llvm::IRBuilder<>& builder, llvm::Value* ci, llvm::Value* pc) const
{
    builder.CreateAlignedStore(pc, field_ptr(builder, ci, offsetof(CallInfo, savedpc)), llvm::Align(alignof(const Instruction*)));
}

void VmTypes::copy_tvalue(llvm::IRBuilder<>& builder, llvm::Value* target, llvm::Value* source) const
{
    const llvm::Align align(alignof(TValue));

    // SROA splits these aggregate transfers back into scalars, so there is no reason to spell out the individual fields
    llvm::Value* value = builder.CreateAlignedLoad(tvalueTy, source, align);
    builder.CreateAlignedStore(value, target, align);
}

void VmTypes::verify_layout(const llvm::DataLayout& layout) const
{
    LUAU_ASSERT(layout.getPointerSize() == sizeof(void*));

    LUAU_ASSERT(layout.getTypeAllocSize(tvalueTy) == sizeof(TValue));
    LUAU_ASSERT(layout.getABITypeAlign(tvalueTy) == llvm::Align(alignof(TValue)));

    const llvm::StructLayout* tvalueLayout = layout.getStructLayout(tvalueTy);

    LUAU_ASSERT(tvalueLayout->getElementOffset(kTValueValue) == offsetof(TValue, value));
    LUAU_ASSERT(tvalueLayout->getElementOffset(kTValueExtra) == offsetof(TValue, extra));
    LUAU_ASSERT(tvalueLayout->getElementOffset(kTValueTt) == offsetof(TValue, tt));

    LUAU_ASSERT(sizeof(Value) == sizeof(uint64_t));
}

} // namespace Luau::Jit
