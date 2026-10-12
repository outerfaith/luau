// This file is part of the Luau programming language and is licensed under MIT License; see LICENSE.txt for details
#pragma once

#include <llvm/IR/IRBuilder.h>

namespace llvm
{
class DataLayout;
}

namespace Luau::Jit
{

struct VmTypes
{
    explicit VmTypes(llvm::LLVMContext& context);

    llvm::Type* voidTy = nullptr;
    llvm::IntegerType* i1Ty = nullptr;
    llvm::IntegerType* i8Ty = nullptr;
    llvm::IntegerType* i16Ty = nullptr;
    llvm::IntegerType* i32Ty = nullptr;
    llvm::IntegerType* i64Ty = nullptr;
    llvm::Type* floatTy = nullptr;
    llvm::Type* doubleTy = nullptr;
    // the components of a vector TValue, which start at its value and run into its extra field
    llvm::FixedVectorType* vectorTy = nullptr;
    llvm::PointerType* ptrTy = nullptr;

    llvm::StructType* tvalueTy = nullptr;

    llvm::FunctionType* nativeEntryTy = nullptr;

    static constexpr unsigned kTValueValue = 0;
    static constexpr unsigned kTValueExtra = 1;
    static constexpr unsigned kTValueTt = 2;

    llvm::Value* field_ptr(llvm::IRBuilder<>& builder, llvm::Value* base, size_t offset, const llvm::Twine& name = "") const;
    llvm::Value* stack_slot(llvm::IRBuilder<>& builder, llvm::Value* base, int reg, const llvm::Twine& name = "") const;

    llvm::Value* load_state_base(llvm::IRBuilder<>& builder, llvm::Value* state, const llvm::Twine& name = "base") const;
    llvm::Value* load_state_ci(llvm::IRBuilder<>& builder, llvm::Value* state, const llvm::Twine& name = "ci") const;

    void store_ci_savedpc(llvm::IRBuilder<>& builder, llvm::Value* ci, llvm::Value* pc) const;
    void copy_tvalue(llvm::IRBuilder<>& builder, llvm::Value* target, llvm::Value* source) const;
    void verify_layout(const llvm::DataLayout& layout) const;
};

} // namespace Luau::Jit
