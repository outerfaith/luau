// This file is part of the Luau programming language and is licensed under MIT License; see LICENSE.txt for details
#include "BuiltinTranslation.h"

#include "BytecodeAnalysis.h"
#include "BytecodeTranslation.h"
#include "VmTypes.h"

#include "Luau/BytecodeUtils.h"
#include "Luau/Common.h"

#include "lgc.h"
#include "lstate.h"

#include <llvm/IR/Constants.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Intrinsics.h>

#include <math.h>
#include <vector>

namespace Luau::Jit
{

namespace
{

// arguments that the builtin reads without checking their type
constexpr int kAnyTag = -1;

struct BuiltinArg
{
    int reg = -1;
    // set instead of 'reg' for the constant of FASTCALL2K
    const TValue* k = nullptr;
};

struct BuiltinCall
{
    LuauBuiltinFunction bfid = LBF_NONE;
    // the function register of the paired CALL, which receives the result
    int ra = -1;
    int nresults = 0;
    int target = -1;
    std::vector<BuiltinArg> args;
    // the tag that each argument needs for the inline path, one per argument that is read
    std::vector<int> tags;
};

bool decode_call(const Proto* proto, const int pcpos, BuiltinCall& call)
{
    const Instruction* pc = &proto->code[pcpos];
    const auto op = LuauOpcode(LUAU_INSN_OP(*pc));

    // every variant places the paired CALL 'C' instructions past the first word of the fastcall
    const Instruction callInsn = pc[1 + LUAU_INSN_C(*pc)];

    call.bfid = LuauBuiltinFunction(LUAU_INSN_A(*pc));
    call.ra = LUAU_INSN_A(callInsn);
    call.nresults = LUAU_INSN_C(callInsn) - 1;
    call.target = getJumpTarget(*pc, pcpos);

    switch (op)
    {
    case LOP_FASTCALL:
    {
        const int nparams = LUAU_INSN_B(callInsn) - 1;
        if (nparams == LUA_MULTRET)
            return false;

        for (int i = 0; i < nparams; i++)
            call.args.push_back({call.ra + 1 + i});
        break;
    }
    case LOP_FASTCALL1:
        call.args.push_back({LUAU_INSN_B(*pc)});
        break;
    case LOP_FASTCALL2:
        call.args.push_back({LUAU_INSN_B(*pc)});
        call.args.push_back({static_cast<int>(pc[1])});
        break;
    case LOP_FASTCALL2K:
        call.args.push_back({LUAU_INSN_B(*pc)});
        call.args.push_back({-1, &proto->k[pc[1]]});
        break;
    case LOP_FASTCALL3:
        call.args.push_back({LUAU_INSN_B(*pc)});
        call.args.push_back({static_cast<int>(LUAU_INSN_AUX_A(pc[1]))});
        call.args.push_back({static_cast<int>(LUAU_INSN_AUX_B(pc[1]))});
        break;
    default:
        return false;
    }

    // a multret result would have to go to the stack and move L->top
    return call.nresults >= 0;
}

// The argument tags that the lbuiltins.cpp fast function checks before it succeeds, for the builtins that have an
// inline form
bool match_signature(BuiltinCall& call)
{
    const int nparams = int(call.args.size());

    // string.byte is the only builtin with an inline form that produces more than one result
    if (call.nresults > 1 && call.bfid != LBF_STRING_BYTE)
        return false;

    auto fixed = [&](std::initializer_list<int> tags)
    {
        if (nparams < int(tags.size()))
            return false;

        call.tags.assign(tags);
        return true;
    };

    auto variadic = [&](int minParams, int tag)
    {
        if (nparams < minParams)
            return false;

        call.tags.assign(nparams, tag);
        return true;
    };

    switch (call.bfid)
    {
    case LBF_ASSERT:
        return call.nresults == 0 && fixed({kAnyTag});
    case LBF_MATH_ABS:
    case LBF_MATH_ACOS:
    case LBF_MATH_ASIN:
    case LBF_MATH_ATAN:
    case LBF_MATH_CEIL:
    case LBF_MATH_COSH:
    case LBF_MATH_COS:
    case LBF_MATH_DEG:
    case LBF_MATH_EXP:
    case LBF_MATH_FLOOR:
    case LBF_MATH_LOG10:
    case LBF_MATH_RAD:
    case LBF_MATH_SINH:
    case LBF_MATH_SIN:
    case LBF_MATH_SQRT:
    case LBF_MATH_TANH:
    case LBF_MATH_TAN:
    case LBF_MATH_SIGN:
    case LBF_MATH_ROUND:
    case LBF_MATH_ISNAN:
    case LBF_MATH_ISINF:
    case LBF_MATH_ISFINITE:
    case LBF_BIT32_BNOT:
    case LBF_BIT32_COUNTLZ:
    case LBF_BIT32_COUNTRZ:
    case LBF_BIT32_BYTESWAP:
        return fixed({LUA_TNUMBER});
    case LBF_MATH_LOG:
        // the base selects between three formulas, which is only worth it when the base is a constant
        if (nparams == 1)
            return fixed({LUA_TNUMBER});

        return call.args[1].k && fixed({LUA_TNUMBER, LUA_TNUMBER});
    case LBF_MATH_ATAN2:
    case LBF_MATH_FMOD:
    case LBF_MATH_POW:
    case LBF_BIT32_ARSHIFT:
    case LBF_BIT32_LSHIFT:
    case LBF_BIT32_RSHIFT:
    case LBF_BIT32_LROTATE:
    case LBF_BIT32_RROTATE:
        return fixed({LUA_TNUMBER, LUA_TNUMBER});
    case LBF_BIT32_EXTRACTK:
        // the compiler only emits it with FASTCALL2K
        return nparams >= 2 && call.args[1].k && fixed({LUA_TNUMBER, LUA_TNUMBER});
    case LBF_MATH_MIN:
    case LBF_MATH_MAX:
    case LBF_BIT32_BAND:
    case LBF_BIT32_BOR:
    case LBF_BIT32_BXOR:
    case LBF_BIT32_BTEST:
        return variadic(2, LUA_TNUMBER);
    case LBF_MATH_CLAMP:
    case LBF_MATH_LERP:
        return fixed({LUA_TNUMBER, LUA_TNUMBER, LUA_TNUMBER});
    case LBF_BIT32_EXTRACT:
        return nparams == 2 ? fixed({LUA_TNUMBER, LUA_TNUMBER}) : fixed({LUA_TNUMBER, LUA_TNUMBER, LUA_TNUMBER});
    case LBF_BIT32_REPLACE:
        return nparams == 3 ? fixed({LUA_TNUMBER, LUA_TNUMBER, LUA_TNUMBER}) : fixed({LUA_TNUMBER, LUA_TNUMBER, LUA_TNUMBER, LUA_TNUMBER});
    case LBF_TYPE:
        return fixed({kAnyTag});
    case LBF_SETMETATABLE:
        // the barrier needs the metatable in a register
        return fixed({LUA_TTABLE, LUA_TTABLE}) && !call.args[1].k;
    case LBF_STRING_LEN:
        return fixed({LUA_TSTRING});
    case LBF_STRING_BYTE:
        // luauF_byte produces exactly as many results as were asked for
        if (call.nresults == 0)
            return false;

        return nparams == 2 ? fixed({LUA_TSTRING, LUA_TNUMBER}) : fixed({LUA_TSTRING, LUA_TNUMBER, LUA_TNUMBER});
    case LBF_BUFFER_READI8:
    case LBF_BUFFER_READU8:
    case LBF_BUFFER_READI16:
    case LBF_BUFFER_READU16:
    case LBF_BUFFER_READI32:
    case LBF_BUFFER_READU32:
    case LBF_BUFFER_READF32:
    case LBF_BUFFER_READF64:
        return fixed({LUA_TBUFFER, LUA_TNUMBER});
    case LBF_BUFFER_WRITEU8:
    case LBF_BUFFER_WRITEU16:
    case LBF_BUFFER_WRITEU32:
    case LBF_BUFFER_WRITEF32:
    case LBF_BUFFER_WRITEF64:
        return call.nresults == 0 && fixed({LUA_TBUFFER, LUA_TNUMBER, LUA_TNUMBER});
    default:
        break;
    }

    // the vector builtins below are written for three components
    if (LUA_VECTOR_SIZE != 3)
        return false;

    switch (call.bfid)
    {
    case LBF_VECTOR:
        return nparams == 2 ? fixed({LUA_TNUMBER, LUA_TNUMBER}) : fixed({LUA_TNUMBER, LUA_TNUMBER, LUA_TNUMBER});
    case LBF_VECTOR_MAGNITUDE:
    case LBF_VECTOR_NORMALIZE:
    case LBF_VECTOR_FLOOR:
    case LBF_VECTOR_CEIL:
    case LBF_VECTOR_ABS:
    case LBF_VECTOR_SIGN:
        return fixed({LUA_TVECTOR});
    case LBF_VECTOR_CROSS:
    case LBF_VECTOR_DOT:
        return fixed({LUA_TVECTOR, LUA_TVECTOR});
    case LBF_VECTOR_MIN:
    case LBF_VECTOR_MAX:
        return variadic(2, LUA_TVECTOR);
    case LBF_VECTOR_CLAMP:
        return fixed({LUA_TVECTOR, LUA_TVECTOR, LUA_TVECTOR});
    case LBF_VECTOR_LERP:
        return fixed({LUA_TVECTOR, LUA_TVECTOR, LUA_TNUMBER});
    default:
        return false;
    }
}

bool analyze_call(const Proto* proto, const int pcpos, BuiltinCall& call)
{
    if (!decode_call(proto, pcpos, call) || !match_signature(call))
        return false;

    // constants are checked here instead of at runtime
    for (size_t i = 0; i < call.tags.size(); i++)
    {
        if (const TValue* k = call.args[i].k; k && call.tags[i] != kAnyTag && ttype(k) != call.tags[i])
            return false;
    }

    return true;
}

class BuiltinLowering
{
public:
    BuiltinLowering(TranslationUnit& unit, const BuiltinCall& call, llvm::BasicBlock* fallback)
        : m_unit(unit)
        , m_build(unit.builder())
        , m_types(unit.types())
        , m_call(call)
        , m_fallback(fallback)
    {
    }

    void lower();

private:
    llvm::Value* number(size_t i)
    {
        const BuiltinArg& arg = m_call.args[i];

        return arg.k ? llvm::ConstantFP::get(m_types.doubleTy, nvalue(arg.k)) : m_unit.load_number(arg.reg);
    }

    llvm::Value* vector(size_t i)
    {
        const BuiltinArg& arg = m_call.args[i];

        if (!arg.k)
            return m_unit.load_vector(arg.reg);

        const float* v = vvalue(arg.k);
        std::vector<llvm::Constant*> components;
        for (int c = 0; c < LUA_VECTOR_SIZE; c++)
            components.push_back(llvm::ConstantFP::get(m_types.floatTy, v[c]));

        return llvm::ConstantVector::get(components);
    }

    // (unsigned)(long long)d and (int)d as cvttsd2si computes them, which is what the casts in lbuiltins.cpp compile to
    // on x86-64: values out of range, NaN included, become the smallest integer
    llvm::Value* to_int(llvm::Value* d, llvm::IntegerType* type)
    {
        const unsigned bits = type->getBitWidth();

        llvm::Value* inRange = m_build.CreateFCmpOLT(
            m_build.CreateUnaryIntrinsic(llvm::Intrinsic::fabs, d), llvm::ConstantFP::get(m_types.doubleTy, ldexp(1.0, int(bits) - 1))
        );

        // the conversion is poison when it is out of range, which the select never picks
        return m_build.CreateSelect(inRange, m_build.CreateFPToSI(d, type), llvm::ConstantInt::get(type, llvm::APInt::getSignedMinValue(bits)));
    }

    llvm::Value* to_unsigned(llvm::Value* d)
    {
        return m_build.CreateTrunc(to_int(d, m_types.i64Ty), m_types.i32Ty);
    }

    llvm::Value* unsigned_arg(size_t i)
    {
        return to_unsigned(number(i));
    }

    llvm::Value* int_arg(size_t i)
    {
        return to_int(number(i), m_types.i32Ty);
    }

    // the shift counts that reach a shift are masked like x86 masks them, which keeps out of range counts defined
    llvm::Value* shift_count(llvm::Value* count)
    {
        return m_build.CreateAnd(count, 31);
    }

    llvm::Value* sum_components(llvm::Value* v)
    {
        llvm::Value* sum = m_build.CreateFAdd(m_build.CreateExtractElement(v, uint64_t(0)), m_build.CreateExtractElement(v, uint64_t(1)));

        return m_build.CreateFAdd(sum, m_build.CreateExtractElement(v, uint64_t(2)));
    }

    llvm::Value* splat(llvm::Value* scalar)
    {
        return m_build.CreateVectorSplat(LUA_VECTOR_SIZE, scalar);
    }

    llvm::Constant* fp(llvm::Type* type, double value)
    {
        return llvm::ConstantFP::get(type, value);
    }

    // continues in a new block when 'cond' holds, and gives the call to the generic path when it doesn't
    void require(llvm::Value* cond)
    {
        llvm::BasicBlock* next = m_unit.new_block("builtin.checked");
        m_build.CreateCondBr(cond, next, m_fallback, m_unit.likely());
        m_build.SetInsertPoint(next);
    }

    void store_number(llvm::Value* value)
    {
        m_unit.store_number(m_call.ra, value);
    }

    void store_unsigned(llvm::Value* value)
    {
        store_number(m_build.CreateUIToFP(value, m_types.doubleTy));
    }

    void store_boolean(llvm::Value* value)
    {
        m_unit.store_boolean(m_call.ra, m_build.CreateZExt(value, m_types.i32Ty));
    }

    void store_vector(llvm::Value* value)
    {
        m_unit.store_vector(m_call.ra, value);
    }

    // min and max keep the earlier value unless the later one compares strictly
    template<typename Load>
    llvm::Value* fold_select(llvm::CmpInst::Predicate pred, Load load)
    {
        llvm::Value* r = load(0);

        for (size_t i = 1; i < m_call.args.size(); i++)
        {
            llvm::Value* a = load(i);
            r = m_build.CreateSelect(m_build.CreateFCmp(pred, a, r), a, r);
        }

        return r;
    }

    template<typename Op>
    llvm::Value* fold_unsigned(Op op)
    {
        llvm::Value* r = unsigned_arg(0);

        for (size_t i = 1; i < m_call.args.size(); i++)
            r = op(r, unsigned_arg(i));

        return r;
    }

    void lower_math_unary(llvm::Intrinsic::ID id)
    {
        store_number(m_build.CreateUnaryIntrinsic(id, number(0)));
    }

    void lower_number();
    void lower_bit32();
    void lower_vector();
    void lower_buffer();
    void lower_string_byte();

    // the address of an access of 'size' bytes at the offset in the second argument, after the bounds check of
    // luauF_readinteger and friends
    llvm::Value* buffer_address(uint64_t size);

    TranslationUnit& m_unit;
    llvm::IRBuilder<>& m_build;
    const VmTypes& m_types;
    const BuiltinCall& m_call;
    llvm::BasicBlock* m_fallback;
};

void BuiltinLowering::lower()
{
    switch (m_call.bfid)
    {
    case LBF_ASSERT:
    {
        // luauF_assert produces no results, and leaves the falsy case to the library function that raises the error
        llvm::Value* falsy = m_unit.is_falsy(m_call.args[0].reg);
        require(m_build.CreateNot(falsy));
        return;
    }
    case LBF_TYPE:
    {
        llvm::Value* global = m_unit.load_field(m_types.ptrTy, m_unit.state(), offsetof(lua_State, global), "global");
        llvm::Value* names = m_types.field_ptr(m_build, global, offsetof(global_State, ttname));

        const BuiltinArg& arg = m_call.args[0];
        llvm::Value* tag = arg.k ? m_build.getInt32(ttype(arg.k)) : m_unit.load_tag(arg.reg);
        llvm::Value* name = m_build.CreateLoad(m_types.ptrTy, m_build.CreateGEP(m_types.ptrTy, names, tag), "ttname");

        // type names are fixed strings that the collector never frees
        m_unit.store_object(m_call.ra, name, LUA_TSTRING);
        return;
    }
    case LBF_SETMETATABLE:
    {
        llvm::Value* table = m_unit.load_pointer(m_call.args[0].reg);
        llvm::Value* readonly = m_unit.load_field(m_types.i8Ty, table, offsetof(LuaTable, readonly));
        llvm::Value* metatable = m_unit.load_field(m_types.ptrTy, table, offsetof(LuaTable, metatable));

        // luauF_setmetatable leaves frozen tables and the __metatable check of replacing a metatable to the library
        require(m_build.CreateAnd(m_build.CreateICmpEQ(readonly, m_build.getInt8(0)), m_build.CreateIsNull(metatable)));

        const int mtReg = m_call.args[1].reg;
        m_unit.store_field(m_unit.load_pointer(mtReg), table, offsetof(LuaTable, metatable));
        m_unit.emit_barrier(table, mtReg, reinterpret_cast<const void*>(&luaC_barrierf));

        m_unit.store_object(m_call.ra, table, LUA_TTABLE);
        return;
    }
    case LBF_STRING_LEN:
    {
        llvm::Value* len = m_unit.load_field(m_types.i32Ty, m_unit.load_pointer(m_call.args[0].reg), offsetof(TString, len));
        store_number(m_build.CreateSIToFP(len, m_types.doubleTy));
        return;
    }
    case LBF_STRING_BYTE:
        lower_string_byte();
        return;
    case LBF_BUFFER_READI8:
    case LBF_BUFFER_READU8:
    case LBF_BUFFER_WRITEU8:
    case LBF_BUFFER_READI16:
    case LBF_BUFFER_READU16:
    case LBF_BUFFER_WRITEU16:
    case LBF_BUFFER_READI32:
    case LBF_BUFFER_READU32:
    case LBF_BUFFER_WRITEU32:
    case LBF_BUFFER_READF32:
    case LBF_BUFFER_WRITEF32:
    case LBF_BUFFER_READF64:
    case LBF_BUFFER_WRITEF64:
        lower_buffer();
        return;
    default:
        break;
    }

    switch (m_call.bfid)
    {
    case LBF_BIT32_ARSHIFT:
    case LBF_BIT32_BAND:
    case LBF_BIT32_BNOT:
    case LBF_BIT32_BOR:
    case LBF_BIT32_BXOR:
    case LBF_BIT32_BTEST:
    case LBF_BIT32_EXTRACT:
    case LBF_BIT32_LROTATE:
    case LBF_BIT32_LSHIFT:
    case LBF_BIT32_REPLACE:
    case LBF_BIT32_RROTATE:
    case LBF_BIT32_RSHIFT:
    case LBF_BIT32_COUNTLZ:
    case LBF_BIT32_COUNTRZ:
    case LBF_BIT32_EXTRACTK:
    case LBF_BIT32_BYTESWAP:
        lower_bit32();
        return;
    case LBF_VECTOR:
    case LBF_VECTOR_MAGNITUDE:
    case LBF_VECTOR_NORMALIZE:
    case LBF_VECTOR_CROSS:
    case LBF_VECTOR_DOT:
    case LBF_VECTOR_FLOOR:
    case LBF_VECTOR_CEIL:
    case LBF_VECTOR_ABS:
    case LBF_VECTOR_SIGN:
    case LBF_VECTOR_CLAMP:
    case LBF_VECTOR_MIN:
    case LBF_VECTOR_MAX:
    case LBF_VECTOR_LERP:
        lower_vector();
        return;
    default:
        lower_number();
        return;
    }
}

void BuiltinLowering::lower_number()
{
    llvm::Type* doubleTy = m_types.doubleTy;

    // luauF_rad and luauF_deg
    const double rpd = 3.14159265358979323846 / 180.0;

    switch (m_call.bfid)
    {
    case LBF_MATH_ABS:
        return lower_math_unary(llvm::Intrinsic::fabs);
    case LBF_MATH_ACOS:
        return lower_math_unary(llvm::Intrinsic::acos);
    case LBF_MATH_ASIN:
        return lower_math_unary(llvm::Intrinsic::asin);
    case LBF_MATH_ATAN:
        return lower_math_unary(llvm::Intrinsic::atan);
    case LBF_MATH_CEIL:
        return lower_math_unary(llvm::Intrinsic::ceil);
    case LBF_MATH_COSH:
        return lower_math_unary(llvm::Intrinsic::cosh);
    case LBF_MATH_COS:
        return lower_math_unary(llvm::Intrinsic::cos);
    case LBF_MATH_EXP:
        return lower_math_unary(llvm::Intrinsic::exp);
    case LBF_MATH_FLOOR:
        return lower_math_unary(llvm::Intrinsic::floor);
    case LBF_MATH_LOG10:
        return lower_math_unary(llvm::Intrinsic::log10);
    case LBF_MATH_SINH:
        return lower_math_unary(llvm::Intrinsic::sinh);
    case LBF_MATH_SIN:
        return lower_math_unary(llvm::Intrinsic::sin);
    case LBF_MATH_SQRT:
        return lower_math_unary(llvm::Intrinsic::sqrt);
    case LBF_MATH_TANH:
        return lower_math_unary(llvm::Intrinsic::tanh);
    case LBF_MATH_TAN:
        return lower_math_unary(llvm::Intrinsic::tan);
    case LBF_MATH_ROUND:
        return lower_math_unary(llvm::Intrinsic::round);
    case LBF_MATH_DEG:
        return store_number(m_build.CreateFDiv(number(0), fp(doubleTy, rpd)));
    case LBF_MATH_RAD:
        return store_number(m_build.CreateFMul(number(0), fp(doubleTy, rpd)));
    case LBF_MATH_LOG:
    {
        llvm::Value* a = number(0);

        if (m_call.args.size() == 1)
            return store_number(m_build.CreateUnaryIntrinsic(llvm::Intrinsic::log, a));

        const double base = nvalue(m_call.args[1].k);

        if (base == 2.0)
            return store_number(m_build.CreateUnaryIntrinsic(llvm::Intrinsic::log2, a));
        if (base == 10.0)
            return store_number(m_build.CreateUnaryIntrinsic(llvm::Intrinsic::log10, a));

        llvm::Value* logBase = m_build.CreateUnaryIntrinsic(llvm::Intrinsic::log, fp(doubleTy, base));
        return store_number(m_build.CreateFDiv(m_build.CreateUnaryIntrinsic(llvm::Intrinsic::log, a), logBase));
    }
    case LBF_MATH_SIGN:
    {
        llvm::Value* v = number(0);
        llvm::Value* negative = m_build.CreateSelect(m_build.CreateFCmpOLT(v, fp(doubleTy, 0.0)), fp(doubleTy, -1.0), fp(doubleTy, 0.0));
        return store_number(m_build.CreateSelect(m_build.CreateFCmpOGT(v, fp(doubleTy, 0.0)), fp(doubleTy, 1.0), negative));
    }
    case LBF_MATH_ISNAN:
    {
        llvm::Value* v = number(0);
        return store_boolean(m_build.CreateFCmpUNO(v, v));
    }
    case LBF_MATH_ISINF:
    {
        llvm::Value* abs = m_build.CreateUnaryIntrinsic(llvm::Intrinsic::fabs, number(0));
        return store_boolean(m_build.CreateFCmpOEQ(abs, llvm::ConstantFP::getInfinity(doubleTy)));
    }
    case LBF_MATH_ISFINITE:
    {
        llvm::Value* abs = m_build.CreateUnaryIntrinsic(llvm::Intrinsic::fabs, number(0));
        return store_boolean(m_build.CreateFCmpOLT(abs, llvm::ConstantFP::getInfinity(doubleTy)));
    }
    case LBF_MATH_ATAN2:
        return store_number(m_build.CreateBinaryIntrinsic(llvm::Intrinsic::atan2, number(0), number(1)));
    case LBF_MATH_FMOD:
        return store_number(m_build.CreateFRem(number(0), number(1)));
    case LBF_MATH_POW:
        return store_number(m_build.CreateBinaryIntrinsic(llvm::Intrinsic::pow, number(0), number(1)));
    case LBF_MATH_MIN:
        return store_number(fold_select(
            llvm::CmpInst::FCMP_OLT,
            [&](size_t i)
            {
                return number(i);
            }
        ));
    case LBF_MATH_MAX:
        return store_number(fold_select(
            llvm::CmpInst::FCMP_OGT,
            [&](size_t i)
            {
                return number(i);
            }
        ));
    case LBF_MATH_CLAMP:
    {
        llvm::Value* v = number(0);
        llvm::Value* min = number(1);
        llvm::Value* max = number(2);

        // min > max raises an error in the library function
        require(m_build.CreateFCmpOLE(min, max));

        llvm::Value* r = m_build.CreateSelect(m_build.CreateFCmpOLT(v, min), min, v);
        return store_number(m_build.CreateSelect(m_build.CreateFCmpOGT(r, max), max, r));
    }
    case LBF_MATH_LERP:
    {
        llvm::Value* a = number(0);
        llvm::Value* b = number(1);
        llvm::Value* t = number(2);

        llvm::Value* r = m_build.CreateFAdd(a, m_build.CreateFMul(m_build.CreateFSub(b, a), t));
        return store_number(m_build.CreateSelect(m_build.CreateFCmpOEQ(t, fp(doubleTy, 1.0)), b, r));
    }
    default:
        LUAU_ASSERT(!"Unsupported builtin");
        LUAU_UNREACHABLE();
    }
}

void BuiltinLowering::lower_bit32()
{
    llvm::IntegerType* i32Ty = m_types.i32Ty;

    switch (m_call.bfid)
    {
    case LBF_BIT32_BAND:
        return store_unsigned(fold_unsigned(
            [&](llvm::Value* a, llvm::Value* b)
            {
                return m_build.CreateAnd(a, b);
            }
        ));
    case LBF_BIT32_BOR:
        return store_unsigned(fold_unsigned(
            [&](llvm::Value* a, llvm::Value* b)
            {
                return m_build.CreateOr(a, b);
            }
        ));
    case LBF_BIT32_BXOR:
        return store_unsigned(fold_unsigned(
            [&](llvm::Value* a, llvm::Value* b)
            {
                return m_build.CreateXor(a, b);
            }
        ));
    case LBF_BIT32_BTEST:
    {
        llvm::Value* r = fold_unsigned(
            [&](llvm::Value* a, llvm::Value* b)
            {
                return m_build.CreateAnd(a, b);
            }
        );
        return store_boolean(m_build.CreateICmpNE(r, m_build.getInt32(0)));
    }
    case LBF_BIT32_BNOT:
        return store_unsigned(m_build.CreateNot(unsigned_arg(0)));
    case LBF_BIT32_LSHIFT:
    case LBF_BIT32_RSHIFT:
    case LBF_BIT32_ARSHIFT:
    {
        llvm::Value* u = unsigned_arg(0);
        llvm::Value* s = int_arg(1);

        // negative counts and counts past the width are handled by the library function
        require(m_build.CreateICmpULT(s, m_build.getInt32(32)));

        if (m_call.bfid == LBF_BIT32_LSHIFT)
            return store_unsigned(m_build.CreateShl(u, s));
        if (m_call.bfid == LBF_BIT32_RSHIFT)
            return store_unsigned(m_build.CreateLShr(u, s));

        return store_unsigned(m_build.CreateAShr(u, s));
    }
    case LBF_BIT32_LROTATE:
    case LBF_BIT32_RROTATE:
    {
        llvm::Value* u = unsigned_arg(0);
        llvm::Value* s = int_arg(1);

        // funnel shifts take the count modulo the width, like the s & 31 of luauF_lrotate and luauF_rrotate
        const llvm::Intrinsic::ID id = m_call.bfid == LBF_BIT32_LROTATE ? llvm::Intrinsic::fshl : llvm::Intrinsic::fshr;
        return store_unsigned(m_build.CreateIntrinsic(id, {i32Ty}, {u, u, s}));
    }
    case LBF_BIT32_EXTRACT:
    {
        llvm::Value* n = unsigned_arg(0);
        llvm::Value* f = int_arg(1);

        if (m_call.args.size() == 2)
        {
            require(m_build.CreateICmpULT(f, m_build.getInt32(32)));
            return store_unsigned(m_build.CreateAnd(m_build.CreateLShr(n, f), 1));
        }

        llvm::Value* w = int_arg(2);

        // f >= 0 && w > 0 && f + w <= 32, with the sum wrapping like it does in the interpreter
        llvm::Value* valid = m_build.CreateAnd(
            m_build.CreateAnd(m_build.CreateICmpSGE(f, m_build.getInt32(0)), m_build.CreateICmpSGT(w, m_build.getInt32(0))),
            m_build.CreateICmpSLE(m_build.CreateAdd(f, w), m_build.getInt32(32))
        );
        require(valid);

        llvm::Value* m = m_build.CreateNot(m_build.CreateShl(m_build.getInt32(0xfffffffe), shift_count(m_build.CreateSub(w, m_build.getInt32(1)))));
        return store_unsigned(m_build.CreateAnd(m_build.CreateLShr(n, shift_count(f)), m));
    }
    case LBF_BIT32_EXTRACTK:
    {
        llvm::Value* n = unsigned_arg(0);

        // the compiler packs an in-range field and width into the constant
        const int fw = int(nvalue(m_call.args[1].k));
        const uint32_t f = fw & 31;
        const uint32_t m = ~(0xfffffffeu << (fw >> 5));

        return store_unsigned(m_build.CreateAnd(m_build.CreateLShr(n, f), m));
    }
    case LBF_BIT32_REPLACE:
    {
        llvm::Value* n = unsigned_arg(0);
        llvm::Value* v = unsigned_arg(1);
        llvm::Value* f = int_arg(2);
        llvm::Value* m = nullptr;

        if (m_call.args.size() == 3)
        {
            require(m_build.CreateICmpULT(f, m_build.getInt32(32)));
            m = m_build.getInt32(1);
        }
        else
        {
            llvm::Value* w = int_arg(3);

            llvm::Value* valid = m_build.CreateAnd(
                m_build.CreateAnd(m_build.CreateICmpSGE(f, m_build.getInt32(0)), m_build.CreateICmpSGT(w, m_build.getInt32(0))),
                m_build.CreateICmpSLE(m_build.CreateAdd(f, w), m_build.getInt32(32))
            );
            require(valid);

            m = m_build.CreateNot(m_build.CreateShl(m_build.getInt32(0xfffffffe), shift_count(m_build.CreateSub(w, m_build.getInt32(1)))));
        }

        f = shift_count(f);

        // (n & ~(m << f)) | ((v & m) << f)
        llvm::Value* cleared = m_build.CreateAnd(n, m_build.CreateNot(m_build.CreateShl(m, f)));
        return store_unsigned(m_build.CreateOr(cleared, m_build.CreateShl(m_build.CreateAnd(v, m), f)));
    }
    case LBF_BIT32_COUNTLZ:
        return store_unsigned(m_build.CreateBinaryIntrinsic(llvm::Intrinsic::ctlz, unsigned_arg(0), m_build.getFalse()));
    case LBF_BIT32_COUNTRZ:
        return store_unsigned(m_build.CreateBinaryIntrinsic(llvm::Intrinsic::cttz, unsigned_arg(0), m_build.getFalse()));
    case LBF_BIT32_BYTESWAP:
        return store_unsigned(m_build.CreateUnaryIntrinsic(llvm::Intrinsic::bswap, unsigned_arg(0)));
    default:
        LUAU_ASSERT(!"Unsupported builtin");
        LUAU_UNREACHABLE();
    }
}

void BuiltinLowering::lower_vector()
{
    llvm::Type* floatTy = m_types.floatTy;

    auto vectors = [&](size_t i)
    {
        return vector(i);
    };

    switch (m_call.bfid)
    {
    case LBF_VECTOR:
    {
        llvm::Value* z = m_call.args.size() >= 3 ? m_build.CreateFPTrunc(number(2), floatTy) : fp(floatTy, 0.0);

        llvm::Value* v = llvm::PoisonValue::get(m_types.vectorTy);
        v = m_build.CreateInsertElement(v, m_build.CreateFPTrunc(number(0), floatTy), uint64_t(0));
        v = m_build.CreateInsertElement(v, m_build.CreateFPTrunc(number(1), floatTy), uint64_t(1));
        v = m_build.CreateInsertElement(v, z, uint64_t(2));
        return store_vector(v);
    }
    case LBF_VECTOR_MAGNITUDE:
    {
        llvm::Value* v = vector(0);
        llvm::Value* length = m_build.CreateUnaryIntrinsic(llvm::Intrinsic::sqrt, sum_components(m_build.CreateFMul(v, v)));
        return store_number(m_build.CreateFPExt(length, m_types.doubleTy));
    }
    case LBF_VECTOR_NORMALIZE:
    {
        llvm::Value* v = vector(0);
        llvm::Value* length = m_build.CreateUnaryIntrinsic(llvm::Intrinsic::sqrt, sum_components(m_build.CreateFMul(v, v)));
        llvm::Value* invSqrt = m_build.CreateFDiv(fp(floatTy, 1.0), length);
        return store_vector(m_build.CreateFMul(v, splat(invSqrt)));
    }
    case LBF_VECTOR_CROSS:
    {
        llvm::Value* a = vector(0);
        llvm::Value* b = vector(1);

        // a.yzx * b.zxy - a.zxy * b.yzx
        const int yzx[] = {1, 2, 0};
        const int zxy[] = {2, 0, 1};

        llvm::Value* left = m_build.CreateFMul(m_build.CreateShuffleVector(a, yzx), m_build.CreateShuffleVector(b, zxy));
        llvm::Value* right = m_build.CreateFMul(m_build.CreateShuffleVector(a, zxy), m_build.CreateShuffleVector(b, yzx));
        return store_vector(m_build.CreateFSub(left, right));
    }
    case LBF_VECTOR_DOT:
    {
        llvm::Value* dot = sum_components(m_build.CreateFMul(vector(0), vector(1)));
        return store_number(m_build.CreateFPExt(dot, m_types.doubleTy));
    }
    case LBF_VECTOR_FLOOR:
        return store_vector(m_build.CreateUnaryIntrinsic(llvm::Intrinsic::floor, vector(0)));
    case LBF_VECTOR_CEIL:
        return store_vector(m_build.CreateUnaryIntrinsic(llvm::Intrinsic::ceil, vector(0)));
    case LBF_VECTOR_ABS:
        return store_vector(m_build.CreateUnaryIntrinsic(llvm::Intrinsic::fabs, vector(0)));
    case LBF_VECTOR_SIGN:
    {
        llvm::Value* v = vector(0);
        llvm::Value* zero = splat(fp(floatTy, 0.0));
        llvm::Value* negative = m_build.CreateSelect(m_build.CreateFCmpOLT(v, zero), splat(fp(floatTy, -1.0)), zero);
        return store_vector(m_build.CreateSelect(m_build.CreateFCmpOGT(v, zero), splat(fp(floatTy, 1.0)), negative));
    }
    case LBF_VECTOR_MIN:
        return store_vector(fold_select(llvm::CmpInst::FCMP_OLT, vectors));
    case LBF_VECTOR_MAX:
        return store_vector(fold_select(llvm::CmpInst::FCMP_OGT, vectors));
    case LBF_VECTOR_CLAMP:
    {
        llvm::Value* v = vector(0);
        llvm::Value* min = vector(1);
        llvm::Value* max = vector(2);

        require(m_build.CreateAndReduce(m_build.CreateFCmpOLE(min, max)));

        llvm::Value* r = m_build.CreateSelect(m_build.CreateFCmpOLT(v, min), min, v);
        return store_vector(m_build.CreateSelect(m_build.CreateFCmpOGT(r, max), max, r));
    }
    case LBF_VECTOR_LERP:
    {
        llvm::Value* a = vector(0);
        llvm::Value* b = vector(1);
        llvm::Value* t = m_build.CreateFPTrunc(number(2), floatTy);

        llvm::Value* r = m_build.CreateFAdd(a, m_build.CreateFMul(m_build.CreateFSub(b, a), splat(t)));
        return store_vector(m_build.CreateSelect(m_build.CreateFCmpOEQ(t, fp(floatTy, 1.0)), b, r));
    }
    default:
        LUAU_ASSERT(!"Unsupported builtin");
        LUAU_UNREACHABLE();
    }
}

void BuiltinLowering::lower_string_byte()
{
    llvm::Value* ts = m_unit.load_pointer(m_call.args[0].reg);
    llvm::Value* i = int_arg(1);
    llvm::Value* j = m_call.args.size() >= 3 ? int_arg(2) : i;
    llvm::Value* len = m_unit.load_field(m_types.i32Ty, ts, offsetof(TString, len), "len");

    // i >= 1 && j >= i && j <= len && j - i + 1 == nresults
    llvm::Value* valid = m_build.CreateAnd(m_build.CreateICmpSGE(i, m_build.getInt32(1)), m_build.CreateICmpSGE(j, i));
    valid = m_build.CreateAnd(valid, m_build.CreateICmpSLE(j, len));
    valid = m_build.CreateAnd(valid, m_build.CreateICmpEQ(m_build.CreateSub(j, i), m_build.getInt32(m_call.nresults - 1)));
    require(valid);

    llvm::Value* data = m_types.field_ptr(m_build, ts, offsetof(TString, data));
    llvm::Value* first = m_build.CreateGEP(m_types.i8Ty, data, m_build.CreateZExt(m_build.CreateSub(i, m_build.getInt32(1)), m_types.i64Ty));

    // every byte is read before any result is written, because the results may overwrite the arguments
    std::vector<llvm::Value*> bytes;
    for (int k = 0; k < m_call.nresults; k++)
    {
        llvm::Value* byte = m_build.CreateLoad(m_types.i8Ty, m_build.CreateConstGEP1_64(m_types.i8Ty, first, k));
        bytes.push_back(m_build.CreateUIToFP(byte, m_types.doubleTy));
    }

    for (int k = 0; k < m_call.nresults; k++)
        m_unit.store_number(m_call.ra + k, bytes[k]);
}

llvm::Value* BuiltinLowering::buffer_address(const uint64_t size)
{
    llvm::Value* buffer = m_unit.load_pointer(m_call.args[0].reg);
    llvm::Value* offset = m_build.CreateZExt(int_arg(1), m_types.i64Ty);
    llvm::Value* len = m_build.CreateZExt(m_unit.load_header(m_types.i32Ty, buffer, offsetof(Buffer, len), "len"), m_types.i64Ty);

    // uint64_t(unsigned(offset)) + (size - 1) < uint64_t(len), which a negative offset never passes
    require(m_build.CreateICmpULT(m_build.CreateAdd(offset, m_build.getInt64(size - 1)), len));

    return m_build.CreateGEP(m_types.i8Ty, m_types.field_ptr(m_build, buffer, offsetof(Buffer, data)), offset);
}

void BuiltinLowering::lower_buffer()
{
    llvm::Type* type = nullptr;
    bool isSigned = false;
    bool isWrite = false;

    switch (m_call.bfid)
    {
    case LBF_BUFFER_READI8:
        isSigned = true;
        [[fallthrough]];
    case LBF_BUFFER_READU8:
        type = m_types.i8Ty;
        break;
    case LBF_BUFFER_WRITEU8:
        type = m_types.i8Ty;
        isWrite = true;
        break;
    case LBF_BUFFER_READI16:
        isSigned = true;
        [[fallthrough]];
    case LBF_BUFFER_READU16:
        type = m_types.i16Ty;
        break;
    case LBF_BUFFER_WRITEU16:
        type = m_types.i16Ty;
        isWrite = true;
        break;
    case LBF_BUFFER_READI32:
        isSigned = true;
        [[fallthrough]];
    case LBF_BUFFER_READU32:
        type = m_types.i32Ty;
        break;
    case LBF_BUFFER_WRITEU32:
        type = m_types.i32Ty;
        isWrite = true;
        break;
    case LBF_BUFFER_READF32:
        type = m_types.floatTy;
        break;
    case LBF_BUFFER_WRITEF32:
        type = m_types.floatTy;
        isWrite = true;
        break;
    case LBF_BUFFER_READF64:
        type = m_types.doubleTy;
        break;
    case LBF_BUFFER_WRITEF64:
        type = m_types.doubleTy;
        isWrite = true;
        break;
    default:
        LUAU_ASSERT(!"Unsupported builtin");
        LUAU_UNREACHABLE();
    }

    const uint64_t size = type->getPrimitiveSizeInBits() / 8;

    // the value is converted before the bounds check like luauF_writeinteger does, which has no observable effect
    llvm::Value* value = nullptr;

    if (isWrite)
    {
        if (type->isFloatingPointTy())
            value = m_build.CreateFPTrunc(number(2), type);
        else
            value = m_build.CreateTrunc(unsigned_arg(2), type);
    }

    llvm::Value* address = buffer_address(size);

    if (isWrite)
    {
        m_unit.mark_buffer_access(m_build.CreateAlignedStore(value, address, llvm::Align(1)));
        return;
    }

    llvm::LoadInst* loaded = m_build.CreateAlignedLoad(type, address, llvm::Align(1));
    m_unit.mark_buffer_access(loaded);

    if (type->isFloatTy())
        store_number(m_build.CreateFPExt(loaded, m_types.doubleTy));
    else if (type->isDoubleTy())
        store_number(loaded);
    else if (isSigned)
        store_number(m_build.CreateSIToFP(loaded, m_types.doubleTy));
    else
        store_number(m_build.CreateUIToFP(loaded, m_types.doubleTy));
}

} // namespace

int inline_builtin_results(const Proto* proto, const int pcpos, int& count)
{
    BuiltinCall call;

    if (!analyze_call(proto, pcpos, call))
        return -1;

    switch (call.bfid)
    {
    case LBF_ASSERT:
    case LBF_BUFFER_WRITEU8:
    case LBF_BUFFER_WRITEU16:
    case LBF_BUFFER_WRITEU32:
    case LBF_BUFFER_WRITEF32:
    case LBF_BUFFER_WRITEF64:
        count = 0;
        break;
    case LBF_STRING_BYTE:
        count = call.nresults;
        break;
    default:
        // the result is written even when the call asks for none, like the fast functions do
        count = 1;
        break;
    }

    return call.ra;
}

bool lower_builtin(TranslationUnit& unit, const Instruction* pc, const int pcpos, llvm::BasicBlock* fallback)
{
    const Proto* proto = unit.bytecode().proto;
    LUAU_ASSERT(pc == &proto->code[pcpos]);

    BuiltinCall call;

    if (!analyze_call(proto, pcpos, call))
        return false;

    llvm::IRBuilder<>& build = unit.builder();
    const VmTypes& types = unit.types();

    llvm::BasicBlock* checks = unit.new_block("builtin.args");
    build.CreateCondBr(unit.is_safe_env(), checks, fallback, unit.likely());
    build.SetInsertPoint(checks);

    std::vector<TypedOperand> operands;
    for (size_t i = 0; i < call.tags.size(); i++)
    {
        if (call.args[i].reg >= 0 && call.tags[i] != kAnyTag)
            operands.push_back({call.args[i].reg, LBC_TYPE_ANY, call.tags[i]});
    }

    // the argument types are what the builtin expects, not something annotated, so they are never speculated on
    llvm::BasicBlock* compute = unit.new_block("builtin");
    check_operand_tags(unit, pcpos, operands, compute, fallback);
    build.SetInsertPoint(compute);

    BuiltinLowering(unit, call, fallback).lower();

    build.CreateBr(unit.block_at(call.target));
    return true;
}

} // namespace Luau::Jit
