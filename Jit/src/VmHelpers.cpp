// This file is part of the Luau programming language and is licensed under MIT License; see LICENSE.txt for details
#include "VmHelpers.h"

#include "Luau/Bytecode.h"
#include "Luau/BytecodeUtils.h"
#include "Luau/Common.h"

#include "lbuiltins.h"
#include "lclass.h"
#include "ldebug.h"
#include "ldo.h"
#include "lfunc.h"
#include "lgc.h"
#include "lnumutils.h"
#include "lstate.h"
#include "lstring.h"
#include "ltable.h"
#include "ltm.h"
#include "lvm.h"

LUAU_FASTFLAG(LuauBackedgeHeapCheck)
LUAU_FASTFLAG(LuauFastpcall)
LUAU_FASTFLAG(LuauFastpcallInterrupt)
LUAU_FASTFLAG(DebugLuauUserDefinedClassesRuntime)
LUAU_FASTFLAG(LuauPromoteProto)

// helpers are ports of the matching VM_CASE in lvmexecute.cpp; 'pc' is the instruction being executed, and 'savedpc' is
// set to what the interpreter's 'pc' would be when it makes the same protected call
#define VM_FRAME(next) \
    const Instruction insn = *pc; \
    CallInfo* ci = L->ci; \
    ci->savedpc = pc + (next); \
    Closure* cl = clvalue(ci->func); \
    StkId base = L->base; \
    TValue* k = ci->p->k

#define VM_REG(i) (&base[i])
#define VM_KV(i) (LUAU_ASSERT(unsigned(i) < unsigned(L->ci->p->sizek)), &k[i])
#define VM_UV(i) (LUAU_ASSERT(unsigned(i) < unsigned(cl->nupvalues)), &cl->l.uprefs[i])

#define VM_PATCH_OP(pc, op) *const_cast<Instruction*>(pc) = (uint8_t(op) | (0xffffff00u & *(pc)))
#define VM_PATCH_C(pc, slot) *const_cast<Instruction*>(pc) = ((uint8_t(slot) << 24) | (0x00ffffffu & *(pc)))
#define VM_PATCH_E(pc, slot) *const_cast<Instruction*>(pc) = ((uint32_t(slot) << 8) | (0x000000ffu & *(pc)))
#define VM_PATCH_AUX(pc, slot) *const_cast<Instruction*>(pc) = uint32_t(slot)
#define VM_PATCH_AUX_SLOT(pc, k, slot) *const_cast<Instruction*>(pc) = ((k) | (uint32_t(slot) << 16))

namespace Luau::Jit
{

// copy of the static helper in lvmexecute.cpp
static LUAU_NOINLINE void setup_cci(lua_State* L, int nresults, StkId fun)
{
    CallInfo* ci = incr_ci(L);

    ci->func = fun;
    ci->p = getproto(clvalue(fun));
    ci->base = fun + 1;
    ci->top = L->top + LUA_MINSTACK;
    ci->savedpc = NULL;
    ci->flags = 0;
    ci->nresults = nresults;

    L->base = fun + 1;

    luaD_checkstackfornewci(L, LUA_MINSTACK);

    LUAU_ASSERT(ci->top <= L->stack_last);
    LUAU_ASSERT(ttisfunction(ci->func));
}

static void call_c_tm(lua_State* L, const TValue* fn, const TValue* lhs, const TValue* rhs, int res)
{
    // note: it's safe to push arguments past top for complicated reasons (see top of lvmexecute.cpp)
    LUAU_ASSERT(L->top + 3 < L->stack + L->stacksize);
    StkId top = L->top;
    setobj2s(L, top + 0, fn);
    setobj2s(L, top + 1, lhs);
    setobj2s(L, top + 2, rhs);
    L->top = top + 3;

    luaV_callTM(L, 2, res);
}

static bool interrupt(lua_State* L, const Instruction* pc)
{
    if (void (*interrupt)(lua_State*, int) = L->global->cb.interrupt)
    {
        // the interrupt hook is called right before we advance pc
        L->ci->savedpc = pc + 1;
        interrupt(L, -1);

        if (L->status != 0)
        {
            L->ci->savedpc = pc;
            return true;
        }
    }

    return false;
}

const TValue* get_fast_tm(lua_State* L, LuaTable* mt, int event)
{
    return fasttm(L, mt, TMS(event));
}

int execute_interrupt(lua_State* L, const Instruction* pc)
{
    return interrupt(L, pc) ? 1 : 0;
}

int execute_safepoint(lua_State* L, const Instruction* pc)
{
    if (interrupt(L, pc))
        return 1;

    if (FFlag::LuauBackedgeHeapCheck && luaC_needsGC(L))
    {
        L->ci->savedpc = pc;
        luaC_step(L, true);
    }

    return 0;
}

void execute_getglobal(lua_State* L, const Instruction* pc)
{
    VM_FRAME(2);
    StkId ra = VM_REG(LUAU_INSN_A(insn));
    TValue* kv = VM_KV(pc[1]);
    LUAU_ASSERT(ttisstring(kv));

    LuaTable* h = cl->env;
    int slot = LUAU_INSN_C(insn) & h->nodemask8;
    LuaNode* n = &h->node[slot];

    if (LUAU_LIKELY(ttisstring(gkey(n)) && tsvalue(gkey(n)) == tsvalue(kv)) && !ttisnil(gval(n)))
    {
        setobj2s(L, ra, gval(n));
        return;
    }

    TValue g;
    sethvalue(L, &g, h);
    L->cachedslot = slot;
    luaV_gettable(L, &g, kv, ra);
    VM_PATCH_C(pc, L->cachedslot);
}

void execute_setglobal(lua_State* L, const Instruction* pc)
{
    VM_FRAME(2);
    StkId ra = VM_REG(LUAU_INSN_A(insn));
    TValue* kv = VM_KV(pc[1]);
    LUAU_ASSERT(ttisstring(kv));

    LuaTable* h = cl->env;
    int slot = LUAU_INSN_C(insn) & h->nodemask8;
    LuaNode* n = &h->node[slot];

    if (LUAU_LIKELY(ttisstring(gkey(n)) && tsvalue(gkey(n)) == tsvalue(kv) && !ttisnil(gval(n)) && !h->readonly))
    {
        setobj2t(L, gval(n), ra);
        luaC_barriert(L, h, ra);
        return;
    }

    TValue g;
    sethvalue(L, &g, h);
    L->cachedslot = slot;
    luaV_settable(L, &g, kv, ra);
    VM_PATCH_C(pc, L->cachedslot);
}

void execute_setupval(lua_State* L, const Instruction* pc)
{
    VM_FRAME(1);
    StkId ra = VM_REG(LUAU_INSN_A(insn));
    TValue* ur = VM_UV(LUAU_INSN_B(insn));
    UpVal* uv = upvalue(ur);

    setobj(L, uv->v, ra);
    luaC_barrier(L, uv, ra);
}

void execute_closeupvals(lua_State* L, const Instruction* pc)
{
    VM_FRAME(1);
    StkId ra = VM_REG(LUAU_INSN_A(insn));

    if (L->openupval && L->openupval->v >= ra)
        luaF_close(L, ra);
}

void execute_getimport(lua_State* L, const Instruction* pc)
{
    VM_FRAME(2);
    StkId ra = VM_REG(LUAU_INSN_A(insn));
    TValue* kv = VM_KV(LUAU_INSN_D(insn));

    if (!ttisnil(kv) && cl->env->safeenv)
    {
        setobj2s(L, ra, kv);
        return;
    }

    luaV_getimport(L, cl->env, k, ra, pc[1], /* propagatenil= */ false);
}

void execute_gettableks(lua_State* L, const Instruction* pc)
{
    VM_FRAME(2);
    StkId ra = VM_REG(LUAU_INSN_A(insn));
    StkId rb = VM_REG(LUAU_INSN_B(insn));
    TValue* kv = VM_KV(pc[1]);
    LUAU_ASSERT(ttisstring(kv));

    if (LUAU_LIKELY(ttistable(rb)))
    {
        LuaTable* h = hvalue(rb);

        int slot = LUAU_INSN_C(insn) & h->nodemask8;
        LuaNode* n = &h->node[slot];

        if (LUAU_LIKELY(ttisstring(gkey(n)) && tsvalue(gkey(n)) == tsvalue(kv) && !ttisnil(gval(n))))
        {
            setobj2s(L, ra, gval(n));
        }
        else if (!h->metatable)
        {
            const TValue* res = luaH_getstr(h, tsvalue(kv));

            if (res != luaO_nilobject)
                VM_PATCH_C(pc, gval2slot(h, res));

            setobj2s(L, ra, res);
        }
        else
        {
            L->cachedslot = slot;
            luaV_gettable(L, rb, kv, ra);
            VM_PATCH_C(pc, L->cachedslot);
        }

        return;
    }

    if (ttisuserdata(rb))
    {
        if (LuaTable* dispatch = L->global->udatadirectfields[uvalue(rb)->tag])
        {
            int slot = LUAU_INSN_C(insn) & dispatch->nodemask8;
            LuaNode* n = &dispatch->node[slot];

#if LUA_VECTOR_DOUBLE == 1
            DirectFieldResult dfr{L, ra};
            void* resultarg = &dfr;
#else
            void* resultarg = ra;
#endif

            if (LUAU_LIKELY(ttisstring(gkey(n)) && tsvalue(gkey(n)) == tsvalue(kv) && !ttisnil(gval(n))))
            {
                lua_UserdataDirectFieldGet fn = reinterpret_cast<lua_UserdataDirectFieldGet>(pvalue(gval(n)));
                fn(uvalue(rb)->data, resultarg);
                return;
            }

            const TValue* fptr = luaH_getstr(dispatch, tsvalue(kv));
            if (!ttisnil(fptr))
            {
                VM_PATCH_C(pc, gval2slot(dispatch, fptr));
                lua_UserdataDirectFieldGet fn = reinterpret_cast<lua_UserdataDirectFieldGet>(pvalue(fptr));
                fn(uvalue(rb)->data, resultarg);
                return;
            }
        }
    }

    const TValue* fn = 0;
    if (ttisuserdata(rb) && (fn = fasttm(L, uvalue(rb)->metatable, TM_INDEX)) && ttisfunction(fn) && clvalue(fn)->isC)
    {
        L->cachedslot = LUAU_INSN_C(insn);
        call_c_tm(L, fn, rb, kv, LUAU_INSN_A(insn));
        VM_PATCH_C(pc, L->cachedslot);
        return;
    }
    else if (ttisvector(rb))
    {
        // fast-path: quick case-insensitive comparison with "X"/"Y"/"Z"
        const char* name = getstr(tsvalue(kv));
        int ic = (name[0] | ' ') - 'x';

#if LUA_VECTOR_SIZE == 4
        // 'w' is before 'x' in ascii, so ic is -1 when indexing with 'w'
        if (ic == -1)
            ic = 3;
#endif

        if (unsigned(ic) < LUA_VECTOR_SIZE && name[1] == '\0')
        {
            const LUA_VECTOR_TYPE* v = vvalue(rb);
            setnvalue(ra, v[ic]);
            return;
        }

        fn = fasttm(L, L->global->mt[LUA_TVECTOR], TM_INDEX);

        if (fn && ttisfunction(fn) && clvalue(fn)->isC)
        {
            L->cachedslot = LUAU_INSN_C(insn);
            call_c_tm(L, fn, rb, kv, LUAU_INSN_A(insn));
            VM_PATCH_C(pc, L->cachedslot);
            return;
        }
    }
    else if (LUAU_UNLIKELY(FFlag::DebugLuauUserDefinedClassesRuntime && ttisobject(rb)))
    {
        uint8_t slot = LUAU_INSN_C(insn);
        LuauObject* inst = objectvalue(rb);

        if (LUAU_LIKELY(slot < inst->lclass->numberofallmembers && tsvalue(kv) == inst->lclass->offsettomember[slot]))
        {
            setobj2s(L, ra, luaR_lookupmemberatoffset(inst, slot));
            return;
        }

        const TValue* offset = luaH_getstr(inst->lclass->memberstooffset, tsvalue(kv));
        if (ttisnil(offset))
            luaG_missingmembererror(L, rb, kv);

        const uint32_t offsetnum = uint32_t(nvalue(offset));
        setobj2s(L, ra, luaR_lookupmemberatoffset(inst, offsetnum));
        VM_PATCH_C(pc, offsetnum);
        return;
    }

    luaV_gettable(L, rb, kv, ra);
}

void execute_settableks(lua_State* L, const Instruction* pc)
{
    VM_FRAME(2);
    StkId ra = VM_REG(LUAU_INSN_A(insn));
    StkId rb = VM_REG(LUAU_INSN_B(insn));
    TValue* kv = VM_KV(pc[1]);
    LUAU_ASSERT(ttisstring(kv));

    if (LUAU_LIKELY(ttistable(rb)))
    {
        LuaTable* h = hvalue(rb);

        int slot = LUAU_INSN_C(insn) & h->nodemask8;
        LuaNode* n = &h->node[slot];

        if (LUAU_LIKELY(ttisstring(gkey(n)) && tsvalue(gkey(n)) == tsvalue(kv) && !ttisnil(gval(n)) && !h->readonly))
        {
            setobj2t(L, gval(n), ra);
            luaC_barriert(L, h, ra);
        }
        else if (fastnotm(h->metatable, TM_NEWINDEX) && !h->readonly)
        {
            TValue* res = luaH_setstr(L, h, tsvalue(kv));
            VM_PATCH_C(pc, gval2slot(h, res));
            setobj2t(L, res, ra);
            luaC_barriert(L, h, ra);
        }
        else
        {
            L->cachedslot = slot;
            luaV_settable(L, rb, kv, ra);
            VM_PATCH_C(pc, L->cachedslot);
        }

        return;
    }

    const TValue* fn = 0;
    if (ttisuserdata(rb) && (fn = fasttm(L, uvalue(rb)->metatable, TM_NEWINDEX)) && ttisfunction(fn) && clvalue(fn)->isC)
    {
        // note: it's safe to push arguments past top for complicated reasons (see top of lvmexecute.cpp)
        LUAU_ASSERT(L->top + 4 < L->stack + L->stacksize);
        StkId top = L->top;
        setobj2s(L, top + 0, fn);
        setobj2s(L, top + 1, rb);
        setobj2s(L, top + 2, kv);
        setobj2s(L, top + 3, ra);
        L->top = top + 4;

        L->cachedslot = LUAU_INSN_C(insn);
        luaV_callTM(L, 3, -1);
        VM_PATCH_C(pc, L->cachedslot);
        return;
    }

    luaV_settable(L, rb, kv, ra);
}

void execute_gettable(lua_State* L, const Instruction* pc)
{
    VM_FRAME(1);
    StkId ra = VM_REG(LUAU_INSN_A(insn));
    StkId rb = VM_REG(LUAU_INSN_B(insn));
    StkId rc = VM_REG(LUAU_INSN_C(insn));

    if (ttistable(rb) && ttisnumber(rc))
    {
        LuaTable* h = hvalue(rb);

        double indexd = nvalue(rc);
        int index = int(indexd);

        if (LUAU_LIKELY(unsigned(index) - 1 < unsigned(h->sizearray) && !h->metatable && double(index) == indexd))
        {
            setobj2s(L, ra, &h->array[unsigned(index - 1)]);
            return;
        }
    }

    luaV_gettable(L, rb, rc, ra);
}

void execute_settable(lua_State* L, const Instruction* pc)
{
    VM_FRAME(1);
    StkId ra = VM_REG(LUAU_INSN_A(insn));
    StkId rb = VM_REG(LUAU_INSN_B(insn));
    StkId rc = VM_REG(LUAU_INSN_C(insn));

    if (ttistable(rb) && ttisnumber(rc))
    {
        LuaTable* h = hvalue(rb);

        double indexd = nvalue(rc);
        int index = int(indexd);

        if (LUAU_LIKELY(unsigned(index) - 1 < unsigned(h->sizearray) && !h->metatable && !h->readonly && double(index) == indexd))
        {
            setobj2t(L, &h->array[unsigned(index - 1)], ra);
            luaC_barriert(L, h, ra);
            return;
        }
    }

    luaV_settable(L, rb, rc, ra);
}

void execute_gettablen(lua_State* L, const Instruction* pc)
{
    VM_FRAME(1);
    StkId ra = VM_REG(LUAU_INSN_A(insn));
    StkId rb = VM_REG(LUAU_INSN_B(insn));
    int c = LUAU_INSN_C(insn);

    if (ttistable(rb))
    {
        LuaTable* h = hvalue(rb);

        if (LUAU_LIKELY(unsigned(c) < unsigned(h->sizearray) && !h->metatable))
        {
            setobj2s(L, ra, &h->array[c]);
            return;
        }
    }

    TValue n;
    setnvalue(&n, c + 1);
    luaV_gettable(L, rb, &n, ra);
}

void execute_settablen(lua_State* L, const Instruction* pc)
{
    VM_FRAME(1);
    StkId ra = VM_REG(LUAU_INSN_A(insn));
    StkId rb = VM_REG(LUAU_INSN_B(insn));
    int c = LUAU_INSN_C(insn);

    if (ttistable(rb))
    {
        LuaTable* h = hvalue(rb);

        if (LUAU_LIKELY(unsigned(c) < unsigned(h->sizearray) && !h->metatable && !h->readonly))
        {
            setobj2t(L, &h->array[c], ra);
            luaC_barriert(L, h, ra);
            return;
        }
    }

    TValue n;
    setnvalue(&n, c + 1);
    luaV_settable(L, rb, &n, ra);
}

void execute_newclosure(lua_State* L, const Instruction* pc)
{
    VM_FRAME(1);
    StkId ra = VM_REG(LUAU_INSN_A(insn));

    Proto* pv = ci->p->p[LUAU_INSN_D(insn)];
    LUAU_ASSERT(unsigned(LUAU_INSN_D(insn)) < unsigned(ci->p->sizep));

    // note: we save closure to stack early in case the code below wants to capture it by value
    Closure* ncl = luaF_newLclosure(L, pv->nups, cl->env, pv);
    setclvalue(L, ra, ncl);

    for (int ui = 0; ui < pv->nups; ++ui)
    {
        Instruction uinsn = pc[1 + ui];
        LUAU_ASSERT(LUAU_INSN_OP(uinsn) == LOP_CAPTURE);

        switch (LUAU_INSN_A(uinsn))
        {
        case LCT_VAL:
            setobj(L, &ncl->l.uprefs[ui], VM_REG(LUAU_INSN_B(uinsn)));
            break;

        case LCT_REF:
            setupvalue(L, &ncl->l.uprefs[ui], luaF_findupval(L, VM_REG(LUAU_INSN_B(uinsn))));
            break;

        case LCT_UPVAL:
            setobj(L, &ncl->l.uprefs[ui], VM_UV(LUAU_INSN_B(uinsn)));
            break;

        default:
            LUAU_ASSERT(!"Unknown upvalue capture type");
            LUAU_UNREACHABLE();
        }
    }

    ci->savedpc = pc + 1 + pv->nups;
    luaC_checkGC(L);
}

void execute_namecall(lua_State* L, const Instruction* pc)
{
    VM_FRAME(2);
    StkId ra = VM_REG(LUAU_INSN_A(insn));
    StkId rb = VM_REG(LUAU_INSN_B(insn));
    TValue* kv = VM_KV(pc[1]);
    LUAU_ASSERT(ttisstring(kv));

    if (LUAU_LIKELY(ttistable(rb)))
    {
        LuaTable* h = hvalue(rb);
        // note: we can't use nodemask8 here because we need to query the main position of the table, and 8-bit nodemask8 only works
        // for predictive lookups
        LuaNode* n = &h->node[tsvalue(kv)->hash & (sizenode(h) - 1)];

        const TValue* mt = 0;
        const LuaNode* mtn = 0;

        if (ttisstring(gkey(n)) && tsvalue(gkey(n)) == tsvalue(kv) && !ttisnil(gval(n)))
        {
            // note: order of copies allows rb to alias ra+1 or ra
            setobj2s(L, ra + 1, rb);
            setobj2s(L, ra, gval(n));
        }
        else if (
            gnext(n) == 0 && (mt = fasttm(L, hvalue(rb)->metatable, TM_INDEX)) && ttistable(mt) &&
            (mtn = &hvalue(mt)->node[LUAU_INSN_C(insn) & hvalue(mt)->nodemask8]) && ttisstring(gkey(mtn)) && tsvalue(gkey(mtn)) == tsvalue(kv) &&
            !ttisnil(gval(mtn))
        )
        {
            setobj2s(L, ra + 1, rb);
            setobj2s(L, ra, gval(mtn));
        }
        else
        {
            setobj2s(L, ra + 1, rb);
            L->cachedslot = LUAU_INSN_C(insn);
            luaV_gettable(L, rb, kv, ra);
            VM_PATCH_C(pc, L->cachedslot);
            // recompute ra since stack might have been reallocated
            ra = &L->base[LUAU_INSN_A(insn)];
            if (ttisnil(ra))
                luaG_methoderror(L, ra + 1, tsvalue(kv));
        }

        return;
    }

    LuaTable* mt = ttisuserdata(rb) ? uvalue(rb)->metatable : L->global->mt[ttype(rb)];
    const TValue* tmi = 0;

    if (const TValue* fn = fasttm(L, mt, TM_NAMECALL))
    {
        setobj2s(L, ra + 1, rb);
        setobj2s(L, ra, fn);

        L->namecall = tsvalue(kv);
    }
    else if ((tmi = fasttm(L, mt, TM_INDEX)) && ttistable(tmi))
    {
        LuaTable* h = hvalue(tmi);
        int slot = LUAU_INSN_C(insn) & h->nodemask8;
        LuaNode* n = &h->node[slot];

        if (LUAU_LIKELY(ttisstring(gkey(n)) && tsvalue(gkey(n)) == tsvalue(kv) && !ttisnil(gval(n))))
        {
            setobj2s(L, ra + 1, rb);
            setobj2s(L, ra, gval(n));
        }
        else
        {
            setobj2s(L, ra + 1, rb);
            L->cachedslot = slot;
            luaV_gettable(L, rb, kv, ra);
            VM_PATCH_C(pc, L->cachedslot);
            ra = &L->base[LUAU_INSN_A(insn)];
            if (ttisnil(ra))
                luaG_methoderror(L, ra + 1, tsvalue(kv));
        }
    }
    else if (LUAU_UNLIKELY(FFlag::DebugLuauUserDefinedClassesRuntime && ttisobject(rb)))
    {
        uint8_t slot = LUAU_INSN_C(insn);
        LuauObject* inst = objectvalue(rb);

        if (slot < inst->lclass->numberofallmembers && tsvalue(kv) == inst->lclass->offsettomember[slot])
        {
            setobj2s(L, ra + 1, rb);
            setobj2s(L, ra, luaR_lookupmemberatoffset(inst, slot));
        }
        else
        {
            const TValue* offset = luaH_getstr(inst->lclass->memberstooffset, tsvalue(kv));
            if (ttisnil(offset))
                luaG_missingmembererror(L, rb, kv);
            const uint32_t offsetnum = uint32_t(nvalue(offset));
            setobj2s(L, ra + 1, rb);
            setobj2s(L, ra, luaR_lookupmemberatoffset(inst, offsetnum));
            VM_PATCH_C(pc, offsetnum);
        }
    }
    else
    {
        setobj2s(L, ra + 1, rb);
        luaV_gettable(L, rb, kv, ra);
        ra = &L->base[LUAU_INSN_A(insn)];
        if (ttisnil(ra))
            luaG_methoderror(L, ra + 1, tsvalue(kv));
    }
}

int execute_call(lua_State* L, const Instruction* pc)
{
    const Instruction insn = *pc;
    const bool feedback = LUAU_INSN_OP(insn) == LOP_CALLFB;
    const Instruction* next = pc + (feedback ? 2 : 1);

    // the slot is read once, so a call through __call still records a hit after the slot gets sealed below
    const Instruction slot = feedback ? pc[1] : LUAU_INSN_FBSLOT_SEALED;

    CallInfo* ci = L->ci;
    Closure* cl = clvalue(ci->func);
    StkId ra = &L->base[LUAU_INSN_A(insn)];

    int nparams = LUAU_INSN_B(insn) - 1;
    int nresults = LUAU_INSN_C(insn) - 1;

    ci->savedpc = next;

    if (nparams != LUA_MULTRET)
        L->top = ra + 1 + nparams;

    if (slot != LUAU_INSN_FBSLOT_SEALED && !ttisfunction(ra))
        VM_PATCH_AUX(pc + 1, LUAU_INSN_FBSLOT_SEALED);

    // luau_precall resolves __call, reallocates the stack, fills missing arguments, and runs C functions to completion
    int status = luau_precall(L, ra, nresults);

    if (status == PCRLUA)
    {
        if (slot != LUAU_INSN_FBSLOT_SEALED && !luaF_recordhit(L, cl, clvalue(L->ci->func), slot))
            VM_PATCH_AUX(pc + 1, LUAU_INSN_FBSLOT_SEALED);

        return kCallEnter;
    }

    if (slot != LUAU_INSN_FBSLOT_SEALED)
        VM_PATCH_AUX(pc + 1, LUAU_INSN_FBSLOT_SEALED);

    if (status == PCRYIELD)
        return kCallYield;

    // luau_precall leaves top past the results, the interpreter only does that for MULTRET calls
    if (nresults != LUA_MULTRET)
        L->top = L->ci->top;

    return kCallDone;
}

int execute_return(lua_State* L, const Instruction* pc)
{
    const Instruction insn = *pc;
    StkId ra = &L->base[LUAU_INSN_A(insn)]; // note: this can point to L->top if b == LUA_MULTRET
    int b = LUAU_INSN_B(insn) - 1;

    CallInfo* ci = L->ci;
    CallInfo* cip = ci - 1;

    StkId res = ci->func;

    StkId vali = ra;
    StkId valend = (b == LUA_MULTRET) ? L->top : ra + b;

    int nresults = ci->nresults;

    int i;
    for (i = nresults; i != 0 && vali < valend; i--)
        setobj2s(L, res++, vali++);
    while (i-- > 0)
        setnilvalue(res++);

    L->ci = cip;
    L->base = cip->base;
    L->top = (nresults == LUA_MULTRET) ? res : cip->top;

    return (ci->flags & LUA_CALLINFO_RETURN) ? 0 : 1;
}

int execute_equal(lua_State* L, const Instruction* pc)
{
    VM_FRAME(1);
    StkId ra = VM_REG(LUAU_INSN_A(insn));
    StkId rb = VM_REG(pc[1]);

    if (ttype(ra) != ttype(rb))
        return 0;

    // the interpreter calls C __eq metamethods of userdata directly, which luaV_equalval does the same way
    return luaV_equalval(L, ra, rb) == 1;
}

int execute_lessequal(lua_State* L, const Instruction* pc)
{
    VM_FRAME(1);
    StkId ra = VM_REG(LUAU_INSN_A(insn));
    StkId rb = VM_REG(pc[1]);

    if (LUAU_LIKELY(ttisnumber(ra) && ttisnumber(rb)))
        return nvalue(ra) <= nvalue(rb);
    else if (ttisstring(ra) && ttisstring(rb))
        return luaV_strcmp(tsvalue(ra), tsvalue(rb)) <= 0;
    else
        return luaV_lessequal(L, ra, rb) == 1;
}

int execute_lessthan(lua_State* L, const Instruction* pc)
{
    VM_FRAME(1);
    StkId ra = VM_REG(LUAU_INSN_A(insn));
    StkId rb = VM_REG(pc[1]);

    if (LUAU_LIKELY(ttisnumber(ra) && ttisnumber(rb)))
        return nvalue(ra) < nvalue(rb);
    else if (ttisstring(ra) && ttisstring(rb))
        return luaV_strcmp(tsvalue(ra), tsvalue(rb)) < 0;
    else
        return luaV_lessthan(L, ra, rb) == 1;
}

void execute_arith(lua_State* L, const Instruction* pc)
{
    VM_FRAME(1);
    StkId ra = VM_REG(LUAU_INSN_A(insn));

    const TValue* rb = nullptr;
    const TValue* rc = nullptr;

    const auto op = LuauOpcode(LUAU_INSN_OP(insn));

    switch (op)
    {
    case LOP_ADD:
    case LOP_SUB:
    case LOP_MUL:
    case LOP_DIV:
    case LOP_IDIV:
    case LOP_MOD:
    case LOP_POW:
        rb = VM_REG(LUAU_INSN_B(insn));
        rc = VM_REG(LUAU_INSN_C(insn));
        break;
    case LOP_ADDK:
    case LOP_SUBK:
    case LOP_MULK:
    case LOP_DIVK:
    case LOP_IDIVK:
    case LOP_MODK:
    case LOP_POWK:
        rb = VM_REG(LUAU_INSN_B(insn));
        rc = VM_KV(LUAU_INSN_C(insn));
        break;
    case LOP_SUBRK:
    case LOP_DIVRK:
        rb = VM_KV(LUAU_INSN_B(insn));
        rc = VM_REG(LUAU_INSN_C(insn));
        break;
    case LOP_MINUS:
        rb = VM_REG(LUAU_INSN_B(insn));
        rc = rb;
        break;
    default:
        LUAU_ASSERT(!"Unsupported arithmetic opcode");
        LUAU_UNREACHABLE();
    }

    // the interpreter calls C metamethods of userdata directly, luaV_doarithimpl ends up making the same call
    switch (op)
    {
    case LOP_ADD:
    case LOP_ADDK:
        return luaV_doarithimpl<TM_ADD>(L, ra, rb, rc);
    case LOP_SUB:
    case LOP_SUBK:
    case LOP_SUBRK:
        return luaV_doarithimpl<TM_SUB>(L, ra, rb, rc);
    case LOP_MUL:
    case LOP_MULK:
        return luaV_doarithimpl<TM_MUL>(L, ra, rb, rc);
    case LOP_DIV:
    case LOP_DIVK:
    case LOP_DIVRK:
        return luaV_doarithimpl<TM_DIV>(L, ra, rb, rc);
    case LOP_IDIV:
    case LOP_IDIVK:
        return luaV_doarithimpl<TM_IDIV>(L, ra, rb, rc);
    case LOP_MOD:
    case LOP_MODK:
        return luaV_doarithimpl<TM_MOD>(L, ra, rb, rc);
    case LOP_POW:
    case LOP_POWK:
        return luaV_doarithimpl<TM_POW>(L, ra, rb, rc);
    case LOP_MINUS:
        return luaV_doarithimpl<TM_UNM>(L, ra, rb, rc);
    default:
        LUAU_UNREACHABLE();
    }
}

void execute_concat(lua_State* L, const Instruction* pc)
{
    VM_FRAME(1);
    int b = LUAU_INSN_B(insn);
    int c = LUAU_INSN_C(insn);

    // this call may realloc the stack, so the registers are read again below
    luaV_concat(L, c - b + 1, c);

    base = L->base;
    setobj2s(L, VM_REG(LUAU_INSN_A(insn)), base + b);
    luaC_checkGC(L);
}

void execute_length(lua_State* L, const Instruction* pc)
{
    VM_FRAME(1);
    StkId ra = VM_REG(LUAU_INSN_A(insn));
    StkId rb = VM_REG(LUAU_INSN_B(insn));

    if (LUAU_LIKELY(ttistable(rb)) && fastnotm(hvalue(rb)->metatable, TM_LEN))
    {
        setnvalue(ra, cast_num(luaH_getn(hvalue(rb))));
    }
    else if (ttisstring(rb))
    {
        setnvalue(ra, cast_num(tsvalue(rb)->len));
    }
    else
    {
        luaV_dolen(L, ra, rb);
    }
}

void execute_newtable(lua_State* L, const Instruction* pc)
{
    VM_FRAME(2);
    StkId ra = VM_REG(LUAU_INSN_A(insn));
    int b = LUAU_INSN_B(insn);

    sethvalue(L, ra, luaH_new(L, pc[1], b == 0 ? 0 : (1 << (b - 1))));
    luaC_checkGC(L);
}

void execute_duptable(lua_State* L, const Instruction* pc)
{
    VM_FRAME(1);
    StkId ra = VM_REG(LUAU_INSN_A(insn));
    TValue* kv = VM_KV(LUAU_INSN_D(insn));

    sethvalue(L, ra, luaH_clone(L, hvalue(kv)));
    luaC_checkGC(L);
}

void execute_setlist(lua_State* L, const Instruction* pc)
{
    VM_FRAME(2);
    StkId ra = VM_REG(LUAU_INSN_A(insn));
    StkId rb = &base[LUAU_INSN_B(insn)]; // note: this can point to L->top if c == LUA_MULTRET
    int c = LUAU_INSN_C(insn) - 1;
    uint32_t index = pc[1];

    if (c == LUA_MULTRET)
    {
        c = int(L->top - rb);
        L->top = L->ci->top;
    }

    // the interpreter returns from luau_execute here to weaken a bytecode tampering primitive; leaving the table alone
    // is the closest equivalent that keeps the frame consistent
    if (!ttistable(ra))
        return;

    LuaTable* h = hvalue(ra);

    int last = index + c - 1;
    if (last > h->sizearray)
        luaH_resizearray(L, h, last);

    TValue* array = h->array;

    for (int i = 0; i < c; ++i)
        setobj2t(L, &array[index + i - 1], rb + i);

    luaC_barrierfast(L, h);
}

int execute_fornprep(lua_State* L, const Instruction* pc)
{
    VM_FRAME(1);
    StkId ra = VM_REG(LUAU_INSN_A(insn));

    if (!ttisnumber(ra + 0) || !ttisnumber(ra + 1) || !ttisnumber(ra + 2))
        luaV_prepareFORN(L, ra + 0, ra + 1, ra + 2);

    double limit = nvalue(ra + 0);
    double step = nvalue(ra + 1);
    double idx = nvalue(ra + 2);

    // make sure the loop condition is exactly the same as in FORNLOOP so that NaN is handled consistently
    return (step > 0 ? idx <= limit : limit <= idx) ? 0 : 1;
}

void execute_forgprep(lua_State* L, const Instruction* pc)
{
    VM_FRAME(1);
    StkId ra = VM_REG(LUAU_INSN_A(insn));

    const auto op = LuauOpcode(LUAU_INSN_OP(insn));

    if (op == LOP_FORGPREP_INEXT)
    {
        if (cl->env->safeenv && ttistable(ra + 1) && ttisnumber(ra + 2) && nvalue(ra + 2) == 0.0)
        {
            setnilvalue(ra);
            setpvalue(ra + 2, reinterpret_cast<void*>(uintptr_t(0)), LU_TAG_ITERATOR);
        }
        else if (!ttisfunction(ra))
        {
            luaG_typeerror(L, ra, "iterate over");
        }

        return;
    }

    if (op == LOP_FORGPREP_NEXT)
    {
        if (cl->env->safeenv && ttistable(ra + 1) && ttisnil(ra + 2))
        {
            setnilvalue(ra);
            setpvalue(ra + 2, reinterpret_cast<void*>(uintptr_t(0)), LU_TAG_ITERATOR);
        }
        else if (!ttisfunction(ra))
        {
            luaG_typeerror(L, ra, "iterate over");
        }

        return;
    }

    LUAU_ASSERT(op == LOP_FORGPREP);

    if (ttisfunction(ra))
        return;

    LuaTable* mt = ttistable(ra) ? hvalue(ra)->metatable : ttisuserdata(ra) ? uvalue(ra)->metatable : cast_to(LuaTable*, NULL);
    const TValue* fn = fasttm(L, mt, TM_ITER);

    if (FFlag::DebugLuauUserDefinedClassesRuntime && LUAU_UNLIKELY(fn == NULL && ttisobject(ra)))
    {
        fn = luaT_gettmbyobj(L, ra, TM_ITER);
        if (ttisnil(fn))
            luaG_typeerror(L, ra, "iterate over");
    }

    if (fn)
    {
        setobj2s(L, ra + 1, ra);
        setobj2s(L, ra, fn);

        L->top = ra + 2; // func + self arg
        LUAU_ASSERT(L->top <= L->stack_last);

        luaD_call(L, ra, 3);
        L->top = L->ci->top;

        // recompute ra since stack might have been reallocated
        ra = &L->base[LUAU_INSN_A(insn)];

        // protect against __iter returning nil, since nil is used as a marker for builtin iteration in FORGLOOP
        if (ttisnil(ra))
            luaG_typeerror(L, ra, "call");
    }
    else if (fasttm(L, mt, TM_CALL))
    {
        // table or userdata with __call, will be called during FORGLOOP
    }
    else if (ttistable(ra))
    {
        // set up registers for builtin iteration
        setobj2s(L, ra + 1, ra);
        setpvalue(ra + 2, reinterpret_cast<void*>(uintptr_t(0)), LU_TAG_ITERATOR);
        setnilvalue(ra);
    }
    else
    {
        luaG_typeerror(L, ra, "iterate over");
    }
}

int execute_forgloop(lua_State* L, const Instruction* pc)
{
    VM_FRAME(1);
    StkId ra = VM_REG(LUAU_INSN_A(insn));
    uint32_t aux = pc[1];

    // note: ra=nil guarantees ra+1=table and ra+2=userdata because of the setup by FORGPREP* opcodes
    if (ttisnil(ra) && ttistable(ra + 1))
    {
        LuaTable* h = hvalue(ra + 1);
        int index = int(reinterpret_cast<uintptr_t>(pvalue(ra + 2)));

        int sizearray = h->sizearray;

        // clear extra variables since we might have more than two
        // note: while aux encodes ipairs bit, when set we always use 2 variables, so it's safe to check this via a signed comparison
        if (LUAU_UNLIKELY(int(aux) > 2))
            for (int i = 2; i < int(aux); ++i)
                setnilvalue(ra + 3 + i);

        // terminate ipairs-style traversal early when encountering nil
        if (int(aux) < 0 && (unsigned(index) >= unsigned(sizearray) || ttisnil(&h->array[index])))
            return kForgloopExit;

        while (unsigned(index) < unsigned(sizearray))
        {
            TValue* e = &h->array[index];

            if (!ttisnil(e))
            {
                setpvalue(ra + 2, reinterpret_cast<void*>(uintptr_t(index + 1)), LU_TAG_ITERATOR);
                setnvalue(ra + 3, double(index + 1));
                setobj2s(L, ra + 4, e);
                return kForgloopContinue;
            }

            index++;
        }

        int sizenode = 1 << h->lsizenode;

        while (unsigned(index - sizearray) < unsigned(sizenode))
        {
            LuaNode* n = &h->node[index - sizearray];

            if (!ttisnil(gval(n)))
            {
                setpvalue(ra + 2, reinterpret_cast<void*>(uintptr_t(index + 1)), LU_TAG_ITERATOR);
                getnodekey(L, ra + 3, n);
                setobj2s(L, ra + 4, gval(n));
                return kForgloopContinue;
            }

            index++;
        }

        return kForgloopExit;
    }

    // note: it's safe to push arguments past top for complicated reasons (see top of lvmexecute.cpp)
    setobj2s(L, ra + 3 + 2, ra + 2);
    setobj2s(L, ra + 3 + 1, ra + 1);
    setobj2s(L, ra + 3, ra);

    L->top = ra + 3 + 3; // func + 2 args (state and index)
    LUAU_ASSERT(L->top <= L->stack_last);

    // luau_finishop completes the instruction on resume, and expects savedpc to point right past it
    if (luaD_performcally(L, ra + 3, uint8_t(aux)))
        return kForgloopYield;

    L->top = L->ci->top;

    ra = &L->base[LUAU_INSN_A(insn)];

    // copy first variable back into the iteration index
    setobj2s(L, ra + 2, ra + 3);

    return ttisnil(ra + 3) ? kForgloopExit : kForgloopContinue;
}

void execute_getvarargs(lua_State* L, const Instruction* pc)
{
    VM_FRAME(1);
    int b = LUAU_INSN_B(insn) - 1;
    int n = cast_int(base - ci->func) - ci->p->numparams - 1;

    if (b == LUA_MULTRET)
    {
        luaD_checkstack(L, n);
        base = L->base;
        StkId ra = VM_REG(LUAU_INSN_A(insn)); // previous call may change the stack

        for (int j = 0; j < n; j++)
            setobj2s(L, ra + j, base - n + j);

        L->top = ra + n;
        return;
    }

    StkId ra = VM_REG(LUAU_INSN_A(insn));

    for (int j = 0; j < b && j < n; j++)
        setobj2s(L, ra + j, base - n + j);
    for (int j = n; j < b; j++)
        setnilvalue(ra + j);
}

void execute_prepvarargs(lua_State* L, const Instruction* pc)
{
    VM_FRAME(1);
    int numparams = LUAU_INSN_A(insn);

    // all fixed parameters are copied after the top so we need more stack space
    luaD_checkstack(L, cl->stacksize + numparams);
    base = L->base;

    // the caller must have filled extra fixed arguments with nil
    LUAU_ASSERT(cast_int(L->top - base) >= numparams);

    // move fixed parameters to final position
    StkId fixed = base; // first fixed argument
    base = L->top;      // final position of first argument

    for (int i = 0; i < numparams; ++i)
    {
        setobj2s(L, base + i, fixed + i);
        setnilvalue(fixed + i);
    }

    // rewire our stack frame to point to the new base
    L->ci->base = base;
    L->ci->top = base + cl->stacksize;

    L->base = base;
    L->top = L->ci->top;
}

void execute_dupclosure(lua_State* L, const Instruction* pc)
{
    VM_FRAME(1);
    StkId ra = VM_REG(LUAU_INSN_A(insn));
    TValue* kv = VM_KV(LUAU_INSN_D(insn));

    Closure* kcl = clvalue(kv);

    // clone closure if the environment is not shared
    // note: we save closure to stack early in case the code below wants to capture it by value
    Closure* ncl = (kcl->env == cl->env) ? kcl : luaF_newLclosure(L, kcl->nupvalues, cl->env, getproto(kcl));
    setclvalue(L, ra, ncl);

    // see LOP_DUPCLOSURE in lvmexecute.cpp for how the existing closure gets reused
    for (int ui = 0; ui < kcl->nupvalues; ++ui)
    {
        Instruction uinsn = pc[1 + ui];
        LUAU_ASSERT(LUAU_INSN_OP(uinsn) == LOP_CAPTURE);
        LUAU_ASSERT(LUAU_INSN_A(uinsn) == LCT_VAL || LUAU_INSN_A(uinsn) == LCT_UPVAL);

        TValue* uv = (LUAU_INSN_A(uinsn) == LCT_VAL) ? VM_REG(LUAU_INSN_B(uinsn)) : VM_UV(LUAU_INSN_B(uinsn));

        if (ncl == kcl && luaO_rawequalObj(&ncl->l.uprefs[ui], uv))
            continue;

        if (ncl == kcl && kcl->preload == 0)
        {
            ncl = luaF_newLclosure(L, kcl->nupvalues, cl->env, getproto(kcl));
            setclvalue(L, ra, ncl);

            ui = -1; // restart the loop to fill all upvalues
            continue;
        }

        setobj(L, &ncl->l.uprefs[ui], uv);
        luaC_barrier(L, ncl, uv);
    }

    // this is a noop if ncl is newly created or shared successfully, but it has to run after the closure is preloaded for the first time
    ncl->preload = 0;

    if (kcl != ncl)
    {
        ci->savedpc = pc + 1 + kcl->nupvalues;
        luaC_checkGC(L);
    }
}

int execute_fastcall(lua_State* L, const Instruction* pc)
{
    const Instruction insn = *pc;
    const auto op = LuauOpcode(LUAU_INSN_OP(insn));

    CallInfo* ci = L->ci;
    Closure* cl = clvalue(ci->func);
    StkId base = L->base;
    TValue* k = ci->p->k;

    if (!cl->env->safeenv)
        return 0;

    // every variant places the paired CALL 'C' instructions past the first word of the fastcall
    Instruction call = pc[1 + LUAU_INSN_C(insn)];
    LUAU_ASSERT(LUAU_INSN_OP(call) == LOP_CALL);

    StkId ra = VM_REG(LUAU_INSN_A(call));

    int nparams = 0;
    int nresults = LUAU_INSN_C(call) - 1;

    TValue* arg = nullptr;
    TValue* args = nullptr;

    switch (op)
    {
    case LOP_FASTCALL:
        nparams = LUAU_INSN_B(call) - 1;
        nparams = (nparams == LUA_MULTRET) ? int(L->top - ra - 1) : nparams;
        arg = ra + 1;
        args = ra + 2;
        break;
    case LOP_FASTCALL1:
        nparams = 1;
        arg = VM_REG(LUAU_INSN_B(insn));
        break;
    case LOP_FASTCALL2:
        nparams = 2;
        arg = VM_REG(LUAU_INSN_B(insn));
        args = VM_REG(pc[1]);
        break;
    case LOP_FASTCALL2K:
        nparams = 2;
        arg = VM_REG(LUAU_INSN_B(insn));
        args = VM_KV(pc[1]);
        break;
    case LOP_FASTCALL3:
    {
        nparams = 3;
        arg = VM_REG(LUAU_INSN_B(insn));

        // note: it's safe to push arguments past top for complicated reasons (see top of lvmexecute.cpp)
        LUAU_ASSERT(L->top + 2 < L->stack + L->stacksize);
        StkId top = L->top;
        setobj2s(L, top, VM_REG(LUAU_INSN_AUX_A(pc[1])));
        setobj2s(L, top + 1, VM_REG(LUAU_INSN_AUX_B(pc[1])));
        args = top;
        break;
    }
    default:
        LUAU_ASSERT(!"Unsupported fastcall opcode");
        LUAU_UNREACHABLE();
    }

    ci->savedpc = pc + getOpLength(op);

    luau_FastFunction f = luauF_table[LUAU_INSN_A(insn)];
    LUAU_ASSERT(f);

    int n = f(L, ra, arg, nresults, args, nparams);

    if (n < 0)
        return 0;

    if (op == LOP_FASTCALL)
        L->top = (nresults == LUA_MULTRET) ? ra + n : L->ci->top;
    else if (nresults == LUA_MULTRET)
        L->top = ra + n;

    return 1;
}

int execute_fastpcall(lua_State* L, const Instruction* pc)
{
    const Instruction insn = *pc;

    // even with compiler flag enabled, runtime can be safely disabled and will execute the fallback
    if (!FFlag::LuauFastpcall)
        return kFastpcallFallback;

    if (FFlag::LuauFastpcallInterrupt && interrupt(L, pc))
        return kFastpcallYield;

    CallInfo* ci = L->ci;
    Closure* cl = clvalue(ci->func);
    StkId base = L->base;

    int pfid = LUAU_INSN_A(insn);
    int skip = LUAU_INSN_C(insn);

    Instruction call = pc[1 + skip];
    LUAU_ASSERT(LUAU_INSN_OP(call) == LOP_CALL);

    StkId ra = VM_REG(LUAU_INSN_A(call));

    int nparams = LUAU_INSN_B(call) - 1;
    int nresults = LUAU_INSN_C(call) - 1;

    nparams = (nparams == LUA_MULTRET) ? int(L->top - ra - 1) : nparams;

    // fast protected calls are only supported in safe environments and in yieldable contexts
    if (LUAU_UNLIKELY(!cl->env->safeenv) || L->nCcalls > L->baseCcalls)
        return kFastpcallFallback;

    int errfunc = luauPF_table[pfid](L, ra, nparams);

    if (errfunc < 0)
        return kFastpcallFallback;

    ci->savedpc = pc + 1 + skip + 1; // return skips the fallback path

    L->top = ra + 1 + nparams;

    // note: creating a call frame can invalidate 'ra' and other StkId
    luau_pushhandlerci(L, ra, errfunc, nresults);

    // prepare target call, note that we request fewer results from the target as pcalls provide 'status'
    StkId callerfunc = L->ci->base + errfunc;
    int calleeresults = (nresults > 0) ? nresults - 1 : nresults;
    int pr = luau_precall(L, callerfunc, calleeresults);

    if (LUAU_LIKELY(pr == PCRLUA))
    {
        // target Luau function must return so that C continuation can be processed
        L->ci->flags |= LUA_CALLINFO_RETURN;
        return kFastpcallEnter;
    }
    else if (pr == PCRC)
    {
        luau_pospcallsuccess(L);
        return kFastpcallDone;
    }

    LUAU_ASSERT(pr == PCRYIELD);
    return kFastpcallYield;
}

void execute_getudataks(lua_State* L, const Instruction* pc)
{
    VM_FRAME(2);
    StkId rb = VM_REG(LUAU_INSN_B(insn));
    uint32_t aux = pc[1];
    uint32_t kidx = LUAU_INSN_AUX_KV16(aux);
    TValue* kv = VM_KV(kidx);

    if (LUAU_LIKELY(ttisuserdata(rb)))
    {
        int utag = uvalue(rb)->tag;
        lua_UdataDirectAccessData& udatadirect = L->global->udatadirect[utag];
        lua_UserdataDirectAccess onudataindex = udatadirect.index;
        TValue* tm = &udatadirect.indextm;

        if (LUAU_LIKELY(onudataindex != nullptr && !ttisnil(tm)))
        {
            void* udata = uvalue(rb)->data;

            // note: it's safe to push arguments past top for complicated reasons (see top of lvmexecute.cpp)
            LUAU_ASSERT(L->top + 3 < L->stack + L->stacksize);
            StkId top = L->top;
            setobj2s(L, top + 0, tm);
            setobj2s(L, top + 1, rb);
            setobj2s(L, top + 2, kv);
            L->top += 3;

            ++L->nCcalls;

            if (L->nCcalls >= LUAI_MAXCCALLS)
                luaD_checkCstack(L);

            setup_cci(L, 1, top);

            uint16_t cachedslot = LUAU_INSN_AUX_SLOT(aux);
            onudataindex(L, udata, tsvalue(kv)->atom, &cachedslot, utag);

            // update cached slot if instruction didn't deoptimize
            if (cachedslot != LUAU_INSN_AUX_SLOT(aux) && LUAU_INSN_OP(*pc) == LOP_GETUDATAKS)
                VM_PATCH_AUX_SLOT(pc + 1, kidx, cachedslot);

            CallInfo* cci = L->ci;
            CallInfo* cip = cci - 1;

            L->ci = cip;
            L->base = cip->base;
            --L->nCcalls;

            // grab result while L->top is still pointed to the previous function frame
            setobj2s(L, &L->base[LUAU_INSN_A(insn)], L->top - 1);

            L->top = cip->top;
            return;
        }
    }

    // the interpreter backpatches the instruction into a regular table access and executes that
    VM_PATCH_OP(pc, LOP_GETTABLEKS);
    VM_PATCH_AUX_SLOT(pc + 1, kidx, 0);

    execute_gettableks(L, pc);
}

void execute_setudataks(lua_State* L, const Instruction* pc)
{
    VM_FRAME(2);
    StkId ra = VM_REG(LUAU_INSN_A(insn));
    StkId rb = VM_REG(LUAU_INSN_B(insn));
    uint32_t aux = pc[1];
    uint32_t kidx = LUAU_INSN_AUX_KV16(aux);
    TValue* kv = VM_KV(kidx);

    if (LUAU_LIKELY(ttisuserdata(rb)))
    {
        int utag = uvalue(rb)->tag;
        lua_UdataDirectAccessData& udatadirect = L->global->udatadirect[utag];
        lua_UserdataDirectAccess onudatanewindex = udatadirect.newindex;
        TValue* tm = &udatadirect.newindextm;

        if (LUAU_LIKELY(onudatanewindex != nullptr && !ttisnil(tm)))
        {
            void* udata = uvalue(rb)->data;

            // note: it's safe to push arguments past top for complicated reasons (see top of lvmexecute.cpp)
            LUAU_ASSERT(L->top + 4 < L->stack + L->stacksize);
            StkId top = L->top;
            setobj2s(L, top + 0, tm);
            setobj2s(L, top + 1, rb);
            setobj2s(L, top + 2, kv);
            setobj2s(L, top + 3, ra);
            L->top += 4;

            ++L->nCcalls;

            if (L->nCcalls >= LUAI_MAXCCALLS)
                luaD_checkCstack(L);

            setup_cci(L, 0, top);

            uint16_t cachedslot = LUAU_INSN_AUX_SLOT(aux);
            onudatanewindex(L, udata, tsvalue(kv)->atom, &cachedslot, utag);

            if (cachedslot != LUAU_INSN_AUX_SLOT(aux) && LUAU_INSN_OP(*pc) == LOP_SETUDATAKS)
                VM_PATCH_AUX_SLOT(pc + 1, kidx, cachedslot);

            CallInfo* cci = L->ci;
            CallInfo* cip = cci - 1;

            L->ci = cip;
            L->base = cip->base;
            L->top = cip->top;
            --L->nCcalls;
            return;
        }
    }

    VM_PATCH_OP(pc, LOP_SETTABLEKS);
    VM_PATCH_AUX_SLOT(pc + 1, kidx, 0);

    execute_settableks(L, pc);
}

int execute_namecalludata(lua_State* L, const Instruction* pc)
{
    VM_FRAME(2);
    StkId ra = VM_REG(LUAU_INSN_A(insn));
    StkId rb = VM_REG(LUAU_INSN_B(insn));
    uint32_t aux = pc[1];
    uint32_t kidx = LUAU_INSN_AUX_KV16(aux);
    TValue* kv = VM_KV(kidx);

    if (LUAU_LIKELY(ttisuserdata(rb)))
    {
        int utag = uvalue(rb)->tag;
        lua_UdataDirectAccessData& udatadirect = L->global->udatadirect[utag];
        lua_UserdataDirectNamecall onudatanamecall = udatadirect.namecall;
        TValue* tm = &udatadirect.namecalltm;

        if (LUAU_LIKELY(onudatanamecall != nullptr && !ttisnil(tm)))
        {
            void* udata = uvalue(rb)->data;

            // note: order of copies allows rb to alias ra+1 or ra
            setobj2s(L, ra + 1, rb);
            setobj2s(L, ra, tm);

            const Instruction call = pc[2];
            LUAU_ASSERT(LUAU_INSN_OP(call) == LOP_CALL || LUAU_INSN_OP(call) == LOP_CALLFB);
            LUAU_ASSERT(LUAU_INSN_A(call) == LUAU_INSN_A(insn));

            int nparams = LUAU_INSN_B(call) - 1;
            int nresults = LUAU_INSN_C(call) - 1;

            ci->savedpc = pc + 2 + getOpLength(LuauOpcode(LUAU_INSN_OP(call)));
            L->namecall = tsvalue(kv);
            L->top = (nparams == LUA_MULTRET) ? L->top : ra + 1 + nparams;

            // note: namecalls do not increase C call number and allow yielding
            setup_cci(L, nresults, ra);

            LUAU_ASSERT(tsvalue(kv)->atom >= 0);

            uint16_t cachedslot = LUAU_INSN_AUX_SLOT(aux);
            int results = onudatanamecall(L, udata, tsvalue(kv)->atom, &cachedslot, utag);

            if (cachedslot != LUAU_INSN_AUX_SLOT(aux) && LUAU_INSN_OP(*pc) == LOP_NAMECALLUDATA)
                VM_PATCH_AUX_SLOT(pc + 1, kidx, cachedslot);

            if (results < 0)
                return kNamecalludataYield;

            CallInfo* cci = L->ci;
            CallInfo* cip = cci - 1;

            StkId res = cci->func;
            StkId vali = L->top - results;
            StkId valend = L->top;

            int i;
            for (i = nresults; i != 0 && vali < valend; i--)
                setobj2s(L, res++, vali++);
            while (i-- > 0)
                setnilvalue(res++);

            L->ci = cip;
            L->base = cip->base;
            L->top = (nresults == LUA_MULTRET) ? res : cip->top;

            return kNamecalludataDone;
        }
    }

    VM_PATCH_OP(pc, LOP_NAMECALL);
    VM_PATCH_AUX_SLOT(pc + 1, kidx, 0);

    execute_namecall(L, pc);
    return kNamecalludataFallback;
}

void execute_newclass(lua_State* L, const Instruction* pc)
{
    VM_FRAME(2);
    StkId ra = VM_REG(LUAU_INSN_A(insn));
    uint8_t super = LUAU_INSN_B(insn);
    TValue* kv = VM_KV(pc[1]);

    LuauClass* newcls = luaR_cloneclass(L, classvalue(kv));
    setclassvalue(L, ra, newcls);
    newcls->isopen = (LUAU_INSN_C(insn) & 0x1u) != 0; // bottom bit of C is the isopen flag

    if (super != 0xff)
    {
        StkId rb = VM_REG(super);

        if (LUAU_UNLIKELY(!ttisclass(rb)))
            luaG_typeerror(L, rb, "extend");

        luaR_inheritclass(L, newcls, classvalue(rb));
    }
}

void execute_newclassmember(lua_State* L, const Instruction* pc)
{
    VM_FRAME(2);
    StkId ra = VM_REG(LUAU_INSN_A(insn));
    TValue* membername = VM_KV(pc[1]);
    LUAU_ASSERT(ttisstring(membername));
    LUAU_ASSERT(LUAU_INSN_B(insn) == 0);
    StkId rc = VM_REG(LUAU_INSN_C(insn));

    luaR_addclassmember(L, classvalue(ra), tsvalue(membername), rc);
}

void execute_construct(lua_State* L, const Instruction* pc)
{
    VM_FRAME(2);
    StkId ra = VM_REG(LUAU_INSN_A(insn));
    StkId rb = VM_REG(LUAU_INSN_B(insn));

    luaR_tryconstructobject(L, ra, rb, pc[1]);
}

int execute_finconstruct(lua_State* L, const Instruction* pc)
{
    VM_FRAME(1);
    TValue* arg = VM_REG(LUAU_INSN_A(insn));

    if (!ttisobject(arg))
        return 0;

    // class construction from a previous CONSTRUCT was successful, so do the work of the CALL and skip it
    Instruction call = pc[1 + LUAU_INSN_C(insn)];
    LUAU_ASSERT(LUAU_INSN_OP(call) == LOP_CALL);

    StkId ra = VM_REG(LUAU_INSN_A(call));
    int nresults = LUAU_INSN_C(call) - 1;

    StkId res = ra;
    setobj2s(L, res++, arg);

    for (int i = 1; i < nresults; ++i)
        setnilvalue(res++);

    L->top = nresults == LUA_MULTRET ? ra + 1 : L->ci->top;
    return 1;
}

int execute_cmpproto(lua_State* L, const Instruction* pc)
{
    VM_FRAME(2);
    StkId ra = VM_REG(LUAU_INSN_A(insn));
    uint32_t funid = pc[1];

    if (LUAU_UNLIKELY(!ttisfunction(ra)))
        return 1;

    Closure* ccl = clvalue(ra);
    return ccl->isC || ccl->l.p->funid != funid;
}

} // namespace Luau::Jit
