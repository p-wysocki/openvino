// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// MegaKernel plugin implementation — single monokernel decoder.
// The entire Qwen3 decoder (all layers) runs in ONE kernel launch per token: a
// persistent grid walks every layer, synchronising between per-layer stages with
// a grid-wide software barrier. Decode is one launch; prefill loops it per token.
// Key techniques: intel_sub_group_block_read, fused RMSNorm, fused RoPE,
// workgroup-cooperative flash-decoding attention, split-K GEMV.

#include "megakernel.hpp"
#include "intel_gpu/primitives/megakernel.hpp"
#include "megakernel_inst.h"
#include "../primitive_ocl_base.hpp"
#include "intel_gpu/runtime/memory.hpp"
#include "intel_gpu/graph/network.hpp"
#include "ocl/ocl_stream.hpp"
#include "ocl/ocl_engine.hpp"
#include "ocl/ocl_memory.hpp"
#include "ocl/ocl_event.hpp"
#include <CL/cl.h>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <vector>

namespace ov::intel_gpu::ocl {

using cldnn::ocl::ocl_engine;
using cldnn::ocl::ocl_stream;
using cldnn::ocl::ocl_event;
using cldnn::ocl::gpu_buffer;

namespace {

// ---------------------------------------------------------------------------
// Kernel source — embedded attempt4 kernels adapted for plugin tensor layouts
// ---------------------------------------------------------------------------
static const char* kKernelSrc = R"CL(
#pragma OPENCL EXTENSION cl_khr_fp16              : enable
#pragma OPENCL EXTENSION cl_intel_subgroups       : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

#define H     1024
#define QDIM  2048
#define KVDIM 1024
#define HD    128
#define NH    16
#define KVH   8
#define HHD   64
#define GQA   2
#define IM    3072
#define EPS   1e-6f
#define SG    16
#define RPS   4
#define NUM_L 28

// Barrier-free RMS: sub_group_reduce_add over each lane's strided H/SG slice
inline float sg_rms(const __global float* h, uint lane) {
    float ss = 0;
    for (uint k = lane * 16; k < H; k += SG * 16) {
        float16 v = vload16(0, h + k);
        float8  q = v.lo*v.lo + v.hi*v.hi;
        ss += q.s0+q.s1+q.s2+q.s3+q.s4+q.s5+q.s6+q.s7;
    }
    return rsqrt(sub_group_reduce_add(ss) / H + EPS);
}

// GEMV with fused RMS + block reads (SIMD16 message per 256-element strip)
inline void sg_gemv_rms(const __global float* h, const __global half* wn, float rms,
                        const __global half* w, uint base, uint lane, float* out) {
    float acc[RPS];
    for (int r = 0; r < RPS; r++) acc[r] = 0;
    for (uint blk = 0; blk < H; blk += SG * 16) {
        const __global uint*   hp = (const __global uint*)(h  + blk);
        float8 hlo = as_float8(intel_sub_group_block_read8(hp));
        float8 hhi = as_float8(intel_sub_group_block_read8(hp + SG * 8));
        const __global ushort* np = (const __global ushort*)(wn + blk);
        float8 xlo = hlo*rms*convert_float8(as_half8(intel_sub_group_block_read_us8(np)));
        float8 xhi = hhi*rms*convert_float8(as_half8(intel_sub_group_block_read_us8(np + SG*8)));
        for (int r = 0; r < RPS; r++) {
            const __global ushort* wp = (const __global ushort*)(w + (ulong)(base+r)*H + blk);
            float8 ylo = convert_float8(as_half8(intel_sub_group_block_read_us8(wp)));
            float8 yhi = convert_float8(as_half8(intel_sub_group_block_read_us8(wp + SG*8)));
            float8 p = xlo*ylo + xhi*yhi;
            acc[r] += p.s0+p.s1+p.s2+p.s3+p.s4+p.s5+p.s6+p.s7;
        }
    }
    for (int r = 0; r < RPS; r++) out[r] = sub_group_reduce_add(acc[r]);
}

#define NPL (HD/SG)   // head-dim elements handled per subgroup lane (attention)

// ===========================================================================
// MONOKERNEL: the entire 28-layer decoder in a SINGLE kernel launch, per token.
// A persistent grid of MONO_WG workgroups (sized to the device's co-resident
// capacity so a software global barrier cannot deadlock) walks all layers,
// synchronising between the per-layer stages with a sense-reversing global
// barrier. Work within each stage is distributed over the grid's subgroups
// with a grid-stride loop. Reuses sg_rms / sg_gemv_rms from above.
//
// Decode runs one launch (S==1). Prefill launches the same kernel once per
// token (tok_off selects the token's slice of hs/h), so the whole plugin uses
// only this kernel. tok_off is the element offset (t*H) into hs and h.
// ===========================================================================

// Plain subgroup GEMV over an fp16 activation (no fused RMS, no split-K):
// each subgroup fully reduces RPS output rows. Used for o-proj / down-proj.
inline void sg_gemv_f16(const __global half* a, const __global half* w, uint IN,
                        uint base, uint lane, float* out) {
    float acc[RPS];
    for (int r=0; r<RPS; r++) acc[r]=0;
    for (uint blk=0; blk<IN; blk+=SG*16) {
        const __global ushort* ap=(const __global ushort*)(a+blk);
        float8 xlo=convert_float8(as_half8(intel_sub_group_block_read_us8(ap)));
        float8 xhi=convert_float8(as_half8(intel_sub_group_block_read_us8(ap+SG*8)));
        for (int r=0; r<RPS; r++) {
            const __global ushort* wp=(const __global ushort*)(w+(ulong)(base+r)*IN+blk);
            float8 ylo=convert_float8(as_half8(intel_sub_group_block_read_us8(wp)));
            float8 yhi=convert_float8(as_half8(intel_sub_group_block_read_us8(wp+SG*8)));
            float8 p=xlo*ylo+xhi*yhi;
            acc[r]+=p.s0+p.s1+p.s2+p.s3+p.s4+p.s5+p.s6+p.s7;
        }
    }
    for (int r=0; r<RPS; r++) out[r]=sub_group_reduce_add(acc[r]);
}

// Sense-reversing global (grid-wide) barrier. Safe ONLY because the host launches
// exactly MONO_WG workgroups, all guaranteed co-resident on the device (see the
// occupancy probe). gbar[0]=arrival counter, gbar[1]=sense flag. The leader
// (local id 0) of each workgroup participates; the whole workgroup converges via
// the surrounding work-group barriers, which also publish this WG's global stores.
inline void grid_barrier(volatile __global atomic_int* gc,
                         volatile __global atomic_int* gs, uint lid, int WG) {
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    if (lid==0) {
        int gen = atomic_load_explicit(gs, memory_order_relaxed, memory_scope_device);
        if (atomic_fetch_add_explicit(gc, 1, memory_order_acq_rel, memory_scope_device) == WG-1) {
            atomic_store_explicit(gc, 0, memory_order_relaxed, memory_scope_device);
            atomic_store_explicit(gs, gen^1, memory_order_release, memory_scope_device);
        } else {
            while (atomic_load_explicit(gs, memory_order_acquire, memory_scope_device) == gen) {}
        }
    }
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
}

__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void mk_mono(
    const __global half* hs, __global float* h,
    const __global half* wn, const __global half* pn,
    const __global half* qw, const __global half* kw, const __global half* vw,
    const __global half* ow, const __global half* gw, const __global half* uw,
    const __global half* dw,
    const __global half* qn, const __global half* kn, const __global half* rf,
    __global float* qb, __global float* kb, __global float* vb,
    __global half* xn, __global half* gbuf,
    __global half* kc, __global half* vc,
    volatile __global atomic_int* gbar,
    int pos, int WG, uint CS, uint tok_off)
{
    // Select this token's slice of the input embedding / residual stream. For
    // decode tok_off==0; for prefill token t it is t*H.
    hs += tok_off;
    h  += tok_off;

    uint lid = get_local_id(0);
    uint l   = get_sub_group_local_id();
    uint nsg = get_num_sub_groups();
    uint gsg = get_group_id(0)*nsg + get_sub_group_id();   // global subgroup id
    uint NSG = (uint)WG * nsg;                              // total subgroups in grid
    volatile __global atomic_int* gc = gbar;
    volatile __global atomic_int* gs = gbar + 1;

    // SLM for stage-BC flash-decoding: per-subgroup partial softmax state
    // (max, sum, accumulator) merged by subgroup 0. Sized for LWS=256 -> 16 subgroups.
    __local float lsm_m[16], lsm_l[16];
    __local float lsm_a[16][NPL][SG];

    const float scl = rsqrt((float)HD);

    // Stage 0: input embedding fp16 -> fp32 residual stream.
    for (uint i = get_global_id(0); i < H; i += get_global_size(0))
        h[i] = convert_float(hs[i]);
    grid_barrier(gc, gs, lid, WG);

    for (uint layer = 0; layer < NUM_L; layer++) {
        uint wn_off=layer*H,       pn_off=layer*H;
        uint qw_off=layer*QDIM*H,  kw_off=layer*KVDIM*H, vw_off=layer*KVDIM*H;
        uint ow_off=layer*H*QDIM,  gw_off=layer*IM*H,    uw_off=layer*IM*H, dw_off=layer*H*IM;
        uint qn_off=layer*HD,      kn_off=layer*HD;

        // Stage A: fused input RMSNorm + Q/K/V projection.
        for (uint gi = gsg; gi < (QDIM+2*KVDIM)/RPS; gi += NSG) {
            uint gr = gi*RPS;
            float rms = sg_rms(h, l), o[RPS];
            if (gr < QDIM) {
                sg_gemv_rms(h, wn+wn_off, rms, qw+qw_off, gr, l, o);
                if (l==0) for (int r=0;r<RPS;r++) qb[gr+r]=o[r];
            } else if (gr < QDIM+KVDIM) {
                uint n=gr-QDIM;
                sg_gemv_rms(h, wn+wn_off, rms, kw+kw_off, n, l, o);
                if (l==0) for (int r=0;r<RPS;r++) kb[n+r]=o[r];
            } else {
                uint n=gr-QDIM-KVDIM;
                sg_gemv_rms(h, wn+wn_off, rms, vw+vw_off, n, l, o);
                if (l==0) for (int r=0;r<RPS;r++) vb[n+r]=o[r];
            }
        }
        grid_barrier(gc, gs, lid, WG);

        // Stage BC (fused RoPE + flash-decoding attention, single barrier).
        //   * Workgroup wg<NH owns query head wg; its subgroups split the KV scan
        //     and merge partials via SLM (flash-decoding), so attention runs in
        //     parallel across the sequence instead of serially per head. Past
        //     positions come from the KV cache (visible after stage A's barrier);
        //     the current position's K/V are rotated on the fly by subgroup 0.
        //   * Workgroups NH..NH+KVH-1 (idle for attention) write the current token's
        //     K/V to the cache for future steps; that target (cache[pos]) is disjoint
        //     from the past positions attention reads, so there is no race, and it
        //     only has to be visible next step (guaranteed by that step's barrier).
        //   * Workgroups >= NH+KVH idle. All reach the same trailing barrier.
        {
            uint wg = get_group_id(0);
            uint sgl = get_sub_group_id();
            uint nsgl = get_num_sub_groups();
            if (wg < NH) {
                uint hq=wg, kv=hq/GQA;
                // RMSNorm + RoPE of this query head (recomputed per subgroup, cheap;
                // qr[j] holds the rotated q at element l+SG*j).
                __global float* qhs = qb + hq*HD;
                float sq=0, qv[8];
                for (int j=0;j<8;j++){ float x=qhs[l+SG*j]; qv[j]=x; sq+=x*x; }
                float iq=rsqrt(sub_group_reduce_add(sq)/HD+EPS);
                for (int j=0;j<8;j++) qv[j]=qv[j]*iq*convert_float(qn[qn_off+l+SG*j]);
                float qr[NPL];
                for (int j=0;j<4;j++){
                    uint d=l+SG*j;
                    float a=(float)pos*convert_float(rf[d]), c=native_cos(a), sn=native_sin(a);
                    float x0=qv[j], x1=qv[j+4];
                    qr[j]=x0*c-x1*sn; qr[j+4]=x1*c+x0*sn;
                }
                ulong base=((ulong)layer*KVH+kv)*(ulong)CS*HD;
                // Split past positions [0,pos) across the workgroup's subgroups.
                uint tile=((uint)pos + nsgl - 1)/nsgl;
                uint s0=sgl*tile, s1=min(s0+tile,(uint)pos);
                float acc[NPL]; for (int j=0;j<NPL;j++) acc[j]=0;
                float m=-INFINITY, ls=0;
                for (uint s=s0;s<s1;s++){
                    float pa=0;
                    for (int j=0;j<NPL;j++) pa+=qr[j]*convert_float(kc[base+(ulong)s*HD+l+SG*j]);
                    float sc=sub_group_reduce_add(pa)*scl;
                    float mn=fmax(m,sc), cr=native_exp(m-mn), p=native_exp(sc-mn);
                    ls=ls*cr+p;
                    for (int j=0;j<NPL;j++) acc[j]=acc[j]*cr+p*convert_float(vc[base+(ulong)s*HD+l+SG*j]);
                    m=mn;
                }
                if (sgl==0){   // current position handled by subgroup 0 (on-the-fly K/V)
                    __global float* khs = kb + kv*HD;
                    float sk=0, kvv[8];
                    for (int j=0;j<8;j++){ float x=khs[l+SG*j]; kvv[j]=x; sk+=x*x; }
                    float ik=rsqrt(sub_group_reduce_add(sk)/HD+EPS);
                    for (int j=0;j<8;j++) kvv[j]=kvv[j]*ik*convert_float(kn[kn_off+l+SG*j]);
                    float kr[NPL];
                    for (int j=0;j<4;j++){
                        uint d=l+SG*j;
                        float a=(float)pos*convert_float(rf[d]), c=native_cos(a), sn=native_sin(a);
                        float x0=kvv[j], x1=kvv[j+4];
                        kr[j]=x0*c-x1*sn; kr[j+4]=x1*c+x0*sn;
                    }
                    float pa=0;
                    for (int j=0;j<NPL;j++) pa+=qr[j]*kr[j];
                    float sc=sub_group_reduce_add(pa)*scl;
                    float mn=fmax(m,sc), cr=native_exp(m-mn), p=native_exp(sc-mn);
                    ls=ls*cr+p;
                    for (int j=0;j<NPL;j++) acc[j]=acc[j]*cr+p*convert_float(vb[kv*HD+l+SG*j]);
                    m=mn;
                }
                // Publish partials to SLM, merge in subgroup 0.
                if (l==0){ lsm_m[sgl]=m; lsm_l[sgl]=ls; }
                for (int j=0;j<NPL;j++) lsm_a[sgl][j][l]=acc[j];
                barrier(CLK_LOCAL_MEM_FENCE);
                if (sgl==0){
                    float M=lsm_m[0], L=lsm_l[0], ac[NPL];
                    for (int j=0;j<NPL;j++) ac[j]=lsm_a[0][j][l];
                    for (uint t=1;t<nsgl;t++){
                        float mn=fmax(M,lsm_m[t]), cr=native_exp(M-mn), p=native_exp(lsm_m[t]-mn);
                        L=L*cr+lsm_l[t]*p;
                        for (int j=0;j<NPL;j++) ac[j]=ac[j]*cr+lsm_a[t][j][l]*p;
                        M=mn;
                    }
                    float il=1.0f/L;
                    for (int j=0;j<NPL;j++) xn[hq*HD+l+SG*j]=convert_half(ac[j]*il);
                }
            } else if (wg < NH+KVH) {
                // KV head: RMSNorm + RoPE current K, write K/V to cache for future steps.
                uint kvh=wg-NH;
                if (sgl==0) {
                    __global float* kh = kb + kvh*HD;
                    float sq=0, kv2[8];
                    for (int j=0;j<8;j++){ float x=kh[l+SG*j]; kv2[j]=x; sq+=x*x; }
                    float iv=rsqrt(sub_group_reduce_add(sq)/HD+EPS);
                    for (int j=0;j<8;j++) kv2[j]=kv2[j]*iv*convert_float(kn[kn_off+l+SG*j]);
                    float ko[8];
                    for (int j=0;j<8;j++) ko[j]=kv2[j];
                    for (int j=0;j<4;j++){
                        uint d=l+SG*j;
                        float a=(float)pos*convert_float(rf[d]), c=native_cos(a), sn=native_sin(a);
                        float x0=kv2[j], x1=kv2[j+4];
                        ko[j]=x0*c-x1*sn; ko[j+4]=x1*c+x0*sn;
                    }
                    ulong cbase=((ulong)layer*KVH+kvh)*(ulong)CS*HD + (ulong)pos*HD;
                    for (int j=0;j<8;j++){
                        uint e=l+SG*j;
                        kc[cbase+e]=convert_half(ko[j]);
                        vc[cbase+e]=convert_half(vb[kvh*HD+e]);
                    }
                }
            }
        }
        grid_barrier(gc, gs, lid, WG);

        // Stage D: O-projection with residual add (h += xn . Wo). Each subgroup
        // owns disjoint output rows, so the residual RMW needs no atomics.
        for (uint gi=gsg; gi<H/RPS; gi+=NSG){
            uint n=gi*RPS; float o[RPS];
            sg_gemv_f16(xn, ow+ow_off, QDIM, n, l, o);
            if (l==0) for (int r=0;r<RPS;r++) h[n+r]+=o[r];
        }
        grid_barrier(gc, gs, lid, WG);

        // Stage E: fused post-attn RMSNorm + gate/up + SiLU.
        for (uint gi=gsg; gi<IM/RPS; gi+=NSG){
            uint n=gi*RPS;
            float rms=sg_rms(h,l), a[RPS], b[RPS];
            sg_gemv_rms(h, pn+pn_off, rms, gw+gw_off, n, l, a);
            sg_gemv_rms(h, pn+pn_off, rms, uw+uw_off, n, l, b);
            if (l==0) for (int r=0;r<RPS;r++)
                gbuf[n+r]=convert_half((a[r]/(1.0f+native_exp(-a[r])))*b[r]);
        }
        grid_barrier(gc, gs, lid, WG);

        // Stage F: down-projection with residual add (h += g . Wdown).
        for (uint gi=gsg; gi<H/RPS; gi+=NSG){
            uint n=gi*RPS; float o[RPS];
            sg_gemv_f16(gbuf, dw+dw_off, IM, n, l, o);
            if (l==0) for (int r=0;r<RPS;r++) h[n+r]+=o[r];
        }
        grid_barrier(gc, gs, lid, WG);
    }
}
)CL";

// ---------------------------------------------------------------------------
// Dispatch constants (tuned on Intel Arc Pro B60)
// ---------------------------------------------------------------------------
static constexpr int  NUM_L = 28, H_DIM = 1024, KVH = 8, HD = 128;
static constexpr int  NH = 16, IM_DIM = 3072;
static constexpr int  QDIM = NH * HD, KVDIM = KVH * HD;
static constexpr int  SG = 16, RPS = 4;
static constexpr int  MAX_SEQ = 4096;  // capacity of the internal KV cache (per layer/head)
// Monokernel launch geometry. The persistent grid must be small enough that all
// MONO_WG workgroups are co-resident on the device, otherwise the grid-wide
// software barrier deadlocks (Intel gives no cross-workgroup forward-progress
// guarantee). The co-resident cap is kernel-specific: this register- and
// SLM-heavy kernel builds in large-GRF mode, so at LWS=256 only a couple of
// workgroups fit per Xe-core. Sweeping (with the SLM flash-decoding attention)
// showed WG=32 is the sweet spot: enough parallelism for the GEMV stages while
// keeping the ~141 grid barriers cheap; more workgroups only add barrier cost
// (and eventually deadlock past the co-residency cap). Tunable via
// OV_MEGAKERNEL_MONO_WG. Attention needs at least NH+KVH=24 workgroups.
static constexpr int  MONO_WG = 32, MONO_LWS = 256;

// ---------------------------------------------------------------------------
// MegaKernelFastImpl
// ---------------------------------------------------------------------------

// Kernel memory argument descriptor: a device allocation that is either a
// classic OpenCL cl_mem buffer (usm == false) or an Intel USM pointer.
struct KArg {
    const void* ptr = nullptr;
    bool usm = false;
};

class MegaKernelFastImpl : public cldnn::primitive_impl {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::MegaKernelFastImpl)

    MegaKernelFastImpl() = default;
    explicit MegaKernelFastImpl(const cldnn::program_node&, const RuntimeParams&) {}
    // Copy constructor: copy the primitive_impl base subobject so metadata such
    // as m_manager and the dynamic flag are preserved (required by the impl
    // caches in ImplementationsFactory). The OpenCL/runtime members below keep
    // their default null/zero initializers so device state is re-created lazily.
    MegaKernelFastImpl(const MegaKernelFastImpl& other) : cldnn::primitive_impl(other) {}

    [[nodiscard]] std::unique_ptr<cldnn::primitive_impl> clone() const override {
        return std::make_unique<MegaKernelFastImpl>(*this);
    }
    bool is_cpu() const override { return false; }
    void save(BinaryOutputBuffer&) const override {}
    void load(BinaryInputBuffer&) override {}
    void init_kernels(const cldnn::kernels_cache&, const cldnn::kernel_impl_params&) override {}
    void set_arguments(cldnn::primitive_inst&) override {}
    void set_arguments(cldnn::primitive_inst&, cldnn::kernel_arguments_data&) override {}
    std::vector<cldnn::BufferDescriptor> get_internal_buffer_descs(const cldnn::kernel_impl_params&) const override {
        return {};
    }

    void ensure_ready(cldnn::primitive_inst& instance) {
        std::lock_guard<std::mutex> g(mu_);
        if (ready_) return;

        auto& eng = downcast<ocl_engine>(instance.get_network().get_engine());
        ctx_ = eng.get_cl_context().get();
        dev_ = eng.get_cl_device().get();

        cl_int err;
        prog_ = clCreateProgramWithSource(ctx_, 1, &kKernelSrc, nullptr, &err);
        OPENVINO_ASSERT(err == CL_SUCCESS, "[MegaKernel] clCreateProgramWithSource: ", err);
        err = clBuildProgram(prog_, 1, &dev_, "-cl-std=CL2.0", nullptr, nullptr);
        if (err != CL_SUCCESS) {
            size_t n = 0;
            clGetProgramBuildInfo(prog_, dev_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &n);
            std::vector<char> log(n);
            clGetProgramBuildInfo(prog_, dev_, CL_PROGRAM_BUILD_LOG, n, log.data(), nullptr);
            OPENVINO_THROW("[MegaKernel] Build failed:\n",
                           std::string(log.begin(), log.end()));
        }

        kMono_ = clCreateKernel(prog_, "mk_mono", &err);
        OPENVINO_ASSERT(err == CL_SUCCESS, "[MegaKernel] clCreateKernel(mk_mono): ", err);

        auto alloc = [&](size_t bytes) {
            cl_mem m = clCreateBuffer(ctx_, CL_MEM_READ_WRITE, bytes, nullptr, &err);
            OPENVINO_ASSERT(err == CL_SUCCESS, "[MegaKernel] clCreateBuffer: ", err);
            return m;
        };
        mQb_ = alloc(QDIM   * 4);
        mKb_ = alloc(KVDIM  * 4);
        mVb_ = alloc(KVDIM  * 4);
        mGb_ = alloc(IM_DIM * 2);
        mXn_ = alloc(QDIM   * 2);
        // Persistent internal KV cache: [NUM_L, KVH, MAX_SEQ, HD] half, K and V.
        // Owned by the megakernel and reused across decode steps, so attention
        // never touches OpenVINO's KV-variable buffers (which are padded/pooled).
        mKC_ = alloc((size_t)NUM_L * KVH * MAX_SEQ * HD * 2);
        mVC_ = alloc((size_t)NUM_L * KVH * MAX_SEQ * HD * 2);
        // Grid-barrier state for the monokernel: [0]=arrival counter, [1]=sense
        // flag, both zero-initialised. The barrier restores the counter to 0 at
        // every release, so it stays valid across decode launches.
        int bar_init[2] = {0, 0};
        mBar_ = clCreateBuffer(ctx_, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                               sizeof(bar_init), bar_init, &err);
        OPENVINO_ASSERT(err == CL_SUCCESS, "[MegaKernel] clCreateBuffer(bar): ", err);
        ready_ = true;
    }

    cldnn::event::ptr execute(const std::vector<cldnn::event::ptr>& events,
                              cldnn::primitive_inst& instance) override {
        ensure_ready(instance);

        auto& eng = downcast<ocl_engine>(instance.get_network().get_engine());
        const auto& usm_helper = eng.get_usm_helper();
        auto& strm = instance.get_network().get_stream();
        auto& ocls = downcast<ocl_stream>(strm);
        cl_command_queue q = ocls.get_cl_queue().get();

        // A kernel memory argument that may be backed either by a classic
        // OpenCL cl_mem buffer or by an Intel USM allocation. buffer_ptr()
        // returns the cl_mem handle for gpu_buffer and the raw device pointer
        // for gpu_usm; the two require different clSetKernelArg* calls.
        auto marg = [](cldnn::memory& m) -> KArg {
            auto at = m.get_allocation_type();
            bool usm = at == cldnn::allocation_type::usm_device ||
                       at == cldnn::allocation_type::usm_host ||
                       at == cldnn::allocation_type::usm_shared;
            return KArg{m.buffer_ptr(), usm};
        };

        KArg hs = marg(instance.input_memory(0));
        KArg qw = marg(instance.input_memory(5));
        KArg kw = marg(instance.input_memory(6));
        KArg vw = marg(instance.input_memory(7));
        KArg ow = marg(instance.input_memory(8));
        KArg gw = marg(instance.input_memory(9));
        KArg uw = marg(instance.input_memory(10));
        KArg dw = marg(instance.input_memory(11));
        KArg il = marg(instance.input_memory(12));
        KArg pl = marg(instance.input_memory(13));
        KArg qn = marg(instance.input_memory(14));
        KArg kn = marg(instance.input_memory(15));
        KArg rf = marg(instance.input_memory(16));
        KArg oh = marg(instance.output_memory(0));
        // Internal scratch + persistent KV cache (plain cl_mem from ensure_ready).
        KArg aQb{static_cast<void*>(mQb_), false};
        KArg aKb{static_cast<void*>(mKb_), false};
        KArg aVb{static_cast<void*>(mVb_), false};
        KArg aGb{static_cast<void*>(mGb_), false};
        KArg aXn{static_cast<void*>(mXn_), false};
        KArg aKC{static_cast<void*>(mKC_), false};
        KArg aVC{static_cast<void*>(mVC_), false};

        // Number of new tokens this step (dim 1 of hidden_states).
        auto hs_ps = instance.input_memory(0).get_layout().get<ov::PartialShape>();
        uint S_new = (uint)hs_ps[1].get_length();

        // Length bookkeeping. The authoritative sequence position is the first
        // new token's position_ids value (input port 1): 0 at the start of every
        // sequence and P for a decode step at absolute position P. The MegaKernel
        // is deliberately stateless with respect to sequence position — it derives
        // S_past solely from position_ids, exactly like the baseline model, so that
        // enabling the MegaKernel never changes how the model responds to a given
        // set of inputs. A fresh sequence simply arrives with pos==0 and overwrites
        // the internal KV slots from the start; no internal length counter is kept.
        //
        // NOTE / KNOWN LIMITATION (not implemented): the MegaKernel keeps its own
        // internal KV cache (mKC_/mVC_) instead of reading the past_key/past_value
        // inputs (ports 3/4). That internal cache is a PoC kernel-design choice and
        // is not wired to OpenVINO's KV state; correctness relies on every slot
        // being (re)written by this sequence before it is read, which position_ids
        // guarantees. Sharing OpenVINO's KV buffers is left unimplemented.
        for (auto& e : events) strm.wait_for_events({e});   // inputs ready before we read them
        // DIAGNOSTIC TOGGLE: OV_MEGAKERNEL_POSREAD=0 disables the per-token blocking
        // read of position_ids and falls back to an internal accumulator (correct
        // only within a single sequence). Used to isolate whether the per-token
        // host<->device sync of the position read is what serialises execution under
        // framework decode loops. Default (unset) reads position_ids every step.
        static const bool pos_read = [] {
            const char* v = std::getenv("OV_MEGAKERNEL_POSREAD");
            return !(v && v[0] == '0');
        }();
        static thread_local uint acc_len = 0;   // fallback accumulator (diagnostic path)
        uint S_past;
        if (pos_read) {
            const auto& pos_mem = instance.input_memory(1);
            const auto pos_dt = pos_mem.get_layout().data_type;
            int64_t pos0 = -1;
            if (pos_dt == ov::element::i64) {
                pos_mem.copy_to(strm, &pos0, 0, 0, sizeof(int64_t), true);
            } else if (pos_dt == ov::element::i32) {
                int32_t p32 = -1;
                pos_mem.copy_to(strm, &p32, 0, 0, sizeof(int32_t), true);
                pos0 = (int64_t)p32;
            }
            OPENVINO_ASSERT(pos0 >= 0,
                            "[MegaKernel] position_ids (input 1) must be i32/i64 and >= 0; "
                            "the MegaKernel derives its sequence position solely from it.");
            S_past = (uint)pos0;
        } else {
            S_past = (S_new > 1) ? 0u : acc_len;
        }
        acc_len = S_past + S_new;

        auto set_arg = [](cl_kernel k, cl_uint i, size_t sz, const void* p) {
            cl_int r = clSetKernelArg(k, i, sz, p);
            OPENVINO_ASSERT(r == CL_SUCCESS, "[MegaKernel] clSetKernelArg: ", r);
        };
        auto set_mem = [&](cl_kernel k, cl_uint i, const KArg& a) {
            cl_int r;
            if (a.usm) {
                r = usm_helper.set_kernel_arg_mem_pointer(cl::Kernel(k, true), i, a.ptr);
            } else {
                cl_mem m = static_cast<cl_mem>(const_cast<void*>(a.ptr));
                r = clSetKernelArg(k, i, sizeof(cl_mem), &m);
            }
            OPENVINO_ASSERT(r == CL_SUCCESS, "[MegaKernel] set mem arg: ", r);
        };
#define SM(k,i,m)  set_mem(k,i,(m))
#define SU(k,i,v)  { cl_uint _vv=(cl_uint)(v); set_arg(k,i,sizeof(cl_uint),&_vv); }
#define SI(k,i,v)  { cl_int  _vi=(cl_int)(v);  set_arg(k,i,sizeof(cl_int), &_vi); }
        auto enq = [&](cl_kernel k, size_t g, size_t l) {
            cl_int r = clEnqueueNDRangeKernel(q, k, 1, nullptr, &g, &l, 0, nullptr, nullptr);
            OPENVINO_ASSERT(r == CL_SUCCESS, "[MegaKernel] enqueue: ", r);
        };

        // ===== Monokernel path: the whole 28-layer model in ONE launch per token.
        // A single persistent grid runs every stage of every layer, using a
        // grid-wide software barrier for inter-stage synchronisation. Decode
        // (S_new==1) is a single launch; prefill loops the same kernel once per
        // token so the plugin uses only this kernel.
        //
        // The grid MUST NOT oversubscribe: all mono_wg workgroups have to be
        // co-resident or the global barrier deadlocks. The safe count depends on
        // this (register- and SLM-heavy) kernel's occupancy, not the device max,
        // so it is tunable via OV_MEGAKERNEL_MONO_WG (default MONO_WG).
        static const int mono_wg = [] {
            const char* v = std::getenv("OV_MEGAKERNEL_MONO_WG");
            int w = v ? atoi(v) : MONO_WG;
            if (w < 1) w = 1;
            if (w > 160) w = 160;
            return w;
        }();
        cl_kernel k = kMono_;
        SM(k,0,hs);  SM(k,1,oh);
        SM(k,2,il);  SM(k,3,pl);
        SM(k,4,qw);  SM(k,5,kw);  SM(k,6,vw);
        SM(k,7,ow);  SM(k,8,gw);  SM(k,9,uw);  SM(k,10,dw);
        SM(k,11,qn); SM(k,12,kn); SM(k,13,rf);
        SM(k,14,aQb); SM(k,15,aKb); SM(k,16,aVb);
        SM(k,17,aXn); SM(k,18,aGb);
        SM(k,19,aKC); SM(k,20,aVC);
        { cl_mem m = mBar_;
          cl_int r = clSetKernelArg(k, 21, sizeof(cl_mem), &m);
          OPENVINO_ASSERT(r == CL_SUCCESS, "[MegaKernel] set bar arg: ", r); }
        SI(k,23,mono_wg); SU(k,24,MAX_SEQ);
        // One launch per new token; the in-order queue serialises them so token
        // t+1 sees the KV cache written by token t. tok_off selects the token's
        // slice of hs/h (t*H). For decode this loop runs exactly once.
        for (uint t = 0; t < S_new; t++) {
            SI(k,22,(int)(S_past + t));
            SU(k,25,t * H_DIM);
            enq(k, (size_t)mono_wg*MONO_LWS, (size_t)MONO_LWS);
        }
#undef SM
#undef SU
#undef SI
        cl_event marker;
        clEnqueueMarkerWithWaitList(q, 0, nullptr, &marker);
        return std::make_shared<ocl_event>(cl::Event(marker, false), 0ULL);
    }

private:
    std::mutex mu_;
    bool ready_ = false;
    cl_context    ctx_  = nullptr;
    cl_device_id  dev_  = nullptr;
    cl_program    prog_ = nullptr;
    cl_kernel kMono_=nullptr;            // single-launch monokernel (whole model)
    cl_mem mQb_=nullptr, mKb_=nullptr, mVb_=nullptr, mGb_=nullptr, mXn_=nullptr;
    cl_mem mKC_=nullptr, mVC_=nullptr;   // persistent internal KV cache (K, V)
    cl_mem mBar_=nullptr;                // monokernel grid-barrier state
};

}  // namespace

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------
std::unique_ptr<cldnn::primitive_impl> MegaKernelImpl::create_impl(
        const cldnn::program_node& node, const RuntimeParams& params) const {
    OPENVINO_ASSERT(node.is_type<cldnn::megakernel>());
    return std::make_unique<MegaKernelFastImpl>(node, params);
}

}  // namespace ov::intel_gpu::ocl

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::megakernel)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::MegaKernelFastImpl)
