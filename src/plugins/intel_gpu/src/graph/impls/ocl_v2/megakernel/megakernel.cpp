// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// MegaKernel plugin implementation — attempt4 algorithms.
// Multi-dispatch: 6 kernels per layer (proj_qkv, qk_rope, attn, o-res, gate_up, down-res).
// Key techniques: intel_sub_group_block_read, fused RMSNorm, split-K, subgroup attention.

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

// fp16 input embedding → fp32 residual
__kernel void mk_to_f32(const __global half* in, __global float* h, uint n) {
    uint i = get_global_id(0);
    if (i < n) h[i] = convert_float(in[i]);
}

// Balanced QKV projection with fused input RMSNorm
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void mk_proj_qkv(
    const __global float* h_base,
    const __global half* wn, const __global half* qw,
    const __global half* kw, const __global half* vw,
    __global float* qb, __global float* kb, __global float* vb,
    uint wn_off, uint qw_off, uint kw_off, uint vw_off, uint tok_off)
{
    uint sg = get_global_id(0)/SG, l = get_sub_group_local_id(), gr = sg*RPS;
    const __global float* h = h_base + tok_off;
    float rms = sg_rms(h, l), o[RPS];
    if (gr < QDIM) {
        sg_gemv_rms(h, wn+wn_off, rms, qw+qw_off, gr, l, o);
        if (l==0) for (int r=0; r<RPS; r++) qb[gr+r]=o[r];
    } else if (gr < QDIM+KVDIM) {
        uint n = gr-QDIM;
        sg_gemv_rms(h, wn+wn_off, rms, kw+kw_off, n, l, o);
        if (l==0) for (int r=0; r<RPS; r++) kb[n+r]=o[r];
    } else {
        uint n = gr-QDIM-KVDIM;
        sg_gemv_rms(h, wn+wn_off, rms, vw+vw_off, n, l, o);
        if (l==0) for (int r=0; r<RPS; r++) vb[n+r]=o[r];
    }
}

// Q/K norm + RoPE + write new token's K/V to this layer's present cache
// Q/K norm + RoPE. Writes the new token's K/V into the persistent internal KV
// cache at its absolute position (fixed stride CS = MAX_SEQ). The megakernel owns
// the cache, so it never touches OpenVINO's KV-variable buffers.
__kernel void mk_qk_rope(
    __global float* qb, __global float* kb,
    const __global half* qn, const __global half* kn,
    const __global half* rf, int pos,
    __global half* kc, __global half* vc, const __global float* vb,
    uint layer, uint abs_pos, uint CS, uint qn_off, uint kn_off)
{
    uint hd = get_group_id(0), d = get_local_id(0);
    __local float s[HD];
    if (hd < NH) {
        __global float* qh = qb + hd*HD;
        s[d] = qh[d]*qh[d];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint t=HD>>1; t>0; t>>=1) { if (d<t) s[d]+=s[d+t]; barrier(CLK_LOCAL_MEM_FENCE); }
        float iv = rsqrt(s[0]/HD+EPS);
        float val = qh[d]*iv*convert_float(qn[qn_off+d]);
        barrier(CLK_LOCAL_MEM_FENCE);
        qh[d] = val;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (d < HHD) {
            float a=(float)pos*convert_float(rf[d]), c=native_cos(a), sn=native_sin(a);
            float x0=qh[d], x1=qh[d+HHD];
            qh[d]=x0*c-x1*sn; qh[d+HHD]=x1*c+x0*sn;
        }
    } else {
        uint kvh = hd-NH;
        __global float* kh = kb + kvh*HD;
        s[d] = kh[d]*kh[d];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint t=HD>>1; t>0; t>>=1) { if (d<t) s[d]+=s[d+t]; barrier(CLK_LOCAL_MEM_FENCE); }
        float iv = rsqrt(s[0]/HD+EPS);
        float val = kh[d]*iv*convert_float(kn[kn_off+d]);
        barrier(CLK_LOCAL_MEM_FENCE);
        kh[d] = val;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (d < HHD) {
            float a=(float)pos*convert_float(rf[d]), c=native_cos(a), sn=native_sin(a);
            float x0=kh[d], x1=kh[d+HHD];
            kh[d]=x0*c-x1*sn; kh[d+HHD]=x1*c+x0*sn;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        half kval = convert_half(kh[d]);
        half vval = convert_half(vb[kvh*HD+d]);
        ulong cbase = ((ulong)layer*KVH + kvh)*(ulong)CS*HD + (ulong)abs_pos*HD;
        kc[cbase+d] = kval; vc[cbase+d] = vval;
    }
}

// Subgroup attention over the internal KV cache: zero barriers, online softmax.
#define NPL (HD/SG)
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void mk_attn(
    const __global float* qb, const __global half* kc, const __global half* vc,
    int SA, __global half* xn, uint layer, uint CS)
{
    uint h=get_group_id(0), l=get_sub_group_local_id(), kv=h/GQA;
    const __global float* qh = qb + h*HD;
    const float scl = rsqrt((float)HD);
    const ulong base = ((ulong)layer*KVH + kv)*(ulong)CS*HD;
    const __global half* ok = kc;
    const __global half* ov = vc;
    float qr[NPL], acc[NPL];
#pragma unroll
    for (int j=0; j<NPL; j++) { qr[j]=qh[l+SG*j]; acc[j]=0; }
    float m=-INFINITY, ls=0;
    for (int s=0; s<SA; s++) {
        float pa=0;
#pragma unroll
        for (int j=0; j<NPL; j++) pa += qr[j]*convert_float(ok[base+(ulong)s*HD+l+SG*j]);
        float sc=sub_group_reduce_add(pa)*scl;
        float mn=fmax(m,sc), cr=native_exp(m-mn), p=native_exp(sc-mn);
        ls=ls*cr+p;
#pragma unroll
        for (int j=0; j<NPL; j++) acc[j]=acc[j]*cr+p*convert_float(ov[base+(ulong)s*HD+l+SG*j]);
        m=mn;
    }
    float il=1.0f/ls;
#pragma unroll
    for (int j=0; j<NPL; j++) xn[h*HD+l+SG*j]=convert_half(acc[j]*il);
}

// Flash-decoding attention: split the S_past key/value scan across TFD sub-groups
// inside one workgroup. Each sub-group runs the same barrier-free online softmax
// over a contiguous tile of keys, producing a partial (max m, denom ls, acc)
// state; the TFD partials are then merged with a log-sum-exp reduction in local
// memory by sub-group 0. This parallelises the attention scan across the GPU as
// the KV cache grows, where the single-subgroup mk_attn is serial-bound.
// TFD is chosen per step on the host (1 for short context -> identical to
// mk_attn, up to MAX_TFD for long context). Empty tiles (s0>=SA) contribute
// nothing: their m stays -INFINITY so exp(m-mn)->0 in the merge.
#define MAX_TFD 8
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void mk_attn_fd(
    const __global float* qb, const __global half* kc, const __global half* vc,
    int SA, __global half* xn, uint layer, uint CS, uint TFD)
{
    uint h  = get_group_id(0);
    uint sg = get_sub_group_id();
    uint l  = get_sub_group_local_id();
    uint kv = h/GQA;
    const float scl = rsqrt((float)HD);
    const ulong base = ((ulong)layer*KVH + kv)*(ulong)CS*HD;

    uint tile = ((uint)SA + TFD - 1) / TFD;
    uint s0 = sg*tile;
    uint s1 = min(s0 + tile, (uint)SA);

    float qr[NPL], acc[NPL];
#pragma unroll
    for (int j=0; j<NPL; j++) { qr[j]=qb[h*HD + l+SG*j]; acc[j]=0; }
    float m=-INFINITY, ls=0;
    for (uint s=s0; s<s1; s++) {
        float pa=0;
#pragma unroll
        for (int j=0; j<NPL; j++) pa += qr[j]*convert_float(kc[base+(ulong)s*HD+l+SG*j]);
        float sc=sub_group_reduce_add(pa)*scl;
        float mn=fmax(m,sc), cr=native_exp(m-mn), p=native_exp(sc-mn);
        ls=ls*cr+p;
#pragma unroll
        for (int j=0; j<NPL; j++) acc[j]=acc[j]*cr+p*convert_float(vc[base+(ulong)s*HD+l+SG*j]);
        m=mn;
    }

    // Publish this tile's partial state, then merge in sub-group 0.
    __local float lm[MAX_TFD], ll[MAX_TFD];
    __local float la[MAX_TFD][NPL][SG];
    if (l==0) { lm[sg]=m; ll[sg]=ls; }
#pragma unroll
    for (int j=0; j<NPL; j++) la[sg][j][l]=acc[j];
    barrier(CLK_LOCAL_MEM_FENCE);

    if (sg==0) {
        float M=lm[0], L=ll[0], ac[NPL];
#pragma unroll
        for (int j=0; j<NPL; j++) ac[j]=la[0][j][l];
        for (uint t=1; t<TFD; t++) {
            float mn=fmax(M, lm[t]), cr=native_exp(M-mn), p=native_exp(lm[t]-mn);
            L=L*cr+ll[t]*p;
#pragma unroll
            for (int j=0; j<NPL; j++) ac[j]=ac[j]*cr+la[t][j][l]*p;
            M=mn;
        }
        float il=1.0f/L;
#pragma unroll
        for (int j=0; j<NPL; j++) xn[h*HD+l+SG*j]=convert_half(ac[j]*il);
    }
}

// Split-K residual GEMV: h += a . w  (o-proj and down-proj)
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void mk_gemv_sk(
    const __global half* a, const __global half* w, __global float* h_base,
    uint IN, int ksplit, uint w_off, uint h_off)
{
    uint wg=get_group_id(0), sgid=get_sub_group_id(), l=get_sub_group_local_id();
    uint n=wg*RPS, chunk=IN/ksplit, kbeg=sgid*chunk;
    __local float red[16*RPS];
    float acc[RPS];
    for (int r=0; r<RPS; r++) acc[r]=0;
    const __global half* wl = w+w_off;
    for (uint blk=kbeg; blk<kbeg+chunk; blk+=SG*16) {
        const __global ushort* ap=(const __global ushort*)(a+blk);
        float8 xlo=convert_float8(as_half8(intel_sub_group_block_read_us8(ap)));
        float8 xhi=convert_float8(as_half8(intel_sub_group_block_read_us8(ap+SG*8)));
        for (int r=0; r<RPS; r++) {
            const __global ushort* wp=(const __global ushort*)(wl+(ulong)(n+r)*IN+blk);
            float8 ylo=convert_float8(as_half8(intel_sub_group_block_read_us8(wp)));
            float8 yhi=convert_float8(as_half8(intel_sub_group_block_read_us8(wp+SG*8)));
            float8 p=xlo*ylo+xhi*yhi;
            acc[r]+=p.s0+p.s1+p.s2+p.s3+p.s4+p.s5+p.s6+p.s7;
        }
    }
    for (int r=0; r<RPS; r++) {
        float sv=sub_group_reduce_add(acc[r]);
        if (l==0) red[sgid*RPS+r]=sv;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sgid==0 && l<RPS) {
        __global float* h=h_base+h_off;
        float t=h[n+l];
        for (int q=0; q<ksplit; q++) t+=red[q*RPS+l];
        h[n+l]=t;
    }
}

// Gate+Up+SiLU with fused post-attn RMSNorm
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void mk_gate_up(
    const __global float* h_base,
    const __global half* wn, const __global half* gw, const __global half* uw,
    __global half* g,
    uint wn_off, uint gw_off, uint uw_off, uint tok_off)
{
    uint sg=get_global_id(0)/SG, l=get_sub_group_local_id(), n=sg*RPS;
    const __global float* h=h_base+tok_off;
    float rms=sg_rms(h,l), a[RPS], b[RPS];
    sg_gemv_rms(h, wn+wn_off, rms, gw+gw_off, n, l, a);
    sg_gemv_rms(h, wn+wn_off, rms, uw+uw_off, n, l, b);
    if (l==0)
        for (int r=0; r<RPS; r++)
            g[n+r]=convert_half((a[r]/(1.0f+native_exp(-a[r])))*b[r]);
}
)CL";

// ---------------------------------------------------------------------------
// Dispatch constants (tuned on Intel Arc Pro B60)
// ---------------------------------------------------------------------------
static constexpr int  NUM_L = 28, H_DIM = 1024, KVH = 8, HD = 128;
static constexpr int  NH = 16, IM_DIM = 3072;
static constexpr int  QDIM = NH * HD, KVDIM = KVH * HD;
static constexpr int  SG = 16, RPS = 4;
static constexpr int  LW_QKV = 64, LW_GU = 128;
static constexpr int  KS_O = 4, KS_DN = 6;
static constexpr int  MAX_SEQ = 4096;  // capacity of the internal KV cache (per layer/head)
static constexpr int  MAX_TFD = 8;     // max flash-decoding tiles (must match kernel MAX_TFD)
static constexpr int  FD_TOKENS_PER_TILE = 32;  // target KV tokens per flash-decoding tile

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
            OPENVINO_THROW("[MegaKernel] Build failed:\n", std::string(log.begin(), log.end()));
        }

        auto gk = [&](const char* nm) {
            cl_kernel k = clCreateKernel(prog_, nm, &err);
            OPENVINO_ASSERT(err == CL_SUCCESS, "[MegaKernel] clCreateKernel(", nm, "): ", err);
            return k;
        };
        kToF32_  = gk("mk_to_f32");
        kProjQKV_= gk("mk_proj_qkv");
        kRope_   = gk("mk_qk_rope");
        kAttn_   = gk("mk_attn");
        kAttnFd_ = gk("mk_attn_fd");
        kGemvSk_ = gk("mk_gemv_sk");
        kGateUp_ = gk("mk_gate_up");

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

        // Two-model decode-only mode: in the two-model PoC setup the prefill phase
        // is served by a separate, unmodified OpenVINO model and only the decode
        // phase is routed to this MegaKernel. When OV_MEGAKERNEL_DECODE_ONLY=1 we
        // therefore refuse any multi-token (prefill) step so the MegaKernel can
        // never accidentally absorb prefill work and pollute decode measurements.
        // The internal KV cache is instead grown one token at a time by decode
        // steps (see cur_len_ below). Default (unset) keeps the multi-token path
        // enabled so single-model users (e.g. GenAI/Optimum) can still self-prime.
        static const bool decode_only = [] {
            const char* v = std::getenv("OV_MEGAKERNEL_DECODE_ONLY");
            return v && v[0] == '1';
        }();
        OPENVINO_ASSERT(!(decode_only && S_new > 1),
                        "[MegaKernel] OV_MEGAKERNEL_DECODE_ONLY=1 but received a multi-token "
                        "(prefill) step with S=", S_new,
                        ". In two-model mode prefill must run on the regular model; the "
                        "MegaKernel model handles decode (S==1) only.");

        // Length bookkeeping via an internal counter: a multi-token step (S_new > 1)
        // starts a fresh sequence (prefill), a single-token step continues decode.
        // This is fully under our control and immune to OV's padded/stale KV shapes.
        uint S_past = (S_new > 1) ? 0u : cur_len_;
        uint S_total = S_past + S_new;

        for (auto& e : events) strm.wait_for_events({e});

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

        // Flash-decoding attention is on by default; OV_MEGAKERNEL_NO_FLASH=1
        // falls back to the original single-subgroup mk_attn for A/B comparison.
        static const bool use_flash = [] {
            const char* v = std::getenv("OV_MEGAKERNEL_NO_FLASH");
            return !(v && v[0] == '1');
        }();

        { uint n = S_new * H_DIM;
          SM(kToF32_,0,hs); SM(kToF32_,1,oh); SU(kToF32_,2,n);
          enq(kToF32_, n, 256); }

        const size_t gQ=(size_t)((QDIM+2*KVDIM)/RPS)*SG, gGI=(size_t)(IM_DIM/RPS)*SG;
        const size_t gQr=(size_t)(NH+KVH)*HD, gAt=(size_t)NH*SG;
        const size_t gOsk=(size_t)(H_DIM/RPS)*KS_O*SG, lOsk=(size_t)KS_O*SG;
        const size_t gDsk=(size_t)(H_DIM/RPS)*KS_DN*SG, lDsk=(size_t)KS_DN*SG;

        for (uint layer = 0; layer < (uint)NUM_L; layer++) {
            for (uint t = 0; t < S_new; t++) {
                uint tok = t*H_DIM, pos_v = S_past+t;
                int pos = (int)pos_v;
                uint ilo=layer*H_DIM, plo=layer*H_DIM, qno=layer*HD, kno=layer*HD;
                uint qwo=layer*QDIM*H_DIM, kwo=layer*KVDIM*H_DIM, vwo=layer*KVDIM*H_DIM;
                uint owo=layer*H_DIM*QDIM, gwo=layer*IM_DIM*H_DIM;
                uint uwo=layer*IM_DIM*H_DIM, dwo=layer*H_DIM*IM_DIM;

                SM(kProjQKV_,0,oh); SM(kProjQKV_,1,il);
                SM(kProjQKV_,2,qw); SM(kProjQKV_,3,kw); SM(kProjQKV_,4,vw);
                SM(kProjQKV_,5,aQb); SM(kProjQKV_,6,aKb); SM(kProjQKV_,7,aVb);
                SU(kProjQKV_,8,ilo); SU(kProjQKV_,9,qwo);
                SU(kProjQKV_,10,kwo); SU(kProjQKV_,11,vwo); SU(kProjQKV_,12,tok);
                enq(kProjQKV_, gQ, LW_QKV);

                SM(kRope_,0,aQb); SM(kRope_,1,aKb);
                SM(kRope_,2,qn); SM(kRope_,3,kn); SM(kRope_,4,rf);
                SI(kRope_,5,pos);
                SM(kRope_,6,aKC); SM(kRope_,7,aVC); SM(kRope_,8,aVb);
                SU(kRope_,9,layer); SU(kRope_,10,pos_v); SU(kRope_,11,MAX_SEQ);
                SU(kRope_,12,qno); SU(kRope_,13,kno);
                enq(kRope_, gQr, HD);

                int sa=(int)(pos_v+1);
                if (use_flash) {
                    // Flash decoding: split the S_past scan across TFD tiles so
                    // the attention kernel fills the GPU as the KV cache grows.
                    // TFD scales with context (1 for short -> matches mk_attn).
                    uint TFD = (uint)sa / FD_TOKENS_PER_TILE;
                    if (TFD < 1u) TFD = 1u;
                    if (TFD > (uint)MAX_TFD) TFD = (uint)MAX_TFD;
                    SM(kAttnFd_,0,aQb); SM(kAttnFd_,1,aKC); SM(kAttnFd_,2,aVC);
                    SI(kAttnFd_,3,sa); SM(kAttnFd_,4,aXn);
                    SU(kAttnFd_,5,layer); SU(kAttnFd_,6,MAX_SEQ); SU(kAttnFd_,7,TFD);
                    enq(kAttnFd_, (size_t)NH*TFD*SG, (size_t)TFD*SG);
                } else {
                    SM(kAttn_,0,aQb); SM(kAttn_,1,aKC); SM(kAttn_,2,aVC);
                    SI(kAttn_,3,sa); SM(kAttn_,4,aXn);
                    SU(kAttn_,5,layer); SU(kAttn_,6,MAX_SEQ);
                    enq(kAttn_, gAt, SG);
                }

                { uint id=QDIM;
                  SM(kGemvSk_,0,aXn); SM(kGemvSk_,1,ow); SM(kGemvSk_,2,oh);
                  SU(kGemvSk_,3,id); SI(kGemvSk_,4,KS_O);
                  SU(kGemvSk_,5,owo); SU(kGemvSk_,6,tok);
                  enq(kGemvSk_, gOsk, lOsk); }

                SM(kGateUp_,0,oh); SM(kGateUp_,1,pl);
                SM(kGateUp_,2,gw); SM(kGateUp_,3,uw); SM(kGateUp_,4,aGb);
                SU(kGateUp_,5,plo); SU(kGateUp_,6,gwo);
                SU(kGateUp_,7,uwo); SU(kGateUp_,8,tok);
                enq(kGateUp_, gGI, LW_GU);

                { uint id=IM_DIM;
                  SM(kGemvSk_,0,aGb); SM(kGemvSk_,1,dw); SM(kGemvSk_,2,oh);
                  SU(kGemvSk_,3,id); SI(kGemvSk_,4,KS_DN);
                  SU(kGemvSk_,5,dwo); SU(kGemvSk_,6,tok);
                  enq(kGemvSk_, gDsk, lDsk); }
            }
        }
        cur_len_ = S_total;   // commit the new cache length
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
    cl_kernel kToF32_=nullptr, kProjQKV_=nullptr;
    cl_kernel kRope_ =nullptr, kAttn_  =nullptr, kGemvSk_ =nullptr;
    cl_kernel kGateUp_=nullptr;
    cl_kernel kAttnFd_=nullptr;          // flash-decoding attention
    cl_mem mQb_=nullptr, mKb_=nullptr, mVb_=nullptr, mGb_=nullptr, mXn_=nullptr;
    cl_mem mKC_=nullptr, mVC_=nullptr;   // persistent internal KV cache (K, V)
    uint32_t cur_len_ = 0;               // number of tokens currently in the cache
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
