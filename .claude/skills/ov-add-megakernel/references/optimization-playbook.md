# MegaKernel optimization playbook

Measured on Arc Pro B60.
Numbers are evidence for the direction, not guarantees - re-measure on your model/device.

A megakernel's whole purpose is the maximum achievable performance for ITS specific model on ITS specific target hardware - not a generic kernel that merely runs everywhere. Do not generalize across models or devices or add flexibility "just in case"; hard-code shapes/dims/dispatch geometry for the one model and query/tune for the one device you were asked to target, and rip out anything that turns out not to help that specific pair. Do not treat this file as a closed list of allowed techniques. Everything below is evidence from past sessions, not a menu to pick from - profile your own model/hardware, form new hypotheses, and try things not written here. Trial and error against real measurements beats reapplying a technique that happened to work on a different model.

## First: know what you are fighting

- Decode at batch 1 is **launch/submission-bound plus memory-bound**, not compute-bound.
  Gemma4 baseline: 1002 kernels/token, 8.5 ms summed device time, 11.7 ms wall. The 3.3 ms gap is host submission;
  wall time tracked primitive COUNT (~4.6 us of wall per removed primitive), not FLOPs. ~530 of the 1002 kernels were
  norms/element-wise glue moving a few KB each.
- Qwen3-0.6B: baseline ~550 primitives/token; MegaKernel = 1 op (hides ~169 internal enqueues in early versions, 1 launch now).
- Therefore: (1) collapse launches, (2) make every GEMV stream weights near the memory roofline, (3) keep DRAM busy
  through sync points (prefetch/preload), (4) remove host syncs.
- Graph-only rewrites (fusing a few norms/transposes) gave Gemma4 only ~4%; the 1.5x came from custom kernels and
  finally one whole-model persistent kernel.

## Decode techniques that WORKED

### Persistent single-launch decode
- **Grid barrier** (Gemma4 `mk_decode`): NWG co-resident work-groups walk all layers + lm_head in ONE launch.
  `grid_arrive` = WG barrier + one `atomic_fetch_add(release, device scope)`; `grid_wait` = spin on relaxed loads with
  `__builtin_IB_eu_thread_pause(8)` then an acquire load. Counters are never reset: target = `epoch * NWG`, epoch
  incremented per launch (no reset kernel). Bounded spin (`SPIN_MAX`) sets an error flag instead of hanging the GPU.
- **Task system** (Qwen06BPOC, `research/preloading_gemv/src/ocl/taskSystem`): a topologically sorted queue of tasks
  (stage x tile) pulled FIFO by persistent workers; per-(layer, stage) completion counters as semaphores
  (`WaitForSemaphore_block`/`SignalSemaphore_block`). Finer-grained dependencies than a grid barrier
  (a task waits only for the stage it consumes). Worker count tuned (`OV_MEGAKERNEL_MONO_WG`, 32 x LWS 512 on B60):
  more workers = more outstanding loads until shared-cursor atomic contention dominates.
- **Co-residency is mandatory** for both: every spinning WG's producers must be resident or it deadlocks. Derive the
  WG count from the device (`CL_DEVICE_MAX_COMPUTE_UNITS / 8` = Xe cores; Gemma4 uses 2 WGs/Xe-core with 256 GRF,
  4 with 128 GRF) and from per-kernel register/SLM usage; clamp.

### Weight preloading / prefetch
Weights do not fit in cache between layers, and barriers leave DRAM idle. Keep it busy:
- Before each grid barrier, every sub-group prefetches the first `PF_RECS` (4) records of the weight slice it will
  stream in the NEXT phase (`intel_sub_group_block_prefetch_ui8`). Examples in `gemma4ModelKernels.h`: dense gate/up
  prefetched at the end of o_proj, expert/down weights as soon as the top-k experts are known, the next layer's
  q|k|v during the current layer's down projection.
- During attention (few WGs busy, DRAM idle) the idle WGs prefetch the o_proj matrix into L2.
- KV rows and small per-layer tensors (q/k norm gammas, inv_freq) are prefetched before the barrier preceding attention.
- `research/preloading_gemv/gemvOpt/gemvBlock.hcl`: warp-specialized GEMV - load warps stream the next phase tile into
  double-buffered SLM while compute warps consume the current one, synchronized by semaphores. Useful when the GEMV itself has to hide its own load latency.
- Smaller effects: loading the first weight vector into registers before the activation prologue was only ~2 us
  (Gemma4 `-DPRE_G`); in compute-bound prefill, explicit prefetch was neutral and SLM preload regressed (occupancy).
  So: prefetch across sync points in memory-bound phases; do not bother inside XMX GEMM loops.

### Flat, balanced weight streaming
- Concatenate all matrices of a phase into one record stream and give every sub-group of the grid an equal contiguous slice -> all sub-groups stream the same bytes, no tail effect. Slices may split a row block; combine with atomics.
- Small-N matrices (router, KV projections): split-K across sub-groups (Gemma4 FFN: DENSE_KS=4, ROUTER_KS=16, MOE_KS=4, DOWN_KS=8) and reduce through SLM or atomics.

### Deterministic split-K
- Float atomics made every run slightly different and flipped near-tied MoE experts. Use Q32.32 fixed point in 64-bit integer atomics (`atomic_addfx`): exact, order-independent sums (|sum| < 2^31).
- Zero accumulators two phases after their last reader instead of adding barriers (`zero_grid`).

### int4 weight repack (transformation time, host, once)
- Stock layout made 16 lanes gather 16 rows' bytes + scale + zp: ~200 GB/s. Repacked into 16-row x 64-col records
  (`gemma4Repack.h`: 128 dwords weights + 16 dwords f16 scale | u8 zp, 576 B) read with
  `intel_sub_group_block_read8` + `intel_sub_group_block_read`: ~450 GB/s (roofline 456).
- Bytes are only reshuffled -> no accuracy change. The repacked Constant replaces the original (single device copy).
- For fp16 weights in XMX prefill, every tested repack (5 layouts) was slower than row-major `vload16` - repack only when lanes otherwise gather scattered rows/metadata.

### Fusion inside the kernel
- Each WG recomputes the RMS of the (small) hidden vector itself instead of a separate stage + barrier.
- Residual add in the o_proj/down epilogue; SiLU/GELU(gate)*up in the gate/up stage; q/k norm + RoPE (computed from `position_ids` and inv_freq, no cos/sin tables) + KV append in the attention stage.
- Quantize activations once per phase into SLM/global (Gemma4: two int8 planes `hi + lo/254` per 64-group, MORE precise than the stock single int8 plane) and reuse across all GEMVs of the phase; integer dot products (`dot_4x8packed_*`).
- Mask construction (~75 tiny host-bound primitives in Gemma4, 0.85 ms GPU idle/step) -> one kernel or computed in-kernel.

### Attention
- Flash-decoding: split keys across WGs/sub-groups, online softmax partials, combine (Gemma4 uses ticket counters so
  the last arriving unit combines). Keys-per-lane tuned (`OV_MEGAKERNEL_ATT_KPS_S/F`).
- KV cache owned by the MK, `[L][KVH][cap][D]` f16; grow by doubling (copy valid rows).

### Compiler / resources
- `-cl-intel-256-GRF-per-thread` when the kernel spills with 128 GRF (Gemma4 attention did). Check spill/SLM/private
  per kernel with `clGetKernelWorkGroupInfo(k, dev, 0x4109 /*CL_KERNEL_SPILL_MEM_SIZE_INTEL*/ ...)`.
- `-cl-mad-enable`; model dims as `-D` so loops fully unroll; `intel_reqd_sub_group_size(16)`.
- Separate kernels per tile shape beat one adaptive kernel (register footprint is static).

### Host side
- No blocking reads/writes per token; upload per-token scalars as kernel args, read positions on device.
- optimum: cap torch threads (`--torch-threads`, ~cores/4..cores-4). With torch using all cores, host sampling ops were
  throttled 2-3x by contention and MK looked slower than baseline (10.6 vs 8.1 ms) - restored to 5.0 vs 6.5 ms.

## Prefill techniques
- Batch all prompt tokens; per-token decode loop was 0.05-0.2x baseline TTFT.
- XMX GEMM (`intel_sub_group_f16_f16_matrix_mad_k16`): best custom tiles M16/N256/K128 (<= 64 tokens),
  M32/N512/K128 (> 64), as separate kernels. Fuse Q|K|V and gate|up launches.
- oneDNN GPU matmul via OpenCL interop on the existing USM weights (transposed-stride descriptors, no copies, cached
  primitives and arg maps): ~2x over the custom GEMM (`OV_MEGAKERNEL_PREFILL_ONEDNN`). Check with `DNNL_VERBOSE=1`.
- Batched attention: oneDNN `Q*K^T` -> causal softmax kernel -> `P*V`: 3229 tokens 941 -> 605 ms.
- Gemma4 int4: DPAS GEMM directly on repacked records, nibble dequant `(w >> 4j) & 0x000F000F | 0x64006400`,
  activations in "P order", 2D block reads (`intel_sub_group_2d_block_read_16b_32r16x1c`, OOB rows read 0, private dst
  cast to `(__private ushort*)`); lm_head over all rows with token groups on grid dim0 for L2 weight reuse 1462 -> 486 ms. genai pf_x 1.4-1.9x.
- MoE prefill: route tokens into per-expert tiles, block-copy gather, grouped GEMMs, scatter.
- Fusing all per-head RoPE/KV-cache WGs of one token into one WG.

## Techniques which did not work (but may in your case)
- lm_head hand-written GEMV (stock runs at ~85% roofline): 4% slower.
- MoE in Gemma4 baseline: only 0.3 ms total.
- `OV_GPU_QUEUE_TYPE=in-order`, static reshape to [1,1,H] (3% only), taskset/P-core pinning, `clFinish` drain inside execute.
- Prefill: large SLM activation tiles (occupancy loss), N >= 768 output blocks, K256/K64 panels, M24 tiles, adaptive
  M16/M32 kernel, explicit prefetch, sub-group RMS reduction, 4-subgroup/sequence-parallel attention for <= 3k tokens,
  vectorized block IO where the compiler already coalesced, precomputed RoPE tables, separate V-copy kernel,
  fused GQA attention with SLM K/V tiles (slower than oneDNN batched), init-time weight packing for fp16 XMX,
  oneDNN sum/eltwise post-ops (corrupted output / failed on B60).
- Prefill as a fine-grained task system: 0.5x of separate kernels (queue + semaphore overhead per task).

## Where to look next on a new model
1. Per-op profile of the baseline: which ops are many-and-tiny (fuse/launch-collapse) vs big-and-off-roofline (better GEMV).
2. Bytes/token / decode time vs 456 GB/s: if far from roofline, the GEMV streaming or barrier bubbles are the problem.
3. In-kernel barrier timestamps: phase span, barrier latency, release skew, prologue time per phase.
4. Is the GPU idle between launches (cltrace) or inside the kernel (barrier profile)?
