# Profiling and accuracy tools for MegaKernels

Profile before optimizing and after every change. Answer in order: how many launches? how much device time vs wall
time? which ops/phases dominate? how far from the memory roofline? where is the GPU idle?

## Host / framework level
- **e2e numbers**: `MEGAKERNEL_POC/python/e2e_performance_measurement.py` (see SKILL.md step 5).
- **Per-op baseline profile**: compile with `{"PERF_COUNT": True, "CACHE_DIR": ""}`, run a decode step,
  `req.get_profiling_info()` -> group by node name with the layer index stripped (`re.sub(r"layers\.\d+", ...)`),
  sum `real_time` and count. Shows primitives per token and the top costs. Limitation: multi-stage primitives
  (e.g. MoE) only report their last kernel, and idle gaps are invisible.
- **Wall vs host CPU**: wrap `req.infer()` with `time.perf_counter()` and `time.process_time()`. ~100% CPU does NOT prove
  host-bound: NEO busy-spins in `clFinish` (`NEO::waitForTimestampsWithinContainer`). Confirm with `perf`.
- **perf**: `perf_event_paranoid` is restrictive on the dev box; `sudo -n sysctl kernel.perf_event_paranoid=1` (restore
  after). `perf record --control fifo:<ctl>` + writing `enable`/`disable` from Python samples only the decode loop.
  Release `.so`s are stripped (plugin frames are anonymous; NEO/libc symbols resolve).
- **Kernel timeline via LD_PRELOAD shim** (not in repo, ~150 lines to write): interpose `clEnqueueNDRangeKernel`,
  force an event, record kernel name (`clGetKernelInfo`), gws/lws; afterwards read
  `CL_PROFILING_COMMAND_{QUEUED,SUBMIT,START,END}`. Also interpose `dlsym` (oneDNN resolves OpenCL entry points via
  `dlopen`+`dlsym`, bypassing plain interposition). The queue must have profiling enabled - `PERF_COUNT=True` does that
  for the plugin queue. Gives GPU busy vs idle per step, kernel count and per-kernel device time including hidden
  sub-kernels. Export start/stop functions and drive them from Python via `ctypes` to trace only selected steps.
- **oneDNN**: `DNNL_VERBOSE=1` shows the chosen implementation (`jit:gemm:any`...) and timings.
- **OpenVINO GPU debug caps** (`OV_GPU_*` dumps, verbose, per-primitive dumps): need a build with
  `-DENABLE_DEBUG_CAPS=ON -DENABLE_GPU_DEBUG_CAPS=ON` (CI builds have it; check `CMakeCache.txt`). See the `ov-debug`
  skill and `src/plugins/intel_gpu/docs/gpu_debug_utils.md`.

## Standalone (no OpenVINO in the loop)
- `MEGAKERNEL_POC/benchmarks/Qwen3_0_6B/` (`run.sh --device=<i|name> --context-tokens=N --iterations=N`): links the
  megakernel runtime directly, random fixed-seed weights, GPU timestamps on an in-order queue, reports latency,
  effective bandwidth and % of B60 456 GB/s. Copy it for a new model to iterate on kernels in seconds without plugin
  rebuilds (also used for Gemma4: build `bench.cpp` + runtime `.cpp`s with `g++ ... -lOpenCL`).
- Roofline: decode bytes/token = all weights read once + KV read + KV write; `bytes / time` vs 456 GB/s (B60).
  lm_head at 85% roofline was not worth touching; per-layer int4 FCs at 50-74% were.

## In-kernel
- **Barrier timestamps** (Gemma4 `OV_MEGAKERNEL_PROF=1` -> `-DMK_PROF`): `__builtin_IB_read_cycle_counter()` written by
  lane 0 of each WG at arrive / release / start-of-streaming per barrier into a USM buffer; host aggregates per phase:
  span (first start .. last arrival), mean WG busy, barrier latency (release_min - arrive_max), release skew, prologue.
  This is how the phase that limits a persistent kernel is found. Reuse the pattern.
- `research/preloading_gemv/src/ocl/common/inkernelProfile.hcl`: `IN_KERNEL_PROFILE(_BLOCK)` macros with cycle counter +
  `printf` for one WG/worker.
- **ISA / registers**: `research/preloading_gemv/get_kernel_asm.sh <kernel>` or manually
  `IGC_ShaderDumpEnable=1 IGC_DumpToCustomDir=<dir>` (dumps `.asm` per kernel; grep `//.kernel <name>`), add
  `-igc_opts 'VISAOptions=-printregusage'` to build options for register usage. Check spills via
  `CL_KERNEL_SPILL_MEM_SIZE_INTEL` (0x4109), SLM via `CL_KERNEL_LOCAL_MEM_SIZE` (Gemma4 prints them with `OV_MEGAKERNEL_DUMP=1`).
- Debug dumps from inside the persistent kernel (`-DMK_DEBUG`: store per-layer residuals into a USM buffer) compared
  against the slow path (`OV_MEGAKERNEL_DEBUG_CMP=1` prints per-layer residual / attention / per-head cosine).

## Not available on the dev box
unitrace / onetrace / VTune are not installed; do not plan around them without checking.

## Accuracy tools (performance at accuracy is the top priority)
- e2e `decode_only`: `argmatch`/`cos` compare the LAST-TOKEN LOGITS OF THE PROMPT pass (prefill path), plus greedy
  generated text. Decode-path correctness therefore shows only in the text - add a direct check:
- **Teacher-forced logits**: generate greedy tokens with the baseline, feed the SAME tokens to both baseline and MK,
  compare logits per decode step (cosine, max abs diff, argmax). Free-running text diverges chaotically after the first
  near-tie, so it is a weak oracle.
- **Independent oracle** when MK and baseline disagree: CPU plugin with
  `{"INFERENCE_PRECISION_HINT": "f32", "DYNAMIC_QUANTIZATION_GROUP_SIZE": 0}` (GPU f32 may lack impls for fused ops).
  In Gemma4 the baseline GPU (int8 dynamic quant for prompts > 64 tokens) was further from f32 than the MK.
- Use real tokens: random-gaussian `inputs_embeds` saturate soft-caps/norms and hide bugs.
- Bisect with env bit masks / per-pass toggles in the same process against identical inputs (toggle
  `OV_MEGAKERNEL_DISABLE` between compiles - the pass reads it at compile time; always `CACHE_DIR=""`).
- Precision parity audit: weights byte-identical (repack only), activation quant group <= stock
  `DYNAMIC_QUANTIZATION_GROUP_SIZE`, same or wider accumulation type.
