# MegaKernel gotchas

## Build / environment
- **Stale binaries are the #1 time sink.** After any build, confirm the process loads what you built:
  `python -c "import openvino as ov; print(ov.__file__, ov.__version__)"`.
- If you're getting SIGSEGV ABI mismatch: GNU Make does not see version-define changes: after a branch switch/rebase `touch src/core/src/version.cpp`, else the
  core keeps an old hash and Python bindings vs core during a GraphRewrite even with `OV_MEGAKERNEL_DISABLE=1`. Rebuild and reinstall core + gpu plugin + `lib<Name>.so` + pyopenvino together.
- openvino.genai must be rebuilt after every core/plugin rebuild; make sure `import openvino_genai` resolves to the
  locally built tree, not a pip wheel (a mismatched genai segfaults silently).
- Benchmark subprocesses use `CACHE_DIR=""` because the blob cache key ignores `OV_MEGAKERNEL_DISABLE` - with a cache
  the MK run silently reuses the baseline blob. Do the same in every ad-hoc script.
- When copying libs into a dist dir, use absolute paths.
- Release `.so`s are stripped: gdb breakpoints by function/line do not hit; use source analysis + targeted prints, or
  a RelWithDebInfo build.

## Transformation
- The pass runs on every model compiled for GPU (also embedding/vision sub-models of the same pipeline): fingerprint +
  full match before any mutation, otherwise return `false` with the graph untouched.
- Snapshot `output.get_target_inputs()` BEFORE creating new consumers on that output, else you rewire your own node (cycle).
- Name-based weight lookup is brittle across exporter versions; assert loudly when a lookup fails.
- Do not use `Equal`/`Select` for shape/length math in the graph (bool eltwise breaks shape-of subgraph marking ->
  "No layout format available for select"); use `Minimum`/`Subtract`.
- `TransposeFusion` needs a separate order Constant per Transpose; a `Reshape` user blocks in-place crops (they become copies).
- Multi-output producers: plugin passes that take `node` instead of `output` break (fixed once in
  `unsqueeze_broadcast_reshape_sdpa_fusion`) - pass `Output<Node>` around.

## Runtime / kernels
- Kernels reaching buffers through pointer tables need `CL_KERNEL_EXEC_INFO_INDIRECT_{DEVICE,HOST,SHARED}_ACCESS_INTEL`,
  otherwise `CL_OUT_OF_RESOURCES` at enqueue.
- Thousands of unblocked enqueues (per-token prefill loop on a 5k-token prompt, ~750k enqueues) exhaust driver command
  storage -> `CL_OUT_OF_RESOURCES`; `clFlush` does not help, `clFinish` every ~64 tokens does.
- Persistent kernels: co-residency is a correctness requirement, not a tuning knob. A WG count fine on B60 can hang a
  smaller GPU. Clamp to device queries, bound the spin, report an error flag.
- Tail tasks must guard inactive lanes instead of returning before the completion barrier/semaphore signal.
- Store races when one kernel serves several tile heights: conditional stores for rows beyond the tile.
- Reductions across >1 sub-group must combine ALL partial sums (a M32 RMS once summed only the first SIMD16 group).
- KV handover/import: each layer and each K/V tensor can have its own padded stride in the stock KV state
  (`get_padded_dims()[2]` per tensor); one shared stride gives structured-but-wrong output (cos ~0.6).
- The runtime is shared by all infer requests of a compiled model, and OpenVINO rebinds variable states per request:
  never cache state handles across `Execute` calls; re-derive `past` per call.
- `scripts/benchmark_app.sh` sends 1-token requests with no prefill; stock KV state then has a dynamic layout -
  `get_shape()` on it throws. Guard with `get_partial_shape().is_static()` (or own the KV cache).
- genai VLM passes `inputs_embeds` as plain `cl_mem`; the plugin stages to USM (blocking copy). Outputs must be USM.
- Last-token logits: genai slices to the last token (`StridedSlice begin=[0,-1]`, output dim1 == 1); optimum may ask for
  all tokens. Support both output shapes.

## Measurement
- Fixed-position `decode_only` replays one token: for MoE it hits the same experts every step and looks faster than
  real decode - use `--decode-mode greedy`.
- `generation_config.json` may set `do_sample=true`; the e2e script forces greedy, ad-hoc scripts must too.
- "OUTPUT MISMATCH" on long prompts can be baseline imprecision or a near-tie flip - prove it with teacher-forced logits
  or the CPU f32 oracle; never assume.
- Prompts > `dynamic_quantization_threshold` (64) make the stock GPU FCs quantize activations to int8; shorter ones run
  f16. Baseline precision therefore differs between short and long prompts.
