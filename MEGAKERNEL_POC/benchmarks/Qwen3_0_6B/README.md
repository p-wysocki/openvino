# Qwen06BPOC random prefill and decode benchmark

Standalone benchmark for the Qwen06BPOC runtime. It needs no captured fixtures:
all FP16 weights and hidden states are
generated from a fixed seed; positions are initialized to zero for context
building and to the context length for decode. The output buffers are allocated
on the GPU and checked for finite values after setup, warmup and timing. This
does not compare against a reference model or test numerical correctness.

Run from the repository root:

```sh
MEGAKERNEL_POC/benchmarks/Qwen3_0_6B/run.sh --device=0 --iterations=200
```

The selected runtime uses the restored task-system decoder on B60, with fixed
34-worker, 1024-thread launch geometry. See
[Task-system overhead comparison](#task-system-overhead-comparison-2026-11-02).

The runner builds in `Qwen3_0_6B/build` (override with `BUILD_DIR`, `JOBS`, or
`CMAKE_BUILD_TYPE`). Options: `--device=<index|name substring>` (first OpenCL
GPU by default), `--list-devices`, `--context-tokens=N` (default 4000),
`--prefill-iterations=N` (default 5), `--prefill-warmup=N` (default 1),
`--iterations=N` (decode, default 100), `--warmup=N` (decode, default 5), `--seed=N`
(default 42). Requires an Intel OpenCL GPU with USM support.
`--context-tokens` must be between 2 and 4095 so the runtime uses its prefill path
and leaves room for a decode token.

Select the megakernel at CMake configure time (default: `Qwen06BPOC_prefill_separate_kernels`):

```sh
cmake -S MEGAKERNEL_POC/benchmarks/Qwen3_0_6B -B /tmp/qwen06b-bench2-build \
  -DMEGAKERNEL_IMPLEMENTATION=Qwen06BPOC_prefill_megakernel
cmake --build /tmp/qwen06b-bench2-build -j8
/tmp/qwen06b-bench2-build/qwen06b_random_decode_benchmark --context-tokens=128
```

The `Qwen06BPOC_prefill_separate_kernels` implementation builds the repository's
oneDNN GPU dependency as part of this standalone CMake project, so its initial
build takes longer than the other implementations.

Prefill and decode are timed separately using GPU timestamps on an in-order
queue. The prefill measurement repeats the same context at position zero after
one setup call and optional warmup, then decode repeats the same single-token
call at the next position. Each section reports average GPU latency, estimated
memory bandwidth, and B60 SOL Memory. Times exclude host setup and transfers.
The prefill transfer estimate counts
weights once, K/V writes once, and input, position and output bytes, but omits
K/V reads that may be served from on-chip storage. Decode counts weights once,
previous K/V reads once, new K/V writes once, and input, position and output
bytes. Effective bandwidth divides each estimate by average GPU latency
(decimal GB/s); it does not measure actual memory traffic. On an Arc Pro B60,
each phase's SOL Memory compares its effective bandwidth with the 456 GB/s peak.

## Historical Experiments

The experiment sections below record earlier implementations and settings.
Their decode environment options and alternate decoder paths have been removed;
those commands no longer reproduce the old configurations. Shared research
GemvBlock helpers and tests remain independent of this runtime. The current
flag-free configuration is described in the final section.

## Decode geometry experiments (2026-10-11)

Five iterations on `Qwen06BPOC_prefill_separate_kernels`, Intel Arc Pro B60,
SIMD32, seed 42. Each candidate used 10 decode warmups and 200 measured calls;
prefill used zero warmups and one measured call after setup. Values below are
average GPU decode latency in milliseconds, not end-to-end generation latency.

| Configuration | Workers | LWS | TF | Context 40 | Context 400 | Context 4000 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Restored baseline | 32 | 512 | 1 | 2.73 | 2.86 | 4.18 |
| 1. More workers | 40 | 512 | 1 | 2.79 | 2.92 | 4.25 |
| 2. More workers | 48 | 512 | 1 | 2.73 | 2.87 | 4.26 |
| 3. Double GEMV work per task | 32 | 512 | 2 | 2.87 | 3.00 | 4.32 |
| 4. Quadruple GEMV work per task | 32 | 512 | 4 | 3.39 | 3.52 | 4.80 |
| 5. Smaller groups, more workers | 64 | 256 | 1 | 2.87 | 3.13 | 5.71 |

No candidate was retained. The initial baseline (5 warmups, 100 calls) was
2.74 / 2.86 / 4.18 ms; rebuilding and rerunning the restored baseline produced
the first row. These are individual timing runs, not statistical confidence
intervals; the 48-worker short-context result is a tie, not a speedup.

Candidates 1-4 and the restored baseline matched saved decode outputs exactly
at all three contexts. Candidate 5 passed the existing per-element tolerance
and relative-L2 limit of 0.002, with relative L2 between 0.000694 and 0.000784
and maximum absolute error at most 0.00296. Its embedding and RMSNorm loops
were adapted to cover all 1024 elements with 256 threads; attention used eight
instead of sixteen subgroups, changing its partitioning and reduction order.
All runs also passed finite-output and warmup-versus-timed determinism checks.
This compares with the starting implementation, not a reference model.

`TF` controls sequential tiles per GEMV task; increasing it reduces task count
without increasing threads per group. Candidate 5 kept total launched threads
at 16384 and reduced declared attention SLM from 8320 to 4160 bytes per group,
but doubled projection task counts and halved attention subgroups per task.
Neither more workers nor heavier tasks improved this workload. Persistent
consumer tasks spin on producers, so larger worker pools must still respect
co-residency requirements.

The [Intel GPU Occupancy Calculator](https://oneapi-src.github.io/oneAPI-samples/Tools/GPU-Occupancy-Calculator/index.html)
reports theoretical occupancy and explicitly notes that higher occupancy need
not improve performance. No hardware occupancy or compiler register-allocation
measurement was collected in these runs; the reported 40% occupancy was not
independently verified. Register/spill and per-stage stall profiling is needed
before attributing these timings to an occupancy bottleneck.

Reproduce a worker-count candidate after creating a baseline reference with
`--write-decode-reference=/tmp/qwen-geometry-baseline-400.txt` and 32 workers:

```sh
OV_MEGAKERNEL_MONO_WG=48 timeout --kill-after=10s 180s \
  MEGAKERNEL_POC/benchmarks/Qwen3_0_6B/build/qwen06b_random_decode_benchmark \
  --device=0 --context-tokens=400 --prefill-iterations=1 --prefill-warmup=0 \
  --iterations=200 --warmup=10 \
  --decode-reference=/tmp/qwen-geometry-baseline-400.txt
```

Raw logs and reference files from this session are under
`/tmp/qwen-geometry-*`. For coarsening experiments, both host and embedded
OpenCL `TF` constants must change together before rebuilding.

## GemvBlock decode experiment (2026-10-11)

Set `OV_MEGAKERNEL_DECODE_GEMV_BLOCK=1` before runtime initialization to replace
all seven decode projections with the existing double-buffered SLM GemvBlock
implementation. Unset it or use `0` for the original subgroup GEMV. Prefill is
unchanged. The switch selects the OpenCL implementation at program build time,
so it does not add a branch inside projection tasks.

The specializations use 32 output rows per task and four compute subgroups:

| Projection | Specialization | Rows per phase |
| --- | --- | ---: |
| Q | `GemvBlock_2048x1024` | 4 |
| K, V | `GemvBlock_1024x1024` | 8 |
| O | `GemvBlock_1024x2048` | 4 |
| Gate, up | `GemvBlock_3072x1024` | 8 |
| Down | `GemvBlock_1024x3072` | 4 |

The shared templates now accept optional `GemvBlock_VECTOR_FP32` and
`GemvBlock_OUTPUT_FP32` flags, both defaulting to zero. Their compute helper
accepts the corresponding `ComputeGemvTile_` flags. Normalized activations stay
FP32, Q/K/V outputs stay FP16, and O/gate/up/down produce FP32 scratch results
before the existing residual or SiLU epilogues. This avoids introducing an
extra FP16 rounding before those epilogues. The enabled path allocates 24 KiB
of device scratch and uses 48 KiB SLM per worker, versus 8320 bytes for the
default path. Static assertions check the specialization SLM requirements.

Median GPU decode latency across three separate process runs, each with 10
warmups and 200 measured calls, B60, 32 workers, LWS 512, seed 42:

| Context tokens | Original (ms) | GemvBlock (ms) | Latency increase |
| ---: | ---: | ---: | ---: |
| 40 | 2.74 | 3.25 | 18.6% |
| 400 | 2.86 | 3.39 | 18.5% |
| 4000 | 4.18 | 4.83 | 15.6% |

The original path remains the default because GemvBlock was slower in all
three contexts. All 18 runs passed finite-output, determinism, and saved decode
reference checks. The original path matched exactly; GemvBlock relative L2
was 0.000714 / 0.000812 / 0.000722, below the 0.002 limit. Reduction order differs,
so GemvBlock is not bit-identical. This is baseline parity testing, not model
accuracy validation. The existing `TaskSystemGemvTest.Gemv2048x1024` also passed
against oneDNN and GemvOpt using the unchanged FP16 template defaults.

Use the previous benchmark command with `OV_MEGAKERNEL_DECODE_GEMV_BLOCK=1`
added to its environment. Logs are in `/tmp/qwen-gemvblock-r*-mode*-*.log`.
Use separate processes for repetitions: `--gtest_repeat=3` stalled at the
second test invocation with the default implementation during this experiment.

## Isolated GemvBlock versus subgroup GEMV (2026-10-11)

The existing research test executable now contains
`TaskSystemGemvTest.IsolatedGemvBlockVsSubgroup`. Its subgroup alternative
reproduces the runtime GEMV's two-row, SIMD32 reduction and first-panel weight
preloading, with FP16 or FP32 activation loads. Both alternatives use the same
persistent task scheduler, 32-row tasks, LWS 512, FP16 row-major weights, and
the same input/output precision. Inputs are ready: there are no producer waits,
residual additions, SiLU, or attention stages. Each kernel reserves its own
required SLM, rather than the whole decoder's maximum SLM allocation.

The sweep covers seven shape/tuning variants, FP16 and decode precision,
32 and 64 workers, and flushed/unflushed cache conditions: 56 configurations.
Each configuration uses three paired repetitions with alternating execution
order, 30 warmups and 200 measured kernel calls per repetition. Medians below
are in microseconds. Timing uses each kernel event's START/END timestamps;
cache saturation is enqueued before the kernel and excluded from its event.
This avoids subtracting an independently timed cache-flush kernel. Flushing
uses the existing L3 saturation helper, not a guarantee that every cache is cold.

Decode-precision results, using the integrated GemvBlock tile settings:

| Shape | Input/output | Workers | Flushed GemvBlock | Flushed subgroup | Unflushed GemvBlock | Unflushed subgroup |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Q, 2048x1024 | f32/f16 | 32 | 18.328 | 19.338 | 16.032 | 6.389 |
| K/V, 1024x1024 | f32/f16 | 32 | 9.710 | 11.464 | 7.661 | 4.865 |
| O, 1024x2048 | f16/f32 | 32 | 14.740 | 19.312 | 13.712 | 5.103 |
| Gate/up, 3072x1024 | f32/f32 | 32 | 21.617 | 27.451 | 19.436 | 8.946 |
| Down, 1024x3072 | f16/f32 | 32 | 21.677 | 37.938 | 20.126 | 6.338 |
| Q, 2048x1024 | f32/f16 | 64 | 14.084 | 20.124 | 13.322 | 6.479 |
| K/V, 1024x1024 | f32/f16 | 64 | 9.826 | 12.150 | 8.565 | 5.673 |
| O, 1024x2048 | f16/f32 | 64 | 15.402 | 20.955 | 12.784 | 5.105 |
| Gate/up, 3072x1024 | f32/f32 | 64 | 19.469 | 28.071 | 20.485 | 9.270 |
| Down, 1024x3072 | f16/f32 | 64 | 23.759 | 41.456 | 23.494 | 7.548 |

GemvBlock wins all five integrated-tuning shapes after flushing; subgroup GEMV
wins all five when reusing the same matrix without flushing. GemvBlock uses
`NontemporalLoad` with `LSC_LDCC_L1C_L3UC` for weights, explicitly bypassing L3;
the subgroup implementation uses default cache policy. The standalone down
tuning (2 phase rows / 2 compute subgroups) took 35.573 us flushed at 32 workers,
versus 21.677 us for 4/4, so reducing that tile is not an isolated improvement.

Both implementations passed double-precision CPU dot-product reference checks
on every repetition, using FP16-rounded weights and the selected activation
precision. Tolerances are `5e-5 + 2e-6 * abs(reference)` for FP32 outputs and
`0.002 + 0.001 * abs(reference)` for FP16 outputs. The original
`TaskSystemGemvTest.Gemv2048x1024` also passed after the harness changes.

These results confirm an isolated advantage for GemvBlock under flushed
conditions, but do not identify the cause of the full-decoder regression.
Repeated 2-6 MiB matrices are not the decoder's roughly 840 MiB weight stream;
the unflushed result must not be treated as proof of decoder weight-cache hits.
Global SLM/register requirements, producer waits, and epilogue overhead remain
integration-specific factors not measured by this isolated comparison.

After building the research `preloading_gemv` target, run:

```sh
timeout --kill-after=10s 180s \
  MEGAKERNEL_POC/research/preloading_gemv/build/preloading_gemv \
  --gtest_filter=TaskSystemGemvTest.IsolatedGemvBlockVsSubgroup
```

The full run took about 46 seconds on B60. Output lines prefixed `ISOLATED,`
contain CSV results; the last column is subgroup time divided by GemvBlock
time, so values above one favor GemvBlock. Session output is preserved in
`/tmp/qwen-isolated-gemv-sweep.log`.

## GemvBlock decode L3 cache policy (2026-10-11)

Compared GemvBlock weight loads using the same
`__builtin_IB_lsc_load_global_uint4` intrinsic with either `LSC_LDCC_L1C_L3UC`
(existing L3 bypass) or `LSC_LDCC_L1C_L3C` (L3 cached). Tiling, 32 workers,
LWS 512, precision, synchronization, and epilogues were unchanged. This tests
explicit L3 caching, not ordinary compiler-selected loads or a policy change
to the original subgroup GEMV.

On B60 device 0, each measurement used an independent process, 10 decode
warmups, and 300 measured iterations. Three rounds rotated the order of the
subgroup control, GemvBlock bypass, and GemvBlock cached paths. Median decode
latencies in milliseconds:

| Context | Subgroup control | GemvBlock L3 bypass | GemvBlock L3 cached | Cached slowdown vs bypass |
| --- | ---: | ---: | ---: | ---: |
| 40 | 2.74 | 3.25 | 3.40 | 4.6% |
| 400 | 2.86 | 3.39 | 3.53 | 4.1% |
| 4000 | 4.18 | 4.83 | 4.96 | 2.7% |

Bypass ranges were 3.24-3.25, 3.39-3.39, and 4.83-4.84 ms; cached ranges were
3.33-3.40, 3.53-3.54, and 4.96-4.96 ms. Explicit L3 caching was slower in every
paired comparison. Bypassing L3 is beneficial for this GemvBlock implementation,
but does not close its gap to subgroup GEMV. These measurements do not establish
cache hit rates or the effect of bypassing L3 in the subgroup implementation.

All 27 primary runs passed the reference tolerances and the warmup-versus-final
output equality check. Fresh bypass references also passed comparison with the
original decoder. One cached context-4000 process differed from its bypass
reference (relative L2 `6.80e-4`, max absolute error `1.98e-3`); the other
17 GemvBlock runs matched exactly. Three additional cached/bypass pairs at
context 4000 all matched exactly, with cached latency 4.95-4.96 ms versus bypass
4.83 ms. The isolated cross-process discrepancy remains unexplained; universal
bit-identical cache-policy parity is not established. The legacy
`TaskSystemGemvTest.Gemv2048x1024` also passed with the unchanged default policy.

Defaults remain subgroup GEMV, with L3 bypass when GemvBlock is enabled. To
reproduce the cache comparison after building the decoder benchmark, run each
policy in a separate process (repeat with contexts 40 and 4000):

```sh
for cached in 0 1; do
  OV_MEGAKERNEL_DECODE_GEMV_BLOCK=1 \
  OV_MEGAKERNEL_GEMV_BLOCK_L3_CACHE="$cached" \
  OV_MEGAKERNEL_MONO_WG=32 \
  timeout --kill-after=10s 90s \
    MEGAKERNEL_POC/benchmarks/Qwen3_0_6B/build/qwen06b_random_decode_benchmark \
    --device=0 --context-tokens=400 --prefill-iterations=1 --prefill-warmup=0 \
    --iterations=300 --warmup=10
done
```

`OV_MEGAKERNEL_GEMV_BLOCK_L3_CACHE=1` affects only the enabled GemvBlock path.
The shared load helper retains L3 bypass unless its compile-time
`NONTEMPORAL_LOAD_CACHE_CONTROL` is overridden. Session logs are in
`/tmp/qwen-cache-r{1,2,3}-{subgroup,bypass,cached}-{40,400,4000}.log`, with
follow-up parity logs in `/tmp/qwen-cache-parity-r{4,5,6}-cache{0,1}-4000.log`
and the legacy result in `/tmp/qwen-cache-legacy.log`.

## GemvBlock decode worker sweep (2026-10-11)

Tested whether GemvBlock needs a different worker count from subgroup GEMV.
Only `OV_MEGAKERNEL_MONO_WG` changed: GemvBlock retained LWS 512, 32-row tiles,
48 KiB SLM per work-group, L3-bypassing weight loads, precision, synchronization,
and epilogues. No runtime or kernel source changes were needed.

On B60 device 0, each independent process used 10 decode warmups and 300 measured
iterations. Context 400 was tested in three rounds with ascending, descending,
and mixed worker ordering, plus a fresh 32-worker subgroup control each round.
Median decode latencies and observed ranges, in milliseconds:

| GemvBlock workers | Median | Range |
| ---: | ---: | ---: |
| 16 | 5.29 | 5.29-5.30 |
| 24 | 4.47 | 4.47-4.47 |
| 32 | 3.39 | 3.39-3.39 |
| 48 | 3.17 | 3.16-3.17 |
| 64 | 3.27 | 3.27-3.27 |
| 80 | 3.27 | 3.27-3.27 |

The subgroup control was 2.86 ms in all three rounds. The best sampled
GemvBlock setting was 48 workers, reducing latency by 6.5% versus 32 workers.
All sampled counts completed without timeout; this is not a general residency
guarantee for other devices or kernels.

Validated 48 workers at contexts 40 and 4000 with three rotated-order rounds
against 32-worker GemvBlock and the subgroup control. Combined median results:

| Context | Subgroup, 32 workers | GemvBlock, 32 workers | GemvBlock, 48 workers | GemvBlock latency reduction |
| --- | ---: | ---: | ---: | ---: |
| 40 | 2.74 | 3.25 | 3.03 | 6.8% |
| 400 | 2.86 | 3.39 | 3.17 | 6.5% |
| 4000 | 4.18 | 4.83 | 4.68 | 3.1% |

The 48-worker ranges were 3.03-3.03 ms at context 40 and 4.67-4.68 ms at
context 4000. All 39 runs passed reference and warmup-versus-final output
checks: all 30 GemvBlock runs matched the saved 32-worker bypass references
exactly, and all nine subgroup controls matched the original references exactly.

The launch configuration explains part of GemvBlock's integration penalty,
but tuned GemvBlock remains 10.6-12.0% slower than subgroup GEMV. Reducing workers
to 16 or 24 does not help; adding workers beyond 48 also regresses at context
400. No instantaneous memory-bandwidth, occupancy, or stall counters were
collected, so these timings do not establish whether load bursts or contention
cause the remaining gap. Worker counts other than 32 and 48 were not swept at
contexts 40 and 4000, and LWS was not varied.

Defaults remain unchanged. To reproduce the worker sweep after building the
decoder benchmark, repeat this loop with reversed and mixed ordering:

```sh
for workers in 16 24 32 48 64 80; do
  OV_MEGAKERNEL_DECODE_GEMV_BLOCK=1 \
  OV_MEGAKERNEL_GEMV_BLOCK_L3_CACHE=0 \
  OV_MEGAKERNEL_MONO_WG="$workers" \
  timeout --kill-after=10s 60s \
    MEGAKERNEL_POC/benchmarks/Qwen3_0_6B/build/qwen06b_random_decode_benchmark \
    --device=0 --context-tokens=400 --prefill-iterations=1 --prefill-warmup=0 \
    --iterations=300 --warmup=10
done
```

Session logs: `/tmp/qwen-block-workers-r{1,2,3}-w{16,24,32,48,64,80}-400.log`,
`/tmp/qwen-block-workers-r{1,2,3}-w{32,48}-{40,4000}.log`, and
`/tmp/qwen-block-workers-r{1,2,3}-subgroup-{40,400,4000}.log`.

## Heavier GemvBlock tasks (2026-10-11)

Added opt-in 64-row tasks, halving projection tasks per layer from 288 to 144
(gate/up share one task). Both heavy variants allocate 64 KiB SLM per worker
and retain 32 workers and L3-bypassing weight loads. Host task counts, device
dependency counts, and residual/SiLU epilogues use the larger row tiles.

| Projection | Rows per phase | Compute subgroups, 512 / 1024 threads | One buffer | Both buffers |
| --- | ---: | ---: | ---: | ---: |
| Q/K/V, gate/up | 16 | 4 / 8 | 32 KiB | 64 KiB |
| O | 8 | 4 / 8 | 32 KiB | 64 KiB |
| Down | 4 | 4 / 4 | 24 KiB | 48 KiB |

The helper double-buffers whole rows. Down has 3072 columns: its next valid
phase size, eight rows, would need 96 KiB for both buffers. It therefore keeps
four-row phases and four compute subgroups, using 48 KiB of the allocated
64 KiB. For the other projections, the 1024-thread variant has 32 SIMD32
subgroups, split into eight compute and 24 loader subgroups. The initial
preload uses the entire work-group, as before.

The 1024-thread path guards embedding and RMS vector accesses beyond the
1024-element activation. RMS handles the additional zero-valued subgroup
partials, and attention partitions its work across 32 rather than 16 subgroups.
Thus its timing and numerical changes are not attributable solely to GEMV.
The compiled kernel's maximum work-group size is checked before execution.

B60 device 0, seed 42: three independent processes per configuration/context,
rotated test order, 10 decode warmups, and 300 measured iterations. All settings
below use 32 workers; median decode latencies are in milliseconds:

| Configuration | Context 40 | Context 400 | Context 4000 |
| --- | ---: | ---: | ---: |
| Subgroup, 512 threads | 2.75 | 2.87 | 4.19 |
| Original GemvBlock, 512 threads, 32-row tasks | 3.21 | 3.36 | 4.80 |
| Heavy GemvBlock, 512 threads, 64-row tasks | 3.87 | 4.03 | 5.45 |
| Heavy GemvBlock, 1024 threads, 64-row tasks | 3.60 | 3.68 | 4.76 |

The heavy/512 ranges were 3.87-3.88, 4.03-4.03, and 5.45-5.45 ms; heavy/1024
ranges were 3.60-3.61, 3.68-3.69, and 4.75-4.76 ms. Increasing threads helps
the heavy layout, but heavy/1024 remains 12.1% and 9.5% slower than the original
GemvBlock at contexts 40 and 400. It is only 0.8% faster at context 4000, where
attention partitioning also changed. Neither heavy variant beats subgroup GEMV.
These are fresh controls; earlier sections report earlier measurements.

All 36 runs passed reference tolerances, finite-output checks, and exact
warmup-versus-final output checks. Subgroup controls matched the original
references exactly. Heavy/1024 relative L2 errors were 0.000750, 0.000833,
and 0.000670 at contexts 40, 400, and 4000; maximum absolute error across them
was 0.00394. The heavy/512 reference error metrics matched the original
GemvBlock metrics. No occupancy, register/spill, or instantaneous bandwidth
measurements were collected; unchanged occupancy at 64 KiB is not established.

Defaults remain unchanged. Enable heavy tasks with
`OV_MEGAKERNEL_GEMV_BLOCK_HEAVY=1` alongside
`OV_MEGAKERNEL_DECODE_GEMV_BLOCK=1`. The optional
`OV_MEGAKERNEL_GEMV_BLOCK_THREADS` accepts `512` or `1024`, defaulting to `1024`
for heavy tasks. These settings do not change the subgroup or original
GemvBlock configurations when heavy mode is disabled.

```sh
for threads in 512 1024; do
  OV_MEGAKERNEL_DECODE_GEMV_BLOCK=1 \
  OV_MEGAKERNEL_GEMV_BLOCK_HEAVY=1 \
  OV_MEGAKERNEL_GEMV_BLOCK_THREADS="$threads" \
  OV_MEGAKERNEL_GEMV_BLOCK_L3_CACHE=0 \
  OV_MEGAKERNEL_MONO_WG=32 \
  timeout --kill-after=10s 60s \
    MEGAKERNEL_POC/benchmarks/Qwen3_0_6B/build/qwen06b_random_decode_benchmark \
    --device=0 --context-tokens=400 --prefill-iterations=1 --prefill-warmup=0 \
    --iterations=300 --warmup=10
done
```

Session logs: `/tmp/qwen-heavy-r{1,2,3}-{subgroup,block32,heavy512,heavy1024}-{40,400,4000}.log`.

## Three-GEMV chain layout in decode (2026-10-12)

Replicated the mixed tile layout from
`ChainTaskSystemGemvTest.ThreeGemvChain`, rather than using 64-row tasks for
every projection. The decoder experiment uses 35 workers, 1024 threads per
worker, 64 KiB SLM, and the same GemvBlock preload/wait/signal mechanism.

| Stage | Rows per task | Rows per phase | Compute subgroups | Tasks per layer |
| --- | ---: | ---: | ---: | ---: |
| O, 1024x2048 | 32 | 8 | 8 | 32 |
| Gate/up, 3072x1024 | 64 | 16 | 8 | 48 |
| Down, 1024x3072 | 32 | 4 | 4 | 32 |

These stages have 112 queued tasks, matching the synthetic chain, versus 80
in the all-heavy decoder configuration. The decoder's gate/up task still runs
two GEMVs plus SiLU; the synthetic middle task runs one GEMV. Q/K/V are absent
from the synthetic benchmark and retain the heavy decoder's 64-row tiles and
16-row phases. Total projection tasks are therefore 176 per layer, or 4928 per
token over 28 layers, versus 144/4032 for all-heavy and 288/8064 for baseline.

Dependencies preserve model semantics. O waits for attention/cache completion;
post-attention RMS waits for all 32 O tasks; gate/up waits for that normalized
activation; down waits for all 48 gate/up tasks. The next layer's normalization
waits for all 32 down tasks. Existing scratch-buffer and epilogue barriers remain.
FP32 normalization and projection epilogues are retained, unlike the synthetic
chain's all-FP16 boundaries. This matches its scheduling pattern and GEMV tile
settings, not its entire computation.

The opt-in chain mode also reproduces the benchmark's IGC flags:
`-igc_opts 'VISAOptions=-hybridRAWithSpill -fastCompileRA'`. A flags-off comparison
isolates their effect while keeping layout and workers identical. The unchanged
synthetic benchmark passed with 112 tasks and 35 workers: task-system 52.07 us,
separate OpenCL 66.57 us, oneDNN 80.74 us (one run, 1000 measured iterations).
Its OpenCL comparator is not the decoder's custom subgroup GEMV.

Decoder timings below are fresh B60 device-0 measurements, seed 42, three
independent processes per setting/context, rotated order, 10 warmups, and 300
measured iterations. GemvBlock weights bypass L3 throughout. Medians in ms:

| Configuration | Workers | Context 40 | Context 400 | Context 4000 |
| --- | ---: | ---: | ---: | ---: |
| Subgroup, 512 threads | 32 | 2.75 | 2.87 | 4.19 |
| Original GemvBlock, 512 threads | 48 | 3.01 | 3.15 | 4.63 |
| All-heavy GemvBlock, 1024 threads | 35 | 3.65 | 3.74 | 4.80 |
| Chain layout, benchmark IGC flags | 35 | 2.94 | 3.04 | 4.18 |
| Chain layout, no IGC overrides | 35 | 2.89 | 2.97 | 4.00 |

At fixed 35 workers and without IGC overrides, changing only O/down task rows
from 64 to 32 (including their task counts, epilogues, and dependency thresholds)
reduces context-400 latency from 3.74 to 2.97 ms, or 20.6%. The chain's compiler
flags hurt this decoder layout. Without those flags, chain layout improves on
the 48-worker original GemvBlock by 4.0%, 5.7%, and 13.6%. It remains 5.1% and
3.5% slower than subgroup at short/medium context, but is 4.5% faster at 4000.
Its timing ranges were 2.89-2.89, 2.97-2.98, and 4.00-4.00 ms. The IGC-on ranges
were 2.93-2.94, 3.04-3.04, and 4.18-4.19 ms.

All 45 decoder runs passed reference tolerances, finite-output checks, and
warmup-versus-final equality checks. Subgroup controls matched the original
references exactly. Across GemvBlock configurations, maximum relative L2 was
0.000833 and maximum absolute error was 0.00394. No occupancy or stall counters
were collected. Comparisons against 512-thread implementations also include the
1024-thread attention partitioning change; the all-heavy versus chain comparison
holds that partitioning fixed.

Defaults remain unchanged. `OV_MEGAKERNEL_GEMV_BLOCK_CHAIN=1` requires
`OV_MEGAKERNEL_DECODE_GEMV_BLOCK=1`, selects the mixed tiles and 1024 threads,
and takes precedence over the heavy/thread settings. Set
`OV_MEGAKERNEL_MONO_WG=35` explicitly to match the chain's worker count; enabling
chain mode alone does not change the existing 32-worker default. IGC overrides
are enabled by default for chain mode; set `OV_MEGAKERNEL_GEMV_BLOCK_CHAIN_IGC=0`
for the faster measured variant:

```sh
OV_MEGAKERNEL_DECODE_GEMV_BLOCK=1 \
OV_MEGAKERNEL_GEMV_BLOCK_CHAIN=1 \
OV_MEGAKERNEL_GEMV_BLOCK_CHAIN_IGC=0 \
OV_MEGAKERNEL_GEMV_BLOCK_L3_CACHE=0 \
OV_MEGAKERNEL_MONO_WG=35 \
timeout --kill-after=10s 60s \
  MEGAKERNEL_POC/benchmarks/Qwen3_0_6B/build/qwen06b_random_decode_benchmark \
  --device=0 --context-tokens=400 --prefill-iterations=1 --prefill-warmup=0 \
  --iterations=300 --warmup=10
```

Session logs: `/tmp/qwen-chain-layout-r{1,2,3}-{subgroup32,block48,heavy35,chain35,chain35noigc}-{40,400,4000}.log`.
The unchanged synthetic chain result is in `/tmp/qwen-chain-synthetic.log`.

## Register-output GemvBlock epilogues (2026-10-21)

GemvBlock remains the projection implementation when
`OV_MEGAKERNEL_DECODE_GEMV_BLOCK=1`. O, gate/up, and down now enable
`GemvBlock_OUTPUT_PRIVATE=1` and pass a pointer to a private scalar instead of
a global output array. Q/K/V retain the default global FP16 output because
their results cross task boundaries. Prefill and the default subgroup path
are unchanged; all existing tile sizes, phase sizes, and worker settings remain.

The private-output contract is:

- Each compute subgroup distributes its block results across its first
  `GemvBlockResultsPerSubgroup_SUFFIX` lanes, one scalar per lane.
- Only those lanes in the first `GemvBlock_COMPUTE_WARPS` subgroups have valid
  results. `GemvBlockOutputRow_SUFFIX(tileId)` maps each valid lane to its
  global output row, accounting for the phase and subgroup layout.
- The block must have at most `WARP_SIZE` results per compute subgroup;
  a static assertion enforces this. FP16/FP32 output selection is unchanged.
- `ComputeGemvTile_OUTPUT_PRIVATE` is the matching internal helper option;
  its additional `outputLaneBase` argument selects the lanes for each phase.
- No final work-group barrier is needed to consume a private result.
  Callers must still synchronize before reusing SLM. Stage E retains a local
  barrier between gate and up; the existing semaphore publishes final writes.

Stages D/F apply residual additions directly to their private FP32 results.
Stage E keeps one gate and one up scalar per valid lane through SiLU. This
removes the 24 KiB global scratch allocation, three global epilogue barriers
per layer, and the global fence from the gate/up SLM-reuse barrier. SLM remains
48 KiB for base GemvBlock and 64 KiB for heavy/chain configurations.
The removed scratch stores and reloads total
`28 * (1024 + 3072 + 3072 + 1024) * 4 * 2 = 1,835,008` bytes (1.75 MiB)
per decode token. This is logical traffic, not measured DRAM traffic; scratch
may previously have hit caches, and streamed weights still dominate.

Three independent processes per setting/context, B60 device 0, seed 42,
10 warmups and 300 measured decode iterations, fixed configuration order:

| Configuration | Workers | Context 40 | Context 400 | Context 4000 |
| --- | ---: | ---: | ---: | ---: |
| Base GemvBlock, private output | 48 | 2.99 | 3.13 | 4.61 |
| Chain, private output, benchmark IGC flags | 35 | 2.91 | 3.02 | 4.17 |
| Chain, private output, no IGC overrides | 35 | 2.92 | 3.00 | 4.03 |

Values are median GPU decode milliseconds. The preceding section's recorded
global-output results were respectively 3.01/3.15/4.63, 2.94/3.04/4.18, and
2.89/2.97/4.00 ms. These are historical comparisons, not interleaved controls:
base/IGC-on differences are small and do not establish a speedup. A fresh
global-output chain/no-IGC control at context 400 measured 2.97 ms (100 calls)
before editing, versus 3.00 ms after (300 calls). Thus the requested scratch
elimination does not improve the fastest chain configuration in these runs.
No default environment settings were changed.

Rejected prototypes at context 400: private arrays per lane (3.04 ms), uniform
private-array assignments (3.13 ms), and SLM output with smaller phase panels
to fit 64 KiB (3.19 ms). The retained scalar-per-lane version avoids those
arrays and preserves the original phase geometry. No hardware memory-traffic
or occupancy counters were collected, and register-spill savings are not claimed.

Validation: all 27 timing runs passed saved-reference tolerances, finite checks,
and warmup/final equality. Maximum relative L2 versus subgroup was 0.000833;
maximum absolute error was 0.00394. Context-400 chain output matched the freshly
saved global-GemvBlock reference exactly. Six additional runs covered subgroup,
heavy/512, and heavy/1024 at contexts 40 and 4000; subgroup matched exactly.
`TaskSystemGemvTest.PrivateOutputMatchesGlobal` passed 32 shape/precision cases
(eight layouts, FP16/FP32 inputs and outputs), checking exact global/private
parity and a CPU double reference across three tiles. The existing
`TaskSystemGemvTest.Gemv2048x1024` also passed.

Use the preceding chain command to exercise the new path; no extra switch is
needed. Build and run the focused template regression with:

```sh
cmake --build MEGAKERNEL_POC/research/preloading_gemv/build --target preloading_gemv -j8
MEGAKERNEL_POC/research/preloading_gemv/build/preloading_gemv \
  --gtest_filter=TaskSystemGemvTest.PrivateOutputMatchesGlobal
```

Session logs: `/tmp/qwen-register-r{1,2,3}-{base,chain,chain_igc}-{40,400,4000}.log`,
`/tmp/qwen-register-check-{subgroup,heavy512,heavy1024}-{40,4000}.log`, and
`/tmp/qwen-private-tests.log`. The fresh global-output reference is
`/tmp/qwen-private-before-400.txt`.

## SLM-output block epilogues (2026-10-22)

Set `OV_MEGAKERNEL_GEMV_BLOCK_SLM_OUTPUT=1` together with
`OV_MEGAKERNEL_DECODE_GEMV_BLOCK=1` to pass O, gate/up, and down projection
results through SLM before their epilogues. Register output remains the default
for A/B comparison. Q/K/V, prefill, and the non-GemvBlock path are unchanged.

`GemvBlockLocal_SUFFIX` calls the private-output GemvBlock, scatters the
per-lane results into contiguous tile-local SLM, and returns that buffer after
a local-memory barrier. It reuses the idle half of the double-buffered weight
storage after the last phase, selecting the half according to phase-count
parity. The other half can still be read by the final compute phase. Static
assertions check output capacity and alignment. No extra SLM allocation or
smaller phase panels are needed: base remains at 48 KiB, heavy/chain at 64 KiB.

D/F consume the returned array using block-local IDs, independent of GEMV
compute-subgroup ownership. E first loads each gate result into its tail
thread's register, then executes a local barrier before up reuses the same
SLM. Up results are consumed from SLM for SiLU. Only the first 32 or 64 threads
have output rows in the current tiles; making the result available to the
whole block does not create work for every thread. The existing semaphore
barrier still publishes final output writes before dependent tasks run.

Fresh B60 comparisons used three separate-process rounds, reversing order in
the second round. Median GPU decode latency in milliseconds:

| Configuration | Workers | Context 40 | Context 400 | Context 4000 |
| --- | ---: | ---: | ---: | ---: |
| Without GemvBlock, subgroup | 32 | 2.75 | 2.87 | 4.19 |
| Chain GemvBlock, register tails, IGC overrides off | 35 | 2.89 | 2.97 | 4.00 |
| Chain GemvBlock, SLM tails, IGC overrides off | 35 | 2.85 | 2.93 | 3.95 |

SLM improves on register tails by 1.4%, 1.3%, and 1.3%, respectively. Compared
with non-GemvBlock decode, it is 3.6% slower at context 40, 2.1% slower at 400,
and 5.7% faster at 4000. SLM/register comparisons keep geometry fixed; subgroup
comparisons also change work-group size and attention partitioning. These are
GPU decode timings, not end-to-end generation measurements. No physical memory
traffic, occupancy, or stall counters were collected.

All 27 timing runs passed output checks; all nine SLM runs matched saved
register outputs exactly. The expanded
`TaskSystemGemvTest.PrivateOutputMatchesGlobal` passed 40 shape/precision
cases with global/register/SLM equality and CPU-reference checks, including
single-phase and odd-phase layouts. Eight additional SLM decode runs covered
base, heavy/512, heavy/1024, and chain with IGC overrides enabled at contexts
40 and 4000; all passed the saved subgroup-reference tolerances.

```sh
OV_MEGAKERNEL_DECODE_GEMV_BLOCK=1 \
OV_MEGAKERNEL_GEMV_BLOCK_CHAIN=1 \
OV_MEGAKERNEL_GEMV_BLOCK_CHAIN_IGC=0 \
OV_MEGAKERNEL_GEMV_BLOCK_SLM_OUTPUT=1 \
OV_MEGAKERNEL_GEMV_BLOCK_L3_CACHE=0 \
OV_MEGAKERNEL_MONO_WG=35 \
timeout --kill-after=10s 180s \
  MEGAKERNEL_POC/benchmarks/Qwen3_0_6B/build/qwen06b_random_decode_benchmark \
  --device=0 --context-tokens=400 --prefill-iterations=1 --prefill-warmup=0 \
  --iterations=300 --warmup=10
```

Session logs: `/tmp/qwen-slm-r{1,2,3}-{subgroup,registers,slm}-{40,400,4000}.log`,
`/tmp/qwen-slm-check-{base,heavy512,heavy1024,chain_igc}-{40,4000}.log`, and
`/tmp/qwen-slm-tests.log`. Register reference outputs are in
`/tmp/qwen-slm-registers-{40,400,4000}.txt`.

## GemvBlock parameter sweep (2026-10-22)

Searched the existing runtime switches without changing kernel source or
defaults. The best sampled setting remains chain layout, SLM output, 35
workers, L3 bypass, and chain IGC overrides disabled. No improvement over the
preceding SLM configuration was found. This is a bounded search, not proof of
a global optimum or an exhaustive joint sweep of every switch.

Screening used B60 device 0, seed 42, 10 warmups and 150 measured decode calls
per independent process. Prefill used zero warmups and one measured call.
SLM output, L3 bypass, and disabled chain IGC overrides were fixed for the
layout/worker sweep. All four layouts were screened at context 400:

| Workers | Base, LWS512 | Heavy, LWS512 | Heavy, LWS1024 | Chain, LWS1024 |
| ---: | ---: | ---: | ---: | ---: |
| 24 | 4.43 | 4.14 | 3.70 | 3.42 |
| 32 | 3.32 | 3.96 | 3.65 | 2.97 |
| 35 | 3.26 | 3.97 | 3.70 | 2.93 |
| 40 | 3.22 | 4.04 | 3.76 | 2.99 |
| 48 | 3.11 | 4.04 | 3.76 | 2.99 |
| 64 | 3.21 | 4.04 | 3.76 | 2.99 |

Values are GPU decode milliseconds from individual screening runs. Chain
fine-sweep results at context 400 were 2.94 ms for 33/34/36 workers, 2.95 ms
for 37, and 2.96 ms for 38. At context 4000, 33 workers tied 35 at 3.95 ms;
34 and 36 measured 3.96 ms. The benchmark prints latency to two decimal places.

The coarse chain worker sweep was also repeated at contexts 40 and 4000:

| Workers | Context 40 | Context 4000 |
| ---: | ---: | ---: |
| 24 | 3.28 | 4.92 |
| 32 | 2.88 | 3.98 |
| 35 | 2.85 | 3.95 |
| 40 | 2.88 | 4.01 |
| 48 | 2.88 | 4.01 |
| 64 | 2.88 | 4.00 |

The best alternative-layout worker counts from the context-400 screen also
lost at context 4000: base/48 measured 4.59 ms, heavy512/32 5.38 ms, and
heavy1024/32 4.75 ms. More workers were not assumed to improve occupancy or
bandwidth. Each process had a 90-second timeout with a 10-second kill grace
period; no run timed out, but these results do not establish safe co-residency
for other devices or arbitrary worker counts.

Additional chain/35 switch screening:

| Output | Chain IGC overrides | L3 policy | Context 400 | Context 4000 |
| --- | --- | --- | ---: | ---: |
| SLM | Off | Bypass | 2.93 | 3.95 |
| SLM | On | Bypass | 2.97 | 4.09 |
| SLM | Off | Cached | 3.05 | 4.04 |
| SLM | On | Cached | 3.21 | 4.24 |
| Registers | Off | Bypass | 2.98 | 3.99 |

The two closest worker settings were then compared across three fresh rounds,
alternating process order and context order, with 10 warmups and 300 measured
calls. Median latency:

| Chain + SLM workers | Context 40 | Context 400 | Context 4000 |
| ---: | ---: | ---: | ---: |
| 33 | 2.86 | 2.94 | 3.95 |
| 35 | 2.85 | 2.93 | 3.95 |

The 35-worker short-context range was 2.84-2.85 ms; all other confirmation
values were unchanged at printed precision. The existing command above
therefore remains the recommendation; there is no new faster command from
this search. No physical bandwidth, occupancy, or stall counters were collected.

All 73 runs (55 screens and 18 confirmations) passed saved-reference,
finite-output, and warmup/final equality checks. References were the saved
chain/register outputs at the matching context. Chain and heavy1024 matched
them exactly. Base and heavy512 passed tolerances, with maximum relative L2
0.000784 and maximum absolute error 0.00288; their attention partitioning
differs because they use 512 threads. These checks are implementation parity,
not validation against a reference model.

To reproduce a coarse-screen point, use the preceding benchmark command with
`--iterations=150`, `--context-tokens=400`, and the desired worker count.
For base set `OV_MEGAKERNEL_GEMV_BLOCK_CHAIN=0` and
`OV_MEGAKERNEL_GEMV_BLOCK_HEAVY=0`; for heavy set chain to `0`, heavy to `1`,
and `OV_MEGAKERNEL_GEMV_BLOCK_THREADS=512` or `1024`. Keep SLM output enabled,
L3 caching disabled, and chain IGC overrides disabled. Use separate processes
and bounded timeouts when sweeping persistent worker counts.

Raw logs are `/tmp/qwen-meta-{coarse,fine,contexts,switches,confirm1,confirm2,confirm3}-*.log`.
Each filename records layout, workers (`w`), context (`c`), IGC override (`i`),
L3 caching (`l`), and SLM output (`s`).

## Rejected GemvBlock QK attention experiment (2026-10-26)

Tried replacing the historical-key dot products in `mk_stageBC` with
GemvBlock. The K cache is row-major `[sequence, 128]`, so QK maps naturally
to a GEMV over the RoPE-transformed query. The existing helper reads 256
elements per SIMD32 iteration; the experiment therefore added a 128-column
`float4` path and an optional SLM input vector. Query and score tiles stayed
FP32 in SLM. RoPE, current-token attention, value accumulation, cache writes,
and partition merging otherwise retained their existing roles. The last
incomplete key tile used the original dot-product loop, without padded reads.

All variants used 128-key blocks. The first candidate preserved the original
contiguous subgroup sequence assignment, but only a subset of subgroups could
consume each score tile. A second candidate distributed each tile's sequences
across all subgroups, changing online-softmax reduction order. Subsequent
variants enlarged phases, used ordinary cached key loads rather than the
projection-oriented non-temporal loads, and increased compute subgroups.

B60 device 0, chain projections with SLM tails, 35 workers, 1024 threads,
projection L3 bypass and chain IGC overrides disabled, seed 42. Single screening
runs used 10 warmups and 150 measured decode calls; times below are GPU decode
milliseconds, not stage-only timings or repeated-run medians:

| Attention implementation | Context 400 | Context 4000 |
| --- | ---: | ---: |
| Original fused subgroup attention, fresh control | 2.93 | 3.95 |
| GemvBlock, phase16/compute8, contiguous assignment | 3.19 | Not measured |
| GemvBlock, phase16/compute8, distributed assignment | 3.13 | Not measured |
| GemvBlock, phase64/compute8, non-temporal keys | 3.03 | 4.86 |
| GemvBlock, phase64/compute8, cached keys | 3.01 | 4.70 |
| GemvBlock, phase64/compute16, cached keys | 2.99 | 4.51 |

The best tested candidate was still 2.0% slower at context 400 and 14.2%
slower at 4000. Unlike the wide projection matrices, QK has a short 128-element
reduction already fused with online softmax and value accumulation. GemvBlock
introduced key staging, score handoffs, and work-group barriers. Those costs
are a plausible explanation, not a conclusion from hardware stall counters;
no such counters were collected. The value product was not replaced: its
required matrix orientation differs from GemvBlock's contiguous output rows.

Temporary template coverage passed 60 shape/precision cases, including
128-column keys, global/SLM queries, and cached loads. The initial contiguous
candidate matched the context-400 decoder reference exactly; distributed
candidates passed the standard context-40/400/4000 tolerances but differed
numerically. Boundary tests passed at contexts 2, 3, 255, 256, 257, and 258.
The context-4095 distributed candidate failed the per-element tolerance
(relative L2 0.00142, max absolute error 0.00493). A further variant cached all
partition scores in SLM to restore original accumulation order, but also
failed that check and measured 4.65 ms at 4095. Tolerances were not relaxed.
A fresh original-attention control was itself non-bit-identical to the saved
4095 reference (relative L2 0.00160, max absolute error 0.00430), although it
passed tolerances. Thus the near-capacity discrepancy was not isolated solely
to the candidate, and its cause remains unresolved.

No attention candidate was retained: the temporary runtime switch, template
extensions, and their additional test cases were removed. Earlier projection
GemvBlock/register/SLM-output changes remain intact. Rebuilding the restored
code yielded 2.85 / 2.93 / 3.95 ms at contexts 40 / 400 / 4000 with exact saved
reference equality. The existing 40-case `PrivateOutputMatchesGlobal` and
legacy `Gemv2048x1024` tests both passed after restoration. The previously
recommended benchmark command remains unchanged.

Session logs: `/tmp/qwen-attn-{qk128*,phase64*,cached*,halfcompute*,edges*,ordered*}.log`,
`/tmp/qwen-attn-control-stability-4095.log`,
`/tmp/qwen-attn-template-final.log`, and
`/tmp/qwen-attn-restored-*.log`. These document rejected experiments, not
an available `OV_MEGAKERNEL_ATTN_GEMV_BLOCK` feature.

## Static decode bandwidth optimization (2026-10-26)

This section records the indexed-static implementation before the scheduler-only
revert described in [Task-system overhead comparison](#task-system-overhead-comparison-2026-11-02).

At that point, the only decoder in `Qwen06BPOC_prefill_separate_kernels` used indexed-static
scheduling. It uses direct subgroup GEMV rather than
GemvBlock, with 1024-thread work-groups and asymmetric projection tiles:
two rows per subgroup for Q/K/V/gate/up (64 rows/task), one row for O/down
(32 rows/task). Each layer has 176 projection tasks and 218 total tasks.
The work-group uses 16,640 bytes of SLM for attention, not weight staging.
Weights remain FP16 and projection accumulation remains FP32.

Workers traverse the topologically ordered task sequence by a fixed stride.
Stage, layer, and tile are derived from the task index, eliminating per-task
queue atomics, descriptor loads, and task-pointer broadcast barriers. Existing
producer/consumer semaphores remain. An acquire-release completion counter
identifies the last worker, which resets synchronization state. The kernel
receives its context directly; there is no host task-descriptor queue or generic
task manager. Worker count is fixed at 34, with 1024 threads per work-group.

Run the default configuration without environment flags:

```sh
MEGAKERNEL_POC/benchmarks/Qwen3_0_6B/run.sh \
  --device=0 --context-tokens=4000 \
  --prefill-iterations=1 --prefill-warmup=0 \
  --iterations=300 --warmup=10
```

The runtime is restricted to Arc Pro B60; other devices are rejected during
initialization. The fixed pool is validated on this driver/device, not derived
from a general occupancy formula. A historical 48-worker static trial stalled.

The original dynamic decoder, decoder GemvBlock specializations, profiling
wrapper, and all decode tuning/selection options have been removed. This
includes `OV_MEGAKERNEL_DECODE_STATIC`, `OV_MEGAKERNEL_DECODE_GEMV_BLOCK`,
`OV_MEGAKERNEL_MONO_WG`, and the `OV_MEGAKERNEL_GEMV_BLOCK_*` options. Setting
these variables no longer changes this runtime. Prefill behavior and its
existing controls are unchanged.

After removal, 300-call checks at contexts 40/400/4000 matched pre-cleanup
fastest-decoder references bit-for-bit, including repeated-launch reset checks.
Timings were 2.58/2.67/3.66 ms; context 4000 reported 365.74 GB/s and 80.21%
estimated bandwidth. Logs: `/tmp/qwen-prune-host-*.log`.

Historical comparison before removal of the alternate decoders: B60 device 0,
seed 42, normal prefill configuration, 10 warmups and 300 measured
decode calls per process. GemvBlock and static values are medians of three
independent processes, with reversed configuration order in the second round.
The original row is a final control after restoring its original vectorized
epilogues. GPU average decode latency in milliseconds:

| Decoder | Workers | Context 40 | Context 400 | Context 4000 |
| --- | ---: | ---: | ---: | ---: |
| Original subgroup, final control | 32 | 2.75 | 2.87 | 4.19 |
| Previous GemvBlock chain + SLM, no IGC override | 35 | 2.85 | 2.93 | 3.95 |
| Indexed-static asymmetric subgroup | 34 | 2.59 | 2.68 | 3.67 |

At context 4000, static decode reported 364.99 / 365.14 / 365.20 GB/s,
or 80.04 / 80.08 / 80.09% of the 456 GB/s peak. This is about 7.1% lower
latency than the previous best GemvBlock path and 12.4% below the final original
control. A post-cleanup run reproduced 3.67 ms and 80.10%. The margin above 80%
is small and is not guaranteed across devices, drivers, or clock conditions.
Contexts 40 and 400 reached only about 75.1% and 75.9%, respectively.
These figures use the benchmark's unchanged minimum-transfer estimate;
physical DRAM traffic, cache hits, and stall counters were not measured.

All 27 confirmation runs passed saved-reference and warmup/final-output checks.
The original and previous GemvBlock paths retained exact reference equality.
Static decode changes attention subgroup partitioning from 16 to 32 subgroups
relative to the original path, so exact cross-configuration equality is not
expected. Final static checks against original references at 40/400/4000 had
relative L2 at most 0.000782 and max absolute error at most 0.00294. No tolerance
was relaxed. Before removing the worker override, counts 1 and 40 passed
repeated small-context checks; oversized, malformed, and mutually exclusive
settings were rejected as expected.

Boundary/reference checks passed at contexts 2, 3, 63, 64, 65, 255, 256, 257,
and 4095 with seed 42, at 4000 and 4095 with seed 7, and at 400 with seed 123.
For these boundary/additional-seed comparisons, the existing alternate prefill
attention path was selected with `OV_MEGAKERNEL_PREFILL_BATCHED_ATTENTION=0`
(presence of this variable disables batched attention). At 4095, the normal
batched-prefill setup made even the original decoder fail against its own fresh
saved reference (relative L2 0.00143, max absolute error 0.00489). The alternate
prefill control repeated exactly and the static candidate passed its reference
(seed 42: relative L2 0.000728, max absolute error 0.00293). Default near-capacity
reference stability remains unresolved; no prefill source was changed.

Rejected probes included contiguous cache-bypassing GEMV loads, full register
preloads, fused gate/up loops, tiled attention, base-2 softmax, and finer attention
partitions. Static scheduling alone was not faster on the original geometry;
it became useful with the asymmetric 1024-thread layout. All rejected probe
switches and implementations were removed, and shared GemvBlock helpers were
not modified in this optimization.

Session logs: `/tmp/qwen-sol-confirm-r*-*-c*.log`,
`/tmp/qwen-sol-final-m*-c*.log`, `/tmp/qwen-sol-boundary-*.log`,
`/tmp/qwen-sol-edge-*.log`, `/tmp/qwen-sol-workers-*.log`, and
`/tmp/qwen-sol-guard-*.log`. Earlier `/tmp/qwen-sol-*.log` files record screening
experiments, including rejected failures, rather than retained configurations.

## Task-system overhead comparison (2026-11-02)

The task system is restored as the only decode scheduler. The CPU builds the
same topologically ordered sequence of 6,105 descriptors (one embedding task
plus 28 layers of 218 tasks) and uploads it once during runtime initialization.
`ExecuteMkTask` dispatches the stored type/layer/tile through the existing
`WorkerMainLoop_block` implementation, including its FIFO atomic cursor, SLM
task-pointer broadcast, and final state reset. `ExecuteIndexedMkTask` and its
static strided loop have been removed.

Only scheduling and its host allocation/cleanup/build dependencies changed.
Both versions use 34 workers, 1024 threads per work-group, the same asymmetric
64/32-row projection tiles, 16,640 bytes of SLM, attention partitioning,
projection math, and producer/consumer semaphores. Prefill is unchanged.
No decode tuning flags or older GemvBlock paths were restored.

Fresh measurements on B60 device 0, seed 42, normal prefill: three independent
processes per configuration/context, each with 10 warmups and 300 timed decode
calls; prefill used one measured iteration and zero warmups. Indexed runs were
collected before the edit, task-system runs afterward, not interleaved.
Median GPU average latency, with deltas calculated from rounded milliseconds:

| Context | Indexed-static | Restored task system | Added latency | Increase vs indexed |
| ---: | ---: | ---: | ---: | ---: |
| 40 | 2.58 ms | 2.71 ms | 0.13 ms | 5.0% |
| 400 | 2.67 ms | 2.79 ms | 0.12 ms | 4.5% |
| 4000 | 3.66 ms | 3.80 ms | 0.14 ms | 3.8% |

All nine restored-task-system runs matched fresh indexed outputs bit-for-bit
(relative L2 and maximum absolute error both zero), and passed finite-output
and warmup/final-output checks. Context 4000 median estimated bandwidth changed
from 365.63 GB/s (80.18% of peak) to 352.56 GB/s (77.32%). These remain
minimum-transfer estimates, not measured physical DRAM throughput.

The roughly 0.12-0.14 ms/token delta is the combined effect of task descriptors,
queue atomics, worker-loop synchronization, dynamic work assignment, and reset
behavior. It does not isolate descriptor decoding alone, nor include CPU queue
construction or the one-time upload, which occur outside GPU decode timing.
At context 4000 the indexed scheduler saved about 3.7% of the restored scheduler's
latency with all other optimizations retained; this does not establish an
independent contribution on the older kernel geometry.

Run the restored decoder without environment flags:

```sh
MEGAKERNEL_POC/benchmarks/Qwen3_0_6B/run.sh \
  --device=0 --context-tokens=4000 \
  --prefill-iterations=1 --prefill-warmup=0 \
  --iterations=300 --warmup=10
```

Logs: `/tmp/qwen-task-indexed-r*-c*.log` and
`/tmp/qwen-task-restored-r*-c*.log`. Fresh references:
`/tmp/qwen-task-indexed-r1-c{40,400,4000}.txt`; pass one with
`--decode-reference=PATH` to repeat the cross-scheduler check.