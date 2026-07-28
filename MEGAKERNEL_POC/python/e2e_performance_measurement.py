"""End-to-end MegaKernel decode performance measurement (two-model PoC).

Usage
-----
    source /opt/home/pwysocki/openvino_dist/setupvars.sh
    /opt/home/pwysocki/.venv/bin/python e2e_performance_measurement.py
    ... --device GPU.1 --frameworks native optimum genai
    ... --decode-iters 200 --max-new-tokens 128
    ... --only-framework native
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
DEFAULT_MODEL_DIR = HERE / "qwen3-0.6b-openvino-ir"
DEFAULT_DEVICE = "GPU.1"
BATCH = 1

PROMPTS: list[dict[str, str]] = [
    {
        "name": "short",
        "text": "What is the capital of France?",
    },
    {
        "name": "medium",
        "text": (
            "Explain, in a few sentences, how a transformer neural network uses "
            "self-attention to process a sequence of tokens, and why key-value "
            "caching makes autoregressive decoding faster than recomputing the "
            "whole sequence at every step."
        ),
    },
    {
        "name": "long",
        "text": (
            "You are a senior systems engineer. Read the following background "
            "carefully and then answer the question at the end.\n\n"
            "Large language models are deployed on a wide range of hardware, from "
            "small integrated GPUs to large data-center accelerators. During "
            "inference the model first runs a prefill phase that processes the "
            "entire prompt in a single forward pass, populating the key-value "
            "cache for every attention layer. After prefill the model enters the "
            "decode phase, generating one token at a time. Each decode step reads "
            "the growing key-value cache, computes attention against all previous "
            "tokens, and appends the new key and value vectors. Because the decode "
            "phase is memory-bandwidth bound and launches many small kernels, it "
            "often dominates end-to-end latency for long generations. A megakernel "
            "fuses the many small per-layer kernels of a decode step into a single "
            "GPU kernel launch, preloading weights for the next operation while the "
            "current one computes, using fine-grained synchronization, and removing "
            "kernel launch overhead and tail effects. This is particularly valuable "
            "for small models on small GPUs where launch overhead is a large "
            "fraction of the total step time.\n\n"
            "Question: Given the description above, explain why fusing the decode "
            "step into a single megakernel is expected to improve latency more than "
            "it improves prefill, and describe one hardware limitation that could "
            "reduce the achievable speedup on a small GPU."
        ),
    },
]


def get_prompts(args) -> list[dict[str, str]]:
    """Prompt set to benchmark: a single user-supplied prompt when --prompt is
    given, otherwise the built-in short/medium/long trio."""
    custom = getattr(args, "prompt", None)
    if custom:
        return [{"name": "custom", "text": custom}]
    return PROMPTS


def prefill_inputs(input_ids: np.ndarray) -> dict[str, np.ndarray]:
    seq_len = input_ids.shape[1]
    return {
        "input_ids": input_ids.astype(np.int64),
        "attention_mask": np.ones((BATCH, seq_len), np.int64),
        "position_ids": np.arange(seq_len, dtype=np.int64).reshape(1, seq_len),
        "beam_idx": np.zeros(BATCH, np.int32),
    }


def single_token_inputs(token_id: int, position: int) -> dict[str, np.ndarray]:
    """One-token step (decode, or one step of token-by-token priming)."""
    return {
        "input_ids": np.array([[token_id]], np.int64),
        "attention_mask": np.ones((BATCH, position + 1), np.int64),
        "position_ids": np.array([[position]], np.int64),
        "beam_idx": np.zeros(BATCH, np.int32),
    }


def load_tokenizer(model_dir: Path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_dir)


def chat_text(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


def prompt_token_ids(tokenizer, prompt: str) -> np.ndarray:
    text = chat_text(tokenizer, prompt)
    ids = tokenizer([text], return_tensors="np").input_ids
    return ids.astype(np.int64)


def stats(latencies_ms: list[float]) -> dict[str, float]:
    lat = sorted(latencies_ms)
    n = len(lat)
    mean = statistics.mean(lat)
    return {
        "mean": mean,
        "median": lat[n // 2],
        "min": lat[0],
        "p90": lat[int(0.90 * (n - 1))],
        "p99": lat[int(0.99 * (n - 1))],
        "tok_s": 1000.0 / mean,
        "count": n,
    }


def native_worker(args) -> list[dict]:
    import openvino as ov

    core = ov.Core()
    dev_name = core.get_property(args.device, "FULL_DEVICE_NAME")
    model = core.read_model(str(Path(args.model_dir) / "openvino_model.xml"))
    t0 = time.perf_counter()
    compiled = core.compile_model(model, args.device)
    compile_s = time.perf_counter() - t0

    tokenizer = load_tokenizer(Path(args.model_dir))
    is_mk = args.path == "megakernel"

    # The MegaKernel primitive keeps an internal KV counter (cur_len_) per impl
    # instance and ignores position_ids, so each prompt must run on a fresh impl
    # starting at cur_len_=0. We therefore (a) never reuse a request for the
    # MegaKernel path and (b) keep every created request alive for the duration
    # of the worker so the plugin cannot recycle a stale (already-advanced) impl.
    keep_alive = []

    if not is_mk:
        # Baseline uses OpenVINO's own KV state (reset on every fresh request), so
        # a one-time throwaway warmup safely compiles kernels and keeps the first
        # measured prefill from paying the lazy-compilation cost.
        warm = compiled.create_infer_request()
        warm.infer(prefill_inputs(np.ones((BATCH, 8), np.int64)))
        for pos in range(8, 12):
            warm.infer(single_token_inputs(1, pos))
        keep_alive.append(warm)

    results = []
    for prompt in get_prompts(args):
        ids = prompt_token_ids(tokenizer, prompt["text"])
        prompt_len = int(ids.shape[1])
        req = compiled.create_infer_request()
        keep_alive.append(req)

        prefill_ms = float("nan")
        if is_mk:
            # Two-model PoC: the MegaKernel model cannot run a real multi-token
            # prefill (DECODE_ONLY refuses S>1), so it "prefills" by growing its
            # internal KV cache one token at a time. Time that priming loop so it
            # is reported as the MegaKernel prefill cost (it is NOT free).
            t = time.perf_counter()
            for pos in range(prompt_len):
                req.infer(single_token_inputs(int(ids[0, pos]), pos))
            prefill_ms = (time.perf_counter() - t) * 1e3
            res = req.results
        else:
            t = time.perf_counter()
            res = req.infer(prefill_inputs(ids))
            prefill_ms = (time.perf_counter() - t) * 1e3

        logits = np.array(res[0])[0, -1, :].astype(np.float32)
        next_id = int(logits.argmax())

        past = prompt_len
        for _ in range(args.warmup):
            req.infer(single_token_inputs(next_id, past))
            past += 1
        lat = []
        for _ in range(args.decode_iters):
            inp = single_token_inputs(next_id, past)
            t = time.perf_counter()
            req.infer(inp)
            lat.append((time.perf_counter() - t) * 1e3)
            past += 1

        # Greedy-generate real output for output similarity comparison.
        # Uses a fresh request so the timed benchmark loop above is not affected.
        text_out = native_generate_text(compiled, tokenizer, ids, prompt_len,
                                        is_mk, args.max_new_tokens, keep_alive)

        results.append({
            "prompt": prompt["name"],
            "prompt_len": prompt_len,
            "prefill_ms": prefill_ms,
            "n_tok": args.decode_iters,
            "decode": stats(lat),
            "argmax": next_id,
            "logits": logits.tolist(),
            "device": dev_name,
            "compile_s": compile_s,
            "text": text_out,
        })
    return results


def native_generate_text(compiled, tokenizer, ids: np.ndarray, prompt_len: int,
                         is_mk: bool, max_new_tokens: int, keep_alive: list) -> str:
    """Greedy-decode real tokens from a fresh request and detokenize."""
    gen_req = compiled.create_infer_request()
    keep_alive.append(gen_req)
    r = None
    if is_mk:
        for pos in range(prompt_len):
            r = gen_req.infer(single_token_inputs(int(ids[0, pos]), pos))
    else:
        r = gen_req.infer(prefill_inputs(ids))
    cur = int(np.array(r[0])[0, -1, :].argmax())
    pos = prompt_len
    eos = tokenizer.eos_token_id
    gen_ids: list[int] = []
    for _ in range(max_new_tokens):
        gen_ids.append(cur)
        if eos is not None and cur == eos:
            break
        r = gen_req.infer(single_token_inputs(cur, pos))
        pos += 1
        cur = int(np.array(r[0])[0, -1, :].argmax())
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


def optimum_worker(args) -> list[dict]:
    from optimum.intel import OVModelForCausalLM
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    t0 = time.perf_counter()
    # CACHE_DIR="" disables the compiled-model blob cache. The cache key does not
    # include the OV_MEGAKERNEL_DISABLE env var, so leaving it on would make the
    # MegaKernel run silently reuse the baseline blob (no transformation).
    model = OVModelForCausalLM.from_pretrained(
        args.model_dir, device=args.device, ov_config={"CACHE_DIR": ""})
    compile_s = time.perf_counter() - t0

    import openvino as ov
    dev_name = ov.Core().get_property(args.device, "FULL_DEVICE_NAME")

    def gen(model_inputs, n_new):
        t = time.perf_counter()
        out = model.generate(
            **model_inputs,
            max_new_tokens=n_new,
            do_sample=False,
            num_beams=1,
        )
        return (time.perf_counter() - t) * 1e3, out

    results = []
    for prompt in get_prompts(args):
        text = chat_text(tokenizer, prompt["text"])
        model_inputs = tokenizer([text], return_tensors="pt")
        prompt_len = int(model_inputs.input_ids.shape[1])

        # warmup
        for _ in range(max(1, args.gen_warmup)):
            gen(model_inputs, args.max_new_tokens)

        # TTFT (prefill only) = generate exactly one new token.
        ttft = []
        for _ in range(args.gen_iters):
            ms, _ = gen(model_inputs, 1)
            ttft.append(ms)
        # Full generation of max_new_tokens.
        full = []
        last_out = None
        for _ in range(args.gen_iters):
            ms, last_out = gen(model_inputs, args.max_new_tokens)
            full.append(ms)

        ttft_mean = statistics.mean(ttft)
        full_mean = statistics.mean(full)
        out_ids = last_out[0][prompt_len:].tolist()
        actual_n_tok = len(out_ids)
        # TTFT measures 1-token generation; remaining (actual_n_tok - 1) tokens
        # are decode steps. Guard against degenerate short outputs.
        n_decode = max(actual_n_tok - 1, 1)
        decode_total = max(full_mean - ttft_mean, 1e-6)
        per_tok_ms = decode_total / n_decode
        argmax = int(out_ids[0]) if out_ids else -1
        gen_text = tokenizer.decode(out_ids, skip_special_tokens=True)

        results.append({
            "prompt": prompt["name"],
            "prompt_len": prompt_len,
            "prefill_ms": ttft_mean,
            "n_tok": actual_n_tok,
            "decode": {
                "mean": per_tok_ms,
                "median": per_tok_ms,
                "tok_s": 1000.0 / per_tok_ms,
                "count": n_decode,
            },
            "argmax": argmax,
            "device": dev_name,
            "compile_s": compile_s,
            "text": gen_text,
        })
    return results


def genai_worker(args) -> list[dict]:
    import openvino_genai as ov_genai
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    t0 = time.perf_counter()
    # CACHE_DIR="" disables the compiled-model blob cache (see optimum_worker).
    # Baseline uses GenAI's default PagedAttention backend.
    pipeline_kwargs: dict = {"CACHE_DIR": ""}
    if args.path == "megakernel":
        pipeline_kwargs["ATTENTION_BACKEND"] = "SDPA"
    pipe = ov_genai.LLMPipeline(args.model_dir, args.device, **pipeline_kwargs)
    compile_s = time.perf_counter() - t0

    import openvino as ov
    dev_name = ov.Core().get_property(args.device, "FULL_DEVICE_NAME")

    cfg = ov_genai.GenerationConfig()
    cfg.max_new_tokens = args.max_new_tokens
    cfg.do_sample = False
    cfg.num_beams = 1

    results = []
    for prompt in get_prompts(args):
        text = chat_text(tokenizer, prompt["text"])
        prompt_len = int(tokenizer([text], return_tensors="np").input_ids.shape[1])

        for _ in range(max(1, args.gen_warmup)):
            pipe.generate([text], cfg)

        ttft_ms, tpot_ms, tput = [], [], []
        gen_text = ""
        for _ in range(args.gen_iters):
            res = pipe.generate([text], cfg)
            pm = res.perf_metrics
            ttft_ms.append(pm.get_ttft().mean)
            tpot_ms.append(pm.get_tpot().mean)     # decode: mean ms / output token
            tput.append(pm.get_throughput().mean)
            gen_text = res.texts[0] if getattr(res, "texts", None) else str(res)

        tpot_mean = statistics.mean(tpot_ms)
        # Count actual output tokens from the last generation by re-tokenizing.
        actual_n_tok = len(tokenizer.encode(gen_text, add_special_tokens=False)) if gen_text else args.max_new_tokens
        results.append({
            "prompt": prompt["name"],
            "prompt_len": prompt_len,
            "prefill_ms": statistics.mean(ttft_ms),
            "n_tok": actual_n_tok,
            "decode": {
                "mean": tpot_mean,
                "median": tpot_mean,
                "tok_s": 1000.0 / tpot_mean,
                "throughput_tok_s": statistics.mean(tput),
                "count": actual_n_tok,
            },
            "device": dev_name,
            "compile_s": compile_s,
            "text": gen_text,
        })
    return results


WORKERS = {"native": native_worker, "optimum": optimum_worker, "genai": genai_worker}


def worker_env(framework: str, path: str) -> dict:
    env = os.environ.copy()
    if path == "baseline":
        env["OV_MEGAKERNEL_DISABLE"] = "1"
    else:
        env["OV_MEGAKERNEL_DISABLE"] = "0"
        # Only the native manual loop is strictly decode-only. Optimum/GenAI run
        # a real (multi-token) prefill inside generate(), so the guard must stay
        # off there — their decode is isolated via TPOT / full-minus-TTFT instead.
        if framework == "native":
            env["OV_MEGAKERNEL_DECODE_ONLY"] = "1"
    return env


def spawn(framework: str, path: str, args) -> list[dict]:
    cmd = [
        sys.executable, __file__, "--worker", framework, "--path", path,
        "--model-dir", str(args.model_dir), "--device", args.device,
        "--warmup", str(args.warmup), "--decode-iters", str(args.decode_iters),
        "--gen-warmup", str(args.gen_warmup), "--gen-iters", str(args.gen_iters),
        "--max-new-tokens", str(args.max_new_tokens),
    ]
    if getattr(args, "prompt", None):
        cmd += ["--prompt", args.prompt]
    out = subprocess.run(cmd, env=worker_env(framework, path), capture_output=True, text=True)
    if out.returncode != 0:
        sys.stdout.write(out.stdout)
        sys.stderr.write(out.stderr)
        raise RuntimeError(f"{framework}/{path} worker failed (exit {out.returncode})")
    lines = [ln for ln in out.stdout.strip().splitlines() if ln.strip()]
    for ln in lines[:-1]:
        print(f"    [{framework}/{path}] {ln}")
    return json.loads(lines[-1])


def cosine(a: list[float], b: list[float]) -> float:
    va, vb = np.asarray(a), np.asarray(b)
    return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9))


def report_framework(framework: str, base: list[dict], mega: list[dict],
                     max_new_tokens: int) -> None:
    print(f"\n{'=' * 92}\n {framework.upper()}\n{'=' * 92}")
    genai_note = "  [baseline=PA, megakernel=SDPA+MK]" if framework == "genai" else ""
    header = (f"  {'prompt':<8} {'ctx':>5} | {'base pf ms':>10} {'mega pf ms':>10} | "
              f"{'base dec(mean)':>14} {'mega dec(mean)':>14} | "
              f"{'dec x':>6} {f'e2e x@{max_new_tokens}':>10}{genai_note}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for b, m in zip(base, mega):
        bd = b["decode"]["mean"]
        md = m["decode"]["mean"]
        sp = bd / md if md else float("nan")
        bpf = b.get("prefill_ms", float("nan"))
        mpf = m.get("prefill_ms", float("nan"))
        n_dec = max(max_new_tokens - 1, 0)
        base_e2e = bpf + n_dec * bd
        mega_e2e = mpf + n_dec * md
        e2e = base_e2e / mega_e2e if mega_e2e else float("nan")
        extra = ""
        if framework == "native" and "logits" in b and "logits" in m:
            extra = f"  argmatch={b['argmax'] == m['argmax']} cos={cosine(b['logits'], m['logits']):.4f}"
        print(f"  {b['prompt']:<8} {b['prompt_len']:>5} | {bpf:>10.3f} {mpf:>10.3f} | "
              f"{bd:>14.3f} {md:>14.3f} | "
              f"{sp:>5.2f}x {e2e:>9.2f}x{extra}")
    # Print generated text from both paths so outputs can be compared directly.
    if any(x.get("text") for x in base + mega):
        print("  Generated text (baseline vs megakernel):")
        for b, m in zip(base, mega):
            print(f"    [{b['prompt']}] baseline  : {(b.get('text') or '').strip()!r}")
            print(f"    [{b['prompt']}] megakernel: {(m.get('text') or '').strip()!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--device", default=DEFAULT_DEVICE, help="GPU.1 = B60 dGPU")
    ap.add_argument("--frameworks", nargs="+", default=["native", "optimum", "genai"],
                    choices=list(WORKERS))
    ap.add_argument("--only-framework", choices=list(WORKERS), default=None,
                    help="Run a single framework (overrides --frameworks).")
    # native decode benchmark
    ap.add_argument("--warmup", type=int, default=20, help="native decode warmup steps")
    ap.add_argument("--decode-iters", type=int, default=200, help="native timed decode steps")
    # optimum / genai generate() benchmark
    ap.add_argument("--gen-warmup", type=int, default=1, help="generate() warmup calls")
    ap.add_argument("--gen-iters", type=int, default=3, help="generate() measured calls")
    ap.add_argument("--max-new-tokens", type=int, default=128, help="tokens per generate() call")
    ap.add_argument("--prompt", default=None,
                    help="Run a single custom prompt (e.g. --prompt \"What's my name?\") "
                         "and print the text the model generates. Overrides the built-in prompts.")
    # internal
    ap.add_argument("--worker", choices=list(WORKERS), default=None, help=argparse.SUPPRESS)
    ap.add_argument("--path", choices=("baseline", "megakernel"), default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.worker:
        args.model_dir = str(args.model_dir)
        results = WORKERS[args.worker](args)
        print(json.dumps(results))
        return

    frameworks = [args.only_framework] if args.only_framework else args.frameworks
    print(f"Device: {args.device}   frameworks: {frameworks}")
    print(f"native decode: warmup={args.warmup} iters={args.decode_iters}   "
          f"generate: warmup={args.gen_warmup} iters={args.gen_iters} "
          f"max_new_tokens={args.max_new_tokens}")

    all_results: dict[str, dict[str, list[dict]]] = {}
    for fw in frameworks:
        all_results[fw] = {}
        for path in ("baseline", "megakernel"):
            print(f"\n>>> running {fw}/{path} ...")
            all_results[fw][path] = spawn(fw, path, args)

    for fw in frameworks:
        report_framework(fw, all_results[fw]["baseline"], all_results[fw]["megakernel"],
                         args.max_new_tokens)

    h0 = f" {'framework':<10} {'prompt':<8} {'in_tok':>6} {'gen b/m':>9}"
    h1 = f" | {'ttft_ms':>8} {'ms/tok':>8} {'decode_ms':>10} {'total_ms':>10}"   # baseline
    h2 = f" | {'ttft_ms':>8} {'ms/tok':>8} {'decode_ms':>10} {'total_ms':>10}"   # megakernel
    h3 = f" | {'decode_x':>8} {'e2e_x':>7}"
    W = len(h0) + len(h1) + len(h2) + len(h3) + 2

    print(f"\n{'=' * W}")
    print(" SUMMARY  (MegaKernel vs baseline, real measured times)")
    # Sub-header labels for the two time blocks
    bl = len(h1)
    base_lbl = "baseline".center(bl - 3)
    mega_lbl = "megakernel".center(bl - 3)
    print(f"{' ' * len(h0)} | {base_lbl} | {mega_lbl} |")
    print(h0 + h1 + h2 + h3)
    print(" " + "-" * (W - 1))
    for fw in frameworks:
        base, mega = all_results[fw]["baseline"], all_results[fw]["megakernel"]
        for b, m in zip(base, mega):
            bd  = b["decode"]["mean"]   # measured mean per-token decode latency (ms)
            md  = m["decode"]["mean"]
            bpf = b.get("prefill_ms", float("nan"))   # prefill / time-to-first-token (ms)
            mpf = m.get("prefill_ms", float("nan"))
            b_n_tok = b.get("n_tok", b["decode"]["count"])   # actual tokens generated
            m_n_tok = m.get("n_tok", m["decode"]["count"])
            # Real total decode time over the actual number of tokens each path
            # generated (the two paths may generate different counts).
            b_dec_ms = max(b_n_tok - 1, 0) * bd
            m_dec_ms = max(m_n_tok - 1, 0) * md
            base_tot = bpf + b_dec_ms
            mega_tot = mpf + m_dec_ms
            dec_x  = bd / md if md else float("nan")
            e2e_x  = base_tot / mega_tot if mega_tot else float("nan")
            gen_bm = f"{b_n_tok}/{m_n_tok}"
            print(f" {fw:<10} {b['prompt']:<8} {b['prompt_len']:>6} {gen_bm:>9}"
                  f" | {bpf:>8.1f} {bd:>8.3f} {b_dec_ms:>10.1f} {base_tot:>10.1f}"
                  f" | {mpf:>8.1f} {md:>8.3f} {m_dec_ms:>10.1f} {mega_tot:>10.1f}"
                  f" | {dec_x:>7.2f}x {e2e_x:>6.2f}x")
    print(f"{'=' * W}")
    print(" Legend:")
    print("   in_tok     input (prompt) token count")
    print("   gen b/m    tokens actually generated (baseline/megakernel); may differ")
    print("              because greedy decoding can stop at different points")
    print("   ttft_ms    prefill latency = time to first token (ms)")
    print("   ms/tok     measured mean per-token decode latency (ms/token)")
    print("   decode_ms  real total decode time = (gen - 1) * ms/tok")
    print("   total_ms   ttft_ms + decode_ms (real end-to-end for the tokens generated)")
    print("   decode_x   per-token decode speedup (baseline ms/tok / megakernel ms/tok)")
    print("   e2e_x      end-to-end speedup (baseline total_ms / megakernel total_ms)")


if __name__ == "__main__":
    main()
