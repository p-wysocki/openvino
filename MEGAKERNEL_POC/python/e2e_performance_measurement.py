"""
Usage
-----
    python e2e_performance_measurement.py
    python e2e_performance_measurement.py --device GPU.1 --iters 200
    python e2e_performance_measurement.py --only baseline
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
DEFAULT_MEGAKERNEL_LIB = None
SEQ_LEN = 8
BATCH = 1
MODEL_ID = "Qwen/Qwen3-0.6B"


# ---------------------------------------------------------------------------
# Model conversion (reuse convert_to_openvino_ir helpers)
# ---------------------------------------------------------------------------
def ensure_model(model_dir: Path) -> None:
    if (model_dir / "openvino_model.xml").exists():
        return
    print(f"[setup] IR not found at {model_dir}; converting {MODEL_ID} (fp16)…")
    import convert_to_openvino_ir as conv

    conv.ensure_exporter_available()
    cmd = ["optimum-cli", "export", "openvino", "--model", MODEL_ID,
           "--task", "text-generation-with-past", "--weight-format", "fp16",
           str(model_dir)]
    subprocess.run(cmd, check=True)
    conv.validate_ir_dir(model_dir)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def prefill_inputs(seq_len: int) -> dict[str, np.ndarray]:
    return {
        "input_ids": np.ones((BATCH, seq_len), np.int64),
        "attention_mask": np.ones((BATCH, seq_len), np.int64),
        "position_ids": np.arange(seq_len, dtype=np.int64).reshape(1, seq_len),
        "beam_idx": np.zeros(BATCH, np.int32),
    }


def decode_inputs(past: int) -> dict[str, np.ndarray]:
    return {
        "input_ids": np.ones((BATCH, 1), np.int64),
        "attention_mask": np.ones((BATCH, past + 1), np.int64),
        "position_ids": np.array([[past]], np.int64),
        "beam_idx": np.zeros(BATCH, np.int32),
    }


# ---------------------------------------------------------------------------
# Worker: time one path in an isolated process and emit JSON on stdout
# ---------------------------------------------------------------------------
def run_worker(model_dir: Path, device: str, warmup: int, iters: int) -> dict:
    import openvino as ov

    core = ov.Core()
    dev_name = core.get_property(device, "FULL_DEVICE_NAME")
    model = core.read_model(str(model_dir / "openvino_model.xml"))
    t0 = time.perf_counter()
    compiled = core.compile_model(model, device)
    compile_s = time.perf_counter() - t0
    req = compiled.create_infer_request()

    # functional decode logits for cross-path verification
    req.infer(prefill_inputs(SEQ_LEN))
    res = req.infer(decode_inputs(SEQ_LEN))
    logits = np.array(res[0])[0, -1, :].astype(np.float32)

    past = SEQ_LEN + 1
    for _ in range(warmup):
        req.infer(decode_inputs(past)); past += 1
    lat = []
    for _ in range(iters):
        inp = decode_inputs(past)
        t = time.perf_counter(); req.infer(inp); lat.append((time.perf_counter() - t) * 1e3)
        past += 1
    lat.sort()
    return {
        "device": dev_name,
        "compile_s": compile_s,
        "mean": statistics.mean(lat),
        "median": lat[len(lat) // 2],
        "min": lat[0],
        "p90": lat[int(0.9 * (len(lat) - 1))],
        "p99": lat[int(0.99 * (len(lat) - 1))],
        "tok_s": 1000.0 / statistics.mean(lat),
        "argmax": int(logits.argmax()),
        "logits": logits.tolist(),
    }


# ---------------------------------------------------------------------------
# Parent: spawn one worker per path with the right env, then compare
# ---------------------------------------------------------------------------
def spawn(path: str, args) -> dict:
    env = os.environ.copy()
    env["OV_MEGAKERNEL_DISABLE"] = "1" if path == "baseline" else "0"
    if path == "megakernel" and args.megakernel_lib:
        lib = str(args.megakernel_lib)
        env["LD_LIBRARY_PATH"] = lib + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    cmd = [sys.executable, __file__, "--worker", path,
           "--model-dir", str(args.model_dir), "--device", args.device,
           "--warmup", str(args.warmup), "--iters", str(args.iters)]
    out = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if out.returncode != 0:
        print(out.stdout); print(out.stderr, file=sys.stderr)
        raise RuntimeError(f"{path} worker failed (exit {out.returncode})")
    return json.loads(out.stdout.strip().splitlines()[-1])


def report(name: str, r: dict) -> None:
    print(f"\n=== {name} on {r['device']} ===")
    print(f"  compile          : {r['compile_s']:.2f} s")
    print(f"  decode mean      : {r['mean']:.3f} ms")
    print(f"  decode median    : {r['median']:.3f} ms")
    print(f"  decode min       : {r['min']:.3f} ms")
    print(f"  decode p90/p99   : {r['p90']:.3f} / {r['p99']:.3f} ms")
    print(f"  throughput       : {r['tok_s']:.1f} tokens/s")
    print(f"  next-token argmax: {r['argmax']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--device", default=DEFAULT_DEVICE, help="GPU.1 = B60 dGPU")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--only", choices=("baseline", "megakernel"), default=None)
    ap.add_argument("--megakernel-lib", type=Path, default=DEFAULT_MEGAKERNEL_LIB,
                    help="dir with the megakernel-enabled GPU plugin .so")
    ap.add_argument("--worker", choices=("baseline", "megakernel"), default=None,
                    help=argparse.SUPPRESS)  # internal child mode
    args = ap.parse_args()

    if args.worker:
        print(json.dumps(run_worker(args.model_dir, args.device, args.warmup, args.iters)))
        return

    ensure_model(args.model_dir)
    paths = [args.only] if args.only else ["baseline", "megakernel"]
    results = {p: spawn(p, args) for p in paths}
    for p in paths:
        report(p, results[p])

    if "baseline" in results and "megakernel" in results:
        b, m = results["baseline"], results["megakernel"]
        sp = b["median"] / m["median"]
        cos = float(np.dot(b["logits"], m["logits"]) /
                    (np.linalg.norm(b["logits"]) * np.linalg.norm(m["logits"]) + 1e-9))
        print("\n=== comparison ===")
        print(f"  baseline median : {b['median']:.3f} ms  ({b['tok_s']:.1f} tok/s)")
        print(f"  megakernel median: {m['median']:.3f} ms ({m['tok_s']:.1f} tok/s)")
        print(f"  speedup          : {sp:.2f}x")
        print(f"  output cosine    : {cos:.6f}  argmax_match={b['argmax'] == m['argmax']}")


if __name__ == "__main__":
    main()
