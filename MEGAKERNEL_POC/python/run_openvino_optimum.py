"""Run a Qwen3 OpenVINO IR model exported with convert_to_openvino_ir.py.

Example:
    python run_openvino_ir.py \
        --model-dir ./qwen3-0.6b-openvino-ir \
        --prompt "Give me a short introduction to large language model." \
        --max-new-tokens 256
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path


DEFAULT_MODEL_DIR = "qwen3-0.6b-openvino-ir"
DEFAULT_PROMPT = "Give me a short introduction to large language model."
THINK_END_TOKEN_ID = 151668


def non_negative_int(value: str) -> int:
    parsed_value = int(value)
    if parsed_value < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed_value


def positive_int(value: str) -> int:
    parsed_value = int(value)
    if parsed_value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed_value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="Directory produced by convert_to_openvino_ir.py.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="User prompt for generation.")
    parser.add_argument("--device", default="GPU", help="OpenVINO device, for example CPU, GPU, or AUTO.")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Maximum number of tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.8, help="Nucleus sampling top-p value.")
    parser.add_argument("--top-k", type=int, default=20, help="Top-k sampling value.")
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling. By default generation is greedy.")
    parser.add_argument("--warmup-iterations", type=non_negative_int, default=5, help="Number of warmup generate calls.")
    parser.add_argument("--benchmark-iterations", type=positive_int, default=10, help="Number of measured generate calls.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code=True while loading.")
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help="Disable Qwen chat-template thinking mode when the tokenizer supports it.",
    )
    parser.add_argument(
        "--raw-prompt",
        action="store_true",
        help="Tokenize --prompt directly instead of wrapping it in the chat template.",
    )
    return parser.parse_args()


def load_model_and_tokenizer(model_dir: Path, device: str, trust_remote_code: bool):
    try:
        from optimum.intel import OVModelForCausalLM
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Missing runtime dependencies. Install them first, for example:\n"
            "  python -m pip install 'optimum-intel[openvino]' transformers accelerate"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=trust_remote_code)
    model = OVModelForCausalLM.from_pretrained(
        model_dir,
        device=device,
        trust_remote_code=trust_remote_code,
    )
    return model, tokenizer


def build_inputs(tokenizer, prompt: str, raw_prompt: bool, enable_thinking: bool):
    if raw_prompt:
        text = prompt
    else:
        messages = [{"role": "user", "content": prompt}]
        template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        try:
            text = tokenizer.apply_chat_template(
                messages,
                enable_thinking=enable_thinking,
                **template_kwargs,
            )
        except TypeError:
            text = tokenizer.apply_chat_template(messages, **template_kwargs)

    return tokenizer([text], return_tensors="pt")


def split_qwen_thinking(output_ids: list[int]) -> tuple[list[int], list[int]]:
    try:
        index = len(output_ids) - output_ids[::-1].index(THINK_END_TOKEN_ID)
    except ValueError:
        return [], output_ids
    return output_ids[:index], output_ids[index:]


def generate_ids(model, model_inputs, args: argparse.Namespace):
    start_time = time.perf_counter()
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )
    latency = time.perf_counter() - start_time
    return generated_ids, latency


def benchmark_generate(model, model_inputs, args: argparse.Namespace) -> tuple[object, list[float]]:
    for index in range(args.warmup_iterations):
        _, latency = generate_ids(model, model_inputs, args)
        print(f"Warmup {index + 1}/{args.warmup_iterations}: {latency:.6f} s")

    generated_ids = None
    latencies = []
    for index in range(args.benchmark_iterations):
        generated_ids, latency = generate_ids(model, model_inputs, args)
        latencies.append(latency)
        print(f"Benchmark {index + 1}/{args.benchmark_iterations}: {latency:.6f} s")

    return generated_ids, latencies


def print_latency_summary(latencies: list[float]) -> None:
    print("model.generate() latency summary:")
    print(f"  iterations: {len(latencies)}")
    print(f"  mean:       {statistics.fmean(latencies):.6f} s")
    print(f"  median:     {statistics.median(latencies):.6f} s")
    print(f"  min:        {min(latencies):.6f} s")
    print(f"  max:        {max(latencies):.6f} s")


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}. Run convert_to_openvino_ir.py first.")

    model, tokenizer = load_model_and_tokenizer(model_dir, args.device, args.trust_remote_code)
    model_inputs = build_inputs(
        tokenizer=tokenizer,
        prompt=args.prompt,
        raw_prompt=args.raw_prompt,
        enable_thinking=not args.no_thinking,
    )

    generated_ids, latencies = benchmark_generate(model, model_inputs, args)
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
    thinking_ids, answer_ids = split_qwen_thinking(output_ids)

    thinking_content = tokenizer.decode(thinking_ids, skip_special_tokens=True).strip("\n")
    answer_content = tokenizer.decode(answer_ids, skip_special_tokens=True).strip("\n")

    if thinking_content:
        print("thinking content:", thinking_content)
    print("content:", answer_content)
    print_latency_summary(latencies)


if __name__ == "__main__":
    main()