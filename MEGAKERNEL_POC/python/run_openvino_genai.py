"""Run a converted Qwen3 OpenVINO model with OpenVINO GenAI.

Example:
    python3 run_openvino_genai.py \
        --model-dir ./qwen3-0.6b-openvino-ir \
        --prompt "Give me a short introduction to large language model." \
        --max-new-tokens 256
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path
from typing import Any, cast


DEFAULT_MODEL_DIR = "qwen3-0.6b-openvino-ir"
DEFAULT_PROMPT = "Give me a short introduction to large language model."
DESCRIPTION = "Run a converted Qwen3 OpenVINO model with OpenVINO GenAI."


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
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="Directory with the converted OpenVINO IR model.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="User prompt for generation.")
    parser.add_argument("--device", default="GPU", help="OpenVINO device, for example CPU, GPU, or AUTO.")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Maximum number of tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.8, help="Nucleus sampling top-p value.")
    parser.add_argument("--top-k", type=int, default=20, help="Top-k sampling value.")
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling. By default generation is greedy.")
    parser.add_argument("--warmup-iterations", type=non_negative_int, default=5, help="Number of warmup generate calls.")
    parser.add_argument("--benchmark-iterations", type=positive_int, default=10, help="Number of measured generate calls.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code=True while loading tokenizer.")
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help="Disable Qwen chat-template thinking mode when the tokenizer supports it.",
    )
    parser.add_argument(
        "--raw-prompt",
        action="store_true",
        help="Send --prompt directly to the GenAI pipeline instead of applying the chat template.",
    )
    return parser.parse_args()


def load_tokenizer(model_dir: Path, trust_remote_code: bool):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Loading the tokenizer requires transformers.") from exc

    tokenizer_kwargs = {
        "trust_remote_code": trust_remote_code,
        "fix_mistral_regex": True,
    }
    try:
        return AutoTokenizer.from_pretrained(model_dir, **tokenizer_kwargs)
    except TypeError:
        tokenizer_kwargs.pop("fix_mistral_regex")
        return AutoTokenizer.from_pretrained(model_dir, **tokenizer_kwargs)


def ensure_openvino_tokenizer(model_dir: Path, tokenizer) -> None:
    tokenizer_path = model_dir / "openvino_tokenizer.xml"
    detokenizer_path = model_dir / "openvino_detokenizer.xml"
    if tokenizer_path.exists() and detokenizer_path.exists():
        return

    try:
        from openvino import save_model
        from openvino_tokenizers import convert_tokenizer
    except ImportError as exc:
        raise RuntimeError(
            "OpenVINO GenAI requires openvino_tokenizer.xml for string prompts. "
            "Install openvino-tokenizers or add the tokenizer IR files to the model directory."
        ) from exc

    ov_tokenizer, ov_detokenizer = cast(tuple[Any, Any], convert_tokenizer(tokenizer, with_detokenizer=True))
    save_model(ov_tokenizer, tokenizer_path)
    save_model(ov_detokenizer, detokenizer_path)
    print(f"Saved OpenVINO tokenizer to: {tokenizer_path}")
    print(f"Saved OpenVINO detokenizer to: {detokenizer_path}")


def build_prompt(tokenizer, prompt: str, raw_prompt: bool, enable_thinking: bool) -> str:
    if raw_prompt:
        return prompt

    messages = [{"role": "user", "content": prompt}]
    template_kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=enable_thinking, **template_kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **template_kwargs)


def build_generation_config(ov_genai, args: argparse.Namespace):
    generation_config = ov_genai.GenerationConfig()
    generation_config.max_new_tokens = args.max_new_tokens
    generation_config.do_sample = args.do_sample
    generation_config.temperature = args.temperature
    generation_config.top_p = args.top_p
    generation_config.top_k = args.top_k
    return generation_config


def generate_text(pipeline, prompt: str, generation_config) -> tuple[str, float]:
    start_time = time.perf_counter()
    result = pipeline.generate(prompt, generation_config)
    latency = time.perf_counter() - start_time

    if isinstance(result, str):
        return result, latency

    texts = getattr(result, "texts", None)
    if texts:
        return texts[0], latency

    return str(result), latency


def benchmark_generate(pipeline, prompt: str, generation_config, warmup_iterations: int, benchmark_iterations: int) -> tuple[str, list[float]]:
    for index in range(warmup_iterations):
        _, latency = generate_text(pipeline, prompt, generation_config)
        print(f"Warmup {index + 1}/{warmup_iterations}: {latency:.6f} s")

    generated_text = ""
    latencies = []
    for index in range(benchmark_iterations):
        generated_text, latency = generate_text(pipeline, prompt, generation_config)
        latencies.append(latency)
        print(f"Benchmark {index + 1}/{benchmark_iterations}: {latency:.6f} s")

    return generated_text, latencies


def print_latency_summary(latencies: list[float]) -> None:
    print("model.generate() latency summary:")
    print(f"  iterations: {len(latencies)}")
    print(f"  mean:       {statistics.fmean(latencies):.6f} s")
    print(f"  median:     {statistics.median(latencies):.6f} s")
    print(f"  min:        {min(latencies):.6f} s")
    print(f"  max:        {max(latencies):.6f} s")


def split_qwen_thinking(text: str) -> tuple[str, str]:
    end_tag = "</think>"
    if end_tag not in text:
        return "", text

    thinking_text, answer_text = text.split(end_tag, 1)
    thinking_text = thinking_text.replace("<think>", "", 1).strip("\n")
    return thinking_text, answer_text.strip("\n")


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}. Run convert_to_openvino_ir.py first.")

    try:
        import openvino_genai as ov_genai
    except ImportError as exc:
        raise RuntimeError("Missing openvino_genai. Install OpenVINO GenAI before running this script.") from exc

    tokenizer = load_tokenizer(model_dir, trust_remote_code=args.trust_remote_code)
    ensure_openvino_tokenizer(model_dir, tokenizer)
    prompt = build_prompt(
        tokenizer=tokenizer,
        prompt=args.prompt,
        raw_prompt=args.raw_prompt,
        enable_thinking=not args.no_thinking,
    )
    pipeline = ov_genai.LLMPipeline(str(model_dir), args.device)
    generation_config = build_generation_config(ov_genai, args)
    generated_text, latencies = benchmark_generate(
        pipeline=pipeline,
        prompt=prompt,
        generation_config=generation_config,
        warmup_iterations=args.warmup_iterations,
        benchmark_iterations=args.benchmark_iterations,
    )
    thinking_content, answer_content = split_qwen_thinking(generated_text)

    if thinking_content:
        print("thinking content:", thinking_content)
    print("content:", answer_content)
    print_latency_summary(latencies)


if __name__ == "__main__":
    main()