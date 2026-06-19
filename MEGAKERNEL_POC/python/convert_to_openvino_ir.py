"""Convert Qwen3-0.6B to OpenVINO IR with Optimum Intel.

Example:
    python convert_to_openvino_ir.py \
        --model-id Qwen/Qwen3-0.6B \
        --output-dir ./qwen3-0.6b-openvino-ir \
        --weight-format fp16
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


DEFAULT_MODEL_ID = "Qwen/Qwen3-0.6B"
DEFAULT_OUTPUT_DIR = "qwen3-0.6b-openvino-ir"
DEFAULT_TASK = "text-generation-with-past"
DESCRIPTION = "Convert Qwen3-0.6B to OpenVINO IR with Optimum Intel."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Hugging Face model id or local model directory.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory where the OpenVINO IR will be saved.")
    parser.add_argument("--task", default=DEFAULT_TASK, help="Optimum export task.")
    parser.add_argument(
        "--weight-format",
        default="fp16",
        choices=("fp32", "fp16", "int8", "int4"),
        help="Weight format passed to optimum-cli export openvino.",
    )
    parser.add_argument("--trust-remote-code", action="store_true", help="Pass --trust-remote-code to the exporter.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove an existing output directory before conversion.",
    )
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Extra argument forwarded to optimum-cli. Repeat for multiple arguments.",
    )
    return parser.parse_args()


def ensure_exporter_available() -> None:
    if shutil.which("optimum-cli") is None:
        raise RuntimeError(
            "optimum-cli was not found. Install the exporter dependencies first, for example:\n"
            "  python -m pip install 'optimum-intel[openvino]' transformers accelerate"
        )


def validate_ir_dir(output_dir: Path) -> None:
    xml_files = sorted(output_dir.glob("*.xml"))
    bin_files = sorted(output_dir.glob("*.bin"))
    if not xml_files:
        raise RuntimeError(f"Conversion finished, but no .xml IR files were found in {output_dir}")

    print("\nOpenVINO IR files:")
    for path in xml_files + bin_files:
        print(f"  {path}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    ensure_exporter_available()

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}\n"
                "Pass --overwrite to replace it, or choose another --output-dir."
            )
        shutil.rmtree(output_dir)

    command = [
        "optimum-cli",
        "export",
        "openvino",
        "--model",
        args.model_id,
        "--task",
        args.task,
        "--weight-format",
        args.weight_format,
    ]
    if args.trust_remote_code:
        command.append("--trust-remote-code")
    command.extend(args.extra_arg)
    command.append(str(output_dir))

    print("Running conversion command:")
    print("  " + " ".join(command))
    subprocess.run(command, check=True)

    validate_ir_dir(output_dir)
    print(f"\nSaved OpenVINO model to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()