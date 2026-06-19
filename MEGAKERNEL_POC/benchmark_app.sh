#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^[1-9][0-9]*$ ]]; then
	echo "Usage: $0 <attention_mask_size>" >&2
	exit 1
fi

attention_mask_size="$1"
input_dir="$(mktemp -d)"
trap 'rm -rf "${input_dir}"' EXIT

INPUT_DIR="${input_dir}" ATTENTION_MASK_SIZE="${attention_mask_size}" python3 - <<'PY'
import os
from pathlib import Path

import numpy as np

input_dir = Path(os.environ["INPUT_DIR"])
attention_mask_size = int(os.environ["ATTENTION_MASK_SIZE"])
np.save(input_dir / "input_ids.npy", np.array([[13]], dtype=np.int64))
np.save(input_dir / "attention_mask.npy", np.ones((1, attention_mask_size), dtype=np.int64))
np.save(input_dir / "position_ids.npy", np.array([[attention_mask_size - 1]], dtype=np.int64))
np.save(input_dir / "beam_idx.npy", np.array([0], dtype=np.int32))
PY

ZE_AFFINITY_MASK=0 /home/REPO/openvino/openvino/bin/intel64/Release/benchmark_app \
	-m /home/REPO/MODELS/Qwen3-06B/fp16/openvino_model.xml \
	-d GPU \
	-hint latency \
	-api sync \
	-nireq 1 \
	-niter 100 \
	-data_shape "input_ids[1,1],attention_mask[1,${attention_mask_size}],position_ids[1,1],beam_idx[1]" \
	-i \
	"input_ids:${input_dir}/input_ids.npy" \
	"attention_mask:${input_dir}/attention_mask.npy" \
	"position_ids:${input_dir}/position_ids.npy" \
	"beam_idx:${input_dir}/beam_idx.npy" \
	-pc