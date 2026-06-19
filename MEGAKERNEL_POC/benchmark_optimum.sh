#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}/python"

ZE_AFFINITY_MASK=0 python3 run_openvino_optimum.py "$@"