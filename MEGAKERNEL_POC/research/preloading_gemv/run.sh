#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
build_dir="${BUILD_DIR:-${script_dir}/build}"
build_type="${CMAKE_BUILD_TYPE:-Release}"

cmake -S "${script_dir}" -B "${build_dir}" -DCMAKE_BUILD_TYPE="${build_type}"
cmake --build "${build_dir}" -j"$(nproc)"

ZE_AFFINITY_MASK=0 "${build_dir}/preloading_gemv" "$@"