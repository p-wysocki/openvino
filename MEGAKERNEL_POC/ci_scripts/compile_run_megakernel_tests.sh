#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
build_dir="$(cd "${repo_root}/.." && pwd)/build"
venv_dir=""

cleanup() {
    local exit_code=$?

    if [[ -n "${venv_dir}" ]] && ! rm -rf -- "${venv_dir}"; then
        ((exit_code == 0)) && exit_code=1
    fi

    exit "${exit_code}"
}

main() {
    venv_dir="$(mktemp -d)"
    trap cleanup EXIT

    #git submodule update --init --recursive

    python3 -m venv --system-site-packages "${venv_dir}"
    source "${venv_dir}/bin/activate"

    chmod +x install_build_dependencies.sh
    ./install_build_dependencies.sh

    python -m pip install --upgrade optimum-intel
    python "${repo_root}/MEGAKERNEL_POC/python/convert_to_openvino_ir.py" \
        --output-dir "${repo_root}/MEGAKERNEL_POC/python/qwen3-0.6b-openvino-ir"

    cmake -S "${repo_root}" -B "${build_dir}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DENABLE_DEBUG_CAPS=ON \
        -DENABLE_CPU_DEBUG_CAPS=OFF \
        -DENABLE_GPU_DEBUG_CAPS=ON \
        -DENABLE_TESTS=ON \
        -DENABLE_INTEL_CPU=ON \
        -DENABLE_INTEL_GPU=ON \
        -DENABLE_OV_ONNX_FRONTEND=OFF \
        -DENABLE_PYTHON=ON \
        -DENABLE_OV_PADDLE_FRONTEND=OFF \
        -DENABLE_OV_PYTORCH_FRONTEND=ON \
        -DENABLE_OV_JAX_FRONTEND=OFF \
        -DENABLE_OV_TF_FRONTEND=OFF \
        -DENABLE_OV_TF_LITE_FRONTEND=OFF \
        -DENABLE_JS=OFF \
        -DENABLE_WHEEL=ON \
        -DENABLE_TEMPLATE_REGISTRATION=OFF
    cmake --build "${build_dir}" --parallel 16

    python -m pip install "${build_dir}"/wheels/*.whl --force-reinstall
    bash "${repo_root}/MEGAKERNEL_POC/benchmark_app.sh"
    python "${repo_root}/MEGAKERNEL_POC/python/e2e_performance_measurement.py" \
        --frameworks decode_only optimum \
        --torch-threads 20
}

main "$@"

