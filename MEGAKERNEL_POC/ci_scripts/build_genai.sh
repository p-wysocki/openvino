#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
build_dir="${repo_root}/build"
venv_dir="${build_dir}/venv"
genai_source_dir="${build_dir}/openvino.genai"
genai_build_dir="${build_dir}/openvino.genai-build"
genai_repository="${GENAI_REPOSITORY:-https://github.com/openvinotoolkit/openvino.genai.git}"
genai_ref="${GENAI_REF:-2026.3.0.0}"

main() {
    source "${venv_dir}/bin/activate"

    rm -rf -- "${genai_source_dir}" "${genai_build_dir}"
    git clone --branch "${genai_ref}" --depth 1 --recurse-submodules --shallow-submodules \
        "${genai_repository}" "${genai_source_dir}"

    python -m pip install --upgrade -r "${genai_source_dir}/requirements-build.txt"
    python -m pip install "${build_dir}"/wheels/*.whl --force-reinstall

    cmake -S "${genai_source_dir}" -B "${genai_build_dir}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DOpenVINO_DIR="${build_dir}" \
        -DENABLE_TESTS=OFF \
        -DENABLE_SAMPLES=OFF \
        -DENABLE_TOOLS=OFF
    cmake --build "${genai_build_dir}" --parallel 16
}

main "$@"