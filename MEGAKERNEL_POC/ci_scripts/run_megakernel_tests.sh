#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
build_dir="${repo_root}/build"
venv_dir="${build_dir}/venv"

main() {
    python3 -m venv --system-site-packages "${venv_dir}"
    source "${venv_dir}/bin/activate"

    python -m pip install "${build_dir}"/wheels/*.whl --force-reinstall
    bash "${repo_root}/MEGAKERNEL_POC/benchmark_app.sh"
    python "${repo_root}/MEGAKERNEL_POC/python/e2e_performance_measurement.py" \
        --frameworks decode_only optimum \
        --torch-threads 20
}

main "$@"