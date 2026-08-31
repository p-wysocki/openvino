
set -e

venv_dir="$(mktemp -d)"
trap 'rm -rf "${venv_dir}"' EXIT
python3 -m venv --system-site-packages "${venv_dir}"
source "${venv_dir}/bin/activate"

pip install -U optimum-intel

cd MEGAKERNEL_POC/python/
python convert_to_openvino_ir.py
cd ../..

cmake -S . -B ../build/ \
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
    #-DMEGAKERNEL_IMPLEMENTATION=Qwen06BPOC_prefill_separate_kernels
cmake --build ../build/ --parallel 16



python -m pip install ../build/wheels/*.whl --force-reinstall

bash MEGAKERNEL_POC/benchmark_app.sh

python MEGAKERNEL_POC/python/e2e_performance_measurement.py --frameworks decode_only optimum --torch-threads 20

