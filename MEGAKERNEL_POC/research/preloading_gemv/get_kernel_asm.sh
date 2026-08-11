#!/usr/bin/env bash

set -euo pipefail

usage() {
  echo "Usage: $0 [--dump-assembly <path>] [--source-file <path>] [--gtest-filter <filter>] <kernel-name>" >&2
}

assembly_dump_path=
source_file=
gtest_filter=ChainTaskSystemGemvTest.ThreeGemvChain
kernel_name=
while [[ $# -gt 0 ]]; do
  case $1 in
    --dump-assembly)
      shift
      if [[ $# -eq 0 || -z $1 ]]; then
        usage
        exit 2
      fi
      assembly_dump_path=$1
      ;;
    --dump-assembly=*)
      assembly_dump_path=${1#*=}
      if [[ -z ${assembly_dump_path} ]]; then
        usage
        exit 2
      fi
      ;;
    --source-file)
      shift
      if [[ $# -eq 0 || -z $1 ]]; then
        usage
        exit 2
      fi
      source_file=$1
      ;;
    --source-file=*)
      source_file=${1#*=}
      if [[ -z ${source_file} ]]; then
        usage
        exit 2
      fi
      ;;
    --gtest-filter)
      shift
      if [[ $# -eq 0 || -z $1 ]]; then
        usage
        exit 2
      fi
      gtest_filter=$1
      ;;
    --gtest-filter=*)
      gtest_filter=${1#*=}
      if [[ -z ${gtest_filter} ]]; then
        usage
        exit 2
      fi
      ;;
    -*)
      usage
      exit 2
      ;;
    *)
      if [[ -n ${kernel_name} ]]; then
        usage
        exit 2
      fi
      kernel_name=$1
      ;;
  esac
  shift
done

if [[ -z ${kernel_name} ]]; then
  usage
  exit 2
fi
if [[ -n ${source_file} ]]; then
  if [[ ! -r ${source_file} ]]; then
    echo "Cannot read OpenCL source file: ${source_file}" >&2
    exit 1
  fi
  if [[ ${source_file} =~ [[:space:]] ]]; then
    echo "OpenCL source file path cannot contain whitespace: ${source_file}" >&2
    exit 2
  fi
  source_file=$(realpath -- "${source_file}")
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
build_dir="${BUILD_DIR:-${script_dir}/build}"
test_binary="${build_dir}/preloading_gemv"

if [[ ! -x "${test_binary}" ]]; then
  echo "Missing test binary: ${test_binary}" >&2
  echo "Build it first with: cmake --build ${build_dir} --target preloading_gemv" >&2
  exit 1
fi

work_dir=$(mktemp -d "${TMPDIR:-/tmp}/chain-kernel-resources.XXXXXX")
trap 'rm -rf -- "${work_dir}"' EXIT

cat >"${work_dir}/inject_visa_options.c" <<'EOF'
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int cl_int;
typedef unsigned int cl_uint;
typedef struct _cl_program* cl_program;
typedef struct _cl_device_id* cl_device_id;
typedef void (*build_callback_t)(cl_program, void*);
typedef cl_int (*cl_build_program_t)(cl_program, cl_uint, const cl_device_id*,
                                     const char*, build_callback_t, void*);

cl_int clBuildProgram(cl_program program, cl_uint num_devices,
                      const cl_device_id* device_list, const char* options,
                      build_callback_t callback, void* user_data) {
  static cl_build_program_t real_cl_build_program;
  if (real_cl_build_program == NULL) {
    real_cl_build_program =
        (cl_build_program_t)dlsym(RTLD_NEXT, "clBuildProgram");
    if (real_cl_build_program == NULL) {
      fprintf(stderr, "Failed to resolve clBuildProgram: %s\n", dlerror());
      exit(EXIT_FAILURE);
    }
  }

  if (options == NULL) {
    return real_cl_build_program(program, num_devices, device_list, options,
                                 callback, user_data);
  }

  const char* visa_options = strstr(options, "VISAOptions=");
  const char* closing_quote =
      visa_options == NULL ? NULL : strchr(visa_options, '\'');
  const char* fallback = " -igc_opts 'VISAOptions=-printregusage'";
  const char* addition = " -printregusage";
    const char* source_file = getenv("OCL_SOURCE_FILE");
    const char* debug_prefix = " -g -s ";
    const int annotate_source = source_file != NULL && source_file[0] != '\0';
  const size_t options_size = strlen(options);
    const size_t visa_addition_size =
      closing_quote == NULL ? strlen(fallback) : strlen(addition);
    const size_t debug_addition_size =
      annotate_source ? strlen(debug_prefix) + strlen(source_file) : 0;
    char* modified_options =
      malloc(options_size + visa_addition_size + debug_addition_size + 1);
  if (modified_options == NULL) {
    fprintf(stderr, "Failed to allocate modified OpenCL build options\n");
    exit(EXIT_FAILURE);
  }

  if (closing_quote == NULL) {
    memcpy(modified_options, options, options_size);
    memcpy(modified_options + options_size, fallback, visa_addition_size + 1);
  } else {
    const size_t prefix_size = (size_t)(closing_quote - options);
    memcpy(modified_options, options, prefix_size);
    memcpy(modified_options + prefix_size, addition, visa_addition_size);
    memcpy(modified_options + prefix_size + visa_addition_size, closing_quote,
           options_size - prefix_size + 1);
  }

  if (annotate_source) {
    const size_t modified_size = options_size + visa_addition_size;
    memcpy(modified_options + modified_size, debug_prefix,
           strlen(debug_prefix));
    memcpy(modified_options + modified_size + strlen(debug_prefix), source_file,
           strlen(source_file) + 1);
  }

  const cl_int status = real_cl_build_program(
      program, num_devices, device_list, modified_options, callback, user_data);
  free(modified_options);
  return status;
}
EOF

cc -shared -fPIC -O2 "${work_dir}/inject_visa_options.c" \
  -o "${work_dir}/inject_visa_options.so" -ldl

set +e
IGC_ShaderDumpEnable=1 \
IGC_DumpToCustomDir="${work_dir}" \
NEO_CACHE_PERSISTENT=0 \
OCL_SOURCE_FILE="${source_file}" \
LD_PRELOAD="${work_dir}/inject_visa_options.so${LD_PRELOAD:+:${LD_PRELOAD}}" \
  "${test_binary}" \
  --gtest_filter="${gtest_filter}" \
  >"${work_dir}/test.log" 2>&1
test_status=$?
set -e

mapfile -t assembly_files < <(
  grep -lFx -- "//.kernel ${kernel_name}" "${work_dir}"/*.asm \
    2>/dev/null | sort
)
if [[ ${#assembly_files[@]} -eq 0 ]]; then
  cat "${work_dir}/test.log" >&2
  echo "IGC did not produce assembly for ${kernel_name}" >&2
  exit 1
fi

assembly_file=${assembly_files[0]}
for candidate in "${assembly_files[@]}"; do
  if [[ ${candidate} == *_1_simd*_entry_*.asm ]]; then
    assembly_file=${candidate}
  fi
done

if [[ -n ${assembly_dump_path} ]]; then
  cp -- "${assembly_file}" "${assembly_dump_path}"
  echo "Assembly saved to ${assembly_dump_path}"
fi

grf_mode=$(sed -n 's|^//\.thread_config numGRF=\([0-9][0-9]*\).*|\1|p' \
  "${assembly_file}")
grf_usage=$(sed -n 's|^//\.GRF count \([0-9][0-9]*\).*|\1|p' \
  "${assembly_file}")
private_bytes=$(sed -n 's|^//\.private memory size \([0-9][0-9]*\).*|\1|p' \
  "${assembly_file}")
spill_bytes=$(sed -n 's|^//\.spill size \([0-9][0-9]*\).*|\1|p' \
  "${assembly_file}")
spill_references=$(sed -n \
  's|^//\.spill GRF est\. ref count \([0-9][0-9]*\).*|\1|p' \
  "${assembly_file}")

echo "${kernel_name} resources:"
echo "  GRFs used / allocation mode: ${grf_usage:-unavailable} / ${grf_mode:-unavailable}"
echo "  Private memory per hardware thread: ${private_bytes:-0} bytes"
echo "  Spill storage per hardware thread: ${spill_bytes:-0} bytes"
echo "  Estimated spill references: ${spill_references:-0}"

if [[ ${test_status} -ne 0 ]]; then
  echo "  Note: benchmark test exited with status ${test_status}; compiler data is still valid." >&2
fi