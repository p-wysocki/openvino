cmake_minimum_required(VERSION 3.16)

set(APPLICATION_NAME "preloading_gemv")

project(${APPLICATION_NAME} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(OPENVINO_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../.." CACHE PATH "Path to the OpenVINO source tree")
set(OPENCL_ICD_LOADER_HEADERS_DIR "${OPENVINO_SOURCE_DIR}/thirdparty/ocl/cl_headers" CACHE PATH "Path to OpenCL C headers")
set(OPENCL_ICD_LOADER_BUILD_SHARED_LIBS ON CACHE BOOL "Build OpenCL ICD loader as shared library")
set(ENABLE_OPENCL_LAYERS OFF CACHE BOOL "Disable optional OpenCL layer info tool")
set(DNNL_CPU_RUNTIME NONE CACHE STRING "Disable oneDNN CPU runtime for the GEMV reference" FORCE)
set(DNNL_GPU_RUNTIME OCL CACHE STRING "Use OpenCL oneDNN GPU runtime for the GEMV reference" FORCE)
set(DNNL_GPU_VENDOR INTEL CACHE STRING "Build oneDNN GPU runtime for Intel GPUs" FORCE)
set(DNNL_BUILD_EXAMPLES OFF CACHE BOOL "Disable oneDNN examples")
set(DNNL_BUILD_TESTS OFF CACHE BOOL "Disable oneDNN tests")
set(ONEDNN_BUILD_GRAPH OFF CACHE BOOL "Disable oneDNN graph component")
set(DNNL_ENABLE_PRIMITIVE "CONCAT;CONVOLUTION;DECONVOLUTION;GATED_MLP;INNER_PRODUCT;MATMUL;REORDER;POOLING;REDUCTION;SDPA;RNN" CACHE STRING "Build oneDNN GPU primitive set needed by the implementation registry" FORCE)
set(BUILD_GMOCK OFF CACHE BOOL "Disable GoogleMock for this standalone test")
set(INSTALL_GTEST OFF CACHE BOOL "Disable GoogleTest install rules")

add_subdirectory("${OPENVINO_SOURCE_DIR}/thirdparty/ocl/icd_loader" "${CMAKE_CURRENT_BINARY_DIR}/opencl_icd_loader")
add_subdirectory("${OPENVINO_SOURCE_DIR}/src/plugins/intel_gpu/thirdparty/onednn_gpu" "${CMAKE_CURRENT_BINARY_DIR}/onednn" EXCLUDE_FROM_ALL)
add_subdirectory("${OPENVINO_SOURCE_DIR}/thirdparty/gtest/gtest" "${CMAKE_CURRENT_BINARY_DIR}/gtest" EXCLUDE_FROM_ALL)

file(GLOB_RECURSE PRELOADING_TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_LIST_DIR}/src/*.cpp")
file(GLOB_RECURSE COMMON_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_LIST_DIR}/../common/*.cpp")

add_executable(${APPLICATION_NAME} ${PRELOADING_TEST_SOURCES} ${COMMON_SOURCES})
target_compile_definitions(${APPLICATION_NAME} PRIVATE
    CL_TARGET_OPENCL_VERSION=300
    OPENCL_KERNEL_SOURCE_PATH="${CMAKE_CURRENT_LIST_DIR}/src/ocl/")
target_link_libraries(${APPLICATION_NAME} PRIVATE OpenCL::OpenCL dnnl gtest_main)

target_include_directories(${APPLICATION_NAME} PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/src/ocl/")

if(UNIX AND NOT APPLE)
    set_target_properties(${APPLICATION_NAME} PROPERTIES BUILD_RPATH "$ORIGIN")
endif()