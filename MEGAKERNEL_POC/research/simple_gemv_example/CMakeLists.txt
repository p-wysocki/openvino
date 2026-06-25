cmake_minimum_required(VERSION 3.16)

set(APPLICATION_NAME "simple_gemv_example")

project(${APPLICATION_NAME} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(OPENVINO_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../.." CACHE PATH "Path to the OpenVINO source tree")
set(OPENCL_ICD_LOADER_HEADERS_DIR "${OPENVINO_SOURCE_DIR}/thirdparty/ocl/cl_headers" CACHE PATH "Path to OpenCL C headers")
set(OPENCL_ICD_LOADER_BUILD_SHARED_LIBS ON CACHE BOOL "Build OpenCL ICD loader as shared library")
set(ENABLE_OPENCL_LAYERS OFF CACHE BOOL "Disable optional OpenCL layer info tool")
set(BUILD_GMOCK OFF CACHE BOOL "Disable GoogleMock for this standalone test")
set(INSTALL_GTEST OFF CACHE BOOL "Disable GoogleTest install rules")

add_subdirectory("${OPENVINO_SOURCE_DIR}/thirdparty/ocl/icd_loader" "${CMAKE_CURRENT_BINARY_DIR}/opencl_icd_loader")
add_subdirectory("${OPENVINO_SOURCE_DIR}/thirdparty/gtest/gtest" "${CMAKE_CURRENT_BINARY_DIR}/gtest" EXCLUDE_FROM_ALL)

file(GLOB_RECURSE PRELOADING_TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_LIST_DIR}/src/*.cpp")
file(GLOB_RECURSE COMMON_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_LIST_DIR}/../common/*.cpp")

add_executable(${APPLICATION_NAME} ${PRELOADING_TEST_SOURCES} ${COMMON_SOURCES})
target_compile_definitions(${APPLICATION_NAME} PRIVATE
    CL_TARGET_OPENCL_VERSION=120
    OPENCL_KERNEL_SOURCE_PATH="${CMAKE_CURRENT_LIST_DIR}/src/ocl/gemv.cl")
target_link_libraries(${APPLICATION_NAME} PRIVATE OpenCL::OpenCL gtest_main)

enable_testing()
add_test(NAME ${APPLICATION_NAME} COMMAND ${APPLICATION_NAME})

if(UNIX AND NOT APPLE)
    set_target_properties(${APPLICATION_NAME} PROPERTIES BUILD_RPATH "$ORIGIN")
endif()