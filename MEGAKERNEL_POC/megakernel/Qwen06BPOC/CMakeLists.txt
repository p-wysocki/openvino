cmake_minimum_required(VERSION 3.16)

project(Qwen06BPOC LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(TASK_SYSTEM_OPENCL_ROOT
    "${CMAKE_CURRENT_LIST_DIR}/../../research/preloading_gemv/src/ocl"
    CACHE PATH "Path to the task-system OpenCL sources")

if(NOT TARGET OpenCL::OpenCL)
    find_package(OpenCL REQUIRED)
endif()

add_library(Qwen06BPOC STATIC
    megakernelImpl.cpp
    impl/qwen06BPOCRuntime.cpp
    "${TASK_SYSTEM_OPENCL_ROOT}/taskSystem/host/taskManagerHost.cpp")
add_library(Qwen06BPOC::Qwen06BPOC ALIAS Qwen06BPOC)

target_compile_definitions(Qwen06BPOC PRIVATE
    CL_TARGET_OPENCL_VERSION=300
    TASK_SYSTEM_OPENCL_ROOT="${TASK_SYSTEM_OPENCL_ROOT}")

target_include_directories(Qwen06BPOC
    PUBLIC "${CMAKE_CURRENT_LIST_DIR}/.."
    PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}/impl"
        "${TASK_SYSTEM_OPENCL_ROOT}")

target_link_libraries(Qwen06BPOC PUBLIC OpenCL::OpenCL)

set_target_properties(Qwen06BPOC PROPERTIES POSITION_INDEPENDENT_CODE ON)