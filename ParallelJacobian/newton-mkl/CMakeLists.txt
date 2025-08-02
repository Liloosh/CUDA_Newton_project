cmake_minimum_required(VERSION 3.16)
project(sparse_cg LANGUAGES CXX)

set(CMAKE_CXX_COMPILER icpx)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED ENV{MKLROOT})
    message(FATAL_ERROR "MKLROOT is not set. Please set MKLROOT environment variable.")
endif()

add_executable(sparse_cg main.cpp)

target_include_directories(sparse_cg PRIVATE $ENV{MKLROOT}/include)

target_compile_options(sparse_cg PRIVATE
    -fsycl
    -fsycl-device-code-split=per_kernel
    -fno-sycl-early-optimizations
)

target_link_libraries(sparse_cg PRIVATE
    -L$ENV{MKLROOT}/lib/intel64
    -lmkl_sycl
    -lmkl_intel_lp64
    -lmkl_sequential
    -lmkl_core
    -lsycl
    -lOpenCL
    -lpthread
    -lm
    -ldl
)

target_link_options(sparse_cg PRIVATE
    -fsycl
    -fsycl-device-code-split=per_kernel
    -fno-sycl-early-optimizations
)
