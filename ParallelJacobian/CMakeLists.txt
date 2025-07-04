cmake_minimum_required(VERSION 3.24)
project(ParallelJacobian LANGUAGES CXX CUDA)

if(WIN32)
    file(GLOB CUDA_DIRS "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*")
    list(SORT CUDA_DIRS)
    list(GET CUDA_DIRS 0 CUDA_PATH)
    string(REGEX MATCH "v([0-9]+\\.[0-9]+)" _cuda_ver_match "${CUDA_PATH}")
    string(REPLACE "v" "" CUDA_VERSION "${_cuda_ver_match}")
    message(STATUS "Detected CUDA path: ${CUDA_PATH}")
    message(STATUS "Detected CUDA version: ${CUDA_VERSION}")

    file(GLOB CUDSS_DIRS "C:/Program Files/NVIDIA cuDSS/v*")
    list(SORT CUDSS_DIRS)
    list(GET CUDSS_DIRS 0 CUDSS_PATH)
    string(REGEX MATCH "v([0-9]+\\.[0-9]+)" _cudss_ver_match "${CUDSS_PATH}")
    string(REPLACE "v" "" CUDSS_VERSION "${_cudss_ver_match}")
    message(STATUS "Detected cuDSS path: ${CUDSS_PATH}")
    message(STATUS "Detected cuDSS version: ${CUDSS_VERSION}")
endif()

if(UNIX)
    set(CUDA_PATH "/usr/local/cuda")
    set(CUDSS_PATH "/usr/local/cudss")

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
endif()

include_directories(
    "${CMAKE_SOURCE_DIR}/Include"
    "${CUDSS_PATH}/include"
    "${CUDA_PATH}/include"
)

if(WIN32)
    link_directories(
        "${CUDSS_PATH}/lib/12"
        "${CUDA_PATH}/lib/x64"
    )
endif()

if(UNIX)
    link_directories(
        "${CUDSS_PATH}/lib/12"
        "${CUDA_PATH}/lib64"
    )
endif()


file(GLOB COMMON_SOURCE_FILES
    "${CMAKE_SOURCE_DIR}/Source/*.cpp"
    "${CMAKE_SOURCE_DIR}/Source/*.cu"
)
list(REMOVE_ITEM COMMON_SOURCE_FILES
     "${CMAKE_SOURCE_DIR}/Source/kernel.cu"
     "${CMAKE_SOURCE_DIR}/Source/kernel2.cu"
)

add_library(my_common_lib STATIC ${COMMON_SOURCE_FILES})

add_executable(ParallelJacobian_DimensionStudy "${CMAKE_SOURCE_DIR}/Source/kernel.cu")
target_link_libraries(ParallelJacobian_DimensionStudy PRIVATE my_common_lib cudss cublas)

add_executable(ParallelJacobian_SparsityStudy "${CMAKE_SOURCE_DIR}/Source/kernel2.cu")
target_link_libraries(ParallelJacobian_SparsityStudy PRIVATE my_common_lib cudss cublas)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
    message(STATUS "Setting default build type: ${CMAKE_BUILD_TYPE}")
endif()

# CUDA_ARCHITECTURES: list of architectures, for example "61;75" or
# one of the following special values: native, all, all-major
if(NOT CUDA_ARCHITECTURES)
    set(CUDA_ARCHITECTURES native)
    message(STATUS "Setting default CUDA Architectures: ${CUDA_ARCHITECTURES}")
endif()

set_target_properties(ParallelJacobian_DimensionStudy ParallelJacobian_SparsityStudy my_common_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON # Generally good practice for large CUDA projects
    CUDA_ARCHITECTURES "${CUDA_ARCHITECTURES}"
)

if(WIN32)
    # This flag can sometimes resolve linking issues with default libraries in MSVC
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:LIBCMT")

    add_custom_command(TARGET ParallelJacobian_DimensionStudy POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CUDSS_PATH}/bin/12/cudss64_0.dll"
        $<TARGET_FILE_DIR:ParallelJacobian_DimensionStudy>
    )

    add_custom_command(TARGET ParallelJacobian_SparsityStudy POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CUDSS_PATH}/bin/12/cudss64_0.dll"
        $<TARGET_FILE_DIR:ParallelJacobian_SparsityStudy>
    )
endif()
