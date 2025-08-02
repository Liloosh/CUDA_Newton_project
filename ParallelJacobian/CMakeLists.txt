cmake_minimum_required(VERSION 3.24)

option(BUILD_CUDA "Build the project with CUDA" ON)
option(BUILD_MKL "Build the project with MKL" ON)

if(BUILD_CUDA)
    project(ParallelJacobian LANGUAGES CXX CUDA)
else()
    project(ParallelJacobian LANGUAGES CXX)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(UNIX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
endif()

include_directories(
    "${CMAKE_SOURCE_DIR}/Include"
)

if(BUILD_CUDA)
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
        set(cudss_DIR "${CUDSS_PATH}/lib/12/cmake/cudss")
        message(STATUS "Detected cuDSS path: ${CUDSS_PATH}")
        message(STATUS "Detected cuDSS version: ${CUDSS_VERSION}")
    endif()

    find_package(cudss)
    message (STATUS "cuDSS version: ${cudss_VERSION}")

    if(UNIX)
        set(CUDA_PATH "/usr/local/cuda")
        set(CUDSS_PATH "/usr/local/cudss")
    endif()

    include_directories(
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

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCFG_SOLVE_CUDA")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DCFG_SOLVE_CUDA")
endif() # BUILD_CUDA

if(BUILD_MKL)
    if(NOT MKL_INTERFACE_FULL)
        set(MKL_INTERFACE_FULL intel_lp64)
        message(STATUS "Set MKL intel_lp64 by default")
    endif()

    find_package(MKL REQUIRED)

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCFG_SOLVE_MKL")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DCFG_SOLVE_MKL")

    if(MKL_INTERFACE_FULL STREQUAL intel_ilp64)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMKL_ILP64")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DMKL_ILP64")
    endif()
endif()

set(COMMON_SOURCE_FILES
    "${CMAKE_SOURCE_DIR}/Source/DataInitializer.cpp"
    "${CMAKE_SOURCE_DIR}/Source/DataInitializerCPU.cpp"
    "${CMAKE_SOURCE_DIR}/Source/EditionalTools.cpp"
    "${CMAKE_SOURCE_DIR}/Source/Equation.cpp"
    "${CMAKE_SOURCE_DIR}/Source/FileOperations.cpp"
    "${CMAKE_SOURCE_DIR}/Source/NewtonSolverCPU.cpp"
    "${CMAKE_SOURCE_DIR}/Source/report.cpp"
    "${CMAKE_SOURCE_DIR}/Source/settings.cpp"
    "${CMAKE_SOURCE_DIR}/Source/system-build-info.cpp"
    "${CMAKE_SOURCE_DIR}/Source/system-info.cpp"
)

if(BUILD_CUDA)
    list(APPEND COMMON_SOURCE_FILES
        "${CMAKE_SOURCE_DIR}/Source/DataInitializerCuDSS.cpp"
        "${CMAKE_SOURCE_DIR}/Source/DataInitializerCUDA.cpp"
        "${CMAKE_SOURCE_DIR}/Source/gpu-monitor.cpp"
        "${CMAKE_SOURCE_DIR}/Source/NewtonSolverCUDA.cu"
        "${CMAKE_SOURCE_DIR}/Source/NewtonSolverCuDSS.cu"
        "${CMAKE_SOURCE_DIR}/Source/NewtonSolverGPUFunctions.cu"
    )
endif()

if(BUILD_MKL)
    list(APPEND COMMON_SOURCE_FILES
        "${CMAKE_SOURCE_DIR}/Source/DataInitializerMKLdss.cpp"
        "${CMAKE_SOURCE_DIR}/Source/DataInitializerMKLlapack.cpp"
        "${CMAKE_SOURCE_DIR}/Source/NewtonSolverMKLdss.cpp"
        "${CMAKE_SOURCE_DIR}/Source/NewtonSolverMKLlapack.cpp"
    )
endif()

add_compile_options(
    "$<$<CONFIG:Release,MinSizeRel>:-DBUILD_CONFIG_RELEASE>"
    "$<$<CONFIG:Debug>:-DBUILD_CONFIG_DEBUG>"
)

add_library(my_common_lib STATIC ${COMMON_SOURCE_FILES})

add_executable(ParallelJacobian_DimensionStudy "${CMAKE_SOURCE_DIR}/Source/main_dim.cpp")
target_link_libraries(ParallelJacobian_DimensionStudy PRIVATE my_common_lib)

add_executable(ParallelJacobian_SparsityStudy "${CMAKE_SOURCE_DIR}/Source/main_spar.cpp")
target_link_libraries(ParallelJacobian_SparsityStudy PRIVATE my_common_lib)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
    message(STATUS "Setting default build type: ${CMAKE_BUILD_TYPE}")
endif()

if(BUILD_CUDA)
    # CUDA_ARCHITECTURES: list of architectures, for example "61;75" or
    # one of the following special values: native, all, all-major
    if(NOT CUDA_ARCHITECTURES)
        set(CUDA_ARCHITECTURES native)
        message(STATUS "Setting default CUDA Architectures: ${CUDA_ARCHITECTURES}")
    endif()

    target_link_libraries(ParallelJacobian_DimensionStudy PRIVATE cudss cublas cusolver)
    target_link_libraries(ParallelJacobian_SparsityStudy PRIVATE cudss cublas cusolver)

    set_target_properties(ParallelJacobian_DimensionStudy ParallelJacobian_SparsityStudy my_common_lib PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON # Generally good practice for large CUDA projects
        CUDA_ARCHITECTURES "${CUDA_ARCHITECTURES}"
        )

    if(WIN32)
        # build main files as CUDA for proper linkage by MSVC
        set_source_files_properties("${CMAKE_SOURCE_DIR}/Source/main_dim.cpp"
            PROPERTIES LANGUAGE CUDA)
        set_source_files_properties("${CMAKE_SOURCE_DIR}/Source/main_spar.cpp"
            PROPERTIES LANGUAGE CUDA)
    endif()
endif()

if(BUILD_MKL)
    target_link_libraries(ParallelJacobian_DimensionStudy PRIVATE MKL::MKL)
    target_link_libraries(ParallelJacobian_SparsityStudy PRIVATE MKL::MKL)
    target_include_directories(ParallelJacobian_DimensionStudy PRIVATE $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
    target_include_directories(ParallelJacobian_SparsityStudy PRIVATE $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
endif()

# Add a custom target to generate version.h
add_custom_target(gen-version-h
    ALL
    COMMAND ${CMAKE_COMMAND}
        -DSOURCE_DIR=${CMAKE_SOURCE_DIR}
        -DBINARY_DIR=${CMAKE_BINARY_DIR}
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DCUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}
        -DCMAKE_CUDA_COMPILER_ID=${CMAKE_CUDA_COMPILER_ID}
        -DCMAKE_CXX_COMPILER_ID=${CMAKE_CXX_COMPILER_ID}
        -DCMAKE_CUDA_COMPILER_VERSION=${CMAKE_CUDA_COMPILER_VERSION}
        -DCMAKE_CXX_COMPILER_VERSION=${CMAKE_CXX_COMPILER_VERSION}
        -DCMAKE_CUDSS_VERSION=${cudss_VERSION}
        -DMKL_INTERFACE_FULL=${MKL_INTERFACE_FULL}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/gen-version.cmake
)
add_dependencies(my_common_lib gen-version-h)
include_directories(${CMAKE_CURRENT_BINARY_DIR})

if(WIN32 AND BUILD_CUDA)
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
