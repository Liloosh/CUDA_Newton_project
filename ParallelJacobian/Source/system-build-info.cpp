#include "system-build-info.h"

#include "version.h"

const std::string SystemBuildInfo::gitVersion = GIT_VERSION_STRING;
const std::string SystemBuildInfo::compilerCudaId = BUILD_COMPILER_CUDA_ID;
const std::string SystemBuildInfo::compilerCudaVer = BUILD_COMPILER_CUDA_VER;
const std::string SystemBuildInfo::compilerCxxId = BUILD_COMPILER_CXX_ID;
const std::string SystemBuildInfo::compilerCxxVer = BUILD_COMPILER_CXX_VER;
const std::string SystemBuildInfo::cuDssVer = BUILD_CUDSS_VERSION;
const std::string SystemBuildInfo::cudaArchs = BUILD_CUDA_ARCHITECTURES;
const std::string SystemBuildInfo::cmakeBuildType = BUILD_CMAKE_TYPE;
const std::string SystemBuildInfo::mklInterface = BUILD_MKL_INTERFACE_FULL;

#if defined(BUILD_CONFIG_DEBUG)
const std::string SystemBuildInfo::buildConfig = "Debug";
#elif defined(BUILD_CONFIG_RELEASE)
const std::string SystemBuildInfo::buildConfig = "Release";
#else
    #error "Define BUILD_CONFIG_RELEASE or BUILD_CONFIG_DEBUG"
#endif

void SystemBuildInfo::dump(std::ostream& s)
{
    s << "Version: " << gitVersion << std::endl;
    s << "CUDA Compiler " << compilerCudaId << " " << compilerCudaVer << std::endl;
    s << "CXX Compiler " << compilerCxxId << " " << compilerCxxVer << std::endl;
    s << "cuDSS Version " << cuDssVer << std::endl;
    s << "CUDA Architectures: " << cudaArchs << std::endl;
    s << "Build config: " << buildConfig << std::endl;

    if (buildConfig != cmakeBuildType) {
        s << "Cmake build type: " << cmakeBuildType << std::endl;
    }

#ifndef CFG_SOLVE_CUDA
    s << "CUDA build disabled" << std::endl;
#endif
#ifndef CFG_SOLVE_MKL
    s << "MKL build disabled" << std::endl;
#else
    s << "MKL Interface: " << mklInterface << std::endl;
#endif
}
