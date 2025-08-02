#include "system-info.h"

#ifdef __linux__
#include <sys/resource.h>
#endif

#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

#ifdef CFG_SOLVE_CUDA
#include <cuda_runtime.h>
#endif

#ifdef CFG_SOLVE_CUDA
#include "gpu-monitor.h"
#endif

struct MemUsage {
    long rss{0};
    long hwm{0};
};

#ifdef __linux__
extern char** environ;
#endif

namespace {

#ifdef CFG_SOLVE_CUDA
void checkCudaErrors(cudaError_t err)
{
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA Error: ") +
            cudaGetErrorName(err) + ", " + 	cudaGetErrorString(err));
    }
}
#endif

#ifdef __linux__
static double timeval_sec(const struct timeval& tv)
{
    return tv.tv_usec * 1e-6 + tv.tv_sec;
}

bool is_prefix(const std::string& line, const std::string& prefix)
{
    return strncmp(line.c_str(), prefix.c_str(), prefix.length()) == 0;
}

bool get_memory_usage(MemUsage& mem_usage)
{
    static const std::string vm_rss = "VmRSS:";
    static const std::string vm_hwm = "VmHWM:";

    std::ifstream status_file("/proc/self/status");

    if (!status_file.is_open()) {
        std::cerr << "Error opening /proc/self/status" << std::endl;
        return false;
    }

    std::string line;
    int cnt = 0;

    while (std::getline(status_file, line) && cnt < 2) {
        if (is_prefix(line, vm_rss)) {
            sscanf(line.c_str(), "%*s %ld", &mem_usage.rss);
            cnt++;
            continue;
        }
        if (is_prefix(line, vm_hwm)) {
            sscanf(line.c_str(), "%*s %ld", &mem_usage.hwm);
            cnt++;
            continue;
        }
    }

    return cnt == 2;
}

static void dump_env_mkl_omp(std::ostream& stream)
{
#define ENV_MKL "MKL_"
#define ENV_OMP "OMP_"

    for (char** env_var = environ; *env_var != nullptr; ++env_var) {
        if (strncmp(*env_var, ENV_MKL, sizeof(ENV_MKL) - 1) != 0 &&
                strncmp(*env_var, ENV_OMP, sizeof(ENV_OMP) - 1) != 0) {
            continue;
        }

        stream << *env_var << std::endl;
    }
}
#endif

}

SystemInfo::SystemInfo(int argc, char* argv[])
#ifdef CFG_SOLVE_CUDA
    : gpu_mon_{std::make_unique<GpuMonitor>()}
#endif
{
    for (int i = 0; i < argc; ++i) {
        if (i !=0)
            cmd_ += ' ';
        cmd_ += argv[i];
    }

    auto now = std::chrono::system_clock::now();
    auto ms_since_epoch =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                now.time_since_epoch());
    long long ms = ms_since_epoch.count() % 1000;
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm* gmt = std::gmtime(&t);

    std::stringstream ss;

    ss << std::put_time(gmt, "%Y%m%d-%H%M%S") << std::setfill('0')
       << std::setw(3) << ms;

    timestamp_ = ss.str();
}

SystemInfo::~SystemInfo()
{
}

std::string SystemInfo::getTimeStamp() const
{
    return timestamp_;
}

void SystemInfo::dump(std::ostream& stream) const
{
    stream << "Launch time: " << timestamp_ << std::endl;
    stream << "Command: " << cmd_ << std::endl;
#ifdef __linux__
    dump_env_mkl_omp(stream);
#endif
    stream << std::endl;
#ifdef CFG_SOLVE_CUDA
    dumpDeviceProps(stream);
#endif
}

#ifdef CFG_SOLVE_CUDA
void SystemInfo::dumpDeviceProps(std::ostream& stream) const
{
    int devId, devCount;
    cudaDeviceProp props;
    const int bsize = 512;
    char buf[bsize];

    checkCudaErrors(cudaGetDevice(&devId));
    checkCudaErrors(cudaGetDeviceCount(&devCount));
    checkCudaErrors(cudaGetDeviceProperties(&props, devId));

    stream << "Dev ID: " << devId << ", Dev count: " << devCount << '\n';

    std::snprintf(buf, sizeof(buf), "CUDA Compute %d.%d, Device: %s",
        props.major, props.minor, props.name);
    stream << buf << std::endl;

    std::snprintf(buf, sizeof(buf),
        "Memory: %zu bytes, Bus Width: %d bits, Clock Rate %d kHz",
        props.totalGlobalMem, props.memoryBusWidth, props.memoryClockRate);
    stream << buf << std::endl;

    stream << "Free memory: " << gpu_mon_->mem_initial_get() << " kB"
           << std::endl;

    stream << "MultiProcessorCount: " << props.multiProcessorCount << std::endl;
    stream << "GPU Cock Rate: " << props.clockRate << " kHz" << std::endl;
}
#endif

void SystemInfo::dump_resource_usage(std::ostream& stream) const
{
#ifdef __linux__
    stream << std::endl;

    struct rusage usage;

    getrusage(RUSAGE_SELF, &usage);

    stream << "User CPU time used: " << timeval_sec(usage.ru_utime)
           << ", Sys CPU time used: " << timeval_sec(usage.ru_stime)
           << std::endl;

    MemUsage mem;
    if (!get_memory_usage(mem)) {
        stream << "Failed to read mem usage" << std::endl;
    } else {
        stream << "RSS: " << mem.rss << " kB, Max RSS: " << mem.hwm << " kB" << std::endl;
    }
#endif
#ifdef CFG_SOLVE_CUDA
    stream << "GPU Mem Usage: " << gpu_mon_->mem_usage_get()
           << " kB, GPU Max Mem Usage: " << gpu_mon_->mem_usage_max_get()
           << " kB" << std::endl;
#endif

    stream << std::endl;
}

size_t SystemInfo::gpu_mem_usage_get()
{
#ifdef CFG_SOLVE_CUDA
    return gpu_mon_->mem_usage_get();
#else
    return 0;
#endif
}

size_t SystemInfo::gpu_mem_usage_max_get()
{
#ifdef CFG_SOLVE_CUDA
    return gpu_mon_->mem_usage_max_get();
#else
    return 0;
#endif
}

size_t SystemInfo::mem_rss_usage_get()
{
#ifdef __linux__
    MemUsage mem;
    if (!get_memory_usage(mem))
        return 0;

    return mem.rss;
#else
    return 0;
#endif
}

size_t SystemInfo::mem_rss_max_usage_get()
{
#ifdef __linux__
    MemUsage mem;
    if (!get_memory_usage(mem))
        return 0;

    return mem.hwm;
#else
    return 0;
#endif
}
