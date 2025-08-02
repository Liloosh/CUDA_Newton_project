#include "gpu-monitor.h"

#include <cuda_runtime.h>

namespace
{

static size_t gpu_free_mem_get()
{
    size_t free;
    size_t total;
    cudaMemGetInfo(&free, &total);

    // return kB
    return free / 1024;
}

}

GpuMonitor::GpuMonitor()
{
    free_base_memory_ = gpu_free_mem_get();
}

size_t GpuMonitor::mem_initial_get() const
{
    return free_base_memory_;
}

size_t GpuMonitor::mem_usage_get()
{
    size_t free = gpu_free_mem_get();

    size_t usage = free < free_base_memory_ ? free_base_memory_ - free : 0;

    if (usage > max_mem_usage_)
        max_mem_usage_ = usage;

    return usage;
}

size_t GpuMonitor::mem_usage_max_get() const
{
    return max_mem_usage_;
}

void GpuMonitor::mem_usage_max_reset()
{
    max_mem_usage_ = gpu_free_mem_get();
}
