#pragma once

#include <cstddef>

class GpuMonitor
{
public:
    GpuMonitor();

    // return memory in kB units
    size_t mem_initial_get() const;

    // return difference from initial free memory
    size_t mem_usage_get();
    size_t mem_usage_max_get() const;
    void mem_usage_max_reset();

private:
    size_t free_base_memory_{0};
    size_t max_mem_usage_{0};
};
