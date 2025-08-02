#pragma once
#include "DataInitializerMKLlapack.h"

#include "settings.h"
#include "system-info.h"

struct NewtonSolverMKLlapack
{
private:
    DataInitializerMKLlapack * data;
    const Settings::SettingsData& settings_;
    SystemInfo& sinfo_;

    void cpu_computeVec();
	double cpu_compute_derivative(int rowIndex, int colIndex);
    void cpu_compute_jacobian();
    void cpu_find_delta();

public:
    NewtonSolverMKLlapack(DataInitializerMKLlapack* data,
                          const Settings::SettingsData& settings,
                          SystemInfo& sinfo);
    ~NewtonSolverMKLlapack();
    void cpu_newton_solve();
};
