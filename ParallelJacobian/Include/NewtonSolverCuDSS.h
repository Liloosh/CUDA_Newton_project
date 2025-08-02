#pragma once
#include <DataInitializerCuDSS.h>

#include "settings.h"
#include "system-info.h"

class NewtonSolverCuDSS {
private:
	DataInitializerCuDSS* data;
	const Settings::SettingsData& settings_;
	SystemInfo& sinfo_;

public:

	NewtonSolverCuDSS(DataInitializerCuDSS*data,
			const Settings::SettingsData& settings,
			SystemInfo& sinfo);
	~NewtonSolverCuDSS();
	void solve(double* matrix_A_h, double* vector_b_d, double* vector_x_h, double* vector_x_d);
	void gpu_newton_solver_cudss();
};
