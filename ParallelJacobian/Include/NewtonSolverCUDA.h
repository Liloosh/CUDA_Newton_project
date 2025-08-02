#pragma once
#include "DataInitializerCUDA.h"

#include "settings.h"
#include "system-info.h"

struct NewtonSolverCUDA {
private:
	DataInitializerCUDA* data;
	const Settings::SettingsData& settings_;
	SystemInfo& sinfo_;

public:
	NewtonSolverCUDA(DataInitializerCUDA* data,
			const Settings::SettingsData& settings, SystemInfo& sinfo);
	~NewtonSolverCUDA();

	void gpu_cublas_solve(DataInitializerCUDA* data);
	void gpu_cusolver_solve(DataInitializerCUDA* data);
	void gpu_newton_solve();
};
