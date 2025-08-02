#pragma once
#include "DataInitializerMKLdss.h"

#include "settings.h"
#include "system-info.h"

class NewtonSolverMKLdss {
private:
	DataInitializerMKLdss* data;
	const Settings::SettingsData& settings_;
	SystemInfo& sinfo_;

	void cpu_computeVec();
	double cpu_compute_derivative(int rowIndex, int colIndex);
	void cpu_compute_jacobian();
	void cpu_mkl_dss_find_delta();
public:

	NewtonSolverMKLdss(DataInitializerMKLdss*data,
			const Settings::SettingsData& settings,
			SystemInfo& sinfo);
	~NewtonSolverMKLdss();
	void cpu_newton_solve();
};
