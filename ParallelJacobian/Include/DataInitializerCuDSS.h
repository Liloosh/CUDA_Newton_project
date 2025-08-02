#pragma once
#include "DataInitializer.h"
#include "cudss.h"

class DataInitializerCuDSS : public DataInitializer
{
public:
	bool analyzed = false;
	double* csr_values_h{ nullptr };
	int* csr_rows_h{ nullptr },
		* csr_cols_h{ nullptr };

	int* csr_rows_d{ nullptr },
		* csr_cols_d{ nullptr };
	double* csr_values_d{ nullptr };

	int non_zero_count{ 0 };



	double* points_d{ nullptr },
		* jacobian_d{ nullptr },
		* delta_d{ nullptr },
		* funcs_value_d{ nullptr },
		* funcs_value_h{ nullptr },
		* delta_h{ nullptr };

	cudssHandle_t handler;
	cudssConfig_t solverConfig;
	cudssData_t solverData;

	cudssMatrix_t x, b;

	cudssMatrix_t A;
	cudssMatrixType_t mtype = CUDSS_MTYPE_GENERAL;
	cudssMatrixViewType_t mvtype = CUDSS_MVIEW_FULL;
	cudssIndexBase_t base = CUDSS_BASE_ZERO;

	DataInitializerCuDSS(int MATRIX_SIZE, int zeros_elements_per_row,
			int file_name, const Settings::SettingsData& s, int power);
	~DataInitializerCuDSS();
};
