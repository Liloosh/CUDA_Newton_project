#pragma once

#include <cublas_v2.h>
#include <cusolverDn.h>

#include "DataInitializer.h"

class DataInitializerCUDA : public DataInitializer
{
public:
	bool is_cublas { false };
	cublasHandle_t cublasContextHandler { nullptr};
	cusolverDnHandle_t cusolverH{ nullptr };
	int* cusolver_pivot{ nullptr };
	int* cublas_info{ nullptr };
	double** cublas_ajacobian_d{ nullptr };
	double** cublas_afunc_values_d{ nullptr };
	double *workspace_d{ nullptr };
	int workspace_size{ 0 };

	double* intermediate_funcs_value_h{ nullptr },
		* delta_h{ nullptr },
		* funcs_value_h{ nullptr },

		* indexes_d{ nullptr },
		* points_d{ nullptr },
		* intermediate_funcs_value_d{ nullptr },
		* jacobian_d{ nullptr },
		* delta_d{ nullptr },
		* funcs_value_d{ nullptr };

	DataInitializerCUDA(int MATRIX_SIZE, int zeros_elements_per_row,
			int file_name, bool is_cublas,
			const Settings::SettingsData& s, int power);
	~DataInitializerCUDA();
};
