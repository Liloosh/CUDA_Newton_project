#pragma once
#include "DataInitializer.h"

class DataInitializerCUDA : public DataInitializer
{
public:
	cublasHandle_t cublasContextHandler;
	int* cublas_pivot{ nullptr }, * cublas_info{ nullptr };
	double** cublas_ajacobian_d{ nullptr }, ** cublas_ainverse_jacobian_d{ nullptr };

	double* intermediate_funcs_value_h{ nullptr },
		* delta_h{ nullptr },
		* funcs_value_h{ nullptr },

		* indexes_d{ nullptr },
		* points_d{ nullptr },
		* intermediate_funcs_value_d{ nullptr },
		* vector_b_d{ nullptr },
		* jacobian_d{ nullptr },
		* inverse_jacobian_d{ nullptr },
		* delta_d{ nullptr },
		* funcs_value_d{ nullptr };

	DataInitializerCUDA(int MATRIX_SIZE, int zeros_elements_per_row, int file_name, int power);
	~DataInitializerCUDA();
};