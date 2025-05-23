#pragma once
#include "Equation.h"
#include "vector"
#include "config.h"
#include "cublas_v2.h"

#define BLOCK_SIZE 64
#define EQURENCY 1e-6
#define TOLERANCE 1e-5
#define SHAFFLE_CONST 0xffffffff

struct DataInitializer {
public:
	Equation* equation{ nullptr };
	int MATRIX_SIZE{ 0 };

	double* indexes_h{ nullptr },
		* points_h{ nullptr },
		* intermediate_funcs_value_h{ nullptr },
		* vector_b_h{ nullptr },
		* jacobian_h{ nullptr },
		* inverse_jacobian_h{ nullptr },
		* delta_h{ nullptr },
		* funcs_value_h{ nullptr },
		* points_check;

	double* indexes_d{ nullptr },
		* points_d{ nullptr },
		* intermediate_funcs_value_d{ nullptr },
		* vector_b_d{ nullptr },
		* jacobian_d{ nullptr },
		* inverse_jacobian_d{ nullptr },
		* delta_d{ nullptr },
		* funcs_value_d{ nullptr };
	int* cublas_pivot{ nullptr }, * cublas_info{ nullptr };
	double** cublas_ajacobian_d{ nullptr }, ** cublas_ainverse_jacobian_d{ nullptr };
	cublasHandle_t cublasContextHandler;

#ifdef INTERMEDIATE_RESULTS
	std::vector<double> intermediate_results;
#endif

#ifdef TOTAL_ELASPED_TIME
	double total_elapsed_time;
#endif

	void initialize_indexes_matrix_and_b();
 
	DataInitializer(int MATRIX_SIZE, int power = 1);
	~DataInitializer();

	friend struct NewtonSolver;
};