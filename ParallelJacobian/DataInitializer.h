#pragma once
#include "vector"
#include "config.h"

#define MATRIX_SIZE 100
#define BLOCK_SIZE 128
#define EQURENCY 1e-7
#define TOLERANCE 1e-6

struct DataInitializer {
public:
	double* indexes_h{ nullptr },
		* points_h{ nullptr },
		* intermediate_funcs_value_h{ nullptr },
		* vector_b_h{ nullptr },
		* jacobian_h{ nullptr },
		* inverse_jacobian_h{ nullptr },
		* delta_h{ nullptr },
		* funcs_value_h{ nullptr };

#ifdef GPU_SOLVER
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
#endif

#ifdef INTERMEDIATE_RESULTS
	std::vector<double> intermediate_results;
#endif

#ifdef TOTAL_ELASPED_TIME
	double total_elapsed_time;
#endif

	void initialize_indexes_matrix_and_b();
 
	DataInitializer();
	~DataInitializer();

	friend struct NewtonSolver;
};