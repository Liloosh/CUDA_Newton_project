#include "DataInitializer.h"
#include "cuda_runtime.h"
#include "EditionalTools.h"
#include "stdlib.h"
#include "iostream"

DataInitializer::DataInitializer(int MATRIX_SIZE, int zeros_elements_per_row, int file_name, int power, bool isCuDSS) {
	this->equation = new Equation(power);
	this->MATRIX_SIZE = MATRIX_SIZE;
	this->file_name = file_name;
	this->zeros_elements_per_row = zeros_elements_per_row;

#ifdef PINNED_MEMORY
	int x_blocks_count = (MATRIX_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;

	cudaMallocHost((double**)&points_h, MATRIX_SIZE * sizeof(double));
	cudaMallocHost((double**)&indexes_h, MATRIX_SIZE * MATRIX_SIZE * sizeof(double));
	cudaMallocHost((double**)&intermediate_funcs_value_h, x_blocks_count * MATRIX_SIZE * sizeof(double));
	cudaMallocHost((double**)&jacobian_h, MATRIX_SIZE * MATRIX_SIZE * sizeof(double));
	cudaMallocHost((double**)&inverse_jacobian_h, MATRIX_SIZE * MATRIX_SIZE * sizeof(double));
	cudaMallocHost((double**)&funcs_value_h, MATRIX_SIZE * sizeof(double));
	cudaMallocHost((double**)&vector_b_h, MATRIX_SIZE * sizeof(double));
	cudaMallocHost((double**)&delta_h, x_blocks_count * MATRIX_SIZE * sizeof(double));
#else
	if (!isCuDSS) {
		indexes_h = new double[MATRIX_SIZE * MATRIX_SIZE];
	}
    points_h = new double[MATRIX_SIZE];
    vector_b_h = new double[MATRIX_SIZE];
	points_check = new double[MATRIX_SIZE];
#endif

#ifdef INTERMEDIATE_RESULTS
    intermediate_results = std::vector<double>(5, 0.0);
#endif

#ifdef TOTAL_ELASPED_TIME
	total_elapsed_time = 0.0;
#endif

    initialize_indexes_matrix_and_b(isCuDSS);
}

DataInitializer::~DataInitializer() {

#ifdef PINNED_MEMORY
	cudaFreeHost(points_h);
	cudaFreeHost(indexes_h);
	cudaFreeHost(intermediate_funcs_value_h);
	cudaFreeHost(jacobian_h);
	cudaFreeHost(inverse_jacobian_h);
	cudaFreeHost(funcs_value_h);
	cudaFreeHost(delta_h);
	cudaFreeHost(vector_b_h);
#else
    delete[] indexes_h;
    delete[] points_h;
    delete[] vector_b_h;
	delete[] points_check;
#endif
    delete equation;
}

void DataInitializer::initialize_indexes_matrix_and_b(bool isCuDSS) {
    //int x_blocks_count = (MATRIX_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int i = 0; i < MATRIX_SIZE; i++) {
        points_h[i] = 10;
#ifdef GPU_SOLVER
        //for (int j = 0; j < x_blocks_count; j++) {
        //    intermediate_funcs_value_h[i * x_blocks_count + j] = 0;
        //}
#endif
    }

	if (!isCuDSS) {
		tools::generate_sparse_initial_indexes_matrix_and_vector_b(indexes_h, vector_b_h, points_check, MATRIX_SIZE, equation, zeros_elements_per_row);
	}
    //tools::generate_sparse_initial_indexes_matrix_and_vector_b(indexes_h, vector_b_h, 500, MATRIX_SIZE);
}
