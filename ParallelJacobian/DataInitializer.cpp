#include "DataInitializer.h"
#include "cuda_runtime.h"
#include "EditionalTools.h"
#include "stdlib.h"
#include "iostream"

DataInitializer::DataInitializer() {
    int x_blocks_count = (MATRIX_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;

#ifdef GPU_SOLVER
    cudaMalloc((void**)&points_d, MATRIX_SIZE * sizeof(double));
    cudaMalloc((void**)&indexes_d, MATRIX_SIZE * MATRIX_SIZE * sizeof(double));
    cudaMalloc((void**)&intermediate_funcs_value_d, x_blocks_count * MATRIX_SIZE * sizeof(double));
    cudaMalloc((void**)&jacobian_d, MATRIX_SIZE * MATRIX_SIZE * sizeof(double));
    cudaMalloc((void**)&inverse_jacobian_d, MATRIX_SIZE * MATRIX_SIZE * sizeof(double));
    cudaMalloc((void**)&delta_d, x_blocks_count * MATRIX_SIZE * sizeof(double));
    cudaMalloc((void**)&funcs_value_d, MATRIX_SIZE * sizeof(double));
    cudaMalloc((void**)&vector_b_d, MATRIX_SIZE * sizeof(double));

    cudaMalloc((void**)&cublas_pivot, MATRIX_SIZE * sizeof(int));
    cudaMalloc((void**)&cublas_info, sizeof(int));
    cudaMalloc((void**)&cublas_ajacobian_d, sizeof(double*));
    cudaMalloc((void**)&cublas_ainverse_jacobian_d, sizeof(double*));
#endif

#ifdef PINNED_MEMORY
	cudaMallocHost((double**)&points_h, MATRIX_SIZE * sizeof(double));
	cudaMallocHost((double**)&indexes_h, MATRIX_SIZE * MATRIX_SIZE * sizeof(double));
	cudaMallocHost((double**)&intermediate_funcs_value_h, x_blocks_count * MATRIX_SIZE * sizeof(double));
	cudaMallocHost((double**)&jacobian_h, MATRIX_SIZE * MATRIX_SIZE * sizeof(double));
	cudaMallocHost((double**)&inverse_jacobian_h, MATRIX_SIZE * MATRIX_SIZE * sizeof(double));
	cudaMallocHost((double**)&funcs_value_h, MATRIX_SIZE * sizeof(double));
	cudaMallocHost((double**)&vector_b_h, MATRIX_SIZE * sizeof(double));
	cudaMallocHost((double**)&delta_h, x_blocks_count * MATRIX_SIZE * sizeof(double));
#else
    indexes_h = new double[MATRIX_SIZE * MATRIX_SIZE];
    jacobian_h = new double[MATRIX_SIZE * MATRIX_SIZE];
    points_h = new double[MATRIX_SIZE];
    funcs_value_h = new double[MATRIX_SIZE];
    inverse_jacobian_h = new double[MATRIX_SIZE * MATRIX_SIZE];
    delta_h = new double[x_blocks_count * MATRIX_SIZE];
    vector_b_h = new double[MATRIX_SIZE];
    intermediate_funcs_value_h = new double[x_blocks_count * MATRIX_SIZE];
#endif

#ifdef INTERMEDIATE_RESULTS
    intermediate_results = std::vector<double>(5, 0.0);
#endif

#ifdef TOTAL_ELASPED_TIME
	total_elapsed_time = 0.0;
#endif

    initialize_indexes_matrix_and_b();
}

DataInitializer::~DataInitializer() {
#ifdef GPU_SOLVER
    cudaFree(points_d);
    cudaFree(indexes_d);
    cudaFree(intermediate_funcs_value_d);
    cudaFree(jacobian_d);
    cudaFree(inverse_jacobian_d);
    cudaFree(funcs_value_d);
	cudaFree(delta_d);
	cudaFree(vector_b_d);
	cudaFree(cublas_pivot);
	cudaFree(cublas_info);
	cudaFree(cublas_ajacobian_d);
	cudaFree(cublas_ainverse_jacobian_d);
#endif

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
    delete[] jacobian_h;
    delete[] points_h;
    delete[] funcs_value_h;
    delete[] inverse_jacobian_h;
    delete[] vector_b_h;
    delete[] delta_h;
	delete[] intermediate_funcs_value_h;
#endif
}

void DataInitializer::initialize_indexes_matrix_and_b() {
    int x_blocks_count = (MATRIX_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int i = 0; i < MATRIX_SIZE; i++) {
        points_h[i] = static_cast<double>(rand()) / RAND_MAX;
        funcs_value_h[i] = 0;
        delta_h[i] = 0;
#ifdef GPU_SOLVER
        for (int j = 0; j < x_blocks_count; j++) {
            intermediate_funcs_value_h[i * x_blocks_count + j] = 0;
        }
#endif

        for (int j = 0; j < MATRIX_SIZE; j++) {
            jacobian_h[i * MATRIX_SIZE + j] = 0;
            inverse_jacobian_h[i * MATRIX_SIZE + j] = 0;
        }
    }

    tools::generate_initial_indexes_matrix_and_vector_b(indexes_h, vector_b_h);
    //tools::generate_sparse_initial_indexes_matrix_and_vector_b(indexes_h, vector_b_h, 500);
}