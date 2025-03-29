#include "DataInitializer.h"
#include "cuda_runtime.h"
#include "EditionalTools.h"
#include "stdlib.h"
#include "iostream"

DataInitializer::DataInitializer() {
    int x_blocks_count = (MATRIX_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaMalloc((void**)&points_d, MATRIX_SIZE * sizeof(double));
    cudaMalloc((void**)&indexes_d, MATRIX_SIZE * MATRIX_SIZE * sizeof(double));
    cudaMalloc((void**)&vector_d, x_blocks_count * MATRIX_SIZE * sizeof(double));
    cudaMalloc((void**)&jacobian_d, MATRIX_SIZE * MATRIX_SIZE * sizeof(double));
    cudaMalloc((void**)&inverse_jacobian_d, MATRIX_SIZE * MATRIX_SIZE * sizeof(double));
    cudaMalloc((void**)&delta_d, x_blocks_count * MATRIX_SIZE * sizeof(double));
    cudaMalloc((void**)&vec_d, MATRIX_SIZE * sizeof(double));
    cudaMalloc((void**)&vector_b_d, MATRIX_SIZE * sizeof(double));

    cudaMalloc((void**)&cublas_pivot, MATRIX_SIZE * sizeof(int));
    cudaMalloc((void**)&cublas_info, sizeof(int));
    cudaMalloc((void**)&cublas_ajacobian_d, sizeof(double*));
    cudaMalloc((void**)&cublas_ainverse_jacobian_d, sizeof(double*));

    indexes_h = new double[MATRIX_SIZE * MATRIX_SIZE];
    jacobian_h = new double[MATRIX_SIZE * MATRIX_SIZE];
    points_h = new double[MATRIX_SIZE];
    vec_h = new double[MATRIX_SIZE];
    inverse_jacobian_h = new double[MATRIX_SIZE * MATRIX_SIZE];
    delta_h = new double[x_blocks_count * MATRIX_SIZE];
    vector_b_h = new double[MATRIX_SIZE];
    vector_h = new double[x_blocks_count * MATRIX_SIZE];

#ifdef INTERMEDIATE_RESULTS
    intermediate_results = std::vector<double>(5, 0.0);
#endif

    initialize_indexes_matrix_and_b();
}

DataInitializer::~DataInitializer() {
    cudaFree(points_d);
    cudaFree(indexes_d);
    cudaFree(vector_d);
    cudaFree(jacobian_d);
    cudaFree(inverse_jacobian_d);
    cudaFree(vec_d);

    delete[] indexes_h;
    delete[] jacobian_h;
    delete[] points_h;
    delete[] vec_h;
    delete[] inverse_jacobian_h;
    delete[] delta_h;
    delete[] vector_b_h;
}

void DataInitializer::initialize_indexes_matrix_and_b() {
    int x_blocks_count = (MATRIX_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int i = 0; i < MATRIX_SIZE; i++) {
        points_h[i] = (rand() % 10) / 10.0;
        vec_h[i] = 0;
        delta_h[i] = 0;
        for (int j = 0; j < x_blocks_count; j++) {
            vector_h[i * x_blocks_count + j] = 0;
        }

        for (int j = 0; j < MATRIX_SIZE; j++) {
            jacobian_h[i * MATRIX_SIZE + j] = 0;
            inverse_jacobian_h[i * MATRIX_SIZE + j] = 0;
        }
    }

    tools::generate_initial_indexes_matrix_and_vector_b(indexes_h, vector_b_h);
}