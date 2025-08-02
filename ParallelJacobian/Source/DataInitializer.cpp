#include "DataInitializer.h"

#include <iostream>

#include "EditionalTools.h"

const std::string DataInitializer::csv_header =
"func_t,jcbn_t,delta_t,update_t,matrix_size,nnz_row,iter_num,mem_rss_max,mem_gpu_max,solver,label";

DataInitializer::DataInitializer(int MATRIX_SIZE, int zeros_elements_per_row,
		int file_name, const Settings::SettingsData& s,
		int power, bool is_csr)
	: settings{s}
{
	this->equation = new Equation(power);
	this->MATRIX_SIZE = MATRIX_SIZE;
	this->file_name = file_name;
	this->zeros_elements_per_row = zeros_elements_per_row;

	nnz_row = MATRIX_SIZE - zeros_elements_per_row;

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
    if (!is_csr) {
		indexes_h = new double[MATRIX_SIZE * MATRIX_SIZE];
	}
    points_h = new double[MATRIX_SIZE];
    vector_b_h = new double[MATRIX_SIZE];
	points_check = new double[MATRIX_SIZE];
#endif

#ifdef INTERMEDIATE_RESULTS
    intermediate_results = std::vector<double>(4, 0.0);
#endif

#ifdef TOTAL_ELASPED_TIME
	total_elapsed_time = 0.0;
#endif

    initialize_indexes_matrix_and_b(s, is_csr);
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
    if (indexes_h) {
        delete[] indexes_h;
    }
    delete[] points_h;
    delete[] vector_b_h;
	delete[] points_check;
#endif
    delete equation;
}

void DataInitializer::initialize_indexes_matrix_and_b(
        const Settings::SettingsData& s, bool is_csr) {
    for (int i = 0; i < MATRIX_SIZE; i++) {
        points_h[i] = 10;
    }

    if (!is_csr) {
        tools::generate_sparse_initial_indexes_matrix_and_vector_b(
                    indexes_h, vector_b_h, points_check, MATRIX_SIZE, equation,
                    zeros_elements_per_row, s);
    }
}
