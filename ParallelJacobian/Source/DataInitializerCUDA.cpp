#include "DataInitializerCUDA.h"

DataInitializerCUDA::DataInitializerCUDA(
		int MATRIX_SIZE, int zeros_elements_per_row, int file_name,
		bool is_cublas, const Settings::SettingsData& s, int power = 1)
	: DataInitializer(MATRIX_SIZE, zeros_elements_per_row, file_name, s, power)
	  , is_cublas{is_cublas}
{
	if (is_cublas) {
		cublasCreate_v2(&cublasContextHandler);
	} else {
		cusolverDnCreate(&cusolverH);
		cudaMalloc((void**)&cusolver_pivot, MATRIX_SIZE * sizeof(int));
	}

	int x_blocks_count = (MATRIX_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
	cudaMalloc((void**)&points_d, MATRIX_SIZE * sizeof(double));
	cudaMalloc((void**)&indexes_d, MATRIX_SIZE * MATRIX_SIZE * sizeof(double));
	cudaMalloc((void**)&intermediate_funcs_value_d, x_blocks_count * MATRIX_SIZE * sizeof(double));
	cudaMalloc((void**)&jacobian_d, MATRIX_SIZE * MATRIX_SIZE * sizeof(double));
	cudaMalloc((void**)&delta_d, x_blocks_count * MATRIX_SIZE * sizeof(double));
	cudaMalloc((void**)&funcs_value_d, MATRIX_SIZE * sizeof(double));
	cudaMalloc((void**)&cublas_info, sizeof(int));

	if (is_cublas) {
		cudaMalloc((void**)&cublas_ajacobian_d, sizeof(double*));
		cudaMalloc((void**)&cublas_afunc_values_d, sizeof(double*));
		cudaMemcpy(cublas_ajacobian_d, &jacobian_d, sizeof(double*), cudaMemcpyHostToDevice);
		cudaMemcpy(cublas_afunc_values_d, &funcs_value_d, sizeof(double*), cudaMemcpyHostToDevice);
	}

	intermediate_funcs_value_h = new double[x_blocks_count * MATRIX_SIZE];
	delta_h = new double[x_blocks_count * MATRIX_SIZE];
	funcs_value_h = new double[MATRIX_SIZE];
}

DataInitializerCUDA::~DataInitializerCUDA() {
	if (is_cublas) {
		cublasDestroy_v2(cublasContextHandler);
	} else {
		cusolverDnDestroy(cusolverH);
		cudaFree(cusolver_pivot);
	}
	cudaFree(points_d);
	cudaFree(indexes_d);
	cudaFree(intermediate_funcs_value_d);
	cudaFree(jacobian_d);
	cudaFree(delta_d);
	cudaFree(funcs_value_d);
	cudaFree(cublas_info);

	if (is_cublas) {
		cudaFree(cublas_ajacobian_d);
		cudaFree(cublas_afunc_values_d);
	}
	if (workspace_d) {
		cudaFree(workspace_d);
	}

	delete[] intermediate_funcs_value_h;
	delete[] delta_h;
	delete[] funcs_value_h;
}
