#include "DataInitializerCuDSS.h"
#include <iostream>
#include "../Include/EditionalTools.h"

DataInitializerCuDSS::DataInitializerCuDSS(int MATRIX_SIZE, int zeros_elements_per_row, int file_name, int power)
	: DataInitializer(MATRIX_SIZE, zeros_elements_per_row, file_name, power, true) {
	// Allocate memory for CSR representation
	non_zero_count = MATRIX_SIZE * (MATRIX_SIZE - zeros_elements_per_row);
	csr_values_h = new double[non_zero_count];
	csr_rows_h = new int[MATRIX_SIZE + 1];
	csr_cols_h = new int[non_zero_count];
	cudaMalloc((void**)&csr_values_d, non_zero_count * sizeof(double));
	cudaMalloc((void**)&csr_rows_d, (MATRIX_SIZE + 1) * sizeof(int));
	cudaMalloc((void**)&csr_cols_d, non_zero_count * sizeof(int));
	// Allocate memory for other CUDA variables
	cudaMalloc((void**)&points_d, MATRIX_SIZE * sizeof(double));
	cudaMalloc((void**)&jacobian_d, non_zero_count * sizeof(double));
	cudaMalloc((void**)&delta_d, MATRIX_SIZE * sizeof(double));
	cudaMalloc((void**)&funcs_value_d, MATRIX_SIZE * sizeof(double));
	delta_h = new double[MATRIX_SIZE];
	funcs_value_h = new double[MATRIX_SIZE];

	tools::generate_sparse_initial_indexes_matrix_and_vector_b(csr_values_h, csr_rows_h, csr_cols_h, vector_b_h, points_check, MATRIX_SIZE, equation, zeros_elements_per_row);

	cudssCreate(&handler);

	cudssConfigCreate(&solverConfig);
	cudssDataCreate(handler, &solverData);

	cudssMatrixCreateDn(&b, MATRIX_SIZE, 1, MATRIX_SIZE, funcs_value_d, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR);
	cudssMatrixCreateDn(&x, MATRIX_SIZE, 1, MATRIX_SIZE, delta_d, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR);

	cudssMatrixCreateCsr(&A, MATRIX_SIZE, MATRIX_SIZE, non_zero_count, csr_rows_d, NULL, csr_cols_d, jacobian_d, CUDA_R_32I, CUDA_R_64F, mtype, mvtype, base);

}

int DataInitializerCuDSS::count_non_zero_elements(double* matrix_A) {
	int non_zero_count = 0;
	for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; i++) {
		if (matrix_A[i] != 0) {
			non_zero_count++;
		}
	}
	return non_zero_count;
}

DataInitializerCuDSS::~DataInitializerCuDSS() {
	// Free CSR memory
	cudaFree(csr_values_d);
	cudaFree(csr_rows_d);
	cudaFree(csr_cols_d);
	// Free other CUDA variables
	cudaFree(points_d);
	cudaFree(jacobian_d);
	cudaFree(delta_d);
	cudaFree(funcs_value_d);
	delete[] delta_h;
	delete[] funcs_value_h;

	cudssMatrixDestroy(A);
	cudssMatrixDestroy(x);
	cudssMatrixDestroy(b);
	cudssDataDestroy(handler, solverData);
	cudssConfigDestroy(solverConfig);
	cudssDestroy(handler);
}