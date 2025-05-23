#include "NewtonSolverCuDSS.h"
#include "NewtonSolverGPUFunctions.h"
#include "DataInitializer.h"
#include "FileOperations.h"
#include "iostream"
#include "math.h"
#include "chrono"
#include "cudss.h"
#include "EditionalTools.h"

NewtonSolverCuDSS::NewtonSolverCuDSS(DataInitializer* data) {
	this->data = data;

	non_zero_count = count_non_zero_elements(data->indexes_h);
	csr_cols_h = new int[non_zero_count];
	csr_rows_h = new int[data->MATRIX_SIZE + 1];
	csr_values_h = new double[non_zero_count];

	cudaMalloc((void**)&csr_values_d, non_zero_count * sizeof(double));
	cudaMalloc((void**)&csr_rows_d, (data->MATRIX_SIZE + 1) * sizeof(int));
	cudaMalloc((void**)&csr_cols_d, non_zero_count * sizeof(int));
}

NewtonSolverCuDSS::~NewtonSolverCuDSS() {
	delete[] csr_cols_h;
	delete[] csr_rows_h;
	delete[] csr_values_h;
	cudaFree(csr_values_d);
	cudaFree(csr_rows_d);
	cudaFree(csr_cols_d);
}

int NewtonSolverCuDSS::count_non_zero_elements(double* matrix_A) {
	int non_zero_count = 0;
	for (int i = 0; i < data->MATRIX_SIZE * data->MATRIX_SIZE; i++) {
		if (matrix_A[i] != 0) {
			non_zero_count++;
		}
	}
	return non_zero_count;
}

void NewtonSolverCuDSS::parse_to_csr(int* csr_cols, int* csr_rows, double* csr_values, double* matrix_A) {
	int non_zero_count = 0;
	csr_rows[0] = 0;
	for (int i = 0; i < data->MATRIX_SIZE; ++i) {
		for (int j = 0; j < data->MATRIX_SIZE; ++j) {
			if (matrix_A[i * data->MATRIX_SIZE + j] != 0) {
				csr_cols[non_zero_count] = j;
				csr_values[non_zero_count] = matrix_A[i * data->MATRIX_SIZE + j];
				non_zero_count++;
			}
		}
		csr_rows[i + 1] = non_zero_count;
	}
}

void NewtonSolverCuDSS::solve(double* matrix_A_h, double* vector_b_d, double* vector_x_h, double* vector_x_d) {
	//parse_to_csr(csr_cols_h, csr_rows_h, csr_values_h, matrix_A_h);

	//cudaMemcpy(csr_cols_d, csr_cols_h, non_zero_count * sizeof(int), cudaMemcpyHostToDevice);
	//cudaMemcpy(csr_rows_d, csr_rows_h, (data->MATRIX_SIZE + 1) * sizeof(int), cudaMemcpyHostToDevice);
	//cudaMemcpy(csr_values_d, csr_values_h, non_zero_count * sizeof(double), cudaMemcpyHostToDevice);

	cudssHandle_t handler;
	cudssConfig_t solverConfig;
	cudssData_t solverData;
	cudssCreate(&handler);

	cudssConfigCreate(&solverConfig);
	cudssDataCreate(handler, &solverData);

	cudssMatrix_t x, b;
	cudssMatrixCreateDn(&b, data->MATRIX_SIZE, 1, data->MATRIX_SIZE, vector_b_d, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR);
	cudssMatrixCreateDn(&x, data->MATRIX_SIZE, 1, data->MATRIX_SIZE, vector_x_d, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR);

	cudssMatrix_t A;
	cudssMatrixType_t mtype = CUDSS_MTYPE_GENERAL;
	cudssMatrixViewType_t mvtype = CUDSS_MVIEW_FULL;
	cudssIndexBase_t base = CUDSS_BASE_ZERO;
	cudssMatrixCreateCsr(&A, data->MATRIX_SIZE, data->MATRIX_SIZE, non_zero_count, csr_rows_d, NULL, csr_cols_d, data->jacobian_d, CUDA_R_32I, CUDA_R_64F, mtype, mvtype, base);

	cudssExecute(handler, CUDSS_PHASE_ANALYSIS, solverConfig, solverData, A, x, b);
	cudssExecute(handler, CUDSS_PHASE_FACTORIZATION, solverConfig, solverData, A, x, b);
	cudssExecute(handler, CUDSS_PHASE_SOLVE, solverConfig, solverData, A, x, b);

	cudssMatrixDestroy(A);
	cudssMatrixDestroy(x);
	cudssMatrixDestroy(b);
	cudssDataDestroy(handler, solverData);
	cudssConfigDestroy(solverConfig);
	cudssDestroy(handler);

	cudaDeviceSynchronize();

	cudaMemcpy(vector_x_h, vector_x_d, data->MATRIX_SIZE * sizeof(double), cudaMemcpyDeviceToHost);
}

__global__ void gpu_compute_func_values_csr(
	double* points_d,
	double* csr_values_d,
	int* csr_cols_d,
	int* csr_rows_d,
	double* vec_d,
	int MATRIX_SIZE,
	int power
) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;

	if (row >= MATRIX_SIZE) return;

	int row_start = csr_rows_d[row];
	int row_end = csr_rows_d[row + 1];

	double sum = 0.0;

	for (int i = row_start; i < row_end; i++) {
		int col = csr_cols_d[i];
		double value = csr_values_d[i];

		double point_pow = 1.0;
		for (int p = 0; p < power; p++) {
			point_pow *= points_d[col];
		}

		sum += value * point_pow;
	}

	vec_d[row] = sum;
}


__global__ void gpu_compute_jacobian_csr(double* csr_values_d, int* csr_columns_d, int* csr_rows_ptr_d, double* points_d,
	double* csr_values_jacobian_d, int power, int matrix_size, int count_of_nnz) {
	int gid = blockIdx.x * blockDim.x + threadIdx.x;

	int start_i = 0;
	int start_index = 0;
	int end_index = 0;

	if (gid < count_of_nnz) {


		for (int i = 0; i < matrix_size; i++) {
			if (csr_rows_ptr_d[i] >= gid) {
				if (csr_rows_ptr_d[i] == gid) {
					start_i = i;
					start_index = csr_rows_ptr_d[i];
					break;
				}
				else {
					start_i = i - 1;
					start_index = csr_rows_ptr_d[i - 1];
					break;
				}
			}
			else {
				start_i = i;
				start_index = csr_rows_ptr_d[i];
			}
		}
		end_index = csr_rows_ptr_d[start_i + 1];


		double f_minus = 0.0;
		double f_plus = 0.0;
		double result = 0.0;

		for (int i = start_index; i < end_index; i++) {
			double element = csr_values_d[i];
			double value = points_d[csr_columns_d[i]];

			if (csr_columns_d[gid] == csr_columns_d[i]) {
				double x_value_plus = 1;
				double x_value_minus = 1;
				for (int i = 0; i < power; i++) {
					x_value_plus *= (value + EQURENCY);
					x_value_minus *= (value - EQURENCY);
				}
				f_minus += x_value_minus * element;
				f_plus += x_value_plus * element;
			}
			else {
				f_minus += value * element;
				f_plus += value * element;
			}
		}

		__syncthreads();

		result = (f_plus - f_minus) / (2 * EQURENCY);

		csr_values_jacobian_d[gid] = result;
	}
}

void NewtonSolverCuDSS::gpu_newton_solver_cudss() {
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	int version = prop.major;
	FileOperations* file_op = new FileOperations();
	std::string file_name = "gpu_cudss_newton_solver_" + std::to_string(data->MATRIX_SIZE) + ".csv";
	file_op->create_file(file_name, 4);
	file_op->append_file_headers("func_value_t,jacobian_value_t,delta_value_t,update_points_t,matrix_size");

	NewtonSolverGPUFunctions::gpu_dummy_warmup << <1, 32 >> > ();
	cudaDeviceSynchronize();
	std::cout << "GPU CuDss Newton solver\n";
	std::cout << "Power: " << data->equation->get_power() << "\n";
	int x_blocks_count = (data->MATRIX_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
	int iterations_count = 0;
	double dx = 0;

	dim3 blockDim(BLOCK_SIZE, 1, 1);
	dim3 gridDim(x_blocks_count, data->MATRIX_SIZE, 1);

	auto start_total = std::chrono::high_resolution_clock::now();

	cudaMemcpy(data->points_d, data->points_h, data->MATRIX_SIZE * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(data->indexes_d, data->indexes_h, data->MATRIX_SIZE * data->MATRIX_SIZE * sizeof(double), cudaMemcpyHostToDevice);
	for (size_t i = 0; i < data->MATRIX_SIZE; ++i) {
		data->points_h[i] += data->delta_h[i];
		dx = std::max(dx, std::abs(data->delta_h[i]));
	}

	parse_to_csr(csr_cols_h, csr_rows_h, csr_values_h, data->indexes_h);
	cudaMemcpy(csr_cols_d, csr_cols_h, non_zero_count * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(csr_rows_d, csr_rows_h, (data->MATRIX_SIZE + 1) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(csr_values_d, csr_values_h, non_zero_count * sizeof(double), cudaMemcpyHostToDevice);
	std::cout << "Non-zero count: " << non_zero_count << "\n";
	do {
		iterations_count++;

#ifdef INTERMEDIATE_RESULTS
		auto start = std::chrono::high_resolution_clock::now();
#endif

		int threads_per_blocks = 256;
		int blockss = (data->MATRIX_SIZE + threads_per_blocks - 1) / threads_per_blocks;

		gpu_compute_func_values_csr << <blockss, threads_per_blocks >> > (
			data->points_d,
			csr_values_d,
			csr_cols_d,
			csr_rows_d,
			data->funcs_value_d,
			data->MATRIX_SIZE,
			data->equation->get_power()
			);
		cudaDeviceSynchronize();

		cudaMemcpy(data->funcs_value_h, data->funcs_value_d, data->MATRIX_SIZE * sizeof(double), cudaMemcpyDeviceToHost);
		for (int i = 0; i < data->MATRIX_SIZE; i++) {
			data->funcs_value_h[i] -= data->vector_b_h[i];
		}

		cudaMemcpy(data->funcs_value_d, data->funcs_value_h, data->MATRIX_SIZE * sizeof(double), cudaMemcpyHostToDevice);

#ifdef INTERMEDIATE_RESULTS
		auto end = std::chrono::high_resolution_clock::now();
		data->intermediate_results[0] = std::chrono::duration<double>(end - start).count();
		start = std::chrono::high_resolution_clock::now();
#endif
		int threads_per_block = 256;
		int blocks = (non_zero_count + threads_per_block - 1) / threads_per_block;
		gpu_compute_jacobian_csr << <blocks, threads_per_block >> > (csr_values_d, csr_cols_d, csr_rows_d, data->points_d, data->jacobian_d, data->equation->get_power(), data->MATRIX_SIZE, non_zero_count);

		cudaMemcpy(data->jacobian_h, data->jacobian_d, data->MATRIX_SIZE * data->MATRIX_SIZE * sizeof(double), cudaMemcpyDeviceToHost);

		//NewtonSolverGPUFunctions::gpu_compute_jacobian << <gridDim, blockDim, 2 * blockDim.x * sizeof(double) >> > (
		//	data->points_d, data->indexes_d, data->jacobian_d, data->MATRIX_SIZE, data->equation->get_power());
		cudaDeviceSynchronize();

		//cudaMemcpy(data->jacobian_h, data->jacobian_d, data->MATRIX_SIZE * data->MATRIX_SIZE * sizeof(double), cudaMemcpyDeviceToHost);

#ifdef INTERMEDIATE_RESULTS
		end = std::chrono::high_resolution_clock::now();
		data->intermediate_results[1] = std::chrono::duration<double>(end - start).count();
		start = std::chrono::high_resolution_clock::now();
#endif

		solve(data->jacobian_d, data->funcs_value_d, data->delta_h, data->delta_d);
#ifdef INTERMEDIATE_RESULTS
		end = std::chrono::high_resolution_clock::now();
		data->intermediate_results[3] = std::chrono::duration<double>(end - start).count();
		start = std::chrono::high_resolution_clock::now();
#endif
		dx = 0.0;
		for (size_t i = 0; i < data->MATRIX_SIZE; ++i) {
			data->points_h[i] -= data->delta_h[i];
			dx = std::max(dx, std::abs(data->delta_h[i]));
		}

#ifdef INTERMEDIATE_RESULTS
		end = std::chrono::high_resolution_clock::now();
		data->intermediate_results[4] = std::chrono::duration<double>(end - start).count();
#endif
		tools::print_intermediate_result(data, iterations_count, dx, true);
		cudaMemcpy(data->points_d, data->points_h, data->MATRIX_SIZE * sizeof(double), cudaMemcpyHostToDevice);
		file_op->append_file_data(data->intermediate_results, data->MATRIX_SIZE);
	} while (dx > TOLERANCE);

	auto end_total = std::chrono::high_resolution_clock::now();
	data->total_elapsed_time = std::chrono::duration<double>(end_total - start_total).count();
	tools::print_solution(data, iterations_count);
}