#include "NewtonSolverCuDSS.h"

#include <memory>

#include "NewtonSolverGPUFunctions.h"
#include "FileOperations.h"
#include "iostream"
#include "math.h"
#include "chrono"
#include "cudss.h"
#include "EditionalTools.h"
#include <DataInitializerCuDSS.h>

NewtonSolverCuDSS::NewtonSolverCuDSS(DataInitializerCuDSS* data,
		const Settings::SettingsData& settings, SystemInfo& sinfo)
	: settings_{settings}
	, sinfo_{sinfo}
{
	this->data = data;
}

NewtonSolverCuDSS::~NewtonSolverCuDSS() {
}

void NewtonSolverCuDSS::solve(double* matrix_A_h, double* vector_b_d, double* vector_x_h, double* vector_x_d) {
	if (!data->analyzed) {
		cudssExecute(data->handler, CUDSS_PHASE_ANALYSIS, data->solverConfig, data->solverData, data->A, data->x, data->b);
		data->analyzed = true;
	}
	cudssExecute(data->handler, CUDSS_PHASE_FACTORIZATION, data->solverConfig, data->solverData, data->A, data->x, data->b);
	cudssExecute(data->handler, CUDSS_PHASE_SOLVE, data->solverConfig, data->solverData, data->A, data->x, data->b);

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


__global__ void gpu_compute_jacobian_csr(double* csr_values_d, int* csr_columns_d, int* csr_rows_ptr_d, double* points_d, double* csr_values_jacobian_d, int power, int matrix_size, int count_of_nnz) {
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	if (gid >= count_of_nnz) return;

	double f_plus = 0.0;
	double f_minus = 0.0;

	double element = csr_values_d[gid];
	double value = points_d[csr_columns_d[gid]];

	double x_value_plus = 1.0;
	double x_value_minus = 1.0;
	for (int i = 0; i < power; i++) {
		x_value_plus *= (value + EQURENCY);
		x_value_minus *= (value - EQURENCY);
	}
	f_plus = x_value_plus * element;
	f_minus = x_value_minus * element;

	double result = (f_plus - f_minus) / (2 * EQURENCY);
	csr_values_jacobian_d[gid] = result;
}

void NewtonSolverCuDSS::gpu_newton_solver_cudss() {
	std::unique_ptr<FileOperations> file_op = std::make_unique<FileOperations>(settings_.path);
	std::string file_name = "gpu_cudss_newton_solver_" + std::to_string(data->file_name) + ".csv";
	file_op->create_file(file_name, 4);
	file_op->append_file_headers(data->csv_header);

	NewtonSolverGPUFunctions::gpu_dummy_warmup << <1, 32 >> > ();
	cudaDeviceSynchronize();
	std::cout << "GPU CuDss Newton solver\n";
	std::cout << "Power: " << data->equation->get_power() << "\n";
	int iterations_count = 0;
	double dx = 0;

	auto start_total = std::chrono::steady_clock::now();

	cudaMemcpy(data->points_d, data->points_h, data->MATRIX_SIZE * sizeof(double), cudaMemcpyHostToDevice);

	cudaMemcpy(data->csr_cols_d, data->csr_cols_h, data->non_zero_count * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(data->csr_rows_d, data->csr_rows_h, (data->MATRIX_SIZE + 1) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(data->csr_values_d, data->csr_values_h, data->non_zero_count * sizeof(double), cudaMemcpyHostToDevice);
	delete[] data->csr_values_h;
	delete[] data->csr_rows_h;
	delete[] data->csr_cols_h;
	std::cout << "Non-zero count: " << data->non_zero_count << "\n";
	do {
		iterations_count++;

#ifdef INTERMEDIATE_RESULTS
		auto start = std::chrono::steady_clock::now();
#endif

		int threads_per_blocks = 256;
		int blockss = (data->MATRIX_SIZE + threads_per_blocks - 1) / threads_per_blocks;

		gpu_compute_func_values_csr << <blockss, threads_per_blocks >> > (
			data->points_d,
			data->csr_values_d,
			data->csr_cols_d,
			data->csr_rows_d,
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
		cudaDeviceSynchronize();
#ifdef INTERMEDIATE_RESULTS
		auto end = std::chrono::steady_clock::now();
		data->intermediate_results[0] = std::chrono::duration<double>(end - start).count();
		start = std::chrono::steady_clock::now();
#endif
		int threads_per_block = 256;
		int blocks = (data->non_zero_count + threads_per_block - 1) / threads_per_block;
		gpu_compute_jacobian_csr << <blocks, threads_per_block >> > (data->csr_values_d, data->csr_cols_d, data->csr_rows_d, data->points_d, data->jacobian_d, data->equation->get_power(), data->MATRIX_SIZE, data->non_zero_count);

		cudaDeviceSynchronize();

#ifdef INTERMEDIATE_RESULTS
		end = std::chrono::steady_clock::now();
		data->intermediate_results[1] = std::chrono::duration<double>(end - start).count();
		start = std::chrono::steady_clock::now();
#endif

		solve(data->jacobian_d, data->funcs_value_d, data->delta_h, data->delta_d);
		cudaDeviceSynchronize();
#ifdef INTERMEDIATE_RESULTS
		end = std::chrono::steady_clock::now();
		data->intermediate_results[2] = std::chrono::duration<double>(end - start).count();
		start = std::chrono::steady_clock::now();
#endif
		dx = 0.0;
		for (size_t i = 0; i < data->MATRIX_SIZE; ++i) {
			data->points_h[i] -= data->delta_h[i];
			dx = std::max(dx, std::abs(data->delta_h[i]));
		}

#ifdef INTERMEDIATE_RESULTS
		end = std::chrono::steady_clock::now();
		data->intermediate_results[3] = std::chrono::duration<double>(end - start).count();
		tools::print_intermediate_result(data, iterations_count, dx);
#endif
		cudaMemcpy(data->points_d, data->points_h, data->MATRIX_SIZE * sizeof(double), cudaMemcpyHostToDevice);

		file_op->append_file_data(
				data->intermediate_results, data->MATRIX_SIZE,
				data->nnz_row, iterations_count,
				sinfo_.mem_rss_usage_get(), sinfo_.gpu_mem_usage_get(),
				"cuDSS", data->settings.label);
	} while (dx > TOLERANCE);

	auto end_total = std::chrono::steady_clock::now();
	data->total_elapsed_time = std::chrono::duration<double>(end_total - start_total).count();
	tools::print_solution(data, iterations_count);
}
