#include "math.h"
#include <iostream>
#include<algorithm>
#include "cublas.h"
#include "NewtonSolver.h"
#include "EditionalTools.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

NewtonSolver::NewtonSolver(DataInitializer* dataInitializer) {
	data = dataInitializer;
}

NewtonSolver::~NewtonSolver() {
}

void NewtonSolver::cpu_computeVec() {
	for (int i = 0; i < MATRIX_SIZE; i++) {
        data->funcs_value_h[i] = -data->vector_b_h[i];
		for (int j = 0; j < MATRIX_SIZE; j++) {
			data->funcs_value_h[i] += tools::calculate_index_xn(data->indexes_h[i * MATRIX_SIZE + j], data->points_h[j]);
        }
	}
}

double NewtonSolver::cpu_compute_derivative(int rowIndex, int colIndex) {
    double temp_plus[MATRIX_SIZE], temp_minus[MATRIX_SIZE];

    for (int i = 0; i < MATRIX_SIZE; ++i) {
        temp_plus[i] = data->points_h[i];
        temp_minus[i] = data->points_h[i];
    }

    double equrency = EQURENCY;

    temp_plus[colIndex] += equrency;
    temp_minus[colIndex] -= equrency;

    double f_plus = 0.0, f_minus = 0.0;
    for (int j = 0; j < MATRIX_SIZE; ++j) {
        f_plus += tools::calculate_index_xn(data->indexes_h[rowIndex * MATRIX_SIZE + j], temp_plus[j]);
        f_minus += tools::calculate_index_xn(data->indexes_h[rowIndex * MATRIX_SIZE + j], temp_minus[j]);
    }

    return (f_plus - f_minus) / (2.0 * equrency);
}

//
//   CPU
//

void NewtonSolver::cpu_compute_jacobian() {
    for (int i = 0; i < MATRIX_SIZE; ++i) {
        for (int j = 0; j < MATRIX_SIZE; ++j) {
            data->jacobian_h[i * MATRIX_SIZE + j] = cpu_compute_derivative(i, j);
        }
    }
}

void NewtonSolver::cpu_inverse() {
    for (int i = 0; i < MATRIX_SIZE; i++) data->inverse_jacobian_h[i * MATRIX_SIZE + i] = 1.0;

    for (int i = 0; i < MATRIX_SIZE; i++) {
        double temp = data->jacobian_h[i * MATRIX_SIZE + i];
        for (int j = 0; j < MATRIX_SIZE; j++) {
            data->jacobian_h[i * MATRIX_SIZE + j] /= temp;
            data->inverse_jacobian_h[i * MATRIX_SIZE + j] /= temp;
        }
        for (int k = 0; k < MATRIX_SIZE; k++) {
            if (k != i) {
                temp = data->jacobian_h[k * MATRIX_SIZE + i];
                for (int j = 0; j < MATRIX_SIZE; j++) {
                    data->jacobian_h[k * MATRIX_SIZE + j] -= data->jacobian_h[i * MATRIX_SIZE + j] * temp;
                    data->inverse_jacobian_h[k * MATRIX_SIZE + j] -= data->inverse_jacobian_h[i * MATRIX_SIZE + j] * temp;
                }
            }
        }
    }
}

void NewtonSolver::cpu_compute_delta() {
    for (int i = 0; i < MATRIX_SIZE; i++) {
        data->delta_h[i] = 0.0;
        for (int j = 0; j < MATRIX_SIZE; j++) {
            data->delta_h[i] -= data->inverse_jacobian_h[i * MATRIX_SIZE + j] * data->funcs_value_h[j];
        }
    }
}

void NewtonSolver::cpu_newton_solve() {
    std::cout << "CPU Newton solver\n";
    double dx = 0.0;
    int iterations_count = 0;

#ifdef TOTAL_ELASPED_TIME
	auto start_total = std::chrono::high_resolution_clock::now();
#endif
    do {
        iterations_count++;

#ifdef INTERMEDIATE_RESULTS
		auto start = std::chrono::high_resolution_clock::now();
#endif
        cpu_computeVec();
#ifdef INTERMEDIATE_RESULTS
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed = end - start;
        data->intermediate_results[0] = elapsed.count();
#endif

#ifdef INTERMEDIATE_RESULTS
		start = std::chrono::high_resolution_clock::now();
#endif
        cpu_compute_jacobian();
#ifdef INTERMEDIATE_RESULTS
		end = std::chrono::high_resolution_clock::now();
		elapsed = end - start;
        data->intermediate_results[1] = elapsed.count();
#endif

#ifdef INTERMEDIATE_RESULTS
		start = std::chrono::high_resolution_clock::now();
#endif
        cpu_inverse();
#ifdef INTERMEDIATE_RESULTS
		end = std::chrono::high_resolution_clock::now();
		elapsed = end - start;
        data->intermediate_results[2] = elapsed.count();
#endif

#ifdef INTERMEDIATE_RESULTS
		start = std::chrono::high_resolution_clock::now();
#endif
        cpu_compute_delta();
#ifdef INTERMEDIATE_RESULTS
		end = std::chrono::high_resolution_clock::now();
		elapsed = end - start;
        data->intermediate_results[3] = elapsed.count();
#endif

#ifdef INTERMEDIATE_RESULTS
        start = std::chrono::high_resolution_clock::now();
#endif
        dx = 0;
        for (size_t i = 0; i < MATRIX_SIZE; ++i) {
            data->points_h[i] += data->delta_h[i];
            //data->points_h[i] = tools::calculate_index_xn(data->indexes_h[i * MATRIX_SIZE + i], data->points_h[i]);
            dx = std::max(dx, std::abs(data->delta_h[i]));
        }
#ifdef INTERMEDIATE_RESULTS
		end = std::chrono::high_resolution_clock::now();
		elapsed = end - start;
		data->intermediate_results[4] = elapsed.count();
#endif

#ifdef INTERMEDIATE_RESULTS
		std::cout << "\nIteration: " << iterations_count << "\n";
        std::cout << "===============================================================\n";
        std::cout << "Intermediate results: \n";
        std::cout << "Compute func values: " << data->intermediate_results[0] << "\n";
        std::cout << "Compute jacobian: " << data->intermediate_results[1] << "\n";
        std::cout << "Compute inverse jacobian: " << data->intermediate_results[2] << "\n";
        std::cout << "Compute delta: " << data->intermediate_results[3] << "\n";
        std::cout << "Update points: " << data->intermediate_results[4] << "\n";
		std::cout << "Error: " << dx << "\n";
        std::cout << "===============================================================\n";
#endif
    } while (dx > TOLERANCE);
#ifdef TOTAL_ELASPED_TIME
	auto end_total = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed_total = end_total - start_total;
	data->total_elapsed_time = elapsed_total.count();
#endif

    print_solution(iterations_count, data->points_h);
}

//
// GPU
//

#ifdef GPU_SOLVER
void gpu_cublasInverse(DataInitializer* data) {
    cudaMemcpy(data->cublas_ajacobian_d, &data->jacobian_d, sizeof(double*), cudaMemcpyHostToDevice);
    cudaMemcpy(data->cublas_ainverse_jacobian_d, &data->inverse_jacobian_d, sizeof(double*), cudaMemcpyHostToDevice);

    cublasHandle_t cublasContextHandler;
    cublasCreate_v2(&cublasContextHandler);

    cublasStatus_t status2 = cublasDgetrfBatched(cublasContextHandler, MATRIX_SIZE, data->cublas_ajacobian_d, MATRIX_SIZE, data->cublas_pivot, data->cublas_info, 1);
    cublasStatus_t status = cublasDgetriBatched(cublasContextHandler, MATRIX_SIZE, (const double**)data->cublas_ajacobian_d, MATRIX_SIZE, data->cublas_pivot, data->cublas_ainverse_jacobian_d, MATRIX_SIZE, data->cublas_info, 1);

    cudaMemcpy(data->inverse_jacobian_h, data->inverse_jacobian_d, MATRIX_SIZE * MATRIX_SIZE * sizeof(double), cudaMemcpyDeviceToHost);
}
#endif

void NewtonSolver::print_solution(int iterations_count, double* result) {
    std::cout << "Total Iterations count: " << iterations_count << "\n";

#ifdef TOTAL_ELASPED_TIME
	std::cout << "Total elapsed time: " << data->total_elapsed_time << "\n";
#endif
#ifdef SOLUTION_PRINT
    std::cout << "Solution: \n";

    for (int i = 0; i < MATRIX_SIZE; i++) {
        std::cout << result[i] << "\n";
    }
#endif
    std::cout << "\n";
}