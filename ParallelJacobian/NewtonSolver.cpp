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
        data->vec_h[i] = -data->vector_b_h[i];
		for (int j = 0; j < MATRIX_SIZE; j++) {
			data->vec_h[i] += tools::calculate_index_xn(data->indexes_h[i * MATRIX_SIZE + j], data->points_h[j]);
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
            data->delta_h[i] -= data->inverse_jacobian_h[i * MATRIX_SIZE + j] * data->vec_h[j];
        }
    }
}

void NewtonSolver::cpu_newton_solve() {
    double dx = 0.0;
    int iterations_count = 0;

    do {
        iterations_count++;

        cpu_computeVec();

        cpu_compute_jacobian();

        cpu_inverse();

        cpu_compute_delta();

        dx = 0;
        for (size_t i = 0; i < MATRIX_SIZE; ++i) {
            data->points_h[i] += data->delta_h[i];
            //data->points_h[i] = tools::calculate_index_xn(data->indexes_h[i * MATRIX_SIZE + i], data->points_h[i]);
            dx = std::max(dx, std::abs(data->delta_h[i]));
        }
    } while (dx > TOLERANCE);

    print_solution(iterations_count, data->points_h);
}

//
// GPU
//

void gpu_cublasInverse(DataInitializer* data) {
    cudaMemcpy(data->cublas_ajacobian_d, &data->jacobian_d, sizeof(double*), cudaMemcpyHostToDevice);
    cudaMemcpy(data->cublas_ainverse_jacobian_d, &data->inverse_jacobian_d, sizeof(double*), cudaMemcpyHostToDevice);

    cublasHandle_t cublasContextHandler;
    cublasCreate_v2(&cublasContextHandler);

    cublasStatus_t status2 = cublasDgetrfBatched(cublasContextHandler, MATRIX_SIZE, data->cublas_ajacobian_d, MATRIX_SIZE, data->cublas_pivot, data->cublas_info, 1);
    cublasStatus_t status = cublasDgetriBatched(cublasContextHandler, MATRIX_SIZE, (const double**)data->cublas_ajacobian_d, MATRIX_SIZE, data->cublas_pivot, data->cublas_ainverse_jacobian_d, MATRIX_SIZE, data->cublas_info, 1);

    cudaMemcpy(data->inverse_jacobian_h, data->inverse_jacobian_d, MATRIX_SIZE * MATRIX_SIZE * sizeof(double), cudaMemcpyDeviceToHost);
}

void NewtonSolver::print_solution(int iterations_count, double* result) {
    std::cout << "Iterations: " << iterations_count << "\n";
    std::cout << "Solution: \n";

    for (int i = 0; i < MATRIX_SIZE; i++) {
        std::cout << result[i] << "\n";
    }
}