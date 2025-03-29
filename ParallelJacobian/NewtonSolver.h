#pragma once
#include "DataInitializer.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "config.h"
#include <chrono>

struct NewtonSolver {
private:
	DataInitializer * data;

	void cpu_computeVec();
	double cpu_compute_derivative(int rowIndex, int colIndex);
	void cpu_compute_jacobian();
	void cpu_inverse();
	void cpu_compute_delta();

	void print_solution(int iterations_count, double* result);

public:
	NewtonSolver(DataInitializer* data);
	~NewtonSolver();

	void cpu_newton_solve();
	void gpu_newton_solve();
};

__global__ void gpu_compute_func_and_delta_values(double* points_d, double* indexes_d, double* vec_d);
__global__ void gpu_compute_jacobian(double* points_d, double* indexes_d, double* jacobian_d);
void gpu_cublasInverse(DataInitializer* data);
