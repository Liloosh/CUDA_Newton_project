﻿#include "NewtonSolver.h"
#include "stdio.h"
#include <iostream>

#ifdef GPU_SOLVER
__global__ void gpu_compute_func_and_delta_values(double* points_d, double* indexes_d, double* vec_d) {
    int x_blocks_count = (MATRIX_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int gidx = blockDim.x * blockIdx.x + threadIdx.x;
    int gidy = blockDim.y * blockIdx.y + threadIdx.y;
    int tidx = threadIdx.x;

    extern __shared__ double shared_points[];

    if (gidx < MATRIX_SIZE) {
        shared_points[threadIdx.x] = points_d[gidx];
        shared_points[threadIdx.x + blockDim.x] = indexes_d[gidy * MATRIX_SIZE + gidx];
        //printf("points: %f %f\n", shared_points[threadIdx.x], shared_points[threadIdx.x + blockDim.x]);
    }
    else {
        shared_points[threadIdx.x] = 0.0;
        shared_points[threadIdx.x + blockDim.x] = 0.0;
    }
    __syncthreads();

    shared_points[tidx] *= shared_points[tidx + blockDim.x];
    __syncthreads();

    if (BLOCK_SIZE >= 1024 && threadIdx.x < 512) {
        shared_points[threadIdx.x] += shared_points[threadIdx.x + 512];
    }

    __syncthreads();

    if (BLOCK_SIZE >= 512 && threadIdx.x < 256) {
        shared_points[threadIdx.x] += shared_points[threadIdx.x + 256];
    }

    __syncthreads();

    if (BLOCK_SIZE >= 256 && threadIdx.x < 128) {
        shared_points[threadIdx.x] += shared_points[threadIdx.x + 128];
    }

    __syncthreads();

    if (BLOCK_SIZE >= 128 && threadIdx.x < 64) {
        shared_points[threadIdx.x] += shared_points[threadIdx.x + 64];
    }

    __syncthreads();

    if (BLOCK_SIZE >= 64 && threadIdx.x < 32) {
        shared_points[threadIdx.x] += shared_points[threadIdx.x + 32];
    }

    __syncthreads();
    if (threadIdx.x < 32) {
        shared_points[threadIdx.x] += shared_points[threadIdx.x + 16]; __syncwarp();
        shared_points[threadIdx.x] += shared_points[threadIdx.x + 8]; __syncwarp();
        shared_points[threadIdx.x] += shared_points[threadIdx.x + 4]; __syncwarp();
        shared_points[threadIdx.x] += shared_points[threadIdx.x + 2]; __syncwarp();
        shared_points[threadIdx.x] += shared_points[threadIdx.x + 1]; __syncwarp();
    }
    __syncthreads();
    if (tidx == 0) {
        vec_d[gidy * x_blocks_count + blockIdx.x] = shared_points[threadIdx.x];
        //printf("%f\n", vec_d[gidy * x_blocks_count + blockIdx.x]);
    }
}

__global__ void gpu_compute_jacobian(double * points_d, double * indexes_d, double * jacobian_d) {
    extern __shared__ double shared_data[];

    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    double result = 0.0;
    double f_minus = 0.0;
    double f_plus = 0.0;

    for (int ph = 0; ph < gridDim.x; ++ph) {
        int global_col = ph * blockDim.x + threadIdx.x;

        if (global_col < MATRIX_SIZE) {
            shared_data[threadIdx.x] = points_d[global_col];
            shared_data[blockDim.x + threadIdx.x] = indexes_d[row * MATRIX_SIZE + global_col];
        }
        else {
            shared_data[threadIdx.x] = 0.0;
            shared_data[blockDim.x + threadIdx.x] = 0.0;
        }

        __syncthreads();

        for (int i = 0; i < blockDim.x; ++i) {
            if (ph * blockDim.x + i >= MATRIX_SIZE) break;

            double value = shared_data[i];
            double element = shared_data[blockDim.x + i];

            if (ph * blockDim.x + i == col) {
                f_minus += (value - EQURENCY) * element;
                f_plus += (value + EQURENCY) * element;
            }
            else {
                f_minus += value * element;
                f_plus += value * element;
            }
        }

        __syncthreads();
    }

    result = (f_plus - f_minus) / (2 * EQURENCY);

    if (row < MATRIX_SIZE && col < MATRIX_SIZE) {
        jacobian_d[row * MATRIX_SIZE + col] = result;
    }
}

__global__ void normalizeRow(double* jacobian, double* inverse, int i, double pivot) {
    int j = threadIdx.x + blockDim.x * blockIdx.x;
    if (j < MATRIX_SIZE) {
        jacobian[i * MATRIX_SIZE + j] /= pivot;
        inverse[i * MATRIX_SIZE + j] /= pivot;
    }
}

__global__ void eliminateColumn(double* jacobian, double* inverse, int i) {
    int k = blockIdx.x;
    int j = threadIdx.x;

    if (k != i && j < MATRIX_SIZE) {
        double factor = jacobian[k * MATRIX_SIZE + i];
        jacobian[k * MATRIX_SIZE + j] -= jacobian[i * MATRIX_SIZE + j] * factor;
        inverse[k * MATRIX_SIZE + j] -= inverse[i * MATRIX_SIZE + j] * factor;
    }
}

__global__ void initIdentity(double* inverse) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < MATRIX_SIZE) {
        for (int j = 0; j < MATRIX_SIZE; ++j)
            inverse[i * MATRIX_SIZE + j] = (i == j) ? 1.0 : 0.0;
    }
}

void gpu_inverse(double* jacobian_d, double* inverse_jacobian_d) {
    int x_blocks_count = (MATRIX_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 blockDim(BLOCK_SIZE, 1, 1);
    dim3 gridDim(x_blocks_count, 1, 1);
    initIdentity << <gridDim, blockDim >> > (inverse_jacobian_d);

    for (int i = 0; i < MATRIX_SIZE; ++i) {
        double pivot;
        cudaMemcpy(&pivot, &jacobian_d[i * MATRIX_SIZE +i], sizeof(double), cudaMemcpyDeviceToHost);
        normalizeRow << <gridDim, blockDim >> > (jacobian_d, inverse_jacobian_d, i, pivot);
        eliminateColumn << <MATRIX_SIZE, MATRIX_SIZE >> > (jacobian_d, inverse_jacobian_d, i);
    }
}

void NewtonSolver::gpu_newton_solve() {
    std::cout << "GPU Newton solver\n";
    int x_blocks_count = (MATRIX_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int iterations_count = 0;
    double dx = 0;

    dim3 blockDim(BLOCK_SIZE, 1, 1);
    dim3 gridDim(x_blocks_count, MATRIX_SIZE, 1);

    double* delta = new double[MATRIX_SIZE];

#ifdef TOTAL_ELASPED_TIME
    auto start_total = std::chrono::high_resolution_clock::now();
#endif

    cudaMemcpy(data->points_d, data->points_h, MATRIX_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(data->indexes_d, data->indexes_h, MATRIX_SIZE * MATRIX_SIZE * sizeof(double), cudaMemcpyHostToDevice);

    do {
        iterations_count++;

#ifdef INTERMEDIATE_RESULTS
        auto start = std::chrono::high_resolution_clock::now();
#endif

        gpu_compute_func_and_delta_values << <gridDim, blockDim, 2 * blockDim.x * sizeof(double) >> > (
            data->points_d, data->indexes_d, data->intermediate_funcs_value_d);
        cudaDeviceSynchronize();

        cudaMemcpy(data->intermediate_funcs_value_h, data->intermediate_funcs_value_d, x_blocks_count * MATRIX_SIZE * sizeof(double), cudaMemcpyDeviceToHost);

        for (int i = 0; i < MATRIX_SIZE; i++) {
            data->funcs_value_h[i] = -data->vector_b_h[i];
            for (int j = 0; j < x_blocks_count; j++) {
                data->funcs_value_h[i] += data->intermediate_funcs_value_h[i * x_blocks_count + j];
            }
        }

#ifdef INTERMEDIATE_RESULTS
        auto end = std::chrono::high_resolution_clock::now();
        data->intermediate_results[0] = std::chrono::duration<double>(end - start).count();
        start = std::chrono::high_resolution_clock::now();
#endif

        gpu_compute_jacobian << <gridDim, blockDim, 2 * blockDim.x * sizeof(double) >> > (
            data->points_d, data->indexes_d, data->jacobian_d);
        cudaDeviceSynchronize();

        cudaMemcpy(data->jacobian_h, data->jacobian_d, MATRIX_SIZE * MATRIX_SIZE * sizeof(double), cudaMemcpyDeviceToHost);

#ifdef INTERMEDIATE_RESULTS
        end = std::chrono::high_resolution_clock::now();
        data->intermediate_results[1] = std::chrono::duration<double>(end - start).count();
        start = std::chrono::high_resolution_clock::now();
#endif
        //gpu_cublasInverse(data);
        //cudaMemcpy(data->inverse_jacobian_d, data->inverse_jacobian_h, MATRIX_SIZE * MATRIX_SIZE * sizeof(double), cudaMemcpyHostToDevice);
		gpu_inverse(data->jacobian_d, data->inverse_jacobian_d);
#ifdef INTERMEDIATE_RESULTS
        end = std::chrono::high_resolution_clock::now();
        data->intermediate_results[2] = std::chrono::duration<double>(end - start).count();
        start = std::chrono::high_resolution_clock::now();
#endif

        cudaMemcpy(data->funcs_value_d, data->funcs_value_h, MATRIX_SIZE * sizeof(double), cudaMemcpyHostToDevice);

        gpu_compute_func_and_delta_values << <gridDim, blockDim, 2 * blockDim.x * sizeof(double) >> > (
            data->funcs_value_d, data->inverse_jacobian_d, data->delta_d);
        cudaDeviceSynchronize();

        cudaMemcpy(data->delta_h, data->delta_d, x_blocks_count * MATRIX_SIZE * sizeof(double), cudaMemcpyDeviceToHost);

        for (int i = 0; i < MATRIX_SIZE; i++) {
            delta[i] = 0;
            for (int j = 0; j < x_blocks_count; j++) {
                delta[i] -= data->delta_h[i * x_blocks_count + j];
            }
        }

#ifdef INTERMEDIATE_RESULTS
        end = std::chrono::high_resolution_clock::now();
        data->intermediate_results[3] = std::chrono::duration<double>(end - start).count();
        start = std::chrono::high_resolution_clock::now();
#endif

        dx = 0.0;
        for (size_t i = 0; i < MATRIX_SIZE; ++i) {
            data->points_h[i] += delta[i];
            dx = std::max(dx, std::abs(delta[i]));
        }

        cudaMemcpy(data->points_d, data->points_h, MATRIX_SIZE * sizeof(double), cudaMemcpyHostToDevice);

#ifdef INTERMEDIATE_RESULTS
        end = std::chrono::high_resolution_clock::now();
        data->intermediate_results[4] = std::chrono::duration<double>(end - start).count();

        std::cout << "\nIteration: " << iterations_count << "\n";
        std::cout << "===============================================================\n";
        std::cout << "Intermediate results: \n";
        std::cout << "Compute func values: " << data->intermediate_results[0] << "s" << "\n";
        std::cout << "Compute jacobian: " << data->intermediate_results[1] << "s" << "\n";
        std::cout << "Compute inverse jacobian: " << data->intermediate_results[2] << "s" << "\n";
        std::cout << "Compute delta: " << data->intermediate_results[3] << "s" << "\n";
        std::cout << "Update points: " << data->intermediate_results[4] << "s" << "\n";
		std::cout << "Error (dx): " << dx << "\n";
        std::cout << "===============================================================\n";
#endif

    } while (dx > TOLERANCE);

#ifdef TOTAL_ELASPED_TIME
    auto end_total = std::chrono::high_resolution_clock::now();
    data->total_elapsed_time = std::chrono::duration<double>(end_total - start_total).count();
#endif

    print_solution(iterations_count, data->points_h);
    delete[] delta;
}

#endif