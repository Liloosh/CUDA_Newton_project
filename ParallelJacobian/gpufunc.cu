#include "NewtonSolver.h"
#include "stdio.h"
#include <iostream>

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

    if (BLOCK_SIZE >= 1024) {
        shared_points[threadIdx.x] += shared_points[threadIdx.x + 512];
    }

    __syncthreads();

    if (BLOCK_SIZE >= 512) {
        shared_points[threadIdx.x] += shared_points[threadIdx.x + 256];
    }

    __syncthreads();

    if (BLOCK_SIZE >= 256) {
        shared_points[threadIdx.x] += shared_points[threadIdx.x + 128];
    }

    __syncthreads();

    if (BLOCK_SIZE >= 128) {
        shared_points[threadIdx.x] += shared_points[threadIdx.x + 64];
    }

    __syncthreads();

    if (BLOCK_SIZE >= 64) {
        shared_points[threadIdx.x] += shared_points[threadIdx.x + 32];
    }

    __syncthreads();
    if (threadIdx.x < 32) {
        shared_points[threadIdx.x] += shared_points[threadIdx.x + 16];
        shared_points[threadIdx.x] += shared_points[threadIdx.x + 8];
        shared_points[threadIdx.x] += shared_points[threadIdx.x + 4];
        shared_points[threadIdx.x] += shared_points[threadIdx.x + 2];
        shared_points[threadIdx.x] += shared_points[threadIdx.x + 1];
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

void NewtonSolver::gpu_newton_solve() {
    int x_blocks_count = (MATRIX_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    double dx = 0;
    int iterations_count = 0;
    dim3 blockDim(BLOCK_SIZE, 1, 1);
    dim3 gridDim(x_blocks_count, MATRIX_SIZE, 1);

    double* delta = new double[MATRIX_SIZE];
    double* delta1 = new double[MATRIX_SIZE];
    do {
        iterations_count++;

        cudaMemcpy(data->points_d, data->points_h, MATRIX_SIZE * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(data->indexes_d, data->indexes_h, MATRIX_SIZE * MATRIX_SIZE * sizeof(double), cudaMemcpyHostToDevice);
        gpu_compute_func_and_delta_values << <gridDim, blockDim, 2 * blockDim.x * sizeof(double) >> > (data->points_d, data->indexes_d, data->vector_d);
        cudaDeviceSynchronize();
        cudaMemcpy(data->vector_h, data->vector_d, x_blocks_count * MATRIX_SIZE * sizeof(double), cudaMemcpyDeviceToHost);

        for (int i = 0; i < MATRIX_SIZE; i++) {
            data->vec_h[i] -= data->vector_b_h[i];
            for (int j = 0; j < x_blocks_count; j++) {
                data->vec_h[i] += data->vector_h[i * x_blocks_count + j];
            }
        }

        gpu_compute_jacobian << <gridDim, blockDim, 2 * blockDim.x * sizeof(double) >> > (data->points_d, data->indexes_d, data->jacobian_d);
        cudaMemcpy(data->jacobian_h, data->jacobian_d, MATRIX_SIZE * MATRIX_SIZE * sizeof(double), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        gpu_cublasInverse(data);

        cudaMemcpy(data->vec_d, data->vec_h, MATRIX_SIZE * sizeof(double), cudaMemcpyHostToDevice);
        gpu_compute_func_and_delta_values << <gridDim, blockDim, 2 * blockDim.x * sizeof(double) >> > (data->vec_d, data->inverse_jacobian_d, data->delta_d);
        cudaDeviceSynchronize();
        cudaMemcpy(data->delta_h, data->delta_d, x_blocks_count * MATRIX_SIZE * sizeof(double), cudaMemcpyDeviceToHost);

        for (int i = 0; i < MATRIX_SIZE; i++) {
            delta[i] = 0;
            for (int j = 0; j < x_blocks_count; j++) {
                delta[i] -= data->delta_h[i * x_blocks_count + j];
            }
        }

        dx = 0.0;

        for (size_t i = 0; i < MATRIX_SIZE; ++i) {
            if (iterations_count == 1) {
                delta1[i] = delta[i];
            }
            data->points_h[i] += delta[i];
            dx = std::max(dx, std::abs(delta[i]));
        }
    } while (dx > TOLERANCE);

    for (size_t i = 0; i < MATRIX_SIZE; ++i) {
        data->points_h[i] -= delta1[i];
    }

    print_solution(iterations_count, data->points_h);
}