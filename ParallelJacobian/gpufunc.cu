#include "NewtonSolver.h"
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

void NewtonSolver::gpu_newton_solve() {
	std::cout << "GPU Newton solver" << "\n";
    int x_blocks_count = (MATRIX_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    double dx = 0;
    int iterations_count = 0;
    dim3 blockDim(BLOCK_SIZE, 1, 1);
    dim3 gridDim(x_blocks_count, MATRIX_SIZE, 1);

    double* delta = new double[MATRIX_SIZE];
    double* delta1 = new double[MATRIX_SIZE];

#ifdef TOTAL_ELASPED_TIME
    auto start_total = std::chrono::high_resolution_clock::now();
#endif
    do {
        iterations_count++;

#ifdef INTERMEDIATE_RESULTS
        auto start = std::chrono::high_resolution_clock::now();
#endif
#ifdef COPY_ACTION
		auto start_copy = std::chrono::high_resolution_clock::now();
#endif
        cudaMemcpy(data->points_d, data->points_h, MATRIX_SIZE * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(data->indexes_d, data->indexes_h, MATRIX_SIZE * MATRIX_SIZE * sizeof(double), cudaMemcpyHostToDevice);
#ifdef COPY_ACTION
		auto end_copy = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed_copy = end_copy - start_copy;
		std::cout << "\nCopy points and indexes to device: " << elapsed_copy.count() << "\n";
#endif
        gpu_compute_func_and_delta_values << <gridDim, blockDim, 2 * blockDim.x * sizeof(double) >> > (data->points_d, data->indexes_d, data->vector_d);
        cudaDeviceSynchronize();
#ifdef COPY_ACTION
        start_copy = std::chrono::high_resolution_clock::now();
#endif
        cudaMemcpy(data->vector_h, data->vector_d, x_blocks_count * MATRIX_SIZE * sizeof(double), cudaMemcpyDeviceToHost);
#ifdef COPY_ACTION
		end_copy = std::chrono::high_resolution_clock::now();
		elapsed_copy = end_copy - start_copy;
		std::cout << "\nCopy vector to host: " << elapsed_copy.count() << "\n";
#endif

        for (int i = 0; i < MATRIX_SIZE; i++) {
            data->vec_h[i] -= data->vector_b_h[i];
            for (int j = 0; j < x_blocks_count; j++) {
                data->vec_h[i] += data->vector_h[i * x_blocks_count + j];
            }
        }
#ifdef INTERMEDIATE_RESULTS
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        data->intermediate_results[0] = elapsed.count();
#endif

#ifdef INTERMEDIATE_RESULTS
        start = std::chrono::high_resolution_clock::now();
#endif
        gpu_compute_jacobian << <gridDim, blockDim, 2 * blockDim.x * sizeof(double) >> > (data->points_d, data->indexes_d, data->jacobian_d);
#ifdef COPY_ACTION
		start_copy = std::chrono::high_resolution_clock::now();
#endif
        cudaMemcpy(data->jacobian_h, data->jacobian_d, MATRIX_SIZE * MATRIX_SIZE * sizeof(double), cudaMemcpyDeviceToHost);
#ifdef COPY_ACTION
		end_copy = std::chrono::high_resolution_clock::now();
		elapsed_copy = end_copy - start_copy;
		std::cout << "\nCopy jacobian to host: " << elapsed_copy.count() << "\n";
#endif
        cudaDeviceSynchronize();
#ifdef INTERMEDIATE_RESULTS
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        data->intermediate_results[1] = elapsed.count();
#endif

#ifdef INTERMEDIATE_RESULTS
        start = std::chrono::high_resolution_clock::now();
#endif
        gpu_cublasInverse(data);
#ifdef INTERMEDIATE_RESULTS
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        data->intermediate_results[2] = elapsed.count();
#endif

#ifdef INTERMEDIATE_RESULTS
        start = std::chrono::high_resolution_clock::now();
#endif
#ifdef COPY_ACTION
		start_copy = std::chrono::high_resolution_clock::now();
#endif
        cudaMemcpy(data->vec_d, data->vec_h, MATRIX_SIZE * sizeof(double), cudaMemcpyHostToDevice);
#ifdef COPY_ACTION
		end_copy = std::chrono::high_resolution_clock::now();
		elapsed_copy = end_copy - start_copy;
		std::cout << "\nCopy vec to device: " << elapsed_copy.count() << "\n";
#endif
        gpu_compute_func_and_delta_values << <gridDim, blockDim, 2 * blockDim.x * sizeof(double) >> > (data->vec_d, data->inverse_jacobian_d, data->delta_d);
        cudaDeviceSynchronize();
#ifdef COPY_ACTION
		start_copy = std::chrono::high_resolution_clock::now();
#endif
        cudaMemcpy(data->delta_h, data->delta_d, x_blocks_count * MATRIX_SIZE * sizeof(double), cudaMemcpyDeviceToHost);
#ifdef COPY_ACTION
		end_copy = std::chrono::high_resolution_clock::now();
		elapsed_copy = end_copy - start_copy;
		std::cout << "\nCopy delta to host: " << elapsed_copy.count() << "\n";
#endif

        for (int i = 0; i < MATRIX_SIZE; i++) {
            delta[i] = 0;
            for (int j = 0; j < x_blocks_count; j++) {
                delta[i] -= data->delta_h[i * x_blocks_count + j];
            }
        }
#ifdef INTERMEDIATE_RESULTS
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        data->intermediate_results[3] = elapsed.count();
#endif

#ifdef INTERMEDIATE_RESULTS
        start = std::chrono::high_resolution_clock::now();
#endif
        dx = 0.0;

        for (size_t i = 0; i < MATRIX_SIZE; ++i) {
            if (iterations_count == 1) {
                delta1[i] = delta[i];
            }
            data->points_h[i] += delta[i];
            dx = std::max(dx, std::abs(delta[i]));
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
        std::cout << "===============================================================\n";
#endif
    } while (dx > TOLERANCE);
#ifdef TOTAL_ELASPED_TIME
    auto end_total = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_total = end_total - start_total;
    data->total_elapsed_time = elapsed_total.count();
#endif

    for (size_t i = 0; i < MATRIX_SIZE; ++i) {
        data->points_h[i] -= delta1[i];
    }


    print_solution(iterations_count, data->points_h);
}
#endif