#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "iostream"
#include "math.h"
#include "vector"
#include "cublas.h"
#include "DataInitializer.h"
#include "NewtonSolver.h"
#include "memory"

__global__ void computeVec(double* points, double* elements, double* vec, int block_count) {
    int gidx = blockDim.x * blockIdx.x + threadIdx.x;
    int gidy = blockDim.y * blockIdx.y + threadIdx.y;
    int tidx = threadIdx.x;

    extern __shared__ double shared_points[];

    if (gidx < MATRIX_SIZE) {
        shared_points[threadIdx.x] = points[gidx];
        shared_points[threadIdx.x + blockDim.x] = elements[gidy * MATRIX_SIZE + gidx];
        //printf("points: %f %f\n", shared_points[threadIdx.x], shared_elements[threadIdx.x]);
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
        vec[gidy * block_count + blockIdx.x] = shared_points[threadIdx.x];
    }
}

__global__ void compute_jacobian(double* points, double* elements, double* jacobian) {
    extern __shared__ double shared_data[];

    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    double result = 0.0;
    double f_minus = 0.0;
    double f_plus = 0.0;

    for (int ph = 0; ph < gridDim.x; ++ph) {
        int global_col = ph * blockDim.x + threadIdx.x;

        if (global_col < MATRIX_SIZE) {
            shared_data[threadIdx.x] = points[global_col];
            shared_data[blockDim.x + threadIdx.x] = elements[row * MATRIX_SIZE + global_col];
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
        jacobian[row * MATRIX_SIZE + col] = result;
    }
}

void cublasInverse(double* jacobian, double* inverse_jacobian_d, double* inverse_jacobian_h) {
    int* pivot;
    int* info;
    double** ajacobian_d;
    double** ainverse_jacobian_d;
    cudaMalloc((void**)&pivot, MATRIX_SIZE);
    cudaMalloc((void**)&ajacobian_d, sizeof(double*));
    cudaMalloc((void**)&ainverse_jacobian_d, sizeof(double*));
    cudaMalloc((void**)&info, sizeof(int));

    cudaMemcpy(ajacobian_d, &jacobian, sizeof(double*), cudaMemcpyHostToDevice);
    cudaMemcpy(ainverse_jacobian_d, &inverse_jacobian_d, sizeof(double*), cudaMemcpyHostToDevice);

    cublasHandle_t cublasContextHandler;
    cublasCreate_v2(&cublasContextHandler);

    cublasDgetrfBatched(cublasContextHandler, MATRIX_SIZE, ajacobian_d, MATRIX_SIZE, pivot, info, 1);
    cublasDgetriBatched(cublasContextHandler, MATRIX_SIZE, (const double**)ajacobian_d, MATRIX_SIZE, pivot, ainverse_jacobian_d, MATRIX_SIZE, info, 1);

    cudaMemcpy(inverse_jacobian_h, inverse_jacobian_d, MATRIX_SIZE * MATRIX_SIZE * sizeof(double), cudaMemcpyDeviceToHost);
}

__global__ void computeDelta(double* inv_jacobian, double* vec, double* delta, int block_count) {
    int gidx = blockDim.x * blockIdx.x + threadIdx.x;
    int gidy = blockDim.y * blockIdx.y + threadIdx.y;
    int tidx = threadIdx.x;

    extern __shared__ double shared_points[];

    if (gidx < MATRIX_SIZE) {
        shared_points[threadIdx.x] = inv_jacobian[gidy * MATRIX_SIZE + gidx];
        shared_points[threadIdx.x + blockDim.x] = vec[gidx];
        //printf("points: %f %f\n", shared_points[threadIdx.x], shared_elements[threadIdx.x]);
    }
    else {
        shared_points[threadIdx.x] = 0;
        shared_points[threadIdx.x + blockDim.x] = 0;
    }
    __syncthreads();

    shared_points[tidx] *= shared_points[tidx + blockDim.x];

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

    if (tidx == 0) {
        delta[gidy * block_count + blockIdx.x] = shared_points[threadIdx.x];
    }
}

void gpy_Newton(double* points, double* elements, double& dx, double* points_d, double* elements_d, double* vector_d, double* jacobian_d, double* inverse_jacobian_d, double* delta_d, double* vec_d) {
    dim3 blockDim(BLOCK_SIZE, 1, 1);
    int x_blocks_count = (MATRIX_SIZE + blockDim.x - 1) / blockDim.x;
    dim3 gridDim(x_blocks_count, MATRIX_SIZE, 1);

    //
    // VECTOR
    //

    double* vector_h = new double[x_blocks_count * MATRIX_SIZE];

    double* vec = new double[MATRIX_SIZE];

    for (int i = 0; i < MATRIX_SIZE; i++) {
        vec[i] = 0;
    }

    cudaMemcpy(points_d, points, MATRIX_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(elements_d, elements, MATRIX_SIZE * MATRIX_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    computeVec << <gridDim, blockDim, 2 * blockDim.x * sizeof(double) >> > (points_d, elements_d, vector_d, x_blocks_count);
    cudaDeviceSynchronize();
    cudaMemcpy(vector_h, vector_d, x_blocks_count * MATRIX_SIZE * sizeof(double), cudaMemcpyDeviceToHost);

    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < x_blocks_count; j++) {
            vec[i] += vector_h[i * x_blocks_count + j];
        }
    }

    //std::cout << "vector:" << std::endl;
    //for (int i = 0; i < MATRIX_SIZE; i++) {
    //    std::cout << vec[i] << std::endl;
    //}

    //
    // JACOBIAN
    //

    double* jacobian_h = new double[MATRIX_SIZE * MATRIX_SIZE];
    compute_jacobian << <gridDim, blockDim, 2 * blockDim.x * sizeof(double) >> > (points_d, elements_d, jacobian_d);
    cudaMemcpy(jacobian_h, jacobian_d, MATRIX_SIZE * MATRIX_SIZE * sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    //for (int i = 0; i < MATRIX_SIZE; i++) {
    //    for (int j = 0; j < MATRIX_SIZE; j++) {
    //        std::cout << jacobian_h[i * MATRIX_SIZE + j] << " ";
    //    }
    //    std::cout << std::endl;
    //}

    //
    // INVERSE JACOBIAN
    //
    double* inverse_jacobian = new double[MATRIX_SIZE * MATRIX_SIZE];
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            inverse_jacobian[i * MATRIX_SIZE + j] = 0.0;
        }
    }
    cublasInverse(jacobian_d, inverse_jacobian_d, inverse_jacobian);
    //cudaMemcpy(inverse_jacobian_d, inverse_jacobian, MATRIX_SIZE * MATRIX_SIZE * sizeof(double), cudaMemcpyHostToDevice);

    //for (int i = 0; i < MATRIX_SIZE; i++) {
    //    for (int j = 0; j < MATRIX_SIZE; j++) {
    //        std::cout << inverse_jacobian_h[i * MATRIX_SIZE + j] << " ";
    //    }
    //    std::cout << std::endl;
    //}

    //
    // DELTA
    //

    double* delta = new double[MATRIX_SIZE];
    double* delta_h = new double[MATRIX_SIZE * x_blocks_count];

    for (int i = 0; i < MATRIX_SIZE; i++) {
        delta[i] = 0;
    }

    cudaMemcpy(vec_d, vec, MATRIX_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    computeDelta << <gridDim, blockDim, 2 * blockDim.x * sizeof(double) >> > (inverse_jacobian_d, vec_d, delta_d, x_blocks_count);
    cudaMemcpy(delta_h, delta_d, x_blocks_count * MATRIX_SIZE * sizeof(double), cudaMemcpyDeviceToHost);

    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < x_blocks_count; j++) {
            delta[i] -= delta_h[i * x_blocks_count + j];
        }
    }

    //std::cout << "delta:" << std::endl;
    //for (int i = 0; i < MATRIX_SIZE; i++) {
    //    std::cout << delta[i] << std::endl;
    //}

    //
    // ADD DELTA
    //

    double* last_point = new double[MATRIX_SIZE];
    for (size_t i = 0; i < MATRIX_SIZE; ++i) {
        last_point[i] = points[i];
        points[i] += delta[i];
    }

    //
    // DX
    //

    dx = 0.0;
    for (size_t i = 0; i < MATRIX_SIZE; ++i) {
        dx = std::max(dx, std::abs(points[i] - last_point[i]));
    }
    cudaThreadSynchronize();
}

int main() {
    //
    // CPY
    //

    std::unique_ptr<DataInitializer> data = std::make_unique<DataInitializer>();
    std::unique_ptr<NewtonSolver> newton_solver = std::make_unique<NewtonSolver>(data.get());
    newton_solver->cpu_newton_solve();

    //
    // GPU
    //

    std::unique_ptr<DataInitializer> data2 = std::make_unique<DataInitializer>();
    std::unique_ptr<NewtonSolver> newton_solver2 = std::make_unique<NewtonSolver>(data2.get());
    newton_solver2->gpu_newton_solve();

    return 0;
}