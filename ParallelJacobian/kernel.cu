#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "iostream"
#include "functions.h"
#include "math.h"
#include "vector"
#include "cublas.h"

#define MATRIX_SIZE 100
#define EQURENCY 1e-7
#define TOLERANCE 1e-6
#define BLOCK_SIZE 64

struct cublasInverseData {
    cublasInverseData()
    {
        cudaMalloc((void**)&pivot, MATRIX_SIZE);
        cudaMalloc((void**)&ajacobian_d, sizeof(double*));
        cudaMalloc((void**)&ainverse_jacobian_d, sizeof(double*));
        cudaMalloc((void**)&info, sizeof(int));

        cublasCreate_v2(&cublasContextHandler);
        ready_ = true;
    }

    ~cublasInverseData()
    {
        if (ready_)
            cublasDestroy_v2(cublasContextHandler);

        cudaFree(info);
        cudaFree(ainverse_jacobian_d);
        cudaFree(ajacobian_d);
        cudaFree(pivot);
    }

    bool ready_{false};
    cublasHandle_t cublasContextHandler;
    int* pivot{nullptr};
    int* info{nullptr};
    double** ajacobian_d{nullptr};
    double** ainverse_jacobian_d{nullptr};
};

__host__ void cpy_computeVec(double* points, double* elements, double* vec) {
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            vec[i] += elements[i * MATRIX_SIZE + j] * points[j];
        }
    }
}

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

__host__ double cpy_compute_derivative(double* points, double* elements, int rowIndex, int colIndex, double equrency) {
    double temp_plus[MATRIX_SIZE], temp_minus[MATRIX_SIZE];

    for (int i = 0; i < MATRIX_SIZE; ++i) {
        temp_plus[i] = points[i];
        temp_minus[i] = points[i];
    }

    temp_plus[colIndex] += equrency;
    temp_minus[colIndex] -= equrency;

    double f_plus = 0.0, f_minus = 0.0;
    for (int j = 0; j < MATRIX_SIZE; ++j) {
        f_plus += elements[rowIndex * MATRIX_SIZE + j] * temp_plus[j];
        f_minus += elements[rowIndex * MATRIX_SIZE + j] * temp_minus[j];
    }

    return (f_plus - f_minus) / (2.0 * equrency);
}

__host__ void cpy_compute_jacobian(double* points, double* elements, double* jacobian) {
    for (int i = 0; i < MATRIX_SIZE; ++i) {
        for (int j = 0; j < MATRIX_SIZE; ++j) {
            jacobian[i * MATRIX_SIZE + j] = cpy_compute_derivative(points, elements, i, j, EQURENCY);
        }
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

__host__ void cpy_inverse(double* a, double* y, int n) {
    for (int i = 0; i < n; i++) y[i * n + i] = 1.0;

    for (int i = 0; i < n; i++) {
        double temp = a[i * n + i];
        for (int j = 0; j < n; j++) {
            a[i * n + j] /= temp;
            y[i * n + j] /= temp;
        }
        for (int k = 0; k < n; k++) {
            if (k != i) {
                temp = a[k * n + i];
                for (int j = 0; j < n; j++) {
                    a[k * n + j] -= a[i * n + j] * temp;
                    y[k * n + j] -= y[i * n + j] * temp;
                }
            }
        }
    }
}

void cublasInverse(double* jacobian, double* inverse_jacobian_d, double* inverse_jacobian_h) {
    cublasInverseData d;

    cudaMemcpy(d.ajacobian_d, &jacobian, sizeof(double*), cudaMemcpyHostToDevice);
    cudaMemcpy(d.ainverse_jacobian_d, &inverse_jacobian_d, sizeof(double*), cudaMemcpyHostToDevice);



    cublasDgetrfBatched(d.cublasContextHandler, MATRIX_SIZE, d.ajacobian_d, MATRIX_SIZE, d.pivot, d.info, 1);
    cublasDgetriBatched(d.cublasContextHandler, MATRIX_SIZE, (const double**)d.ajacobian_d,
                        MATRIX_SIZE, d.pivot, d.ainverse_jacobian_d, MATRIX_SIZE, d.info, 1);

    cudaMemcpy(inverse_jacobian_h, inverse_jacobian_d, MATRIX_SIZE * MATRIX_SIZE * sizeof(double), cudaMemcpyDeviceToHost);    
}

__host__ void cpy_computeDelta(double* inv_jacobian, double* vec, double* delta) {
    for (int i = 0; i < MATRIX_SIZE; i++) {
        delta[i] = 0.0;
        for (int j = 0; j < MATRIX_SIZE; j++) {
            delta[i] -= inv_jacobian[i * MATRIX_SIZE + j] * vec[j];
        }
    }
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

void cpy_Newton(double* points, double* elements, double& dx) {

    //
    // VECTOR
    //

    double* vec = new double[MATRIX_SIZE];
    for (int i = 0; i < MATRIX_SIZE; i++) {
        vec[i] = 0;
    }
    cpy_computeVec(points, elements, vec);

    //std::cout << "-----------------" << std::endl;
    //std::cout << "Vector" << std::endl;
    //for (int i = 0; i < MATRIX_SIZE; i++) {
    //    std::cout << vec[i] << std::endl;
    //}

    //
    // JACOBIAN
    //

    double* jacobian = new double[MATRIX_SIZE * MATRIX_SIZE];
    cpy_compute_jacobian(points, elements, jacobian);

    //std::cout << "Jacobian" << std::endl;
    //for (int i = 0; i < MATRIX_SIZE; i++) {
    //    for (int j = 0; j < MATRIX_SIZE; j++) {
    //        std::cout << jacobian[i * MATRIX_SIZE + j] << " ";
    //    }
    //    std::cout << std::endl;
    //}

    //
    // INVERSE
    //

    double* inverse_jacobian = new double[MATRIX_SIZE * MATRIX_SIZE];
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            inverse_jacobian[i * MATRIX_SIZE + j] = 0.0;
        }
    }
    cpy_inverse(jacobian, inverse_jacobian, MATRIX_SIZE);


    //std::cout << "Inverse Jacobian" << std::endl;
    //for (int i = 0; i < MATRIX_SIZE; i++) {
    //    for (int j = 0; j < MATRIX_SIZE; j++) {
    //        std::cout << inverse_jacobian[i * MATRIX_SIZE + j] << " ";
    //    }
    //    std::cout << std::endl;
    //}

    //
    // DELTA
    //

    double* delta = new double[MATRIX_SIZE];
    cpy_computeDelta(inverse_jacobian, vec, delta);

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

    delete[] vec;
    delete[] jacobian;
    delete[] inverse_jacobian;
    delete[] delta;
    delete[] last_point;
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

    delete[] inverse_jacobian;
    delete[] jacobian_h;
    delete[] last_point;
    delete[] delta;
    delete[] delta_h;
    delete[] vector_h;
    delete[] vec;
}

int main() {
    double* elements_h = new double[MATRIX_SIZE * MATRIX_SIZE];
    double* jacobian_h = new double[MATRIX_SIZE * MATRIX_SIZE];
    double* points_h = new double[MATRIX_SIZE];
    double* points_h1 = new double[MATRIX_SIZE];

    for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; i++) {
        elements_h[i] = elements[i];
        jacobian_h[i] = 0;
    }

    for (int i = 0; i < MATRIX_SIZE; i++) {
        points_h[i] = 10.0;
    }

    /*
    //
    //   JACOBIAN
    //

    float duration = 0.0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    cpy_compute_jacobian(points_h, elements_h, jacobian_h);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&duration, start, stop);

    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            std::cout << jacobian_h[i * MATRIX_SIZE + j] << " ";
        }
        std::cout << std::endl;
    }

    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            jacobian_h[i * MATRIX_SIZE + j] = 0;
        }
    }

    std::cout << "Time: " << duration << " ms" << std::endl;
    std::cout << std::endl;

    cudaMalloc((void**)&elements_d, MATRIX_SIZE * MATRIX_SIZE * sizeof(double));
    cudaMalloc((void**)&jacobian_d, MATRIX_SIZE * MATRIX_SIZE * sizeof(double));
    cudaMalloc((void**)&points_d, MATRIX_SIZE * MATRIX_SIZE * sizeof(double));

    cudaMemcpy(elements_d, elements_h, MATRIX_SIZE * MATRIX_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(points_d, points_h, MATRIX_SIZE * MATRIX_SIZE * sizeof(double), cudaMemcpyHostToDevice);

    cudaEventRecord(start);
    dim3 blockDim(32, 1, 1);
    int x_blocks_count = (MATRIX_SIZE + blockDim.x - 1) / blockDim.x;
    dim3 gridDim(x_blocks_count, MATRIX_SIZE, 1);
    printf("%d, %d \n", gridDim.x, gridDim.y);
    compute_jacobian << <gridDim, blockDim, 2 * blockDim.x * sizeof(double) >> > (points_d, elements_d, jacobian_d);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);

    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            jacobian_h[i * MATRIX_SIZE + j] = 0;
        }
    }


    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            std::cout << jacobian_h[i * MATRIX_SIZE + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    cudaMemcpy(jacobian_h, jacobian_d, MATRIX_SIZE * MATRIX_SIZE * sizeof(double), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&duration, start, stop);

    std::cout << "Time: " << duration << " ms" << std::endl;

    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            std::cout << jacobian_h[i * MATRIX_SIZE + j] << " ";
        }
        std::cout << std::endl;
    }

    //
    // INVERSION
    //

    double* inverse_jacobian_h = new double[MATRIX_SIZE * MATRIX_SIZE];
    double* inverse_jacobian_d;

    cudaMalloc((void**)&inverse_jacobian_d, MATRIX_SIZE* MATRIX_SIZE * sizeof(double));
    cudaMemcpy(inverse_jacobian_h, inverse_jacobian_d, MATRIX_SIZE* MATRIX_SIZE * sizeof(double), cudaMemcpyDeviceToHost);

    cudaEventRecord(start);
    cpy_inverse(jacobian_h, inverse_jacobian_h, MATRIX_SIZE);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&duration, start, stop);

    std::cout << std::endl;

    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            std::cout << inverse_jacobian_h[i * MATRIX_SIZE + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "Time: " << duration << std::endl;
    std::cout << std::endl;

    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            inverse_jacobian_h[i * MATRIX_SIZE + j] = 0;
        }
    }
    
    cudaEventRecord(start);
    cublasInverse(jacobian_d, inverse_jacobian_d, inverse_jacobian_h);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&duration, start, stop);

    std::cout << std::endl;

    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            std::cout << inverse_jacobian_h[i * MATRIX_SIZE + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "Time: " << duration << std::endl;

    //
    // VECTOR
    //

    double* vec = new double[MATRIX_SIZE];

    for (int i = 0; i < MATRIX_SIZE; i++) {
        vec[i] = 0;
    }

    cpy_computeVec(points_h, elements_h, vec);

    for (int i = 0; i < MATRIX_SIZE; i++) {
        std::cout << vec[i] << std::endl;
    }

    for (int i = 0; i < MATRIX_SIZE; i++) {
        vec[i] = 0;
    }


    double* vector_h = new double[x_blocks_count * MATRIX_SIZE];

    double* vector_d;
    cudaMalloc((void**)&vector_d, x_blocks_count * MATRIX_SIZE * sizeof(double));
    cudaMemcpy(points_d, points_h, MATRIX_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(elements_d, elements_h, MATRIX_SIZE * MATRIX_SIZE * sizeof(double), cudaMemcpyHostToDevice);

    computeVec<<<gridDim, blockDim>>>(points_d, elements_d, vector_d, x_blocks_count);

    cudaMemcpy(vector_h, vector_d, x_blocks_count * MATRIX_SIZE * sizeof(double), cudaMemcpyDeviceToHost);

    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < x_blocks_count; j++) {
            vec[i] += vector_h[i * x_blocks_count + j];
        }
    }

    std::cout << "vector:" << std::endl;
    for (int i = 0; i < MATRIX_SIZE; i++) {
        std::cout << vec[i] << std::endl;
    }

    //
    // DELTA
    //

    double* delta = new double[MATRIX_SIZE];

    cpy_computeDelta(inverse_jacobian_h, vec, delta);
    std::cout << "delta:" << std::endl;
    for (int i = 0; i < MATRIX_SIZE; i++) {
        std::cout << delta[i] << std::endl;
    }

    for (int i = 0; i < MATRIX_SIZE; i++) {
        delta[i] = 0;
    }

    double* delta_h = new double[MATRIX_SIZE * x_blocks_count];
    double* delta_d;
    double* vec_d;

    cudaMalloc((void**)&delta_d, x_blocks_count * MATRIX_SIZE * sizeof(double));
    cudaMalloc((void**)&vec_d, MATRIX_SIZE * sizeof(double));

    cudaMemcpy(vec_d, vec, MATRIX_SIZE * sizeof(double), cudaMemcpyHostToDevice);

    computeDelta<<<gridDim, blockDim>>>(inverse_jacobian_d, vec_d, delta_d, x_blocks_count);
    cudaMemcpy(delta_h, delta_d, x_blocks_count * MATRIX_SIZE * sizeof(double), cudaMemcpyDeviceToHost);

    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < x_blocks_count; j++) {
            delta[i] -= delta_h[i * x_blocks_count + j];
        }
    }

    std::cout << "delta:" << std::endl;
    for (int i = 0; i < MATRIX_SIZE; i++) {
        std::cout << delta[i] << std::endl;
    }*/

    //
    // NEWTON
    //

    //
    // CPY
    //

    std::cout << "\nCPU" << std::endl;

    double dx = 0.0;
    int i = 0;
    float duration = 0.0f;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    do {
        i++;
        cpy_Newton(points_h, elements_h, dx);
    } while (dx > TOLERANCE);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&duration, start, stop);

    std::cout << "Iterations: " << i << std::endl;
    std::cout << "\nSolution:" << std::endl;
    for (int i = 0; i < MATRIX_SIZE; i++) {
        std::cout << points_h[i] << std::endl;
    }

    std::cout << "Res check" << std::endl;
    double res = 0.0;
    for (int i = 0; i < MATRIX_SIZE; i++) {
        res = 0.0;
        for (int j = 0; j < MATRIX_SIZE; j++) {
            res += points_h[j] * elements_h[i * MATRIX_SIZE + j];
        }
        std::cout << res << std::endl;
    }
    std::cout << "Time: " << duration << " ms\n" << std::endl;

    //
    // GPU
    //

    std::cout << "\nGPU" << std::endl;
    int x_blocks_count = (MATRIX_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;

    double* elements_d, * points_d, * vector_d, * jacobian_d, * inverse_jacobian_d, * delta_d, * vec_d;

    cudaMalloc((void**)&points_d, MATRIX_SIZE * sizeof(double));
    cudaMalloc((void**)&elements_d, MATRIX_SIZE * MATRIX_SIZE * sizeof(double));
    cudaMalloc((void**)&vector_d, x_blocks_count * MATRIX_SIZE * sizeof(double));
    cudaMalloc((void**)&jacobian_d, MATRIX_SIZE * MATRIX_SIZE * sizeof(double));
    cudaMalloc((void**)&inverse_jacobian_d, MATRIX_SIZE * MATRIX_SIZE * sizeof(double));
    cudaMalloc((void**)&delta_d, x_blocks_count * MATRIX_SIZE * sizeof(double));
    cudaMalloc((void**)&vec_d, MATRIX_SIZE * sizeof(double));


    for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; i++) {
        elements_h[i] = elements[i];
    }

    for (int i = 0; i < MATRIX_SIZE; i++) {
        points_h1[i] = 10.0;
    }

    dx = 0.0;
    i = 0;
    duration = 0.0f;
    cudaEventRecord(start);
    do {
        i++;
        gpy_Newton(points_h1, elements_h, dx, points_d, elements_d, vector_d, jacobian_d, inverse_jacobian_d, delta_d, vec_d);
    } while (dx > TOLERANCE);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&duration, start, stop);

    cudaFree(points_d);
    cudaFree(elements_d);
    cudaFree(vector_d);
    cudaFree(jacobian_d);
    cudaFree(inverse_jacobian_d);
    cudaFree(delta_d);
    cudaFree(vec_d);

    std::cout << "Iterations: " << i << std::endl;
    std::cout << "\nSolution:" << std::endl;
    for (int i = 0; i < MATRIX_SIZE; i++) {
        std::cout << points_h1[i] << std::endl;
    }

    std::cout << "Res check" << std::endl;
    res = 0.0;
    for (int i = 0; i < MATRIX_SIZE; i++) {
        res = 0.0;
        for (int j = 0; j < MATRIX_SIZE; j++) {
            res += points_h1[j] * elements_h[i * MATRIX_SIZE + j];
        }
        std::cout << res << std::endl;
    }
    std::cout << "Time: " << duration << " ms" << std::endl;

    bool check = true;
    for (int i = 0; i < MATRIX_SIZE; i++) {
        if ((int)points_h[i] != (int)points_h1[i]) {
            std::cout << i << std::endl;
            check = false;
            break;
        }
    }

    delete[] elements_h;
    delete[] jacobian_h;
    delete[] points_h;
    delete[] points_h1;

    std::cout << "Points are the same: " << check << std::endl;
    std::cin.get();
    return 0;
}
