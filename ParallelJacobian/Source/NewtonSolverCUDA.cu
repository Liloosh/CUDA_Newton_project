#include "stdio.h"
#include <iostream>
#include <memory>
#include <string>
#include "cuda_runtime.h"
#include "FileOperations.h"
#include "NewtonSolverGPUFunctions.h"
#include <EditionalTools.h>
#include "NewtonSolverCUDA.h"
#include <chrono>

NewtonSolverCUDA::NewtonSolverCUDA(DataInitializerCUDA* dataInitializer,
        const Settings::SettingsData& settings, SystemInfo& sinfo)
    : settings_{settings}
    , sinfo_{sinfo}
{
	data = dataInitializer;
}

NewtonSolverCUDA::~NewtonSolverCUDA() {
}
void NewtonSolverCUDA::gpu_cublas_solve(DataInitializerCUDA* data) {
    cublasStatus_t s1 = cublasDgetrfBatched(
                data->cublasContextHandler, data->MATRIX_SIZE,
                data->cublas_ajacobian_d, data->MATRIX_SIZE,
                nullptr, //data->cublas_pivot,
                data->cublas_info, 1);

    const double alpha = 1.0;
    cublasStatus_t s2 = cublasDtrsm(
                data->cublasContextHandler, CUBLAS_SIDE_LEFT,
                CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT,
                data->MATRIX_SIZE, 1, &alpha,
                data->jacobian_d, data->MATRIX_SIZE,
                data->funcs_value_d, data->MATRIX_SIZE);

    cublasStatus_t s3 = cublasDtrsm(
                data->cublasContextHandler, CUBLAS_SIDE_LEFT,
                CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                data->MATRIX_SIZE, 1, &alpha,
                data->jacobian_d, data->MATRIX_SIZE,
                data->funcs_value_d, data->MATRIX_SIZE);

    std::cout << "TRF/TRS status: " << s1 << " " << s2 << " " << s3 << std::endl;
}

void NewtonSolverCUDA::gpu_cusolver_solve(DataInitializerCUDA* data) {
    int workspace_size;
    int n = data->MATRIX_SIZE;
    cusolverStatus_t s1 = cusolverDnDgetrf_bufferSize(
                data->cusolverH, n, n, data->jacobian_d, n, &workspace_size);

    if (workspace_size > data->workspace_size) {
        if (data->workspace_d) {
            cudaFree(data->workspace_d);
        }
        cudaMalloc(reinterpret_cast<void **>(&data->workspace_d),
                   sizeof(double) * workspace_size);
        data->workspace_size = workspace_size;
    }

    cusolverStatus_t s2 = cusolverDnDgetrf(
                data->cusolverH, n, n, data->jacobian_d, n, data->workspace_d,
                data->cusolver_pivot, data->cublas_info);

    cusolverStatus_t s3 = cusolverDnDgetrs(
                data->cusolverH, CUBLAS_OP_N, n, 1, data->jacobian_d, n,
                data->cusolver_pivot, data->funcs_value_d, n,
                data->cublas_info);

    std::cout << "cuSOLVER status: " << s1 << " " << s2 << " " << s3 << std::endl;
}

void NewtonSolverCUDA::gpu_newton_solve() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int version = prop.major;
    std::unique_ptr<FileOperations> file_op = std::make_unique<FileOperations>(settings_.path);
    std::string file_name = data->is_cublas ? "gpu_cublas_" : "gpu_cusolver_";
    file_name = file_name + std::to_string(data->file_name) + ".csv";
    file_op->create_file(file_name, 4);
    file_op->append_file_headers(data->csv_header);

    NewtonSolverGPUFunctions::gpu_dummy_warmup << <1, 32 >> > ();
    cudaDeviceSynchronize();
    std::cout << (data->is_cublas ? "GPU cuBLAS\n" : "GPU cuSOLVER\n");
    int x_blocks_count = (data->MATRIX_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int iterations_count = 0;
    double dx = 0;

    dim3 blockDim(BLOCK_SIZE, 1, 1);
    dim3 gridDim(x_blocks_count, data->MATRIX_SIZE, 1);

    double* delta = new double[data->MATRIX_SIZE];

	auto start_total = std::chrono::steady_clock::now();

    cudaMemcpy(data->points_d, data->points_h, data->MATRIX_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(data->indexes_d, data->indexes_h, data->MATRIX_SIZE * data->MATRIX_SIZE * sizeof(double), cudaMemcpyHostToDevice);

    cudaStream_t myStream;
    cudaStreamCreate(&myStream);

    do {
        iterations_count++;

#ifdef INTERMEDIATE_RESULTS
		auto start = std::chrono::steady_clock::now();
#endif
		std::cout << "Power: " << data->equation->get_power() << "\n";
        NewtonSolverGPUFunctions::gpu_compute_func_values << <gridDim, blockDim, blockDim.x * sizeof(double) >> > (
            data->points_d, data->indexes_d, data->intermediate_funcs_value_d, data->MATRIX_SIZE, version, data->equation->get_power());
        cudaDeviceSynchronize();

        cudaMemcpy(data->intermediate_funcs_value_h, data->intermediate_funcs_value_d, x_blocks_count * data->MATRIX_SIZE * sizeof(double), cudaMemcpyDeviceToHost);

        for (int i = 0; i < data->MATRIX_SIZE; i++) {
            data->funcs_value_h[i] = -data->vector_b_h[i];
            for (int j = 0; j < x_blocks_count; j++) {
                data->funcs_value_h[i] += data->intermediate_funcs_value_h[i * x_blocks_count + j];
            }
        }
        cudaMemcpy(data->funcs_value_d, data->funcs_value_h, data->MATRIX_SIZE * sizeof(double), cudaMemcpyHostToDevice);

#ifdef INTERMEDIATE_RESULTS
		auto end = std::chrono::steady_clock::now();
        data->intermediate_results[0] = std::chrono::duration<double>(end - start).count();
		start = std::chrono::steady_clock::now();
#endif

        NewtonSolverGPUFunctions::gpu_compute_jacobian << <gridDim, blockDim >> > (
            data->points_d, data->indexes_d, data->jacobian_d, data->MATRIX_SIZE, data->equation->get_power());
        cudaDeviceSynchronize();

        //cudaMemcpy(data->jacobian_h, data->jacobian_d, data->MATRIX_SIZE * data->MATRIX_SIZE * sizeof(double), cudaMemcpyDeviceToHost);

#ifdef INTERMEDIATE_RESULTS
        end = std::chrono::steady_clock::now();
        data->intermediate_results[1] = std::chrono::duration<double>(end - start).count();
        start = std::chrono::steady_clock::now();
#endif
        if (data->is_cublas) {
            gpu_cublas_solve(data);
        } else {
            gpu_cusolver_solve(data);
        }
        cudaDeviceSynchronize();
#ifdef INTERMEDIATE_RESULTS
        end = std::chrono::steady_clock::now();
        data->intermediate_results[2] = std::chrono::duration<double>(end - start).count();
        start = std::chrono::steady_clock::now();
#endif

        cudaMemcpy(data->funcs_value_h, data->funcs_value_d, data->MATRIX_SIZE * sizeof(double), cudaMemcpyDeviceToHost);
        dx = 0.0;
        for (size_t i = 0; i < data->MATRIX_SIZE; ++i) {
            data->points_h[i] -= data->funcs_value_h[i];
            dx = std::max(dx, std::abs(data->funcs_value_h[i]));
        }

        cudaMemcpy(data->points_d, data->points_h, data->MATRIX_SIZE * sizeof(double), cudaMemcpyHostToDevice);

#ifdef INTERMEDIATE_RESULTS
        end = std::chrono::steady_clock::now();
        data->intermediate_results[3] = std::chrono::duration<double>(end - start).count();

        tools::print_intermediate_result(data, iterations_count, dx);
#endif
        file_op->append_file_data(
                    data->intermediate_results, data->MATRIX_SIZE,
                    data->nnz_row, iterations_count,
                    sinfo_.mem_rss_usage_get(), sinfo_.gpu_mem_usage_get(),
                    data->is_cublas ? "cuBLAS" : "cuSOLVER",
                    data->settings.label);
    } while (dx > TOLERANCE);
	file_op->close_file();

	auto end_total = std::chrono::steady_clock::now();
    data->total_elapsed_time = std::chrono::duration<double>(end_total - start_total).count();

    tools::print_solution(data, iterations_count);
    cudaStreamDestroy(myStream);
    delete[] delta;
}
