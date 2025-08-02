#include "math.h"
#include <iostream>
#include <memory>
#include <chrono>
#include "mkl.h"

#include "FileOperations.h"
#include "NewtonSolverMKLlapack.h"
#include "EditionalTools.h"

NewtonSolverMKLlapack::NewtonSolverMKLlapack(DataInitializerMKLlapack* dataInitializer,
        const Settings::SettingsData& settings, SystemInfo& sinfo)
    : settings_{settings}
    , sinfo_{sinfo}
{
	data = dataInitializer;
}

NewtonSolverMKLlapack::~NewtonSolverMKLlapack() {
}

void NewtonSolverMKLlapack::cpu_computeVec() {
    const int N = data->MATRIX_SIZE;

    for (int i = 0; i < N; ++i) {
        double sum = 0.0;
        for (int j = 0; j < N; ++j) {
            double index = data->indexes_h[i * N + j];
            double x = data->points_h[j];
            sum += data->equation->calculate_term_value(index, x);
        }

        data->funcs_value_h[i] = sum - data->vector_b_h[i];
    }
}

double NewtonSolverMKLlapack::cpu_compute_derivative(int rowIndex, int colIndex) {
    double value = data->points_h[colIndex];
    double element = data->indexes_h[rowIndex * data->MATRIX_SIZE + colIndex];

    double f_plus =  data->equation->calculate_term_value(element, value + EQURENCY);
    double f_minus = data->equation->calculate_term_value(element, value - EQURENCY);;

    return (f_plus - f_minus) / (2.0 * EQURENCY);
}

void NewtonSolverMKLlapack::cpu_compute_jacobian() {
    for (int i = 0; i < data->MATRIX_SIZE; ++i) {
        for (int j = 0; j < data->MATRIX_SIZE; ++j) {
            data->jacobian_h[i * data->MATRIX_SIZE + j] = cpu_compute_derivative(i, j);
        }
    }
}

void NewtonSolverMKLlapack::cpu_find_delta(){
    lapack_int *ipiv = new lapack_int[data->MATRIX_SIZE];
    int info;
    info = LAPACKE_dgetrf(LAPACK_COL_MAJOR, data->MATRIX_SIZE, data->MATRIX_SIZE, data->jacobian_h, data->MATRIX_SIZE, ipiv);
    
    if (info != 0) {
        std::cerr << "dgetrf error: " << info << std::endl;
    exit(EXIT_FAILURE);
    }   

    info = LAPACKE_dgetrs(LAPACK_COL_MAJOR, 'T', data->MATRIX_SIZE, 1, data->jacobian_h, data->MATRIX_SIZE,
        ipiv, data->funcs_value_h, data->MATRIX_SIZE);
    if (info != 0) {
    std::cerr << "dgetrs error: " << info << std::endl;
    exit(EXIT_FAILURE);
}
    delete[] ipiv;
}

void NewtonSolverMKLlapack::cpu_newton_solve() {
    std::cout << "CPU Newton solver\n";
    double dx = 0.0;
    int iterations_count = 0;
    std::unique_ptr<FileOperations> file_op = std::make_unique<FileOperations>(settings_.path);
    std::string file_name = "cpu_mkl_lapack_newton_solver_" + std::to_string(data->file_name) + ".csv";
    file_op->create_file(file_name, 4);
    file_op->append_file_headers(data->csv_header);


#ifdef TOTAL_ELASPED_TIME
    auto start_total = std::chrono::steady_clock::now();
#endif
    do {
        iterations_count++;

#ifdef INTERMEDIATE_RESULTS
        auto start = std::chrono::steady_clock::now();
#endif
        cpu_computeVec();
#ifdef INTERMEDIATE_RESULTS
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        data->intermediate_results[0] = elapsed.count();
#endif

#ifdef INTERMEDIATE_RESULTS
        start = std::chrono::steady_clock::now();
#endif
        cpu_compute_jacobian();
#ifdef INTERMEDIATE_RESULTS
        end = std::chrono::steady_clock::now();
        elapsed = end - start;
        data->intermediate_results[1] = elapsed.count();
#endif

#ifdef INTERMEDIATE_RESULTS
        start = std::chrono::steady_clock::now();
#endif
        cpu_find_delta();
#ifdef INTERMEDIATE_RESULTS
        end = std::chrono::steady_clock::now();
        elapsed = end - start;
        data->intermediate_results[2] = elapsed.count();
#endif

#ifdef INTERMEDIATE_RESULTS
        start = std::chrono::steady_clock::now();
#endif
        dx = 0;
        for (int i = 0; i < data->MATRIX_SIZE; ++i) {
            data->points_h[i] -= data->funcs_value_h[i];
            dx = std::max(dx, std::abs(data->funcs_value_h[i]));
        }
#ifdef INTERMEDIATE_RESULTS
        end = std::chrono::steady_clock::now();
        elapsed = end - start;
        data->intermediate_results[3] = elapsed.count();
        tools::print_intermediate_result(data, iterations_count, dx);
#endif

        file_op->append_file_data(
                    data->intermediate_results, data->MATRIX_SIZE,
                    data->nnz_row, iterations_count,
                    sinfo_.mem_rss_usage_get(), sinfo_.gpu_mem_usage_get(),
                    "MKL_LAPACK", data->settings.label);
    } while (dx > TOLERANCE);
    file_op->close_file();

	auto end_total = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsed_total = end_total - start_total;
	data->total_elapsed_time = elapsed_total.count();

    tools::print_solution(data, iterations_count);
}
