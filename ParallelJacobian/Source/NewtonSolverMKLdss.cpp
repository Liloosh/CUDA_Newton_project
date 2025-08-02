#include "NewtonSolverMKLdss.h"
#include "FileOperations.h"
#include "iostream"
#include "chrono"
#include <memory>
#include "EditionalTools.h"
#include "mkl.h"


void NewtonSolverMKLdss::cpu_computeVec() {
    for (int i = 0; i < data->MATRIX_SIZE; ++i) {
        double sum = 0.0;
        int row_start = data->csr_rows_h[i];
        int row_end = data->csr_rows_h[i + 1];

        for (int idx = row_start; idx < row_end; ++idx) {
            int col = data->csr_cols_h[idx];
            double index = data->csr_values_h[idx];
            double x = data->points_h[col];

            sum += data->equation->calculate_term_value(index, x);
        }

        data->funcs_value_h[i] = (sum - data->vector_b_h[i]);
    }
}

double NewtonSolverMKLdss::cpu_compute_derivative(int elementIndex, int colIndex) {
    double value = data->points_h[colIndex];
    double element = data->csr_values_h[elementIndex];

    double f_plus =  data->equation->calculate_term_value(element, value + EQURENCY);
    double f_minus = data->equation->calculate_term_value(element, value - EQURENCY);

    return (f_plus - f_minus) / (2.0 * EQURENCY);
}

void NewtonSolverMKLdss::cpu_compute_jacobian() {

    for (int i = 0; i < data->MATRIX_SIZE; ++i) {
        int row_start = data->csr_rows_h[i];
        int row_end = data->csr_rows_h[i + 1];

        for (int idx = row_start; idx < row_end; ++idx) {
            int j = data->csr_cols_h[idx];
            data->jacobian[idx] = cpu_compute_derivative(idx, j);
        }
    }
}

void NewtonSolverMKLdss::cpu_mkl_dss_find_delta() {
	if (!data->analyzed) {
		dss_define_structure(
				data->handle, data->sym, data->csr_rows_h,
				data->matrix_size, data->matrix_size, data->csr_cols_h,
				data->non_zero_count);

		dss_reorder(data->handle, data->order, 0);
		data->analyzed = true;
	}

	dss_factor_real(data->handle, data->type, data->jacobian);

	static const MKL_INT nrhs = 1;
	dss_solve_real(data->handle, data->opt, data->funcs_value_h, nrhs, data->delta_h);
}

void NewtonSolverMKLdss::cpu_newton_solve() {
	std::unique_ptr<FileOperations> file_op = std::make_unique<FileOperations>(settings_.path);
	std::string file_name = "cpu_mkl_dss_newton_solver_" + std::to_string(data->file_name) + ".csv";
	file_op->create_file(file_name, 4);
	file_op->append_file_headers(data->csv_header);

	std::cout << "CPU MKL dss Newton solver\n";
	std::cout << "Power: " << data->equation->get_power() << "\n";
	int iterations_count = 0;
	double dx = 0;

	auto start_total = std::chrono::steady_clock::now();

	std::cout << "Non-zero count: " << data->non_zero_count << "\n";
	do {
		iterations_count++;

#ifdef INTERMEDIATE_RESULTS
		auto start = std::chrono::steady_clock::now();
#endif

		cpu_computeVec();	

#ifdef INTERMEDIATE_RESULTS
		auto end = std::chrono::steady_clock::now();
		data->intermediate_results[0] = std::chrono::duration<double>(end - start).count();
		start = std::chrono::steady_clock::now();
#endif
		cpu_compute_jacobian();

#ifdef INTERMEDIATE_RESULTS
		end = std::chrono::steady_clock::now();
		data->intermediate_results[1] = std::chrono::duration<double>(end - start).count();
		start = std::chrono::steady_clock::now();
#endif

		cpu_mkl_dss_find_delta();
#ifdef INTERMEDIATE_RESULTS
		end = std::chrono::steady_clock::now();
		data->intermediate_results[2] = std::chrono::duration<double>(end - start).count();
		start = std::chrono::steady_clock::now();
#endif
		dx = 0.0;
		for (int i = 0; i < data->MATRIX_SIZE; ++i) {
			data->points_h[i] -= data->delta_h[i];
			dx = std::max(dx, std::abs(data->delta_h[i]));
		}

#ifdef INTERMEDIATE_RESULTS
		end = std::chrono::steady_clock::now();
		data->intermediate_results[3] = std::chrono::duration<double>(end - start).count();
		tools::print_intermediate_result(data, iterations_count, dx);
#endif
		file_op->append_file_data(
				data->intermediate_results, data->MATRIX_SIZE,
				data->nnz_row, iterations_count,
				sinfo_.mem_rss_usage_get(), sinfo_.gpu_mem_usage_get(),
				"MKL_DSS", data->settings.label);
	} while (dx > TOLERANCE);

	auto end_total = std::chrono::steady_clock::now();
	data->total_elapsed_time = std::chrono::duration<double>(end_total - start_total).count();
	tools::print_solution(data, iterations_count);
}

NewtonSolverMKLdss::NewtonSolverMKLdss(DataInitializerMKLdss* data,
		const Settings::SettingsData& settings, SystemInfo& sinfo)
	: settings_{settings}
	, sinfo_{sinfo}
{
	this->data = data;
}

NewtonSolverMKLdss::~NewtonSolverMKLdss() {
}
