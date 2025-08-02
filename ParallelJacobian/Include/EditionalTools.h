#pragma once
#include "Equation.h"
#include "DataInitializer.h"

namespace tools {
	void generate_sparse_initial_indexes_matrix_and_vector_b(
		double* matrix,
		double* b,
		double* points,
		int MATRIX_SIZE,
		Equation* equation,
		int zero_elements_per_row,
		const Settings::SettingsData& s);

    template<typename T>
	void generate_sparse_initial_indexes_matrix_and_vector_b(
		double* csr_values,
		T* csr_rows,
		T* csr_cols,
		double* b,
		double* points,
		int MATRIX_SIZE,
		Equation* equation,
		int zero_elements_per_row,
		const Settings::SettingsData& s);
	double calculate_index_xn(double index, double x);

	//
	// RESULTS PRINT
	//
	void print_solution(DataInitializer* data, int iterations_count);
	void print_intermediate_result(DataInitializer* data, int iteration_number, double error);
}
