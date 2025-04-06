#pragma once

namespace tools {
	void generate_initial_indexes_matrix_and_vector_b(double* matrix, double* b);
	double calculate_index_xn(double index, double x);
	void generate_sparse_initial_indexes_matrix_and_vector_b(double* matrix, double* b, int zeros_per_row);
}