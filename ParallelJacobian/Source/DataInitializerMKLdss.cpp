#include "DataInitializerMKLdss.h"

#include <iostream>

#include "EditionalTools.h"

DataInitializerMKLdss::DataInitializerMKLdss(
		int MATRIX_SIZE, int zeros_elements_per_row, int file_name,
		const Settings::SettingsData& s, int power)
	: DataInitializer(MATRIX_SIZE, zeros_elements_per_row, file_name, s, power,
			true) {
	// Allocate memory for CSR representation
	matrix_size = MATRIX_SIZE;
	non_zero_count = MATRIX_SIZE * (MATRIX_SIZE - zeros_elements_per_row);
	csr_values_h = new double[non_zero_count];
	jacobian = new double[non_zero_count];
	csr_rows_h = new MKL_INT[MATRIX_SIZE + 1];
	csr_cols_h = new MKL_INT[non_zero_count];

    delta_h = new double[MATRIX_SIZE];
	funcs_value_h = new double[MATRIX_SIZE];

    tools::generate_sparse_initial_indexes_matrix_and_vector_b(
                csr_values_h, csr_rows_h, csr_cols_h, vector_b_h, points_check,
                MATRIX_SIZE, equation, zeros_elements_per_row, s);
    non_zero_count = csr_rows_h[MATRIX_SIZE];

    dss_create(handle, opt);
}

DataInitializerMKLdss::~DataInitializerMKLdss(){
    delete[] csr_values_h;
	delete[] jacobian;
    delete[] csr_cols_h;
    delete[] csr_rows_h;

	delete[] delta_h;
	delete[] funcs_value_h;

    dss_delete(handle, opt);
}
