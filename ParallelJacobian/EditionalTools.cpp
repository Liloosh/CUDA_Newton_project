#include "EditionalTools.h"
#include "DataInitializer.h"
#include "time.h"
#include "stdlib.h"

void tools::generate_initial_indexes_matrix_and_vector_b(double* matrix, double* b) {
	int value = 0;
	double sum = 0;
	for (int i = 0; i < MATRIX_SIZE; i++) {

		b[i] = 1;

		value = 0;
		sum = 0;
		for (int j = 0; j < MATRIX_SIZE - 1; j++) {
			matrix[i * MATRIX_SIZE + j] = static_cast<double>(rand()) / RAND_MAX;
			sum += matrix[i * MATRIX_SIZE + j];
		}
		matrix[i * MATRIX_SIZE + MATRIX_SIZE - 1] = 10 - sum;
	}
}

void tools::generate_sparse_initial_indexes_matrix_and_vector_b(double* matrix, double* b, int zeros_per_row) {
    for (int i = 0; i < MATRIX_SIZE; i++) {
        b[i] = 1;

        double sum = 0.0;
        for (int j = 0; j < MATRIX_SIZE; j++) {
            matrix[i * MATRIX_SIZE + j] = static_cast<double>(rand()) / RAND_MAX;
            sum += matrix[i * MATRIX_SIZE + j];
        }
        bool zeroed[MATRIX_SIZE] = { false };

        int zeros_set = 0;
        while (zeros_set < zeros_per_row) {
            int idx = rand() % MATRIX_SIZE;
            if (!zeroed[idx]) {
                sum -= matrix[i * MATRIX_SIZE + idx];
                matrix[i * MATRIX_SIZE + idx] = 0.0;
                zeroed[idx] = true;
                zeros_set++;
            }
        }

        if (sum > 0) {
            for (int j = 0; j < MATRIX_SIZE; j++) {
                if (!zeroed[j]) {
                    matrix[i * MATRIX_SIZE + j] *= 10.0 / sum;
                }
            }
        }
    }
}


double tools::calculate_index_xn(double index, double x) {
	return index * x;
}