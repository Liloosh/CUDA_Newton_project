#include "EditionalTools.h"
#include "DataInitializer.h"
#include "time.h"
#include "stdlib.h"

void tools::generate_initial_indexes_matrix_and_vector_b(double* matrix, double* b) {
	//srand(time(0));
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

double tools::calculate_index_xn(double index, double x) {
	return index * x;
}