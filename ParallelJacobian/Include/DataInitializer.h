#pragma once
#include "Equation.h"
#include "vector"
#include "config.h"
#include "cublas_v2.h"

#define BLOCK_SIZE 64
#define EQURENCY 1e-6
#define TOLERANCE 1e-5
#define SHAFFLE_CONST 0xffffffff

struct DataInitializer {
public:
	int file_name;
	int zeros_elements_per_row;
	Equation* equation{ nullptr };
	int MATRIX_SIZE{ 0 };

	double* indexes_h{ nullptr },
		* points_h{ nullptr },
		* vector_b_h{ nullptr },
		* points_check{ nullptr };

#ifdef INTERMEDIATE_RESULTS
	std::vector<double> intermediate_results;
#endif

#ifdef TOTAL_ELASPED_TIME
	double total_elapsed_time;
#endif

	void initialize_indexes_matrix_and_b(bool isCuDSS = false);
 
	DataInitializer(int MATRIX_SIZE, int zeros_elements_per_row, int file_name, int power = 1, bool isCuDSS = false);
	~DataInitializer();
};