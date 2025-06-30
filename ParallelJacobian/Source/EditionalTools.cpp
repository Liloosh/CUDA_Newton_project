#include "EditionalTools.h"

#include <iostream>
#include <numeric>
#include <random> 
#include <vector>

#include "DataInitializer.h"

template<class It, class G>
void random_shuffle(It first, It last, G& gen)
{
    for (auto i = last - first - 1; i > 0; --i)
    {
        std::swap(first[i], first[gen() % (i + 1)]);
    }
}

void tools::generate_initial_indexes_matrix_and_vector_b(double* matrix, double* b, double* points, int MATRIX_SIZE, Equation* equation) {
    std::mt19937 gen;
    auto rand_max = gen.max();

    for (int i = 0; i < MATRIX_SIZE; i++) {
        points[i] = static_cast<double>(gen()) / rand_max;
        for (int j = 0; j < MATRIX_SIZE; j++) {
            matrix[i * MATRIX_SIZE + j] = static_cast<double>(gen()) / rand_max;
        }
    }

    for (int i = 0; i < MATRIX_SIZE; i++) {
        b[i] = 0;
        for (int j = 0; j < MATRIX_SIZE; j++) {
            b[i] += equation->calculate_term_value(matrix[i * MATRIX_SIZE + j], points[j]);
        }
    }
}

void tools::generate_sparse_initial_indexes_matrix_and_vector_b(
    double* matrix,
    double* b,
    double* points,
    int MATRIX_SIZE,
    Equation* equation,
    int zero_elements_per_row)
{
    if (zero_elements_per_row < 0 || zero_elements_per_row >= MATRIX_SIZE) {
        zero_elements_per_row = 0;
    }

    std::mt19937 gen;
    auto rand_max = gen.max();

    for (int i = 0; i < MATRIX_SIZE; i++) {
        points[i] = static_cast<double>(gen()) / rand_max;
    }

    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            matrix[i * MATRIX_SIZE + j] = static_cast<double>(gen()) / rand_max;
        }


        if (zero_elements_per_row > 0) {
            std::vector<int> positions(MATRIX_SIZE);
            std::iota(positions.begin(), positions.end(), 0);
            random_shuffle(positions.begin(), positions.end(), gen);

            for (int k = 0; k < zero_elements_per_row; k++) {
                int j = positions[k];
                matrix[i * MATRIX_SIZE + j] = 0.0;
            }
        }
    }

    for (int i = 0; i < MATRIX_SIZE; i++) {
		double sum = 0.0;
		double diagonal_value = 0.0;
        for (int j = 0; j < MATRIX_SIZE; j++) {
            if (j != i) {
                sum += matrix[i * MATRIX_SIZE + j];
            }
        }

        diagonal_value = matrix[i * MATRIX_SIZE + i];
        if ((sum + 1e-7) >= diagonal_value) {
			matrix[i * MATRIX_SIZE + i] = sum - diagonal_value + 1.0;
        }
    }

    for (int i = 0; i < MATRIX_SIZE; i++) {
        b[i] = 0;
        for (int j = 0; j < MATRIX_SIZE; j++) {
            if (matrix[i * MATRIX_SIZE + j] != 0.0) {
                b[i] += equation->calculate_term_value(matrix[i * MATRIX_SIZE + j], points[j]);
            }
        }
    }
}

void tools::generate_sparse_initial_indexes_matrix_and_vector_b(
    double* csr_values,
    int* csr_rows,
    int* csr_cols,
    double* b,
    double* points,
    int MATRIX_SIZE,
    Equation* equation,
    int zero_elements_per_row)
{
    if (zero_elements_per_row < 0 || zero_elements_per_row >= MATRIX_SIZE) {
        zero_elements_per_row = 0;
    }

    std::mt19937 gen;
    auto rand_max = gen.max();

    for (int i = 0; i < MATRIX_SIZE; i++) {
        points[i] = static_cast<double>(gen()) / rand_max;
    }

    csr_rows[0] = 0;
    int current_pos = 0;

    for (int i = 0; i < MATRIX_SIZE; i++) {
        std::vector<int> positions;
        positions.reserve(MATRIX_SIZE);

        positions.push_back(i);

        for (int j = 0; j < MATRIX_SIZE; j++) {
            if (j != i) {
                positions.push_back(j);
            }
        }

        std::shuffle(positions.begin() + 1, positions.end(), gen);

        int non_zero_count = MATRIX_SIZE - zero_elements_per_row;
        std::vector<int> selected_positions(positions.begin(), positions.begin() + non_zero_count);
        std::sort(selected_positions.begin(), selected_positions.end());

        for (int col : selected_positions) {
            csr_cols[current_pos] = col;
            csr_values[current_pos] = static_cast<double>(gen()) / rand_max;
            current_pos++;
        }

        csr_rows[i + 1] = current_pos;
    }

    for (int i = 0; i < MATRIX_SIZE; i++) {
        double sum = 0.0;
        int diagonal_idx = -1;

        for (int idx = csr_rows[i]; idx < csr_rows[i + 1]; idx++) {
            if (csr_cols[idx] == i) {
                diagonal_idx = idx;
            }
            else {
                sum += std::abs(csr_values[idx]);
            }
        }

        if (diagonal_idx != -1) {
            if ((sum + 1e-7) >= std::abs(csr_values[diagonal_idx])) {
                csr_values[diagonal_idx] = sum + 1.0;
            }
        }
    }

    for (int i = 0; i < MATRIX_SIZE; i++) {
        b[i] = 0.0;
        for (int idx = csr_rows[i]; idx < csr_rows[i + 1]; idx++) {
            int col = csr_cols[idx];
            double val = csr_values[idx];
            b[i] += equation->calculate_term_value(val, points[col]);
        }
    }
}



double tools::calculate_index_xn(double index, double x) {
	return index * x;
}

void tools::print_solution(DataInitializer* data, int iterations_count = 0) {
    std::cout << "Total Iterations count: " << iterations_count << "\n";

#ifdef TOTAL_ELASPED_TIME
    std::cout << "Total elapsed time: " << data->total_elapsed_time << "\n";
#endif
#ifdef SOLUTION_PRINT
    std::cout << "Solution: \n";

    for (int i = 0; i < data->MATRIX_SIZE; i++) {
        std::cout << data->points_h[i] << " " << data->points_check[i] << "\n";
    }
#endif
    std::cout << "\n";

    bool isCorrect = true;

    for (int i = 0; i < data->MATRIX_SIZE; i++) {
        if (std::abs(data->points_h[i] - data->points_check[i]) > 1e-5) {
            isCorrect = false;
        }
    }

    if (isCorrect) {
        std::cout << "Solution is correct!" << "\n";
    }
    else {
        std::cout << "Solution is incorrect!" << "\n";
    }
}

void tools::print_intermediate_result(DataInitializer* data, int iteration_number, double error, bool isCudss) {
    std::cout << "\nIteration: " << iteration_number << "\n";
    std::cout << "===============================================================\n";
    std::cout << "Intermediate results: \n";
    std::cout << "Compute func values: " << data->intermediate_results[0] << "\n";
    std::cout << "Compute jacobian: " << data->intermediate_results[1] << "\n";
    std::cout << "Compute inverse jacobian: " << data->intermediate_results[2] << "\n";
    std::cout << "Compute delta: " << data->intermediate_results[3] << "\n";
    if (!isCudss) {
        std::cout << "Update points: " << data->intermediate_results[4] << "\n";
    }
    std::cout << "Error: " << error << "\n";
    std::cout << "===============================================================\n";
}
