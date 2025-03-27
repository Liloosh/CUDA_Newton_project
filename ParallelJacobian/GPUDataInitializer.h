#pragma once

struct GPUDataInitializer {
public:
	double* indexes_d{ nullptr },
		* points_d{ nullptr },
		* vector_d{ nullptr },
		* vector_b_d{ nullptr },
		* jacobian_d{ nullptr },
		* inverse_jacobian_d{ nullptr },
		* delta_d{ nullptr },
		* vec_d{ nullptr };
	int* cublas_pivot{nullptr}, * cublas_info{ nullptr };
	double** cublas_ajacobian_d{ nullptr }, ** cublas_ainverse_jacobian_d{ nullptr };
};
