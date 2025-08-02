#pragma once
#include "DataInitializer.h"

class DataInitializerCPU : public DataInitializer
{
public:
	double* funcs_value_h{ nullptr },
		* jacobian_h{ nullptr },
		* inverse_jacobian_h{ nullptr },
		* delta_h{ nullptr };

	DataInitializerCPU(int MATRIX_SIZE, int zeros_elements_per_row, int file_name,
			const Settings::SettingsData& s, int power);
	~DataInitializerCPU();
};
