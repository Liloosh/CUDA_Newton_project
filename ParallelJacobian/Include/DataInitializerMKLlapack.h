#pragma once
#include "DataInitializer.h"

class DataInitializerMKLlapack : public DataInitializer {
public:
	double* funcs_value_h{ nullptr },
		* jacobian_h{ nullptr };
    
    DataInitializerMKLlapack(int MATRIX_SIZE, int zeros_elements_per_row,
                             int file_name, const Settings::SettingsData& s, int power);
    ~DataInitializerMKLlapack();
};
