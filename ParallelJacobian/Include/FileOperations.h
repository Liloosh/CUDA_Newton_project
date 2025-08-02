#pragma once

#include <fstream>
#include <string>
#include <vector>

class FileOperations {
private:
	bool isFileExist;
	std::ofstream my_file;
	int number_of_columns;
	std::string path_;

public: 
	FileOperations(std::string path);

	void create_file(std::string file_name, int number_of_columns);
	void close_file();
	void append_file_data(
			const std::vector<double>& row, int MATRIX_SIZE, size_t mem_rss,
			size_t mem_gpu, std::string label);
	void append_file_data(
			const std::vector<double>& row, int MATRIX_SIZE, int nnz_row,
			int iter_num, size_t mem_rss, size_t mem_gpu, std::string solver,
			std::string label);
	void append_file_headers(std::string headers);
};
