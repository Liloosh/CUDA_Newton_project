#include <iostream>
#include <vector>
#include <random>
#include <numeric>
#include <sycl/sycl.hpp>
#include "oneapi/mkl.hpp"

constexpr int64_t MATRIX_SIZE = 100;

using namespace oneapi;

template <typename fp, typename intType>
static void diagonal_mv(sycl::queue& queue,
                        intType nrows,
                        sycl::buffer<fp, 1>& d_buffer,
                        sycl::buffer<fp, 1>& t_buffer) {
    queue.submit([&](sycl::handler& cgh) {
        auto d = d_buffer.template get_access<sycl::access::mode::read>(cgh);
        auto t = t_buffer.template get_access<sycl::access::mode::read_write>(cgh);
        cgh.parallel_for(sycl::range<1>(nrows), [=](sycl::id<1> idx) {
            t[idx] *= d[idx];
        });
    });
}

int count_non_zero_elements(const std::vector<double>& matrix) {
    return std::count_if(matrix.begin(), matrix.end(), [](double val) { return val != 0.0; });
}

void generate_diagonally_dominant_matrix(std::vector<double>& matrix, std::vector<double>& b, std::vector<double>& x_true) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    const int N = MATRIX_SIZE;

    matrix.resize(N * N);
    x_true.resize(N);
    b.resize(N);

    for (int i = 0; i < N; ++i) {
        x_true[i] = dist(gen);
        double row_sum = 0.0;

        for (int j = 0; j < N; ++j) {
            if (i != j) {
                double val = dist(gen);
                matrix[i * N + j] = val;
                row_sum += std::abs(val);
            }
        }
        matrix[i * N + i] = row_sum + 1.0;
    }

    for (int i = 0; i < N; ++i) {
        b[i] = 0.0;
        for (int j = 0; j < N; ++j) {
            b[i] += matrix[i * N + j] * x_true[j];
        }
    }
}

void dense_to_csr(const std::vector<double>& dense,
                  std::vector<int64_t>& ia,
                  std::vector<int64_t>& ja,
                  std::vector<double>& a) {
    const int N = MATRIX_SIZE;
    ia.resize(N + 1);
    ia[0] = 0;

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            double val = dense[i * N + j];
            if (val != 0.0) {
                ja.push_back(j);
                a.push_back(val);
            }
        }
        ia[i + 1] = static_cast<int64_t>(ja.size());
    }
}

template <typename fp>
void run_sparse_cg_example(const sycl::device& dev) {
    using intType = int64_t;

    const intType size = MATRIX_SIZE;

    std::vector<double> dense_matrix, b, x_true;
    generate_diagonally_dominant_matrix(dense_matrix, b, x_true);

    std::vector<intType> ia, ja;
    std::vector<fp> a;
    dense_to_csr(dense_matrix, ia, ja, a);

    intType nrows = size;
    intType nnz = a.size();

    std::vector<fp> x(size, 0.0f);

    sycl::queue queue(dev, [](sycl::exception_list el) {
        for (auto& e : el) {
            try {
                std::rethrow_exception(e);
            } catch (sycl::exception& ex) {
                std::cerr << "SYCL exception: " << ex.what() << "\n";
            }
        }
    });

    sycl::buffer<intType> ia_buf(ia.data(), ia.size());
    sycl::buffer<intType> ja_buf(ja.data(), ja.size());
    sycl::buffer<fp> a_buf(a.data(), a.size());
    sycl::buffer<fp> x_buf(x.data(), x.size());
    sycl::buffer<fp> b_buf(b.data(), b.size());
    sycl::buffer<fp> r_buf(size), w_buf(size), p_buf(size);
    sycl::buffer<fp> t_buf(size), d_buf(size), temp_buf(1);

    mkl::sparse::matrix_handle_t handle;
    mkl::sparse::init_matrix_handle(&handle);

    mkl::sparse::set_csr_data(queue, handle, nrows, nrows, mkl::index_base::zero, ia_buf, ja_buf, a_buf);
    mkl::sparse::set_matrix_property(handle, mkl::sparse::property::symmetric);
    mkl::sparse::set_matrix_property(handle, mkl::sparse::property::sorted);

    mkl::sparse::optimize_gemv(queue, mkl::transpose::nontrans, handle);
    mkl::sparse::optimize_trsv(queue, mkl::uplo::lower, mkl::transpose::nontrans, mkl::diag::nonunit, handle);
    mkl::sparse::optimize_trsv(queue, mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::nonunit, handle);

    queue.submit([&](sycl::handler& cgh) {
        auto ia_acc = ia_buf.template get_access<sycl::access::mode::read>(cgh);
        auto ja_acc = ja_buf.template get_access<sycl::access::mode::read>(cgh);
        auto a_acc = a_buf.template get_access<sycl::access::mode::read>(cgh);
        auto d_acc = d_buf.template get_access<sycl::access::mode::write>(cgh);

        cgh.parallel_for(sycl::range<1>(nrows), [=](sycl::id<1> idx) {
            intType row = idx[0];
            for (intType i = ia_acc[row]; i < ia_acc[row + 1]; ++i) {
                if (ja_acc[i] == row) {
                    d_acc[row] = a_acc[i];
                    break;
                }
            }
        });
    });

    mkl::blas::copy(queue, size, b_buf, 1, r_buf, 1);

    mkl::sparse::trsv(queue, mkl::uplo::lower, mkl::transpose::nontrans, mkl::diag::nonunit, handle, r_buf, t_buf);
    diagonal_mv<fp, intType>(queue, size, d_buf, t_buf);
    mkl::sparse::trsv(queue, mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::nonunit, handle, t_buf, w_buf);
    mkl::blas::copy(queue, size, w_buf, 1, p_buf, 1);

    fp norm_corr = 0, initial_norm = 0, alpha = 0, beta = 0, temp = 0;
    int iter = 0;

    mkl::blas::nrm2(queue, size, w_buf, 1, temp_buf);
    initial_norm = temp_buf.template get_access<sycl::access::mode::read>()[0];
    mkl::blas::dot(queue, size, r_buf, 1, w_buf, 1, temp_buf);
    temp = temp_buf.template get_access<sycl::access::mode::read>()[0];

    while (iter < 100) {
        mkl::sparse::gemv(queue, mkl::transpose::nontrans, 1.0f, handle, p_buf, 0.0f, t_buf);
        mkl::blas::dot(queue, size, p_buf, 1, t_buf, 1, temp_buf);
        alpha = temp / temp_buf.template get_access<sycl::access::mode::read>()[0];

        mkl::blas::axpy(queue, size, alpha, p_buf, 1, x_buf, 1);
        mkl::sparse::gemv(queue, mkl::transpose::nontrans, -alpha, handle, p_buf, 1.0f, r_buf);

        mkl::sparse::trsv(queue, mkl::uplo::lower, mkl::transpose::nontrans, mkl::diag::nonunit, handle, r_buf, t_buf);
        diagonal_mv<fp, intType>(queue, size, d_buf, t_buf);
        mkl::sparse::trsv(queue, mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::nonunit, handle, t_buf, w_buf);
        mkl::blas::nrm2(queue, size, w_buf, 1, temp_buf);

        norm_corr = temp_buf.template get_access<sycl::access::mode::read>()[0];
        std::cout << "Iter " << iter++ << ", rel norm = " << norm_corr / initial_norm << "\n";

        if (norm_corr / initial_norm < 1e-3) break;

        mkl::blas::dot(queue, size, r_buf, 1, w_buf, 1, temp_buf);
        beta = temp_buf.template get_access<sycl::access::mode::read>()[0] / temp;
        temp = temp_buf.template get_access<sycl::access::mode::read>()[0];

        mkl::blas::scal(queue, size, beta, p_buf, 1);
        mkl::blas::axpy(queue, size, 1.0f, w_buf, 1, p_buf, 1);
    }

    mkl::sparse::release_matrix_handle(queue, &handle);

    auto result = x_buf.template get_access<sycl::access::mode::read>();
    std::cout << "\nSample solution:\n";
    for (int i = 0; i < size; ++i) {
        std::cout << "x[" << i << "] = " << result[i] << " " << x_true[i] << "\n";
    }
}

int main() {
    std::cout << "### oneMKL PCG Sparse Solver ###\n";
     sycl::device dev{sycl::default_selector{}};
    std::cout << "Device: " << dev.get_info<sycl::info::device::name>() << "\n";

    run_sparse_cg_example<double>(dev);
    return 0;
}
