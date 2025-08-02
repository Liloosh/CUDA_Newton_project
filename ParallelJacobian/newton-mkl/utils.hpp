//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

template <typename fp> fp rand_scalar() {
    return fp(std::rand()) / fp(RAND_MAX) - fp(0.5);
}

// Підрахунок ненульових елементів у квадратній матриці 5-пунктового шаблону
template <typename intType>
intType count_nnz(const intType nx, const intType ny) {
    intType nnz = 0;
    for (intType iy = 0; iy < ny; iy++) {
        for (intType ix = 0; ix < nx; ix++) {
            intType count = 1; // центр
            if (ix > 0) count++;      // вліво
            if (ix < nx - 1) count++; // вправо
            if (iy > 0) count++;      // вгору
            if (iy < ny - 1) count++; // вниз
            nnz += count;
        }
    }
    return nnz;
}

// Генерація квадратної розрідженої матриці (5-point stencil) в форматі CSR
template <typename fp, typename intType>
void generate_sparse_matrix(const intType nx,
                            std::vector<intType> &ia,
                            std::vector<intType> &ja,
                            std::vector<fp> &a)
{
    intType ny = nx;
    intType nnz = 0;
    ia[0] = 0;

    for (intType iy = 0; iy < ny; iy++) {
        for (intType ix = 0; ix < nx; ix++) {
            intType current_row = iy * nx + ix;

            // Ліворуч
            if (ix > 0) {
                ja[nnz] = current_row - 1;
                a[nnz++] = -1.;
            }

            // Вгору
            if (iy > 0) {
                ja[nnz] = current_row - nx;
                a[nnz++] = -1.;
            }

            // Центр
            ja[nnz] = current_row;
            a[nnz++] = 4.;

            // Вниз
            if (iy < ny - 1) {
                ja[nnz] = current_row + nx;
                a[nnz++] = -1.;
            }

            // Праворуч
            if (ix < nx - 1) {
                ja[nnz] = current_row + 1;
                a[nnz++] = -1.;
            }

            ia[current_row + 1] = nnz;
        }
    }
}
