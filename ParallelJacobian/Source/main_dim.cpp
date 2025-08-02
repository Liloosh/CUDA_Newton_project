#include <iostream>
#include <memory>
#include <vector>

#include "FileOperations.h"
#include "NewtonSolverCPU.h"
#ifdef CFG_SOLVE_CUDA
#include "NewtonSolverCUDA.h"
#include "NewtonSolverCuDSS.h"
#endif
#ifdef CFG_SOLVE_MKL
#include "NewtonSolverMKLdss.h"
#include "NewtonSolverMKLlapack.h"
#endif
#include "config.h"
#include "report.h"
#include "settings.h"
#include "system-build-info.h"
#include "system-info.h"

int main(int argc, char* argv[]) {
    SystemBuildInfo::dump(std::cout);
    std::cout << std::endl;

    Settings s;
    if (!s.parse(argc, argv))
        return 1;

    Settings::SettingsData& sd = s.settings;
    SystemInfo sinfo(argc, argv);

    if (s.settings.report_subdir) {
        std::string path = s.settings.path + "/" + sinfo.getTimeStamp();
        if (!Report::createReportDir(path))
            return 2;

        s.settings.path = path;
    }

    Report::RedirectOut rdout;
    if (s.settings.redirect_out) {
        rdout.redirect(s.settings.path);

        SystemBuildInfo::dump(std::cout);
        std::cout << std::endl;
    }

    sinfo.dump(std::cout);

    unsigned matrix_size_max = s.settings.max;
    unsigned matrix_size_min = s.settings.min;
    int stride = s.settings.stride;
    int power = s.settings.power;

    if (s.settings.size != 0) {
        // use only one matrix size
        matrix_size_max = s.settings.size;
        matrix_size_min = s.settings.size;
    }

    std::unique_ptr<FileOperations> file_op = std::make_unique<FileOperations>(s.settings.path);
    std::string header =
            "CPU,cuBLAS,cuSOLVER,cuDSS,MKL_Lapack,MKL_DSS,matrix_size,mem_rss_max,mem_gpu_max,label";
    file_op->create_file("total_statistic.csv", 6);
    file_op->append_file_headers(header);
    std::vector<double> row(6, 0);

    for (unsigned size = matrix_size_min; size <= matrix_size_max; size += stride) {
        unsigned num_zeros = s.settings.nnz != 0 && s.settings.nnz < size ?
                    size - s.settings.nnz : 0;

        //
        // CPU
        //
        if (s.settings.is_cpu)
        {
            sinfo.dump_resource_usage(std::cout);
            auto data = std::make_unique<DataInitializerCPU>(
                        size, num_zeros, size, sd, power);
            auto newton_solver = std::make_unique<NewtonSolverCPU>(
                        data.get(), s.settings, sinfo);
            newton_solver->cpu_newton_solve();
            row[0] = data->total_elapsed_time;
            sinfo.dump_resource_usage(std::cout);
        }

#ifdef CFG_SOLVE_CUDA
        //
        // cuBLAS
        //
        if (s.settings.is_cublas)
        {
            sinfo.dump_resource_usage(std::cout);
            auto data2 = std::make_unique<DataInitializerCUDA>(
                        size, num_zeros, size, true, sd, power);
            auto newton_solver2 = std::make_unique<NewtonSolverCUDA>(
                        data2.get(), s.settings, sinfo);
            newton_solver2->gpu_newton_solve();
            row[1] = data2->total_elapsed_time;;
            sinfo.dump_resource_usage(std::cout);
        }

        //
        // cuSOLVER
        //
        if (s.settings.is_cusolver)
        {
            sinfo.dump_resource_usage(std::cout);
            auto data = std::make_unique<DataInitializerCUDA>(
                        size, num_zeros, size, false, sd, power);
            auto newton_solver = std::make_unique<NewtonSolverCUDA>(
                        data.get(), s.settings, sinfo);
            newton_solver->gpu_newton_solve();
            row[2] = data->total_elapsed_time;
            sinfo.dump_resource_usage(std::cout);
        }

        //
        // cuDSS
        //
        if (s.settings.is_cudss)
        {
            sinfo.dump_resource_usage(std::cout);
            auto data3 = std::make_unique<DataInitializerCuDSS>(
                        size, num_zeros, size, sd, power);
            auto cuDssSolver = std::make_unique<NewtonSolverCuDSS>(
                        data3.get(), s.settings, sinfo);
            cuDssSolver->gpu_newton_solver_cudss();
            row[3] = data3->total_elapsed_time;;
            sinfo.dump_resource_usage(std::cout);
        }
#endif
#ifdef CFG_SOLVE_MKL
        //
        // MKL Lapack
        //
        if (s.settings.is_mkl_lapack)
        {
            sinfo.dump_resource_usage(std::cout);
            auto data = std::make_unique<DataInitializerMKLlapack>(
                        size, num_zeros, size, sd, power);
            auto mklLapackSolver = std::make_unique<NewtonSolverMKLlapack>(
                        data.get(), s.settings, sinfo);
            mklLapackSolver->cpu_newton_solve();
            row[4] = data->total_elapsed_time;;
            sinfo.dump_resource_usage(std::cout);
        }

        //
        // MKL DSS
        //
        if (s.settings.is_mkl_dss)
        {
            sinfo.dump_resource_usage(std::cout);
            auto data = std::make_unique<DataInitializerMKLdss>(
                        size, num_zeros, size, sd, power);
            auto mklDssSolver = std::make_unique<NewtonSolverMKLdss>(
                        data.get(), s.settings, sinfo);
            mklDssSolver->cpu_newton_solve();
            row[5] = data->total_elapsed_time;;
            sinfo.dump_resource_usage(std::cout);
        }
#endif

        file_op->append_file_data(row, size, sinfo.mem_rss_max_usage_get(),
                                  sinfo.gpu_mem_usage_max_get(), sd.label);
    }

    sinfo.dump_resource_usage(std::cout);

    return 0;
}
