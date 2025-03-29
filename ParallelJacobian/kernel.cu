#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "iostream"
#include "math.h"
#include "vector"
#include "cublas.h"
#include "DataInitializer.h"
#include "NewtonSolver.h"
#include "memory"
#include "config.h"

int main() {

    //
    // CPY
    //
#ifdef CPU_SOLVER
    {
        std::unique_ptr<DataInitializer> data = std::make_unique<DataInitializer>();
        std::unique_ptr<NewtonSolver> newton_solver = std::make_unique<NewtonSolver>(data.get());
        newton_solver->cpu_newton_solve();
    }
#endif

    //
    // GPU
    //
#ifdef GPU_SOLVER
    {
        std::unique_ptr<DataInitializer> data2 = std::make_unique<DataInitializer>();
        std::unique_ptr<NewtonSolver> newton_solver2 = std::make_unique<NewtonSolver>(data2.get());
        newton_solver2->gpu_newton_solve();
    }
#endif

    return 0;
}