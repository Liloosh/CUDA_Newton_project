#include "settings.h"

#include <cstring>
#include <iostream>

Settings::Settings()
{
    pmap_ = std::map<std::string, Parser>{
        {"--cpu", { parseBool, &settings.is_cpu }},
#ifdef CFG_SOLVE_CUDA
        {"--cublas", { parseBool, &settings.is_cublas }},
        {"--cudss", { parseBool, &settings.is_cudss }},
        {"--cusolver", { parseBool, &settings.is_cusolver }},
#endif
#ifdef CFG_SOLVE_MKL
        {"--mkl-dss", { parseBool, &settings.is_mkl_dss }},
        {"--mkl-lapack", { parseBool, &settings.is_mkl_lapack }},
#endif
        {"--max", { parseUnsigned, &settings.max }},
        {"--min", { parseUnsigned, &settings.min }},
        {"--size", { parseUnsigned, &settings.size }},
        {"--nnz", { parseUnsigned, &settings.nnz }},
        {"--seed", { parseUnsigned, &settings.seed }},
        {"--power", { parseUnsigned, &settings.power }},
        {"--rand-sparse-pos", { parseBool, &settings.rand_sparse_pos }},
        {"--stride", { parseUnsigned, &settings.stride }},
        {"--redirect-out", { parseBool, &settings.redirect_out }},
        {"--report-subdir", { parseBool, &settings.report_subdir }},
        {"--result-path", { parseString, &settings.path }},
        {"--label", { parseString, &settings.label }}
    };
}

void Settings::help()
{
    std::cout << "Supported options:" << std::endl;
    for (auto& i : pmap_) {
        std::cout << "\t" << i.first << std::endl;
    }
}

bool Settings::parse(int argc, char* argv[])
{
    if (argc == 2 && (!strcmp(argv[1], "-h") || !strcmp(argv[1], "--help"))) {
        help();
        return false;
    }

    int i = 1;

    while (i < argc) {
        auto iter = pmap_.find(argv[i]);
        if (iter == pmap_.end()) {
            std::cerr << "Unknown argument:" << argv[i] << std::endl;
            return false;
        }

        auto& p = iter->second;
        int parsed = p.f(p.item, argc - i, &argv[i]);

        if (parsed <= 0) {
            std::cerr << "Failed to parse argument: " << argv[i] << std::endl;
            return false;
        }

        i += parsed;
    }

    if (!settings.is_cpu && !settings.is_cublas && !settings.is_cudss &&
            !settings.is_cusolver && !settings.is_mkl_dss &&
            !settings.is_mkl_lapack) {
        settings.is_cpu = true;
        settings.is_cublas = true;
        settings.is_cudss = true;
        settings.is_cusolver = true;
        settings.is_mkl_dss = true;
        settings.is_mkl_lapack = true;
    }

    return true;
}

int Settings::parseUnsigned(ItemType& item, int argc, char* argv[])
{
    if (argc < 2)
        return -1;

    try {
        int val = std::stoi(argv[1]);
        if (val < 0)
            return -1;

        unsigned *res = std::get<unsigned*>(item);
        *res = val;
    } catch(...) {
        return -1;
    }

    return 2;
}

int Settings::parseString(ItemType& item,  int argc, char* argv[])
{
    if (argc < 2)
        return -1;

    try {
        std::string *res = std::get<std::string*>(item);
        *res = argv[1];
    } catch(...) {
        return -1;
    }

    return 2;
}

int Settings::parseBool(ItemType& item, int argc, char* argv[])
{
    try {
        bool *res = std::get<bool*>(item);
        *res = true;
    } catch(...) {
        return -1;
    }

    return 1;
}
