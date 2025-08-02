#include "report.h"

#include <filesystem>
#include <iostream>
#include <system_error>

namespace fs = std::filesystem;

namespace Report {

RedirectOut::RedirectOut()
{
}

RedirectOut::~RedirectOut()
{
    if (orig_) {
        std::cout.rdbuf(orig_);
    }
}

void RedirectOut::redirect(std::string dir)
{
    if (orig_) {
        return;
    }
    std::string fname = dir + "/console-output";
    out_.open(fname);
    orig_ = std::cout.rdbuf();
    std::cout.rdbuf(out_.rdbuf());
}

bool createReportDir(std::string path)
{
    if (fs::exists(path)) {
        std::cerr << "Path already exists: " << path << std::endl;
        return false;
    }

    bool ret = false;

    try {
        ret = fs::create_directory(path);
    } catch (std::system_error const& e) {
        std::cerr << e.what() << std::endl;

        return false;
    }

    return ret;
}

} // namespace Report
