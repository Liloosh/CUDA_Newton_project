#pragma once

#include <fstream>
#include <string>

namespace Report {

class RedirectOut {
public:
    RedirectOut();
    ~RedirectOut();

    void redirect(std::string dir);

private:
    std::ofstream out_;
    std::basic_streambuf<char, std::char_traits<char>>* orig_{nullptr};
};

bool createReportDir(std::string path);

}
