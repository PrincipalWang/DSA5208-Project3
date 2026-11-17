#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

struct Options {
    int n = 1024;
    int iterations = 500;
    int threads = 0;
    std::string communication = "blocking";
    std::string output;
    bool profile = false;
    bool help = false;
};

inline int parse_integer(const std::string& text) {
    std::size_t used = 0;
    long long value = std::stoll(text, &used);
    if (used != text.size() || value < 0 ||
        value > std::numeric_limits<int>::max() - 2) {
        throw std::runtime_error("Invalid non-negative integer: " + text);
    }
    return static_cast<int>(value);
}

inline Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help") options.help = true;
        else if (arg == "--profile") options.profile = true;
        else {
            if (i + 1 >= argc) throw std::runtime_error("Missing value for " + arg);
            std::string value = argv[++i];
            if (arg == "--n") options.n = parse_integer(value);
            else if (arg == "--iterations") options.iterations = parse_integer(value);
            else if (arg == "--threads") options.threads = parse_integer(value);
            else if (arg == "--comm") options.communication = value;
            else if (arg == "--output") options.output = value;
            else throw std::runtime_error("Unknown option: " + arg);
        }
    }
    if (options.n < 4) throw std::runtime_error("--n must be at least 4");
    if (options.communication != "blocking" && options.communication != "overlap") {
        throw std::runtime_error("--comm must be blocking or overlap");
    }
    if (!options.output.empty() &&
        static_cast<long long>(options.n) * options.n > std::numeric_limits<int>::max()) {
        throw std::runtime_error("The requested grid is too large for CSV gathering");
    }
    return options;
}

inline void print_help() {
    std::cout << "Options: --n N --iterations K --threads T "
                 "--comm blocking|overlap --profile --output grid.csv\n"
                 "N is the number of interior points on each axis (N >= 4).\n"
                 "All runs perform exactly K Jacobi iterations.\n";
}

struct Grid {
    int n;
    int rows;
    int first_row;
    std::size_t stride;
    double h;
    std::vector<double> current;
    std::vector<double> next;
    std::vector<double> rhs;

    Grid(int size, int local_rows, int first)
        : n(size), rows(local_rows), first_row(first),
          stride(static_cast<std::size_t>(size) + 2), h(1.0 / (size + 1.0)),
          current((static_cast<std::size_t>(rows) + 2) * stride, 0.0),
          next(current.size(), 0.0), rhs(current.size(), 0.0) {
        const double pi = std::acos(-1.0);
        for (int i = 1; i <= rows; ++i) {
            const double y = (first_row + i - 1) * h;
            for (int j = 1; j <= n; ++j) {
                const double x = j * h;
                rhs[static_cast<std::size_t>(i) * stride + j] = h * h * pi * pi *
                    (2.0 * std::sin(pi * x) * std::sin(pi * y) +
                     3.25 * std::sin(2.0 * pi * x) * std::sin(3.0 * pi * y));
            }
        }
    }
};

inline void update_row(const Grid& grid, const double* current, double* next, int i) {
    const std::size_t start = static_cast<std::size_t>(i) * grid.stride;
    for (int j = 1; j <= grid.n; ++j) {
        const std::size_t p = start + j;
        next[p] = 0.25 * (current[p - grid.stride] + current[p + grid.stride] +
                          current[p - 1] + current[p + 1] + grid.rhs[p]);
    }
}

struct Metrics {
    double iteration_error = 0.0;
    double solution_error = 0.0;
    double residual = 0.0;
    double checksum = 0.0;
};

inline double mode_amplitude(int n, int iterations, int a, int b, double weight) {
    const double pi = std::acos(-1.0);
    const double h = 1.0 / (n + 1.0);
    const double eigenvalue = 0.5 * (std::cos(a * pi * h) + std::cos(b * pi * h));
    return weight * h * h * pi * pi * (a * a + b * b) * 0.25 *
           (1.0 - std::pow(eigenvalue, iterations)) / (1.0 - eigenvalue);
}

inline Metrics measure(const Grid& grid, int iterations) {
    Metrics metrics;
    const double pi = std::acos(-1.0);
    const double a11 = mode_amplitude(grid.n, iterations, 1, 1, 1.0);
    const double a23 = mode_amplitude(grid.n, iterations, 2, 3, 0.25);
    for (int i = 1; i <= grid.rows; ++i) {
        const double y = (grid.first_row + i - 1) * grid.h;
        for (int j = 1; j <= grid.n; ++j) {
            const double x = j * grid.h;
            const double phi11 = std::sin(pi * x) * std::sin(pi * y);
            const double phi23 = std::sin(2.0 * pi * x) * std::sin(3.0 * pi * y);
            const std::size_t p = static_cast<std::size_t>(i) * grid.stride + j;
            const double value = grid.current[p];
            metrics.iteration_error = std::max(metrics.iteration_error,
                std::abs(value - a11 * phi11 - a23 * phi23));
            metrics.solution_error = std::max(metrics.solution_error,
                std::abs(value - phi11 - 0.25 * phi23));
            const double residual = (4.0 * value - grid.current[p - grid.stride] -
                grid.current[p + grid.stride] - grid.current[p - 1] -
                grid.current[p + 1] - grid.rhs[p]) / (grid.h * grid.h);
            metrics.residual = std::max(metrics.residual, std::abs(residual));
            metrics.checksum += value;
        }
    }
    return metrics;
}

inline void write_grid(const std::string& path, int n, const std::vector<double>& values) {
    std::ofstream file(path);
    if (!file) throw std::runtime_error("Cannot write grid: " + path);
    file << "row,column,value\n" << std::setprecision(17);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            file << i + 1 << ',' << j + 1 << ','
                 << values[static_cast<std::size_t>(i) * n + j] << '\n';
        }
    }
    if (!file) throw std::runtime_error("Failed while writing grid: " + path);
}

inline void print_result(const Options& options, const std::string& implementation,
                         const std::string& communication, int ranks, int threads,
                         double seconds, double mpi_seconds, const Metrics& metrics) {
    const double updates = static_cast<double>(options.n) * options.n * options.iterations;
    std::cout << "implementation,communication,n,iterations,ranks,threads,profile,"
                 "solve_seconds,mpi_call_seconds_max,updates_per_second,"
                 "iteration_error_max,solution_error_max,residual_max,checksum\n";
    std::cout << std::setprecision(17) << implementation << ',' << communication << ','
              << options.n << ',' << options.iterations << ',' << ranks << ',' << threads
              << ',' << options.profile << ',' << seconds << ',' << mpi_seconds << ','
              << (seconds > 0.0 ? updates / seconds : 0.0) << ',' << metrics.iteration_error
              << ',' << metrics.solution_error << ',' << metrics.residual << ','
              << metrics.checksum << '\n';
}
