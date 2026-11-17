#include "common.hpp"

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        if (options.help) {
            print_help();
            return 0;
        }
        if (options.threads > 1) throw std::runtime_error("The serial program uses one thread");
        Grid grid(options.n, options.n, 1);
        const auto start = std::chrono::steady_clock::now();
        for (int step = 0; step < options.iterations; ++step) {
            for (int i = 1; i <= grid.rows; ++i) {
                update_row(grid, grid.current.data(), grid.next.data(), i);
            }
            grid.current.swap(grid.next);
        }
        const double seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();
        const Metrics metrics = measure(grid, options.iterations);
        if (!options.output.empty()) {
            std::vector<double> values(static_cast<std::size_t>(options.n) * options.n);
            for (int i = 1; i <= options.n; ++i) {
                std::copy_n(grid.current.data() + i * grid.stride + 1, options.n,
                            values.data() + static_cast<std::size_t>(i - 1) * options.n);
            }
            write_grid(options.output, options.n, values);
        }
        print_result(options, "serial", "none", 1, 1, seconds, 0.0, metrics);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
