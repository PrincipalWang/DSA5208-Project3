#include "common.hpp"
#define OMPI_SKIP_MPICXX 1
#define MPICH_SKIP_MPICXX 1
#include <mpi.h>
#include <omp.h>

void exchange_blocking(double* values, const Grid& grid, int up, int down) {
    MPI_Sendrecv(values + grid.stride, grid.n + 2, MPI_DOUBLE, up, 0,
                 values + (grid.rows + 1) * grid.stride, grid.n + 2, MPI_DOUBLE, down, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(values + grid.rows * grid.stride, grid.n + 2, MPI_DOUBLE, down, 1,
                 values, grid.n + 2, MPI_DOUBLE, up, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

void start_exchange(double* values, const Grid& grid, int up, int down,
                    MPI_Request* requests) {
    MPI_Irecv(values, grid.n + 2, MPI_DOUBLE, up, 1, MPI_COMM_WORLD, &requests[0]);
    MPI_Irecv(values + (grid.rows + 1) * grid.stride, grid.n + 2, MPI_DOUBLE, down, 0,
              MPI_COMM_WORLD, &requests[1]);
    MPI_Isend(values + grid.stride, grid.n + 2, MPI_DOUBLE, up, 0,
              MPI_COMM_WORLD, &requests[2]);
    MPI_Isend(values + grid.rows * grid.stride, grid.n + 2, MPI_DOUBLE, down, 1,
              MPI_COMM_WORLD, &requests[3]);
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0;
    int ranks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    try {
        const Options options = parse_options(argc, argv);
        if (options.help) {
            if (rank == 0) print_help();
            MPI_Finalize();
            return 0;
        }
        if (provided < MPI_THREAD_FUNNELED) {
            throw std::runtime_error("MPI_THREAD_FUNNELED support is required");
        }
        if (ranks > options.n) throw std::runtime_error("MPI ranks must not exceed N");
        omp_set_dynamic(0);
        if (options.threads > 0) omp_set_num_threads(options.threads);

        const int base = options.n / ranks;
        const int extra = options.n % ranks;
        const int rows = base + (rank < extra ? 1 : 0);
        const int first = rank * base + std::min(rank, extra) + 1;
        const int up = rank == 0 ? MPI_PROC_NULL : rank - 1;
        const int down = rank + 1 == ranks ? MPI_PROC_NULL : rank + 1;
        Grid grid(options.n, rows, first);
        double* current = grid.current.data();
        double* next = grid.next.data();
        MPI_Request requests[4];
        double mpi_seconds = 0.0;
        int threads = 1;

        MPI_Barrier(MPI_COMM_WORLD);
        const double start = MPI_Wtime();
        #pragma omp parallel shared(current, next, requests, mpi_seconds, threads)
        {
            #pragma omp master
            threads = omp_get_num_threads();
            for (int step = 0; step < options.iterations; ++step) {
                if (options.communication == "blocking") {
                    #pragma omp master
                    {
                        const double before = options.profile ? MPI_Wtime() : 0.0;
                        exchange_blocking(current, grid, up, down);
                        if (options.profile) mpi_seconds += MPI_Wtime() - before;
                    }
                    #pragma omp barrier
                    #pragma omp for schedule(static)
                    for (int i = 1; i <= rows; ++i) update_row(grid, current, next, i);
                } else {
                    #pragma omp master
                    {
                        const double before = options.profile ? MPI_Wtime() : 0.0;
                        start_exchange(current, grid, up, down, requests);
                        if (options.profile) mpi_seconds += MPI_Wtime() - before;
                    }
                    #pragma omp barrier
                    #pragma omp for schedule(static)
                    for (int i = 2; i < rows; ++i) update_row(grid, current, next, i);
                    #pragma omp master
                    {
                        const double before = options.profile ? MPI_Wtime() : 0.0;
                        MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
                        if (options.profile) mpi_seconds += MPI_Wtime() - before;
                    }
                    #pragma omp barrier
                    #pragma omp for schedule(static)
                    for (int side = 0; side < 2; ++side) {
                        if (side == 0 || rows > 1) {
                            update_row(grid, current, next, side == 0 ? 1 : rows);
                        }
                    }
                }
                #pragma omp master
                std::swap(current, next);
                #pragma omp barrier
            }
        }
        const double local_seconds = MPI_Wtime() - start;
        double seconds = 0.0;
        double maximum_mpi_seconds = 0.0;
        MPI_Reduce(&local_seconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&mpi_seconds, &maximum_mpi_seconds, 1, MPI_DOUBLE, MPI_MAX, 0,
                    MPI_COMM_WORLD);
        if (current != grid.current.data()) grid.current.swap(grid.next);

        // Refresh the final halo for residual evaluation, outside the timed loop.
        exchange_blocking(grid.current.data(), grid, up, down);
        const Metrics local = measure(grid, options.iterations);
        Metrics global;
        double local_maxima[3] = {local.iteration_error, local.solution_error, local.residual};
        double global_maxima[3] = {};
        MPI_Reduce(local_maxima, global_maxima, 3, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local.checksum, &global.checksum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        global.iteration_error = global_maxima[0];
        global.solution_error = global_maxima[1];
        global.residual = global_maxima[2];

        if (!options.output.empty()) {
            std::vector<double> packed(static_cast<std::size_t>(rows) * options.n);
            for (int i = 1; i <= rows; ++i) {
                std::copy_n(grid.current.data() + i * grid.stride + 1, options.n,
                            packed.data() + static_cast<std::size_t>(i - 1) * options.n);
            }
            std::vector<int> counts(ranks), offsets(ranks);
            for (int r = 0; r < ranks; ++r) {
                counts[r] = (base + (r < extra ? 1 : 0)) * options.n;
                offsets[r] = (r * base + std::min(r, extra)) * options.n;
            }
            std::vector<double> values;
            if (rank == 0) values.resize(static_cast<std::size_t>(options.n) * options.n);
            MPI_Gatherv(packed.data(), static_cast<int>(packed.size()), MPI_DOUBLE,
                        values.data(), counts.data(), offsets.data(), MPI_DOUBLE, 0,
                        MPI_COMM_WORLD);
            if (rank == 0) write_grid(options.output, options.n, values);
        }
        if (rank == 0) {
            print_result(options, threads == 1 ? "mpi" : "hybrid", options.communication,
                         ranks, threads, seconds, maximum_mpi_seconds, global);
        }
        MPI_Finalize();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Rank " << rank << ": " << error.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
