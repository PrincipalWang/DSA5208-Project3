import argparse
import csv
import math
from pathlib import Path
import tempfile

from run_common import ROOT, append_result, execute, mpi_command, read_result


def read_grid(path, n):
    with Path(path).open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if len(rows) != n * n:
        raise RuntimeError("The output grid has an incorrect number of points")
    for index, row in enumerate(rows):
        if (int(row["row"]), int(row["column"])) != (index // n + 1, index % n + 1):
            raise RuntimeError("The output grid is not in global row order")
    values = [float(row["value"]) for row in rows]
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("The output grid contains a non-finite value")
    return values


def reference_grid(n, iterations):
    # An independent Python stencil, without MPI or the C++ update routine.
    h = 1.0 / (n + 1)
    current = [[0.0] * (n + 2) for _ in range(n + 2)]
    rhs = [[0.0] * (n + 2) for _ in range(n + 2)]
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            x, y = j * h, i * h
            rhs[i][j] = h * h * math.pi ** 2 * (
                2 * math.sin(math.pi * x) * math.sin(math.pi * y) +
                3.25 * math.sin(2 * math.pi * x) * math.sin(3 * math.pi * y)
            )
    for _ in range(iterations):
        next_grid = [[0.0] * (n + 2) for _ in range(n + 2)]
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                next_grid[i][j] = 0.25 * (
                    current[i - 1][j] + current[i + 1][j] +
                    current[i][j - 1] + current[i][j + 1] + rhs[i][j]
                )
        current = next_grid
    return [current[i][j] for i in range(1, n + 1) for j in range(1, n + 1)]


def maximum_difference(left, right):
    return max(abs(a - b) for a, b in zip(left, right))


def main():
    parser = argparse.ArgumentParser(description="Pointwise Jacobi correctness checks")
    parser.add_argument("--output", default="results/local/validation.csv")
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if output.exists():
        raise SystemExit(f"Choose a new output path; existing results are preserved: {output}")
    cases = [(4, 0, 4), (4, 1, 4), (5, 7, 1), (7, 2, 4), (17, 31, 3), (31, 300, 4)]
    checks = 0
    with tempfile.TemporaryDirectory(prefix="jacobi-validation-") as temporary:
        temporary = Path(temporary)
        for n, iterations, ranks in cases:
            expected = reference_grid(n, iterations)
            serial_path = temporary / "serial.csv"
            serial = read_result(execute([
                str(ROOT / "bin/jacobi_serial"), "--n", str(n),
                "--iterations", str(iterations), "--output", str(serial_path)
            ]))
            serial_values = read_grid(serial_path, n)
            serial_error = maximum_difference(serial_values, expected)
            if serial_error > 1e-11:
                raise RuntimeError(f"Serial / Python mismatch: {serial_error}")
            append_result(output, dict(n=n, iterations=iterations, ranks=1, threads=1,
                                      communication="serial", max_vs_python=serial_error,
                                      max_vs_serial=0.0,
                                      analytical_error=serial["iteration_error_max"], status="PASS"))
            checks += 1
            for threads in (1, 2, 4):
                for communication in ("blocking", "overlap"):
                    grid_path = temporary / "parallel.csv"
                    command = mpi_command(ranks) + [
                        str(ROOT / "bin/jacobi_parallel"), "--n", str(n),
                        "--iterations", str(iterations), "--threads", str(threads),
                        "--comm", communication, "--output", str(grid_path)
                    ]
                    row = read_result(execute(command, threads))
                    if int(row["threads"]) != threads or int(row["ranks"]) != ranks:
                        raise RuntimeError("The requested process/thread count was not used")
                    values = read_grid(grid_path, n)
                    python_error = maximum_difference(values, expected)
                    serial_error = maximum_difference(values, serial_values)
                    if max(python_error, serial_error) > 1e-11:
                        raise RuntimeError(f"Parallel mismatch: {n}, {ranks}, {threads}, {communication}")
                    append_result(output, dict(n=n, iterations=iterations, ranks=ranks,
                        threads=threads, communication=communication,
                        max_vs_python=python_error, max_vs_serial=serial_error,
                        analytical_error=row["iteration_error_max"], status="PASS"))
                    checks += 1
            print(f"PASS: N={n}, iterations={iterations}, ranks={ranks}, threads=1/2/4", flush=True)
    print(f"All {checks} pointwise checks passed. Results: {output}")


if __name__ == "__main__":
    main()
