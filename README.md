# DSA5208 Project 3: MPI + OpenMP Jacobi solver

C++17 serial and hybrid MPI/OpenMP implementations of Jacobi iteration for the
two-dimensional Poisson equation, with zero Dirichlet boundaries. The parallel
solver uses row decomposition and supports blocking halo exchange (`blocking`)
and nonblocking exchange overlapped with interior updates (`overlap`).

## Requirements

- Linux, GNU Make, a C++17 compiler with OpenMP, and Open MPI.
- Python 3.8 or later for validation and experiment drivers.
- NumPy and Matplotlib 3.9 or later for plotting and supplementary analysis.

The Vanda experiments used GCC 13.3.0 / Open MPI 5.0.3 (`gompi/2024a`).
Compile and run numerical experiments on compute nodes, not the Vanda login node.

## Build and run

Run from the repository root inside a compute allocation with at least 8 CPU cores:

```bash
module load gompi/2024a  # Vanda only
make -j2
./bin/jacobi_serial --n 1024 --iterations 1000
export OMP_PLACES=cores OMP_PROC_BIND=close OMP_DYNAMIC=FALSE
mpiexec -n 4 --map-by ppr:4:node:PE=2 --bind-to core ./bin/jacobi_parallel --n 1024 --iterations 1000 --threads 2 --comm blocking
mpiexec -n 4 --map-by ppr:4:node:PE=2 --bind-to core ./bin/jacobi_parallel --n 1024 --iterations 1000 --threads 2 --comm overlap
```

`--n` is the number of interior grid points per axis. Both programs output a CSV
header and result row. `--output path.csv` also writes the interior grid, useful
for small correctness checks. Measurements use a fixed iteration count; large-grid
timings do not imply convergence to the steady solution.

## Correctness checks

After building, run inside the compute allocation:

```bash
python3 scripts/validate.py --output results/new_validation.csv
python3 scripts/check_convergence.py --directory results/new_convergence
```

Validation compares an independent Python reference, the serial C++ solver, and
both parallel communication modes. The convergence check compares against the
manufactured continuous solution on successively finer grids. Choose new output
paths when repeating checks.

## Full Vanda experiments

Submit from the repository root on the Vanda login node:

```bash
one=$(qsub jobs/vanda_single.pbs)
qsub -W depend=afterok:$one jobs/vanda_two_nodes.pbs
qstat -u "$USER"
```

The PBS scripts request one or two nodes, 8 CPU cores and 8 GB RAM per node,
using `auto_free`. Two-node placement is scattered across physical nodes.
They build and benchmark on compute nodes, then generate plots using the
`matplotlib/3.10.3-gfbf-2025a` module. Each job uses a separate scratch directory
and backs up results under `$HOME/project3_backups` by default. Optional
`PROJECT3_RUN_ROOT` and `PROJECT3_BACKUP_ROOT` override these locations.

Strong scaling fixes the problem size; weak scaling increases the grid with the
worker count. Initial experiments use three shuffled repetitions. The follow-up
study uses 3000 iterations, seven repetitions, adjacent randomized communication
pairs, and matched serial/MPI baseline controls.

To submit the follow-up from the login node:

```bash
cd supplementary
python3 scripts/submit_supplement.py --plan-only
python3 scripts/submit_supplement.py
```

`--plan-only` prints the plan without submitting jobs. Actual submission schedules
single-node, two-node and analysis jobs sequentially, with unique log/output paths.
See [supplementary/README.md](supplementary/README.md) for details.

## Files and results

- `src/`: numerical C++ sources; `Makefile`: serial and parallel builds.
- `scripts/`: correctness checks, experiment drivers, CPU binding and analysis.
- `jobs/`: Vanda PBS templates for the initial study.
- `supplementary/`: unchanged numerical sources and follow-up scripts/PBS templates.

`solve_seconds` measures the iteration loop, taking the maximum across MPI ranks;
it excludes initialization, MPI startup and result gathering. Speedup uses the
serial median from the same allocation; efficiency divides speedup by the number
of MPI ranks times threads per rank. Shared Vanda nodes can cause variability.
Nonblocking communication did not show a stable improvement in the follow-up.

This repository contains source code and run instructions. Existing experiment
records, cluster metadata and the PDF report are retained separately. New runs
generate their own measurements, metadata, summaries and plots.
