# DSA5208 Project 3

Jacobi iteration for the 2D Poisson equation, implemented in C++17 using
MPI and OpenMP. `src/` contains the serial and parallel solvers, `scripts/`
contains the experiments and analysis, and `jobs/` contains Vanda job scripts.

## Build and run

Use Linux, GCC, Open MPI and Make. On Vanda, compile and run inside a compute
allocation. These parallel examples need at least 8 allocated CPU cores.

```bash
module load gompi/2024a
make -j2
./bin/jacobi_serial --n 1024 --iterations 1000
export OMP_PLACES=cores OMP_PROC_BIND=close OMP_DYNAMIC=FALSE
mpiexec -n 4 --map-by ppr:4:node:PE=2 --bind-to core ./bin/jacobi_parallel --n 1024 --iterations 1000 --threads 2 --comm blocking
mpiexec -n 4 --map-by ppr:4:node:PE=2 --bind-to core ./bin/jacobi_parallel --n 1024 --iterations 1000 --threads 2 --comm overlap
```

`--n` gives the interior grid size per axis. `--iterations` sets the number of
Jacobi updates. The programs print CSV results; `--output grid.csv` also saves
the grid for small validation cases.

## Check correctness

After building, run these inside the compute allocation. Python 3.8+ is required.
Choose fresh output paths when repeating the checks.

```bash
python3 scripts/validate.py --output results/new_validation.csv
python3 scripts/check_convergence.py --directory results/new_convergence
```

## Run the experiments on Vanda

Submit from this directory on the login node. The jobs build and run on compute
nodes, requesting 8 CPU cores and 8 GB RAM per node. Plotting uses NumPy and
Matplotlib 3.9+; the job scripts load the Vanda plotting module automatically.

For the initial strong/weak scaling experiments:

```bash
one=$(qsub jobs/vanda_single.pbs)
qsub -W depend=afterok:$one jobs/vanda_two_nodes.pbs
```

For the longer, seven-repeat supplementary experiments:

```bash
one=$(qsub jobs/supplement_single.pbs)
two=$(qsub -W depend=afterok:$one jobs/supplement_two.pbs)
qsub -W depend=afterok:$two -v SUPPLEMENT_SINGLE=$one,SUPPLEMENT_TWO=$two jobs/supplement_analysis.pbs
qstat -u "$USER"
```

Initial runs save working files under `/scratch/$USER/project3_runs` and backups
under `$HOME/project3_backups`. Supplementary runs use
`/scratch/$USER/project3_supplement_runs`; their PBS logs show the backup location.
Supplementary summaries and plots go under `$HOME/project3_supplement_results`.

The benchmarks use fixed iteration counts. `solve_seconds` measures the iteration
loop using the slowest MPI rank, excluding initialization and result gathering.
Existing measurements and the report are kept separately from this source folder.
