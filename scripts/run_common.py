import csv
import io
import math
import os
from pathlib import Path
import shlex
import sys
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def execute(command, threads=1, timeout=180):
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS=str(threads), OMP_DYNAMIC="FALSE",
                       OMP_PLACES="cores", OMP_PROC_BIND="close")
    result = subprocess.run(command, cwd=ROOT, env=environment, text=True,
                            capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(
            f"Command failed: {shlex.join(map(str, command))}\n"
            f"{result.stdout}\n{result.stderr}"
        )
    # Preserve launcher warnings, especially affinity or resource warnings.
    if result.stderr.strip():
        print(result.stderr.strip(), file=__import__("sys").stderr)
    return result.stdout


def mpi_command(ranks, threads=1, nodes=1, binding=False):
    command = shlex.split(os.environ.get("MPIEXEC", "mpiexec"))
    command += shlex.split(os.environ.get("MPIEXEC_FLAGS", ""))
    command += ["-n", str(ranks)]
    if binding:
        if ranks % nodes:
            raise ValueError("The number of ranks must be divisible by the node count")
        command += ["--map-by", f"ppr:{ranks // nodes}:node:PE={threads}",
                    "--bind-to", "core"]
    else:
        # Validation does not measure speed. Let OpenMP use the available cores.
        command += ["--bind-to", "none"]
    return command


def read_result(text):
    rows = list(csv.DictReader(io.StringIO(text)))
    if len(rows) != 1 or "iteration_error_max" not in rows[0]:
        raise RuntimeError(f"Unexpected program output:\n{text}")
    row = rows[0]
    for name in ("solve_seconds", "iteration_error_max", "solution_error_max",
                 "residual_max", "checksum"):
        if not math.isfinite(float(row[name])):
            raise RuntimeError(f"Non-finite result for {name}: {row}")
    if float(row["iteration_error_max"]) > 1e-9:
        raise RuntimeError(f"The analytical iteration check failed: {row}")
    return row


def append_result(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    if exists:
        with path.open(newline="", encoding="utf-8") as file:
            if next(csv.reader(file)) != list(row):
                raise RuntimeError(f"Existing CSV has a different schema: {path}")
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
