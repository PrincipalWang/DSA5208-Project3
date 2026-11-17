import argparse
from collections import defaultdict
import csv
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys

from run_common import ROOT, append_result, execute, mpi_command, read_result
from validate import maximum_difference, read_grid, reference_grid

parser = argparse.ArgumentParser()
parser.add_argument("--nodes", type=int, choices=(1, 2), required=True)
parser.add_argument("--output", default="results/vanda")
args = parser.parse_args()
out = (ROOT / args.output).resolve()
out.mkdir(parents=True, exist_ok=True)
os.environ.pop("JACOBI_BINDING", None)

node_code = """import json,os,socket,subprocess; print(json.dumps(dict(host=socket.gethostname(),cpus=sorted(os.sched_getaffinity(0)),cpu_details=subprocess.check_output(['lscpu'],text=True))))"""
prefix = shlex.split(os.environ.get("MPIEXEC", "mpiexec"))
prefix += shlex.split(os.environ.get("MPIEXEC_FLAGS", ""))
node_command = prefix + ["-n", str(args.nodes), "--map-by", "ppr:1:node", "--bind-to", "none", sys.executable, "-c", node_code]
hosts = [json.loads(line) for line in execute(node_command).splitlines() if line.startswith("{")]
if len({h["host"] for h in hosts}) != args.nodes:
    raise RuntimeError("The requested number of distinct physical hosts was not used")
if any(len(h["cpus"]) != 8 for h in hosts):
    raise RuntimeError("Unexpected CPU allocation; stop before timing to check PBS binding")
(out / "hosts.json").write_text(json.dumps(hosts, indent=2))
print("Verified distinct hosts:", [(h["host"], h["cpus"]) for h in hosts], flush=True)

binding_code = """import json,os,socket; print(json.dumps(dict(host=socket.gethostname(),rank=int(os.environ['OMPI_COMM_WORLD_RANK']),cpus=sorted(os.sched_getaffinity(0)))))"""
bound = [json.loads(line) for line in execute(mpi_command(2 * args.nodes, 2, args.nodes, binding=True) +
         [sys.executable, "-c", binding_code], threads=2).splitlines() if line.startswith("{")]
allowed = {h["host"]: set(h["cpus"]) for h in hosts}
used = defaultdict(set)
for rank in bound:
    cpus = set(rank["cpus"])
    if len(cpus) != 2 or not cpus <= allowed[rank["host"]] or cpus & used[rank["host"]]:
        raise RuntimeError("Invalid or overlapping rank binding")
    used[rank["host"]].update(cpus)
(out / "rank_binding.json").write_text(json.dumps(bound, indent=2))

if args.nodes == 2:
    for n, iterations in [(4, 1), (17, 31)]:
        expected = reference_grid(n, iterations)
        for threads in (1, 2):
            for communication in ("blocking", "overlap"):
                grid_path = out / "distributed_grid.csv"
                command = mpi_command(4, threads, 2, binding=True) + [
                    str(ROOT / "bin/jacobi_parallel"), "--n", str(n), "--iterations", str(iterations),
                    "--threads", str(threads), "--comm", communication, "--output", str(grid_path)]
                row = read_result(execute(command, threads))
                error = maximum_difference(read_grid(grid_path, n), expected)
                if error > 1e-11:
                    raise RuntimeError("Distributed pointwise validation failed")
                append_result(out / "distributed_validation.csv", dict(n=n, iterations=iterations,
                    ranks=4, threads=threads, communication=communication,
                    max_vs_python=error, analytical_error=row["iteration_error_max"], status="PASS"))
    print("Eight cross-node pointwise checks passed", flush=True)

subprocess.run([sys.executable, str(ROOT / "scripts/check_convergence.py"),
                "--directory", str(out)], cwd=ROOT, check=True)

configs = ("1x1,2x1,1x2,4x1,2x2,1x4,8x1,4x2,2x4,1x8" if args.nodes == 1 else
           "2x1,4x1,8x1,16x1,2x2,4x2,8x2,2x4,4x4")
label = "vanda-" + str(args.nodes) + "-node"
common = [sys.executable, str(ROOT / "scripts/benchmark.py"), "--nodes", str(args.nodes),
          "--repeats", "3", "--configs", configs, "--label", label]
subprocess.run(common + ["--mode", "strong", "--sizes", "1024,2048", "--iterations", "1000",
                        "--output", str(out / "strong_small.csv")], cwd=ROOT, check=True)
subprocess.run(common + ["--mode", "strong", "--sizes", "4096", "--iterations", "300",
                        "--output", str(out / "strong_large.csv")], cwd=ROOT, check=True)
records = []
for name in ("strong_small.csv", "strong_large.csv"):
    with (out / name).open(newline="") as file:
        records.extend(csv.DictReader(file))
with (out / "strong.csv").open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)
subprocess.run(common + ["--mode", "weak", "--weak-base", "1024", "--iterations", "1000",
                        "--output", str(out / "weak.csv")], cwd=ROOT, check=True)
profile_configs = "8x1,4x2,2x4,1x8" if args.nodes == 1 else "16x1,8x2,4x4"
subprocess.run([sys.executable, str(ROOT / "scripts/benchmark.py"), "--nodes", str(args.nodes),
    "--mode", "strong", "--sizes", "4096", "--iterations", "300", "--repeats", "3",
    "--configs", profile_configs, "--profile", "--label", label + "-profile",
    "--output", str(out / "profile.csv")], cwd=ROOT, check=True)
print("VANDA_TIMING_COMPLETE", flush=True)
