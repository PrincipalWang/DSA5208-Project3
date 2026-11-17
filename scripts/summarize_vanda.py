import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import tarfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--single", required=True)
parser.add_argument("--two", required=True)
parser.add_argument("--pilot", required=True)
parser.add_argument("--source", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
source = Path(args.source)
target = Path(args.output)
target.mkdir(parents=True, exist_ok=True)
figdir = target / "report_figures"
figdir.mkdir(exist_ok=True)

def load_csv(path):
    with Path(path).open(newline="") as file:
        return list(csv.DictReader(file))

integers = {"n", "iterations", "ranks", "threads", "workers", "repeats"}
def typed(row):
    result = {}
    for key, value in row.items():
        if value in ("", None):
            result[key] = None
        else:
            try:
                result[key] = int(value) if key in integers else float(value)
            except ValueError:
                result[key] = value
    return result

facts = {"runs": {}, "validation": {}}
raw_modes = {}
models = set()
for name, path, nodes, counts in [
    ("single", Path(args.single), 1, {"strong":189, "weak":63, "profile":27}),
    ("two", Path(args.two), 2, {"strong":171, "weak":57, "profile":21}),
]:
    results = path / "results/vanda"
    summaries = [typed(row) for row in load_csv(results / "summary.csv")]
    hosts = json.loads((results / "hosts.json").read_text())
    if len({h["host"] for h in hosts}) != nodes:
        raise RuntimeError("Host topology does not match the claimed experiment")
    for host in hosts:
        for line in host["cpu_details"].splitlines():
            if line.startswith("Model name:"):
                models.add(line.split(":", 1)[1].strip())
    all_raw = {}
    for mode, expected in counts.items():
        rows = load_csv(results / (mode + ".csv"))
        if len(rows) != expected:
            raise RuntimeError(name + " " + mode + " has incomplete measurements")
        groups = defaultdict(list)
        for row in rows:
            if not math.isfinite(float(row["solve_seconds"])) or float(row["iteration_error_max"]) >= 1e-9:
                raise RuntimeError("Invalid timing or analytical correctness check")
            groups[(row["n"], row["iterations"], row["ranks"], row["threads"], row["communication"])].append(int(row["repeat"]))
        if any(sorted(repeats) != [1,2,3] for repeats in groups.values()):
            raise RuntimeError("Each configuration must have three distinct repeats")
        all_raw[mode] = rows
    metadata_files = list(results.glob("*.environment.json"))
    for file in metadata_files:
        metadata = json.loads(file.read_text())
        for name_in_source, digest in metadata["source_sha256"].items():
            if hashlib.sha256((source / name_in_source).read_bytes()).hexdigest() != digest:
                raise RuntimeError("Numerical source differs from the measured source")
    serial = {str(row["n"]): row["median_seconds"] for row in summaries
              if row["mode"] == "strong" and row["communication"] == "none"}
    best = {}
    for n in (1024, 2048, 4096):
        choices = [r for r in summaries if r["mode"] == "strong" and r["n"] == n
                   and r["workers"] == nodes * 8 and r["communication"] != "none"]
        best[str(n)] = min(choices, key=lambda r: r["median_seconds"])
    job_id = path.name.split("vanda_single_", 1)[-1] if name == "single" else path.name.split("vanda_two_nodes_", 1)[-1]
    facts["runs"][name] = dict(job_id=job_id, nodes=nodes, cores=nodes*8, hosts=hosts,
        summary=summaries, serial=serial, best=best, records=counts)
    raw_modes[name] = all_raw
    dest = target / "results" / ("vanda_" + name)
    shutil.copytree(results, dest, dirs_exist_ok=True)
    for filename in ("environment.txt", "pbs_record.txt"):
        shutil.copy2(path / filename, dest / filename)
    final_status = subprocess.check_output(["qstat", "-fx", job_id], text=True)
    (dest / "pbs_final.txt").write_text(final_status)
    if "Exit_status = 0" not in final_status:
        raise RuntimeError("A reported final job did not exit successfully")

pilot_rows = load_csv(Path(args.pilot) / "results/pilot/validation.csv")
distributed = load_csv(Path(args.two) / "results/vanda/distributed_validation.csv")
if len(pilot_rows) != 42 or len(distributed) != 8 or any(r["status"] != "PASS" for r in pilot_rows + distributed):
    raise RuntimeError("Pointwise validation record is incomplete")
shutil.copy2(Path(args.pilot) / "results/pilot/validation.csv", target / "results/validation_single.csv")
facts["validation"] = dict(single_checks=42, distributed_checks=8,
    max_pointwise_error=max([float(r["max_vs_python"]) for r in pilot_rows + distributed]),
    max_analytical_error=max(float(r["iteration_error_max"]) for run in raw_modes.values() for rows in run.values() for r in rows),
    convergence=[dict(n=int(r["n"]), solution_error_max=float(r["solution_error_max"]),
        observed_order=float(r["observed_order"]) if r["observed_order"] else None)
        for r in load_csv(Path(args.single) / "results/vanda/convergence.csv")])
facts["hardware_description"] = "; ".join(sorted(models)) + "; hyper-threading disabled on the measured nodes."

comparisons = []
for name, ranks, threads in [("single",8,1),("single",4,2),("two",16,1),("two",8,2)]:
    summary = facts["runs"][name]["summary"]
    times = {r["communication"]: r["median_seconds"] for r in summary
        if r["mode"] == "strong" and r["n"] == 4096 and r["ranks"] == ranks and r["threads"] == threads}
    profile = raw_modes[name]["profile"]
    api = {comm: statistics.median(float(r["mpi_call_seconds_max"]) for r in profile
           if int(r["ranks"]) == ranks and int(r["threads"]) == threads and r["communication"] == comm)
           for comm in ("blocking", "overlap")}
    comparisons.append(dict(nodes=facts["runs"][name]["nodes"], ranks=ranks, threads=threads,
        blocking_s=times["blocking"], overlap_s=times["overlap"],
        change_pct=100*(times["blocking"]-times["overlap"])/times["blocking"],
        mpi_blocking_s=api["blocking"], mpi_overlap_s=api["overlap"]))
facts["overlap_comparisons"] = comparisons
(target / "report_data.json").write_text(json.dumps(facts, indent=2))

plt.rcParams.update({"font.size":8.5, "axes.spines.top":False, "axes.spines.right":False,
                     "axes.titlesize":9, "axes.labelsize":8.5, "legend.fontsize":8})
series = [("single", 1, "1 node: MPI", "#186a8b", "-"),
          ("single", 2, "1 node: 2 threads/rank", "#186a8b", "--"),
          ("two", 1, "2 nodes: MPI", "#c47532", "-"),
          ("two", 2, "2 nodes: 2 threads/rank", "#c47532", "--")]
ticks = [1,2,4,8,16]
fig, axes = plt.subplots(2, 3, figsize=(7.05,4.1), sharex=True)
for col, n in enumerate((1024,2048,4096)):
    for name, strategy, label, color, style in series:
        points = [r for r in facts["runs"][name]["summary"]
            if r["mode"] == "strong" and r["n"] == n and r["communication"] == "blocking"
            and r["threads"] == strategy]
        points.sort(key=lambda r:r["workers"])
        x = [r["workers"] for r in points]
        axes[0,col].plot(x,[r["speedup"] for r in points],style,color=color,marker="o",ms=3,label=label)
        axes[1,col].plot(x,[r["efficiency"] for r in points],style,color=color,marker="o",ms=3)
    axes[0,col].plot([1,16],[1,16],":",color="0.7",lw=1)
    axes[1,col].axhline(1,color="0.7",ls=":",lw=1)
    axes[0,col].set_title("N="+str(n)+", K="+str(1000 if n<4096 else 300))
    for row in range(2):
        axes[row,col].set_xscale("log",base=2)
        axes[row,col].set_xticks(ticks)
        axes[row,col].set_xticklabels(ticks)
        axes[row,col].grid(alpha=.2)
    axes[1,col].set_xlabel("Physical cores")
axes[0,0].set_ylabel("Speedup")
axes[1,0].set_ylabel("Efficiency")
fig.legend(*axes[0,0].get_legend_handles_labels(), loc="upper center",ncol=2,frameon=False)
fig.tight_layout(rect=(0,0,1,.83),w_pad=1.2,h_pad=1)
fig.savefig(figdir/"strong_summary.png",dpi=240,bbox_inches="tight")
plt.close(fig)

fig, axes = plt.subplots(1,2,figsize=(7.05,2.5))
for name,strategy,label,color,style in series:
    points=[r for r in facts["runs"][name]["summary"] if r["mode"]=="weak"
        and r["communication"]=="blocking" and r["threads"]==strategy]
    points.sort(key=lambda r:r["workers"])
    x=[r["workers"] for r in points]
    axes[0].plot(x,[r["median_seconds"] for r in points],style,color=color,marker="o",ms=3,label=label)
    axes[1].plot(x,[r["weak_efficiency"] for r in points],style,color=color,marker="o",ms=3)
for ax,ylabel in zip(axes,["Runtime (s)","Weak efficiency"]):
    ax.set_xscale("log",base=2); ax.set_xticks(ticks); ax.set_xticklabels(ticks)
    ax.set_xlabel("Physical cores"); ax.set_ylabel(ylabel); ax.grid(alpha=.2)
axes[1].axhline(1,ls=":",color=".7")
fig.legend(*axes[0].get_legend_handles_labels(),loc="upper center",ncol=2,frameon=False)
fig.tight_layout(rect=(0,0,1,.76))
fig.savefig(figdir/"weak_summary.png",dpi=240,bbox_inches="tight")
plt.close(fig)

grid=load_csv(Path(args.single)/"results/vanda/converged_grid.csv")
n=max(int(r["row"]) for r in grid)
values=np.array([float(r["value"]) for r in grid]).reshape(n,n)
h=1/(n+1)
x=np.arange(1,n+1)*h
xx,yy=np.meshgrid(x,x)
exact=np.sin(np.pi*xx)*np.sin(np.pi*yy)+.25*np.sin(2*np.pi*xx)*np.sin(3*np.pi*yy)
fig,axes=plt.subplots(1,2,figsize=(7.05,2.25))
for ax,array,title in zip(axes,[values,np.abs(values-exact)],["Numerical solution, N=63","Absolute error vs continuous solution"]):
    im=ax.imshow(array,origin="lower",extent=(h/2,1-h/2,h/2,1-h/2),cmap="viridis")
    ax.set_xlim(0,1);ax.set_ylim(0,1);ax.set_xlabel("x");ax.set_ylabel("y");ax.set_title(title)
    fig.colorbar(im,ax=ax,fraction=.046,pad=.04)
fig.tight_layout()
fig.savefig(figdir/"validation.png",dpi=240,bbox_inches="tight")
plt.close(fig)

for name in ("src","scripts"):
    destination=target/name
    destination.mkdir(exist_ok=True)
    for path in (source/name).glob("*"):
        if path.is_file() and path.suffix in (".py",".cpp",".hpp") and path.name != "finalize_checkpoint.py":
            shutil.copy2(path,destination/path.name)
shutil.copy2(source/"Makefile",target/"Makefile")
(target/"jobs").mkdir(exist_ok=True)
(target/"logs").mkdir(exist_ok=True)
for old,new in [("vanda_single_verified.pbs","vanda_single.pbs"),("vanda_two_nodes_verified.pbs","vanda_two_nodes.pbs")]:
    original=(source/"jobs"/old).read_text()
    (target/"logs"/old).write_text(original)
    portable=original.replace('RUN="/scratch/e1538476/dsa5208_project3_20261116/$NAME"',
        'RUN_ROOT="${PROJECT3_RUN_ROOT:-/scratch/$USER/project3_runs}"\nRUN="$RUN_ROOT/$NAME"')
    portable=portable.replace('BACKUP="/nfs/home/svu/e1538476/dsa5208_project3_20261116/runs/$NAME"',
        'BACKUP_ROOT="${PROJECT3_BACKUP_ROOT:-$HOME/project3_backups}"\nBACKUP="$BACKUP_ROOT/$NAME"')
    (target/"jobs"/new).write_text(portable)
for name in ("single_serialized.log","two_serialized.log","pilot.log","scheduling_note.txt"):
    shutil.copy2(source/"results/vanda"/name,target/"logs"/name)

readme="""# DSA5208 Project 3 - MPI + OpenMP Jacobi iteration

Wang Chen - A0326645Y

## Build and quick run
Use Linux, GCC with C++17/OpenMP, Open MPI and make. Numerical programs need no
Python packages. Validation/benchmark scripts need Python >=3.8. Plotting needs
NumPy and Matplotlib. The examples below require an allocation with at least
8 CPU cores. On Vanda, request a compute allocation before building or running.

    make -j2
    ./bin/jacobi_serial --n 1024 --iterations 1000
    export OMP_PLACES=cores OMP_PROC_BIND=close OMP_DYNAMIC=FALSE
    mpiexec -n 4 --map-by ppr:4:node:PE=2 --bind-to core ./bin/jacobi_parallel --n 1024 --iterations 1000 --threads 2 --comm blocking
    mpiexec -n 4 --map-by ppr:4:node:PE=2 --bind-to core ./bin/jacobi_parallel --n 1024 --iterations 1000 --threads 2 --comm overlap

N counts interior grid points per axis. The programs print a CSV header and one
result row. --output path.csv additionally writes the full interior grid for small
correctness experiments. Runs perform a fixed iteration count; the large-grid
benchmark is not claimed to have reached the steady solution.

## Validation
    python3 scripts/validate.py --output results/new_validation.csv
    python3 scripts/check_convergence.py --directory results/new_convergence

The pointwise script checks independent Python, serial and parallel results.
The Vanda driver also validates communication across two distinct physical hosts.
iteration_error_max checks the exact finite-iteration sine-mode reference;
solution_error_max checks the continuous steady solution. They differ by design.

## Reproduce on Vanda
Keep NUS VPN connected when off campus. Submit from this project directory.
Compile and benchmark on compute nodes, following the Vanda welcome instructions.

    one=$(qsub jobs/vanda_single.pbs)
    qsub -W depend=afterok:$one jobs/vanda_two_nodes.pbs
    qstat -u "$USER"

The scripts use auto_free without a project name and load gompi/2024a
(GCC 13.3.0 and Open MPI 5.0.3). Each node reserves 8 CPU cores and 8 GB RAM.
The two-node job uses place=scatter. RUN_ROOT and BACKUP_ROOT can be configured
with PROJECT3_RUN_ROOT and PROJECT3_BACKUP_ROOT in the scripts or PBS environment.
Each job builds into a distinct scratch directory and backs up its results to home.
The plotting phase loads matplotlib/3.10.3-gfbf-2025a after measurements finish.

Strong scaling uses 1000 iterations at N=1024/2048 and 300 at N=4096.
Weak scaling uses 1000 iterations and N=round(1024*sqrt(ranks*threads)).
Every case has a warm-up and 3 shuffled repetitions. Profiled runs are separate.
The two jobs run sequentially to avoid contention between our own experiments.
They still use shared cluster nodes.

## Results and interpretation
results/vanda_single and results/vanda_two contain raw repetitions, summaries,
environment/source/binary hashes, allocated CPU masks, rank-binding checks,
convergence data and final PBS records. logs/ contains original submission scripts
and final job logs. Earlier simultaneous runs were cancelled and excluded.
report_figures/ contains the compact report charts; report_data.json contains
the verified values used by the report.

solve_seconds times the iteration loop and takes the maximum across MPI ranks.
It excludes initialization, MPI startup, gathering and checks.
Speedup uses the serial median from the same allocation; efficiency divides it
by ranks*threads. mpi_call_seconds_max measures time inside halo MPI calls and
waits. It is diagnostic API time, not total network time or an additive wall-time
decomposition. No trained ML model is required for this project.

The README and numerical sources are sufficient for independent execution.
The PDF is a review draft: review the analysis, authorship and AI-use statement
before submitting. No credentials or SSH keys belong in the submission.
"""
(target/"README.md").write_text(readme)
manifest={"single_job":facts["runs"]["single"]["job_id"],"two_node_job":facts["runs"]["two"]["job_id"],
          "pointwise_checks":50,"timed_records":sum(sum(r["records"].values()) for r in facts["runs"].values()),
          "source_sha256":{str(p.relative_to(target)):hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sorted((target/"src").glob("*")) if p.is_file()}}
(target/"experiment_manifest.json").write_text(json.dumps(manifest,indent=2))
archive=target.parent/"project3_vanda_data.tar.gz"
with tarfile.open(archive,"w:gz") as tar:
    tar.add(target,arcname="project3")
print(json.dumps(manifest,indent=2))
print("POSTPROCESS_COMPLETE",archive)
