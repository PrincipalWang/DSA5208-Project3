import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import socket
import subprocess
import time

from run_common import ROOT, append_result, execute, mpi_command, read_result


def capture(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=15)
        return result.stdout + result.stderr
    except (OSError, subprocess.TimeoutExpired) as error:
        return str(error)


def record_environment(path, args):
    nodefile = os.environ.get("PBS_NODEFILE")
    metadata = dict(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        hostname=socket.gethostname(), platform=platform.platform(),
        cpu_count=os.cpu_count(), arguments=vars(args),
        cpu_details=capture(["lscpu"]), compiler=capture(["g++", "--version"]),
        mpi_compiler=capture(["mpicxx", "--showme"]),
        mpi_version=capture(["mpiexec", "--version"]),
        pbs_job_id=os.environ.get("PBS_JOBID"),
        pbs_nodefile=Path(nodefile).read_text() if nodefile and Path(nodefile).exists() else None,
        binding=os.environ.get("JACOBI_BINDING", "native-openmpi"), allowed_cpus=sorted(os.sched_getaffinity(0)),
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted((ROOT / "src").glob("*"))},
        binary_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted((ROOT / "bin").glob("jacobi_*")) if p.is_file()},
    )
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Fixed-work Jacobi strong/weak scaling")
    parser.add_argument("--mode", choices=("strong", "weak"), default="strong")
    parser.add_argument("--sizes", default="1024,2048,4096")
    parser.add_argument("--weak-base", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--configs", default="1x1,2x1,1x2,4x1,2x2,1x4,8x1,4x2,2x4,1x8,16x1,8x2,4x4,1x16")
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--label", default="local-not-for-report")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--output", default="results/local/strong.csv")
    args = parser.parse_args()
    sizes = [int(value) for value in args.sizes.split(",")]
    configs = [tuple(map(int, value.split("x"))) for value in args.configs.split(",")]
    if args.iterations < 1 or args.repeats < 1 or args.nodes < 1 or args.weak_base < 4:
        parser.error("Iterations, repeats and nodes must be positive; weak-base must be >= 4")
    if min(sizes) < 4 or any(len(c) != 2 or min(c) < 1 for c in configs):
        parser.error("Invalid grid size or rank/thread configuration")
    if any(ranks % args.nodes for ranks, _ in configs):
        parser.error("Each rank count must be divisible by --nodes")
    output = (ROOT / args.output).resolve()
    metadata_path = output.with_suffix(".environment.json")
    if output.exists() or metadata_path.exists():
        parser.error("Choose a new output path; previous experiments are preserved")
    version = capture(["mpiexec", "--version"])
    if "Open MPI" not in version and "OpenRTE" not in version:
        parser.error("This affinity setup requires Open MPI. Check the Vanda MPI module first.")
    output.parent.mkdir(parents=True, exist_ok=True)
    record_environment(metadata_path, args)
    cases = []
    if args.mode == "strong":
        cases.extend((n, 1, 1, "none") for n in sizes)
        for n in sizes:
            for ranks, threads in configs:
                cases.extend((n, ranks, threads, comm) for comm in ("blocking", "overlap"))
    else:
        cases.append((args.weak_base, 1, 1, "none"))
        for ranks, threads in configs:
            n = round(args.weak_base * (ranks * threads) ** 0.5)
            cases.extend((n, ranks, threads, comm) for comm in ("blocking", "overlap"))
    if any(ranks > n for n, ranks, _, _ in cases):
        parser.error("Each grid must have at least one interior row per rank")

    def run_case(case, iterations):
        n, ranks, threads, communication = case
        if communication == "none":
            # The serial baseline is also pinned to one physical core.
            command = mpi_command(1, 1, 1, binding=True) + [str(ROOT / "bin/jacobi_serial")]
        else:
            command = mpi_command(ranks, threads, args.nodes, binding=True) + [
                str(ROOT / "bin/jacobi_parallel"), "--threads", str(threads),
                "--comm", communication
            ]
        command += ["--n", str(n), "--iterations", str(iterations)]
        if args.profile:
            command.append("--profile")
        row = read_result(execute(command, threads, timeout=1800))
        if int(row["ranks"]) != ranks or int(row["threads"]) != threads:
            raise RuntimeError("The requested rank/thread count was not used")
        return row

    order = list(cases)
    random.Random(5208).shuffle(order)
    for case in order:
        run_case(case, min(20, args.iterations))
    total = len(cases) * args.repeats
    completed = 0
    for repeat in range(1, args.repeats + 1):
        random.Random(5208 + repeat).shuffle(order)
        for case in order:
            row = run_case(case, args.iterations)
            row = dict(label=args.label, mode=args.mode, repeat=repeat,
                       requested_nodes=args.nodes if case[3] != "none" else 1,
                       hostname=socket.gethostname(),
                       pbs_job_id=os.environ.get("PBS_JOBID", ""),
                       weak_base=args.weak_base if args.mode == "weak" else "", **row)
            append_result(output, row)
            completed += 1
            print(f"[{completed}/{total}] {case}: {float(row['solve_seconds']):.6f} s", flush=True)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
