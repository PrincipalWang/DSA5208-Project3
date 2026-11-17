import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shlex
import sys

from benchmark import record_environment
from run_common import ROOT, append_result, execute, mpi_command, read_result
from validate import maximum_difference, read_grid, reference_grid


EXPECTED = {
    'src/common.hpp': '053af3e8d69dfdf3f24905a2f287b6383a8c2891e3570ec93fb67768b45519c6',
    'src/parallel.cpp': '73f830c2d575de0272366dd3f40dbef50edbb1b693cc7e545e77045303ed02bd',
    'src/serial.cpp': '163969e4001b20efd0b233eb1c01a439b135f57f91980a7d9d684cf74b2600ce',
}


def make_groups(nodes):
    groups = []
    if nodes == 1:
        groups.append(('baseline_2048', [
            dict(n=2048, ranks=1, threads=1, communication='none', requested_nodes=1),
            dict(n=2048, ranks=1, threads=1, communication='blocking', requested_nodes=1),
            dict(n=2048, ranks=1, threads=8, communication='blocking', requested_nodes=1),
        ]))
    groups.append(('baseline_4096', [
        dict(n=4096, ranks=1, threads=1, communication='none', requested_nodes=1),
        dict(n=4096, ranks=1, threads=1, communication='blocking', requested_nodes=1),
    ]))
    configs = [(8, 1), (4, 2), (2, 4), (1, 8)] if nodes == 1 else [(16, 1), (8, 2), (4, 4)]
    for ranks, threads in configs:
        groups.append((f'communication_{ranks}x{threads}', [
            dict(n=4096, ranks=ranks, threads=threads, communication=comm, requested_nodes=nodes)
            for comm in ('blocking', 'overlap')
        ]))
    return groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nodes', type=int, choices=(1, 2), required=True)
    parser.add_argument('--iterations', type=int, default=3000)
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--output', default='results/supplement')
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    if args.iterations < 1 or args.repeats < 3:
        parser.error('Positive iteration count and at least three repetitions required')
    groups = make_groups(args.nodes)
    plan = dict(nodes=args.nodes, cores=8 * args.nodes, iterations=args.iterations,
                repeats=args.repeats, measured_records=sum(len(c) for _, c in groups) * args.repeats,
                groups=groups, source_sha256=EXPECTED)
    for name, expected in EXPECTED.items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Numerical source changed: ' + name)
    if args.plan_only:
        print(json.dumps(plan, indent=2))
        return
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=False)
    (output / 'plan.json').write_text(json.dumps(plan, indent=2))
    os.environ.pop('JACOBI_BINDING', None)
    record_environment(output / 'environment.json', args)
    node_code = "import json,os,socket,subprocess; print(json.dumps(dict(host=socket.gethostname(),cpus=sorted(os.sched_getaffinity(0)),cpu_details=subprocess.check_output(['lscpu'],text=True))))"
    launcher = shlex.split(os.environ.get('MPIEXEC', 'mpiexec'))
    launcher += shlex.split(os.environ.get('MPIEXEC_FLAGS', ''))
    hosts = [json.loads(line) for line in execute(launcher + ['-n', str(args.nodes), '--map-by',
        'ppr:1:node', '--bind-to', 'none', sys.executable, '-c', node_code]).splitlines()
        if line.startswith('{')]
    if len(hosts) != args.nodes or len({h['host'] for h in hosts}) != args.nodes:
        raise RuntimeError('Unexpected physical host allocation')
    if any(len(h['cpus']) != 8 for h in hosts):
        raise RuntimeError('Exactly eight allowed cores per node are required')
    (output / 'hosts.json').write_text(json.dumps(hosts, indent=2))
    allowed = {h['host']: set(h['cpus']) for h in hosts}
    binding_code = "import json,os,socket; print(json.dumps(dict(host=socket.gethostname(),rank=int(os.environ['OMPI_COMM_WORLD_RANK']),cpus=sorted(os.sched_getaffinity(0)))))"
    bindings = []
    checked = set()
    for _, cases in groups:
        for case in cases:
            key = (case['ranks'], case['threads'], case['requested_nodes'])
            if key in checked:
                continue
            checked.add(key)
            ranks, threads, nodes = key
            rows = [json.loads(line) for line in execute(mpi_command(ranks, threads, nodes, True) +
                [sys.executable, '-c', binding_code], threads).splitlines() if line.startswith('{')]
            if len(rows) != ranks or len({r['host'] for r in rows}) != nodes:
                raise RuntimeError('Requested rank/host count was not used')
            used = {}
            for row in rows:
                cpus = set(row['cpus'])
                occupied = used.setdefault(row['host'], set())
                if len(cpus) != threads or not cpus <= allowed[row['host']] or cpus & occupied:
                    raise RuntimeError('Incorrect or overlapping rank binding')
                occupied.update(cpus)
            bindings.append(dict(ranks=ranks, threads=threads, nodes=nodes, bindings=rows))
    (output / 'rank_bindings.json').write_text(json.dumps(bindings, indent=2))
    print('HOSTS_AND_ALL_BINDINGS_VERIFIED', [(h['host'], h['cpus']) for h in hosts], flush=True)

    for n, iterations in [(4, 1), (17, 31)]:
        reference = reference_grid(n, iterations)
        for comm in ('blocking', 'overlap'):
            path = output / 'validation_grid.csv'
            result = read_result(execute(mpi_command(4, 2, args.nodes, True) + [
                str(ROOT / 'bin/jacobi_parallel'), '--n', str(n), '--iterations', str(iterations),
                '--threads', '2', '--comm', comm, '--output', str(path)], 2))
            error = maximum_difference(read_grid(path, n), reference)
            if not math.isfinite(error) or error > 1e-11:
                raise RuntimeError('Supplement pointwise validation failed')
            append_result(output / 'validation.csv', dict(n=n, iterations=iterations, ranks=4,
                threads=2, communication=comm, max_vs_python=error, status='PASS'))
    print('FOUR_POINTWISE_CHECKS_PASSED', flush=True)

    def run(case, iterations):
        ranks, threads, nodes = case['ranks'], case['threads'], case['requested_nodes']
        command = mpi_command(ranks, threads, nodes, True)
        if case['communication'] == 'none':
            command += [str(ROOT / 'bin/jacobi_serial')]
        else:
            command += [str(ROOT / 'bin/jacobi_parallel'), '--threads', str(threads),
                        '--comm', case['communication']]
        row = read_result(execute(command + ['--n', str(case['n']), '--iterations', str(iterations)],
                                  threads, timeout=1800))
        if int(row['ranks']) != ranks or int(row['threads']) != threads:
            raise RuntimeError('Unexpected rank/thread count')
        if float(row['solve_seconds']) <= 0:
            raise RuntimeError('Nonpositive elapsed time')
        return row

    warmups = [case for _, cases in groups for case in cases]
    random.Random(20261117).shuffle(warmups)
    for case in warmups:
        run(case, min(50, args.iterations))
    print('WARMUPS_COMPLETE', len(warmups), flush=True)
    completed = 0
    for repeat in range(1, args.repeats + 1):
        order = list(groups)
        random.Random(20261117 + repeat).shuffle(order)
        for group, cases in order:
            ordered_cases = list(cases)
            random.Random(731 * repeat + sum(map(ord, group))).shuffle(ordered_cases)
            for within_pair, case in enumerate(ordered_cases, 1):
                row = run(case, args.iterations)
                append_result(output / 'timings.csv', dict(study=group, repeat=repeat,
                    within_pair=within_pair, allocation_nodes=args.nodes,
                    timestamp_utc=datetime.now(timezone.utc).isoformat(),
                    requested_nodes=case['requested_nodes'], pbs_job_id=os.environ.get('PBS_JOBID', ''), **row))
                completed += 1
                print(f"[{completed}/{plan['measured_records']}] repeat={repeat} {group} "
                      f"{case['communication']} {case['ranks']}x{case['threads']}: "
                      f"{float(row['solve_seconds']):.6f} s", flush=True)
    with (output / 'timings.csv').open(newline='') as file:
        if len(list(csv.DictReader(file))) != plan['measured_records']:
            raise RuntimeError('Incomplete timings')
    print('SUPPLEMENT_TIMING_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
