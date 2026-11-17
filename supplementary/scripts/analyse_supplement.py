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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read_csv(path):
    with path.open(newline='') as file:
        return list(csv.DictReader(file))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--single', required=True)
    parser.add_argument('--two', required=True)
    parser.add_argument('--project', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    project = Path(args.project)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'figures').mkdir()
    facts = dict(runs={}, comparisons=[], baseline_controls=[], limitations=[
        'Shared nodes; longer runs and seven repetitions do not establish exclusive-node reproducibility.',
        'Pairs are adjacent within each repetition, with deterministic random ordering; MPI progress was not directly profiled.',
        'All timings measure fixed-work iteration loops, not a tolerance-based end-to-end solve.',
        'New timings use 3000 iterations and cannot be compared directly with original 300/1000-iteration runtimes.',
    ])
    all_groups = {}
    total = 0
    for name, run_path, count, nodes in [('single', Path(args.single), 91, 1),
                                          ('two', Path(args.two), 56, 2)]:
        directory = run_path / 'results/supplement'
        plan = json.loads((directory / 'plan.json').read_text())
        rows = read_csv(directory / 'timings.csv')
        if len(rows) != count or plan['repeats'] != 7 or plan['iterations'] != 3000:
            raise RuntimeError('Unexpected or incomplete experiment plan')
        total += len(rows)
        groups = defaultdict(list)
        for row in rows:
            for field in ['solve_seconds', 'iteration_error_max']:
                if not math.isfinite(float(row[field])):
                    raise RuntimeError('Non-finite timing or numerical error')
            if float(row['solve_seconds']) <= 0 or float(row['iteration_error_max']) > 1e-9:
                raise RuntimeError('Invalid timing or failed correctness check')
            groups[(int(row['n']), int(row['ranks']), int(row['threads']), row['communication'])].append(row)
        if any(sorted(int(r['repeat']) for r in values) != list(range(1, 8)) for values in groups.values()):
            raise RuntimeError('Seven distinct repeats required for every configuration')
        metadata = json.loads((directory / 'environment.json').read_text())
        for filename, expected in metadata['source_sha256'].items():
            if hashlib.sha256((project / filename).read_bytes()).hexdigest() != expected:
                raise RuntimeError('Measured numerical source mismatch')
        hosts = json.loads((directory / 'hosts.json').read_text())
        if len({h['host'] for h in hosts}) != nodes:
            raise RuntimeError('Unexpected node topology')
        validation = read_csv(directory / 'validation.csv')
        if len(validation) != 4 or any(r['status'] != 'PASS' for r in validation):
            raise RuntimeError('Supplement pointwise checks incomplete')
        destination = output / 'results' / name
        shutil.copytree(directory, destination)
        for filename in ['environment.txt', 'pbs_record.txt']:
            shutil.copy2(run_path / filename, destination / filename)
        job_id = rows[0]['pbs_job_id']
        status = subprocess.check_output(['qstat', '-fx', job_id], text=True)
        if 'Exit_status = 0' not in status:
            raise RuntimeError('Measured PBS job did not finish successfully')
        (destination / 'pbs_final.txt').write_text(status)
        facts['runs'][name] = dict(nodes=nodes, job_id=job_id, hosts=hosts,
            iterations=3000, repeats=7, timed_records=len(rows), pointwise_checks=4,
            max_analytical_error=max(float(r['iteration_error_max']) for r in rows))
        all_groups[name] = groups
        for key, values in groups.items():
            n, ranks, threads, comm = key
            if n != 4096 or comm != 'blocking' or ranks * threads != nodes * 8:
                continue
            blocking = {int(r['repeat']): float(r['solve_seconds']) for r in values}
            overlap = {int(r['repeat']): float(r['solve_seconds']) for r in groups[(n, ranks, threads, 'overlap')]}
            changes = [100 * (blocking[i] - overlap[i]) / blocking[i] for i in range(1, 8)]
            serial = statistics.median(float(r['solve_seconds']) for r in groups[(n, 1, 1, 'none')])
            b, o = statistics.median(blocking.values()), statistics.median(overlap.values())
            facts['comparisons'].append(dict(topology=name, ranks=ranks, threads=threads,
                blocking_seconds=b, overlap_seconds=o,
                ratio_of_medians_reduction_pct=100 * (b-o)/b,
                median_paired_reduction_pct=statistics.median(changes),
                paired_reductions_pct=changes, overlap_wins=sum(c > 0 for c in changes),
                blocking_runs=[blocking[i] for i in range(1, 8)],
                overlap_runs=[overlap[i] for i in range(1, 8)],
                serial_baseline_seconds=serial, blocking_speedup=serial/b, overlap_speedup=serial/o))
    for n in [2048, 4096]:
        groups = all_groups['single']
        serial_runs = [float(r['solve_seconds']) for r in groups[(n, 1, 1, 'none')]]
        mpi_runs = [float(r['solve_seconds']) for r in groups[(n, 1, 1, 'blocking')]]
        row = dict(n=n, serial_runs=serial_runs, mpi_one_runs=mpi_runs,
                   serial_median=statistics.median(serial_runs), mpi_one_median=statistics.median(mpi_runs))
        row['serial_range'] = [min(serial_runs), max(serial_runs)]
        row['mpi_one_range'] = [min(mpi_runs), max(mpi_runs)]
        if n == 2048:
            threaded = [float(r['solve_seconds']) for r in groups[(n, 1, 8, 'blocking')]]
            row.update(threaded_runs=threaded, threaded_median=statistics.median(threaded),
                speedup_vs_serial=statistics.median(serial_runs)/statistics.median(threaded),
                speedup_vs_mpi_one=statistics.median(mpi_runs)/statistics.median(threaded))
        facts['baseline_controls'].append(row)
    facts['comparisons'].sort(key=lambda row: (row['topology'], -row['ranks']))
    facts['timed_records'] = total
    facts['pointwise_checks'] = 8
    (output / 'supplement_summary.json').write_text(json.dumps(facts, indent=2))
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(8.5, 3.9))
    labels = []
    for i, row in enumerate(facts['comparisons']):
        values = row['paired_reductions_pct']
        ax.scatter([i + (j-3)*0.035 for j in range(7)], values, color='#21758b', alpha=.7, s=24)
        ax.scatter([i], [row['median_paired_reduction_pct']], color='#c66b2b', marker='D', s=45, zorder=3)
        nodes = 1 if row['topology'] == 'single' else 2
        labels.append(f"{nodes} node(s)\n{row['ranks']} x {row['threads']}")
    ax.axhline(0, color='.4', linestyle='--', linewidth=1)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_xlabel('MPI ranks x OpenMP threads per rank')
    ax.set_ylabel('Paired runtime reduction with overlap (%)')
    ax.set_title('N=4096, K=3000; seven adjacent blocking/overlap pairs\nDots: repetitions; diamonds: medians; positive favours overlap')
    ax.grid(axis='y', alpha=.2)
    fig.tight_layout()
    fig.savefig(output / 'figures/paired_communication.png', dpi=220)
    plt.close(fig)
    control = facts['baseline_controls'][0]
    fig, ax = plt.subplots(figsize=(6.3, 3.5))
    ax.boxplot([control['serial_runs'], control['mpi_one_runs'], control['threaded_runs']],
               tick_labels=['Serial C++', 'MPI: 1 rank x 1 thread', 'MPI: 1 rank x 8 threads'], showmeans=True)
    ax.set_ylabel('Iteration-loop time (s)')
    ax.set_title('Baseline control: N=2048, K=3000, seven repetitions\nSame allocation; unchanged numerical sources')
    ax.grid(axis='y', alpha=.2)
    fig.tight_layout()
    fig.savefig(output / 'figures/baseline_control.png', dpi=220)
    plt.close(fig)
    report = ['# Supplementary experiment results', '',
              'Unchanged numerical C++ sources; 3000 fixed iterations, seven repetitions.',
              'Blocking and overlap cases are adjacent within each repetition, with randomized order.', '',
              '| Allocation | MPI x threads | Blocking (s) | Overlap (s) | Median paired reduction | Overlap wins |',
              '|---|---:|---:|---:|---:|---:|']
    for r in facts['comparisons']:
        report.append(f"| {r['topology']} | {r['ranks']} x {r['threads']} | {r['blocking_seconds']:.4f} | "
                      f"{r['overlap_seconds']:.4f} | {r['median_paired_reduction_pct']:+.1f}% | {r['overlap_wins']}/7 |")
    report += ['', 'Positive reduction means shorter runtime with overlap. The paired median is not the ratio of medians.', '',
               '## Baseline control', '',
               f"At N=2048 the serial, MPI 1x1, and MPI 1x8 medians were {control['serial_median']:.4f}, "
               f"{control['mpi_one_median']:.4f}, and {control['threaded_median']:.4f} seconds.",
               f"The 1x8 speedup was {control['speedup_vs_serial']:.2f} against serial C++ and "
               f"{control['speedup_vs_mpi_one']:.2f} against MPI 1x1.", '', '## Limitations', '']
    report += ['- ' + text for text in facts['limitations']]
    (output / 'RESULTS.md').write_text('\n'.join(report) + '\n')
    for name in ['src', 'scripts', 'jobs']:
        shutil.copytree(project / name, output / name, ignore=shutil.ignore_patterns('__pycache__'))
    for name in ['Makefile', 'README.md']:
        shutil.copy2(project / name, output / name)
    print('SUPPLEMENT_ANALYSIS_COMPLETE', json.dumps(facts['comparisons'], indent=2))


if __name__ == '__main__':
    main()
