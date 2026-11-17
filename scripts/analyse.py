import argparse
import csv
from collections import defaultdict
import json
from pathlib import Path
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--directory', default='results/vanda')
parser.add_argument('--platform', default='Vanda')
args = parser.parse_args()
out = Path(args.directory).resolve()
figures = out / 'figures'
figures.mkdir(exist_ok=True)
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
summary = []
for mode in ('strong', 'weak'):
    with (out / (mode + '.csv')).open(newline='') as file:
        records = list(csv.DictReader(file))
    groups = defaultdict(list)
    for row in records:
        key = (int(row['n']), int(row['iterations']), int(row['ranks']),
               int(row['threads']), row['communication'])
        groups[key].append(float(row['solve_seconds']))
    serial = {key[0]: statistics.median(values) for key, values in groups.items() if key[4] == 'none'}
    weak_baseline = next(iter(serial.values())) if mode == 'weak' else None
    for (n, iterations, ranks, threads, communication), values in sorted(groups.items()):
        median = statistics.median(values)
        baseline = serial[n] if mode == 'strong' else weak_baseline
        speedup = baseline / median if mode == 'strong' else ''
        summary.append(dict(mode=mode, n=n, iterations=iterations, ranks=ranks, threads=threads,
            workers=ranks * threads, communication=communication, repeats=len(values),
            median_seconds=median, min_seconds=min(values), max_seconds=max(values),
            stdev_seconds=statistics.stdev(values) if len(values) > 1 else 0,
            speedup=speedup, efficiency=speedup / (ranks * threads) if mode == 'strong' else '',
            weak_efficiency=baseline / median if mode == 'weak' else '',
            cells_per_worker=n * n / (ranks * threads)))
with (out / 'summary.csv').open('w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=list(summary[0]))
    writer.writeheader()
    writer.writerows(summary)

def select_series(rows):
    specs = [('MPI', '#186a8b', lambda r: r['threads'] == 1),
             ('OpenMP (1 rank)', '#ca6c23', lambda r: r['ranks'] == 1),
             ('Hybrid (2 threads/rank)', '#33855b', lambda r: r['threads'] == 2)]
    for name, color, condition in specs:
        for communication, style in [('blocking', '-'), ('overlap', '--')]:
            points = sorted((r for r in rows if r['communication'] == communication and condition(r)),
                            key=lambda r: r['workers'])
            if points:
                yield name + ' / ' + communication, color, style, points

worker_ticks = sorted({r['workers'] for r in summary})
sizes = sorted({r['n'] for r in summary if r['mode'] == 'strong'})
fig, axes = plt.subplots(len(sizes), 3, figsize=(14, 3.4 * len(sizes)), squeeze=False)
for index, n in enumerate(sizes):
    rows = [r for r in summary if r['mode'] == 'strong' and r['n'] == n and r['communication'] != 'none']
    for label, color, style, points in select_series(rows):
        workers = [r['workers'] for r in points]
        for ax, field in zip(axes[index], ['median_seconds', 'speedup', 'efficiency']):
            ax.plot(workers, [r[field] for r in points], style, color=color, marker='o', label=label)
    maximum = max(r['workers'] for r in rows)
    axes[index, 1].plot([1, maximum], [1, maximum], ':', color='0.65', label='Ideal')
    axes[index, 2].axhline(1, linestyle=':', color='0.65')
    for ax, ylabel in zip(axes[index], ['Runtime (s)', 'Speedup vs serial', 'Efficiency']):
        ax.set_xscale('log')
        ax.set_xticks(worker_ticks)
        ax.set_xticklabels(worker_ticks)
        ax.set_xlabel('Physical cores used')
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)
        ax.set_title('N = ' + str(n))
fig.legend(*axes[0, 1].get_legend_handles_labels(), loc='upper center', ncol=4, fontsize=8)
fig.suptitle(args.platform + ' strong scaling', y=1.015, fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(figures / 'strong_scaling.png', dpi=180, bbox_inches='tight')
plt.close(fig)

rows = [r for r in summary if r['mode'] == 'weak' and r['communication'] != 'none']
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
for label, color, style, points in select_series(rows):
    workers = [r['workers'] for r in points]
    axes[0].plot(workers, [r['median_seconds'] for r in points], style, marker='o', color=color, label=label)
    axes[1].plot(workers, [r['weak_efficiency'] for r in points], style, marker='o', color=color)
for ax, ylabel in zip(axes, ['Runtime (s)', 'Weak efficiency vs serial']):
    ax.set_xscale('log')
    ax.set_xticks(worker_ticks)
    ax.set_xticklabels(worker_ticks)
    ax.set_xlabel('Physical cores used')
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2)
axes[1].axhline(1, color='0.65', linestyle=':')
fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', ncol=3, fontsize=8)
fig.suptitle(args.platform + ' weak scaling: approximately constant cells per core', y=1.06)
fig.tight_layout(rect=(0, 0, 1, 0.86))
fig.savefig(figures / 'weak_scaling.png', dpi=180, bbox_inches='tight')
plt.close(fig)

largest = max(sizes)
rows = [r for r in summary if r['mode'] == 'strong' and r['n'] == largest and r['workers'] == max(worker_ticks)]
configs = sorted({(r['ranks'], r['threads']) for r in rows}, reverse=True)
fig, ax = plt.subplots(figsize=(7.4, 4.0))
for side, communication in enumerate(['blocking', 'overlap']):
    lookup = {(r['ranks'], r['threads']): r['median_seconds'] for r in rows if r['communication'] == communication}
    ax.bar(np.arange(len(configs)) + (side - 0.5) * 0.35,
           [lookup[c] for c in configs], width=0.35, label=communication)
ax.set_xticks(np.arange(len(configs)))
ax.set_xticklabels([str(r) + ' ranks x ' + str(t) + ' threads' for r, t in configs])
ax.set_ylabel('Median runtime (s)')
ax.set_title('Same ' + str(max(worker_ticks)) + ' physical cores, N = ' + str(largest))
ax.legend()
fig.tight_layout()
fig.savefig(figures / 'hybrid_comparison.png', dpi=180)
plt.close(fig)

with (out / 'converged_grid.csv').open(newline='') as file:
    grid = list(csv.DictReader(file))
n = max(int(r['row']) for r in grid)
values = np.array([float(r['value']) for r in grid]).reshape(n, n)
x = np.arange(1, n + 1) / (n + 1)
xx, yy = np.meshgrid(x, x)
exact = np.sin(np.pi * xx) * np.sin(np.pi * yy) + 0.25 * np.sin(2 * np.pi * xx) * np.sin(3 * np.pi * yy)
fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
for ax, data, title in zip(axes, [values, np.abs(values - exact)],
                         ['Converged solution (N = 63)', 'Absolute error vs continuous solution']):
    im = ax.imshow(data, origin='lower', extent=(0, 1, 0, 1), cmap='viridis')
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.colorbar(im, ax=ax, shrink=0.85)
fig.tight_layout()
fig.savefig(figures / 'solution_check.png', dpi=180)
plt.close(fig)
best = {str(n): min((r for r in summary if r['mode'] == 'strong' and r['n'] == n
        and r['communication'] != 'none'), key=lambda r: r['median_seconds']) for n in sizes}
(out / 'best_results.json').write_text(json.dumps(best, indent=2))
print('Summaries and figures saved:', figures)
for n, result in best.items():
    print('N=' + n, result)
