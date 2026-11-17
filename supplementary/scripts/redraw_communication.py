import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--directory', default=str(Path(__file__).resolve().parents[1]))
args = parser.parse_args()
directory = Path(args.directory)
facts = json.loads((directory / 'supplement_summary.json').read_text())
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
fig, ax = plt.subplots(figsize=(8.5, 4.1))
labels = []
for i, row in enumerate(facts['comparisons']):
    ax.scatter([i + (j-3)*.035 for j in range(7)], row['paired_reductions_pct'],
               color='#21758b', alpha=.7, s=24)
    ax.scatter([i], [row['median_paired_reduction_pct']], color='#c66b2b', marker='D', s=45, zorder=3)
    nodes = 1 if row['topology'] == 'single' else 2
    labels.append(f"{nodes} node(s)\n{row['ranks']} x {row['threads']}")
ax.axhline(0, color='.4', linestyle='--', linewidth=1)
ax.set_xticks(range(len(labels)), labels)
ax.set_xlabel('MPI ranks x OpenMP threads per rank')
ax.set_ylabel('Paired runtime reduction (%)')
ax.set_title('N=4096, K=3000; seven adjacent blocking/overlap pairs\nDots: repetitions; diamonds: medians; positive favours overlap')
ax.grid(axis='y', alpha=.2)
fig.tight_layout()
fig.savefig(directory / 'figures/paired_communication.png', dpi=220)
plt.close(fig)
