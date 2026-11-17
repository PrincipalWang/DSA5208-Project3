import argparse
import csv
import math
from run_common import ROOT, execute, read_result

parser = argparse.ArgumentParser()
parser.add_argument('--directory', default='results/vanda')
args = parser.parse_args()
out = ROOT / args.directory
out.mkdir(parents=True, exist_ok=True)
rows = []
for n, iterations in [(15, 5000), (31, 10000), (63, 40000)]:
    command = [str(ROOT / 'bin/jacobi_serial'), '--n', str(n), '--iterations', str(iterations)]
    if n == 63:
        command += ['--output', str(out / 'converged_grid.csv')]
    result = read_result(execute(command))
    rows.append(dict(n=n, iterations=iterations, h=1 / (n + 1),
        solution_error_max=float(result['solution_error_max']),
        iteration_error_max=float(result['iteration_error_max']),
        residual_max=float(result['residual_max'])))
for i, row in enumerate(rows):
    row['observed_order'] = '' if i == 0 else math.log(
        rows[i - 1]['solution_error_max'] / row['solution_error_max'], 2)
    if i and row['observed_order'] < 1.8:
        raise RuntimeError('Second-order spatial convergence was not observed')
with (out / 'convergence.csv').open('w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print('Spatial convergence checks passed:', rows)
