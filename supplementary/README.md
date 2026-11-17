# Supplementary Vanda experiments

The numerical C++ files are identical to the root `src/` files. The follow-up
uses 3000 fixed iterations and seven repetitions, pairing blocking and overlap
runs adjacently with randomized order. It also checks serial C++, MPI 1x1 and
MPI 1x8 baselines at N=2048. Full-core layouts are 8x1, 4x2, 2x4, 1x8 on one
node, and 16x1, 8x2, 4x4 on two nodes. There are 147 timed runs and eight small
pointwise checks, with CPU binding verified before measurements.

From this directory on the Vanda login node:

```bash
python3 scripts/submit_supplement.py --plan-only
python3 scripts/submit_supplement.py
```

The first command submits nothing. The second schedules one-node and two-node
compute jobs sequentially, followed by analysis. Job IDs are saved under a unique
`logs/<run-tag>/submission.json`. Analysis is written to a unique directory under
`$HOME/project3_supplement_results/<run-tag>/`, preserving previous results.
Compilation and timing occur only on compute nodes. The jobs load `gompi/2024a`;
analysis loads `matplotlib/3.10.3-gfbf-2025a`.

After the analysis job finishes, its output directory contains a generated
`supplement_summary.json` and `figures/`. To redraw that run's communication chart
from its summary, pass the analysis output directory:

```bash
python3 scripts/redraw_communication.py --directory /path/to/analysis/output
```

The image is written under that directory as `figures/paired_communication.png`.
Existing measurements and cluster metadata are not included in this code repository.
The analysis no longer copies the archived reference-report JSON; numerical
calculations and experiment settings are unchanged. Positive paired
runtime reduction means overlap was faster. The median of paired reductions is
not the percentage computed from two separate median runtimes. Seven repeats
describe stability under these conditions; they do not establish universal gains
or isolate cache, NUMA and MPI progress mechanisms. Initial and follow-up runs
used different iteration counts and allocations, so absolute times across the two
batches should not be compared directly.
