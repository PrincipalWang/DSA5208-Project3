import os
from pathlib import Path
import sys

threads = int(sys.argv[1])
local_rank = int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', '0'))
local_size = int(os.environ.get('OMPI_COMM_WORLD_LOCAL_SIZE', '1'))
cores = {}
for cpu in sorted(os.sched_getaffinity(0)):
    topology = Path('/sys/devices/system/cpu') / ('cpu' + str(cpu)) / 'topology'
    key = ((topology / 'physical_package_id').read_text().strip(),
           (topology / 'core_id').read_text().strip())
    cores.setdefault(key, cpu)
cpus = list(cores.values())
if local_size * threads > len(cpus):
    raise SystemExit('Requested more physical cores than the inherited CPU allocation')
selected = cpus[local_rank * threads:(local_rank + 1) * threads]
os.sched_setaffinity(0, selected)
if set(os.sched_getaffinity(0)) != set(selected):
    raise SystemExit('Failed to apply the requested rank affinity')
if os.environ.get('JACOBI_REPORT_BINDING') == '1':
    print('local_rank=' + str(local_rank) + ', CPUs=' + str(selected), file=sys.stderr)
os.execvp(sys.argv[2], sys.argv[2:])
