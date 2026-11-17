import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import uuid

parser = argparse.ArgumentParser()
parser.add_argument('--plan-only', action='store_true')
args = parser.parse_args()
project = Path(__file__).resolve().parents[1]
tag = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6]
logs = project / 'logs' / tag
output = Path.home() / 'project3_supplement_results' / tag / 'project3_supplement'
if args.plan_only:
    print(json.dumps(dict(project=str(project), logs=str(logs), output=str(output),
                         sequence=['single', 'two (afterok:single)', 'analysis (afterok:two)']), indent=2))
    raise SystemExit(0)
if ',' in str(logs) or ',' in str(output):
    raise SystemExit('PBS environment paths must not contain commas')
logs.mkdir(parents=True, exist_ok=False)
jobs = dict(output=str(output), logs=str(logs))
def save():
    (logs / 'submission.json').write_text(json.dumps(jobs, indent=2))
def submit(arguments):
    return subprocess.check_output(['qsub'] + arguments, cwd=project, text=True).strip()
save()
jobs['single'] = submit(['-o', str(logs / 'single.log'), 'jobs/single.pbs'])
save()
jobs['two'] = submit(['-W', 'depend=afterok:' + jobs['single'], '-o', str(logs / 'two.log'), 'jobs/two.pbs'])
save()
environment = ','.join(['SUPPLEMENT_SINGLE=' + jobs['single'], 'SUPPLEMENT_TWO=' + jobs['two'],
                        'SUPPLEMENT_LOG_DIR=' + str(logs), 'SUPPLEMENT_OUTPUT_DIR=' + str(output)])
jobs['analysis'] = submit(['-W', 'depend=afterok:' + jobs['two'], '-v', environment,
                           '-o', str(logs / 'analysis.log'), 'jobs/analyse.pbs'])
save()
print(json.dumps(jobs, indent=2))
