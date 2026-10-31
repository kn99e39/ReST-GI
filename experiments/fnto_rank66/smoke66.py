"""NON-EVIDENCE smoke harness: whole pipeline in a scratch root with K=4 replicas of 2M-photon packets."""
import os,sys,json,subprocess,shutil
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=Path(sys.argv[1]);assert ROOT!=Path.home()/'rest_rank66'
sys.path.insert(0,str(HERE));from chain66 import RUNENV
env=dict(os.environ,**RUNENV,R66_ROOT=str(ROOT),TMPDIR=str(ROOT/'tmp'),P53_TMPDIR=str(ROOT/'tmp'));(ROOT/'tmp').mkdir(parents=True,exist_ok=True)
def run(*cmd,**kw):print('SMOKE_STAGE',cmd,flush=True);subprocess.run([sys.executable,*map(str,cmd)],env=dict(env,**kw),cwd=HERE,check=True)
stages=sys.argv[2].split(',')
if 'ports' in stages:run(HERE/'ports66.py')
if 'tests' in stages:run(HERE/'test66.py')
if 'preflight' in stages:
    run(HERE/'probes66.py','preflight','--source-commit','smoke')
    b=json.loads((ROOT/'budget.json').read_text());b.update(replicas=4,packets=b['probes']*4,t_quantile=3.182446,smoke_override=True);(ROOT/'budget.json').write_text(json.dumps(b))
if 'generate' in stages:run(HERE/'probes66.py','generate','--source-commit','smoke',R66_SMOKE_N='2000000')
if 'model' in stages:run(HERE/'model66.py')
if 'analyze' in stages:run(HERE/'analyze66.py')
if 'export' in stages:run(HERE/'export66.py')
