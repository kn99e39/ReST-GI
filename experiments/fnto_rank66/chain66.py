"""Durable serial WL66 chain. 'pre': ports -> tests -> resource preflight/budget (then commit budget). 'main': generation ->
frozen model outputs -> analysis -> export. Stops on any failure; completed stage markers are never re-run."""
import os,sys,json,subprocess,time,traceback
from pathlib import Path
HOME=Path.home();ROOT=Path(os.environ.get('R66_ROOT',HOME/'rest_rank66'));HERE=Path(__file__).resolve().parent;PY=sys.executable
# The frozen WL59 run.sh exports, applied to every stage.
RUNENV=dict(P53_MI_VARIANT='cuda_ad_rgb',DRJIT_LIBLLVM_PATH='/usr/lib64/libLLVM-11.so',PYTHONUNBUFFERED='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
            WL57_WALKER='fast',WL57_ACTION_CHUNK='32000000',WL57_EDGE_CHUNK='65536',WL57_EVAL_BATCH='4',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='8')
def stamp():return time.strftime('%Y-%m-%d %H:%M:%S')
def status(text):(ROOT/'chain.status').write_text(f'{text} {stamp()}\n')
def gpu():return subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True)
def run(name,cmd):
    marker=ROOT/'stages'/f'{name}.json'
    if marker.exists():return json.loads(marker.read_text())
    status(f'running {name}');before=gpu();t=time.time()
    with open(ROOT/f'{name}.log','w') as f:
        subprocess.run(list(map(str,cmd)),env=dict(os.environ,**RUNENV,TMPDIR=str(ROOT/'tmp'),P53_TMPDIR=str(ROOT/'tmp')),cwd=HERE,stdout=f,stderr=subprocess.STDOUT,check=True)
    rec=dict(stage=name,started=t,ended=time.time(),seconds=time.time()-t,GPU_processes_before=before,GPU_processes_after=gpu(),command=list(map(str,cmd)))
    marker.parent.mkdir(exist_ok=True);marker.write_text(json.dumps(rec,indent=2));return rec

def main(phase):
    assert 'R66_SMOKE_N' not in os.environ
    (ROOT/'tmp').mkdir(parents=True,exist_ok=True);commit=(ROOT/'source_commit.txt').read_text().strip()
    if phase=='pre':
        run('ports',[PY,HERE/'ports66.py']);run('tests',[PY,HERE/'test66.py'])
        run('preflight',[PY,HERE/'probes66.py','preflight','--source-commit',commit]);status('preflight done; budget fixed by rule; awaiting budget commit');return
    assert (ROOT/'budget_commit.txt').exists(),'Budget must be committed before evidence generation'
    run('generate',[PY,HERE/'probes66.py','generate','--source-commit',commit])
    run('model',[PY,HERE/'model66.py']);run('analyze',[PY,HERE/'analyze66.py']);run('export',[PY,HERE/'export66.py'])
    a=json.loads((ROOT/'analysis.json').read_text())
    (ROOT/'final_status.json').write_text(json.dumps(dict(state='COMPLETE',outcome=a['outcome'],precision_passed=a['precision']['passed'],time=time.time()),indent=2))
    status(f'done outcome {a["outcome"]}')

if __name__=='__main__':
    try:main(sys.argv[1])
    except Exception:
        (ROOT/'failure.txt').write_text(traceback.format_exc());status('failed see failure.txt');raise
