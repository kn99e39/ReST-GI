"""Durable serial extension chain: tests -> top-up -> Gate 2 -> (READY only) U65 -> A/B -> evaluation. Stops on any failure."""
import os,sys,json,subprocess,time,traceback
from pathlib import Path
HOME=Path.home();ROOT=Path(os.environ.get('C65X_ROOT',HOME/'rest_clean65_ext'));HERE=Path(__file__).resolve().parent;ORIG=HOME/'rest_persist_v1'
PY=sys.executable;RUN=ORIG/'repo/experiments/fnto_persist59/run.sh'
# Exactly the frozen WL59 run.sh exports (minus per-root P59/WL57/targets), for every non-run.sh stage too.
RUNENV=dict(P53_MI_VARIANT='cuda_ad_rgb',DRJIT_LIBLLVM_PATH='/usr/lib64/libLLVM-11.so',PYTHONUNBUFFERED='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',WL57_WALKER='fast',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='8')
POLICY0=dict(WL57_ACTION_CHUNK='32000000',WL57_EDGE_CHUNK='65536',WL57_EVAL_BATCH='4',P59_ATTEMPT='0')

def stamp():return time.strftime('%Y-%m-%d %H:%M:%S')
def status(text):(ROOT/'chain.status').write_text(f'{text} {stamp()}\n')
def gpu():return subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True)
def run(name,cmd,env=None,cwd=HERE):
    marker=ROOT/'stages'/f'{name}.json'
    if marker.exists():return json.loads(marker.read_text())
    status(f'running {name}');before=gpu();t=time.time()
    with open(ROOT/f'{name}.log','w') as f:subprocess.run(list(map(str,cmd)),env=env or dict(os.environ,**RUNENV,TMPDIR=str(ROOT/'tmp'),P53_TMPDIR=str(ROOT/'tmp')),cwd=cwd,stdout=f,stderr=subprocess.STDOUT,check=True)
    rec=dict(stage=name,started=t,ended=time.time(),seconds=time.time()-t,GPU_processes_before=before,GPU_processes_after=gpu(),command=list(map(str,cmd)))
    marker.parent.mkdir(exist_ok=True);marker.write_text(json.dumps(rec,indent=2));return rec

def arm_env(arm):return dict(os.environ,P59=str(ROOT/'arms'/arm),**POLICY0)

def main():
    assert 'C65X_SMOKE_STEPS' not in os.environ
    (ROOT/'tmp').mkdir(exist_ok=True)
    commit=(ROOT/'source_commit.txt').read_text().strip()
    run('tests_phaseA',[PY,HERE/'test65.py']);run('tests_ext',[PY,HERE/'test65b.py'])
    run('topup_preflight',[PY,HERE/'topup65.py','preflight','--root',ROOT,'--source-commit',commit])
    run('topup_generate',[PY,HERE/'topup65.py','generate','--root',ROOT,'--source-commit',commit])
    run('qualify',[PY,HERE/'qualify65.py','--root',ROOT]);run('label_noise',[PY,HERE/'labelnoise65.py'])
    q=json.loads((ROOT/'qualification_status.json').read_text())
    if q['status']!='READY_FOR_CONTROLLED_TRAINING':
        outcome='C' if q['status']=='STOP_INSUFFICIENT_PRECISION' else 'D'
        run('export',[PY,HERE/'export65.py',ROOT,ROOT/'review'])
        (ROOT/'final_status.json').write_text(json.dumps(dict(state=q['status'],outcome=outcome,training='not conducted',time=time.time()),indent=2));status(f'stopped {q["status"]} outcome {outcome}');return
    run('fresh_U65',[PY,HERE/'fresh65.py']);run('arms',[PY,HERE/'arms65.py'])
    for seed,order in ((5901,('A','B')),(5902,('B','A'))):
        for arm in order:run(f'train_{arm}_{seed}',['bash',RUN,'train.py','P',seed,6000],env=arm_env(arm))
    for arm in ('A','B'):run(f'evaluate_{arm}',['bash',RUN,'evaluate.py'],env=arm_env(arm))
    for arm in ('A','B'):run(f'cost_{arm}',['bash',RUN,'cost.py'],env=arm_env(arm))
    run('invariants',[PY,HERE/'invariants65.py']);run('compare',[PY,HERE/'compare65.py'])
    run('export',[PY,HERE/'export65.py',ROOT,ROOT/'review'])
    c=json.loads((ROOT/'comparison.json').read_text())
    (ROOT/'final_status.json').write_text(json.dumps(dict(state='COMPLETE',outcome=c['outcome'],sub=c['outcome_B_sublabel'],coupled_GI='not started',time=time.time()),indent=2))
    status(f'done outcome {c["outcome"]}')

if __name__=='__main__':
    try:main()
    except Exception:
        tb=traceback.format_exc();(ROOT/'failure.txt').write_text(tb);status('failed see failure.txt');raise
