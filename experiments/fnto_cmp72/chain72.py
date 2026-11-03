"""WL72 server chain (whole batch end to end, no session dependency): pt0 symmetry geometries -> confirmation ports (te2 identity) ->
gates/tests -> 4 candidate fits (2 lanes) -> confirmation references -> bundle -> evaluation -> export. Any failure stops the chain."""
import sys,os,time,signal,subprocess
from pathlib import Path
from common72 import CFG,ROOT

HERE=Path(__file__).resolve().parent;PY=sys.executable
RUNENV=dict(P53_MI_VARIANT='cuda_ad_rgb',WL57_WALKER='fast',PYTHONUNBUFFERED='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
            LD_LIBRARY_PATH=str(Path.home()/'nvlibs/optix595'),R72_ROOT=str(ROOT))
def status(msg):(ROOT/'chain.status').write_text(f'{msg} {time.strftime("%Y-%m-%d %H:%M:%S")}\n');print(msg,flush=True)
def stage(name,args,done=None,env=None):
    if done and (ROOT/done).exists():status(f'skip {name} (complete)');return
    status(f'running {name}');log=open(ROOT/'logs'/f'{name}.log','a')
    r=subprocess.run([PY,*args],cwd=HERE,env=dict(os.environ,**RUNENV,**(env or {})),stdout=log,stderr=subprocess.STDOUT)
    if r.returncode:status(f'FAILED {name} rc={r.returncode}');raise SystemExit(1)
def sampler():
    cmd=f'while true; do echo "$(date +%s) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits) | $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr "\\n" ";")" >> {ROOT}/logs/gpu_sampler.log; sleep 15; done'
    return subprocess.Popen(['bash','-c',cmd],start_new_session=True)
def fits():
    lanes=int(os.environ.get('R72_LANES','2'));todo=[r for r in CFG['runs'] if not (ROOT/'runs'/f'{r}.pt').exists()];live={}
    while todo or live:
        while todo and len(live)<lanes:
            rid=todo.pop(0);status(f'running fit_{rid} (lanes {lanes})')
            live[rid]=subprocess.Popen([PY,'fit72.py','run',rid],cwd=HERE,env=dict(os.environ,**RUNENV,R72_LANE=f'{lanes} lanes'),stdout=open(ROOT/'logs'/f'fit_{rid}.log','a'),stderr=subprocess.STDOUT)
        time.sleep(10)
        for rid,p in list(live.items()):
            if p.poll() is not None:
                del live[rid]
                if p.returncode:
                    for q in live.values():q.wait()
                    status(f'FAILED fit_{rid} rc={p.returncode}');raise SystemExit(1)
def main():
    stage('geoms',['fit72.py','geoms'],'geom/variants.json');stage('ports',['ports72.py'],'ports.json');stage('tests',['test72.py'])
    fits()
    if not (ROOT/'refs/complete.json').exists():stage('generate',['probes72.py','generate']);stage('bundle',['probes72.py','bundle'],'refs/complete.json')
    stage('eval',['eval72.py'],'eval/analysis.json');stage('export',['export72.py'],'final/summary.json');status('done')
if __name__=='__main__':
    (ROOT/'logs').mkdir(parents=True,exist_ok=True);s=sampler()
    try:main()
    except SystemExit:raise
    except Exception as e:status(f'FAILED chain {e!r}');raise
    finally:os.killpg(s.pid,signal.SIGTERM)
