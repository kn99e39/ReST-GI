"""WL71 server chain. Phase 'pre': ports -> pre tests (then the port identities are committed). Phase 'main' (requires
ports_commit.txt): reference preflight -> generation -> bundle -> fit targets -> post tests -> fits (R71_LANES concurrent) ->
evaluation -> export. Every stage logs to logs/<stage>.log and chain.status; any failure stops the chain with evidence preserved."""
import sys,os,json,time,signal,subprocess
from pathlib import Path
from common71 import CFG,ROOT

HERE=Path(__file__).resolve().parent;PY=sys.executable
RUNENV=dict(P53_MI_VARIANT='cuda_ad_rgb',WL57_WALKER='fast',PYTHONUNBUFFERED='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
            LD_LIBRARY_PATH=str(Path.home()/'nvlibs/optix595'),R71_ROOT=str(ROOT))
def status(msg):(ROOT/'chain.status').write_text(f'{msg} {time.strftime("%Y-%m-%d %H:%M:%S")}\n');print(msg,flush=True)
def foreign():
    out=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True).stdout.split()
    return [p for p in out if p.strip()]
def wait_idle():
    ok=0
    while ok<3:ok=ok+1 if not foreign() else 0;time.sleep(20 if ok<3 else 0)
def stage(name,args,done=None,env=None):
    if done and (ROOT/done).exists():status(f'skip {name} (complete)');return
    status(f'running {name}');log=open(ROOT/'logs'/f'{name}.log','a')
    r=subprocess.run([PY,*args],cwd=HERE,env=dict(os.environ,**RUNENV,**(env or {})),stdout=log,stderr=subprocess.STDOUT)
    if r.returncode:status(f'FAILED {name} rc={r.returncode}');raise SystemExit(1)
def sampler():
    cmd=f'while true; do echo "$(date +%s) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits) | $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr "\\n" ";")" >> {ROOT}/logs/gpu_sampler.log; sleep 15; done'
    return subprocess.Popen(['bash','-c',cmd],start_new_session=True)

def fits():
    lanes=int(os.environ.get('R71_LANES','1'));todo=[r for r in CFG['runs'] if not (ROOT/'runs'/f'{r}.pt').exists()];live={}
    while todo or live:
        while todo and len(live)<lanes:
            rid=todo.pop(0);status(f'running fit_{rid} (lanes {lanes})');log=open(ROOT/'logs'/f'fit_{rid}.log','a')
            live[rid]=subprocess.Popen([PY,'fit71.py','run',rid],cwd=HERE,env=dict(os.environ,**RUNENV,R71_LANE=f'{lanes} concurrent' if lanes>1 else 'exclusive'),stdout=log,stderr=subprocess.STDOUT)
        time.sleep(10)
        for rid,p in list(live.items()):
            if p.poll() is not None:
                del live[rid]
                if p.returncode:
                    for q in live.values():q.wait()
                    status(f'FAILED fit_{rid} rc={p.returncode}');raise SystemExit(1)

def pre():
    stage('ports',['ports71.py'],'ports.json');stage('tests_pre',['test71.py','pre']);status('pre_done awaiting ports commit')

def main():
    assert (ROOT/'ports_commit.txt').exists(),'commit ports.json first'
    stage('preflight',['probes71.py','preflight'],'refs/budget71.json')
    if not (ROOT/'refs/complete.json').exists():
        stage('generate',['probes71.py','generate']);stage('bundle',['probes71.py','bundle'],'refs/complete.json')
    stage('prepare',['fit71.py','prepare'],'fit_targets71.pt');stage('tests_post',['test71.py','post'])
    fits();stage('eval',['eval71.py'],'eval/analysis.json');stage('export',['export71.py'],'final/summary.json');status('done')

if __name__=='__main__':
    (ROOT/'logs').mkdir(parents=True,exist_ok=True);s=sampler()
    try:status(f'waiting_gpu_idle {sys.argv[1]}');wait_idle();{'pre':pre,'main':main}[sys.argv[1]]()
    except SystemExit:raise
    except Exception as e:status(f'FAILED chain {e!r}');raise
    finally:os.killpg(s.pid,signal.SIGTERM)
