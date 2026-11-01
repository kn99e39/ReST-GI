"""WL68 server chain (one detached process per phase). Every stage writes logs/<stage>.log and chain.status; any failure
stops the chain with the failed stage preserved. Phase 'pre' ends after the budget is fixed; 'main' requires budget_commit.txt."""
import sys,os,json,time,subprocess
from pathlib import Path
from common68 import CFG,ROOT

HERE=Path(__file__).resolve().parent;PY=sys.executable
RUNENV=dict(P53_MI_VARIANT='cuda_ad_rgb',WL57_WALKER='fast',PYTHONUNBUFFERED='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
            LD_LIBRARY_PATH=str(Path.home()/'nvlibs/optix595'),R68_ROOT=str(ROOT))
def status(msg):(ROOT/'chain.status').write_text(f'{msg} {time.strftime("%Y-%m-%d %H:%M:%S")}\n');print(msg,flush=True)
def foreign():
    out=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True).stdout.split()
    return [p for p in out if p.strip() and int(p)!=os.getpid()]
def wait_idle():
    ok=0
    while ok<3:
        ok=ok+1 if not foreign() else 0;time.sleep(20 if ok<3 else 0)
def stage(name,args,done=None):
    if done and (ROOT/done).exists():status(f'skip {name} (complete)');return
    status(f'running {name}');log=open(ROOT/'logs'/f'{name}.log','a')
    r=subprocess.run([PY,*args],cwd=HERE,env=dict(os.environ,**RUNENV),stdout=log,stderr=subprocess.STDOUT)
    if r.returncode:status(f'FAILED {name} rc={r.returncode}');raise SystemExit(1)
def sampler():
    return subprocess.Popen(['bash','-c',f'while true; do echo "$(date +%s) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits) | $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr "\\n" ";")" >> {ROOT}/logs/gpu_sampler.log; sleep 15; done'])

def pre():
    stage('prepare',['fit68.py','prepare'],'inputs/fit_targets.pt')
    stage('tests',['test68.py'])
    stage('ports',['ports68.py'],'heldout/ports.json')
    stage('wl66_reproduction',['eval68.py','repro'],'wl66_reproduction.json')
    stage('cross_machine',['probes68.py','cross'],'heldout/cross_machine.json')
    stage('preflight',['probes68.py','preflight'],'heldout/budget.json')
    status('pre_done awaiting budget commit')

def main():
    assert (ROOT/'budget_commit.txt').exists(),'commit budget.json first'
    b=json.loads((ROOT/'heldout/budget.json').read_text())
    if (ROOT/'heldout/execution_amendment.json').exists():b.update(json.loads((ROOT/'heldout/execution_amendment.json').read_text()))
    stage('prepare',['fit68.py','prepare'],'inputs/fit_targets.pt')
    if not (ROOT/'heldout/complete.json').exists():
        status(f"running generate x{b['workers']}");procs=[]
        for w in range(b['workers']):
            procs.append(subprocess.Popen([PY,'probes68.py','generate',str(w)],cwd=HERE,env=dict(os.environ,**RUNENV),stdout=open(ROOT/'logs'/f'generate_w{w}.log','a'),stderr=subprocess.STDOUT))
        rc=[p.wait() for p in procs]
        if any(rc):status(f'FAILED generate rc={rc}');raise SystemExit(1)
        stage('bundle',['probes68.py','bundle'],'heldout/complete.json')
    order=[r for r,v in CFG['runs'].items() if v['target']=='phys' and v['kind']=='net']+[r for r,v in CFG['runs'].items() if v['target']=='phys' and v['kind']=='coef']
    order+=list(CFG['convex'])+[r for r in CFG['runs'] if CFG['runs'][r]['target'] not in ('phys','T2')]+[r for r in CFG['runs'] if CFG['runs'][r]['target']=='T2']
    for rid in order:
        stage(f'fit_{rid}',['fit68.py','convex' if rid in CFG['convex'] else 'run',rid],f'runs/{rid}.pt')
    stage('eval',['eval68.py','main'],'eval/analysis.json')
    stage('export',['export68.py'],'final/summary.json')
    status('done')

if __name__=='__main__':
    (ROOT/'logs').mkdir(parents=True,exist_ok=True);s=sampler()
    try:
        status(f'waiting_gpu_idle {sys.argv[1]}');wait_idle();{'pre':pre,'main':main}[sys.argv[1]]()
    except SystemExit:raise
    except Exception as e:status(f'FAILED chain {e!r}');raise
    finally:s.terminate()
