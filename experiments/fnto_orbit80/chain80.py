"""WL80 server chain B (whole remaining batch, no session dependency), started after chain A (Stages 0-2):
[Stage 2 gate] -> tests -> full-scale probe (non-evidence; lanes) -> six arm fits -> supervised evaluation + fit gate -> mechanism (always) ->
[>= 1 arm passes fit AND mechanism] Stage 5 population freeze -> references -> bundle -> Stage 5 -> Stage 6 deformation (always) -> cost ->
export. Any failure stops."""
import sys,os,time,signal,subprocess,json
from pathlib import Path
from common80 import CFG,ROOT,dev

HERE=Path(__file__).resolve().parent;PY=sys.executable
RUNENV=dict(P53_MI_VARIANT='cuda_ad_rgb',WL57_WALKER='fast',PYTHONUNBUFFERED='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
            LD_LIBRARY_PATH=str(Path.home()/'nvlibs/optix595'),R80_ROOT=str(ROOT))
FORCE=dev() and os.environ.get('R80_DEV_FORCE')=='1'
def status(msg):(ROOT/'chain.status').write_text(f'{msg} {time.strftime("%Y-%m-%d %H:%M:%S")}\n');print(msg,flush=True)
def stage(name,args,done=None,env=None):
    if done and (ROOT/done).exists():status(f'skip {name} (complete)');return
    status(f'running {name}');log=open(ROOT/'logs'/f'{name}.log','a')
    r=subprocess.run([PY,*args],cwd=HERE,env=dict(os.environ,**RUNENV,**(env or {})),stdout=log,stderr=subprocess.STDOUT)
    if r.returncode:status(f'FAILED {name} rc={r.returncode}');raise SystemExit(1)
def spawn(name,args,env):
    return subprocess.Popen([PY,*args],cwd=HERE,env=dict(os.environ,**RUNENV,**env),stdout=open(ROOT/'logs'/f'{name}.log','a'),stderr=subprocess.STDOUT)
def lanes(jobs,n):
    todo=[j for j in jobs if not (ROOT/j[2]).exists()];live={}
    while todo or live:
        while todo and len(live)<n:
            name,args,_=todo.pop(0);status(f'running {name} (lanes {n})');live[name]=spawn(name,args,dict(R75_LANE=f'{n} lanes'))
        time.sleep(10)
        for name,p in list(live.items()):
            if p.poll() is not None:
                del live[name]
                if p.returncode:
                    for q in live.values():q.wait()
                    status(f'FAILED {name} rc={p.returncode}');raise SystemExit(1)
def probe():
    f=ROOT/'probe/lanes.json'
    if f.exists():return json.loads(f.read_text())['lanes']
    (ROOT/'probe').mkdir(exist_ok=True);steps=str(int(os.environ.get('R80_PROBE_STEPS',CFG['execution']['probe_steps'])));runs=list(CFG['execution']['probe_runs'])
    status('running probe (2 lanes)');ps={r:spawn(f'probe2_{r}',['fit80.py','probe',r,steps],dict(R75_LANE='2 lanes',R75_PROBE_TAG='_2lanes')) for r in runs}
    rc={r:p.wait() for r,p in ps.items()}
    total=int(subprocess.run(['nvidia-smi','--query-gpu=memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True).stdout.split()[0])*2**20
    two={r:json.loads((ROOT/'probe'/f'{r}_2lanes.json').read_text()) for r in runs if rc[r]==0}
    ok2=len(two)==len(runs) and sum(v['peak_GPU_reserved'] for v in two.values())<=.85*total;one=None
    if not ok2:
        status('running probe (1 lane)');r=runs[0];p=spawn(f'probe1_{r}',['fit80.py','probe',r,steps],dict(R75_LANE='1 lane',R75_PROBE_TAG='_1lane'))
        if p.wait():status('FAILED probe (single lane)');raise SystemExit(1)
        one=json.loads((ROOT/'probe'/f'{r}_1lane.json').read_text())
    n=2 if ok2 else 1
    f.write_text(json.dumps(dict(lanes=n,two_lane=two,return_codes=rc,single_lane=one,gpu_memory_total=total,rule=CFG['execution']['lanes_rule']),indent=1))
    status(f'probe done: {n} lanes');return n
def main():
    g2=json.loads((ROOT/'stage2/gate.json').read_text())
    if not g2['passed'] and not FORCE:
        status(f"Stage 2: {g2['classification']} - no learned training (protocol)");stage('export',['export80.py'],'final/summary.json');status('done');return
    stage('tests',['test80.py'],'tests/tests.json',env=dict(R80_TEST_RECORD='1'))
    if 'FAIL' in (ROOT/'tests/tests.json').read_text():status('FAILED tests');raise SystemExit(1)
    n=probe()
    lanes([(f'fit_{r}',['fit80.py','run',r],f'runs/{r}.pt') for r in CFG['runs']],n)
    (ROOT/'stage3').mkdir(exist_ok=True);stage('stage3',['eval80.py','stage3'],'stage3/analysis.json')
    (ROOT/'mech').mkdir(exist_ok=True);stage('mech',['eval80.py','mech'],'mech/mech.json')
    g3=json.loads((ROOT/'stage3/gate.json').read_text());g4=json.loads((ROOT/'mech/gate.json').read_text())
    ok=[a for a in CFG['arms'] if g3['arms'][a]['passed'] and g4['arms'][a]]
    if ok or FORCE:
        if FORCE and not ok:status('DEV ONLY: Stage 5 forced to exercise the path')
        stage('ports80',['ports80.py','freeze'],'ports80.json')
        stage('refs_generate',['probes80.py','generate'],'refs/complete.json',env=dict(R74_SMOKE_N='1000000') if dev() else None)
        stage('refs_bundle',['probes80.py','bundle'],'refs/complete.json')
        (ROOT/'stage5').mkdir(exist_ok=True);stage('stage5',['eval80.py','stage5'],'stage5/analysis.json')
    else:status(f"gates: fit {g3['classification']} / mechanism {g4['arms']} - Stage 5 not run (protocol)")
    (ROOT/'stage6').mkdir(exist_ok=True);stage('stage6',['eval80.py','stage6'],'stage6/deformation.json')
    (ROOT/'cost').mkdir(exist_ok=True);stage('cost',['cost80.py','final'],'cost/final.json')
    stage('export',['export80.py'],'final/summary.json');status('done')
if __name__=='__main__':
    (ROOT/'logs').mkdir(parents=True,exist_ok=True)
    try:main()
    except SystemExit:raise
    except Exception as e:status(f'FAILED chain {e!r}');raise
