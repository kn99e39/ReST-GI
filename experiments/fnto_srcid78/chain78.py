"""WL78 server chain (whole batch, no session dependency): tests -> cost gate (initial W+UV-Q) -> full-scale probe (non-evidence; lanes) ->
two W+UV-Q fits -> Stage 3 (fit gate) -> mechanism diagnostics (development data) -> [gate passed: Stage 4 on the WL77A Part 0 references] ->
final cost -> export. Any failure stops."""
import sys,os,time,signal,subprocess,json
from pathlib import Path
from common78 import CFG,ROOT,dev

HERE=Path(__file__).resolve().parent;PY=sys.executable
RUNENV=dict(P53_MI_VARIANT='cuda_ad_rgb',WL57_WALKER='fast',PYTHONUNBUFFERED='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
            LD_LIBRARY_PATH=str(Path.home()/'nvlibs/optix595'),R78_ROOT=str(ROOT))
def status(msg):(ROOT/'chain.status').write_text(f'{msg} {time.strftime("%Y-%m-%d %H:%M:%S")}\n');print(msg,flush=True)
def stage(name,args,done=None,env=None):
    if done and (ROOT/done).exists():status(f'skip {name} (complete)');return
    status(f'running {name}');log=open(ROOT/'logs'/f'{name}.log','a')
    r=subprocess.run([PY,*args],cwd=HERE,env=dict(os.environ,**RUNENV,**(env or {})),stdout=log,stderr=subprocess.STDOUT)
    if r.returncode:status(f'FAILED {name} rc={r.returncode}');raise SystemExit(1)
SAMPLER=r'''import subprocess,time,sys
q=lambda a:subprocess.run(['nvidia-smi',a,'--format=csv,noheader'+(',nounits' if 'gpu=' in a else '')],capture_output=True,text=True).stdout.strip().replace('\n',';')
while True:
    open(sys.argv[1],'a').write(f"{int(time.time())} {q('--query-gpu=utilization.gpu,memory.used')} | {q('--query-compute-apps=pid,used_memory')}\n");time.sleep(15)'''
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
    """Non-evidence full-scale probe (protocol execution.lanes_rule): real configuration, a few steps; decides 1 or 2 lanes."""
    f=ROOT/'probe/lanes.json'
    if f.exists():return json.loads(f.read_text())['lanes']
    (ROOT/'probe').mkdir(exist_ok=True);steps=str(int(os.environ.get('R78_PROBE_STEPS',CFG['execution']['probe_steps'])));runs=list(CFG['runs'])
    status('running probe (2 lanes)');ps={r:spawn(f'probe2_{r}',['fit78.py','probe',r,steps],dict(R75_LANE='2 lanes',R75_PROBE_TAG='_2lanes')) for r in runs}
    rc={r:p.wait() for r,p in ps.items()}
    total=int(subprocess.run(['nvidia-smi','--query-gpu=memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True).stdout.split()[0])*2**20
    two={r:json.loads((ROOT/'probe'/f'{r}_2lanes.json').read_text()) for r in runs if rc[r]==0}
    ok2=len(two)==len(runs) and sum(v['peak_GPU_reserved'] for v in two.values())<=.85*total;one=None
    if not ok2:
        status('running probe (1 lane)');r=runs[0];p=spawn(f'probe1_{r}',['fit78.py','probe',r,steps],dict(R75_LANE='1 lane',R75_PROBE_TAG='_1lane'))
        if p.wait():status('FAILED probe (single lane)');raise SystemExit(1)
        one=json.loads((ROOT/'probe'/f'{r}_1lane.json').read_text())
    n=2 if ok2 else 1
    f.write_text(json.dumps(dict(lanes=n,two_lane=two,return_codes=rc,single_lane=one,gpu_memory_total=total,rule=CFG['execution']['lanes_rule']),indent=1))
    status(f'probe done: {n} lanes');return n
def main():
    stage('tests',['test78.py'],'tests/tests.json',env=dict(R78_TEST_RECORD='1'))
    if 'FAIL' in (ROOT/'tests/tests.json').read_text():status('FAILED tests');raise SystemExit(1)
    (ROOT/'cost').mkdir(exist_ok=True);stage('cost_gate',['cost78.py','gate'],'cost/gate.json')
    if not json.loads((ROOT/'cost/gate.json').read_text())['passed']:status('COST INFEASIBLE at the cost gate (protocol): stop');stage('export',['export78.py'],'final/summary.json');return
    n=probe()
    lanes([(f'fit_{r}',['fit78.py','run',r],f'runs/{r}.pt') for r in CFG['runs']],n)
    (ROOT/'stage3').mkdir(exist_ok=True);stage('stage3',['eval78.py','stage3'],'stage3/analysis.json')
    (ROOT/'mech').mkdir(exist_ok=True);stage('mech',['mech78.py',str(ROOT/'mech/mech.json')],'mech/mech.json')
    gate=json.loads((ROOT/'stage3/gate.json').read_text());force=dev() and os.environ.get('R78_DEV_FORCE_GATE')=='1'
    if gate['passed'] or force:
        if force and not gate['passed']:status('DEV ONLY: Stage 4 forced to exercise the path')
        (ROOT/'stage4').mkdir(exist_ok=True);stage('stage4',['eval78.py','stage4'],'stage4/analysis.json')
    else:status(f"fit gate: {gate['classification']} - Stage 4 not run (protocol)")
    stage('cost_final',['cost78.py','final'],'cost/final.json');stage('export',['export78.py'],'final/summary.json');status('done')
if __name__=='__main__':
    (ROOT/'logs').mkdir(parents=True,exist_ok=True);s=subprocess.Popen([PY,'-c',SAMPLER,str(ROOT/'logs/gpu_sampler.log')],start_new_session=True)
    try:main()
    except SystemExit:raise
    except Exception as e:status(f'FAILED chain {e!r}');raise
    finally:os.killpg(s.pid,signal.SIGTERM)
