"""WL76 server chain (whole remaining batch, no session dependency): Stage 1 result check -> Stage 4 population freeze (geometry only) ->
tests -> full-scale probe (non-evidence; lanes) -> two FIA fits -> Stage 3 + gates -> [gates passed: references -> bundle -> Stage 4] ->
export. Any failure stops."""
import sys,os,time,json,subprocess
from pathlib import Path
from common76 import CFG,ROOT,dev

HERE=Path(__file__).resolve().parent;PY=sys.executable
RUNENV=dict(P53_MI_VARIANT='cuda_ad_rgb',WL57_WALKER='fast',PYTHONUNBUFFERED='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
            LD_LIBRARY_PATH=str(Path.home()/'nvlibs/optix595'),R76_ROOT=str(ROOT))
def status(msg):(ROOT/'chain.status').write_text(f'{msg} {time.strftime("%Y-%m-%d %H:%M:%S")}\n');print(msg,flush=True)
def spawn(name,args,env=None):
    return subprocess.Popen([PY,*args],cwd=HERE,env=dict(os.environ,**RUNENV,**(env or {})),stdout=open(ROOT/'logs'/f'{name}.log','a'),stderr=subprocess.STDOUT)
def stage(name,args,done=None,env=None):
    if done and (ROOT/done).exists():status(f'skip {name} (complete)');return
    status(f'running {name}')
    if spawn(name,args,env).wait():status(f'FAILED {name}');raise SystemExit(1)
SAMPLER=r'''import subprocess,time,sys
q=lambda a:subprocess.run(['nvidia-smi',a,'--format=csv,noheader'+(',nounits' if 'gpu=' in a else '')],capture_output=True,text=True).stdout.strip().replace('\n',';')
while True:
    open(sys.argv[1],'a').write(f"{int(time.time())} {q('--query-gpu=utilization.gpu,memory.used')} | {q('--query-compute-apps=pid,used_memory')}\n");time.sleep(15)'''
def lanes(jobs,n):
    todo=[j for j in jobs if not (ROOT/j[2]).exists()];live={}
    while todo or live:
        while todo and len(live)<n:name,args,_=todo.pop(0);status(f'running {name} (lanes {n})');live[name]=spawn(name,args,dict(R76_LANE=f'{n} lanes'))
        time.sleep(10)
        for name,p in list(live.items()):
            if p.poll() is not None:
                del live[name]
                if p.returncode:
                    for q in live.values():q.wait()
                    status(f'FAILED {name}');raise SystemExit(1)
def probe():
    f=ROOT/'probe/lanes.json'
    if f.exists():return json.loads(f.read_text())['lanes']
    (ROOT/'probe').mkdir(exist_ok=True);steps=os.environ.get('R76_PROBE_STEPS',str(CFG['execution']['probe_steps']));runs=list(CFG['runs'])
    status('running probe (2 lanes)');ps={r:spawn(f'probe2_{r}',['fit76.py','probe',r,steps],dict(R76_LANE='2 lanes',R76_PROBE_TAG='_2lanes')) for r in runs};rc={r:p.wait() for r,p in ps.items()}
    total=int(subprocess.run(['nvidia-smi','--query-gpu=memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True).stdout.split()[0])*2**20
    two={r:json.loads((ROOT/'probe'/f'{r}_2lanes.json').read_text()) for r in runs if rc[r]==0}
    ok=len(two)==len(runs) and sum(v['peak_GPU_reserved'] for v in two.values())<=.85*total;one=None
    if not ok:
        status('running probe (1 lane)');p=spawn(f'probe1_{runs[0]}',['fit76.py','probe',runs[0],steps],dict(R76_LANE='1 lane',R76_PROBE_TAG='_1lane'))
        if p.wait():status('FAILED probe (single lane)');raise SystemExit(1)
        one=json.loads((ROOT/'probe'/f'{runs[0]}_1lane.json').read_text())
    n=2 if ok else 1;f.write_text(json.dumps(dict(lanes=n,two_lane=two,return_codes=rc,single_lane=one,gpu_memory_total=total),indent=1));status(f'probe done: {n} lanes');return n
def main():
    s1=ROOT/'stage1/physics.json' if not dev() else Path(os.environ.get('R76_STAGE1',str(ROOT/'stage1/physics.json')))
    if not s1.exists():status('STOP: Stage 1 record missing');raise SystemExit(1)
    g1=json.loads(s1.read_text())['gate'];am=[a for a in CFG['amendments'] if a['id']==1]
    ok=g1['passed'] or (am and sorted(map(list,g1['failures']))==sorted(map(list,am[0]['accepted_failures'])))   # amendment 1 (post hoc, user decision)
    if not ok:status('STOP: Stage 1 physical gate not passed');raise SystemExit(1)
    stage('ports',['ports76.py','freeze'],'ports76.json');stage('tests',['test76.py'],env=dict(R76_TEST_RECORD='1'))
    n=probe();lanes([(f'fit_{r}',['fit76.py','run',r],f'runs/{r}.pt') for r in CFG['runs']],n)
    stage('stage3',['eval76.py','stage3'],'stage3/analysis.json')
    gate=json.loads((ROOT/'stage3/gate.json').read_text());force=dev() and os.environ.get('R76_DEV_FORCE_GATE')=='1'
    if gate['passed'] or force:
        if force and not gate['passed']:status('DEV ONLY: Stage 4 forced')
        assert dev() or 'R74_SMOKE_N' not in os.environ
        if not (ROOT/'refs/complete.json').exists():stage('generate',['probes76.py','generate']);stage('bundle',['probes76.py','bundle'],'refs/complete.json')
        stage('stage4',['eval76.py','stage4'],'stage4/analysis.json')
    else:status(f"Stage 3 gates: {gate['classification']} - no Stage 4 references (protocol)")
    stage('export',['export76.py'],'final/summary.json');status('done')
if __name__=='__main__':
    (ROOT/'logs').mkdir(parents=True,exist_ok=True);s=subprocess.Popen([PY,'-c',SAMPLER,str(ROOT/'logs/gpu_sampler.log')])
    try:main()
    except SystemExit:raise
    except Exception as e:status(f'FAILED chain {e!r}');raise
    finally:s.terminate()
