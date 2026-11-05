"""WL74 server chain (whole batch, no session dependency): Stage 4 ports check (geometry only) -> tests -> Stage 1 bases (G0, G1 x3, G2 x3;
2 lanes) -> Stage 1 scores -> Stage 2 per basis (2 lanes) -> decision -> [only if the Stage 3 gate passes: 2 spatial fits -> Stage 4
references -> bundles -> evaluation] -> export. Any failure stops."""
import sys,os,time,signal,subprocess,json
from pathlib import Path
from common74 import CFG,ROOT,dev

HERE=Path(__file__).resolve().parent;PY=sys.executable
RUNENV=dict(P53_MI_VARIANT='cuda_ad_rgb',WL57_WALKER='fast',PYTHONUNBUFFERED='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
            LD_LIBRARY_PATH=str(Path.home()/'nvlibs/optix595'),R74_ROOT=str(ROOT))
def status(msg):(ROOT/'chain.status').write_text(f'{msg} {time.strftime("%Y-%m-%d %H:%M:%S")}\n');print(msg,flush=True)
def stage(name,args,done=None):
    if done and (ROOT/done).exists():status(f'skip {name} (complete)');return
    status(f'running {name}');log=open(ROOT/'logs'/f'{name}.log','a')
    r=subprocess.run([PY,*args],cwd=HERE,env=dict(os.environ,**RUNENV),stdout=log,stderr=subprocess.STDOUT)
    if r.returncode:status(f'FAILED {name} rc={r.returncode}');raise SystemExit(1)
def sampler():
    cmd=f'while true; do echo "$(date +%s) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits) | $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr "\\n" ";")" >> {ROOT}/logs/gpu_sampler.log; sleep 15; done'
    return subprocess.Popen(['bash','-c',cmd],start_new_session=True)
def lanes(jobs):
    """jobs: list of (name, args, done_path); at most R74_LANES concurrently."""
    n=int(os.environ.get('R74_LANES','2'));todo=[j for j in jobs if not (ROOT/j[2]).exists()];live={}
    while todo or live:
        while todo and len(live)<n:
            name,args,_=todo.pop(0);status(f'running {name} (lanes {n})')
            live[name]=subprocess.Popen([PY,*args],cwd=HERE,env=dict(os.environ,**RUNENV,R74_LANE=f'{n} lanes'),stdout=open(ROOT/'logs'/f'{name}.log','a'),stderr=subprocess.STDOUT)
        time.sleep(10)
        for name,p in list(live.items()):
            if p.poll() is not None:
                del live[name]
                if p.returncode:
                    for q in live.values():q.wait()
                    status(f'FAILED {name} rc={p.returncode}');raise SystemExit(1)
def main():
    sys.path.insert(0,str(HERE));import ato74
    stage('ports4',['ports74.py','freeze'],'ports4.json');stage('tests',['test74.py'])
    ids=ato74.basis_ids()
    lanes([(f'basis_{b}',['ato74.py','basis',b.split('_r')[0],b.split('_r')[1]],f'bases/{b}.pt') for b in ids])
    stage('stage1',['ato74.py','stage1'],'stage1/scores.json')
    lanes([(f'stage2_{b}',['ato74.py','stage2',b],f'stage2/{b}.json') for b in ids])
    stage('decide',['ato74.py','decide'],'decision.json')
    d=json.loads((ROOT/'decision.json').read_text());force=dev() and os.environ.get('R74_DEV_FORCE_GATE')=='1'
    if d['stage3_gate'] or force:
        if force and not d['stage3_gate']:status('DEV ONLY: Stage 3/4 forced to exercise the path')
        lanes([(f'spatial_{r}',['spatial74.py','run',r],f'runs/{r}.pt') for r in CFG['stage3']['runs']])
        if not (ROOT/'refs/complete.json').exists():stage('generate4',['probes74.py','generate']);stage('bundle4',['probes74.py','bundle'],'refs/complete.json')
        stage('eval4',['eval74.py'],'eval4/analysis.json')
    else:status('Stage 3 gate not passed: oracle study complete, no Stage 3/4 (protocol)')
    stage('export',['export74.py'],'final/summary.json');status('done')
if __name__=='__main__':
    (ROOT/'logs').mkdir(parents=True,exist_ok=True);s=sampler()
    try:main()
    except SystemExit:raise
    except Exception as e:status(f'FAILED chain {e!r}');raise
    finally:os.killpg(s.pid,signal.SIGTERM)
