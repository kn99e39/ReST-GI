"""WL73 server chain (whole batch end to end, no session dependency): ports (frozen-list check + te0 identity) -> tests (pre) ->
Stage 1 reproduction (evaluation identity vs WL72, 500-step adapter replays vs the archived WL71/WL72 histories) -> references
(training + confirmation) -> bundles -> fit targets -> tests (post) -> 6 fits (2 lanes) -> evaluation -> export. Any failure stops."""
import sys,os,time,signal,subprocess,json
from pathlib import Path
from common73 import CFG,ROOT,write_new

HERE=Path(__file__).resolve().parent;PY=sys.executable
RUNENV=dict(P53_MI_VARIANT='cuda_ad_rgb',WL57_WALKER='fast',PYTHONUNBUFFERED='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
            LD_LIBRARY_PATH=str(Path.home()/'nvlibs/optix595'),R73_ROOT=str(ROOT))
def status(msg):(ROOT/'chain.status').write_text(f'{msg} {time.strftime("%Y-%m-%d %H:%M:%S")}\n');print(msg,flush=True)
def stage(name,args,done=None):
    if done and (ROOT/done).exists():status(f'skip {name} (complete)');return
    status(f'running {name}');log=open(ROOT/'logs'/f'{name}.log','a')
    r=subprocess.run([PY,*args],cwd=HERE,env=dict(os.environ,**RUNENV),stdout=log,stderr=subprocess.STDOUT)
    if r.returncode:status(f'FAILED {name} rc={r.returncode}');raise SystemExit(1)
def sampler():
    cmd=f'while true; do echo "$(date +%s) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits) | $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr "\\n" ";")" >> {ROOT}/logs/gpu_sampler.log; sleep 15; done'
    return subprocess.Popen(['bash','-c',cmd],start_new_session=True)
def lanes(kind,ids,outdir,prefix=''):
    n=int(os.environ.get('R73_LANES','2'));todo=[r for r in ids if not (ROOT/outdir/f'{prefix}{r}.pt').exists()];live={}
    while todo or live:
        while todo and len(live)<n:
            rid=todo.pop(0);status(f'running {kind}_{rid} (lanes {n})')
            live[rid]=subprocess.Popen([PY,'fit73.py',kind,rid],cwd=HERE,env=dict(os.environ,**RUNENV,R73_LANE=f'{n} lanes'),stdout=open(ROOT/'logs'/f'{kind}_{rid}.log','a'),stderr=subprocess.STDOUT)
        time.sleep(10)
        for rid,p in list(live.items()):
            if p.poll() is not None:
                del live[rid]
                if p.returncode:
                    for q in live.values():q.wait()
                    status(f'FAILED {kind}_{rid} rc={p.returncode}');raise SystemExit(1)
def stage1_summary():
    if (ROOT/'stage1/summary.json').exists():return
    sys.path.insert(0,str(HERE));os.environ.update(RUNENV);import fit73
    rows=[fit73.compare_replay(r) for r in CFG['replays']];ev=json.loads((ROOT/'stage1/eval_identity.json').read_text())
    ok=all(r['passed'] for r in rows) and ev['passed'];write_new(ROOT/'stage1/summary.json',dict(replays=rows,evaluation=ev,passed=ok))
    if not ok:status('FAILED stage1 reproduction gate');raise SystemExit(1)
def main():
    stage('ports',['ports73.py','freeze'],'ports.json');stage('tests_pre',['test73.py','pre'])
    stage('stage1_eval',['eval73.py','stage1'],'stage1/eval_identity.json');lanes('replay',list(CFG['replays']),'stage1','replay_');stage1_summary()
    if not (ROOT/'refs/complete_conf.json').exists():
        stage('generate',['probes73.py','generate']);stage('bundle_train',['probes73.py','bundle','train'],'refs/complete_train.json')
        stage('bundle_conf',['probes73.py','bundle','conf'],'refs/complete_conf.json')
    stage('prepare',['fit73.py','prepare'],'fit_targets73.pt');stage('tests_post',['test73.py','post'])
    lanes('run',list(CFG['runs']),'runs')
    stage('eval',['eval73.py','main'],'eval/analysis.json');stage('export',['export73.py'],'final/summary.json');status('done')
if __name__=='__main__':
    (ROOT/'logs').mkdir(parents=True,exist_ok=True);s=sampler()
    try:main()
    except SystemExit:raise
    except Exception as e:status(f'FAILED chain {e!r}');raise
    finally:os.killpg(s.pid,signal.SIGTERM)
