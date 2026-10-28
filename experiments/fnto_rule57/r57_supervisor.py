"""Server-owned durable experiment chain with bounded resource fallbacks.

No architecture/threshold/step-budget changes. OOM retries resume atomic checkpoints.
Unknown errors fail closed; completed stages and all attempt logs are retained.
"""
import os,sys,json,time,fcntl,subprocess,shutil,traceback,signal,ctypes
from pathlib import Path

ROOT=Path(os.environ.get('WL57',Path.home()/'rest_wl57'));HERE=ROOT/'repo/experiments/fnto_rule57';OUT=ROOT/'out';LOGS=ROOT/'logs';STATE=ROOT/'stages'
FALLBACKS=[{},
 dict(WL57_EDGE_CHUNK='32768',WL57_CHECKPOINT_EDGES='1',WL57_RAY_CHUNK='1000000',WL57_TR_CHUNK='1000000',WL57_EVAL_BATCH='2'),
 dict(WL57_EDGE_CHUNK='8192',WL57_CHECKPOINT_EDGES='1',WL57_MICROBATCH='1',WL57_STREAM_GEOMETRY='1',WL57_RAY_CHUNK='250000',WL57_TR_CHUNK='250000',WL57_EVAL_BATCH='1')]
OOM=('cuda out of memory','cuda error: out of memory','cublas_status_alloc_failed','cuda_error_out_of_memory','outofmemoryerror')
def parent_death_signal():
    # Linux server only: a supervisor crash cannot leave an untracked stage writer.
    ctypes.CDLL(None).prctl(1,signal.SIGTERM)
def write(x,p):
    p=Path(p);tmp=p.with_name(p.name+'.partial');tmp.write_text(json.dumps(x,indent=1,allow_nan=False)+'\n');os.replace(tmp,p)
def status(stage,state,**kw):
    rec=dict(time=time.strftime('%Y-%m-%dT%H:%M:%S%z'),stage=stage,status=state,pid=os.getpid(),**kw);write(rec,ROOT/'status.json');(ROOT/'status.txt').write_text(f"{rec['time']} stage={stage} status={state}\n");print(json.dumps(rec),flush=True)
def run(stage,args,allow_gate=False):
    done=STATE/f'{stage}.done.json'
    if done.exists():status(stage,'REUSED_COMPLETED_STAGE');return 0
    old=list(LOGS.glob(f'{stage}.attempt*.json'));begin=max([int(p.stem.split('attempt')[-1]) for p in old]+[-1])+1
    for level,extra in enumerate(FALLBACKS):
        attempt=begin+level;log=LOGS/f'{stage}.attempt{attempt}.log';meta=log.with_suffix('.json');env=dict(os.environ,**extra);env['WL57_SUPERVISED']='1'
        status(stage,'RUNNING',attempt=attempt,resource_policy=level,environment=extra,log=str(log));t0=time.time()
        before=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
        runner=Path(os.environ.get('WL57_RUNNER',HERE/'run.sh'))
        with open(log,'wb') as f:p=subprocess.Popen(['bash',str(runner)]+args,cwd=HERE,env=env,stdout=f,stderr=subprocess.STDOUT,preexec_fn=parent_death_signal);rc=p.wait()
        rec=dict(stage=stage,args=args,attempt=attempt,resource_policy=level,environment=extra,returncode=rc,wall_seconds=time.time()-t0,GPU_processes_before=before,commit=(ROOT/'repo_commit.txt').read_text().strip(),log=str(log));write(rec,meta)
        if rc==0:write(rec,done);status(stage,'DONE',attempt=attempt,wall_seconds=rec['wall_seconds']);return 0
        # No memory fallback is selected on validation/accuracy. Only explicit allocation errors qualify.
        tail=log.read_text(errors='replace')[-20000:].lower();oom=any(s in tail for s in OOM)
        if allow_gate and not oom and (OUT/'gate1.json').exists() and not json.load(open(OUT/'gate1.json'))['proceed']:
            status(stage,'STOP_GATE1');return rc
        if not oom:raise RuntimeError(f'{stage} failed rc={rc}; no automatic retry for an unknown defect; see {log}')
        saved=ROOT/'failed_attempts'/f'{stage}_{attempt}';saved.mkdir(parents=True,exist_ok=True)
        for p in (ROOT/'checkpoints').glob('*_progress.pt'):shutil.copy2(p,saved/p.name)
        for filename in ('heldout.json','cost.json'):
            if (OUT/filename).exists():shutil.copy2(OUT/filename,saved/filename)
        status(stage,'OOM_FALLBACK',failed_attempt=attempt,next_resource_policy=level+1)
        if level+1<len(FALLBACKS):time.sleep(10)
    raise RuntimeError(f'{stage}: all three resource policies exhausted; original attempts/checkpoints preserved')

def main():
    for p in (ROOT,OUT,LOGS,STATE,ROOT/'tmp',ROOT/'checkpoints'):p.mkdir(parents=True,exist_ok=True)
    lock=open(ROOT/'supervisor.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    gate=OUT/'preflight_v2/decision.json'
    if not gate.exists() or json.load(open(gate)).get('verdict')!='GO':raise RuntimeError('Primary disabled until corrected preflight GO')
    if json.load(open(gate))['primary_commit']!=(ROOT/'repo_commit.txt').read_text().strip():raise RuntimeError('Preflight commit does not match primary source')
    if (ROOT/'primary_disabled.json').exists():raise RuntimeError('Explicit user preflight stop is still active')
    write(dict(pid=os.getpid(),sid=os.getsid(0),ppid=os.getppid(),start=time.time(),commit=(ROOT/'repo_commit.txt').read_text().strip()),ROOT/'supervisor.json')
    status('batch','STARTING')
    if shutil.disk_usage(ROOT).free<60*1024**3:raise RuntimeError('Less than60GiB scratch free; stop without deleting shared files')
    run('environment',['r57_environment.py'])
    run('focused',['tests/test_r57.py',str(OUT/'tests/focused.json')])
    run('supervisor_tests',['tests/test_supervisor.py',str(OUT/'tests/supervisor.json')])
    # Completed unchanged historical suite may be adopted after the wrapper-only transition.
    reg=OUT/'tests/regressions.json'
    if reg.exists() and all(r['returncode']==0 for r in json.load(open(reg))):
        write(dict(adopted_completed_unchanged_regression=True,source=str(reg)),STATE/'regressions.done.json')
    run('regressions',['r57_regressions.py'])
    run('supervision',['r57_data.py','build'])
    if run('data_gate',['r57_data.py','gate'],allow_gate=True)!=0:
        run('report',['r57_report.py']);status('batch','STOPPED_GATE1');return
    for arm,seed in [('D',5701),('R',5701),('R',5702),('D',5702)]:run(f'train_{arm}_{seed}',['r57_train.py',arm,str(seed)])
    run('heldout',['r57_eval.py','evaluate']);run('cost',['r57_eval.py','cost']);run('composition',['r57_compose.py']);run('report',['r57_report.py']);status('batch','COMPLETE')
if __name__=='__main__':
    try:main()
    except BlockingIOError:
        print('Another WL57 supervisor owns the lock; leave its status and processes unchanged.',flush=True);sys.exit(73)
    except Exception:
        err=traceback.format_exc();status('batch','FAILED',error=err);(ROOT/'FAILURE.txt').write_text(err);print(err,flush=True);sys.exit(1)
