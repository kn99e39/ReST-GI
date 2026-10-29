"""Durable bounded server chain: independent of client session, no scope escalation."""
import sys,os,json,time,subprocess,fcntl,traceback,shutil
from common import *
import r57_supervisor as inherited

POLICIES=[
 dict(WL57_ACTION_CHUNK='32000000',WL57_EDGE_CHUNK='65536',WL57_EVAL_BATCH='4',P59_ATTEMPT='0'),
 dict(WL57_ACTION_CHUNK='16000000',WL57_EDGE_CHUNK='32768',WL57_EVAL_BATCH='2',WL57_CHECKPOINT_EDGES='1',P59_ATTEMPT='1'),
 dict(WL57_ACTION_CHUNK='8000000',WL57_EDGE_CHUNK='8192',WL57_EVAL_BATCH='1',WL57_CHECKPOINT_EDGES='1',WL57_MICROBATCH='1',WL57_STREAM_GEOMETRY='1',P59_ATTEMPT='2'),
 dict(WL57_ACTION_CHUNK='2000000',WL57_EDGE_CHUNK='8192',WL57_EVAL_BATCH='1',WL57_CHECKPOINT_EDGES='1',WL57_MICROBATCH='1',WL57_STREAM_GEOMETRY='1',P59_ATTEMPT='3')]

def status(stage,state,**kw):
    r=dict(stage=stage,status=state,pid=os.getpid(),time=time.strftime('%Y-%m-%dT%H:%M:%S%z'),**kw)
    dump(r,ROOT/'status.json');print(json.dumps(r),flush=True)

def run(stage,args):
    marker=ROOT/'stages'/f'{stage}.json';identity=dict(source=(ROOT/'repo_commit.txt').read_text().strip(),protocol=sha(HERE/'protocol/persist_v1.json'),args=args)
    if marker.exists():
        assert json.loads(marker.read_text())['identity']==identity,'Never reuse a completed stage from a different experiment identity'
        return
    prior=len(list((ROOT/'logs').glob(f'{stage}.*.log')))
    for level,policy in enumerate(POLICIES):
        log=ROOT/'logs'/f'{stage}.{prior+level}.log'
        status(stage,'RUNNING',attempt=prior+level,resource_policy=level)
        env=dict(os.environ,**policy);started=time.time()
        with log.open('wb') as f:
            proc=subprocess.Popen(['bash',str(HERE/'run.sh')]+args,cwd=HERE,env=env,stdout=f,stderr=subprocess.STDOUT,preexec_fn=inherited.parent_death_signal)
            rc=proc.wait()
        record=dict(identity=identity,returncode=rc,wall_seconds=time.time()-started,resource_policy=level,environment=policy,log=str(log))
        dump(record,log.with_suffix('.json'))
        if rc==0:
            dump(record,marker);status(stage,'DONE',wall_seconds=record['wall_seconds']);return
        text=log.read_text(errors='replace')[-24000:].lower()
        if not any(s in text for s in inherited.OOM):raise RuntimeError(f'{stage}: implementation or primary gate failure; see {log}')
        failed=ROOT/'failed_attempts'/f'{stage}_{prior+level}';failed.mkdir(parents=True,exist_ok=True)
        for ck in (ROOT/'checkpoints').glob('*progress.pt'):shutil.copy2(ck,failed/ck.name)
        status(stage,'OOM_FALLBACK',failed_attempt=prior+level,next_policy=level+1)
    raise RuntimeError(f'{stage}: finite OOM policies exhausted; original logs/checkpoints retained')

def main():
    for p in (ROOT,OUT,ROOT/'logs',ROOT/'stages',ROOT/'checkpoints',ROOT/'tmp'):p.mkdir(parents=True,exist_ok=True)
    lock=(ROOT/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    dump(dict(pid=os.getpid(),ppid=os.getppid(),sid=os.getsid(0),source=(ROOT/'repo_commit.txt').read_text().strip()),ROOT/'supervisor.json')
    assert shutil.disk_usage(ROOT).free>=60*1024**3,'Stop without deleting shared server files'
    run('environment',['environment.py'])
    run('preflight',['tests.py'])
    run('prepare',['data.py','prepare'])
    run('regressions',[str(HERE.parent/'fnto_rule57/r57_regressions.py')])
    run('reference',['data.py','reference'])
    run('reference_check',['reference_check.py'])
    for seed in CFG['training']['seeds']:run(f'train_{seed}',['train.py','P',str(seed),'6000'])
    run('heldout',['evaluate.py'])
    run('cost',['cost.py'])
    run('composition',['compose.py'])
    run('report',['report.py'])
    status('batch','COMPLETE')

if __name__=='__main__':
    try:main()
    except BlockingIOError:print('Live supervisor already holds the batch lock; leave it unchanged.',flush=True);sys.exit(73)
    except Exception:
        error=traceback.format_exc();status('batch','FAILED',error=error);(ROOT/'FAILURE.txt').write_text(error)
        try:
            import report
            report.main()
        except Exception:print(traceback.format_exc(),flush=True)
        print(error,flush=True);sys.exit(1)
