"""A2 server-owned diagnosis-backed single confirmation -> conditional full primary.
Source/config stage identities are mandatory; no old failed Gate1 override.
"""
import sys,os,json,time,fcntl,traceback,hashlib
from pathlib import Path
import r57_geometry as RG
from r57_data import ROOT,OUT,dump,sha
import r57_supervisor as S

CFG_PATH=Path(os.environ['WL57_CONFIRM_PROTOCOL']);CFG=json.load(open(CFG_PATH))
def identity(args):return dict(commit=(ROOT/'repo_commit.txt').read_text().strip(),protocol_sha256=sha(CFG_PATH),args=args)
def run(name,args,gate=False):
    marker=S.STATE/f'{name}.done.json';expected=identity(args)
    if marker.exists():assert json.load(open(marker)).get('recovery_identity')==expected,'Incompatible completed stage: preserve and investigate, never silently reuse'
    rc=S.run(name,args,allow_gate=gate)
    if rc==0:
        r=json.load(open(marker));r['recovery_identity']=expected;S.write(r,marker)
    return rc
def main():
    for p in (ROOT,OUT,S.LOGS,S.STATE,ROOT/'tmp',ROOT/'checkpoints'):p.mkdir(parents=True,exist_ok=True)
    lock=open(ROOT/'supervisor.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert sha(RG.HERE/'r57_model.py')==CFG['primary']['model_sha256'],'Frozen model source changed'
    execution=ROOT/'execution_policy.json'
    if execution.exists():
        ep=json.load(open(execution));assert ep['validation_passed']
        assert ep['scientific_protocol_sha256']==sha(CFG_PATH) and ep['scientific_commit']==(ROOT/'repo_commit.txt').read_text().strip()
        for filename,digest in ep['execution_source_sha256'].items():assert sha(RG.HERE/filename)==digest,'Execution source does not match validated policy'
        assert json.load(open(ROOT/'performance/large_equivalence.json'))['passed']
        # Bounded larger-batch OOM descent; effective training batch/model stay fixed.
        S.FALLBACKS=[{},dict(WL57_ACTION_CHUNK='16000000',WL57_CHECKPOINT_EDGES='1',WL57_EDGE_CHUNK='32768',WL57_EVAL_BATCH='2'),
            dict(WL57_ACTION_CHUNK='8000000',WL57_CHECKPOINT_EDGES='1',WL57_EDGE_CHUNK='8192',WL57_MICROBATCH='1',WL57_STREAM_GEOMETRY='1',WL57_EVAL_BATCH='1'),
            dict(WL57_ACTION_CHUNK='2000000',WL57_CHECKPOINT_EDGES='1',WL57_EDGE_CHUNK='8192',WL57_MICROBATCH='1',WL57_STREAM_GEOMETRY='1',WL57_EVAL_BATCH='1'),
            dict(WL57_ACTION_CHUNK='250000',WL57_CHECKPOINT_EDGES='1',WL57_EDGE_CHUNK='8192',WL57_MICROBATCH='1',WL57_STREAM_GEOMETRY='1',WL57_EVAL_BATCH='1')]
    dump(dict(pid=os.getpid(),ppid=os.getppid(),sid=os.getsid(0),identity=identity([]),execution_policy_sha256=sha(execution) if execution.exists() else None),ROOT/'supervisor.json')
    for i,c in enumerate((None,'1000000','250000')):
        if c and not execution.exists():S.FALLBACKS[i]['WL57_ACTION_CHUNK']=c
    S.status('recovery','STARTING_SINGLE_ATTEMPT')
    run('environment',['r57_environment.py'])
    run('focused',['tests/test_r57.py',str(OUT/'focused.json')])
    run('recovery_controls',['tests/test_supervisor.py',str(OUT/'supervisor_tests.json')])
    run('prepare',['r57_recovery_data.py','prepare'])
    run('pilot',['r57_recovery_checks.py','pilot'])
    run('confirm_focus',['r57_recovery_data.py','focus'])
    if run('confirm_gate',['r57_recovery_checks.py','gate'],gate=True):
        run('report',['r57_recovery_report.py']);S.status('recovery','NO_GO_CONFIRMATION');return
    S.status('recovery','GO_AUTOMATIC_PRIMARY')
    run('confirm_remaining',['r57_recovery_data.py','complete'])
    for arm,seed in [('D',5701),('R',5701),('R',5702),('D',5702)]:run(f'train_{arm}_{seed}',['r57_train.py',arm,str(seed)])
    run('heldout',['r57_eval.py','evaluate'])
    run('cost',['r57_eval.py','cost'])
    run('composition',['r57_compose.py'])
    run('report',['r57_recovery_report.py']);S.status('recovery','COMPLETE')
if __name__=='__main__':
    try:main()
    except BlockingIOError:
        print('Existing recovery supervisor holds lock; do not alter it.',flush=True);sys.exit(73)
    except Exception:
        err=traceback.format_exc();S.status('recovery','FAILED',error=err);(ROOT/'FAILURE.txt').write_text(err)
        # Preserve useful evidence even when an implementation/resource stop prevents a verdict.
        try:
            from r57_recovery_report import report
            report(failure=err)
        except Exception:print(traceback.format_exc(),flush=True)
        raise
