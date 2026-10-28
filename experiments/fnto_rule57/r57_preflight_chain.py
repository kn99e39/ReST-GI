"""Only focused preflight first. Latest user instruction: automatically launch primary IF final GO."""
import sys,os,json,subprocess,time,fcntl,traceback
from pathlib import Path
from r57_data import ROOT,OUT,dump
from r57_resource import PF

def main():
    PF.mkdir(parents=True,exist_ok=True);lock=open(ROOT/'preflight.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    steps=[('focused',['tests/test_r57.py',str(PF/'focused.json')]),('controls',['tests/test_preflight.py',str(PF/'controls.json')]),('recovery',['tests/test_supervisor.py',str(PF/'recovery.json')]),('resources',['r57_resource.py'])]
    records=[];go=True
    for stage,args in steps:
        dump(dict(stage='preflight_'+stage,status='RUNNING',pid=os.getpid()),ROOT/'status.json');t0=time.time()
        with open(PF/f'{stage}.log','wb') as f:rc=subprocess.run([sys.executable]+args,cwd=Path(__file__).parent,stdout=f,stderr=subprocess.STDOUT).returncode
        records.append(dict(stage=stage,rc=rc,wall_seconds=time.time()-t0));dump(records,PF/'stages.json')
        if rc:go=False;break
    commit=(ROOT/'repo_commit.txt').read_text().strip()
    dump(dict(verdict='GO' if go else 'NO-GO',primary_commit=commit,stages=records,claim='Matched-context comparison of the SPECIFIC learned graph-state mechanism vs direct raw-context decoder, not a causal proof that abstract state separation itself helps.',user_auto_launch_authorization='Latest user instruction: final GO starts server chain automatically',time=time.strftime('%Y-%m-%dT%H:%M:%S%z')),PF/'decision.json')
    if not go:
        dump(dict(stage='preflight',status='NO_GO',details=str(PF/'decision.json')),ROOT/'status.json');return
    # The previous user stop remains an auditable artifact, never silently discarded.
    stopped=ROOT/'primary_disabled.json'
    if stopped.exists():os.replace(stopped,PF/'prior_primary_stop.json')
    # Release GPU allocations by exec into a fresh runtime; detached server ownership persists.
    os.execv('/bin/bash',['bash',str(Path(__file__).parent/'chain.sh')])
if __name__=='__main__':
    try:main()
    except Exception:
        PF.mkdir(parents=True,exist_ok=True);dump(dict(verdict='NO-GO',error=traceback.format_exc()),PF/'decision.json');raise
