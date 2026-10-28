"""Fault-injected orchestration test; no real GPU fault/process termination."""
import sys,os,json,uuid
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import r57_supervisor as S

def main(out):
    root=out.parent/('supervisor_faults_'+uuid.uuid4().hex[:8]);root.mkdir(parents=True)
    for d in ('logs','stages','out','checkpoints'): (root/d).mkdir()
    (root/'repo_commit.txt').write_text('fault-injection-only\n')
    S.ROOT=root;S.LOGS=root/'logs';S.STATE=root/'stages';S.OUT=root/'out';calls=[];queue=[]
    class Fake:
        def __init__(self,args,**kw):
            self.rc,msg=queue.pop(0);kw['stdout'].write(msg.encode());calls.append(dict(args=args,env=kw['env']))
        def wait(self):return self.rc
    with patch.object(S.subprocess,'Popen',Fake),patch.object(S.subprocess,'check_output',return_value='test GPU'),patch.object(S.time,'sleep'):
        queue[:]=[(1,'torch.OutOfMemoryError: CUDA out of memory'),(0,'success')];assert S.run('recover',['fake.py'])==0 and len(calls)==2
        assert calls[-1]['env']['WL57_CHECKPOINT_EDGES']=='1'
        n=len(calls);S.run('recover',['fake.py']);assert len(calls)==n
        queue[:]=[(1,'ValueError: deliberate unknown defect')]
        try:S.run('unknown',['fake.py']);raise AssertionError('Unknown error retried')
        except RuntimeError:pass
        assert len(calls)==n+1
        queue[:]=[(1,'CUDA out of memory')]*3;n=len(calls)
        try:S.run('exhausted',['fake.py']);raise AssertionError('Exhaustion ignored')
        except RuntimeError:pass
        assert len(calls)==n+3 and calls[-1]['env']['WL57_MICROBATCH']=='1'
    out.write_text(json.dumps(dict(passed=4,total=4,OOM_retry=True,completed_stage_skip=True,unknown_error_no_retry=True,finite_resource_exhaustion=True,artifacts=str(root)),indent=1)+'\n')
if __name__=='__main__':main(Path(sys.argv[1]))
