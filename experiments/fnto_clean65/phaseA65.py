"""Bounded reference-only supervisor. No training command or automatic Phase B exists."""
import os,sys,time,json,subprocess,fcntl,traceback
from pathlib import Path
from common65 import HERE,write_new

def main():
    root=Path(os.environ['C65_ROOT']);orig=Path(os.environ['C65_ORIGINAL']);commit=os.environ['C65_COMMIT'];root.mkdir(parents=True,exist_ok=True)
    lock=open(root/'phaseA.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (root/'phaseA_status.json').exists()
    stages=[('generate',[HERE/'run65.py','generate','--root',root,'--original',orig,'--source-commit',commit]),('analyze',[HERE/'analyze65.py','--root',root,'--original',orig]),('report',[HERE/'report65.py','--root',root,'--out',root/'review'])]
    for name,args in stages:
        (root/'status.json').write_text(json.dumps({'phase':'A','stage':name,'state':'RUNNING','whole_batch_complete':False,'phase_B_authorized':False}))
        t=time.perf_counter();peers=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True)
        with open(root/f'{name}.log','w') as f:subprocess.run([sys.executable,*map(str,args)],stdout=f,stderr=subprocess.STDOUT,check=True)
        write_new(root/f'{name}_stage.json',dict(seconds=time.perf_counter()-t,GPU_processes_before=peers,GPU_processes_after=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True)))
    s=json.loads((root/'phaseA_status.json').read_text());(root/'status.json').write_text(json.dumps({'state':s['status'],'phase':'A','phaseA_complete':True,'whole_batch_complete':False,'phase_B_authorized':False}));print(s['status'],flush=True)
if __name__=='__main__':
    try:main()
    except Exception:
        r=Path(os.environ['C65_ROOT']);tb=traceback.format_exc();(r/'failure.txt').write_text(tb)
        text=''
        for p in r.glob('*.log'):text+=p.read_text(errors='replace')[-12000:]
        resource=any(t in text for t in ('out of memory','Storage gate','Deadline reached','GPU memory gate'))
        s={'state':'STOP_RESOURCE_OR_SCOPE' if resource else 'STOP_INVALID_REFERENCE','phase':'A','whole_batch_complete':False,'phase_B_authorized':False,'error':tb}
        (r/'status.json').write_text(json.dumps(s,indent=2));print(tb,flush=True);raise
