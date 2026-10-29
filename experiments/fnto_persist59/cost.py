import os,sys,time,json,subprocess,resource
import numpy as np
import torch
from common import *
from runtime import full, selective, apply
from evaluate import model_of
from r57_memory import bytes_of

def peers():
    text=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
    rows=[v for v in text.splitlines() if v.strip() and int(v.split(',')[0])!=os.getpid()]
    return rows

def worker(policy,seed,edit,out):
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
    m=model_of('P',seed);x=torch.load(DATA/'lights.pt')['test'].cuda();records=[];before=peers()
    with torch.no_grad():
        for rep in range(7):
            torch.cuda.synchronize();t=time.perf_counter();g0=geometry('rest');c=full(m,g0);torch.cuda.synchronize();setup=time.perf_counter()-t
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.perf_counter()
            g=geometry(PRIMARY[0]) if edit=='dent' else rigid_geometry(ballistic=policy=='F')
            torch.cuda.synchronize()
            geometry_seconds=time.perf_counter()-t
            audit={}
            if policy=='F':c=full(m,g)
            else:
                c,audit=selective(m,c,g);audit.pop('masks')
            torch.cuda.synchronize();update=time.perf_counter()-t
            cumulative={}
            for i in range(64):
                y=apply(m,c,x[i:i+1])
                if i+1 in (1,4,16,64):
                    torch.cuda.synchronize();cumulative[str(i+1)]=time.perf_counter()-t
            records.append(dict(rep=rep,initial_rest_seconds=setup,geometry_and_B_seconds=geometry_seconds,update_seconds=update,
                                lifecycle_seconds=cumulative,persistent_GPU_bytes=bytes_of(c),
                                peak_GPU_allocated=torch.cuda.max_memory_allocated(),peak_GPU_reserved=torch.cuda.max_memory_reserved(),
                                CPU_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,audit=audit))
            del c,g,g0,y
            torch.cuda.empty_cache()
    dump(dict(policy=policy,seed=seed,edit=edit,records=records,peers_before=before,peers_after=peers(),uncontended=not before and not peers()),out)

def main():
    attempt=os.environ.get('P59_ATTEMPT','0');folder=OUT/f'cost_attempt_{attempt}';folder.mkdir(exist_ok=True)
    records={}
    # Alternate F/S order across seeds; every worker is a fresh process, no inherited allocator peak.
    for seed in CFG['training']['seeds']:
        for edit in ('dent','rigid'):
            for p in (('F','S') if seed==5901 else ('S','F')):
                path=folder/f'{seed}_{edit}_{p}.json'
                if not path.exists():subprocess.run([sys.executable,__file__,'worker',p,str(seed),edit,str(path)],check=True)
                records[f'{seed}_{edit}_{p}']=json.loads(path.read_text())
    rng=np.random.default_rng(592001);comparisons=[]
    for seed in CFG['training']['seeds']:
        for edit in ('dent','rigid'):
            f=records[f'{seed}_{edit}_F'];s=records[f'{seed}_{edit}_S']
            for n in (1,4,16,64):
                a=np.array([r['lifecycle_seconds'][str(n)] for r in f['records']]);b=np.array([r['lifecycle_seconds'][str(n)] for r in s['records']])
                ratio=float(np.median(b)/np.median(a));ix=rng.integers(0,7,(10000,7));boot=np.median(b[ix],axis=1)/np.median(a[ix],axis=1)
                comparisons.append(dict(seed=seed,edit=edit,queries=n,F_seconds=float(np.median(a)),S_seconds=float(np.median(b)),
                                        ratio=ratio,bootstrap95=np.quantile(boot,[.025,.975]).tolist(),uncontended=f['uncontended'] and s['uncontended']))
    dump(dict(attempt=attempt,workers=records,comparisons=comparisons,kind='actual intrinsic update+queries; C/solve eligibility recorded separately'),OUT/'cost.json')

if __name__=='__main__':
    if len(sys.argv)>1:worker(sys.argv[2],int(sys.argv[3]),sys.argv[4],Path(sys.argv[5]))
    else:main()
    sys.stdout.flush();os._exit(0)
