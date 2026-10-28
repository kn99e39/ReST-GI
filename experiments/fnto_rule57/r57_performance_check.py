"""Execution-only profiling/validation on OLD diagnostic fixtures, never new signals.
Fixed sampler/photons/materials. Larger chunks change RNG partition only; choose by
wall time and memory, never target accuracy. Reference-vs-fast at identical chunk
must preserve RNG state, counts and responses before using the specialization.
"""
import os,sys,json,time,gc,subprocess
from pathlib import Path
import torch
import r57_geometry as RG
from r57_action_oracle import action
from r57_data import dump
OLD=Path.home()/'rest_wl57';DEST=Path.home()/'rest_wl57_recovery/performance'
def invoke(dom,x,N,seed,chunk,walker):
    os.environ['WL57_ACTION_CHUNK']=str(chunk);os.environ['WL57_WALKER']=walker
    gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
    out=action(dom,x,N,seed);torch.cuda.synchronize()
    out['wall_seconds']=time.perf_counter()-t;out['peak_reserved']=torch.cuda.max_memory_reserved();out['peak_allocated']=torch.cuda.max_memory_allocated();return out
def equivalence():
    x=torch.load(OLD/'data/lights.pt')['test'];rows=[]
    for name in ('rest','te0'):
        m,V=RG.mesh(name);fm,VC,_=RG.KO.refined(m,V,4);dom=RG.KO.DomainX(fm,VC)
        for fi in (0,48):
            a=invoke(dom,x[fi],1_000_000,580001+fi,2_000_000,'reference');b=invoke(dom,x[fi],1_000_000,580001+fi,2_000_000,'fast')
            er=float((a['response']-b['response']).abs().sum()/a['response'].abs().sum());s2=float((a['second']-b['second']).abs().sum()/a['second'].abs().sum())
            counts=all(a['stats'][k]==b['stats'][k] for k in ('photons','exits','leaks','wrong_side','truncated','steps','collisions'))
            ok=er<=1e-6 and s2<=1e-6 and a['rng_state_sha256']==b['rng_state_sha256'] and counts and a['physical_passed'] and b['physical_passed']
            rows.append(dict(pose=name,field=fi,response_relative_L1=er,second_relative_L1=s2,identical_RNG=a['rng_state_sha256']==b['rng_state_sha256'],identical_counts=counts,reference_seconds=a['wall_seconds'],fast_seconds=b['wall_seconds'],passed=ok))
            dump(dict(passed=all(r['passed'] for r in rows),controls=rows),DEST/'equivalence.json');print(rows[-1],flush=True);assert ok
        del dom;torch.cuda.empty_cache()
def bench(chunk):
    m,V=RG.mesh('rest');fm,VC,_=RG.KO.refined(m,V,4);dom=RG.KO.DomainX(fm,VC);x=torch.load(OLD/'data/lights.pt')['test'][0]
    # Real32M unit matches primary half. No estimator/sample/geometry sweep.
    out=invoke(dom,x,32_000_000,580101,chunk,'fast')
    rec={k:v for k,v in out.items() if k not in ('response','second')};dump(rec,DEST/f'batch_{chunk}.json');print(json.dumps(rec,indent=1),flush=True)
def profile(walker):
    m,V=RG.mesh('rest');fm,VC,_=RG.KO.refined(m,V,4);dom=RG.KO.DomainX(fm,VC);x=torch.load(OLD/'data/lights.pt')['test'][0]
    os.environ['WL57_WALKER']=walker;os.environ['WL57_ACTION_CHUNK']='2000000'
    from torch.profiler import profile as profiling,ProfilerActivity
    with profiling(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as p:action(dom,x,250_000,580201)
    (DEST/f'profile_{walker}.txt').write_text(p.key_averages().table(sort_by='self_cpu_time_total',row_limit=20))
    rec={k:sum(e.count for e in p.key_averages() if k in e.key) for k in ('aten::item','aten::_local_scalar_dense','cudaStreamSynchronize','cudaDeviceSynchronize','aten::nonzero','aten::cat')}
    dump(rec,DEST/f'profile_{walker}.json');print(rec,flush=True)
if __name__=='__main__':
    DEST.mkdir(parents=True,exist_ok=True);torch.set_num_threads(8)
    {'equivalence':equivalence,'bench':lambda:bench(int(sys.argv[2])),'profile':lambda:profile(sys.argv[2])}[sys.argv[1]]();sys.stdout.flush();os._exit(0)
