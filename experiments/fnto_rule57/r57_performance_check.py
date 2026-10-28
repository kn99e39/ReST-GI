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
def equivalence(N=1_000_000,chunk=2_000_000,label='equivalence'):
    x=torch.load(OLD/'data/lights.pt')['test'];rows=[]
    for name in ('rest','te0'):
        m,V=RG.mesh(name);fm,VC,_=RG.KO.refined(m,V,4);dom=RG.KO.DomainX(fm,VC)
        for fi in (0,48):
            a=invoke(dom,x[fi],N,580001+fi,chunk,'reference');b=invoke(dom,x[fi],N,580001+fi,chunk,'fast')
            er=float((a['response']-b['response']).abs().sum()/a['response'].abs().sum());s2=float((a['second']-b['second']).abs().sum()/a['second'].abs().sum())
            counts=all(a['stats'][k]==b['stats'][k] for k in ('photons','exits','leaks','wrong_side','truncated','steps','collisions'))
            ok=er<=1e-6 and s2<=1e-6 and a['rng_state_sha256']==b['rng_state_sha256'] and counts and a['physical_passed'] and b['physical_passed']
            rows.append(dict(pose=name,field=fi,N=N,chunk=chunk,response_relative_L1=er,second_relative_L1=s2,identical_RNG=a['rng_state_sha256']==b['rng_state_sha256'],identical_counts=counts,reference_seconds=a['wall_seconds'],fast_seconds=b['wall_seconds'],fast_peak_reserved=b['peak_reserved'],passed=ok))
            dump(dict(passed=all(r['passed'] for r in rows),controls=rows),DEST/f'{label}.json');print(rows[-1],flush=True);assert ok
        del dom;torch.cuda.empty_cache()
def bench(chunk,walker='fast'):
    m,V=RG.mesh('rest');fm,VC,_=RG.KO.refined(m,V,4);dom=RG.KO.DomainX(fm,VC);x=torch.load(OLD/'data/lights.pt')['test'][0]
    # Real32M unit matches primary half. No estimator/sample/geometry sweep.
    out=invoke(dom,x,32_000_000,580101,chunk,walker)
    rec={k:v for k,v in out.items() if k not in ('response','second')};dump(rec,DEST/f'batch_{walker}_{chunk}.json');print(json.dumps(rec,indent=1),flush=True)
def profile(walker):
    m,V=RG.mesh('rest');fm,VC,_=RG.KO.refined(m,V,4);dom=RG.KO.DomainX(fm,VC);x=torch.load(OLD/'data/lights.pt')['test'][0]
    os.environ['WL57_WALKER']=walker;os.environ['WL57_ACTION_CHUNK']='2000000'
    # CUPTI/torch.profiler crashed both unchanged reference and fast OptiX/Dr.Jit
    # contexts on this stack; retain failed logs and use non-invasive Python profiling.
    import cProfile,pstats,io
    p=cProfile.Profile();p.runcall(action,dom,x,250_000,580201);buf=io.StringIO();stats=pstats.Stats(p,stream=buf)
    stats.sort_stats('cumulative').print_stats(24);(DEST/f'profile_{walker}.txt').write_text(buf.getvalue())
    rec={k:sum(v[1] for name,v in stats.stats.items() if k in name[2]) for k in ("'item'","'cpu'","'nonzero'","'any'",'torch.cat')}
    rec['scope']='Python call counts/cumulative time only; no claimed CUDA kernel timeline'
    dump(rec,DEST/f'profile_{walker}.json');print(rec,flush=True)
if __name__=='__main__':
    DEST.mkdir(parents=True,exist_ok=True);torch.set_num_threads(8)
    {'equivalence':equivalence,'large_equivalence':lambda:equivalence(32_000_000,32_000_000,'large_equivalence'),'bench':lambda:bench(int(sys.argv[2]),sys.argv[3] if len(sys.argv)>3 else 'fast'),'profile':lambda:profile(sys.argv[2])}[sys.argv[1]]();sys.stdout.flush();os._exit(0)
