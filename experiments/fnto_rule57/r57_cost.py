"""Fresh process for EACH model/seed/pose/cache policy: isolated CUDA peaks."""
import sys,os,json,time,subprocess,resource,gc
import numpy as np
import torch
import r57_geometry as RG
from r57_data import ROOT,OUT,DATA,dump
from r57_model import Transport
from r57_memory import cache_record

def timed(fn,n):
    values=[];result=None
    for _ in range(n):
        del result;gc.collect();torch.cuda.synchronize();t=time.perf_counter();result=fn();torch.cuda.synchronize();values.append(time.perf_counter()-t)
    return result,dict(median=float(np.median(values)),p10=float(np.quantile(values,.1)),p90=float(np.quantile(values,.9)),all=values)

def worker(name,arm,seed,policy,path,untrained=False):
    torch.set_num_threads(8);torch.manual_seed(int(seed));torch.backends.cuda.matmul.allow_tf32=False
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        g,gt=timed(lambda:RG.geometry(name),1 if untrained else 3);geometry_peak=torch.cuda.max_memory_allocated()
        model=Transport(arm).cuda()
        if not untrained:
            ck=torch.load(ROOT/'checkpoints'/f'{arm}_{seed}.pt',map_location='cpu');model.load_state_dict(ck['model']);del ck
        model.eval()
        # Inputs are caller-owned; compact/full caches see identical fields. No target IO.
        x=torch.rand(64,g['P'],16,3,device='cuda') if untrained else torch.load(DATA/'lights.pt')['test'].cuda()
        x=x/x.sum((1,2),keepdim=True)
        torch.cuda.synchronize();baseline=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
        st,stt=timed(lambda:model.state(g),2 if untrained else 20)
        co,ct=timed(lambda:model.coefficients(g,st),2 if untrained else 20) if policy=='full' else (None,None)
        retained=cache_record(g,st,co,model.parameters());prep_peak=torch.cuda.max_memory_allocated()
        def query():return model.apply(g,x[:1],co) if co is not None else model(g,x[:1],st)
        for _ in range(1 if untrained else 5):query()
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();y,qt=timed(query,2 if untrained else 20)
        query_peak=torch.cuda.max_memory_allocated();out_bytes=y.numel()*y.element_size();del y,st,co
        # No persistent full coefficients retained during compact lifecycle (or vice versa).
        gc.collect();torch.cuda.empty_cache();cycles={}
        for N in ((1,) if untrained else (1,4,16,64)):
            def life():
                state=model.state(g);coeff=model.coefficients(g,state) if policy=='full' else None
                for j in range(N):
                    y0=model.apply(g,x[j:j+1],coeff) if coeff is not None else model(g,x[j:j+1],state)
                return y0
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();y,lat=timed(life,1 if untrained else 3);peak=torch.cuda.max_memory_allocated();reserved=torch.cuda.max_memory_reserved();del y
            cycles[N]=dict(lifecycle=lat,peak_allocated_bytes=peak,peak_reserved_bytes=reserved)
        dump(dict(pose=name,arm=arm,seed=seed,policy=policy,untrained=untrained,geometry=gt,geometry_peak_bytes=geometry_peak,
            memory=retained,baseline_model_geometry_inputs_bytes=baseline,state_prepare=stt,coefficient_prepare=ct,query=qt,
            cache_preparation_peak_bytes=prep_peak,query_peak_bytes=query_peak,query_output_bytes=out_bytes,CPU_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            cycles=cycles,eligible=retained['full_total_bytes' if policy=='full' else 'compact_total_bytes']<=512_000_000,
            parameters=sum(p.numel() for p in model.parameters()),process_id=os.getpid(),peak_scope='Fresh process per model/pose/policy; reset before cache preparation, query and lifecycle; absolute CUDA allocated+reserved separately.'),path)

def benchmark():
    results={};dest=OUT/'cost_workers';dest.mkdir(exist_ok=True)
    for pose in RG.TEST:
        rows={};workers={}
        for arm in ('D','R'):
            for seed in (5701,5702):
                tag=f'{arm}_{seed}';workers[tag]={}
                for policy in ('compact','full'):
                    p=dest/f'{pose}_{tag}_{policy}.json'
                    subprocess.run([sys.executable,__file__,'worker',pose,arm,str(seed),policy,str(p)],check=True)
                    workers[tag][policy]=json.load(open(p))
        common=[v for pair in workers.values() for w in pair.values() for v in w['geometry']['all']];gt=dict(median=float(np.median(common)),p10=float(np.quantile(common,.1)),p90=float(np.quantile(common,.9)),all=common)
        for tag,pair in workers.items():
            f,c=pair['full'],pair['compact'];q={}
            for N in ('1','4','16','64'):
                val={p:gt['median']+w['cycles'][N]['lifecycle']['median'] for p,w in pair.items() if w['eligible']};assert val
                selected=min(val,key=val.get)
                q[N]=dict(full=gt['median']+f['cycles'][N]['lifecycle']['median'],compact=gt['median']+c['cycles'][N]['lifecycle']['median'],selected=selected,best=val[selected],full_lifecycle=f['cycles'][N]['lifecycle'],compact_lifecycle=c['cycles'][N]['lifecycle'])
            rows[tag]=dict(geometry=gt,queries=q,workers=pair,cache_policy_total_bytes={p:w['memory']['full_total_bytes' if p=='full' else 'compact_total_bytes'] for p,w in pair.items()},parameters=f['parameters'])
        results[pose]=rows;dump(results,OUT/'cost.json')

if __name__=='__main__':
    if sys.argv[1]=='worker':worker(*sys.argv[2:7],untrained=len(sys.argv)>7 and sys.argv[7]=='untrained')
    else:benchmark()
    sys.stdout.flush();os._exit(0)
