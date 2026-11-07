"""WL78 Stage 5 computational cost (measured on the same GPU, run alone at the end of the chain): A75_5901 vs Q78_5901 (and the initial
W+UV-Q before training at the cost-gate stage). Parameters; geometry-dependent cached state (W, U, V floats + the geometry input tensors);
geometry rebuild = W+UV-Q input construction (frames + boundary probes + frame-expressed pair offsets) + coefficient generation; 64-light
(batches of 4) and 1-light apply; peak GPU memory; training time from the run records. Median of repeated timings after warm-up.
    python cost78.py gate | final"""
import sys,os,json,time,subprocess
import torch
from common78 import CFG,ROOT,R75,INP68,write_new,environment,dev,load
RG=environment()
import rank75 as RK,q78 as Q,fit71

def med(f,n,warm=3):
    for _ in range(warm):f()
    ts=[]
    for _ in range(n):torch.cuda.synchronize();t=time.perf_counter();f();torch.cuda.synchronize();ts.append(time.perf_counter()-t)
    ts.sort();return dict(median=ts[len(ts)//2],min=ts[0],max=ts[-1],n=n)

def measure(m,g,lights,n,q):
    out=dict(parameters=RK.parameter_counts(m))
    with torch.no_grad():
        if q:
            out['feature_build_seconds']=med(lambda:Q.features(RG,g),n)
            def rebuild():Q._CACHE.clear();return m.coefficients(dict(g))
        else:
            out['feature_build_seconds']=dict(median=0.,min=0.,max=0.,n=0)
            def rebuild():return m.coefficients(dict(g))
        co=m.coefficients(g);out['state_floats']={k:int(v.numel()) for k,v in co.items()};out['state_bytes_WUV']=sum(out['state_floats'][k] for k in ('W','U','V'))*4
        gg=Q.QG(RG,g) if q else g;out['geometry_input_bytes']=int(gg['node'][:,:16].numel()*4+gg['pair'].numel()*4)
        del co;torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();base=torch.cuda.memory_allocated()
        out['rebuild_seconds']=med(rebuild,n);co=m.coefficients(g)
        out['apply64_seconds']=med(lambda:[m.apply(g,lights[i:i+4],co) for i in range(0,64,4)],max(3,n//2))
        out['apply1_seconds']=med(lambda:m.apply(g,lights[:1],co),n)
        out['peak_GPU_above_geometry']=torch.cuda.max_memory_allocated()-base;del co
    torch.cuda.empty_cache();return out

def main(kind):
    torch.backends.cuda.matmul.allow_tf32=False;torch.set_grad_enabled(False);out=ROOT/'cost'/f'{kind}.json'
    if out.exists():raise FileExistsError('immutable')
    apps=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
    g=RG.device(fit71.guarded('geometry_rest'));lights=load(INP68/'wl59/lights.pt')['test'].cuda();n=3 if dev() else 10;res={}
    m=RK.load_run(R75/'runs/A75_5901.pt');res['A75_5901']=measure(m,g,lights,n,False);del m
    if kind=='gate':m,_=Q.init_model(RG,'P_5901',7801);m.eval();tag='Q78_init_5901'
    else:m=Q.load_run(RG,ROOT/'runs/Q78_5901.pt');tag='Q78_5901'
    res[tag]=measure(m,g,lights,n,True);del m;torch.cuda.empty_cache()
    a,b=res['A75_5901'],res[tag]
    ratio=dict(apply64=b['apply64_seconds']['median']/a['apply64_seconds']['median'],apply1=b['apply1_seconds']['median']/a['apply1_seconds']['median'],
               rebuild_incl_features=(b['rebuild_seconds']['median'])/a['rebuild_seconds']['median'],
               geometry_state_bytes=(b['state_bytes_WUV']+b['geometry_input_bytes'])/(a['state_bytes_WUV']+int(g['node'][:,:16].numel()*4+g['pair'].numel()*4)),
               peak_GPU=b['peak_GPU_above_geometry']/max(a['peak_GPU_above_geometry'],1))
    lim=CFG['stage5']['cost_limit'];passed=all(ratio[k]<=lim for k in CFG['stage5']['gated'])
    tr=None
    if kind=='final':tr={n_:{k:load(ROOT/'runs'/f'{n_}.pt').get(k) for k in ('seconds','peak_GPU_allocated','concurrent','trainable')} for n_ in CFG['runs']}
    write_new(out,json.loads(json.dumps(dict(models=res,ratio=ratio,limit=lim,gated=CFG['stage5']['gated'],passed=passed,training=tr,
        A75_training=dict(seconds=4084,note='WL75 local RTX 5080, 2 lanes (run records); not the same GPU'),gpu=torch.cuda.get_device_name(0),other_gpu_processes_at_start=apps,timing_repeats=n),default=float)))
    print('COST_DONE',kind,'passed' if passed else 'EXCEEDED',json.dumps(ratio),flush=True)

if __name__=='__main__':main(sys.argv[1]);sys.stdout.flush();os._exit(0)
