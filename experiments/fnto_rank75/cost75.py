"""WL75 Stage 5 computational feasibility (measured, same GPU, run alone at the end of the chain): rank 8 (AR_5901 through the historical
runtime class and through the rank-generic class) vs rank 16 (A75_5901). Parameters, angular parameters, geometry-dependent state
(W, U, V floats), full coefficient rebuild from geometry (state + W/U/V/A/O), apply latency (64 test lights in batches of 4; one light),
peak GPU memory; rank-independent geometry preparation (cf1 build with ballistic B, once); training time / peak from the run records;
physical reference cost; angular pretraining (G2 basis fit). Median of repeated timings after warm-up.
    python cost75.py"""
import sys,os,json,time,subprocess
import torch
from common75 import CFG,ROOT,R74,R73,INP68,write_new,environment,dev
import rank75 as RK

def load(p,mmap=False):return torch.load(p,map_location='cpu',weights_only=False,mmap=mmap)
def med(f,n,warm=3):
    for _ in range(warm):f()
    ts=[]
    for _ in range(n):torch.cuda.synchronize();t=time.perf_counter();f();torch.cuda.synchronize();ts.append(time.perf_counter()-t)
    ts.sort();return dict(median=ts[len(ts)//2],min=ts[0],max=ts[-1],n=n)

def measure(m,g,lights,n):
    out=dict(parameters=RK.parameter_counts(m) if hasattr(m,'M') else dict(learned_parameters_excl_fixed_AO=sum(p.numel() for p in m.parameters())-m.in_logits.numel()-m.out_logits.numel(),
        fixed_AO_logits=m.in_logits.numel()+m.out_logits.numel(),total=sum(p.numel() for p in m.parameters())))
    with torch.no_grad():
        co=m.coefficients(g);out['state_floats']={k:int(v.numel()) for k,v in co.items()};out['state_floats_WUV']=sum(out['state_floats'][k] for k in ('W','U','V'))
        out['state_bytes_WUV']=out['state_floats_WUV']*4;del co;torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();base=torch.cuda.memory_allocated()
        out['rebuild_seconds']=med(lambda:m.coefficients(g),n);co=m.coefficients(g)
        out['apply64_seconds']=med(lambda:[m.apply(g,lights[i:i+4],co) for i in range(0,64,4)],max(3,n//2))
        out['apply1_seconds']=med(lambda:m.apply(g,lights[:1],co),n)
        out['peak_GPU_above_geometry']=torch.cuda.max_memory_allocated()-base;del co
    torch.cuda.empty_cache();return out

def main():
    RG=environment();torch.backends.cuda.matmul.allow_tf32=False;torch.set_grad_enabled(False);out=ROOT/'cost/cost.json'
    if out.exists():raise FileExistsError('immutable')
    apps=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
    g=RG.device(load(INP68/'wl59/geometry_rest.pt'));lights=load(INP68/'wl59/lights.pt')['test'].cuda();n=3 if dev() else 10;res={}
    from runtime import Transport
    m=Transport().cuda();m.load_state_dict(load(R73/'runs/AR_5901.pt')['model']);m.eval();res['rank8_AR_5901_historical']=measure(m,g,lights,n);del m
    m=RK.TransportM(8).cuda();m.load_state_dict(load(R73/'runs/AR_5901.pt')['model']);m.eval();res['rank8_AR_5901_generic']=measure(m,g,lights,n);del m
    m=RK.load_run(ROOT/'runs/A75_5901.pt');res['rank16_A75_5901']=measure(m,g,lights,n);del m;torch.cuda.empty_cache()
    t=time.perf_counter();_=RG.geometry(CFG['stage3']['pose'],ballistic=True);geo=time.perf_counter()-t
    r8=res['rank8_AR_5901_generic'];r16=res['rank16_A75_5901']
    ratio={k:r16[k]['median']/r8[k]['median'] for k in ('rebuild_seconds','apply64_seconds','apply1_seconds')}
    ratio.update(state_floats=r16['state_floats_WUV']/r8['state_floats_WUV'],learned_parameters=r16['parameters']['learned_parameters_excl_fixed_AO']/r8['parameters']['learned_parameters_excl_fixed_AO'],
                 angular_parameters=r16['parameters']['fixed_AO_logits']/r8['parameters']['fixed_AO_logits'],peak_GPU=r16['peak_GPU_above_geometry']/max(r8['peak_GPU_above_geometry'],1))
    tr={n_:{k:load(ROOT/'runs'/f'{n_}.pt').get(k) for k in ('seconds','peak_GPU_allocated','concurrent','trainable')} for n_ in ('A75_5901','A75_5902')}
    tr.update({n_:{k:load(R73/'runs'/f'{n_}.pt').get(k) for k in ('seconds','peak_GPU_allocated','concurrent','trainable')} for n_ in ('AR_5901','AR_5902')})
    refs=json.loads((ROOT/'refs/complete.json').read_text()) if (ROOT/'refs/complete.json').exists() else None
    b=load(R74/'bases'/f"{CFG['basis']['selected']}.pt",mmap=True)
    cd=CFG['stage5'];dom=any(ratio[k]>cd['dominates_ratio'] for k in cd['dominates_keys'])
    write_new(out,json.loads(json.dumps(dict(models=res,ratio_rank16_over_rank8=ratio,cost_dominates_rule=cd,cost_ratio_exceeds=dom,geometry_preparation_seconds_rank_independent=geo,
        training=tr,references=dict(stage3_packets=refs['packets'] if refs else None,stage3_seconds_sum=refs['seconds_sum'] if refs else None),
        angular_pretraining=dict(basis=CFG['basis']['selected'],seconds=float(b['seconds']),note='WL74 G2 fit (shared logits + per-source T, float64); rank-8 G0 basis needs none'),
        gpu=torch.cuda.get_device_name(0),other_gpu_processes_at_start=apps,timing_repeats=n),default=float)))
    print('COST_DONE',json.dumps(ratio),flush=True)

if __name__=='__main__':main();sys.stdout.flush();os._exit(0)
