"""WL79 Stage 6 computational cost (same GPU, run alone at the end of the chain): Q78_5901 (primary control) vs S79_5901 (same runtime
architecture), A75_5901 for reference. Parameters; geometry-dependent cached state; geometry rebuild (shared geometry preparation + model-specific
inputs + coefficient generation, WL78 amendment-1 definition); coefficient generation; 1-light and 64-light apply; peak GPU memory; training time
and peak memory from the run records; the additional augmentation cost (SymRest construction and the per-step presentation rebuild).
    python cost79.py final"""
import sys,os,json,time,subprocess
import torch
from common79 import CFG,ROOT,R75,R78,INP68,write_new,environment,dev,load
import common68 as C68
import common78
common78.environment=lambda:C68.environment(ROOT/'repo')        # WL78 modules use this repository copy
RG=environment()
import rank75 as RK,q78 as Q,fit71
import cost78 as C78

def main(kind='final'):
    torch.backends.cuda.matmul.allow_tf32=False;torch.set_grad_enabled(False);out=ROOT/'cost'/f'{kind}.json'
    if out.exists():raise FileExistsError('immutable')
    apps=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
    g=RG.device(fit71.guarded('geometry_rest'));lights=load(INP68/'wl59/lights.pt')['test'].cuda();n=3 if dev() else 10;res={}
    m=RK.load_run(R75/'runs/A75_5901.pt');res['A75_5901']=C78.measure(m,g,lights,n,False);del m
    m=Q.load_run(RG,R78/'runs/Q78_5901.pt');res['Q78_5901']=C78.measure(m,g,lights,n,True);del m
    m=Q.load_run(RG,ROOT/'runs/S79_5901.pt');res['S79_5901']=C78.measure(m,g,lights,n,True);del m;torch.cuda.empty_cache()
    prep=C78.med(lambda:RG.geometry('rest',ballistic=False),max(3,n//2),warm=1);pm=prep['median'];a,b=res['Q78_5901'],res['S79_5901']
    ratio=dict(apply64=b['apply64_seconds']['median']/a['apply64_seconds']['median'],apply1=b['apply1_seconds']['median']/a['apply1_seconds']['median'],
               rebuild_incl_shared_preparation=(pm+b['feature_build_seconds']['median']+b['rebuild_seconds']['median'])/(pm+a['feature_build_seconds']['median']+a['rebuild_seconds']['median']),
               coefficients=b['rebuild_seconds']['median']/a['rebuild_seconds']['median'],state_bytes=b['state_bytes_WUV']/a['state_bytes_WUV'],
               peak_GPU=b['peak_GPU_above_geometry']/max(a['peak_GPU_above_geometry'],1),parameters=b['parameters']['total']/a['parameters']['total'])
    # augmentation cost (training only): SymRest construction and the per-step presentation (fit79), on the real S16-R targets
    import fit73,fit79
    torch.cuda.synchronize();t=time.perf_counter();pb=fit73.Problem73(RG,'S16R');torch.cuda.synchronize();t_problem=time.perf_counter()-t
    t=time.perf_counter();sr=fit79.SymRest(RG,pb.tg['rest'],pb.ports,pb.gs['rest']);torch.cuda.synchronize();t_sym=time.perf_counter()-t
    step=C78.med(lambda:sr.set(7),n)
    tr={n_:{k:load(ROOT/'runs'/f'{n_}.pt').get(k) for k in ('seconds','peak_GPU_allocated','concurrent','trainable')} for n_ in CFG['runs']}
    q78={n_:{k:load(R78/'runs'/f'{n_}.pt').get(k) for k in ('seconds','peak_GPU_allocated','concurrent')} for n_ in ('Q78_5901','Q78_5902')}
    dev_=max(abs(ratio[k]-1) for k in ('apply64','apply1','state_bytes','parameters'))
    write_new(out,json.loads(json.dumps(dict(models=res,ratio=ratio,shared_geometry_preparation=prep,augmentation=dict(problem_build_seconds=t_problem,symrest_build_seconds=t_sym,
        presentation_rebuild_per_step=step),training=tr,Q78_training=q78,significant_runtime_deviation=dev_>CFG['stage6']['cost_deviation'],gpu=torch.cuda.get_device_name(0),
        other_gpu_processes_at_start=apps,timing_repeats=n),default=float)))
    print('COST_DONE',json.dumps(ratio),flush=True)

if __name__=='__main__':main(*sys.argv[1:]);sys.stdout.flush();os._exit(0)
