"""WL80 Stage 6 computational cost (same GPU, run alone at the end of the chain): Q78_5901 (A0B0) vs the 5901 run of every trained arm (same
runtime architecture; only the frozen basis values differ), A75_5901 for reference: parameters, geometry-dependent state, geometry rebuild
(shared preparation + inputs + coefficients), coefficients, 1- / 64-light apply, peak GPU; training time / memory (run records); angular
pretraining (Stage 1 basis restarts, records) and the symmetry presentation (SymRest build, per-step rebuild).  python cost80.py final"""
import sys,os,json,time,subprocess
import torch
from common80 import CFG,ROOT,R75,R78,INP68,write_new,environment,dev,load
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
    runs=[r for r in CFG['runs'] if r.endswith('_5901') and (ROOT/'runs'/f'{r}.pt').exists()]
    for r in runs:m=Q.load_run(RG,ROOT/'runs'/f'{r}.pt');res[r]=C78.measure(m,g,lights,n,True);del m;torch.cuda.empty_cache()
    prep=C78.med(lambda:RG.geometry('rest',ballistic=False),max(3,n//2),warm=1);pm=prep['median'];a=res['Q78_5901'];ratio={}
    for r in runs:
        b=res[r];ratio[r]=dict(apply64=b['apply64_seconds']['median']/a['apply64_seconds']['median'],apply1=b['apply1_seconds']['median']/a['apply1_seconds']['median'],
            rebuild_incl_shared_preparation=(pm+b['feature_build_seconds']['median']+b['rebuild_seconds']['median'])/(pm+a['feature_build_seconds']['median']+a['rebuild_seconds']['median']),
            coefficients=b['rebuild_seconds']['median']/a['rebuild_seconds']['median'],state_bytes=b['state_bytes_WUV']/a['state_bytes_WUV'],
            peak_GPU=b['peak_GPU_above_geometry']/max(a['peak_GPU_above_geometry'],1),parameters=b['parameters']['total']/a['parameters']['total'])
    import fit73,fit79
    pb=fit73.Problem73(RG,'S16R');torch.cuda.synchronize();t=time.perf_counter();sr=fit79.SymRest(RG,pb.tg['rest'],pb.ports,pb.gs['rest']);torch.cuda.synchronize();t_sym=time.perf_counter()-t
    step=C78.med(lambda:sr.set(7),n)
    basis={f.stem:dict(seconds=load(f,mmap=True)['seconds'],peak_GPU=load(f,mmap=True)['peak_GPU']) for f in sorted((ROOT/'basis').glob('B80_r*.pt'))}
    tr={n_:{k:load(ROOT/'runs'/f'{n_}.pt').get(k) for k in ('seconds','peak_GPU_allocated','concurrent','trainable')} for n_ in CFG['runs'] if (ROOT/'runs'/f'{n_}.pt').exists()}
    q78={n_:{k:load(R78/'runs'/f'{n_}.pt').get(k) for k in ('seconds','peak_GPU_allocated','concurrent')} for n_ in ('Q78_5901','Q78_5902')}
    dev_=max([abs(v[k]-1) for v in ratio.values() for k in ('apply64','apply1','state_bytes','parameters')] or [0])
    write_new(out,json.loads(json.dumps(dict(models=res,ratio=ratio,shared_geometry_preparation=prep,angular_pretraining=basis,symmetry_presentation=dict(symrest_build_seconds=t_sym,
        rebuild_per_step=step),training=tr,Q78_training=q78,significant_runtime_deviation=dev_>CFG['stage6']['cost_deviation'],gpu=torch.cuda.get_device_name(0),
        other_gpu_processes_at_start=apps,timing_repeats=n),default=float)))
    print('COST_DONE',json.dumps(ratio),flush=True)
if __name__=='__main__':main(*sys.argv[1:]);sys.stdout.flush();os._exit(0)
