"""Resumable evaluation of immutable released scenes/checkpoints, one method/process."""
import argparse, hashlib, importlib, json, os, platform, subprocess, sys, time
from pathlib import Path

def atomic_json(path, data):
    path=Path(path); temp=path.with_suffix('.tmp'); temp.write_text(json.dumps(data,indent=2)+'\n'); temp.replace(path)

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(); p.add_argument('--upstream',required=True); p.add_argument('--out',required=True)
    p.add_argument('--asset',required=True); p.add_argument('--method',choices=['neural','pt'],required=True)
    p.add_argument('--spp',type=int,required=True); p.add_argument('--rep',type=int,default=0)
    p.add_argument('--res',type=int,default=800); p.add_argument('--chunk',type=int,default=4)
    p.add_argument('--vcall',type=int,default=0)
    p.add_argument('--workers',default='1',help="Process count, or 'auto' to use CPU/VRAM measurements")
    p.add_argument('--tile-size',type=int,default=0)
    p.add_argument('--task-timeout',type=float,default=120)
    p.add_argument('--cpu-threads',type=int,default=2)
    a=p.parse_args()
    a.auto_workers = a.workers == 'auto'
    try:
        a.workers = len(os.sched_getaffinity(0)) if a.auto_workers else int(a.workers)
    except ValueError:
        p.error("workers must be a positive integer or 'auto'")
    if a.workers < 1 or a.tile_size < 0 or a.task_timeout <= 0 or a.cpu_threads < 1:
        p.error('workers/timeout/cpu-threads must be positive and tile-size nonnegative')
    if a.auto_workers or a.workers > 1 or a.tile_size > 0:
        from parallel_render import run
        return run(a)
    start=time.perf_counter(); root=Path(a.upstream).resolve(); out=Path(a.out).resolve()
    out.mkdir(parents=True,exist_ok=True); os.chdir(root/'demo'); sys.path.insert(0,str(root))
    import numpy as np
    import torch
    import mitsuba as mi
    mi.set_variant('cuda_ad_rgb')
    import drjit as dr
    import models.integrator
    dr.set_flag(dr.JitFlag.LoopRecord,False); dr.set_flag(dr.JitFlag.VCallRecord,bool(a.vcall))
    key=f'{a.asset}_{a.method}_r{a.rep}_{a.res}'
    npz=out/(key+'.npz'); meta_path=out/(key+'.json')
    prior=json.loads(meta_path.read_text()) if meta_path.exists() else None
    if prior and prior['completed_spp']>=a.spp:
        print('already complete',key,prior['completed_spp'],flush=True); return
    scene=mi.load_dict(importlib.import_module('scenes.'+a.asset).scene1(a.res))
    integrator=scene.integrator() if a.method=='pt' else mi.load_dict({'type':'neuralpath','max_depth':10,'rr_depth':5,'device':0,'model_type':'8dna','asset_instance0':str(root/'checkpoints'/f'{a.asset}.ckpt')})
    load_s=time.perf_counter()-start
    meta=prior or dict(key=key,arguments=vars(a),completed_spp=0,chunks=[],created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),environment=dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,mitsuba=mi.__version__,drjit=dr.__version__,gpu=torch.cuda.get_device_name(),platform=platform.platform()),checkpoint_sha256=sha(root/'checkpoints'/f'{a.asset}.ckpt'),scene_source_sha256=sha(root/'scenes'/a.asset/'__init__.py'),upstream_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),jit_flags=dict(LoopRecord=False,VCallRecord=bool(a.vcall)))
    if prior:
        if prior['arguments']['chunk']!=a.chunk or prior['jit_flags']['VCallRecord']!=bool(a.vcall): raise ValueError('Resume configuration changed')
        with np.load(npz) as z: total=z['sum'].copy(); first128=z['first128'].copy()
    else: total=np.zeros((a.res,a.res,3),dtype=np.float64); first128=np.zeros_like(total)
    print(json.dumps({'key':key,'load_s':load_s,'resume_spp':meta['completed_spp'],'target':a.spp}),flush=True)
    warm=time.perf_counter(); warm_img=mi.render(scene,integrator=integrator,spp=a.chunk,seed=900000+a.rep*1000); dr.eval(warm_img); torch.cuda.synchronize()
    meta.setdefault('invocations',[]).append(dict(load_s=load_s,warmup_s=time.perf_counter()-warm,target_spp=a.spp,gpu_start=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip()))
    del warm_img; dr.sync_thread()
    seedbase=(210000 if a.method=='neural' else 410000)+100000*a.rep
    checkpoints={128,512,2048,4096,8192,16384,32768,a.spp}
    t0=time.perf_counter()
    for offset in range(meta['completed_spp'],a.spp,a.chunk):
        count=min(a.chunk,a.spp-offset); ts=time.perf_counter()
        image=mi.render(scene,integrator=integrator,spp=count,seed=seedbase+offset//a.chunk)
        dr.eval(image); torch.cuda.synchronize(); elapsed=time.perf_counter()-ts
        arr=np.array(image,dtype=np.float64)
        if not np.isfinite(arr).all(): raise ValueError('nonfinite render output')
        total+=arr*count
        if offset<128: first128+=arr*count
        meta['chunks'].append(dict(seed=seedbase+offset//a.chunk,spp=count,wall_s=elapsed))
        meta['completed_spp']=offset+count
        del image,arr
        if meta['completed_spp'] in checkpoints:
            temp=npz.with_suffix('.tmp.npz'); np.savez_compressed(temp,sum=total,first128=first128); temp.replace(npz)
            avg=(total/meta['completed_spp']).astype(np.float32)
            exr=out/(key+f'_s{meta["completed_spp"]}.exr'); mi.Bitmap(avg).write(str(exr))
            meta['torch_peak_bytes']=torch.cuda.max_memory_allocated(); meta['last_checkpoint_sha256']=sha(exr)
            meta['last_checkpoint_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
            meta['timed_wall_s']=sum(c['wall_s'] for c in meta['chunks'])
            atomic_json(meta_path,meta)
            print(json.dumps({'key':key,'spp':meta['completed_spp'],'render_wall_s':meta['timed_wall_s'],'invocation_s':time.perf_counter()-t0,'mean':float(avg.mean())}),flush=True)
    meta['gpu_end']=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip()
    atomic_json(meta_path,meta)

if __name__=='__main__': main()
