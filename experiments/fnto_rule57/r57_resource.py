"""Bounded resource smoke only: untrained forward/backward, no reference generation or optimizer steps."""
import sys,os,json,subprocess,resource,time
from pathlib import Path
import torch
import r57_geometry as RG
from r57_data import ROOT,OUT,dump
from r57_model import Transport

PF=OUT/'preflight_v2'
def gradient(arm,path):
    torch.set_num_threads(8);torch.manual_seed(5701);torch.backends.cuda.matmul.allow_tf32=False
    torch.cuda.reset_peak_memory_stats();a=RG.geometry('rest',q=32);b=RG.geometry('te0',q=32);m=Transport(arm).cuda()
    x=torch.rand(2,a['P'],16,3,device='cuda');x/=x.sum((1,2),keepdim=True)
    y=m(a,x);z=m(b,x);loss=y.square().mean()+z.square().mean()+(y-z).square().mean();loss.backward();torch.cuda.synchronize()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
    dump(dict(arm=arm,paired_geometries=2,batch=2,oracle_calls=0,optimizer_steps=0,finite=True,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),CPU_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),path)

def main():
    import shutil
    PF.mkdir(parents=True,exist_ok=True);workers={}
    for arm in ('D','R'):
        for policy in ('compact','full'):
            path=PF/f'memory_{arm}_{policy}.json'
            subprocess.run([sys.executable,str(RG.HERE/'r57_cost.py'),'worker','rest',arm,'5701',policy,str(path),'untrained'],check=True)
            workers[arm+'_'+policy]=json.load(open(path))
        path=PF/f'gradient_{arm}.json';subprocess.run([sys.executable,__file__,'gradient',arm,str(path)],check=True);workers[arm+'_gradient']=json.load(open(path))
    P=3840;poses=len(RG.POSES)+1+len(RG.TEST);photons=P*16*2048
    # At most one exit per photon; uncoalesced idx+3 weights=20B, conservatively one record/photon.
    raw=poses*photons*20;targets=poses*(96+32+64)*P*64*3*4;geometry=len(RG.POSES)*max(w['memory']['geometry_bytes'] for k,w in workers.items() if 'memory' in w)
    disk_required=int(1.25*(raw+targets+geometry)+2*1024**3);disk=shutil.disk_usage(ROOT)
    mem={}
    for line in Path('/proc/meminfo').read_text().splitlines():
        k,v=line.split(':',1);mem[k]=int(v.strip().split()[0])*1024
    cpu_required=max(24*1024**3,int((len(RG.TRAIN)+len(RG.VAL))*(96+32+64)*P*64*3*4*1.5+geometry+4*1024**3))
    free,total=torch.cuda.mem_get_info();gpu_required=max(w['peak_reserved_bytes'] for k,w in workers.items() if k.endswith('gradient'))+int(1.2*geometry)+512*1024**2
    feasible=disk.free>=disk_required and mem['MemAvailable']>=cpu_required and free>=gpu_required and all(w.get('eligible',True) for w in workers.values())
    dump(dict(feasible=feasible,scratch=str(ROOT.resolve()),disk_free_bytes=disk.free,disk_required_conservative_bytes=disk_required,CPU_available_bytes=mem['MemAvailable'],CPU_required_conservative_bytes=cpu_required,GPU_free_bytes=free,GPU_total_bytes=total,GPU_required_conservative_bytes=gpu_required,
        raw_operator_bound_bytes=raw,target_bytes=targets,geometry_bytes=geometry,workers=workers,limitations='Untrained paired-gradient smoke plus conservative resident-geometry allowance; not a measured full training peak. Full per-model/policy cost peaks are isolated in fresh processes. No new TR/Cycles target generation.'),PF/'resources.json')
    assert feasible,'Resource preflight NO-GO'
if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='gradient':gradient(sys.argv[2],sys.argv[3])
    else:main()
    sys.stdout.flush();os._exit(0)
