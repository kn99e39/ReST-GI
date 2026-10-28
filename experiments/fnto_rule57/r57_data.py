"""WL57 offline supervision and data/physics gate. Candidate runtime never imports target-loading functions."""
import sys,os,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
import r57_geometry as RG
import k55_ops as KO
import p53_geom as G

ROOT=Path(os.environ.get('WL57',Path.home()/'rest_wl57'));OUT=ROOT/'out';DATA=ROOT/'data';OPS=ROOT/'ops'

def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,(np.floating,float)):return float(x) if np.isfinite(x) else None
    if isinstance(x,np.integer):return int(x)
    if isinstance(x,np.bool_):return bool(x)
    return x
def dump(x,p):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(clean(x),indent=1,allow_nan=False)+'\n')
def log(*a):print(time.strftime('%H:%M:%S'),*a,flush=True)
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(16*1024*1024),b''):h.update(b)
    return h.hexdigest()

def aggregate(y,parent,coarse=True):
    """Flux sums, no averaging: 64->16 angular bins and optionally 2.5mm->1cm material cells."""
    bm=torch.as_tensor(G.bin_parent(8,4),device=y.device)
    z=torch.zeros(y.shape[0],y.shape[1],16,3,device=y.device,dtype=y.dtype).index_add_(2,bm,y)
    if not coarse:return z
    return torch.zeros(y.shape[0],int(parent.max())+1,16,3,device=y.device,dtype=y.dtype).index_add_(1,parent,z)

def light_fields(geometry,n,seed,split):
    rng=np.random.default_rng(seed);xyz=geometry['rest_xyz'].cpu().numpy();P=len(xyz);face=geometry['coarse']['port_face'][geometry['parent'].cpu().numpy()]
    ar=geometry['area'].cpu().numpy();dirs=G.bin_sample(np.arange(16),4,np.full(16,.5),np.full(16,.5))
    widths=[.0025,.0075,.02] if split=='test' else [.005,.0125,.03]
    kappas=[0,4,16,32] if split=='test' else [0,2,8]
    fields=[];meta=[]
    for i in range(n):
        typ=i%4;f=int(rng.integers(6));axis=f//2;other=[k for k in range(3) if k!=axis]
        xy=xyz[:,other];mu=rng.uniform(-.04,.04,2);width=widths[int(rng.integers(len(widths)))];sp=np.exp(-((xy-mu)**2).sum(1)/(2*width**2))*(face==f)
        if typ==1:
            sp=np.exp(rng.uniform(-1,1)*np.sin(xyz[:,0]*rng.uniform(30,200))+rng.uniform(-1,1)*np.cos(xyz[:,1]*rng.uniform(30,200)))
        elif typ==2:
            sp=np.exp(-(xy[:,0]+.05)/rng.choice([.005,.01,.02]))*(face==f)
        elif typ==3:
            # Whole-boundary excitation, including hidden/side cells; checkerboard prevents a single-emitter shortcut.
            sp=.05+(np.sin(xyz[:,0]*rng.uniform(100,500))*np.cos(xyz[:,1]*rng.uniform(100,500))>0).astype(float)
        if i<16:
            # Fixed central illumination population, not adjusted to the current dent or any result.
            f=5 if i%2==0 else 4;mu=np.array([[-.02,-.02],[-.02,.02],[.02,-.02],[.02,.02],[0,0],[0,.02],[.02,0],[0,-.02]][i//2])+rng.uniform(-.0025,.0025,2)
            xy=xyz[:,:2];sp=np.exp(-((xy-mu)**2).sum(1)/(2*width**2))*(face==f);typ=4
        direction=dirs[int(rng.integers(16))];k=kappas[int(rng.integers(len(kappas)))];ang=np.exp(k*(dirs@direction-1))
        z=(sp*ar)[:,None]*ang[None];z/=max(z.sum(),1e-30)
        rgb=np.stack([z*np.exp(rng.uniform(-.5,.5)*xyz[:,c]/.05)[:,None] for c in range(3)],-1)
        rgb/=np.maximum(rgb.sum((0,1),keepdims=True),1e-30)
        fields.append(rgb.astype(np.float32));meta.append(dict(family=typ,face=f,width=width,kappa=k,direction=direction.tolist()))
    # Last 16 fields: equal-spatial/equal-total-flux pairs with different incoming directions.
    for i in range(n-16,n,2):
        base=fields[i].sum(1);aa=int(rng.integers(16));bb=(aa+5)%16
        a=np.zeros_like(fields[i]);b=a.copy();a[:,aa]=base;b[:,bb]=base;fields[i]=a;fields[i+1]=b
        meta[i].update(pair=i+1,equal_irradiance=True);meta[i+1]=dict(meta[i],pair=i)
    return torch.tensor(np.stack(fields)),meta

def prepare_inputs():
    for d in ('geometry','targets'): (DATA/d).mkdir(parents=True,exist_ok=True)
    for name in RG.POSES:
        p=DATA/'geometry'/f'{name}.pt'
        if p.exists():continue
        g=RG.geometry(name);torch.save(RG.serial(g),p);dump(g['meta'],OUT/'geometry'/f'{name}.json');log('GEOMETRY',name,g['meta']['total_seconds'])
        if name=='rest':
            lights={};metas={}
            for split,n,seed in [('train',96,571001),('validation',32,571002),('test',64,571003)]:lights[split],metas[split]=light_fields(g,n,seed,split)
            torch.save(lights,DATA/'lights.pt');dump(metas,OUT/'light_manifest.json')
        del g;torch.cuda.empty_cache()

def oracle_path(name,rep):return OPS/f'{name}_r{rep}.pt'
def raw_oracle(name,rep):
    path=oracle_path(name,rep)
    if path.exists():return torch.load(path)
    OPS.mkdir(exist_ok=True);ix=list(RG.POSES).index(name);t0=time.perf_counter()
    src=None
    if name=='rest':src=Path.home()/'rest_wl55/ops'/f'slab_L4_s{rep+1}.pt'
    if name=='te0':src=Path.home()/'rest_wl55/ops'/f'dent_L4_s{rep+1}.pt'
    if src:
        a=torch.load(src);r={k:v for k,v in a.items() if k not in ('KC','S2','cams','cam_keys')};del a
    else:
        m,V=RG.mesh(name);fm,VC,parent=KO.refined(m,V,4);dom=KO.DomainX(fm,VC,name=name)
        r=KO.tr_sparse(dom,64,16,2048,57700+10*ix+rep,None,parent,len(m['port_area']),log=log,s2=False)
        del dom
    torch.save(r,path);dump(dict(pose=name,rep=rep,seed=r['seed'],source=str(src) if src else 'fresh isolated TR',source_sha256=sha(src) if src else None,
        bytes=path.stat().st_size,seconds=r['seconds'],stage_wall=time.perf_counter()-t0,stats=r['stats'],sha256=sha(path)),OUT/'oracle'/f'{name}_r{rep}.json')
    return r

def actions(name,rep=0):
    p=DATA/'targets'/f'{name}_r{rep}.pt'
    if p.exists():return
    lights=torch.load(DATA/'lights.pt');r=raw_oracle(name,rep);K=KO.KOp(r['idx'],r['val'],r['shape']);out={};t0=time.perf_counter()
    for split,x in lights.items():
        arr=[]
        for i in range(0,len(x),8):
            xx=x[i:i+8].to('cuda');q=[]
            for c in range(3):q.append((K[c].csr@xx[:,:,:,c].reshape(len(xx),-1).T).T.reshape(len(xx),-1,64).cpu())
            arr.append(torch.stack(q,-1))
        out[split]=torch.cat(arr)
    col=K.colsum();st=r['stats'];phy=dict(passed=bool(min(float(v.csr.values().min()) for v in K.ch)>=0 and float(col.max())<=1+1e-6 and (st['leaks']+st['wrong_side'])/st['photons']<=1e-5 and st['truncated']==0),max_colsum=col.max(1).values.cpu().tolist(),leak_rate=(st['leaks']+st['wrong_side'])/st['photons'])
    torch.save(out,p);dump(dict(pose=name,rep=rep,seconds=time.perf_counter()-t0,sha256=sha(p),splits={k:list(v.shape) for k,v in out.items()},physics=phy),OUT/'target_manifest'/f'{name}_r{rep}.json')
    log('ACTIONS',name,rep,time.perf_counter()-t0)

def build_targets():
    prepare_inputs()
    for name in RG.POSES:actions(name,0)
    for name in ['rest']+RG.TEST:actions(name,1)

def data_gate():
    g=RG.device(torch.load(DATA/'geometry/rest.pt'));records=[];physics=[]
    rest=[torch.load(DATA/'targets'/f'rest_r{r}.pt')['test'] for r in (0,1)]
    rr=[aggregate(x.to('cuda'),g['parent']).cpu() for x in rest]
    for name in RG.TEST:
        cur=[torch.load(DATA/'targets'/f'{name}_r{r}.pt')['test'] for r in (0,1)]
        cur=[aggregate(x.to('cuda'),g['parent']).cpu() for x in cur]
        delta=.5*(cur[0]+cur[1]-rr[0]-rr[1]);noise=(cur[0]-cur[1])/(2**.5);rnoise=(rr[0]-rr[1])/(2**.5)
        snr=delta.flatten(1,2).norm(dim=1)/torch.hypot(noise.flatten(1,2).norm(dim=1),rnoise.flatten(1,2).norm(dim=1)).clamp(min=1e-30)
        # Concentrated families 0 and2, plus directional pairs whose base is spatially concentrated.
        ids=list(range(16))
        frac=(snr[ids]>=3).float().mean(0)
        records.append(dict(pose=name,median_snr=snr.median(0).values.tolist(),concentrated_fraction_ge3=frac.tolist(),
            meaningful_channels=(frac>=.25).tolist(),target_noise_rel=((cur[0]-cur[1]).flatten(1,2).norm(dim=1)/(.5*(cur[0]+cur[1])).flatten(1,2).norm(dim=1).clamp(min=1e-30)).mean(0).tolist()))
    for name in RG.POSES:
        phy=json.load(open(OUT/'target_manifest'/f'{name}_r0.json'))['physics'];physics.append(dict(pose=name,**phy))
    proceed=all(r['passed'] for r in physics) and sum(sum(r['meaningful_channels'])>=2 for r in records)>=3
    dump(dict(proceed=proceed,signal=records,physics=physics),OUT/'gate1.json');log('GATE1',proceed,records)
    assert proceed,'Stop: target physics or edit signal is not adequate; no training'

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True);DATA.mkdir(parents=True,exist_ok=True)
    {'build':build_targets,'gate':data_gate}[sys.argv[1]]();sys.stdout.flush();os._exit(0)
