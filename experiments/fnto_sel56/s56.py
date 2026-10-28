"""WL56 isolated exact domain-selective oracle pipeline. See protocol/sel56.json.

run.sh {prepare,tests,reference_gate,update,replay,audit} ...
Complete TR builds only occur for update F/S, never hidden in replay.
"""
import sys, os, json, time, hashlib, resource, subprocess
from pathlib import Path
import numpy as np
import torch
import s56_fixtures as SF
import k55_common as CM
import k55_ops as KO
import k55_compose as KP
import c54_casp as CA
import p53_compose as CP
import p53_geom as G
import p53_fixtures as FX

ROOT = Path(os.environ.get('WL56', Path.home() / 'rest_wl56'))
OUT = ROOT / 'out'
CACHE = ROOT / 'cache'
OPS = ROOT / 'ops'
HIST = Path.home() / 'rest_wl55'

def dump(x,p):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True); CM.dump(x,p)

def log(*x): print(time.strftime('%H:%M:%S'),*x,flush=True)

def timed(fn):
    torch.cuda.synchronize(); a=time.perf_counter(); r=fn(); torch.cuda.synchronize()
    return r,time.perf_counter()-a

def digest(r):
    h=hashlib.sha256()
    for k in ('idx','val','KC'):
        a=r[k].detach().cpu().contiguous().numpy(); h.update(k.encode()); h.update(memoryview(a).cast('B'))
    h.update(str(r['shape']).encode()); h.update(str(r['cam_keys']).encode())
    return h.hexdigest()

def hash_file(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(16*1024*1024),b''): h.update(b)
    return h.hexdigest()

def rest_path(seed): return CACHE / f'rest_{seed}.pt'

def raw_path(state,arm,rep,ci): return OPS / f'{state}_{arm}_r{rep}_d{ci}.pt'

def rest_seed(rep,ci): return (5561,5562)[rep if ci==0 else 1-rep]

def new_seed(state,rep,ci): return (5565,5566)[rep] if state=='dent' and ci==1 else rest_seed(rep,ci)

def build(state,ci,seed,path,nper=2048):
    if path.exists(): raise FileExistsError(f'Preserve existing oracle: {path}; use replay, never overwrite a run')
    m,V=SF.intrinsic(state,ci); fm,VC,par=KO.refined(m,V,4)
    dom=KO.DomainX(fm,VC,name=f'{state}/d{ci}/{seed}')
    r=KO.tr_sparse(dom,64,16,nper,seed,np.array([FX.DIRS[k] for k in SF.KEYS]),par,len(m['port_area']),log=log,s2=False)
    r.update(cam_keys=list(SF.KEYS),level=4,parent=par,cell_area=dom.port_area.double().cpu().numpy(),signature=SF.signature(state,ci))
    r['content_hash']=digest(r)
    path.parent.mkdir(parents=True,exist_ok=True)
    _,write=timed(lambda:torch.save(r,path)); meta=dict(path=str(path),seed=seed,content_hash=r['content_hash'],signature=r['signature'],
        bytes=path.stat().st_size,TR_seconds=r['seconds'],write_seconds=write,nnz=len(r['idx']),stats=r['stats'])
    dump(meta,path.with_suffix('.json')); del dom
    return r,meta

def prepare():
    CACHE.mkdir(parents=True,exist_ok=True); manifest={}
    for j,seed in enumerate((5561,5562),1):
        p=HIST/'ops'/f'slab_L4_s{j}.pt'; r,load=timed(lambda:torch.load(p))
        r['KC']=r['KC'][[r['cam_keys'].index(k) for k in SF.KEYS]].clone(); r['cam_keys']=list(SF.KEYS)
        r['signature']=SF.signature('rest',0); r['content_hash']=digest(r)
        q=rest_path(seed); _,write=timed(lambda:torch.save(r,q))
        manifest[str(seed)]=dict(source=str(p),source_sha256=hash_file(p),path=str(q),content_hash=r['content_hash'],
             bytes=q.stat().st_size,load_seconds=load,write_seconds=write,original_TR_seconds=r['seconds'],original_cam_keys=5,new_cam_keys=list(SF.KEYS))
        del r; log('prepared',seed)
    dump(manifest,OUT/'initial_cache.json')

def load_packages(state,arm,rep,do_build=False,independent=False):
    started=time.perf_counter()
    data={}; meta={}; total=0
    for ci in (0,1):
        needs=arm=='F' or (arm=='S' and ci in SF.dirty(state))
        p=raw_path(state,arm,rep,ci) if needs else rest_path(rest_seed(rep,ci))
        if independent and ci==0: p=rest_path(rest_seed(1-rep,0)); needs=False
        if do_build and needs:
            a=time.perf_counter(); r,md=build(state,ci,new_seed(state,rep,ci),p); torch.cuda.synchronize(); dt=time.perf_counter()-a
            md['complete_build_wall']=dt; md['recomputed']=True
        else:
            r,dt=timed(lambda:torch.load(p)); md=dict(path=str(p),load_seconds=dt,recomputed=False,bytes=p.stat().st_size)
        expected=SF.signature('rest' if arm=='T' else state,ci)
        assert r.get('signature',expected)==expected,'cache signature mismatch'
        # Hash only once per load; the persisted digest is also checked against actual memory.
        a=time.perf_counter(); hh=digest(r); md['identity_check_seconds']=time.perf_counter()-a
        assert hh==r['content_hash']; md.update(content_hash=hh,signature=expected,seed=int(r['seed']))
        total+=dt+md['identity_check_seconds']; data[ci]=r; meta[str(ci)]=md
    return data,meta,time.perf_counter()-started

def compose(state,raw,rep,arm,external_N=64_000_000,prepared_K=None):
    """Rebuild every current boundary/C/input; solve all domains. Timings disjoint, CUDA fenced."""
    begun=time.perf_counter()
    sd=SF.scene(state); timings={}; levels=[np.full(len(m['port_area']),4 if k=='medium' else 1) for n,k,m,V in sd['comps']]
    (ss,parent,_),timings['boundary']=timed(lambda:KP.build_ss(sd,levels))
    Kops={}; cs={}; stats={}
    def make_ops():
        for ci,r in raw.items():
            Kops[ci]=prepared_K[ci] if prepared_K is not None else KO.KOp(r['idx'],r['val'],r['shape'])
    _,timings['CSR_and_transfer']=timed(make_ops)
    diag_start=time.perf_counter()
    for ci,r in raw.items():
        col=Kops[ci].colsum(); cs[str(ci)]=col.max(1).values.tolist()
        stats[str(ci)]=dict(leak_rate=(r['stats']['leaks']+r['stats']['wrong_side'])/r['stats']['photons'],truncated=r['stats']['truncated'])
    torch.cuda.synchronize();diag1=time.perf_counter()-diag_start
    cnt=np.bincount(parent); mp=np.maximum(1,np.round(128/cnt[parent])).astype(int)
    for ci,c in enumerate(ss.comps):
        if c.kind=='black': mp[ss.comp_ports(ci)]=1
    (C,esc,cst),timings['C']=timed(lambda:CA.cast_adaptive(ss,mp,5681))
    def field():
        gen=torch.Generator(device='cuda'); gen.manual_seed(5661+rep)
        v=torch.zeros(ss.Ptot*16,device='cuda',dtype=torch.float64)
        for i in range(0,external_N,8_000_000):
            n=min(8_000_000,external_N-i); x,d,w=KP.emitter_rays(ss,sd,n,gen); w=w*n/external_N
            t,p=ss.isect(x,d); ok=torch.isfinite(t); j=ok.nonzero().squeeze(1); j=j[(d[j]*ss.n[p[j]]).sum(1)<0]; pj=p[j]
            b=G.local_to_bin(G.to_local(-d[j],ss.n[pj],ss.t[pj],ss.b[pj]),ss.Nin)
            v.index_add_(0,ss.tri_port[pj]*16+b,w[j].double())
        return v[:,None].expand(-1,3).float()
    ext,timings['illumination']=timed(field)
    phi0,timings['K_direct']=timed(lambda:CP.apply_K(ss,Kops,ext))
    sol,timings['solve']=timed(lambda:CP.solve(ss,Kops,C,phi0))
    ph=ext+sol['phi_in_ind']; Pc=int(parent.max())+1; par=torch.as_tensor(parent,device='cuda'); pred={}
    def readout():
        for ki,key in enumerate(SF.KEYS):
            d=torch.tensor(FX.DIRS[key],device='cuda',dtype=torch.float32)
            Ln=torch.zeros(Pc,3,device='cuda',dtype=torch.float64); Lb=torch.zeros_like(Ln); Lbin=torch.zeros_like(Ln)
            for ci,r in raw.items():
                sl=ss.comp_ports(ci); c0=int(parent[sl.start]); c1=int(parent[sl.stop-1])+1; pin=ph[sl.start*16:sl.stop*16]
                kc=r['KC'][r['cam_keys'].index(key)].to('cuda')
                Ln[c0:c1]=torch.stack([kc[k].double()@pin[:,k].double() for k in range(3)],1)
                KB=KP.ballistic_KB(ss,ci,parent,FX.DIRS[key],seed=5700+ci*13+ki)
                Lb[c0:c1]=KP.kb_apply(KB,pin)
                tris=(ss.tri_comp==ci).nonzero().squeeze(1); cell=ss.tri_port[tris]
                ob=G.local_to_bin(G.to_local(d.expand(len(tris),3),ss.n[tris],ss.t[tris],ss.b[tris]),ss.Nout)
                lt=sol['phi_out'][cell*64+ob].double()/(ss.port_area[cell].double()*np.pi/64)[:,None]
                w=(ss.area[tris]*(ss.n[tris]*d).sum(1).clamp(min=0)).double()
                num=torch.zeros(Pc,3,device='cuda',dtype=torch.float64).index_add_(0,par[cell],lt*w[:,None])
                den=torch.zeros(Pc,device='cuda',dtype=torch.float64).index_add_(0,par[cell],w)
                Lbin[c0:c1]=(num/den.clamp(min=1e-30)[:,None])[c0:c1]
            pred[key+'|nee']=Ln.cpu().numpy(); pred[key+'|ball']=Lb.cpu().numpy(); pred[key+'|bin']=Lbin.cpu().numpy()
    _,timings['readout']=timed(readout)
    diag_start=time.perf_counter()
    closure=torch.sparse.sum(C,dim=0).to_dense().double()+esc.double()
    # Diagnostics timed separately and excluded from dynamic update totals.
    rho,trho=timed(lambda:CA.rho_robust(ss,Kops,C))
    residual=sol['phi_out']-phi0-CP.apply_K(ss,Kops,CP.apply_C(C,sol['phi_out']))
    rel_residual=float(residual.abs().sum()/sol['phi_out'].abs().sum())
    torch.cuda.synchronize();diag2=time.perf_counter()-diag_start
    pred['phi_out']=sol['phi_out'].cpu().numpy(); pred['phi_in_ind']=sol['phi_in_ind'].cpu().numpy()
    shared_elapsed=time.perf_counter()-begun-diag1-diag2
    timings['other_bookkeeping_and_state_transfer']=shared_elapsed-sum(timings.values())
    meta=dict(timings=timings,update_shared_seconds=shared_elapsed,rho=rho,rho_diagnostic_seconds=trho,validation_diagnostic_seconds=diag1+diag2,
       converged=sol['converged'],iterations=sol['iters'],last_rel=sol['last_rel'],fixed_point_rel=rel_residual,
       closure_max=float((closure-1).abs().max()),back_hits=cst['back_hits'],max_colsum=cs,transport_stats=stats,
       cells=ss.Ptot,C_nnz=C._nnz(),C_bytes=C._indices().numel()*8+C._values().numel()*4,
       external_flux=ext.sum(0).cpu().tolist(),field_seed=5661+rep,source_not_target=True,
       gpu_allocated_peak=torch.cuda.max_memory_allocated(),gpu_reserved_peak=torch.cuda.max_memory_reserved(),cpu_rss_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    return pred,meta

def update(state,arm,rep,replay=False,independent=False):
    torch.cuda.reset_peak_memory_stats(); a=time.perf_counter()
    raw,ownership,package_time=load_packages(state,arm,rep,do_build=not replay,independent=independent)
    pred,meta=compose(state,raw,rep,arm)
    meta.update(state=state,arm=arm,rep=rep,replay=replay,independent_clean=independent,packages=ownership,dirty_domains=SF.dirty(state),
       package_seconds=package_time,total_update_seconds=package_time+meta['update_shared_seconds'],observed_wall_seconds=time.perf_counter()-a,
       start_commit=(ROOT/'repo_commit.txt').read_text().strip(),hardware=subprocess.run(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used','--format=csv,noheader'],capture_output=True,text=True).stdout.strip())
    tag=f'{state}_{arm}_r{rep}'+('_ind' if independent else '_replay' if replay else '')
    out=OUT/'runs';out.mkdir(parents=True,exist_ok=True); np.savez(out/(tag+'.npz'),**pred);dump(meta,out/(tag+'.json'))
    log('UPDATE',tag,meta['total_update_seconds'],meta['timings'],meta['converged'])

def verify(state,rep):
    """Hard stop on ownership/numerical disagreement before continuing the primary chain."""
    ad=OUT/'runs';a=np.load(ad/f'{state}_F_r{rep}.npz');b=np.load(ad/f'{state}_S_r{rep}.npz');record={}
    for k in a.files:
        rel=float(np.abs(a[k]-b[k]).sum()/max(float(np.abs(a[k]).sum()),1e-30))
        mx=float(np.abs(a[k]-b[k]).max()/max(float(np.abs(a[k]).max()),1e-30))
        record[k]=dict(relative_L1=rel,normalized_max=mx)
    ok=all(v['relative_L1']<=1e-6 and v['normalized_max']<=1e-5 for v in record.values())
    identities={}
    for arm in ('F','S'):
        md=json.load(open(ad/f'{state}_{arm}_r{rep}.json'))
        ok &= md['converged'] and md['closure_max']<=1e-6 and md['back_hits']==0 and max(md['rho'])<1
        ok &= all(max(v)<=1+1e-6 for v in md['max_colsum'].values())
        ok &= all(v['leak_rate']<=1e-5 and v['truncated']==0 for v in md['transport_stats'].values())
        identities[arm]={ci:p['content_hash'] for ci,p in md['packages'].items()}
    dump(dict(passed=ok,fields=record,package_hashes=identities),OUT/f'verify_{state}_r{rep}.json')
    assert ok,'Shared-operator ownership/numerical gate failed: stop for focused diagnosis'
    log('VERIFIED',state,rep)

def resident():
    """Actual repeated rigid updates with resident GPU K; excludes one-time cache/CSR initialization."""
    raw,ownership,lt=load_packages('rigid','S',0)
    ops,ct=timed(lambda:{ci:KO.KOp(r['idx'],r['val'],r['shape']) for ci,r in raw.items()})
    records=[];baseline=np.load(OUT/'runs/rigid_S_r0.npz')
    for i in range(3):
        p,md=compose('rigid',raw,0,'S_resident',prepared_K=ops)
        rel=max(float(np.abs(p[k]-baseline[k]).sum()/max(float(np.abs(baseline[k]).sum()),1e-30)) for k in p)
        assert rel<=1e-6
        records.append(dict(meta=md,relative_to_disk_cache_S=rel))
    dump(dict(initial_cache_load_and_check=lt,initial_CSR=ct,records=records,
        note='Measured rigid updates only. Do not add a hypothetical removed TR cost and call it a measured dent resident update.'),OUT/'resident_rigid.json')

def extract(state):
    sd=SF.scene(state); ss=CM.classify_sceneset(sd); geo=json.load(open(OUT/'cycles'/f'ccam_{state}.json')); res={}
    for cam,g in geo.items():
        fs=sorted((OUT/'cycles').glob(f'cimg_{state}_{cam}_s*.npy'))
        res[cam]=CM.cycles_ports(ss,g,[np.load(f) for f in fs])
    return res

def reference_gate():
    R,D=extract('rest'),extract('dent'); lab=CM.labels_of(SF.scene('rest')); m=SF.intrinsic('rest',0)[0]
    disc=np.concatenate([np.hypot(m['port_center'][:,0],m['port_center'][:,1])<=.025,np.zeros(len(m['port_area'])+400,dtype=bool)])
    out={}; passed=False
    for unit in ('amended','declared'):
        r,d=R['gapdown'][unit],D['gapdown'][unit]; sel=r['sel']&d['sel']&(lab=='lower:top')
        for name,msk in (('face',sel),('disc',sel&disc)):
            x=r['seeds'][:,msk];y=d['seeds'][:,msk]; delta=y-x; n=len(x)
            means=delta.mean(1); base=x.mean((0,1)); change=means.mean(0)/base; se=means.std(0,ddof=1)/np.sqrt(n)/base
            dm=delta.mean(0); noise=delta.std(0,ddof=1)/np.sqrt(n)
            l2=np.linalg.norm(dm,axis=0); nl2=np.linalg.norm(noise,axis=0)
            gate=(np.abs(change)>np.maximum(.002,3*se))&(l2>3*nl2)
            out[unit+'|'+name]=dict(n_ports=int(msk.sum()),relative_change=change.tolist(),se=se.tolist(),l2_signal_over_noise=(l2/np.maximum(nl2,1e-30)).tolist(),detected=gate.tolist())
            if unit=='amended':passed|=bool(gate.any())
    dump(dict(proceed=passed,groups=out),OUT/'reference_gate.json');log('REFERENCE_GATE',passed,out)

def audit():
    import platform
    files={}
    for folder in ('phys_ref53','fnto_c54','fnto_k55'):
        for p in sorted((Path(__file__).resolve().parent.parent/folder).rglob('*')):
            if p.is_file() and '__pycache__' not in str(p): files[str(p.relative_to(Path(__file__).resolve().parent.parent))]=hash_file(p)
    dump(dict(platform=platform.platform(),python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,source_hashes=files,
        historical_repo_commit=(HIST/'repo_commit.txt').read_text(),gpu=subprocess.run(['nvidia-smi'],capture_output=True,text=True).stdout),OUT/'environment.json')

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True);cmd=sys.argv[1]
    if cmd in ('prepare','reference_gate','audit','resident'):globals()[cmd]()
    elif cmd=='verify':verify(sys.argv[2],int(sys.argv[3]))
    elif cmd in ('update','replay'):update(sys.argv[2],sys.argv[3],int(sys.argv[4]),replay=cmd=='replay',independent='ind' in sys.argv[5:])
    sys.stdout.flush();os._exit(0)
