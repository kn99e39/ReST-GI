"""WL57 runtime geometry only: material cells, current metric and exact no-collision paths.
No reference K or targets are read by this module.
"""
import sys,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree

HERE=Path(__file__).resolve().parent
for d in ('phys_ref53','fnto_c54','fnto_k55','fnto_sel56'):sys.path.insert(0,str(HERE.parent/d))
import p53_geom as G
import p53_fixtures as FX
import p53_tracer as TR
import k55_ops as KO

DEV='cuda'; DIN=16;DOUT=64;NEAR=128;RANK=16;MIX=8

# (depth metres, radius metres, centre x/y metres). No scene/light/target input.
POSES={
 'rest':(0.,.02,0.,0.),
 'tr00':(.002,.015,-.02,-.01),'tr01':(.004,.02,-.02,.01),'tr02':(.006,.025,.02,-.01),
 'tr03':(.002,.02,.02,.01),'tr04':(.004,.025,-.01,-.02),'tr05':(.006,.015,.01,.02),
 'tr06':(.002,.025,0.,-.02),'tr07':(.004,.015,0.,.02),'tr08':(.006,.02,-.02,0.),
 'tr09':(.002,.015,.01,-.01),'tr10':(.004,.02,-.01,.01),'tr11':(.006,.025,.01,0.),
 'va0':(.003,.0175,-.0125,-.0075),'va1':(.005,.0225,.0175,.0125),
 'te0':(.005,.02,0.,0.),'te1':(.0035,.0175,.0125,-.0075),
 'te2':(.0055,.0225,-.0175,.0125),'te3':(.006,.015,0.,0.)}
TRAIN=['rest']+[f'tr{i:02}' for i in range(12)]; VAL=['va0','va1'];TEST=['te0','te1','te2','te3']

def mesh(name):
    m=FX.slab_mesh();dep,rad,x,y=POSES[name]
    return m,G.map_dent(m['V'],dep,rad,FX.H,(x,y)) if dep else m['V'].copy()

def geometry(name,ballistic=True,q=256):
    """Same 2.5 mm cells /16 in/64 out as WL56. All coordinates in the object's supplied rigid material frame."""
    t0=time.perf_counter();m,V=mesh(name);fm,VC,parent=KO.refined(m,V,4)
    fr=G.tri_frames(fm['V'],VC,fm['F'],fm['tri_port'],fm['port_tangent']);tp=fm['tri_port'];P=len(parent)
    area=np.bincount(tp,weights=fr['area'],minlength=P)
    def avg(v):return np.stack([np.bincount(tp,weights=v[:,k]*fr['area'],minlength=P)/area for k in range(v.shape[1])],1)
    xyz=avg(fr['centroid']);normal=avg(fr['n']);normspread=1-np.linalg.norm(normal,axis=1);normal/=np.linalg.norm(normal,axis=1)[:,None]
    fr0=G.tri_frames(fm['V'],fm['V'],fm['F'],tp,fm['port_tangent']);ar0=np.bincount(tp,weights=fr0['area'],minlength=P)
    xyz0=np.stack([np.bincount(tp,weights=fr0['centroid'][:,k]*fr0['area'],minlength=P)/ar0 for k in range(3)],1)
    # Local height-field thickness from corresponding top/bottom cells, not the dent descriptor.
    top=fm['port_face']==5;bot=fm['port_face']==4;tree=cKDTree(xyz0[top,:2]);ti=tree.query(xyz0[:,:2])[1]
    btree=cKDTree(xyz0[bot,:2]);bi=btree.query(xyz0[top,:2])[1]
    thick_top=xyz[top,2]-xyz[bot,2][bi];thick=thick_top[ti]
    d=xyz-xyz0
    local=np.column_stack([xyz0/.05,xyz/.05,d/.006,normal,area/ar0,thick/.01,normspread,fm['port_face']/5])
    assert local.shape==(P,16)
    v=np.linspace(-.05,.05,11);xx,yy=np.meshgrid(v,v,indexing='ij');gt=tree.query(np.stack([xx.ravel(),yy.ravel()],1))[1]
    globalf=np.r_[d[top,2][gt]/.006,[d[:,2].min()/.006,d[:,2].max()/.006,d[:,2].mean()/.006]]
    node=np.column_stack([local,np.broadcast_to(globalf,(P,124))]);assert node.shape==(P,140)
    idx=cKDTree(xyz0).query(xyz0,k=NEAR)[1]
    src=np.repeat(np.arange(P),NEAR);dst=idx.reshape(-1)
    dr=(xyz0[dst]-xyz0[src])/.01;dist=np.linalg.norm(xyz[dst]-xyz[src],axis=1)
    pair=np.column_stack([dr,np.linalg.norm(dr,axis=1),dist/.01,(normal[dst]*normal[src]).sum(1)])
    lengths=np.array([.0181,.0058,.0018])
    prior=np.exp(-dist[:,None]/lengths[None])*.0025**2/(dist[:,None]**2+.0025**2)
    cv=lambda x,dtype=torch.float32:torch.as_tensor(x,dtype=dtype,device=DEV)
    h=hashlib.sha256(np.ascontiguousarray(VC).tobytes()).hexdigest()
    out=dict(name=name,node=cv(node),src=cv(src,torch.long),dst=cv(dst,torch.long),pair=cv(pair),prior=cv(prior),
        graph=cv(idx[:,:16],torch.long),parent=cv(parent,torch.long),P=P,signature=h,area=cv(area),rest_xyz=cv(xyz0),xyz=cv(xyz),
        frame=fr,mesh=fm,Vcur=VC,coarse=m,meta=dict(signature=h,P=P,preprocess_seconds=time.perf_counter()-t0))
    if ballistic:
        B,beta,st=ballistic_matrix(fm,VC,q=q);out.update(B=B,beta=beta);out['meta']['ballistic']=st
    out['meta']['total_seconds']=time.perf_counter()-t0
    return out

def ballistic_matrix(m,V,q=256,seed=57501):
    """Analytic exp(-sigma_t distance) integrated over cells/bins by fixed-seed geometry sampling. No scattered transport oracle."""
    dom=KO.DomainX(m,V);ns=dom.P*DIN;gen=torch.Generator(device=DEV);gen.manual_seed(seed)
    rows=[];cols=[];vals=[];miss=wrong=0;t0=time.perf_counter()
    for i in range(0,ns,max(1,2_000_000//q)):
        cells=torch.arange(i,min(i+max(1,2_000_000//q),ns),device=DEV)
        x,d,n,w,src=TR.cell_source(dom,cells,q,DIN,gen)
        t,p=dom.isect(x-dom.eps*n,d);hit=torch.isfinite(t);miss+=int((~hit).sum());j=hit.nonzero().squeeze(1)
        good=(d[j]*dom.n[p[j]]).sum(1)>0;wrong+=int((~good).sum());j=j[good];pj=p[j]
        b=G.local_to_bin(G.to_local(d[j],dom.n[pj],dom.t[pj],dom.b[pj]),8)
        rows.append(dom.tri_port[pj]*64+b);cols.append(src[j]);vals.append(torch.exp(-dom.sigma_t*t[j].double())/q)
    B=torch.sparse_coo_tensor(torch.stack([torch.cat(rows),torch.cat(cols)]),torch.cat(vals),(dom.P*64,ns)).coalesce()
    beta=torch.zeros(ns,device=DEV,dtype=torch.float64).index_add_(0,B.indices()[1],B.values()).float().view(dom.P,16)
    assert float(beta.max())<=1+1e-7
    B=torch.sparse_coo_tensor(B.indices(),B.values().float(),B.shape).coalesce()
    torch.cuda.synchronize()
    return B,beta,dict(seconds=time.perf_counter()-t0,q=q,seed=seed,rays=ns*q,miss=miss,wrong=wrong,nnz=B._nnz(),
        bytes=B.indices().numel()*8+B.values().numel()*4+beta.numel()*4,semantics='zero-collision first exit, geometry+material only')

def serial(g):
    return {k:(v.detach().cpu() if isinstance(v,torch.Tensor) else v) for k,v in g.items() if k not in ('frame','mesh','Vcur','coarse')}

def device(g):return {k:v.to(DEV) if isinstance(v,torch.Tensor) else v for k,v in g.items()}
