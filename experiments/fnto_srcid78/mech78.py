"""WL78 mechanism diagnostics (archived, already-scored development data only; never the WL77A Part 0 references):
 M1 (frozen criterion) the Stage 0 correspondence pairs 101<-50, 105<-52, 107<-53 (exact slab operation + in-face translation, exact port and
    bin maps): near-field (0-5 mm) relative difference between the model's far-source prediction and its mapped training-source prediction,
    native and fine16 contrast, worst channel, mean over the three pairs; W+UV-Q vs A75 per seed (5901 vs 5901, 5902 vs 5902).
    improved iff ratio <= 0.5 for native and contrast in both seeds.
 M2 per-stratum model error on those far sources and their supervised analogues (descriptive), W/UV split.
 M3 dependence on absolute coordinates: a constant origin offset of every coordinate tensor of the geometry (physically identical input).
 M4 WL75 Stage 3 population (archived, scored in WL75): per-source native / fine16 contrast error, development only (descriptive).
    python mech78.py <out.json>"""
import sys,os,json
import numpy as np
import torch
import torch.nn.functional as F
from common78 import ROOT,R75,R75L,R73,R71,R68,INP68,environment,load
RG=environment();torch.set_grad_enabled(False)
import fit71,fit73,rank75 as RK,ato74 as AT,family68 as FM,sym72 as SY,probes68 as P68,q78 as Q
from scipy.spatial import cKDTree
dev='cuda';TR,C=16,3
g=RG.device(fit71.guarded('geometry_rest'));P=g['P'];x=g['rest_xyz'].double().cpu().numpy()
dom=P68.domain(RG,'rest');tri=dom.port_tris[:,0];Tf=dom.t[tri].double();Bf=dom.b[tri].double();Nf=dom.n[tri].double()
parent=g['parent'];bm=FM.bin_map(dev);tree=cKDTree(x)
PORTS={s:p for s,p in fit73.populations()['S16R']}
PORTS.update({int(k):v['port'] for k,v in json.loads((R75L/'ports4.json').read_text())['ports'].items()})
def bundle_path(s):
    if s>=100:return R75L/'refs/bundles'/f'rest_s{s}.pt'
    if s<4:return INP68/'fit'/f'rest_s{s}.pt'
    if s<8:return R68/'heldout/bundles'/f'rest_s{s}.pt'
    if s<16:return R71/'refs/bundles'/f'rest_s{s}.pt'
    return R73/'refs/bundles'/f'rest_s{s}.pt'
def bundle(s):
    b=load(bundle_path(s));p=int(b['port']) if 'port' in b else PORTS[s];assert p==PORTS[s];return b['S'].double(),p
def bin_dirs(N):
    k=torch.arange(N*N,dtype=torch.float64);u=((k%N)+.5)/N;v=((k//N)+.5)/N;xx,yy=RG.G.square_to_disk(u,v)
    return torch.stack([xx,yy,torch.sqrt((1-xx*xx-yy*yy).clamp(min=0))],1).to(dev)
def bin_map(src,dstp,M,N):
    L=bin_dirs(N);s=torch.as_tensor(src,device=dev);q=torch.as_tensor(dstp,device=dev)
    w=L[None,:,0,None]*Tf[s][:,None]+L[None,:,1,None]*Bf[s][:,None]+L[None,:,2,None]*Nf[s][:,None];w=w@torch.as_tensor(M,device=dev,dtype=torch.float64).T
    lq=torch.stack([(w*Tf[q][:,None]).sum(-1),(w*Bf[q][:,None]).sum(-1),(w*Nf[q][:,None]).sum(-1)],-1)
    mp=RG.G.local_to_bin(lq.float(),N).long();ex=all(len(set(r.tolist()))==N*N for r in mp.cpu());return mp,ex
def correspondence(ps,pf):
    best=[]
    for op in SY.ops(True):
        M=SY.matrix(op);d=x[pf]-M@x[ps]
        if (np.abs(d)>1e-9).sum()<=1 and abs(d@Nf[pf].cpu().numpy())<1e-9:
            dd,ii=tree.query(x@M.T+d);best.append((float(np.abs(d).sum()),op,M,d,np.where(dd<1e-6,ii,-1)))
    best.sort(key=lambda t:t[0]);return best[0]
def map_tensor(X,ps,pf,M,mp):
    mi,exi=bin_map([ps],[pf],M,4);J=np.nonzero(mp>=0)[0];Jp=mp[J];mo,exo=bin_map(J,Jp,M,8);assert exi and exo
    out=torch.zeros_like(X);Jt=torch.as_tensor(J);Jpt=torch.as_tensor(Jp);mo=mo.cpu();mi=mi[0].cpu()
    for a in range(16):out[:,mi[a],Jpt[:,None],mo]=X[:,a,Jt[:,None],torch.arange(64)[None]]
    return out,mi
EDGES=[0,5,10,20,1e9]
def strata(pf):
    r=np.linalg.norm(x-x[pf],axis=1)*1e3;return [(f'{a}-{b if b<1e8 else "inf"}mm',torch.as_tensor((r>=a)&(r<b))) for a,b in zip(EDGES[:-1],EDGES[1:])]
def contrast(Y,d):
    Yf=FM.aggregate(Y,'fine16',parent.cpu(),bm.cpu());return (Yf-Yf.mean(-4,keepdim=True))/d.view(*([1]*(Yf.dim()-4)),16,1,1,1)
def nrm(X,mask):return X[...,mask,:,:].square().sum(dim=tuple(i for i in range(X.dim()) if i!=X.dim()-1)).sqrt()
def predict(m,gg,port):
    cols=(lambda G_,p:Q.columns(m,G_,p)) if isinstance(m,Q.TransportQ) else (lambda G_,p:RK.columns(m,G_,p))
    T,A,O=cols(gg,[port]);return AT.predict_m(T,(1-gg['beta'][[port]]).float(),A,O,'native')[0].double().cpu()

names=sys.argv[2].split(',') if len(sys.argv)>2 else ['A75_5901','A75_5902','Q78_5901','Q78_5902']
def model(n):return RK.load_run(R75/'runs'/f'{n}.pt') if n.startswith('A75') else Q.load_run(RG,ROOT/'runs'/f'{n}.pt')
models={n:model(n) for n in names};out=dict(note='WL78 mechanism diagnostics on archived development data; non-confirmatory',models=names,pairs={})
for st,sf in ((50,101),(52,105),(53,107)):
    Ss,ps=bundle(st);Sf,pf=bundle(sf);_,op,M,delta,mp=correspondence(ps,pf);mapped=torch.zeros(P,dtype=torch.bool);mapped[torch.as_tensor(mp[mp>=0])]=True
    df=1-g['beta'][pf].double().cpu();A=Sf.mean(0);cA=contrast(Sf,df).mean(0);row=dict(op=list(map(str,op)),delta_mm=(delta*1e3).round(3).tolist(),models={})
    for n,m in models.items():
        hs=predict(m,g,ps);hf=predict(m,g,pf);hm,_=map_tensor(hs[None],ps,pf,M,mp);hm=hm[0];chf=contrast(hf[None],df)[0];chm=contrast(hm[None],df)[0]
        Sm,_=map_tensor(Ss,ps,pf,M,mp);B=Sm.mean(0);cB=contrast(Sm,df).mean(0);r={}
        for name,mask in strata(pf):
            mk=mask&mapped;a=nrm(A,mk);ca=nrm(cA,mk)
            r[name]=dict(model_diff=(nrm(hf-hm,mk)/a).tolist(),con_model_diff=(nrm(chf-chm,mk)/ca).tolist(),phys_diff=(nrm(A-B,mk)/a).tolist(),
                err_far=(nrm(hf-A,mk)/a).tolist(),err_train_mapped=(nrm(hm-B,mk)/a).tolist(),con_err_far=(nrm(chf-cA,mk)/ca).tolist(),con_err_train_mapped=(nrm(chm-cB,mk)/ca).tolist())
        row['models'][n]=r
    out['pairs'][f'{st}->{sf}']=row;print('PAIR',st,sf,{n:[round(max(v['0-5mm']['model_diff']),3),round(max(v['0-5mm']['con_model_diff']),3)] for n,v in row['models'].items()},flush=True)
    del Ss,Sf
# M1 criterion (frozen)
crit={}
if all(n in models for n in ('A75_5901','A75_5902','Q78_5901','Q78_5902')):
    for sd in ('5901','5902'):
        mn=lambda n,k:float(np.mean([max(out['pairs'][p]['models'][n]['0-5mm'][k]) for p in out['pairs']]))
        crit[sd]=dict(native_ratio=mn(f'Q78_{sd}','model_diff')/mn(f'A75_{sd}','model_diff'),contrast_ratio=mn(f'Q78_{sd}','con_model_diff')/mn(f'A75_{sd}','con_model_diff'),
                      A75=[mn(f'A75_{sd}','model_diff'),mn(f'A75_{sd}','con_model_diff')],Q78=[mn(f'Q78_{sd}','model_diff'),mn(f'Q78_{sd}','con_model_diff')])
    crit['improved']=all(crit[s]['native_ratio']<=.5 and crit[s]['contrast_ratio']<=.5 for s in ('5901','5902'))
out['criterion']=crit
# M3 origin offset of every coordinate tensor
shift={}
for dm in ((10.,0,0),(0,22.5,0)):
    sh=torch.tensor(dm,device=dev,dtype=torch.float32)/1e3;node=g['node'].clone();node[:,0:3]+=sh/.05;node[:,3:6]+=sh/.05
    gs=dict(g,node=node,xyz=g['xyz']+sh,rest_xyz=g['rest_xyz']+sh);row={}
    for n,m in models.items():
        for s in (0,50,101):
            p=PORTS[s];a=predict(m,g,p);b=predict(m,gs,p);al=torch.ones(P,dtype=torch.bool);row[f'{n}_s{s}']=(nrm(b-a,al)/nrm(a,al)).tolist()
    shift[str(dm)]=row
out['origin_offset']=shift;print('SHIFT',json.dumps(shift)[:600],flush=True)
# M4 WL75 Stage 3 population (development)
s4={}
for s in range(100,116):
    S,p=bundle(s);A=S.mean(0);d=1-g['beta'][p].double().cpu();cA=contrast(S,d).mean(0);al=torch.ones(P,dtype=torch.bool);row={}
    for n,m in models.items():
        h=predict(m,g,p);row[n]=dict(native=(nrm(h-A,al)/nrm(A,al)).tolist(),con_fine16=(nrm(contrast(h[None],d)[0]-cA,al)/nrm(cA,al)).tolist())
    s4[s]=row;del S
out['wl75_stage3_population_dev']=s4
json.dump(out,open(sys.argv[1],'w'),indent=1);print('MECH_DONE',json.dumps(crit),flush=True)
