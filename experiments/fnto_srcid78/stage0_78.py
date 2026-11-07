"""WL78 Stage 0 development diagnostics (archived, already-scored data only; the WL77A Part 0 references are never read here).
 D1 translate/image correspondence of the three boundary-far WL74 Stage 4 sources and their S16-R analogues (101<-50, 105<-52, 107<-53):
    exact slab operation + in-face translation, exact receiver-port map, exact incoming/outgoing bin maps from the port frames;
    full-response comparison of the replica-mean references by distance stratum, with replica noise, unmapped flux and beta rows.
 D2 the same comparison for the preserved A75 predictions (both seeds), W and UV paths separately; per-stratum model error.
 D3 reference collided energy and fine16 contrast signal outside the W edge support (128-NN of the source) for S16-R and the scored S4 sources.
 D4 origin-shift sensitivity of A75: a constant offset of the absolute-coordinate node features (physically identical input).
    python stage0_78.py <out.json>"""
import sys,os,json,math
import numpy as np
import torch
import torch.nn.functional as F
from common78 import ROOT,R75,R75L,R73,R71,R68,INP68,environment,load
RG=environment();torch.set_grad_enabled(False)
import fit71,fit73,rank75 as RK,ato74 as AT,family68 as FM,sym72 as SY,probes68 as P68
from scipy.spatial import cKDTree
dev='cuda';TR,C=16,3
g=RG.device(fit71.guarded('geometry_rest'));P=g['P'];x=g['rest_xyz'].double().cpu().numpy()
dom=P68.domain(RG,'rest');tri=dom.port_tris[:,0];Tf=dom.t[tri].double();Bf=dom.b[tri].double();Nf=dom.n[tri].double()
parent=g['parent'];bm=FM.bin_map(dev);face=np.rint(g['node'][:,15].cpu().numpy()*5).astype(int)
tree=cKDTree(x);srcE=g['src'];dstE=g['dst']

def bundle_path(s):
    if s>=100:return R75L/'refs/bundles'/f'rest_s{s}.pt'
    if s<4:return INP68/'fit'/f'rest_s{s}.pt'
    if s<8:return R68/'heldout/bundles'/f'rest_s{s}.pt'
    if s<16:return R71/'refs/bundles'/f'rest_s{s}.pt'
    return R73/'refs/bundles'/f'rest_s{s}.pt'
PORTS={s:p for s,p in fit73.populations()['S16R']}
PORTS.update({int(k):v['port'] for k,v in json.loads((R75L/'ports4.json').read_text())['ports'].items()})
def bundle(s):
    b=load(bundle_path(s));p=int(b['port']) if 'port' in b else PORTS[s];assert p==PORTS[s],('port',s,p);return b['S'].double(),p

def bin_dirs(N):
    k=torch.arange(N*N,dtype=torch.float64);u=((k%N)+.5)/N;v=((k//N)+.5)/N;xx,yy=RG.G.square_to_disk(u,v)
    return torch.stack([xx,yy,torch.sqrt((1-xx*xx-yy*yy).clamp(min=0))],1).to(dev)
def bin_map(src,dstp,M,N):
    """bin o of port src -> bin of port dstp under the direction map M (3x3); exact-bijection flag."""
    L=bin_dirs(N);s=torch.as_tensor(src,device=dev);q=torch.as_tensor(dstp,device=dev)
    w=L[None,:,0,None]*Tf[s][:,None]+L[None,:,1,None]*Bf[s][:,None]+L[None,:,2,None]*Nf[s][:,None];w=w@torch.as_tensor(M,device=dev,dtype=torch.float64).T
    lq=torch.stack([(w*Tf[q][:,None]).sum(-1),(w*Bf[q][:,None]).sum(-1),(w*Nf[q][:,None]).sum(-1)],-1)
    mp=RG.G.local_to_bin(lq.float(),N).long();ex=all(len(set(r.tolist()))==N*N for r in mp.cpu());return mp,ex

def correspondence(ps,pf):
    """slab op k and in-face translation delta with op(x[ps]) + delta = x[pf], delta along one in-face axis (the edge direction)."""
    best=[]
    for op in SY.ops(True):
        M=SY.matrix(op);y=M@x[ps];d=x[pf]-y;nz=np.abs(d)>1e-9
        if nz.sum()<=1 and abs(d@Nf[pf].cpu().numpy())<1e-9:
            yj=x@M.T+d;dd,ii=tree.query(yj);mp=np.where(dd<1e-6,ii,-1);best.append((float(np.abs(d).sum()),op,M,d,mp))
    best.sort(key=lambda t:t[0]);return best

def map_tensor(X,ps,pf,M,mp):
    """X (K,16,P,64,3) response of source ps -> response of source pf under the correspondence (unmapped receivers stay 0)."""
    mi,exi=bin_map([ps],[pf],M,4);J=np.nonzero(mp>=0)[0];Jp=mp[J];mo,exo=bin_map(J,Jp,M,8);assert exi and exo,'bins do not permute'
    out=torch.zeros_like(X);Jt=torch.as_tensor(J);Jpt=torch.as_tensor(Jp);mo=mo.cpu();mi=mi[0].cpu()
    for a in range(16):out[:,mi[a],Jpt[:,None],mo]=X[:,a,Jt[:,None],torch.arange(64)[None]]
    return out,mi

EDGES=[0,5,10,20,40,1e9]
def strata(pf):
    r=np.linalg.norm(x-x[pf],axis=1)*1e3;return [(f'{a}-{b if b<1e8 else "inf"}mm',torch.as_tensor((r>=a)&(r<b))) for a,b in zip(EDGES[:-1],EDGES[1:])]
def contrast(Y,d):
    """fine16 contrast over the incoming bins: (Y[a]-mean_a Y)/d[a], Y (...,16,P,64,3) native."""
    Yf=FM.aggregate(Y,'fine16',parent.cpu(),bm.cpu());return (Yf-Yf.mean(-4,keepdim=True))/d.view(*([1]*(Yf.dim()-4)),16,1,1,1)
def nrm(X,mask):return X[...,mask,:,:].square().sum(dim=tuple(i for i in range(X.dim()) if i!=X.dim()-1)).sqrt()   # per channel

def columns_split(m,ports):
    """RK.columns split into the W (edge) and UV (latent) parts of T."""
    import family71 as FW
    M=m.M;st=m.state(g);z=st['z'];nh=m.node_head(z)
    U=torch.softmax(nh[:,:TR*M*C].view(P,TR,M,C),dim=0);V=F.softplus(nh[:,TR*M*C:2*TR*M*C].view(P,TR,M,C));mass=torch.sigmoid(nh[:,2*TR*M*C:].view(P,M,C))
    e=FW.edges(g,ports);src=g['src'][e];dst=g['dst'][e];n=e.shape[0]
    h=st['s'][src]+st['d'][dst]+m.pair_layer(g['pair'][e]);Wr=F.softplus(m.edge(h)).view(n,-1,M,C)*g['prior'][e][:,:,None,:]
    pi=torch.as_tensor(ports,device=dev);scale=mass[pi]/(Wr.sum(1)+V[pi].sum(1)+1e-12)
    Tw=torch.zeros(n,P,M,C,device=dev).scatter_add(1,dst[:,:,None,None].expand(-1,-1,M,C),Wr*scale[:,None])
    Tu=torch.einsum('jkrc,nkrc->njrc',U,V[pi]*scale[:,None]);return Tw,Tu,torch.softmax(m.in_logits,0),torch.softmax(m.out_logits,1)
def predict(m,port,split=False):
    Tw,Tu,A,O=columns_split(m,[port]);d=(1-g['beta'][[port]]).float()
    pw=AT.predict_m(Tw,d,A,O,'native')[0].double().cpu();pu=AT.predict_m(Tu,d,A,O,'native')[0].double().cpu()
    return (pw,pu) if split else pw+pu

models={s:RK.load_run(R75/'runs'/f'A75_{s}.pt') for s in (5901,5902)}
out=dict(note='WL78 Stage 0 development diagnostics on archived, already-scored data; non-confirmatory',models={s:str(R75/'runs'/f'A75_{s}.pt') for s in models})

# ---------------------------------------------------------------- D1 / D2
PAIRS=[(50,101),(52,105),(53,107)];out['pairs']={}
for st,sf in PAIRS:
    Ss,ps=bundle(st);Sf,pf=bundle(sf);cands=correspondence(ps,pf);assert cands,('no correspondence',st,sf)
    rows=[]
    for dist,op,M,delta,mp in cands[:2]:
        Sm,mi=map_tensor(Ss,ps,pf,M,mp);mapped_f=torch.zeros(P,dtype=torch.bool);mapped_f[torch.as_tensor(mp[mp>=0])]=True
        A=Sf.mean(0);B=Sm.mean(0);Ef=Sf.mean(0).sum((0,2))      # collided energy per receiver (P,3)
        Es=Ss.mean(0).sum((0,2));un_s=(Es[torch.as_tensor(mp<0)].sum(0)/Es.sum(0)).tolist();un_f=(Ef[~mapped_f].sum(0)/Ef.sum(0)).tolist()
        vA=Sf.var(0)/Sf.shape[0];vB=Sm.var(0)/Ss.shape[0]
        df=1-g['beta'][pf].double().cpu();ds=(1-g['beta'][ps].double().cpu())[mi.argsort()]
        cA=contrast(Sf,df);cB=contrast(Sm,df);cAm=cA.mean(0);cBm=cB.mean(0);cvA=cA.var(0)/cA.shape[0];cvB=cB.var(0)/cB.shape[0]
        res=dict(op=list(map(str,op)),delta_mm=(delta*1e3).round(4).tolist(),unmapped_flux_far=un_f,unmapped_flux_train=un_s,
                 beta_far_vs_mapped_train_rel=float((df-ds).norm()/df.norm()),strata={})
        tot=nrm(A,mapped_f).square()
        for name,mask in strata(pf):
            mk=mask&mapped_f
            if mk.sum()==0:continue
            a=nrm(A,mk);res['strata'][name]=dict(n=int(mk.sum()),share=(a.square()/tot).tolist(),phys_rel=(nrm(A-B,mk)/a).tolist(),
                noise_rel=(nrm(vA+vB,mk).sqrt()/a if False else (vA+vB)[...,mk,:,:].sum(dim=(0,1,2)).sqrt()/a).tolist(),
                con_phys_rel=(nrm(cAm-cBm,mk)/nrm(cAm,mk)).tolist(),con_noise_rel=((cvA+cvB)[...,mk,:,:].sum(dim=(0,1,2)).sqrt()/nrm(cAm,mk)).tolist(),
                con_share=(nrm(cAm,mk).square()/nrm(cAm,mapped_f).square()).tolist())
        # model side
        mres={}
        for sd,m in models.items():
            pw_s,pu_s=predict(m,ps,True);pw_f,pu_f=predict(m,pf,True);ps_=pw_s+pu_s;pf_=pw_f+pu_f
            pm,_=map_tensor(ps_[None],ps,pf,M,mp);pm=pm[0];cpf=contrast(pf_[None],df)[0];cpm=contrast(pm[None],df)[0]
            Ssm=Ss.mean(0);cSs=contrast(Ss,ds if False else (1-g['beta'][ps].double().cpu())).mean(0);cps=contrast(ps_[None],1-g['beta'][ps].double().cpu())[0]
            mm={}
            for name,mask in strata(pf):
                mk=mask&mapped_f
                if mk.sum()==0:continue
                a=nrm(A,mk);ca=nrm(cAm,mk)
                mm[name]=dict(model_diff_rel=(nrm(pf_-pm,mk)/a).tolist(),err_far_rel=(nrm(pf_-A,mk)/a).tolist(),err_trainmapped_rel=(nrm(pm-B,mk)/a).tolist(),
                    con_model_diff_rel=(nrm(cpf-cpm,mk)/ca).tolist(),con_err_far_rel=(nrm(cpf-cAm,mk)/ca).tolist(),con_err_trainmapped_rel=(nrm(cpm-cBm,mk)/ca).tolist(),
                    uv_share_far=(nrm(pu_f,mk).square()/(nrm(pw_f,mk).square()+nrm(pu_f,mk).square())).tolist())
            uvs=lambda pw,pu:(pu.sum((0,1,2))/(pw+pu).sum((0,1,2))).tolist()
            mres[sd]=dict(strata=mm,uv_energy_share_far=uvs(pw_f,pu_f),uv_energy_share_train=uvs(pw_s,pu_s),
                         err_far_total=(nrm(pf_-A,mapped_f)/nrm(A,mapped_f)).tolist(),err_train_total=(nrm(ps_-Ssm,torch.ones(P,dtype=torch.bool))/nrm(Ssm,torch.ones(P,dtype=torch.bool))).tolist(),
                         con_err_far_total=(nrm(cpf-cAm,mapped_f)/nrm(cAm,mapped_f)).tolist(),con_err_train_total=(nrm(cps-cSs,torch.ones(P,dtype=torch.bool))/nrm(cSs,torch.ones(P,dtype=torch.bool))).tolist())
        res['models']=mres;rows.append(res);print('PAIR',st,sf,json.dumps(res['strata'].get('0-5mm',{}))[:400],flush=True)
    out['pairs'][f'{st}->{sf}']=dict(train_port=ps,far_port=pf,correspondences=rows);del Ss,Sf;torch.cuda.empty_cache();json.dump(out,open(sys.argv[1],'w'),indent=1)

# ---------------------------------------------------------------- D3 support
slots=[s for s,_ in fit73.populations()['S16R']]+list(range(100,116));out['support']={}
for s in slots:
    S,p=bundle(s);A=S.mean(0);e=(srcE==p).nonzero().squeeze(1);ins=torch.zeros(P,dtype=torch.bool);ins[dstE[e].cpu()]=True;ins[p]=True
    E=A.sum((0,2));d=1-g['beta'][p].double().cpu();cA=contrast(S,d).mean(0)
    rmax=float(np.linalg.norm(x[dstE[e].cpu().numpy()]-x[p],axis=1).max()*1e3)
    out['support'][s]=dict(port=p,face=int(face[p]),edge_radius_mm=rmax,energy_outside=(E[~ins].sum(0)/E.sum(0)).tolist(),
        con_signal_outside=(nrm(cA,~ins).square()/nrm(cA,torch.ones(P,dtype=torch.bool)).square()).tolist(),
        native_signal_outside=(nrm(A,~ins).square()/nrm(A,torch.ones(P,dtype=torch.bool)).square()).tolist())
    print('SUPPORT',s,json.dumps(out['support'][s]),flush=True);del S;json.dump(out,open(sys.argv[1],'w'),indent=1)

# ---------------------------------------------------------------- D4 origin shift (A75)
out['origin_shift']={}
g0=g;base={}
for sd,m in models.items():
    for s in (0,50,101):
        _,p=(None,{0:2085,50:3439,101:224}[s]);base[(sd,s)]=predict(m,p)
for dm in ((2.5,0,0),(10,0,0),(0,22.5,0)):
    sh=torch.tensor(dm,device=dev)/1e3/.05;node=g0['node'].clone();node[:,0:3]+=sh;node[:,3:6]+=sh;g=dict(g0,node=node);row={}
    for sd,m in models.items():
        for s,p in ((0,2085),(50,3439),(101,224)):
            y=predict(m,p);S,_=bundle(s);A=S.mean(0);al=torch.ones(P,dtype=torch.bool)
            row[f'{sd}_s{s}']=dict(change_rel_to_ref=(nrm(y-base[(sd,s)],al)/nrm(A,al)).tolist(),err_base=(nrm(base[(sd,s)]-A,al)/nrm(A,al)).tolist(),err_shift=(nrm(y-A,al)/nrm(A,al)).tolist())
    out['origin_shift'][str(dm)]=row;g=g0;print('SHIFT',dm,json.dumps(row)[:300],flush=True)
json.dump(out,open(sys.argv[1],'w'),indent=1)
print('STAGE0_DONE',flush=True)
