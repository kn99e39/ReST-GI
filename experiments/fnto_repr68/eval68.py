"""WL68 evaluation (reads fits, frozen checkpoints and physical references; trains nothing except the labelled
held-out angular-transfer ORACLE). Physical errors are unbiased cross-replica statistics from bin x bin Grams:
for a bin-space map L and deterministic prediction h, E = tr(L hh'L') - 2 mean_k tr(L h R_k' L') + mean_{k!=l} tr(L R_k R_l' L'),
S = mean_{k!=l} tr(L R_k R_l' L'); abs L=I, directional contrast L = H D^-1 (WL66 Z normalisation, model budget D).
Fitting-population statistics use ONLY replicas 4-7 (disjoint from fitting replicas 0-3). Jackknife, t(K-1)."""
import sys,os,json,time,itertools,argparse
from pathlib import Path
import numpy as np
import torch
from common68 import CFG,ROOT,INP,FIT_GEOMS,NBIN,sha,write_new,save_new,environment,verify_inputs
import family68 as FM

SPACES=('native','fine16','coarse');TOT=('tot_fine16','tot_coarse')
STD=[('abs_native','abs','native'),('abs_fine16','abs','fine16'),('abs_coarse','abs','coarse'),('con_native','con','native'),
     ('con_fine16','con','fine16'),('con_coarse','con','coarse'),('tot_fine16','abs','tot_fine16'),('tot_coarse','abs','tot_coarse')]
COARSE_ONLY=[('con_coarse','con','coarse')]
def Hm():return torch.eye(NBIN,dtype=torch.float64)-torch.full((NBIN,NBIN),1./NBIN,dtype=torch.float64)
def c16(Xf,parent):return torch.zeros(*Xf.shape[:-3],240,16,3,device=Xf.device,dtype=Xf.dtype).index_add(Xf.dim()-3,parent,Xf)

def grams(X,Y):
    """X (k,a,Q,C), Y (l,b,Q,C) -> (k,l,a,b,C) float64 on CPU (chunked over Q)."""
    X=X.reshape(X.shape[0],X.shape[1],-1,X.shape[-1]);Y=Y.reshape(Y.shape[0],Y.shape[1],-1,Y.shape[-1]);out=None
    for s in range(0,X.shape[-2],65536):
        g=torch.einsum('kaqc,lbqc->klabc',X[...,s:s+65536,:].double(),Y[...,s:s+65536,:].double());out=g if out is None else out+g
    return out.cpu()

class Ref:
    """One probe matrix (geometry, port): K evaluation replicas of collided S (K,16,P,64,3) and zero-collision B16."""
    def __init__(self,key,S,B16,d,parent,bm,ball16,coarse_only=False):
        self.key=key;self.S=S;self.B16=B16;self.K=S.shape[0];self.d=d.double().cpu();self.parent=parent;self.bm=bm;self.ball16=ball16
        Sg,Bg=self.gpu();self.RR={}
        for sp in SPACES:X=FM.aggregate(Sg,sp,parent,bm);self.RR[sp]=grams(X,X)
        Tf=FM.aggregate(Sg,'fine16',parent,bm)+Bg;self.RR['tot_fine16']=grams(Tf,Tf);Tc=c16(Tf,parent);self.RR['tot_coarse']=grams(Tc,Tc)
        self.coarse=FM.aggregate(Sg,'coarse',parent,bm).double().cpu();self.energy=Sg.double().sum((2,3)).cpu()
        del Sg,Bg,Tf,Tc;torch.cuda.empty_cache()
    def gpu(self):return self.S.cuda(),self.B16.cuda()

def cross(ref,Sg,Bg,h):
    """Candidate Grams vs replicas. h: native prediction (16,P,64,3) on GPU, or ('coarse', (16,240,16,3))."""
    o={'hh':{},'hR':{}}
    if isinstance(h,tuple):
        hc=h[1].double();X=FM.aggregate(Sg,'coarse',ref.parent,ref.bm)
        o['hh']['coarse']=grams(hc[None],hc[None])[0,0];o['hR']['coarse']=grams(X,hc[None])[:,0];o['coarse']=hc.cpu();return o
    for sp in SPACES:
        x=FM.aggregate(h,sp,ref.parent,ref.bm);X=FM.aggregate(Sg,sp,ref.parent,ref.bm)
        o['hh'][sp]=grams(x[None],x[None])[0,0];o['hR'][sp]=grams(X,x[None])[:,0]
    hf=FM.aggregate(h,'fine16',ref.parent,ref.bm)+ref.ball16;Tf=FM.aggregate(Sg,'fine16',ref.parent,ref.bm)+Bg
    for nm,x,X in (('tot_fine16',hf,Tf),('tot_coarse',c16(hf,ref.parent),c16(Tf,ref.parent))):
        o['hh'][nm]=grams(x[None],x[None])[0,0];o['hR'][nm]=grams(X,x[None])[:,0]
    o['energy']=h.double().sum((1,2)).cpu();o['coarse']=FM.aggregate(h,'coarse',ref.parent,ref.bm).double().cpu()
    return o

def Lmap(kind,d):return torch.eye(NBIN,dtype=torch.float64) if kind=='abs' else Hm()@torch.diag(1/d)
def quad(L,G):return torch.einsum('ab,...bez,ae->...z',L,G,L)
def signal(ref,L,space,S):
    pr=[(k,l) for k in S for l in S if k!=l];return sum(quad(L,ref.RR[space][k,l]) for k,l in pr)/len(pr)
def error(ref,cg,L,space,S):
    return quad(L,cg['hh'][space])-2*sum(quad(L,cg['hR'][space][k]) for k in S)/len(S)+signal(ref,L,space,S)

def jackknife(fn,K):
    t=CFG['t_quantiles'][str(K-1)];full=fn(list(range(K)));loo=[fn([j for j in range(K) if j!=k]) for k in range(K)];res={}
    for n,v in full.items():
        xs=torch.tensor([l[n] for l in loo],dtype=torch.float64);se=float(((K-1)/K*(xs-xs.mean()).square().sum()).sqrt())
        res[n]=dict(est=float(v),se=se,lo=float(v)-t*se,hi=float(v)+t*se)
    return res
def sq(x):return float(np.sqrt(max(x,0.)))

def pooled(refs,cgs,metrics):
    """Per-channel pooled relative error rho = sqrt(sum E / sum S) with jackknife 95% bounds over replicas."""
    K=refs[0].K;assert all(r.K==K for r in refs)
    def fn(S):
        out={}
        for name,kind,space in metrics:
            E=sum(error(r,cg,Lmap(kind,r.d),space,S) for r,cg in zip(refs,cgs));Sg=sum(signal(r,Lmap(kind,r.d),space,S) for r in refs)
            for c in range(3):out[f'{name}/{c}']=float(E[c]/Sg[c]);out[f'{name}/{c}/S']=float(Sg[c])
        return out
    j=jackknife(fn,K);res={}
    for name,_,_ in metrics:
        res[name]=[dict(rho=sq(j[f'{name}/{c}']['est']),lo=sq(j[f'{name}/{c}']['lo']),hi=sq(j[f'{name}/{c}']['hi']),ratio=j[f'{name}/{c}']['est'],
                        ratio_se=j[f'{name}/{c}']['se'],signal=j[f'{name}/{c}/S']['est'],signal_over_SE=j[f'{name}/{c}/S']['est']/max(j[f'{name}/{c}/S']['se'],1e-300)) for c in range(3)]
    return res

def floors(refs,A,O):
    """Fixed-projector (unbiased) floors for given angular bases, pooled, coarse contrast: relaxed rank-8 tail (L7, low-biased),
    held-out split subspace (U7, high-biased), signed frozen-A (F_A) and signed frozen A+O with free signed spatial
    coefficients (PAOc) -- WL66 definitions -- and a per-bin-pair table for WL63's Pair7 bins (8,13)."""
    K=refs[0].K;out={}
    import estimators66 as E66
    for c in range(3):
        TA=E66.basis(Hm()@A[:,:,c].T.double(),tol=1e-5,cap=7) if A is not None else None
        if O is not None:
            Ot=torch.zeros(8,16,dtype=torch.float64).index_add(1,FM.bin_map(),O[:,:,c].double())
            Pcon=E66.block_projector(torch.einsum('ar,ro->aor',Hm()@A[:,:,c].T.double(),Ot).reshape(256,-1))
        def fn(S):
            GC=0.;FA=0.;PAO=0.
            for r in refs:
                L=Lmap('con',r.d);G=sum(L@r.RR['coarse'][k,l][...,c]@L.T for k in S for l in S if k!=l)/(len(S)*(len(S)-1));GC=GC+G
                if TA is not None:R=torch.eye(16,dtype=torch.float64)-TA@TA.T;FA=FA+float(torch.trace(R@G@R.T))
                if O is not None:
                    C=torch.einsum('ab,kbjo->kjao',L,r.coarse[...,c]).reshape(r.K,240,256);Y=(C-C@Pcon).reshape(r.K,-1)
                    PAO=PAO+sum(float(Y[k]@Y[l]) for k in S for l in S if k!=l)/(len(S)*(len(S)-1))
            sc=float(torch.trace(GC));o=dict(L7=E66.tail(GC,7)/sc,S_C=sc)
            if TA is not None:o['F_A']=FA/sc
            if O is not None:o['PAOc']=PAO/sc
            u=torch.zeros(16,dtype=torch.float64);u[8]=1;u[13]=-1;o['pair8_13_phys']=float(u@GC@u)
            return o
        j=jackknife(fn,K);out[c]={k:dict(rho=sq(v['est']),hi=sq(v['hi']),lo=sq(v['lo'])) if k not in ('S_C','pair8_13_phys') else v for k,v in j.items()}
    return out

def per_bin(ref,cg,space='coarse'):
    """Point estimates of per-probe (bin) relative abs error, all replicas."""
    S=list(range(ref.K));pr=[(k,l) for k in S for l in S if k!=l]
    RR=sum(ref.RR[space][k,l] for k,l in pr)/len(pr);hR=sum(cg['hR'][space][k] for k in S)/len(S)
    e=torch.diagonal(cg['hh'][space],dim1=0,dim2=1)-2*torch.diagonal(hR,dim1=0,dim2=1)+torch.diagonal(RR,dim1=0,dim2=1)
    s=torch.diagonal(RR,dim1=0,dim2=1);return (e.clamp(min=0)/s).sqrt().T.tolist()  # (16,3)

def corr_slope(refs,cgs):
    out=[]
    for c in range(3):
        x=torch.cat([r.coarse.mean(0)[...,c].flatten() for r in refs]);y=torch.cat([cg['coarse'][...,c].flatten() for cg in cgs])
        xc=x-x.mean();yc=y-y.mean();out.append(dict(corr=float(xc@yc/(xc.norm()*yc.norm())),slope=float(xc@yc/(xc@xc))))
    return out

def Q(p):
    th=CFG['criteria']['quality'];chk={}
    for c in range(3):
        chk[c]=dict(abs_coarse=p['abs_coarse'][c]['hi']<=th['abs_coarse_hi'],abs_fine16=p['abs_fine16'][c]['hi']<=th['abs_fine16_hi'],
                    con_coarse=p['con_coarse'][c]['hi']<=th['con_coarse_hi'])
    return dict(per_channel=chk,passed=all(all(v.values()) for v in chk.values()))

# ------------------------------------------------------------------ data
def geoms(RG):
    return {g:RG.device(torch.load(INP/'wl59'/f'geometry_{g}.pt',map_location='cpu')) for g in ('rest','pt0','cf3')}

def ports():
    fit=json.loads((INP/'wl66_ports.json').read_text())['ports'];ho=json.loads((ROOT/'heldout/ports.json').read_text())['ports']
    return {s['slot']:s['port'] for s in fit}|{s['slot']:s['port'] for s in ho}

def ball16(g,i,bm):
    x=torch.zeros(g['P']*16,16,device='cuda');x[i*16+torch.arange(16,device='cuda'),torch.arange(16,device='cuda')]=1.
    y=torch.sparse.mm(g['B'],x).T.reshape(16,g['P'],64);z=torch.zeros(16,g['P'],16,device='cuda').index_add(2,bm,y)
    return z[...,None].expand(16,g['P'],16,3).contiguous()

def load_refs(G,P):
    """FIT: WL66 replicas 4-7 (disjoint from fitting replicas 0-3). HELD-OUT: all replicas of the new references."""
    bm=FM.bin_map('cuda');refs={};fr=CFG['populations']['fit']['eval_replicas']
    for g in FIT_GEOMS:
        for s in range(4):
            b=torch.load(INP/'fit'/f'{g}_s{s}.pt',map_location='cpu');i=P[s];d=(1-G[g]['beta'][i]).float()
            refs[(g,s)]=Ref((g,s),b['S'][fr],b['B16'][fr],d,G[g]['parent'],bm,ball16(G[g],i,bm));del b
    for g,slots in CFG['populations']['heldout']['matrices'].items():
        for s in slots:
            b=torch.load(ROOT/'heldout/bundles'/f'{g}_s{s}.pt',map_location='cpu');i=P[s];d=(1-G[g]['beta'][i]).float()
            refs[(g,s)]=Ref((g,s),b['S'],b['B16'],d,G[g]['parent'],bm,ball16(G[g],i,bm));del b
    return refs

def delta_refs(refs):
    """Signed deformation references Delta_k = R(geom)_k - R(rest)_k (independent packets), first min(K) replicas."""
    bm=FM.bin_map('cuda');out={}
    for name,v in CFG['populations']['deformation'].items():
        for s in v['slots']:
            a,b=refs[(v['geometry'],s)],refs[('rest',s)];K=min(a.K,b.K)
            out[(name,s)]=Ref((name,s),a.S[:K]-b.S[:K],a.B16[:K]-b.B16[:K],a.d,a.parent,bm,a.ball16-b.ball16)
    return out

# ------------------------------------------------------------------ candidates
def net_model(spec):
    from runtime import Transport
    m=Transport().cuda()
    if spec.startswith('P_'):m.load_state_dict(torch.load(INP/'wl59'/f'{spec}.pt',map_location='cpu')['model'])
    else:m.load_state_dict(torch.load(ROOT/'runs'/f'{spec}.pt',map_location='cpu')['model'])
    return m.eval()

def fcoef(st):
    return FM.FCoef(st['in_logits'],st['out_logits'],[st[f'mass.{k}'] for k in range(2)],[st[f'sp.{k}'] for k in range(2)]).cuda()

def candidates():
    c={'frozen_5901':('net','P_5901'),'frozen_5902':('net','P_5902')}
    for r,v in CFG['runs'].items():
        if v['target']=='phys':c[r]=(v['kind'],r)
    for r in CFG['convex']:c[r]=('convex',r)
    return c

def predictor(kind,spec,G,P):
    """f(geom, slot) -> native prediction (16,P,64,3) on GPU, ('coarse', (16,240,16,3)), or None when undefined there."""
    if kind=='net':
        m=net_model(spec)
        def f(g,s):
            T,A,O=FM.network_columns(m,G[g],[P[s]]);return FM.predict(T,(1-G[g]['beta'][[P[s]]]).float(),A,O,'native')[0]
        return f,m
    if kind=='coef':
        fc=fcoef(torch.load(ROOT/'runs'/f'{spec}.pt',map_location='cpu')['model'])
        def f(g,s):
            if g not in FIT_GEOMS or s>3:return None
            T,A,O=fc.columns(FIT_GEOMS.index(g));return FM.predict(T[[s]],(1-G[g]['beta'][[P[s]]]).float(),A,O,'native')[0]
        return f,fc
    run=torch.load(ROOT/'runs'/f'{spec}.pt',map_location='cpu');A=run['A'].float().cuda();O=run['O'].float().cuda();obj=run['spec']['objective']
    def f(g,s):
        if g not in FIT_GEOMS or s>3:return None
        x=run['result'][g]['x'][[s]].cuda();d=(1-G[g]['beta'][[P[s]]]).float()
        if obj=='J':return FM.predict(x,d,A,O,'native')[0]
        if obj=='con_AO':
            O16=torch.zeros(8,16,3,device='cuda').index_add(1,FM.bin_map('cuda'),O);return ('coarse',torch.einsum('na,rac,njrc,roc->najoc',d,A,x,O16)[0])
        return ('coarse',torch.einsum('na,rac,njroc->najoc',d,A,x)[0])
    return f,run

# ------------------------------------------------------------------ teacher / physics / extended fields
def teacher_eval(G,P):
    """Noise-free student-vs-teacher errors (initial and final) for every synthetic same-family recovery control."""
    import fit68
    RG=environment();out={};bm=FM.bin_map('cuda');fp=[P[s] for s in range(4)]
    for rid,v in CFG['runs'].items():
        if v['target']=='phys':continue
        T=fit68.teacher_targets(RG,CFG['teachers'][v['target']]);run=torch.load(ROOT/'runs'/f'{rid}.pt',map_location='cpu');res={}
        for stage in ('initial','final'):
            with torch.no_grad():
                if v['kind']=='net':
                    m=net_model(v['init'] if stage=='initial' else rid);cols=lambda g:FM.network_columns(m,G[g],fp)
                elif stage=='initial':
                    src=net_model(v['init']);cols=lambda g:FM.network_columns(src,G[g],fp)
                else:
                    fc=fcoef(run['model']);cols=lambda g:fc.columns(FIT_GEOMS.index(g))
                per={}
                for g in FIT_GEOMS:
                    Tm,A,O=cols(g);d=(1-G[g]['beta'][fp]).float();y=FM.predict(Tm,d,A,O,'native').double();t=T[g].cuda().double();row={}
                    for sp in SPACES:
                        a=FM.aggregate(y,sp,G[g]['parent'],bm);b=FM.aggregate(t,sp,G[g]['parent'],bm)
                        row[f'abs_{sp}']=((a-b).square().sum((0,1,2,3))/b.square().sum((0,1,2,3))).sqrt().tolist()
                    Hd=lambda z:torch.einsum('ab,nbqoc->naqoc',Hm().cuda(),z/d.double()[:,:,None,None,None])
                    a=Hd(FM.aggregate(y,'coarse',G[g]['parent'],bm));b=Hd(FM.aggregate(t,'coarse',G[g]['parent'],bm))
                    row['con_coarse']=((a-b).square().sum((0,1,2,3))/b.square().sum((0,1,2,3))).sqrt().tolist();per[g]=row
                res[stage]=per
        th=CFG['criteria']['teacher'];fin=res['final'];ini=res['initial']
        passed=all(fin[g]['con_coarse'][c]<=th['con_coarse'] and fin[g]['abs_coarse'][c]<=th['abs_coarse'] and fin[g]['abs_fine16'][c]<=th['abs_fine16'] for g in FIT_GEOMS for c in range(3))
        informative=any(ini[g]['con_coarse'][c]>=th['informative_con_coarse'] or ini[g]['abs_coarse'][c]>=th['informative_abs_coarse'] for g in FIT_GEOMS for c in range(3))
        out[rid]=dict(errors=res,passed=passed,informative=informative,kind=v['kind'],converged=run['converged'],J_final=run['J_final'],seconds=run['seconds'])
    return out

def passivity(m,G,lights):
    from r57_eval import physics
    out={}
    with torch.no_grad():
        for g in ('rest','pt0','cf3'):co=m.coefficients(G[g]);out[g]=physics(m,G[g],lights[:2].cuda(),co);del co
    return out

def extended(m,G,lights):
    """Original WL59 F-path metrics/thresholds (fnto_persist59/evaluate.py) on the historical held-out extended fields:
    never fitted in WL68, but inspected in WL59-65 (not fresh confirmation)."""
    from r57_eval import scores,rel
    from r57_data import aggregate
    preds={};refs={};noise={};rows={}
    with torch.no_grad():
        for g in ('rest','pt0','cf3'):
            co=m.coefficients(G[g]);preds[g]=torch.cat([m.apply(G[g],lights[i:i+4].cuda(),co).cpu() for i in range(0,64,4)]);del co
            a=torch.load(INP/'wl59'/f'targets_{g}_r0.pt')['test'];b=torch.load(INP/'wl59'/f'targets_{g}_r1.pt')['test'];refs[g]=.5*(a+b);noise[g]=(a-b)/2**.5
            mt=scores(preds[g].cuda(),refs[g].cuda(),G[g])
            rows[g]=dict(metrics=mt,static_useful=all(v<=.10 for v in mt['coarse']['mean']) and all(v<=.20 for v in mt['coarse']['p90']) and all(v<=.15 for v in mt['fine16']['mean']))
        par=G['rest']['parent'];td=aggregate((refs['pt0']-refs['rest']).cuda(),par);yd=aggregate((preds['pt0']-preds['rest']).cuda(),par)
        nd=torch.hypot(aggregate(noise['pt0'].cuda(),par),aggregate(noise['rest'].cuda(),par));snr=td.flatten(1,2).norm(dim=1)/nd.flatten(1,2).norm(dim=1).clamp(min=1e-30);dyn=[]
        for c in range(3):
            ids=(snr[:16,c]>=3).nonzero().flatten();xx=td[:16,:,:,c][ids].flatten();yy=yd[:16,:,:,c][ids].flatten()
            er=float((xx-yy).norm()/xx.norm().clamp(min=1e-30));xc=xx-xx.mean();yc=yy-yy.mean();co_=float(xc@yc/(xc.norm()*yc.norm()).clamp(min=1e-30))
            dyn.append(dict(n_fields=len(ids),error=er,corr=co_,slope=float(xc@yc/(xc@xc).clamp(min=1e-30)),passed=bool(len(ids)) and er<=.35 and co_>=.90))
        par=G['pt0']['parent'];yp=aggregate((preds['pt0'][48::2]-preds['pt0'][49::2]).cuda(),par);tp=aggregate((refs['pt0'][48::2]-refs['pt0'][49::2]).cuda(),par)
        pn=aggregate((noise['pt0'][48::2]-noise['pt0'][49::2]).cuda(),par);er=rel(yp,tp);det=tp.flatten(1,2).norm(dim=1)>3*pn.flatten(1,2).norm(dim=1)
        pairs=dict(errors=er.cpu().tolist(),detectable=det.cpu().tolist(),passed=bool(det.any() and ((er<=.25)|~det).all()))
    eligible=rows['rest']['static_useful'] and rows['pt0']['static_useful'] and all(d['passed'] for d in dyn) and pairs['passed']
    return dict(poses=rows,dynamic=dyn,directional_pairs=pairs,F_path_eligible=eligible,note='historical held-out fields; WL59 reference-precision and F/S gates not re-run')

def transfer_oracle(A,O,G,refs,iters):
    """LABELLED ORACLE (uses held-out references): best admissible spatial coefficients (T>=0, sum_j T<=1) for FIXED angular
    bases A/O, fitted on held-out replica half 0 and scored on half 1; per held-out group."""
    res={};bm=FM.bin_map('cuda');A=A.double().cuda();O=O.double().cuda();Hc=Hm().cuda()
    proj=lambda T:FM.project_capped(T.permute(0,2,3,1)).permute(0,3,1,2)
    for pop,keys in CFG['populations']['heldout']['groups'].items():
        subs=[];cgs=[]
        for g,s in keys:
            r=refs[(g,s)];K=r.K;nf=K//2 if K>=6 else 1;fh,eh=list(range(nf)),list(range(nf,K))
            tg=FM.Target(r.S[fh].double().mean(0)[None].cuda(),r.d[None].cuda(),r.parent,bm)
            def f(T,tg=tg):
                v=sum((FM.predict(T,tg.d,A,O,sp,tg.parent,bm)-tg.S[sp]).square().sum((1,2,3))/tg.den[sp] for sp in SPACES)/3
                zc=FM.predict(T,tg.d,A,O,'coarse',tg.parent,bm)/tg.d[:,:,None,None,None]
                return (v+(torch.einsum('ab,nbqoc->naqoc',Hc,zc)-tg.HZ).square().sum((1,2,3))/tg.den['con']).sum()
            x,_=FM.fista(f,torch.full((1,G[g]['P'],8,3),.5/G[g]['P'],dtype=torch.float64,device='cuda'),proj,iters)
            h=FM.predict(x,tg.d,A,O,'native')[0].float();sub=Ref(r.key,r.S[eh],r.B16[eh],r.d,r.parent,bm,r.ball16);Sg,Bg=sub.gpu()
            subs.append(sub);cgs.append(cross(sub,Sg,Bg,h));del Sg,Bg,tg;torch.cuda.empty_cache()
        res[pop]=pooled(subs,cgs,[('con_coarse','con','coarse'),('abs_coarse','abs','coarse'),('abs_fine16','abs','fine16')])
    return res

# ------------------------------------------------------------------ decision and main
def decide(res,teacher):
    """Predeclared outcome mapping (protocol 'decision'); no threshold or population is chosen after results."""
    phys=[r for r,v in CFG['runs'].items() if v['target']=='phys' and v['kind']=='net' and v['group']=='all']
    primary=min(phys,key=lambda r:res[r]['J_final'])
    S_fit=all(res[primary]['fit'][g]['Q']['passed'] for g in FIT_GEOMS)
    groups=list(CFG['populations']['heldout']['groups'])
    S_gen=all(res[primary]['heldout'][p]['Q']['passed'] for p in groups)
    net_t=[t for t,v in teacher.items() if v['kind']=='net'];inf=[t for t in net_t if teacher[t]['informative']]
    adequate=bool(inf) and all(teacher[t]['passed'] for t in inf);conv=res[primary]['converged']
    outcome='A' if S_fit and S_gen else 'B' if S_fit else 'C' if adequate and conv else 'D'
    coef=[r for r,v in CFG['runs'].items() if v['target']=='phys' and v['kind']=='coef'];best=min(coef,key=lambda r:res[r]['J_final'])
    coef_t=[t for t,v in teacher.items() if v['kind']=='coef']
    verdict={'A':'still a viable candidate on the tested physical population','B':'in need of a specific additional falsification (independent generalization)',
             'C':'unsupported for the tested physical requirements (constructive attainment failed with adequate optimizer controls; impossibility not established)',
             'D':'in need of a specific additional falsification (representation vs optimization/reference not separated)'}[outcome]
    return dict(primary_candidate=primary,primary_selection='lowest final training objective among physical net/all runs',S_fit=S_fit,S_gen=S_gen,
                heldout_groups={p:res[primary]['heldout'][p]['Q']['passed'] for p in groups},optimizer_adequate=adequate,informative_teachers=inf,
                primary_converged=conv,outcome=outcome,family_status=verdict,
                coefficient_class=dict(best=best,Q_fit={g:res[best]['fit'][g]['Q']['passed'] for g in FIT_GEOMS},
                                       teacher={t:dict(passed=teacher[t]['passed'],informative=teacher[t]['informative']) for t in coef_t}))

def main():
    RG=environment();torch.backends.cuda.matmul.allow_tf32=False;torch.set_num_threads(8);torch.set_grad_enabled(False)
    out=ROOT/'eval/analysis.json'
    if out.exists():raise FileExistsError('analysis is immutable')
    verify_inputs();t0=time.perf_counter();G=geoms(RG);P=ports();lights=torch.load(INP/'wl59/lights.pt')['test']
    refs=load_refs(G,P);drefs=delta_refs(refs);print('REFS',round(time.perf_counter()-t0),flush=True)
    qual={};prec={}
    groups={'fit':{g:[(g,s) for s in range(4)] for g in FIT_GEOMS}}|{'heldout':{p:[tuple(k) for k in v] for p,v in CFG['populations']['heldout']['groups'].items()}}
    res={};coarse_out={}
    for name,(kind,spec) in candidates().items():
        f,obj=predictor(kind,spec,G,P);cg={};hs={}
        for key,r in refs.items():
            h=f(*key)
            if h is None:continue
            Sg,Bg=r.gpu();cg[key]=cross(r,Sg,Bg,h);del Sg,Bg
            if not isinstance(h,tuple):hs[key]=h
        metrics=COARSE_ONLY if kind=='convex' and CFG['convex'][spec]['objective']!='J' else STD
        row=dict(kind=kind,fit={},heldout={},per_matrix={},per_bin={},energy_ratio={})
        if kind in ('net','coef'):
            run=None if spec.startswith('P_') else torch.load(ROOT/'runs'/f'{spec}.pt',map_location='cpu')
            row.update(J_final=None if run is None else run['J_final'],converged=None if run is None else run['converged'],
                       seconds=None if run is None else run['seconds'],peak_GPU_allocated=None if run is None else run['peak_GPU_allocated'])
        for key in cg:
            p=pooled([refs[key]],[cg[key]],metrics);row['per_matrix'][f'{key[0]}_s{key[1]}']=p
            if metrics is STD:
                row['per_bin'][f'{key[0]}_s{key[1]}']=per_bin(refs[key],cg[key])
                e=refs[key].energy.mean(0);row['energy_ratio'][f'{key[0]}_s{key[1]}']=(cg[key]['energy']/e).tolist()
            if name=='frozen_5901':prec[f'{key[0]}_s{key[1]}']=[c['signal_over_SE'] for c in p['con_coarse']]
            coarse_out[f'{name}|{key[0]}_s{key[1]}']=cg[key]['coarse'].float()
        if name=='frozen_5901':
            th=CFG['gates']['precision_signal_over_SE'];qual={k:min(v)>=th for k,v in prec.items()}
        for part,gr in groups.items():
            for gname,keys in gr.items():
                keys=[k for k in keys if k in cg and qual.get(f'{k[0]}_s{k[1]}',True)]
                if not keys:continue
                p=pooled([refs[k] for k in keys],[cg[k] for k in keys],metrics)
                row[part][gname]=dict(pooled=p,Q=Q(p) if metrics is STD else None,matrices=[f'{k[0]}_s{k[1]}' for k in keys],
                                      corr_slope=corr_slope([refs[k] for k in keys],[cg[k] for k in keys]))
        if metrics is STD and hs:
            row['deformation']={}
            for dk,dr in drefs.items():
                v=CFG['populations']['deformation'][dk[0]];ka,kb=(v['geometry'],dk[1]),('rest',dk[1])
                if ka in hs and kb in hs:
                    Sg,Bg=dr.gpu();row['deformation'][f'{dk[0]}_s{dk[1]}']=dict(pooled=pooled([dr],[c:=cross(dr,Sg,Bg,hs[ka]-hs[kb])],STD[:3]+STD[6:]),corr_slope=corr_slope([dr],[c]))
                    del Sg,Bg
        if kind!='convex' or obj is not None:
            if kind=='net':A,O=torch.softmax(obj.in_logits,0).cpu(),torch.softmax(obj.out_logits,1).cpu()
            elif kind=='coef':A,O=torch.softmax(obj.in_logits,0).cpu(),torch.softmax(obj.out_logits,1).cpu()
            else:A,O=obj['A'].float(),obj['O'].float()
            row['bases']={f'{part}/{gname}':floors([refs[k] for k in keys if qual.get(f'{k[0]}_s{k[1]}',True)],A,O)
                          for part,gr in groups.items() for gname,keys in gr.items() if all(k in refs for k in keys)}
        if kind=='net':
            row['passivity']=passivity(obj,G,lights);row['extended_fields']=extended(obj,G,lights)
        res[name]=row;del hs,cg;torch.cuda.empty_cache();print('EVAL',name,round(time.perf_counter()-t0),flush=True)
    teacher=teacher_eval(G,P);print('TEACHER',round(time.perf_counter()-t0),flush=True)
    dec=decide(res,teacher)
    oracle={}
    for nm in ('frozen_5901','frozen_5902',dec['primary_candidate'],dec['coefficient_class']['best']):
        kind,spec=candidates()[nm];_,obj=predictor(kind,spec,G,P)
        A,O=torch.softmax(obj.in_logits,0).detach().cpu(),torch.softmax(obj.out_logits,1).detach().cpu()
        oracle[nm]=transfer_oracle(A,O,G,refs,int(os.environ.get('R68_DEV_ORACLE_ITERS',CFG['oracle']['iters'])))
        print('ORACLE',nm,round(time.perf_counter()-t0),flush=True)
    save_new(ROOT/'eval/coarse_predictions.pt',coarse_out)
    write_new(out,dict(results=res,teacher=teacher,decision=dec,angular_transfer_oracle=oracle,precision=dict(per_matrix_signal_over_SE=prec,qualified=qual),
        seconds=time.perf_counter()-t0,peak_GPU_allocated=torch.cuda.max_memory_allocated(),inputs_verified=True))
    print('EVAL_DONE',dec['outcome'],round(time.perf_counter()-t0),flush=True)


def reproduce_wl66():
    """Identity gate before any fit: the WL68 estimators on ALL 8 fit replicas reproduce WL66's pooled coarse frozen-model
    statistics (model contrast error E_Mc, F_A, PAOc, relaxed L7) for both WL59 checkpoints."""
    RG=environment();torch.set_grad_enabled(False);out=ROOT/'wl66_reproduction.json'
    if out.exists():raise FileExistsError('immutable')
    G=geoms(RG);fit=json.loads((INP/'wl66_ports.json').read_text())['ports'];P={s['slot']:s['port'] for s in fit};bm=FM.bin_map('cuda');refs={}
    for g in FIT_GEOMS:
        for s in range(4):
            b=torch.load(INP/'fit'/f'{g}_s{s}.pt',map_location='cpu');i=P[s]
            refs[(g,s)]=Ref((g,s),b['S'],b['B16'],(1-G[g]['beta'][i]).float(),G[g]['parent'],bm,ball16(G[g],i,bm));del b
    w66=json.loads((ROOT/'repo/results/fnto_rank66/final/analysis.json').read_text())['pooled'];rows={};ok=True
    for seed in (5901,5902):
        f,m=predictor('net',f'P_{seed}',G,P);cgs=[]
        for k,r in refs.items():Sg,Bg=r.gpu();cgs.append(cross(r,Sg,Bg,f(*k)));del Sg,Bg
        p=pooled(list(refs.values()),cgs,[('con_coarse','con','coarse')]);fl=floors(list(refs.values()),torch.softmax(m.in_logits,0).cpu(),torch.softmax(m.out_logits,1).cpu())
        for c in range(3):
            w=w66[f'coarse/{c}'];ck=f'WL59_P_{seed}/'
            pairs={'E_Mc_rel':(p['con_coarse'][c]['ratio'],w[ck+'E_Mc_rel']['est']),'F_A_rel':(fl[c]['F_A']['rho']**2,w[ck+'F_A_rel']['est']),
                   'PAOc_rel':(fl[c]['PAOc']['rho']**2,w[ck+'PAOc_rel']['est']),'L7_rel':(fl[c]['L7']['rho']**2,w['L7_rel']['est'])}
            for n,(a,b) in pairs.items():
                r_=abs(a-b)/abs(b);rows[f'{seed}/{c}/{n}']=dict(wl68=a,wl66=b,relative_difference=r_);ok&=r_<=1e-4
    write_new(out,dict(rows=rows,passed=bool(ok),tolerance=1e-4,replicas='all 8 (WL66 definition)'))
    if not ok:raise RuntimeError('STOP: WL66 statistics not reproduced')
    print('WL66_REPRODUCED',flush=True)

if __name__=='__main__':
    {'repro':reproduce_wl66,'main':main}[sys.argv[1] if len(sys.argv)>1 else 'main']();sys.stdout.flush();os._exit(0)
