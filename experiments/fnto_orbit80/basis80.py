"""WL80 Stage 1: orbit-balanced global rank-16 angular basis (DIAGNOSTIC FITS; per-image T are latent fitted variables, never a learned rule).

Fitting population (no new physical label): the S16-R fitting-replica-0-3 means (fit_targets73.pt) at rest, presented under every exact rest
operation (sym72.ops(True), 16) with the WL79 maps (sym79: receiver ports, incoming and outgoing bins, budget d' = transformed d), deduplicated
per source by image port (an operation in the stabiliser of a source gives the same physical source again: the first operation in sym72 order
is kept; two S16-R sources that lie in one orbit keep separate image targets, each source with its own weight, as in S16-R); plus the 16 pt0 originals (identity only: dented maps are not exact). Weighting (predeclared): J = 0.5 x rest term + 0.5 x pt0 term
(the WL74 geometry average); rest term = mean over the 16 sources of the mean over that source's distinct images (orbit multiplicity never
changes a source's weight; boundary / interior balance identical to the original S16-R population: 4 / 12); pt0 term = mean over 16 sources.
Per-image objective = the WL68 objective of ato74.objective_m ((native + fine16 + coarse) / 3 + coarse contrast, relative, channel mean).

Operator / constraints unchanged: K = d A T O, A = softmax over r (rank 16), O = softmax over o, T >= 0 on the capped simplex (family68
projection), 16 / 64 bins. Algorithm (WL74 G1/G2 schedule): rounds x [per-image T: t_iters FISTA steps, warm start | shared logits: ao_steps
Adam steps, lr ao_lr, T fixed], then polish_iters FISTA steps of T at the final basis; float64. The A/O step uses exact sufficient statistics
of the fixed T (objective identity verified by test T2), so the 208 targets need not reside on the GPU; the T step streams chunks.
    python basis80.py prepare | fit <restart> | select"""
import sys,os,json,time,math
import numpy as np
import torch
from common80 import CFG,ROOT,R73,R74,environment,load,save_new,write_new,sha,tensor_sha,dev
RG=environment()
import fit71,fit73,family68 as FM,ato74 as AT,sym72 as SY,sym79 as S9
BC=CFG.get('basis',{});NB=16;C=3

# ------------------------------------------------------------------ fitting population
def prepare():
    """targets80.pt: transformed rest images (float32, CPU) + pt0 originals, budgets, weights, provenance (geometry-only maps)."""
    out=ROOT/'basis/targets80.pt'
    if out.exists():raise FileExistsError('immutable')
    out.parent.mkdir(parents=True,exist_ok=True)
    gs=fit71.geometries(RG);pops=fit73.populations()['S16R'];slots=[s for s,_ in pops];ports=[p for _,p in pops]
    t=load(R73/'fit_targets73.pt',mmap=True);assert t['ports']==ports;Sr=t['S']['r0123']['rest'];Sp=t['S']['r0123']['pt0']
    x0=gs['rest']['rest_xyz'].double().cpu().numpy();beta=gs['rest']['beta'].double().cpu();maps=[S9.Map(RG,'rest',op,x0) for op in SY.ops(True)]
    rows=[];S=[];D=[];seen_all={}
    for k,(s,p) in enumerate(zip(slots,ports)):
        seen=set();imgs=[]
        for mp in maps:
            mp.check();q=int(mp.perm[p])
            if q in seen:continue
            seen.add(q);seen_all.setdefault(q,[]).append(s)   # cross-source coincidences (sources in one orbit) are kept: each source keeps weight 1
            imgs.append((mp,q))
        for mp,q in imgs:
            S.append(mp.response(Sr[k].cuda(),p).float().cpu());D.append(mp.budget(1-beta[p],p).float())
            rows.append(dict(geom='rest',slot=s,source_port=p,op=list(mp.op),image_port=q,weight=.5/len(slots)/len(imgs)))
    bp=gs['pt0']['beta'].double().cpu()
    for k,(s,p) in enumerate(zip(slots,ports)):
        S.append(Sp[k].float().clone());D.append((1-bp[p]).float());rows.append(dict(geom='pt0',slot=s,source_port=p,op=['id',False],image_port=p,weight=.5/len(slots)))
    S=torch.stack(S);D=torch.stack(D);w=torch.tensor([r['weight'] for r in rows],dtype=torch.float64);assert abs(float(w.sum())-1)<1e-12
    shared={q:v for q,v in seen_all.items() if len(v)>1};print('SHARED_ORBIT_IMAGES',len(shared),sorted({tuple(v) for v in shared.values()}),flush=True)
    save_new(out,dict(S=S,d=D,rows=rows,weights=w,shared_images={int(q):v for q,v in shared.items()},source_sha256=sha(R73/'fit_targets73.pt'),layout='S (N, 16, P, 64, RGB) float32 collided fitting means; rows: geometry, source, operation, image port, weight'))
    print('PREPARED',len(rows),'targets',sum(r['geom']=='rest' for r in rows),'rest images',flush=True)

# ------------------------------------------------------------------ per-target objective and its sufficient-statistics form
def obj_vec(T,A,O,tg):
    """Per-target WL68 objective (n,) = ato74.objective_m before the (source, channel) mean (identical terms)."""
    n,P=T.shape[:2];O16=torch.zeros(O.shape[0],NB,C,device=O.device,dtype=O.dtype).index_add(1,tg.bm,O);Sn=AT._Shape((n,16,P,64,C))
    nat=FM.sq_error(T,tg.d,A,O,Sn,tg.den['native'],tg.Sc['native'])/tg.den['native']
    f16=FM.sq_error(T,tg.d,A,O16,tg.S['fine16'],tg.den['fine16'],tg.Sc['fine16'])/tg.den['fine16']
    pc=AT.predict_m(T,tg.d,A,O,'coarse',tg.parent,tg.bm);co=(pc-tg.S['coarse']).square().sum((1,2,3))/tg.den['coarse']
    con=(torch.einsum('ab,nbqoc->naqoc',FM.Hmat(T.device,T.dtype),pc/tg.d[:,:,None,None,None])-tg.HZ).square().sum((1,2,3))/tg.den['con']
    return ((nat+f16+co)/3+con).mean(1)

def stats(T,tg):
    """Exact sufficient statistics of the per-target objective at fixed T (all small)."""
    n,P,M,_=T.shape;Sc=tg.Sc['native'].view(C,n,16,P,64);Tc=torch.zeros(n,240,M,C,device=T.device,dtype=T.dtype).index_add(1,tg.parent,T)
    return dict(Xn=torch.einsum('cnajo,njrc->naroc',Sc,T),Xf=torch.einsum('najfc,njrc->narfc',tg.S['fine16'],T),
                Xc=torch.einsum('naqfc,nqrc->narfc',tg.S['coarse'],Tc),Y=torch.einsum('naqfc,nqrc->narfc',tg.HZ,Tc),
                GT=torch.einsum('njrc,njsc->nrsc',T,T),GTc=torch.einsum('nqrc,nqsc->nrsc',Tc,Tc),d=tg.d.clone(),
                dn=tg.den['native'].clone(),df=tg.den['fine16'].clone(),dc=tg.den['coarse'].clone(),dh=tg.den['con'].clone())

def obj_stats(A,O,st,bm,H):
    O16=torch.zeros(O.shape[0],NB,C,device=O.device,dtype=O.dtype).index_add(1,bm,O);d=st['d'];d2=d*d
    GO=torch.einsum('roc,soc->rsc',O,O);G16=torch.einsum('roc,soc->rsc',O16,O16);HA=torch.einsum('ab,rbc->rac',H,A)
    nat=(torch.einsum('na,rac,sac,nrsc,rsc->nc',d2,A,A,st['GT'],GO)-2*torch.einsum('na,rac,naroc,roc->nc',d,A,st['Xn'],O)+st['dn'])/st['dn']
    f16=(torch.einsum('na,rac,sac,nrsc,rsc->nc',d2,A,A,st['GT'],G16)-2*torch.einsum('na,rac,narfc,rfc->nc',d,A,st['Xf'],O16)+st['df'])/st['df']
    co=(torch.einsum('na,rac,sac,nrsc,rsc->nc',d2,A,A,st['GTc'],G16)-2*torch.einsum('na,rac,narfc,rfc->nc',d,A,st['Xc'],O16)+st['dc'])/st['dc']
    con=(torch.einsum('rac,sac,nrsc,rsc->nc',HA,HA,st['GTc'],G16)-2*torch.einsum('rac,narfc,rfc->nc',HA,st['Y'],O16)+st['dh'])/st['dh']
    return ((nat+f16+co)/3+con).mean(1)

def cat_stats(parts):return {k:torch.cat([p[k] for p in parts]) for k in parts[0]}

# ------------------------------------------------------------------ fit
def init_logits(restart):
    """Restart 0: the frozen G2 basis (WL75 selection, float64 as stored); restarts >= 1: N(0, 1) logits from the predeclared seeds."""
    bc=json.loads((ROOT/'repo/experiments/fnto_rank75/protocol/rank75_v1.json').read_text())['basis'];f=R74/'bases'/f"{bc['selected']}.pt";assert sha(f)==bc['sha256']
    b=load(f);a,o=b['in_logits'].double(),b['out_logits'].double()
    if restart==0:return a,o,dict(init='G2',id=bc['selected'],sha256=bc['sha256'])
    g=torch.Generator().manual_seed(BC['seeds'][restart]);return torch.randn(a.shape,generator=g,dtype=torch.float64),torch.randn(o.shape,generator=g,dtype=torch.float64),dict(init='N(0,1)',seed=BC['seeds'][restart])

def fit(restart):
    restart=int(restart);out=ROOT/'basis'/f'B80_r{restart}.pt'
    if out.exists():raise FileExistsError('immutable')
    torch.backends.cuda.matmul.allow_tf32=False;torch.set_num_threads(4)
    R,ti,ai,lr,pol,ch=BC['rounds'],BC['t_iters'],BC['ao_steps'],BC['ao_lr'],BC['polish_iters'],BC['chunk']
    if 'R80_DEV_ITERS' in os.environ:assert dev();R,ti,ai,pol=2,int(os.environ['R80_DEV_ITERS']),int(os.environ['R80_DEV_ITERS']),int(os.environ['R80_DEV_ITERS'])
    tf=load(ROOT/'basis/targets80.pt',mmap=True);S,D,rows,w=tf['S'],tf['d'],tf['rows'],tf['weights'].cuda();N=len(rows)
    if 'R80_DEV_TARGETS' in os.environ:
        assert dev();keep=[i for i in range(N) if i%int(os.environ['R80_DEV_TARGETS'])==0];S,D,rows=S[keep],D[keep],[rows[i] for i in keep];w=w[keep]/w[keep].sum();N=len(rows)
    gs=fit71.geometries(RG);bm=FM.bin_map('cuda');H=FM.Hmat('cuda',torch.float64);par={g:gs[g]['parent'] for g in ('rest','pt0')}
    chunks=[];cur=[]
    for i,r in enumerate(rows):   # chunks never mix geometries (coarse parents)
        if cur and (len(cur)==ch or rows[cur[0]]['geom']!=r['geom']):chunks.append(cur);cur=[]
        cur.append(i)
    chunks.append(cur)
    a,o,meta=init_logits(restart);a=a.cuda().requires_grad_(True);o=o.cuda().requires_grad_(True);M=a.shape[0];P=S.shape[2]
    T=[torch.full((len(c),P,M,C),.5/P,device='cuda',dtype=torch.float64) for c in chunks]
    def target(c):return AT.target(S[c].cuda().double(),D[c].cuda().double(),par[rows[c[0]]['geom']],bm)
    def tstep(iters):
        A,O=(x.detach() for x in AT.AO(a,o));parts=[]
        for i,c in enumerate(chunks):
            tg=target(c);T[i],_=AT.fista_T(tg,A,O,T[i],iters)
            with torch.no_grad():parts.append(stats(T[i],tg))
            del tg;torch.cuda.empty_cache()
        return cat_stats(parts)
    hist=[];t0=time.perf_counter();torch.cuda.reset_peak_memory_stats();opt=torch.optim.Adam([a,o],lr=lr)
    for rnd in range(R):
        st=tstep(ti)
        with torch.no_grad():Jt=float((obj_stats(*AT.AO(a,o),st,bm,H)*w).sum())
        for _ in range(ai):
            opt.zero_grad(set_to_none=True);L=(obj_stats(*AT.AO(a,o),st,bm,H)*w).sum();L.backward();opt.step()
        with torch.no_grad():Ja=float((obj_stats(*AT.AO(a,o),st,bm,H)*w).sum())
        hist.append(dict(round=rnd,J_after_T=Jt,J_after_AO=Ja,seconds=time.perf_counter()-t0));print('ROUND',restart,rnd,Jt,Ja,round(time.perf_counter()-t0),flush=True)
    A,O=(x.detach() for x in AT.AO(a,o));Jv=torch.empty(N,dtype=torch.float64,device='cuda')
    for i,c in enumerate(chunks):
        tg=target(c);T[i],_=AT.fista_T(tg,A,O,T[i],pol)
        with torch.no_grad():Jv[c]=obj_vec(T[i],A,O,tg)
        del tg;torch.cuda.empty_cache()
    Jf=float((Jv*w).sum());geo=np.array([r['geom'] for r in rows])
    per=dict(rest=float((Jv*w)[torch.as_tensor(geo=='rest',device='cuda')].sum()*2),pt0=float((Jv*w)[torch.as_tensor(geo=='pt0',device='cuda')].sum()*2))
    save_new(out,dict(restart=restart,rank=M,init=meta,in_logits=a.detach().cpu(),out_logits=o.detach().cpu(),J_final=Jf,J_by_geometry=per,J_per_target=Jv.cpu(),
        history=hist,seconds=time.perf_counter()-t0,peak_GPU=torch.cuda.max_memory_allocated(),targets=N,chunk=ch,settings=dict(rounds=R,t_iters=ti,ao_steps=ai,ao_lr=lr,polish_iters=pol)))
    print('BASIS_DONE',restart,Jf,per,round(time.perf_counter()-t0),flush=True)

# ------------------------------------------------------------------ selection and alignment
def select():
    """Predeclared: the restart with the lowest final fitting objective J (fitting replicas only); its mixtures are re-indexed by the optimal
    assignment (Hungarian, joint normalised (A_r, O_r) patterns, channel-summed) onto the G2 mixtures; a common rank permutation of A and O is an
    exact symmetry of the operator (T re-indexed alike), so the basis is unchanged; it only fixes the latent-coordinate identity used by the
    Q78 initialisation rule."""
    from scipy.optimize import linear_sum_assignment as lsa
    out=ROOT/'basis/G1.pt'
    if out.exists():raise FileExistsError('immutable')
    rs=[load(ROOT/'basis'/f'B80_r{r}.pt') for r in range(len(BC['seeds']))];J={r['restart']:r['J_final'] for r in rs};best=min(J,key=J.get);b=rs[best]
    a0,o0,_=init_logits(0);A0,O0=AT.AO(a0,o0);A1,O1=AT.AO(b['in_logits'],b['out_logits'])
    def pat(A,O):return torch.cat([(A/A.norm(dim=1,keepdim=True)).permute(0,2,1).reshape(A.shape[0],-1),(O/O.norm(dim=1,keepdim=True)).permute(0,2,1).reshape(O.shape[0],-1)],1)
    Cm=torch.cdist(pat(A0,O0),pat(A1,O1)).square().numpy();r_,c_=lsa(Cm);perm=torch.as_tensor(c_)[np.argsort(r_)]
    a1=b['in_logits'][perm].clone();o1=b['out_logits'][perm].clone()
    Aa,Oa=AT.AO(a1,o1);assert torch.allclose(Aa,A1[perm]) and torch.allclose(Oa,O1[perm])
    norm=dict(A_colsum_err=float((Aa.sum(0)-1).abs().max()),O_rowsum_err=float((Oa.sum(1)-1).abs().max()),A_min=float(Aa.min()),O_min=float(Oa.min()))
    rel=lambda X,Y:float((X-Y).norm()/Y.norm())
    corr=dict(assignment=perm.tolist(),matched_cost_per_rank=[float(Cm[i,c_[i]]) for i in range(len(r_))],A_rel_change_vs_G2=rel(Aa,A0),O_rel_change_vs_G2=rel(Oa,O0))
    spread=(max(J.values())-min(J.values()))/min(J.values())
    save_new(out,dict(in_logits=a1,out_logits=o1,selected_restart=best,J_by_restart=J,restart_spread=spread,correspondence=corr,normalisation=norm,
        in_logits_sha256=tensor_sha(a1),out_logits_sha256=tensor_sha(o1),source={r['restart']:dict(init=r['init'],J_by_geometry=r['J_by_geometry'],seconds=r['seconds']) for r in rs}))
    write_new(ROOT/'basis/select.json',json.loads(json.dumps(dict(selected_restart=best,J_by_restart=J,restart_spread=spread,correspondence=corr,normalisation=norm,
        G1_sha256=sha(out),in_logits_sha256=tensor_sha(a1),out_logits_sha256=tensor_sha(o1),
        histories={r['restart']:r['history'] for r in rs},J_by_geometry={r['restart']:r['J_by_geometry'] for r in rs}),default=float)))
    print('SELECTED',best,J,spread,flush=True)

if __name__=='__main__':
    {'prepare':prepare,'fit':fit,'select':select}[sys.argv[1]](*sys.argv[2:]);sys.stdout.flush();os._exit(0)
