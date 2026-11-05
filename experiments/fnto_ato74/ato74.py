"""WL74 angular representability oracles (LABELLED ORACLES: per-source T are latent fitted variables, never a learned rule).
  K_scatter[i,j,a,o,c] = d[i,a] sum_r A[r,a,c] T[i,j,r,c] O[r,o,c],  A = softmax(in_logits, r), O = softmax(out_logits, o),
  T >= 0 with sum_j T[i,j,r,c] <= 1 (capped simplex)  =>  positive, linear, sum_{j,o} K <= d <= 1 (passive) for any rank M.
G0: fixed WL68 net_all_5902 A/O (M = 8). G1: shared trainable A/O, M = 8. G2: shared trainable A/O, M = 16 (diagnostic capacity bound).
Objective J (WL68, per source and channel, relative): mean[(native + fine16 + coarse)/3 + coarse contrast].
    python ato74.py basis <arm> <restart> | stage1 | stage2 <basis_id> | decide"""
import sys,os,json,time,math
from pathlib import Path
import torch
from common74 import CFG,ROOT,R73,INP68,R68,write_new,save_new,sha,tensor_sha,environment,dev
import family68 as FM

NB=16;OC=CFG.get('oracle',{})
def load(p,mmap=False):return torch.load(p,map_location='cpu',weights_only=False,mmap=mmap)

# ------------------------------------------------------------------ operator (rank-generic copies of family68 algebra)
def predict_m(T,d,A,O,space,parent=None,bm=None):
    """family68.predict with M taken from T (family68 hard-codes M = 8 in the coarse path)."""
    C=T.shape[-1];M=T.shape[2]
    if space!='native':O=torch.zeros(O.shape[0],NB,C,device=O.device,dtype=O.dtype).index_add(1,bm,O)
    if space=='coarse':T=torch.zeros(T.shape[0],240,M,C,device=T.device,dtype=T.dtype).index_add(1,parent,T)
    return torch.einsum('na,rac,njrc,roc->najoc',d,A,T,O)

class _Shape:
    def __init__(self,s):self.shape=s
def objective_m(T,A,O,tg):
    """family68.objective (same terms and normalisation) for any rank; tg.S['native'] may be released (only its shape is used)."""
    n,P=T.shape[:2];C=T.shape[-1];Sn=tg.S['native'] if 'native' in tg.S else _Shape((n,16,P,64,C))
    O16=torch.zeros(O.shape[0],NB,C,device=O.device,dtype=O.dtype).index_add(1,tg.bm,O)
    nat=FM.sq_error(T,tg.d,A,O,Sn,tg.den['native'],tg.Sc['native'])/tg.den['native']
    f16=FM.sq_error(T,tg.d,A,O16,tg.S['fine16'],tg.den['fine16'],tg.Sc['fine16'])/tg.den['fine16']
    pc=predict_m(T,tg.d,A,O,'coarse',tg.parent,tg.bm);co=(pc-tg.S['coarse']).square().sum((1,2,3))/tg.den['coarse']
    con=(torch.einsum('ab,nbqoc->naqoc',FM.Hmat(T.device,T.dtype),pc/tg.d[:,:,None,None,None])-tg.HZ).square().sum((1,2,3))/tg.den['con']
    return ((nat+f16+co)/3+con).mean()

def AO(a,o):return torch.softmax(a,0),torch.softmax(o,1)
def proj(T):return FM.project_capped(T.permute(0,2,3,1)).permute(0,3,1,2)
def target(S,d,parent,bm):
    tg=FM.Target(S,d,parent,bm);tg.S.pop('native');return tg

def fista_T(tg,A,O,T0,iters):
    """Convex per-source problem for fixed A/O (sources are separable; one shared backtracking step)."""
    f=lambda T:objective_m(T,A,O,tg)*T.shape[0]
    x,h=FM.fista(f,T0,proj,iters);return x.detach(),h

def wl68_logits():
    st=load(R68/'runs/net_all_5902.pt')['model'];return st['in_logits'].clone(),st['out_logits'].clone()

def init_logits(arm,restart):
    """Prospective initialisations: restart 0 = WL68 bases (G2: each WL68 mixture duplicated + N(0, 0.01), i.e. the G0 operator class);
    restarts 1, 2 = N(0, 1) logits from fixed seeds."""
    a,o=wl68_logits();M=CFG['arms'][arm]['rank']
    if arm=='G0':return a,o
    g=torch.Generator().manual_seed(CFG['arms'][arm]['seeds'][restart])
    if restart==0:
        if M==8:return a,o
        k=M//8;return (a.repeat(k,1,1)+.01*torch.randn(M,*a.shape[1:],generator=g)),(o.repeat(k,1,1)+.01*torch.randn(M,*o.shape[1:],generator=g))
    return torch.randn(M,*a.shape[1:],generator=g),torch.randn(M,*o.shape[1:],generator=g)

# ------------------------------------------------------------------ data
def training(RG,dtype=torch.float32):
    """S16-R fitting targets (means of fitting replicas 0-3; WL73 fit_targets73.pt) and model budgets at rest / pt0."""
    import fit71,fit73
    gs=fit71.geometries(RG);pops=fit73.populations();ports=[p for _,p in pops['S16R']];slots=[s for s,_ in pops['S16R']]
    t=load(R73/'fit_targets73.pt',mmap=True);assert t['ports']==ports;bm=FM.bin_map('cuda');tg={}
    for g in ('rest','pt0'):
        S=t['S']['r0123'][g].to(dtype).cuda();d=(1-gs[g]['beta'][ports]).to(dtype);tg[g]=target(S,d,gs[g]['parent'],bm);del S
    torch.cuda.empty_cache();return gs,ports,slots,tg

# ------------------------------------------------------------------ Stage 1: shared bases
def basis(arm,restart):
    """G0: T only. G1/G2: block-coordinate fit of shared logits (Adam) and per-source T (FISTA), then a T polish at the final basis."""
    RG=environment();torch.backends.cuda.matmul.allow_tf32=False;torch.set_num_threads(4);bid=f'{arm}_r{restart}'
    out=ROOT/'bases'/f'{bid}.pt'
    if out.exists():raise FileExistsError('immutable')
    R,ti,ai,lr,pol=OC['rounds'],OC['t_iters'],OC['ao_steps'],OC['ao_lr'],OC['polish_iters']
    if 'R74_DEV_ITERS' in os.environ:assert dev();R,ti,ai,pol=2,int(os.environ['R74_DEV_ITERS']),int(os.environ['R74_DEV_ITERS']),int(os.environ['R74_DEV_ITERS'])
    gs,ports,slots,tg=training(RG,torch.float64);a,o=init_logits(arm,restart);M=a.shape[0]   # amendment 1: float64 (float32 FISTA backtracking hangs at the noise floor)
    a=a.double().cuda().requires_grad_(arm!='G0');o=o.double().cuda().requires_grad_(arm!='G0')
    T={g:torch.full((16,gs[g]['P'],M,3),.5/gs[g]['P'],device='cuda',dtype=torch.float64) for g in tg};hist=[];t0=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    def J(T_):
        A,O=AO(a,o);return sum(objective_m(T_[g],A,O,tg[g]) for g in tg)/len(tg)
    if arm!='G0':
        opt=torch.optim.Adam([a,o],lr=lr)
        for rnd in range(R):
            A,O=(x.detach() for x in AO(a,o))
            for g in tg:T[g],_=fista_T(tg[g],A,O,T[g],ti)
            for _ in range(ai):opt.zero_grad(set_to_none=True);L=J(T);L.backward();opt.step()
            with torch.no_grad():hist.append(dict(round=rnd,J=float(J(T)),seconds=time.perf_counter()-t0))
            print('ROUND',bid,rnd,hist[-1]['J'],flush=True)
    A,O=(x.detach() for x in AO(a,o));pol=pol+(R*ti if arm=='G0' else 0)   # amendment 1: G0 gets the same total T iterations as G1/G2
    for g in tg:T[g],_=fista_T(tg[g],A,O,T[g],pol)
    with torch.no_grad():Jf=float(J(T))
    conv=None
    if arm=='G0':   # optimisation adequacy of the convex T problem: the same budget again from the solution
        T2={g:fista_T(tg[g],A,O,T[g],pol)[0] for g in tg}
        with torch.no_grad():J3=float(J(T2))
        conv=dict(J_polish=Jf,J_3x=J3,rel_change=(Jf-J3)/Jf)
    save_new(out,dict(arm=arm,restart=restart,rank=M,in_logits=a.detach().cpu(),out_logits=o.detach().cpu(),T={g:v.cpu() for g,v in T.items()},ports=ports,slots=slots,
        J_final=Jf,history=hist,convergence_G0=conv,seconds=time.perf_counter()-t0,peak_GPU=torch.cuda.max_memory_allocated(),
        parameters=dict(shared_logits=a.numel()+o.numel(),per_source_T=int(sum(v.numel() for v in T.values())))))
    print('BASIS_DONE',bid,Jf,round(time.perf_counter()-t0),flush=True)

def basis_ids():return [f'{arm}_r{r}' for arm,c in CFG['arms'].items() for r in range(c['restarts'])]

# ------------------------------------------------------------------ scoring (eval68 estimators, unchanged)
def score(E,refs,keys,hs):
    """Per matrix and pooled groups for predictions hs[key] (native, float32) against the held-out replicas in refs[key]."""
    cg={}
    for k in keys:
        r=refs[k];Sg,Bg=r.gpu();cg[k]=E.cross(r,Sg,Bg,hs[k]);del Sg,Bg
    per={f'{a}_s{b}':{m:[c['hi'] for c in v] for m,v in E.pooled([refs[k]],[cg[k]],[E.STD[0],E.STD[1],E.STD[2],E.STD[3],E.STD[5]]).items()} for (a,b) in keys for k in [(a,b)]}
    return cg,per

def pooled(E,refs,cg,keys):
    p=E.pooled([refs[k] for k in keys],[cg[k] for k in keys],E.STD);return dict(pooled=p,Q=E.Q(p),matrices=[f'{a}_s{b}' for a,b in keys])

def stage1():
    """Every basis on the S16-R training sources: T fitted on fitting replicas 0-3 (Stage 1), scored on held-out replicas 4-7."""
    RG=environment();torch.backends.cuda.matmul.allow_tf32=False;torch.set_grad_enabled(False);out=ROOT/'stage1/scores.json'
    if out.exists():raise FileExistsError('immutable')
    import eval73 as E73,eval68 as E
    G=E73.geoms(RG);P=E73.slot_ports();R,_=E73.load_refs(G,P);GR=E73.groups();res={}
    keys={g:[(g,s) for s in [s for s,_ in __import__('fit73').populations()['S16R']]] for g in ('rest','pt0')}
    for bid in basis_ids():
        b=load(ROOT/'bases'/f'{bid}.pt');A,O=(x.float().cuda() for x in AO(b['in_logits'].double(),b['out_logits'].double()));hs={}
        for g in ('rest','pt0'):
            for i,k in enumerate(keys[g]):hs[k]=predict_m(b['T'][g][i:i+1].float().cuda(),(1-G[g]['beta'][[P[k[1]]]]).float(),A,O,'native')[0]
        kk=keys['rest']+keys['pt0'];cg,per=score(E,R,kk,hs)
        res[bid]=dict(J_final=b['J_final'],per_matrix=per,groups={gn:pooled(E,R,cg,GR[gn]) for gn in ('TRAIN_NEW_rest','TRAIN_NEW_pt0','TRAIN_KEEP_rest','TRAIN_KEEP_pt0')},
            passivity=passivity(b,G),seconds=b['seconds'],convergence_G0=b['convergence_G0'],parameters=b['parameters'])
        print('S1',bid,flush=True);torch.cuda.empty_cache()
    write_new(out,json.loads(json.dumps(res,default=float)))

def passivity(b,G):
    """Structural checks of the fitted operator: A, O normalised and positive; T >= 0 and column sums <= 1 (=> sum_{j,o} K <= d)."""
    A,O=AO(b['in_logits'],b['out_logits']);T=torch.cat([v.reshape(-1,*v.shape[-2:]) for v in b['T'].values()])
    return dict(A_colsum_err=float((A.sum(0)-1).abs().max()),O_rowsum_err=float((O.sum(1)-1).abs().max()),A_min=float(A.min()),O_min=float(O.min()),
        T_min=float(min(float(v.min()) for v in b['T'].values())),T_max_colsum=float(max(float(v.sum(1).max()) for v in b['T'].values())),
        passed=bool((A.sum(0)-1).abs().max()<1e-5 and (O.sum(1)-1).abs().max()<1e-5 and A.min()>=0 and O.min()>=0 and min(float(v.min()) for v in b['T'].values())>=0
                    and max(float(v.sum(1).max()) for v in b['T'].values())<=1+1e-5))

# ------------------------------------------------------------------ Stage 2: cross-source representability with frozen bases
def stage2(bid):
    """Frozen basis; WL73 confirmation sources (development): T fitted on replica 0, scored on replicas 1-3 (eval72 oracle split)."""
    RG=environment();torch.backends.cuda.matmul.allow_tf32=False;torch.set_num_threads(4);out=ROOT/'stage2'/f'{bid}.json'
    if out.exists():raise FileExistsError('immutable')
    import eval73 as E73,eval68 as E
    G=E73.geoms(RG);P=E73.slot_ports();GR=E73.groups();bm=FM.bin_map('cuda')
    keys=sorted({k for gn in CFG['stage2']['groups'] for k in GR[gn]})
    b=load(ROOT/'bases'/f'{bid}.pt');A,O=(x.detach().cuda() for x in AO(b['in_logits'].double(),b['out_logits'].double()));M=A.shape[0]
    iters=OC['stage2_iters'] if 'R74_DEV_ITERS' not in os.environ else int(os.environ['R74_DEV_ITERS'])
    if 'R74_DEV_ITERS' in os.environ:assert dev()
    hs={};subs={};conv={};t0=time.perf_counter()
    for g in sorted({k[0] for k in keys}):
        kg=[k for k in keys if k[0]==g]
        for c0 in range(0,len(kg),OC['chunk']):
            ch=kg[c0:c0+OC['chunk']];bs=[load(R73/'refs/bundles'/f'{g}_s{s}.pt') for _,s in ch]
            for (gg,s),bb in zip(ch,bs):assert bb['port']==P[s] and bb['K']==4
            S=torch.stack([bb['S'][0] for bb in bs]).double().cuda();d=torch.stack([(1-G[g]['beta'][P[s]]).double() for _,s in ch])
            tg=target(S,d,G[g]['parent'],bm);del S
            T0=torch.full((len(ch),G[g]['P'],M,3),.5/G[g]['P'],dtype=torch.float64,device='cuda');T,_=fista_T(tg,A,O,T0,iters)
            with torch.no_grad():Ja=float(objective_m(T,A,O,tg))
            T2,_=fista_T(tg,A,O,T,iters)                       # adequacy: double the budget from the solution
            with torch.no_grad():Jb=float(objective_m(T2,A,O,tg))
            conv[f'{g}_{ch[0][1]}-{ch[-1][1]}']=dict(J=Ja,J_2x=Jb,rel_change=(Ja-Jb)/Ja)
            for i,(gg,s) in enumerate(ch):
                hs[(g,s)]=predict_m(T[i:i+1].float(),d[i:i+1].float(),A.float(),O.float(),'native')[0].detach()
                subs[(g,s)]=E.Ref((g,s),bs[i]['S'][1:],bs[i]['B16'][1:],(1-G[g]['beta'][P[s]]).float(),G[g]['parent'],bm,E.ball16(G[g],P[s],bm))
            del tg,T,T2;torch.cuda.empty_cache()
    with torch.no_grad():
        cg,per=score(E,subs,keys,hs);groups={gn:pooled(E,subs,cg,GR[gn]) for gn in CFG['stage2']['groups']}
        groups['BND_DEV']=pooled(E,subs,cg,sorted({k for gn in ('BND_NEAR','BND_FAR','UNS') for k in GR[gn]}))
    write_new(out,json.loads(json.dumps(dict(basis=bid,rank=M,per_matrix=per,groups=groups,convergence=conv,seconds=time.perf_counter()-t0),default=float)))
    print('S2_DONE',bid,round(time.perf_counter()-t0),flush=True)

# ------------------------------------------------------------------ decision (protocol 'decision')
def decide():
    out=ROOT/'decision.json'
    if out.exists():raise FileExistsError('immutable')
    s1=json.loads((ROOT/'stage1/scores.json').read_text());s2={b:json.loads((ROOT/'stage2'/f'{b}.json').read_text()) for b in basis_ids()}
    dc=CFG['decision'];M=('abs_coarse','con_coarse')
    hi=lambda grp,m:max(c['hi'] for c in grp['pooled'][m]);lo=lambda grp,m:max(c['lo'] for c in grp['pooled'][m])
    def g1(b,g):return s1[b]['groups'][g]
    G0='G0_r0';rows={}
    for b in basis_ids():
        tr=all(hi(g1(b,f'TRAIN_NEW_{g}'),'abs_coarse')<=dc['gate_abs_coarse'] and hi(g1(b,f'TRAIN_NEW_{g}'),'con_coarse')<=dc['gate_con_coarse'] for g in ('rest','pt0'))
        rel=all(hi(g1(b,f'TRAIN_NEW_{g}'),m)<=dc['improve_factor']*hi(g1(G0,f'TRAIN_NEW_{g}'),m) for g in ('rest','pt0') for m in M)
        dev_=all(hi(s2[b]['groups']['BND_DEV'],m)<lo(s2[G0]['groups']['BND_DEV'],m) for m in M)
        reg={gn:all(hi(src[gn],m)<=dc['interior_factor']*hi(ref[gn],m) for m in M) for gn,src,ref in
             [(g,s1[b]['groups'],s1[G0]['groups']) for g in ('TRAIN_KEEP_rest','TRAIN_KEEP_pt0')]+[(g,s2[b]['groups'],s2[G0]['groups']) for g in ('INT_DED','REM_NEAR','FP_rest')]}
        valid=s1[b]['passivity']['passed']
        gate=valid and (tr or rel) and dev_ and all(reg.values())
        meets=all(hi(g1(b,f'TRAIN_NEW_{g}'),'abs_coarse')<=.10 and hi(g1(b,f'TRAIN_NEW_{g}'),'abs_fine16')<=.15 and hi(g1(b,f'TRAIN_NEW_{g}'),'con_coarse')<=.25 for g in ('rest','pt0')) and \
              hi(s2[b]['groups']['BND_DEV'],'abs_coarse')<=.10 and hi(s2[b]['groups']['BND_DEV'],'abs_fine16')<=.15 and hi(s2[b]['groups']['BND_DEV'],'con_coarse')<=.25
        rows[b]=dict(valid=valid,train_gate=tr,train_rel20=rel,dev_beyond_uncertainty=dev_,interior_ok=reg,gate=gate,meets_WL59_boundary=meets,J_final=s1[b]['J_final'])
    def arm(a):return [b for b in basis_ids() if b.startswith(a+'_')]
    g1pass=[b for b in arm('G1') if rows[b]['gate']];gate=len(g1pass)>=dc['min_restarts']
    chosen=min(g1pass,key=lambda b:rows[b]['J_final']) if gate else None   # selection on the fitting objective only
    def better(X,Y):   # every restart of X <= 0.8 x every restart of Y on the boundary groups, both metrics
        return all(hi(s1[x]['groups'][f'TRAIN_NEW_{g}'],m)<=dc['improve_factor']*hi(s1[y]['groups'][f'TRAIN_NEW_{g}'],m) for x in X for y in Y for g in ('rest','pt0') for m in M) and \
               all(hi(s2[x]['groups']['BND_DEV'],m)<=dc['improve_factor']*hi(s2[y]['groups']['BND_DEV'],m) for x in X for y in Y for m in M)
    J=lambda a:[rows[b]['J_final'] for b in arm(a)];spread={a:(max(J(a))-min(J(a)))/min(J(a)) for a in ('G1','G2')}
    s1g0=s1[G0]['convergence_G0']['rel_change'];s2conv=max(abs(v['rel_change']) for b in basis_ids() for v in s2[b]['convergence'].values())
    uncertain=max(spread.values())>dc['restart_spread_max'] or abs(s1g0)>dc['convex_change_max'] or s2conv>dc['convex_change_max']
    A_=gate and sum(rows[b]['meets_WL59_boundary'] for b in arm('G1'))>=dc['min_restarts']
    B_=sum(1 for x in arm('G2') if better([x],arm('G1')))>=dc['min_restarts']
    G1imp=sum(1 for x in arm('G1') if better([x],[G0]))>=dc['min_restarts'];G2imp=sum(1 for x in arm('G2') if better([x],[G0]))>=dc['min_restarts']
    cls=(['A: rank-8 refit removes the boundary shortfall'] if A_ else [])+(['B: rank 16 adds substantial capability'] if B_ else [])+ \
        (['C: neither improves sufficiently'] if not (gate or G1imp or G2imp or B_) else [])+(['D: optimisation uncertainty'] if uncertain else [])
    write_new(out,dict(per_basis=rows,G1_restarts_passing=g1pass,stage3_gate=gate,chosen_basis=chosen,G1_better_than_G0=G1imp,G2_better_than_G0=G2imp,
        G2_better_than_G1=B_,restart_spread=spread,G0_convex_change=s1g0,stage2_convex_change_max=s2conv,optimisation_uncertain=uncertain,classes=cls))
    print('DECISION',json.dumps(dict(gate=gate,chosen=chosen,classes=cls)),flush=True)

if __name__=='__main__':
    m=sys.argv[1]
    if m=='basis':basis(sys.argv[2],int(sys.argv[3]))
    elif m=='stage1':stage1()
    elif m=='stage2':stage2(sys.argv[2])
    elif m=='decide':decide()
    sys.stdout.flush();os._exit(0)
