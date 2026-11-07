"""WL79 Stage 1: angular representability of exact symmetry images under the FROZEN rank-16 G2 A/O (LABELLED ORACLES: per-source T are
fitted latent variables, never a learned rule; nothing here trains or scores a network).
  closure  algebraic analysis of A (16 ranks x 16 incoming bins) and O (16 x 64 outgoing bins) under the 8 D4 tangent maps of the bin square
           (rank-assignment closure of A, O and of the joint rank-1 patterns (A_r, O_r) for every (source map, receiver map) pair; descriptive)
  oracle   for the S16-R rest training sources: per-source T fitted (WL74 Stage 2 procedure: float64 FISTA from 0.5/P, ato74.objective_m,
           3,000 iterations, then the same budget again as a convergence check) on the means of fitting replicas 0-3, scored with the eval68
           estimators on held-out replicas 4-7, for the original responses (op id) and for their exact images under the predeclared
           representative operations (protocol stage1.ops). Labels, budgets, zero-collision B16 and ball16 are transformed by the exact maps
           of sym79 (rest: every map exact, verified in Stage 0).
  decide   predeclared gate (protocol stage1.gate).
    python oracle79.py closure | oracle | decide"""
import sys,os,json,time
import numpy as np
import torch
from common79 import CFG,ROOT,R73,R74,R71,R68,INP68,environment,load,write_new,sha
RG=environment();torch.backends.cuda.matmul.allow_tf32=False
import fit71,fit73,family68 as FM,ato74 as AT,eval68 as E,sym72 as SY,sym79 as S9
S1=CFG.get('stage1',{})

def basis64():
    """The frozen G2 basis (WL75 selection) in its stored float64 form; provenance as rank75.basis()."""
    bc=json.loads((S9_root()/'experiments/fnto_rank75/protocol/rank75_v1.json').read_text())['basis']
    f=R74/'bases'/f"{bc['selected']}.pt";assert sha(f)==bc['sha256'];b=load(f);return b['in_logits'].double(),b['out_logits'].double(),dict(id=bc['selected'],sha256=bc['sha256'])
def S9_root():return ROOT/'repo'
def opname(op):return op[0]+('+z' if op[1] else '')
def op_of(name):return (name[:-2],True) if name.endswith('+z') else (name,False)

# ------------------------------------------------------------------ algebraic closure (descriptive)
D4T={'I':((1,0),(0,1)),'R90':((0,-1),(1,0)),'R180':((-1,0),(0,-1)),'R270':((0,1),(-1,0)),'Mt':((1,0),(0,-1)),'Mb':((-1,0),(0,1)),'S':((0,1),(1,0)),'AS':((0,-1),(-1,0))}
def tangent_maps(N):
    L=S9.centres(RG.G,N);out={}
    for k,m in D4T.items():
        Q=torch.eye(3,dtype=torch.float64);Q[:2,:2]=torch.tensor(m,dtype=torch.float64);w=L@Q.T;b=RG.G.local_to_bin(w,N).long()
        assert float((w-L[b]).norm(dim=1).max())<1e-9 and len(set(b.tolist()))==N*N;out[k]=b
    return out
def closure():
    from scipy.optimize import linear_sum_assignment as lsa
    a,o,meta=basis64();A,O=AT.AO(a,o);mi=tangent_maps(4);mo=tangent_maps(8);res=dict(basis=meta,A={},O={},joint={})
    def assign(X,Y):   # X, Y (M, bins): best rank assignment of Y's rows onto X's rows; relative Frobenius error
        C=torch.cdist(X,Y).square().numpy();r,c=lsa(C);return float(np.sqrt(C[r,c].sum())/X.norm().item()),c.tolist()
    for c in range(3):
        for k in D4T:
            Ai=torch.empty_like(A[:,:,c]);Ai[:,mi[k]]=A[:,:,c];res['A'][f'{k}/{c}']=assign(A[:,:,c],Ai)[0]
            Oi=torch.empty_like(O[:,:,c]);Oi[:,mo[k]]=O[:,:,c];res['O'][f'{k}/{c}']=assign(O[:,:,c],Oi)[0]
        for ks in D4T:
            for kr in D4T:
                Ai=torch.empty_like(A[:,:,c]);Ai[:,mi[ks]]=A[:,:,c];Oi=torch.empty_like(O[:,:,c]);Oi[:,mo[kr]]=O[:,:,c]
                X=torch.cat([A[:,:,c]/A[:,:,c].norm(),O[:,:,c]/O[:,:,c].norm()],1);Y=torch.cat([Ai/A[:,:,c].norm(),Oi/O[:,:,c].norm()],1)
                res['joint'][f'{ks},{kr}/{c}']=assign(X,Y)[0]
    nonI=[v for k,v in res['joint'].items() if not k.startswith('I,I/')]
    res['summary']=dict(A_max=max(v for k,v in res['A'].items() if not k.startswith('I/')),O_max=max(v for k,v in res['O'].items() if not k.startswith('I/')),
                        joint_max=max(nonI),joint_median=float(np.median(nonI)),
                        note='0 = the basis is closed under the bin permutation (an exact rank relabelling exists); descriptive only')
    out=ROOT/'stage1/closure.json';out.parent.mkdir(parents=True,exist_ok=True);write_new(out,res);print('CLOSURE',json.dumps(res['summary']),flush=True)

# ------------------------------------------------------------------ oracle
def bundle_path(s):
    if s<4:return INP68/'fit'/f'rest_s{s}.pt'
    if s<8:return R68/'heldout/bundles'/f'rest_s{s}.pt'
    if s<16:return R71/'refs/bundles'/f'rest_s{s}.pt'
    return R73/'refs/bundles'/f'rest_s{s}.pt'

def oracle():
    out=ROOT/'stage1/oracle.json'
    if out.exists():raise FileExistsError('immutable')
    torch.set_num_threads(4);t0=time.perf_counter()
    g=RG.device(fit71.guarded('geometry_rest'));P=g['P'];x0=g['rest_xyz'].double().cpu().numpy();bm=FM.bin_map('cuda');parent=g['parent']
    pops=fit73.populations()['S16R'];slots=[s for s,_ in pops];ports=[p for _,p in pops]
    t=load(R73/'fit_targets73.pt',mmap=True);assert t['ports']==ports;Sfit=t['S']['r0123']['rest']
    a,o,meta=basis64();A,O=(v.cuda() for v in AT.AO(a,o));iters=int(os.environ.get('R79_ORACLE_ITERS',S1['oracle']['iters']))
    if 'R79_ORACLE_ITERS' in os.environ:assert ROOT.name.endswith('_dev')
    ops=[('id',False)]+[op_of(n) for n in S1['ops']];chunk=S1['oracle']['chunk'];res=dict(basis=meta,ops=[opname(op) for op in ops],slots=slots,ports=ports,
        per_matrix={},groups={},convergence={},iters=iters)
    # evaluation replicas 4-7 (never fitted)
    ev=[4,5,6,7];EV={}
    for s,p in zip(slots,ports):
        b=load(bundle_path(s));assert int(b.get('port',p))==p and b['S'].shape[0]==8,(s,p);EV[s]=(b['S'][ev].clone(),b['B16'][ev].clone());del b
    beta=g['beta'].double().cpu()
    # boundary sources (protocol stage1.groups): nearest port with a perpendicular normal within bnd_mm (current = rest positions)
    xx=g['xyz'].double();nn=torch.nn.functional.normalize(g['node'][:,9:12].double(),dim=1);D=torch.cdist(xx,xx)
    r1=(torch.where((nn@nn.T).abs()<.5,D,torch.full_like(D,float('inf'))).min(1).values*1e3).cpu();del D
    bnd={s for s,p in zip(slots,ports) if float(r1[p])<=S1['groups']['bnd_mm']};res['boundary_mm']={s:float(r1[p]) for s,p in zip(slots,ports)}
    for op in ops:
        mp=S9.Map(RG,'rest',op,x0);mp.check();refs=[];cgs=[];keys=[]
        for c0 in range(0,len(slots),chunk):
            ss=slots[c0:c0+chunk];pp=ports[c0:c0+chunk]
            S=torch.stack([mp.response(Sfit[slots.index(s)].double().cuda(),p) for s,p in zip(ss,pp)])
            d=torch.stack([mp.budget(1-beta[p],p) for p in pp]).cuda()
            tg=AT.target(S,d,parent,bm);del S
            T0=torch.full((len(ss),P,A.shape[0],3),.5/P,dtype=torch.float64,device='cuda');T,_=AT.fista_T(tg,A,O,T0,iters)
            with torch.no_grad():Ja=float(AT.objective_m(T,A,O,tg))
            T2,_=AT.fista_T(tg,A,O,T,iters)
            with torch.no_grad():Jb=float(AT.objective_m(T2,A,O,tg))
            res['convergence'][f'{opname(op)}:{ss[0]}-{ss[-1]}']=dict(J=Ja,J_2x=Jb,rel_change=(Ja-Jb)/Ja)
            with torch.no_grad():
                for i,(s,p) in enumerate(zip(ss,pp)):
                    h=AT.predict_m(T[i:i+1].float(),d[i:i+1].float(),A.float(),O.float(),'native')[0]
                    Se,Be=EV[s];Sm=mp.response(Se.cuda(),p).cpu();Bm=mp.fine16(Be.cuda(),p).cpu();b16=mp.fine16(E.ball16(g,p,bm),p)
                    ref=E.Ref(('rest',s,opname(op)),Sm,Bm,d[i].float().cpu(),parent,bm,b16);Sg,Bg=ref.gpu();cg=E.cross(ref,Sg,Bg,h);del Sg,Bg
                    ref.S=None;ref.B16=None;refs.append(ref);cgs.append(cg);keys.append(s)
                    pm=E.pooled([ref],[cg],E.STD);res['per_matrix'][f'{opname(op)}:s{s}']={m:[c['hi'] for c in v] for m,v in pm.items()}
                    res['per_matrix'][f'{opname(op)}:s{s}']['rho']={m:[c['rho'] for c in v] for m,v in pm.items()}
            del tg,T,T2;torch.cuda.empty_cache();print('ORACLE',opname(op),ss[0],ss[-1],Ja,(Ja-Jb)/Ja,round(time.perf_counter()-t0),flush=True)
        with torch.no_grad():
            grp={'ALL':keys,'BND':[s for s in keys if s in bnd],'INT':[s for s in keys if s not in bnd]}
            for gn,gk in grp.items():
                idx=[keys.index(s) for s in gk];pl=E.pooled([refs[i] for i in idx],[cgs[i] for i in idx],E.STD)
                res['groups'][f'{opname(op)}:{gn}']=dict(pooled=pl,slots=gk)
        del refs,cgs;json.dump(res,open(ROOT/'stage1/oracle_partial.json','w'),default=float)
    res['seconds']=time.perf_counter()-t0;write_new(out,json.loads(json.dumps(res,default=float)));print('ORACLE_DONE',round(res['seconds']),flush=True)

# ------------------------------------------------------------------ decision
def decide():
    out=ROOT/'stage1/gate.json'
    if out.exists():raise FileExistsError('immutable')
    r=json.loads((ROOT/'stage1/oracle.json').read_text());gc=S1['gate'];hi=lambda grp,m:max(c['hi'] for c in grp['pooled'][m])
    rows={};fail=[]
    for opn in S1['ops']:
        for gn in gc['groups']:
            o=r['groups'][f'id:{gn}'];t=r['groups'][f'{opn}:{gn}'];row={}
            for m in gc['metrics']:
                ratio=hi(t,m)/max(hi(o,m),1e-30);row[m]=dict(original_hi=hi(o,m),transformed_hi=hi(t,m),ratio=ratio,ok=ratio<=gc['factor'])
                if not row[m]['ok']:fail.append((opn,gn,m))
            for m,th in gc['wl59'].items():
                po=hi(o,m)<=th;pt=hi(t,m)<=th;row[f'wl59_{m}']=dict(original_pass=po,transformed_pass=pt,ok=(pt or not po))
                if po and not pt:fail.append((opn,gn,f'wl59_{m}'))
            rows[f'{opn}:{gn}']=row
    # descriptive per-source ratios (rho, worst channel)
    pm=r['per_matrix'];per={}
    for opn in S1['ops']:
        for s in r['slots']:
            o=pm[f'id:s{s}']['rho'];t=pm[f'{opn}:s{s}']['rho'];per[f'{opn}:s{s}']={m:max(t[m])/max(max(o[m]),1e-30) for m in gc['metrics']}
    over=[k for k,v in per.items() if any(x>gc['factor'] for x in v.values())]
    conv=max(abs(v['rel_change']) for v in r['convergence'].values())
    g0=json.loads((ROOT/'stage0/geometry.json').read_text())['maps'];r0=json.loads((ROOT/'stage0/refs.json').read_text())['summary']
    rest=[v for k,v in g0.items() if k.startswith('rest:')]
    s0=dict(rest_maps=len(rest),rest_all_exact=all(v['exact'] for v in rest),rest_all_commute=all(all(v['commutation'].values()) for v in rest),references_valid=bool(r0['valid']))
    s0['valid']=s0['rest_maps']==16 and s0['rest_all_exact'] and s0['rest_all_commute'] and s0['references_valid']
    passed=not fail and s0['valid']
    cls='STAGE 0 CONTRACT INVALID (stop)' if not s0['valid'] else 'REPRESENTABLE (Stage 2 may proceed)' if passed else 'ANGULAR BASIS LIMIT (stop: brief Stage 1)'
    write_new(out,dict(rows=rows,failures=fail,stage0=s0,passed=passed,per_source_ratio=per,per_source_over_factor=over,max_convergence_change=conv,classification=cls))
    print('STAGE1_GATE',cls,fail[:10],len(over),conv,flush=True)

if __name__=='__main__':
    {'closure':closure,'oracle':oracle,'decide':decide}[sys.argv[1]]();sys.stdout.flush();os._exit(0)
