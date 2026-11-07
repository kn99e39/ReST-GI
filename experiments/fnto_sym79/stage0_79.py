"""WL79 Stage 0: physical symmetry contract and its verification (archived data only; no model is trained or scored).
  geometry <out.json>  for rest (D4h, 16), pt0 (D4 -> pt0 / pt0v images) and te0 (D4, self): port maps, frame maps, bin-map exactness and
                       bijectivity, aggregation commutation (fine16, coarse cells), refined-mesh identity of op(G) and the image geometry,
                       beta equivariance (quadrature noise), W+UV-Q (Q78) input equivariance (node and pair blocks; probe-tie ports counted)
  refs <out.json>      reference controls on archived, already-scored bundles:
     POSITIVE  exact pairs: two referenced sources q = perm[p] in the same rest geometry under each of the 15 non-identity operations
               (the first CAP = 2 pairs per operation, sorted by port), and every self-fixed source (q = p; replica halves). The transformed
               reference of p is compared with q. (Dented poses have no exact bin map: geometry contract.)
     NEGATIVE  on the same pairs (i) port map with identity bin maps, (ii) the bin maps composed with an extra t <-> b swap in the image frame
               (a wrong angular map with the right port map); (iii) class-2 translation pairs (exact op + in-face translation) whose
               boundary context differs (nearest perpendicular face <= 5 mm vs >= 15 mm), WL78 correspondence rule.
  Predeclared reading (written before the run): a positive control PASSES when, in the 0-5 mm and 5-10 mm strata around the image source,
  the native and fine16-contrast relative differences of the replica means are <= 2 x their combined replica-noise level (worst channel);
  a negative control DETECTS when the 0-5 mm native or fine16-contrast difference is >= 3 x noise (worst channel). Negative controls are
  RULED on boundary sources only (nearest perpendicular face <= 10 mm: an interior response is locally rotation-symmetric, so a wrong in-plane
  bin map can be physically undetectable there; interior rows are reported); the identity-bin control is not applicable where the source's
  tangent map is the identity. The contract is VALID when every positive control passes and every ruled negative control detects; otherwise
  Stage 0 reports the failure (brief stop condition).
    python stage0_79.py geometry <out.json> | refs <out.json>"""
import sys,json,math,time
import numpy as np
import torch
from common79 import ROOT,R72,R73,INP68,environment,load,bundles
RG=environment();torch.set_grad_enabled(False)
import sym72 as SY,family68 as FM,q78 as Q,sym79 as S9

def geometry(name):
    if name in ('rest','pt0','cf3'):return RG.device(load(INP68/'wl59'/f'geometry_{name}.pt'))
    if name.startswith('pt0v'):return RG.device(load(R72/'geom'/f'{name}.pt'))
    if name=='te0':return RG.device(load(R73/'inputs/geometry_te0.pt'))
    raise KeyError(name)
_G={}
def G(name):
    if name not in _G:_G[name]=geometry(name)
    return _G[name]
REST=G('rest');X0=REST['rest_xyz'].double().cpu().numpy();P=REST['P'];BM=FM.bin_map('cpu');FACE=np.rint(REST['node'][:,15].cpu().numpy()*5).astype(int)
SPACES={'rest':SY.ops(True),'pt0':SY.ops(False),'te0':SY.ops(False)};CAP=2
def opname(op):return op[0]+('+z' if op[1] else '')

# ------------------------------------------------------------------ geometry contract
def q78_equivariance(mp):
    """node16[perm p] vs the frame map applied to the vector blocks of node16[p] (scalars equal); pair features on mapped edges."""
    a=Q.QG(RG,G(mp.src));b=Q.QG(RG,G(mp.dst));na=a['node'][:,:16].double().cpu();nb=b['node'][:,:16].double().cpu();exp=na.clone()
    for s0 in (0,6,10):exp[:,s0:s0+3]=torch.einsum('pij,pj->pi',mp.Q,na[:,s0:s0+3])
    dn=(nb[mp.perm]-exp).abs().max(1).values;info=a['_q78_info']
    E=a['src'].cpu()*P+a['dst'].cpu();Eb=b['src'].cpu()*P+b['dst'].cpu();key={int(k):i for i,k in enumerate(Eb.tolist())}
    img=(mp.perm[a['src'].cpu()]*P+mp.perm[a['dst'].cpu()]).tolist();idx=[key.get(int(k),-1) for k in img];ok=torch.tensor(idx)>=0
    pa=a['pair'].double().cpu();pb=b['pair'].double().cpu();pe=pa.clone();pe[:,:3]=torch.einsum('eij,ej->ei',mp.Q[a['src'].cpu()],pa[:,:3])
    dp=(pb[torch.tensor(idx).clamp(min=0)]-pe).abs().max(1).values[ok]
    return dict(node_max_diff=float(dn.max()),node_ports_exact_1e5=int((dn<=1e-5).sum()),node_ports=P,node_ports_over_1e3=int((dn>1e-3).sum()),
                probe1_tie_ports_src=info['ties1'],edges_mapped_fraction=float(ok.double().mean()),pair_max_diff_mapped=float(dp.max()) if len(dp) else None)

def geometry_contract(out):
    res={}
    for src,ops in SPACES.items():
        for op in ops:
            mp=S9.Map(RG,src,op,X0);gs=G(src);gd=G(mp.dst);row=dict(dst=mp.dst,exact=mp.exact,residual={str(k):v for k,v in mp.residual.items()},
                bijective={str(k):v for k,v in mp.bijective.items()},normal_err=mp.normal_err,tangent_err=mp.tangent_err)
            row['tangent_type_by_face']={int(f):sorted({mp.tangent_type(p) for p in np.nonzero(FACE==f)[0]}) for f in range(6)}
            row['commutation']={k:v for k,v in S9.commutation(mp,gs['parent'],BM).items() if k!='coarse_perm'}
            row['mesh']=S9.mesh_check(RG,src,mp.dst,mp.M)
            bs=gs['beta'].double().cpu();bd=gd['beta'].double().cpu()
            bmap=torch.stack([mp.budget(1-bs[p],p) for p in range(P)]);bdd=1-bd[mp.perm]
            row['beta']=dict(max_abs_diff_d=float((bmap-bdd).abs().max()),rel_norm=float((bmap-bdd).norm()/bdd.norm()),
                             identity_op_diff=None)
            row['q78_inputs']=q78_equivariance(mp)
            res[f'{src}:{opname(op)}']=row;print('GEOM',src,opname(op),mp.dst,mp.exact,row['mesh'],row['commutation'],row['beta']['rel_norm'],
                                                  row['q78_inputs']['node_max_diff'],flush=True)
            json.dump(res,open(out,'w'),indent=1)
    summary=dict(all_exact=all(r['exact'] for r in res.values()),all_commute=all(all(r['commutation'].values()) for r in res.values()),
                 mesh_identical=all(r['mesh']['vertex_max_dist_m']<1e-9 and r['mesh']['triangle_match_fraction']==1.0 for r in res.values()),
                 non_exact=[k for k,r in res.items() if not r['exact']],non_identical_mesh=[k for k,r in res.items() if not (r['mesh']['vertex_max_dist_m']<1e-9 and r['mesh']['triangle_match_fraction']==1.0)])
    json.dump(dict(maps=res,summary=summary),open(out,'w'),indent=1);print('GEOMETRY_DONE',json.dumps(summary),flush=True)

# ------------------------------------------------------------------ reference controls
EDGES=[0,5,10,20,40,1e9]
def strata(q):
    r=np.linalg.norm(X0-X0[q],axis=1)*1e3;return [(f'{a}-{b if b<1e8 else "inf"}mm',torch.as_tensor((r>=a)&(r<b))) for a,b in zip(EDGES[:-1],EDGES[1:])]
def contrast(Y,d,parent):
    Yf=FM.aggregate(Y,'fine16',parent.cpu(),BM);return (Yf-Yf.mean(-4,keepdim=True))/d.view(*([1]*(Yf.dim()-4)),16,1,1,1)
def nrm(X,mask):return X[...,mask,:,:].square().sum(dim=tuple(i for i in range(X.dim()) if i!=X.dim()-1)).sqrt()
def compare(Sq,Sm,dq,parent,mask=None,Kq=None,Km=None):
    """Replica-mean comparison of the image source's reference Sq (K,16,P,64,3) with a mapped tensor Sm (K',...) per stratum."""
    A=Sq.mean(0);B=Sm.mean(0);vA=Sq.var(0)/Sq.shape[0];vB=Sm.var(0)/Sm.shape[0];cA=contrast(Sq,dq,parent);cB=contrast(Sm,dq,parent)
    cAm=cA.mean(0);cBm=cB.mean(0);cv=cA.var(0)/cA.shape[0]+cB.var(0)/cB.shape[0];out={}
    for name,m in strata_q:
        mk=m if mask is None else m&mask
        if mk.sum()==0:continue
        a=nrm(A,mk);ca=nrm(cAm,mk)
        out[name]=dict(n=int(mk.sum()),native_diff=(nrm(A-B,mk)/a).tolist(),native_noise=((vA+vB)[...,mk,:,:].sum(dim=(0,1,2)).sqrt()/a).tolist(),
                       con_diff=(nrm(cAm-cBm,mk)/ca).tolist(),con_noise=(cv[...,mk,:,:].sum(dim=(0,1,2)).sqrt()/ca).tolist())
    return out
def ratio(st,k):
    return max(d/max(n,1e-30) for d,n in zip(st[f'{k}_diff'],st[f'{k}_noise']))

def refs(out):
    B=bundles();ref={}
    for b in B:ref.setdefault((b['geom'],b['port']),b)          # first bundle per (geometry, port)
    res=dict(bundles=len(B),positive=[],negative=[],translation=[]);t0=time.perf_counter()
    x=REST['xyz'].double();nn=torch.nn.functional.normalize(REST['node'][:,9:12].double(),dim=1)
    D=torch.cdist(x,x);perp=(nn@nn.T).abs()<.5;r1=(torch.where(perp,D,torch.full_like(D,float('inf'))).min(1).values*1e3).cpu().numpy();del D
    def S_of(b):return load(b['path'])['S'].double()
    global strata_q
    # ---- exact pairs and self-fixed sources under self-symmetries
    for gname in ('rest',):     # dented poses: bin maps are not exact (geometry contract), so no exact transformed label exists there
        g=G(gname);ports={p for (gg,p) in ref if gg==gname}
        for op in SPACES[gname]:
            if op==('id',False):continue
            mp=S9.Map(RG,gname,op,X0)
            if mp.dst!=gname:continue
            npair=0
            for p in sorted(ports):
                q=int(mp.perm[p])
                if q not in ports:continue
                if q<p and int(mp.perm[q])==p and q!=p:continue     # unordered pair seen once (involutions)
                if q!=p:
                    npair+=1
                    if npair>CAP:continue                            # bounded: the first CAP pairs per operation (sorted by port)
                bp,bq=ref[(gname,p)],ref[(gname,q)];Sp=S_of(bp);dq=1-g['beta'][q].double().cpu();strata_q=strata(q)
                if q==p:     # self-fixed: replica halves
                    K=Sp.shape[0];h=K//2;Sq=Sp[:h];Sm=mp.response(Sp[h:],p);kind='self_fixed'
                else:Sq=S_of(bq);Sm=mp.response(Sp,p);kind='pair'
                row=dict(geom=gname,op=opname(op),p=p,q=q,slot_p=bp['slot'],slot_q=bq['slot'],kind=kind,K=[int(Sq.shape[0]),int(Sm.shape[0])],
                         face_p=int(FACE[p]),face_q=int(FACE[q]),tangent=mp.tangent_type(p),strata=compare(Sq,Sm,dq,g['parent']))
                st=[row['strata'][s] for s in ('0-5mm','5-10mm') if s in row['strata']]
                row['passed']=all(ratio(s,'native')<=2 and ratio(s,'con')<=2 for s in st)
                res['positive'].append(row);print('POS',gname,opname(op),p,q,kind,row['passed'],[round(ratio(s,'native'),2) for s in st],[round(ratio(s,'con'),2) for s in st],flush=True)
                # negative controls on the same pair (image frame t <-> b swap; identity bin maps)
                for nm,post in (('swap_tb',[[0,1,0],[1,0,0],[0,0,1]]),):
                    mn=S9.Map(RG,gname,op,X0,post=post);Sn,mk=mn.response_partial(Sp if kind=='pair' else Sp[Sp.shape[0]//2:],p)
                    n0=compare(Sq,Sn,dq,g['parent'],mk);s5=n0.get('0-5mm')
                    res['negative'].append(dict(control=nm,geom=gname,op=opname(op),p=p,q=q,boundary_mm=float(r1[p]),ruled=bool(r1[p]<=10),strata=n0,
                                                detected=bool(s5 and (ratio(s5,'native')>=3 or ratio(s5,'con')>=3))))
                Sid=torch.zeros_like(Sm);Sx=Sp if kind=='pair' else Sp[Sp.shape[0]//2:];Sid[...,mp.perm,:,:]=Sx
                n1=compare(Sq,Sid,dq,g['parent']);s5=n1.get('0-5mm')
                res['negative'].append(dict(control='identity_bins',geom=gname,op=opname(op),p=p,q=q,boundary_mm=float(r1[p]),
                                            ruled=bool(r1[p]<=10 and mp.tangent_type(p)!=(1,0,0,1)),strata=n1,detected=bool(s5 and (ratio(s5,'native')>=3 or ratio(s5,'con')>=3))))
                print('NEG',gname,opname(op),p,q,[(n['control'],n['detected']) for n in res['negative'][-2:]],flush=True)
                del Sp,Sq,Sm;json.dump(res,open(out,'w'),indent=1)
    # ---- class-2 translation pairs with different boundary context (rest only)
    rp=sorted({p for (gg,p) in ref if gg=='rest'});pairs=[]
    from scipy.spatial import cKDTree
    tree=cKDTree(X0)
    for i,p in enumerate(rp):
        for q in rp:
            if q==p or FACE[p]==FACE[q] and False:continue
            if not ((r1[p]<=5 and r1[q]>=15) or (r1[q]<=5 and r1[p]>=15)):continue
            for op in SPACES['rest']:
                M=SY.matrix(op);d=X0[q]-M@X0[p];nq=REST['node'][q,9:12].double().cpu().numpy()
                if (np.abs(d)>1e-9).sum()==1 and abs(d@nq)<1e-9:pairs.append((ref[('rest',p)]['slot'],ref[('rest',q)]['slot'],p,q,op,d));break
    pairs.sort(key=lambda t:(t[0],t[1]));sel=pairs[:3]
    for sp,sq,p,q,op,d in sel:
        mp=S9.Map(RG,'rest',op,X0,delta=d);Sp=S_of(ref[('rest',p)]);Sq=S_of(ref[('rest',q)]);strata_q=strata(q)
        Sm,mk=mp.response_partial(Sp,p);dq=1-REST['beta'][q].double().cpu();row=dict(p=p,q=q,slot_p=sp,slot_q=sq,op=opname(op),delta_mm=(d*1e3).round(3).tolist(),
            boundary_mm=[float(r1[p]),float(r1[q])],strata=compare(Sq,Sm,dq,REST['parent'],mk))
        s5=row['strata'].get('0-5mm');row['detected']=bool(s5 and (ratio(s5,'native')>=3 or ratio(s5,'con')>=3));res['translation'].append(row)
        print('TRANS',sp,sq,opname(op),row['delta_mm'],row['boundary_mm'],row['detected'],flush=True);json.dump(res,open(out,'w'),indent=1)
    res['translation_candidates']=len(pairs)
    res['summary']=dict(positive_n=len(res['positive']),positive_passed=sum(r['passed'] for r in res['positive']),
        negative_n=len(res['negative'])+len(res['translation']),negative_detected=sum(r['detected'] for r in res['negative']+res['translation']),
        negative_ruled=sum(1 for r in res['negative'] if r['ruled'])+len(res['translation']),
        negative_ruled_detected=sum(1 for r in res['negative'] if r['ruled'] and r['detected'])+sum(r['detected'] for r in res['translation']),
        valid=bool(res['positive']) and all(r['passed'] for r in res['positive']) and all(r['detected'] for r in res['negative'] if r['ruled'])
              and all(r['detected'] for r in res['translation']),
        seconds=time.perf_counter()-t0)
    json.dump(res,open(out,'w'),indent=1);print('REFS_DONE',json.dumps(res['summary']),flush=True)

if __name__=='__main__':
    {'geometry':geometry_contract,'refs':refs}[sys.argv[1]](sys.argv[2])
    sys.stdout.flush();import os;os._exit(0)
