"""WL72 implementation gates and synthetic contracts (run before training). python test72.py"""
import sys,os,json,math
import numpy as np
import torch
from common72 import CFG,ROOT,R71,R68,INP68,environment
import family68 as FM
import family71 as FW
import sym72 as SY

RG=environment();torch.backends.cuda.matmul.allow_tf32=False
def load(p):return torch.load(p,map_location='cpu',weights_only=False)
G={g:RG.device(load(INP68/'wl59'/f'geometry_{g}.pt')) for g in ('rest','pt0')}
def rel(a,b):return float((a-b).double().norm()/b.double().norm().clamp(min=1e-30))
def models():
    import fit71,rsk72
    a,o=fit71.angular();torch.manual_seed(7201);b=rsk72.RSK(a,o).cuda().eval();return {'A':fit71.model_from('P_5902').eval(),'B':b}

def test_runtime_contract():
    """Both candidates: same A/O (net_all_5902), zero input -> 0, linear, nonnegative, passive (also perturbed), nonzero source responses,
    complete runtime K (coefficients + apply) consistent with the per-source columns (one-hot factorisation)."""
    import fit71
    a,o=fit71.angular();lights=load(INP68/'wl59/lights.pt')['test'][:2].cuda();from r57_eval import physics
    for k,m in models().items():
        assert torch.equal(m.in_logits.cpu(),a) and torch.equal(m.out_logits.cpu(),o)
        for gn,g in G.items():
            with torch.no_grad():
                co=m.coefficients(g);ph=physics(m,g,lights,co);assert ph['passed'],(k,gn,ph)
                cols=(lambda gg,pp:m.columns(gg,pp)) if k=='B' else (lambda gg,pp:FW.network_columns(m,gg,pp))
                T,A,O=cols(g,[2085,2707]);assert float(T.min())>=0 and float(T.sum((1,2,3)).min())>0 and float(T.sum(1).max())<=1+1e-6
                i=2707;x=torch.zeros(16,g['P'],16,3,device='cuda');x[torch.arange(16),i,torch.arange(16)]=1.
                tot=torch.cat([m.apply(g,x[s:s+4],co) for s in range(0,16,4)])
                ball=torch.stack([torch.sparse.mm(g['B'],x[:,:,:,c].reshape(16,-1).T).T.reshape(16,g['P'],64) for c in range(3)],-1)
                y=FM.predict(T[[1]],(1-g['beta'][[i]]).float(),A,O,'native')[0];assert rel(y,tot-ball)<=1e-5,(k,gn,rel(y,tot-ball))
    torch.manual_seed(72072);m=models()['B']
    with torch.no_grad():
        for p in m.parameters():
            if p.requires_grad:p.add_(torch.randn_like(p)*.5)
        co=m.coefficients(G['pt0']);assert float(m.column_sums(G['pt0'],co).max())<=1+1e-6

def test_symmetry_exact_and_labels():
    """Port permutations exact for all 16 ops; pose images stay in the family; transformed label of a mirror pair equals the
    physical response of the mirrored source within Monte Carlo noise (development data: rest slot 2 -> slot 8 under mx)."""
    x0=G['rest']['rest_xyz'].cpu().numpy()
    for op in SY.ops(True):p=SY.permutation(x0,op);assert sorted(p.tolist())==list(range(len(x0)))
    P71={int(k):v['port'] for k,v in json.loads((R71/'ports.json').read_text())['slots'].items()}
    perm=SY.permutation(x0,('mx',False));assert perm[P71[2]]==P71[8],('mirror pair',P71[2],P71[8],perm[P71[2]])
    a=load(INP68/'fit/rest_s2.pt')['S'];b=load(R71/'refs/bundles/rest_s8.pt')['S']
    Fa=SY.integrated(a.double().mean(0)[None])[0];Fb=SY.integrated(b.double().mean(0)[None])[0];Ft=torch.empty_like(Fa);Ft[torch.as_tensor(perm)]=Fa
    na=SY.integrated(a[:4].double().mean(0)[None])[0]-SY.integrated(a[4:].double().mean(0)[None])[0]
    nb=SY.integrated(b[:4].double().mean(0)[None])[0]-SY.integrated(b[4:].double().mean(0)[None])[0]
    noise=float(((na.square()+nb.square()).sum()/4).sqrt()/Fb.norm())   # SE of the difference of two 8-replica means (from half-split differences)
    d=rel(Ft,Fb);print('   mirror pair rel diff',d,'noise',noise);assert d<=4*noise+1e-9

def test_rsk_relational_contract():
    """RSK: rigid translation of the whole geometry leaves T unchanged (no absolute position), boundary context changes it (edge vs
    centre), far term carries mass beyond 2 cm, geometry change needs no retraining (te2 finite, passive)."""
    import rsk72
    m=models()['B'];g=G['rest'];gt={k:(v.clone() if isinstance(v,torch.Tensor) else v) for k,v in g.items() if not k.startswith('_')}
    sh=torch.tensor([.3,-.2,.1],device='cuda');gt['xyz']=gt['xyz']+sh;gt['rest_xyz']=gt['rest_xyz']+sh
    with torch.no_grad():
        T1,_,_=m.columns(g,[2085,1123]);T2,_,_=m.columns(gt,[2085,1123]);assert rel(T2,T1)<=1e-5,rel(T2,T1)
        st=rsk72.prepare(g);assert float(st['desc'][1123,4])<float(st['desc'][2085,4])   # corner port is closer to a side boundary
        x0=g['xyz'];far=(x0-x0[2085]).norm(dim=1)>.02;assert float(T1[0][far].sum())>0
        te2=RG.device(load(ROOT/'inputs/geometry_te2.pt'));T3,_,_=m.columns(te2,[2085]);assert torch.isfinite(T3).all() and float(T3.sum(1).max())<1

def test_training_reads_only_allowed():
    """fit72 data path (fit71.Problem + symmetry data + model init) never opens confirmation, cf2/cf3, te2 or held-out files."""
    import fit71,fit72
    seen=[];orig=torch.load
    def spy(f,*a,**k):seen.append(str(f));return orig(f,*a,**k)
    torch.load=spy
    try:
        pb=fit71.Problem(RG,dict(pop='S16',replicas=[0,1,2,3],target='phys'));gs=dict(pb.gs)
        S=orig(R71/'fit_targets71.pt',map_location='cpu',weights_only=False)['S']['r0123'];F={g:SY.integrated(S[g]) for g in ('rest','pt0')};del S
        fit72.sym_data(RG,gs,pb.ports,F);fit71.model_from('P_5902');fit71.angular()
    finally:torch.load=orig
    bad=[s for s in seen if any(t in s for t in ('rest_cmp72/refs','bundles','cf2','cf3','te2','heldout'))]
    assert not bad,bad

TESTS=[test_runtime_contract,test_symmetry_exact_and_labels,test_rsk_relational_contract,test_training_reads_only_allowed]
if __name__=='__main__':
    ok=bad=0
    for t in TESTS:
        try:t();ok+=1;print('PASS',t.__name__,flush=True)
        except Exception as e:
            bad+=1;import traceback;traceback.print_exc();print('FAIL',t.__name__,repr(e),flush=True)
        torch.cuda.empty_cache()
    print('TESTS',ok,'passed',bad,'failed',flush=True);sys.stdout.flush();os._exit(1 if bad else 0)
