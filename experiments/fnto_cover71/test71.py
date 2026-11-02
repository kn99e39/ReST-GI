"""WL71 focused tests on the execution server. python test71.py pre (before references) | post (after bundles/targets, before fits)."""
import sys,os,json,builtins
from pathlib import Path
import numpy as np
import torch
from common71 import CFG,ROOT,INP68,R68,TRAIN_GEOMS,sha,environment,seed
import family68 as FM
import family71 as FW

RG=environment();torch.backends.cuda.matmul.allow_tf32=False
from runtime import Transport
def load(p):return torch.load(p,map_location='cpu',weights_only=False)
def geom(g):return RG.device(load(ROOT/'inputs/geometry_cf2.pt') if g=='cf2' else load(INP68/'wl59'/f'geometry_{g}.pt'))
G={g:geom(g) for g in ('rest','pt0','cf2')}
def ck(spec):
    m=Transport().cuda();m.load_state_dict(load(INP68/'wl59'/f'{spec}.pt' if spec.startswith('P_') else R68/'runs'/f'{spec}.pt')['model']);return m.eval()
def rel(a,b):return float((a-b).double().norm()/b.double().norm())
P_FIT=[2085,2441,1123,2069]

def g64(g):return {k:(v.double() if isinstance(v,torch.Tensor) and v.is_floating_point() else v) for k,v in g.items() if not k.startswith('_')}

def test_batched_columns_equal_wl68():
    """family71.network_columns == family68.network_columns in float64 (values and every parameter gradient <= 1e-10) for a frozen
    and a fitted member at every geometry; float32 values agree to 1e-6 (float32 gradients differ only by summation order)."""
    for spec in ('P_5902','net_all_5902'):
        m=ck(spec);ports=P_FIT+[2148,1025,7,3000]
        with torch.no_grad():
            for g in G:
                for x,y in zip(FW.network_columns(m,G[g],ports),FM.network_columns(m,G[g],ports)):assert rel(x,y)<=1e-6
        m=m.double().train();gg=g64(G['rest'])
        def grads(f):
            for p in m.parameters():p.grad=None
            T,A,O=f(m,gg,ports);w=torch.linspace(0,1,T.numel(),device='cuda',dtype=T.dtype).view_as(T);(T*w).sum().backward()
            return T.detach(),{n:p.grad.clone() for n,p in m.named_parameters() if p.grad is not None}
        Ta,ga=grads(FW.network_columns);Tb,gb=grads(FM.network_columns)
        assert rel(Ta,Tb)<=1e-12 and set(ga)==set(gb) and all(rel(ga[n],gb[n])<=1e-10 for n in ga)

def test_objective_wrapper_identity():
    """fit71.objective (native target held channel-major only) == family68.objective: identical float32 value; float64 gradient <= 1e-10."""
    import fit71
    m=ck('net_all_5902').train();g=G['pt0'];ports=P_FIT;bm=FM.bin_map('cuda')
    with torch.no_grad():T,A,O=FW.network_columns(m,g,ports);y=FM.predict(T,(1-g['beta'][ports]).float(),A,O,'native')
    S=(y*(1+.1*torch.randn_like(y))).clamp(min=0);d=(1-g['beta'][ports]).float()
    t1=FM.Target(S,d,g['parent'],bm);t2=FM.Target(S.clone(),d,g['parent'],bm);del t2.S['native']
    with torch.no_grad():T,A,O=FW.network_columns(m,g,ports);assert float(FM.objective(T,A,O,t1))==float(fit71.objective(T,A,O,t2,False))
    m=m.double();gg=g64(g);S=S.double();d=d.double()
    t1=FM.Target(S,d,g['parent'],bm);t2=FM.Target(S.clone(),d,g['parent'],bm);del t2.S['native']
    T,A,O=FW.network_columns(m,gg,ports);g1=torch.autograd.grad(FM.objective(T,A,O,t1),[m.in_logits,m.edge[-1].weight,m.src_layer.weight])
    T,A,O=FW.network_columns(m,gg,ports);g2=torch.autograd.grad(fit71.objective(T,A,O,t2,False),[m.in_logits,m.edge[-1].weight,m.src_layer.weight])
    assert all(rel(a,b)<=1e-10 for a,b in zip(g1,g2))

def test_parameters_teachers_and_passivity():
    """157,024 parameters; WL68 angular bases installed; TB = P_5902 + those bases; T >= 0, sum < 1, column sums <= 1 (also perturbed)."""
    import fit71
    a,o=fit71.angular();ref=ck('net_all_5902')
    for spec in ('P_5901','P_5902'):
        m=fit71.model_from(spec);assert sum(p.numel() for p in m.parameters())==157024
        assert torch.equal(m.in_logits.cpu(),a) and torch.equal(m.out_logits.cpu(),o)
        base=ck(spec);assert all(torch.equal(x,y) for (n,x),(_,y) in zip(m.state_dict().items(),base.state_dict().items()) if n not in ('in_logits','out_logits'))
    ta=fit71.teacher('TA');assert all(torch.equal(x,y) for x,y in zip(ta.state_dict().values(),ref.state_dict().values()))
    tb=fit71.teacher('TB');assert torch.equal(tb.in_logits.cpu(),a)
    with torch.no_grad():
        for m in (ta,tb,fit71.model_from('P_5901')):
            for p in list(m.parameters())[::3]:p.add_(torch.randn_like(p)*.3)
            for g in G.values():
                T,A,O=FW.network_columns(m,g,P_FIT);assert float(T.min())>=0 and float(T.sum(1).max())<1
                assert float(m.column_sums(g,m.coefficients(g)).max())<=1+1e-6

def test_ports_rule_and_correspondence():
    """Final ports by the geometry rule: distinct, never a previously referenced port, >= 1 cm (same face) from all training and earlier ports;
    slots 0-7 equal WL66/WL68 ports; material correspondence of rest_xyz/face across rest, pt0, cf2."""
    pj=json.loads((ROOT/'ports.json').read_text());P={int(k):v['port'] for k,v in pj['slots'].items()}
    fit=json.loads((INP68/'wl66_ports.json').read_text())['ports'];wl68=json.loads((R68/'heldout/ports.json').read_text())['ports']
    for s in fit+wl68:assert P[s['slot']]==s['port']
    x0=G['rest']['rest_xyz'].cpu().numpy();face=np.rint(G['rest']['node'][:,15].cpu().numpy()*5).astype(int)
    for g in G.values():assert torch.equal(g['rest_xyz'],G['rest']['rest_xyz']) and torch.equal(g['node'][:,15],G['rest']['node'][:,15])
    fin=[P[s] for s in range(20,28)];tr=[P[s] for s in range(16)];old=[P[s] for s in range(8)]
    assert len(set(fin+tr))==24 and not set(fin)&set(old)
    for i in fin:
        for j in tr+[k for k in fin if k!=i]:
            if face[i]==face[j]:assert np.hypot(*(x0[i,:2]-x0[j,:2]))>=.01-1e-9
        assert min(float((1-G[g]['beta'][i]).min()) for g in ('rest','cf2'))>=.5
    for i in [P[s] for s in range(8,16)]:assert min(float((1-G[g]['beta'][i]).min()) for g in ('rest','pt0'))>=.5

def test_seeds_and_packets():
    """Packet seeds distinct and disjoint from WL66 (66xxxxxxx), WL68 (683/688xxxxxx) and smoke (719xxxxxx) ranges."""
    import probes71
    s=[p[-1] for p in probes71.packets()];assert len(set(s))==len(s)==8*2*16*8+8*2*16*4
    assert all(713000000<=x<716000000 for x in s) and seed(2,27,15,3)<716000000

def test_deformation_sign_and_precision():
    """DEF reference = cf2 - rest (independent packets); signal/SE large for a low-noise matrix and small for pure noise."""
    import eval68 as E,eval71 as V
    bm=FM.bin_map('cuda');g=G['rest'];P=g['P'];X=torch.rand(16,P,64,3)*1e-3
    def R(S,gg,key):return E.Ref(key,S,torch.zeros(S.shape[0],16,P,16,3),torch.ones(16)*.9,g['parent'],bm,torch.zeros(16,P,16,3,device='cuda'))
    noise=lambda K,s:torch.randn(K,16,P,64,3)*s
    rr=R(X[None]+noise(4,1e-6),'rest',('rest',20));rc=R(2*X[None]+noise(4,1e-6),'cf2',('cf2',20))
    old=V.CFG['references']['final']['slots'];V.CFG['references']['final']['slots']=[20]
    try:d=V.delta_refs({('rest',20):rr,('cf2',20):rc})[('DEF',20)]
    finally:V.CFG['references']['final']['slots']=old
    assert torch.allclose(d.S,rc.S-rr.S) and rel(d.S.mean(0),X)<=1e-2
    assert min(V.signal_over_SE(rr))>=10 and max(V.signal_over_SE(R(noise(4,1e-3),'rest',('rest',21))))<3

def test_fit_stage_reads_only_allowed():
    """Constructing every fit problem reads only fit_targets71, ports, training geometries, WL59 checkpoints and the WL68 run."""
    import fit71
    seen=[];orig=torch.load
    def spy(f,*a,**k):seen.append(str(f));return orig(f,*a,**k)
    torch.load=spy
    try:
        for rid in ('C_5901','D2_5902','TA_S4'):fit71.Problem(RG,CFG['runs'][rid]);torch.cuda.empty_cache()
    finally:torch.load=orig
    allowed={str(p) for p in fit71.ALLOWED.values()}
    assert seen and all(s in allowed for s in seen),[s for s in seen if s not in allowed]
    assert not any('bundles' in s or '/refs/' in s or 'cf2' in s or 'cf3' in s for s in seen)

def test_fit_targets_identity():
    """fit_targets71: r0123 = mean of replicas 0-3 and r0 = replica 0 of each training slot; S4 part equals WL68 fit_targets exactly."""
    ft=load(ROOT/'fit_targets71.pt');w68=load(INP68/'fit_targets.pt')
    for g in TRAIN_GEOMS:
        assert torch.equal(ft['S']['r0123'][g][:4],w68['S'][g])
        for s in (0,5,9,15):
            f=INP68/'fit'/f'{g}_s{s}.pt' if s<4 else R68/'heldout/bundles'/f'{g}_s{s}.pt' if s<8 else ROOT/'refs/bundles'/f'{g}_s{s}.pt'
            S=load(f)['S'];assert torch.equal(ft['S']['r0123'][g][s],S[[0,1,2,3]].double().mean(0).float()) and torch.equal(ft['S']['r0'][g][s],S[0])
            assert ft['sources'][f'{g}_s{s}']['sha256']==sha(f)

PRE=[test_batched_columns_equal_wl68,test_objective_wrapper_identity,test_parameters_teachers_and_passivity,test_ports_rule_and_correspondence,
     test_seeds_and_packets,test_deformation_sign_and_precision]
POST=[test_fit_targets_identity,test_fit_stage_reads_only_allowed]
if __name__=='__main__':
    ok=0;bad=0
    for t in {'pre':PRE,'post':POST}[sys.argv[1]]:
        try:t();ok+=1;print('PASS',t.__name__,flush=True)
        except Exception as e:
            bad+=1;import traceback;traceback.print_exc();print('FAIL',t.__name__,repr(e),flush=True)
        torch.cuda.empty_cache()
    print('TESTS',ok,'passed',bad,'failed',flush=True);sys.stdout.flush();os._exit(1 if bad else 0)
