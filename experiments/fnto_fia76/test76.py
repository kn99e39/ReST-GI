"""WL76 focused tests (python test76.py): quadrature energy split, FIA operator contract (positivity, zero input, linearity, column sums
<= 1, rebuild identity) for initial and perturbed parameters, source independence of the learned response, exact slab-symmetry
equivariance of the learned part (I1 up to quadrature noise), objective identity with family68 at rank 8, Stage 4 isolation, decisions."""
import sys,os,json,inspect
import torch
from common76 import CFG,ROOT,R73,R74,INP68,sha,environment
import family68 as FM
import fia76 as FI

RG=environment();torch.backends.cuda.matmul.allow_tf32=False;torch.set_grad_enabled(False)
def load(p,mmap=False):return torch.load(p,map_location='cpu',weights_only=False,mmap=mmap)
def rel(a,b):return float((a-b).double().norm()/b.double().norm().clamp(min=1e-30))
_G={}
def geo(g='rest'):
    if g not in _G:
        import fit71,probes68 as P68
        gs=fit71.geometries(RG);_G[g]=(gs[g],FI.Geo(gs[g],P68.domain(RG,g)))
    return _G[g]
def perturbed(seed=76):
    torch.manual_seed(seed);m=FI.FIA(CFG['candidate']['hidden']).cuda()
    with torch.no_grad():
        for p in m.parameters():p.add_(torch.randn_like(p)*.3)
    return m

def test_quadrature_energy_split():
    """sum_k W + sum_{j,o} SS = 1 - beta_B exactly per cell after the budget rescaling (interior, edge and corner ports)."""
    g,G=geo();q=FI.quad_ports(G,[995,3437,3815,362],0,CFG['candidate']['quadrature_q'])
    for p in (995,3437,3815,362):
        for a in (0,5,15):
            c=p*16+a;s=float(q['W'][2][q['W'][0]==c].sum()+q['SS'][2][q['SS'][0]==c].sum());assert abs(s-(1-float(g['beta'][p,a])))<=1e-9,(p,a,s)

def test_operator_contract():
    """Runtime physics for the initial and a perturbed model on rest: zero, positivity, linearity <= 1e-5, column sums <= 1, rebuild identity."""
    g,G=geo();L=load(INP68/'wl59/lights.pt')['test'][:2].cuda()
    for m in (FI.FIA(CFG['candidate']['hidden']).cuda(),perturbed()):
        rt=FI.Runtime(m,G,0,256);r=FI.physics(rt,g,L,G.dom.albedo);assert r['passed'],r
        rt2=FI.Runtime(m,G,0,256);y1=rt.apply(g,L,G.dom.albedo);assert rel(rt2.apply(g,L,G.dom.albedo),y1)<=1e-6   # GPU index_add order: not bitwise

def test_source_independence():
    """The learned response is a function of the anchor only: identical R for an anchor computed in different batches / for different sources."""
    g,G=geo();m=perturbed();k=torch.tensor([10,500,4001],device='cuda');a1,b1=m.response(G,k);a2,b2=m.response(G,torch.tensor([7,4001,500,10,99],device='cuda'))
    assert torch.allclose(a1[1],a2[2],atol=1e-6) and torch.allclose(a1[2],a2[1],atol=1e-6) and torch.allclose(b1[0],b2[3],atol=1e-6)
    assert 'cells' not in inspect.signature(m.response).parameters

def test_equivariance():
    """Exact slab symmetries: the angle-integrated predicted field of source i vs its image (I1) is at the quadrature-noise level: I1 <= 2 x the
    same statistic between two quadrature seeds of source i itself + 0.01."""
    import sym72 as SY,ports73 as P73
    g,G=geo();m=perturbed();sl=P73.Slab(load(INP68/'wl59/geometry_rest.pt'))
    def field(port,gi):
        q=FI.quad_ports(G,[port],gi,4096);return FI.predict(m,G,q['W'],q['SS'],torch.arange(16,device='cuda')+16*port,G.dom.albedo).sum((0,2))
    for op,perm in zip(SY.ops(True),sl.perms):
        if op not in (('r90',False),('mx',True)):continue
        for i in (995,3437):
            j=int(perm[i]);pt=torch.as_tensor(perm,device='cuda');Fi=field(i,0);noise=rel(field(i,9),Fi);I1=rel(field(j,0)[pt],Fi)
            assert I1<=2*noise+.01,(op,i,I1,noise)

def test_objective_identity():
    """fia76.objective on a dense prediction == family68.objective on the factorised rank-8 prediction (AR_5901, rest, S16-R)."""
    import fit71,fit73,family71 as FW
    from runtime import Transport
    gs=fit71.geometries(RG);ports=[p for _,p in fit73.populations()['S16R']][:4];t=load(R73/'fit_targets73.pt',mmap=True);S=t['S']['r0123']['rest'][:4].float().cuda()
    d=(1-gs['rest']['beta'][ports]).float().cuda();bm=FM.bin_map('cuda');h=Transport().cuda();h.load_state_dict(load(R73/'runs/AR_5901.pt')['model'])
    T,A,O=FW.network_columns(h,gs['rest'],ports);ref=float(FM.objective(T,A,O,FM.Target(S,d,gs['rest']['parent'],bm)))
    Sh=FM.predict(T,d,A,O,'native');tg=FI.target(S,d,gs['rest']['parent'],bm);J,_=FI.objective(Sh,tg,torch.arange(4,device='cuda'));J=float(J)/(4*3)
    assert abs(J-ref)<=1e-3*abs(ref),(J,ref)   # float32: family68 uses the Gram identity, fia76 the direct difference

def test_stage4_isolation():
    """Stage 4 ports (protocol) disjoint from S16-R, WL73 confirmation, WL74 Stage 4, every earlier referenced port and all their images."""
    import fit73,ports73 as P73
    s4={v for v in CFG['stage4']['ports'].values() if v is not None};w=json.loads((R73/'ports.json').read_text());w74=json.loads((R74/'ports4.json').read_text())
    prev=set(p for _,p in fit73.populations()['S16R'])|{v['port'] for v in w['confirmation'].values()}|{v['port'] for v in w74['ports'].values()}
    p71,p72=P73.referenced();prev|=set(p71.values())|set(p72.values());sl=P73.Slab(load(INP68/'wl59/geometry_rest.pt'))
    assert not (s4&set(sl.images(list(prev)))),'Stage 4 leakage';assert len(s4)==len([v for v in CFG['stage4']['ports'].values() if v is not None])
    assert 'fit_targets73.pt' in inspect.getsource(__import__('fit76').problem)

def test_decisions():
    import eval68 as E,eval76 as EV
    MS=[s[0] for s in E.STD]
    def grp(v):return dict(pooled={m:[dict(hi=v,lo=v*.9,rho=v) for _ in range(3)] for m in MS},Q=dict(passed=v<=.1))
    def rec(vals):return dict(groups={g:grp(v) for g,v in vals.items()},physics={'rest':dict(passed=True)},passivity={'te1':dict(passed=True)})
    gs=list(CFG['stage4']['decision']['regression'])+['BND_FAR','BND_NEAR']
    def d4(far,other=.2):
        r={n:rec({g:(.9 if g=='BND_FAR' else .5 if g=='BND_NEAR' else .2) for g in gs}) for n in EV.AR+EV.A75}
        r.update({n:rec({g:(v if g=='BND_FAR' else .5 if g=='BND_NEAR' else other) for g in gs}) for n,v in zip(EV.F76,far)});return EV.decide4(r)['classes'][0]
    assert d4((.4,.45)).startswith('TRANSFERABLE') and d4((.85,.8)).startswith('NO INDEPENDENT') and d4((.7,.75)).startswith('INCONCLUSIVE') and d4((.4,.45),.3).startswith('TRADE-OFF')

if __name__=='__main__':
    T=[test_quadrature_energy_split,test_operator_contract,test_source_independence,test_equivariance,test_objective_identity,test_stage4_isolation,test_decisions];fails=0;res={}
    for t in T:
        try:t();print('PASS',t.__name__,flush=True);res[t.__name__]='PASS'
        except Exception as e:fails+=1;print('FAIL',t.__name__,repr(e),flush=True);res[t.__name__]=f'FAIL {e!r}'
    if os.environ.get('R76_TEST_RECORD'):p=ROOT/'tests/tests.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(res,indent=1))
    print('TESTS',len(T)-fails,'passed',fails,'failed',flush=True);sys.stdout.flush();os._exit(1 if fails else 0)
