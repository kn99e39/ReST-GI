"""WL77 focused tests (python test77.py): the WL76 tests on the FIA-AZ model, plus bin azimuths, per-pair normalisation / positivity of the
azimuthal lobe, synthetic azimuthal expressivity vs ring lobes, and full native D4 x z-flip equivariance with exact bin permutations. WL76 list: quadrature energy split, FIA operator contract (positivity, zero input, linearity, column sums
<= 1, rebuild identity) for initial and perturbed parameters, source independence of the learned response, exact slab-symmetry
equivariance of the learned part (I1 up to quadrature noise), objective identity with family68 at rank 8, Stage 4 isolation, decisions."""
import sys,os,json,inspect
import torch
from common77 import CFG,ROOT,R73,R74,INP68,sha,environment
import family68 as FM
import fia77 as FI

RG=environment();torch.backends.cuda.matmul.allow_tf32=False;torch.set_grad_enabled(False)
def load(p,mmap=False):return torch.load(p,map_location='cpu',weights_only=False,mmap=mmap)
def rel(a,b):return float((a-b).double().norm()/b.double().norm().clamp(min=1e-30))
_G={}
def geo(g='rest'):
    if g not in _G:
        import fit71,probes68 as P68
        gs=fit71.geometries(RG);_G[g]=(gs[g],FI.Geo(gs[g],P68.domain(RG,g)))
    return _G[g]
def perturbed(seed=77):
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
        rt=FI.Runtime(m,G,0,256,keep_reference=True);r=FI.physics(rt,g,L,G.dom.albedo);assert r['passed'],r
        assert rel(rt.apply(g,L,G.dom.albedo),rt.apply_reference(g,L,G.dom.albedo))<=1e-5,'CSR apply == reference scatter apply (amendment 5)'
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
    """fia77.objective on a dense prediction == family68.objective on the factorised rank-8 prediction (AR_5901, rest, S16-R)."""
    import fit71,fit73,family71 as FW
    from runtime import Transport
    gs=fit71.geometries(RG);ports=[p for _,p in fit73.populations()['S16R']][:4];t=load(R73/'fit_targets73.pt',mmap=True);S=t['S']['r0123']['rest'][:4].float().cuda()
    d=(1-gs['rest']['beta'][ports]).float().cuda();bm=FM.bin_map('cuda');h=Transport().cuda();h.load_state_dict(load(R73/'runs/AR_5901.pt')['model'])
    T,A,O=FW.network_columns(h,gs['rest'],ports);ref=float(FM.objective(T,A,O,FM.Target(S,d,gs['rest']['parent'],bm)))
    Sh=FM.predict(T,d,A,O,'native');tg=FI.target(S,d,gs['rest']['parent'],bm);J,_=FI.objective(Sh,tg,torch.arange(4,device='cuda'));J=float(J)/(4*3)
    assert abs(J-ref)<=1e-3*abs(ref),(J,ref)   # float32: family68 uses the Gram identity, fia77 the direct difference

def test_stage4_isolation():
    """Stage 4 ports (protocol) disjoint from S16-R, WL73 confirmation, WL74 Stage 4, every earlier referenced port and all their images."""
    import fit73,ports73 as P73
    s4={v for v in CFG['stage4']['ports'].values() if v is not None};w=json.loads((R73/'ports.json').read_text());w74=json.loads((R74/'ports4.json').read_text())
    prev=set(p for _,p in fit73.populations()['S16R'])|{v['port'] for v in w['confirmation'].values()}|{v['port'] for v in w74['ports'].values()}
    p71,p72=P73.referenced();prev|=set(p71.values())|set(p72.values());sl=P73.Slab(load(INP68/'wl59/geometry_rest.pt'))
    assert not (s4&set(sl.images(list(prev)))),'Stage 4 leakage';assert len(s4)==len([v for v in CFG['stage4']['ports'].values() if v is not None])
    assert 'fit_targets73.pt' in inspect.getsource(__import__('fit77').problem)

def test_decisions():
    import eval68 as E,eval77 as EV
    MS=[s[0] for s in E.STD]
    def grp(v):return dict(pooled={m:[dict(hi=v,lo=v*.9,rho=v) for _ in range(3)] for m in MS},Q=dict(passed=v<=.1))
    def rec(vals):return dict(groups={g:grp(v) for g,v in vals.items()},physics={'rest':dict(passed=True)},passivity={'te1':dict(passed=True)})
    gs=list(CFG['stage4']['decision']['regression'])+['BND_FAR','BND_NEAR']
    def d4(far,other=.2):
        r={n:rec({g:(.9 if g=='BND_FAR' else .5 if g=='BND_NEAR' else .2) for g in gs}) for n in EV.AR+EV.A75}
        r.update({n:rec({g:(v if g=='BND_FAR' else .5 if g=='BND_NEAR' else other) for g in gs}) for n,v in zip(EV.F76,far)});return EV.decide4(r)['classes'][0]
    assert d4((.4,.45)).startswith('TRANSFERABLE') and d4((.85,.8)).startswith('NO INDEPENDENT') and d4((.7,.75)).startswith('INCONCLUSIVE') and d4((.4,.45),.3).startswith('TRADE-OFF')


# ------------------------------------------------------------------ WL77 angular-representation tests
def az_model(seed=77):
    """Perturbed FIA-AZ with a strongly weighted, strongly anisotropic azimuthal lobe."""
    m=perturbed(seed)
    with torch.no_grad():
        b=m.fA[-1].bias.view(FI.LOBES+3,FI.C);b[FI.LOBES]+=4.;b[FI.LOBES+1:]+=3.   # alpha_az -> ~1, a1, a2 -> ~0.5 tanh(3)
    return m

def test_bin_azimuth():
    """Bin-centre directions of the concentric 8x8 grid map back to their own bin (local_to_bin) and phi_o = atan2(y, x) of that centre."""
    import r57_geometry as R
    phi,st=FI.bin_azimuth();b=torch.arange(64);u=((b%8).double()+.5)/8;v=((b//8).double()+.5)/8;x,y=R.G.square_to_disk(u,v);z=torch.sqrt((1-x*x-y*y).clamp(min=0))
    assert torch.equal(R.G.local_to_bin(torch.stack([x,y,z],1),8),b) and float((torch.atan2(y,x).float()-phi).abs().max())<1e-6

def test_normalisation_positivity():
    """Per pair: sum over the 64 bins of the outgoing distribution = rho (1 - tau) D (every component normalised), all bin values >= 0."""
    g,G=geo();m=az_model();k=torch.tensor([3,777,4001,9000],device='cuda');loc,_=m.response(G,k);B=m.basis()
    O=torch.einsum('anqc,qoc->anoc',loc,B);assert float(O.min())>=-1e-7,'negative outgoing flux'
    rho=torch.sigmoid(m.rho(G.afeat[k]));tau=torch.sigmoid(m.tau(G.afeat[k]));D=O.sum(2)/(rho*(1-tau))[:,None]
    assert float((D.sum(1)-1).abs().max())<1e-4,'local spatial distribution not normalised'
    az=(loc[:,:,FI.LOBES:]).abs().sum()/loc.abs().sum();assert float(az)>.3,'azimuthal lobe inactive in the test model'

def test_azimuthal_expressivity_synthetic():
    """A strongly azimuthal synthetic outgoing distribution nu(1 + 0.9 cos(phi_o - 0.7)) is fitted by the FIA-AZ lobe form (residual < 0.05)
    but not by ring-uniform lobes (residual > 0.3): the representation relaxes the WL76 restriction."""
    phi,_=FI.bin_azimuth();phi=phi.double();ring=torch.nn.functional.one_hot(FI.RING,4).double();nu=ring@torch.tensor([.1,.2,.3,.4],dtype=torch.float64)/FI.RCOUNT.double()[FI.RING]
    y=nu*(1+.9*torch.cos(phi-.7));y=y/y.sum()
    def lsq(B):Q,_=torch.linalg.qr(B);return float((y-Q@(Q.T@y)).norm()/y.norm())
    d=phi-.7;Baz=torch.cat([ring,ring*torch.cos(d)[:,None],ring*torch.cos(2*d)[:,None]],1)
    assert lsq(Baz)<.05 and lsq(ring)>.3,(lsq(Baz),lsq(ring))

def bin_perm(G,op,N):
    """For each port p and its image q under the slab operation, the bin map o -> o' (N x N grid) from the actual port frames; returns
    (P, N*N) long and a bool 'exact' (every port's map is a bijection)."""
    import r57_geometry as R,sym72 as SY
    t,b=FI.port_frames(G);n=G.n;M=torch.as_tensor(SY.matrix(op),dtype=torch.float32,device='cuda');sl=torch.as_tensor(__import__('ports73').Slab(load(INP68/'wl59/geometry_rest.pt')).perms[[o for o in SY.ops(True)].index(op)],device='cuda')
    k=torch.arange(N*N);u=((k%N).double()+.5)/N;v=((k//N).double()+.5)/N;x,y=R.G.square_to_disk(u,v);z=torch.sqrt((1-x*x-y*y).clamp(min=0));L=torch.stack([x,y,z],1).float().cuda()
    w=L[None,:,0,None]*t[:,None]+L[None,:,1,None]*b[:,None]+L[None,:,2,None]*n[:,None]          # (P, N*N, 3) world directions
    w=w@M.T;q=sl;lq=torch.stack([(w*t[q][:,None]).sum(-1),(w*b[q][:,None]).sum(-1),(w*n[q][:,None]).sum(-1)],-1)
    mp=R.G.local_to_bin(lq,N);exact=all(len(set(r.tolist()))==N*N for r in mp.cpu());return mp,sl,exact

def test_native_equivariance():
    """Full native D4 x z-flip equivariance where the bins permute exactly: K(s(i))[a', s(j), o'] vs K(i)[a, j, o] (strongly azimuthal model)
    within 2 x the quadrature-seed noise of K(i) + 0.01."""
    import sym72 as SY
    g,G=geo();m=az_model();FI.attach_frames(G)
    def K(port,gi):q=FI.quad_ports(G,[port],gi,4096);return FI.predict(m,G,q['W'],q['SS'],torch.arange(16,device='cuda')+16*port,G.dom.albedo)
    for op in (('r90',False),('mx',True)):
        mo,sl,ex64=bin_perm(G,op,8);mi,_,ex16=bin_perm(G,op,4);assert ex64 and ex16,('bins do not permute',op)
        for i in (995,3437):
            j=int(sl[i]);Ki=K(i,0);Kn=K(i,9);Kj=K(j,0)
            out=torch.zeros_like(Kj);out[mi[i][:,None,None],sl[None,:,None],mo[None]]=Ki   # K_i[a,j,o] -> (a', s(j), o')
            I=rel(Kj,out);noise=rel(Kn,Ki);assert I<=2*noise+.01,(op,i,I,noise)

if __name__=='__main__':
    T=[test_quadrature_energy_split,test_operator_contract,test_source_independence,test_equivariance,test_objective_identity,test_stage4_isolation,test_decisions,
       test_bin_azimuth,test_normalisation_positivity,test_azimuthal_expressivity_synthetic,test_native_equivariance];fails=0;res={}
    for t in T:
        try:t();print('PASS',t.__name__,flush=True);res[t.__name__]='PASS'
        except Exception as e:fails+=1;print('FAIL',t.__name__,repr(e),flush=True);res[t.__name__]=f'FAIL {e!r}'
    if os.environ.get('R77_TEST_RECORD'):p=ROOT/'tests/tests.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(res,indent=1))
    print('TESTS',len(T)-fails,'passed',fails,'failed',flush=True);sys.stdout.flush();os._exit(1 if fails else 0)
