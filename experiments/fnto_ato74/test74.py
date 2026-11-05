"""WL74 focused tests (python test74.py): rank-generic algebra == family68 at rank 8; rank-16 duplication contains the G0 class; A/O
normalisation and positivity; zero input / linearity; passivity of the oracle operator; capped-simplex projection; global basis sharing;
G0 / G1-restart-0 identity with the WL68 bases; replica independence of every fit/score split; training / confirmation isolation."""
import sys,os,json
import torch
from common74 import CFG,ROOT,R73,R68,INP68,environment
import family68 as FM
import ato74 as AT

RG=environment();torch.backends.cuda.matmul.allow_tf32=False
def load(p,mmap=False):return torch.load(p,map_location='cpu',weights_only=False,mmap=mmap)
def rel(a,b):return float((a-b).double().norm()/b.double().norm().clamp(min=1e-30))

def small_target():
    gs,ports,slots,tg=AT.training(RG,torch.float64);return gs,ports,tg

def test_algebra_and_rank16_duplication():
    """objective_m / predict_m == family68 at M = 8 (float64); duplicated rank-16 bases + duplicated T reproduce the rank-8 operator exactly."""
    gs,ports,tg=small_target();a,o=AT.wl68_logits();A,O=(x.double().cuda() for x in AT.AO(a,o));g=torch.Generator(device='cuda').manual_seed(74)
    for name,t in tg.items():
        n=t.d.shape[0];T=AT.proj(torch.rand(n,gs[name]['P'],8,3,device='cuda',dtype=torch.float64,generator=g)*(2./gs[name]['P']))
        t.S['native']=None;t.S.pop('native')
        j1=float(AT.objective_m(T,A,O,t));tt=type('X',(),{})();Sfull=load(R73/'fit_targets73.pt',mmap=True)['S']['r0123'][name].double().cuda()
        ref=FM.Target(Sfull,t.d,gs[name]['parent'],t.bm);j2=float(FM.objective(T,A,O,ref));assert abs(j1-j2)<=1e-9*abs(j2),(name,j1,j2);del ref,Sfull
        for sp in ('native','fine16','coarse'):assert rel(AT.predict_m(T,t.d,A,O,sp,t.parent,t.bm),FM.predict(T,t.d,A,O,sp,t.parent,t.bm))<=1e-12
        Ad,Od=(x.cuda() for x in AT.AO(a.double(),o.double()));A16,O16=(x.cuda() for x in AT.AO(a.repeat(2,1,1).double(),o.repeat(2,1,1).double()));T16=T.repeat(1,1,2,1)   # softmax in float64 (float32 rounding differs by copy count)
        assert rel(AT.predict_m(T16,t.d,A16,O16,'native'),AT.predict_m(T,t.d,Ad,Od,'native'))<=1e-12,'rank-16 duplication'
        break

def test_basis_normalisation_positivity_passivity_linearity():
    """A columns and O rows sum to 1, nonnegative, for random logits of rank 8 and 16; projected T is capped; then sum_{j,o} K <= d,
    K(0) = 0 and K is linear in T (oracle operator contract)."""
    g=torch.Generator().manual_seed(7401)
    for M in (8,16):
        a,o=torch.randn(M,16,3,generator=g)*3,torch.randn(M,64,3,generator=g)*3;A,O=AT.AO(a,o)
        assert float((A.sum(0)-1).abs().max())<1e-6 and float((O.sum(1)-1).abs().max())<1e-6 and float(A.min())>=0 and float(O.min())>=0
        T=AT.proj(torch.rand(2,300,M,3,generator=g)*.02);assert float(T.min())>=0 and float(T.sum(1).max())<=1+1e-6
        d=torch.rand(2,16,generator=g);K=AT.predict_m(T,d,A,O,'native')
        assert float(K.sum((2,3)).sub(d[:,:,None]).max())<=1e-6,'passivity'
        assert float(AT.predict_m(torch.zeros_like(T),d,A,O,'native').abs().max())==0.
        assert rel(AT.predict_m(2*T+T,d,A,O,'native'),3*K)<=1e-6

def test_basis_identities_and_sharing():
    """G0 and G1 restart 0 start from the WL68 net_all_5902 bases (bitwise); G2 restart 0 = duplicated WL68 + small noise; one logit pair
    is shared by every source and geometry (no per-source angular parameter)."""
    a,o=AT.wl68_logits();a0,o0=AT.init_logits('G0',0);a1,o1=AT.init_logits('G1',0);a2,o2=AT.init_logits('G2',0)
    assert torch.equal(a0,a) and torch.equal(o0,o) and torch.equal(a1,a) and torch.equal(o1,o)
    assert a2.shape==(16,16,3) and o2.shape==(16,64,3) and float((a2[:8]-a).abs().max())<.1 and float((a2[8:]-a).abs().max())<.1
    st=load(R68/'runs/net_all_5902.pt')['model'];assert st['in_logits'].shape==(8,16,3) and st['out_logits'].shape==(8,64,3)

def test_isolation_and_replica_independence():
    """Bases are fitted only on fit_targets73 (fitting replicas 0-3 of S16-R); Stage 1 scores replicas 4-7; Stage 2 fits replica 0 and
    scores 1-3; S16-R, WL73 confirmation and Stage 4 ports are pairwise disjoint, Stage 4 ports are not images of any previous port."""
    import fit73,ports73 as P73
    t=load(R73/'fit_targets73.pt',mmap=True);assert 'replica 0-3' in t['layout'] or 'replicas 0-3' in t['layout']
    import inspect,eval73
    assert 'ev=[4,5,6,7]' in inspect.getsource(eval73.load_refs)
    src=inspect.getsource(AT.stage2);assert "bb['S'][0]" in src and "bs[i]['S'][1:]" in src
    w=json.loads((R73/'ports.json').read_text());S16R={p for _,p in fit73.populations()['S16R']};conf={v['port'] for v in w['confirmation'].values()}
    s4=set(CFG['frozen']['stage4_ports'].values());assert not (S16R&conf) and not (s4&conf) and not (s4&S16R)
    sl=P73.Slab(load(INP68/'wl59/geometry_rest.pt'));p71,p72=P73.referenced();prev=set(p71.values())|set(p72.values())|S16R|conf
    assert not (s4&set(sl.images(list(prev)))),'Stage 4 leakage'

if __name__=='__main__':
    T=[test_algebra_and_rank16_duplication,test_basis_normalisation_positivity_passivity_linearity,test_basis_identities_and_sharing,test_isolation_and_replica_independence];fails=0
    for t in T:
        try:t();print('PASS',t.__name__,flush=True)
        except Exception as e:fails+=1;print('FAIL',t.__name__,repr(e),flush=True)
    print('TESTS',len(T)-fails,'passed',fails,'failed',flush=True);sys.stdout.flush();os._exit(1 if fails else 0)
