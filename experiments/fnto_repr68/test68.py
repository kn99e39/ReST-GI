"""WL68 focused tests (run on the execution server before evidence). python test68.py"""
import sys,os,json,math,builtins
from pathlib import Path
import numpy as np
import torch
from common68 import CFG,ROOT,INP,FIT_GEOMS,sha,environment,verify_sources,verify_inputs
import family68 as FM

RG=environment();torch.backends.cuda.matmul.allow_tf32=False;torch.set_grad_enabled(False)
from runtime import Transport
P_FIT=[s['port'] for s in sorted(json.loads((INP/'wl66_ports.json').read_text())['ports'],key=lambda s:s['slot'])]
def geom(g):return RG.device(torch.load(INP/'wl59'/f'geometry_{g}.pt',map_location='cpu'))
def ck(seed):m=Transport().cuda();m.load_state_dict(torch.load(INP/'wl59'/f'P_{seed}.pt',map_location='cpu')['model']);return m.eval()
G={g:geom(g) for g in FIT_GEOMS}
def rel(a,b):return float((a-b).double().norm()/b.double().norm())

def test_sources_and_inputs():
    verify_sources();m=verify_inputs();assert len(m['files'])>=30
    assert torch.equal(FM.bin_map(),torch.as_tensor(RG.G.bin_parent(8,4)))

def test_parameter_constraints():
    """A column-stochastic, O row-stochastic, T >= 0 and sum_j T = mass in (0,1); passivity column sums <= 1 (also for perturbed weights)."""
    for m in (ck(5901),ck(5902)):
        with torch.no_grad():
            for p in list(m.parameters())[::3]:p.add_(torch.randn_like(p)*.3)
        for g in FIT_GEOMS:
            T,A,O=FM.network_columns(m,G[g],P_FIT)
            assert torch.allclose(A.sum(0),torch.ones(16,3,device='cuda'),atol=1e-6) and torch.allclose(O.sum(1),torch.ones(8,3,device='cuda'),atol=1e-6)
            assert float(T.min())>=0 and float(T.sum(1).max())<1 and float(T.sum(1).min())>0
            co=m.coefficients(G[g]);assert float(m.column_sums(G[g],co).max())<=1+1e-6

def test_factorisation_identity():
    """network_columns+predict == Transport.apply(one-hot) - B x (the WL66 factorisation) for both checkpoints, all fit ports."""
    for seed in (5901,5902):
        m=ck(seed)
        for g in FIT_GEOMS:
            co=m.coefficients(G[g]);T,A,O=FM.network_columns(m,G[g],P_FIT);d=(1-G[g]['beta'][P_FIT]).float()
            for n,i in enumerate(P_FIT):
                x=torch.zeros(16,G[g]['P'],16,3,device='cuda');x[torch.arange(16),i,torch.arange(16)]=1.
                tot=torch.cat([m.apply(G[g],x[k:k+4],co) for k in range(0,16,4)])
                ball=torch.stack([torch.sparse.mm(G[g]['B'],x[:,:,:,c].reshape(16,-1).T).T.reshape(16,G[g]['P'],64) for c in range(3)],-1)
                assert rel(FM.predict(T[[n]],d[[n]],A,O,'native')[0],tot-ball)<=1e-5

def test_frozen_replay():
    """WL59 stored pt0 actions (fields 8,9,62,63) and WL66 frozen coarse probe outputs are reproduced on this machine."""
    lights=torch.load(INP/'wl59/lights.pt')['test'];fp=torch.load(INP/'wl66_frozen_coarse.pt');bm=FM.bin_map('cuda')
    for seed in (5901,5902):
        m=ck(seed);co=m.coefficients(G['pt0']);st=torch.load(INP/'wl59'/f'replay_{seed}.pt')['F']
        y=torch.cat([m.apply(G['pt0'],lights[[f]].cuda(),co).cpu() for f in (8,9,62,63)]);assert rel(y,st)<=2e-5
        for g in FIT_GEOMS:
            T,A,O=FM.network_columns(m,G[g],P_FIT);d=(1-G[g]['beta'][P_FIT]).float();yc=FM.predict(T,d,A,O,'coarse',G[g]['parent'],bm).double().cpu()
            for s in range(4):assert rel(yc[s],fp[f'WL59_P_{seed}/{g}/s{s}'])<=1e-5

def test_split_and_reference_identity():
    """fit_targets = mean of replicas 0-3 exactly; eval replicas 4-7 disjoint; seeds = WL66 formula; ballistic/collided disjoint tags."""
    ft=torch.load(INP/'fit_targets.pt');fr,er=CFG['populations']['fit']['fit_replicas'],CFG['populations']['fit']['eval_replicas']
    assert set(fr).isdisjoint(er) and sorted(fr+er)==list(range(8)) and ft['replicas']==fr and ft['ports']==P_FIT
    for gi,g in enumerate(FIT_GEOMS):
        for s in range(4):
            b=torch.load(INP/'fit'/f'{g}_s{s}.pt');assert torch.equal(b['S'][fr].double().mean(0).float(),ft['S'][g][s])
            assert b['seeds']==[680000000+gi*100000+s*10000+a*100+k for k in range(8) for a in range(16)]
    for f in (INP/'cross').glob('*.pt'):
        r=torch.load(f,weights_only=False);t=r['component_first']
        assert float((t['T']-t['B']-t['Scat']).abs().max())<=1e-12*float(t['T'].abs().sum())

def test_projection_and_fista():
    """Capped-simplex projection is the Euclidean projection (KKT); FISTA reaches a bounded-least-squares optimum."""
    from scipy.optimize import lsq_linear
    g=torch.Generator().manual_seed(1);v=torch.randn(50,40,generator=g,dtype=torch.float64)*.2;w=FM.project_capped(v)
    assert float(w.min())>=0 and float(w.sum(-1).max())<=1+1e-12
    for k in range(50):
        if float(v[k].clamp(min=0).sum())<=1:assert torch.allclose(w[k],v[k].clamp(min=0))
        else:
            th=float((v[k]-w[k])[w[k]>0].mean());assert torch.allclose((v[k]-w[k])[w[k]>0],torch.full_like(w[k][w[k]>0],th),atol=1e-10)
    X=torch.rand(60,12,generator=g,dtype=torch.float64);y=torch.rand(60,generator=g,dtype=torch.float64)
    ref=lsq_linear(X.numpy(),y.numpy(),bounds=(0,np.inf)).x;f=lambda t:(X@t[0]-y).square().sum()
    x,_=FM.fista(f,torch.zeros(1,12,dtype=torch.float64),lambda t:t.clamp(min=0),3000)
    assert float(f(x))<=float(((X.numpy()@ref-y.numpy())**2).sum())*(1+1e-6)+1e-12

def test_synthetic_representability():
    """A same-family coefficient-level target is exactly recovered by the coefficient optimizer from another member."""
    torch.set_grad_enabled(True);g=G['rest'];bm=FM.bin_map('cuda');gen=torch.Generator().manual_seed(3)
    with torch.no_grad():
        T,_,_=FM.network_columns(ck(5902),g,P_FIT[:1]);A=torch.softmax(torch.randn(8,16,3,generator=gen),0).cuda();O=torch.softmax(torch.randn(8,64,3,generator=gen),1).cuda()
        d=(1-g['beta'][P_FIT[:1]]).float();tg=FM.Target(FM.predict(T,d,A,O,'native'),d,g['parent'],bm)
        T0,A0,O0=FM.network_columns(ck(5901),g,P_FIT[:1]);ml,sl=FM.FCoef.logits_from(T0)
    fc=FM.FCoef(torch.log(A0),torch.log(O0),[ml],[sl]).cuda();opt=torch.optim.Adam(fc.parameters(),lr=1e-2);J0=None
    for it in range(3000):
        opt.zero_grad();J=FM.objective(*fc.columns(0),tg);J.backward();opt.step();J0=J0 or float(J)
    assert float(J)<=1e-3*J0,(J0,float(J))
    torch.set_grad_enabled(False)

def test_objective_identity():
    """Gram-identity objective equals the materialised objective (float64) and its gradient."""
    torch.set_grad_enabled(True);g=G['rest'];bm=FM.bin_map('cuda');gen=torch.Generator(device='cuda').manual_seed(7)
    T1,A1,O1=FM.network_columns(ck(5901),g,P_FIT);d=(1-g['beta'][P_FIT]).double()
    syn=FM.predict(T1.double(),d,A1.double(),O1.double(),'native');syn=syn*(1+.2*torch.rand(syn.shape,generator=gen,device='cuda',dtype=torch.float64))
    m=ck(5902);T,A,O=[x.double() for x in FM.network_columns(m,g,P_FIT)]
    tg=FM.Target(syn,d,g['parent'],bm);T=T.detach().requires_grad_(True)
    a=FM.objective(T,A,O,tg);b=FM.objective_direct(T,A,O,tg);ga,=torch.autograd.grad(a,T);gb,=torch.autograd.grad(b,T)
    torch.set_grad_enabled(False);assert abs(float(a-b))<=1e-9*abs(float(b)) and rel(ga,gb)<=1e-7,(float(a),float(b),rel(ga,gb))

def test_estimator_unbiased():
    """Cross-replica error/signal are unbiased and the contrast map is WL66's H D^-1 (synthetic replicas)."""
    import eval68 as EV
    gen=torch.Generator().manual_seed(5);truth=torch.rand(16,120,3,generator=gen,dtype=torch.float64);h=truth+.05*torch.randn(16,120,3,generator=gen,dtype=torch.float64)
    d=torch.rand(16,generator=gen,dtype=torch.float64)*.5+.5;acc=[]
    for t in range(300):
        R=truth[None]+.3*torch.randn(4,16,120,3,generator=gen,dtype=torch.float64)
        RR=EV.grams(R,R);hR=EV.grams(R,h[None])[:,0];hh=EV.grams(h[None],h[None])[0,0]
        class Rf:pass
        r=Rf();r.RR={'s':RR};cg={'hh':{'s':hh},'hR':{'s':hR}};L=EV.Lmap('con',d);acc.append(EV.error(r,cg,L,'s',list(range(4))))
    Ht=torch.einsum('ab,bqc->aqc',EV.Hm()@torch.diag(1/d),h-truth);true=Ht.square().sum((0,1))
    assert torch.allclose(torch.stack(acc).mean(0),true,rtol=.05)

def test_no_heldout_leakage():
    """The fit stage refuses held-out/eval paths and, in a short real run, opens only the declared fitting inputs."""
    import fit68
    for bad in ('heldout/bundles/cf3_s4.pt','fit/rest_s0.pt','eval/analysis.json'):
        try:fit68.guarded(bad);raise AssertionError(bad)
        except AssertionError as e:
            if str(e)==bad:raise
    opened=[];orig=torch.load
    def spy(p,*a,**k):opened.append(str(p));return orig(p,*a,**k)
    torch.load=spy
    try:
        pb=fit68.Problem(RG);m=fit68.model_from('P_5901');torch.set_grad_enabled(True)
        J=pb.loss(lambda g:FM.network_columns(m,pb.gs[g],pb.ports));J.backward()
    finally:torch.load=orig;torch.set_grad_enabled(False)
    assert opened and all(any(p.endswith(a) for a in fit68.ALLOWED) for p in opened),opened
    assert not any('heldout' in p or '/fit/' in p for p in opened)

if __name__=='__main__':
    names=[n for n in dir() if n.startswith('test_')];fails=0
    for n in sys.argv[1:] or names:
        try:globals()[n]();print('PASS',n,flush=True)
        except Exception as e:
            import traceback;fails+=1;print('FAIL',n,repr(e),flush=True);traceback.print_exc()
    print('TESTS',len(sys.argv[1:] or names)-fails,'passed',fails,'failed',flush=True);sys.stdout.flush();os._exit(1 if fails else 0)
