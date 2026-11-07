"""WL79 focused tests (python test79.py), before any candidate fitting:
 T1 group structure of the exact maps (rest, all 16 x 16): Map(g2) after Map(g1) == Map(g2 g1) for ports, incoming and outgoing bins;
    inverse maps; every map exact
 T2 transformed labels: SymRest.set(t) reproduces, for every source and its scheduled operation, an independently built family68.Target of
    the exact image (sym79.Map.response of the native fitting target, image-port budget): every field and the objective on a random T;
    identity() restores the original tensors bitwise
 T3 schedule: every step presents all 16 operations once; every source sees every operation once per 16 steps; identity share 1/16
 T4 initialisation and capacity: the S79 initial state equals the Q78 initialisation rule bitwise (q78.init_model, same seeds); parameter
    names / shapes / counts equal to Q78; no parameter has a port-count dimension (no source-specific lookup); A/O = frozen G2
 T5 physics (r57_eval.physics: positivity, zero input, linearity, passivity, cache identity) at rest / pt0 for the initial and a perturbed model
 T6 geometry cache: W+UV-Q inputs recomputed from a fresh geometry object are bitwise equal; operator from a copied geometry equals the cached one
 T7 no confirmation-label leakage: fit79 / fit75 / fit73 sources reference no evaluation or confirmation root; the frozen Stage 5 population is
    disjoint from the training orbit and from every earlier referenced / scored port and its images
 T8 transformed labels vs physics: on the rest exact pairs among referenced S16-R sources, the transformed fitting-target mean of p equals the
    fitting-target mean of q = perm[p] within replica noise (independent MC)
 T9 decision logic: synthetic records through the fit gate, the mechanism criterion, the Stage 5 decision and the deformation flag"""
import sys,os,json
import numpy as np
import torch
from common79 import CFG,ROOT,R73,R77A,INP68,environment,load
RG=environment();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
import sym72 as SY,sym79 as S9,q78 as Q,rank75 as RK,family68 as FM,ato74 as AT,fit71,fit73
G=RG.device(fit71.guarded('geometry_rest'));X0=G['rest_xyz'].double().cpu().numpy();OPS=SY.ops(True)
_M={}
def M(op):
    if op not in _M:_M[op]=S9.Map(RG,'rest',op,X0)
    return _M[op]
def compose(a,b):
    """op c with matrix(c) = matrix(a) @ matrix(b)."""
    m=SY.matrix(a)@SY.matrix(b)
    for c in OPS:
        if np.allclose(SY.matrix(c),m):return c
    raise KeyError
def rel(a,b):return float((a-b).double().norm()/b.double().norm().clamp(min=1e-30))

def test_group_structure():
    P=G['P']
    for a in OPS:
        assert M(a).exact
        for b in OPS:
            c=compose(a,b);ma,mb,mc=M(a),M(b),M(c)
            assert torch.equal(ma.perm[mb.perm],mc.perm),('ports',a,b)
            p=torch.arange(P);q=mb.perm
            for N,f in ((4,'pin'),(8,'pout')):
                comp=torch.gather(getattr(ma,f)[q],1,getattr(mb,f));assert torch.equal(comp,getattr(mc,f)),('bins',N,a,b)

def test_symrest_labels():
    import fit79
    t=load(R73/'fit_targets73.pt',mmap=True);ports=[p for _,p in fit73.populations()['S16R']];assert t['ports']==ports
    S=t['S']['r0123']['rest'].cuda();bm=FM.bin_map('cuda');d0=(1-G['beta'][ports]).float()
    tg=FM.Target(S.clone(),d0,G['parent'],bm);del tg.S['native'];orig={k:v.clone() for k,v in tg.Sc.items()};sr=fit79.SymRest(RG,tg,ports,G)
    gen=torch.Generator(device='cuda').manual_seed(79);T=torch.rand(len(ports),G['P'],16,3,device='cuda',generator=gen)/G['P'];a,o,_=RK.basis();A,O=AT.AO(a.cuda(),o.cuda())
    for step in (0,1,5,15):
        img=sr.set(step);ok=sr.ops_at(step)
        ref=FM.Target(torch.stack([M(OPS[o_]).response(S[k],ports[k]) for k,o_ in enumerate(ok)]),torch.stack([(1-G['beta'][q]).float() for q in img]),G['parent'],bm)
        assert img==[int(M(OPS[o_]).perm[ports[k]]) for k,o_ in enumerate(ok)]
        for f in ('native','fine16'):assert rel(tg.Sc[f],ref.Sc[f])<=1e-6,('Sc',f,step,rel(tg.Sc[f],ref.Sc[f]))
        for f in ('fine16','coarse'):assert rel(tg.S[f],ref.S[f])<=1e-6,('S',f,step)
        assert rel(tg.HZ,ref.HZ)<=1e-6 and torch.equal(tg.d,ref.d)
        for f in ref.den:assert rel(tg.den[f],ref.den[f])<=1e-5,('den',f)
        ref.S.pop('native');ja=float(AT.objective_m(T,A,O,tg));jb=float(AT.objective_m(T,A,O,ref));assert abs(ja-jb)<=1e-5*abs(jb),(ja,jb)
    sr.identity()
    for k,v in orig.items():assert torch.equal(tg.Sc[k],v),k

def test_schedule():
    import fit79
    class D:pass
    sr=D();sr.n=16;sr.ops=OPS;f=fit79.SymRest.ops_at
    for t in range(40):assert sorted(f(sr,t))==list(range(16))
    for k in range(16):assert sorted(f(sr,t)[k] for t in range(16))==list(range(16))

def test_init_capacity():
    for rid,spec in CFG['runs'].items():
        a,_=Q.init_model(RG,spec['init'],spec['reinit_seed'],spec['rank']);qs=CFG['controls_init'][rid]
        b,_=Q.init_model(RG,qs['init'],qs['reinit_seed'],16)
        assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items()),'init differs from the Q78 rule'
        q78=Q.load_run(RG,Path_q78(qs['control']));assert {k:tuple(v.shape) for k,v in a.state_dict().items()}=={k:tuple(v.shape) for k,v in q78.state_dict().items()}
        assert RK.parameter_counts(a)==RK.parameter_counts(q78);assert not any(G['P'] in tuple(v.shape) for v in a.state_dict().values())
        x,y,_=RK.basis();assert torch.equal(a.in_logits.cpu(),x) and torch.equal(a.out_logits.cpu(),y)
def Path_q78(n):
    from common79 import R78;return R78/'runs'/f'{n}.pt'

def test_physics():
    from r57_eval import physics
    L=load(INP68/'wl59/lights.pt')['test'][:2].cuda();spec=list(CFG['runs'].values())[0];m,_=Q.init_model(RG,spec['init'],spec['reinit_seed'],16);m.eval()
    gen=torch.Generator(device='cuda').manual_seed(79)
    for tag in ('init','perturbed'):
        if tag=='perturbed':
            with torch.no_grad():
                for p in (m.edge[-1].weight,m.node_head[-1].weight,m.encoder.weight):p.add_(torch.randn(p.shape,device='cuda',generator=gen)*.5)
        for name in ('rest','pt0'):
            g=RG.device(fit71.guarded(f'geometry_{name}'));r=physics(m,g,L,m.coefficients(g));assert r['passed'],(tag,name,r)

def test_geometry_cache():
    a=Q.features(RG,G);g2=RG.device(fit71.guarded('geometry_rest'));b=Q.features(RG,g2)
    assert torch.equal(a[0],b[0]) and torch.equal(a[1],b[1])
    spec=list(CFG['runs'].values())[0];m,_=Q.init_model(RG,spec['init'],spec['reinit_seed'],16);m.eval()
    with torch.no_grad():c1=m.coefficients(G);c2=m.coefficients(dict(G))
    for k in c1:assert rel(c1[k],c2[k])<=1e-6,k

def test_no_leakage():
    here=os.path.dirname(os.path.abspath(__file__))
    for f in ('fit79.py',):
        src=open(os.path.join(here,f)).read()
        for bad in ('R77A','arch77a','R75L','rank75L','refs/bundles','ports79','stage5'):assert bad not in src.split('"""',2)[-1],(f,bad)
    import ports79 as P9
    rows,prev,orbit=P9.design();got={k:v['port'] for k,v in rows.items()};assert got==CFG['stage5']['ports'],'frozen population differs'
    import ports73 as P73
    sl=P73.Slab(load(INP68/'wl59/geometry_rest.pt'));pts=[v for v in got.values() if v is not None]
    assert not set(pts)&set(orbit),'overlaps the training orbit';assert not set(pts)&set(sl.images(prev)),'overlaps an earlier scored port or image'
    assert len(set(sl.orbit_key(p) for p in pts))==len(pts),'two confirmation ports in one orbit'

def test_transformed_vs_physics():
    t=load(R73/'fit_targets73.pt',mmap=True);ports=[p for _,p in fit73.populations()['S16R']];S=t['S']['r0123']['rest'];n=0
    for op in OPS[1:]:
        mp=M(op)
        for i,p in enumerate(ports):
            q=int(mp.perm[p])
            if q in ports and q!=p:
                j=ports.index(q);a=mp.response(S[i].double(),p);b=S[j].double();r=rel(a,b);assert r<=.10,(op,p,q,r);n+=1
    assert n>0,'no exact pair among the training sources'

def test_decision_logic():
    import eval79 as EV
    MS=['abs_native','abs_fine16','abs_coarse','con_native','con_fine16','con_coarse','tot_fine16','tot_coarse']
    def grp(v,lo=None):return dict(pooled={m:[dict(hi=v,lo=v*.9 if lo is None else lo,rho=v) for _ in range(3)] for m in MS},Q=dict(passed=v<=.1))
    tr=CFG['stage3']['gate']['groups']
    def rec(vals,ok=True,ao=True):return dict(groups={g:grp(v) for g,v in vals.items()},passivity={'rest':dict(passed=ok)},run=dict(AO_identity=ao))
    res={n:rec({g:.2 for g in tr}) for n in EV.Q78};res.update({n:rec({g:.24 for g in tr}) for n in EV.S79})
    assert EV.fit_gate(res)['classification']=='FIT GATE PASSED'
    r2=dict(res);r2[EV.S79[1]]=rec({g:.26 for g in tr});assert EV.fit_gate(r2)['classification']=='INSUFFICIENT FIT'
    r3=dict(res);r3[EV.S79[0]]=rec({g:.2 for g in tr},ok=False);assert EV.fit_gate(r3)['classification']=='PHYSICAL VALIDITY FAILURE'
    def s5(new,old=(.8,.8),a75=(.9,.9),int_new=.2,lo=None):
        gr=lambda v,i=.2:{'BND_FAR':v,'BND_NEAR':.5,'INT':i}
        r={n:rec(gr(v)) for n,v in zip(EV.Q78,old)};r.update({n:rec(gr(v)) for n,v in zip(EV.A75,a75)});r.update({n:rec(gr(.95)) for n in EV.AR})
        if lo is not None:
            for n in EV.Q78:
                for c in r[n]['groups']['BND_FAR']['pooled']['con_coarse']:c['lo']=lo
        r.update({n:rec(gr(v,int_new)) for n,v in zip(EV.S79,new)})
        for n in r:r[n]['physics']={'rest':dict(passed=True)}
        return EV.decide5(r)
    v=lambda x:x['verdict']
    assert v(s5((.4,.45)))=='SYMMETRY TRAINING BENEFIT',s5((.4,.45))
    assert v(s5((.75,.78)))=='EQUIVARIANCE IMPROVED, TRANSFER UNCHANGED'
    assert v(s5((.5,.7)))=='INCONCLUSIVE'
    assert v(s5((.55,.6),lo=.55))=='INCONCLUSIVE'
    x=s5((.4,.45),int_new=.3);assert x['verdict']=='INCONCLUSIVE' and x['qualifiers'],x
    y=s5((.4,.45),old=(.95,.95),a75=(.6,.6));assert y['stronger_control']['BND_FAR']==EV.A75[:3] or y['stronger_control']['BND_FAR']==EV.A75,y['stronger_control']

if __name__=='__main__':
    T=[test_group_structure,test_symrest_labels,test_schedule,test_init_capacity,test_physics,test_geometry_cache,test_no_leakage,test_transformed_vs_physics,test_decision_logic]
    fails=0;res={}
    for t in T:
        try:t();print('PASS',t.__name__,flush=True);res[t.__name__]='PASS'
        except Exception as e:fails+=1;print('FAIL',t.__name__,repr(e)[:400],flush=True);res[t.__name__]=f'FAIL {e!r}'[:2000]
    if os.environ.get('R79_TEST_RECORD'):p=ROOT/'tests/tests.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(res,indent=1))
    print('TESTS',len(T)-fails,'passed',fails,'failed',flush=True);sys.stdout.flush();os._exit(1 if fails else 0)
