"""WL80 focused tests (python test80.py), after the Stage 2 gate and before any learned fit:
 T1 basis provenance and alignment: G1 = the selected restart re-indexed by the recorded assignment (bitwise); A column sums / O row sums = 1,
    A, O > 0; the rank permutation is an exact operator symmetry (objective of a random T unchanged when T is re-indexed alike)
 T2 initialisation per arm: trunk bitwise the Q78 initial state (q78.init_model, same seeds); logits bitwise the arm's basis (float32);
    parameter names / shapes / counts equal to Q78; no port-indexed parameter
 T3 physics (r57_eval.physics) of the initial A1 model at rest / pt0 and of a perturbed copy
 T4 symmetry presentation: fit79.SymRest reproduces independently built exact image targets (sym79.Map.response) and restores bitwise
 T5 Stage 5 provenance: ports80 design == the WL79 frozen list; no WL79 reference / ports file / Stage 5 output; disjoint from the training
    orbit and earlier scored ports + images; seeds unique
 T6 decision logic: synthetic records through the fit gate, the mechanism arm rule and the Stage 5 factorial classification
 T7 no confirmation leakage in the training / basis code (no evaluation or confirmation root referenced)"""
import sys,os,json
import numpy as np
import torch
from common80 import CFG,ROOT,R73,R79,INP68,environment,load
RG=environment();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
import q78 as Q,rank75 as RK,family68 as FM,ato74 as AT,fit71,fit73,fit80,sym72 as SY,sym79 as S9
G=RG.device(fit71.guarded('geometry_rest'))
def rel(a,b):return float((a-b).double().norm()/b.double().norm().clamp(min=1e-30))

def test_basis():
    g1=load(ROOT/'basis/G1.pt');sel=json.loads((ROOT/'basis/select.json').read_text());src=load(ROOT/'basis'/f"B80_r{sel['selected_restart']}.pt")
    perm=torch.as_tensor(sel['correspondence']['assignment']);assert torch.equal(g1['in_logits'],src['in_logits'][perm]) and torch.equal(g1['out_logits'],src['out_logits'][perm])
    A,O=AT.AO(g1['in_logits'],g1['out_logits']);assert (A.sum(0)-1).abs().max()<1e-12 and (O.sum(1)-1).abs().max()<1e-12 and A.min()>0 and O.min()>0
    t=load(R73/'fit_targets73.pt',mmap=True);S=t['S']['r0123']['rest'][:2].double().cuda();ports=t['ports'][:2];bm=FM.bin_map('cuda');d=(1-G['beta'][ports]).double()
    tg=AT.target(S,d,G['parent'],bm);gen=torch.Generator(device='cuda').manual_seed(80);T=torch.rand(2,G['P'],16,3,device='cuda',dtype=torch.float64,generator=gen)/G['P']
    A0,O0=AT.AO(src['in_logits'].cuda(),src['out_logits'].cuda());A1,O1=AT.AO(g1['in_logits'].cuda(),g1['out_logits'].cuda())
    j0=float(AT.objective_m(T,A0,O0,tg));j1=float(AT.objective_m(T[:,:,perm.cuda()],A1,O1,tg));assert abs(j0-j1)<=1e-12*abs(j0),(j0,j1)

def test_init():
    q=Q.load_run(RG,ROOT.parent/'rest_srcid78/runs/Q78_5901.pt') if (ROOT.parent/'rest_srcid78/runs/Q78_5901.pt').exists() else None
    for rid,spec in CFG['runs'].items():
        m,_=fit80.init_model(spec);b,_=Q.init_model(RG,spec['init'],spec['reinit_seed'],16);a,o,_=fit80.basis(spec['arm'])
        for k,v in m.state_dict().items():
            if k in ('in_logits','out_logits'):continue
            assert torch.equal(v,b.state_dict()[k]),(rid,k)
        assert torch.equal(m.in_logits.cpu(),a) and torch.equal(m.out_logits.cpu(),o)
        if q is not None:assert {k:tuple(v.shape) for k,v in m.state_dict().items()}=={k:tuple(v.shape) for k,v in q.state_dict().items()} and RK.parameter_counts(m)==RK.parameter_counts(q)
        assert not any(G['P'] in tuple(v.shape) for v in m.state_dict().values())

def test_physics():
    from r57_eval import physics
    L=load(INP68/'wl59/lights.pt')['test'][:2].cuda();spec=[s for s in CFG['runs'].values() if s['arm'].startswith('A1')][0];m,_=fit80.init_model(spec);m.eval()
    gen=torch.Generator(device='cuda').manual_seed(80)
    for tag in ('init','perturbed'):
        if tag=='perturbed':
            with torch.no_grad():
                for p in (m.edge[-1].weight,m.node_head[-1].weight,m.encoder.weight):p.add_(torch.randn(p.shape,device='cuda',generator=gen)*.5)
        for name in ('rest','pt0'):
            g=RG.device(fit71.guarded(f'geometry_{name}'));r=physics(m,g,L,m.coefficients(g));assert r['passed'],(tag,name,r)

def test_symrest():
    import fit79
    t=load(R73/'fit_targets73.pt',mmap=True);ports=[p for _,p in fit73.populations()['S16R']];S=t['S']['r0123']['rest'].cuda();bm=FM.bin_map('cuda')
    tg=FM.Target(S.clone(),(1-G['beta'][ports]).float(),G['parent'],bm);del tg.S['native'];orig={k:v.clone() for k,v in tg.Sc.items()};sr=fit79.SymRest(RG,tg,ports,G);OPS=SY.ops(True)
    for step in (0,3,11):
        img=sr.set(step);ok=sr.ops_at(step)
        ref=FM.Target(torch.stack([sr.maps[o].response(S[k],ports[k]) for k,o in enumerate(ok)]),torch.stack([(1-G['beta'][q]).float() for q in img]),G['parent'],bm)
        for f in ('native','fine16'):assert rel(tg.Sc[f],ref.Sc[f])<=1e-6,(f,step)
        assert rel(tg.HZ,ref.HZ)<=1e-6 and torch.equal(tg.d,ref.d)
    sr.identity()
    for k,v in orig.items():assert torch.equal(tg.Sc[k],v),k

def test_stage5_provenance():
    import ports79 as P9,ports73 as P73
    w79=json.loads((ROOT/'repo/experiments/fnto_sym79/protocol/sym79_v1.json').read_text())['stage5']['ports'];rows,prev,orbit=P9.design();got={k:v['port'] for k,v in rows.items()}
    assert got==w79==CFG['stage5']['ports'];assert not (R79/'refs').exists() and not (R79/'ports79.json').exists() and not (R79/'stage5').exists()
    sl=P73.Slab(load(INP68/'wl59/geometry_rest.pt'));pts=[v for v in got.values() if v is not None]
    assert not set(pts)&set(orbit) and not set(pts)&set(sl.images(prev))
    import probes80 as PB
    seeds=[PB.seed(0,int(s),a,k) for s in got if got[s] is not None for a in range(16) for k in range(4)];assert len(set(seeds))==len(seeds)

def test_decision_logic():
    import eval80 as EV
    MS=['abs_native','abs_fine16','abs_coarse','con_native','con_fine16','con_coarse','tot_fine16','tot_coarse']
    def grp(v,lo=None):return dict(pooled={m:[dict(hi=v,lo=v*.9 if lo is None else lo,rho=v) for _ in range(3)] for m in MS},Q=dict(passed=v<=.1))
    tr=CFG['stage4']['fit_gate']['groups']
    def rec(vals,ok=True,ao=True):return dict(groups={g:grp(v) for g,v in vals.items()},passivity={'rest':dict(passed=ok)},run=dict(AO_identity=ao))
    res={n:rec({g:.2 for g in tr}) for n in EV.Q78};res.update({n:rec({g:.24 for g in tr}) for n in EV.RUNS})
    gt=EV.fit_gate(res);assert gt['any_passed'] and all(v['passed'] for v in gt['arms'].values())
    r2=dict(res);r2['A1B0_5902']=rec({g:.26 for g in tr});gt=EV.fit_gate(r2);assert not gt['arms']['A1B0']['passed'] and gt['arms']['A0B1']['passed']
    def s5(vals,old=(.8,.8),a75=(.9,.9),mech=None,int_new=.2):
        gr=lambda v,i=.2:{'BND_FAR':v,'BND_NEAR':.5,'INT':i}
        r={n:rec(gr(v)) for n,v in zip(EV.Q78,old)};r.update({n:rec(gr(v)) for n,v in zip(EV.A75,a75)})
        for a,(v1,v2) in vals.items():r[f'{a}_5901']=rec(gr(v1,int_new));r[f'{a}_5902']=rec(gr(v2,int_new))
        for n in r:r[n]['physics']={'rest':dict(passed=True)}
        return EV.decide5(r,list(vals),mech or {a:True for a in vals})['verdicts']
    assert s5({'A0B1':(.75,.78),'A1B0':(.4,.45),'A1B1':(.75,.78)})==['BASIS CHANGE BENEFIT','INDEPENDENT SOURCE TRANSFER SUPPORTED']
    assert s5({'A0B1':(.4,.45),'A1B0':(.75,.78),'A1B1':(.75,.78)})==['SYMMETRY PRESENTATION BENEFIT','INDEPENDENT SOURCE TRANSFER SUPPORTED']
    assert s5({'A0B1':(.75,.78),'A1B0':(.75,.78),'A1B1':(.4,.45)})==['INTERACTION BENEFIT','INDEPENDENT SOURCE TRANSFER SUPPORTED']
    assert s5({'A0B1':(.75,.78),'A1B0':(.75,.78),'A1B1':(.75,.78)})==['ORBIT-ONLY TRANSFER']
    assert s5({'A0B1':(.5,.7),'A1B0':(.75,.78),'A1B1':(.75,.78)})==['INCONCLUSIVE']
    assert 'INDEPENDENT SOURCE TRANSFER SUPPORTED' not in s5({'A0B1':(.75,.78),'A1B0':(.4,.45),'A1B1':(.75,.78)},int_new=.3)

def test_no_leakage():
    here=os.path.dirname(os.path.abspath(__file__))
    for f in ('fit80.py','basis80.py'):
        src=open(os.path.join(here,f)).read().split('"""',2)[-1]
        for bad in ('R77A','arch77a','R75L','rank75L','refs/bundles','ports80','stage5','[4,5,6,7]'):assert bad not in src,(f,bad)

if __name__=='__main__':
    T=[test_basis,test_init,test_physics,test_symrest,test_stage5_provenance,test_decision_logic,test_no_leakage];fails=0;res={}
    for t in T:
        try:t();print('PASS',t.__name__,flush=True);res[t.__name__]='PASS'
        except Exception as e:fails+=1;print('FAIL',t.__name__,repr(e)[:400],flush=True);res[t.__name__]=f'FAIL {e!r}'[:2000]
    if os.environ.get('R80_TEST_RECORD'):p=ROOT/'tests/tests.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(res,indent=1))
    print('TESTS',len(T)-fails,'passed',fails,'failed',flush=True);sys.stdout.flush();os._exit(1 if fails else 0)
