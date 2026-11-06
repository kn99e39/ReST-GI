"""WL75 focused Stage 0 tests (python test75.py): rank-generic class == historical rank-8 runtime (columns, coefficients, apply, column
sums, resident rebuild); synthetic rank-16 embedding of rank-8 coefficients reproduces the physical response; positivity / zero input /
linearity / passivity / cache identity at rank 16 (embedded, initial candidate, perturbed); rank-generic training objective == the
historical fit73 objective at rank 8; G2 basis provenance and selection rule; predeclared initialisation; frozen A/O; Stage 3 isolation."""
import sys,os,json,inspect
import torch
from common75 import CFG,ROOT,R74,R73,R68,INP68,sha,tensor_sha,environment
import family68 as FM
import family71 as FW
import rank75 as RK
import ato74 as AT

RG=environment();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
def load(p,mmap=False):return torch.load(p,map_location='cpu',weights_only=False,mmap=mmap)
def rel(a,b):return float((a-b).double().norm()/b.double().norm().clamp(min=1e-30))
G=None;L=None
def geom():
    global G,L
    if G is None:
        import fit71
        G=fit71.geometries(RG);L=load(INP68/'wl59/lights.pt')['test'][:4].cuda()
    return G,L
def ar8():
    from runtime import Transport
    st=load(R73/'runs/AR_5901.pt')['model'];h=Transport().cuda();h.load_state_dict(st);h.eval();m=RK.TransportM(8).cuda();m.load_state_dict(st);m.eval();return h,m,st
PORTS=[100,1716,3485,224,2059]

def test_rank8_reproduction():
    """TransportM(8) with the AR_5901 state == runtime.Transport (columns / coefficients / apply / column sums / runtime.full)."""
    import runtime
    g,_=geom();h,m,_=ar8();x=L
    with torch.no_grad():
        for name in ('rest','pt0'):
            gg=g[name];a=FW.network_columns(h,gg,PORTS);b=RK.columns(m,gg,PORTS)
            for u,v in zip(a,b):assert rel(v,u)<=1e-6,('columns',name)
            ch=h.coefficients(gg);cm=m.coefficients(gg)
            for k in ch:assert rel(cm[k],ch[k])<=1e-6,('coefficients',k)
            assert rel(m.apply(gg,x,cm),h.apply(gg,x,ch))<=1e-6,'apply'
            assert rel(m.column_sums(gg,cm),h.column_sums(gg,ch))<=1e-6,'column sums'
            cf=runtime.full(h,gg)['co'];assert all(rel(cm[k],cf[k])<=1e-6 for k in cf),'resident full rebuild'

def test_rank16_embedding():
    """Synthetic rank-16 embedding (mixture r <- r mod 8, logits repeated) gives the same physical response as the rank-8 operator."""
    g,_=geom();h,_,st=ar8();m=RK.TransportM(16).cuda();m.load_state_dict(RK.embed_rank(st,16));m.eval()
    with torch.no_grad():
        for name in ('rest','pt0'):
            gg=g[name];ch=h.coefficients(gg);cm=m.coefficients(gg)
            assert rel(m.apply(gg,L,cm),h.apply(gg,L,ch))<=1e-5,('apply',name)
            assert rel(m.column_sums(gg,cm),h.column_sums(gg,ch))<=1e-6,('column sums',name)
            T8,A8,O8=FW.network_columns(h,gg,PORTS);T16,A16,O16=RK.columns(m,gg,PORTS);d=(1-gg['beta'][PORTS]).float()
            assert rel(FM.predict(T16,d,A16,O16,'native'),FM.predict(T8,d,A8,O8,'native'))<=1e-5,('predict',name)
            assert torch.equal(T16[:,:,:8],T16[:,:,8:]) and rel(T16[:,:,:8],T8)<=1e-6,'duplicated T'

def test_physics_contract_rank16():
    """Positivity, zero input, linearity, column sums <= 1 and cache identity (r57_eval.physics) at rank 16: embedded AR, the initial
    candidate (P_5901 trunk + G2) and a randomly perturbed initial candidate."""
    from r57_eval import physics
    g,_=geom();_,_,st=ar8();ms=[]
    m=RK.TransportM(16).cuda();m.load_state_dict(RK.embed_rank(st,16));ms.append(('embedded',m))
    m,_=RK.init_model('P_5901');ms.append(('init',m))
    m2,_=RK.init_model('P_5902');gen=torch.Generator(device='cuda').manual_seed(75)
    with torch.no_grad():
        for p in (m2.edge[-1].weight,m2.edge[-1].bias,m2.node_head[-1].weight,m2.node_head[-1].bias):p.add_(torch.randn(p.shape,device='cuda',generator=gen)*.5)
    ms.append(('perturbed',m2))
    with torch.no_grad():
        for tag,m in ms:
            for name in ('rest','pt0'):
                r=physics(m,g[name],L[:2],m.coefficients(g[name]));assert r['passed'],(tag,name,r)

def test_objective_identity_rank8():
    """fit75's rank-generic loss (ato74.objective_m, rank75.columns) == the historical fit73 loss (fit71.objective, family71) for AR_5901."""
    import fit71,fit72,fit73,sym72 as SY
    g,_=geom();h,m,_=ar8();pb=fit73.Problem73(RG,'S16R');gs=dict(pb.gs);sym=fit72.sym_data(RG,gs,pb.ports,pb.F)
    with torch.no_grad():
        for gname in ('rest','pt0'):
            a=float(fit71.objective(*FW.network_columns(h,gs[gname],pb.ports),pb.tg[gname],False));b=float(AT.objective_m(*RK.columns(m,gs[gname],pb.ports),pb.tg[gname]))
            assert abs(a-b)<=1e-6*abs(a),('J',gname,a,b)
        key,src,lab=sym[0];ss=src[:64];d=(1-gs[key]['beta'][ss]).float()
        ya=SY.model_integrated(*FW.network_columns(h,gs[key],ss)[:2],d);yb=SY.model_integrated(*RK.columns(m,gs[key],ss)[:2],d);assert rel(yb,ya)<=1e-6,'L_sym term'

def test_basis_provenance_and_selection():
    """G2 basis = argmin of the WL74 fitting objective J_final over the G2 restarts (basis files only); hash, rank, J recorded in the
    protocol; float64 logits rounded once to float32; restart 0 is the duplicated WL68 basis start (WL74 init rule)."""
    a,o,meta=RK.basis();bc=CFG['basis'];assert meta['id']==bc['selected'] and meta['sha256']==bc['sha256']
    assert meta['id']==min(meta['J_final_by_restart'],key=meta['J_final_by_restart'].get)
    b=load(R74/'bases'/f"{bc['selected']}.pt");assert torch.equal(a,b['in_logits'].float()) and torch.equal(o,b['out_logits'].float())
    A,O=AT.AO(a,o);assert float((A.sum(0)-1).abs().max())<1e-6 and float((O.sum(1)-1).abs().max())<1e-6
    assert b['restart']==0 and CFG['basis']['selected']=='G2_r0'

def test_initialisation_and_frozen_AO():
    """Predeclared init: trunk == P_590x bitwise, rank-dependent rows == P_590x rows r mod 8, logits == G2; 'net' group excludes A/O;
    parameter counts equal the protocol."""
    import fit71,fit68
    for init in ('P_5901','P_5902'):
        st8=fit71.guarded(init)['model'];m,meta=RK.init_model(init);st=m.state_dict();e,nh=RK.rank_rows(16)
        for k,v in st8.items():
            if k in ('in_logits','out_logits'):continue
            if k.startswith('edge.3'):assert torch.equal(st[k].cpu(),v[e]),k
            elif k.startswith('node_head.2'):assert torch.equal(st[k].cpu(),v[nh]),k
            else:assert torch.equal(st[k].cpu(),v),k
        a,o,_=RK.basis();assert torch.equal(st['in_logits'].cpu(),a) and torch.equal(st['out_logits'].cpu(),o)
        params=fit68.group_params(m,'net');assert all(p is not m.in_logits and p is not m.out_logits for p in params)
        assert not m.in_logits.requires_grad and not m.out_logits.requires_grad
        pc=RK.parameter_counts(m);assert pc==CFG['stage0']['parameter_counts']['rank16'],pc
    _,m8,_=ar8();assert RK.parameter_counts(m8)==CFG['stage0']['parameter_counts']['rank8']

def test_stage3_isolation():
    """Stage 3 population = WL74 ports4.json (hash) = WL74 protocol frozen list; disjoint from S16-R, WL73 confirmation, every previously
    referenced port and all their exact images; WL74 never generated Stage 4 references; training targets are fitting replicas 0-3;
    Stage 2 scores replicas 4-7."""
    import fit73,ports73 as P73,eval73
    assert sha(R74/'ports4.json')==CFG['stage3']['inputs']['ports4.json'] and not (R74/'refs').exists()
    s4={int(k):v['port'] for k,v in json.loads((R74/'ports4.json').read_text())['ports'].items()};assert s4=={int(k):v for k,v in CFG['stage3']['ports'].items()}
    assert s4=={int(k):v for k,v in json.loads((R74/'repo/experiments/fnto_ato74/protocol/ato74_v1.json').read_text())['frozen']['stage4_ports'].items()} if (R74/'repo').exists() else True
    w=json.loads((R73/'ports.json').read_text());S16R={p for _,p in fit73.populations()['S16R']};conf={v['port'] for v in w['confirmation'].values()};S4=set(s4.values())
    sl=P73.Slab(load(INP68/'wl59/geometry_rest.pt'));p71,p72=P73.referenced();prev=set(p71.values())|set(p72.values())|S16R|conf
    assert not (S4&set(sl.images(list(prev)))),'Stage 3 leakage'
    t=load(R73/'fit_targets73.pt',mmap=True);assert 'replicas 0-3' in t['layout']
    assert 'ev=[4,5,6,7]' in inspect.getsource(eval73.load_refs)

def test_decision_logic():
    """Synthetic records through the predeclared Stage 2 gate and Stage 3 decision (every branch that the evidence could take)."""
    import eval68 as E,eval75 as EV
    MS=[s[0] for s in E.STD]
    def grp(v,lo=None):return dict(pooled={m:[dict(hi=v,lo=v*.9 if lo is None else lo,rho=v) for _ in range(3)] for m in MS},Q=dict(passed=v<=.1))
    def rec(vals,ok=True):
        r=dict(groups={g:grp(v) for g,v in vals.items()},passivity={'rest':dict(passed=ok)},run=dict(AO_identity=True),deformation={'DEF4':dict(unmeasured=True)});return r
    tr=['TRAIN_NEW_rest','TRAIN_NEW_pt0','TRAIN_KEEP_rest','TRAIN_KEEP_pt0']
    orc={b:dict(groups={g:grp(.1) for g in tr}) for b in ('G0_r0',CFG['basis']['selected'])};prov=dict(passed=True)
    res={n:rec({g:.27 for g in tr}) for n in EV.OLD};res.update({n:rec({**{g:.27 for g in tr},'TRAIN_NEW_rest':.15,'TRAIN_NEW_pt0':.15}) for n in EV.NEW})
    # the synthetic candidate improves every metric on TRAIN_NEW (incl. abs) -> passes; equal elsewhere -> no regression
    assert EV.fit_gate(res,orc,prov)['classification']=='FIT GATE PASSED'
    res2=dict(res);res2['A75_5902']=rec({g:.27 for g in tr});assert EV.fit_gate(res2,orc,prov)['classification']=='INSUFFICIENT FIT'
    res3=dict(res);res3['A75_5901']=rec({**{g:.27 for g in tr},'TRAIN_NEW_rest':.15,'TRAIN_NEW_pt0':.15},ok=False);assert EV.fit_gate(res3,orc,prov)['classification']=='PHYSICAL VALIDITY FAILURE'
    g4=['BND4_FAR','BND4_NEAR','INT4',f'FP4_{EV.POSE}']
    def s3(new_far,old_far=.9,int_new=.2,old_lo=None):
        r={n:rec({'BND4_FAR':old_far,'BND4_NEAR':.9,'INT4':.2,f'FP4_{EV.POSE}':.2}) for n in EV.OLD}
        if old_lo is not None:
            for n in EV.OLD:
                for c in r[n]['groups']['BND4_FAR']['pooled']['con_coarse']:c['lo']=old_lo
        r.update({n:rec({'BND4_FAR':v,'BND4_NEAR':.9,'INT4':int_new,f'FP4_{EV.POSE}':.2}) for n,v in zip(EV.NEW,new_far)});return EV.decide3(r,{})
    c=lambda x:x['classes'][0]
    assert c(s3((.4,.45))).startswith('TRANSFERABLE'),s3((.4,.45))
    assert c(s3((.85,.8))).startswith('NO INDEPENDENT'),s3((.85,.8))
    assert c(s3((.7,.75))).startswith('INCONCLUSIVE'),s3((.7,.75))                 # one seed clears 0.8x, the other does not
    assert c(s3((.6,.65),old_lo=.5)).startswith('INCONCLUSIVE'),s3((.6,.65),old_lo=.5)   # clears 0.8x but not beyond the controls' lower bound
    assert c(s3((.4,.45),int_new=.3)).startswith('TRADE-OFF'),s3((.4,.45),int_new=.3)

if __name__=='__main__':
    T=[test_rank8_reproduction,test_rank16_embedding,test_physics_contract_rank16,test_objective_identity_rank8,test_basis_provenance_and_selection,
       test_initialisation_and_frozen_AO,test_stage3_isolation,test_decision_logic];fails=0;res={}
    for t in T:
        try:t();print('PASS',t.__name__,flush=True);res[t.__name__]='PASS'
        except Exception as e:fails+=1;print('FAIL',t.__name__,repr(e),flush=True);res[t.__name__]=f'FAIL {e!r}'
    if os.environ.get('R75_TEST_RECORD'):
        p=ROOT/'tests'/f'{os.environ["R75_TEST_RECORD"]}.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(res,indent=1))
    print('TESTS',len(T)-fails,'passed',fails,'failed',flush=True);sys.stdout.flush();os._exit(1 if fails else 0)
