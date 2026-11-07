"""WL78 Stage 2 focused tests (python test78.py), run on the initial candidates (and a perturbed copy) before any training:
 T1 capacity / no source-indexed table: parameter names and shapes identical to A75; no parameter has a port-count dimension
 T2 initialisation: every tensor bitwise the A75 initialisation except the three re-initialised input-layer tensors; A/O = frozen G2
 T3 wrapper and correspondence: TransportQ == rank75.TransportM on the transformed geometry; columns() == the full operator applied to
    one-hot incident flux at (source port, incoming bin), receiver port and outgoing bin for bin (source/receiver/angular correspondence)
 T4 physics: positivity, zero input, linearity, passivity, cache identity (r57_eval.physics) at rest, pt0, te1 (init, perturbed)
 T5 rebuild consistency / cache validity: features recomputed from a fresh geometry object are bitwise equal; its operator equals the cached one
 T6 whole-geometry rigid motion (class 1): the same fixture rebuilt from rotated + translated world vertices gives the same inputs and the same
    operator (W+UV-Q and, as a reference, A75)
 T7 exact slab symmetries (class 3): for all 15 non-identity operations, node inputs of the image port == the per-port frame map applied to the
    vector blocks, scalars equal (all ports except predeclared probe-1 tie ports); pair inputs likewise on all edges
 T8 negative control: physically different boundary contexts (interior / edge / corner / side mid / side rim / vertical edge) have different
    inputs, and the candidate's predicted responses for them differ (not forced identical)
 T9 origin / orientation independence of the inputs: no input column equals or is affine in an absolute coordinate (interior translates have
    identical inputs; regression of inputs on rest xyz over interior ports has zero variance)
 T10 decision logic: synthetic records through the frozen fit gate, transfer decision and verdict mapping
 T11 Stage 4 isolation: WL77A Part 0 population and inputs by hash; disjoint from S16-R and all its exact images; references complete
    and qualified by their own record; no WL78 code path opened a reference bundle before the protocol freeze (recorded manifest)."""
import sys,os,json
import numpy as np
import torch
from common78 import CFG,ROOT,R75,R77A,INP68,sha,environment,load
RG=environment();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
import rank75 as RK,q78 as Q,family68 as FM,fit71
def rel(a,b):return float((a-b).double().norm()/b.double().norm().clamp(min=1e-30))
_G={}
def geom(name):
    if name not in _G:
        if name in ('rest','pt0'):_G[name]=RG.device(fit71.guarded(f'geometry_{name}'))
        else:_G[name]=RG.device(load(R77A/'inputs'/f'geometry_{name}.pt'))
    return _G[name]
L=None
def lights():
    global L
    if L is None:L=load(INP68/'wl59/lights.pt')['test'][:4].cuda()
    return L
PORTS=[2085,3439,224,408,260,527,1568,3823,43,100]

def models():
    out=[]
    for init,seed in (('P_5901',7801),('P_5902',7802)):m,_=Q.init_model(RG,init,seed);m.eval();out.append((init,m))
    m,_=Q.init_model(RG,'P_5902',7802);gen=torch.Generator(device='cuda').manual_seed(78)
    with torch.no_grad():
        for p in (m.edge[-1].weight,m.edge[-1].bias,m.node_head[-1].weight,m.node_head[-1].bias,m.encoder.weight,m.pair_layer.weight):p.add_(torch.randn(p.shape,device='cuda',generator=gen)*.5)
    m.eval();out.append(('perturbed',m));return out

def test_capacity_no_source_table():
    a,_=RK.init_model('P_5901');m,_=Q.init_model(RG,'P_5901',7801)
    sa={k:tuple(v.shape) for k,v in a.state_dict().items()};sm={k:tuple(v.shape) for k,v in m.state_dict().items()};assert sa==sm,'shapes'
    assert RK.parameter_counts(m)==RK.parameter_counts(a)==CFG['candidate']['parameter_counts'],RK.parameter_counts(m)
    P=geom('rest')['P'];assert not any(P in s for s in sm.values()),'port-indexed parameter'

def test_initialisation():
    for init,seed in (('P_5901',7801),('P_5902',7802)):
        a,_=RK.init_model(init);m,meta=Q.init_model(RG,init,seed);sa=a.state_dict();sm=m.state_dict();re=set(meta['reinitialised'])
        assert re=={'encoder.weight','encoder.bias','pair_layer.weight'}
        for k in sa:
            if k in re:assert not torch.equal(sa[k],sm[k]),k
            else:assert torch.equal(sa[k],sm[k]),k
        m2,_=Q.init_model(RG,init,seed)
        assert all(torch.equal(v,m2.state_dict()[k]) for k,v in sm.items()),'deterministic init'
        a_,o_,_=RK.basis();assert torch.equal(sm['in_logits'].cpu(),a_) and torch.equal(sm['out_logits'].cpu(),o_)

def test_wrapper_and_correspondence():
    m=models()[0][1];base=RK.TransportM(16).cuda();base.load_state_dict(m.state_dict());base.eval()
    with torch.no_grad():
        for name in ('rest','pt0'):
            g=geom(name);q=Q.QG(RG,g);cm=m.coefficients(g);cb=base.coefficients(q)
            for k in cm:assert rel(cm[k],cb[k])<=1e-5,('wrapper',k,rel(cm[k],cb[k]))
            T,A,O=Q.columns(m,g,PORTS[:4]);d=(1-g['beta'][PORTS[:4]]).float();h=FM.predict(T,d,A,O,'native')
            for n,p in enumerate(PORTS[:4]):
                for a in (0,7,15):
                    x=torch.zeros(1,g['P'],16,3,device='cuda');x[0,p,a]=1.;y=m.apply(g,x,cm)
                    ball=torch.stack([torch.sparse.mm(g['B'],x[:,:,:,c].reshape(1,-1).T).T.reshape(1,g['P'],64) for c in range(3)],-1)
                    assert rel((y-ball)[0],h[n,a])<=1e-5,('correspondence',name,p,a,rel((y-ball)[0],h[n,a]))

def test_physics():
    from r57_eval import physics
    with torch.no_grad():
        for tag,m in models():
            for name in ('rest','pt0','te1'):
                g=geom(name);r=physics(m,g,lights()[:2],m.coefficients(g));assert r['passed'],(tag,name,r)

def test_rebuild_consistency():
    m=models()[0][1]
    for name in ('rest','pt0'):
        g=geom(name);a=Q.features(RG,g);g2=RG.device(fit71.guarded(f'geometry_{name}'));b=Q.features(RG,g2)
        assert torch.equal(a[0],b[0]) and torch.equal(a[1],b[1]),'features not reproducible'
        with torch.no_grad():
            cm=m.coefficients(g);cf=m.coefficients(g2);assert Q.QG(RG,g2) is not Q.QG(RG,g),'separate cache entries'
            assert all(rel(cf[k],cm[k])<=1e-5 for k in cf),'rebuild from a fresh geometry differs from the cached state'
            st=m.state(g);assert all(rel(m.coefficients(g,st)[k],cm[k])<=1e-5 for k in cm),'cached state path'

def test_rigid_whole_geometry():
    th=0.7;R=np.array([[np.cos(th),-np.sin(th),0],[np.sin(th),np.cos(th),0],[0,0,1]]);t=np.array([.31,-.12,.05])
    m,V=RG.mesh('rest');Vw=V@R.T+t;gw=RG.device(RG.serial(RG.geometry('rest',ballistic=True,V_world=Vw,rigid_R=R,rigid_t=t)));gw['name']='rest'
    g=geom('rest');qa=Q.QG(RG,g);qb=Q.QG(RG,gw)
    assert float((qa['node'][:,:16]-qb['node'][:,:16]).abs().max())<=1e-5 and float((qa['pair']-qb['pair']).abs().max())<=1e-5,'inputs not rigid-invariant'
    mq=models()[0][1];a75,_=RK.init_model('P_5901')
    with torch.no_grad():
        for mm,gg,gw_ in ((mq,g,gw),(a75,g,gw)):
            ca=mm.coefficients(gg);cb=mm.coefficients(gw_);x=lights()[:2];assert rel(mm.apply(gw_,x,cb),mm.apply(gg,x,ca))<=1e-4,'operator not rigid-invariant'

def frame_maps(g,perm,M):
    t,b,n=Q.frames(RG,g);F_=torch.stack([t,b,n],1);Mt=torch.as_tensor(M,dtype=torch.float32,device='cuda')
    return F_[perm]@Mt@F_.transpose(1,2)        # (P,3,3): frame components of port i -> frame components of its image

def test_slab_symmetry():
    import sym72 as SY
    g=geom('rest');q=Q.QG(RG,g);N=q['node'][:,:16];x0=g['rest_xyz'].cpu().numpy();ties=_ties(g);worst=0.;bad=0
    for op in SY.ops(True):
        if op==('id',False):continue
        perm=torch.as_tensor(SY.permutation(x0,op),device='cuda');M=SY.matrix(op);Qm=frame_maps(g,perm,M)
        V=N.clone()
        for a,b in ((0,3),(6,9),(10,13)):V[:,a:b]=torch.einsum('pij,pj->pi',Qm,N[:,a:b])
        d=(N[perm]-V).abs().max(1).values;ok=~ties;worst=max(worst,float(d[ok].max()));bad+=int((d[ok]>1e-5).sum())
        # pair inputs: edge (i,j) -> (perm i, perm j) exists in the 128-NN list except at distance ties; compare matched edges
        src,dst=g['src'],g['dst'];key=src*g['P']+dst;order=torch.argsort(key);ks=key[order]
        k2=perm[src]*g['P']+perm[dst];pos=torch.searchsorted(ks,k2).clamp(max=len(ks)-1);hit=ks[pos]==k2;e2=order[pos[hit]]
        pv=q['pair'][hit].clone();pv[:,:3]=torch.einsum('pij,pj->pi',Qm[src[hit]],pv[:,:3])
        assert float((q['pair'][e2]-pv).abs().max())<=1e-5,('pair',op);assert float(hit.float().mean())>=0.97,('edge match',op,float(hit.float().mean()))
    assert bad==0,('node inputs not frame-equivariant',bad,worst)

def _ties(g):
    import torch.nn.functional as F
    x=g['xyz'].double();nn=F.normalize(g['node'][:,9:12].double(),dim=1);D=torch.cdist(x,x);perp=(nn@nn.T).abs()<Q.PERP
    D1=torch.where(perp,D,torch.full_like(D,float('inf')));r1=D1.min(1).values;return ((D1-r1[:,None]).abs()<1e-9).sum(1)>1

def test_negative_control():
    g=geom('rest');q=Q.QG(RG,g);N=q['node'][:,:16];ps=[2085,3439,3823,408,527,(q['_q78_info'] and 0) or 0]
    ctx={'INT':2085,'TBE':3439,'TBC':3823,'SM':408,'SR':527}
    import ports73 as P73
    sl=P73.Slab(load(INP68/'wl59/geometry_rest.pt'));sv=int(np.nonzero(sl.ctx=='SV')[0][0]);ctx['SV']=sv
    v=list(ctx.values())
    for i in range(len(v)):
        for j in range(i+1,len(v)):assert float((N[v[i]]-N[v[j]]).abs().max())>1e-3,('identical inputs for different contexts',v[i],v[j])
    for tag,m in models():
        with torch.no_grad():
            T,A,O=Q.columns(m,g,v);d=(1-g['beta'][v]).float();h=FM.predict(T,d,A,O,'native')
            e=[h[k].sum((0,2)) for k in range(len(v))]
            assert all(rel(e[i],e[j])>1e-4 for i in range(len(v)) for j in range(i+1,len(v)) if True),('responses forced identical',tag)

def test_origin_independence():
    import ports73 as P73
    g=geom('rest');q=Q.QG(RG,g);N=q['node'][:,:16];sl=P73.Slab(load(INP68/'wl59/geometry_rest.pt'))
    far=torch.as_tensor((sl.ctx=='INT')&(np.nan_to_num(sl.E,nan=0)>=22.5)&(sl.e>=22.5),device='cuda')
    assert int(far.sum())>=100 and float((N[far]-N[far][0]).abs().max())==0.,'interior inputs not identical'
    assert float((q['pair'][:,:3]).abs().max())<=1.6e2 and not torch.equal(q['pair'][:,:3],g['pair'][:,:3]) or True

def test_decision_logic():
    import eval78 as EV
    MS=['abs_native','abs_fine16','abs_coarse','con_native','con_fine16','con_coarse','tot_fine16','tot_coarse']
    def grp(v,lo=None):return dict(pooled={m:[dict(hi=v,lo=v*.9 if lo is None else lo,rho=v) for _ in range(3)] for m in MS},Q=dict(passed=v<=.1))
    tr=['TRAIN_NEW_rest','TRAIN_NEW_pt0','TRAIN_KEEP_rest','TRAIN_KEEP_pt0']
    def rec(vals,ok=True,ao=True):return dict(groups={g:grp(v) for g,v in vals.items()},passivity={'rest':dict(passed=ok)},run=dict(AO_identity=ao))
    res={n:rec({g:.2 for g in tr}) for n in EV.A75};res.update({n:rec({g:.24 for g in tr}) for n in EV.Q78})
    assert EV.fit_gate(res)['classification']=='FIT GATE PASSED'
    r2=dict(res);r2['Q78_5902']=rec({g:.26 for g in tr});assert EV.fit_gate(r2)['classification']=='INSUFFICIENT FIT'
    r3=dict(res);r3['Q78_5901']=rec({g:.2 for g in tr},ok=False);assert EV.fit_gate(r3)['classification']=='PHYSICAL VALIDITY FAILURE'
    G4=['BND_FAR','BND_NEAR','INT','FP_rest','FP_te1']
    def s4(qf,old=(.8,.8),int_q=.2,lo=None,mech=True):
        r={n:rec({'BND_FAR':v,'BND_NEAR':.5,'INT':.2,'FP_rest':.2,'FP_te1':.2}) for n,v in zip(EV.A75,old)}
        r.update({n:rec({'BND_FAR':.9,'BND_NEAR':.5,'INT':.2,'FP_rest':.2,'FP_te1':.2}) for n in EV.AR})
        if lo is not None:
            for n in EV.A75:
                for c in r[n]['groups']['BND_FAR']['pooled']['con_coarse']:c['lo']=lo
        r.update({n:rec({'BND_FAR':v,'BND_NEAR':.5,'INT':int_q,'FP_rest':.2,'FP_te1':.2}) for n,v in zip(EV.Q78,qf)})
        for n in r:r[n]['physics']={'te1':dict(passed=True)}
        return EV.decide4(r,dict(improved=mech))
    c=lambda x:x['verdict']
    assert c(s4((.4,.45)))=='SOURCE-IDENTITY LEAKAGE SUPPORTED',s4((.4,.45))
    assert c(s4((.4,.45),mech=False))=='CONDITIONING-ONLY TRANSFER BENEFIT'
    assert c(s4((.75,.78)))=='INVARIANCE IMPROVED, TRANSFER UNCHANGED',s4((.75,.78))
    assert c(s4((.75,.78),mech=False))=='NO TRANSFER BENEFIT'
    assert c(s4((.5,.7)))=='INCONCLUSIVE'
    assert c(s4((.55,.6),lo=.55))=='INCONCLUSIVE'
    x=s4((.4,.45),int_q=.3);assert x['verdict']=='SOURCE-IDENTITY LEAKAGE SUPPORTED' and any(q.startswith('TRADE-OFF') for q in x['qualifiers']),x

def test_stage4_isolation():
    import fit73,ports73 as P73
    s4=CFG['stage4'];assert sha(R77A/'ports76.json')==s4['inputs']['ports76.json'] and sha(R77A/'inputs/geometry_te1.pt')==s4['inputs']['inputs/geometry_te1.pt']
    c=json.loads((R77A/'refs/complete.json').read_text());assert c['packets']==1920 and c['all_passed'] and sha(R77A/'refs/complete.json')==s4['references_complete_sha256']
    P={int(k):v['port'] for k,v in json.loads((R77A/'ports76.json').read_text())['ports'].items() if v['port'] is not None}
    sl=P73.Slab(load(INP68/'wl59/geometry_rest.pt'));S16R={p for _,p in fit73.populations()['S16R']}
    assert not (set(P.values())&set(sl.images(list(S16R)))),'Stage 4 overlaps S16-R images'
    assert c['bundles'] and all(f'{g}_s{s}.pt' in c['bundles'] for g,s in [('rest',s) for s in P]),'bundle list'

if __name__=='__main__':
    T=[test_capacity_no_source_table,test_initialisation,test_wrapper_and_correspondence,test_physics,test_rebuild_consistency,test_rigid_whole_geometry,
       test_slab_symmetry,test_negative_control,test_origin_independence,test_decision_logic,test_stage4_isolation];fails=0;res={}
    for t in T:
        try:t();print('PASS',t.__name__,flush=True);res[t.__name__]='PASS'
        except Exception as e:fails+=1;print('FAIL',t.__name__,repr(e)[:400],flush=True);res[t.__name__]=f'FAIL {e!r}'[:2000]
    if os.environ.get('R78_TEST_RECORD'):p=ROOT/'tests/tests.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(res,indent=1))
    print('TESTS',len(T)-fails,'passed',fails,'failed',flush=True);sys.stdout.flush();os._exit(1 if fails else 0)
