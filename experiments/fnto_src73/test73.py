"""WL73 focused tests. python test73.py pre | post
pre : candidate identities (A/O, constructors, parameter counts, run specs), population / budget / leakage / seeds, exact symmetries
      of the new ports, adapter identity (Problem73 on S16 == fit71.Problem), runtime contract at boundary ports on rest/pt0/te0.
post: fit_targets73 identity (retained rows bitwise == fit_targets71, new rows == bundle means of replicas 0-3) and the fit-stage
      read whitelist on S16-R."""
import sys,os,json
import numpy as np
import torch
from common73 import CFG,ROOT,R71,R72,INP68,R68,environment,seed
import family68 as FM
import family71 as FW
import sym72 as SY

RG=environment();torch.backends.cuda.matmul.allow_tf32=False
def load(p,mmap=False):return torch.load(p,map_location='cpu',weights_only=False,mmap=mmap)
POSE=CFG['stage0']['deformation_pose']['selected']
G={g:RG.device(load(INP68/'wl59'/f'geometry_{g}.pt')) for g in ('rest','pt0')};G[POSE]=RG.device(load(ROOT/'inputs'/f'geometry_{POSE}.pt'))
PJ=json.loads((ROOT/'ports.json').read_text())
def rel(a,b):return float((a-b).double().norm()/b.double().norm().clamp(min=1e-30))

def test_candidate_identities():
    """D1/A: WL59 P_590x with the WL68 net_all_5902 A/O, 'net' group; B: RSK from the WL72 seeds; identical A/O; run specs as frozen."""
    import fit71,fit72
    a,o=fit71.angular();n71=load(R71/'runs/D1_5901.pt');n72=load(R72/'runs/A_5901.pt');b72=load(R72/'runs/B_7201.pt')
    for rid,sp in CFG['runs'].items():
        m,params,_=fit72.model_for(dict(candidate='A' if sp['family'] in ('D1','A') else 'B',init=sp.get('init'),seed=sp.get('seed')),RG)
        assert torch.equal(m.in_logits.cpu(),a) and torch.equal(m.out_logits.cpu(),o),rid
        n=sum(p.numel() for p in params);ref=n71['trainable'] if sp['family']=='D1' else n72['trainable'] if sp['family']=='A' else b72['trainable']
        assert n==ref,(rid,n,ref);assert sp['lambda_sym']==(1.0 if sp['family']=='A' else 0.0) and sp['population']=='S16R'
        if sp['family']=='B':
            torch.manual_seed(sp['seed']);import rsk72;m2=rsk72.RSK(a,o).cuda()
            assert all(torch.equal(x,y) for x,y in zip(m.state_dict().values(),m2.state_dict().values()))
    print('   trainable D1',n71['trainable'],'A',n72['trainable'],'B',b72['trainable'])

def test_population_budget_leakage_seeds():
    """16 sources (12 kept + 4 new, 4 removed), no new/confirmation port equals an image of a referenced, training or other confirmation
    port; same per-matrix photon budget and replica counts as S16; seeds unique and disjoint from WL66-72."""
    import fit73,probes73,ports73
    pops=fit73.populations();assert len(pops['S16R'])==16 and len({p for _,p in pops['S16R']})==16
    kept=[s for s,_ in pops['S16R'] if s<16];assert len(kept)==12 and sorted(set(range(16))-set(kept))==CFG['frozen']['removed_slots']
    sl=ports73.Slab(load(INP68/'wl59/geometry_rest.pt'))
    p71,p72=ports73.referenced();R0=set(p71.values())|set(p72.values());new=[int(v) for v in CFG['frozen']['new_ports'].values()]
    conf=[int(v) for v in CFG['frozen']['confirmation_ports'].values()]
    assert not set(new)&set(sl.images(list(R0)))
    train=[p for _,p in pops['S16R']]+[p for _,p in pops['S16']]
    assert not set(conf)&set(sl.images(train+list(R0))),'confirmation leakage'
    for i,c in enumerate(conf):assert not set(sl.images([c]))&set(conf[:i]+conf[i+1:]),('confirmation duplicates',c)
    pk=probes73.packets();ntr=sum(1 for p in pk if p[0]=='train');nc=len(pk)-ntr
    nst=len(CFG['stage0']['confirmation_static']);nfp=len(CFG['stage0']['footprint']['ports'])
    assert ntr==4*2*16*8 and nc==(nst*16+nfp*2*16)*4,(ntr,nc);assert probes73.N==32_000_000
    sd=[p[-1] for p in pk];assert len(set(sd))==len(sd) and all(733000000<=x<738000000 for x in sd)
    print('   packets train',ntr,'confirmation',nc)

def test_symmetry_new_ports():
    """Exact permutations; images of the new boundary sources stay in their boundary context."""
    import ports73
    sl=ports73.Slab(load(INP68/'wl59/geometry_rest.pt'))
    for p in sl.perms:assert sorted(p.tolist())==list(range(sl.P))
    for s,v in PJ['new'].items():
        ctx={str(sl.ctx[j]) for j in sl.images([v['port']])};assert ctx=={v['context']},(s,ctx)

def test_adapter_identity():
    """Problem73 on S16 builds exactly the WL71 problem: same ports, same targets, same J for the D1 / A / B initial models."""
    import fit71,fit72,fit73
    pb=fit73.Problem73(RG,'S16');ref=fit71.Problem(RG,dict(pop='S16',replicas=[0,1,2,3],target='phys'))
    assert pb.ports==ref.ports
    for g in ('rest','pt0'):
        for k in ('fine16','coarse'):assert rel(pb.tg[g].S[k],ref.tg[g].S[k])<=1e-6,(g,k,rel(pb.tg[g].S[k],ref.tg[g].S[k]))   # GPU index_add aggregation is not bitwise deterministic
        assert torch.equal(pb.tg[g].d,ref.tg[g].d)
    with torch.no_grad():
        for sp in (dict(candidate='A',init='P_5901'),dict(candidate='B',seed=7201)):
            m,_,cols=fit72.model_for(sp,RG)
            J1=sum(fit71.objective(*cols(pb.gs[g],pb.ports),pb.tg[g],False) for g in ('rest','pt0'))/2
            J2=sum(fit71.objective(*cols(ref.gs[g],ref.ports),ref.tg[g],False) for g in ('rest','pt0'))/2
            assert abs(float(J1)-float(J2))<=1e-6*abs(float(J2)),(sp,float(J1),float(J2))
    del pb,ref;torch.cuda.empty_cache()

def test_runtime_contract_boundary():
    """Initial D1/A and B models at the new boundary ports and a footprint port on rest / pt0 / te0: A/O fixed, nonnegative, passive
    (column sums <= 1), nonzero; one-hot factorisation of the complete runtime K (coefficients + apply) equals the column prediction."""
    import fit71,rsk72
    a,o=fit71.angular();torch.manual_seed(7201);ms={'W+UV':fit71.model_from('P_5902').eval(),'RSK':rsk72.RSK(a,o).cuda().eval()}
    ports=[v['port'] for v in PJ['new'].values()]+[PJ['confirmation']['80']['port']];lights=load(INP68/'wl59/lights.pt')['test'][:2].cuda();from r57_eval import physics
    for k,m in ms.items():
        for gn,g in G.items():
            with torch.no_grad():
                co=m.coefficients(g);ph=physics(m,g,lights,co);assert ph['passed'],(k,gn,ph)
                cols=(lambda gg,pp:m.columns(gg,pp)) if k=='RSK' else (lambda gg,pp:FW.network_columns(m,gg,pp))
                T,A,O=cols(g,ports);assert float(T.min())>=0 and float(T.sum((1,2,3)).min())>0 and float(T.sum(1).max())<=1+1e-6
                for j,i in enumerate(ports[:2]):
                    x=torch.zeros(16,g['P'],16,3,device='cuda');x[torch.arange(16),i,torch.arange(16)]=1.
                    tot=torch.cat([m.apply(g,x[s:s+4],co) for s in range(0,16,4)])
                    ball=torch.stack([torch.sparse.mm(g['B'],x[:,:,:,c].reshape(16,-1).T).T.reshape(16,g['P'],64) for c in range(3)],-1)
                    y=FM.predict(T[[j]],(1-g['beta'][[i]]).float(),A,O,'native')[0];assert rel(y,tot-ball)<=1e-5,(k,gn,i,rel(y,tot-ball))

def test_fit_targets73_identity():
    """Retained rows bitwise equal to fit_targets71; new rows equal the replica 0-3 means of the new bundles; sources hashed."""
    import fit73
    t=load(ROOT/'fit_targets73.pt');f71=load(R71/'fit_targets71.pt',mmap=True);pops=fit73.populations();assert t['ports']==[p for _,p in pops['S16R']]
    for g in ('rest','pt0'):
        for j,(s,p) in enumerate(pops['S16R']):
            if s<16:assert torch.equal(t['S']['r0123'][g][j],f71['S']['r0123'][g][s]),(g,s)
            else:
                b=load(ROOT/'refs/bundles'/f'{g}_s{s}.pt');assert torch.equal(t['S']['r0123'][g][j],b['S'][[0,1,2,3]].double().mean(0).float()),(g,s)

def test_fit_reads_only_allowed():
    """The S16-R fit stage (problem, model constructors, symmetry data) reads only allowed files (never a bundle, packet or evaluation replica)."""
    import fit71,fit72,fit73
    allowed={str(ROOT/'fit_targets73.pt'),str(INP68/'wl59/geometry_rest.pt'),str(INP68/'wl59/geometry_pt0.pt'),str(INP68/'wl59/P_5901.pt'),
             str(INP68/'wl59/P_5902.pt'),str(R68/'runs/net_all_5902.pt')}|{str(p) for p in (R72/'geom').glob('*.pt')}
    seen=[];orig=torch.load
    def spy(f,*a,**k):seen.append(str(f));return orig(f,*a,**k)
    torch.load=spy
    try:
        pb=fit73.Problem73(RG,'S16R');gs=dict(pb.gs)
        for sp in (dict(candidate='A',init='P_5901'),dict(candidate='B',seed=7201)):fit72.model_for(sp,RG)
        fit72.sym_data(RG,gs,pb.ports,pb.F)
    finally:torch.load=orig
    bad=[s for s in seen if s not in allowed];assert not bad,bad

if __name__=='__main__':
    which=sys.argv[1];T={'pre':[test_candidate_identities,test_population_budget_leakage_seeds,test_symmetry_new_ports,test_adapter_identity,test_runtime_contract_boundary],
       'post':[test_fit_targets73_identity,test_fit_reads_only_allowed]}[which];fails=0
    for t in T:
        try:t();print('PASS',t.__name__,flush=True)
        except Exception as e:fails+=1;print('FAIL',t.__name__,repr(e),flush=True)
    print('TESTS',len(T)-fails,'passed',fails,'failed',flush=True);sys.stdout.flush();os._exit(1 if fails else 0)
