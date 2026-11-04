"""WL73 physical references: unchanged WL66/68/71/72 estimator (tagged first-exit action of the unchanged WL57 fast walker), packet
gates and 32M -> 16M OOM fallback (probes72.trace). Training: S16-R replacement slots 50-53 on rest and pt0, 8 replicas (fit 0-3,
evaluation 4-7, as S16). Confirmation: static slots 60-75 on rest, footprint slots 80-85 on rest and te0, 4 replicas.
    python probes73.py generate | bundle"""
import sys,os,json,time,socket
from pathlib import Path
from common73 import CFG,ROOT,sha,write_new,save_new,environment,seed,dev
import common68 as C68
import probes68 as P68
import probes72 as PR72

NBIN=16;N=32_000_000

def ports():
    r=json.loads((ROOT/'ports.json').read_text());P={int(k):v['port'] for k,v in r['new'].items()};P.update({int(k):v['port'] for k,v in r['confirmation'].items()});return P

def parts():
    """(kind, geometry, geometry index, slots, replicas) in generation order."""
    fz=CFG['frozen'];new=sorted(int(s) for s in fz['new_ports']);conf=sorted(int(s) for s in fz['confirmation_ports'])
    foot=[s['slot'] for s in CFG['stage0']['footprint']['ports']];static=[s for s in conf if s not in foot];pose=CFG['stage0']['deformation_pose']['selected']
    return [('train','rest',0,new,8),('train','pt0',1,new,8),('conf','rest',0,static+foot,4),('conf',pose,1,foot,4)]

def packets():
    out=[(kind,g,s,a,k,seed(kind,gi,s,a,k)) for kind,g,gi,slots,K in parts() for s in slots for a in range(NBIN) for k in range(K)]
    assert len({p[-1] for p in out})==len(out);return out

def setup():
    import torch
    RG=environment();import r57_action_oracle as AO
    import r57_fastwalk as FW
    from tag_reference import instrument_walker,tagged_action
    torch.set_num_threads(4);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
    _,proof=instrument_walker(FW);return RG,AO,FW,tagged_action,proof,dict(walker='WL57 fast walker (unchanged)',module=FW.__file__,module_sha256=sha(FW.__file__))

def generate():
    import torch
    RG,AO,FW,tagged,proof,info=setup();P=ports();n=N
    if 'R73_SMOKE_N' in os.environ:assert dev(),'smoke override never in the evidence root';n=int(os.environ['R73_SMOKE_N'])
    m=ROOT/'refs/manifest.json'
    if not m.exists():
        write_new(m,dict(walker=info,AST_equivalence=proof,verified_sources=C68.verify_sources(ROOT/'repo'),protocol_sha256=sha(Path(__file__).parent/'protocol/src73_v1.json'),
            torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),host=socket.gethostname(),photons=n,
            scripts={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}))
    doms={}
    for kind,g,s,a,k,sd in packets():
        p=ROOT/'refs/packets'/f'{g}_s{s}_a{a:02d}_k{k}.pt';rc=p.with_suffix('.json')
        if p.exists():
            if rc.exists() and json.loads(rc.read_text())['sha256']==sha(p):continue
            q=ROOT/'refs/quarantine'/str(time.time_ns());q.mkdir(parents=True);os.replace(p,q/p.name)
        if g not in doms:doms.clear();torch.cuda.empty_cache();doms[g]=P68.domain(RG,g)
        r,chunk=PR72.trace(AO,FW,tagged,doms[g],P[s],a,n,sd,32_000_000);x=P68.one_hot(doms[g].P,P[s],a);ok,det=P68.gates(r,x,n)
        keep={k_:r[k_] for k_ in ('N','seed','seconds','chunk','walker','execution_source_sha256','fastwalk_source_sha256','rng_state_sha256','stats',
              'expected_input','output_energy','energy_SE','physical_passed','proposal_sum','importance_ratio_max','peak_GPU_bytes','gates_A_B','event_counts')}
        keep['component_first']={c:r['component_first'][c][0].float() for c in ('Scat','B')};keep['component_second']={c:r['component_second'][c][0] for c in ('Scat',)}
        save_new(p,keep);write_new(rc,dict(sha256=sha(p),kind=kind,geometry=g,slot=s,port=P[s],bin=a,replica=k,seed=sd,N=n,chunk=chunk,seconds=r['seconds'],gates=det,passed=ok,
            walker=info['walker'],energy=r['output_energy'],energy_SE=r['energy_SE'],peak_GPU_bytes=r['peak_GPU_bytes']))
        print('PROBE',kind,g,s,a,k,chunk,round(r['seconds'],2),ok,flush=True)
        if not ok:raise RuntimeError('STOP_PHYSICAL_OR_IDENTITY_GATE; failed packet preserved')

def bundle(which):
    import torch
    RG=environment();bm=torch.as_tensor(RG.G.bin_parent(8,4));recs=[];P=ports()
    for kind,g,gi,slots,K in parts():
        if kind!=which:continue
        for s in slots:
            S=[];B=[]
            for k in range(K):
                rs=[];rb=[]
                for a in range(NBIN):
                    p=ROOT/'refs/packets'/f'{g}_s{s}_a{a:02d}_k{k}.pt';rc=json.loads(p.with_suffix('.json').read_text())
                    assert rc['sha256']==sha(p) and rc['passed'] and rc['port']==P[s] and rc['kind']==kind;recs.append(rc);x=torch.load(p,map_location='cpu',weights_only=False)
                    rs.append(x['component_first']['Scat'].float());bb=x['component_first']['B'].double()
                    rb.append(torch.zeros(bb.shape[0],16,3,dtype=torch.float64).index_add_(1,bm,bb).float())
                S.append(torch.stack(rs));B.append(torch.stack(rb))
            save_new(ROOT/'refs/bundles'/f'{g}_s{s}.pt',dict(S=torch.stack(S),B16=torch.stack(B),K=K,geometry=g,slot=s,port=P[s]))
    write_new(ROOT/f'refs/complete_{which}.json',dict(packets=len(recs),all_passed=all(r['passed'] for r in recs),photons=sum(r['N'] for r in recs),seconds_sum=sum(r['seconds'] for r in recs),
        chunks={str(c):sum(r['chunk']==c for r in recs) for c in {r['chunk'] for r in recs}},walkers=sorted({r['walker'] for r in recs}),
        bundles={p.name:sha(p) for p in sorted((ROOT/'refs/bundles').glob('*.pt')) if any(p.name==f'{g}_s{s}.pt' for kind,g,gi,sl,K in parts() if kind==which for s in sl)}))

if __name__=='__main__':
    {'generate':generate,'bundle':lambda:bundle(sys.argv[2])}[sys.argv[1]]();sys.stdout.flush();os._exit(0)
