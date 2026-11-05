"""WL74 Stage 4 (conditional) references: unchanged estimator, gates, walker and 32M/16M chunk rule (probes73.setup, probes72.trace,
probes68.gates); static slots on rest, footprint slots on rest and the Stage 4 pose; K = 4 replicas; seeds 743/744xxxxxx.
    python probes74.py generate | bundle"""
import sys,os,json,time,socket
from pathlib import Path
from common74 import CFG,ROOT,sha,write_new,save_new,environment,seed,dev
import common68 as C68
import probes68 as P68
import probes72 as PR72

NBIN=16;N=32_000_000;K=4
def ports():return {int(k):v['port'] for k,v in json.loads((ROOT/'ports4.json').read_text())['ports'].items()}
def parts():
    st=[s['slot'] for s in CFG['stage4']['static']];fp=[s['slot'] for s in CFG['stage4']['footprint']];pose=CFG['stage4']['pose']['selected']
    return [('rest',0,st+fp),(pose,1,fp)]
def packets():
    out=[(g,s,a,k,seed(gi,s,a,k)) for g,gi,slots in parts() for s in slots for a in range(NBIN) for k in range(K)]
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
    if 'R74_SMOKE_N' in os.environ:assert dev();n=int(os.environ['R74_SMOKE_N'])
    m=ROOT/'refs/manifest.json'
    if not m.exists():
        write_new(m,dict(walker=info,AST_equivalence=proof,verified_sources=C68.verify_sources(ROOT/'repo'),protocol_sha256=sha(Path(__file__).parent/'protocol/ato74_v1.json'),
            torch=torch.__version__,gpu=torch.cuda.get_device_name(0),host=socket.gethostname(),photons=n,scripts={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}))
    doms={}
    for g,s,a,k,sd in packets():
        p=ROOT/'refs/packets'/f'{g}_s{s}_a{a:02d}_k{k}.pt';rc=p.with_suffix('.json')
        if p.exists():
            if rc.exists() and json.loads(rc.read_text())['sha256']==sha(p):continue
            q=ROOT/'refs/quarantine'/str(time.time_ns());q.mkdir(parents=True);os.replace(p,q/p.name)
        if g not in doms:doms.clear();torch.cuda.empty_cache();doms[g]=P68.domain(RG,g)
        r,chunk=PR72.trace(AO,FW,tagged,doms[g],P[s],a,n,sd,32_000_000);x=P68.one_hot(doms[g].P,P[s],a);ok,det=P68.gates(r,x,n)
        keep={k_:r[k_] for k_ in ('N','seed','seconds','chunk','walker','execution_source_sha256','fastwalk_source_sha256','rng_state_sha256','stats',
              'expected_input','output_energy','energy_SE','physical_passed','proposal_sum','importance_ratio_max','peak_GPU_bytes','gates_A_B','event_counts')}
        keep['component_first']={c:r['component_first'][c][0].float() for c in ('Scat','B')};keep['component_second']={c:r['component_second'][c][0] for c in ('Scat',)}
        save_new(p,keep);write_new(rc,dict(sha256=sha(p),geometry=g,slot=s,port=P[s],bin=a,replica=k,seed=sd,N=n,chunk=chunk,seconds=r['seconds'],gates=det,passed=ok,
            walker=info['walker'],energy=r['output_energy'],energy_SE=r['energy_SE'],peak_GPU_bytes=r['peak_GPU_bytes']))
        print('PROBE',g,s,a,k,chunk,round(r['seconds'],2),ok,flush=True)
        if not ok:raise RuntimeError('STOP_PHYSICAL_OR_IDENTITY_GATE; failed packet preserved')

def bundle():
    import torch
    RG=environment();bm=torch.as_tensor(RG.G.bin_parent(8,4));recs=[];P=ports()
    for g,gi,slots in parts():
        for s in slots:
            S=[];B=[]
            for k in range(K):
                rs=[];rb=[]
                for a in range(NBIN):
                    p=ROOT/'refs/packets'/f'{g}_s{s}_a{a:02d}_k{k}.pt';rc=json.loads(p.with_suffix('.json').read_text())
                    assert rc['sha256']==sha(p) and rc['passed'] and rc['port']==P[s];recs.append(rc);x=torch.load(p,map_location='cpu',weights_only=False)
                    rs.append(x['component_first']['Scat'].float());bb=x['component_first']['B'].double()
                    rb.append(torch.zeros(bb.shape[0],16,3,dtype=torch.float64).index_add_(1,bm,bb).float())
                S.append(torch.stack(rs));B.append(torch.stack(rb))
            save_new(ROOT/'refs/bundles'/f'{g}_s{s}.pt',dict(S=torch.stack(S),B16=torch.stack(B),K=K,geometry=g,slot=s,port=P[s]))
    write_new(ROOT/'refs/complete.json',dict(packets=len(recs),all_passed=all(r['passed'] for r in recs),photons=sum(r['N'] for r in recs),seconds_sum=sum(r['seconds'] for r in recs),
        chunks={str(c):sum(r['chunk']==c for r in recs) for c in {r['chunk'] for r in recs}},walkers=sorted({r['walker'] for r in recs}),
        bundles={p.name:sha(p) for p in sorted((ROOT/'refs/bundles').glob('*.pt'))}))

if __name__=='__main__':
    {'generate':generate,'bundle':bundle}[sys.argv[1]]();sys.stdout.flush();os._exit(0)
