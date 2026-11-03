"""WL72 confirmation references: unchanged WL66/WL68/WL71 estimator, packet gates and OOM-fallback trace (probes71/probes68),
new ports (slots 40-47) on rest and te2, seeds 723xxxxxx. Walker: WL57 fast walker unless the validated WL71C rework is present.
    python probes72.py preflight | generate | bundle"""
import sys,os,json,time,importlib.util
from pathlib import Path
from common72 import CFG,ROOT,sha,write_new,save_new,environment,seed
import probes68 as P68

GI={'rest':0,'te2':1};NBIN=16;N=32_000_000;K=4;HOME=Path.home()
def ports():return {int(k):v['port'] for k,v in json.loads((ROOT/'ports.json').read_text())['slots'].items()}
def packets():
    out=[(g,s['slot'],a,k,seed(GI[g],s['slot'],a,k)) for g in ('rest','te2') for s in CFG['confirmation']['slots'] for a in range(NBIN) for k in range(K)]
    assert len({p[-1] for p in out})==len(out);return out

def walker():
    """Validated WL71C walker if its record says PASSED at generation start, else the WL57 fast walker (recorded)."""
    v=HOME/'rest_walker71c/validated.json'
    if v.exists():
        rec=json.loads(v.read_text())
        if rec.get('status')=='PASSED':
            spec=importlib.util.spec_from_file_location('r57_fastwalk',rec['module']);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
            return mod,dict(walker='WL71C',module=rec['module'],module_sha256=sha(rec['module']),record_sha256=sha(v))
    import r57_fastwalk as FW
    return FW,dict(walker='WL57 fast walker (unchanged)',module=FW.__file__,module_sha256=sha(FW.__file__))

def setup():
    import torch
    RG=environment();import r57_action_oracle as AO
    from tag_reference import instrument_walker,tagged_action
    FW,info=walker();torch.set_num_threads(4);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
    _,proof=instrument_walker(FW);return RG,AO,FW,tagged_action,proof,info

def trace(AO,FW,tagged,dom,i,a,n,sd,chunk):
    import torch,gc
    os.environ['WL57_ACTION_CHUNK']=str(chunk)
    try:return tagged(AO,FW,dom,P68.one_hot(dom.P,i,a),n,sd),chunk
    except torch.cuda.OutOfMemoryError:
        if chunk<=16_000_000:raise
    gc.collect();torch.cuda.empty_cache();os.environ['WL57_ACTION_CHUNK']='16000000'
    return tagged(AO,FW,dom,P68.one_hot(dom.P,i,a),n,sd),16_000_000

def generate():
    import torch
    RG,AO,FW,tagged,proof,info=setup();P=ports();n=N
    if 'R72_SMOKE_N' in os.environ:assert ROOT!=HOME/'rest_cmp72','smoke override never in the evidence root';n=int(os.environ['R72_SMOKE_N'])
    m=ROOT/'refs/manifest.json'
    if not m.exists():
        write_new(m,dict(walker=info,AST_equivalence=proof,protocol_sha256=sha(Path(__file__).parent/'protocol/cmp72_v1.json'),torch=torch.__version__,
            scripts={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}))
    doms={}
    for g,s,a,k,sd in packets():
        p=ROOT/'refs/packets'/f'{g}_s{s}_a{a:02d}_k{k}.pt';rc=p.with_suffix('.json')
        if p.exists():
            if rc.exists() and json.loads(rc.read_text())['sha256']==sha(p):continue
            q=ROOT/'refs/quarantine'/str(time.time_ns());q.mkdir(parents=True);os.replace(p,q/p.name)
        if g not in doms:doms.clear();torch.cuda.empty_cache();doms[g]=P68.domain(RG,g)
        r,chunk=trace(AO,FW,tagged,doms[g],P[s],a,n,sd,32_000_000);x=P68.one_hot(doms[g].P,P[s],a);ok,det=P68.gates(r,x,n)
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
    for g in ('rest','te2'):
        for s in CFG['confirmation']['slots']:
            s=s['slot'];S=[];B=[]
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
