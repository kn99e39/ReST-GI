"""WL71 physical references: unchanged WL66/WL68 estimator and packet gates (probes68.gates/one_hot/domain), new ports and seeds.
preflight -> budget71.json (chunk rule); generate -> refs/packets (receipt per packet); bundle -> refs/bundles/{g}_s{slot}.pt."""
import sys,os,json,time,platform,subprocess
from pathlib import Path
from common71 import CFG,ROOT,NBIN,sha,write_new,save_new,environment,seed,verify_sources
import probes68 as P68

GI={'rest':0,'pt0':1,'cf2':2}
def setup():
    import torch
    RG=environment();import r57_action_oracle as AO,r57_fastwalk as FW
    from tag_reference import instrument_walker,tagged_action
    torch.set_num_threads(4);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
    _,proof=instrument_walker(FW);return RG,AO,FW,tagged_action,proof

def ports():return {int(k):v['port'] for k,v in json.loads((ROOT/'ports.json').read_text())['slots'].items()}

def packets():
    out=[]
    for part in ('training_new','final'):
        c=CFG['references'][part]
        for g in c['geometries']:
            for s in c['slots']:
                for a in range(NBIN):
                    for k in range(c['replicas']):out.append((g,s,a,k,seed(GI[g],s,a,k)))
    assert len({p[-1] for p in out})==len(out),'distinct seeds';return out

def trace(RG,AO,FW,tagged,dom,i,a,N,sd,chunk):
    """One packet at the given chunk; on CUDA OOM at 32M the packet is retried from its own seed at 16M (protocol)."""
    import torch
    import gc
    os.environ['WL57_ACTION_CHUNK']=str(chunk)
    try:return tagged(AO,FW,dom,P68.one_hot(dom.P,i,a),N,sd),chunk
    except torch.cuda.OutOfMemoryError:
        if chunk<=16_000_000:raise
    # amendment 1: retry only after the handler has exited, so the failed attempt's tensors (held by the traceback) are freed
    gc.collect();torch.cuda.empty_cache();os.environ['WL57_ACTION_CHUNK']='16000000'
    return tagged(AO,FW,dom,P68.one_hot(dom.P,i,a),N,sd),16_000_000

def preflight():
    """Non-evidence smoke timing (seeds 719000000+) of 32M and 16M chunks on rest at a final-slot port; chooses the chunk by rule."""
    import torch
    if (ROOT/'refs/budget71.json').exists():raise FileExistsError('immutable')
    RG,AO,FW,tagged,proof=setup();dom=P68.domain(RG,'rest');i=ports()[CFG['references']['final']['slots'][0]];rows={}
    for chunk in (32_000_000,16_000_000):
        torch.cuda.reset_peak_memory_stats();t0=time.time();ok=True
        try:
            for k in range(3):
                os.environ['WL57_ACTION_CHUNK']=str(chunk);r=tagged(AO,FW,dom,P68.one_hot(dom.P,i,k),CFG['references']['packet_photons'],719000000+chunk//1000000*10+k)
                ok&=P68.gates(r,P68.one_hot(dom.P,i,k),r['N'])[0]
        except torch.cuda.OutOfMemoryError:ok=False
        rows[chunk]=dict(seconds_per_packet=(time.time()-t0)/3,peak=torch.cuda.max_memory_allocated(),passed=bool(ok));torch.cuda.empty_cache()
        print('PREFLIGHT',chunk,rows[chunk],flush=True)
    best=32_000_000 if rows[32_000_000]['passed'] and rows[32_000_000]['seconds_per_packet']<rows[16_000_000]['seconds_per_packet'] else 16_000_000
    n=len(packets());write_new(ROOT/'refs/budget71.json',dict(smoke=rows,chunk=best,fallback_16M_on_OOM=best==32_000_000,packets=n,
        predicted_hours=n*rows[best]['seconds_per_packet']/3600,evidence=False))

def manifest(proof):
    import torch
    return dict(scripts={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},protocol_sha256=sha(Path(__file__).parent/'protocol/cover71_v1.json'),
        verified_sources=verify_sources(ROOT/'repo'),AST_equivalence=proof,python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True),host=platform.node())

def generate():
    import torch
    b=json.loads((ROOT/'refs/budget71.json').read_text());RG,AO,FW,tagged,proof=setup();P=ports();N=CFG['references']['packet_photons']
    if 'R71_SMOKE_N' in os.environ:assert ROOT!=Path.home()/'rest_cover71','smoke override never in the evidence root';N=int(os.environ['R71_SMOKE_N'])
    m=ROOT/'refs/manifest.json'
    if not m.exists():write_new(m,manifest(proof))
    doms={}
    for g,s,a,k,sd in packets():
        p=ROOT/'refs/packets'/f'{g}_s{s}_a{a:02d}_k{k}.pt';rc=p.with_suffix('.json')
        if p.exists():
            if rc.exists() and json.loads(rc.read_text())['sha256']==sha(p):continue
            q=ROOT/'refs/quarantine'/str(time.time_ns());q.mkdir(parents=True);os.replace(p,q/p.name)
        if g not in doms:doms.clear();torch.cuda.empty_cache();doms[g]=P68.domain(RG,g)
        r,chunk=trace(RG,AO,FW,tagged,doms[g],P[s],a,N,sd,b['chunk']);x=P68.one_hot(doms[g].P,P[s],a);ok,det=P68.gates(r,x,N)
        keep={k_:r[k_] for k_ in ('N','seed','seconds','chunk','walker','execution_source_sha256','fastwalk_source_sha256','rng_state_sha256','stats',
              'expected_input','output_energy','energy_SE','physical_passed','proposal_sum','importance_ratio_max','peak_GPU_bytes','gates_A_B','event_counts')}
        keep['component_first']={c:r['component_first'][c][0].float() for c in ('Scat','B')};keep['component_second']={c:r['component_second'][c][0] for c in ('Scat',)}
        save_new(p,keep);write_new(rc,dict(sha256=sha(p),geometry=g,slot=s,port=P[s],bin=a,replica=k,seed=sd,N=N,chunk=chunk,seconds=r['seconds'],gates=det,passed=ok,
            energy=r['output_energy'],energy_SE=r['energy_SE'],peak_GPU_bytes=r['peak_GPU_bytes']))
        print('PROBE',g,s,a,k,chunk,round(r['seconds'],2),ok,flush=True)
        if not ok:raise RuntimeError('STOP_PHYSICAL_OR_IDENTITY_GATE; failed packet preserved')

def bundle():
    import torch
    RG=environment();bm=torch.as_tensor(RG.G.bin_parent(8,4));recs=[];P=ports()
    for part in ('training_new','final'):
        c=CFG['references'][part]
        for g in c['geometries']:
            for s in c['slots']:
                S=[];B=[]
                for k in range(c['replicas']):
                    rs=[];rb=[]
                    for a in range(NBIN):
                        p=ROOT/'refs/packets'/f'{g}_s{s}_a{a:02d}_k{k}.pt';rc=json.loads(p.with_suffix('.json').read_text())
                        assert rc['sha256']==sha(p) and rc['passed'] and rc['port']==P[s];recs.append(rc);x=torch.load(p,map_location='cpu',weights_only=False)
                        rs.append(x['component_first']['Scat'].float());bb=x['component_first']['B'].double()
                        rb.append(torch.zeros(bb.shape[0],16,3,dtype=torch.float64).index_add_(1,bm,bb).float())
                    S.append(torch.stack(rs));B.append(torch.stack(rb))
                save_new(ROOT/'refs/bundles'/f'{g}_s{s}.pt',dict(S=torch.stack(S),B16=torch.stack(B),K=c['replicas'],geometry=g,slot=s,port=P[s]))
    write_new(ROOT/'refs/complete.json',dict(packets=len(recs),all_passed=all(r['passed'] for r in recs),photons=sum(r['N'] for r in recs),
        seconds_sum=sum(r['seconds'] for r in recs),chunks={str(c):sum(r['chunk']==c for r in recs) for c in {r['chunk'] for r in recs}},
        bundles={p.name:sha(p) for p in sorted((ROOT/'refs/bundles').glob('*.pt'))}))

if __name__=='__main__':
    {'preflight':preflight,'generate':generate,'bundle':bundle}[sys.argv[1]]();sys.stdout.flush();os._exit(0)
