"""Held-out physical references on the execution server: cross-machine identity gate, resource preflight, budget rule,
multi-worker generation (packet-level seeds, so results do not depend on worker assignment) and bundling.
Estimator unchanged from WL66: WL57 first-exit importance action, fast walker, WL63 observational tags, one-hot proposal."""
import sys,os,json,time,argparse,platform,subprocess,fcntl
from pathlib import Path
import numpy as np
from common68 import CFG,ROOT,INP,NBIN,sha,write_new,save_new,environment,verify_sources,heldout_seed

def one_hot(P,i,a):
    import torch
    x=torch.zeros(P,NBIN,3);x[i,a,:]=1.;return x

def gates(r,x,N):
    import torch
    tol=1e-12;g=r['gates_A_B']
    tag=all(v<=tol for v in g.values() if not isinstance(v,bool)) and all(v for v in g.values() if isinstance(v,bool))
    onehot=int((x>0).sum())==3 and abs(r['proposal_sum']-1)<1e-12 and abs(r['importance_ratio_max']-1)<1e-12
    unit=all(abs(v-1.)<1e-12 for v in r['expected_input'][0])
    finite=all(bool(torch.isfinite(y).all() and (y>=0).all()) for y in r['component_first'].values())
    phys=bool(r['physical_passed']) and r['stats']['photons']==N and r['N']==N
    return phys and tag and onehot and unit and finite,dict(physical=phys,tag_closure=tag,one_hot_proposal=onehot,unit_input=unit,finite_nonnegative=finite)

def domain(RG,name):
    mesh,V=RG.mesh(name);fm,VC,parent=RG.KO.refined(mesh,V,4);return RG.KO.DomainX(fm,VC)

def packets():
    """Deterministic packet list: (geometry index, geometry, slot, bin, replica, seed)."""
    pr=json.loads((ROOT/'heldout/ports.json').read_text());b=json.loads((ROOT/'heldout/budget.json').read_text());out=[]
    gi={g:k for k,g in enumerate(CFG['heldout_references']['geometry_order'])}
    for g,slots in CFG['populations']['heldout']['matrices'].items():
        for s in slots:
            for a in range(NBIN):
                for k in range(b['replicas']):out.append((gi[g],g,s,a,k,heldout_seed(gi[g],s,a,k)))
    return out

def port_of(slot):
    fit=json.loads((INP/'wl66_ports.json').read_text())['ports'];ho=json.loads((ROOT/'heldout/ports.json').read_text())['ports']
    return {s['slot']:s['port'] for s in fit+ho}[slot]

def setup():
    import torch
    RG=environment();import r57_action_oracle as AO,r57_fastwalk as FW
    from tag_reference import instrument_walker,tagged_action
    torch.set_num_threads(4);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
    _,proof=instrument_walker(FW);return RG,AO,FW,tagged_action,proof

def manifest(proof):
    import torch
    return dict(scripts={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},protocol_sha256=sha(Path(__file__).parent/'protocol/repr68_v1.json'),
        verified_sources=verify_sources(),AST_equivalence=proof,python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,
        gpu=subprocess.check_output(['nvidia-smi'],text=True),host=platform.node(),
        env={k:os.environ.get(k) for k in ('WL57_WALKER','WL57_ACTION_CHUNK','P53_MI_VARIANT','LD_LIBRARY_PATH','OMP_NUM_THREADS')})

def cross_gate():
    """Re-run the predeclared WL66 packets with their original seeds and chunk on this machine; compare with LabServer69."""
    import torch
    out=ROOT/'heldout/cross_machine.json'
    if out.exists():raise FileExistsError('immutable')
    os.environ['WL57_ACTION_CHUNK']='32000000';RG,AO,FW,tagged,proof=setup();rows=[];th=CFG['gates']['cross_machine'];doms={}
    for f in sorted((INP/'cross').glob('*.pt')):
        o=torch.load(f,map_location='cpu',weights_only=False);g,s,a,k=f.stem.split('_');slot=int(s[1:]);b=int(a[1:])
        dom=doms.setdefault(g,domain(RG,g));x=one_hot(dom.P,port_of(slot),b);r=tagged(AO,FW,dom,x,o['N'],o['seed']);ok,det=gates(r,x,o['N'])
        row=dict(packet=f.name,seed=o['seed'],gates=det,passed_gates=ok,seconds=r['seconds'])
        for comp in ('Scat','B'):
            m1,m2=r['component_first'][comp][0],o['component_first'][comp][0];N=o['N']
            v1=((N*r['component_second'][comp][0]-m1.square())/(N-1)).clamp(min=0);v2=((N*o['component_second'][comp][0]-m2.square())/(N-1)).clamp(min=0)
            e1,e2=m1.sum((0,1)),m2.sum((0,1));se=(v1.sum((0,1))+v2.sum((0,1))).sqrt();z=((e1-e2)/se.clamp(min=1e-300)).tolist()
            bm=torch.as_tensor(RG.G.bin_parent(8,4));parent=torch.as_tensor(np.asarray(RG.KO.refined(*RG.mesh(g),4)[2]))
            agg=lambda y:torch.zeros(240,16,3,dtype=y.dtype).index_add(0,parent,torch.zeros(y.shape[0],16,3,dtype=y.dtype).index_add(1,bm,y))
            dm=agg(m1)-agg(m2);dv=agg(v1)+agg(v2);mask=dv>0;chi=float((dm[mask].square()/dv[mask]).mean()) if mask.any() else 0.
            row[comp]=dict(bitwise_equal=bool(torch.equal(m1,m2)),max_abs_diff=float((m1-m2).abs().max()),energy_z=z,coarse_mean_z2=chi,cells=int(mask.sum()))
        row['passed']=ok and all(max(abs(v) for v in row[c]['energy_z'])<=th['energy_z'] and row[c]['coarse_mean_z2']<=th['coarse_mean_z2'] for c in ('Scat','B'))
        rows.append(row);print('CROSS',f.name,row['passed'],row['Scat']['bitwise_equal'],row['Scat']['energy_z'],flush=True)
    res=dict(manifest(proof),packets=rows,passed=all(r['passed'] for r in rows),rule=th);write_new(out,res)
    if not res['passed']:raise RuntimeError('STOP: cross-machine reference identity not verified')

def preflight():
    """Time the predeclared worker configurations with non-evidence smoke packets; fix K and configuration by rule."""
    import torch
    if (ROOT/'heldout/budget.json').exists():raise FileExistsError('budget is immutable')
    cfg=CFG['heldout_references'];smoke=[];n_probe=sum(len(v) for v in CFG['populations']['heldout']['matrices'].values())*NBIN
    for conf in cfg['configurations']:
        procs=[];t0=time.time()
        for w in range(conf['workers']):
            env=dict(os.environ,WL57_ACTION_CHUNK=str(conf['chunk']))
            procs.append(subprocess.Popen([sys.executable,__file__,'smoke',str(conf['workers']),str(w),str(conf['chunk'])],env=env,stdout=subprocess.PIPE,text=True))
        outs=[json.loads(p.communicate()[0].strip().splitlines()[-1]) for p in procs];wall=time.time()-t0
        if any(p.returncode for p in procs):smoke.append(dict(conf,failed=True));continue
        per=[o['seconds'] for o in outs];rate=sum(len(o['packets']) for o in outs)/max(o['span'] for o in outs)
        smoke.append(dict(conf,packets_per_hour=3600*rate,packet_seconds=per,peak_GPU_bytes=[o['peak'] for o in outs],wall=wall,failed=False))
        print('PREFLIGHT',conf,3600*rate,flush=True)
    ok=[s for s in smoke if not s['failed']];best=max(ok,key=lambda s:s['packets_per_hour'])
    pred={K:n_probe*K/best['packets_per_hour'] for K in (8,4)};K=8 if pred[8]<=cfg['max_hours'] else 4 if pred[4]<=cfg['max_hours'] else None
    write_new(ROOT/'heldout/preflight.json',dict(smoke=smoke,probes=n_probe,predicted_hours=pred,chosen=best,evidence=False))
    if K is None:raise RuntimeError('STOP_RESOURCE by predeclared budget rule')
    write_new(ROOT/'heldout/budget.json',dict(rule=cfg['budget_rule'],replicas=K,workers=best['workers'],chunk=best['chunk'],packet_photons=cfg['packet_photons'],
        probes=n_probe,packets=n_probe*K,total_photons=n_probe*K*cfg['packet_photons'],predicted_hours=pred[K]))

def smoke(W,w,chunk):
    """Non-evidence throughput probe: each worker traces its share of 2 rest packets per worker with smoke seeds."""
    import torch
    RG,AO,FW,tagged,proof=setup();dom=domain(RG,'rest');i=port_of(0);rec=[];t0=time.time();torch.cuda.reset_peak_memory_stats()
    for k in range(2):
        s=688000000+w*10+k;r=tagged(AO,FW,dom,one_hot(dom.P,i,(w*2+k)%16),CFG['heldout_references']['packet_photons'],s);rec.append(s)
        assert gates(r,one_hot(dom.P,i,(w*2+k)%16),r['N'])[0]
    print(json.dumps(dict(packets=rec,seconds=time.time()-t0,span=time.time()-t0,peak=torch.cuda.max_memory_allocated())),flush=True)

def generate(w):
    """Worker w of W: packets with list index % W == w. Each packet has its own seed and receipt; failed gate => STOP."""
    import torch
    b=json.loads((ROOT/'heldout/budget.json').read_text())
    am=ROOT/'heldout/execution_amendment.json'
    if am.exists():b.update({k:v for k,v in json.loads(am.read_text()).items() if k in ('workers','chunk')})  # amendment 1 (OOM)
    W=b['workers'];os.environ['WL57_ACTION_CHUNK']=str(b['chunk'])
    RG,AO,FW,tagged,proof=setup();lock=open(ROOT/f'heldout/worker{w}.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    doms={};N=b['packet_photons']
    if 'R68_SMOKE_N' in os.environ:assert ROOT!=Path.home()/'rest_repr68','smoke override never in the evidence root';N=int(os.environ['R68_SMOKE_N'])
    for idx,(gi,g,s,a,k,seed) in enumerate(packets()):
        if idx%W!=w:continue
        p=ROOT/'heldout/packets'/f'{g}_s{s}_a{a:02d}_k{k}.pt';rc=p.with_suffix('.json')
        if p.exists():
            if rc.exists() and json.loads(rc.read_text())['sha256']==sha(p):continue
            q=ROOT/'heldout/quarantine'/str(time.time_ns());q.mkdir(parents=True);os.replace(p,q/p.name)
        if g not in doms:doms.clear();torch.cuda.empty_cache();doms[g]=domain(RG,g)
        x=one_hot(doms[g].P,port_of(s),a);r=tagged(AO,FW,doms[g],x,N,seed);ok,det=gates(r,x,N)
        keep={k_:r[k_] for k_ in ('N','seed','seconds','chunk','walker','execution_source_sha256','fastwalk_source_sha256','rng_state_sha256','stats',
              'expected_input','output_energy','energy_SE','physical_passed','proposal_sum','importance_ratio_max','peak_GPU_bytes','gates_A_B','event_counts')}
        keep['component_first']={c:r['component_first'][c][0].float() for c in ('Scat','B')};keep['component_second']={c:r['component_second'][c][0] for c in ('Scat',)}
        save_new(p,keep);write_new(rc,dict(sha256=sha(p),geometry=g,slot=s,port=port_of(s),bin=a,replica=k,seed=seed,N=N,seconds=r['seconds'],gates=det,passed=ok,worker=w,
            energy=r['output_energy'],energy_SE=r['energy_SE'],peak_GPU_bytes=r['peak_GPU_bytes']))
        print('PROBE',g,s,a,k,round(r['seconds'],2),ok,flush=True)
        if not ok:raise RuntimeError('STOP_PHYSICAL_OR_IDENTITY_GATE; failed packet preserved')

def bundle():
    """Verify every receipt and stack per (geometry, slot): S (K,16,P,64,3), B16 (K,16,P,16,3), float32."""
    import torch
    RG=environment();bm=torch.as_tensor(RG.G.bin_parent(8,4));b=json.loads((ROOT/'heldout/budget.json').read_text());K=b['replicas'];recs=[]
    for g,slots in CFG['populations']['heldout']['matrices'].items():
        for s in slots:
            S=[];B=[]
            for k in range(K):
                rs=[];rb=[]
                for a in range(NBIN):
                    p=ROOT/'heldout/packets'/f'{g}_s{s}_a{a:02d}_k{k}.pt';rc=json.loads(p.with_suffix('.json').read_text())
                    assert rc['sha256']==sha(p) and rc['passed'];recs.append(rc);x=torch.load(p,map_location='cpu',weights_only=False)
                    rs.append(x['component_first']['Scat'].float());bb=x['component_first']['B'].double()
                    rb.append(torch.zeros(bb.shape[0],16,3,dtype=torch.float64).index_add_(1,bm,bb).float())
                S.append(torch.stack(rs));B.append(torch.stack(rb))
            save_new(ROOT/'heldout/bundles'/f'{g}_s{s}.pt',dict(S=torch.stack(S),B16=torch.stack(B),K=K,geometry=g,slot=s))
    write_new(ROOT/'heldout/complete.json',dict(packets=len(recs),all_passed=all(r['passed'] for r in recs),photons=sum(r['N'] for r in recs),
        seconds_sum=sum(r['seconds'] for r in recs),bundles={p.name:sha(p) for p in sorted((ROOT/'heldout/bundles').glob('*.pt'))}))

if __name__=='__main__':
    a=sys.argv[1:]
    {'cross':lambda:cross_gate(),'preflight':lambda:preflight(),'smoke':lambda:smoke(int(a[1]),int(a[2]),int(a[3])),
     'generate':lambda:generate(int(a[1])),'bundle':lambda:bundle()}[a[0]]();sys.stdout.flush();os._exit(0)
