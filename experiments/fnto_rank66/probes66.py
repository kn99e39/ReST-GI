"""Resource preflight (budget rule) and fixed-budget generation of one-hot localized collided/zero-collision probes."""
import sys,json,time,argparse,platform,subprocess,os,fcntl
from pathlib import Path
import numpy as np
from common66 import CFG,ROOT,ORIG,GEOMS,SLOTS,NBIN,sha,write_new,save_new,environment,verify_sources,seed,ports

def one_hot(P,i,a):
    import torch
    x=torch.zeros(P,NBIN,3);x[i,a,:]=1.;return x

def packet_gates(r,x,N):
    """Physical, normalization and tag-closure gates for one packet; returns (ok, details)."""
    import torch
    tol=CFG['gates']['tag_closure_relative_L1'];g=r['gates_A_B']
    tag=all(v<=tol for v in g.values() if not isinstance(v,bool)) and all(v for v in g.values() if isinstance(v,bool))
    onehot=int((x>0).sum())==3 and abs(r['proposal_sum']-1)<1e-12 and abs(r['importance_ratio_max']-1)<1e-12
    unit=all(abs(v-1.)<1e-12 for v in r['expected_input'][0])
    finite=all(bool(torch.isfinite(y).all() and (y>=0).all()) for y in r['component_first'].values())
    phys=bool(r['physical_passed']) and r['stats']['photons']==N and r['N']==N
    return phys and tag and onehot and unit and finite,dict(physical=phys,tag_closure=tag,one_hot_proposal=onehot,unit_input=unit,finite_nonnegative=finite)

def domain(RG,name):
    mesh,V=RG.mesh(name);fm,VC,parent=RG.KO.refined(mesh,V,4);return RG.KO.DomainX(fm,VC)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['preflight','generate']);ap.add_argument('--source-commit',required=True);a=ap.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True);lock=open(ROOT/'probes.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    import torch
    C59,RG=environment();import r57_action_oracle as AO,r57_fastwalk as FW
    from tag_reference import instrument_walker,tagged_action
    torch.set_num_threads(8);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
    hashes=verify_sources();_,proof=instrument_walker(FW);pr=ports();N=CFG['probes']['packet_photons']
    manifest=dict(source_commit=a.source_commit,protocol_sha256=sha(Path(__file__).parent/'protocol/rank66_v1.json'),
        scripts={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},verified_hashes=hashes,ports_sha256=sha(ROOT/'ports.json'),AST_equivalence=proof,
        python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,
        gpu=subprocess.check_output(['nvidia-smi'],text=True),compute_processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True),
        cpu=subprocess.check_output(['lscpu'],text=True),env={k:os.environ.get(k) for k in ('WL57_WALKER','WL57_ACTION_CHUNK','P53_MI_VARIANT','DRJIT_LIBLLVM_PATH','OMP_NUM_THREADS')})
    if a.stage=='preflight':
        if (ROOT/'preflight.json').exists():raise FileExistsError('preflight is immutable')
        smoke=[]
        for gi,name in enumerate(GEOMS):
            dom=domain(RG,name);i=pr['ports'][0]['port'];x=one_hot(dom.P,i,0);torch.cuda.reset_peak_memory_stats()
            s=689000000+gi;r=tagged_action(AO,FW,dom,x,N,s);ok,det=packet_gates(r,x,N)
            p=ROOT/'preflight'/f'smoke_{name}.pt';save_new(p,r)
            smoke.append(dict(geometry=name,port=i,bin=0,seed=s,N=N,seconds=r['seconds'],peak_GPU_bytes=r['peak_GPU_bytes'],gates=det,passed=ok,sha256=sha(p),bytes=p.stat().st_size,evidence=False))
            print('SMOKE',name,r['seconds'],ok,flush=True)
            if not ok:raise RuntimeError('STOP: preflight packet failed physical/identity gates')
            del dom;torch.cuda.empty_cache()
        t=float(np.mean([s['seconds'] for s in smoke]));probes=len(GEOMS)*len(SLOTS)*NBIN
        pred={k:t*probes*k/3600 for k in (8,4)}
        K=8 if pred[8]<=4 else 4 if pred[4]<=4 else None
        write_new(ROOT/'preflight.json',dict(manifest,smoke=smoke,mean_packet_seconds=t,predicted_hours=pred))
        if K is None:raise RuntimeError('STOP_RESOURCE by predeclared budget rule')
        b=dict(rule=CFG['probes']['budget_rule'],replicas=K,packet_photons=N,probes=probes,packets=probes*K,photons_per_probe=K*N,total_photons=probes*K*N,
               predicted_hours=pred[K],bytes_per_packet=smoke[0]['bytes'],predicted_bytes=smoke[0]['bytes']*probes*K,t_quantile=CFG['t_quantiles'][str(K-1)])
        write_new(ROOT/'budget.json',b);print('BUDGET',json.dumps(b),flush=True);return
    pre=json.loads((ROOT/'preflight.json').read_text());b=json.loads((ROOT/'budget.json').read_text())
    for k in ('source_commit','protocol_sha256','verified_hashes','ports_sha256','AST_equivalence'):assert pre[k]==manifest[k],k
    if (ROOT/'probes_complete.json').exists():raise FileExistsError('Completed reference is immutable')
    K=b['replicas'];records=[];start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    if 'R66_SMOKE_N' in os.environ:
        # Non-evidence smoke harness only: never in the evidence root.
        assert ROOT!=Path.home()/'rest_rank66';N=int(os.environ['R66_SMOKE_N'])
    for gi,name in enumerate(GEOMS):
        dom=domain(RG,name)
        for s in pr['ports']:
            i=s['port']
            for a_ in range(NBIN):
                x=one_hot(dom.P,i,a_)
                for k in range(K):
                    p=ROOT/'probes'/f'{name}_s{s["slot"]}_a{a_:02d}_k{k}.pt';rc=p.with_suffix('.json')
                    if p.exists():
                        if rc.exists():
                            r=json.loads(rc.read_text());assert r['sha256']==sha(p) and r['source_commit']==a.source_commit;records.append(r);continue
                        q=ROOT/'quarantine'/str(time.time_ns());q.mkdir(parents=True);os.replace(p,q/p.name)
                    sd=seed(gi,s['slot'],a_,k);r=tagged_action(AO,FW,dom,x,N,sd);ok,det=packet_gates(r,x,N)
                    save_new(p,r)
                    rec=dict(path=str(p),sha256=sha(p),geometry=name,slot=s['slot'],port=i,bin=a_,replica=k,seed=sd,N=r['N'],seconds=r['seconds'],gates=det,passed=ok,
                             gates_A_B=r['gates_A_B'],event_counts=r['event_counts'],stats=r['stats'],energy=r['output_energy'],energy_SE=r['energy_SE'],
                             rng_state_sha256=r['rng_state_sha256'],peak_GPU_bytes=r['peak_GPU_bytes'],source_commit=a.source_commit,
                             reference_source_sha256=r['execution_source_sha256'],walker_source_sha256=r['fastwalk_source_sha256'])
                    write_new(rc,rec);records.append(rec);print('PROBE',name,s['slot'],a_,k,round(r['seconds'],2),ok,flush=True)
                    if not ok:raise RuntimeError('STOP_PHYSICAL_OR_IDENTITY_GATE; failed packet preserved')
        del dom;torch.cuda.empty_cache()
    assert len(records)==b['packets'] and sum(r['N'] for r in records)==(b['total_photons'] if 'R66_SMOKE_N' not in os.environ else b['packets']*N)
    import resource
    write_new(ROOT/'probes_complete.json',dict(source_commit=a.source_commit,packets=len(records),photons=sum(r['N'] for r in records),all_passed=all(r['passed'] for r in records),
        seconds=time.perf_counter()-start,peak_GPU_allocated=torch.cuda.max_memory_allocated(),peak_GPU_reserved=torch.cuda.max_memory_reserved(),
        CPU_maxrss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,records=records))
    print('PROBES_COMPLETE',time.perf_counter()-start,flush=True)

if __name__=='__main__':main();sys.stdout.flush();os._exit(0)
