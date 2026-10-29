"""Isolated fixed-budget physical component generation; no model or estimator tuning."""
import argparse,fcntl,json,os,platform,subprocess,sys,time,traceback
from pathlib import Path
from common63 import CFG,HERE,sha,dump_new,reference

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['preflight','generate']);ap.add_argument('--root',type=Path,required=True);ap.add_argument('--original',type=Path,required=True);ap.add_argument('--source-commit',required=True);a=ap.parse_args();root=a.root;orig=a.original;repo=orig/'repo';root.mkdir(exist_ok=True,parents=True)
    lock=open(root/'lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    import torch,numpy as np
    from tag_reference import instrument_walker,tagged_action
    RG,AO,FW=reference(orig);torch.set_num_threads(8);torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    identity={};sources={}
    for rel,expected in CFG['source_expected_sha256'].items():
        assert sha(repo/rel)==expected,rel;sources[rel]=expected
    for rel,expected in CFG['input_expected_sha256'].items():
        p=orig/rel;assert sha(p)==expected,rel;identity[rel]=dict(sha256=expected,bytes=p.stat().st_size,resolved=str(p.resolve()))
    for rel,expected in CFG['frozen_replay_dependencies'].items():assert sha(HERE/'frozen60'/Path(rel).name)==expected
    load=lambda p:torch.load(p,map_location='cpu',weights_only=False)
    lights=load(orig/'data/lights.pt')['test'];field_hashes={str(f):__import__('hashlib').sha256(lights[f].numpy().tobytes()).hexdigest() for f in CFG['fields']};assert field_hashes==CFG['field_sha256']
    g=load(orig/'data/geometry/pt0.pt');fresh=RG.geometry('pt0',ballistic=False)
    for k in ('xyz','rest_xyz','area','node','pair','prior','parent'):assert torch.equal(fresh[k].cpu(),g[k]),k
    mesh,V=RG.mesh('pt0');fm,VC,parent=RG.KO.refined(mesh,V,4);assert np.array_equal(parent,g['parent'].numpy())
    _,astproof=instrument_walker(FW)
    manifest=dict(source_commit=a.source_commit,protocol_sha256=sha(HERE/'protocol/components_v1.json'),scripts={p.name:sha(p) for p in HERE.glob('*.py')},inputs=identity,sources=sources,fields=field_hashes,geometry_signature=g['signature'],geometry_features_bitwise_equal=True,AST_equivalence=astproof,python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,gpu=subprocess.check_output(['nvidia-smi'],text=True),compute_processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True),cpu=subprocess.check_output(['lscpu'],text=True),material=dict(sigma_t=float(RG.FX.SIGMA_T),albedo=list(map(float,RG.FX.ALBEDO)),g=float(RG.FX.G_PHASE)))
    if a.stage=='preflight':dump_new(root/'preflight.json',manifest);print('PREFLIGHT_GO',flush=True);return
    pre=json.loads((root/'preflight.json').read_text())
    for k in ('source_commit','protocol_sha256','inputs','sources','fields','geometry_signature','AST_equivalence'):assert pre[k]==manifest[k],k
    if (root/'reference_complete.json').exists():raise FileExistsError('Completed reference is immutable')
    dom=RG.KO.DomainX(fm,VC);records=[];start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    for field in CFG['fields']:
        for rep in range(2):
            for half in range(2):
                p=root/'halves'/f'pt0_f{field}_r{rep}_h{half}.pt';receipt=p.with_suffix('.json')
                if p.exists():
                    assert receipt.exists();r=json.loads(receipt.read_text());assert r['sha256']==sha(p) and r['source_commit']==a.source_commit;records.append(r);continue
                seed=630000000+field*100+rep*10+half;print('START',field,rep,half,seed,flush=True)
                result=tagged_action(AO,FW,dom,lights[field],CFG['N_per_half'],seed)
                p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.pt.partial');torch.save(result,tmp);os.replace(tmp,p)
                gates=result['gates_A_B'];tol=CFG['gate_A_B_relative_L1']
                gate_ok=all(v<=tol for k,v in gates.items() if not isinstance(v,bool)) and all(v for k,v in gates.items() if isinstance(v,bool))
                physical=bool(result['physical_passed'] and all(torch.isfinite(y).all() and (y>=0).all() for y in result['component_first'].values()) and result['stats']['photons']==CFG['N_per_half'])
                r=dict(path=str(p),sha256=sha(p),field=field,replica=rep,half=half,seed=seed,N=result['N'],chunk=result['chunk'],seconds=result['seconds'],physical_passed=physical,gate_A_B_passed=gate_ok,gates_A_B=gates,source_commit=a.source_commit,reference_source_sha256=result['execution_source_sha256'],original_walker_source_sha256=result['fastwalk_source_sha256'],instrumentation=result['instrumentation'],event_counts=result['event_counts'],rng_state_sha256=result['rng_state_sha256'],stats=result['stats'],peak_GPU_bytes=result['peak_GPU_bytes'])
                dump_new(receipt,r);records.append(r);print('DONE',field,rep,half,result['seconds'],physical,gate_ok,flush=True)
                if not physical or not gate_ok:raise RuntimeError('STOP_PHYSICAL_OR_COMPONENT_GATE; failed evidence preserved')
    assert len(records)==8 and sum(r['N'] for r in records)==CFG['total_photons']
    import resource
    dump_new(root/'reference_complete.json',dict(source_commit=a.source_commit,photons=sum(r['N'] for r in records),physical_passed=all(r['physical_passed'] for r in records),gates_A_B_passed=all(r['gate_A_B_passed'] for r in records),seconds=time.perf_counter()-start,peak_GPU_allocated=torch.cuda.max_memory_allocated(),peak_GPU_reserved=torch.cuda.max_memory_reserved(),CPU_maxrss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,records=records))
    print('REFERENCE_COMPLETE',time.perf_counter()-start,flush=True)
if __name__=='__main__':
    try:main()
    except Exception:traceback.print_exc();raise
