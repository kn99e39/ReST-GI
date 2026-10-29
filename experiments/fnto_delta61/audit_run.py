"""Generate exactly the committed WL61 population with unchanged WL59 reference code."""
import argparse,fcntl,hashlib,json,os,platform,subprocess,sys,time,traceback
from pathlib import Path

HERE=Path(__file__).resolve().parent
CFG=json.loads((HERE/'protocol/audit_v1.json').read_text())

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(16*1024*1024),b''):h.update(b)
    return h.hexdigest()

def write_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(f'Preserve completed evidence: {path}')
    tmp=path.with_suffix(path.suffix+'.partial');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['preflight','generate']);ap.add_argument('--root',type=Path,required=True);ap.add_argument('--original',type=Path,required=True);ap.add_argument('--source-commit',required=True);a=ap.parse_args()
    root=a.root;orig=a.original;repo=orig/'repo';root.mkdir(exist_ok=True,parents=True)
    lock=open(root/'audit.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if a.stage=='generate' and (root/'reference_complete.json').exists():raise FileExistsError('Completed audit is immutable')
    os.environ['P59']=str(orig);os.environ['WL57']=str(orig);os.environ['WL57_CONFIRM_PROTOCOL']=str(repo/'experiments/fnto_rule57/protocol/rule57_recovery_a2.json')
    os.environ['WL57_WALKER']='fast';os.environ['WL57_ACTION_CHUNK']=str(CFG['action_chunk'])
    sys.path.insert(0,str(repo/'experiments/fnto_rule57'))
    import torch,numpy as np
    import r57_geometry as RG
    from r57_action_oracle import action
    torch.set_num_threads(8);torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    inputs={};sources={}
    for rel,expected in CFG['source_expected_sha256'].items():
        p=repo/rel;actual=sha(p);assert actual==expected,f'Original source mismatch: {rel}'
        sources[rel]=actual
    for rel,expected in CFG['input_expected_sha256'].items():
        p=orig/rel;actual=sha(p);assert actual==expected,f'Original input mismatch: {rel}'
        inputs[rel]=dict(sha256=actual,resolved=str(p.resolve()),bytes=p.stat().st_size)
    for rel,expected in CFG['frozen_replay_dependencies'].items():
        p=HERE/'frozen60'/Path(rel).name;assert sha(p)==expected
    assert sha(HERE/'baseline_wl60_replay.json')==CFG['baseline_replay_sha256']
    load=lambda p:torch.load(p,map_location='cpu',weights_only=False)
    lights=load(orig/'data/lights.pt')['train'];fields=lights[CFG['fields']]
    old_lights=load(orig.parent/'rest_wl57/data/lights.pt')['train']
    assert torch.equal(fields,old_lights[CFG['fields']]),'Original train-field lineage mismatch'
    field_hashes={str(f):hashlib.sha256(lights[f].numpy().tobytes()).hexdigest() for f in CFG['fields']}
    meshes={};geometry={}
    parents=[]
    for pose in CFG['poses']:
        g=load(orig/'data/geometry'/f'{pose}.pt');mesh,V=RG.mesh(pose);fm,VC,parent=RG.KO.refined(mesh,V,4)
        vv=np.round(VC,12);vv[np.abs(vv)<.5e-12]=0.
        sig=hashlib.sha256(np.ascontiguousarray(vv).tobytes()).hexdigest();assert sig==g['signature'],f'Geometry mismatch: {pose}'
        assert np.array_equal(parent,g['parent'].numpy());assert fields.shape[1:]==(g['P'],16,3)
        parents.append(g['parent']);meshes[pose]=(fm,VC)
        geometry[pose]=dict(signature=sig,pose_parameters=list(RG.POSES[pose]),P=g['P'],parent_sha256=hashlib.sha256(g['parent'].numpy().tobytes()).hexdigest())
    assert all(torch.equal(parents[0],p) for p in parents[1:]),'Material correspondence differs'
    peers=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True)
    manifest=dict(source_commit=a.source_commit,protocol_sha256=sha(HERE/'protocol/audit_v1.json'),scripts={p.name:sha(p) for p in HERE.glob('*.py')},inputs=inputs,sources=sources,fields=field_hashes,geometry=geometry,material=dict(sigma_t=float(RG.FX.SIGMA_T),albedo=list(map(float,RG.FX.ALBEDO)),phase_g=float(RG.FX.G_PHASE)),python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,gpu=subprocess.check_output(['nvidia-smi'],text=True),external_processes=peers,cpu=subprocess.check_output(['lscpu'],text=True),new_reference=True,estimator='Existing importance-sampled action + validated fast first-exit walk, unchanged')
    if a.stage=='preflight':write_json(root/'preflight.json',manifest);print('PREFLIGHT_GO',flush=True);return
    old=json.loads((root/'preflight.json').read_text())
    for k in ('source_commit','protocol_sha256','inputs','sources','fields','geometry','material'):assert old[k]==manifest[k],f'Preflight identity changed: {k}'
    files=[];start=time.perf_counter();N=CFG['photons_per_half'];total=0
    for pi,pose in enumerate(CFG['poses']):
        fm,VC=meshes[pose];dom=RG.KO.DomainX(fm,VC)
        for fi,field in enumerate(CFG['fields']):
            for rep in range(2):
                for half in range(2):
                    p=root/'halves'/f'{pose}_f{field}_r{rep}_h{half}.pt';receipt=p.with_suffix('.json')
                    if p.exists():
                        assert receipt.exists(),'Partial/orphan evidence: preserve and stop'
                        r=json.loads(receipt.read_text());assert r['sha256']==sha(p) and r['source_commit']==a.source_commit
                        files.append(r);total+=N;continue
                    seed=610000000+pi*10000+field*100+rep*10+half
                    print('HALF_START',pose,field,rep,half,seed,flush=True)
                    result=action(dom,fields[fi],N,seed)
                    assert result['N']==N and result['stats']['photons']==N
                    physical=bool(result['physical_passed'] and torch.isfinite(result['response']).all() and torch.isfinite(result['second']).all() and (result['second']>=0).all())
                    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.pt.partial');torch.save(result,tmp);os.replace(tmp,p)
                    r=dict(path=str(p),sha256=sha(p),pose=pose,field=field,replica=rep,half=half,seed=seed,N=N,seconds=result['seconds'],physical_passed=physical,source_commit=a.source_commit,protocol_sha256=manifest['protocol_sha256'],reference_implementation_sha256=result['execution_source_sha256'],fastwalk_sha256=result['fastwalk_source_sha256'],stats=result['stats'],expected_input=result['expected_input'],output_energy=result['output_energy'],energy_SE=result['energy_SE'])
                    write_json(receipt,r);files.append(r);total+=N
                    print('HALF_DONE',pose,field,rep,half,result['seconds'],physical,flush=True)
                    if not physical:raise RuntimeError('STOP_REFERENCE_PHYSICAL_VALIDATION; preserve failed packet')
        del dom;torch.cuda.empty_cache()
    assert total==CFG['total_photons'] and len(files)==24
    write_json(root/'reference_complete.json',dict(total_photons=total,halves=len(files),physical_passed=all(r['physical_passed'] for r in files),seconds=time.perf_counter()-start,source_commit=a.source_commit,files=files))
    print('REFERENCE_COMPLETE',total,time.perf_counter()-start,flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        traceback.print_exc();raise
