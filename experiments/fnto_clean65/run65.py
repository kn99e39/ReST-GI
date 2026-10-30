"""Phase A identity/resource preflight and exactly832 independent packets; no model execution."""
import argparse,os,sys,time,json,hashlib,platform,subprocess,shutil,fcntl,datetime,traceback
from pathlib import Path
from common65 import CFG,HERE,sha,write_new,save_new,reference

def manifest(orig):
    import torch,numpy as np
    RG,AO=reference(orig);torch.set_num_threads(8);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert RG.TRAIN==CFG['poses'];sources={};inputs={}
    for rel,expected in CFG['source_expected_sha256'].items():assert sha(orig/'repo'/rel)==expected,rel;sources[rel]=expected
    for rel,expected in CFG['input_expected_sha256'].items():
        p=orig/rel;assert sha(p)==expected,rel;inputs[rel]=dict(sha256=expected,bytes=p.stat().st_size,resolved=str(p.resolve()))
    lights=torch.load(orig/'data/lights.pt')['train'];old=torch.load(orig.parent/'rest_wl57/data/lights.pt')['train'];assert torch.equal(lights,old);assert len(lights)==96
    inputs['lineage_lights']=dict(path=str(orig.parent/'rest_wl57/data/lights.pt'),sha256=sha(orig.parent/'rest_wl57/data/lights.pt'))
    oldr1=orig.parent/'rest_wl57/data/targets/rest_r1.pt';assert sha(oldr1)==CFG['old_rest_r1_sha256'];inputs['old_rest_r1']=dict(path=str(oldr1),sha256=sha(oldr1))
    field_hash={str(f):hashlib.sha256(lights[f].numpy().tobytes()).hexdigest() for f in range(96)};selected=[]
    for p in range(8):assert torch.allclose(lights[80+2*p].sum(1),lights[81+2*p].sum(1),atol=1e-8,rtol=1e-6);assert not torch.equal(lights[80+2*p],lights[81+2*p])
    for f in CFG['fields']:
        x=lights[f].double().reshape(-1,3);q=x.mean(1);q=q/q.sum();prob=x/x.sum(0);z=q>0;ess=1/(prob[z].square()/q[z,None]).sum(0)
        selected.append(dict(field=f,source_ESS_fraction_RGB=ess.tolist(),per_half_source_ESS_RGB=(ess*CFG['N_per_half']).tolist(),max_source_importance_ratio=float((x[z]/q[z,None]).max()),support_cells=int(z.sum())))
    geometries={};parents=[]
    for pose in CFG['poses']:
        g=torch.load(orig/'data/geometry'/f'{pose}.pt');m,V=RG.mesh(pose);fm,VC,parent=RG.KO.refined(m,V,4);raw=hashlib.sha256(np.ascontiguousarray(VC).tobytes()).hexdigest();canonical=np.round(VC,12);canonical[abs(canonical)<.5e-12]=0.;cs=hashlib.sha256(np.ascontiguousarray(canonical).tobytes()).hexdigest()
        assert raw==g['signature'],pose;fresh=RG.geometry(pose,ballistic=False);assert fresh['signature']==cs
        for k in ('xyz','rest_xyz','area','node','pair','prior','parent'):assert torch.equal(fresh[k].cpu(),g[k]),(pose,k)
        assert np.array_equal(parent,g['parent'].numpy());assert lights.shape[1:]==(g['P'],16,3);parents.append(g['parent'])
        geometries[pose]=dict(raw_signature=raw,canonical_signature=cs,derived_geometry_bitwise=True,P=g['P'],parent_sha256=hashlib.sha256(g['parent'].numpy().tobytes()).hexdigest(),pose_parameters=list(RG.POSES[pose]));del g,fresh
    assert all(torch.equal(parents[0],v) for v in parents)
    external=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True);peers=[s for s in external.splitlines() if s.strip() and int(s.split(',')[0])!=os.getpid()];assert not peers,peers
    return dict(protocol_sha256=sha(HERE/'protocol/clean_v1.json'),scripts={p.name:sha(p) for p in HERE.glob('*.py')},sources=sources,inputs=inputs,all96_train_field_sha256=field_hash,selected_field_proposals=selected,geometries=geometries,material=dict(sigma_t=float(RG.FX.SIGMA_T),albedo=list(map(float,RG.FX.ALBEDO)),phase_g=float(RG.FX.G_PHASE)),environment=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,gpu=subprocess.check_output(['nvidia-smi'],text=True),CPU=subprocess.check_output(['lscpu'],text=True)),external_GPU_processes=peers),RG,AO,lights

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['preflight','generate']);ap.add_argument('--root',type=Path,required=True);ap.add_argument('--original',type=Path,required=True);ap.add_argument('--source-commit',required=True);a=ap.parse_args();r=a.root;r.mkdir(parents=True,exist_ok=True)
    lock=open(r/'reference.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    import torch
    from moments65 import observed_action
    m,RG,AO,lights=manifest(a.original)
    free=shutil.disk_usage(r).free;assert free>=CFG['resource']['require_free_GiB']*1024**3
    if a.stage=='preflight':
        deadline=datetime.datetime.fromisoformat(CFG['resource']['deadline_UTC']);hours=(deadline-datetime.datetime.now(datetime.timezone.utc)).total_seconds()/3600;assert hours>2
        m.update(source_commit=a.source_commit,free_bytes=free,deadline_hours_remaining=hours,feasibility=dict(expected_execution_minutes=CFG['resource']['expected_minutes'],raw_packet_bytes=CFG['resource']['estimated_raw_packet_bytes'],label_plus_variance_bytes=CFG['resource']['estimated_labels_plus_variance_bytes'],GPU_peak_estimate_GiB=16,precision_estimate='Coarse absolute precision expected sub-percent; fine16 absolute a few percent for broad fields. Weak contrast/delta precision not guaranteed: fixed-budget gates must measure it. Source ESS is diagnostic, not exit-action ESS. No batching or N dilution.'))
        write_new(r/'preflight.json',m);print('PREFLIGHT_GO',flush=True);return
    old=json.loads((r/'preflight.json').read_text())
    for k in ('protocol_sha256','scripts','sources','inputs','all96_train_field_sha256','selected_field_proposals','geometries','material'):assert old[k]==m[k],k
    assert not (r/'reference_complete.json').exists();records=[];t0=time.perf_counter()
    for pi,pose in enumerate(CFG['poses']):
        mesh,V=RG.mesh(pose);fm,VC,parent=RG.KO.refined(mesh,V,4);dom=RG.KO.DomainX(fm,VC)
        for field in CFG['fields']:
            for rep in range(2):
                for half in range(2):
                    assert datetime.datetime.now(datetime.timezone.utc)<datetime.datetime.fromisoformat(CFG['resource']['deadline_UTC']),'Deadline reached: preserve partial outputs'
                    assert shutil.disk_usage(r).free>20*1024**3,'Storage gate'
                    p=r/'halves'/f'{pose}_f{field}_r{rep}_h{half}.pt';receipt=p.with_suffix('.json');seed=650000000+pi*100000+field*100+rep*10+half
                    if p.exists():
                        assert receipt.exists();row=json.loads(receipt.read_text());assert sha(p)==row['sha256'] and row['seed']==seed and row['source_commit']==a.source_commit;records.append(row);continue
                    torch.cuda.reset_peak_memory_stats();started=time.perf_counter();value=observed_action(AO,dom,lights[field],CFG['N_per_half'],seed)
                    physical=bool(value['physical_passed'] and all(value['observer_checks'].values()) and all(torch.isfinite(value[k]).all() and (value[k]>=0).all() for k in ('response','first64','second')) and value['N']==CFG['N_per_half'] and value['stats']['photons']==CFG['N_per_half'])
                    save_new(p,value);peak=torch.cuda.max_memory_allocated();row=dict(pose=pose,field=field,replica=rep,half=half,seed=seed,N=value['N'],sha256=sha(p),bytes=p.stat().st_size,walker_seconds=value['seconds'],wall_seconds=time.perf_counter()-started,physical=physical,observer_checks=value['observer_checks'],stats=value['stats'],peak_GPU_bytes=peak,rng_sha256=value['rng_state_sha256'],source_commit=a.source_commit,reference_source_sha256=value['execution_source_sha256'],walker_source_sha256=value['fastwalk_source_sha256']);write_new(receipt,row);records.append(row)
                    status=dict(stage='generate',packets=len(records),photons=sum(v['N'] for v in records),pose=pose,field=field,elapsed=time.perf_counter()-t0);(r/'progress.json').write_text(json.dumps(status))
                    print('PACKET',len(records),pose,field,rep,half,round(row['wall_seconds'],3),physical,flush=True)
                    if not physical:raise RuntimeError('STOP_INVALID_REFERENCE')
                    if peak>CFG['resource']['execution_GiB_limit']*1024**3:raise RuntimeError('STOP_RESOURCE_OR_SCOPE: GPU memory gate')
                    del value
        del dom;torch.cuda.empty_cache()
    assert len(records)==832 and sum(v['N'] for v in records)==26624000000
    write_new(r/'reference_complete.json',dict(source_commit=a.source_commit,packets=len(records),photons=sum(v['N'] for v in records),seconds=time.perf_counter()-t0,physical=True,records=records));print('REFERENCE_COMPLETE',flush=True)
if __name__=='__main__':
    try:main()
    except Exception:traceback.print_exc();raise
