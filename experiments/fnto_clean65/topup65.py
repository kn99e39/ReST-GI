"""Gate 1: identity preflight and exactly 624 fixed top-up packets; Phase A read-only, no model execution."""
import argparse,os,sys,time,json,shutil,fcntl,datetime,traceback
from pathlib import Path
from ext65 import CFG,CFG1,TOP,PHASE_A,ORIG,HOME,sha,write_new,save_new,packets,topup_seed

def expand(p):return Path(p.replace('~',str(HOME),1))

def identities(orig):
    """Re-verify Phase A evidence and every original/external identity named by v1 and v2."""
    out=dict(phaseA_artifacts={},phaseA_packets=0,phaseA_banks={},sources={},inputs={},external={})
    for rel,expected in CFG['phase_A']['sha256'].items():
        assert sha(PHASE_A/rel)==expected,rel;out['phaseA_artifacts'][rel]=expected
    receipts={(v['pose'],v['field'],v['replica'],v['half']):v for v in json.loads((PHASE_A/'reference_complete.json').read_text())['records']}
    assert len(receipts)==CFG['phase_A']['packets']
    for (pose,field,rep,half),v in receipts.items():
        p=PHASE_A/'halves'/f'{pose}_f{field}_r{rep}_h{half}.pt';assert sha(p)==v['sha256'],p;assert v['physical'] and v['N']==CFG1['N_per_half']
    out['phaseA_packets']=len(receipts)
    for row in json.loads((PHASE_A/'labels_manifest.json').read_text())['labels']:
        for k in ('','variance_'):
            p=Path(row[k+'path']);assert sha(p)==row[k+'sha256'],p;out['phaseA_banks'][str(p)]=row[k+'sha256']
    for rel,expected in CFG['source_expected_sha256'].items():
        assert sha(orig/'repo'/rel)==expected,rel;out['sources'][rel]=expected
    for rel,expected in CFG['input_expected_sha256'].items():
        assert sha(orig/rel)==expected,rel;out['inputs'][rel]=expected
    for rel,expected in CFG['external_expected_sha256'].items():
        assert sha(expand(rel))==expected,rel;out['external'][rel]=expected
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['preflight','generate']);ap.add_argument('--root',type=Path,required=True);ap.add_argument('--source-commit',required=True);a=ap.parse_args();r=a.root;r.mkdir(parents=True,exist_ok=True)
    lock=open(r/'topup.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    import torch
    from run65 import manifest
    from moments65 import observed_action
    m,RG,AO,lights=manifest(ORIG)
    old=json.loads((PHASE_A/'preflight.json').read_text())
    # Same physical semantics as Phase A: identical source/input/field/geometry/material manifest.
    for k in ('protocol_sha256','sources','inputs','all96_train_field_sha256','selected_field_proposals','geometries','material'):assert old[k]==m[k],k
    free=shutil.disk_usage(r).free;assert free>=CFG['resource']['require_free_GiB']*1024**3
    if a.stage=='preflight':
        ids=identities(ORIG)
        seeds=[topup_seed(pi,f,rep,k) for pi in range(13) for f in TOP['fields'] for rep in range(2) for k in range(6)]
        old_seeds={v['seed'] for v in json.loads((PHASE_A/'reference_complete.json').read_text())['records']}
        assert len(set(seeds))==TOP['total_packets'] and not old_seeds.intersection(seeds)
        write_new(r/'topup_preflight.json',dict(source_commit=a.source_commit,protocol_v2_sha256=sha(Path(__file__).parent/'protocol/topup_ab_v2.json'),phaseA_manifest_keys_identical=True,identities=ids,manifest={k:m[k] for k in ('protocol_sha256','scripts','sources','inputs','geometries','material','environment','external_GPU_processes')},
            selected_field_proposals=[v for v in m['selected_field_proposals'] if v['field'] in TOP['fields']],free_bytes=free,seed_count=len(seeds),seeds_disjoint_from_phaseA=True))
        print('TOPUP_PREFLIGHT_GO',flush=True);return
    pre=json.loads((r/'topup_preflight.json').read_text());assert pre['source_commit']==a.source_commit
    for k in ('protocol_sha256','sources','inputs','geometries','material'):assert pre['manifest'][k]==m[k],k
    assert not (r/'topup_complete.json').exists();records=[];t0=time.perf_counter()
    for pi,pose in enumerate(TOP['poses']):
        mesh,V=RG.mesh(pose);fm,VC,parent=RG.KO.refined(mesh,V,4);dom=RG.KO.DomainX(fm,VC)
        for field in TOP['fields']:
            for kind,rep,p,N,seed in packets(pose,field,root_t=r):
                if kind!='topup':continue
                assert shutil.disk_usage(r).free>20*1024**3,'Storage gate'
                receipt=p.with_suffix('.json');k=int(p.stem.split('_k')[1]);assert seed==topup_seed(pi,field,rep,k)
                if p.exists():
                    assert receipt.exists();row=json.loads(receipt.read_text());assert sha(p)==row['sha256'] and row['seed']==seed and row['source_commit']==a.source_commit;records.append(row);continue
                torch.cuda.reset_peak_memory_stats();started=time.perf_counter();value=observed_action(AO,dom,lights[field],N,seed)
                physical=bool(value['physical_passed'] and all(value['observer_checks'].values()) and all(torch.isfinite(value[q]).all() and (value[q]>=0).all() for q in ('response','first64','second')) and value['N']==N and value['stats']['photons']==N)
                save_new(p,value);peak=torch.cuda.max_memory_allocated()
                row=dict(pose=pose,field=field,replica=rep,k=k,seed=seed,N=value['N'],sha256=sha(p),bytes=p.stat().st_size,walker_seconds=value['seconds'],wall_seconds=time.perf_counter()-started,physical=physical,observer_checks=value['observer_checks'],stats=value['stats'],peak_GPU_bytes=peak,rng_sha256=value['rng_state_sha256'],source_commit=a.source_commit,reference_source_sha256=value['execution_source_sha256'],walker_source_sha256=value['fastwalk_source_sha256'])
                write_new(receipt,row);records.append(row)
                (r/'progress.json').write_text(json.dumps(dict(stage='topup',packets=len(records),photons=sum(v['N'] for v in records),pose=pose,field=field,elapsed=time.perf_counter()-t0)))
                print('PACKET',len(records),pose,field,rep,k,round(row['wall_seconds'],3),physical,flush=True)
                if not physical:raise RuntimeError('STOP_INVALID_REFERENCE')
                if peak>CFG['resource']['GPU_limit_GiB']*1024**3:raise RuntimeError('STOP_RESOURCE_OR_SCOPE: GPU memory gate')
                del value
        del dom;torch.cuda.empty_cache()
    assert len(records)==TOP['total_packets'] and sum(v['N'] for v in records)==TOP['total_photons']
    phaseA_sources={(v['reference_source_sha256'],v['walker_source_sha256']) for v in json.loads((PHASE_A/'reference_complete.json').read_text())['records']}
    assert {(v['reference_source_sha256'],v['walker_source_sha256']) for v in records}==phaseA_sources,'Top-up estimator source differs from Phase A'
    write_new(r/'topup_complete.json',dict(source_commit=a.source_commit,packets=len(records),photons=sum(v['N'] for v in records),seconds=time.perf_counter()-t0,physical=True,estimator_source_identical_to_phaseA=True,records=records));print('TOPUP_COMPLETE',flush=True)
if __name__=='__main__':
    try:main()
    except Exception:traceback.print_exc();raise
