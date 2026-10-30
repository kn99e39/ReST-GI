"""Final provenance and small review bundle; large immutable evidence remains hashed on server."""
import json,hashlib,os,sys,subprocess,time,tarfile
from pathlib import Path
R=Path.home()/'rest_direction64';O=Path.home()/'rest_persist_v1'
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(16*1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    assert json.loads((R/'status.json').read_text())['state']=='COMPLETE';assert (R/'review_complete.json').exists();assert not (R/'provenance.json').exists()
    cal=json.loads((R/'calibration.json').read_text());checked={}
    for rel,expected in cal['identities'].items():
        p=O/rel;actual=sha(p);assert actual==expected,rel;checked[str(p)]=actual
    for rel,expected in cal['sources'].items():assert sha(O/'repo'/rel)==expected,rel
    frozen=json.loads((R/'source_manifest.json').read_text())
    for rel,expected in frozen.items():assert sha(R/'source'/rel)==expected,rel
    comp=json.loads((R/'comparison.json').read_text())
    for arm in ('A','B'):
        for seed in (5901,5902):assert sha(R/'arms'/arm/'checkpoints'/f'P_{seed}.pt')==comp['models'][f'{arm}_{seed}']['checkpoint_sha256']
    ref=json.loads((R/'independent_complete.json').read_text())
    for row in ref['receipts']:assert sha(R/'independent_halves'/f"f{row['field']}_r{row['replica']}_h{row['half']}.pt")==row['sha256']
    files={}
    for p in sorted(R.rglob('*')):
        rel=p.relative_to(R)
        if not p.is_file() or p.is_symlink() or any(k in rel.parts for k in ('tmp','__pycache__','data')):continue
        if p.name in ('provenance.json','review_bundle.tar.gz','collect64.log'):continue
        files[str(rel)]=dict(bytes=p.stat().st_size,sha256=sha(p))
    (R/'provenance.json').write_text(json.dumps(dict(scientific_source_commit='05bb858',calibration_commit='7594e35',report_source_commit=(R/'report_commit.txt').read_text().strip(),branch='Selective_Recompute',original_inputs_rehashed_unchanged=checked,original_sources_unchanged=True,frozen_execution_source_unchanged=True,files=files,large_evidence_location=str(R),complete_epoch=time.time(),final_GPU_processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True),scope='one objective, same model/data/budget; no adoption or automatic successor'),indent=2)+'\n')
    selected=[R/'independent_lights.pt',R/'source_05bb858.tar.gz']
    for p in R.iterdir():
        if p.is_file() and p.suffix in ('.json','.npz','.txt','.log'):selected.append(p)
    for arm in ('A','B'):
        ar=R/'arms'/arm
        for seed in (5901,5902):selected.append(ar/'checkpoints'/f'P_{seed}.pt')
        selected.extend((ar/'out').rglob('*.json'))
    selected.extend(p for p in (R/'review_v1').rglob('*') if p.is_file())
    with tarfile.open(R/'review_bundle.tar.gz','w:gz') as tf:
        for p in sorted(set(selected)):tf.add(p,arcname=str(p.relative_to(R)))
    print('BUNDLE_COMPLETE',sha(R/'review_bundle.tar.gz'),(R/'review_bundle.tar.gz').stat().st_size,flush=True)
if __name__=='__main__':main()
