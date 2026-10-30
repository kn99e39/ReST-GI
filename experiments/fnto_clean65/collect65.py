"""Final Phase-A provenance/bundle. Never enables or executes model training."""
import argparse,json,hashlib,subprocess,sys,tarfile,time,importlib.metadata as metadata
from pathlib import Path

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(16*1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--original',type=Path,required=True);ap.add_argument('--report-commit',required=True);a=ap.parse_args();r=a.root;orig=a.original
    assert not (r/'provenance.json').exists();s=json.loads((r/'status.json').read_text());assert s['state'] in ('READY_FOR_CONTROLLED_TRAINING','STOP_INSUFFICIENT_PRECISION');assert not s['whole_batch_complete'] and not s['phase_B_authorized']
    pre=json.loads((r/'preflight.json').read_text());original={}
    for rel,v in pre['inputs'].items():
        p=Path(v.get('path',orig/rel));assert sha(p)==v['sha256'],rel;original[str(p)]=v['sha256']
    for rel,v in pre['sources'].items():assert sha(orig/'repo'/rel)==v,rel
    for rel,v in json.loads((r/'source_manifest.json').read_text()).items():assert sha(r/'source'/rel)==v,rel
    known={};complete=json.loads((r/'reference_complete.json').read_text());assert complete['photons']==26624000000 and complete['packets']==832
    for row in complete['records']:
        p=r/'halves'/f"{row['pose']}_f{row['field']}_r{row['replica']}_h{row['half']}.pt";assert sha(p)==row['sha256'];known[str(p.relative_to(r))]=dict(bytes=p.stat().st_size,sha256=row['sha256'])
    labels=json.loads((r/'labels_manifest.json').read_text());assert not labels['training_authorized']
    for row in labels['labels']:
        for path,expected in ((row['path'],row['sha256']),(row['variance_path'],row['variance_sha256'])):
            p=Path(path);assert sha(p)==expected;known[str(p.relative_to(r))]=dict(bytes=p.stat().st_size,sha256=expected)
    files={}
    for p in sorted(r.rglob('*')):
        rel=p.relative_to(r)
        if not p.is_file() or any(v in rel.parts for v in ('tmp','__pycache__')) or p.name in ('provenance.json','phaseA_bundle.tar.gz','collect.log'):continue
        files[str(rel)]=known.get(str(rel)) or dict(bytes=p.stat().st_size,sha256=sha(p))
    provenance=dict(phase='A',phaseA_complete=True,whole_batch_complete=False,phase_B_authorized=False,readiness=s['state'],scientific_source_commit=(r/'source_commit.txt').read_text().strip(),report_commit=a.report_commit,original_inputs_rehashed_unchanged=original,original_sources_unchanged=True,frozen_source_unchanged=True,files=files,actual_artifact_bytes=sum(v['bytes'] for v in files.values()),raw_packet_bytes=sum(v['bytes'] for k,v in known.items() if k.startswith('halves/')),label_and_variance_bytes=sum(v['bytes'] for k,v in known.items() if not k.startswith('halves/')),environment=dict(python=sys.version,executable=sys.executable,CPU=subprocess.check_output(['lscpu'],text=True),GPU=subprocess.check_output(['nvidia-smi'],text=True),packages=sorted(f'{d.metadata["Name"]}=={d.version}' for d in metadata.distributions())),GPU_processes_final=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True),location=str(r),completed_epoch=time.time(),future='Explicit post-reset user continuation required; no trainer, model eval or coupled GI launched')
    (r/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    selected=[p for p in r.iterdir() if p.is_file() and p.suffix in ('.json','.log','.txt') and p.name!='collect.log']
    selected.extend(p for folder in ('reduced','review') for p in (r/folder).rglob('*') if p.is_file());selected.append(r/'source_866ce7d.tar.gz')
    with tarfile.open(r/'phaseA_bundle.tar.gz','w:gz') as tf:
        for p in sorted(set(selected)):tf.add(p,arcname=str(p.relative_to(r)))
    print('BUNDLE_COMPLETE',sha(r/'phaseA_bundle.tar.gz'),(r/'phaseA_bundle.tar.gz').stat().st_size,flush=True)
if __name__=='__main__':main()
