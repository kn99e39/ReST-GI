"""Export provenance and verify immutable inputs without importing GPU code."""
import datetime
import hashlib
import json
from pathlib import Path
import sys


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(root,out):
    root,out=Path(root),Path(out)
    code=Path(__file__).resolve().parent
    repository=code.parent.parent
    after_rows=(root/'frozen_model_hashes_after.txt').read_text(encoding='utf-8-sig').strip().splitlines()
    after={r.split(None,1)[1].strip():r.split(None,1)[0] for r in after_rows}
    start=(root/'frozen_model_hashes.txt').read_text().strip().splitlines()
    checks=[]
    for row in start:
        digest,path=row.split(None,1)
        path=path.strip()
        checks.append({'file':path,'before_sha256':digest,'after_sha256':after[path],'unchanged':after[path]==digest})
    assert all(c['unchanged'] for c in checks),'STOP_FROZEN_INPUT_CHANGED'
    files=[]
    for p in sorted((root/'results').rglob('*')):
        if p.is_file() and 'preview' not in p.parts:
            files.append({'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':sha(p)})
    render=[json.loads(p.read_text()) for p in sorted((root/'results/renders').glob('*_1024_*.json'))]
    physical=[r for r in render if 'material' in r]
    assert len(physical)==20, f'Incomplete physical frames: {len(physical)}'
    assert len(render)==28, f'Incomplete model/physical frames: {len(render)}'
    assert len({r['scene_sha256'] for r in render})==1
    assert len({r['original_table_sha256'] for r in physical})==1
    assert len({r['protocol_sha256'] for r in physical})==1
    assert len({r['script_sha256'] for r in physical})==1
    base=repository/'experiments'
    deps=['ecro86/pt86.py','ecro86/scene86.py','fnto_compiled83/closure83_dr.py',
          'ecro_local93d/addendum_checker/checker93d.py','ecro_local93d/addendum_checker/scene_ck2.json']
    remote_rows=(root/'remote_dependency_hashes.txt').read_text(encoding='utf-8-sig').strip().splitlines()
    remote={r.split(None,1)[1].strip():r.split(None,1)[0] for r in remote_rows}
    assert all(sha(base/f)==remote['experiments/'+f] for f in deps)
    local_start=json.loads((root/'local_start_manifest.json').read_text(encoding='utf-8-sig'))
    preservation=[{'file':f,'before_sha256':v,'after_sha256':sha(repository/f),
                   'unchanged':sha(repository/f)==v} for f,v in local_start['frozen_inputs'].items()]
    assert all(c['unchanged'] for c in preservation),'STOP_LOCAL_PRESERVATION_CHANGED'
    manifest={'exported_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'local_base_commit':'3cf69733d91e0defa4a4648bb64565e45dcd9c47',
              'server_dependency_bundle_commit':'d8d092fd9634921190e7fcc438cb57f2adb79e36',
              'server_root':'LabServer63 /home/nd/rest_nm93e','local_export_root':str(root.resolve()),
              'command':'run93e.sh ROOT full; overlapping independent high-material worker run_high93e.sh ROOT; final viz93e.py/export93e.py LOCAL_ROOT evaluation/93E',
              'environment':{'physical_render_example':physical[0],
                             'gpu':'NVIDIA GeForce RTX 3080 Ti, driver 595.71.05, 12288 MiB',
                             'gpu_start_file':(root/'gpu_start.txt').read_text()},
              'freeze_file':(root/'freeze.txt').read_text(),
              'status':{name:(root/f'{name}.status').read_text() for name in ['preview','full','high']},
              'new_code':[{'file':p.name,'sha256':sha(p)} for p in sorted(code.iterdir()) if p.is_file()],
              'analysis_execution':'Local isolated Windows Python venv; numpy 1.26.4, scipy 1.14.1, matplotlib 3.10.9, Pillow 12.3.0. Additional script upload rejected by automatic approval review; no further code sent. Downloaded existing renders for local analysis. Final visualization labels clarify mirror scope, without changing renderer, raw arrays, protocol conditions or metric formulas.',
              'frozen_dependency_files':[{'path':'experiments/'+f,'sha256':sha(base/f),'server_match':True} for f in deps],
              'local_preservation_checks':preservation,
              'immutable_models_and_tables':checks,
              'actual_geometry_lighting_camera_scene_hash_identical':True,
              'original_state_table_hash_identical_all_physical_arms':True,
              'physical_frame_count':len(physical),'model_frame_count':len(render)-len(physical),
              'raw_results':files,
              'qualification':json.loads((out/'metrics_nearmirror93e.json').read_text())['qualification'],
              'limitations':['finite six-event reference','mirror e2 qualified; full/h exclude light-side delta caustic connector',
                             'frozen ECRO has no new-GGX/mirror contract; S1 checker comparison itself is OOD',
                             'concurrent unrelated GPU work; no uncontended performance or architecture verdict']}
    out.mkdir(parents=True,exist_ok=True)
    (out/'execution_manifest.json').write_text(json.dumps(manifest,indent=2))
    for src,dst in [('preflight.json','preflight.json'),('analytic_material_checks.json','analytic_material_checks.json')]:
        (out/dst).write_bytes((root/'results'/src).read_bytes())
    (out/'protocol.json').write_bytes((code/'protocol.json').read_bytes())
    (out/'scene.json').write_bytes((code/'scene.json').read_bytes())
    # Regenerate after every reviewer artifact exists, including provenance.
    artifacts=[{'file':p.name,'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(out.iterdir()) if p.is_file() and p.name!='manifest_nearmirror93e.json']
    (out/'manifest_nearmirror93e.json').write_text(json.dumps({'artifacts':artifacts,
        'raw_source':'results/ecro_nearmirror93e/results/renders (complete local export) and LabServer63 /home/nd/rest_nm93e/results/renders',
        'provenance':'execution_manifest.json; script/config/input hashes, exact seeds, renderer and environment',
        'architecture_verdict':'NOT APPLICABLE'},indent=2))
    print('EXPORT_VERIFIED',len(artifacts),'artifacts',len(files),'raw files',flush=True)


if __name__=='__main__':main(*sys.argv[1:3])
