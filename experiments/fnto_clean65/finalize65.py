"""One-shot final collection/download; notify the existing thread only on exit."""
import argparse,subprocess,json,time,shutil
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--thread',required=True);ap.add_argument('--log-dir',type=Path,required=True);a=ap.parse_args();a.log_dir.mkdir(parents=True,exist_ok=True);log=a.log_dir/'finalize.log';archive=Path('/private/tmp/wl65-phaseA-bundle.tar.gz')
command='~/NeuralShaderDemo/external/8dna26/.venv/bin/python ~/rest_clean65/collect65.py --root ~/rest_clean65 --original ~/rest_persist_v1 --report-commit c77c03f'
with open(log,'w') as f:
    rc=subprocess.run(['ssh','-o','BatchMode=yes','LabServer69',command],stdout=f,stderr=subprocess.STDOUT).returncode
    if rc==0:rc=subprocess.run(['scp','LabServer69:rest_clean65/phaseA_bundle.tar.gz',str(archive)],stdout=f,stderr=subprocess.STDOUT).returncode
result={'event':'phaseA_provenance_download_completed' if rc==0 else 'phaseA_provenance_failed','returncode':rc,'archive':str(archive),'log':str(log.resolve()),'log_tail':log.read_text()[-2500:],'time':time.time()};(a.log_dir/'result.json').write_text(json.dumps(result,indent=2))
message='[WL65 final collection watcher event] '+json.dumps(result)+'\nHandle this terminal event only. On success validate bundle/manifest, export reviewer artifacts to results/evaluation/65, append final seven-label Phase-A report and commit. Status STOP_INSUFFICIENT_PRECISION; whole batch remains incomplete. No polling, additional photons, model training or Phase B. On failure inspect the concrete collection error once.'
p=subprocess.run([shutil.which('codex'),'queue','--thread',a.thread,'--message',message],capture_output=True,text=True);(a.log_dir/'queue_result.json').write_text(json.dumps({'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr},indent=2))
