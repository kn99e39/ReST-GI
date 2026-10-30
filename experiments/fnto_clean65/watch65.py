"""Local event bridge: persistent SSH -> existing Codex thread queue, no model polling."""
import argparse,json,subprocess,time,os
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--thread',required=True);ap.add_argument('--codex',required=True);ap.add_argument('--log-dir',type=Path,required=True);a=ap.parse_args();a.log_dir.mkdir(parents=True,exist_ok=True)
statefile=a.log_dir/'delivered.json';delivered=json.loads(statefile.read_text()) if statefile.exists() else [];finished=False
while not finished:
    p=subprocess.Popen(['ssh','-o','BatchMode=yes','-o','ServerAliveInterval=30','-o','ServerAliveCountMax=3','LabServer69','python -u ~/rest_clean65/watch_server65.py ~/rest_clean65'],stdout=subprocess.PIPE,stderr=open(a.log_dir/'ssh.log','a'),text=True)
    for line in p.stdout:
        event=json.loads(line);eid=event['id']
        if eid in delivered:continue
        msg='[WL65 stage watcher event] '+json.dumps(event)+'\nReact only to this completed/failed stage. Do not poll or wait for progress. Server chain remains authoritative. No model training/Phase B. If report completed, finish authorized Phase-A provenance/export/report; if failed, inspect once and report/repair only a concrete in-scope defect. Otherwise acknowledge briefly and return idle.'
        result=subprocess.run([a.codex,'queue','--thread',a.thread,'--message',msg],capture_output=True,text=True)
        with open(a.log_dir/'events.jsonl','a') as f:f.write(json.dumps(dict(event=event,queue_returncode=result.returncode,queue_stdout=result.stdout,queue_stderr=result.stderr))+'\n')
        if result.returncode:
            p.terminate();break
        delivered.append(eid);statefile.write_text(json.dumps(delivered));finished=eid in ('report','failure')
    p.wait()
    if not finished:time.sleep(20)
(a.log_dir/'complete.json').write_text(json.dumps({'delivered':delivered,'thread':a.thread,'time':time.time()}))
