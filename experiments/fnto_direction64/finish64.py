"""Wait for the fixed chain, then run only report/invariant measurements."""
import json,sys,time,os,subprocess
from pathlib import Path
r=Path.home()/'rest_direction64'
while True:
    s=json.loads((r/'status.json').read_text()) if (r/'status.json').exists() else {}
    if s.get('state')=='FAILED':raise RuntimeError('Scientific chain failed; do not auto continue')
    if s.get('state')=='COMPLETE':break
    time.sleep(5)
for name in ('invariants64','risk64'):
    with open(r/f'{name}.log','w') as f:subprocess.run([sys.executable,str(r/f'{name}.py')],stdout=f,stderr=subprocess.STDOUT,check=True)
with open(r/'export64.log','w') as f:subprocess.run([sys.executable,str(r/'export64.py'),str(r),str(r/'review_v1')],stdout=f,stderr=subprocess.STDOUT,check=True)
(r/'review_complete.json').write_text(json.dumps({'state':'COMPLETE','ended':time.time(),'report_source_commit':(r/'report_commit.txt').read_text().strip()}))
