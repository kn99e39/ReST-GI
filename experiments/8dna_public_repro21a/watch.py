"""Keep review artifacts and machine-readable status current beside the GPU queue."""
import json,subprocess,sys,time
from pathlib import Path
base=Path(sys.argv[1]).resolve(); last=None
while True:
    paths=sorted((base/'results').glob('*.json'))
    stamp=[(p.name,p.stat().st_mtime_ns) for p in paths]
    queue=(base/'queue.status').read_text().strip() if (base/'queue.status').exists() else 'starting'
    if stamp!=last or queue.startswith('complete'):
        with open(base/'analysis.log','w') as log:
            done=subprocess.run([sys.executable,str(base/'scripts'/'analyze.py'),'--base',str(base),'--export'],stdout=log,stderr=subprocess.STDOUT)
        (base/'analysis.status').write_text(json.dumps({'updated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'exit':done.returncode,'queue':queue})+'\n')
        if done.returncode==0: last=stamp
    if queue.startswith('complete'): break
    time.sleep(20)
