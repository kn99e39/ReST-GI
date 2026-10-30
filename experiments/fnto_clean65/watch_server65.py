"""Token-free server observer: emits stage completion/failure only; never mutates science."""
import json,time,os,sys
from pathlib import Path
root=Path(sys.argv[1]);seen=set()
def read(p):
    try:return json.loads(p.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return {}
while True:
    for stage in ('generate','analyze','report'):
        p=root/f'{stage}_stage.json'
        if p.exists() and stage not in seen:
            seen.add(stage);event=dict(id=stage,stage=stage,event='completed',receipt=read(p),root=str(root))
            if stage=='generate':event['reference']={k:v for k,v in read(root/'reference_complete.json').items() if k!='records'}
            if stage in ('analyze','report'):event['phaseA']=read(root/'phaseA_status.json')
            print(json.dumps(event),flush=True)
    state=read(root/'status.json');pid=read(root/'launch.json').get('pid');alive=True
    if pid:
        try:os.kill(pid,0)
        except ProcessLookupError:alive=False
    if (root/'failure.txt').exists() or (state.get('state')=='RUNNING' and not alive):
        print(json.dumps(dict(id='failure',event='failed',state=state,error=(root/'failure.txt').read_text()[-5000:] if (root/'failure.txt').exists() else 'Chain exited without terminal marker',root=str(root))),flush=True);break
    if 'report' in seen:break
    time.sleep(5)
