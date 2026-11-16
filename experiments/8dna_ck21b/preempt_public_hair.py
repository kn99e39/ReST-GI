"""User-prioritized training: preempt only the verified public Hair renderer tree."""
import json
import os
import signal
import time
from pathlib import Path
BASE=Path.home()/'rest_8dna_ck21b'
PID=3857935
expected=['python','-u','scripts/render.py','--upstream','/home/nd/NeuralShaderDemo/external/8dna26',
    '--out','/home/nd/rest_8dna_public21a/results','--asset','hair','--method','neural','--rep','0','--spp','128']
def inspect(pid):
    root=Path('/proc')/str(pid)
    cmd=root.joinpath('cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
    stat=root.joinpath('stat').read_text().rsplit(')',1)[1].split()
    return {'pid':pid,'ppid':int(stat[1]),'state':stat[0],'starttime':stat[19],'argv':cmd}
root=inspect(PID)
if root['argv']!=expected: raise RuntimeError('Public renderer identity mismatch')
nodes={}
for path in Path('/proc').iterdir():
    if path.name.isdigit():
        try: nodes[int(path.name)]=inspect(int(path.name))
        except (FileNotFoundError,ProcessLookupError,PermissionError): pass
selected={PID}
while True:
    expanded=selected|{pid for pid,record in nodes.items() if record['ppid'] in selected}
    if expanded==selected: break
    selected=expanded
for pid in selected:
    if nodes[pid]['argv']!=expected: raise RuntimeError(f'Unrecognized child; leave tree untouched: {nodes[pid]}')
record={'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
    'reason':'User explicitly requested resuming CK2 training; public Hair render is deferred to free GPU memory.',
    'processes':[nodes[pid] for pid in sorted(selected)],
    'retry':'The existing public dispatcher later revisits this same asset/method/rep at target4096, regenerating the 128spp checkpoint before continuing.'}
(BASE/'public_hair_preemption.json').write_text(json.dumps(record,indent=2)+'\n')
for pid in sorted(selected,reverse=True):
    try: os.kill(pid,signal.SIGTERM)
    except ProcessLookupError: pass
time.sleep(2)
for pid in selected:
    try:
        current=inspect(pid)
        if current['starttime']==nodes[pid]['starttime'] and current['argv']==expected and current['state']!='Z':
            os.kill(pid,signal.SIGKILL)
    except (FileNotFoundError,ProcessLookupError): pass
print('PREEMPTED_VERIFIED_PUBLIC_HAIR_TREE',len(selected),flush=True)
