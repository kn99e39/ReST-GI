"""Stop the verified public evaluation queue at the user's request."""
import json
import os
import signal
import time
from pathlib import Path

base = Path.home() / 'rest_8dna_public21a'
queue_pid = 3579312
render_pid = 3958588
queue_argv = ['bash', str(base / 'scripts/launch.sh')]
render_argv = [
    'python', '-u', 'scripts/render.py', '--upstream',
    '/home/nd/NeuralShaderDemo/external/8dna26', '--out', str(base / 'results'),
    '--asset', 'fabric', '--method', 'neural', '--rep', '0', '--spp', '128',
]

def inspect(pid):
    proc = Path('/proc') / str(pid)
    argv = proc.joinpath('cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
    stat = proc.joinpath('stat').read_text().rsplit(')', 1)[1].split()
    return {'pid': pid, 'ppid': int(stat[1]), 'state': stat[0],
            'starttime': stat[19], 'argv': argv}

queue = inspect(queue_pid)
render = inspect(render_pid)
if queue['argv'] != queue_argv or render['argv'] != render_argv:
    raise RuntimeError('Process identity changed; inspect before stopping')
os.kill(queue_pid, signal.SIGSTOP)
os.kill(render_pid, signal.SIGSTOP)
processes = {}
for proc in Path('/proc').iterdir():
    if proc.name.isdigit():
        try:
            processes[int(proc.name)] = inspect(int(proc.name))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
selected = {queue_pid}
while True:
    expanded = selected | {pid for pid, data in processes.items() if data['ppid'] in selected}
    if expanded == selected:
        break
    selected = expanded
if render_pid not in selected:
    os.kill(render_pid, signal.SIGCONT)
    os.kill(queue_pid, signal.SIGCONT)
    raise RuntimeError('Renderer is outside verified queue process tree')
utc = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
record = {'utc': utc, 'reason': 'User requested stopping the Fabric render.',
          'queue_status': 'stopped_by_user',
          'processes': [processes[pid] for pid in sorted(selected)]}
record_path = base / 'user_stop_fabric_20261117.json'
record_path.write_text(json.dumps(record, indent=2) + '\n')
# The dispatcher is frozen, so no subsequent evaluation can start.
# End stopped roots immediately, and allow active workers a brief clean exit.
for pid in selected:
    try:
        current = inspect(pid)
        if current['starttime'] == processes[pid]['starttime']:
            sig = signal.SIGKILL if pid in (queue_pid, render_pid) else signal.SIGTERM
            os.kill(pid, sig)
    except (FileNotFoundError, ProcessLookupError):
        pass
time.sleep(2)
for pid in selected:
    try:
        current = inspect(pid)
        if current['starttime'] == processes[pid]['starttime'] and current['state'] != 'Z':
            os.kill(pid, signal.SIGKILL)
    except (FileNotFoundError, ProcessLookupError):
        pass
(base / 'queue.status').write_text(f'stopped_by_user {utc}\n')
(base / 'current.status').write_text(f'stopped_by_user fabric_neural_r0_s128 {utc}\n')
print(json.dumps({'status': 'stopped_by_user', 'process_count': len(selected),
                  'record': str(record_path)}), flush=True)
