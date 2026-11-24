"""WL111: run an UNCHANGED TransGI reproduction script with a fixed argument list (avoids Windows cmd JSON quoting).
python tg_run.py NAME   (NAME in the protocol lr3_detail.transgi_runs)"""
import json
import os
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
L = json.loads((HERE / 'protocol' / 'bench111_lr3.json').read_text(encoding='utf-8'))
run = L['transgi_runs'][sys.argv[1]]
repo = os.environ['TRANSGI_REPO']
os.chdir(repo)
sys.path.insert(0, repo)
ck = str(Path(os.environ['WL111_LR3']) / 'ckpt' / run['ckpt'] / 'checkpoint.pt')
args = [a.replace('{CKPT}', ck).replace('{PO}', json.dumps(L['transgi_probe_override'])) for a in run['args']]
sys.argv = [run['script']] + args
print('TG_RUN', sys.argv, flush=True)
runpy.run_path(run['script'], run_name='__main__')
