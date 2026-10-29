import sys,os,platform,subprocess
import torch,numpy
from common import *

def main():
    files={}
    for folder in ('phys_ref53','fnto_c54','fnto_k55','fnto_sel56','fnto_rule57','fnto_persist59'):
        for p in (HERE.parent/folder).rglob('*.py'):files[str(p.relative_to(HERE.parent.parent))]=sha(p)
    dump(dict(python=platform.python_version(),torch=torch.__version__,numpy=numpy.__version__,cuda=torch.version.cuda,
              gpu=subprocess.check_output(['nvidia-smi'],text=True),cpu=subprocess.check_output(['lscpu'],text=True),
              code_commit=(ROOT/'repo_commit.txt').read_text().strip(),protocol_sha256=sha(HERE/'protocol/persist_v1.json'),
              files=files),OUT/'environment.json')

if __name__=='__main__':main()
