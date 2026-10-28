"""Environment and immutable source manifest; no renderer execution."""
import sys,os,platform,subprocess,json
from pathlib import Path
import torch,numpy,scipy
import r57_geometry as RG
from r57_data import ROOT,OUT,dump,sha
import mitsuba,drjit
mitsuba.set_variant(os.environ.get('P53_MI_VARIANT','cuda_ad_rgb'))

def main():
    files={}
    for d in ('phys_ref53','fnto_c54','fnto_k55','fnto_sel56','fnto_rule57'):
        for p in sorted((RG.HERE.parent/d).rglob('*')):
            if p.is_file() and p.suffix in ('.py','.json','.sh','.md'):files[str(p.relative_to(RG.HERE.parent.parent))]=sha(p)
    d=dict(commit=(ROOT/'repo_commit.txt').read_text().strip(),python=sys.version,platform=platform.platform(),torch=torch.__version__,cuda=torch.version.cuda,
        numpy=numpy.__version__,scipy=scipy.__version__,mitsuba=mitsuba.__version__,drjit=drjit.__version__,GPU=torch.cuda.get_device_name(),
        GPU_status=subprocess.check_output(['nvidia-smi'],text=True),source_files=files,poses=RG.POSES,splits=dict(train=RG.TRAIN,val=RG.VAL,test=RG.TEST),
        renderer='Inherited official Blender3.5.1 e1ccd9d4a1d3 CUDA scope. No new Blender smoke validation.',runtime_oracle_access='Neither model implementation nor geometry preprocessing loads target matrices. Offline and evaluation explicitly separate.')
    dump(d,OUT/'environment.json');print(json.dumps({k:v for k,v in d.items() if k!='source_files'},indent=1))
if __name__=='__main__':main();sys.stdout.flush();os._exit(0)
