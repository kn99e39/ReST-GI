import os, sys, json, time, importlib, argparse
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--upstream', required=True)
p.add_argument('--asset',default='seal')
p.add_argument('--scene',default='scene1')
p.add_argument('--res',type=int,default=64)
p.add_argument('--spp',type=int,default=1)
p.add_argument('--vcall',type=int,default=0)
p.add_argument('--method',default='neural')
p.add_argument('--out',required=True)
a=p.parse_args()
root=Path(a.upstream).resolve(); os.chdir(root/'demo'); sys.path.insert(0,str(root))
import torch
import mitsuba as mi
mi.set_variant('cuda_ad_rgb')
import drjit as dr
import models.integrator
dr.set_flag(dr.JitFlag.LoopRecord,False)
dr.set_flag(dr.JitFlag.VCallRecord,bool(a.vcall))
print(json.dumps({'torch':torch.__version__,'mitsuba':mi.__version__,'drjit':dr.__version__,'gpu':torch.cuda.get_device_name(),'args':vars(a)}),flush=True)
s=mi.load_dict(getattr(importlib.import_module('scenes.'+a.asset),a.scene)(a.res))
i=s.integrator() if a.method=='pt' else mi.load_dict({'type':'neuralpath','max_depth':10,'rr_depth':5,'device':0,'model_type':'8dna','asset_instance0':str(root/'checkpoints'/f'{a.asset}.ckpt')})
print('constructed',flush=True)
t=time.perf_counter(); img=mi.render(s,integrator=i,spp=a.spp,seed=0); dr.eval(img); torch.cuda.synchronize()
import numpy as np
x=np.array(img)
Path(a.out).parent.mkdir(parents=True,exist_ok=True)
mi.Bitmap(x).write(a.out)
print(json.dumps({'render_s':time.perf_counter()-t,'finite':bool(np.isfinite(x).all()),'mean':float(x.mean()),'max':float(x.max()),'peak_torch_bytes':torch.cuda.max_memory_allocated()}),flush=True)
