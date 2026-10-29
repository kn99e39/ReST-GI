"""Complete the declared trained-checkpoint rigid control; no new fit/threshold."""
import sys,os
import torch
from common import *
from evaluate import model_of
from runtime import full,selective,apply

def main():
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
    x=torch.load(DATA/'lights.pt')['test'];rows={}
    with torch.no_grad():
        for seed in CFG['training']['seeds']:
            m=model_of('P',seed);rest=geometry('rest');world=rigid_geometry(ballistic=True)
            f=full(m,world);s=full(m,rest);s,audit=selective(m,s,rigid_geometry(ballistic=False));audit.pop('masks')
            errors=[]
            for i in range(0,64,4):
                a=apply(m,f,x[i:i+4].cuda());b=apply(m,s,x[i:i+4].cuda())
                errors.extend(((a-b).abs().sum((1,2))/a.abs().sum((1,2)).clamp(min=1e-30)).cpu().tolist())
            maximum=max(v for r in errors for v in r)
            rows[str(seed)]=dict(max_relative_L1=maximum,per_field_channel=errors,audit=audit,
                                passed=maximum<=CFG['criteria']['rigid_relative_L1'] and audit['raw_retained_fraction']==1.)
            del m,rest,world,f,s;torch.cuda.empty_cache()
    dump(dict(passed=all(v['passed'] for v in rows.values()),seeds=rows,
              status='Declared rigid correctness control completed on final checkpoints; distinct from pre-training control and timing-only workers'),OUT/'trained_rigid.json')
    print(rows.keys(),[v['max_relative_L1'] for v in rows.values()],flush=True)

if __name__=='__main__':main();sys.stdout.flush();os._exit(0)
