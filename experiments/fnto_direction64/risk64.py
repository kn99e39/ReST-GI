"""Final train/validation pair-risk measurement only; no updates or selection."""
import os,sys,json,time
from pathlib import Path
ROOT=Path.home()/'rest_direction64';ORIG=Path.home()/'rest_persist_v1'
os.environ['P59']=str(ORIG);os.environ['WL57']=str(ORIG)
sys.path.insert(0,str(ORIG/'repo/experiments/fnto_persist59'));sys.path.insert(0,str(ROOT/'source/experiments/fnto_direction64'))
import torch,numpy as np
from common import RG
from runtime import Transport
from train import loss_metric
from objective64 import contrast
from r57_data import dump,aggregate

def main():
    assert not (ROOT/'final_pair_risk.json').exists();torch.set_num_threads(8);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
    fixed=json.loads((ROOT/'fixed_objective.json').read_text());lights=torch.load(ORIG/'data/lights.pt');results={};t0=time.perf_counter()
    for arm in ('A','B'):
        for seed in (5901,5902):
            m=Transport().cuda().eval();m.load_state_dict(torch.load(ROOT/'arms'/arm/'checkpoints'/f'P_{seed}.pt')['model']);rows={}
            for split,names,start in (('train',RG.TRAIN,80),('validation',['rest']+RG.VAL,16)):
                for name in names:
                    g=RG.device(torch.load(ORIG/'data/geometry'/f'{name}.pt'));tt=torch.load(ORIG/'data/training_targets'/f'{name}_r0.pt')[split][start:].cuda();xx=lights[split][start:].cuda();co=m.coefficients(g)
                    yy=torch.cat([m.apply(g,xx[i:i+2],co) for i in range(0,16,2)]);dirloss=[]
                    for p in range(8):dirloss.append(float(contrast(yy[2*p:2*p+2],tt[2*p:2*p+2],g['parent'],fixed['eta'])))
                    pairerr={}
                    for label,coarse in (('coarse',True),('fine16',False)):
                        a,b=aggregate(yy,g['parent'],coarse),aggregate(tt,g['parent'],coarse);d=a[::2]-a[1::2];q=b[::2]-b[1::2]
                        pairerr[label]=((d-q).flatten(1,2).norm(dim=1)/q.flatten(1,2).norm(dim=1).clamp(min=1e-30)).tolist()
                    rows[f'{split}/{name}']=dict(pair_objective=dirloss,mean_objective=float(np.mean(dirloss)),absolute_pair_action_loss=float(loss_metric(yy,tt,g['parent'])),pair_error=pairerr,label='original r0; unqualified noisy supervision; not physical test')
                    del g,tt,xx,co,yy;torch.cuda.empty_cache()
            results[f'{arm}_{seed}']=rows;del m;torch.cuda.empty_cache()
    dump(dict(models=results,seconds=time.perf_counter()-t0,selection=False),ROOT/'final_pair_risk.json');print('FINAL_PAIR_RISK_COMPLETE',flush=True)
if __name__=='__main__':main();sys.stdout.flush();os._exit(0)
