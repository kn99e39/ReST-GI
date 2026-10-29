"""POST HOC stored-output seed sensitivity; no fit, inference or gate changes."""
import sys,os,inspect
import torch
from common import *
from evaluate import agg

def main():
    torch.set_num_threads(8)
    g=dict(parent=torch.load(DATA/'geometry/rest.pt')['parent'].cuda())
    rs={}
    for name in ('rest',PRIMARY[0]):
        rs[name]=[agg(torch.load(DATA/'targets'/f'{name}_r{r}.pt')['test'].cuda(),g).cpu().double() for r in (0,1)]
    d=[rs[PRIMARY[0]][r]-rs['rest'][r] for r in (0,1)]
    mean=.5*(d[0]+d[1])
    # Reproduce the SAME conservative primary qualification mask from independent per-pose replica differences.
    noise=torch.hypot((rs[PRIMARY[0]][0]-rs[PRIMARY[0]][1])/(2**.5),(rs['rest'][0]-rs['rest'][1])/(2**.5))
    snr=mean.flatten(1,2).norm(dim=1)/noise.flatten(1,2).norm(dim=1).clamp(min=1e-30)
    result={}
    for seed in CFG['training']['seeds']:
        kw=dict(map_location='cpu')
        if 'mmap' in inspect.signature(torch.load).parameters:kw['mmap']=True
        obj=torch.load(OUT/f'actions_{seed}.pt',**kw)
        pred={name:agg(obj['predictions'][name]['F'].cuda(),g).cpu().double() for name in ('rest',PRIMARY[0])}
        yd=pred[PRIMARY[0]]-pred['rest'];rows=[]
        for c in range(3):
            ids=(snr[:16,c]>=3).nonzero().flatten()
            yy=yd[:16,:,:,c][ids].flatten()
            targets=[v[:16,:,:,c][ids].flatten() for v in (mean,d[0],d[1])]
            errors=[float((yy-v).norm()/v.norm()) for v in targets]
            se=(targets[1]-targets[2])/2
            rows.append(dict(channel=c,n_fields=len(ids),relative_delta_errors_mean_rep0_rep1=errors,
                             reference_mean_delta_noise_norm_ratio=float(se.norm()/targets[0].norm())))
        pp=pred[PRIMARY[0]][48::2]-pred[PRIMARY[0]][49::2]
        tt=[v[48::2]-v[49::2] for v in rs[PRIMARY[0]]];tm=.5*(tt[0]+tt[1])
        er=lambda a,b:((a-b).flatten(1,2).norm(dim=1)/b.flatten(1,2).norm(dim=1)).tolist()
        result[str(seed)]=dict(dynamic=rows,directional_errors_mean_rep0_rep1=[er(pp,t) for t in (tm,tt[0],tt[1])],
            directional_reference_mean_noise_ratio=((tt[0]-tt[1]).flatten(1,2).norm(dim=1)/2/tm.flatten(1,2).norm(dim=1)).tolist())
        del obj,pred
    dump(dict(post_hoc=True,changes_primary_verdict=False,method='Stored outputs vs each independent reference replica with fixed original masks; descriptive sensitivity, not a confidence interval',seeds=result),OUT/'review/noise_sensitivity.json')

if __name__=='__main__':main();sys.stdout.flush();os._exit(0)
