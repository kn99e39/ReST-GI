import sys,os
import torch
from common import *
from evaluate import agg,rel

def main():
    records={};refs={};noises={}
    g=loaded_geometry('rest')
    for name in ['rest']+PRIMARY:
        a=torch.load(DATA/'targets'/f'{name}_r0.pt')['test'].cuda();b=torch.load(DATA/'targets'/f'{name}_r1.pt')['test'].cuda()
        refs[name]=agg(.5*(a+b),g);noises[name]=agg((a-b)/2**.5,g)
        ratio=rel(noises[name],refs[name]);records[name]=dict(mean_relative_noise=ratio.mean(0).tolist(),per_field=ratio.cpu().tolist())
    for name in PRIMARY:
        delta=refs[name]-refs['rest'];noise=torch.hypot(noises[name],noises['rest'])
        snr=delta.flatten(1,2).norm(dim=1)/noise.flatten(1,2).norm(dim=1).clamp(min=1e-30)
        records[name]['dynamic_qualified_first16']=(snr[:16]>=3).sum(0).tolist()
    passed=all(all(v<=.02 for v in row['mean_relative_noise']) for row in records.values())
    dump(dict(passed=passed,poses=records,method='independent replica-difference noise; not a claimed confidence interval'),OUT/'reference_check.json')
    assert passed, 'Primary reference precision prevents quality attribution'

if __name__=='__main__':main();sys.stdout.flush();os._exit(0)
