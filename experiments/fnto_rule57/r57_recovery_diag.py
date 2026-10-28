"""Stage A: immutable original WL57 evidence only. No new rays/targets/fitting."""
import os,sys,json,time
from pathlib import Path
import numpy as np
import torch
import r57_geometry as RG
from r57_data import aggregate,dump,sha

OLD=Path.home()/'rest_wl57';DEST=Path(os.environ.get('WL57_DIAG',Path.home()/'rest_wl57_recovery/diagnostic'))
def norm(x):return x.flatten(1,2).norm(dim=1)
def coarse_operator(name,rep):
    r=torch.load(OLD/'ops'/f'{name}_r{rep}.pt',map_location='cpu');out=torch.zeros(3,240*240,device='cuda',dtype=torch.float64)
    for i in range(0,len(r['idx']),2_000_000):
        ix=r['idx'][i:i+2_000_000].cuda();v=r['val'][:,i:i+2_000_000].cuda().double();row=ix//r['shape'][1];col=ix%r['shape'][1]
        out.index_add_(1,(row//64//16)*240+(col//16//16),v/256)
    return out.view(3,240,240)
def main():
    DEST.mkdir(parents=True,exist_ok=True);torch.set_num_threads(8)
    g=RG.device(torch.load(OLD/'data/geometry/rest.pt'));parent=g['parent'];x=torch.load(OLD/'data/lights.pt')['test'].cuda();xyz=g['rest_xyz'];meta=json.load(open(OLD/'out/light_manifest.json'))['test']
    def target(name,r):return torch.load(OLD/'data/targets'/f'{name}_r{r}.pt')['test'].cuda()
    rest=[target('rest',r) for r in (0,1)];rr=[aggregate(t,parent) for t in rest];rm=.5*(rr[0]+rr[1]);rn=(rr[0]-rr[1])/2**.5
    pc=torch.zeros(240,3,device='cuda').index_add_(0,parent,xyz)/16;face=torch.tensor(RG.mesh('rest')[0]['port_face'],device='cuda')
    res={};a,b=coarse_operator('rest',0),coarse_operator('rest',1)
    for name in RG.TEST:
        cur=[target(name,r) for r in (0,1)];cc=[aggregate(t,parent) for t in cur];cm=.5*(cc[0]+cc[1]);cn=(cc[0]-cc[1])/2**.5;delta=cm-rm
        dep,rad,cx,cy=RG.POSES[name];loc=(torch.hypot(pc[:,0]-cx,pc[:,1]-cy)<=rad+.0025);lf=loc[parent]
        noise=norm(cn).square()+norm(rn).square();sn=norm(delta)/noise.sqrt();debiased=(norm(delta).square()-.5*noise).clamp(min=0)
        field=dict(SNR=sn.tolist(),relative_change=(norm(delta)/norm(rm)).tolist(),noise_relative=(noise.sqrt()/norm(rm)).tolist(),
            debiased_change_relative=(debiased.sqrt()/norm(rm)).tolist(),incident_local_mass=x[:,lf].sum((1,2)).tolist(),
            effective_stratified_photons=(2048/x.square().sum((1,2))).tolist(),families=meta)
        regions={}
        for label,mask in [('local',loc),('complement',~loc)]+[(f'face_{f}',face==f) for f in range(6)]:
            dd=delta[:,mask];nn=norm(cn[:,mask]).square()+norm(rn[:,mask]).square();ee=(norm(dd).square()-.5*nn)
            regions[label]=dict(debiased_signal_energy=ee.tolist(),SNR=(norm(dd)/nn.sqrt().clamp(min=1e-30)).tolist())
        # Equal projected-angle rings, fully reported; no favorable subset promoted to primary.
        rings=[]
        for k in range(2):
            ids=torch.tensor([j for j in range(16) if int(max(abs(j%4-1.5),abs(j//4-1.5)))==k],device='cuda')
            rings.append(dict(ring=k,SNR=(norm(delta[:,:,ids])/torch.hypot(norm(cn[:,:,ids]),norm(rn[:,:,ids])).clamp(min=1e-30)).tolist()))
        c,d=coarse_operator(name,0),coarse_operator(name,1);dm=.5*(c+d-a-b);on=torch.hypot((c-d).flatten(1).norm(dim=1),(a-b).flatten(1).norm(dim=1))/2**.5
        op=dict(spatial_pair_SNR=(dm.flatten(1).norm(dim=1)/on).tolist(),relative_change=(dm.flatten(1).norm(dim=1)/(.5*(a+b)).flatten(1).norm(dim=1)).tolist(),
            comparison_WL55='1cm spatial-only pairs; WL55 dent6.7/6.8/3.6 used single-seed change vs rest-seed difference; this uses two-seed mean and pooled noise, so definitions differ.')
        # Null controls from independent seeds, same geometry, same spatial/directional estimand.
        null=dict(rest_noise_relative=(norm(rn)/norm(rm)).tolist(),current_noise_relative=(norm(cn)/norm(cm)).tolist(),
            null_expected_mean_delta_to_single_seed_noise_norm=2**-.5,noise_corrected_signal_energy=debiased.tolist())
        res[name]=dict(field=field,regions=regions,angular=rings,operator=op,noise_control=null)
        dump(dict(kind='POST_HOC_DIAGNOSTIC_ONLY',source_commit='dc6a5326b3abf0d962c18f3536e7b1cf376fe624',poses=res),DEST/'diagnosis.json')
        print(name,op,'focused SNR fourth',torch.sort(sn[:16],dim=0,descending=True).values[3].tolist(),flush=True)
        del cur,cc,cm,cn,c,d;torch.cuda.empty_cache()
    print('DIAGNOSTIC COMPLETE; no replacement targets generated',flush=True)
if __name__=='__main__':main();sys.stdout.flush();os._exit(0)
