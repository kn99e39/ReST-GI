"""WL78 Stage 0 geometry check (no reference data): port bin frames per face at rest and pt0; boundary probes."""
import json,sys
import numpy as np,torch
from common78 import environment
RG=environment();torch.set_grad_enabled(False)
import fit71,probes68 as P68
out={}
for name in ('rest','pt0'):
    g=RG.device(fit71.guarded(f'geometry_{name}'))
    if name not in RG.POSES:RG.POSES[name]=tuple(__import__('fit72').PT0)
    dom=P68.domain(RG,name);tri=dom.port_tris[:,0];t=dom.t[tri].double();b=dom.b[tri].double();n=dom.n[tri].double()
    face=np.rint(g['node'][:,15].cpu().numpy()*5).astype(int);row={}
    for f in range(6):
        m=torch.as_tensor(face==f,device='cuda')
        row[f]=dict(n=int(m.sum()),t=t[m].mean(0).tolist(),t_std=float(t[m].std(0).max()),b=b[m].mean(0).tolist(),nrm=n[m].mean(0).tolist(),
            node_normal_vs_dom=float((n[m]-g['node'][m,9:12].double()).norm(dim=1).max()))
    row['handed_min']=float((torch.cross(t,b,dim=1)*n).sum(1).min());row['P_dom']=int(dom.P);row['P_g']=int(g['P']);out[name]=row
json.dump(out,open(sys.argv[1],'w'),indent=1);print(json.dumps(out)[:3000])
