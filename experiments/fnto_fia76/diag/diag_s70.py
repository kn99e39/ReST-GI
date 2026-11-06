"""NON-EVIDENCE diagnostic after the Stage 1 failure: rest s70 (SR, port 519) bin 0, single-scatter quadrature vs MC single tally with fresh seeds."""
import os,sys,json,torch
sys.path.insert(0,os.path.expanduser('~/rest_fia76/repo/experiments/fnto_fia76'))
import phys76 as P
RG=P.environment();torch.set_grad_enabled(False)
import probes68 as P68
dom=P68.domain(RG,'rest');parent=torch.as_tensor(RG.KO.refined(*RG.mesh('rest'),4)[2]);bm=torch.as_tensor(RG.G.bin_parent(8,4));N=32_000_000;C=lambda y:P.coarse(y,parent,bm);out=[]
for rep in (11,12,13):
    r=P.decomposed(dom,519,0,N,769000000+rep);sm,sv=P.mv(r['single'],N);q=P.single_quadrature(dom,519,0,N,769100000+rep);qm,qv=P.mv(q[:2],N)
    c=P.compare(C(qm),C(qv),C(sm),C(sv),C(qm),N,30);out.append(dict(rep=rep,z2=c['coarse_mean_z2'],cells=c['cells'],energy_z=c['energy_z']));print(out[-1],flush=True)
open(os.path.expanduser('~/rest_fia76/diag_s70_nonevidence.json'),'w').write(json.dumps(out,indent=1))
