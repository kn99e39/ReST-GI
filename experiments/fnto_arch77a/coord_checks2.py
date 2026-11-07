"""WL77A coordinator check (NON-EVIDENCE): projection floors of M = ref - SS_q at q = 4096, 16384, 65536 for two training sources, and a
1/q extrapolation of the squared residual (residual^2 = true^2 + c/q) to the noise-free floors.   python coord_checks2.py <out.json>"""
import os,sys,json,torch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'fnto_fia77'))
import common77 as C
RG=C.environment();torch.set_grad_enabled(False)
import fia77 as FI,gate77 as GT,fit71,probes68 as P68
gs=fit71.geometries(RG);g=gs['rest'];G=FI.attach_frames(FI.Geo(g,P68.domain(RG,'rest')));phi_o,_=FI.bin_azimuth();phi_o=phi_o.double().cuda();t,b=G.frames;out={}
def SS(p,gi,q):
    Q=FI.quad_ports(G,[p],gi,q);r,c,v=Q['SS'];cells=torch.arange(16,device='cuda')+16*p
    return torch.zeros(16,G.P*64,3,device='cuda',dtype=torch.float64).index_put((r-cells[0],c),v[:,None]*G.dom.albedo.double()[None],accumulate=True).view(16,G.P,64,3)
for s,p in ((0,2085),(50,3439)):
    S=torch.load(GT.bundle(s),map_location='cpu',weights_only=False)['S'].double();ref=S.mean(0).cuda()
    rr=G.x.double()-G.x[p].double();rn=(rr*G.n.double()).sum(1,keepdim=True);rt=rr-rn*G.n.double()
    Bs=GT.basis_sets(phi_o,torch.atan2((rt*b.double()).sum(1),(rt*t.double()).sum(1)));row={}
    for q in (4096,16384,65536):
        M=ref-SS(p,11,q);row[q]={bn:float(GT.residual(M,B).norm()/M.norm()) for bn,B in Bs.items()};print(s,q,row[q],flush=True)
    row['extrapolated_noise_free']={bn:float(max(0.,(row[65536][bn]**2*65536-row[16384][bn]**2*16384)/(65536-16384)))**.5 for bn in Bs}
    out[f'rest_s{s}']={str(k):v for k,v in row.items()};print(s,'extrap',row['extrapolated_noise_free'],flush=True)
json.dump(out,open(sys.argv[1],'w'),indent=1)
