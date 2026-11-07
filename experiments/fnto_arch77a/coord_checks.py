"""WL77A coordinator checks (NON-EVIDENCE, labelled; read-only on every historical file; no Gate 4 reference is read).
(1) SS-quadrature noise in the WL77 Stage 1 floors (objection A6): recompute the gate77 projection floors for two training sources with the
    single-scatter term SS at q = 1024 (the WL76/77 seeds), at q = 1024 with independent seeds, and at q = 16384; also the floor of the
    pure SS-noise difference.
(2) Translate identity (objection A1 / B1): WL74 Stage 4 far-boundary source s101 (port 224) vs its training counterpart s50 (port 3439):
    per-incoming-bin collided energy (sorted) and tangential radial energy profile, against the replica noise.
    python coord_checks.py <out.json>"""
import os,sys,json,torch
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'fnto_fia77'))
import common77 as C
RG=C.environment();torch.set_grad_enabled(False)
import fia77 as FI,gate77 as GT,family68 as FM,fit71,probes68 as P68
gs=fit71.geometries(RG);g=gs['rest'];G=FI.attach_frames(FI.Geo(g,P68.domain(RG,'rest')));phi_o,_=FI.bin_azimuth();phi_o=phi_o.double().cuda();t,b=G.frames;out={}
def SS(p,gi,q):
    Q=FI.quad_ports(G,[p],gi,q);r,c,v=Q['SS'];cells=torch.arange(16,device='cuda')+16*p
    return torch.zeros(16,G.P*64,3,device='cuda',dtype=torch.float64).index_put((r-cells[0],c),v[:,None]*G.dom.albedo.double()[None],accumulate=True).view(16,G.P,64,3)
for s,p in ((0,2085),(50,3439)):
    bb=torch.load(GT.bundle(s),map_location='cpu',weights_only=False);S=bb['S'].double();ref=S.mean(0).cuda()
    rr=G.x.double()-G.x[p].double();rn=(rr*G.n.double()).sum(1,keepdim=True);rt=rr-rn*G.n.double()
    Bs=GT.basis_sets(phi_o,torch.atan2((rt*b.double()).sum(1),(rt*t.double()).sum(1)))
    variants={'q1024_wl77':SS(p,0,1024),'q1024_indep':SS(p,7,1024),'q16384':SS(p,8,16384)}
    row={}
    for k,ss in variants.items():
        M=ref-ss;row[k]={bn:float(GT.residual(M,B).norm()/M.norm()) for bn,B in Bs.items()}
    d=(variants['q1024_wl77']-variants['q1024_indep'])/2**.5;M=ref-variants['q16384']
    row['ss_q1024_noise_floor']={bn:float(GT.residual(d,B).norm()/M.norm()) for bn,B in Bs.items()}
    row['ss_q1024_noise_rel_to_M']=float(d.norm()/M.norm());out[f'rest_s{s}']=row;print(s,json.dumps(row),flush=True)
# (2) translate check
def prof(S,p):
    x=G.x.double();rr=x-x[p];rn=(rr*G.n[p].double()).sum(1,keepdim=True);rt=(rr-rn*G.n[p].double()).norm(dim=1)*1e3   # mm, tangential
    e=S.sum((3,4))                                                                                                     # (K,16,P)
    edges=torch.tensor([0,1.5,3,5,8,12,20,30,60,200],dtype=torch.float64)
    idx=torch.bucketize(rt.cpu(),edges)-1
    return torch.stack([e[...,idx==k].sum(-1) for k in range(len(edges)-1)],-1)                                       # (K,16,nbins)
a=torch.load(GT.bundle(50),map_location='cpu',weights_only=False)['S'].double();bpath=Path.home()/'rest_rank75/refs/bundles/rest_s101.pt'
if bpath.exists():
    bt=torch.load(bpath,map_location='cpu',weights_only=False);assert bt['port']==224;B_=bt['S'].double()
    pa=prof(a,3439);pb=prof(B_,224);ea=pa.sum(-1).mean(0).sort().values;eb=pb.sum(-1).mean(0).sort().values
    sea=pa.sum(-1).std(0)/a.shape[0]**.5;seb=pb.sum(-1).std(0)/B_.shape[0]**.5
    ra=pa.mean(0).sum(0);rb=pb.mean(0).sum(0)
    out['translate_s101_vs_s50']=dict(sorted_bin_energy_rel_diff=float((ea-eb).norm()/ea.norm()),bin_energy_SE_rel=float(torch.hypot(sea.norm(),seb.norm())/ea.norm()),
        radial_profile_rel_diff=float((ra-rb).norm()/ra.norm()),radial_profile_s50=ra.tolist(),radial_profile_s101=rb.tolist(),
        note='energy invariants only (bin-permutation free); s101 bundle from the local WL75 root (K=4), s50 from the WL73 training bundle')
    print('TRANSLATE',json.dumps({k:v for k,v in out['translate_s101_vs_s50'].items() if 'profile_s' not in k}),flush=True)
json.dump(out,open(sys.argv[1],'w'),indent=1)
