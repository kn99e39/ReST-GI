"""WL77A coordinator check (NON-EVIDENCE): error that the q = 1024 single-scatter quadrature noise ALONE would put into an FIA-type prediction
whose learned multiple-scatter part were perfect: prediction = (ref - SS_true) + SS_q1024, SS_true ~ SS_q262144. Reported as relative L2 of
(i) native, fine16, coarse collided response and (ii) the coarse directional contrast H D^-1 Z (eval68 normalisation) against the reference.
Point statistics of the replica mean (not the eval68 jackknife upper bound).   python coord_checks3.py <out.json>"""
import os,sys,json,torch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'fnto_fia77'))
import common77 as C
RG=C.environment();torch.set_grad_enabled(False)
import fia77 as FI,gate77 as GT,family68 as FM,fit71,probes68 as P68
gs=fit71.geometries(RG);g=gs['rest'];G=FI.Geo(g,P68.domain(RG,'rest'));bm=FM.bin_map('cuda');H=FM.Hmat('cuda',torch.float64);out={}
def SS(p,gi,q):
    Q=FI.quad_ports(G,[p],gi,q);r,c,v=Q['SS'];cells=torch.arange(16,device='cuda')+16*p
    return torch.zeros(16,G.P*64,3,device='cuda',dtype=torch.float64).index_put((r-cells[0],c),v[:,None]*G.dom.albedo.double()[None],accumulate=True).view(16,G.P,64,3)
A=lambda y,sp:FM.aggregate(y,sp,g['parent'],bm);rel=lambda a,b:float((a-b).norm()/b.norm())
for s,p in ((0,2085),(50,3439),(10,1699)):
    S=torch.load(GT.bundle(s),map_location='cpu',weights_only=False)['S'].double();ref=S.mean(0).cuda();K=S.shape[0]
    true=sum(SS(p,20+k,65536) for k in range(4))/4;noisy=SS(p,0,1024);pred=ref-true+noisy;d=(1-g['beta'][p]).double()
    con=lambda y:torch.einsum('ab,bqoc->aqoc',H,A(y,'coarse')/d[:,None,None,None])
    half=(S[:K//2].mean(0)-S[K//2:].mean(0)).cuda()/2
    out[f'rest_s{s}']=dict(native=rel(pred,ref),fine16=rel(A(pred,'fine16'),A(ref,'fine16')),coarse=rel(A(pred,'coarse'),A(ref,'coarse')),con_coarse=rel(con(pred),con(ref)),
        ref_noise_con=float(con(half).norm()/con(ref).norm()),ref_noise_native=float(half.norm()/ref.norm()))
    print(s,json.dumps({k:round(v,4) for k,v in out[f'rest_s{s}'].items()}),flush=True)
json.dump(out,open(sys.argv[1],'w'),indent=1)
