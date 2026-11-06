"""WL76 POST HOC diagnostic (labelled; does not change the Stage 3 verdict): where does the FIA training-source error come from?
For the S16-R rest sources (replica means of the archived bundles):
  ss_share       exact single-scatter share of the reference collided energy (deployable SS term)
  err_total      FIA error (coarse / fine16 / native, relative L2)
  err_ms         error of the learned multiple-scatter part vs the reference minus the exact SS
  ring_floor     native residual of the reference multiple-scatter part after projecting each receiver's 64 bins onto ring-uniform
                 distributions (what ring lobes can never represent), with the replica-noise floor of the same statistic
  inc_profile    relative L2 of the per-incoming-bin collided energy (FIA vs reference) - the incident-angle budget
    python posthoc76.py <run_id>"""
import os,sys,json,torch
sys.path.insert(0,os.path.expanduser('~/rest_fia76/repo/experiments/fnto_fia76'))
import common76 as C
RG=C.environment();torch.set_grad_enabled(False)
import family68 as FM,fia76 as FI,fit71,fit73,probes68 as P68
run=sys.argv[1];gs=fit71.geometries(RG);g=gs['rest'];G=FI.Geo(g,P68.domain(RG,'rest'));bm=FM.bin_map('cuda')
x=torch.load(C.ROOT/'runs'/f'{run}.pt',map_location='cpu',weights_only=False);m=FI.FIA(C.CFG['candidate']['hidden']).cuda();m.load_state_dict(x['model']);m.eval()
pops=fit73.populations()['S16R'];P71=C.R71;out={}
def bundle(s):
    if s<4:return C.INP68/'fit'/f'rest_s{s}.pt'
    if s<8:return C.R68/'heldout/bundles'/f'rest_s{s}.pt'
    if s<16:return P71/'refs/bundles'/f'rest_s{s}.pt'
    return C.R73/'refs/bundles'/f'rest_s{s}.pt'
RING=FI.RING.cuda();rel=lambda a,b:float((a-b).norm()/b.norm().clamp(min=1e-30))
def ringproj(y):   # y (..., 64, C) -> ring-uniform projection
    s=torch.zeros(*y.shape[:-2],4,y.shape[-1],device=y.device,dtype=y.dtype).index_add(y.dim()-2,RING,y);return (s/FI.RCOUNT.cuda().to(y.dtype)[:,None])[...,RING,:]
for s,p in pops:
    b=torch.load(bundle(s),map_location='cpu',weights_only=False);S=b['S'].double();K=S.shape[0];ref=S.mean(0).cuda()
    half=(S[:K//2].mean(0)-S[K//2:].mean(0)).cuda()/2   # replica-noise proxy of the mean
    q=FI.quad_ports(G,[p],0,C.CFG['candidate']['quadrature_q']);cells=torch.arange(16,device='cuda')+16*p
    h=FI.predict(m,G,q['W'],q['SS'],cells,G.dom.albedo).double()
    r,c,v=q['SS'];ss=torch.zeros(16,G.P*64,3,device='cuda',dtype=torch.float64).index_put((r-cells[0],c),v[:,None]*G.dom.albedo.double()[None],accumulate=True).view(16,G.P,64,3)
    A=lambda y,sp:FM.aggregate(y,sp,g['parent'],bm);msr=ref-ss;msh=h-ss
    rf=msr-ringproj(msr);nf=half-ringproj(half)
    out[f'rest_s{s}']=dict(port=p,ss_share=float(ss.sum()/ref.sum()),
        err_total={sp:rel(A(h,sp),A(ref,sp)) for sp in ('native','fine16','coarse')},err_ms={sp:rel(A(msh,sp),A(msr,sp)) for sp in ('native','fine16','coarse')},
        ring_floor=float(rf.norm()/msr.norm()),ring_noise=float(nf.norm()/msr.norm()),noise_native=float(half.norm()/ref.norm()),
        inc_profile=rel(h.sum((1,2)),ref.sum((1,2))),energy_ratio=float(h.sum()/ref.sum()))
    print(s,json.dumps({k:(round(v,3) if isinstance(v,float) else v) for k,v in out[f'rest_s{s}'].items() if k not in ('err_total','err_ms')}),
          {k:round(v,3) for k,v in out[f'rest_s{s}']['err_ms'].items()},flush=True)
    del S,ref,h,ss;torch.cuda.empty_cache()
agg={k:sum(v[k] for v in out.values())/len(out) for k in ('ss_share','ring_floor','ring_noise','inc_profile','energy_ratio')}
agg.update({f'err_ms_{sp}':sum(v['err_ms'][sp] for v in out.values())/len(out) for sp in ('native','fine16','coarse')})
json.dump(dict(run=run,label='POST HOC diagnostic (non-verdict), S16-R rest sources, archived bundle replica means',per_source=out,mean=agg),open(C.ROOT/'diag_posthoc76.json','w'),indent=1)
print('MEAN',json.dumps({k:round(v,3) for k,v in agg.items()}),flush=True)
