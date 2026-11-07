"""WL77 Stage 1 (angular expressivity gate, training-reference replicas only) and Stage 2 (full experimental cost probe).
expr: for the 16 S16-R rest sources, the reference multiple-scatter part M = mean(S) - albedo SS (exact single scatter, same quadrature) is
      projected per (incoming bin, receiver, channel) onto (i) ring-uniform functions (4; the WL76 restriction), (ii) the FIA-AZ span
      {ring indicator x (1, cos(phi_o - phi_r), cos 2(phi_o - phi_r))} (12; phi_r = azimuth of the source -> receiver direction in the
      receiver frame; a relaxation of the trained family), (iii) the free-phase span {ring x (1, cos, sin, cos2, sin2)(phi_o)} (20, reference).
      Residual relative L2 per source and its replica-noise floor (half-split difference). Gate (protocol stage1.gate): mean (ii) floor
      <= 0.5 x mean (i) floor, else STOP (the angular restriction is not relaxed).
cost: a full FIA-AZ runtime (initial parameters; cost does not depend on trained weights) on rest and pt0: state, rebuild, 1/64-light apply,
      peak memory, physics; the rank-16 baseline apply in the same process. Gate: state <= 4 GiB and 64-light apply <= 10 x A75, else STOP.
    python gate77.py expr | cost"""
import sys,os,json,time
import torch
from common77 import CFG,ROOT,R68,R71,R73,INP68,write_new,environment,dev
import family68 as FM
import fia77 as FI

def load(p,mmap=False):return torch.load(p,map_location='cpu',weights_only=False,mmap=mmap)
def bundle(s):
    if s<4:return INP68/'fit'/f'rest_s{s}.pt'
    if s<8:return R68/'heldout/bundles'/f'rest_s{s}.pt'
    if s<16:return R71/'refs/bundles'/f'rest_s{s}.pt'
    return R73/'refs/bundles'/f'rest_s{s}.pt'

def basis_sets(phi_o,phi_r):
    """phi_o (64,), phi_r (P,) -> dict of (P, 64, nb) bases."""
    ring=torch.nn.functional.one_hot(FI.RING.to(phi_o.device),4).double()                                    # (64, 4)
    d=phi_o[None,:]-phi_r[:,None]
    R4=ring[None].expand(len(phi_r),-1,-1)
    AZ=torch.cat([R4,R4*torch.cos(d)[...,None],R4*torch.cos(2*d)[...,None]],2)
    h=torch.stack([torch.ones_like(phi_o),torch.cos(phi_o),torch.sin(phi_o),torch.cos(2*phi_o),torch.sin(2*phi_o)],1)   # (64,5)
    FREE=(ring[:,:,None]*h[:,None,:]).reshape(64,20)[None].expand(len(phi_r),-1,-1)
    return dict(ring=R4,az=AZ,free=FREE)

def residual(Y,B):
    """Y (n_a, P, 64, C), B (P, 64, nb): per-receiver least-squares residual (same B for every a, c)."""
    Q,_=torch.linalg.qr(B.double());Yd=Y.double().permute(1,2,0,3).reshape(B.shape[0],64,-1)                  # (P,64,a*C)
    return Yd-torch.bmm(Q,torch.bmm(Q.transpose(1,2),Yd))

def expr():
    RG=environment();torch.set_grad_enabled(False);out=ROOT/'stage1/expressivity.json'
    if out.exists():raise FileExistsError('immutable')
    import fit71,fit73,probes68 as P68
    gs=fit71.geometries(RG);g=gs['rest'];G=FI.attach_frames(FI.Geo(g,P68.domain(RG,'rest')));phi_o,_=FI.bin_azimuth();phi_o=phi_o.double().cuda();t,b=G.frames;rows={}
    for s,p in fit73.populations()['S16R']:
        bb=load(bundle(s));S=bb['S'].double();K=S.shape[0];ref=S.mean(0).cuda();half=(S[:K//2].mean(0)-S[K//2:].mean(0)).cuda()/2
        q=FI.quad_ports(G,[p],0,CFG['candidate']['quadrature_q']);r,c,v=q['SS'];cells=torch.arange(16,device='cuda')+16*p
        ss=torch.zeros(16,G.P*64,3,device='cuda',dtype=torch.float64).index_put((r-cells[0],c),v[:,None]*G.dom.albedo.double()[None],accumulate=True).view(16,G.P,64,3)
        M=ref-ss;rr=G.x.double()-G.x[p].double();rn=(rr*G.n.double()).sum(1,keepdim=True);rt=rr-rn*G.n.double()
        phi_r=torch.atan2((rt*b.double()).sum(1),(rt*t.double()).sum(1));Bs=basis_sets(phi_o,phi_r);den=M.norm()
        row={}
        for k,B in Bs.items():
            row[k]=float(residual(M,B).norm()/den);row[k+'_noise']=float(residual(half,B).norm()/den)
        row['ss_share']=float(ss.sum()/ref.sum());rows[f'rest_s{s}']=row;print('EXPR',s,{k:round(v,4) for k,v in row.items()},flush=True)
        del S,ref,half,ss,M;torch.cuda.empty_cache()
    mean={k:sum(v[k] for v in rows.values())/len(rows) for k in next(iter(rows.values()))}
    gc=CFG['stage1']['gate'];passed=mean['az']<=gc['factor']*mean['ring']
    write_new(out,dict(per_source=rows,mean=mean,gate=dict(rule=gc['rule'],az=mean['az'],ring=mean['ring'],factor=gc['factor'],passed=passed),
        label='training-reference replicas only (archived bundles); projection floors are expressivity bounds of linear relaxations, not trained-model accuracy'))
    print('STAGE1_EXPR',passed,json.dumps({k:round(v,4) for k,v in mean.items()}),flush=True)
    if not passed and not (dev() and os.environ.get('R77_DEV_SKIP_GATES')=='1'):raise SystemExit('STOP: the azimuth-resolving representation does not relax the ring restriction (protocol stage1.gate)')

def cost():
    RG=environment();torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;out=ROOT/'stage2/cost_probe.json'
    if out.exists():raise FileExistsError('immutable')
    import fit71,probes68 as P68,eval77 as EV
    gs=fit71.geometries(RG);lights=load(INP68/'wl59/lights.pt')['test'].cuda();torch.manual_seed(7700);m=FI.FIA(CFG['candidate']['hidden']).cuda().eval();res={}
    def med(f,n=5):
        f();ts=[]
        for _ in range(n):torch.cuda.synchronize();t0=time.perf_counter();f();torch.cuda.synchronize();ts.append(time.perf_counter()-t0)
        return sorted(ts)[n//2]
    mb,_=EV.baseline_h('A75_5901');co=mb.coefficients(gs['rest']);a75=med(lambda:[mb.apply(gs['rest'],lights[i:i+4],co) for i in range(0,64,4)]);del mb,co;torch.cuda.empty_cache()
    for gn in ('rest','pt0'):
        g=gs[gn];torch.cuda.reset_peak_memory_stats();G=FI.Geo(g,P68.domain(RG,gn));rt=FI.Runtime(m,G,CFG['geometry_index'][gn],CFG['candidate']['quadrature_q']);alb=G.dom.albedo
        ph=FI.physics(rt,g,lights[:2],alb)
        res[gn]=dict(build=rt.seconds,bytes=rt.bytes,state_bytes=sum(rt.bytes.values()),apply64_s=med(lambda:[rt.apply(g,lights[i:i+4],alb) for i in range(0,64,4)]),
                     apply1_s=med(lambda:rt.apply(g,lights[:1],alb)),peak_GPU=torch.cuda.max_memory_allocated(),physics=ph)
        del rt,G;torch.cuda.empty_cache();print('COST',gn,json.dumps({k:v for k,v in res[gn].items() if k!='bytes'},default=float),flush=True)
    gc=CFG['stage2']['gate'];worst=max(res,key=lambda k:res[k]['state_bytes'])
    gate=dict(state_bytes=res[worst]['state_bytes'],state_max=gc['state_bytes_max'],apply64_s=res['rest']['apply64_s'],A75_apply64_s=a75,ratio=res['rest']['apply64_s']/a75,ratio_max=gc['apply64_ratio_max'],
              physics=all(v['physics']['passed'] for v in res.values()))
    gate['passed']=gate['state_bytes']<=gate['state_max'] and gate['ratio']<=gate['ratio_max'] and gate['physics']
    write_new(out,json.loads(json.dumps(dict(geometries=res,gate=gate,note='initial parameters; cost depends on structure, not weights; the WL76 quadrature (q 1024) is unchanged'),default=float)))
    print('STAGE2_COST',gate['passed'],json.dumps(gate,default=float),flush=True)
    if not gate['passed'] and not (dev() and os.environ.get('R77_DEV_SKIP_GATES')=='1'):raise SystemExit('STOP: COST INFEASIBLE (protocol stage2.gate)')

if __name__=='__main__':
    {'expr':expr,'cost':cost}[sys.argv[1]]();sys.stdout.flush();os._exit(0)
