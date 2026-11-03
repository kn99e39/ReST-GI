"""WL72 training (diagnostic copies; WL71 D1 data/objective/optimizer). Candidate A = WL59 W+UV ('net' group, fixed A/O) with the exact
symmetry-consistency term; Candidate B = RSK (rsk72) from a seed. Reads only WL71 fit targets/ports, training geometries, the
symmetry-variant geometries built here (geometry only), WL59 checkpoints and the WL68 net_all_5902 A/O.
    python fit72.py geoms | run <run_id>"""
import sys,os,json,time,math
from pathlib import Path
import torch
from common72 import CFG,ROOT,R71,INP68,save_new,sha,environment
import family68 as FM
import family71 as FW
import sym72 as SY
TRAIN=('rest','pt0');PT0=(0.00575,0.0225,0.005,-0.005)

def load(p):return torch.load(p,map_location='cpu',weights_only=False)

def variant_name(pose):return 'pt0' if tuple(pose)==PT0 else 'pt0v_%+.4f_%+.4f'%(pose[2],pose[3])

def build_geoms():
    """Symmetry images of pt0 (geometry only, unchanged r57_geometry with the image pose); saved once with hashes."""
    RG=environment();out=ROOT/'geom';out.mkdir(parents=True,exist_ok=True);rec={}
    for op in SY.ops(False):
        pose=SY.pose_image(PT0,op);nm=variant_name(pose)
        if nm=='pt0' or (out/f'{nm}.pt').exists():continue
        RG.POSES[nm]=pose;g=RG.geometry(nm,ballistic=True);save_new(out/f'{nm}.pt',RG.serial(g));rec[nm]=dict(pose=pose,sha256=sha(out/f'{nm}.pt'),meta=g['meta'])
        print('GEOM',nm,pose,flush=True)
    (out/'variants.json').write_text(json.dumps(rec,indent=1,default=str))

def geom(RG,name):
    if name in ('rest','pt0'):return RG.device(load(INP68/'wl59'/f'geometry_{name}.pt'))
    return RG.device(load(ROOT/'geom'/f'{name}.pt'))

def sym_data(RG,gs,ports,F):
    """List of (geometry key, transformed source ports, labels (n,P,3)) for every non-identity exact symmetry of the training data."""
    rest_xyz=gs['rest']['rest_xyz'].cpu().numpy();groups={}
    for gname in TRAIN:
        pose=(0.,.02,0.,0.) if gname=='rest' else PT0
        for op in SY.ops(gname=='rest'):
            if op==('id',False):continue
            perm=torch.as_tensor(SY.permutation(rest_xyz,op));key='rest' if gname=='rest' else variant_name(SY.pose_image(PT0,op))
            lab=torch.empty_like(F[gname]);lab[:,perm]=F[gname]
            src=[int(perm[p]) for p in ports];groups.setdefault(key,([],[]));groups[key][0].extend(src);groups[key][1].append(lab)
    out=[]
    for key,(src,labs) in groups.items():
        if key not in gs:gs[key]=geom(RG,key)
        out.append((key,src,torch.cat(labs).cuda()))
    return out

def model_for(spec,RG):
    import fit71
    if spec['candidate'] in ('A',):
        m=fit71.model_from(spec['init']);params=__import__('fit68').group_params(m,'net');cols=lambda g,ports:FW.network_columns(m,g,ports);return m,params,cols
    import rsk72
    a,o=fit71.angular();torch.manual_seed(spec['seed']);m=rsk72.RSK(a,o).cuda();params=[p for n,p in m.named_parameters() if p.requires_grad]
    return m,params,lambda g,ports:m.columns(g,ports)

def run(rid):
    spec=CFG['runs'][rid];RG=environment();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_num_threads(4);out=ROOT/'runs'/f'{rid}.pt'
    if out.exists():raise FileExistsError('completed run is immutable')
    import fit71
    steps=CFG['optimizer']['steps']
    if 'R72_DEV_STEPS' in os.environ:assert ROOT!=Path.home()/'rest_cmp72','dev override never in the evidence root';steps=int(os.environ['R72_DEV_STEPS'])
    pb=fit71.Problem(RG,dict(pop='S16',replicas=[0,1,2,3],target='phys'));gs=dict(pb.gs)
    m,params,cols=model_for(spec,RG);lam=spec.get('lambda_sym',0.)
    sym=[]
    if lam:
        S=load(R71/'fit_targets71.pt')['S']['r0123'];F={g:SY.integrated(S[g]) for g in TRAIN};del S
        sym=sym_data(RG,gs,pb.ports,F);del F
    def loss(parts=False):
        Js=[fit71.objective(*cols(gs[g],pb.ports),pb.tg[g],False) for g in TRAIN];J=sum(Js)/len(Js);L=torch.zeros((),device='cuda')
        if sym:
            terms=[]
            for key,src,lab in sym:
                for c0 in range(0,len(src),64):
                    ss=src[c0:c0+64];T,A,_=cols(gs[key],ss);d=(1-gs[key]['beta'][ss]).float();y=SY.model_integrated(T,A,d);t=lab[c0:c0+64]
                    terms.append(((y-t).square().sum(1)/t.square().sum(1).clamp(min=1e-30)))
            L=torch.cat(terms).mean()
        return J+lam*L,J,L
    a0=m.in_logits.detach().clone();o0=m.out_logits.detach().clone()
    opt=torch.optim.Adam(params,lr=CFG['optimizer']['lr']);hist=[];t0=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():tot0,J0,L0=(float(v) for v in loss())
    ck=set(CFG['optimizer']['checkpoints'])
    for step in range(steps):
        for q in opt.param_groups:q['lr']=fit71.cosine(step,steps,CFG['optimizer']['lr'],CFG['optimizer']['lr_min'])
        opt.zero_grad(set_to_none=True);tot,J,L=loss();tot.backward();gn=torch.nn.utils.clip_grad_norm_(params,CFG['optimizer']['clip']);opt.step()
        if step%25==0 or step==steps-1:
            v=(float(tot),float(J),float(L))
            if not all(math.isfinite(x) for x in v):raise RuntimeError('nonfinite objective: preserve failed run')
            hist.append(dict(step=step,loss=v[0],J=v[1],L_sym=v[2],grad_norm=float(gn),seconds=time.perf_counter()-t0))
            if step%1000==0:print('FIT',rid,step,v,flush=True)
        if step+1 in ck and step+1<steps:
            save_new(ROOT/'runs'/f'{rid}@{step+1}.pt',dict(model={k:v.detach().cpu() for k,v in m.state_dict().items()},step=step+1,spec=spec))
    torch.cuda.synchronize();sec=time.perf_counter()-t0
    with torch.no_grad():totf,Jf,Lf=(float(v) for v in loss())
    assert torch.equal(m.in_logits,a0) and torch.equal(m.out_logits,o0),'A/O changed'
    w=[h['loss'] for h in hist];n=len(w);aa=w[int(.8*n):int(.9*n)] or w[:1];bb=w[int(.9*n):] or w[-1:]
    rel=abs(sum(bb)/len(bb)-sum(aa)/len(aa))/max(sum(bb)/len(bb),1e-30)
    save_new(out,dict(model={k:v.detach().cpu() for k,v in m.state_dict().items()},spec=spec,history=hist,loss_initial=tot0,J_initial=J0,L_sym_initial=L0,
        loss_final=totf,J_final=Jf,L_sym_final=Lf,last_decile_relative_change=rel,converged=rel<.01,seconds=sec,peak_GPU_allocated=torch.cuda.max_memory_allocated(),
        trainable=sum(p.numel() for p in params),sym_examples=sum(len(s[1]) for s in sym),concurrent=os.environ.get('R72_LANE')))
    print('RUN_DONE',rid,totf,Jf,Lf,round(sec),flush=True)

if __name__=='__main__':
    environment();{'geoms':lambda:build_geoms(),'run':lambda:run(sys.argv[2])}[sys.argv[1]]();sys.stdout.flush();os._exit(0)
