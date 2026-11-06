"""WL76 Stage 3 fit: the FIA learned response R_theta on the S16-R training sources (fitting replicas 0-3 means, fit_targets73.pt) at rest
and pt0, objective = the WL68 objective J (family68 definition, computed on the dense prediction), Adam, cosine lr 3e-3 -> 3e-5, clip 1.0,
20,000 full-batch steps, checkpoints 2,000 / 6,000 / 10,000, final iterate (no selection). W / SS are fixed geometry-only quadratures
(q per protocol, per-port seeds) and are not trained. No symmetry term: the learned part is exactly equivariant by construction.
    python fit76.py run <run_id> | probe <run_id> <steps>"""
import sys,os,json,time,math
import torch
from common76 import CFG,ROOT,R73,TRAIN,save_new,write_new,environment,dev
import family68 as FM
import fia76 as FI

def load(p,mmap=False):return torch.load(p,map_location='cpu',weights_only=False,mmap=mmap)

def problem(RG):
    """Training geometries, Geo state, quadrature rows of the 16 S16-R ports and native targets (rest, pt0)."""
    import fit71,fit73,probes68 as P68
    gs=fit71.geometries(RG);ports=[p for _,p in fit73.populations()['S16R']];t=load(R73/'fit_targets73.pt',mmap=True);assert t['ports']==ports
    bm=FM.bin_map('cuda');q=CFG['candidate']['quadrature_q'];out={}
    for gi,g in enumerate(TRAIN):
        G=FI.Geo(gs[g],P68.domain(RG,g));Q=FI.quad_ports(G,ports,CFG['geometry_index'][g],q)
        cells=(torch.as_tensor(ports,device='cuda')[:,None]*16+torch.arange(16,device='cuda')[None]).reshape(-1)
        S=t['S']['r0123'][g].float().cuda();d=(1-gs[g]['beta'][ports]).float().cuda()
        out[g]=dict(G=G,Q=Q,cells=cells,tg=FI.target(S,d,gs[g]['parent'],bm),alb=G.dom.albedo,n=len(ports));del S;torch.cuda.empty_cache()
    return out,ports

def loss_and_backward(model,pb,chunk,backward=True):
    """Full-batch J (mean over geometries of the mean over (source, channel)), accumulated over source chunks."""
    tot=0.;parts={}
    for g,p in pb.items():
        n=p['n'];cnt=n*3*len(pb)
        for s0 in range(0,n,chunk):
            idx=torch.arange(s0,min(s0+chunk,n),device='cuda');cells=p['cells'].view(n,16)[idx].reshape(-1)
            Sh=FI.predict(model,p['G'],p['Q']['W'],p['Q']['SS'],cells,p['alb']).view(len(idx),16,p['G'].P,64,3)
            J,pt=FI.objective(Sh,p['tg'],idx);L=J/cnt
            if backward:L.backward()
            tot+=float(L);parts.setdefault(g,[]).append({k:v.cpu() for k,v in pt.items()});del Sh
    return tot,parts

def run(rid,probe=None):
    import fit71
    RG=environment();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_num_threads(4)
    spec=CFG['runs'][rid];oc=CFG['optimizer'];out=ROOT/'runs'/f'{rid}.pt' if probe is None else ROOT/'probe'/f'{rid}{os.environ.get("R76_PROBE_TAG","")}.json'
    if out.exists():raise FileExistsError('completed run is immutable')
    if probe is None:
        orphans=sorted((ROOT/'runs').glob(f'{rid}@*.pt'))
        if orphans:
            q=ROOT/'runs/quarantine'/str(time.time_ns());q.mkdir(parents=True)
            for p in orphans:os.replace(p,q/p.name)
    full=oc['steps'];steps=full if probe is None else int(probe)
    if 'R76_DEV_STEPS' in os.environ and probe is None:assert dev();steps=int(os.environ['R76_DEV_STEPS'])
    torch.cuda.reset_peak_memory_stats();t0=time.perf_counter();pb,ports=problem(RG);prep=time.perf_counter()-t0
    torch.manual_seed(spec['seed']);model=FI.FIA(CFG['candidate']['hidden']).cuda();params=list(model.parameters())
    opt=torch.optim.Adam(params,lr=oc['lr']);hist=[];ck={c*steps//full:c for c in oc['checkpoints']} if probe is None else {};tstep=[]
    with torch.no_grad():J0,_=loss_and_backward(model,pb,oc['chunk'],False)
    t0=time.perf_counter()
    for step in range(steps):
        torch.cuda.synchronize();ts=time.perf_counter()
        for q in opt.param_groups:q['lr']=fit71.cosine(step,full if probe is not None else steps,oc['lr'],oc['lr_min'])
        opt.zero_grad(set_to_none=True);J,_=loss_and_backward(model,pb,oc['chunk']);gn=torch.nn.utils.clip_grad_norm_(params,oc['clip']);opt.step()
        torch.cuda.synchronize();tstep.append(time.perf_counter()-ts)
        if not math.isfinite(J):raise RuntimeError('nonfinite objective: preserve failed run')
        if step%25==0 or step==steps-1:
            hist.append(dict(step=step,J=J,grad_norm=float(gn),seconds=time.perf_counter()-t0))
            if step%1000==0:print('FIT',rid,step,J,flush=True)
        if step+1 in ck and step+1<steps:save_new(ROOT/'runs'/f'{rid}@{ck[step+1]}.pt',dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},step=step+1,spec=spec))
    torch.cuda.synchronize();sec=time.perf_counter()-t0
    if probe is not None:
        w=sorted(tstep[3:]) or tstep;med=w[len(w)//2]
        write_new(out,dict(run=rid,steps=steps,seconds_per_step_median=med,extrapolated_hours=med*full/3600,peak_GPU_allocated=torch.cuda.max_memory_allocated(),
            peak_GPU_reserved=torch.cuda.max_memory_reserved(),J_initial=J0,prep_seconds=prep,concurrent=os.environ.get('R76_LANE'),note='non-evidence full-scale probe'))
        print('PROBE_DONE',rid,med,torch.cuda.max_memory_reserved(),flush=True);return
    with torch.no_grad():Jf,parts=loss_and_backward(model,pb,oc['chunk'],False)
    w=[h['J'] for h in hist];n=len(w);aa=w[int(.8*n):int(.9*n)] or w[:1];bb=w[int(.9*n):] or w[-1:]
    rel=abs(sum(bb)/len(bb)-sum(aa)/len(aa))/max(sum(bb)/len(bb),1e-30)
    save_new(out,dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},spec=spec,history=hist,J_initial=J0,J_final=Jf,parts_final=parts,
        last_decile_relative_change=rel,converged=rel<.01,seconds=sec,prep_seconds=prep,peak_GPU_allocated=torch.cuda.max_memory_allocated(),
        trainable=sum(p.numel() for p in params),ports=ports,concurrent=os.environ.get('R76_LANE')))
    print('RUN_DONE',rid,Jf,round(sec),flush=True)

if __name__=='__main__':
    environment()
    if sys.argv[1]=='run':run(sys.argv[2])
    elif sys.argv[1]=='probe':run(sys.argv[2],int(sys.argv[3]))
    sys.stdout.flush();os._exit(0)
