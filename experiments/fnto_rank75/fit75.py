"""WL75 Stage 1: rank-16 W+UV with the frozen G2 basis, exactly the WL73 AR recipe otherwise (fit73.run / spatial74.run):
S16-R fitting targets (replicas 0-3), objective J + 1.0 L_sym (exact-symmetry consistency, WL72 Candidate A), 'net' group (A/O frozen),
Adam, cosine 3e-3 -> 3e-5, 20,000 steps, clip 1.0, checkpoints 2,000 / 6,000 / 10,000, final iterate (no selection). The only change is
the rank-generic objective copy (ato74.objective_m == family68.objective at M = 8, test75) and the rank-16 network (rank75).
    python fit75.py run <run_id> | probe <run_id> <steps>"""
import sys,os,json,time,math
import torch
from common75 import CFG,ROOT,TRAIN,save_new,write_new,environment,dev
import rank75 as RK
import ato74 as AT
import sym72 as SY

def build(rid,RG):
    import fit68,fit72,fit73
    spec=CFG['runs'][rid];pb=fit73.Problem73(RG,'S16R');gs=dict(pb.gs)
    m,meta=RK.init_model(spec['init'],spec['rank']);params=fit68.group_params(m,'net')
    assert not m.in_logits.requires_grad and not m.out_logits.requires_grad
    cols=lambda g,ports:RK.columns(m,g,ports);lam=spec['lambda_sym'];sym=fit72.sym_data(RG,gs,pb.ports,pb.F);del pb.F
    def loss():
        Js=[AT.objective_m(*cols(gs[g],pb.ports),pb.tg[g]) for g in TRAIN];J=sum(Js)/len(Js);terms=[]
        for key,src,lab in sym:
            for c0 in range(0,len(src),64):
                ss=src[c0:c0+64];T,A,_=cols(gs[key],ss);d=(1-gs[key]['beta'][ss]).float();y=SY.model_integrated(T,A,d);t=lab[c0:c0+64]
                terms.append(((y-t).square().sum(1)/t.square().sum(1).clamp(min=1e-30)))
        L=torch.cat(terms).mean();return J+lam*L,J,L
    return spec,m,meta,params,loss,pb,sym

def run(rid,probe=None):
    import fit71
    RG=environment();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_num_threads(4)
    out=ROOT/'runs'/f'{rid}.pt' if probe is None else ROOT/'probe'/f'{rid}{os.environ.get("R75_PROBE_TAG","")}.json'
    if out.exists():raise FileExistsError('completed run is immutable')
    if probe is None:
        orphans=sorted((ROOT/'runs').glob(f'{rid}@*.pt'))
        if orphans:
            q=ROOT/'runs/quarantine'/str(time.time_ns());q.mkdir(parents=True)
            for p in orphans:os.replace(p,q/p.name)   # a crashed attempt is preserved; the run restarts from its initialisation
    oc=CFG['optimizer'];full=oc['steps'];steps=full if probe is None else int(probe)
    if 'R75_DEV_STEPS' in os.environ and probe is None:assert dev(),'dev override never in the evidence root';steps=int(os.environ['R75_DEV_STEPS'])
    torch.cuda.reset_peak_memory_stats();spec,m,meta,params,loss,pb,sym=build(rid,RG)
    a0=m.in_logits.detach().clone();o0=m.out_logits.detach().clone()
    opt=torch.optim.Adam(params,lr=oc['lr']);hist=[];t0=time.perf_counter()
    with torch.no_grad():tot0,J0,L0=(float(v) for v in loss())
    sched=full if probe is not None else steps   # probe: the real 20k schedule for its first steps
    ck={c*steps//full:c for c in oc['checkpoints']} if probe is None else {}
    tstep=[]
    for step in range(steps):
        torch.cuda.synchronize();ts=time.perf_counter()
        for q in opt.param_groups:q['lr']=fit71.cosine(step,sched,oc['lr'],oc['lr_min'])
        opt.zero_grad(set_to_none=True);tot,J,L=loss();tot.backward();gn=torch.nn.utils.clip_grad_norm_(params,oc['clip']);opt.step()
        if probe is not None:torch.cuda.synchronize();tstep.append(time.perf_counter()-ts)
        if step%25==0 or step==steps-1:
            v=(float(tot),float(J),float(L))
            if not all(math.isfinite(x) for x in v):raise RuntimeError('nonfinite objective: preserve failed run')
            hist.append(dict(step=step,loss=v[0],J=v[1],L_sym=v[2],grad_norm=float(gn),seconds=time.perf_counter()-t0))
            if step%1000==0:print('FIT',rid,step,v,flush=True)
        if step+1 in ck and step+1<steps:
            save_new(ROOT/'runs'/f'{rid}@{ck[step+1]}.pt',dict(model={k:v.detach().cpu() for k,v in m.state_dict().items()},step=step+1,spec=spec,rank=m.M))
    torch.cuda.synchronize();sec=time.perf_counter()-t0
    assert torch.equal(m.in_logits,a0) and torch.equal(m.out_logits,o0),'A/O changed'
    if probe is not None:
        w=sorted(tstep[5:]) or tstep;med=w[len(w)//2]
        write_new(out,dict(run=rid,steps=steps,seconds_per_step_median=med,extrapolated_20k_hours=med*full/3600,peak_GPU_allocated=torch.cuda.max_memory_allocated(),
            peak_GPU_reserved=torch.cuda.max_memory_reserved(),J_initial=J0,L_sym_initial=L0,loss_initial=tot0,concurrent=os.environ.get('R75_LANE'),note='non-evidence full-scale probe'))
        print('PROBE_DONE',rid,med,torch.cuda.max_memory_allocated(),flush=True);return
    with torch.no_grad():totf,Jf,Lf=(float(v) for v in loss())
    w=[h['loss'] for h in hist];n=len(w);aa=w[int(.8*n):int(.9*n)] or w[:1];bb=w[int(.9*n):] or w[-1:]
    rel=abs(sum(bb)/len(bb)-sum(aa)/len(aa))/max(sum(bb)/len(bb),1e-30)
    save_new(out,dict(model={k:v.detach().cpu() for k,v in m.state_dict().items()},spec=spec,rank=m.M,basis=meta,history=hist,loss_initial=tot0,J_initial=J0,L_sym_initial=L0,
        loss_final=totf,J_final=Jf,L_sym_final=Lf,last_decile_relative_change=rel,converged=rel<.01,seconds=sec,peak_GPU_allocated=torch.cuda.max_memory_allocated(),
        trainable=sum(p.numel() for p in params),parameters=RK.parameter_counts(m),sym_examples=sum(len(s[1]) for s in sym),ports=pb.ports,slots=pb.slots,
        concurrent=os.environ.get('R75_LANE')))
    print('RUN_DONE',rid,totf,Jf,Lf,round(sec),flush=True)

if __name__=='__main__':
    environment()
    if sys.argv[1]=='run':run(sys.argv[2])
    elif sys.argv[1]=='probe':run(sys.argv[2],int(sys.argv[3]))
    sys.stdout.flush();os._exit(0)
