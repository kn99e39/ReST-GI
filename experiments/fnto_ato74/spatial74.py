"""WL74 Stage 3 (conditional): WL72 Candidate A (W+UV + exact symmetry L_sym, lambda 1) on S16-R with the frozen G1 shared rank-8 A/O
instead of the WL68 A/O; everything else exactly as the WL73 AR runs (fit73.run): P_590x spatial initialisation, 'net' group, objective J,
Adam, cosine 3e-3 -> 3e-5, 20,000 steps, clip 1.0, final iterate. Controls: frozen WL73 AR_5901 / AR_5902.
    python spatial74.py run <run_id>"""
import sys,os,json,time,math
import torch
from common74 import CFG,ROOT,R73,save_new,environment,dev
import family71 as FW
import sym72 as SY

def load(p):return torch.load(p,map_location='cpu',weights_only=False)

def basis_logits():
    d=json.loads((ROOT/'decision.json').read_text());bid=d['chosen_basis']
    if not (d['stage3_gate'] and bid):
        assert dev() and os.environ.get('R74_DEV_FORCE_GATE')=='1','Stage 3 not authorised by the gate';bid='G1_r0'   # dev smoke only
    b=load(ROOT/'bases'/f"{bid}.pt");assert b['rank']==8;return b['in_logits'],b['out_logits'],bid

def run(rid):
    import fit71,fit72,fit73
    spec=CFG['stage3']['runs'][rid];RG=environment();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_num_threads(4);out=ROOT/'runs'/f'{rid}.pt'
    if out.exists():raise FileExistsError('completed run is immutable')
    orphans=sorted((ROOT/'runs').glob(f'{rid}@*.pt'))
    if orphans:
        q=ROOT/'runs/quarantine'/str(time.time_ns());q.mkdir(parents=True)
        for p in orphans:os.replace(p,q/p.name)
    oc=CFG['stage3']['optimizer'];steps=oc['steps']
    if 'R74_DEV_STEPS' in os.environ:assert dev();steps=int(os.environ['R74_DEV_STEPS'])
    a_new,o_new,bid=basis_logits()
    pb=fit73.Problem73(RG,'S16R');gs=dict(pb.gs)
    m,params,cols=fit72.model_for(dict(candidate='A',init=spec['init']),RG)
    with torch.no_grad():m.in_logits.copy_(a_new.cuda());m.out_logits.copy_(o_new.cuda())
    lam=spec['lambda_sym'];sym=fit72.sym_data(RG,gs,pb.ports,pb.F);del pb.F
    def loss():
        Js=[fit71.objective(*cols(gs[g],pb.ports),pb.tg[g],False) for g in ('rest','pt0')];J=sum(Js)/len(Js);terms=[]
        for key,src,lab in sym:
            for c0 in range(0,len(src),64):
                ss=src[c0:c0+64];T,A,_=cols(gs[key],ss);d=(1-gs[key]['beta'][ss]).float();y=SY.model_integrated(T,A,d);t=lab[c0:c0+64]
                terms.append(((y-t).square().sum(1)/t.square().sum(1).clamp(min=1e-30)))
        L=torch.cat(terms).mean();return J+lam*L,J,L
    a0=m.in_logits.detach().clone();o0=m.out_logits.detach().clone()
    opt=torch.optim.Adam(params,lr=oc['lr']);hist=[];t0=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():tot0,J0,L0=(float(v) for v in loss())
    ck={c*steps//oc['steps']:c for c in oc['checkpoints']}
    for step in range(steps):
        for q in opt.param_groups:q['lr']=fit71.cosine(step,steps,oc['lr'],oc['lr_min'])
        opt.zero_grad(set_to_none=True);tot,J,L=loss();tot.backward();gn=torch.nn.utils.clip_grad_norm_(params,oc['clip']);opt.step()
        if step%25==0 or step==steps-1:
            v=(float(tot),float(J),float(L))
            if not all(math.isfinite(x) for x in v):raise RuntimeError('nonfinite objective: preserve failed run')
            hist.append(dict(step=step,loss=v[0],J=v[1],L_sym=v[2],grad_norm=float(gn),seconds=time.perf_counter()-t0))
            if step%1000==0:print('FIT',rid,step,v,flush=True)
        if step+1 in ck and step+1<steps:
            save_new(ROOT/'runs'/f'{rid}@{ck[step+1]}.pt',dict(model={k:v.detach().cpu() for k,v in m.state_dict().items()},step=step+1,spec=spec))
    torch.cuda.synchronize();sec=time.perf_counter()-t0
    with torch.no_grad():totf,Jf,Lf=(float(v) for v in loss())
    assert torch.equal(m.in_logits,a0) and torch.equal(m.out_logits,o0),'A/O changed'
    w=[h['loss'] for h in hist];n=len(w);aa=w[int(.8*n):int(.9*n)] or w[:1];bb=w[int(.9*n):] or w[-1:]
    rel=abs(sum(bb)/len(bb)-sum(aa)/len(aa))/max(sum(bb)/len(bb),1e-30)
    save_new(out,dict(model={k:v.detach().cpu() for k,v in m.state_dict().items()},spec=spec,basis=bid,history=hist,loss_initial=tot0,J_initial=J0,L_sym_initial=L0,
        loss_final=totf,J_final=Jf,L_sym_final=Lf,last_decile_relative_change=rel,converged=rel<.01,seconds=sec,peak_GPU_allocated=torch.cuda.max_memory_allocated(),
        trainable=sum(p.numel() for p in params),concurrent=os.environ.get('R74_LANE')))
    print('RUN_DONE',rid,totf,Jf,Lf,round(sec),flush=True)

if __name__=='__main__':
    environment();run(sys.argv[2]);sys.stdout.flush();os._exit(0)
