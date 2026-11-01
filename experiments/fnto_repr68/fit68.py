"""Constructive representability fits (diagnostic objective, experimental copies only). Reads ONLY fit_targets.pt
(mean of fitting replicas 0-3 of the WL66 probes), geometry and checkpoints; never a held-out reference or eval replica.
Run: python fit68.py prepare | python fit68.py run <run_id>."""
import sys,os,json,time,math,argparse
from pathlib import Path
import torch
from common68 import CFG,ROOT,INP,FIT_GEOMS,sha,tensor_sha,write_new,save_new,environment
import family68 as FM

ALLOWED=('fit_targets.pt','wl59/geometry_rest.pt','wl59/geometry_pt0.pt','wl59/P_5901.pt','wl59/P_5902.pt','wl66_ports.json')
def guarded(rel):
    assert rel in ALLOWED or rel.startswith('runs/'),('fit stage may not read',rel)
    assert 'heldout' not in rel and 'eval' not in rel
    p=INP/rel if not rel.startswith('runs/') else ROOT/rel
    return torch.load(p,map_location='cpu',weights_only=False) if p.suffix=='.pt' else json.loads(p.read_text())

def fit_ports():return [s['port'] for s in sorted(guarded('wl66_ports.json')['ports'],key=lambda s:s['slot'])]

def prepare():
    """Mean of the declared fitting replicas, per geometry (n=4 slots). The only physical data the fits can see."""
    out=INP/'fit_targets.pt';K=CFG['populations']['fit']['fit_replicas'];data={}
    for g in FIT_GEOMS:
        S=[torch.load(INP/'fit'/f'{g}_s{s}.pt',map_location='cpu')['S'][K].double().mean(0).float() for s in range(4)]
        data[g]=torch.stack(S)
    save_new(out,dict(S=data,replicas=K,ports=fit_ports()));print('PREPARED',sha(out),flush=True)

def geometries(RG):
    gs={}
    for g in FIT_GEOMS:
        x=RG.device(guarded(f'wl59/geometry_{g}.pt'));gs[g]=x
    return gs

class Problem:
    def __init__(self,RG,targets=None):
        self.gs=geometries(RG);self.ports=fit_ports();self.bm=FM.bin_map('cuda')
        self.d={g:(1-self.gs[g]['beta'][self.ports]).float() for g in FIT_GEOMS}
        S=guarded('fit_targets.pt')['S'] if targets is None else targets
        self.tg={g:FM.Target(S[g].cuda(),self.d[g],self.gs[g]['parent'],self.bm) for g in FIT_GEOMS}
    def loss(self,cols,parts=False):
        vals=[FM.objective(*cols(g),self.tg[g],parts=parts) for g in FIT_GEOMS]
        if not parts:return sum(vals)/len(vals)
        return sum(v[0] for v in vals)/len(vals),{g:{k:x.cpu().tolist() for k,x in v[1].items()} for g,v in zip(FIT_GEOMS,vals)}

def model_from(spec):
    from runtime import Transport
    m=Transport().cuda()
    if spec.startswith('P_'):m.load_state_dict(guarded(f'wl59/{spec}.pt')['model'])
    elif spec.startswith('scratch'):
        torch.manual_seed(int(spec.split('_')[1]));m=Transport().cuda()
    elif spec.startswith('runs/'):m.load_state_dict(guarded(spec)['model'])
    else:raise ValueError(spec)
    return m

def group_params(m,group):
    names={'A':{'in_logits'},'O':{'out_logits'},'AO':{'in_logits','out_logits'}}
    sel=[]
    for n,p in m.named_parameters():
        on=(n in names[group]) if group in names else (n not in names['AO']) if group=='net' else True if group=='all' else None
        assert on is not None,group;p.requires_grad_(on)
        if on:sel.append(p)
    return sel

def teacher_targets(RG,spec):
    """Noise-free same-family targets from a KNOWN admissible parameter setting (synthetic recovery controls)."""
    gs=geometries(RG);ports=fit_ports();out={}
    with torch.no_grad():
        if spec['kind']=='net':
            t=model_from(spec['model']).eval()
            if 'angular_from' in spec:
                src=best_coef() if spec['angular_from']=='best_physical_coef' else spec['angular_from']
                fc=guarded(src)['model'];t.in_logits.copy_(fc['in_logits']);t.out_logits.copy_(fc['out_logits'])
            for g in FIT_GEOMS:
                T,A,O=FM.network_columns(t,gs[g],ports);d=(1-gs[g]['beta'][ports]).float()
                out[g]=FM.predict(T,d,A,O,'native').cpu()
        else:  # coefficient-level teacher: random global A/O logits, spatial T of a WL59 checkpoint
            gen=torch.Generator().manual_seed(spec['seed']);t=model_from(spec['spatial_from']).eval()
            A=torch.softmax(torch.randn(8,16,3,generator=gen)*spec['logit_sd'],0).cuda();O=torch.softmax(torch.randn(8,64,3,generator=gen)*spec['logit_sd'],1).cuda()
            for g in FIT_GEOMS:
                T,_,_=FM.network_columns(t,gs[g],ports);d=(1-gs[g]['beta'][ports]).float();out[g]=FM.predict(T,d,A,O,'native').cpu()
    return out

def best_coef():
    """Predeclared: the physical coefficient-level restart with the lowest final TRAINING objective."""
    ids=[r for r,v in CFG['runs'].items() if v['kind']=='coef' and v['target']=='phys']
    J={r:guarded(f'runs/{r}.pt')['J_final'] for r in ids};return f'runs/{min(J,key=J.get)}.pt'

def cosine(step,steps,lr,lr_min):return lr_min+.5*(lr-lr_min)*(1+math.cos(math.pi*step/steps))

def run(rid):
    spec=CFG['runs'][rid];RG=environment();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_num_threads(8);out=ROOT/'runs'/f'{rid}.pt'
    if out.exists():raise FileExistsError('completed run is immutable')
    targets=None if spec['target']=='phys' else teacher_targets(RG,CFG['teachers'][spec['target']])
    pb=Problem(RG,targets);torch.cuda.reset_peak_memory_stats();t0=time.perf_counter();hist=[]
    opt_cfg=CFG['optimizer'][spec['kind']];steps=opt_cfg['steps']
    if 'R68_DEV_STEPS' in os.environ:assert ROOT!=Path.home()/'rest_repr68','dev override never in the evidence root';steps=int(os.environ['R68_DEV_STEPS'])
    if spec['kind']=='net':
        m=model_from(spec['init']);m.train();params=group_params(m,spec['group'])
        cols=lambda g:FM.network_columns(m,pb.gs[g],pb.ports);state=lambda:{k:v.detach().cpu() for k,v in m.state_dict().items()}
    elif spec['kind']=='coef':
        with torch.no_grad():
            if spec['init'].startswith('P_'):
                src=model_from(spec['init']);Ts={g:FM.network_columns(src,pb.gs[g],pb.ports)[0] for g in FIT_GEOMS}
                Al,Ol=src.in_logits.detach(),src.out_logits.detach()
            else:
                gen=torch.Generator().manual_seed(int(spec['init'].split('_')[1]))
                Al=(torch.randn(8,16,3,generator=gen)*.05).cuda();Ol=(torch.randn(8,64,3,generator=gen)*.05).cuda()
                Ts={g:torch.full((4,pb.gs[g]['P'],8,3),.5/pb.gs[g]['P'],device='cuda') for g in FIT_GEOMS}
            ml,sl=zip(*[FM.FCoef.logits_from(Ts[g]) for g in FIT_GEOMS])
        m=FM.FCoef(Al,Ol,ml,sl).cuda();params=list(m.parameters());idx={g:k for k,g in enumerate(FIT_GEOMS)}
        cols=lambda g:m.columns(idx[g]);state=lambda:{k:v.detach().cpu() for k,v in m.state_dict().items()}
    else:raise ValueError(spec['kind'])
    opt=torch.optim.Adam(params,lr=opt_cfg['lr']);J0,parts0=pb.loss(cols,parts=True);J0=float(J0)
    for step in range(steps):
        for q in opt.param_groups:q['lr']=cosine(step,steps,opt_cfg['lr'],opt_cfg['lr_min'])
        opt.zero_grad(set_to_none=True);J=pb.loss(cols)
        if not torch.isfinite(J):raise RuntimeError('nonfinite objective: preserve failed run')
        J.backward();gn=float(torch.nn.utils.clip_grad_norm_(params,opt_cfg['clip']));opt.step()
        if step%opt_cfg['log_every']==0 or step==steps-1:
            hist.append(dict(step=step,J=float(J),grad_norm=gn,lr=opt.param_groups[0]['lr'],seconds=time.perf_counter()-t0))
            if step%(opt_cfg['log_every']*20)==0:print('FIT',rid,step,float(J),flush=True)
    torch.cuda.synchronize();sec=time.perf_counter()-t0
    with torch.no_grad():Jf,parts=pb.loss(cols,parts=True)
    w=[h['J'] for h in hist];n=len(w);a=w[int(.8*n):int(.9*n)] or w[:1];b=w[int(.9*n):] or w[-1:]
    rel_change=abs(sum(b)/len(b)-sum(a)/len(a))/max(sum(b)/len(b),1e-30)
    save_new(out,dict(model=state(),spec=spec,history=hist,J_initial=J0,parts_initial=parts0,J_final=float(Jf),parts_final=parts,
        last_decile_relative_change=rel_change,converged=rel_change<CFG['optimizer']['convergence_relative_change'],
        seconds=sec,peak_GPU_allocated=torch.cuda.max_memory_allocated(),trainable=sum(p.numel() for p in params)))
    print('RUN_DONE',rid,float(Jf),round(sec),flush=True)

def convex(rid):
    """Exact convex floors at FROZEN checkpoint angular bases (float64 FISTA, capped-simplex projection)."""
    spec=CFG['convex'][rid];RG=environment();torch.backends.cuda.matmul.allow_tf32=False;out=ROOT/'runs'/f'{rid}.pt'
    if out.exists():raise FileExistsError('completed run is immutable')
    pb=Problem(RG);src=model_from(spec['init']).eval();t0=time.perf_counter();res={}
    with torch.no_grad():A=torch.softmax(src.in_logits,0).double();O=torch.softmax(src.out_logits,1).double()
    for g in FIT_GEOMS:
        tg=pb.tg[g];d=tg.d.double();Hm=FM.Hmat('cuda',torch.float64);HZ=tg.HZ.double();den=tg.den['con'].double()
        with torch.no_grad():T0=FM.network_columns(src,pb.gs[g],pb.ports)[0].double()
        if spec['objective']=='J':
            S={s:v.double() for s,v in tg.S.items()};dn={s:tg.den[s].double() for s in S}
            def f(T):
                v=sum((FM.predict(T,d,A,O,s,tg.parent,tg.bm)-S[s]).square().sum((1,2,3))/dn[s] for s in S)/3
                zc=FM.predict(T,d,A,O,'coarse',tg.parent,tg.bm)/d[:,:,None,None,None]
                return (v+(torch.einsum('ab,nbqoc->naqoc',Hm,zc)-HZ).square().sum((1,2,3))/den).sum()
            proj=lambda T:FM.project_capped(T.permute(0,2,3,1)).permute(0,3,1,2);x0=T0
        elif spec['objective']=='con_AO':
            O16=torch.zeros(8,16,3,dtype=torch.float64,device='cuda').index_add(1,tg.bm,O)
            def f(Tc):return (torch.einsum('ab,nbqoc->naqoc',Hm,torch.einsum('rac,njrc,roc->najoc',A,Tc,O16))-HZ).square().sum((1,2,3)).div(den).sum()
            proj=lambda T:FM.project_capped(T.permute(0,2,3,1)).permute(0,3,1,2)
            x0=torch.zeros(4,240,8,3,dtype=torch.float64,device='cuda').index_add(1,tg.parent,T0)
        elif spec['objective']=='con_A_freeout':
            def f(Mb):return (torch.einsum('ab,nbqoc->naqoc',Hm,torch.einsum('rac,njroc->najoc',A,Mb))-HZ).square().sum((1,2,3)).div(den).sum()
            proj=lambda X:FM.project_capped(X.permute(0,2,4,1,3).reshape(4,8,3,-1)).reshape(4,8,3,240,16).permute(0,3,1,4,2)
            O16=torch.zeros(8,16,3,dtype=torch.float64,device='cuda').index_add(1,tg.bm,O)
            x0=torch.einsum('njrc,roc->njroc',torch.zeros(4,240,8,3,dtype=torch.float64,device='cuda').index_add(1,tg.parent,T0),O16)
        iters=spec['iters']
        if 'R68_DEV_STEPS' in os.environ:assert ROOT!=Path.home()/'rest_repr68';iters=20
        x,info=FM.fista(f,x0,proj,iters);res[g]=dict(x=x.float().cpu(),info=info)
    save_new(out,dict(spec=spec,result=res,A=A.cpu(),O=O.cpu(),seconds=time.perf_counter()-t0,peak_GPU_allocated=torch.cuda.max_memory_allocated()))
    print('CONVEX_DONE',rid,{g:r['info']['final'] for g,r in res.items()},flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('what',choices=['prepare','run','convex']);ap.add_argument('rid',nargs='?');a=ap.parse_args()
    environment()
    {'prepare':lambda:prepare(),'run':lambda:run(a.rid),'convex':lambda:convex(a.rid)}[a.what]();sys.stdout.flush();os._exit(0)
