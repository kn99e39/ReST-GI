"""WL73 training (diagnostic copies on S16-R; WL71/WL72 objectives, optimiser, schedule and model constructors unchanged).
D1 = WL59 W+UV 'net' group with fixed A/O and objective J (lambda_sym 0); A = the same + exact-symmetry consistency (lambda 1);
B = RSK from its WL72 seed, objective J. The fit stage reads only the S16-R fit targets (fitting replicas 0-3), ports, the training
geometries, the WL72 pt0 symmetry-image geometries (A only), the WL59 checkpoints and the WL68 net_all_5902 A/O (test73 spies it).
    python fit73.py prepare | run <run_id> | replay <run_id>"""
import sys,os,json,time,math
from pathlib import Path
import torch
from common73 import CFG,ROOT,R71,R72,TRAIN,save_new,write_new,sha,tensor_sha,environment,dev
import family68 as FM
import family71 as FW
import sym72 as SY

def load(p,mmap=False):return torch.load(p,map_location='cpu',weights_only=False,mmap=mmap)

def populations():
    """slot -> port for S16 (WL71) and S16-R (frozen; kept slots in WL71 order, then the replacements)."""
    p71={int(k):v['port'] for k,v in json.loads((R71/'ports.json').read_text())['slots'].items()}
    fz=CFG['frozen'];kept=[s for s in range(16) if s not in fz['removed_slots']]
    S16R=[(s,p71[s]) for s in kept]+[(int(s),int(p)) for s,p in sorted(fz['new_ports'].items(),key=lambda t:int(t[0]))]
    return dict(S16=[(s,p71[s]) for s in range(16)],S16R=S16R)

def prepare():
    """fit_targets73.pt: S16-R means of fitting replicas 0-3; retained rows copied bitwise from fit_targets71, new rows from the new bundles."""
    out=ROOT/'fit_targets73.pt';pops=populations();f71=load(R71/'fit_targets71.pt',mmap=True);src={};data={}
    for g in TRAIN:
        rows=[]
        for s,p in pops['S16R']:
            if s<16:rows.append(f71['S']['r0123'][g][s].clone());src[f'{g}_s{s}']=dict(file=str(R71/'fit_targets71.pt'),row=s,sha256_row=tensor_sha(rows[-1]))
            else:
                f=ROOT/'refs/bundles'/f'{g}_s{s}.pt';b=load(f);assert b['port']==p and b['K']==8;rows.append(b['S'][[0,1,2,3]].double().mean(0).float())
                src[f'{g}_s{s}']=dict(file=str(f),sha256=sha(f));del b
        data[g]=torch.stack(rows)
    save_new(out,dict(S={'r0123':data},slots=[s for s,_ in pops['S16R']],ports=[p for _,p in pops['S16R']],sources=src,
        layout='S[\'r0123\'][geometry]: (16 slots in S16-R order, 16 bins, P, 64, RGB) collided means of fitting replicas 0-3 only'))
    print('PREPARED',sha(out),flush=True)

class Problem73:
    """fit71.Problem with an explicit population (identical construction: archived training geometries, d = 1 - beta, FM.Target)."""
    def __init__(self,RG,pop):
        import fit71
        self.gs=fit71.geometries(RG);P=populations()[pop];self.slots=[s for s,_ in P];self.ports=[p for _,p in P];self.bm=FM.bin_map('cuda')
        if pop=='S16':S=dict(load(R71/'fit_targets71.pt',mmap=True)['S']['r0123'])
        else:t=load(ROOT/'fit_targets73.pt');assert t['ports']==self.ports;S=dict(t['S']['r0123'])
        self.F={g:SY.integrated(S[g]) for g in TRAIN}   # angle-integrated labels for A's symmetry term (same tensor, same row order)
        self.tg={}
        for g in TRAIN:
            d=(1-self.gs[g]['beta'][self.ports]).float();self.tg[g]=FM.Target(S[g].cuda(),d,self.gs[g]['parent'],self.bm);S[g]=None
            del self.tg[g].S['native']
        torch.cuda.empty_cache()

def run(rid,replay=False,tag='replay_'):
    import fit71,fit72
    spec=CFG['replays' if replay else 'runs'][rid];RG=environment();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_num_threads(4);out=ROOT/('stage1' if replay else 'runs')/f'{tag if replay else ""}{rid}.pt'
    if out.exists():raise FileExistsError('completed run is immutable')
    orphans=sorted((ROOT/'runs').glob(f'{rid}@*.pt')) if not replay else []
    if orphans:
        q=ROOT/'runs/quarantine'/str(time.time_ns());q.mkdir(parents=True)
        for p in orphans:os.replace(p,q/p.name)   # a crashed attempt is preserved, the run restarts from its initialisation
    oc=CFG['optimizer'];full=oc['steps'];steps=CFG['stage1']['replay_steps'] if replay else full
    if 'R73_DEV_STEPS' in os.environ:assert dev(),'dev override never in the evidence root';steps=min(steps,int(os.environ['R73_DEV_STEPS']))
    pb=Problem73(RG,spec['population']);gs=dict(pb.gs)
    m,params,cols=fit72.model_for(dict(candidate='A' if spec['family'] in ('D1','A') else 'B',init=spec.get('init'),seed=spec.get('seed')),RG)
    lam=spec['lambda_sym'];sym=fit72.sym_data(RG,gs,pb.ports,pb.F) if lam else [];del pb.F
    def loss():
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
    opt=torch.optim.Adam(params,lr=oc['lr']);hist=[];t0=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():tot0,J0,L0=(float(v) for v in loss())
    ck={c*steps//full:c for c in oc['checkpoints']} if not replay else {}   # nominal checkpoints (proportional in dev; identity in evidence)
    for step in range(steps):
        for q in opt.param_groups:q['lr']=fit71.cosine(step,full if replay else steps,oc['lr'],oc['lr_min'])   # replay: the historical 20k schedule
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
    save_new(out,dict(model={k:v.detach().cpu() for k,v in m.state_dict().items()},spec=spec,history=hist,loss_initial=tot0,J_initial=J0,L_sym_initial=L0,
        loss_final=totf,J_final=Jf,L_sym_final=Lf,last_decile_relative_change=rel,converged=rel<.01,seconds=sec,peak_GPU_allocated=torch.cuda.max_memory_allocated(),
        trainable=sum(p.numel() for p in params),sym_examples=sum(len(s[1]) for s in sym),ports=pb.ports,slots=pb.slots,concurrent=os.environ.get('R73_LANE')))
    print('RUN_DONE',rid,totf,Jf,Lf,round(sec),flush=True)

def compare_replay(rid):
    """Stage 1 adapter identity: initial objective and the first replay_steps of the archived history (WL71 D1 / WL72 A, B) through this path."""
    spec=CFG['replays'][rid];r=load(ROOT/'stage1'/f'replay_{rid}.pt');arc=load(Path(spec['archived']).expanduser());tol=CFG['stage1']
    a={h['step']:h for h in arc['history']};b={h['step']:h for h in r['history']};common=[s for s in sorted(b) if s in a]
    traj=max(abs(b[s]['J']-a[s]['J'])/abs(a[s]['J']) for s in common)
    init=dict(J=abs(r['J_initial']-arc['J_initial'])/abs(arc['J_initial']))
    if 'loss_initial' in arc:init['loss']=abs(r['loss_initial']-arc['loss_initial'])/abs(arc['loss_initial'])
    ok=max(init.values())<=tol['initial_rel_tol'] and traj<=tol['trajectory_rel_tol']
    return dict(run=rid,archived=spec['archived'],steps_compared=len(common),initial_rel=init,trajectory_max_rel=traj,passed=ok)

def compare_replay2(rid):
    """Amendment 1: the archived-vs-replay trajectory difference must lie within the run-to-run envelope of the same code (a second
    independent replay): max_s |J_r1 - J_arch| / J_arch <= factor * max_s |J_r1 - J_r2| / J_r2, plus the unchanged initial-objective identity."""
    spec=CFG['replays'][rid];am=CFG['amendments'][0];r1=load(ROOT/'stage1'/f'replay_{rid}.pt');r2=load(ROOT/'stage1'/f'replay2_{rid}.pt');arc=load(Path(spec['archived']).expanduser())
    a={h['step']:h['J'] for h in arc['history']};b={h['step']:h['J'] for h in r1['history']};c={h['step']:h['J'] for h in r2['history']};common=[s for s in sorted(b) if s in a and s in c]
    d_arch=max(abs(b[s]-a[s])/abs(a[s]) for s in common);d_self=max(abs(b[s]-c[s])/abs(c[s]) for s in common);d_arch2=max(abs(c[s]-a[s])/abs(a[s]) for s in common)
    first=next((s for s in common if abs(b[s]-a[s])/abs(a[s])>1e-3),None);bias=sum((b[s]-a[s])/a[s] for s in common)/len(common)
    init=dict(J=abs(r1['J_initial']-arc['J_initial'])/abs(arc['J_initial']),J_replay2=abs(r2['J_initial']-arc['J_initial'])/abs(arc['J_initial']))
    ok=max(init.values())<=CFG['stage1']['initial_rel_tol'] and d_arch<=am['envelope_factor']*d_self and d_arch2<=am['envelope_factor']*d_self
    return dict(run=rid,steps_compared=len(common),initial_rel=init,replay_vs_archive_max_rel=d_arch,replay2_vs_archive_max_rel=d_arch2,replay_vs_replay2_max_rel=d_self,
        first_step_beyond_1e_3=first,mean_signed_rel_replay_vs_archive=bias,passed=ok)

if __name__=='__main__':
    environment();{'prepare':lambda:prepare(),'run':lambda:run(sys.argv[2]),'replay':lambda:run(sys.argv[2],True),'replay2':lambda:run(sys.argv[2],True,'replay2_')}[sys.argv[1]]();sys.stdout.flush();os._exit(0)
