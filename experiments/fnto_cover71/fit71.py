"""WL71 fits (diagnostic objective on experimental copies; WL68 objective and optimizer). The fit stage reads ONLY
fit_targets71.pt (means of fitting replicas), training geometries, WL59 checkpoints and the WL68 net_all_5902 run (angular
bases / teacher TA); never a final reference or an evaluation replica (guarded loader, test71).
Run: fit71.py prepare | fit71.py run <run_id>."""
import sys,os,json,time,math
from pathlib import Path
import torch
from common71 import CFG,ROOT,INP68,R68,TRAIN_GEOMS,sha,save_new,environment
import family68 as FM
import family71 as FW

ALLOWED={'fit_targets71.pt':ROOT/'fit_targets71.pt','ports.json':ROOT/'ports.json','geometry_rest':INP68/'wl59/geometry_rest.pt','geometry_pt0':INP68/'wl59/geometry_pt0.pt',
         'P_5901':INP68/'wl59/P_5901.pt','P_5902':INP68/'wl59/P_5902.pt','net_all_5902':R68/'runs/net_all_5902.pt'}
def guarded(name):
    p=ALLOWED[name];return torch.load(p,map_location='cpu',weights_only=False) if p.suffix=='.pt' else json.loads(p.read_text())

def slot_ports(pop):return [int(guarded('ports.json')['slots'][str(s)]['port']) for s in CFG['populations'][pop]['slots']]

def prepare():
    """Means of the declared fitting replicas for every training slot: 'r0123' (replicas 0-3) and 'r0' (replica 0); slots 0-15."""
    out=ROOT/'fit_targets71.pt';data={'r0123':{},'r0':{}};src={}
    def file(g,s):
        if s<4:return INP68/'fit'/f'{g}_s{s}.pt'
        if s<8:return R68/'heldout/bundles'/f'{g}_s{s}.pt'
        return ROOT/'refs/bundles'/f'{g}_s{s}.pt'
    for g in TRAIN_GEOMS:
        rows={'r0123':[],'r0':[]}
        for s in CFG['populations']['S16']['slots']:
            f=file(g,s);S=torch.load(f,map_location='cpu',weights_only=False)['S'];src[f'{g}_s{s}']=dict(file=str(f),sha256=sha(f))
            rows['r0123'].append(S[[0,1,2,3]].double().mean(0).float());rows['r0'].append(S[0].clone());del S
        for k in rows:data[k][g]=torch.stack(rows[k])
    save_new(out,dict(S=data,slots=CFG['populations']['S16']['slots'],ports=slot_ports('S16'),sources=src,
        layout='S[replica_set][geometry]: (16 slots, 16 bins, P, 64, RGB) collided means of fitting replicas only'))
    print('PREPARED',sha(out),flush=True)

def geometries(RG):return {g:RG.device(guarded(f'geometry_{g}')) for g in TRAIN_GEOMS}

def angular():
    st=guarded('net_all_5902')['model'];return st['in_logits'].clone(),st['out_logits'].clone()

def model_from(spec):
    from runtime import Transport
    m=Transport().cuda();m.load_state_dict(guarded(spec)['model'])
    if spec.startswith('P_'):
        a,o=angular()
        with torch.no_grad():m.in_logits.copy_(a);m.out_logits.copy_(o)
    return m

def teacher(name):
    """TA: WL68 net_all_5902 complete; TB: P_5902 network with the WL68 net_all_5902 angular bases."""
    return (model_from('net_all_5902') if name=='TA' else model_from('P_5902')).eval()

def teacher_targets(RG,name,ports,G):
    t=teacher(name);out={}
    with torch.no_grad():
        for g in TRAIN_GEOMS:
            T,A,O=FW.network_columns(t,G[g],ports);out[g]=FM.predict(T,(1-G[g]['beta'][ports]).float(),A,O,'native')
    return out

class Problem:
    def __init__(self,RG,spec):
        self.gs=geometries(RG);pop=CFG['populations'][spec['pop']];idx=[CFG['populations']['S16']['slots'].index(s) for s in pop['slots']]
        self.ports=slot_ports(spec['pop']);self.bm=FM.bin_map('cuda')
        if spec['target']=='phys':
            rs='r0123' if spec['replicas']==[0,1,2,3] else 'r0' if spec['replicas']==[0] else None;assert rs
            S={g:v[idx] for g,v in guarded('fit_targets71.pt')['S'][rs].items()}
        else:S=teacher_targets(RG,spec['target'],self.ports,self.gs)
        self.tg={}
        for g in TRAIN_GEOMS:
            d=(1-self.gs[g]['beta'][self.ports]).float();self.tg[g]=FM.Target(S[g].cuda(),d,self.gs[g]['parent'],self.bm);S[g]=None
            del self.tg[g].S['native']  # only its shape is used (sq_error reads Sc); frees memory, values unchanged
        torch.cuda.empty_cache()
    def loss(self,m,parts=False):
        vals=[]
        for g in TRAIN_GEOMS:
            T,A,O=FW.network_columns(m,self.gs[g],self.ports);tg=self.tg[g]
            vals.append(objective(T,A,O,tg,parts))
        if not parts:return sum(vals)/len(vals)
        return sum(v[0] for v in vals)/len(vals),{g:{k:x.cpu().tolist() for k,x in v[1].items()} for g,v in zip(TRAIN_GEOMS,vals)}

class _Shape:
    def __init__(self,shape):self.shape=shape
def objective(T,A,O,tg,parts):
    """family68.objective with the native target held only in channel-major form (identical arithmetic)."""
    S=tg.S;n=T.shape[0];P=T.shape[1]
    S['native']=_Shape((n,16,P,64,3))
    try:return FM.objective(T,A,O,tg,parts=parts)
    finally:del S['native']

def cosine(step,steps,lr,lr_min):return lr_min+.5*(lr-lr_min)*(1+math.cos(math.pi*step/steps))

def run(rid):
    spec=CFG['runs'][rid];RG=environment();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_num_threads(4);out=ROOT/'runs'/f'{rid}.pt'
    if out.exists():raise FileExistsError('completed run is immutable')
    import fit68
    oc=dict(lr=3e-3,lr_min=3e-5,steps=20000,clip=1.0,log_every=25);steps=oc['steps']
    if 'R71_DEV_STEPS' in os.environ:assert ROOT!=Path.home()/'rest_cover71','dev override never in the evidence root';steps=int(os.environ['R71_DEV_STEPS'])
    pb=Problem(RG,spec);torch.cuda.reset_peak_memory_stats();m=model_from(spec['init']);m.train();params=fit68.group_params(m,spec['group'])
    opt=torch.optim.Adam(params,lr=oc['lr']);J0,parts0=pb.loss(m,parts=True);J0=float(J0);hist=[];t0=time.perf_counter()
    for step in range(steps):
        for q in opt.param_groups:q['lr']=cosine(step,steps,oc['lr'],oc['lr_min'])
        opt.zero_grad(set_to_none=True);J=pb.loss(m);J.backward();gn=torch.nn.utils.clip_grad_norm_(params,oc['clip']);opt.step()
        if step%oc['log_every']==0 or step==steps-1:
            Jf=float(J)
            if not math.isfinite(Jf):raise RuntimeError('nonfinite objective: preserve failed run')
            hist.append(dict(step=step,J=Jf,grad_norm=float(gn),lr=opt.param_groups[0]['lr'],seconds=time.perf_counter()-t0))
            if step%(oc['log_every']*20)==0:print('FIT',rid,step,Jf,flush=True)
    torch.cuda.synchronize();sec=time.perf_counter()-t0
    with torch.no_grad():Jf,parts=pb.loss(m,parts=True)
    w=[h['J'] for h in hist];n=len(w);a=w[int(.8*n):int(.9*n)] or w[:1];b=w[int(.9*n):] or w[-1:]
    rel=abs(sum(b)/len(b)-sum(a)/len(a))/max(sum(b)/len(b),1e-30)
    save_new(out,dict(model={k:v.detach().cpu() for k,v in m.state_dict().items()},spec=spec,history=hist,J_initial=J0,parts_initial=parts0,J_final=float(Jf),
        parts_final=parts,last_decile_relative_change=rel,converged=rel<.01,seconds=sec,peak_GPU_allocated=torch.cuda.max_memory_allocated(),
        trainable=sum(p.numel() for p in params),concurrent=os.environ.get('R71_LANE')))
    print('RUN_DONE',rid,float(Jf),round(sec),flush=True)

if __name__=='__main__':
    environment();{'prepare':lambda:prepare(),'run':lambda:run(sys.argv[2])}[sys.argv[1]]();sys.stdout.flush();os._exit(0)
