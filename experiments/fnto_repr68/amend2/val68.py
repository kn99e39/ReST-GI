"""Amendment-2 validation (non-evidence): old einsum path (Sc=None) vs channel-major path on the real fitting targets.
(1) objective/gradient agreement vs a float64 reference, (2) s/step per run kind, (3) 1000-step Adam trajectories."""
import sys,time,json,math,torch
sys.path.insert(0,'.')
from common68 import CFG,environment
import fit68 as F,family68 as FM
RG=environment();torch.backends.cuda.matmul.allow_tf32=False;torch.set_num_threads(8)
pb=F.Problem(RG);OLD={'flag':False}
_sq=FM.sq_error
def sq_switch(T,d,A,O,S,den,Sc=None):return _sq(T,d,A,O,S,den,None if OLD['flag'] else Sc)
FM.sq_error=sq_switch
def setup(kind,group,init='P_5901'):
    torch.manual_seed(0)
    if kind=='net':
        m=F.model_from(init);m.train();params=F.group_params(m,group);cols=lambda g:FM.network_columns(m,pb.gs[g],pb.ports)
    else:
        with torch.no_grad():
            src=F.model_from(init);Ts={g:FM.network_columns(src,pb.gs[g],pb.ports)[0] for g in F.FIT_GEOMS}
            ml,sl=zip(*[FM.FCoef.logits_from(Ts[g]) for g in F.FIT_GEOMS])
        m=FM.FCoef(src.in_logits.detach(),src.out_logits.detach(),ml,sl).cuda();params=list(m.parameters());idx={g:k for k,g in enumerate(F.FIT_GEOMS)}
        cols=lambda g:m.columns(idx[g])
    return m,params,cols
out={}
# (1) agreement: same parameters, J and gradient, old vs new vs float64 (direct objective)
for kind,group in [('net','all'),('coef',None)]:
    m,params,cols=setup(kind,group);res={}
    for name,old in (('old',True),('new',False)):
        OLD['flag']=old
        for p in params:p.grad=None
        J=pb.loss(cols);J.backward();res[name]=(float(J),torch.cat([p.grad.flatten() for p in params]).double())
    OLD['flag']=False
    for p in params:p.grad=None
    J64=0
    for g in F.FIT_GEOMS:
        T,A,O=cols(g);tg=pb.tg[g]
        tg64=FM.Target(tg.S['native'].double(),tg.d.double(),tg.parent,tg.bm)
        J64=J64+FM.objective_direct(T.double(),A.double(),O.double(),tg64)/2;del tg64
    J64.backward();g64=torch.cat([p.grad.flatten() for p in params]).double();J64=float(J64)
    rel=lambda x,y:float((x-y).norm()/y.norm())
    out[f'{kind}_{group}']=dict(J_old=res['old'][0],J_new=res['new'][0],J_f64=J64,
        J_rel_old=abs(res['old'][0]-J64)/J64,J_rel_new=abs(res['new'][0]-J64)/J64,
        grad_rel_old_vs_f64=rel(res['old'][1],g64),grad_rel_new_vs_f64=rel(res['new'][1],g64),grad_rel_new_vs_old=rel(res['new'][1],res['old'][1]))
    print(f'{kind}_{group}',out[f'{kind}_{group}'],flush=True);torch.cuda.empty_cache()
# (2) step time
def step(opt,params,cols):
    opt.zero_grad(set_to_none=True);J=pb.loss(cols)
    if not torch.isfinite(J):raise RuntimeError
    J.backward();float(torch.nn.utils.clip_grad_norm_(params,1.));opt.step();return J
for kind,group in [('net','net'),('net','AO'),('net','all'),('coef',None)]:
    for name,old in (('old',True),('new',False)):
        OLD['flag']=old;m,params,cols=setup(kind,group);opt=torch.optim.Adam(params,lr=1e-6)
        for _ in range(5):step(opt,params,cols)
        torch.cuda.synchronize();t=time.perf_counter()
        for _ in range(40):step(opt,params,cols)
        torch.cuda.synchronize();out[f'time_{kind}_{group}_{name}']=(time.perf_counter()-t)/40
        print(kind,group,name,'s/step',round(out[f'time_{kind}_{group}_{name}'],4),flush=True)
# (3) trajectories: 1000 steps of the protocol optimizer (net all lr schedule compressed to 1000 steps)
for kind,group in [('net','all'),('coef',None)]:
    oc=CFG['optimizer'][kind];traj={}
    for name,old in (('old',True),('new',False)):
        OLD['flag']=old;m,params,cols=setup(kind,group);opt=torch.optim.Adam(params,lr=oc['lr']);h=[]
        for s in range(1000):
            for q in opt.param_groups:q['lr']=F.cosine(s,1000,oc['lr'],oc['lr_min'])
            J=step(opt,params,cols)
            if s%100==0 or s==999:h.append(float(J))
        traj[name]=h
    out[f'traj_{kind}_{group}']=traj;print(kind,group,'traj',traj,flush=True)
OLD['flag']=False
json.dump(out,open('/tmp/nd_val68.json','w'),indent=1)
