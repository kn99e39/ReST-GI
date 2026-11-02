"""Profile one fit step of fit68 (net groups + coef) to attribute low GPU utilisation. Non-evidence, profiling root only."""
import sys,time,torch
sys.path.insert(0,'.')
from common68 import CFG,environment
import fit68 as F,family68 as FM
RG=environment();torch.backends.cuda.matmul.allow_tf32=False;torch.set_num_threads(8)
pb=F.Problem(RG)
def setup(kind,group,init):
    if kind=='net':
        m=F.model_from(init);m.train();params=F.group_params(m,group);cols=lambda g:FM.network_columns(m,pb.gs[g],pb.ports)
    else:
        with torch.no_grad():
            src=F.model_from(init);Ts={g:FM.network_columns(src,pb.gs[g],pb.ports)[0] for g in F.FIT_GEOMS}
            ml,sl=zip(*[FM.FCoef.logits_from(Ts[g]) for g in F.FIT_GEOMS])
        m=FM.FCoef(src.in_logits.detach(),src.out_logits.detach(),ml,sl).cuda();params=list(m.parameters());idx={g:k for k,g in enumerate(F.FIT_GEOMS)}
        cols=lambda g:m.columns(idx[g])
    return m,params,cols
def step(opt,params,cols,sync=True):
    opt.zero_grad(set_to_none=True);J=pb.loss(cols)
    if sync and not torch.isfinite(J):raise RuntimeError
    J.backward();gn=torch.nn.utils.clip_grad_norm_(params,1.)
    if sync:float(gn)
    opt.step();return J
for kind,group in [('net','net'),('net','all'),('net','AO'),('coef',None)]:
    m,params,cols=setup(kind,group,'P_5901');opt=torch.optim.Adam(params,lr=1e-6)
    for _ in range(10):step(opt,params,cols)
    torch.cuda.synchronize()
    for sync in (True,False):
        t=time.perf_counter()
        for _ in range(50):step(opt,params,cols,sync)
        torch.cuda.synchronize();wall=(time.perf_counter()-t)/50
        print(kind,group,'sync' if sync else 'nosync','s/step',round(wall,4),flush=True)
    from torch.profiler import profile,ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as prof:
        for _ in range(5):step(opt,params,cols)
        torch.cuda.synchronize()
    ev=prof.key_averages();gpu=sum(e.self_cuda_time_total for e in ev)/5e6
    nk=sum(e.count for e in ev if e.device_type==torch.autograd.DeviceType.CUDA)/5
    print(kind,group,'GPU kernel s/step',round(gpu,4),'kernels/step',nk,flush=True)
    print(ev.table(sort_by='self_cuda_time_total',row_limit=12),flush=True)
