"""Per-term timing of the WL68 objective (all group) to locate the GPU time. Non-evidence."""
import sys,time,torch
sys.path.insert(0,'.')
from common68 import environment
import fit68 as F,family68 as FM
RG=environment();torch.backends.cuda.matmul.allow_tf32=False
pb=F.Problem(RG);m=F.model_from('P_5901');m.train();params=F.group_params(m,'all')
g='rest';tg=pb.tg[g]
def tm(fn,n=20):
    fn();torch.cuda.synchronize();t=time.perf_counter()
    for _ in range(n):fn()
    torch.cuda.synchronize();return round((time.perf_counter()-t)/n*1e3,2)
def cols():return FM.network_columns(m,pb.gs[g],pb.ports)
def fb(f):
    def run():
        for p in params:p.grad=None
        T,A,O=cols();f(T,A,O).sum().backward()
    return run
print('columns fwd+bwd ms',tm(fb(lambda T,A,O:T.sum()+A.sum()+O.sum())))
print('sq native ms',tm(fb(lambda T,A,O:FM.sq_error(T,tg.d,A,O,tg.S['native'],tg.den['native']))))
O16f=lambda O:torch.zeros(O.shape[0],16,3,device=O.device,dtype=O.dtype).index_add(1,tg.bm,O)
print('sq fine16 ms',tm(fb(lambda T,A,O:FM.sq_error(T,tg.d,A,O16f(O),tg.S['fine16'],tg.den['fine16']))))
print('coarse ms',tm(fb(lambda T,A,O:(FM.predict(T,tg.d,A,O,'coarse',tg.parent,tg.bm)-tg.S['coarse']).square().sum())))
print('full objective ms',tm(fb(lambda T,A,O:FM.objective(T,A,O,tg))))
# pieces of sq_error native
S=tg.S['native']
def hs_pieces(T,A,O):
    X=torch.einsum('najoc,roc->najrc',S,O);return torch.einsum('na,rac,njrc,najrc->nc',tg.d,A,T,X)
print('hs native ms',tm(fb(hs_pieces)))
print('SO only ms',tm(fb(lambda T,A,O:torch.einsum('najoc,roc->najrc',S,O)+0*T.sum())))
from torch.profiler import profile,ProfilerActivity
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    fb(lambda T,A,O:torch.einsum('najoc,roc->najrc',S,O)+0*T.sum())();torch.cuda.synchronize()
print(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=8))
print('opt_einsum',torch.backends.opt_einsum.enabled,torch.backends.opt_einsum.is_available())
