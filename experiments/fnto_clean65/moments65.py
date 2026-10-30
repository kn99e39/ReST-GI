"""Observe existing AO accumulator after unchanged action; no path/RNG/sink patch."""
def observed_action(AO,dom,field,N,seed):
    import torch
    original=AO.Acc;created=[]
    class Observed(original):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs);created.append(self)
    try:
        AO.Acc=Observed;result=AO.action(dom,field,N,seed)
    finally:AO.Acc=original
    assert len(created)==1;a=created[0]
    unpack=lambda x:x.permute(0,2,1).reshape(len(a.fields),dom.P,64,3).cpu()
    first=unpack(a.values);second=unpack(a.second)
    result['observer_checks']=dict(original_response_exact=bool(torch.equal(first.float(),result['response'])),original_second_exact=bool(torch.equal(second,result['second'])),no_source_or_walker_patch=True)
    result['first64']=first
    return result

def aggregate(y,parent,coarse=True):
    import torch
    y=y.double();mapping=(torch.arange(64,device=y.device)%8)//2+4*((torch.arange(64,device=y.device)//8)//2)
    z=torch.zeros(*y.shape[:-2],16,3,dtype=y.dtype,device=y.device).index_add_(-2,mapping,y)
    if not coarse:return z
    return torch.zeros(len(y),int(parent.max())+1,16,3,dtype=y.dtype,device=y.device).index_add_(1,parent,z)
def variance(first,second,N):return (second-first.square()/N)*(N/(N-1))
def norm(x):return x.flatten(-3,-2).norm(dim=-2)
def ratio(x,y):return norm(x)/norm(y).clamp(min=1e-300)
def loss_scale(delta,rest):return (norm(delta).square()+1e-4*norm(rest).square()).sqrt()
