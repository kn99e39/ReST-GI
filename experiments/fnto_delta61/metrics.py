"""Flux aggregation and independent-half uncertainty; no estimator/model changes."""
import torch

def aggregate(y,parent,coarse=True):
    y=y.double();bm=torch.arange(64,device=y.device);bm=(bm%8)//2+4*((bm//8)//2)
    z=torch.zeros(*y.shape[:-2],16,3,dtype=y.dtype,device=y.device).index_add_(-2,bm,y)
    if not coarse:return z
    return torch.zeros(len(y),int(parent.max())+1,16,3,dtype=y.dtype,device=y.device).index_add_(1,parent,z)

def half_variance(mean,second,N):
    # second=sum of squared photon contributions; mean=sum of contributions.
    return (second.double()-mean.double().square()/N).clamp(min=0)*(N/(N-1))

def mean_variance(halves,variances):
    count=len(halves)
    return torch.stack(halves).mean(0),torch.stack(variances).sum(0)/(count*count)

def channel_norm(x):
    return x.reshape(-1,3).norm(dim=0)

def relative(y,t):return (channel_norm(y-t)/channel_norm(t).clamp(min=1e-30)).tolist()

def per_field_relative(y,t):
    return ((y-t).flatten(1,2).norm(dim=1)/t.flatten(1,2).norm(dim=1).clamp(min=1e-30)).tolist()

def paired(values,pose):
    a,b=values[pose],values['rest']
    if a.shape!=b.shape:raise ValueError('Rest/current correspondence shape mismatch')
    return a-b

def alignment(y,t):
    rows=[]
    for c in range(3):
        a=t[...,c].flatten();b=y[...,c].flatten();ac=a-a.mean();bc=b-b.mean()
        rows.append(dict(correlation=float(ac@bc/(ac.norm()*bc.norm()).clamp(min=1e-30)),slope=float(ac@bc/(ac@ac).clamp(min=1e-30))))
    return rows
