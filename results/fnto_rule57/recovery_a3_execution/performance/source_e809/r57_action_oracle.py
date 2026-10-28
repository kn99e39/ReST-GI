"""Unbiased incident-field importance oracle, unchanged WL53 first-exit walk.

q(s) is a mixture of prescribed incident fluxes, never target-dependent. Trace RGB
unit weights; apply x(s)/q(s) ONLY at exit accumulation, so RR/path decisions are
independent of the queried weights for fixed q. No full K, air continuation or
post-normalisation. Float64 first/second moment accumulators.
"""
import os,time
import torch
import r57_geometry as RG
TR=RG.TR

class Ignore:
    def index_add_(self,*args):pass
class Sink:
    def __init__(self,owner):self.a=owner
    def index_add_(self,dim,idx,we):
        a=self.a;row=idx//a.nsrc;col=idx%a.nsrc
        for k in range(len(a.fields)):
            weighted=we*a.ratio[k,col].T
            a.values[k].index_add_(1,row,weighted)
            a.second[k].index_add_(1,row,weighted.square())
class Acc:
    def __init__(self,dom,fields,q):
        self.dom=dom;self.Dout=64;self.Din=16;self.Nout=8;self.nsrc=dom.P*16;self.psrc=1;self.KC=None
        self.fields=fields;self.ratio=fields.double().reshape(len(fields),-1,3)/q.clamp(min=1e-300)[None,:,None]
        self.values=torch.zeros(len(fields),3,dom.P*64,device='cuda',dtype=torch.float64);self.second=torch.zeros_like(self.values)
        self.T=Sink(self);self.S2=Ignore()
        import numpy as np
        self.stats=dict(photons=0,exits=0,leaks=0,wrong_side=0,truncated=0,nee_invalid=0,steps=0,collisions=0,flux_in=np.zeros(3),flux_out=np.zeros(3),flux_leak=np.zeros(3))

def action(dom,fields,N,seed,proposal=None):
    if fields.ndim==3:fields=fields[None]
    fields=fields.cuda();assert fields.shape[1:]==(dom.P,16,3) and bool((fields>=0).all())
    q=fields.double().mean((0,3)).reshape(-1) if proposal is None else proposal.double().reshape(-1).cuda()
    if float(q.sum())==0:
        return dict(response=torch.zeros(len(fields),dom.P,64,3),second=torch.zeros(len(fields),dom.P,64,3),N=N,seed=seed,zero=True,seconds=0.)
    q=q/q.sum();assert bool(((fields.double().reshape(len(fields),-1,3).sum((0,2))>0)<= (q>0)).all())
    a=Acc(dom,fields,q);gen=torch.Generator(device='cuda').manual_seed(seed);t0=time.perf_counter();chunk=int(os.environ.get('WL57_ACTION_CHUNK','2000000'))
    for start in range(0,N,chunk):
        n=min(chunk,N-start);cells=torch.multinomial(q,n,replacement=True,generator=gen)
        x,d,normal,w,src=TR.cell_source(dom,cells,1,16,gen);w.fill_(1./N)
        TR.walk(a,x,d,normal,w,src,gen,continue_air=False)
    torch.cuda.synchronize()
    y=a.values.permute(0,2,1).reshape(len(fields),dom.P,64,3).float().cpu()
    s2=a.second.permute(0,2,1).reshape(len(fields),dom.P,64,3).cpu()
    energy=y.double().sum((1,2));v=(s2.sum((1,2))-energy.square()/N).clamp(min=0)*N/(N-1)
    expected=fields.double().sum((1,2)).cpu();stats={k:v.tolist() if hasattr(v,'tolist') else v for k,v in a.stats.items()}
    passed=bool(y.min()>=0 and ((energy-expected)<=5*v.sqrt()+1e-6).all() and (stats['leaks']+stats['wrong_side'])/N<=1e-5 and stats['truncated']==0)
    return dict(response=y,second=s2,N=N,seed=seed,seconds=time.perf_counter()-t0,chunk=chunk,stats=stats,
        expected_input=expected.tolist(),output_energy=energy.tolist(),energy_SE=v.sqrt().tolist(),physical_passed=passed,
        proposal_sum=float(q.sum()),importance_ratio_max=float(a.ratio.max()),peak_GPU_bytes=torch.cuda.max_memory_allocated())
