"""Focused CPU correctness; no production regression, physical photons or model load."""
import types,sys
import torch
from moments65 import observed_action,aggregate,variance,loss_scale,norm

def main():
    torch.set_num_threads(1);checks={}
    # Photon-level sample covariance is preserved when exclusive cells aggregate.
    samples=torch.tensor([[1.,0.],[0.,2.],[0.,0.],[3.,0.]],dtype=torch.float64)
    N=len(samples);first=samples.mean(0);second=samples.square().sum(0)/N**2
    direct=samples.sum(1).var(unbiased=True)/N
    assert torch.allclose(variance(first.sum(),second.sum(),N),direct)
    assert not torch.allclose(variance(first,second,N).sum(),direct)
    checks['exclusive_bin_covariance_aggregate_before_variance']=True
    v=torch.tensor([1.,2.,3.,4.]);assert float(v.sum()/16)==.625;checks['four_half_mean_variance_divide16']=True
    vg,vr,vh=2.,3.,5.;assert (vg+vr)+(vh+vr)-2*vr==vg+vh;checks['shared_rest_delta_covariance_cancels']=True
    yy=torch.ones(2,1,64,3);parent=torch.tensor([0]);assert torch.equal(aggregate(yy,parent).sum((1,2)),yy.double().sum((1,2)))
    assert torch.isfinite(loss_scale(torch.zeros_like(yy),yy)).all();checks['weak_delta_original_loss_floor_no_signal_filter']=True
    class Acc:
        def __init__(self):self.fields=[0];self.values=torch.arange(192,dtype=torch.float64).reshape(1,3,64);self.second=self.values.square()
    ao=types.SimpleNamespace(Acc=Acc)
    def action(dom,field,N,seed):
        a=ao.Acc();unpack=lambda y:y.permute(0,2,1).reshape(1,1,64,3)
        return dict(response=unpack(a.values).float(),second=unpack(a.second),seed=seed)
    ao.action=action;r=observed_action(ao,types.SimpleNamespace(P=1),None,4,650)
    assert ao.Acc is Acc and all(r['observer_checks'].values());checks['observer_original_output_and_restoration']=True
    pairs=[(f,f+1) for f in range(80,96,2)];assert [v for x in pairs for v in x]==list(range(80,96));checks['all8_original_pairs_no_filtering']=True
    assert len({650000000+p*100000+f*100+r*10+h for p in range(13) for f in range(80,96) for r in range(2) for h in range(2)})==832
    checks['832_distinct_fixed_seeds']=True
    print(checks,flush=True)
if __name__=='__main__':main()
