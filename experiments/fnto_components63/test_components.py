import copy,itertools,unittest
from pathlib import Path
import torch
from common63 import reference
from tag_reference import instrument_walker,tagged_accumulator
from metrics63 import variance,covariance_BS,decompose

class ComponentTests(unittest.TestCase):
    def test_moment_covariance_closure(self):
        B=torch.tensor(.25,dtype=torch.float64);S=B.clone();B2=torch.tensor(.0625,dtype=torch.float64);S2=torch.tensor(.03125,dtype=torch.float64)
        cov=covariance_BS(B,S,4);self.assertLess(float(cov),0)
        torch.testing.assert_close(variance(B+S,B2+S2,4),variance(B,B2,4)+variance(S,S2,4)+2*cov,atol=1e-15,rtol=1e-14)
    def test_four_half_covariance_scaling(self):
        cov=torch.tensor(-.02,dtype=torch.float64)
        self.assertAlmostEqual(float(torch.stack([cov]*4).sum()/16),-.005)
    def test_cancellation_not_magnitude_attribution(self):
        b=torch.tensor([[2.,2.,2.]],dtype=torch.float64);s=torch.tensor([[.5,.5,.5]],dtype=torch.float64);br=torch.ones(1,3,dtype=torch.float64);sr=br.clone()
        row=decompose(b,s,br,sr)[0];self.assertAlmostEqual(row['error_cosine'],-1);self.assertLess(row['cross_over_total_signal'],0);self.assertEqual(row['classification'],'C_INTERACTING_OR_MIXED');self.assertLess(row['error_energy_identity_abs'],1e-14)
    def test_deterministic_original_tagged_paths_and_rng(self):
        RG,AO,FW=reference(Path.home()/'rest_persist_v1');tagwalk,proof=instrument_walker(FW)
        self.assertTrue(proof['strip_added_assignment_restores_original_AST']);self.assertEqual(proof['added_RNG_calls'],0)
        dev='cuda'
        class Domain:
            P=1;eps=0.;sigma_t=1.;g=0.
            def __init__(self):
                self.albedo=torch.tensor([.999,.99,.9],device=dev);self.n=torch.tensor([[0.,0.,1.]],device=dev);self.t=torch.tensor([[1.,0.,0.]],device=dev);self.b=torch.tensor([[0.,1.,0.]],device=dev);self.tri_port=torch.tensor([0],device=dev);self.trace=[]
            def isect(self,x,d):
                t=torch.zeros(len(x),device=dev)
                if not self.trace:t[1:]=1e6
                prim=torch.zeros(len(x),dtype=torch.long,device=dev)
                self.trace.append(tuple(v.clone() for v in (x,d,t,prim)));return t,prim
        fields=torch.ones(1,1,16,3,device=dev)/16;q=torch.ones(16,device=dev)/16
        x=torch.zeros(3,3,device=dev);d=torch.zeros_like(x);d[:,2]=1;n=-d;w=torch.ones_like(x)/3;src=torch.tensor([0,1,2],device=dev)
        created=[];Tagged=tagged_accumulator(AO,created);d0=Domain();d1=Domain();plain=AO.Acc(d0,fields,q);tag=Tagged(d1,fields,q)
        oldphase=RG.TR.phase_sample
        def scripted_phase(dd,gg,gen):
            torch.rand(len(dd),device=dd.device,generator=gen);torch.rand(len(dd),device=dd.device,generator=gen);return dd
        try:
            RG.TR.phase_sample=scripted_phase
            g0=torch.Generator(device=dev).manual_seed(630123);g1=torch.Generator(device=dev).manual_seed(630123)
            FW.walk(plain,x,d,n,w,src,g0);tagwalk(tag,x,d,n,w,src,g1);FW.flush(plain);FW.flush(tag)
        finally:RG.TR.phase_sample=oldphase
        self.assertEqual(len(d0.trace),len(d1.trace))
        for aa,bb in zip(d0.trace,d1.trace):
            for av,bv in zip(aa,bb):self.assertTrue(torch.equal(av,bv))
        self.assertTrue(torch.equal(g0.get_state(),g1.get_state()))
        torch.testing.assert_close(plain.values,tag.values,atol=1e-14,rtol=1e-12);torch.testing.assert_close(plain.second,tag.second,atol=1e-14,rtol=1e-12)
        torch.testing.assert_close(tag.tag_first['B']+tag.tag_first['Scat'],plain.values,atol=1e-14,rtol=1e-12)
        self.assertEqual(tag.event_counts,dict(B=1,Scat=2));self.assertEqual(plain.stats['collisions'],tag.stats['collisions']);self.assertEqual(plain.stats['exits'],3)
        torch.testing.assert_close(tag.tag_first['B'].sum((0,2)),torch.ones(3,device=dev,dtype=torch.float64)/3,atol=1e-8,rtol=1e-6)
        torch.testing.assert_close(tag.tag_first['Scat'].sum((0,2)),2*d1.albedo.double()/3,atol=1e-8,rtol=1e-6)

if __name__=='__main__':unittest.main()
