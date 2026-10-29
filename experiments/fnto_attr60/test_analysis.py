import unittest
import torch
from analyze_stored import angular,agg,project,rank_floor,relative

class AnalysisTests(unittest.TestCase):
    def test_flux_aggregation(self):
        y=torch.arange(2*4*64*3,dtype=torch.float64).reshape(2,4,64,3)
        z=agg(y,torch.tensor([0,0,1,1]))
        torch.testing.assert_close(z.sum((1,2)),y.sum((1,2)))
        self.assertEqual(tuple(z.shape),(2,2,16,3))
        self.assertEqual(float(angular(torch.ones(1,64,3))[0,0,0]),4.)
    def test_span_bound_and_rank(self):
        basis=torch.eye(4,dtype=torch.float64)[:,:2]
        y=torch.tensor([[1.,2.,3.,4.]],dtype=torch.float64)
        p,r=project(y,basis)
        self.assertEqual(r,2)
        torch.testing.assert_close(p,torch.tensor([[1.,2.,0.,0.]],dtype=torch.float64))
        self.assertLessEqual(float((p-y).norm()),float((torch.tensor([[3.,5.,0.,0.]])-y).norm()))
    def test_rank_floor(self):
        y=torch.diag(torch.tensor([4.,3.,2.,1.],dtype=torch.float64))
        floor,_=rank_floor(y,2)
        self.assertAlmostEqual(floor,5**.5)
    def test_effective_component_error_identity(self):
        t=torch.tensor([2.,5.]);p=torch.tensor([3.,4.]);b=torch.tensor([.5,1.])
        torch.testing.assert_close((p-b)-(t-b),p-t)

if __name__=='__main__':unittest.main()
