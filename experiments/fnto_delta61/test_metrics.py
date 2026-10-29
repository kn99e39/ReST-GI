import unittest
import torch
from metrics import aggregate,half_variance,mean_variance,paired,relative,channel_norm

class MetricsTests(unittest.TestCase):
    def test_noise_is_not_prediction_error(self):
        noise=torch.tensor([[3.,4.,0.]],dtype=torch.float64);signal=torch.tensor([[30.,40.,10.]],dtype=torch.float64)
        torch.testing.assert_close(channel_norm(noise)/channel_norm(signal),torch.tensor([.1,.1,0.],dtype=torch.float64))
    def test_half_variance_known_bernoulli(self):
        # 2 unit-weight Bernoulli samples: [0,1]; contributions [0,.5], mean .5.
        self.assertAlmostEqual(float(half_variance(torch.tensor(.5),torch.tensor(.25),2)),.25)
    def test_four_half_scaling(self):
        vals=[torch.tensor([2.,3.,4.])]*4;var=[torch.ones(3)]*4
        mean,v=mean_variance(vals,var);torch.testing.assert_close(v,torch.ones(3)*.25)
        # Independent rest/current variance adds, not subtracts; shared rest covariance is retained separately.
        torch.testing.assert_close(v+v,torch.ones(3)*.5)
    def test_flux_aggregation(self):
        y=torch.arange(2*4*64*3,dtype=torch.float64).reshape(2,4,64,3)
        z=aggregate(y,torch.tensor([0,0,1,1]));torch.testing.assert_close(z.sum((1,2)),y.sum((1,2)))
    def test_paired_same_rest(self):
        values={'rest':torch.ones(2,3),'tr11':torch.ones(2,3)*2,'va1':torch.ones(2,3)*4}
        torch.testing.assert_close(paired(values,'va1')-paired(values,'tr11'),values['va1']-values['tr11'])
        with self.assertRaises(ValueError):paired({'rest':torch.ones(2,3),'bad':torch.ones(3,3)},'bad')
    def test_error_denominator(self):
        self.assertEqual(relative(torch.tensor([[2.,4.,6.]]),torch.tensor([[1.,2.,3.]])),[1.,1.,1.])
if __name__=='__main__':unittest.main()
