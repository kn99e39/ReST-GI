"""Focused matching, geometry and accounting regressions."""
import unittest

import numpy as np
import torch

from common108 import P, assessment107
from model108 import Reconstructor, base, loss
from qualify108 import foreground_statistic, synthetic_control
from train108 import append_aux


class Contracts(unittest.TestCase):
    def test_foreground(self):
        np.testing.assert_array_equal(foreground_statistic(np.array([0., 1., 3.]), np.array([0, 2, 4])),
                                      [0, .5, .75])
        self.assertEqual(synthetic_control()["kernel_predictions"], [1.0, 1.05])

    def test_guide_order(self):
        aux = np.broadcast_to(np.arange(14, dtype=np.float32)[:, None, None], (14, 2, 2))
        field = np.ones((2, 2), np.float32) * 17
        merged = append_aux(aux, field)
        np.testing.assert_array_equal(merged[:14], aux)
        np.testing.assert_array_equal(merged[14], field)

    def test_matched_kernels_and_neutral_replay(self):
        torch.manual_seed(24)
        old = base.Reconstructor("R3").eval()
        torch.manual_seed(24)
        new = Reconstructor().eval()
        self.assertEqual(base.n_params(new) - base.n_params(old), 414)
        for key, value in old.state_dict().items():
            tensor = new.state_dict()[key]
            if tensor.shape != value.shape:
                self.assertTrue(torch.equal(tensor[:, :-1], value))
                self.assertEqual(int(tensor[:, -1].count_nonzero()), 0)
            else:
                self.assertTrue(torch.equal(tensor, value))
        comp, aux = torch.rand(1, 9, 16, 16), torch.rand(1, 14, 16, 16)
        with torch.no_grad():
            actual, parts = new(comp, torch.cat((aux, torch.zeros_like(aux[:, :1])), 1))
            reference, _ = old(comp, aux)
        torch.testing.assert_close(actual, reference, atol=2e-6, rtol=2e-6)
        self.assertTrue(torch.equal(actual, parts["low"] + parts["high"]))
        for network in (new.net_low, new.net_high):
            gx = torch.rand(1, 18, 16, 16)
            kernel = network(gx).softmax(1)
            self.assertGreaterEqual(float(kernel.min()), 0)
            torch.testing.assert_close(kernel.sum(1), torch.ones((1,16,16)))

    def test_descriptor_reaches_kernel(self):
        model = Reconstructor()
        with torch.no_grad():
            for p in model.parameters():
                p.zero_()
            low = model.net_low
            low.enc[0][0].weight[0, -1, 1, 1] = 1
            low.enc[0][2].weight[0, 0, 1, 1] = 1
            skip_offset = low.enc[1][0].out_channels
            low.dec[-1][0].weight[0, skip_offset, 1, 1] = 1
            low.dec[-1][2].weight[0, 0, 1, 1] = 1
            low.head.weight[0, 0, 0, 0] = 1
        guide = torch.zeros(1, 18, 16, 16)
        guide[:, -1] = .25
        a = low(guide).softmax(1)[:, 0]
        guide[:, -1] = .34
        b = low(guide).softmax(1)[:, 0]
        self.assertGreater(float((b-a).min()), 0)

    def test_loss_isolation(self):
        target = torch.zeros(1,9,1,1)
        parts = {"low": torch.ones(1,3,1,1), "high": torch.ones(1,3,1,1)}
        pred = parts["low"]+parts["high"]
        aux = torch.zeros(1,15,1,1)
        v = loss(pred, parts, target, aux, False)
        self.assertTrue(torch.equal(v, loss(pred, parts, target, aux, True)))
        aux[:,6] = .5
        torch.testing.assert_close(loss(pred, parts, target, aux, True), v + .25*4/4.01)

    def test_covariance_and_training_contract(self):
        parts = np.array([1, 3, 2, 0], dtype=np.float32).reshape(2,2,1,1,1).repeat(3,axis=2)
        out = parts.sum(axis=1)
        assess = assessment107()
        lo, hi, twice, variance = assess.variance_components(out, parts)
        np.testing.assert_allclose(lo+hi+twice, assess.luma(out).var(axis=0,ddof=1))
        np.testing.assert_allclose(variance, assess.luma(out).var(axis=0,ddof=1))
        self.assertEqual(len(set(P["training"]["train_seeds"])), 3)
        self.assertEqual(len(set(tuple(f) for f in P["arms"].values())), 4)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
