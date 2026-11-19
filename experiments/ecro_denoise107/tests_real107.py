"""Focused numerical and predictor checks for the WL107 real-scene code."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from common107 import total
from oracle107 import min_adaptive_variance, solve_factor_torch
from reference107 import assemble


class RealContracts(unittest.TestCase):
    def test_reference_block_uncertainty(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            for h, v in enumerate((1.0, 3.0)):
                np.savez(path / f"target_half{h}.npz", comp=np.full((9, 2, 2), v))
            mask = np.ones((2, 2), dtype=bool)
            target, info = assemble(path, None, {"primary": mask}, {"primary": mask})
            np.testing.assert_array_equal(total(target), 6.0)
            self.assertAlmostEqual(info["relative_rgb_rms_se"]["primary"], 0.5)
            self.assertAlmostEqual(info["relative_signed_luma_se"]["primary"], 0.5)

    def test_min_adaptive_variance(self):
        lo = np.array([[0., 0.], [4., 0.], [2., 1.]])
        hi = np.array([[2., 3.], [5., 4.], [4., 3.]])
        # First column can attain (2, 4, 3). The second admits a common value.
        np.testing.assert_allclose(min_adaptive_variance(lo, hi), [1., 0.], atol=1e-12)

    def test_solver_reference(self):
        import torch
        torch.set_num_threads(2)
        a = np.zeros((1, 2, 450))
        a[0, 0, 112] = 1
        a[0, 1, 337] = 1
        offset = np.array([[-0.4, -0.7]])
        weights, value, lb, gap = solve_factor_torch(a, offset, "cpu", steps=80)
        np.testing.assert_allclose(weights.sum(-1), 1, atol=1e-10)
        np.testing.assert_allclose(weights[:, :, 112], [[0.4, 0.7]], atol=1e-6)
        self.assertLess(value.max(), 1e-10)
        self.assertLess(gap.max(), 1e-6)
        self.assertTrue((lb <= value).all())

    def test_covariance_factor_matches_applied_samples(self):
        from common107 import stencil
        from real107 import compose_taps, fit_design
        rng = np.random.default_rng(18)
        components = rng.uniform(0, 1, (6, 9, 16, 16))
        taps = stencil(np.array([34, 128]), 16, 16)
        x = compose_taps(components, taps)
        _, var, offset = fit_design(x.mean(0), x, 4)
        w = rng.uniform(0, 1, (2, 2, 225))
        w /= w.sum(-1, keepdims=True)
        fitted = np.einsum("nrc,nc->nr", var, w.reshape(2, -1)) + offset
        evaluated = np.einsum("rgcpt,pgt->rpc", x, w)
        np.testing.assert_allclose(np.square(fitted).sum(1), evaluated.var(0, ddof=1).mean(-1) * 8)

    def test_matched_predictor(self):
        import torch
        from predict107 import SignedDiagnostic, M, apply_signed, signed_weights
        torch.manual_seed(8)
        diagnostic = SignedDiagnostic()
        torch.manual_seed(8)
        r3 = M.Reconstructor("R3")
        self.assertEqual(M.n_params(diagnostic), M.n_params(r3))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(diagnostic.parameters(), r3.parameters())))
        logits = torch.randn(1, 225, 16, 16, requires_grad=True)
        w = signed_weights(logits)
        self.assertLess(float(w.min().detach()), 0.)
        torch.testing.assert_close(w.sum(1), torch.ones_like(w[:, 0]))
        image = torch.ones(1, 3, 16, 16)
        before = image.clone()
        actual = apply_signed(image, logits)
        torch.testing.assert_close(actual, image)
        actual.mean().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        torch.testing.assert_close(image, before)


if __name__ == "__main__":
    unittest.main(argv=sys.argv)
