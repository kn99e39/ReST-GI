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

    def test_variance_check_distinguishes_roundoff_from_composition(self):
        from real107 import luma, variance_components
        from oracle107 import empirical_parts
        low = [[15.565797, 15.544671, 15.528677], [15.567665, 15.547737, 15.533305],
               [15.5663185, 15.546453, 15.530813], [15.567047, 15.544995, 15.529252],
               [15.549951, 15.531599, 15.517065], [15.580654, 15.563723, 15.550202],
               [15.561088, 15.544073, 15.5306015], [15.565063, 15.54156, 15.526003]]
        high = [[.997481, .9559686, .92776746], [1.0068092, .9623786, .9318276],
                [1.0013227, .9582906, .9287806], [.9953201, .95242494, .9232255],
                [1.0003884, .9583996, .92975783], [.9966406, .9554455, .9272064],
                [.99926066, .9570166, .9283791], [.99227965, .9536151, .9271259]]
        # These 8 independent predictions share one bright V0 pixel.
        parts = np.asarray((low, high), dtype=np.float32).transpose(1, 0, 2)[..., None, None]
        out = parts.sum(axis=1).astype(np.float64)
        previous_total = empirical_parts(luma(parts[:, 0]), luma(parts[:, 1]))[-1]
        self.assertFalse(np.allclose(previous_total, luma(out).var(axis=0, ddof=1), rtol=1e-4, atol=1e-10))
        restored = variance_components(out, parts)
        np.testing.assert_allclose(np.sum(restored[:3], axis=0), restored[-1], rtol=1e-12)
        invalid = out.copy()
        invalid[0, 0, 0, 0] += 1e-4
        with self.assertRaises(AssertionError):
            variance_components(invalid, parts)

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
