"""Focused tests for WL107's experiment-local analytical contracts."""
import unittest

import numpy as np

from common107 import coverage_masks, low_high, stencil, take_stencil, total
from oracle107 import (adaptive_intervals, adaptive_risk_bound, empirical_parts, project_simplex,
                       solve_numpy, synthetic_risk)
from synthetic107 import counterexamples, guides, midpoint_reference, truth


class Contracts(unittest.TestCase):
    def test_fold_and_order(self):
        x = np.arange(18 * 20).reshape(18, 20)
        centers = np.array([0, 19, 20 * 9 + 8, x.size - 1])
        got = take_stencil(x, stencil(centers, *x.shape))
        ref = np.pad(x, 7, mode="reflect")
        expected = np.stack([ref[y:y + 15, c:c + 15].ravel() for y, c in zip(*np.divmod(centers, 20))])
        np.testing.assert_array_equal(got, expected)

    def test_composition_and_immutability(self):
        x = np.arange(9 * 16 * 16, dtype=np.float64).reshape(9, 16, 16)
        before = x.copy()
        np.testing.assert_array_equal(low_high(x).sum(axis=0), total(x))
        np.testing.assert_array_equal(x, before)

    def test_coverage_semantics(self):
        cov = np.array([0, 1/64, 0.125, 0.875, 63/64, 1])
        bands = np.stack(coverage_masks(cov, (cov > 0) & (cov < 1)))
        np.testing.assert_array_equal(bands.sum(axis=0), [0, 1, 1, 1, 1, 0])
        self.assertTrue(bands[2, 2])
        self.assertTrue(bands[-1, 4])
        exact = 0.7
        self.assertAlmostEqual(guides(0.3)[0][0, 32], 0.75)
        self.assertGreater(abs(exact - guides(0.3)[0][0, 32]), 0.01)

    def test_analytic_truth(self):
        sample = (0.3, "dark", "rim", 0.25, True)
        x = truth(sample)
        self.assertLess(np.abs(x - midpoint_reference(sample)).max(), 0.002)
        self.assertEqual(x.shape, (2, 3, 64, 64))
        self.assertTrue((x > 0).all())
        a = (0.125, "bright", "constant", None, False)
        y = truth(a).sum(axis=0)
        expected = 0.875 * y[:, 0, 33] + 0.125 * y[:, 0, 31]
        np.testing.assert_allclose(y[:, 0, 32], expected, atol=1e-14)

    def test_simplex_and_duality_gap(self):
        v = np.array([[-10., 4, 6], [-2, 1, 1]])
        np.testing.assert_allclose(project_simplex(v), [[0, 0, 1], [0, 0.5, 0.5]])
        z = np.array([[[1., 2, 3], [4, -1, -2]]])
        f = lambda w: (np.square(w - z).sum(axis=(1, 2)), 2 * (w - z))
        w, best, lower, gap = solve_numpy(f, np.array([2]), np.full_like(z, 1/3), steps=10)
        np.testing.assert_allclose(w, project_simplex(z))
        np.testing.assert_allclose(best, lower)
        np.testing.assert_allclose(gap, [0])

    def test_exact_cross_covariance(self):
        mean = np.tile(np.array([1., 2.])[None, :, None, None], (1, 1, 225, 3))
        w = np.zeros((1, 2, 225))
        w[:, :, 112] = 1
        f, _ = synthetic_risk(mean, 4, 1)
        expected = ((0.42 * 3) ** 2 + 0.63 ** 2 + (0.51 * 2) ** 2 + (0.10 * 3) ** 2) / 4
        self.assertAlmostEqual(f(w)[0][0], expected)
        l = np.array([1., -1, 1, -1])
        vl, vh, cov, vt = empirical_parts(l, 2 * l)
        self.assertAlmostEqual(cov, 4 * vl)
        self.assertAlmostEqual(vt, vl + vh + cov)

    def test_directional_gradient(self):
        rng = np.random.default_rng(1)
        x = rng.uniform(0.5, 2.5, (2, 2, 225, 3))
        f, _ = synthetic_risk(x, 4, 1)
        w = np.full((2, 2, 225), 1/225)
        delta = rng.normal(size=w.shape)
        delta -= delta.mean(axis=-1, keepdims=True)
        eps = 1e-6
        observed = (f(w + eps * delta)[0] - f(w - eps * delta)[0]) / (2 * eps)
        predicted = (f(w)[1] * delta).sum(axis=(1, 2))
        np.testing.assert_allclose(observed, predicted, atol=1e-9)

    def test_adaptive_and_information_bounds(self):
        taps = np.array([[[0.2, 0.5], [1, 2]]])
        lo, hi = adaptive_intervals(taps)
        np.testing.assert_allclose(adaptive_risk_bound(lo, hi, [0.2]), [1])
        ex = counterexamples()
        self.assertAlmostEqual(ex["adaptive_extremum"]["ratio_floor"], 0.64)
        for s in (4, 16, 32):
            row = ex["equal_guide_pair"]["view"][str(s)]
            self.assertGreater(row["existing Bayes MSE"], 0)
            self.assertGreater(row["prob_common_given_B"], 0)
            self.assertEqual(row["privileged Bayes MSE"], 0)

    def test_seed_moment_and_determinism(self):
        from synthetic107 import draws
        config = (0.125, "bright", "constant", None, False)
        args = (config, 32)
        a = draws(*args, np.random.default_rng(1), 256)
        b = draws(*args, np.random.default_rng(1), 256)
        np.testing.assert_array_equal(a, b)
        self.assertGreaterEqual(a.min(), 0)
        self.assertLess(np.abs(a.mean(axis=0) - truth(config)).mean(), 0.01)
        vl, vh, c, vt = empirical_parts(a[:, 0, 0], a[:, 1, 0])
        self.assertGreater(c.mean(), 0)
        np.testing.assert_allclose(vl + vh + c, vt, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
