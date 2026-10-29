import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import angular_preflight as A


class AnalyticContracts(unittest.TestCase):
    def test_map_roundtrip(self):
        b = np.arange(64)
        w = A.square_direction((b%8+.37)/8, (b//8+.61)/8)
        np.testing.assert_array_equal(A.direction_bin(w, 8), b)
        np.testing.assert_allclose(np.square(w).sum(1), 1, atol=1e-14)

    def test_reciprocity_and_normal_peak(self):
        v = A.analytic_checks()
        self.assertLess(v['reciprocity_relative_max'], 1e-10)
        self.assertLess(v['normal_peak_relative_max'], 1e-10)

    def test_binned_measure_and_optimal_projection(self):
        s = A.field_stats(np.array([0., 0., 1.]), .5, 64, 16)
        self.assertAlmostEqual(sum(s['output_flux']), s['energy'], places=13)
        self.assertGreater(s['energy'], 0.)
        self.assertLessEqual(s['energy'], 1.001)
        self.assertGreaterEqual(s['best_output_relative_l2'], 0.)
        self.assertLessEqual(s['best_output_relative_l2'], 1.)

    def test_pair_lower_bound(self):
        a, b = np.array([1., 0.]), np.array([0., 1.])
        self.assertAlmostEqual(A.pair_bound(a, b), 1/math.sqrt(2))
        self.assertEqual(A.pair_bound(a, a), 0.)
        # The equal-norm midpoint attains this lower bound.
        z = .5*(a+b)
        self.assertAlmostEqual(A.rel(z, a), A.pair_bound(a, b))

    def test_uniform_bin_sources_have_identical_input(self):
        for cell in (18, 27):
            a = A.square_direction((cell%8+.2)/8, (cell//8+.5)/8)
            b = A.square_direction((cell%8+.8)/8, (cell//8+.5)/8)
            self.assertEqual(int(A.direction_bin(a, 8)), cell)
            self.assertEqual(int(A.direction_bin(b, 8)), cell)
            self.assertGreater(np.linalg.norm(a-b), 0.)


if __name__ == '__main__':
    unittest.main()
