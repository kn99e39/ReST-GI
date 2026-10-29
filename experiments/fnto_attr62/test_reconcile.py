import math,unittest
from reconcile import moment_sensitivity,assert_distinct
class ReconciliationTests(unittest.TestCase):
    def test_zero_noise(self):self.assertAlmostEqual(moment_sensitivity(.4,0)['relative_error'],.4)
    def test_squared_correction(self):self.assertAlmostEqual(moment_sensitivity(.5,.1)['relative_squared'],.24/.99)
    def test_negative_not_clamped(self):
        r=moment_sensitivity(.01,.1);self.assertLess(r['relative_squared'],0);self.assertIsNone(r['relative_error'])
    def test_undefined_signal(self):self.assertIsNone(moment_sensitivity(.5,1)['relative_squared'])
    def test_population_alias_guard(self):
        assert_distinct(['train8'],['test8'])
        with self.assertRaises(ValueError):assert_distinct(['same'],['same'])
if __name__=='__main__':unittest.main()
