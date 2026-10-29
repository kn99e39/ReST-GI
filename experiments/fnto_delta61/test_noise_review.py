import itertools,unittest
import numpy as np
from noise_review import cross_squared
class CrossSquaredTests(unittest.TestCase):
    def test_independent_sign_enumeration(self):
        d=np.array([3.,4.]);p=np.array([2.,5.]);errors=[];signals=[]
        for a,b in itertools.product([-1.,1.],repeat=2):
            e,s=cross_squared(p,d+a*np.array([.5,1.]),d+b*np.array([1.,-.25]));errors.append(e);signals.append(s)
        self.assertAlmostEqual(float(np.mean(errors)),float(np.sum((p-d)**2)))
        self.assertAlmostEqual(float(np.mean(signals)),float(np.sum(d*d)))
if __name__=='__main__':unittest.main()
