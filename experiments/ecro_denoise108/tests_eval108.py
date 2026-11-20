"""Check the paired bootstrap, disjoint mask accounting and sealed cameras."""
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from common108 import P, read
from confirm108 import check_cameras
from eval108 import regions
from metrics108 import draws, effect


class EvaluationContracts(unittest.TestCase):
    def test_paired_bootstrap_identical_ratios(self):
        shape = 3,6,8
        old = np.random.default_rng(112).uniform(.25,2,size=shape)
        indices = draws(6,8)
        rank = effect(.8*old,old,indices)
        np.testing.assert_allclose([rank["point"],*rank["ci95"]],.2,atol=1e-12)
        ratio = effect(old.mean(2),old.mean(2),indices,"ratio")
        np.testing.assert_array_equal([ratio["point"],*ratio["ci95"]],[1.,1.,1.])

    def test_partition(self):
        primary = np.ones((12,12),dtype=bool)
        primary[0,0] = False
        cover = np.zeros((9,12,12),dtype=np.float32)
        cover[6,2:10,2:10] = 1
        cover[6,2,2:10] = .5
        silhouette = (cover[6] > 0)&(cover[6] < 1)
        fake = type("Fake",(),{"profile_masks":lambda _,aux,original:dict(original)})()
        rois = {"silhouette":silhouette,"primary":primary}
        with patch("eval108.rois",return_value=rois):
            _, parts = regions({"cover":cover},np.arange(144).reshape(12,12),fake)
        np.testing.assert_array_equal(np.stack(list(parts.values())).sum(0),primary)
        self.assertGreater(parts["inside_beyond_3"].sum(),0)
        self.assertEqual(parts["silhouette"].sum(),8)

    def test_camera_duplicate_rejected(self):
        cams = P["confirmation"]["cameras"]
        check_cameras(cams,[])
        with self.assertRaises(AssertionError):
            check_cameras(cams,[cams["I1"]])
        local_manifest = Path("results/ecro_denoise105/data/run_manifest.json")
        if local_manifest.is_file():
            check_cameras(cams,[v["camera"] for v in read(local_manifest)["views"].values()])


if __name__=="__main__":
    unittest.main()
