"""WL55 amendment A1.2: reference pixel-size diagnostic for the oblique side-face groups (rule fixed before the renders).
    P53_MI_VARIANT=cuda_ad_rgb python k55_pixdiag.py <out json>"""
import json
import sys
from pathlib import Path

import numpy as np

import k55_analyze as KA
import k55_common as KCm
import k55_fixtures as KF

ROOT = KA.ROOT / "out"


def ports(scene, cdir, cam, seeds):
    sd = KF.scene_def(scene); ss = KCm.classify_sceneset(sd); g = json.load(open(cdir / f"ccam_{scene}.json"))[cam]
    return KCm.cycles_ports(ss, g, [np.load(cdir / f"cimg_{scene}_{cam}_s{s}.npy") for s in seeds])["amended"], KCm.labels_of(sd), sd


def main(out):
    res = {}
    for scene, seeds0, s1 in (("gap8", (5501, 5502), 5541), ("gap4", (5501, 5502), 5541), ("kf_gap8_U", (5511, 5512), 5542), ("kf_gap4_U", (5511, 5512), 5542), ("gap6s", (5521, 5522), 5543)):
        for cam in ("obot", "obotx"):
            if not (ROOT / "cycles_px0.125" / f"cimg_{scene}_{cam}_s{s1}.npy").exists():
                continue
            u05, lab, sd = ports(scene, ROOT / "cycles_cuda", cam, seeds0); rec = {}
            us = {"0.25": ports(scene, ROOT / "cycles_px0.25", cam, (s1,))[0], "0.125": ports(scene, ROOT / "cycles_px0.125", cam, (s1,))[0]}
            med = KF.medium_ports(sd)
            for gn in np.unique(lab[u05["sel"] & med]):
                m = u05["sel"] & (lab == gn) & us["0.125"]["sel"]
                if m.sum() < 5:
                    continue
                sig = {px: u["mean"][m].sum(0) for px, u in (("0.5", u05), *us.items())}
                ch = (sig["0.125"] - sig["0.5"]) / sig["0.5"]
                rec[gn] = dict(n=int(m.sum()), signal={k: v.tolist() for k, v in sig.items()}, change_0p5_to_0p125=ch.tolist(), not_converged=bool(np.max(np.abs(ch)) >= 0.03))
            res[f"{scene}|{cam}"] = rec
            print(scene, cam, {g: [round(100 * x, 1) for x in v["change_0p5_to_0p125"]] for g, v in rec.items()})
    KCm.dump(res, out)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
