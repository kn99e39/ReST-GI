"""WL111A Stage B (H1): 8 x 8 sub-pixel guides for the 48 WL111 ladder views (same engine.guides function and cameras
as the WL111 sub-4 guides; only the sub-sampling differs). The ladder component frames are reused from WL111 unchanged.

python guides8_r111a.py      -> OUT/ladder8/<view>_guides.npz (+ manifest)
"""
import time

import numpy as np

import common_r111a as A
import common_r111 as C
import engine_r111 as E
import seq_r111 as SQ


def main():
    rt, Bd, B, audit = E.admit(print, replay=False)
    eng = E.Engine(rt, Bd, B)
    cams = SQ.wl105_cameras()
    lad = A.WL111_EVID / "data" / "ladder"
    names = sorted(p.stem for p in lad.glob("*.npz") if not p.stem.endswith("_guides"))
    assert len(names) == 48
    d = A.OUT / "ladder8"
    d.mkdir(parents=True, exist_ok=True)
    man = {"sub": 8, "views": {}}
    for n in names:
        g = eng.guides(E.cam_dict(cams[n]), 8)
        f = d / f"{n}_guides.npz"
        np.savez(f, **{k: v.cpu().numpy() for k, v in g.items()})
        # the sub-4 file's camera must be the same view: compare the ladder record's camera
        rec = C.read(lad / f"{n}.json")
        assert rec["camera"] == cams[n], n
        man["views"][n] = {"sha256": C.sha(f), "sub4_sha256": C.sha(lad / f"{n}_guides.npz")}
    C.write(d / "manifest.json", man)
    print("GUIDES8_COMPLETE", len(names), flush=True)


if __name__ == "__main__":
    main()
