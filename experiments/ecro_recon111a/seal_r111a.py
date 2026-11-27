"""WL111A final evaluation inputs: newly sealed views N1-N6 (protocol views.sealed_new_rule) and their T_ECRO references
(32 x 8192 spp, per-component variance of the mean). Resumable per view.

python seal_r111a.py      -> OUT/final/sealed_new.json, OUT/final/refs/N<k>.npz
"""
import sys
import time

import numpy as np

import common_r111a as A
import common_r111 as C
import engine_r111 as E
import seq_r111 as SQ
import stageA_r111a as SA

P = C.read(A.HERE / "protocol" / "recon111a.json")
NBLK, SPPB = 32, 8192


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def main():
    out = A.OUT / "final"
    (out / "refs").mkdir(parents=True, exist_ok=True)
    eng, audit = SA.setup()
    sys.path.insert(0, str(C.EXP / "ecro_denoise105"))
    import cameras105 as CAM
    f = out / "sealed_new.json"
    if not f.exists():
        k111 = C.read(A.WL111_EVID / "data" / "sealed_views.json")["views"]
        forbidden = list(SQ.wl105_cameras().values()) + list(k111.values())
        cands = CAM.random_cameras(200, P["seeds"]["sealed_camera_seed"], SQ.rows()[0]["geom"], forbidden)
        names = P["views"]["sealed_new"]
        acc, rec = [], []
        for i, cam in enumerate(cands):
            c = E.cam_dict(cam)
            vf = float(eng.guides(c, 2)["valid_frac"].mean())
            sph = int((eng.guides(c, 8)["cover"][6] >= 1.0).sum())
            ok = vf >= 0.9 and sph >= 2000
            rec.append({"index": i, "camera": cam, "valid_fraction_2x2": vf, "sphere_full_pixels": sph, "accepted": ok})
            if ok:
                acc.append(cam)
            if len(acc) == len(names):
                break
        assert len(acc) == len(names), "not enough sealed views"
        C.write(f, {"views": dict(zip(names, acc)), "candidates_examined": rec, "admission": audit})
    views = C.read(f)["views"]
    for vi, (name, cam) in enumerate(views.items()):
        p = out / "refs" / f"{name}.npz"
        if p.exists():
            continue
        t0 = time.perf_counter()
        c = E.cam_dict(cam)
        sums = {k: [0.0, 0.0] for k in ("low", "high", "total")}
        mean = 0.0
        for b in range(NBLK):
            comp = eng.blocks(c, SPPB, A.seed("ref_new", vi, 0, b), b=32)
            parts = {"low": comp[0:3] + comp[3:6], "high": comp[6:9]}
            parts["total"] = parts["low"] + parts["high"]
            mean = mean + comp
            for k, x in parts.items():
                sums[k][0] = sums[k][0] + x
                sums[k][1] = sums[k][1] + x * x
        mean = mean / NBLK
        var = {k: np.maximum(s[1] / NBLK - (s[0] / NBLK) ** 2, 0.0) * NBLK / (NBLK - 1) / NBLK for k, s in sums.items()}
        aux = eng.roi_aux(c)
        np.savez(p, mean=mean.astype(np.float32), var_total=var["total"].astype(np.float32),
                 **{f"var_{k}": v.astype(np.float32) for k, v in var.items() if k != "total"},
                 blocks=NBLK, spp_block=SPPB, **{f"aux_{k}": x for k, x in aux.items()})
        log("ref", name, round(time.perf_counter() - t0, 1), "s")
    print("SEAL_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
