"""WL111 training data (Stage D) and sealed-view acceptance. GPU, frozen renderer only (no reconstruction, no reference).

python data_r111.py OUT DESIGN_JSON [--smoke]
  OUT/sealed_views.json          K1-K6 accepted by the geometry-only rule (protocol views.sealed)
  OUT/ladder/<view>.npz          comps (12 levels, 2 replicates, 9, H, W) float16: spp ladder 1 .. 2048, ladder seed family
  OUT/ladder/<view>_guides.npz   candidate guides (sub from DESIGN_JSON): cover, normal, depth, valid_frac, pure
  OUT/data_manifest.json         seeds, hashes, timings
"""
import argparse
import time
from pathlib import Path

import numpy as np

import common_r111 as C
import engine_r111 as E
import seq_r111 as SQ

LEVELS = 12
REPS = 2


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def seal(eng, out, smoke):
    s = C.P["views"]["sealed"]
    cands = SQ.sealed_candidates()
    acc, rec = [], []
    for i, cam in enumerate(cands):
        c = E.cam_dict(cam)
        g2 = eng.guides(c, 2)
        g8 = eng.guides(c, 8)
        vf = float(g2["valid_frac"].mean())
        sph = int((g8["cover"][6] >= 1.0).sum())
        ok = vf >= 0.9 and sph >= 2000
        rec.append({"index": i, "camera": cam, "valid_fraction_2x2": vf, "sphere_full_pixels": sph, "accepted": ok})
        if ok:
            acc.append(cam)
        if len(acc) == len(s["names"]):
            break
    assert len(acc) == len(s["names"]), "not enough sealed views"
    res = {"rule": s["rule"], "views": dict(zip(s["names"], acc)), "candidates_examined": rec}
    C.write(out / "sealed_views.json", res)
    return res


def ladder(eng, out, sub, smoke):
    cams = SQ.wl105_cameras()
    names = sorted(k for k in cams if k.startswith("T")) + sorted(k for k in cams if k.startswith("Q"))
    assert len(names) == 48
    if smoke:
        names = [names[0], names[40]]
    d = out / "ladder"
    d.mkdir(parents=True, exist_ok=True)
    man = {}
    for vi, name in enumerate(names):
        f = d / f"{name}.npz"
        if f.exists():
            man[name] = C.read(d / f"{name}.json")
            continue
        t0 = time.perf_counter()
        cam = E.cam_dict(cams[name])
        comps = np.zeros((LEVELS, REPS, 9, C.RES, C.RES), np.float16)
        seeds = {}
        for lv in range(LEVELS if not smoke else 4):
            spp = 2 ** lv
            for r in range(REPS):
                sd = C.seed("ladder", vi // 10, (vi % 10) * 10 + r, lv, smoke=smoke)
                comps[lv, r] = eng.blocks(cam, spp, sd, b=min(32, spp)).astype(np.float16)
                seeds[f"{lv}_{r}"] = sd
        np.savez(f, comps=comps)
        g = eng.guides(cam, sub)
        np.savez(d / f"{name}_guides.npz", **{k: v.cpu().numpy() for k, v in g.items()})
        rec = {"view": name, "view_index": vi, "camera": cam, "seeds": seeds, "sha256": C.sha(f),
               "guides_sha256": C.sha(d / f"{name}_guides.npz"), "seconds": time.perf_counter() - t0,
               "finite": bool(np.isfinite(comps).all()), "max": float(np.abs(comps.astype(np.float32)).max())}
        C.write(d / f"{name}.json", rec)
        man[name] = rec
        log("ladder", name, round(rec["seconds"], 1))
    return man


def main(out, design, smoke):
    out.mkdir(parents=True, exist_ok=True)
    dsg = C.read(design)
    assert smoke or not dsg["cost_futility_stop"], "cost futility: no training"
    rt, Bd, B, audit = E.admit(log, replay=not smoke)
    eng = E.Engine(rt, Bd, B)
    man = {"protocol_sha256": C.sha(C.PROTOCOL), "design": dsg, "admission": audit, "smoke": smoke,
           "code_sha256": {f.name: C.sha(f) for f in sorted(C.HERE.glob("*.py"))}}
    man["sealed"] = seal(eng, out, smoke)
    man["ladder"] = ladder(eng, out, dsg["sub"], smoke)
    C.write(out / "data_manifest.json", man)
    print("DATA_COMPLETE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("design", type=Path)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    main(a.out.resolve(), a.design.resolve(), a.smoke)
