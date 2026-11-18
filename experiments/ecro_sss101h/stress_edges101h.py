"""WL101H Q1b: native attachment at planar face edges as a function of metric inset (membership band check).

For every planar patch side, points are placed at metric insets from that side and attached through the
unchanged WL101E single-shape attachment. Interior patch sides lie inside a face and always attach; only
true face edges can miss. The declared metric band b_m = 16 ulp32(3 m) = 3.8147e-6 m is accepted if the
largest inset with any miss is <= b_m / 2.

python stress_edges101h.py WL101_ROOT OUT_DIR
"""
from pathlib import Path
import json
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ecro_sss101g"))
sys.path.insert(0, str(HERE))
argv = sys.argv[:]
sys.argv = sys.argv[:1]
import run_generalization as RG  # noqa: E402
from oracle_field import ExposedOracle  # noqa: E402

INSETS = [0.0, 1e-8, 2e-8, 5e-8, 1e-7, 2e-7, 5e-7, 1e-6, 2e-6, 4e-6]
B_M = 16 * np.spacing(np.float32(3.0)).astype(np.float64)


def main():
    root, out = Path(argv[1]).resolve(), Path(argv[2]).resolve()
    reader, *_ = RG.F101.inputs(root)
    R, rt, dr, mi = reader.R, reader.rt, reader.dr, reader.mi
    out.mkdir(parents=True, exist_ok=False)
    geom = rt.inc.G
    oracle = ExposedOracle(R, rt)
    planar = np.flatnonzero(geom.PSID != 6)
    rs = np.random.default_rng(1012)
    rows = []
    for pid in planar:
        po, e1, e2 = geom.PO[pid], geom.PE1[pid], geom.PE2[pid]
        l1, l2 = np.linalg.norm(e1), np.linalg.norm(e2)
        n = np.cross(e1, e2)
        n /= np.linalg.norm(n)
        for side in range(4):
            for inset in INSETS:
                t = rs.random(8)
                if side == 0: u0, u1 = np.full(8, inset / l1), t
                elif side == 1: u0, u1 = np.full(8, 1 - inset / l1), t
                elif side == 2: u0, u1 = t, np.full(8, inset / l2)
                else: u0, u1 = t, np.full(8, 1 - inset / l2)
                p = po[None] + u0[:, None] * e1[None] + u1[:, None] * e2[None]
                rows.append((pid, side, inset, p, np.broadcast_to(n, p.shape)))
    pid = np.concatenate([np.full(len(r[3]), r[0]) for r in rows])
    side = np.concatenate([np.full(len(r[3]), r[1]) for r in rows])
    inset = np.concatenate([np.full(len(r[3]), r[2]) for r in rows])
    p = np.concatenate([r[3] for r in rows]).astype(np.float32)
    n = np.concatenate([r[4] for r in rows]).astype(np.float32)
    sid = geom.PSID[pid]
    miss = np.zeros(len(p), bool)
    for shape in np.unique(sid):
        m = sid == shape
        Pf = lambda a: mi.Point3f(*[R.F(np.ascontiguousarray(a[m, k])) for k in range(3)])
        Vf = lambda a: mi.Vector3f(*[R.F(np.ascontiguousarray(a[m, k])) for k in range(3)])
        ray = mi.Ray3f(Pf(p + 0.002 * n), Vf(-n))
        ray.maxt = 0.003
        hit = oracle.shape_scenes[int(shape)].ray_intersect(ray)
        dr.eval(hit)
        miss[m] = ~np.asarray(hit.is_valid())
    by_inset = {f"{x:.0e}": {"points": int(np.sum(inset == x)), "misses": int(np.sum(miss & (inset == x)))} for x in INSETS}
    worst = float(inset[miss].max()) if miss.any() else 0.0
    result = {"b_m_m": float(B_M), "by_inset": by_inset, "largest_missing_inset_m": worst,
              "accepted": bool(worst <= B_M / 2), "patches": int(len(planar)), "points": int(len(p)),
              "missing_shapes": {int(s): int(np.sum(miss & (sid == s))) for s in np.unique(sid[miss])}}
    np.savez_compressed(out / "edge_stress.npz", pid=pid, side=side, inset=inset, miss=miss)
    (out / "edge_stress.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=1), flush=True)


if __name__ == "__main__":
    main()
