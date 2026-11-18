"""WL101I development investigations on WL101H (development) seeds. Read-only geometry/visibility replay.

python dev101i.py d4    WL101_ROOT OUT SEED SLOT [SLOT ...]   candidate visibility flips at offset m vs m/10
python dev101i.py d3    WL101_ROOT OUT SEED LAUNCH SLOT       replay one unregistered event (candidate)
python dev101i.py flips WL101_ROOT OUT SEED                   subcell address under two kernel compositions
"""
from pathlib import Path
import json
import sys
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ecro_sss101g"))
sys.path.insert(0, str(HERE.parent / "ecro_sss101h"))
argv = sys.argv[:]
sys.argv = sys.argv[:1]
import run_generalization as RG  # noqa: E402
import gpu101g as GG  # noqa: E402
import semantics101h as SEM  # noqa: E402
import instrument101h as I  # noqa: E402
from oracle_field import ExposedOracle  # noqa: E402

H_MIN = json.loads((HERE.parent / "ecro_sss101h/protocol/qualification.json").read_text())["candidate"]["h_min_m"]
B_M = json.loads((HERE.parent / "ecro_sss101h/protocol/qualification.json").read_text())["membership"]["b_m_m"]


def setup(root):
    reader, *_ = RG.F101.inputs(root)
    R, rt = reader.R, reader.rt
    contact = SEM.install(R, rt, H_MIN)
    import construct101h as CO
    geom = rt.inc.G
    lengths = (R.F(np.linalg.norm(geom.PE1, axis=1).astype(np.float32)), R.F(np.linalg.norm(geom.PE2, axis=1).astype(np.float32)))
    oracle = CO.metric_oracle(ExposedOracle, B_M, lengths)(R, rt)
    return reader, R, rt, contact, oracle


def d4(root, out, seed, slots):
    reader, R, rt, contact, oracle = setup(root)
    dr, mi = R.dr, R.mi
    target = R.U(np.array(sorted(slots), np.uint32))
    rows = []
    for launch in range(128):
        sl, ray, hits, rng, sd, info = oracle.probes(32, seed, launch)
        hit_slot = dr.zeros(mi.Bool, dr.width(sl))
        for s in slots:
            hit_slot |= dr.eq(sl, s)
        ids = dr.compress(info["valid"] & hit_slot)
        if dr.width(ids) == 0:
            continue
        g = lambda x: dr.gather(type(x), x, ids)
        hit, r, req = g(hits), g(ray), dr.gather(R.U, sl, ids)
        dr.eval(hit, r, req)
        w = -r.d
        vray, _, _ = contact.spawn(hit, w)
        y = rt.scene.ray_intersect(vray)
        dr.eval(y)
        o_small = hit.p + (vray.o - hit.p) * 0.1
        ys = rt.scene.ray_intersect(mi.Ray3f(o_small, w))
        dr.eval(ys)
        front = y.is_valid() & (y.to_local(-w).z > 1e-6)
        front_s = ys.is_valid() & (ys.to_local(-w).z > 1e-6)
        rows.append({k: v for k, v in {
            "slot": np.asarray(req), "p": I.P(hit.p), "n": I.P(hit.n), "w": I.P(w), "o": I.P(vray.o), "o_small": I.P(o_small),
            "hit": np.asarray(y.is_valid()), "front": np.asarray(front), "shape": np.asarray(rt.shape_id(y)), "t": np.asarray(y.t),
            "yp": I.P(y.p), "yn": I.P(y.n),
            "hit_s": np.asarray(ys.is_valid()), "front_s": np.asarray(front_s), "shape_s": np.asarray(rt.shape_id(ys)),
            "t_s": np.asarray(ys.t), "yp_s": I.P(ys.p), "yn_s": I.P(ys.n)}.items()})
    data = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"d4_{seed}.npz", **data)
    flip = data["front"] != data["front_s"]
    print(json.dumps({"samples": int(len(flip)), "flips": int(flip.sum())}))


def d3(root, out, seed, launch, slot):
    reader, R, rt, contact, oracle = setup(root)
    dr, mi = R.dr, R.mi
    sl, ray, hits, rng, sd, info = oracle.probes(32, seed, launch)
    ids = dr.compress(info["valid"] & dr.eq(sl, slot))
    g = lambda x: dr.gather(type(x), x, ids)
    hit, r = g(hits), g(ray)
    dr.eval(hit, r)
    w = -r.d
    vnat = hit.spawn_ray(w)
    vray, sing, need = contact.spawn(hit, w)
    res = []
    for label, rr in (("native", vnat), ("candidate", vray)):
        y = rt.scene.ray_intersect(rr)
        dr.eval(y)
        o = I.P(rr.o)
        res.append({"label": label, "o": o.tolist(), "hit": np.asarray(y.is_valid()).tolist(),
                    "shape": np.asarray(rt.shape_id(y)).tolist(), "t": np.asarray(y.t).tolist(), "y": I.P(y.p).tolist()})
    rec = {"seed": seed, "launch": launch, "slot": slot, "pid": slot // 64, "bin": slot % 64,
           "sid": int(rt.inc.G.PSID[slot // 64]), "p": I.P(hit.p).tolist(), "n": I.P(hit.n).tolist(), "w": I.P(w).tolist(),
           "candidate_singular": np.asarray(sing).tolist(), "candidate_moved": np.asarray(need).tolist(), "rays": res}
    out.mkdir(parents=True, exist_ok=True)
    (out / f"d3_{seed}_{launch}_{slot}.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec)[:3000])


def flips(root, out, seed):
    """Subcell address of every valid sample computed (A) as in WL101G (gather, then addresses in one fused
    kernel) and (B) as in WL101H (gathered fields evaluated first). Records disagreements."""
    reader, *_ = RG.F101.inputs(root)
    R, rt = reader.R, reader.rt
    dr = R.dr
    oracle = GG.amended_oracle(ExposedOracle, 8e-6)(R, rt)
    tol = 3e-5
    moved = []
    total = 0
    for launch in range(128):
        sl, ray, hits, rng, sd, info = oracle.probes(32, seed, launch)
        ids = dr.compress(info["valid"])
        g = lambda x: dr.gather(type(x), x, ids)
        hitA, rA, reqA = g(hits), g(ray), dr.gather(R.U, sl, ids)
        qA = SimpleNamespace(point=hitA.p, ray=rA, pid=reqA // 64, bn=reqA % 64)
        _, sA, tA, _ = reader.addresses(qA, tol)
        hitB, rB, reqB = g(hits), g(ray), dr.gather(R.U, sl, ids)
        dr.eval(hitB, rB, reqB)
        qB = SimpleNamespace(point=hitB.p, ray=rB, pid=reqB // 64, bn=reqB % 64)
        _, sB, tB, _ = reader.addresses(qB, tol)
        a, b, c, d = reader.coordinates(qB.point, -qB.ray.d, qB.pid, qB.bn)
        sA, tA, sB, tB = (np.asarray(x) for x in (sA, tA, sB, tB))
        diff = (sA != sB) | (tA != tB)
        total += len(sA)
        if diff.any():
            idx = np.flatnonzero(diff)
            moved.append({"launch": launch, "slot": np.asarray(reqB)[idx].tolist(), "sA": sA[idx].tolist(), "sB": sB[idx].tolist(),
                          "tA": tA[idx].tolist(), "tB": tB[idx].tolist(),
                          "abcd": [np.asarray(x)[idx].tolist() for x in (a, b, c, d)]})
    out.mkdir(parents=True, exist_ok=True)
    n = sum(len(m["slot"]) for m in moved)
    (out / f"flips_{seed}.json").write_text(json.dumps({"seed": seed, "valid": total, "moved": n, "records": moved}))
    print(json.dumps({"seed": seed, "valid": total, "moved": n}))


def near_border(root, out, seed, slots):
    """GPU float32 subcell coordinates of samples in the given slots whose a/b/c/d lies within 1e-5 of 0.5,
    with an independent float64 recomputation of the azimuth coordinate d."""
    reader, *_ = RG.F101.inputs(root)
    R, rt = reader.R, reader.rt
    dr = R.dr
    oracle = GG.amended_oracle(ExposedOracle, 8e-6)(R, rt)
    target = set(slots)
    rows = []
    for launch in range(128):
        sl, ray, hits, rng, sd, info = oracle.probes(32, seed, launch)
        mask = dr.zeros(R.mi.Bool, dr.width(sl))
        for s in slots:
            mask |= dr.eq(sl, s)
        ids = dr.compress(info["valid"] & mask)
        if dr.width(ids) == 0:
            continue
        g = lambda x: dr.gather(type(x), x, ids)
        hit, r, req = g(hits), g(ray), dr.gather(R.U, sl, ids)
        dr.eval(hit, r, req)
        q = SimpleNamespace(point=hit.p, ray=r, pid=req // 64, bn=req % 64)
        a, b, c, d = (np.asarray(x, np.float64) for x in reader.coordinates(q.point, -q.ray.d, q.pid, q.bn))
        tx, ty, tz = rt.patch_frame(q.pid)
        w = -I.P(r.d)
        T = (I.P(tx), I.P(ty))
        bn = np.asarray(q.bn)
        d64 = (np.arctan2(np.einsum("ij,ij->i", w, T[1]), np.einsum("ij,ij->i", w, T[0])) + np.pi) * (16 / (2 * np.pi)) - (bn % 16)
        close = np.min(np.abs(np.stack([a, b, c, d]) - 0.5), 0) < 1e-5
        idx = np.flatnonzero(close)
        rows.append({"launch": launch, "slot": np.asarray(req)[idx].tolist(), "a": a[idx].tolist(), "b": b[idx].tolist(),
                     "c": c[idx].tolist(), "d": d[idx].tolist(), "d64": d64[idx].tolist()})
    out.mkdir(parents=True, exist_ok=True)
    (out / f"near_border_{seed}.json").write_text(json.dumps(rows))
    print(json.dumps({"near_border_samples": sum(len(r["slot"]) for r in rows)}))


if __name__ == "__main__":
    cmd, root, out = argv[1], Path(argv[2]).resolve(), Path(argv[3]).resolve()
    if cmd == "d4":
        d4(root, out, int(argv[4]), [int(s) for s in argv[5:]])
    elif cmd == "d3":
        d3(root, out, int(argv[4]), int(argv[5]), int(argv[6]))
    elif cmd == "near":
        near_border(root, out, int(argv[4]), json.loads(Path(argv[5]).read_text()))
    else:
        flips(root, out, int(argv[4]))
