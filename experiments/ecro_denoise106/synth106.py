"""WL106 Part B: synthetic boundary contracts with known targets.

Each scene is a 2-D arrangement of surfaces A (mapped to shape channel 6, the sphere), B (channel 2, back wall),
C (channel 0, floor) or empty (miss). The guide is the renderer's: coverage on the fixed 8 x 8 midpoint subpixel grid
(render105.aux_buffers semantics; the frame program samples uniformly in the same pixel square, pt86.gen_rays).
Targets:
  T8  = mean of the radiance at the 8 x 8 guide points (guide-consistent: for per-surface constants exactly
        sum_s c8_s L_s; isolates the operator's support / application from guide quantization),
  Tex = mean over a 32 x 32 midpoint grid (the expected pixel value of uniform pixel sampling, to < 1e-3).
Kernels (logits per pixel; identical for both operators): K1 box, K2 Gaussian sigma 2.5 px, K3 random N(0, 2^2)
(fixed seed), K4 the frozen WL105 R3 LOW network on Monte-Carlo noisy inputs (out-of-distribution, report only).
Operators: U = unconstrained softmax application (WL105), C = coverage-moment-constrained (tilt106).

python synth106.py OUT [R3_WEIGHTS]
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CH = {"A": 6, "B": 2, "C": 0}
BINS = [0.0, 1 / 64, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 63 / 64, 1.0 + 1e-9]
GATED = ("disk", "disk_empty", "edge17", "corner3")          # generic configurations (synthetic gate)
REPORT_ONLY = ("edge0", "strip_thin", "strip_wide")           # grid-aligned / sub-pixel features (identifiability)


def membership(name, x, y, n):
    """Surface id per point: 0 = A, 1 = B, 2 = C, -1 = empty. x right, y down, pixel units."""
    c = n / 2
    if name == "edge0":
        return np.where(x < c + 0.3, 0, 1)
    if name == "edge17":
        a = math.radians(17)
        return np.where((x - c) * math.cos(a) + (y - c) * math.sin(a) < 0.137, 0, 1)
    if name in ("disk", "disk_empty"):
        inside = (x - (c + 0.21)) ** 2 + (y - (c - 0.37)) ** 2 < (0.3 * n) ** 2
        return np.where(inside, 0, 1 if name == "disk" else -1)
    if name in ("strip_thin", "strip_wide"):
        a = math.radians(8)
        w = 0.6 if name == "strip_thin" else 2.5
        d = -(x - c) * math.sin(a) + (y - c) * math.cos(a)
        return np.where(np.abs(d - 0.11) < w / 2, 0, 1)
    if name == "corner3":
        cx, cy = c + 0.43, c - 0.29
        a = math.radians(11)
        u = (x - cx) * math.cos(a) + (y - cy) * math.sin(a)
        v = -(x - cx) * math.sin(a) + (y - cy) * math.cos(a)
        return np.where(u < 0, 0, np.where(v < 0, 1, 2))
    raise ValueError(name)


def radiance(field, sid, x, y, name, n):
    """Per-surface radiance. const: A 3.0, B 0.75, C 1.5; const_inv: A 0.75, B 3.0, C 1.5;
    smooth: low-frequency variation inside each surface; rim: A darkens steeply within 6 px of its boundary."""
    base = {"const": (3.0, 0.75, 1.5), "const_inv": (0.75, 3.0, 1.5), "smooth": (3.0, 0.75, 1.5), "rim": (3.0, 0.75, 1.5)}[field]
    L = np.zeros_like(x, dtype=np.float64)
    for s in range(3):
        L = np.where(sid == s, base[s], L)
    if field == "smooth":
        L = L * np.where(sid == 0, 1 + 0.3 * np.sin(2 * np.pi * x / 40) * np.cos(2 * np.pi * y / 53),
                         1 + 0.5 * (x / n - 0.5))
    if field == "rim":
        dist = boundary_distance(name, x, y, n)
        L = np.where(sid == 0, L * (0.5 + 0.5 * np.minimum(1.0, dist / 6.0)), L)
    return np.where(sid < 0, 0.0, L)


def boundary_distance(name, x, y, n):
    c = n / 2
    if name in ("disk", "disk_empty"):
        return np.abs(np.hypot(x - (c + 0.21), y - (c - 0.37)) - 0.3 * n)
    if name == "edge0":
        return np.abs(x - (c + 0.3))
    if name == "edge17":
        a = math.radians(17)
        return np.abs((x - c) * math.cos(a) + (y - c) * math.sin(a) - 0.137)
    if name in ("strip_thin", "strip_wide"):
        a = math.radians(8)
        w = 0.6 if name == "strip_thin" else 2.5
        return np.abs(np.abs(-(x - c) * math.sin(a) + (y - c) * math.cos(a) - 0.11) - w / 2)
    cx, cy = c + 0.43, c - 0.29
    a = math.radians(11)
    u = (x - cx) * math.cos(a) + (y - cy) * math.sin(a)
    v = -(x - cx) * math.sin(a) + (y - cy) * math.cos(a)
    return np.where(u < 0, np.abs(u), np.minimum(np.abs(u), np.abs(v)))


def grid(n, sub):
    o = (np.arange(sub) + 0.5) / sub
    py, px = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    x = px[:, :, None, None] + o[None, None, None, :]
    y = py[:, :, None, None] + o[None, None, :, None]
    return np.broadcast_to(x, (n, n, sub, sub)), np.broadcast_to(y, (n, n, sub, sub))


FIELDS = ("const", "const_inv", "smooth", "rim")


def scenes(n=128):
    out = {}
    for name in GATED + REPORT_ONLY:
        rec = {}
        for sub, key in ((8, "8"), (32, "ex")):
            x, y = grid(n, sub)
            sid = membership(name, x, y, n)
            cover = np.zeros((9, n, n), np.float32)
            for s, ch in enumerate(("A", "B", "C")):
                cover[CH[ch]] = (sid == s).mean((2, 3))
            rec["cover" + key] = cover
            for f in FIELDS:
                rec.setdefault("T" + key, {})[f] = radiance(f, sid, x, y, name, n).mean((2, 3)).astype(np.float64)
        out[name] = rec
    return out


def kernels(n, kind, seed=106001):
    K = 15
    r = K // 2
    if kind == "K1":
        return np.zeros((K * K, n, n), np.float32)
    if kind == "K2":
        dy, dx = np.meshgrid(np.arange(K) - r, np.arange(K) - r, indexing="ij")
        l = -(dy ** 2 + dx ** 2) / (2 * 2.5 ** 2)
        return np.broadcast_to(l.reshape(K * K, 1, 1), (K * K, n, n)).astype(np.float32).copy()
    rs = np.random.default_rng(seed)
    return (rs.standard_normal((K * K, n, n)) * 2).astype(np.float32)


def noisy(name, field, n, spp, rs):
    """Uniform pixel sampling, per-sample exponential noise factor (mean 1): one Monte-Carlo frame (mean of spp)."""
    py, px = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    acc = np.zeros((n, n))
    for _ in range(spp):
        x = px + rs.random((n, n))
        y = py + rs.random((n, n))
        sid = membership(name, x, y, n)
        acc += radiance(field, sid, x, y, name, n) * rs.exponential(1.0, (n, n))
    return acc / spp


def metrics(E, Tt, cov8, c10_np):
    """E expected output, Tt target (H,W); partial = pixels covered by >= 2 of (A, B, C, empty)."""
    partial = (c10_np > 0).sum(0) >= 2
    near = np.zeros_like(partial)
    H, W = partial.shape
    ys, xs = np.nonzero(partial)
    for dy in range(-7, 8):
        for dx in range(-7, 8):
            yy, xx = np.clip(ys + dy, 0, H - 1), np.clip(xs + dx, 0, W - 1)
            near[yy, xx] = True
    full_near = near & ~partial & (Tt > 0)
    rel = np.where(Tt > 0, (E - Tt) / np.maximum(Tt, 1e-12), 0.0)
    a = cov8[6]
    bins = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = partial & (a >= lo) & (a < hi)
        bins.append(None if m.sum() < 20 else float((E - Tt)[m].sum() / Tt[m].sum()))
    return {"band_signed_energy": float((E - Tt)[partial].sum() / Tt[partial].sum()),
            "band_max_abs_rel": float(np.abs(rel[partial]).max()),
            "near_full_signed_energy": float((E - Tt)[full_near].sum() / Tt[full_near].sum()),
            "near_full_max_abs_rel": float(np.abs(rel[full_near]).max()),
            "n_partial": int(partial.sum()), "coverage_bins_signed": bins}


def main(out, r3_weights=None):
    import torch
    sys.path.insert(0, str(HERE))
    import model106 as M6
    import tilt106 as T
    M = M6.M
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    n = 128
    sc = scenes(n)
    res = {"n": n, "gated": GATED, "report_only": REPORT_ONLY, "fields": FIELDS, "rows": []}
    const_err = {}

    def apply(img, logits, cov, constrained):
        with torch.no_grad():
            if constrained:
                return M6.constrained_apply(img, logits, cov)
            M.FUSED = False
            return M.apply_kernel(img, logits)

    for name, rec in sc.items():
        cov = torch.from_numpy(rec["cover8"])[None].to(dev)
        c10 = T.coverage10(cov)[0].cpu().numpy()
        st = {}
        for kind in ("K1", "K2", "K3"):
            lg = torch.from_numpy(kernels(n, kind))[None].to(dev)
            with torch.no_grad():
                zc = T.constrain_logits(lg, cov, stats=st)
            # mixing: coverage-moment mismatch of the applied kernels
            mix = {}
            c12 = torch.from_numpy(np.concatenate([c10, np.zeros((2, n, n), np.float32)]))[None].to(dev)
            part = (c10 > 0).sum(0) >= 2
            for op, z in (("U", lg), ("C", zc)):
                mom = torch.cat([apply(c12[:, j:j + 3].contiguous(), z, cov, False) for j in (0, 3, 6, 9)], 1)[0, :10]
                mm = np.abs(mom.cpu().numpy() - c10).sum(0)
                mix[op] = {"band_mean_L1": float(mm[part].mean()), "all_max_L1": float(mm.max())}
            one = torch.full((1, 3, n, n), 0.37, device=dev)
            for op in ("U", "C"):
                const_err[f"{name}_{kind}_{op}"] = float((apply(one, lg, cov, op == "C") - 0.37).abs().max())
            for f in FIELDS:
                for tkey in ("T8", "Tex"):
                    Tt = rec[tkey][f]
                    img = torch.from_numpy(np.repeat(Tt[None], 3, 0).astype(np.float32))[None].to(dev)
                    for op in ("U", "C"):
                        E = apply(img, lg, cov, op == "C")[0, 0].double().cpu().numpy()
                        res["rows"].append({"scene": name, "kernel": kind, "field": f, "truth": tkey, "op": op,
                                            "mixing": mix[op], **metrics(E, Tt, rec["cover8"], c10)})
        res.setdefault("constraint_stats", {})[name] = {"active_fraction": st["active_fraction"], "tilted_fraction": st["tilted_fraction"],
                                                         "max_own_surfaces": st["max_own_surfaces"], "residual_max": float(st["residual"].max())}
        print("synthetic", name, flush=True)
    res["constant_field_max_abs_error"] = const_err
    # K4: frozen R3 LOW network on Monte-Carlo noisy inputs (out of distribution; report only).
    if r3_weights:
        r3 = M.Reconstructor("R3").to(dev)
        r3.load_state_dict(torch.load(r3_weights, map_location=dev))
        r3.eval()
        rs = np.random.default_rng(106002)
        reps = 64
        for name in ("disk", "edge17", "corner3"):
            rec = sc[name]
            cov = torch.from_numpy(rec["cover8"])[None].to(dev)
            nrm = np.zeros((3, n, n), np.float32)
            nrm[2] = 1.0
            dep = np.full((1, n, n), 0.4, np.float32)
            for spp in (4, 32):
                aux = torch.from_numpy(np.concatenate([rec["cover8"], nrm, dep, np.full((1, n, n), np.log2(spp) / 5, np.float32)]))[None].to(dev)
                for f in ("const", "rim"):
                    acc = {"U": 0.0, "C": 0.0}
                    sq = {"U": 0.0, "C": 0.0}
                    mixacc = {"U": 0.0, "C": 0.0}
                    Tt = rec["Tex"][f]
                    for _ in range(reps):
                        x = noisy(name, f, n, spp, rs)
                        img = torch.from_numpy(np.repeat(x[None], 3, 0).astype(np.float32))[None].to(dev)
                        with torch.no_grad():
                            lg = r3.net_low(torch.cat([torch.log1p(img), aux], 1))
                        timg = torch.from_numpy(np.repeat(Tt[None], 3, 0).astype(np.float32))[None].to(dev)
                        for op in ("U", "C"):
                            y = apply(img, lg, cov, op == "C")[0, 0].double().cpu().numpy()
                            acc[op] += y / reps
                            sq[op] += y ** 2 / reps
                            mixacc[op] += apply(timg, lg, cov, op == "C")[0, 0].double().cpu().numpy() / reps
                    c10 = T.coverage10(cov)[0].cpu().numpy()
                    for op in ("U", "C"):
                        mc_se = np.sqrt(np.maximum(sq[op] - acc[op] ** 2, 0) / reps)
                        part = (c10 > 0).sum(0) >= 2
                        res["rows"].append({"scene": name, "kernel": f"K4_spp{spp}", "field": f, "truth": "Tex", "op": op,
                                            "mc_se_band_energy": float(np.sqrt((mc_se[part] ** 2).sum()) / Tt[part].sum()),
                                            "mixing_term": metrics(mixacc[op], Tt, rec["cover8"], c10)["band_signed_energy"],
                                            **metrics(acc[op], Tt, rec["cover8"], c10)})
                print("K4", name, spp, flush=True)
    (out / "synthetic106.json").write_text(json.dumps(res, indent=1))
    gate(res, out)


def gate(res, out):
    """Predeclared synthetic gate (protocol denoise106.json, synthetic_gate)."""
    rows = res["rows"]
    g = {}
    g["SG1_constant_exact"] = max(res["constant_field_max_abs_error"].values()) <= 1e-5
    sg2 = [r for r in rows if r["truth"] == "T8" and r["op"] == "C" and r["field"] in ("const", "const_inv") and r["kernel"] in ("K1", "K2", "K3")]
    g["SG2_guide_consistent_exact"] = max(max(r["band_max_abs_rel"], r["near_full_max_abs_rel"]) for r in sg2) <= 1e-3
    sg3 = [r for r in rows if r["truth"] == "Tex" and r["field"] in ("const", "const_inv") and r["kernel"] in ("K1", "K2", "K3") and r["scene"] in GATED]
    ok3 = True
    for r in sg3:
        if r["op"] == "C":
            u = next(q for q in sg3 if q["op"] == "U" and all(q[k] == r[k] for k in ("scene", "kernel", "field")))
            ok3 &= abs(r["band_signed_energy"]) <= 0.01 and abs(r["near_full_signed_energy"]) <= 0.01
            ok3 &= abs(r["band_signed_energy"]) <= abs(u["band_signed_energy"])
    g["SG3_exact_area_generic"] = bool(ok3)
    g["pass"] = bool(g["SG1_constant_exact"] and g["SG2_guide_consistent_exact"] and g["SG3_exact_area_generic"])
    res["gate"] = g
    (out / "synthetic106.json").write_text(json.dumps(res, indent=1))
    print("SYNTHETIC GATE", json.dumps(g), flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve(), sys.argv[2] if len(sys.argv) > 2 else None)
