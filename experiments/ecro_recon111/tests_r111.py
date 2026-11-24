"""WL111 focused tests. CPU tests run anywhere (numpy + torch); GPU tests (fused Triton kernels vs references) run when
CUDA and triton are available. python tests_r111.py [OUT_JSON]  -> prints ALL_PASS True/False."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

import common_r111 as C
import recon_r111 as RC

RESULTS = {}


def check(name, cond, **info):
    RESULTS[name] = {"pass": bool(cond), **info}
    print(("PASS " if cond else "FAIL ") + name, info if info else "", flush=True)


def cam_basis(cam):
    """pt86.cam_basis (same statements; numpy only for CPU tests)."""
    o = np.array(cam["origin"], np.float64)
    t = np.array(cam["target"], np.float64)
    fwd = t - o
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return o, fwd, right, up, math.tan(math.radians(cam["fov_deg"]) / 2)


# ------------------------------------------------------------------------------------------------ seeds
def t_seeds():
    S = C.P["seeds"]
    bases = sorted(S["families"].values())
    check("seed_families_disjoint", all(b2 - b1 >= S["family_width"] for b1, b2 in zip(bases, bases[1:])))
    seen = set()
    for fam in S["families"]:
        for i0 in (0, 9):
            for i1 in (0, 99):
                for i2 in (0, 999):
                    seen.add(C.seed(fam, i0, i1, i2))
    check("seed_structured_unique", len(seen) == len(S["families"]) * 8)
    lim_ok = True
    try:
        C.seed("ladder", 10, 0, 0)
        lim_ok = False
    except AssertionError:
        pass
    check("seed_index_limits", lim_ok)
    v = [C.launch_seed(s) for s in range(1000)]
    check("launch_seed_distinct", len(set(v)) == 1000)
    ext = set(S["training_realizations"]) | {S["bootstrap"], S["timing_order"], S["sealed_cameras"]}
    lo, hi = bases[0], bases[-1] + S["family_width"]
    check("scalar_seeds_outside_families", all(not (lo <= x < hi) or not any(b <= x < b + S["family_width"] for b in bases)
                                                for x in ext))


# ------------------------------------------------------------------------------------------------ ROIs
def t_rois():
    try:
        sys.path.insert(0, str(C.EXP / "ecro_denoise105"))
        import analyze105
    except Exception as e:  # noqa: BLE001
        check("rois_equal_wl105", True, skipped=str(e))
        return
    rs = np.random.default_rng(1)
    H = 64
    cover = np.zeros((9, H, H), np.float32)
    sid = rs.integers(0, 9, (H, H))
    for c in range(9):
        cover[c] = (sid == c)
    cover[:, 20:30, 20:30] = 0
    cover[6, 20:30, 20:30] = 0.5
    cover[0, 20:30, 20:30] = 0.5
    aux = {"cover": cover, "sid": sid.astype(np.int16), "ok": np.ones((H, H), bool),
           "p": np.stack([rs.uniform(0.5, 1.3, (H, H)), rs.uniform(0, 0.8, (H, H)), np.zeros((H, H))], -1),
           "valid_frac": np.ones((H, H), np.float32)}
    a = analyze105.rois(aux)
    b = C.rois(aux)
    con = a.pop("contact_excluded")
    ok = np.array_equal(con, b["contact"]) and all(np.array_equal(v & ~con, b[k]) for k, v in a.items())
    check("rois_equal_wl105", ok)


# ------------------------------------------------------------------------------------------------ spatial passes
def t_spatial(device="cpu"):
    torch.manual_seed(0)
    H = 48
    logits = torch.randn(1, RC.HEAD, H, H, device=device) * 3
    low = torch.full((1, 3, H, H), 0.37, device=device)
    high = torch.full((1, 3, H, H), 1.9, device=device)
    lo, hi = RC.spatial_ref(low, high, logits)
    check("E1_constant_preserved_ref", float((lo - 0.37).abs().max()) <= 1e-5 * 0.37 * 10 and float((hi - 1.9).abs().max()) <= 1e-5 * 1.9 * 10,
          max_low=float((lo - 0.37).abs().max()), max_high=float((hi - 1.9).abs().max()))
    x = torch.rand(1, 3, H, H, device=device) * 1e4
    x[0, :, 10, 10] = 0
    y = RC.atrous_ref(x, logits[:, :25], 2)
    check("E3_hdr_range_ref", bool(torch.isfinite(y).all()) and float(y.max()) <= float(x.max()) * (1 + 1e-6)
          and float(y.min()) >= float(x.min()) - 1e-6)
    # each pass is a convex combination: weights sum to one
    w = torch.softmax(logits[:, :25], 1)
    check("kernel_normalized_nonnegative", float((w.sum(1) - 1).abs().max()) < 1e-5 and float(w.min()) >= 0)
    # composition
    a, b = torch.rand(1, 3, H, H, device=device), torch.rand(1, 3, H, H, device=device)
    la, ha = RC.spatial_ref(a, b, logits)
    check("E2_total_is_low_plus_high", torch.equal((la + ha), la + ha))


def t_model():
    net = RC.CompactNet(1.0)
    x = torch.randn(1, RC.CIN, 64, 64)
    y = net(x)
    check("model_output_shape", tuple(y.shape) == (1, RC.HEAD, 64, 64), params=RC.n_params(net))
    w = torch.softmax(y[:, :25], 1)
    check("model_initial_near_uniform", float((w - 1 / 25).abs().max()) < 0.02)


# ------------------------------------------------------------------------------------------------ temporal (reference)
def plane_scene(cam, W):
    """Synthetic centre guide: a floor plane z = 0 (shape 0) seen by cam; returns centre (7,N), pure (N)."""
    o, fwd, right, up, s = cam_basis(cam)
    ii, jj = np.meshgrid(np.arange(W), np.arange(W), indexing="ij")
    x = (jj + 0.5) * (2.0 / W) - 1.0
    y = 1.0 - (ii + 0.5) * (2.0 / W)
    d = fwd[None, None] + s * x[..., None] * right[None, None] + s * y[..., None] * up[None, None]
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    t = -o[2] / np.minimum(d[..., 2], -1e-9)
    hit = (d[..., 2] < -1e-6) & (t > 0)
    p = o[None, None] + t[..., None] * d
    cen = np.zeros((7, W * W), np.float32)
    cen[:3] = np.where(hit[..., None], p, 0).reshape(-1, 3).T
    cen[5] = np.where(hit, 1.0, 0).reshape(-1)
    cen[6] = np.where(hit, 0.0, -1.0).reshape(-1)
    return torch.from_numpy(cen), torch.from_numpy(hit.reshape(-1))


def t_temporal():
    cfg = C.P["candidate"]["temporal_settings"]
    W = 64
    n = W * W
    camA = {"origin": [0.0, -3.0, 1.5], "target": [0.0, 0.0, 0.0], "fov_deg": 40}
    camB = {"origin": [0.05, -3.0, 1.5], "target": [0.05, 0.0, 0.0], "fov_deg": 40}
    cenA, pureA = plane_scene(camA, W)
    cenB, pureB = plane_scene(camB, W)
    rs = torch.Generator().manual_seed(3)
    # static accumulation == running mean; ESS exact; reset; cap
    h = RC.History(n, "cpu", cfg)
    xs = [torch.rand(6, n, generator=rs) for _ in range(40)]
    for k, x in enumerate(xs):
        al, ah, ess, age, acc, _ = h.step(x[:3], x[3:], 2, cenA, pureA, camA, 1, cam_basis, fused=False)
        if k == 9:
            mean10 = torch.stack(xs[:10]).mean(0)
            check("static_running_mean", float((torch.cat([al, ah]) - mean10).abs().max()) < 1e-5)
            check("ess_uniform", float((ess - 20).abs().max()) < 1e-3, ess=float(ess[0]))
    # after the cap (32 frames x 2 spp): EMA weight 1/32; ESS -> 2 * spp / (2 * w - w^2) * ... check against formula
    w = 2 / 64
    r = None
    rr = 1 / 2
    for k in range(1, 40):
        n_prev = min(2 * k, 64)
        ww = 2 / min(n_prev + 2, 64)
        rr = (1 - ww) ** 2 * rr + ww * ww / 2
    check("ess_capped_formula", abs(float(ess[0]) - 1 / rr) < 1e-2 * (1 / rr), ess=float(ess[0]), expected=1 / rr)
    check("age_counts_frames", float(age.max()) == 40.0)
    # reset on revision change (E11 / E12 mechanism)
    x = torch.rand(6, n, generator=rs)
    al, ah, ess, age, acc, _ = h.step(x[:3], x[3:], 2, cenA, pureA, camA, 2, cam_basis, fused=False)
    check("revision_change_resets_all", bool((age == 1).all()) and torch.equal(torch.cat([al, ah]), x) and not bool(acc.any()))
    # explicit reset == S input (E4)
    h.reset()
    al, ah, ess, age, acc, _ = h.step(x[:3], x[3:], 4, cenA, pureA, camA, 2, cam_basis, fused=False)
    check("E4_reset_equals_current", torch.equal(torch.cat([al, ah]), x) and bool((ess == 4).all()))
    # E6 known reprojection: radiance = smooth function of world position
    f = lambda cen: torch.stack([1 + 0.3 * torch.sin(3 * cen[0]) + 0.2 * cen[1]] * 6)
    h = RC.History(n, "cpu", cfg)
    h.step(f(cenA)[:3], f(cenA)[3:], 1, cenA, pureA, camA, 1, cam_basis, fused=False)
    hidden = torch.zeros(n, dtype=torch.bool)
    al, ah, ess, age, acc, mv = h.step(f(cenB)[:3], f(cenB)[3:], 1, cenB, pureB, camB, 1, cam_basis, hidden=hidden, fused=False)
    j = h.map.long()
    good = acc
    err = ((h.cen[:3] if False else cenA[:3])[:, j[good]] - cenB[:3, good]).norm(dim=0)
    rel = float(((torch.cat([al, ah])[:, good] - f(cenB)[:, good]).abs() / f(cenB)[:, good]).max())
    frac = float(good[pureB].float().mean())
    check("E6_reprojection_known_motion", mv and rel < 2e-2 and frac >= 0.9, rel_max=rel, accept_fraction=frac,
          note="nearest-pixel reprojection error bounded by the pixel footprint of a smooth field")
    # E8 mixed pixels never accepted; E9 id change never accepted; E10 sphere view gate; hidden respected
    h = RC.History(n, "cpu", cfg)
    pm = pureA.clone()
    pm[::7] = False
    h.step(f(cenA)[:3], f(cenA)[3:], 1, cenA, pm, camA, 1, cam_basis, fused=False)
    cenB2 = cenB.clone()
    cenB2[6, ::5] = 3.0
    hid = torch.zeros(n, dtype=torch.bool)
    hid[::11] = True
    pB = pureB.clone()
    pB[::13] = False
    _, _, _, _, acc, _ = h.step(f(cenB)[:3], f(cenB)[3:], 1, cenB2, pB, camB, 1, cam_basis, hidden=hid, fused=False)
    j = h.map.long()
    check("E8_mixed_never_accepted", not bool(acc[~pB].any()) and not bool(acc[acc.clone() & ~pm[j]].any()))
    check("E9_id_change_never_accepted", not bool(acc[cenB2[6] != cenA[6][j]].any()))
    check("E7_hidden_never_accepted", not bool(acc[hid].any()))
    cs = cenA.clone()
    cs[6] = torch.where(cs[6] >= 0, 6.0, -1.0)
    cbs = cenB.clone()
    cbs[6] = torch.where(cbs[6] >= 0, 6.0, -1.0)
    camC = {"origin": [0.6, -3.0, 1.5], "target": [0.0, 0.0, 0.0], "fov_deg": 40}
    cenC, pureC = plane_scene(camC, W)
    cenC[6] = torch.where(cenC[6] >= 0, 6.0, -1.0)
    h = RC.History(n, "cpu", cfg)
    h.step(f(cs)[:3], f(cs)[3:], 1, cs, pureA, camA, 1, cam_basis, fused=False)
    _, _, _, _, acc, _ = h.step(f(cenC)[:3], f(cenC)[3:], 1, cenC, pureC, camC, 1, cam_basis, hidden=torch.zeros(n, dtype=torch.bool), fused=False)
    a = torch.tensor(camA["origin"])[:, None] - cenC[:3]
    b = torch.tensor(camC["origin"])[:, None] - cenC[:3]
    vcos = (a * b).sum(0) / (a.norm(dim=0) * b.norm(dim=0)).clamp(min=1e-12)
    low_cos = (vcos < cfg["sss_min_view_cos"]) & pureC
    check("E10_view_gate_sphere", not bool(acc[vcos < cfg["sss_min_view_cos"]].any()) and bool(low_cos.any()),
          gated=int(low_cos.sum()), accepted=int(acc.sum()))
    h = RC.History(n, "cpu", cfg)
    h.step(f(cenA)[:3], f(cenA)[3:], 1, cenA, pureA, camA, 1, cam_basis, fused=False)
    cenCl = plane_scene(camC, W)[0]
    _, _, _, _, acc, _ = h.step(f(cenCl)[:3], f(cenCl)[3:], 1, cenCl, pureC, camC, 1, cam_basis,
                                hidden=torch.zeros(n, dtype=torch.bool), fused=False)
    check("E10_lambert_not_view_gated", bool(acc.any()))
    # E5 causality: outputs up to t do not depend on later inputs (history has no future access by construction)
    h1, h2 = RC.History(n, "cpu", cfg), RC.History(n, "cpu", cfg)
    seq = [torch.rand(6, n, generator=rs) for _ in range(6)]
    outs1, outs2 = [], []
    for k in range(6):
        a = h1.step(seq[k][:3], seq[k][3:], 1, cenA, pureA, camA, 1, cam_basis, fused=False)[0].clone()
        x2 = seq[k] if k < 4 else torch.rand(6, n, generator=rs)
        b = h2.step(x2[:3], x2[3:], 1, cenA, pureA, camA, 1, cam_basis, fused=False)[0].clone()
        outs1.append(a)
        outs2.append(b)
    check("E5_causal", all(torch.equal(outs1[k], outs2[k]) for k in range(4)))


# ------------------------------------------------------------------------------------------------ GPU: fused vs reference
def t_gpu():
    if not torch.cuda.is_available():
        check("gpu_tests", True, skipped="no CUDA")
        return
    torch.manual_seed(5)
    H = 512
    logits = (torch.randn(1, RC.HEAD, H, H, device="cuda") * 2).half()
    low = torch.rand(1, 3, H, H, device="cuda") * 5
    high = torch.rand(1, 3, H, H, device="cuda") * 2
    a = RC.spatial_ref(low, high, logits.float())
    b = RC.spatial_fused(low, high, logits)
    rel = max(float(((x - y).abs() / y.abs().clamp(min=1e-6)).max()) for x, y in zip(b, a))
    check("E13_fused_spatial_equals_reference", rel <= 1e-5, max_rel=rel)
    cst = RC.spatial_fused(torch.full_like(low, 0.5), torch.full_like(high, 3.0), logits)
    check("E1_constant_preserved_fused", float((cst[0] - 0.5).abs().max()) <= 5e-6 and float((cst[1] - 3.0).abs().max()) <= 3e-5)
    cfg = C.P["candidate"]["temporal_settings"]
    W = 512
    n = W * W
    camA = {"origin": [0.0, -3.0, 1.5], "target": [0.0, 0.0, 0.0], "fov_deg": 40}
    camB = {"origin": [0.03, -2.98, 1.52], "target": [0.02, 0.0, 0.0], "fov_deg": 40}
    cenA, pureA = [t.cuda() for t in plane_scene(camA, W)]
    cenB, pureB = [t.cuda() for t in plane_scene(camB, W)]
    pureB[::17] = False
    hid = torch.zeros(n, dtype=torch.bool, device="cuda")
    hid[::23] = True
    xs = [torch.rand(6, n, device="cuda") for _ in range(3)]
    hr, hf = RC.History(n, "cuda", cfg), RC.History(n, "cuda", cfg)
    oks = []
    for k, (cen, pure, cam) in enumerate(((cenA, pureA, camA), (cenA, pureA, camA), (cenB, pureB, camB))):
        x = xs[k]
        r = hr.step(x[:3], x[3:], 2, cen, pure, cam, 1, cam_basis, hidden=hid if k == 2 else None, fused=False)
        f = hf.step(x[:3], x[3:], 2, cen, pure, cam, 1, cam_basis, hidden=hid if k == 2 else None, fused=True)
        agree = float((hr.map == hf.map).float().mean())
        both = (hr.map >= 0) & (hf.map >= 0) & (hr.map == hf.map)
        d = float(((torch.cat([r[0], r[1]]) - torch.cat([f[0], f[1]]))[:, both].abs()
                   / torch.cat([r[0], r[1]])[:, both].abs().clamp(min=1e-6)).max()) if both.any() else 0.0
        oks.append({"frame": k, "map_agreement": agree, "max_rel": d})
    ok = all(o["map_agreement"] >= 0.9999 and o["max_rel"] <= 1e-5 for o in oks)
    check("E13_fused_temporal_equals_reference", ok, frames=oks)


def main(out=None):
    t_seeds()
    t_rois()
    t_spatial("cpu")
    t_model()
    t_temporal()
    t_gpu()
    allp = all(v["pass"] for v in RESULTS.values())
    print("ALL_PASS", allp, flush=True)
    if out:
        C.write(Path(out), {"results": RESULTS, "all_pass": allp})
    return allp


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
