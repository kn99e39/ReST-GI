"""WL111 Stage A: baseline recovery (G0), program equivalence, cost attribution and the predeclared design rule.

python profile_r111.py OUT [--smoke]
Writes OUT/stageA.json (gates, measurements) and OUT/design.json (sub, m, cost futility decision).
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

import common_r111 as C
import engine_r111 as E
import recon_r111 as RC
import r3_r111
import arms_r111 as AR

torch, R, dr, mi = E.torch, E.R, E.dr, E.mi
W = C.RES
N = W * W
LOG = None


def log(*a):
    s = time.strftime("%Y-%m-%d %H:%M:%S ") + " ".join(str(x) for x in a)
    print(s, flush=True)
    if LOG:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(s + "\n")


def rel_l1(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.abs(a - b).sum() / max(np.abs(b).sum(), 1e-30))


def cams105():
    import seq_r111
    return seq_r111.wl105_cameras()


# ------------------------------------------------------------------------------------------------ G0d / G0e / G0f
def g0d_r3(model, smoke):
    raw = C.read(C.RAW105)["raw_files"]
    sys.path.insert(0, str(C.EXP / "ecro_denoise105"))
    import data105
    rows, worst = [], 0.0
    views = C.P["views"]["historical"][: 1 if smoke else 6]
    for v in views:
        d = C.D105 / "eval" / v
        assert C.sha(d / "aux.npz") == raw[f"data/eval/{v}/aux.npz"]
        for spp in (4, 16, 32):
            aux = torch.from_numpy(data105.aux_tensor(d / "aux.npz", spp))[None].cuda()
            for r in range(2 if smoke else 8):
                p = d / f"noisy_spp{spp}_r{r}.npz"
                assert C.sha(p) == raw[f"data/eval/{v}/noisy_spp{spp}_r{r}.npz"]
                comp = torch.from_numpy(np.load(p)["comp"])[None].cuda()
                y = r3_r111.forward(model, comp, aux)[0].cpu().numpy()
                ref = np.load(C.E105 / f"{v}_spp{spp}_r{r}.npz")["R3"]
                e = rel_l1(y, ref)
                worst = max(worst, e)
                rows.append({"view": v, "spp": spp, "r": r, "rel_l1": e})
    G = C.P["gates"]["G0"]
    grade = "EXACT" if worst <= G["r3_replay_relative_l1_exact"] else ("HOST_NUMERIC" if worst <= G["r3_replay_relative_l1_host"] else "FAIL")
    return {"rows": rows, "max_rel_l1": worst, "grade": grade}


def reference107(view):
    """runtime109.reference statements: WL107 48 x 8192-spp frozen-ECRO reference (mean comps, variance of the mean)."""
    m = C.REF107 / "reference107.json"
    rec = C.read(m)["views"][view]
    target = C.REF107 / f"{view}_target.npy"
    assert C.sha(target) == rec["target_sha256"]
    cap = C.read(C.CAP107 / "capture_manifest.json")["files"]
    raw = C.read(C.RAW105)["raw_files"]
    blocks = []
    for h in range(2):
        p = C.D105 / "eval" / view / f"target_half{h}.npz"
        assert C.sha(p) == raw[f"data/eval/{view}/target_half{h}.npz"]
        blocks.append(np.load(p)["comp"].astype(np.float64))
    for h in range(46):
        key = f"{view}/assessment_{h:02d}.npy"
        p = C.CAP107 / key
        assert C.sha(p) == cap[key]["sha256"], key
        blocks.append(np.load(p, allow_pickle=False).astype(np.float64))
    comp = np.stack(blocks)
    mean = comp.mean(0)
    rgb = comp[:, 0:3] + comp[:, 3:6] + comp[:, 6:9]
    var = rgb.var(0, ddof=1) / comp.shape[0]
    stored = np.load(target).astype(np.float64)
    assert np.allclose(stored, mean, rtol=2e-6, atol=1e-7)
    return mean, var


def g0e_quality(model, smoke):
    """R3 primary / silhouette RMSE against the WL107 references from the archived inputs (reproduces WL108's R3 row)."""
    sys.path.insert(0, str(C.EXP / "ecro_denoise105"))
    import data105
    out = {}
    for v in C.P["views"]["historical"][: 1 if smoke else 6]:
        mean, var = reference107(v)
        ref = mean[0:3] + mean[3:6] + mean[6:9]
        with np.load(C.D105 / "eval" / v / "aux.npz") as z:
            aux = {k: z[k] for k in z.files}
        masks = C.rois(aux)
        for spp in (4, 16, 32):
            auxt = torch.from_numpy(data105.aux_tensor(C.D105 / "eval" / v / "aux.npz", spp))[None].cuda()
            vals = {k: [] for k in ("primary", "silhouette")}
            for r in range(2 if smoke else 8):
                comp = torch.from_numpy(np.load(C.D105 / "eval" / v / f"noisy_spp{spp}_r{r}.npz")["comp"])[None].cuda()
                y = r3_r111.forward(model, comp, auxt)[0].double().cpu().numpy()
                for k in vals:
                    d = (y - ref)[:, masks[k]]
                    vals[k].append(math.sqrt((d ** 2).mean() / (ref[:, masks[k]] ** 2).mean()))
            out[f"{v}_{spp}"] = {k: float(np.mean(x)) for k, x in vals.items()}
    pooled = {spp: {k: float(np.mean([out[f"{v}_{spp}"][k] for v in C.P["views"]["historical"][: 1 if smoke else 6]]))
                    for k in ("primary", "silhouette")} for spp in (4, 16, 32)}
    return {"per_view": out, "pooled_mean_per_draw": pooled,
            "wl108_reference_values": {"primary": [0.01470, 0.00937, 0.00775], "silhouette": [0.0778, 0.0596, 0.0500]}}


def g0f_svgf(smoke):
    """Replay the frozen WL101B SVGF on its archived static sequences (raw frames + archived centre guide)."""
    base = C.INPUTS / "wl101b_primary"
    if not base.exists():
        return {"status": "NOT_AVAILABLE", "reason": "archived WL101B frames not present"}
    import svgf as SV
    import json as _json
    S = AR.SVGF_SETTINGS
    with np.load(base / "guide_0_A.npz") as z:
        p, n, sid = z["p"], z["n"], z["sid"]
    guide = torch.from_numpy(np.concatenate([p.reshape(N, 3).T, n.reshape(N, 3).T, sid.reshape(1, N).astype(np.float32)])).float().cuda()
    cam = dict(C.P["views"]["V0"], res=W)
    worst, rows = 0.0, []
    for spp in (1, 4):
        for rep in range(1 if smoke else 4):
            den = SV.SVGF(W, "cuda", R.PT.cam_basis, S)
            for f in range(24):
                with np.load(base / f"static_{spp}spp_r{rep}_f{f:02d}.npz") as z:
                    raw = torch.from_numpy(z["raw"].reshape(N, 3).T.copy()).float().cuda()
                    ref = z["reconstructed"]
                out, _, _ = den.apply(raw, guide, cam, 1)
                y = out.detach().cpu().numpy().T.reshape(W, W, 3)
                e = rel_l1(y, ref)
                worst = max(worst, e)
                rows.append({"spp": spp, "rep": rep, "frame": f, "rel_l1": e})
    ok = worst <= C.P["gates"]["G0"]["svgf_replay_relative_l1"]
    return {"status": "REPRODUCED" if ok else "NOT_REPRODUCED", "max_rel_l1": worst, "rows": rows}


# ------------------------------------------------------------------------------------------------ program equivalence
def equivalence(eng, smoke):
    """Camera-input programs vs the originals, cycling the camera V0 -> H1 -> H2 -> H3 -> V0 so that every replay of a
    frozen program sees a camera different from the one it was recorded with (a stale-input replay fails here)."""
    out = {}
    rt = eng.rt
    cams = cams105()
    order = ["V0", "H1", "H2", "H3", "V0"]
    for step, v in enumerate(order):
        cam = E.cam_dict(cams[v])
        for spp in (1, 4):
            rt._frames = {}
            a = rt.frame(eng.Bd, cam, W, spp, 12345 + step)                   # original program: seed * 1000003
            a = torch.stack([o.torch() for o in a]).clone()
            b = eng.frame_raw(cam, spp, (12345 + step) * 1000003)
            b = torch.stack([o.torch() for o in b]).clone()
            out[f"frame_{step}_{v}_{spp}"] = {"bitwise": bool(torch.equal(a, b)), "max_abs": float((a - b).abs().max())}
        rt._frames = {}
        g = eng.guides(cam, 8)
        with np.load(C.D105 / "eval" / v / "aux.npz") as z:
            cover, nrm, dep, vf = z["cover"], z["normal"], z["depth"], z["valid_frac"]
        gc = g["cover"].cpu().numpy()
        out[f"guide_{step}_{v}"] = {"cover_exact_fraction": float((np.abs(gc - cover) < 1e-7).all(0).mean()),
                                    "cover_max_abs": float(np.abs(gc - cover).max()),
                                    "normal_max_abs": float(np.abs(g["normal"].cpu().numpy() - nrm).max()),
                                    "depth_max_abs": float(np.abs(g["depth"].cpu().numpy() - dep).max()),
                                    "valid_max_abs": float(np.abs(g["valid_frac"].cpu().numpy() - vf).max())}
        prev = E.cam_dict(cams[order[step - 1]]) if step else None
        p = eng.pack(cam, 4, prev, True)
        gb = rt.gbuffer(dict(cam, res=W), W)
        sid = np.where(gb["ok"], gb["sid"], -1).reshape(-1)
        cen = p["centre"].cpu().numpy()
        out[f"pack_centre_{step}_{v}"] = {"sid_mismatch": float(np.mean(cen[6].astype(np.int64) != sid))}
        if prev is not None:
            h_ref = eng.hidden_from(prev, p["centre"][:3])
            out[f"pack_hidden_{step}_{v}"] = {"equal_separate_query": bool(torch.equal(h_ref, p["hidden"]))}
    base = C.INPUTS / "wl101b_primary"
    if base.exists():
        cen = eng.centre(dict(C.P["views"]["V0"], res=W)).cpu().numpy()
        with np.load(base / "guide_0_A.npz") as z:
            sid, pp, n = z["sid"].reshape(N), z["p"].reshape(N, 3).T, z["n"].reshape(N, 3).T
        mask = sid >= 0
        out["centre_V0"] = {"sid_mismatch": float(np.mean(cen[6].astype(np.int16) != sid)),
                            "p_rms": float(np.sqrt(((cen[:3] - pp)[:, mask] ** 2).mean())),
                            "n_rms": float(np.sqrt(((cen[3:6] - n)[:, mask] ** 2).mean()))}
    ok = all(v["bitwise"] for k, v in out.items() if k.startswith("frame_"))
    ok &= all(v["cover_exact_fraction"] >= 0.9999 and v["normal_max_abs"] <= 1e-4 and v["depth_max_abs"] <= 1e-4
              for k, v in out.items() if k.startswith("guide_"))
    ok &= all(v["sid_mismatch"] <= 1e-4 for k, v in out.items() if k.startswith("pack_centre_"))
    ok &= all(v["equal_separate_query"] for k, v in out.items() if k.startswith("pack_hidden_"))
    if "centre_V0" in out:
        c = out["centre_V0"]
        ok &= c["sid_mismatch"] <= 1e-4 and c["p_rms"] <= 1e-4 and c["n_rms"] <= 1e-4
    out["pass"] = bool(ok)
    return out


# ------------------------------------------------------------------------------------------------ timing helpers
def tms(fn, warm, rounds):
    for _ in range(warm):
        fn()
    E.sync()
    v = []
    for _ in range(rounds):
        E.sync()
        t0 = time.perf_counter()
        fn()
        E.sync()
        v.append(1e3 * (time.perf_counter() - t0))
    return v


def st(v):
    v = np.asarray(v, np.float64)
    return {"median": float(np.median(v)), "p95": float(np.percentile(v, 95)), "mean": float(v.mean()), "n": int(v.size)}


def conv_flops(model, x):
    """Analytic multiply-accumulates and output bytes of every Conv2d for one forward of input x."""
    macs, outb = [0], [0]

    def hook(m, i, o):
        k = m.kernel_size[0] * m.kernel_size[1]
        macs[0] += o.numel() * (m.in_channels // m.groups) * k
        outb[0] += o.numel() * o.element_size()
    hs = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
    with torch.no_grad():
        model(x)
    for h in hs:
        h.remove()
    return macs[0], outb[0]


def bandwidth():
    a = torch.empty(64 * 1024 * 1024, device="cuda")      # 256 MB
    b = torch.empty_like(a)
    v = tms(lambda: b.copy_(a), 3, 10)
    return 2 * a.numel() * 4 / (np.median(v) * 1e-3) / 1e9


def timing(eng, r3, smoke):
    P = C.P["stageA"]
    warm, rounds = (1, 3) if smoke else (P["warmup"], P["rounds"])
    cams = cams105()
    res = {}
    seed = lambda slot, i: C.seed("stageA", 0, slot, i, smoke=smoke)
    v0 = E.cam_dict(cams["V0"])
    camb = dict(v0, origin=[v0["origin"][0] + 0.08] + v0["origin"][1:])
    # ECRO frames
    fr = {}
    for vi, v in enumerate(P["views"]):
        cam = E.cam_dict(cams[v])
        for spp in P["ecro_frame_spp"]:
            k = [0]

            def f():
                k[0] += 1
                eng.frame_raw(cam, spp, C.launch_seed(seed(vi, k[0] % 1000)))
            fr[f"{v}_{spp}"] = st(tms(f, warm, rounds))
    res["ecro_frame_ms"] = fr
    rt = eng.rt
    for spp in (1, 4):
        rt._frames = {}
        res[f"ecro_frame_original_V0_{spp}"] = st(tms(lambda: dr.eval(rt.frame(eng.Bd, v0, W, spp, 777)), warm, rounds))
    rt._frames = {}
    # guides
    gt = {}
    for sub in (8, 4, 2):
        gt[f"sub{sub}"] = st(tms(lambda: eng.guides(v0, sub), warm, rounds))
    gt["centre"] = st(tms(lambda: eng.centre(v0), warm, rounds))
    cen = eng.centre(v0)
    gt["visibility"] = st(tms(lambda: eng.hidden_from(camb, cen[:3]), warm, rounds))
    for sub in (4, 8):
        gt[f"pack_sub{sub}_camera"] = st(tms(lambda: eng.pack(v0, sub, camb, True), warm, rounds))
    res["guide_ms"] = gt
    # R3 staged
    r3t = {}
    g8 = eng.guides(v0, 8)
    for spp in C.P["stageA"]["r3_spp"]:
        comp = eng.frame(v0, spp, seed(9, spp))[None]
        aux = r3_r111.aux14(g8, spp)
        for label, kw in (("fp32_fused", {}), ("fp16_fused_diag", {"fp16": True}), ("fp32_unfold_diag", {"fused": False})):
            sg = r3_r111.Staged(r3, **kw)
            torch.cuda.reset_peak_memory_stats()
            before = torch.cuda.memory_allocated()
            stages = []
            for _ in range(warm):
                sg(comp, aux)
            for _ in range(rounds):
                _, s_, extra = sg(comp, aux)
                stages.append(s_)
            peak = torch.cuda.max_memory_allocated() - before
            tot = tms(lambda: r3_r111.forward(r3, comp, aux) if label == "fp32_fused" else sg(comp, aux), warm, rounds)
            r3t[f"{label}_{spp}"] = {"stages_median_ms": {k: float(np.median([s[k] for s in stages])) for k in stages[0]},
                                     "total_wall": st(tot), "peak_extra_bytes": int(peak), **extra}
            if smoke and label != "fp32_fused":
                break
    # R3 as one CUDA graph (common execution contract): equality with the eager forward and its time
    comp = eng.frame(v0, 4, seed(9, 4))[None]
    aux = r3_r111.aux14(g8, 4)
    r3g = AR.R3Graph(r3)
    yg = r3g(comp, aux).clone()
    ye = r3_r111.forward(r3, comp, aux)
    r3t["graph_4"] = {"bitwise_equal_eager": bool(torch.equal(yg, ye)), "rel_l1_vs_eager": float((yg - ye).abs().sum() / ye.abs().sum()),
                      "total_wall": st(tms(lambda: r3g(comp, aux), warm, rounds))}
    # analytic costs
    x = torch.zeros(1, 17, W, W, device="cuda")
    ml, bl = conv_flops(r3.net_low, x)
    mh, bh = conv_flops(r3.net_high, x)
    r3t["analytic"] = {"net_low_macs": ml, "net_high_macs": mh, "conv_output_bytes_fp32": bl + bh,
                       "params": r3_r111.n_params(r3), "logits_bytes_per_net": 225 * N * 4}
    res["r3"] = r3t
    res["copy_bandwidth_GBps"] = float(bandwidth())
    # SVGF (frozen) static and camera form
    arm = AR.SVGFArm(R.PT.cam_basis)
    comp4 = eng.frame(v0, 4, seed(9, 100))
    cache = {}
    fa = AR.Frame(eng, comp4, v0, 4, 1, None, False, 8)
    arm(fa, cache)
    res["svgf_static_ms"] = st(tms(lambda: arm(AR.Frame(eng, comp4, v0, 4, 1, v0, False, 8), cache), warm, rounds))
    flip = [0]

    def svgf_cam():
        flip[0] ^= 1
        cam, prev = (camb, v0) if flip[0] else (v0, camb)
        c2 = {}
        arm(AR.Frame(eng, comp4, cam, 4, 1, prev, False, 8), c2)
    res["svgf_camera_ms"] = st(tms(svgf_cam, warm, rounds))
    # candidate cost grid (untrained weights)
    cg = {}
    for m in (1.0, 1.5, 2.0):
        net = RC.CompactNet(m)
        cand = RC.Candidate(net, C.P["candidate"]["temporal_settings"])
        xin = torch.zeros(1, RC.CIN, W, W, device="cuda")
        macs, ob = conv_flops(cand.net.float(), xin)
        cand.net.to(memory_format=torch.channels_last)
        for sub in (4, 8):
            st_arm = AR.CandidateArm(cand, R.PT.cam_basis, True, "ST")
            s_arm = AR.CandidateArm(cand, R.PT.cam_basis, False, "S")
            cache = {}
            st_arm(AR.Frame(eng, comp4, v0, 4, 1, None, False, sub), cache)
            static = tms(lambda: st_arm(AR.Frame(eng, comp4, v0, 4, 1, v0, False, sub), cache), warm, rounds)
            s_static = tms(lambda: s_arm(AR.Frame(eng, comp4, v0, 4, 1, v0, False, sub), cache), warm, rounds)
            flip = [0]

            def cam_form():
                flip[0] ^= 1
                cam, prev = (camb, v0) if flip[0] else (v0, camb)
                st_arm(AR.Frame(eng, comp4, cam, 4, 1, prev, False, sub), {})
            camera = tms(cam_form, warm, rounds)
            cg[f"m{m}_sub{sub}"] = {"ST_static": st(static), "S_static": st(s_static), "ST_camera_form": st(camera),
                                    "params": RC.n_params(net), "macs": macs, "conv_output_bytes_fp32": ob}
            if smoke:
                break
        if smoke:
            break
    res["candidate_grid"] = cg
    # candidate component breakdown, camera form, m = 1.0: (a) eager attribution of the same operations, (b) the
    # executed form (guide pack + CUDA graph); CUDA events + synchronized wall
    br = {}
    net = RC.CompactNet(1.0)
    cand = RC.Candidate(net, C.P["candidate"]["temporal_settings"])
    for sub in (4, 8):
        h = RC.History(N, "cuda", cand.cfg)
        gc = RC.GraphCandidate(cand, N, True)
        rows_, rows_g = [], []
        for i in range(warm + rounds):
            cam, prev = (camb, v0) if i % 2 == 0 else (v0, camb)
            if i == 0:
                prev = None
            E.sync()
            w0 = time.perf_counter()
            ev = [torch.cuda.Event(enable_timing=True) for _ in range(6)]
            ev[0].record()
            p = eng.pack(cam, sub, prev, True)
            ev[1].record()
            low, high = RC.raw_low_high(comp4)
            al, ah, ess, _, _, _ = h.step(low.reshape(3, N), high.reshape(3, N), 4, p["centre"], p["pure"].reshape(N), cam, 1,
                                          R.PT.cam_basis, hidden=p["hidden"] if prev is not None else None)
            ev[2].record()
            x = RC.features(al.reshape(1, 3, W, W), ah.reshape(1, 3, W, W), p["cover"][None], p["normal"][None],
                            p["depth"][None, None], ess.view(1, 1, W, W))
            ev[3].record()
            with torch.autocast("cuda", dtype=torch.float16):
                lg = cand.net(x.to(memory_format=torch.channels_last))
            ev[4].record()
            RC.spatial_fused(al.reshape(1, 3, W, W), ah.reshape(1, 3, W, W), lg)
            ev[5].record()
            E.sync()
            wall = 1e3 * (time.perf_counter() - w0)
            E.sync()
            w1 = time.perf_counter()
            eg = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
            eg[0].record()
            p2 = eng.pack(cam, sub, prev, True)
            eg[1].record()
            gc.run(low, high, 4, p2, cam, 1, R.PT.cam_basis)
            eg[2].record()
            E.sync()
            wall_g = 1e3 * (time.perf_counter() - w1)
            if i >= warm:
                names = ("guide_pack", "temporal", "features", "network", "spatial_passes")
                r_ = {n: ev[k].elapsed_time(ev[k + 1]) for k, n in enumerate(names)}
                r_["wall_total"] = wall
                rows_.append(r_)
                rows_g.append({"guide_pack": eg[0].elapsed_time(eg[1]), "graph": eg[1].elapsed_time(eg[2]), "wall_total": wall_g})
        br[f"sub{sub}_eager"] = {k: float(np.median([r[k] for r in rows_])) for k in rows_[0]}
        br[f"sub{sub}_graph"] = {k: float(np.median([r[k] for r in rows_g])) for k in rows_g[0]}
    res["candidate_breakdown_camera_form_m1"] = br
    log("candidate breakdown", br)
    # R3 subsystem in camera form: 8 x 8 guide acquisition + R3 (fused, fp32) every frame
    r3arm = AR.R3Arm(r3)
    flip = [0]

    def r3_cam():
        flip[0] ^= 1
        cam = camb if flip[0] else v0
        r3arm(AR.Frame(eng, comp4, cam, 4, 1, None, False, 8), {})
    res["r3_camera_form_ms_4spp"] = st(tms(r3_cam, warm, rounds))
    res["r3_static_form_ms_4spp"] = st(tms(lambda: r3arm(AR.Frame(eng, comp4, v0, 4, 1, None, False, 8), cache), warm, rounds))
    r3e = AR.R3Arm(r3, graph=False)
    flip = [0]

    def r3e_cam():
        flip[0] ^= 1
        cam = camb if flip[0] else v0
        r3e(AR.Frame(eng, comp4, cam, 4, 1, None, False, 8), {})
    res["r3_eager_camera_form_ms_4spp_report"] = st(tms(r3e_cam, warm, rounds))
    return res


def design_rule(t):
    rule = C.P["candidate"]
    g8 = t["guide_ms"]["sub8"]["median"]
    sub = 8 if g8 <= 1.0 else 4
    m_sel = 1.0
    for m in (2.0, 1.5, 1.0):
        k = f"m{m}_sub{sub}"
        if k in t["candidate_grid"] and t["candidate_grid"][k]["ST_camera_form"]["median"] <= 3.0:
            m_sel = m
            break
    c10 = t["candidate_grid"].get(f"m1.0_sub{sub}", {}).get("ST_camera_form", {}).get("median")
    r3c = t["r3_camera_form_ms_4spp"]["median"]
    futile = c10 is not None and c10 >= 0.70 * r3c
    return {"sub": sub, "m": m_sel, "guide8_median_ms": g8, "candidate_m1_camera_form_ms": c10,
            "r3_camera_form_ms": r3c, "futility_ratio": (c10 / r3c) if c10 else None, "cost_futility_stop": bool(futile),
            "rule": rule["stageA_rule"], "futility_rule": rule["cost_futility"]}


def main(out, smoke):
    global LOG
    out.mkdir(parents=True, exist_ok=True)
    LOG = out / "stageA.log"
    res = {"protocol_sha256": C.sha(C.PROTOCOL), "smoke": smoke, "code_sha256": {f.name: C.sha(f) for f in sorted(C.HERE.glob("*.py"))}}
    eng_ok = None
    rt, Bd, B, audit = E.admit(log, replay=not smoke)
    res["G0abc"] = audit
    eng = E.Engine(rt, Bd, B)
    res["equivalence"] = equivalence(eng, smoke)
    log("equivalence", res["equivalence"])
    assert res["equivalence"]["pass"] or smoke, "frame / guide program equivalence failed: STOP"
    r3 = r3_r111.load_r3()
    res["G0d_r3_replay"] = g0d_r3(r3, smoke)
    log("G0d", res["G0d_r3_replay"]["grade"], res["G0d_r3_replay"]["max_rel_l1"])
    assert res["G0d_r3_replay"]["grade"] != "FAIL" or smoke, "R3 not reproduced: STOP"
    if smoke and not (C.CAP107 / "capture_manifest.json").exists():
        res["G0e_r3_quality"] = {"skipped_in_smoke": "WL107 capture not yet copied"}
    else:
        res["G0e_r3_quality"] = g0e_quality(r3, smoke)
    log("G0e", res["G0e_r3_quality"]["pooled_mean_per_draw"])
    res["G0f_svgf"] = g0f_svgf(smoke)
    log("G0f", res["G0f_svgf"].get("status"), res["G0f_svgf"].get("max_rel_l1"))
    C.write(out / "stageA.json", res)
    t, rec = E.timed_block(lambda: timing(eng, r3, smoke), "stageA", log) if not smoke else (timing(eng, r3, smoke), [])
    res["timing"] = t
    res["timing_exclusivity"] = rec
    res["environment"] = E.versions()
    C.write(out / "stageA.json", res)
    d = design_rule(t)
    C.write(out / "design.json", d)
    log("design", d)
    print("STAGE_A_COMPLETE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    main(a.out.resolve(), a.smoke)
