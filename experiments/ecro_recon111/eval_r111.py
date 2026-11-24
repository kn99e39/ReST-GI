"""WL111 Stages F / G evidence: references, static protocol, dynamic sequences, end-to-end timing.

python eval_r111.py ROOT STAGE [--smoke]
  STAGE refs    frozen-ECRO references (sealed views, Q2 keyframes, Q3 / Q5 states, every Q4 frame) + sequence qualification
  STAGE static  static protocol (historical + sealed views, spp grid, 65-frame streams) for R0, R3, SVGF, S, ST (2 realizations)
  STAGE seq     sequences Q2-Q5 for every arm, metrics at reference frames, history diagnostics (D1-D6 inputs)
  STAGE timing  exclusive-GPU E2E latency: static, camera motion (Q2 path), continuous geometry update (Q4 path)
ROOT holds design.json, data/ (sealed_views.json), models/ (cand_r0.pt, cand_r1.pt); outputs go to ROOT/<stage>/.
Every metric is computed at capture time by the frozen metric code (common_r111.frame_metrics); raw outputs of
replicate 0 are kept for reviewer artifacts.
"""
import argparse
import math
import time
from pathlib import Path

import numpy as np

import common_r111 as C
import engine_r111 as E
import recon_r111 as RC
import r3_r111
import arms_r111 as AR
import seq_r111 as SQ

torch, R = E.torch, E.R
W = C.RES
N = W * W
ROOT = None
SMOKE = False


def log(*a):
    s = time.strftime("%Y-%m-%d %H:%M:%S ") + " ".join(str(x) for x in a)
    print(s, flush=True)
    with open(ROOT / "eval.log", "a", encoding="utf-8") as f:
        f.write(s + "\n")


# ------------------------------------------------------------------------------------------------ setup
def setup():
    rt, Bd, B, audit = E.admit(log, replay=not SMOKE)
    eng = E.Engine(rt, Bd, B)
    eng.state = (rows()[0]["geom"], rows()[0]["lights"][0])
    eng.revision = 1
    return eng, audit


def rows():
    return SQ.rows()


def views():
    sealed = C.read(ROOT / "data" / "sealed_views.json")["views"]
    hist = SQ.wl105_cameras()
    out = {v: E.cam_dict(hist[v]) for v in C.P["views"]["historical"]}
    out.update({k: E.cam_dict(c) for k, c in sealed.items()})
    if SMOKE:
        out = {k: out[k] for k in ("V0", "K1")}
    return out


def arms(eng, with_svgf=True):
    dsg = C.read(ROOT / "design.json")
    r3 = r3_r111.load_r3()
    out = {"R0": AR.R0Arm(), "R3": AR.R3Arm(r3, graph=True)}
    if with_svgf:
        out["SVGF"] = AR.SVGFArm(R.PT.cam_basis)
    for ri in (0, 1):
        cand = AR.load_candidate(ROOT / "models" / f"cand_r{ri}.pt", dsg["m"])
        out[f"S{ri}"] = AR.CandidateArm(cand, R.PT.cam_basis, False, f"S{ri}")
        out[f"ST{ri}"] = AR.CandidateArm(cand, R.PT.cam_basis, True, f"ST{ri}")
    return out, dsg["sub"]


# ------------------------------------------------------------------------------------------------ references
def make_ref(eng, cam, nblocks, spp_block, seed_fn, path):
    """Mean component means over nblocks independent blocks, variance of the mean of TOTAL, ROI aux; resumable."""
    if path.exists():
        return
    mean = None
    m2 = None
    mt = None
    for k in range(nblocks):
        comp = eng.blocks(cam, spp_block, seed_fn(k), b=min(32, spp_block))
        tot = comp[0:3] + comp[3:6] + comp[6:9]
        if mean is None:
            mean, mt, m2 = comp.copy(), tot.copy(), np.zeros_like(tot)
        else:
            mean += (comp - mean) / (k + 1)
            d = tot - mt
            mt += d / (k + 1)
            m2 += d * (tot - mt)
    var_mean = m2 / (nblocks - 1) / nblocks
    aux = eng.roi_aux(cam)
    np.savez(path, mean=mean.astype(np.float32), var_total=var_mean.astype(np.float32), blocks=nblocks, spp_block=spp_block,
             **{f"aux_{k}": v for k, v in aux.items()})


def load_ref(path):
    with np.load(path) as z:
        aux = {k[4:]: z[k] for k in z.files if k.startswith("aux_")}
        return z["mean"].astype(np.float64), z["var_total"].astype(np.float64), aux


def ref_quality(mean, var, aux):
    ref = mean[0:3] + mean[3:6] + mean[6:9]
    m = C.rois(aux)
    return {k: float(math.sqrt(var[:, v].mean() / (ref[:, v] ** 2).mean())) for k, v in m.items() if v.any()}


def stage_refs(eng):
    out = ROOT / "refs"
    out.mkdir(exist_ok=True)
    Q = C.P["sequences"]
    seal = C.P["views"]["sealed"]
    nb = lambda n: 2 if SMOKE else n
    sb = lambda s: 64 if SMOKE else s
    rec = {}
    # sealed static views (cfg0, light 0)
    eng.set_state(rows()[0]["geom"], rows()[0]["lights"][0])
    for vi, (name, cam) in enumerate((k, c) for k, c in views().items() if k.startswith("K")):
        p = out / f"{name}.npz"
        make_ref(eng, cam, nb(32), sb(8192), lambda k: C.seed("ref_sealed", vi, 0, k, smoke=SMOKE), p)
        rec[name] = ref_quality(*load_ref(p))
        log("ref", name, rec[name]["primary"])
    # sequences
    for si, name in enumerate(("Q2", "Q3", "Q4", "Q5")):
        frames = SQ.reference_frames(name)
        if name == "Q3":
            frames.update({f"Q3_cfg{Q['Q3']['fallback_edit_to_config']}_l0":
                           dict(SQ.sequence("Q3")[0], geom=rows()[Q["Q3"]["fallback_edit_to_config"]]["geom"])})
        for st_i, (key, fr) in enumerate(sorted(frames.items())):
            if SMOKE and st_i > 1:
                break
            p = out / f"{key}.npz"
            if key == "Q3_cfg0_l0" or key == "Q5_cfg0_l0":
                continue                     # V0 cfg0 light 0 = WL107 historical V0 reference
            eng.set_state(fr["geom"], fr["light"])
            cam = E.cam_dict(fr["cam"])
            if name in ("Q3", "Q5"):
                make_ref(eng, cam, nb(Q[name]["reference_blocks_state"]), sb(Q[name]["reference_spp_state"] // Q[name]["reference_blocks_state"]),
                         lambda k: C.seed("ref_seq", si, st_i, k, smoke=SMOKE), p)
            else:
                make_ref(eng, cam, nb(Q[name]["reference_blocks"]), sb(Q[name]["reference_spp"] // Q[name]["reference_blocks"]),
                         lambda k: C.seed("ref_seq", si, st_i, k, smoke=SMOKE), p)
            rec[key] = ref_quality(*load_ref(p))
            log("ref", key, rec[key]["primary"])
    eng.set_state(rows()[0]["geom"], rows()[0]["lights"][0])
    C.write(out / "refs.json", rec)
    C.write(out / "qualification.json", qualify(out))


_REF_CACHE = {}


def ref_for(key):
    """Memoized _ref_for (references are immutable inputs; loaded and verified once per process)."""
    if key not in _REF_CACHE:
        _REF_CACHE[key] = _ref_for(key)
    return _REF_CACHE[key]


def _ref_for(key):
    """key -> (mean comps, var_total, aux). V0 cfg0 light 0 states map to the WL107 historical V0 reference."""
    if key in C.P["views"]["historical"] or key in ("Q3_cfg0_l0", "Q5_cfg0_l0"):
        import profile_r111
        v = "V0" if key.startswith("Q") else key
        mean, var = profile_r111.reference107(v)
        with np.load(C.D105 / "eval" / v / "aux.npz") as z:
            aux = {k: z[k] for k in z.files}
        return mean, var, aux
    return load_ref(ROOT / "refs" / f"{key}.npz")


def stationary_mask(aux_a, aux_b):
    """Stationary receivers: centre-ray shape is not a box (7 / 8) in either state, valid, contact disk excluded."""
    ma, mb = C.rois(aux_a), C.rois(aux_b)
    sa, sb = aux_a["sid"], aux_b["sid"]
    return (sa >= 0) & (sb >= 0) & (sa < 7) & (sb < 7) & ~ma["contact"] & ~mb["contact"]


def change_stats(a, b):
    (ma, va, xa), (mb, vb, xb) = a, b
    m = stationary_mask(xa, xb)
    ta, tb = ma[0:3] + ma[3:6] + ma[6:9], mb[0:3] + mb[3:6] + mb[6:9]
    ia, ib = ma[3:6] + ma[6:9], mb[3:6] + mb[6:9]
    rms = lambda d, t: float(math.sqrt((d[:, m] ** 2).mean() / (t[:, m] ** 2).mean()))
    noise = float(math.sqrt((va + vb)[:, m].mean() / (ta[:, m] ** 2).mean()))
    return {"total_change": rms(tb - ta, ta), "indirect_change": rms(ib - ia, ia), "noise_total": noise,
            "pixels": int(m.sum())}


def qualify(out):
    Q = C.P["sequences"]
    res = {}
    v0 = ref_for("Q3_cfg0_l0")
    for cfg in (Q["Q3"]["edit_to_config"], Q["Q3"]["fallback_edit_to_config"]):
        p = out / f"Q3_cfg{cfg}_l0.npz"
        if p.exists():
            s = change_stats(v0, load_ref(p))
            s["qualifies"] = s["total_change"] >= 0.05 and s["indirect_change"] >= 0.03 and s["total_change"] >= 10 * s["noise_total"]
            res[f"Q3_cfg{cfg}"] = s
    res["Q3_selected_config"] = next((c for c in (Q["Q3"]["edit_to_config"], Q["Q3"]["fallback_edit_to_config"])
                                      if res.get(f"Q3_cfg{c}", {}).get("qualifies")), None)
    p5 = out / f"Q5_cfg0_l{Q['Q5']['light_index']}.npz"
    if p5.exists():
        s = change_stats(v0, load_ref(p5))
        s["qualifies"] = s["total_change"] >= 0.05 and s["total_change"] >= 10 * s["noise_total"]
        res["Q5"] = s
    n4 = Q["Q4"]["frames"]
    ks = [out / f"Q4_anim{k:02d}.npz" for k in range(n4)]
    if all(k.exists() for k in ks):
        first, last = load_ref(ks[0]), load_ref(ks[-1])
        s = change_stats(first, last)
        steps = []
        prev = first
        for k in ks[1:]:
            cur = load_ref(k)
            c = change_stats(prev, cur)
            steps.append(c["total_change"] >= 3 * c["noise_total"])
            prev = cur
        s["per_step_resolved_fraction"] = float(np.mean(steps))
        s["qualifies"] = (s["total_change"] >= 0.05 and s["indirect_change"] >= 0.03 and s["total_change"] >= 10 * s["noise_total"]
                          and s["per_step_resolved_fraction"] >= 0.5)
        res["Q4"] = s
    q2 = []
    for key in sorted(SQ.reference_frames("Q2")):
        p = out / f"{key}.npz"
        if p.exists():
            _, _, aux = load_ref(p)
            q2.append(float(aux["valid_frac"].mean()) >= 0.9 and int((aux["cover"][6] >= 1).sum()) >= 2000)
    res["Q2"] = {"keyframes_ok_fraction": float(np.mean(q2)) if q2 else None, "qualifies": bool(q2) and all(q2)}
    return res


# ------------------------------------------------------------------------------------------------ static protocol
def np3(t):
    return t.detach().double().cpu().numpy().reshape(3, W, W)


def rel_l1(a, b):
    return float((a - b).abs().sum() / b.abs().sum().clamp(min=1e-30))


def stage_static(eng):
    S = C.P["static_protocol"]
    out = ROOT / "static"
    out.mkdir(exist_ok=True)
    A, sub = arms(eng)
    grid = S["spp_grid"] if not SMOKE else [1, 4]
    reps = S["replicates"] if not SMOKE else 1
    nfr = S["frames"] if not SMOKE else 6
    rec_frames = set(S["recorded_frames"]) if not SMOKE else {1, 2, 5}
    warm = nfr - 1
    flick = list(range(warm - 7, warm + 1))
    eng.set_state(rows()[0]["geom"], rows()[0]["lights"][0])
    for vname, cam in views().items():
        hist = vname in C.P["views"]["historical"]
        fam = "static_hist" if hist else "static_sealed"
        vi = (C.P["views"]["historical"] if hist else C.P["views"]["sealed"]["names"]).index(vname)
        mean, var, aux = ref_for(vname) if hist else load_ref(ROOT / "refs" / f"{vname}.npz")
        ref = mean[0:3] + mean[3:6] + mean[6:9]
        masks = C.rois(aux)
        for si, spp in enumerate(grid):
            for rep in range(reps):
                f = out / f"{vname}_spp{spp:02d}_r{rep}.json"
                if f.exists():
                    continue
                for a in A.values():
                    a.reset()
                cache = {}
                res = {"view": vname, "spp": spp, "rep": rep, "metrics": {}, "flicker": {}, "reset_check": {}}
                prev_out = {}
                fl = {k: [] for k in A}
                keep = {}
                for fr in range(1, nfr + 1):
                    sd = C.seed(fam, vi, C.P["static_protocol"]["spp_grid"].index(spp) * 10 + rep, fr, smoke=SMOKE)
                    comp = eng.frame(cam, spp, sd)
                    frm = AR.Frame(eng, comp, cam, spp, eng.revision, cam, False, sub)
                    if fr == nfr:                      # forced reset frame: ST must equal S on the same input
                        for ri in (0, 1):
                            A[f"ST{ri}"].reset()
                            y_st = A[f"ST{ri}"](frm, cache)
                            y_s = A[f"S{ri}"](frm, cache)
                            res["reset_check"][f"ST{ri}"] = rel_l1(y_st, y_s)
                        break
                    temporal = [k for k in A if k.startswith("ST") or k == "SVGF"]
                    active = list(temporal)
                    if fr == 1 or fr in flick:
                        active += [k for k in A if k not in temporal]
                    for k in active:
                        y = A[k](frm, cache)
                        rec = (k in temporal and fr in rec_frames) or (k not in temporal and fr in (1, warm))
                        if rec:
                            res["metrics"][f"{k}_f{fr}"] = C.frame_metrics(np3(y), ref, var, masks)
                        if fr in flick:
                            if k in prev_out and fr - 1 in flick:
                                d = (y - prev_out[k]).double()
                                fl[k].append(float((d.view(3, N)[:, torch.from_numpy(masks["primary"].reshape(-1)).cuda()] ** 2).mean().item()))
                            prev_out[k] = y.clone()
                        if (rep == 0 or vname in ("V0", "H1", "H2")) and spp in (1, 4) and fr in (1, warm):
                            keep[f"{k}_f{fr}"] = np3(y).astype(np.float16)
                    if fr in rec_frames:
                        for ri in (0, 1):
                            ag = A[f"ST{ri}"].last["age"]
                            res.setdefault("st_age", {})[f"ST{ri}_f{fr}"] = float(ag.float().mean())
                r2 = float((ref[:, masks["primary"]] ** 2).mean())
                res["flicker"] = {k: (math.sqrt(np.mean(v) / r2) if v else None) for k, v in fl.items()}
                C.write(f, res)
                if keep:
                    np.savez_compressed(out / f"{vname}_spp{spp:02d}_r{rep}_images.npz", **keep)
            log("static", vname, spp)


# ------------------------------------------------------------------------------------------------ sequences
def stage_seq(eng):
    Q = C.P["sequences"]
    out = ROOT / "seq"
    out.mkdir(exist_ok=True)
    A, sub = arms(eng)
    qual = C.read(ROOT / "refs" / "qualification.json")
    grid = Q["candidate_spp_grid_dynamic"] if not SMOKE else [1, 4]
    reps = Q["replicates"] if not SMOKE else 1
    for si, name in enumerate(("Q2", "Q3", "Q4", "Q5")):
        if name == "Q3":
            cfg = qual.get("Q3_selected_config")
            if cfg is None:
                log("Q3 not qualifying: skipped (reported)")
                continue
            Q["Q3"]["edit_to_config"] = cfg
        seq = SQ.sequence(name)
        if SMOKE:
            seq = seq[:6] if name != "Q3" and name != "Q5" else seq[Q[name]["edit_frame"] - 2:Q[name]["edit_frame"] + 3]
        refkeys = {}
        for key, fr in SQ.reference_frames(name).items():
            refkeys.setdefault(fr["key"], key) if name in ("Q2", "Q4") else None
        for spi, spp in enumerate(grid):
            for rep in range(reps):
                f = out / f"{name}_spp{spp}_r{rep}.json"
                if f.exists():
                    continue
                for a in A.values():
                    a.reset()
                eng.state = None
                res = {"sequence": name, "spp": spp, "rep": rep, "frames": {}}
                prev_cam, last_edit = None, 0
                outs = {}
                keep = {}
                cache = {}
                for fr in seq:
                    k = fr["key"]
                    changed = eng.state is None or eng.state != (fr["geom"], fr["light"])
                    if changed:
                        eng.set_state(fr["geom"], fr["light"])
                        last_edit = k
                        cache = {}
                    cam = E.cam_dict(fr["cam"])
                    if prev_cam is not None and cam != prev_cam:
                        cache = {}
                    sd = C.seed("seq", si, grid.index(spp) * 10 + rep, k, smoke=SMOKE)
                    comp = eng.frame(cam, spp, sd)
                    frm = AR.Frame(eng, comp, cam, spp, eng.revision, prev_cam, changed, sub)
                    ys = {an: a(frm, cache) for an, a in A.items()}
                    row = {"since_edit": k - last_edit, "state": fr["state"]}
                    for ri in (0, 1):
                        L = A[f"ST{ri}"].last
                        row[f"ST{ri}_age_max"] = float(L["age"].max())
                        row[f"ST{ri}_accept"] = float(L["accept"].float().mean())
                        row[f"ST{ri}_eq_S"] = rel_l1(ys[f"ST{ri}"], ys[f"S{ri}"]) if k == last_edit else None
                        if name == "Q2" and L.get("moving"):
                            cen = frm.pack(sub, True, cache)["centre"]
                            pc = prev_cam
                            hid = eng.hidden_from(pc, cen[:3], oracle=True)
                            import svgf as SV
                            x, y, z, _, _ = SV.projection(cen, pc, W, R.PT.cam_basis)
                            offscreen = (z <= 0) | (x < -0.5) | (x >= W - 0.5) | (y < -0.5) | (y >= W - 0.5)
                            dis = (cen[6] >= 0) & (hid | offscreen)
                            row[f"ST{ri}_disoccluded"] = int(dis.sum())
                            row[f"ST{ri}_disoccluded_retained"] = int((dis & L["accept"]).sum())
                    if name in ("Q3", "Q5"):
                        rk = f"{name}_{fr['state']}"
                    else:
                        rk = refkeys.get(k)
                    if rk is not None and SMOKE and rk not in ("Q3_cfg0_l0", "Q5_cfg0_l0") and not (ROOT / "refs" / f"{rk}.npz").exists():
                        rk = None                      # smoke builds only a few references
                    if rk is not None:
                        mean, var, aux = ref_for(rk)
                        ref = mean[0:3] + mean[3:6] + mean[6:9]
                        masks = C.rois(aux)
                        row["metrics"] = {an: C.frame_metrics(np3(y), ref, var, masks) for an, y in ys.items()}
                        if name in ("Q2", "Q4"):
                            outs[k] = {an: np3(y).astype(np.float32) for an, y in ys.items()}
                            outs[k]["_ref"] = (ref.astype(np.float32), var.astype(np.float32), masks)
                    if rep == 0 and spp in (1, 4):
                        keep[f"f{k:02d}"] = np.stack([np3(ys[an]) for an in sorted(ys)]).astype(np.float16)
                        L0 = A["ST0"].last
                        keep[f"age{k:02d}"] = L0["age"].clamp(max=255).to(torch.uint8).view(W, W).cpu().numpy()
                        keep[f"acc{k:02d}"] = L0["accept"].view(W, W).cpu().numpy()
                    res["frames"][str(k)] = row
                    prev_cam = cam
                res["arms_order"] = sorted(A)
                res["temporal"] = temporal_pairs(name, outs)
                C.write(f, res)
                if keep:
                    np.savez_compressed(out / f"{name}_spp{spp}_r{rep}_frames.npz", **keep)
                log("seq", name, spp, rep)
    eng.set_state(rows()[0]["geom"], rows()[0]["lights"][0])


def temporal_pairs(name, outs):
    """Temporal-change error and signed-change slope on referenced frame pairs (Q2: d = 4 and the extra pairs; Q4: d = 1)."""
    if not outs:
        return {}
    keys = sorted(outs)
    if name == "Q2":
        pairs = [(a, a + 4) for a in keys if a + 4 in outs] + [(a - 1, a) for a in C.P["sequences"]["Q2"]["reference_extra"]
                                                               if a - 1 in outs and a in outs]
    else:
        pairs = [(a, a + 1) for a in keys if a + 1 in outs]
    res = {}
    for a, b in pairs:
        ra, va, ma = outs[a]["_ref"]
        rb, vb, mb = outs[b]["_ref"]
        dT = rb.astype(np.float64) - ra
        noise = np.sqrt(va + vb)
        m = ma["primary"] & mb["primary"]
        lT = C.luma(dT)
        changed = m & (np.abs(lT) > 3 * C.luma(noise))
        r2 = float((rb[:, m].astype(np.float64) ** 2).mean())
        row = {}
        for an in outs[a]:
            if an == "_ref":
                continue
            dY = outs[b][an].astype(np.float64) - outs[a][an]
            e = float(((dY - dT)[:, m] ** 2).mean() - (va + vb)[:, m].mean())
            lY = C.luma(dY)
            slope = float((lY[changed] * lT[changed]).sum() / max((lT[changed] ** 2).sum(), 1e-30)) if changed.any() else None
            row[an] = {"tce": math.sqrt(max(e, 0.0) / r2), "tce_raw_mse": e, "slope": slope, "changed_pixels": int(changed.sum())}
        res[f"{a}-{b}"] = row
    return res


# ------------------------------------------------------------------------------------------------ timing
def stage_timing(eng):
    out = ROOT / "timing"
    out.mkdir(exist_ok=True)
    T = C.P["timing"]
    warm, rounds = (1, 3) if SMOKE else (T["warmup"], T["rounds"])
    A, sub = arms(eng)
    A["R3e"] = AR.R3Arm(A["R3"].model, graph=False)        # historical eager execution of the same R3 (report only)
    use = ["R0", "R3", "R3e", "SVGF", "S0", "ST0"]
    v0 = E.cam_dict(C.P["views"]["V0"])
    grid = C.P["static_protocol"]["spp_grid"] if not SMOKE else [1, 4]
    sd = lambda reg, slot, i: C.seed("timing", reg, slot, i % 1000, smoke=SMOKE)

    def sync():
        E.sync()

    def static_block():
        """Interleaved: per spp, every arm warms up, then each round times one frame of every arm in a shuffled order."""
        import random
        rs = random.Random(C.P["seeds"]["timing_order"])
        res = {}
        eng.set_state(rows()[0]["geom"], rows()[0]["lights"][0])
        for si, spp in enumerate(grid):
            caches = {an: {} for an in use}
            for an in use:
                A[an].reset()
            times = {an: [] for an in use}
            for i in range(warm + rounds):
                order = list(use)
                rs.shuffle(order)
                for an in order:
                    sync()
                    t0 = time.perf_counter()
                    comp = eng.frame(v0, spp, sd(0, si, i))
                    A[an](AR.Frame(eng, comp, v0, spp, eng.revision, v0, False, sub), caches[an])
                    sync()
                    if i >= warm:
                        times[an].append(1e3 * (time.perf_counter() - t0))
            for an in use:
                res[f"{an}_{spp}"] = times[an]
            res[f"load1_after_{spp}"] = E.load1()
        return res

    def path_block(name, regime):
        """Per spp: two passes over the sequence for every arm, arms alternating pass by pass (shared CPU conditions)."""
        seq = SQ.sequence(name)
        res = {}
        dgrid = C.P["sequences"]["candidate_spp_grid_dynamic"] if not SMOKE else [1, 4]
        frames = seq if not SMOKE else seq[:5]
        for si, spp in enumerate(dgrid):
            times = {an: [] for an in use}
            upd = {an: [] for an in use}
            for rep in range(2 if not SMOKE else 1):
                order = use if rep == 0 else list(reversed(use))
                for an in order:
                    a = A[an]
                    a.reset()
                    eng.state = None
                    eng.set_state(frames[0]["geom"], frames[0]["light"])
                    prev = None
                    for j, fr in enumerate(frames):
                        sync()
                        t0 = time.perf_counter()
                        changed = eng.state != (fr["geom"], fr["light"])
                        if changed:
                            eng.set_state(fr["geom"], fr["light"])
                        t1 = time.perf_counter()
                        cam = E.cam_dict(fr["cam"])
                        comp = eng.frame(cam, spp, sd(regime, si, fr["key"] + 100 * rep))
                        a(AR.Frame(eng, comp, cam, spp, eng.revision, prev, changed, sub), {})
                        sync()
                        if j >= len(frames) // 4:                      # first quarter of each pass = warm-up
                            times[an].append(1e3 * (time.perf_counter() - t0))
                            upd[an].append(1e3 * (t1 - t0))
                        prev = cam
            for an in use:
                res[f"{an}_{spp}"] = times[an]
                res[f"{an}_{spp}_update_part"] = upd[an]
            res[f"load1_after_{spp}"] = E.load1()
        return res

    def components_block():
        """CUDA-event spans at 4 spp along the Q2 path (camera form): the candidate's guide pack (guides + centre +
        visibility, one Dr.Jit program) and its graph (temporal + features + network + passes, inputs copied in); R3's
        8 x 8 guides and its graph; plus R3's eager stage breakdown (static form)."""
        res = {"candidate": [], "r3_graph": [], "r3_eager_stages": []}
        st_arm, r3arm = A["ST0"], A["R3"]
        st_arm.reset()
        comp = eng.frame(v0, 4, sd(9, 0, 0))
        cams = [E.cam_dict(fr["cam"]) for fr in SQ.sequence("Q2")[:40]]
        prev = None
        for i, cam in enumerate(cams):
            frm = AR.Frame(eng, comp, cam, 4, eng.revision, prev, False, sub)
            ev = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
            ev[0].record()
            p = frm.pack(sub, True, {})
            ev[1].record()
            low, high = RC.raw_low_high(comp)
            st_arm.g.run(low, high, 4, p, cam, eng.revision, R.PT.cam_basis)
            ev[2].record()
            p8 = frm.pack(8, False, {})
            ev[3].record()
            r3arm.graph(comp[None], r3_r111.aux14(p8, 4))
            ev[4].record()
            torch.cuda.synchronize()
            if i >= 5:
                res["candidate"].append({"guide_pack": ev[0].elapsed_time(ev[1]), "graph": ev[1].elapsed_time(ev[2])})
                res["r3_graph"].append({"guides_8x8": ev[2].elapsed_time(ev[3]), "graph": ev[3].elapsed_time(ev[4])})
            prev = cam
        sg = r3_r111.Staged(r3arm.model)
        aux = r3_r111.aux14(eng.guides(v0, 8), 4)
        for i in range(warm + rounds):
            _, s_, _ = sg(comp[None], aux)
            if i >= warm:
                res["r3_eager_stages"].append(s_)
        return res

    blocks = {"static": static_block, "camera_Q2": lambda: path_block("Q2", 1), "continuous_Q4": lambda: path_block("Q4", 2),
              "components": components_block}
    res = {}
    for name, fn in blocks.items():
        f = out / f"{name}.json"
        if f.exists():
            continue
        if SMOKE:
            r = fn()
            ex = []
        else:
            r, ex = E.timed_block(fn, name, log)
        C.write(f, {"times_ms": r, "exclusivity": ex, "environment": E.versions()})
        log("timing", name)
    eng.set_state(rows()[0]["geom"], rows()[0]["lights"][0])


def main(root, stage, smoke):
    global ROOT, SMOKE
    ROOT, SMOKE = root, smoke
    root.mkdir(parents=True, exist_ok=True)
    eng, audit = setup()
    C.write(root / f"admission_{stage}.json", audit)
    {"refs": stage_refs, "static": stage_static, "seq": stage_seq, "timing": stage_timing}[stage](eng)
    print(f"EVAL_{stage.upper()}_COMPLETE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("stage")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    main(a.root.resolve(), a.stage, a.smoke)
