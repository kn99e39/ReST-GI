"""WL111A Stage A (development data only): frozen identities, WL111 replay, dev references, and the cold-frame bias /
variance recovery of R0 / R3 / S111 r0 / r1 on the WL105 validation views Q00-Q07.

python stageA_r111a.py [--smoke] [--n N]
  OUT/stageA/identities.json    admission, checkpoint hashes, WL111 static-item replay
  OUT/devref/<view>.npz         64 x 1024-spp blocks: component means (9), variance of the LOW / HIGH / TOTAL mean, ROI aux
  OUT/stageA/devset.json        per view / spp / arm / ROI / component: bias^2, variance, debiased MSE (+ signed bias,
                                silhouette gradient ratio of the mean image)
  OUT/stageA/maps_<view>_spp<k>.npz  mean / variance images for reviewer maps (fp16; selected views only)
"""
import argparse
import json
import math
import time

import numpy as np

import common_r111a as A
import common_r111 as C
import engine_r111 as E
import recon_r111 as RC
import r3_r111
import arms_r111 as AR
import seq_r111 as SQ

torch, R = E.torch, E.R
W = C.RES
REF_BLOCKS, REF_SPP = 64, 1024
COMPS = ("low", "high", "total")
MAP_VIEWS = ("Q00", "Q03")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def setup():
    rt, Bd, B, audit = E.admit(log, replay=True)
    eng = E.Engine(rt, Bd, B)
    row = SQ.rows()[0]
    eng.state = (row["geom"], row["lights"][0])
    eng.revision = 1
    eng.set_state(row["geom"], row["lights"][0])
    return eng, audit


def identities(eng, audit, out):
    rec = {"admission": audit, "r3_sha256": C.sha(C.M105 / "R3.pt"), "protocol111_sha256": C.sha(C.PROTOCOL),
           "s111": {ri: C.sha(A.wl111_weights(ri)) for ri in (0, 1)}}
    for ri in (0, 1):
        lg = C.read(A.WL111_EVID / "models" / f"cand_r{ri}_log.json")
        rec["s111"][f"{ri}_matches_training_log"] = lg["weights_sha256"] == rec["s111"][ri]
    # replay one WL111 static item, frame 1 (cold): R3 graph and both S arms through the WL111 arm code
    import eval_r111 as EV
    dsg = C.read(A.WL111_EVID / "design.json")
    cam = E.cam_dict(SQ.wl105_cameras()["V0"])
    spp, rep, fr = 4, 0, 1
    sd = C.seed("static_hist", 0, C.P["static_protocol"]["spp_grid"].index(spp) * 10 + rep, fr)
    comp = eng.frame(cam, spp, sd)
    mean, var, aux = EV._ref_for("V0")
    ref = mean[0:3] + mean[3:6] + mean[6:9]
    masks = C.rois(aux)
    r3 = r3_r111.load_r3()
    arms = {"R3": AR.R3Arm(r3, graph=True)}
    for ri in (0, 1):
        arms[f"S{ri}"] = AR.CandidateArm(AR.load_candidate(A.wl111_weights(ri), dsg["m"]), R.PT.cam_basis, False, f"S{ri}")
    frm = AR.Frame(eng, comp, cam, spp, eng.revision, cam, False, dsg["sub"])
    stored = C.read(A.WL111_EVID / "static" / "V0_spp04_r0.json")["metrics"]
    rep_ = {}
    cache = {}
    for k, a in arms.items():
        y = a(frm, cache)
        mt = C.frame_metrics(y.detach().double().cpu().numpy().reshape(3, W, W), ref, var, masks)
        st = stored[f"{k}_f1"]
        rep_[k] = {"primary_rel_rmse": mt["primary"]["rel_rmse"], "stored": st["primary"]["rel_rmse"],
                   "max_abs_diff_all_rois": max(abs(mt[r]["rel_rmse"] - st[r]["rel_rmse"]) for r in st)}
    rec["wl111_replay_V0_spp04_r0_f1"] = rep_
    rec["replay_exact"] = all(v["max_abs_diff_all_rois"] == 0.0 for v in rep_.values())
    C.write(out / "identities.json", rec)
    log("identities", json.dumps({k: v for k, v in rec.items() if k != "admission"})[:600])
    del arms
    return rec


def dev_ref(eng, cams, out, smoke):
    d = A.OUT / "devref"
    d.mkdir(parents=True, exist_ok=True)
    nb = 4 if smoke else REF_BLOCKS
    for vi, v in enumerate(A.DEV_VIEWS):
        p = d / f"{v}.npz"
        if p.exists():
            continue
        t0 = time.perf_counter()
        mean = None
        sums = {c: [None, None] for c in COMPS}
        for k in range(nb):
            comp = eng.blocks(cams[v], REF_SPP, A.seed("devref", vi, k // 1000, k % 1000), b=32)
            parts = {"low": comp[0:3] + comp[3:6], "high": comp[6:9]}
            parts["total"] = parts["low"] + parts["high"]
            mean = comp.copy() if mean is None else mean + comp
            for c in COMPS:
                if sums[c][0] is None:
                    sums[c] = [parts[c].copy(), parts[c] ** 2]
                else:
                    sums[c][0] += parts[c]
                    sums[c][1] += parts[c] ** 2
        mean /= nb
        var = {}
        for c in COMPS:
            m = sums[c][0] / nb
            var[c] = np.maximum(sums[c][1] / nb - m * m, 0.0) * nb / (nb - 1) / nb     # variance of the mean
        aux = eng.roi_aux(cams[v])
        np.savez(p, mean=mean.astype(np.float32), **{f"var_{c}": var[c].astype(np.float32) for c in COMPS},
                 blocks=nb, spp_block=REF_SPP, **{f"aux_{k}": x for k, x in aux.items()})
        log("devref", v, round(time.perf_counter() - t0, 1), "s")


def load_devref(v):
    with np.load(A.OUT / "devref" / f"{v}.npz") as z:
        aux = {k[4:]: z[k] for k in z.files if k.startswith("aux_")}
        m = z["mean"].astype(np.float64)
        ref = {"low": m[0:3] + m[3:6], "high": m[6:9]}
        ref["total"] = ref["low"] + ref["high"]
        return ref, {c: z[f"var_{c}"].astype(np.float64) for c in COMPS}, aux


class Recons:
    """Cold-frame reconstructions of one frame, per component: R0 (raw), R3 (frozen eager forward; equal to its graph),
    S111 r0 / r1 (Candidate.spatial: the graph body's operations, fused kernels)."""

    def __init__(self, extra=None):
        dsg = C.read(A.WL111_EVID / "design.json")
        self.sub = dsg["sub"]
        self.r3 = r3_r111.load_r3()
        self.cands = {f"S111_r{ri}": AR.load_candidate(A.wl111_weights(ri), dsg["m"]) for ri in (0, 1)}
        self.extra = extra or {}                                   # name -> (Candidate, sub) for Stage B arms

    @torch.no_grad()
    def __call__(self, comp, spp, g4, g8, gx=None):
        low, high = RC.raw_low_high(comp)
        out = {"R0": (low, high)}
        aux = r3_r111.aux14(g8, spp)
        _, parts = self.r3(comp[None], aux)
        out["R3"] = (parts["low"][0], parts["high"][0])
        ess = torch.full((W, W), float(spp), device=comp.device)
        for k, cand in self.cands.items():
            out[k] = cand.spatial(low, high, ess, g4)
        for k, (cand, sub) in self.extra.items():
            out[k] = cand.spatial(low, high, ess, (gx or {})[sub])
        return out


def devset(eng, cams, out, n, smoke, rec=None, label="devset", only=None):
    """Bias / variance over n independent cold frames per (view, spp). rec: a Recons; only: restrict to these arms."""
    rec = rec or Recons()
    res = {}
    views = A.DEV_USED[:2] if smoke else A.DEV_USED
    for v in views:
        vi = A.DEV_VIEWS.index(v)
        ref, var_ref, aux = load_devref(v)
        masks = C.rois(aux)
        cam = cams[v]
        g4, g8 = eng.guides(cam, 4), eng.guides(cam, 8)
        gx = {4: g4, 8: g8}
        for si, spp in enumerate(A.DEV_SPP):
            t0 = time.perf_counter()
            mom = {}
            for k in range(n):
                comp = eng.frame(cam, spp, A.seed("devset", vi, si, k))
                outs = rec(comp, spp, g4, g8, gx)
                for arm, (lo, hi) in outs.items():
                    if only and arm not in only:
                        continue
                    for c, y in (("low", lo), ("high", hi), ("total", lo + hi)):
                        key = (arm, c)
                        if key not in mom:
                            mom[key] = A.Moments((3, W, W), torch)
                        mom[key].add(y)
            cell = {}
            for (arm, c), mo in mom.items():
                m, s2 = mo.mean_var()
                d = cell.setdefault(arm, {}).setdefault(c, {})
                for roi, mk in masks.items():
                    if not mk.any():
                        continue
                    d[roi] = A.decompose(m, s2, n, ref[c], var_ref[c], mk)
                    if c == "total":
                        d[roi]["signed_bias_of_mean"] = C.signed_bias(m, ref["total"], mk)
                        if roi in ("silhouette", "material_boundary"):
                            d[roi]["grad_ratio_of_mean"] = C.grad_ratio(m, ref["total"], mk)
                if c == "total" and v in MAP_VIEWS and spp in (1, 4) and label == "devset":
                    np.savez_compressed(out / f"maps_{v}_spp{spp}_{arm}.npz", mean=m.astype(np.float16), var=s2.astype(np.float16),
                                        ref=ref["total"].astype(np.float16))
            res.setdefault(v, {})[str(spp)] = cell
            C.write(out / f"{label}.partial.json", {"n": n, "cells": res})
            log(label, v, spp, round(time.perf_counter() - t0, 1), "s",
                {a: round(cell[a]["total"]["primary"]["rel_rmse"] or 0.0, 4) for a in cell})
    C.write(out / f"{label}.json", {"n": n, "views": list(views), "spp": list(A.DEV_SPP), "cells": res})
    return res


def main(smoke, n):
    out = A.OUT / "stageA"
    out.mkdir(parents=True, exist_ok=True)
    eng, audit = setup()
    cams = {k: E.cam_dict(c) for k, c in SQ.wl105_cameras().items()}
    identities(eng, audit, out)
    dev_ref(eng, cams, out, smoke)
    devset(eng, cams, out, 4 if smoke else n, smoke)
    print("STAGEA_COMPLETE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n", type=int, default=64)
    a = ap.parse_args()
    main(a.smoke, a.n)
