"""WL111A final quality evidence (protocol/recon111a.json): cold current-frame reconstructions of R0, R3, S111 r0 / r1 and
S111A r0 / r1 on the historical views, the WL111 sealed views K1-K6, the new sealed views N1-N6 and the edited states
(camera V0), at the protocol spp grid, 8 independent replicates per (item, spp). Every arm receives the identical
current-frame component means of each seed (paired). No history exists in any arm.

python evalF_r111a.py [--smoke] [--only KEY ...]
  -> OUT/final/quality/<key>_spp<k>.json      per replicate: TOTAL frame_metrics vs T_ECRO (and T_PHYSICAL where it
                                              exists), LOW / HIGH component errors per ROI; per item: bias / variance
                                              decomposition over the replicates (per arm, component, ROI); graph == eager
                                              checks; reference precision
  -> OUT/final/maps/<key>_spp<k>.npz         per-pixel mean / variance of TOTAL over replicates, replicate-0 outputs,
                                              reference (selected keys, fp16)
  -> OUT/final/states.json                   edited-state change statistics (qualification input, references only)

Quality is computed eagerly per component (R3: the frozen eager forward; candidates: Candidate.spatial, the graph body's
fused operations). Each item's replicate 0 also runs the CUDA-graph executions used in timing (R3Graph, GraphCandidate
S-mode) and records rel. L1(graph TOTAL, eager TOTAL).
Items (key, spp) are fully determined by their seeds; RECON111A_SHARD=i/n keeps items whose running index is i mod n.
"""
import argparse
import math
import os
import time

import numpy as np

import common_r111a as A
import common_r111 as C
import engine_r111 as E
import recon_r111 as RC
import recon_r111a as RA
import r3_r111
import arms_r111 as AR
import seq_r111 as SQ
import stageA_r111a as SA

torch, R = E.torch, E.R
W = C.RES
P = C.read(A.HERE / "protocol" / "recon111a.json")
SHARD = tuple(int(x) for x in os.environ.get("RECON111A_SHARD", "0/1").split("/"))
COMPS = ("low", "high", "total")
MAP_KEYS = ("V0", "H1", "K1", "K4", "N1", "N4", "E_cfg1", "E_l1", "E_a24")
MAP_SPP = (1, 4)
PHYS = {"V0": "S0_V0", "H1": "S0_H1", "H2": "S0_H2", "E_cfg1": "S1_V0", "E_l1": "SL_V0"}


def log(*a):
    print(time.strftime("%H:%M:%S"), f"[{SHARD[0]}/{SHARD[1]}]" if SHARD[1] > 1 else "", *a, flush=True)


# ------------------------------------------------------------------------------------------------ items and references
def states():
    """Edited states (camera V0): key -> (geom, light, WL111 reference key)."""
    rows = SQ.rows()
    l0 = rows[0]["lights"][0]
    q4 = {fr["state"]: fr for fr in SQ.sequence("Q4")}
    out = {"E_cfg1": (rows[1]["geom"], l0, "Q3_cfg1_l0"),
           "E_cfg2": (rows[2]["geom"], l0, "Q3_cfg2_l0"),
           "E_l1": (rows[0]["geom"], rows[0]["lights"][1], "Q5_cfg0_l1")}
    for k in (8, 16, 24, 32, 40):
        fr = q4[f"anim{k:02d}"]
        out[f"E_a{k:02d}"] = (fr["geom"], fr["light"], f"Q4_anim{k:02d}")
    assert list(out) == list(P["views"]["edited_states"]["states"])
    # the WL111 sequences built the same states
    assert SQ.reference_frames("Q5")["Q5_cfg0_l1"]["light"] == out["E_l1"][1]
    return out


def items(smoke):
    V = P["views"]
    hist, kv, nv = V["historical"], V["sealed_wl111"], V["sealed_new"]
    keys = [("static", g, i, k) for g, names in enumerate((hist, kv, nv)) for i, k in enumerate(names)]
    keys += [("edit", 0, i, k) for i, k in enumerate(V["edited_states"]["states"])]
    if smoke:
        keys = [k for k in keys if k[3] in ("V0", "E_cfg1")]
    return keys


def load_ref(key, st):
    """-> ref (dict comp -> (3,H,W)), var (dict comp -> (3,H,W) or None), aux, provenance."""
    import eval_r111 as EV
    if key in P["views"]["historical"]:
        mean, var, aux = EV._ref_for(key)
        src = "WL107 historical reference (eval_r111._ref_for)"
        cv = {}
    elif key.startswith("K"):
        p = A.WL111_EVID / "refs" / f"{key}.npz"
        mean, var, aux = EV.load_ref(p)
        src, cv = str(p), {}
    elif key.startswith("N"):
        p = A.OUT / "final" / "refs" / f"{key}.npz"
        mean, var, aux = EV.load_ref(p)
        with np.load(p) as z:
            cv = {c: z[f"var_{c}"].astype(np.float64) for c in ("low", "high")}
        src = str(p)
    else:
        p = A.WL111_EVID / "refs" / f"{st[key][2]}.npz"
        mean, var, aux = EV.load_ref(p)
        src, cv = str(p), {}
    ref = {"low": mean[0:3] + mean[3:6], "high": mean[6:9]}
    ref["total"] = ref["low"] + ref["high"]
    return ref, {"total": var, **cv}, aux, src


def load_phys(key):
    """T_PHYSICAL (descriptive): WL110 REF_FULL halves -> mean (3,H,W), per-pixel variance of the mean ((h0 - h1)^2 / 4)."""
    if key not in PHYS:
        return None
    d = C.ROOT / "inputs" / "phys110"
    hs = []
    for h in (0, 1):
        with np.load(d / f"{PHYS[key]}_FULL_h{h}.npz") as z:
            hs.append((z["e1"] + z["e2"] + z["h"]).astype(np.float64).transpose(2, 0, 1))
    return (hs[0] + hs[1]) / 2, (hs[0] - hs[1]) ** 2 / 4, {f"h{h}": C.sha(d / f"{PHYS[key]}_FULL_h{h}.npz") for h in (0, 1)}


def comp_metrics(y, ref, var, masks):
    """Component error per ROI: raw MSE, reference power, reference variance of the mean (None if the reference has no
    per-component variance) and the relative RMSE (debiased only when the variance exists)."""
    out = {}
    for name, m in masks.items():
        if not m.any():
            continue
        mse = float(((y - ref)[:, m] ** 2).mean())
        r2 = float((ref[:, m] ** 2).mean())
        vr = float(var[:, m].mean()) if var is not None else None
        out[name] = {"mse": mse, "ref2": r2, "var_ref": vr,
                     "rel_rmse": math.sqrt(max(mse - (vr or 0.0), 0.0) / r2) if r2 > 0 else None}
    return out


def ref_precision(ref, var, masks):
    return {k: math.sqrt(float(var[:, m].mean()) / float((ref[:, m] ** 2).mean())) for k, m in masks.items()
            if m.any() and float((ref[:, m] ** 2).mean()) > 0}


# ------------------------------------------------------------------------------------------------ arms
class Arms:
    """Eager per-component reconstructions (quality) plus the timing executions (graph check)."""

    def __init__(self):
        dsg = C.read(A.WL111_EVID / "design.json")
        assert dsg["sub"] == 4
        self.r3 = r3_r111.load_r3()
        self.cands = {f"S111_r{ri}": AR.load_candidate(A.wl111_weights(ri), dsg["m"]) for ri in (0, 1)}
        for ri in (0, 1):
            self.cands[f"S111A_r{ri}"] = RA.load_candidate(A.OUT / "models" / f"S111A_r{ri}.pt", "fullres")
        self.r3g = AR.R3Graph(self.r3)
        self.graphs = {k: RC.GraphCandidate(c, W * W, False) for k, c in self.cands.items()}

    @torch.no_grad()
    def eager(self, comp, spp, g4, g8):
        low, high = RC.raw_low_high(comp)
        out = {"R0": (low, high)}
        _, parts = self.r3(comp[None], r3_r111.aux14(g8, spp))
        out["R3"] = (parts["low"][0], parts["high"][0])
        ess = torch.full((W, W), float(spp), device=comp.device)
        for k, cand in self.cands.items():
            out[k] = cand.spatial(low, high, ess, g4)
        return out

    @torch.no_grad()
    def graph_check(self, eng, comp, spp, cam, eager):
        low, high = RC.raw_low_high(comp)
        p4 = eng.pack(cam, 4, None, False)
        res = {}
        y = self.r3g(comp[None], r3_r111.aux14(eng.guides(cam, 8), spp))[0]
        res["R3"] = rel_l1(y, eager["R3"][0] + eager["R3"][1])
        for k, g in self.graphs.items():
            g.reset()
            y = g.run(low, high, spp, p4, cam, eng.revision, R.PT.cam_basis)
            res[k] = rel_l1(y, eager[k][0] + eager[k][1])
        return res


def rel_l1(a, b):
    return float((a.double() - b.double()).abs().sum() / b.double().abs().sum().clamp(min=1e-30))


def np3(t):
    return t.detach().double().cpu().numpy().reshape(3, W, W)


# ------------------------------------------------------------------------------------------------ main
def state_stats(st, out):
    """Edited-state change statistics against the cfg0 / light-0 V0 reference (WL111 change_stats, references only)."""
    import eval_r111 as EV
    v0 = EV._ref_for("Q3_cfg0_l0")
    res = {}
    for k, (_, _, rk) in st.items():
        ref = EV.load_ref(A.WL111_EVID / "refs" / f"{rk}.npz")
        s = EV.change_stats(v0, ref)
        s["reference_key"] = rk
        s["ref_precision"] = ref_precision(ref[0][0:3] + ref[0][3:6] + ref[0][6:9], ref[1], C.rois(ref[2]))
        res[k] = s
    C.write(out / "states.json", res)


def main(smoke, only):
    out = A.OUT / ("final_smoke" if smoke else "final")
    for d in ("quality", "maps"):
        (out / d).mkdir(parents=True, exist_ok=True)
    eng, audit = SA.setup()
    C.write(out / f"admission_quality_shard{SHARD[0]}of{SHARD[1]}.json", audit)
    st = states()
    if SHARD[0] == 0 and not (out / "states.json").exists():
        state_stats(st, out)
    arms = Arms()
    fam = lambda kind: "smoke" if smoke else ("eval_static" if kind == "static" else "eval_edit")
    grid = [1, 4] if smoke else P["quality_protocol"]["spp_grid"]
    nrep = 2 if smoke else P["quality_protocol"]["replicates"]
    row0 = SQ.rows()[0]
    hist = SQ.wl105_cameras()
    sealed_k = C.read(A.WL111_EVID / "data" / "sealed_views.json")["views"]
    sealed_n = C.read(A.OUT / "final" / "sealed_new.json")["views"] if not smoke else {}
    idx = -1
    for kind, g, vi, key in items(smoke):
        if only and key not in only:
            continue
        work = []
        for si, spp in enumerate(grid):
            idx += 1
            f = out / "quality" / f"{key}_spp{spp:02d}.json"
            if not f.exists() and idx % SHARD[1] == SHARD[0]:
                work.append((si, spp, f))
        if not work:
            continue
        if kind == "static":
            eng.set_state(row0["geom"], row0["lights"][0])
            cam = E.cam_dict({**hist, **sealed_k, **sealed_n}[key])
        else:
            eng.set_state(st[key][0], st[key][1])
            cam = E.cam_dict(C.P["views"]["V0"])
        ref, var, aux, src = load_ref(key, st)
        masks = C.rois(aux)
        phys = load_phys(key)
        g4, g8 = eng.guides(cam, 4), eng.guides(cam, 8)
        # the reference's ROI inputs must be this camera / state's own geometry
        assert np.array_equal(aux["valid_frac"] > 0, g8["valid_frac"].cpu().numpy() > 0), key
        for si, spp, f in work:
            t0 = time.perf_counter()
            P_ = P["quality_protocol"]["spp_grid"].index(spp) if not smoke else si
            rec = {"key": key, "kind": kind, "spp": spp, "camera": cam, "reference": src, "reps": [],
                   "ref_precision": ref_precision(ref["total"], var["total"], masks),
                   "component_var_available": "low" in var, "device": torch.cuda.get_device_name(0)}
            if phys is not None:
                rec["physical"] = {"sha256": phys[2],
                                   "ref_precision": ref_precision(phys[0], phys[1], masks),
                                   "phys_minus_ecro": {k: math.sqrt(max(float(((phys[0] - ref["total"])[:, m] ** 2).mean())
                                                                        - float(phys[1][:, m].mean()) - float(var["total"][:, m].mean()), 0.0)
                                                                    / float((ref["total"][:, m] ** 2).mean()))
                                                       for k, m in masks.items() if m.any()},
                                   "phys_minus_ecro_signed": {k: C.signed_bias(phys[0], ref["total"], m) for k, m in masks.items() if m.any()}}
            mom = {}
            keep = {}
            for rep in range(nrep):
                sd = A.seed(fam(kind), g + (5 if smoke and kind == "edit" else 0), vi * 10 + P_, rep)
                comp = eng.frame(cam, spp, sd)
                outs = arms.eager(comp, spp, g4, g8)
                r = {"seed": sd}
                for arm, (lo, hi) in outs.items():
                    tot = lo + hi
                    y = np3(tot)
                    m_ = {"total": C.frame_metrics(y, ref["total"], var["total"], masks),
                          "low": comp_metrics(np3(lo), ref["low"], var.get("low"), masks),
                          "high": comp_metrics(np3(hi), ref["high"], var.get("high"), masks)}
                    if phys is not None:
                        m_["physical"] = C.frame_metrics(y, phys[0], phys[1], masks)
                    r[arm] = m_
                    for c, t in (("low", lo), ("high", hi), ("total", tot)):
                        if (arm, c) not in mom:
                            mom[(arm, c)] = A.Moments((3, W, W), torch)
                        mom[(arm, c)].add(t)
                    if rep == 0 and key in MAP_KEYS and spp in MAP_SPP:
                        keep[f"{arm}_rep0"] = y.astype(np.float16)
                if rep == 0:
                    r["graph_vs_eager_rel_l1"] = arms.graph_check(eng, comp, spp, cam, outs)
                rec["reps"].append(r)
            dec = {}
            for (arm, c), mo in mom.items():
                m, s2 = mo.mean_var()
                vr = var.get(c)
                d = dec.setdefault(arm, {}).setdefault(c, {})
                for roi, mk in masks.items():
                    if mk.any() and float((ref[c][:, mk] ** 2).mean()) > 0:
                        d[roi] = A.decompose(m, s2, nrep, ref[c], vr if vr is not None else np.zeros_like(m), mk)
                        if c == "total":
                            d[roi]["signed_bias_of_mean"] = C.signed_bias(m, ref["total"], mk)
                if c == "total" and key in MAP_KEYS and spp in MAP_SPP:
                    keep[f"{arm}_mean"] = m.astype(np.float16)
                    keep[f"{arm}_var"] = s2.astype(np.float16)
            rec["decomposition"] = dec
            C.write(f, rec)
            if keep:
                keep["ref"] = ref["total"].astype(np.float16)
                for roi in ("primary", "silhouette", "material_boundary", "sss_interior", "contact"):
                    keep[f"mask_{roi}"] = masks[roi]                  # geometry-only ROI masks (reviewer crops)
                if phys is not None:
                    keep["phys"] = phys[0].astype(np.float16)
                np.savez_compressed(out / "maps" / f"{key}_spp{spp:02d}.npz", **keep)
            pr = {a: round(rec["reps"][0][a]["total"]["primary"]["rel_rmse"], 4) for a in outs}
            log("quality", key, spp, round(time.perf_counter() - t0, 1), "s", pr,
                "graph", max(rec["reps"][0]["graph_vs_eager_rel_l1"].values()))
    eng.set_state(row0["geom"], row0["lights"][0])
    print("EVALF_COMPLETE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    main(a.smoke, a.only)
