"""WL104K Stage F (conditional): fixed-proposal scene validation of A / T / P through an experiment-local path.

fixed  : WL104I / WL104H exact replay of the archived WL104G launches (E1 at the fixed light, E2 with the frozen native R0 proposal and
         its actual pdf, F/p recomputed per arm). Closures: A0 = historical R0 (compiled runtime), A_r = compiled ER-v1-GF runtime,
         T_r / P_r = common104k.DrSDM, r = the 4 primary starts.
analyze: identity gates, objects stable RMS per event, start-mean paired contrasts (P - A, T - A, P - T) with replicate jackknife.

Usage: python scenes104k.py fixed|analyze <scene> <fit_dir> <out_dir> [--reps a:b] [--smoke]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

import common104k as CM

sys.path.insert(0, str(CM.HERE.parent / "ecro_wl84_104i"))
import scenes104i as SI  # noqa: E402  (FixedPass, settings, identity gates)

IO, GF, S8 = CM.IO, CM.GF, SI.S8
H, C, RT, mi, dr = SI.H, SI.C, SI.RT, SI.mi, SI.dr
from metrics104e import event_metrics  # noqa: E402
from stats104b import jk_ci  # noqa: E402

SC = SI.SC                                  # WL104I scene seeds / spp (unchanged)
ROIS = SI.ROIS


def arm_states(fit_dir, n):
    fj = CM.read(fit_dir / "fit.json")
    R = len(fj["start_keys"])
    with np.load(fit_dir / "states.npz") as z:
        S = {k: z[k] for k in z.files}
    out = {}
    for arm in CM.ARMS:
        y = S[f"{arm}_y"].reshape(R, n, 58)
        g = S[f"{arm}_g"].reshape(R, n, *S[f"{arm}_g"].shape[1:]) if arm != "A" else None
        for r in range(R):
            comp = GF.compile_states(y[r])
            comp = comp.cpu().numpy() if torch.is_tensor(comp) else np.asarray(comp)
            out[f"{arm}_{r}"] = (arm, comp, None if g is None else g[r])
    return out, R


class Fixed104K(SI.FixedPass):
    def __init__(self, tag, r0_states, arms, i):
        super().__init__(tag, {"A0": r0_states}, i)
        for name, (arm, comp, g) in arms.items():
            self.cl[name] = H.CD.DrClosure83("ER", comp[i]) if arm == "A" else CM.DrSDM(comp[i], g[i])


def setup(scene):
    i, row = S8.scene_row(scene)
    if "held" in SI.SC_ROLE(scene):
        C.CAM = IO.read(CM.HERE.parent / "ecro_wl84_104d/protocol.json")["population"]["held_out_view"]
    return i, row


def run_fixed(scene, fit_dir, out, reps, smoke, k=64):
    out.mkdir(parents=True, exist_ok=True)
    rows, raw, compiled, _, _, _ = CM.inputs()
    i, row = setup(scene)
    arms, R = arm_states(fit_dir, len(rows))
    ctx = Fixed104K(row["tag"] + "_" + scene + "_k", compiled["R0"], arms, i)
    light = RT.point_lights(row["light"])
    seed0 = SC["scenes"][scene]
    if smoke:
        nat = H.HA.render(ctx, scene, 4, seed0, "cam", b=4)
        img, chk = ctx.replicate(seed0, 4, 2, light, None, launch=4)
        res = {"e1": S8.l1(img["N|e1"] + img["A0|e1"], nat["e1"]), "e2": S8.l1(img["N|e2"] + img["A0|e2"], nat["e2"]), "chk": chk}
        tp = time.monotonic()
        sd = dr.opaque(mi.UInt64, seed0 * 1000003)
        for c in range(S8.LAUNCH // k):
            ctx.chunk(sd, c * C.RES * C.RES * k, k, light)
        res["full_launch_seconds"] = time.monotonic() - tp
        CM.write(out / "fixed_smoke.json", res)
        print(res, flush=True)
        return
    spp = SI.spp_of(scene)
    log = []
    for r in range(*reps):
        t = time.monotonic()
        img, chk = ctx.replicate(seed0 + r, spp, k, light, None)
        chk["seconds"] = time.monotonic() - t
        log.append(chk)
        np.savez_compressed(out / f"fixed_r{r:03d}.npz", **{q: v.astype(np.float32) for q, v in img.items()})
        print("fixed", scene, r, round(chk["seconds"], 1), flush=True)
    CM.write(out / f"fixed_{reps[0]:02d}_{reps[1]:02d}.json", {"checks": log, "replicates": list(range(*reps)), "chunk_spp": k,
                                                               "commit": CM.commit(), "starts": R})


def paired(metrics, event, x, y, R, roi="objects"):
    dx, dy = [], []
    for r in range(R):
        a, b = metrics[f"{y}_{r}"][roi]["rms"][event], metrics[f"{x}_{r}"][roi]["rms"][event]
        dy.append((a["point"], np.asarray(a["loo"]))); dx.append((b["point"], np.asarray(b["loo"])))
    Y = np.mean([q[0] for q in dy]); X = np.mean([q[0] for q in dx])
    Yl = np.mean([q[1] for q in dy], 0); Xl = np.mean([q[1] for q in dx], 0)
    rel = jk_ci((X - Y) / Y, (Xl - Yl) / Yl)
    rel[f"{y}_rms"], rel[f"{x}_rms"] = float(Y), float(X)
    rel["per_start_rel"] = [float((dx[j][0] - dy[j][0]) / dy[j][0]) for j in range(R)]
    return rel


def analyze(scene, out):
    arch, masks, seen = S8.archived(scene, None)
    ref = {"e1": arch["reference_e1"], "e2": arch["reference_e2"], "h": arch["reference_h"]}
    fx = {}
    for r in range(12):
        with np.load(out / f"fixed_r{r:03d}.npz") as z:
            for q in z.files:
                fx.setdefault(q, []).append(z[q])
    fx = {q: np.stack(v).astype(np.float64) for q, v in fx.items()}
    chk = [c for p in sorted(out.glob("fixed_[0-9]*_*.json")) for c in CM.read(p)["checks"]]
    R = CM.read(sorted(out.glob("fixed_[0-9]*_*.json"))[0])["starts"]
    ident = {"A0_e1": max(S8.l1(fx["N|e1"][r] + fx["A0|e1"][r], arch["R0_e1"][r]) for r in range(12)),
             "A0_e2": max(S8.l1(fx["N|e2"][r] + fx["A0|e2"][r], arch["R0_e2"][r]) for r in range(12)),
             "lane_e1_rel": max(c["e1_A0_minus_native"] / max(c["e1_native_peak"], 1e-30) for c in chk),
             "lane_e2_weight_pdf_rel": max(c["e2_weight_pdf_minus_eval"] / max(c["e2_native_peak"], 1e-30) for c in chk)}
    ok = ident["A0_e1"] <= 2e-5 and ident["A0_e2"] <= 2e-5 and ident["lane_e1_rel"] <= 2e-6 and ident["lane_e2_weight_pdf_rel"] <= 2e-6
    names = [q.split("|")[0] for q in fx if q.endswith("|e1") and not q.startswith("N")]
    roi_masks = {k: masks[k] for k in ROIS if k in masks and masks[k].any()}
    fixed = {}
    for name in names:
        buf = {"e1": fx["N|e1"] + fx[f"{name}|e1"], "e2": fx["N|e2"] + fx[f"{name}|e2"], "h": arch["R0_h"]}
        fixed[name], _ = event_metrics(buf, ref, roi_masks, SI.GATES)
    _, precision = event_metrics({"e1": arch["R0_e1"], "e2": arch["R0_e2"], "h": arch["R0_h"]}, ref, roi_masks, SI.GATES)
    prim = {f"{x}-{y}": {ev: paired(fixed, ev, x, y, R) for ev in ("e1", "e2", "total")} for x, y in (("P", "A"), ("T", "A"), ("P", "T"))}
    maps = {}
    for name in ("A0", "A_0", "T_0", "P_0"):
        for ev in ("e1", "e2"):
            maps[f"{name}_{ev}_err"] = (fx[f"N|{ev}"] + fx[f"{name}|{ev}"] - ref[ev]).mean(0).astype(np.float32)
    for ev in ("e1", "e2"):
        maps[f"P_minus_A_{ev}"] = np.mean([fx[f"P_{r}|{ev}"] - fx[f"A_{r}|{ev}"] for r in range(R)], 0).mean(0).astype(np.float32)
        maps[f"abs_err_reduction_P_vs_A_{ev}"] = (np.abs(np.mean([fx[f"N|{ev}"] + fx[f"A_{r}|{ev}"] for r in range(R)], 0).mean(0) - ref[ev].mean(0))
                                                   - np.abs(np.mean([fx[f"N|{ev}"] + fx[f"P_{r}|{ev}"] for r in range(R)], 0).mean(0) - ref[ev].mean(0))).astype(np.float32)
    maps["reference_e1"] = ref["e1"].mean(0).astype(np.float32)
    maps["reference_e2"] = ref["e2"].mean(0).astype(np.float32)
    np.savez_compressed(out / "maps.npz", **maps)
    secs = [c["seconds"] for c in chk]
    res = {"scene": scene, "identity": ident, "identity_passed": bool(ok), "baseline_precision_failures": precision, "fixed": fixed,
           "primary": prim, "replicate_seconds_median": float(np.median(secs)), "inputs_sha256": seen, "commit": CM.commit()}
    CM.write(out / "analysis.json", res)
    print("analysis", scene, ok, {k: (round(v["total"]["point"], 4), round(v["total"]["lo"], 4), round(v["total"]["hi"], 4)) for k, v in prim.items()},
          flush=True)
    if not ok:
        raise IO.Stop("SCENE_REPLAY_IDENTITY")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("fixed", "analyze"))
    ap.add_argument("scene")
    ap.add_argument("fit", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--reps", default="0:12")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--k", type=int, default=128)
    args = ap.parse_args()
    if args.mode == "fixed":
        run_fixed(args.scene, args.fit, args.out, tuple(int(x) for x in args.reps.split(":")), args.smoke, args.k)
    else:
        analyze(args.scene, args.out)


if __name__ == "__main__":
    main()
