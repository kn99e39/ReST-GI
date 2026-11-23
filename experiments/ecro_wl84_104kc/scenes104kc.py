"""WL104K-C scene stage (protocol stage_D_scenes): new independent scene references and fixed-proposal replays of the WL104K-C arms.

reference : one predeclared new scene (protocol order) through the unchanged scenes104g.study_reference (WL104D held-out camera,
            LAY1024 layered reference + native R0 + masks, 12 x 4096 spp, mask pass 2048 spp at seed + 990), hashed into
            <root>/scenes_new/manifest.json; at most two new scenes qualify; fallbacks in protocol order.
fixed     : WL104K-B fixed-proposal replay (scenes104kb.FixedKB, grouped passes, A0 in every pass) at the scene's seeds.
            NEW WL104K-C scenes: closures P_hist x 4, O0 x 8 (Stage A R8), E2_B x 8 (WL104K-B E2), CR x 8, O0C x 8.
            Previously examined scenes (archived iid_1 / hard_1 / iid_159 and WL104K-B N1 hard_53 / N2 block_186): closures CR x 8,
            O0C x 8 and O0_0, O0_1 (replay-determinism check against the saved WL104K-B images); O0, E2_B and P_hist images are the
            saved WL104K-B replicates (authenticated by evaluation/104K-B/manifest.json raw_sha256).
analyze   : identity gates (A0 vs R0 <= 2e-5 relative L1, lane F / p <= 2e-6 of peak, grouped-pass repeat <= 2e-5; merged scenes also
            native lanes and O0_0 / O0_1 vs the WL104K-B images <= 2e-5), objects stable RMS per closure and event (e1, fixed e2,
            total), X vs O0 contrasts with replicate jackknife t(11), labels IMPROVED (hi < 0) / REGRESSED (lo > +0.02) / NEUTRAL /
            UNRESOLVED_NOISE_FLOOR, peak-pixel concentration, maps.
fixed / analyze require the hashed WL104K-C fit decision and states equal to its fits_sha256 (SCENE_STATE_MISMATCH otherwise).
TOTAL = E1 + fixed E2 + R0 HIGH (HIGH fixed): NOT an end-to-end ECRO validation.

Usage: python scenes104kc.py reference <scene> <root>
       python scenes104kc.py fixed <scene> <root> [--reps a:b] [--smoke]
       python scenes104kc.py analyze <scene> <root>
"""
import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import common104kc as C

KB, CM, GF, KA = C.KB, C.CM, C.GF, C.KA
import scenes104kb as SB  # noqa: E402  (unchanged WL104K-B scene path)

S8, SI, SK = SB.S8, SB.SI, SB.SK
Stop = C.Stop
NEW = dict(C.P.get("seeds", {}).get("scene_seed_bases", {}))          # WL104K-C predeclared new scenes, in order
KB_NEW = dict(SB.NEW)                                                    # WL104K-B new scenes (hard_53_L2, block_186_L2, ...)
KB_SCENES = ("test83_iid_1_L2", "test83_hard_1_L1", "test83_iid_159_L2", "test83_hard_53_L2", "test83_block_186_L2")
EVENTS = SB.EVENTS
CHECK_O0 = (0, 1)


def is_new(scene):
    return scene in NEW


def config(scene):
    if is_new(scene):
        return {"seed": NEW[scene], "spp": SB.NEW_SPP, "held": True, "source": "WL104K-C new"}
    if scene in KB_SCENES:
        c = SB.scene_config(scene)
        return {**c, "source": "WL104K-B"}
    raise Stop(f"SCENE_NOT_PREDECLARED: {scene}")


def new_scene_inputs(scene, root):
    """WL104K-C new reference authenticated by <root>/scenes_new/manifest.json (as scenes104kb.new_scene)."""
    return SB.new_scene(scene, root)


def scene_inputs(scene, root):
    if is_new(scene):
        return new_scene_inputs(scene, root)
    if scene in KB_NEW:
        return SB.new_scene(scene, C.KB_ROOT)
    return S8.archived(scene, None)


# ------------------------------------------------------------------ reference
def run_reference(scene, root):
    if scene not in NEW:
        raise Stop(f"SCENE_NOT_PREDECLARED: {scene}")
    d = root / "scenes_new"
    d.mkdir(parents=True, exist_ok=True)
    man_path = d / "manifest.json"
    man = C.read(man_path) if man_path.exists() else {"batch": "WL104K-C", "order": list(NEW), "scenes": {}}
    done, order = man["scenes"], list(NEW)
    if [s for s, v in done.items() if v.get("status") == "started"]:
        raise Stop("SCENE_REFERENCE_INCOMPLETE")
    if scene in done or (d / scene).exists():
        raise Stop(f"SCENE_REFERENCE_EXISTS: {scene}")
    missing = [s for s in order[:order.index(scene)] if s not in done]
    if missing:
        raise Stop(f"SCENE_ORDER: {scene} before {missing}")
    if sum(v.get("precision_passed") is True for v in done.values()) >= 2:
        raise Stop("NEW_SCENE_LIMIT")
    sys.path.insert(0, str(C.KB_DIR.parent / "ecro_wl84_104g"))
    sys.path.insert(0, str(C.KB_DIR.parent / "ecro_wl84_104f"))
    import scenes104g as G
    cfg = G.CFG
    want = {"replicates": SB.NEW_REPS, "held_spp": SB.NEW_SPP, "held_mask_spp": SB.NEW_MASK_SPP,
            "held_mask_offset": SB.NEW_MASK_OFFSET, "batch": SB.LAUNCH}
    if any(cfg[k] != v for k, v in want.items()):
        raise Stop("SCENE_REFERENCE_CONFIG")
    rows, _, compiled, desc, _, seen = CM.inputs()
    i, row = S8.scene_row(scene)
    t0 = time.monotonic()
    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    done[scene] = {"image": scene, "status": "started", "seed_base": NEW[scene], "created": created, "commit": C.commit()}
    C.write(man_path, man)
    try:
        _, _, ready = G.study_reference(row, desc[i], compiled["R0"][i], NEW[scene], d / scene)
        files = {f"{scene}/{p.name}": C.sha(p) for p in sorted((d / scene).iterdir()) if p.is_file()}
    except Exception as e:
        done[scene] = {"image": scene, "status": "error", "precision_passed": False, "error": f"{type(e).__name__}: {e}",
                       "seed_base": NEW[scene], "created": created, "commit": C.commit(), "seconds": time.monotonic() - t0}
        C.write(man_path, man)
        raise
    done[scene] = {"image": scene, "status": "complete", "material": row["tag"], "cohort": row["cohort"], "seed_base": NEW[scene],
                   "precision_passed": bool(ready["precision_passed"]), "failures": ready["failures"], "spp": SB.NEW_SPP,
                   "replicates": SB.NEW_REPS, "mask_spp": SB.NEW_MASK_SPP, "mask_seed": NEW[scene] + SB.NEW_MASK_OFFSET,
                   "camera": ready["camera"], "reference": "scenes104g.study_reference, LAY1024", "files": files,
                   "inputs_sha256": seen, "protocol_sha256": C.sha(C.PROTOCOL), "seconds": time.monotonic() - t0,
                   "commit": C.commit(), "created": created}
    C.write(man_path, man)
    print(("QUALIFIED " if ready["precision_passed"] else "UNQUALIFIED ") + scene, flush=True)


# ------------------------------------------------------------------ closures
def fit_decision(root):
    fd, side = root / "fit_decision/fit_decision.json", root / "fit_decision/fit_decision.json.sha256"
    if not (fd.exists() and side.exists()) or C.sha(fd) != side.read_text(encoding="utf-8").split()[0]:
        raise Stop("LEAK_GUARD: no hashed WL104K-C fit decision")
    return C.read(fd), C.sha(fd)


def closure_set(root, scene, fd):
    n = C.N_MAT
    arms = {"O0": root / "stageA/fits/R8", "CR": root / "fits/CR", "O0C": root / "fits/O0C"}
    out, prov = {}, {}
    for a, d in arms.items():
        h = C.sha(d / "states.npz")
        if h != fd["fits_sha256"][a]:
            raise Stop(f"SCENE_STATE_MISMATCH: {a}")
        st = C.load_states(d / "states.npz", 8)
        prov[a] = {"states_sha256": h, "start_keys": KB.START_KEYS}
        rs = range(8) if (is_new(scene) or a != "O0") else CHECK_O0
        for r in rs:
            out[f"{a}_{r}"] = ("P", SB.compiled_np(st[r]["y"]), st[r]["g"])
    if is_new(scene):
        seen = C.authenticate_kb(["primary/fits/E2/states.npz"])
        st = C.load_states(C.KB_ROOT / "fits/E2/states.npz", 8)
        for r in range(8):
            out[f"E2_B_{r}"] = ("P", SB.compiled_np(st[r]["y"]), st[r]["g"])
        prov["E2_B"] = {"states_sha256": list(seen.values())[0], "start_keys": KB.START_KEYS}
        hist, _, fj = KA.states(C.K_FIT, n)
        for r in range(len(fj["start_keys"])):
            out[f"P_hist_{r}"] = ("P", SB.compiled_np(hist["P"][r]["y"]), hist["P"][r]["g"])
        prov["P_hist"] = {"start_keys": fj["start_keys"]}
    return out, prov


def run_fixed(scene, root, reps, smoke):
    out = root / "scenes" / scene
    out.mkdir(parents=True, exist_ok=True)
    cfg = config(scene)
    fd, fd_sha = fit_decision(root)
    seen = scene_inputs(scene, root)[2] if (is_new(scene) or scene in KB_NEW) else {}
    rows, _, compiled, _, _, seen_in = CM.inputs()
    seen.update(seen_in)
    i, row = S8.scene_row(scene)
    cl, prov = closure_set(root, scene, fd)
    if cfg["held"]:
        SB.C.CAM = SB.held_camera()
    ctx = SB.FixedKB(row["tag"] + "_" + scene + "_kc", compiled["R0"], cl, i)
    light = SB.RT.point_lights(row["light"])
    groups = SB.groups_of(list(cl), 8)
    seed0 = cfg["seed"]
    log = {"scene": scene, "closures": ["A0", *[q for g in groups for q in g]], "groups": groups, "chunk_spp": 64, "spp": cfg["spp"],
           "seed_base": seed0, "camera": SB.C.CAM, "fits": prov, "inputs_sha256": seen, "fit_decision_sha256": fd_sha,
           "source": cfg["source"], "commit": C.commit()}
    checks = []
    spp = 4 if smoke else cfg["spp"]
    for r in range(*reps):
        t = time.monotonic()
        img, chk = ctx.replicate_groups(seed0 + r, spp, 2 if smoke else 64, light, groups, **({"launch": 4} if smoke else {}))
        chk["seconds"] = time.monotonic() - t
        checks.append(chk)
        np.savez_compressed(out / f"fixed_r{r:03d}.npz", **{q: v.astype(np.float32) for q, v in img.items()})
        print("fixed", scene, r, round(chk["seconds"], 1), flush=True)
    C.write(out / f"fixed_{reps[0]:02d}_{reps[1]:02d}.json", {**log, "checks": checks, "replicates": list(range(*reps)),
                                                             "smoke": smoke, "allocator": SB.allocator()})


# ------------------------------------------------------------------ analysis
def load_replicates(d, files=None):
    fx = {}
    for r in range(12):
        p = d / f"fixed_r{r:03d}.npz"
        with np.load(p) as z:
            for q in z.files:
                fx.setdefault(q, []).append(z[q])
    return {q: np.stack(v) for q, v in fx.items()}


def kb_replicates(scene):
    """Saved WL104K-B replicate images of a previously examined scene, authenticated; O0 / E2 / P_hist closures only."""
    rels = [f"primary/scenes/{scene}/fixed_r{r:03d}.npz" for r in range(12)]
    seen = C.authenticate_kb(rels)
    fx = load_replicates(C.KB_ROOT / "scenes" / scene)
    keep = {}
    for q, v in fx.items():
        name = q.split("|")[0]
        if q.startswith("N|") or q.startswith("A0|"):
            keep["KB_" + q] = v
        elif name.startswith("O0_") or name.startswith("P_hist_"):
            keep[q] = v
        elif name.startswith("E2_"):
            keep["E2_B_" + name.split("_")[1] + "|" + q.split("|")[1]] = v
    return keep, seen


def analyze(scene, root, smoke=False):
    t0 = time.monotonic()
    fd, fd_sha = fit_decision(root)
    out = root / "scenes" / scene
    arch, masks, seen = scene_inputs(scene, root)
    ref = {"e1": arch["reference_e1"], "e2": arch["reference_e2"], "h": arch["reference_h"]}
    logs = [C.read(p) for p in sorted(out.glob("fixed_[0-9]*_*.json"))]
    if sorted(r for lg in logs for r in lg["replicates"]) != list(range(12)) or (any(lg.get("smoke") for lg in logs) and not smoke):
        raise Stop("SCENE_REPLICATES_INCOMPLETE")
    if any(lg["fit_decision_sha256"] != fd_sha for lg in logs):
        raise Stop("SCENE_STATE_MISMATCH: replays not made under the current fit decision")
    chk = [c for lg in logs for c in lg["checks"]]
    fx = load_replicates(out)
    ident = {"A0_e1": max(S8.l1(fx["N|e1"][r] + fx["A0|e1"][r], arch["R0_e1"][r]) for r in range(12)),
             "A0_e2": max(S8.l1(fx["N|e2"][r] + fx["A0|e2"][r], arch["R0_e2"][r]) for r in range(12)),
             "lane_e1_rel": max(c["e1_A0_minus_native"] / max(c["e1_native_peak"], 1e-30) for c in chk),
             "lane_e2_weight_pdf_rel": max(c["e2_weight_pdf_minus_eval"] / max(c["e2_native_peak"], 1e-30) for c in chk),
             "grouped_pass_repeat_rel_l1": max(c.get("grouped_pass_repeat_rel_l1", 0.0) for c in chk)}
    merged = not is_new(scene)
    kb_seen = {}
    if merged:
        kb, kb_seen = kb_replicates(scene)
        for ev in ("e1", "e2"):
            ident[f"native_{ev}_vs_WL104KB"] = max(S8.l1(fx[f"N|{ev}"][r], kb[f"KB_N|{ev}"][r]) for r in range(12))
            for r0 in CHECK_O0:
                ident[f"O0_{r0}_{ev}_vs_WL104KB"] = max(S8.l1(fx[f"N|{ev}"][r] + fx[f"O0_{r0}|{ev}"][r],
                                                              kb[f"KB_N|{ev}"][r] + kb[f"O0_{r0}|{ev}"][r]) for r in range(12))
        for q, v in kb.items():
            if not q.startswith("KB_") and q not in fx:
                fx[q] = v
    ok = (ident["A0_e1"] <= SB.IDENTITY_L1 and ident["A0_e2"] <= SB.IDENTITY_L1 and ident["lane_e1_rel"] <= SB.LANE_REL
          and ident["lane_e2_weight_pdf_rel"] <= SB.LANE_REL and ident["grouped_pass_repeat_rel_l1"] <= SB.IDENTITY_L1
          and all(v <= SB.IDENTITY_L1 for k, v in ident.items() if k.endswith("_vs_WL104KB")))
    res = {"scene": scene, "source": config(scene)["source"], "new_independent": is_new(scene), "identity": ident,
           "identity_passed": bool(ok), "merged_with_WL104KB_images": merged,
           "scope": "fixed-proposal replay: TOTAL = E1 + fixed E2 + R0 HIGH (HIGH fixed); not an end-to-end ECRO validation",
           "inputs_sha256": {**seen, **kb_seen}, "fit_decision_sha256": fd_sha, "commit": C.commit()}
    if not ok and smoke:
        res["smoke_identity_ignored"] = True
    elif not ok:
        C.write(out / "analysis104kc.json", {**res, "stop": "SCENE_REPLAY_IDENTITY"})
        raise Stop("SCENE_REPLAY_IDENTITY")
    names = sorted({q.split("|")[0] for q in fx if q.endswith("|e1") and not q.startswith("N|")})
    roi_masks = {q: masks[q] for q in SB.ROIS if q in masks and masks[q].any()}
    h = arch["R0_h"]
    M = {}
    for name in names:
        buf = {ev: fx[f"N|{ev}"].astype(np.float64) + fx[f"{name}|{ev}"] for ev in ("e1", "e2")}
        M[name], _ = SB.event_metrics({**buf, "h": h}, ref, roi_masks, SB.GATES)
    _, precision = SB.event_metrics({"e1": arch["R0_e1"], "e2": arch["R0_e2"], "h": h}, ref, roi_masks, SB.GATES)
    starts = {}
    for name in names:
        if name != "A0":
            a, r = name.rsplit("_", 1)
            starts.setdefault(a, []).append(int(r))
    starts = {a: sorted(v) for a, v in starts.items()}
    summary = {}
    for a, rs in starts.items():
        cl = [f"{a}_{r}" for r in rs]
        summary[a] = {"n_starts": len(rs), "objects_rms": {}, "energy": {}}
        for ev in EVENTS:
            v = [M[q]["objects"]["rms"][ev]["point"] for q in cl]
            summary[a]["objects_rms"][ev] = {"start_mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else None,
                                             "per_start": [float(x) for x in v]}
        for roi in roi_masks:
            summary[a]["energy"][roi] = {ev: float(np.mean([M[q][roi]["energy"][ev]["point"] for q in cl])) for ev in EVENTS}
    o0 = starts["O0"]
    pairs = {"CR": "O0", "O0C": "O0", "E2_B": "O0", "CR_vs_O0C": "O0C"}
    contrasts = {}
    for key, base in pairs.items():
        a = key.split("_vs_")[0]
        if a not in starts or base not in starts:
            continue
        common = [r for r in starts[a] if r in starts[base]]
        contrasts[key] = {ev: SB.rel_change(M, ev, [f"{a}_{r}" for r in common], [f"{base}_{r}" for r in common]) for ev in EVENTS}
    desc = {}
    if "P_hist" in starts:
        common = [r for r in starts["P_hist"] if r in o0]               # P_hist start keys = O0 start keys 1-4 (WL104K-B)
        desc["P_hist"] = {ev: SB.rel_change(M, ev, [f"P_hist_{r}" for r in common], [f"O0_{r}" for r in common]) for ev in EVENTS}
    # absolute errors (always reported, also where a relative change is undefined)
    absolute = {a: {ev: summary[a]["objects_rms"][ev]["start_mean"] for ev in EVENTS} for a in summary}
    refl = {"e1": ref["e1"].mean(0) @ KB.LUMA, "total": (ref["e1"] + ref["e2"] + ref["h"]).mean(0) @ KB.LUMA}
    h_mean = h.mean(0)
    lum = {q: SB.lum_images(fx, q, h_mean) for q in names}
    peak = {a: {str(r): SB.concentration(lum[f"{a}_{r}"], lum[f"O0_{r}"], refl, masks) for r in starts[a] if r in o0}
            for a in ("CR", "O0C", "E2_B") if a in starts}
    maps = {"reference_e1_lum": refl["e1"], "reference_total_lum": refl["total"], "objects_mask": masks["objects"]}
    for a, rs in starts.items():
        for ev in ("e1", "total"):
            m = np.mean([lum[f"{a}_{r}"][ev] for r in rs], 0)
            maps[f"{a}_{ev}_lum"] = m
            maps[f"{a}_{ev}_abs_err"] = np.abs(m - refl[ev])
    np.savez_compressed(out / "maps.npz", **{q: v.astype(np.float32) if v.dtype != bool else v for q, v in maps.items()})
    res.update({"contrasts": contrasts, "contrasts_descriptive_P_hist": desc, "arm_summary": summary, "absolute_objects_rms": absolute,
                "per_closure": SB.strip(M), "peak_pixels": peak, "baseline_precision_failures": precision,
                "replicate_seconds_median": float(np.median([c["seconds"] for c in chk])),
                "label_rule": "relative change of start-mean objects stable RMS vs the base arm (paired starts), jackknife t(11) 95 %: "
                              "IMPROVED hi < 0, REGRESSED lo > +0.02, otherwise NEUTRAL; UNRESOLVED_NOISE_FLOOR when the base RMS "
                              "is clipped at 0", "seconds": time.monotonic() - t0})
    C.write(out / ("analysis_smoke.json" if smoke else "analysis104kc.json"), res)
    print("analysis", scene, {k: (v["total"]["point"], v["total"]["label"]) for k, v in contrasts.items()}, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("reference", "fixed", "analyze"))
    ap.add_argument("scene")
    ap.add_argument("root", type=Path)
    ap.add_argument("--reps", default="0:12")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.mode == "reference":
        run_reference(a.scene, a.root)
    elif a.mode == "fixed":
        run_fixed(a.scene, a.root, tuple(int(x) for x in a.reps.split(":")), a.smoke)
    else:
        analyze(a.scene, a.root, a.smoke)


if __name__ == "__main__":
    main()
