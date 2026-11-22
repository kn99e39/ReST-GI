"""WL104K-B Stage G (protocol stage_G_scenes): new held-out scene references and fixed-proposal scene replay of every fitted arm.

reference : one predeclared new scene (N1 test83_hard_53_L2, N2 test83_block_186_L2, fallbacks test83_block_161_L1 /
            test83_hard_76_L1, in that order, at most two qualified) through the unchanged scenes104g.study_reference (WL104D held-out
            camera, LAY1024 layered reference + native R0 (WL83 ER_DF compiled state) + masks, 12 x 4096 spp, mask pass 2048 spp at
            seed + 990) into <root>/scenes_new/<scene>/; every produced file is hashed into <root>/scenes_new/manifest.json.
fixed     : WL104K / WL104I fixed-proposal replay at the scene's seeds (archived WL104H seeds, or the new seed base). Closures A0 (R0
            identity), A_hist_r / P_hist_r (WL104K 12k states, 4 starts) and <arm>_r for every physical fit in <root>/fits/{O0,O1,D,E2,F}
            (P family through common104k.DrSDM, A through closure83_dr.DrClosure83). Closures are evaluated in GROUPED PASSES over the same
            replicate (identical seeds and launch layout; A0 and the native lanes are re-evaluated in every pass and must agree).
analyze   : identity gates (A0 vs R0 <= 2e-5 relative L1, lane F / p <= 2e-6 of peak, grouped-pass repeat <= 2e-5), per-closure objects
            stable RMS and region signed energy, arm start-mean / SD, X vs O0 contrasts with replicate jackknife t(11) and the protocol
            labels, E1-selected starts from the hashed dev_eval/dev.json (+ descriptive companion: selected O0 start vs the mean of the
            other 7 starts), peak-pixel squared-error concentration (descriptive), maps for figures.
fixed / analyze require the hashed dev decision (LEAK_GUARD; --smoke alone may bypass it when none exists) and replayed states equal
to dev fits_sha256 / fit_json_sha256 (F: dev stage_F run with its spec; test_eval fits_sha256 when present), else SCENE_STATE_MISMATCH.
reference writes a 'started' manifest entry before rendering; an exception is recorded as status 'error' (not qualified) so the
predeclared order continues; a leftover 'started' entry stops every later reference call (SCENE_REFERENCE_INCOMPLETE).
TOTAL = E1 + fixed E2 + R0 HIGH (HIGH fixed from R0): NOT an end-to-end ECRO validation.

Usage: python scenes104kb.py reference <scene> <root>
       python scenes104kb.py fixed <scene> <root> <out_dir> [--reps a:b] [--k 64] [--group 8] [--smoke] [--dup N]
       python scenes104kb.py analyze <scene> <root> <out_dir> [--smoke]
"""
import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import common104kb as KB

CM, IO, GF, KA = KB.CM, KB.IO, KB.GF, KB.KA
sys.path.insert(0, str(KB.HERE.parent / "ecro_wl84_104k"))
import scenes104k as SK  # noqa: E402  (WL104K fixed-proposal scene path: SI.FixedPass, S8.archived / scene_row, metrics)

SI, S8 = SK.SI, SK.S8
H, C, RT, mi, dr = SK.H, SK.C, SK.RT, SK.mi, SK.dr
event_metrics, jk_ci = SK.event_metrics, SK.jk_ci

ARCHIVED = dict(SI.SC["scenes"])                               # WL104I / WL104K archived scene seeds
NEW = dict(KB.P["seeds"]["scene_seed_bases"])                  # predeclared order: N1, N2, fallback 1, fallback 2
NEW_SPP, NEW_REPS, NEW_MASK_SPP, NEW_MASK_OFFSET, LAUNCH = 4096, 12, 2048, 990, 512
INDEPENDENT = {"test83_iid_159_L2", *NEW}
PHYS = ("O0", "O1", "D", "E2", "F")
ARM_META = {"O0": ("adam", "original", "original"), "O1": ("lbfgs", "original", "original"),
            "D": ("adam", "coverage", "original"), "E2": ("adam", "original", "e2")}
GATES, ROIS = SI.GATES, SI.ROIS
EVENTS = ("e1", "e2", "total")
IDENTITY_L1, LANE_REL = 2e-5, 2e-6
PEAK_REGIONS = ("highlight_core", "highlight_ring", "sphere_lower", "grazing", "cube")
Stop = KB.Stop


# ------------------------------------------------------------------ scenes
def held_camera():
    return IO.read(KB.HERE.parent / "ecro_wl84_104d/protocol.json")["population"]["held_out_view"]


def scene_config(scene):
    if scene in ARCHIVED:
        return {"seed": ARCHIVED[scene], "spp": SI.spp_of(scene), "held": "held" in SI.SC_ROLE(scene), "new": False}
    if scene in NEW:
        return {"seed": NEW[scene], "spp": NEW_SPP, "held": True, "new": True}
    raise Stop(f"SCENE_NOT_PREDECLARED: {scene}")


def new_scene(scene, root):
    """New reference authenticated by the WL104K-B manifest (iid_159 held-out layout: r###.npz + region_masks.npz)."""
    d = root / "scenes_new"
    entry = KB.read(d / "manifest.json")["scenes"].get(scene) if (d / "manifest.json").exists() else None
    if entry is None:
        raise Stop(f"NO_FROZEN_HASH: scenes_new/{scene}")
    if entry.get("status") != "complete":
        raise Stop(f"SCENE_REFERENCE_UNQUALIFIED: {scene} (manifest status {entry.get('status')})")
    files = [d / scene / f"r{i:03d}.npz" for i in range(NEW_REPS)]
    masks_path, ready_path = d / scene / "region_masks.npz", d / scene / "readiness.json"
    seen = {}
    for p in (*files, masks_path, ready_path):
        rel = f"{scene}/{p.name}"
        if rel not in entry["files"]:
            raise Stop(f"NO_FROZEN_HASH: scenes_new/{rel}")
        h = KB.sha(p)
        if h != entry["files"][rel]:
            raise Stop(f"FROZEN_HASH_MISMATCH: scenes_new/{rel}")
        seen[f"scenes_new/{rel}"] = h
    if not KB.read(ready_path)["precision_passed"] or not entry["precision_passed"]:
        raise Stop(f"SCENE_REFERENCE_UNQUALIFIED: {scene}")
    ev = {k: [] for k in ("reference_e1", "reference_e2", "reference_h", "R0_e1", "R0_e2", "R0_h")}
    for f in files:
        with np.load(f) as z:
            for k in ev:
                ev[k].append(z[k])
    out = {k: np.stack(v).astype(np.float64) for k, v in ev.items()}
    with np.load(masks_path) as z:
        masks = {k: z[k].astype(bool) for k in z.files}
    return out, masks, seen


def scene_inputs(scene, root):
    return new_scene(scene, root) if scene_config(scene)["new"] else S8.archived(scene, None)


def run_reference(scene, root):
    if scene not in NEW:
        raise Stop(f"SCENE_NOT_PREDECLARED: {scene}")
    d = root / "scenes_new"
    d.mkdir(parents=True, exist_ok=True)
    man_path = d / "manifest.json"
    man = KB.read(man_path) if man_path.exists() else {"batch": "WL104K-B", "order": list(NEW), "scenes": {}}
    done, order = man["scenes"], list(NEW)
    started = [s for s, v in done.items() if v.get("status") == "started"]
    if started:                                                # a run died without completion / error record: never resumed silently
        raise Stop(f"SCENE_REFERENCE_INCOMPLETE: {[(s, 'readiness.json present' if (d / s / 'readiness.json').exists() else 'no readiness') for s in started]}")
    if scene in done or (d / scene).exists():
        raise Stop(f"SCENE_REFERENCE_EXISTS: {scene}")
    missing = [s for s in order[:order.index(scene)] if s not in done]
    if missing:
        raise Stop(f"SCENE_ORDER: {scene} before {missing}")
    if sum(v.get("precision_passed") is True for v in done.values()) >= 2:
        raise Stop("NEW_SCENE_LIMIT: two new scenes already qualified")
    sys.path.insert(0, str(KB.HERE.parent / "ecro_wl84_104g"))
    sys.path.insert(0, str(KB.HERE.parent / "ecro_wl84_104f"))
    import scenes104g as G
    cfg = G.CFG
    want = {"replicates": NEW_REPS, "held_spp": NEW_SPP, "held_mask_spp": NEW_MASK_SPP, "held_mask_offset": NEW_MASK_OFFSET,
            "batch": LAUNCH}
    if any(cfg[k] != v for k, v in want.items()):
        raise Stop(f"SCENE_REFERENCE_CONFIG: {[(k, cfg[k], v) for k, v in want.items()]}")
    rows, _, compiled, desc, _, seen = CM.inputs()
    i, row = S8.scene_row(scene)
    if rows[i]["image"] != scene:
        raise Stop("SCENE_ROW_ORDER")
    t0 = time.monotonic()
    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    done[scene] = {"image": scene, "status": "started", "seed_base": NEW[scene], "created": created, "commit": KB.commit()}
    KB.write(man_path, man)                                    # recorded before rendering (crash -> 'error'; kill -> 'started')
    try:
        _, _, ready = G.study_reference(row, desc[i], compiled["R0"][i], NEW[scene], d / scene)
        files = {f"{scene}/{p.name}": KB.sha(p) for p in sorted((d / scene).iterdir()) if p.is_file()}
    except Exception as e:                                     # not qualified; the predeclared order continues with the next scene
        done[scene] = {"image": scene, "status": "error", "precision_passed": False, "error": f"{type(e).__name__}: {e}",
                       "seed_base": NEW[scene], "created": created, "commit": KB.commit(), "seconds": time.monotonic() - t0,
                       "failed": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        KB.write(man_path, man)
        print("UNQUALIFIED " + scene + " (reference error)", flush=True)
        raise
    done[scene] = {"image": scene, "status": "complete", "material": row["tag"], "cohort": row["cohort"], "seed_base": NEW[scene],
                   "precision_passed": bool(ready["precision_passed"]), "failures": ready["failures"], "spp": NEW_SPP,
                   "replicates": NEW_REPS, "mask_spp": NEW_MASK_SPP, "mask_seed": NEW[scene] + NEW_MASK_OFFSET, "camera": ready["camera"],
                   "R0_state": "WL83 ER_DF compiled state (evaluation/104D/amendment1/compiled_states.npz, = WL104G response R0)",
                   "reference": "scenes104g.study_reference, LAY1024 (data81 layered walk, depth 1024)", "files": files,
                   "inputs_sha256": seen, "protocol_sha256": KB.sha(KB.PROTOCOL), "seconds": time.monotonic() - t0,
                   "commit": KB.commit(), "created": created,
                   "completed": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    KB.write(man_path, man)
    print(("QUALIFIED " if ready["precision_passed"] else "UNQUALIFIED ") + scene, flush=True)


# ------------------------------------------------------------------ closures
def compiled_np(y):
    comp = GF.compile_states(y)
    return comp.cpu().numpy() if torch.is_tensor(comp) else np.asarray(comp)


def closure_set(root, n):
    """name -> (kind, compiled [n, 69], gates [n, 8, 3, 5] | None) and provenance per arm."""
    seen = KA.authenticate_k()
    hist, R, fj = KA.states(KA.K_ROOT / "fit_primary", n)
    out = {}
    prov = {"A_hist": {"start_keys": fj["start_keys"], "inputs_sha256": seen}, "P_hist": {"start_keys": fj["start_keys"]}}
    for r in range(R):
        out[f"A_hist_{r}"] = ("A", compiled_np(hist["A"][r]["y"]), None)
    for r in range(R):
        out[f"P_hist_{r}"] = ("P", compiled_np(hist["P"][r]["y"]), hist["P"][r]["g"])
    for arm in PHYS:
        d = root / "fits" / arm
        if not (d / "states.npz").exists():
            continue
        fa = KB.read(d / "fit.json")
        meta = fa["arm"]
        if meta.get("synthetic") or (arm in ARM_META and (meta["method"], meta["data"], meta["objective"]) != ARM_META[arm]):
            raise Stop(f"FIT_ARM_MISMATCH: {arm} {meta}")
        if fa["start_keys"] != KB.START_KEYS:
            raise Stop(f"FIT_START_KEYS: {arm} {fa['start_keys']} != {KB.START_KEYS}")
        Ra = len(fa["start_keys"])
        with np.load(d / "states.npz") as z:
            if z["P_y"].shape != (Ra * n, 58) or z["P_g"].shape != (Ra * n, 8, 3, 5):
                raise Stop(f"STATE_LAYOUT: {arm}")
        st, _ = KB.EV.load_states(d / "states.npz", Ra, n)
        for r in range(Ra):
            out[f"{arm}_{r}"] = ("P", compiled_np(st["P"][r]["y"]), st["P"][r]["g"])
        prov[arm] = {"start_keys": fa["start_keys"], "arm": meta, "states_sha256": KB.sha(d / "states.npz"),
                     "fit_json_sha256": KB.sha(d / "fit.json")}
    return out, prov


def dev_decision(root, smoke=False):
    """The hashed dev decision (as eval104kb.leak_guard). Only --smoke may run without one (recorded as bypassed)."""
    dp, sp = root / "dev_eval/dev.json", root / "dev_eval/dev.json.sha256"
    if not dp.exists() or not sp.exists():
        if smoke:
            return None, {"available": False, "bypassed": "smoke only: no hashed dev decision"}
        raise Stop("LEAK_GUARD: no hashed dev decision (dev_eval/dev.json + dev.json.sha256)")
    h = KB.sha(dp)
    if h != sp.read_text(encoding="utf-8").split()[0].strip():
        raise Stop("LEAK_GUARD: dev.json does not match dev.json.sha256")
    return KB.read(dp), {"available": True, "authenticated": True, "sha256": h}


def verify_states(root, prov, dev):
    """Replayed physical states == states hashed in the dev decision (fits_sha256, fit_json_sha256); F only if eligible at dev
    with the dev spec; and == the test evaluation's fits_sha256 when test_eval/test.json exists. Mismatch -> SCENE_STATE_MISMATCH."""
    phys = sorted(a for a in prov if a in PHYS)
    bad = []
    dev_arms = sorted(dev["fits_sha256"])
    if [a for a in phys if a != "F"] != dev_arms:
        bad.append(f"replayed arms {[a for a in phys if a != 'F']} != dev arms {dev_arms}")
    for a in [a for a in phys if a != "F"]:
        if prov[a]["states_sha256"] != dev["fits_sha256"].get(a):
            bad.append(f"{a} states.npz != dev fits_sha256")
        fj = dev.get("fit_json_sha256", {}).get(a)
        if fj is not None and prov[a]["fit_json_sha256"] != fj:
            bad.append(f"{a} fit.json != dev fit_json_sha256")
    sf = dev["stage_F"]
    if "F" in phys:
        spec = sf.get("spec") or {}
        got = tuple(prov["F"]["arm"].get(k) for k in ("method", "data", "objective"))
        if not sf.get("run") or got != tuple(spec.get(k) for k in ("method", "data", "objective")):
            bad.append(f"F {got} not the dev stage_F spec (run {sf.get('run')}, spec {spec or None})")
    elif sf.get("run"):
        bad.append("F eligible at dev but fits/F not replayed")
    tp, test = root / "test_eval/test.json", None
    if tp.exists():
        test = KB.read(tp)
        th = test.get("fits_sha256", {})
        if sorted(a for a in th if a in PHYS) != phys:
            bad.append(f"replayed arms {phys} != test arms {sorted(a for a in th if a in PHYS)}")
        bad += [f"{a} states.npz != test fits_sha256" for a in phys if th.get(a) != prov[a]["states_sha256"]]
    if bad:
        raise Stop("SCENE_STATE_MISMATCH: " + "; ".join(bad))
    return {"checked_arms": phys, "dev_fits_sha256": True, "dev_fit_json_sha256": bool(dev.get("fit_json_sha256")),
            "F": "dev stage_F spec" if "F" in phys else "not run at dev",
            "test_eval": {"checked": test is not None, **({"sha256": KB.sha(tp)} if test is not None else {})}}


class FixedKB(SI.FixedPass):
    """WL104K Fixed104K with closures evaluated in groups; A0 (identity + lane checks) is part of every group."""

    def __init__(self, tag, r0_states, closures, i):
        super().__init__(tag, {"A0": r0_states}, i)
        self.a0 = self.cl["A0"]
        self.all = {name: H.CD.DrClosure83("ER", comp[i]) if kind == "A" else CM.DrSDM(comp[i], g[i])
                    for name, (kind, comp, g) in closures.items()}

    def use(self, names):
        self.cl = {"A0": self.a0, **{q: self.all[q] for q in names}}

    def replicate_groups(self, seed, spp, k, light, groups, launch=S8.LAUNCH):
        img, chk, rep = {}, {}, 0.0
        for names in groups:
            self.use(names)
            im, ch = self.replicate(seed, spp, k, light, None, launch=launch)
            for q, v in im.items():
                if q in img:
                    rep = max(rep, S8.l1(v, img[q]))
                else:
                    img[q] = v
            for q, v in ch.items():
                chk[q] = max(chk.get(q, 0.0), v)
        chk["grouped_pass_repeat_rel_l1"] = rep
        return img, chk


def groups_of(names, size):
    rest = [q for q in names if q != "A0"]
    return [rest[j:j + size] for j in range(0, len(rest), size)] or [[]]


def allocator():
    return [ln.strip() for ln in dr.whos_str().splitlines() if "peak" in ln]


def run_fixed(scene, root, out, reps, smoke, k, group, dup):
    out.mkdir(parents=True, exist_ok=True)
    cfg = scene_config(scene)
    seen = new_scene(scene, root)[2] if cfg["new"] else {}
    rows, _, compiled, _, _, seen_in = CM.inputs()
    seen.update(seen_in)
    i, row = S8.scene_row(scene)
    cl, prov = closure_set(root, len(rows))
    dev, dev_info = dev_decision(root, smoke)
    state_check = verify_states(root, prov, dev) if dev is not None else {"bypassed": "smoke only: no hashed dev decision"}
    if dup:
        if not smoke:
            raise Stop("--dup is a smoke-only memory / time emulation")
        base = [q for q in cl if q.startswith("O0_")]
        for j in range(dup):
            cl.update({f"O0dup{j}_{q.split('_')[1]}": cl[q] for q in base})
    if cfg["held"]:
        C.CAM = held_camera()
    ctx = FixedKB(row["tag"] + "_" + scene + "_kb", compiled["R0"], cl, i)
    light = RT.point_lights(row["light"])
    groups = groups_of(list(cl), group)
    seed0 = cfg["seed"]
    log = {"scene": scene, "closures": ["A0", *[q for g in groups for q in g]], "groups": groups, "chunk_spp": k, "spp": cfg["spp"],
           "seed_base": seed0, "camera": C.CAM, "fits": prov, "grouped_passes": len(groups) > 1, "inputs_sha256": seen,
           "dev_decision": dev_info, "state_check": state_check, "commit": KB.commit()}
    if smoke:
        nat = H.HA.render(ctx, scene, 4, seed0, "cam", b=4)
        img, chk = ctx.replicate_groups(seed0, 4, 2, light, groups, launch=4)
        res = {**log, "smoke_e1_A0_vs_native": S8.l1(img["N|e1"] + img["A0|e1"], nat["e1"]),
               "smoke_e2_A0_vs_native": S8.l1(img["N|e2"] + img["A0|e2"], nat["e2"]), "smoke_checks": chk, "replicates": []}
        for r in range(*reps):
            t = time.monotonic()
            img, chk = ctx.replicate_groups(seed0 + r, cfg["spp"], k, light, groups)
            chk["seconds"] = time.monotonic() - t
            res["replicates"].append({"replicate": r, **chk})
            print("smoke replicate", r, round(chk["seconds"], 1), flush=True)
        res["allocator"] = allocator()
        res["n_closures"] = len(log["closures"])
        KB.write(out / "fixed_smoke.json", res)
        print({q: v for q, v in res.items() if q not in ("fits", "inputs_sha256", "groups", "closures")}, flush=True)
        return
    checks = []
    for r in range(*reps):
        t = time.monotonic()
        img, chk = ctx.replicate_groups(seed0 + r, cfg["spp"], k, light, groups)
        chk["seconds"] = time.monotonic() - t
        checks.append(chk)
        np.savez_compressed(out / f"fixed_r{r:03d}.npz", **{q: v.astype(np.float32) for q, v in img.items()})
        print("fixed", scene, r, round(chk["seconds"], 1), flush=True)
    KB.write(out / f"fixed_{reps[0]:02d}_{reps[1]:02d}.json",
             {**log, "checks": checks, "replicates": list(range(*reps)), "allocator": allocator()})


# ------------------------------------------------------------------ analysis
def strip(o):
    if isinstance(o, dict):
        return {k: strip(v) for k, v in o.items() if k != "loo"}
    return o


def label(ci):
    return "IMPROVED" if ci["hi"] < 0 else "REGRESSED" if ci["lo"] > 0.02 else "NEUTRAL"


def rms_of(M, name, ev, roi="objects"):
    a = M[name][roi]["rms"][ev]
    return a["point"], np.asarray(a["loo"])


def rel_change(M, ev, xs, ys):
    """Relative change of the mean objects stable RMS over closures xs vs ys (paired lists), replicate jackknife t(11)."""
    X = [rms_of(M, q, ev) for q in xs]
    Y = [rms_of(M, q, ev) for q in ys]
    Xp, Yp = np.mean([a[0] for a in X]), np.mean([a[0] for a in Y])
    Xl, Yl = np.mean([a[1] for a in X], 0), np.mean([a[1] for a in Y], 0)
    out = jk_ci((Xp - Yp) / Yp, (Xl - Yl) / Yl)
    out["x_rms"], out["o0_rms"], out["n_starts"] = float(Xp), float(Yp), len(xs)
    if len(xs) == len(ys):
        per = [(a[0] - b[0]) / b[0] for a, b in zip(X, Y)]
        out["per_start_rel"] = [float(v) for v in per]
        out["per_start_rel_sd"] = float(np.std(per, ddof=1)) if len(per) > 1 else None
    out["label"] = label(out)
    return out


def dev_selections(dev, i, tag):
    """{arm: dev-selected start index for material i} from the eval104kb dev.json 'selections' layout ONLY:
    {arm: [start index per material]} (one entry per dev material, int in [0, len(START_KEYS))). Anything else raises."""
    mats, sel = dev.get("materials"), dev.get("selections")
    if not isinstance(mats, list) or not isinstance(sel, dict) or not sel:
        raise Stop("DEV_SELECTIONS_FORMAT: dev.json 'materials' / 'selections' missing or not the eval104kb layout")
    if not 0 <= i < len(mats) or mats[i] != tag:
        raise Stop(f"DEV_SELECTIONS_MATERIAL: dev material {i} is {mats[i] if 0 <= i < len(mats) else None}, scene {tag}")
    out = {}
    for a, per in sel.items():
        if (not isinstance(per, list) or len(per) != len(mats)
                or any(type(v) is not int or not 0 <= v < len(KB.START_KEYS) for v in per)):
            raise Stop(f"DEV_SELECTIONS_FORMAT: {a}: expected a list of {len(mats)} start indices, got {per!r}")
        out[a] = per[i]
    return out


def lum_images(fx, name, h):
    e1 = (fx["N|e1"] + fx[f"{name}|e1"]).mean(0)
    return {"e1": e1 @ KB.LUMA, "total": (e1 + (fx["N|e2"] + fx[f"{name}|e2"]).mean(0) + h) @ KB.LUMA}


def concentration(img_x, img_o, refl, masks):
    """descriptive104ka squared-error concentration of X vs O0 (luminance of replicate-mean images, objects)."""
    obj = masks["objects"]
    out = {}
    for ev in ("e1", "total"):
        eX, eO = img_x[ev] - refl[ev], img_o[ev] - refl[ev]
        d = eX ** 2 - eO ** 2
        se = float((eO ** 2)[obj].sum())
        top = np.sort(d[obj])[::-1]
        ab = np.sort(np.abs(d[obj]))[::-1]
        share = (lambda m: float(ab[:m].sum() / ab.sum())) if ab.sum() > 0 else (lambda m: None)   # identical closures: undefined
        out[ev] = {"total_change_of_O0_SE": float(d[obj].sum() / se),
                   "top20_increase_of_O0_SE": float(top[:20].sum() / se), "top100_increase_of_O0_SE": float(top[:100].sum() / se),
                   "top20_share_of_abs_change": share(20), "top100_share_of_abs_change": share(100),
                   "by_region_of_O0_SE": {q: float(d[masks[q] & obj].sum() / se) for q in PEAK_REGIONS if q in masks}}
    return out


def analyze(scene, root, out, smoke=False):
    t0 = time.monotonic()
    dev, dev_info = dev_decision(root, smoke)                  # LEAK_GUARD before any scene array is read
    arch, masks, seen = scene_inputs(scene, root)
    ref = {"e1": arch["reference_e1"], "e2": arch["reference_e2"], "h": arch["reference_h"]}
    logs = [KB.read(p) for p in sorted(out.glob("fixed_[0-9]*_*.json"))]
    if sorted(r for lg in logs for r in lg["replicates"]) != list(range(12)):
        raise Stop("SCENE_REPLICATES_INCOMPLETE")
    chk = [c for lg in logs for c in lg["checks"]]
    keys = ("closures", "fits", "seed_base", "spp", "camera", "groups", "chunk_spp", "dev_decision")
    if any(tuple(lg.get(q) for q in keys) != tuple(logs[0].get(q) for q in keys) for lg in logs):
        raise Stop("SCENE_REPLAY_INCONSISTENT: replicate ranges used different closures / fits / seeds / spp / camera / groups / "
                   "chunk_spp / dev decision")
    if dev is not None:
        if (logs[0].get("dev_decision") or {}).get("sha256") != dev_info["sha256"]:
            raise Stop("SCENE_STATE_MISMATCH: replay logs were not made under the current hashed dev decision")
        state_check = verify_states(root, logs[0]["fits"], dev)
    else:
        state_check = {"bypassed": "smoke only: no hashed dev decision"}
    fx = {}
    for r in range(12):
        with np.load(out / f"fixed_r{r:03d}.npz") as z:
            for q in z.files:
                fx.setdefault(q, []).append(z[q])
    fx = {q: np.stack(v) for q, v in fx.items()}
    ident = {"A0_e1": max(S8.l1(fx["N|e1"][r] + fx["A0|e1"][r], arch["R0_e1"][r]) for r in range(12)),
             "A0_e2": max(S8.l1(fx["N|e2"][r] + fx["A0|e2"][r], arch["R0_e2"][r]) for r in range(12)),
             "lane_e1_rel": max(c["e1_A0_minus_native"] / max(c["e1_native_peak"], 1e-30) for c in chk),
             "lane_e2_weight_pdf_rel": max(c["e2_weight_pdf_minus_eval"] / max(c["e2_native_peak"], 1e-30) for c in chk),
             "grouped_pass_repeat_rel_l1": max(c.get("grouped_pass_repeat_rel_l1", 0.0) for c in chk)}
    ok = (ident["A0_e1"] <= IDENTITY_L1 and ident["A0_e2"] <= IDENTITY_L1 and ident["lane_e1_rel"] <= LANE_REL
          and ident["lane_e2_weight_pdf_rel"] <= LANE_REL and ident["grouped_pass_repeat_rel_l1"] <= IDENTITY_L1)
    res = {"scene": scene, "independent": scene in INDEPENDENT, "identity": ident, "identity_passed": bool(ok),
           "scope": "fixed-proposal replay: TOTAL = E1 + fixed E2 + R0 HIGH (HIGH fixed); not an end-to-end ECRO validation",
           "inputs_sha256": seen, "fits": logs[0]["fits"], "grouped_passes": logs[0].get("groups"), "dev_decision": dev_info,
           "state_check": state_check, "smoke": bool(smoke), "commit": KB.commit()}
    if not ok:
        KB.write(out / "analysis104kb.json", {**res, "stop": "SCENE_REPLAY_IDENTITY", "seconds": time.monotonic() - t0})
        raise Stop("SCENE_REPLAY_IDENTITY")
    names = [q.split("|")[0] for q in fx if q.endswith("|e1") and not q.startswith("N|")]
    roi_masks = {q: masks[q] for q in ROIS if q in masks and masks[q].any()}
    h = arch["R0_h"]
    M = {}
    for name in names:
        buf = {ev: fx[f"N|{ev}"].astype(np.float64) + fx[f"{name}|{ev}"] for ev in ("e1", "e2")}
        M[name], _ = event_metrics({**buf, "h": h}, ref, roi_masks, GATES)
    _, precision = event_metrics({"e1": arch["R0_e1"], "e2": arch["R0_e2"], "h": h}, ref, roi_masks, GATES)
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
    if "O0" not in starts:
        raise Stop("SCENE_NO_CONTROL_ARM: O0")
    o0 = starts["O0"]
    o0_keys = res["fits"]["O0"]["start_keys"]
    contrasts, descriptive = {}, {}
    for a in [x for x in starts if x not in ("O0", "A_hist", "P_hist")]:
        common = [r for r in starts[a] if r in o0]
        contrasts[a] = {ev: rel_change(M, ev, [f"{a}_{r}" for r in common], [f"O0_{r}" for r in common]) for ev in EVENTS}
    for a in ("A_hist", "P_hist"):
        hk = res["fits"][a]["start_keys"]
        common = [r for r in starts.get(a, []) if r in o0 and r < len(o0_keys) and o0_keys[r] == hk[r]]
        if common:
            descriptive[a] = {ev: rel_change(M, ev, [f"{a}_{r}" for r in common], [f"O0_{r}" for r in common]) for ev in EVENTS}
            descriptive[a]["paired_start_keys"] = [hk[r] for r in common]
    # E1-selected starts (dev selection, no scene-based selection)
    i, row = S8.scene_row(scene)
    selected, e1_other = {"source": dev_info, "material": row["tag"]}, None
    if dev is not None:
        if "O0" not in dev.get("selections", {}):
            raise Stop("DEV_SELECTIONS_FORMAT: no O0 selection in dev.json")
        for a, s in dev_selections(dev, i, row["tag"]).items():
            if a not in starts or s not in starts[a]:
                raise Stop(f"SCENE_STATE_MISMATCH: dev-selected start {a}_{s} was not replayed")
            q = f"{a}_{s}"
            keys = res["fits"][a]["start_keys"]
            others = [f"{a}_{r}" for r in starts[a] if r != s]
            selected[a] = {"start": s, "start_key": keys[s],
                           "objects_rms": {ev: strip(M[q]["objects"]["rms"][ev]) for ev in EVENTS},
                           "vs_own_start_mean": {ev: rel_change(M, ev, [q], [f"{a}_{r}" for r in starts[a]]) for ev in EVENTS},
                           "vs_own_other_starts_mean": {ev: rel_change(M, ev, [q], others) for ev in EVENTS},
                           "n_other_starts": len(others),
                           "vs_O0_start_mean": {ev: rel_change(M, ev, [q], [f"O0_{r}" for r in o0]) for ev in EVENTS}}
        contrasts["E1"] = selected["O0"]["vs_O0_start_mean"]
        e1_other = selected["O0"]["vs_own_other_starts_mean"]
    # peak-pixel concentration and maps
    refl = {"e1": (ref["e1"].mean(0)) @ KB.LUMA, "total": (ref["e1"] + ref["e2"] + ref["h"]).mean(0) @ KB.LUMA}
    h_mean = h.mean(0)
    lum = {q: lum_images(fx, q, h_mean) for q in names}
    peak = {}
    for a in [x for x in starts if x != "O0"]:
        hk = res["fits"].get(a, {}).get("start_keys")
        rs = [r for r in starts[a] if r in o0 and (a not in ("A_hist", "P_hist") or (hk and o0_keys[r] == hk[r]))]
        peak[a] = {str(r): concentration(lum[f"{a}_{r}"], lum[f"O0_{r}"], refl, masks) for r in rs}
    maps = {"reference_e1_lum": refl["e1"], "reference_total_lum": refl["total"], "objects_mask": masks["objects"],
            "A0_e1_lum": lum["A0"]["e1"], "A0_total_lum": lum["A0"]["total"]}
    for a, rs in starts.items():
        for ev in ("e1", "total"):
            m = np.mean([lum[f"{a}_{r}"][ev] for r in rs], 0)
            maps[f"{a}_{ev}_lum"] = m
            maps[f"{a}_{ev}_abs_err"] = np.abs(m - refl[ev])
    if "O0" in selected:
        for ev in ("e1", "total"):
            maps[f"E1_{ev}_lum"] = lum[f"O0_{selected['O0']['start']}"][ev]
            maps[f"E1_{ev}_abs_err"] = np.abs(maps[f"E1_{ev}_lum"] - refl[ev])
    np.savez_compressed(out / "maps.npz", **{q: v.astype(np.float32) if v.dtype != bool else v for q, v in maps.items()})
    res.update({"contrasts": contrasts, "contrasts_descriptive_hist": descriptive, "contrasts_descriptive_E1": e1_other,
                "arm_summary": summary,
                "contrast_notes": {"E1": "O0 at its dev-selected start for this material vs the O0 start-mean (same replays; "
                                         "no scene-based selection)",
                                   "contrasts_descriptive_E1": "descriptive companion: the dev-selected O0 start "
                                                         "vs the mean of the other 7 O0 starts (same replays)",
                                   "A_hist / P_hist": "descriptive, paired with O0 starts of equal key"},
                "per_closure": strip(M), "selected_E1": selected, "peak_pixels": peak, "baseline_precision_failures": precision,
                "replicate_seconds_median": float(np.median([c["seconds"] for c in chk])),
                "label_rule": "relative change of start-mean objects stable RMS vs O0 (paired starts), jackknife t(11) 95 %: "
                              "IMPROVED hi < 0, REGRESSED lo > +0.02, otherwise NEUTRAL",
                "seconds": time.monotonic() - t0})
    KB.write(out / "analysis104kb.json", res)
    print("analysis", scene, ok, {a: (round(v["total"]["point"], 4), round(v["total"]["lo"], 4), round(v["total"]["hi"], 4),
                                     v["total"]["label"]) for a, v in contrasts.items()}, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("reference", "fixed", "analyze"))
    ap.add_argument("scene")
    ap.add_argument("root", type=Path)
    ap.add_argument("out", type=Path, nargs="?")
    ap.add_argument("--reps", default="0:12")
    ap.add_argument("--k", type=int, default=64, help="sub-chunk spp (results do not depend on it)")
    ap.add_argument("--group", type=int, default=8, help="closures per pass besides A0 (8: kernel compile ~3 min vs ~20 min at 16)")
    ap.add_argument("--smoke", action="store_true", help="fixed / analyze: smoke only (may run without a hashed dev decision)")
    ap.add_argument("--dup", type=int, default=0, help="smoke only: add N copies of the O0 closures (memory / time emulation)")
    a = ap.parse_args()
    if a.mode == "reference":
        run_reference(a.scene, a.root)
    elif a.out is None:
        ap.error("out_dir is required")
    elif a.mode == "fixed":
        run_fixed(a.scene, a.root, a.out, tuple(int(x) for x in a.reps.split(":")), a.smoke, a.k, a.group, a.dup)
    else:
        analyze(a.scene, a.root, a.out, a.smoke)


if __name__ == "__main__":
    main()
