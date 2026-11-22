"""WL104K-B angular evaluation: dev-only decisions (E1 selection, stage-F eligibility) and the independent final angular test.

  python eval104kb.py dev  <root> [--smoke]   dev lanes + dev peak grid ONLY. Arms O0 / O1 / D / E2 (8 starts) and the
                                    synthetic S_O0 (4 starts, noisy synthetic dev targets); O0 / O1 / D / E2 and S_O0 / S_O1 / S_D /
                                    S_E2 must exist (hashes recorded) and every fit's training data must be this run's data
                                    (FIT_DATA_MISMATCH), unless --smoke. Writes <root>/dev_eval/dev.json, dev.json.sha256 and
                                    dev_eval/arrays.npz; prints 'ELIGIBLE <list|NONE>' and 'F_SPEC ...' (or 'F_SPEC NOT_RUN').
  python eval104kb.py test <root> [--smoke]   LEAK_GUARD (data104kb.test_lock + local checks): refuses unless the hashed dev
                                    decision exists, the fits it recorded (physical and synthetic) are unchanged, fits/F exists when
                                    F is eligible, and the protocol / eval104kb.py / decision104kb.py / dev data / frozen input hashes
                                    equal those recorded at dev time. Test lanes + test peak grid (final), WL104D Amendment-1
                                    reference (secondary, repeatedly inspected). Arms O0 / O1 / D / E2 / F, P_hist, A_hist; synthetic
                                    controls S_O0 / S_O1 / S_D / S_E2 against noise-free truth. Writes test_eval/test.json + arrays.npz.
--smoke: pipeline check only (missing arms and unauthenticated data allowed and recorded; a smoke dev decision is refused by a
         non-smoke test evaluation).

root = results/ecro_wl84_104kb/primary (data/, fits/<arm>/). Metric definitions: protocol.json metrics / angular_verdicts /
synthetic_controls / stage_F. Interpretation choices that the protocol leaves open are listed in DEFINITIONS (written into the
outputs). Predictions are float64 (common104k components, batched over materials and starts).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

import common104kb as KB
import fit104kb as FK

CM, IO, GF, EV, KA = KB.CM, KB.IO, KB.GF, KB.EV, KB.KA
DEV = KB.DEV
LUMA = KB.LUMA
Stop = KB.Stop
PHYS = ("O0", "O1", "D", "E2")
MECH = ("O1", "D", "E2")
SYN = ("S_O0", "S_O1", "S_D", "S_E2")
SPEC = {"O0": ("adam", "original", "original", False), "O1": ("lbfgs", "original", "original", False),
        "D": ("adam", "coverage", "original", False), "E2": ("adam", "original", "e2", False),
        "S_O0": ("adam", "original", "original", True), "S_O1": ("lbfgs", "original", "original", True),
        "S_D": ("adam", "coverage", "original", True), "S_E2": ("adam", "original", "e2", True)}
CONTROL = {"O1": "SC_O1", "D": "SC_D", "E2": "SC_E2"}
NPAR = 178
NB = 10000
GRAZING_DEG = 75.0
RGB_REGIONS = {"e_core": (0.0, FK.CORE_DEG), "e_shoulder": (FK.CORE_DEG, FK.SHOULDER_DEG)}
REGIONS = {"core": (0.0, FK.CORE_DEG), "shoulder": (FK.CORE_DEG, FK.SHOULDER_DEG), "tail": (FK.SHOULDER_DEG, 181.0)}
CON_KEYS = ("el1", "ealb", "eshape", "e_core", "e_shoulder", "crit", "e_prof", "e_h", "core_l1", "shoulder_l1", "tail_l1",
            "el1_grazing")
CHUNK = 1 << 18
DATA_SETS = ("all", "isolation", "train_coverage", "dev", "dev_peak", "test", "test_peak")   # data104kb data.json 'qualified'
CODE_FILES = ("eval104kb.py", "decision104kb.py")
ORIGINAL_DATA_KEY = "results/ecro_wl84_104j/primary/data/fit_set.npz"   # train_original (authenticated by CM.inputs)
O0_REPRO_OK, O0_REPRO_UNQUALIFIED = 1e-4, 1e-2                          # protocol arms.O0
O0_FLAG = "O0_REPRODUCTION_DISCREPANCY"
DEFINITIONS = {
    "targets": "Lane metrics against the mean of the two reference halves; every *_a / *_b key repeats it against one half.",
    "e_core / e_shoulder": "RGB region-normalized relative L1 over lanes with mirror offset in [0, 3) / [3, 12) deg, importance "
                           "weighted (fit104kb.Data formula, denominator clamped at 1e-12).",
    "region l1 / signed": "Luminance region relative L1 and signed energy (eval104k.metrics), region edges 3 / 12 deg.",
    "crit": "Per-probe el1 + ealb + e_core + e_shoulder; S_dev = mean over the 20 probes (protocol selection_criterion_dev).",
    "el1_grazing": "el1 on probes with incident polar >= 75 deg, 0 elsewhere (pooled sum = sum over grazing probes).",
    "peak": "Luminance on all USED grid points (use mask; radius 0 .. 8 deg on the new grids, 0 .. 4 on WL104D): s_h = max F / "
            "max R - 1, e_h = |s_h|, e_prof = sum |F - R| / sum R.",
    "ring residual": "Per ring: mean over used azimuths of F / mean of R - 1.",
    "instability": "SD over starts (ddof = 1) of s_h per (material, probe); I = mean over probes then materials.",
    "pooled contrast": "Per start: sum over (material, probe) of X / same sum of O0 - 1 (paired starts); median over starts; "
                       "cluster bootstrap over materials (10,000 resamples of the material index from one generator seeded 1041350 "
                       "per cohort, reused by every contrast: common random numbers), 2.5 / 97.5 % of the bootstrap medians.",
    "resolved": "Contrast recomputed against half a and half b: same sign as each other and as the mean-target contrast, and "
                "|c_a - c_b| / 2 <= |c| / 4.",
    "BROAD_PRESERVED vs P_hist": "median over X starts of pooled el1 / median over the 4 P_hist starts of pooled el1 - 1 (unpaired).",
    "material regression": "Per material: median over starts of the per-start mean-over-probes el1, X / O0 - 1, applied to all "
                           "nine materials (protocol pre_freeze_clarifications).",
    "median-start pooled (E1)": "Sum over materials of the per-material median over starts of the material-pooled metric.",
    "spearman": "Average ranks; an undefined correlation (constant input) counts as 0 (not > 0).",
    "synthetic SC medians": "Ratio of the medians over starts of the pooled metric (S_X median / S_O0 median).",
    "O0 reproduction": "Per unit (WL104K starts 1-4 x material) |O0 final training objective - WL104K P final objective| / |WL104K P|, "
                       "recorded per unit: max <= 1e-4 reproduced; 1e-4 < max <= 1e-2 disclosed, every label carries "
                       "O0_REPRODUCTION_DISCREPANCY ('flags' list and '+O0_REPRODUCTION_DISCREPANCY' label suffix) and stays "
                       "qualified; max > 1e-2 -> UNQUALIFIED (protocol arms.O0).",
    "fit data": "Every fit's fit.json data_sha256 must equal this run's training data: train_original = the CM.inputs-authenticated "
                "WL104J fit_set.npz hash; train_coverage = data_dir equal to <root>/data and the sha256 of "
                "<root>/data/train_coverage.npz (FIT_DATA_MISMATCH otherwise).",
    "leak guard": "Test mode refuses unless data104kb.test_lock passes (when present) and the local checks pass: hashed dev "
                  "decision, unchanged physical + synthetic fits, F present when eligible, protocol / eval104kb.py / "
                  "decision104kb.py / dev.npz / dev_peak.npz / every dev-time inputs_sha256 entry unchanged, and a non-smoke "
                  "dev decision for a non-smoke test.",
    "data authentication": "Outside --smoke: data/data.json must exist, the qualification flag of every used set must be stated "
                           "(not None) and every loaded npz must be hash-verified against data.json outputs_sha256.",
    "test arms": "Physical arms are compared with O0 on paired starts; per-row labels are gated here (UNQUALIFIED on a data, "
                 "isolation or O0-reproduction failure); the decision adds only contract / leak / dev-hash gating.",
}


# ------------------------------------------------------------------ small utilities
def clean(o):
    """JSON-safe copy: numpy scalars / arrays -> python, NaN / inf -> None."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    return o


def write(p, o):
    KB.write(p, clean(o))


def data_status(root, files, used=(), smoke=False):
    """data/data.json: per-set qualification (None when the report does not state it) and sha256 verification of the loaded npz.
    Outside smoke the authentication is mandatory: data.json present, the 'used' sets' qualification stated, every file hashed."""
    p = root / "data/data.json"
    if not p.exists():
        if not smoke:
            raise Stop("DATA_UNAUTHENTICATED: data/data.json missing")
        return {"available": False, "qualified": {s: None for s in DATA_SETS}, "hash_verified": {f: None for f in files},
                "smoke": True}
    dj = KB.read(p)
    q = dj.get("qualified", dj.get("qualification"))

    def setq(name):
        v = q.get(name) if isinstance(q, dict) else q
        if isinstance(v, dict):
            v = v.get("all", v.get("passed", v.get("qualified")))
        return None if v is None else bool(v)
    hashes = {}
    for key in ("outputs_sha256", "sha256", "npz_sha256", "files_sha256"):
        if isinstance(dj.get(key), dict):
            hashes.update({Path(k).name: v for k, v in dj[key].items()})
    ver = {}
    for f in files:
        h = hashes.get(f)
        if h is None:
            ver[f] = None
        elif KB.sha(root / "data" / f) != h:
            raise Stop(f"FROZEN_HASH_MISMATCH: data/{f}")
        else:
            ver[f] = True
    qual = {s: setq(s) for s in DATA_SETS}
    if not smoke:
        unstated = [s for s in used if qual[s] is None]
        unhashed = [f for f, v in ver.items() if v is not True]
        if unstated or unhashed:
            raise Stop(f"DATA_UNAUTHENTICATED: qualification not stated {unstated}, hash not recorded {unhashed}")
    return {"available": True, "qualified": qual, "hash_verified": ver, "data_json_sha256": KB.sha(p), "smoke": bool(smoke)}


def fit_data_context(root, seen):
    """Expected training-data hashes of this run (train_original from the authenticated inputs; train_coverage hashed lazily)."""
    return {"original": seen.get(ORIGINAL_DATA_KEY), "coverage_dir": (root / "data").resolve(),
            "coverage_path": root / "data/train_coverage.npz", "coverage": None}


def check_fit_data(arm, fj, ctx):
    """fit.json data_sha256 (and data_dir for coverage) vs this run root's data -> Stop('FIT_DATA_MISMATCH')."""
    kind = fj["arm"]["data"]
    if kind == "coverage":
        if ctx["coverage"] is None:
            if not ctx["coverage_path"].exists():
                raise Stop(f"FIT_DATA_MISMATCH: {arm} uses train_coverage but {ctx['coverage_path']} is missing")
            ctx["coverage"] = KB.sha(ctx["coverage_path"])
        dd = fj.get("data_dir")
        if dd is None or Path(dd).resolve() != ctx["coverage_dir"]:
            raise Stop(f"FIT_DATA_MISMATCH: {arm} data_dir {dd} != {ctx['coverage_dir']}")
        want = ctx["coverage"]
    else:
        want = ctx["original"]
    if want is None or fj.get("data_sha256") != want:
        raise Stop(f"FIT_DATA_MISMATCH: {arm} data_sha256 {fj.get('data_sha256')} != {want} ({kind})")
    return want


def load_sets(root, name, n):
    """Lane set + peak grid of one new set ('dev' / 'test'), layout checked."""
    import data104kb as DK
    lp, pp = root / f"data/{name}.npz", root / f"data/{name}_peak.npz"
    ds = DK.load_lanes(lp, n) if hasattr(DK, "load_lanes") else KB.load_lane_set(lp, n)   # test_* lanes: test lock
    with np.load(lp) as z:
        polar = z["polar_deg"].reshape(n, -1)
        inc = z["incidence"].reshape(n, -1, 3)
        mid = z["mid"].reshape(n, -1, z["mid"].shape[-1])
    if not (mid == np.arange(n)[:, None, None]).all():
        raise Stop(f"LAYOUT: {name} mid is not material-major")
    if not np.allclose(ds["wi"][:, :, 0], inc, atol=1e-12):
        raise Stop(f"LAYOUT: {name} incidence")
    pk = DK.load_peak(pp, n)
    if not np.allclose(pk["peak_wi"][:, :, 0], inc, atol=1e-12):
        raise Stop(f"LAYOUT: {name} peak incidence differs from the lane set")
    ds["rad"] = KB.mirror_radius(ds["wi"], ds["wo"])
    ds["graze"] = (polar >= GRAZING_DEG).astype(float)
    ds["polar"] = polar
    return ds, pk


def hist_sets(ref, n):
    """WL104D Amendment-1 reference in the same layout (secondary)."""
    ev, peak = CM.eval_reference(ref, n)
    ev["rad"] = CM.mirror_radius(ev["wi"], ev["wo"])
    polar = np.broadcast_to(CM.POLAR, ev["w"].shape[:2])
    ev["graze"], ev["polar"] = (polar >= GRAZING_DEG).astype(float), polar
    pa = ref["peak_a"].reshape(n, CM.NP, *ref["peak_a"].shape[1:]) @ LUMA
    pb = ref["peak_b"].reshape(n, CM.NP, *ref["peak_b"].shape[1:]) @ LUMA
    rad = np.round(CM.mirror_radius(peak["wi"], peak["wo"])[0, 0], 6)
    pk = {"peak_wi": peak["wi"], "peak_wo": peak["wo"], "ref": peak["ref"], "ref_a": pa, "ref_b": pb,
          "use": peak["wo"][..., 2] > 1e-7, "radius_deg": rad}
    return ev, pk


def load_fit(root, arm, n, R, spec=None, ctx=None):
    """fits/<arm>: hash, arm spec, capacity, budget, start keys and (ctx given) training data checked. None when absent."""
    d = root / "fits" / arm
    if not (d / "states.npz").exists():
        return None
    fj = KB.read(d / "fit.json")
    sha = KB.sha(d / "states.npz")
    if sha != fj["states_sha256"]:
        raise Stop(f"STATES_HASH: {arm}")
    a = fj["arm"]
    want = spec if spec is not None else SPEC[arm]
    if (a["method"], a["data"], a["objective"], bool(a["synthetic"])) != tuple(want):
        raise Stop(f"ARM_SPEC_MISMATCH: {arm} {a} vs {want}")
    if fj["parameters_per_material"] != NPAR or not fj["budget_respected"]:
        raise Stop(f"BUDGET_OR_CAPACITY: {arm}")
    if fj["start_keys"] != KB.START_KEYS[:R]:
        raise Stop(f"START_KEYS: {arm} {fj['start_keys']}")
    data_sha = check_fit_data(arm, fj, ctx) if ctx is not None else None
    st, truth = EV.load_states(d / "states.npz", R, n)
    for s in st["P"]:
        if not (np.isfinite(s["y"]).all() and np.isfinite(s["g"]).all()):
            raise Stop(f"NONFINITE_STATE: {arm}")
    return {"states": st["P"], "truth": truth or None, "fit": fj, "sha": sha, "R": R, "data_sha256": data_sha}


# ------------------------------------------------------------------ prediction (float64, batched over starts and materials)
def predict_units(kind, states, wi, wo, chunk=CHUNK):
    """kind 'P' (ER-v1-SDM) or 'A' (certified ER); states: list over starts of {'y': [n, 58], 'g': [n, 8, 3, 5]};
    wi / wo [n, ..., 3] -> F [R, n, ..., 3] (numpy float64). Same operations as common104k.predict."""
    R, n = len(states), wi.shape[0]
    lanes = int(np.prod(wi.shape[1:-1]))
    y = np.concatenate([s["y"] for s in states])
    g = np.concatenate([s["g"] for s in states]) if kind == "P" else None
    out = np.empty((R * n, lanes, 3))
    WI = KB.to_t(wi.reshape(n, lanes, 3))
    WO = KB.to_t(wo.reshape(n, lanes, 3))
    per = max(1, chunk // lanes)
    with torch.no_grad():
        p, gt = CM.decode(kind, {"y": y, "g": g})
        for u0 in range(0, R * n, per):
            us = torch.arange(u0, min(u0 + per, R * n), device=DEV)
            ix = us.repeat_interleave(lanes)
            q = {k: v[ix] for k, v in p.items()}
            gg = None if gt is None else gt[ix]
            m = us % n
            F = CM.lane_eval(kind, q, gg, WI[m].reshape(-1, 3), WO[m].reshape(-1, 3))
            out[u0:u0 + len(us)] = F.reshape(len(us), lanes, 3).cpu().numpy()
    return out.reshape(R, *wi.shape[:-1], 3)


# ------------------------------------------------------------------ metrics
def crit_terms(F, tgt, w, rad):
    """WL82 el1 / ealb / eshape plus the RGB region terms e_core / e_shoulder: F [..., n, P, L, 3] -> [..., n, P]."""
    out = dict(CM.wl82_terms(F, tgt, w))
    for name, (lo, hi) in RGB_REGIONS.items():
        W = (w * ((rad >= lo) & (rad < hi)))[..., None]
        out[name] = (np.abs(F - tgt) * W).sum((-2, -1)) / np.maximum((tgt * W).sum((-2, -1)), 1e-12)
    out["crit"] = out["el1"] + out["ealb"] + out["e_core"] + out["e_shoulder"]
    return out


def lane_metrics(F, ds):
    out = {}
    for h, key in (("", "mean"), ("_a", "a"), ("_b", "b")):
        for k, v in crit_terms(F, ds[key], ds["w"], ds["rad"]).items():
            out[k + h] = v
        out["el1_grazing" + h] = out["el1" + h] * ds["graze"]
    pl, rl = (F @ LUMA) * ds["w"], (ds["mean"] @ LUMA) * ds["w"]
    tot = np.maximum(rl.sum(-1), 1e-20)
    for k, (lo, hi) in REGIONS.items():
        m = (ds["rad"] >= lo) & (ds["rad"] < hi)
        den = np.maximum((rl * m).sum(-1), 1e-20)
        out[f"{k}_signed"] = ((pl - rl) * m).sum(-1) / den
        out[f"{k}_l1"] = (np.abs(pl - rl) * m).sum(-1) / den
        out[f"{k}_l1_of_total"] = (np.abs(pl - rl) * m).sum(-1) / tot
    return out


def peak_metrics(Fl, pk):
    """Fl luminance [..., n, P, K] on the peak grid."""
    use = pk["use"]
    out = {}
    Fu = np.where(use, Fl, 0.0)
    for h, key in (("", "ref"), ("_a", "ref_a"), ("_b", "ref_b")):
        Ru = np.where(use, pk[key], 0.0)
        s = Fu.max(-1) / np.maximum(Ru.max(-1), 1e-20) - 1
        out["s_h" + h], out["e_h" + h] = s, np.abs(s)
        out["e_prof" + h] = np.abs(Fu - Ru).sum(-1) / np.maximum(Ru.sum(-1), 1e-20)
    return out


def rings(pk):
    r = np.round(np.asarray(pk["radius_deg"], float), 6)
    return np.unique(r), r


def ring_means(Xl, pk):
    """Mean over used grid points of each ring: [..., n, P, K] -> [..., n, P, n_ring]."""
    radii, r = rings(pk)
    use = pk["use"].astype(float)
    out = []
    for rr in radii:
        m = (r == rr) * use
        out.append((Xl * m).sum(-1) / np.maximum(m.sum(-1), 1e-20))
    return np.stack(out, -1)


def evaluate(kind, states, ds, pk):
    """All lane + peak metrics [R, n, P] and ring profiles [R, n, P, n_ring] for one arm on one set."""
    m = lane_metrics(predict_units(kind, states, ds["wi"], ds["wo"]), ds)
    Fl = predict_units(kind, states, pk["peak_wi"], pk["peak_wo"]) @ LUMA
    m.update(peak_metrics(Fl, pk))
    prof = ring_means(Fl, pk) / np.maximum(ring_means(pk["ref"], pk), 1e-20) - 1
    return m, prof


def instability(s):
    """s_h [R, n, P] -> per material I_m [n] (SD over starts, ddof 1, mean over probes)."""
    return s.std(0, ddof=1).mean(-1)


# ------------------------------------------------------------------ contrasts
def boot_index(cohorts):
    return {c: np.random.default_rng(KB.P["seeds"]["bootstrap"]).integers(0, len(ms), (NB, len(ms))) for c, ms in cohorts.items()}


def pooled_contrast(X, Y, ms, idx):
    """Paired-start pooled relative change of X vs Y ([R, n, P]) over materials ms."""
    Xm, Ym = X[:, ms].sum(-1), Y[:, ms].sum(-1)
    per = Xm.sum(1) / np.maximum(Ym.sum(1), 1e-300) - 1
    bx, by = Xm[:, idx].sum(-1), Ym[:, idx].sum(-1)
    med = np.median(bx / np.maximum(by, 1e-300) - 1, 0)
    return {"per_start": per, "median": float(np.median(per)), "lo": float(np.quantile(med, .025)),
            "hi": float(np.quantile(med, .975))}


def inst_contrast(sX, sY, ms, idx):
    IX, IY = instability(sX)[ms], instability(sY)[ms]
    b = IX[idx].mean(-1) / np.maximum(IY[idx].mean(-1), 1e-300) - 1
    return {"I_X": float(IX.mean()), "I_ref": float(IY.mean()), "median": float(IX.mean() / max(IY.mean(), 1e-300) - 1),
            "lo": float(np.quantile(b, .025)), "hi": float(np.quantile(b, .975))}


def resolved(c, ca, cb):
    return bool(np.sign(ca) == np.sign(cb) == np.sign(c) and c != 0 and abs(ca - cb) / 2 <= abs(c) / 4)


def contrast_table(MX, MY, cohorts, idx, keys=CON_KEYS):
    out = {}
    for k in keys:
        out[k] = {}
        for coh, ms in cohorts.items():
            c = pooled_contrast(MX[k], MY[k], ms, idx[coh])
            if f"{k}_a" in MX:
                c["median_a"] = pooled_contrast(MX[k + "_a"], MY[k + "_a"], ms, idx[coh])["median"]
                c["median_b"] = pooled_contrast(MX[k + "_b"], MY[k + "_b"], ms, idx[coh])["median"]
                c["resolved"] = resolved(c["median"], c["median_a"], c["median_b"])
            out[k][coh] = c
    out["instability"] = {}
    for coh, ms in cohorts.items():
        c = inst_contrast(MX["s_h"], MY["s_h"], ms, idx[coh])
        c["median_a"] = inst_contrast(MX["s_h_a"], MY["s_h_a"], ms, idx[coh])["median"]
        c["median_b"] = inst_contrast(MX["s_h_b"], MY["s_h_b"], ms, idx[coh])["median"]
        c["resolved"] = resolved(c["median"], c["median_a"], c["median_b"])
        out["instability"][coh] = c
    return out


def summary(M, cohorts):
    """Pooled per-start sums, their median, per-material median-start means and instability for one arm."""
    out = {}
    for k in CON_KEYS + ("crit", "s_h", "core_signed", "shoulder_signed", "tail_signed"):
        if k not in M:
            continue
        per = M[k].sum((1, 2))
        out[k] = {"pooled_per_start": per, "pooled_median": float(np.median(per)),
                  "material_median_start_mean": np.median(M[k], 0).mean(-1),
                  "cohort_pooled_median": {c: float(np.median(M[k][:, ms].sum((1, 2)))) for c, ms in cohorts.items()}}
    if M["s_h"].shape[0] > 1:
        Im = instability(M["s_h"])
        out["instability"] = {"I": float(Im.mean()), "I_material": Im, "I_cohort": {c: float(Im[ms].mean()) for c, ms in cohorts.items()}}
    return out


# ------------------------------------------------------------------ dev
def selection_dev(M):
    """S_dev [R, n] = mean over probes of crit; selected start per material (argmin, ties -> lower start index)."""
    S = M["crit"].mean(-1)
    return S, np.argmin(S, 0)


def selection_training(fj, R, n):
    obj = np.asarray(fj["final_terms"]["objective"], float).reshape(R, n)
    return obj, np.argmin(obj, 0)


def synthetic_truth_check(fits, raw):
    tr = KB.synthetic_truth(raw["R0"])
    for arm, f in fits.items():
        if f is None:
            continue
        t = f["truth"]
        if t is None or np.abs(t["y"] - tr["y"]).max() > 1e-12 or np.abs(t["g"] - tr["g"]).max() > 1e-12:
            raise Stop(f"SYNTHETIC_TRUTH_MISMATCH: {arm}")
    return tr


def code_sha():
    return {f: KB.sha(KB.HERE / f) for f in CODE_FILES}


def require_arms(fits, smoke):
    """Every listed fit must exist outside smoke (Stop 'MISSING: fits/<arm>'); returns the missing arms (smoke only)."""
    missing = [a for a, f in fits.items() if f is None]
    if missing and not smoke:
        raise Stop(f"MISSING: fits/{missing[0]} (all missing: {missing})")
    return missing


def run_dev(root, smoke=False):
    out = root / "dev_eval"
    if (out / "dev.json").exists():
        raise Stop("DEV_DECISION_EXISTS: dev_eval/dev.json is frozen once written")
    if (root / "test_eval").exists():
        raise Stop("LEAK_GUARD: test_eval exists before the dev decision")
    t0 = time.monotonic()
    rows, raw, _, _, _, seen = CM.inputs()
    n = len(rows)
    cohorts = {"all": list(range(n)), **{c: [i for i, r in enumerate(rows) if r["cohort"] == c] for c in ("development", "held_out")}}
    code = code_sha()
    dstat = data_status(root, ("dev.npz", "dev_peak.npz"), ("dev", "dev_peak", "isolation", "train_coverage"), smoke)
    bad = [s for s in ("dev", "dev_peak", "isolation") if dstat["qualified"][s] is False]
    if bad:
        raise Stop(f"DATA_UNQUALIFIED: {bad}")
    ctx = fit_data_context(root, seen)
    fits = {arm: load_fit(root, arm, n, 8, ctx=ctx) for arm in PHYS}
    syn_fits = {S: load_fit(root, S, n, 4, ctx=ctx) for S in SYN}
    if fits["O0"] is None:
        raise Stop("MISSING: fits/O0")
    missing = require_arms({**fits, **syn_fits}, smoke)
    syn = syn_fits["S_O0"]
    ds, pk = load_sets(root, "dev", n)
    idx = boot_index(cohorts)
    M, prof, timing = {}, {}, {}
    for arm, f in fits.items():
        if f is None:
            continue
        t = time.monotonic()
        M[arm], prof[arm] = evaluate("P", f["states"], ds, pk)
        timing[arm] = time.monotonic() - t
        print("dev evaluated", arm, round(timing[arm], 1), "s", flush=True)
    # E1 selection (dev criterion) and the training-objective selection, every arm
    sel, S_dev, tsel, tobj = {}, {}, {}, {}
    for arm in M:
        S_dev[arm], sel[arm] = selection_dev(M[arm])
        tobj[arm], tsel[arm] = selection_training(fits[arm]["fit"], 8, n)
    # synthetic E1 control: S_O0 on noisy synthetic dev targets
    syn_out = {"available": syn is not None}
    if syn is not None:
        synthetic_truth_check({"S_O0": syn}, raw)
        tgt = predict_units("P", [syn["truth"]], ds["wi"], ds["wo"])[0]
        sigma = 0.5 * np.abs(ds["a"] - ds["b"])
        tgt = tgt + sigma * np.random.default_rng(KB.P["seeds"]["synthetic_dev_noise"]).standard_normal(tgt.shape)
        Fs = predict_units("P", syn["states"], ds["wi"], ds["wo"])
        cs = crit_terms(Fs, tgt, ds["w"], ds["rad"])
        Ss = cs["crit"].mean(-1)
        syn_out.update({"S_dev": Ss, "selected_start": np.argmin(Ss, 0), "states_sha256": syn["sha"],
                        "noise_seed": KB.P["seeds"]["synthetic_dev_noise"]})
    # stage F eligibility (dev only)
    elig = {}
    for X in MECH:
        if X not in M:
            elig[X] = {"status": "NOT_AVAILABLE", "eligible": False}
            continue
        cp = pooled_contrast(M[X]["e_prof"], M["O0"]["e_prof"], cohorts["all"], idx["all"])
        ce = pooled_contrast(M[X]["el1"], M["O0"]["el1"], cohorts["all"], idx["all"])
        ci = inst_contrast(M[X]["s_h"], M["O0"]["s_h"], cohorts["all"], idx["all"])
        rule = {"e_prof_median_le_-10pct": cp["median"] <= -0.10, "e_prof_hi_lt_0": cp["hi"] < 0,
                "el1_median_le_+5pct": ce["median"] <= 0.05, "instability_le_+10pct": ci["median"] <= 0.10}
        elig[X] = {"status": "EVALUATED", "e_prof": cp, "el1": ce, "instability": ci, "rule": rule, "eligible": all(rule.values())}
    eligible = [X for X in MECH if elig[X]["eligible"]]
    run_F = len(eligible) >= 2
    spec = ({"method": "lbfgs" if "O1" in eligible else "adam", "data": "coverage" if "D" in eligible else "original",
             "objective": "e2" if "E2" in eligible else "original"} if run_F else None)
    con = {X: contrast_table(M[X], M["O0"], cohorts, idx) for X in MECH if X in M}
    out.mkdir(parents=True, exist_ok=True)
    arrays = {f"{arm}__{k}": v for arm in M for k, v in M[arm].items()}
    arrays.update({f"profile__{arm}": v for arm, v in prof.items()})
    arrays.update({"profile_radius_deg": rings(pk)[0], "polar_deg": ds["polar"]})
    np.savez_compressed(out / "arrays.npz", **arrays)
    res = {"stage": "dev", "materials": [r["tag"] for r in rows], "cohorts": cohorts, "start_keys": KB.START_KEYS,
           "arms_evaluated": list(M), "arms_missing": [a for a in PHYS if a not in M],
           "fits_sha256": {arm: f["sha"] for arm, f in fits.items() if f is not None},
           "fit_json_sha256": {arm: KB.sha(root / "fits" / arm / "fit.json") for arm, f in fits.items() if f is not None},
           "synthetic_fits_sha256": {S: f["sha"] for S, f in syn_fits.items() if f is not None},
           "synthetic_fit_json_sha256": {S: KB.sha(root / "fits" / S / "fit.json") for S, f in syn_fits.items() if f is not None},
           "fits_data_sha256": {a: f["data_sha256"] for a, f in {**fits, **syn_fits}.items() if f is not None},
           "fits_missing_smoke": missing, "smoke": bool(smoke), "code_sha256": code,
           "data": dstat, "data_sha256": {f: KB.sha(root / "data" / f) for f in ("dev.npz", "dev_peak.npz")},
           "summary": {arm: summary(M[arm], cohorts) for arm in M}, "contrasts_vs_O0": con,
           "selection": {"S_dev": S_dev, "dev_criterion": sel, "training_objective_final": tobj, "training_objective": tsel},
           "selections": sel,                          # {arm: dev-selected start per material}: the scenes104kb interface
           "synthetic_selection": syn_out,
           "stage_F": {"eligibility": elig, "eligible": eligible, "run": run_F, "spec": spec},
           "definitions": DEFINITIONS, "arrays_sha256": KB.sha(out / "arrays.npz"), "timing_seconds": timing,
           "commit": KB.commit(), "protocol_sha256": KB.sha(KB.PROTOCOL), "inputs_sha256": seen, "seconds": time.monotonic() - t0}
    if code_sha() != code:
        raise Stop("CODE_CHANGED: eval104kb.py / decision104kb.py changed during the dev evaluation")
    write(out / "dev.json", res)
    (out / "dev.json.sha256").write_text(KB.sha(out / "dev.json") + "\n", encoding="utf-8")
    print("ELIGIBLE " + (",".join(eligible) if eligible else "NONE"), flush=True)
    print(("F_SPEC method={method} data={data} objective={objective}".format(**spec)) if run_F else "F_SPEC NOT_RUN", flush=True)


# ------------------------------------------------------------------ test
def leak_guard(root, smoke=False):
    """Test lock (data104kb.test_lock when present) plus the local checks; nothing under data/test* is opened here."""
    import data104kb as DK
    dp, sp = root / "dev_eval/dev.json", root / "dev_eval/dev.json.sha256"
    if not dp.exists() or not sp.exists():
        raise Stop("LEAK_GUARD: no hashed dev decision (dev_eval/dev.json + dev.json.sha256)")
    if KB.sha(dp) != sp.read_text(encoding="utf-8").split()[0].strip():
        raise Stop("LEAK_GUARD: dev.json does not match dev.json.sha256")
    lock = "data104kb.test_lock" if hasattr(DK, "test_lock") else (
        "data104kb.require_dev_decision" if hasattr(DK, "require_dev_decision") else "local only")
    try:
        if hasattr(DK, "test_lock"):
            DK.test_lock(root)
        elif hasattr(DK, "require_dev_decision"):
            DK.require_dev_decision(root)
    except Stop as e:
        raise Stop(f"LEAK_GUARD: {lock} refused ({e})")
    dev = KB.read(dp)
    same = lambda p, h: h is not None and Path(p).exists() and KB.sha(p) == h
    if dev.get("smoke") and not smoke:
        raise Stop("LEAK_GUARD: the dev decision is a smoke decision")
    if dev.get("protocol_sha256") != KB.sha(KB.PROTOCOL):
        raise Stop("LEAK_GUARD: protocol.json changed after the dev decision")
    code = dev.get("code_sha256")
    if not isinstance(code, dict) or set(code) != set(CODE_FILES):
        raise Stop("LEAK_GUARD: dev decision does not record code_sha256")
    for f, h in code.items():
        if not same(KB.HERE / f, h):
            raise Stop(f"LEAK_GUARD: {f} changed after the dev decision")
    for f in ("dev.npz", "dev_peak.npz"):
        h = dev.get("data_sha256", {}).get(f)
        if not same(root / "data" / f, h):
            raise Stop(f"LEAK_GUARD: data/{f} changed after the dev decision (or not recorded)")
    for rel, h in dev.get("inputs_sha256", {}).items():
        if not same(KB.REPO / rel, h):
            raise Stop(f"LEAK_GUARD: frozen input {rel} changed after the dev decision")
    if dev["stage_F"]["run"] and not (root / "fits/F/states.npz").exists():
        raise Stop("LEAK_GUARD: F is eligible but fits/F/states.npz does not exist")
    recorded = {**dev["fits_sha256"], **dev.get("synthetic_fits_sha256", {})}
    if not smoke and not set(PHYS + SYN) <= set(recorded):
        raise Stop(f"LEAK_GUARD: dev decision lacks fits {sorted(set(PHYS + SYN) - set(recorded))}")
    for arm, h in recorded.items():
        if not same(root / "fits" / arm / "states.npz", h):
            raise Stop(f"LEAK_GUARD: fits/{arm} changed after the dev decision")
    for arm, h in {**dev.get("fit_json_sha256", {}), **dev.get("synthetic_fit_json_sha256", {})}.items():
        if not same(root / "fits" / arm / "fit.json", h):
            raise Stop(f"LEAK_GUARD: fits/{arm}/fit.json changed after the dev decision")
    for arm in SYN:
        if (root / "fits" / arm / "states.npz").exists() and arm not in recorded:
            raise Stop(f"LEAK_GUARD: synthetic fit {arm} appeared after the dev decision")
    dev["_lock"] = lock
    return dev


def spearman(x, y):
    def rank(a):
        o = np.argsort(a, kind="stable")
        r = np.empty(len(a))
        r[o] = np.arange(len(a), dtype=float)
        for v in np.unique(a):
            m = a == v
            r[m] = r[m].mean()
        return r
    rx, ry = rank(np.asarray(x, float)), rank(np.asarray(y, float))
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def selection_eval(M, sel, tsel, S_dev, half=""):
    """Pooled test metrics of the dev-selected starts vs the per-material median start and the training-objective selection."""
    n = len(sel)
    out = {}
    for k in ("e_prof", "el1"):
        Xm = M[k + half].sum(-1)                                   # [R, n]
        out[k] = {"selected": float(Xm[sel, np.arange(n)].sum()), "median_start": float(np.median(Xm, 0).sum()),
                  "training_selected": float(Xm[tsel, np.arange(n)].sum()),
                  "median_start_elementwise": float(np.median(M[k + half], 0).sum())}
    if not half:
        e = M["e_prof"].sum(-1)
        rho = [spearman(S_dev[:, m], e[:, m]) for m in range(n)]
        out["spearman_S_dev_vs_test_e_prof"] = rho
    return out


def e1_labels(s, sa, sb):
    rho = np.nan_to_num(np.asarray(s["spearman_S_dev_vs_test_e_prof"], float), nan=0.0)
    predictive = bool((rho > 0).sum() >= 7 and np.median(rho) >= 0.5)
    c = s["e_prof"]["selected"] / s["e_prof"]["median_start"] - 1
    ca = sa["e_prof"]["selected"] / sa["e_prof"]["median_start"] - 1
    cb = sb["e_prof"]["selected"] / sb["e_prof"]["median_start"] - 1
    rule = {"e_prof_le_0.9_median_start": s["e_prof"]["selected"] <= 0.9 * s["e_prof"]["median_start"],
            "e_prof_le_training_selected": s["e_prof"]["selected"] <= s["e_prof"]["training_selected"],
            "el1_le_1.05_median_start": s["el1"]["selected"] <= 1.05 * s["el1"]["median_start"],
            "resolved": resolved(c, ca, cb)}
    return {"PREDICTIVE": predictive, "n_positive_rho": int((rho > 0).sum()), "median_rho": float(np.median(rho)),
            "SELECTION_EFFECTIVE": bool(all(rule.values())), "rule": rule, "contrast_vs_median_start": c,
            "contrast_a": ca, "contrast_b": cb}


def angular_label(con, broad_facts, control, qualified, flags=()):
    pe, st = con["e_prof"]["all"], con["instability"]["all"]
    peak = bool(pe["median"] <= -0.15 and pe["hi"] < 0 and pe["resolved"])
    stab = bool(st["median"] <= -0.25 and st["hi"] < 0 and st["resolved"])
    broad = bool(all(broad_facts["rule"].values()))
    gains = peak + stab
    if gains == 2 and broad:
        lab = "EFFECTIVE"
    elif gains == 1 and broad:
        lab = "PARTIAL"
    elif gains >= 1:
        lab = "TRADE_OFF"
    else:
        lab = "INEFFECTIVE" if broad else "REGRESSION"
    measured = lab
    if lab in ("INEFFECTIVE", "REGRESSION") and control is False:
        lab += "+CONTROL_NOT_DEMONSTRATED"
    if not qualified:
        lab = "UNQUALIFIED"
    else:
        lab += "".join("+" + f for f in flags)
    return {"PEAK_GAIN": peak, "STABILITY_GAIN": stab, "BROAD_PRESERVED": broad, "measured_label": measured, "label": lab,
            "flags": list(flags), "qualified": bool(qualified),
            "control_passed": control, "positive_with_failed_control": bool(gains >= 1 and control is False)}


def broad_facts(MX, MO, MP, con):
    px = MX["el1"].sum((1, 2))
    pp = MP["el1"].sum((1, 2))
    vs_p = float(np.median(px) / np.median(pp) - 1)
    # per material: median over starts of the per-start mean-over-probes el1 (protocol pre_freeze_clarifications), all materials
    mat = np.median(MX["el1"].mean(-1), 0) / np.median(MO["el1"].mean(-1), 0) - 1
    rule = {"el1_vs_O0_le_+5pct": con["el1"]["all"]["median"] <= 0.05, "el1_vs_P_hist_le_+5pct": vs_p <= 0.05,
            "shoulder_l1_le_+10pct": con["shoulder_l1"]["all"]["median"] <= 0.10,
            "tail_l1_le_+10pct": con["tail_l1"]["all"]["median"] <= 0.10,
            "grazing_el1_le_+10pct": con["el1_grazing"]["all"]["median"] <= 0.10,
            "no_material_el1_gt_+10pct": bool((mat <= 0.10).all())}
    return {"el1_vs_P_hist": vs_p, "material_el1_change": mat, "rule": {k: bool(v) for k, v in rule.items()}}


def synthetic_controls(root, raw, n, ds, pk, dev, ctx=None):
    fits = {S: load_fit(root, S, n, 4, ctx=ctx) for S in SYN}
    out = {"available": {S: f is not None for S, f in fits.items()}}
    if all(f is None for f in fits.values()):
        return out, {}
    synthetic_truth_check(fits, raw)
    truth = next(f["truth"] for f in fits.values() if f is not None)
    tgt = predict_units("P", [truth], ds["wi"], ds["wo"])[0]
    tpk = predict_units("P", [truth], pk["peak_wi"], pk["peak_wo"])[0] @ LUMA
    tpk_set = {**pk, "ref": tpk, "ref_a": tpk, "ref_b": tpk}
    M, arrays = {}, {}
    for S, f in fits.items():
        if f is None:
            continue
        F = predict_units("P", f["states"], ds["wi"], ds["wo"])
        m = dict(crit_terms(F, tgt, ds["w"], ds["rad"]))
        m.update(peak_metrics(predict_units("P", f["states"], pk["peak_wi"], pk["peak_wo"]) @ LUMA, tpk_set))
        obj = np.asarray(f["fit"]["final_terms"]["objective"], float).reshape(4, n)
        M[S] = {"el1": m["el1"].sum((1, 2)), "e_prof": m["e_prof"].sum((1, 2)), "e_prof_material": m["e_prof"].sum(-1),
                "objective": obj.sum(1), "fit_seconds": f["fit"]["seconds"]}
        arrays.update({f"synthetic__{S}__{k}": m[k] for k in ("el1", "e_prof", "s_h")})
    med = lambda S, k: float(np.median(M[S][k])) if S in M else None
    sc = {}
    if "S_O0" in M and "S_O1" in M:
        sc["SC_O1"] = {"objective_S_O1": med("S_O1", "objective"), "objective_S_O0": med("S_O0", "objective"),
                       "el1_ratio": med("S_O1", "el1") / med("S_O0", "el1"),
                       "passed": bool(med("S_O1", "objective") <= med("S_O0", "objective") and med("S_O1", "el1") <= 1.05 * med("S_O0", "el1"))}
    for ctl, S in (("SC_D", "S_D"), ("SC_E2", "S_E2")):
        if "S_O0" in M and S in M:
            r = med(S, "e_prof") / med("S_O0", "e_prof")
            sc[ctl] = {"e_prof_ratio": r, "paired_per_start_ratio": M[S]["e_prof"] / M["S_O0"]["e_prof"], "passed": bool(r <= 0.8)}
    ss = dev.get("synthetic_selection", {})
    if "S_O0" in M and ss.get("available"):
        sel = np.asarray(ss["selected_start"], int)
        e = M["S_O0"]["e_prof_material"]                               # [4, n]
        ok = e[sel, np.arange(n)] <= np.median(e, 0)
        sc["SC_E1"] = {"selected_start": sel, "material_ok": ok, "n_ok": int(ok.sum()), "passed": bool(ok.sum() >= 6)}
    for ctl in ("SC_O1", "SC_D", "SC_E2", "SC_E1"):
        sc.setdefault(ctl, {"passed": None, "status": "NOT_AVAILABLE"})
    out.update({"pooled": M, "controls": sc})
    return out, arrays


def optimization_report(f):
    fj = f["fit"]
    pg = np.asarray(fj["terminal_projected_gradient_norm"], float)
    ev = np.asarray(fj["evaluations_per_unit"], float)
    return {"stationary_fraction_0.5pct": fj["stationary_fraction_0.5pct"], "projected_gradient_median": float(np.median(pg)),
            "projected_gradient_max": float(pg.max()), "gate_fraction_at_bounds": fj["gate_fraction_at_bounds"],
            "final_objective_median": float(np.median(fj["final_terms"]["objective"])), "budget": fj["budget"],
            "evaluations_max": float(ev.max()), "evaluations_median": float(np.median(ev)),
            "gradient_evaluations_median": float(np.median(fj["gradient_evaluations_per_unit"])),
            "terminated_early": int(np.sum(fj["terminated_early"])),
            "memory_resets_median": (float(np.median(fj["memory_resets_per_unit"])) if "memory_resets_per_unit" in fj else None),
            "seconds": fj["seconds"], "peak_allocated_bytes": fj["peak_allocated_bytes"]}


def run_test(root, smoke=False):
    dev = leak_guard(root, smoke)                                   # before ANY test array is opened
    out = root / "test_eval"
    if (out / "test.json").exists():
        raise Stop("TEST_EXISTS: test_eval/test.json")
    t0 = time.monotonic()
    rows, raw, _, _, ref, seen = CM.inputs()
    changed = sorted(k for k, h in dev.get("inputs_sha256", {}).items() if seen.get(k) != h)
    if changed:
        raise Stop(f"LEAK_GUARD: frozen inputs differ from the dev decision {changed}")
    ref_sha = KB.sha(IO.ANGULAR)
    seen.update(KA.authenticate_k())
    n = len(rows)
    cohorts = {"all": list(range(n)), **{c: [i for i, r in enumerate(rows) if r["cohort"] == c] for c in ("development", "held_out")}}
    dstat = data_status(root, ("test.npz", "test_peak.npz", "dev.npz", "dev_peak.npz"),
                        ("test", "test_peak", "dev", "dev_peak", "isolation", "train_coverage"), smoke)
    bad = [s for s in ("test", "test_peak", "isolation") if dstat["qualified"][s] is False]
    if bad:
        raise Stop(f"DATA_UNQUALIFIED: {bad}")
    ctx = fit_data_context(root, seen)
    fits = {arm: load_fit(root, arm, n, 8, ctx=ctx) for arm in PHYS}
    F_spec = dev["stage_F"]["spec"]
    F_present = (root / "fits/F/states.npz").exists()
    if dev["stage_F"]["run"]:
        fits["F"] = load_fit(root, "F", n, 8, (F_spec["method"], F_spec["data"], F_spec["objective"], False), ctx=ctx)
    for arm in PHYS:
        if (fits[arm] is None) != (arm not in dev["fits_sha256"]):
            raise Stop(f"LEAK_GUARD: arm set changed after the dev decision ({arm})")
    fits = {a: f for a, f in fits.items() if f is not None}
    kst, _, kfj = KA.states(KA.K_ROOT / "fit_primary", n)
    ds, pk = load_sets(root, "test", n)
    hev, hpk = hist_sets(ref, n)
    idx = boot_index(cohorts)
    arms = {**{a: ("P", f["states"]) for a, f in fits.items()}, "P_hist": ("P", kst["P"]), "A_hist": ("A", kst["A"])}
    M, H, prof, hprof, timing = {}, {}, {}, {}, {}
    for arm, (kind, sts) in arms.items():
        t = time.monotonic()
        M[arm], prof[arm] = evaluate(kind, sts, ds, pk)
        H[arm], hprof[arm] = evaluate(kind, sts, hev, hpk)
        timing[arm] = time.monotonic() - t
        print("test evaluated", arm, round(timing[arm], 1), "s", flush=True)
    # descriptive dev criterion for F (dev data only; F was fitted after the dev decision)
    F_dev = None
    if "F" in fits:
        dds, dpk = load_sets(root, "dev", n)
        Fm, _ = evaluate("P", fits["F"]["states"], dds, dpk)
        S, s = selection_dev(Fm)
        F_dev = {"S_dev": S, "dev_criterion": s, "note": "computed in test mode from dev data only (descriptive E1 for F)"}
    # synthetic controls
    syn, syn_arrays = synthetic_controls(root, raw, n, ds, pk, dev, ctx)
    sc = syn.get("controls", {})
    # qualification
    qual = {f"data_{s}": dstat["qualified"][s] for s in ("test", "test_peak", "dev", "dev_peak", "isolation")}
    budgets = {a: f["fit"]["budget"] for a, f in fits.items()}
    qual["budgets_equal"] = len(set(budgets.values())) == 1
    o0 = np.asarray(fits["O0"]["fit"]["final_terms"]["objective"], float).reshape(8, n)[:4]
    kP = np.asarray(kfj["P"]["final_objective"], float).reshape(4, n)
    if kfj["start_keys"] != KB.START_KEYS[:4]:
        raise Stop(f"O0_REPRODUCTION_KEYS: WL104K P start keys {kfj['start_keys']} != {KB.START_KEYS[:4]}")
    rep = np.abs(o0 - kP) / np.abs(kP)                              # [4 WL104K starts, n materials]
    rep_max = float(rep.max())
    rep_status = ("REPRODUCED" if rep_max <= O0_REPRO_OK else O0_FLAG if rep_max <= O0_REPRO_UNQUALIFIED else "UNQUALIFIED")
    flags = (O0_FLAG,) if rep_status == O0_FLAG else ()
    qual["O0_reproduction_within_1e-2"] = bool(rep_max <= O0_REPRO_UNQUALIFIED)
    qualified = all(v is not False for v in qual.values())
    # contrasts and labels
    con, hcon, labels, broad = {}, {}, {}, {}
    for X in [a for a in ("O1", "D", "E2", "F") if a in M]:
        con[X] = contrast_table(M[X], M["O0"], cohorts, idx)
        hcon[X] = contrast_table(H[X], H["O0"], cohorts, idx)
        broad[X] = broad_facts(M[X], M["O0"], M["P_hist"], con[X])
        if X == "F":
            cs = [sc.get(CONTROL[m], {}).get("passed") for m in dev["stage_F"]["eligible"]]
            control = False if False in cs else (None if None in cs else True)
        else:
            control = sc.get(CONTROL[X], {}).get("passed")
        uses_cov = X == "D" or (X == "F" and F_spec["data"] == "coverage")
        q = qualified and not (uses_cov and dstat["qualified"]["train_coverage"] is False)
        labels[X] = angular_label(con[X], broad[X], control, q, flags)
    # descriptive: historical baselines vs O0 (unpaired 4 vs 8 starts -> pooled medians only)
    base = {a: {k: float(np.median(M[a][k].sum((1, 2))) / np.median(M["O0"][k].sum((1, 2))) - 1) for k in ("el1", "e_prof", "e_h")}
            for a in ("P_hist", "A_hist")}
    # E1 (primary on O0, selections read from dev.json only)
    e1 = {}
    for arm in [a for a in ("O0", "O1", "D", "E2") if a in M]:
        sel = np.asarray(dev["selection"]["dev_criterion"][arm], int)
        tsel = np.asarray(dev["selection"]["training_objective"][arm], int)
        S_dev = np.asarray(dev["selection"]["S_dev"][arm], float)
        s = selection_eval(M[arm], sel, tsel, S_dev)
        s_a, s_b = selection_eval(M[arm], sel, tsel, S_dev, "_a"), selection_eval(M[arm], sel, tsel, S_dev, "_b")
        e1[arm] = {**s, "labels": e1_labels(s, s_a, s_b), "selected_start": sel, "training_selected_start": tsel,
                   "role": "primary" if arm == "O0" else "descriptive"}
    if F_dev is not None:
        sel = F_dev["dev_criterion"]
        tsel = selection_training(fits["F"]["fit"], 8, n)[1]
        s = selection_eval(M["F"], sel, tsel, F_dev["S_dev"])
        e1["F"] = {**s, "labels": e1_labels(s, selection_eval(M["F"], sel, tsel, F_dev["S_dev"], "_a"),
                                            selection_eval(M["F"], sel, tsel, F_dev["S_dev"], "_b")),
                   "selected_start": sel, "training_selected_start": tsel, "role": "descriptive (selection computed after dev decision)"}
    e1_primary = {**e1["O0"]["labels"], "control_SC_E1": sc.get("SC_E1", {}).get("passed"), "qualified": bool(qualified),
                  "flags": list(flags)}
    out.mkdir(parents=True, exist_ok=True)
    arrays = {f"test__{a}__{k}": v for a in M for k, v in M[a].items()}
    arrays.update({f"hist__{a}__{k}": H[a][k] for a in H for k in ("el1", "ealb", "eshape", "e_core", "e_shoulder", "e_prof", "s_h", "e_h")})
    arrays.update({f"profile__{a}": v for a, v in prof.items()})
    arrays.update({f"hist_profile__{a}": v for a, v in hprof.items()})
    arrays.update({"profile_radius_deg": rings(pk)[0], "hist_profile_radius_deg": rings(hpk)[0], "polar_deg": ds["polar"],
                   "profile_ref_luminance": ring_means(pk["ref"], pk), "hist_profile_ref_luminance": ring_means(hpk["ref"], hpk)})
    arrays.update(syn_arrays)
    np.savez_compressed(out / "arrays.npz", **arrays)
    res = {"stage": "test", "materials": [r["tag"] for r in rows], "cohorts": cohorts, "arms_evaluated": list(M),
           "F": {"eligible_at_dev": dev["stage_F"]["run"], "spec": F_spec, "evaluated": "F" in M,
                 "present_but_not_eligible": bool(F_present and not dev["stage_F"]["run"])},
           "qualification": qual, "qualified": qualified, "flags": list(flags), "smoke": bool(smoke), "leak_lock": dev["_lock"],
           "O0_reproduction_max_relative": rep_max,
           "O0_reproduction": {"relative_per_unit": rep, "start_keys": KB.START_KEYS[:4], "max": rep_max, "status": rep_status,
                               "rule": "<= 1e-4 reproduced; (1e-4, 1e-2] O0_REPRODUCTION_DISCREPANCY flag (qualified); "
                                       "> 1e-2 UNQUALIFIED (protocol arms.O0)"},
           "labels": labels, "broad_preserved_facts": broad, "contrasts_vs_O0": con, "historical_baselines_vs_O0": base,
           "summary": {a: summary(M[a], cohorts) for a in M},
           "secondary_historical": {"note": "WL104D Amendment-1 reference: secondary, repeatedly inspected; descriptive only.",
                                    "summary": {a: summary(H[a], cohorts) for a in H}, "contrasts_vs_O0": hcon},
           "selection_E1": {"primary_O0": e1_primary, "per_arm": e1},
           "synthetic": syn, "optimization": {a: optimization_report(f) for a, f in fits.items()},
           "data": dstat, "data_sha256": {f: KB.sha(root / "data" / f) for f in ("test.npz", "test_peak.npz", "dev.npz", "dev_peak.npz")},
           "dev_json_sha256": KB.sha(root / "dev_eval/dev.json"), "fits_sha256": {a: f["sha"] for a, f in fits.items()},
           "fits_data_sha256": {a: f["data_sha256"] for a, f in fits.items()}, "code_sha256": code_sha(),
           "definitions": DEFINITIONS, "arrays_sha256": KB.sha(out / "arrays.npz"), "timing_seconds": timing,
           "commit": KB.commit(), "protocol_sha256": KB.sha(KB.PROTOCOL), "inputs_sha256": seen, "seconds": time.monotonic() - t0}
    if KB.sha(IO.ANGULAR) != ref_sha:
        raise Stop("EVALUATION_REFERENCE_CHANGED")
    if code_sha() != dev["code_sha256"] or KB.sha(KB.PROTOCOL) != dev["protocol_sha256"]:
        raise Stop("CODE_CHANGED: evaluation code or protocol changed during the test evaluation")
    write(out / "test.json", res)
    print("test done", {X: v["label"] for X, v in labels.items()}, "E1", e1_primary["PREDICTIVE"], e1_primary["SELECTION_EFFECTIVE"],
          round(time.monotonic() - t0, 1), "s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("dev", "test"))
    ap.add_argument("root", type=Path)
    ap.add_argument("--smoke", action="store_true", help="pipeline check only (see module doc); recorded in the outputs")
    args = ap.parse_args()
    try:
        (run_dev if args.mode == "dev" else run_test)(args.root.resolve(), args.smoke)
    except Stop as e:
        print("STOP", e, flush=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
