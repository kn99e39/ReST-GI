"""WL104K-B data stage: generate and qualify the three NEW angular lane sets (train_coverage, dev, test) and the two peak grids
(dev_peak, test_peak) exactly per protocol.data. Every set uses incidence_stratified directions (per-set direction seed) and the
proposal_refined lanes (1/2 WL83 defensive mixture, 1/2 radially stratified in 8 equal annuli around the mirror direction,
one-sample balance-heuristic q). Physical responses come from the frozen layered walk (DA.ref_eval, two independent halves) on
ACTIVE lanes / grid points only, with the protocol initstates and pair0 offsets. Before each walk the stream is checked against the
explicit registry of every recorded range (and the other new ranges). Qualification: precision_rule, support_rule, isolation_rule.

  python data104kb.py <out_dir> [--smoke]     (smoke: 64 lanes, 512 walks, 512 peak walks; same code path)

Outputs in <out_dir> (normally results/ecro_wl84_104kb/primary/data):
  train_coverage.npz, dev.npz, test.npz : wi, wo [n*P, L, 3]; q, use [n*P, L]; a, b [n*P, L, 3] (walk halves); mid [n*P, L];
                                          incidence [n*P, 3]; polar_deg, azimuth_deg [n*P]   (material-major, then probe)
  dev_peak.npz, test_peak.npz           : peak_wi, peak_wo, peak_a, peak_b [n*P, K, 3]; use [n*P, K]; radius_deg [K]
  data.json                             : qualification, seconds, sequence ranges, per-material precision, sha256 of outputs,
                                          code_sha256 of the generator code, git status of the frozen code.
Non-smoke runs refuse to start (UNFROZEN_CODE) while git lists data104kb.py, common104kb.py or protocol.json as modified or
untracked. Any unqualified set stops the whole data stage (DATA_UNQUALIFIED; protocol pre_freeze_clarifications).
Loaders: load_lanes(path, n), load_peak(path, n); test_lock(root) is the single test-set lock definition (also called by
eval104kb), require_dev_decision(root) delegates to it and is used by both loaders for test_* files.
"""
import argparse
import re
import subprocess
import time
from pathlib import Path

import numpy as np

import common104kb as KB
from data104d import oriented, mirror

CM, IO, DA = KB.CM, KB.IO, KB.DA
D, S = KB.P["data"], KB.P["seeds"]
LUMA = np.asarray(KB.LUMA, np.float64)
POLAR_EDGES = (1.0, 12.5, 27.5, 42.5, 53.75, 58.75, 61.25, 66.25, 75.0, 83.0, 88.0)
NPB = 2 * (len(POLAR_EDGES) - 1)                 # 20 probes per material (2 per stratum)
ANNULI = (0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 4.5, 7.0, 12.0)
SUPPORT = (0.0, 0.5, 1.0, 3.0, 12.0)
BAND, GRAZING = (53.75, 66.25), 75.0
N_AZ = 16
HORIZON = 1e-7
MIN_SEQUENCE = 2 ** 44
LANE_SETS = ("train_coverage", "dev", "test")
# Repo-relative generator code recorded in data.json, and the subset that must be committed before a non-smoke run.
CODE_FILES = ("experiments/ecro_wl84_104kb/data104kb.py", "experiments/ecro_wl84_104kb/common104kb.py",
              "experiments/ecro_wl84_104d/data104d.py", "experiments/fnto_compiled81/data81.py")
FROZEN_CODE = ("experiments/ecro_wl84_104kb/data104kb.py", "experiments/ecro_wl84_104kb/common104kb.py",
               "experiments/ecro_wl84_104kb/protocol.json")
STAGEA_JSON = KB.OUT / "stageA" / "stageA.json"
# Every recorded layered-walk stream: (label, initstate, start, end) in global PCG32 sequence indices (pair0 * walks + lane).
HISTORICAL = [("WL104D lanes", 104460, 0, 735870976),
              ("WL104D peak grid", 104470, 0, 189480960),
              ("WL104D Amendment 1", 10441901, 12288000000, 12363497472),
              ("WL104J fit_set", 1041011, 2 ** 40, 2 ** 40 + 1470906368),
              ("WL104J transfer sets", 1041021, 2 ** 41, 2201824591872)]
HISTORICAL += [(f"WL104K-A scene lanes si={si}", 1041210 + si, 2 ** 42 + si * 32768000000,
                2 ** 42 + si * 32768000000 + 105398 * 8192) for si in range(3)]
# Stage A default (stageA104kb: initstate 1041301, pair0 2^31, 65536 walks, 8190 pairs); replaced by load_registry() with the
# range recorded in results/ecro_wl84_104kb/stageA/stageA.json when that file exists.
STAGE_A = ("WL104K-B Stage A", 1041301, 2 ** 31 * 65536, 2 ** 31 * 65536 + 8190 * 65536)
REGISTRY = HISTORICAL + [STAGE_A]


# ------------------------------------------------------------------ directions
def incidence_stratified(n, rng):
    """protocol incidence_stratified: per material 10 polar strata x 2 probes (azimuth [0, 90) and [90, 180)), stratum-major.
    -> polar_deg, azimuth_deg [n, 20], incidence [n * 20, 3]."""
    lo, hi = np.array(POLAR_EDGES[:-1]), np.array(POLAR_EDGES[1:])
    polar = rng.uniform(lo[None, :, None], hi[None, :, None], (n, len(lo), 2)).reshape(n, NPB)
    azimuth = (rng.uniform(0.0, 90.0, (n, len(lo), 2)) + np.array([0.0, 90.0])).reshape(n, NPB)
    return polar, azimuth, KB.sph(polar, azimuth).reshape(-1, 3)


def mixture_pdf(desc_lane, wi, wo):
    """Density of DA.mixture (WL83 defensive mixture) at arbitrary wo (same expression as data81.mixture)."""
    a1, a2 = desc_lane[:, 0], np.minimum(1.0, desc_lane[:, 1] * desc_lane[:, 9])
    w = DA.W_MIX
    return (w[0] * np.maximum(wo[:, 2], 0) / np.pi + w[1] / (2 * np.pi) + w[2] * DA.vndf_pdf(wi, wo, a1)
            + w[3] * DA.vndf_pdf(wi, wo, a2))


def proposal_refined(desc_g, incidence, lanes, rng):
    """protocol proposal_refined. desc_g [G, 10], incidence [G, 3] -> wi, wo [G, lanes, 3], q, use [G, lanes], check dict.
    Lanes [0, lanes/2): DA.mixture; then 8 blocks of lanes/16, block k uniform in solid angle in annulus k around the mirror."""
    if lanes % 16:
        raise ValueError("proposal_refined needs lanes divisible by 16")
    G, half, per = len(incidence), lanes // 2, lanes // 16
    wi = np.ascontiguousarray(np.broadcast_to(incidence[:, None, :], (G, lanes, 3)))
    wo = np.empty_like(wi)
    use = np.ones((G, lanes), bool)
    wo_m, q_m, use_m = DA.mixture(np.repeat(desc_g, half, axis=0), wi[:, :half].reshape(-1, 3), rng)
    wo[:, :half], use[:, :half] = wo_m.reshape(G, half, 3), use_m.reshape(G, half)
    c = np.cos(np.radians(ANNULI))
    centre = mirror(incidence)[:, None, :]
    for k in range(len(ANNULI) - 1):
        cos_r = rng.uniform(c[k + 1], c[k], (G, per))
        phi = rng.uniform(0.0, 2 * np.pi, (G, per))
        wo[:, half + k * per: half + (k + 1) * per] = oriented(centre, np.arccos(cos_r), phi)
    use &= wo[..., 2] > HORIZON
    wo[~use] = (0.0, 0.0, 1.0)                    # zero-weight placeholder for every invalid draw
    q_mix = mixture_pdf(np.repeat(desc_g, lanes, axis=0), wi.reshape(-1, 3), wo.reshape(-1, 3)).reshape(G, lanes)
    dot = np.sum(mirror(wi) * wo, axis=-1)
    q = 0.5 * q_mix
    for k in range(len(ANNULI) - 1):
        q += (1 / 16) * ((dot > c[k + 1]) & (dot <= c[k])) / (2 * np.pi * (c[k] - c[k + 1]))
    ok = use[:, :half].reshape(-1)
    rel = np.abs(q_mix[:, :half].reshape(-1)[ok] / q_m[ok] - 1)
    return wi, wo, q, use, {"mixture_pdf_max_rel_diff": float(rel.max()) if rel.size else 0.0,
                            "active_fraction": float(use.mean()), "min_q": float(q.min())}


def peak_radii(text):
    """'0 (exact mirror) + 0.25 .. 4.0 step 0.25 [+ 5, 6, 8]' -> ring radii (deg) without the 0 point."""
    m = re.fullmatch(r"0 \(exact mirror\) \+ 0\.25 \.\. 4\.0 step 0\.25(?: \+ ([\d., ]+))?", text.strip())
    if m is None:
        raise ValueError(f"unparsed peak radii: {text}")
    extra = [float(x) for x in m.group(1).split(",")] if m.group(1) else []
    return np.r_[0.25 * np.arange(1, 17), extra]


def peak_layout(incidence, radii):
    """WL104D peak-grid layout: exact mirror, then per ring 16 azimuths (k + 0.5) * 22.5 deg. -> wi, wo [G, K, 3], radius_deg [K]."""
    rad = np.r_[0.0, np.repeat(radii, N_AZ)]
    az = np.r_[0.0, np.tile((np.arange(N_AZ) + 0.5) * (360.0 / N_AZ), len(radii))]
    wo = oriented(mirror(incidence)[:, None, :], np.radians(rad)[None, :], np.radians(az)[None, :])
    return np.ascontiguousarray(np.broadcast_to(incidence[:, None, :], wo.shape)), wo, rad


# ------------------------------------------------------------------ stream registry and walks
def load_registry():
    """Set REGISTRY = HISTORICAL + Stage A (range read from stageA.json when present). Stops if a historical (non-Stage-A) range
    reaches 2^44 or the Stage A initstate differs from protocol seeds.stageA_walks. -> provenance dict for data.json."""
    late = [r[0] for r in HISTORICAL if r[3] > MIN_SEQUENCE]       # end is exclusive: [start, end) must lie below 2^44
    if late:
        raise IO.Stop(f"STREAM_REGISTRY historical range not below 2^44: {late}")
    entry, src = STAGE_A, "built-in default (stageA.json absent)"
    if STAGEA_JSON.exists():
        j = KB.read(STAGEA_JSON)
        seed, p0 = int(j["seeds"]["walk"]), int(j["seeds"]["pair0"])
        w, npair = int(j["grid"]["walks"]), int(j["grid"]["pairs"])
        entry = ("WL104K-B Stage A", seed, p0 * w, (p0 + npair) * w)
        src = {"path": STAGEA_JSON.relative_to(KB.REPO).as_posix(), "sha256": KB.sha(STAGEA_JSON)}
    if entry[1] != int(S["stageA_walks"]):
        raise IO.Stop(f"STREAM_REGISTRY Stage A initstate {entry[1]} != protocol {S['stageA_walks']}")
    REGISTRY[:] = HISTORICAL + [entry]
    return {"stage_A": list(entry), "stage_A_source": src, "stage_A_equals_default": entry == STAGE_A,
            "historical_end_max": int(max(r[3] for r in HISTORICAL)), "historical_below_2^44": True}


def check_stream(label, initstate, rng_range, new, smoke):
    """Registry rule: new initstate, range >= 2^44, disjoint from every recorded and every other new range. Stops (non-smoke)."""
    s0, s1 = rng_range
    used = {r[1] for r in REGISTRY} | {r[1] for r in new}
    clash = [r[0] for r in REGISTRY + new if s0 < r[3] and r[2] < s1]
    res = {"label": label, "initstate": int(initstate), "start": int(s0), "end": int(s1),
           "initstate_new": initstate not in used, "at_or_above_2^44": s0 >= MIN_SEQUENCE, "overlaps": clash}
    res["passed"] = bool(res["initstate_new"] and res["at_or_above_2^44"] and not clash)
    if not res["passed"] and not smoke:
        raise IO.Stop(f"STREAM_REGISTRY {res}")
    new.append((label, int(initstate), int(s0), int(s1)))
    return res


def walk(desc, mid, wi, wo, use, walks, seed, pair0, label, new, smoke):
    """DA.ref_eval on active entries only; inactive entries get a = b = 0. Arrays [..., 3] with use [...]."""
    act = use.ravel()
    npair = int(act.sum())
    reg = check_stream(label, seed, (pair0 * walks, (pair0 + npair) * walks), new, smoke)
    t0 = time.monotonic()
    a, b = DA.ref_eval(desc, mid.ravel()[act], wi.reshape(-1, 3)[act], wo.reshape(-1, 3)[act], walks, seed, pair0=pair0)
    secs = time.monotonic() - t0
    A, B = np.zeros((act.size, 3)), np.zeros((act.size, 3))
    A[act], B[act] = a, b
    info = {"pairs_walked": npair, "walks_per_pair": int(walks), "walks": npair * int(walks), "seconds_walks": secs,
            "walks_per_second": npair * walks / max(secs, 1e-9), "initstate": int(seed), "pair0": int(pair0),
            "sequence_range": [reg["start"], reg["end"]], "registry": reg}
    return A.reshape(use.shape + (3,)), B.reshape(use.shape + (3,)), info


def make_set(name, desc, lanes, walks, new, smoke):
    cfg = D[name]
    n = len(desc)
    dseed = int(S[f"{name}_directions"])
    if int(cfg["incidence"].split("seed")[-1]) != dseed or int(S[f"{name}_walks"]) != cfg["walk_seed"]:
        raise ValueError(f"{name}: seed mismatch between protocol data and seeds")
    rng = np.random.default_rng(dseed)
    polar, azimuth, inc = incidence_stratified(n, rng)
    wi, wo, q, use, chk = proposal_refined(np.repeat(desc, NPB, axis=0), inc, lanes, rng)
    mid = np.ascontiguousarray(np.broadcast_to(np.repeat(np.arange(n), NPB)[:, None], use.shape)).astype(np.int64)
    a, b, info = walk(desc, mid, wi, wo, use, walks, cfg["walk_seed"], cfg["walk_pair0"], name, new, smoke)
    d = {"wi": wi, "wo": wo, "q": q, "use": use, "a": a, "b": b, "mid": mid, "incidence": inc,
         "polar_deg": polar.reshape(-1), "azimuth_deg": azimuth.reshape(-1)}
    info.update({"direction_seed": dseed, "lanes_per_probe": int(lanes), "proposal_check": chk})
    return d, info


def make_peak(name, desc, lane_set, walks, new, smoke):
    cfg = D[name]["peak_grid"]
    if not cfg["azimuths"].startswith("(k + 0.5) * 22.5 deg"):
        raise ValueError(f"{name}: unexpected peak azimuth layout")
    if int(S[f"{name}_peak_walks"]) != cfg["walk_seed"]:
        raise ValueError(f"{name}: peak seed mismatch between protocol data and seeds")
    radii = peak_radii(cfg["radii_deg"])
    wi, wo, rad = peak_layout(lane_set["incidence"], radii)
    use = wo[..., 2] > HORIZON
    mid = np.ascontiguousarray(np.broadcast_to(lane_set["mid"][:, :1], use.shape)).astype(np.int64)
    a, b, info = walk(desc, mid, wi, wo, use, walks, cfg["walk_seed"], cfg["walk_pair0"], name + "_peak", new, smoke)
    d = {"peak_wi": wi, "peak_wo": wo, "peak_a": a, "peak_b": b, "use": use, "radius_deg": rad}
    info.update({"points_per_probe": int(wo.shape[1]), "radii_deg": [0.0] + radii.tolist()})
    return d, info


# ------------------------------------------------------------------ qualification
def lane_precision(d, n):
    """precision_rule per material (half split a vs b, WL82 terms with w = use / q)."""
    w = np.where(d["use"], 1.0 / d["q"], 0.0)
    t = CM.wl82_terms(d["a"], d["b"], w)
    sh, al, e1 = (t[k].reshape(n, NPB) for k in ("eshape", "ealb", "el1"))
    pol = d["polar_deg"].reshape(n, NPB)
    rows = []
    for i in range(n):
        band = (pol[i] >= BAND[0]) & (pol[i] <= BAND[1])
        graze = pol[i] >= GRAZING
        r = {"band_shape": float(0.5 * sh[i, band].mean()), "grazing_shape": float(0.5 * sh[i, graze].mean()),
             "albedo": float(0.5 * al[i].mean()), "all_shape": float(0.5 * sh[i].mean()),
             "el1_half_noise": float(0.5 * e1[i].mean()), "band_probes": int(band.sum()), "grazing_probes": int(graze.sum())}
        r["passed"] = bool(r["band_shape"] <= 0.015 and r["grazing_shape"] <= 0.025 and r["albedo"] <= 0.008)
        rows.append(r)
    return rows


def peak_precision(p, n):
    """Per material: median / max over probes of 0.5 |h_a - h_b| / h (h = max luminance of the half mean over the grid)."""
    la, lb = p["peak_a"] @ LUMA, p["peak_b"] @ LUMA
    ha, hb, h = la.max(-1), lb.max(-1), (0.5 * (la + lb)).max(-1)
    r = (0.5 * np.abs(ha - hb) / np.maximum(h, 1e-300)).reshape(n, -1)
    rows = []
    for i in range(n):
        x = {"median": float(np.median(r[i])), "max": float(r[i].max()), "min_h": float(h.reshape(n, -1)[i].min()),
             "active_points_min": int(p["use"].reshape(n, -1, p["use"].shape[-1])[i].sum(-1).min())}
        x["passed"] = bool(x["median"] <= 0.005 and x["max"] <= 0.02)
        rows.append(x)
    return rows


def support(d):
    rad = CM.mirror_radius(d["wi"], d["wo"])
    out = {}
    for lo, hi in zip(SUPPORT[:-1], SUPPORT[1:]):
        out[f"[{lo},{hi})"] = int((d["use"] & (rad >= lo) & (rad < hi)).sum(-1).min())
    return {"min_active_per_probe": out, "passed": bool(min(out.values()) >= 16)}


def row_keys(x):
    """Exact float64 row identity (-0.0 folded to +0.0)."""
    x = np.ascontiguousarray(np.asarray(x, np.float64).reshape(-1, 3) + 0.0)
    return {r.tobytes() for r in x}


def isolation(groups, incid, n):
    """groups: {name: [n, N_m] list of active outgoing [N, 3] per material}; incid: {name: [M, 3]} incident directions."""
    out_pairs, inc_pairs = {}, {}
    names = list(groups)
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            out_pairs[f"{x}|{y}"] = int(sum(len(row_keys(groups[x][m]) & row_keys(groups[y][m])) for m in range(n)))
    names = list(incid)
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            if not (x.startswith("historical") and y.startswith("historical")):   # historical sets share the WL104D grid by design
                inc_pairs[f"{x}|{y}"] = len(row_keys(incid[x]) & row_keys(incid[y]))
    within = {k: int(len(v) - len(row_keys(v))) for k, v in incid.items() if not k.startswith("historical")}
    return {"exact_equal_outgoing_active": out_pairs, "exact_equal_incidence": inc_pairs, "duplicate_incidence_within_set": within,
            "passed": bool(not any(out_pairs.values()) and not any(inc_pairs.values()) and not any(within.values()))}


def frozen_check(seen):
    bad = [k for k, h in seen.items() if KB.sha(KB.REPO / k) != h]
    return {"files": len(seen), "changed": bad, "passed": not bad}


def code_sha():
    return {rel: KB.sha(KB.REPO / rel) for rel in CODE_FILES}


def unfrozen_code():
    """git status --porcelain lines for FROZEN_CODE (modified, staged or untracked). A failing git call stops."""
    try:
        out = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all", "--", *FROZEN_CODE], cwd=KB.REPO,
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as e:
        raise IO.Stop(f"UNFROZEN_CODE: git status failed ({e})")
    return [ln for ln in out.splitlines() if ln.strip()]


# ------------------------------------------------------------------ test-set lock and loaders
def _resolve_code(key):
    """code_sha256 key -> file: absolute, repo-relative, or relative to this directory."""
    for p in (Path(key), KB.REPO / key, Path(__file__).resolve().parent / key):
        if p.is_absolute() and p.exists():
            return p
    raise IO.Stop(f"TEST_LOCKED: recorded code file {key} not found")


def test_lock(root):
    """THE test-set lock (isolation_rule + pre_freeze_clarifications), shared by the loaders here and eval104kb.run_test.
    Refuses (IO.Stop 'TEST_LOCKED: ...') unless <root>/dev_eval/dev.json exists with a matching dev.json.sha256 sidecar, fits/F
    exists when dev.json stage_F.run is true, every fits_sha256 / synthetic_fits_sha256 entry equals the current
    fits/<arm>/states.npz, protocol_sha256 equals the current protocol, and (when dev.json records them) every code_sha256
    (repo-relative or relative to this directory), data_sha256 (relative to <root>/data) and inputs_sha256 (repo-relative) entry is
    unchanged. Reads no test array. -> the dev decision dict."""
    root = Path(root)
    dev = root / "dev_eval" / "dev.json"
    side = dev.with_name("dev.json.sha256")
    if not (dev.exists() and side.exists()):
        raise IO.Stop(f"TEST_LOCKED: hashed dev decision missing under {root}")
    if KB.sha(dev) != side.read_text(encoding="utf-8").split()[0].strip():
        raise IO.Stop(f"TEST_LOCKED: dev decision hash mismatch {dev}")
    d = KB.read(dev)
    for key in ("stage_F", "fits_sha256", "protocol_sha256"):
        if key not in d:
            raise IO.Stop(f"TEST_LOCKED: dev.json lacks {key}")
    if not d["fits_sha256"]:
        raise IO.Stop("TEST_LOCKED: dev.json records no fits")
    if d["stage_F"]["run"] and not (root / "fits/F/states.npz").exists():
        raise IO.Stop("TEST_LOCKED: F is eligible but fits/F/states.npz does not exist")
    for key in ("fits_sha256", "synthetic_fits_sha256"):
        for arm, h in (d.get(key) or {}).items():
            p = root / "fits" / arm / "states.npz"
            if not p.exists() or KB.sha(p) != h:
                raise IO.Stop(f"TEST_LOCKED: fits/{arm} changed after the dev decision ({key})")
    if d["protocol_sha256"] != KB.sha(KB.PROTOCOL):
        raise IO.Stop("TEST_LOCKED: protocol changed after the dev decision")
    checks = (("code_sha256", _resolve_code), ("data_sha256", lambda k: root / "data" / k), ("inputs_sha256", lambda k: KB.REPO / k))
    for key, where in checks:
        for k, h in (d.get(key) or {}).items():
            p = where(k)
            if not p.exists() or KB.sha(p) != h:
                raise IO.Stop(f"TEST_LOCKED: {key} {k} changed after the dev decision")
    return d


def require_dev_decision(root):
    """Test-set lock used by the loaders for test_* files: delegates to test_lock (single definition)."""
    return test_lock(root)


def load_lanes(path, n, guard=True):
    """Lane set npz -> common104kb.load_lane_set layout. For a test_* file the test lock of the run root (path/../..) must pass
    unless guard=False (smoke only)."""
    path = Path(path)
    if guard and path.name.startswith("test"):
        require_dev_decision(path.parent.parent)
    return KB.load_lane_set(path, n)


def load_peak(path, n, guard=True):
    """Peak grid npz -> {'peak_wi','peak_wo' [n, P, K, 3], 'ref','ref_a','ref_b' luminance [n, P, K], 'use' [n, P, K], 'radius_deg' [K]}.
    For a test_* file the test lock of the run root (path/../..) must pass unless guard=False (smoke only)."""
    path = Path(path)
    if guard and path.name.startswith("test"):
        require_dev_decision(path.parent.parent)
    with np.load(path) as z:
        r = lambda k: z[k].reshape(n, -1, *z[k].shape[1:])
        la, lb = r("peak_a") @ LUMA, r("peak_b") @ LUMA
        return {"peak_wi": r("peak_wi"), "peak_wo": r("peak_wo"), "ref": 0.5 * (la + lb), "ref_a": la, "ref_b": lb,
                "use": r("use"), "radius_deg": z["radius_deg"].copy()}


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", type=Path)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    dirty = unfrozen_code()
    if dirty and not args.smoke:
        raise IO.Stop(f"UNFROZEN_CODE {dirty}")
    reg_info = load_registry()
    args.out.mkdir(parents=True, exist_ok=False)
    t0 = time.monotonic()
    proto_sha = KB.sha(KB.PROTOCOL)
    code0 = code_sha()
    rows, raw, compiled, desc, ref, seen = CM.inputs()
    n = len(rows)
    new, sets, info, files = [], {}, {}, {}
    for name in LANE_SETS:
        lanes = 64 if args.smoke else D[name]["lanes_per_probe"]
        walks = 512 if args.smoke else D[name]["walks_per_lane"]
        ts = time.monotonic()
        sets[name], info[name] = make_set(name, desc, lanes, walks, new, args.smoke)
        np.savez(args.out / f"{name}.npz", **sets[name])
        info[name]["seconds"] = time.monotonic() - ts
        files[name] = f"{name}.npz"
        print(name, {k: info[name][k] for k in ("pairs_walked", "walks", "seconds_walks")}, flush=True)
        if "peak_grid" in D[name]:
            pk = f"{name}_peak"
            pw = 512 if args.smoke else D[name]["peak_grid"]["walks"]
            ts = time.monotonic()
            sets[pk], info[pk] = make_peak(name, desc, sets[name], pw, new, args.smoke)
            np.savez(args.out / f"{pk}.npz", **sets[pk])
            info[pk]["seconds"] = time.monotonic() - ts
            files[pk] = f"{pk}.npz"
            print(pk, {k: info[pk][k] for k in ("pairs_walked", "walks", "seconds_walks")}, flush=True)

    # qualification
    for name in LANE_SETS:
        info[name]["precision"] = lane_precision(sets[name], n)
        info[name]["support"] = support(sets[name])
        pk = f"{name}_peak"
        if pk in sets:
            info[pk]["precision"] = peak_precision(sets[pk], n)
            if not np.array_equal(sets[pk]["peak_wi"][:, 0], sets[name]["incidence"]):
                raise IO.Stop(f"PEAK_INCIDENCE_LAYOUT {pk}")
    original = CM.load_set("fit_set", n)               # train_original (WL104J fit set; O0 / O1 / E2 training data)
    act = lambda d, m: d["wo"].reshape(n, -1, 3)[m][d["use"].reshape(n, -1)[m]]
    groups = {"train_original": [original["wo"][m][original["w"][m] > 0] for m in range(n)],
              "train_coverage": [act(sets["train_coverage"], m) for m in range(n)]}
    for name in ("dev", "test"):
        pk = sets[f"{name}_peak"]
        groups[name] = [np.concatenate([act(sets[name], m), pk["peak_wo"].reshape(n, -1, 3)[m][pk["use"].reshape(n, -1)[m]]])
                        for m in range(n)]
    incid = {"historical_train_original": original["wi"][:, :, 0].reshape(-1, 3),
             "historical_wl104d_reference": ref["wi"].reshape(-1, ref["wi"].shape[-2], 3)[:, 0]}
    incid.update({name: sets[name]["incidence"] for name in LANE_SETS})
    iso = isolation(groups, incid, n)
    iso["registry"] = {"recorded": [list(r) for r in REGISTRY], "new": [list(r) for r in new],
                       "checks": [info[k]["registry"] for k in info], "passed": all(info[k]["registry"]["passed"] for k in info),
                       **reg_info}
    iso["frozen_inputs"] = frozen_check(seen)
    iso["protocol_sha256_unchanged"] = KB.sha(KB.PROTOCOL) == proto_sha
    iso["code_sha256_unchanged"] = code_sha() == code0
    iso_ok = bool(iso["passed"] and iso["registry"]["passed"] and iso["frozen_inputs"]["passed"] and iso["protocol_sha256_unchanged"]
                  and iso["code_sha256_unchanged"])

    q = {}
    for name in LANE_SETS:
        q[name] = bool(all(r["passed"] for r in info[name]["precision"]) and info[name]["support"]["passed"] and iso_ok)
        if f"{name}_peak" in info:
            q[f"{name}_peak"] = bool(all(r["passed"] for r in info[f"{name}_peak"]["precision"]) and iso_ok)
    q["isolation"] = iso_ok
    q["all"] = bool(all(q.values()))
    report = {"qualified": q, "sets": info, "isolation": iso, "materials": [r["tag"] for r in rows],
              "cohort": [r["cohort"] for r in rows], "smoke": args.smoke,
              "walks_total": int(sum(v["walks"] for v in info.values())),
              "seconds_walks_total": float(sum(v["seconds_walks"] for v in info.values())),
              "commit": KB.commit(), "protocol_sha256": proto_sha, "inputs_sha256": seen, "code_sha256": code0,
              "git_status_frozen_code": dirty,
              "outputs_sha256": {f: KB.sha(args.out / f) for f in files.values()}, "seconds": time.monotonic() - t0}
    KB.write(args.out / "data.json", report)
    print("data", q, {k: round(v["seconds_walks"], 1) for k, v in info.items()},
          f"walks {report['walks_total']:.3e} in {report['seconds_walks_total']:.1f} s", flush=True)
    if not q["all"] and not args.smoke:                # global stop (declared pre-freeze clarification)
        raise IO.Stop(f"DATA_UNQUALIFIED {q}")


if __name__ == "__main__":
    main()
