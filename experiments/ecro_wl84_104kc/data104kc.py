"""WL104K-C data stage: generate and qualify the NEW independent angular test set (test_c lanes + test_c_peak grid), disjoint from
every earlier stream and direction set, with the unchanged WL104K-B generator functions (incidence_stratified, proposal_refined,
peak_layout, walk = data81.ref_eval layered reference) and the WL104K-B precision / support rules (protocol data).

  python data104kc.py <out_dir> [--smoke]

Outputs <out_dir>/test_c.npz, test_c_peak.npz (WL104K-B layouts) and data.json (qualification, stream registry, isolation,
hashes). Non-smoke runs refuse unless protocol.json, this file and common104kc.py are committed (UNFROZEN_CODE).
The response arrays (a, b, peak_a, peak_b) are read here only by the precision rule (half-split noise; no model evaluation).
"""
import argparse
import subprocess
import time
from pathlib import Path

import numpy as np

import common104kc as C

KB, CM, IO = C.KB, C.CM, C.IO
import data104kb as DK  # noqa: E402

FROZEN = ("experiments/ecro_wl84_104kc/data104kc.py", "experiments/ecro_wl84_104kc/common104kc.py",
          "experiments/ecro_wl84_104kc/protocol.json")


def wl104kb_ranges():
    """WL104K-B recorded new streams (data.json) + its Stage A range."""
    dj = C.read(C.KB_ROOT / "data/data.json")
    out = []
    for name, v in dj["sets"].items():
        out.append((f"WL104K-B {name}", int(v["initstate"]), int(v["sequence_range"][0]), int(v["sequence_range"][1])))
    return out, C.sha(C.KB_ROOT / "data/data.json")


def unfrozen():
    out = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all", "--", *FROZEN], cwd=C.REPO,
                         capture_output=True, text=True, check=True).stdout
    return [ln for ln in out.splitlines() if ln.strip()]


def make(desc, cfg, seeds, lanes, walks, peak_walks, new, smoke):
    n = len(desc)
    rng = np.random.default_rng(int(seeds["test_c_directions"]))
    polar, azimuth, inc = DK.incidence_stratified(n, rng)
    wi, wo, q, use, chk = DK.proposal_refined(np.repeat(desc, DK.NPB, axis=0), inc, lanes, rng)
    mid = np.ascontiguousarray(np.broadcast_to(np.repeat(np.arange(n), DK.NPB)[:, None], use.shape)).astype(np.int64)
    a, b, info = DK.walk(desc, mid, wi, wo, use, walks, int(seeds["test_c_walks"]), int(cfg["walk_pair0"]), "WL104K-C test_c", new,
                         smoke)
    lane = {"wi": wi, "wo": wo, "q": q, "use": use, "a": a, "b": b, "mid": mid, "incidence": inc,
            "polar_deg": polar.reshape(-1), "azimuth_deg": azimuth.reshape(-1)}
    info.update({"direction_seed": int(seeds["test_c_directions"]), "lanes_per_probe": int(lanes), "proposal_check": chk})
    radii = DK.peak_radii(cfg["peak_grid"]["radii_deg"])
    pwi, pwo, rad = DK.peak_layout(inc, radii)
    puse = pwo[..., 2] > DK.HORIZON
    pmid = np.ascontiguousarray(np.broadcast_to(mid[:, :1], puse.shape)).astype(np.int64)
    pa, pb, pinfo = DK.walk(desc, pmid, pwi, pwo, puse, peak_walks, int(seeds["test_c_peak_walks"]),
                            int(cfg["peak_grid"]["walk_pair0"]), "WL104K-C test_c_peak", new, smoke)
    peak = {"peak_wi": pwi, "peak_wo": pwo, "peak_a": pa, "peak_b": pb, "use": puse, "radius_deg": rad}
    pinfo.update({"points_per_probe": int(pwo.shape[1]), "radii_deg": [0.0] + radii.tolist()})
    return lane, info, peak, pinfo


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", type=Path)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    dirty = unfrozen()
    if dirty and not args.smoke:
        raise C.Stop(f"UNFROZEN_CODE {dirty}")
    cfg, seeds = C.P["data"]["test_c"], C.P["seeds"]
    reg_info = DK.load_registry()                                # WL104D ... WL104K-A + WL104K-B Stage A
    kb_ranges, kb_dj_sha = wl104kb_ranges()
    DK.REGISTRY[:] = DK.REGISTRY + kb_ranges
    args.out.mkdir(parents=True, exist_ok=False)
    t0 = time.monotonic()
    proto_sha = C.sha(C.PROTOCOL)
    rows, raw, compiled, desc, ref, seen = CM.inputs()
    n = len(rows)
    new = []
    lanes = 64 if args.smoke else int(cfg["lanes_per_probe"])
    walks = 512 if args.smoke else int(cfg["walks_per_lane"])
    pw = 512 if args.smoke else int(cfg["peak_grid"]["walks"])
    lane, info, peak, pinfo = make(desc, cfg, seeds, lanes, walks, pw, new, args.smoke)
    np.savez(args.out / "test_c.npz", **lane)
    np.savez(args.out / "test_c_peak.npz", **peak)
    info["precision"] = DK.lane_precision(lane, n)
    info["support"] = DK.support(lane)
    pinfo["precision"] = DK.peak_precision(peak, n)
    # isolation against every earlier direction set
    original = CM.load_set("fit_set", n)
    act = lambda d, m: d["wo"].reshape(n, -1, 3)[m][d["use"].reshape(n, -1)[m]]
    kb = {}
    for name in ("train_coverage", "dev", "test"):
        with np.load(C.KB_ROOT / f"data/{name}.npz") as z:
            kb[name] = {k: z[k] for k in ("wo", "use", "incidence")}
    for name in ("dev", "test"):
        with np.load(C.KB_ROOT / f"data/{name}_peak.npz") as z:
            kb[name + "_peak"] = {"wo": z["peak_wo"], "use": z["use"]}
    groups = {"train_original": [original["wo"][m][original["w"][m] > 0] for m in range(n)],
              "wl104kb_train_coverage": [act(kb["train_coverage"], m) for m in range(n)]}
    for name in ("dev", "test"):
        groups[f"wl104kb_{name}"] = [np.concatenate([act(kb[name], m), act(kb[name + "_peak"], m)]) for m in range(n)]
    groups["test_c"] = [np.concatenate([act(lane, m), peak["peak_wo"].reshape(n, -1, 3)[m][peak["use"].reshape(n, -1)[m]]])
                        for m in range(n)]
    incid = {"historical_train_original": original["wi"][:, :, 0].reshape(-1, 3),
             "historical_wl104d_reference": ref["wi"].reshape(-1, ref["wi"].shape[-2], 3)[:, 0],
             "wl104kb_train_coverage": kb["train_coverage"]["incidence"], "wl104kb_dev": kb["dev"]["incidence"],
             "wl104kb_test": kb["test"]["incidence"], "test_c": lane["incidence"]}
    iso = DK.isolation(groups, incid, n)
    # earlier sets are compared only with test_c (their mutual relations were qualified in WL104K-B)
    iso["exact_equal_outgoing_active"] = {k: v for k, v in iso["exact_equal_outgoing_active"].items() if "test_c" in k}
    iso["exact_equal_incidence"] = {k: v for k, v in iso["exact_equal_incidence"].items() if "test_c" in k}
    iso["duplicate_incidence_within_set"] = {"test_c": iso["duplicate_incidence_within_set"].get("test_c", 0)}
    iso["passed"] = bool(not any(iso["exact_equal_outgoing_active"].values()) and not any(iso["exact_equal_incidence"].values())
                         and not any(iso["duplicate_incidence_within_set"].values()))
    iso["registry"] = {"recorded": [list(r) for r in DK.REGISTRY], "new": [list(r) for r in new],
                       "checks": [info["registry"], pinfo["registry"]],
                       "passed": bool(info["registry"]["passed"] and pinfo["registry"]["passed"]), **reg_info,
                       "wl104kb_data_json_sha256": kb_dj_sha}
    iso["frozen_inputs"] = DK.frozen_check(seen)
    iso["protocol_sha256_unchanged"] = C.sha(C.PROTOCOL) == proto_sha
    iso_ok = bool(iso["passed"] and iso["registry"]["passed"] and iso["frozen_inputs"]["passed"] and iso["protocol_sha256_unchanged"])
    q = {"test_c": bool(all(r["passed"] for r in info["precision"]) and info["support"]["passed"] and iso_ok),
         "test_c_peak": bool(all(r["passed"] for r in pinfo["precision"]) and iso_ok), "isolation": iso_ok}
    q["all"] = bool(all(q.values()))
    rep = {"qualified": q, "sets": {"test_c": info, "test_c_peak": pinfo}, "isolation": iso, "materials": [r["tag"] for r in rows],
           "smoke": args.smoke, "commit": C.commit(), "protocol_sha256": proto_sha, "inputs_sha256": seen,
           "code_sha256": {k: C.sha(C.REPO / k) for k in FROZEN[:2]}, "git_status_frozen_code": dirty,
           "outputs_sha256": {f: C.sha(args.out / f) for f in ("test_c.npz", "test_c_peak.npz")},
           "seconds": time.monotonic() - t0}
    C.write(args.out / "data.json", rep)
    print("data", q, round(rep["seconds"], 1), flush=True)
    if not q["all"] and not args.smoke:
        raise C.Stop(f"DATA_UNQUALIFIED {q}")


if __name__ == "__main__":
    main()
