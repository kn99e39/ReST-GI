"""WL104K-A Stage A: reproduce and qualify the existing WL104K evidence (angular reproduction, per-start scene metrics,
uncertainty decomposition, event ownership).

Usage: python stageA104ka.py <out_json>
"""
import sys
from pathlib import Path

import numpy as np

import common104ka as KA

CM, EV, IO = KA.CM, KA.EV, KA.IO


def main():
    out = Path(sys.argv[1])
    out.parent.mkdir(parents=True, exist_ok=True)
    seen = KA.authenticate_k()
    # post hoc scene outputs: record hashes before any new evidence (they were not in a frozen manifest)
    ph = KA.K_ROOT / "posthoc_scenes"
    ph_hash = {str(p.relative_to(KA.K_ROOT)).replace("\\", "/"): CM.sha(p) for p in sorted(ph.rglob("*")) if p.is_file()}
    rows, raw, _, _, ref, seen2 = CM.inputs()
    seen.update(seen2)
    n = len(rows)
    ev, _ = CM.eval_reference(ref, n)
    E = KA.read(KA.K_ROOT / "eval/eval.json")
    st, R, fj = KA.states(KA.K_ROOT / "fit_primary", n)
    # ---- angular reproduction
    rep = {}
    for arm in CM.ARMS:
        el = np.stack([CM.wl82_terms(CM.predict(arm, s, ev["wi"], ev["wo"]), ev["mean"], ev["w"])["el1"] for s in st[arm]])
        arch = np.array(E["tables"][arm]["el1_per_start"])
        rep[arm] = float(np.max(np.abs(el - arch) / np.abs(arch)))
    ok_rep = max(rep.values()) <= 1e-9
    # ---- uncertainty decomposition (angular)
    coh = E["cohorts"]
    unc = {}
    for c, ms in coh.items():
        unc[c] = {"reference_half_split_el1_ratio_of_A": E["reference_resolution"][c]["ratio"]}
        for k in ("T-A", "P-T", "P-A"):
            ps = E["contrasts"][k]["eval:el1"][c]["per_start"]
            unc[c][f"{k}_across_start_range_pp"] = float(100 * (max(ps) - min(ps)))
            unc[c][f"{k}_material_bootstrap_ci_width_pp"] = float(100 * (E["contrasts"][k]["eval:el1"][c]["hi"] - E["contrasts"][k]["eval:el1"][c]["lo"]))
    # ---- scenes: per-start stable RMS and contrasts; replicate jackknife half-widths
    scenes = {}
    for s in KA.SCENES:
        an = KA.read(ph / s / "analysis.json")
        fx = an["fixed"]
        per = {}
        for arm in CM.ARMS:
            per[arm] = {ev_: [fx[f"{arm}_{r}"]["objects"]["rms"][ev_]["point"] for r in range(R)] for ev_ in ("e1", "e2", "total")}
        jk = {arm: {ev_: [float(np.std(fx[f"{arm}_{r}"]["objects"]["rms"][ev_]["loo"]) * np.sqrt(11)) for r in range(R)]
                    for ev_ in ("e1", "e2", "total")} for arm in CM.ARMS}
        scenes[s] = {"identity": an["identity"], "identity_passed": an["identity_passed"], "per_start_stable_rms": per,
                     "per_start_jackknife_se": jk, "contrasts": an["primary"],
                     "A0_stable_rms": {ev_: fx["A0"]["objects"]["rms"][ev_]["point"] for ev_ in ("e1", "e2", "total")}}
    ok_scene = all(v["identity_passed"] for v in scenes.values())
    res = {"angular_reproduction_max_relative": rep, "angular_reproduction_passed": bool(ok_rep), "uncertainty": unc,
           "scenes": scenes, "scene_identity_passed": bool(ok_scene), "posthoc_scene_sha256": ph_hash,
           "event_ownership": KA.P["stage_A"]["ownership"], "inputs_sha256": seen, "commit": CM.commit()}
    KA.write(out, res)
    print("stageA", rep, ok_rep, ok_scene, flush=True)
    if not (ok_rep and ok_scene):
        raise IO.Stop("STAGE_A_REPRODUCTION")


if __name__ == "__main__":
    main()
