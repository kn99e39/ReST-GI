"""WL104J POST HOC (after the frozen decision; does not change any label): why did the S1 amplitude-variation control fail?

Hypothesis tested: the frozen chain hands B the shared parameters (lobe shapes, attenuation, eta) of A's fit to the S1 target, and A
compensates the incident-dependent amplitudes with those shared parameters; B with the TRUE shared parameters frozen would recover
the amplitude variation. Same noisy S1 fit target, optimizer, steps and evaluation as the frozen S1 run.
Variant B_truth_frozen: B with the truth's shapes / attenuation / eta frozen, amplitudes started from the unmodulated truth (R0).

Usage: python posthoc104j_s1.py <run_root> <out_json>
"""
import sys
from pathlib import Path

import numpy as np

import common104j as CM
import fit104j as FT

IO = CM.IO


def main():
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    rows, raw, _, desc, ref, _, seen = CM.inputs()
    n = len(rows)
    with np.load(root / "data/fit_set.npz") as z:
        d = {k: z[k] for k in z.files}
    y_star = CM.GF.from_original(raw["R0"])
    truth = FT.synthetic_truth("S1", y_star)
    tgt = CM.predict(truth["arm"], truth["state"], d["wi"].reshape(n, CM.NP, -1, 3), d["wo"].reshape(n, CM.NP, -1, 3), CM.PROBE_GROUP)
    sigma = 0.5 * np.abs(d["a"] - d["b"]).reshape(tgt.shape)
    noisy = tgt + sigma * np.random.default_rng(CM.P["seeds"]["synthetic_noise"]).standard_normal(tgt.shape)
    data = FT.FitData(d, n, 1, target=noisy.reshape(-1, d["wi"].shape[1], 3))
    init = FT.local_from_A(y_star)                     # truth shapes / s_n / eta; unmodulated amplitudes
    log = {}
    term, _ = FT.run_arm(data, "B", init, 12000, log)
    R4 = {k: ref[k].reshape(n, CM.NP, -1, *ref[k].shape[2:]) for k in ("wi", "wo", "use", "q")}
    w = R4["use"] / R4["q"]
    et = CM.predict(truth["arm"], truth["state"], R4["wi"], R4["wo"], CM.PROBE_GROUP)
    e = CM.wl82_terms(CM.predict("B", term, R4["wi"], R4["wo"], CM.PROBE_GROUP), et, w)["el1"]
    e0 = CM.wl82_terms(CM.predict("B", init, R4["wi"], R4["wo"], CM.PROBE_GROUP), et, w)["el1"]
    frozen = IO.read(root / "eval/eval.json")["synthetic"]["S1"]["heldout_el1_mean"]
    res = {"post_hoc": True, "variant": "B_truth_frozen", "heldout_el1_mean": float(e.mean()),
           "heldout_el1_mean_at_start_unmodulated": float(e0.mean()), "frozen_S1_heldout_el1_mean": frozen,
           "ratio_to_frozen_A": float(e.mean() / frozen["A"]), "fit": log["B"], "commit": CM.commit(), "inputs_sha256": seen}
    CM.write(out, res)
    print("posthoc S1", res["heldout_el1_mean"], res["ratio_to_frozen_A"], flush=True)


if __name__ == "__main__":
    main()
