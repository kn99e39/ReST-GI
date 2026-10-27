"""WL47 post-hoc metric repair (no new data, no refit choices): two predeclared ratio metrics are undefined where their
true denominator is 0.

  (1) Stage A nonlocal pair difference used ||Y_A,nl|| as denominator; pairs whose A field sends nothing nonlocally divide
      by 0. Repaired: max(||Y_A,nl||, ||Y_B,nl||), pairs with both 0 excluded (count reported).
  (2) Stage C pair discrimination ||dY_hat|| / ||dY|| on S0, where dY = 0 exactly. Repaired: spurious pair difference
      ||dY_hat|| / ||Y_pairs|| for every condition and regime (refit with the identical deterministic Stage C code).

    python diag47.py <results/fnto_smoke47> [<mir T npz>]
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import smoke47_core as K  # noqa: E402


def main():
    root = Path(sys.argv[1]); rec = {"stage_a_nonlocal_pair_diff": {}, "stage_c_spurious_pair_diff": {}}
    for name in ("S0", "S1"):
        d = np.load(root / "stage_a" / f"stage_a_{name}.npz"); T = d["T"]
        nl = np.ones_like(T, bool)
        for q in range(K.P):
            nl[q * K.J:(q + 1) * K.J, q * K.J:(q + 1) * K.J] = False
        a, b = (T * nl) @ d["LA"].T, (T * nl) @ d["LB"].T
        den = np.maximum(np.linalg.norm(a, axis=0), np.linalg.norm(b, axis=0)); ok = den > 1e-12
        D = np.linalg.norm(a - b, axis=0)[ok] / den[ok]
        rec["stage_a_nonlocal_pair_diff"][name] = {"mean": float(D.mean()) if ok.any() else None, "median": float(np.median(D)) if ok.any() else None,
                                                   "n_pairs": int(ok.sum()), "n_excluded_both_zero": int((~ok).sum())}
    regimes = {"S0": (K.oracle_s0(), K.OMP, K.P, K.J, K.P, np.tile(np.isin(np.arange(K.J), [0, 1, 4, 5]), K.P), K.lap_synthetic()),
               "S1": (K.oracle_s1(), K.OMP, K.P, K.J, K.P, np.tile(np.isin(np.arange(K.J), [0, 1, 4, 5]), K.P), K.lap_synthetic())}
    if len(sys.argv) > 2:
        nd = len(K.MIR_T)
        regimes["MIR"] = (np.load(sys.argv[2])["T1"], np.ones(nd), 2, nd, 4, np.tile(np.isin(K.MIR_T, K.PM["visible_dirs"]), 4), K.lap_chain(4, nd))
    for name, (T, cvec, P_in, J_in, n_out, mask, lap) in regimes.items():
        Emap = K.emap(P_in, cvec); rt = np.random.default_rng(K.PC["data"]["test_seed"])
        K.sample_fields(K.PC["data"]["test_random"], P_in, J_in, cvec, rt, nonzero=True)                 # same stream as Stage C
        LA = K.sample_fields(4 * K.PC["data"]["test_pairs"], P_in, J_in, cvec, rt, nonzero=True)
        LA = LA[(Emap @ LA.T).sum(0) > 0][:K.PC["data"]["test_pairs"]]; LB = K.pair_fields(LA, P_in, J_in, cvec, rt)
        ny = np.sqrt((np.linalg.norm(T @ LA.T) ** 2 + np.linalg.norm(T @ LB.T) ** 2) / 2)
        out = {"truth": float(np.linalg.norm(T @ (LA - LB).T) / ny)}
        Ltr = K.sample_fields(K.PC["data"]["train"], P_in, J_in, cvec, np.random.default_rng(K.PC["data"]["train_seeds"][0]), nonzero=True).T
        for cond in ("A0", "A1", "B0", "B1"):
            m = mask if cond.endswith("0") else np.ones_like(mask)
            Z = Emap @ Ltr if cond[0] == "A" else Ltr; Wm = K.fit(Z, T @ Ltr, m, lap)
            Wrad = Wm @ Emap if cond[0] == "A" else Wm
            out[cond] = float(np.linalg.norm(Wrad @ (LA - LB).T) / ny)
        rec["stage_c_spurious_pair_diff"][name] = out
    (root / "diag_posthoc.json").write_text(json.dumps(rec, indent=1)); print(json.dumps(rec, indent=1))


main()
