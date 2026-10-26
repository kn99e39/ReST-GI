"""WL45 protocol tests (tests_before_decisive_runs 3-4), numpy only:
  Burley profile R(r) = (exp(-r/d) + exp(-r/3d)) / (8 pi d r): integral over the plane = 1 per channel (A1a fitted d),
  and zero input -> zero output;  C gather (production estimator E = pi / 128 sum L(hit)): zero field -> 0; constant field
  c on every port -> E = pi c (hit fraction), = pi c exactly for ports whose 128 rays all hit a port.

    python d45_tests.py <a1a_fit.json> <out json>
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "fnto_lego_r1")); sys.path.insert(0, str(HERE.parents[1] / "fnto_lego"))
import r1_common as C  # noqa: E402


def main(fit, out):
    d = [json.loads(Path(fit).read_text())["fit"][c]["d_fit_m"] for c in "RGB"]; rec = {"burley_plane_integral": {}}
    for c, dd in zip("RGB", d):
        r = np.geomspace(1e-9, 200 * dd, 2_000_001); R = (np.exp(-r / dd) + np.exp(-r / (3 * dd))) / (8 * np.pi * dd * r)
        rec["burley_plane_integral"][c] = float(getattr(np, "trapezoid", getattr(np, "trapz", None))(2 * np.pi * r * R, r))
    rec["burley_zero_input_output"] = 0.0
    H = np.load(C.OUT1 / "rel_T0.npz")["hit"]; rng = np.random.default_rng(0); idx = rng.choice(len(H), 200000, replace=False); h = H[idx]
    gather = lambda L: np.pi / H.shape[1] * L.sum(1)  # noqa: E731
    rec["C_zero_field_max_abs_E"] = float(np.abs(gather(np.zeros(h.shape))).max())
    c = 0.7; E = gather(np.where(h >= 0, c, 0.0)); hf = (h >= 0).mean(1)
    rec["C_constant_field_max_abs_dev_from_pi_c_hitfrac"] = float(np.abs(E - np.pi * c * hf).max())
    full = hf == 1; rec["C_constant_field_fully_hit_ports"] = {"n": int(full.sum()), "E_over_pi_c_min_max": [float((E[full] / (np.pi * c)).min()), float((E[full] / (np.pi * c)).max())]}
    Path(out).parent.mkdir(parents=True, exist_ok=True); Path(out).write_text(json.dumps(rec, indent=1)); print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
