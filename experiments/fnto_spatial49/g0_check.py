"""WL49 Gate 0 - reference / evaluation readiness checks and the Cycles Yellow exitance ratio from the bakes.

    python g0_check.py <out root> <bake json> [<bake json> ...]
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s49_data as S  # noqa: E402
import r1_common as C  # noqa: E402


@torch.no_grad()
def main(root, bakes):
    root = Path(root); out = root / "gate0_reference"; out.mkdir(parents=True, exist_ok=True); dev = torch.device("cuda")
    L = S.Lego(dev); sp = S.splits(); rf, rh = S.receivers(L); rec = {"checks": {}, "exitance": {}}
    a3 = json.loads((C.ROOT / "results/fnto_diag45/track_a/a3_oracle.json").read_text())
    rec["checks"]["scale_provenance"] = {"s_c_obj": a3["source_sets"]["obj"]["scale_per_channel"], "fit_views": "0-7 (a3_burley_oracle.py: tr = view < 8)", "test_views": "8-15"}
    rec["checks"]["object_scope_materials_in_sources"] = sorted(np.unique(L.mat).tolist())
    rec["checks"]["split_disjoint"] = {"fit_val": int(len(np.intersect1d(sp["fit"], sp["val"]))), "fit_test": "different prep splits (train vs holdout)",
                                       "receivers_fit_held": int(len(np.intersect1d(rf, rh)))}
    xi = torch.as_tensor(rf[:5], device=dev); E1 = torch.ones(L.nA, 1, device=dev)
    yS, yT = S.oracle(L, xi, E1, S.dch()[0])
    r = torch.cdist(L.pos[xi], L.pos).clamp(min=2e-3); k = S.burley(r, S.dch()[0]) * (r < 16 * S.dch()[0]) * L.area[None]
    with_self = k.sum(1); self_term = k[torch.arange(5), xi]
    rec["checks"]["self_exclusion"] = {"rel_diff_vs_bruteforce_without_self": float(((yS + yT)[:, 0] - (with_self - self_term)).abs().max() / with_self.max()),
                                       "self_share_of_response": float((self_term / with_self).mean())}
    rec["checks"]["readout"] = "oracle and candidates: isotropic scalar per receiver / channel; K44 historical: K_only pixel path toward the camera"
    # exitance: incident sun power on Yellow / Lego ports for the baked view
    white = [json.loads(Path(bj).read_text()) for bj in bakes if json.loads(Path(bj).read_text()).get("white")]
    if white:
        w = white[0]; rec["mesh_incident"] = {"lego_mesh": w["all_exit_flux_RGB"][0], "yellow_mesh": w["yellow_exit_flux_RGB"][0], "spp": w["spp"], "seed": w["seed"],
                                              "note": "white Lambertian bake: pi * sum(area * L) = incident sun power on the mesh (Cycles visibility)"}
    for bj in bakes:
        b = json.loads(Path(bj).read_text()); v = b["view"]
        if b.get("white"):
            continue
        E = S.view_fields(L, "T0")[:, v]; A_ = L.area; yel = torch.as_tensor(L.mat == 0, device=dev)
        Pin_lego = float((A_ * E).sum()); Pin_yel = float((A_ * E * yel).sum())
        fl = np.array(b["yellow_exit_flux_RGB"])
        rec["exitance"][Path(bj).stem] = {"spp": b["spp"], "seed": b["seed"], "yellow_exit_flux_RGB": fl.tolist(),
                                          "incident_sun_power_lego_ports": Pin_lego, "incident_sun_power_yellow_ports": Pin_yel,
                                          "injected_into_yellow_sss_estimate": 0.98 * Pin_yel,
                                          "exit_over_incident_lego_RGB": (fl / Pin_lego).tolist(), "exit_over_incident_yellow_RGB": (fl / Pin_yel).tolist(),
                                          "exit_over_injected_yellow_RGB": (fl / (0.98 * Pin_yel)).tolist(),
                                          "area_check_yellow_mesh_vs_ports": [b["yellow_area_m2"], float((A_ * yel).sum())],
                                          "area_check_lego_mesh_vs_ports": [b["total_area_m2"], float(A_.sum())], "t_bake_s": b["t_bake_s"]}
        if white:
            m = rec["mesh_incident"]; rec["exitance"][Path(bj).stem].update({"exit_over_incident_lego_MESH_RGB": (fl / m["lego_mesh"]).tolist(),
                                                                             "exit_over_incident_yellow_MESH_RGB": (fl / m["yellow_mesh"]).tolist(),
                                                                             "exit_over_injected_yellow_MESH_RGB": (fl / (0.98 * m["yellow_mesh"])).tolist()})
    ks = [k for k in rec["exitance"] if rec["exitance"][k]["spp"] > 1]
    if len(ks) >= 2:
        a, b = (np.array(rec["exitance"][k]["yellow_exit_flux_RGB"]) for k in ks[:2])
        rec["exitance_replicate_rel_diff_RGB"] = (np.abs(a - b) / ((a + b) / 2)).tolist()
        rec["exitance_valid"] = bool(np.all(np.abs(a - b) / ((a + b) / 2) < 0.03))
    (out / "gate0.json").write_text(json.dumps(rec, indent=1)); print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
