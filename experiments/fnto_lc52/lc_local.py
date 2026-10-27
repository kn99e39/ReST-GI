"""WL52 amendment 3 (POST HOC diagnostic, after the primary evaluation): well-conditioned localized error of the coupled
solutions. The predeclared P2 metric (per-port RELATIVE error, 99th percentile) is dominated by ports whose oracle
irradiance is ~0 (e.g. floor under a box 0.5 mm above it) and is reported unchanged; this diagnostic adds
  e_a(r) = |E_a(r) - E_ORA(r)| / mean_r |E_ORA|      (absolute, normalized by the configuration's mean indirect irradiance)
p99 / max over all / near-contact (< 1 cm) / strongly occluded / visibility-transition ports (trajectories), the share of the
total squared error held by the worst 1 % of ports (sparse vs spread), and the learned-only part |E_C1 - E_C0p| (C0p = the
same patch representation, ray traced). Coupled solves as lc_eval (Lambertian K, sun of the configuration).

    python lc_local.py <ckpt> <out dir>
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lc_eval as LE  # noqa: E402
import lc_model as LM  # noqa: E402
import lc_scene as LS  # noqa: E402
import lc_trace as TR  # noqa: E402

DEV = LE.DEV


def stats(e, m):
    if m.sum() < 20:
        return None
    x = e[m]; s2 = (x ** 2).sort(descending=True).values
    return {"p99": float(torch.quantile(x, 0.99)), "max": float(x.max()), "mean": float(x.mean()), "top1pct_sq_share": float(s2[:max(1, len(x) // 100)].sum() / s2.sum().clamp(min=1e-30)), "n": int(m.sum())}


def main(ckpt, outd):
    outd = Path(outd); outd.mkdir(parents=True, exist_ok=True); model = LM.GLT().to(DEV); model.load_state_dict(torch.load(ckpt, map_location=DEV)); model.eval()
    sets = {"test_iid": LE.SETS["test_iid"][:LE.COUPLED_IID], "traj": [f for j, nm in enumerate(LS.TRAJ) for f in LS.trajectory(nm, 400000 + j)]}; rec = {}; t0 = time.time(); maps = {}
    for sn, cfgs in sets.items():
        R = []; prevH = None; prevT = None
        for i, cfg in enumerate(cfgs):
            S = LS.build(cfg); G = TR.Geo(S, DEV); Es = TR.sun_irr(G, cfg["sun"]); ap, Ho, To, _ = LE.arms(model, G, cfg); mk = LE.masks(G, Ho, To)
            if sn == "traj":
                mk["transition"] = ((Ho != prevH).float().mean(1) > 0.1) if (prevH is not None and prevT == cfg["traj"]) else torch.zeros(S.N, dtype=torch.bool, device=DEV)
                prevH, prevT = Ho, cfg["traj"]
            sol = {a: TR.solve(G, Es, ap[a])["Eind"] for a in ("ORA", "C0", "C0p", "C1")}; nrm = sol["ORA"].norm(dim=1).mean().clamp(min=1e-12)
            r = {"seed": cfg["seed"], "traj": cfg.get("traj"), "frame": cfg.get("frame")}
            for a in ("C0", "C0p", "C1"):
                e = (sol[a] - sol["ORA"]).norm(dim=1) / nrm; r[a] = {m: stats(e, mm) for m, mm in mk.items() if m in ("all", "near_contact_lt1cm", "strongly_occluded", "transition")}
            e = (sol["C1"] - sol["C0p"]).norm(dim=1) / nrm; r["C1_minus_C0p"] = {m: stats(e, mm) for m, mm in mk.items() if m in ("all", "near_contact_lt1cm", "strongly_occluded", "transition")}
            if (sn == "traj" and cfg["frame"] in (0, 29)) or (sn == "test_iid" and i < 3):
                pid = TR.camera_ports(G, *TR.CAMS[0], res=256); bg = pid < 0
                maps[f"{sn}_{cfg['seed']}_{cfg.get('traj', '')}_{cfg.get('frame', '')}"] = {a: torch.where(bg, torch.zeros_like(pid, dtype=torch.float32), ((sol[a] - sol["ORA"]).norm(dim=1) / nrm)[pid.clamp(min=0)]).cpu().numpy().astype(np.float16) for a in ("C0", "C0p", "C1")}
            R.append(r); print(sn, i, json.dumps({a: (r[a]["all"]["p99"], (r[a].get("near_contact_lt1cm") or {}).get("p99")) for a in ("C0", "C0p", "C1")}), flush=True)
            del G; torch.cuda.empty_cache()
        rec[sn] = R; (outd / "local.json").write_text(json.dumps(rec, indent=1))
    np.savez_compressed(outd / "errmaps.npz", **{f"{k}__{a}": v for k, d in maps.items() for a, v in d.items()})
    rec["wall_s"] = time.time() - t0; (outd / "local.json").write_text(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
