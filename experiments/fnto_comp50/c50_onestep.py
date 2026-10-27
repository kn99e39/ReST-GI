"""WL50 Gate 2 - single-flight K / C coupling (worklog-46 one-step semantics) and first-event sender energy.

Held-out T0 views (all 16): K_only (L = 0) and ONE production gather step (rel_T0; each gather reads the condition's
first-event sender value), production render. Per class (Yellow / opaque / glass / box): the single-flight indirect
(one-step - K_only) as a luminance ratio to the reference indirect (full_r1 - first_r1, ALL bounces - an upper reference)
and its correlation; display PSNR of the one-step image vs full_r1. Sender energy (first event): sum over Yellow ports of
pi * A * (K44 rho / pi E_sun + glob + [band + mid | N_arm] at the port normal) / sum over Lego ports A * E_sun (port areas;
Gate 0 of worklog 49: port areas are 1.5x the mesh, the bake ratio is mesh-based).

    python c50_onestep.py <out dir> <L0 ckpt> <L1 ckpt>
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import c50_common as CC  # noqa: E402
import r1_common as C  # noqa: E402

LUM = np.array([0.2126, 0.7152, 0.0722]); CLS = {"Yellow": [0], "opaque": [1, 3, 5, 6, 7], "glass": [2, 4], "box": [8]}


def psnr(p, r):
    return float(10 * np.log10(1 / max(((C.ldr3(p) - C.ldr3(r)) ** 2).mean(), 1e-30)))


@torch.no_grad()
def main(out, ck0, ck1):
    from r1_eval import load_kernels
    out = Path(out); out.mkdir(parents=True, exist_ok=True); dev = torch.device("cuda"); t00 = time.time()
    KA, KB = load_kernels(str(C.OUT1 / "K_k44P/kernels.pt"), dev); T = CC.SV.Tables(KA, KB, dev); geo = CC.Geo(dev); yp = CC.yellow_ports()
    arms = {"H": None, "L0C": CC.ArmOp("L0", CC.load_net(ck0, dev), geo, yp), "L1C": CC.ArmOp("L1", CC.load_net(ck1, dev), geo, yp)}
    H = np.load(C.OUT1 / "rel_T0.npz")["hit"]; vis = np.load(C.OUT1 / "eval_vis_k44.npz"); px = np.load(C.OUT1 / "eval_px/heldout_T0.npz")
    full = np.load(C.OUT1 / "refs/heldout_T0_full.npz"); first = np.load(C.OUT1 / "refs/heldout_T0_first.npz")
    nA = int(T.dom[0].numel()); rec = {"per_view": {}, "pooled": {}, "energy": {}}; acc = {}
    for v in range(int(px["nv"])):
        val = px[f"v{v}_valid"].astype(bool); mat = np.where(px[f"v{v}_dom"][val] == 1, 8, px[f"v{v}_mat"][val])
        pxd = {k: torch.as_tensor(px[f"v{v}_{k}"][val].astype(np.float32 if k in ("pos", "nrm", "cam", "vis") else np.int64), device=dev) for k in ("pos", "nrm", "cam", "vis", "dom", "port")}
        rf, r0 = full[f"v{v}_rgb"][val].astype(np.float64), first[f"v{v}_rgb"][val].astype(np.float64); rind = (rf - r0) @ LUM
        Vf = vis[f"heldout_T0_v{v}_T0"].astype(np.float32); r = {}
        for cn, arm in arms.items():
            CC.activate(arm)
            S = CC.SV.State(T, H, Vf, px[f"v{v}_light"]); Eind, h = S.aggregates(); I0 = S.outgoing(Eind, h); p0 = CC.SV.render_pixels(T, S, pxd)[0].cpu().numpy()
            if v in (6, 15):                                                                     # first-event sender energy (port areas)
                Ar = T.area[:nA]; ypt = torch.as_tensor(yp, device=dev)
                rec["energy"].setdefault(cn, {})[f"view{v}"] = {"yellow_iso_out_over_lego_sun_RGB": (math.pi * (I0[ypt] * Ar[ypt, None]).sum(0) / (Ar * S.Esun[:nA]).sum()).tolist()}
            S.step(); Eind, h = S.aggregates(); S.outgoing(Eind, h); p1 = CC.SV.render_pixels(T, S, pxd)[0].cpu().numpy()
            pind = (p1 - p0) @ LUM; rr = {"psnr_onestep_vs_full": psnr(p1, rf), "finite": bool(np.isfinite(p1).all())}
            for k, ids in CLS.items():
                m = np.isin(mat, ids)
                if m.sum() > 100:
                    rr[k] = {"indirect_ratio_to_ref": float(pind[m].sum() / max(rind[m].sum(), 1e-30)), "indirect_corr": float(np.corrcoef(pind[m], rind[m])[0, 1]),
                             "psnr_onestep": psnr(p1[m], rf[m]), "psnr_K_only_vs_first": psnr(p0[m], r0[m])}
                    a = acc.setdefault((cn, k), [0.0, 0.0, [], []]); a[0] += float(pind[m].sum()); a[1] += float(rind[m].sum()); a[2].append(pind[m]); a[3].append(rind[m])
            r[cn] = rr; del S; torch.cuda.empty_cache()
        rec["per_view"][f"view{v}"] = r; print(v, json.dumps({cn: {k: (round(x["indirect_ratio_to_ref"], 3) if isinstance(x, dict) else round(x, 2)) for k, x in rr_.items() if k != "finite"} for cn, rr_ in r.items()}), flush=True)
    for (cn, k), a in acc.items():
        pi_, ri_ = np.concatenate(a[2]), np.concatenate(a[3])
        rec["pooled"].setdefault(cn, {})[k] = {"indirect_ratio_to_ref": a[0] / max(a[1], 1e-30), "indirect_corr": float(np.corrcoef(pi_, ri_)[0, 1])}
    CC.deactivate(); rec["cost"] = {"wall_s": time.time() - t00, "peak_GB": torch.cuda.max_memory_allocated() / 1e9}
    (out / "onestep.json").write_text(json.dumps(rec, indent=1)); print(json.dumps(rec["pooled"], indent=1), json.dumps(rec["energy"]))


if __name__ == "__main__":
    main(*sys.argv[1:4])
