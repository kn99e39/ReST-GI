"""WL106 Part A: qualify the WL105 silhouette failure on frozen inputs and checkpoints (no training, no rendering).

python qualify106.py DATA105 MODELS105 EVAL105 OUT

For every frozen arm (R1, R2, R3) and every noisy input y = sum_q w_q x_q is decomposed exactly as
    y - T = (sum_q w_q T_q - T)            mixing term: the kernel applied to the expectation (support / mixing)
          + sum_q w_q (x_q - T_q)          weight-noise term: weights correlated with the noise they average
with T the frozen-ECRO target of the same component(s). The coverage moment of every kernel, sum_q w_q c_q - c_p
(c = 9 shape coverages + miss fraction), measures cross-surface mixing directly. Evaluation views use the WL105 target
(mean of two 8192-spp halves); validation views (4096 spp) additionally receive the diagnostic coverage-moment
constraint applied to the FROZEN R3 logits (mechanism prediction; evaluation views are never constrained here).
"""
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "ecro_denoise105"))
import model106 as M6  # noqa: E402
import tilt106 as T  # noqa: E402
import data105 as D  # noqa: E402
from analyze105 import rois, LUMA  # noqa: E402
from train105 import D_aux, sha  # noqa: E402
M = M6.M
BINS = [0.0, 1 / 64, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 63 / 64, 1.0 + 1e-9]
ROI_KEYS = ("primary", "silhouette", "material_boundary", "sss_interior", "control")


def luma(x):
    return np.tensordot(LUMA, x, 1)


def roi_bias(d, Y, m):
    return float(luma(d)[m].mean() / luma(Y)[m].mean()) if m.any() else float("nan")


def bin_bias(d, Y, cover_sph, m):
    out = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        mm = m & (cover_sph >= lo) & (cover_sph < hi)
        out.append(float(luma(d)[mm].sum() / luma(Y)[mm].sum()) if mm.sum() >= 20 else None)
    return out


def moments(logits, c10, constrained=False, cov=None):
    """sum_q w_q c_q for the 10 coverage channels with the (optionally constrained) kernel."""
    c12 = torch.cat([c10, torch.zeros_like(c10[:, :2])], 1)
    parts = []
    for j in (0, 3, 6, 9):
        x = c12[:, j:j + 3].contiguous()
        parts.append(T.constrained_apply_fused(x, logits, cov) if constrained else M.apply_kernel(x, logits))
    return torch.cat(parts, 1)[:, :10]


def kernels(net, arm, comp, aux):
    low, high, total = M.features(comp, aux)
    if arm == "R1":
        return [(total, net.net(torch.cat([torch.log1p(total), aux], 1)), (0, 9))]
    if arm == "R2":
        k = net.net(torch.cat([torch.log1p(low), torch.log1p(high), aux], 1))
        return [(low, k[:, :M.K * M.K], (0, 6)), (high, k[:, M.K * M.K:], (6, 9))]
    return [(low, net.net_low(torch.cat([torch.log1p(low), aux], 1)), (0, 6)),
            (high, net.net_high(torch.cat([torch.log1p(high), aux], 1)), (6, 9))]


def comp_target(Tc, rng):
    a, b = rng
    t = Tc[a:a + 3]
    for s in range(a + 3, b, 3):
        t = t + Tc[s:s + 3]
    return t


def run_view(nets, d, reps, eval_split, constrained_diag, dev):
    base, Tc, halves, nz = D.load_view(d, reps, eval_split=eval_split)
    with np.load(d / "aux.npz") as a:
        auxd = {k: a[k] for k in a.files}
    m = rois(auxd)
    Tc = Tc.astype(np.float64)
    Y = Tc[0:3] + Tc[3:6] + Tc[6:9]
    sph = auxd["cover"][6]
    out = {}
    for spp in D.SPPS:
        aux = torch.from_numpy(D_aux(base, spp))[None].to(dev)
        cov = aux[:, :9]
        c10 = T.coverage10(cov)
        c10n = c10[0].cpu().numpy()
        for r in range(reps):
            comp = torch.from_numpy(nz[spp][r])[None].to(dev)
            for arm, net in nets.items():
                ops = ["U"] + (["C"] if constrained_diag and arm == "R3" else [])
                for op in ops:
                    y = np.zeros_like(Y)
                    mix = np.zeros_like(Y)
                    mom = {}
                    for img, lg, rng in kernels(net, arm, comp, aux):
                        tgt = torch.from_numpy(comp_target(Tc, rng).astype(np.float32))[None].to(dev)
                        if op == "U":
                            y += M.apply_kernel(img, lg)[0].double().cpu().numpy()
                            mix += M.apply_kernel(tgt, lg)[0].double().cpu().numpy()
                        else:
                            y += T.constrained_apply_fused(img, lg, cov)[0].double().cpu().numpy()
                            mix += T.constrained_apply_fused(tgt, lg, cov)[0].double().cpu().numpy()
                        mm = (moments(lg, c10, op == "C", cov)[0].cpu().numpy() - c10n)
                        mom["LOW" if rng == (0, 6) else "HIGH" if rng == (6, 9) else "TOTAL"] = mm
                    key = f"{arm}{'+C' if op == 'C' else ''}"
                    rec = out.setdefault(key, {}).setdefault(str(spp), {"bias": {}, "mix": {}, "bins_bias": [], "bins_mix": [],
                                                                         "moment_L1_silhouette": {}, "moment_sphere_bins": {}, "y_rmse_sil": []})
                    for roi in ROI_KEYS:
                        rec["bias"].setdefault(roi, []).append(roi_bias(y - Y, Y, m[roi]))
                        rec["mix"].setdefault(roi, []).append(roi_bias(mix - Y, Y, m[roi]))
                    rec["bins_bias"].append(bin_bias(y - Y, Y, sph, m["silhouette"]))
                    rec["bins_mix"].append(bin_bias(mix - Y, Y, sph, m["silhouette"]))
                    rec["y_rmse_sil"].append(math.sqrt(((y - Y)[:, m["silhouette"]] ** 2).mean() / (Y[:, m["silhouette"]] ** 2).mean()))
                    for kname, mm in mom.items():
                        l1 = np.abs(mm).sum(0)
                        rec["moment_L1_silhouette"].setdefault(kname, []).append(float(l1[m["silhouette"]].mean()))
                        rec["moment_sphere_bins"].setdefault(kname, []).append(
                            [float(mm[6][m["silhouette"] & (sph >= lo) & (sph < hi)].mean()) if (m["silhouette"] & (sph >= lo) & (sph < hi)).sum() >= 20 else None
                             for lo, hi in zip(BINS[:-1], BINS[1:])])
                    if eval_split and op == "U" and r == 0:
                        rec.setdefault("y_for_check", []).append(y)
    return out, Y, halves


def main(data, models, ev, out):
    dev = "cuda"
    torch.backends.cudnn.deterministic = True
    M.FUSED = True
    out.mkdir(parents=True, exist_ok=False)
    eman = json.loads((ev / "eval_manifest.json").read_text())
    dman = json.loads((data / "run_manifest.json").read_text())
    rec = {"checks": {}}
    rec["checks"]["weights_sha256_match_WL105"] = {a: sha(models / f"{a}.pt") == eman["weights_sha256"][a] for a in ("R1", "R2", "R3")}
    rec["checks"]["data_manifest_sha256_match"] = sha(data / "run_manifest.json") == eman["data_manifest_sha256"]
    bad = []
    for key, v in dman["views"].items():
        if key.split("/")[0] not in ("eval", "val"):
            continue
        for f, h in v["files"].items():
            if f.startswith("noisy_") or f.startswith("target") or f == "aux.npz":
                if sha(data / key / f) != h:
                    bad.append(f"{key}/{f}")
    rec["checks"]["input_hashes_identical"] = {"mismatches": bad, "pass": not bad}
    assert all(rec["checks"]["weights_sha256_match_WL105"].values()) and rec["checks"]["data_manifest_sha256_match"] and not bad
    nets = {}
    for a in ("R1", "R2", "R3"):
        n = M.Reconstructor(a).to(dev)
        n.load_state_dict(torch.load(models / f"{a}.pt", map_location=dev))
        n.eval()
        nets[a] = n
    res = {"eval": {}, "val": {}}
    maxdiff = 0.0
    with torch.no_grad():
        for n in eman["views"]:
            o, Y, halves = run_view(nets, data / "eval" / n, eman["reps"], True, False, dev)
            for arm in ("R1", "R2", "R3"):       # reproduction: identical outputs to the stored WL105 evaluation
                for i, spp in enumerate(D.SPPS):
                    with np.load(ev / f"{n}_spp{spp}_r0.npz") as f:
                        st = f[arm].astype(np.float64)
                    y = o[arm][str(spp)].pop("y_for_check")[0]
                    maxdiff = max(maxdiff, float(np.abs(y - st).max() / np.abs(st).max()))
            res["eval"][n] = o
            print("eval", n, flush=True)
        val_names = sorted(k.split("/")[1] for k in dman["views"] if k.startswith("val/"))
        for n in val_names:
            o, Y, _ = run_view(nets, data / "val" / n, 4, False, True, dev)
            res["val"][n] = o
            print("val", n, flush=True)
    rec["checks"]["reproduces_WL105_eval_outputs_max_rel"] = maxdiff
    rec["results"] = res
    (out / "qualify106.json").write_text(json.dumps(rec))
    print("QUALIFY COMPLETE", json.dumps(rec["checks"]), flush=True)


if __name__ == "__main__":
    main(*[Path(a).resolve() for a in sys.argv[1:5]])
