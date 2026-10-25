"""WL45 Track B2 / B3a - frozen C (and frozen K_k44P) against the emit45 Blender oracle.

B2 (C only, no K): for sampled non-Yellow pixels of held-out T0 views, the frozen relation table rel_T0 gathers the
known outgoing field f (analytic, emit45_render.field_numpy) as C does in production:
  E_C(port)      = pi / 128 * sum_k f(origin of the hit port)        (production: value carried by the hit PORT)
  E_C_hit(port)  = pi / 128 * sum_k f(hit point o_k + t_k d_k)       (output projection removed; same rays)
  misses (-1), inside faces (-2) and Yellow hits carry 0 (Yellow does not emit in emit45)
and the oracle E_in(pixel) = pi (P_emit - P_emitonly). Reported: pooled ratio, median |relative error|, correlation,
per material, and by the receiver port's near-hit / inside / miss fractions.
B3a (one K/C step): State(T, rel_T0) with the sun off and L set to the production gathered field; aggregates ->
outgoing (sun off) -> render_pixels with pixel sun visibility 0 gives K_k44P's response to that incident field at
Yellow pixels, compared with the emit45 Yellow radiance (the real Yellow material under the same field).

    python b2_b3_emit_eval.py <emit45 dir> <out dir> [views=6,15,2,10] [px_per_view=20000]
    env D45_B2_ONLY=1 (B2 only), D45_SUFFIX (output name suffix), D45_EMIT_ALL=1 (B3a against the 'emitall' oracle)
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "fnto_lego_r1")); sys.path.insert(0, str(HERE.parents[1] / "fnto_lego")); sys.path.insert(0, str(HERE.parents[1] / "fnto_lego_k44"))
import r1_common as C  # noqa: E402
import r1_relations as RR  # noqa: E402

MATS = C.MATERIALS[:8] + ["box"]
B2_ONLY = bool(int(__import__("os").environ.get("D45_B2_ONLY", "0")))            # re-run of B2 (no GPU) after the hit-point fix
SUFFIX = __import__("os").environ.get("D45_SUFFIX", "")
EMIT_ALL = bool(int(__import__("os").environ.get("D45_EMIT_ALL", "0")))          # 'emitall' oracle: Yellow ports also carry f; B3a only
EMIT_NAME = "emitall" if EMIT_ALL else "emit"


def field_numpy(p):
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    return np.maximum(0.05, 1 + 0.6 * np.sin(2 * np.pi * x / 0.4) * np.sin(2 * np.pi * y / 0.4) + 0.3 * np.sin(2 * np.pi * z / 0.25))


def port_world(A):
    off = np.zeros((len(A["obj"]), 3)); off[A["obj"] == 1] = np.asarray(C.offsets("T0")[1], np.float64)
    return A["pos"].astype(np.float64) + off


def main(edir, out, views=(6, 15, 2, 10), npx=20000):
    edir, out = Path(edir), Path(out); out.mkdir(parents=True, exist_ok=True)
    A = RR.load(); N = len(A["obj"]); pw = port_world(A)
    emits = (A["mat"] != 0) | (A["obj"] == 1)                                  # Yellow (mat 0, Lego) does not emit
    if EMIT_ALL:
        emits = np.ones_like(emits)
    f_port = np.where(emits, field_numpy(pw), 0.0)
    rel = np.load(C.OUT1 / "rel_T0.npz"); H, Tt = rel["hit"], rel["t"]
    px = np.load(C.OUT1 / "eval_px/heldout_T0.npz"); rng = np.random.default_rng(4502)
    acc = {"E_in": [], "E_C": [], "E_C_hit": [], "mat": [], "near": [], "inside": [], "miss": [], "view": []}
    for v in (views if not EMIT_ALL else ()):                                   # emitall: B2 does not apply (Yellow pixels only)
        e = np.load(edir / f"emit_v{v}.npz")["rgb"].astype(np.float64); eo = np.load(edir / f"emitonly_v{v}.npz")["rgb"].astype(np.float64)
        val = px[f"v{v}_valid"].astype(bool); dom = px[f"v{v}_dom"]; mat = np.where(dom == 1, 8, px[f"v{v}_mat"])
        cand = np.flatnonzero(val & (mat != 0)); sel = np.sort(rng.choice(cand, min(npx, len(cand)), replace=False))
        E_in = np.pi * (e[sel] - eo[sel]).mean(1)
        prt = px[f"v{v}_port"][sel].astype(np.int64); up, inv = np.unique(prt, return_inverse=True)
        o, d = RR.ray_setup(A, up)                                             # [u, 128, 3] world (offsets: box ports need T0 offset)
        boxu = A["obj"][up] == 1; o[boxu] += np.asarray(C.offsets("T0")[1], np.float64)
        h = H[up]; t = Tt[up].astype(np.float64)
        fp = np.where(h >= 0, f_port[np.clip(h, 0, None)], 0.0)
        hp = o + np.where(h >= 0, t, 0.0)[..., None] * d; fh = np.where((h >= 0) & emits[np.clip(h, 0, None)], field_numpy(hp), 0.0)   # misses: t = inf
        Ec = math.pi / H.shape[1] * fp.sum(1); Ech = math.pi / H.shape[1] * fh.sum(1)
        acc["E_in"].append(E_in); acc["E_C"].append(Ec[inv]); acc["E_C_hit"].append(Ech[inv]); acc["mat"].append(mat[sel]); acc["view"].append(np.full(len(sel), v))
        acc["near"].append(((t < 3e-4) & (h >= 0)).mean(1)[inv]); acc["inside"].append((h == -2).mean(1)[inv]); acc["miss"].append((h == -1).mean(1)[inv])
        print("B2 view", v, len(sel), flush=True)
    d = {k: np.concatenate(x) if x else np.zeros(0) for k, x in acc.items()}

    def stats(a, b, m):
        a, b = a[m], b[m]
        return {"px": int(m.sum()), "pooled_ratio": float(a.sum() / max(b.sum(), 1e-30)), "median_abs_rel": float(np.median(np.abs(a - b) / np.maximum(b, 1e-6))),
                "corr": float(np.corrcoef(a, b)[0, 1]) if m.sum() > 2 else None, "mean_signed_rel": float(np.mean((a - b) / np.maximum(b, 1e-6)))}
    rec = {"views": list(views), "B2": {}, "oracle": EMIT_NAME}
    for nm, key in ((("C_production", "E_C"), ("C_hitpoint_values", "E_C_hit")) if not EMIT_ALL else ()):
        r = {"all": stats(d[key], d["E_in"], np.ones(len(d["E_in"]), bool))}
        for i in range(1, 9):
            m = d["mat"] == i
            if m.sum() > 100:
                r[MATS[i]] = stats(d[key], d["E_in"], m)
        for lab, m in (("receiver_near_hit_frac>0.05", d["near"] > 0.05), ("near_hit_frac==0", d["near"] == 0), ("inside_frac>0.10", d["inside"] > 0.10),
                       ("inside_frac==0", d["inside"] == 0), ("miss_frac>0.5", d["miss"] > 0.5)):
            if m.sum() > 100:
                r[lab] = stats(d[key], d["E_in"], m)
        rec["B2"][nm] = r
    if not EMIT_ALL:
        rec["B2"]["receiver_fractions_mean"] = {"near_hit_lt_0.3mm": float(d["near"].mean()), "inside": float(d["inside"].mean()), "miss": float(d["miss"].mean())}
    if B2_ONLY:
        (out / f"b2{SUFFIX}.json").write_text(json.dumps(rec, indent=1)); print(json.dumps(rec, indent=1))
        np.savez_compressed(out / f"b2_arrays{SUFFIX}.npz", **{f"b2_{k}": v for k, v in d.items()}); return
    # B3a: one K/C step with the production field, Yellow pixels
    import r1_solve as SV
    from r1_eval import load_kernels
    dev = torch.device("cuda"); KA, KB = load_kernels(str(C.OUT1 / "K_k44P/kernels.pt"), dev); T = SV.Tables(KA, KB, dev)
    S = SV.State(T, H, np.zeros(N), np.array([0.0, 0.0, 1.0]))
    with torch.no_grad():
        Lf = torch.as_tensor(np.where(H >= 0, f_port[np.clip(H, 0, None)], 0.0).astype(np.float16), device=dev)
        S.L = Lf[..., None].expand(-1, -1, 3).contiguous()
        Eind, hh = S.aggregates(); S.Eind, S.h = Eind, hh; S.outgoing(Eind, hh, sun_on=False)
        b3 = {"ref": [], "pred": [], "parts": [], "view": []}
        for v in views:
            e = np.load(edir / f"{EMIT_NAME}_v{v}.npz")["rgb"].astype(np.float64)
            val = px[f"v{v}_valid"].astype(bool); dom = px[f"v{v}_dom"]; mat = px[f"v{v}_mat"]
            yel = np.flatnonzero(val & (dom == 0) & (mat == 0))
            pxd = {k: torch.as_tensor(px[f"v{v}_{k}"][yel].astype(np.float32 if k in ("pos", "nrm", "cam", "vis") else np.int64), device=dev)
                   for k in ("pos", "nrm", "cam", "vis", "dom", "port")}
            pxd["vis"] = torch.zeros_like(pxd["vis"])
            pr, parts = SV.render_pixels(T, S, pxd)
            b3["ref"].append(e[yel]); b3["pred"].append(pr.cpu().numpy()); b3["parts"].append(parts.cpu().numpy()); b3["view"].append(np.full(len(yel), v))
    dd = {k: np.concatenate(x) for k, x in b3.items()}; Dd = C.ldr3; LUM = np.array([0.2126, 0.7152, 0.0722])
    rec["B3a_yellow_one_step"] = {"px": int(len(dd["ref"])), "display_psnr": float(10 * np.log10(1 / ((Dd(dd["pred"]) - Dd(dd["ref"])) ** 2).mean())),
                                 "pred_over_ref_lum": float((dd["pred"] @ LUM).sum() / (dd["ref"] @ LUM).sum()),
                                 "pred_over_ref_rgb": (dd["pred"].sum(0) / dd["ref"].sum(0)).tolist(),
                                 "corr_lum": float(np.corrcoef(dd["pred"] @ LUM, dd["ref"] @ LUM)[0, 1]),
                                 "parts_mean_lum_loc_ind_glob_bandmid": [float((dd["parts"][:, k] @ LUM).mean()) for k in range(4)],
                                 "ref_mean_lum": float((dd["ref"] @ LUM).mean())}
    (out / f"b2_b3a{SUFFIX}.json").write_text(json.dumps(rec, indent=1)); print(json.dumps(rec, indent=1))
    np.savez_compressed(out / f"b2_b3a_arrays{SUFFIX}.npz", **{f"b2_{k}": v for k, v in d.items()}, **{f"b3_{k}": v for k, v in dd.items()})


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], tuple(int(x) for x in a[2].split(",")) if len(a) > 2 else (6, 15, 2, 10), int(a[3]) if len(a) > 3 else 20000)
