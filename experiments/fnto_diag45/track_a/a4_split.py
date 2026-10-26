"""WL45 Track A4 (part 2) - where Yellow's first-event light comes from, by source distance, for the physics oracle vs
the frozen K_k44P component split; and A2's remaining localisation axes (view sun elevation, receiver n.w).

Oracle (A3, object-scoped source set, s_c and d_c from a3_oracle.json): per test-view (8-15) Yellow pixel, the share of
the oracle's output from source ports at |x - x'| < 5 cm (K_band reach), 5-30 cm (K_mid reach), > 30 cm (only K_glob,
rank 64). K_k44P: the share of its K-only output in loc / ind / glob / band+mid (eval views.npz columns 3-14).
Both for lit and shadow Yellow pixels, luminance, summed over pixels.
A2 axes: K-only vs first_r1 Yellow PSNR and K / ref by view (with sun elevation) and by receiver n.w bins (lit pixels).

    python a4_split.py <a3_oracle.json> <out dir> [px_per_view=6000]
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "fnto_lego_r1")); sys.path.insert(0, str(HERE.parents[1] / "fnto_lego"))
import r1_common as C  # noqa: E402
from a3_burley_oracle import burley  # noqa: E402

LUM = np.array([0.2126, 0.7152, 0.0722])
EDGES = [0.0, 0.05, 0.30, 1e9]


@torch.no_grad()
def main(a3_json, out, npx=6000):
    out = Path(out); out.mkdir(parents=True, exist_ok=True); dev = torch.device("cuda")
    a3 = json.loads(Path(a3_json).read_text()); dch = a3["d_m"]; sc = np.array(a3["source_sets"]["obj"]["scale_per_channel"])
    A = np.load(C.OUT1 / "anchors.npz"); nA = int(A["n_A"])
    pos = torch.as_tensor(A["pos"][:nA], device=dev).float(); nrm = torch.as_tensor(A["nrm"][:nA], device=dev).float(); area = torch.as_tensor(A["area"][:nA], device=dev).float()
    px = np.load(C.OUT1 / "eval_px/heldout_T0.npz"); first = np.load(C.OUT1 / "refs/heldout_T0_first.npz")
    V = np.load(C.OUT1 / "eval_K_k44P/views.npz"); vis = np.load(C.OUT1 / "eval_vis_k44.npz"); D = C.ldr3
    rng = np.random.default_rng(4504); acc = {k: [] for k in ("orc", "kparts", "lit")}; per_view = []
    for v in range(int(px["nv"])):
        val = px[f"v{v}_valid"].astype(bool); dom = px[f"v{v}_dom"][val]; matp = px[f"v{v}_mat"][val]; y = (dom == 0) & (matp == 0)
        K = V[f"heldout_T0_v{v}_K_only"].astype(np.float64); R = first[f"v{v}_rgb"][val].astype(np.float64); w = np.asarray(px[f"v{v}_light"], np.float64)
        lit = px[f"v{v}_vis"][val] > 0.5; ndw = px[f"v{v}_nrm"][val].astype(np.float64) @ w
        rec_v = {"view": v, "sun_elevation_deg": float(np.degrees(np.arcsin(np.clip(w[2], -1, 1)))),
                 "yellow_psnr": float(10 * np.log10(1 / ((D(K[y, :3]) - D(R[y])) ** 2).mean())), "K_over_ref_lum": float((K[y, :3] @ LUM).sum() / (R[y] @ LUM).sum()),
                 "lit_by_ndotw": []}
        for lo, hi in ((0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)):
            m = y & lit & (ndw >= lo) & (ndw < hi)
            if m.sum() > 200:
                rec_v["lit_by_ndotw"].append({"ndotw": [lo, hi], "px": int(m.sum()), "psnr": float(10 * np.log10(1 / ((D(K[m, :3]) - D(R[m])) ** 2).mean())),
                                              "K_over_ref_lum": float((K[m, :3] @ LUM).sum() / (R[m] @ LUM).sum())})
        per_view.append(rec_v)
        if v < 8:
            continue
        yi = np.flatnonzero(y); sel = np.sort(rng.choice(yi, min(npx, len(yi)), replace=False))
        frac = torch.as_tensor(vis[f"heldout_T0_v{v}_T0"].astype(np.float32)[:nA], device=dev)
        EA = C.E_SUN * (nrm @ torch.as_tensor(w, device=dev).float()).clamp(min=0) * frac * area
        xr = torch.as_tensor(px[f"v{v}_pos"][val][sel], device=dev).float(); o = torch.zeros(len(sel), 3, 3, device=dev)
        for c0 in range(0, len(sel), 96):
            r = torch.cdist(xr[c0:c0 + 96], pos).clamp(min=2e-3)
            for ci, d in enumerate(dch):
                k = burley(r, d) * (r < 16 * d) * EA[None]
                for bi in range(3):
                    o[c0:c0 + 96, bi, ci] = (k * ((r >= EDGES[bi]) & (r < EDGES[bi + 1]))).sum(1)
        acc["orc"].append(o.cpu().numpy() * sc[None, None]); acc["kparts"].append(K[sel, 3:15].reshape(-1, 4, 3)); acc["lit"].append(lit[sel])
        print("view", v, flush=True)
    d = {k: np.concatenate(x) for k, x in acc.items()}; rec = {"per_view": per_view, "source_distance_split": {}}
    for lab, m in (("lit", d["lit"]), ("shadow", ~d["lit"])):
        ob = (d["orc"][m] @ LUM).sum(0); kp = (d["kparts"][m] @ LUM).sum(0)
        ob_rgb = d["orc"][m].sum(0)
        rec["source_distance_split"][lab] = {"px": int(m.sum()), "oracle_share_lt5cm_5to30cm_gt30cm_lum": (ob / ob.sum()).tolist(),
                                             "oracle_share_by_channel_RGB": {c: (ob_rgb[:, i] / max(ob_rgb[:, i].sum(), 1e-30)).tolist() for i, c in enumerate("RGB")},
                                             "K_share_loc_ind_glob_bandmid_lum": (kp / kp.sum()).tolist()}
    (out / "a4_split.json").write_text(json.dumps(rec, indent=1)); print(json.dumps(rec["source_distance_split"], indent=1))
    print(json.dumps([{k: x[k] for k in ("view", "sun_elevation_deg", "yellow_psnr", "K_over_ref_lum")} for x in per_view]))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 6000)
