"""WL45 Track A2 - where the frozen K_k44P K-only error on Yellow lives (all valid pixels of the 16 held-out T0 views,
regime first_r1; no learned parameter is touched).

Squared display error (C.ldr3) of K_only vs first_r1 summed over Lego pixels, split by
  material (Yellow share of the Lego K-only SE), Yellow lit / shadow (pixel sun visibility), Yellow distance to the
  nearest sun-lit Lego port of that view (E_sun > 0 with the eval_vis_k44 fractions), channel;
with the signed luminance bias K / ref per bin.

    python a2_localize.py <out dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "fnto_lego_r1")); sys.path.insert(0, str(HERE.parents[1] / "fnto_lego"))
import r1_common as C  # noqa: E402

LUM = np.array([0.2126, 0.7152, 0.0722])
BINS = [0, 0.003, 0.01, 0.03, 0.10, 0.30, 10]


def main(out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    A = np.load(C.OUT1 / "anchors.npz"); nA = int(A["n_A"]); pos, nrm = A["pos"][:nA].astype(np.float64), A["nrm"][:nA].astype(np.float64)
    px = np.load(C.OUT1 / "eval_px/heldout_T0.npz"); first = np.load(C.OUT1 / "refs/heldout_T0_first.npz")
    V = np.load(C.OUT1 / "eval_K_k44P/views.npz"); vis = np.load(C.OUT1 / "eval_vis_k44.npz"); D = C.ldr3
    acc = {k: [] for k in ("se", "mat", "lit", "dist", "k", "r", "view")}
    for v in range(int(px["nv"])):
        val = px[f"v{v}_valid"].astype(bool); dom = px[f"v{v}_dom"][val]; mat = px[f"v{v}_mat"][val]
        lego = dom == 0
        K = V[f"heldout_T0_v{v}_K_only"][:, :3].astype(np.float64)[lego]; R = first[f"v{v}_rgb"][val].astype(np.float64)[lego]
        w = np.asarray(px[f"v{v}_light"], np.float64)
        lit_ports = pos[(nrm @ w > 0) & (vis[f"heldout_T0_v{v}_T0"][:nA] > 0)]
        dist = cKDTree(lit_ports).query(px[f"v{v}_pos"][val][lego].astype(np.float64))[0]
        acc["se"].append((D(K) - D(R)) ** 2); acc["mat"].append(mat[lego]); acc["lit"].append(px[f"v{v}_vis"][val][lego] > 0.5)
        acc["dist"].append(dist); acc["k"].append(K @ LUM); acc["r"].append(R @ LUM); acc["view"].append(np.full(lego.sum(), v))
        print("view", v, int(lego.sum()), flush=True)
    d = {k: np.concatenate(x) for k, x in acc.items()}
    se = d["se"].sum(1); y = d["mat"] == 0; tot = se.sum()

    def row(m):
        return {"px": int(m.sum()), "px_share": float(m.sum() / y.sum()), "se_share_of_yellow": float(se[m].sum() / se[y].sum()),
                "psnr": float(10 * np.log10(1 / max(d["se"][m].mean(), 1e-30))), "K_over_ref_lum": float(d["k"][m].sum() / max(d["r"][m].sum(), 1e-30)),
                "se_share_RGB": (d["se"][m].sum(0) / max(d["se"][m].sum(), 1e-30)).tolist()}
    rec = {"regime": "first_r1, held-out T0, all 16 views, display space (ldr3)", "yellow_se_share_of_lego": float(se[y].sum() / tot),
           "per_material_se_share": {C.MATERIALS[i]: float(se[d["mat"] == i].sum() / tot) for i in range(8)},
           "yellow_all": row(y), "yellow_lit": row(y & d["lit"]), "yellow_shadow": row(y & ~d["lit"]), "yellow_by_distance_to_lit_port": []}
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        for lab, m in (("shadow", y & ~d["lit"]), ("lit", y & d["lit"])):
            mm = m & (d["dist"] >= lo) & (d["dist"] < hi)
            if mm.sum() > 100:
                rec["yellow_by_distance_to_lit_port"].append({"m": [lo, hi], "pixels": lab, **row(mm)})
    rec["yellow_by_view_se_share"] = [float(se[y & (d["view"] == v)].sum() / se[y].sum()) for v in range(int(px["nv"]))]
    (out / "a2_localize.json").write_text(json.dumps(rec, indent=1)); print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
