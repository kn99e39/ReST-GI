"""WL45 Track B3f (amendment 4) - spatial spread of K_k44P's Yellow response to a LOCALIZED incident field.

Field: f = 10 exp(-|x - c|^2 / (2 (2 cm)^2)) emitted (non-camera rays) by every Lego / box surface including Yellow
('emitall' semantics, D45_FIELD=bump); Blender renders of views 6 / 15 at 1024 spp with two seeds (noise estimate:
half difference). K_k44P one K/C step with the C-gathered field (b2_b3_emit_eval.py, D45_EMIT_ALL=1 D45_FIELD=bump).
Per Yellow pixel distance to c (bins 0-2, 2-5, 5-10, 10-20, 20-40 cm): pred / ref luminance and per channel,
reference seed-to-seed relative difference, and K's component shares (loc / ind / glob / band+mid).

    python b3f_bump.py <bump render dir (emitall_v*.npz, emitall_seed100_v*.npz)> <b2_b3a_arrays_bump.npz> <out dir>
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "fnto_lego_r1")); sys.path.insert(0, str(HERE.parents[1] / "fnto_lego"))
import r1_common as C  # noqa: E402

LUM = np.array([0.2126, 0.7152, 0.0722]); CEN = np.array([0.26708984, -0.45532227, 0.54345703]); BINS = [0, 0.02, 0.05, 0.10, 0.20, 0.40]


def main(rdir, arr, out):
    rdir, out = Path(rdir), Path(out); out.mkdir(parents=True, exist_ok=True)
    Z = np.load(arr); px = np.load(C.OUT1 / "eval_px/heldout_T0.npz"); rec = {"center_m": CEN.tolist(), "sigma_m": 0.02, "bins": []}
    acc = {k: [] for k in ("d", "r1", "r2", "p", "parts")}
    for v in sorted(set(Z["b3_view"].tolist())):
        val = px[f"v{v}_valid"].astype(bool); yel = np.flatnonzero(val & (px[f"v{v}_dom"] == 0) & (px[f"v{v}_mat"] == 0)); m = Z["b3_view"] == v
        mk = px[f"v{v}_mk"]; a1 = np.load(rdir / f"emitall_v{v}.npz")["rgb"].astype(np.float64); a2 = np.load(rdir / f"emitall_seed100_v{v}.npz")["rgb"].astype(np.float64)
        acc["d"].append(np.linalg.norm(px[f"v{v}_pos"][yel].astype(np.float64) - CEN, axis=1)); acc["r1"].append(a1[yel]); acc["r2"].append(a2[yel])
        acc["p"].append(Z["b3_pred"][m].astype(np.float64)); acc["parts"].append(Z["b3_parts"][m].astype(np.float64))
    d = {k: np.concatenate(x) for k, x in acc.items()}; ref = 0.5 * (d["r1"] + d["r2"])
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (d["d"] >= lo) & (d["d"] < hi)
        if m.sum() < 50:
            continue
        rl, pl = ref[m] @ LUM, d["p"][m] @ LUM; pt = (d["parts"][m] @ LUM).sum(0)
        rec["bins"].append({"m": [lo, hi], "px": int(m.sum()), "ref_mean_lum": float(rl.mean()), "pred_over_ref_lum": float(pl.sum() / max(rl.sum(), 1e-30)),
                            "pred_over_ref_RGB": (d["p"][m].sum(0) / np.maximum(ref[m].sum(0), 1e-30)).tolist(),
                            "ref_seed_rel_diff_lum": float(np.abs(((d["r1"][m] - d["r2"][m]) @ LUM).sum()) / max(rl.sum(), 1e-30)),
                            "corr_lum": float(np.corrcoef(pl, rl)[0, 1]) if m.sum() > 2 else None, "K_part_shares_loc_ind_glob_bandmid": (pt / max(pt.sum(), 1e-30)).tolist()})
    (out / "b3f_bump.json").write_text(json.dumps(rec, indent=1)); print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:4])
