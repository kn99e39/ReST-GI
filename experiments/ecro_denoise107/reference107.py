"""Independent reference-block precision and immutable WL105 pilot verification."""
import argparse
from pathlib import Path

import numpy as np

from common107 import LUMA, SPPS, VIEWS, coverage_masks, read, rois, sha, total, write

DECISION_ROIS = ("primary", "silhouette", "material_boundary")


def reference_se(var, target, mask, n):
    return float(np.sqrt(var[:, mask].mean() / n / (target[:, mask] ** 2).mean()))


def ref_blocks(data_view, capture_view):
    for h in range(2):
        with np.load(data_view / f"target_half{h}.npz") as file:
            yield file["comp"]
    if capture_view is not None:
        for h in range(46):
            yield np.load(capture_view / f"assessment_{h:02d}.npy", allow_pickle=False)


def assemble(data_view, capture_view, masks, signed_masks):
    mean_comp = None
    mean_rgb = None
    sq_diff_rgb = None
    signed = {k: [] for k in signed_masks}
    for n, block in enumerate(ref_blocks(data_view, capture_view), 1):
        comp = block.astype(np.float64)
        rgb = total(comp)
        if mean_rgb is None:
            mean_comp = comp.copy()
            mean_rgb = rgb.copy()
            sq_diff_rgb = np.zeros_like(mean_rgb)
        else:
            mean_comp += (comp - mean_comp) / n
            diff = rgb - mean_rgb
            mean_rgb += diff / n
            sq_diff_rgb += diff * (rgb - mean_rgb)
        for name, mask in signed_masks.items():
            signed[name].append(float(LUMA @ rgb[:, mask].mean(axis=1)))
    sq_diff_rgb /= n - 1
    result = {"blocks": n, "relative_rgb_rms_se": {k: reference_se(sq_diff_rgb, mean_rgb, m, n)
                                                   for k, m in masks.items() if m.any()},
              "relative_signed_luma_se": {k: float(np.std(rows, ddof=1) / np.sqrt(n) / np.mean(rows))
                                          for k, rows in signed.items()}}
    return mean_comp.astype(np.float32), result


def rel_rmse(pred, target, mask):
    return float(np.sqrt(((pred - target)[:, mask] ** 2).mean() / (target[:, mask] ** 2).mean()))


def qualify(baseline, output, capture, anchor, replay):
    archive_only = capture is None
    if not archive_only:
        assert replay is not None and read(replay)["repeat_bitwise"]
        cap = read(capture / "capture_manifest.json")
        assert cap["complete"]
        assert all(sha(capture / f) == r["sha256"] for f, r in cap["files"].items() if "/assessment_" in f)
    anchors = read(anchor)
    raw = anchors["raw_files"]
    records = {}
    output.mkdir(parents=True, exist_ok=False)
    for view in VIEWS:
        d = baseline / "data/eval" / view
        for f in ("aux.npz", "target_half0.npz", "target_half1.npz"):
            assert sha(d / f) == raw[f"data/eval/{view}/{f}"]
        with np.load(d / "aux.npz") as z:
            aux = {k: z[k] for k in z.files}
        masks = rois(aux)
        bins = coverage_masks(aux["cover"][6], masks["silhouette"])
        signed = {r: masks[r] for r in DECISION_ROIS if masks[r].any()}
        signed.update({f"coverage_{i}": m for i, m in enumerate(bins) if m.sum() >= 20})
        target, rec = assemble(d, None if archive_only else capture / view, masks, signed)
        if not archive_only:
            np.save(output / f"{view}_target.npy", target, allow_pickle=False)
            rec["target_sha256"] = sha(output / f"{view}_target.npy")
            cur = rec["R3_RMSE"] = {}
            for spp in SPPS:
                stack = []
                for i in range(8):
                    with np.load(baseline / "eval" / f"{view}_spp{spp}_r{i}.npz") as f:
                        stack.append(f["R3"])
                cur[str(spp)] = {r: float(np.mean([rel_rmse(y, total(target), masks[r]) for y in stack]))
                                 for r in DECISION_ROIS if masks[r].any()}
        records[view] = rec
    rms_se = {r: float(np.mean([v["relative_rgb_rms_se"][r] for v in records.values()
                                if r in v["relative_rgb_rms_se"]])) for r in DECISION_ROIS}
    if archive_only:
        published_path = anchor.parent / "metrics.json"
        assert sha(published_path) == anchors["review_files"]["metrics.json"]
        published = read(published_path)
        rmse = {str(spp): {r: published["summary"]["R3"][str(spp)][r]["rel_rmse"]
                           for r in DECISION_ROIS} for spp in SPPS}
    else:
        rmse = {str(spp): {r: float(np.mean([v["R3_RMSE"][str(spp)][r] for v in records.values()
                                            if r in v["R3_RMSE"][str(spp)]]))
                           for r in DECISION_ROIS} for spp in SPPS}
    ratios = {spp: {r: rms_se[r] / e for r, e in row.items()} for spp, row in rmse.items()}
    max_signed_se = max(value for rec in records.values() for value in rec["relative_signed_luma_se"].values())
    passed = max(value for row in ratios.values() for value in row.values()) <= 0.10 and max_signed_se <= 0.0025
    report = {"archive_pilot": archive_only, "views": records, "pooled_reference_rgb_rms_se": rms_se,
              "pooled_archived_or_recomputed_R3_rmse": rmse, "reference_noise_over_R3_rmse": ratios,
              "max_view_stratum_signed_relative_se": max_signed_se, "preliminary_pass": bool(passed),
              "rgb_se_limit_fraction_of_RMSE": 0.10, "signed_stratum_se_limit": 0.0025,
              "archive_anchor_sha256": sha(anchor)}
    write(output / "reference107.json", report)
    print("Reference precision", "PASS" if passed else "FAIL", ratios, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for field in ("baseline", "output", "anchor"):
        p.add_argument(field, type=Path)
    p.add_argument("--capture", type=Path)
    p.add_argument("--replay", type=Path)
    a = p.parse_args()
    qualify(a.baseline, a.output, a.capture, a.anchor, a.replay)
