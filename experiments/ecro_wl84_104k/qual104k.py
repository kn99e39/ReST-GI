"""WL104K POST HOC qualitative package (owner request 2026-11-22): rendered scene comparison of reference / A / T / P plus the
angular material response, built from the post hoc fixed-proposal scene replay. Presentation only; the frozen verdict is unchanged.

Usage: python qual104k.py <run_root> <out_dir>
"""
import argparse
import html
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import FuncNorm  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common104k as CM  # noqa: E402
sys.path.insert(0, str(HERE.parent / "ecro_wl84_104h"))
import scenes104h as S8  # noqa: E402

LUMA = CM.LUMA
SCENES = {"test83_iid_1_L2": "iid_1 (development material)", "test83_hard_1_L1": "hard_1 (development material)",
          "test83_iid_159_L2": "iid_159 (held-out material, held-out camera)"}
ARMS = {"reference": "Layered reference (physical)", "A": "A  shared ER (baseline)", "T": "T  one shared gate", "P": "P  per-component gates (ER-v1-SDM)"}
SLUG = {"reference": "layered_reference", "A": "A_shared_ER", "T": "T_shared_gate", "P": "P_per_component_gates"}
EVENTS = {"first_bounce": "first bounce (direct light on the objects)", "first_and_second_bounce": "first + second bounce",
          "total": "total (first + second + higher-order)"}
TAG = "POST HOC · owner-requested qualitative material · frozen WL104K verdict unchanged"


def srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def ssqrt(M):
    return FuncNorm((lambda x: np.sign(x) * (np.abs(x) / M) ** 0.25, lambda y: np.sign(y) * y ** 4 * M), vmin=-M, vmax=M)


def root4(M):
    return FuncNorm((lambda x: (np.clip(x, 0, None) / M) ** 0.25, lambda y: y ** 4 * M), vmin=0, vmax=M)


def load_scene(root, scene, start=0):
    d = root / "posthoc_scenes" / scene
    fx = {}
    for r in range(12):
        with np.load(d / f"fixed_r{r:03d}.npz") as z:
            for q in z.files:
                fx.setdefault(q, []).append(z[q])
    fx = {q: np.stack(v).astype(np.float64).mean(0) for q, v in fx.items()}          # mean over the 12 replicates
    arch, masks, _ = S8.archived(scene, None)
    ref = {e: arch[f"reference_{e}"].mean(0) for e in ("e1", "e2", "h")}
    r0h = arch["R0_h"].mean(0)
    img = {"reference": {"first_bounce": ref["e1"], "first_and_second_bounce": ref["e1"] + ref["e2"], "total": ref["e1"] + ref["e2"] + ref["h"]}}
    for a in ("A", "T", "P"):
        e1, e2 = fx["N|e1"] + fx[f"{a}_{start}|e1"], fx["N|e2"] + fx[f"{a}_{start}|e2"]
        img[a] = {"first_bounce": e1, "first_and_second_bounce": e1 + e2, "total": e1 + e2 + r0h}
    an = CM.read(d / "analysis.json")
    return img, masks, an


def bbox(mask, pad=6):
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    y0, y1, x0, x1 = ys.min() - pad, ys.max() + pad, xs.min() - pad, xs.max() + pad
    H, W = mask.shape
    return max(0, y0), min(H, y1 + 1), max(0, x0), min(W, x1 + 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    out = args.out
    (out / "renders").mkdir(parents=True, exist_ok=True)
    data = {s: load_scene(args.root, s) for s in SCENES}
    # one absolute error scale for every error map of this package (luminance)
    M = 0.0
    for s, (img, masks, an) in data.items():
        for ev in EVENTS:
            r = img["reference"][ev] @ LUMA
            for a in ("A", "T", "P"):
                M = max(M, float(np.abs(img[a][ev] @ LUMA - r)[masks["objects"]].max()))
    files, figs, metrics = {}, [], {}
    for s, (img, masks, an) in data.items():
        sl = s.replace("test83_", "")
        expo = {ev: float(np.percentile(img["reference"][ev][masks["objects"]] @ LUMA, 99.5)) for ev in EVENTS}
        # ---- individual tone-mapped renders (same exposure for all arms of a scene and event)
        for ev in EVENTS:
            for a in ARMS:
                f = f"renders/{sl}_{ev}_{SLUG[a]}.png"
                plt.imsave(out / f, srgb(img[a][ev] / expo[ev]))
                files[(s, ev, a)] = f
            for a in ("A", "T", "P"):
                err = np.abs(img[a][ev] @ LUMA - img["reference"][ev] @ LUMA)
                f = f"renders/{sl}_{ev}_{SLUG[a]}_abs_error.png"
                fig = plt.figure(figsize=(4, 4), dpi=100); ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
                ax.imshow(np.where(masks["objects"], err, np.nan), cmap="magma", norm=root4(M))
                fig.savefig(out / f); plt.close(fig)
                files[(s, ev, a, "err")] = f
        # ---- overview figure per scene: renders + abs error + reduction
        fig, axs = plt.subplots(3, 7, figsize=(24, 11.2))
        for i, ev in enumerate(EVENTS):
            r = img["reference"][ev]
            for j, a in enumerate(ARMS):
                axs[i, j].imshow(srgb(img[a][ev] / expo[ev])); axs[i, j].set_xticks([]); axs[i, j].set_yticks([])
                if i == 0:
                    axs[i, j].set_title(ARMS[a], fontsize=10)
            ea = np.abs(img["A"][ev] @ LUMA - r @ LUMA); ep = np.abs(img["P"][ev] @ LUMA - r @ LUMA)
            et = np.abs(img["T"][ev] @ LUMA - r @ LUMA)
            nm = root4(M)
            im1 = axs[i, 4].imshow(np.where(masks["objects"], ea, np.nan), cmap="magma", norm=nm)
            axs[i, 5].imshow(np.where(masks["objects"], ep, np.nan), cmap="magma", norm=nm)
            im2 = axs[i, 6].imshow(np.where(masks["objects"], ea - ep, np.nan), cmap="PiYG", norm=ssqrt(M))
            for k in (4, 5, 6):
                axs[i, k].set_xticks([]); axs[i, k].set_yticks([])
            if i == 0:
                axs[i, 4].set_title("|A − reference| (objects)", fontsize=10)
                axs[i, 5].set_title("|P − reference| (objects)", fontsize=10)
                axs[i, 6].set_title("error reduction |A err| − |P err|\n(green = P closer)", fontsize=10)
            axs[i, 0].set_ylabel(EVENTS[ev], fontsize=10)
            mm = masks["objects"]
            metrics.setdefault(s, {})[ev] = {a: float(np.sqrt(np.mean(((img[a][ev] - r) @ LUMA)[mm] ** 2))) / float(np.mean((r @ LUMA)[mm])) for a in ("A", "T", "P")}
        fig.colorbar(im1, ax=axs[:, 4:6], shrink=0.5, label=f"absolute luminance error (shared scale 0–{M:.3g}, 4th-root display)")
        fig.colorbar(im2, ax=axs[:, 6], shrink=0.5, label="reduction of absolute error (luminance)")
        fig.suptitle(f"{SCENES[s]}: rendered comparison, layered reference vs A / T / P (start 1 = generic, mean of 12 replicates; same exposure per row)",
                     x=0.01, ha="left", fontsize=12, weight="bold")
        fig.text(0.995, 0.995, TAG, ha="right", va="top", fontsize=8, color="white", bbox=dict(boxstyle="round", fc="#b7791f", ec="none"))
        fig.text(0.01, 0.005, "All arms are evaluated at the same sampled directions (common qualified R0 proposal; F/p recomputed per arm). "
                              "The higher-order part of every model's 'total' row is the archived R0 HIGH, so differences in that row come from the first "
                              "and second bounce only. Floor and background use the same native BSDF in every column.", fontsize=8)
        f = f"{sl}_scene_reference_vs_A_T_P_renders_and_error.png"
        fig.savefig(out / f, dpi=100); plt.close(fig)
        figs.append((f, f"{SCENES[s]}: full-frame renders and error maps"))
        # ---- zoom crops: highlight core and grazing region (first bounce)
        crops = [(k, lab) for k, lab in (("highlight_core", "highlight core"), ("grazing", "grazing view region"), ("sphere_lower", "lower sphere"))
                 if k in masks and masks[k].any()]
        fig, axs = plt.subplots(len(crops), 6, figsize=(20, 3.6 * len(crops)), squeeze=False)
        ev = "first_bounce"
        for i, (k, lab) in enumerate(crops):
            y0, y1, x0, x1 = bbox(masks[k])
            r = img["reference"][ev]
            loc_exp = float(np.percentile((r @ LUMA)[masks[k]], 99))
            for j, a in enumerate(ARMS):
                axs[i, j].imshow(srgb(img[a][ev][y0:y1, x0:x1] / loc_exp), interpolation="nearest")
                axs[i, j].set_title(f"{ARMS[a]}" if i == 0 else "", fontsize=9)
            mk = masks[k][y0:y1, x0:x1]
            d = np.where(mk, ((img["P"][ev] - r) @ LUMA)[y0:y1, x0:x1], np.nan)
            da = np.where(mk, ((img["A"][ev] - r) @ LUMA)[y0:y1, x0:x1], np.nan)
            lim = float(np.nanmax(np.abs(np.stack([d, da]))))
            axs[i, 4].imshow(da, cmap="RdBu_r", norm=ssqrt(lim), interpolation="nearest")
            im = axs[i, 5].imshow(d, cmap="RdBu_r", norm=ssqrt(lim), interpolation="nearest")
            if i == 0:
                axs[i, 4].set_title("A − reference (signed)", fontsize=9); axs[i, 5].set_title("P − reference (signed)", fontsize=9)
            axs[i, 0].set_ylabel(f"{lab}\n(local exposure;\nmaps: region pixels only)", fontsize=9)
            fig.colorbar(im, ax=axs[i, 4:6], shrink=0.8, label=f"±{lim:.3g} lum.\n(row scale, 4th root)")
            for a_ in axs[i]:
                a_.set_xticks([]); a_.set_yticks([])
        fig.suptitle(f"{SCENES[s]}: zoomed regions, first bounce (red = model brighter than the reference)", x=0.01, ha="left", fontsize=12, weight="bold")
        fig.text(0.995, 0.995, TAG, ha="right", va="top", fontsize=8, color="white", bbox=dict(boxstyle="round", fc="#b7791f", ec="none"))
        f = f"{sl}_scene_zoomed_highlight_and_grazing_regions.png"
        fig.savefig(out / f, dpi=100, bbox_inches="tight"); plt.close(fig)
        figs.append((f, f"{SCENES[s]}: zoomed highlight / grazing regions"))
    # ---- summary chart: objects relative RMS per scene and event (start 1) + frozen-pass contrasts (start-mean, jackknife)
    fig, axs = plt.subplots(1, 2, figsize=(16, 5))
    xs = np.arange(len(SCENES) * 3)
    short_ev = {"first_bounce": "1st", "first_and_second_bounce": "1st+2nd", "total": "total"}
    lab = [f"{s.replace('test83_', '').rsplit('_', 1)[0]}\n{short_ev[e]}" for s in SCENES for e in EVENTS]
    for j, a in enumerate(("A", "T", "P")):
        axs[0].bar(xs + (j - 1) * 0.27, [100 * metrics[s][e][a] for s in SCENES for e in EVENTS], 0.26, color={"A": "#1f77b4", "T": "#e08214", "P": "#2ca02c"}[a], label=ARMS[a])
    axs[0].set_xticks(xs, lab, fontsize=7.5); axs[0].set_ylabel("objects RMS error / mean reference (%)"); axs[0].legend(fontsize=8)
    axs[0].set_title("objects error of the shown renders (start 1, replicate mean)", fontsize=10)
    rows = []
    for s in SCENES:
        an = data[s][2]
        for ev, evn in (("e1", "first bounce"), ("e2", "second bounce"), ("total", "total")):
            v = an["primary"]["P-A"][ev]
            rows.append((f"{s.replace('test83_', '')}\n{evn}", v["point"], v["lo"], v["hi"]))
    y = np.arange(len(rows))
    axs[1].errorbar([100 * r[1] for r in rows], y, xerr=[[100 * (r[1] - r[2]) for r in rows], [100 * (r[3] - r[1]) for r in rows]], fmt="o", color="#2ca02c", capsize=3)
    axs[1].axvline(0, color="k", lw=0.8); axs[1].set_yticks(y, [r[0] for r in rows], fontsize=8); axs[1].invert_yaxis()
    axs[1].set_xlabel("P − A change of objects stable RMS vs reference (%), mean of 4 starts, jackknife 95 % CI")
    axs[1].set_title("descriptive scene contrast P − A (post hoc, not a frozen test); stable RMS is noise-debiased,\n−100 % = P error below the replicate-noise estimate", fontsize=9)
    fig.text(0.995, 0.995, TAG, ha="right", va="top", fontsize=8, color="white", bbox=dict(boxstyle="round", fc="#b7791f", ec="none"))
    fig.suptitle("Scene-level error: does the angular improvement show up in rendered images?", x=0.01, ha="left", fontsize=12, weight="bold")
    fig.subplots_adjust(left=0.06, right=0.99, top=0.85, bottom=0.2, wspace=0.35)
    f = "scene_error_summary_A_T_P.png"
    fig.savefig(out / f, dpi=110); plt.close(fig)
    figs.insert(0, (f, "Scene-level error summary"))
    # ---- interactive page
    sw = []
    for s in SCENES:
        sl = s.replace("test83_", "")
        for ev in EVENTS:
            opts = "".join(f"<button onclick=\"show('{sl}_{ev}','{files[(s, ev, a)]}',this)\">{html.escape(ARMS[a])}</button>" for a in ARMS)
            eopts = "".join(f"<button onclick=\"show('{sl}_{ev}','{files[(s, ev, a, 'err')]}',this)\">|{a} − reference|</button>" for a in ("A", "T", "P"))
            sw.append(f"<div class='sw'><h4>{html.escape(SCENES[s])} — {html.escape(EVENTS[ev])}</h4><div>{opts}</div><div class='e'>{eopts}</div>"
                      f"<img id='{sl}_{ev}' src='{files[(s, ev, 'reference')]}'></div>")
    an_rows = "".join(f"<tr><td>{html.escape(SCENES[s])}</td>" + "".join(
        f"<td>{100 * data[s][2]['primary']['P-A'][e]['point']:+.2f}% [{100 * data[s][2]['primary']['P-A'][e]['lo']:+.2f}, {100 * data[s][2]['primary']['P-A'][e]['hi']:+.2f}]</td>"
        for e in ("e1", "e2", "total")) + f"<td>{data[s][2]['identity_passed']}</td></tr>" for s in SCENES)
    gal = "".join(f"<section><h3>{html.escape(c)}</h3><img src='{html.escape(f)}' loading='lazy'></section>" for f, c in figs)
    page = f"""<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>WL104K qualitative</title><style>
:root{{--bg:#fff;--fg:#1a1a1a;--mut:#555;--bd:#ddd;--btn:#f0f0f0}}
@media (prefers-color-scheme: dark){{:root:not([data-theme=light]){{--bg:#16181d;--fg:#e8e8e8;--mut:#aaa;--bd:#333;--btn:#2a2d35}}}}
body{{background:var(--bg);color:var(--fg);font-family:system-ui,sans-serif;max-width:1400px;margin:0 auto;padding:16px}}
img{{max-width:100%;border:1px solid var(--bd)}} .sw img{{width:512px;max-width:100%;image-rendering:pixelated}}
button{{margin:2px;padding:4px 8px;background:var(--btn);color:var(--fg);border:1px solid var(--bd);border-radius:4px;cursor:pointer}}
button.on{{outline:2px solid #2ca02c}} .e button{{font-size:12px}} .sw{{display:inline-block;vertical-align:top;margin:8px;max-width:540px}}
.note{{background:#b7791f;color:#fff;padding:6px 10px;border-radius:4px;display:inline-block}} p{{color:var(--mut)}} td,th{{border:1px solid var(--bd);padding:3px 6px}}
table{{border-collapse:collapse}}</style>
<script>function show(id,src,b){{document.getElementById(id).src=src;for(const x of b.parentElement.parentElement.querySelectorAll('button'))x.classList.remove('on');b.classList.add('on');}}</script>
</head><body><h1>WL104K — qualitative comparison on the existing scene data set</h1>
<p class='note'>{html.escape(TAG)}</p>
<p>Three existing WL104 scenes (sphere + cube with the coated-conductor material) rendered with the layered physical reference and with the material
response of A (shared ER-v1-GF), T (one shared symmetric gate) and P (per-component symmetric gates, the ER-v1-SDM candidate), all fitted on the
WL104J angular fit set. All arms are evaluated at the same sampled directions (common R0 proposal, F/p recomputed per arm), so the images differ only
through the material response. The frozen WL104K verdict (INCONCLUSIVE; scene stage not run under the frozen rule) is unchanged; these renders are
owner-requested qualitative material.</p>
<h2>Click to switch (same exposure within each panel)</h2>{''.join(sw)}
<h2>Descriptive scene contrast P − A (objects stable RMS vs reference, mean of 4 starts, 12-replicate jackknife 95 % CI)</h2>
<table><tr><th>scene</th><th>first bounce</th><th>second bounce</th><th>total</th><th>replay identity</th></tr>{an_rows}</table>
<p>Negative = P closer to the reference. "Total" uses the archived R0 higher-order term for every model.</p>
<h2>Figures</h2>{gal}</body></html>"""
    (out / "qualitative.html").write_text(page, encoding="utf-8")
    man = {"post_hoc": True, "owner_request": "2026-11-22", "error_scale_luminance": M, "objects_relative_rms_shown_renders": metrics,
           "scene_contrasts": {s: data[s][2]["primary"] for s in SCENES}, "identity": {s: data[s][2]["identity"] for s in SCENES},
           "commit": CM.commit(), "script_sha256": CM.sha(Path(__file__)),
           "files_sha256": {str(p.relative_to(out)).replace("\\", "/"): CM.sha(p) for p in sorted(out.rglob("*")) if p.suffix in (".png", ".html")}}
    CM.write(out / "qualitative_manifest.json", man)
    print("qualitative done", len(figs), json.dumps({s: {e: {a: round(v, 4) for a, v in d.items()} for e, d in m.items()} for s, m in metrics.items()}), flush=True)


if __name__ == "__main__":
    main()
