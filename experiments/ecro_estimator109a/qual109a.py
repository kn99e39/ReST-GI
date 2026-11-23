"""Qualitative comparison sheets for WL109 / WL109A reviewers (CPU, descriptive; no decision, no new render).

Arms, all from the authenticated WL109 frames:
  R0      native ECRO Monte Carlo renderer (no denoiser), 4 spp, paired replicate r
  C1      frozen C1 (M = 1), same seed: TOTAL = LOW(R0 frame) + C1 HIGH
  R0-eq   native R0 at the lowest grid spp whose WL109 median frame time is >= C1's 4-spp time (R0 gets at least C1's time)
References:
  physical   WL101 archived six-event physical reference of config 0 / light 0 / V0 (two 4096-spp halves; V0 only)
  ECRO-exp   WL107 qualified frozen-ECRO expectation (48 x 8192 spp), the WL109 metric target (all views)

python qual109a.py WL109_EVIDENCE_DIR PHYS_REF_DIR OUT
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from common109a import EVAL_VIEWS, LUMA, all_masks, read
import runtime109 as RT

SPP = 4
REP = 0
CROP_ROI = {"V0": "silhouette", "H1": "material_boundary", "H2": "material_boundary", "H3": "material_boundary",
            "H4": "material_boundary", "H5": "material_boundary"}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def tonemap(x, exposure):
    y = np.clip(x * exposure, 0, None)
    y = y / (1 + y)
    return np.clip(y, 0, 1) ** (1 / 2.2)


def rgb(x, e):
    return tonemap(x, e).transpose(1, 2, 0)


def relrmse(y, ref, m):
    return float(np.sqrt(((y - ref)[:, m] ** 2).mean() / (ref[:, m] ** 2).mean()))


def crop_box(m, size=96):
    r, c = np.nonzero(m)
    k = len(r) // 2
    order = np.argsort(r * 512 + c)
    rc, cc = r[order[k]], c[order[k]]
    r0 = int(np.clip(rc - size // 2, 0, 512 - size)); c0 = int(np.clip(cc - size // 2, 0, 512 - size))
    return r0, r0 + size, c0, c0 + size


def main(ev, phys, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    out.mkdir(parents=True, exist_ok=True)
    run = ev / "run"
    man = read(run / "manifest.json")["files"]
    met = read(ev / "metrics/metrics109.json")
    used = {}

    def load(rel):
        p = run / "raw" / rel
        assert sha(p) == man[rel], rel
        used[f"wl109:{rel}"] = man[rel]
        return np.load(p).astype(np.float64)

    pf = [phys / f"REF_0_{j}.npz" for j in range(2)]
    for p in pf:
        used[f"wl101_phys:{p.name}"] = sha(p)
    PH = np.mean([(lambda f: (f["e1"] + f["e2"] + f["h"]).astype(np.float64))(np.load(p)) for p in pf], 0).transpose(2, 0, 1)
    summary = {}
    exp_rows = []
    for view in EVAL_VIEWS:
        masks = all_masks(RT.aux(view))
        prim = masks["primary"]
        refc, vref, rec = RT.reference(view)
        ref = refc[0:3] + refc[3:6] + refc[6:9]
        tm = met["views"][view]["timing_median_ms"]
        tc = tm[f"C1_{SPP}"]
        levels = sorted(int(k[3:]) for k in tm if k.startswith("R0_"))
        eq = min(s for s in levels if tm[f"R0_{s}"] >= tc)
        r0 = load(f"paired/{view}/spp{SPP}/r{REP:02d}_R0.npy")
        c1h = load(f"paired/{view}/spp{SPP}/r{REP:02d}_C1H.npy")
        T0 = r0[0:3] + r0[3:6] + r0[6:9]
        T1 = r0[0:3] + r0[3:6] + c1h
        g = load(f"grid/{view}/spp{eq:03d}/r{REP:02d}.npy")
        Teq = g[0:3] + g[3:6] + g[6:9]
        # 32-replicate statistics (paired) for numbers and the expectation sheet
        R = met["paired_replicates"]
        s0, s1, seq = [], [], []
        sumH0 = np.zeros((3, 512, 512)); sumH1 = np.zeros_like(sumH0); sumD2 = np.zeros_like(sumH0); sumT0 = np.zeros_like(sumH0); sumT1 = np.zeros_like(sumH0)
        for r in range(R):
            a = load(f"paired/{view}/spp{SPP}/r{r:02d}_R0.npy"); b = load(f"paired/{view}/spp{SPP}/r{r:02d}_C1H.npy")
            t0 = a[0:3] + a[3:6] + a[6:9]; t1 = a[0:3] + a[3:6] + b
            s0.append(relrmse(t0, ref, prim)); s1.append(relrmse(t1, ref, prim))
            d = b - a[6:9]
            sumH0 += a[6:9]; sumH1 += b; sumD2 += d * d; sumT0 += t0; sumT1 += t1
        Rg = met["grid_replicates"]
        for r in range(Rg):
            gg = load(f"grid/{view}/spp{eq:03d}/r{r:02d}.npy")
            seq.append(relrmse(gg[0:3] + gg[3:6] + gg[6:9], ref, prim))
        Dm = (sumH1 - sumH0) / R
        Dsd = np.sqrt(np.maximum(sumD2 / R - Dm ** 2, 0) * R / (R - 1))
        z = (LUMA @ Dm.reshape(3, -1)).reshape(512, 512) / np.maximum((LUMA @ Dsd.reshape(3, -1)).reshape(512, 512) / np.sqrt(R), 1e-12)
        lref = (LUMA @ ref.reshape(3, -1)).reshape(512, 512)
        exp_rows.append((view, sumT0 / R, sumT1 / R, ref, Dm, z, lref, masks))
        info = {"C1_4spp_ms": tc, "R0_equal_time_spp": eq, "R0_equal_time_ms": tm[f"R0_{eq}"], "R0_4spp_ms": tm["R0_4"],
                "relrmse_vs_ECROexp_primary_mean32": {"R0_4": float(np.mean(s0)), "C1_4": float(np.mean(s1)), f"R0_{eq}": float(np.mean(seq))},
                "relrmse_vs_ECROexp_primary_shown_frame": {"R0_4": relrmse(T0, ref, prim), "C1_4": relrmse(T1, ref, prim), f"R0_{eq}": relrmse(Teq, ref, prim)}}
        if view == "V0":
            pm = prim & ~masks.get("contact", np.zeros_like(prim))
            info["relrmse_vs_physical_primary_shown_frame"] = {"ECRO_exp": relrmse(ref, PH, pm), "R0_4": relrmse(T0, PH, pm),
                                                               "C1_4": relrmse(T1, PH, pm), f"R0_{eq}": relrmse(Teq, PH, pm)}
        summary[view] = info
        # ------------------------------------------------------------ per-view sheet
        e = 1.0 / max(np.percentile(lref[prim], 90), 1e-6)
        cols = [("physical reference (WL101, 8192 spp)" if view == "V0" else "ECRO expectation (WL107, 48x8192 spp)\nno physical reference for this view", PH if view == "V0" else ref),
                (f"R0 native ECRO, {SPP} spp ({tm['R0_4']:.1f} ms)", T0),
                (f"C1 (M = 1), {SPP} spp ({tc:.1f} ms)", T1),
                (f"R0 at >= C1 time: {eq} spp ({tm[f'R0_{eq}']:.1f} ms)", Teq)]
        roi = CROP_ROI[view]
        rb = crop_box(masks[roi])
        fig, axs = plt.subplots(3, 4, figsize=(17, 13.6), constrained_layout=True)
        for j, (title, img) in enumerate(cols):
            ax = axs[0, j]; ax.imshow(rgb(img, e)); ax.set_title(title, fontsize=9)
            ax.add_patch(Rectangle((rb[2], rb[0]), rb[3] - rb[2], rb[1] - rb[0], fill=False, ec="#ffcc00", lw=1.2))
            if j:
                k = ["R0_4", "C1_4", f"R0_{eq}"][j - 1]
                txt = f"relRMSE vs ECRO-exp (primary)\nthis frame {info['relrmse_vs_ECROexp_primary_shown_frame'][k]:.3f}\nmean of {R if j < 3 else Rg} {info['relrmse_vs_ECROexp_primary_mean32'][k]:.3f}"
                if view == "V0":
                    txt += f"\nvs physical (no contact) {info['relrmse_vs_physical_primary_shown_frame'][k]:.3f}"
                ax.text(4, 508, txt, color="w", fontsize=7, va="bottom", bbox=dict(fc="k", alpha=0.55, lw=0))
            ax.axis("off")
        # row 2: relative luminance error vs ECRO expectation (V0 col 0: ECRO expectation vs physical)
        lim = 1.0
        for j, (title, img) in enumerate(cols):
            ax = axs[1, j]
            if j == 0:
                if view == "V0":
                    er = ((LUMA @ ref.reshape(3, -1)) - (LUMA @ PH.reshape(3, -1))).reshape(512, 512) / np.maximum((LUMA @ PH.reshape(3, -1)).reshape(512, 512), 1e-3)
                    im = ax.imshow(er, cmap="RdBu_r", vmin=-lim, vmax=lim)
                    ax.set_title(f"ECRO expectation vs physical (signed rel. luma)\nrelRMSE primary w/o contact {info['relrmse_vs_physical_primary_shown_frame']['ECRO_exp']:.3f}", fontsize=8)
                else:
                    ov = rgb(ref, e).copy()
                    for nm, cc in (("silhouette", (1, 0.2, 0.2)), ("material_boundary", (0.2, 1, 0.2)), ("contact", (0.3, 0.5, 1))):
                        if nm in masks:
                            ov[masks[nm]] = 0.4 * ov[masks[nm]] + 0.6 * np.asarray(cc)
                    ax.imshow(ov); ax.set_title("ROIs: silhouette red, material boundary green, contact blue", fontsize=8)
            else:
                er = ((LUMA @ img.reshape(3, -1)) - (LUMA @ ref.reshape(3, -1))).reshape(512, 512) / np.maximum(lref, 1e-3)
                im = ax.imshow(er, cmap="RdBu_r", vmin=-lim, vmax=lim)
                ax.set_title("signed relative luma error vs ECRO expectation", fontsize=8)
            ax.axis("off")
        fig.colorbar(im, ax=axs[1, :], shrink=0.6, label="(Y - ref) / ref, luma")
        # row 3: crops
        for j, (title, img) in enumerate(cols):
            ax = axs[2, j]
            ax.imshow(rgb(img[:, rb[0]:rb[1], rb[2]:rb[3]], e), interpolation="nearest")
            ax.contour(masks[roi][rb[0]:rb[1], rb[2]:rb[3]].astype(float), levels=[0.5], colors="#ffcc00", linewidths=0.6)
            ax.set_title(f"crop: {roi} (yellow contour)", fontsize=8); ax.axis("off")
        fig.suptitle(f"{view}: physical / ECRO references vs native ECRO R0 vs C1 (WL109 frames, replicate {REP}, same seed for R0 and C1). Descriptive only.", fontsize=10)
        fig.savefig(out / f"compare_{view}_4spp.png", dpi=100); plt.close(fig)
        print(view, "done", flush=True)
    # ---------------------------------------------------------------- expectation sheet (32-replicate means)
    fig, axs = plt.subplots(len(exp_rows), 4, figsize=(15, 3.9 * len(exp_rows)), constrained_layout=True)
    for i, (view, m0, m1, ref, Dm, z, lref, masks) in enumerate(exp_rows):
        prim = masks["primary"]
        e = 1.0 / max(np.percentile(lref[prim], 90), 1e-6)
        axs[i, 0].imshow(rgb(m0, e)); axs[i, 0].set_title(f"{view}: mean of 32 R0 frames, 4 spp", fontsize=8)
        axs[i, 1].imshow(rgb(m1, e)); axs[i, 1].set_title(f"{view}: mean of 32 C1 frames, 4 spp", fontsize=8)
        rd = (LUMA @ Dm.reshape(3, -1)).reshape(512, 512) / np.maximum(lref, 1e-3)
        im2 = axs[i, 2].imshow(rd, cmap="RdBu_r", vmin=-0.2, vmax=0.2); axs[i, 2].set_title("paired mean (C1 - R0) HIGH / ref luma", fontsize=8)
        im3 = axs[i, 3].imshow(z, cmap="RdBu_r", vmin=-4, vmax=4); axs[i, 3].set_title("per-pixel z of the paired mean (32 pairs)", fontsize=8)
        for a in axs[i]:
            a.axis("off")
    fig.colorbar(im2, ax=axs[:, 2], shrink=0.3); fig.colorbar(im3, ax=axs[:, 3], shrink=0.3)
    fig.suptitle("Expectation view (WL109 paired frames): per-pixel paired mean difference. Pixel z values are not a test; the Q / B1 decisions are in metrics109.json", fontsize=10)
    fig.savefig(out / "expectation_all_views_4spp.png", dpi=90); plt.close(fig)
    rec = {"description": __doc__, "summary": summary, "inputs_sha256": used, "code_sha256": sha(Path(__file__)),
           "outputs": {p.name: sha(p) for p in sorted(out.glob("*.png"))}}
    (out / "qualitative109a.json").write_text(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), Path(sys.argv[3]).resolve())
