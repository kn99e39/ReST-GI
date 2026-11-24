"""WL111 reviewer artifacts (not part of any decision): frontier curves, runtime breakdown, convergence, per-frame latency,
identically exposed images and signed error maps, silhouette crops, history age / acceptance maps, sequence animations,
abrupt-edit before / after strips, per-region tables. Reads the evidence only.

python figs_r111.py ROOT OUT
Display transform for every image: LDR = clip(x, 0, 1) ^ (1 / 2.2) at exposure 1 (WL110 convention). Signed error maps:
(luminance(y) - luminance(ref)) / mean ROI luminance of the reference, fixed symmetric scale per figure.
"""
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import common_r111 as C

COL = {"R0": "#9e9e9e", "R3": "#d62728", "SVGF": "#8c564b", "S0": "#1f77b4", "ST0": "#2ca02c", "S1": "#aec7e8", "ST1": "#98df8a"}
LABEL = {"R0": "R0 native", "R3": "R3 (frozen)", "SVGF": "SVGF (WL101B)", "S0": "S (spatial)", "ST0": "ST (spatiotemporal)",
         "S1": "S r1", "ST1": "ST r1"}


def ldr(x):
    return np.clip(np.transpose(np.asarray(x, np.float32), (1, 2, 0)), 0, 1) ** (1 / 2.2)


def lum(x):
    return np.tensordot(C.LUMA, np.asarray(x, np.float64), 1)


def frontier(an, out):
    for regime, title in (("static_warm", "static warm (sealed K1-K6, frame 64)"), ("cold_reset", "cold / reset (sealed, frame 1)"),
                          ("static_warm_historical", "static warm (historical V0/H1-H5)"),
                          ("continuous_edit", "continuous geometry motion Q4"), ("camera_motion", "camera orbit Q2")):
        reg = an["regimes"].get(regime, {})
        r = reg.get("anchor4_r0")
        if not r or not r.get("points"):
            continue
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        pts = r["points"]
        ax.plot([p["e2e_ms"] for p in pts], [p["rmse_ratio"] for p in pts], "o-", color=COL["ST0"], label=f"{r['candidate']} (r0) / R3(4 spp)")
        for p in pts:
            ax.annotate(f"{p['spp']}", (p["e2e_ms"], p["rmse_ratio"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
            ax.plot([p["e2e_ms"]] * 2, p["ci95"], color=COL["ST0"], lw=0.8)
        r1 = reg.get("anchor4_r1")
        if r1 and r1.get("points"):
            ax.plot([p["e2e_ms"] for p in r1["points"]], [p["rmse_ratio"] for p in r1["points"]], "s--", color=COL["ST1"], ms=3,
                    label="realization r1")
        ax.axhline(1.0, color=COL["R3"], lw=1)
        ax.axhline(1.05, color=COL["R3"], lw=0.6, ls=":")
        ax.axvline(r["R3_e2e_ms"], color=COL["R3"], lw=1, label=f"R3 4 spp E2E {r['R3_e2e_ms']:.1f} ms")
        ax.axvline(0.85 * r["R3_e2e_ms"], color=COL["R3"], lw=0.6, ls=":")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("median end-to-end frame time (ms)")
        ax.set_ylabel("primary RMSE ratio vs R3 at 4 spp (GM)")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / f"frontier_{regime}.png", dpi=130)
        plt.close(fig)


def runtime(root, out):
    p = root / "timing" / "components.json"
    sA = root / "stageA" / "stageA.json"
    if not p.exists() or not sA.exists():
        return
    comp = C.read(p)["times_ms"]
    med_ = lambda rows: {k: float(np.median([r[k] for r in rows])) for k in rows[0]}
    cand, r3g, r3e = med_(comp["candidate"]), med_(comp["r3_graph"]), med_(comp["r3_eager_stages"])
    bars = [("R3 eager stages (static)", r3e), ("R3 graph + 8x8 guides (camera)", r3g), ("ST graph + guide pack (camera)", cand)]
    keys = []
    for _, d in bars:
        for k in d:
            if k not in keys:
                keys.append(k)
    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    seen = set()
    for i, (name, d) in enumerate(bars):
        left = 0
        for k in keys:
            v = d.get(k, 0)
            if v:
                ax.barh(i, v, left=left, color=cmap(keys.index(k)), label=None if k in seen else k)
                seen.add(k)
                left += v
        ax.text(left, i, f" {left:.2f} ms", va="center", fontsize=8)
    ax.set_yticks(range(len(bars)))
    ax.set_yticklabels([b[0] for b in bars], fontsize=8)
    ax.set_xlabel("GPU span per frame at 512 x 512, 4 spp (ms, CUDA events, median)")
    ax.legend(fontsize=6, ncol=3, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "runtime_breakdown.png", dpi=130)
    plt.close(fig)
    C.write(out / "runtime_breakdown.json", {"candidate_camera_form": cand, "r3_graph_camera_form": r3g, "r3_eager_stages": r3e,
                                             "stageA_candidate_breakdown": C.read(sA)["timing"].get("candidate_breakdown_camera_form_m1")})


def convergence(root, out):
    views = C.P["views"]["sealed"]["names"]
    frames = C.P["static_protocol"]["recorded_frames"]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for spp, ls in ((1, "-"), (2, "--"), (4, ":")):
        ys = []
        for fr in frames:
            vals = []
            for v in views:
                for f in (root / "static").glob(f"{v}_spp{spp:02d}_r*.json"):
                    m = C.read(f)["metrics"].get(f"ST0_f{fr}", {}).get("primary")
                    if m:
                        vals.append(m["rel_rmse"])
            ys.append(np.mean(vals) if vals else np.nan)
        ax.plot(frames, ys, ls, color=COL["ST0"], label=f"ST {spp} spp")
    r3 = []
    for v in views:
        for f in (root / "static").glob(f"{v}_spp04_r*.json"):
            m = C.read(f)["metrics"].get("R3_f1", {}).get("primary")
            if m:
                r3.append(m["rel_rmse"])
    if r3:
        ax.axhline(np.mean(r3), color=COL["R3"], label="R3 4 spp (any frame)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("frame (static camera, history accumulating)")
    ax.set_ylabel("primary relative HDR RMSE (sealed views, mean)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "temporal_convergence.png", dpi=130)
    plt.close(fig)


def latency(root, out):
    for name in ("camera_Q2", "continuous_Q4"):
        p = root / "timing" / f"{name}.json"
        if not p.exists():
            continue
        t = C.read(p)["times_ms"]
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        for an in ("R0", "R3", "SVGF", "S0", "ST0"):
            v = t.get(f"{an}_4")
            if v:
                ax.plot(v, lw=0.8, color=COL[an], label=f"{LABEL[an]} median {np.median(v):.1f} / p95 {np.percentile(v, 95):.1f} ms")
        for y, l in ((10, "10"), (15, "15"), (20, "20")):
            ax.axhline(y, color="k", lw=0.4, ls=":")
        ax.set_xlabel("measured frame (both passes, warm-up removed)")
        ax.set_ylabel("E2E ms (4 spp)")
        ax.set_title(name, fontsize=10)
        ax.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(out / f"latency_{name}.png", dpi=130)
        plt.close(fig)


def images(root, out):
    def ref_np(v):
        if v in C.P["views"]["historical"]:
            mean = np.load(C.REF107 / f"{v}_target.npy").astype(np.float64)
            with np.load(C.D105 / "eval" / v / "aux.npz") as z:
                aux = {k: z[k] for k in z.files}
            return mean, aux
        with np.load(root / "refs" / f"{v}.npz") as z:
            return z["mean"].astype(np.float64), {k[4:]: z[k] for k in z.files if k.startswith("aux_")}
    for v in ("V0", "K1", "H3"):
        for spp in (1, 4):
            p = root / "static" / f"{v}_spp{spp:02d}_r0_images.npz"
            if not p.exists():
                continue
            mean, aux = ref_np(v)
            ref = mean[0:3] + mean[3:6] + mean[6:9]
            masks = C.rois(aux)
            z = np.load(p)
            panels = [("reference (T_ECRO)", ref)] + [(f"{LABEL[a]} f{fr}", z[f"{a}_f{fr}"].astype(np.float64))
                                                      for a, fr in (("R0", 1), ("R3", 1), ("SVGF", 64), ("S0", 1), ("ST0", 64)) if f"{a}_f{fr}" in z]
            mlum = lum(ref)[masks["primary"]].mean()
            fig, axs = plt.subplots(2, len(panels), figsize=(2.6 * len(panels), 5.4))
            for i, (name, y) in enumerate(panels):
                axs[0, i].imshow(ldr(y))
                axs[0, i].set_title(name, fontsize=7)
                axs[1, i].imshow((lum(y) - lum(ref)) / mlum, cmap="RdBu_r", vmin=-0.3, vmax=0.3)
                for a in axs[:, i]:
                    a.axis("off")
            fig.suptitle(f"{v} {spp} spp: LDR (exposure 1) and signed luminance error / mean (±0.3)", fontsize=9)
            fig.tight_layout()
            fig.savefig(out / f"images_{v}_{spp}spp.png", dpi=110)
            plt.close(fig)
            sil = masks["silhouette"]
            ys, xs = np.nonzero(sil)
            if len(xs):
                k = np.argmin(xs)
                cy, cx = int(ys[k]), int(xs[k])
                y0, x0 = max(cy - 24, 0), max(cx - 24, 0)
                fig, axs = plt.subplots(2, len(panels), figsize=(2.2 * len(panels), 4.6))
                for i, (name, y) in enumerate(panels):
                    axs[0, i].imshow(ldr(y[:, y0:y0 + 48, x0:x0 + 48]), interpolation="nearest")
                    axs[0, i].set_title(name, fontsize=7)
                    axs[1, i].imshow(((lum(y) - lum(ref)) / mlum)[y0:y0 + 48, x0:x0 + 48], cmap="RdBu_r", vmin=-0.3, vmax=0.3,
                                     interpolation="nearest")
                    for a in axs[:, i]:
                        a.axis("off")
                fig.suptitle(f"{v} {spp} spp left-silhouette crop (48 px)", fontsize=9)
                fig.tight_layout()
                fig.savefig(out / f"silhouette_{v}_{spp}spp.png", dpi=120)
                plt.close(fig)


def sequences(root, out):
    from PIL import Image, GifImagePlugin  # noqa: F401  (registers the GIF writer)
    for name in ("Q2", "Q4", "Q3", "Q5"):
        for spp in (1, 4):
            p = root / "seq" / f"{name}_spp{spp}_r0_frames.npz"
            j = root / "seq" / f"{name}_spp{spp}_r0.json"
            if not p.exists():
                continue
            z = np.load(p)
            order = C.read(j)["arms_order"]
            show = [a for a in ("R0", "R3", "SVGF", "S0", "ST0") if a in order]
            keys = sorted(k for k in z.files if k.startswith("f"))
            tiles = []
            for k in keys:
                st = z[k]
                row = np.concatenate([ldr(st[order.index(a)]) for a in show], 1)
                tiles.append(Image.fromarray((row * 255).astype(np.uint8)).resize((row.shape[1] // 2, row.shape[0] // 2)))
            if name in ("Q2", "Q4"):
                tiles[0].save(out / f"video_{name}_{spp}spp.gif", format="GIF", save_all=True, append_images=tiles[1:], duration=80, loop=0)
            else:
                ed = C.P["sequences"][name]["edit_frame"]
                pick = [i for i, k in enumerate(keys) if int(k[1:]) in (ed - 1, ed, ed + 1, ed + 8, ed + 32)]
                strip = np.concatenate([np.asarray(tiles[i]) for i in pick], 0)
                Image.fromarray(strip).save(out / f"edit_{name}_{spp}spp_before_after.png")
            if name == "Q2":
                ak = sorted(k for k in z.files if k.startswith("age"))
                if ak:
                    mid = ak[len(ak) // 2]
                    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
                    axs[0].imshow(z[mid], cmap="viridis", vmin=0, vmax=32)
                    axs[0].set_title(f"ST history age (frames), {name} frame {mid[3:]}", fontsize=8)
                    axs[1].imshow(z[mid.replace("age", "acc")], cmap="gray")
                    axs[1].set_title("history accepted (white) / rejected (black)", fontsize=8)
                    for a in axs:
                        a.axis("off")
                    fig.tight_layout()
                    fig.savefig(out / f"history_maps_{name}_{spp}spp.png", dpi=110)
                    plt.close(fig)
        # header of the tiles
    C.write(out / "sequence_tiles_order.json", {"columns": ["R0", "R3", "SVGF", "S0", "ST0"], "rows": "frames in order"})


def main(root, out):
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    an = C.read(root / "analysis.json")
    for f in (frontier, ):
        f(an, out)
    for f in (runtime, convergence, latency, images, sequences):
        try:
            f(root, out)
        except Exception as e:  # noqa: BLE001  (reviewer figures never decide anything; report and continue)
            print("figure failed", f.__name__, repr(e), flush=True)
    print("FIGS_COMPLETE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
