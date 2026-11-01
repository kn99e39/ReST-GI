"""WL69 reviewer figures (report-only; reads analysis.json and the stored images/timings, computes no new metric).

01_quality_vs_latency.png  T1 (edited state) max |group bias| over the ruled cameras vs edit-to-output latency of those views.
02_underup_T1.png          underup view of T1 (slab underside at 4 mm clearance): reference, CY (QM_contract and QM vs RG_CAK), D8;
                           row 2 per-pixel |error| on one shared sqrt scale (max over the figure; radiance units); row 3 blue-channel
                           signed error of 8x8-pixel block means (noise-reduced bias view) on one shared symmetric scale (radiance units).

    cd <repo>/experiments/fnto_k55 && python ../xbench69/x_figures.py <root> <out_dir>
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

CYC = Path.home() / "rest_wl55" / "out" / "cycles_cuda"
RULED = ("top", "underup", "underdown")


def main():
    root = Path(sys.argv[1]); out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
    A = json.load(open(root / "analysis.json")); rg = json.load(open(root / "rg" / "rg.json"))
    man = [json.loads(x) for d in sorted((root / "cy").glob("spp*")) for x in open(d / "render_manifest.jsonl")]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    # CY
    for r, mk in ((0, "o"), (1, "s")):
        xs, ys = [], []
        for spp, v in sorted(A["arms"].get("CY", {}).items(), key=lambda kv: int(kv[0])):
            if str(r) not in v:
                continue
            seeds = sorted({m["seed"] for m in man if m["scene"] == "floor_c4" and m["spp"] == int(spp)})
            t = sum(m["seconds"] for m in man if m["scene"] == "floor_c4" and m["spp"] == int(spp) and m["cam"] in RULED and m["seed"] == seeds[r])
            xs.append(t); ys.append(100 * v[str(r)]["T1"]["M2"])
        ax.plot(xs, ys, mk + "-", color="#3b6ea8", ms=4, lw=1, label="Full Recompute (Cycles)" if r == 0 else None)
    # D8
    for name, col in (("D8_s0", "#c0504d"), ("D8_s1", "#e39b97")):
        if name not in A["arms"]:
            continue
        ev = json.load(open(root / "d8" / f"d8_eval_{name[3:]}.json")); xs, ys = [], []
        for spp, v in sorted(A["arms"][name].items(), key=lambda kv: int(kv[0])):
            t = sum(x["seconds"] for x in ev["renders"] if x["state"] == "T1" and x["spp"] == int(spp) and x["r"] == 0 and x["cam"] in RULED)
            t += ev["t_scene_load_T1_s"] + ev["t_load_model_T1_s"]
            if "0" in v:
                xs.append(t); ys.append(100 * v["0"]["T1"]["M2"])
        ax.plot(xs, ys, "^-", color=col, ms=4, lw=1, label=f"8DNA ({name[3:]}, frozen asset moved)")
    # RG
    for arm, mk, col in (("CAK", "*", "#2e8b57"), ("P10", "D", "#8fbc8f")):
        runs = [v for k, v in rg["runs"].items() if k.startswith("floor_c4/") and arm in k]
        t = float(np.mean([v["t_update_s"] for v in runs]))
        ax.plot([t], [100 * A["arms"][f"RG_{arm}"]["T1"]["M2"]], mk, color=col, ms=11 if mk == "*" else 6, label=f"ReST-GI {arm} (rest K reused)")
    ax.axhline(3, color="0.5", ls=":", lw=1); ax.text(0.01, 3.15, "3 % contract", color="0.4", fontsize=8, transform=ax.get_yaxis_transform())
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("T1 edit-to-output latency for top + underup + underdown [s]")
    ax.set_ylabel("T1 max |group bias| [%]"); ax.set_title("C1 floor -> floor_c4: quality vs latency (LabServer69, RTX 3090)", fontsize=10)
    ax.legend(fontsize=7, frameon=False); fig.tight_layout(); fig.savefig(out / "01_quality_vs_latency.png", dpi=160); plt.close(fig)
    # images
    ref = np.mean([np.load(f)[..., :3] for f in sorted(CYC.glob("cimg_floor_c4_underup_s*.npy"))], 0)
    panels = [("Reference (Cycles 2x1024 spp)", ref)]
    for lab, qm in (("QM contract", A.get("QM", {}).get("CY", {}).get("contract")), ("QM vs CAK", A.get("QM", {}).get("CY", {}).get("RG_CAK"))):
        if qm:
            f = sorted((root / "cy" / f"spp{qm}").glob("cimg_floor_c4_underup_s*.npy"))[0]; panels.append((f"Full Recompute {qm} spp ({lab})", np.load(f)[..., :3]))
    for tag in ("s0", "s1"):
        f = root / "d8" / f"d8_{tag}_T1_underup_spp1024_r0.npy"
        if f.exists():
            panels.append((f"8DNA {tag} 1024 spp", np.load(f)))
    errs = [np.abs(p - ref).mean(-1) for _, p in panels[1:]]; emax = max(float(e.max()) for e in errs) if errs else 1.0
    vmax = float(np.percentile(ref, 99.5))
    def blk(x, b=8):
        H, W = (x.shape[0] // b) * b, (x.shape[1] // b) * b
        return x[:H, :W].reshape(H // b, b, W // b, b).mean((1, 3))
    rb = blk(ref[..., 2]); sgn = [blk(p[..., 2]) - rb for _, p in panels[1:]]; smax = max(float(np.abs(e).max()) for e in sgn) if sgn else 1.0
    fig, axs = plt.subplots(3, len(panels), figsize=(3.0 * len(panels), 9.2), squeeze=False)
    for j, (t, p) in enumerate(panels):
        axs[0, j].imshow(np.clip(p / vmax, 0, 1) ** (1 / 2.2), origin="lower"); axs[0, j].set_title(t, fontsize=8); axs[0, j].axis("off")
        if j == 0:
            axs[1, j].axis("off"); axs[2, j].axis("off"); continue
        im2 = axs[2, j].imshow(sgn[j - 1], origin="lower", cmap="RdBu_r", vmin=-smax, vmax=smax); axs[2, j].axis("off")
        axs[2, j].set_title(f"blue 8x8 signed, mean {sgn[j - 1][rb > 0.05 * rb.max()].mean():+.4f}", fontsize=8)
        im = axs[1, j].imshow(np.sqrt(errs[j - 1]), origin="lower", cmap="magma", vmin=0, vmax=np.sqrt(emax)); axs[1, j].axis("off")
        axs[1, j].set_title(f"|error| mean {errs[j - 1].mean():.4f}", fontsize=8)
    if errs:
        cb = fig.colorbar(im, ax=axs[1, 1:].tolist(), shrink=0.8); tk = np.linspace(0, np.sqrt(emax), 5); cb.set_ticks(tk); cb.set_ticklabels([f"{x * x:.3g}" for x in tk])
        cb.set_label("|error| (radiance, sqrt scale)", fontsize=8)
        cb2 = fig.colorbar(im2, ax=axs[2, 1:].tolist(), shrink=0.8); cb2.set_label("blue signed error (radiance)", fontsize=8)
    fig.suptitle("T1 underup (slab underside, 4 mm clearance)", fontsize=10); fig.savefig(out / "02_underup_T1.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
    main()
