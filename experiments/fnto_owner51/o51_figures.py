"""WL51 reviewer figures -> results/evaluation/51/ (slab path-semantics renders on one shared absolute scale; rho by mask).

    python o51_figures.py <repo root>
"""
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main(root):
    root = Path(root); D = root / "results/fnto_ownership51/gate2_renderer"; out = root / "results/evaluation/51"; out.mkdir(parents=True, exist_ok=True)
    imgs = [("J_below: camera vertex (top face)", "Ltop_J_below_5101.npy"), ("J_below: via diffuse patch\n(diffuse ancestor)", "Lpdiff_J_below_5101.npy"),
            ("J_below: via mirror patch\n(singular ancestor)", "Lpmirr_J_below_5101.npy"), ("S_below: camera vertex, top face\n(separate objects)", "Ltop_S_below_5101.npy"),
            ("S_below: camera vertex in the gap\n(bottom slab top face)", "Lgap_S_below_5101.npy"), ("L_above: Lambert control\nvia diffuse patch", "Lpdiff_L_above_5101.npy")]
    A = [np.load(D / f)[..., 0] for _, f in imgs]; vmax = max(float(a.max()) for a in A)
    fig, ax = plt.subplots(2, 3, figsize=(13, 8.5))
    for a, (t, _), x in zip(A, imgs, ax.ravel()):
        im = x.imshow(np.sqrt(np.clip(a, 0, None)), cmap="magma", vmin=0, vmax=np.sqrt(vmax)); x.set_title(t, fontsize=9); x.axis("off")
    cb = fig.colorbar(im, ax=ax.ravel().tolist(), shrink=0.8); tk = np.linspace(0, np.sqrt(vmax), 5); cb.set_ticks(tk); cb.set_ticklabels([f"{v * v:.2f}" for v in tk])
    cb.set_label("red radiance [W m^-2 sr^-1], sqrt scale shared by all panels")
    fig.suptitle("WL51 gate 2 - Cycles 3.5.1 Yellow Burley path semantics (seed 5101, red channel)")
    fig.savefig(out / "01_slab_path_semantics.png", dpi=110, bbox_inches="tight"); plt.close(fig)
    g1 = json.loads((root / "results/fnto_ownership51/gate1_operator/g1.json").read_text())
    masks = ["none", "noYY", "noYO", "noOY", "noOO", "onlyYY", "noYY_t3mm", "noYY_t3cm", "noCross", "SEM"]
    fig, ax = plt.subplots(figsize=(11, 4.5)); w = 0.27
    for i, c in enumerate(["H_5011", "L0C_5011", "L1C_5011"]):
        ax.bar(np.arange(len(masks)) + (i - 1) * w, [g1[c][m]["rho_RGB"][0] for m in masks], w, label=c.replace("_5011", ""))
    ax.axhline(1, color="k", lw=0.8, ls="--"); ax.set_yscale("log"); ax.set_xticks(range(len(masks))); ax.set_xticklabels(masks, rotation=30)
    ax.set_ylabel("red-channel spectral radius (log)"); ax.legend()
    ax.set_title("WL51 gate 1 - red rho under diagnostic edge masks and the SEM counterfactual (diagnostic only, not production)", fontsize=9)
    fig.savefig(out / "02_rho_by_mask.png", dpi=110, bbox_inches="tight"); plt.close(fig)
    src = {"01_slab_path_semantics.png": [str((D / f).relative_to(root)) for _, f in imgs], "02_rho_by_mask.png": ["results/fnto_ownership51/gate1_operator/g1.json"]}
    man = {"worklog": "docs/worklogs/51_WORKLOG_FNTO_TRANSPORT_OWNERSHIP.md", "generator": "experiments/fnto_owner51/o51_figures.py", "artifacts": {}}
    for k, v in src.items():
        man["artifacts"][k] = {"sha256": hashlib.sha256((out / k).read_bytes()).hexdigest(), "sources": v,
                               "note": "shared absolute sqrt scale across panels" if k.startswith("01") else "log scale; masks and SEM are diagnostic counterfactuals"}
    (out / "manifest.json").write_text(json.dumps(man, indent=1)); print("ok", vmax)


if __name__ == "__main__":
    main(sys.argv[1])
