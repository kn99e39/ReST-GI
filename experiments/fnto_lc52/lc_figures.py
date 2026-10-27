"""WL52 reviewer figures -> results/evaluation/52/ (renders on one display transform per configuration; error maps on ONE shared
absolute scale across every panel of the worklog's error figure; sqrt map with a real-unit colour bar).

    python lc_figures.py <repo root>
"""
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ARMS = ["REF", "ORA", "C0", "C0p", "C1"]


def main(root):
    root = Path(root); R = root / "results/fnto_lc52"; out = root / "results/evaluation/52"; out.mkdir(parents=True, exist_ok=True); man = {"worklog": "docs/worklogs/52_WORKLOG_LEARNED_INTERDOMAIN_TRANSPORT.md", "artifacts": {}}
    im = np.load(R / "eval/images.npz"); keys = sorted({k.rsplit("__", 1)[0] for k in im.files})
    pick = [k for k in keys if k.startswith("test_iid")][:2] + [k for k in keys if k.startswith("traj")][:6:2] + [k for k in keys if k.startswith("scale_obj")][:1] + [k for k in keys if k.startswith("combo")][:1]
    fig, ax = plt.subplots(len(pick), 5, figsize=(15, 3 * len(pick)))
    for i, k in enumerate(pick):
        for j, a in enumerate(ARMS):
            x = ax[i, j]; x.imshow(np.clip(im[f"{k}__{a}"].astype(np.float32), 0, 1)); x.set_xticks([]); x.set_yticks([])
            if i == 0:
                x.set_title(a)
            if j == 0:
                x.set_ylabel(k.replace("____", " ").replace("__", " "), fontsize=7)
    fig.suptitle("WL52 coupled GI, camera 0: REF path-traced | ORA 2048 gathers | C0 128 gathers | C0p ray-traced patch transfer | C1 learned (display: clip(x * e)^(1/2.2), e = 1 / p99 of REF)", fontsize=9)
    fig.savefig(out / "01_coupled_renders.png", dpi=90, bbox_inches="tight"); plt.close(fig)
    em = np.load(R / "local/errmaps.npz"); ek = sorted({k.rsplit("__", 1)[0] for k in em.files}); vmax = max(float(em[f].max()) for f in em.files)
    fig, ax = plt.subplots(len(ek), 3, figsize=(10, 3.2 * len(ek)))
    for i, k in enumerate(ek):
        for j, a in enumerate(["C0", "C0p", "C1"]):
            x = ax[i, j]; h = x.imshow(np.sqrt(em[f"{k}__{a}"].astype(np.float32)), cmap="magma", vmin=0, vmax=np.sqrt(vmax)); x.set_xticks([]); x.set_yticks([])
            if i == 0:
                x.set_title(a)
            if j == 0:
                x.set_ylabel(k.replace("____", " ").replace("__", " "), fontsize=7)
    cb = fig.colorbar(h, ax=ax.ravel().tolist(), shrink=0.6); tk = np.linspace(0, np.sqrt(vmax), 5); cb.set_ticks(tk); cb.set_ticklabels([f"{v * v:.2f}" for v in tk])
    cb.set_label("|E_ind - E_ind,ORA| / mean |E_ind,ORA|  (sqrt scale, shared by all panels)")
    fig.suptitle("WL52 coupled indirect-irradiance error vs the oracle solution (post-hoc diagnostic, amendment 3)", fontsize=9)
    fig.savefig(out / "02_coupled_error_maps.png", dpi=90, bbox_inches="tight"); plt.close(fig)
    ev = json.loads((R / "eval/eval.json").read_text()); sets = ["train_cfg", "test_iid", "combo", "scale_obj", "traj"]; fields = ["smooth", "hot", "first", "noise"]
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.8), sharey=True)
    for j, fn in enumerate(fields):
        for k, a in enumerate(["ORA2", "C0", "C0p", "C1"]):
            v = [np.mean([r["single_flight"][fn][a]["all"] for r in ev["sets"][s]]) for s in sets]; ax[j].bar(np.arange(len(sets)) + (k - 1.5) * 0.2, v, 0.2, label=a)
        ax[j].set_xticks(range(len(sets))); ax[j].set_xticklabels(sets, rotation=30, fontsize=8); ax[j].set_title(f"single flight, field '{fn}'")
    ax[0].set_ylabel("relative L2 of E_ind vs ORA"); ax[0].legend(fontsize=8)
    fig.savefig(out / "03_single_flight_errors.png", dpi=90, bbox_inches="tight"); plt.close(fig)
    for f in sorted(out.glob("*.png")):
        man["artifacts"][f.name] = {"sha256": hashlib.sha256(f.read_bytes()).hexdigest(), "sources": ["results/fnto_lc52/eval/images.npz", "results/fnto_lc52/local/errmaps.npz", "results/fnto_lc52/eval/eval.json"]}
    (out / "manifest.json").write_text(json.dumps(man, indent=1)); print("ok", vmax)


if __name__ == "__main__":
    main(sys.argv[1])
