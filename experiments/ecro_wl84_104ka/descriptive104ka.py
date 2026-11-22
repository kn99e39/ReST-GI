"""WL104K-A ADDITIONAL DESCRIPTIVE ANALYSIS (not predeclared; written after the frozen decision, changes no label):
(1) where the rendered first-bounce squared-error change P vs A comes from (regions, top-pixel concentration), per start;
(2) per-start peak height on the independent WL104D peak grid for the scene materials.

Usage: python descriptive104ka.py <run_root> <out_dir>
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import common104ka as KA  # noqa: E402

CM = KA.CM
sys.path.insert(0, str(KA.HERE.parent / "ecro_wl84_104k"))
import scenes104k as SK  # noqa: E402

L = CM.LUMA
SCN = {"test83_iid_1_L2": ("iid_1 scene", "test83_iid_1"), "test83_hard_1_L1": ("hard_1 scene", "test83_hard_1"),
       "test83_iid_159_L2": ("iid_159 scene (held out)", "test83_iid_159")}
REG = ("highlight_core", "highlight_ring", "sphere_lower", "grazing", "cube")


def main():
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    rows, raw, _, _, ref, _ = CM.inputs()
    n = len(rows)
    _, peak = CM.eval_reference(ref, n)
    res = {"post_hoc_descriptive": True, "squared_error": {}, "peak": {}}
    for tag, base in (("12k", KA.K_ROOT / "posthoc_scenes"), ("extended", root / "scenes_ext")):
        for s in KA.SCENES:
            arch, masks, _ = SK.S8.archived(s, None)
            fx = {}
            for r in range(12):
                with np.load(base / s / f"fixed_r{r:03d}.npz") as z:
                    for q in z.files:
                        if q.endswith("|e1"):
                            fx.setdefault(q, []).append(z[q] @ L)
            fx = {q: np.mean(v, 0) for q, v in fx.items()}
            refm = (arch["reference_e1"] @ L).mean(0)
            obj = masks["objects"]
            rows_ = []
            for r in range(4):
                eA = fx["N|e1"] + fx[f"A_{r}|e1"] - refm
                eP = fx["N|e1"] + fx[f"P_{r}|e1"] - refm
                d = eP ** 2 - eA ** 2
                seA = (eA ** 2)[obj].sum()
                top = np.sort(d[obj])[::-1]
                rows_.append({"total_change_of_A_SE": float(d[obj].sum() / seA),
                              "by_region": {k: float(d[masks[k] & obj].sum() / seA) for k in REG if k in masks},
                              "top20_pixels": float(top[:20].sum() / seA), "top100_pixels": float(top[:100].sum() / seA)})
            res["squared_error"].setdefault(tag, {})[s] = rows_
    for tag, base in (("12k", KA.K_ROOT / "fit_primary"), ("extended", root / "ext")):
        st, R, _ = KA.states(base, n)
        for s, (_, mt) in SCN.items():
            m = [i for i, r in enumerate(rows) if r["tag"] == mt][0]
            d = {}
            for arm in CM.ARMS:
                d[arm] = []
                for r in range(R):
                    s1 = {k: v[m:m + 1] for k, v in st[arm][r].items()}
                    pk = CM.predict(arm, s1, peak["wi"][m:m + 1], peak["wo"][m:m + 1])[0] @ L
                    d[arm].append((pk.max(-1) / peak["ref"][m].max(-1) - 1).tolist())
            res["peak"].setdefault(tag, {})[s] = d
    CM.write(out / "descriptive.json", res)
    # figure: hard_1 per start: squared-error concentration and peak height vs incidence
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    s = "test83_hard_1_L1"
    sq = res["squared_error"]["12k"][s]
    x = np.arange(4)
    axs[0].bar(x - 0.2, [100 * q["total_change_of_A_SE"] for q in sq], 0.4, color="#2ca02c", label="all object pixels")
    axs[0].bar(x + 0.2, [100 * q["top20_pixels"] for q in sq], 0.4, color="#c0392b", label="the 20 pixels with the largest increase")
    axs[0].axhline(0, color="k", lw=0.8); axs[0].legend(fontsize=8)
    axs[0].set_xticks(x, ["start 1", "start 2", "start 3", "start 4"])
    axs[0].set_ylabel("change of first-bounce squared error, P vs A\n(% of A's squared error, objects)")
    axs[0].set_title("hard_1: the image regression is a handful of highlight-peak pixels", fontsize=10)
    pol = CM.POLAR
    for arm, cl in (("A", "#1f77b4"), ("P", "#2ca02c")):
        for r in range(4):
            v = np.array(res["peak"]["12k"][s][arm][r]).reshape(-1, 2).mean(1)
            axs[1].plot(CM.ANGLES, 100 * v, "o-" if arm == "P" else "s--", ms=3, color=cl, alpha=0.4 + 0.15 * r,
                        label=f"{arm} start {r + 1}")
    axs[1].axhline(0, color="k", lw=0.8); axs[1].axvspan(20, 36, color="#ffe9a8", alpha=0.6, label="incidence range of the hard_1 highlight")
    axs[1].set_xlabel("incident polar of the angular probe (deg)"); axs[1].set_ylabel("peak height model / physical − 1 (%)")
    axs[1].set_title("hard_1 material: mirror-peak height on the independent peak grid", fontsize=10); axs[1].legend(fontsize=6.5, ncol=2)
    for k, (sc, (nm, _)) in enumerate(SCN.items()):
        q = res["squared_error"]["12k"][sc]
        axs[2].plot(x, [100 * v["total_change_of_A_SE"] for v in q], "o-", label=nm)
    axs[2].axhline(0, color="k", lw=0.8); axs[2].set_xticks(x, ["start 1", "start 2", "start 3", "start 4"])
    axs[2].set_ylabel("change of first-bounce squared error, P vs A (%)"); axs[2].legend(fontsize=8)
    axs[2].set_title("all scenes: per-start change (12k states)", fontsize=10)
    fig.suptitle("Additional descriptive analysis (not predeclared): why hard_1's image error rises while its angular error falls",
                 x=0.01, ha="left", weight="bold")
    fig.text(0.01, 0.005, "Squared error is dominated by the brightest pixels; the angular el1 metric is energy-weighted L1 and barely sees a few-percent "
                          "overshoot at the mirror peak. Starts 1–2 overshoot the peak on the angular peak grid too (20–35° probes), starts 3–4 do not.",
             fontsize=8)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.85, bottom=0.14, wspace=0.3)
    fig.savefig(out / "fig09_hard1_peak_pixels_and_peak_height_per_start.png", dpi=115)
    plt.close(fig)
    print("descriptive done", flush=True)


if __name__ == "__main__":
    main()
