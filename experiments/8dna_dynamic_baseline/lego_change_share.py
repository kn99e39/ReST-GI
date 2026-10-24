"""Share of the ReST-GI Lego T0 -> T1 stationary-receiver change carried by first-event transport (worklog 43; numpy only).

Existing Cycles data only (no render): the historical held-out targets (B4 'hist' regime, mean of two 2048-spp
realisations, clamped d20/i5; via results/fnto_lego/eval_px/heldout_T{0,1}.npz) and the worklog-41 first-event references
(results/fnto_lego/firstevent/heldout_T{0,1}_first.npz: one intrinsic event, 1024 spp, clamp d20), with the B4 regions
(results/rna_selective_reconstruction_lego/eval/views.npz). Same cameras and suns in T0 and T1, so pixels pair by index.

For stationary Lego pixels (Lego in both states; box pixels and 'A_uncorresponded' excluded):
  dG_full  = target_T1 - target_T0          dG_first = first_T1 - first_T0
  share    = <dG_first, dG_full> / <dG_full, dG_full>   (linear RGB, pooled over views)
reported overall, per material, per region and lit/shadow. Interpretation for an 8DNA Lego asset with an explicit box
inside its AABB (code case b): first-event changes on non-SSS materials are first-vertex events, which 8DNA evaluates with
the real BSDF and current visibility; Cycles' first event on Yellow includes Burley SSS of direct light, which 8DNA would
learn (>= 2 events) and connect to the sun from its AABB exit, where the box is not seen. This is a descriptive bound,
regimes as labelled; it is not an 8DNA measurement.

    python -I lego_change_share.py <eval_px dir> <firstevent dir> <B4 views.npz> <out.json>
"""

import json
import sys

import numpy as np

MATS = ["Yellow", "Gray", "transparent", "Black", "Red_Glass", "Light_brown", "Sand", "Brown", "box"]
REGS = ["A_changed", "A_weak", "A_unchanged"]


def main(px_dir, first_dir, views_path, out_path):
    P0 = np.load(f"{px_dir}/heldout_T0.npz")
    P1 = np.load(f"{px_dir}/heldout_T1.npz")
    F0 = np.load(f"{first_dir}/heldout_T0_first.npz")
    F1 = np.load(f"{first_dir}/heldout_T1_first.npz")
    Z = np.load(views_path)
    nv = int(P1["nv"])
    acc = {}

    def add(key, dfu, dfi):
        a = acc.setdefault(key, [0.0, 0.0, 0.0, 0])
        a[0] += float(np.sum(dfi * dfu))
        a[1] += float(np.sum(dfu * dfu))
        a[2] += float(np.sum(dfi * dfi))
        a[3] += len(dfu)

    for v in range(nv):
        m1, m0 = P1[f"v{v}_mk"], P0[f"v{v}_mk"]
        f1i, f0i = F1[f"v{v}_mk"], F0[f"v{v}_mk"]
        common = np.intersect1d(np.intersect1d(m1, m0), np.intersect1d(f1i, f0i))
        i1, i0 = np.searchsorted(m1, common), np.searchsorted(m0, common)
        j1, j0 = np.searchsorted(f1i, common), np.searchsorted(f0i, common)
        lego = (P1[f"v{v}_dom"][i1] == 0) & (P0[f"v{v}_dom"][i0] == 0)
        dfu = (P1[f"v{v}_tgt"][i1] - P0[f"v{v}_tgt"][i0]).astype(np.float64)
        dfi = (F1[f"v{v}_rgb"][j1] - F0[f"v{v}_rgb"][j0]).astype(np.float64)
        mat = P1[f"v{v}_mat"][i1]
        lit = P1[f"v{v}_vis"][i1].astype(bool)
        reg = {r: Z[f"heldout_T1_v{v}_reg_{r}"].ravel()[common] for r in REGS + ["A_uncorresponded", "A_vis_changed"]}
        keep = lego & ~reg["A_uncorresponded"].astype(bool)
        sel_sets = {"all_stationary": keep}
        for r in REGS + ["A_vis_changed"]:
            sel_sets[r] = keep & reg[r].astype(bool)
        for name, s in sel_sets.items():
            add(name, dfu[s], dfi[s])
            for mi in np.unique(mat[s]):
                add(f"{name}/{MATS[mi]}", dfu[s & (mat == mi)], dfi[s & (mat == mi)])
            add(f"{name}/lit_T1", dfu[s & lit], dfi[s & lit])
            add(f"{name}/shadow_T1", dfu[s & ~lit], dfi[s & ~lit])
    res = {}
    for k, (fu_fi, fu2, fi2, n) in sorted(acc.items()):
        res[k] = {"pixels": n, "first_event_share": None if fu2 == 0 else fu_fi / fu2,
                  "rms_ratio_first_over_full": None if fu2 == 0 else float(np.sqrt(fi2 / fu2)),
                  "change_energy": fu2}
    tot = res["A_changed"]["change_energy"]
    for k in res:
        if k.startswith("A_changed/"):
            res[k]["fraction_of_A_changed_energy"] = res[k]["change_energy"] / tot
    out = {"regime": "Cycles; full = B4 hist targets (clamp d20/i5, 2x2048 spp); first = WL41 first-event refs (1 intrinsic event, "
                     "clamp d20, 1024 spp); PC-rendered (both)", "views": nv, "results": res}
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1)
    for k in ("all_stationary", "A_changed", "A_vis_changed", "A_changed/Sand", "A_changed/Yellow", "A_changed/lit_T1", "A_changed/shadow_T1"):
        if k in res:
            print(k, res[k])


if __name__ == "__main__":
    main(*sys.argv[1:5])
