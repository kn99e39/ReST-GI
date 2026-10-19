"""Worklog 37 smoke, POST-HOC analysis (not predeclared; user question 2026-10-19): split the frozen model's error on physically
changed A surfaces into (a) same visibility as T0 = in-distribution inputs with a changed target, (b) visibility changed
(off-support head), (c) A pixels not corresponded to the same T0 view (includes surfaces never seen at T0).
Writes results/evaluation/37/smoke_meeting/ood_split_posthoc.json."""
import numpy as np, json
z = np.load("/mnt/c/Projects/NeuralShaderDemo/results/rna_selective_reconstruction_lego/smoke/eval_smoke2/views.npz")
arms = {"frozen": "SO_T0-PU", "selective": "B_sel_smoke-PU", "full": "A_full_smoke-PU"}
acc = {}
for v in range(16):
    k = f"heldout_T1_v{v}"; ref = z[k + "_ref"].astype(np.float32); own = z[k + "_own"]
    ch, vc, unc = z[k + "_reg_A_changed"], z[k + "_reg_A_vis_changed"], z[k + "_reg_A_uncorresponded"]
    regs = {"(a) changed, same visibility (in-distribution input)": ch & ~vc, "(b) visibility changed (new shadow / unshadow)": vc,
            "(c) A surface not visible at T0 (old box footprint etc.)": unc}
    for r, m in regs.items():
        for a, n in arms.items():
            e = ((z[k + "_" + n].astype(np.float32) - ref) ** 2).mean(-1)[m]
            s = acc.setdefault(r, {}).setdefault(a, [0.0, 0])
            s[0] += float(e.sum()); s[1] += int(m.sum())
out = {}
tot = sum(acc[r]["frozen"][0] for r in acc)
for r, d in acc.items():
    out[r] = {a: round(d[a][0] / max(d[a][1], 1), 6) for a in d}; out[r]["px"] = d["frozen"][1]
    out[r]["share_of_frozen_error_energy_in_these_three"] = round(d["frozen"][0] / tot, 3)
print(json.dumps(out, indent=1))
open("/mnt/c/Projects/NeuralShaderDemo/results/evaluation/37/smoke_meeting/ood_split_posthoc.json", "w").write(json.dumps({"post_hoc": True, "source": "results/rna_selective_reconstruction_lego/smoke/eval_smoke2/views.npz", "metric": "display MSE, PU, 16 held-out T1 views", "regions": out}, indent=1))
