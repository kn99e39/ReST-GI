"""WL54 consolidated report (local, numpy + matplotlib): amended Lambert composition table with reference-validity flags, summary.json and figures.
    python c54_report.py <results/fnto_c54> <results/evaluation/54>
Reads: final/{amend1_tables, amend1_pixel_size, amendE_tables, known_field, cost, change, dnoise, profiles, rho_check, tests_final}.json, stageC/*.json, kfine/*.json, audit/*.json."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

R, E = Path(sys.argv[1]), Path(sys.argv[2]); E.mkdir(parents=True, exist_ok=True); F = R / "final"
J = lambda p: json.load(open(p))  # noqa: E731
tab, pix, tabE, kf, cost, chg, dno, prof, rho, tests = (J(F / f) for f in ("amend1_tables.json", "amend1_pixel_size.json", "amendE_tables.json", "known_field.json", "cost.json", "change.json", "dnoise.json", "profiles.json", "rho_check.json", "tests_final.json"))
C = {n: J(R / "stageC" / f"{n}_stageC.json") for n in ("gap4", "gap8", "gap30", "floor_lam", "lblock_lam", "rot_lam")}
KC = J(R / "kfine" / "kfine_control.json"); WL53_ZERO_SIGNAL = {("gap4", "obot", "slab_hi:-y"), ("gap4", "obot", "slab_lo:-y")}
# pixel-converged reference (0.25 mm, amendment 1) replaces the 0.5 mm one for these two gap4 groups
CONV = {("gap4", "gapdown", "slab_lo:top"): pix["gapdown@0.25"], ("gap4", "gapup", "slab_hi:bot"): pix["gapup@0.25"]}
TOL = 0.03
COL = dict(C0="#6b6b6b", C1="#1f5fa8", F="#111111", U="#d98200", G="#8a4fb3")


def lambert_table():
    rows = []
    for fx in tab:
        for cam, rec in tab[fx].items():
            for tag in ("amended", "partial"):
                if tag not in rec:
                    continue
                for g, v in rec[tag]["C0prod"].items():
                    if tag == "amended" and not v["ruled"]:
                        continue
                    if tag == "partial" and v["n"] < 5:
                        continue
                    f16 = rec[tag]["F16"][g]; c1 = rec[tag]["C1"][g]; se = v["se"]
                    row = dict(fixture=fx, cam=cam, group=g, kind=tag, n=v["n"], share=v["share"], se=se, C0=v["bias"], C1=c1["bias"], F16=f16["bias"], G16=rec[tag]["G16"][g]["bias"], ref="Cycles 0.5 mm")
                    if (fx, cam, g) in CONV:
                        p = CONV[(fx, cam, g)]; row.update(C0=p["C0prod"], C1=p["C1"], F16=p["F16"], G16=p["G16"], ref="Cycles 0.25 mm (pixel-converged)")
                    row["zero_signal"] = bool((fx, cam, g) in WL53_ZERO_SIGNAL and abs(row["C0"] + 1) < 1e-6) or (abs(row["C0"] + 1) < 1e-6 and abs(row["C1"] + 1) < 1e-6)
                    row["ref_validated"] = bool(abs(row["F16"]) <= TOL + 3 * se)
                    row["C0_pass_vs_ref"] = bool(abs(row["C0"]) <= TOL + 3 * se); row["C1_pass_vs_ref"] = bool(abs(row["C1"]) <= TOL + 3 * se)
                    row["C1_pass_vs_F16"] = bool(abs(row["C1"] - row["F16"]) <= TOL); row["C0_pass_vs_F16"] = bool(abs(row["C0"] - row["F16"]) <= TOL)
                    rows.append(row)
    return rows


rows = lambert_table(); amended = [r for r in rows if r["kind"] == "amended" and not r["zero_signal"]]
signif = [r for r in amended if r["ref_validated"]]
summary = dict(
    lambert=dict(ruled_groups=len(amended), reference_validated=len(signif), C0_fail_validated=sum(not r["C0_pass_vs_ref"] for r in signif), C1_fail_validated=sum(not r["C1_pass_vs_ref"] for r in signif),
                 unvalidated=[(r["fixture"], r["cam"], r["group"], round(r["F16"], 4), round(r["C0"], 4), round(r["C1"], 4)) for r in amended if not r["ref_validated"]],
                 C1_fail_validated_groups=[(r["fixture"], r["cam"], r["group"], round(r["C1"], 4)) for r in signif if not r["C1_pass_vs_ref"]],
                 C0_fail_validated_groups=[(r["fixture"], r["cam"], r["group"], round(r["C0"], 4)) for r in signif if not r["C0_pass_vs_ref"]],
                 partial=[(r["fixture"], r["cam"], r["group"], r["n"], round(r["C0"], 3), round(r["C1"], 3), round(r["F16"], 3)) for r in rows if r["kind"] == "partial" and abs(r["C0"]) > TOL]),
    worst_C0=max(((abs(r["C0"]), r["fixture"], r["cam"], r["group"]) for r in signif)), worst_C1=max(((abs(r["C1"]), r["fixture"], r["cam"], r["group"]) for r in signif)))
# C1 vs F16 over validated + unvalidated ruled groups (judged against the fine reference where Cycles is not validated)
summary["lambert"]["C1_vs_F16_max_abs"] = max(abs(r["C1"] - r["F16"]) for r in amended)
summary["lambert"]["C0_vs_F16_max_abs"] = max(abs(r["C0"] - r["F16"]) for r in amended)
json.dump(dict(rows=rows, summary=summary), open(F / "lambert_table.json", "w"), indent=1)

# --------------------------------------------------------------------------- figures
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
cases = [("floor_lam", "underup", "dom:bot", "floor_lam\nbox underside"), ("rot_lam", "underup", "dom:bot", "rot_lam\nbox underside"), ("lblock_lam", "underup", "dom:bot", "lblock_lam\nbox underside"),
         ("gap4", "gapup", "slab_hi:bot", "gap4\nupper bottom"), ("gap4", "gapdown", "slab_lo:top", "gap4\nlower top"), ("gap8", "gapup", "slab_hi:bot", "gap8\nupper bottom"), ("gap8", "gapdown", "slab_lo:top", "gap8\nlower top"), ("gap30", "gapup", "slab_hi:bot", "gap30\nupper bottom")]
look = {(r["fixture"], r["cam"], r["group"]): r for r in rows if r["kind"] == "amended"}
fig, ax = plt.subplots(1, 1, figsize=(9.2, 3.8)); w = 0.26; x = np.arange(len(cases))
for k, (lab, key, c) in enumerate((("C0 production", "C0", COL["C0"]), ("C1 = CASP", "C1", COL["C1"]), ("exact Galerkin coarse (A)", "G16", COL["G"]))):
    vals = [100 * (look[(a, b, g)][key] - look[(a, b, g)]["F16"]) for a, b, g, _ in cases]; ax.bar(x + (k - 1) * w, vals, w, label=lab, color=c)
ax.axhline(3, color="k", lw=0.7, ls="--"); ax.axhline(-3, color="k", lw=0.7, ls="--"); ax.set_xticks(x); ax.set_xticklabels([c[3] for c in cases], fontsize=7.5); ax.set_ylabel("pooled bias vs fine reference F16 (%)")
ax.set_title("Lambert, exact K: error of the coarse interface and of CASP against the converged fine solution (dashed: 3 %)", fontsize=9); ax.legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig(E / "01_lambert_interface_error.png", dpi=160); plt.close(fig)

fig, ax = plt.subplots(1, 1, figsize=(6.4, 3.8)); p = prof["gap4_profile"]; xx = np.array(p["x"]) * 100 + 5
for k, lab, c, ls in (("F32", "fine solution (s = 32)", COL["F"], "-"), ("G16", "1 cm interface, exact transfer (Galerkin)", COL["G"], "--"), ("C0prod", "C0 production", COL["C0"], ":"), ("C1", "CASP", COL["C1"], "-")):
    ax.semilogy(xx, np.array(p[k]) / p["F32"][0], ls, color=c, marker="o", ms=3, label=lab)
ax.set_xlabel("distance from the open (sunward) edge, 1 cm port centres (cm)"); ax.set_ylabel("port radiance / first port of the fine solution"); ax.set_title("gap4 lower-slab top face: decay into the gap", fontsize=9); ax.legend(frameon=False, fontsize=7.5)
fig.tight_layout(); fig.savefig(E / "02_gap4_decay_profile.png", dpi=160); plt.close(fig)

fig, axs = plt.subplots(1, 4, figsize=(11.5, 3.3), sharey=True); sel = ["gap4:slab_hi:bot", "gap4:slab_lo:top", "floor_lam:dom:bot", "gap8:slab_hi:bot"]
for a, k in zip(axs, sel):
    e = prof["pareto"][k]; fl = 0.1
    pts = [(e["coarse_C0inf"]["cells"], e["coarse_C0inf"]["bias"], "C0 interface (M→∞)", COL["C0"], "s")] + [(v["cells"], v["bias"], "uniform s=2,4,8" if s == "2" else None, COL["U"], "o") for s, v in e["uniform"].items()]
    for cells, b, lab, c, m in pts:
        a.loglog(cells, max(abs(100 * b), fl), m, color=c, label=lab, ms=6)
    a.loglog([v["cells"] for v in e["uniform"].values()], [max(abs(100 * v["bias"]), fl) for v in e["uniform"].values()], "-", color=COL["U"], lw=0.8)
    for kk, lab in (("C1", "CASP c=2 (primary)"), ("C1_c1", "CASP c=1"), ("C1_c4", "CASP c=4"), ("C1_b4", "CASP, 4× rays")):
        v = e["casp"][kk]; a.loglog(v["cells"], max(abs(100 * v["bias"]), fl), "D" if kk == "C1" else "^", color=COL["C1"], mfc=(COL["C1"] if kk == "C1" else "none"), label=lab if k == sel[0] else None, ms=6)
    a.axhline(3, color="k", lw=0.7, ls="--"); a.set_title(k.replace(":", "  "), fontsize=9); a.set_xlabel("unknowns (cells)")
axs[0].set_ylabel("|pooled bias vs F16| (%), floor 0.1"); axs[0].legend(frameon=False, fontsize=6.5); fig.suptitle("Accuracy against unknowns: uniform refinement vs CASP (equal ray budget)", fontsize=9); fig.tight_layout(); fig.savefig(E / "03_accuracy_vs_unknowns.png", dpi=160); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(10, 3.4)); res = cost["results"]; byt = {(r["tag"], r["arm"]): r for r in res}
for arm, c, lab in (("C0", COL["C0"], "C0"), ("CASP", COL["C1"], "CASP"), ("uniform4", COL["U"], "uniform s=4")):
    xs_ = [byt[(t, arm)]["coarse_ports"] for t in ("floor10", "floor20", "floor30", "floor60")]; axs[0].loglog(xs_, [byt[(t, arm)]["median"]["total"] * 1e3 for t in ("floor10", "floor20", "floor30", "floor60")], "o-", color=c, label=lab)
axs[0].set_xlabel("coarse ports (floor edge 10 / 20 / 30 / 60 cm)"); axs[0].set_ylabel("geometry update + 20 iterations, GPU (ms)"); axs[0].legend(frameon=False); axs[0].set_title("scaling with port count", fontsize=9)
gaps = ("gap4", "gap8", "gap30"); xw = np.arange(3)
for k, (arm, c) in enumerate((("C0", COL["C0"]), ("CASP", COL["C1"]), ("uniform4", COL["U"]))):
    axs[1].bar(xw + (k - 1) * 0.26, [byt[(g, arm)]["median"]["total"] * 1e3 for g in gaps], 0.26, color=c)
axs[1].set_xticks(xw); axs[1].set_xticklabels(["gap 4 mm", "gap 8 mm", "gap 30 mm"]); axs[1].set_ylabel("ms"); axs[1].set_title("near-contact density (480 coarse ports)", fontsize=9)
fig.tight_layout(); fig.savefig(E / "04_cost_scaling.png", dpi=160); plt.close(fig)

fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.2), sharey=True)
for a, ch, nm in zip(axs, "012", "RGB"):
    labs, v1, v2 = [], [], []
    for sc, cam, g in (("gap4", "gapdown", "slab_lo:top"), ("gap4", "gapup", "slab_hi:bot"), ("gap4", "bot", "slab_lo:bot"), ("gap8", "gapdown", "slab_lo:top"), ("gap8", "gapup", "slab_hi:bot"), ("gap8", "bot", "slab_lo:bot")):
        rc = KC[sc]["cams"][cam]; labs.append(f"{sc}\n{cam}"); v1.append(100 * rc["K1_amended"][ch][g]["bias"]); v2.append(100 * rc["K2_amended"][ch][g]["bias"])
    x = np.arange(len(labs)); a.bar(x - 0.2, v1, 0.4, color=COL["C0"], label="K and C at 1 cm"); a.bar(x + 0.2, v2, 0.4, color=COL["C1"], label="K and C at 5 mm (control)"); a.axhline(3, color="k", lw=0.7, ls="--"); a.axhline(-3, color="k", lw=0.7, ls="--")
    a.set_xticks(x); a.set_xticklabels(labs, fontsize=6.5); a.set_title(f"{nm} channel", fontsize=9)
axs[0].set_ylabel("pooled bias vs Cycles volume (%)"); axs[0].legend(frameon=False, fontsize=7); fig.suptitle("Matched-resolution K control (labelled control, not a candidate): medium-to-medium gap faces", fontsize=9); fig.tight_layout(); fig.savefig(E / "05_k_resolution_control.png", dpi=160); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.4), sharey=True)
for a, sc in zip(axs, ("bend90", "dent")):
    labs, D1, N, D0 = [], [], [], []
    for cam, g, ncam in (("top", "dom:top", "top"), ("otop", "dom:top", "otop"), ("underup", "dom:bot", "bot"), ("underdown", "floor", "top")):
        rec = tabE[sc].get(cam, {}).get("differential")
        if rec is None or g not in rec["C1"]["0"]:
            continue
        for ch in "012":
            labs.append(f"{cam}\n{g}\n{'RGB'[int(ch)]}"); D1.append(100 * rec["C1"][ch][g]["D"]); D0.append(100 * rec["C0rebuilt"][ch][g]["D"]); N.append(100 * abs(dno[sc].get(ncam, {}).get(g, {"noise": [0, 0, 0]})["noise"][int(ch)]))
    x = np.arange(len(labs)); a.bar(x - 0.19, D0, 0.38, color=COL["C0"], label="C0"); a.bar(x + 0.19, D1, 0.38, color=COL["C1"], label="C1 = CASP", yerr=N, error_kw=dict(ecolor="k", lw=0.8))
    a.axhline(0, color="k", lw=0.5); a.set_xticks(x); a.set_xticklabels(labs, fontsize=5); a.set_title(f"{sc}: stale - current K (error bars: K-oracle seed noise)", fontsize=8.5)
axs[0].set_ylabel("paired differential D (% of group signal)"); axs[0].legend(frameon=False, fontsize=7); fig.tight_layout(); fig.savefig(E / "06_stale_vs_current_differential.png", dpi=160); plt.close(fig)

summary.update(known_field={k: dict(solution_effect=v["fields"]["solution"]["rel_effect"], by_distance=v["fields"]["solution"]["by_distance_rel_to_total"], synthetic={f: v["fields"][f]["rel_effect"] for f in ("uniform", "ramp", "exp_edge_5mm", "step_in_port")}) for k, v in kf.items()},
               tests=dict(passed=tests["passed"], n=tests["n"]), rho_check=rho)
json.dump(summary, open(F / "summary.json", "w"), indent=1)
print(json.dumps(summary["lambert"], indent=1)[:4000])
