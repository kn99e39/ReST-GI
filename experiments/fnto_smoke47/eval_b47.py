"""WL47 Stage B evaluation (numpy): MIR transfer matrices, paired-field metrics, superposition; SSS band metrics.

    python eval_b47.py <results/fnto_smoke47>
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import smoke47_core as K  # noqa: E402

PM, PS, CR = K.PM, K.PROTO["stage_b_sss"], K.PROTO["stage_b_criteria"]
ND = len(K.MIR_T)


def vec(v):
    """[8 cams, 4 ports] -> port-major output vector (q * 8 + k)."""
    return np.asarray(v, float).T.reshape(-1)


def classify(D, noise, dE, abs_floor=0.10):
    if dE > 0.03:
        return "CONFOUNDED"
    bound = 3 * noise + dE
    if D >= abs_floor and D > bound:
        return "SENSITIVE"
    if D <= bound:
        return "INSENSITIVE" if bound <= 0.05 else "INCONCLUSIVE"
    return "INCONCLUSIVE"


def mir(root):
    b = [np.load(root / "mir" / f"mir_basis_rep{r}.npz") for r in (0, 1)]
    Ts = []
    for d in b:
        V = d["vals"]; T = np.zeros((4 * ND, 2 * ND))
        for p in range(2):
            for i in range(ND):
                T[:, p * ND + i] = vec(V[p, i]) / PM["basis_E"]
        Ts.append(T)
    np.savez(root / "mir" / "mir_T.npz", T1=Ts[0], T2=Ts[1])
    F = np.load(root / "mir" / "mir_fields.npz"); irr = json.loads((root / "mir" / "mir_fields_irr.json").read_text())
    rec = {"T_rel_noise_per_set": K.rel(Ts[1], Ts[0]) / math.sqrt(2), "T_shape": list(Ts[0].shape),
           "T_column_norms": np.linalg.norm(Ts[0], axis=0).round(5).tolist(), "basis_wall_s": [float(d["wall_s"]) for d in b]}
    for pn, pr in PM["pairs"].items():
        Y = {f"{s}{r}": vec(F[f"{pn}_{s}_rep{r}_vals"]) for s in "AB" for r in (0, 1)}
        EA, EB = np.array(irr[f"{pn}_A"]["E_ports"]), np.array(irr[f"{pn}_B"]["E_ports"]); op = pr["open"]
        mean_ = [p for p in op if EA[p] >= 0.1 * EA[op].max()]
        dE = float(max(abs(EA[p] - EB[p]) / EA[p] for p in mean_))
        den = max(np.linalg.norm(Y["A0"]), np.linalg.norm(Y["B0"]))
        D = float(np.linalg.norm(Y["A0"] - Y["B0"]) / den)
        noise = float(math.sqrt((np.linalg.norm(Y["A0"] - Y["A1"]) ** 2 + np.linalg.norm(Y["B0"] - Y["B1"]) ** 2) / 2) / den)
        nlp = [q for q in range(4) if q not in op]; sel = np.concatenate([np.arange(q * ND, (q + 1) * ND) for q in nlp])
        dnl = max(np.linalg.norm(Y["A0"][sel]), np.linalg.norm(Y["B0"][sel]))
        Dnl = float(np.linalg.norm(Y["A0"][sel] - Y["B0"][sel]) / dnl) if dnl > 0 else float("nan")
        nnl = float(math.sqrt((np.linalg.norm((Y["A0"] - Y["A1"])[sel]) ** 2 + np.linalg.norm((Y["B0"] - Y["B1"])[sel]) ** 2) / 2) / dnl) if dnl > 0 else float("nan")
        sup = {}
        for s in "AB":
            pred = np.zeros(4 * ND)
            for p in op:
                for t, E in pr[s]:
                    pred += E * Ts[0][:, p * ND + int(np.argmin(np.abs(K.MIR_T - t)))]
            rn = float(np.linalg.norm(Y[f"{s}0"] - Y[f"{s}1"]) / np.linalg.norm(Y[f"{s}0"]))
            err = K.rel(pred, Y[f"{s}0"])
            sup[s] = {"rel_err": err, "replicate_noise": rn, "pass": bool(err <= 3 * rn + 0.02)}
        rec[pn] = {
            "E_A": EA.tolist(), "E_B": EB.tolist(), "delta_E": dE, "D_rel": D, "noise_D": noise, "bound": 3 * noise + dE,
            "class": classify(D, noise, dE), "nonlocal_ports": nlp, "D_rel_nonlocal": Dnl, "noise_D_nonlocal": nnl,
            "class_nonlocal": classify(Dnl, nnl, dE),
            "port_totals_A": np.asarray(F[f"{pn}_A_rep0_vals"]).sum(0).round(5).tolist(),
            "port_totals_B": np.asarray(F[f"{pn}_B_rep0_vals"]).sum(0).round(5).tolist(),
            "dir_totals_A": np.asarray(F[f"{pn}_A_rep0_vals"]).sum(1).round(5).tolist(),
            "dir_totals_B": np.asarray(F[f"{pn}_B_rep0_vals"]).sum(1).round(5).tolist(),
            "superposition": sup,
        }
    rec["fields_wall_s"] = irr["wall_s"]
    return rec


def sss(root):
    d = np.load(root / "sss" / "sss.npz"); meta = json.loads((root / "sss" / "sss_meta.json").read_text())
    xt = d["x_top"].mean(0); lit = (xt >= -0.10) & (xt <= -0.02)
    E = {f: math.pi * d[f"{f}_irr_top"].mean(0)[lit].mean(0) for f in "AB"}             # RGB irradiance on the lit side
    dE_rgb = np.abs(E["A"] - E["B"]) / E["A"]; dE = float(dE_rgb.max())
    rec = {"meta": meta, "E_lit_A": E["A"].tolist(), "E_lit_B": E["B"].tolist(), "delta_E_rgb": dE_rgb.tolist(), "bands": {}}
    for cam in PS["cameras"]:
        x = d[f"x_{cam}"].mean(0)
        prof = {k: d[f"{k}_{cam}"].mean(0) for k in ("A_rep0", "A_rep1", "B_rep0", "B_rep1")}   # [64, 3]
        for lo, hi in PS["bands_m"] + [[-0.10, -0.02]]:
            m = (x >= lo) & (x < hi); v = {k: p[m].mean(0) for k, p in prof.items()}
            den = np.maximum(v["A_rep0"], v["B_rep0"])
            D = np.abs(v["A_rep0"] - v["B_rep0"]) / den
            nz = np.sqrt(((v["A_rep0"] - v["A_rep1"]) ** 2 + (v["B_rep0"] - v["B_rep1"]) ** 2) / 2) / den
            key = f"{cam}_{lo:+.2f}_{hi:+.2f}"
            rec["bands"][key] = {"n_px": int(m.sum()), "A": v["A_rep0"].tolist(), "B": v["B_rep0"].tolist(),
                                 "D_rel_rgb": D.tolist(), "noise_D_rgb": nz.tolist(), "bound_rgb": (3 * nz + dE).tolist(),
                                 "signed_B_over_A_rgb": (v["B_rep0"] / v["A_rep0"]).tolist(),
                                 "class_rgb": [classify(float(D[c]), float(nz[c]), dE) for c in range(3)],
                                 "role": "local lit side (not evidence)" if lo < 0 else "nonlocal shadow band"}
    return rec


if __name__ == "__main__":
    root = Path(sys.argv[1]); out = {"mir": mir(root), "sss": sss(root)}
    (root / "stage_b.json").write_text(json.dumps(out, indent=1))
    m = out["mir"]; print("T noise", m["T_rel_noise_per_set"])
    for pn in PM["pairs"]:
        r = m[pn]; print(pn, {k: r[k] for k in ("delta_E", "D_rel", "noise_D", "class", "D_rel_nonlocal", "class_nonlocal")}, r["superposition"])
    print("SSS dE", out["sss"]["delta_E_rgb"])
    for k, v in out["sss"]["bands"].items():
        print(k, np.round(v["D_rel_rgb"], 4), np.round(v["bound_rgb"], 4), v["class_rgb"])
