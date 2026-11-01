"""WL69 Stage 4 analysis (protocol xbench69_v1 + amendment 1). Every arm is scored by the unchanged WL55 machinery: Cycles-ordered
images -> KCm.cycles_ports (amended unit) -> KCm.group_bias against the WL55 Cycles references of the same state; ReST-GI port
predictions -> group_bias directly (WL55 k55_analyze form: mean of the two K seeds, SE_total with the seed half-difference).

M1 passes among ruled group x channel tests of the ruled cameras (top, underup, underdown); M2 max |bias|; M3 per-port relative L2
over selected ports; M4 stationary-receiver signed change on floor ports of top + underdown selected in both states.
QM: lowest spp at which both realizations of CY (and D8) are >= the arm on M1 and <= on M2, M3, M4; QM_contract: all ruled pass.

    cd <repo>/experiments/fnto_k55 && python ../xbench69/x_analyze.py <root> <out_json>
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "fnto_k55"))
import k55_common as KCm  # noqa: E402
import k55_fixtures as KF  # noqa: E402

CYC = Path.home() / "rest_wl55" / "out" / "cycles_cuda"
STATES = {"T0": "floor", "T1": "floor_c4"}
RULED = ("top", "underup", "underdown")
ALLC = RULED + ("otop",)
LADDER = (1, 4, 16, 64, 256, 1024)
RECV = {"top": "floor", "underdown": "floor"}


def refs():
    R = {}
    for st, scene in STATES.items():
        sd = KF.scene_def(scene); geo = json.load(open(CYC / f"ccam_{scene}.json")); ss = KCm.classify_sceneset(sd); lab = KCm.labels_of(sd)
        R[st] = dict(sd=sd, geo=geo, ss=ss, lab=lab, cams={c: KCm.cycles_ports(ss, geo[c], [np.load(f)[..., :3] for f in sorted(CYC.glob(f"cimg_{scene}_{c}_s*.npy"))])["amended"]
                                                         for c in ALLC})
    return R


def score(R, st, preds, se_extra=None):
    """preds: {cam: (P, 3) port prediction}; returns per-cam groups + M1-M3."""
    out = dict(cams={}); ruled = []; rel = []
    for cam, P in preds.items():
        u = R[st]["cams"][cam]; g = KCm.group_bias(P, u["mean"], u["sel"], R[st]["lab"], u["se"])
        if se_extra is not None and cam in se_extra:
            for gn in g:
                so = se_extra[cam].get(gn)
                if so is not None:
                    g[gn]["se_total"] = [float(np.hypot(a, b)) for a, b in zip(g[gn]["se"], so)]
                    g[gn]["passed"] = [bool(abs(b) <= KCm.TOL + 3 * t) if rl else None for b, t, rl in zip(g[gn]["bias"], g[gn]["se_total"], g[gn]["ruled"])]
        out["cams"][cam] = g
        if cam in RULED:
            ruled += [(cam, gn, ch, v["bias"][ch], v["passed"][ch]) for gn, v in g.items() for ch in range(3) if v["ruled"][ch]]
            m = u["sel"]; rel.append(((P[m] - u["mean"][m]) ** 2).sum(0) / (u["mean"][m] ** 2).sum(0))
    out["M1"] = sum(bool(p) for *_, p in ruled); out["n_ruled"] = len(ruled)
    out["M2"] = max(abs(b) for _, _, _, b, _ in ruled) if ruled else None
    out["M3"] = np.sqrt(np.sum(rel, 0)).tolist() if rel else None
    out["fails"] = [(c, g, ch, round(100 * b, 2)) for c, g, ch, b, p in ruled if not p]
    return out


def change(R, P0, P1):
    """M4 on stationary-receiver (floor) ports of top + underdown selected in both states."""
    dP, dR = [], []
    for cam, grp in RECV.items():
        u0, u1 = R["T0"]["cams"][cam], R["T1"]["cams"][cam]; lab = R["T0"]["lab"]
        m = u0["sel"] & u1["sel"] & (lab == grp)
        dP.append(P1[cam][m] - P0[cam][m]); dR.append(u1["mean"][m] - u0["mean"][m])
    dP, dR = np.concatenate(dP), np.concatenate(dR)
    err = np.linalg.norm(dP - dR, axis=0) / np.linalg.norm(dR, axis=0)
    slope = (dP * dR).sum(0) / (dR * dR).sum(0); corr = [float(np.corrcoef(dP[:, k], dR[:, k])[0, 1]) for k in range(3)]
    return dict(n_ports=int(len(dR)), err=err.tolist(), slope=slope.tolist(), corr=corr)


def image_ports(R, st, files):
    return {cam: KCm.cycles_ports(R[st]["ss"], R[st]["geo"][cam], [np.load(f)[..., :3]])["amended"]["mean"] for cam, f in files.items()}


def main():
    root = Path(sys.argv[1]); R = refs(); res = dict(protocol="xbench69_v1", arms={})
    # ReST-GI (mean of the two K seeds; SE_oracle = half difference, WL55 form)
    for arm in ("CAK", "P10"):
        P = {}; A = {}
        for st, scene in STATES.items():
            z = [np.load(root / "rg" / f"{scene}__{arm}_s{i}.npz") for i in (1, 2)]
            seeds = [{cam: zz[f"{KCm.dirkey(R[st]['sd']['cams'][cam]['dir'])}|ball"] for cam in ALLC} for zz in z]
            P[st] = {cam: 0.5 * (seeds[0][cam] + seeds[1][cam]) for cam in ALLC}
            sc = [score(R, st, s_) for s_ in seeds]
            so = {cam: {gn: [abs(a - b) / 2 for a, b in zip(sc[0]["cams"][cam][gn]["bias"], sc[1]["cams"][cam][gn]["bias"])] for gn in sc[0]["cams"][cam]} for cam in ALLC}
            A[st] = score(R, st, P[st], so)
        A["M4"] = change(R, P["T0"], P["T1"]); res["arms"][f"RG_{arm}"] = A
    # STALE: T0 reference ports as the T1 prediction
    P0 = {cam: R["T0"]["cams"][cam]["mean"] for cam in ALLC}
    res["arms"]["STALE"] = dict(T1=score(R, "T1", P0), M4=change(R, P0, P0))
    # Ladders: CY (Cycles Full Recompute) and D8 (8DNA, per training seed tag)
    lad = {}
    for d in sorted((root / "cy").glob("spp*")):
        spp = int(d.name[3:])
        for st, scene in STATES.items():
            for r in (0, 1):
                f = {cam: next(iter(sorted(d.glob(f"cimg_{scene}_{cam}_s*.npy"))[r:r + 1]), None) for cam in ALLC}
                if all(f.values()):
                    lad.setdefault(("CY", spp, r), {})[st] = image_ports(R, st, f)
    for f in sorted((root / "d8").glob("d8_*_T0_top_spp*_r*.npy")):
        tag = f.name.split("_")[1]; spp = int(f.name.split("spp")[1].split("_")[0]); r = int(f.stem[-1])
        for st in STATES:
            fs = {cam: root / "d8" / f"d8_{tag}_{st}_{cam}_spp{spp}_r{r}.npy" for cam in ALLC}
            if all(p.exists() for p in fs.values()):
                lad.setdefault((f"D8_{tag}", spp, r), {})[st] = image_ports(R, st, fs)
    for (name, spp, r), P in lad.items():
        if len(P) < 2:
            continue
        e = res["arms"].setdefault(name, {}).setdefault(str(spp), {})
        e[str(r)] = dict(T0=score(R, "T0", P["T0"]), T1=score(R, "T1", P["T1"]), M4=change(R, P["T0"], P["T1"]))
    # quality matching
    def q(a):
        return dict(M1=min(a["T0"]["M1"], a["T1"]["M1"]), M2=max(a["T0"]["M2"], a["T1"]["M2"]), M3=float(np.max([a["T0"]["M3"], a["T1"]["M3"]])), M4=float(np.max(a["M4"]["err"])))
    res["quality"] = {}
    for name, A in res["arms"].items():
        if name.startswith("RG_"):
            res["quality"][name] = q(A)
        elif name.startswith(("CY", "D8")):
            res["quality"][name] = {spp: [q(v) for v in A[spp].values()] for spp in A}
    nr = res["arms"]["RG_CAK"]["T1"]["n_ruled"]
    for ladname in [n for n in res["quality"] if n.startswith(("CY", "D8"))]:
        L = res["quality"][ladname]; qm = {}
        qm["contract"] = next((int(s) for s in sorted(L, key=int) if len(L[s]) == 2 and all(x["M1"] == nr for x in L[s])), None)
        for arm in [n for n in res["quality"] if n.startswith("RG_")] + [n for n in res["quality"] if n.startswith("D8") and n != ladname]:
            ref = res["quality"][arm]
            refq = ref if isinstance(ref, dict) and "M1" in ref else None
            if refq is None:
                continue
            qm[arm] = next((int(s) for s in sorted(L, key=int) if len(L[s]) == 2 and all(x["M1"] >= refq["M1"] and x["M2"] <= refq["M2"] and x["M3"] <= refq["M3"]
                                                                                     and x["M4"] <= refq["M4"] for x in L[s])), None)
        res.setdefault("QM", {})[ladname] = qm
    KCm.dump(res, sys.argv[2]); print(json.dumps(res["quality"], indent=0)[:3000]); print(json.dumps(res.get("QM")))
    sys.stdout.flush(); os._exit(0)


if __name__ == "__main__":
    main()
