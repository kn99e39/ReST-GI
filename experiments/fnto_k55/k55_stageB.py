"""WL55 Stage B driver (protocol fnto_k55, stage_B): operators, shared sun terms, matched composition arms, COR, known-field arms,
outgoing test. Outputs under $WL55 (default ~/rest_wl55): ops/*.pt, out/stageB/<scene>/... (npz predictions + json meta).

    P53_MI_VARIANT=cuda_ad_rgb python k55_stageB.py ops <name,...|all>
    ... sun <scene>            ... compose <scene> <arm,...|all>        ... cor <scene> <seed,...>
    ... kf <kf scene> <arm,...|all>                                     ... btest <gap4|gap8>
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

import k55_common as KCm
import k55_compose as KP
import k55_cor as COR
import k55_fixtures as KF
import k55_ops as KO
import p53_fixtures as FX
import p53_compose as CP

DEV = "cuda"
ROOT = Path(os.environ.get("WL55", Path.home() / "rest_wl55")); OPS = ROOT / "ops"; OUT = ROOT / "out" / "stageB"
WL53 = Path.home() / "rest_wl53" / "results"
CAM3 = ("top", "bot", "otop")
OPDEF = {   # name: (geometry, level, seed, cams, Din)
    "slab_L4_s1": ("slab", 4, 5561, KCm.CAM5, 16), "slab_L4_s2": ("slab", 4, 5562, KCm.CAM5, 16),
    "slab_L1d_s1": ("slab", 1, 5541, KCm.CAM5, 16), "slab_L1d_s2": ("slab", 1, 5542, KCm.CAM5, 16),
    "slab_L2d_s1": ("slab", 2, 5551, KCm.CAM5, 16), "slab_L1d_din64": ("slab", 1, 5543, KCm.CAM5, 64),
    "bend90_L4_s1": ("bend90", 4, 5563, CAM3, 16), "bend90_L4_s2": ("bend90", 4, 5564, CAM3, 16),
    "dent_L4_s1": ("dent", 4, 5565, CAM3, 16), "dent_L4_s2": ("dent", 4, 5566, CAM3, 16),
}
SCENE_CAMS = {"gap4": ("gapdown", "gapup", "obot", "obotx", "bot", "top", "otop"), "gap8": ("gapdown", "gapup", "obot", "obotx", "bot", "top", "otop"),
              "gap6s": ("gapdown", "gapup", "obot", "obotx", "bot"), "floor": ("top", "otop", "underup", "underdown"),
              "bend90": ("top", "otop", "underup", "underdown"), "dent": ("top", "otop", "underup", "underdown"), "floor_c4": ("top", "otop", "underup", "underdown")}
SCENE_K = {"gap4": "slab", "gap8": "slab", "gap6s": "slab", "floor": "slab", "floor_c4": "slab", "bend90": "bend90", "dent": "dent"}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def cam_dirs_of(sd, cams):
    """camera id -> direction key (FX.DIRS) for a scene."""
    return [KCm.dirkey(sd["cams"][c]["dir"]) for c in cams]


# ------------------------------------------------------------------ operators


def st_ops(names):
    OPS.mkdir(parents=True, exist_ok=True)
    for name in (OPDEF if names == ["all"] else names):
        p = OPS / f"{name}.pt"
        if p.exists():
            log("skip", name); continue
        geo, lev, seed, cams, din = OPDEF[name]; m, V = FX.domain_geometry(geo)
        fm, VC, par = KO.refined(m, V, lev); dom = KO.DomainX(fm, VC, name=name)
        dirs = np.array([FX.DIRS[k] for k in cams])
        r = KO.tr_sparse(dom, 64, din, 2048, seed, dirs, par, len(m["port_area"]), log=log, s2=(lev <= 2))
        r.update(geo=geo, level=lev, cam_keys=list(cams), parent=par, cell_area=dom.port_area.double().cpu().numpy())
        KO.save_op(r, p); log(name, "done", f"{r['seconds']:.0f}s nnz {len(r['idx'])} peak {r['peak_gpu_gb']:.1f} GB", json.dumps({k: r["stats"][k] for k in ("photons", "leaks", "wrong_side", "truncated")}))
        del r, dom; torch.cuda.empty_cache()


def load_op(name):
    return KO.load_op(OPS / f"{name}.pt")


# ------------------------------------------------------------------ sun


def sun_path(scene):
    return OUT / scene / "sun.pt"


def st_sun(scene):
    (OUT / scene).mkdir(parents=True, exist_ok=True); sd = KF.scene_def(scene)
    keys = sorted(set(cam_dirs_of(sd, SCENE_CAMS.get(scene, list(sd["cams"])))))
    s = KP.sun_term(sd, keys); torch.save(s, sun_path(scene)); log("sun", scene, {ci: v["seconds"] for ci, v in s.items()})


# ------------------------------------------------------------------ composition arms


def uniform_levels(sd, l):
    return [np.full(len(m["port_area"]), l) for n, k, m, V in sd["comps"]]


def cak_levels(sd, c=2.0, lmax=4, seed=5801):
    """stage_C.level_rule: the CASP geometry-only rule on the current geometry of all components, l_max = 4."""
    import c54_casp as CA
    lev, nu, d = CA.assign_levels(KCm.classify_sceneset(sd), c=c, lmax=lmax, seed=seed); out, off = [], 0
    for n, k, m, V in sd["comps"]:
        P = len(m["port_area"]); out.append(lev[off:off + P].astype(int)); off += P
    return out


def arms_for(scene):
    k = SCENE_K[scene]
    A = {"P10_s1": (1, f"{k}_L4_s1", None), "P10_s2": (1, f"{k}_L4_s2", None), "P05_s1": (2, f"{k}_L4_s1", None), "P05_s2": (2, f"{k}_L4_s2", None),
         "P025_s1": (4, f"{k}_L4_s1", None), "P025_s2": (4, f"{k}_L4_s2", None), "P10_c2": (1, f"{k}_L4_s1", dict(seed_C=5582))}
    A.update({"CAK_s1": ("cak", f"{k}_L4_s1", None), "CAK_s2": ("cak", f"{k}_L4_s2", None)})     # stage_C candidate (run only after gate_C)
    if k in ("bend90", "dent"):                                                                     # stage_E: stale rest K_fine reused on the deformed geometry
        A.update({"P10st_s1": (1, "slab_L4_s1", None), "P10st_s2": (1, "slab_L4_s2", None), "CAKst_s1": ("cak", "slab_L4_s1", None), "CAKst_s2": ("cak", "slab_L4_s2", None),
                  "P025st_s1": (4, "slab_L4_s1", None), "P025st_s2": (4, "slab_L4_s2", None)})
    if k == "slab":
        A.update({"P10d_s1": (1, "slab_L1d_s1", None), "P10d_s2": (1, "slab_L1d_s2", None), "P05d_s1": (2, "slab_L2d_s1", None),
                  "P10_din64": (1, "slab_L1d_din64", dict(din=64))})
    return A


def st_compose(scene, arms):
    od = OUT / scene; od.mkdir(parents=True, exist_ok=True); sd = KF.scene_def(scene); sun = torch.load(sun_path(scene))
    cams = SCENE_CAMS[scene]; keys = sorted(set(cam_dirs_of(sd, cams))); A = arms_for(scene); cache = {}
    for arm in (A if arms == ["all"] else arms):
        if (od / f"{arm}.npz").exists():
            log("skip", scene, arm); continue
        lev, opname, kw = A[arm]; kw = kw or {}
        if arms == ["all"] and (arm.startswith("CAK") or "st_" in arm):
            continue                                         # the candidate runs only when named explicitly (after gate_C)
        if opname not in cache:
            cache.clear(); torch.cuda.empty_cache(); cache[opname] = load_op(opname)
        Kf = cache[opname]
        levels = cak_levels(sd) if lev == "cak" else uniform_levels(sd, lev); Kops = {}; reuse = {}
        direct = (lev != "cak") and Kf["level"] == lev
        for ci, (n, k, m, V) in enumerate(sd["comps"]):
            if k != "medium":
                continue
            key = levels[ci].tobytes()                       # same rest domain + same levels -> same operator (translation only)
            if key in reuse:
                Kops[ci] = reuse[key]; continue
            if direct:
                Kops[ci] = dict(K=KO.KOp(Kf["idx"], Kf["val"], Kf["shape"]), KC=Kf["KC"].to(DEV), cam_keys=list(Kf["cam_keys"]))
            else:
                Kops[ci] = KP.arm_operator(Kf, m, V, levels[ci])
            reuse[key] = Kops[ci]
        r = KP.compose(sd, levels, Kops, sun, keys, label=f"{scene}/{arm}", **kw)
        np.savez(od / f"{arm}.npz", **{f"{key}|{ro}": v[ro] for key, v in r["pred"].items() for ro in ("nee", "ball", "bin")})
        KCm.dump(dict(r["meta"], op=opname, level=lev, kw=kw, levels_per_comp=[np.bincount(l, minlength=5).tolist() for l in levels]), od / f"{arm}.json")
        del Kops, r; torch.cuda.empty_cache()


# ------------------------------------------------------------------ COR


def st_cor(scene, seeds, N=64_000_000):
    od = OUT / scene; od.mkdir(parents=True, exist_ok=True); sd = KF.scene_def(scene)
    keys = sorted(set(cam_dirs_of(sd, list(sd["cams"])))); cor = COR.COR(sd, keys)
    src = "emitter" if sd["emit"] else "sun"
    for seed in seeds:
        if (od / f"COR_{seed}.npz").exists():
            log("skip", scene, seed); continue
        r = cor.run(N, seed, source=src, log=log)
        ball = np.stack([cor.ballistic(k, seed=seed + 100 + k) for k in range(len(keys))])
        np.savez(od / f"COR_{seed}.npz", L=r["L"], ball=ball, arr_tri=r["arr_tri"], arr_port=r["arr_port"], esc=r["esc"], keys=np.array(keys))
        KCm.dump(dict(stats=r["stats"], keys=keys, N=N, source=src), od / f"COR_{seed}.json"); log("COR", scene, seed, json.dumps(r["stats"]))


# ------------------------------------------------------------------ known-field arms


def st_kf(scene, arms):
    od = OUT / scene; od.mkdir(parents=True, exist_ok=True); sd = KF.scene_def(scene); cams = list(sd["cams"]); keys = sorted(set(cam_dirs_of(sd, cams)))
    A = {"K10_s1": (1, "slab_L4_s1"), "K10_s2": (1, "slab_L4_s2"), "K05_s1": (2, "slab_L4_s1"), "K05_s2": (2, "slab_L4_s2"), "K025_s1": (4, "slab_L4_s1"),
         "K025_s2": (4, "slab_L4_s2"), "K10d_s1": (1, "slab_L1d_s1")}
    field = scene.split("_")[-1]
    for arm in (A if arms == ["all"] else arms):
        if (od / f"{arm}.npz").exists():
            continue
        lev, opname = A[arm]; Kf = load_op(opname)
        levels = [np.full(len(m["port_area"]), lev if n == "slab_lo" else 1) for n, k, m, V in sd["comps"]]
        m0, V0 = sd["comps"][0][2], sd["comps"][0][3]
        Kops = {0: dict(K=None, KC=Kf["KC"].to(DEV), cam_keys=list(Kf["cam_keys"]))} if Kf["level"] == lev else {0: KP.arm_operator(Kf, m0, V0, levels[0])}
        if field == "S":
            ss, parent, _ = KP.build_ss(sd, levels); s, E = sd["lights"][0]; Ep, phi = CP.sun_external(ss, s, E, 4096, 5603)
            pred = KP.kf_readout(ss, parent, phi.double(), Kops, keys)
            meta = dict(level=lev, op=opname, field="S_binned_sun")
        else:
            ss, parent, phi = KP.kf_field(sd, levels)
            pred = KP.kf_readout(ss, parent, phi, Kops, keys)
            meta = dict(level=lev, op=opname, field=field, phi_total=float(phi.sum()))
        np.savez(od / f"{arm}.npz", **{f"{key}|{ro}": v[ro] for key, v in pred.items() for ro in ("nee", "ball")})
        KCm.dump(meta, od / f"{arm}.json"); log("kf", scene, arm, meta); del Kf, Kops; torch.cuda.empty_cache()


def st_btest(geo):
    scene = f"kf_{geo}_S"; od = OUT / scene; od.mkdir(parents=True, exist_ok=True); sd = KF.scene_def(scene)
    if not sun_path(scene).exists():
        st_sun(scene)
    sun = torch.load(sun_path(scene)); res = {}
    for lev in (1, 2, 4):
        r = KP.outgoing_test(sd, lev, sun); res[f"L{lev}"] = r
    np.savez(od / "btest.npz", **{f"{k}|{kk}": v for k, d in res.items() for kk, v in d.items() if isinstance(v, np.ndarray)})
    log("btest", geo, {k: float(v["arr"].sum()) for k, v in res.items()})


def main():
    st = sys.argv[1]
    if st == "ops":
        st_ops(sys.argv[2].split(","))
    elif st == "sun":
        st_sun(sys.argv[2])
    elif st == "compose":
        st_compose(sys.argv[2], sys.argv[3].split(","))
    elif st == "cor":
        st_cor(sys.argv[2], [int(s) for s in sys.argv[3].split(",")])
    elif st == "kf":
        st_kf(sys.argv[2], sys.argv[3].split(","))
    elif st == "btest":
        st_btest(sys.argv[2])
    sys.stdout.flush(); os._exit(0)


if __name__ == "__main__":
    main()
