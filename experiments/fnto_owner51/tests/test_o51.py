"""WL51 focused tests (GPU, LabServer69). python tests/test_o51.py <out json>"""
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import o51_ops as O  # noqa: E402
from o51_ops import CC, C  # noqa: E402

R = {}


def check(n, ok, **i):
    R[n] = {"pass": bool(ok), **i}; print(n, R[n], flush=True)


@torch.no_grad()
def main(outp):
    dev = torch.device("cuda")
    sha = hashlib.sha256((C.OUT1 / "K_k44P/kernels.pt").read_bytes()).hexdigest()
    ck = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:8] for p in sorted(O.CK.glob("ckpt_*.pt"))}
    exp = {"ckpt_L0_seed5011.pt": "0dfc7793", "ckpt_L0_seed5012.pt": "3b40a940", "ckpt_L1_seed5011.pt": "6041f0b4", "ckpt_L1_seed5012.pt": "1aa75e89"}
    check("baseline_identity", sha.startswith("30d55da373a01ab8") and ck == exp, k44=sha[:16], ckpts=ck)
    T, H, Rl = O.setup(dev); cls = O.classes(T); tT = torch.as_tensor(Rl["t"], device=dev)
    # classification vs anchors
    A = np.load(C.OUT1 / "anchors.npz"); obj, mat = A["obj"], A["mat"]; nA = int(A["n_A"])
    exp_cls = np.select([obj == 1, mat == 0, np.isin(mat, [2, 4])], [3, 0, 2], 1)
    yp = CC.yellow_ports()
    check("class_classification", bool((cls.cpu().numpy() == exp_cls).all()) and int((cls == 0).sum()) == len(yp) and bool((obj[:nA] == 0).all() and (obj[nA:] == 1).all()),
          counts={n: int((cls == k).sum()) for k, n in enumerate(O.NAMES)})
    # port areas
    ar = T.area.cpu().numpy()
    check("port_area_sums", abs(ar[:nA].sum() - 40.83549118041992) < 1e-2 and abs(ar[yp].sum() - 15.874663352966309) < 1e-2, lego=float(ar[:nA].sum()), yellow=float(ar[yp].sum()))
    # production unchanged; mask none = production (compared on a fixed port subset: no full fp32 copies)
    rows = torch.as_tensor(np.sort(np.random.default_rng(7).choice(T.N, 40000, replace=False)), device=dev)
    CC.deactivate(); S0 = O.sun_off_state(T, H); g = torch.Generator(device=dev).manual_seed(3)
    L0 = (torch.rand(S0.L.shape, generator=g, device=dev, dtype=torch.float16) * S0.valid[..., None]).half(); S0.L = L0.clone(); S0.step(sun_on=False); ref = S0.L[rows].float()
    S0.L = L0.clone(); S0.step(sun_on=False); dprod = float((S0.L[rows].float() - ref).abs().max())
    del S0; torch.cuda.empty_cache()
    CC.activate(None); S1 = O.sun_off_state(T, H); keep = O.edge_keep(S1, cls, "none", tT); O.apply_mask(S1, keep); S1.L = L0.clone(); S1.step(sun_on=False)
    d1 = float((S1.L[rows].float() - ref).abs().max())
    check("mask_none_equals_production", bool(torch.equal(keep, S1.valid)) and d1 <= max(4 * dprod, 1e-3 * float(ref.abs().max())), d=d1, prod_vs_prod=dprod)
    del keep; torch.cuda.empty_cache()
    # edge-mask correctness on a port subset (exact)
    sub = np.random.default_rng(5).choice(T.N, 3000, replace=False); Hs = H[sub].astype(np.int64); ok = Hs >= 0
    cn = cls.cpu().numpy(); r = cn[sub][:, None]; s = cn[np.clip(Hs, 0, None)]; t = Rl["t"][sub]; Yr, Ys = r == 0, s == 0
    expect = {"noYY": Yr & Ys, "noYO": Ys & ~Yr, "noOY": ~Ys & Yr, "noOO": ~Ys & ~Yr, "onlyYY": ~(Yr & Ys), "noYY_t3mm": Yr & Ys & (t < 0.003),
              "noYY_t3cm": Yr & Ys & (t < 0.03), "noCross": (r == 3) != (s == 3)}
    bad = {}
    for mk, m in expect.items():
        kk = O.edge_keep(S1, cls, mk, tT)[torch.as_tensor(sub, device=dev)].cpu().numpy()
        bad[mk] = int((kk != (ok & ~m)).sum()); torch.cuda.empty_cache()
    check("edge_mask_correctness", all(v == 0 for v in bad.values()), mismatches=bad)
    del S1, tT; torch.cuda.empty_cache()
    # channel-diagonal, homogeneity (L1 arm seed 5011)
    geo = CC.Geo(dev); arm = O.make_arm("L1C", dev, geo); CC.activate(arm); S2 = O.sun_off_state(T, H)
    S2.L = L0.clone(); S2.step(sun_on=False); a = S2.L[rows].float()
    Lp = L0.clone(); Lp[..., 0] = (Lp[..., 0].float() * 0.5).half(); S2.L = Lp; del Lp; S2.step(sun_on=False); b = S2.L[rows].float()
    S2.L = L0.clone(); S2.step(sun_on=False); a2 = S2.L[rows].float(); del S2; torch.cuda.empty_cache()
    nd = float((a2 - a).abs().max()); dgb = float((b[..., 1:] - a[..., 1:]).abs().max()); hom = float(((b[..., 0] - 0.5 * a[..., 0]).abs().max()) / a[..., 0].abs().max())
    check("channel_diagonal_homogeneous", dgb <= max(4 * nd, 1e-3 * float(a[..., 1:].abs().max())) and hom < 5e-3, GB_change_when_R_scaled=dgb, nondet=nd, R_homogeneity_rel=hom)
    # SEM override: only Yellow sender values change, and equal the formula
    S3 = O.sun_off_state(T, H); S3.L = L0.clone(); Ei, h = S3.aggregates(); Ip = S3.outgoing(Ei, h, sun_on=False).clone(); O.sem_on(S3); Is = S3.outgoing(Ei, h, sun_on=False)
    ypt = torch.as_tensor(yp, device=dev); ny = torch.ones(T.N, dtype=torch.bool, device=dev); ny[ypt] = False
    form = torch.tensor(O.RHO_Y_INT, device=dev) / math.pi * Ei[ypt] + T.kap[ypt].sum(1) / math.pi * Ei[ypt]
    check("sem_override_scope", float((Is[ny] - Ip[ny]).abs().max()) <= 1e-6 * float(Ip.abs().max()) and float((Is[ypt] - form).abs().max()) <= 1e-6 * float(form.abs().max()),
          nonyellow_maxabs=float((Is[ny] - Ip[ny]).abs().max()), yellow_formula_maxabs=float((Is[ypt] - form).abs().max()))
    # units: uniform unit radiance on every valid gather -> E_ind = pi x valid fraction
    S3.L = S3.valid[..., None].expand(-1, -1, 3).half().contiguous(); Ei, _ = S3.aggregates()
    fr = S3.valid.float().mean(1) * math.pi
    check("unit_consistency_uniform_radiance", float((Ei - fr[:, None]).abs().max()) < 1e-4, maxabs=float((Ei - fr[:, None]).abs().max()))
    CC.deactivate()
    Path(outp).parent.mkdir(parents=True, exist_ok=True); Path(outp).write_text(json.dumps(R, indent=1))
    print("ALL PASS" if all(v["pass"] for v in R.values()) else "FAILURES", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
