"""WL54 amendment 2 (audit finding): validate the robust spectral-radius estimator against scipy Arnoldi and compare the WL53 estimator.
    python c54_rho_check.py <out json>"""
import sys
from pathlib import Path

import numpy as np
import scipy.sparse.linalg as sla
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "phys_ref53"))
import c54_casp as CA  # noqa: E402
import c54_common as CC  # noqa: E402
import c54_stageE as SE  # noqa: E402
import p53_compose as CP  # noqa: E402
import p53_fixtures as FX  # noqa: E402
import p53_oracle as OR  # noqa: E402

R = Path.home() / "rest_wl53/results/oracle"; out = {}
for scene, T in (("floor", "slab_s1"), ("gap4", "slab_s1")):
    sd = FX.scene_def(scene); Tc = OR.load_T(R, T); ss0 = OR.scene_set(sd); Tm = {ci: Tc["T"] for ci, c in enumerate(ss0.comps) if c.kind == "medium"}
    for s_uni in (1, 2):
        ov, par = SE.comps_with_levels(sd, None, s_uniform=s_uni) if s_uni > 1 else SE.comps_with_levels(sd, None)
        ss = OR.scene_set(sd, ov); par_all = np.concatenate(par); m = np.maximum(1, np.round(128 / np.bincount(par_all)[par_all])).astype(int)
        C, esc, st = CA.cast_adaptive(ss, m, 5340); n = ss.Ptot * ss.Dout
        def mv(x):
            v = torch.zeros(n, 3, device="cuda"); v[:, 0] = torch.tensor(x.astype(np.float32), device="cuda")
            return CP.apply_K(ss, Tm, CP.apply_C(C, v))[:, 0].double().cpu().numpy()
        vals = sla.eigs(sla.LinearOperator((n, n), matvec=mv, dtype=np.float64), k=4, which="LM", tol=1e-6, maxiter=3000, return_eigenvectors=False)
        out[f"{scene}_s{s_uni}"] = dict(arnoldi_R=sorted(np.abs(vals).tolist(), reverse=True), robust_A2=CA.rho_robust(ss, Tm, C), wl53_power300=CP.spectral_radius(ss, Tm, C), n=n)
        print(scene, s_uni, out[f"{scene}_s{s_uni}"], flush=True); del C, ss; torch.cuda.empty_cache()
CC.dump(out, Path(sys.argv[1]))
