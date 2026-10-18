"""Worklog 37: cross-machine check (5080 PC vs LabServer63) on the same T0 oracle views (same cameras, lights, seeds):
noise-debiased RMS display difference = sqrt(mean((T(L_pc,1)-T(L_srv,1)) * (T(L_pc,2)-T(L_srv,2)))) and mean signed
difference per channel. Decides whether server-rendered T0 held-out references can be paired with PC-rendered T1 ones."""
import json, sys, h5py, numpy as np
T = lambda x: (np.clip(x, 0, None) / (1 + np.clip(x, 0, None))) ** (1 / 2.2)
pc, sv = sys.argv[1], sys.argv[2]
f = {(m, k): h5py.File(f"{d}/oracle_T0_s{k}_val.h5", "r") for m, d in (("pc", pc), ("sv", sv)) for k in (1, 2)}
S = n = 0; mean = np.zeros(3); aov = 0.0; nb = 0
for v in range(f[("pc", 1)]["color"].shape[0]):
    a = np.ones(f[("pc", 1)]["alpha"].shape[1], bool)
    for key in f: a &= f[key]["alpha"][v][:, 0].astype(np.float32) >= 0.999
    d1 = T(f[("pc", 1)]["color"][v][a].astype(np.float32)) - T(f[("sv", 1)]["color"][v][a].astype(np.float32))
    d2 = T(f[("pc", 2)]["color"][v][a].astype(np.float32)) - T(f[("sv", 2)]["color"][v][a].astype(np.float32))
    S += float((d1 * d2).sum()); n += d1.size; mean += (0.5 * (d1 + d2)).sum(0); nb += a.sum()
    aov = max(aov, float(np.abs(f[("pc", 1)]["position"][v][a].astype(np.float32) - f[("sv", 1)]["position"][v][a].astype(np.float32)).max()))
print(json.dumps({"rms_machine_difference": float(np.sqrt(max(S / n, 0))), "ms": S / n, "mean_signed_rgb": (mean / nb).tolist(), "pixels": int(nb),
                  "position_aov_max_abs_diff": aov, "tau": 0.005}, indent=1))
