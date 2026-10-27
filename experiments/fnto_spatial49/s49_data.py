"""WL49 shared data (torch, GPU): Lego port tables, K44 probe / view irradiance fields, receivers, the worklog-45 Burley
oracle (object scope, self port excluded) split into in-support (r < 30 cm) and tail responses, and the area-mean source
aggregates K_band / K_mid read (protocol fnto_spatial49)."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "fnto_lego_r1")); sys.path.insert(0, str(HERE.parent / "fnto_lego"))
import r1_common as C  # noqa: E402

PROTO = json.loads((HERE / "protocol" / "spatial49.json").read_text())
STAGEA = C.OUT1 / "stagea"
R_IN = C.R_MID                                  # in-support boundary (30 cm)


def burley(r, d):
    return (torch.exp(-r / d) + torch.exp(-r / (3 * d))) / (8 * math.pi * d * r)


def dch():
    fit = json.loads((C.ROOT / "results/fnto_diag45/track_a/a1a_fit.json").read_text())["fit"]
    return [fit[c]["d_fit_m"] for c in "RGB"]


def s_c():
    return json.loads((C.ROOT / "results/fnto_diag45/track_a/a3_oracle.json").read_text())["source_sets"]["obj"]["scale_per_channel"]


class Lego:
    def __init__(self, dev):
        A = np.load(C.OUT1 / "anchors.npz"); self.nA = nA = int(A["n_A"]); self.dev = dev
        t = lambda x, dt=torch.float32: torch.as_tensor(np.asarray(x), device=dev).to(dt)  # noqa: E731
        self.pos, self.nrm, self.area = t(A["pos"][:nA]), t(A["nrm"][:nA]), t(A["area"][:nA])
        self.mat = A["mat"][:nA].astype(np.int64); self.comp = A["comp"][:nA]
        self.f2b, self.f2c = t(A["f2b"][:nA], torch.long), t(A["f2c"][:nA], torch.long)
        self.NB, self.NC = int(A["bpos"].shape[0]), int(A["cpos"].shape[0])
        self.bpos, self.cpos = t(A["bpos"]), t(A["cpos"]); self.bnrm, self.cnrm = t(A["bnrm"]), t(A["cnrm"])
        self.nbr_np, self.nbrA_np, self.mid_np, self.midA_np = A["nbr"], A["nbrA"], A["mid"], A["midA"]
        self.nsamp = t(np.isfinite(A["spos"][:nA, :, 0]).sum(1).clip(1, None))
        self.barea_m = torch.zeros(self.NB, device=dev).index_add_(0, self.f2b, self.area)    # member area sums
        self.carea_m = torch.zeros(self.NC, device=dev).index_add_(0, self.f2c, self.area)
        atom = self.f2b * self.NC + self.f2c; self.atom_u, self.atom = torch.unique(atom, return_inverse=True)
        self.atom_area = torch.zeros(len(self.atom_u), device=dev).index_add_(0, self.atom, self.area)

    def rep_lists(self, ports):
        p = np.asarray(ports)
        t = lambda x, dt=torch.float32: torch.as_tensor(np.asarray(x), device=self.dev).to(dt)  # noqa: E731
        return t(self.nbr_np[p], torch.long), t(self.nbrA_np[p]), t(self.mid_np[p], torch.long), t(self.midA_np[p])

    def agg_means(self, E):
        """Area-mean irradiance of every band aggregate [NB, F] and mid cell [NC, F] for fields E [nA, F]."""
        AE = E * self.area[:, None]
        return (torch.zeros(self.NB, E.shape[1], device=self.dev).index_add_(0, self.f2b, AE) / self.barea_m[:, None].clamp(min=1e-20),
                torch.zeros(self.NC, E.shape[1], device=self.dev).index_add_(0, self.f2c, AE) / self.carea_m[:, None].clamp(min=1e-20))

    def atom_project(self, E):
        """Collision field: area-mean over band x mid atoms (keeps every band / mid aggregate mean)."""
        AE = E * self.area[:, None]
        m = torch.zeros(len(self.atom_u), E.shape[1], device=self.dev).index_add_(0, self.atom, AE) / self.atom_area[:, None].clamp(min=1e-20)
        return m[self.atom]


def probe_fields(L, split, ids):
    """K44 probe configurations -> Lego port irradiance [nA, len(ids)] (r1_train.light_dirs_E semantics)."""
    import r1_train as TR
    d = STAGEA / f"prep2_{split}"; pv = np.load(d / "port_vis.npy", mmap_mode="r"); LT = np.load(d / "lights.npy"); nl = np.load(d / "nlights.npy")
    out = torch.zeros(L.nA, len(ids), device=L.dev)
    for i, k in enumerate(ids):
        J = int(nl[k]); frac = torch.as_tensor(np.asarray(pv[k, :J, :L.nA]), device=L.dev).float() / L.nsamp[None]
        _, E = TR.light_dirs_E(torch.as_tensor(LT[k, :J], device=L.dev).float(), L.pos, L.nrm, frac); out[:, i] = E.sum(0)
    return out


def view_fields(L, state="T0"):
    px = np.load(C.OUT1 / f"eval_px/heldout_{state}.npz"); vis = np.load(C.OUT1 / "eval_vis_k44.npz"); nv = int(px["nv"])
    E = torch.stack([C.E_SUN * (L.nrm @ torch.as_tensor(px[f"v{v}_light"], device=L.dev).float()).clamp(min=0)
                     * torch.as_tensor(vis[f"heldout_{state}_v{v}_{state}"].astype(np.float32)[:L.nA], device=L.dev) for v in range(nv)], 1)
    return E


def splits():
    rng = np.random.default_rng(4901); perm = rng.permutation(1000)
    return {"fit": np.sort(perm[:480]), "val": np.sort(perm[480:600]), "test_probe": np.arange(120)}


def receivers(L):
    yel = np.flatnonzero(L.mat == 0)
    fit = np.sort(np.random.default_rng(4902).choice(yel, 2000, replace=False))
    rest = np.setdiff1d(yel, fit); held = np.sort(np.random.default_rng(4903).choice(rest, 1000, replace=False))
    return fit, held


@torch.no_grad()
def oracle(L, xi, E, d, ch=1 << 14, extra=None):
    """In-support y_S30 and tail y_T [n, F] for receiver ports xi (self excluded). extra: optional second field set."""
    xp = L.pos[xi]; F_ = E.shape[1]
    yS = torch.zeros(len(xi), F_, device=L.dev, dtype=torch.float64); yT = torch.zeros_like(yS)
    AE = (E * L.area[:, None])
    for s0 in range(0, L.nA, ch):
        s1 = min(s0 + ch, L.nA); r = torch.cdist(xp, L.pos[s0:s1]).clamp(min=2e-3)
        k = burley(r, d) * (r < 16 * d) * (xi[:, None] != torch.arange(s0, s1, device=L.dev)[None])
        kin = k * (r < R_IN)
        yS += (kin @ AE[s0:s1]).double(); yT += ((k - kin) @ AE[s0:s1]).double()
    return yS, yT


@torch.no_grad()
def dist_to_lit(L, xi, E, ch=1 << 15):
    """Distance from each receiver to the nearest lit Lego port, per field [n, F] (inf if none)."""
    out = torch.full((len(xi), E.shape[1]), float("inf"), device=L.dev); xp = L.pos[xi]
    for f in range(E.shape[1]):
        lp = L.pos[E[:, f] > 0]
        for s0 in range(0, len(lp), ch):
            out[:, f] = torch.minimum(out[:, f], torch.cdist(xp, lp[s0:s0 + ch]).min(1).values)
    return out


@torch.no_grad()
def cell_avg(L, xi, cells, members, ptr, d, lo, hi):
    """Area-weighted mean of k * [lo <= r < hi] over the members of cells[i, j] (self excluded); and member area sums."""
    xp = L.pos[xi]; n, J = cells.shape; cc = cells.clamp(min=0); a, b = ptr[cc], ptr[cc + 1]; cnt = (b - a) * (cells >= 0)
    Lm = int(cnt.max()); avg = torch.zeros(n, J, device=L.dev)
    for j0 in range(0, J, 8):
        sl = slice(j0, j0 + 8); ar = torch.arange(Lm, device=L.dev)
        idx = a[:, sl, None] + ar; ok = ar < cnt[:, sl, None]; fid = members[idx.clamp(max=len(members) - 1)]
        r = (L.pos[fid] - xp[:, None, None]).norm(dim=-1).clamp(min=2e-3)
        k = burley(r, d) * (r < 16 * d) * (r >= lo) * (r < hi) * ok * (fid != xi[:, None, None])
        A_ = L.area[fid] * ok
        avg[:, sl] = (k * A_).sum(-1) / A_.sum(-1).clamp(min=1e-20)
    return avg


def csr(f2x, n):
    o = torch.argsort(f2x, stable=True); cnt = torch.bincount(f2x, minlength=n); ptr = torch.zeros(n + 1, dtype=torch.long, device=f2x.device); ptr[1:] = torch.cumsum(cnt, 0)
    return o, ptr


def rel(a, b, m=None):
    m = torch.ones_like(b, dtype=torch.bool) if m is None else m
    m = m.double(); a, b = a.double(), b.double()
    return float(torch.sqrt((((a - b) ** 2) * m).sum() / ((b ** 2) * m).sum().clamp(min=1e-60)))
