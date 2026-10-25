"""WL44 candidate training (protocol fnto_k44): r1_train with the renderer-consistent incident field.

Everything is r1_train (kernel family, losses, optimiser, steps, batch, passivity hinge, seed, chunked checkpointing)
except, by monkey-patching four module functions / classes of r1_train:
  Split        reads stagea/prep2_<split> (k44_prep: 0.3 mm visibility rule) and attaches each pixel's box offset
  to_t         carries that offset to the device
  batch_inputs port incident fields of the BOX domain evaluated at world positions (canonical + offset)
  predict      K_loc point / spot light directions and 1 / r^2 evaluated at world positions (canonical + offset)
For the Lego domain the offset is 0, so only the visibility rule differs from r1 there.

    python k44_train.py --out <dir> [r1_train options, e.g. --passivity 1.0 --seed 0]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import k44_common as K  # noqa: F401  (path setup)
import r1_common as C
import r1_train as TR
import fnto_model as M

D = C.OUT1 / "stagea"


class Split2(TR.Split):
    def __init__(self, name):
        od = D / f"prep2_{name}"
        self.px = {k: np.load(od / f"px_{k}.npy", mmap_mode="r") for k in ("pos", "nrm", "cam", "col", "dom", "port", "vbits")}
        self.start = np.load(od / "start.npy"); self.n = len(self.start) - 1
        self.lights = np.load(od / "lights.npy"); self.nl = np.load(od / "nlights.npy")
        self.pvis = np.load(od / "port_vis.npy", mmap_mode="r"); self.family = np.load(od / "family.npy")
        self.boxoff = np.nan_to_num(np.load(od / "boxoff.npy"))

    def pixels(self, j, sel=None):
        out = super().pixels(j, sel)
        out["off"] = np.where((out["dom"] == 1)[:, None], self.boxoff[j][None], 0).astype(np.float32)
        return out


_to_t = TR.to_t


def to_t(d, dev):
    out = _to_t(d, dev)
    out["off"] = torch.as_tensor(np.ascontiguousarray(d["off"])).to(dev).float() if "off" in d else torch.zeros_like(out["pos"])
    return out


def port_incident_off(P, o, L, nl, pv, offs):
    ii = P.idx[o]; n = P.nrm[ii]; td, bd = P.detf[0][ii], P.detf[1][ii]
    Es, Hs = [], []
    for k in range(L.shape[0]):
        J = int(nl[k]); x = P.pos[ii] + (offs[k][None] if o == 1 else 0)
        frac = pv[k, :J].float() / P.nsamp[ii][None]
        w, E = TR.light_dirs_E(L[k, :J], x, n, frac)
        b = M.hemi_bin(torch.stack([(w * td[None]).sum(-1), (w * bd[None]).sum(-1), (w * n[None]).sum(-1)], -1))
        h = torch.zeros(x.shape[0], TR.NB, device=x.device); h.scatter_add_(1, b.T, E.T)
        Es.append(E.sum(0)); Hs.append(h)
    return torch.stack(Es), torch.stack(Hs)


def batch_inputs(S, js, P, dev):
    L = torch.as_tensor(S.lights[js]).to(dev); nl = torch.as_tensor(S.nl[js]).to(dev).long()
    pv = torch.as_tensor(np.stack([np.asarray(S.pvis[j]) for j in js])).to(dev)
    offs = torch.as_tensor(np.stack([S.boxoff[j] for j in js])).to(dev).float()
    inc = []
    for o in (0, 1):
        ii = P.idx[o]; inc.append(port_incident_off(P, o, L, nl, pv[:, :, ii[0]:ii[-1] + 1], offs))
    return L, nl, inc


def predict(Km, P, tab, o, px, L, nl):
    canon, n, wo, port, im = px["pos"], px["nrm"], px["cam"], px["port"], px["im"]
    world = canon + px["off"]
    mat = P.mat[port] if o == 0 else torch.zeros_like(port)
    h = Km.feat(canon, mat); rho, kap, alpha = Km.bsdf_params(h); p, _ = Km.p_q(h); hx = Km.proj(h)
    out = (p * tab["gamma"][im]).sum(1)
    Jm = int(nl.max()); Li = L[im][:, :Jm]
    bits = ((px["vbits"][:, None] >> torch.arange(Jm, device=n.device)[None]) & 1).float()
    bits = bits * (torch.arange(Jm, device=n.device)[None] < nl[im][:, None]).float()
    typ = Li[..., 0]; v = Li[..., 1:4] - world[:, None]; r2 = (v * v).sum(-1).clamp(min=1e-8)
    w = torch.where((typ == 0)[..., None], F.normalize(Li[..., 1:4], dim=-1), v / r2.sqrt()[..., None])
    cos = (w * n[:, None]).sum(-1).clamp(min=0)
    E = torch.where(typ == 0, Li[..., 8] * cos, Li[..., 8] * cos / r2) * bits
    out = out + Km.loc_response(n, wo, rho, kap, alpha, w, E[..., None].expand(-1, -1, 3))
    nb = P.nbr[port]; use = (nb >= 0).float(); nbc = nb.clamp(min=0)
    W = Km.band_w(hx, tab["hs"][nbc], n, P.bnrm[nbc], (P.bpos[nbc] - canon[:, None]) / C.R_LOCAL, wo)
    out = out + torch.einsum("nkbc,nkb->nc", W * (P.nbrA[port] * use)[..., None, None], tab["hB"][im[:, None], nbc])
    mc = P.mid[port]; usem = (mc >= 0).float(); mcc = mc.clamp(min=0)
    Wm = Km.mid_w(hx, tab["hc"][mcc], n, P.cnrm[mcc], (P.cpos[mcc] - canon[:, None]) / C.R_MID, wo)
    out = out + (Wm * (P.midA[port] * usem)[..., None] * tab["Ec"][im[:, None], mcc][..., None]).sum(1)
    return out


TR.Split, TR.to_t, TR.batch_inputs, TR.predict = Split2, to_t, batch_inputs, predict

if __name__ == "__main__":
    out = Path(sys.argv[sys.argv.index("--out") + 1]); out.mkdir(parents=True, exist_ok=True)
    (out / "k44_correction.json").write_text(json.dumps({"protocol": "fnto_k44", "prep": "stagea/prep2_<split> (k44_prep, T_MIN 0.3 mm)",
                                                         "box_world_frame": True, "argv": sys.argv[1:]}, indent=1))
    TR.main()
