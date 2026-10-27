"""WL52 learned inter-domain transport C1 (protocol fnto_lc52): geometry-conditioned latent transport (GLT).

    E_ind(r) = sum_{s in cand(r)} F_rs(G) * Lbar_s,     F_r. = pi * softmax([l_null(r), l_rs])  over NULL + k candidates

  geometry G   patch tokens (area-weighted centroid, mean normal, log area, Fourier position features) -> transformer encoder
               (global geometry context: the only place occlusion can be inferred; no rays)
  receiver r   port position / normal + the latent of its own patch
  candidates   EVERY source patch with positive analytic UNOCCLUDED point-to-patch transfer
               FF0 = a cos_r+ cos_s+ / (pi d^2 + a) (pure geometry, no visibility); dense receiver x patch evaluation
               (a top-k set failed the predeclared 99 % coverage rule: k = 64 / 128 / 192 / 256 -> 71 / 84 / 91 / 95 %)
  pair logit   l_rs = log(FF0_rs / pi) + g_phi(q_r, z_s, relative geometry); the network learns visibility and quadrature
  operator     linear in the source radiance (patch area means Lbar), F >= 0, sum_s F_rs <= pi (passive; = pi * (1 - p_null))
Cost per geometry: encoder O(M^2 d), FF0 O(N M), pair head O(N M d) - DENSE in receiver x patch, so the asymptotic scaling is
not improved over explicit gathers O(N K log T); per GI iteration: dense O(N M) matrix product.
"""
import math

import torch
import torch.nn as nn

NF = 6                                                     # Fourier frequencies per axis (scene ~2 m)
SCALE = 2.0


def fourier(x):
    f = (2.0 ** torch.arange(NF, device=x.device, dtype=x.dtype)) * (math.pi / SCALE)
    a = x[..., None] * f
    return torch.cat([torch.sin(a), torch.cos(a)], -1).flatten(-2)


def frame(n):
    a = torch.where(n[..., 2:3].abs() < 0.9, n.new_tensor([0.0, 0, 1]).expand_as(n), n.new_tensor([1.0, 0, 0]).expand_as(n))
    t = torch.nn.functional.normalize(torch.cross(a, n, dim=-1), dim=-1); return t, torch.cross(n, t, dim=-1)


@torch.no_grad()
def ff0(pos, nrm, pcen, pnrm, parea):
    """Analytic unoccluded point-to-patch transfer [n, M] (0 where either side faces away)."""
    d = pcen[None] - pos[:, None]; d2 = (d * d).sum(-1).clamp(min=1e-10); dh = d / d2.sqrt()[..., None]
    cr = (dh * nrm[:, None]).sum(-1).clamp(min=0); cs = (-(dh * pnrm[None]).sum(-1)).clamp(min=0)
    return parea[None] * cr * cs / (math.pi * d2 + parea[None])


@torch.no_grad()
def candidates(pos, nrm, pcen, pnrm, parea, k, chunk=4096):
    """Top-k source patches by analytic unoccluded transfer FF0 per receiver -> idx [n, k], FF0 [n, k]."""
    I, Fv = [], []
    for s in range(0, len(pos), chunk):
        p, n = pos[s:s + chunk], nrm[s:s + chunk]; d = pcen[None] - p[:, None]; d2 = (d * d).sum(-1).clamp(min=1e-10); dh = d / d2.sqrt()[..., None]
        cr = (dh * n[:, None]).sum(-1).clamp(min=0); cs = (-(dh * pnrm[None]).sum(-1)).clamp(min=0)
        ff = parea[None] * cr * cs / (math.pi * d2 + parea[None]); v, i = torch.topk(ff, min(k, ff.shape[1]), dim=1); I.append(i); Fv.append(v)
    return torch.cat(I), torch.cat(Fv)


def ff0_total(pos, nrm, pcen, pnrm, parea, chunk=4096):
    out = []
    for s in range(0, len(pos), chunk):
        p, n = pos[s:s + chunk], nrm[s:s + chunk]; d = pcen[None] - p[:, None]; d2 = (d * d).sum(-1).clamp(min=1e-10); dh = d / d2.sqrt()[..., None]
        out.append((parea[None] * (dh * n[:, None]).sum(-1).clamp(min=0) * (-(dh * pnrm[None]).sum(-1)).clamp(min=0) / (math.pi * d2 + parea[None])).sum(1))
    return torch.cat(out)


class GLT(nn.Module):
    def __init__(self, d=128, layers=6, heads=4, ff=256, hid=128):
        super().__init__()
        tin = 3 + 3 + 1 + 3 * 2 * NF
        self.tok = nn.Sequential(nn.Linear(tin, d), nn.GELU(), nn.Linear(d, d))
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, heads, ff, dropout=0.0, batch_first=True, norm_first=True, activation="gelu"), layers)
        self.rq = nn.Sequential(nn.Linear(3 + 3 + 3 * 2 * NF + d, d), nn.GELU(), nn.Linear(d, d))
        self.A, self.B, self.Cr = nn.Linear(d, hid), nn.Linear(d, hid, bias=False), nn.Linear(10, hid, bias=False)
        self.pair = nn.Sequential(nn.GELU(), nn.Linear(hid, hid), nn.GELU(), nn.Linear(hid, 1))
        self.null = nn.Sequential(nn.Linear(d + 1, hid), nn.GELU(), nn.Linear(hid, 1))

    def encode(self, pcen, pnrm, parea):
        x = torch.cat([pcen, pnrm, torch.log(parea.clamp(min=1e-8))[..., None], fourier(pcen)], -1)
        return self.enc(self.tok(x)[None])[0]                                                    # [M, d]

    def transfer(self, z, pos, nrm, patch_of, pcen, pnrm, parea, pobj, robj, ffm=None):
        """Dense over source patches -> F [n, M] (x pi, 0 where FF0 = 0), F_null [n], probabilities [n, M + 1]."""
        ffm = ff0(pos, nrm, pcen, pnrm, parea) if ffm is None else ffm
        q = self.rq(torch.cat([pos, nrm, fourier(pos), z[patch_of]], -1))                       # [n, d]
        d = pcen[None] - pos[:, None]; dist = d.norm(dim=-1).clamp(min=1e-5); dh = d / dist[..., None]
        t, b = frame(nrm); cr = (dh * nrm[:, None]).sum(-1); cs = -(dh * pnrm[None]).sum(-1)
        rel = torch.stack([(dh * t[:, None]).sum(-1), (dh * b[:, None]).sum(-1), cr, cs, torch.log(dist), torch.log(ffm.clamp(min=1e-10)),
                           (pobj[None] == robj[:, None]).float(), (nrm[:, None] * pnrm[None]).sum(-1), torch.log(parea.clamp(min=1e-8))[None] - 2 * torch.log(dist),
                           torch.sqrt(parea)[None] / dist], -1)
        h = self.A(q)[:, None] + self.B(z)[None] + self.Cr(rel)
        logit = torch.where(ffm > 0, torch.log(ffm.clamp(min=1e-12) / math.pi) + self.pair(h)[..., 0], torch.full_like(ffm, float("-inf")))
        ln = self.null(torch.cat([q, torch.log1p(-(ffm.sum(1) / math.pi).clamp(max=0.999))[:, None]], -1))[:, 0]
        P = torch.softmax(torch.cat([ln[:, None], logit], 1), 1)
        return math.pi * P[:, 1:], math.pi * P[:, 0], P

    @staticmethod
    def apply(F, Lbar):
        """E [n, 3] = F [n, M] @ Lbar [M, 3]."""
        return F @ Lbar


def n_params(m):
    return sum(p.numel() for p in m.parameters())
