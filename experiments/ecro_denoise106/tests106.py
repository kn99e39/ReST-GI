"""WL106 focused tests (operator, equivalence, model contract). python tests106.py"""
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import model106 as M6  # noqa: E402
import tilt106 as T  # noqa: E402
import synth106 as S  # noqa: E402
M = M6.M


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name, detail, flush=True)
    if not ok:
        raise SystemExit(1)


def main():
    dev = "cuda"
    g = torch.Generator().manual_seed(106)
    scenes = S.scenes(96)
    cov = torch.stack([torch.from_numpy(sc["cover8"]) for sc in scenes.values()]).to(dev)          # (B,9,H,W)
    B, _, H, W = cov.shape
    logits = (torch.randn(B, M.K * M.K, H, W, generator=g) * 2).to(dev)
    # 1 gather order equals the unfold order used by the kernel application
    c10 = T.coverage10(cov)
    idx = torch.tensor([[0, 0, 0], [1, 40, 47], [2, H - 1, W - 3]], device=dev)
    r = M.K // 2
    unf = torch.nn.functional.unfold(torch.nn.functional.pad(c10, (r, r, r, r), mode="reflect"), M.K).view(B, 10, M.K * M.K, H, W)
    ref = torch.stack([unf[i[0], :, :, i[1], i[2]] for i in idx])
    check("window gather == unfold order", torch.equal(T.gather_window(c10, idx), ref))
    # 2 moment constraint satisfied, masked support exact
    st = {}
    with torch.no_grad():
        zc = T.constrain_logits(logits, cov, stats=st)
    names = list(scenes)
    res = {}
    for i, nm in enumerate(names):
        s1 = {}
        with torch.no_grad():
            T.constrain_logits(logits[i:i + 1], cov[i:i + 1], stats=s1)
        res[nm] = float(s1["residual"].max())
    check("moment residual <= 1e-4 on generic synthetic coverage", max(res[n] for n in S.GATED + ("edge0", "strip_wide")) <= 1e-4, res)
    check("moment residual <= 5e-3 at extreme-point pixels (thin strip, asymptotic)", res["strip_thin"] <= 5e-3, res["strip_thin"])
    # 3 untouched where the window holds one coverage vector
    act = T.active_pixels(c10)
    check("inactive pixels bit-identical", torch.equal(zc.permute(0, 2, 3, 1)[~act], logits.permute(0, 2, 3, 1)[~act]))
    # 4 constant field exact
    img = torch.full((B, 3, H, W), 0.37, device=dev)
    with torch.no_grad():
        y = M6.constrained_apply(img, logits, cov)
    check("constant field exact (reference path)", float((y - 0.37).abs().max()) < 1e-6, float((y - 0.37).abs().max()))
    # 5 Triton solver == torch solver
    cp = c10[idx[:, 0], :, idx[:, 1], idx[:, 2]]
    sel = (T.coverage10(cov) > 0).sum(1) >= 2
    sidx = sel.nonzero()
    sq = T.gather_window(T.support_bits(c10), sidx)[:, 0].long()
    sp = T.support_bits(c10)[sidx[:, 0], 0, sidx[:, 1], sidx[:, 2]].long()
    z = logits[sidx[:, 0], :, sidx[:, 1], sidx[:, 2]] + T.MASK * ((sq & ~sp[:, None]) != 0)
    cps = c10[sidx[:, 0], :, sidx[:, 1], sidx[:, 2]]
    two = (cps > 0).sum(1) <= 3
    T.USE_TRITON = True
    with torch.no_grad():
        la, ch, cq = T.tilt_rows(z[two], c10, sidx[two], cps[two])
    T.USE_TRITON = False
    with torch.no_grad():
        lb, _, _ = T.tilt_rows(z[two], c10, sidx[two], cps[two])
    T.USE_TRITON = True
    wa = torch.softmax(torch.log_softmax(z[two], 1) + torch.einsum("ns,nsk->nk", la, cq), 1)
    wb = torch.softmax(torch.log_softmax(z[two], 1) + torch.einsum("ns,nsk->nk", lb, cq), 1)
    check("Triton solver == torch solver (kernel weights, abs <= 1e-4)", float((wa - wb).abs().max()) <= 1e-4, (float((wa - wb).abs().max()), int(two.sum())))
    # 6 fused application == reference path
    img = (torch.rand(B, 3, H, W, generator=g) * 5).to(dev)
    with torch.no_grad():
        ref = M6.constrained_apply(img, logits, cov)
        fus = T.constrained_apply_fused(img, logits, cov)
    err = float(((fus - ref).abs() / ref.abs().clamp_min(1e-3)).max())
    check("fused constrained application == reference", err < 1e-4, err)
    # 7 implicit gradient: autograd vs central finite differences of one tilted pixel's output
    p = sidx[sidx[:, 0] == 0][len(sidx[sidx[:, 0] == 0]) // 3]
    def yp(l):
        return M6.constrained_apply(img[:1], l, cov[:1])[0, 0, p[1], p[2]]
    lg = logits[:1].clone().requires_grad_(True)
    yp(lg).backward()
    fds, ans = [], []
    for k in range(0, 225, 4):
        d = torch.zeros_like(lg); d[0, k, p[1], p[2]] = 3e-2
        with torch.no_grad():
            fd = float((yp(logits[:1] + d) - yp(logits[:1] - d)) / 6e-2)
        fds.append(fd); ans.append(float(lg.grad[0, k, p[1], p[2]]))
    rel = float(np.linalg.norm(np.array(fds) - np.array(ans)) / np.linalg.norm(ans))
    check("implicit gradient matches finite differences (vector rel <= 2e-2, 57 taps)", rel <= 2e-2, rel)
    # 8 model contract: R4 == R3 parameters / names / init, deterministic, input unmodified, HDR finite
    torch.manual_seed(10550001); r3 = M.Reconstructor("R3")
    torch.manual_seed(10550001); r4 = M6.R4()
    check("R4 state_dict keys and shapes == R3", [(k, v.shape) for k, v in r3.state_dict().items()] == [(k, v.shape) for k, v in r4.state_dict().items()])
    check("R4 init == R3 init under the same seed", all(torch.equal(a, b) for a, b in zip(r3.state_dict().values(), r4.state_dict().values())))
    check("R4 parameter count == R3", M.n_params(r3) == M.n_params(r4), M.n_params(r4))
    r4 = r4.to(dev).eval()
    comp = (torch.rand(B, 9, H, W, generator=g) * 2).to(dev)
    aux = torch.cat([cov, torch.rand(B, 5, H, W, generator=g).to(dev)], 1)
    before = comp.clone()
    for fused in (False, True):
        M6.FUSED = fused
        with torch.no_grad():
            y1, _ = r4(comp, aux); y2, _ = r4(comp, aux)
        check(f"R4 deterministic (fused={fused})", torch.equal(y1, y2))
        check(f"R4 input unmodified (fused={fused})", torch.equal(before, comp))
        with torch.no_grad():
            yc, _ = r4(torch.full_like(comp, 0.7), aux)
        check(f"R4 constant exact (fused={fused})", float((yc - 2.1).abs().max()) < 1e-5, float((yc - 2.1).abs().max()))
        with torch.no_grad():
            yh, _ = r4(comp * 500, aux)
        check(f"R4 HDR finite (fused={fused})", bool(torch.isfinite(yh).all()) and float(yh.max()) > 1)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
