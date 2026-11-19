"""Focused checks on synthetic data. These execute before evidence collection."""

import numpy as np
import torch

from common104d import CL, DA, F, H, STATES
from data104d import proposal, radius_deg
from fit104d import inputs, objective, optimize
from response104d import terms
from scenes104d import EVENTS, pack_metrics


def run_tests():
    rng = np.random.default_rng(104401)
    wi1 = DA.sph(np.radians(np.array([56., 60., 64., 86.])), np.radians(np.array([10., 100., 190., 280.])))
    desc = np.tile([0.05, 0.2, 0.35, 0.8, 0.7, 0.5, 0.1, 0.12, 0.16, 1.45], (4, 1))
    wi, wo, q, use = proposal(desc, wi1, 64, rng)
    radius = radius_deg(wi, wo)
    assert np.all(radius[:3, 32:48] < 3) and np.all((radius[:3, 48:] >= 3) & (radius[:3, 48:] < 12))
    assert np.array_equal(use[:3, 32:], np.ones((3, 32), bool))
    assert np.allclose(radius, radius_deg(wo, wi))
    assert np.isfinite(q).all() and (q > 0).all()
    assert np.array_equal(wo[..., 2] > 0, np.ones(use.shape, bool))

    dev = torch.device("cuda")
    x = torch.tensor([[2., 3., 5.], [5., 4., 3.]], dtype=torch.float64, device=dev, requires_grad=True)
    y = torch.tensor([[1., 2., 3.], [4., 5., 6.]], dtype=torch.float64, device=dev)
    w = torch.ones(2, dtype=torch.float64, device=dev)
    g = F.group_terms(x, y, w, 2)
    assert torch.allclose(objective(g, torch.tensor(False, device=dev)), g["metric"])
    assert torch.allclose(objective(g, torch.tensor(True, device=dev)), g["shape"] + g["e2"])
    obj = lambda pred: objective(F.group_terms(pred, y, w, 2), torch.tensor(True, device=dev))
    assert torch.autograd.gradcheck(obj, (x,))
    display = terms(x.detach().cpu().numpy()[None], y.cpu().numpy()[None], np.ones((1, 2, 1)))
    for key, other in (("e1", "el1"), ("e2", "ealb"), ("shape", "eshape")):
        assert np.allclose(g[key].detach().cpu().numpy(), display[other], rtol=1e-6, atol=1e-8)
    identity = terms(y.cpu().numpy()[None], y.cpu().numpy()[None], np.ones((1, 2, 1)))
    assert all(identity[k][0] == 0 for k in ("el1", "ealb", "eshape"))

    unit_fit = {"wi": wi[:1][None], "wo": wo[:1][None], "q": q[:1][None], "use": use[:1][None],
                "f": np.broadcast_to([0.31, 0.34, 0.37], (1, 1, 64, 3)).copy()}
    batch = inputs(unit_fit, unit_fit, dev, starts=1)
    z = CL.init_raw("ER", "cert", len(STATES) - 2, dev)
    fit1, loss1 = optimize(batch, z, 8, 8)
    fit2, loss2 = optimize(batch, z, 8, 8)
    assert torch.equal(fit1, fit2) and np.array_equal(loss1[8], loss2[8])
    assert torch.equal(fit1[0], fit1[2]) and torch.equal(fit1[1], fit1[3])
    decoded = CL.decode("ER", "cert", fit1)
    assert float(decoded["lam"].max()) <= 1 and bool((decoded["lam"].sum(-1) <= 1.000001).all())

    raw = {"reference": {k: np.ones((4, 2, 2, 3)) * (i + 1) for i, k in enumerate(EVENTS)}}
    raw.update({arm: {k: value * 1.1 for k, value in raw["reference"].items()} for arm in STATES})
    result = pack_metrics(raw, {"objects": np.ones((2, 2), bool)})
    point = result["arms"]["A"]["signed"]["objects"]
    assert abs(sum(point[k]["point"] for k in EVENTS) - point["total"]["point"]) < 1e-12
    assert abs(result["arms"]["A"]["stable_rms"]["objects"]["point"] - 0.1) < 1e-12
    print("PASS synthetic coverage, proposal, autograd, metric, optimizer determinism, certificate, event statistics", flush=True)


if __name__ == "__main__":
    run_tests()
