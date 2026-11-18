"""Focused WL101G checks that need no renderer: projection normalization, fallbacks,
adaptive compact indexing, refinement energy and the noise-corrected RMSE estimator."""
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tables101g as T  # noqa: E402


def synthetic_packet(rs, slots=40):
    shape = (slots, 4, 4)
    field = rs.random((*shape, 3)) + 0.05
    n = rs.integers(0, 9, size=shape).astype(np.float64)
    v = np.floor(n * rs.random(shape))
    g = v * rs.random(shape)
    n[0] = 0; v[0] = 0; g[0] = 0               # empty slot
    v[1] = 0; g[1] = 0; n[1] = np.maximum(n[1], 1)  # no visible segment
    g[2, 1] = 0                                 # one empty spatial subcell
    raw = {"n": n, "v": v, "g": g, "L": n[..., None] * field, "LL": n[..., None] * field ** 2,
           "vL": v[..., None] * field, "gL": g[..., None] * field}
    return raw, field


def check_independent(rs):
    raw, field = synthetic_packet(rs)
    tab = T.independent_tables(raw)
    n, v, g = raw["n"], raw["v"], raw["g"]
    def mean(w, axes):
        num = (w[..., None] * field).sum(axes)
        den = w.sum(axes)[..., None]
        return num, den
    num, den = mean(n, (1, 2))
    un = np.where(den > 0, num / np.where(den > 0, den, 1), 0)
    assert np.allclose(tab["Un"][0], un, rtol=1e-6, atol=1e-7)
    assert not tab["Un"][1][0].any()
    num, den = mean(g, (1, 2))
    gg = np.where(den > 0, num / np.where(den > 0, den, 1), un)
    assert np.allclose(tab["G"][0], gg, rtol=1e-6, atol=1e-7)
    assert np.array_equal(tab["G"][0][1], tab["Un"][0][1]) and not tab["G"][1][1].any()
    num, den = mean(g, 2)
    sg = np.where(den > 0, num / np.where(den > 0, den, 1), gg[:, None, :])
    assert np.allclose(tab["S_G"][0], sg, rtol=1e-6, atol=1e-6)
    assert not tab["S_G"][1][2, 1].any() and np.allclose(tab["S_G"][0][2, 1], tab["G"][0][2])
    full = np.where(g[..., None] > 0, field, gg[:, None, None, :])
    assert np.allclose(tab["SA_G"][0], full, rtol=1e-6, atol=1e-6)
    # Etendue-weighted mean of the subcells reproduces the coarse G exactly (nested projection).
    w = g.sum(2)
    recon = np.divide((tab["S_G"][0] * w[..., None]).sum(1), w.sum(1)[:, None],
                      out=np.array(tab["G"][0], np.float64), where=w.sum(1)[:, None] > 0)
    assert np.allclose(recon, tab["G"][0], rtol=1e-5, atol=1e-6)
    return True


def check_adaptive(rs):
    raw, _ = synthetic_packet(rs)
    tab = T.independent_tables(raw)
    refined = rs.random(len(raw["n"])) < 0.3
    table, native, offsets = T.adaptive_table(*tab["G"], *tab["S_G"], refined)
    for slot in range(len(refined)):
        for s in range(4):
            row = T.adaptive_index(offsets, refined, np.array([slot]), np.array([s]))[0]
            expect = tab["S_G"][0][slot, s] if refined[slot] else tab["G"][0][slot]
            assert np.array_equal(table[row], expect)
    assert len(table) == T.adaptive_layout(refined)[1] == int(len(refined) + 3 * refined.sum())
    return True


def check_energy(rs):
    slots = 30
    shape = (slots, 4, 4)
    sub = np.ones((*shape, 3))
    sub[:10, 0] = 3.0                              # structured slots
    g = np.ones(shape)
    raws = [{"g": g, "gL": g[..., None] * sub} for _ in range(4)]
    measure = np.full(slots, 2.0)
    energy = T.refinement_energy(raws, measure, proposals=16)
    lum = sub.sum(2) @ T.LUMA / 4
    parent = lum.mean(1)
    expected = (4 * 4 * 2.0 / (4 * 16)) * ((lum - parent[:, None]) ** 2).sum(1)
    assert np.allclose(energy, expected) and np.all(energy[10:] == 0) and np.all(energy[:10] > 0)
    assert T.select_refined(energy, 10 / 30).sum() == 10 and T.select_refined(energy, 10 / 30)[:10].all()
    return True


def check_cross_rmse(rs):
    h = w = 200
    ref_true = rs.random((h, w, 3)) + 0.5
    bias = 0.05 * np.sin(np.linspace(0, 6, h))[:, None, None] * np.ones((h, w, 3))
    mask = np.ones((h, w), bool)
    stream = [0.05 * rs.standard_normal((h, w, 3)) for _ in range(2)]
    ref = np.stack([ref_true + 0.05 * rs.standard_normal((h, w, 3)) for _ in range(2)])
    arm = np.stack([np.stack([ref[j] + bias + 0.02 * rs.standard_normal((h, w, 3)) * (i + 1) / 4 + stream[j]
                              for j in range(2)]) for i in range(4)])
    est = T.cross_rmse(arm, ref, mask)["relative"]
    truth = np.sqrt(np.mean((bias + stream[0] * 0) ** 2)) / np.sqrt(np.mean(ref_true ** 2))
    # stream terms are shared within j only and independent across j, so they cancel in expectation.
    assert abs(est - truth) / truth < 0.05, (est, truth)
    return {"estimate": est, "truth": float(truth)}


def run_all(seed=101703):
    rs = np.random.default_rng(seed)
    return {"independent_tables": check_independent(rs), "adaptive_index": check_adaptive(rs),
            "refinement_energy": check_energy(rs), "cross_rmse": check_cross_rmse(rs), "seed": seed}


if __name__ == "__main__":
    print(run_all())
