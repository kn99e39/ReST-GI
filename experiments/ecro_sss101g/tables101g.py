"""WL101G table builders and estimators (numpy only, no renderer import).

Nested construction sums are indexed (slot, s, t): slot = pid*64+bin, s = 2x2 spatial
subcell, t = 2x2 directional subcell (WL101F address convention).
"""
import math

import numpy as np

LUMA = np.array((0.2126, 0.7152, 0.0722))
NSLOT_BINS = 64


def projected(numerator, denominator, fallback):
    """Weighted mean with explicit fallback where the denominator is zero (WL101F rule)."""
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.broadcast_to(np.asarray(denominator, dtype=np.float64), numerator.shape[:-1])
    valid = np.broadcast_to((denominator > 0)[..., None], numerator.shape)
    assert np.all(numerator[~valid] == 0)
    out = np.array(np.broadcast_to(fallback, numerator.shape), dtype=np.float64)
    np.divide(numerator, np.broadcast_to(denominator[..., None], numerator.shape), out=out, where=valid)
    return out.astype(np.float32), np.array(valid)


def slot_measure(psid, parea, sphere_sid=6):
    """Proposal measure per slot (patch area x lookup-preimage solid angle, m^2 sr)."""
    omega = np.full((len(psid), NSLOT_BINS), 2 * math.pi / NSLOT_BINS)
    sphere = np.asarray(psid) == sphere_sid
    omega[np.ix_(sphere, np.arange(16))] = 2 * math.pi * 1.25 / 16
    return (np.asarray(parea, dtype=np.float64)[:, None] * omega).ravel()


def independent_tables(raw, names=("Un", "V", "G", "S_G", "A_G", "SA_G", "S_U")):
    """Camera-independent tables from one replica's nested sums.

    raw keys (shapes S x 4 x 4 [x 3]): n, v, g, L, vL, gL.
    """
    n, v, g = (np.asarray(raw[k], dtype=np.float64) for k in ("n", "v", "g"))
    L, vL, gL = (np.asarray(raw[k], dtype=np.float64) for k in ("L", "vL", "gL"))
    out = {}
    un_t, un_v = projected(L.sum((1, 2)), n.sum((1, 2)), 0.0)
    out["Un"] = (un_t, un_v)
    v_t, v_v = projected(vL.sum((1, 2)), v.sum((1, 2)), un_t)
    out["V"] = (v_t, v_v)
    g_t, g_v = projected(gL.sum((1, 2)), g.sum((1, 2)), un_t)
    out["G"] = (g_t, g_v)
    parent_g = g_t.astype(np.float64)
    if "S_G" in names:
        out["S_G"] = projected(gL.sum(2), g.sum(2), parent_g[:, None, :])
    if "A_G" in names:
        out["A_G"] = projected(gL.sum(1), g.sum(1), parent_g[:, None, :])
    if "SA_G" in names:
        out["SA_G"] = projected(gL, g, parent_g[:, None, None, :])
    if "S_U" in names:
        out["S_U"] = projected(L.sum(2), n.sum(2), un_t.astype(np.float64)[:, None, :])
    return {k: out[k] for k in names}


def refinement_energy(raws, measure, proposals):
    """Cross-validated G-measure between-subcell energy per slot (luminance).

    raws: four replica sum packets. Halves {0,1} and {2,3}. Subcell etendue from all
    four replicas. Returns energy (S,) with possibly negative cross-product values.
    """
    assert len(raws) == 4
    def half(ids):
        g = sum(np.asarray(raws[i]["g"], np.float64).sum(2) for i in ids)
        gL = sum(np.asarray(raws[i]["gL"], np.float64).sum(2) for i in ids)
        lum = gL @ LUMA
        parent = np.divide(lum.sum(1), g.sum(1), out=np.zeros(len(g)), where=g.sum(1) > 0)
        sub = np.divide(lum, g, out=np.zeros_like(g), where=g > 0)
        return sub - parent[:, None], g > 0
    da, ok_a = half((0, 1))
    db, ok_b = half((2, 3))
    g_all = sum(np.asarray(r["g"], np.float64).sum(2) for r in raws)
    etendue = g_all * (np.asarray(measure, np.float64)[:, None] / (len(raws) * proposals))
    both = ok_a & ok_b
    return np.where(both, etendue * da * db, 0.0).sum(1)


def select_refined(energy, fraction):
    count = int(math.ceil(fraction * len(energy)))
    order = np.argsort(-energy, kind="stable")
    refined = np.zeros(len(energy), dtype=bool)
    refined[order[:count]] = True
    return refined


def adaptive_layout(refined):
    rows = np.where(refined, 4, 1)
    offsets = np.concatenate(([0], np.cumsum(rows)[:-1])).astype(np.uint32)
    return offsets, int(rows.sum())


def adaptive_table(parent, parent_native, spatial, spatial_native, refined):
    """Compact table: refined slots carry four spatial subcells, others the parent."""
    offsets, total = adaptive_layout(refined)
    table = np.zeros((total, 3), dtype=np.float32)
    native = np.zeros((total, 3), dtype=bool)
    keep = ~refined
    table[offsets[keep]] = parent[keep]
    native[offsets[keep]] = parent_native[keep]
    idx = offsets[refined][:, None] + np.arange(4)[None, :]
    table[idx] = spatial[refined]
    native[idx] = spatial_native[refined]
    return table, native, offsets


def adaptive_index(offsets, refined, slot, s):
    slot = np.asarray(slot)
    return offsets[slot] + np.where(refined[slot], np.asarray(s), 0)


# ---------------------------------------------------------------- metrics
def cross_mse(arm, ref, mask, same_replica=False):
    """Noise-corrected squared bias on mask.

    arm: (I, 2, H, W, 3) or (2, H, W, 3) for single-table arms; ref: (2, H, W, 3).
    Uses ordered replica pairs i != i' across the two independent streams.
    """
    ref0, ref1 = (np.asarray(ref[j][mask], np.float64) for j in (0, 1))
    power = float(np.mean(ref0 * ref1))
    if arm.ndim == 4:
        d0 = np.asarray(arm[0][mask], np.float64) - ref0
        d1 = np.asarray(arm[1][mask], np.float64) - ref1
        prod = d0 * d1
        return float(np.mean(prod)), power, [], float(prod.std(ddof=1) / np.sqrt(prod.size))
    n = arm.shape[0]
    d0 = [np.asarray(arm[i, 0][mask], np.float64) - ref0 for i in range(n)]
    d1 = [np.asarray(arm[i, 1][mask], np.float64) - ref1 for i in range(n)]
    pairs = {}
    prod = 0.
    for i in range(n):
        for k in range(n):
            if (i == k) == same_replica:
                product = d0[i] * d1[k]
                pairs[(i, k)] = float(np.mean(product))
                prod = prod + product
    prod = prod / len(pairs)
    pixel_se = float(prod.std(ddof=1) / np.sqrt(prod.size))
    value = float(np.mean(list(pairs.values())))
    jack = []
    if not same_replica and n >= 3:
        for drop in range(n):
            jack.append(float(np.mean([v for (i, k), v in pairs.items() if drop not in (i, k)])))
    return value, power, jack, pixel_se


def signed_relative(mse, power):
    return float(math.copysign(math.sqrt(abs(mse) / power), mse)) if power > 0 else float("nan")


def cross_rmse(arm, ref, mask, same_replica=False):
    mse, power, jack, pixel_se = cross_mse(arm, ref, mask, same_replica)
    rel = signed_relative(mse, power)
    se = None
    if len(jack) >= 3:
        rels = np.array([signed_relative(j, power) for j in jack])
        n = len(rels)
        se = float(math.sqrt((n - 1) / n * np.sum((rels - rels.mean()) ** 2)))
    return {"relative": rel, "mse": mse, "reference_power": power, "jackknife_se": se,
            "pixel_product_se_relative_mse": pixel_se / power if power > 0 else None}
