"""Budget-matched coverage replacement and independent layered response probes."""

import time

import numpy as np

from common104d import DA, H, P, S, relative_difference, write


def theta_deg(w):
    return np.degrees(np.arccos(np.clip(w[..., 2], -1, 1)))


def mirror(wi):
    return wi * [-1, -1, 1]


def radius_deg(wi, wo):
    return np.degrees(np.arccos(np.clip(np.sum(mirror(wi) * wo, axis=-1), -1, 1)))


def oriented(center, radius, phi):
    tangent = np.cross(center, [0, 0, 1.0])
    tangent /= np.linalg.norm(tangent, axis=-1, keepdims=True)
    bitangent = np.cross(center, tangent)
    return np.cos(radius)[..., None] * center + np.sin(radius)[..., None] * (
        np.cos(phi)[..., None] * tangent + np.sin(phi)[..., None] * bitangent
    )


def proposal(desc, incidence, lanes, rng):
    """Stratified balance-heuristic mixture. Invalid draws carry zero weight."""
    n_groups = len(incidence)
    if lanes % 4:
        raise ValueError("The declared proposal requires four equal lane blocks")
    half, quarter = lanes // 2, lanes // 4
    wi = np.broadcast_to(incidence[:, None, :], (n_groups, lanes, 3))
    old_wi = wi[:, :half].reshape(-1, 3)
    old_wo, _, old_use = DA.mixture(np.repeat(desc, half, axis=0), old_wi, rng)
    wo = np.empty_like(wi)
    wo[:, :half] = old_wo.reshape(n_groups, half, 3)
    use = np.ones(wi.shape[:2], dtype=bool)
    use[:, :half] = old_use.reshape(n_groups, half)
    edges = np.cos(np.radians([0, 3, 12]))
    for j in range(2):
        block = slice(half + j * quarter, half + (j + 1) * quarter)
        radius = np.arccos(rng.uniform(edges[j + 1], edges[j], size=(n_groups, quarter)))
        phi = rng.uniform(0, 2 * np.pi, size=radius.shape)
        wo[:, block] = oriented(mirror(incidence)[:, None, :], radius, phi)
    use &= wo[..., 2] > 1e-7
    wo[wo[..., 2] <= 1e-7] = [0.0, 0.0, 1.0]
    density = DA.W_MIX[0] * wo[..., 2] / np.pi + DA.W_MIX[1] / (2 * np.pi)
    wi_flat, wo_flat = wi.reshape(-1, 3), wo.reshape(-1, 3)
    for fraction, alpha in zip(DA.W_MIX[2:], (desc[:, 0], np.minimum(1, desc[:, 1] * desc[:, 9]))):
        density += fraction * DA.vndf_pdf(wi_flat, wo_flat, np.repeat(alpha, lanes)).reshape(use.shape)
    dot = np.sum(mirror(wi) * wo, axis=-1)
    q = 0.5 * density
    q += 0.25 * (dot > edges[1]) / (2 * np.pi * (edges[0] - edges[1]))
    q += 0.25 * ((dot > edges[2]) & (dot <= edges[1])) / (2 * np.pi * (edges[1] - edges[2]))
    return wi, wo, q, use


def walks(desc, mid, wi, wo, use, count, seed):
    """Return independently walked halves in the original shape."""
    active = use.ravel()
    a, b = DA.ref_eval(desc, mid.ravel()[active], wi.reshape(-1, 3)[active], wo.reshape(-1, 3)[active], count, seed)
    halves = np.zeros((2, active.size, 3), dtype=np.float64)
    halves[0, active], halves[1, active] = a, b
    return halves.reshape((2,) + use.shape + (3,))


def coverage(desc, original, out):
    config = P["training_data"]
    limits = config["qualification"]
    rng = np.random.default_rng(S["coverage_directions"])
    n = len(desc)
    shape = (n, 16)
    theta = 55 + 2.5 * (np.repeat(np.arange(4), 4) + rng.random(shape))
    phi = 90 * (np.tile(np.arange(4), 4) + rng.random(shape))
    inc = DA.sph(np.radians(theta.ravel()), np.radians(phi.ravel()))
    wi, wo, q, use = proposal(np.repeat(desc, 16, axis=0), inc, config["lanes_per_group"], rng)
    mid = np.repeat(np.arange(n), 16 * config["lanes_per_group"]).reshape(use.shape)
    start = time.monotonic()
    # Like WL83, evaluate the placeholder direction of a rejected draw and give it zero weight.
    a, b = walks(desc, mid, wi, wo, np.ones_like(use), config["walks_per_lane"], S["fit_walks"])
    added = {"wi": wi, "wo": wo, "q": q, "use": use, "fA": a, "fB": b, "f": 0.5 * (a + b)}
    data = {key: val.copy() for key, val in original.items()}
    original_theta = theta_deg(original["wi"][:, :, 0])
    changed = []
    for i, angles in enumerate(original_theta):
        low = np.flatnonzero(angles < 55)
        high = np.flatnonzero((angles > 65) & (angles < 80))
        if min(len(low), len(high)) < 8 or np.any((angles >= 55) & (angles <= 65)):
            raise ValueError("Original groups cannot support the frozen replacement rule")
        replace = np.r_[low[np.argsort(angles[low])[-8:]], high[np.argsort(angles[high])[:8]]]
        changed.append(replace)
        for key in data:
            data[key][i, replace] = added[key][16 * i:16 * (i + 1)]
    np.savez(out / "train_covered.npz", **data, replacement=np.stack(changed))
    elapsed = time.monotonic() - start

    radii = radius_deg(wi, wo)
    bands = (radii < 3, (radii >= 3) & (radii < 12))
    quadrature = use.astype(np.float64) / q
    results = []
    for j in range(n):
        rows = slice(16 * j, 16 * (j + 1))
        polar = np.floor((theta[j] - 55) / 2.5).astype(int)
        azimuth = (phi[j] // 90).astype(int)
        cells = len(set(zip(polar, azimuth)))
        row_help = {"occupied_incident_cells": cells, "strata": {}}
        row_ok = cells == 16
        for label, region in zip(("core", "shoulder"), bands):
            w = quadrature[rows] * region[rows]
            contributions = (0.5 * (a[rows] + b[rows]) @ H.LUMA) * w
            sums = contributions.reshape(4, -1).sum(axis=1)
            ess = sums**2 / np.maximum((contributions.reshape(4, -1)**2).sum(axis=1), 1e-20)
            pa = ((a[rows] @ H.LUMA) * w).reshape(4, -1).sum(axis=1)
            pb = ((b[rows] @ H.LUMA) * w).reshape(4, -1).sum(axis=1)
            half = relative_difference(pa, pb)
            hits = (w > 0).sum(axis=1)
            row_help["strata"][label] = {"ess": ess.tolist(), "half_relative_difference": half.tolist(),
                                         "minimum_hits_per_group": int(hits.min())}
            row_ok &= bool(hits.min() >= 16 and ess.min() >= limits["minimum_stratum_effective_samples"]
                           and half.max() <= limits["maximum_reference_half_relative_difference"])
        results.append({**row_help, "support_pass": row_ok})

    # Balanced fixed subset of injected core/shoulder draws. Compare f, not f*cos(theta_o).
    quarter = config["lanes_per_group"] // 4
    lanes = np.array([2 * quarter, 2 * quarter + 1, 3 * quarter, 3 * quarter + 1])
    first, second = wi[:, lanes], wo[:, lanes]
    rec_mid = mid[:, lanes]
    active = np.ones(rec_mid.shape, bool)
    rec_a, rec_b = walks(desc, rec_mid, first, second, active, limits["reciprocity_walks"], S["reciprocal_walks"])
    sw_a, sw_b = walks(desc, rec_mid, second, first, active, limits["reciprocity_walks"], S["reciprocal_walks"] + 1)
    rec_forward = (np.stack([rec_a, rec_b]) @ H.LUMA) / second[..., 2]
    rec_reverse = (np.stack([sw_a, sw_b]) @ H.LUMA) / first[..., 2]
    weight = quadrature[:, lanes].reshape(n, 16, 2, 2)
    fr = (rec_forward.reshape(2, n, 16, 2, 2) * weight).sum(axis=(2, 4))
    rv = (rec_reverse.reshape(2, n, 16, 2, 2) * weight).sum(axis=(2, 4))
    fr_mean, rv_mean = fr.mean(0), rv.mean(0)
    denom = 0.5 * (fr_mean + rv_mean)
    rec_gap = relative_difference(fr_mean, rv_mean)
    rec_unc = 0.5 * np.hypot(fr[0] - fr[1], rv[0] - rv[1]) / np.maximum(denom, 1e-20)
    for i, row in enumerate(results):
        limit = limits["reciprocity_absolute_margin"] + 4 * rec_unc[i]
        row["reciprocity"] = {"relative_error": rec_gap[i].tolist(), "se_proxy": rec_unc[i].tolist(),
                              "passed": bool(np.all(rec_gap[i] <= limit))}
        row["grazing_groups_preserved"] = bool(np.sum(original_theta[i] >= 80) ==
                                               np.sum(theta_deg(data["wi"][i, :, 0]) >= 80))
        row["replaced_incident_theta"] = original_theta[i, changed[i]].tolist()
        row["added_incident_theta"] = theta[i].tolist()
        row["added_incident_phi"] = phi[i].tolist()
    report = {"materials": results, "seconds_for_new_fit_walks": elapsed,
              "lanes_per_material_both": config["groups"] * config["lanes_per_group"],
              "walks_per_material_both": config["groups"] * config["lanes_per_group"] * config["walks_per_lane"],
              "new_fit_walks": wi.shape[0] * wi.shape[1] * config["walks_per_lane"],
              "reciprocity_audit_walks": 2 * active.size * limits["reciprocity_walks"]}
    report["passed"] = all(row["support_pass"] and row["reciprocity"]["passed"] and row["grazing_groups_preserved"]
                           for row in results)
    write(out / "coverage.json", report)
    return data, report


def response_reference(desc, out):
    config = P["response"]
    theta, phi = np.meshgrid(config["angles_deg"], config["azimuths_deg"], indexing="ij")
    incidence = np.tile(DA.sph(np.radians(theta.ravel()), np.radians(phi.ravel())), (len(desc), 1))
    n_groups = theta.size
    wi, wo, q, use = proposal(np.repeat(desc, n_groups, axis=0), incidence, config["lanes"],
                              np.random.default_rng(S["evaluation_directions"]))
    mid = np.broadcast_to(np.repeat(np.arange(len(desc)), n_groups)[:, None], use.shape)
    a, b = walks(desc, mid, wi, wo, use, config["walks"], S["evaluation_walks"])
    radii = np.r_[0.0, np.repeat(np.arange(0.25, 4.01, 0.25), 16)]
    azimuth = np.r_[0.0, np.tile((np.arange(16) + 0.5) * (2 * np.pi / 16), 16)]
    peak_wo = oriented(mirror(incidence)[:, None, :], np.radians(radii[None, :]), azimuth[None, :])
    peak_wi = np.broadcast_to(incidence[:, None, :], peak_wo.shape)
    peak_mid = np.broadcast_to(mid[:, :1], peak_wo.shape[:2])
    peak_a, peak_b = walks(desc, peak_mid, peak_wi, peak_wo, peak_wo[..., 2] > 1e-7,
                           config["peak_walks"], S["peak_walks"])
    data = {"wi": wi, "wo": wo, "q": q, "use": use, "a": a, "b": b, "mid": mid,
            "peak_wi": peak_wi, "peak_wo": peak_wo, "peak_a": peak_a, "peak_b": peak_b}
    np.savez(out / "independent_response.npz", **data)
    return data
