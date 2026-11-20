"""Fresh paired moment qualification at the sole WL104F operator failure."""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import qmc

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ecro_wl84_104f"))
from cross104f import C, H, frozen_inputs, mi, dr, read, sha_ok, write

P = read(HERE / "protocol.json")
OP = P["operator"]
EXACT = np.array([1., 1., 1., 4. / 3.])


def vector(array):
    return np.stack([np.asarray(array[i], dtype=np.float64) for i in range(3)], axis=-1)


def moments(points):
    return np.column_stack((np.ones(len(points)), points))


def statistics(values, margin, reference=0., reference_se=0.):
    mean = values.mean(axis=0)
    se = np.sqrt(values.var(axis=0, ddof=1) / len(values) + np.asarray(reference_se)**2)
    error = mean - reference
    upper = np.abs(error) + OP["sigma"] * se
    return {"mean": mean.tolist(), "error": error.tolist(), "se": se.tolist(),
            "abs_4sem_bound": upper.tolist(), "margin": margin, "passed": bool(np.all(upper <= margin))}


def native_blocks(closure, wi, rng):
    known, measured, pdf_max, pdf_p999, pdf_median, weight = [], [], [], [], [], []
    n = OP["mc_lanes_per_block"]
    for _ in range(OP["mc_blocks"]):
        u = [mi.Float(v) for v in rng.random((3, n), dtype=np.float32)]
        wo, density, throughput, valid = closure.sample(wi, u[0], mi.Point2f(u[1], u[2]))
        good = np.asarray(valid, dtype=bool)
        points = vector(wo)[good]
        p = np.asarray(density, dtype=np.float64)
        f, evaluated = closure.eval_pdf(wi, wo, mi.Bool(valid))
        f = vector(f)
        w = vector(throughput)
        p_eval = np.asarray(evaluated, dtype=np.float64)
        if (not np.isfinite(p).all() or not np.isfinite(f).all() or not np.isfinite(w).all()
                or np.any(p[good] <= 0.) or np.any(p[~good] != 0.) or np.any(w[~good] != 0.)
                or np.any(w < 0.) or np.any(f < 0.) or np.any(points[:, 2] <= 1e-7)):
            raise ValueError("Invalid support, density or throughput")
        errors = np.abs(p_eval[good] - p[good]) / p[good]
        pdf_max.append(errors.max())
        median, p999 = np.quantile(errors, [.5, .999])
        pdf_p999.append(p999)
        pdf_median.append(median)
        weight.append(np.max(np.abs(w * p[:, None] - f)) / f.max())
        measured.append(moments(points).sum(axis=0) / n)
        integrands = 1. + .5 * points
        cosine = points[:, 2] / (np.pi * p[good])
        known.append(np.column_stack((cosine, cosine[:, None] * integrands)).sum(axis=0) / n)
    parity = {"max_relative_pdf": float(max(pdf_max)), "max_block_p999_relative_pdf": float(max(pdf_p999)),
              "max_block_median_relative_pdf": float(max(pdf_median)),
              "max_relative_peak_weight_residual": float(max(weight)),
              "passed": bool(max(pdf_median) <= 1e-5 and max(pdf_p999) <= 1e-3 and max(weight) <= 2e-5)}
    return np.asarray(known), np.asarray(measured), parity


def integrate(closure, wi):
    output = []
    for i in range(OP["sobol_scrambles"]):
        u = qmc.Sobol(2, scramble=True, seed=OP["sobol_seed"] + i).random_base2(OP["sobol_power"])
        z, theta = u[:, 0], 2 * np.pi * u[:, 1]
        r = np.sqrt(1 - z*z)
        xyz = np.column_stack((r * np.cos(theta), r * np.sin(theta), z))
        wo = mi.Vector3f(*(mi.Float(a.astype(np.float32)) for a in xyz.T))
        pdf = np.asarray(closure.pdf(wi, wo), dtype=np.float64)
        if not np.isfinite(pdf).all() or np.any(pdf[z > 1e-7] <= 0):
            raise ValueError("Reported PDF has invalid support in the independent integration")
        output.append(2 * np.pi * (moments(xyz) * pdf[:, None]).mean(axis=0))
    return np.asarray(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    _, rows, _, states = frozen_inputs()
    historical = C.REPO / "evaluation/104F"
    manifest = read(historical / "manifest.json")
    for name in ("decision.json", "operator_audit.json", "status.json"):
        sha_ok(historical / name, manifest["packaged_sha256"][name])
    sha_ok(HERE.parent / "ecro_wl84_104f/protocol.json", read(historical / "status.json")["protocol_sha256"])
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    index = next(i for i, row in enumerate(rows) if row["tag"] == OP["material"])
    t, p = np.radians([OP["theta"], OP["phi"]])
    wi = mi.Vector3f(float(np.sin(t)*np.cos(p)), float(np.sin(t)*np.sin(p)), float(np.cos(t)))
    records = {}
    raw = {}
    for i, arm in enumerate(OP["arms"]):
        state = states[arm][index]
        probs = state[45:53]
        if not np.isfinite(state).all() or probs[0] <= 0. or probs.min() < 0. or abs(probs.sum() - 1.) > 1e-6:
            raise ValueError(f"Invalid proposal mixture for {arm}")
        closure = H.CD.DrClosure83("ER", state)
        known, empirical, parity = native_blocks(closure, wi, np.random.default_rng(OP["mc_seed"] + i))
        numerical = integrate(closure, wi)
        known_result = statistics(known, OP["known_margin"], EXACT)
        dist_result = statistics(empirical, OP["distribution_margin"], numerical.mean(axis=0),
                                 numerical.std(axis=0, ddof=1) / np.sqrt(len(numerical)))
        records[arm] = {"known_cosine": known_result, "raw_distribution": dist_result, "parity": parity,
                        "independent_pdf_integrals": numerical.mean(axis=0).tolist(),
                        "passed": all(x["passed"] for x in (known_result, dist_result, parity))}
        raw.update({f"{arm}_known": known, f"{arm}_empirical": empirical, f"{arm}_sobol": numerical})
    np.savez(args.output / "operator_blocks.npz", **raw)
    payload = {"passed": all(row["passed"] for row in records.values()), "arms": records, "config": OP,
               "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               "protocol_sha256": C.sha(HERE / "protocol.json"), "seconds": time.monotonic()-started,
               "device_allocator": [line.strip() for line in dr.whos_str().splitlines() if "used (peak:" in line]}
    write(args.output / "operator.json", payload)
    print("operator qualified:", payload["passed"], flush=True)


if __name__ == "__main__":
    main()
