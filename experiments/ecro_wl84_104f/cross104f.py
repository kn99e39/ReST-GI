"""Cross the first ER evaluation with the actual native A or C proposal."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ecro_wl84_104e"))
from trace104e import C, H, PT, RT, Counterfactual, dr, mi
from metrics104e import event_metrics, stable_rms
from run104e import PROV, l1_ratio, load_case, read, sha_ok, write
from stats104a import ratio_jk
from stats104b import jk_ci, ms_loo

P = read(HERE / "protocol.json")
PARENT = C.REPO / "results/ecro_wl84_104e"
ROOT = C.REPO / "evaluation/104D/amendment1"
VARIANTS = ("AA", "AC", "CA", "CC")  # F followed by p


def frozen_inputs():
    meta = read(ROOT / "provenance.json")
    for name in ("compiled_states.npz", "materials.json"):
        sha_ok(ROOT / name, meta["output_sha256"][name])
    descriptor = C.W83 / "data/materials83.npz"
    sha_ok(descriptor, meta["parent_source_sha256"]["data/materials83.npz"])
    for label, mesh in (("sphere", C.SPHERE_PLY), ("quad", C.QUAD_PLY)):
        sha_ok(mesh, C.P["scene"][f"{label}_ply_sha256"])
    rows = read(ROOT / "materials.json")
    with np.load(ROOT / "compiled_states.npz") as z:
        states = {k: z[k] for k in ("A", "C", "ER_DF")}
    with np.load(descriptor) as z:
        materials = [z[row["set"]][row["index"]] for row in rows]
    return meta, rows, materials, states


class Cross(Counterfactual):
    def evaluate_cross(self, seed, proposal, light, spp, launch_spp):
        acc = {}
        for launch, start in enumerate(range(0, spp, launch_spp)):
            batch = min(launch_spp, spp - start)
            n = C.RES * C.RES * batch
            s = dr.opaque(mi.UInt64, seed * 1000003 + launch)
            rng = mi.PCG32(size=n, initstate=s, initseq=mi.UInt64(0))
            ray, pix, _ = self.gen_rays(batch, rng)
            si, sid, _, ok = self.surface(ray, mi.Bool(True))
            active = ok & (si.wi.z > 1e-6)
            native = self.bsdf(si, sid, proposal)
            bs, w = native.sample(self.bctx, si, rng.next_float32(),
                                  mi.Point2f(rng.next_float32(), rng.next_float32()), active)
            used = active & (bs.pdf > 0) & (dr.max(w) > 0)
            proper_pdf = dr.select(used, bs.pdf, 1.0)
            wo = dr.select(used, bs.wo, mi.Vector3f(0, 0, 1))
            weights = {
                f: dr.select(used, self.bsdf(si, sid, f).eval(self.bctx, si, wo, used) / proper_pdf, 0.0)
                for f in "AC"
            }
            other, other_sid, _, ok = self.surface(si.spawn_ray(si.to_world(bs.wo)), used)
            receiver, _, _ = self.terminal(other, other_sid, light, used & ok & (other.wi.z > 1e-6), ("C",))
            lanes = {f"{f}{k}": (weights[f] * receiver["C"])[k] for f in "AC" for k in range(3)}
            a = self.accum(lanes, pix, C.RES * C.RES)
            for key, x in a.items():
                acc[key] = acc[key] + x if key in acc else x
        return {f + proposal: PT.pack({f"e1{k}": acc[f"{f}{k}"] for k in range(3)}, C.RES, C.RES, spp)["e1"]
                for f in "AC"}


def blocks(value, shape):
    return np.asarray(value, dtype=np.float64).reshape(shape)


def operator_case(frozen, theta, phi, seed, settings):
    """Check mass with an independent cosine component and exact smooth integrals."""
    shape = (settings["blocks"], settings["samples_per_proposal_per_block"])
    rg = np.random.default_rng(seed)
    t, p = np.radians([theta, phi])
    wi = mi.Vector3f(float(np.sin(t) * np.cos(p)), float(np.sin(t) * np.sin(p)), float(np.cos(t)))
    closures = {name: H.CD.DrClosure83("ER", st) for name, st in frozen.items()}
    samples, accepted = {}, {}
    for name in ("A", "C", "cosine"):
        u = [mi.Float(rg.random(np.prod(shape), dtype=np.float32)) for _ in range(3)]
        if name == "cosine":
            out = mi.warp.square_to_cosine_hemisphere(mi.Point2f(u[1], u[2]))
            eligible = out.z > 1e-7
        else:
            out, _, _, eligible = closures[name].sample(wi, u[0], mi.Point2f(u[1], u[2]))
        samples[name] = dr.select(eligible, out, mi.Vector3f(0, 0, 1))
        accepted[name] = eligible

    pn, fn, gate, exact = {}, {}, {}, {}
    opt = settings
    for key, wo in samples.items():
        use = accepted[key]
        result = {name: obj.eval_pdf(wi, wo, use) for name, obj in closures.items()}
        valid = np.asarray(use, dtype=bool).reshape(shape)
        gate[key] = valid
        pn[key] = {name: blocks(pdf, shape) for name, (_, pdf) in result.items()}
        fn[key] = {name: np.stack([blocks(f[k], shape) for k in range(3)], axis=-1) for name, (f, _) in result.items()}
        z = blocks(wo.z, shape)
        comp = np.stack((np.ones(shape), 1 + .5 * blocks(wo.x, shape),
                         1 + .5 * blocks(wo.y, shape), 1 + .5 * z), axis=-1)
        exact[key] = (z / np.pi)[..., None] * comp

    def weighted(a, pdf, use):
        pdf = np.where(use, pdf, 1)
        return np.where(use[..., None], a / pdf[..., None], 0.).mean(axis=1)

    def check(v, limit):
        mean = np.asarray(v).mean(axis=0)
        se = np.asarray(v).std(axis=0, ddof=1) / np.sqrt(shape[0])
        return {"difference": mean.tolist(), "se": se.tolist(),
                "passed": bool(np.all(np.abs(mean) <= np.maximum(limit, opt["sigma_multiplier"] * se)))}

    normalization, known, expectation = {}, {}, {}
    valid_support = all(np.isfinite(a).all() and np.all(a[gate[key]] > 0)
                        for key, pair in pn.items() for a in pair.values())
    if not valid_support:
        raise ValueError("Native evaluated density failed the common-support check")
    for name in "AC":
        integral = np.zeros(shape[0])
        for key, wo in samples.items():
            z = blocks(wo.z, shape)
            q = (pn[key]["A"] + pn[key]["C"] + z / np.pi) / 3
            integral += weighted(pn[key][name][..., None], q, gate[key])[:, 0] / 3
        accepted_mass = gate[name].mean(axis=1)
        normalization[name] = {**check(integral - accepted_mass, opt["integration_abs_tolerance"]),
                               "acceptance": float(accepted_mass.mean()), "integrated_pdf": float(integral.mean())}
        moments = weighted(exact[name], pn[name][name], gate[name])
        known[name] = check(moments - np.array([1, 1, 1, 4 / 3]), opt["integration_abs_tolerance"])
    for target in "AC":
        a = weighted(fn["A"][target], pn["A"]["A"], gate["A"])
        c = weighted(fn["C"][target], pn["C"]["C"], gate["C"])
        mean = (a + c).mean(axis=0) * 0.5
        expectation[target] = {**check((c - a) / mean, opt["f_expectation_rel_tolerance"]), "rgb": mean.tolist()}
    return {"theta": theta, "phi": phi, "seed": seed, "support_passed": valid_support,
            "pdf_mass": normalization, "known_cosine_moments": known, "fixed_f_albedo": expectation,
            "passed": all(x["passed"] for group in (normalization, known, expectation) for x in group.values())}


def audit_states(rows, states):
    opt = P["part_a"]["operator_audit"]
    histories = {}
    for j, i in enumerate((0, 2, P["held_out"]["index"])):
        source = {arm: state[i] for arm, state in states.items() if arm in "AC"}
        if not all(np.isfinite(v).all() and v[45] > 0 and np.all(v[45:53] >= 0)
                   and abs(v[45:53].sum() - 1) < 1e-6 for v in source.values()):
            raise ValueError("Invalid positive Lambert sampling component")
        records = []
        for t, theta in enumerate(opt["theta_deg"]):
            for p, phi in enumerate(opt["phi_deg"]):
                records.append(operator_case(source, theta, phi, opt["seeds"] + 100*j + 2*t + p, opt))
        histories[rows[i]["tag"]] = {"min_lambert_probability": min(float(v[45]) for v in source.values()),
                                     "probes": records, "passed": all(r["passed"] for r in records)}
    return histories


def difference_statistics(x, y, ref, mask):
    ref_total = sum(ref[k][:, mask] for k in ("e1", "e2", "h"))
    d = x[:, mask] - y[:, mask]
    den = (ref_total.mean(0)**2).mean()
    d_luma = (d @ H.LUMA).sum(axis=1)
    return {
        "rms": stable_rms(d.reshape(len(d), -1), den),
        "signed": ratio_jk(d_luma, (ref_total @ H.LUMA).sum(axis=1)),
    }


def projections(result, mask):
    d = {k: value[:, mask].reshape(len(value), -1) for k, value in result.items()}
    original = d["CC"] - d["AA"]
    den, dlo = ms_loo(original, original)
    if den <= 0 or np.any(dlo <= 0):
        return {}
    out = {}
    for p in "AC":
        a, lo = ms_loo(d["C" + p] - d["A" + p], original)
        out[p] = jk_ci(a / den, lo / dlo)
    return out


def cross_scene(parent, case, row, descriptor, states, out, smoke):
    fields, masks, baselines = load_case(parent / "parent", case, read(PROV / "provenance.json"))
    b = P["budget"]
    g = P["part_a"]
    seed, light = b["seed_base"][case], RT.point_lights(row["light"])
    ctx = Cross(row["tag"], descriptor, states["A"], states["C"])
    ref = fields["reference"]
    focused = {}
    for proposal, title in (("A", "AA"), ("C", "CC")):
        s = b["focused_spp"]
        a = ctx.evaluate_cross(seed, proposal, light, s, s)
        repeat = ctx.evaluate_cross(seed, proposal, light, s, s)
        native = ctx.evaluate(seed, proposal, light, s, s)["AC" if proposal == "A" else "CC"]
        ident = l1_ratio(a[title], native)
        det = max(l1_ratio(a[k], repeat[k]) for k in a)
        if max(ident, det) > g["identity_relative_l1"]:
            raise ValueError(f"Part A focused identity: {case}/{proposal} {ident}, {det}")
        focused[proposal] = {"native_relative_l1": ident, "repeat_relative_l1": det}
    write(out / "focused.json", focused)
    if smoke:
        return focused

    saved = read(parent / "primary/provenance.json")
    values = {k: [] for k in VARIANTS}
    identities = []
    for i in range(b["replicates"]):
        packet = {}
        for proposal in "AC":
            packet.update(ctx.evaluate_cross(seed+i, proposal, light, b["spp"], b["batch"]))
        path = parent / "primary" / case / f"r{i:03d}.npz"
        hashes = {key.replace("\\", "/"): digest for key, digest in saved["output_sha256"].items()}
        sha_ok(path, hashes[f"{case}/{path.name}"])
        with np.load(path) as old:
            same = {"AA": l1_ratio(packet["AA"], old["AC"]), "CC": l1_ratio(packet["CC"], old["CC"])}
        np.savez(out / f"r{i:03d}.npz", **packet)
        if max(same.values()) > g["identity_relative_l1"]:
            raise ValueError(f"Part A native replay: {case}/r{i} {same}")
        identities.append(same)
        for k, x in packet.items():
            values[k].append(x)
        print(case, "cross replicate", i, flush=True)
    values = {k: np.stack(x).astype(np.float64) for k, x in values.items()}
    changes = {"fixed_f_A": ("AC", "AA"), "fixed_f_C": ("CC", "CA"),
               "fixed_p_A": ("CA", "AA"), "fixed_p_C": ("CC", "AC")}
    comparisons = {
        roi: {name: difference_statistics(values[x], values[y], ref, mask) for name, (x, y) in changes.items()}
        for roi, mask in masks.items() if mask.any()
    }
    physics = {}
    failures = []
    gates = {"precision_energy_half_width": P["part_d"]["gates"]["energy_half_width"],
             "precision_rms_half_width": P["part_d"]["gates"]["rms_half_width"]}
    for name in VARIANTS:
        paired = {"e1": fields[name[0]]["e1"], "e2": values[name], "h": fields["C"]["h"]}
        metrics, failed = event_metrics(paired, ref, masks, gates, compact=True)
        failures.extend({"arm": name, **r} for r in failed)
        physics[name] = metrics
    # Direct E1 is point-light evaluation, with a unique camera sample and no continuation proposal.
    direct = {
        arm: difference_statistics(fields[arm]["e1"], ref["e1"], ref, masks["objects"]) for arm in ("A", "C", "ER_DF")
    }
    object_mask = masks["objects"]
    response = {
        "identity": identities, "archived_baseline": baselines, "cross_E2_and_recombined_TOT": physics,
        "comparisons": comparisons, "fixed_p_response_projections": projections(values, object_mask),
        "proposal_variance_object_E2": {key: float(np.var(v[:, object_mask], axis=0, ddof=1).mean()) for key, v in values.items()},
        "direct_e1": direct, "precision_passed": not failures, "precision_failures": failures,
    }
    write(out / "metrics.json", response)
    return response


def judge(results, audit):
    primary = results[P["primary"]]
    comparisons = primary["comparisons"]["objects"]
    proposal = [comparisons[k] for k in ("fixed_f_A", "fixed_f_C")]
    response = [comparisons[k] for k in ("fixed_p_A", "fixed_p_C")]
    accepted = all(r["signed"]["lo"] >= -.0025 and r["signed"]["hi"] <= .0025 and r["rms"]["hi"] < .01
                   for r in proposal)
    spatial = min(x["rms"]["lo"] for x in response) > max(x["rms"]["hi"] for x in proposal)
    projections = primary["fixed_p_response_projections"]
    spatial &= len(projections) == 2 and all(r["lo"] >= .75 for r in projections.values())
    physical = all(r["passed"] for r in audit.values())
    precision = all(r["precision_passed"] for r in results.values())
    return {"sampling_audit_passed": physical, "proposal_practical_equivalence": accepted,
            "target_response_dominant": bool(spatial), "scene_precision_passed": precision,
            "qualifies_correction": bool(accepted and spatial and physical and precision)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    _, rows, materials, states = frozen_inputs()
    results = {}
    audit = {} if args.smoke else audit_states(rows, states)
    if audit:
        write(args.output / "operator_audit.json", audit)
    for case in P["cases"]:
        i, row = next((i, r) for i, r in enumerate(rows) if r["image"] == case)
        out = args.output / case
        out.mkdir()
        results[case] = cross_scene(PARENT, case, row, materials[i],
                                    {a: x[i] for a, x in states.items()}, out, args.smoke)
    if not args.smoke:
        write(args.output / "decision.json", judge(results, audit))
    write(args.output / "status.json", {
        "elapsed_seconds": time.monotonic()-started, "smoke": args.smoke,
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "protocol_sha256": H.C.sha(HERE / "protocol.json"), "host": "local", "cases": list(results),
    })
    print("PART A", "smoke" if args.smoke else judge(results, audit), flush=True)


if __name__ == "__main__":
    main()
