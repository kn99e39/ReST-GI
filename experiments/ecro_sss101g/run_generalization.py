"""Frozen WL101G cross-view, camera-independent HIGH projection run.

Stage 0  views, gbuffers, per-view event/query consistency and LOW invariance
Stage 1  camera-independent construction (Un, V, G, refinements, adaptive layout) and its validation
Stage 1b per-view lookup consistency
Stage 2  per-view camera-conditioned controls P_v / S_v
         -> every construction table is hashed into the manifest before Stage 3
Stage 3  two paired 4,096-spp evaluation streams per view with every arm
Stage 4  isolated lookup timing
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
for sibling in ("ecro_sss101a", "ecro_sss101d", "ecro_sss101e", "ecro_sss101f"):
    sys.path.insert(0, str(HERE.parent / sibling))
sys.path.insert(0, str(HERE))

import run_attribution as F101  # noqa: E402  WL101F frozen input validation
from oracle_field import ExposedOracle, FirstSurface  # noqa: E402
from run_oracle import A, array_hash, foreign_gpu, gpu_query, rel_l1, sha, wait_for_gpu, write_json  # noqa: E402
from projections import make_tables  # noqa: E402
from query_readout import QueryReadout  # noqa: E402
from regions101g import contact_valid, view_cameras, view_regions  # noqa: E402
import gpu101g as GG  # noqa: E402
import selftest101g  # noqa: E402
import tables101g as T  # noqa: E402

PROTOCOL = HERE / "protocol/generalization.json"
AMENDMENT1 = HERE / "protocol/amendment1.json"
AMENDMENT2 = HERE / "protocol/amendment2.json"
P = json.loads(PROTOCOL.read_text())
G = P["gates"]
CI = P["camera_independent_construction"]
VIEW_INDEX = P["camera_conditioned_controls"]["view_index"]
def load_baselines(root, wl101e):
    base = root.parent / "results/ecro_sss101e/recovery2"
    f101 = root.parent / P["baselines"]["wl101f_manifest_file"]
    assert sha(f101) == P["baselines"]["wl101f_manifest_sha256"]
    wl101f = json.loads(f101.read_text())
    assert wl101f["protocol_sha256"] == P["baselines"]["wl101f_protocol_sha256"]
    fdir = f101.parent
    uniform, uniform_native, uniform_stats = [], [], []
    for row in wl101e["oracle_tables"]:
        path = base / row["file"]
        assert sha(path) == row["sha256"]
        with np.load(path) as f:
            assert array_hash(f["B"]) == row["array_sha256"]
            uniform.append(f["B"].copy())
            uniform_native.append(np.broadcast_to((f["valid"] > 0)[..., None], f["B"].shape).copy())
            uniform_stats.append({"mean": f["B"].astype(np.float64), "var": f["sample_variance"].astype(np.float64),
                                  "n": f["valid"].astype(np.float64)})
    ftables = {}
    for name in ("tables_B.npz", "tables_C.npz"):
        assert sha(fdir / name) == wl101f["files"][name]["sha256"]
        with np.load(fdir / name) as f:
            ftables.update({k: f[k].copy() for k in f.files})
    return base, fdir, wl101f, uniform, uniform_native, uniform_stats, ftables


def validate_construction(packet, meta, stats):
    m, seg = meta["membership"], meta["segments"]
    eligible = sum(m.values()) - m["buried"] - m["back"]
    assert m["invalid"] == 0, ("Construction membership", m)
    assert m["tie"] <= G["uniform_audit_tie_fraction"] * eligible, ("Construction ties", m)
    assert meta["amendment1_ties"] <= 1e-6 * eligible, ("Amendment 1 recurrence stop", meta["amendment1_ties"], m)
    assert meta["coordinate_excursion"] <= G["address_coordinate_tolerance"]
    assert seg["unexplained"] == 0, ("Containment", seg)
    assert seg["seam"] <= 2e-5 * m["valid"], ("Seam leaks", seg)
    assert seg["tangency"] <= 2e-5 * m["valid"], ("Tangency misses", seg)
    return {"tie_fraction_of_eligible": m["tie"] / eligible, "amendment1_fraction_of_eligible": meta["amendment1_ties"] / eligible, "seam_fraction": seg["seam"] / m["valid"],
            "tangency_fraction": seg["tangency"] / m["valid"], "back_receiver_fraction": seg["back_receiver"] / m["valid"]}


def cos_normalization(raws, sbase):
    rows = []
    for k in range(4):
        c = sum(r["c"][:sbase * 64].reshape(sbase, 64, 16)[:, 16 * k:16 * (k + 1)].sum() for r in raws)
        n = sum(r["n"][:sbase * 64].reshape(sbase, 64, 16)[:, 16 * k:16 * (k + 1)].sum() for r in raws)
        mean, expected = c / n, (2 * k + 1) / 8
        se = 0.25 / np.sqrt(12) / np.sqrt(n)
        rows.append({"ring": k, "mean_cos": float(mean), "expected": expected, "standard_errors": float((mean - expected) / se),
                     "samples": float(n)})
    assert all(abs(r["standard_errors"]) <= 5 for r in rows), rows
    return rows


def uniform_identity(raws, stats):
    zs = []
    for raw, ue in zip(raws, stats):
        n = raw["n"].sum((1, 2))
        L, LL = raw["L"].sum((1, 2)), raw["LL"].sum((1, 2))
        mean = np.divide(L, n[:, None], out=np.zeros_like(L), where=n[:, None] > 0)
        var = np.divide(LL - L * mean, (n - 1)[:, None], out=np.zeros_like(L), where=n[:, None] > 1)
        un, vn = ue["mean"].reshape(-1, 3), ue["var"].reshape(-1, 3)
        nn = ue["n"].reshape(-1)
        ok = (n >= 32) & (nn >= 32)
        se = np.sqrt(var[ok] / n[ok, None] + vn[ok] / nn[ok, None])
        good = se > 0
        zs.append(((mean[ok] - un[ok])[good] / se[good]).ravel())
    z = np.concatenate(zs)
    result = {"entries": int(z.size), "median_abs_z": float(np.median(np.abs(z))),
              "fraction_abs_z_gt_6": float(np.mean(np.abs(z) > 6)), "rms_z": float(np.sqrt(np.mean(z * z)))}
    assert 0.6745 * 0.85 <= result["median_abs_z"] <= 0.6745 * 1.15, ("Uniform identity", result)
    assert result["fraction_abs_z_gt_6"] <= 0.005, ("Uniform identity", result)
    return result


def run(root, out, commit, smoke):
    reader0, learned, base_unused, wl101e, prior, history, reference, manifest = F101.inputs(root)
    R, rt = reader0.R, reader0.rt
    out.mkdir(parents=True, exist_ok=False)
    run = {**manifest, "commit": commit, "protocol_sha256": sha(PROTOCOL), "amendment1_sha256": sha(AMENDMENT1),
           "amendment2_sha256": sha(AMENDMENT2),
           "smoke_mode": smoke,
           "stages": {}, "files": {}, "gpu_memory_observations_mib": [], "selftest": None}
    e_base, fdir, wl101f, uniform, uniform_native, uniform_stats, ftables = load_baselines(root, wl101e)
    run["baseline_manifests"] = {"wl101e": sha(e_base / "run_manifest.json"), "wl101f": sha(fdir / "run_manifest.json")}
    run["selftest"] = selftest101g.run_all()
    geom = reader0.geom

    def save():
        write_json(out / "progress.json", run)

    def store(filename, **arrays):
        path = out / filename
        np.savez_compressed(path, **arrays)
        row = {"file": filename, "sha256": sha(path), "bytes": path.stat().st_size}
        run["files"][filename] = row
        return row

    peak = 0
    def monitor(message=None):
        nonlocal peak
        if foreign_gpu():
            wait_for_gpu()
        memory = int(gpu_query("gpu", "memory.used").split()[0])
        peak = max(peak, memory)
        if message is not None:
            print(f"{message} ({memory} MiB)", flush=True)

    def timed(label, fn, *args, **kwargs):
        nonlocal peak
        peak = 0
        monitor(f"start {label}")
        R.sync()
        stamp = time.perf_counter()
        result = fn(*args, **kwargs)
        R.sync()
        seconds = time.perf_counter() - stamp
        monitor(f"done {label} {seconds:.2f}s")
        run["gpu_memory_observations_mib"].append({"stage": label, "max_observed_mib": peak, "seconds": seconds})
        return result, seconds

    # ---------------------------------------------------------------- stage 0
    cams = view_cameras(reader0.cam)
    readers = {"V0": reader0}
    run["views"] = {}
    regions = {}
    for k, (vid, cam) in enumerate(cams.items()):
        if vid != "V0":
            readers[vid] = QueryReadout(R, rt, cam)
        gfile = f"gbuffer_{vid}.npz"
        store(gfile, **rt.gbuffer(cam, cam["res"]))
        row = {"camera": cam, "index": VIEW_INDEX[vid]}
        if vid == "V0":
            with np.load(reference / "MAP_0.npz") as a, np.load(out / gfile) as b:
                assert np.array_equal(a["ok"], b["ok"]) and np.array_equal(a["sid"], b["sid"]), "V0 gbuffer ids"
                dp = float(np.max(np.abs(a["p"][a["ok"]] - b["p"][b["ok"]])))
                assert dp <= G["gbuffer_position_m"], ("V0 gbuffer position", dp)
                row["gbuffer_reproduction_max_position_m"] = dp
        reg, bands = view_regions(reference / "MAP_0.npz" if vid == "V0" else out / gfile, vid == "V0")
        regions[vid] = reg
        row["region_pixels"] = {name: int(m.sum()) for name, m in reg.items()}
        row["band_pixels"] = [int(b.sum()) for b in bands]
        row["contact_valid"] = contact_valid(reg)
        seed = P["smoke_seeds_start"] + VIEW_INDEX[vid]
        row["smoke"] = readers[vid].smoke(seed, FirstSurface, G)
        row["low_invariance"] = GG.low_invariance(readers[vid], learned, seed)
        run["views"][vid] = row
        print(vid, row["region_pixels"], row["contact_valid"], flush=True)
        save()
    run["stages"]["0"] = {"passed": True}

    # ---------------------------------------------------------------- stage 1
    oracle = GG.amended_oracle(ExposedOracle, json.loads(AMENDMENT1.read_text())["window"])(R, rt)
    cfg = {**CI, "gates": G}
    det_seed = 10203001
    first, m1 = GG.construct(readers["V0"], oracle, cfg, det_seed, None, launches=1)
    again, m2 = GG.construct(readers["V0"], oracle, cfg, det_seed, None, launches=1)
    other, m3 = GG.construct(readers["H1"], oracle, cfg, det_seed, None, launches=1)
    def compare(a, b, ma, mb):
        exact = all(np.array_equal(a[k], b[k]) for k in ("n", "v")) and \
            ma["membership"] == mb["membership"] and ma["segments"] == mb["segments"]
        rel = max(float(np.max(np.abs(a[k] - b[k])) / max(float(np.max(np.abs(a[k]))), 1e-30))
                  for k in ("c", "g", "L", "LL", "vL", "gL"))
        return {"integer_columns_and_counts_identical": exact, "max_relative_real_difference": rel}
    independence = {"same_reader_repeat": compare(first, again, m1, m2),
                    "V0_vs_H1_reader": compare(first, other, m1, m3)}
    for row in independence.values():
        assert row["integer_columns_and_counts_identical"] and row["max_relative_real_difference"] <= 1e-5, \
            ("Camera-independent construction depends on the reader camera", independence)
    run["stages"]["1"] = {"camera_independence": independence, "replicas": []}
    launches = 2 if smoke else None
    seeds = CI["replica_seeds"][:1] if smoke else CI["replica_seeds"]
    raws = []
    for i, seed in enumerate(seeds):
        (packet, meta), seconds = timed(f"construction_{i}", GG.construct, readers["V0"], oracle, cfg, seed, monitor, launches)
        meta.update(seconds=seconds, **validate_construction(packet, meta, uniform_stats[i]))
        meta.update(store(f"construction_{i}.npz", **packet))
        run["stages"]["1"]["replicas"].append(meta)
        raws.append(packet)
        save()
    proposals = CI["launch_proposals"] * (2 if smoke else CI["proposals_per_slot"] // CI["launch_proposals"])
    if not smoke:
        run["stages"]["1"]["cos_normalization"] = cos_normalization(raws, geom.SBASE)
        run["stages"]["1"]["uniform_identity"] = uniform_identity(raws, uniform_stats)
    measure = T.slot_measure(geom.PSID, geom.PAREA)
    if len(raws) == 4:
        energy = T.refinement_energy(raws, measure, proposals)
    else:
        energy = T.refinement_energy([raws[0]] * 4, measure, proposals)
    refined = T.select_refined(energy, 0.10)
    offsets, rows = T.adaptive_layout(refined)
    independent = []
    for i, packet in enumerate(raws):
        tabs = T.independent_tables(packet)
        g_t, g_n = tabs["G"]
        sg_t, sg_n = tabs["S_G"]
        tabs["S_Gad"] = T.adaptive_table(g_t, g_n, sg_t, sg_n, refined)[:2]
        independent.append(tabs)
    positive = energy[energy > 0]
    run["stages"]["1"]["adaptive"] = {
        "refined_slots": int(refined.sum()), "rows": rows, "size_ratio": rows / len(refined),
        "captured_positive_energy_fraction": float(energy[refined & (energy > 0)].sum() / positive.sum()),
        "refined_by_shape": {int(sid): int(np.count_nonzero(refined.reshape(-1, 64)[geom.PSID == sid]))
                             for sid in np.unique(geom.PSID)}}
    store("tables_independent.npz", refined=refined, offsets=offsets, energy=energy,
          **{f"{name}_{i}": v[0] for i, tabs in enumerate(independent) for name, v in tabs.items()},
          **{f"{name}_{i}:native": v[1] for i, tabs in enumerate(independent) for name, v in tabs.items()})
    if not smoke:
        g = np.stack([t["G"][0] for t in independent]).astype(np.float64)
        half = 0.5 * (g[:2].mean(0) - g[2:].mean(0))
        run["stages"]["1"]["G_half_difference_rms_over_table_rms"] = float(np.sqrt(np.mean(half ** 2)) / np.sqrt(np.mean(g.mean(0) ** 2)))
    run["stages"]["1"]["passed"] = True
    save()

    run["stages"]["1b"] = {}
    for vid in cams:
        run["stages"]["1b"][vid] = GG.lookup_checks(readers[vid], learned, P["smoke_seeds_start"] + 50 + VIEW_INDEX[vid],
                                                    G["address_coordinate_tolerance"], refined, offsets)
    save()

    # ---------------------------------------------------------------- stage 2
    run["stages"]["2"] = {"views": {}}
    per_view = {}
    spp_v = 32 if smoke else 64
    for vid in (["V0"] if smoke else cams):
        vi = VIEW_INDEX[vid]
        per_view[vid] = []
        rows_v = []
        for i in range(1 if smoke else 4):
            (packet, meta), seconds = timed(f"view_{vid}_{i}", GG.train_view, readers[vid], FirstSurface,
                                            10203200 + 10 * vi + i, 10203300 + 10 * vi + i, spp_v, 32,
                                            G["address_coordinate_tolerance"], monitor)
            tabs = make_tables(packet, uniform[i], ("P", "S"))
            per_view[vid].append(tabs)
            meta.update(seconds=seconds, **store(f"view_usage_{vid}_{i}.npz", **packet))
            rows_v.append(meta)
        run["stages"]["2"]["views"][vid] = rows_v
        save()
    store("tables_per_view.npz", **{f"{kind}v_{vid}_{i}": tabs[kind][0] for vid, lst in per_view.items()
                                    for i, tabs in enumerate(lst) for kind in ("P", "S")},
          **{f"{kind}v_{vid}_{i}:native": tabs[kind][1] for vid, lst in per_view.items()
             for i, tabs in enumerate(lst) for kind in ("P", "S")})
    run["construction_frozen_before_evaluation"] = {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": {k: run["files"][k]["sha256"] for k in ("tables_independent.npz", "tables_per_view.npz")}}
    save()

    # ---------------------------------------------------------------- stage 3
    arms = GG.DeviceArms(reader0, (offsets, refined))
    nrep = len(raws)
    for i in range(nrep if not smoke else 1):
        arms.add(f"U_E_{i}", uniform[i], uniform_native[i], "slot")
        for fam, kind in (("P", "slot"), ("Q", "slot"), ("S", "S"), ("A", "A"), ("SA", "SA")):
            arms.add(f"{fam}_F_{i}", ftables[f"{fam}_{i}"], ftables[f"{fam}_{i}:native"], kind)
        tabs = independent[i]
        for fam, kind in (("Un", "slot"), ("V", "slot"), ("G", "slot"), ("S_G", "S"), ("A_G", "A"),
                          ("SA_G", "SA"), ("S_U", "S"), ("S_Gad", "AD")):
            arms.add(f"{fam}_{i}", tabs[fam][0], tabs[fam][1], kind)
        for vid, lst in per_view.items():
            if i < len(lst):
                arms.add(f"Pv_{vid}_{i}", lst[i]["P"][0], lst[i]["P"][1], "slot")
                arms.add(f"Sv_{vid}_{i}", lst[i]["S"][0], lst[i]["S"][1], "S")
    run["arms"] = list(arms.arms)
    run["stages"]["3"] = {"streams": []}
    spp = 32 if smoke else 4096
    for vid in (["V0"] if smoke else cams):
        vi = VIEW_INDEX[vid]
        for j in range(1 if smoke else 2):
            seed = P["evaluation_seeds"][vid][j]
            (images, meta), seconds = timed(f"eval_{vid}_{j}", GG.camera_pass, readers[vid], FirstSurface, seed, spp, 32,
                                            learned, arms, geom.PSID, G["address_coordinate_tolerance"], monitor)
            meta.update(view=vid, stream=j, seconds=seconds)
            if vid == "V0" and not smoke:
                meta["replay"] = replay_v0(images, j, seed, history, prior, reference, e_base, wl101e, fdir, wl101f, regions["V0"])
            meta.update(store(f"camera_{vid}_{j}.npz", **images))
            run["stages"]["3"]["streams"].append(meta)
            save()
            del images

    # ---------------------------------------------------------------- stage 4
    run["stages"]["4"], _ = timed("benchmark", GG.benchmark, reader0, arms, P["benchmark_seed"], G["address_coordinate_tolerance"])
    run["table_bytes"] = {
        "coarse_rgb_float32": int(rt.N * 64 * 3 * 4), "spatial_2x2": int(rt.N * 64 * 4 * 3 * 4),
        "angular_2x2": int(rt.N * 64 * 4 * 3 * 4), "joint_4x4": int(rt.N * 64 * 16 * 3 * 4),
        "adaptive_rows": int(rows * 3 * 4), "adaptive_index_offsets_uint32_plus_flag": int(rt.N * 64 * 5)}
    run["completed"] = True
    write_json(out / "run_manifest.json", run)
    print("WL101G run complete", flush=True)


def replay_v0(images, j, seed, history, prior, reference, e_base, wl101e, fdir, wl101f, reg):
    """V0 stream reproduces the archived physical HIGH, learned readout, LOW and WL101E/F readouts."""
    result = {}
    archived = prior / history["matched"][j]["path"]
    assert sha(archived) == history["matched"][j]["sha256"]
    with np.load(archived) as f:
        result["learned_low_rel_l1"] = {k: rel_l1(images[k], f[k]) for k in ("e1", "e2")}
        result["learned_high_rel_l1"] = rel_l1(images["learned"], f["hl"])
    with np.load(reference / f"REF_0_{j}.npz") as f:
        result["physical_low_rel_l1"] = {k: rel_l1(images[k], f[k]) for k in ("e1", "e2")}
        result["query_vs_reference_rel_l1"] = rel_l1(images["query"], f["h"])
        luma = np.array([0.2126, 0.7152, 0.0722])
        endpoints = [float((images["query"] @ luma)[reg[m]].mean() - (f["h"] @ luma)[reg[m]].mean()) for m in ("near", "flank")]
        result["query_endpoint_differences"] = endpoints
    replays = {}
    for row in (r for r in wl101e["readouts"] if r["camera_half"] == j):
        with np.load(e_base / row["file"]) as f:
            replays[f"U_E_{row['oracle_index']}"] = rel_l1(images[f"U_E_{row['oracle_index']}"], f["hl"])
    for stage, fams in (("B", ("P", "Q")), ("C", ("S", "A", "SA"))):
        name = f"camera_{stage}_{seed}.npz"
        assert sha(fdir / name) == wl101f["files"][name]["sha256"]
        with np.load(fdir / name) as f:
            for fam in fams:
                for i in range(4):
                    replays[f"{fam}_F_{i}"] = rel_l1(images[f"{fam}_F_{i}"], f[f"{fam}_{i}"])
    result["wl101ef_readout_replay_rel_l1"] = replays
    assert max(*result["learned_low_rel_l1"].values(), *result["physical_low_rel_l1"].values()) <= G["low_relative_l1"], result
    assert max(result["learned_high_rel_l1"], result["query_vs_reference_rel_l1"]) <= G["matched_rgb_relative_l1"], result
    assert max(abs(x) for x in endpoints) <= G["matched_roi_max_absolute"], result
    assert max(replays.values()) <= G["wl101ef_readout_replay_relative_l1"], result
    return result


if __name__ == "__main__":
    arg = argparse.ArgumentParser(description=__doc__)
    arg.add_argument("wl101", type=Path)
    arg.add_argument("output", type=Path)
    arg.add_argument("commit")
    arg.add_argument("--smoke", action="store_true")
    a = arg.parse_args()
    try:
        run(a.wl101.resolve(), a.output.resolve(), a.commit, a.smoke)
    except AssertionError as error:
        print("STOP:", error, flush=True)
        raise
