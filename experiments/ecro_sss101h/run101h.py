"""WL101H runner. One semantics per process (isolation).

python run101h.py construct WL101_ROOT OUT {native|candidate} {a1|metric} SEED [SEED ...]
python run101h.py camera    WL101_ROOT OUT {native|candidate} TABLE_DIR [TABLE_DIR ...]
"""
from pathlib import Path
import json
import platform
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ecro_sss101g"))
sys.path.insert(0, str(HERE))
argv = sys.argv[:]
sys.argv = sys.argv[:1]
import run_generalization as RG  # noqa: E402  WL101F inputs, WL101G helpers and paths
from run_oracle import A, array_hash, sha, write_json  # noqa: E402
from oracle_field import ExposedOracle, FirstSurface  # noqa: E402
from query_readout import QueryReadout  # noqa: E402
import gpu101g as GG  # noqa: E402
import tables101g as T  # noqa: E402
import semantics101h as SEM  # noqa: E402
import construct101h as CO  # noqa: E402

PROTOCOL = HERE / "protocol/qualification.json"
P = json.loads(PROTOCOL.read_text())
SMOKE = __import__("os").environ.get("WL101H_SMOKE") == "1"   # code-path check only: reduced scale, no archive replay
if SMOKE:
    P["construction"]["proposals_per_slot"] = 64
    P["camera"]["spp"] = 32
EXPERIMENT_CODE = ["experiments/ecro_sss101h/" + f for f in
                   ("run101h.py", "construct101h.py", "semantics101h.py", "instrument101h.py")] + \
                  ["experiments/ecro_sss101g/gpu101g.py", "experiments/ecro_sss101g/tables101g.py",
                   "experiments/ecro_sss101e/oracle_field.py", "experiments/ecro_sss101f/query_readout.py",
                   "experiments/ecro_sss101f/projections.py", "experiments/ecro_sss101f/run_attribution.py"]


class CandidateReadout(QueryReadout):
    def __init__(self, R, rt, camera):
        super().__init__(R, rt, camera)
        import types
        self.observed = types.MethodType(SEM.observed_trace(R), rt)


def setup(root, semantics):
    reader, learned, base, wl101e, prior, history, reference, manifest = RG.F101.inputs(root)
    R, rt = reader.R, reader.rt
    contact = SEM.Contact(R, rt, P["candidate"]["h_min_m"])
    if semantics == "candidate":
        contact = SEM.install(R, rt, P["candidate"]["h_min_m"])
        reader = CandidateReadout(R, rt, reader.cam)
    repo = root.parent
    manifest["experiment_code_sha256"] = {f: sha(repo / f) for f in EXPERIMENT_CODE if (repo / f).exists()}
    manifest["protocol_sha256"] = sha(PROTOCOL)
    manifest["semantics"] = semantics
    manifest["python"] = platform.python_version()
    return reader, learned, wl101e, prior, history, reference, manifest, contact


def construct_cmd(root, out, semantics, membership, seeds):
    reader, learned, wl101e, prior, history, reference, manifest, contact = setup(root, semantics)
    R, rt = reader.R, reader.rt
    out.mkdir(parents=True, exist_ok=False)
    geom = rt.inc.G
    b_m = P["membership"]["b_m_m"]
    if membership == "a1":
        oracle = GG.amended_oracle(ExposedOracle, 8e-6)(R, rt)
    else:
        lengths = (R.F(np.linalg.norm(geom.PE1, axis=1).astype(np.float32)),
                   R.F(np.linalg.norm(geom.PE2, axis=1).astype(np.float32)))
        oracle = CO.metric_oracle(ExposedOracle, b_m, lengths)(R, rt)
    solids = CO.Solids(R, rt, A["geometry"])
    run = {**manifest, "command": "construct", "membership": membership, "seeds": seeds, "replicas": [], "files": {}}
    C = P["construction"]
    tol = P["gates"]["address_coordinate_tolerance"]

    def store(name, **arrays):
        np.savez_compressed(out / name, **arrays)
        run["files"][name] = {"sha256": sha(out / name), "bytes": (out / name).stat().st_size}

    # Camera independence of the construction under these semantics (V0 vs H1 reader, one launch).
    cams = RG.view_cameras(reader.cam)
    other = (CandidateReadout if semantics == "candidate" else QueryReadout)(R, rt, cams["H1"])
    a, ma = CO.construct(reader, oracle, solids, contact, semantics, P["construction"]["independence_seed"], 32, 32, tol, b_m, launches=1)
    b, mb = CO.construct(reader, oracle, solids, contact, semantics, P["construction"]["independence_seed"], 32, 32, tol, b_m, launches=1)
    c, mc = CO.construct(other, oracle, solids, contact, semantics, P["construction"]["independence_seed"], 32, 32, tol, b_m, launches=1)
    ints = [k for k in a if k in ("n", "v", "v_small") or k.startswith(("k_", "vl_", "slot_"))]
    reals = [k for k in a if k not in ints]
    def compare(x, y, mx, my):
        exact = all(np.array_equal(x[k], y[k]) for k in ints) and mx["membership"] == my["membership"] and \
            mx["classes"] == my["classes"] and mx["labels"] == my["labels"]
        rel = max(float(np.max(np.abs(x[k] - y[k])) / max(float(np.max(np.abs(x[k]))), 1e-30)) for k in reals)
        return {"integer_columns_and_counts_identical": exact, "max_relative_real_difference": rel}
    run["camera_independence"] = {"same_reader_repeat": compare(a, b, ma, mb), "V0_vs_H1_reader": compare(a, c, ma, mc)}
    unregistered = []
    for i, seed in enumerate(seeds):
        stamp = time.perf_counter()
        packet, meta = CO.construct(reader, oracle, solids, contact, semantics, seed, C["proposals_per_slot"],
                                    C["launch_proposals"], tol, b_m, unregistered=unregistered)
        meta["seconds"] = time.perf_counter() - stamp
        tabs = T.independent_tables(packet, names=("Un", "V", "G", "S_G"))
        store(f"construction_{seed}.npz", **packet)
        store(f"tables_{seed}.npz", **{k: v[0] for k, v in tabs.items()}, **{f"{k}:native": v[1] for k, v in tabs.items()})
        run["replicas"].append(meta)
        print(json.dumps(meta), flush=True)
        write_json(out / "progress.json", run)
    run["unregistered_events"] = unregistered
    write_json(out / "run_manifest.json", run)
    print("construct complete", flush=True)


def camera_cmd(root, out, semantics, table_dirs):
    reader, learned, wl101e, prior, history, reference, manifest, contact = setup(root, semantics)
    R, rt = reader.R, reader.rt
    out.mkdir(parents=True, exist_ok=False)
    run = {**manifest, "command": "camera", "table_dirs": [str(d) for d in table_dirs], "streams": [], "files": {}}
    G = P["gates"]
    cams = RG.view_cameras(reader.cam)
    # Event/query consistency and LOW invariance of this semantics at every view camera (64^2, 1 spp).
    run["consistency"] = {}
    for vid, cam in cams.items():
        rd = reader if vid == "V0" else (CandidateReadout if semantics == "candidate" else QueryReadout)(R, rt, cam)
        seed = P["camera"]["smoke_seed_start"] + list(cams).index(vid)
        run["consistency"][vid] = {"smoke": rd.smoke(seed, FirstSurface, G), "low": GG.low_invariance(rd, learned, seed)}
    arms = GG.DeviceArms(reader)
    base = root.parent / "results/ecro_sss101e/recovery2"
    for row in wl101e["oracle_tables"]:
        with np.load(base / row["file"]) as f:
            arms.add(f"U_E_{row['file'].split('_')[1].split('.')[0]}", f["B"], (f["valid"] > 0)[..., None], "slot")
    for d in table_dirs:
        man = json.loads((d / "run_manifest.json").read_text())
        tag = "N" if man["semantics"] == "native" else "C"
        for i, seed in enumerate(man["seeds"]):
            name = f"tables_{seed}.npz"
            assert sha(d / name) == man["files"][name]["sha256"]
            with np.load(d / name) as f:
                for fam, kind in (("Un", "slot"), ("V", "slot"), ("G", "slot"), ("S_G", "S")):
                    arms.add(f"{fam}{tag}_{i}", f[fam], f[f"{fam}:native"], kind)
    run["arms"] = list(arms.arms)
    for j, seed in enumerate(P["camera"]["V0_seeds"]):
        stamp = time.perf_counter()
        images, meta = GG.camera_pass(reader, FirstSurface, seed, P["camera"]["spp"], 32, learned, arms,
                                      rt.inc.G.PSID, G["address_coordinate_tolerance"], None)
        meta["seconds"] = time.perf_counter() - stamp
        if semantics == "native" and not SMOKE:
            archived = prior / history["matched"][j]["path"]
            with np.load(archived) as f, np.load(reference / f"REF_0_{j}.npz") as ref:
                meta["replay"] = {"learned_low_rel_l1": {k: RG.rel_l1(images[k], f[k]) for k in ("e1", "e2")},
                                  "query_vs_archived_reference_rel_l1": RG.rel_l1(images["query"], ref["h"])}
            assert max(meta["replay"]["learned_low_rel_l1"].values()) <= G["low_relative_l1"], meta
            assert meta["replay"]["query_vs_archived_reference_rel_l1"] <= G["matched_rgb_relative_l1"], meta
        name = f"camera_V0_{j}.npz"
        np.savez_compressed(out / name, **images)
        meta.update(file=name, sha256=sha(out / name))
        run["streams"].append(meta)
        write_json(out / "progress.json", run)
        del images
    write_json(out / "run_manifest.json", run)
    print("camera complete", flush=True)


if __name__ == "__main__":
    cmd, root, out, semantics = argv[1], Path(argv[2]).resolve(), Path(argv[3]).resolve(), argv[4]
    if cmd == "construct":
        construct_cmd(root, out, semantics, argv[5], [int(s) for s in argv[6:]])
    else:
        camera_cmd(root, out, semantics, [Path(d).resolve() for d in argv[5:]])
