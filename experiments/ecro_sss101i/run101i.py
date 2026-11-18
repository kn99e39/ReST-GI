"""WL101I construction runner (contract v2 instruments). One semantics per process.

python run101i.py WL101_ROOT OUT {native|candidate} {a1|metric} H_MIN_OVERRIDE|- SEED [SEED ...]
H_MIN_OVERRIDE is only for the declared positive control (expanded singular region); '-' keeps the
frozen WL101H candidate h_min.
"""
from pathlib import Path
import json
import platform
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ecro_sss101g"))
sys.path.insert(0, str(HERE.parent / "ecro_sss101h"))
sys.path.insert(0, str(HERE))
argv = sys.argv[:]
sys.argv = sys.argv[:1]
import run_generalization as RG  # noqa: E402
from run_oracle import A, sha, write_json  # noqa: E402
from oracle_field import ExposedOracle  # noqa: E402
from query_readout import QueryReadout  # noqa: E402
import gpu101g as GG  # noqa: E402
import semantics101h as SEM  # noqa: E402
import construct101h as CO_H  # noqa: E402  (metric oracle, solids)
import construct101i as CO  # noqa: E402

PROTOCOL = HERE / "protocol/contract_v2.json"
P = json.loads(PROTOCOL.read_text())
SMOKE = __import__("os").environ.get("WL101I_SMOKE") == "1"   # code-path check on development seeds only
if SMOKE:
    P["proposals_per_slot"] = 64
    P["final_seeds"] = [10204101, 10204102, 10204103, 10204104]
H_PROTOCOL = json.loads((HERE.parent / "ecro_sss101h/protocol/qualification.json").read_text())
CODE = ["experiments/ecro_sss101i/" + f for f in ("run101i.py", "construct101i.py")] + \
       ["experiments/ecro_sss101h/" + f for f in ("semantics101h.py", "construct101h.py", "run101h.py", "instrument101h.py")] + \
       ["experiments/ecro_sss101g/gpu101g.py", "experiments/ecro_sss101g/tables101g.py", "experiments/ecro_sss101e/oracle_field.py",
        "experiments/ecro_sss101f/query_readout.py", "experiments/ecro_sss101f/projections.py", "experiments/ecro_sss101f/run_attribution.py"]


def main(root, out, semantics, membership, h_override, seeds):
    reader, learned, base, wl101e, prior, history, reference, manifest = RG.F101.inputs(root)
    R, rt = reader.R, reader.rt
    h_min = H_PROTOCOL["candidate"]["h_min_m"] if h_override == "-" else float(h_override)
    contact = SEM.Contact(R, rt, h_min)
    if semantics == "candidate":
        contact = SEM.install(R, rt, h_min)
    out.mkdir(parents=True, exist_ok=False)
    b_m = H_PROTOCOL["membership"]["b_m_m"]
    geom = rt.inc.G
    if membership == "a1":
        oracle = GG.amended_oracle(ExposedOracle, 8e-6)(R, rt)
    else:
        lengths = (R.F(np.linalg.norm(geom.PE1, axis=1).astype(np.float32)), R.F(np.linalg.norm(geom.PE2, axis=1).astype(np.float32)))
        oracle = CO_H.metric_oracle(ExposedOracle, b_m, lengths)(R, rt)
    solids = CO_H.Solids(R, rt, A["geometry"])
    repo = root.parent
    run = {**manifest, "semantics": semantics, "membership": membership, "h_min_m": h_min, "h_min_override": h_override != "-",
           "seeds": seeds, "protocol_sha256": sha(PROTOCOL), "python": platform.python_version(),
           "experiment_code_sha256": {f: sha(repo / f) for f in CODE if (repo / f).exists()}, "replicas": [], "files": {}}
    tol = P["address_coordinate_tolerance"]
    if semantics == "candidate" and h_override == "-":
        cams = RG.view_cameras(reader.cam)
        cls = QueryReadout
        other = cls(R, rt, cams["H1"])
        sd = P["independence_seed"]
        a, ma = CO.construct(reader, oracle, solids, contact, semantics, sd, 32, 32, tol, b_m, launches=1)
        b, mb = CO.construct(reader, oracle, solids, contact, semantics, sd, 32, 32, tol, b_m, launches=1)
        c, mc = CO.construct(other, oracle, solids, contact, semantics, sd, 32, 32, tol, b_m, launches=1)
        ints = [k for k in a if k in ("n", "v", "sh_n", "sh_v") or k.startswith(("k_", "vl_", "slot_", "nf"))]
        reals = [k for k in a if k not in ints]
        def compare(x, y, mx, my):
            exact = all(np.array_equal(x[k], y[k]) for k in ints) and mx["membership"] == my["membership"] and mx["classes"] == my["classes"]
            rel = max(float(np.max(np.abs(x[k] - y[k])) / max(float(np.max(np.abs(x[k]))), 1e-30)) for k in reals)
            return {"integer_columns_and_counts_identical": exact, "max_relative_real_difference": rel}
        run["camera_independence"] = {"same_reader_repeat": compare(a, b, ma, mb), "V0_vs_H1_reader": compare(a, c, ma, mc)}
    if semantics == "candidate" and h_override == "-":
        # Instrument consistency of the candidate semantics, re-measured here (not reused from WL101H).
        import run101h as RH
        import gpu101g as G2
        from oracle_field import FirstSurface
        gates = {k: P["smoke_gates"][k] for k in P["smoke_gates"]}
        run["consistency"] = {}
        for k, (vid, cam) in enumerate(RG.view_cameras(reader.cam).items()):
            rd = RH.CandidateReadout(R, rt, cam)
            sd = P["smoke_seed_start"] + k
            run["consistency"][vid] = {"smoke": rd.smoke(sd, FirstSurface, gates), "low": G2.low_invariance(rd, learned, sd)}
    unregistered = []
    for seed in seeds:
        stamp = time.perf_counter()
        packet, meta = CO.construct(reader, oracle, solids, contact, semantics, seed, P["proposals_per_slot"], P["launch_proposals"],
                                    tol, b_m, unregistered=unregistered)
        meta["seconds"] = time.perf_counter() - stamp
        name = f"construction_{seed}.npz"
        np.savez_compressed(out / name, **packet)
        run["files"][name] = {"sha256": sha(out / name), "bytes": (out / name).stat().st_size}
        run["replicas"].append(meta)
        print(json.dumps(meta), flush=True)
        write_json(out / "progress.json", run)
    run["unregistered_events"] = unregistered
    write_json(out / "run_manifest.json", run)
    print("construct complete", flush=True)


if __name__ == "__main__":
    main(Path(argv[1]).resolve(), Path(argv[2]).resolve(), argv[3], argv[4], argv[5], [int(s) for s in argv[6:]])
