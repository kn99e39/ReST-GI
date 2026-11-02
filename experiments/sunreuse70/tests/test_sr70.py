"""WL70 focused tests (protocol sunreuse70_v1 tests list). Small photon counts on development seeds (90xx), never evidence seeds.
Run on LabServer69 from experiments/fnto_k55 with the 8dna26 venv (plain runner; pytest is not installed there):
    python ../sunreuse70/tests/test_sr70.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import sr70_state as SS  # noqa: E402
import sr70_scenes as SC  # noqa: E402
import sr70_stats as ST  # noqa: E402
import k55_common as KCm  # noqa: E402
import k55_ops as KO  # noqa: E402
import k55_stageB as SB  # noqa: E402
import p53_tracer as TR  # noqa: E402

N_DEV = 1_000_000
_CACHE = {}


def keys():
    return sorted(set(SB.cam_dirs_of(SC.state("T1")[0], SB.SCENE_CAMS["floor_c4"])))


def cache():
    if "st" not in _CACHE:
        sun, sts = SS.sun_term_rec(SC.state("T0")[0], keys(), n_sun=N_DEV, seed=9001); _CACHE["st"] = sts[0]; _CACHE["sun"] = sun
    return _CACHE["st"]


def test_rigid_material_correspondence():
    m0, V0 = SC.state("T0")[0]["comps"][0][2:]; m1, V1 = SC.state("T1")[0]["comps"][0][2:]
    f0, c0, p0 = KO.refined(m0, V0, 4); f1, c1, p1 = KO.refined(m1, V1, 4)
    assert SS.mesh_hash(m0) == SS.mesh_hash(m1) and np.array_equal(p0, p1) and np.array_equal(f0["tri_port"], f1["tri_port"])
    assert np.array_equal(f0["V"][f0["F"]], f1["V"][f1["F"]])                          # identical rest (material) corners per triangle
    d = c1[f1["F"]] - c0[f0["F"]]; assert np.abs(d - d.reshape(-1, 3)[0]).max() < 1e-12   # current corners differ by one translation
    R, t, res = SS.rigid_fit(m1["V"], V1); assert res < 1e-12 and np.abs(R - np.eye(3)).max() < 1e-12


def test_source_replica_bitwise():
    for name in ("T0", "T1"):
        sd = SC.state(name)[0]; ss = KCm.classify_sceneset(sd); n, k, m, V = sd["comps"][0]; fm, VC, _ = KO.refined(m, V, 4)
        dom = KO.DomainX(fm, VC); s, E = sd["lights"][0]
        g1 = torch.Generator(device="cuda"); g1.manual_seed(9002); g2 = torch.Generator(device="cuda"); g2.manual_seed(9002)
        a = TR.sun_source(dom, s, E, 200_000, g1, shadow=ss.isect); b = SS.sun_source_rec(dom, s, E, 200_000, g2, ss.isect)
        assert all(torch.equal(x, y) for x, y in zip(a, b[:5])) and torch.equal(g1.get_state(), g2.get_state())


def test_run_sun_replica_matches_historical():
    sd = SC.state("T1")[0]; ss = KCm.classify_sceneset(sd); n, k, m, V = sd["comps"][0]; fm, VC, _ = KO.refined(m, V, 4)
    dom = KO.DomainX(fm, VC); s, E = sd["lights"][0]; dirs = np.array([SS.FX.DIRS[kk] for kk in keys()])
    a = TR.run_sun(dom, s, E, N_DEV, 9003, 64, cams=dirs, shadow=ss.isect); b = SS.run_sun_rec(dom, s, E, N_DEV, 9003, 64, dirs, ss.isect)
    for kk in ("T", "KC"):
        assert float((a[kk] - b[kk]).abs().max()) <= 1e-6 * float(a[kk].abs().max()), kk


def test_valid_state_accepted():
    sd, med = SC.state("T1"); acc, rep = SS.validate(cache(), sd, 0, keys(), medium=med)
    assert acc and rep["visibility_mismatch"] == 0 and rep["visibility_samples"] == N_DEV, rep


def test_visibility_dependency_detected():
    sd, med = SC.state("N2_occluder"); acc, rep = SS.validate(cache(), sd, 0, keys(), medium=med)
    assert not acc and rep["checks"]["D6_visibility"] is False and rep["visibility_mismatch"] > 0
    sd, med = SC.state("P1_offpath"); acc, rep = SS.validate(cache(), sd, 0, keys(), medium=med); assert acc and rep["visibility_mismatch"] == 0


def test_light_and_material_identity():
    sd, med = SC.state("N1_sun"); acc, rep = SS.validate(cache(), sd, 0, keys(), medium=med); assert not acc and rep["checks"]["D4_light_material_frame"] is False
    sd, med = SC.state("N3_medium"); acc, rep = SS.validate(cache(), sd, 0, keys(), medium=med); assert not acc and rep["checks"]["D2_medium"] is False
    sd, med = SC.state("T1"); n, k, m, V = sd["comps"][0]; m2 = dict(m); m2["V"] = m["V"].copy(); m2["V"][0, 0] += 1e-6; sd["comps"][0] = (n, k, m2, V)
    acc, rep = SS.validate(cache(), sd, 0, keys(), medium=med); assert not acc and rep["checks"]["D1_rest_mesh"] is False


def test_camera_frame_and_rotation():
    sd, med = SC.state("N4_rotate"); acc, rep = SS.validate(cache(), sd, 0, keys(), medium=med)
    assert not acc and rep["checks"]["D3_rigid"] and not rep["checks"]["D4_light_material_frame"]
    sd, med = SC.state("T1"); acc, rep = SS.validate(cache(), sd, 0, keys() + ["obot"], medium=med)
    assert not acc and rep["checks"]["D5_cameras_material_frame"] is False


def test_nonrigid_rejected():
    sd, med = SC.state("T1"); n, k, m, V = sd["comps"][0]; V2 = V.copy(); V2[0, 2] += 1e-4; sd["comps"][0] = (n, k, m, V2)
    acc, rep = SS.validate(cache(), sd, 0, keys(), medium=med); assert not acc and rep["checks"]["D3_rigid"] is False


def test_cached_flux_normalisation_and_L0_order():
    st = cache(); o = st["sun"]; r = SS.reuse(st, list(reversed(keys())))
    assert torch.equal(r["phi_f"], o["phi_f"]) and torch.equal(r["L0"], o["L0"].flip(0))
    out = o["phi_f"].double().sum(0).numpy(); assert np.allclose(out, o["stats"]["flux_out"], rtol=1e-5), (out, o["stats"]["flux_out"])
    fin = np.asarray(o["stats"]["flux_in"]); assert np.all(out <= fin * (1 + 1e-6)) and np.all(out > 0)
    ok = st["rec"]["ok"].double().sum().item() * st["rec"]["w"][0]; assert abs(ok - fin[0]) <= 1e-6 * fin[0]


def test_accounting():
    b = SS.state_bytes(cache()); assert b["total"] == b["source_record"] + b["rest_geometry"] + b["outputs"] and b["source_record"] == 13 * N_DEV
    sd, med = SC.state("T1"); acc, rep = SS.validate(cache(), sd, 0, keys(), medium=med)
    assert set(rep["timing"]) == {"identity_s", "rigid_s", "directions_s", "shadow_scene_build_s", "visibility_replay_s"}
    assert "floor" not in (HERE / "sr70_state.py").read_text().split('"""')[2]          # the contract never consults a scene name


def test_statistics():
    rng = np.random.default_rng(0); x = lambda: 1 + 0.01 * rng.standard_normal((4000, 3))  # noqa: E731
    r0 = ST.r_test(x(), x(), x(), x()); assert all(r["passed"] for r in r0) and ST.null_ok(r0)
    r1 = ST.r_test(x() * 1.01, x() * 1.01, x(), x()); assert not any(r["passed"] for r in r1)
    lab = np.repeat(np.arange(4), 1000); g = ST.group_test(x(), x(), x(), x(), lab); assert all(r["passed"] and r["precise"] for r in g)


if __name__ == "__main__":
    fails = 0
    for name, fn in [(k, v) for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]:
        try:
            fn(); print("PASS", name, flush=True)
        except Exception as e:  # noqa: BLE001
            import traceback
            fails += 1; print("FAIL", name, type(e).__name__, e, traceback.format_exc()[-600:], flush=True)
    print(f"{sum(1 for k in globals() if k.startswith('test_')) - fails} passed, {fails} failed"); sys.exit(1 if fails else 0)
