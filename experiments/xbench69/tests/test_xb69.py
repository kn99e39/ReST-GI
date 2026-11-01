"""WL69 focused tests (protocol xbench69_v1 tests list). Run on LabServer69 from experiments/fnto_k55 with the 8dna26 venv:
    python -m pytest -q ../xbench69/tests/test_xb69.py
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "fnto_k55"))
import d8_scene as DS  # noqa: E402
import k55_fixtures as KF  # noqa: E402
import p53_fixtures as FX  # noqa: E402

CYC = Path.home() / "rest_wl55" / "out" / "cycles_cuda"
UP = Path.home() / "NeuralShaderDemo" / "external" / "8dna26"
WORK = Path.home() / "rest_xbench69" / "8dna_work"


def test_similarity_native_convention():
    s, c = DS.similarity(); V, F = DS.asset_mesh()
    assert abs(np.linalg.norm(V.max(0) - V.min(0)) - 2.0) < 1e-9 and np.abs(0.5 * (V.min(0) + V.max(0))).max() < 1e-9
    assert np.linalg.norm(V, axis=1).max() <= 1.0 + 1e-9          # inside 8DNA's unit incident sphere


def test_asset_geometry_equals_fixture():
    s, c = DS.similarity(); m, V0 = [(m, V) for n, k, m, V in KF.scene_def("floor")["comps"] if k == "medium"][0]
    V, F = DS.asset_mesh(); assert np.abs(V / s + c - V0).max() < 1e-12 and np.array_equal(F, m["F"])


def test_t1_is_rigid_translation_of_t0():
    t = DS.pose_translation("T1"); s, _ = DS.similarity()
    assert np.allclose(t / s, (0.0023, 0.0017, 0.004 - 0.01), atol=1e-9)


def test_floor_and_light_identical_across_states():
    a, b = KF.scene_def("floor"), KF.scene_def("floor_c4")
    fa = [V for n, k, m, V in a["comps"] if k == "lambert"][0]; fb = [V for n, k, m, V in b["comps"] if k == "lambert"][0]
    assert np.array_equal(fa, fb) and a["lights"] == b["lights"]


def test_medium_parameters():
    s, _ = DS.similarity(); med = DS.medium(s)
    assert med["sigma_t"] == 1.0 and abs(med["scale"] - FX.SIGMA_T / s) < 1e-12
    assert med["albedo"]["value"] == list(FX.ALBEDO) and med["phase"]["type"] == "isotropic" and FX.G_PHASE == 0.0


def test_reference_files_and_manifest():
    man = [json.loads(x) for x in open(CYC / "render_manifest.jsonl")]
    for scene, seeds in (("floor", (5501, 5502)), ("floor_c4", (5531, 5532))):
        for cam in ("top", "otop", "underup", "underdown"):
            for sd in seeds:
                assert (CYC / f"cimg_{scene}_{cam}_s{sd}.npy").exists()
                e = [m for m in man if m["scene"] == scene and m["cam"] == cam and m["seed"] == sd][-1]
                assert e["spp"] == 1024 and e["device"] == "CUDA" and e["build_hash"] == "e1ccd9d4a1d3"


def test_seed_disjointness():
    ref = {5501, 5502, 5531, 5532}
    cy = {690000 + 1000 * si + 10 * li + r for si in (0, 1) for li in range(6) for r in (0, 1)}
    d8 = {7_000_000 + 100_000 * si + 1000 * li + 500 * r + k for si in (0, 1) for li in range(6) for r in (0, 1) for k in range(64)}
    assert len(d8) == 2 * 6 * 2 * 64 and not (ref & cy) and not (ref & d8) and not (cy & d8)


def test_8dna_code_unchanged():
    assert hashlib.sha256((WORK / "train.py").read_bytes()).hexdigest() == hashlib.sha256((UP / "train.py").read_bytes()).hexdigest()
    for p in ("models", "utils", "configs"):
        assert (WORK / p).resolve() == (UP / p).resolve()


def test_no_reference_access_in_predictors():
    for f in ("d8_eval.py", "x_rg.py", "d8_scene.py"):
        src = (HERE / f).read_text()
        assert "cimg_" not in src, f                      # predictors never open reference images (ccam geometry only)


def test_rg_readout_is_wl55_nee_plus_ball():
    """Amendment 2: the analysed ReST-GI port prediction is nee + ball (k55_analyze.load_arm), equal to WL55's stored arm."""
    import k55_common as KCm
    root = Path.home() / "rest_xbench69"; w55 = Path.home() / "rest_wl55" / "out" / "stageB"
    for scene in ("floor", "floor_c4"):
        sd = KF.scene_def(scene)
        for cam in ("top", "underup", "underdown"):
            k = KCm.dirkey(sd["cams"][cam]["dir"]); z = np.load(root / "rg" / f"{scene}__CAK_s1.npz"); r = np.load(w55 / scene / "CAK_s1.npz")
            p = z[f"{k}|nee"] + z[f"{k}|ball"]; q = r[f"{k}|nee"] + r[f"{k}|ball"]
            assert np.abs(p).max() > 0 and np.abs(p - q).max() <= 1e-6 * np.abs(q).max()
            assert np.abs(z[f"{k}|ball"]).max() < 1e-2 * np.abs(p).max()      # ball alone is the small unscattered term


if __name__ == "__main__":                                    # pytest is not installed on LabServer69 (no internet): plain runner
    fails = 0
    for name, fn in sorted((k, v) for k, v in globals().items() if k.startswith("test_") and callable(v)):
        try:
            fn(); print("PASS", name)
        except Exception as e:  # noqa: BLE001
            fails += 1; print("FAIL", name, type(e).__name__, e)
    print(f"{sum(1 for k in globals() if k.startswith('test_')) - fails} passed, {fails} failed"); sys.exit(1 if fails else 0)
