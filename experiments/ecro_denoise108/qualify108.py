"""Qualify immutable data and a geometry-only foreground grazing statistic."""
import argparse
from pathlib import Path
import time

import numpy as np

from common108 import P, PROTOCOL, VIEWS, read, sha, write
from capture107 import admit
from synthetic107 import counterexamples

CENTRE = np.array([0.9, 0.4, 0.4])
RADIUS = 0.4


def foreground_statistic(squared_sum, count):
    assert np.all((count >= 0) & (count <= 64))
    return (squared_sum / np.maximum(count, 1)).astype(np.float32)


def extract(r, R, rt, camera):
    """WL105 camera rays and shape ownership, without radiometric program calls."""
    start = time.monotonic()
    dr, mi = R.dr, R.mi
    F, U, I = mi.Float, mi.UInt32, mi.Int32
    res = r.RES
    o, fwd, right, up, s = R.PT.cam_basis(dict(camera, res=res))
    pixel = dr.arange(U, res*res)
    row, col = pixel//res, pixel % res
    count = np.zeros((res, res), np.int32)
    squared = np.zeros((res, res), np.float64)
    analytic = np.zeros_like(squared)
    to_center = CENTRE - np.array(o, dtype=np.float64)
    for k in range(64):
        jx, jy = ((k % 8) + 0.5)/8, ((k // 8) + 0.5)/8
        x = (F(col)+jx)*(2.0/res)-1.0
        y = 1.0-(F(row)+jy)*(2.0/res)
        direction = dr.normalize(mi.Vector3f(*[float(fwd[j]) + s*x*float(right[j]) + s*y*float(up[j])
                                              for j in range(3)]))
        ray = mi.Ray3f(mi.Point3f(*[F(float(o[j])) for j in range(3)]), direction)
        hit = rt.scene.ray_intersect(ray)
        front = hit.is_valid() & (I(rt.shape_id(hit)) == 6)
        cosine = dr.select(front, dr.clip(dr.dot(hit.n, direction), -1, 1), 0)
        dr.eval(front, cosine, direction)
        is_foreground = np.asarray(front).reshape(res, res)
        count += is_foreground
        squared += np.asarray(cosine, dtype=np.float64).reshape(res, res)**2
        # Analytic normal at the first intersection of a unit-direction sphere ray.
        direction_cpu = np.stack([np.asarray(direction[j], dtype=np.float64) for j in range(3)], axis=-1)
        direction_cpu /= np.linalg.norm(direction_cpu, axis=-1, keepdims=True)
        along = np.einsum("ij,j->i", direction_cpu, to_center)
        closed = np.maximum(along**2 - to_center @ to_center + RADIUS**2, 0) / RADIUS**2
        analytic += np.where(is_foreground, np.minimum(closed.reshape(res,res), 1), 0)
    result = foreground_statistic(squared, count)
    analytic = foreground_statistic(analytic, count)
    assert np.isfinite(result).all() and result.min() >= 0 and result.max() <= 1
    return result, count, float(np.max(np.abs(result-analytic))), time.monotonic()-start


def quadratic_features(x):
    i,j = np.triu_indices(x.shape[1])
    return np.column_stack((np.ones(len(x)), x, x[:, i]*x[:, j]))


def synthetic_control():
    item = counterexamples()["equal_guide_pair"]
    a = foreground_statistic(np.array([32*0.5**2, 16*(0.2**2+0.8**2)]), np.array([32,32]))
    np.testing.assert_allclose(a, item["foreground_mean_squared_cosine"], rtol=1e-7)
    # Identical observations contain two convex taps bracketing both answers.
    # A softmax kernel using a learned affine descriptor logit fits both targets.
    targets = np.array(item["targets"])
    low, high = 0.9, 1.2
    desired = (targets-low)/(high-low)
    coefficients = np.linalg.solve(np.column_stack((np.ones(2), a)), np.log(desired/(1-desired)))
    observed = low + (high-low)/(1+np.exp(-(coefficients[0]+coefficients[1]*a)))
    np.testing.assert_allclose(observed, targets, atol=1e-12)
    return {"descriptor": a.tolist(), "kernel_predictions": observed.tolist(),
            "neutral_channel_twoworld_mse_floor": float(targets.var())}


def authenticate(base, prior, anchor):
    assert sha(anchor) == P["frozen"]["wl105_anchor"]
    data, ev, files = admit(base, anchor)
    sources = {"wl107_reference": "reference/reference107.json", "wl107_capture": "capture/capture_manifest.json",
               "wl107_real": "real_amend1/real107.json", "wl107_replay": "replay.json"}
    for key, path in sources.items():
        assert sha(prior / path) == P["frozen"][key], path
    real = read(prior / sources["wl107_real"])
    assert real["complete"] and tuple(real["views"]) == VIEWS
    reference = read(prior / sources["wl107_reference"])
    capture = read(prior / sources["wl107_capture"])
    assert reference["preliminary_pass"] and capture["seed_and_stream_disjoint"] and capture["complete"]
    assert real["capture_manifest"] == P["frozen"]["wl107_capture"]
    assert real["qualified_reference_manifest"] == P["frozen"]["wl107_reference"]
    for view in VIEWS:
        target = prior / "reference" / (view + "_target.npy")
        assert sha(target) == reference["views"][view]["target_sha256"]
    return data, ev, files


def qualify(base, prior, anchor, wl101, out, commit):
    out.mkdir(parents=True, exist_ok=False)
    synthetic = synthetic_control()
    data, ev, files = authenticate(base, prior, anchor)
    import render105 as r
    import torch
    reader, _, _, _, _, _, _, scene = r.RG.F101.inputs(wl101)
    R, rt = reader.R, reader.rt
    record = {"commit": commit, "protocol_sha256": sha(PROTOCOL), "synthetic": synthetic,
              "files_verified": files, "checkpoint_sha256": ev["weights_sha256"]["R3"],
              "base_data_sha256": sha(base / "data/run_manifest.json"),
              "reference_sha256": P["frozen"]["wl107_reference"],
              "renderer": scene, "torch": torch.__version__, "device": torch.cuda.get_device_name(0), "files": {}}
    regression = {"train": [], "val": []}
    assert {k: sum(n.startswith(k+"/") for n in data["views"]) for k in ("train","val","eval")} == {
        "train": 40, "val": 8, "eval": 6}
    for key, view in data["views"].items():
        split, name = key.split("/")
        result, count, error, seconds = extract(r, R, rt, view["camera"])
        with np.load(base / "data" / key / "aux.npz") as frozen:
            np.testing.assert_array_equal(count, 64*frozen["cover"][6])
            mask = (count > 0) & (count < 64)
            if split in regression:
                old = np.concatenate((frozen["cover"], frozen["normal"], frozen["depth"][None]/5))
                regression[split].append((old[:, mask].T, result[mask]))
        assert error <= .002, (key, error)
        folder = out / split
        folder.mkdir(exist_ok=True)
        path = folder / (name + ".npy")
        np.save(path, result, allow_pickle=False)
        record["files"][key] = {"sha256": sha(path), "seconds": seconds, "mixed_pixels": int(mask.sum()),
                                  "mixed_std": float(result[mask].std()) if mask.any() else None,
                                  "sphere_analytic_max_abs": error}
        if key == next(iter(data["views"])):
            repeated, again, _, _ = extract(r, R, rt, view["camera"])
            assert np.array_equal(result, repeated) and np.array_equal(count, again)
            record["repeat_exact"] = True
    assert max(row["mixed_std"] or 0 for key,row in record["files"].items() if key.startswith("train/")) > .001
    train, val = ({k: np.concatenate([r[i] for r in regression[k]]) for k in regression} for i in range(2))
    scale = train["train"].std(0)
    keep = scale > 1e-7
    centre = train["train"][:,keep].mean(0)
    mat = {k: quadratic_features((x[:,keep]-centre)/scale[keep]) for k,x in train.items()}
    coeff = np.linalg.lstsq(mat["train"], val["train"], rcond=1e-6)[0]
    residual = mat["val"] @ coeff - val["val"]
    record["guide_quadratic_val"] = {"rmse": float(np.sqrt(np.mean(residual**2))),
                                    "r2": float(1-np.mean(residual**2)/np.var(val["val"]))}
    record["pass"] = True
    write(out / "qualification.json", record)
    print("Qualified geometry and data", len(record["files"]), record["guide_quadratic_val"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline","prior","anchor","wl101","output"):
        p.add_argument(name, type=Path)
    p.add_argument("commit")
    a = p.parse_args()
    qualify(a.baseline,a.prior,a.anchor,a.wl101,a.output,a.commit)
