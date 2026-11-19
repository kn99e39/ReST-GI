"""Replay the WL104B event tracer with all factorial arms and physical references."""

import time

import numpy as np
from scipy.ndimage import binary_dilation

from common104d import H, P, S, STATES, write
from stats104a import ratio_jk
from stats104b import jk_ci, ms_loo

EVENTS = ("e1", "e2", "h")
ROIS = ("objects", "highlight_core", "highlight_ring", "grazing")


def total(events):
    return sum(events[event] for event in EVENTS)


def scene_masks(ctx, reference):
    sid, _, nrm, names = H.HA.gbuffer(ctx)
    ray, _, _ = ctx.gen_rays(1, None)
    d = np.stack([np.asarray(ray.d[i]) for i in range(3)], axis=-1).reshape(nrm.shape)
    sphere, cube = sid == names.index("sphere"), sid == names.index("cube")
    objects = sphere | cube
    luminance = reference @ H.LUMA
    core = objects & (luminance >= 3 * np.median(luminance[objects]))
    ring = sphere & binary_dilation(sphere & core, structure=np.ones((3, 3)), iterations=6) & ~core
    return {"objects": objects, "highlight_core": core, "highlight_ring": ring,
            "grazing": objects & (np.abs(np.sum(d * nrm, axis=-1)) <= 0.25),
            "sphere": sphere, "cube": cube, "full": np.ones_like(sphere)}


def pack_metrics(raw, masks):
    ref = raw["reference"]
    result = {"roi_counts": {key: int(mask.sum()) for key, mask in masks.items()}, "arms": {}}
    ref_total = total(ref)
    ref_average = ref_total.mean(0)
    for arm in STATES:
        value = raw[arm]
        difference = {key: value[key] - ref[key] for key in EVENTS}
        difference["total"] = total(difference)
        abs_closure = np.max(np.abs(total(value) - ref_total - difference["total"]))
        if abs_closure > 1e-12:
            raise ValueError(f"Event decomposition identity failed: {abs_closure}")
        output = {"signed": {}, "stable_rms": {}, "object_material_relative_l1": 0.0, "event_identity_error": float(abs_closure)}
        for name, region in masks.items():
            if not region.any():
                continue
            energy = (ref_total[:, region] @ H.LUMA).sum(axis=1)
            output["signed"][name] = {
                key: ratio_jk((values[:, region] @ H.LUMA).sum(axis=1), energy)
                for key, values in difference.items()
            }
            residual = difference["total"][:, region].reshape(ref_total.shape[0], -1)
            norm = (ref_average[region]**2).mean()
            ms, loo = ms_loo(residual, residual)
            rms, rms_loo = np.sqrt(max(0.0, ms) / norm), np.sqrt(np.maximum(0.0, loo) / norm)
            output["stable_rms"][name] = {**jk_ci(rms, rms_loo), "loo": rms_loo.tolist()}
        mask = masks["objects"]
        output["object_material_relative_l1"] = float(np.abs(difference["total"].mean(0)[mask]).sum() /
                                                       np.abs(ref_average[mask]).sum())
        result["arms"][arm] = output
    p = P["scene"]
    result["precision_passed"] = all(
        (row["stable_rms"]["objects"]["hi"] - row["stable_rms"]["objects"]["lo"]) / 2 <= p["precision_rms_half_width"]
        and all((ci["hi"] - ci["lo"]) / 2 <= p["precision_energy_half_width"]
                for name in ROIS for ci in row["signed"].get(name, {}).values())
        for row in result["arms"].values()
    )
    return result


def render_case(row, descriptor, states, case, out):
    config = P["scene"]
    image, tag, light = row["image"], row["tag"], row["light"]
    initial_camera = H.C.CAM
    if row["cohort"] == "held_out":
        H.C.CAM = P["population"]["held_out_view"]
    dest = out / "scenes" / image
    dest.mkdir(parents=True)
    seed_base = S["images"] + 1000 * case
    H.R81.REG[f"lay_{tag}"] = descriptor
    ctx_ref = H.RT.Ctx(H.scene(tag, "LAY1024"))
    ctx = {"reference": ctx_ref}
    for name, state in states.items():
        key = f"wl104d_{tag}_{name}"
        H.RR.REG[key] = ("ER", state)
        mat = {"type": "cmp83", "key": key}
        ctx[name] = H.RT.Ctx(H.mi.load_dict(H.C.scene_dict(H.mi, H.RT.T, mat, with_sensor=True)))
    if row["cohort"] == "development":
        refs = H.C.ROOT / "evidence/refs" / image
        with np.load(refs / "LAY65_h0.npz") as h0, np.load(refs / "LAY65_h1.npz") as h1:
            mask_ref = 0.5 * (h0["mean"].astype(np.float64) + h1["mean"].astype(np.float64))
    else:
        mask_ref = total(H.HA.render(ctx_ref, image, 2048, seed_base + 990, "cam", pass_seed=True))
    masks = scene_masks(ctx_ref, mask_ref)
    np.savez_compressed(dest / "region_masks.npz", **masks)
    times = {key: 0.0 for key in ctx}
    raw = {key: {event: [] for event in EVENTS} for key in ctx}
    for repeat in range(config["replicates"]):
        flat = {}
        for name, context in ctx.items():
            start = time.monotonic()
            events = H.HA.render(context, image, config["spp_per_replicate"], seed_base + repeat, "cam",
                                 b=config["launch_spp"], pass_seed=name == "reference")
            times[name] += time.monotonic() - start
            if set(events) != set(EVENTS):
                raise ValueError(f"Unexpected pt102 events {list(events)}")
            for event, values in events.items():
                buffer = values.astype(np.float32)
                raw[name][event].append(buffer)
                flat[f"{name}_{event}"] = buffer
        np.savez(dest / f"r{repeat:03d}.npz", **flat)
        print("scene", image, "replicate", repeat, flush=True)
    raw = {key: {event: np.stack(buffers).astype(np.float64) for event, buffers in values.items()} for key, values in raw.items()}
    result = pack_metrics(raw, masks)
    result.update({"seed_base": seed_base, "seconds_by_arm": times, "camera": H.C.CAM, "light": light,
                   "spp_per_replicate": config["spp_per_replicate"], "replicates": config["replicates"]})
    write(dest / "metrics.json", result)
    means = {f"{arm}_{key}": a.mean(axis=0).astype(np.float32)
             for arm, events in raw.items() for key, a in events.items()}
    np.savez(dest / "means.npz", **means)
    for name in states:
        H.RR.REG.pop(f"wl104d_{tag}_{name}")
    H.R81.REG.pop(f"lay_{tag}")
    H.C.CAM = initial_camera
    return result
