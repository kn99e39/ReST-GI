"""Render, reconstruct, and time WL101B using the original WL101 package."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

from svgf import SVGF, compose, projection

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "protocol/reconstruction.json").read_text())
W = P["resolution"]


def sha(path):
    with path.open("rb") as stream:
        h = hashlib.sha256()
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def stamp(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def gpu_busy():
    output = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True, timeout=15,
    )
    return [int(x.strip()) for x in output.splitlines() if x.strip() and int(x) != os.getpid()]


def wait_idle():
    while gpu_busy():
        print("GPU occupied by another job. Waiting.", flush=True)
        time.sleep(10)


def frame_rgb(tensor):
    return tensor.detach().cpu().numpy().T.reshape(W, W, 3).copy()


class Runtime:
    def __init__(self, R, start):
        self.R = R
        self.rt = R.RT(start[0], "sss", R.load_model("sss"), start[1])
        self.light, self.geometry = start[1], start[0]
        self.table = None
        self.revision = 0
        self.guides, self.frames = {}, {}
        self.update(start[0], start[1])

    def update(self, geometry, light):
        rt, R = self.rt, self.R
        if geometry != self.geometry or light != self.light:
            self.revision += 1
        boxes, old = rt.set_geom(geometry)
        rt.update_state(old, boxes)
        rt.set_light(light)
        table, _ = rt.respond(rt.source())
        self.table = R.t2d(table.view(-1))
        self.light, self.geometry = light, geometry

    def trace(self, cam, spp, seed):
        key = tuple(cam["origin"]) + tuple(cam["target"]) + (cam["fov_deg"],)
        self.rt._frames = self.frames.setdefault(key, {})
        return self.rt.frame(self.table, cam, W, spp, seed)

    def frame(self, cam, spp, seed):
        channels = [c.torch() for c in self.trace(cam, spp, seed)]
        return compose(channels, spp)

    def _guide_program(self, scene, ptrs, cam):
        R = self.R
        self.rt.ptrs = ptrs
        ray, _ = R.PT.gen_rays(cam, W, 1, None)
        si = scene.ray_intersect(ray)
        front = si.is_valid() & (si.to_local(-ray.d).z > 1e-6)
        return [R.dr.select(front, a, 0) for a in (*si.p, *si.n)] + [
            R.dr.select(front, R.F(self.rt.shape_id(si)), -1.0)
        ]

    def guide(self, cam):
        R = self.R
        key = tuple(cam["origin"]) + tuple(cam["target"]) + (cam["fov_deg"],)
        if key not in self.guides:
            self.guides[key] = R.maybe_freeze(lambda scene, ptrs: self._guide_program(scene, ptrs, cam))
        arrays = self.guides[key](self.rt.scene, list(self.rt.ptrs))
        R.dr.eval(arrays)
        return R.torch.stack([a.torch() for a in arrays])

    def hidden_in_previous(self, current, old_cam, oracle=False):
        R = self.R
        x, y, z, origin, _ = projection(current, old_cam, W, R.PT.cam_basis)
        p = R.mi.Point3f(*[R.t2d(channel) for channel in current[:3]])
        o = R.mi.Point3f(*[float(v) for v in origin])
        d = p - o
        length = R.dr.norm(d)
        ray = R.mi.Ray3f(o, d / R.dr.maximum(length, 1e-6))
        limit = R.dr.maximum(length - 0.0002, 0.)
        if oracle:
            hit = self.rt.scene.ray_intersect(ray)
            hidden = hit.is_valid() & (hit.t < limit)
        else:
            ray.maxt = limit
            hidden = self.rt.scene.ray_test(ray)
        return ((current[6] >= 0) & (hidden.torch() | (z <= 0) |
                (x < -0.5) | (x >= W-0.5) | (y < -0.5) | (y >= W-0.5)))

    def denoiser(self):
        return SVGF(W, self.R.dev, self.R.PT.cam_basis, P["reconstruction"])


def preflight(rt, cam, cam_b, historical_map):
    R = rt.R
    torch = R.torch
    guide = rt.guide(cam)
    check = {}
    with np.load(historical_map) as old:
        mask = old["ok"] & (old["sid"] >= 0)
        sid = guide[6].cpu().numpy().reshape(W, W)
        p = guide[:3].cpu().numpy().T.reshape(W, W, 3)
        n = guide[3:6].cpu().numpy().T.reshape(W, W, 3)
        check["guide_sid_mismatch_fraction"] = float(np.mean(sid != old["sid"]))
        check["guide_point_rmse"] = float(np.mean((p[mask]-old["p"][mask])**2)**0.5)
        check["guide_normal_rmse"] = float(np.mean((n[mask]-old["n"][mask])**2)**0.5)
    assert check["guide_sid_mismatch_fraction"] <= 1e-4
    assert check["guide_point_rmse"] <= 1e-4 and check["guide_normal_rmse"] <= 1e-4

    check["frame_relative_l1"] = {}
    for spp in (1, 4):
        c = [x.torch() for x in rt.trace(cam, spp, 10170900)]
        candidate = compose(c, spp)
        expected = torch.stack([(c[i]+c[i+3]+c[i+6])/spp for i in range(3)])
        error = (candidate-expected).abs().sum().item()/expected.abs().sum().item()
        check["frame_relative_l1"][spp] = error
        assert error <= 1e-6
        # Repeat through the original WL101 entry point and paired seed.
        rt.rt._frames = {}
        baseline = compose([x.torch() for x in rt.rt.frame(rt.table, cam, W, spp, 10170900)], spp)
        assert (baseline-candidate).abs().sum().item()/expected.abs().sum().item() <= 1e-6
    den = rt.denoiser()
    raw = rt.frame(cam, 1, 10170910)
    filtered, count, _ = den.apply(raw, guide, cam, 0)
    assert bool(torch.isfinite(filtered).all()) and bool((count == 1).all())
    assert torch.equal(den.history[:3], raw)
    still = torch.full_like(raw, 0.42)
    c, count, _ = den.apply(still, guide, cam, 0, reset=True)
    assert bool((count == 1).all()) and (c-still).abs().max().item() < 2e-6
    c, count, _ = den.apply(still, guide, cam, 0)
    assert bool((count == 2).all()) and (c-still).abs().max().item() < 2e-6
    g2 = rt.guide(cam_b)
    blocked = rt.hidden_in_previous(g2, cam)
    _, count, mapping = den.apply(still, g2, cam_b, 0, blocked=blocked)
    hidden = rt.hidden_in_previous(g2, cam, oracle=True)
    assert torch.equal((count > 1), (mapping >= 0))
    accepted = mapping >= 0
    assert torch.equal(g2[6, accepted], guide[6, mapping[accepted].long()])
    assert 0 < int(accepted.sum()) < W*W and int(hidden.sum()) > 0
    check["camera_hidden_pixels"] = int(hidden.sum())
    check["camera_hidden_reuse_fraction"] = float(accepted[hidden].float().mean())
    assert check["camera_hidden_reuse_fraction"] <= 0.005
    _, count, _ = den.apply(raw, guide, cam, 1)
    assert bool((count == 1).all()) and torch.equal(den.history[:3], raw)
    check["constant_and_reset"] = True
    check["reconstruction_reference_access"] = "No reference image import or argument in SVGF. Only gbuffer enters preflight."
    return check


def save_guide(path, guide):
    G = guide.detach().cpu().numpy().reshape(7, W, W)
    np.savez_compressed(path, ok=G[6] >= 0, sid=G[6].astype(np.int16),
                        p=G[:3].transpose(1, 2, 0), n=G[3:6].transpose(1, 2, 0))


def quality(rt, out, start, warm, tour, cam, cam_b):
    R = rt.R
    for spp in P["spp"]:
        for seq, name in enumerate(("static", "tour", "camera")):
            for rep in range(P["evaluation"]["repeats"]):
                wait_idle()
                den = rt.denoiser()
                cold = rt.denoiser() if name != "static" else None
                rt.update(start[0], start[1])
                if name == "tour":
                    for g, light, _ in warm:
                        rt.update(g, light)
                    records = [(g, light, sid, cam) for g, light, sid in tour]
                else:
                    g, light, sid = tour[0]
                    rt.update(g, light)
                    records = [(g, light, sid, cam_b if name == "camera" and 16 <= t < 20 else cam)
                               for t in range(P["evaluation"]["static_frames"])]
                last_cam, last_guide = None, None
                rows = []
                for frame, (g, light, sid, camera) in enumerate(records):
                    if name == "tour":
                        rt.update(g, light)
                    seed = 10171000 + 100000*seq + 10000*spp + 1000*rep + frame
                    if last_cam != camera or name == "tour":
                        guide = rt.guide(camera)
                    else:
                        guide = last_guide
                    cam_id = "B" if camera == cam_b else "A"
                    guide_file = out / f"guide_{sid}_{cam_id}.npz"
                    if not guide_file.exists():
                        save_guide(guide_file, guide)
                    raw = rt.frame(camera, spp, seed)
                    moved_camera = name == "camera" and last_cam is not None and last_cam != camera
                    blocked = rt.hidden_in_previous(guide, last_cam) if moved_camera else None
                    recon, count, _ = den.apply(raw, guide, camera, rt.revision, blocked=blocked)
                    row = {"seed": seed, "state": sid, "frame": frame, "camera": cam_id}
                    previous_camera = last_cam
                    last_cam, last_guide = camera, guide
                    if name == "camera" and frame < 16:
                        rows.append(row)
                        continue
                    arrays = {"raw": frame_rgb(raw), "reconstructed": frame_rgb(recon),
                              "count": count.cpu().numpy().astype(np.uint8).reshape(W, W)}
                    recovered = name == "camera" and previous_camera != camera
                    if recovered or name == "tour":
                        control, _, _ = cold.apply(raw, guide, camera, rt.revision, reset=True)
                        if recovered:
                            arrays["spatial_reset"] = frame_rgb(control)
                    if recovered:
                        hidden = rt.hidden_in_previous(guide, previous_camera, oracle=True)
                        arrays["disoccluded"] = hidden.cpu().numpy().reshape(W, W)
                    if name == "tour":
                        assert np.all(arrays["count"] == 1)
                        row["exact_reset"] = np.array_equal(arrays["reconstructed"], frame_rgb(control))
                        assert row["exact_reset"]
                    assert np.isfinite(arrays["raw"]).all() and np.isfinite(arrays["reconstructed"]).all()
                    path = out / f"{name}_{spp}spp_r{rep}_f{frame:02d}.npz"
                    np.savez(path, **arrays)
                    rows.append({**row, "file": path.name, "sha256": sha(path)})
                stamp(out / f"{name}_{spp}spp_r{rep}.json", rows)
                print(f"Quality {name} {spp} spp replica {rep} complete", flush=True)


def timing(rt, out, start, warm, tour, cam, cam_b):
    R = rt.R
    torch = R.torch
    data = {}
    for spp in P["spp"]:
        data[spp] = {}
        for reconstruction in (False, True):
            name = "reconstructed" if reconstruction else "raw"
            den = rt.denoiser() if reconstruction else None
            rt.update(start[0], start[1])
            for g, light, sid in warm + tour:
                rt.update(g, light)
                raw = rt.frame(cam, spp, 10170901 + spp)
                if den is not None:
                    den.apply(raw, rt.guide(cam), cam, rt.revision)
            R.sync()
            wait_idle()
            torch.cuda.reset_peak_memory_stats()
            before = torch.cuda.memory_allocated()
            resident = 0
            events = []
            for i, (g, light, sid) in enumerate(tour * 4):
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                R.sync()
                start_time = time.perf_counter()
                rt.update(g, light)
                raw = rt.frame(cam, spp, 10190000 + 1000*spp + i)
                if den is not None:
                    guide = rt.guide(cam)
                    start_event.record()
                    image, _, _ = den.apply(raw, guide, cam, rt.revision)
                    end_event.record()
                    resident = den.buffers_bytes + guide.numel()*guide.element_size()
                R.sync()
                elapsed = (time.perf_counter()-start_time)*1e3
                events.append({"ms": elapsed, "state": sid,
                               "denoiser_gpu_ms": start_event.elapsed_time(end_event) if den else 0.})
            foreign = gpu_busy()
            data[spp][name] = {"events": events, "foreign_gpu_pids": foreign,
                               "peak_torch_bytes": torch.cuda.max_memory_allocated(),
                               "begin_torch_allocated": before, "reconstruction_resident_bytes": resident}
            print(f"Timing {name} {spp} spp complete", flush=True)
        rt.update(*tour[0][:2])
        den = rt.denoiser()
        first = rt.guide(cam)
        den.apply(rt.frame(cam, spp, 10190001), first, cam, rt.revision)
        camera_events = []
        previous = cam
        for i in range(20):
            current = cam_b if i % 2 == 0 else cam
            R.sync()
            t = time.perf_counter()
            raw = rt.frame(current, spp, 10200000+spp*100+i)
            guide = rt.guide(current)
            blocked = rt.hidden_in_previous(guide, previous)
            den.apply(raw, guide, current, rt.revision, blocked=blocked)
            R.sync()
            camera_events.append((time.perf_counter()-t)*1e3)
            previous = current
        data[spp]["camera_only_ms"] = camera_events[2:]
    stamp(out / "timing.json", data)


def main(wl101, out, commit, only_preflight):
    wait_idle()
    sys.path.insert(0, str(wl101/"experiments/ecro_dyn101"))
    import rt101 as R
    import seq101 as Q
    assert R.FREEZE
    ck = R.ROOT100/"sss/model.pt"
    assert sha(ck) == P["checkpoint_sha256"]
    import triton
    assert (R.mi.__version__, R.dr.__version__, R.torch.__version__, triton.__version__) == (
            "3.8.0", "1.3.1", "2.3.1+cu121", "2.3.1")
    assert R.torch.cuda.get_device_name() == "NVIDIA GeForce RTX 3080 Ti"
    out.mkdir(parents=True, exist_ok=False)
    case, start, events = Q.workload("V_SSS_TOUR")
    warm, tour = events[:Q.WARM], events[Q.WARM:Q.WARM+8]
    assert case == "sss" and [sid for _, _, sid in tour] == [0,2,3,4,5,6,7,1]
    cam = dict(R.C100.P["datasets"]["sss"]["camera"], res=W)
    cam_b = dict(cam, origin=[cam["origin"][0]+0.08, *cam["origin"][1:]])
    rt = Runtime(R, start)
    rt.update(*tour[0][:2])
    lib = next(Path(row.split()[-1]) for row in Path("/proc/self/maps").read_text().splitlines()
               if "/libnvoptix.so." in row)
    assert lib.name == "libnvoptix.so.595.71.05"
    arch_map = wl101/"results/ecro_dyn101/evidence/quality/sss/MAP_0.npz"
    assert sha(arch_map) == "8c269050c351027ceae4a06cc63ca3a401ba64c37d341a90c345830089393fbc"
    print("WL101 renderer loaded. Beginning focused preflight.", flush=True)
    checks = preflight(rt, cam, cam_b, arch_map)
    assert sha(wl101/"experiments/ecro_dyn101/rt101.py") == "417fd6c041603b6760aa72179bb383ac44ca1e3d031715f8b76fe0bb419eedf6"
    archived_b = arch_map.with_name("PRED_B_0.npz")
    assert sha(archived_b) == "fb43bb808a48f8ad5293603d4b4404c7275d2d2e6e5d085ec2778b4635eb143a"
    with np.load(archived_b) as prior:
        table = rt.table.torch().cpu().numpy().reshape(prior["B"].shape)
        checks["archived_table_relative_l1"] = float(np.abs(table-prior["B"]).sum()/np.abs(prior["B"]).sum())
        assert checks["archived_table_relative_l1"] <= 1e-6
    meta = {"commit": commit, "checkpoint_sha256": sha(ck), "protocol_sha256": sha(HERE/"protocol/reconstruction.json"),
            "renderer_sha256": sha(wl101/"experiments/ecro_dyn101/rt101.py"),
            "configs_sha256": sha(R.ROOT100/"configs.json"), "optix": str(lib), "optix_sha256": sha(lib),
            "versions": {"python": platform.python_version(), "os": platform.platform(),
                         "gpu": R.torch.cuda.get_device_name(), "mitsuba": R.mi.__version__,
                         "drjit": R.dr.__version__, "torch": R.torch.__version__,
                         "triton": triton.__version__, "numpy": np.__version__},
            "guide_arrays": "center-ray world position, world normal, geometric shape ID; derived depth and rigid camera reprojection",
            "camera": {"A":cam, "B":cam_b}, "tour": tour, "preflight": checks}
    stamp(out/"run_manifest.json", meta)
    print("Preflight", checks, flush=True)
    if only_preflight:
        return
    timing(rt, out, start, warm, tour, cam, cam_b)
    quality(rt, out, start, warm, tour, cam, cam_b)
    print("WL101B evidence complete.", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("wl101", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("commit")
    p.add_argument("--preflight", action="store_true")
    a = p.parse_args()
    main(a.wl101.resolve(), a.out.resolve(), a.commit, a.preflight)
