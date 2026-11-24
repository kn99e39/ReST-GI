"""WL111 sealed cameras and sequences (numpy only). Every definition here is fixed in the protocol before any candidate
training, model selection or evaluation; the sealed-view acceptance uses geometry only (no radiance).

Views
  historical V0, H1-H5   WL105 evaluation cameras (inspected by WL105-WL109: historical-workload comparison only)
  training / validation  WL105 T00-T39 / Q00-Q07 cameras (WL105 4096-spp frozen-ECRO targets)
  sealed K1-K6           new random room views (cameras105.random_cameras, protocol seed), rejected near any WL105 view
Sequences (all 512 x 512, SSS case, frozen ECRO)
  Q1 static          stationary camera / geometry / light: V0 and K1 (static protocol, 64 frames + forced reset)
  Q2 camera orbit    fixed scene state (cfg0, light 0); orbit around the held-out target, azimuth -60 -> +25 deg over 96 frames
  Q3 abrupt edit     V0; cfg0 frames 0-39, then the edit geometry (frames 40-79); light 0
  Q4 continuous      V0; boxes interpolate cfg2 -> cfg1 (WL101 V_SSS_ANIM endpoints) over 48 frames, one new pose per frame
  Q5 light change    V0; cfg0; light 0 frames 0-39, light 1 (WL100 test-0 second light) frames 40-79
"""
import copy
import math

import numpy as np

import common_r111 as C

Q = C.P["sequences"]


def rows():
    return C.read(C.WL101 / "results" / "ecro_material100" / "configs.json")["sss"]["test"]


def lerp_geom(g0, g1, t):
    """seq101._lerp_geom (same statements)."""
    g = copy.deepcopy(g0)
    for b in "AB":
        g[b]["c"] = [float((1 - t) * a + t * c) for a, c in zip(g0[b]["c"], g1[b]["c"])]
        g[b]["yaw"] = float((1 - t) * g0[b]["yaw"] + t * g1[b]["yaw"])
    return g


def wl105_cameras():
    """name -> camera dict for every WL105 view (train T*, val Q*, eval V0/H*) from the frozen WL105 run manifest."""
    m = C.read(C.D105 / "run_manifest.json")
    return {k.split("/")[1]: dict(v["camera"]) for k, v in m["views"].items()}


def orbit_camera(k):
    o = Q["Q2"]
    n = o["frames"]
    az = math.radians(o["azimuth_deg"][0] + (o["azimuth_deg"][1] - o["azimuth_deg"][0]) * k / (n - 1))
    el = math.radians(o["elevation_deg"])
    d = o["horizontal_m"]
    T = np.array(o["target"])
    org = T + np.array([d * math.sin(az), -d * math.cos(az), d * math.tan(el)])
    return {"origin": [float(x) for x in org], "target": [float(x) for x in T], "fov_deg": float(o["fov_deg"])}


def sealed_candidates():
    """Candidate sealed views in a fixed order (geometry acceptance happens on the GPU: valid fraction and sphere size)."""
    import sys
    sys.path.insert(0, str(C.EXP / "ecro_denoise105"))
    import cameras105 as CAM
    s = C.P["views"]["sealed"]
    forbidden = list(wl105_cameras().values())
    geom = rows()[0]["geom"]
    return CAM.random_cameras(s["candidates"], s["camera_seed"], geom, forbidden)


def sequence(name):
    """-> list of frames: dict(cam, geom, light, state) where state labels the transport state (for references)."""
    r = rows()
    light0 = r[0]["lights"][0]
    v0 = dict(C.P["views"]["V0"])
    q = Q[name]
    if name == "Q2":
        return [{"cam": orbit_camera(k), "geom": r[0]["geom"], "light": light0, "state": "cfg0_l0", "key": k}
                for k in range(q["frames"])]
    if name == "Q3":
        g1 = r[q["edit_to_config"]]["geom"]
        return [{"cam": v0, "geom": r[0]["geom"] if k < q["edit_frame"] else g1, "light": light0,
                 "state": "cfg0_l0" if k < q["edit_frame"] else f"cfg{q['edit_to_config']}_l0", "key": k}
                for k in range(q["frames"])]
    if name == "Q4":
        g0, g1 = r[q["from_config"]]["geom"], r[q["to_config"]]["geom"]
        n = q["frames"]
        return [{"cam": v0, "geom": lerp_geom(g0, g1, k / (n - 1)), "light": light0, "state": f"anim{k:02d}", "key": k}
                for k in range(n)]
    if name == "Q5":
        l1 = r[0]["lights"][q["light_index"]]
        return [{"cam": v0, "geom": r[0]["geom"], "light": light0 if k < q["edit_frame"] else l1,
                 "state": "cfg0_l0" if k < q["edit_frame"] else f"cfg0_l{q['light_index']}", "key": k}
                for k in range(q["frames"])]
    raise KeyError(name)


def reference_frames(name):
    """Frames of a sequence that receive their own frozen-ECRO reference (keyframes; static-camera states share one)."""
    q = Q[name]
    seq = sequence(name)
    if name == "Q2":
        keys = sorted(set(range(0, q["frames"], q["reference_every"])) | set(q["reference_extra"]))
        return {f"Q2_k{k:02d}": seq[k] for k in keys}
    if name in ("Q3", "Q5"):
        out = {}
        for fr in seq:
            out.setdefault(f"{name}_{fr['state']}", fr)
        return out
    if name == "Q4":
        return {f"Q4_{fr['state']}": fr for fr in seq}
    raise KeyError(name)
