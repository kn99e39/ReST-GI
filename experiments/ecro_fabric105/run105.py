"""Freeze, fit and render ECRO on the native public Fabric scene."""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from fabric105 import Fabric, HERE, ROOT, UPSTREAM, P, NB, dr, mi, load_scene, make_geometry


def sha(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def commit(path):
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_npz(path, **fields):
    temp = path.with_suffix(".tmp.npz")
    np.savez(temp, **fields)
    temp.replace(path)


def checked(values):
    if any(not np.isfinite(v).all() or (v < 0).any() for v in values):
        raise ValueError("Negative or nonfinite radiance")


def image(fields, high="h"):
    return fields["e0"] + fields["e1"] + fields["e2"] + fields[high]


def relative_l1(a, b, mask=None):
    if mask is not None:
        a,b = a[mask],b[mask]
    return float(np.abs(a-b).sum()/max(float(np.abs(b).sum()),1e-20))


def signed_energy(a, b, mask):
    lum = np.asarray([.2126,.7152,.0722])
    return float(((a[mask]-b[mask])@lum).sum()/max(float((b[mask]@lum).sum()),1e-20))


def coarse(a):
    h, w, _ = a.shape
    return a.reshape(h//8, 8, w//8, 8, 3).mean((1,3))


def provenance():
    files = [UPSTREAM/"scenes"/p for p in ("fabric/__init__.py", "fabric/fabric.m3curve", "textures/directional.exr")]
    code = [*sorted(HERE.glob("*.py")), HERE/"protocol.json",
            HERE.parent/"ecro86/model86.py", HERE.parent/"ecro_lr3_102/train102.py"]
    return {"repo_commit": commit(ROOT), "upstream_commit": commit(UPSTREAM),
            "source_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in files+code},
            "protocol_sha256": sha(HERE/"protocol.json"), "platform": platform.platform(),
            "python": platform.python_version(), "mi": mi.__version__, "drjit": dr.__version__}


def preflight(root):
    config = P["preflight"]
    root.mkdir(parents=True, exist_ok=False)
    t = time.monotonic()
    scene = load_scene(config["resolution"])
    g, info = make_geometry(scene)
    ctx = Fabric(scene, g)
    hit, dbin, state = ctx.state()
    g.update({"hit": hit, "dbin": dbin})
    save_npz(root/"geometry.npz", **g)
    geometry_s = time.monotonic()-t
    q = []
    r = []
    masks = []
    for seed in config["seeds"]:
        q0 = ctx.pixels(config["spp_each"], seed)
        n = mi.render(scene, spp=config["spp_each"], seed=seed+1000).numpy()
        checked([*q0.values(), n])
        q.append(image(q0))
        r.append(n)
        masks.append(q0["fabric_mask"])
        print("PREFLIGHT",seed,flush=True)
    mask = np.mean(masks,axis=0)>.5
    qa, ra = np.mean(q,axis=0), np.mean(r,axis=0)
    table = dr.zeros(mi.Float,ctx.n*NB*3)
    cross = ctx.pixels(4,config["seeds"][0],table=table)
    physical = ctx.pixels(4,config["seeds"][0])
    held_geom = Fabric(load_scene(48, **P["evaluation"]["held_camera"]),g)
    extra = held_geom.pixels(4,config["seeds"][0],table=table)
    fallback = max(float(a["fallback"].sum()/max(a["lookup"].sum(),1.)) for a in (cross,extra))
    low_gap = max(relative_l1(cross[a],physical[a]) for a in ("e0","e1","e2"))
    native_flags = ctx.pool.bsdf().flags()
    specimen = {
        "transmission": bool(dr.all(mi.has_flag(native_flags,mi.BSDFFlags.Transmission))),
        "anisotropic": bool(dr.all(mi.has_flag(native_flags,mi.BSDFFlags.Anisotropic))),
        "backside": bool(dr.all(mi.has_flag(native_flags,mi.BSDFFlags.BackSide))),
    }
    metrics = {
        "native_flags": specimen, "geometry": {**info, **state, "seconds": geometry_s},
        "signed_energy": signed_energy(qa,ra,mask), "coarse_rel_l1": relative_l1(coarse(qa),coarse(ra)),
        "native_coarse_half_l1": relative_l1(coarse(r[0]),coarse(r[1])),
        "adapter_coarse_half_l1": relative_l1(coarse(q[0]),coarse(q[1])),
        "zero_table_max_low_l1": low_gap, "camera_lookup_fallback_max": fallback,
        "geometry_sha256": sha(root/"geometry.npz"), "seconds": time.monotonic()-t,
        "provenance": provenance(),
    }
    metrics["passed"] = bool(abs(metrics["signed_energy"]) <= config["max_signed_energy_error"]
                             and metrics["coarse_rel_l1"] <= config["max_coarse_relative_l1"]
                             and fallback <= P["held_camera_max_fallback"] and low_gap <= 1e-6
                             and state["state_exact_hit_fraction"] >= 1-P["held_camera_max_fallback"]
                             and all(specimen.values()) and np.count_nonzero(mask) > 100)
    save_npz(root/"preflight_images.npz", adapter=np.stack(q), reference=np.stack(r), mask=mask)
    write(root/"preflight.json", metrics)
    print("PREFLIGHT",json.dumps({k:v for k,v in metrics.items() if k!="provenance"}),flush=True)
    if not metrics["passed"]:
        raise ValueError("The predeclared physical cross-check did not qualify")


def open_geometry(root):
    record = read(root/"preflight.json")
    if not record["passed"] or sha(root/"geometry.npz") != record["geometry_sha256"]:
        raise ValueError("Geometry must be validated and hash verified")
    if sha(HERE/"protocol.json") != record["provenance"]["protocol_sha256"]:
        raise ValueError("Protocol differs from preflight")
    with np.load(root/"geometry.npz") as z:
        return dict(z)


def rotations(config):
    rng = np.random.default_rng(config["seed"])
    result = []
    for _ in range(config["groups"] * config["lights_per_group"]):
        while True:
            angles = (float(rng.uniform(-180,180)), float(rng.uniform(-135,135)))
            delta = lambda a,b: (a-b+180)%360-180
            valid = all(np.hypot(delta(angles[0],t[0]),delta(angles[1],t[1]))>=30
                        for t in (P["evaluation"]["default_rotation"],P["evaluation"]["held_light_rotation"]))
            if valid:
                result.append(angles)
                break
    n = config["lights_per_group"]
    return [result[i:i+n] for i in range(0,len(result),n)]


def generate(root):
    g = open_geometry(root)
    ctx = Fabric(load_scene(32),g)
    complete = {}
    for split in ("train","val"):
        config = P[split]
        out = root/split
        out.mkdir(exist_ok=True)
        times = []
        group_rotations = rotations(config)
        for i, angles in enumerate(group_rotations):
            path = out/f"g{i:03d}.npz"
            if path.exists():
                with np.load(path) as prior:
                    if not np.array_equal(prior["rotations"],angles) or not all(np.isfinite(prior[k]).all() for k in ("s","f2","fh")):
                        raise ValueError(f"Invalid cached item {path}")
                continue
            t = time.monotonic()
            s, f2, fh, f2b, fhb = [],[],[],[],[]
            for j, (yaw,tilt) in enumerate(angles):
                ctx.light(yaw,tilt)
                seed = config["seed"]*1000+100*i+10*j
                s.append(ctx.source(seed))
                a,b = ctx.field(config["spp"],seed+1)
                f2.append(a)
                fh.append(b)
                if config["halves"] == 2:
                    a,b = ctx.field(config["spp"],seed+2)
                    f2b.append(a)
                    fhb.append(b)
            data = {n: np.stack(v) for n,v in (("s",s),("f2",f2),("fh",fh),("f2b",f2b),("fhb",fhb)) if v}
            checked(list(data.values()))
            save_npz(path, rotations=angles, **data)
            times.append(time.monotonic()-t)
            print("DATA",split,i,round(times[-1],3),flush=True)
        complete[split] = {"seconds_this_invocation": sum(times), "rotations": group_rotations,
                           "group_seconds": times, "sha256": {f.name:sha(f) for f in sorted(out.glob("g*.npz"))}}
    write(root/"data.json",complete)


def import_training():
    os.environ["WL102_SEED"] = str(P["fit"]["seed"])
    os.environ["WL102_AREA_FLOOR"] = str(P["fit"]["area_input_floor"])
    sys.path.insert(0,str(HERE.parent/"ecro_lr3_102"))
    import train102 as t
    return t


def train(root):
    meta = read(root/"data.json")
    open_geometry(root)
    for split, group in meta.items():
        if len(group["sha256"]) != P[split]["groups"] or any(sha(root/split/k)!=h for k,h in group["sha256"].items()):
            raise ValueError("Frozen training data mismatch")
    model = root/"fit"
    if (model/"model.pt").exists():
        raise ValueError("Fit already exists")
    import_training().main(root,model,P["fit"]["steps"])


def render_view(root, out, name, geom, model, gs, factors, camera, angles):
    r = P["evaluation"]
    scene = load_scene(r["resolution"],**camera)
    ctx = Fabric(scene,geom)
    if tuple(angles) != (0,0):
        ctx.light(*angles)
    t0 = time.monotonic()
    sources = [ctx.source(105551+i*117) for i in range(4)]
    torch = import_training().torch
    with torch.no_grad():
        source = torch.as_tensor(np.mean(sources,axis=0)[None],device="cuda")
        f2 = model.apply(factors,gs,source,use_global=False)[0].cpu().numpy()
        fh = model.apply(factors,gs,source,use_single=False)[0].cpu().numpy()
    B = mi.Float(np.ascontiguousarray(f2+fh,dtype=np.float32).ravel())
    prep = time.monotonic()-t0
    ss = {"reference":0., "physical":0., "ecro":0.}
    replicas = []
    for j, seed in enumerate(r["seeds"]):
        raw = {}
        for arm in ss:
            started = time.monotonic()
            if arm=="reference":
                raw["reference"] = mi.render(scene,spp=r["spp_each"],seed=seed+10000).numpy()
            else:
                fields = ctx.pixels(r["spp_each"],seed,table=B if arm=="ecro" else None)
                raw.update({arm+"_"+k:v for k,v in fields.items()})
            ss[arm] += time.monotonic()-started
            print("RENDER",name,j,arm,round(ss[arm],3),flush=True)
        checked(list(raw.values()))
        save_npz(out/f"{name}_r{j}.npz",**raw)
        replicas.append(raw)
    a = {k:np.mean([m[k] for m in replicas],axis=0) for k in replicas[0]}
    mask = a["physical_fabric_mask"]>.5
    low = a["ecro_e0"]+a["ecro_e1"]+a["ecro_e2"]
    ecro = low+a["ecro_hl"]
    full = a["physical_e0"]+a["physical_e1"]+a["physical_e2"]+a["physical_h"]
    ref = a["reference"]
    fallback = float(a["ecro_fallback"].sum()/max(float(a["ecro_lookup"].sum()),1.))
    gap = max(relative_l1(a["ecro_"+k],a["physical_"+k]) for k in ("e0","e1","e2"))
    field_seed = 105561 if name=="public" else 105569
    field_ref = [ctx.field(64,field_seed+i) for i in range(2)]
    rf2, rfh = (np.mean([it[j] for it in field_ref],axis=0) for j in range(2))
    save_npz(out/f"{name}.npz",**a,mask=mask,low=low,ecro=ecro,full=full,
             field_ref_f2=rf2,field_ref_fh=rfh,field_f2=f2,field_fh=fh)
    quality = {
        "mask_pixels": int(mask.sum()), "fallback_fraction":fallback,"physical_low_parity_max_l1":gap,
        "reference_half_fabric_l1":relative_l1(replicas[0]["reference"],replicas[1]["reference"],mask),
        "fabric_l1":{k:relative_l1(v,ref,mask) for k,v in (("ecro",ecro),("low",low),("physical",full))},
        "fabric_signed_energy":{k:signed_energy(v,ref,mask) for k,v in (("ecro",ecro),("low",low),("physical",full))},
        "full_l1":{k:relative_l1(v,ref) for k,v in (("ecro",ecro),("low",low),("physical",full))},
        "high_l1_vs_physical":relative_l1(a["ecro_hl"],a["physical_h"],mask),
        "high_signed_energy_vs_physical":signed_energy(a["ecro_hl"],a["physical_h"],mask),
        "high_fraction_physical":float(a["physical_h"][mask].sum()/max(float(full[mask].sum()),1e-20)),
        "patch_field_l1": {"single":relative_l1(f2,rf2),"global":relative_l1(fh,rfh),"total":relative_l1(f2+fh,rf2+rfh)},
        "patch_field_halves_l1":relative_l1(sum(field_ref[0]),sum(field_ref[1])),
    }
    if fallback>P["held_camera_max_fallback"] or gap>1e-5:
        raise ValueError(f"Camera transport parity failed: {name} {quality}")
    write(out/f"{name}.json",{"quality":quality,"angles":angles,"camera":camera,"resolution":r["resolution"],
                             "spp_each":r["spp_each"],"seeds":r["seeds"],"seconds":ss,"table_seconds":prep})
    return quality


def evaluate(root):
    T = import_training()
    torch, M = T.torch,T.M
    g = open_geometry(root)
    gs, rows = T.geostate(root)
    p = root/"fit/model.pt"
    model = M.ECRO().to(T.DEV)
    model.mat_states = rows
    model.load_state_dict(torch.load(p,map_location=T.DEV,weights_only=True))
    model.eval()
    with torch.no_grad():
        factors = model.factors(T.input_state(gs))
    path = root/"renders"
    path.mkdir(exist_ok=False)
    eval_cfg = P["evaluation"]
    started = time.monotonic()
    output = {}
    for view,camera,angles in (("public",{},eval_cfg["default_rotation"]),
                               ("held",eval_cfg["held_camera"],eval_cfg["held_light_rotation"])):
        output[view] = render_view(root,path,view,g,model,gs,factors,camera,angles)
    write(path/"summary.json",{"quality":output,"seconds":time.monotonic()-started,"model_sha256":sha(p),
                               "device_allocator":[line.strip() for line in dr.whos_str().splitlines() if "used (peak:" in line]})


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("stage",choices=("preflight","generate","train","evaluate"))
    cli.add_argument("root",type=Path)
    arg = cli.parse_args()
    {"preflight":preflight,"generate":generate,"train":train,"evaluate":evaluate}[arg.stage](arg.root)


if __name__=="__main__":
    main()
