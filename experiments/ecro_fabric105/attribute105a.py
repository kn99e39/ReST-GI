"""Separate Fabric HIGH table prediction from patch pooling and readout."""

import argparse
import time
from pathlib import Path

import numpy as np

from fabric105 import Fabric, F, HERE, NB, PT, U, dr, mi, load_scene
from run105 import P, coarse, commit, open_geometry, read, relative_l1, save_npz, sha, signed_energy, write, ROOT

C = read(HERE / "attribution105a.json")
PREC = read(HERE / "precision105.json")
PUBLIC, HELD = "public", "held"
VIEWS = {PUBLIC: {}, HELD: P["evaluation"]["held_camera"]}
LIGHTS = {PUBLIC: P["evaluation"]["default_rotation"], HELD: P["evaluation"]["held_light_rotation"]}


def parent_data(root, name):
    directory = root / PREC["output"]
    hashes = read(ROOT / "evaluation/105/high_spp/manifest.json")["raw_sha256"]

    def load(path):
        if sha(path) != hashes[path.relative_to(root).as_posix()]:
            raise ValueError(f"Frozen artifact mismatch: {path}")
        with np.load(path) as z:
            return dict(z)

    data = load(directory / f"{name}.npz")
    old = [load(directory / f"{name}_r{i}.npz") for i in range(2)]
    return data["field_f2"] + data["field_fh"], data["mask"], old


def context(geom, view, light, size):
    ctx = Fabric(load_scene(size, **VIEWS[view]), geom)
    if tuple(LIGHTS[light]) != (0, 0):
        ctx.light(*LIGHTS[light])
    return ctx


def entry(ctx, w, h, rng, n):
    """Replay the physical camera's first scattering to get a weighted second hit."""
    pix = dr.arange(U, n) % (w*h)
    u = mi.Point2f((F(pix % w)+rng.next_float32())/w, (F(pix//w)+rng.next_float32())/h)
    sensor = ctx.scene.sensors()[0]
    ray, _ = sensor.sample_ray(0., .5, u, mi.Point2f(.5,.5))
    first = ctx.scene.ray_intersect(mi.Ray3f(ray))
    active = first.is_valid()
    # Use the exact two RNG draws spent by Fabric.trace on first-hit NEE.
    rng.next_float32()
    rng.next_float32()
    bs, thr = first.bsdf().sample(ctx.bctx, first, rng.next_float32(),
                                  mi.Point2f(rng.next_float32(),rng.next_float32()), active)
    used = active & (bs.pdf > 0) & (dr.max(thr) > 0)
    second = ctx.scene.ray_intersect(first.spawn_ray(first.to_world(bs.wo)), used)
    return pix, second, dr.select(used & second.is_valid(), thr, 0.)


def fit_camera_table(ctx, samples, seed, area):
    sensor = ctx.scene.sensors()[0]
    w, h = (int(v) for v in sensor.film().crop_size())
    pixels = w*h
    bins = ctx.n*NB
    launch = max(1, (1 << 19)//pixels)
    accum = {k: np.zeros((bins,3), np.float64) for k in ("num", "den")}
    num_replay = np.zeros((pixels,3), np.float64)
    for i, start in enumerate(range(0,samples,launch)):
        n = pixels*min(launch,samples-start)
        rng = mi.PCG32(size=n, initstate=dr.opaque(mi.UInt64, seed*1000003+i))
        pix, si, thr = entry(ctx,w,h,rng,n)
        pid, bid, _ = ctx.address(si.p, si.to_world(si.wi))
        after = ctx.trace(si, rng, P["field_bounces"])
        high = after["e2"] + after["h"]
        wt = thr*thr
        values = {"num": wt*high, "den": wt}
        fields = PT.Ctx.accum({k+str(c):v[c] for k,v in values.items() for c in range(3)}, pid*NB+bid,bins)
        for k in accum:
            accum[k] += np.stack([fields[k+str(c)] for c in range(3)], axis=-1)
        weighted = thr*high
        re = PT.Ctx.accum({str(c):weighted[c] for c in range(3)},pix,pixels)
        num_replay += np.stack([re[str(c)] for c in range(3)], axis=-1)
    den = accum["den"]
    fitted = np.divide(accum["num"], den, out=area.reshape(-1,3).astype(np.float64).copy(), where=den>0)
    return fitted.reshape(area.shape).astype(np.float32), den, num_replay.reshape(h,w,3)/samples


def area_table(ctx, seed):
    single, multiple = ctx.field(C["area_field_spp_per_half"],seed)
    return single+multiple


def render_table(ctx, spp, seed, table, old):
    t = mi.Float(np.ascontiguousarray(table,dtype=np.float32).ravel())
    result = ctx.pixels(spp,seed,table=t)
    err = max(relative_l1(result[k], old["physical_"+k]) for k in ("e0","e1","e2"))
    if err > 1e-6:
        raise ValueError(f"Paired oracle readout changed LOW: {err}")
    return result["hl"], err


def detail(a):
    return a - coarse(a).repeat(8,axis=0).repeat(8,axis=1)


def mean_product(a, b, mask):
    if mask is not None:
        a,b = a[mask],b[mask]
    return float(np.mean(a*b))


def scores(ref, low, arms, mask):
    physical = ref[0]/2 + ref[1]/2
    low_mean = (low[0]+low[1])/2
    row = {}
    for name, pair in arms.items():
        v = (pair[0]+pair[1])/2
        row[name] = {
            "signed_HIGH_luminance": signed_energy(v,physical,mask),
            "HIGH_relative_l1": relative_l1(v,physical,mask),
            "total_pixel_l1_vs_physical": relative_l1(low_mean+v,low_mean+physical,mask),
            "total_coarse_l1_vs_physical": relative_l1(coarse(low_mean+v),coarse(low_mean+physical)),
            "detail_l1_vs_physical_HIGH": float(np.abs(detail(v-physical)[mask]).sum()/np.abs(detail(physical)[mask]).sum()),
            "detail_amplitude_fraction": float(np.abs(detail(v)[mask]).sum()/np.abs(detail(physical)[mask]).sum()),
        }
    terms = {}
    for transform in ("pixels","coarse","detail"):
        fn = {"pixels":lambda v:v,"coarse":coarse,"detail":detail}[transform]
        subset = None if transform=="coarse" else mask
        a = [fn(x-y) for x,y in zip(arms["area"],ref)]
        b = [fn(x-y) for x,y in zip(arms["learned"],arms["area"])]
        total = [x+y for x,y in zip(a,b)]
        rr = mean_product(fn(ref[0]),fn(ref[1]),subset)
        av, bv, tv = (mean_product(*v,subset) / rr for v in (a,b,total))
        camera = [fn(x-y) for x,y in zip(arms["camera"],ref)]
        terms[transform] = {"table_error_mean_square":bv,"pooling_error_mean_square":av,
                            "cross_term":tv-av-bv,"full_mean_square":tv,
                            "camera_oracle_error_mean_square":mean_product(*camera,subset)/rr,"HIGH_energy_denom":rr}
    return row, terms


def main_oracles(root, out, geometry, name):
    t = time.monotonic()
    learned,mask,old = parent_data(root,name)
    width = P["evaluation"]["resolution"]
    ctx = context(geometry,name,name,width)
    arms = {"learned":[m["ecro_hl"] for m in old],"area":[],"camera":[]}
    native = [m["physical_h"] for m in old]
    low = [m["physical_e0"]+m["physical_e1"]+m["physical_e2"] for m in old]
    area, camera, support, parity = [],[],[],[]
    fit_behaviour = []
    for i, seed in enumerate(PREC["seeds"]):
        a = area_table(ctx,C["area_seeds"][name][i])
        b,d,replayed = fit_camera_table(ctx,C["camera_oracle_fit_spp_per_half"],C["camera_fit_seeds"][name][i],a)
        support.append(float(np.mean(np.sum(d,axis=1)>0)))
        if not (np.isfinite(a).all() and np.isfinite(b).all() and np.isfinite(replayed).all() and np.min(b)>=0):
            raise ValueError("Invalid diagnostic table")
        # Replay the first-hit segmentation on independent fit seeds and verify H.
        if i == 0:
            direct = ctx.pixels(C["camera_oracle_fit_spp_per_half"], C["camera_fit_seeds"][name][i])
            g = relative_l1(replayed,direct["h"])
            if g>1e-5:
                raise ValueError(f"Camera fitting uses the wrong scattering event: {g}")
            fit_behaviour.append(g)
        for mode,table in (("area",a),("camera",b)):
            img,err = render_table(ctx,PREC["spp_each"],seed,table,old[i])
            arms[mode].append(img)
            parity.append(err)
        area.append(a)
        camera.append(b)
        print("ORACLE",name,i,round(time.monotonic()-t,2),flush=True)
    metrics,scale = scores(native,low,arms,mask)
    payload = {"comparison":metrics,"mean_square":scale,"camera_bin_support":support,"low_parity":parity,
               "camera_high_identity":fit_behaviour,
               "area_tables_half_l1":relative_l1(*area),"camera_tables_half_l1":relative_l1(*camera),
               "learnt_vs_area_mean_l1":relative_l1(learned,np.mean(area,axis=0)),
               "seconds":time.monotonic()-t}
    save_npz(out/f"{name}.npz",mask=mask,physical=np.stack(native),low=np.stack(low),
             **{a:np.stack(v) for a,v in arms.items()},area_tables=np.stack(area),camera_tables=np.stack(camera))
    write(out/f"{name}.json",payload)
    return learned,payload


def crossover(out,geometry,tables):
    config = C["light_view_cross"]
    result = {}
    for view in VIEWS:
        for light in LIGHTS:
            ctx = context(geometry,view,light,config["resolution"])
            table = mi.Float(np.ascontiguousarray(tables[light],dtype=np.float32).ravel())
            rows, values = [], []
            for seed in config["seeds"]:
                n = ctx.pixels(config["spp_per_half"],seed)
                l = ctx.pixels(config["spp_per_half"],seed,table=table)
                mask = n["fabric_mask"]>.5
                low = n["e0"]+n["e1"]+n["e2"]
                if max(relative_l1(l[k],n[k]) for k in ("e0","e1","e2")) > 1e-6:
                    raise ValueError("LOW is not paired in the crossover test")
                rows.append({"signed_HIGH":signed_energy(l["hl"],n["h"],mask),
                             "signed_TOTAL":signed_energy(low+l["hl"],low+n["h"],mask)})
                values.append((low,n["h"],l["hl"],n["fabric_mask"]))
            low,high,pred,coverage = (np.mean([v[j] for v in values],axis=0) for j in range(4))
            mask = coverage>.5
            result[view+"_"+light] = {"signed_HIGH":signed_energy(pred,high,mask),
                                     "signed_TOTAL":signed_energy(low+pred,low+high,mask),"halves":rows}
            print("CROSS",view,light,flush=True)
    write(out/"crossover.json",result)
    return result


def reuse_view(parent, directory, view):
    learned, mask, old = parent_data(parent,view)
    record = read(directory/f"{view}.json")
    with np.load(directory/f"{view}.npz") as saved:
        fresh = {"mask":mask,"physical":np.stack([r["physical_h"] for r in old]),
                 "learned":np.stack([r["ecro_hl"] for r in old]),
                 "low":np.stack([r["physical_e0"]+r["physical_e1"]+r["physical_e2"] for r in old])}
        if not all(np.array_equal(saved[k],v) for k,v in fresh.items()):
            raise ValueError(f"Saved {view} comparison differs from the frozen camera evidence")
        score, ms = scores(saved["physical"],saved["low"],
                           {a:saved[a] for a in ("learned","area","camera")},mask)
    if score != record["comparison"] or ms != record["mean_square"]:
        raise ValueError(f"Saved {view} metrics do not reproduce")
    return learned,record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("--resume-cross",action="store_true")
    args = parser.parse_args()
    directory = args.output
    geometry = open_geometry(args.parent)
    if args.resume_cross:
        if any((directory/f"{n}.json").exists() for n in ("crossover","summary")):
            raise ValueError("Crossover output already exists")
    else:
        directory.mkdir(exist_ok=False,parents=True)
    started = time.monotonic()
    tables,results = {},{}
    for view in VIEWS:
        if args.resume_cross:
            tables[view],results[view] = reuse_view(args.parent,directory,view)
        else:
            tables[view],results[view] = main_oracles(args.parent,directory,geometry,view)
    cross = crossover(directory,geometry,tables)
    write(directory/"summary.json",{"views":results,"crossover":cross,"seconds_this_invocation":time.monotonic()-started,
                                     "resumed_crossover":args.resume_cross,
                                     "execution_commit":commit(ROOT),"protocol_sha256":sha(HERE/"attribution105a.json")})


if __name__ == "__main__":
    main()
