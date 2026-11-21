"""Matched 6 cm Fabric adapter trial using the frozen 18 cm reference."""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import run105 as base
from attribute105a import PREC, VIEWS, context, detail, mean_product, parent_data
from fabric105 import HERE, NB, ROOT, mi

CFG = HERE/"refinement105b.json"
B = base.read(CFG)
P = base.P
NAMES = ("coarse_learned","fine_learned","coarse_area","fine_area")


def check_sha(path, expected):
    if base.sha(path) != expected:
        raise ValueError(f"Hash mismatch for {path}")


def parent_artifacts(parent):
    manifest = base.read(ROOT/"evaluation/105/high_spp/manifest.json")["raw_sha256"]
    for name in ("geometry.npz","fit/model.pt","fit/training.json"):
        check_sha(parent/name,manifest[name])
    return base.open_geometry(parent)


def cell_coordinates(g):
    return np.stack(np.unravel_index(g["key"],g["grid_dim"]),axis=-1)+g["grid_min"]


def verify_partition(parent, candidate):
    ratio = P["patch_cell_m"]/B["fine_cell_m"]
    if not np.isclose(ratio,round(ratio)):
        raise ValueError("The candidate grid should be nested")
    coarse = {tuple(p):float(a) for p,a in zip(cell_coordinates(parent),parent["parea"][::2]*2)}
    merged = dict.fromkeys(coarse,0.)
    for point,area in zip(cell_coordinates(candidate)//round(ratio),candidate["parea"][::2]*2):
        key = tuple(point)
        merged[key] = merged.get(key,0.)+float(area)
    diff = sum(abs(a-coarse.get(k,0.)) for k,a in merged.items())/sum(coarse.values())
    if diff > B["validity"]["parent_area_partition_max_l1"] or not np.array_equal(parent["rows"],candidate["rows"]):
        raise ValueError(f"Grid refinement did not preserve weight or input material: {diff}")
    return float(diff)


def preflight(parent, out):
    g = parent_artifacts(parent)
    base.preflight(out,cell=B["fine_cell_m"])
    fine = base.open_geometry(out)
    info = {"protocol_sha256":base.sha(CFG),"commit":base.commit(ROOT),"parent_partition_rel_l1":verify_partition(g,fine),
            "parent_patches":len(g["pc"]),"fine_patches":len(fine["pc"]),
            "direction_bins":NB,"parent_checkpoint_sha256":base.sha(parent/"fit/model.pt"),
            "fine_geometry_sha256":base.sha(out/"geometry.npz")}
    base.write(out/"refinement.json",info)


def candidate_geometry(out):
    info = base.read(out/"refinement.json")
    check_sha(CFG,info["protocol_sha256"])
    check_sha(out/"geometry.npz",info["fine_geometry_sha256"])
    geom = base.open_geometry(out)
    if float(geom["grid_cell_m"]) != B["fine_cell_m"]:
        raise ValueError("The candidate geometry is at the wrong refinement")
    return geom


def frozen_view(parent, attribution, name):
    _,mask,old = parent_data(parent,name)
    src = attribution/f"{name}.npz"
    hashes = base.read(ROOT/"evaluation/105A/manifest.json")["raw_sha256"]
    check_sha(src,hashes[src.name])
    native = np.stack([r["physical_h"] for r in old])
    low = np.stack([r["physical_e0"]+r["physical_e1"]+r["physical_e2"] for r in old])
    learned = np.stack([r["ecro_hl"] for r in old])
    with np.load(src) as z:
        if not np.array_equal(mask,z["mask"]) or not np.array_equal(native,z["physical"]):
            raise ValueError("WL105 and WL105A references differ")
        coarse_area = z["area"]
    return mask,old,native,low,learned,coarse_area


def render_table(ctx, table, ref, seed):
    t = time.monotonic()
    buf = mi.Float(np.ascontiguousarray(table,dtype=np.float32).ravel())
    ans = ctx.pixels(PREC["spp_each"],seed,table=buf)
    gap = max(base.relative_l1(ans[name],ref["physical_"+name]) for name in ("e0","e1","e2"))
    fallback = float(ans["fallback"].sum()/max(ans["lookup"].sum(),1))
    base.checked([ans["hl"]])
    if gap>B["validity"]["native_low_max_rel_l1"] or fallback>B["validity"]["camera_fallback_max"]:
        raise ValueError(f"Unqualified camera readout low gap={gap} fallback={fallback}")
    return ans["hl"],{"seconds":time.monotonic()-t,"physical_low_gap":gap,"fallback":fallback}


def target_table(ctx, seed):
    single,multi = ctx.field(4096,seed)
    return single+multi


def oracle(parent, attribution, out, view, geometry):
    start = time.monotonic()
    _,old,_,_,_,_ = frozen_view(parent,attribution,view)
    ctx = context(geometry,view,view,P["evaluation"]["resolution"])
    areas = [target_table(ctx,seed) for seed in B["area_seeds"][view]]
    rendered, controls = zip(*(render_table(ctx,table,raw,seed) for table,raw,seed in zip(areas,old,PREC["seeds"])))
    path = out/f"oracle_{view}.npz"
    base.save_npz(path,area=np.stack(areas),image=np.stack(rendered))
    base.write(out/f"oracle_{view}.json",{"seconds":time.monotonic()-start,"camera":controls,
                                           "area_fields_half_l1":base.relative_l1(*areas),"sha256":base.sha(path)})
    print("ORACLE",view,flush=True)


def learned_table(ctx,out):
    T = base.import_training()
    torch = T.torch
    gs,rows = T.geostate(out)
    model = T.M.ECRO().to(T.DEV)
    model.mat_states = rows
    model.load_state_dict(torch.load(out/"fit/model.pt",map_location=T.DEV,weights_only=True))
    model.eval()
    started = time.monotonic()
    with torch.no_grad():
        factors = model.factors(T.input_state(gs))
        sources = [ctx.source(105551+i*117) for i in range(4)]
        source = torch.as_tensor(np.mean(sources,axis=0)[None],device=T.DEV)
        single = model.apply(factors,gs,source,use_global=False)[0]
        multi = model.apply(factors,gs,source,use_single=False)[0]
        table = (single+multi).cpu().numpy()
    return table,time.monotonic()-started


def arm_scores(a, mask, native, low):
    target = low+native
    value = low+a
    detail_target = [detail(n) for n in native]
    delta = [detail(x-y) for x,y in zip(a,native)]
    mean = lambda x: np.mean(x,axis=0)
    return {"total_pixel_l1":base.relative_l1(mean(value),mean(target),mask),
            "total_pixel_l1_by_half":[base.relative_l1(x,y,mask) for x,y in zip(value,target)],
            "total_coarse_l1":base.relative_l1(base.coarse(mean(value)),base.coarse(mean(target))),
            "total_signed_bias":base.signed_energy(mean(value),mean(target),mask),
            "high_signed_bias":base.signed_energy(mean(a),mean(native),mask),
            "high_detail_mse":mean_product(*delta,mask)/mean_product(*detail_target,mask)}


def evaluate(parent, attribution, out, view, geometry):
    t = time.monotonic()
    mask,old,native,low,coarse_learned,coarse_area = frozen_view(parent,attribution,view)
    file = out/f"oracle_{view}.npz"
    audit = base.read(out/f"oracle_{view}.json")
    check_sha(file,audit["sha256"])
    with np.load(file) as z:
        fine_area,area_tables = z["image"],z["area"]
    ctx = context(geometry,view,view,P["evaluation"]["resolution"])
    table,prep = learned_table(ctx,out)
    rendered, controls = zip(*(render_table(ctx,table,r,s) for r,s in zip(old,PREC["seeds"])))
    values = dict(zip(NAMES,(coarse_learned,np.stack(rendered),coarse_area,fine_area)))
    stats = {k:arm_scores(v,mask,native,low) for k,v in values.items()}
    path = out/f"view_{view}.npz"
    base.save_npz(path,**values,mask=mask,physical=native,low=low,fine_table=table)
    base.write(out/f"view_{view}.json",{"seconds":time.monotonic()-t,"table_seconds":prep,"camera":controls,
                                         "arms":stats,"fine_prediction_area_l1":base.relative_l1(table,area_tables.mean(0)),
                                         "sha256":base.sha(path)})
    print("EVAL",view,flush=True)


def summarize(parent,out):
    base_fit = base.read(parent/"fit/training.json")
    fine_fit = base.read(out/"fit/training.json")
    if any(base_fit[k]!=fine_fit[k] for k in ("model86_sha256","params","steps","seed","objective","recipe")):
        raise ValueError("The model or matched fit configuration changed")
    views = {view:base.read(out/f"view_{view}.json") for view in VIEWS}
    gate = B["interpretation"]
    checks = {}
    for view,result in views.items():
        arms = result["arms"]
        old,new = (arms[k] for k in ("coarse_learned","fine_learned"))
        area_old,area_new = (arms[k] for k in ("coarse_area","fine_area"))
        checks[view] = {
            "learned_detail_gain_relative":1-new["high_detail_mse"]/old["high_detail_mse"],
            "area_detail_gain_relative":1-area_new["high_detail_mse"]/area_old["high_detail_mse"],
            "learned_pixel_gain":old["total_pixel_l1"]-new["total_pixel_l1"],
            "area_pixel_gain":area_old["total_pixel_l1"]-area_new["total_pixel_l1"],
            "both_halves_pixel_gain":all(a>b for a,b in zip(old["total_pixel_l1_by_half"],new["total_pixel_l1_by_half"])),
            "absolute_bias_regression":abs(new["total_signed_bias"])-abs(old["total_signed_bias"]),
        }
        c = checks[view]
        c["qualified_recovery"] = bool(c["learned_detail_gain_relative"]>=gate["minimum_detail_gain_relative"]
                                     and c["learned_pixel_gain"]>=gate["minimum_learned_total_pixel_gain"]
                                     and c["both_halves_pixel_gain"]
                                     and c["absolute_bias_regression"]<=gate["held_out_absolute_bias_regression_max"])
    result = {"commit":base.commit(ROOT),"protocol_sha256":base.sha(CFG),
              "geometry":base.read(out/"refinement.json"),"native_valid":True,"views":views,"checks":checks,
              "joint_qualified_recovery":all(c["qualified_recovery"] for c in checks.values()),
              "parent_fit_s":base_fit["train_s"],"candidate_fit_s":fine_fit["train_s"],"candidate_fit_peak_mib":fine_fit["peak_mib"]}
    base.write(out/"summary.json",result)
    print("WL105B",result["joint_qualified_recovery"],flush=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("parent",type=Path)
    cli.add_argument("attribution",type=Path)
    cli.add_argument("output",type=Path)
    cli.add_argument("stage",nargs="?",default="all",
                     choices=("all","preflight","generate","train","oracle-public","oracle-held","eval-public","eval-held","summary"))
    a = cli.parse_args()
    parent_artifacts(a.parent)
    if a.stage=="all":
        for stage in ("preflight","oracle-public","oracle-held","generate","train","eval-public","eval-held","summary"):
            subprocess.run([sys.executable,"-B",__file__,str(a.parent),str(a.attribution),str(a.output),stage],check=True)
    elif a.stage=="preflight":
        preflight(a.parent,a.output)
    else:
        geometry = candidate_geometry(a.output)
        if a.stage in ("generate","train"):
            {"generate":base.generate,"train":base.train}[a.stage](a.output)
        elif a.stage=="summary":
            summarize(a.parent,a.output)
        else:
            kind,view = a.stage.split("-")
            (oracle if kind=="oracle" else evaluate)(a.parent,a.attribution,a.output,view,geometry)


if __name__=="__main__":
    main()
