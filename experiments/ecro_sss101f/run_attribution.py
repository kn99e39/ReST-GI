"""Frozen WL101F physical query, projection and conditional refinement run."""
import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
from scipy.stats import t

from projections import make_tables, render, train, uniform_geometry, verify_projection

from query_readout import QueryReadout, LUMA

HERE = Path(__file__).resolve().parent
for sibling in ("ecro_sss101e", "ecro_sss101d", "ecro_sss101a"):
    sys.path.insert(0, str(HERE.parent / sibling))
from oracle_field import ExposedOracle, FirstSurface
from run_oracle import (A, array_hash, fingerprint, foreign_gpu, gpu_query, rel_l1,
                        sha, validate_inputs, wait_for_gpu, write_json)
from common import masks

PROTOCOL = HERE / "protocol/attribution.json"
P = json.loads(PROTOCOL.read_text())
G = P["gates"]
NEAR_LUMA = np.array(LUMA)


def inputs(root):
    reference = root / A["references"]["path_in_wl101"]
    sources = validate_inputs(root, reference)
    prior = root.parent / "results/ecro_sss101a/qualified63"
    base = root.parent / "results/ecro_sss101e/recovery2"
    assert sha(base/"run_manifest.json") == P["baseline_manifest_sha256"]
    wl101e = json.loads((base/"run_manifest.json").read_text())
    assert sha(prior/"run_manifest.json") == wl101e["prior_manifest_sha256"]
    assert sources == wl101e["renderer_sources_sha256"]
    history = json.loads((prior/"run_manifest.json").read_text())
    wait_for_gpu()
    assert [p.strip() for p in gpu_query("gpu","name,driver_version").split(",")] == [
        "NVIDIA GeForce RTX 3080 Ti", "595.71.05"]
    sys.path.insert(0, str(root/"experiments/ecro_dyn101"))
    os.environ["FREEZE"] = "1"
    import rt101 as R
    import seq101 as seq
    assert (R.mi.__version__,R.dr.__version__,R.torch.__version__)==("3.8.0","1.3.1","2.3.1+cu121")
    case,start,events = seq.workload(A["workload"])
    cam = dict(R.C100.P["datasets"][case]["camera"],res=512)
    assert cam==A["camera"] and events[seq.WARM]==(A["geometry"],A["light"],0)
    rt=R.RT(start[0],case,R.load_model(case),start[1])
    for geo, light, _ in events[:seq.WARM+1]:
        moved,old=rt.set_geom(geo)
        rt.update_state(old,moved)
        rt.set_light(light)
        learned,_=rt.respond(rt.source())
    learned=learned.cpu().numpy().copy()
    scene = fingerprint({k:A[k] for k in ("geometry","camera","sphere","light")})
    assert scene == wl101e["scene_sha256"] and array_hash(learned)==wl101e["learned_table_sha256"]
    optix=next(Path(line.split()[-1]) for line in Path("/proc/self/maps").read_text().splitlines()
               if "/libnvoptix.so." in line)
    assert optix.name=="libnvoptix.so.595.71.05" and sha(optix)==wl101e["environment"]["optix_sha256"]
    manifest={k:wl101e[k] for k in ("scene_sha256","checkpoint_sha256","cfg_sha256",
                                   "learned_table_sha256","input_sha256","renderer_sources_sha256")}
    manifest["environment"] = {"gpu":R.torch.cuda.get_device_name(),"driver":"595.71.05",
                               "optix_sha256":sha(optix),"mitsuba":R.mi.__version__,
                               "drjit":R.dr.__version__,"torch":R.torch.__version__,
                               "python":platform.python_version(),"numpy":np.__version__,
                               "os":platform.platform()}
    return QueryReadout(R,rt,cam), learned, base, wl101e, prior, history, reference, manifest


def scalar(rgb, regions):
    y=np.asarray(rgb,dtype=np.float64)@NEAR_LUMA
    near=float(y[regions["near"]].mean())
    return np.array([near,near-float(y[regions["flank"]].mean())])


def interval(values):
    n,m=values.shape
    u,v=values.mean(1).var(ddof=1)/n,values.mean(0).var(ddof=1)/m
    half=0.
    df=None
    if u+v>0:
        df=(u+v)**2/(u*u/(n-1)+v*v/(m-1))
        half=float(t.ppf(.975,df)*np.sqrt(u+v))
    avg=float(values.mean())
    return {"mean":avg,"ci95":[avg-half,avg+half],"half_width95":half,
            "field_se":float(np.sqrt(u)),"camera_se":float(np.sqrt(v)),
            "degrees_of_freedom":df,"cells":values.tolist()}


def intervals(values):
    return {name:interval(values[...,i]) for i,name in enumerate(("near","concentration"))}


def primary(reader,learned,base,wl101e,prior,history,reference,out,run):
    R=reader.R
    run["stages"]={}
    run["files"]={}
    run["gpu_memory_observations_mib"]=[]
    run["projection_normalization_control"]=verify_projection()
    try:
        run_pipeline(reader,learned,base,wl101e,prior,history,reference,out,run)
    except AssertionError as error:
        run["stop"]=" ".join(str(a) for a in error.args) or "Unclassified assertion failure; see execution log."
        write_json(out/"run_manifest.json",run)
        raise
    finally:
        R.sync()


def run_pipeline(reader,learned,base,wl101e,prior,history,reference,out,run):
    R=reader.R
    seeds=P["camera_seeds"]
    all_seeds=[*seeds,*P["training_seeds"],*P["training_suffix_seeds"],
               P["query_control_seed"],P["uniform_audit_seed"]]
    assert len(set(all_seeds))==len(all_seeds)
    assert seeds==[r["seed"] for r in history["matched"]]
    masks_file=reference/"MAP_0.npz"
    regions=masks(masks_file)
    with np.load(masks_file) as m:
        pos=m["p"]
    center=np.asarray(A["sphere"]["center"])
    r=np.linalg.norm(pos[..., :2]-center[:2],axis=-1)
    front=regions["floor"] & (pos[...,1]<=center[1])
    regions["near"]=front&(r<.18)
    regions["flank"]=front&(r>=.26)&(r<.40)
    assert regions["near"].sum()==1318 and regions["flank"].sum()==6042
    assert np.all(regions["contact_shadow"][regions["near"]])

    active_stage="initial"
    peak=0
    def monitor(message=None):
        nonlocal peak
        if foreign_gpu():
            wait_for_gpu()
        memory=int(gpu_query("gpu","memory.used").split()[0])
        peak=max(peak,memory)
        if message is not None:
            print(f"{active_stage}: {message} ({memory} MiB)",flush=True)

    def timing(label,action,*args,**kwargs):
        nonlocal active_stage,peak
        active_stage,peak=label,0
        monitor(f"starting {label}")
        R.sync()
        stamp=time.perf_counter()
        result=action(*args,**kwargs)
        R.sync()
        seconds=time.perf_counter()-stamp
        monitor(f"finished {label} in {seconds:.2f}s")
        run["gpu_memory_observations_mib"].append({"stage":label,"max_observed_mib":peak,"seconds":seconds})
        return result,seconds

    def store(filename,**arrays):
        path=out/filename
        np.savez_compressed(path,**arrays)
        row={"file":filename,"sha256":sha(path),"bytes":path.stat().st_size}
        run["files"][filename]=row
        return row

    def save_progress():
        write_json(out/"progress.json",run)

    uniform=[]
    for row in wl101e["oracle_tables"]:
        file=base/row["file"]
        assert sha(file)==row["sha256"]
        with np.load(file) as f:
            field=f["B"]
        assert array_hash(field)==row["array_sha256"]
        uniform.append(field)

    run["stages"]["A"]={"streams":[],"passed":False}
    original=[]
    legacy=np.empty((4,2,2),np.float64)
    for j,seed in enumerate(seeds):
        archpath=prior/history["matched"][j]["path"]
        assert sha(archpath)==history["matched"][j]["sha256"]
        with np.load(reference/f"REF_0_{j}.npz") as f:
            ref=f["h"]
            physical_low=f["e1"],f["e2"]
        original.append(scalar(ref,regions))
        # The primary baseline is the four archived WL101E uniform field readouts.
        for row in (r for r in wl101e["readouts"] if r["camera_half"]==j):
            path=base/row["file"]
            assert sha(path)==row["sha256"] and row["camera_seed"]==seed
            with np.load(path) as f:
                legacy[row["oracle_index"],j]=scalar(f["hl"],regions)
        first=wl101e["readouts"][j]
        assert first["oracle_index"]==0 and first["camera_seed"]==seed
        baseline={"U_0":(uniform[0],np.ones_like(uniform[0],dtype=bool))}
        (result,row),elapsed=timing(f"A_camera_{seed}",render,reader,FirstSurface,P,seed,
                                    baseline,monitor,query_control=True,learned=learned)
        with np.load(archpath) as f:
            errors={k:rel_l1(result[k],f[k]) for k in ("e1","e2")}
            learned_error=rel_l1(result["learned"],f["hl"])
        with np.load(base/first["file"]) as f:
            uniform_error=rel_l1(result["U_0"],f["hl"])
        p_errors={k:rel_l1(result[k],p) for k,p in zip(("e1","e2"),physical_low)}
        query_error=rel_l1(result["query"],ref)
        query_roi=scalar(result["query"],regions)-original[-1]
        row.update(seconds=elapsed,learned_low_rel_l1=errors,physical_low_rel_l1=p_errors,
                   old_learned_high_rel_l1=learned_error,old_uniform_high_rel_l1=uniform_error,
                   query_high_rel_l1=query_error,query_high_endpoints=query_roi.tolist(),
                   **store(f"query_{seed}.npz",**result))
        run["stages"]["A"]["streams"].append(row)
        save_progress()
        assert max([*errors.values(),*p_errors.values()])<=G["low_relative_l1"], ("LOW replay",row)
        assert max(learned_error,uniform_error,query_error)<=G["matched_rgb_relative_l1"], ("Matched HIGH",row)
        assert max(abs(query_roi))<=G["matched_roi_max_absolute"], ("Physical reference endpoints",row)
    run["stages"]["A"]["passed"]=True
    original=np.stack(original)
    legacy-=original[None]
    base_error=np.abs(legacy.mean((0,1)))
    run["baseline_residual"]=intervals(legacy)
    save_progress()

    run["stages"]["B"]={"passed":False,"training":[]}
    table_samples={}
    (population,geom_meta),elapsed=timing("B_uniform_audit",uniform_geometry,reader,
                                          ExposedOracle(R,reader.rt),P,monitor)
    run["stages"]["B"]["uniform_audit"]={**geom_meta,"seconds":elapsed,
                                         **store("uniform_geometry.npz",counts=population)}
    for i,(seed,tail_seed) in enumerate(zip(P["training_seeds"],P["training_suffix_seeds"])):
        (packet,row),elapsed=timing(f"B_training_{i}",train,reader,FirstSurface,regions,P,seed,tail_seed,monitor)
        run["stages"]["B"]["training"].append({**row,"seconds":elapsed,**store(f"usage_{i}.npz",**packet)})
        table_samples[i]=packet
        save_progress()

    def evaluate(stage,names):
        gathered={}
        active={}
        frames=run["stages"][stage].setdefault("streams",[])
        for i,packet in table_samples.items():
            tables=make_tables(packet,uniform[i],names)
            gathered.update({f"{key}_{i}":value for key,value in tables.items()})
            active.update({f"{key}_{i}":int(np.count_nonzero(value[1].all(-1))) for key,value in tables.items()})
        run["stages"][stage]["native_rgb_supported_cells"]=active
        store(f"tables_{stage}.npz",**{key:value[0] for key,value in gathered.items()},
              **{f"{key}:native":value[1] for key,value in gathered.items()})
        values={name:np.zeros((4,2,2),dtype=np.float64) for name in names}
        for j,seed in enumerate(seeds):
            (images,row),elapsed=timing(f"{stage}_camera_{seed}",render,reader,FirstSurface,P,seed,gathered,monitor)
            # Full prefix history and reduction must stay the same at every phase.
            with np.load(out/f"query_{seed}.npz") as old:
                assert rel_l1(images["total_weight"],old["total_weight"])<=1e-6, "Camera use replay"
            weights=images["total_weight"].astype(np.float64)
            coverage={}
            for key in gathered:
                name,i=key.split("_")
                values[name][int(i),j]=scalar(images[key],regions)-original[j]
                marked=images[key+":weight"].astype(np.float64)
                coverage[key]={region:float(marked[mask].sum()/weights[mask].sum())
                               for region,mask in regions.items()}
            frames.append({**row,"seconds":elapsed,"native_transport_weight_fraction":coverage,
                           **store(f"camera_{stage}_{seed}.npz",**images)})
            save_progress()
        return values

    def validate_coverage(stage,names):
        for r in run["stages"][stage]["streams"]:
            for name,fractions in r["native_transport_weight_fraction"].items():
                if name.split("_")[0] in names:
                    assert min(fractions["near"],fractions["flank"])>=G["min_held_out_native_weight_fraction"], (
                        "Held-out native weight coverage",stage,r["seed"],name,fractions)

    def qualified(families):
        for name,grid in families.items():
            bars=intervals(grid)
            half=np.array([bar["half_width95"] for bar in bars.values()])
            assert np.all(half<=G["precision_fraction_of_wl101e_residual"]*base_error), ("Precision", name,bars)

    projected=evaluate("B",("P","Q"))
    run["stages"]["B"]["residuals"]={k:intervals(v) for k,v in projected.items()}
    run["stages"]["B"]["reductions"]={"U_minus_P":intervals(legacy-projected["P"]),
                                       "U_minus_Q":intervals(legacy-projected["Q"]),
                                       "Q_minus_P":intervals(projected["Q"]-projected["P"])}
    save_progress()
    validate_coverage("B",("P","Q"))
    qualified({**projected,"U_minus_P":legacy-projected["P"]})
    run["stages"]["B"]["passed"]=True
    limits=G["material_fraction_of_wl101e_residual"]*base_error
    p_ci=np.array([a["ci95"] for a in intervals(projected["P"]).values()])
    insufficient=bool(np.any((p_ci[:,0]>limits)|(p_ci[:,1]<-limits)))
    sufficient=bool(np.all((p_ci[:,0]>=-limits)&(p_ci[:,1]<=limits)))
    if sufficient:
        run["verdict"]="FIXED_LAYOUT_PROJECTION_SUFFICIENT"
        run["stages"]["C"]={"executed":False,"reason":"Gate B residuals within the material-equivalence band."}
    elif insufficient:
        run["stages"]["C"]={"executed":True,"passed":False}
        variants=evaluate("C",("S","A","SA"))
        p=projected["P"]
        effects={f"P_minus_{k}":p-v for k,v in variants.items()}
        effects["interaction_P_minus_S_minus_A_plus_SA"]=p-variants["S"]-variants["A"]+variants["SA"]
        effects["A_at_refined_space"]=variants["S"]-variants["SA"]
        effects["S_at_refined_direction"]=variants["A"]-variants["SA"]
        run["stages"]["C"]["residuals"]={k:intervals(v) for k,v in variants.items()}
        run["stages"]["C"]["effects"]={k:intervals(v) for k,v in effects.items()}
        save_progress()
        validate_coverage("C",tuple(variants))
        qualified({**variants,**effects})
        run["stages"]["C"]["passed"]=True
        resolved={}
        for name in ("P_minus_S","P_minus_A","P_minus_SA"):
            # Move the signed residual toward zero, including a possible P overshoot.
            effect=np.abs(p)-np.abs(variants[name.removeprefix("P_minus_")])
            lo=np.array([item["ci95"][0] for item in intervals(effect).values()])
            resolved[name]=(lo>=G["minimum_reduction_fraction_of_wl101e_residual"]*base_error).tolist()
        run["stages"]["C"]["material_reductions"]=resolved
        run["verdict"]="REFINEMENT_ISOLATED" if any(any(v) for v in resolved.values()) else "UNRESOLVED_RESIDUAL"
    else:
        run["verdict"]="INCONCLUSIVE_PRECISION"
        run["stages"]["C"]={"executed":False,"reason":"Fixed-layout equivalence and material residual both unresolved."}
    run["precision_limits"]=(G["precision_fraction_of_wl101e_residual"]*base_error).tolist()
    run["material_equivalence_band"]=limits.tolist()
    write_json(out/"run_manifest.json",run)
    print(run["verdict"],flush=True)


def main(root,out,commit,smoke):
    reader,learned,base,wl101e,prior,history,reference,run = inputs(root)
    out.mkdir(parents=True, exist_ok=False)
    run.update(commit=commit, protocol_sha256=sha(PROTOCOL),
               baseline_manifest_sha256=sha(base/"run_manifest.json"))
    started=time.perf_counter()
    print("Running exact split query and original six-event comparison",flush=True)
    run["smoke"] = reader.smoke(P["query_control_seed"],FirstSurface,G)
    run["smoke"]["seconds"] = time.perf_counter()-started
    print(run["smoke"],flush=True)
    write_json(out/"progress.json",run)
    if smoke:
        return
    primary(reader,learned,base,wl101e,prior,history,reference,out,run)


if __name__=="__main__":
    arg=argparse.ArgumentParser(description=__doc__)
    arg.add_argument("wl101",type=Path)
    arg.add_argument("output",type=Path)
    arg.add_argument("commit")
    arg.add_argument("--smoke",action="store_true")
    a=arg.parse_args()
    main(a.wl101.resolve(),a.output.resolve(),a.commit,a.smoke)
