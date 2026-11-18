"""Run recovered E2–E5 Oracle and paired unmodified ECRO camera readout."""
import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

from oracle_field import ExposedOracle, suffix_controls

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"ecro_sss101d"))
from run_oracle import (A, array_hash, fingerprint, foreign_gpu, gpu_query, readout_preflight,
                        rel_l1, sha, unpack, validate_inputs, wait_for_gpu, write_json)

PROTOCOL=HERE/"protocol/recovery.json"
P=json.loads(PROTOCOL.read_text())


def main(root,out,commit,preflight):
    reference=root/A["references"]["path_in_wl101"]
    sources=validate_inputs(root,reference)
    wait_for_gpu()
    assert [p.strip() for p in gpu_query("gpu","name,driver_version").split(",")]==[
        "NVIDIA GeForce RTX 3080 Ti","595.71.05"]
    sys.path.insert(0,str(root/"experiments/ecro_dyn101"))
    os.environ["FREEZE"]="1"
    import rt101 as R
    import seq101 as seq
    assert (R.mi.__version__,R.dr.__version__,R.torch.__version__)==("3.8.0","1.3.1","2.3.1+cu121")
    case,start,events=seq.workload(A["workload"])
    cam=dict(R.C100.P["datasets"][case]["camera"],res=512)
    assert cam==A["camera"] and events[seq.WARM]==(A["geometry"],A["light"],0)
    rt=R.RT(start[0],case,R.load_model(case),start[1])
    for geo,light,_ in events[:seq.WARM+1]:
        moved,old=rt.set_geom(geo)
        rt.update_state(old,moved)
        rt.set_light(light)
        learned,_=rt.respond(rt.source())
    learned=learned.cpu().numpy().copy()
    assert array_hash(learned)=="d7e40954304bfc2b03f679af0ef6a39876f21ffc21d3a07fe3dfc3de7bd18543"
    optix=next(Path(line.split()[-1]) for line in Path("/proc/self/maps").read_text().splitlines() if "/libnvoptix.so." in line)
    assert optix.name=="libnvoptix.so.595.71.05"
    out.mkdir(parents=True,exist_ok=False)
    history=root.parent/"results/ecro_sss101a/qualified63"
    prior=json.loads((history/"run_manifest.json").read_text())
    assert prior["response_sha256"]==array_hash(learned)
    assert prior["active_optix_sha256"]==sha(optix)
    scene=fingerprint({k:A[k] for k in ("geometry","camera","sphere","light")})
    assert scene==prior["scene_sha256"]

    run={"commit":commit,"protocol_sha256":sha(PROTOCOL),"scene_sha256":scene,
         "checkpoint_sha256":A["checkpoint_sha256"],"cfg_sha256":A["configs_sha256"],
         "learned_table_sha256":array_hash(learned),"prior_manifest_sha256":sha(history/"run_manifest.json"),
         "input_sha256":A["references"]["sha256"],"renderer_sources_sha256":sources,
         "environment":{"gpu":R.torch.cuda.get_device_name(),"driver":"595.71.05","optix_sha256":sha(optix),
                        "mitsuba":R.mi.__version__,"drjit":R.dr.__version__,"torch":R.torch.__version__,
                        "python":platform.python_version(),"numpy":np.__version__,"os":platform.platform()},
         "oracle_tables":[],"readouts":[]}

    def save(name, **data):
        path=out/name
        np.savez_compressed(path,**data)
        return {"file":name,"sha256":sha(path),"bytes":path.stat().st_size}

    def timing(fn):
        wait_for_gpu()
        R.sync()
        stamp=time.perf_counter()
        answer=fn()
        R.sync()
        return answer,time.perf_counter()-stamp

    run["B_theta"]=save("B_theta.npz",B=learned)
    oracle,build_s=timing(lambda: ExposedOracle(R,rt))
    print(f"Attached native shape views in {build_s:.2f}s",flush=True)
    address=readout_preflight(rt,R,cam)
    seed=P["qualification_seed"]
    print("Checking matched suffix ownership",flush=True)
    suffix,suffix_s=timing(lambda: suffix_controls(oracle,seed))
    print(f"Suffix control {suffix} in {suffix_s:.2f}s",flush=True)
    test,qual_s=timing(lambda: oracle.accumulate(P["qualification_spp"],seed,16))
    run["qualification"]={"address":address,"suffix":suffix,"build_seconds":build_s,"suffix_seconds":suffix_s,
                          "qualification_seconds":qual_s,"floor":oracle.floor_check(test),
                          **save("qualification.npz",**test)}
    write_json(out/"progress.json",run)
    print("Qualification",run["qualification"],flush=True)
    if preflight:
        return

    seeds=P["seeds"]
    assert len(set(seeds))==4
    assert set(seeds).isdisjoint({seed,*P["camera_seeds"]})
    fields,counts=[],[]
    for i,seed in enumerate(seeds):
        def mark(_,total):
            if total%256==0:
                print(f"Oracle {i}: {total}/{P['spp_per_slot_replica']} spp",flush=True)
            if foreign_gpu():
                wait_for_gpu()
        packet,seconds=timing(lambda:oracle.accumulate(P["spp_per_slot_replica"],seed,P["batch_spp"],mark))
        fields.append(packet["B"])
        counts.append(packet["valid"])
        row={**save(f"oracle_{i}.npz",**packet),"seed":seed,"seconds":seconds,
             "spp":P["spp_per_slot_replica"],"array_sha256":array_hash(packet["B"]),
             "floor":oracle.floor_check(packet),
             "counts":{k:int(packet[k].sum()) for k in
                       ("proposed","valid","buried","back","tie","invalid","negative_ring_zero")},
             "unsupported_slots":int((packet["valid"]==0).sum())}
        run["oracle_tables"].append(row)
        write_json(out/"progress.json",run)
        print("Oracle complete",i,row,flush=True)

    run["halves"]=[save(f"half_{j}.npz",B=np.mean(fields[j*2:j*2+2],axis=0,dtype=np.float64).astype(np.float32),
                       counts=np.stack(counts[j*2:j*2+2])) for j in range(2)]
    empty=np.any(np.stack(counts)==0,axis=0)
    renderer=lambda table,s:unpack(R.render_hq(rt,R.PT.fl(table),cam,512,P["camera_spp"],s,b=P["camera_launch_spp"]),
                                   P["camera_spp"])
    run["support_camera"]=[]
    assert P["camera_seeds"]==[row["seed"] for row in prior["matched"]]
    if empty.any():
        flag=np.repeat(empty[...,None],3,axis=2).astype(np.float32)
        for s in P["camera_seeds"]:
            raw,seconds=timing(lambda:renderer(flag,s))
            marker=raw[...,2]
            row={**save(f"unsupported_marker_{s}.npz",weight=marker),"seed":s,"seconds":seconds,
                 "empty_slots":int(empty.sum()),"max_weight":float(marker.max()),
                 "pixels":int(np.count_nonzero(marker.max(-1)))}
            run["support_camera"].append(row)
            print("Support",row,flush=True)
        if any(r["pixels"]>0 for r in run["support_camera"]):
            run["stop"]="Camera uses physically empty Oracle slots."
            write_json(out/"run_manifest.json",run)
            return
    for i,table in enumerate(fields):
        for j,s in enumerate(P["camera_seeds"]):
            raw,seconds=timing(lambda:renderer(table,s))
            e1,e2,hl=np.moveaxis(raw,-1,0)
            row=prior["matched"][j]
            prior_path=history/row["path"]
            assert row["seed"]==s and sha(prior_path)==row["sha256"]
            with np.load(prior_path) as arch,np.load(reference/f"REF_0_{j}.npz") as ref:
                err={k:rel_l1(v,arch[k]) for k,v in (("e1",e1),("e2",e2))}
                physical={k:rel_l1(v,ref[k]) for k,v in (("e1",e1),("e2",e2))}
            assert max(err.values())<=1e-7 and max(physical.values())<=1e-4
            packet=save(f"readout_{i}_{j}.npz",e1=e1,e2=e2,hl=hl,low=e1+e2,full=e1+e2+hl)
            run["readouts"].append({**packet,"oracle_index":i,"camera_half":j,"camera_seed":s,
                                    "seconds":seconds,"spp":P["camera_spp"],
                                    "learned_low_rel_l1":err,"physical_low_rel_l1":physical})
            write_json(out/"progress.json",run)
            print("Readout complete",i,j,seconds,flush=True)
    write_json(out/"run_manifest.json",run)
    print("WL101E final evidence complete",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("wl101",type=Path)
    p.add_argument("output",type=Path)
    p.add_argument("commit")
    p.add_argument("--preflight",action="store_true")
    a=p.parse_args()
    main(a.wl101.resolve(),a.output.resolve(),a.commit,a.preflight)
