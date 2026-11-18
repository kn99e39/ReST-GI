"""WL101D physical target capture and unchanged matched WL101 readout."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/"ecro_sss101a"))
from common import P as A, sha, fingerprint, rel_l1, write_json
from run import validate_inputs, wait_for_gpu, foreign_gpu, gpu_query

PROTOCOL = HERE/"protocol/oracle.json"
P = json.loads(PROTOCOL.read_text())


def array_hash(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def unpack(acc, spp, res=512):
    return (acc.reshape(3,3,res,res).transpose(2,3,1,0)/spp).astype(np.float32)


def readout_preflight(rt, R, cam):
    n = rt.N*64
    base = np.arange(n,dtype=np.float32).reshape(rt.N,64)
    table = np.stack((1+base/n,1+(base%64)/64,1+(base//64)/rt.N),axis=-1)
    counts = []
    old = rt.patch_bin

    def captured(*args):
        result = old(*args)
        counts.append(result)
        return result

    def frame(B):
        rt.patch_bin = captured
        counts.clear()
        try:
            out = rt._frame_prog(rt.scene,rt.tabs(),rt.light_args(),R.PT.fl(B),
                                 R.dr.opaque(R.mi.UInt64, 2110041), (cam,64,1))
            assert len(counts)==2
            pid, bn = counts[1]
            R.dr.eval(out,pid,bn)
            return np.stack([np.asarray(v) for v in out]), np.asarray(pid),np.asarray(bn)
        finally:
            rt.patch_bin=old

    zero, _, _ = frame(np.zeros_like(table))
    ones, _, _ = frame(np.ones_like(table))
    encoded, pid, bn = frame(table)
    summed, _, _ = frame(table+1)
    expected = ones[6:]*table[pid,bn].T
    assert np.count_nonzero(ones[6:])>5000
    assert np.all(zero[6:]==0)
    assert all(np.array_equal(zero[:6],image[:6]) for image in (ones,encoded,summed))
    assert np.max(np.abs(encoded[6:]-expected))<=3e-7
    assert np.max(np.abs(summed[6:]-encoded[6:]-ones[6:]))<=4e-7
    return {"valid_lookup_values":int(np.count_nonzero(ones[6:])),
            "addressing_max_absolute":float(np.max(np.abs(encoded[6:]-expected))),
            "linearity_max_absolute":float(np.max(np.abs(summed[6:]-encoded[6:]-ones[6:]))),
            "zero_and_low_invariance":True}


def field(rt,R,spp,seed,batch,audit=False):
    """Call the original estimator and retain its actual valid denominators."""
    old_accum, old_trace, old_lookup = rt.accum, rt.trace, rt.patch_bin
    captured_counts, mismatches = [], []
    observed = []
    max_event_error = 0.

    def accumulate(outputs,index,n):
        sums = old_accum(outputs,index,n)
        captured_counts.append(sums["count"])
        if audit:
            expected = R.dr.arange(R.U,R.dr.width(index))%(rt.N*64)
            pid, bn = observed[0]
            mismatch = outputs["count"]>0
            mismatch &= (pid*64+bn)!=expected
            values = old_accum({"mismatch":R.dr.select(mismatch,R.F(1),R.F(0))},index,n)
            mismatches.append(values["mismatch"])
        return sums

    def observe(*args):
        value = old_lookup(*args)
        observed.append(value)
        return value

    def audited_trace(ray,N,rng,sd,ref,depth,lights,mode,**kwargs):
        nonlocal max_event_error
        assert mode=="probe" and depth==5
        observed.clear()
        # Copy the same post-sampling RNG state for a five-event camera-mode suffix.
        duplicate = R.mi.PCG32(size=N)
        duplicate.state = R.mi.UInt64(rng.state)
        duplicate.inc = R.mi.UInt64(rng.inc)
        original, ok = old_trace(ray,N,rng,sd,ref,depth,lights,mode,**kwargs)
        probe_route = list(observed)
        camera, _ = old_trace(ray,N,duplicate,sd,ref,depth,lights,"cam",probe_ok=kwargs["probe_ok"])
        observed[:] = probe_route
        for j in range(3):
            for a,c in (("f2","e2"),("fh","h")):
                delta = R.dr.select(ok,R.dr.abs(original[a+str(j)]-camera[c+str(j)]),0.)
                max_event_error = max(max_event_error,float(R.dr.max(delta)[0]))
        return original,ok

    rt.accum=accumulate
    if audit:
        rt.trace=audited_trace
        rt.patch_bin=observe
    try:
        numeric = rt.field_components(rt.lt,spp,seed,b=batch)
    finally:
        rt.accum,rt.trace,rt.patch_bin = old_accum,old_trace,old_lookup
    count = np.sum(captured_counts,axis=0).reshape(rt.N,64)
    f2, fh = numeric
    assert f2.shape==fh.shape==(1270,64,3)
    assert np.isfinite(f2).all() and np.isfinite(fh).all() and min(f2.min(),fh.min())>=0
    assert np.all((count>=0)&(count<=spp))
    result = {"f2":f2,"fh":fh,"B":f2+fh,"valid_count":count.astype(np.uint32)}
    if audit:
        result["misrouted_count"] = np.sum(mismatches,axis=0).reshape(rt.N,64).astype(np.uint32)
        assert max_event_error<=1e-7, max_event_error
    return result,max_event_error


def support_camera(rt,R,cam,suspect,out):
    table=R.PT.fl(np.repeat(suspect[...,None],3,axis=2).astype(np.float32))
    rows=[]
    for seed in A["references"]["seed_by_half"]:
        wait_for_gpu()
        R.sync()
        started=time.perf_counter()
        values=unpack(R.render_hq(rt,table,cam,512,4096,seed),4096)[...,2]
        seconds=time.perf_counter()-started
        path=out/f"support_readout_{seed}.npz"
        np.savez_compressed(path,weight=values)
        row={"seed":seed,"file":path.name,"sha256":sha(path),
             "nonzero_pixels":int(np.count_nonzero(values.max(axis=-1))),
             "max_weight":float(values.max()),"mean_weight":float(values.mean()),
             "seconds":seconds,"foreign_gpu_pids_after":foreign_gpu()}
        rows.append(row)
        print("SUPPORT",row,flush=True)
    return rows


def classify_first_hit(rt,R,spp,seed):
    """Replay only the original probe's four RNG draws and the first hit."""
    ns=rt.N*64
    count=ns*spp
    rng=R.mi.PCG32(size=count,initstate=R.dr.opaque(R.mi.UInt64,seed*1000003),initseq=R.mi.UInt64(0))
    slot=R.dr.arange(R.U,count)%ns
    requested_pid,requested_bin=slot//64,slot%64
    p,nn,(tx,ty,tz)=rt.patch_point(requested_pid,rng.next_float32(),rng.next_float32())
    local=rt.dir_in_bin(requested_bin,rng.next_float32(),rng.next_float32())
    world=tx*local.x+ty*local.y+tz*local.z
    front=R.dr.dot(world,nn)>1e-3
    ray=R.mi.Ray3f(p+world*2e-3,-world)
    hit=rt.scene.ray_intersect(ray,front)
    actual_sid=rt.shape_id(hit)
    actual_pid,actual_bin=rt.patch_bin(hit,actual_sid,-ray.d)
    accepted=front & hit.is_valid() & (R.dr.norm(hit.p-p)<1e-3)
    back=hit.to_local(-ray.d).z<=1e-6
    values=[slot,actual_sid,actual_pid,actual_bin,accepted,back,R.dr.norm(hit.p-p)]
    R.dr.eval(values)
    R.sync()
    idx,arr_sid,arr_pid,arr_bin,valid,back,dist=[np.asarray(item) for item in values]
    requested_pid,requested_bin=idx//64,idx%64
    requested_sid=rt.inc.G.PSID[requested_pid]
    wrong_address=valid & ((requested_pid!=arr_pid)|(requested_bin!=arr_bin))
    groups={
        "different_shape":wrong_address&(requested_sid!=arr_sid),
        "same_shape_different_patch":wrong_address&(requested_sid==arr_sid)&(requested_pid!=arr_pid),
        "same_patch_different_bin":wrong_address&(requested_pid==arr_pid)&(requested_bin!=arr_bin)
    }
    assert np.array_equal(wrong_address,np.logical_or.reduce(list(groups.values())))
    accumulate=lambda mask:np.bincount(idx[mask].astype(np.int64),minlength=ns).reshape(rt.N,64)
    matrix=np.zeros((9,9),np.int64)
    cross=groups["different_shape"]
    np.add.at(matrix,(requested_sid[cross],arr_sid[cross]),1)
    return {"valid":accumulate(valid),"wrong":accumulate(wrong_address),
            "backface_accepted":accumulate(valid&back),"cross_shape_matrix":matrix,
            "max_distance_by_kind_m":np.array([np.max(dist[m],initial=0.) for m in groups.values()]),
            "sample_count_by_kind":np.array([m.sum() for m in groups.values()]),
            **{k:accumulate(mask) for k,mask in groups.items()}}


def main(root,out,commit,development,audit_support,inspect_membership):
    archive = root/A["references"]["path_in_wl101"]
    sources = validate_inputs(root,archive)
    wait_for_gpu()
    assert [p.strip() for p in gpu_query("gpu","name,driver_version").split(",")]==[
        "NVIDIA GeForce RTX 3080 Ti","595.71.05"]
    sys.path.insert(0,str(root/"experiments/ecro_dyn101"))
    os.environ["FREEZE"]="1"
    import rt101 as R
    import seq101 as seq
    assert (R.mi.__version__, R.dr.__version__, R.torch.__version__)==("3.8.0","1.3.1","2.3.1+cu121")
    case,start,events=seq.workload(A["workload"])
    cam=dict(R.C100.P["datasets"][case]["camera"],res=512)
    assert cam==A["camera"] and events[seq.WARM]==(A["geometry"],A["light"],0)
    rt=R.RT(start[0],case,R.load_model(case),start[1])
    for geom,light,_ in events[:seq.WARM+1]:
        moved,old=rt.set_geom(geom)
        rt.update_state(old,moved)
        rt.set_light(light)
        learned,_=rt.respond(rt.source())
    learned=learned.cpu().numpy().copy()
    with np.load(archive/"PRED_B_0.npz") as f:
        assert np.array_equal(f["B"],learned)
    assert array_hash(learned)=="d7e40954304bfc2b03f679af0ef6a39876f21ffc21d3a07fe3dfc3de7bd18543"
    paths=Path("/proc/self/maps").read_text().splitlines()
    optix=next(Path(line.split()[-1]) for line in paths if "/libnvoptix.so." in line)
    assert optix.name=="libnvoptix.so.595.71.05"

    out.mkdir(parents=True,exist_ok=False)
    run={"commit":commit,"protocol_sha256":sha(PROTOCOL),"source_sha256":sources,
         "scene_sha256":fingerprint({k:A[k] for k in ("geometry","camera","sphere","light")}),
         "checkpoint_sha256":A["checkpoint_sha256"],"cfg_sha256":A["configs_sha256"],
         "learned_table_sha256":array_hash(learned),"archived_inputs_sha256":A["references"]["sha256"],
         "environment":{"GPU":R.torch.cuda.get_device_name(),"driver":"595.71.05","optix_sha256":sha(optix),
                        "mitsuba":R.mi.__version__,"drjit":R.dr.__version__,"torch":R.torch.__version__,
                        "python":platform.python_version(),"numpy":np.__version__,"os":platform.platform()},
         "oracle_tables":[],"readouts":[]}

    def save(name, **data):
        p=out/name
        np.savez_compressed(p,**data)
        return {"file":name,"sha256":sha(p),"bytes":p.stat().st_size}

    run["learned"]=save("B_theta.npz",B=learned)
    if inspect_membership is not None:
        previous=json.loads((inspect_membership/"progress.json").read_text())
        assert previous["scene_sha256"]==run["scene_sha256"]
        info=previous["preflight"]
        original_probe=inspect_membership/info["probe_packet"]["file"]
        assert sha(original_probe)==info["probe_packet"]["sha256"]
        stamp=time.perf_counter()
        routing=classify_first_hit(rt,R,info["audit_spp"],info["audit_seed"])
        with np.load(original_probe) as saved:
            assert np.array_equal(routing["valid"],saved["valid_count"])
            assert np.array_equal(routing["wrong"],saved["misrouted_count"])
        run["membership_replay"]={**save("first_hit_membership.npz",**routing),
            "seed":info["audit_seed"],"spp":info["audit_spp"],
            "seconds":time.perf_counter()-stamp,
            "source_probe_sha256":info["probe_packet"]["sha256"],
            "categories":["different_shape","same_shape_different_patch","same_patch_different_bin"],
            "category_counts":routing["sample_count_by_kind"].tolist(),
            "cross_shape_matrix":routing["cross_shape_matrix"].tolist(),
            "backface_accepted":int(routing["backface_accepted"].sum())}
        write_json(out/"run_manifest.json",run)
        print("MEMBERSHIP",run["membership_replay"],flush=True)
        return
    if audit_support is not None:
        preflight=json.loads((audit_support/"progress.json").read_text())
        assert preflight["scene_sha256"]==run["scene_sha256"]
        assert preflight["learned_table_sha256"]==run["learned_table_sha256"]
        packet=preflight["preflight"]["probe_packet"]
        assert sha(audit_support/packet["file"])==packet["sha256"]
        with np.load(audit_support/packet["file"]) as archive_probe:
            count=archive_probe["valid_count"]
            misrouted=archive_probe["misrouted_count"]
        suspect=misrouted>0
        g=rt.inc.G
        run["support_input"]=packet
        run["support_slot_info"]=save("slot_metadata.npz",patch_shape=g.PSID,patch_material=g.PMAT,
            patch_center=g.PC,patch_normal=g.PN,valid_count=count,misrouted_count=misrouted)
        run["support_camera_test"]=support_camera(rt,R,cam,suspect,out) if suspect.any() else []
        if any(row["nonzero_pixels"]>0 for row in run["support_camera_test"]):
            run["stop"]="Camera reads a geometrically misrouted WL100 supervision slot."
        write_json(out/"run_manifest.json",run)
        return
    synthetic=readout_preflight(rt,R,cam)
    developmental_seed=10190901
    probe, event_error=field(rt,R,32,developmental_seed,32,audit=True)
    run["preflight"]={**synthetic,"audit_spp":32,"audit_seed":developmental_seed,
                      "matched_probe_event_max_absolute":event_error,
                      "slots_with_zero_count":int((probe["valid_count"]==0).sum()),
                      "slots_with_misrouting":int((probe["misrouted_count"]>0).sum()),
                      "misrouted_samples":int(probe["misrouted_count"].sum()),
                      "probe_packet":save("probe_preflight.npz",**probe)}
    write_json(out/"progress.json",run)
    print("PREFLIGHT",run["preflight"],flush=True)
    if development:
        return
    audited_bad=probe["misrouted_count"]>0
    if audited_bad.any():
        run["support_camera_test"]=support_camera(rt,R,cam,audited_bad,out)
        if any(row["nonzero_pixels"]>0 for row in run["support_camera_test"]):
            run["stop"]="Camera reads a geometrically misrouted WL100 supervision slot."
            write_json(out/"run_manifest.json",run)
            print(run["stop"],flush=True)
            return
    arrays=[]
    assert sorted(set(P["oracle_seeds"]))==P["oracle_seeds"]
    assert set(P["oracle_seeds"]).isdisjoint(A["references"]["seed_by_half"])
    unsupported=np.zeros_like(audited_bad)
    for replica,seed in enumerate(P["oracle_seeds"]):
        wait_for_gpu()
        now=time.perf_counter()
        packet,_=field(rt,R,P["oracle_spp_per_replica"],seed,P["oracle_launch_spp"])
        elapsed=time.perf_counter()-now
        arrays.append(packet["B"])
        unsupported |= packet["valid_count"]==0
        run["oracle_tables"].append({**save(f"oracle_{replica}.npz",**packet),"seed":seed,
            "spp_per_slot":P["oracle_spp_per_replica"],"launch_spp":P["oracle_launch_spp"],
            "seconds":elapsed,"foreign_gpu_pids_after":foreign_gpu(),"table_sha256":array_hash(packet["B"]),
            "slots_with_zero_count":int((packet["valid_count"]==0).sum()),
            "minimum_count":int(packet["valid_count"].min())})
        write_json(out/"progress.json",run)
        print(f"Oracle {replica} done in {elapsed:.2f}s",flush=True)
    run["oracle_halves"]=[save(f"oracle_half_{i}.npz",B=np.mean(arrays[2*i:2*i+2],axis=0,dtype=np.float64).astype(np.float32))
                         for i in range(2)]

    seeds=A["references"]["seed_by_half"]
    run["support_camera_test"]=[]
    if unsupported.any():
        run["support_camera_test"]=support_camera(rt,R,cam,unsupported,out)
        if any(row["nonzero_pixels"]>0 for row in run["support_camera_test"]):
            run["stop"]="Camera reads an unsupported Oracle slot."
            write_json(out/"run_manifest.json",run)
            print(run["stop"],flush=True)
            return
    # Frozen learned and physical controls are read here only for LOW/readout validation.
    history=root.parent/"results/ecro_sss101a/qualified63"
    prior=json.loads((history/"run_manifest.json").read_text())
    assert prior["response_sha256"]==array_hash(learned)
    for r,table in enumerate(arrays):
        device=R.PT.fl(table)
        for j,seed in enumerate(seeds):
            wait_for_gpu()
            R.sync()
            start_time=time.perf_counter()
            e1,e2,hl=np.moveaxis(unpack(R.render_hq(rt,device,cam,512,4096,seed),4096),-1,0)
            R.sync()
            elapsed=time.perf_counter()-start_time
            match=prior["matched"][j]
            historical=history/match["path"]
            assert match["seed"]==seed and sha(historical)==match["sha256"]
            with np.load(historical) as control, np.load(archive/f"REF_0_{j}.npz") as ref:
                low_learned={k:rel_l1(v,control[k]) for k,v in (("e1",e1),("e2",e2))}
                low_physical={k:rel_l1(v,ref[k]) for k,v in (("e1",e1),("e2",e2))}
            assert max(low_learned.values())<=1e-7 and max(low_physical.values())<=1e-4
            run["readouts"].append({**save(f"readout_{r}_{j}.npz",e1=e1,e2=e2,hl=hl,
                low=e1+e2,full=e1+e2+hl),"oracle_index":r,"camera_half":j,"camera_seed":seed,
                "spp":4096,"seconds":elapsed,"learned_low_rel_l1":low_learned,"physical_low_rel_l1":low_physical,
                "foreign_gpu_pids_after":foreign_gpu()})
            write_json(out/"progress.json",run)
            print(f"Readout {r}/{j} done in {elapsed:.2f}s",flush=True)
    write_json(out/"run_manifest.json",run)
    print("WL101D complete",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wl101",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("commit")
    parser.add_argument("--preflight",action="store_true")
    parser.add_argument("--audit-support",type=Path)
    parser.add_argument("--inspect-membership",type=Path)
    a=parser.parse_args()
    main(a.wl101.resolve(),a.output.resolve(),a.commit,a.preflight,a.audit_support,a.inspect_membership)
