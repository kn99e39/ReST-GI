"""Render the prespecified confirmation cameras only after a diagnostic pass."""
import argparse
import math
from pathlib import Path
import time

import numpy as np

from common108 import P, PROTOCOL, SPPS, VIEWS, read, sha, write
from common107 import coverage_masks, rois
from qualify108 import extract
from renderer108 import frozen_runtime
from reference107 import assemble, DECISION_ROIS


def forward(camera):
    d = np.asarray(camera["target"]) - camera["origin"]
    return d / np.linalg.norm(d)


def check_cameras(cameras, existing):
    cos_limit = math.cos(math.radians(15))
    for camera in cameras.values():
        assert all(np.linalg.norm(np.asarray(camera["origin"])-other["origin"]) >= .75 or
                   forward(camera) @ forward(other) <= cos_limit for other in existing)
        existing.append(camera)


def confirm(base, prior, wl101, first, output):
    initial = read(first)
    assert initial["complete"] and initial["protocol_sha256"] == sha(PROTOCOL)
    initial_decision = initial["decision"]
    assert initial_decision["reference_qualified"] and (
        initial_decision["original_input_pass"] or initial_decision["descriptor_incremental_pass"])
    output.mkdir(parents=True, exist_ok=False)
    manifest = read(base / "data/run_manifest.json")
    config = P["confirmation"]
    cameras = config["cameras"]
    check_cameras(cameras, [v["camera"] for v in manifest["views"].values()])
    import render105 as r
    R, rt, worker, scene = frozen_runtime(r, wl101, radiance=True)
    seen = {f["seed"] for f in read(prior/"capture/capture_manifest.json")["files"].values()}
    seen |= {s for values in r.seed_sets(40,8,len(VIEWS)).values() for s in values}
    seen |= {r.P["seeds"]["determinism"], *r.P["seeds"]["replay"]}
    descriptor = {"files": {}, "protocol_sha256": sha(PROTOCOL)}
    reference = {"views": {}, "geometry": scene, "inputs": {}, "protocol_sha256": sha(PROTOCOL)}
    regions = {}
    raw_seeds = set()
    for i, (name, camera) in enumerate(cameras.items()):
        parent = output/"data"/name
        parent.mkdir(parents=True)
        guides = {**r.aux_buffers(R,rt,camera), **r.centre_gbuffer(R,rt,camera)}
        mask = rois(guides)
        assert all(mask[k].any() for k in DECISION_ROIS) and mask["silhouette"].sum() >= 20, name
        geom, counts, err, seconds = extract(r,R,rt,camera)
        np.testing.assert_array_equal(counts,64*guides["cover"][6])
        assert err <= .002
        np.savez_compressed(parent/"aux.npz",**guides)
        desc_file = output/"descriptors/eval"/(name+".npy")
        desc_file.parent.mkdir(parents=True,exist_ok=True)
        np.save(desc_file,geom,allow_pickle=False)
        descriptor["files"][f"eval/{name}"] = {"sha256":sha(desc_file),"seconds":seconds,"analytic_error":err}
        regions[name] = (mask, guides["cover"][6])
        add = i * config["seed_stride_per_view"]
        raw_seeds |= {config["assessment_seed_base"]+add+h for h in range(48)}
        raw_seeds |= {config["noisy_seed_base"]+add+100*j+k for j in range(len(SPPS)) for k in range(8)}
    assert len(raw_seeds) == len(cameras)*(48+3*8) and not (raw_seeds & seen)
    # seed*1000003+launch is injective for unique seeds and <=256 launches.
    assert config["block_spp"]//32 < 1000003
    write(output/"descriptors/descriptors.json",descriptor)

    ref_root = output/"reference"
    for i, (name, camera) in enumerate(cameras.items()):
        data_view = output/"data"/name
        blocks = ref_root/name
        blocks.mkdir(parents=True)
        add = i*config["seed_stride_per_view"]
        files = {}
        for h in range(config["reference_blocks"]):
            seed = config["assessment_seed_base"]+add+h
            block = r.frame_components(R,rt,worker,camera,config["block_spp"],seed)
            assert block.shape==(9,512,512) and np.isfinite(block).all() and block.min()>=0
            if h<2:
                filename = data_view/f"target_half{h}.npz"
                np.savez_compressed(filename,comp=block)
            else:
                filename = blocks/f"assessment_{h-2:02d}.npy"
                np.save(filename,block,allow_pickle=False)
            files[str(filename.relative_to(output))] = {"sha256":sha(filename),"seed":seed}
            print("Confirmation reference",name,h,flush=True)
        for j,spp in enumerate(SPPS):
            for rep in range(8):
                seed = config["noisy_seed_base"]+add+100*j+rep
                comp = r.frame_components(R,rt,worker,camera,spp,seed)
                assert np.isfinite(comp).all() and comp.min()>=0
                f = data_view/f"noisy_spp{spp}_r{rep}.npz"
                np.savez_compressed(f,comp=comp,seed=seed)
                files[str(f.relative_to(output))] = {"sha256":sha(f),"seed":seed}
        mask, cover = regions[name]
        signed = {k: mask[k] for k in DECISION_ROIS}
        signed.update({f"coverage_{i}":m for i,m in enumerate(coverage_masks(cover,mask["silhouette"])) if m.sum()>=20})
        target, row = assemble(data_view,blocks,mask,signed)
        dest = ref_root/(name+"_target.npy")
        np.save(dest,target,allow_pickle=False)
        reference["views"][name] = {**row,"target_sha256":sha(dest)}
        reference["inputs"][name] = files
        write(ref_root/"progress.json",reference)
    reference["pooled_reference_rgb_rms_se"] = {
        k:float(np.mean([v["relative_rgb_rms_se"][k] for v in reference["views"].values() if k in v["relative_rgb_rms_se"]]))
        for k in DECISION_ROIS}
    reference["max_view_stratum_signed_relative_se"] = max(x for v in reference["views"].values()
                                                            for x in v["relative_signed_luma_se"].values())
    write(ref_root/"reference107.json",reference)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ("baseline","prior","wl101","original_metrics","output"):
        p.add_argument(arg,type=Path)
    a=p.parse_args()
    confirm(a.baseline,a.prior,a.wl101,a.original_metrics,a.output)
