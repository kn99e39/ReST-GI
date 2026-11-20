"""Evaluate frozen R3 and the 12 paired factorial fits on immutable paired frames."""
import argparse
import math
from pathlib import Path
import time

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

from common108 import ARMS, P, PROTOCOL, SEEDS, SPPS, VIEWS, assessment107, read, sha, write
from common107 import low_high, panel, rois, total
from model108 import Reconstructor, base
from train108 import append_aux, load_feature
from train105 import D_aux
from metrics108 import decide, summarize

ORACLE_ANCHOR = "b5e49c7296cb007077674de37e821231c4400cfa60b423bc0f80e1375de82c55"


def regions(aux, descriptor, old):
    original = rois(aux)
    masks = old.profile_masks(aux,original)
    rim = original["silhouette"]
    distance = distance_transform_edt(~rim).astype(np.int16)
    inside = (aux["cover"][6] >= 1) & original["primary"]
    masks.update({f"inside_{k}": inside & (distance == k) for k in (1,2,3)})
    masks["inside_beyond_3"] = inside & (distance > 3)
    cuts = np.quantile(descriptor[rim],(1/3,2/3))
    bins = np.searchsorted(cuts,descriptor,side="right")
    masks.update({f"grazing_{k}":rim & (bins==k) for k in range(3)})
    groups = ("silhouette","inside_1","inside_2","inside_3","inside_beyond_3")
    remainder = original["primary"] & ~np.logical_or.reduce([masks[k] for k in groups])
    partition = {**{k:masks[k] for k in groups}, "remaining_primary":remainder}
    np.testing.assert_array_equal(np.stack(list(partition.values())).sum(0),original["primary"])
    return masks,partition


def analyze(pred, parts, target, masks, partition, old, oracle=None):
    y, components = np.stack(pred).astype(np.float64), np.stack(parts)
    result = old.metric_rows(y,components,target,masks)
    draws = len(y)
    for k, row in result.items():
        row["seed_corrected_bias_power_RGB"] = row["squared_bias_RGB_ensemble"] - row["linear_RGB_variance"]/draws
    pixel_error = np.square(y-target).sum(1)
    mass = pixel_error[:,masks["primary"]].sum(1)
    error_share = {k:(pixel_error[:,mask].sum(1)/mass).tolist() for k,mask in partition.items()}
    np.testing.assert_allclose(sum(np.asarray(x) for x in error_share.values()),1,atol=1e-10)
    record = {"rois":result,"primary_RGB_SSE_fraction_draws":error_share}
    if oracle is not None:
        ix,label = oracle["panel_indices"],oracle["target"]
        reconstruction = y.reshape(draws,3,-1)[:,:,ix].transpose(0,2,1)
        record["WL107_panel_RGB_RMSE_draws"] = np.sqrt(np.mean((reconstruction-label)**2,axis=(1,2)) /
                                                        np.mean(label**2,dtype=np.float64)).tolist()
    return record


@torch.no_grad()
def timing(model, data, guides):
    for _ in range(10):
        model(data,guides)
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    samples = []
    for _ in range(50):
        start = time.perf_counter()
        model(data,guides)
        torch.cuda.synchronize()
        samples.append((time.perf_counter()-start)*1000)
    return {"median_ms":float(np.median(samples)), "p95_ms":float(np.quantile(samples,.95)),
            "peak_extra_MiB":(torch.cuda.max_memory_allocated()-baseline)/2**20,
            "params":base.n_params(model)}


def models(base_path, fit_path, descriptor_qual):
    checkpoint = base_path / "models/R3.pt"
    assert sha(checkpoint) == descriptor_qual["checkpoint_sha256"]
    sizes = {"R3":{"baseline": checkpoint}}
    for arm in ARMS:
        sizes[arm] = {}
        for seed in map(str,SEEDS):
            folder = fit_path / arm / seed
            path = folder / "weights.pt"
            fit = read(folder / "train.json")
            assert fit["complete"] and fit["steps"] == P["training"]["iterations"]
            assert fit["protocol_sha256"] == sha(PROTOCOL) and fit["weights_sha256"] == sha(path)
            assert fit["data_manifest_sha256"] == descriptor_qual["base_data_sha256"]
            assert fit["initial_sha256"] == read(fit_path/"A0"/seed/"train.json")["initial_sha256"]
            assert fit["hardware"] == torch.cuda.get_device_name(0) and fit["n_params"] == 1056038
            sizes[arm][seed] = path
    return sizes


def description_path(path, phase):
    return path / ("qualification.json" if phase=="original" else "descriptors.json")


@torch.no_grad()
def evaluate(base_path, prior, fit_path, descriptor_path, relative_output, oracle_anchor, commit, confirmation):
    phase = "confirmation" if confirmation else "original"
    relative_output.mkdir(parents=True, exist_ok=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    base.FUSED = True
    old = assessment107()
    qual_root = fit_path.parent / "descriptors"
    q = read(qual_root / "qualification.json")
    size = models(base_path,fit_path,q)
    test_names = tuple(P["confirmation"]["cameras"]) if confirmation else VIEWS
    reference_root = (confirmation or prior) / "reference"
    reference_manifest = reference_root / "reference107.json"
    reference = read(reference_manifest)
    if not confirmation:
        assert sha(reference_manifest) == P["frozen"]["wl107_reference"] and reference["preliminary_pass"]
        assert sha(oracle_anchor) == ORACLE_ANCHOR
        anchor = read(oracle_anchor)
    dm_path = description_path(descriptor_path,phase)
    descriptor_manifest = read(dm_path)
    dataset = (confirmation / "data") if confirmation else (base_path / "data/eval")
    result = {"phase":phase,"protocol_sha256":sha(PROTOCOL),"commit":commit,"complete":False,
              "descriptor_sha256":sha(dm_path),"reference_sha256":sha(reference_manifest),
              "models":{k:{s:sha(p) for s,p in v.items()} for k,v in size.items()},
              "timing_model_only_512x512_4spp":{}, "views":{},"roi_definition":"WL107 with WL108 diagnostics"}
    for view in test_names:
        folder = dataset / view
        target_file = reference_root / (view+"_target.npy")
        assert sha(target_file) == reference["views"][view]["target_sha256"]
        target = total(np.load(target_file)).astype(np.float64)
        aux = dict(np.load(folder/"aux.npz"))
        descriptor = load_feature(descriptor_path,descriptor_manifest,"eval",view)
        masks, partition = regions(aux, descriptor, old)
        if confirmation:
            oracle_file = None
        else:
            panel_file = prior/"real_amend1"/(view+"_panel.npz")
            assert sha(panel_file) == anchor["raw_files_sha256"][f"real_amend1/{view}_panel.npz"]
            oracle_file = dict(np.load(panel_file))
            idx = oracle_file["panel_indices"]
            np.testing.assert_array_equal(idx,panel(aux))
            np.testing.assert_array_equal(oracle_file["target"],target.reshape(3,-1)[:,idx].T)
        result["views"][view] = {}
        maps = {"target":target.astype(np.float32),"silhouette":masks["silhouette"],"descriptor":descriptor}
        for spp in SPPS:
            rows = {}
            frames = []
            for rep in range(P["evaluation"]["reps"]):
                with np.load(folder / f"noisy_spp{spp}_r{rep}.npz") as f:
                    frames.append(f["comp"])
            comp = [torch.from_numpy(a)[None].cuda() for a in frames]
            init_aux = D_aux(aux,spp)
            g0 = torch.from_numpy(append_aux(init_aux,np.zeros_like(descriptor)))[None].cuda()
            g1 = torch.from_numpy(append_aux(init_aux,descriptor))[None].cuda()
            original = g0[:,:14].contiguous()
            rows["R0"] = {"baseline":analyze([total(a) for a in frames],[low_high(a) for a in frames],
                                           target,masks,partition,old,oracle_file)}
            for arm, weights in size.items():
                rows[arm] = {}
                for seed, path in weights.items():
                    model = (base.Reconstructor("R3") if arm=="R3" else Reconstructor()).cuda()
                    model.load_state_dict(torch.load(path,map_location="cuda"))
                    model.eval()
                    guide = original if arm=="R3" else g1 if P["arms"][arm][0] else g0
                    timed = not confirmation and view==test_names[0] and spp==4
                    if timed:
                        result["timing_model_only_512x512_4spp"].setdefault(arm,{})[seed] = timing(model,comp[0],guide)
                    pred, parts = [], []
                    for t in comp:
                        out, piece = model(t,guide)
                        rgb = out.cpu().numpy()[0]
                        assert np.isfinite(rgb).all()
                        pred.append(rgb)
                        parts.append(np.stack([piece[k].cpu().numpy()[0] for k in ("low","high")]))
                    if arm=="R3" and not confirmation:
                        for rep, arr in enumerate(pred):
                            with np.load(base_path / "eval" / f"{view}_spp{spp}_r{rep}.npz") as frozen:
                                ref = frozen["R3"]
                            assert np.abs(arr-ref).sum()/np.abs(ref).sum() <= 1e-6, (view,spp,rep)
                    rows[arm][seed] = analyze(pred,parts,target,masks,partition,old,oracle_file)
                    if seed in ("baseline",str(SEEDS[0])):
                        maps[f"{arm}_{spp}"] = np.mean(pred,axis=0,dtype=np.float64).astype(np.float32)
                    del model
            if oracle_file is not None:
                key = f"O1_{spp}_1"
                pred_panel = oracle_file[key].astype(np.float64).sum(1)
                truth = oracle_file["target"].astype(np.float64)
                rows["O1"] = {"lambda":1,"RGB_RMSE_draws":np.sqrt(np.mean((pred_panel-truth)**2,axis=(1,2)) /
                                                                 np.mean(truth**2)).tolist()}
            result["views"][view][str(spp)] = rows
            print("Scored",view,spp,flush=True)
        np.savez_compressed(relative_output / (view +"_maps.npz"),**maps)
        write(relative_output / "metrics.json",result)
    result["pooled"] = summarize(result["views"],reference)
    result["decision"] = decide(result["pooled"],reference,len(test_names))
    if not confirmation:
        result["WL107_panel_mean_relative_RMSE"] = {
            str(s):{arm:float(np.mean([score for v in result["views"].values()
                                       for r in v[str(s)][arm].values() for score in r["WL107_panel_RGB_RMSE_draws"]]))
                    for arm in ("R0","R3",*ARMS)}
            for s in SPPS
        }
        for spp in SPPS:
            result["WL107_panel_mean_relative_RMSE"][str(spp)]["O1"] = float(np.mean([
                np.mean(v[str(spp)]["O1"]["RGB_RMSE_draws"]) for v in result["views"].values()]))
    result["complete"] = True
    write(relative_output / "metrics.json",result)
    print("Decision", result["decision"], flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ("baseline","prior","fits","descriptors","output","oracle_anchor"):
        p.add_argument(arg,type=Path)
    p.add_argument("commit")
    p.add_argument("--confirmation",type=Path)
    a=p.parse_args()
    evaluate(a.baseline,a.prior,a.fits,a.descriptors,a.output,a.oracle_anchor,a.commit,a.confirmation)
