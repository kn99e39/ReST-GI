"""Full matched WL105 crop-order training of one arm and seed."""
import argparse
import hashlib
import math
from pathlib import Path
import platform
import time

import numpy as np
import torch

from common108 import ARMS, P, PROTOCOL, SEEDS, SPPS, read, sha, write
import data105 as data_api
from train105 import D_aux
from model108 import Reconstructor, base, loss


def feature_path(descriptors, split, name):
    return descriptors / split / (name + ".npy")


def load_feature(descriptors, desc_manifest, split, name):
    path = feature_path(descriptors, split, name)
    assert sha(path) == desc_manifest["files"][f"{split}/{name}"]["sha256"], path
    return np.load(path, allow_pickle=False)


def append_aux(aux, field):
    return np.concatenate((aux, field[None]), axis=0).astype(np.float32, copy=False)


@torch.no_grad()
def validation(model, records):
    model.eval()
    errors = {}
    for spp in SPPS:
        ms, magnitude = [], []
        for aux, target, frames in records:
            input_field = torch.from_numpy(frames[spp][0])[None].cuda()
            guide = torch.from_numpy(aux[spp])[None].cuda()
            pred, _ = model(input_field, guide)
            _, _, total = base.features(torch.from_numpy(target)[None].cuda(), guide)
            ms.append(float((pred-total).square().mean()))
            magnitude.append(float(total.square().mean()))
        errors[str(spp)] = math.sqrt(sum(ms) / sum(magnitude))
    model.train()
    return errors


def train(data_root, descriptors, output, arm, seed, commit):
    out = output / arm / str(seed)
    out.mkdir(parents=True, exist_ok=False)
    manifest = read(data_root / "run_manifest.json")
    dm = read(descriptors / "qualification.json")
    assert dm["pass"] and dm["base_data_sha256"] == sha(data_root / "run_manifest.json")
    assert dm["protocol_sha256"] == sha(PROTOCOL)
    groups = {k: sorted(p.split("/")[1] for p in manifest["views"] if p.startswith(k + "/"))
              for k in ("train", "val")}
    assert len(groups["train"]) == 40 and len(groups["val"]) == 8
    g1, l1 = P["arms"][arm]
    torch.manual_seed(seed)
    genotype = torch.Generator().manual_seed(seed)
    model = Reconstructor().cuda()
    param_count = base.n_params(model)
    assert param_count == 1056038
    base.FUSED = True
    initial_digest = hashlib.sha256()
    for param in model.parameters():
        initial_digest.update(param.detach().cpu().numpy().tobytes())
    dataset = data_api.TrainSet(data_root / "train", groups["train"], 4, "cuda")
    extra = np.stack([load_feature(descriptors, dm, "train", n) for n in groups["train"]])
    if not g1:
        extra.fill(0)
    dataset.aux = torch.cat((dataset.aux, torch.from_numpy(extra[:, None]).to("cuda", torch.float16)), 1)
    validation_data = []
    for name in groups["val"]:
        field = load_feature(descriptors, dm, "val", name)
        if not g1:
            field.fill(0)
        guides, target, _, noisy = data_api.load_view(data_root / "val" / name, 4)
        validation_data.append(({spp: append_aux(D_aux(guides, spp), field) for spp in SPPS}, target, noisy))

    config = P["training"]
    steps = config["iterations"]
    opt = torch.optim.Adam(model.parameters(), lr=config["adam_lr"])
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=config["cosine_final_lr"])
    record = {
        "arm": arm, "seed": seed, "commit": commit, "n_params": param_count, "input_channels": 15,
        "initial_sha256": initial_digest.hexdigest(), "data_manifest_sha256": dm["base_data_sha256"],
        "guide_manifest_sha256": sha(descriptors / "qualification.json"), "protocol_sha256": sha(PROTOCOL),
        "hardware": torch.cuda.get_device_name(0), "torch": torch.__version__, "python": platform.python_version(),
        "train_views": groups["train"], "validation_views": groups["val"], "trace": [], "loss": []
    }
    start = time.monotonic()
    for i in range(steps):
        comp, target, aux = dataset.batch(genotype, config["batch"], config["crop"])
        # WL105 batch appends log-spp after all resident channels.
        aux = torch.cat((aux[:, :13], aux[:, 14:15], aux[:, 13:14]), dim=1)
        y, components = model(comp, aux)
        objective = loss(y, components, target, aux, bool(l1))
        opt.zero_grad(set_to_none=True)
        objective.backward()
        opt.step()
        schedule.step()
        if i % 100 == 0:
            value = float(objective.detach())
            assert math.isfinite(value), (arm, seed, i)
            record["loss"].append([i, value])
        if (i + 1) % 2000 == 0:
            metrics = validation(model, validation_data)
            record["trace"].append({"iteration": i + 1, "validation_primary_unmasked": metrics})
            write(out / "train.json", record)
            print(arm, seed, i + 1, metrics, flush=True)
    path = out / "weights.pt"
    torch.save(model.state_dict(), path)
    record.update({"seconds": time.monotonic()-start, "steps": steps, "weights_sha256": sha(path), "complete": True})
    write(out / "train.json", record)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "descriptors", "output"):
        p.add_argument(name, type=Path)
    p.add_argument("arm", choices=ARMS)
    p.add_argument("seed", type=int, choices=SEEDS)
    p.add_argument("commit")
    args = p.parse_args()
    train(args.data, args.descriptors, args.output, args.arm, args.seed, args.commit)
