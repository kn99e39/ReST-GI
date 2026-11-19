"""Shared paths and frozen kernels for the WL104D factorial."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ecro_wl84_104b"))
sys.path.insert(0, str(HERE.parent / "ecro_wl84_104c"))
import h104b as H
import fit82 as F

P = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
S = P["seeds"]
CL, DA, CD = H.CL, H.DA, H.CD
ARMS = tuple(P["arms"])
STATES = (*ARMS, "ER_DF", "ER_L")
REPO = HERE.parents[1]


def write(path, result):
    H.C.write_json(path, result)


def byte_hash(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def population(source):
    rng = np.random.default_rng(S["material_draw"])
    config = P["population"]
    rows = []
    with np.load(source / "data/materials83.npz") as archive:
        for name, index, light in config["development"]:
            rows.append((name, index, light, "development", archive[name][index]))
        for name in config["held_out_sets"]:
            descriptors = archive[name]
            excluded = 2 if name in ("test83_iid", "test83_hard") else 0
            eligible = np.arange(excluded, len(descriptors))
            for index in rng.choice(eligible, config["held_out_draws_per_set"], replace=False):
                rows.append((name, int(index), 1 + (len(rows) - len(config["development"])) % 2, "held_out", descriptors[index]))
    table = [
        {"set": name, "index": index, "light": light, "cohort": cohort, "tag": f"{name}_{index}",
         "image": f"{name}_{index}_L{light}", "descriptor_sha256": byte_hash(desc)}
        for name, index, light, cohort, desc in rows
    ]
    descriptors = np.stack([row[4] for row in rows])
    if len(np.unique(descriptors, axis=0)) != len(table):
        raise ValueError("Duplicate material descriptor in WL104D population")
    return table, descriptors


def load_original(source, rows):
    fields = ("wi", "wo", "q", "use", "f", "fA", "fB")
    fits = []
    settings = P["training_data"]
    nw, no = settings["groups"], settings["lanes_per_group"]
    for name in sorted({row["set"] for row in rows}):
        with np.load(source / "data" / f"fit82_{name}.npz") as archive:
            if (archive["n_wi"], archive["n_wo"]) != (nw, no):
                raise ValueError("Unexpected WL83 fit budget")
            indices = {row["index"] for row in rows if row["set"] == name}
            arrays = {key: archive[key] for key in fields}
            groups = {key: a.reshape((-1, nw, no) + a.shape[1:]) for key, a in arrays.items()}
            fits.extend(((name, i), {k: group[i] for k, group in groups.items()}) for i in indices)
    indexed = dict(fits)
    return {k: np.stack([indexed[row["set"], row["index"]][k] for row in rows]) for k in fields}


def historical_states(source, rows):
    result = {}
    for arm, suffix in (("ER_DF", "_ER_cert_metric_12000_generic"), ("ER_L", "_L_ER_s8301")):
        paths = {row["set"]: source / "fit" / f"fit_{row['set']}{suffix}.npz" for row in rows}
        states = {}
        for name, path in paths.items():
            with np.load(path) as archive:
                states[name] = archive["z"]
        result[arm] = np.stack([states[row["set"]][row["index"]] for row in rows])
    return result


def relative_difference(a, b):
    return np.abs(a - b) / np.maximum(0.5 * (np.abs(a) + np.abs(b)), 1e-20)


class StopRun(RuntimeError):
    """A frozen evidence gate did not admit the next stage."""

    def __init__(self, stage, status):
        self.stage = stage
        self.status = status
        super().__init__(f"{stage}: {status}")
