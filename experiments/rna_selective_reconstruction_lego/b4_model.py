"""Architecture Batch 4 (worklog 37): surface ownership under a rigid geometry edit (RNA venv).

Isolated edit adapter around the unchanged worklog-32/36 representation (protocol b4_v1):
  * EditOwner: surface identity on the combined mesh of the CURRENT state (official Lego triangles + B's 12
    triangles, each object at its state offset), with the worklog-36 LegoOwner rule (candidates from centroids and
    auxiliary on-surface points, exact point-triangle distance, |n_t . n| >= 0.3). Unit: A point -> nearest L16 centre
    (A centres only) of its on-surface point; B point -> unit 16. Component: the triangle's (B = 1872, its own).
  * canonical frame: every object's T0 placement. canonical = world - offset(object, state); the edit is a rigid
    translation, so directions and normals are unchanged and a unit sees identical inputs for the same physical
    surface point in every state.
  * EditOwnedSurfaceDataset: upstream dataset + canonical positions (normalised with the canonical AABB) + owner id,
    exactly the worklog-32 position layout, so rso_model.OwnedTriplaneSet routes unchanged.
  * edit_unit_specs: the worklog-36 matched specs for the 16 A units (unchanged) + B's spec by the same texel rule.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "rna_surface_ownership_lego"))
sys.path.insert(0, str(HERE.parent / "rna_surface_ownership"))
import b3_model as B3  # noqa: E402
import rso_model as M  # noqa: E402

ROOT = HERE.parents[1]
SCENE = json.loads((ROOT / "results/rna_selective_reconstruction_lego/scene/scene.json").read_text()) \
    if (ROOT / "results/rna_selective_reconstruction_lego/scene/scene.json").exists() else None
B_UNIT = 16
B_COMP = 1872


def state_offsets(state, delta=None):
    """-> (offset of A, offset of B) of a state relative to the canonical (T0) placement."""
    d = np.asarray(delta if delta is not None else SCENE["delta"], np.float64)
    return {"T0": (0 * d, 0 * d), "T1": (0 * d, d), "T1b": (-d, 0 * d)}[state]


class EditOwner(B3.LegoOwner):
    """LegoOwner over the combined state mesh; .units = 17; see the module doc."""

    def __init__(self, partition_npz, mesh_npz, b_mesh_npz, a_off, b_off, n_aux=B3.N_AUX):
        from scipy.spatial import cKDTree

        p = np.load(partition_npz); m = np.load(mesh_npz); bm = np.load(b_mesh_npz)
        self.a_off, self.b_off = np.asarray(a_off, np.float64), np.asarray(b_off, np.float64)
        self.centres_A = p["centres"].astype(np.float64)                       # canonical (= official) frame
        assert len(self.centres_A) == B_UNIT
        self.centres = self.centres_A + self.a_off                             # world frame of this state
        self.units = B_UNIT + 1
        VA, FA = m["V"].astype(np.float64) + self.a_off, m["F"]
        VB, FB = bm["V"].astype(np.float64) + self.b_off, bm["F"]
        self.n_tri_A = len(FA)
        V = np.concatenate([VA, VB]); F = np.concatenate([FA, FB + len(VA)])
        self.tri_comp = np.concatenate([m["comp"].astype(np.int64), np.full(len(FB), B_COMP, np.int64)])
        self.a, self.b, self.c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        n = np.cross(self.b - self.a, self.c - self.a)
        self.tn = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)
        self.tree = cKDTree((self.a + self.b + self.c) / 3)
        area = 0.5 * np.linalg.norm(n, axis=1)
        rng = np.random.default_rng(0)
        n_aux = min(n_aux, 100 * len(F))
        t = rng.choice(len(F), n_aux, p=area / area.sum())
        u, v = rng.random(n_aux), rng.random(n_aux); fl = u + v > 1; u[fl], v[fl] = 1 - u[fl], 1 - v[fl]
        self.aux_tri = t
        self.aux_tree = cKDTree(self.a[t] + (self.b[t] - self.a[t]) * u[:, None] + (self.c[t] - self.a[t]) * v[:, None])
        h = hashlib.sha256(self.centres_A.tobytes() + bm["V"].astype(np.float64).tobytes() + self.a_off.tobytes() + self.b_off.tobytes())
        self.key = "b4_" + h.hexdigest()[:12]

    def __call__(self, world, normals=None, return_tri=False):
        lab, dist, (tri, comp, onsurf) = super().__call__(world, normals, return_tri=True)
        isB = tri >= self.n_tri_A
        lab = np.where(isB, B_UNIT, lab)          # A labels came from the A centres only (self.centres has 16 rows)
        return (lab, dist, (tri, comp, onsurf)) if return_tri else (lab, dist)

    def canonical(self, world, lab):
        """World positions -> canonical (T0-placement) positions, by the owner's object."""
        return world - np.where((lab == B_UNIT)[:, None], self.b_off[None], self.a_off[None])


def make_owner(state, root=ROOT):
    a_off, b_off = state_offsets(state)
    return EditOwner(root / "results/rna_surface_ownership_lego/partition_L16.npz", root / "results/rna_surface_ownership_lego/mesh/lego_mesh.npz",
                     root / "results/rna_selective_reconstruction_lego/scene/B_box_T0.npz", a_off, b_off)


class EditOwnedSurfaceDataset(M.rdatasets.NeuralSurfaceDataset):
    """Upstream dataset; positions -> canonical, normalised with the canonical AABB, + owner id (worklog-32 layout)."""

    def __init__(self, *args, owner=None, datafile=None, canonical_aabb=None, **kwargs):
        import torch

        super().__init__(*args, datafile=datafile, **kwargs)
        cmin, cmax = (torch.tensor(np.asarray(a), dtype=torch.float32, device=self.aabb_min.device) for a in canonical_aabb)
        self.owner_counts = np.zeros(owner.units, np.int64)
        self.owner_max_dist = 0.0
        self.shifted_px = 0
        for i in range(self.total_slices):
            p = self.positions[i]
            world = (p.to(self.ops_device) * (self.aabb_max - self.aabb_min) + self.aabb_min).cpu().numpy().astype(np.float64)
            nrm = self.normals[i].cpu().numpy().astype(np.float64)
            lab, d = B3.cached_owners(owner, datafile, world, nrm, i)
            lab = lab.astype(np.int64)
            canon = owner.canonical(world, lab)
            self.shifted_px += int((np.abs(canon - world).max(1) > 0).sum())
            pc = (torch.tensor(canon, dtype=torch.float32, device=cmin.device) - cmin) / (cmax - cmin)
            self.owner_counts += np.bincount(lab, minlength=owner.units)
            self.owner_max_dist = max(self.owner_max_dist, float(d.max()) if len(d) else 0.0)
            self.positions[i] = torch.cat([pc.to(p.dtype).to(p.device), torch.tensor(lab, dtype=p.dtype, device=p.device)[:, None]], -1)


class EditOwnedDataModule(M.rinterfaces.NeuralSurfaceDataModule):
    def __init__(self, conf, owner=None, canonical_aabb=None, **kw):
        super().__init__(conf, **kw)
        self.owner, self.canonical_aabb = owner, canonical_aabb

    def setup(self, stage=None):
        mk = lambda f: EditOwnedSurfaceDataset(datafile=f, use_for_visibility=self.use_for_visibility, use_positions=True, device=self.device,
                                               ops_device=self.ops_device, owner=self.owner, canonical_aabb=self.canonical_aabb)
        self.trainset = mk(self.conf.training_dataset)
        self.valset = mk(self.conf.validation_dataset) if self.conf.validation_dataset else None


def edit_unit_specs(partition_npz, base_model_params, dataset_aabb, b_mesh_npz, matched=True):
    """A: rso_model.unit_specs on L16 (the worklog-36 specs, K = 16 for the width rule); B: same texel rule on its AABB."""
    specs = M.unit_specs(partition_npz, base_model_params, dataset_aabb, matched=matched)
    assert len(specs) == B_UNIT
    V = np.load(b_mesh_npz)["V"].astype(np.float64)
    lo, hi = V.min(0), V.max(0); ext = np.maximum(hi - lo, 1e-3)
    lo, hi = lo - M.AABB_MARGIN * ext, hi + M.AABB_MARGIN * ext
    dmin, dmax = (np.asarray(a, np.float64) for a in dataset_aabb)
    p = dict(base_model_params)
    p["resolution"] = [int(np.clip(round(int(base_model_params["resolution"][0]) * (hi - lo).max() / (dmax - dmin).max()), 64,
                                   int(base_model_params["resolution"][0])))] * 2
    p["mlp_hidden_layer_neurons"] = int(round(int(base_model_params["mlp_hidden_layer_neurons"]) / math.sqrt(B_UNIT)))
    specs.append({"params": p, "lo": lo, "hi": hi})
    return specs
