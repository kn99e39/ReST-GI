"""Geometry-only check of the ReST-GI Lego + red box edit against 8DNA's empty-envelope condition (worklog 43).

8DNA bakes every path inside an asset's bounding box (code audit: the learned exit lies on the AABB, so the AABB is the
effective forbidden region; the paper states the convex hull). For the official B4 edit (box at T0 and T1 = T0 + 0.44 x),
this reports:
  1. whether the box lies inside the whole-Lego AABB;
  2. which connected components' AABBs the box enters (those parts cannot be separate 8DNA assets without cutting them);
  3. how many assets a decomposition along the box's face planes would need, if every remaining component keeps its own
     AABB outside both box positions (an illustrative count, not an optimal decomposition).
numpy only.

    python -I lego_envelope_analysis.py <lego_mesh.npz> <B_box_T0.npz> <out.json>
"""

import json
import sys

import numpy as np


def main(mesh_path, box_path, out_path):
    m = np.load(mesh_path)
    V, F, comp, mat, area = m["V"], m["F"], m["comp"], m["material"], m["area"]
    names = ["Yellow", "Gray", "transparent", "Black", "Red_Glass", "Light_brown", "Sand", "Brown"]
    bx = np.load(box_path)["V"]
    boxes = {"T0": (bx.min(0), bx.max(0)), "T1": (bx.min(0) + [0.44, 0, 0], bx.max(0) + [0.44, 0, 0])}

    lo_all, hi_all = V.min(0), V.max(0)
    res = {"lego_aabb": [lo_all.tolist(), hi_all.tolist()],
           "box": {k: [a.tolist(), b.tolist()] for k, (a, b) in boxes.items()},
           "box_inside_lego_aabb": {k: bool(np.all(a >= lo_all) and np.all(b <= hi_all)) for k, (a, b) in boxes.items()}}

    nc = int(comp.max()) + 1
    tri = V[F]  # (nF, 3, 3)
    tlo, thi = tri.min(1), tri.max(1)
    clo = np.full((nc, 3), np.inf)
    chi = np.full((nc, 3), -np.inf)
    np.minimum.at(clo, comp, tlo)
    np.maximum.at(chi, comp, thi)
    carea = np.bincount(comp, weights=area, minlength=nc)
    cmat = np.array([np.bincount(mat[comp == c], weights=area[comp == c]).argmax() if (comp == c).any() else -1 for c in range(nc)])

    def overlaps(lo, hi, a, b, margin=0.0):
        return np.all(lo <= b + margin, axis=1) & np.all(hi >= a - margin, axis=1)

    hit = np.zeros(nc, bool)
    for k, (a, b) in boxes.items():
        h = overlaps(clo, chi, a, b)
        hit |= h
        res[f"components_whose_aabb_contains_part_of_box_{k}"] = int(h.sum())
    idx = np.nonzero(hit)[0]
    res["conflicting_components"] = [{"comp": int(c), "material": names[cmat[c]] if cmat[c] >= 0 else None, "area_m2": float(carea[c]),
                                      "aabb": [clo[c].tolist(), chi[c].tolist()]} for c in idx[np.argsort(-carea[idx])]]
    res["conflicting_area_fraction"] = float(carea[hit].sum() / carea.sum())
    res["conflicting_area_by_material"] = {names[i]: float(carea[hit & (cmat == i)].sum()) for i in range(8) if (hit & (cmat == i)).any()}

    # regions bounded by the box face planes: every component entirely inside one box-free slab can share an asset
    a0, b0 = boxes["T0"]
    a1, b1 = boxes["T1"]
    zlo, zhi = a0[2], b0[2]
    ylo, yhi = a0[1], b0[1]
    regions = {
        "below_box_bottom": chi[:, 2] < zlo,
        "above_box_top": clo[:, 2] > zhi,
        "front_of_boxes(y<min)": chi[:, 1] < ylo,
        "behind_boxes(y>max)": clo[:, 1] > yhi,
        "left_of_T0(x<min)": chi[:, 0] < a0[0],
        "between_T0_T1": (clo[:, 0] > b0[0]) & (chi[:, 0] < a1[0]),
        "right_of_T1(x>max)": clo[:, 0] > b1[0],
    }
    assigned = np.zeros(nc, bool)
    groups = {}
    for name, mask in regions.items():
        sel = mask & ~assigned & ~hit
        if sel.any():
            groups[name] = {"components": int(sel.sum()), "area_m2": float(carea[sel].sum())}
            assigned |= sel
    left = ~assigned & ~hit
    res["slab_decomposition"] = {"groups": groups, "n_groups": len(groups),
                                 "components_in_no_single_slab_without_conflict": int(left.sum()),
                                 "their_area_m2": float(carea[left].sum()),
                                 "assets_needed_lower_bound_illustrative": len(groups) + int(hit.sum()) + int(left.sum()),
                                 "note": "groups whose AABB stays outside both box positions; conflicting components would still contain the box in their own AABB and would have to be cut; this is an illustrative count, not an optimal decomposition"}
    # 8DNA also needs every OTHER asset's geometry outside each asset's AABB (code case b between learned assets).
    # (a) the slab groups above: how much other-group surface lies inside each group's AABB
    cent = tri.mean(1)
    gid = np.full(nc, -1)
    for gi, (name, mask) in enumerate(regions.items()):
        gid[mask & (gid < 0) & ~hit] = gi
    tg = gid[comp]
    cross = {}
    for gi, name in enumerate(regions):
        sel = gid == gi
        if not sel.any():
            continue
        glo, ghi = clo[sel].min(0), chi[sel].max(0)
        inside = np.all((cent >= glo) & (cent <= ghi), axis=1) & (tg != gi)
        cross[name] = {"aabb": [glo.tolist(), ghi.tolist()], "other_group_area_inside_m2": float(area[inside].sum()),
                       "other_group_area_fraction": float(area[inside].sum() / area.sum())}
    res["slab_decomposition"]["other_groups_inside_each_group_aabb"] = cross

    # (b) the finest partition of whole components into clusters with pairwise disjoint AABBs: components whose AABBs
    # overlap must share a cluster (otherwise the cluster AABBs overlap); merge until no two cluster AABBs overlap
    parent = np.arange(nc)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    lab = np.arange(nc)
    for it in range(100):
        ids = np.unique(lab)
        blo = np.array([clo[lab == k].min(0) for k in ids])
        bhi = np.array([chi[lab == k].max(0) for k in ids])
        ov = np.all(blo[:, None, :] <= bhi[None, :, :], axis=2) & np.all(bhi[:, None, :] >= blo[None, :, :], axis=2)
        np.fill_diagonal(ov, False)
        if not ov.any():
            break
        parent[:] = np.arange(nc)
        for a_, b_ in zip(*np.nonzero(np.triu(ov))):
            ra, rb = find(ids[a_]), find(ids[b_])
            if ra != rb:
                parent[ra] = rb
        roots = np.array([find(k) for k in ids])
        remap = dict(zip(ids, roots))
        lab = np.array([remap[v] for v in lab])
    ids = np.unique(lab)
    clusters = []
    for k in ids:
        sel = lab == k
        a_, b_ = clo[sel].min(0), chi[sel].max(0)
        clusters.append({"components": int(sel.sum()), "area_m2": float(carea[sel].sum()), "aabb": [a_.tolist(), b_.tolist()],
                         "overlaps_box": {t: bool(np.all(a_ <= bb) and np.all(b_ >= aa)) for t, (aa, bb) in boxes.items()}})
    clusters.sort(key=lambda c: -c["area_m2"])
    res["disjoint_aabb_partition"] = {"iterations": it, "n_clusters": len(clusters), "largest": clusters[:10],
                                      "clusters_overlapping_a_box": int(sum(any(c["overlaps_box"].values()) for c in clusters)),
                                      "note": "finest partition of intact connected components with pairwise disjoint AABBs; a box overlapping a cluster AABB means no such decomposition keeps the box outside every learned envelope without cutting meshes"}
    res["n_components"] = nc
    with open(out_path, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "conflicting_components"}, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:4])
