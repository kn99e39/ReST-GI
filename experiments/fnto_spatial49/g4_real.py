"""WL49 Gate 4 - real-scene Yellow first-event fidelity vs the first_r1 Cycles references (held-out views 8-15, T0 and T1).

Per Yellow pixel, evaluated at its port (self excluded) with the SAME shared terms for every arm:
  pred_arm = s_c * (arm in-support response + oracle > 30 cm tail + self term k_c(2 mm) A_port E_port)
arms: L0 / L1 (Gate 3 checkpoints, both seeds), oracle in-support (reference operator); K44 K_only (historical, camera
readout). s_c = worklog 45 'obj' scales (fitted on T0 views 0-7). Metrics: display PSNR (ldr3) lit / shadow / 3-30 cm,
linear rel L2, luminance ratio, corr, per view; T0 -> T1 change on common pixels. Two panel views use all Yellow pixels.

    python g4_real.py <out root>
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s49_data as S  # noqa: E402
import dasq as DQ  # noqa: E402
import g3_learned as G3  # noqa: E402
import r1_common as C  # noqa: E402

LUM = np.array([0.2126, 0.7152, 0.0722]); TEST_VIEWS = list(range(8, 16)); PANEL_VIEWS = [8, 12]; NPX = 8000


def psnr(p, r):
    return float(10 * np.log10(1 / max(((C.ldr3(p) - C.ldr3(r)) ** 2).mean(), 1e-30)))


@torch.no_grad()
def main(root):
    root = Path(root); out = root / "gate4_real_scene"; out.mkdir(parents=True, exist_ok=True); dev = torch.device("cuda"); t00 = time.time()
    L = S.Lego(dev); T = DQ.Octree(L.pos, L.area); DQ.set_normals(T, L.nrm); sc = np.array(S.s_c()); dch = S.dch()
    px = {st: np.load(C.OUT1 / f"eval_px/heldout_{st}.npz") for st in ("T0", "T1")}
    ref = {st: np.load(C.OUT1 / f"refs/heldout_{st}_first.npz") for st in ("T0", "T1")}
    V = np.load(C.OUT1 / "eval_K_k44P/views.npz")
    sel = {}; rng = np.random.default_rng(4904)
    for v in TEST_VIEWS:                                                       # pixels Yellow + valid in BOTH states (by image index mk)
        ids = []
        for st in ("T0", "T1"):
            p = px[st]; val = p[f"v{v}_valid"].astype(bool); y = val & (p[f"v{v}_dom"] == 0) & (p[f"v{v}_mat"] == 0); ids.append(set(p[f"v{v}_mk"][y].tolist()))
        common = np.array(sorted(ids[0] & ids[1]))
        sel[v] = common if v in PANEL_VIEWS else np.sort(rng.choice(common, min(NPX, len(common)), replace=False))
    rows = {}; ports_all = set()
    for st in ("T0", "T1"):
        for v in TEST_VIEWS:
            p = px[st]; val = np.flatnonzero(p[f"v{v}_valid"].astype(bool)); mk = p[f"v{v}_mk"]
            pos_of = {int(m): i for i, m in enumerate(mk)}; idx = np.array([pos_of[int(m)] for m in sel[v]])       # rows in the eval_px arrays
            vrow = np.searchsorted(val, idx)                                                                   # rows among valid pixels (views.npz / refs)
            rows[(st, v)] = (idx, vrow); ports_all.update(p[f"v{v}_port"][idx].astype(np.int64).tolist())
    ports = np.array(sorted(ports_all)); pmap = {p_: i for i, p_ in enumerate(ports)}; xi = torch.as_tensor(ports, device=dev)
    rec = {"ports": int(len(ports)), "pixels_per_view": {str(v): int(len(sel[v])) for v in TEST_VIEWS}, "s_c": sc.tolist(), "states": {}}
    ck = {(a, s_): torch.load(root / f"gate3_learned_ab/ckpt_{a}_seed{s_}.pt") for a in ("L0", "L1") for s_ in G3.SEEDS}
    preds = {}
    for st in ("T0", "T1"):
        E = S.view_fields(L, st); Ev = E[:, TEST_VIEWS]
        dist = S.dist_to_lit(L, xi, Ev)                                                # [P, 8]
        base = torch.zeros(len(ports), 3, len(TEST_VIEWS), device=dev, dtype=torch.float64); orin = torch.zeros_like(base)
        for ci, d in enumerate(dch):
            yS, yT = S.oracle(L, xi, Ev, d); selfk = float(S.burley(torch.tensor(2e-3, dtype=torch.float64), d))
            base[:, ci] = yT + selfk * (L.area[xi, None] * Ev[xi]).double(); orin[:, ci] = yS
        arm_in = {}
        for arm in ("L0", "L1"):
            D, meas, valid, F = G3.arm_tables(arm, L, T, ports, {"v": Ev})
            for s_ in G3.SEEDS:
                net = G3.Net().to(dev); net.load_state_dict(ck[(arm, s_)])
                arm_in[(arm, s_)] = torch.cat([G3.predict(net, D[i:i + 200], meas[i:i + 200], valid[i:i + 200], F["v"][i:i + 200]) for i in range(0, len(ports), 200)]).double()
            del D, meas, valid, F; torch.cuda.empty_cache()
        scale = torch.as_tensor(sc, device=dev)[None, :, None]
        cond = {"oracle": scale * (orin + base), **{f"{a}_s{s_}": scale * (arm_in[(a, s_)] + base) for a in ("L0", "L1") for s_ in G3.SEEDS}}
        for v_i, v in enumerate(TEST_VIEWS):
            idx, vrow = rows[(st, v)]; pp = np.array([pmap[int(q)] for q in px[st][f"v{v}_port"][idx]])
            r_ = ref[st][f"v{v}_rgb"][idx].astype(np.float64)                    # refs rows align with the eval_px arrays (amendment 6 fix); lit = px[st][f"v{v}_vis"][idx].astype(bool); dl = dist[pp, v_i].cpu().numpy()
            preds[(st, v, "ref")] = r_; preds[(st, v, "K44")] = V[f"heldout_{st}_v{v}_K_only"][vrow][:, :3].astype(np.float64)
            for cn, cv in cond.items():
                preds[(st, v, cn)] = cv[pp, :, v_i].cpu().numpy()
            preds[(st, v, "lit")] = lit; preds[(st, v, "dist")] = dl
    conds = ["K44", "oracle"] + [f"{a}_s{s_}" for a in ("L0", "L1") for s_ in G3.SEEDS]
    for st in ("T0", "T1"):
        r = {"per_view": {}, "pooled": {}}
        cat = lambda key: np.concatenate([preds[(st, v, key)] for v in TEST_VIEWS])  # noqa: E731
        R_, lit, dl = cat("ref"), cat("lit"), cat("dist")
        masks = {"all": np.ones(len(R_), bool), "lit": lit, "shadow": ~lit, "shadow_3_30cm": (~lit) & (dl >= 0.03) & (dl < 0.30)}
        for cn in conds:
            P_ = cat(cn)
            r["pooled"][cn] = {**{f"psnr_{m}": psnr(P_[mm], R_[mm]) for m, mm in masks.items()},
                               "rel_L2_lin": float(np.linalg.norm(P_ - R_) / np.linalg.norm(R_)), "lum_ratio": float((P_ @ LUM).sum() / (R_ @ LUM).sum()),
                               "corr_lum": float(np.corrcoef(P_ @ LUM, R_ @ LUM)[0, 1]),
                               "per_channel_psnr": [psnr(P_[:, c], R_[:, c]) for c in range(3)]}
            r["per_view"][cn] = [psnr(preds[(st, v, cn)], preds[(st, v, "ref")]) for v in TEST_VIEWS]
        bs = np.random.default_rng(1)
        for s_ in G3.SEEDS:
            d_ = np.array(r["per_view"][f"L1_s{s_}"]) - np.array(r["per_view"][f"L0_s{s_}"]); boots = [d_[bs.integers(0, len(d_), len(d_))].mean() for _ in range(400)]
            r[f"L1_minus_L0_psnr_seed{s_}"] = {"mean": float(d_.mean()), "ci95": np.quantile(boots, [0.025, 0.975]).tolist(), "n_views_pos": int((d_ > 0).sum())}
        rec["states"][st] = r; print(st, json.dumps({cn: {k: round(v, 3) for k, v in r["pooled"][cn].items() if isinstance(v, float)} for cn in conds}), flush=True)
    ed = {}
    for cn in ["ref"] + conds:
        ed[cn] = np.concatenate([(preds[("T1", v, cn)] - preds[("T0", v, cn)]) @ LUM for v in TEST_VIEWS])
    dr = ed["ref"]; top = np.abs(dr) >= np.quantile(np.abs(dr), 0.9)
    rec["edit_T0_T1"] = {cn: {"corr": float(np.corrcoef(ed[cn], dr)[0, 1]), "slope": float((ed[cn] * dr).sum() / (dr * dr).sum()),
                              "corr_top10pct": float(np.corrcoef(ed[cn][top], dr[top])[0, 1])} for cn in conds}
    rec["edit_T0_T1"]["ref_change_rms"] = float(np.sqrt((dr ** 2).mean()))
    np.savez_compressed(out / "g4_panel_views.npz", **{f"{st}_v{v}_{cn}": preds[(st, v, cn)].astype(np.float32) for st in ("T0", "T1") for v in PANEL_VIEWS for cn in ["ref"] + conds},
                        **{f"{st}_v{v}_mk": sel[v] for st in ("T0", "T1") for v in PANEL_VIEWS})
    rec["composition"] = "NOT APPLICABLE: integrating L0 / L1 into the coupled r1 solve would modify production K / C (not authorised); first-event only"
    rec["cost"] = {"wall_s": time.time() - t00, "peak_GB": torch.cuda.max_memory_allocated() / 1e9}
    (out / "gate4.json").write_text(json.dumps(rec, indent=1)); print(json.dumps(rec["edit_T0_T1"], indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
