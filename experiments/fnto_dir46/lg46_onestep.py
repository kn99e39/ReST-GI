"""WL46 LG one-step diagnostics (protocol amendment 1: R3 / R5 are not production-capable - rho 9.4 / 17.1, solves diverge -
so their Lego evidence is the FIRST gather only, for all conditions alike).

Held-out T0 views (default all 16): from L = 0, one DirState step (each gather reads the sun-driven first-bounce sender value
of the condition), then the production render. Per view and condition: display PSNR vs full_r1 (all / Yellow / opaque / box)
and the single-bounce indirect (prediction - frozen K_only) per class as a luminance ratio to the reference indirect
(full_r1 - first_r1, all bounces) and its correlation. Two-factor interaction (A outgoing x B incoming) on every metric:
R5 - R3 - R4 + R0. Basis error (R3x, view 6): one-step ray values of the 19-bin basis vs the exact per-ray query.

    python lg46_onestep.py <out dir> [views=all]
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dir46_solve as DS  # noqa: E402
import r1_common as C  # noqa: E402
import fnto_model as M  # noqa: E402
from lg46_run import CONDS  # noqa: E402

LUM = np.array([0.2126, 0.7152, 0.0722]); CLS = {"Yellow": [0], "opaque": [1, 3, 5, 6, 7], "glass": [2, 4], "box": [8]}


def psnr(p, r):
    return float(10 * np.log10(1 / max(((C.ldr3(p) - C.ldr3(r)) ** 2).mean(), 1e-30)))


@torch.no_grad()
def main(out, views=None):
    import r1_solve as SV
    from r1_eval import load_kernels
    out = Path(out); out.mkdir(parents=True, exist_ok=True); dev = torch.device("cuda")
    KA, KB = load_kernels(str(C.OUT1 / "K_k44P/kernels.pt"), dev); dig0 = M.digest(KA, KB); T = SV.Tables(KA, KB, dev)
    H = np.load(C.OUT1 / "rel_T0.npz")["hit"]; vis = np.load(C.OUT1 / "eval_vis_k44.npz"); px = np.load(C.OUT1 / "eval_px/heldout_T0.npz")
    first = np.load(C.OUT1 / "refs/heldout_T0_first.npz"); full = np.load(C.OUT1 / "refs/heldout_T0_full.npz"); Vk = np.load(C.OUT1 / "eval_K_k44P/views.npz")
    conds = ("R0", "R3", "R4", "R5"); rec = {"per_view": {}, "conds": conds}
    for v in (range(int(px["nv"])) if views is None else views):
        val = px[f"v{v}_valid"].astype(bool); mat = np.where(px[f"v{v}_dom"][val] == 1, 8, px[f"v{v}_mat"][val])
        pxd = {k: torch.as_tensor(px[f"v{v}_{k}"][val].astype(np.float32 if k in ("pos", "nrm", "cam", "vis") else np.int64), device=dev) for k in ("pos", "nrm", "cam", "vis", "dom", "port")}
        fu = full[f"v{v}_rgb"][val].astype(np.float64); fr = first[f"v{v}_rgb"][val].astype(np.float64); ko = Vk[f"heldout_T0_v{v}_K_only"][:, :3].astype(np.float64)
        rind = (fu - fr) @ LUM; r = {}
        for c in conds:
            S = DS.DirState(T, H, vis[f"heldout_T0_v{v}_T0"].astype(np.float32), px[f"v{v}_light"], **CONDS[c]); S.step()
            Eind, h = S.aggregates(); S.Eind, S.h = Eind, h; S.outgoing(Eind, h)
            pr = SV.render_pixels(T, S, pxd)[0].cpu().numpy().astype(np.float64); pind = (pr - ko) @ LUM
            m_ = {"all": psnr(pr, fu)}
            for cn, ids in CLS.items():
                m = np.isin(mat, ids)
                if m.sum() > 100:
                    m_[f"psnr_{cn}"] = psnr(pr[m], fu[m]); m_[f"ind_ratio_{cn}"] = float(pind[m].sum() / max(rind[m].sum(), 1e-30))
                    m_[f"ind_corr_{cn}"] = float(np.corrcoef(pind[m], rind[m])[0, 1])
            m_["max_L"] = float(S.L.float().abs().max()); r[c] = m_
            if v == 6 and c == "R3":
                Lb = S.L.cpu(); del S; torch.cuda.empty_cache()
                Sx = DS.DirState(T, H, vis[f"heldout_T0_v{v}_T0"].astype(np.float32), px[f"v{v}_light"], **CONDS["R3x"]); Sx.step(); Lx = Sx.L.cpu(); del Sx
                rec["basis_error_view6"] = {"rel_l2": float((Lb.float() - Lx.float()).norm() / Lx.float().norm()),
                                            "lum_ratio_basis_over_exact": float((Lb.float() @ torch.tensor(LUM, dtype=torch.float32)).sum() / (Lx.float() @ torch.tensor(LUM, dtype=torch.float32)).sum())}
                print("basis error", rec["basis_error_view6"], flush=True)
            else:
                del S
            torch.cuda.empty_cache()
        r["interaction_R5-R3-R4+R0"] = {k: r["R5"][k] - r["R3"][k] - r["R4"][k] + r["R0"][k] for k in r["R0"] if k in r["R5"] and k != "max_L"}
        rec["per_view"][f"view{v}"] = r; print(v, json.dumps({c: {k: round(x, 3) for k, x in r[c].items()} for c in conds}), flush=True)
        (out / "lg46_onestep.json").write_text(json.dumps(rec, indent=1))
    keys = [k for k in rec["per_view"][next(iter(rec["per_view"]))]["R0"] if k != "max_L"]
    rec["mean_over_views"] = {c: {k: float(np.mean([pv[c][k] for pv in rec["per_view"].values() if k in pv[c]])) for k in keys} for c in conds}
    rng = np.random.default_rng(46); vs = list(rec["per_view"])
    rec["paired_vs_R0"] = {}
    for c in conds[1:]:
        d = np.array([rec["per_view"][x][c]["all"] - rec["per_view"][x]["R0"]["all"] for x in vs])
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
        rec["paired_vs_R0"][c] = {"mean_dB": float(d.mean()), "ci95": np.quantile(bs, [0.025, 0.975]).tolist(), "n_pos": int((d > 0).sum()), "n": len(d)}
    rec["frozen_digest_identical"] = M.digest(KA, KB) == dig0
    (out / "lg46_onestep.json").write_text(json.dumps(rec, indent=1)); print(json.dumps({k: rec[k] for k in ("mean_over_views", "paired_vs_R0")}, indent=1))


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], None if len(a) < 2 or a[1] == "all" else [int(x) for x in a[1].split(",")])
