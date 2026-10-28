"""WL54 Stage E amended tables (medium K, coarse K interface): group biases under the amended coverage rule, all three channels, plus the paired
stale-vs-current K differential with the ranking rule of the protocol.
    P53_MI_VARIANT=cuda_ad_rgb python c54_amendE.py <out json> <wl53 results> <stageE dir> <scene,...>"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "phys_ref53"))
import c54_amend1 as AM  # noqa: E402
import c54_common as CC  # noqa: E402
import c54_stageB as SB  # noqa: E402
import c54_stageE as SE  # noqa: E402
import p53_compose as CP  # noqa: E402
import p53_fixtures as FX  # noqa: E402
import p53_oracle as OR  # noqa: E402
import p53_report as RP  # noqa: E402


def stats3(ref, pred, ok, groups):
    return {c: SB.group_stats(pred[:, c], ref[:, :, c].mean(0), ok, groups, ref[:, :, c].std(0, ddof=1) / np.sqrt(len(ref))) for c in range(3)}


def main():
    out = Path(sys.argv[1]); R = Path(sys.argv[2]); edir = Path(sys.argv[3]); res = {}
    for scene in sys.argv[4].split(","):
        cfg = SE.SCENES[scene]; sd = FX.scene_def(scene); groups = RP.port_groups(scene); ss0 = OR.scene_set(sd); cdir = R / "cycles"; geo = json.load(open(cdir / f"ccam_{scene}.json"))
        P = dict(np.load(edir / f"{scene}_preds.npz")); stored = np.load(R / "oracle" / "pred" / f"{scene}__{'current' if scene in ('bend90', 'dent') else 'part_s1'}.npz"); res[scene] = {}
        for cam, g in geo.items():
            files = sorted(cdir.glob(f"cimg_{scene}_{cam}_*.npy")); pid = CP.classify(ss0, CC.blender_ortho_rays(g, SB.V1_OFF))
            means = [CP.port_means(pid, np.load(f).reshape(-1, 3), ss0.Ptot) for f in files]; ref = np.stack([x[0] for x in means]); cnt = means[0][1]
            cov = cnt / np.maximum(AM.expected_pixels(ss0, g), 1e-9); ok_a = (cnt >= SB.MINPIX) & (cov >= AM.COV); part = (cnt >= SB.MINPIX) & (cov < AM.COV)
            key = f"{RP.dirkey(sd['cams'][cam]['dir'])}__nee"; rec = dict(n_amended=int(ok_a.sum()), n_partial=int(part.sum()), amended={}, partial={})
            arms = {"C0stored": stored[key]} | {a: P[f"{a}|{key}"] for a in ("C0rebuilt", "C1", "C1_r0", "C1_r1", "C1_r2", "Cfine_s4") if f"{a}|{key}" in P}
            for tag, ok in (("amended", ok_a), ("partial", part)):
                if ok.any():
                    rec[tag] = {a: stats3(ref, v, ok, groups) for a, v in arms.items()}
            if "stale" in cfg:
                kb = f"{RP.dirkey(sd['cams'][cam]['dir'])}__bin"; rec["differential"] = {}
                for a in ("C0rebuilt", "C1"):
                    cur, st = P[f"{a}|cur_bin|{kb}"], P[f"{a}|stale_bin|{kb}"]; D = st - cur
                    rec["differential"][a] = {}
                    for c in range(3):
                        se = ref[:, :, c].std(0, ddof=1) / np.sqrt(len(ref)); rm = ref[:, :, c].mean(0); gs = {}
                        for gn in np.unique(groups[ok_a]):
                            m = ok_a & (groups == gn); sig = rm[m].sum()
                            if sig <= 0 or m.sum() < SB.MINPORTS:
                                continue
                            Dg = float(D[m, c].sum() / sig); Bc = float((cur[m, c] - rm[m]).sum() / sig); Bs = float((st[m, c] - rm[m]).sum() / sig); sg = float(np.sqrt((se[m] ** 2).sum()) / sig)
                            gs[gn] = dict(D=Dg, B_current=Bc, B_stale=Bs, se=sg, share=float(sig / rm[ok_a].sum()), rank_reliable=bool(abs(Dg) > 2 * (abs(Bc) + 3 * sg)), n=int(m.sum()))
                        rec["differential"][a][c] = gs
            res[scene][cam] = rec
        print("done", scene, flush=True)
    CC.dump(res, out)


if __name__ == "__main__":
    main()
