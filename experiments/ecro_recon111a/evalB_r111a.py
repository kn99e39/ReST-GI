"""WL111A Stage B: cold-frame bias / variance of the factorial arms on the Stage A development realization sets (identical
seeds, hence paired with Stage A's R3 / S111 cells).

python evalB_r111a.py [--label L] ARM:SUB[:NET] ...   (each arm has realizations r0 and r1 in OUT/models; NET default ctak)
  -> OUT/stageB/<L>.json (default devset_B), OUT/stageB/init_check_<L>.json
"""
import os
import sys

import common_r111a as A
import common_r111 as C
import engine_r111 as E
import arms_r111 as AR
import seq_r111 as SQ
import stageA_r111a as SA
import recon_r111a as RA


def main(specs, label="devset_B"):
    out = A.OUT / "stageB"
    out.mkdir(parents=True, exist_ok=True)
    eng, audit = SA.setup()
    cams = {k: E.cam_dict(c) for k, c in SQ.wl105_cameras().items()}
    extra, chk = {}, {}
    for s in specs:
        arm, sub, net = (s.split(":") + ["ctak"])[:3]
        for ri in (0, 1):
            lg = C.read(A.OUT / "models" / f"{arm}_r{ri}_log.json")
            w111 = C.read(A.WL111_EVID / "models" / f"cand_r{ri}_log.json")
            chk[f"{arm}_r{ri}"] = {"net": net, "init_equals_wl111": lg["init_sha256"] == w111["init_sha256"],
                                   "weights_sha256": C.sha(A.OUT / "models" / f"{arm}_r{ri}.pt"), "seconds": lg["seconds"]}
            extra[f"{arm}_r{ri}"] = (RA.load_candidate(A.OUT / "models" / f"{arm}_r{ri}.pt", net), int(sub))
    C.write(out / ("init_check.json" if label == "devset_B" else f"init_check_{label}.json"), chk)
    print("init", chk, flush=True)
    smoke = os.environ.get("RECON111A_SMOKE") == "1"
    SA.devset(eng, cams, out, 4 if smoke else 64, smoke, rec=SA.Recons(extra), label=label, only=set(extra) | {"R3", "S111_r0", "S111_r1"})
    print("EVALB_COMPLETE", flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    lab = "devset_B"
    if args and args[0] == "--label":
        lab, args = args[1], args[2:]
    main(args, lab)
