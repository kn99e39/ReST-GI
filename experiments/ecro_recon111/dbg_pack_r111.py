"""Diagnostic (not evidence): do two frozen guide programs called alternately replay stale inputs?"""
import sys
import engine_r111 as E
import seq_r111 as SQ
rt, Bd, B, audit = E.admit(lambda *a: None, replay=False)
eng = E.Engine(rt, Bd, B)
c = SQ.sealed_candidates()
pattern = sys.argv[1]
for i, cam in enumerate(c[:6]):
    cd = E.cam_dict(cam)
    row = [i]
    for sub in [int(x) for x in pattern.split(",")]:
        g = eng.guides(cd, sub)
        row.append(round(float(g["valid_frac"].mean()), 4))
    print(pattern, *row, flush=True)
