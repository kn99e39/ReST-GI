"""WL101 evidence chain (local, sequential, resumable: a step whose output exists is skipped).
  python chain101.py <out_root>
"""
import os, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = str(HERE.parent.parent / 'external' / '8dna26' / '.venv-cu128' / 'Scripts' / 'python.exe')
O = Path(sys.argv[1]).resolve()
(O / 'timing').mkdir(parents=True, exist_ok=True); (O / 'quality').mkdir(parents=True, exist_ok=True); (O / 'scaling').mkdir(parents=True, exist_ok=True)
LOG = open(O / 'chain.log', 'a', buffering=1)


def run(out, *args):
    if Path(out).exists():
        LOG.write(f'SKIP {out}\n'); return
    t0 = time.time()
    LOG.write(f'START {time.strftime("%Y-%m-%d %H:%M:%S")} {" ".join(map(str, args))}\n')
    with open(str(out) + '.log', 'w') as f:
        r = subprocess.run([PY, '-u', *map(str, args)], cwd=str(HERE), stdout=f, stderr=subprocess.STDOUT)
    LOG.write(f'END rc={r.returncode} {time.time() - t0:.1f}s {out}\n')
    if r.returncode != 0 or not Path(out).exists():
        LOG.write(f'FAILED {out}\n'); (O / 'chain.status').write_text(f'FAILED {out}\n'); sys.exit(1)


T = O / 'timing'
for wl in ['SEL', 'V_SSS_TOUR', 'V_NEAR_TOUR', 'V_SSS_ANIM']:
    run(T / f'{wl}_A_sel_r1.json', 'run_a101.py', wl, 'A_sel', T / f'{wl}_A_sel_r1')
    run(T / f'{wl}_B_sel_r1.json', 'run_b101.py', wl, 'B_sel', T / f'{wl}_B_sel_r1', f'--cmp={T / f"{wl}_A_sel_r1_dump.npz"}')
    run(T / f'{wl}_A_sel_r2.json', 'run_a101.py', wl, 'A_sel', T / f'{wl}_A_sel_r2', '--nodump')
    run(T / f'{wl}_B_sel_r2.json', 'run_b101.py', wl, 'B_sel', T / f'{wl}_B_sel_r2', f'--cmp={T / f"{wl}_A_sel_r1_dump.npz"}')
    run(T / f'{wl}_A_full_r1.json', 'run_a101.py', wl, 'A_full', T / f'{wl}_A_full_r1')
    run(T / f'{wl}_B_full_r1.json', 'run_b101.py', wl, 'B_full', T / f'{wl}_B_full_r1')
    run(T / f'{wl}_B_eager_r1.json', 'run_b101.py', wl, 'B_eager', T / f'{wl}_B_eager_r1')
Qd = O / 'quality'
for wl in ['V_SSS_TOUR', 'V_NEAR_TOUR']:
    case = 'sss' if 'SSS' in wl else 'near'
    d = Qd / case
    run(d / 'MAP_1.npz', 'qual_a101.py', 'ref', wl, d)
    run(d / 'PRED_A_1.json', 'qual_a101.py', 'pred', wl, d)
    run(d / 'PRED_B_1.json', 'qual_b101.py', wl, d)
Sd = O / 'scaling'
for s in ['1.5', '2.0', '3.0']:
    run(Sd / f'V_SSS_TOUR_A_sel_s{s}.json', 'run_a101.py', 'V_SSS_TOUR', 'A_sel', Sd / f'V_SSS_TOUR_A_sel_s{s}', f'--scale={s}')
    run(Sd / f'V_SSS_TOUR_B_sel_s{s}.json', 'run_b101.py', 'V_SSS_TOUR', 'B_sel', Sd / f'V_SSS_TOUR_B_sel_s{s}', f'--scale={s}')
    run(Sd / f'V_SSS_TOUR_B_full_s{s}.json', 'run_b101.py', 'V_SSS_TOUR', 'B_full', Sd / f'V_SSS_TOUR_B_full_s{s}', f'--scale={s}')
(O / 'chain.status').write_text('DONE\n')
LOG.write('CHAIN DONE\n')
