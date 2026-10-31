# WL67 — Baseline Superiority Evidence (`baseline_sup67_v1`)

Isolated comparison adapters; no historical source is edited. Protocol: `protocol/bsup67.json` (predeclared).

| File | Role |
|---|---|
| `b67_w1_cycles.py` | W1 (WL56 two-slab medium): Cycles volume Full Recompute spp ladder + RNA-native bounce/clamp check (Blender 3.5.1 CUDA) |
| `chain.sh` | server chain; W1 oracle re-execution calls the unchanged `fnto_sel56/s56.py` with `WL56=~/rest_bsup67/w1oracle` |
| `b67_w2_pt.py` | W2 (WL22 teaset common light): Mitsuba path-traced Full Recompute ladder, cross-machine check, released 8DNA re-execution |
| `b67_blockers.py` | RNA training-view coverage of W1 queries and cited native-generator settings |
| `b67_w1_analyze.py`, `b67_w2_analyze.py` | scoring with the unchanged WL56 `group_stats` and WL22 `refit_block` |
| `tests/test_b67.py` | focused identity / seed / reference tests, run first in the chain |

Server root `~/rest_bsup67` (LabServer69). Nothing under `fnto_rank66` / WL66 is read or written.
