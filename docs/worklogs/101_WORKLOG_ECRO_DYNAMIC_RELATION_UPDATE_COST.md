# Worklog — ECRO Dynamic Relation-Update Cost: Attribution and Bounded Runtime Optimization (2026-11-17)

> Append-only execution record. Brief: "Prioritize ECRO dynamic-scene viability" (user, 2026-11-17), local execution preferred.
> Batch class: regular **WL101** — an independent question (dynamic-update cost of the existing ECRO) that is neither a closure of WL100
> nor part of the concurrent RB-DTO-E work (WL99 / WL99A). The owner may reclassify; nothing else is renumbered.

## Scope and invariants

- **Question.** Can ECRO keep its learned higher-order transport across geometry-relation changes without retraining while the
  end-to-end dynamic frame approaches 20 / 15 / 10 ms, and is any remaining limit implementation-level or architectural?
- **Preserved.** ECRO v0 architecture (`model86.py`), the WL100 checkpoints (`results/ecro_material100/{near,sss}/model.pt`),
  physical LOW (events 1–2) / learned higher order (events ≥ 3) ownership, the WL100 physical material adapters, the source files
  `pt86.py`, `pt100.py`, `closure83_dr.py`, `scene86.py`, `model86.py`, `common100.py` (byte-identical; hashes recorded in every
  output). No training, no capacity / material / target / threshold change. RB-DTO-E results are not used.
- **Changed.** Only the dynamic update and execution organisation: data residency, dirty-ray scheduling, reductions, hand-over and
  kernel replay. Code: `experiments/ecro_dyn101/`; raw outputs `results/ecro_dyn101/`; reviewer artifacts `evaluation/101/`.
- **Hardware.** Local PC: RTX 5080 16 GB (driver 596.49), AMD Ryzen 9 9950X3D, 64 GB, Windows 11. Arm A: venv
  `external/8dna26/.venv-cu128` (Mitsuba 3.5.1, Dr.Jit 0.4.6, torch 2.8.0+cu128). Arm B: same venv + Mitsuba 3.9.1 / Dr.Jit 1.5.0
  installed with `pip --target external/mi39_dyn101 --no-deps` (the frozen venv is untouched).
- **Terminology.** The comparator "C" is the same-checkpoint internal **full transport-state rebuild** (all state rays). No external
  Neural Full Recompute is involved or claimed; ECRO reuses its parameters natively across rigid edits (geometry-conditioned
  generator). Path-traced images are the physical quality reference only.

## 1. Where the historical "~80 ms" came from

- WL86 Stage 4 (RTX 3080 Ti, LabServer63): ECRO per edit 76–78 ms wall = geometry 31 + state 21 + source 5 + operator 11 + table
  transfer 1.5 ms, **excluding** the frame. WL96A / WL97-E5 reproduced the same lifecycle on CK2 (edit → 512² × 64 spp frame
  146–149 ms; no-frame 87.7 ms) and measured that **69–81 % of the lifecycle is CPU-side overhead** (numpy geometry tables,
  host copies, Python / Dr.Jit re-tracing); GPU kernels were 19–31 %.

## 2. Reproduced baseline on this PC (development profile, selection case only; non-evidence)

`dev_profile101.py`, SSS checkpoint, WL100 test configs 0 ↔ 1 with their own lights, 512² × 1 spp, warm medians:

| part | wall ms | Dr.Jit GPU ms | cause of the wall time |
|---|---|---|---|
| geometry update (`set_geom`) | 12.7–18 | 0.04 | `scene86.Geom` rebuilds all 1,270 patches in a Python loop; constant sample-offset tables (4 × N × K floats, 10 MB) re-uploaded every edit |
| relation update (`update_state`) | 9–12 | 0.08 | segment table (15.6 MB) uploaded for the GPU dirty test, ids copied to the host, re-trace results copied to numpy, three 650 k-entry array copies, Python re-tracing |
| source encoding | 2.0–2.7 | 0.04 | per-light host reduction (`np.bincount`) |
| generator + apply (`respond`) | 3.8–4.5 | — (torch) | CPU casts of the state into a new GeoState, host copy of the table |
| table upload | 0.4–0.6 | — | host → device copy of the table just copied down |
| frame 512² × 1 spp | 25–35 | 1.0 | **20 ms Python re-tracing of the LOW + lookup program on every call** + 4.8 ms launch / float64 host accumulation |
| **edit → frame** | **55–74** | | |

The frame alone (no edit) costs ~25 ms, so the frozen stack cannot meet 20 ms even with a free update. Frame GPU time is 1.0 ms
(1 spp), 3.1 ms (4 spp), 11.8 ms (16 spp), 46 ms (64 spp) on the SSS scene.

**Bottleneck finding.** Every dominant cost is implementation-level: host round trips of data that already lives on the GPU,
re-uploading constants, Python loops over unchanged patches, and Dr.Jit 0.4.6 re-tracing each program on every call (it has no
kernel replay). The mathematically necessary per-edit work is small: BVH update of the moved boxes, table rows of the moved boxes,
re-trace of the dirty state rays (~90 k of 650 k), 20 k source shadow rays, one generator pass (its message passing has a global
receptive field and a global-mean term, so every patch's factors change after any edit; no selective factor update is exact), one
linear application and the frame.

## 3. Implementation (arm B, `rt101.py`)

- **Geometry:** `IncGeom` rewrites only the moved boxes' patch / rect rows (bit-identical to `scene86.Geom` + WL100 checker; tested
  indirectly through state equality); constant tables stay resident; BVH update by `params.update()` as before, only when a box moved.
- **Relation:** state rays resident on the GPU (hit, bin, segment). Dirty test on the GPU with the frozen rule (origin on a moved box,
  or segment through the old / new 1 mm-padded box); only dirty rays are re-traced and scattered in place. For kernel replay the dirty
  list is padded to a power-of-two capacity by repeating a dirty ray (rewrites the same value).
- **Source:** same samples and shadow rays; float64 reduction on the GPU.
- **Generator:** unchanged `model86` factors / apply on resident tensors; no CPU GeoState.
- **Hand-over:** the table is shared torch → Dr.Jit by DLPack (zero copy).
- **Frame:** the unchanged `pt100.Ctx.trace` LOW + lookup estimator; the image stays on the GPU.
- **Replay:** every Dr.Jit program (state rays, source, frame) is wrapped in `dr.freeze` (Dr.Jit ≥ 1.1): kernels are recorded once
  and replayed without Python re-tracing; geometry tables, scene, lights, table, seeds and the material table are explicit inputs so
  that replays see the current data. `FREEZE=0` runs the same code eagerly (attribution control).
- **Compatibility shims (no frozen source edited):** `dr.eq / dr.neq` → `== / !=`; `mi.load_dict(..., optimize=False)` (3.6+ merges
  same-BSDF meshes, which would break the shape ids); `pt100.Ctx.sss_walk` re-expressed with `dr.while_loop` (same statements, same
  RNG order; `mi.Loop` no longer exists); the 0.4.6 PCG32 seeding replicated (1.x seeds lanes differently, which made same-seed
  frames differ by 3 %–110 % before the fix).
- **Not done:** CUDA-graph capture of the generator (would save ≈ 0.8 ms of launch overhead) needs a mask-free rewrite of
  `model86.form_factors`; left out to keep the model code untouched.

Development checks on the selection case (non-evidence): selective = full rebuild exactly; source and table bit-identical to arm A
(one of 650,240 state rays differs by a cross-version hit tie in one state, table max rel 3.7e-5); same-seed frames agree to
2e-8 – 8e-6 mean relative difference; frozen replay = eager re-trace exactly. Dev timing: edit → 512² × 1 spp frame ≈ 7.4–8.4 ms.

## 4. Freeze and execution record

- Protocol `experiments/ecro_dyn101/protocol/dyn101.json` and all code frozen in commit `bf9c6b4` before any validation run.
  Tolerances were set from the selection-case observations only.
- Execution fix `8a31a80` (non-scientific): the first chain launch started while another project's WSL training held the GPU at
  83–99 %; its single completed output (SEL A_sel r1) is quarantined under `results/ecro_dyn101/evidence/_contended_invalid/` and not
  used. The chain now starts each step only after the GPU has stayed below 5 % for 60 s.

## Evidence (append below)
