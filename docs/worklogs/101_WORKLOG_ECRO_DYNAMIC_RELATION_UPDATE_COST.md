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

- **Amendment 1** (`protocol/amendment1.json`, commits `f503f73`, `de48c52`; execution only, before any validation evidence): the
  user moved execution to **LabServer69** ("로컬에서 진행시키지 말고, 69번 서버로 돌려"). Consequences:
  - Machine: RTX 3090 24 GB, driver 550.54.15, CentOS 8.4 (glibc 2.28). Workspace `~/rest_dyn101` (git archive of the commit;
    WL100 configs and the two checkpoints copied, sha256 equal). The PC numbers in sections 2–3 stay development-only.
  - Arm A: `~/NeuralShaderDemo/external/8dna26/.venv` (Mitsuba 3.5.1, Dr.Jit 0.4.6, **torch 2.3.1**+cu121).
  - Arm B: same venv + **Mitsuba 3.8.0 / Dr.Jit 1.3.1** (`--target external/mi_3.8.0`). Mitsuba 3.9.1 / Dr.Jit 1.5.0 aborts on this
    driver (its OptiX module uses `optix.ptx.copysign.f32`, rejected by the 550 OptiX compiler); 3.7.0, 3.7.1 and 3.8.0 pass a
    freeze + ray-tracing test, the newest one is used. Dr.Jit ≥ 1.3 needs `libatomic.so.1`, absent on the host: a copy from the
    user's conda gcc-11 toolchain is placed in the workspace and put on `LD_LIBRARY_PATH` (no system change, no root).
  - References: all tour states are rendered on the server (WL100's PC-rendered REF_0 is not reused, so no cross-machine pairing).
  - Server SEL smoke (selection workload, non-evidence): every correctness check passed (selective = full, revisits equal,
    cross-stack hit mismatch 1 / 650,240, table rel L1 2.9e-7, source identical, same-seed frame mean rel ≤ 6.7e-6, replay = eager);
    dynamic frame 512² × 1 spp median A 100.7 ms vs B 15.8 ms.
  - The full chain (timing, correctness, quality, scaling, analysis) runs as one detached `setsid` process (`launch69.sh`).

## Evidence (append below)

### Execution record on LabServer69 (2026-11-17)

- Chain `chain101.py` ran 14:43–16:03 KST (one `setsid` process). Execution-only interventions, all committed:
  `6ba31cb` idle gate 60 → 10 s (the user noted the server idling between ~5–25 s steps); `eb1fa4c` the chain had stopped at the
  first quality step because the quality output directory was not created before its log was opened (fixed; quality renders of
  the two scenes then ran concurrently without the idle gate; timing steps kept it). Completed timing outputs were skipped on
  restart; no timing step was repeated or selected. Every timing step started with the GPU idle and no foreign compute process.
- Outputs: server `~/rest_dyn101/results/ecro_dyn101/evidence/`; JSON and figures copied to `results/ecro_dyn101/evidence69/`
  (quality NPZ arrays stay on the server); reviewer artifacts `evaluation/101/` (manifest with sha256).

### Measurement — timing (LabServer69, RTX 3090, N = 1,270 patches, 650,240 state rays)

Dynamic frame = geometry edit → every update → frame complete. Medians over 2 processes × 24 events (V_SSS_ANIM 2 × 48);
p99 in brackets.

| workload | A_sel 512² × 1 | B_sel 512² × 1 | B_full 512² × 1 | B_eager 512² × 1 | A / B_sel | B_sel 512² × 4 | B_sel 1024² × 1 |
|---|---|---|---|---|---|---|---|
| SEL (selection) | 101.0 (104.1) | **15.65 (16.15)** | 12.62 | 27.4 | 6.45× | 23.6 | 20.9 |
| V_SSS_TOUR | 101.8 (103.8) | **15.69 (16.01)** | 12.62 | 27.1 | 6.48× | 23.6 | 20.9 |
| V_NEAR_TOUR | 84.4 (89.6) | **14.20 (14.44)** | 10.95 | 21.8 | 5.94× | 15.0 | 14.6 |
| V_SSS_ANIM (48 frames) | 100.4 (104.0) | **15.66 (16.15)** | 12.47 | 26.9 | 6.41× | 23.6 | 20.9 |

B_full at 512² × 4 / 1024² × 1: 20.4–20.6 / 17.7–17.9 ms (SSS), 11.7 / 11.4 ms (near). Edit without frame: A 52.3–54.2 ms,
B_sel 13.1–13.5 ms (3.9–4.1×).

Part medians (V_SSS_TOUR r1; ms wall / Dr.Jit GPU):

| part | A_sel | B_sel | B_full |
|---|---|---|---|
| geometry update | 23.75 / 0.05 | 2.93 | 2.78 |
| relation update | 15.38 / 0.07 | 3.77 / 0.09 (+ torch dirty test) | 0.98 / 0.15 |
| source | 3.84 / 0.06 | 0.97 / 0.11 | 0.96 / 0.08 |
| generator + apply (torch) | 8.03 | 5.26 (CUDA-event span 5.24) | 5.19 |
| table upload / hand-off | 0.84 | 0.07 | 0.07 |
| frame 512² × 1 | 46.7 | 2.56 / 2.04 | 2.51 / 2.03 |

- Static frame (no edit) at 512² × 1 / × 4 / 1024² × 1: A 46.6 / 53.0 / 65.2 ms (SSS) and 31.4 / 32.5 / 45.4 (near);
  B 2.44 / 10.49 / 7.69 (SSS) and 0.63 / 1.46 / 1.12 (near).
- Memory: B torch peak 235 MiB, device delta 1.57 GB (A 169 MiB / 1.50 GB); resident relation state 20.8 MB.
- Kernel recordings per process: state 3, source 1, each frame configuration 1.
- Dirty rays per edit (median): 85,571 (SSS tour), 103,356 (near tour), 81,152 (animation) of 650,240.

Scaling (V_SSS_TOUR; timing only, same checkpoint, quality not evaluated):

| N | A_sel | B_sel | B_full | B generator | B_sel relation | B_full relation |
|---|---|---|---|---|---|---|
| 1,270 | 99.2 | 15.7 | 12.6 | 5.2 | 3.8 | 1.0 |
| 2,802 | 144.2 | 22.7 | 17.7 | 9.5 | 6.1 | 1.3 |
| 5,080 | 212.3 | 34.3 | 26.2 | 17.2 | 9.7 | 1.6 |
| 11,430 | 406.0 | 69.4 | 52.5 | 41.2 | 19.6 | 2.9 |

Frame (2.6 ms) and source (1.0 ms) stay flat in N, and geometry update grows 2.9 → 4.6 ms. The generator grows about linearly
(exponent 0.94 over 9× N) and is 79 % of B_full at N = 11,430. B_full device memory grows 1.5 → 4.9 GB.

### Measurement — correctness (all predeclared tolerances PASS on every validation sequence and both processes)

| | SEL | V_SSS_TOUR | V_NEAR_TOUR | V_SSS_ANIM |
|---|---|---|---|---|
| selective = full rebuild (A and B, every event) | 24/24 | 24/24 | 24/24 | 48/48 |
| revisit digest = first visit | 24/24 | 20/20 | 20/20 | 35/35 |
| B vs A state mismatch (max fraction; tol 1e-4) | 1.5e-6 | 1.5e-6 | 1.5e-6 | 0 |
| B vs A table rel L1 / max rel (tol 1e-4 / 1e-2) | 2.9e-7 / 3.7e-5 | 2.9e-7 / 3.7e-5 | 1.1e-7 / 7.2e-6 | 0 / 0 |
| B vs A source max rel (tol 1e-5) | 0 | 0 | 0 | 0 |
| same-seed frame mean rel, 72 checks (tol 1e-4) | ≤ 6.7e-6 | ≤ 5.1e-6 | ≤ 1.2e-5 | ≤ 7.1e-6 |
| replay = eager re-trace | exact | exact | exact | exact |
| table changes after every dirty edit | yes | yes | yes | yes |

The one mismatching state ray (of 650,240, in one SSS state and one near state) is a cross-version hit tie; none occurs in the
animation.

### Measurement — quality (4096-spp predictions vs 2 × 4096-spp physical references rendered on LabServer69, first tour pass)

- A vs B images: relative L1 2.1e-7 – 3.0e-6 (total); tables ≤ 2.9e-7. **B reproduces A's rendered result.**
- Higher-order change qualification (predeclared): near-mirror tour **7 of 8** transitions qualify (R_h 0.10–0.28 vs noise
  0.05–0.06); SSS tour **0 of 8** (R_h 0.025–0.056 vs a conservative noise of 0.033: box moves change the SSS sphere's ≥ 3-event
  light by less than this criterion resolves).
- Learned higher-order error in the material ROI, Σ|hl − h_ref| / Σ h_ref, updated B vs STALE (previous state's table, LOW
  current): near 0.045–0.050 vs 0.048–0.061 (stale worse in 7 of 8 states, one tie); SSS 0.026–0.027 vs 0.026–0.046 (stale worse in
  7 of 8; in one state 0.026 vs 0.027). E_H in the higher-order change region: near 0.009–0.013 (B) vs 0.010–0.015 (stale);
  SSS 0.0035–0.0069 vs 0.0052–0.0127.
- ROI total rel RMSE in the near-mirror case is dominated by reference / prediction MC noise (reference ROI noise 0.08–0.17, heavy
  tails, as in WL100); SSS ROI 0.018–0.019 (B) vs reference noise 0.010.

## Final report

### Implementation fact

- Arm B keeps the ECRO v0 representation, the WL100 checkpoints, the physical LOW / learned higher-order ownership and the frame
  estimator; the frozen sources are imported unchanged. What changed: incremental box tables, GPU-resident relation state, GPU
  dirty test with the frozen rule and re-trace of dirty rays only, GPU source reduction, generator on resident tensors, DLPack
  hand-off, and recorded kernel replay (`dr.freeze`) of every Dr.Jit program on Mitsuba 3.8.0 / Dr.Jit 1.3.1.
- Every dynamic frame recomputes the BVH and tables of the moved boxes, the relation state, the source encoding and the physical
  LOW (recomputed physical information). It evaluates the generator and the linear application on the current state (newly
  evaluated neural information) and reuses the trained parameters unchanged (reused learned information). No retraining and no
  stale table.

### Observation

- The historical ~80 ms (WL86, without frame) and the ~100 ms dynamic frame of the frozen lifecycle here are dominated by
  host-side work; the GPU kernels of geometry, relation and source take < 0.2 ms in total in both arms.
- Replay is the largest single factor: the same restructured code re-traced eagerly costs 27 ms against 15.7 ms.
- At N = 1,270 a full state rebuild (0.98 ms) is cheaper than the selective update (3.8 ms): re-tracing all 650 k rays on the
  GPU costs less than the torch dirty test plus its count synchronisation. The selective update stays more expensive up to
  N = 11,430 (19.6 vs 2.9 ms).

### Interpretation

- **Original cost.** CPU-side implementation overhead: per-call Python / Dr.Jit re-tracing (frame 46.7 ms for ~2 ms of GPU),
  a numpy rebuild of every patch table (23.8 ms), host round trips of the relation state and the table, and per-edit re-upload of
  constants.
- **Eliminated or reduced.** Re-tracing (replay), host round trips (residency, DLPack), constant uploads, and the full numpy table
  rebuild (incremental rows). Edit → 512² × 1 spp frame is 6.4–6.5× faster (5.9× near); the frame alone 18×; the edit without
  frame 4×.
- **What remains and is necessary.** BVH update, one relation re-trace, source shadow rays, the frame, and a **full generator
  pass after every edit**. The generator's message passing has a global receptive field and a global-mean term, so all per-patch,
  per-bin factors change after any edit, and no exact selective factor update exists. The generator is the largest part at
  N = 1,270 (5.2 ms) and grows about linearly with N (41 ms at N = 11,430).
- **Remaining implementation-level overhead** (not removed here): the torch dirty test (a full rebuild is cheaper at every tested
  N), ~0.8 ms of generator launch overhead (CUDA graphs would need a mask-free `form_factors`), ~2 ms of host work in the
  geometry update.
- **Quality.** Preserved exactly: B reproduces A's tables and per-sample estimator, so the learned higher-order error is that of the
  WL100 checkpoint. The update matters: the stale table is worse in 14 of 16 states.

### Unresolved question

- The kernel-time split of the generator at large N (per-bin heads versus dense N × N form-factor message passing) was not measured
  on the server; whether a sparse message pass would change the slope is untested.
- No representative production workload was tested: one room family, two boxes, one light, N ≤ 11,430, 512² / 1024², ≤ 4 spp
  without denoising, RTX 3090 with a server CPU. The server is slower than the development PC (A ≈ 100 ms vs ≈ 58 ms on the 5080).
- The SSS tour produced no transition that qualifies as a material higher-order change under the predeclared criterion.

### Architecture verdict (scope above)

- **Learned higher-order transport retained across geometry-relation changes without retraining:** SUPPORTED, as exact equivalence
  to the existing ECRO lifecycle (the update is required: stale is worse), with the WL100 checkpoint's own fidelity.
- **Practical real-time latency:** at N = 1,270 and 512² × 1 spp the end-to-end dynamic frame is 11.0–15.7 ms: **below 20 ms on
  every workload, and below 15 ms with the full-rebuild variant** (SSS 12.5–12.6 ms, near 11.0 ms). 10 ms is not reached. SSS at
  512² × 4 spp (20.4–23.6 ms) misses 20 ms. **General real-time capability is not supported**: the SSS dynamic frame exceeds
  15 ms at N ≥ 2,802 and 20 ms at N ≥ 5,080.
- **Remaining limit: mainly architectural at scale.** Every edit re-evaluates the global, geometry-conditioned generator for all
  patches and bins (linear in N); the smaller implementation-level overheads are listed above.

### Reproducibility

- Code: freeze `bf9c6b4`; Amendment 1 `f503f73`, `de48c52`; execution fixes `8a31a80`, `6ba31cb`, `eb1fa4c`. Server workspace
  `~/rest_dyn101` (`DEPLOYED_COMMIT`, then the execution-fix files). Protocol `experiments/ecro_dyn101/protocol/dyn101.json` +
  `amendment1.json`.
- Checkpoints (WL100): near sha256 `b3550787…`, SSS `2d9cb580…`. Frozen-source hashes are in every timing JSON.
- Environment: LabServer69 RTX 3090, driver 550.54.15. Arm A: Mitsuba 3.5.1 / Dr.Jit 0.4.6 / torch 2.3.1+cu121. Arm B:
  Mitsuba 3.8.0 / Dr.Jit 1.3.1 + the same torch, `libatomic.so.1` from the workspace.
- Seeds: frame timing 9000 + event; same-seed checks 7000 + event; predictions 101500 + 10 × state; references 101100 / 101101 +
  10 × state.
- Re-run: `bash experiments/ecro_dyn101/launch69.sh` in the workspace (resumable; the last step runs `analyze101.py`).
- Artifacts: [evaluation/101/index.html](../../evaluation/101/index.html), `analysis101.json`, `manifest101.json`.
