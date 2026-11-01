# Worklog 69 — Common Workload and External Baseline: Feasibility Gate and Direct Comparison on the WL55 Floor Fixture (2026-11-01 – 2026-11-02)

**Authorisation:** the user's 2026-11-01 brief "TRACK B — Common Workload / External Baseline: Independent Benchmark Feasibility Experiment", run in parallel with Track A (WL68). Branch `Selective_Recompute`; no branch or worktree. Track B's paths (`experiments/xbench69`, `results/xbench69`, `results/evaluation/69`, server root `~/rest_xbench69`) do not overlap Track A's, and no Track A result was read or used. The user approved LabServer69 for this batch ("그럼 69번 써."), as an exception to the LabServer63-main rule.

**Protocol:** `experiments/xbench69/protocol/xb69.json` (`xbench69_v1`).
- Preregistered at `390f267` before any comparative outcome.
- Amendment 1 (`e4c0870`) holds the Stage-3 gate results and the frozen Stage-4 settings. It was committed before any Stage-4 output.
- Amendment 2 (`32ded46`) is **post hoc and labelled**: an analysis read defect, found after the first `analysis.json` had been read. It is described below.

## Completion answer

> Can the currently implemented ReST-GI and at least one external Neural GI baseline execute the same physically meaningful geometry-edit workload without a new architecture? If yes, is there a measured quality/cost advantage?

**Yes, on one workload, and that workload passes the gate: C1 × 8DNA.**
- **C1** is the WL55 floor fixture: a 10×10×1 cm homogeneous medium slab above a Lambert floor, rigidly moved from 1 cm to 4 mm clearance.
- **8DNA** is the released code, unchanged, with one slab asset trained natively and moved by its instance transform without retraining.
- **Excluded pairs:** RNA is blocked on this workload by its input semantics at the transparent medium boundary. Neural Radiosity has no medium transport. TransGI has no released code.

**Outcome B** under the predeclared verdict map: the workload is established and executed, but no ReST-GI superiority meets class A. The details:

- **ReST-GI against 8DNA:**
  - ReST-GI CAK (WL55 interface, rest oracle K reused) meets the complete 3 % contract on both states: **12/12 and 12/12**, max |group bias| 1.37 % / 0.76 %.
  - 8DNA fails it at every spp for both training seeds: **3–7 / 12**, max 15–20 %. This is mostly a blue-channel deficit on the medium faces (−14 … −20 %) plus −3 … −7 % on the in-gap floor. The deficit is the same before and after the edit.
  - 8DNA is therefore INELIGIBLE for a superiority claim, and never reaches ReST-GI's quality at any tested spp.
- **ReST-GI against Full Recompute — this is what prevents A:**
  - Quality-matched Full Recompute (Cycles volume path tracing with the reference renderer) **dominates** ReST-GI.
  - Cycles reaches every ReST-GI quality metric (M1–M4) at **16 spp, in 2.3–2.4 s** for the three ruled T1 views. It already meets the contract at **1 spp (1.2 s)**.
  - The ReST-GI T1 update takes **24.0 s**, of which 21.7 s is the Monte Carlo sun term. The other parts take 1.0 s (aggregation) and 1.2 s (C build, solve and readout).

## Implementation Fact

### Stage 1 — capability matrix (source and execution evidence)

Status vocabulary: EXECUTED (run in this project), SUPPORTED-UNQUALIFIED, UNSUPPORTED, UNKNOWN; BLOCKED marks a code path that is unavailable or fails a gate.

| Arm | Homogeneous medium | Rigid edit without retraining | Stationary receiver / in-gap views | Collimated sun | Status on C1 |
|---|---|---|---|---|---|
| ReST-GI (WL55 oracle K + explicit C, CAK / P10) | EXECUTED (WL53/55) | EXECUTED (rest K reused; re-aggregation + C rebuild + solve) | EXECUTED (WL55 face-on views) | EXECUTED | EXECUTED here. No surface/glossy K exists; the learned WL59 K has never run in a coupled solve |
| 8DNA (released code, `8dna26`) | SUPPORTED natively (homogeneous volumetric assets); EXECUTED here | EXECUTED here (asset instance `to_world`); non-rigid needs retraining | EXECUTED (Mitsuba renders any view; exterior is path traced) | EXECUTED (directional emitter) | EXECUTED (gate passed) |
| RNA (`66b5b098`, custom Cycles wheel) | BLOCKED: X4a, the position AOV accumulates over transparent-boundary crossings | UNSUPPORTED (edit = refit / retrain, WL22 definition) | Training views on the exterior upper hemisphere only (WL67) | SUPPORTED (distant sun) | BLOCKED (G3/G6; G10 not met on LabServer69) |
| Neural Radiosity | UNSUPPORTED (surface residual only) | UNSUPPORTED (no dynamic conditioning) | — | — | UNSUPPORTED |
| TransGI | UNKNOWN (no code) | paper: rigid motion only | — | — | BLOCKED (no released implementation, 2026-11-01 search) |
| Full Recompute (Cycles 3.5.1 CUDA volume PT, reference settings) | EXECUTED | EXECUTED | EXECUTED | EXECUTED | EXECUTED |

### Stage 2 — candidate workloads

**C1 floor rigid**
- Scene: σt 1000 m⁻¹, albedo 0.999/0.99/0.9, isotropic, index-matched null boundary, Lambert floor ρ 0.8.
- Light: sun E = 1 at zenith 40°, azimuth 20°.
- Edit: T0 floor → T1 floor_c4.
- Cameras: top, underup and underdown are ruled; otop is reported only.
- Reference: WL55 Cycles references, 1024 spp × 2 seeds per state.
- C1 is the only candidate that a priori meets every brief requirement.

The other candidates were assessed from inherited evidence only, and none was carried to Stage 4:

| Candidate | Inherited evidence |
|---|---|
| C2 (W1 two-slab) | WL67 blockers |
| C3 (Lego + box) | K44 vs RNA comparison; path-history-dependent Burley reference; glass-transmission gap |
| C4 (teaset) | No ReST-GI K |
| C5 (floor_lam) | Only analytic Lambert K; ReST-GI would reduce to radiosity |

### Stage 3 — gate (G1–G10) and checks

**G8 signal.** The T0→T1 reference change on the stationary receiver exceeds the contract. In the WL55 references, the underdown floor group changes by 3.2 / 7.7 / 11.2 %; the STALE control measures this. The in-gap slab underside changes by 12–70 %.

**X1 — 8DNA renderer physics: PASS.**
- Mitsuba 3.5.1 volpath was run at 8DNA's native unit convention: asset centred, diagonal 2.0 units, s = 14.1069 units/m.
- It passes 12/12 ruled groups, max 0.85 %.
- At metres it shows the WL53 boundary-layer bias (−8.24 %), so the check discriminates.

**X2 — environment: PASS.**
- Toolchain: user-space conda-forge GCC 11.4 (glibc 2.17 sysroot) and ninja, installed with micromamba on LabServer63 and copied to the same path on LabServer69. No root was used, and LabServer69 has no outbound network.
- The unchanged 8DNA CUDA extension builds.
- The released candle renders at the demo settings (max_depth 10).
- At 128² / max_depth 64 the released code crashed on an empty batch (`cuvrqs_sample` reshape of 0 lanes).

**X3 — asset semantics and cost: PASS.** Training samples cover the whole boundary and the full hemisphere. One epoch takes 371 s, which projects to about 3.1 h per seed.

**X4a — RNA AOV semantics: FAIL.**
- Only 91.8 / 91.0 / 94.9 % of slab pixels have a position within 1 mm of the analytic hit; the predeclared threshold is ≥ 99 %.
- The failing pixels sit 1–4 px from the silhouette, and their position z (27–30 mm) lies above the slab top (20 mm).
- Cause: Cycles accumulates shader AOVs at every primary transparent hit.

**X5 — RNA cost.**
- Rendering projects to about 34 h of CPU time on LabServer69 (85–205 s per 1024² view), before HQ training.
- This exceeds G10 (36 h).

**Pair verdicts:**
- C1 × 8DNA: PASS.
- C1 × RNA: BLOCKED.
- C1 × Neural Radiosity: UNSUPPORTED.
- C1 × TransGI: BLOCKED.

### Stage 4 — arms (frozen in amendment 1)

- **CY:** `k55_cycles` unchanged, official Blender 3.5.1 CUDA. spp ladder {1, 4, 16, 64, 256, 1024} × 2 realizations. Seeds 690000 + 1000·state + 10·level + r.
- **RG_CAK / RG_P10:** WL55 CAK and the 1 cm P10 control, with both rest K seeds.
  - For T1 the rest K is reused (rigid).
  - Predictions equal WL55's stored arrays bit for bit (max relative deviation 0.0).
  - Components are timed.
- **STALE:** T0 reference ports used as the T1 prediction.
- **D8:** two training seeds (0, 1) with `configs/default.yaml`, unchanged except `scene=xbslab`.
  - Training: max_epochs 30, `final.ckpt`.
  - Rendering: `neuralvolpath` with max_depth 10, rr_depth 5, LoopRecord and VCallRecord off.
  - One integrator per state, because `prepare_scene` caches the asset (WL21 audit). This was fixed before any scored output.
  - spp is accumulated in 16-spp chunks with seeds 7,000,000 + 100,000·state + 1000·level + 500·r + k.
- **Scoring:** the WL55 amended unit and the unchanged `group_bias`; |bias| ≤ 3 % + 3 SE.
  - M1: passes.
  - M2: max |bias|.
  - M3: per-port relative L2.
  - M4: stationary-floor signed change (top + underdown).
  - QM: the lowest spp at which both realizations reach the arm on all of M1–M4.

### Execution, deviations and corrections

**Machine.** Everything ran on LabServer69 (RTX 3090 24 GB), with the user's approval.
- The WL55 references, operators and environment live only on LabServer69.
- LabServer63's Mitsuba cuda film accumulation is wrong.
- The RNA checks X4/X5 were also moved to LabServer69, because Track A was running on LabServer63.

**Chain.** One detached `setsid` chain, `experiments/xbench69/chain.sh`, ran from 2026-11-01 18:40 to 2026-11-02 01:11 KST.
- Before every stage it gated on an idle GPU (three empty checks).
- A 15 s sampler logged every GPU process: 1,567 samples, never more than one process. Every PID belongs to this chain (rg, 5 Blender, 2 trainings, 2 evaluations, analysis). The runs were uncontended.

**Native 8DNA failures, recorded and not retried.**
- The released code raised the X2 empty-batch error on 6/96 renders (seed 0) and 9/96 (seed 1). All were on in-gap cameras, mostly at 1024 spp.
- An spp level is scored only when all views of both states exist for a realization. So D8 at 1024 spp is unscored for both seeds, and some seed-1 levels have a single realization.
- 8DNA fails the contract at every scored level for both seeds, so the missing levels cannot change the outcome.

**Amendment 2 (post hoc, labelled).** `x_analyze.py` read the ReST-GI prediction as the `ball` array alone.
- WL55's ballistic readout is `nee + ball` (`k55_analyze.load_arm`), and `ball` is only the small unscattered add-on.
- The first analysis therefore scored RG as zero predictions: −100 % in every group. It is kept as `results/xbench69/stage4/analysis_defective_a1.json`.
- The fix reads `nee + ball`. A focused test, `test_rg_readout_is_wl55_nee_plus_ball`, checks it against WL55's stored arrays.
- The analysis was re-run on the unchanged stored outputs. CY, D8 and STALE scores are unaffected; they had been seen before the fix.

**Tests.** 10/10 focused tests pass on LabServer69 with a plain runner, because pytest is not installed there:
- similarity convention;
- fixture identity;
- rigid T1;
- floor and light identity;
- medium parameters;
- reference manifest;
- seed disjointness;
- 8DNA code identity (sha256 of `train.py`, symlinked models/utils/configs);
- no reference access by predictors;
- RG readout.

**Precheck disclosure.** An exploratory signal read of the exterior views happened before the protocol (P2; `results/xbench69/precheck/`). The gate criteria are generic and were not tuned to it.

## Measurement

All results are on LabServer69 (RTX 3090), measured uncontended. Biases are % against the WL55 Cycles references. M3 and M4 are per channel (R / G / B).

### Quality

| Arm | T0 M1 | T1 M1 | max \|bias\| T0 / T1 | T1 M3 | M4 change error | M4 slope | M4 corr |
|---|---|---|---|---|---|---|---|
| RG_CAK (mean of 2 K seeds) | 12/12 | 12/12 | 1.37 / 0.76 | 0.023 / 0.026 / 0.033 | 0.205 / 0.200 / 0.205 | 1.02 / 1.05 / 1.06 | 0.980 / 0.982 / 0.983 |
| RG_P10 (1 cm ports) | 10/12 | 6/12 | 4.33 / 19.44 | 0.102 / 0.131 / 0.171 | 0.297 / 0.283 / 0.217 | 1.01 / 0.99 / 1.03 | 0.961 / 0.962 / 0.980 |
| STALE | — | 6/12 | — / 70.2 | 0.165 / 0.266 / 0.507 | 1 (no change) | 0 | — |
| CY 1 spp (r0 / r1) | 12/12 | 12/12 | ≤ 1.02 / ≤ 2.36 | 0.090 / 0.078 / 0.073 | 0.67 / 0.48 / 0.24 | 0.89–1.06 | — |
| CY 16 spp | 12/12 | 12/12 | ≤ 0.53 / ≤ 0.42 | 0.022 / 0.018 / 0.020 | 0.165 / 0.117 / 0.057 | 0.99–1.00 | — |
| CY 64 spp | 12/12 | 12/12 | ≤ 0.17 / ≤ 0.16 | 0.011 / 0.009 / 0.010 | 0.076 / 0.053 / 0.029 | 0.99–1.00 | ≥ 0.996 |
| D8 seed 0, 256 spp (r0 / r1) | 6/12 | 6/12 | 17.8–17.9 / 15.8 | — | 0.24 / 0.18 / 0.21 | 1.07 / 1.02 / 0.99 | 0.98 / 0.98 / 0.98 |
| D8 seed 1, 256 spp (r1) | 4/12 | 3/12 | 20.0 / 17.1 | — | 0.25 / 0.18 / 0.21 | 1.07 / 1.02 / 0.98 | 0.98 / 0.98 / 0.98 |
| D8, every scored spp (1–256) | 6–7 (s0), 3–4 (s1) | 6–7 (s0), 3–4 (s1) | 15–20 % | — | improves with spp | — | — |

**Where 8DNA fails** (seed 0, 256 spp; per-group biases in `analysis.json`):
- Blue channel on the medium faces: top dom:top −15.8 %, underup dom:bot −13.6 … −17.8 %.
- In-gap floor (underdown): −3.3 … −6.8 %.
- Seed 1 is 2–3 points lower on red and green.
- The top-view floor (−0.2 … −0.4 %) and otop floor pass.

**ReST-GI's only out-of-rule group** is otop dom:+y (+2.1 … +3.6 % at T1), which is reported but not ruled.

**Quality matching (QM):**
- CY reaches the contract at 1 spp.
- CY matches RG_CAK and RG_P10 on all of M1–M4 at 16 spp.
- D8 never reaches the contract or the quality of either RG arm at any scored spp. D8 has no QM.

### Cost

Latency is the time from the edit to output on the three ruled T1 views.

| Arm | T1 latency | Breakdown |
|---|---|---|
| RG_CAK | **24.0 / 24.1 s** | sun term 21.7–21.9 s (Monte Carlo first-order term, WL55 seed); levels 0.01 s; aggregation 1.0 s; compose 1.2 s (C 0.04 s, solve 0.08 s / 51 iterations, readout); K load 4.1–4.2 s, resident and not counted |
| RG_P10 | 23.0 / 23.2 s | — |
| CY | 1.2 s (1 spp), 1.8 s (4), 2.3–2.4 s (16), 4.0–4.1 s (64), 10.9–11.1 s (256), 40.0–40.1 s (1024) | Per-view Cycles render call including session sync; Blender start and Python scene construction not timed |
| D8 | 0.98 s update + 0.82 s (1 spp), 5.1 s (16), 81 s (256) | Update = scene load 0.93–0.95 s + integrator construction 0.03 s; 1024 spp: 218 s for two views |

**Offline cost, persistent state and peak GPU memory:**

| Arm | Offline cost | Persistent state | Peak GPU memory |
|---|---|---|---|
| ReST-GI | Rest oracle K: about 380 s per configuration (WL55 measurement, not repeated) | 2.04 GB per K seed (`slab_L4_s1.pt`) | 6.8 GB torch / 8.0 GB nvidia-smi |
| 8DNA | Training 2 h 37 min per seed, 30 epochs × 8,192 steps, including dataset build; final train NLL −1.56 / −1.57 | 8.1 MB per asset | Training 16.5 GB (nvidia-smi); evaluation 5.0 GB torch / 19.6 GB nvidia-smi |
| CY | None | — | — |

## Observation

- **The edit signal is large where the workload is hardest.** The stale T1 prediction fails 6/12, by up to 70 % on the in-gap slab underside (blue).
- **CAK and P10 differ on the edit.** ReST-GI's CAK interface tracks the edit within the contract. The 1 cm P10 control does not (T1 6/12, up to 19 %), as in WL54/55.
- **8DNA's deficit is static and structured.**
  - It is the same before and after the edit.
  - In the block-mean blue error of the underup view (figure 02), it concentrates at the slab edges.
  - Its signed-change prediction on the stationary floor at 256 spp (error 0.18–0.25, slope 0.98–1.07, corr 0.98) is of the same order as RG_CAK's (0.20–0.21).
- **Cycles noise averages out quickly at port-group level.** Group biases reach the 3 % contract at 1 spp. Per-port M3 and M4 need 16 spp to match RG_CAK, and at 64 spp CY's M4 error is 2.7–7× lower than RG_CAK's.

## Interpretation

- **RG against 8DNA.** RG_CAK beats 8DNA on physical fidelity at every comparable point. This is not a cost-matched comparison: 8DNA never reaches RG's quality, so no quality-matched 8DNA cost exists.
- **Where 8DNA's error comes from.** X1 shows that Mitsuba volpath under the same unit convention and scene passes 12/12. So most of 8DNA's 15–20 % deficit is attributable to the learned asset, not to the renderer. Two factors were not isolated:
  - **Render depth.** The evaluation used the demo max_depth 10, because depth 64 crashes in the released code. On the directly lit slab top (sun → first free flight → one neural step → camera), a 10-vertex cut-off is not a plausible cause of −16 % in blue.
  - **Training budget.** Only the native 30-epoch schedule was run.
- **Why A is not reached.** ReST-GI is dominated on this workload by quality-matched Full Recompute: about 10× slower at matched quality (24 s vs 2.4 s), and about 20× slower than the cheapest point that meets the contract.
- **Where the ReST-GI update time goes.** 90 % is the Monte Carlo sun term. The K/C part alone (aggregation + compose, 2.2 s) is comparable to CY at 16 spp. It does not beat CY at 1–4 spp, which already meets the contract.
- **Scope.** Small fixture, one rigid edit, one machine. The result does not generalise to large scenes, other materials or non-rigid edits.

## Unresolved Question

- Whether 8DNA's blue/edge deficit comes from:
  - asset capacity;
  - the training sample distribution at the null boundary;
  - the 30-epoch schedule;
  - the demo render depth.

  None was varied, and the brief forbids tuning.
- Whether the empty-batch crash (`cuvrqs_sample`) is purely a robustness defect, or interacts with sparse deep paths in the in-gap views. It only appears there.
- How ReST-GI's cost against Full Recompute scales with scene size, view count or edit locality. One small fixture was measured.
- Whether any C2–C5 blocker can be removed without a new architecture. RNA would need a medium input/AOV semantics change and about 17–34 h of data; teaset/Lego would need a K for surfaces/BSSRDF. These are decisions, not measurements made here.

## Architecture Verdict

**Outcome B** (predeclared map).
- The common workload is physically aligned and was executed with an external neural baseline (8DNA), ReST-GI, Full Recompute and stale/no-update controls.
- ReST-GI CAK meets the contract where 8DNA (INELIGIBLE) does not.
- Quality-matched Full Recompute dominates ReST-GI in latency, so no class-A superiority is measured.

No architecture, K or baseline role changes. No successor is authorised.

## Reproducibility

- **Code and protocol:** `experiments/xbench69/`, at commits `390f267` (protocol), `e4c0870` (amendment 1, frozen settings) and `32ded46` (amendment 2 + figures). Reviewer-figure edits come after that.
- **Server:** LabServer69, `~/rest_xbench69`.
  - `repo/` is a `git -c core.autocrlf=false archive` of HEAD.
  - `env_8dna.sh` sets the toolchain/CUDA path and the 8dna26 venv.
  - `8dna_work/` holds the unchanged `train.py` copy (sha256 2833e42e…) with symlinked models/utils/configs.
  - Run `bash repo/experiments/xbench69/chain.sh`. It is resumable through `launcher/*.done`.
- **Inputs:** WL55 references at `~/rest_wl55/out/cycles_cuda` (floor 5501/5502, floor_c4 5531/5532) and operators at `~/rest_wl55/ops/slab_L4_s{1,2}.pt`; all read-only.
- **Outputs in the repo:**
  - Gate: `results/xbench69/gate/`.
  - Precheck: `results/xbench69/precheck/`.
  - Stage 4: `results/xbench69/stage4/` — `analysis.json` (sha256 b1e0ffa7…), `analysis_defective_a1.json`, `rg.json`, `d8_eval_s{0,1}.json`, the CY render manifest, `steps.log` and `gpu_sampler.log`.
  - Figures: `results/evaluation/69/` (manifest with sha256).
- **Left on the server:** large arrays (CY/D8 images, RG npz, checkpoints).
