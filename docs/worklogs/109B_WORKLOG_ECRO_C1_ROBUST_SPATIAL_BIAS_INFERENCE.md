# Worklog — Robust Spatial Bias Inference and Frozen ECRO C1 Dynamic Closure (2026-11-23)

**Batch class:** suffix batch of WL109 (owner brief, 2026-11-23). It closes the evidence gap left by WL109 and WL109A. It
adds no estimator, denoiser or architecture.

**Unchanged verdicts.** WL109 stays **NOT_SUPPORTED**. WL109A stays **INCONCLUSIVE**. Their data, code, protocols and
worklogs are not edited.

**Question.** Can frozen ECRO C1 (M = 1) be shown, with a calibrated spatial-bias inference procedure, to preserve the
expected image within the original tolerance and to reduce TOTAL MSE at equal GPU cost, including after the frozen P7
geometry edit?

**What may change.** Only the confidence procedure for Q, its calibration, independent evidence collection, statistical
reporting and dynamic-closure instrumentation.

**What stays the same:**
- Q estimand: mean_{ROI, RGB} μ_D² / mean Var(TOTAL_R0), with D = H_C1 − H_R0.
- Limits: 0.01, and 0.02 for silhouette.
- B1: ±0.5 %.
- C1, M, table, checkpoint, renderer, ROIs and every WL109 gate.

## Stage B — development and the chosen procedure (before any WL109B evidence)

### Development sources (not qualification evidence)

- WL109A synthetic generators with fresh development seeds (77,000,001+): `experiments/ecro_estimator109b/dev109b.py`.
- Raw small-ROI pixels from WL109A's independent R0 null pool.
- WL109A's reviewer JSON.

These sources shaped the choice of procedure. None of them is used to qualify it.

### Why the WL109 bound fails

WL109 computes Q_g = mean(D̄_g² − s_g²/n) per group and puts a t interval on 8 groups.
- D̄_g² is chi-square-like and skewed to the right.
- When a realization happens to be low, the estimated SE is low as well, so the upper bound under-covers.
- With more replicates per group, s_g²/n becomes nearly constant and no longer offsets the skew. This explains why WL109A's
  heteroscedastic under-coverage got worse at larger R.
- With strongly heteroscedastic noise, a few pixels carry the whole statistic. That raises the skew further.

### Development round 1: WL109 Q compared with XJ

XJ is the cross-product U-statistic over distinct groups with a delete-one-group jackknife. 600 realizations per cell.

| Family | WL109 Q: worst Q = 0 non-coverage | XJ: worst Q = 0 non-coverage |
|---|---|---|
| heteroscedastic 1.5 | 0.100 | ≤ 0.043 |
| any family | — | ≤ 0.043 |

- XJ drops the diagonal term, and that removes the Q = 0 under-coverage.
- XJ still under-covers when a bias is injected along the noise (heteroscedastic noise with a noise-proportional bias):
  0.07–0.11.
- Cause: the bias sits on the few highest-variance elements. The jackknife SE then scales with |μ + ē| and collapses
  together with the estimate. This is the usual weakness of a Wald interval for a squared mean.

### Real heteroscedasticity (WL109A null pool, development)

- Per-pixel log-SD standard deviation in the material-boundary and silhouette ROIs is **0.04–0.45**.
- The synthetic stress families use 1.0 and 1.5. At those values one or two pixels effectively carry all the variance.
- Both stress families are nevertheless **kept as hard gates**. They are the families that sank WL109A.

### Development round 2: XJL

- **Change from XJ.** Add the worst-case variance of the linear term, b·q with b = 4λ/(R·X·scale), and invert the bound in
  q.
- **Coverage.** One-sided non-coverage is 0.000–0.002 in every family, shape, Q_true, pattern and design.
- **Cost.** The bound is conservative. In the stress families, with Var(D)/scale = 2, Q = 0 often fails the limit and
  synthetic power is low.
- **Real R0 null noise at R = 512.** Q = 0 fail rate 0; power at 2 × limit 1.0.
- **Bonferroni-level power on real noise, by G:**

| G | Power |
|---|---|
| 8 | 0.71–0.73 |
| 16 | 0.997 |
| 32 | 1.000 |

### Chosen procedure (frozen): XJL with G = 32

**Grouping**
- The R paired frames, in seed order, form R/16 subgroups of 16 pairs.
- These are merged into G = 32 contiguous groups.
- X is the number of ROI pixels × 3 RGB channels.

**Estimator**
- U = Σ_{g≠h} ⟨D̄_g, D̄_h⟩ / (G(G−1)·X·scale).
- U is unbiased for Q because distinct groups are independent.
- No clipping is applied, so U may be negative.

**Variance**
- V(q) = se²_jack + b·q.
- se²_jack is the delete-one-group jackknife variance of U.
- b = 4λ/(R·X·scale). λ is 16 × the largest eigenvalue of the sample covariance of the subgroup means over the ROI elements.
- b·q bounds the variance of the linear term 2⟨μ, ē⟩/(X·scale) for every bias field μ with mean square q·scale. The bound
  also holds under spatially correlated noise, because it uses λ_max.
- λ is biased upward for wide ROIs, which is conservative.

**Bounds**
- Upper bound: sup{q ≥ 0 : U ≥ q − t₀.₉₇₅·√V(q)}.
- Lower bound for the BIASED claim: inf{q ≥ 0 : U ≤ q + t·√V(q)}, with t at Bonferroni family α = 0.05/84.
- Both use Student t with G − 1 degrees of freedom and have closed-form roots.

**Multiplicity**
- EQUIVALENT requires every one-sided 97.5 % upper bound to be at or below its limit. This is an intersection-union test, so
  a false overall equivalence claim has probability at most 2.5 % without adjustment.
- Per-ROI upper bounds are per-ROI claims, not simultaneous intervals.
- BIASED is family-wise 5 % (Bonferroni).

**Assumptions**
- Replicates are independent across seeds.
- Groups are identically distributed within a condition.
- Nothing is assumed about pixel independence, spatial correlation, heteroscedasticity or symmetry of D.

**Expected failure modes**
- Very few groups combined with strongly skewed group means.
- λ underestimated when a rare tail never appears in the R pairs.
- Conservatism, which lowers precision rather than coverage.

The original WL109 Q (G = 8, t₇) is still reported, for historical comparison only.

## Frozen protocol summary

Protocol: `experiments/ecro_estimator109b/protocol109b.json`.

### Stage A
- Check the evaluation/109A hashes.
- Rerun the unchanged `calib109a` pool and hist. Every leaf must be exact.
- Compute the true median and 99th percentile of the per-draw upper bound (WL109A's label defect).
- Reproduce the WL109-design null fail rates.
- STOP on any failure.

### Stage C: sealed qualification

**Synthetic families**
- Families: gauss_iid, mc_like_4, mc_like_32, asym_mc4, heteroscedastic 1.0 and 1.5, het_mc4, correlated.
- Six real ROI shapes; R = 512 and R = 1024.
- 2,000 realizations per cell, with new seeds.
- Injections: Q_true ∈ {0, 0.5, 1, 2} × limit, with four patterns.
- Each entry's non-coverage is classified by its 95 % Clopper–Pearson interval:

| Class | Condition |
|---|---|
| PASS | CP upper ≤ 0.05 |
| FAIL | CP lower > 0.05 |
| BORDERLINE | otherwise |

- If no design qualifies and any entry is BORDERLINE, every cell is escalated uniformly by 6,000 more realizations.
- After escalation, an entry passes if its pooled rate is ≤ 0.05. There is no second escalation.

**GPU null pool**
- New R0 frames: 4,096 per view × spp, stored as 128 blocks of 16 pairs.
- 400 random block re-groupings per design.

**Gates on the null pool**

| Gate | Requirement |
|---|---|
| Coverage | Pooled non-coverage ≤ 0.05 for every Q_true and pattern |
| Precision | Per-setting P(upper > limit \| Q = 0) ≤ 0.01; product over the 84 settings of P(upper ≤ limit) ≥ 0.80 |
| Power | Pooled P(lower_family > limit \| 2 × limit) ≥ 0.80 for every pattern |
| Direct null | No BIASED state in two disjoint R0-versus-R0 decisions |

**Budget rule**
- R = 512 if every gate passes. Otherwise R = 1024 if every gate passes there. Otherwise **STOP INCONCLUSIVE**, and no C1
  frame is rendered.

### Stage D
- Paired R0/C1 at R for all six views × {4, 16, 32} spp, from new seeds.
- Bias decision:
  - any BIASED → Stage E;
  - otherwise any UNRESOLVED → INCONCLUSIVE;
  - otherwise → Stage F.

### Stage E
- Re-capture the flagged units from the replication seed family.
- If the flagged condition persists → BIAS_SUPPORTED, followed by the P versus P′ diagnostic, with its code committed first.
- If it does not persist → INCONCLUSIVE.

### Stage F
- New R0 grid and timing.
- WL109 rules: reference precision, grid bracketing, P2 ≤ 0.85, P3 upper bound < 1.0, P5 and P6.
- Mapping: INCONCLUSIVE, NOT_SUPPORTED (practical utility) or STATIC_PASS.

### Stage G
- P7 edit. D1–D3, D5 and D6 unchanged; D4 uses XJL.
- Mapping as in the brief.

### Seeds

| Family | Base |
|---|---|
| null | 31 M |
| paired | 32 M |
| grid | 33 M |
| timing | 33.6 M |
| dynamic | 34 M |
| dynamic reference | 34.95 M |
| mechanism | 35 M |
| replication | 38 M |
| smoke offset | +70 M |

- Every family is disjoint from WL101–WL109A and from the development seeds. `tests109b.py` checks this.
- A rebase of the replication family from 32.7 M to 38 M was found by this test, before anything was committed.

### GPU ownership

- Every GPU unit runs under the unchanged WL109 guard.
- The guard waits on the CPU, without a GPU library loaded, until no tmux session exists and the GPU has had no compute
  process for 120 s.
- When a foreign compute process appears, the guard kills only its own child's process group.
- Units are atomic and are skipped on restart.
- CPU stages run with `CUDA_VISIBLE_DEVICES=`.

## Execution record

(appended below as the chain runs)

### 2026-11-23 — chain start and smoke

- The chain `run109b.sh evidence cc8fb24` started on LabServer69 at 18:34:06 (+09:00).
  - Server output: `results/ecro_estimator109b/evidence/`.
  - Protocol sha256: `ba1f19a6…`.
- `tests109b.py` passed.
- Two CPU jobs started in the background: Stage A (WL109A reproduction) and synthetic stage 1.
- The guard waited for a short foreign GPU process to end. It then admitted each smoke unit:

| Smoke unit | Duration (s) |
|---|---|
| null | 160 |
| paired | 100 |
| grid | 41 |
| timing | 43 |
| dynamic | 37 |

- Every code path ran to the end:
  - smoke synthetic, pool and decide (outcome `SMOKE`; the frozen rule would have asked for escalation on the 3-realization
    smoke cells);
  - the bias phase and the static phase, which the smoke run forces through;
  - dynamic D1–D6.
- All smoke numbers are smoke data, not evidence. They use R = 64 and G = 4.
- `SMOKE PASS` at 18:52:37. The independent R0 null pool (4,096 frames per view × spp) started under the guard.

### 2026-11-23 / 24 — null pool, Stage A, Stage C qualification, paired capture in progress

**Null pool.** 4,096 R0 frames for each of the 18 view × spp settings. One attempt with no preemption, 18:54–20:34
(6,012 s).

**Stage A: PASS.**
- Artifact hashes match.
- The unchanged `calib109a` pool and hist reruns reproduce every leaf: 0 mismatches each. The budget decision is equal.
- The label defect is confirmed: in every setting, the archived `upper_median` equals the mean of the per-draw upper bounds.
  The true quantiles are in `stageA/hist_upper_quantiles.json`.
- The WL109-design null fail rates for H3/H4 material boundary reproduce.

**Stage C qualification: PROCEED with R = 512.**
- Synthetic stage 1 took 3,614 s. The real-null pool analysis ran 20:34 → 06:27 on the CPU.
- No escalation was needed.

| Gate | R = 512 | R = 1024 |
|---|---|---|
| Synthetic entries | 624 / 624 PASS (0 FAIL, 0 BORDERLINE); worst one-sided non-coverage 0.000 | 624 / 624 PASS; worst 0.0005 |
| Real-null pooled non-coverage | 0.000 for every Q_true and pattern | 0.000 |
| Precision | max per-setting fail 0.000; joint pass probability 1.000 | max per-setting fail 0.000 |
| Pooled Bonferroni power at 2 × limit | ≥ 0.99997 for every pattern | 1.0 |
| Direct null | 56 / 56 states EQUIVALENT, none BIASED | 56 / 56 EQUIVALENT |

- By the budget rule (the smallest design that passes every gate), **R = 512** was selected. No C1 frame existed before
  this decision.
- The measured coverage is far more conservative than the nominal 0.025. This is the expected price of the worst-case
  linear-term variance. It lowers synthetic precision in the stress families. On real noise it did not.

**Paired capture (Stage D), R = 512.**
- The guard waited for foreign GPU work from 06:27, then admitted the unit at 10:49.
- It preempted itself three times, each time killing only its own child when foreign work appeared:

| Time | Foreign work | WL109B run time before preemption |
|---|---|---|
| 10:50 | foreign PID | 33 s |
| 10:56 | foreign PID | 45 s |
| 14:37 | tmux `b28` | 1,332 s |

- At 14:55, 16 of 18 units are complete. H5 at 16 and 32 spp remain.
- The guard is waiting while `b28` and its GPU processes run, and holds no GPU memory.

## WL109B Amendment 1 — owner-directed host move LabServer69 → LabServer70 (2026-11-24)

**Owner direction, 2026-11-24.**
- LabServer69 is held by TransGI reproduction work until the evening.
- Execution therefore moves to LabServer70, which has the same RTX 3090.
- The owner expects no material effect on the final decision.
- LabServer63 is no longer used.

**Classification.** This is a non-scientific execution change inside the same ongoing protocol, so it is an amendment.
- The estimator, the inference procedure, the budget R = 512, every threshold and every decision rule are unchanged.
- Stages A–C and their outputs stand as completed on LabServer69.

**State at the move.**
- On LabServer69 the chain was waiting in the guard behind tmux `b28`. It held no GPU memory.
- 16 of 18 paired units were complete. H5 at 16 and 32 spp were missing.
- The operator stopped the chain processes (`run109b.sh` and its guard, by PID; no foreign process was touched) and
  appended an `OPERATOR STOP` line to `chain.state`.

**What was copied to LabServer70, paths unchanged, nothing on 69 modified or deleted:**
- the Python environment (venv and uv Python), Mitsuba 3.8.0 and `rest_dyn101`;
- the ReST-GI code copy;
- the frozen inputs (WL105/107/108);
- WL109 manifests, dev and metrics;
- the WL109B evidence directory without the null pool and smoke outputs, which later stages do not read.

Transfer script: `results/ecro_estimator109b/xfer70.sh` on LabServer69.

**LabServer70 differences.**
- NVIDIA driver 525.60.13; LabServer69 uses 550.54.15.
- A second GPU, an RTX 3080 Ti. GPU commands are pinned to the RTX 3090 by UUID, but the guard watches every GPU and
  every tmux session on the host.

**Predeclared cross-host rule** (`xcheck109b.py`, run under the guard before any further capture):
1. The unchanged WL109 admission must pass on LabServer70. It checks sources, checkpoint, scene and table hash (which must
   equal the WL109 manifest), and replays the archived 4096-spp V0 frame.
2. The first replicate of each of the 16 completed paired units is re-rendered with the same seed. It is compared with the
   stored `frame0_R0` and `frame0_C1H`.
   - **PASS** if every relative L1 is ≤ 1e-6 (the WL109 structural tolerance). The 16 units are kept, and only the 2
     missing units are captured on LabServer70.
   - **FAIL** otherwise. The LabServer69 units are moved aside and kept as descriptive only, and all 18 units are
     recaptured on LabServer70 with the same seeds.

**Rest of the chain.**
- Grid, timing and dynamic all run on LabServer70 (`run109b_amend1.sh`), so every equal-time comparison stays within one
  host and GPU.
- WL109's 0.792 equal-time ratio was measured on LabServer69. It is compared only descriptively.

### Amendment 1 execution: the frozen admission refuses LabServer70 (2026-11-24)

- **Transfer, 69 → 70.** Finished 16:40–17:23 on LabServer69's clock. LabServer70's clock runs a few minutes behind.
  All 16 transferred paired units match their `unit.json` hashes.
- **`tests109b.py`** passed on LabServer70.
- **`xcheck` stopped in the frozen admission after 6 s.** The guard admitted it after 120 s of free GPU. No GPU work had
  started.
  - Immediate cause: `renderer108.frozen_runtime` parses a single `nvidia-smi` row, but LabServer70 has two GPUs.
  - Underlying cause: the frozen WL108 admission pins LabServer69. It asserts GPU = RTX 3090 with driver **550.54.15**,
    asserts the OptiX library is `libnvoptix.so.550.54.15`, and labels the host `LabServer69`.
  - LabServer70 runs driver 525.60.13, so the frozen admission cannot pass there even with the parse fixed.
- The owner was asked to choose among three options:
  - wait for LabServer69;
  - replace the admission for LabServer70;
  - align LabServer70's driver.

## WL109B Amendment 2 — LabServer70 admission (owner decision, 2026-11-24)

**Decision.** The owner chose to replace the admission for LabServer70.

**What changed.**
- `experiments/ecro_estimator109b/admission70.py` is a copy of `renderer108.frozen_runtime`. Only its environment pins
  differ:
  - The GPU row is selected by the UUID in `CUDA_VISIBLE_DEVICES`. The name must still be RTX 3090.
  - The driver must equal LabServer70's declared driver, 525.60.13.
  - The OptiX library must match that driver.
  - The host label and the OptiX sha256 are recorded as found.
- The diff against the frozen function contains only these lines.

**What is unchanged and still executed.**
- Baseline and prior manifests.
- Renderer source hashes.
- Mitsuba 3.8.0, Dr.Jit 1.3.1, Torch 2.3.1+cu121, NumPy and Python versions.
- Workload, camera and scene fingerprint.
- Learned HIGH table hash.
- The archived 4096-spp V0 replay within the WL105 tolerance.

**How it is installed.**
- The frozen file is not edited.
- `common109b` installs the copy in place of `renderer108.frozen_runtime`, inside WL109B processes only, when
  `WL109B_ADMISSION=lab70` is set.
- The chain `run109b_amend2.sh` is `run109b_amend1.sh` with that flag set.

**What still guards reproducibility.**
- The Amendment 1 cross-host rule is unchanged: re-render the first replicate of each of the 16 LabServer69 paired units.
  - All relative L1 ≤ 1e-6: keep the 16 units.
  - Otherwise: recapture all 18 units on LabServer70.
- If Mitsuba/Dr.Jit OptiX cannot run on driver 525, the admission fails and nothing is captured. That would be reported.

### Amendment 2 execution and Amendment 3 probe: LabServer70 cannot run the frozen renderer (2026-11-24)

**Amendment 2 chain (`run109b_amend2.sh`, commit 5c4bdeb).**
- The first launch carried a placeholder commit label. It was stopped before any GPU work, recorded in `chain.state`, and
  relaunched.
- `xcheck` then stopped while importing Dr.Jit: `libatomic.so.1` is missing.
  - The WL108 runtime directory holds a symlink to LabServer69's `/usr/local/anaconda3/lib/libatomic.so.1`.
  - LabServer70 has no anaconda installation.
- A copy of the real file, `runtime70/lib`, removed that error. Two further incompatibilities then appeared:
  - LabServer70 runs CentOS 7, with glibc 2.17 and the GCC 4.8 `libstdc++`.
  - The frozen Dr.Jit 1.3.1 binaries need GLIBC 2.27 (`libdrjit-core.so`) and GLIBCXX 3.4.22.

**Amendment 3 probe (owner decision: try Docker).**
- Command: `docker run --rm` with the `python:3.10-bookworm` image (glibc 2.36).
  - GPU: the RTX 3090 only, selected by UUID, with `NVIDIA_DRIVER_CAPABILITIES=all`.
  - User `nd` (uid 1014); `/home/nd` mounted at the same path.
  - Before the run, no foreign GPU process or tmux session was present.
- Inside the container the frozen stack loads with the frozen versions: Dr.Jit 1.3.1, Mitsuba 3.8.0, Torch 2.3.1+cu121,
  NumPy 1.26.4, Python 3.10.21.
- OptiX initialization fails: *"OptiX 8.0 requires driver revision R535 or newer"*. LabServer70's driver is 525.60.13.

**Conclusion.**
- The frozen renderer cannot run on LabServer70 while its driver is older than R535, whether natively or in a container.
- Nothing was captured on LabServer70. No WL109B evidence or decision depends on it.
- Amendments 1–3 are recorded as **not executable on LabServer70**.
- The transferred copies remain on LabServer70 and are unused.
- LabServer69's data and state are unchanged: 16 of 18 paired units done; Stages A–C complete.
