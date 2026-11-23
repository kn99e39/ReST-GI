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
