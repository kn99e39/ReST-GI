# Worklog — Frozen ECRO C1 Spatial Bias Qualification and Conditional Dynamic Closure (2026-11-23)

**Batch class:** suffix batch of WL109 (owner brief, 2026-11-23). It closes WL109's residual spatial-bias uncertainty and,
only after a static pass, its unexecuted dynamic stage. Estimator, renderer and architecture stay as frozen in WL109. No new
mechanism.

**WL109 stays NOT_SUPPORTED.** WL109's data, code, protocol and worklog are not changed. WL109A answers its own question
under its own protocol.

**Question.** Does frozen C1 (M = 1) preserve the per-pixel expected radiance of frozen R0 within the WL109 spatial bias
tolerance, keep its cost-normalized TOTAL variance benefit, and does that survive the frozen P7 geometry edit?

**Description rule.** At M = 1, C1 is a conditional-resampling / LOW–HIGH decorrelation estimator: a finite, unbiased,
one-sample inner estimate. It is not exact conditional integration. WL109 measured that its gain comes from removing the
LOW–HIGH covariance, while Var(HIGH) is essentially unchanged.

## Corrections to the WL109 record (found while preparing this batch)

Both errors are in WL109's prose. WL109's JSON is correct. The owner brief for this batch repeats both.

1. **Number of bias tests.** H3 and H4 have no `control` ROI. WL109 therefore ran **28** view × bias-ROI tests, not 30
   (`evaluation/109/metrics109.json`, key `bias`). The correct counts are B1 **28 / 28** EQUIVALENT and B2 **26 / 28**
   within limit. WL109's verdict is unchanged.
2. **"No Q point exceeds its limit" is false.** The H4 material boundary has a 16-spp Q point of **0.0136 ± 0.0042**,
   which is above the 0.01 limit. WL109's own table shows this value. Its upper95 at that spp is 0.0236. What is true is
   that neither failing ROI has a Q point above the limit at the spp that set its maximum upper95: 4 spp for H4, 32 spp for
   H3.

## Design (frozen before any WL109A evidence)

Protocol: `experiments/ecro_estimator109a/protocol109a.json`. Code: `experiments/ecro_estimator109a/`. WL109 code is
imported unchanged.

### Stages

**Stage A — reproduction (CPU).**
- Checks WL109 source and protocol hashes against the WL109 run manifest.
- Reruns the unchanged `analyze109.py` on the 4,992 authenticated frames, with torch on the CPU. A shim maps
  `device="cuda"` to `"cpu"`; the analysis file itself is not edited.
- Pass requires every numeric leaf to match (rel ≤ 1e-8 or abs ≤ 1e-12), and the decision and verdict to match exactly.
- The first GPU admission also checks checkpoint, scene and table bytes, and that the table sha equals the WL109 manifest
  value.

**Stage B — calibration of the original Q bound.**
- *Historical null splits.* Each (view, bias spp) has 64 iid R0 frames from WL109: 32 paired R0 frames and 32 independent
  grid frames at the same spp. Random pairings give an exact-zero-expectation D = H_a − H_b at WL109's design
  (R = 32, G = 8, n = 4).
- *Synthetic known-truth families* at five real small-ROI shapes:
  - Gaussian;
  - heavy-tailed firefly-like at 4 and 32 spp;
  - heteroscedastic;
  - spatially correlated.
- *Independent GPU null pool:* 2,048 R0 frames per (view, spp); no C1 frame. It gives:
  - n = 64 full-frame groups;
  - raw small-ROI pixels for n = 4 … 64;
  - two disjoint R = 512 R0-versus-R0 runs of the whole decision.
- *Injected bias.* Uniform, noise-proportional, localized and signed-smooth patterns at Q_true = 0, 0.5, 1 and 2 × limit.
  The exact identity Q_g(q) = Q_g(0) + 2√(q/scale)·mean(b·D̄_g) + q is evaluated on the stored group sums.

**Calibration gates.**

| Gate | Requirement |
|---|---|
| Coverage | One-sided non-coverage P(upper95 < Q_true) ≤ 0.05, pooled over null-pool settings and in every synthetic cell at the design |
| Power | P(lower95 > limit \| Q_true = 2 × limit) ≥ 0.80, pooled null pool |
| Precision | Predicted per-setting P(upper95 > limit \| Q = 0) ≤ 0.01, and the pooled estimate ≤ 0.01 |
| Direct null | No BIASED state in either R0-versus-R0 run |

- A coverage or direct-null failure stops the batch as INCONCLUSIVE.
- **Budget rule:** R = 512 if all gates pass at n = 64; otherwise R = 1024 if its predictions pass; otherwise INCONCLUSIVE.
  The rule applies uniformly to every view and spp. It is decided from null and development data only.

**Stage C — independent evidence.**
- New seed families: paired 22,000,000 (other families: null 21M, grid 23M, timing 23.6M, dynamic 24M). Every family is
  above every WL101–WL109 range, including WL109's reserved synthetic range up to 18.31M.
- R paired R0 / C1 frames per view and bias spp (4, 16, 32), on all six views and all 28 decision pairs. Each frame checks
  the LOW and original-HIGH identity.
- Storage is sufficient statistics: subgroup sums of D and D², global moment sums, per-frame per-mask statistics, raw
  small-ROI pixels and the first full frame. Raw frames are not stored.
- A new R0 grid (18 levels × 64) and the WL109 Stage 3 timing block, on an exclusive GPU.

**Stage E — decision mapping.**
- Q state: EQUIVALENT if every upper95 ≤ limit; BIASED if any lower95 > limit; otherwise UNRESOLVED. B1 follows the WL109
  rule.
- The mapping, in order:

| Condition | Verdict |
|---|---|
| Any BIASED state | BIAS_SUPPORTED |
| Otherwise, any UNRESOLVED state, or reference precision fails, or the grid does not bracket | INCONCLUSIVE |
| Otherwise, P2, P3, P5 or P6 fails (WL109 thresholds, new data) | NOT_SUPPORTED |
| Otherwise, a Stage D flag is raised | Stage D |
| Otherwise | static pass, then Stage F |

**Stage D trigger.** Stage D runs on any of:
- any non-EQUIVALENT state;
- a resolved positive Q at family α = 0.05 / 84;
- a resolved positive Q in one of the four WL109 concern regions (H3/H4/H5 material boundary, V0 silhouette) at
  α = 0.05 / 12;
- a B1 interval excluding zero at α = 0.05 / 28.

Stage D first replicates the flagged pattern with fresh seeds. It then compares the replayed main-stream continuation P with
the independent inner continuation P′ under the same U. Its code is committed before it runs.

**Stage F — dynamic closure.**
- D1–D3, D5 and D6 are unchanged from `dynamic109.py`.
- D4 uses the qualified R at 4 spp.
- Outcomes: D1–D3 fail → NOT_SUPPORTED; D4 BIASED → BIAS_SUPPORTED; D4 UNRESOLVED or D5 fail → STATIC_ONLY_SUPPORTED;
  otherwise QUALIFIED_STATIC_AND_DYNAMIC.

### GPU ownership

- Every GPU command runs under the unchanged WL109 guard (`experiments/ecro_estimator109/guard109.py`).
- The guard waits, with no GPU library loaded, until no tmux session exists and no compute process has been listed for
  120 s. When a foreign compute process appears, it kills only its own child's process group.
- Capture units are written atomically and skipped on restart.
- CPU stages run with `CUDA_VISIBLE_DEVICES=` and `nice`.
- The WL109 Amendment 1 incident record stays in WL109.

## Execution record

(appended below as the chain runs)

### 2026-11-23 — chain start, tests and Stage A

- Chain `run109a.sh evidence 4e694af` started on LabServer69 at 11:22:21 (+09:00). Output:
  `results/ecro_estimator109a/evidence/` on the server.
- `tests109a.py` passed.
- **Stage A PASS** (CPU, torch 2.3.1, 6 threads, 6,930 s).
  - WL109 code equals the WL109 run manifest; protocol and table sha equal (`d7e40954…`).
  - The unchanged `analyze109.py` reproduces all 14,295 numeric leaves of `metrics109.json` within tolerance: 0
    mismatches; decision and verdict (NOT_SUPPORTED) equal.
  - Largest absolute difference: 1.3e-15. Largest relative difference among leaves with |value| > 1e-9: 7.8e-14.
  - The recorded `max_relative_difference` of 4.0 comes from identity-residual leaves whose published value is ~1e-17
    (abs difference 2.8e-17). It is float round-off around zero, within the abs ≤ 1e-12 rule.
- Stage B synthetic families finished (106 cells, 1,077 s). The historical-split analysis is running on the CPU.
- The first GPU unit (`smoke_null`) is waiting under the guard from 13:17:56: a foreign tmux session (`b21`) exists. No
  WL109A process holds a CUDA context while waiting.
