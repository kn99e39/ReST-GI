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

### 2026-11-23 — smoke stop and implementation fix (no evidence affected)

- The guard admitted the smoke units after the foreign work ended: null, paired, grid and timing each passed
  (14:56–15:05, 35–59 s each). The smoke pool and smoke analysis ran on the CPU. Their numbers are smoke data, not evidence.
- `smoke_dynamic` passed D1–D3, then stopped at D4 (`ValueError: cannot reshape … (8,0,3,…)`). The chain stopped at
  15:08:19 with `STOP stage=smoke_gpu exit=1`. No evidence unit had started.
- **Cause (implementation bug).** `dynamic109a.py` fixed G = 8 groups of 16-frame subgroups. The smoke R = 32 has only two
  subgroups. With the qualified R (512 or 1024) there are 32 or 64 subgroups, so the evidence path is not affected.
- **Fix.** `G = 8 if nsub >= 8 else nsub`, the rule `analyze109a.py` already uses. The D4 code path was dry-run on the CPU
  with the saved smoke accumulators before the restart. Protocol, thresholds and every R ≥ 128 path are unchanged.
- Stage B CPU analyses finished before the restart: historical splits (`calib/hist.json`) and synthetic families
  (`calib/synth.json`). Finished outputs and capture units are skipped on restart. The partial smoke dynamic directory is
  moved aside by the guard's `--before` hook.

### 2026-11-23 — restart, null pool and Stage B budget decision

- The chain was restarted at 15:09:40 with commit 8477b96. Every smoke unit passed, including smoke dynamic (D1–D4 code
  path to the end); `SMOKE PASS` at 15:22:27.
- **Independent R0 null pool:** 2,048 R0 frames for each of the 6 views × 3 bias spp. Run under the guard from 15:24:27 to
  16:11:37 (2,829 s).
- No foreign work appeared during any WL109A GPU unit, so the guard had no preemption event. Total WL109A GPU time:
  ≈ 57 min.
- `calib109a.py pool` ran on the CPU (81 s). **Budget outcome: `STOP_INCONCLUSIVE`**, with the reason "Q calibration cannot
  be qualified (coverage)".
- By `run109a.sh` and the protocol's budget rule, the chain ended with `COMPLETE verdict=INCONCLUSIVE (Stage B)` at
  16:12:58.
- Stages C–F were not executed: no paired C1 frame, no new grid or timing block, no dynamic edit.

## Final report

### Implementation Fact

- The frozen WL109 C1 (M = 1), the HIGH table (sha `d7e40954…`), the checkpoint, the scene, R0 and the WL109 thresholds were
  used unchanged. WL109 code was imported unchanged.
- Protocol `protocol109a.json` (sha `2504c982…`) and all decision code were committed in 4e694af before any WL109A evidence.
- Two code changes were made after commit 4e694af:
  1. 8477b96: the smoke-only group count in `dynamic109a.py` (described above). It affects no evidence path.
  2. After the decision, `fig109a.py` was reordered so the Stage B figures can be drawn without Stage C metrics, and a
     synthetic-coverage figure was added. It is descriptive and computes no decision.
- GPU ownership:
  - Every GPU unit ran under the unchanged WL109 guard and was admitted only after no tmux session existed and the GPU had
    no compute process for 120 s.
  - While waiting (13:17–14:56, behind TransGI `b21` and another session's `verify110.py`), WL109A held no CUDA context and
    no VRAM.
  - No foreign process was killed or interrupted. The WL109 Amendment 1 incident record is unchanged.
- **Reporting defect in `hist.json` (non-decision fields).**
  - `q0.upper_median` and `q0.upper_q99` are each computed per draw, from one subset, and then averaged over draws. Both are
    therefore the **mean upper95 over draws**, not a median or a 99th percentile.
  - `fail`, `noncov`, `power`, `null_upper_percentile` and `null_point_percentile` are computed correctly over all draws.
  - The budget decision does not read these two fields. The same fields in `pool.json` are correct, because they are taken
    over the 12,870 subsets of one setting.

### Measurement

**Stage A.** WL109's metrics were reproduced exactly. Details are in the Stage A entry above.

**Stage B, historical splits at the WL109 design (R = 32, G = 8, n = 4; exact-null R0-versus-R0 re-pairings of 64 WL109 R0
frames).**

H3 and H4 material boundary:

| setting | pixels | null P(upper95 > limit) | power at 2 × limit | WL109 C1 upper95 | percentile of C1 upper95 in null upper bounds | percentile of C1 point |
|---|---|---|---|---|---|---|
| H3 mb 4 spp | 103 | 0.340 | 0.50 | 0.0018 | 0.08 | 0.09 |
| H3 mb 16 spp | 103 | 0.384 | 0.42 | 0.0125 | 0.79 | 0.63 |
| H3 mb 32 spp | 103 | 0.375 | 0.44 | 0.0136 | 0.86 | 0.95 |
| H4 mb 4 spp | 48 | 0.682 | 0.21 | 0.0268 | 0.96 | 0.77 |
| H4 mb 16 spp | 48 | 0.680 | 0.20 | 0.0236 | 0.91 | 0.99 |
| H4 mb 32 spp | 48 | 0.646 | 0.24 | 0.0127 | 0.50 | 0.36 |

- Every other view × ROI × spp setting has null P(fail) ≤ 0.001 at this design, including V0 silhouette at 0.000.
- Mean one-sided non-coverage over all 84 settings: 0.026.

**Stage B, independent null pool (n = 64 per group, the R = 512 design).**

| quantity | value |
|---|---|
| Pooled non-coverage at Q_true = 0 | 0.028 |
| Pooled non-coverage at injected Q_true = 0.5–2 × limit (4 patterns) | 0.020–0.027 |
| Pooled null fail | 0.000 |
| Pooled power at 2 × limit | 1.000 for every pattern |
| Predicted per-setting fail at n = 64 | 0.000 everywhere |
| Two disjoint R0-versus-R0 decision runs (Q and B1) | 56 / 56 view × ROI × half states EQUIVALENT |
| H3 / H4 material-boundary median upper95 | 0.0004–0.0011, limit 0.01 |

- Per-setting non-coverage estimates exceed 0.05 in 11 of 84 settings (maximum 0.35, at H5 silhouette 32 spp and H1
  silhouette 16 spp). Each estimate comes from a single 16-group pool, so it varies strongly between pools.
- The frozen gate uses the pooled value.

**Stage B, synthetic known-truth families** (106 cells; 600 realizations at R ≥ 512, 2,000 below; binomial SE at 0.05 is
≈ 0.009 for 600 realizations):

- **Heteroscedastic** (per-pixel SD exp(N(0, 1))):
  - At R512G8, non-coverage at Q_true = 0 is 0.055–0.083 on all five ROI shapes; the worst injected cell is 0.098.
  - It is also above 0.05 at R128G8 (0.051–0.079) and R512G16 (0.040–0.067). At R32G8 it is 0.037–0.052.
- **Correlated:**
  - At R512G8, Q_true = 0 is 0.028–0.047.
  - Injected cells reach 0.053 (H4 mb) and 0.063 (V0 mb).
- **gauss_iid, mc_like_4, mc_like_32:** ≤ 0.040 at Q_true = 0 for every design. The single exception is gauss_iid H4 mb at
  R1024G8, which reaches 0.053 on two injected cells.
- Under the frozen skip rules, the R1024G8 cells are gauss_iid and mc_like_4 only.

**Budget rule (frozen).**

| design | coverage | other gates | result |
|---|---|---|---|
| R = 512 | Fails on 7 synthetic cells (5 heteroscedastic, 2 correlated) | Power, precision and direct null pass | Not qualified |
| R = 1024 | Fails on the gauss_iid H4 mb cell (0.053) | Precision and power predictions pass | Not qualified |

The rule therefore gives **STOP, INCONCLUSIVE**.

Figures:
- `evaluation/109A/calibration.png`: WL109-design null fail rates, and null-pool fail and coverage against n.
- `evaluation/109A/synthetic_coverage.png`.

### Observation

- At WL109's own design, the WL109 B2 bound fails on an exact zero-bias R0-versus-R0 difference with probability:
  - 0.65–0.68 at the 48-pixel H4 material boundary;
  - 0.34–0.38 at the 103-pixel H3 material boundary.
- At the same design, the bound would detect a bias of twice the limit with probability only 0.20–0.50 there.
- WL109's observed C1 upper bounds lie inside the null distribution of R0-versus-R0 upper bounds, at percentiles 0.08–0.96.
  None is beyond every null draw.
- With n = 64, the same Q bound on real R0 noise is precise:
  - median upper95 at H3/H4 material boundary 0.0004–0.0011;
  - pooled coverage 0.972;
  - power 1.0.
- The one-sided Student-t bound with 8 groups under-covers when per-pixel noise levels differ strongly (heteroscedastic
  family): non-coverage reaches 0.083 at Q_true = 0 and 0.098 in an injected cell, against the 0.05 gate. This gets worse
  with more replicates per group, not better: 0.037–0.052 at R32G8, 0.055–0.083 at R512G8.
- The R1024 miss (0.053 in 600 realizations) is within binomial noise of 0.05. The frozen rule has no Monte Carlo tolerance.

### Interpretation

- **WL109's two B2 failures (H3/H4 material boundary) are what an unbiased estimator produces most of the time at WL109's
  design.** That bound could not distinguish C1 from R0 at those ROIs, so WL109's failures do not show a C1 spatial bias.
- This is a statement about the precision of WL109's test. It does not show that C1 is unbiased, because no new C1
  evidence was collected.
- The question WL109A was frozen to answer, equivalence at the 0.01 / 0.02 limits with a qualified bound, is **not
  answered**. The frozen coverage gate was not met.
- The cause is a property of the inference method on synthetic heteroscedastic noise: an 8-group t bound on a skewed group
  statistic. It is not a defect seen in the R0 null pool, where both direct R0-versus-R0 runs were EQUIVALENT and pooled
  coverage passed.
- Whether real C1 − R0 differences at H3/H4 are heteroscedastic enough for this under-coverage to matter is not measured.

### Unresolved

- C1-versus-R0 spatial equivalence at the WL109 limits with a qualified bound: Stage C was not executed.
- The cost-normalized TOTAL variance benefit on fresh seeds (P2, P3, P5, P6) was not re-measured. WL109's 0.792
  equal-time ratio is the latest measurement.
- Dynamic closure under the P7 edit (D1–D6) was not executed. Only the smoke run touched that code path; its numbers are
  not evidence.
- A coverage-qualified inference method for heteroscedastic small ROIs is open: more groups, or a skew-robust or bootstrap
  bound. Choosing one would change the frozen inference method, which needs an owner decision. It is not a WL109A
  amendment.
- `hist.json` upper-bound quantile fields are mislabelled (see Implementation Fact). They can be recomputed post hoc if
  needed.

### Architecture Verdict

**INCONCLUSIVE** (Stage B budget rule: the Q calibration could not be qualified on coverage).

- WL109 stays **NOT_SUPPORTED**.
- C1 is not adopted.
- No successor batch, inference-method change or amendment is started.
- Living documents are unchanged; no architecture decision follows.

### Reproducibility

| item | value |
|---|---|
| Code | `experiments/ecro_estimator109a/` at 4e694af (frozen), plus 8477b96 (smoke fix) and the descriptive `fig109a.py` reorder |
| Protocol | sha `2504c98256f9…` |
| Server run | LabServer69 `~/rest_sss101a/results/ecro_estimator109a/evidence/` |
| Chain | `run109a.sh evidence <commit>` |
| CPU-stage environment | `CUDA_VISIBLE_DEVICES=`, `nice -n 10`, 6 threads |
| Seeds | null family base 21,000,000; calibration RNG 26,900,001; splitmix64 launch seeds |
| Local mirror (gitignored, no raw frames) | `results/ecro_estimator109a/lab69/` |
| Reviewer artifacts | `evaluation/109A/` (manifest `manifest109a.json` with sha256 and source paths) |
| `pool.json` sha | `fdaf884a…`, equal to `budget109a.json`'s `pool_json_sha256` |

Software: Mitsuba 3.8.0, Dr.Jit 1.3.1, Torch 2.3.1+cu121, one RTX 3090.
