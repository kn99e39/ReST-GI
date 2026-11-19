# Worklog — ECRO Grazing-Rim Operator and Input Qualification (2026-11-19)

WL107, Denoising Track, on the existing `Denoise_Reconstructor` branch at base
`a1d5c82b4be0ab18b7c5e02aeb408955b1b30e17`. The bounded Architecture question is whether the
existing ECRO inputs carry usable grazing-rim information and whether R3's normalized nonnegative
kernel class, or its learned weight prediction, can exploit it while reducing variance.

The [frozen WL107 protocol](../../experiments/ecro_denoise107/protocol/denoise107.json) keeps the WL105
40/8/6 view split, 4/16/32 spp inputs, renderer expectation, contact exclusion, 15 × 15 reflect-padded
support and independently normalized LOW/HIGH kernels. LOW is E1 + E2. HIGH is the frozen higher-order
table readout. TOTAL is their sum. The covariance of the paired component estimates belongs in TOTAL
risk. WL103 measured a contribution of 2Cov(LOW,HIGH) of about 23% of raw TOTAL silhouette variance
at 4 spp on its fixture. [WL103](103_WORKLOG_ECRO_SSS_LOW_HIGH_NOISE_ATTRIBUTION.md)

[WL105](105_WORKLOG_ECRO_COUPLED_LOW_HIGH_RECONSTRUCTOR.md) owns R0 through R3, the published
targets and the original qualification failure. [WL106](106_WORKLOG_ECRO_BOUNDARY_SAFE_COVERAGE_MOMENT_RECONSTRUCTION.md)
owns the coverage-moment mechanism and its closed INCONCLUSIVE result. Its SG2 failure and untrained
R4 status are historical controls. WL106's 4-spp validation bias shifted from +1.22% to +3.62% when
enforcing coverage moments. Its held-out mixing attribution and validation-only constraint result
use different splits.

## Frozen arms

| Arm | Scientific purpose and privileges |
|---|---|
| O0 | Frozen R3, two learned kernels, one per component, RGB-shared weights |
| O1 | Target- and covariance-aware fixed two-simplex optimization and a separate samplewise bound that also applies to noise-dependent convex weights |
| O2 | Matched R3-sized diagnostic with a sum-zero signed residual to the softmax weights. Same observed inputs, fit recipe and channels reaching the same branches |
| O3 | Conditional information control. A matched analytic predictor receives either the original observations or one privileged geometry statistic in the two-world ambiguity family |
| O4 | One conditional runtime candidate only after a qualified existing-input result identifies a specific operator defect |

O1 minimizes expected TOTAL RGB squared bias plus a fixed multiplier of TOTAL RGB variance. Its
stencil includes the center. Nonnegative weights sum to one independently for LOW and HIGH.
The synthetic risk uses the exact additive-noise covariance with a shared pixel term and a shared
frame term. The real implementation would factor the empirical joint covariance from 64 independent
32-spp draws per view. The first-order simplex gap bounds the minimum of the fitted convex objective.
A feasible point with low bias and variance is a capacity witness. A loose gap or the absence of a
witness on the seven-point multiplier grid leaves that case unresolved.

The real controls require independent fitting and assessment references. They retain WL105's two
assessment halves and budget 46 independent additional assessment blocks per view, plus eight
fit-only blocks. The original renderer, tables and data remain read-only. The assessment RMS
noise ceiling is 10% of the eligible-arm RMSE in the decision ROIs. The signed target standard
error ceiling is 0.25 percentage points in each occupied decision stratum.

## Implementation Fact

- Ran all 64 analytical edge fixtures at 4, 16 and 32 spp with 256 independent controlled draws
  per fixture and spp. Fixtures cross four subpixel offsets, two foreground contrasts, constant
  and smooth fields, and three dark-rim widths with flat and wavy tangential profiles. No frozen
  case was dropped. The stencil test checked the 225-tap ordering and reflect padding against
  explicit padded indexing. The independent midpoint quadrature check passed in every fixture.
- Executed O1 with exact synthetic means and joint covariance, including spatial covariance.
  Evaluated each fixed weight solution on independent draws. Executed an adaptive convex-hull
  bound on the isolated dark-pixel construction.
- Ran the two-world existing-input information test and its matched O3 analytic control. Both
  estimators see the same scenes, sampling distribution, equal prior and loss. The additional
  O3 statistic is the foreground subpixel mean squared view-normal cosine. The estimator knows
  the two synthetic data-generating laws.
- Verified every archived WL105 held-out input (162 files, including 144 disjoint noisy seeds),
  authenticated the WL105 data manifest and found zero duplicate camera records across the 54
  archived viewpoints. All six WL106 reviewer artifacts matched their manifest and SG2 remained
  false. The original worklogs, evidence and production files match the base commit.
- Prepared and locally tested the reference assembler, covariance solver and matched O2 training
  and evaluation code. The real-scene GPU stages were not executed. Automatic approval review
  rejected the source and WL105 hash-manifest transfer and background job on LabServer63. The
  server has the frozen payload and RTX 3080 Ti. The local mirror lacks the three R1–R3 model
  files and frozen eval manifest. Approval for the remote payload and execution was requested.
- No O4 qualified for implementation. No renderer or input contract changed.

## Measurement

### Synthetic capacity and risk

The largest absolute reference discrepancy between the analytic integral and independent 1,024-point
midpoint integration along each axis was `3.18e-4` linear units, within the implementation check of
`0.002`. The frozen test required at most 1% absolute signed band bias and 1% band profile RMS
bias, plus at least 50% luminance variance reduction.

| spp | Cases with a feasible point on the frozen grid | Dark-rim cases | Median lowest feasible variance / raw variance |
|---|---:|---:|---:|
| 4 | 62/64 | 46/48 | 3.29% |
| 16 | 64/64 | 48/48 | 3.31% |
| 32 | 64/64 | 48/48 | 3.36% |

The remaining two grid cases were both 1-pixel rims at offset 0.125 with the dark foreground
contrast, one flat and one wavy. At 4 spp and the two adjacent multipliers 0.125 and 0.5, their
profile RMS bias / variance ratios were 0.52% / 59.1% and 1.55% / 15.2% for the flat case, and
0.42% / 56.3% and 1.24% / 13.5% for the wavy case. Their simplex gap was too loose to exclude
feasible weights. The raw evidence retains the objective, lower bound and gap for every point.
The median relative gap over all seven multipliers and all cases was 3.0%, 45.9% and 93.4% by spp.
The achieved curves provide feasibility points, with limited precision on the location of the optimum
where the gap is large.

**Post hoc mathematical check on the entire 64-case family.** A separate construction used only
same-column taps, normalized and matched to the known center moment. Its expected RGB component
error stayed below `2.0e-15` across the inspected panel. Exact covariance retained 8.93–15.42% of
raw band variance across cases at each spp, including the two 4-spp grid misses. The construction
uses the known vertical edge orientation and expected radiance. This is additional existence
evidence under the fixed operator contract. The frozen 62/64 point-search result is preserved.

**Adaptive counterexample.** At a synthetic isolated trough, the center is 0.95 or 1.05 with
equal probability, its target is 1.00, and every other tap is at least 1.95. Both LOW and HIGH
have proportional positive shares of these values. On the bright half of the draws, even an
adaptive convex kernel's output is at least 1.05. If the total mean is at most 1.01, the
conditional-means variance alone is at least `0.04² = 0.0016`, or **64% of raw variance**. The
minimum is attainable in this idealized case by lifting the low output to 0.97. The best reduction
under the 1% bias ceiling is 36%, below the 50% requirement.

### Existing-input ambiguity and O3

Two synthetic worlds share the coverage, mean camera-facing normal and mean depth on the entire
stencil. The first uses foreground cosine 0.5 and the second uses an equal mix of 0.2 and 0.8.
Opposite lateral components cancel in both worlds. A rare response atom appears only in the
second world. When it is absent, all observed E1/E2/HIGH means and guides can coincide, while
the expected total is 1.00 in the first world and 1.05 in the second.

| spp | Second world yields the common observation | Existing-input Bayes MSE | Privileged Bayes MSE |
|---|---:|---:|---:|
| 4 | 77.25% | 0.0005448 | 0 |
| 16 | 35.61% | 0.0003282 | 0 |
| 32 | 12.68% | 0.0001407 | 0 |

The richer geometry scalar is 0.25 versus 0.34 and distinguishes the pair. The conditional input
collision is exact on a positive-probability event. The unconditional distributions differ. The
test establishes an information gap under this synthetic law and squared-error loss, independent of
network training and kernel class. It does not establish that the gap alone violates the joint
bias and variance contract on the real scene. The privileged score assumes the known two-world
prior and response law.

### Archived real-scene precision and baseline context

The WL105 archive yielded RGB target RMS SE of 0.3063% on the primary ROI, 0.4057% on the
silhouette and 0.3341% on material boundaries. Its largest signed-luminance stratum SE was 0.145
percentage points. The WL107 10% primary noise criterion failed at all three spp against
the published R3 primary RMSE. The archive pilot used the original two halves and historical
R3 metrics, with the original contact disk omitted from WL107 decision masks.

| spp | Published R3 primary RMSE | Published R3 silhouette bias | Published R3 material-boundary bias | Published R3 SSS interior RMSE | Reference noise / R3 primary RMSE |
|---|---:|---:|---:|---:|---:|
| 4 | 1.50% | +2.23% | +0.74% | 1.86% | 20.4% |
| 16 | 0.99% | +1.90% | +0.60% | 1.18% | 31.0% |
| 32 | 0.83% | +1.65% | +0.47% | 0.98% | 36.7% |

Published silhouette luminance variance for R3 was 0.01055 / 0.00447 / 0.00298 linear HDR²,
versus 0.36444 / 0.09146 / 0.04580 for raw. These are the WL105 measurements, taken before the
new precision audit. There are no new held-out O1/O2 profiles, model comparisons or confidence
intervals. Historical signed crops at original resolution remain in the WL105 gallery.
[WL105 metrics](../../evaluation/105/metrics.json) · [WL105 crops](../../evaluation/105/hires/index.html)

**Cost.** The archived R3 numbers are 10.72 ms median reconstruction, 428 MiB peak extra GPU
memory and 1,055,624 parameters. Archived renderer medians were 11.30 / 42.05 / 82.81 ms, giving
sum-of-medians estimates of 22.02 / 52.77 / 93.53 ms. No new deployable arm, timing or cost-matched
quality result exists for WL107.

## Observation

In the regular synthetic rim family, a target-aware operator can take substantial weight from
along the boundary while retaining the rim. Some risk multipliers move energy into the mixed
column and take it out of adjacent full-coverage columns. In the isolated trough, every neighbor
is brighter than the center and convex averaging can only raise the bright draw. In the
two-world ambiguity, even a predictor unconstrained by the kernel class has the same prediction
for the exact common observation.

[Trade-off curves](../../evaluation/107/synthetic_tradeoffs.png) ·
[Oracle and raw examples with signed HDR maps](../../evaluation/107/synthetic_error_maps.png) ·
[Signed rim profiles](../../evaluation/107/synthetic_rim_profiles.png) ·
[Archived target precision](../../evaluation/107/archived_reference_precision.png)

## Interpretation

Convex kernel capacity is adequate in the tested regular analytical fields when oracle weights
have access to the expected radiance. It fails the specified bias–variance contract in the
constructed adaptive extremum. The finite-spp information collision is an independent mechanism
with a positive loss even under exact Bayesian prediction. The archive precision failure and
blocked replay prevent attributing the real grazing rim to either mechanism or to learned
weight prediction.

## Unresolved Question

The held-out ECRO camera set still needs a qualified reference, reproduction of the learned
baselines, independent O1 covariance and the exact-input O2 comparison. Whether its rim contains
the isolated extrema or finite-spp ambiguities exhibited by the synthetic tests is unmeasured.
The relative importance of those cases to the existing R3 failure remains open.

## Architecture Verdict

**INCONCLUSIVE** on the held-out ECRO grazing-rim failure. Both kinds of limitation are proven
within their respective synthetic constructions. The real-scene attribution stopped at baseline
replay access and reference qualification. The result supplies no basis to adopt a new input
representation or operator.

## Reproducibility

- Protocol and initial capture freeze `d4c7f4b6796be80bb914f7df34be6d420cde84d7`, protocol SHA-256
  `0dd810ff7cb4cccf63a5a065b7af59552352fdd6621b89f6e2ef02248e79c5be`.
  Frozen synthetic execution `c7315a212a5cb288d470ef3863d2c84116ffb91f`.
- Held-out pipeline freeze `f9c375c3e2327c2617826011f438e7b66457cf5a` and independent O1 invocation
  `ce593843826be4db530dbea7988652c05667a3fb`. Archive audit and review commit
  `4c62a5af39b5796eeee6a0221822f45b9898859b`. Supplementary same-column check
  `80ddd7e85f7555d6cf1213b1ac59d53890b7eff8`.
- Synthetic RNG `11740001`, reviewer draw `11740002`, original WL105 seeds and staged disjoint
  GPU seeds are in the frozen protocol. The frozen WL105 checkpoint hashes are R1
  `cfa468c72747abf8b6bb689370380af7689f6d503959054766c8b883b70b8d72`, R2
  `3e420609e0b2c98355d4e4aedcf782f11798b8dc22cb581d9ccb376034222c01`, and R3
  `d51b69199b99a8608d8bfbd1f17375cd837894350cb33d63d9170be7b29264e1`. The archived ECRO checkpoint and HIGH table hashes
  begin `2d9cb580` and `d7e40954`. The complete identities and renderer source hashes are in the verified
  WL105 archive manifest and its data manifest.
- Local analysis ran on arm64 Darwin 25.6.0, Python 3.13.9, NumPy 2.3.5, SciPy 1.16.3 and
  Matplotlib 3.10.6. The matched predictor and tensor solver unit tests ran with CPU PyTorch
  2.5.1 on Python 3.11.15. Nine analytical and five tensor/reference checks passed. Planned GPU
  path uses LabServer63 RTX 3080 Ti, Mitsuba 3.8.0, Dr.Jit 1.3.1 and Torch 2.3.1+cu121.
- Executed `synthetic107.py results/ecro_denoise107/synthetic evaluation/106/synthetic106.json <synthetic commit>`,
  `reference107.py results/ecro_denoise105 results/ecro_denoise107/archive_precision evaluation/105/raw_manifest105.json`,
  `archive107.py results/ecro_denoise105 evaluation/105 evaluation/106 results/ecro_denoise107/archive_audit107.json`,
  `tangent107.py results/ecro_denoise107/synthetic/synthetic107.json results/ecro_denoise107/tangent107.json`
  and `review107.py <synthetic107.json> <reference107.json> evaluation/107 --audit <archive_audit107.json>`.
  Scripts are under `experiments/ecro_denoise107/`. The prepared `run107.sh capture|train|assess|oracle <commit>`
  was unexecuted.
- Complete curves including covariance terms and solver bounds, two-world results, per-view
  target precision and post hoc witnesses are indexed and hashed in the reviewer manifest.
  [Evaluation manifest](../../evaluation/107/manifest107.json)

## Owner approval and held-out execution resumed (2026-11-19)

The owner explicitly approved the LabServer63 transfer, GPU replay and reference/diagnostic
execution, and retrieval of WL107 results. The earlier stop above describes the evidence
available before this approval. The real-scene qualification is pending while the resumed
job runs through the frozen gates.

- Deployed commit `f8a8554` to `LabServer63:~/rest_sss101a/experiments/ecro_denoise107` with
  `evaluation/105/raw_manifest105.json`. The staged protocol and anchor matched their local
  SHA-256 digests. The staging-only change restricts the archived WL105 metrics dependency
  to the local two-half audit. It changes no fresh assessment estimator, inputs, threshold
  or synthetic evidence.
- The remote archive preflight verified 983 files, including the frozen checkpoint, input and
  output identities. This check precedes the actual R0–R3 output replay.
- The detached chain launched at 17:12 KST and entered the GPU queue after the archive check.
  It waits for the existing GPU workload to clear. It then runs `capture`, `train` and `assess`
  from the frozen source, with the reference and compute stops in force.
- The remote root is `~/rest_sss101a/results/ecro_denoise107/`. `pipeline.state`,
  `pipeline.log`, `chain107.sh` and the stage logs track execution.

**Resumed C0/C1 verification.** GPU execution began at 17:17 KST. All four reconstruction arms
replayed bitwise on the 144 frozen held-out inputs, with input immutability and repeat checks
passing. The frozen ECRO 4096-spp renderer replay matched all six archived component comparisons
exactly. The seed and launch-stream disjointness check passed. R3 measured 10.848 ms median and
11.147 ms p95 reconstruction, 427 MiB extra peak memory and 1,055,624 parameters on the RTX
3080 Ti. The replay JSON SHA-256 is
`75147fd7bc7189d81e037f53f37cb8a12fa394641407c52ee939f30dfc205863`.
All 384 covariance draws were captured. Independent fit and assessment reference capture continues.

**Reference and diagnostic execution through the arithmetic stop (2026-11-19 KST).** C6 reference
capture finished with 708 fresh files: 384 covariance draws, 48 fitting reference blocks and 276
assessment reference blocks. The 48-block assessment means per view passed the preliminary precision
gate. Relative target RMS standard error on primary / silhouette / material boundary was
0.06296% / 0.08366% / 0.06694%. Primary noise divided by the recomputed R3 RMSE was
4.28% / 6.72% / 8.12% at 4 / 16 / 32 spp. The largest signed stratum standard error was 0.0181
percentage points. The frozen limit is 10% for the RMS ratios and 0.25 percentage points for the
signed stratum standard error. The candidate-relative target check awaits the full assessment.

The O2 diagnostic used the unchanged 20,000-step schedule and completed in 1,410.6 s. Its last
validation relative RMSE was 0.01583 / 0.01093 / 0.00977. All 144 held-out O2 inferences completed.
The trained and inference manifests agree on O2 checkpoint SHA-256
`60d12537f7156419e75b389651c1efb7e9ac372e121144972561430351599389`.

The real metrics process stopped on its covariance consistency assertion at V0, 16 spp, at
20:42 KST. The authorized diagnostic retrieval reproduced one O2 pixel whose variance from
separately promoted components was `8.11208358e-5`, while the variance from the rounded float32
TOTAL was `8.11303553e-5`. Every saved O2 TOTAL in the affected eight frames exactly equaled
LOW + HIGH evaluated in the saved float32 dtype. The assertion had compared variances across
different rounding orders. The original failed assessment and the completed O2 inference
artifacts are retained. Approval was requested for an evaluation-only arithmetic check that
verifies the stored composition exactly and compares the two covariance calculations using the
same precision. No real-scene verdict is assigned at this execution stop.

## Amendment 1 — same-precision variance consistency check (2026-11-20 KST)

The owner explicitly approved the evaluation-only correction and retry. The scientific design,
metric formulas, decision thresholds, fit schedule, sample counts and frozen artifacts are
unchanged. The revised assertion independently checks exact LOW + HIGH composition in the saved
dtype and covariance closure on the components summed in float64. Both consistency checks retain
their respective exact equality and numerical tolerance. The regression test uses the eight
component pairs from the observed failure and verifies rejection of a perturbed TOTAL.

The retry uses the frozen 708 capture files, assessment means, R0–R3 inputs and saved O2 outputs
from the first execution. It starts a new raw assessment path, preserves the failed attempt, and
runs within the remaining 60-minute evaluation allowance. No capture, O2 fitting or O2 inference
will be repeated under this amendment.
