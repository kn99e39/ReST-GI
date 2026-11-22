# Worklog 108 — ECRO Grazing-Rim Input and Learnability (2026-11-20)

The owner authorized WL108 on the existing `Denoise_Reconstructor` branch. Its base is
`69c8f23557e83d69890eeb98a9858524c536b20d`, the completed WL107 Amendment 1 assessment.
The owner designated the regular identifier. The working tree and LR3 branch were checked before
implementation. The two factorial variables are observable grazing geometry and the learning
objective. The operator is the original R3 pair of independently normalized nonnegative 15×15
kernels on LOW and HIGH.

## Preregistration

WL103 motivates paired covariance accounting. WL105 supplies R3, the training sampler and 40/8/6
camera split. WL106 supplies the frozen boundary and contact exclusions. WL107 supplies independently
qualified references, target-informed O1 panels and the negative O2 outcome. The WL107 fit proves
a feasible weight assignment on its panels, leaving runtime prediction open.

The [frozen protocol](../../experiments/ecro_denoise108/protocol108.json) specifies A0, A1, B0 and
B1, three paired seeds, thresholds, reference precision, conditionally sealed confirmation cameras,
compute budgets and stops. A0 and A1 receive a neutral zero channel. B0 and B1 receive the
foreground-conditioned mean squared view-normal cosine from the original 8×8 grid. All models
have 1,056,038 parameters, including 414 weights from the symmetric first-convolution extension.
The original initialized weights are copied and each new channel starts at zero.

L0 is the original component loss. L1 adds one term at coefficient 0.25, the relative squared
TOTAL error on mixed sphere-coverage pixels. A per-observation reconstruction loss and the
original full-frame component losses both remain active. No oracle target enters training.
The four arms use the same 20,000 Adam steps, cosine schedule, crop sequence by seed and final-step
selection. Held-out evaluation is run only after all twelve models are frozen.

## LabServer69 execution destination (owner direction, 2026-11-20)

The owner directed WL108 to LabServer69 after preregistration. LabServer63's first preflight
stopped before data qualification because its historical environment was missing the interpreter
and shared objects. An isolated version-pinned recovery reached the six-test suite. The suite
caught platform-dependent float32 promotion in the analytic synthetic positive control, before
qualification and training. Changing that calculation to explicit float64 reproduces the
predeclared target pair and keeps the same numerical assertion tolerance. The old logs are
preserved on LabServer63. No WL108 fit or reference render ran there.

LabServer69 uses the native RTX 3090 / 550.54.15 stack with Torch 2.3.1+cu121, Triton 2.3.1,
Mitsuba 3.8.0, Dr.Jit 1.3.1 and NumPy 1.26.4. An analytic CUDA sphere intersection returned
the known distance of 2.0 using the native OptiX library. The only local library search addition
is an experiment-scoped link to the installed libatomic. The frozen WL105/WL107 inputs are
copied to the WL108 inputs directory, with the full original identity gates applied there.
The historical scripts are read-only. The WL108 loader admits the requested device by validating
the original renderer/config/checkpoint bytes and canonical scene. Stage A uses only the scene
geometry and checks geometry against frozen per-pixel sphere hit counts. Conditional fresh
radiometric confirmation additionally requires exact archived learned-table identity and paired
historical renderer replay within the original WL105 tolerance before any new reference frames.

The original protocol file and sealed camera list retain their hashes. The host override and
synthetic arithmetic fix precede actual qualification or training. Both factors, all thresholds,
the frozen reference, the seeds and the timing budgets retain their original definitions.

### Stage A result

The direct authenticated LabServer63-to-69 transfer completed. Initial LabServer69 admission
verified the 983 WL105 files and stopped on the absent original WL101 deployment identity
sidecar. The sidecar was copied from the frozen snapshot and matched its SHA-256. The previous
qualification directory and logs were kept under `preflight69_missing_commit/`. The retry used
the same code and thresholds and qualified before training.

On LabServer69, commit `6b1d8adf0591019d95a68365953cd9f54aec63e7` authenticated the WL105 and
WL107 sources, reference, original R3 checkpoint
`d51b69199b99a8608d8bfbd1f17375cd837894350cb33d63d9170be7b29264e1`, and all 54 exact frozen
sphere-coverage maps. A repeated extraction was exactly identical. The maximum absolute
difference from the analytic sphere cosine statistic was `7.642e-7`. The two-world synthetic
inputs with equal frozen guides yielded 0.25 and 0.34 on the new channel and the softmax
positive control reproduced 1.00 and 1.05. The nine focused checks passed.

A fixed quadratic map of the old pointwise guides fit on training geometry attained validation
R² 0.835 and absolute RMSE 0.06199 on mixed pixels. This measures predictability under one
pointwise diagnostic. A neighborhood, camera-specific priors and nonlinear alternatives can
carry information absent from this statistic. The qualification extraction, including analytic
cross-check and host readback, had median 4.024 s per camera, range 3.705–4.337 s. Its
SHA-256 is `628686a630e57733dc31a2b4aeaca6e85fe400c1f7ffe8ee879680d36b2aebc8`.

The detached `chain69.sh` started the first paired training run at 15:05:43 KST. Results are
under `LabServer69:~/rest_sss101a/results/ecro_denoise108/`. Stage C is chained behind all
12 completed logged fits. Stage D is conditional on the frozen diagnostic trigger.

## Completed result — verified 2026-11-22 KST

### Implementation Fact

LabServer69 completed **12/12 fits** at 19:09:14 KST on November 20. Each fit used 20,000
steps with the frozen final-step checkpoint policy. The three paired training seeds were
`10550001`, `10850002` and `10850003`. Within each seed the four recorded initialized states
have the same SHA-256, and all final checkpoint hashes match the evaluator and the retrieved
files. The 6-view, 3-spp, 8-draw scoring finished at 19:17:18 KST. Frozen R3 replay passed its
per-image tolerance, and reference qualification passed. The conditional independent-camera
rendering threshold was missed by both comparisons.

### Measurement

Each tuple is ordered 4 / 16 / 32 spp. RMSE is relative linear HDR RGB and bias is signed
luminance on the WL107 silhouette mask. Reference RMS noise divided by the smallest eligible
primary RMSE was 4.44% / 6.94% / 8.35%, below the 10% limit. All other decision masks and the
signed-reference-stratum check passed.

| Arm | Primary RMSE | Silhouette RMSE | Silhouette signed bias | SSS interior RMSE | Material-boundary RMSE |
|---|---|---|---|---|---|
| R3 | 1.470% / 0.937% / 0.775% | 7.78% / 5.96% / 5.00% | +2.25% / +1.93% / +1.69% | 1.793% / 1.071% / 0.854% | 7.37% / 5.40% / 4.56% |
| A0 | 1.435% / 0.934% / 0.779% | 9.93% / 7.75% / 6.44% | +3.21% / +2.87% / +2.38% | 1.720% / 1.044% / 0.846% | 6.76% / 4.74% / 3.87% |
| A1 | 1.819% / 1.141% / 0.947% | 11.95% / 10.15% / 9.15% | +5.55% / +5.03% / +4.58% | 2.243% / 1.293% / 1.016% | 7.44% / 5.38% / 4.47% |
| B0 | 1.418% / 0.907% / 0.754% | 8.93% / 6.80% / 5.74% | +2.21% / +1.89% / +1.60% | 1.705% / 1.017% / 0.818% | 6.49% / 4.57% / 3.73% |
| B1 | 1.775% / 1.127% / 0.934% | 11.04% / 9.49% / 8.55% | +4.97% / +4.55% / +4.14% | 2.217% / 1.293% / 1.023% | 7.31% / 5.33% / 4.46% |

**Paired silhouette RMSE effects.** Positive percentages indicate improvement. Brackets are
the preregistered 95% crossed bootstrap intervals over training seeds, views and paired noisy
draws.

| Factor contrast | 4 spp | 16 spp | 32 spp |
|---|---|---|---|
| A1 over A0, adding L1 under G0 | −20.31% [−40.89, +23.15] | −31.02% [−56.28, +16.32] | −42.06% [−68.53, +11.00] |
| B0 over A0, adding G1 under L0 | +10.13% [−4.17, +18.14] | +12.31% [−7.33, +22.19] | +10.96% [−9.90, +20.85] |
| B1 over A1, adding G1 under L1 | +7.61% [−6.10, +15.76] | +6.58% [−5.82, +14.14] | +6.59% [−5.82, +15.56] |
| B1 over B0, adding L1 under G1 | −23.68% [−48.45, +23.18] | −39.57% [−67.99, +16.08] | −49.04% [−83.33, +8.80] |

The descriptor-by-objective interaction on the additive RMSE scale was −0.097 [−1.827, +1.262],
−0.285 [−2.391, +1.281] and −0.104 [−2.090, +1.598] percentage points. At 4 spp the descriptor
missed the required +15% improvement and positive interval lower bound. Its primary
nonregression, higher-spp direction and view/shell criteria passed. At 32 spp B1 retained
106.49% of A1's silhouette variance, above the 105% matched-control limit. The G0 objective
comparison missed the 4-spp effect, higher-spp direction, primary nonregression and localization
checks. A1 primary RMSE exceeded A0 by 26.76%, 22.15% and 21.49%, with all three paired intervals
excluding zero. No arm met the full WL107 production comparison.

**Bias and variance.** On the silhouette, adding L1 under G0 retained 43.79%, 41.56% and
40.88% of A0 variance while increasing signed bias and bias power. At 4 spp the
finite-draw-corrected linear RGB bias power rose from 0.02645 to 0.04979. The corresponding
B0/B1 powers were 0.02006 and 0.04413. The R3 silhouette variance was 2.92%, 4.94% and 6.57%
of raw, while A1 was 1.57%, 2.49% and 3.26%. The twice-covariance share in the learned arms
was 13.75–15.77% on the silhouette. The saved per-view measurements retain LOW variance,
HIGH variance, twice-covariance and TOTAL variance. All identities passed on retrieval.

**WL107 capacity gap and original-fit sensitivity.** On the exact original O1 panels, the
mean per-draw RGB RMSE was 3.69% / 2.13% / 1.73% for O1, 7.84% / 5.99% / 5.04% for frozen R3
and 8.86% / 6.81% / 5.78% for B0, the lowest learned-arm panel error of this batch. WL108
averages the eight per-draw RMSEs for every arm. WL107's historical panel table took the RMS
across the draws before averaging views. With its original seed, A0 obtained 1.478% primary
and 10.37% silhouette RMSE at 4 spp, compared with frozen R3's 1.470% and 7.78%. A0 is the
matched retraining control. Frozen R3 supplies the historical comparison.

### Observation

H3 and H4 drove much of the adverse change from L1. At 4 spp H3 silhouette RMSE was 17.99% for
R3 and 29.22%, 39.13%, 23.27%, 35.44% for A0, A1, B0, B1. H4 measured 9.13%, 9.75%, 15.49%,
9.44% and 12.96%. On V0 and H2, adding the same objective under G0 lowered silhouette RMSE.
At 4 spp B1 over A1 reduced silhouette RMSE in four of six views and in the first inner
shell, with small degradations in the second and third shells. Geometry, coverage, normal
and depth strata are available in the per-view and pooled raw measurements. Mean per-view
absolute signed bias for R3/A0/A1/B0/B1 at 4 spp was 2.75% / 4.09% / 5.79% / 3.25% / 5.14%.

The disjoint rim and first three inner shells accounted for 15.52% / 20.11% / 21.03% of R3's
primary RGB squared error at 4 / 16 / 32 spp. At 4 spp the respective contributions by
shell were 6.11%, 5.10%, 2.75% and 1.56%. The deeper full-coverage sphere contributed 29.23%
and the remaining primary image 55.25%. These are means of actual squared-error shares per
draw and view. The frozen contact exclusion applies.

**Timing and attribution.** Training took 1,184.7–1,189.6 s per fit, excluding loading,
within the 30-minute caps. Evaluation took 484 s. On LabServer69 the median of synchronized
512×512 4-spp model-only measurements was 10.869 ms for R3 and 10.924–10.925 ms across the
matched-arm seed medians. Peak additional allocation was 429 MiB for each. The Stage A
geometry timing (4.024 s median) includes 64 primary-hit query passes, host readback,
reduction and the independent analytic validation. It measures the offline diagnostic
construction path. The measured model-only latency excludes construction and rendering.

### Interpretation

The tested boundary-aware loss with the original inputs reduced variance while increasing
rim bias and primary error. The grazing descriptor showed a consistent positive point estimate
against the matched original-input controls, but the paired evidence was too uncertain and
the predeclared effect and variance criteria were missed. The two-world qualification
establishes descriptor responsiveness in its synthetic control. The real fit comparisons
measure learnability under a fixed optimization recipe. The best learned variant retains a
clear gap to the reference-informed nonnegative O1 assignment.

### Unresolved Question

Whether some other learned training rule can recover the O1 margin with the original guides
remains unresolved, as does the real-scene incremental value beyond this particular exposure
and sample regime. The six views previously motivated WL108. The independent confirmation
trigger failed, leaving their generalization beyond these views unmeasured.

### Architecture Verdict

**NOT SUPPORTED for the preregistered WL108 practical hypotheses.** A qualified negative
result closes the batch. Input sufficiency in general remains unresolved. The architecture
and LR3 branch retain their existing dispositions.

### Reproducibility

- Protocol SHA-256 `b6d4da10487025b43fae067316a3cfae732b8114a2824b1e40e1a903670817de`.
  LabServer69 fit and scoring commit `6b1d8adf0591019d95a68365953cd9f54aec63e7`.
- `chain69.sh`, `run_training108.sh`, and `run_evaluation108.sh` ran under the LabServer69
  native runtime recorded in Stage A. No branch, renderer or frozen dataset changes were
  required to produce the results.
- The verified raw mirror is `results/ecro_denoise108/lab69/` and contains all 12 checkpoints,
  their training provenance and full per-view/per-draw metrics. The raw metrics SHA-256 is
  `f439f2f793ffd5336962d84fa86573c41d340fd5bb0b66b926553fdf5c4235b4`.
- [Artifact manifest](../../evaluation/108/manifest108.json) ·
  [Pooled metrics](../../evaluation/108/results108.json) ·
  [Quality and intervals](../../evaluation/108/original_quality.png) ·
  [4-spp signed-error crops](../../evaluation/108/original_crops.png)
