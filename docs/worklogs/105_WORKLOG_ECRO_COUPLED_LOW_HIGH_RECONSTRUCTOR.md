# Worklog 105 — Coupled LOW/HIGH vs combined-RGB single-frame reconstruction (Denoising Track) (2026-11-18)

## Scope and frozen baseline

This batch is the Denoising Track's first reconstructor-architecture hypothesis. It runs on the existing branch `Denoise_Reconstructor`, base `04a93e7` (WL103), with no new branch or worktree. WL105 is the next number that is free on every branch: WL103 is here and WL104 is on `ecro_wl84_scene`. The contact-region systematic HIGH bias belongs to the main track. This batch does not diagnose, compensate or optimize it. The contact region is excluded from the primary criterion, and the raw observations are kept.

**Hypothesis.** With identical noisy ECRO inputs and comparable reconstruction cost, a coupled LOW/HIGH reconstructor reduces stochastic error more than a qualified combined-RGB reconstructor. It must do so without systematic bias and without damaging geometric or radiometric boundaries.

**Frozen inputs.**
- The WL100 SSS checkpoint and its learned table.
- The WL101 renderer, with the WL101F input validation: scene, checkpoint `2d9cb580…`, table `d7e40954…`, renderer sources, and the stack. HIGH generation and lookup, the LOW estimator, camera sampling, geometry, materials, lights and the physical references are unchanged.

**Execution host.** LabServer63, on the WL101 stack used by WL101A–C and WL101B: Mitsuba 3.8.0, Dr.Jit 1.3.1, Torch 2.3.1, Triton 2.3.1, RTX 3080 Ti. The RTX 5080 workstation's checkout is on `ecro_wl84_scene` with an unpushed WL104A freeze, so it was not touched.

**Prior evidence used.**
- WL103: LOW and HIGH both carry material variance. Cov(LOW, HIGH) is positive and is 23–41% of total variance, coming entirely from C(E2, HIGH) through the shared first-event sample. HIGH noise is readout noise.
- WL103 G7: its stage attribution is invalid and is not used.
- WL101B: SVGF is a historical negative control. Filtering random subpixel coverage with a single centre-ray guide left a dark silhouette ring. It is cited, not rerun.
- WL101C/WL103: the contact HIGH bias is excluded from the primary criterion.
- WL104 (Sharp Interreflection): it lives on another branch with another stack. It is not used, so the sharp-highlight stress test is not applicable here.

## Design (frozen before any evaluation output)

The [protocol](../../experiments/ecro_denoise105/protocol/denoise105.json) and the code in [experiments/ecro_denoise105](../../experiments/ecro_denoise105/) are frozen.

**Target and data.**
- **Target:** the expected output of the frozen ECRO renderer, E1 + E2 + HL, from high-spp renders of the same renderer. Training and validation views use 4096 spp. Each evaluation view uses two independent 8192-spp halves, so the target uncertainty is known.
- **Physical reference:** the archived six-event halves, V0 only. They serve only for independent error accounting.
- **Views:** 40 random training views and 8 validation views are rejected if they near-duplicate an evaluation view. The evaluation views are V0 and WL101G's held-out H1–H5; no table, model or reconstructor has ever read them.
- **Samples:** 4, 16 and 32 spp. Training and validation have 4 replicates; evaluation has 8. All seed sets are disjoint, including at the launch-stream level.
- **Noisy inputs:** the original frame program's component means E1, E2 and HL.
- **Aux guides:** geometry only, on an 8 × 8 subpixel grid: coverage of each of the 9 shapes, mean normal, and mean depth, plus log2 spp. Coverage-aware guides remove WL101B's centre-ray silhouette mechanism by construction.

**Arms.** All arms share a 1.05M-parameter U-Net that predicts per-pixel 15 × 15 softmax kernels. The kernels preserve the local mean and are exact on constant regions.

| Arm | Input | Kernels | How it is trained |
|---|---|---|---|
| R0 | raw TOTAL | none | — |
| R1 (RGB baseline) | log1p(TOTAL) + aux | one kernel on TOTAL | relative MSE on the total output |
| R2 (coupled candidate) | log1p(LOW), log1p(HIGH) + aux | two jointly predicted kernels, applied to LOW and HIGH and summed into one image | relative MSE on the summed output |
| R3 (ablation: separation without coupling) | LOW + aux and HIGH + aux, in two independent half-width U-Nets | one kernel each, summed | each on its own component target |

R1, R2 and R3 are trained identically: 20,000 iterations, batch 16, 128² crops, Adam 2e-4 with cosine decay, seed 10550001. The final weights are evaluated; there is no checkpoint selection.

**Rejected alternatives:**
- direct radiance prediction, because of the bias and hallucination risk;
- an explicit per-pixel covariance input, because frames expose only component sums and a 4-spp covariance is unusable;
- temporal models, which are out of scope.

**Decision rule.** The primary measure is relative HDR RMSE against the frozen-ECRO target over all valid pixels minus the contact disk (floor within 0.25 m of the support point). Uncertainty comes from paired two-level bootstrap intervals, over views and then replicates.
- **R1 qualification:** R1 must beat R0, keep its silhouette gradient ratio within [0.8, 1.2], and keep |bias| ≤ 1%. If it fails, the batch stops with no valid baseline.
- **D (equal spp):** material if D ≥ 0.05 and its interval lower bound is > 0.
- **D_cost (equal end-to-end time):** R1 is given the extra spp s* that matches R2's frame-plus-reconstruction time. s* comes from a linear fit of frame time against spp, and R1's error at s* is interpolated log-log.
- **Safety:** |bias| ≤ 1% in every ROI; boundary RMSE ≤ 1.05 × R1; silhouette gradient ratio within [0.8, 1.2].
- **Target trust:** target noise must be ≤ 10% of R1's error.
- **Verdict:**
  - SUPPORTED if D, D_cost and safety all pass at every spp;
  - PARTIALLY SUPPORTED if they pass at some spp;
  - QUALITY-ONLY if D is material but D_cost never is;
  - otherwise NOT SUPPORTED or INCONCLUSIVE.
  - R3 only attributes; it never decides the verdict.

### Development before the freeze (code-path smoke, offset seeds +5,000,000, not interpreted)

- Model tests pass: LOW + HIGH = TOTAL exactly; deterministic inference; inputs unmodified; constant regions exact (≤ 1.4e-6); HDR preserved without clamping.
- Data tests pass: seed sets are disjoint, and the frozen-renderer gates G2/G3 pass. G1, the replay of the archived V0 4096-spp predictions, runs in the primary.
- **Per-camera frames.** WL101's `frame()` and `render_hq()` cache their frozen program by (res, spp) and capture the first camera. The data generator therefore calls the original `_frame_prog` per camera, with `render_hq`'s launch and seed convention.
- **Kernel application.** The smoke run measured unfold-based kernel application at about 6.6 ms per kernel: R1 11.6 ms, R2 18.2 ms. A fused Triton application, equal to the unfold reference within 1.2e-6 and used for inference in every arm, lowers this to R1 6.6 ms and R2 8.2 ms.
- **Cost rule.** Because the coupled design intrinsically applies two kernels, the initial rule (R2 latency ≤ 1.10 × R1) was replaced by the cost-matched comparison above. This happened before any evaluation output existed.

## Amendment 1 — seed renumbering (owner-approved, 2026-11-18)

The frozen chain (commit `42f386c`) stopped at its own seed-disjointness gate G4 before any training or evaluation data existed. Only the G1 archived-replay gate had run.

The frozen seed bases were badly chosen. With 40 views, train-noisy seeds (10510000 + 1000·v + 100·i + r) reach 10549303. That range overlaps the train-target (10520000), eval-noisy (10530000) and eval-target (10540000) families.

The owner approved re-basing the six families 100,000 apart:

| Family | New base |
|---|---|
| train-noisy | 11000000 |
| val-noisy | 11100000 |
| eval-noisy | 11200000 |
| train-target | 11300000 |
| val-target | 11400000 |
| eval-target | 11500000 |

All 785 seeds are unique under the same rule (independent check). Nothing else changes:
- camera, determinism, training and bootstrap seeds;
- views, spp, models, thresholds and analysis.

The failed log is kept as `results/ecro_denoise105/failed_g4/`. The chain reruns from the start in a fresh output directory.

## Evidence (Amendment 1 chain, commit `dda00cd`, LabServer63)

### Implementation fact

The chain ran five stages, all with exit 0 and no OptiX warning: render (with G1–G4), data tests, training of R1/R2/R3, held-out evaluation, and frozen analysis.

**Renderer gates.**
- **G1:** replaying the archived V0 4096-spp ECRO predictions at seeds 101100 and 101101 gives relative L1 = **0.0** for E1, E2 and HL, so the frozen renderer and checkpoint are bit-identical.
- **G2:** LOW + HIGH = TOTAL exactly.
- **G3:** same-seed replays are deterministic.
- **G4:** all seed sets are disjoint (480 / 96 / 144 noisy, 40 / 8 / 12 target).

**Data.** 54 views were generated.

**ECRO frame cost** on the frozen path (V0, 512², RTX 3080 Ti): 11.3 / 42.1 / 82.8 ms at 4 / 16 / 32 spp.

**Training.** Parameter counts were matched. The last validation row is relative RMSE on the validation views at 4 / 16 / 32 spp; it is logged only and selects nothing.

| Arm | Parameters | Training time | Final validation rel. RMSE (4 / 16 / 32 spp) |
|---|---|---|---|
| R1 | 1,045,729 | 10.9 min | 0.0155 / 0.0114 / 0.0102 |
| R2 | 1,054,018 | 18.3 min | 0.0156 / 0.0109 / 0.0096 |
| R3 | 1,055,624 | 19.8 min | 0.0144 / 0.0101 / 0.0090 |

**Inference.** Deterministic. Every input hash is unchanged after reconstruction.

| Arm | Reconstruction latency (512², fused kernels) | Peak extra memory |
|---|---|---|
| R1 | 6.79 ms | 442 MiB |
| R2 | 8.19 ms | 540 MiB |
| R3 | 10.72 ms | 428 MiB |

### Measurement

Held-out views are V0 and H1–H5, with 8 seeds per spp. The target is the frozen-ECRO expectation (2 × 8192 spp). Values are relative HDR RMSE, with signed luminance bias relative to the ROI target mean in parentheses. The primary region excludes the contact disk.

| spp | Arm | Primary | SSS interior | Silhouette | Material boundary | Control | Contact (excluded) |
|---|---|---|---|---|---|---|---|
| 4 | R0 | 0.2246 | 0.3415 | 0.2885 | 0.2574 | 0.1381 | 0.4982 |
| 4 | R1 | 0.0159 (−0.04%) | 0.0174 | 0.1489 (**+5.71%**) | 0.0798 (**+1.92%**) | 0.0071 | 0.0400 |
| 4 | R2 | 0.0160 (−0.11%) | 0.0188 | 0.1099 (**+3.71%**) | 0.0686 (+0.13%) | 0.0074 | 0.0410 |
| 4 | R3 | 0.0150 | 0.0186 | 0.0769 (+2.23%) | 0.0738 | 0.0074 | 0.0391 |
| 16 | R1 | 0.0115 | 0.0111 | 0.1350 (**+5.33%**) | 0.0573 (**+1.30%**) | 0.0043 | 0.0274 |
| 16 | R2 | 0.0111 | 0.0118 | 0.0930 (**+3.38%**) | 0.0537 | 0.0045 | 0.0287 |
| 16 | R3 | 0.0099 | 0.0118 | 0.0589 (+1.90%) | 0.0541 | 0.0045 | 0.0252 |
| 32 | R1 | 0.0100 | 0.0094 | 0.1227 (**+4.75%**) | 0.0460 | 0.0037 | 0.0227 |
| 32 | R2 | 0.0096 | 0.0100 | 0.0838 (**+3.14%**) | 0.0460 | 0.0039 | 0.0249 |
| 32 | R3 | 0.0083 | 0.0098 | 0.0494 (+1.65%) | 0.0458 | 0.0039 | 0.0212 |

**Target noise.** The relative RMS of the 16,384-spp evaluation target is 0.31% over the primary region. That is 19%, 27% and 31% of R1's primary error at 4, 16 and 32 spp, against a frozen limit of ≤ 10%.

**Paired comparisons.** The 95% intervals come from the two-level bootstrap over views and replicates.

| spp | D: R2 vs R1, equal spp | D_cost: equal end-to-end time | R3 vs R1 | R2 vs R3 |
|---|---|---|---|---|
| 4 | −0.6% [−6.4, +4.1] | −3.7% [−10.2, +1.6], s* = 4.5 | +5.4% [−3.8, +13.6] | **−6.4% [−12.0, −0.7]** |
| 16 | +3.6% [−7.5, +11.4] | +3.0% [−8.2, +10.9], s* = 16.5 | +14.1% [−1.5, +26.9] | **−12.2% [−22.5, −2.1]** |
| 32 | +4.0% [−6.7, +11.7] | +3.7% [−7.1, +11.5], s* = 32.5 | **+16.7% [+0.3, +29.9]** | **−15.2% [−27.4, −3.7]** |

- **Gradient ratios.** At the silhouette, every reconstructor is within [0.99, 1.02]; R0's is 2.1 / 1.4 / 1.2. At material boundaries R1–R3 range from 0.91 to 1.09.
- **V0 against the physical reference** (secondary, contact excluded). Raw 11.4 / 5.7 / 4.0%; R1 0.88 / 0.65 / 0.58%; R2 0.96 / 0.73 / 0.65%; R3 0.91 / 0.65 / 0.58%. The frozen-ECRO target itself sits 0.40% from physical, so the reconstructors do not move the image toward or away from physics beyond that floor.

### Frozen verdict: STOP — two pre-registered stop conditions

1. **Untrustworthy targets.** The 16,384-spp target noise is 19–31% of R1's error, above the 10% limit. The reconstructors reach about 1% error, so the target floor is not negligible. The limit was set without anticipating that error level.
2. **No qualified RGB baseline.** R1 has a signed silhouette bias of +5.7 / +5.3 / +4.8% and a material-boundary bias of +1.9 / +1.3% at 4 / 16 spp, against a frozen limit of ≤ 1%. The coverage-aware guides did not remove the silhouette error mode by construction.

R2's own safety check also fails on silhouette bias: +3.7 / +3.4 / +3.1%. The comparisons below are descriptive only.

### Post hoc (after the verdict; descriptive; `posthoc_silhouette105.py`)

The first silhouette crop in the frozen figures was centred on the median silhouette pixel, which falls inside the sphere for a circle, and used a raw-noise colour scale, so it does not show the edge. A post hoc figure anchors the crop at the left silhouette and uses R1's crop-error scale.

- **Crops.** Every kernel arm leaves a one-pixel dark line exactly on the silhouette column. Low-frequency mottling (±0.05–0.15) sits on the wall and inside the sphere. R1's line is the strongest and R3's the weakest.
- **Signed residual by sphere-coverage bin** (pooled over views and seeds, coverage 1/64–63/64), relative to the bin's target:
  - R1: +2.3% to +6.1% at 4 spp; +1.9% to +5.4% at 32 spp; largest around coverage 0.5.
  - R2: +1.2% to +3.4%.
  - R3: +1.1% to +1.9% at 4 spp; +0.6% to +1.5% at 32 spp.
- [Profile](../../evaluation/105/posthoc_silhouette_profile.png) · [Crops](../../evaluation/105/posthoc_silhouette_crops.png)

## Final report

**Implementation fact.** Built and executed on the frozen WL101 ECRO renderer:
- per-camera component-mean data generation with renderer gates;
- coverage-aware geometry guides;
- three parameter-matched kernel-predicting reconstructors (R1 combined RGB, R2 coupled LOW/HIGH, R3 separated ablation), trained identically;
- a fused Triton kernel application used for inference by every arm;
- held-out evaluation on six views and three spp, a frozen paired-bootstrap analysis, and one labelled post hoc silhouette diagnostic.

ECRO, its checkpoint, the HIGH table, the LOW estimator and the contact semantics were not changed. Amendment 1, a seed re-base approved before any data existed, is the only protocol change.

**Measurement.** As above.

**Observation.**
- All reconstructors remove about 93–95% of raw primary error at 4 spp and keep gradient magnitudes near the target, so there is no oversmoothing at the silhouette by gradient ratio.
- All of them leave a systematic, positive (coverage-pooled) silhouette bias, plus a thin dark boundary line and low-frequency mottling.
- The coupled R2 roughly halves R1's silhouette bias but does not lower the primary error.
- The uncoupled R3 has the lowest error and the smallest silhouette bias, at 1.6 × R1's reconstruction latency.

**Interpretation.**
- The frozen verdict is STOP: the targets are not trustworthy at this error level and the RGB baseline failed qualification. The hypothesis is therefore **not established** by this batch.
- Descriptively, three points hold:
  1. **No coupling advantage over RGB.** R2 shows no resolved primary-error difference from R1 at equal spp or at equal end-to-end time.
  2. **Separation beats coupling.** Where R2 and R3 differ, the separated R3 is better, significantly so at every spp.
  3. **Boundary bias depends on the input formulation.** Silhouette bias falls from RGB to coupled to separated inputs.
- In this design, the useful ingredient appears to be **component separation** (each component filtered by its own kernel from its own statistics), not joint coupling. Joint prediction from both components did not help, and it hurt relative to separation. This is a descriptive reading under a stopped protocol, not a qualified finding.

**Unresolved questions.**
- Whether R3's advantage survives:
  - a qualified, bias-free silhouette treatment (the shared kernel-boundary failure mode is unresolved);
  - lower-noise targets (2 × 8192 spp is insufficient at ~1% error);
  - cost-matched width (R3's two half-width networks cost 1.6 × R1).
- Whether coverage-aware guides, kernel size or the relative-MSE loss cause the dark boundary line and the positive mixed-pixel bias.
- No sharp-highlight evidence: the fixture is diffuse/SSS, and WL104's sharp fixture lives on another branch and stack.
- One scene and checkpoint, single-frame only.

**Architecture verdict.**
- **Is coupled LOW/HIGH reconstruction useful?** Not established, and not supported descriptively: it gives no primary-error gain over RGB and is worse than separated components.
- **Is it preferable to combined RGB at matched cost?** No resolved difference: D_cost intervals contain 0 at every spp.
- **Is it worth retaining for temporal reconstruction?** The evidence favours retaining the **separated LOW/HIGH** formulation (R3-type) as a candidate, conditional on fixing the shared silhouette bias and on evaluation with lower-noise targets. It does not favour joint coupling.

No reconstructor architecture is accepted and no living document changes. No temporal work or successor batch is started.

**Reproducibility.**
- Commits: branch `Denoise_Reconstructor`; freeze `42f386c`; Amendment 1 `dda00cd`; this report is the next commit.
- Stack: LabServer63 RTX 3080 Ti; Mitsuba 3.8.0 / Dr.Jit 1.3.1 / Torch 2.3.1 / Triton 2.3.1 / Python 3.10.21 (`~/NeuralShaderDemo/external/8dna26/.venv`); `chain105.sh`, using `run105.sh`.
- Weights SHA-256: R1 `cfa468c7…`, R2 `3e420609…`, R3 `d51b6919…`.
- Seeds: protocol `seeds` after Amendment 1.
- Raw results: data 5.4 GB, eval outputs 1.5 GB, models, logs and manifests, all under `results/ecro_denoise105/`.
- Reviewer artifacts: `evaluation/105/`, with frozen review, post hoc figures and hashes.
