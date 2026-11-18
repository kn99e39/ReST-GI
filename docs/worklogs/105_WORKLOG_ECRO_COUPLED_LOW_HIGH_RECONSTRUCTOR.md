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
