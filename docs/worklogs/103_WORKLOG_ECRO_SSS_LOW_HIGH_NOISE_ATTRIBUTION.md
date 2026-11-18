# Worklog 103 — ECRO SSS LOW/HIGH noise attribution (Denoising Track DN-1) (2026-11-18)

> Renumbered from WL101H to **WL103** by the research owner on 2026-11-18. This happened after evidence collection and before reporting; see "Owner renumbering" below. The freeze text that follows is kept as committed, including its original numbering paragraph.

## Authorization and freeze, 2026-11-18 KST

The user authorized one bounded LOW/HIGH noise-attribution study of the current ECRO implementation. It is the first diagnostic batch of a separate Denoising Track, and its purpose is evidence for a future ECRO-aware reconstruction architecture. The brief excludes implementing or training a denoiser, repairing WL101B's SVGF, changing the HIGH generator, readout, LOW sampling or estimator, and tuning away the HIGH bias. The user directed work on a new branch `Denoise_Reconstructor`, created from `ecro_lr3_adapter` at `c0daf39`, and execution on the local workstation (RTX 5080) instead of LabServer63.

**Numbering.** Under the AGENTS.md policy this is a suffix batch. It is a dependent evidence-gap follow-up of WL101A. WL101A's 32-spp attribution was INCONCLUSIVE, and that worklog recommended evaluating the full readout under LOW–HIGH correlation and measuring the persistent HIGH error separately. This batch uses the same fixture, checkpoint and estimator as WL101A–C and introduces no mechanism. The owner may reclassify it. Files would then be renamed with the change recorded, not silently renumbered. The track label is DN-1.

**Baseline and its validity.** The study uses the WL100 SSS checkpoint (`model.pt`, SHA-256 `2d9cb580…`) and WL100 test configuration 0 with light 0. This is the WL101A/WL101C cfg0. The original WL101 optimized renderer (`rt101.py`, byte-identical to the LabServer69/63 deployment hash `417fd6c0…`) is imported unchanged. WL101D–G concern Oracle-table construction and validation code (membership, containment classifiers, projection). They did not change the raw camera renderer or the checkpoint. Their control defects therefore do not invalidate this comparison. No WL101D–G artifact is used.

**Signal path, from the code.** The ECRO frame traces two optical-domain events per sample (`pt100.trace`, mode `ecro`, kmax 2):
- **E1:** NEE at the first camera hit, plus the SSS entry-to-first-exit walk NEE on the sphere.
- **E2:** the same at the second hit, plus the ballistic through-sphere term generated at the first hit.
- **HIGH = HL:** throughput × B[patch(second hit), bin(−direction)], read at the second hit before that event's SSS walk.
- **Composed output:** `(E1 + E2 + HL) / spp`.

One PCG32 stream per sample lane feeds three stages in order:
- **J:** pixel jitter.
- **P:** the k = 1 SSS walk and k = 1 closure sample or exit, which fix the second hit and its throughput.
- **S:** the k = 2 SSS walk, which affects E2 only.

HIGH therefore depends on J and P only. LOW shares J and P with HIGH and adds S.

The HIGH table B is generated from fixed state-ray and source-point offsets (`table_seed` 86001) and carries no runtime RNG. Development checks found it bitwise identical across repeated generation, and within 3.0e-7 relative L1 of the archived WL100 table. In the normal renderer, any "HIGH state sampling" is therefore a fixed, seed-independent quadrature realization, not render noise.

**Design.** All arms keep geometry, material, light, camera and weights fixed and vary seeds only.
- **J arm (primary):** the unchanged frame path at 1, 4 and 16 spp with 64 independent seeds each. It yields Var(LOW), Var(HIGH), the signed Cov(LOW,HIGH) and Var(TOTAL), with TOTAL taken from the contract-composed image. The identity is checked numerically.
- **PF arm (conditional stage diagnostic):** a copy of the frame program whose only change is the RNG wiring, giving J, P and S separate streams. Pick-freeze triples (J,P,S), (J,P,S′), (J,P′,S″) decompose the variance by stage. The arm is admitted only if it is bitwise-structurally correct and statistically equivalent to the normal arm (G6, G7).
- **HQ arm:** eight independent normal 4096-spp renders against the two local WL100 physical halves (seeds 100901/100902, six events, the same physical definition as WL101's halves). This gives a stable, unbiased squared-bias cross product for LOW, HIGH and TOTAL, with reference-noise accounting.
- **Q diagnostic (conditional):** re-draws the frozen source or state offsets four times each and reads ΔHIGH on common random numbers. It shows how large the frozen quadrature realization is relative to the persistent HIGH bias.
- **Dynamic check:** not run. The local references cover only configurations that differ in both geometry and light.

**ROIs**, fixed from geometry before any noise statistic:
- **Primary:**
  - SSS interior (45,692 px).
  - Sphere silhouette (870 mixed-coverage pixels from an 8 × 8 subpixel id buffer).
  - WL101C near-contact floor (1,318).
  - A back-wall control (86,118).
- **Secondary:** the WL101A/C regions, whose counts were reproduced exactly.

The [protocol](../../experiments/ecro_noise101h/protocol/noise101h.json) freezes seeds, gates G0–G8, estimators, bootstrap rules, verdict rules and display. Code is in [experiments/ecro_noise101h](../../experiments/ecro_noise101h/), raw output in `results/ecro_noise101h/` and reviewer artifacts in [evaluation/101H](../../evaluation/101H/).

### Development checks before freeze

- **Runtime.** Local `rt101` loads Mitsuba 3.9.1 / Dr.Jit 1.5.0 / torch 2.8.0+cu128 on the RTX 5080. WL101A–C used Mitsuba 3.8.0 / Dr.Jit 1.3.1 on LabServer63, so cross-stack equivalence is an admission gate, not an assumption.
- **Geometry.** The own center-ray id buffer equals WL100 `map_0` exactly. The WL101A/C mask counts are reproduced exactly.
- **Split arm.** When only the suffix stream changes, E1 and HL stay bitwise identical while E2 changes. A same-seed replay is bitwise identical.
- **Estimator tests.** Synthetic tests confirm the known-moment, signed-covariance, pick-freeze, stable-bias and identity estimators.
- **Bootstrap interval.** A coverage test showed that percentile or basic bootstrap intervals of replicate-resampled ddof=1 variances under-cover badly when the ROI has far more elements than replicates (0.52–0.60 at R = 64, M = 400). Normal intervals with the bootstrap SE cover at 0.95–0.99. The protocol uses the latter.
- **Pipeline smoke.** A smoke run used 4 seeds offset by +5,000,000 so that no evidence seed was consumed. It passed every gate end to end, including the cross-stack replays (worst 2.4e-5 relative L1 against the 1e-4 gate). Its noise statistics were not inspected.

## Owner renumbering, 2026-11-18 KST

After the evidence run and before any reporting, the owner directed that this batch be numbered **WL103** (a regular number) instead of the suffix WL101H. Nothing scientific changed. The worklog file and the reviewer folder now carry 103. The frozen identifiers stay as committed in `ea1dc7a`, because the protocol hash and the run manifest reference them:
- protocol `"id": "WL101H"`;
- code in `experiments/ecro_noise101h/`;
- raw output in `results/ecro_noise101h/`.

Read every `WL101H` / `evaluation/101H` inside those frozen files as WL103 / `evaluation/103`.

## Evidence, 2026-11-18 KST

### Implementation fact

- **Normal arm, unchanged.** Every normal-arm frame came from the original `rt101.RT.frame` / `render_hq`. Instrumentation is limited to:
  - reading back its native per-pixel E1, E2 and HL channel sums;
  - a float32 contract composition `(E1 + E2 + HL) / spp`;
  - a geometry-only id buffer for the ROIs.
- **Split arm, diagnostic only.** A diagnostic subclass re-wires the three RNG stages and changes nothing else.
- **No reconstruction.** No reconstruction, filtering or temporal accumulation exists anywhere in the pipeline.
- **Run.** Commit `ea1dc7a` on a clean experiment directory, finished 18:41 KST.
  - Wall time: 64 × 3 normal frames in 3.2 s, 64 × 3 × 3 split frames in 6.7 s, 8 × 4096 spp in 22 s.
  - The raw tree is 7.6 GB.

| Gate | Result |
|---|---|
| G0 inputs | all 16 frozen hashes match; experiment directory clean |
| ROIs | all 11 frozen counts reproduced; own center-ray ids = WL100 `map_0` (0 mismatches) |
| G1 HIGH state | table vs archived WL100 table: 2.98e-7 relative L1. Three regenerations: source and table bitwise identical. Table bitwise unchanged after all frames. |
| G2 replay of PRED_0 (seed 100903, 4096 spp; Mitsuba 3.5.1 eager → 3.9.1 frozen) | worst 2.37e-5 relative L1 over full / LOW / HIGH / E1 / E2 × full frame and primary ROIs (gate 1e-4) |
| G3 paired LOW vs both physical halves | worst 2.32e-5 (gate 1e-4) |
| G4 determinism | every same-seed replay bitwise identical (normal and split) |
| G5 recomposition | per frame ≤ 2.70e-8 relative L1. Variance identity residual ≤ 4.9e-9 · V_T in every ROI and spp. |
| G6 split structure | E1 and HL bitwise identical between suffix-only pairs in all 192 triples; values finite and ≥ 0 |
| Q restore / CRN | original table restored bitwise; LOW identical on all eight redraw readouts |
| **G7 split equivalence** | **FAIL, 5 of 36 tests** (see below). The stage decomposition is therefore **INVALID** and not interpreted. |
| G8 tests | pass |

**G7 failure.** Every mean test passed (max |z| 2.35 against a critical 3.20). The failed variance-ratio tests (split / normal) were:
- **near:**
  - TOTAL at 1 spp: 1.077 (z 3.41);
  - HIGH at 16 spp: 1.034 (z 3.86).
- **control at 16 spp:**
  - TOTAL: 1.039 (z 9.29);
  - LOW: 1.038 (z 8.96);
  - HIGH: 1.032 (z 7.72).

Control passes at 1 and 4 spp (ratios 0.996 and 0.990). The normal arm's control variance scales as 1/spp almost exactly (n·V_T 2.073 / 2.074 / 2.079 in the units below). The deviation is therefore inferred, not tested, to lie in the split arm's RNG wiring, at about 4% at 16 spp. No cause was identified. One untested hypothesis is a dependence between the role-offset PCG streams. Per the frozen rule, no stage (jitter / first-event / k = 2 walk) attribution is reported. The stage figures carry the invalid label.

### Measurement

**Units and reading.** Shares are signed fractions of Var(TOTAL): s_L + s_H + s_C = 1, where s_C = 2 Cov(LOW, HIGH) / Var(TOTAL). Each was measured across 64 independent seeds of the normal arm. The shares are invariant across spp within ±0.004, except near contact (s_H 0.360 → 0.348 from 1 to 16 spp; see the slopes below). 4 spp is shown. ρ is the ROI-aggregated LOW–HIGH correlation. "Bias share" is stable b²_TOTAL / (Var_TOTAL(n) + b²_TOTAL). Intervals are 95%.

| Primary ROI (px) | s_L | s_H | s_C | ρ | Bias share at 1 / 4 / 16 spp | Class at 1 / 4 / 16 spp |
|---|---:|---:|---:|---:|---|---|
| sss_interior (45,692) | 0.498 [0.497, 0.499] | 0.173 [0.172, 0.173] | 0.329 [0.328, 0.329] | 0.56 | 0.000 / 0.000 / 0.001 | MIXED / MIXED / MIXED |
| silhouette (870) | 0.602 [0.598, 0.606] | 0.169 [0.166, 0.171] | 0.229 [0.226, 0.232] | 0.36 | 0.002 / 0.008 / 0.033 [0.016, 0.049] | LOW / LOW / LOW-DOMINANT |
| near contact (1,318) | 0.321 [0.318, 0.325] | 0.350 [0.346, 0.355] | 0.329 [0.325, 0.332] | 0.49 | 0.057 / 0.187 / **0.476 [0.351, 0.601]** | MIXED / MIXED / MIXED |
| control wall (86,118) | 0.439 [0.439, 0.439] | 0.152 [0.151, 0.152] | 0.410 [0.409, 0.410] | 0.79 | 0.000 / 0.000 / 0.000 | MIXED / MIXED / MIXED |

**Secondary ROIs.** These classes are reported per ROI; the overall verdict uses the primary ROIs only.

| Secondary ROI | s_L | Class |
|---|---:|---|
| Full frame | 0.506 | LOW-DOMINANT |
| Floor | 0.610 | LOW-DOMINANT |
| Sphere boundary | 0.532 | LOW-DOMINANT |
| WL101A contact/shadow | 0.501 | LOW-DOMINANT at 4 and 16 spp; MIXED at 1 spp (lower bound ≈ 0.500) |
| Flank | 0.479 | MIXED |

At every ROI and spp, s_H is 0.10–0.35 and s_C is a resolved positive 0.23–0.41. No ROI is HIGH-DOMINANT, COVARIANCE-DOMINATED or BIAS-DOMINANT under the frozen rules. The frozen overall verdict is **REGION-DEPENDENT** at 1, 4 and 16 spp.

**Where the covariance sits.** Absolute variances at 4 spp, linear HDR²:

| ROI | V_E1 | V_E2 | V_H | C(E1,H) | C(E2,H) | C(E1,E2) |
|---|---:|---:|---:|---:|---:|---:|
| sss_interior | 0.0298 | 0.0821 | 0.0371 | −0.0044 | 0.0397 | −0.0025 |
| silhouette | 0.224 | 0.314 | 0.074 | −0.056 | 0.107 | −0.136 |
| near | 1.6e-6 | 0.00808 | 0.00881 | ≈ 0 | 0.00413 | ≈ 0 |
| control | 8.4e-5 | 0.227 | 0.079 | ≈ 0 | 0.106 | ≈ 0 |

Two patterns stand out:
- **Covariance is E2–HIGH.** The whole LOW–HIGH covariance is C(E2, HIGH). On diffuse first hits (near, control), E1 is almost deterministic and LOW variance is E2 variance.
- **Negative terms at the sphere.** At the sphere, E1 has a negative covariance with HIGH and with E2. At the silhouette, C(E1,E2) is the largest negative term.

**SPP scaling.** Fitted log-variance slopes:
- Full frame, sss_interior and control: −1.000 ± 0.002 for LOW, HIGH, 2Cov and TOTAL.
- Silhouette: −0.997 ± 0.005.
- **Near contact: −0.977 ± 0.006 (TOTAL).** n·V rises 0.0956 → 0.1007 → 0.1018 from 1 to 16 spp.

The single-render 4096-spp variance there is 0.67–0.72 of the 1/n extrapolation. A heavy-tailed per-sample distribution in the contact region would produce both patterns. In that case the variance and its bootstrap SE at low total sample counts are biased low. This is inferred, not tested.

**Bias** (8 × 4096-spp ECRO against two independent WL100 physical halves; signed values in Rec.709 luminance, Welch 95%):

| ROI | Signed TOTAL bias | Signed HIGH bias | Signed LOW bias | Stable HIGH RMS (of own physical HIGH RMS) |
|---|---|---|---|---:|
| near | +0.06994 [0.06986, 0.07002] (**+26.5%**) | +0.06993 (**+32.3%**) | +0.00001 [−0.0010, 0.0010] | 32.2% |
| contact_shadow | +1.18% | +2.15% | ≈ 0 | 4.4% |
| floor | +0.24% | +0.69% | ≈ 0 | 2.6% |
| sss_interior | +0.13% | +0.28% | 0.0000 | 1.6% |
| silhouette | −0.02% (interval contains 0) | — | — | 5.9% (spatially signed, cancels in the mean) |
| sphere_boundary | — | — | — | 4.9% |
| control | −0.11% | **−0.87%** (resolved) | — | 0.84% |

Near-contact bias:
- **Independent replication of WL101C.** WL101C measured +26.49% / +32.32% using the WL101 reference halves, Mitsuba 3.8 and other seeds. This run reproduces those values with independent references, a different stack and fresh seeds.
- **Size relative to noise.** Reference-mean noise is 3.3e-5 linear HDR² and HQ-mean noise is 2.1e-6, both well below the stable b²_TOTAL of 5.8e-3.

LOW stable b² estimates are ≤ 1% of LOW RMS everywhere and slightly negative (e.g. control −1.3e-5 [−1.5e-5, −1.2e-5]). A negative squared bias outside its interval means the 16 × 16 block-bootstrap SEs are somewhat optimistic for small b². Read the "bias resolved" flags in low-bias ROIs with that caution.

**HIGH-state quadrature (conditional Q diagnostic).**
- **Table change.** Re-drawing the frozen offsets changed the table by:
  - 0.4–1.0% relative L1 for source offsets;
  - 2.1–2.5% for state offsets.
- **Image-space ΔHIGH RMS** (common random numbers, relative to own physical HIGH RMS):

| ROI | Source redraws | State redraws | Stable HIGH bias |
|---|---:|---:|---:|
| near | 3.0% | 4.9% | 32.2% |
| sss_interior | 0.7% | 1.1% | 1.6% |
| control | 0.6% | 0.5% | 0.84% |

- **Near mean shifts.** Mean near shifts per draw range from −4.0% to +6.7%.

These offset redraws also feed the generator inputs it was not trained with. The values therefore indicate the effect size, not a clean quadrature error.

**Timing and instrumentation overhead** (local RTX 5080, shared with the desktop display; descriptive only). nvidia-smi read 99% / 79% utilization at the start and end of the timing stage, so contention cannot be excluded. Median ms at 1 / 4 / 16 spp:

| Step | 1 spp | 4 spp | 16 spp |
|---|---:|---:|---:|
| Frame | 0.82 | 3.03 | 10.99 |
| 9-channel readback | 0.53 | 0.54 | 1.10 |
| Contract composition | 0.31 | 0.32 | 0.48 |
| Split frame + readback (eager, unfrozen) | 4.0 | 6.3 | 14.9 |

Component capture adds only the readback. The normal frame program itself is unchanged.

### Observation

[Variance maps](../../evaluation/103/variance_maps.png):
- **LOW std** is high on the directly lit walls and floor (E2 / indirect sampling) and in the sphere.
- **HIGH std** is lower and spatially smoother.
- **The signed 2Cov map** is positive over the whole frame. It is strongest on lit diffuse surfaces and weakest on the sphere interior.

[Contact crop](../../evaluation/103/crop_contact.png): at the contact, LOW and HIGH std both fall toward the contact shadow. The residual maps show the same two-lobed HIGH overbrightness under the sphere that WL101C reported.

[Silhouette crop](../../evaluation/103/crop_silhouette.png): LOW std is elevated in a one-pixel ring on the sphere outline. The bias crop shows no comparable ring. It shows a wall-seam line and seed-level speckle.

[Residual maps](../../evaluation/103/bias_maps.png): the HIGH residual adds positive seams (floor–wall and the right-hand corner) and negative box faces. The LOW residual is visually uniform.

Other figures: [SPP scaling](../../evaluation/103/spp_scaling.png) · [MSE budget](../../evaluation/103/mse_budget.png) · [raw components](../../evaluation/103/raw_components.png) · [ROI masks](../../evaluation/103/roi_masks.png) · [gallery](../../evaluation/103/index.html)

### Interpretation

- **HIGH has no runtime noise source of its own** (measured + code). In the current ECRO, HIGH's stochastic variance is readout noise: the deterministic table is read at a sampled second hit, with sampled throughput. The table is bitwise reproducible and has no runtime RNG.
- **The LOW–HIGH covariance comes from the shared first-event sample** (measured + code). The k = 1 direction or exit, the second hit and the throughput feed both E2 and the HL lookup. C(E2, HIGH) accounts for the whole Cov(LOW, HIGH). The resulting covariance is positive and large: 23–41% of TOTAL variance, ρ 0.36–0.79.
- **The components cannot be treated as independent noise sources.** Var(LOW) + Var(HIGH) understates Var(TOTAL) by that 23–41%.
- **Region dependence** (measured): the silhouette (and floor / full frame) is LOW-dominant. SSS interior, contact and the control wall have no single dominant variance source.
- **Bias at contact is a separate, persistent error** (measured): near the contact a systematic HIGH bias of about +32% sits beside the variance. Its share of TOTAL MSE is 6% at 1 spp, 19% at 4 spp and 48% [35%, 60%] at 16 spp.
  - By 1/n extrapolation, bias would match variance near ~17 spp and dominate above it. This is inferred, not measured between 16 and 4096 spp.
  - Elsewhere, stable bias is ≤ 3.3% of TOTAL MSE at ≤ 16 spp (silhouette 3.3% at 16 spp).
- **The frozen quadrature draw explains little of the contact excess** (conditional): redrawing it moves HIGH by an order of magnitude less than the contact bias.

### Unresolved question

- **Stage decomposition (G7).** The jitter / first-event / k = 2-walk split failed G7, so the following remain unmeasured:
  - how much silhouette variance is pixel-footprint (coverage) integration versus transport sampling;
  - how much of LOW is the HIGH-independent k = 2 walk.

  The split arm's ~4% variance excess at 16 spp is unexplained.
- **Heavy tails at contact.** Low-sample-count variances there may be underestimated (slope −0.977, HQ ratio 0.7). Its share and slope values carry that caveat.
- **Bias interval width.** Block-bootstrap bias SEs appear optimistic for small b².
- **Physical interpretation limits.** The reference is the six-event finite WL100 definition. The HIGH bias is not separated into generator weights, patch/bin discretization and readout. WL100's limited native-volpath qualification still bounds physical interpretation.
- **Scope.**
  - One fixture, camera and checkpoint; static scene, so temporal instability equals inter-seed variance.
  - No dynamic edit check: no single-factor edit pair with local references.
  - Spp beyond 16 was observed only at 4096.

### Architecture verdict

Measured responsibilities, for this fixture only:
- **Variance across the whole image:** reconstruction must deal with LOW and HIGH variance. Neither is negligible in any ROI, with s_H 0.10–0.35.
- **Covariance:** the two are strongly positively correlated through E2. A per-component treatment that assumes independent LOW and HIGH noise would misstate 23–41% of TOTAL variance.
- **Silhouette:** variance is LOW-led there.
- **Persistent bias:** the contact HIGH bias (+32%) is not sampling noise, does not decrease with spp, and is invisible to any reference-free filter. At ≥ 16 spp in the contact region it is about half of the error.

What profiling alone cannot decide:
- Whether a reconstructor should consume LOW, HIGH, their composition, or the noise-free table as auxiliary input. Measured variance shares do not establish denoiser utility, bias, latency, or behaviour under edits.
- Whether the silhouette error mode WL101B introduced relates to footprint variance; the stage split that would show this is invalid.

No reconstructor architecture is selected or adopted. No denoiser was implemented. No successor batch is authorized by this result.

### Reproducibility

- **Commits.** Branch `Denoise_Reconstructor` from `ecro_lr3_adapter` `c0daf39`.
  - Freeze and evidence: `ea1dc7a`.
  - The reporting commit changes only figure layout (`constrained_layout`) and the gallery title in `analyze_h.py`. Re-running the analysis reproduced `metrics.json` byte-identically (SHA-256 `7e359fc1…`).
- **Stack.** Local RTX 5080 16 GB, driver 596.49, Windows 11. `external/8dna26/.venv-cu128` (Python 3.11.9, torch 2.8.0+cu128, NumPy 1.26.4, SciPy 1.17.1, Matplotlib 3.10). Mitsuba 3.9.1 / Dr.Jit 1.5.0 from `external/mi39_dyn101`, cuda_ad_rgb, FREEZE=1.
- **Inputs.** Checkpoint SHA-256 `2d9cb580…`. References `REF_0_0/1` (seeds 100901 / 100902). Prediction `PRED_0` (100903). Table `table_0`, gbuffer `map_0`. All hashes are in the protocol.
- **Seeds.**
  - Normal arm: 10183000 + 100·spp + r.
  - Split arm: base 10185000 + 1000·i + 10·r.
  - HQ: 10189001–10189008.
  - Q: 10189201–4 and 10189301–4, readout seed 10189401.
  - Bootstrap: [10189999, arm, spp].
- **Commands**, from `experiments/ecro_noise101h`:
  - `<venv> tests_h.py <map_0.npz> <coverage.npz>`
  - `<venv> run_h.py ../../results/ecro_noise101h/evidence`
  - `<venv> analyze_h.py ../../results/ecro_noise101h/evidence ../../evaluation/103`
- **Process status.** The run's process exit status is not reliable on this Windows stack, because the interpreter can abort at `os._exit` after completion. Completion is the `finished` field and the `RUN COMPLETE` line.
- **Outputs.**
  - Raw (gitignored, 7.6 GB): `results/ecro_noise101h/evidence/`, containing `manifest.json` with gates, timings and every raw SHA-256, plus the logs.
  - The pre-freeze smoke tree `results/ecro_noise101h/smoke/` is retained as non-evidence.
  - Reviewer artifacts: [evaluation/103](../../evaluation/103/), with `metrics.json`, `evaluation_manifest.json` (file hashes and display scales: stochastic family max 3.67, residual family max 0.263 linear HDR, square-root mapping) and the gallery.
