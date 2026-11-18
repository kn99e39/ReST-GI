# Worklog 101H — ECRO SSS LOW/HIGH noise attribution (Denoising Track DN-1) (2026-11-18)

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
