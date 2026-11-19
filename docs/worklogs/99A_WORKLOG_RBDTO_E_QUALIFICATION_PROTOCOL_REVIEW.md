# Worklog — WL99A: RB-DTO-E Workload Qualification Protocol Review and Amendment (2026-11-17)

> Append-only. Suffix batch of WL99. WL99's protocol (`rbdto99_v1`), results and verdict STOPPED_WORKLOAD_UNQUALIFIED are preserved unchanged; nothing in WL99's worklog, protocol files or reference outputs was modified.

## Origin and scope

**Brief.** The user's brief: "Continue the WL99 RB-DTO-E investigation by reviewing and, if justified, amending the workload qualification protocol."

**Question.** Was WL99's qualification failure caused by an inadequate measurement protocol? Can a valid amended protocol enable an interpretable RB-DTO-E architecture evaluation?

**Batch class.** Suffix (WL99A). It resolves WL99's residual uncertainty under preserved architectural contracts.

**Allowed changes.** Q3 measurement semantics, qualification resolution, and the evaluation implementation. Thresholds were not tuned to outcomes. No scene was changed to force a pass. The RB-DTO-E model and its contracts were not touched.

## What was done

1. **Amended Q3 (v2) declared and frozen before any v2 value entered the evidence.** Protocol `rbdto99a_v2`, commit `75838db`.
   - Metric: `experiments/rbdto99/q3v2.py`.
   - Physical definition:
     - C = the relative RMS of the body self-transport image Sd that on-surface averaging at λ_c = 2 cm would remove.
     - Implemented as a mask-normalised Gaussian high-pass, with s = λ_c / √12 / footprint, where footprint is the median on-surface pixel size from the camera geometry.
     - Noise is removed without bias by the cross-product of the two independent reference halves.
     - The ROI is eroded 3 px (scaled with resolution) away from non-body pixels, so geometric silhouettes and occluder bars are excluded.
   - Gates:
     - structure: C ≥ 0.05 at T0 and T1;
     - resolved: λ_c ≥ 4 px, and |C_384 − C_768| / C_768 ≤ 0.20 (or 768 against 1536);
     - reliable: block-bootstrap SE ≤ 0.25 C.
     - The WL99 angular (exact-direction) part is unchanged.
   - Every constant comes from the frozen WL99 Stage 1 protocol, not from candidate outcomes:
     - λ_c = 2 cm is the coarsest TwinP spatial cell;
     - c_min = τ1 = 0.05 is the Stage 1 fidelity tier.
   - Rationale for c_min: structure whose removal costs less than the tier cannot make a representation that erases it fail.
2. **Metric validation on synthetic images only** (`tests99a.py`; all pass). The six cases:
   - a smooth ramp gives C ≈ 0.002;
   - a λ_c sinusoid of amplitude 0.10 gives 0.0563, against 0.0569 analytic;
   - the same scene at 2× resolution gives 0.0564 (converged);
   - a sub-pixel step converges between 384 and 768;
   - noise only gives 0.004;
   - a silhouette control gives 0.000.
3. **Disclosed draft revisions, made before any candidate value.**
   - A first band-pass draft (G_s − G_2s) responded with only ~0.29 of a λ_c sinusoid's amplitude, so "C ≥ τ1" would not have meant what the rationale says. The synthetic test S2 exposed this, and the definition was changed to the high-pass.
   - A first draft of the independent-candidate generator placed the occluder bar along its own long axis, so it almost always blocked the light at the body centre. Its offset was moved across the bar's long axis before any render.
   - **Disclosure.** The pipeline dev smoke, run before the freeze commit but after the protocol file was written, computed v2 once on WL99 candidate A1's stored reference (T0 C = 0.012). No threshold was changed afterwards.
4. **Diagnosis D1–D5 and v2 qualification** (`qual99a.py`, LabServer63, RTX 3080 Ti, Mitsuba 3.5.1 / Dr.Jit 0.4.6).
   - D1: the WL99 Q3 values were reproduced from the stored 384 px references; all 26 development values are equal (`wl99a_summary.json`).
   - Matched 768 px references (8192 spp per half, same scene / camera / transport) were rendered for T0 and T1 of all 22 non-null candidates. Where v2 changed by more than 0.20 between 384 and 768, T0 was also rendered at 1536 px (A9, A10).
   - 8 **independent held-out candidates** were drawn by a seeded generator (`gen_indep99a.py`, seed 990901), frozen before any render and never inspected before their run. They were qualified from scratch with the WL99 v1 references and gates plus v2.
   - The WL99 Q4 comparison was redone at matched spp (the co-moved reference at the T0 spp): A1 0.0064 (limit 0.02), A10 0.043 (limit 0.090). Both pass; WL99's A1 Q4 miss was the noise-mismatch artefact.
5. **Execution incidents** (no evidence affected):
   - At 04:03 the GPU went idle between renders because of per-launch Python tracing and CPU analysis. Fix (`688367b`): scipy Gaussian, larger launches, and the WL99 v1 statistics deferred to a CPU process pool.
   - A 38-million-lane launch at 384 px exhausted GPU memory and stalled from 05:23 to 06:09. Fix (`46b96d1`): launches capped at about 24 million lanes.
   - One agent `pgrep -f` matched its own ssh command line and killed it. No other process was affected.

## Measurement

`evaluation/99A/wl99a_summary.json`, `fig99A_resolution.png`, `fig99A_structure.png`; 44 candidate-states (22 non-null candidates × T0 / T1).

**WL99 v1 Q3 is resolution-locked.**
- The edge width in px never scales with resolution. For a resolved feature it should double from 384 to 768 px; in all 44 states w768 < 1.5 × w384, with median w768 / w384 = 0.84.
- The local contrast falls with resolution: median c768 / c384 = 0.53.
- So v1 measures pixel-scale texture plus noise, not a physical feature width. Its 2 px / 0.30 thresholds therefore admit or reject a candidate depending on the pixel grid.
- On single halves (half the spp) the width falls by about 3–10 %, a noise effect of secondary size.

**Q3 v2 is resolution-stable and noise-unbiased.**
- C_384 vs C_768: 37 of 44 states converge within 0.20 (median change 0.09).
- 6 states do not converge: A1 T1, B1 T1, A10 T0 / T1, I3 T1, I8 T1.

**Structure at the 2 cm scale is small almost everywhere.**
- Median C is 0.016. Only 6 of 44 states reach C ≥ 0.05: A11 T1, I1 T0, I2 T0 / T1, I4 T0 / T1.
- Of those, three are not reliable, because the structure is concentrated in a few blocks:
  - A11 T1: 0.081 ± 0.023;
  - I1 T0: 0.057 ± 0.021;
  - I4 T1: 0.086 ± 0.024.

**Eligibility.**

| | v1 (WL99 rules) | v2 |
|---|---|---|
| admitted | A5, B3, I1, I2 (+ C null) | I2 only (+ C null) |
| A5 (groove S1 oblique) | admitted | C 0.016 / 0.024 (T1 unreliable): structure < τ1 |
| B3 (groove S2 oblique) | admitted | C 0.013 / 0.013: structure < τ1 |
| I1 | admitted | T0 0.057 unreliable, T1 0.025 |
| I2 (groove S1, 101°) | admitted | 0.071 / 0.072, converged, reliable: **admitted** |
| I4 (channel S1) | Q3 fail | 0.139 / 0.086, T1 SE 0.024 > 0.25 C: not admitted |
| required set | not met (no development-geometry A) | not met: development A, development B, independent non-groove A, second independent missing |

**Failure classes of the WL99 v1 Q3 rejections**, using the predeclared A / B / C / D mapping:
- mostly **A (insufficient structure; reliable and converged)**: A2, A3, A4, A6, A8, A9, A12, B2, I5, I6, I7;
- **A + B (under-resolved)**: A1, B1;
- **A or C + B**: A10, I3, I8;
- **A or C (unreliable)**: A11;
- unclassified: I4. It has structure, but T1 fails reliability.
- **No WL99 v1 rejection is a pure D (false rejection)** under v2.

## Observation

- **Localisation.** The higher-order transport change T1 vs stale (|Sd_T1 − Sd_stale|) is spread over whole plates (`fig99A_structure.png`, right column). It is not confined to sharp features.
- The 2 cm high-pass residual is concentrated along plate junctions and the edges of occluder shadows. Inside the plates the self-reflections are smooth, at the scale of the frozen S1 / S2 lobes.

## Interpretation

- **The WL99 measurement protocol was inadequate.** Its Q3 edge width and contrast are pixel-locked. Its verdicts on individual candidates (A5, B3, I1 admitted; A4, A8 rejected) were not physically grounded. The Q4 limit ignored the co-moved reference's noise.
- **That inadequacy did not cause the coverage failure.** Under a physically defined, noise-unbiased, resolution-converged measure with protocol-derived thresholds, 38 of 44 states show genuinely small structure at the 2 cm scale (C < 0.05). Most are well below: median 0.016.
- The dominant reason this family (S1 / S2 ER bodies of the WL99 kind) does not qualify is therefore **genuine insufficiency of sharp higher-order structure (A)**. Under-resolution (B) and noise (C) are secondary and explain a minority of states.
- Only one candidate (independent I2) qualifies. That is short of the required development / held-out coverage, so Stage 1–3 were **not** executed.

## Unresolved Question

- Whether the reliability criterion (block-bootstrap SE ≤ 0.25 C) is too strict for spatially concentrated structure (A11 T1, I1 T0, I4 T1). Changing it after seeing these values would be outcome-dependent, so it was not changed.
- Whether a different body or material family (sharper than the frozen S1, or near-mirror) would provide qualifying structure. That changes the physical workload family and is a scope decision for the owner, not a protocol repair.
- H1, H2 and H3 of WL99 remain untested.

## Architecture Verdict

- **STOPPED_WORKLOAD_UNQUALIFIED_V2.**
- The amended protocol is valid. It shows that the coverage failure is mostly genuine insufficiency of 2 cm-scale structure in the current workload family, and only partly a measurement artefact.
- This is **not** an RB-DTO-E result. There is no adoption and no training, and no successor is authorised.
- Protocol qualification, model training, higher-order representation quality, geometry-edit reuse and rendering utility are all **untested** for RB-DTO-E.

## Reproducibility

- **Commits.**
  - `75838db`: freeze of protocol `rbdto99a_v2` (sha 78e24695…), the independent candidates (sha 0cfa1057…), and the metric and diagnosis code.
  - `688367b`, `46b96d1`: execution fixes.
- **Server.** LabServer63, `~/rest_x99/results/wl99a/`. References at 768 px use 8192 spp per half (b about 24 M lanes); A9 / A10 T0 at 1536 px use 4096 spp per half. Independent candidates at 384 px use 16384 spp per half, with the v1 extension up to 65536.
- **Seeds.** Halves A / B: 990001+ / 990501+, offset by a hash of the configuration name, plus 7000 + resolution for the resolution study. Independent generator seed: 990901.
- **Reviewer artifacts.** `evaluation/99A/`: `wl99a_summary.json` (D1 check, per-state v1 / v2 table), `qual99a.json`, `tests99a.json`, `fig99A_resolution.png`, `fig99A_structure.png`.
