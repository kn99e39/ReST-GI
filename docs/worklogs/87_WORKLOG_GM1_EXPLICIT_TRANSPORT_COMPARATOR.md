# Worklog — GM1 Learning-free Global Transport Comparator and Architecture Discrimination (2026-11-11)

## Origin and scope

- **Brief.** "WL87 — Learning-free Global Transport Comparator and Architecture Discrimination" (user, 2026-11-11). One bounded implementation and experiment batch.
- **Question.** At the same patch × bin representation, frozen ER scattering, geometry state and camera readout, does the learned global response operator of WL86 (ECRO v0) give a quality-matched advantage over explicitly computed physical transport, and what representation limits remain?
- **Hypotheses.**
  - H0 (explicit sufficiency): an explicit operator on the same state reaches comparable fidelity at practical cost.
  - H1 (learned utility): explicit transport represents the response, but learning saves measurable computation.
  - H2 (representation bottleneck): the state cannot carry sharp transport even with physical coefficients.
- **Frozen.**
  - WL84 ER closures. The WL86 code (`experiments/ecro86/`) is imported unchanged.
  - Checkpoints `ecro_s8601` / `ecro_s8602`, the WL86 representation, references, masks, thresholds and verdicts.
  - The WL86A diagnostics and WL86B study.
- **Not done.** No learned operator is trained or tuned. No H-P, subspace or successor architecture is built. No new branch or worktree.
- **Where.**
  - Branch `Selective_Recompute`; LabServer63.
  - `experiments/gm1_explicit87/`, `results/gm1_explicit87/`, `results/evaluation/87/`.
  - Server `~/rest_gm187`, which reads `~/rest_ecro86` and `~/rest_ecro86a`.
- **Comparator terminology.** All comparators are physical recomputations with the same frozen ER (path tracing, table fill, explicit transport). No external neural baseline is run. No Neural Full Recompute or Native Reuse claim is made.

## Deployed WL86 contract, verified against the code

| Item | Deployed value (file) |
|---|---|
| Transport | b = S_ER(e + C_G b), q = S_ER e, T = S_ER C_G. Reference: next-event estimation at ≤ 6 camera vertices, no Russian roulette (`pt86.Ctx.trace`) |
| Event convention | A next-event contribution at camera vertex k has k scattering events: e1 = q, e2 = T q, h = events 3–6 |
| Table B | `[1270, 64, 3]`: outgoing radiance of **2–5** events per patch and outgoing bin (`pt86.Ctx.field`, probe mode: no next-event estimation at the probe vertex, k = 2…5) |
| Readout | At the second camera vertex: `throughput(x1) × B[patch(x2), bin(x2 → x1)]` (`pt86` mode `ecro`). Camera paths therefore get 3–6 events from the table, and q at x2 (= e2) is not in B |
| Bins | 4 rings uniform in cos θ × 16 azimuth sectors of the patch-local frame (t, b, n) = equal solid angle 2π / 64. `bin_of` and `dir_in_bin` are consistent |
| State | Per patch 512 rays = 8 per incident bin, uniform within the bin. **One draw of offsets `default_rng(86001).random((4, 1, 512))` is shared by all patches**. Each ray records its hit patch (front face only) and the outgoing bin at the hit |
| ER | `eval(wi, wo) = f(wi, wo) cos(wo)`; f is reciprocal (symmetric GGX lobes and s(μ) factors; test T_SYM, max relative difference 1.7e-5). The Lambert walls and boxes are ER states with `lam_0 = albedo`, so f = ρ / π exactly |
| ECRO source encoding | Patch-average direct irradiance per incident bin from 16 fixed points (`pt86.Ctx.source`) |

There is no discrepancy with the WL86 / WL86A worklogs. One clarification: the per-patch state rays are not stratified independently per patch. All patches use the same 512 local directions, and only the origin point differs per ray.

## Comparators (frozen in `protocol/gm1_explicit87.json`)

**Primary: EXP (C1, same-state receiver gather).**

- L₁ = q_d is the once-scattered radiance of the actual lights, projected onto patch × outgoing bin.
  - 16 stratified points per patch, each with a shadow ray and the exact light direction.
  - Lambert patches: ρ / π · E, exact.
  - ER patches: the bin average over outgoing directions uses 1,024 MIS-mixture samples per point (½ ER sampling, ½ uniform hemisphere). It is computed in the point's shading frame and binned in the patch frame.
- L_{k+1}[i, b_o] = Σ_r W_ray[m_i, r, b_o] · L_k[hit_r, dbin_r] over the 512 deployed state rays of patch i.
  - W_ray = (1/ΔΩ) ∫_{b_o} f(v, ω_r) dv · cos ω_r · ΔΩ / 8, at each ray's **exact** local direction ω_r.
  - Estimated with 65,536 MIS-mixture samples per (material, ray direction). Lambert rows are exact.
  - W depends only on the materials and the realisation's directions, not on the geometry. It is built once (initialisation).
- **Curved sphere patches.** The 8 patches of a sphere band are rotations about z, so W has one row set per band (4 extra rows).
  - Each row is evaluated at that ray's origin point: local shading frame, local cosine (0 below the local horizon), and outgoing directions above the local horizon.
  - Each row is normalised by the bin's mean valid solid angle, because the oracle field averages over valid directions. q of sphere patches is normalised the same way.
  - The C2-ENC and C3 controls use a flat patch-frame table for sphere receivers. This approximation is declared.
- B_exp = L₂ + L₃ + L₄ + L₅ = Σ_{j=1}^{4} T_d^j q_d. It is read by the unchanged WL86 LOW + lookup renderer with the WL86 ECRO render seeds (86600 / 86601).
  - Because the seeds are common random numbers, EXP − ECRO differences come from the tables alone.
- No renormalisation, clamping or energy correction.

**Diagnostic controls (same output table, same readout).**

| Control | Changes | Isolates |
|---|---|---|
| C2-BINQ | each ray's W replaced by the mean over the 8 rays of its incident bin (same quadrature samples) | incident-direction information inside the bin (scattering quadrature) |
| C2-ENC | L₁ from ECRO's own 16-point encoding s | source projection at the first event (identical input to ECRO) |
| C2-RAYSEND | first gather reads the exact once-scattered radiance at each ray's hit point | sender projection of the 1-event family (light → panel → receiver) |
| C3 | sender-side deposition: per order 650,240 rays drawn in proportion to emitted power, traced, deposited at the receiver's dense incident cell | receiver-side sampling placement. This is **not** an identical-state comparison: there is a separate relation per order and per source |
| POOL | T_d averaged over 16 independent state realisations (8,192 rays per patch) | convergence diagnostic. Not matched budget and not a candidate |
| TF\<b\> | path-traced 2–5-event table at b samples per slot (`pt86.Ctx.field`), with independent seeds and the full fill cost counted | is the speed-up the table's or the learning's? |
| PT / PT_RR | the frozen-ER path tracer (re-measured), and the same tracer with Russian roulette before vertices ≥ 3 (survival min(0.95, max throughput)) | physical full-path comparators |

## Prospective rules (summary; full text in the protocol)

- **Stage 0 gate.** These tests must all pass, otherwise `STOPPED_INVALID_COMPARISON`:
  - WL86 recomputation and re-render readout identity;
  - ER reciprocity; the MIS estimator against an independent uniform estimator;
  - Lambert normalisation; q against analytic irradiance;
  - torch against a numpy loop; zero / no-visibility, linearity, non-negativity, furnace;
  - state identity under null edit, move-and-return and selective = full;
  - realisation reproducibility; RR expectation; seed disjointness.
- **Stage 1.**
  - Measures: WL86 E_H with the WL86 thresholds (F1 stationary ≤ 0.03 with capture ≤ 0.25; F2 A / C / D ≤ 0.06 at T1; F3 slope 0.75–1.25 and correlation ≥ 0.8 in D; F4 unseen sources ≤ 0.04).
  - Also: field-level and per-order errors against path-traced truth, and the region-C reading of the light → panel → Lambert family (WL86A definitions).
  - Classification: NUMERICALLY_INVALID / ACCURATE_UNDER_FROZEN_REPRESENTATION / DISCRETISATION_LIMITED / INSUFFICIENTLY_RESOLVED.
- **Stage 2.**
  - 16 independent realisations (seeds 87201–87216), same rule and budget as the deployed state, plus the deployed state.
  - Per realisation: E_H and signed error per region, T1 − T0 change, panel hit counts, missed important relations, BIN / RAY family readings. Mean, SD and 95 % t interval.
  - Classes:
    - A (systematic): the pooled-state error keeps ≥ 50 % of the realisation-mean error, and that mean differs from 0.
    - B (variance): realisation SD ≥ |mean| and the pooled error ≤ 50 % of it.
    - C (receiver sampling): C3 halves the region-C signed error in ≥ 2 of 3 family cases.
    - D: a mixture of these.
- **Stage 3.**
  - Measured alone on the GPU. Ladders 4–16,384 spp × 3 replicates for every arm at T1.
  - Quality: relative L1 against the pooled REF-ER on stationary pixels at 0.10 / 0.05 / 0.03 / g. g is evaluated only if the pooled reference noise is ≤ g / 2.
  - WL86 region rule: region-matched if A, C, D are each ≤ 1.1 × the path tracer's error. Per-region time-to-quality is reported for C and D.
  - Matching uses log-log interpolation only, with no extrapolation.
  - Per-edit, source-only, cold and memory costs are measured, with amortisation over V = 1 / 4 / 16 frames.
- **Untouched condition E3.**
  - E3 is the first of three predeclared candidates whose geometry-only masks give ≥ 1,000 pixels in A, C, D and E, and whose geometry passes the WL86 admissibility checks.
  - No radiance of any candidate existed before the choice.
  - E3 needs new references (8 × 8,192 spp REF-ER and REF-P, seeds 87100+ / 87120+) and the WL86 admission criteria Q1–Q4, Q6, Q7. If E3 is not admitted it is reported, and the batch does not stop.
  - The dev smoke used a plumbing-only stand-in (E3dev: E1 geometry, another light) so that no E3-candidate radiance existed before the freeze.

## Pre-freeze development (non-evidence)

All development ran on LabServer63 in `~/rest_gm187dev`. It used a dev geometry from the WL86 dev split (seed 86300) with a dev light, or the plumbing-only E3dev stand-in. No E1 / E2 result value was read. No radiance of an E3 candidate was computed.

1. **ER sampler consistency.** Three independent directional-albedo estimators agree for B, S1 and S2 at three angles: uniform hemisphere, ER importance sampling and the MIS mixture.
   - The first per-bin test failure came from the MIS estimator's own noise at the dev sample count (about 64 samples per bin).
   - The test was rewritten to include both estimators' split-half noise before any evidence existed.
2. **Lazy-scatter defect, again.** The first GPU-resident q table came out 1.9× too large on every class. The cause was Dr.Jit 0.4.6 scattering unevaluated expressions (the WL84 "lazy scatter 2×" defect). Values and indices are now evaluated before every scatter.
3. **Cost of q.** The first q implementation (numpy round trips) took 0.85 s per call. The GPU-resident one takes 25 ms. This was done so that the comparator is not handicapped by implementation overhead.
   - Per call, after the first: state 30 ms, q 25 ms, four-order recurrence 10 ms.
   - The material W tables take 0.4–1.3 s once per realisation (geometry-independent).
4. **Dev geometry against a 512-spp path-traced field** (flux ratio / relative L1, which includes truth noise):
   - EXP: floor 0.98 / 0.10, panel 1.01 / 0.09, walls 1.00 / 0.06, boxes 1.00 / 0.05.
   - Sphere: 0.98 / 0.16 with the flat-frame approximation, and 0.95 / 0.12 with the curved-patch rows (item 6).
   - Per order (walls), flux ratios are 0.99–1.00.
5. **End-to-end dev smoke of the chain** (low spp, E3dev). Every stage ran and the visuals were produced. One analysis crash on the dev stand-in's geometry lookup was fixed.
6. **Independent code review** (one subagent, before the freeze). The following were fixed before the freeze:
   - **Curved sphere patches** (above). Before the fix, W used the patch-centre frame for the sphere rays: wrong cosine and lobe orientation up to about 40° from the local frame, and no local horizon. The oracle field has no such approximation.
   - **Common random numbers.** The table renders used 64-spp launches against ECRO's 32-spp launches, so only half the samples were shared. Renders now use 32.
   - **Verdict mapping.** It was documented but not coded; it is now implemented (`analyze87.py verdict`).
   - **Analysis rules brought in line with the protocol:**
     - INSUFFICIENTLY_RESOLVED requires every failure to be within 1.5 × noise.
     - DISCRETISATION_LIMITED requires the localisation evidence; otherwise FAILING_UNLOCALISED.
     - The receiver-sampling class uses the rendered errors.
     - The systematic class requires the same sign.
     - The capture criterion is included in the realisation pass fraction.
     - The region errors of a first ladder point are kept.
     - C3 GPU time uses wall time as a proxy.
     - Second-view amortisation is reported.
     - Per-realisation quadrature seeds are used.
   - C2-BINQ was redefined as the in-bin mean of the per-ray weights (same samples) instead of a separate dense table.
