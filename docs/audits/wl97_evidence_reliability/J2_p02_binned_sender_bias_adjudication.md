# WL97 Phase 2 - Track J2: adjudication of P0-2 (binned M64 / 64-bin sender bias)

Date: 2026-11-16. Independent adversarial reviewer report; kept verbatim. No existing file was modified.

Own files:
- `experiments/audit97/J2/j2_flux_sign.py` (numpy only; imports Track A's ER port `experiments/audit97/A/er_np.py` read-only, own quadrature).
- `results/audit97/J2/j2_flux_sign.json`, `results/audit97/J2/j2_flux_sign.log` (run time 279 s, local CPU).

## 1. The decisive question and the check

Track A (A-03, A-04) says about half of the binned over-delivery to box A is the solid-angle-mean averaging convention of the table. E2 / E3 attribute it to outgoing-direction binning as such, and E3-02 additionally states that "the ideal table conserves the panel's emitted flux by construction". These cannot all be true.

**Mechanism (code-level).** The table holds one radiance per (patch, outgoing bin). The 4 x 16 bins are uniform in z = cos(theta), so the exact integral of cos over a bin equals ZB * DOMEGA (`er_np.py:84-87`, E3 `ana97.py:6-8`). The flight (WL95A, A, E2, E3) multiplies the table value by the exact sender cosine of each ray, so the flux a table entry emits is q * ZB * DOMEGA. Two piecewise-constant projections are possible:
- `q_sa = (1/DOMEGA) int_bin f dOmega`: the solid-angle mean (quad87 qtable / pt86 field / M64 convention). It is exact for the bin-averaged radiance. Its emitted flux is ZB * int_bin f, which differs from int_bin f cos whenever f is not uniform in z inside the ring.
- `q_cos = int_bin f cos / int_bin cos`: the cosine-weighted (projected-solid-angle) mean. It is the unique piecewise-constant value that conserves each bin's flux under the exact-cosine flight.

So the solid-angle-mean table is flux-conserving only for lobes that are symmetric in z about the ring centre. A grazing lobe at z of about 0.05 inside ring 0 (z in [0, 0.25], ZB = 0.125) is over-emitted. This is arithmetic, not a numerical question; only the magnitude needs checking.

**Decisive check (J2, independent quadrature).** A dense uniform (z, phi) midpoint grid (1,024 x 8,192 per incident direction), and a separate far-field computation of the whole CK2 S1 panel intensity (42 patches x 16 sender points, tables from a 48 x 48 in-bin grid, 256 x 1,024 output directions). This shares only the ER port with Track A; it does not use A's log-azimuth grid.

| quantity | J2 | Track A | E3 |
|---|---|---|---|
| CK2 panel emitted flux, solid-angle-mean table / physical | **1.295** | 1.298 | not computed ("conserves by construction" asserted) |
| CK2 panel emitted flux, cos-weighted table / physical | 1.000 (by construction) | 1.000 | - |
| table flux M64 / q_sa | - | 0.940 | 175.9 / 188.0 = 0.936 |

Single incident directions (S1 material, flux of a solid-angle-mean table / physical, RGB):
cos_i 0.025: 1.79 / 1.41 / 1.57; 0.047: 1.67 / 1.33 / 1.44; 0.10: 1.30 / 1.14 / 1.17; 0.154: 1.06 / 1.04 / 1.04; 0.40: 1.00 / 1.01 / 1.01; 0.80: 1.02 / 1.02 / 1.02. The ring-0 ratio alone is 1.95 to 1.76 at grazing; rings 1 to 3 carry +3 to +9 %.

**Sign / receiver dependence (far field).** With the cos-weighted table the total panel intensity is exact, yet the table is below the exact intensity in 63 % of the exact-intensity-weighted directions and above it in 37 %. At the direction of peak exact intensity (local z 0.09, the grazing glint) the table gives 0.12x (cos) or 0.19x (solid-angle mean). Exact-weighted 10 / 50 / 90 % quantiles of the table / exact ratio are 0.37 / 0.78 / 1.93 (cos) and 0.49 / 0.96 / 2.58 (sa). Box A lies 0.52 m from the panel centre, off the glint (centre bin 56, ring 3; it gathers 30 / 44 / 22 / 4 % of its exact irradiance from sender rings 0-3, Track A). The table over-delivers to box A because box A sits in the lobe tails. A receiver in the lobe core would be under-delivered by a factor of several. Far-field is a principle check; box A itself is in the near field.

**Ruling on the tension.**
- (i) Track A's counterfactual is computed correctly (flux ratios 1.298 and 1.000 reproduced independently; its code applies `q_cos = sum El * int f cos / WT` with `WT = ZB * DOMEGA = int_bin cos`, `a3_ck2_delivery.py:80`). It is a legitimate alternative projection of the same 64 numbers per patch: flux-exact per bin under ECRO's own exact-cosine flight / gather, passive whenever the BRDF is (row albedo = the physical albedo), and requiring nothing else of the receiver. It does not repair the camera reading: a camera looking into the glint sees 0.12x (cos) instead of 0.19x (sa), and neither convention gives pointwise radiance. It is therefore an alternative, not a correction; neither convention is "the" physical table.
- E3-02's claim "the ideal table conserves the panel's emitted flux by construction" is **wrong**. E3's table is the solid-angle mean (`quad97.py:16`), and its 188.02 emitted flux is about 1.30x the physical flux. E3's numbers are right; only that interpretive sentence is wrong. E2 / E3 and A do not contradict each other numerically. They decompose along different axes: E2 / E3 along the pipeline (incident / position / outgoing); A splits the outgoing step into convention and in-bin redistribution.
- (ii) Decomposition of the box-A excess over the exact oracle (Track A, ns5_np40_nr3, confirmed by E3's bindir 1.397 and E2's 1.412 at 4 x 16):
  - **solid-angle-mean convention: 1.396 -> 1.191, about 52 % of the excess**. It is grazing-specific: the flux ratio is 1.00-1.02 at cos_i >= 0.15-0.4.
  - **in-bin redistribution of a lobe narrower than a bin, intrinsic to any flux-conserving 4 x 16 piecewise-constant outgoing table: about +19 % at box A**. Its sign is receiver-dependent (J2 far field).
  - **resolution**: at 8 x 32 the solid-angle-mean per-ray ideal table already gives 1.024 (E2-F9); both parts shrink with finer rings.
  - patch-position averaging: about 0 in total (E3-03); incident binning +5-7 points (E2-F8, E3-03).
  - These shares are not additive invariants. They hold for this receiver, this light and this material.

## 2. Rulings

| id | ruling | short reason |
|---|---|---|
| E2-F1 | UPHELD | Independent chain reproduces deployed arrays to 1e-6; consistent with F0-10 and C2 recomputations. |
| E2-F2 | UPHELD | Pooled 1.53 (4,096 draws) agrees with A (1.49 for M64(s), deterministic), E3 (1.484) and WL95A (1.544 +- 0.035). |
| E2-F3 | UPHELD | Plateau at 0.506; deterministic oracles (A, E3) need no relation sampling at all. |
| E2-F8 | UPHELD_WITH_MODIFICATION | The chain is right, but "outgoing direction +0.356" contains about 0.2 of solid-angle-mean over-emission (convention) and about 0.19 of in-bin redistribution. It is not "binning as such" alone. |
| E2-F9 | UPHELD | 4 x 16 specific; 8 x 32 gives 1.024. Its ideal table uses the solid-angle-mean convention, so finer rings remove both parts. |
| E2-F10 | UPHELD | e2_p is trustworthy at its stated noise; three independent oracles (E1, E2, E3/A) agree (0.985-1.015). |
| E3-01 | UPHELD | Deterministic oracle = reference within noise (1.015 / 0.068, n0 0.059). |
| E3-02 | UPHELD_WITH_MODIFICATION | The deterministic over-delivery is upheld. The sentence "ideal table conserves the panel's emitted flux by construction" is REFUTED: it emits 1.30x physical (J2 1.295, A 1.298). Roughly half the excess is that over-emission. |
| E3-03 | UPHELD_WITH_MODIFICATION | "bindir" carries the whole excess; but "outgoing-direction binning" here means solid-angle-mean binning, about half of which is the convention. |
| E3-07 | UPHELD_WITH_MODIFICATION | No longer purely untested. J2 measured that the sign is receiver-dependent (0.12-0.19x in the glint core, over-delivery in the tails) and that the convention part vanishes for non-grazing incidence. WL96 scratch (line 65) measured a smaller binned bias for other lights (0.05-0.22 rel L1). Magnitude and sign are configuration-specific; existence for sub-bin lobes is general. |
| E3-09 | UPHELD | Code reading and an oracle that does not use the path tracer agree. |
| E3-10 | UPHELD | Deployed draw at the 13-22nd percentile; WL95A itself disclosed this (worklog :136, roadmap). Not material for the bias. |
| C2-02 | UPHELD_WITH_MODIFICATION | Systematic and sampling-independent: yes. "Cause is the outgoing 64-bin projection" should read "the solid-angle-mean 4 x 16 outgoing projection (about half convention, half in-bin redistribution on box A)". |
| C2-03 | UPHELD | The FLIGHT_ATTRIBUTION label rule mixes a single-draw L1 with a pooled L1. The label is draw-dependent (22 / 30 / 12 of 64), as stored in `results/audit97/C2/wl95a_verdict_sensitivity.json`. Bias conclusion unaffected. |
| A-03 | UPHELD_WITH_MODIFICATION | M64 = accurate bin average; not flux-conserving at grazing (1.30x / 1.22x verified). Verdict INVALID_COMPARISON applies only to statements that read distance-to-M64 as physical error (WL94D teacher "exact sender", WL92B-WL94D substitutions on grazing CK2). It does not apply to WL95A B2, which compares against the path-traced e2_p. |
| A-04 | UPHELD_WITH_MODIFICATION | Computation correct and reproduced; the cos-weighted table is a legitimate alternative (flux-exact, passive), not a correction (camera reading still 0.12x in the glint). "About half" holds for box A only; the residual's sign depends on the receiver. |
| F-MAP-06 | UPHELD | `pt86.py:114-127` broadcasts the same K offsets and PSRC source points to every patch (verified). Pre-WL95A single-draw comparisons are draw-specific. |
| F-MAP-07 | UPHELD_WITH_MODIFICATION | M64 as "exact sender" is biased only for grazing outgoing lobes and sub-bin lobes. On non-grazing workloads WL93C measured M64(flight(q)) vs path-traced t2 at rel L1 0.064 (E2 T1, `93C_WORKLOG:84`). So the dependency is material for CK2-type grazing evaluations (WL94D CK2 teacher / flight floor, WL95 rank bounds vs M64) and minor for E2 / val-geometry table conclusions. |
| F0-10 | UPHELD | Reproduction from raw draws; consistent with all deterministic oracles. |
| E1-F2 | UPHELD | Independent q, M64 and box-A reference; M64 4.5 % grazing-ring residual vs own operator is consistent with the convention sensitivity at ring 0. |

## 3. P0-2 audit verdict

**P0-2 "The M64/64-bin sender introduces systematic sharp-delivery bias that is not eliminated by relation sampling": CONDITIONALLY_VALID.**

It holds under these restrictions:
- **Scope.** CK2 'none': scene86 room, one point light 5 cm in front of the S1 panel (incidence cos 0.025-0.15, all incident ring 0), GGX alpha 0.026-0.075, receiver box A (Lambert), exactly-2-event S1-mediated delivery.
- **Table.** The patch x 4 x 16 outgoing table with the solid-angle-mean bin convention (M64(s), q256).
- **What holds there.** The expected over-delivery is +40-50 % (deterministic 1.39-1.47x vs the exact oracle; 1.42-1.54x vs the reference). It is independent of relation sampling (VERIFIED by three deterministic oracles and 4,096 draws).

It does not hold, or is not shown, beyond that:
- (a) The magnitude is **not** a property of "64 bins" alone. About half of the box-A excess is the solid-angle-mean convention's grazing over-emission (+30 % panel flux). A flux-conserving cos-weighted table on the same 64 bins gives about +19-21 %.
- (b) The sign is receiver-dependent. The table under-delivers severely (0.12-0.19x) to receivers in the glint.
- (c) At 8 x 32 bins the per-ray ideal table is within 2-6 %.
- (d) It is not tested for other scenes, lights, materials or receivers beyond WL96 scratch indications.

The general property that is justified: a fixed 4 x 16 piecewise-constant outgoing table cannot represent a lobe narrower than a bin, so it smears it; the delivered error is large and has a receiver-dependent sign. The solid-angle-mean convention additionally over-emits grazing lobes.

## 4. Affected statements

1. **WL95A `95A:146`** "binned patch x bin sender table ... carries a systematic over-delivery to box A of about +40-50 % in expectation".
   - Defect / limit: The magnitude is specific to the solid-angle-mean convention, the grazing light and the box-A placement.
   - Corrected interpretation: Systematic, sampling-independent bias on CK2 box A. About half is convention; the flux-conserving remainder is about +20 %. The sign is receiver-dependent.
   - Dependents: FLIGHT_ATTRIBUTION; WL96 "carrier property"; roadmap WL95A entry.
   - Action: QUALIFY.
2. **WL95A `95A:149`** [INF] "The loss is the patch-constant and bin-constant outgoing representation of a sharp, position-dependent lobe".
   - Defect / limit: Patch-constant position contributes about 0 to the total (E3-03, E2-F8 +0.04). The bin-constant part splits into convention plus in-bin redistribution.
   - Corrected interpretation: "The loss is the bin-constant (solid-angle-mean) outgoing representation of a lobe narrower than a 4 x 16 bin; position averaging is minor for the total."
   - Dependents: WL96 mechanism statements.
   - Action: QUALIFY.
3. **WL95A `95A:167`** FLIGHT_ATTRIBUTION = ANGULAR_PROJECTION_EVIDENCED.
   - Defect / limit: The label rule uses a single deployed draw; it is draw-dependent (22 / 30 / 12 of 64; C2-03). The label term "angular projection" includes the convention.
   - Corrected interpretation: The bias evidence stands (pooled / deterministic). The label as a rule outcome is draw-specific.
   - Dependents: roadmap WL95A.
   - Action: QUALIFY.
4. **WL95A `95A:155`** "evidence about the patch x 64-bin carrier for a grazing sharp lobe ... not a statement about other bin layouts".
   - Defect / limit: Correctly scoped. It omits the convention.
   - Corrected interpretation: Add "with the solid-angle-mean convention".
   - Action: PRESERVE (with a note).
5. **WL96 `96:43-46`** "binned patch x bin sender table has an expected-value delivery bias that relation sampling does not remove".
   - Defect / limit: Measured correctly; same scope limits.
   - Action: PRESERVE.
6. **WL96 `96:60`** "an exact rank-16 copy of M64 inherits M64's +42-54 % binned-outgoing bias" (makes RANK16 "moot for design").
   - Defect / limit: The magnitude is convention- and receiver-specific. A flux-conserving 64-bin target would carry about +20 % on box A.
   - Corrected interpretation: "Inherits M64's convention-specific CK2 box-A bias." Mootness relative to M64 stands; it is not a property of every 64-bin target.
   - Action: QUALIFY.
7. **WL96 `96:62-65`** "the dominant mechanism is outgoing-direction quantisation".
   - Defect / limit: True, provided quantisation includes the solid-angle-mean over-emission (about half on box A).
   - Action: QUALIFY.
8. **Roadmap WL96 entry** "the binned bias is a carrier property reproduced by two independent deterministic oracles".
   - Defect / limit: "Carrier property" overgeneralises. It is a property of the 4 x 16 solid-angle-mean table for sub-bin grazing lobes at this receiver. The sign is receiver-dependent; it disappears at 8 x 32.
   - Action: QUALIFY.
9. **Roadmap WL95A entry** "expected over-delivery 1.54x".
   - Defect / limit: Scope missing (box A, CK2 grazing, solid-angle-mean M64(s16)).
   - Action: QUALIFY.
10. **WL96A `96A:173`** "unlike the 1.5x over-delivery of the grazing e2_p in WL95A".
    - Defect / limit: Correctly scoped to grazing e2_p.
    - Action: PRESERVE.
11. **WL94D** M64 teacher as "exact sender"; the CK2 flight floor; "over-emits (1.75 / 1.71) ... delivers more than the exact sender (1.28 / 1.19)" (`94D:35,64,146`).
    - Defect / limit: On CK2 the "exact sender" over-emits about 1.22x physical and over-delivers 1.21x (deployed) / 1.54x (pooled) vs the reference.
    - Corrected interpretation: These are distances to a biased binned operator. NO_RELIABLE_IMPROVEMENT is valid as measured but does not test physically correct sender supervision. Over-emission "vs exact" understates the physical over-emission (about 1.75 x 1.22).
    - Action: QUALIFY.
12. **WL95 / WL95A** rank-16 bounds measured against M64 (RANK16_DELIVERY "0.63x / 0.16x of M64's flux").
    - Defect / limit: The target itself over-emits at grazing.
    - Corrected interpretation: The flux ratios are relative to a 1.22x-over-emitting target. The "peak <= 0.02" footprint failure is unaffected.
    - Action: QUALIFY.
13. **WL92B** X_M1 / X_M12 / X_Q substitution attributions ("exact q closes 96 % of the seed gap"); **WL93C** ORA_s / R* oracle; **WL93D** "wrong vs frozen-ER M64"; **WL94 C2_srcS1**.
    - Defect / limit: The M64 / q tables use the same solid-angle-mean convention. The bias is material only for grazing / sub-bin lobes. WL93C measured M64(flight(q)) vs t2 at 0.064 on E2 T1.
    - Corrected interpretation: On E2 / val geometries these are distances to a near-physical operator. Their CK2 (grazing) parts are distances to a biased one. All are also single deployed-draw estimates (F-MAP-06).
    - Action: QUALIFY (CK2 parts); PRESERVE (E2 / val parts, pending draw sensitivity).
14. **Reference e2_p (WL94 CKF_none)** trustworthy.
    - Corrected interpretation: Confirmed by three independent oracles; the only shared component is the ER closure and the scene geometry.
    - Action: PRESERVE.

## 5. Unresolved

- Box-A-like delivery for other lights, receivers and scenes was not measured here. J2's sign result is far-field only.
- The convention share for WL93C / WL94 non-grazing table metrics was not recomputed (expected small from the single-direction ratios 1.00-1.02 at cos >= 0.4).
- Which convention the downstream ECRO receiver / camera read should use is a design question, outside this audit.
