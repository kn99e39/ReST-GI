# WL97 audit, track E3: deterministic reproduction of P0-2 and P0-3 (no relation sampling)

Independent reviewer report. Date 2026-11-16. Repository HEAD efcf86a (branch Selective_Recompute). Nothing outside
`docs/audits/wl97_evidence_reliability/`, `experiments/audit97/E3/`, `results/audit97/E3/` and `~/rest_x97/E3/` (LabServer63) was created
or changed. Nothing was trained. No file of another audit track was opened.

Continuation note: a first attempt at this track was interrupted before reporting. Its scripts and server outputs were continued, not
trusted: the decisive numbers were re-run, and one bug of that attempt (my own code, not audited material) was found and fixed, see section 7.

## 1. Scope and method

Claims under audit (CK2 scene, variant `none`, point light at [-1.45, -0.6, 1.7], rgb 30):

- P0-2: "The M64 / 64-bin sender introduces systematic sharp-delivery bias that is not eliminated by relation sampling."
- P0-3: "Continuous exact-direction and hit-point ER evaluation substantially closes the CK2 delivery error."

WL95A measured both with Monte Carlo relation draws. WL96 quoted deterministic designer oracles (ratio 1.015, rel L1 0.068, n0 0.059;
binned M64 1.486 / 0.487). I did not read or run those scripts, nor `a95.py` as code to reuse (I read `a95.py` lines 20-136 and 396-412
only to learn the metric and arm definitions that I had to match).

Quantity: the exactly-2-event panel-mediated exitance on the 54 box-A patches (1130..1183): light -> S1 panel (ER) -> Lambert box A
(albedo 0.5) -> exitance. Computed by deterministic midpoint quadrature in area form,

    E_j = (1 / A_j) int_{A_j} dA_r int_{panel} dA_s  L(x_s -> x_r) cos_s cos_r / r^2 V,   exitance_j = 0.5 E_j,

with own code: `geo97.py` (geometry from the fixture JSON files), `er97.py` (float64 port of the ER closure, written from the formula in
the docstring of `closure83_dr.py`), `quad97.py` (quadrature, float64 torch). Five sender models share the same quadrature nodes:

| model | sender radiance L(x_s -> x_r) |
|---|---|
| `exact` | ER at the exact light direction and distance of x_s, exact outgoing direction x_s -> x_r |
| `anchor` | exact outgoing direction x_s -> x_r, but light direction / cosine / distance taken at the centre of the 20 cm panel patch of x_s (this is WL95A's X1c definition, `a95.py:90-92`) |
| `bindir` | exact sender position, outgoing radiance averaged over the 64-bin cell that contains the outgoing direction (angular binning alone) |
| `qbin` | patch x 64-bin table of the true once-scattered radiance (patch average and bin average): what an ideal table of q delivers |
| `m64` | M64(s) built from ER by me: s = patch-average irradiance per incident bin, Mq = incident-bin average of the outgoing-bin average of f (definition of `quad87.py:240-242`, not its code or its tables) |

Reference: `e2_p` of `~/rest_x94/results/fields/CKF_none.npz` (sha256 3a6238a8a42d1db4, 2 halves x 4096 spp, seeds 94301 / 94302), mean of
the two halves, reduced to per-patch exitance. Metrics as WL95A (`a95.py:397-405`): rgb-summed exitance per patch; ratio = sum(arm) /
sum(ref); abs rel L1 = sum|arm - ref| / sum(ref); Pearson corr; n0 = 0.5 sum|ref_a - ref_b| / sum(ref).

## 2. Facts verified before measuring

All on LabServer63 against a private copy of the frozen code (commit a1718f7; sha256 prefixes pt86.py 912e68ac, scene86.py b75a2d9c,
closure83_dr.py 3eee5909, mats86.npz d3c5be85, scene_ck2.json 977ba122, ecro86.json ecca6724). Outputs `val_dr.json`, `cmp_er.json`, `tabtest.json`.

| check | result |
|---|---|
| Geometry (`T_GEOM`) | my patch origins / edges / normals for patches 1088..1183 equal `scene86.Geom` exactly (max abs err 0.0); panel PMAT = 1 (S1), box A PMAT = 4 (Lambert 0.5) |
| Shading frame (`T_FRAME`) | Mitsuba local frame on the panel is (s, t, n) = (+y, +z, +x); box-A faces resolve to shape 7 |
| ER convention (`T_COS`) | `DrClosure83.eval(wi, wo)` = f * cos(SECOND argument): for the Lambert state it equals rho / pi * cos(wo) with relative error 0.0; the first-argument hypothesis fails (max rel 952). In `pt86a.field_src` the second argument is the light direction (`pt86a.py:131`), so the cosine is the light cosine at the sender. I use the same |
| Reciprocity (`T_SYM_S1`) | f of S1 is symmetric: median rel 5e-8, max 1.2e-5 (float32) |
| ER port (`cmp_er.json`) | my float64 port vs production closure, 3 direction families x 4 materials, 1.05 M pairs each: max rel 1.6e-6, sum ratio 1 +- 6e-8 |
| Visibility (`T_VIS`) | light -> panel: 0 of 2,097,152 rays occluded (light cosine 0.024..0.164, mean 0.047: all light arrives in incident ring 0). Panel -> box A, 2,097,152 pairs per face, production ray test vs my analytic rule V = [cos_s > 0][cos_r > 0]: the only mismatches are 483 pairs with cos_r <= 1e-3 (weight share 2e-6) and 758 pairs whose path inside box A is shorter than the 1e-4 m ray epsilon. No other occluder lies between light, panel and box A |
| Bin weights | derived from `pt86.Ctx.dir_in_bin` (`pt86.py:195-200`): a table entry is a mean over directions uniform in solid angle inside a bin (z = cos theta uniform in ring [iz/4, (iz+1)/4], 16 sectors). Hence exitance = sum_b T[j,b] * int_bin cos dOmega with int_bin cos dOmega = (2 pi / 16) ((iz+1)^2 - iz^2) / 32. This equals WL95A's W_T (ring midpoint x 2 pi / 64) exactly and sums to pi. My reduction of the reference equals WL95A's stored `ref_a` / `ref_b` to 2e-16 per patch |
| Bin-average quadrature (`tabtest.json`) | half-vector quadrature vs brute-force (z, phi) grid for S1 at six CK2 light directions: bin L1 6.5e-4 at 384 x 256, 2.1e-4 at 768 x 512. Caveat: for a pure Lambert state the same grid is poor in single bins at grazing (max bin error 5.5 %); the S1 result and the delivery-level doubling test below are the relevant evidence |

## 3. Results

### 3.1 Exact oracle versus the path-traced reference

Finest run `s4r24` (4 mm sender grid = 105,000 panel nodes, 24 x 24 receiver nodes per patch, 9.1e8 pairs).

| quantity | value |
|---|---|
| reference total (rgb-summed exitance, 54 patches) | 15.1311 (halves 14.7561 / 15.5061) |
| exact oracle total | 15.3580 |
| ratio | 1.0150 +- 0.0274 (SE of the reference total from the scatter of its 128 independent bin estimates per patch; the half difference gives +-0.025) |
| abs rel L1 | 0.0680 |
| reference noise L1 | n0 = 0.0593 (half split); 0.0624 (bin scatter, Gaussian); bootstrap over bins: mean 0.0665, 95th percentile 0.099 |
| corr | 0.9951 |
| panel-facing face only | ratio 1.0111, L1 0.0679 |
| peak | patch 5 in both; oracle 0.977x the reference there |

The oracle agrees with the path tracer within the reference noise. WL96's numbers (1.015 / 0.068 / n0 0.059 / corr 0.995) are reproduced
to three digits by independent code.

Where the residual sits: 70 % on the panel-facing face (which holds 71 % of the signal) and 26 % on the +z face (16.5 % of the signal).
Largest single deviations: patch 51 (+z face; ref 0.397 +- 0.044, oracle 0.550), patch 3 (0.761 +- 0.048 vs 0.886), patch 6
(0.738 +- 0.079 vs 0.931). These are the patches with the largest relative reference noise. A naive chi-square with bin-scatter SEs gives
79.4 / 36, but that statistic is not calibrated here: the reference's own half-versus-half test with the same SEs gives mean z^2 = 3.75
(median 0.67, sum-of-squares ratio 0.73), i.e. the per-patch SEs are underestimated in a few patches because the estimator is heavy-tailed
(a sharp lobe sampled by cosine sampling). I therefore do not read 79.4 / 36 as a mismatch. L1 against each half separately is 0.083 /
0.104 against an expected 0.084.

Self-check of the oracle that does not use the path tracer (`mc97.py`): a Monte Carlo estimate in solid-angle form (uniform receiver point,
cosine direction, analytic panel hit; 2.9e10 samples, seed 97001) gives total 15.3615 +- 0.0157; quadrature / MC = 0.99977; per-patch rel
L1 0.0027; rms z 1.08; max |z| 2.2. In particular patches 3, 6 and 51 are 0.8829 +- 0.0027, 0.9232 +- 0.0061, 0.5576 +- 0.0055: the
oracle is right there and the reference is low by noise. The area-form Jacobian, masks and convergence are thus verified to about 0.1 %
(total) and 0.3 % (footprint).

### 3.2 Sender models, same quadrature (finest run; `bindir` at 8 mm)

Against the path-traced reference (reference noise applies: ratio +-0.027, L1 floor about 0.06):

| model | total | ratio | abs rel L1 | corr | value at reference peak | peak patch (ref 5) |
|---|---|---|---|---|---|---|
| exact | 15.358 | 1.015 | 0.068 | 0.995 | 0.98 | 5 |
| anchor | 16.394 | 1.084 | 0.303 | 0.939 | 1.08 | 8 |
| bindir | 21.449 | 1.418 | 0.437 | 0.934 | 1.03 | 8 |
| qbin | 21.391 | 1.414 | 0.424 | 0.952 | 1.17 | 8 |
| m64 | 22.449 | 1.484 | 0.485 | 0.973 | 1.35 | 8 |

Against my exact oracle (no reference noise at all; this comparison does not involve the path tracer):

| model | ratio | abs rel L1 | corr | L1 after rescaling to equal energy | ratio on faces -x / -y / +z |
|---|---|---|---|---|---|
| anchor | 1.068 | 0.266 | 0.942 | 0.242 | 1.02 / 0.81 / 1.47 |
| bindir | 1.397 | 0.415 | 0.950 | 0.236 | 1.39 / 1.21 / 1.55 |
| qbin | 1.393 | 0.401 | 0.967 | 0.193 | 1.43 / 1.11 / 1.47 |
| m64 | 1.462 | 0.462 | 0.982 | 0.149 | 1.47 / 1.23 / 1.58 |

Reading:

- The binned sender over-delivers deterministically: +39 % (ideal table of q) to +46 % (M64(s)) of the exact value, footprint error 0.40
  to 0.46. No relation ray exists in this computation, so the excess is a property of the table content, not of relation sampling.
- Angular binning of the outgoing direction alone, with the exact sender position, already gives +40 % / 0.415 (`bindir`). Averaging the
  position over the 20 cm patch adds nothing to the total (1.393 vs 1.397). Binning the incident direction as well (M64(s)) adds about
  +7 points.
- The emission-anchor effect alone is a footprint error of 0.27 with a small net energy error (+6.8 %): it moves energy between faces
  (-y face 0.81x, +z face 1.47x) and moves the peak from patch 5 to patch 8.
- Only exact direction and exact point together reach the reference noise floor.
- Diagnostic: the ideal table conserves the panel's emitted flux by construction (my flux diagnostic 188.02), and M64(s) even emits 6.5 %
  less (175.9), yet both deliver about 40 to 46 % more to box A. The bias is a redistribution of the narrow lobe over the bin's solid
  angle into directions that reach box A. Its sign and size therefore depend on where the receiver sits relative to the true lobe.

### 3.3 Agreement with the audited numbers

| quantity | WL95A (pooled 64 relation draws, +-SE) | WL96 (designer deterministic) | this audit vs reference | this audit vs exact oracle |
|---|---|---|---|---|
| exact direction and point (X1) | 1.022 / 0.111 (0.041) / 0.99 | 1.015 / 0.068 / 0.995 | 1.015 / 0.068 / 0.995 | - |
| exact direction, patch centre (X1c) | 1.037 / 0.268 (0.045) / 0.95 | - | 1.084 / 0.303 / 0.94 | 1.068 / 0.266 |
| q table (q256) | 1.422 / 0.433 (0.033) / 0.94 | 1.420 / 0.428 | 1.414 / 0.424 / 0.95 | 1.393 / 0.401 |
| M64 (M64_spool) | 1.495 / 0.500 (0.034) / 0.96 | 1.486 / 0.487 | 1.484 / 0.485 / 0.97 | 1.462 / 0.462 |

Per-patch comparison of my deterministic values with WL95A's stored pooled arms (`results/ecro_attr95a/deliv.npz`, sha256 f2324a32effc80ca),
using WL95A's own per-draw scatter as SE: exact vs X1: total ratio 1.007, rel L1 0.098, rms z 0.90; anchor vs X1c: 0.957, 0.097, 1.02;
qbin vs q256: 1.006, 0.060, 0.96; m64 vs M64_spool: 1.008, 0.056, 0.95. All four agree within WL95A's Monte Carlo noise. I also recomputed
WL95A's table values from its stored draws and they reproduce (X1 1.0224 / 0.1106 / 0.989; M64_spool 1.4949 / 0.4996; q256 1.4221 / 0.4331;
X1c 1.0371 / 0.2677).

Consequence: WL95A's pooled X1 residual of 0.111 is Monte Carlo noise of the 64 pooled draws (0.098 against the noise-free oracle) plus
reference noise (0.068), not a remaining bias of exact evaluation.

### 3.4 Convergence evidence

Totals (exitance, rgb-summed). Midpoint rule throughout.

| run | sender grid | receiver grid | exact | anchor | qbin | m64 |
|---|---|---|---|---|---|---|
| s16r12 | 16 mm | 12 x 12 | 15.3571 | 16.0034 | 20.9872 | 21.9848 |
| s8r12 | 8 mm | 12 x 12 | 15.3570 | 16.3958 | 21.3874 | 22.4457 |
| s4r12 | 4 mm | 12 x 12 | 15.3581 | 16.3957 | 21.3934 | 22.4516 |
| s2r12 | 2 mm | 12 x 12 | 15.3585 | 16.3956 | 21.3920 | 22.4502 |
| s4r6 | 4 mm | 6 x 6 | 15.3588 | 16.4000 | 21.3864 | 22.4441 |
| s4r24 | 4 mm | 24 x 24 | 15.3580 | 16.3944 | 21.3906 | 22.4488 |

- Exact: totals stable to 1e-4 from 16 mm down; largest per-patch change (27 patches above 1 % of the peak) versus s4r24 is 6e-4 at
  s8r12, 3e-4 at s4r12 and s2r12, 1.6e-3 at s4r6. Receiver refinement 6 -> 12 -> 24 changes the total by 7e-4 then 1.6e-4 (second order).
- Anchor and tables need 8 mm or finer (they are discontinuous at patch borders): the 16 mm grid is off by up to 5 % per patch; from 8 mm
  on, per-patch changes are below 1e-3.
- Table construction: 32 x 32 instead of 16 x 16 points per patch and 32 x 32 instead of 8 x 8 incident sub-cells (`tables_c`) changes
  qbin by 3.6e-4 and m64 by 2.7e-4 in total (max per patch 1.9e-3 / 5.6e-3). Doubling the half-vector quadrature (`tables_b`, 768 x 512)
  changes qbin by 6e-6 and m64 by 3.2e-4 in total (max per patch 1.2e-4 / 7.6e-4). `bindir`: 21.4431 (384 x 256) vs 21.4429 (768 x 512) at 16 mm; 21.4491 at 8 mm.
- Independent estimator: the 2.9e10-sample solid-angle Monte Carlo of section 3.1 (0.02 % total, 0.27 % L1).
- Two runs with deliberately off-centre nodes (`s8r12o1`, `s8r12o2`; a first-order rule, not convergence evidence) move the exact total
  by +-1.5 %; the midpoint ladder and the Monte Carlo are the evidence used.

All numerical errors are at or below 0.5 %, against claimed effects of 40 to 48 % (ratio) and 0.27 to 0.49 (L1).

### 3.5 Is any part of the "bias" explained by the reference?

No, on two independent grounds.

1. Numerical: the +39 to +46 % and the 0.40 to 0.46 footprint error appear against my own oracle, which does not use the path tracer.
   The oracle in turn matches the path tracer at 1.015 +- 0.027 and matches an independent Monte Carlo at 0.9998.
2. Code reading of the reference generator (`mat94.py:289-299` -> patched `pt86a.field_src`, `pt86a.py:73-198`, patch `mat94.py:126-139`):
   - Clamping: none on radiance. The only `maximum` calls are denominators and an 'ang'-mode cosine that the 'true' mode does not execute.
   - kmax: `e2_p` collects k = 2 only (`pt86a.py:176-181`); it does not depend on kmax.
   - 2e-3 m probe offset: the probe ray starts at p + 2e-3 w and must return to within 1e-3 m of p (`pt86a.py:96, 109`). It only gates
     acceptance at k = 1 and the count that normalises the bin mean; it does not displace the receiver point. Box A touches nothing.
   - Per-patch material lookup at k = 1: the patch forces PMAT[pid0] when the hit is on the probe patch's own shape. In variant `none`
     every box-A patch is material 4 and every panel patch material 1 (checked in the stored PMAT), so the lookup cannot change anything.
   - Cell rounding: receiver patches are assigned by construction (slot -> pid0), not by `patch_bin`; the panel split uses `mid == 1`,
     uniform over the panel.
   - Estimator: cosine sampling at the Lambert receiver, next-event estimation at the panel with the exact light direction and a shadow
     ray. It is unbiased for the quantity I integrate, but heavy-tailed: 524,288 samples per patch leave 2.7 % noise on the total.

What the reference does limit is resolution: agreement of oracle and path tracer cannot be established below about +-5 % (total, 2 SE) or
about 0.06 (L1) from the stored reference. WL96's own caveat ("up to about 0.03 of unresolved oracle bias") is fair. My Monte Carlo removes
the quadrature side of that doubt; an error of a few per cent in the production path tracer itself would remain undetected, and would not
change either verdict.

## 4. Findings

| id | category | material | claim | original number | independent number | evidence | decision-level impact | proposed verdict | confidence |
|---|---|---|---|---|---|---|---|---|---|
| E3-01 | confirmation | yes | P0-3 (and WL96 "oracle validated") | WL96: ratio 1.015, rel L1 0.068, n0 0.059, corr 0.995 (`96_WORKLOG...md:45`) | 1.0150 +- 0.027, 0.0680, 0.0593, 0.9951; quadrature vs own MC 0.9998 / L1 0.0027 | `results/audit97/E3/ana_s4r24_s8r12.json`, `mc_l23b32s97001.npz` | The deterministic exact oracle is a sound reference-consistent evaluation on CK2; the WL96 statement stands | VERIFIED | high |
| E3-02 | confirmation | yes | P0-2 | WL95A pooled M64_spool 1.495 / 0.500 (0.034), q256 1.422 / 0.433 (`95A_WORKLOG...md:80-81`); WL96 1.486 / 0.487, 1.420 / 0.428 (`96_WORKLOG...md:46`) | m64 1.484 / 0.485, qbin 1.414 / 0.424 vs reference; 1.462 / 0.462 and 1.393 / 0.401 vs noise-free oracle | `conv_models.json`, `ana_s4r24_s8r12.json` | The over-delivery exists with zero relation sampling, so more or better relation samples cannot remove it | VERIFIED | high |
| E3-03 | confirmation | yes | P0-2 attribution ("64-bin sender") | WL95A: "q256 -> X1c is outgoing angular + average-to-centre" (`95A_WORKLOG...md:152`), not separated | exact position + binned direction: 1.397 / 0.415; adding patch averaging: 1.393 / 0.401; adding incident binning: 1.462 / 0.462 | `deliv_s8r12.npz` (`E_bindir`), `conv_models.json` | Outgoing-direction binning alone carries the over-delivery; position averaging is not the cause of the energy excess | VERIFIED | high |
| E3-04 | confirmation | yes | P0-3 ("and hit-point") | WL95A pooled X1c 1.037 / 0.268 (0.045) (`95A_WORKLOG...md:82`) | anchor 1.068 / 0.266 vs oracle; 1.084 / 0.303 vs reference | `conv_models.json` | Exact direction without exact incident state leaves a 0.27 footprint error; both parts of "exact-direction and hit-point" are needed | VERIFIED | high |
| E3-05 | confirmation | no | WL95A pooled table B2 | X1 1.022 / 0.111, X1c, q256, M64_spool as above | recomputed from stored draws: identical; my deterministic values agree per patch within WL95A's MC noise (rms z 0.90 to 1.02) | `results/ecro_attr95a/deliv.npz`, `wl95a_pooled_boxA.npz` | WL95A's pooled Monte Carlo measurement of these four arms is reliable; its X1 residual 0.111 is noise, not bias | VERIFIED | high |
| E3-06 | statistical_weakness | yes | P0-3 as a statement about a usable estimator | WL95A per-draw X1 L1 median 0.617, deployed X1 0.764 / 0.627 (`95A_WORKLOG...md:83, 132`) | recomputed from stored draws: X1 per-draw L1 median 0.617 (0.616 vs my oracle), M64_spool 0.665; not re-simulated | `results/ecro_attr95a/deliv.npz` | "Closes the error" is true for the expectation (deterministic or pooled evaluation). At the WL95A relation budget a single exact-direction draw is no closer to the reference than a binned one. Any decision that reads P0-3 as "switching the per-ray lookup fixes delivery" needs the variance qualifier | CONDITIONALLY_VALID | medium (per-draw part rests on stored draws) |
| E3-07 | dataset_limitation | yes | P0-2 generality ("systematic sharp-delivery bias") | one configuration (WL96 itself: "Oracle validation is one configuration", `96_WORKLOG...md:137-140`) | one light 5 cm in front of the panel plane, incident cosine 0.024 to 0.164 (all in incident ring 0), one Lambert receiver box, 2-event term only | `val_dr.json` (`T_VIS.light_to_panel`) | The existence of a carrier bias for a lobe narrower than a bin is general; the +40 to +48 % size and the positive sign are specific to this geometry and must not be quoted as a general magnitude | UNTESTED_INFERENCE (for any generalisation beyond CK2 `none`) | high |
| E3-08 | statistical_weakness | no | precision of oracle validation | n0 0.059; WL96 "up to about 0.03 unresolved" | reference total SE 2.7 %; bootstrap L1 mean 0.0665, p95 0.099; per-patch SEs from bin scatter are not calibrated (half-vs-half mean z^2 3.75) | section 3.1 | Agreement finer than about 5 % (total) / 0.06 (L1) is not resolved by the stored reference; far below the audited effects | NONE | high |
| E3-09 | confirmation | no | reference integrity for `e2_p` (`none`) | - | no clamp, kmax-independent, probe offset and k = 1 lookup inert for `none`, cell rounding not involved | `pt86a.py:73-198`, `mat94.py:126-139, 289-299` | No reference-side explanation of the binned bias | VERIFIED | high (code reading plus E3-01 / E3-02) |
| E3-10 | statistical_weakness | no | WL95A "deployed" column | deployed M64(s_pool) 1.158 / 0.483, A4 1.208 / 0.524, X1 0.764 / 0.627 | the deployed relation's totals sit at the 13th / 14th / 22nd percentile of the 64 pooled draws; deterministic values are 1.484 and 1.015 | `results/ecro_attr95a/deliv.npz` | Single-draw ("deployed") delivery ratios understate the binned bias and should not be used as the carrier's bias; WL95A already says so (`95A_WORKLOG...md:86, 128`) and that statement is confirmed | VERIFIED | high |

## 5. P0 assessment within this track's lens

**P0-2: VERIFIED, scoped to CK2 `none`, the exactly-2-event panel-mediated term on box A.** A patch x 64-bin sender, whether it is the
ideal table of the true once-scattered radiance or M64(s), delivers 1.39x to 1.46x the exact exitance (1.41x to 1.48x the path-traced
reference) with a footprint error of 0.40 to 0.49, in a computation with no relation sampling. The numbers of WL95A (pooled) and WL96
(deterministic) are reproduced by independent code within 0.2 to 0.5 % (WL96) and within Monte Carlo noise (WL95A). The cause is the
bin average of the outgoing direction. Restriction: one geometry, one light, one receiver object; the magnitude and the sign are not
general (E3-07).

**P0-3: VERIFIED as a statement about the expectation of the evaluation; CONDITIONALLY_VALID if read as a statement about a single
relation draw.** Exact direction and exact hit point give ratio 1.015 +- 0.027 and L1 0.068 against a reference whose own noise floor is
0.059 to 0.067, versus 1.48 / 0.49 for M64: the delivery error is closed down to the reference noise, and an independent Monte Carlo
confirms the deterministic value to 0.3 %. Both ingredients are needed (exact direction with patch-centre incident state leaves 0.27).
Restrictions: closure below 0.06 L1 cannot be shown with the stored reference; per-draw behaviour at the WL95A ray budget does not show
closure (taken from WL95A's stored draws, not re-simulated); one configuration.

P0-1, P0-4 and P0-5 are outside this track; nothing here bears on them except that the delivery numbers they build on (M64 pooled
over-delivery, exact-evaluation closure) are sound on CK2 `none`.

## 6. Special checks

| check | result |
|---|---|
| Angular weighting of the reference reduction | Derived from `dir_in_bin`; exact weights equal WL95A's W_T; my reduction equals WL95A's stored reference to 2e-16. For a Lambert receiver any weight set summing to pi gives the same expectation, so this metric is insensitive to the earlier W94 / W_T issue (WL95A reports a 4e-3 difference; plausible, not re-measured) |
| Which argument carries the cosine | Second argument of `eval`; verified with a Lambert state on the production closure (`T_COS`) |
| Receiver mask | 36 of 54 patches receive signal (faces -x, -y, +z, and 0.03 % on -z); the +x and +y faces are exactly 0 in the reference, the oracle and the Monte Carlo |
| Comparison semantics of X1c | My `anchor` follows `a95.py:90-92` (incident state at the patch centre, same outgoing direction); agreement with pooled X1c within its noise |
| Visibility | Real scene ray tests against the analytic rule, 2.1e6 pairs per face and 2.1e6 light rays (`T_VIS`) |
| Convergence | Section 3.4; plus an independent Monte Carlo estimator |
| Reference-side explanations | Section 3.5: none found |

## 7. Errors in my own work (disclosed)

- First attempt, `quad97_buggy_anchor_v1.py`: the `anchor` model indexed the patch-centre array, which is ordered iu * 7 + iv, with the
  patch id iu + 6 iv. It evaluated the light state of the wrong patch and gave ratio 0.408 / L1 0.735. Fixed in `quad97.py` (inverse
  permutation plus an assertion that every node lies within half a patch diagonal of its centre). The fixed value (1.068 / 0.266 vs oracle)
  agrees with WL95A's independent X1c. The other models never used that array and are bit-identical before and after the fix.
- A first version of the analysis divided by a zero SE on the 18 patches that never see the panel; fixed by restricting z statistics to
  patches with signal.

## 8. Not verified, and why

- Any light, variant or receiver other than CK2 `none` / box A / the single point light; `e2_o`, 3+ event terms, image-level (ROI)
  relevance. Out of the stated task and budget.
- The production M64 tables themselves (`quad87` Mq, MIS-sampled) and the production relation code were not loaded. My `m64` is my own
  construction from the stated definition. Its agreement with WL95A's pooled M64_spool is within that arm's 3.5 % Monte Carlo SE, so a
  difference of a few per cent between my table and the production table is not excluded.
- Per-draw variance (finding E3-06) was read from WL95A's stored draws, not re-simulated.
- Agreement of the production path tracer with exact transport below about 5 % (total). A 16x larger reference would cost about 9 GPU-min
  with production `field_src`; it could not change either verdict, so it was not run.
- The WL96 designer scripts were not read (by instruction); only their reported numbers were compared.
- WL95A's statement that W_T and W94 agree to 4e-3 on box A.

## 9. Reproducibility appendix

Local files (sha256 prefix): `experiments/audit97/E3/` quad97.py 3b3d7acbc583, er97.py 355a44ca6340, geo97.py fed79f9d0ea2,
val97_dr.py cd653e62707a, cmp97.py 1609b47f3492, mc97.py d6895b87d6e0, ana97.py 7aacfa7ae4f8, conv97.py d3388f87c318, run_e3.sh 8aca75e53b42,
job1.sh 10e5057bbb0a, job2.sh 55040fd47f79, job3.sh a3cf9f79f380, quad97_buggy_anchor_v1.py 41020f004f72 (kept for the record).

Server: LabServer63, `~/rest_x97/E3/` (code copy of commit a1718f7 under `code/`, outputs under `out/`), environment `. ~/rest_x95a/env.sh`,
one GPU process at a time, peak 1.16 GB, total GPU time about 6 minutes.

    # facts (section 2)
    ./run_e3.sh val97_dr.py /home/nd/rest_x97/E3/code /home/nd/rest_x97/E3/out
    ./run_e3.sh cmp97.py /home/nd/rest_x97/E3/out
    ./run_e3.sh quad97.py tabtest $C $O
    # tables: <tag> <points per patch side> <nth> <nph> <incident sub-cells per side>
    ./run_e3.sh quad97.py tables $C $O a 16 384 256 8
    ./run_e3.sh quad97.py tables $C $O c 32 384 256 32
    ./run_e3.sh quad97.py tables $C $O b 16 768 512 8
    # deliveries: <tag> <sender spacing mm> <receiver nodes per side>          (job2.sh holds the full list)
    ./run_e3.sh quad97.py deliver $C $O s8r12 8 12 --tables=$O/tables_a.npz --bindir=384,256
    ./run_e3.sh quad97.py deliver $C $O s4r24 4 24 --tables=$O/tables_a.npz
    # Monte Carlo self-check (seed 97001)
    ./run_e3.sh mc97.py $C $O 23 32 97001
    # reference slice: e2_p_a / e2_p_b rows 1130..1183 of ~/rest_x94/results/fields/CKF_none.npz -> out/ref_boxA.npz (sha256 559ac125c2bd)
    # local analysis (numpy only)
    python experiments/audit97/E3/ana97.py results/audit97/E3 s4r24 s8r12

Result files in `results/audit97/E3/` (sha256 prefix): deliv_s4r24.npz 74d72fbe4f7d, deliv_s8r12.npz 2475c6b8d664, deliv_s4r12.npz
231247903271, deliv_s2r12.npz f6201ccca518, deliv_s4r6.npz 24a2289aa122, deliv_s16r12.npz a60987357d81, deliv_s16r12hv.npz bd1da92a839c,
deliv_s4r12b.npz 85fe2da0f858, deliv_s4r12c.npz dd99153b90c4, deliv_s8r12o1.npz 600b14822c3f, deliv_s8r12o2.npz 2a6c92c8deae,
mc_l23b32s97001.npz 998d1b5183eb, ref_boxA.npz 559ac125c2bd, wl95a_pooled_boxA.npz ced741a15c7b, ana_s4r24_s8r12.json 8251051b3a7e,
conv_models.json f12def5ca842, val_dr.json cc2ec677fe91, cmp_er.json b945b6c1ade6, tabtest.json 6eeb2d94b548; logs job1.log, job2.log,
job3.log; `_interrupted_run_log.md` (coordinator's extract of the first attempt).

Audited inputs read (sha256 prefix): pt86a.py cfc109cb61c7, mat94.py 15268a6a598e, a95.py df3a14c22e05 (definitions only),
quad87.py cfa9c5c4e854 (definitions only), results/ecro_attr95a/deliv.npz f2324a32effc80ca.

    python experiments/audit97/E3/conv97.py results/audit97/E3 results/ecro_attr95a/deliv.npz

`conv97.py` (sha256 d3388f87c318) writes `conv_models.json`: convergence ladder, model-vs-oracle metrics, comparison with WL95A's stored
pooled arms, per-draw statistics, reference noise tests (bootstrap with `numpy.random.default_rng(97)`), quadrature vs Monte Carlo.
