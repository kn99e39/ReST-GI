# WL97 audit, track C1: measurement and statistical audit, WL92 through WL94D

Auditor: independent track C1. This run resumed an interrupted first attempt of the same track; its scripts and outputs were
re-checked before use (section 8). No other track's report or directory was opened. Date: 2026-11-16. Repository HEAD efcf86a.

## 1. Scope and method

- Scope: the measurements behind the headline claims of WL92, WL92A, WL92B, WL93, WL93A-D, WL94, WL94A-D. Metric definitions and
  implementation, angular weights, receiver masks, normalisation, split-half noise, paired versus independent comparisons, seed spread.
- Method: read the metric code, then recompute from stored raw artifacts with own numpy code that does not call the audited metric
  functions. Each recomputation first reproduces the stored value with the definition *as coded* (identity check), then changes exactly
  one thing (weights, mask, norm, noise statistic) and reports the difference and the decision-level effect.
- Nothing was trained, tuned, modified, moved, committed. One short GPU job (model forward passes only, 226 MiB peak, under one minute)
  dumped the learned S1 factors of 10 stored checkpoints; everything else ran on CPU.
- Labels used below: MEASURED = recomputed by me from raw artifacts; JSON = read from a stored result JSON without raw recomputation;
  INFERENCE = my reasoning from measured numbers.

Depth actually reached per batch:

| batch | raw recomputation by me | JSON-level review only |
|---|---|---|
| WL92 | region-C E_H of 11 arms x 2 conditions from raw renders (identity to 5 digits), MC standard error, noise floor, mask composition | fits, identifiability, local operator tables |
| WL92A, WL92B | none | region-C E_H tables, criteria, E4 / E5 |
| WL93, WL93A-C | none | preflight, criteria, contrasts |
| WL93D | none | criteria (G3, C2 cells) |
| WL94, 94A, 94C, 94D | all CK2 table metrics of 18 model x seed tables (identity 2e-7), all image E_H of 41 render families with MC error, sender field / operator / delivery metrics of 10 checkpoints (identity 1e-7 to 2.5e-7), directional albedo of the learned S1 operator | population metrics, training curves |
| WL94B | none (gate inputs are unweighted; code path read) | audit.json gates |

## 2. Findings

Verdict vocabulary and categories are those of the audit brief. "Material" = could change an interpretation, an acceptance result or an
architecture decision.

### C1-01. an94.W is coded as 8 rings x 8 sectors; the protocol layout is 4 rings x 16 sectors (implementation_defect; material for specific numbers, no gate flips)

- Evidence. `experiments/ecro_mat94/an94.py:19` sets `NB, NPHI, NZ = 64, 8, 8`; `an94.py:32` builds `W = ((arange(64) // 8) + 0.5) / 8 * 2 pi / 64`.
  The bin layout is 4 rings in cos(theta) x 16 azimuth sectors (`experiments/ecro86/protocol/ecro86.json:47`; every WL86-WL93D weight uses
  `S.NPHI`, `S.NZ`, e.g. `experiments/ecro_attr92/diag92.py:114-119`). With bin = ring * 16 + sector, the coded cosine equals the true ring
  cosine minus 0.0625 for sectors 0-7 and plus 0.0625 for sectors 8-15: a factor 0.5 / 1.5 in the grazing ring, 0.83 / 1.17, 0.9 / 1.1,
  0.93 / 1.07 in the others. The azimuthal mean is unchanged, so azimuthally balanced fields are almost unaffected and fields
  concentrated in a few grazing sectors (the CK2 panel lobe) are strongly affected.
- Where it is used (MEASURED by grep and code reading): `an94.ex` and `an94.bin_err` (table amplitude / corr / error of WL94, 94A, 94C, 94D),
  `b94.py:47,85` (the `ex` path labels of WL94B, reused by WL94C `q94.py:85` and WL94D `s94.py:149`), `s94.py:129,157,287` (WL94D G_PHYS
  check and every sender field / delivery metric). NOT used by: WL92-WL93D; any image E_H; the F2, RET, SSR, RENDER_IMP gates;
  the WL94B G_EXPOSURE gate (unweighted normalised loss mass `nm`, `b94.py:82-83`); the WL94D teacher loss (`s94.py:57-60`, unweighted);
  the population metrics (`c94.py` `exitance_w` uses `S.NPHI`).
- Recomputation (own code, `c1_tables.py`, `c1_sender_metrics.py`; coded definition reproduces the stored JSON to 2.1e-7 over 16,000+ values
  and the stored sender JSON to 2.5e-7):

| quantity (WL) | stored / coded W | true W | gate effect |
|---|---|---|---|
| box-A amplitude of B, all 18 tables x 5 variants (94, 94A, 94C, 94D) | e.g. own s8601 U03 0.100 | 0.093; largest change 0.009 | MR flag identical in 18 / 18 |
| table corr | - | changes up to 0.12, only on the uniform variants U03 / U07 where abs(corr) is below 0.6 and carries no gate | none |
| sender split error of the S1 part vs e2_p on box A (94) | 0.594 / 0.873 (own s8601 / s8602) | 0.595 / 0.866; largest change 0.010 | none |
| WL94D CK2 sender energy ratio, T | 1.75 / 1.71 | 1.37 / 1.39 | none |
| same, D | 0.87 / 1.62 | 0.67 / 1.25 | none |
| same, P | 1.01 / 1.67 | 0.77 / 1.28 | none |
| same, frozen own | 0.56 / 1.12 | 0.43 / 0.89 | none |
| WL94D CK2 sender field rel L1, T | 1.54 / 1.48 | 1.32 / 1.30 | none |
| same, own | 1.08 / 1.22 | 1.04 / 1.12 | none |
| same, P | 1.29 / 1.63 | 1.19 / 1.40 | none |
| WL94D val sender field rel L1 / energy | 0.85-0.89 / 0.83-1.09 | changes of 0.015 or less | none |
| SENDER_IMP ratio T / min(R, P), val | 0.964 / 0.977 | 0.968 / 0.978 | False -> False |
| SENDER_IMP ratio, CK2 | 1.193 / 0.943 | 1.110 / 0.932 | False -> False |
| G_PHYS CK2 diagnostic, box A (ratio, rel L1) | 1.211, 0.528 | 1.208, 0.524 | none |

- Decision-level impact. No gate or batch verdict of WL94-WL94D changes. The WL94D wording "over-emits (1.75 / 1.71)" should read
  1.37 / 1.39; the frozen own s8601 sender is at 0.43 (not 0.56) of the exact sender's energy on CK2. Note that the sign of "over-emission"
  depends on the cosine weighting at all: with unweighted bin sums the T energy ratio is 0.86 / 0.90.
- Not verified: the training-item half of G_PHYS (pooled ratio 0.953, corr 0.641) with true weights. The path labels were stored only
  as W-weighted exitances, so a correction needs re-rendering.
- Proposed verdict: IMPLEMENTATION_DEFECT (confirmed), effect bounded as above. Confidence high.

### C1-02. The E_H "noise_bound" is an L1 noise floor, not an uncertainty; it is 100 to 280 times the Monte Carlo standard error and is used as a significance margin (evaluation_defect; material)

- Evidence. `an94.py:182-189`, `analyze92.py:54-65`, `analyze86.py:120-134`: `noise_bound = (0.5 sum|x_a - x_b| + 0.5 sum|h_a - h_b|) / den`
  over the pixels of the mask. That is a sum of absolute per-pixel noise. It estimates the value E_H would take if the true error were
  zero. It does not shrink with the number of pixels and is not the uncertainty of E_H or of a difference of two E_H values.
- MEASURED (`c1_wl92_eh.py`, `c1_images.py`; per-pixel replicate variances, pixels independent):

| mask | E_H example | coded noise_bound | L1 floor (true error 0) | MC standard error of E_H | disjoint-half values |
|---|---|---|---|---|---|
| WL92 E2_T1 region C (3,974 px), frozen s8601 | 0.07172 | 0.01262 | 0.00974 | 0.00012 | 0.07183 / 0.07209 |
| WL92 E2_T1 region C, oracle ORA2 | 0.01269 | 0.01302 | 0.00987 | 0.00013 | 0.01607 / 0.01612 |
| WL94 CK2 ROI_T (58,035 px), own s8601 C0 | 0.04724 | 0.00834 | 0.00693 | 0.00003 | 0.04756 / 0.04759 |
| paired difference C0 - C2_srcS1, own s8601 | 0.01622 | (rule uses 2 x 0.00836) | - | 0.00007 or less | 0.01603 / 0.01604 |

- Decision-level impacts.
  1. **WL94 SSR gate, seed 8601, and the H-A label.** `an94.py:236-240`: pass if `d >= max(2 nb, 0.005)` and share >= 0.30. Stored: d = 0.016216,
     2 nb = 0.016727, share 0.343 -> FAIL by 0.0005. The difference is resolved at more than 200 standard errors and reproduces on disjoint
     data halves. With the MC standard error as the noise term the criterion passes on both seeds (shares 0.34 / 0.54). By the frozen
     mapping (`an94.py:333-355`: `all(ssr_f) and (not mr or ssrc)` -> SUPPORTED, with `mr` False) **the H-A label flips from INCONCLUSIVE to
     SUPPORTED.** The worklog's "s8601 at the noise limit" (WL94 line 253) is not correct as a statement about measurement noise.
  2. WL93D G3 cell E2 (`analyze93d.py:92-98`): C1_both 0.0836 vs C0 0.0769 marked "differs: false" against margin 0.0126. INFERENCE from
     the WL92 standard error on the same mask: about 40 sigma. The label does not flip, because the rule needs two cells and the E5 cell
     differs by 0.0012, below the rule's own 0.005 floor.
  3. Values close to the floor cannot be ranked by E_H: oracle 0.006-0.013, WL92A own on E4 0.0130 / 0.0124 against oracle 0.0096,
     exact-sender arms at 0.017-0.035 on CK2 against a floor of 0.007. Relative statements such as "E4 C halved" understate the change in
     true error by an unknown amount.
  4. In the other direction, every arm-versus-arm E_H difference I checked inside WL94-WL94D (26 pairs) is resolved at 40 sigma or more.
     Monte Carlo noise is not what limits those comparisons; training variability is (C1-03).
- Proposed verdict: the E_H values are VERIFIED as numbers; significance statements built on `noise_bound` are CONDITIONALLY_VALID
  (conservative for differences, uninformative near the floor); the WL94 SSR / H-A outcome is decided by a mis-scaled noise term.
  Confidence high for the measurement, medium for the label consequence (the frozen rule names this statistic, so the stored label is
  rule-conformant).

### C1-03. Seed and recipe variability of the sharp CK2 error is larger than every treatment effect and changes sign; n = 2 (statistical_weakness; material)

- MEASURED (CK2, variant none, ROI_T, deployed arm C0; E_H and signed error; MC standard error 3e-5):

| model | s8601 E_H (signed) | s8602 E_H (signed) | share removed by exact S1 sender, s8601 / s8602 |
|---|---|---|---|
| own (frozen WL92A) | 0.0472 (-0.040) | 0.0420 (+0.028) | 0.34 / 0.54 |
| A | 0.0390 (+0.003) | 0.0932 (+0.088) | 0.30 / 0.72 |
| C | 0.0842 (-0.082) | 0.0669 (+0.058) | 0.10 / 0.66 |
| D (WL94C) | 0.0439 (-0.030) | 0.0889 (+0.086) | 0.25 / 0.78 |
| E (control) | 0.0858 (-0.083) | 0.0659 (+0.057) | 0.10 / 0.67 |
| P | 0.0443 (-0.022) | 0.0890 (+0.087) | 0.22 / 0.80 |
| R | 0.0438 (+0.003) | 0.1010 (+0.095) | 0.31 / 0.78 |
| T (WL94D) | 0.0619 (+0.040) | 0.0965 (+0.095) | 0.45 / 0.81 |

- The same contrast has opposite signs on the two seeds: D - C = -0.040 / +0.022; T - P = +0.018 / +0.007; T - own = +0.015 / +0.054.
- The learned S1 sender's delivery to box A relative to the exact (M64) sender on CK2 is 0.41 (own s8601) and 1.27 (own s8602), a
  factor 3 between two seeds of one recipe; across the ten checkpoints it spans 0.41 to 2.07 (true weights).
- On the WL86 workload the same pattern appears (JSON, WL92A): E2_T1 region C, own 0.0769 / 0.0395; frozen ECRO 0.0717 / 0.0634.
- Decision-level impact. The sharp error of these checkpoints is not a stable under-delivery. It is a scale error of the
  panel-mediated light whose size and sign are set by the training run. Two-seed arm comparisons therefore cannot estimate a treatment
  effect. The labels INCONCLUSIVE (WL94A, WL94C) and NO_RELIABLE_IMPROVEMENT (WL94D) are consistent with this. A stronger reading such as
  "sender supervision cannot repair the sharp error" or "the residual is a representation limit" is not supported: a variance-dominated
  error on a condition that training barely covers (C1-07) points at least as much to identifiability and coverage as to capacity.
- Proposed verdict: INCONCLUSIVE-type labels VERIFIED; any capacity inference from these arms UNTESTED_INFERENCE. Confidence high.

### C1-04. WL94D mislabels two delivery numbers as "box A" (evaluation_defect; not material for the gate, material for the quoted magnitude)

- Evidence. `experiments/ecro_s94d/s94.py:309` sets `bx[1130:1238]`: patches 1130-1183 are box A (PSID 7) and 1184-1237 are box B (PSID 8)
  (`masks94.npz`, MEASURED). `delivery_ratio` (`s94.py:311`) is taken over all non-panel patches. WL94D lines 110-112 and 146 report both
  as "CK2 delivery to box A": "T 1.05 vs P 0.62 (delivery ratio 1.28 vs 0.70)".
- Recomputed with true weights (the delivery numbers are insensitive to W):

| CK2, learned vs exact S1 sender through the exact receiver | reported as box A | box A + B (what was computed) | box A only | all receivers |
|---|---|---|---|---|
| T s8601 rel L1 | 1.05 | 1.052 | 1.036 | 1.091 |
| P s8601 rel L1 | 0.62 | 0.618 | 0.614 | 0.945 |
| T s8601 ratio | 1.28 | 1.78 | **1.76** | 1.28 |
| P s8601 ratio | 0.70 | 0.83 | **0.82** | 0.70 |
| T s8602 ratio | 1.19 | 2.09 | **2.07** | 1.18 |

- Effect. The rel L1 mislabel is immaterial on CK2 (0.02 or less) but not on validation geometries (own s8602: A + B 0.600, box A 0.837).
  The ratio mislabel understates the over-delivery to box A (1.76 / 2.07, not 1.28 / 1.19). The direction of the WL94D statement stands and
  becomes stronger. SENDER_IMP does not use these numbers.
- All other masks agree with the claims: ROI_T is 58,035 pixels, all on the panel (sid 5), whose mirror ray hits box A only (PSID 7, no
  box B); DIFF is 9,039 floor / wall pixels, disjoint from ROI_T; table metrics use T = box A (54 patches) and T_panel_face (9 patches).
  WL86-WL93C use pixel regions of masks86; region C of E2 is 3,974 pixels, all of the mirror-cone type (C_footprint 0), 3,349 on the panel
  and 625 on the sphere, none on a box.
- Proposed verdict: IMPLEMENTATION_DEFECT in labelling, CONDITIONALLY_VALID claim. Confidence high.

### C1-05. Matched-norm comparison of the learned S1 operator with M64: the rank-16 gap is real; "zero-operator level" holds in flux L1 only (invalid_inference in wording; material for interpretation)

- WL94D line 109 sets the learned operators' flux-L1 distance (0.96-1.04) beside a rank-16 Eckart-Young residual in the flux-Frobenius
  norm (0.46-0.52). The worklog flags the norm difference; WL95 later states "zero-operator level" in matched L1.
- MEASURED in matched norms (flux-transfer weighting with true ring cosines; my SVD from the dumped M64 S1 matrix, re-checked locally):

| operator vs M64 S1 | flux L1 | flux Frobenius (R / G / B) | mass ratio | peak-bin retention |
|---|---|---|---|---|
| zero | 1.000 | 1.000 | 0.00 | 0.00 |
| learned, 10 checkpoints (val and CK2 geometry) | 0.96-1.04 | 0.77-0.83 | 0.82-1.06 | 0.17-0.28 |
| SVD rank 16 (Frobenius optimum) | 0.891 (0.683 clipped at 0) | 0.52 / 0.46 / 0.46 | 0.94 | 0.54 |
| NMF rank 16 (non-negative, Frobenius fit) | 0.606 | 0.55 / 0.49 / 0.49 | 1.03 | 0.52 |
| SVD rank 32 | 0.548 | 0.32 / 0.23 / 0.24 | 0.99 | 0.79 |

- Reading. (a) Confirmed: in matched norms the learned operators are well short of what a non-negative rank-16 operator can reach for
  this one material (L1 1.0 vs 0.61, Frobenius 0.80 vs 0.50, peak 0.2 vs 0.5). The gap is not a rank-16 capacity bound. (b) Flux L1
  saturates near 1 for any operator that carries the right mass in the wrong bins; the Frobenius-optimal rank-16 operator itself scores
  0.89. "At zero-operator level" is therefore true in L1 and misleading as a description: the learned operators carry 82-106 % of the
  mass and are 17-23 % better than zero in Frobenius norm. (c) The reference is M64, which is itself biased against the physical
  single flight (C1-06).
- Proposed verdict: the 0.46-0.52 and 0.96-1.04 numbers VERIFIED; the WL94D juxtaposition INVALID_COMPARISON as written; the corrected
  conclusion (large gap to an admissible rank-16 operator) VERIFIED; "zero-operator level" CONDITIONALLY_VALID (L1 only). Confidence high.

### C1-06. The M64 "exact sender" is not a physical oracle on CK2; the WL94D G_PHYS gate passed only because its training-item labels were noisy (evaluation_defect; material)

- Evidence. `s94.py:136-166`. Rule as frozen: pooled energy ratio in [0.8, 1.25] and corr(exact, label) >= corr(label, independent label).
  The first-draft rule (corr >= 0.9) failed at 0.641 and was replaced before the freeze after that result was seen
  (`audit.json` field `first_draft_rule`). The label reliability is 0.525, so the bar became 0.525.
- On CK2, where the reference is precise (2 x 4,096 spp), the same quantity fails by a wide margin (stored and MEASURED, true weights):

| CK2 single flight through the exact receiver vs path label e2_p | ratio | corr | rel L1 | half-noise rel L1 | corr of the two reference halves |
|---|---|---|---|---|---|
| M64 sender, box A (54 patches) | 1.208 | 0.720 | 0.524 | 0.059 | 0.983 |
| M64 sender, panel-facing face (9 patches) | 1.201 | -0.321 | 0.591 | 0.070 | 0.867 |
| M64 sender, all receivers | 1.373 | 0.714 | 0.937 | 0.125 | 0.930 |
| learned sender own s8601, box A | 0.495 | 0.759 | 0.588 | 0.059 | - |
| learned sender own s8602, box A | 1.534 | 0.751 | 0.901 | 0.059 | - |
| learned sender T s8601 / s8602, box A | 2.125 / 2.501 | 0.83 / 0.82 | 1.18 / 1.58 | 0.059 | - |

- Reading. (a) The 64-bin exact sender is 9 times the reference noise away from the physical single flight on box A and anti-correlated
  with it on the face nearest the panel. Within my lens this supports P0-2 at the magnitude level. (b) The WL94D teacher was this
  operator. "Oracle-supervised" in WL94D means supervised toward a target that over-delivers by 21 % with the wrong spatial pattern on
  the evaluation scene. (c) Measured against the physical label, the frozen own s8601 sender is only 0.06 worse than M64 on box A
  (0.588 vs 0.524) and equal on the panel-facing face (0.597 vs 0.591); own s8602 is 0.38 worse. Part of what WL94-WL95 call sender
  error "relative to the exact sender" is distance to a biased reference. (d) Even so, substituting M64 at S1 lowers the rendered ROI
  error in all 16 model x seed cases (C1-03 table), so the substitution result itself is solid.
- Proposed verdict: G_PHYS pass CONDITIONALLY_VALID (noise-limited gate, threshold revised after a failed first draft, disclosed);
  "sender oracle" naming INVALID_COMPARISON if read as physical truth; WL94D outcome NO_RELIABLE_IMPROVEMENT VERIFIED as measured
  but UNTESTED_INFERENCE for "sender supervision cannot help". Confidence high.

### C1-07. The CK2 chain rests on one post hoc scene at the edge of the training distribution (dataset_limitation; material)

- CK2 was built in the WL93D post hoc addendum by adjusting object placement until the reflection was visible; box A's corner is at
  x = -1.345 against the training rule's limit -1.35, with the panel-facing face about 0.22 m from the panel; one point light, one
  camera (`docs/worklogs/93D_WORKLOG_ADDENDUM_CHECKERBOARD_QUALITATIVE.md:21-40`). It did not pass a frozen workload admission; the
  frozen E6 admission of WL93D failed 3 / 3.
- WL94B's own audit (JSON, `results/ecro_exp94b/audit.json`): panel-mediated transport is 3.4 % of the training loss mass (e2_p 2.1 %,
  e3_p 1.3 %); 4 of 96 audited items (0.042) reach CK2's box-A level; status UNDERREPRESENTED. The same file records that the base
  objective supervises the panel's once-scattered response only through receivers (field `semantics`), i.e. the S1 sender is not
  directly identified by the training loss. That is consistent with the 3-fold seed spread of its CK2 output (C1-03).
- On the two held-out conditions that were measured inside my range, the picture differs from CK2 (JSON, region C, F2 level 0.06):

| condition | frozen ECRO s8601 / s8602 | OWN s8601 / s8602 | explicit X_EXP | oracle |
|---|---|---|---|---|
| E5 T1 (WL92B, admitted) | 0.0215 / 0.0236 | 0.0327 / 0.0274 | 0.0299 | 0.0062 |
| E5 T0 | 0.0368 / 0.0295 | 0.0339 / 0.0169 | 0.0174 | 0.0099 |
| E4 T1 (WL92A, not admitted: Q4) | 0.0181 / 0.0188 | 0.0130 / 0.0124 | 0.0268 | 0.0096 |
| E2 T1 (WL86 gate condition; MEASURED except OWN) | 0.0717 / 0.0634 | 0.0769 / 0.0395 | 0.0491 | 0.0127 |

  Every learned model is below 0.06 on E5 and E4; the frozen ECRO is below the explicit 64-bin recurrence arm X_EXP
  (`diag92.py:292`) on E5 T1 and E4 T1, and above it on E5 T0 and E2 T1. The sharp
  failure is specific to E2 (0.076-0.086 on the panel pixels, 0.007-0.009 on the sphere pixels of region C) and to CK2.
- Decision-level impact. All negative evidence of WL93D-addendum through WL94D (and, by construction, anything later that reuses CK2)
  describes one hand-placed, weakly covered configuration. It supports "these checkpoints fail on CK2", not a statement about the
  representation class over the workload distribution.
- Proposed verdict: CONDITIONALLY_VALID (scene-specific). Confidence high for the facts, medium for how far the restriction reaches
  outside my range.

### C1-08. Gates decided by margins inside noise or by rule mechanics rather than by the measured quantity (statistical_weakness; material for three of them)

| batch | decisive gate | value vs threshold | measured noise / spread | assessment |
|---|---|---|---|---|
| WL86 F2 as replayed in WL92 | region C E_H <= 0.06 on E2 | T1 0.0717 / 0.0634; T0 0.0603 / 0.0588 | MC se 0.0001; seed spread 0.008; WL92 arms with exact 64-bin local operators 0.041-0.052, explicit X_EXP 0.049, WL93D exact-operator arm 0.030 | Exceedance real in MC terms at T1; margin (0.003-0.012) is of the order of the seed spread; s8602 passes at T0. Arms that contain no learned local operator already use 50-87 % of the 0.06 budget on this condition |
| WL92 | G_rep: best in family <= 0.06 | R16 construct 0.084 / 0.073; per-geometry rank-16 fit 0.036 / 0.017 excluded as "not converged" (`analyze92.py:197-198`) | MC se 0.0001 | STOP_INTERVENTION_G_rep came from the convergence flag (relative change 0.06 / 0.12), not from fidelity. The worklog's verdict A = INCONCLUSIVE is the right reading; "rank-16 loses the panel lobe" as a capacity statement is not supported by the gate |
| WL92A | E4 admission (Q4), then HELDOUT_SHARP | Q4 value 0.0101, fails; own 0.0130 / 0.0124 vs sum 0.0245 / 0.0241 passes on both seeds | oracle 0.0096; floor about 0.010 | The only positive held-out result of the range (OWN at the oracle floor on E4, both seeds) was set aside by workload admission. Outcome D_STOPPED is rule-conformant; the evidence is not negative |
| WL92A | NO_REGRESSION | own s8602, E1_T1_LSPOT stationary 0.0360 vs 0.0216 | one seed of two | one-seed decision |
| WL93 | preflight: irreducible <= 0.5 x learned | 0.508 vs 0.485 / 0.483 | implied single-q noise 0.145 pooled; q-vs-q 0.29 pooled but 0.07 or less on 49 of 55 validation geometries (outliers up to 0.91); per-geometry irreducible median 0.36 against learned 0.78-1.23 | Decided by a 0.023 margin inside target noise that a few geometries dominate. Per geometry the rule is met on 52 of 55 (both seeds). STOP is rule-conformant and produced no evidence on the architecture |
| WL93A | signal: dM <= -2 se | dM +0.051 / +0.125 | se 0.060 / 0.105; predeclared delta 0.099 | Low power: the detectable effect (0.12 / 0.21) exceeds the predeclared effect size. NO_SIGNAL is the right label; H_TARGET "NOT SUPPORTED" is stronger than an underpowered null |
| WL93B / 93C | contrasts on pooled tables | e.g. UQ - TQ WHOLE -0.0032, se 0.0005 | run-to-run 0; 2 seeds | Paired bootstrap se is small and both seeds agree in sign; smoke scale. No objection at JSON level |
| WL93D | G3 / C2 cells | E5 cell: C2_both 0.0314 vs C0 0.0327 "not better" | explicit X_EXP 0.0299, oracle 0.0062 on the same cell | Ceiling effect: on E5 the learned C0 is already at the 64-bin level, so the exact-operator arm cannot beat it by the margin. COADAPTATION_LIMITED also holds through the validation branch, so the label does not depend on this cell |
| WL94 | SSR s8601 | d 0.01622 vs 2 nb 0.01673 | MC se 0.00007 | See C1-02: the H-A label flips with a correctly scaled noise term |
| WL94 | MR_frozen | amplitudes 0.09-0.69 vs [0.85, 1.15] | weight change 0.009 or less | robust |
| WL94A | RET, F2, MR per seed | C s8601 0.0842 vs A 0.0390; C s8602 0.0669 vs A 0.0932 | seed sign flip | INCONCLUSIVE is correct; decided by seed disagreement |
| WL94B | G_EXPOSURE >= 1.25 | 1.2316 / 1.2361 | deterministic allocation; PM rank reliability 0.999 | Missed by 1.1-1.5 %. A design-validity stop; candidate D was not trained; no evidence on the architecture |
| WL94C | sharp vs own / vs C | D - C: -0.040 / +0.022 | MC se 0.00007; seed sign flip | INCONCLUSIVE is correct |
| WL94D | SENDER_IMP: T <= 0.8 x min(R, P) on val and CK2 | val 0.97 / 0.98; CK2 1.11 / 0.93 (true W) | deterministic | robust to weights and masks; False on both seeds |
| WL94D | RENDER_IMP | T 0.0619 / 0.0965 vs P 0.0443 / 0.0890, R 0.0438 / 0.1010 | MC se 0.00007 | robust; False |

### C1-09. The learned S1 local operator violates directional-albedo passivity in every checkpoint examined (confirmation of P0-1's second clause at operator level; material)

- MEASURED (`c1_sender_dump.py`, `c1_passivity.py`). Operator A = sum_r a c^T on the 42 panel patches of the CK2 geometry, directional
  albedo per incident bin in the same convention as the WL94D teacher check (`s94.py:179-182`), true ring cosines.

| operator | albedo max | mean | share of (patch, incident bin, channel) above 1 | at CK2's lit incident bin 12, mean over patches |
|---|---|---|---|---|
| M64 S1 (teacher) | 0.825 | 0.635 | 0 | 0.60-0.69 |
| own s8601 / s8602 | 2.26 / 3.08 | 0.56 / 0.55 | 10.4 % / 15.4 % | 1.28 / 1.27 |
| D s8601 / s8602 | 1.89 / 2.54 | 0.56 / 0.59 | 8.1 % / 16.2 % | 1.24 / 1.42 |
| T s8601 / s8602 | 2.24 / 2.93 | 0.72 / 0.68 | 16.7 % / 19.7 % | 1.38 / 1.60 |
| P, R (4 checkpoints) | 1.87-2.68 | 0.60-0.62 | 8.8-16.6 % | 1.26-1.68 |

- All entries are non-negative (minimum 1e-6 to 1e-5). The violations concentrate at grazing incidence (ring 0 and 1: max 1.7-3.1).
  CK2's four lit incident bins (12-15) are in the grazing ring; the mean albedo there exceeds 1 at bin 12 for all ten checkpoints.
- Scope limits. This is the local single-scatter operator of one material on one geometry. The aggregate exitance of the learned
  sender on the actual CK2 field is 0.43-1.39 times that of the passive M64 sender, so no net energy gain over the incident flux
  follows for this field (INFERENCE: 1.39 x 0.69 is below 1). I did not test the composed deployed operator (local plus global term).
  ECRO applies the operator once, so non-passivity is a physical-validity defect here, not a divergence mechanism. No contract test in
  WL92-WL94D checks passivity of the learned operator (the stored contracts are superposition, scaling, zero, minimum).
- Proposed verdict: VERIFIED at local-operator level; UNTESTED for the composed operator. Confidence high for the numbers.

### C1-10. Confirmations

- WL92 region-C E_H of 11 arms x 2 conditions reproduced from raw renders to 5 digits (e.g. 0.07172, 0.06339, 0.08433, 0.04487,
  0.01269); the WL86 renders give the same values as the WL92 re-renders.
- WL94 Stage B / Stage C, WL94A, WL94C, WL94D table JSONs reproduced to 2.1e-7 (18 comparisons, 728 to 1,718 values each).
- WL94 image E_H reproduced (own s8601 C0 0.047238). The "34 % / 54 %" exact-sender shares reproduced.
- WL94D sender JSONs reproduced to 2.5e-7 for 10 checkpoints; SVD rank-16 floor 0.46-0.52 and peak retention 0.49-0.56 reproduced.
- WL92-WL93D angular weights use the correct 4 x 16 layout (no hard-coded 8 outside `an94.py` and the two later scripts that replicate
  the bug on purpose to quantify it: `ecro_attr95a/a95.py:28`, `wl95calc/wl95_calc.py:17`).
- Reference precision is adequate where it was used for table metrics: e2_p half-noise on box A is 0.059 against errors of 0.5-1.6.
- Event content of the WL94 sender split is matched (single flight from the S1 part against the two-event path label through the
  panel); the same masks and weights are applied to both sides.

## 3. Priority-0 claims within this track's lens

| claim | assessment | restrictions |
|---|---|---|
| P0-1 learned S1 sender has large avoidable error; may violate passivity | First clause CONDITIONALLY_VALID. Verified: field rel L1 against M64 0.84-0.88 (val), 1.04-1.40 (CK2); operator gap to an admissible rank-16 operator in matched norms (L1 1.0 vs 0.61, Frobenius 0.80 vs 0.50, peak 0.2 vs 0.5); exact-sender substitution lowers the ROI error in 16 / 16 cases; H-A would be SUPPORTED with a correct noise term. Second clause VERIFIED at local-operator level (albedo up to 3.1; 8-20 % of entries above 1) | "Avoidable" is shown against M64 for one material, not by a trained model; M64 is itself 0.52 rel L1 from the physical single flight on CK2 box A; the size and sign of the sender error vary 3-fold between seeds; CK2 is weakly covered by training; composed-operator passivity untested |
| P0-2 M64 / 64-bin sender has a systematic sharp-delivery bias | Magnitude VERIFIED within my lens: CK2 box A ratio 1.21, rel L1 0.52 at 9 times reference noise, panel-face corr -0.32; WL92 64-bin exact-operator arms 0.041-0.052 with signed error about equal to minus E_H; WL93 per-geometry irreducible 0.29-0.39 | "Not eliminated by relation sampling" is outside my range |
| P0-3 continuous exact evaluation closes the CK2 delivery error | Not assessed (WL95A / WL96A). Consistent with the fact that the 64-bin sender is the error carrier in C1-06 | - |
| P0-4 no headroom for a learned higher-order resolvent on CK2 | Not assessed. Relevant restriction from my range: single scene, C1-07 | - |
| P0-5 negative results justify rejecting the tested mechanisms but not a general impossibility | Second half supported. First half CONDITIONALLY_VALID: the WL92-WL94D record contains no completed, powered test that rejects a mechanism. Its outcomes are a gate stop by a convergence flag (92), an admission stop over a positive held-out result (92A), a noise-decided preflight stop (93), smokes with 2 seeds and sign flips (93A-94D), a design-validity stop (94B), all on E2 or CK2. On E4 and E5 the learned models pass the 0.06 level | The supported statement is "the trained checkpoints fail on E2 panel pixels and on CK2; smoke-scale interventions did not reliably fix CK2" |

## 4. Special checks

1. **an94.W 8 x 8 vs 4 x 16.** Confirmed. Every affected number family identified and recomputed (C1-01). No gate or verdict of
   WL93D-WL94D changes. Largest numerical effects: WL94D CK2 sender energy (minus 22 %) and field rel L1 (minus 0.04 to 0.23). WL92x
   and WL93x do not use this weight or a similar one.
2. **Box A vs box A + B.** CK2 image and table masks are box-A only. One mixed mask exists: WL94D `delivery_box` is box A + B, and the
   quoted "delivery ratio" is over all receivers; both are reported as box A (C1-04). No gate uses them.
3. **Unmatched norms (WL94D, WL95).** Recomputed in matched norms (C1-05). The gap to rank 16 stands; "zero-operator level" is an L1-only
   statement; the Frobenius-optimal rank-16 operator itself is at 0.89 in that L1.
4. **Noise and thresholds.** The stored noise_bound is not an uncertainty (C1-02). Gate-by-gate table in C1-08. Decided inside noise or
   by one seed: WL93 preflight, WL94 SSR s8601, WL92A NO_REGRESSION, WL94A / 94C seed disagreement. Decided by rule mechanics: WL92
   G_rep, WL92A E4 admission, WL94B exposure.

Metric compatibility between ECRO outputs, explicit comparator outputs and field references: same masks, same reference, same
normalisation in every E_H comparison I traced (WL92 `analyze92.eh`, WL94 `an94.eh`). The one incompatibility found is the use of M64
single flight as if it were the physical once-scattered field (C1-06).

## 5. Not verified, and why

- WL92A, WL92B, WL93, WL93A-D numbers were reviewed at JSON level only (budget); their raw renders and tables are on the server and
  were not recomputed. Their E_H code path is the one I reproduced for WL92.
- The standard error of the WL93D G3 cells is inferred from WL92's on the same mask, not measured.
- G_PHYS on the 96 training items with true weights (labels stored only as weighted exitances).
- Passivity of the composed deployed operator (local plus global term, with flights).
- Population metrics (`pop/*.json`), training curves, WL94B exposure ratio noise (64-spp independent labels not recomputed).
- The exact threshold of the WL92A Q4 admission clause (value 0.0101 read; rule text not traced).
- Whether true-weight exitances change the WL94C query weights (`q94.py:85,115-116` use `ex` only for a positivity mask and a
  log-regression diagnostic; code read, not recomputed).
- Debiased E_H near the floor (would need a cross-half estimator; not built).

## 6. What I read and ran

Read (targeted): `experiments/ecro_mat94/an94.py`, parts of `mat94.py`, `c94.py`; `experiments/ecro_s94d/s94.py`;
`experiments/ecro_exp94b/b94.py`; `experiments/ecro_mat94r/r94.py`; `experiments/ecro_attr92/analyze92.py`, `diag92.py`;
`experiments/ecro86/analyze86.py`, `masks86.py`; `experiments/ecro_local93d/analyze93d.py`; worklogs 94, 94B, 94D, 95 (sections),
93, 93D addendum (sections); the final JSONs of all batches in scope.

Commands (all from `c:/Projects/ReST-GI`; server `LabServer63`, `. ~/rest_x95a/env.sh`):

```
# server, CPU
$PY c1_tables.py tables_recompute.json                         # 71 s; reads ~/rest_x94{,r,c,d}/results/{tables,fields,pre}
$PY c1_sender_metrics.py snd ~/rest_x94d sender_metrics.json
$PY c1_passivity.py snd passivity.json
$PY c1_images.py ~/rest_x94d images_recompute.json            # about 2 min
$PY c1_wl92_eh.py ~/rest_ecro86 ~/rest_x92 wl92_eh.json
# server, GPU (forward passes only, 226 MiB peak, under 1 min)
$PY c1_sender_dump.py ~/rest_x94d/code ~/rest_x94d ...        # writes ~/rest_x97/C1/snd/
# local
python experiments/audit97/C1/c1_compare_tables.py > results/audit97/C1/tables_compare.txt
python experiments/audit97/C1/c1_sender_report.py  > results/audit97/C1/sender_report.txt
```

No random seeds are involved (deterministic recomputation from stored arrays).

## 7. Reproducibility appendix (sha256 prefixes)

Scripts (`experiments/audit97/C1/`): c1_tables.py 3f1781abf20d1432; c1_compare_tables.py 3159110edc6f93da; c1_sender_dump.py
be8298007daaa2a1; c1_sender_metrics.py f7b9380c0d1956fe; c1_sender_report.py e15084c0c866bb4d; c1_passivity.py c577a2e34f7e80e2;
c1_images.py 72c98a50e92f3208; c1_wl92_eh.py 14f2a23b64d2c32e.

Outputs (`results/audit97/C1/`): tables_recompute.json 437d6296a7b26dda; tables_compare.txt 1c734e01aded4c2a; sender_metrics.json
70d86fedc182e01c; sender_report.txt 1f190c3472c1b09c; sender_dump_meta.json f2041b0ad33d34a1; M64_S1.npy fc61b55222c9513b;
passivity.json df9531a27146de59; images_recompute.json 624519a7aecc9919; wl92_eh.json 8d8f5bdcb8a018f1. Server copies and the
factor dumps are in `~/rest_x97/C1/` (snd/SND_own_s8601.npz d969fc0a71bb0ded).

Inputs (server, read-only): rest_x94/results/pre/masks94.npz b9a7d46b1a23d8c4; rest_x94/results/fields/CKF_none.npz
3a6238a8a42d1db4; rest_x94/results/tables/FZ_own_s8601.npz 44fade056130ec75, FZ_own_s8602.npz 03183dd2e6e013dc;
rest_x94d/results/tables/CD_T_s8601.npz fb55ba850e0cf498; rest_x94d/results/models/T_s8601.pt ccefd37f1e6add5b;
rest_ecro86/results/masks/E2.npz 0ba5435f94e77cf0; rest_x92/results/renders/X92_B_ecro_s8601_E2_T1_r0.npz 90920ee0d10d491a.
Audited code: experiments/ecro_mat94/an94.py 7110b5da503d4638; experiments/ecro_s94d/s94.py c865c56ae0d18820. WL94D chain commit
a08c9bf.

## 8. Resumption note

The first attempt left the table, sender, passivity and image recomputations and an action log
(`results/audit97/C1/_interrupted_run_log.md`) but no conclusions. Before using its numbers I checked: the identity lines (coded
definition equals stored JSON to 2e-7) in `tables_compare.txt` and the sender log; the stored `audit.json` G_PHYS diagnostic against the
recomputed one (1.2115 / 0.5277 stored, 1.211 / 0.528 mine); the stored image E_H (0.047238) against `images_recompute.json`; the
SVD floors, by an independent local recomputation from `M64_S1.npy` (Frobenius 0.520 / 0.458 / 0.461, L1 0.87-0.93 per channel, albedo
maximum 0.8247). The image job it had launched had finished; its output was fetched. No file named `*_buggy_*` exists. Added in this
run: `c1_wl92_eh.py`, the mask composition checks, the gate table, and all conclusions.
