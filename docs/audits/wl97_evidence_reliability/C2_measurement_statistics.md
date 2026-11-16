# WL97 Track C2 — Measurement and Statistical Audit (WL87–WL91, WL95, WL95A, WL96, WL96A)

Independent reviewer report. Date 2026-11-16. Repository HEAD `efcf86a` (branch `Selective_Recompute`), tree unmodified except the audit
directories. No training, no tuning, no existing file changed. This track did not open any other track's report.

Resumption note: a first attempt at this track was interrupted by a usage limit. Its scripts and outputs were inherited
(`experiments/audit97/C2/`, `results/audit97/C2/`, `~/rest_x97/C2/` on LabServer63). Every decisive number below was either re-run or
independently re-derived in this session (section 9 lists which); the server chain launched by the first attempt had completed
(`chain.out`: TABLES OK, 4 renders, no `*_FAILED` line) and its outputs were fetched and inspected here.

Labels used below: **[MEAS]** = recomputed by this track from raw artifacts with its own numpy code; **[RERUN]** = an audited script
re-executed on a copy; **[READ]** = read from a stored JSON / code without recomputation; **[INF]** = inference.

---

## 1. Scope and method

- (a) WL95A delivery statistics: recomputed from `results/ecro_attr95a/deliv.npz` (per-patch exitance, deployed and 64 draws per arm, reference
  halves) with own code; no audited metric function called. Image-level E_H recomputed on the server from the stored renders.
- (b) WL95 / WL96 numpy evidence: rank-16 ring-0 counterexample re-derived; deterministic oracle re-run on a copy and compared with the
  path-traced reference; carrier-ladder scratch outputs read; WL96 Amdahl arithmetic recomputed from `life91.json`.
- (c) WL96A: field metrics recomputed from `tails.npz` against `CKF_none.npz` (copied from the server); image decision recomputed from the
  stored renders; four additional audit renders (1 rep, 4,096 spp, about 10 s GPU each, own directory) to isolate the e3_o-only reference
  noise and the e3_o image share; sensitivity to debiasing, threshold and mask.
- (d) WL87–WL91: quoted-number and rule checks from stored JSONs; WL89 stop / amendment artifacts inspected.

Verdict vocabulary: VERIFIED | CONDITIONALLY_VALID | INVALID_COMPARISON | IMPLEMENTATION_DEFECT | UNREPRODUCIBLE | UNTESTED_INFERENCE.

---

## 2. Headline results

1. **Every stored number that was recomputed reproduces** (max deviation 0 to 3e-17 for WL95A delivery, WL95A image E_H, WL96A field metrics
   and the WL96A image decision). No arithmetic or metric-code defect was found in WL95A or WL96A. WL95A uses the corrected angular weight
   W_T (4 rings x 16 sectors); the `an94.W` 8 x 8 layout defect is real but is confined to WL94–WL94D numbers.
2. **The 64-bin sender bias is real and is not a sampling artifact** (P0-2): pooled box-A flux ratio 1.544 +/- 0.035 (64 independent relation
   draws), and an independent noise-free deterministic quadrature reproduces the same bias (1.42 for a 64-bin table of the physical
   once-scattered field, unchanged under spatial refinement, removed by 1,024 angular bins).
3. **The deployed-arm numbers of WL95A are single-draw values of a wide distribution.** The deployed relation draw 86001 is a low draw
   (A4 flux 1.208 vs pooled 1.544; z = -1.2; its abs rel L1 0.524 is lower than 83 % of draws). The frozen label
   `FLIGHT_ATTRIBUTION = ANGULAR_PROJECTION_EVIDENCED` follows from the frozen rule, but the rule compares a single-draw L1 with a pooled L1;
   had another pooled draw been the deployed one, the same rule gives ANGULAR 22 / MIXED 30 / SAMPLING_DOMINANT 12 of 64.
4. **"Exact per-ray evaluation closes the delivery error" holds in expectation only** (P0-3): X1 pooled 1.022 +/- 0.050, abs rel L1 0.111
   (reference noise 0.059); a single X1 draw at the deployed budget has mean abs rel L1 0.737 (A4: 0.761), and the deployed X1 draw is worse
   than the deployed A4 draw (0.627 vs 0.524).
5. **"RANK16 INADEQUATE" is measured against M64, which is itself 1.5x biased against the reference.** Against the reference, the
   rank-32 arm delivers better than M64 (L1 0.386 vs 0.546, z = -5.1), the rank-16 NMF arm has a better flux (0.975 vs 1.544) and a lower ROI
   E_H (0.0209 vs 0.0245), and the learned sender s8601 is statistically tied with M64 (L1 +0.023 +/- 0.035). The ring-0 counterexample is
   correct but nearly trivial (16 input bins) and cuts both ways.
6. **WL96A S1_NO_HEADROOM is robust** (P0-4): the debiasing term is statistically imperfect (it subtracts e3_p noise that the tested
   difference does not contain, 0.00146 vs the appropriate 0.00071), but without any debiasing the statistic is 0.00069, a factor 7 below
   M_img = 0.005, in every mask examined (maximum 0.0024). The explicit e3_o reproduces the reference e3_o image contribution to about 1 %.
   It is an oracle-prefix ceiling on one light.
7. **Quotation errors**: "0 better / 3 worse / 36 equal" (WL96) should read 0 / 3 / 33 of 36; the WL96 source-update Amdahl ratio uses
   the edit event's sweep time (10.6 ms) where the source event's own is 9.4 ms (1.14x, not 1.16x). Neither changes a decision.

---

## 3. Findings

Format: id — category — material — claim affected — original vs independent — evidence — decision impact — proposed verdict — confidence.

### C2-01 — confirmation — material: no — WL95A delivery table (all arms)
- Original (`results/ecro_attr95a/an.json`) vs recomputed (`results/audit97/C2/wl95a_delivery_recompute.json`): identical to the printed
  precision for all arms (deviation 0.0e+00).

  | arm | deployed ratio / L1 | pooled ratio (SE) | pooled L1 (jackknife SE) |
  |---|---|---|---|
  | A0 learned s8601 | 0.495 / 0.588 | 0.517 (0.005) | 0.569 (0.004) |
  | A0 learned s8602 | 1.534 / 0.901 | 1.630 (0.014) | 0.965 (0.014) |
  | A1 NMF16 | 0.921 / 0.623 | 0.975 (0.010) | 0.655 (0.008) |
  | A2 L1-NMF16 | 0.231 / 0.779 | 0.244 (0.003) | 0.762 (0.003) |
  | A3 rank-32 | 1.048 / 0.480 | 1.127 (0.018) | 0.386 (0.018) |
  | A4 = M64 | 1.208 / 0.524 | 1.544 (0.035) | 0.546 (0.035) |
  | M64_spool | 1.158 / 0.483 | 1.495 (0.035) | 0.500 (0.034) |
  | q256 | 1.077 / 0.534 | 1.422 (0.034) | 0.433 (0.033) |
  | X1c | 0.719 / 0.388 | 1.037 (0.063) | 0.268 (0.045) |
  | X1 | 0.764 / 0.627 | 1.022 (0.050) | 0.111 (0.041) |

  Reference half-split noise n0 = 0.0593 (box A), 0.0698 (panel face). Attribution: E0 0.4645, Rs 0.0242 (SE 0.034), Ra 0.2319 (SE 0.047),
  Rp 0.1571 (SE 0.060), threshold 0.25 E0 = 0.1161 — all identical to the stored values [MEAS].
- Evidence: `experiments/audit97/C2/c2_wl95a_delivery.py`, `c2_wl95a_verdict_sensitivity.py`; metric definition
  `experiments/ecro_attr95a/protocol/ecro_attr95a.json` (`metrics.delivery`, `decisions`).
- Proposed verdict: VERIFIED (numbers). Confidence: high.

### C2-02 — confirmation — material: yes — P0-2 (64-bin sender bias not removed by relation sampling)
- Pooled A4 flux ratio 1.544, SE 0.035, bootstrap 95 % CI [1.475, 1.612]; halves of the 64 draws 1.541 / 1.547; pilot 16 draws 1.562; lag-1
  autocorrelation of draw totals -0.05 (draws use independent seeds 950000 + d) [MEAS]. Against the deterministic oracle: ratio 1.521,
  z = 14.9; chi-square per patch 156 [MEAS, first attempt, re-read].
- Source-point sampling (A4 -> M64_spool, paired over common draws): L1 -0.047 +/- 0.002, ratio -0.049 +/- 0.002: significant, small.
- Independent estimator [READ, scratch outputs of the WL96 designers, noise-free deterministic quadrature, no relation sampling]:
  64-bin patch table of the physical once-scattered field: ratio 1.420, L1 0.428 (`results/wl96_calc/scratch/d3/res_ladder_L0_0.json`);
  spatial subdivision x4 / x16 / x64 at 64 bins: 1.419 / 1.420 / 1.383; 1,024 bins: 1.048 / 0.103; 1,024 bins + x4 cells: 1.012 / 0.027;
  "outgoing bin-centre direction, exact position": 1.423 / 0.527 (`d4/ck2_model2_ck2.json`). The pipeline's pooled q256 (same quantity
  with Monte Carlo relations) is 1.422 / 0.433: agreement to three digits between two code paths.
- Cross-workload [MEAS from `results/gm1_gap90/results/final90.json`]: on GM1 W1 T1 R_mirror the explicit-vs-oracle gap 0.0226 equals the
  angular-only projection 0.0226; spatial-only -0.0009 (paired noise 1e-4 or less).
- Decision impact: supports the claim as worded. Restrictions: one CK2 light (grazing omni, incidence 80.6–88.6 degrees from the panel normal,
  i.e. only 4 of 64 incident bins, all in ring 0), alpha about 0.03–0.18, Lambert receivers.
- Proposed verdict: VERIFIED (on the tested workload). Confidence: high.

### C2-03 — statistical_weakness — material: yes — WL95A `FLIGHT_ATTRIBUTION = ANGULAR_PROJECTION_EVIDENCED`; every "deployed" number
- The frozen rule defines E0 = E(A4, **deployed single draw**) and Rs = E0 - E(M64_spool, **pooled**). An abs-rel-L1 of a single draw
  contains the draw's variance; the L1 of a 64-draw mean does not. The two are not commensurable, so Rs is not an estimate of "what
  sampling contributes".
- The deployed relation draw 86001 is not one of the 64 pooled draws (minimum L1 distance to any pooled draw 0.72 of its flux) and is a
  low-flux draw: A4 ratio 1.208 vs pooled 1.544 (draw SD 0.283, z = -1.18, 55 of 64 draws higher). Its L1 0.524 is below the pooled-mean
  L1 0.546 and below 83 % of single draws (mean single-draw L1 0.761) because the low flux partly cancels the positive bias [MEAS].
- Re-applying the frozen rule with each pooled draw in the role of the deployed one: ANGULAR_PROJECTION_EVIDENCED 22, MIXED 30,
  SAMPLING_DOMINANT 12 of 64; median E0 0.667 (deployed 0.464), median Rs 0.227 = 34 % of E0 [MEAS,
  `results/audit97/C2/wl95a_verdict_sensitivity.json`].
- Commensurable decomposition of a typical single draw's L1 (all vs the reference): mean single draw 0.761 -> pooled 0.546 (relation
  variance 0.215) -> M64_spool 0.500 (source points 0.047) -> X1c 0.268 (direction 0.232) -> X1 0.111 (position 0.157) -> n0 0.059.
- Image level shows the same single-draw effect: ROI E_H excess over the oracle-table floor is +0.0034 for the deployed A4 table and
  +0.0126 for the pooled A4 table (floor 0.0118) [MEAS on the server, identical to `an.json`].
- Sixteen fixed source points: 64 source redraws at the deployed relation give M64 ratio 1.158 +/- 0.090 (deployed 1.208) and learned
  s8601 0.480 +/- 0.036 (deployed 0.495, rank 0.73), s8602 1.479 +/- 0.102 (deployed 1.534) [MEAS, own numpy flight on the server,
  `results/audit97/C2/learned95a.json`; the same code reproduces the stored deployed values 0.4954 / 0.5879, 1.5342 / 0.9007, 1.2082 / 0.5237].
- Decision impact: the *bias* conclusion (C2-02) is unaffected. The *label* "sampling does not count" is specific to draw 86001; for a
  typical draw, relation-sampling variance and angular projection contribute comparable L1. Statements of the form "deployed A4 = 1.21x"
  or "dF = 0.0034" must not be read as properties of the estimator.
- Proposed verdict: CONDITIONALLY_VALID (label follows the frozen rule; interpretation "sampling is not a contributor" is not supported).
  Confidence: high.

### C2-04 — invalid_inference / statistical_weakness — material: yes — P0-3 (continuous exact-direction and hit-point ER evaluation closes the error)
- Pooled over 64 draws: X1 ratio 1.022 +/- 0.050 (bootstrap [0.93, 1.13]), L1 0.111 (jackknife SE 0.041; expected L1 under "unbiased" given
  its own SE and the reference noise is about 0.15, 90 % range 0.10–0.22) -> consistent with no bias. Against the deterministic oracle:
  ratio 1.007, L1 0.098, chi-square per patch 0.80 [MEAS].
- Per draw at the deployed budget (512 rays per patch, 8 per bin): X1 mean single-draw L1 0.737, median 0.617, worst 2.21; ratio median 0.92,
  67 % of draws below 1, kurtosis 7.3 (X1c: kurtosis 27, one draw at 4.25x). Deployed draw: X1 0.764 / 0.627 vs A4 1.208 / 0.524.
  On the same draw index X1 has the lower single-draw L1 than A4 in 69 % of the 64 draws (median single-draw L1 0.617 vs 0.726): a
  modest per-draw advantage, far from the pooled 0.111 vs 0.546 [MEAS].
- Step attribution A4 -> q256 -> X1c -> X1 (pooled flux ratio steps, paired jackknife): A4 -> M64_spool -0.049 +/- 0.002;
  M64_spool -> q256 -0.073 +/- 0.004 (incident side); q256 -> X1c -0.385 +/- 0.065 (outgoing direction); X1c -> X1 -0.015 +/- 0.078
  (not significant in flux). In L1: Rp = 0.157 +/- 0.060 (bootstrap CI [0.001, 0.26], p(<= 0) = 0.025); dropping the 1 / 2 / 4 largest
  X1c draws gives 0.140 / 0.118 / 0.109 [MEAS]. The noise-free oracle gives a patch-centre ("X1c-like") L1 of 0.269 against exact
  (`d3/res_exact_L0.json`), so the spatial term is real; its pooled Monte Carlo estimate is only marginally resolved.
- Cleanliness of the steps: q256 -> X1c changes two things at once — outgoing direction becomes exact per ray, and the incident
  geometry collapses from 256 stratified points per patch to the patch centre. The step is therefore "outgoing angular exactness minus
  an incident-position collapse", not a pure outgoing-direction effect. The sign and size of the outgoing effect are nevertheless
  unambiguous from the oracle ladder (C2-02). A4 -> q256 also changes the source sampling (16 points -> 256) together with the incident
  projection; M64_spool separates these.
- Decision impact: "closes the delivery error" is true for the estimator's expectation (bias removed) and false for a deployable
  single-realization delivery at the tested ray budget. WL95A's own text says this ("high variance"); P0-3 as worded omits it. WL91
  reached the same split on GM1 (pooled X1 = oracle; held-out single draws fail).
- Proposed verdict: CONDITIONALLY_VALID (bias closure VERIFIED; per-realization closure not shown). Confidence: high.

### C2-05 — invalid_inference — material: yes — WL95A `RANK16_DELIVERY = INADEQUATE_ON_TESTED_WORKLOAD` and its use as a capacity statement
- The label follows the frozen criteria [MEAS]: A1 flux vs A4 0.631 (SE 0.014, outside [0.9, 1.1]), map L1 vs A4 0.649 (SE 0.010, above 0.10);
  A2 0.158 / 0.842; criterion (iv) (ROI E_H) passes for both.
- The yardstick is fidelity to A4 = M64, which is itself biased against the reference (C2-02). Against the reference [MEAS,
  `wl95a_verdict_sensitivity.json`, pooled, paired jackknife]:
  - A3 (rank 32) - A4: L1 -0.161 +/- 0.031 (z = -5.1): better than the full-rank bin operator;
  - A1 (rank 16) - A4: L1 +0.109 +/- 0.035 (z = 3.1), but flux 0.975 vs 1.544 and ROI E_H 0.0209 vs 0.0245 (paired image noise 0.0002);
  - learned s8601 - A4: L1 +0.023 +/- 0.035 (z = 0.6): tied; learned s8601 - A1: -0.086 +/- 0.008 (learned better than the offline bound);
  - learned s8602 - A4: +0.418 +/- 0.036.
- Ring-0 counterexample (WL96, `results/wl96_calc/ring_alloc.py`) re-derived [MEAS, `results/audit97/C2/rank16_ring0.json`]:
  - CK2 panel incidence: bins 12–15 only, also under dense sampling of the whole panel (shares 0.023 / 0.205 / 0.356 / 0.417; original
    16-point values 0.023 / 0.209 / 0.356 / 0.412).
  - M64 restricted to ring-0 inputs has rank 16 per channel; restricted to the CK2 support, rank 4. An exact non-negative rank-16
    factorisation on ring-0 inputs exists (error 0.0, albedo max 0.686). This is true by counting: ring 0 has 16 input bins, so any
    16 x 64 block has rank at most 16 (factor = identity on the ring-0 inputs times the ring-0 rows of M64).
  - The same factorisation has relative error 1.0 on rings 1–3. A ring-0-weighted (x10) signed rank-16 fit gives ring-0 error 0.0045 with
    rings 1–3 at 0.79–0.99. Signed SVD truncation with uniform flux weighting needs rank about 48 for 5 % on the CK2 support
    (rank 16 / 24 / 32 / 48: 0.585 / 0.411 / 0.204 / 0.052). Ring-0 rows carry 6.4 % of the A1 fit weight.
  - What it implies: on the CK2 workload rank 16 is not the binding capacity; the A1 / A2 failure is an allocation property of an
    all-incidence fit objective. What it does not imply: that one rank-16 material operator valid for all incidence can represent M64
    (the evidence says it cannot), or that a rank-16 operator equal to M64 would deliver correctly (it would inherit the 1.5x bias).
- Decision impact: "rank 16 is inadequate" should not be used as the cause of the sharp delivery error, in either direction. The
  tested rank-16 fits fail to reproduce M64; M64 fails to reproduce the reference; a higher-rank fit (A3) is closer to the reference than
  M64 by compensation of errors. The dominant error is the 64-bin outgoing interface shared by all table arms.
- Proposed verdict: CONDITIONALLY_VALID for the label as defined ("tested rank-16 fits do not reproduce M64 delivery");
  UNTESTED_INFERENCE for "rank 16 limits sharp delivery". Confidence: high (numbers), medium (interpretation).

### C2-06 — statistical_weakness / confirmation — material: yes — P0-1 within the delivery lens (learned S1 sender error; passivity)
- Error vs the reference is large for both seeds and has opposite sign: s8601 0.517x (L1 0.569), s8602 1.630x (L1 0.965); relation-draw
  SD only 0.036 / 0.108 and source-draw SD 0.036 / 0.102, so these are not single-draw artifacts [MEAS]. The two training seeds disagree by
  a factor 3.2 in delivered flux.
- "Avoidable" depends on the comparator [MEAS / READ `main.json`]:
  - vs the exact per-ray carrier X1 (L1 0.111): avoidable 0.46–0.85, but only by leaving the 64-bin interface;
  - vs the exact 64-bin operator M64 (L1 0.546): s8601 not distinguishable (+0.023 +/- 0.035), s8602 +0.418;
  - vs the offline rank-16 bound A1 (L1 0.655): s8601 is better (-0.086 +/- 0.008);
  - at the sender-field level vs M64 s16 (W_T): learned rel L1 1.04 / 1.12, energy 0.43 / 0.89, peak 0.038 / 0.129; the rank-16 bound A1
    is 0.957 / 0.458 / 0.018. The learned sender is at the level of its own class bound and at the zero-operator level (rel L1 about 1).
- Passivity [MEAS, own computation from the frozen factors, `results/audit97/C2/learned95a.json`; reproduces the stored maxima exactly]:
  directional albedo sum_d A[m, d] W_T[d] of the learned S1 operators: max 2.264 (s8601) / 3.077 (s8602); 10.4 % / 15.4 % of the 8,064
  (patch, incident bin, channel) entries exceed 1; every S1 patch has at least one entry above 1; on the bins actually lit in CK2 the
  maximum is 1.44 / 1.79 with 4.2 % / 41 % of lit entries above 1; irradiance-weighted effective albedo on CK2 0.28 / 0.58 overall
  (M64: 0.65), per-patch maximum 1.12 / 1.44. Over all 1,270 patches: 0.7 % / 1.7 % of entries above 1, 27 % / 41 % of patches with at
  least one. M64 itself: albedo max 0.825 (passive).
- What was not measured by anyone: passivity of the complete deployed operator (local + global term) as a coupled-solve property
  (spectral radius or energy balance on realistic fields). The violation shown is per incident bin of the S1 sender factor; on the CK2
  field the aggregate S1 output is below the incident energy, and one patch emits 1.12x / 1.44x its incident energy.
- Proposed verdict: P0-1 first half CONDITIONALLY_VALID ("large error" VERIFIED; "avoidable" holds against a continuous carrier or a
  CK2-support-aware allocation, not against the tested rank-16 class bound or, for s8601, against M64); second half VERIFIED as
  "the learned S1 sender factor violates the per-bin directional albedo bound, including on CK2-lit bins"; "the deployed operator violates
  passivity" as a system property is UNTESTED_INFERENCE. Confidence: high (numbers), medium (scope of "avoidable").

### C2-07 — evaluation_defect (weak power) — material: yes for `NEXT_DESIGN_TARGET`, no for other labels — WL95A image-level tolerance
- The frozen E_H tolerance is tol = max(2 x max(noise bounds), 0.005) = 0.0167, where the noise bound is the an94.eh unpaired half-split
  bound (about 0.0083). The arm tables are rendered with identical camera random numbers, so arm-vs-floor differences are paired:
  the half-split of the *difference* is 0.0001–0.0003 [MEAS, `img95a_recompute.json`]. The tolerance is about 60–80x the noise of the
  quantity being compared.
- Consequences: (i) no image-level component difference can count unless it exceeds 0.0167 (pooled A4 excess 0.0126, learned s8601
  0.0130, learned-global s8601 0.0133 all "within noise" although each is about 50 paired SE from zero); (ii) rank-16 criterion (iv)
  passes trivially; (iii) `NEXT_DESIGN_TARGET = UNDETERMINED` is partly produced by the tolerance. Re-evaluating with a paired tolerance
  of 0.005: seed 8601 dS 0.0130, dF 0.0034, dG 0.0133 — still no single dominant component; seed 8602 dS 0.0354 dominates. The label stays
  UNDETERMINED (both seeds required), so the outcome is unchanged, but for a different reason (seed disagreement, not noise).
- dF uses the deployed A4 draw (0.0034); with the pooled A4 table it is 0.0126 (C2-03).
- WL96A corrected this by using the paired statistic.
- Proposed verdict: CONDITIONALLY_VALID (label follows the rule; the rule has almost no power). Confidence: high.

### C2-08 — confirmation — material: yes — deterministic oracle vs path-traced reference (WL96: 1.015 / 0.068 vs n0 0.059)
- Re-run of an unmodified copy of `results/wl96_calc/scratch/d3/toy96.py validate` (sha256 044efed109a6c299): ratio 1.01516, rel L1
  0.06813, corr 0.99507, reference half noise 0.05927 — identical to the stored `res_validate.json` [RERUN, 23 s CPU].
- n0 = 0.5 sum|a - b| / sum is the expected L1 of the two-half mean against the truth, so 0.068 against 0.059 is the right comparison.
  Cross-validated squared bias (oracle - a)(oracle - b) summed over patches is -0.0009 of the oracle's squared norm (noise term 0.0081):
  no resolvable bias; total flux difference -0.6 SE of the reference total [MEAS, `wl95a_consistency.json`].
- Limits: the reference is 2 x 4,096 spp with 5.9 % box-A L1 noise and a 2.5 % total-flux half difference, so an oracle or reference
  bias of up to about 3 % cannot be excluded (WL96 states the same). One light, one geometry. The toy's ER model was not re-derived
  by this track (physics correctness of the oracle belongs to another track); it is, however, a separate code path from pt86 / pt86a.
- Proposed verdict: VERIFIED (agreement within reference noise, about +/- 3 % resolution). Confidence: medium-high.

### C2-09 — confirmation with restrictions — material: yes — P0-4 accuracy (WL96A `S1_NO_HEADROOM`)
- Field metrics from `tails.npz` vs `CKF_none.npz`: maximum deviation from `tails.json` 2.9e-17; render tables rebuilt to 4e-8 [MEAS].
- Image decision recomputed from the stored renders [MEAS, `results/audit97/C2/img96a.json`]: dI(XOD) 0.000646, n_img 0.001464,
  nb 0.0000218, H -0.000817, U -0.000774 — identical to `s1.json`. Outcome S1_NO_HEADROOM (U < 0.005).
- Validity of the debiasing n_img = 0.5 dI(Fa, Fb):
  - Fa / Fb are built from the half tables of the whole e3 = e3_p + e3_o (`w96a.py:314`), whereas XOD differs from F only in e3_o
    (reference e3_p is shared). n_img therefore includes e3_p reference noise that dI(XOD) cannot contain. An audit render with only the
    e3_o halves varied gives n_img(e3_o only) = 0.00071 in ROI_T (frozen 0.00146), 0.00103 in DIFF (frozen 0.00101) [MEAS, 4 own renders].
  - Subtracting an L1 noise from an L1 difference is not an unbiased estimator of the L1 of the noise-free difference
    (|bias + noise| is not |bias| + |noise|); H is negative here, which an error magnitude cannot be.
  - With the appropriate noise term H = -0.00006; with no debiasing U_raw = dI + 2 nb = 0.00069. Both are a factor 7 or more below M_img.
  - A cross-validated L2 estimate (XOD - Foa)(XOD - Fob) gives an XOD-vs-truth image error of about 0 (-0.1 %) relative to the e3_o image
    in ROI_T, with a noise level of 1.1 % [MEAS].
- Paired statistic nb: appropriate for render noise (paired camera seeds); it does not cover relation-draw variability, which is small here
  (pooled XOP dI 0.00060; per-realization e3_o table L1 median 0.038, worst 0.053 against n0 0.032).
- M_img = 0.005: inherited convention ("E_H tolerance floor used by every WL93D–WL95A rule"), predeclared, not derived from a perceptual or
  task criterion. In relative terms it is 2.8 % of the tail's image contribution in ROI_T (tail share 0.178) and 8 % of the e3_o
  contribution (share 0.062) [MEAS]. The outcome is S1_NO_HEADROOM for any M_img >= 0.0007 (1.1 % of the e3_o image).
- Mask sensitivity (U_raw, no debiasing) [MEAS]: ROI_T 0.00069; DIFF 0.00130; whole frame 0.00100; panel pixels 0.00133; complement of
  ROI_T 0.00107; per-surface masks 0.00066–0.00243. No mask approaches 0.005. For comparison in ROI_T: zero tail 0.178, learned
  global terms G8601 / G8602 0.0255 / 0.0126, full explicit tail XD 0.0029.
- Regions the ROI does not read: the panel receivers' own e3_o tables have a deployed table excess 0.043 (0.090 - 0.047) and pooled 0.016
  (jackknife SE 0.002; below the frozen n0 / 2 = 0.024 flag; the subtraction of n0 is itself approximate, so the excess is nominally but
  not securely above zero; cross-validated relative L2 bias +0.038 pooled, +0.079 deployed).
  The camera does see the panel: on panel pixels dI(XOD) is 0.00124 against an e3_o-only noise of 0.00093, i.e. no headroom at 0.005 for
  this view. For other views of the panel the claim is not measured; an upper estimate from the table excess times the e3_o share on
  the panel (0.139) is about 0.006 deployed / 0.002 pooled of the panel total [INF].
- Restrictions that the label does not carry: (i) the explicit tail is fed the **reference** 2-event field (`w96a.py:250`,
  `tail(op, mp, e2(rf))`) — an oracle-prefix ceiling by design, not a deployed-system comparison; (ii) one light, one geometry, no edit;
  (iii) the tail is truncated at three sweeps (flux 0.993–0.996 of the reference); (iv) N = 1,270 patches, 512 rays per patch.
- Proposed verdict: VERIFIED on the tested CK2 workload with restrictions (i)–(iv). Confidence: high.

### C2-10 — confirmation with restrictions — material: yes — P0-4 lifecycle cost (WL96A U; WL96 Amdahl estimate)
- WL96A [MEAS from `cost_s1.0.json`]: U = wall(R0) / wall(arm) recomputed identical: ZERO 1.038 (edit) / 1.043 (source); LAM24 0.932 / 1.040;
  LAM64 0.921 / 1.038; criterion 1.25. The three tail sweeps are 3.3 ms = 2.3 % of the 144.1 ms edit and 3.7 % of the 88.5 ms source
  event; render 59.5 ms (41 %), q table 25.7 ms, geometry update 30.2 ms, relation update 23 ms.
- Weaknesses: medians of 6 edits / 16 source events with no dispersion stored; the ZERO edit gain (5.2 ms) exceeds the removed stage
  (3.3 ms), and the q-table component is 31–33 ms in the LAM edit arms against 25.5–25.7 ms in R0 / ZERO, so timing noise or arm-order
  effects are about 2–7 ms, the size of the effect under test. This cannot move U by the factor needed (1.04 -> 1.25).
- WL96 Amdahl anchors [MEAS from `results/gm1_c3urx1_91/results/econ/life91.json`]: 124.7 / 69.4 / 75.8 / 21.4 ms are the **C3URX1** arm
  (edit wall, edit minus render, single-light source wall, source minus render); sweeps 10.6 ms is C3URX1's `transport_orders` in the
  **edit** event. 124.7 / 114.1 = 1.093 (as quoted). For the source event the arm's own sweep time is 9.4 ms: 75.8 / 66.4 = 1.142, not
  1.16; "source without render" 21.4 / 12.0 = 1.78, not 1.98. Other arms: C3UR 1.090 (edit) / 1.135 (source); R0 1.040 / 1.070.
- Is C3URX1 the right denominator? It is the cheapest explicit arm of WL91 and one that WL91 itself did not adopt
  (BASELINE ADOPTION EVIDENCE INSUFFICIENT), measured on GM1 W1, not on CK2 and not in an ECRO pipeline. Choosing it makes the ceiling
  slightly *higher* than the adopted-class arms would (R0 1.04), so the choice is not favourable to the negative conclusion. The estimate
  is nevertheless a property of this implementation and scale: GPU kernel time is 17–18 % of the edit wall, a 64-spp 512 x 512 frame
  render is 38–44 %, and the sweeps act on 1,270 patches. WL96A's direct CK2 measurement (1.04) supersedes the cross-workload estimate.
- Not tested: larger N (the scaling extension was gated on U(ZERO) >= 1.15 and did not run), more bounces, a pipeline in which the frame
  render is not the dominant term.
- Proposed verdict: VERIFIED at the tested size and implementation; UNTESTED_INFERENCE beyond it. Confidence: high (arithmetic), medium
  (generality).

### C2-11 — provenance_gap / quotation — material: no — "0 better / 3 worse / 36 equal" (WL96 line 149, quoting WL87)
- `final87.json`: 36 region rows; rule-based (both seeds beyond the larger unpaired noise bound) ECRO better 0, worse 3
  (E1_T1 obj_panel, E2_T1 C, E2_T1 obj_panel), within bound **33**. WL87's own table says "0 / 3 / 36" with 36 as the row count.
- Raw signs: both seeds worse than explicit in 23 rows, both better in 5 (E2_T0 / E2_T1 / E3_T0 / E3_T1 obj_sphere, E3_T0 obj_panel),
  mixed in 8; mean difference ECRO - EXP +0.0029 / +0.0021 per seed [MEAS, `wl87_91_96_checks.json`].
- "Explicit same-state transport >= ECRO everywhere" is therefore an overstatement of a rule outcome whose "equal" class uses a
  conservative unpaired noise bound: the accurate statement is "never significantly better, significantly worse in 3 of 36, nominally
  worse in about two thirds". WL87's predeclared label for LEARNED_OPERATOR_ADDITIONAL_UTILITY was SUPPORTED (source-update cost), which
  later summaries cite only for its fidelity half.
- Proposed verdict: CONDITIONALLY_VALID (direction supported; "everywhere" and "36 equal" are not exact). Confidence: high.

### C2-12 — confirmation — material: no — "accuracy headroom about 0.001 (WL90 P2plus <= 0.0009)"
- `final90.json`, R_mirror E_H, P2plus - EE: +0.00090 (W1 T1), +0.00071 (W1 T0), -0.00043 (W2 T1), paired noise 1e-5 to 2e-5 [MEAS]. Correctly
  quoted. It measures the cost of the 64-bin projection applied only at orders >= 2 in an explicit GM1 pipeline; WL95 itself labels it
  cross-regime. WL96A's direct CK2 value (dI(XOD) 0.00065) is of the same size.
- Proposed verdict: VERIFIED (quotation), scope GM1. Confidence: high.

### C2-13 — confirmation (procedural) — material: no for P0 — WL89 stop, strict-digest false positive, post-freeze amendment
- Stored artifacts [READ]: `results/gm1_relation89/results/orig_stop/chain.status` ends `GATE TESTS FAIL`, `STOP STOPPED_INVALID_COMPARISON`
  at 2026-11-12T04:35:33Z, verdict step rc = 1; `amend1/audit89.json`: strict digests differ, operational digests equal; mismatches: 2
  entries, `dbin` only, both invalid in selective and full, weights 0; 22 operator tensors equal; CPU transport outputs bitwise equal;
  perturbation of 157,538 invalid entries (random / 255 / NaN) leaves operator tensors and outputs unchanged; `AUDIT_PASS: true`.
  Relation / transport tables in `results/tables` carry timestamps after the preserved stop.
- Assessment: the stored evidence supports "the stop was caused by a test that hashed non-semantic lanes". The amendment changed a test
  criterion, not an outcome rule, before any candidate table existed, was disclosed and user-approved. The audit evidence was produced
  by the conflicted author; this track did not re-execute it (it needs the WL89 run directory and GPU).
- Downstream: WL89's SUPPORTED-scoped C3UR result, and WL90 / WL91 which reuse its workload, rest on the amended gate. The P0 claims use
  from this chain only (i) the WL91 timing anchors (C2-10) and (ii) the WL90 P2plus value (C2-12), neither of which depends on the amended
  test. WL89 additionally records that its scaling *update* times are invalid (first-call compilation) — they are not used in WL96.
- Proposed verdict: CONDITIONALLY_VALID (procedurally sound on the stored evidence; not independently re-executed). Confidence: medium.

### C2-14 — statistical_weakness — material: no for P0 — WL91 utility rule (a) and held-out single-draw variance
- WL91 QUALITY-MATCHED ECONOMIC UTILITY = NOT SUPPORTED rests on rule (a) ("no slowdown > 1.25x at any common tier") violated by one
  deployed draw (W2 T1, R_mirror tier 0.06: 0.74 s vs 0.24 s), while rules (b) and (c) are met [READ, worklog 91 lines 285–290]. A
  maximum-type rule over draws x tiers applied to a heavy-tailed single-draw estimator has a high false-fail probability; the result
  reflects the estimator's variance (consistent with C2-04), not a mean cost disadvantage (lifecycle 0.57–0.90x C3UR).
- It concerns an explicit comparator's adoption, not ECRO; later P0 claims do not depend on it.
- Proposed verdict: CONDITIONALLY_VALID. Confidence: medium (not recomputed from raw tiers).

### C2-15 — implementation_defect (historical, confirmed) — material: yes for WL94–WL94D numbers, no for WL95A / WL96A — `an94.W`
- `experiments/ecro_mat94/an94.py:19` sets `NB, NPHI, NZ = 64, 8, 8` and line 32 builds W from it; the representation is 4 rings x 16
  sectors (`experiments/ecro86/scene86.py:15`, bin index = ring x 16 + sector). W94 weights eight pseudo-rings of eight bins.
- WL95A states "W_T (corrected) everywhere" and this track's recomputation with W_T = ((d // 16) + 0.5) / 4 x 2 pi / 64 reproduces every
  WL95A number, so WL95A and WL96A are on the correct weights [MEAS].
- Size of the effect on learned-arm sender metrics [READ `main.json`, bridge values]: energy vs M64 s16 with W94 / W_T: s8601 0.557 / 0.433,
  s8602 1.123 / 0.891 (crosses 1); rel L1 1.075 / 1.041 and 1.221 / 1.118. WL95's statement "<= 0.02 L1, <= 0.04 energy" applies to
  the bound arms only; for the learned arms the shift is up to 0.23 in energy.
- Proposed verdict: IMPLEMENTATION_DEFECT for WL94–WL94D exitance-weighted metrics (magnitude changes, direction of conclusions
  unchanged in the numbers seen here); WL95A onward unaffected. Confidence: high.

### C2-16 — dataset_limitation — material: yes — single workload behind P0-2 / P0-3 / P0-4
- All CK2 results are one geometry, one light (grazing, 4 incident bins of ring 0), one camera, one material variant ("none"), two
  training seeds, one deployed relation realization, two reference halves. The cross-workload support is GM1 (WL90 / WL91), same scene
  family and code base. Statements about architecture classes rest on this one family.
- Proposed verdict: applies as a restriction to every P0 verdict. Confidence: high.

---

## 4. Priority-0 assessment (within this track's lens)

| claim | verdict | basis |
|---|---|---|
| P0-1 learned S1 sender has large avoidable error; deployed learned operator may violate passivity | **CONDITIONALLY_VALID** | Large error VERIFIED (0.52x / 1.63x, L1 0.57 / 0.97, not a draw artifact; seeds disagree 3.2x). "Avoidable" holds against a continuous carrier, not against the tested rank-16 bound (learned s8601 beats A1) nor, for s8601, against M64. Per-bin albedo violation of the S1 sender factor VERIFIED (max 2.26 / 3.08; 10–15 % of entries; exercised on CK2-lit bins); system-level passivity untested. |
| P0-2 M64 / 64-bin sender introduces systematic sharp-delivery bias not eliminated by relation sampling | **VERIFIED** (tested workload) | 1.544 +/- 0.035 over 64 independent draws; noise-free oracle ladder 1.42 at 64 bins irrespective of spatial refinement; GM1 WL90 angular-only = whole gap. |
| P0-3 continuous exact-direction and hit-point ER evaluation substantially closes the CK2 delivery error | **CONDITIONALLY_VALID** | Closes the bias (pooled 1.022 +/- 0.050, L1 0.111 vs n0 0.059; oracle 1.007). Does not close single-realization error at the tested budget (mean single-draw L1 0.737; deployed X1 worse than deployed A4). Hit-point (spatial) step marginal in the Monte Carlo data (Rp 0.157 +/- 0.060), confirmed by the oracle (0.27). |
| P0-4 learned broad higher-order resolvent has no meaningful accuracy or lifecycle-cost headroom over explicit recurrence on the tested CK2 workload | **VERIFIED** as worded, with restrictions | Accuracy: U_raw 0.0007 vs 0.005 in every mask; robust to the (imperfect) debiasing, to M_img down to 0.0007 and to the mask. Cost: Amdahl ceiling 1.04 measured. Restrictions: oracle (reference) 2-event prefix, one light, N = 1,270, frame-render-dominated lifecycle, scaling not run. |
| P0-5 negatives justify rejecting the tested ECRO mechanisms but not a general impossibility | **CONDITIONALLY_VALID** (statistical lens only) | The measurements support: the 64-bin outgoing table interface is biased on sharp delivery; the tested learned sender instances are far from the reference and non-passive per bin; an explicit broad tail leaves no measurable headroom given an exact prefix. They do not isolate *rank* as a cause (C2-05), do not show that a continuous-carrier learned sender is infeasible, and are one scene family. Rejection scoped to "the tested instances and the 64-bin interface on this workload" is supported; anything broader is UNTESTED_INFERENCE. |

---

## 5. Special checks owned by this track

| check | result |
|---|---|
| Rank-16 ring-0 counterexample | Reproduced (support bins 12–15; rank 16 on ring 0, rank 4 on the CK2 support; exact non-negative factorisation, error 0.0, albedo 0.686). Trivial by dimension count; zero outside ring 0. Shows A1 / A2 failure is allocation, not capacity, on CK2; does not show a deployable all-incidence rank-16 operator. See C2-05. |
| Source-point sampling and fixed random patterns | Relation draw 86001: A4 flux z = -1.2 in the pooled distribution, L1 at the 17th percentile; not among the pooled draws. 16 source points: +/- 8 % (M64) and +/- 7 % (learned) flux SD over 64 redraws. Deployed-arm numbers for A4 / q256 / X1c / X1 are dominated by draw variance (draw SD 0.27–0.50 of the reference flux); learned-arm numbers are stable (SD 0.04–0.11). See C2-03. |
| X1 outgoing-direction and hit-point changes | A4 -> M64_spool: source points only (clean). M64_spool -> q256: incident projection plus physical MIS table (clean enough; SE 0.004). q256 -> X1c: outgoing direction exact **and** incident position collapsed to the patch centre (not clean; dominant step -0.385 +/- 0.065). X1c -> X1: position restored (clean; flux n.s., L1 0.157 +/- 0.060, heavy-tailed). See C2-04. |
| Reference noise and significance thresholds | n0 box A 0.0593 recomputed; n0 is the correct expected-L1 scale for the two-half mean. E(X) = L1 - n0 is an approximate debias. WL95A image tolerance 0.0167 is 60–80x the paired noise (C2-07). WL96A n_img over-subtracts (C2-09). M_img 0.005 is conventional. |
| WL89 invalid comparison and downstream effects | Stored stop and amendment evidence is internally consistent with a non-semantic test false positive; no P0 number depends on the amended gate. See C2-13. |

---

## 6. Could not verify, and why

- The physics of the deterministic oracle (`toy96.py`, `er_np.py`) and of the path-traced event-split reference (`pt86a.py`): not
  re-derived here; only their mutual agreement and noise statistics were checked.
- Re-execution of the WL95A flight for the 64 pooled draws (GPU, Mitsuba): not repeated; `deliv.npz` was taken as raw data. The deployed
  delivery of the table arms was re-derived with an own numpy flight for M64 and both learned senders (exact match), which validates the
  flight / receive code path for the deployed realization only.
- WL89 amendment audit: read, not re-executed.
- WL91 tier timings behind rule (a) and the held-out 1 / 3 result: read from the worklog, not recomputed from raw files.
- WL87 per-region E_H values: taken from `final87.json`; the renders were not re-read. Whether explicit and ECRO renders in WL87 share
  camera seeds (which would allow a paired, more powerful comparison) was not checked.
- WL96A Stage 2 (conditional, not triggered) and the scaling extension (not triggered): nothing to audit.
- WL95 calculation files (`results/wl95_calc/wl95_calc*.json`): keys listed only; the "learned S1 operator at zero-operator level" claim
  was checked indirectly through the WL95A sender metrics (rel L1 1.04 / 1.12) rather than from these files.
- WL88 qualification statistics (Q5 SNR 3.7x vs 4x): out of the P0 dependency path, skipped.
- System-level passivity of the deployed ECRO operator: no artifact measures it.

---

## 7. Notes on inference versus measurement

- The statement "the outgoing 64-bin projection is the dominant cause" combines a Monte Carlo step that is not clean (q256 -> X1c) with a
  deterministic ladder that is clean; the conclusion rests on the ladder.
- The bound "about 0.006 of the panel total" for unread panel views (C2-09) is an estimate from table excess times image share, not a
  render.
- "Rank is not the binding constraint" (C2-05) is supported by measurements against the reference; it is specific to the CK2 incident
  support (ring 0 only).

---

## 8. Files produced by this track

- Scripts: `experiments/audit97/C2/` — `c2_wl95a_delivery.py`, `c2_wl95a_consistency.py`, `c2_wl95a_verdict_sensitivity.py`,
  `c2_rank16_ring0.py`, `c2_wl96a_fields.py`, `c2_wl87_91_96_checks.py`, server: `c2_img95a_server.py`, `c2_img96a_server.py`,
  `c2_srv_tables96a.py`, `c2_srv_learned95a.py`, `c2_srv_chain.sh`.
- Outputs: `results/audit97/C2/` — `wl95a_delivery_recompute.{json,log}`, `wl95a_consistency.{json,log}`,
  `wl95a_verdict_sensitivity.{json,log}`, `rank16_ring0.{json,log}`, `wl96a_fields_recompute.{json,log}`, `img95a_recompute.json`,
  `img96a.{json,log}`, `learned95a.{json,log}`, `wl87_91_96_checks.{json,log}`, `chain.out`, `CKF_none.npz` (copy of the server
  reference), `toy_copy/` (copy of the WL96 oracle script and its re-run output), `_interrupted_run_log.md` (coordinator's log of the
  first attempt).
- Server: `~/rest_x97/C2/` (own directory only; `R/` holds 4 audit renders and the audit table; existing `~/rest_*` directories were
  only read; `R/data` is a symlink to `~/rest_x94/data`).

---

## 9. Reproducibility appendix

Environment: local Python 3.11 + numpy (Git Bash, repository root); server LabServer63, `. ~/rest_x95a/env.sh; $PY`.

Input hashes (sha256 prefix):

| file | sha256[:16] |
|---|---|
| results/ecro_attr95a/deliv.npz | f2324a32effc80ca |
| results/ecro_attr95a/ops95a.npz | d4d02e5a2f423ce9 |
| results/ecro_attr95a/an.json | d8f4ebdc72d78472 |
| results/ecro_attr95a/main.json | 28d104991bcad6f7 |
| results/ecro_attr95a/s0.json | ea95017a511dcb0a |
| results/ecro_util96a/tails.npz | 04da33a8b57eeb32 |
| results/ecro_util96a/tails.json | 395f0166deedeb5b |
| results/ecro_util96a/s1.json | 92f5e763126d83b6 |
| results/ecro_util96a/T96A.npz | 8019cb39d03421dd |
| results/ecro_util96a/cost_s1.0.json | 103b775ed571ed26 |
| results/ecro_util96a/final.json | 1c625581262250e4 |
| results/audit97/C2/CKF_none.npz (= ~/rest_x94/results/fields/CKF_none.npz) | 3a6238a8a42d1db4 |
| results/gm1_c3urx1_91/results/econ/life91.json | 2f3fc40b5afb28cd |
| results/gm1_explicit87/results/final87.json | 3b0f8ed8389bea61 |
| results/gm1_gap90/results/final90.json | 837265aef23a76b0 |
| results/gm1_relation89/results/amend1/audit89.json | fcc881f341fbc4aa |
| results/wl96_calc/ring_alloc.json | 4c259a3ca68877a4 |
| results/wl96_calc/ck2_support_E.npy | e6ed8cdd04d6c309 |
| results/wl96_calc/scratch/d3/toy96.py | 044efed109a6c299 |
| results/wl96_calc/scratch/d3/res_validate.json | 35c5d32107e1ac61 |

Commands (repository root unless noted):

```
python experiments/audit97/C2/c2_wl95a_delivery.py            # bootstrap seed inside the script; -> wl95a_delivery_recompute.{json,log}
python experiments/audit97/C2/c2_wl95a_consistency.py         # oracle vs reference, chi-square, heavy-tail robustness
python experiments/audit97/C2/c2_wl95a_verdict_sensitivity.py # re-run in this session
python experiments/audit97/C2/c2_rank16_ring0.py              # rng 9702 (dense panel sampling)
python experiments/audit97/C2/c2_wl96a_fields.py
python experiments/audit97/C2/c2_wl87_91_96_checks.py         # re-run in this session
( cd results/audit97/C2/toy_copy && python toy96_copy.py validate )   # 23 s CPU

# server (own directory ~/rest_x97/C2; one GPU process at a time; total GPU about 1 min, peak 2.9 GB)
scp experiments/audit97/C2/c2_*server.py experiments/audit97/C2/c2_srv_* LabServer63:rest_x97/C2/
ssh LabServer63 'cd ~/rest_x97/C2 && . ~/rest_x95a/env.sh && CUDA_VISIBLE_DEVICES= $PY c2_img95a_server.py out/img95a.json'
ssh LabServer63 'cd ~/rest_x97/C2 && setsid nohup bash c2_srv_chain.sh > chain.out 2>&1 &'
#   chain: c2_srv_tables96a.py (audit tables F / Fp / Foa / Fob) -> mat94.py render (4 x 4,096 spp, seed 86600, about 10–14 s each)
#          -> c2_srv_learned95a.py (frozen OWN s8601 / s8602 factors; source redraws rng 951000 + d; torch peak 210 MiB)
#          -> c2_img96a_server.py (CPU)
```

Verified again in this session (not only inherited): n0, pooled / deployed ratios and L1 for A4, X1, X1c, q256, M64_spool and both learned
arms; E0, Rs, Ra, Rp and their jackknife SEs; the 22 / 30 / 12 label simulation; pairwise L1 differences; WL96A decision values against
`s1.json`; WL96A cost ratios; WL91 anchors and Amdahl arithmetic; WL87 counts; WL90 P2plus; WL89 stop / amendment artifacts; the
`an94.W` definition; completion and outputs of the server chain. Inherited from the first attempt and read but not re-executed here:
bootstrap intervals, chi-square per patch, heavy-tail drop-k table, WL96A field recomputation log, the rank-16 script output, the
oracle re-run, the two server analyses.
