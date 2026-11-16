# WL97 Phase 2 - Track J6: Adjudication of metric and evaluation defects in WL92-WL95A

Date: 2026-11-16. Role: adversarial adjudicator. Audit only: nothing trained, tuned, moved or modified.
Own files: `experiments/audit97/J6/j6_decisive.py` (numpy, reads stored JSON only), output `results/audit97/J6/j6_decisive.json`.
No server job was needed: every decisive quantity existed in stored results or in phase-1 recomputations (C1, A, E1, C2), which I spot-checked against the stored originals.

Findings adjudicated: F-MAP-08, A-02, C1-01, C2-15, E1-F9 (an94.W cluster); C1-02, C2-07 (noise statistic); C1-04 (mask); C1-05, C1-06, F-MAP-07, F0-13 (norms / M64 as truth); C1-08 (stops inside noise); C1-10 (confirmations).

## 1. an94.W (8 x 8 coded, 4 rings x 16 sectors true)

**Defect: confirmed (IMPLEMENTATION_DEFECT).** `experiments/ecro_mat94/an94.py:19` `NB, NPHI, NZ = 64, 8, 8`; `:32` `W = ((arange(NB)//NPHI)+0.5)/NZ*2pi/NB`. The protocol (`experiments/ecro86/protocol/ecro86.json:47`) and `scene86.py` `bin_dirs` use bin = iz*16 + iphi with z = (iz+0.5)/4. My own derivation (j6_decisive.json `weight_ratio_coded_over_true`) gives these coded/true ratios:
- ring 0: 0.5 for sectors 0-7, 1.5 for sectors 8-15
- ring 1: 0.833 / 1.167
- ring 2: 0.9 / 1.1
- ring 3: 0.929 / 1.071

The sum of W is identical (pi). Azimuth-uniform fields, which include all Lambert outgoing exitance, are unbiased.

**Consumers (union of MAP, A, C1 and my greps).**
- WL94: `an94.ex` and the Stage B/C table metrics.
- WL94A: `r94` via table_metrics.
- WL94B: `b94.py:47` stored exitance and the path-label `ex`. These are auxiliary; the G_EXPOSURE gate uses unweighted loss masses.
- WL94C: `q94` table_metrics and face_ext.
- WL94D: `s94.py:129` (G_PHYS), `:157` (CK2 flight-floor diagnostic), `:287-312` (S1 field rel L1 / energy, delivery).
- WL95 design calc: W94 primary, with W_T variants reported.

Not used by:
- WL92-WL93D, which use S.NPHI/S.NZ;
- the WL93D CK2 addendum checker (`experiments/ecro_local93d/addendum_checker/` has no exitance weighting);
- any image E_H, F2, RET, SSR or RENDER_IMP;
- the WL94D teacher loss, or its operator metric (correct zb, `s94.py:179-181,288`);
- WL95A / WL96A, which use W_T.

**Corrected numbers.** The C1 recomputation reproduces the stored JSON to 2.5e-7. The bridge values for the own arms agree independently across A, E1 and C2.

| quantity (CK2 unless stated) | stored (W94) | corrected (W_T) | unweighted |
|---|---|---|---|
| T S1 sender energy vs M64, s8601 / s8602 (WL94D:106-107,146 "over-emits 1.75 / 1.71") | 1.749 / 1.709 | 1.370 / 1.388 | 0.862 / 0.902 |
| D energy s8601 / s8602 | 0.867 / 1.618 | 0.668 / 1.249 | 0.412 / 0.744 |
| P energy s8601 / s8602 | 1.010 / 1.665 | 0.773 / 1.281 | 0.468 / 0.762 |
| R energy s8601 / s8602 | 1.261 / 1.653 | 0.963 / 1.347 | 0.601 / 0.879 |
| own energy s8601 / s8602 (stored only in WL95A bridge JSON) | 0.557 / 1.123 | 0.433 / 0.891 | 0.273 / 0.553 |
| S1 field rel L1, T / D / P s8602 | 1.484 / 1.587 / 1.629 | 1.297 / 1.368 / 1.395 | 1.048 / 1.128 / 1.141 |
| SENDER_IMP ratio T/min(R,P), CK2 s8601 / s8602 (rule <= 0.8) | 1.193 / 0.943 | 1.110 / 0.932 | - |
| SENDER_IMP ratio, val s8601 / s8602 | 0.964 / 0.977 | 0.968 / 0.978 | - |
| val-geometry field metrics (all 10 checkpoints) | - | shift <= 0.015 L1, <= 0.023 energy | - |
| delivery through exact receivers (all masks) | - | identical to 3 decimals (Lambert receivers) | - |
| G_PHYS CK2 diagnostic box A ratio / rel L1 | 1.211 / 0.528 | 1.208 / 0.524 | - |
| WL94/94A/94C/94D box-A MR inputs | - | max amplitude change 0.009; MR flag identical 18/18 | - |
| panel-face (S1 receiver) table corr, ungated | - | changes up to 0.08 | - |

**Reconciliation of C1-01, A-02, E1-F9 and C2-15.** These findings do not contradict each other; they describe different quantities.
- **(a) Gates and labels.** No gate outcome or verdict label of WL94, 94A, 94B, 94C or 94D changes, so C1-01 is correct here. SENDER_IMP stays false on both seeds under either weight, so WL94D NO_RELIABLE_IMPROVEMENT does not depend on the weight. This answers the open question in F-MAP-08.
- **(b) Magnitudes.** On the grazing CK2 S1 sender field the shifts are large: energy 0.12-0.38 and rel L1 0.03-0.23. The largest are T s8601 energy (0.38) and P s8602 L1 (0.23). A-02 and E1-F9 quote only the own arms, and C2-15's "up to about 0.2" understates.
- **(c) Sign relative to 1.**
  - Survives: T over-emission (1.37/1.39); own s8601 and D s8601 under-emission.
  - Changes side: own s8602 (1.12 -> 0.89), R s8601 (1.26 -> 0.96) and P s8601 (1.01 -> 0.77).
  - E1-F9's "unreliable in sign" is therefore right only for those three arms.
- **(d) WL95 did not understate.** WL95:225-226 restricted "<= 0.02 L1, <= 0.04 energy" to the bound arms and said the effect on learned numbers was unmeasured. A-02's "larger than WL95 stated" should read "larger than the bound-arm figure WL95 reported; WL95 left the learned-arm effect unmeasured".
- **(e) Weighting dependence.** The energy ratio depends strongly on the weighting itself: every unweighted value is below 1. Statements about the "energy level" of the S1 sender on CK2 are meaningful only with the protocol cosine weight.

**Narrated conclusions that change:** none in direction. The only side changes (P/R s8601 and own s8602) are not narrated as a conclusion in any worklog. The magnitude wording at WL94D:106-107,146 must be corrected.

## 2. E_H "noise_bound" and decisions taken with it

**What the statistic measures: confirmed.** In `an94.py:182-189` (also `viz_checker93d.py:56` and `analyze92.py`):

noise_bound = [0.5 sum|x_halfA - x_halfB| + 0.5 sum|h_halfA - h_halfB|]/den

- This is the expected L1 of the replicate-mean noise of arm plus reference, i.e. an estimate of E_H at zero true error (an L1 floor).
- It is not the standard error of a paired difference of two E_H values, which share the reference and the camera seeds.
- The C1 paired bound is se(d) <= 6.5e-5 for every image pair.
- Disjoint-half differences reproduce d to about 1e-5. For WL94 SSR s8601: d = 0.016216, halves 0.016031 / 0.016039.

The frozen rule `max(2 x noise_bound, 0.005)` therefore works as a 0.0167 practical-significance threshold that is mislabelled as noise.

**Recomputed decisions** (j6_decisive.json; tolerance = max(2 se, 0.005), and also without the 0.005 floor):

| decision | margin (stored) | frozen tol | label stored | label with paired stat |
|---|---|---|---|---|
| WL94 SSR s8601 (share 0.343 >= 0.30) | d 0.01622 | 0.01673 (fail by 0.0005) | SSR fail -> **H-A INCONCLUSIVE** | SSR pass both seeds -> **H-A SUPPORTED** (frozen mapping `an94.labels`, GATE=RUN_C, FIT false so mr false) |
| WL94 SSR s8602 | d 0.02258 | 0.01680 | pass | pass |
| WL94D RENDER_IMP s8601 (T vs P / R) | -0.0176 / -0.0181 | 0.017 | T worse, False | T worse, False |
| WL94D RENDER_IMP s8602 | -0.0074 / +0.0046 | 0.017 | False | False (T worse than P; T vs R below 0.005 floor; without floor T better than R but still worse than P) |
| WL94D outcome | SENDER_IMP false both seeds | - | NO_RELIABLE_IMPROVEMENT | unchanged |
| WL94C D vs C s8601 / s8602 | +0.040 / -0.022 | 0.017 | better / worse | unchanged -> INCONCLUSIVE unchanged |
| WL94C E vs C s8601 / s8602 | -0.0016 / +0.0010 | 0.017 | tie / tie | tie with 0.005 floor; without floor E worse / better (still mixed) |
| WL94A RET C s8601 | E_H C 0.084 vs A 0.039 | 0.017 | False | False -> INCONCLUSIVE unchanged |
| WL94A SSR (not in outcome) A s8601 | d 0.0116, share **0.298** | - | fail | fail on share by 0.002 (rule mechanics, not noise) |
| WL95A NEXT_DESIGN_TARGET (C2-07) | s8601 dS 0.0130 / dG 0.0133 / dF 0.0034; s8602 dS 0.0354 | 0.0167 | UNDETERMINED | UNDETERMINED (seeds disagree: s8602 LEARNED_SENDER-like, s8601 dS ~ dG) |
| WL92A HELDOUT_SHARP | margin 0.0101 both seeds | - | pass | pass |

WL93D G3 cells: C1 reports no flip (two-cell rule, 0.005 floor); I did not re-verify this one.

**Only one label differs: WL94 H-A changes from INCONCLUSIVE to SUPPORTED.** Caveats on that change:
1. The stored label is conformant with the frozen rule text. The SUPPORTED reading is post hoc and must be labelled as such.
2. C2_srcS1 substitutes the binned M64 sender (`mat94.py:249`), which itself over-delivers on CK2 (section 4). H-A SUPPORTED therefore means: "replacing the learned S1 sender by the exact 64-bin sender removes >= 30 % of the untextured sharp-ROI error on both seeds, measured against the path-traced reference". It does not mean that learning can achieve a physically correct sender.
3. WL94's narrated conclusion already said this (WL94:253-255, "34 % (s8601; at the noise limit) and 54 %"). Only the qualifier "at the noise limit" is wrong.

C1-02's secondary claim, "values near the floor cannot be ranked by E_H", is modified. Paired differences near the floor are statistically resolvable. However, L1 compresses true differences toward zero near the floor, so these differences cannot be read as ratios of true error.

## 3. C1-04 WL94D "CK2 delivery to box A"

Confirmed: `s94.py:309` `bx[1130:1238]` covers boxes A and B (108 patches), and delivery_ratio uses all non-panel receivers. WL95 section 2.2 item 2 already disclosed this.

Box-A-only values (independent of W):

| seed | T rel L1 (ratio) | P rel L1 (ratio) |
|---|---|---|
| s8601 | 1.036 (1.76) | 0.614 (0.82) |
| s8602 | 1.221 (2.07) | 1.213 (1.87) |

WL94D's direction survives: T over-delivers to box A, and on s8602 T ~ P. The original wording understates it. No gate uses these numbers, so the finding is immaterial.

## 4. Unmatched norms and M64 as truth (C1-05, C1-06, F-MAP-07, F0-13)

**C1-05.** The matched-norm values reproduce (C1, WL95:230):

| operator | flux L1 | Frobenius | other |
|---|---|---|---|
| zero | 1.000 | 1.000 | - |
| learned | 0.96-1.04 | 0.77-0.83 | mass 0.82-1.06 |
| SVD16 | 0.891 | 0.46-0.52 | - |
| NMF16 | 0.606 | 0.49-0.55 | - |

WL94D:147 already flagged "(a different norm)", and WL95 section 2.2 item 3 recorded the matched references.

The remaining defect is wording. "Zero-operator level" at WL95:244, 491 and 599 is true in flux-L1 only; in Frobenius and mass the learned operator is clearly non-zero. The conclusion survives in every matched norm: the learned operator is far from an admissible rank-16 operator and is therefore not at a rank-16 representation limit.

**C1-06.** The facts check out against `results/ecro_s94d/audit.json`. G_PHYS was revised before the freeze, after a first-draft rule (corr >= 0.9) failed at 0.641. WL94D:46-52 disclosed this, together with the CK2 diagnostic (WL94D:60: box A 1.21 / 0.72 / 0.53 / noise 0.06).

My addition (j6_decisive.json) quantifies how weak the revised rule is:
- The rule corr(X, label256) >= corr(label256, label64) passes any deterministic predictor whose correlation with the noise-free label exceeds 0.62.
- The M64 flight's implied noise-free correlation is 0.76 from the 256-spp label and 0.72 from the 64-spp label, which are consistent.
- G_PHYS therefore certifies only moderate agreement plus a pooled energy of 0.95 on the 96 training items.

On CK2 the M64 flight over-delivers: 1.21x to box A and 1.37x to all receivers, with panel-face corr -0.32 (half-noise 0.06-0.13).

Rel L1 against the physical label on CK2 box A (single flight, exact receiver):

| sender | rel L1 | ratio |
|---|---|---|
| M64 | 0.524 | 1.21 |
| own s8601 | 0.588 | 0.50 |
| D s8601 | 0.590 | - |
| P s8601 | 0.646 | - |
| R s8601 | 0.761 | - |
| **T s8601** | **1.182** | **2.13** |
| all s8602 arms | 0.90-1.64 | - |

Hence WL94D NO_RELIABLE_IMPROVEMENT is valid as measured, but it tests supervision toward M64, not physically correct sender supervision. Its narrated finding that T makes s8601 worse also holds against the physical label.

**F-MAP-07.** Modified on three points:
- **WL92B "X_Q closes 96 % of the seed gap"** uses the exact path-traced once-scattered table q (`cf92b.py:194` `q = D['q']`; WL92B:27,110), not M64. It is a rendered E_H against the path-traced floor, so it does not depend on M64.
- **WL93C ORACLE_FEASIBLE and WL92B X_M\*** are rendered E_H against path-traced references, so they are valid as render-level counterfactuals.
- **Operator-level "error against M64"** (WL93D:38,52, "physical control is M64") is valid as distance to the exact 64-bin projection of the frozen-ER BRDF, which is the class the learned local operator represents. It is invalid only where it is narrated as physical delivery truth: WL93D's label "physical control", the WL94D teacher and G_PHYS "positive physical control", and WL94's "exact sender".

**F0-13.**
- Items 1, 2 and 5 are upheld, with the corrections above. Item 5's "unmeasured effect" is now measured: no gate flips.
- Items 3 and 4 are outside this cluster and are not adjudicated here. Item 3 is WL87 "explicit >= ECRO" feeding the WL96 cost argument; item 4 is the WL95A delivery bias used as a WL96 premise.

## 5. C1-08: stops and labels decided inside noise, by one seed or by rule mechanics

| batch / decision | decisive margin | noise / context | reading |
|---|---|---|---|
| WL92 G_rep STOP | best owned R16 0.084 / 0.073 vs 0.06; per-geometry rank-16 fit 0.036 / 0.017 excluded as `OWN_k16_converged: false` | an unconverged fit is still an existence upper bound below 0.06 | stop by convergence-flag mechanics; not evidence of a representation limit |
| WL92A not admitted | HELDOUT_SHARP passes both seeds (margin 0.010); E4 admission clause failed (Q4 SNR 3.9 vs 4, per stored admission / C1) | NO_REGRESSION fails on one seed: own_s8602 E1_T1_LSPOT 0.0360 vs 0.0216 (real, large) | held-out result positive but unadmitted; the regression is real, single-seed |
| WL93 preflight STOP | 0.508 vs bound 0.485 (margin 0.023) | implied single-q noise 0.145; vs mean of two q 0.496 (`posthoc_ident.json`) | margin inside target noise; no model evidence |
| WL93A H_TARGET "NOT SUPPORTED" | detectable effect 0.12 / 0.21 (2 se) > predeclared 0.099 (C1; not re-verified by J6) | underpowered | should read "not detected" |
| WL94 H-A INCONCLUSIVE | d 0.01622 vs 0.01673 | se(d) 6.5e-5 | decided by mis-scaled margin (section 2) |
| WL94A outcome INCONCLUSIVE | seed split (RET fails s8601 by 0.045) | real | conformant, real seed dependence |
| WL94B STOP_INVALID_INTERVENTION | G_EXPOSURE 1.2316 / 1.2361 vs 1.25 | no noise term in the gate; ~1.5 % below threshold | rule mechanics; candidate never trained, no model evidence |
| WL95A NDT UNDETERMINED | see section 2 | stated reason (noise) wrong; actual reason seed disagreement | label preserved |
| WL86 F2 replay (WL92) | 0.003-0.012 over 0.06; seed spread 0.008 (C1) | within seed spread | weak |

Robust decisions (margin well above both noise and seed spread):
- WL94 MR_frozen
- WL94D SENDER_IMP
- WL94D RENDER_IMP
- WL94C D-vs-C seed split
- WL92A HELDOUT_SHARP

## 6. C1-10 confirmations

My spot checks agree. The WL94 SSR values and the WL94D gate values read from stored JSON equal C1's recomputed values, and the own-arm W94/W_T bridge values agree across C1, A, E1 and C2. VERIFIED.

## 7. Rulings (summary)

| finding | ruling | classification |
|---|---|---|
| F-MAP-08 | UPHELD_WITH_MODIFICATION (impact now measured: no label change) | IMPLEMENTATION_DEFECT |
| A-02 | UPHELD_WITH_MODIFICATION (WL95 did not understate; largest learned-arm shifts are T/D/P/R, not own) | IMPLEMENTATION_DEFECT |
| C1-01 | UPHELD_WITH_MODIFICATION ("bounded" true for gates/tables, not for CK2 S1 field magnitudes) | IMPLEMENTATION_DEFECT |
| C2-15 | UPHELD_WITH_MODIFICATION (up to 0.38 energy, not 0.2) | IMPLEMENTATION_DEFECT |
| E1-F9 | UPHELD_WITH_MODIFICATION (sign unreliable only for arms near 1) | IMPLEMENTATION_DEFECT |
| C1-02 | UPHELD_WITH_MODIFICATION (WL94 H-A is the only label affected; post hoc; M64-substitution scope) | CONDITIONALLY_VALID (stored labels), evaluation defect in the statistic |
| C2-07 | UPHELD | CONDITIONALLY_VALID |
| C1-04 | UPHELD (immaterial; disclosed in WL95) | CONDITIONALLY_VALID |
| C1-05 | UPHELD_WITH_MODIFICATION (WL94D/WL95 partly disclosed; residual = "zero-operator level" wording) | INVALID_COMPARISON (wording only) |
| C1-06 | UPHELD_WITH_MODIFICATION (disclosed; weak rule quantified) | CONDITIONALLY_VALID |
| F-MAP-07 | UPHELD_WITH_MODIFICATION (WL92B X_Q is path-traced, not M64) | CONDITIONALLY_VALID |
| F0-13 | UPHELD_WITH_MODIFICATION for items 1, 2, 5; items 3-4 not adjudicated | CONDITIONALLY_VALID |
| C1-08 | UPHELD_WITH_MODIFICATION (margins above) | CONDITIONALLY_VALID |
| C1-10 | UPHELD | VERIFIED |

## 8. P0 verdicts decided by this cluster

**P0-1, first half ("learned S1 sender has large avoidable error"): CONDITIONALLY_VALID.**
- Large error vs M64 holds under the corrected weight and in every matched norm:
  - CK2 field rel L1 1.04 / 1.12 for the own arms and 1.30-1.40 for T/D/P/R;
  - operator Frobenius 0.77-0.83 vs rank-16 0.46-0.55.
- "Avoidable" holds relative to the exact 64-bin operator class.
- Against the physical label at CK2 box-A delivery, own s8601 (0.588) is only 0.06 worse than M64 (0.524), with the opposite energy sign. Own s8602 (0.901) is clearly worse.
- Restrictions: CK2 grazing configuration; two seeds of one recipe; single flight with exact receiver; energy-direction statements only with W_T.
- The passivity half is not decided here.

**P0-2, first half ("M64 introduces systematic sharp-delivery bias"): CONDITIONALLY_VALID.**
- On CK2, single-flight M64 over-delivers 1.21x to box A (corr 0.72, rel L1 0.52 vs half-noise 0.06) and 1.37x to all receivers, with panel-face corr -0.32. This does not depend on the weight.
- On the 96 training items the pooled energy is 0.95 (implied true corr 0.72-0.76).
- "Systematic" is therefore restricted to grazing, CK2-like incidence.
- The relation-sampling half is not decided here.

**P0-5: CONDITIONALLY_VALID** for "justify rejecting the tested mechanisms", restricted to "the geometry-conditioned rank-16 tied S1 operator as trained at the tested budgets". Within this cluster none of the labels is a rejection:
- WL92, WL93 and WL94B are stops inside noise or by rule mechanics, with no model evidence.
- WL94 H-A becomes SUPPORTED under a paired statistic (post hoc).
- WL94D tests supervision toward a CK2-biased M64 teacher, not physically correct sender supervision.

"No general impossibility": VERIFIED.

## 9. Affected worklog statements

Each item reads: original claim -> defect or limit -> corrected interpretation -> dependents -> action.

1. **WL94D:106-107,146 "T over-emits 1.75 / 1.71".**
   - Defect: an94.W.
   - Corrected: 1.37 / 1.39, so T still over-emits. The D/P table energies change from 0.87/1.62 and 1.01/1.67 to 0.67/1.25 and 0.77/1.28.
   - Dependents: none (SENDER_IMP unaffected).
   - Action: QUALIFY.
2. **WL94D:110-112 "CK2 delivery to box A: T 1.05 vs P 0.62 (ratio 1.28 vs 0.70); s8602 1.24 vs 1.23".**
   - Defect: A+B mask and all-receiver ratio.
   - Corrected (box A only): s8601 1.04 vs 0.61 (ratio 1.76 vs 0.82); s8602 1.22 vs 1.21 (2.07 vs 1.87).
   - Dependents: none.
   - Action: QUALIFY.
3. **WL94D:178 "training-field error 0.82 -> 0.72".**
   - Defect: W94-weighted.
   - Corrected value not recomputed; val shifts are <= 0.015.
   - Dependents: none.
   - Action: QUALIFY (magnitude unverified, immaterial).
4. **WL94D:46-52 G_PHYS pass as "positive physical control of the teacher quantity".**
   - Defect: the revised rule is weak (it passes any predictor with true corr > 0.62), and the M64 flight is biased on CK2.
   - Corrected: NO_RELIABLE_IMPROVEMENT is scoped to M64 supervision.
   - Dependents: WL95 / WL95A reliance on WL94D as "sender supervision does not help".
   - Action: QUALIFY.
5. **WL94D outcome NO_RELIABLE_IMPROVEMENT.**
   - Limit: none found; robust to W and to the paired statistic.
   - Action: PRESERVE, with the scope from item 4.
6. **WL94D:147 / WL95:244,491,599 "operator at the zero-operator level relative to M64 / far above the rank-16 floor (different norm)".**
   - Defect: unmatched or L1-only comparison.
   - Corrected (matched norms): L1 ~1.0 vs 0.61-0.89; Frobenius 0.77-0.83 vs 0.46-0.55.
   - Dependents: the WL95 "evidence-favoured constraint", which survives.
   - Action: QUALIFY.
7. **WL94 H-A INCONCLUSIVE (results/ecro_mat94/final.json); WL94:253 / :340 "s8601 at the noise limit / 0.0005 short of the noise rule".**
   - Defect: noise_bound is an L1 floor; se(d) is 6.5e-5.
   - Corrected: under a paired statistic, H-A = SUPPORTED by the frozen mapping (post hoc), scoped to M64 substitution.
   - Dependents: WL94's narrated item 4, and the WL94A-94D premise that the residual sits in panel-delivered input, are strengthened.
   - Action: QUALIFY (stored label kept as rule-conformant; add the post hoc paired reading).
8. **WL95:225-226 "bias <= 0.02 L1, <= 0.04 energy on the bound arms; learned effect unmeasured".**
   - Limit: correct as scoped. The learned effect is now measured: energy 0.12-0.38 and L1 0.03-0.23 on CK2.
   - Action: PRESERVE (append the measured value).
9. **WL95A NEXT_DESIGN_TARGET UNDETERMINED, with "within noise" reasoning.**
   - Defect: the tolerance is 60-80x the paired noise.
   - Corrected: the label is unchanged; the actual reason is seed disagreement.
   - Action: QUALIFY.
10. **WL93D:38,52 "physical control is M64 / E_phys".**
    - Defect: M64 is the exact 64-bin operator, not a physical delivery oracle on CK2.
    - Dependents: the WL94D teacher choice.
    - Action: QUALIFY (rename; operator-level results preserved).
11. **WL92 STOP_INTERVENTION_G_rep.**
    - Defect: decided by the convergence flag; the per-geometry rank-16 fit reached 0.036 / 0.017, below 0.06.
    - Corrected: must not be read as a representation failure.
    - Action: QUALIFY (label PRESERVE).
12. **WL93 preflight STOP (0.508 vs 0.485).**
    - Limit: the margin is inside target noise, and there is no model evidence.
    - Action: PRESERVE the label, QUALIFY the interpretation.
13. **WL94B STOP_INVALID_INTERVENTION (1.2316 / 1.2361 vs 1.25).**
    - Limit: rule mechanics with no noise term; no model evidence.
    - Action: PRESERVE.
14. **WL93A H_TARGET NOT SUPPORTED.**
    - Limit: underpowered (from C1, not re-verified).
    - Corrected: "not detected".
    - Action: QUALIFY.
15. **WL94 / 94A / 94C table and MR results.**
    - Limit: the W change is <= 0.009 and the flags are unchanged.
    - Action: PRESERVE.
16. **WL92B "X_Q closes 96 % of the seed gap".**
    - Limit: none; it uses path-traced q, not M64.
    - Action: PRESERVE.

## 10. Unresolved

- WL94D training-field error 0.82 -> 0.72 under W_T was not recomputed. This needs the training-item sender fields; it is immaterial to any label.
- The WL94D G_PHYS training-item pooled ratio and corr under W_T were not recomputed. The receivers are mostly Lambert and the margins are 0.15 / 0.12, so the risk is low.
- The WL93D G3 and WL93A power figures were taken from C1 without independent re-verification.
- F0-13 items 3-4 belong to the cost / WL96 tracks.

Status: COMPLETE.
