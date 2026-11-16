# WL97 audit - Track E4: independent reproduction of the ACCURACY half of P0-4

Auditor: independent track E4 (resumed after an interrupted first attempt of the same track; all numbers below were re-derived or
spot-checked in this run, see section 8). Date 2026-11-16. Repository HEAD efcf86a, audited code at freeze commit a1718f7
(`~/rest_x96a/code`, read-only). Nothing outside `docs/audits/wl97_evidence_reliability/`, `experiments/audit97/E4/`,
`results/audit97/E4/` and `~/rest_x97/E4/` was created or modified. Nothing was trained.

Claim under audit (P0-4, accuracy half): *"The learned broad higher-order resolvent has no meaningful accuracy ... headroom over explicit
recurrence on the tested CK2 workload."* Source: WL96A (`docs/worklogs/96A_WORKLOG_LEARNED_HIGHER_ORDER_UTILITY_CEILING.md`, sections B1-B3,
C, D; `experiments/ecro_util96a/w96a.py`; `results/ecro_util96a/{s1.json,tails.json,tails.npz,T96A.npz}`; server `~/rest_x96a/results`).

## 1. Verdict in one paragraph

**CONDITIONALLY_VALID.** Every decisive number of WL96A's Stage 1 was reproduced with independent code (image level to 5 digits, field
level to 3-4 digits), the event accounting was confirmed by experiment, the explicit tail was re-derived with an own gather-scatter and
own receiver weights, and the decision (ROI change of the explicit broad-last tail far below M_img = 0.005) is robust to camera seeds,
to new independent field references, to the mask, to the statistic, to the noise subtraction, to pooled vs deployed state and to prefix
noise. Against a new event-split path-traced camera reference, the explicit tail readout is as accurate as the "ceiling" table itself
(L1 0.00535 vs 0.00532 of the ROI total). The restrictions are important and are part of the verdict:

1. the "ceiling" is a 1270 x 64 patch x bin TABLE; it is not the path-traced image. The tail table readout itself deviates from the
   path-traced tail image by a structured 0.0036-0.0053 of the ROI total (about M_img, 6-8x the explicit-vs-ceiling gap). The result
   therefore bounds table-output operators only; WL96A's sentence "no representation can buy anything on this workload" exceeds what
   was tested (UNTESTED_INFERENCE for non-table outputs);
2. the explicit recurrence is fed the path-traced (oracle, bin-averaged) 2-event prefix; nothing is said about a deployable prefix;
3. "broad-last tail e3_o" is 76 % (box A) / 83 % (all receivers) Lambert-last-sender flux; at most 1.9 % of it passes a binned panel
   sender. Its glossy-last sub-families do show real explicit errors (floor-last about 5 %, sphere-last about 25 % of the sub-family);
4. the statement is an image-ROI statement for one camera, one light, one geometry, 3-5 scattering events. Receivers the camera does not
   read have real field-level explicit errors (panel "sharp readers": 8 % deployed / 4 % pooled debiased table L2; sphere receivers:
   -4.6 % / -2.3 % flux, not reported in WL96A; sphere and box B cover 0 camera pixels);
5. the learned OWN arms (G8601 / G8602) are descriptive, not like-for-like, and play no role in the decision (the decision rests on the
   oracle-output ceiling, which is conservative in favour of the learned side).

## 2. Scope and method

Trace followed: protocol (`experiments/ecro_util96a/protocol/ecro_util96a.json`) -> code (`w96a.py`, `ecro_mat94/{mat94,an94}.py`,
`ecro86/{pt86,scene86}.py`, `ecro86a/pt86a.py`, `gm1_explicit87/{xport87,quad87,fx87}.py`) -> input artifacts (`CKF_none.npz`,
`masks94.npz`, `T96A.npz`, `tails.npz`, renders) -> raw numbers -> decision rule -> inference.
Own code only for all metrics (no import of `w96a.py`, `a95.py`, `an94.py`). `xport87.XOp` / `quad87.Quad` (the comparator under test)
were used to produce the explicit tail and were cross-checked with an own numpy float64 gather-scatter and own Monte-Carlo ER receiver
weights. Scene / closure / ray primitives of `pt86` were used (Ctx, patch_point, dir_in_bin, DrClosure83, Ctx.render).

Definitions verified in code (`pt86.py:217-271`): 'cam' render: e1 = NEE at camera vertex 1, e2 = vertex 2, h = vertices 3..6 (D = 6);
'ecro' render: kmax 2 plus table lookup `hl = thr * B[patch, bin]` at camera vertex 2. A table holding 2..5 scattering events leaving
vertex 2 therefore corresponds exactly to h (3..6 camera events). Denominators: all image numbers are relative to the sum of the stored
reference total (e1 + e2 + h, mean of 4 reps) inside the mask, as in WL96A.

New audit data (server LabServer63, about 9 GPU-min in total, never more than one own GPU process):
- 4 tagged field halves, own tracer `e4_field.py`, seeds 97401-97404, 4,096 spp each, kmax 6, tags = (class of the last sender s1) x
  (panel at any vertex beyond s1) per order k = 2..6 (25 s each);
- production field tracer with kmax = 2, 3, 5 at the stored half-a seed 94301 (`e4_kmax.py`);
- explicit tails from 9 prefixes, deployed state 86001 and the 16 pooled realizations (`e4_tail.py`);
- 65 table-lookup re-renders (`e4_render.py`): identity render seed 86600; camera seeds 97600-97603 x 16 tables (4,096 spp);
- 8 event-split camera references (`e4_ref.py`): seeds 97700-97703, kmax 3 and kmax 6 with identical random streams, 4,096 spp.

## 3. Original vs reproduced numbers

Image level (ROI_T unless stated; "stored" = WL96A renders analysed with own code; "new" = audit re-renders / audit fields):

| quantity | WL96A | audit, stored renders | audit, new seeds / independent data |
|---|---|---|---|
| Z (tail dropped) dI | 0.178 | 0.17789 | 0.17789 (new camera seeds) |
| higher-order share h of ROI total | 0.32 | 0.3207 | 0.3206 (new REF k6) |
| dI(XOD) | 0.00065 | 0.00065 | 0.00065 (new camera seeds, each of 4 reps 0.00065) |
| dI(XOD), prefix AND reference from audit fields | - | - | 0.00058 |
| dI(XOD), cross-independent (prefix ckf / ref audit; prefix audit / ref ckf) | - | - | 0.00054 / 0.00066 |
| dI(XOD) vs 6-half reference | - | - | 0.00050 (signed +0.00035) |
| signed(XOD) | +0.0006 | +0.00057 | +0.00057 |
| paired noise nb | 0.00002 | 0.00002 | 0.00002 |
| n_img = 0.5 dI(Fa, Fb) | 0.00146 | 0.00146 | 0.00146 |
| matched noise n_img_o (e3_o halves only) | not computed | - | 0.00071 (ckf 2 halves), 0.00046 (audit 4 halves) |
| H = dI - n_img ; U = H + 2 nb | -0.00082 ; -0.00077 | -0.00082 ; -0.00077 | matched: -0.00006 ; -0.00002 |
| dI + 2 nb (no subtraction) | 0.0007 | 0.00069 | 0.00069 |
| XOP (pooled) dI | 0.0006 | 0.00060 | 0.00060 |
| XD / XP dI | 0.0029 / 0.0031 | 0.00290 / 0.00308 | XD 0.00290 |
| XSD / XSP (explicit e3_p) dI | 0.0027 / 0.0031 | 0.00272 / 0.00312 | - |
| G8601 / G8602 dI (signed) | 0.025 / 0.013 (-0.019 / -0.009) | 0.02546 / 0.01262 (-0.01887 / -0.00904) | - |
| E_H(F) vs REF h (noise bound) | 0.0118 (0.0082) | 0.01183 (0.00815) | 0.0140 vs new REF (null 0.0091) |
| R = E_H(XOD) - E_H(F) | 0.00015 | 0.00014 | vs new event-split REF, tail only: +0.00003 |
| DIFF dI(XOD) | 0.0011 | 0.00107 | 0.00106 |
| outcome | S1_NO_HEADROOM | S1_NO_HEADROOM | all variants below M_img |

Field level (W-weighted patch x bin x channel rel L1 with own weights; n0 = half-split noise of the reference mean):

| arm / family / region | WL96A flux ; table (n0) | audit vs CKF_none | audit vs own 4 new halves | vs all 6 halves |
|---|---|---|---|---|
| XD e3_o box A | 0.993 ; 0.036 (0.032) | 0.9934 ; 0.0357 (0.0315) | 0.9930 ; 0.0293 (0.0242) | 0.9931 +- 0.0012 ; 0.0266 (0.0217) |
| XD e3_o box A exitance | 0.015 (0.009) | 0.0150 (0.0091) | 0.0136 (0.0058) | 0.0129 (0.0059) |
| XP e3_o box A | 0.999 ; 0.035 (0.032) | 0.9986 ; 0.0348 (0.0315) | - | 0.9984 ; 0.0245 (0.0217) |
| XD e3_o panel face | 1.004 ; 0.060 (0.055) | 1.0039 ; 0.0598 (0.0547) | 1.009 ; 0.043 (0.033) | 1.007 ; 0.042 (0.030) |
| XD e3_o S1 (sharp readers) | 1.002 ; 0.090 (0.047) | 1.0015 ; 0.0903 (0.0471) | 1.0018 ; 0.0840 (0.0359) | 1.0017 ; 0.0796 (0.0309) |
| XP e3_o S1 | 1.011 ; 0.064 (0.047) | 1.0111 ; 0.0635 (0.0471) | 1.0113 ; 0.0521 (0.0359) | 1.0112 ; 0.0476 (0.0309) |
| XD e3_o all | 0.996 ; 0.032 (0.025) | 0.9957 ; 0.0323 (0.0249) | 0.9954 ; 0.0279 (0.0187) | 0.9955 ; 0.0259 (0.0158) |
| XD e3_p box A | 1.008 ; 0.049 (0.035) | 1.0084 ; 0.0487 (0.0352) | 1.0057 ; 0.0455 (0.0280) | 1.0066 ; 0.0410 (0.0242) |
| XD / XP e3_p all | 0.235 / 0.123 (0.100) | 0.2354 / 0.1234 (0.1001) | 0.2215 (0.0686) / - | 0.2183 / 0.0922 (0.0598) |
| cross-half box A (explicit a vs tail b ; tail a vs tail b) | 0.044 ; 0.063 | 0.0444 ; 0.0630 | - | - |
| box A exitance e2 / e3_p / e3_o | 151.3 / 25.7 / 133.3 | 151.31 / 25.68 / 133.33 | 151.42 / 25.75 / 133.38 (own tracer) | - |

No numerical disagreement with WL96A was found in any reproduced quantity.

## 4. Findings

Each finding: id, category, material, claim affected, numbers, evidence, decision-level impact, proposed verdict, confidence.

### E4-F1 (confirmation, not material as a defect) - event accounting of CKF_none is as claimed
- Claim: e2 = exactly 2 scattering events, e3 = 3..5 events, `_p` = the vertex adjacent to the receiver (s1) is the panel.
- Experiment A (production tracer, `e4_kmax.py`, seed 94301, 4,096 spp): kmax 2 reproduces stored e2_p / e2_o half a to rel L1
  1.1e-8 / 1.7e-8 and gives e3 = 0 exactly; kmax 5 reproduces e3_p / e3_o to 8e-9 / 1.6e-8; kmax 3 gives 57 % / 56 % of the stored e3
  sums (874.7 of 1531.6; 44,846.8 of 79,753.2). So e3 is not "3 only" and nothing beyond 5 events is inside it.
- Experiment B (own tagged tracer, new seeds, 4 x 4,096 spp, independent code and seeding rule): flux ratio audit / CKF: e2_o 0.99989,
  e3_o 1.00026 (all receivers), e3_o box A 1.0004, e2_p box A 1.008, e3_p box A 1.003; table rel L1 audit-vs-CKF equals the null
  expectation from the two half-split noises (e3_o all 0.0331 vs 0.0311; box A 0.0450 vs 0.0397; e3_p all 0.126 vs 0.121).
  Alternative definitions are rejected: "e3 = k3..4" flux 0.851, "k3..6" 1.078, "k3 only" 0.562; "e3_p = panel at any earlier vertex"
  1.52, "panel anywhere" 2.42 (claimed definition 0.994).
- One borderline value: e3_p all-receiver flux CKF 76.25 / 76.36 vs audit 76.06 / 75.72 / 75.83 / 75.92 (-0.55 %, 1.97 pooled-SD units,
  t(4) p about 0.12; the z = -4.6 in `fieldan.json` uses a 2-half SE and is an artefact). Not significant; no decision impact.
- Evidence: `results/audit97/E4/kmax.json`, `fieldan.json` (event_accounting, alternatives), `extra.json` (per_half_flux);
  `experiments/ecro86/pt86.py:217-271`, `experiments/ecro86a/pt86a.py` (field_src split).
- Verdict VERIFIED. Confidence high.

### E4-F2 (confirmation) - explicit tail: reproduction, own sweep, non-overlap, no double counting
- Stored `tails.npz` dep_o / dep_p reproduced to rel L1 2.1e-7 / 8.4e-8 from CKF_none with `xport87.XOp` / `quad87.Quad` at the
  protocol seeds; pooled mean to 5.9e-8; the audit tables F / XOD / XOP / Z equal `T96A.npz` to 0 / 9.6e-8 / 2.7e-8 / 0.
- Own numpy float64 gather-scatter with the comparator's weights equals `XOp.apply` to 1.8e-7 (indexing, miss handling, scatter).
  Own Monte-Carlo ER receiver weights (independent quadrature of the frozen closure; floor / panel) reproduce the comparator's
  weights to sum ratio 0.9999 / 1.0019 and the third-order table to rel L1 0.05 % overall (floor receivers 0.19 %, panel receivers
  0.65 %, Lambert receivers 2e-7; analytic Lambert weights identical).
- Non-overlap: the class split sums to the unsplit operator (9e-8). The prefix holds exactly 2 events (F1), each sweep adds exactly one
  event; per-order flux of the explicit e3_o part against the tagged reference: k3 0.9963, k4 0.9951, k5 0.9942 (all receivers), box A
  0.9948 / 0.9931 / 0.9902. A double-counted prefix would shift a whole order by a factor, not by 0.4-1 %. Reference order ratios
  k4/k3 0.513, k5/k4 0.518; explicit 0.511, 0.518.
- Mild systematic drift of about -0.1 % of flux per order (no visible compounding of table error: rel L1 by order at box A 0.040 /
  0.029 / 0.036 against n0 0.033 / 0.026 / 0.035).
- Evidence: `results/audit97/E4/tails97.json`, `fieldan.json` (field_by_class_and_order).
- Verdict VERIFIED. Confidence high (the ER closure itself is a shared primitive and was not re-derived).

### E4-F3 (confirmation) - image-level decision reproduced and not seed-specific
- Own statistics on the stored renders reproduce every row of WL96A table B2 (section 3). Identity render of the audit F table with
  the production seed equals the stored file to 1.7e-7 of the image mean.
- New camera seeds (97600-97603): dI(XOD) 0.00065 in each of the 4 single frames; n_img 0.00146; the new frames are different noise
  realizations (F new vs stored rel L1 0.0072, like stored r0 vs r1 0.0074). Camera sampling is NOT the relevant variance (nb 2e-5);
  the relevant variances are the field reference and the state realization.
- New field data: with prefix and reference both from the audit's 4 new halves dI = 0.00058; cross-independent 0.00054 and 0.00066;
  against the 6-half reference 0.00050. Pooled state 0.00060.
- Evidence: `results/audit97/E4/img.json` (A_decision_recomputed, B_audit_renders.decision_new_seeds, independent_reference),
  `extra.json` (seed_sanity).
- Verdict VERIFIED. Confidence high.

### E4-F4 (statistical_weakness, NOT material) - noise subtraction is mismatched and L1 norms do not subtract
- WL96A: H = dI(XOD) - n_img = -0.00082 with n_img = 0.5 dI(Fa, Fb) = 0.00146. Fa and Fb differ in the e3_p halves too (their image
  noise alone is 0.0011 in ROI_T), but XOD and F share the same e3_p mean, so n_img overstates the noise in XOD - F by 2x. Matched
  noise (e3_o halves only, re-rendered): 0.00071 -> H_matched = -0.00006. The same applies to "table excess = rel L1 - n0" at field
  level: an L1 norm of signal plus noise is not signal + noise.
- Proper estimates: debiased relative L2 from two independent references gives a real explicit e3_o readout error of 0.00058 (rms,
  relative to the rms of the ROI total); the signed ROI change is +0.00057 (CKF reference) / +0.00035 (6-half reference) with a paired
  SE of about 2e-5 plus reference noise 0.0003. Field level, box A: debiased table rel L2 0.6-1.9 %, exitance rel L2 1.0-1.3 %, flux
  0.9931 +- 0.0012 (a real -0.7 % deficit). So the explicit broad tail is NOT exactly at the ceiling; its real error is about 1 % of
  the e3_o component (0.0004-0.0006 of the ROI total), roughly 10x below M_img.
- Decision impact: none (every variant stays below M_img; WL96A itself states the conclusion without subtraction). The phrase "at the
  reference noise level" should be read as "indistinguishable by L1 at this reference precision; real residual about 1 % of e3_o".
- Evidence: `img.json` (n_img_o, independent_reference), `fieldan.json` (table_rel_l2_debiased, flux_ratio_se).
- Verdict CONDITIONALLY_VALID (statistic), confidence high.

### E4-F5 (dataset_limitation, MATERIAL for scope) - what "broad-last tail e3_o" actually contains
Measured with the tagged tracer (shares of W-weighted e3_o flux, 4 x 4,096 spp, SE of the panel share 0.2 points at box A):

| receivers | last sender Lambert | floor | sphere | panel at ANY vertex beyond s1 | of which only possible through a binned panel sender (k = 4, 5) | order share k3 / k4 / k5 |
|---|---|---|---|---|---|---|
| box A | 76.2 % | 23.1 % | 0.6 % | 3.8 % | <= 1.9 % | 57 / 28 / 15 % |
| panel-facing face | 79.6 % | 20.4 % | 0 | 2.6 % | <= 2.1 % | 55 / 29 / 15 % |
| all receivers | 83.0 % | 15.5 % | 1.5 % | 2.7 % | <= 1.9 % | 56 / 29 / 15 % |
| S1 (panel) receivers | 83.4 % | 15.7 % | 0.8 % | 28.0 % | <= 18.0 % | 53 / 32 / 15 % |

(k = 3 with the panel at s2 is inside the path-traced prefix and exact; for k = 4, 5 the tag cannot separate "panel is the vertex whose
e2 table is the prefix or a later sender" from "panel only at the light-side NEE vertex", so the k = 4, 5 column is an upper bound.)

Explicit vs tagged reference by class of the last sender (deployed, prefix = all 6 halves; exitance rel L1 (n0); flux ratio):

| last sender | box A | all receivers |
|---|---|---|
| Lambert | 0.014 (0.006) ; 0.993 | 0.019 (0.004) ; 0.998 |
| floor (moderately glossy ER) | 0.024 (0.008) ; 0.995 | 0.050 (0.008) ; 0.985 ; debiased table L2 0.065 |
| sphere (ER) | 0.112 (0.013) ; 1.008 | 0.239 (0.014) ; 0.969 ; debiased table L2 0.28 |
| panel (= e3_p, for comparison) | 0.032 (0.007) ; 1.006 | 0.202 (0.013) ; 1.003 ; debiased table L2 0.098 (pooled 0.031) |

- Interpretation (measurement): the aggregate agreement of e3_o is carried by Lambert-last transport. The glossy-last sub-families have
  real explicit errors (floor-last about 5 %, sphere-last about 24 % of the sub-family over all receivers), diluted to <= 1 % of e3_o by
  their flux shares. Binned sharp-panel senders are essentially absent from the e3_o test (<= 1.9 % of flux outside the panel's own
  receivers). WL96A states the untagged-predicate caveat qualitatively; these are the missing magnitudes.
- Decision impact: none for the CK2 outcome. It restricts what the outcome can be cited for: it is NOT evidence that explicit 64-bin
  recurrence is accurate for higher-order transport through glossy or sharp last senders, and a workload whose tail is dominated by
  such paths was not tested.
- Evidence: `fieldan.json` (e3_o_composition, field_by_class_and_order); spot check of the box A panel share from the raw TF files
  0.0376.
- Verdict CONDITIONALLY_VALID. Confidence high for the shares, medium for the per-class errors of the small sphere class.

### E4-F6 (evaluation_defect + invalid_inference, MATERIAL for interpretation) - the floor table F is not the path-traced image
- Check requested: does the table readout F reproduce the path-traced REF h within noise? **No.** ROI_T, 8 F frames vs 4 stored REF
  reps: E_H 0.0118 against a null expectation of 0.0069 (WL96A's own "noise bound" 0.0082 is also exceeded and this is not remarked);
  signed +0.0026 +- 0.00005 (F / h = 1.008); debiased rel L2 0.010 of the ROI-total rms (3.3 % of the rms of h); after 16 x 16 pixel
  block aggregation L1 0.0080 against a noise level of 0.0006 (structured, not noise). The stored reference is not at fault: the
  audit's new REF (seeds 97700-97703) agrees with it within noise (signed +0.00006, block-16 L1 0.00093 vs noise 0.00098) and gives
  the same F deviation (+0.0027, block-16 0.0081).
- Split by event count with the event-split references (kmax 3 vs 6, identical streams; ROI_T):

| comparison | L1 (null) | signed | ratio | block-16 L1 (noise) | debiased rel L2 vs ROI total |
|---|---|---|---|---|---|
| table e2 read (Z) vs 3-event camera reference | 0.0124 (0.0080) | +0.0040 | 1.028 | 0.0070 (0.0006) | 0.0089 |
| tail table read (F - Z) vs 4..6-event reference, paired | 0.0053 (0.0032) | -0.0013 | 0.9926 | 0.0036 (0.0004) | 0.0047 |
| explicit tail read (XOD - Z) vs the same reference | 0.0054 (0.0032) | -0.0008 | 0.9957 | 0.0036 (0.0004) | 0.0047 |

  Whole image: tail table 0.0033 (0.0024), block-16 0.0014 (0.0003); box A pixels signed +0.00055 (ratio 1.0096) for the ceiling table
  and -0.00003 (0.9995) for the explicit tail.
- Consequences.
  (a) Supporting: against path-traced truth the explicit tail readout is as accurate as the ceiling table (L1 +0.00003, block-16
  -0.00002, debiased L2 equal). The no-headroom result does not depend on using the table as the yardstick.
  (b) Restricting: the "ceiling" is the best any operator with a 1270 x 64 table OUTPUT can do. The tail's table-readout error against
  truth (0.0036-0.0053 of the ROI total, about 2-3 % of the tail image) is of the size of M_img and 6-8x the explicit-vs-ceiling gap. It
  is common to explicit and learned table operators and was not attributed (patch / bin averaging at camera vertex 2 is the obvious
  candidate; not isolated here). WL96A D.1 "With no headroom over explicit, no representation can buy anything on this workload" is
  therefore supported only for representations that output this table; for any other output parameterization (learned or explicit) it
  is an untested inference.
  (c) `T_READOUT` in WL96A (1.7e-7) tests render reproducibility, not proxy validity; no test of proxy validity was part of the batch.
- Evidence: `img.json` (A_F_vs_REF_h), `extra.json` (table_readout_proxy); `pt86.py:228-233` (table lookup at camera vertex 2).
- Verdict: core claim unaffected (supports); the generalizing sentence UNTESTED_INFERENCE. Confidence high for the measurements,
  medium for the attribution to patch / bin averaging.

### E4-F7 (dataset_limitation, MATERIAL for scope) - receivers the ROI does not read
- Decomposition of the image change XOD - F by receiver class of the table rows (re-rendered masked difference tables): in ROI_T
  Lambert receivers 0.00065 of 0.00065, floor 0.00005, sphere 0.00005, S1 (panel) receivers 0.00000; whole image S1 0.00007, sphere
  0.00005; box A pixels S1 0.00009. Sphere and box B pixels: 0 in this camera.
- Field level, e3_o at receivers the camera does not read:
  - panel (S1) receivers: table rel L1 0.090 (n0 0.047) deployed, 0.064 (0.047) pooled - reproduced; debiased table rel L2 0.079-0.085
    deployed, 0.038-0.043 pooled; exitance fine (rel L1 0.010-0.013). WL96A calls the deployed excess "mostly variance"; the pooled
    residual (about 4 % L2) is real, so it is roughly half state-realization variance, half systematic;
  - sphere receivers: flux 0.954 +- 0.0003 deployed, 0.977 pooled; exitance rel L1 0.052 (n0 0.002); debiased table L2 0.094. This real
    -4.6 % / -2.3 % deficit is not reported in WL96A.
- Per-pixel view of the decisive arm: ROI_T max per-pixel relative change 0.30 %; box A pixels p95 0.55 %, 10 % of pixels above
  0.5 %, none above 1 % (one-signed, -1.0 % of the e3_o component as read there).
- Decision impact: "no accuracy headroom" holds for what this camera reads (Lambert receivers seen directly or through the panel). At
  the field level there IS explicit error on sharp and curved glossy readers; whether it would be image-relevant under another camera
  or geometry was not tested. This limits the claim to the tested readout, as its wording ("on the tested CK2 workload") allows.
- Evidence: `img.json` (receiver_class_decomposition, mask_pixels, stats), `fieldan.json` (field: S1, sphere).
- Verdict CONDITIONALLY_VALID. Confidence high.

### E4-F8 (invalid_inference risk, MATERIAL for how the claim may be cited) - input semantics of the comparison
- Like-for-like? The two sides are (i) "learned resolvent ceiling" = the path-traced tail table itself (an oracle OUTPUT, independent
  of any input and of any learnability), (ii) explicit recurrence fed the path-traced, bin-averaged 2-event table. The asymmetry is
  conservative in favour of the learned side, so the direction of the conclusion is safe. Three conditions come with it:
  1. exact prefix: with a deployable prefix (explicit q / 64-bin sender, the subject of P0-2 / P0-3) the errors upstream are larger
     than anything measured here, and a learned higher-order term trained end to end could in principle compensate prefix error; that
     role was not measured (by design);
  2. truncation: "tail" = 3..5 events. The 6-event order alone carries 7.8 % of the e3 flux (order fluxes k2..k6: 4559 / 2270 / 1164 /
     603 / 316); orders >= 6 are absent from the reference, the ceiling and the explicit arm alike. Explicit recurrence needs one more
     sweep per order; its accuracy beyond 5 events is unmeasured (drift so far -0.1 % of flux per order);
  3. one light, one geometry, one camera; no edit was reference-validated in this batch.
- The learned OWN arms G8601 / G8602 (ROI dI 0.0255 / 0.0126) have a different input and ownership (global term on a learned prefix);
  they are reproduced numerically but are not a measurement of "a learned resolvent's ceiling" and are not used by the decision rule.
  Citing them as evidence that learning is worse than explicit would be an INVALID_COMPARISON; WL96A labels them descriptive.
- Verdict CONDITIONALLY_VALID. Confidence high.

### E4-F9 (confirmation) - robustness of the decision and meaning of M_img
- Mask sensitivity of dI(XOD) (matched reference noise n_img_o in brackets): ROI_T 0.00065 (0.00071), DIFF 0.00107 (0.00101), whole
  image 0.00089 (0.00052), panel pixels 0.00124 (0.00093), panel outside ROI 0.00250 (0.00138), box A pixels 0.00058 (0.00013), floor
  pixels 0.00206 (0.00156), wall pixels 0.00100 (0.00078). Largest value 0.0025 = half of M_img. Box A pixels show the clearest real
  effect: one-signed -0.00058, i.e. -1.0 % of the e3_o component read there, consistent with the field flux 0.993.
- Statistic: L1 0.00065, luminance L1 0.00064, signed +0.00057, rms 0.00075, per-pixel relative mean 0.07 % / p99 0.22 % (ROI_T).
- Pooled instead of deployed state: 0.00060 (signed +0.00007).
- Noisy prefix: the gather averages prefix noise away. Explicit e3_o from independent prefixes differs by table rel L1 0.26-0.37 % at
  box A (prefix table noise 3.5 %) and by 0.00007 in the ROI image. Cross-independent pairings (prefix and reference tail from
  independent path sets) give 0.00054 / 0.00066 against 0.00065 / 0.00058 for same-source pairings: no bias in favour of explicit from
  shared noise. Because the explicit tail is smoother than the reference tail, dI(XOD) is floored by the REFERENCE tail's noise.
- What M_img = 0.005 means here: 0.5 % of the ROI radiance (direct + indirect) = 1.6 % of the higher-order image h = 2.8 % of the tail
  as read (0.178) = a uniform 8.0 % error of the e3_o component as read in the ROI (its share is 0.0623; box A pixels 8.8 %, DIFF 3.0 %,
  whole image 4.9 %). The measured explicit error is 1.0 % of that component. Detectability: paired camera noise 2e-5, reference tail
  noise 0.0005-0.0007 -> an effect of about 0.001 of the ROI total (about 2 % of e3_o) was detectable; the outcome is the same for any
  threshold down to about 0.001. "7x below M_img" is verified (0.00069 vs 0.005).
- Verdict VERIFIED. Confidence high.

### E4-F10 (provenance_gap, minor, not material)
- `results/` is not tracked for the audit outputs; the decisive raw inputs (renders, CKF_none, masks) exist only on the server
  (`~/rest_x96a`, `~/rest_x94`). Hash prefixes are in section 8. The WL96A worklog's field table and `s1.json` agree with the artifacts.
- Not re-derived: the 16-realization spread table (only realization 0 and the pooled mean were reproduced, to 2e-7 / 6e-8), the field
  metrics of the G arms, and the Stage 3 lifecycle cost (other half of P0-4, outside this track).

## 5. P0 assessment within this lens

- **P0-4, accuracy half: CONDITIONALLY_VALID.** Reproduced; decision robust. Valid as: "on CK2 `none` (one grazing point light, this
  camera), given an exact path-traced 2-event prefix table, three explicit 64-bin sweeps on the deployed state reproduce the 3..5-event
  broad-last tail so closely (ROI change 0.0005-0.0007, real residual about 1 % of that component) that a perfect operator with the
  same 1270 x 64 table output could not gain a meaningful image improvement (M_img = 0.005, any threshold >= 0.001)". Not valid as: a
  statement about other output representations, about glossy- or sharp-last higher-order transport, about receivers this camera does
  not read, about orders beyond 5 events, about a deployable prefix, or about other lights / geometries.
- **P0-4, cost half:** not assessed (outside track).
- **P0-5 (partial, from this lens only):** the accuracy result supports rejecting "a broad higher-order learned resolvent in ECRO table
  form as a source of accuracy gain on CK2"; it does not establish anything general. E4-F6(b) shows one concrete over-extension in the
  worklog text. Consistent with the second half of P0-5; the first half is only as strong as restrictions 1-5 of section 1.
- **Side observation for P0-2 (not this track, low weight):** with the panel as last sender in the 3..5-event tail (e3_p), the explicit
  error is real but mostly state-realization variance (all receivers debiased table L2 0.098 deployed vs 0.031 pooled; flux 1.00-1.01),
  unlike the systematic over-delivery reported for the grazing e2_p lobe. This says nothing about e2_p itself.

## 6. Special checks requested by the track

| check | result |
|---|---|
| (1) e2 = 2 events, e3 = 3..5, `_p` = adjacent vertex | confirmed by kmax experiment (bit-level) and by an independent tagged tracer (E4-F1) |
| (1) own reference vs CKF_none | flux ratios 0.9999-1.0004 for e2_o / e3_o; table differences at the null expectation |
| (1) earlier sharp interaction inside e3_o | 3.8 % at box A, 2.7 % overall, 28 % at panel receivers; binned panel sender <= 1.9 % (E4-F5) |
| (1) glossy floor / sphere as last sender | 23.1 % + 0.6 % at box A; 15.5 % + 1.5 % overall |
| (2) explicit tail from the prefix with own weights | reproduced to 2e-7; own sweep 1.8e-7; own ER weights 0.05 % (E4-F2) |
| (2) prefix / tail non-overlap, no double count | confirmed (kmax 2 -> e3 = 0; per-order flux ratios 0.994-0.996) |
| (3) paired statistics from stored renders | all reproduced (section 3) |
| (3) re-render with different camera seeds | dI(XOD) 0.00065 in 4 of 4 new frames; identity render 1.7e-7 |
| (3) F reproduces REF h within noise? | NO: structured deviation 0.008-0.010 of the ROI total, +0.8 % of h; tail part 0.0036-0.0053 (E4-F6) |
| (4) mask, statistic, noise subtraction, M_img, pooled state, noisy prefix | decision unchanged in every variant (E4-F4, E4-F9) |
| (5) receivers the ROI does not read | field-level explicit errors on panel and sphere receivers exist and are unread (E4-F7) |
| (5) like-for-like input semantics | asymmetric but conservative for the learned side; conditional on an exact prefix (E4-F8) |

## 7. Not verified, and why

- Lifecycle-cost half of P0-4 (other track). Stage 2 of WL96A was never run, so there is nothing to reproduce there.
- Field metrics of the learned G arms and the per-realization spread table (descriptive, not decisive; skipped for budget).
- Cause of the table-readout deviation (E4-F6) and of the sphere-receiver flux deficit (E4-F7): measured, not attributed.
- Exact share of e3_o that passes a binned panel sender (only the upper bound from the available tag).
- Orders beyond 5 events, other lights, other geometries or an edited geometry, other cameras: no data exist; none were generated
  (out of audit scope).
- The frozen ER closure (`DrClosure83`) and the Mitsuba ray intersection were used as trusted primitives.
- Whether a learned operator could compensate errors of a deployable prefix: not measurable from WL96A's design.

## 8. Reproducibility appendix

Inputs (sha256 prefixes): `~/rest_x94/results/fields/CKF_none.npz` 3a6238a8a42d1db4; `~/rest_x96a/results/pre/masks94.npz`
b9a7d46b1a23d8c4; `~/rest_x96a/results/tables/T96A.npz` 8019cb39d03421dd (= local `results/ecro_util96a/T96A.npz`);
`~/rest_x96a/results/tails.npz` 04da33a8b57eeb32; `~/rest_x96a/results/s1.json` 92f5e763126d83b6; stored renders
`~/rest_x96a/results/renders/{CK_none_<arm>_X_CK2_T1_r0..3,REF_ER_CK_none_CK2_T1_r0..3}.npz`; code `~/rest_x96a/code` (COMMIT a1718f7).

Audit scripts (`experiments/audit97/E4/`, copies in `~/rest_x97/E4/scripts/`; sha256 prefixes): e4_field.py f909c4e807bff6c5,
e4_kmax.py 3de919436f0fa22d, e4_tail.py 8b5e49b965880f47, e4_fieldan.py 2c9e7b1803397f38, e4_render.py d0bebc97e987b488,
e4_img.py 43f0b14579c9b78d, e4_ref.py 563c16f7ea7bcdef, e4_extra.py 6a17c5f640018c51.

Audit outputs: local `results/audit97/E4/` {kmax.json 836132edb41fb63a, tails97.json c3973d99f7f2f48c, fieldan.json f747505dd0035e16,
img.json 37dd2160e303518e, extra.json c54751079b9bdcfd, `_interrupted_run_log.md` (coordinator's log of the first attempt)}; server
`~/rest_x97/E4/results/` {TF_97401..4.npz 5d710f2b / a30d78c7 / c34f0544 / 14a9cc73, tails97.npz ceb92449e318a2bd, tables97.npz
91ddb26fdefd8741, renders/A_<key>_s<seed>.npz (65 files), refa/REFA_k{3,6}_s97700..3.npz, kmax_kmax3.npz}.

Commands (server, `cd ~/rest_x97/E4; . ~/rest_x95a/env.sh; C=~/rest_x96a/code/experiments`):

```
for s in 97401 97402 97403 97404; do $PY scripts/e4_field.py $C results/TF_$s.npz $s 4096 6 64; done
$PY scripts/e4_kmax.py $C ~/rest_x97/E4/Rk ~/rest_x94/results/fields/CKF_none.npz results/kmax.json 4096
$PY scripts/e4_tail.py $C ~/rest_x94/results/fields/CKF_none.npz ~/rest_x96a/results/tails.npz results/tails97 results/TF_9740{1,2,3,4}.npz
$PY scripts/e4_fieldan.py ~/rest_x94/results/fields/CKF_none.npz results/tails97.npz results/fieldan.json results/tables97.npz results/TF_9740{1,2,3,4}.npz
$PY scripts/e4_render.py $C results/tables97.npz results/renders 4096 86600 1 F
$PY scripts/e4_render.py $C results/tables97.npz results/renders 4096 97600 4 F,Fa,Fb,XOD,dOab
$PY scripts/e4_render.py $C results/tables97.npz results/renders 4096 97600 4 XD,XOP,Z,Pc,Fn
$PY scripts/e4_render.py $C results/tables97.npz results/renders 4096 97600 4 Pn,XODn,D_S1,D_floor,D_sphere,dOn
$PY scripts/e4_img.py ~/rest_x96a/results/renders ~/rest_x96a/results/pre/masks94.npz results/renders results/img.json
$PY scripts/e4_ref.py $C results/refa 4096 97700 4
$PY scripts/e4_extra.py ~/rest_x96a/results/renders ~/rest_x96a/results/pre/masks94.npz results/renders results/refa ~/rest_x94/results/fields/CKF_none.npz results/extra.json results/TF_9740{1,2,3,4}.npz
```

Seeds: tagged fields 97401-97404 (own seeding rule `seed * 7919 + 13 * launch + 5`, initseq 97); production tracer check 94301; state
realization 86001 (deployed) and 960000 + d, d = 0..15 (pooled), quadrature seed 87001 (+ 1000 (d + 1)); own ER weights 97100 + m;
table renders 86600 (identity) and 97600-97603; event-split camera references 97700-97703.

Resumption checks of this run (first-attempt outputs treated as unverified notes): `e4_fieldan.py` re-run -> byte-identical
`fieldan.json`; stored-tail reproduction recomputed from `tails97.npz` (2.08e-7 / 8.4e-8); audit tables vs `T96A.npz` (0 to 9.6e-8);
box A earlier-panel share recomputed from the raw TF files (0.0376); `e4_img.py` had a syntax error (one missing parenthesis) and
was fixed and run for the first time in this run; `e4_ref.py` and `e4_extra.py` are new. Table lookups use plain `pt86` (shape
materials); for the `none` variant this equals the mat94 per-patch map (identity render 1.7e-7; new REF vs stored REF within noise).

Weights and metrics (own): W_T[bin] = (ring + 0.5) / 4 x 2 pi / 64 (sums to pi); table rel L1 = sum |X - ref| W_T / sum |ref| W_T;
n0 = 0.5 x the same for (mean of first half-group - mean of second half-group); debiased L2: (||X - ref||^2 - sum var(ref mean)) / ||ref||^2;
image dI = sum over mask and channels |mean_reps(arm - F)| / sum(reference total); block-b L1 = the same after summing b x b pixel blocks.
