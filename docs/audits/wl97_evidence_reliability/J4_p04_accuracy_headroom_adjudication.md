# WL97 Phase 2 - Track J4: adjudication of the P0-4 accuracy half

Status: FINAL (2026-11-16). Independent adversarial adjudicator. Audit only: nothing trained, tuned, moved or edited outside the J4
namespaces.

Cluster: P0-4 accuracy half - "the learned broad higher-order resolvent has no meaningful accuracy headroom over explicit recurrence on
the tested CK2 workload" (WL96A, Stage 1 `S1_NO_HEADROOM`; worklog `docs/worklogs/96A_WORKLOG_LEARNED_HIGHER_ORDER_UTILITY_CEILING.md`).

Findings adjudicated: E4-F1 ... E4-F9, C2-09, B-08, F-MAP-11, F0-05, A-08.

## 0. Own decisive check (E4-F6)

Scripts (unchanged frozen code imported read-only from `~/rest_x96a/code`):

- `experiments/audit97/J4/j4_attr.py` - on the SAME camera prefixes (x1, x2) as the `ecro` camera mode
  (`hl = throughput(x1) * B[patch(x2), bin(x2 -> x1)]`, `experiments/ecro86/pt86.py:228-233`) it replaces the table lookup by fresh probe
  estimates of the continuation, with common random numbers:
  A = exact hit point x2, exact direction (path-traced continuation = truth);
  B = random point of patch(x2), exact direction (spatial patch averaging only);
  C = exact point, random direction in bin(x2 -> x1) (angular bin averaging only);
  D = random point in the patch and random direction in the bin (= the table's expectation as the field tracer defines it,
  `pt86.py:327-351`, without table sampling noise).
  Each with probe kmax 2 (camera 3 events = table e2) and kmax 5 (camera 4..6 events = table e3). 8 reps x 2,048 spp (seeds 97900-97907).
- `j4_tread.py` - reads, on identical lanes (max lane difference 3e-8), the stored CKF_none tables (Tc) and the INDEPENDENT audit tables of
  track E4 (Tn, 4 new field halves). Includes a patch-layout check (`layout_check.json`: every random patch point within 0.98 patch diagonal
  of the hit, normals identical).
- `j4_an.py`, `j4_an2.py` - numpy statistics relative to the ROI total of the stored path-traced reference (e1 + e2 + h); debiased L2 from
  cross products of independent rep halves; 16 x 16 block L1 with a noise realization.
- `j4_split.py` - split by camera-first-vertex class.

Outputs: `results/audit97/J4/j4_attr.{txt,json}`, `j4_tables.{txt,json}`, `j4_split.txt`, `layout_check.json` (server `~/rest_x97/J4/out`).

### Results (ROI_T; all numbers are fractions of the ROI_T reference total)

| check | value | reading |
|---|---|---|
| tracer sanity: A (kmax 2) vs E4's new 3-event REF | ratio 1.0008, block-16 L1 0.0010 vs noise 0.0010 | camera-side event accounting exact |
| tracer sanity: A tail (kmax 5 - 2) vs E4's REF h6 - h3 | ratio 0.9997, block-16 L1 0.0005 vs noise 0.0005 | no event-order mismatch between table (2..5) and REF (camera 4..6) |
| stored table read (T2 + T3) vs stored REF h | signed +0.0026, block-16 L1 0.0080 (noise 0.0006), debiased rel L2 0.0101 | E4-F6's error reproduced |
| D - A, e2 content (camera 3 events) | debiased rel L2 0.0074, block-16 L1 0.0059 (noise 0.0010) | structural |
| D - A, tail (camera 4..6) | debiased rel L2 0.0037, block-16 L1 0.0026 (noise 0.0005), L1-equivalent 0.0019 | structural |
| B - A (spatial only) | = D - A to 1e-5 in every mask | the cause is patch spatial averaging |
| C - A (angular bin only) | rel L2 0.0001; interaction 0.0002 | 64-bin angular averaging irrelevant here |
| table-noise-free error common to two independent tables: <Tc - A, Tn - A> | e2 0.0069, tail 0.0031 | confirms D - A without any noise model |
| systematic table-vs-D: <Tc - D, Tn - D> | -0.0022 / -0.0019 (no positive systematic part) | the tables are D + estimation noise |
| CKF-table-specific part (Tc - D) | e2 0.0053, tail 0.0034 (audit 4-half table Tn - D: 0.0055 / 0.0020) | reference-table estimation noise, shrinks with more halves |
| ROI_T split by camera first vertex | ROI_T lies 100 % inside the panel mask | the ROI is the sharp panel's reflection of box A |
| directly viewed box A pixels (BOXA_PIX) | D - A rel L2 0.0004 (e2) / 0.0004 (tail) of their total | readout error is small where x1 is Lambert |
| DIFF (diffuse reader) | tail D - A block-16 0.0016 vs noise 0.0017 | not resolved above noise |
| WHOLE frame | D - A rel L2 0.0026 / 0.0016 | |

Conclusion of the check. The common-mode table-readout error is REAL. It is the patch spatial averaging of the 1,270-patch output table at
camera vertex x2, imaged "in focus" through the sharp panel reflection that defines ROI_T (on Lambert-first pixels it averages out). It is not
an event-order mismatch (sanity ratios 1.0008 / 0.9997), not reference-noise mis-scaling (stored and new references agree; block noise
0.0006), and not angular bin discretisation. A second, smaller part of the stored-table readout error is the CKF reference tables' own
estimation-noise realization, frozen in every arm that reads them. Of the tail readout error of the stored table (rel L2 0.0048), about
0.003-0.0037 is irreducible structure of the patch parameterization and the rest is reference-table noise. Note: half-split noise
estimates of these heavy-tailed tables are unstable (e.g. Tc - Tn exceeds the summed half-split noise in the tail, falls below it in e2),
so table-noise magnitudes are reliable only to about +-0.002; the structural numbers (D - A and the cross product) are not affected.

## 1. Rulings

### E4-F6 - UPHELD_WITH_MODIFICATION (material)

Corrected statement: The floor table F reproduces the path-traced REF h only in table space. Against truth its ROI_T image carries a
structured readout error (debiased rel L2 0.010 of the ROI total; block-16 L1 0.008 vs noise 0.0006), almost entirely from the
1,270-patch spatial averaging at camera vertex x2 seen through the sharp panel (e2 content 0.007, tail 0.003-0.004) plus the CKF tables'
estimation noise. The error is linear in the table and enters the explicit arm and the ceiling identically, so it cancels exactly in
dI(XOD, F) = 0.00065: the S1_NO_HEADROOM decision is unaffected. Modifications to E4's wording: (i) "the tail-table readout alone deviates
by about M_img" - the irreducible structural part is about 0.0019-0.0026 in L1-equivalent / block terms (about 0.4-0.5 M_img); the rest is
reference-table noise; (ii) the error source is identified (patch spatial averaging), it is not unattributed any more; (iii) a gain from
another output parameterization is not learned-vs-explicit headroom - it is equally available to an explicit recurrence with the same
finer output, and the largest part (e2 content, 0.007) lies in the prefix, outside the resolvent domain.

Consequences:
- (a) WL96A S1_NO_HEADROOM: stands (table-space comparison, common-mode error cancels).
- (b) WL95A "oracle-table floor" (`docs/worklogs/95A_*.md:44,92,134`): E_H(floor) 0.0118 exceeds the per-arm noise bound (~0.0083)
  because of this readout error; neither WL95A nor WL96A remarked on it (WL96A table B2 lists "0.0118 (0.0082)"). The floor is a
  table-space floor, not an image-level truth floor, and not a lower bound (an arm's error could partially cancel it). "Pooled X1 =
  floor" means equivalence to the path-traced reference table read through the same patch lookup, not agreement with path-traced truth to
  within noise. No WL95A ROI conclusion flips (all differences are compared against a tolerance of 0.0167).
- (c) What a learned resolvent could gain: with the same 1,270 x 64 output, nothing measurable on this ROI (decision stands). With another
  output parameterization (hit-point / sub-patch output), the tail readout structure available on ROI_T is about 0.002-0.004 of the ROI
  total - below M_img - and is not specific to learning. "No representation can buy anything on this workload" is an untested inference
  beyond the patch x bin output; the bounded figure above replaces it.

### E4-F1 - UPHELD (not material)
Event accounting confirmed independently by J4 on the camera side (A vs 3-event REF ratio 1.0008; tail ratio 0.9997) and by A-08 (code).

### E4-F2 - UPHELD (not material)
Explicit tail reproduction is independently confirmed by C2-09 (field recomputation 2.9e-17). Caveat retained: the ER closure
(DrClosure83) is a shared primitive not re-derived; mild -0.1 %/order flux drift (orders 3..5 flux 0.996/0.995/0.994).

### E4-F3 - UPHELD (not material)
dI(XOD) 0.00065 reproduced by E4 and independently by C2-09 (0.000646); new camera seeds and audit references give 0.0005-0.0007.

### E4-F4 - UPHELD (not material)
n_img (0.00146) is the half-split of the whole e3 (w96a.py:314) while XOD - F differs only in e3_o; matched noise 0.00071 (E4 and C2
agree), H -0.00006 instead of -0.00082; "L1 minus L1" is not a valid debiasing estimator. WL96A already stated that the decision does not
depend on the subtraction (raw dI + 2 nb = 0.0007). Corrected reading of "at the reference noise level": the explicit e3_o error is at
most about 1 % of the e3_o image (0 to 0.0006 of the ROI total; E4 debiased 0.00058, C2 0 +- 1.1 % of e3_o - consistent within
resolution), about 7-8x below M_img.

### E4-F5 - UPHELD (material for citation only)
Verified against stored `results/audit97/E4/fieldan.json`: box A e3_o last-sender shares Lambert 0.762 / floor 0.231 / sphere 0.006;
earlier panel 0.038 (SE 0.002). Per-class explicit errors exist on glossy-last paths (box A floor-last exitance rel L1 0.024 vs n0 0.008,
flux 0.995; all-receiver floor-last flux 0.985). WL96A disclosed the untagged mixture qualitatively (worklog lines 26, "Untagged
predicate") but not the magnitudes. The CK2 outcome is unaffected; the result is not evidence about higher-order transport through
glossy or sharp last senders.

### E4-F7 - UPHELD_WITH_MODIFICATION (material for scope)
Verified in `fieldan.json` across three reference sets (CKF, audit, all-6): sphere receivers e3_o flux 0.954 (deployed) / 0.977
(pooled), exitance rel L1 0.052 vs n0 0.002-0.004 - real and not reported in WL96A (the only sphere mention is the domain definition,
line 26); sphere and box B cover 0 camera pixels. S1 (panel) receivers: deployed table debiased L2 0.075-0.085; pooled residual 0.023-0.043
depending on the prefix source (E4 quoted only the CKF-prefix 0.038-0.043; with the all-6 prefix it is 0.023-0.028, so part of it is
prefix noise). Exitance at S1 receivers is accurate (rel L1 0.009-0.014 vs n0 0.007-0.011). WL96A's "mostly variance" for S1 is
correct for the deployed-vs-pooled change but leaves a real pooled residual of 2-4 % (table L2). Corrected statement: "no accuracy
headroom" holds for what the ROI_T and DIFF masks read (box A via the panel reflection and floor/wall diffuse readers), not for every
receiver of the declared e3_o domain.

### E4-F8 - UPHELD (material for citation)
Verified: the ceiling is the path-traced e3 table (output), the explicit arm is fed the path-traced 2-event table (w96a.py:250;
A-08, C2-09); orders k2..k6 = 4559/2270/1164/603/316 so the 6th table event is 7.8 % of e3 flux and absent from reference, ceiling and
explicit alike (the camera reference is truncated identically, cf. J4 tail ratio 0.9997). The asymmetry is conservative for the learned
side; G8601/G8602 have different input and ownership and are descriptive only. Cite only with: exact (oracle) prefix; 3..5 table events;
one light, geometry, camera.

### E4-F9 - UPHELD (not material)
Mask/statistic robustness agrees with C2-09's independent sweep; M_img 0.005 = 8 % of e3_o as read in ROI_T. One addition from J4: because
the ROI is a sharp reflection reader, it is the most sensitive reader of patch-level output error in the frame; robustness in ROI_T is
therefore not weakened by the readout finding.

### C2-09 - UPHELD (decision part not material; debias-term defect real but inconsequential)
Same numbers as E4-F3/F4 from an independent recomputation. "Material=True" is overstated: the flawed debias term does not change the
outcome under any mask or any M_img >= 0.0007.

### A-08 - UPHELD (material for framing)
Code reading confirmed numerically by J4 (camera side). "No headroom" is a statement about propagating a correct 2-event field on CK2,
not about a self-contained explicit method.

### B-08 - UPHELD (not material)
M_img provenance verified (`ecro_local93d.json:33,36`, `ecro_s94d.json:39`, `ecro_util96a.json:44`). Modification: in WL93D-WL95A 0.005
was a lower clamp of a detection tolerance, max(noise, 0.005); WL96A re-purposes it as a "minimum meaningful effect". This is a reframing,
harmless here because the outcome holds for any threshold >= ~0.0007-0.001 (E4-F9, C2-09).

### F-MAP-11 - UPHELD_WITH_MODIFICATION (material for framing only)
Timeline and probe-vs-frozen identity accepted as reported (not re-run). Modification: "thresholds were set with the prediction in hand"
does not imply tailoring - M_img was inherited from five earlier protocols and the 1.25x lifecycle rule from WL96 §F; the protocol
criteria are identical between dev/probe and freeze. The pre-freeze exposure (dev figure with real reference and real explicit operator,
disclosed at worklog lines 254-259) weakens the "frozen before evidence" label to "criteria frozen before the frozen chain ran"; it cannot
have changed the decision (7x margin; outcome invariant down to M_img 0.0007). Decision reliable; prospective framing overstated.

### F0-05 - UPHELD (material for the WL96 dissent statement)
Verified against `docs/worklogs/96_*.md:198-236`. WL96A states it "closely follows WL96 §F, restricted to CK2" but does not list the
departures. Unanswered parts of WL96 §F:
1. Workloads: the T0/T1 box edit with references (WL96A edit is timing-only); the prequalified non-grazing second light with a new
   peaked-class reference; the WL89 W1 rotating-slab workload (no availability check reported).
2. Stage 1 V0 contract (deterministic e2_p oracle vs path-traced e2_p) - not run.
3. Stage 2 sharp-prefix relevance at a render spp able to resolve the predicted 0.0127 - not run.
4. Stage 3: the in-sample rank-24/64 tail fit as a standing arm (only conditional, not triggered); field-level per-region H_acc as the
   decision quantity (reported only); cost at N = 2,802 / 5,080 (only N = 1,270).
5. Decision mapping: under WL96's rule ("H_acc <= max(2 SE, n0/2) in every region") the deployed S1 e3_o region is flagged (0.043 vs
   0.024), so WL96's confirmation condition is NOT met; ONE_FALSIFIABLE_CANDIDATE needs a flag on two workloads, which one workload cannot
   produce. Under WL96's own mapping the result is formally indeterminate; WL96A's own (different) frozen mapping gives
   NO_MEANINGFUL_HEADROOM.

## 2. Verdict on P0-4, accuracy half

**CONDITIONALLY_VALID.** "A perfect broad-last (e3_o) resolvent has no meaningful accuracy headroom over three explicit 64-bin sweeps" holds
under all of the following restrictions:
1. CK2 `none`, static, one grazing point light, one camera; deployed state realization (and 16-draw pool); N = 1,270 patches x 64 bins.
2. Oracle prefix: the explicit tail is seeded with the path-traced 2-event table; the "ceiling" is the path-traced e3 table itself. No
   learned resolvent was fitted or trained - a ceiling argument, not a test of a learned operator.
3. Table space: operators whose output is the same patch x bin table read at camera vertex 2. Image-level truth agreement of the floor
   itself is not established (structured readout error ~0.010 rel L2 of ROI total, from patch spatial averaging); another output
   parameterization was not tested (bounded tail gain ~0.002-0.004 of the ROI total, not learning-specific).
4. What the ROI_T / DIFF masks read (box A via the panel reflection; diffuse floor/wall). Real explicit field errors remain on unread
   receivers (sphere flux 0.954 / 0.977; S1 pooled table L2 0.02-0.04; glossy-last sub-families).
5. 3..5 table events; the 6th order (7.8 % of e3 flux) is absent everywhere.
6. Image-level rel-L1 metric; robust for any threshold >= ~0.001.
7. It does not satisfy WL96 §F's confirmation rule and does not answer WL96's dissent beyond one view's image-level ROI.

Corrected wording of WL96A interpretation sentences:
- D, "No accuracy headroom" (line 187): "On CK2 (one grazing light, this camera), with the path-traced 2-event field as prefix, a resolvent
  that outputs the exact path-traced e3_o table in the same 1,270 x 64 patch x bin parameterization would change the ROI_T image by at most
  0.0007 of the ROI total (explicit residual <= ~1 % of the e3_o image), about 7x below M_img. Both arms share a table-readout error against
  path-traced truth (patch spatial averaging, ~0.010 rel L2), which cancels in this comparison."
- D, claim 1 (line 192): "Representability of the optimistic free fit: not tested. For outputs in the patch x bin table there is nothing
  to gain on this ROI; other output parameterizations were not tested (their tail gain on this ROI is bounded by ~0.002-0.004 of the ROI
  total and is equally available to explicit recurrence)."
- C, "at the reference noise level" (line 170): "about 1 % of the e3_o image (0-0.0006 of the ROI total); n_img overstated the relevant
  noise by ~2x."
- C, S1 receivers (section C, "panel receivers' own e3_o tables" bullet; field rule at line 119): add "pooled residual 2-4 % table L2 is real; sphere receivers (not imaged) under-deliver e3_o by 4.6 % / 2.3 %
  (deployed / pooled)."
- E, WL96 dissent (line 212): "answered for the broad-last resolvent at the image level for one CK2 view with an oracle prefix; WL96 §F's
  edit, second light, field-region and N-scaling conditions remain open, and under WL96's own field rule the deployed S1 region is flagged."
- WL95A floor (lines 44, 92, 134): "oracle-table floor (table space; its own image error vs path-traced truth is 0.0118 L1 against a
  ~0.0083 noise bound, from patch readout discretisation)."

## 3. Affected statements

| worklog | original claim | defect / limit | corrected interpretation | dependents | action |
|---|---|---|---|---|---|
| WL96A D l.192 | "no representation can buy anything on this workload" | valid only for patch x bin output; floor has a structured readout error vs truth | as in section 2 | P0-4, WL96 dissent closure | QUALIFY |
| WL96A D l.187 | no accuracy headroom | restrictions 1-7 | as in section 2 | P0-4, P0-5, roadmap entry for WL96A | QUALIFY |
| WL96A C l.170 | explicit e3_o "at the reference noise level" | mismatched noise term (E4-F4/C2-09) | ~1 % of e3_o, ~7x below M_img | none | QUALIFY |
| WL96A B2 table | E_H(F) 0.0118 (0.0082) without comment | floor exceeds its own bound | readout discretisation (attributed by J4) | WL95A floor | QUALIFY |
| WL96A C (panel receivers bullet) | S1 e3_o deployed excess "mostly variance" | pooled residual 2-4 % real; sphere deficit unreported | add the residuals; outside ROI | P0-5 citations | QUALIFY |
| WL96A E l.212 | WL96 dissent "now answered" | narrower than WL96 §F; §F rule not met | answered for one view's image ROI only | WL96 verdict status | QUALIFY |
| WL96A disclosures | "frozen before evidence" | dev figure / probe pre-freeze (disclosed) | criteria frozen before the frozen chain; decision unaffected | none | QUALIFY |
| WL95A l.44/92/134 | "oracle-table floor"; "pooled X1 = floor" | table-space floor, not truth floor | table-space equivalence | WL95A ROI statements | QUALIFY |
| WL96A S1_NO_HEADROOM decision value | U < M_img | none (common-mode error cancels; noise-term flaw inconsequential) | unchanged | - | PRESERVE |
| WL96A G-arm statements (descriptive) | G 9-40x further from floor | different input/ownership | descriptive only, not evidence against a learned resolvent | - | PRESERVE (as descriptive) |

## 4. Unresolved
- Whether a small systematic CKF-vs-audit difference exists in the tail tables (Tc - Tn exceeds summed half-split noise: 0.0047 vs
  ~0.0024 rel L2) or the half-split noise estimates of these heavy-tailed tables are simply unstable (e2 shows the opposite sign).
  Needs more independent field halves; it does not affect the decision (common-mode).
- The e3_o vs e3_p split of the structural readout error (probe arms were not split by the panel tag).
- The image relevance of the sphere / S1-receiver explicit errors under another camera, light or geometry (untested).
