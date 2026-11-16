# WL97 Phase 2 - Track J7: data, protocol integrity and the statistical basis of the founding failure

Adversarial adjudicator report (independent reviewer). Date 2026-11-16. Repository HEAD efcf86a (branch Selective_Recompute) plus untracked audit files. No existing file was modified, moved or deleted.

Findings adjudicated: B-02, B-03, B-04, B-09, B-10, B-11, MAP-01, MAP-05, MAP-09, MAP-10, MAP-13, MAP-14, C1-03, C1-07, C2-11, C2-13, C2-14, C2-16, F0-01, F0-02, F0-03, F0-09, F0-14.

Own artefacts:

- `experiments/audit97/J7/founding_failure_check.py` -> `results/audit97/J7/founding_failure_check.json` (decisive recomputation of the WL86 F2 failure, WL87 state-draw resampling, WL92A E4 and WL92B E5 tables, WL87 region-row counts; stored JSON only).
- `experiments/audit97/J7/make_manifest.py` -> `results/audit97/J7/p0_input_manifest.json` (P0 input identity manifest, MAP-13).

All other evidence was read from the stored result files and code cited below. No server job was run: the questions could be decided from local raw JSON.

## 1. Decisive check: the founding failure (F0-01, F0-02, C2-11)

The WL86 F2 rule is frozen as `E_H <= 0.06 in A, C and D at T1, E1 and E2` (`experiments/ecro86/protocol/ecro86.json:97,102`). The code applies it only at T1 (`experiments/ecro86/analyze86.py:210-212`). Every other F2 cell is at most 0.0126. The only failing cell is E2 T1 region C:

| quantity | s8601 | s8602 |
|---|---|---|
| WL86 deployed draw (86001), E2 T1 C E_H | 0.0717 | 0.0634 |
| excess over 0.06, in units of the stored reference noise bound | 0.93 | 0.27 |

WL87 (`results/gm1_explicit87/results/stage2_87.json`, per_edit/E2) re-evaluated **only the s8601 checkpoint** over 17 relation-state draws: the deployed draw 86001 and 16 fresh draws 87201-87216. The deployed row reproduces WL86 exactly (0.07172).

| E2 region C, s8601 | T0 | T1 |
|---|---|---|
| draws passing 0.06 (all 17) | 12 | 9 |
| draws passing, fresh 16 only (= WL87's stored `frac_pass` 0.5625 at T1) | 12/16 | 9/16 |
| mean / min / max over 17 draws | 0.0466 / 0.0175 / 0.0682 | 0.0621 / 0.0412 / 0.0809 |
| paired ECRO - EXP, mean +- se (n = 17) | +0.0054 +- 0.0039 (11/17 worse) | **+0.0174 +- 0.0036 (15/17 worse)** |
| explicit EXP draws passing | 14/17 | 13/17 |

Rulings on this evidence:

- **F0-01 is UPHELD_WITH_MODIFICATION.**
  - The F2 failure is a correct application of the frozen point rule to the deployed draw. For s8601, though, it is a near coin flip over the relation-state draw: 9 of 16 fresh draws pass, and the 17-draw mean 0.0621 has a confidence interval [0.055, 0.068] that spans the rule.
  - Corrections to F0-01:
    - The resampling exists for s8601 only. s8602 was never resampled, so "56 % over state draws" is a one-checkpoint statement.
    - F2 is a T1-only rule. The 0.03-noise-bound "miss" F0-01 cites is the T0 value 0.0603, which no gate tests.
  - The noise bound is a reference-noise scale, not a state-draw uncertainty. The rule makes no noise allowance.
  - Two statements are solid:
    1. The deployed WL86 operator fails F2 as frozen. Both seeds fail on their deployed draw.
    2. At E2 T1, ECRO s8601 has a paired, resolved deficit against the same-state explicit operator (about 4.8 se).
  - The narrated premise "ECRO v0 systematically fails held-out sharp transport" is stronger than this.
- **F0-02 is UPHELD_WITH_MODIFICATION.** The WL92B E5k995 table (the only independently admitted, never-inspected held-out edit; 0 margin violations against every training population, B-03) gives:

  | E5k995 | T0 region C | T1 region C |
  |---|---|---|
  | frozen ECRO s8601 / s8602 | 0.0368 / 0.0295 | 0.0215 / 0.0236 |
  | OWN s8601 / s8602 | 0.0339 / 0.0169 | 0.0327 / 0.0274 |
  | explicit X_EXP | 0.0174 | 0.0299 |
  | ECRO - X_EXP | +0.019 / +0.012 | -0.008 / -0.006 |
  | noise bound | 0.008 | 0.0074 |

  - On E5 panel pixels at T1, frozen ECRO scores 0.029 / 0.033 against explicit 0.042.
  - On E4 at T1 (not admitted; descriptive), learned scores 0.012-0.025 against explicit 0.027.
  - E2 is therefore the only edit on which any ECRO-lineage model misses the 0.06 rule (E1, E3, E4 and E5 all pass).
  - "Explicit >= ECRO everywhere" is not supported beyond WL87's three edits:
    - At E5 T1 the sign reverses by about one noise bound.
    - At E5 T0, explicit is better by 1.5-2.4 noise bounds.
  - Correction to F0-02: the E5 and E4 comparisons are single deployed draws. WL87 shows the state-draw SD of region-C E_H is about 0.012, the same size as the E5 differences. So E5 *fails to replicate* explicit dominance; it does not statistically *contradict* it.
  - Leakage caveat: frozen ECRO's E4 value is not a clean held-out, because the WL86 population has 96 files that violate the E4 margin rule (B-03). Only OWN/SUM are E4-disjoint.
- **C2-11 is UPHELD.** The recount from `final87.json` gives 36 rows: ECRO better 0, worse 3, within the unpaired bound 33. So "36 equal" should read "33 within noise", and "everywhere" overstates a within-noise outcome.

**What WL86 and WL87 legitimately establish.**

- WL86 (NOT SUPPORTED, frozen):
  - The deployed operator, at its deployed state, fails F2 on E2 T1 region C for both seeds by 0.27-0.93 noise bounds.
  - It passes every other fidelity gate (F1, F3, F4).
  - That is a valid frozen outcome. It is not evidence of a systematic class deficit.
- WL87:
  - At E2 T1, the learned operator has a paired, statistically resolved deficit relative to same-state explicit transport, on one checkpoint.
  - Elsewhere on E1-E3, learned and explicit are equal within noise.
  - The learned operator never wins significantly.
- Neither establishes that ECRO fails held-out sharp transport in general. The independent held-out E5 shows no failure, including on panel pixels.

## 2. Data and population rulings (B-02, B-03, B-04, B-10, C1-07)

### B-02: UPHELD_WITH_MODIFICATION (not material)

- Confirmed from `results/audit97/B/heldout_disjointness.json`:
  - Under the single-condition margin rule (box centre < 0.15 m or light < 0.3 m), CK2 has 17 violating files in the 928-file OWN (WL92A) population and 20 in WL86's 1,024. It has 0 in WL94's 96 (the WL94 rule, `experiments/ecro_mat94/protocol/ecro_mat94.json` stage_C, was applied only there).
  - The nearest *joint* geometry is 0.94 m away.
- Direction: proximity of single objects or lights would favour the learned model. CK2 conclusions are unfavourable to it, so the leakage does not weaken them.
- No worklog claims that CK2 was margin-excluded for the OWN checkpoints. WL94 only claims T_LEAK for its own 96-file fine-tune population, which is true.
- Corrected statement: CK2 is not a training item of any checkpoint and no training file is jointly near it. 17-20 training files place a single box or light within the margin, so for WL86/OWN checkpoints CK2 is "not trained on", not "margin-excluded". Its novelty is in transport (B-04), not in object placement.
- No "held-out" wording is invalidated.

### B-03: UPHELD_WITH_MODIFICATION (not material for WL87; relevant to E4)

- E3a has 25 and E4 (=E3b) has 96 single-condition violations against the WL86 population. E5k995 has 0.
- WL87's "untouched" means "radiance not inspected" (WL87:96, 338). It never meant margin-disjoint.
- Direction: possible leakage favours frozen ECRO on E3 and E4. E3 parity and E4 learned-better-than-explicit for *frozen ECRO* are therefore not clean held-out evidence.
- OWN/SUM are E4-disjoint (WL92A removed the 96 files; T_DISJOINT).

### B-04: UPHELD_WITH_MODIFICATION (material)

- `results/audit97/B/panel_coverage.json` and `results/ecro_exp94b/audit.json` confirm:
  - Ring-0 (grazing) mirror-to-box proxy >= CK2: 4 of 3,712 OWN lightings (0.11 %), 0 of 384 WL94 lightings.
  - Panel-mediated loss mass 3.4 % (e2_p 0.021 + e3_p 0.013).
  - 4 of 96 items reach CK2's panel share.
- Modification: the rarity is specific to **grazing** incidence. The same coverage file shows 1,727 of 3,712 OWN lightings (47 %) with a non-grazing panel mirror-to-box proxy >= CK2's. Sharp panel-to-box transport in general is common in training; sharp grazing panel delivery is not.
- "Never directly supervised" is correct in the sense that the objective reaches the S1 once-scattered response only through receivers.
- Consequence: CK2 tests an out-of-support *incidence regime* of the trained checkpoints. It does not test a transport type absent from training.

### B-10: UPHELD_WITH_MODIFICATION (material)

- Confirmed in `docs/worklogs/93D_WORKLOG_ADDENDUM_CHECKERBOARD_QUALITATIVE.md`:
  - The scene was built post hoc and is non-numbered. Revision 1 was withdrawn.
  - It went through five render probes, p01-p05 (l.43-50).
  - The box-A corner x is -1.345 against the rule limit -1.35 (l.38).
  - The protocol declares "labels: none" (`protocol_checker93d.json`); the addendum says "Architecture Verdict: None" (l.159-161).
- Modification: the iteration criterion was visibility of the checker in the **reference** reflection (l.52: reference LOW ratio). It was not error of the learned model. So there is no selection *for learned failure*. There is a selection *for maximal grazing sharpness at the edge of the training rule*, which is exactly the regime B-04 shows to be unsupported.
- The scene was then used as the sole quantitative workload of WL94-WL96A without a frozen admission. WL93D's E6 candidates were 0 of 3 admitted; E4's Q4 was 0.00472 vs 4 x 0.00121.

### C1-07: UPHELD_WITH_MODIFICATION (material)

- E4/E5 values verified (section 1). The E5 panel pixels also show no learned failure.
- Modifications:
  - Only E5 is a qualified held-out. E4 is not admitted, and for frozen ECRO it is not margin-disjoint.
  - The E5 comparison is one draw.
  - E2's failing pixels are panel pixels (frozen ECRO obj_panel 0.0555 at E2 T1, sphere 0.0106).
- The "held-out edits do not show the failure" statement stands.

## 3. Learnability and statistics (B-09, F0-09, MAP-14, C1-03, F0-14, C2-14)

### B-09: UPHELD_WITH_MODIFICATION (material)

- WL94B G_EXPOSURE is 1.232 / 1.236 vs 1.25. The pre-freeze note (`ecro_exp94b.json` /pre_freeze_exposure) records that G_EXPOSURE needs contrast >= 2.33 and binds before G_CONTRAST. The dev subset had already failed both gates. Evidence contrast was 2.20.
- The WL92 G_rep stop (`results/ecro_attr92/results/gate.json`):
  - The only in-family fit below 0.06 (OWN_k16 0.036 / 0.017) was flagged unconverged.
  - "best_in_family_owned" therefore fell back to R16 0.084, above the level.
  - So the stop is rule mechanics on an unconverged fit, and the evidence it rests on leans *towards* representability.
- Modification: learnability was not tested at **full scale**. It was tested only by 1,000-2,000-step smoke fine-tunes:
  - WL94C query weighting: INCONCLUSIVE.
  - WL94D sender supervision with a teacher (M64) known before the freeze to over-deliver on CK2: NO_RELIABLE_IMPROVEMENT.
  - Step counts per protocol: WL92A 24,000; WL93A/B/C, WL94, WL94A, WL94C 2,000; WL94D 1,000.
- These smokes are non-results, not negative results.
- WL94B's stop is a correct predeclared validity stop, not a defect. Its consequence is only that exposure was never trained.

### F0-09: UPHELD

Same evidence. WL95:313 files WL94D and the "zero-operator level" as weakening evidence for the learned-sender class, while WL95 itself lists five unseparated causes. That is a stronger reading than the labels allow.

### MAP-14: UPHELD

The label census and training scale are verified (step counts above). Of its decisive results (WL95A, WL96A), WL96A is learning-free, and WL95A is learning-free apart from evaluating the two frozen OWN checkpoints' learned sender (no training). Both are CK2-only.

### C1-03 and F0-14: UPHELD_WITH_MODIFICATION

- Both quantities are stored and consistent once normalisation is taken into account:
  - C1-03 normalises box-A delivery by M64: 0.41 vs 1.27.
  - F0-14 normalises by the reference: 0.52 vs 1.63.
  - The ratio of the two normalisers is 1.27-1.28 in both seeds, which is M64's own over-delivery.
- So the two findings agree: s8601 under-delivers and s8602 over-delivers relative to truth. The variation is not a normalisation artefact.
- The "26 pairs >= 40 sigma" uses Monte Carlo render noise only. It shows that training variability, not render noise, is the limit.
- Modification to C1-03's label: the reported INCONCLUSIVE and NO_RELIABLE_IMPROVEMENT labels are themselves correct. The UNTESTED_INFERENCE applies only to later readings of those arms as representation limits (WL95:313).

### C2-14: UPHELD (not material for P0)

- WL91:285-290: rule (a) is violated by one deployed draw (W2 T1, 3.1x), while (b) and (c) are met.
- The rule as written is a maximum over draws and tiers, so the label follows the rule. The weakness is the rule design, not an error.
- It concerns the explicit C3URX1 comparator, not ECRO.

### Verdict labels resting on n = 2 checkpoints or on one draw

| batch | label | basis |
|---|---|---|
| WL86 | NOT SUPPORTED (F2) | one deployed draw per seed; s8601 resampled pass rate 9/16 |
| WL87 | ECRO_worse at E2_T1:C and obj_panel; EXP sharp_by_edit E2 pass = false | deployed draw (EXP realization pass 0.75) |
| WL88 | STOPPED_WORKLOAD_UNQUALIFIED | one Q5 SNR value, 3.7 vs 4 |
| WL91 | QUALITY-MATCHED ECONOMIC UTILITY NOT SUPPORTED | one draw |
| WL92 | STOP_INTERVENTION_G_rep | unconverged-fit rule |
| WL92A | E4 not admitted / D_STOPPED | Q4 3.9 vs 4; OWN 2 seeds with E2 T1 0.077 vs 0.040 |
| WL92B | outcome C (see MAP-09); OWN_CONSISTENT_ADVANTAGE false | 2 seeds, 1 of 4 cases pass |
| WL93 | STOPPED | one preflight value, 0.508 vs 0.485 |
| WL93A, WL93C | NO_SIGNAL_AT_SMOKE_SCALE | 2 seeds |
| WL93B | SHARING_SIGNAL | 2 seeds |
| WL93D | REPRESENTATION_OR_OPTIMIZATION_DOMINANT + COADAPTATION_LIMITED | 2 seeds |
| WL94 | H-M / H-A INCONCLUSIVE | 2 seeds |
| WL94A | INCONCLUSIVE; forgetting answer CONTRADICTED | forgetting answer reads s8601 only; s8602 gives SUPPORTED (disclosed, WL94A:83,155) |
| WL94B | STOP_INVALID_INTERVENTION | 2 seeds |
| WL94C | INCONCLUSIVE | seed disagreement |
| WL94D | NO_RELIABLE_IMPROVEMENT | 2 seeds |
| WL95A | RANK16 INADEQUATE; FLIGHT_ATTRIBUTION; NEXT_DESIGN_TARGET | 2 checkpoints; flight label depends on the deployed draw (C2-03); next target differs by seed |
| WL96A | S1_NO_HEADROOM | 2 checkpoints; robust to debiasing, threshold and mask per C2-09 (not re-adjudicated here) |

## 4. Protocol integrity (MAP-01, MAP-09, MAP-10, C2-13, B-11, MAP-13)

### MAP-01: UPHELD_WITH_MODIFICATION

- `results/audit97/MAP/code_revision_check.json` confirms per-run identity, except:
  - WL86 (78 of 86 files) and WL86A (most of 99 files) differ from the stated commit only by CRLF (see `crlf_check.json`).
  - WL88, WL89 and WL93 have documented post-freeze files.
- Modification: what this shows is that the *server code tree at audit time* equals the stated commit. That excludes a wrong revision as an explanation. It does not by itself prove the tree was unchanged while the runs executed. That is supported only by the matching hash prefixes of outputs (MAP-04/13).

### MAP-09: UPHELD_WITH_MODIFICATION (material)

- Written rule (`experiments/ecro_cf92b/protocol/ecro_cf92b.json` l.37): "H_interact SUPPORTED if no single counterfactual is DOMINANT in both E2 cases but at least two have f >= 0.25".
- Implementation (`analyze92b.py:110`): checks only H_local (X_M12) and H_global (X_L3).
- `results/ecro_stage93/recon92b.json`: X_L2 and X_Q are DOMINANT in both E2 cases. Under the written rule H_interact is therefore **not SUPPORTED**. The rule has no else-clause, so "INCONCLUSIVE" or "NOT SUPPORTED" are both readings.
- With H_geometry the only SUPPORTED hypothesis, no clause of the written outcome mapping (l.57-62) applies:
  - C needs H_interact or more than one supported hypothesis.
  - A and B need H_local, H_global or H_source as the dominant mechanism.
  - D's parenthetical "every hypothesis INCONCLUSIVE / NOT SUPPORTED" is also false.
- **Outcome under the written rule: undefined; the code's fall-through gives D.**
- Substantive correction: the measurement itself (X_Q and X_L2 each individually dominant) argues for a *single-component* reading (source-side once-scattered stage / single-flight term). It does not argue for an "interaction". The WL92B narrative "source-side once-scattered response x tied factors" is supported only in its first half.
- The roadmap's "Outcome C by the frozen mapping" should be qualified.

### MAP-10 and C2-13: UPHELD (consistent with each other)

- The original stop (`results/gm1_relation89/results/orig_stop/chain.status`, 04:35:33Z, CHAIN_FAILED) was a strict-digest test failure.
- In `amend1/audit89.json`, the only mismatches are 2 `dbin` entries:
  - Both are invalid in both relations, with weight 0.
  - Their stale value comes from a re-traced lane whose first hit changed between T0 and T1.
  - The operator tensors and CPU outputs are bitwise equal.
- This is a non-semantic false positive.
- `git diff 7515a29 5d6bb5c`:
  - `analyze89.py` gained only a stopped-path branch (+14 lines).
  - `tests89.py` changed its predicate.
  - `audit89.py` was added; the protocol got its amendment entry.
  - No generator code, threshold or workload changed.
- Dependence: WL89's comparative quantities exist only because the amended gate let the chain resume. That is a **procedural** dependence, not a numerical one; the numbers would be bitwise identical under either predicate.
- Downstream numbers (from MAP):
  - WL90 uses pre-amendment artefacts plus `stage3_89.json`.
  - WL91 uses pre-amendment artefacts plus WL90's `F_W1_T1`.
  - WL96's design-stage Amdahl anchors (124.7 / 75.8 ms) come from WL91 `life91.json`.
  - WL92-WL96A use none of them.
- Human approval is asserted in protocol text (`amendments[0].approval`) and cannot be verified from artefacts.

### B-11: UPHELD

- All four items are disclosed in the worklogs or the WL93 reconciliation; for WL94A see WL94A:83,155.
- The sub-labels should not be cited as findings. The measurements stand.

### MAP-13: UPHELD (provenance; material for reproducibility only)

- `results/audit97/J7/p0_input_manifest.json` records the 18 P0 input identities with the sha256 prefixes that phase-1 MAP computed (17/18 match the prefix recorded in a worklog or result).
- `~/rest_x94d/results/audit94d/audit.json` (1d03d8432892f5ff) has no prior record to compare against.
- J7 did not re-hash and did not move anything. The prefixes are 16 hex digits; full digests are still absent.

## 5. Scope findings (MAP-05, C2-16, F0-03)

### MAP-05 and C2-16: UPHELD

The CK2 lineage rests on one geometry, one grazing point light, one camera, one reference set (four 8,192-spp renders, two 4,096-spp field halves) and two checkpoints. There is no across-scene uncertainty.

### F0-03: UPHELD_WITH_MODIFICATION

T0 -> T1 edit response measured, per batch (reference in brackets):

| batch | edit response measured |
|---|---|
| WL86 | yes: E1/E2 dH F3 and stale control [path-traced REF] |
| WL86A | E2 T1 attribution, static per state; no edit response |
| WL87 | yes: E1-E3 DYNAMIC_EDIT_VALIDITY, dH [path-traced] |
| WL88, WL89 | GM1 edits: state tests T_EDIT, edit validity [path-traced]; WL88 stopped at workload |
| WL90 | W1 T1 field; gap attribution at the edited state; no response metric |
| WL91 | held-out edits W2 T1 etc.: SOURCE / EDIT VALIDITY SUPPORTED [path-traced] |
| WL92 | E2 T0/T1 static per state |
| WL92A | yes: E4 region-C slope/corr, F3 on E1/E2/E4 [path-traced] |
| WL92B | yes: E5 slope 0.75-1.00, corr >= 0.96 [path-traced] |
| WL93B, WL93C | yes: E2/E5 slopes (inspected edits) |
| WL93D | yes: E2/E5/E6 T0 and T1 per arm |
| WL93D addendum, WL94, WL94A-D, WL95A | **none** (no T1 term in WL94-WL95A) |
| WL95, WL96 | design-only |
| WL96A | edit **timing only**: box A +- 0.15 m, "not a reference-validated geometry change" (WL96A:202) |

Modification: the last batch with a reference-validated edit response is WL93D's main run (inspected edits), and the last on a never-inspected held-out is WL92B (E5). Architecture-level labels after WL93D are static, single-scene statements. P0-4's lifecycle-cost half is valid as timing.

## 6. P0 verdicts within this cluster

### P0-1: CONDITIONALLY_VALID (data and statistics restriction only; the sender error itself is adjudicated by J1)

- What the claim can mean: "the learned S1 sender" means **the two trained OWN/ECRO checkpoints**, on CK2's grazing incidence regime. That regime is about 0.1 % of training lightings, never directly supervised, and the error has opposite sign between seeds. It is not a statement about the representation class or about learnability.
- "Avoidable" holds only against the explicit in-family M64 operator, which itself over-delivers.

### P0-5, first half ("justify rejecting the tested ECRO representation mechanisms"): CONDITIONALLY_VALID, narrowly

- Justified: rejecting **the frozen WL86 v0 operator as deployed, under its frozen F2 rule**. Even this is a near coin flip over relation-state draws: 9/16 pass for s8601, and s8602 was never resampled.
- Justified: rejecting **the fitted rank-16 tied local factor and the 64-bin patch carrier as instances on CK2**.
- Not justified:
  - Rejecting the mechanisms as classes.
  - Rejecting sender learnability. It was never tested at full scale, and every intervention after WL92A is a smoke non-result.
  - Rejecting "learned held-out sharp transport". The only qualified never-inspected held-out (E5) shows no failure, and there ECRO is not worse than explicit at T1.
- The founding premise "ECRO fails held-out sharp transport" reduces to: on E2 T1 region C, a paired deficit against same-state explicit transport (+0.017 +- 0.004, one checkpoint).

### P0-5, second half ("do not establish a general impossibility"): VERIFIED

Single workload, two checkpoints, edge-of-support incidence, smoke-scale learnability tests, and no edit response after WL93D.

## 7. What the chain WL92-WL94D legitimately establishes about the learned ECRO lineage

- WL92: with an exact local factor the interface carries enough information. The G_rep stop is rule mechanics on an unconverged fit (0.036 / 0.017 < 0.06), not evidence of non-representability.
- WL92A: one full-scale retraining (OWN). Ownership was recovered. E4 not admitted. Seed spread at E2 T1 is 0.077 vs 0.040.
- WL92B: X_Q and X_L2 are individually dominant on E2. The official C/"interaction" label is not supported by the written rule (outcome undefined, D by fall-through). E5 was admitted and all six learned models pass it.
- WL93-WL93D: stops and smoke labels. Only WL93D measures that seed effects (26-32 %) exceed geometry effects (4-14 %).
- WL94-WL94D: smoke fine-tunes (1,000-2,000 steps) on CK2 / static.
  - Labels: INCONCLUSIVE / STOP / NO_RELIABLE_IMPROVEMENT.
  - The material-generalisation and query-weighting effects are seed-dependent.
  - The WL94D teacher was biased.
  - an94.W mis-weighting (8x8 layout) affects WL94-WL94D sender energies (not adjudicated here).
- **Net:** the lineage establishes that the two trained checkpoints mis-deliver grazing sharp panel transport on CK2, with seed-signed error. It does not establish a representation or learnability limit, and it adds no edit-response evidence after WL92B/WL93D.

## 8. Affected worklog statements

| original claim | defect or limit | corrected interpretation | dependents | action |
|---|---|---|---|---|
| WL86: "NOT SUPPORTED", held-out sharp failure (F2, E2 T1 C) | one deployed draw; s8601 passes 9/16 fresh draws; misses are 0.27-0.93 noise bounds; F2 is T1-only | frozen outcome stands; it is a marginal, draw-dependent miss, plus a resolved paired deficit vs explicit at E2 T1 (s8601) | WL86A-WL96A premise "ECRO fails sharp held-out" | QUALIFY |
| WL87 / WL95:314 / WL96:150: "Explicit >= ECRO everywhere (0 better / 3 worse / 36 equal)" | 33 within noise, not 36; E5 T1 (WL92B) reverses sign within about 1 noise bound | explicit is never significantly worse on E1-E3 and is better at E2 T1; at E5 T1 learned <= explicit (single draw) | WL95 / WL96 adjudication premise (explicit dominance) | QUALIFY |
| WL92B / RESEARCH_ROADMAP: "Outcome C by the frozen mapping" (H_interact SUPPORTED) | implemented rule differs from the written rule | written rule: H_interact not supported; outcome undefined / D by fall-through; X_Q and X_L2 individually dominant | S1-sender lineage WL93-WL94D | QUALIFY |
| WL93D addendum -> WL94-WL96A: CK2 as the quantitative sharp workload | post hoc, reference-iterated, edge-of-support (grazing about 0.1 % of training lightings), no admission, no labels | all CK2 results are single-scene, out-of-support-incidence statements about the trained checkpoints | P0-1..P0-4 magnitudes | QUALIFY |
| WL95:313: WL94D / zero-operator level filed as weakening learned-sender evidence | smoke-scale non-results, biased teacher, five unseparated causes | not class evidence; the checkpoints are wrong as trained; learnability untested | WL96 NO_VIABLE_LEARNED_MECHANISM | QUALIFY |
| WL92: STOP_INTERVENTION_G_rep read as a representation limit | fit below the level was unconverged | stop is rule mechanics; it neither shows nor excludes representability | WL92A design | QUALIFY |
| WL89: STOPPED_INVALID_COMPARISON -> amendment | strict-digest false positive; procedural dependence only | WL89-WL91 numbers valid; human approval not verifiable from artefacts | WL90, WL91, WL96 anchors | PRESERVE |
| WL91: QUALITY-MATCHED ECONOMIC UTILITY NOT SUPPORTED | one heavy-tailed draw decides a max-type rule | label follows the rule; mean lifecycle 0.57-0.90x | WL96 cost reasoning (explicit comparator only) | QUALIFY |
| WL87 "E3 untouched" / WL92A E4 (frozen ECRO) | not margin-disjoint for the WL86 population | "radiance not inspected"; frozen-ECRO E3/E4 parity may be leakage-favoured | none on direction | QUALIFY |
| WL94A: forgetting CONTRADICTED | seed not named; s8601 only | seed-dependent (s8602 SUPPORTED); disclosed | none | PRESERVE (disclosed) |
| WL94-WL96A architecture labels implying dynamic reuse | no edit response measured after WL93D | static single-scene statements; WL96A edit is timing only | living documents | QUALIFY |

## 9. Unresolved

- s8602 relation-state resampling at E2 T1 was never done. Whether its F2 miss (0.27 nb) is systematic is unknown.
- State-draw uncertainty of the E5 learned-vs-explicit differences is unmeasured (single draw).
- Human approval of the WL89 amendment is asserted in text only.
- Full sha256 digests of the P0 inputs (16-hex prefixes only).

Status: COMPLETE.
