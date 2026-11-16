# WL97 — Coordinator synthesis and P0 evidence ledger

**Author.** The coordinating session (the same session that authored WL86–WL96A, so it has a conflict of interest).

**Status.** This file is a reconciliation of the independent reports. It is **not** an independent review.

- Every verdict below cites the phase-1 report that measured the quantity and the phase-2 adjudication that ruled on it.
- Where the coordinator ruled alone (§3), this is marked.
- Where reviewers disagreed, the dissent is kept (§5).

**Inputs.**
- Phase-1 reports: `MAP_*.md`, `A_*.md`, `B_*.md`, `C1_*.md`, `C2_*.md`, `D_*.md`, `F0_*.md`, `E1_*.md` … `E5_*.md` (this directory). Structured findings: `results/audit97/phase1_results.json`.
- Phase-2 adjudications: `J1_*.md` … `J7_*.md`. Structured rulings: `results/audit97/phase2_results.json`.
- Viewer: `python experiments/audit97/_tools/show.py <finding ids> | --track <T> | --index`.

---

## 1. Executive reliability verdict

**Measurement layer: reliable.**
- Every evidence run executed its stated revision. The only exceptions are CRLF-only differences and documented post-freeze files (MAP-01/02; J7).
- The P0 input artefacts have the recorded identities (MAP-04; J7 manifest).
- Every P0 headline number was reproduced by independently written code, most of them twice by different routes:
  - E1, E2, E3 (deterministic), A (numpy, no path tracer), E4, E5, C1, C2.
- No fabricated, mis-attributed or wrong-code result was found.

**Metric / evaluation layer: four implementation or evaluation defects.**
- None of them reverses a frozen verdict label as stored.
- Two would change a label or a stated number under a corrected rule:
  - WL94 H-A becomes SUPPORTED post hoc (J6).
  - WL92B's outcome C is undefined under its written rule (J7).

**Inference layer: not reliable as currently worded.**
- The chain repeatedly states conclusions at a wider scope than it tested:
  - one static scene, one grazing light, two checkpoints of one recipe;
  - no geometry-edit response after WL93D;
  - an internal explicit null as the only comparator.
- Five statements must be retracted (§4).
- 57 must be qualified. The full list is in the J reports, under "affected statements" (88 statements in total: 5 RETRACT, 57 QUALIFY, 26 PRESERVE).

**Can the chain support architecture decision-making?**
- Yes, for decisions about the *specific tested instances*. The concrete instances are listed under P0-5 in the matrix below.
- No, as the basis for rejecting *mechanism classes* or "learned mechanisms" in general. WL96's label NO_VIABLE_LEARNED_MECHANISM is an untested inference at the level of its label. Its own text is scoped (§3).

## 2. Claim-level audit matrix (P0)

| claim | verdict | binding restrictions | decided by |
|---|---|---|---|
| **P0-1a** learned S1 sender has *large* error | **VERIFIED** | two WL92A OWN checkpoints; CK2, one grazing light, static. Field rel L1 1.04 / 1.12 vs M64(s16) and 1.08 / 1.19 vs physical q; box-A delivery 0.52× / 1.63×, seed-signed. In-distribution (non-grazing) error ≈ 0.8 rel L1 at roughly correct energy | E1 (reproduced to rounding), C1, C2, J1 |
| **P0-1b** … *avoidable* | **UNTESTED_INFERENCE** | "Avoidable" exists only *inside the 64-bin class at field level vs M64*: an admissible passive non-negative rank-16 operator is exact on CK2 (J1). It was not shown by learning (WL94D smoke). At delivery vs the path-traced reference, s8601 is not worse than M64 (0.569 vs 0.546). The class optimum vs the reference was never computed | J1, E1-F6/F7, J6, J7 |
| **P0-1c** deployed learned operator *may violate passivity* | **VERIFIED** (local factor pair) / **CONDITIONALLY_VALID** (as "the deployed operator") | worst-row directional albedo > 1 on 10–16 % of S1 rows in 14 checkpoints (WL86 deployed 2.70 / 2.21; OWN 2.26 / 3.08; SUM; WL94-series). Energy on the CK2 input is sub-unity; the operator is applied in fixed depth, never iterated; no causal or stability consequence was tested; passivity of the composed operator is undefined | E1, A-06, B-05, C1-09, J1 |
| **P0-2** M64 / 64-bin sender has systematic sharp-delivery bias not removed by relation sampling | **CONDITIONALLY_VALID**. The sub-claim "not removed by relation sampling" is **VERIFIED** on CK2 | patch × 4×16 table with the **solid-angle-mean** bin convention; CK2, light 5 cm in front of the panel (cos_i 0.025–0.15), Lambert receiver box A, 2-event S1-mediated term. +40–50 % expected over-delivery (three deterministic oracles plus 4,096 draws). **About half** of the magnitude is the convention's over-emission at grazing incidence (+30 % panel flux); a flux-conserving table on the same bins gives ≈ +20 %. The **sign is receiver-dependent** (glint receivers under-delivered 5–8×). The bias essentially vanishes at 8×32 bins. Other scenes untested | A-03/04, E2, E3, C2-02, J2, J3 |
| **P0-3** exact direction + hit-point evaluation substantially closes the CK2 delivery error | **CONDITIONALLY_VALID** | **expectation only** (64 pooled draws, or deterministic quadrature): ratio 1.02, abs rel L1 0.11 vs n0 0.059. ROI E_H 0.0245 → 0.0124 (floor 0.0118), resolved by a paired statistic (≈ 14 block-bootstrap SE). Direction carries ≈ 70 % of the closure; the hit point is second order (≈ 0.08). **No average gain from a single draw at the deployed budget** (heavy tail; ≥ 16× more draws needed). Other receivers untested; no deployable low-variance exact estimator costed | E2, E3, A-05, J3 |
| **P0-4 (accuracy)** no meaningful accuracy headroom for a learned ≥ 3-event resolvent over explicit recurrence on CK2 | **CONDITIONALLY_VALID** | (1) CK2 static, one light, one camera; N 1,270 × 64. (2) **Oracle prefix**: this is a ceiling argument, since no resolvent was fitted or trained. (3) **Table space**: the floor table itself has a ≈ 0.010 rel-L2 structured readout error vs path-traced truth (patch spatial averaging at camera vertex 2, seen through the panel reflection); it cancels in the comparison. Another output parameterization could gain at most ≈ 0.002–0.004 of the ROI, and that gain is not learning-specific. (4) Only what ROI_T / DIFF read: unread receivers have real explicit errors (sphere e3_o flux 0.954 / 0.977, unreported; S1 tables 0.02–0.04). (5) 3–5 events; the 6th order (7.8 % of e3 flux) is absent everywhere. (6) It does not meet WL96 §F's own two-workload confirmation rule | E4 (reproduced dI 0.00065), C2-09, J4 |
| **P0-4 (cost)** … no lifecycle-cost headroom | **CONDITIONALLY_VALID**, with one **IMPLEMENTATION_DEFECT** component and one **UNTESTED_INFERENCE** sentence | The frozen criterion outcome (U < 1.25) is **VERIFIED as measured**. The zero-cost ceiling is 1.02 edit / 1.04 source (≤ 1.13 without frame). It holds only for a **resolvent-only** element that still pays for the q table, the first order, the relation update and the 512² × 64-spp frame, in this prototype's wall time, at J = 4, one light. Audit runs extend N to 11,430 with no approach to 1.25. "Generator edits 0.92–0.93×" is an OpenMP spin-wait artefact (53–57 % of the deficit); the corrected value ≈ 0.97× is parity. "No scaling hypothesis reaches 1.25×" misquotes scale91 | D, E5, C2-10, J5 |
| **P0-5a** the negative results justify rejecting the tested ECRO mechanisms | **CONDITIONALLY_VALID** | Justified only for the **instances** tested: (a) the frozen WL86 v0 operator under its frozen F2 rule. This miss is marginal and draw-dependent (s8601 passes 9 / 16 fresh state draws; s8602 was never resampled), backed by a resolved paired deficit vs explicit at E2 T1. (b) The geometry-conditioned rank-16 tied S1 local operator *as trained* at the tested budgets. (c) The solid-angle-mean 4×16 sender table at grazing CK2 incidence. **Not** justified for mechanism classes, for sender learnability (never tested beyond 1–2 k-step smokes), for "held-out sharp transport" in general (E5, the only qualified never-inspected held-out, shows learned ≤ explicit at T1), or for dynamic reuse (not measured after WL93D) | J1, J6, J7, F0, coordinator §3 |
| **P0-5b** … but do not establish a general impossibility for reusable learned light transport | **VERIFIED** | No WL86–WL96A measurement bears on learned transport outside these instances, the CK2 / scene86 scope and the explicit-null comparison | J6, J7, F0, coordinator §3 |

## 3. Coordinator rulings on findings no adjudicator was assigned (NOT independent)

The phase-2 lens reviews (L1 false-rejection, L2 false-acceptance, CC completeness critic) were **not run as agents**. The owner decided this on 2026-11-16 to save budget, and the coordinator took them over. Three material F0 findings had been routed only to those reviews. The coordinator checked each against the cited text.

**F0-04: UPHELD_WITH_MODIFICATION.**
- WL96's verdict *text* is scoped: "the learned mechanisms that could be specified in this study"; scene86 / GM1, point lights, rigid edits, α 0.03–0.18; provisional; dissent recorded (WL96:30–37).
- The *label* NO_VIABLE_LEARNED_MECHANISM and the roadmap entry ("Learned sharp carriers are excluded or dominated", filed under a "measured" evidence record with scope "rigid edits") read as a class-level, measured result. But WL96 was design-only, trained nothing, evaluated no edit, and its capacity argument covers only linear, light-independent, separable forms (critic_physics.md).
- Corrected reading: no candidate specified in WL96 placed sharp structure in a learned component, and the one remaining learned element (the broad resolvent) was predicted to have little utility. WL96A then measured that utility on CK2 (P0-4). Class-level exclusion of learned sharp carriers is an argument, not a tested result.
- Verdict: **UNTESTED_INFERENCE at label level; CONDITIONALLY_VALID for the scoped text.**

**F0-07: UPHELD.**
- `docs/BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md` has no role entry for EXP / C3UR. Their comparator role exists only in the 2026-11-13 human decision (Architecture.md:13).
- The chain respects the terminology gate in wording. But the 1.25× utility rule, "decorative" and "no viable learned mechanism" judge learning solely against explicit recomputation in a 1,270-patch room.
- Neural Full Recompute and Native Reuse are UNMEASURED throughout WL86–WL96A.
- WL95 / WL96 classify every design that fixes sharp delivery as "G1 class, comparator under current canon". The WL96 negative is therefore produced jointly by evidence and by that admissibility classification (owner Gate A, which this audit does not adjudicate).

**F0-16: UPHELD.**
- The statements concerned are Architecture.md:13 ("sharp/held-out fidelity failures", plural and general) and the roadmap WL86 / WL87 / WL92 / WL95A / WL96 / WL96A entries.
- Specific annotations are proposed in `PROPOSED_LIVING_DOCUMENT_ANNOTATIONS.md`.
- Two points were refined by the J reports:
  - J7: the WL86 F2 miss is a correct frozen-rule outcome on the deployed draw, but draw-dependent.
  - J7: E5 T1 learned ≤ explicit is ≈ 1 noise bound on single deployed draws, so "opposite sign" is at the noise level.

## 4. Statements to RETRACT

All five were adjudicated in phase 2.

| # | worklog statement | why | dependents |
|---|---|---|---|
| R1 | WL95A:133–138, 166, 175, 178: image-level effects are "below the 2 × 4,096 spp render resolution"; renders cannot rank the components | That tolerance (0.0167) is an L1 noise floor, not an uncertainty of a paired difference. The paired SE is ≤ 2e-4–1.7e-3, so every difference ≥ 0.003 is resolved (J3, C2-07). The frozen labels stay valid as rule outputs; NEXT_DESIGN_TARGET stays UNDETERMINED because the seeds disagree | WL96:69, 161, 216; RANK16 criterion (iv) |
| R2 | WL96:161, 216: "2 × 4,096 spp cannot resolve a 0.013 effect" | Resolved by > 10 bootstrap SE (J3) | WL96 §F smoke design (WL96A already used a paired statistic) |
| R3 | WL96A:194: "tail orders grow slower than the q table and relation build, so no scaling hypothesis reaches 1.25×" | scale91: orders 1.70×, q 1.67×, relation update 1.66×. They grow alike; only the full relation build (in no lifecycle event) grows faster (J5, D-09). Audit runs to N = 11,430 show no approach to 1.25×, but depth, multiple lights and work-basis conventions are untested | none (the extension decision is unaffected) |
| R4 | WL96:149, critic_cost.md §2.2: "with an ECRO-class generator (~11 ms) the per-edit ratio is ≤ 1, a guaranteed loss" | Within the critic's own range the ratio is 0.96–1.05 (parity). The 11 ms is WL86's whole respond stage, including a CPU cast (J5, D-06). The same critic also has a source-ceiling slip (1.16 / 1.98 used the edit event's sweep time) | the cost half of WL96's "predicted decorative"; survives only as "no demonstrated gain" |
| R5 | WL96:114, 153, critic_cost.md §1.3: "0.3 m box-A shift dirties 60 % (~5.5 M re-tests, ~60 ms), ~136–140 ms per edit, ~1.8× ECRO" | Only 10–22 % of the pairs need a new visibility decision, which is within designer d2's estimate. The per-edit cost is unmeasured; at the critic's own rate it would be 1.1–1.3× (J5, D-07) | the "explicit carriers carry real costs the designers underestimated" item. It concerns the explicit comparator only |

## 5. Defects (separated by kind)

**Implementation defects.**
- **an94.W uses an 8 × 8 bin layout; the representation is 4 × 16** (A-02, C1-01, C2-15, E1-F9, MAP-08, J6).
  - No gate, outcome or verdict label of WL94–WL94D changes.
  - The WL94D SENDER_IMP rule stays false on both seeds under true weights.
  - CK2 learned-sender *energies* change by up to 0.38 (e.g. T s8601 1.749 → 1.370), and rel L1 by up to 0.23. Energies near 1 change side (own s8602 1.12 → 0.89; R s8601 1.26 → 0.96; P s8601 1.01 → 0.77). None of these sign changes was narrated as a conclusion.
  - WL95A uses the correct W_T, as do WL92–WL93D.
- **WL96A LAM edit timing.** A GeoState int64 cast on the CPU wakes an OpenMP pool, which slows the next single-threaded q-table trace by 5.6–9 ms. This accounts for 53–57 % of the 7–8 % deficit; corrected U(edit) ≈ 0.97 (D-02, E5-3, J5).

**Evaluation defects.**
- **an94.eh `noise_bound`** is an expected-L1 noise floor. It was used as the tolerance for *paired differences* whose SE is ≤ 6.5e-5 (C1-02, C2-07, J6, J3).
  - Re-applying every frozen rule with a paired tolerance max(2 SE, 0.005) changes exactly one decision: **WL94 H-A, INCONCLUSIVE → SUPPORTED (post hoc)**. SSR s8601 diff 0.016216; disjoint halves 0.01603 / 0.01604; share 0.343.
  - This strengthens WL94's own reading that the residual sits in panel-delivered input.
  - WL94A / C / D, WL95A NDT and WL92A HELDOUT_SHARP are unchanged.
- **WL92B outcome C** came from `analyze92b.py:110`, which checks only X_M12 / X_L3 (MAP-09, J7).
  - Under the written mapping H_interact is not supported; H_geometry is the only SUPPORTED hypothesis, and no clause A–D applies, so the outcome is undefined.
  - The measurements favour a single-component (source-side once-scattered) reading.
- **WL94D "CK2 delivery to box A"** was computed on box A + B or on all receivers (C1-04); stated as a qualification in J6.

**Statistical weaknesses.**
- Two seeds of one recipe throughout. Seed effects exceed treatment effects and flip sign: box-A delivery 0.52× vs 1.63× (C1-03, F0-14, J7).
- Single deployed draws decide labels:
  - WL95A FLIGHT_ATTRIBUTION is ANGULAR in 23, MIXED in 31 and SAMPLING_DOMINANT in 10 of 64 leave-one-out "deployed" choices (J3).
  - WL91 utility rule (a).
  - The WL86 F2 miss (draw-dependent).
- Stops decided by rule mechanics or inside noise: WL92 G_rep (convergence flag), WL93 preflight (0.508 vs 0.485 inside the q noise 0.145), WL94B exposure (1.232 vs 1.25) (C1-08, J6).
- The WL96A n_img debias uses whole-e3 noise instead of e3_o noise; this is immaterial (7× margin) (E4-F4).

**Dataset limitations.**
- CK2 is post hoc and hand-iterated (five probes, corner at −1.345 vs the −1.35 rule). It has no admission labels, yet it became the **sole** quantitative workload of WL94–WL96A (B-10, MAP-05, C2-16, J7).
- CK2's grazing ring-0 sharp transport occurs in 4 of 3,712 training lightings (0.11 %) and is never directly supervised. Non-grazing panel-to-box sharp transport is common (47 %) (B-04, J7).
- CK2 was not excluded from the OWN checkpoints' training population (B-02). This leakage would favour the learned model, so it does not weaken the failure evidence; it does invalidate "held-out" wording for CK2.

**Invalid inference or overgeneralization.**
- RANK16_DELIVERY "INADEQUATE" read as capacity or as physical-target inadequacy (J1). The rule compared against M64, which is 1.54× the reference; rank-32 A3 "fails" while closer to the reference than M64. The shortfall is fit-distribution allocation, not capacity.
- WL95's learned S1 operator "at zero-operator level" holds only in flux-L1. In Frobenius norm it is 0.77–0.83 (J6).
- M64 narrated as "exact sender" or "physical control" (WL93D, WL94, WL94D). M64 is biased at grazing incidence (J2, J6). WL92B X_Q and WL93C ORACLE_FEASIBLE are *not* affected; they use path-traced q and references.
- Static-scene evidence (WL93D addendum → WL96A) carrying dynamic-reuse conclusions (F0-03, J7).
- WL96 label scope (§3).
- Cost "nothing to amortise" read beyond the resolvent-only role (J5, D-04).

**Provenance gaps.**
- P0 inputs exist only on LabServer63, and only 16-hex sha prefixes were recorded. A manifest is now in `results/audit97/J7/p0_input_manifest.json`.
- The WL89 amendment's human approval is verifiable only from the protocol text. The WL89 stop was a strict-digest false positive; the operators and outputs are bitwise equal, and downstream use (WL90, WL91, WL96 Amdahl anchors) is procedural, not numerical (J7).

## 6. Special checks required by the brief (coverage, coordinator completeness pass)

| check | status | where (independent recomputation unless noted) |
|---|---|---|
| an94.W 8×8 vs 4×16 | COVERED | A-02, C1-01, C2-15, E1-F9; decisive recompute J6 |
| Box A vs Box A + B mask | COVERED | C1-04 (recompute); J6 |
| WL94D unmatched Frobenius / L1 | COVERED | C1-05 (matched-norm recompute); J6 |
| rank-16 ring-0 counterexample | COVERED | C2-05, B-06, E1-F6; J1 constructive operator (`results/audit97/J1/j1_rank16_*.json`) |
| source points / fixed random patterns | COVERED | E1-F3 (32 source and 6 state redraws), MAP-06, A-09, C2-03; J3 leave-one-out label |
| M64 teacher vs exact directional response | COVERED | A-03 / 04, E3, E2-F8, C1-06; J2 |
| X1 direction and hit-point changes | COVERED | E2 (4,096 draws), E3 (deterministic), A-05; J3 |
| e2 / e3 counting, e3_p / e3_o tagging | COVERED | A-08 (code), E4-F1 (kmax runs); J4 camera-side |
| e3_o containing earlier sharp interactions | COVERED | E4-F5: box-A e3_o has an earlier panel vertex 0.038, floor-last 0.231, sphere-last 0.006 |
| reference noise and significance thresholds | COVERED | C1-02, C2-07, E2-F10/11, E3-08; J3, J6 paired re-application |
| WL89 invalid comparison, downstream | COVERED | MAP-10, C2-13; J7 |
| WL96 / WL96A cost ceilings and timing boundaries | COVERED | D, E5 (re-timed, A-B-A blocks, N to 11,430); J5 |
| timing-only edits vs dynamic-GI references | COVERED (negative) | D-13, E5-7, F0-03; J7: no T1 reference exists for CK2; post-WL93D labels are static |

**Coordinator spot-checks** (2026-11-16), each verified against the stored file:
- WL94 SSR s8601 diff 0.016216 and noise_bound 0.008363 (`results/ecro_mat94/stageB.json`), matching J6's `j6_decisive.json`.
- J4 floor readout (`results/audit97/J4/j4_attr.txt`): T2 + T3 vs stored REF h, debiased rel L2 0.00996.
- WL96A recorded walls R0 144.13 / 88.53 ms and U(LAM24 edit) 0.932 (`results/audit97/J5/j5_check.json`).
- WL86 F2 E2 T1 region C: 0.0717 / 0.0634, with noise bound 0.0126 (`results/audit97/J7/founding_failure_check.json`).
- J3 pooled X1 paired dE_H vs floor 0.000596, block-bootstrap SE 0.00022 (`results/audit97/J3/j3_img_paired.json`).

Every one matches. This is a limited completeness pass, not the independent CC review the original design specified.

## 7. Dependency impact (which conclusions to preserve, qualify, revalidate)

**PRESERVE (measurement-level, reproduced).**
- The learned sharp-panel sender of the trained checkpoints is far from the physical response at CK2.
- The learned local factor pair has rows with directional albedo > 1.
- A solid-angle-mean 4×16 sender table over-delivers grazing sharp light to the CK2 box A, independent of relation sampling.
- Exact direction (plus hit point) removes that bias in expectation.
- Three explicit 64-bin sweeps from a path-traced 2-event prefix reproduce the CK2 ≥ 3-event tail in table space, within the reference noise, for the regions the ROI reads.
- The WL96A frozen cost criterion was not met at the tested conventions.

**QUALIFY (scope wording).**
- WL86 / WL87 roadmap entries: the draw-dependence of F2; E5.
- WL92: rank-16 norm.
- WL93D / WL94 / WL94D: M64 as "exact" or "physical"; an94.W energies.
- WL94: H-A SUPPORTED post hoc under a paired statistic.
- WL95: zero-operator level in flux-L1 only.
- WL95A: RANK16 and FLIGHT_ATTRIBUTION readings; image-resolution sentences (R1).
- WL96: label scope; R2, R4, R5.
- WL96A: the oracle-prefix and table-space ceiling; the sphere and S1 receivers; R3; the corrected generator U.
- Architecture.md:13 wording.

**REVALIDATE before any further use as a premise.**
- Any statement that a *class* of learned sharp carriers is excluded.
- Any statement that learned transport has no cost headroom beyond the resolvent-only role.
- Any held-out sharp-failure generalization beyond E2.
- Any statement about dynamic geometry reuse from WL93D-addendum-to-WL96A evidence.

**RETRACT.** R1–R5 (§4).

## 8. Smallest missing evidence (per claim)

- **P0-1b.** The delivery-optimal patch × 64-bin operator against the path-traced reference on CK2 (the class floor); this needs no training.
- **P0-1c.** A defined passivity criterion for the *composed* ECRO operator and its value.
- **P0-2.** The same deterministic oracle on one non-grazing light and one second receiver.
- **P0-3.** One single-draw (deployed-budget) exact-sender render, and the cost of a low-variance exact estimator.
- **P0-4.** One second workload with a T1 reference; one output parameterization finer than patch × bin read at the camera; one depth J > 4 or multi-light timing.
- **P0-5.** Any reference-validated geometry-edit measurement after WL93D; any comparison against the project's declared external baselines (Neural Full Recompute / Native Reuse are UNMEASURED); s8602 state-draw resampling at E2 T1.

## 9. Process record and independence

| phase | agents | model | note |
|---|---|---|---|
| 1 | MAP, A, B, C1, C2, D, F0, E1–E5 | Claude Fable 5.2 | first attempt interrupted by a usage limit; each track resumed from its own artefacts plus an auto-extracted action log (`results/audit97/<T>/_interrupted_run_log.md`); no cross-track access |
| 2a | J1, J5 | Claude Fable 5.2 | completed |
| 2a | J4 | Claude Opus 5.5 | interrupted when the session ended; resumed from its own artefacts |
| 2a | J2, J3, J6, J7 | Claude Opus 5.5 | owner budget decision |
| 2b | L1, L2, CC | **not run** | owner budget decision; the coordinator took over (§3, §6). The false-rejection lens is partly covered by phase-1 F0 (independent); the false-acceptance lens is covered only by the adjudicators' instruction to refute findings, and by this non-independent synthesis |

All reviewer reports in this directory are kept verbatim as their agents wrote them.
