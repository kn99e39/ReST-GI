# Worklog — WL97: ReST-GI Historical Evidence Reliability Audit (2026-11-16)

## Origin and scope

- **Brief.** "WL97 — ReST-GI Historical Evidence Reliability Audit" (user), run with ultracode multi-agent orchestration.
- **Batch class.** An audit, not an architecture-development batch. It is a regular number because it independently re-examines the evidence of ten batches; it does not close the evidence gap of a single worklog.
- **Question.** Which recent negative findings are reproducible and methodologically valid, which are conditional or inconclusive, and which cannot support their stated architecture conclusions?
- **Primary scope.** WL92 / 92A / 92B, WL93–93D, WL94–94D, WL95, WL95A, WL96, WL96A.
- **Supporting scope.** WL86–WL91, where a primary conclusion relies on them.
- **Unchanged.**
  - The research direction (reusable learned light transport, geometry-edit adaptation without retraining).
  - ER and ownership contracts, every checkpoint, baseline, protocol, result and historical worklog.
  - No training, tuning, production-code change, new architecture or successor experiment.
- **Conflict of interest.** The coordinating session authored the audited worklogs. Agents were therefore instructed not to treat any worklog verdict as ground truth, and the synthesis keeps every reviewer report verbatim.
- **Where.**
  - Audit directory: `docs/audits/wl97_evidence_reliability/` (index: `README.md`).
  - Scripts: `experiments/audit97/`.
  - Outputs: `results/audit97/`.
  - Server work: LabServer63 `~/rest_x97/` (every historical run directory read-only).
  - Repository at audit start: `efcf86a`, clean tree.

## Process

1. **Phase 1: twelve independent tracks, run without cross-talk.**
   - Seven static tracks:
     - MAP (dependency map, executed revisions);
     - A (maths and physics contract);
     - B (data and protocol);
     - C1 / C2 (measurement and statistics);
     - D (systems and cost);
     - F0 (inference chain, formed before other findings).
   - Five independent reproductions, each with newly written measurement code:
     - E1: P0-1;
     - E2: P0-2 / P0-3 by relation sampling, 4,096 draws;
     - E3: P0-2 / P0-3 by deterministic area quadrature;
     - E4: P0-4 accuracy, with new tagged references and re-renders;
     - E5: P0-4 cost, re-timed alone on the GPU with N up to 11,430.
   - Output: 147 findings, 101 of them material.
2. **Phase 2a: seven adversarial adjudicators (J1–J7).**
   - One adjudicator per issue cluster, each instructed to refute every finding and to settle conflicts with one decisive check of its own.
   - Rulings: 102 in total. 47 UPHELD, 55 UPHELD_WITH_MODIFICATION, 0 REFUTED, 0 UNRESOLVED as rulings (open questions are listed separately). No finding was refuted outright, but many were narrowed; the modifications are recorded in the J reports.
   - 88 affected worklog statements were recorded: 5 RETRACT, 57 QUALIFY, 26 PRESERVE.
3. **Phase 2b (lens reviews L1 / L2 and completeness critic CC): not run as agents.** The owner decided this for budget reasons. The coordinator took them over in `COORD_synthesis_and_p0_ledger.md` §3 and §6, which is not independent.
4. **Interruptions.**
   - Two usage-limit stops and one session end interrupted agents mid-track.
   - Each interrupted track was resumed by a fresh agent of the same track. The new agent continued from its own track's artefacts plus an auto-extracted action log, and had no access to other tracks.
   - Models: phase 1 and J1 / J5 ran on Claude Fable 5.2; J2 / J3 / J4 / J6 / J7 ran on Claude Opus 5.5 (owner budget decision).

## A. Executive reliability verdict

**The experimental measurements of WL86–WL96A are reliable. Their architecture-level interpretations are not reliable as worded.**

- **Measurements.**
  - Every evidence run executed its stated revision, and the P0 inputs have their recorded identities.
  - Every P0 headline number was reproduced by independently written code, most of them by two different routes.
  - No result was fabricated, mis-attributed or produced by the wrong code.
- **Defects.** Four implementation or evaluation defects were found (§D). None reverses a stored verdict label. Two change a label or a stated number under a corrected rule:
  - WL94 H-A becomes SUPPORTED post hoc;
  - WL92B outcome C is undefined under its written rule.
- **Inference.** The chain repeatedly states conclusions wider than its tests:
  - one static scene, one grazing light, two checkpoints of one recipe;
  - no geometry-edit response measured after WL93D;
  - only an internal explicit null as comparator.
  - Five statements are retracted (§E); 57 need qualification.

**Is the chain fit for architecture decisions?**
- It is sufficient for decisions about the **specific tested instances**:
  - the frozen WL86 v0 operator;
  - the rank-16 tied S1 local operator as trained;
  - the solid-angle-mean 4×16 sender table at grazing CK2 incidence;
  - a resolvent-only learned tail on CK2.
- It is **not** sufficient to reject mechanism classes, learned sharp carriers in general, learnability of the sender, or dynamic reuse. Nor does it establish any general impossibility of reusable learned light transport.

## B. Claim-level audit matrix

The full restrictions are in `COORD_synthesis_and_p0_ledger.md` §2.

| claim | verdict |
|---|---|
| P0-1a learned S1 sender has large error | **VERIFIED**: CK2, two OWN checkpoints, seed-signed |
| P0-1b … avoidable | **UNTESTED_INFERENCE**: only within the 64-bin class at field level vs M64. A passive non-negative rank-16 operator exact on CK2 exists, but neither training to it nor the class optimum vs the reference was tested |
| P0-1c deployed operator may violate passivity | **VERIFIED** for the local factor pair (14 checkpoints, 10–16 % of S1 rows > 1). **CONDITIONALLY_VALID** as "deployed operator": energy on the CK2 input is sub-unity and no consequence was tested |
| P0-2 binned M64 sender bias not removed by relation sampling | **CONDITIONALLY_VALID** ("not removed by sampling": **VERIFIED** on CK2). About half is the solid-angle-mean convention; the sign is receiver-dependent; the bias vanishes at 8×32; single configuration |
| P0-3 exact direction plus hit point closes the CK2 error | **CONDITIONALLY_VALID**: expectation only. No average gain at the deployed single-draw budget |
| P0-4 accuracy: no headroom for a learned ≥ 3-event resolvent | **CONDITIONALLY_VALID**: oracle prefix, table space, regions read by ROI_T / DIFF, CK2 only |
| P0-4 cost: no lifecycle headroom | **CONDITIONALLY_VALID**. The frozen outcome is VERIFIED as measured. "Generator 0.92–0.93×" is an **IMPLEMENTATION_DEFECT** (corrected ≈ 0.97×). The scaling sentence is **UNTESTED_INFERENCE** and is retracted |
| P0-5a negative results justify rejecting the tested ECRO mechanisms | **CONDITIONALLY_VALID**: valid for the tested instances only |
| P0-5b … no general impossibility established | **VERIFIED** |
| WL96 label NO_VIABLE_LEARNED_MECHANISM | **UNTESTED_INFERENCE** at label level (design-only, class-level). **CONDITIONALLY_VALID** for its scoped text (coordinator ruling, not independent) |

## C. Independently reproduced evidence (selected numbers; original vs independent)

**P0-1 (E1).**
- The learned sender reconstructs the stored arrays exactly (difference 0.0).
- Sender vs M64(s16): rel L1 / energy / peak 1.041 / 0.433 / 0.038 and 1.118 / 0.891 / 0.129. Reproduced as 1.0409 / 0.4330 / 0.0378 and 1.1182 / 0.8912 / 0.1286.
- Sender vs an independent Monte-Carlo physical table: 1.0830 / 0.4204 / 0.0316 and 1.1912 / 0.8652 / 0.1079, against the original 1.083 / 0.420 / 0.032 and 1.190 / 0.864 / 0.108.
- Maximum directional albedo: 2.26 / 3.08, reproduced as 2.2637 / 3.0774.
- J1 extended the passivity check to the WL86 deployed checkpoints: 2.70 / 2.21.

**P0-2 / P0-3.**
- E3 (deterministic oracle, no relation sampling): the exact oracle agrees with the path-traced e2_p within noise. The binned table over-delivers deterministically.
- A (numpy quadrature, no path tracer):
  - binned q 1.417× / 0.429;
  - M64 1.490× / 0.491;
  - the original values were 1.422 / 0.433 and 1.495 / 0.500;
  - a flux-conserving projection on the same bins gives 1.21× / 0.24.
- E2 (4,096 independent draws): the pooled over-delivery of ≈ 1.5× is real and does not fall with sampling. Exact evaluation is unbiased in expectation, but a single draw at the deployed budget gives no average gain.

**P0-3 image level (J3).**
- Paired re-analysis of the stored renders: pooled A4 − pooled X1 = +0.0120 E_H, about 14 block-bootstrap SE. The 2 × 4,096-spp renders therefore resolve every arm difference of 0.003 or more.

**P0-4 accuracy (E4, C2, J4).**
- dI(XOD) 0.00065 was reproduced by two tracks and with new camera seeds.
- Event accounting was confirmed with kmax runs.
- The floor table carries a structured 0.010 rel-L2 readout error against new path-traced references. It comes from patch-spatial averaging at camera vertex 2 and cancels in the comparison.
- The sphere's e3_o flux is 0.954 / 0.977; WL96A did not report it.

**P0-4 cost (E5, D, J5).**
- R0 144.1 / 88.5 ms reproduced.
- U(ZERO) edit 1.022 → 1.009 and source 1.04 → 1.05 for N 1,270 → 11,430.
- A-B-A thread-pool blocks show that an OpenMP spin-wait causes 53–57 % of the LAM edit deficit.

**Commands.** Each report's reproducibility appendix gives exact commands. The decisive scripts are:
- `experiments/audit97/E1/e1_audit.py`
- `experiments/audit97/E2/e2_chain.py`
- `experiments/audit97/E3/quad97.py`
- `experiments/audit97/A/a3_ck2_delivery.py`
- `experiments/audit97/E4/e4_*.py`
- `experiments/audit97/E5/e5_time.py`
- `experiments/audit97/J1/`, `J3/`, `J4/j4_attr.py`, `J5/`, `J6/j6_decisive.py`, `J7/`

## D. Defects and confounds

**Implementation defects.**
1. **`an94.W` uses an 8×8 bin layout; the representation is 4×16.**
   - No gate, outcome or label of WL94–WL94D changes.
   - CK2 learned-sender energies are off by up to 0.38, and rel L1 by up to 0.23.
   - Energies near 1 can change side of 1.
   - WL92–WL93D and WL95A use the correct weights.
2. **The WL96A LAM edit timing is contaminated by a CPU dtype cast that wakes an OpenMP pool.** The corrected U(edit) is ≈ 0.97 instead of 0.92–0.93, i.e. parity.

**Evaluation defects.**
1. **`an94.eh` noise_bound** is an L1 noise floor, but it was used as the tolerance for paired differences (SE ≤ 6.5e-5).
   - Under a paired tolerance, only WL94 H-A flips: INCONCLUSIVE → SUPPORTED, post hoc, scoped to substituting the binned M64 sender.
   - WL95A's "below render resolution" sentences rest on the same misuse and are retracted.
2. **WL92B outcome C** came from code that checks a subset of its written rule. Under the written rule the outcome is undefined.
3. **WL94D "delivery to box A"** was computed on box A+B or on all receivers.

**Statistical weaknesses.**
- Two seeds of one recipe, with seed effects that exceed treatment effects and flip sign.
- Single deployed draws decide labels:
  - WL95A FLIGHT_ATTRIBUTION gives ANGULAR / MIXED / SAMPLING in 23 / 31 / 10 of 64 draw choices;
  - WL91 rule (a);
  - the WL86 F2 miss (s8601 passes 9 / 16 fresh draws).
- Stops were decided by rule mechanics or inside noise (WL92 G_rep, WL93 preflight, WL94B exposure).

**Dataset limitations.**
- CK2 is post hoc and hand-iterated, without admission labels, yet it is the sole quantitative workload of WL94–WL96A.
- Grazing ring-0 sharp transport occurs in 4 of 3,712 training lightings (0.11 %) and is never directly supervised.
- CK2 was not excluded from the OWN training population. This leakage would favour the learned model.

**Invalid inference and overgeneralization.**
- RANK16_DELIVERY read as capacity: it is allocation, judged against a biased comparator.
- "Zero-operator level" holds in flux-L1 only.
- M64 was narrated as "exact" or "physical".
- Static CK2 evidence carries dynamic-reuse conclusions; no edit was measured after WL93D.
- The WL96 label is class-level.
- The cost result is read beyond the resolvent-only role.
- Only the internal explicit null was used as comparator; Neural Full Recompute and Native Reuse are UNMEASURED.

## E. Dependency impact

**RETRACT.**
- R1: WL95A image-resolution sentences.
- R2: WL96 "cannot resolve 0.013".
- R3: WL96A l.194 scaling sentence.
- R4: WL96 cost critic "generator ⇒ guaranteed loss".
- R5: WL96 cost critic "60 % dirty ⇒ 1.8× ECRO".

**PRESERVE (measurement level).**
- The learned sharp sender of these checkpoints is far from physical at CK2.
- The local factor pair has non-passive rows.
- The solid-angle-mean 4×16 sender over-delivers grazing sharp light to box A, independently of sampling.
- Exact evaluation removes this bias in expectation.
- Explicit sweeps from a path-traced 2-event prefix reproduce the CK2 tail in table space.
- The WL96A frozen cost criterion was not met.

**QUALIFY.**
- Roadmap entries WL86, WL87, WL92, WL93D, WL94, WL94D, WL95, WL95A, WL96, WL96A.
- `docs/Architecture.md:13`.
- Proposed wordings: `PROPOSED_LIVING_DOCUMENT_ANNOTATIONS.md`.

**REVALIDATE before any further use as a premise.**
- Any class-level exclusion of learned sharp carriers.
- Any cost-headroom statement beyond the resolvent-only role.
- Any held-out sharp-failure generalization beyond E2.
- Any dynamic-reuse statement drawn from WL93D-addendum-to-WL96A evidence.

**Upstream chain.** No downstream conclusion depends on a quantity invalidated by a defect, apart from the items above:
- WL89's stop was a false positive with bitwise-equal operators, so WL90 / WL91 / WL96 anchors are procedurally, not numerically, affected.
- WL92B X_Q and WL93C ORACLE_FEASIBLE use path-traced quantities, not M64.

## F. Scientific interpretation

- **ECRO.** The frozen v0 operator missed its held-out sharp gate on one region class of one edit. That miss is marginal and draw-dependent, but there is a resolved paired deficit against explicit transport at E2 T1. Every later batch measures *trained checkpoints at an out-of-support grazing incidence*, not a representation class.
- **Sharp transport representation.** On CK2, a solid-angle-mean 4×16 outgoing table over-delivers grazing sharp light. Half of that magnitude is the averaging convention. The sign depends on the receiver. The effect disappears at 8×32.
  - Exact directional evaluation removes the bias in expectation but needs about 16× more relation draws to help a single estimate.
  - None of this tests whether a *learned* carrier with another basis, conditioning or training could carry sharp structure.
- **Learned higher-order transport.** On CK2, given a path-traced 2-event prefix, there is no accuracy or cost headroom for a resolvent-only learned tail over three explicit sweeps, in table space and at this prototype's overheads. Untested beyond that:
  - learned operators that own the prefix, q table or relation update;
  - output parameterizations other than patch × bin;
  - other scenes, lights, depths or comparators.
- **Dynamic geometry reuse.** No reference-validated edit response was measured from the WL93D addendum through WL96A. These batches carry no evidence about reuse under edits.
- **Research direction.** WL86–WL96A neither support nor refute reusable learned light transport in general. They reject specific instances at specific scope.

## G. Unresolved questions (smallest missing evidence)

- **P0-1b.** The delivery-optimal patch × 64-bin operator against the path-traced reference on CK2 (no training needed).
- **P0-1c.** A defined passivity criterion for the composed operator.
- **P0-2.** One non-grazing light and one second receiver with the deterministic oracle.
- **P0-3.** A single-draw exact-sender render and the cost of a low-variance estimator.
- **P0-4.**
  - A second workload with a T1 reference.
  - An output parameterization finer than patch × bin.
  - Depth J > 4 or multi-light timing.
- **P0-5.**
  - Any reference-validated edit after WL93D.
  - Any declared external-baseline comparison.
  - s8602 state-draw resampling at E2 T1.
- Full sha256 digests of the P0 inputs; only 16-hex prefixes are recorded.

None of these is authorised by this audit.

## H. Reproducibility

- **Repository.** `efcf86a` at the start; audit files committed in the WL97 commit (this worklog). No existing file was modified apart from the dated roadmap audit record.
- **Executed-revision checks.** `results/audit97/MAP/` (code_revision_check.json, crlf_check.json, dependency_map.json).
- **Input manifest.** `results/audit97/J7/p0_input_manifest.json`: 18 artefacts, 16-hex sha256 prefixes, 17 matching prior records. Selected entries:
  - own_s8601 739e7eb50b25d329; own_s8602 cee9a09a289fca8c;
  - ecro_s8601 9d332cd61a7118bd; ecro_s8602 4d993121e41dd0bb;
  - CKF_none 3a6238a8a42d1db4.
- **Environment.**
  - LabServer63: RTX 3080 Ti 12 GB, Mitsuba cuda_ad_rgb, OptiX libs, `~/rest_x95a/env.sh`.
  - Local: Python 3.11 with numpy only.
  - Seeds are listed per track in each report's appendix; audit seeds are 97xxx.
- **Reviewer reports.** Verbatim in `docs/audits/wl97_evidence_reliability/`. Structured returns: `results/audit97/phase1_results.json` and `phase2_results.json`.
- **Not committed (size, duplicate).** `results/audit97/C2/CKF_none.npz`, a byte copy of `~/rest_x94/results/fields/CKF_none.npz` (sha prefix 3a6238a8a42d1db4).
- **Deviations from the brief.**
  - The lens reviews and the completeness critic were not run as independent agents (owner budget decision). The coordinator's substitute is non-independent.
  - Some agents were resumed after interruptions, as described under Process.
  - Two models were used.
