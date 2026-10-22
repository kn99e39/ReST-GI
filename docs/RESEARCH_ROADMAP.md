# Research Roadmap — Surface-Owned Selective Neural Transport Recomputation

**Updated:** 2026-10-22 (Batch 5 SPT-1 authorised and contract recorded; worklog 38 latest reviewed)  
**Branch:** Selective_Recompute  
**Status:** Architecture Batches 1–3 reported. Batch 3 (native RNA Lego, worklog 36): independent owned units pass static screening (Case A) with exact ownership and unit reconstruction; Batch 4 (worklog 37, physical Lego edit, oracle D_T): **CONDITIONAL** — D* = 8/17 units (7 stationary receivers + the moved box), selective recompute recovers R = 0.98 of the changed-region error, V* bit-identical and valid (1.03× full), output change exactly on the predicted support; whole-image −0.31 dB vs full recompute misses the −0.26 dB criterion at seed 0. The seed replicate (worklog 38) is WITHIN_MARGIN: mean −0.08 dB over 3 seeds, and the unit-3 effect is not systematic  
**Venue planning:** SIGGRAPH 2027, official paper deadline to be separately verified.

## 0. Decision Record

- **2026-10-12:** the user clarified that "selective" means recomputing complete owned models, never intra-model parameter subsets. The archived object-level-only framing is at docs/history/ARCHITECTURE_OBJECT_LEVEL_SELECTION_2026-10-12.md.
- **2026-10-13:** the user adopted ten decisions, listed in full in docs/RESEARCH_CENTRIC_TOPIC.md §2:
  - RNA is the initial baseline;
  - ownership may be finer than objects;
  - Surface-Owned Transport Units are the active hypothesis;
  - each invalidated unit's full state may be recomputed;
  - object-level recompute is the coarse comparison;
  - at least five architecture batches come before a broad viability judgment;
  - a single negative result does not falsify the hypothesis;
  - 2007–2013 dynamic GI is the priority design reference;
  - the 1997 line-space hierarchy is not the starting point;
  - efficiency and scaling come after feasibility.

- **2026-10-19 (long-horizon design direction, not implementation approval):** treat interreflection, including nonlocal/multi-bounce dependencies, as a future representation/selector requirement. Prefer an algorithmically constructed, interpretable spatio-directional candidate hierarchy with explicit position and directional bounds over learning an entire scene hierarchy from scratch. Investigate learned relation scoring/dirty-unit selection only after validating candidate coverage and against algorithm-only controls. The specific tree, features and model remain hypotheses. The user reports Batch 4 already running; do not alter its protocol.
- **2026-10-22 (Batch 5, user brief):** design and implement a factorized Spatial-Perspective Neural Transport representation: persistent learned state + configuration-dependent transport state + reusable operator, with explicit update semantics and a held-out physical edit. This is not another prerequisite diagnostic. The Surface-Owned contract stays the historical baseline. The candidate SPT-1 contract is in docs/Architecture.md §9a (worklog 39).
- **2026-10-19 (documentation policy):** record substantive user-approved research-direction decisions in the relevant living documents immediately, without waiting for a Worklog or batch close. Preserve historical records and label unvalidated proposals explicitly.

Active contracts: docs/RESEARCH_CENTRIC_TOPIC.md, docs/Architecture.md, docs/BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md.

## 1. Central Intent

> Can learned neural transport be owned by independently reconstructible surface units such that, after a geometry edit, recomputing the complete state of only the transport-invalid units restores high-quality full-scene GI while all valid units are reused unchanged?

## 2. Evidence So Far

| Worklogs | Established | Not established |
|---|---|---|
| 21–25 | stale nonlocal GI at stationary receivers under a part edit; indirect occlusion and moved-part reflection dominate | that every dynamic neural GI method fails |
| 26–27 | physical and historical whole-asset neural recompute are costly (different regimes) | a quality-matched unit-level cost |
| 28–29 | two sparse K=32 relational-state candidates fail held-out near-mirror T3 | that owned-unit recompute fails |
| 30 | triplane-cell masking inside one 8DNA model (shared flows frozen) fails | anything about complete owned units |
| 31 | explicit per-part surface units (no learned decoder) reach 8DNA-level T0 and recover T3 when refitted; object-level D_T = all objects on the compact teaset; no wall-clock saving in that implementation | unit-level D_T sparsity; neural units |
| 35 | RNA qualified on its published asset (Lego). Native generation and training with unchanged upstream code. Retraining matches the released checkpoints within 0.2 dB on the same val set. The gap to Table 1 (−1.9 HQ / −0.8 small) is explained by 1024- vs 4096-spp validation targets (+1.7 / +0.6 dB). Overall COMPARABLE | performance claims (production renderer unreleased) |
| 34 | Batch 2: seams = independent per-unit error fields (decorrelated across cuts) exposed by hard selection; not supervision starvation. P16-PU composition removes 86% of the seam excess on held-out C2 with lower boundary error, exact output-influence accounting (S_i ∪ B_i), and REINIT intact | recovery of P16's interior fidelity loss (−0.20 dB, logL1 1.18×); C1 accuracy gain |
| 33 | RNA baseline audit: integration and evaluation faithful to upstream (no defect); C1 failure = RNA representation limit on near-mirror Ni (by exclusion: not noise, estimator, distribution, capacity, resolution, convergence or data density 100-200 views); C2 gate miss dominated by 128-spp target noise (A vs converged ref 26.7 dB on 4 views) | a credible static reference (needs a prospective protocol); a positive test of the C1 representation limit |
| 36 | Batch 3 on the credible native Lego: 16 independent complete RNA units (1.54× A's parameters, capacity-confounded upward) score −0.32 dB vs native RNA HQ on 16 held-out 4096-spp views (logL1 1.05×, hf 1.06×, off-band 0.996×); PU removes 91% of the seam excess; REINIT of one unit: 15 others bit-identical, output change exactly S_i ∪ B_i (1.1% in neighbour bands), unit recovered. Screening Case A. Teaset interior loss does not recur | radiometric recovery after a physical edit; D_T sparsity; capacity-equal B; seed variance; the residual hf/band mechanism |
| 32 | RNA split into complete independent units: exact parameter/output preservation and full-state unit reconstruction; P16 costs 1.15× HDR log error and visible seam steps vs RNA (C2), not fixed by 16× capacity | static fidelity under a credible baseline (gate failed: 17.97 / 24.92 dB); recovery after a physical edit; cost |

## 3. Architecture Batch Program (≥ 5 batches before a broad viability judgment)

| Batch | Status | Question | Outcome / stop |
|---|---|---|---|
| 1 — RNA independent units (worklog 32) | **REPORTED; stopped on the baseline-credibility condition** | Can RNA be split into independently owned units with full-state reconstruction and preservation, without unacceptable static-GI loss? | Ownership exact; fidelity verdict not adjudicated; costs: HDR log +12–15%, seam steps |
| 2 — boundary fidelity (worklog 34) | **REPORTED** | Why do independent units lose fidelity at boundaries, and can one representation change fix it without breaking reconstruction? | H2 + H3 supported, H1 starvation contradicted; P16-PU fixes seams, not interior loss; Batch 3 needs a user decision (recommended: immutable shared decoder + owned features under PU) |
| 3 — native Lego generalisation (worklog 36) | **REPORTED: Case A** | Does the independent architecture keep credible static fidelity and reconstructible ownership on a native, paper-comparable RNA asset? | B −0.32 / C −0.31 dB, logL1 1.05×, off-band unchanged, seams fixed by PU, REINIT exact. Batch 4 needs a user decision (recommended: physical geometry-edit reconstruction on Lego, oracle D_T units vs all units vs object) |
| 2 (original plan) | superseded by the above | Recommended, after resolving credibility: radiometric validity of unit-level full recompute under a physical T0 → T3 edit (oracle dirty units vs all units vs whole objects). Alternative: a representation variant (frozen shared decoder + owned features; or blended overlapping units) | — |
| 4 — physical Lego geometry edit (worklog 37) | **REPORTED: CONDITIONAL** | Can oracle-selected transport-dirty units recover physical post-edit radiance, and is D_T meaningfully sparse vs full/object recomputation? | D_T ≠ D_G (7 of 8 dirty units stationary); D* 8/17 units, 39% of surface, 99% of change energy; R = 0.98; V* exact and valid; B_sel −0.313 dB vs A_full (rule ≥ −0.26), gap concentrated in weakly dirty unit 3; cost 0.785× wall-clock for 45.5% of parameters. Seed replicate (worklog 38): WITHIN_MARGIN, mean gap −0.082 dB over seeds 0–2 (A_full seed spread 0.23 dB), unit 3 not systematic, all seeds exact on V\* and support. Batch 5 needs a user decision (recommended: a second physical edit in a glossy or multi-bounce interreflection regime, ≥ 2 seeds per arm) |
| 5 — Spatial-Perspective Transport Factorization (worklog 39) | **AUTHORISED 2026-10-22, in progress** | Can a factorized representation (persistent learned state + current relation-dependent incident state + reusable operator) absorb a held-out physical edit by state recomputation, without re-learning persistent parameters or unacceptable fidelity loss? | Arms: original RNA, Surface-Owned (worklogs 37–38), SPT-1 updated, SPT-1 stale-state control, SPT-1 full refit; verdict SUPPORTED / CONDITIONAL / NOT SUPPORTED |
| 6 and later | FUTURE (requires user decision) | Select from observed failure/success: representation variant, controlled interreflection relation observability, or additional assets/materials | No precommitted mechanism |
| After feasibility | FUTURE (direction adopted; implementation not authorised) | Algorithmic spatio-directional candidate hierarchy plus optional learned transport-relation scoring/dirty-unit selection; measure candidate coverage, stale-GI false negatives, and full selector+recompute cost vs oracle and algorithm-only controls | Requires Batch 4 interpretation, physical-oracle evidence and a user-approved bounded batch |

Gates retained from the object-level framing:
- **P:** phenomenon, CLOSED.
- **C:** cost motivation, CALIBRATED.
- **H:** compact state, CLOSED NEGATIVE for the tested cases.
- **X:** intra-object masking, CLOSED NEGATIVE and off-target.
- **O0/O1:** object-level feasibility and oracle refit. Worklog 31 answered them on an explicit substrate (O0 pass; O1 dense at object granularity). They remain a coarse comparison for unit-level results.

## 3a. Future Relation-Selection Decision Frontier (direction adopted 2026-10-19)

Interreflection is an intended stress capability: static spatial neighborhoods and normal similarity do not guarantee transport dependency. Algorithmic geometric hierarchy and an optional learned selector have different scientific roles:

- **Index/candidate stage:** physically interpretable spatial and directional bounds (e.g., node regions, normal cones, BRDF summaries) discover transport-relevant candidates, including nonlocal receiver changes and third-party occlusion. A search tree is not the multi-bounce dependency graph. The design must not assume normal-only or position-only grouping is sufficient.
- **Selection stage:** test whether a learned relation/invalidation scorer reduces cost or stale-radiance error relative to an algorithm-only selector on the *same candidate set*. The model should not be required to learn the whole scene hierarchy. Nothing is implemented or guaranteed yet.
- **Attribution:** first oracle dirty-set sparsity and radiometric recovery (Batch 4); then physically grounded relation coverage, candidate-recall failure accounting, algorithmic-only baseline, learned-selection ablation, total traversal/inference/refit cost and false-negative radiometric penalties. If oracle D_T is dense or candidate discovery misses important receivers, report the limitation before fitting the predictor.

A later work batch should derive its architecture hypothesis from the Batch 4 evidence; this decision record does **not** authorise selector or hierarchy implementation and does not modify the active Batch 4 experiment.

## 4. Precondition Before Batch 2: Static Credibility

**Resolved for Lego (worklogs 35–36):** native RNA on the official Lego is qualified (worklog 35), and Batch 3 used it with converged (4096-spp) held-out references under the predeclared protocol `b3_v1`. The Teaset history below is kept as the record.

Worklog 32's original RNA failed the predeclared 25 dB static-credibility bar:
- C1 (canonical near-mirror teaset): 17.97 dB;
- C2 (rough nickel α 0.3): 24.92 dB.

The batch therefore stopped on its first stop condition. Worklog 33 attributed the failure:
- **C1:** RNA's representation limit on near-mirror conductors. No integration defect; more data, capacity or resolution did not help.
- **C2:** target noise in the measurement. 37.5% of A's error is 128-spp GT noise; against converged targets A scored 26.7 dB on 4 views.

The user chooses, before any Batch-2 training, between:
- **(a)** a predeclared scene/material configuration where original RNA clears the bar; or
- **(b)** a prospectively revised credibility criterion, written as a new protocol and never applied retroactively to Batch 1. Worklog 33's recommendation: the C2 material regime scored against converged (≥ 8× spp) validation targets, pooled, final weights.

Do not select scenes, gates or thresholds after seeing candidate results.

## 5. Rules for Every Batch

- **Controls.** Compare A (original substrate) against B (owned candidate), changing only the ownership boundary and the changes strictly necessary for it. Account for capacity openly; label a comparison capacity-confounded instead of forcing equivalence.
- **Static GI evaluation:**
  - full-image metrics plus per-unit, boundary/interior and seam metrics;
  - an HDR-appropriate metric beyond PSNR;
  - high-frequency/glossy regions;
  - parameters, convergence, load/inference time and memory.
- **Unit reconstruction.** Report parameter preservation, output preservation and radiometric validity separately. A synthetic intervention proves semantics, not GI recovery.
- **Exports.** Reviewer exports go in `results/evaluation/<worklog>/` with a manifest. Tests must cover mapping, ownership, isolation, checkpoint, dispatch, boundary evaluation and composition.
- **Interpretation.** Do not treat one negative candidate as falsifying the hypothesis. Do not claim speedups from parameter counts. Do not advance automatically.

## 6. Preserve

- Worklogs 21–32 unchanged; worklog 30 as the historical intra-object masking result.
- Original RNA/8DNA outputs, checkpoints and scripts as historical baselines.
- Archived architectures under docs/history/.

## 7. SIGGRAPH 2027 Planning

Replan durations from measured training and reference wall-clock. For reference, one P16 RNA arm takes about 1.5 h on the RTX 5080 and about 2.8 h on LabServer63. Human coding burden is not a veto for a scientifically better architecture. Publication aims never override predeclared stop conditions.

## 8. Document Synchronisation

Update this file, docs/Architecture.md, docs/RESEARCH_CENTRIC_TOPIC.md and docs/BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md when a substantive architecture hypothesis/direction is adopted or rejected, a baseline role changes, a batch closes, or implementation is authorised. **Do this as soon as the user makes the direction decision, not only on Worklog completion.** Label decisions vs hypotheses vs evidence, and never revise the currently running experiment by documentation alone.

This document does not authorise any additional batch.
