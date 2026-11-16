# Worklog — WL98: Learned Directional Transport after ECRO, Design Review (2026-11-16)

> **STATUS: COMPLETE (design-only).** Verdict: **DESIGN_READY_FOR_HUMAN_REVIEW, conditional**, with a predeclared conversion to **NO_DEFENSIBLE_CANDIDATE (scoped to scene86 / GM1 under the current canon)** if the owner declines the named human decisions. Not adoption, not experiment approval. Append-only from its commit on.

## Origin and scope

**Brief.** The user's brief, "ReST-GI의 기존 연구 방향을 유지하면서 ECRO 이후의 Learned Directional Transport Representation을 설계하라" (keep the existing research direction and design a learned directional transport representation after ECRO).

**Batch class.** A regular number. The batch opens a new representation hypothesis at design level.

**Mode.** Design-only. Nothing is implemented, trained, rendered or adopted.

**Question.** Can spatially distributed, high-frequency transport, learned in a continuous or adaptive angular representation, be reused after a rigid geometry edit without retraining, with a meaningful quality or cost advantage over explicit transport? If so, how does it differ from ECRO and from the explicit comparator, and what minimal experiment would refute or support that difference?

**Central evidence.** The WL97 audit (`docs/audits/wl97_evidence_reliability/`): the COORD ledger and J1–J7. The WL97 proposed annotations are **not adopted** and are used only as cited evidence.

**Unchanged.**
- The research direction: reusable learned transport, sharp interreflection fidelity, rigid-edit reuse without retraining.
- K/C ownership.
- Every result, verdict and historical baseline.
- Architecture.md.

**Where.**
- Design documents: `docs/plans/wl98_learned_directional_transport/`.
- Numpy scratch calculations: `experiments/wl98_calc/`, `results/wl98_calc/`. CPU only, no GPU, no server.

## Process

1. **Coordinator draft.** `DESIGN_REVIEW.md` was written before any agent ran.
   - Primary: **FLS** (learned flux-allocated spatio-angular lobe state per surface element, a shared rule composed with explicit cubature delivery).
   - Alternative: **ECRO-FC**.
   - The coordinator authored WL86–WL96A, so the draft carries a conflict of interest.
2. **The user authorised a workflow,** stating that reliability and quality outweigh token cost in this design batch.
3. **Three independent designers (workflow phase 1), each blind to the coordinator draft.**
   - D1: physics and measure first.
   - D2: learning and generalization first.
   - D3: systems and cost first.
   - All three finished. Each wrote a full report and numpy scratch calculations.
4. **Four adversarial critics** over all four designs: X1 physics / measure / ownership, X2 learnability and evidence, X3 systems and cost, X4 prior art with web search. All four finished.
   - Usage-limit and session interruptions stopped the critique phase three times.
   - X2 and X4 finished on the second attempt. X1 and X3 finished on the fourth attempt, resumed from their on-disk reports and an extracted action log of the interrupted attempts (`results/wl98_calc/{X1,X3}/_interrupted_run_log.md`).
5. **Two independent judges** finished: J1 weighted defensibility and falsifiability, J2 weighted the research objective. The first judge attempt was also interrupted.
6. **Coordinator synthesis** (this worklog and `DESIGN_REVIEW.md` §11). Dissent is preserved.

## Established evidence used

The full table with sources is in `DESIGN_REVIEW.md` §1 (E1–E12). Short form:

- **Learned S1 sender error.** The error of the two OWN checkpoints at CK2 is large. "Avoidable" was never tested.
- **Rank 16.** A passive rank-16 operator exact on the CK2 incident support exists. All-incidence fits trade fidelity.
- **Grazing coverage** is 0.11 % of training lightings.
- **Solid-angle-mean 4×16 bias.** +40–50 % at box A. About half of it is the averaging convention; its sign depends on the receiver; it vanishes at 8×32.
- **Exact direction + hit point.** Unbiased in expectation; no gain from a single draw at the deployed budget.
- **WL96A.** The headroom result holds only in table space, with an oracle prefix and a resolvent-only learned element. Its floor table has a 0.010 patch-spatial readout error.
- **Cost.** The resolvent-only ceiling is ≈ 1.02–1.04 with the frame included.
- **Dynamic edits.** No reference-validated dynamic edit exists after WL93D.
- **ECRO local factors** are non-passive.
- **Scene family.** GM1 / scene86 contains only planar and convex objects.

## Candidates

### Coordinator draft (written before the designers)

- **FLS (family C, patch / element level).**
  - What is learned: a shared rule from the delivered incident state to K anisotropic spatio-angular lobes with flux allocation to a NULL sink. It is passive by construction and equivariant in local frames.
  - Delivery: explicit deterministic lobe cubature on the current geometry.
  - Learning-free twin: ELT (moment-matched explicit lobes).
  - Admitted weakness: not exactly additive across sources.
- **ECRO-FC (alternative).** The ECRO rule with a flux-conserving adaptive table, coverage-stratified training and a passivity constraint.

### Independent designers

| designer | primary | alternative | verdict proposed |
|---|---|---|---|
| D1 physics / measure | **XQT**, exact-sample quadrature transport. A learned, sparse, column-stochastic redistribution W_θ of exact-ER flux quanta from explicit relation rays to visibility-consistent neighbouring receiver cells, on a flux-exact projected-disk carrier. It is an *estimator*: learning buys variance reduction (kernel adaptivity) over the unpooled exact gather and over its fixed-kernel twin. | **LLT**, learned lobe form factors from sparse explicit probes. It must infer footprint visibility between probes, so it runs only behind an information gate. | DESIGN_READY_FOR_HUMAN_REVIEW |
| D2 learning / generalization | **LIEM-S**, Learned Intrinsic Exit Measure. A shared, object-local learned kernel for the *collided intrinsic transport of rigid scattering-medium domains with a rough-dielectric interface*: exit position and direction density, mass, analytic anchors; exact interface closure; ballistic transport explicit. | **LIEM-B**, the same kernel as a sparse multi-resolution categorical exit distribution. This is a representation-level alternative, not an independent mechanism. | DESIGN_READY_FOR_HUMAN_REVIEW |
| D3 systems / cost | **RB-DTO**, Rigid-Body Directional Transfer Operator. One shared model compiles, per *non-convex* rigid body, a passive, measure-valued, body-local operator for self-transport after ≥ 1 self-hit (N_b = (I − K_b C_self)⁻¹ K_b − K_b), with continuous sharp lobes plus a broad channel. A self-path certificate is tested on every edit. | **LER-Delta**, a learned static-environment resolvent with explicit edit deltas (M − B). It needs per-environment learned state [HUM]. | DESIGN_READY_FOR_HUMAN_REVIEW |

### Convergent findings across the three independent designers

All three reached these without seeing each other or the coordinator draft. They are still [PRED / ARITH], not measured; each design gates them with a learning-free stage before any training.

1. **In the GM1 / scene86 family (planar and convex objects, pointwise frozen ER, explicit visibility), no rigid-invariant nonlocal transport exists to be learned.**
   - Every learned element there would re-learn ER (local BRDF compression, excluded by the brief), learn inter-body C (WL52 class: slower, fails at transitions), or be a resolvent whose utility ceiling is ≤ ≈ 1.1–1.3 depending on convention.
   - Statement and derivation by designer: D1 Proposition I; D2 structural finding (ceiling ≤ 1.12 by scratch arithmetic); D3 ownership finding (positive H3 impossible on GM1 under engineered conventions).
   - Consequence: CK2 and GM1 can attribute H1 / H2 for the ECRO lineage but cannot carry a positive H3.
2. **The coordinator's FLS (patch-level lobes) is rejected in kind by all three.**
   - D1 (C1: "amortized ER fitting = BRDF compression in a spatial wrapper").
   - D2 (R-C: brief exclusion; violates P1 "learn only what cannot be computed exactly at the target cost").
   - D3 (C patch-level: "re-learns ER").
   - The coordinator's argument that FLS learns "distributed incident → transport state, not a BRDF" is not accepted by any independent designer.
   - Confirmed by all four critics and both judges (see Critique). The coordinator withdraws FLS (see Architecture verdict).
3. **Where a learned representation *can* carry rigid-reusable, nonlocal, sharp transport:**
   - in the **intrinsic transport of non-convex rigid bodies** (D3), self-hit paths;
   - in the **intrinsic transport of scattering-medium domains with a rough-dielectric interface** (D2), through-object, total-internal-reflection and displaced exits;
   - or, in a different role, as an **estimator** amortizing the variance of explicit exact-sample delivery (D1).
   - The first two require **a workload family that GM1 does not contain**: a human decision.
4. **The learning-free fix of the angular carrier is mandatory as a control, not a candidate.** All three, and the coordinator draft, include the following controls:
   - solid-angle-mean 4×16;
   - flux-conserving 4×16;
   - 8×32;
   - adaptive / quadtree.
5. **ECRO-lineage retraining (H1 arm) is diagnostic only.** Its predicted H3 content for geometry edits in GM1 is nil; whether to run it is a human decision (D1 C1, D2 HD8, D3 D8).
6. **The ownership refinement is a canon question.** Learning same-asset self-paths inside a body operator (D3 D1; D1 O1 / HD-8) or per-environment learned state (D3 D4) touches Architecture §2. Only the user can open it.

### Divergences (to be resolved or preserved)

- **Role of learning.**
  - D1: learning as an *estimator* over exact samples in GM1 itself. Its admissibility as "learned transport" is HD-1.
  - D2 / D3: learning as an *intrinsic nonlocal operator* outside GM1.
- **Domain.**
  - D2: participating media behind rough dielectric interfaces. This needs a qualified refractive-media reference; Cycles was validated only for index-matched media.
  - D3: opaque non-convex glossy bodies. This needs a self-path certificate and an ownership refinement.
- **The coordinator's FLS.** Kept by the coordinator draft as primary; rejected by D1–D3, X1–X4 and J1–J2. **Withdrawn by the coordinator** (see Architecture verdict).

## Falsification structure (designers' common skeleton; the judges' merged plan is in the Validation contract below)

The common skeleton across all designs:

- **Track G: ECRO-lineage attribution on GM1** (H1 / H2; H3 at ceiling level).
  - Admitted workloads:
    - E2;
    - E5;
    - ≥ 1 new margin-disjoint grazing workload with a path-traced **T1** reference;
    - CK2 as diagnostic only.
  - Learning-free G0 first:
    - the explicit quality–cost frontier: binned controls, flux-conserving variants, 8×32, adaptive, X1 at 1–16× draws, splat / DQ;
    - role-specific utility ceilings;
    - the patch × 64 class floor against the path-traced reference (the missing WL97 P0-1b evidence).
  - STOP before any training if no learnable role reaches the human-set utility margin.
- **Track B: the candidate, on its own admitted workload family** (non-convex rigid bodies or medium domains; human decision).
  - Learning-free headroom and capacity gates first. Examples: D2 I0 / I1, D3 Stage 0 Amdahl and the oracle floor, D1 S1-c twin vs exact-field oracle.
  - Then a shared-model training with ≥ 3 seeds, against the learning-free twin.
  - Then T0 → T1 rigid edits with fixed weights.
  - Then quality-matched cost against explicit comparators.
- **Binding rules.**
  - M64 is never truth; truth is path-traced, event- or class-tagged references in independent halves.
  - Paired statistics with block-bootstrap SE (never the an94.eh L1 floor).
  - ≥ 3 seeds plus state-draw resampling.
  - Admission and stopping are frozen before any arm output.
  - Dev runs use non-evidence inputs.
  - A full-scale probe of every stage is run before the freeze.

## Human decisions (designers' union; consolidated list under Architecture verdict → Human Decisions)

1. Admissibility of each learned role as "reusable learned light transport":
   - an estimator over exact samples (D1 HD-1);
   - an element-local transport-state rule (the coordinator's HD-1);
   - an intrinsic body / medium operator (D2 / D3).
2. Workload extension beyond GM1: non-convex glossy rigid bodies and/or scattering media behind rough dielectric interfaces (D2 HD2, D3 D2 / D7). A reference renderer for refractive media (D2 HD7).
3. Ownership refinement: body self-paths inside a learned body operator; per-environment learned state (D3 D1 / D4; D1 HD-8; D2 HD5). This is a canon change, outside the batch.
4. Whether Track G ECRO-lineage retraining runs at all (D1, D2 HD8, D3 D8).
5. All numerical thresholds: utility margin, minimum meaningful effect, gate tolerances, grazing quota, seeds, budgets (all designers).
6. Gate A / Gate B carry-over from WL96: explicit carriers inside the method or only as comparators.
7. External baselines (Neural Full Recompute / Native Reuse) or explicitly UNMEASURED.
8. WL97 annotations: not adopted.

## Critique (phase 2)

The four adversarial critics read all four designs (the coordinator draft and D1–D3) in full. Reports: `X1_critic_physics_measure.md`, `X2_critic_learnability.md`, `X3_critic_systems_cost.md`, `X4_critic_prior_art.md`. Structured output: `results/wl98_calc/critics_structured.json`. Scratch numbers are [CALC / ARITH / SCRATCH], design input only.

| critic | proposed verdict | survivors (conditional) |
|---|---|---|
| X1 physics / measure / ownership | DESIGN_READY_FOR_HUMAN_REVIEW, conditional | XQT (in canon, needs HD-1); LIEM-S (needs GM2 scope and a reference qualification). RB-DTO needs a canon change. |
| X2 learnability / evidence | NO_DEFENSIBLE_CANDIDATE (scoped to GM1 and the current canon) | LIEM-S and RB-DTO, only outside the current scope |
| X3 systems / cost | INSUFFICIENT_EVIDENCE | RB-DTO and LIEM-S have a possible but narrow regime in workloads that do not exist yet |
| X4 prior art / novelty | NO_DEFENSIBLE_CANDIDATE (current canon) | RB-DTO (needs a canon refinement), LIEM-S (needs GM2 scope) |

**Findings shared by all four critics.**
- **FLS fails as a learned candidate.**
  - [CALC, X1] Under the CK2 light (cos_i ≈ 0.08) the mirror direction sweeps ≈ 18° (up to 28°) across one 0.2 m S1 patch, against a lobe width of ≈ 1°. About 10 separable lobes are needed per element per light, and 79 % of the irradiance falls on patches that need more than 8.
  - A sheared image-source lobe fixes this, but it is the analytic mirror construction, so nothing is left to learn.
  - In GM1 its learned part is ER × incident-deposit compression plus mixture reduction, and its delivery is explicit cubature (X1, X2, X3, X4).
  - Its runtime equals its ELT twin's by construction (X3).
  - It is nonlinear, which forfeits the only measured learned lever, the source-only update (3.5–3.9× in WL87).
  - Its "selective re-trace" claim is false: any edit changes every lobe (X3).
  - X4 places it in the published VSL / VSGL / SG-light class with learned lobe fitting.
- **ECRO-FC fails as a candidate.**
  - Its generator conditions on world pose, a scene-mean token and form-factor message passing [CODE model86.py:83-104].
  - It changes the carrier, the coverage and the passivity at once.
  - It is usable only as factorised diagnostic arms.
- **No design shows learning headroom over a fair twin in GM1** (X2):
  - D1's flatland oracle / twin ratio is 1.0–1.33;
  - D2's slab has high-frequency exit content only where walks are short;
  - D3's utility rests on placeholder constants and a uniform-binned (weak) twin.
- **Missed alternatives.** All four critics independently name **learned guiding / learned sampling densities driving an unbiased explicit estimator** as an alternative no design compared. Further missed twins:
  - compressed precomputed per-body transfer (all-frequency PRT class) for RB-DTO;
  - edge-aware kernels, spatial reuse with exact re-evaluation, sender-side deposition and MIS for XQT;
  - Woodbury / Krylov edit updates for LER-Delta.

**Experiment-plan defects** (the merged plan below adopts the fixes):
1. **Reference instruments.** The existing reference machinery (pt86a.field_src: solid-angle-mean 64-bin tallies, e2 / e3 × panel / other tags, kmax 5) cannot validate sub-bin claims. New exact-direction, cos-weighted, incident-flux and path-tagged probes must be qualified first (X1).
2. **Floors and tests.**
   - Class floors must be fitted on one reference half and scored on the other.
   - H1 vs H2 must be split three ways: optimisation, allocation and operator capacity, with a rank axis and a light-conditioned control.
   - 3 seeds detect only effects ≥ 2.4–11.6×, so a seed pilot must set n (X2).
3. **Cost accounting** (X3).
   - In GM1 the only large quality-matched lever is the table bias floor seen through the frame.
   - Whole-image tiers do not separate the table arms (within 1–2 %), and ROI tiers near a floor are ill-conditioned.
   - Explicit draws pool over frames at ≈ 3 % of a 64-spp frame.
   - One kernel-time basis is needed, and recompilation must be counted as cost.
   - An engineered explicit comparator is missing.
4. **Prior art** (X4).
   - The closest prior art must be a measured comparator.
   - OSF / TransGI-type object assets are Native Reuse and must be measured or labelled UNMEASURED.
   - For FLS, XQT, ECRO-FC and LLT, "fixed weights at T1" is rule reuse with all state regenerated, so a T0→T1 test cannot show transport reuse.
5. **Coordinator draft.** `DESIGN_REVIEW.md` §10 cited a CRITIQUE.md and a worklog that did not exist at the time (X1, X2, X4). Both pointers are corrected in §11 of that file.

## Judges (phase 3)

Reports: `JUDGE1_report.md`, `JUDGE2_report.md`. Structured output: `results/wl98_calc/judges_structured.json`. J2's scratch: `experiments/wl98_calc/J2/rb_twin_split.py` → `results/wl98_calc/J2/rb_twin_split.json`.

| | J1 (defensibility / falsifiability) | J2 (research objective) |
|---|---|---|
| proposed verdict | DESIGN_READY_FOR_HUMAN_REVIEW | DESIGN_READY_FOR_HUMAN_REVIEW |
| primary | **RB-DTO-C**: D3's RB-DTO with the critics' corrections. Needs HA, a certified rigid self-coupling canon refinement. | **RB-DTO-E**: D3's RB-DTO reformulated to consume cached explicit self-relation records validated by the canonical dirty test. C keeps ownership of C_self (canon-compatible by J2's reading; owner to confirm). |
| alternative | **XQT** (D1), if HB / HD-1 admits a learned estimator; otherwise none within GM1, with LIEM-S as the GM2 reserve | **LIEM-S** (D2), if HD-S opens GM2 media; LIEM-B only as its representation control |
| ranking (0–10) | RB-DTO-C 5 > LIEM-S 3.5 = XQT 3.5 > ECRO lineage 2 = LIEM-B 2 > FLS 1.5 > … | RB-DTO-E 4.5 > LIEM-S 4.0 > ECRO lineage 2.5 = XQT 2.5 > FLS 2.0 > LLT 1.5 > LER-Delta 1.0 |
| conversion rule | if HA is rejected, HB is "no" and GM2 stays closed → NO_DEFENSIBLE_CANDIDATE (scoped) | if HD-L, HD-W and HD-S are declined → NO_DEFENSIBLE_CANDIDATE (scoped) |

**Judge findings beyond the critics.**
- **J1 corrects the reason GM1 has no candidate.**
  - The deciding factor is not "information beyond explicit queries": D1's Proposition I applies equally to media and to rigid-body self-transport.
  - It is whether the amortised quantity stays invariant under the edit, so that computed state can be reused and not only weights (Architecture.md:56).
  - In GM1 only rule reuse is possible. On the kernel basis, regenerating the learned state per edit costs 0.85–0.97 of the explicit work it replaces (D3 / X3 arithmetic).
- **J2's "pincer"** [PRED, merging D1 Proposition I, D2 §3, D3 §2.5 and the J2 scratch]:
  - (a) State that is not rigid-invariant must be regenerated per edit and competes with cheap explicit evaluation.
  - (b) Rigid-invariant transport (body self-transport, intrinsic media) has reusable learned state. But a learning-free per-asset precomputed table (TwinP / Twin-T) is equally reusable and no more expensive at edit time.
  - So the edit-time lever belongs to precomputation. What is specific to learning is static: memory (≈ 2 MB vs 35–180 MB per asset at α 0.03, placeholder constants), one-off precompute and generalisation to unseen assets.
  - J2 therefore requires the utility credit to be split: U_pre = U(TwinP) / U(per-update explicit), and U_learn = U(learned) / U(TwinP).
- **Disagreement among reviewers on ECRO.** J2 disagrees with X1, X2 and X4 on one point: ECRO's Ft is an explicit visibility input, not learned visibility. J2 places the fatal ECRO-lineage defects in whole-scene conditioning, full per-edit regeneration and the re-learning of ER.
- **Prediction (both judges).** Every survivor is more likely to stop at a learning-free Stage-1 gate than to reach training.

## Architecture verdict

The brief asked for these report sections; they are recorded here so the worklog is the durable copy.

### Implementation Fact

N/A. Nothing was implemented, trained, rendered or run on a GPU. The only computations are CPU numpy scratch calculations labelled [SCRATCH / CALC / ARITH / PRED], not [MEAS].

### Established Evidence

The evidence is the E1–E12 table of `DESIGN_REVIEW.md` §1 and the short form above. WL97 annotations were used only as cited evidence, not adopted. Nothing in WL98 is new measured evidence.

### Architecture Hypothesis

Coordinator synthesis of the two judges:
- **Primary: RB-DTO (rigid-body directional transfer operator), in a corrected form.**
  - It is a shared learned operator for the sharp multi-bounce *self*-transport of a rigid non-convex body: the exit measure after ≥ 1 self-hit, N_b = (I − K_b C_self)⁻¹ K_b − K_b.
  - The first event stays exact ER. The operator is reused as computed state under that body's rigid motion.
  - Required corrections (J1 and J2 agree):
    - linear per-incident-sample pushforward, with no nonlinear per-patch mixture reduction;
    - an exact escape budget against the cached explicit self-relation;
    - sender-side delivery of sharp exits into the current-geometry C;
    - a whole-body explicit fallback on intrusion;
    - a convex-body N_b = 0 null test;
    - path-tag ownership disjointness.
  - The two judges differ on the ownership form. **RB-DTO-E** (J2) consumes cached explicit self-relation records under the canonical dirty test, so C keeps C_self. **RB-DTO-C** (J1, D3's original) infers self-visibility under a geometric certificate, which needs a refinement of Architecture §2. The choice is a human decision (HA / HD-C).
  - It is the only candidate whose success would answer the brief's question (sharp, nonlocal, reused as state after a rigid edit). It is also not ECRO-lineage (HD-L).
- **Independent alternative.** The two judges disagree, and the split is preserved.
  - **LIEM-S** (D2) is preferred by J2, X2 and X4. It is a learned intrinsic exit measure of scattering media behind a rough dielectric interface, with canonical ownership. It needs GM2 to be opened and a refractive-medium reference to be qualified. It does not learn sharp interreflection, since sharpness comes from the exact interface. Its closest prior art is Vicini 2019.
  - **XQT** (D1) is preferred by J1 and X1. It is a learned flux-conserving quadrature over exact samples, in GM1 and within canon. It needs HD-1 to admit a learned estimator as "learned transport", and it is predicted to stop at its learning-free gate (oracle / twin 1.0–1.33).
  - Each alternative is opened by a different human decision; neither is ranked above the other here.
- **Withdrawn by the coordinator: FLS** (the coordinator's own draft primary).
  - Every independent reviewer (3 designers, 4 critics, 2 judges) rejected it, for different reasons that converge:
    - the CK2 glint needs ≈ 10 lobes per element per light;
    - its learned content is ER compression plus mixture reduction;
    - delivery is explicit and its runtime equals ELT's;
    - it is nonlinear;
    - it belongs to the VSL / VSGL class.
  - The coordinator's argument that its rule learns "distributed incident → transport state, not a BRDF" did not survive. It is recorded as a minority position only.
  - ELT, upgraded with image-source shear, is kept as an explicit comparator.
- **ECRO-FC and ECRO-lineage arms**: diagnostic only (factorised H1 / H2 arms), not candidates.
- **Rejected:** LLT (learned visibility between probes), LER-Delta (per-scene state, convergence and ghosting contracts fail), LIEM-B as an independent mechanism.

### Feasibility

- **Physically sound once corrected.** RB-DTO is linear per sample, the first event is exact, passivity holds with an exact E_esc, and the measure is defined. LIEM-S is passive by construction, but needs an exact ballistic term and a declared interface model.
- **Not feasible in the current workload family.** scene86 / GM1 contains no non-convex glossy rigid body and no medium. Both candidates need a workload extension (HD-W / HC).
- **Reference instruments must be built and qualified before any sub-bin claim.** This means exact-direction probes, cos-weighted cells, incident flux per cell and self-path tags.

### Expected Performance

[PRED; no measurement]
- RB-DTO's regime is narrow. X3's symmetric recount of D3's own 20-body example gives U 0.16–1.21, with break-even at J_int·K_int ≳ 800–21,000 per patch.
- LIEM-S needs ≥ 10–30 internal events per exit with a network of ≲ 2·10⁴ FLOP.
- By J2's pincer, the geometry-edit advantage attributable to *learning* (U_learn, over a precomputed table) is structurally small. The learning-specific levers are memory, one-off precompute and generalisation to unseen assets.
- Both judges predict a stop at the learning-free Stage 1 as the most likely outcome.

### Failure Modes

- **RB-DTO:**
  - partial intrusion into the self-path region forces a whole-body fallback;
  - shallow cavities or low albedo leave no self-transport share (U0 < U_min);
  - the oracle does not beat a compressed precomputed per-body transfer at matched memory;
  - nonlinear reductions break the source-update lever;
  - with RB-DTO-C, inferred self-visibility can fail on collision pairs.
- **LIEM-S:**
  - with short walks, the exit structure is high-frequency and expensive to learn;
  - with long walks, it is smooth, and a tabulated twin reaches the same result;
  - the rigid edit cannot tell it apart from Twin-T;
  - the refractive reference is unqualified;
  - the WL68–WL80 shape-generalisation history is negative.
- **XQT:**
  - it is passive only in expectation, not per draw;
  - visibility consistency at element level leaks at intra-element edges;
  - pooling cannot help glints narrower than a receiver patch (0.46 rays per patch per draw at the CK2 glint).
- **All candidates:**
  - comparing against a weak twin inflates the learning margin;
  - quality tiers near a floor are ill-conditioned;
  - deterministic arms are over-credited without a draw-pooling convention.

### Alternative Comparison

| candidate | learned object | reuse type | in canon / scope | twin it must beat | J1 / J2 score |
|---|---|---|---|---|---|
| RB-DTO-C / -E | sharp body self-transport | state (rigid) | needs HA (C) / owner confirmation (E); GM1 extension | cached self-relation + sender-side chain tracing; compressed precomputed per-body transfer (PRT / OSF / TransGI class) | 5 / 4.5 |
| LIEM-S | collided intrinsic exit measure (media) | state (rigid) | canonical ownership; needs GM2 | Twin-W walks, Twin-T table, Vicini 2019 | 3.5 / 4.0 |
| XQT | flight estimator weights | rule only (W regenerated on dirty fields) | in canon, GM1; needs HD-1 | edge-aware kernel, spatial reuse with re-evaluation, sender-side deposition, MIS | 3.5 / 2.5 |
| ECRO lineage | factorised diagnostic arms | rule only | GM1 | weighting-matched class floors | 2 / 2.5 |
| FLS (withdrawn) | ER × deposit lobe compression | rule only | brief exclusion | ELT with image-source shear | 1.5 / 2.0 |

### Validation Contract

The two judges' plans agree in structure. This is the merged form; every number is [HUM].
- **S0, learning-free and mandatory.**
  - Reference-instrument qualification against the deterministic exact-ER quadrature, with independent halves. M64 is never truth. A failure stops the batch as STOPPED_REFERENCE_UNQUALIFIED.
  - Workload admission on geometry-only or learning-free rules before any arm output:
    - GM1 Track G: grazing W-G2 admitted on the delivered sharp share at glint-core and tail receivers; a non-grazing W-N; a distribution-shifted held-out W-H;
    - CK2 diagnostic only, with a new T1 reference;
    - Track B, if HA / HD-W: grazing cavity, broad, held-out body class and light class, adversarial;
    - rigid edits with path-traced T0 and T1 references.
    - A failure stops the batch as STOPPED_UNQUALIFIED_WORKLOAD.
  - Engineered explicit frontier and cost anatomy:
    - one kernel basis plus wall time; compile and cache-miss counts;
    - draw-pooling convention (time-to-tier from the edit);
    - tiers away from floors, with a NOT-REACHED rule;
    - lights 1 / 4 / 16; N ≈ 1.3k / 5k / 11k.
- **S1, learning-free decisions.**
  - ECRO-lineage carrier floors and operator-class floors on a rank axis at matched weighting.
  - Role ceilings. Gate G: if no role reaches U_min on ≥ 2 workloads, H3 is supported at ceiling for the lineage.
  - RB-DTO:
    - U0 against a quality-matched Twin 1;
    - the per-body oracle floor against TwinP at matched memory and apply time;
    - a collision-pair test.
    - Stop if U0 < U_min or the gap is within paired noise.
  - XQT (if HB): a field-independent oracle against the strongest twins, with common random numbers, ≥ 16 draws and equal cost.
  - LIEM-S (if HD-S): I0.1 reference, I0.2 headroom with measured U0, then floors.
- **S2, training (survivors only).**
  - A seed pilot (≥ 5 seeds) sets n.
  - Factorised ECRO arms, only by HD-T.
  - RB-DTO arms:
    - natural vs designed grazing coverage;
    - a per-body fitted arm;
    - an exposure-bias metric.
  - Contract tests before evaluation: measure, per-realisation passivity, superposition, rigid equivariance, convex N_b = 0, path-tag disjointness, furnace, reciprocity. A failure is STOPPED_CONTRACT.
  - H-L stop if learned ≤ twin within paired noise on ≥ 2 workloads.
- **S3, fixed-weight T0 → T1** against STALE, TwinP and quality-matched explicit arms.
  - Measures: changed-region fidelity, signed-change slope / correlation, leakage outside the predicted support, extra T1 − T0 error, selective = full.
- **S4, economics.**
  - The U_pre / U_learn split.
  - A positive claim needs two workloads, including the held-out one.
  - Native Reuse is measured or reported as UNMEASURED.
  - Neural Full Recompute applies only where a representation is actually invalidated.
- **Statistics.**
  - Common random numbers; block bootstrap; independent reference halves.
  - Seed as a random effect; never an L1 noise floor as a tolerance; heavy-tail guards.
  - All rules frozen before results.

### Unresolved Questions

- Does any admitted workload have a quality-matched intra-body self-transport share giving U0 ≥ U_min, *and* a per-body learned oracle that beats compressed precomputed per-body transfer at matched memory and apply time?
- Does any medium configuration have both long walks and learned exit structure above MC noise, against a qualified refractive-medium reference?
- Does the delivery-optimal ECRO operator class (rank axis, matched weighting) meet the target against a path-traced reference at all (WL97 P0-1b)? If not, H1 coverage training is moot.
- Does learned guiding (sampling density only) count as a candidate, a comparator or out of scope? No design compared it, although all four critics named it.
- Is ECRO's Ft an explicit visibility input (J2) or learned visibility coupling (X1, X2, X4)? The answer changes the reason, not the verdict, for the ECRO lineage.

### Human Decisions

Consolidated from J1 (HA–HJ) and J2 (HD-*). None is decided by this worklog.
1. **Lineage (HD-L).** The 2026-11-13 decision makes ECRO central for GM1. May the GM1 follow-up hypothesis be non-ECRO-lineage (RB-DTO)? Do the ECRO training arms run for attribution even if the learning-free Gate G closes the lineage (HD-T)?
2. **Canon (HA / HD-C).** Choose between RB-DTO-E (explicit self-relation records as input; owner to confirm canon compatibility) and RB-DTO-C (learned self-visibility under a certificate with whole-body fallback; refines Architecture §2 ownership of exposed same-asset paths).
3. **Estimator admissibility (HB / HD-1).** Is a learned estimator over explicit samples (XQT), or learned guiding, a candidate, or a comparator only?
4. **Scope (HC / HD-S / HD-W).**
   - Should GM1 be extended with non-convex glossy rigid bodies and an adversarial workload?
   - Should GM2 media (LIEM-S) open before an interpretable GM1 result, with a rough-dielectric closure and a choice of reference renderer?
5. **Comparator definition (HD-E).** Does a per-asset precomputed transfer (TwinP / PRT class) count as "explicit transport" for H3? If so, the U_pre / U_learn split is binding.
6. **Numbers and conventions (HE / HF / HD-N / HD-U).**
   - Thresholds: U_min, tiers, m_req, tolerances, admission values.
   - Seeds and draws: seed count via the pilot.
   - Cost conventions: kernel vs wall basis, the engineered comparator, frame spp, pooling horizon, light and scale axes.
7. **External baselines (HG / HD-X).** Native Reuse (OSF / TransGI / 8DNA-type per-object assets) measured or UNMEASURED; Neural Full Recompute only where a representation is actually invalidated.
8. **Novelty (HI).** Is the narrow RB-DTO niche (sharp + shared compiler + passive measure contract) worth testing against PRT, OSF, TransGI and MRT prior art?
9. **Batch class and budget (HJ / HD-B).** Batch class, compute and server.
10. **WL97 annotations.** Remain unadopted.

### Verdict

**DESIGN_READY_FOR_HUMAN_REVIEW, conditional.**

Tally of proposed verdicts:
- DESIGN_READY (conditional): D1, D2, D3, X1, J1, J2.
- NO_DEFENSIBLE_CANDIDATE (scoped): X2, X4.
- INSUFFICIENT_EVIDENCE: X3.

The label means that a corrected primary contract (RB-DTO), two conditional alternatives (LIEM-S, XQT) and a merged learning-free-first validation contract are ready for the owner's decisions. **It is not an expectation of success.** Every reviewer who scored the survivors predicts that a learning-free Stage-1 stop is more likely than reaching training.

**Conversion rule (predeclared).** The verdict becomes NO_DEFENSIBLE_CANDIDATE, scoped to scene86 / GM1 under the current canon, if the owner does all of the following:
- keeps the ECRO-lineage requirement (HD-L);
- rejects both RB-DTO forms (HA / HD-C);
- rules a learned estimator inadmissible (HB);
- keeps GM2 closed.

All nine reviewers agree with that scoped reading of planar / convex GM1. It is not a class-level falsification of reusable learned transport (WL97 P0-5b).

**Dissent preserved:**
- X2 and X4: NO_DEFENSIBLE_CANDIDATE now, since no candidate has demonstrated headroom over a fair twin.
- X3: INSUFFICIENT_EVIDENCE. The regime exists only in workloads that do not exist yet, and the decisive anatomy is cheap and learning-free.
- X1: RB-DTO violates Architecture §2 in its original form; within canon only XQT is physically defensible.
- D2: canonical ownership should decide the primary, which would make LIEM-S primary.
- D1: in GM1 the honest learned transport is an estimator.
- The coordinator's minority position on FLS.

Not adoption. No implementation, training, experiment or successor batch is authorised.

## Disclosures

- The coordinator draft (`DESIGN_REVIEW.md`) was written before the designers. The designers were instructed not to open it. The critics and judges see it as one of four designs.
- The workflow was interrupted several times (session end, usage limits). The designers' results were kept from cache. The critics and judges were restarted, and the interrupted critics resumed from their on-disk reports plus extracted action logs.
- `results/wl98_calc/critics_structured.json` was first written mid-run with the X2 / X4 results only. It was overwritten with the final four-critic output; the interrupted-run logs keep the intermediate state.
- The scratch calculations are numpy on the local PC (no GPU, no server). Designers label them [SCRATCH / ARITH / PRED], not [MEAS].
- Untracked files from other sessions (`docs/worklogs/21B_*`, `evaluation/21B/`, `experiments/8dna_ck21b/`, `experiments/ecro_nearmirror93e/`) are not part of WL98 and are left untouched.

## Reproducibility

- Design documents: `docs/plans/wl98_learned_directional_transport/`. The reviewer reports are kept verbatim.
- Structured outputs: `results/wl98_calc/{designers,critics,judges}_structured.json`.
- Workflow run `wf_0eabece4-2eb` (script persisted in the session directory; not part of the repository).
- Scratch scripts and outputs:
  - `experiments/wl98_calc/D1/flat_pool.py` → `results/wl98_calc/D1/flat_pool.json`
  - `experiments/wl98_calc/D2/{amdahl_ceiling,angular_capacity,slab_tension}.py` → `results/wl98_calc/D2/*.json`
  - `experiments/wl98_calc/D3/cost_model.py` → `results/wl98_calc/D3/cost_model.json`
  - `experiments/wl98_calc/X1/{physics_checks,ck2_spread}.py` → `results/wl98_calc/X1/*.json`
  - `experiments/wl98_calc/X2/learnability_calc.py` → `results/wl98_calc/X2/learnability_calc.json`
  - `experiments/wl98_calc/X3/cost_check{,2}.py` → `results/wl98_calc/X3/*.json`
  - `experiments/wl98_calc/J2/rb_twin_split.py` → `results/wl98_calc/J2/rb_twin_split.json`
- Prior-art check (coordinator, quick; the critic X4 does the focused check):
  - [TransGI (2025), object-centric neural transfer for dynamic scenes, weak on glossy↔glossy interreflection](https://alphaxiv.org/overview/2506.09909v1)
  - [Iwasaki et al., PRT for dynamic scenes with interreflection](https://diglib.eg.org/items/f008ee57-4485-440a-966d-d8c57caa5334/full)
  - [Precomputed Radiance Transfer Field for interreflections in dynamic scenes](https://www.bohrium.com/paper-details/precomputed-radiance-transfer-field-for-rendering-interreflections-in-dynamic-scenes/811961696636633088-2661)
  - [Sloan et al. 2002, PRT](https://www.microsoft.com/en-us/research/publication/precomputed-radiance-transfer-real-time-rendering-dynamic-low-frequency-lighting-environments/)
  - [Currius et al. 2020, learned SG lobe prediction](https://www.cse.chalmers.se/~ramon/spherical-gaussians-eg-2020.pdf)
  - [A Generalizable Light Transport 3D Embedding for GI (2025)](https://arxiv.org/html/2510.18189v1)
  - [Neural Global Illumination (Gao et al.)](https://cg.cs.tsinghua.edu.cn/people/~kun/2022neuralGI/NeuralGlobalIllumination.pdf)
