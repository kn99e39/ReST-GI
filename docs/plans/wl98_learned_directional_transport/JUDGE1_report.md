# WL98 — Independent judge J1: defensibility and falsifiability

**Date.** 2026-11-16. **Role.** Independent judge J1 of the design-only batch WL98 (learned directional transport after ECRO). **Weighting.** Scientific defensibility and falsifiability first: which candidate makes the sharpest claim that a minimal, cheap experiment can refute, with clean H1 / H2 / H3 / learned-vs-twin separation.

**Status.** Design judgement only. Nothing was implemented, trained, rendered or adopted. No existing file was changed. No scratch calculation was run by J1; every number below is quoted from its source. A verdict here is not adoption and not experiment approval.

**Read in full.** `DESIGN_REVIEW.md` (coordinator draft: FLS + ECRO-FC), `D1_physics_measure_first.md` (XQT + LLT), `D2_learning_generalization_first.md` (LIEM-S + LIEM-B), `D3_systems_cost_first.md` (RB-DTO + LER-Δ), `X1` … `X4` critiques.

**Verified by J1 in the canon / code** (not taken from the critics):
- `experiments/ecro86/model86.py:83-104`: `factors()` builds the dense current form-factor matrix `Ft` from the state rays, feeds `opn = 1 − ΣFt`, world centre / normal / tangent, `Ft @ msg` message passing and a scene-mean token into every per-patch `c`, `a` [CODE].
- `docs/Architecture.md:20, 25`: C "includes exposed same-asset and same-domain exit/re-entry paths"; "Object identity must not freeze exposed interreflection or merge editable air transport into K" [CANON].
- `docs/Architecture.md:56`: "Reusing network weights is different from preserving computed transport state" [CANON]. `:70`: S "preserves certified-valid state and rebuilds every invalidated component" [CANON].
- `docs/RESEARCH_CENTRIC_TOPIC.md:48` novelty rule (no novelty from "BSSRDF approximation or operator compression alone") [CANON].
- `docs/RESEARCH_ROADMAP.md:128` human decision 2026-11-13: ECRO is the central learned-global-transport implementation base for GM1; "the next architecture decision must isolate one falsifiable ECRO-lineage representation/training/ownership hypothesis" [HUM, recorded].
- WL97 `COORD_synthesis_and_p0_ledger.md` §1, §2, §8 (P0 matrix and smallest missing evidence) [AUDIT].

**Labels.** [MEAS] measured with source; [AUDIT] WL97 interpretation; [CODE] read from frozen code; [CANON] living-document text; [ARITH] someone's arithmetic on recorded values (owner named); [PRED] prediction (J1's unless attributed); [HUM] human decision.

---

## 0. Verdict in brief

1. **Unanimous finding I accept after checking it.** All three independent designers (D1 Proposition I, D2 §3, D3 §2.5) and all four critics conclude that the current scene86 / GM1 family (planar and convex objects, pointwise frozen ER) contains **no rigid-reusable learned transport object**: intrinsic transport there is exactly ER, and every nonlocal path is editable inter-object C. I add one correction to the *reason* (§1): the discriminator is not "information beyond explicit queries" (Proposition I is equally true of media and of rigid-body self-transport, where everything is also a deterministic functional of geometry and physics). It is **whether the amortised quantity is invariant under the edit class**, so that computed *state*, not only weights, can be reused (`Architecture.md:56`). In GM1-convex only rule reuse (fixed weights, state regenerated per edit) exists, and on the engineered kernel basis a per-edit learned regeneration costs as much as or more than the explicit work it replaces [ARITH D3 §2.3: 0.91–0.97; X3 §4.1: 0.85–0.87 for an 8×32 ECRO-FC].
2. **The coordinator's pair does not survive.** FLS is element-local ER × incident-field lobe compression plus a learned mixture reducer, delivered by explicit sender-side cubature (all three designers rejected this class blind; X1 §1.1 shows K ≤ 8 separable lobes cannot hold the CK2 point-light glint, about 10 per element per light). ECRO-FC keeps the visibility-coupled generator [CODE] and confounds three treatments. Both are retained only in useful learning-free / diagnostic roles (ELT with image-source shear as an explicit comparator; factorised ECRO arms for H1 / H2).
3. **Ranking (merged across designs):** RB-DTO (certified rigid-body sharp self-transport) > LIEM-S ≈ XQT > ECRO-lineage arms (diagnostic only) > LIEM-B (control) > FLS ≈ LLT > LER-Δ. Learned guiding (raised by all four critics, designed by none) is outside the "transport representation" frame and needs a scope ruling.
4. **Primary: RB-DTO-C** (D3's RB-DTO with the critics' corrections, §3.1), **conditional on a canon refinement [HUM HA]**. It is the only candidate whose learned output is high-frequency, nonlocal, *state-reusable* under rigid edits, and replaces iterated explicit sharp work rather than ER. Its decisive questions are learning-free and can stop it before any training.
5. **Independent alternative: XQT** (D1), **conditional on HD-1 [HUM HB]**. It is canon-compliant, sits in GM1 on existing workloads and machinery, and has the cheapest, cleanest learned-vs-twin gate in the batch (identical inputs, common random numbers). It is predicted to stop at that gate (D1's own scratch, X1 §2.6, X3 §4.2). If HD-1 is "no", the alternative is **none** within GM1; LIEM-S is the GM2 reserve only if the owner opens GM2 [HUM HC] (X2 and X4 prefer LIEM-S as the alternative; preserved as dissent).
6. **Proposed batch verdict: DESIGN_READY_FOR_HUMAN_REVIEW**, with an explicit conversion rule: under the current canon (no certified self-coupling) **and** HD-1 = "an estimator is not learned transport" **and** no GM2, the correct verdict is **NO_DEFENSIBLE_CANDIDATE** (scoped). The design is "ready" in the sense that every remaining branch is decided by named human decisions and by learning-free gates that are specified below; it is not ready for adoption, and every surviving candidate is predicted more likely to stop than to pass.

---

## 1. What the batch actually established (J1's reading of the evidence)

### 1.1 The three WL97 problems, kept apart

| problem | what is measured | what it implies for a design | who addresses it |
|---|---|---|---|
| P1 learned-sender training / fit allocation | learned S1 sender far from physical at CK2 (field rel L1 1.08 / 1.19 vs physical q; box-A 0.52× / 1.63×, seed-signed) [MEAS, AUDIT P0-1a]; grazing ring-0 in 4 of 3,712 lightings (0.11 %) [MEAS B-04]; rank-16 exact on CK2 support but all-incidence capacity rank 46–59 for 10 % [AUDIT J1 / B-06]; class optimum vs the path-traced reference never computed [AUDIT P0-1b] | H1 is only testable against a **class floor computed against the reference**, at a matched incidence weighting, with a rank axis; otherwise allocation and capacity are confounded (X2 X-1) | ECRO-lineage diagnostic arms (all designs; X2's three-way split) |
| P2 fixed-bin grazing delivery bias | SA-mean 4×16 over-delivers +40–50 % at CK2 box A; about half is convention (FC ≈ +20 %); receiver-dependent sign (glint receivers 5–8× under); gone at 8×32; one configuration [MEAS + AUDIT P0-2] | the carrier fix is **learning-free**; any learned representation must beat FC 8×32 / adaptive FC and an exact-direction route, not the biased table | carrier floors (all designs) |
| P3 no headroom for resolvent-only | table space, oracle prefix, CK2 static; zero-cost ceiling 1.02 / 1.04; floor table has ≈ 0.010 rel-L2 readout error from patch-spatial averaging seen through the sharp reflection [AUDIT P0-4] | the bound is **role-specific** (D3 §2.3, X3 L1); it does not bind a full operator or a different amortisation target, but no GM1 role reaches 1.25 on the engineered kernel basis with a 64-spp frame | role-specific ceilings (D2 G0.3, D3 Stage 0, X3 L1) |

### 1.2 The discriminator: rule reuse versus state reuse

- Proposition I (D1) is correct but proves less than it seems. In any scene with known local physics, every transport quantity is a deterministic functional of physics + geometry. That holds for LIEM's media and RB-DTO's bodies too; learning there also only amortises (walks, bounces) [PRED].
- What differs is **edit invariance of the amortised quantity**:
  - GM1-convex: the amortised quantity (multi-bounce transport between bodies) depends on relative pose, so it is regenerated after every edit. Only the rule is reused. Its utility is capped by regeneration cost vs explicit work [ARITH D3 / X3].
  - Rigid non-convex bodies / rigid media: the amortised quantity is a function of one body's own geometry in its own frame. It is reusable state under rigid motion, which is what `Architecture.md:56, 70` calls preserving certified-valid computed state.
- Consequence for the brief's hypothesis: "reused after a geometry edit without retraining" has two readings. The rule-reuse reading is predicted to give no geometry-edit utility at GM1 scale (testable learning-free, §5 Stage 1c). The state-reuse reading needs transport that is both sharp and edit-invariant, which GM1-convex does not contain. This is why every defensible candidate needs either a workload extension or a scope / canon decision.

### 1.3 Tension with the recorded human decision

The 2026-11-13 decision makes ECRO the central base and asks for "one falsifiable ECRO-lineage hypothesis" [HUM, `RESEARCH_ROADMAP.md:128`]. The design evidence predicts H3 supported at ceiling for every ECRO-lineage role at GM1 scale on the engineered kernel basis (D2 §4, D3 §2.3, X3 L1) [ARITH, PRED]. The ECRO lineage can still deliver the H1 / H2 attribution the brief asks for. Whether a non-ECRO-lineage candidate (RB-DTO-C, XQT) may be the follow-up's candidate is a human decision [HUM HD]. One bridging reading, for the owner only: RB-DTO-C is the ECRO idea (geometry-conditioned, source-independent, linear learned operator over explicit relations) *scoped to a rigid sub-scene so that its output becomes certified reusable state*. J1 does not claim this makes it ECRO-lineage.

---

## 2. Candidate ranking (merged across designs)

Scores are J1's 0–10 judgement of defensibility and falsifiability under this brief. None is adoption.

| rank | candidate (merged) | sources | score | survives? |
|---|---|---|---|---|
| 1 | **RB-DTO-C** certified rigid-body sharp self-transport operator | D3 RB-DTO (primary); same class as D2 R-D and D1 O1 (both rejected on canon); twins from X4 MA-1, X2 §3.10, X3 §4.4 | 5.0 | CONDITIONAL on HA (canon), workload extension, learning-free gates |
| 2 | **LIEM-S** learned intrinsic exit measure (media) | D2 primary; X1–X4 conditional | 3.5 | CONDITIONAL on GM2 scope, reference qualification, novelty vs Vicini 2019 |
| 3 | **XQT** learned flux-conserving quadrature / estimator of the flight | D1 primary; X1 conditional, X2 conditional, X3 conditional, X4 no | 3.5 | CONDITIONAL on HD-1; predicted Stage-1 stop |
| 4 | ECRO-lineage arms (ECRO-COV, ECRO-SUP, ECRO-FC as carrier-only, ECRO-R) | DR ECRO-FC; D1 C1 / C1f; D2 G2; D3 G1 | 2.0 as candidate | NO as candidate; YES as factorised H1 / H2 diagnostic |
| 5 | LIEM-B hierarchical categorical exit distribution | D2 alternative | 2.0 | control only (not mechanism-independent) |
| 6 | FLS flux-allocated spatio-angular lobe state | DR primary; same class as D1 C1, D2 R-C, D3 §3.5 | 1.5 | NO (learned part); ELT-shear twin kept as explicit comparator |
| 7 | LLT learned lobe transfer from sparse probes | D1 alternative | 1.5 | NO; its collision-pair information test is grafted |
| 8 | LER-Δ learned environment resolvent + explicit Δ | D3 alternative | 1.0 | NO |
| — | learned guiding (sampling density only) | missed alternative in X1 §5, X2 §5.1, X3 §6.4, X4 MA-3 | n/a | outside "representation"; scope ruling [HUM HH] |

### 2.1 RB-DTO-C (rank 1)

- **Physical soundness (6).** Linear per incident sample; first local event exact ER; learned part is the body's ≥ 1-self-hit exit measure, genuinely nonlocal (convex-body N_b ≡ 0 is a falsifiable contract). Defects found by critics and fixed in §3.1: passivity budget depends on an exact escape integral E_esc (X1 §2.3); per-patch mixture reduction is nonlinear (X1 §2.2, X3 L5); partial intrusion forces whole-body fallback (X1 §2.4); sharp exits delivered by receiver-anchored gather re-create the P3 heavy tail (X3 §4.4).
- **Falsifiability (7).** The decisive unknowns are learning-free: (i) the explicit intra-body share of the lifecycle against the *best* explicit twin (cost anatomy U0); (ii) the per-body oracle floor against a compressed precomputed transfer twin at matched memory / apply time. Both can stop it before training. Training then has a clean B2 (per-body oracle) / B5 (per-asset fit) / B4 (shared, designed coverage) / B3 (shared, natural coverage) ladder that separates H1 (coverage), capacity and shared-model generalisation.
- **Learned value (4).** Against Twin 1 (explicit sweeps with cached rigid self-relation and sender-side chain tracing) it amortises J_int sweeps of iterated glossy transport whose explicit cost grows with α⁻² or with branching cubature [PRED]. Against compressed precomputed per-body transfer (PRT / radiance-transfer-field class), both reuse state exactly; learning buys only no per-asset precompute, memory and continuity, and the learned apply is likely slower per sample (X3 L2: 1–20 ns vs 0.4–0.8 ns per table entry) [ARITH X3]. That is a narrow niche.
- **Edit reuse (8).** Exact state reuse under rigid motion of the body; stationary bodies keep state when only their incident field changes. This is the only candidate where a T0 → T1 test exercises reused learned state rather than reused weights (X4 X-3).
- **Cost plausibility (3).** D3's regime map (U 4.8–48) used a uniform-binned twin and placeholder t_q; X3's symmetric recount of D3's own 20-body example gives U 0.16–1.21, break-even J_int · K_int ≳ 790–21,000 per patch [ARITH X3]. Plausible only for deep, high-albedo glossy cavities or many instanced bodies.
- **Novelty (3).** PRT self-transfer, radiance transfer fields, shadow fields, OSF, TransGI, MRT are close (X4 §6.1). The only open niche is sharp (α ≈ 0.03–0.15) self-transport + one shared compiler + a passive measure contract.
- **Decisive unknown.** Is there any admitted workload where the best explicit intra-body twin is a large lifecycle share (U0 ≥ U_min) **and** a per-body learned oracle beats the compressed precomputed twin at matched memory and apply time?

### 2.2 LIEM-S (rank 2, tied)

- Strongest learning construction in the batch (designed impulse measure as a treatment, event-likelihood supervision, exact rigid equivariance, canonical K ownership, passive by construction) [X2 §3.6].
- But it answers a different question: the sharp exit lobe is the *exact* interface closure, and D2's own slab toy puts learned angular structure at noise exactly where walks are long enough to amortise (27–50 events) [SCRATCH D2 §5.2]. That is close to the "sharp to exact physics, learned smooth residual" pattern the brief rules out as a default (X4 S-LIEM-1).
- Its geometry edit cannot discriminate it from Twin-T (any rigid-invariant K is reused exactly; X2 X-9); its kill rule passes on memory by construction (X1 #14, X3 §4.3).
- No qualified reference for refractive-interface media (WL53 index-matched only; Mitsuba float32 boundary layer; LabServer63 Mitsuba cuda film wrong) [MEAS WL53, memory note].
- GM2 depends on an interpretable GM1 result (`RESEARCH_ROADMAP.md:17`) [CANON].
- **Decisive unknown.** Does any admitted medium configuration have both long walks (Twin-W cost) and learned angular / spatial structure above MC noise (D2 I0.2), with a qualified reference?

### 2.3 XQT (rank 2, tied)

- Cleanest canon compliance in GM1: linear, flux-exact, rigid-invariant inputs, ER exact, visibility only from explicit records [D1 §4.1].
- Defects: passive only in expectation, not per relation draw (X1 §1.3: P(delivered > emitted) up to 0.16–0.40 in a 0.46-hit Poisson toy); neighbour redistribution infers point-level visibility from element-level co-visibility (X1 §2.4); per flight 8–26× an explicit order (X3 §4.2); D1's S1-c oracle sees the field while W does not, inflating headroom (X2 §3.2).
- Geometric argument against its lever on CK2-type glints: lobe footprint ≈ 3.5 cm vs 0.2 m patches, neighbours ≈ 11° away from a ≈ 1° lobe, so pooling either moves glint flux to where the lobe does not reach or pools nothing (X1 §2.6) [PRED]; D1's flatland scratch gives oracle / twin only 1.0–1.33 [SCRATCH D1].
- It is an estimator: the brief's criterion "the learned operator does not substantially replace explicit transport" applies unless the owner rules otherwise (HD-1). Prior art is learned MC kernels / deep KDE (X4 §4.1).
- **Why J1 still keeps it as the alternative.** Its decisive learned-vs-twin test is the cheapest and most cleanly separated in the batch: learning-free, on existing GM1 workloads and existing exact-sample machinery (C3URX1, A-04 deterministic oracle), with common random numbers and a pre-named twin. A clean negative here closes the last in-canon GM1 learned lever (estimation) with evidence rather than argument; a positive would be a real surprise worth knowing.
- **Decisive unknown.** Does a field-independent, per-geometry optimal kernel (the honest bound for a linear W) beat the *strongest* learning-free twin (edge-aware analytic kernel, spatial reuse with exact re-evaluation, sender-side deposition, MIS) at matched rays in the sharp ROI by a declared margin?

### 2.4 Lower-ranked candidates

- **ECRO-lineage arms (DR ECRO-FC; D1 C1 / C1f; D2 ECRO-COV / -FC; D3 G1).** NO as a geometry-edit candidate: per-patch "local" factors depend on current form factors, world pose and a scene token [CODE], so no local validity certificate exists and every factor is regenerated per edit; the regeneration costs ≥ the replaced explicit work on the kernel basis [ARITH D3 / X3]. YES as diagnostic arms, but only factorised (coverage only; physical S1 target only; carrier only with matched state rays per bin; rank axis) and read against weighting-matched class floors (X2 X-1). J1's nuance: the generator receives *explicit* current visibility as input, so this is not learned visibility; the defect is that the representation playing the intrinsic role is not intrinsic, which breaks the brief's ownership separation.
- **LIEM-B.** Representation control for LIEM-S; not mechanism-independent (D2 says so); autoregressive per-sample cost 16–160 ns vs a 0.7–13 ns walk (X3 §4.3).
- **FLS.** J1 agrees with all four critics: element-local learned map = ER × deposits + K-lobe reduction (X2 §2.3); separable lobes cannot hold the sheared point-light glint at CK2 incidence (X1 §1.1); radiance formula ambiguous near the horizon (X1 §1.2); the NULL sink is a learned absorption beyond exact ER; superposition lost, which forfeits the only measured learned lever (source-only update 3.5–3.9× [MEAS WL87]); no reusable state after an edit (X3 F-FLS-2); learned-vs-ELT headroom tested only after training. Its learning-free twin (ELT, upgraded with image-source / sheared lobes, VSL / VSGL class) is a valuable explicit comparator.
- **LLT.** Either its probes determine delivery (then the photon twin has the same information) or it infers unprobed visibility (learned C; WL52 failed at transitions) (X2 §3.3). Its collision-pair information test is the graft (with X4's caveat that pairs must not be drawn only from the training generator).
- **LER-Δ.** Per-environment learned state (Native Reuse class), signed out-of-span Δ-fields (WL41 / WL44 failure mode), and a mis-stated convergence condition: the iteration matrix is (I − T_E)⁻¹(M − B), not H_E(M − B) (X1 §1.4).

---

## 3. Primary and independent alternative

### 3.1 Primary: RB-DTO-C (corrected RB-DTO)

**Learned physical meaning.** For a rigid body b in its own frame: the non-local part N_b = (I − K_b C_self,b)⁻¹ K_b − K_b of its response, i.e. the exit measure (exit point × outgoing direction, per unit incident flux) of light that leaves the body after ≥ 1 self-hit, where C_self,b is free-space transport that leaves and re-enters b without meeting foreign geometry. One shared compiler maps the body mesh (+ ER descriptors) to per-body latents once; an apply network maps each incident sample (x_i, ω_i) to a passive pushforward.

**Corrections J1 requires before review (merged from X1–X4):**
1. **Linearity kept.** No incident-field-dependent per-patch mixture reduction in the state. The output is a per-incident-sample pushforward (linear in Φ_i); any caching is a linear operator per incident cell. This keeps superposition, impulse validation and the source-update lever (X1 §2.2, X3 L5).
2. **Exact budget.** E_esc (first-event escape without self-hit) is computed by explicit rays against the cached self-relation, and that cost is charged to the method (X1 §2.3, X2 §3.10 #7). The output simplex is {sharp lobes, broad channel, absorbed-by-exact-ER}; no free learned NULL beyond the exact budget.
3. **Sharp exits delivered sender-side** (deposition from the exit lobes into current-geometry C_ext), not by receiver-anchored gather (X3 §4.4).
4. **Whole-body fallback** on any certified-region intrusion; costed (X1 §2.4).
5. **Self-visibility explicit and cached** (rigid-invariant), shared with the twin; if any self-visibility is inferred, a collision-pair information test is a Stage-1 kill gate (X2 §3.10 #1, D1 S1-d).
6. **Strongest twins** (X2, X3, X4): Twin 1 = explicit intra-body sweeps on the cached rigid self-relation with sender-side chain tracing; Twin 2 = compressed precomputed per-body transfer (wavelet / SG / CPCA class) built from the same exit samples. Uniform-binned twins are not admissible comparators.
7. **Measured prior art** (X4 X-1): compressed per-body transfer is a measured arm; OSF / TransGI-type per-object neural assets are Native Reuse comparators, measured or explicitly UNMEASURED with a reason.

**Ownership [HUM HA].** RB-DTO-C places C_self,b segments inside a learned body operator under a geometric dependency certificate. `Architecture.md:25` forbids *object identity* freezing exposed interreflection; J1 reads the certificate (bit-identical self-relation inputs in the body frame, no foreign primitive in the self-path region, else whole-body explicit fallback) as the same validity principle as the F / S contract (`Architecture.md:70`), not identity freezing. That is J1's reading only; X1 (and D1 O1, D2 R-D) read it as a canon conflict. The owner decides.

**Workload [HUM HC].** This is a GM1 workload extension (non-convex glossy rigid bodies in the room), not GM2: GM1's question is sharp multi-bounce editable-visibility interreflection under rigid edits, which glossy cavities contain. Admission must include an adversarial workload (moderate self-transport, frequent intrusion), because admission rules that select for the lever bias the result (X4 D3-P2).

**Predicted outcome [PRED].** More likely to stop at Stage 1 (U0 below threshold against the best explicit twin, or no headroom over compressed transfer) than to pass. That is acceptable: the stop is cheap and learning-free, and either outcome is informative about the state-reuse reading of the hypothesis.

### 3.2 Independent alternative: XQT (conditional on HD-1)

Independent of RB-DTO-C in regime (GM1 inter-object flights vs intra-body), ownership (C's estimator vs certified self-coupling), reuse type (rule + certified W entries vs body state) and failure mode (estimator headroom vs regime / generalisation).

**Corrections required:** "passive by construction" reworded to "passive in expectation" with a per-draw passivity test (X1 §1.3); point-level visibility consistency (shadow ray at the reassigned point) or the leak measured (X1 §2.4); field-independent oracle (X2 §3.2); strongest twins (X1 §5, X3 §6.2, X4 D1-P1/P2); S1-e ceiling defined as U(1 draw + XQT) vs U(m pooled draws) at equal quality with the twin at equal cost (X3 §5.3); steady-state draw pooling allowed to the explicit arm (X3 L4); J4 readout bias at receivers seen through the sharp reflection reported separately (X2 §3.2 #3).

If HD-1 = "an estimator is not learned transport", XQT is out and the alternative is **none** in GM1. LIEM-S is then the reserve only if GM2 is opened.

---

## 4. Grafts

- **D1:** flux-exact projected-disk carrier rules M1–M3 (store flux per projected-measure cell; never tabulate sharp outgoing radiance; emit / read at the exact point); event-count ownership partition; dependency certificate tested as selective = full; contract tests T1–T10; W-G2 admission on the deterministic delivered sharp difference at glint-core *and* tail receivers; collision-pair information test (S1-d).
- **D2:** principles P1–P8 (learn only what is expensive; condition only on edit-invariant inputs; designed coverage measure as a treatment; unbiased physical targets, never M64; energy as an output-family property; disjoint supervised path families; name the twin before training; held-out by factors); impulse supervision with a grazing quota (applied to RB-DTO-C incident samples); ECRO-R fresh-seed arm as the seed pilot; G0 stop rule (no non-learned-C role reaches U_min → no ECRO-lineage training for architecture purposes); G1 class floors.
- **D3:** Stage-0 cost anatomy before training; role-specific ceilings; path-tag ownership identity test; convex-body N_b ≡ 0; certificate-violation control edit; B2 / B3 / B4 / B5 attribution ladder; Track G / Track B separation.
- **DR (coordinator):** ELT as the learning-free lobe twin, upgraded with image-source shear (VSL / VSGL class) and used as an explicit comparator; clipped-hemisphere measure lesson; contract tests T-a … T-r style; sealed held-out discipline.
- **X1:** reference-instrument qualification gate (exact-direction probes, cos-weighted tallies, incident flux per cell, self-path tags, matched event truncation, CK2 T1 reference); per-draw passivity and spectral-radius tests for every stochastic explicit arm on glossy loops; radiance-measure definition for any lobe family.
- **X2:** three-way H1 attribution (optimisation vs allocation vs operator capacity, with a rank axis and a light-conditioned control); fit on reference half A, score on half B; seed-power pilot that sets n (3 seeds detect only ≈ 2.4× at WL93D-sized spread, ≈ 11.6× at the OWN box-A spread) [SCRATCH X2]; oracles restricted to the learned model's information and linearity; exposure-bias metric; per-instance fitted arm for every shared candidate; held-out with a declared distribution shift.
- **X3:** quality-matched utility in GM1 is decided by the table floor seen through the frame (whole-image tiers identical across table arms within 1–2 %; R_mirror floors 0.062 / 0.045 / 0.030) [ARITH on MEAS WL91]; predeclared tiers away from every floor with a NOT-REACHED rule; progressive draw pooling for explicit arms; unified kernel timeline; compile / recompile accounting; engineered explicit comparator; multi-light axis (1 / 4 / 16).
- **X4:** closest prior art as measured comparators; novelty niche stated in advance (sharp + shared + passive contract); external Native Reuse baselines measured or UNMEASURED.

---

## 5. Merged minimal falsification plan

**Principle.** Learning-free gates first; every training stage is conditional on a learning-free result that makes it informative; every stop is final for the batch (no rescue). Numerical thresholds are [HUM]; none is invented here.

### Stage 0 — Freeze, instruments, admission (learning-free, mandatory)

- **Reference instruments** (X1 §2.1): exact-direction outgoing probes (pt86 probe mode) qualified against the deterministic exact-ER first-flight quadrature (A-04 / E3 class) and against existing binned tallies where they overlap; cos-weighted cell tallies; incident flux per receiver cell; event-count and segment-owner (self-path) tags; depth matched to every arm or deepened with the truncation residual reported; independent reference halves. Fail → `STOPPED_REFERENCE_UNQUALIFIED`.
- **Workloads** (admitted on geometry-only / learning-free rules before any arm output exists):
  - GM1 Track G: W-G2 grazing sharp (admission on delivered sharp difference at glint-core and tail receivers, not on sender irradiance; X2 X-7), W-N non-grazing sharp, W-H independent held-out with a declared shift (another geometry family or a never-inspected partition), CK2 diagnostic only and excluded from gate arithmetic (needs a T1 reference if used).
  - Track B (only if HA): W-Bg grazing internal incidence into a glossy cavity, W-Bn non-grazing, W-Bh held-out body class and light class, W-Badv adversarial (moderate self-transport, frequent intrusion).
  - Edits for every evidence workload: rigid occluder translation that enables / blocks a sharp path, rigid motion of a sharp object / body, and (Track B) one certificate-violation control. Path-traced references at T0 **and** T1.
  - Change SNR on an independent qualification partition (WL86 precedent 4×, value [HUM]). Fail → `STOPPED_UNQUALIFIED_WORKLOAD`, no substitution after inspection.
- **Train / validation / test** separated by geometry family, body class, light class and margins, applied to every trained arm (WL97 B-02 lesson). No T1 data and no evaluation-workload data in training or selection.
- **Timing harness**: unified GPU-busy timeline plus wall; compile / recompile counts per update; engineered explicit comparator; full-scale probe of every stage for wall and GPU utilisation before the freeze.

### Stage 1 — Learning-free decomposition (decides whether anything is trained)

- **1a H2-carrier (GM1).** Floors vs the reference for SA 4×16, FC 4×16, FC 8×32, adaptive FC, pooled exact per-ray, ELT-shear; per receiver class (glint core, tails, extended); fit on half A, score on half B; cost recorded.
- **1b H2-operator class / H1 prerequisite (GM1).** Delivery-optimal patch × 64 operator vs the reference (P0-1b missing evidence); light-independent non-negative rank r ∈ {16, 48, 64} under the declared incidence weightings; one light-conditioned control. Reading: if no admissible weighting meets the target at rank 16, ECRO's sharp failure is operator-capacity-limited (H2-op) and coverage training is moot.
- **1c H3 ceiling (GM1).** Role ceilings (resolvent-only; full table operator with resident factors; full operator + per-edit generator) on the unified kernel basis and wall, N ∈ {1,270, ~5k, ~11k}, lights {1, 4, 16}, frame / no frame; quality-matched through per-arm spp ladders at predeclared tiers with draw pooling. **Stop rule:** no role that does not learn C reaches U_min at zero cost at matched quality on ≥ 2 admitted workloads → H3 SUPPORTED at ceiling for the ECRO lineage on GM1.
- **1d XQT gate (only if HB = yes).** Field-independent per-geometry optimal kernel (fit over the declared source distribution on half A) vs the strongest twin set (edge-aware analytic kernel; spatial reuse with exact re-evaluation + shadow ray; sender-side deposition; MIS of gather and deposition), CRN, ≥ 16 draws, sharp ROI, equal cost. **STOP XQT** if the margin is below m_req [HUM] on two evidence workloads, or if U(1 draw + XQT) / U(m pooled draws) is below its threshold [HUM].
- **1e RB-DTO-C gates (only if HA = yes).** Anatomy U0 = L / (L − A_intra) with Twin 1 as the explicit comparator; **STOP_NO_HEADROOM** if U0 < U_min on the admitted Track B workloads. Then per-body oracle floor (direct fit to tagged exit samples, half A / B) vs Twin 2 (compressed precomputed per-body transfer) at matched memory and apply time; **STOP_NO_LEARNING_HEADROOM** if the oracle floor does not beat Twin 2 beyond paired noise. Collision-pair test if any self-visibility is inferred.

### Stage 2 — H1 attribution of the ECRO lineage (training; conditional)

Runs only if 1b shows that some admissible weighting of the ECRO operator class meets the target on the admitted workloads, or if the owner wants the attribution for the record [HUM HD]. Arms share the scene86 generator family, budget and reference; frozen WL86 / WL92A checkpoints are a historical reference point (shared reference only).
- Pilot: ECRO-R (unchanged recipe) with ≥ 5 fresh seeds → seed SD on the gate metric → n per arm for the declared minimum effect.
- ECRO-COV (coverage only), ECRO-SUP (physical S1 target only: path-traced or validated quadrature, never M64), ECRO-FC (carrier only, no passivity constraint, matched state rays per bin), optional ECRO-PASS.
- Readings (X2 X-1): learned vs weighting-matched class floor = H1-optimisation; floor shift with weighting = H1-allocation; floor across weightings and ranks = H2-operator; Stage-1a floors = H2-carrier.

### Stage 3 — Candidate training (survivors of Stage 1 only)

- RB-DTO-C: B3 shared, natural coverage; B4 shared, designed impulse measure with grazing quota; B5 per-asset fit; same exit-sample generator and budget; seeds from the pilot.
- XQT: trained on the same relation draws used at runtime (no exposure gap) with noise-free deterministic targets.
- Contract tests on the production code before any evaluation: zero input, superposition, non-negativity; passivity per draw and in expectation; composed spectral radius with a valid estimator; rigid equivariance to float tolerance; convex-body N_b ≡ 0; path-tag disjointness; certificate invalidation; furnace; reciprocity measured. Failure → `STOPPED_CONTRACT`, no repair in the batch.
- Exposure-bias metric: learned-map error on reference inputs vs pipeline-delivered inputs, per sweep.
- **H-L stop:** learned ≤ twin within paired noise on ≥ 2 admitted workloads → learned part decorative.

### Stage 4 — Fixed-weight T0 → T1 (no retraining, no fine-tuning)

All surviving arms + STALE + explicit comparators at quality-matched budgets, on W-H / W-Bh (held-out) and the other evidence workloads. Evidence: angular response fidelity at exact probes; peak / lobe position and width; spatially distributed error per receiver class; energy and passivity; sampling variance; image error vs reference; sharp-interreflection ROI; changed-region fidelity and signed-change slope / correlation; leakage / ghosting outside the predicted change support; extra T1 − T0 error; selective = full for every certificate. External baselines: Native Reuse measured or UNMEASURED with a reason; Neural Full Recompute only where a representation is actually invalidated; path tracing as reference only.

### Stage 5 — Economics

Time-to-tier from the edit (explicit arms may pool draws progressively), steady-state frame, learned inference, relation / visibility recompute, certificate and fallback cost, memory and precompute, training as one-time cost; utility U vs the quality-matched explicit comparator; scaling over body count, instancing, J_int and α (Track B) or N and lights (Track G). A positive claim needs two admitted workloads including the held-out one.

### What can and cannot share data, budget and reference

| comparison | training data | budget | physical reference |
|---|---|---|---|
| frozen ECRO vs any retrained ECRO arm | no (historical) | no | yes |
| ECRO-R / COV / SUP / FC | same generator family; COV changes the sampling measure (the treatment) | yes | yes |
| learning-free floors / explicit comparators vs anything | n/a | matched cost or quality points, not training budget | yes, with CRN state draws |
| XQT vs its twins | identical inputs and draws | twin untrained; equal runtime cost | yes |
| RB-DTO-C B3 / B4 / B5 / B2 / Twin 2 | same exit samples (B2, Twin 2 built from them) | matched sample budget and memory | yes (Track B references) |
| Track G vs Track B | **no** | **no** | only renderer, metric definitions, statistics and timing conventions |

### Statistics (binding)

Paired evaluation with common random numbers; block-bootstrap SE over pixels / patches; independent reference halves with debiased L2; seed as a random effect, n from the pilot; ≥ 16 draws for every draw-dependent arm, no label from one draw; never an L1 noise floor as a paired tolerance; per-geometry tables and heavy-tail guards; fit / score split halves for every oracle and floor; decision rules frozen before any result; amendments append-only.

### Synthetic vs real

Synthetic contract tests (analytic, noise-free) certify measure, energy, linearity, equivariance, ownership and certificate behaviour, and catch compensating errors that scene metrics hide (WL86A). Real-scene evaluation is the only place where coverage of scene-induced incidence, composition, readout, edit response and cost exist. Neither alone can approve a candidate.

---

## 6. Human decisions required

- **HA. Canon:** may a rigid body's free-space self-paths C_self be owned by a learned body operator under a geometric dependency certificate with whole-body explicit fallback (D3 D1)? Without it, RB-DTO-C is inadmissible.
- **HB. HD-1:** is a learned estimator over explicit visibility samples (XQT) admissible as the project's learned transport, or excluded by "the learned operator does not substantially replace explicit transport"?
- **HC. Workloads / scope:** add non-convex glossy rigid bodies to GM1 (Track B)? Open GM2 media (LIEM-S) now, despite GM2's dependence on an interpretable GM1 result?
- **HD. Lineage:** given the 2026-11-13 ECRO-central decision, may the follow-up's candidate be non-ECRO-lineage? Do ECRO H1 training arms run for attribution if Stage 1b / 1c already decide them?
- **HE. Numbers:** U_min; quality tiers and ROI definitions; m_req and the learned-vs-twin margin; η_req; admission values (SNR, sharp share, self-transport share, grazing share); passivity / reciprocity tolerances; minimum meaningful image effect; leakage and extra-T1 limits; coverage quota; seed count via the pilot and draw counts.
- **HF. Conventions:** kernel vs wall basis; engineered explicit comparator required; frame spp; draw-pooling horizon; multi-light axis; edit → frame vs edit → valid table.
- **HG. External baselines:** Native Reuse (OSF / TransGI / 8DNA-type per-object asset) measured or UNMEASURED; Neural Full Recompute only where invalidated.
- **HH. Learned guiding** (sampling-density-only learning): explicitly in or out of scope.
- **HI. Novelty:** accept the narrow RB-DTO-C niche (sharp + shared + passive contract) as worth testing against PRT / OSF / TransGI / MRT prior art.
- **HJ.** Batch class (regular vs suffix) and compute / server.

---

## 7. Dissent the final synthesis must preserve

1. **X2 and X4:** batch verdict NO_DEFENSIBLE_CANDIDATE under the current canon and scope; no candidate has demonstrated headroom over a fair twin.
2. **X3:** INSUFFICIENT_EVIDENCE; RB-DTO's regime is narrow (symmetric recount U 0.16–1.21 in D3's own example; break-even J_int · K_int ≳ 790–21,000 per patch).
3. **X1:** RB-DTO is NO under the current canon; within canon only XQT is physically defensible.
4. **D2 vs D3:** D2 rejects rigid-cluster self-transport (its twin is the cached explicit self-relation, equally rigid-invariant; learning buys only sweep amortisation); D3 makes it primary. J1 sides with D3 on *which* candidate to test and with D2 on *what it must beat*.
5. **Unanimous (D1–D3, X1–X4):** no rigid-reusable learned object in GM1 planar / convex scenes; FLS NO as a learned candidate; ECRO-FC NO as a candidate.
6. **X2 and X4 prefer LIEM-S** as the independent alternative; J1 chose XQT for falsifiability and canon / GM1 fit.
7. **Coordinator conflict of interest:** DR's author ran WL86–WL96A; DR §10 points to `CRITIQUE.md` and a WL98 worklog that do not exist in the directory (X2 X-13, X4 DR-P3).
8. **All four critics** name learned guiding as a missed alternative.
9. **Tension with the 2026-11-13 decision:** the design evidence predicts H3 supported at ceiling for every ECRO-lineage role at GM1 scale on the engineered kernel basis.
10. **J1's own:** the discriminator is edit invariance of amortised state, not information beyond explicit queries; Proposition I applies equally to LIEM and RB-DTO.
11. **D1:** if HD-1 = no, D1's own verdict for GM1 becomes NO_DEFENSIBLE_CANDIDATE except LLT, which J1 and three critics reject.

---

## 8. Proposed batch verdict

**DESIGN_READY_FOR_HUMAN_REVIEW** (not adoption, not experiment approval).

- One primary (RB-DTO-C) and one independent alternative (XQT) have complete, corrected contracts; each is conditional on one named human decision (HA, HB).
- The merged plan separates H1 (factorised ECRO arms read against weighting-matched class floors), H2 (carrier floors and operator-class floors with a rank axis), H3 (role-specific learning-free ceilings, then quality-matched fixed-weight T0 → T1) and learned-vs-twin (learning-free oracle-vs-strongest-twin gates **before** training), and keeps CK2 diagnostic and M64 out of truth.
- **Conversion rule:** if the owner rejects HA, answers HB "no" and does not open GM2, the verdict for this batch is **NO_DEFENSIBLE_CANDIDATE** (scoped to scene86 / GM1 and the current canon). This is not a class-level falsification of reusable learned transport (WL97 P0-5b).
- **Prediction carried with the verdict [PRED]:** each surviving candidate is more likely to stop at a learning-free Stage-1 gate than to reach training.
