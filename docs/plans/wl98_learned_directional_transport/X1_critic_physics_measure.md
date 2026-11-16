# WL98 Critic X1: physics, measure and ownership (adversarial)

**Date.** 2026-11-16. **Role.** Independent adversarial critic X1 of the design-only batch WL98. Lens: energy and measure contracts, flux bookkeeping, passivity under composition, linearity, reciprocity, ownership (does a learned part own visibility or inter-object geometry?), rigid transforms, and whether validation against independent path-traced references is possible with the repository's machinery.

**Status.** Design critique only. Nothing was implemented, trained or rendered. No existing file was changed. No commit. This report resumes an interrupted attempt; that attempt left one scratch script and no report. Its decisive number was re-derived and corrected (§1.1).

**Inputs read in full.** `DESIGN_REVIEW.md` (coordinator draft: FLS + ECRO-FC), `D1_physics_measure_first.md` (XQT + LLT), `D2_learning_generalization_first.md` (LIEM-S + LIEM-B), `D3_systems_cost_first.md` (RB-DTO + LER-Δ).

**Code read for facts.**
- `experiments/ecro86/model86.py` lines 40–126: generator inputs and `apply`.
- `experiments/ecro86/pt86.py` header and `render` / `field` signatures.
- `experiments/ecro86a/pt86a.py` lines 1–140: `field_src`.
- `experiments/ecro86/scene86.py`: panel split.
- `experiments/ecro86/protocol/ecro86.json`: fixture and light sets.
- `experiments/ecro_local93d/addendum_checker/scene_ck2.json`: CK2.

**WL97 claims verified.** `J2` (+19–21 % flux-conserving remainder; 8 × 32 gives 1.024 at box A) and `COORD` P0-3 (≥ 16× draws).

**Labels.**
- [MEAS] measured, with its source.
- [AUDIT] a WL97 interpretation.
- [CODE] read from frozen code.
- [CALC] X1 numpy scratch. Design input only, not evidence.
- [PRED] my prediction or argument.
- [HUM] a human decision.

---

## 0. Summary

1. **The coordinator's primary (FLS) does not survive as a learned candidate (NO).**
   - As specified (separable spatial anchor × directional lobe, K ≤ 8 per 0.2 m element), it cannot hold the reflected image of a point light on the planar S1 panel at CK2 incidence. The mirror direction sweeps about 18° (E-weighted, up to 28°) across one panel patch, while the S1 lobe at that incidence is about 1° wide. That needs about 10 separable lobes per element per light, and 79 % of the panel irradiance falls on patches that need more than 8 [CALC §1.1].
   - A sheared (image-source) lobe fixes this with one lobe, but that lobe is the analytic mirror construction. There is nothing left to learn for direct light.
   - FLS's radiance formula is measure-inconsistent near the horizon (§1.2).
   - Its NULL sink lets the learned part delete flux that ER says is scattered.
   - It breaks superposition.
   - In GM1 its only learned content is mixture reduction of exactly known ER lobes. D2 already showed that generic lobes cannot represent the grazing S1 lobe (lobe L1 0.35–0.6 at K ≤ 4).
2. **ECRO-FC does not survive as a candidate (NO). It survives as a diagnostic arm (YES).** It keeps the generator, whose "local" factors are conditioned on current visibility (openness and form-factor message passing), world coordinates and a scene-global token [CODE `model86.py:93-104`]. That is dynamic visibility coupled into the per-patch response, which is a brief stop condition. As an arm it is confounded, because it changes carrier, coverage and passivity at once.
3. **Cross-cutting physics findings that bind several designs.**
   - **(a) Receiver-anchored gather estimators are passive only in expectation.** This covers XQT, EXP / C3UR and C3URX1. Sender-anchored deposition is passive per realisation. In a toy anchored on WL96's 0.46 expected glint rays per receiver per draw, a single draw delivers more flux than the sharp sender emits with probability up to 0.16–0.40 [CALC §1.3]. D1's "passive by construction" claim for XQT is false per draw.
   - **(b) The existing path-traced field references cannot validate sub-bin claims.** `pt86a.field_src` tallies solid-angle-mean radiance in exactly the 4 × 16 bins whose convention WL97 faulted. Its event / class split is e2 / e3 × last-sender panel / other, and all references are depth-truncated: kmax 5 for fields, D 6 for the camera [CODE]. None of the following is measurable with the existing tallies:
     - lobe peak position and width;
     - flux-conserving (cos-weighted) cells;
     - incident flux per receiver cell;
     - body self-path tags;
     - media;
     - all-order transport.

     `pt86.trace` already has a probe mode that takes an arbitrary (y, w). Exact-direction probes are therefore a small extension, but one that must be qualified.
   - **(c) Learned-minus-explicit cancellation produces ghosting by construction** (LER-Δ). D3's convergence condition for LER-Δ is also wrong (§1.4).
4. **Survivors (all conditional).**
   - **XQT (D1), CONDITIONAL.** Conditions: HD-1 must admit a learned *estimator* as "learned transport"; the plan must add a stronger learning-free twin; and the per-draw passivity claim must be corrected. Its own author's scratch predicts that it stops at Stage 1. My geometry argument (§2.6) predicts that its pooling lever vanishes for CK2-type glints.
   - **LIEM-S (D2), CONDITIONAL.** It has the cleanest physics contract of all designs. Conditions: a scope decision (GM2 media), a reference qualification that does not exist yet, and a headroom admission. It does not learn interreflection at all.
   - **RB-DTO (D3), NO under current canon.** It puts editable same-asset air paths into a learned body operator, which `Architecture.md` §2 forbids and the brief tells us to keep. It is CONDITIONAL only if the owner changes canon.
   - **LLT (D1), CONDITIONAL only as an information test.** It is not a candidate.
   - **LER-Δ (D3), NO.** It has a wrong convergence contract, inherent cancellation ghosting, and scene-baked learned state.
   - **LIEM-B, CONDITIONAL as a representation control only.**
5. **Proposed verdict: DESIGN_READY_FOR_HUMAN_REVIEW, with dissent.** It holds only if the batch's primary / alternative pair is *not* FLS / ECRO-FC.
   - Within the current canon and the GM1 workload family, the only physically defensible learned candidate is XQT, and its admissibility (HD-1) is a human question.
   - If HD-1 is answered "an estimator is not learned transport" and GM2 is not opened, the honest batch verdict is **NO_DEFENSIBLE_CANDIDATE**. All three designers' own GM1 analyses (D1 Proposition I, D2 §3, D3 §2.5) say the same.

---

## 1. Scratch calculations (design input only)

Scripts:
- `experiments/wl98_calc/X1/ck2_spread.py` (interrupted run, sha256 64bdadcb0618ca03) → `results/wl98_calc/X1/ck2_spread.json` (6392c9f21c6a1a49);
- `experiments/wl98_calc/X1/physics_checks.py` (489e4372492a4693) → `results/wl98_calc/X1/physics_checks.json` (c6c864bb8e59f226).

Local CPU, numpy, seed 98111.

### 1.1 Point light on the planar S1 panel: spatio-angular shear versus lobe width [CALC]

**Setup.**
- Panel: x = −1.5, normal +x, 1.2 × 1.4 m, split 6 × 7 into 0.2 m patches [CODE `scene86.py:38`].
- Lights:
  - CK2 at (−1.45, −0.6, 1.7), 5 cm in front of the panel plane;
  - L1 (1.4, −0.8, 2.5);
  - L2 (−0.9, −1.2, 2.3).
- Per patch, 8,192 points. At each point the specular (mirror) direction is computed via the image source. I report the irradiance-weighted principal extents of that direction set (uniform-equivalent full width 2√3·SD).
- The S1 lobe half-width at the patch's median cos_i is the cone-equivalent of D2's half-max solid angle, interpolated in log cos_i. This ignores lobe anisotropy, so the numbers are an order of magnitude.
- "Separable lobes needed" = max(1, major extent / lobe width) × max(1, minor extent / lobe width).

**Results.**

| light | E-weighted cos_i | mirror-direction extent across one patch (major / minor) | lobe half-width | separable lobes per element | irradiance share on patches needing > 8 |
|---|---|---|---|---|---|
| CK2 | 0.08 | 18.4° / 1.8° (top patch 28° / 3.5°) | 0.96° | 10.6 (top patch 16) | 0.79 |
| L1 | 0.84 | 3.3° / 2.8° | 5.4° | 1.0 | 0 |
| L2 | 0.32 | 6.2° / 2.1° | 2.6° | 1.2 | 0 |

**Reading.**
- Under grazing incidence, the outgoing field of one sharp element under one point light is a sheared 4D distribution: direction is a function of position.
- A separable "anchor + spread × direction + concentration" lobe can carry it only by spending about 10 lobes per element per light, or by sub-dividing the element about 10× along one axis.
- Otherwise the delivered glint is blurred along one axis by about 18° × distance, about 10× the lobe's own footprint.
- At non-grazing incidence one lobe suffices. So the K budget fails exactly in the regime that motivates the design.
- **Correction.** The interrupted run compared the spread with the cos_i = 0.025 lobe (0.66°) and quoted ratios of 10–20. The cos_i-matched comparison above (about 10) supersedes it.

### 1.2 Lobe measure near the horizon [CALC]

- The draft defines p_k as "unit flux in the projected-solid-angle × area measure" and then writes L = Σ Φ_k p_k / cos θ.
  - If p is a density in projected measure, then L = Φ p (no 1/cos).
  - If p is a solid-angle density of flux, then L = Φ p / cos θ. In that case the radiance is unbounded at the horizon wherever the lobe has mass there.
- Toy: elevation profile of a flux density, Gaussian with σ 1°.

| lobe centre elevation | lower cut-off of the integral | ∫ L dω | ∫ L² dω | L ratio between the two 1σ edges |
|---|---|---|---|---|
| 1° | 0.1° → 0.01° → 0.001° | 74 → 112 → 150 (log-divergent) | 2.5e5 → 1.7e6 → 1.6e7 (≈ 1/ε) | 349 |
| 3° | same | 22.7 → 23.9 | 7.5e3 → 1.1e4 | 2.0 |
| 6° | same | 9.86 (converged) | 1.55e3 | 1.4 |

- CK2 outgoing elevations are 4–7° (§1.1). Ring-0 grazing (cos_i < 0.05) puts lobes at or below about 3°.
- So under the second reading:
  - radiance varies 1.4–2× across the lobe from the measure factor alone. A lobe fitted in one measure has the wrong shape in the other.
  - near-horizon lobes give unbounded radiance and infinite-variance radiance estimators: any readout that samples ω, and the camera reading the panel through a sharp vertex.
- Defining the *radiance* shape and normalising its projected integral (no closed form for clipped anisotropic lobes, so a numerical normaliser) removes this. As written, contract T-a ("flux integral equals Φ_k to 1e-4") can pass while radiance readout is singular. T-a tests the wrong quantity.

### 1.3 Per-draw passivity of receiver-anchored gather [CALC]

- Model: a sharp sender whose lobe footprint covers n receiver patches.
  - Each receiver's K relation rays hit the lobe a Poisson number of times with mean 0.46 [MEAS WL96 §A, CK2 glint].
  - The expected delivered flux is a fraction f of the sender's emitted flux.
- The per-draw total delivered is an unbiased estimate. P(delivered > emitted):

| glint receivers n | f = 0.3 | f = 0.5 | f = 0.8 | P(no glint ray at all) |
|---|---|---|---|---|
| 1 | 0.08 | 0.37 | 0.37 | 0.63 |
| 3 | 0.013 | 0.16 | 0.40 | 0.25 |
| 6 | 0.001 | 0.06 | 0.30 | 0.06 |

- So a fixed relation draw gives an operator whose sender columns can exceed 1 in flux. Iterating it with that draw (C3UR / XQT sweeps reuse the draw) is not passive per realisation.
- In GM1 this rarely matters: one sharp sender, Lambert albedo ≤ 0.7, no glossy-to-glossy loop. In the glossy-to-glossy workloads (WGG, D3's Track B) it can raise the per-draw spectral radius.
- Sender-anchored deposition (FLS cubature, LLT photons, C3-type sender placement) deposits each emitted quantum exactly once, so it is passive per realisation.

### 1.4 LER-Δ splitting [CALC]

- b = (I + H_E)[e + (M − B) b], so the iteration matrix is (I + H_E)(M − B) = (I − T_E)⁻¹(M − B), not H_E(M − B) as written in D3 §3.2.
- Scalar environment gain t with a blocked share β (M = 0): factor −βt / (1 − t). It converges iff β < (1 − t)/t.

| t | correct β_max | β_max implied by D3's written condition | factor at β = 0.5 |
|---|---|---|---|
| 0.5 | 1.00 | 2.00 | 0.50 |
| 0.6 | 0.67 | 1.11 | 0.75 |
| 0.7 | 0.43 | 0.61 | 1.17 (diverges) |
| 0.8 | 0.25 | 0.31 | 2.00 (diverges) |

- D3's stated test accepts divergent cases.
- The iterates alternate in sign. Negative intermediate radiance is unavoidable, and clamping is forbidden.

---

## 2. Cross-cutting physics findings

### 2.1 Reference machinery: what can and cannot be validated

**Exists [CODE].**
- `pt86` camera references: depth-truncated at D = 6 events, no Russian roulette, `--kmax` configurable.
- `pt86a.field_src`: outgoing radiance per (patch, 64-bin) from 2..5 events.
  - Directions are drawn by `dir_in_bin`, uniform in z = cos θ and φ, i.e. **solid-angle-uniform**. The tally is therefore the solid-angle-mean bin convention, the same convention as M64.
  - `split=True` gives e2_p / e2_o / e3_p / e3_o: event count 2 or 3..kmax × last sender on the panel or elsewhere.
- Controlled incident fields: true, 'ang', 'enc', beams.
- The deterministic exact-ER first-flight quadrature oracle (WL97 A-04 / E3). It is independent of any candidate but uses the same frozen ER, which is truth by definition.

**Does not exist.**
1. Exact-direction outgoing probes at arbitrary (y, ω). `pt86.trace` has a probe mode with `probe_y`, so this is a launcher extension, but it needs its own qualification.
2. Cos-weighted (flux-conserving) cell tallies.
3. Incident flux per receiver cell, or the incident directional field (XQT's target, FLS's incident set).
4. Body self-path tags (RB-DTO's path-tag identity test).
5. Participating media and refractive interfaces (LIEM). WL53 qualified only index-matched media, in Cycles.
6. All-order transport. Every candidate that owns "all orders" (XQT iterated, LIEM's internal walk, RB-DTO's (I − K C_self)⁻¹, LER-Δ's H_E) must be compared at a matched event truncation or against deeper references.
7. A CK2 T1 reference [AUDIT J7, D-13].

**Consequence.**
- Every representation-level claim below bin resolution (peak position and width, sub-element structure, glint footprints) needs new reference code first.
- Every design's evidence list asks for such metrics. None budgets their qualification as a gate, although D1 names "exact (x, ω) probes".
- Camera references are adequate for image-level claims in all designs.

### 2.2 Linearity and superposition

- Physical transport is linear in emission. Linearity is what makes these valid: the T0 / T1 difference decomposition, per-light superposition, the source-only update, and impulse (column) validation of an operator.
- Nonlinear candidates:
  - **FLS:** allocation depends on the whole incident set;
  - **RB-DTO:** per-patch mixture reduction to ≤ M_p lobes in the output cache.

  For these:
  - (i) superposition must be tested and cannot be imposed;
  - (ii) the state is not an operator, so it cannot be validated by impulses and must be validated per full incident configuration;
  - (iii) the T-s tolerance has no evidential basis [HUM]. It would be invented.
- Linear by construction: ECRO / ECRO-FC, XQT, LLT (per incident direction), LIEM-S / B, RB-DTO's per-sample pushforward before reduction, and LER-Δ.

### 2.3 Energy: what each "passive by construction" actually guarantees

| candidate | guarantee as claimed | what actually holds |
|---|---|---|
| FLS | Σ Φ_k ≤ Σ a(ω_j) Φ_j | Holds in flux. But the NULL sink is learned absorption beyond exact ER: the exact scattered flux Σ a_j Φ_j is known, so the NULL is an unphysical degree of freedom. It reopens the WL86A-type compensation, where flux is dumped to offset misplacement. Furnace T-b needs π_NULL ≡ 0. Radiance readout is singular near the horizon (§1.2) |
| ECRO-FC | "albedo-bounded allocation on the local factor pair" | The bound needs a declared measure: column sums of a c^T weighted by the projected cell measure of the FC table. Feasible. The global term stays unconstrained |
| XQT | column-stochastic redistribution × ER ≤ 1 | Passive in expectation only (§1.3). Per draw, a sender's delivered flux can exceed its emission |
| LLT | Σ Q + Q_miss = 1, ∫ p = 1 | Holds per lobe |
| LIEM-S | m_c ≤ 1 − m_bal; normalised density | Holds, provided m_bal(x_i, ω_i) is exact. That needs explicit tracing of the ballistic Fresnel / TIR tree per query. The chart mass beyond edges is counted as absorption: passive, but a measure leak concentrated at edges, where learning is supposed to beat Twin-D. The "exact" rough-dielectric closure must be declared (single- versus multiple-scatter microfacet BTDF, η² radiance scaling) and matched to the reference's interface model |
| RB-DTO | outputs ≤ Φ_i[ρ_ER − E_esc] | E_esc(x_i, ω_i) is a 4D escape-visibility integral of a sharp ER lobe. If it is approximated or "bounded" from below, the budget is overestimated and exact first-event escape plus N_b can exceed ρ_ER. Passivity then depends on a visibility quadrature, not on construction |
| LER-Δ | H_E passive | Holds for H_E. The Δ-iteration is not passive (signed iterates), and its convergence condition is mis-stated (§1.4) |

**Reciprocity.** No design imposes it, and all measure it. That is acceptable. For the estimator designs (XQT, C3UR) the pairwise check F_ij A_i = F_ji A_j is cheap and catches measure errors. It should be a contract test, not just a diagnostic.

### 2.4 Ownership: where learning touches visibility or inter-object geometry

- **ECRO / ECRO-FC.** The per-patch factors c and a are generated from:
  - openness = 1 − Σ form factors from the *current* state rays;
  - Ft-weighted message passing;
  - world centre, normal and tangent;
  - a scene-mean token [CODE `model86.py:93-104`].

  The "local" response therefore depends on current visibility and on the whole scene. Rigid equivariance and a local validity certificate are impossible [CODE; D1 §4.7; D3 §3.3].
- **XQT.** D1 claims "XQT never predicts a hit". But moving quantum r, which arrived at receiver i′ via hit point x_j, to a neighbour receiver i asserts that x_j is visible from i and that its radiance toward i equals its radiance toward i′. The visibility check is element-level ("i's records contain a hit on the same element within an angular tolerance"), not point-level.
  - Partial occlusion inside one sender element leaks. This happens in the box-A edit, where the box shadows part of the panel.
  - For lobes narrower than the receiver-to-receiver angle, the moved value is wrong by the lobe ratio.

  P7 is therefore only partly satisfied.
- **LLT.** It infers visibility of the footprint between probes. D1 says so and gates it.
- **LIEM-S.** None. External geometry never enters, and contact invalidates the certificate. This is the cleanest ownership of all designs.
- **RB-DTO.** C_self (air paths that leave and re-enter the same body) is moved into a learned body operator.
  - `Architecture.md` §2: C "includes exposed same-asset and same-domain exit / re-entry paths", and "Object identity must not freeze exposed interreflection or merge editable air transport into K".
  - The brief: keep the intrinsic / geometry-dependent ownership separation.

  The certificate makes self-visibility static only while nothing enters Ω_b. On partial intrusion, the invaded paths cannot be subtracted from a learned N_b (D2 §6.6 makes the same point). The fallback must therefore be whole-body explicit, not "the invaded part" as D3 writes.
- **LER-Δ.** H_E owns static environment visibility, learned per environment. B subtracts the blocked paths *explicitly*, but H_E contains its *learned approximation* of them. The residual H_E[blocked] − explicit B is a ghost: a blurred sharp beam minus an exact sharp beam leaves a negative core and a positive halo. This is structural leakage, not a tuning issue. It is exactly the brief's "leakage / ghosting" metric.

### 2.5 What there is to learn in GM1 (D1 Proposition I, which I accept)

- In scene86 / GM1, intrinsic transport equals pointwise ER, and every nonlocal path is free-space C. Any learned component can therefore only:
  - estimate with fewer queries;
  - amortise repeated application;
  - or infer unqueried information, which for C means visibility.
- **This proposition binds FLS too.** The draft claims Λ_θ owns three things:
  - "element-local scattering of distributed incidence": ER does this exactly per component;
  - "sub-element spatial–angular structure": analytic for planar elements under point lights (image source, §1.1);
  - "closure": mixture reduction of known lobes, whose learning-free twin is EM or moment matching.

  None is transport information that the explicit side lacks. HD-1 is therefore not a matter of framing. Under the brief's exclusion, FLS in GM1 *is* ER compression plus mixture reduction.

### 2.6 Why receiver-side pooling (XQT) cannot fix CK2-type glints [PRED, geometric]

- Lobe footprint on a receiver ≈ distance × lobe angle: about 1 m × 2 × 1° ≈ 3.5 cm, against 0.2 m patches.
- Neighbouring receiver patches subtend about 11° from the sender, far outside a 1° lobe.
- The heavy tail of P0-3 is the count of rays from *one* receiver that hit the glint region of the sender. Redistributing quanta across receivers either moves glint flux to places the lobe does not reach (bias) or pools nothing.
- The learned W can at best learn to pool zero for sharp senders. XQT's lever therefore exists only for lobes whose footprint spans several receivers, i.e. broad transport, where WL96A found no headroom.
- D1's flatland result is consistent with this: grazing α 0.03 still has rel L2 0.24–0.74 after the best pooling, and the oracle over twin is only 1.0–1.27.

---

## 3. Per-design, per-candidate verdicts

### 3.1 DESIGN_REVIEW (coordinator)

**FLS (primary): survives NO (as a learned candidate).**

*Fatal objections:*
- **F1. The state contract cannot hold the motivating case.** K ≤ 8 separable lobes per 0.2 m element cannot keep a point-light glint on S1 at CK2 incidence at lobe resolution: about 10 lobes are needed per element per light, and 79 % of the irradiance is on patches that need more than 8 [CALC §1.1].
  - The working fix is a sheared (image-source) parameterisation, or sub-elements about 10× finer.
  - For direct light the sheared lobe is the analytic mirror construction, so nothing is learned. Sub-elements multiply the K·Q·N cubature rays and approach per-point explicit evaluation.
  - This is the brief's stop condition "representing the sharp response effectively needs full explicit transport".
- **F2. The learned operator does not replace or reuse any explicit transport work.**
  - Delivery is explicit lobe cubature on current geometry: K·Q·N rays per sweep, ≈ C3URX1's count by the draft's own arithmetic.
  - Λ_θ replaces only ER evaluation and lobe fitting.
  - This is the brief's stop condition 1, and §2.5.

*Serious objections:*
- The measure formula is inconsistent; T-a tests flux, not radiance (§1.2).
- The NULL sink is an unphysical learned absorption (§2.3).
- The design is nonlinear. T-s needs a tolerance with no basis.
- The lobe family fails at grazing even for a single ER lobe: lobe L1 0.35–0.6 at K ≤ 4, 2–7× peak overshoot [CALC by D2 §5.1].
- Per-source states multiply cost by the number of lights.
- Cubature error at occlusion edges (F2 in the draft).
- Validation of lobe position and width is impossible with `field_src` (§2.1).

*Strengths:*
- Deterministic sender-anchored delivery is passive per realisation and has no single-draw variance.
- Rigid equivariance in local frames.
- Visibility is explicit.

*Disposition:* its learning-free twin **ELT** should be kept as a comparator. It must include the sheared, image-source lobe (§5). Moment-matched lobes alone are a weak twin.

**Candidate A (draft, rejected as primary): agreed, NO.** The nonlocal form owns visibility; the local form is BRDF compression.

**Candidate B (draft, kept as a control): agreed.** NO as a learned candidate, YES as the flux-conserving control.

**ECRO-FC (alternative): survives NO as a candidate, YES as a diagnostic arm.**

*Fatal objections:*
- The generator conditions the per-patch response on current visibility (openness, form-factor message passing), world coordinates and a scene-mean token [CODE `model86.py:93-104`]. That is dynamic visibility coupled into the per-patch response, and there is no rigid-equivariance certificate.
- The sender half re-learns ER, which is BRDF compression.

*Serious objections:*
- As an H2 arm it changes three treatments at once (carrier, coverage, passivity), so neither H1 nor H2 can be attributed. The comparison against "ECRO + coverage" mixes carrier and passivity.
- Its global term keeps WL86A's compensation risk.

*Strengths:* it is the right lineage diagnostic, once it is factorised.

### 3.2 D1 (physics and measure first)

**XQT (primary): survives CONDITIONAL.**

*Fatal:* none under the current canon, if HD-1 admits an estimator.

*Serious objections:*
- (i) "Passive by construction" is false per draw (§1.3). It holds only in expectation. This is a contract-wording defect, and it matters on glossy loops.
- (ii) Neighbour redistribution infers point-level visibility from element-level co-visibility, and carries radiance evaluated toward i′, not toward i (§2.4). The result is leakage at intra-element occluder edges and wrong values for sub-receiver lobes.
- (iii) Its lever vanishes for CK2-type glints (§2.6). Its author's own scratch shows oracle over twin of only 1.0–1.33.
- (iv) The learning-free twin is under-specified. A fixed-kernel pooling is weaker than spatial reuse with exact re-evaluation plus a shadow ray (ReSTIR-GI-type), or than sender-anchored deposition. Against a weak twin, H-L is biased toward learning.
- (v) The incident-flux-per-cell target has no existing reference tally (§2.1).

*Strengths:*
- Linear and source-independent.
- ER is exact.
- Visibility comes only from explicit records.
- Rigid-invariant inputs, with a dependency certificate for reuse.
- Honest about information content (Proposition I).

*Condition:* HD-1, plus the corrections above.

**LLT (alternative): survives CONDITIONAL, only as the S1-d information test, not as a candidate.**

*Serious objections:*
- The only way it can beat its twin (photons through the same K_p probes) is inferred visibility between probes. That is learned C, with a WL52 negative prior on transitions.
- It costs per light, like the explicit lobe tracing it would replace.

*Strengths:*
- Passive per lobe.
- Linear per incident direction.
- Its collision-pair information test is a sound, learning-free falsifier.

**A1 / A2 / B1 / C1 rejected, E0 as a control, O1 rejected under canon: all agreed.**
- O1's rejection on Architecture §2 grounds is consistent with mine for RB-DTO.

### 3.3 D2 (learning and generalisation first)

**LIEM-S (primary): survives CONDITIONAL.**

*Fatal:* none under the current canon. Its ownership is canonical: interior collided transport goes to K, air to C, ballistic transport stays explicit.

*Serious objections:*
- (i) It does not learn interreflection or spatially distributed high-frequency *free-space* transport. Its sharpness comes from the exact interface, and its learned part is smooth exactly where it saves work (D2's own slab tension). The brief's central hypothesis is therefore tested only in a different physical regime (GM2), and "complex-interreflection fidelity" is not addressed by learning [HUM HD2].
- (ii) The energy bound depends on an exact per-query m_bal (an explicit ballistic Fresnel / TIR tree). The chart mass lost beyond edges is a measure leak concentrated where learning is supposed to win.
- (iii) The "exact interface closure" must be specified and matched to the reference's rough-dielectric model:
  - single- versus multiple-scatter microfacet BTDF;
  - η² radiance scaling;
  - reciprocity of the BTDF form.
- (iv) No qualified reference exists:
  - WL53 covers index-matched media only.
  - Mitsuba float32 skips about 89 µm at null interfaces (∝ σ_t).
  - LabServer63's Mitsuba cuda film is wrong.
  - Cycles BSSRDF closures are path-history-dependent and must not be used; only the volume RTE inside a rough-glass boundary is admissible, and only after qualification.
- (v) The float64 walk is training generator, Twin-W and "independent check" at once. Independence must come from the Cycles / Mitsuba cross-check.

*Strengths:*
- Linear, impulse-supervised.
- Passive by construction with a declared measure.
- Exactly rigid-equivariant, with no visibility inside.
- Coverage is a declared treatment.
- Three named twins.

*Condition:* HD2 (scope), I0.1 reference qualification, I0.2 headroom.

**LIEM-B (alternative): survives CONDITIONAL, as a representation control only.**
- Air-side direction cells without the exact interface must resolve α-scale exit lobes by cells (the α⁻² growth again).
- It is not mechanism-independent, as D2 states.

**R-A, R-B, R-C, R-D, R-E rejected: agreed.**
- R-D is the same canon conflict as RB-DTO.

### 3.4 D3 (systems and cost first)

**RB-DTO (primary): survives NO under current canon. CONDITIONAL only after an explicit canon change [HUM D1].**

*Fatal objection:* it moves editable same-asset air transport (C_self) into a learned body operator. The canon forbids this, and the brief says to keep the ownership separation. The certificate does not cure partial intrusion, because invaded paths cannot be subtracted from a learned N_b. The fallback is therefore whole-body explicit.

*Serious objections:*
- Passivity depends on a 4D escape-visibility quadrature, E_esc (§2.3).
- The per-patch lobe cache reduction is nonlinear (§2.2).
- The sheared exit distributions of glossy cavities face §1.1's lobe-count problem.
- Validation needs new self-path tags, references deeper than D = 6 (no RR) for multi-bounce cavities, and a new, unqualified scene family. It is identically zero on every currently qualified workload (E2, E5, CK2).
- Twin B1 is costed with uniform binning. A hierarchical or low-rank precomputed body transfer is a stronger explicit twin and is missing.

*Strengths:*
- It identifies the only rigid-invariant nonlocal transport that could pay: self-transport of non-convex bodies.
- The first local event stays exact ER.
- Path-tagged component supervision.
- A cost anatomy that runs before training.

**LER-Δ (alternative): survives NO.**

*Fatal objections:*
- (i) The convergence contract is mis-stated. The correct condition, ρ((I + H_E)(M − B)) < 1, fails for bright rooms or large occluders (§1.4).
- (ii) Learned-minus-explicit cancellation leaves structural ghosts of blocked sharp beams (§2.4).
- (iii) H_E is per-environment learned state, close to the scene-baked class the Architecture excludes and to the external Native Reuse class.

*Serious objections:*
- H_E's fidelity on signed Δ-fields outside its training span is untested (WL41 / WL44 precedent).
- Signed intermediate iterates cannot be clamped.

*Strength:* the edit enters exactly and explicitly through B and M. A learning-free variant with a Woodbury / Krylov update on a precomputed explicit environment operator is a legitimate *comparator*.

**A, B, C at patch level rejected: agreed.** D3's observation that ECRO's local factors depend on current visibility matches my code reading.

---

## 4. Falsification-plan defects

### 4.1 Shared by all designs

1. **The reference-machinery gap is not budgeted as a gate.**
   - No plan states that `field_src` tallies the solid-angle-mean 64-bin convention, that it is depth-truncated, and that it has no cos-weighted, incident, sub-bin, self-path or media tallies.
   - Every sub-bin representation metric (peak position, width, glint footprint) therefore depends on new, unqualified probe code.
   - A Stage 0 "reference-instrument qualification" is needed: exact-direction probes against deterministic quadrature and against the existing binned tallies where they overlap.
2. **H2 class floors are computed against 64-bin references.**
   - A floor of a 64-bin representation fitted to a 64-bin reference cannot show what is lost below the bin.
   - H2 needs references at a finer resolution than the finest carrier tested.
3. **Event truncation is not matched.** All-order candidates must be compared at the reference's kmax, or the reference deepened, with the truncation residual estimated.
4. **CK2 T1.** Any edit use of CK2 needs a new T1 reference. Its absence is acknowledged by D1 and D2 but missing from the draft's Stage 3 prerequisites.

### 4.2 DESIGN_REVIEW

5. **The STOP FLS criterion uses an insensitive observable.**
   - "Optimal-fit lobes vs FC 8 × 32 in delivery error on WG" measures delivery to extended Lambert receivers. There 8 × 32 is already within 2–6 % at box A [AUDIT J2].
   - Lobes can win only at glint receivers and through camera reads of the panel.
   - Without predeclared receiver classes (glint core / tails / extended), the gate is uninformative.
6. **ELT (moment-matched lobes) is a weak twin.** It must include exact per-component ER lobes with image-source shear (sender-side lobe tracing). Otherwise H-L is biased toward learning.
7. **ECRO-FC is confounded** (three treatments). It should be factorised: coverage only; carrier only (C1f-type); passivity only.
8. **Missing references.** The draft references `CRITIQUE.md` and a §10 response, but neither exists in the directory.
9. **WGG / glossy loops have no per-draw passivity or spectral-radius test** for the explicit comparators (§1.3).

### 4.3 D1

10. **The H1 arm C1 changes two things:** the coverage distribution and the added physical sender target (identifiability). H1 (coverage) is then not separated from supervision. It needs C1a (coverage only) and C1b (sender target only).
11. **The XQT twin should include exact-re-evaluation spatial reuse** (shadow ray plus ER at the new direction) and sender-anchored deposition at matched rays. The fixed analytic kernel is the weakest admissible twin.
12. **The passivity contract test T5 must be run per relation draw**, not in expectation.
13. **The projected-disk carrier** has no reference tally to compare against. Cos-weighted sampling must be added to the probe code.

### 4.4 D2

14. **The I3 kill rule is nearly unkillable.** LIEM survives if any of (a) time-to-quality vs Twin-W, (b) accuracy vs Twin-D, or (c) memory vs Twin-T holds. But (c), memory of a shared network vs a 2 GB per-asset table, is guaranteed for any shared model and says nothing about geometry-edit utility. Utility must require (a) or (b), with (c) reported but not decisive.
15. **The walk generator is used three times** (training data, Twin-W, independent check). Independence must be defined as agreement with an external renderer at I0.1, before any use.
16. **The interface model must be declared before I0.1.** Otherwise "exact closure" and "reference" can silently differ.

### 4.5 D3

17. **The LER-Δ convergence test is wrong** (§1.4).
18. **The path-tag identity test** requires tracer instrumentation that does not exist.
19. **Track B references:** glossy cavities need deep paths and high spp. D = 6 without RR truncates the transport RB-DTO claims to own.
20. **The B1 twin is costed with uniform binning.** A stronger explicit twin is missing, so Stage 0's U0 is over-estimated.
21. **The fallback is costed as "invaded part explicit".** Physically it must be whole-body explicit (§2.4), so the fallback cost is under-estimated.

### 4.6 Statistical and threshold hygiene

All four designs correctly leave thresholds to [HUM], keep M64 out of truth, and keep CK2 diagnostic. The draft's grazing criterion (≥ 50 % of S1 irradiance at cos_i ≤ 0.15) is offered as an example value and is correctly marked [HUM].

---

## 5. Missed alternatives

- **Sheared / image-source lobe (virtual-source) parameterisation** for planar or near-planar sharp senders. It is learning-free and exact for planar senders and point or area-light lobes (ER blur on a virtual source). It must be the FLS / ELT twin and an explicit comparator. It also answers §1.1 at K = 1 per light.
- **Spatial reuse with exact re-evaluation** (hit-point reuse across receivers, plus a shadow ray, plus exact ER toward the new receiver; ReSTIR-GI class). It is learning-free, unbiased with proper weights, and the right twin for XQT.
- **Woodbury / Krylov edit update** on a precomputed explicit operator. It is robust where LER-Δ's fixed-point splitting diverges, and is a learning-free comparator for any "frozen environment plus Δ" idea.
- **Learned proposal densities only** (parametric-mixture guiding over lobes or receivers, with exact ER and explicit visibility). Every physics contract holds automatically, because learning error costs variance, never bias or energy, and the densities can be conditioned on local frames for rigid reuse.
  - It is family C used as a sampling distribution, not as a carrier.
  - It would face the same HD-1 question as XQT. WL46 already separated sampling as a conditional variable.
  - I list it because none of the four designs compares it, not to recommend it.
- **Reciprocity-symmetric parameterisation** of any learned operator (A^{1/2}-scaled symmetric form). It turns a measured property into a constructed one for linear estimators.

---

## 6. Proposed batch verdict

**DESIGN_READY_FOR_HUMAN_REVIEW, with an explicit dissent and conditions.**
- The batch contains physically defensible conditional candidates:
  - **XQT:** within the current canon and GM1, contingent on HD-1;
  - **LIEM-S:** canonical ownership, contingent on the GM2 scope and reference qualification.
- Both have learning-free stop gates that can end the line before training.
- The coordinator's draft pair should not go forward as primary / alternative:
  - **FLS:** NO, by §1.1, §1.2 and §2.5;
  - **ECRO-FC:** NO as a candidate, by §2.4.
- **RB-DTO** is admissible only after a canon change, and **LER-Δ** fails its own contracts.

**Dissent / scoped reading.** For the current canon and the scene86 / GM1 family alone, every designer and this critic agree that no learned candidate has a physical information advantage (D1 Prop. I, D2 §3, D3 §2.5, §2.5–2.6 here). If the owner answers HD-1 with "a learned estimator over explicit visibility samples is not the project's learned transport" and does not open GM2, the correct batch verdict is **NO_DEFENSIBLE_CANDIDATE**.

**Conditions before any freeze.** These must hold whichever candidate is reviewed:
- the reference-instrument gate (§4.1, items 1–3);
- a corrected per-draw passivity contract;
- the strong learning-free twins of §5;
- factorised H1 / H2 arms.

None of this is adoption or experiment approval.
