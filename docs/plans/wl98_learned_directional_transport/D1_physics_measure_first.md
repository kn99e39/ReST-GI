# WL98 — Designer D1: physics and measure first (independent report, final)

**Date.** 2026-11-16. **Role.** Independent designer D1 of the design-only batch WL98 (learned directional transport representation after ECRO). Starting angle: the rendering equation, the K / C ownership contract and the WL97 measure / energy failures.

**Independence.** I did not open any other file in `docs/plans/wl98_learned_directional_transport/` (no other designer's or coordinator's draft). Sources read: `AGENTS.md` (in context), `docs/Architecture.md`, `docs/BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md`, `docs/RESEARCH_ROADMAP.md` (frontier, GM1, WL86–WL96A records), the WL97 audit (`README.md`, `COORD_synthesis_and_p0_ledger.md`, `A_math_physics_contract.md`, `J1`, `J2`, `J4` §0, `J5` §1, `D` D-04), worklogs 86, 96, 96A (event-split and ceiling tables), 89 (V3 workload), and code `experiments/ecro86/model86.py`, `experiments/ecro86/protocol/ecro86.json` (fixture).

**What this is.** Design only. Nothing was implemented, trained or rendered; no existing file was changed; no commit. One CPU numpy scratch calculation (a flatland toy, §9) was run as design input; it is not evidence of record.

**Labels.** [MEAS] measured, with its source. [AUDIT] a WL97 interpretation or adjudication. [CODE] read from code. [PRED] my new prediction or argument (untested). [HUM] needs a human decision. [SCRATCH] my numpy toy (design input only).

---

## 0. Summary

1. **Physics first.** A learned transport state that will be composed with ER and explicit visibility must preserve, by construction where possible: the flux measure of every stored coefficient (projected solid angle × area), non-negativity, linearity in the source, passivity of every factor that is iterated, rigid equivariance in object-local frames, single ownership of every path segment, and explicit (never learned) visibility of editable free space. Each of the WL97 measure / energy failures is a violation of one of these rules (§2.3): the solid-angle-mean bin over-emission (measure mismatch between stored coefficient and flight), non-passive learned local factors (a learned local operator where ER is exact), the emission-anchor / patch-spatial readout errors (stored state cannot be read pointwise), the latent unsupervised sender (an un-identified intermediate with a physical name). Two further violations are visible in the ECRO v0 code and are not in WL97: absolute world coordinates and a scene-global mean token as generator inputs (`model86.py:95, 99`), so the learned "local" factors are neither rigid-invariant nor dependency-local [CODE].
2. **Information argument [PRED, derived].** In the GM1 regime (surface-only scene86 / GM1 family: planar and convex objects, point lights, pointwise frozen ER), K reduces exactly to ER at every vertex: there is no intrinsic nonlocal transport to learn. Every nonlocal transport quantity is a deterministic functional of (ER, current geometry and visibility, emission). A learned component therefore cannot add physical information in this regime; it can only (a) estimate an integral with fewer explicit queries (variance / resolution amortization), (b) amortize repeated application (order amortization), or (c) infer what was not queried (which, for free-space delivery, means inferring visibility). (c) conflicts with "learning never owns visibility"; (b) has a measured ceiling on CK2 for the resolvent-only role [AUDIT P0-4]; (a) is where the measured residual of the explicit exact-sample route sits (single-draw variance at the deployed budget, P0-3) [AUDIT].
3. **Candidates.** A (continuous directional learned kernel): **rejected** in both readings, local (BRDF compression of ER) and nonlocal (learned gather / learned C without a by-construction measure). B (sparse multi-resolution angular representation): **kept as the carrier**, in a flux-exact projected-disk form, but rejected as a stand-alone learned candidate (its learned part would be a resolvent only). C (learned lobe / distribution representation): **rejected** on the sender side (ER compression), **kept on the delivery side** as the independent alternative.
   - **Primary: XQT — exact-sample quadrature transport.** One shared, geometry-conditioned learned operator that is the *flight operator itself*: a sparse, non-negative, column-stochastic redistribution of exact per-ray flux quanta (ER evaluated exactly at the exact hit point and direction) among visibility-consistent receiver cells of a flux-exact projected-disk carrier, iterated with exact ER for all orders. Linear, flux-exact, passive and source-independent by construction; visibility only from explicit relation records; rigid-invariant inputs; learned state (the sparse weights) reusable after rigid edits with a dependency certificate. It carries the sharp, spatially distributed first-flight delivery and every later flight. Its only lever is estimation efficiency (fewer relation rays for equal error).
   - **Alternative: LLT — learned lobe transfer from sparse probes.** Sender-anchored, per-lobe learned delivery (flux fractions over candidate receivers with a miss class, plus a normalized footprint density) conditioned on a small set of explicit probe rays. It tests the other information regime: whether a learned representation can carry α-scale delivery *beyond* what is explicitly queried. Its central risk is implicit visibility inference.
4. **Prior.** At the currently qualified scale (N ≈ 1,270 patches, 1–2 lights) the removable explicit transport work is about 3.3 ms of a 125 ms edit-to-frame lifecycle and 67–79 % of each lifecycle is non-kernel time [AUDIT D]; even a zero-cost full learned operator reaches only about 1.24–1.37 on source events [AUDIT D-04]. A cost-side utility at that scale is therefore unlikely for any candidate [PRED]. Utility must come from **quality-matched cost** (explicit needs many more draws to reach a sharp-region quality tier, e.g. WL91's 3.1× at one tier [MEAS WL91]) or from a declared **scale axis**. The flatland scratch (§9) gives a design-input estimate of how much a geometry-adaptive pooling could gain over its learning-free twin.
5. **Falsification plan (§6).** Stage 0 workload / reference admission; Stage 1 learning-free decomposition that can stop the batch before any training (representation floors for H2, explicit quality-cost frontier, candidate oracle / twin ceilings, LLT information test, utility ceilings for H3); Stage 2 the H1 arm (ECRO with corrected coverage and physical sender supervision, same representation); Stage 3 candidate training; Stage 4 held-out T0 → T1 with fixed weights; Stage 5 quality-matched economics.
6. **Proposed verdict: DESIGN_READY_FOR_HUMAN_REVIEW**, conditional on human decision HD-1 (whether an estimator-type learned operator over explicit visibility samples is admissible as the project's learned transport) and with the explicit prediction that a positive H3 result at the current workload scale is unlikely. Not adoption, not experiment approval.

---

## 1. Problem statement in operator form

### 1.1 Rendering equation with K / C ownership

Notation: boundary points x, y; directions ω in the local frame of the point; L outgoing radiance, L_in incident radiance; dω⊥ = cos θ dω (projected solid angle).

```text
L_in(y, ω)  = (C L)(y, ω) = L(h(y, ω), -ω)              # free-space flight, h = first hit (visibility)
L(x, ω_o)   = e(x, ω_o) + (K L_in)(x, ω_o)
(K L_in)(x, ω_o) = ∫_H f_x(ω_i, ω_o) L_in(x, ω_i) dω_i⊥   # ER, pointwise (GM1 regime)
L = Σ_j (K C)^j e                                       # Neumann series, convergent iff ||K C|| < 1 in the flux norm
```

- **K** (Architecture §2) owns local surface response and interior transport. In the GM1 regime every object is a single interface with pointwise frozen ER (`Architecture.md` §3, "Current local-scattering foundation"); there is no interior. So K = ER exactly, and ER is already learned, passive and reciprocal [MEAS WL82–84].
- **C** owns every free-space segment, including same-asset exit / re-entry (Architecture §2). C is a function of current geometry only: h(y, ω), distances, cosines.
- ECRO's event partition [CODE `model86.py:110-124`, AUDIT A §2]: the camera prefix (e1 at x1, flight x1 → x2, e2 at x2) is explicit; the table B[x2] carries 2–5 events at x2, of which the "single" term is exactly one learned flight (2 events) and the "global" term events 3–5.

### 1.2 Consequence: what is there to learn in GM1

**Proposition I [PRED, derivation].** In a scene where (i) every optical domain is a surface interface with a known pointwise closure (ER), (ii) emitters are known analytically, and (iii) visibility is explicitly computable, every transport quantity (any table, field, operator power or image) is a deterministic functional of (ER, geometry, visibility, emission). A learned function g_θ(X) of any declared input X can be no more accurate in expectation than the exact functional computed from the same X *plus whatever X omits*. Hence a learned component can contribute only:

| lever | what the learned model does | explicit twin | WL evidence |
|---|---|---|---|
| (a) estimation | turns a small set of explicit queries into a low-variance estimate (pooling, quadrature weights, denoising) | the same queries with a fixed estimator, or more queries | X1 exact samples are unbiased but heavy-tailed: single draws at K = 512 give no gain, ≥ 16× draws needed [AUDIT P0-3]; X1 per-draw variance is a property of receiver-anchored gathering [MEAS WL96 §A] |
| (b) amortization of repeated application | replaces J explicit sweeps by one learned application | explicit recurrence | resolvent-only ceiling 1.02 / 1.04 on CK2 [AUDIT P0-4]; full learned operator source ceiling ≈ 1.24–1.37 at zero cost [AUDIT D-04] |
| (c) inference of what was not queried | predicts transport that depends on unqueried visibility or unresolved angles | query it (rays, finer bins, exact evaluation) | WL52 learned C ≈ ray-traced patch C0p, fails at contacts / transitions [MEAS WL52] |

Lever (c) is the only one that could carry information "beyond explicit"; for free-space delivery its content is visibility, which the brief forbids the intrinsic representation to own and which the canon gives to C. So a defensible learned sharp-transport candidate in GM1 must live mainly on lever (a), and should use lever (c) only where an information test shows the queried evidence determines the answer.

**Where a learned *intrinsic* representation would have physical content** (outside GM1, noted for completeness): optical domains with interior nonlocal transport (media, layered slabs with lateral transport, refractive solids with internal reflection) whose response is rigid-invariant in the material frame. That is GM2 territory and the WL41–WL80 lineage; I do not redesign it here. The asset-level variant that would put a concave asset's exposed self-interreflection into K contradicts Architecture §2 (exposed same-asset paths belong to C) and is rejected in §4.8 unless the owner changes canon [HUM].

---

## 2. What a learned transport state must preserve

### 2.1 Required properties

| id | property | definition used here | by construction? |
|---|---|---|---|
| P1 | linearity / superposition | output is a linear map of the emission / incident field; operator weights do not depend on radiance values | yes, if weights are functions of geometry, relation records and material only |
| P2 | non-negativity | every stored flux and operator entry ≥ 0 | yes (softplus / softmax) |
| P3 | measure consistency | every stored coefficient is a flux Φ = ∫_cell ∫_area L dω⊥ dA, and every flight integrates with the same measure; pointwise readout is never taken from a coefficient that stores a cell mean of a sub-cell lobe | yes, with a projected-disk (Nusselt) carrier and exact-ER readout (§3.2) |
| P4 | passivity | each iterated factor has flux-norm ≤ 1: ER directional albedo ≤ 1, flight closure ≤ 1, redistribution column sums = 1; the composed operator's spectral radius is then < 1 | yes for XQT (product of passive factors); for any learned resolvent only by a generating construction |
| P5 | reciprocity | ER is reciprocal [MEAS WL82–84]; for flights, F_ij A_i = F_ji A_j in expectation | ER yes; learned estimators no (tested, not imposed) |
| P6 | rigid equivariance | learned inputs are relative / object-local (frames of the sender and receiver elements, distances, cosines, material descriptors); no world coordinates, no unbounded global pooling | yes, if the input list is restricted (§3.3) |
| P7 | visibility ownership | which surface a direction hits is decided only by explicit ray queries on current geometry; learned functions may weight queried hits, not predict unqueried ones | yes for XQT; only partially for LLT (tested, §6 S1-d) |
| P8 | single path ownership | every path family (by event count and segment owner) has one owner; prefix / learned / explicit split disjoint | yes, by event-count partition as in ECRO (§3.4) |
| P9 | identifiability | any learned intermediate named as a physical quantity is either supervised against that quantity or exactly constrained to it | yes for XQT (no latent physical intermediate) |
| P10 | independent validation | every learned output has a physical counterpart computable by an independent method (path tracing, deterministic quadrature validated against path tracing) | yes; M64 never used as ground truth |

### 2.2 Measure: the projected-disk carrier

- Map the local hemisphere to the unit disk by (u, v) = (sin θ cos φ, sin θ sin φ). The Jacobian gives dω⊥ = du dv, so **projected solid angle equals disk area**. A cell of the disk with area a_c carries flux Φ_c = a_c · L̄_c · A, where L̄_c is the projected-solid-angle (cosine-weighted) mean of the radiance. This is the flux-conserving convention J2 identifies as the unique piecewise-constant value that conserves each bin's flux under the exact-cosine flight [AUDIT J2 §1].
- The 4 × 16 rings uniform in z = cos θ of scene86 have exact cosine integrals ZB · DΩ [CODE AUDIT A-01], so a flux-exact coefficient can also be defined on the existing bins; the disk form is preferred because a quadtree / Haar refinement of the disk keeps parent flux = sum of children exactly (multi-resolution flux consistency), and grazing directions occupy a thin rim whose projected measure is small.
- **Rule M1:** store flux per projected-measure cell, never solid-angle means converted to flux by a cell-centre cosine. This alone removes about half of the CK2 box-A over-delivery (solid-angle mean +40 %, flux-exact +19–21 %) [AUDIT A-04, J2].
- **Rule M2:** a piecewise-constant cell of any convention cannot carry a lobe narrower than the cell; the residual +19 % at box A is in-bin redistribution with a receiver-dependent sign (glint receivers 0.12–0.19×) [AUDIT J2]. So sharp *outgoing* radiance is never stored in cells. Sharp outgoing radiance is evaluated pointwise by ER at the exact direction from the stored *incident* state (exact light directions for direct light; flux-exact incident cells, refined by an explicit rule, for indirect light).
- **Rule M3:** readout through a sharp reflection images the receiver's field "in focus" (J4: patch spatial averaging at x2 costs 0.010 rel-L2 in ROI_T, while angular bin averaging costs 0.0001 there) [AUDIT J4]. The carrier at receivers seen through sharp reflections needs sub-element spatial structure or pointwise evaluation; at Lambert-first pixels it averages out (0.0004).

### 2.3 Mapping the WL97 failures to violated rules

| WL97 finding | violated property | design consequence |
|---|---|---|
| solid-angle-mean 4×16 sender over-emits +30 % panel flux at CK2 grazing; +40–50 % box-A over-delivery, about half convention [AUDIT P0-2, A-03/04, J2] | P3 (stored solid-angle mean flown with projected measure) | M1: flux per projected-measure cell |
| in-bin redistribution +19 %, receiver-dependent sign; gone at 8×32 [AUDIT J2, E2-F9] | P3 (sub-cell lobe in a cell) | M2: never tabulate sharp outgoing; exact ER at read; refine incident cells by rule |
| emission anchor (X1c 0.234 vs X1 0.113) and patch-spatial readout 0.010 rel-L2 in ROI_T [MEAS WL96 §A, AUDIT J4] | P3 (position measure) | M3: emit / read at the exact point; sub-element structure where seen through sharp reflections |
| learned S1 local factors: worst-row albedo 2.2–3.1 on 10–16 % of rows; azimuth-dependent albedo 0.04–1.7 for an isotropic material [AUDIT P0-1c, A-06, J1] | P4, and K-ownership (a learned local operator where ER is exact) | no learned local scattering factor anywhere; ER only |
| sender Bq never supervised, identified only through a product [AUDIT A-06, A-07] | P9 | no named latent; every learned output has a measurable target |
| M64 used as "exact sender" / teacher [AUDIT J2 item 11, J6] | P10 | references are path-traced or deterministic quadrature validated against path tracing |
| geometry-conditioned factors from world centre, world normal / tangent and a scene-global mean token (`model86.py:95, 99`) [CODE; not a WL97 finding] | P6 and dependency locality | inputs restricted to local frames and a bounded receptive field |
| two seeds, seed effects exceed treatment effects and flip sign [AUDIT §5] | statistics | ≥ 3 training seeds and paired, draw-resampled evaluation (§6.7) [HUM: exact count] |

---

## 3. Common physical frame for the kept candidates

### 3.1 Owners

| component | owner | learned? |
|---|---|---|
| local scattering at every vertex | frozen learned ER (WL84), `eval / sample / pdf` at exact directions | no new learning |
| visibility, hit points, distances | explicit ray queries on current geometry (relation records, probe rays, shadow rays) | never learned |
| direct incidence | exact per light and node: direction and irradiance via shadow rays | no |
| delivery / flight estimation | **the learned operator** (XQT: redistribution weights; LLT: lobe transfer) | yes |
| camera prefix | explicit as in ECRO (e1 at x1, flight x1 → x2, e2 at x2, table read at x2) | no |

### 3.2 Carrier

- Per element e (patch; sharp elements may carry sub-element nodes by an explicit rule), incident flux coefficients Φ^in_{e,c} on a projected-disk quadtree. Refinement depth is an explicit function of the element's own ER state (lobe width) and of nothing learned. Lambert elements: one cell (only irradiance matters for their emission).
- Outgoing radiance is never stored for elements whose ER is not Lambert. L(x, ω_o) = Σ_c Φ^in_{e,c} f̄_c(ω_o) / A_e + exact direct term, where f̄_c(ω_o) is the projected-mean of f over incident cell c at the exact outgoing direction (a per-material function, precomputed finely per incident cell; not per patch).
- This is family B used as a **state**, not as the learned object.

### 3.3 Allowed learned inputs (P6, P7)

Per relation ray or probe: hit element id class (material descriptor, not identity), hit point in the hit element's frame, direction in both frames, distance, both cosines, the exact per-ray sample value only where linearity permits (see each candidate). Per element: material descriptor, area, shape class, refinement level, and offsets of neighbours expressed in its own frame. **Excluded:** world positions or orientations, light identity or position, source identity, global scene statistics, T1 data, reference images, M64.

### 3.4 Event ownership

Same partition as ECRO (camera e1, e2 explicit; table at x2 holds ≥ 2 events at x2). The kept candidates own the table-level transport: XQT every flight; LLT only the first flight out of sharp senders, with every other flight explicit or XQT. Any arm mixing owners declares the event-count boundary; WL96A's `_p` / `_o` last-sender predicate is reused for attribution only [CODE AUDIT A-08].

---

## 4. Candidate contracts

Every candidate below has the full field list of the brief. Dispositions: PRIMARY (§4.1), ALTERNATIVE (§4.2), REJECTED (§4.3–§4.8).

### 4.1 PRIMARY — XQT: exact-sample quadrature transport (family: other; B carrier + learned flight estimator)

**Exact physical meaning of the learned transport.**
For one free-space flight, the explicit relation (K rays per receiver element, each with an exact per-ray sample of the sender's outgoing radiance evaluated by ER at the exact hit point and direction, as WL95A X1 / WL91 C3URX1) gives per-ray flux quanta φ_r = s_r · ω⊥_r · A_i', where ω⊥_r is the ray's projected-measure quadrature weight and i' its receiver. The explicit estimator assigns φ_r wholly to (i', cell of r): unbiased per receiver, heavy-tailed for sharp lobes. XQT's learned object is a sparse, non-negative, column-stochastic redistribution W_θ(G): each quantum φ_r is split among a bounded set of receiver cells (i, c) in a visibility-consistent neighbourhood of (i', r), with Σ_{(i,c)} W_{(i,c),r} = 1. Delivered incident flux Φ^in_{i,c} = Σ_r W_{(i,c),r} φ_r. Physically W_θ is a **learned quadrature rule for the flight integral**: it encodes how transport fields cohere across neighbouring receivers and directions (footprint shape of a lobe of known width at a known distance and angle, smoothness of broad fields, edge preservation where the records show a visibility change) and nothing else. The geometry of a reassignment (the direction cell in receiver i's frame of the hit point x_j) is computed exactly, not learned.

**Spatial and angular input / output.**
- Input per relation ray: the allowed records (§3.3) plus the hit element's ER descriptor; W does **not** take s_r (P1).
- Input per receiver: material descriptor, frame, area, neighbour offsets in its own frame, refinement level.
- Output: sparse weights W (stored as cached transport coefficients, Architecture §4 "Cached transport coefficients"). Per flight the field update is the sparse product Φ^in = W φ.
- Angular: inputs at exact directions (continuous); outputs on the flux-exact projected-disk cells at receivers; pointwise exact-ER emission for non-Lambert senders.

**Local / nonlocal ownership.** Nonlocal only: XQT is C's estimator, applied at every flight of the table-level transport (orders 1..J of the table, i.e. camera events 3..J+2). Local scattering is ER. No learned term owns any local response or any interior transport.

**Object-local coordinates and rigid transforms.** All inputs are relative or in element frames (§3.3). W is exactly invariant under a global rigid motion. Under a rigid edit of one object, W entries whose receptive field (the records of the rays and neighbours it reads) is bit-identical are reused unchanged; the rest are regenerated. This is a dependency certificate in the sense of Architecture §4 / §6 and WL70 (identical inputs ⇒ identical deterministic output), testable as selective = full.

**Interface to dynamic visibility.** The relation is built and selectively updated explicitly (rays crossing the old / new volume of a moved object or starting on it, as in ECRO / C3UR [MEAS WL86, WL89]). Redistribution is masked: quantum r of receiver i' may go to neighbour i only if i's own records contain a hit on the same hit element within an angular tolerance of the re-projected direction (explicit visibility consistency). XQT never predicts a hit.

**Energy / passivity contract.** Per quantum exact conservation (column sums 1), non-negativity (softmax), ER albedo ≤ 1 [MEAS WL82–84], flight closure ≤ 1 (missed rays leave). Every flight + scatter step has flux-norm ≤ max albedo < 1, so J-fold iteration converges; the composed spectral radius is checked with a valid estimator (A² iteration or Arnoldi, BASELINE doc) as a test, not as a constraint. Linearity exact. Reciprocity not imposed; measured.

**Training target and data generation.**
- Target: for a training geometry and an emitting state, the **expected** incident flux per receiver cell for that flight, from a deterministic exact-ER sender-area quadrature with visibility rays at fine resolution (the A-04 / E3 class of oracle, which matched the path-traced CK2 e2_p at 1.015 / 0.068 vs half-noise 0.059 [AUDIT A-04, A-05]); validated on a declared subset against path-traced fields. For flights of order ≥ 2 the target is the same deterministic quadrature applied to the exact previous-order field of that geometry (or path-traced order fields at qualified noise). Noise-free targets avoid the WL93 target-noise problems [AUDIT §5, WL93A].
- Training inputs: one or more relation draws per geometry with the per-ray exact samples. Loss: flux-weighted L2 per receiver cell plus a relative term on sharp-receiver cells, in the projected measure.
- Data: random geometries of the family, random emitting states stratified by incidence (the grazing share at sharp elements at least the evaluation workloads' share; WL92A training had ring-0 mirror-to-box share 0.027 % and 4 of 3,712 lightings at CK2 level [AUDIT B]), sender α range, sender–receiver distance, occluder edges inside footprints. All evaluation workloads excluded with geometric margins (fixing the CK2 leakage, B-02).

**Mechanism for sharp features.** Sharp angular structure is never approximated: ER is evaluated at exact directions per ray. Spatially distributed sharp delivery (glint footprints on receivers, footprint edges) is carried by the learned anisotropic, edge-aware pooling of the sparse hits on the lobe across receivers that see it. A learned kernel can be narrow where the lobe footprint is narrow (d · α) and refuse to pool across a recorded visibility change.

**Runtime evaluation and memory complexity.**
- Relation: N · K rays (650 k at N = 1,270, K = 512) as in ECRO; selective update as in C3UR.
- Exact samples per flight: N · K ER evaluations (direct light: × number of lights; indirect: × nonzero incident cells of the hit element).
- W: nnz ≈ N · K · n_nb (n_nb neighbourhood size, e.g. 9 → 5.9 M weights, ≈ 24 MB float32 at N = 1,270). Generation: one small MLP per (ray, neighbour) on a rebuild (≈ 6 M evaluations); after an edit only dirty receptive fields.
- Per flight apply: one sparse product (≈ 6 M MAC × 3 channels).
- Scaling: linear in N · K · n_nb. The claimed gain is a smaller K (or fewer draws) for equal error than the explicit gather.

**Scene conditions likely to fail.**
- Visibility structure finer than the neighbourhood sampling inside a footprint (contacts, thin occluders, near-coincident geometry): pooling is masked to safety and the gain disappears (WL52-type transitions).
- Lobes so narrow or senders so small that even the pooled neighbourhood holds < 1 expected hit (the CK2 glint is 0.46 expected rays per receiver patch per draw [MEAS WL96 §A]).
- Curved receivers / senders where neighbour re-projection is poor (sphere S2).
- A moving sharp sender: every record involving it is dirty; W regeneration dominates the edit.
- Distribution shift in α, grazing share or distances.
- Frame / overhead-dominated lifecycles: fidelity gain without lifecycle gain (Amdahl, §0 item 4).

**What learning buys over the best learning-free twin.** The twin is the same masked, flux-conserving redistribution with an analytic kernel (bandwidth from distance × lobe width, isotropic, fixed edge rule). Learning buys only a better-adapted kernel: anisotropy, footprint shape, edge decisions and order-dependent smoothing. If the learned weights do not beat the twin by a declared margin [HUM], XQT is an explicit estimator with a learned name and must be rejected (criterion 1 of the brief). §9 gives the scratch estimate: in the flatland toy, an oracle per-element adaptive kernel improves on the best fixed kernel only modestly; most of the pooling gain is available to the learning-free twin.

**Why it is not BRDF compression.** ER is used verbatim; W has no material response, only spatial / angular redistribution of already-exact flux.

**Unsuitability check (brief's criteria).**
1. Replaces / reuses explicit transport? It replaces the explicit *estimator budget* (draws) and is the only flight operator; the explicit relation remains its information. Whether that counts as "learned transport" is HD-1. Fails if learned η ≤ twin η by the declared margin.
2. Reuse after rigid edit: yes (weights fixed; clean W entries reused under the certificate).
3. Visibility not inside an intrinsic representation: yes.
4. Sharp response needs full explicit transport? It needs explicit visibility queries at a reduced budget and exact ER; it fails this criterion if the budget at which it meets the quality tier is not substantially below the explicit one (S1-c / S5).
5. Energy / measure contract: defined, by construction.
6. Independent reference: path tracing and the deterministic oracle.

**Disposition: PRIMARY.** It is the only kept design that is information-sufficient by construction (it reads every queried hit and invents none), measure-exact and passive by construction, rigid-invariant, and a direct response to the measured residual of the exact-sample route (P0-3: unbiased in expectation, heavy-tailed per draw). It is also the honest expression of Proposition I: in GM1 the defensible learned transport is an estimator.

### 4.2 ALTERNATIVE — LLT: learned lobe transfer from sparse probes (family C, delivery side)

**Exact physical meaning.** For a sharp sender node x with one exact incident direction ω_i (one light, or one exact incident cell), the exitant flux Φ_out = E(x) · a_ER(ω_i) is exact (ER albedo). LLT learns the **lobe form factor**: the fraction Q_θ(r | x, ω_i, G) of that flux arriving at each candidate receiver element r (softmax over candidates plus a miss class), and a normalized footprint density p_θ(y, ω | r, ...) over position within r and incoming direction (a continuous lobe / distribution representation with ∫ p = 1). Received flux: Φ_out · Q_θ(r) · p_θ.

**Spatial / angular I/O.** Inputs: the sender node frame, the exact ER lobe descriptor for ω_i (α, Fresnel term, exact lobe mean direction and width moments from ER, not learned), candidate receivers from an explicit coarse cone query or a coarse sender-anchored relation (relative pose in the sender frame), and K_p explicit probe rays drawn deterministically at stratified quantiles of the exact ER lobe with their hit records. Outputs: Q over candidates + miss; per received candidate, a continuous density (mixture of anisotropic Gaussians on the receiver's local plane × projected-disk lobes).

**Ownership.** First flight out of sharp senders only (the table-level e2 from sharp senders and, if the incident cells are exact, the sharp-last flight). Every other flight: explicit, or XQT in a combined system (not combined in the falsification plan, to keep the hypotheses separate).

**Rigid handling.** Inputs in the sender frame; candidate poses relative. Under a rigid edit, a lobe's Q / p is reused if its probes and candidate set are unchanged (certificate), otherwise re-evaluated.

**Visibility interface.** Explicit probes and candidate queries only; between probes LLT *infers* visibility of the footprint. This is the only candidate where a learned function goes beyond queried visibility (lever (c)); it is allowed only if the S1-d information test shows the probe records determine the delivery within tolerance on the evaluation distribution.

**Energy / passivity.** Σ_r Q + Q_miss = 1, ∫ p = 1, Φ_out exact and ≤ incident flux (ER albedo ≤ 1). Linear in the incident flux of each exact incident direction. Not iterated by itself.

**Training target and data.** Deterministic lobe form factors and footprint densities by dense quadrature (exact ER × dense visibility rays) on random geometries: noise-free, independent of M64, validated against path-traced e2 on a declared subset. Stratified by incidence (grazing), α, distance, occluder edges in the footprint.

**Sharp mechanism.** The lobe is exact (ER); the learned part represents where it lands, at a resolution not limited by any bin, from K_p probes plus learned priors on footprint shape.

**Runtime / memory.** Per (sender node, incident direction): K_p rays + |candidates| MLP evaluations. Cost ∝ nodes × lights (or × nonzero exact incident cells), per light like photons. Memory: per lobe a few floats per candidate; cacheable under the certificate.

**Failure conditions.** Occluders thinner than probe spacing or edges inside the footprint between probes (information limit); contacts; curved receivers; many lights (per-light cost, like the explicit lobe-tracing comparator); α outside training range; large candidate sets (wide lobes).

**Learned vs learning-free twin.** The twin traces the same K_p rays as stratified photons from the sender (each carries Φ_out / K_p to its hit) with a fixed footprint splat. LLT buys only within-footprint shape and inferred partial visibility; it must reach the twin's error with fewer probes by a declared factor [HUM].

**Why not BRDF compression.** The lobe is ER; the learned object is its delivery through geometry.

**Unsuitability check.** (1) Replaces explicit lobe tracing if η_LLT ≥ threshold; (2) reuse yes; (3) visibility partially inferred: the decisive risk; (4) fails if the probes it needs approach α-resolution tracing; (5) defined; (6) yes.

**Disposition: ALTERNATIVE.** Independent of XQT in information regime (sparse probes vs full relation), anchoring (sender vs receiver), scope (first sharp flight vs all flights) and failure mode (implicit visibility inference vs estimator efficiency). Not primary because it must infer unqueried visibility to beat its twin, it costs per light like the explicit comparator, and the nearest tested relative (WL52 learned C) failed at transitions. It is the cleanest test of the brief's "learned high-frequency delivery beyond explicit queries" and should run only if S1-d passes.

### 4.3 REJECTED — A1: continuous local directional kernel (family A, local reading)

- **Physical meaning:** a neural field k_θ(x, ω_i, ω_o | material) per element: the local scattering response, optionally spatially varying over the element.
- **I/O:** position in the element frame, incident and outgoing directions → radiance per irradiance.
- **Ownership:** local (K).
- **Rigid:** trivially in the material frame.
- **Visibility:** none needed.
- **Energy:** passivity by penalty or by a normalized-lobe parameterization.
- **Training:** fit to ER.
- **Sharp mechanism:** capacity of the field.
- **Cost:** one network per query versus one ER evaluation (69 floats per material [MEAS WL96 §D]).
- **Failure:** any α below the fitted capacity; it duplicates ER.
- **Learned vs explicit:** none; ER is already the exact, compact, passive local rule (and is itself the project's learned local closure).
- **Reason for rejection:** this is local BRDF compression, excluded by the brief, with no information gain over ER (WL96 d1 E-L+ reached the same conclusion).

### 4.4 REJECTED — A2: continuous nonlocal learned kernel / learned gather field (family A, nonlocal reading)

- **Physical meaning:** k_θ(y, ω | context) predicts the incident radiance at receiver y from direction ω (or a continuous transfer kernel from sender states to receivers), conditioned on geometry tokens or relation summaries.
- **I/O:** continuous receiver position and direction → incident radiance; context from sender states and geometry.
- **Ownership:** C (a learned C).
- **Rigid:** possible with relative inputs.
- **Visibility:** if conditioned on geometry tokens, it learns visibility (owns C); if conditioned only on queried rays it reduces to an estimator, but an unnormalized one.
- **Energy:** an unnormalized continuous kernel has no by-construction flux measure; passivity only by penalty, which is the WL86–WL95 failure mode (non-passive rows [AUDIT P0-1c]).
- **Training:** path-traced or deterministic incident fields.
- **Sharp mechanism:** network capacity in (y, ω).
- **Cost:** network evaluations per receiver query; WL52's dense learned C was 8–60× slower than explicit GPU C [MEAS WL52].
- **Failure:** contacts, visibility transitions [MEAS WL52].
- **Learned vs explicit:** amortizes visibility, which the canon and the brief give to explicit queries.
- **Reason for rejection:** either it owns visibility or it is a weaker (non-measure-exact) version of XQT. Its useful content is kept in XQT, which adds the measure and passivity constraints by construction.

### 4.5 REJECTED as a learned candidate — B1: sparse multi-resolution angular state with a learned resolvent (family B)

- **Physical meaning:** state = flux-exact projected-disk quadtree per element (§3.2); learned object = a geometry-conditioned operator for orders ≥ 2 (resolvent), with explicit first flight.
- **I/O:** sparse coefficients in, sparse coefficients out.
- **Ownership:** learned owns events ≥ 3 at the camera (orders ≥ 2 of the table); first flight explicit.
- **Rigid:** with local inputs.
- **Visibility:** explicit relation for the first flight; the resolvent consumes relation records.
- **Energy:** passivity of a learned resolvent is not by construction unless generated from passive factors.
- **Training:** order fields.
- **Sharp mechanism:** carrier refinement at sharp elements by rule; learned only for higher orders.
- **Cost:** sparse operator application.
- **Failure:** low higher-order share; cheap explicit sweeps.
- **Learned vs explicit:** order amortization only. Measured ceiling on CK2 1.02 / 1.04 (resolvent-only, frozen convention) [AUDIT P0-4]; explicit e3_o reproduces the reference tail within noise where the ROI reads it [AUDIT P0-4].
- **Reason:** the learned part is the higher-order resolvent behind an explicit sharp first flight: the default the brief excludes, with a measured null ceiling on CK2. **Kept:** the carrier (inside XQT) and the sharp-last observation (explicit binned e3_p error 0.235 / 0.123 deployed / pooled vs noise 0.100 [MEAS WL96A]), which XQT owns at every order.

### 4.6 REJECTED — C1: learned sender lobe / distribution representation (family C, sender side)

- **Physical meaning:** a learned encoder maps (incident state at a sharp element, ER descriptor) to a mixture of outgoing lobes (direction, width, flux) per element or node.
- **I/O:** incident field → lobe parameters; readout by lobe evaluation.
- **Ownership:** local K (re-expression of ER × incident field).
- **Rigid:** material frame.
- **Visibility:** none.
- **Energy:** flux-exact if lobes are normalized and amplitudes carry exact flux.
- **Training:** fit to ER responses.
- **Sharp mechanism:** lobe parameters.
- **Cost:** lobe evaluation per query.
- **Failure:** multi-lobe and grazing ER structure (Fresnel, refracted inner lobes of ER [MEAS WL96 §Verified facts]).
- **Learned vs explicit:** none for direct incidence (ER at exact directions is exact); for indirect incidence it is an approximation of Σ_c Φ_c f̄_c, which is explicit.
- **Reason for rejection:** amortized ER fitting, i.e. BRDF compression in a spatial wrapper.

### 4.7 CONTROL, not a candidate — E0: ECRO v0 lineage as frozen (WL86 / WL92A)

- **Physical meaning:** geometry-conditioned per-patch rank-16 local factor pair (a, c) used as sender, receiver and global decoder; one learned flight through 512 fixed state rays; rank-24 global term [CODE `model86.py:91-124`, AUDIT A §2].
- **I/O:** 64-bin solid-angle irradiance-by-incident-bin source encoding → 64-bin solid-angle-mean outgoing table.
- **Ownership:** learned local factors (overlapping ER's role), learned flight weights through explicit rays, learned global term.
- **Rigid:** not invariant: world centre, world normal / tangent and a scene-global mean token are generator inputs (`model86.py:95, 99`) [CODE]; every factor depends on the whole scene, so no local validity certificate exists after an edit.
- **Visibility:** explicit state rays; form factors also feed message passing.
- **Energy:** non-negative and linear by construction; passivity not imposed (non-passive rows [AUDIT P0-1c]); never iterated.
- **Training:** path-traced 2–5 event tables; sender latent unsupervised.
- **Sharp mechanism:** rank-16 factors in 64 solid-angle bins.
- **Cost:** generator 11 ms per edit, apply ~0.8 ms [MEAS WL87 / WL96A via AUDIT D].
- **Failure:** grazing sharp delivery (CK2), held-out sharp F2 (draw-dependent) [AUDIT P0-1, P0-5].
- **Use here:** frozen control C0 and the H1 arm (ECRO-COV, §6.3), never the candidate.

### 4.8 REJECTED under current canon — O1: asset-level reflectance-field K

- **Physical meaning:** for a rigid asset, a learned operator from the incident light field on a bounding proxy to the outgoing light field on the proxy, including exposed self-interreflection inside the proxy.
- **I/O:** 4D light field in, 4D light field out (proxy frame).
- **Ownership:** puts exposed same-asset free-space paths into K.
- **Rigid:** exactly reusable under rigid motion of the asset (proxy frame).
- **Visibility:** intra-proxy visibility owned by the asset; only valid while no foreign geometry enters the proxy.
- **Energy:** passivity constrained by normalization.
- **Training:** per-asset fit (offline).
- **Sharp mechanism:** network capacity over 4D × 4D.
- **Cost:** large (α⁻⁴-type content per asset).
- **Failure:** any intrusion into the proxy, contact, deformation.
- **Learned vs explicit:** memory compression of a rigid-invariant operator; the only family with genuine state reuse.
- **Reason:** (i) Architecture §2 assigns exposed same-asset paths to C and forbids merging editable air transport into K; (ii) in the scene86 / GM1 family every object is planar or convex, so the proxy operator equals ER and has no content; (iii) close to the external 8DNA neural-asset class, whose rigid case is Native Reuse. Changing (i) is a canon decision [HUM, outside this batch].

---

## 5. Comparison

| | A1 | A2 | B1 | C1 | **XQT** | **LLT** | E0 | O1 |
|---|---|---|---|---|---|---|---|---|
| learned object | local kernel | nonlocal kernel | resolvent | sender lobes | flight quadrature weights | lobe form factor + footprint | local factors + flight + global | asset light-field K |
| information beyond explicit queries | none (ER) | visibility | none | none | none (estimator) | partial (inferred footprint visibility) | none / mixed | intra-asset visibility |
| measure by construction | no | no | carrier yes | yes | yes | yes | no | possible |
| passive by construction | penalty | penalty | no | yes | yes | yes | no | possible |
| linear in source | yes | possible | yes | no (encoder) | yes | yes per incident direction | yes | yes |
| rigid-invariant inputs | yes | possible | possible | yes | yes | yes | **no** (world coords) | yes |
| owns visibility | no | **yes** | no | no | no | partly | no | **yes** (intra-proxy) |
| carries sharp spatially distributed transport in the learned part | no | yes | no | no | yes (delivery) | yes (delivery) | yes (failed) | yes |
| measured negative prior | WL96 | WL52 | WL96A | WL96 | none direct; X1 variance is the target | WL52 | WL86–WL95 | none in GM1 |
| disposition | REJECTED | REJECTED | REJECTED (carrier kept) | REJECTED | **PRIMARY** | **ALTERNATIVE** | control | REJECTED (canon) |

---

## 6. Falsification plan

### 6.1 Hypotheses

- **H1 (training coverage / optimisation).** The CK2-type sharp failure of the frozen ECRO representation is caused by training coverage (grazing ring-0 sharp transport in 4 of 3,712 lightings, 0.11 % [AUDIT B-04]) and by the unsupervised latent sender, not by its angular representation.
- **H2 (angular representation).** The failure is caused by the representation: the solid-angle-mean 4 × 16 table and its patch-constant position (convention + in-bin redistribution + spatial readout [AUDIT J2, J4]).
- **H3 (utility).** An improved learned representation that meets fidelity with fixed weights still gives no geometry-edit utility over a quality-matched explicit comparator.
- **H_X (XQT).** Learned redistribution reaches a declared sharp-region quality tier with ≥ η_req times fewer relation rays than the explicit exact-sample gather and beats its learning-free twin by a declared margin, on held-out geometry, at T0 and T1, with fixed weights.
- **H_L (LLT).** K_p explicit probes per lobe carry enough information to determine sharp delivery within tolerance on the evaluation distribution (information test), and LLT then beats its twin by a declared factor.

H1 and H2 are not exclusive; the plan measures each share (§6.6).

### 6.2 Workloads and admission (all declared before any reference is rendered)

| id | role | content | admission |
|---|---|---|---|
| W-G | grazing sharp, **diagnostic only** | CK2 'none' (scene86 room, light 5 cm in front of S1) with the CKF T0 / T1 box-A edit | needs a new path-traced **T1** reference (none exists [AUDIT D-13, J7]); excluded from all training (the CK2 leakage B-02 is removed); cannot approve an architecture on its own |
| W-G2 | grazing sharp, **evidence** | a new grazing configuration in a different geometry of the family (sharp sender, near-field light at cos_i < 0.2, Lambert and glossy receivers, at least one receiver in the glint core and one in the tails, so that both signs of the J2 receiver dependence appear) | geometry-only prequalification: deterministic exact vs solid-angle 4×16 delivery difference ≥ d_min [HUM]; SNR ≥ 4 at the reference budget (WL86 / WL96 rule); sharp share of the T0 → T1 change ≥ s_min [HUM] |
| W-N | non-grazing sharp, evidence | panel / slab lit at cos_i > 0.4 (E1 / E2-type lights; the binned bias is small there [AUDIT J2 single-direction ratios]) with a rigid occluder edit that enables / blocks a sharp path | same rules |
| W-H | independent held-out, evidence | a geometry never inspected during design or training: the WL89 V3 / W2 moving-sharp-slab family if a never-used partition exists (a rigid rotation of the *sharp sender*, which tests rigid equivariance and full dirtiness of a moving sharp object), otherwise a new member generated by a frozen rule | same rules; inspected by nobody before the frozen chain |
| W-S | synthetic contracts | analytic fixtures (§6.8) | exact answers |
| W-X (optional) | scale axis | the W-N geometry at N ∈ {1,270, ~5,000, ~11,000} and 1 / 4 / 16 lights | [HUM]: include or not; it is the only place a cost-side H3 ceiling can rise [AUDIT D-04, E5-5] |

**Edits:** for each evidence workload, T0 → T1 rigid: one occluder translation (enables / blocks a sharp path) and, in W-H, one rigid motion of a sharp object. **Train / validation / test separation:** training geometries and lights by a frozen generator with exclusion margins around every evaluation configuration (ECRO used 0.15 m per box and 0.3 m per light [MEAS WL86]); validation geometries from the generator, disjoint; tests are W-G2, W-N, W-H only. No T1 data and no evaluation-workload data at any training or selection step.

**References:** Cycles is not used here; the project's own path tracer `pt86` family with frozen ER (validated against Mitsuba's path integrator within 0.6 % [MEAS WL86 V1]) for images and fields; two independent halves per reference; the deterministic exact-ER quadrature oracle as an independent check of first-flight fields (it matched path tracing on CK2 [AUDIT A-04, E3]). **M64 is never a reference or a teacher.**

### 6.3 Arms

| arm | content | trained? |
|---|---|---|
| C0 frozen ECRO | WL86 v0 and WL92A OWN checkpoints, frozen | no |
| C1 ECRO-COV (H1 arm) | same `model86` architecture, parameter count, steps and optimiser; only the training distribution (grazing-stratified lights so that the evaluation workloads' incidence classes are covered at a declared share) and an added *physical* sender target (deterministic exact-ER first-scatter field projected flux-exactly to the same 64 bins, never M64) change | yes, ≥ 3 seeds [HUM] |
| C1f ECRO-COV-fx | as C1, but targets and table in the flux-exact 4 × 16 convention (separates the convention part of H2 inside the learned arm) | yes, ≥ 3 seeds (optional [HUM]) |
| R0 representation floors (H2) | direct projection of the reference fields into: solid-angle 4×16 (M64 convention), flux-exact 4×16, flux-exact 8×32, flux-exact projected-disk adaptive (the XQT / B carrier), exact per-ray (X1 pooled / deterministic) | no |
| X explicit comparators | EXP / C3UR as frozen; C3URX1 (exact first gather) at 1, 2, 4, 8, 16× relation draws; learning-free sharp carriers allowed as comparators by canon (photons, deterministic quadrature) [HUM: which] | no |
| T-XQT twin | XQT structure with the analytic kernel | no |
| XQT | §4.1 | yes, ≥ 3 seeds |
| T-LLT twin / LLT | §4.2, only if S1-d passes | yes, ≥ 3 seeds |
| STALE | T0 state at T1 for each trained arm | no |

### 6.4 What can and cannot share data, budget and reference

| comparison | training data | budget | physical reference | visibility budget |
|---|---|---|---|---|
| C0 vs C1 | **no** (C1 changes the distribution by definition) | same parameters, steps, optimiser | shared | same state rays |
| C1 vs C1f | same geometries and lights; different target projection | same | shared | same |
| C1 vs XQT | same training geometries, lights and generation compute class; targets differ in form (64-bin table vs per-cell flight fluxes) and are compared only at common interfaces (exact (x, ω) probes and images) | matched training compute and wall [HUM] | shared | XQT declares its K; C1 uses its 512 state rays; reported in rays |
| XQT vs twin | identical inputs and relation draws (common random numbers) | twin has no training | shared | identical |
| XQT vs explicit X | no training on the explicit side | **quality-matched** cost comparison (not equal budget) | shared | explicit varies its draws |
| LLT vs XQT | different information regimes; never compared at equal rays without stating it | separate | shared | stated per arm |
| R0 floors vs any learned arm | no training | none | shared | n/a |

### 6.5 Stages, gates and stop rules

All thresholds marked [HUM] are not determined by current evidence; they must be frozen before any Stage 0 render.

**S0 — Admission (learning-free).** Workload rules of §6.2, references, two-half noise, deterministic-oracle agreement on first-flight fields (ratio within 2 SE, rel L1 ≤ n0 + 2 SE as WL96 §F V0). Any failure → `STOPPED_UNQUALIFIED_WORKLOAD`; an amendment needs explicit human approval (AGENTS.md amendment rule).

**S1 — Learning-free decomposition (decides whether training is warranted).**
- **S1-a (H2 floors).** Every R0 floor on every evidence workload at T0 and T1: field errors at exact-direction probes, delivered flux per receiver group (glint-core and tail receivers separately, because the sign differs [AUDIT J2]), and oracle-table image error in the sharp ROI.
- **S1-b (explicit frontier).** Error versus measured cost for X arms at 1–16× draws, paired over ≥ 16 relation draws, T0 and T1; the cost of reaching each quality tier.
- **S1-c (XQT ceilings).** On the same draws: the twin, and an oracle pooling (per-source-cell kernels chosen with the exact field; optimistic bound for any learned weights of the XQT form). η_twin, η_oracle and the oracle / twin error ratio per region. **Stop XQT** if η_oracle < η_req [HUM] in the sharp ROI of two evidence workloads, or if the oracle / twin ratio < m_req [HUM] (learning cannot buy a meaningful margin even in principle).
- **S1-d (LLT information test).** For each test lobe, the irreducible error of any function of the K_p probe records, estimated by collision pairs (configurations with identical probe records and different unprobed visibility, sampled from the training generator) and by the fraction of footprints containing an unprobed visibility transition. **Stop LLT** if this floor exceeds the sharp-ROI tolerance [HUM] at the K_p that would make LLT cheaper than its twin.
- **S1-e (H3 ceilings).** For each surviving candidate: zero-cost replacement of its role → U_max for edit → valid frame and source → frame, measured uncontended with A-B-A blocks and compilation excluded (WL96A / E5 harness); plus the quality-matched ceiling from S1-b (explicit cost to reach the tier the oracle candidate reaches at its budget). **Stop** a candidate if both ceilings are below U_req [HUM] on two evidence workloads (and on the scale axis, if W-X is included).

**S2 — H1 arm.** Train C1 (and C1f) on the declared generator, ≥ 3 seeds; evaluate at T0 / T1 on all evidence workloads with frozen weights. No stop by itself; it feeds the decomposition.

**S3 — Candidate training.** Only candidates that passed S1. Same generator and compute class as C1. Contract tests (§6.8) must pass before evaluation; a failing contract test stops that candidate (`STOPPED_CONTRACT`), no repair inside the batch.

**S4 — Held-out evaluation, fixed weights.** W-G2, W-N, W-H at T0 and T1 (and W-G as diagnostic), all arms. Selective = full (W / Q reuse certificate) bitwise or within float tolerance. STALE control must be worse than updated somewhere in the changed region (the WL86 control form).

**S5 — Economics.** Quality-matched edit → valid frame and source → frame versus the cheapest X arm that reaches the same tier, per region; full-scale GPU utilisation probe of every stage before freeze (memory note: utilisation is a hard requirement); training and data-generation cost reported separately as one-time cost.

### 6.6 Decision mapping (frozen before S0)

Let e(·) be the error in the declared sharp-ROI metric at T1 on the evidence workloads, paired, seed-averaged with seed spread reported.

| observation | reading |
|---|---|
| e(C1) − e(R0 floor of the 64-bin solid-angle representation) within tolerance, and e(C0) − e(C1) resolved | **H1 supported** for that workload (coverage / identifiability explains the gap) |
| e(R0, 64 solid-angle) above tolerance and e(R0, flux-exact or finer) within it | **H2 supported** (representation floor); C1f splits the convention part |
| both | shares reported: gap_H1 = e(C0) − e(C1), gap_opt = e(C1) − e(R0 floor), gap_H2 = e(R0, 64 sa) − e(R0, candidate carrier) |
| a candidate meets fidelity at T0 and T1 with fixed weights, U (quality-matched, S5) ≥ U_req on two evidence workloads, and beats its twin by the declared margin | **H3 refuted** for that candidate on the tested scope: candidate eligible for an owner decision (not adoption) |
| a candidate meets fidelity but U < U_req, or does not beat its twin | **H3 supported** for that candidate (no learned utility on the tested scope) |
| no candidate meets fidelity | report attribution by H1 / H2 shares; no candidate |
| W-G only | diagnostic; cannot change any label above |

### 6.7 Evidence to collect

**Representation level.**
- Angular response fidelity at exact (x, ω) probes against the deterministic oracle and path tracing (flux-weighted rel L1 / L2 per receiver cell).
- Sharp lobe / footprint position and width: angular or spatial error of the peak, second-moment width ratio, peak retention.
- Spatially distributed transport error per receiver group (glint core, tails, broad, occluder-edge cells).
- Energy: emitted / incident flux per sender, closure per receiver, composed spectral radius (valid estimator); passivity violations counted (must be 0 by construction for XQT / LLT).
- Sampling variance: per-draw SD over ≥ 16 relation draws.

**Image level.**
- Error against the high-quality reference: E_H as in WL86 (shared prefix noise excluded).
- Sharp-interreflection ROI: defined from geometry before references.
- Changed-region fidelity (regions A / C of `masks86`).
- Leakage / ghosting: radiance in regions that must be dark or unchanged at T1 (blocked beam, stale glint), measured against the reference.
- Extra T1 error over T0 per arm with fixed weights: Δe = e_T1 − e_T0, paired.

**System level.**
- Update critical path (edit → valid state), steady-state frame cost, learned inference cost, relation / visibility recompute, W / Q regeneration after edit, memory, precomputation and training as one-time cost.
- Utility versus the quality-matched explicit comparator, both edit → frame and source → frame.

**Statistics.**
- Paired evaluation with common random numbers across arms.
- ≥ 3 training seeds [HUM: exact number]; seed spread reported per region; no label decided by one seed.
- Relation / state-draw resampling (≥ 16 draws) for draw-dependent arms; no label decided by a single deployed draw (WL95A FLIGHT_ATTRIBUTION flipped in 23 / 31 / 10 of 64 leave-one-out choices [AUDIT §5]).
- Reference noise from two independent halves; debiased L2 from cross products; block bootstrap over pixels / patches.
- Paired tolerance max(2 SE, τ_floor) [HUM: τ_floor]; never an L1 noise floor as a paired tolerance (R1, R2 [AUDIT §4]).

### 6.8 Synthetic contract tests (W-S) and why both test kinds are needed

| test | fixture | pass condition |
|---|---|---|
| T1 measure | analytic lobe on a plane; projected-disk cells | stored cell flux = analytic projected integral to float tolerance; parent = Σ children |
| T2 grazing flux | single grazing incidence (cos_i 0.025–0.15) on an S1-like ER | emitted flux / ER albedo = 1 (solid-angle convention fails this by 1.3–1.8 [AUDIT J2]) |
| T3 linearity | two lights vs their sum | exact superposition |
| T4 zero / non-negativity | zero emission; random fields | zero out; no negative entry |
| T5 passivity | closed white Lambert enclosure (furnace) and a glossy enclosure | iteration converges; radiance = analytic furnace value; spectral radius < 1 with a valid estimator |
| T6 reciprocity | swapped sender / receiver pairs | F_ij A_i = F_ji A_j within MC tolerance (measured for estimators) |
| T7 rigid equivariance | whole scene rigidly moved | identical outputs in local frames; one object moved with untouched receptive fields → identical W / Q entries |
| T8 visibility counterfactual | insert an occluder that blocks no queried ray / one that blocks queried rays | output unchanged / changed only through the affected records |
| T9 planar-mirror exactness | α → small, planar sender, point light | delivery = image-source construction blurred by the ER lobe |
| T10 event accounting | prefix + learned tail vs path tracing at fixed depth | no double count (V1-type check of WL86) |

Synthetic tests are needed because they isolate contract violations (measure, passivity, linearity, equivariance, visibility ownership) with exact answers and without Monte Carlo noise; the WL86–WL95 chain carried measure and passivity defects for many batches because real-scene metrics hid them. Real-scene evaluation is needed because contracts say nothing about fidelity on realistic distributions, grazing and edge configurations, composition over orders, edit response or economics, and only path-traced references there are independent of the candidate.

---

## 7. Human decisions required

- **HD-1 (admissibility).** In the GM1 surface regime, Proposition I says the defensible learned role is an *estimator* of the geometry-dependent transport operator that reads explicit visibility samples. Is that admissible as the project's core learned transport (XQT), or must the learned part carry information beyond explicit queries (only LLT tries, at the cost of inferred visibility)?
- **HD-2 (comparators).** Which learning-free sharp carriers are comparators in S1-b / S5 (C3URX1 draws, photons, deterministic quadrature)? Canon allows them as comparators (Gate A), not as the method.
- **HD-3 (thresholds).** d_min, s_min, sharp-ROI tolerances, η_req, oracle / twin margin m_req, U_req (the historical 1.25 rule or another), τ_floor, number of seeds.
- **HD-4 (workloads).** Geometry of W-G2 and W-N; whether a never-used partition of the WL89 V3 family qualifies as W-H; the new CK2 T1 reference.
- **HD-5 (generator inputs).** Adopt the restriction of learned inputs to local frames and bounded receptive fields (no world coordinates, no scene-global pooling) for the ECRO lineage. It changes the lineage's contract (E0 violates it).
- **HD-6 (scale axis).** Include W-X? Without it, a cost-side H3 result at N ≈ 1,270 is predetermined to be small by the measured Amdahl ceilings.
- **HD-7 (LLT).** Whether LLT runs at all, given that it infers footprint visibility between probes.
- **HD-8 (canon, outside this batch).** Whether an asset-level K owning exposed self-interreflection inside a certified proxy (O1) should ever be considered (GM2 question).
- **HD-9 (batch class and compute).** Regular vs suffix numbering of the follow-up (by AGENTS.md this is a new mechanism, but the owner decides), server and GPU budget.

---

## 8. Proposed verdict

**DESIGN_READY_FOR_HUMAN_REVIEW** (not adoption, not experiment approval).

Rationale.
- A primary (XQT) and an independent alternative (LLT) have complete contracts that satisfy the physics requirements P1–P10 by construction where the physics allows, keep visibility explicit (fully in XQT, partly in LLT under a gate), use rigid-invariant inputs, carry the sharp spatially distributed delivery in the learned part, and have independent reference methods.
- The falsification plan separates H1 (C0 vs C1 vs R0 floors), H2 (R0 ladder and C1f) and H3 (S1-e ceilings and S5 quality-matched economics), lets the learning-free Stage 1 stop each candidate before training, and keeps CK2 diagnostic.
- Qualifications I want carried with the verdict:
  - [PRED] Proposition I limits learned value in GM1 to estimation and amortization; neither candidate can add physical information except LLT's gated visibility inference.
  - [PRED] At the current scale the cost-side utility ceiling is near the measured 1.24–1.37 even for a zero-cost full operator [AUDIT D-04]; a positive H3 needs quality-matched gains or the scale axis.
  - [SCRATCH] In the flatland toy (§9), flux-conserving pooling gains 8–50× in ray-equivalent terms over the unpooled exact gather. Almost all of that gain is available to the learning-free twin: an exact-field oracle kernel improves on it by only 1.0–1.33× in error (isotropic family; edge-aware and anisotropic kernels untested). If that holds in S1-c, XQT stops at Stage 1 and the batch would correctly end with no learned candidate in this regime.
- If the owner rules under HD-1 that an estimator-type learned operator is not the project's learned transport, my verdict for the GM1 surface regime becomes **NO_DEFENSIBLE_CANDIDATE** except for LLT, whose value depends on the S1-d information test.

---

## 9. Scratch calculation (design input only, not evidence)

**Purpose.** Estimate, before any 3D work, how much a geometry-adaptive, flux-conserving pooling of exact-sample gathers (XQT's learned role) can gain over (a) the unpooled explicit gather and (b) its learning-free twin (one fixed kernel).

**Setup** (`experiments/wl98_calc/D1/flat_pool.py`). Flatland. A glossy sender segment carries a cos-weighted Gaussian lobe in angle (α 0.03 / 0.10, passive) under a flatland point light. A Lambert receiver line is divided into 128 elements, and 75 % of the configurations contain an opaque occluder.
- Two families: **grazing** (cos_i ≈ 0.04–0.35, receiver line low so that both glint-core and tail receivers occur) and **non-grazing** (lobe at 15–45°).
- 40 configurations per family × α: 20 to choose the twin's kernel, 20 for test.
- Receiver gather: cosine-weighted, with exact sender evaluation (the X1 analogue).
- Expected errors are computed analytically as bias² + variance of the linear pooling. The inputs are the exact element field (dense quadrature) and per-sample variances (32,768 samples per element; the median rel L1 between the MC mean and the exact field is 0.02–0.10).
- Pooling is column-stochastic, so each element's estimated flux is conserved exactly.
- The "oracle" chooses the bandwidth per source element with the exact field (coordinate descent). It is an optimistic bound for a learned kernel **within this isotropic family**. It is not edge-aware and not anisotropic.

**Results** (median over 20 test configurations; errors are relative L2 over the receiver field; η = (raw / arm)², the ray multiple the unpooled gather needs to match):

| family, α, K rays / element | raw | twin (fixed kernel) | oracle per-element kernel | η twin | η oracle | twin / oracle error |
|---|---|---|---|---|---|---|
| grazing 0.03, 16 / 64 / 256 | 5.45 / 2.73 / 1.36 | 0.74 / 0.52 / 0.36 | 0.72 / 0.42 / 0.24 | 50 / 34 / 20 | 50 / 41 / 30 | 1.00 / 1.11 / 1.27 |
| grazing 0.10, 16 / 64 / 256 | 3.21 / 1.61 / 0.80 | 0.47 / 0.29 / 0.17 | 0.44 / 0.24 / 0.14 | 47 / 31 / 20 | 53 / 44 / 33 | 1.06 / 1.21 / 1.26 |
| non-grazing 0.03, 16 / 64 / 256 | 1.93 / 0.96 / 0.48 | 0.40 / 0.27 / 0.18 | 0.35 / 0.21 / 0.13 | 25 / 13 / 7.6 | 33 / 20 / 12 | 1.16 / 1.29 / 1.33 |
| non-grazing 0.10, 16 / 64 / 256 | 1.08 / 0.54 / 0.27 | 0.27 / 0.17 / 0.10 | 0.21 / 0.13 / 0.08 | 18 / 11 / 7.9 | 26 / 17 / 11 | 1.26 / 1.30 / 1.19 |

**Reading** [SCRATCH, design input only].
- Flux-conserving pooling of exact samples is worth a large ray multiple over the unpooled exact gather: 8–50×, largest at grazing and low K. This is consistent in kind with the measured "≥ 16× draws needed" of P0-3 [AUDIT].
- **Almost all of that gain is available to the learning-free twin.** Even an oracle that knows the exact field improves on the best fixed kernel by only a factor of 1.0–1.33 in error (1.0–1.6 in ray-equivalent terms) within the isotropic family.
- The toy does not measure edge-aware or anisotropic kernels. Those are the places a learned 3D kernel could add more, and they are exactly what S1-c must measure with a richer oracle on the real workloads.
- If S1-c reproduces a twin / oracle ratio in this range, XQT should stop at Stage 1: learning would then buy too little over its explicit twin to count as learned transport (brief criterion 1).
- Absolute errors stay large in this toy (rel L2 0.08–0.74 after pooling) because a 1D receiver line with 128 elements and sub-element glints is harsh. Only the ratios are design-relevant.

**Hashes.**
- `flat_pool.py`: da7c779913f3656c (sha256, first 16 hex).
- `flat_pool.json`: eae63e05187b7668.
- Seed 98001. Local CPU, numpy 2.4.6, about 15 min.

---

## 10. Provenance

- Code read: `experiments/ecro86/model86.py` (lines 40-124: state-ray weights, `factors` inputs at line 95, message passing at 98, global mean token at 99, `apply`), `experiments/ecro86/protocol/ecro86.json` (fixture, E1 / E2, light sets).
- Audit inputs: `docs/audits/wl97_evidence_reliability/{README, COORD_synthesis_and_p0_ledger, A_math_physics_contract, J1_*, J2_*, J4_* §0, J5_* §1, D_* D-04}.md`.
- Worklogs: 86 (operator, fixture, costs), 89 (V3 workload), 96 (carrier facts, X1 variance, cost anchors), 96A (event split, explicit e3_p / e3_o table).
- Scratch: `experiments/wl98_calc/D1/flat_pool.py` → `results/wl98_calc/D1/flat_pool.json` (local CPU, numpy 2.4.6, seed 98001).
