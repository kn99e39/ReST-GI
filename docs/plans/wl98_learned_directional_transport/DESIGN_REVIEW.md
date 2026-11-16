# WL98 Design Review — Learned Directional Transport after ECRO

**Status.** Design only, unadopted.
- Nothing is implemented, trained or rendered in this batch.
- Nothing here changes `docs/Architecture.md`, the research direction or any verdict.
- Adoption, workload extension, numerical thresholds and any experiment are human decisions (§9).

**Author.** The coordinating session. It authored WL86–WL96A and has a conflict of interest. This draft was written before the WL98 workflow ran. Its independent assessment is in the workflow reports in this directory (designers D1–D3 were blind to it; critics X1–X4 and judges JUDGE1/2 review it as one of four designs); see §10.

**Labels.**
- [MEAS] a measured value with its source.
- [AUDIT] a WL97 audit interpretation. The WL97 proposed annotations are *not adopted*.
- [PRED] a new prediction made here, not measured.
- [HUM] a human decision.

---

## 0. Question

> Can spatially distributed, high-frequency light transport, learned with a continuous or adaptive angular representation, be reused after a geometry edit without neural retraining, with a meaningful quality or cost advantage over explicit transport?

## 1. Established evidence (what the design must respect)

| # | evidence | kind | source |
|---|---|---|---|
| E1 | The learned S1 (panel) sender of the two OWN checkpoints is far from physical at CK2: field rel L1 1.08 / 1.19 vs physical q, box-A delivery 0.52× / 1.63× (seed-signed). In distribution its error is ≈ 0.8 rel L1 vs M64 at roughly correct energy | [MEAS] reproduced by WL97 E1 | WL95A B1; `docs/audits/wl97_evidence_reliability/E1_*.md`, `J1_*.md` |
| E2 | A passive non-negative rank-16 operator exists that is exact on the CK2 incident support. A light-independent rank-16 fit under training incidence weights fails CK2 (0.90); up-weighting CK2 trades away fidelity elsewhere (51 % → 0.59; CK2-only → 0.98 elsewhere) | [MEAS] + [AUDIT] (allocation, not capacity, for CK2; an all-incidence trade-off remains) | `J1_*.md` (F0-08 / B-06 ruling), `results/audit97/J1/`, `results/audit97/B/rank16_allocation.json` |
| E3 | Grazing ring-0 sharp transport occurs in 4 of 3,712 training lightings (0.11 %) and is never directly supervised. Non-grazing panel-to-box sharp transport occurs in 47 % | [MEAS] | `J7_*.md` (B-04) |
| E4 | The solid-angle-mean 4×16 sender over-delivers to CK2 box A by +40–50 % deterministically. About half is the convention's grazing over-emission (a flux-conserving 4×16 table gives ≈ +20 %). The sign depends on the receiver (glint receivers are under-delivered 5–8×). At 8×32 the bias is essentially gone. Single configuration | [MEAS] + [AUDIT] | `J2_*.md`, `A_*.md` (A-03 / 04), `E3_*.md` |
| E5 | Exact-direction plus hit-point evaluation removes the bias in expectation (pooled ratio 1.02, abs rel L1 0.11 vs n0 0.059; ROI E_H 0.0245 → 0.0124). A single draw at the deployed budget gives no average gain (heavy tail); ≥ 16× the draws are needed. Direction carries ≈ 70 % of the closure | [MEAS] | `J3_*.md`, `E2_*.md` |
| E6 | WL96A: explicit 3-sweep recurrence from a path-traced 2-event prefix reproduces the ≥ 3-event tail at reference noise, but only in table space, with an oracle prefix, CK2, for what ROI_T reads. The floor table itself has a ≈ 0.010 rel-L2 structured error vs path-traced truth from **patch-spatial averaging read through the sharp reflection**. Unread receivers carry real explicit errors (sphere e3_o flux 0.954 / 0.977) | [MEAS] + [AUDIT] | `J4_*.md`, `E4_*.md` |
| E7 | The resolvent-only cost ceiling is 1.02 edit / 1.04 source with frame (≤ 1.13 without frame) at N 1,270–11,430, in this prototype's wall time. The q table (24 ms) and frame (58 ms) dominate. The generator-arm "0.92×" was an OpenMP artefact (corrected ≈ 0.97×) | [MEAS] + [AUDIT] | `J5_*.md`, `E5_*.md`, `D_*.md` |
| E8 | No reference-validated dynamic edit exists from the WL93D addendum to WL96A. The only qualified never-inspected held-out edit (E5) shows learned ≤ explicit within about one noise bound at T1 (single draws) | [MEAS] | `J7_*.md` (F0-03, F0-01 / 02) |
| E9 | The learned ECRO local factor pair has non-passive rows (directional albedo up to 2.2–3.1) in all 14 checkpoints examined. Energy on the CK2 input is sub-unity; no consequence was tested | [MEAS] | `J1_*.md`, `C1_*.md` |
| E10 | The scene86 / GM1 family contains only planar or convex objects (walls, S1 panel, S2 sphere, two Lambert boxes). There is no object-intrinsic nonlocal transport: all nonlocal transport is inter-object (free space, "C") | [MEAS] fixture definition | `experiments/ecro86/protocol/ecro86.json` "fixture" |
| E11 | Learned inter-domain C (WL52) was valid and stable but 8–60× slower than GPU explicit C and failed contact / visibility-transition locality | [MEAS] | WL52 worklog; memory note wl52 |
| E12 | Prior art (not ReST-GI novelty). PRT and object-local transfer for rigid dynamic scenes with interreflection (Sloan 2002; Iwasaki 2007; Pan 2007 radiance-transfer fields). SG / vMF lobe representations and learned lobe prediction (Wang 2009; Currius 2020; parametric-mixture path guiding, Vorba 2014 / Dong 2023). Neural radiance caching (Müller 2021, online, i.e. retrained). Neural PRT (Rainer 2022). Object-centric neural transfer for dynamic scenes (TransGI 2025: stated weakness, bright glossy↔glossy interreflection). Scene-generalizable learned transport embeddings (arXiv 2510.18189, mainly diffuse) | literature; quick check, not exhaustive | see the Sources list in the WL98 worklog |

**Consequences for the design.**
- (a) The angular *carrier* problem (E4) is solvable without learning, by a flux-conserving table at higher resolution. A learned representation must therefore justify itself against **8×32 flux-conserving** and against an **explicit lobe twin**, not against the biased 4×16 table.
- (b) The *sender* problem (E1–E3) is confounded by coverage and allocation. Any learned rule needs stratified grazing coverage and a learning-free class floor before failure is read as capacity.
- (c) The exact-evaluation route (E5) has a variance problem; deterministic delivery is the lever.
- (d) In GM1, with only convex and planar objects (E10), "intrinsic" learned transport has no nonlocal content. The reusable learned element must be a **shared rule** that composes with explicit, current-geometry coupling. A per-asset intrinsic state test needs assets with self-transport (§9 HD-2).
- (e) A learned C over pairs is disfavoured (E11).
- (f) Utility can only be judged on reference-validated T0 → T1 edits (E8).

## 2. Candidates

All candidates share one skeleton, so that they differ only where the hypothesis is.

- **State.** The directional radiance state per surface element *e* (patch or sub-patch), expressed in *e*'s local shading frame.
- **Coupling C(G).** Explicit and learning-free. It delivers the outgoing state of senders to receivers through **current geometry visibility** (rays traced in the current scene; dirty-update allowed). It is never learned.
- **Learned rule Λ_θ.** One shared model for all elements, materials and scenes. It maps (the incident state of *e*, *e*'s descriptor) to the outgoing state of *e*.
  - The descriptor is the frozen ER material state (69 floats), local geometry scalars (element area, curvature, sub-element normals if any) and nothing about other objects.
  - Rigid edits change only C's visibility and the frames of moved elements. Λ_θ is reused unchanged.
- **Multi-bounce.** Iterate Λ_θ and C (J sweeps), or use an optional learned lobe-space resolvent, gated in §7, Stage 1.
- **Camera readout.** At the camera vertex, exact ER is evaluated against the incident state at x1, or the outgoing state at x2 is read through the sharp vertex.

### Candidate A — continuous directional learned kernel (neural field)

- **Definition.** A per-element latent code z_e is decoded by a shared directional decoder f_θ(z_e, ω, u), where u is the position inside the element, into radiance. Λ_θ maps incident codes to outgoing codes. Alternatively, as a nonlocal kernel: K_θ(x, ω_o; x', ω_i).
- **Physical meaning.** The outgoing radiance field of element *e* given its incident field: element-local transport.
- **Delivery.** Radiance at exact query directions (as X1 does). Integration over a receiver needs sampled queries, so the variance problem of E5 returns unless a deterministic quadrature over receivers is added (cost ∝ receivers × quadrature).
- **Energy contract.** There is no closed-form flux integral of a neural decoder, so passivity is checked numerically, not guaranteed. This is E9's failure mode carried over.
- **The nonlocal kernel form** must learn occlusion between x and x', i.e. dynamic visibility inside the learned part. That is a stop condition, with WL52 as prior evidence.
- **Verdict: rejected as primary.**
  - The nonlocal form violates the ownership separation.
  - The local form has no structural energy contract and keeps the delivery-variance problem.
  - Its useful idea, a continuous directional encoding (e.g. IDE-style), is kept as an *implementation option inside C's predictor*.

### Candidate B — sparse multi-resolution angular representation

- **Definition.** The outgoing state is a sparse set of flux coefficients on an adaptive angular hierarchy (e.g. an octahedral quadtree or HEALPix tree with **cos-weighted, flux-conserving** cells), refined where the lobe is narrow (e.g. 4×16 → 8×32 → 16×64 in grazing / peak regions). Λ_θ predicts coefficients up to a maximum depth.
- **Delivery.** Through a relation per cell, as in the current flight, with exact cos-weighted cell measure.
- **Energy contract.** Non-negative flux coefficients with an albedo-bounded allocation give passivity by construction.
- **Sharpness.** Bounded by maximum depth. Cells, relation rays and memory grow as α⁻².
- **Learned part.**
  - A variable-support sparse output is awkward for a shared network (prediction of support plus values).
  - The learning-free version of B, a flux-conserving table at 8×32 or adaptive, is exactly the control E4 calls for.
- **Verdict.** Not primary. It is kept in two roles:
  - (i) as the **explicit flux-corrected / high-resolution binned control**;
  - (ii) as the **independent alternative ECRO-FC** (below), the minimal-change lineage test.

### Candidate C — learned flux-allocated directional lobe state (primary: **FLS**)

**State.** Per element *e*, K anisotropic **spatio-angular lobes**. Each lobe k carries:
- a direction mean μ_k (local frame);
- concentration and anisotropy (κ_k, β_k, and an orientation);
- a spatial anchor a_k in element coordinates, with spread σ_k;
- RGB flux Φ_k.

Each lobe density p_k(u, ω) is **normalised to unit flux in the projected-solid-angle × area measure, over the clipped upper hemisphere**. The clipping matters: an unclipped normalisation reproduces the grazing over-emission of E4. Hence L_e(u, ω) = Σ_k Φ_k p_k(u, ω) / cos θ.

**Physical meaning of what is learned.** Λ_θ maps the *delivered incident field* of an element to its compact outgoing transport state. That incident field is a set of incident lobes or point-deposits, each carrying direction, flux and sub-element position, possibly partially visible. Λ_θ owns three things:
1. element-local scattering of spatially distributed, partially occluded incidence (ER is its physical model but is not replaced as the local evaluator);
2. sub-element spatial–angular structure (the emission anchor: X1c vs X1, 0.27 vs 0.11 in E5);
3. the **closure**, i.e. reduction of a many-component outgoing field to K lobes chosen to minimise *downstream delivery error*.

It is trained against path-traced delivered fields, not against a BRDF fit. **It owns no visibility and no inter-object geometry.**

**Why this is not "local BRDF compression".** ER is not re-learned as a function. It is used exactly for evaluation and for training targets. The learned map is from *distributed incident transport states* to *transport states*, and its loss is measured after explicit transport.

The critic disputes this. See §10, Q1, and HD-1.

**Energy / measure contract (structural).**
- For each incident component j (flux Φ_j, direction ω_j), the exact ER directional albedo a(ω_j) bounds the scattered flux.
- Λ_θ outputs allocation fractions π_{jk} ≥ 0 over K lobes plus a NULL sink, with Σ_k π_{jk} + π_{j,NULL} = 1 (softmax).
- Φ_k = Σ_j a(ω_j) Φ_j π_{jk}. Hence Σ_k Φ_k ≤ Σ_j a(ω_j) Φ_j: **passive by construction**, with exact flux bookkeeping.
- This is the WL52 π-softmax-with-NULL idea moved to the sender side.

**Homogeneity and superposition.**
- Φ is 1-homogeneous in the incident flux.
- The lobe *shapes* depend on the shape of the incident distribution, so the outgoing field is **not exactly additive** in independent sources. Physical light transport is linear. This is a real contract cost of C (ECRO was exactly linear).
- Mitigations:
  - (i) per-source states (cost ∝ lights);
  - (ii) declare a superposition-error contract metric with a human-set tolerance [HUM].

**Delivery (explicit, deterministic).**
- Each lobe is integrated with a fixed **lobe cubature** of Q nodes (u_i, ω_i, w_i) obtained by mapping a fixed spherical design through the lobe's parameters.
- Rays are traced in the current geometry. The flux Φ_k w_i is deposited at hits, recording position, direction and element.
- There are no random draws: no single-draw variance (addresses E5). The error is cubature error.
- That error is small for smooth lobes and **large when an occlusion boundary cuts the lobe footprint** (failure mode F2).

**Rigid edits.**
- The state is in local frames. A rigid transform of a moved object rotates its frames exactly, so Λ_θ outputs are equivariant by construction (contract test T-c).
- After an edit only C changes. Rays are re-traced for lobes whose cubature rays were dirty, or simply re-traced each update (§6).
- **No learned weights change.**

**Sharpness.**
- Continuous κ, with no bin floor.
- A narrow lobe is cheap to *represent*. Delivering it accurately past visibility edges is the hard part.

**Memory [PRED].** N 1,270 × K 8 × ≈ 12 floats ≈ 0.12 M floats. The current table holds 0.24 M (1,270 × 64 × 3).

**Runtime [PRED, order of magnitude].**
- Λ_θ: a set encoder over ≤ 32 incident components plus a small MLP head, ≈ 0.2–1 MFLOP per element. That is ≈ 0.3–1.3 GFLOP per sweep at N 1,270: sub-millisecond GPU work.
- Delivery: K·Q·N rays per sweep. At K 8, Q 13: 132 k rays per sweep, 0.53 M for 4 sweeps. That equals the C3URX1 per-light count of WL91 [MEAS: WL91, 0.53 M + 0.53 M]. Raw tracing costs about 1 ms at 0.5–3 G rays/s.
- In the current Python / Dr.Jit prototype, overhead dominates similar workloads (11–14 ms for 0.26–2.1 M rays [MEAS: WL91 scale]).

### Independent alternative — ECRO-FC (minimal-change lineage test)

- **What it keeps.** The ECRO v0 rule: geometry-conditioned generator, source encoding, single learned flight over state rays, global term. Ownership and visibility are unchanged.
- **What it changes, only:**
  - (i) the sender / receiver carrier becomes the flux-conserving cos-weighted table, adaptive 4×16 → 8×32 per element;
  - (ii) the training distribution is coverage-stratified by incident ring, with a declared grazing share [HUM];
  - (iii) a passivity constraint (albedo-bounded allocation) on the local factor pair.
- **Purpose.** It directly answers whether the ECRO-lineage failure was coverage (H1) and the carrier (H2), with everything else held fixed.
- **Its weakness:** it stays exactly linear but table-bound (α⁻² growth), and its global term inherits WL86A's compensation risk.

## 3. Ownership contract (FLS)

| transport part | owner | learned? | depends on geometry? | reused after a rigid edit? |
|---|---|---|---|---|
| local scattering evaluation (camera vertex, training targets) | frozen ER | no (frozen learned ER from WL84) | no | yes |
| incident → outgoing transport state of an element, incl. sub-element structure and closure | Λ_θ | **yes (shared rule)** | only through the delivered incident state and own frame | yes, weights unchanged |
| free-space delivery, visibility, contact, re-entry | C (lobe cubature on current geometry) | no | yes | recomputed (dirty or full) |
| multi-bounce composition | J sweeps of Λ_θ ∘ C; optional learned resolvent R_θ in lobe space (gated) | sweeps no; R_θ yes, if admitted | through C | yes |
| emission and direct light | explicit NEE / light lobes | no | yes | recomputed |

**Invariant (contract test T-v).** Λ_θ receives no geometry outside its element. Two elements with identical local inputs produce identical outputs in any scene. A violation is a STOP: implicit visibility.

## 4. Physical / mathematical contract (testable)

| id | contract | why |
|---|---|---|
| T-a | lobe normalisation over the clipped hemisphere: the flux integral equals Φ_k to ≤ 1e-4 for κ up to the sharpest ER lobe, including grazing means | E4's convention over-emission |
| T-b | furnace: with a(ω) = 1 and a closed white enclosure, energy is conserved; with a < 1 the total stays ≤ the input | E9 |
| T-c | rigid equivariance: a rotated / translated scene gives identical local states, with outputs transformed to 1e-6 | rigid reuse |
| T-v | locality: no access to non-local geometry (input audit plus scene-swap test) | ownership |
| T-s | superposition error ‖L(s1 + s2) − L(s1) − L(s2)‖ / ‖L(s1 + s2)‖ on controlled pairs, reported against a declared tolerance [HUM] | nonlinearity of the closure |
| T-f | known-field single flight: analytic incident lobe → plane / box receivers vs analytic projected-solid-angle integrals, open and with one occluder edge crossing the lobe | cubature error, F2 |
| T-r | exactness ladder: K → large and Q → large converge to exact-ER delivery (X1 pooled) | consistency with E5 |

**Synthetic tests vs real scenes.**
- Contract tests certify the mechanism's mathematics (measure, energy, equivariance, locality) independently of scene statistics. They catch compensating errors that a scene metric can hide; WL86A's global-term compensation is the example.
- Real-scene evaluation is still needed for what contracts cannot show: generalisation of a learned rule, fidelity of spatially distributed sharp multi-bounce, edit utility, and cost.
- Neither replaces the other.

## 5. Failure analysis (where FLS is expected to break)

- **F1. Near-mirror senders** (α ≲ 0.01) with spatially varying reflected detail finer than an element. The anchor and spread cannot carry an image; K would have to grow with image complexity. This is the J4 / E6 readout floor in another form.
- **F2. Occlusion edges inside a lobe footprint.** Cubature with Q nodes produces visible error steps. Adaptive Q costs rays.
- **F3. Curved glossy senders** (focusing / caustic-like). The lobe direction varies across the element faster than one anchor can model.
- **F4. Multi-modal incidence** from many sources or bright neighbours. The closure saturates at K; non-additivity grows (T-s).
- **F5. Near field / contact.** Receiver distance comparable to element size breaks a far-field lobe delivery unless sub-element anchors are resolved. This has a WL54 / WL55 precedent.
- **F6. Out-of-support configurations** for the shared rule. The H1 analogue: coverage must be stratified and verified.
- **F7. Exposure bias.** Trained on path-traced incident states, run on C-delivered states; the error compounds over sweeps.
- **F8. Cost.** If K·Q tracing per update is not cached, steady-state update cost is ~ C3URX1. The advantage would then be quality at equal cost, not speed. In this prototype the q table and frame dominate (E7).

## 6. Expected performance and uncertainty [PRED]

- **Carrier fidelity at grazing.** An optimal-fit lobe oracle (K ≤ 8) should remove the SA-mean bias, *as FC 8×32 also does* (E4). Whether lobes beat FC 8×32 at equal memory / time is **unknown**. The likely gain is at α below the 8×32 cell width and in memory.
- **Variance.** Deterministic delivery removes the single-draw variance that defeats X1 at the deployed budget (E5). A quality gain over X1 at equal rays is plausible; the size is unknown.
- **Learned vs explicit twin (ELT, moment-matched lobes from exact ER).** This is the decisive unknown. The prior expectation is a small gain, by the WL96 / WL96A pattern. A null result would make Λ_θ decorative.
- **Edit utility.** In GM1 (convex objects, cheap explicit sweeps, overhead-dominated prototype) a raw speed advantage is **not predicted**. A quality-at-equal-cost advantage over *binned* explicit transport in sharp / grazing regions is plausible. Advantage over FC 8×32 or ELT is uncertain.
- **Higher-order sharp transport** (glossy ↔ glossy) is where binned recurrence compounds bias and sampling variance multiplies. The GM1 family barely contains it (E10, one sharp panel), so it needs a workload extension (HD-2).

## 7. Falsification plan (follow-up experiment design; nothing authorised)

**Hypotheses.**
- **H1.** ECRO's failure is training coverage / optimisation.
- **H2.** It is the angular representation.
- **H3.** An improved learned representation still gives no geometry-edit utility.
- **H-L.** Learning adds value over its explicit twin.

**Arms.**

| arm | training | same data / budget / reference as FLS? | role |
|---|---|---|---|
| Frozen ECRO (WL86 / OWN checkpoints) | historical | no (historical data) | reference point |
| ECRO + coverage (same representation) | retrained | **yes** | H1 |
| ECRO-FC (flux-conserving adaptive table + coverage + passivity) | retrained | **yes** | H2 within the lineage; independent alternative |
| FLS (Λ_θ) | trained | — | candidate |
| ELT (explicit lobes: moment-matched from exact ER, same K, Q, C) | none | same reference | learning-free twin; decides H-L |
| explicit binned FC 4×16 / 8×32 / adaptive | none | same reference | carrier control (H2) |
| EXP / C3UR (+ X1 pooled and deployed budget; DQ where feasible) | none | same reference | explicit comparator, quality-matched |
| class floor: delivery-optimal patch × 64 operator vs the path-traced reference | fit, no learning | same reference | separates H1 from H2 for the ECRO lineage (WL97 P0-1b missing evidence) |

M64 is **never** used as ground truth. Truth is path-traced `field_src` references (event- and class-tagged) and camera references, with independent split halves.

**Workloads.** All are admitted by predeclared, geometry-only or SNR rules *before* any arm output exists.
- **WG (grazing sharp).** CK2 (diagnostic only, never an admission basis) plus ≥ 2 configurations from a seeded sampler with a declared grazing criterion, e.g. ≥ 50 % of S1 irradiance at cos_i ≤ 0.15 [HUM value].
- **WN (non-grazing sharp).** E2-type configurations from the same sampler.
- **WH (independent held-out).** Sealed configurations (sampler seeds disjoint and recorded), never inspected before the final evaluation, with E5-style qualification. Training exclusion is by geometry / light margins, applied to *every* trained arm (the WL97 B-02 lesson).
- **WGG (glossy ↔ glossy, optional).** Two facing glossy surfaces or a glossy box facing the panel; needs a scene-family extension [HD-2].
- **T0 → T1 rigid edit** on each, with path-traced references at **both** T0 and T1. T1 references must exist (WL97 F0-03 / J7).

**Stages and stop rules.**

0. **Freeze.**
   - Protocol, sampler seeds, admission rules, train / validation / held-out split with exclusion margins, reference budgets (split halves), metrics, statistics and stop rules.
   - A full-scale probe of every stage for wall time and GPU utilisation runs before the freeze (standing rule).
   - Dev runs use non-evidence inputs, so that dev figures cannot expose outcomes (the WL96A lesson).
1. **Learning-free gate (decisive and cheap).**
   - Contract tests T-a … T-r on the production code.
   - Then deterministic delivery error vs path-traced references on WG / WN for: SA-mean 4×16, FC 4×16, FC 8×32, adaptive FC, optimal-fit lobes K ∈ {1, 2, 4, 8} (per element, fitted to the exact outgoing field), ELT; plus the patch × 64 class floor. Cost (rays, ms) is recorded for each.
   - **STOP FLS** if optimal-fit lobes at K ≤ 8 do not beat FC 8×32 in delivery error at ≤ its cost on WG (sharp advantage absent), or if T-a / T-b / T-c / T-v fail.
   - **STOP the learned resolvent** if the explicit sweeps on the lobe state are already at reference noise (the WL96A pattern).
2. **Learned, static T0.**
   - ECRO + coverage, ECRO-FC and FLS, ≥ 3 seeds each, identical data and budget (parameter- or compute-matched as declared [HUM]).
   - Representation metrics (§8) and image metrics on WG / WN / WH at T0.
   - **H1:** ECRO + coverage vs frozen ECRO vs class floor.
   - **H2:** ECRO-FC vs ECRO + coverage; FLS vs ECRO-FC.
   - **H-L:** FLS vs ELT (paired).
   - **STOP FLS** if FLS ≤ ELT within the paired noise on all admitted workloads (learning decorative).
3. **Edit utility, fixed weights.**
   - T0 → T1 on WH (and WG / WN), with no retraining and no fine-tuning.
   - Changed-region fidelity, added T1 − T0 error, leakage / ghosting, vs explicit comparators at quality-matched budgets (equal error → compare cost; equal cost → compare error).
   - **H3** is supported if no learned arm shows a predeclared utility margin [HUM].
4. **System.**
   - Update critical path, steady-state frame, learned inference, visibility / relation recompute, memory and precompute.
   - Run alone on the GPU; wall time and GPU-kernel bases reported separately, with implementation overhead separated (WL97 D / E5 lessons).
   - Frame conventions are predeclared.
   - External baselines (Neural Full Recompute / Native Reuse) are included or explicitly marked UNMEASURED [HUM, per the baseline gate].

**Statistics** (WL97 lessons, binding).
- Paired arms with common random numbers.
- Block-bootstrap SE for image metrics. **Never** the an94.eh L1 noise floor as a difference tolerance.
- ≥ 3 training seeds plus state-draw resampling where any explicit arm has stochastic relations.
- Reference noise estimated from independent halves and kept separate from treatment differences.
- No decision rests on one seed or one draw.

## 8. Evidence to collect

- **Representation.**
  - angular response fidelity (cos-weighted L1 / L2 vs a dense exact directional reference);
  - peak / lobe position and width error;
  - spatially distributed transport error (sub-element);
  - energy and passivity (T-b, directional albedo of the learned map);
  - superposition error (T-s);
  - sampling variance (explicit arms).
- **Image.**
  - error vs high-quality reference;
  - sharp-interreflection ROI;
  - changed-region fidelity;
  - leakage / ghosting (signed error outside the changed support);
  - T1 − T0 added error.
- **System.**
  - update critical path; steady-state frame cost; learned inference cost;
  - visibility / relation recompute; memory and precompute;
  - utility vs quality-matched explicit comparators.
- **Statistical.**
  - paired differences with block-bootstrap SE; seed and state-draw spread;
  - reference noise vs treatment difference.

## 9. Human decisions required

- **HD-1.** Is a *shared element-local transport-state rule* composed with explicit coupling an admissible form of "reusable learned light transport"? The alternative reading is that it is the forbidden reduction to "local BRDF compression". This decides whether FLS is in scope at all.
- **HD-2.** Workload extension.
  - GM1 has no object-intrinsic nonlocal transport and almost no glossy ↔ glossy higher-order transport (E10).
  - Without an extension (concave glossy asset, facing glossy pair), the hypothesis can be tested only for first-flight sharp delivery and the broad tail. WL96A found no headroom in the broad tail on CK2.
- **HD-3.** Numerical criteria:
  - minimum meaningful image effect;
  - utility margin (the 1.25× of WL96 is a precedent, not a derivation);
  - Stage-1 gate tolerance;
  - superposition tolerance;
  - grazing coverage share;
  - parameter / compute matching;
  - number of seeds.
- **HD-4.** Gate A / Gate B carry-over from WL96: FLS's delivery is explicit lobe cubature. Is that admissible inside the method, or only as a comparator?
- **HD-5.** External baselines in Stage 4 (Neural Full Recompute / Native Reuse) or explicitly UNMEASURED.
- **HD-6.** Adopting or not the WL97 proposed annotations. They are not adopted, and this design does not depend on adopting them.
- **HD-7.** Compute budget and server (LabServer63 by the standing rule).

## 10. Independent critique and response

Planned single critique file was superseded by the user-authorised workflow (2026-11-16): independent designs D1_*.md, D2_*.md, D3_*.md; critiques X1_*.md … X4_*.md; judges JUDGE1_report.md, JUDGE2_report.md (all verbatim, this directory). The coordinator's synthesis and the disposition of this draft are recorded in the WL98 worklog. This draft is kept unchanged apart from this pointer.

## 11. Coordinator disposition after critique and judges (2026-11-16)

- **FLS is withdrawn as a learned candidate.** All three blind designers, all four critics and both judges rejected it:
  - at CK2 incidence the glint needs ≈ 10 separable lobes per element per light (X1);
  - in GM1 its learned content is ER compression plus mixture reduction, and delivery is explicit cubature at the same runtime as ELT;
  - it is nonlinear, so it gives up the source-only update lever;
  - it belongs to the VSL / VSGL class.
- The argument of §3 and HD-1, that the rule learns a distributed incident-to-transport-state map rather than a BRDF, is kept only as a recorded minority position.
- **ELT is kept as an explicit comparator**, upgraded with image-source shear.
- **ECRO-FC is not a candidate.** Its factors are conditioned on current visibility, world pose and a scene token. It survives only as factorised H1 / H2 diagnostic arms.
- §10's earlier pointer to a separate critique file was corrected above. The synthesis, final verdict, merged validation contract and consolidated human decisions are in `docs/worklogs/98_WORKLOG_LEARNED_DIRECTIONAL_TRANSPORT_DESIGN_REVIEW.md`.
- Sections §1–§9 are otherwise unchanged.
