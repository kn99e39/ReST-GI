# GM1 design — Track C — Alternative global transport operator

**Status:** design-only proposal for review (2026-11-10), revision `Selective_Recompute` @ `c7b0698`. Nothing was implemented, run or measured. Sealed track: of `docs/plans/gm1_architecture_review/` only `00_PROBLEM_BRIEF.md` was read.

**Candidate name:** LPTG — Lobe-Partitioned Transport Graph.

**Labels:** FACT = ESTABLISHED FACT; HIST = HISTORICAL EVIDENCE (original scope only); DERIV = DERIVATION; INF = INFERENCE; HYP = HYPOTHESIS; SPEC = SPECULATION. Every cost figure not cited to a worklog is an unmeasured estimate.

**Verdict in one line:** a conditional candidate. It is credible as a representation and update contract, has a real but narrow economic window, and is not novel as an algorithm class. It should pass a zero-render analytic admission gate (§12, Stage 0) before any implementation is considered.

## 0. Design space considered

| Option | Disposition | Reason |
|---|---|---|
| Geometry-conditioned learned transfer between surface elements | Rejected | HIST WL52: learned C was valid and stable but 8–19× slower per update than GPU explicit C (about 5 ms at 10k ports), dense O(N·M), and failed at contacts and visibility transitions. Explicit tracing is the cheap part. |
| Finite directional interface for all lobes (bins, SH, hash-grid radiance cache) | Rejected as the sole carrier | HIST WL58: a 64-in / 256-out bin interface loses 52–80 % relative L2 on GGX α 0.04 (proof-level, capacity-independent), and 10–18 % at α 0.25. HIST WL46: the directional interface is correct only with a passive sender and lobe-adaptive sampling. |
| Scene-level neural radiance cache with validity tracking | Rejected | Stores radiance, not an operator: after an edit the stored values are globally stale and recovery is retraining (Müller et al. 2021). No certificate of validity. Also conflicts with "learned response, not baked radiance" (FACT, `AGENTS.md` research invariants). |
| VPL / lightcuts / virtual ray lights with ER lobes | Rejected | Known clamping and spike failure on sharp glossy receivers (Walter et al. 2005; Hašan et al. 2009 address it by blurring). The directional accuracy GM1 asks for is where many-light methods are weakest. |
| Neural Radiosity residual solve restricted to invalid regions | Rejected | Update is SGD on a global network (Hadadan et al. 2021); shared parameters make "invalid region" ill-defined (HIST WL30: shared decoder owns transport). |
| Exact path-sample store with random-number replay | Kept as a **control**, not the mechanism | Exact (equals REF-ER at the stored spp) but view-dependent, memory ∝ pixels × spp × depth, and its saving is a bounded constant. It is the GM1 analogue of the WL56 exact selective oracle: an upper bound on avoided work. |
| **Operator factorisation: analytic visibility links + reused material angular part, with sharp lobes traced** | **Chosen** | Uses what the evidence says is cheap (explicit GPU links, ER eval), avoids what the evidence says fails (sharp lobes through a finite interface), and maps one-to-one onto the K/C boundary equation of `docs/Architecture.md` §3. |

## 1. Central mechanism

HYP. The frozen ER closure is an explicit sum of R = 8 lobes (FACT, WL82: Lambert + 7 anisotropic GGX, inner group attenuated by s(μ_i)s(μ_o), ER inner lobes at coat-refracted directions). Split the lobes of every material once, by a predeclared roughness threshold α\*, into a **rough set R** (Lambert and GGX with effective outside roughness ≥ α\*) and a **sharp set S**, so that f_ER = f_R + f_S exactly. Every scattering vertex of every light path is then an R-event or an S-event, which partitions path space exactly. LPTG persists a **view-independent sparse transport graph** whose unknowns are the band-limited *indirect incident radiance* at surface cache points on a fixed set of K material-frame directions; R-events are applied through a per-material K × K matrix **B** compiled from ER's rough lobes; S-events are never discretised and are carried as **stored, replayable ray chains** sampled from ER's sharp lobes, both inside graph links and at camera readout. Direct light is always evaluated exactly with next-event estimation through the full ER. After a rigid edit, stored ray segments are certified valid or invalid by geometric tests against the mover, invalid chains are regenerated from their first invalid segment with the stored random numbers, and the field is re-solved by warm-started sparse iteration.

Falsifiable claims:

- **H-rep.** There is an α\* and K for which the composed result meets a prospective directional / highlight gate against REF-ER, while the same system with all lobes forced through the field does not.
- **H-edit.** The incremental update S equals the same-architecture full rebuild F to numerical tolerance, with an invalid link fraction well below 1 for a small-footprint mover; the stale state fails on stationary receivers.
- **H-econ.** At a predeclared quality level, edit-to-valid-output GPU time of S is below that of path-traced re-rendering with frozen ER.

Each can fail independently, and §12 says what each failure means.

## 2. Persistent state and its information content

| State | Learned? | Spatial content | Angular content | Owner |
|---|---|---|---|---|
| ER closure parameters (69 floats per material) | Learned, frozen (WL83 compiler) | none (pointwise) | continuous | K |
| Lobe class mask (R / S) per material | Derived from ER by a fixed rule | none | split at α\* | K |
| **B** per material: B[m, j] = f_R(ω_j → ω_m) cos θ_j w_j | Compiled from ER by quadrature; source-independent | none | K × K nodes in the tangent frame; resolves lobes of width ≥ α\* | K |
| Cache points | No | material coordinates (triangle + barycentrics); base density plus a geometry-only proximity refinement | — | interface |
| **Links**: for cache point c and node j, the stored ray chain | No | sub-triangle hit positions in material coordinates, continuous | first direction = node j; later directions continuous (ER sharp-lobe samples, replayable from the link index) | C |
| Per-vertex direct term and its shadow segments | No | exact vertex | exact ER eval toward the previous vertex | C + e |
| Field I(c, j): indirect incident radiance | No (solved) | cache resolution | K nodes, band limit ≈ α\* | radiance state |

DERIV. With one piecewise-constant material per object, B costs 3K² floats per material (K = 128: about 0.2 MB). The global mechanism adds **no learned parameters**; its only learned content is ER and what is compiled from ER. This is stated as a weakness in §14.

## 3. Lifecycle and invalidation under a rigid edit

Let A be the mover (A0 at T0, A1 at T1) and Σ the static geometry.

**Never invalidated by a rigid edit:** ER parameters, lobe masks, B (FACT: K is fixed under supported edits, `docs/Architecture.md` §4), cache points in material coordinates away from changed proximity.

**Segment validity (DERIV).** A stored segment is valid at T1 iff re-tracing the same ray returns the same hit in material coordinates.

| Origin → stored hit | Valid at T1 iff | Discovery |
|---|---|---|
| Σ → Σ (or escape, or light sample) | the segment does not intersect A1 before its stored hit (A0 cannot matter: it was not hit at T0) | segment vs OBB(A1), then any-hit against A's own BVH for candidates |
| Σ → A0 | never | index lookup by hit object, no test |
| A → A (exposed self-interreflection, stored in A's frame) | the transformed segment does not intersect Σ | O(1) certificate if bounds(A1) overlap no static bounds; otherwise any-hit against the overlapping static objects only |
| A → Σ, or A → escape | never | index lookup by origin object |

The third row is the FNTO counterexample handled correctly: same-asset links live in C, are reused only under a certificate, and are invalidated by a third-party intrusion.

**Chain rule.** A chain is retained up to its first invalid segment and regenerated from there with the stored random numbers. Shadow segments are validated independently as leaves.

**Cache refinement.** The proximity rule is a deterministic function of current geometry, so points are added near A1 and may be removed near the vacated region; new points trace all K links.

**Field.** Always re-solved, warm-started from the T0 solution.

**F ≡ S (DERIV).** F at T1 with the same cache points, nodes and random numbers generates each chain as a deterministic function of (origin, direction, ξ, T1 geometry). A retained segment returns the stored hit by definition, the ER sample at that hit depends only on the local frame and ξ, and induction over the chain gives identical links. Equality is to floating-point tolerance, not bitwise, for segments re-expressed in a moved frame (compare HIST WL59: 1e-7).

**Limit (INF, from HIST WL70 dependency D6).** Validity is exact at the estimator's own sample resolution: an occluder that falls between all K nodes of a receiver changes nothing in the field, in S and in F alike.

**Discovery cost (DERIV).** Index lookups are output-sensitive. The Σ → Σ test is linear in stored segments if done by brute force, or sub-linear with a segment hierarchy (Bala et al. 1999). In a closed room long segments defeat hierarchical culling, so the honest worst case is linear. It is arithmetic on stored data, without scene-BVH traversal or shading.

**Invalid fraction (DERIV).** With N cache points, N_A of them on the mover, and ω̄ the mean fraction of a static receiver's nodes whose rays reach A0 or OBB(A1):

f ≈ N_A / N + (1 − N_A / N) · ω̄(A0 ∪ A1), plus the corresponding share of chain and shadow segments.

f → 1 when the mover is large, encloses the scene, or carries the light.

## 4. Newly created and destroyed paths

- **Destroyed:** links that hit A0 (regenerated, now reaching what was behind it) and links newly blocked by A1 (regenerated, now ending on A1).
- **Created:** the regenerated suffixes above; every outward link of A's own cache points; links of new refinement points near A1.
- **Newly exposed surface** (the old contact footprint): its cache points already exist; their links hit A0 and are regenerated.
- **Changed direct light:** shadow segments crossing A1 or formerly ending on A0 are re-tested; ER eval at the vertex is unchanged and reused.
- **Not handled by the field:** a path feature narrower than the node spacing or cache spacing (§10).

## 5. Directional information across repeated transport

DERIV. Label each vertex of a path R or S.

- **S → S → …** is carried at continuous directions with ER's own sharp-lobe sampler; it never passes through a finite basis, so sharpness is not lost by repetition. This is the bypass the WL58 candidate deliberately lacked (HIST WL58: "no continuous-direction/specular bypass").
- **R → R** passes through K nodes each bounce. The loss does not compound in sharpness because every R lobe is itself band-limited at ≥ α\*; whether K nodes resolve that band is the Stage 0 gate, not an assumption.
- **R → S → R** is one link containing a chain; the S vertex is exact.
- **Light → vertex** is exact ER evaluation at every vertex of every chain (collimated or small-source incidence never enters the finite interface, which was the WL58 probe's failing input).
- **S → R with a small source** (reflected caustic onto a rough receiver) is sampled only by NEE at the chain vertex and by K fixed receiver directions: this is the weak path class (§10).

## 6. Coupling and multi-bounce computation

Define Λ(x, ω) = direct(x, ω) + Φ_R(x, ω) + Φ_S(x, ω), the non-emitted radiance leaving x toward ω.

- Φ_R(x, ω) ≈ Σ_j B[m(ω), j] · I(c(x), j), interpolated over the nearest cache points and outgoing nodes.
- Φ_S(x, ω) is the stored one-sample estimate (f_S cos / p_S)(ξ) · Λ(x′, ω′), which is the next chain vertex.
- I(c, j) = Λ(x_cj, −ω_j), with x_cj the first hit of link (c, j).

This is i = e + C·b, b = K·i of `docs/Architecture.md` §3 with K := ER (B for R-events, the sharp-lobe sampler for S-events) and C := links. Ownership is disjoint by construction: segments belong to C, vertex events to K (DERIV).

One iteration is a batched K × K product per cache point followed by a sparse gather along links and chain vertices. Termination is by declared residual, not a fixed bounce count. The spectral radius of the realised operator must be estimated with a valid estimator (HIST WL54) because Monte Carlo chain weights f/p can exceed 1 per sample even though ER is passive.

**Readout (view-dependent, per frame).** Primary ray; exact direct light; Φ_R at the visible point from interpolated I with **exact** ER rough-lobe evaluation toward the camera (K evals per pixel, no outgoing basis); Φ_S by s fresh sharp-lobe chains per pixel that terminate into the field. Sharp reflections therefore have full image resolution.

## 7. Memory and computation scaling

Symbols: N cache points, K nodes, L̄ mean chain length (geometric in the sharp-lobe albedo), n_ℓ light samples per vertex, P pixels, s readout chains per pixel, n the path tracer's spp at matched error, d̄ its mean depth.

DERIV (counts, exact up to constants):

| Quantity | Count |
|---|---|
| Links | N·K |
| Stored segments | N·K·(1 + L̄)·(1 + n_ℓ) |
| Field | 3·N·K floats (incident) |
| Full build F | N·K·(1 + L̄)·(1 + n_ℓ) rays and as many ER evals / samples |
| Update S | discovery (≤ one pass over segments) + f × build |
| One solve iteration | N·K² multiply-adds per channel + O(N·K·(1 + L̄)) gathers |
| Iterations, warm start | ≈ log(ε / ‖Δ‖) / log ρ; grows without bound as ρ → 1 (high-reflectance conductors) |
| Readout | P·s·(1 + L̄)·(1 + n_ℓ) rays + P·K ER evals |
| Path-traced re-render | P·n·d̄·(1 + n_ℓ) rays and ER samples |

K scales as c / α\*² (DERIV: a lobe of width α\* subtends ≈ π α\*² sr of a 2π hemisphere), so lowering α\* to capture sharper lobes in the field costs memory ∝ 1/α\*² and solve time ∝ 1/α\*⁴. This is the hard wall behind WL58, and the reason sharp lobes are traced instead.

ESTIMATE (unmeasured, order of magnitude only): N = 2·10⁵, K = 128 gives 2.6·10⁷ links and about 1–3 GB with chains at roughly 32 bytes per segment; this fits 12 GB.

**Honest worst cases.**

1. f → 1: S costs F plus discovery.
2. Readout dominates: S and F are then indistinguishable in total cost, repeating HIST WL59 (common work dominated the selective saving).
3. Lobe census puts nearly all energy in S: L̄ grows, the field carries little, and the method is a path tracer with overhead.
4. ρ near 1: solve iterations dominate.
5. In the current Python-traced Dr.Jit harness each launch costs 0.4–1.2 s regardless of size (FACT, WL84), so wall-clock would be set by launch count, not by this accounting. Comparisons must use GPU kernel time and few large launches, as WL84 did.

No speedup figure is claimed.

## 8. Nonlocal visibility and stationary receivers

A stationary receiver's geometry and cache points are untouched, but its state changes through three separate channels, all without supervision:

1. its own links that A1 newly blocks or A0 no longer blocks (segment test);
2. links that stay valid but whose sender radiance changed (field re-solve; this propagates to any bounce depth);
3. direct light at its link vertices (shadow-segment test).

Third-party occlusion is detected because validity is tested on segments, not on endpoint proximity (FACT requirement, `docs/BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md`: normal similarity or local distance cannot guarantee coverage). Geometry-dirty and transport-dirty sets differ here by construction: the transport-dirty set is the union of receivers with an invalid segment and everything downstream in the solve.

## 9. Integration boundary with frozen ER

Unchanged: parameters, compiler, checkpoint, WL83 corrected numerics, WL84 provenance.

| ER query | Where | Purpose |
|---|---|---|
| lobe parameters (read-only) | once per material | R / S mask by the fixed α\* rule, using effective outside roughness for refracted inner lobes |
| `eval` restricted to R | once per material, K² pairs | compile B |
| `eval` restricted to R | readout, K per visible point | exact rough response toward the camera |
| `eval` (full) | every chain vertex, link hit and camera hit | direct light (NEE) |
| `sample` / `pdf` restricted to S | link hits, chain vertices, camera vertex | chain continuation and MIS against NEE |

The lobe-group restriction is a runtime entry point, not a model change. It needs three identity tests before use: f_R + f_S = f to float tolerance; χ² and pdf consistency of the S-restricted sampler (WL83 R2 style); passivity audit of B's column sums against the ER certificate. If the restriction cannot be added without altering ER numerics, this design is blocked.

Scope restriction: one material per object (B per material); spatially varying descriptors are out of GM1 scope for this design.

## 10. Principal failure conditions and most dangerous counterexamples

1. **Loose quality gate (most dangerous, economic).** HIST WL67 / WL69 / WL70: path-traced re-rendering met the contracts at 1–16 spp in seconds. If the GM1 gate sits where a path tracer needs few spp, no cached structure wins. LPTG can only win in the window between its own bias floor and the path tracer's noise at equal time; the window may be empty.
2. **Moving reflected caustic on a rough stationary receiver** (light → sharp sender → rough receiver, sender or blocker moved). It is a spatially sharp feature in the field, undersampled by cache spacing and K nodes. It is also exactly a GM1 target phenomenon.
3. **Degenerate lobe census.** If the WL84 coated-conductor family has little energy in lobes with α ≥ α\*, the field is nearly empty. The census of the frozen closures is unknown to me (not checked).
4. **Mid-roughness lobes (roughly α 0.05–0.15, SPEC on the range).** Too sharp to cache affordably, too broad to trace cheaply. Classified S they cost what a path tracer costs: no fidelity failure, no advantage.
5. **Near-contact structure.** HIST WL54 / WL55: 1 cm ports lost convex sub-port structure (errors up to 42 %); adaptive refinement removed most but not all, and unknowns grew 7.35× on a 4 mm gap.
6. **High-albedo recurrence.** ρ near 1 with Monte Carlo link weights; no clamping or damping is allowed, so a realised ρ ≥ 1 is a stop.
7. **Thin or small mover between nodes** (§3 limit): indirect shadowing missed by the field although the readout chains see the mover.
8. **Large, enclosing or light-carrying mover:** f → 1.
9. **Frozen-sample error is spatially correlated** (low-frequency blotches), which a pooled relative-L1 metric under-reports and a path tracer's white noise does not have.
10. **Non-rigid edits:** every A → A link is invalid; B survives, nothing else is claimed.

## 11. Testable advantage and why it could exist

Controls and what each isolates:

| Control | Isolates |
|---|---|
| Stale (T0 links and field, T1 primary visibility) | whether the workload needs an update at all; must fail on changed stationary receivers |
| Same-architecture full rebuild F | update correctness (F ≡ S) and the work the selective path avoids |
| REF-ER path-traced re-render, spp ladder | economics at matched quality, free of ER's local bias |
| REF-P | ER's local bias in this multi-bounce scene (not a property of LPTG) |

Where reuse saves work a path tracer must redo (DERIV for the structure, HYP for the magnitude):

1. **Broad-lobe Monte Carlo integration.** The path tracer re-estimates the rough-lobe multi-bounce integral per pixel, per frame, from nothing. LPTG keeps (1 − f) of those samples with a validity certificate.
2. **Recursion.** Deep high-albedo bounces become sparse arithmetic with a warm start; the path tracer pays rays and ER samples for each depth.
3. **View independence.** One field serves all pixels, views and frames; the path tracer's cost scales with P·n.
4. **Decoupling.** Field resolution follows transport smoothness (α\*, cache spacing), not image resolution.

What reuse does not save: sharp-lobe chains at readout cost the same as in a path tracer, and the selective part can be immaterial next to readout. An exact identity here (the path partition, F ≡ S) is not a cost advantage; only the measured time-to-quality curves decide H-econ.

External comparison note (FACT, baseline-role contract): beating REF-ER re-rendering is the path-tracing cost axis only. It says nothing about Neural Full Recompute, which stays UNMEASURED.

## 12. One bounded decisive falsification experiment

**Scene.** One "glossy court": floor and two walls, one stationary curved receiver panel, one rigid box (the mover), one small rectangular area light, closed enough for several bounces. All surfaces use frozen ER materials taken unchanged from the WL84 set (at least one iid material and the hard material `test83_hard_1`). Two fixed cameras; one frames the panel highlight.

**Edit.** One rigid translation + rotation of the box, chosen from geometry alone so that it (a) newly blocks a wall → panel interreflection, (b) unblocks another, (c) brings the box's own reflection into the panel, (d) creates a near-contact with the floor. Plus one null edit and one off-path move as invalidation controls (WL70 style).

**References.** REF-P (layered) and REF-ER at T0 and T1, independent seeds, depth by Russian roulette rather than WL84's static depth 4, uncertainty qualified as in WL84. The REF-P vs REF-ER gap is reported first as ER's local bias.

**Stages and stop rules (all thresholds frozen before the stage runs).**

- **Stage 0 — analytic admission, no rendering.** (i) Lobe census of the frozen closures: share of albedo in R for the predeclared α\*. (ii) B quadrature error for band-limited incident fields at the declared K, including grazing. (iii) The three ER identity tests of §9. Stop here if the R share is negligible or B misses its gate.
- **Stage 1 — references and REF-ER spp ladder** with GPU-kernel time-to-quality at T0 and T1. The quality levels used later are fixed from this ladder before any LPTG image exists.
- **Stage 2 — static T0 fidelity.** LPTG-F against REF-ER, plus two ablations: all lobes through the field (expected to fail highlights; if it passes, the chain machinery is unnecessary) and a final-gather readout (separates cache-spacing error from the rest).
- **Stage 3 — edit.** Stale, F and S at T1.
- **Stage 4 — cost.** Uncontended, split into discovery, retrace, solve, readout, memory; optional exact path-replay control as the avoided-work upper bound.

**Measurements.** Error vs REF-ER by region (highlight ROI, changed stationary ROI, unaffected ROI, mover); signed T1 − T0 change slope and correlation on stationary receivers; F vs S difference; invalid fraction f; realised ρ and iterations; all costs.

**What falsifies what.**

| Result | Meaning |
|---|---|
| Stage 0 fails | Rejected before implementation for this material family |
| F misses the directional / highlight gate while REF-ER at equal GPU time meets it | H-rep falsified: representation failure |
| The all-field ablation passes | Mechanism not needed; a plain directional cache suffices for this scope |
| S ≠ F beyond tolerance, or an invalidation control is accepted when it should be rejected | H-edit falsified: update contract broken |
| Stale passes the changed stationary ROI | Experiment invalid (edit not discriminating), not a result |
| At every predeclared quality level S is not cheaper than REF-ER | H-econ falsified |
| S cheaper than REF-ER but not cheaper than F | The advantage is the cache, not the reuse: the selective-update claim must be dropped |

## 13. Closest prior art and honest difference

Verified by search this session: Bala, Dorsey, Teller 1999 (Eurographics Workshop on Rendering); Dmitriev, Brabec, Myszkowski, Seidel 2002 (Eurographics Workshop on Rendering); Sillion, Arvo, Westin, Greenberg 1991 (SIGGRAPH); Deng, Hašan, Carr, Xu, Marschner 2021 (ACM TOG, SIGGRAPH Asia). The others are cited from memory and were not re-verified.

| Work | Overlap | Difference |
|---|---|---|
| Wallace, Cohen, Greenberg 1987, "A two-pass solution to the rendering equation" (SIGGRAPH); Sillion et al. 1991, "A global illumination solution for general reflectance distributions" (SIGGRAPH) | Finite-basis solve for diffuse / directional-diffuse transport with specular transport traced separately. **This is the same decomposition.** | Static scenes; the split is by ideal BRDF class, not by the lobes of a learned compiled closure. |
| Křivánek et al. 2005, "Radiance caching" (IEEE TVCG); Gautron et al. 2007, "Temporal radiance caching" (IEEE TVCG) | Directional cache for low-frequency glossy; temporal reuse | They cache radiance, not operator links; validity is heuristic (gradients / ageing), not a segment certificate. |
| Drettakis and Sillion 1997, line-space hierarchy (SIGGRAPH); Chen 1990, incremental radiosity (SIGGRAPH) | Link invalidation for moving objects; incremental re-solve | Diffuse. The topic document already ranks the 1997 work as prior art, not a starting point. |
| Bala et al. 1999, ray segment trees | Finding stored ray segments affected by an object edit; reports 3–4× over its base ray tracer | Radiance interpolants for a Whitted-style ray tracer, not a multi-bounce glossy operator. |
| Dmitriev et al. 2002, selective photon tracing | Selective re-tracing of the sample subset affected by a dynamic object | Light-side photons with density estimation; no directional operator. My replay rule is in this family. |
| Müller et al. 2021, real-time neural radiance caching (ACM TOG) | Trace sharp bounces, terminate into a cache once the path is spread out; handles dynamic scenes by online training | Learned radiance, no validity certificate, convergence lag. In practice it is the strongest competitor, and it is not among the designated controls: UNMEASURED. |
| Deng et al. 2021, path graphs | Fixed set of traced paths reused through iterative propagation | Static, per-frame filtering; no edits. |
| Surface-cache-plus-traced-reflection engines (e.g. Lumen, SIGGRAPH 2022 course) | Same split in production form | Diffuse-oriented cache, heuristic invalidation. |

Honest position: every ingredient is known, and the combination is a modernised two-pass directional radiosity with certified link reuse. I do not infer novelty. What would be specific to ReST-GI is (a) the partition taken from the frozen learned closure's lobe structure with an exact sum, (b) F ≡ S on a glossy operator with third-party and same-asset validity, (c) quality-matched lifecycle evidence. None of these is a new algorithm.

## 14. Self-assessed weaknesses

1. **Nothing global is learned.** The mechanism is learning-free above ER. HIST WL80B judged a learning-free operator "PHYSICAL BASELINE ONLY" and inadmissible as a learned K under FNTO v0.1, and FNTO v0.1 §1 warns against replacing learned shading with "a limited explicit radiosity solver". Here K is the learned ER and C is explicit (which HIST WL52 favours on cost), but whether an explicit, unlearned C satisfies the research contract is a **human decision and a possible contract conflict**, reported rather than assumed away.
2. **The economic window may not exist** (§10.1), and the selective part may be immaterial next to readout (§7.2). The most likely outcome on current evidence (INF from WL59, WL67–70) is "correct update, no decisive cost advantage".
3. **The lobe census is unknown.** The whole design rests on a meaningful rough share in the WL84 family.
4. **GM1's hardest phenomenon is my weakest path class** (§10.2). The design gives exact sharp reflections seen by the camera, but not sharp indirect features on rough receivers.
5. **Bias is structural** (node spacing, cache spacing, frozen samples), so beyond some spp the path tracer always wins, as the layered reference does against ER beyond about 1024 spp (FACT, WL84).
6. **α\* and K are free design parameters.** They must be fixed before Stage 0 from lobe geometry alone, with no sweep; one failed setting ends the candidate.
7. **Rigid only.** Deformation invalidates all same-object links; nothing in this design helps GM3.
8. **Restricted materials:** one material per object.
9. **ER's own bias (0.035–0.04 image L1, static, depth 4) is unqualified for deep multi-bounce.** The REF-P vs REF-ER gap may be the dominant error in the experiment and is outside this mechanism's control.

**Recommendation of this track:** do not adopt. Treat LPTG as a narrowed candidate whose admission depends on Stage 0, and treat the exact path-replay store as a useful control for whichever mechanism is chosen.
