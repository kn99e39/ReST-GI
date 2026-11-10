# GM1 review — Physical transport correctness

**Status:** design-only adversarial review (2026-11-10), revision `Selective_Recompute` @ `c7b0698`. Nothing was implemented, run or measured. This file is the only file written. It authorises nothing.

**Reviewed (in full):** `10_design_trackA_directional_delta_transport.md` (DDT / DDT-D), `11_design_trackB_path_space_reuse.md` (ERPS: PR1 / PR2 / PR3), `12_design_trackC_alternative_operator.md` (LPTG). No other review file in this folder was read.

**Originals read for this review:** `00_PROBLEM_BRIEF.md`; `docs/Architecture.md` (whole); `docs/RESEARCH_ROADMAP.md` (GM1 row, current-direction and next-decision paragraphs); `AGENTS.md`; targeted passages of worklogs 46, 58, 70, 81, 82, 83, 84 (the specific numbers the designs rely on were checked against the worklog text). Worklogs 51–56, 59, 67, 69 are used only through `AGENTS.md` / `docs/Architecture.md` summaries and are cited as such. `docs/FNTO_ARCHITECTURE_CONTRACT_V0_1.md` was not re-read; the K/C contract is taken from `docs/Architecture.md` §2–§6.

**Literature:** no web search was run in this session. Every literature reference below is from memory (authors / year / venue believed correct, details unverified) and is used only for standard results, never for a number.

**Labels:** FACT = ESTABLISHED FACT, HIST = HISTORICAL EVIDENCE (original scope), DERIV = DERIVATION, INF = INFERENCE, HYP = HYPOTHESIS, SPEC = SPECULATION.
**Severity:** FATAL = the mechanism as written gives a physically wrong result on a GM1 target case and no in-scope repair is stated; MAJOR = wrong or undefined on a GM1 target case, repair exists but changes cost or the claim; MINOR = under-specification or a bounded error.

**Axis:** physical transport correctness only. Novelty, admissibility as a "learned" mechanism and economics are out of scope except where a cost claim rests on a transport statement.

## 0. Summary of verdicts

| Design | Verdict on this axis | Binding defects |
|---|---|---|
| A — DDT identity | SOUND (exact identity) | none; it is not an algorithm |
| A — DDT-D (narrowed mechanism) | **CONDITIONALLY SOUND, under-specified**; unsound as written on sharp chains | A-1 tolerance norm, A-2 discovery coverage (S ≠ F), A-3 no defined propagation through sub-resolution lobes, A-4 stochastic residuals |
| B — PR1 / PR2 | **SOUND** as estimators of REF-ER | B-1 destroyed-highlight ghost in PR2 (variance, not bias); depth-truncation of the reference |
| B — PR3 | CONDITIONALLY SOUND | B-3 T0-domain shift for vertices on the mover is unspecified |
| C — LPTG | **CONDITIONALLY SOUND as a consistent estimator; biased by construction** at finite K | C-1 point-sampled incident field is not band-limited, C-2 frozen-sample fixed point is biased and can be non-contractive, C-3 discretised B does not inherit the ER certificate, C-4 flux-conservation contract of `docs/Architecture.md` §3 not met |

No design contains a K/C double count or an ownership gap at the level of path segments. No FATAL defect was found in B. A and C each have one defect that is FATAL **for a specific GM1 target path class** (A-3, C-1 combined with C-2 on receiver-side caustic-type chains) unless the stated repair is adopted.

## 1. Common ground used below

**Operator form (FACT, `docs/Architecture.md` §3).** `i = e + C(G) b`, `b = K i`. With frozen ER, `(K i)(x, ω_o) = ∫ f(x; ω_i, ω_o) i(x, ω_i) cos θ_i dω_i` and `(C b)(x, ω_i) = b(y(x, ω_i), −ω_i)` with `y` the first hit (radiance is invariant along a free-space ray). `e = C L_e` is direct incident radiance. `b` is non-emitted outgoing radiance.

**Two norms and what ER's certificate gives (DERIV; standard, cf. Arvo 1995, EGWR, and Veach 1997, PhD thesis — from memory).**

- Flux norm `‖b‖_Φ = ∫∫ |b| cos θ dω dA`. `C` preserves the measure `cos θ dω dA` along rays, so `‖C b‖_Φ ≤ ‖b‖_Φ` (equality in a closed scene). `‖K i‖_Φ ≤ sup_ω a(ω) · ‖i‖_Φ`, `a` = directional albedo. Hence `‖K C‖_Φ ≤ sup a ≤ 1` under the ER certificate (FACT, WL82: every component albedo ≤ 1, softmax Σλ ≤ 1).
- Radiance sup norm. `‖K C‖_∞ = sup_{x, ω_o} ∫ f cos θ_i dω_i`, which equals the albedo at `ω_o` **only because ER is reciprocal** (FACT, WL82: exact by construction). So `‖K C‖_∞ ≤ 1` as well.
- Consequence: `‖K C‖ ≤ 1` in both norms, **not** `< 1`. A strict contraction needs `sup a < 1` uniformly, or escape with a uniform lower bound, or a bound on `‖(K C)^n‖` for some `n`. "Any absorption exists" (Track A §6) is not sufficient for a rate; it gives convergence of the Neumann series for non-degenerate scenes but an arbitrarily slow one. The WL83 hard cluster is defined by high reflectance (FACT, WL83 population table), so the relevant `ρ` is near 1 by design of the test set (INF).
- The two norms are not interchangeable for sharp transport: a perturbation of flux `ε` concentrated in a lobe of solid angle `Ω` over area `A` has radiance `≈ ε / (A Ω cos θ)`. Flux-small is not radiance-small.

**Reference depth (FACT, WL84: integrator `path`, max depth 4).** REF-ER as used in WL84 is a truncated Neumann series. Any field solve to a residual tolerance (A, C) computes a different quantity. With albedo 0.9 the mass beyond four scattering events is of order `0.9⁴ ≈ 0.66` of the fourth-order term per further bounce, not negligible (DERIV, illustrative arithmetic only). A reference with a fixed depth cap makes a correct converged solve look biased.

**Counterexample fixtures used in §2–§4.**

- **CE-1 (occluder between stationary glossy surfaces).** Stationary glossy plates G1, G2 facing each other, a small light on G1's side so that G2 shows G1's highlight by one interreflection. A rigid mover O slides into the G1–G2 gap. Endpoints of every G1↔G2 ray are unchanged.
- **CE-2 (new specular chain).** Light → mirror-like mover M → stationary glossy G → stationary rough receiver W → camera. At T0 the chain does not exist; at T1 a rotation of M creates it. W never sees M.
- **CE-3 (self-shadow-only edit).** A concave rigid mover is rotated in place under a fixed light; no stationary surface is within reach of its reflected light at either pose (dark surroundings), and it occludes no stationary-to-stationary ray differently (convex hull unchanged in the relevant directions).
- **CE-4 (destroyed highlight).** A pixel / cell that holds a sharp bright interreflection highlight at T0 and is fully blocked at T1.
- **CE-5 (thin mover).** A rod whose angular width seen from a receiver is below the receiver's angular sampling or state resolution.
- **CE-6 (small bright indirect source).** A narrow directly lit strip on a stationary surface (HIST analogue: WL54's 2.3 mm sunlit strip aliased by a point-sampled reference) seen by a rough receiver.

## 2. Track A — Directional Delta Transport

### 2.1 Verified sound

1. **Identity (DERIV, checked).** From `b0 = K(e0 + C0 b0)`, `b1 = K(e1 + C1 b1)`: `db = K(de + C1 db + dC b0)`, so `(I − K C1) db = K(de + dC b0)`. Correct, provided `b` is a field on the material manifold `(x_mat, ω_mat)` and `C0`, `C1` both act on that manifold. The design does this (§2, mover lookups in `R⁻¹(−ω_i)`).
2. **`dC b0` on a ray (DERIV, checked).** `b0(y1, −ω_i) − b0(y0, −ω_i)` is the correct radiance form; no cosine or `1/r²` factor belongs there because the unknown is radiance, not flux.
3. **First-order support.** `dC` is non-zero exactly on rays whose first hit differs, which requires crossing `O0 ∪ O1`. CE-1 is inside this set although endpoints are unchanged. Correct.
4. **Feed-forward error statement (§5).** `b1_est − b1 = δ + (I − K C1)⁻¹ K dC δ` for the additive form. Checked. A state error cannot enter the operator.
5. **Additive deltas rejected for values.** The ghost / negative-radiance argument is correct (it is CE-4 in state form) and the overwrite decision removes it.
6. **K/C ownership.** K is pointwise ER; every segment is C. No double count. Direct light is `e`, emission never enters `b`; a revealed emitter is handled by `de`, not by `dC b0`. Consistent.
7. **Non-rigid extension term** `dK i0`. Correct.

### 2.2 Defects

**A-1 (MAJOR). The stopping tolerance has no declared norm, and the natural one is wrong for sharp lobes.**
Step 3 propagates "source magnitude … until it falls under a predeclared radiometric tolerance"; φ_τ and the cost claim are defined through it. DERIV (§1): a flux-norm or cell-averaged tolerance does not bound radiance. Counterexample CE-2: the new highlight on W, and before it the mirror image of the light on G, carry small flux (small light, small M) in a thin `(x, ω_o)` set but order-one radiance for the pixels that see them. Any flux-thresholded or cell-averaged dirty mark drops exactly the GM1 headline feature. Second amplification (DERIV): cells left clean carry a dropped `db_clean`; the dirty-subset fixed point then has error up to `‖db_clean‖ / (1 − ρ)`, so the tolerance must be divided by `(1 − ρ)`, which is large on the high-reflectance cluster.
*Required change:* declare the tolerance in the sup-radiance norm at the consumer's angular resolution (per cell peak, not per cell mean), and state the `1/(1 − ρ)` amplification in the bias budget. This raises φ_τ; the design's "ratio bounded by 1/φ_τ" then has to be re-argued.

**A-2 (MAJOR). Discovery from the mover's surface does not certify coverage of receiver cells; S ≠ F, and the "independent of scene size" cost statement is false.**
The design samples points and directions on `O0`, `O1` and traces outward (§3). The objects to be marked are receiver state cells `(x-cell, ω-cell)`. DERIV: every receiver spatial cell that sees `O0 ∪ O1` must receive at least one discovery ray, so the ray count is bounded below by the number of such cells, which grows with the visible scene area at state resolution — not with mover area alone. With fewer rays, far or grazing receivers are missed; the missed cell keeps `b0` while a full rebuild's receiver-side gather would see the mover. Counterexample CE-5 and its mirror image: a distant glossy wall cell that sees a small mover inside one lobe. Unlike WL70's D6 (HIST: "exact at the estimator's own sample resolution" because discovery replays the estimator's own samples), here discovery samples and gather samples are different sets, so the WL70 argument does not transfer and same-architecture F/S equivalence (`docs/Architecture.md` §6) is not guaranteed.
*Required change:* either conservative geometric discovery (shaft / bounds of `O0 ∪ O1` against every receiver cell's direction set — scene-size dependent), or persist each cell's gather rays and test those (which converts the state into Track C's stored links).

**A-3 (FATAL for chains through sub-resolution lobes unless specified; MAJOR otherwise). No transport of dirtiness is defined where the state is band-limited.**
§2 allows a band-limited `b0` with "path continuation through lobes narrower than the stored resolution". That rule is defined only for **value lookups in T0 geometry**. Dirty-mark propagation (§1 step 3, §8) is defined only on stored cells. For a surface whose lobe is below the stored resolution there are no cells, so there is nothing to mark and nothing that forwards a mark. Counterexample (variant of CE-2 with a stationary mirror): rough stationary W gathers through stationary mirror S toward the region the mover vacates. `D` contains rays S → `O0`; S has no stored cells at that bandwidth; W's cells are never marked; W's stored `b0` stays stale although its gather through S now reaches different radiance. The same hole exists at second order for every sharp stationary surface that sees a first-order receiver.
*Required change:* discovery chains must continue from the mover through every sub-resolution lobe (reverse ER sampling, valid by reciprocity) until they land in a stored cell, and the same for higher orders. That is forward tracing of a signed quantity with a connection / density-estimation step at the landing cell (the design's own §4 last bullet, but there limited to light caustics). The discovery cost then contains a specular-chain tracer; "discovery is bookkeeping" no longer holds.

**A-4 (MAJOR). Residual-driven ordering with stochastic gathers confuses estimator noise with physical change.**
§6 orders work by descending magnitude (Southwell). After the first regather, the only available magnitude for a downstream cell is `|regathered − stored|` at its senders, which contains gather noise and the static solver error of `b0` (the design concedes that `b0` must be "treated as an exact T0 fixed point"). DERIV: with gather variance `σ²/g` the residual floor is `σ/√g` per cell; any threshold below it marks everything reachable (φ → 1), any threshold above it drops real changes of that size — and for small motions the design itself notes that the net signal can fall below state error. Negative-delta cancellation adds to this: for a slightly moved sharp reflection the **signed** cell-mean of `db` is near zero while the pointwise change is order one, so signed or cell-averaged propagation under-marks.
*Required change:* propagate a deterministic **upper bound** of `|s|` (unsigned, peak-per-cell) through `K C1`, never differences of estimators; or regather with common random numbers so that unchanged gathers reproduce exactly. The first over-invalidates; the second again is the stored-link design.

**A-5 (MINOR). Mixed-state consistency.** Overwriting marked cells while unmarked cells keep `b0` yields a state that is not the fixed point of any single operator; adjacent clean / regathered cells differ by the static error and gather noise of `b0`. This is a visible seam on smooth receivers and a non-physical signed change at the dirty boundary. It is bounded by the static error (HIST analogue: per-unit error fields exposed by hard selection, WL34 per `AGENTS.md`).

**A-6 (MINOR). `e` invalidation list is incomplete.** §3 lists only shadow rays crossing `O0 ∪ O1`. On the mover `de` is dense for any rotation, and for any translation under a finite-distance light (direction, cosine and distance change in the material frame). The `b` bullet of the same section says so; the two bullets should agree. CE-3 is covered once this is fixed: the whole mover is regathered, and with dark surroundings nothing else is marked. Correct behaviour, cost = one dense mover regather.

**A-7 (MINOR). Convergence wording.** "Contraction whenever any absorption or escape exists" should read `‖K C1‖ ≤ 1` in both norms with a scene-dependent rate (§1). No stability risk follows — the solve cannot diverge with certified ER — only an unbounded sweep count as `ρ → 1`.

**A-8 (MAJOR for the experiment, not the mechanism). Arm A3 does not test DDT-D.** A3 is "DDT-D with oracle `b0` … no stored field". DDT-D's output is an overwritten state; with no state, A3 can only output `image_T0 + Δ`, the additive form §5 rejects, i.e. residual path tracing. Falsifier F1 ("the identity has no economic content even with a perfect state") would then be drawn from a different estimator with different variance behaviour (CE-4 applies to it). Also no depth convention is stated for A1/A2 versus the converged state arms (§1, reference depth).

### 2.3 Threat patterns

| Threat | Finding |
|---|---|
| New glossy path absent from cached support | A-3; and the design's own never-queried-direction precondition (§2 of the design) |
| Changed occlusion, unchanged endpoints | Covered at first order (CE-1); coverage not certified (A-2) |
| Narrow lobes lost by angular compression | A-1, A-3 |
| Broad multi-bounce propagation | Correctly stated as dense; tolerance amplification `1/(1 − ρ)` missing |
| Negative-delta cancellation | A-4 (in discovery), removed for values by overwrite |
| Estimator noise mistaken for change | A-4 |
| Hidden full path tracing | Present by the design's own admission when `b0` is on demand; also inside discovery after the A-3 repair |

### 2.4 Verdict (Track A)

The identity is **SOUND**. DDT-D is **CONDITIONALLY SOUND and under-specified**: correct on rough-to-rough first-order changes, undefined on changes carried through lobes sharper than the state (A-3), and without an F ≡ S guarantee (A-2). Each physically adequate repair (receiver-side stored gather rays, common random numbers, chain continuation in discovery) moves it toward Track C's stored-link structure plus a forward specular tracer.

## 3. Track B — Path-space / reservoir reuse

### 3.1 Verified sound

1. **PR1 validity rule (DERIV, checked case by case).** A stored segment that ended on stationary geometry cannot have been blocked by `O0`, so only `O1` can change it; a segment that ended on `O0` is dirty; an escaped ray must be tested as an infinite ray against `O1`; a shadow segment must be tested against **both** placements because an any-hit shadow query does not record which occluder stopped it (the design tests both). A shadow segment occluded by stationary geometry stays occluded. All cases are covered. CE-1 is detected by the segment test; CE-3 is detected because every path vertex on the mover is dirty (light direction changes in the material frame).
2. **Suffix replay equals a same-seed T1 render (DERIV)** if random-number consumption is a function of vertex index only. Then PR1 is an unbiased T1 estimator with the T1 sampler's own variance.
3. **PR2 unbiasedness (DERIV).** In primary-sample space `E_u[f1(u)] = I1` and `E_u[f0(u)] = I0`, each `f_t` carrying its own pdf, so `I0 + mean(f1 − f0)` is unbiased when the `n` paths are an unselected subset.
4. **No operator, hence no composition instability.** One path is one Neumann term; every weight is an ER `f cos / p` product. Passivity questions of A and C do not arise.
5. **Ownership and direct / indirect split.** Segments are C, vertices are K, direct light is next-event estimation with MIS at each vertex; identical to REF-ER. No double count is possible that REF-ER does not have.
6. **Honest statement that births are sampled at the fresh sampler's rate** (CE-2 costs exactly what it costs REF-ER).

### 3.2 Defects

**B-1 (MAJOR for PR2 as a comparator; not a bias). Destroyed-highlight ghost with negative radiance.**
CE-4: `f1 ≡ 0` on every path of the pixel. PR2 returns `I0(N) − mean_n(f0)`, a zero-mean residual with standard deviation `σ0 · sqrt(1/n − 1/N)` and either sign, where a cold T1 render returns exactly 0 with zero variance. For a sharp interreflection highlight `σ0` is large and heavy-tailed, so the removed highlight leaves a signed ghost at its old location (DERIV). The design's general remark on decorrelated differences and the control-variate weight (§7) covers this in principle; it should be named as a required gate, because a difference image with ghost residue is precisely the failure the review is asked to detect ("negative-delta cancellation"). A per-pixel weight estimated from the same samples is biased at small `n` (INF; known for image-space control variates, Rousselle et al. 2016, from memory).
*Required change:* PR2's per-pixel fallback must be decided from geometry (pixel has only dirty paths → fresh), not from the sampled covariance; report ghost / negative mass per mask.

**B-2 (MINOR). Selection bias in the stored subset.** If the `n ≤ N` stored paths are chosen by any contribution-dependent rule (reservoir, importance, "keep the bright ones"), `mean(f1 − f0)` is biased. The design does not say how the subset is chosen.

**B-3 (MAJOR for PR3 only). The T0-domain evaluation for paths with a vertex on the mover is not a visibility test.**
The design reduces the T0 target evaluation to "visibility against the T0 placement … one extra instance transform". For a T1 candidate with a vertex on the mover, the T0 counterpart has a different world position, different segment directions at both neighbours, a different geometry term and different ER arguments; the shift (world-anchored versus material-anchored vertex) and its Jacobian must be defined, and a shift that is undefined (the material point is occluded or back-facing at T0) must be rejected symmetrically in both directions. If the reservoir instead keeps the stored world-space reconnection vertex, that point lies in empty space at T1. An asymmetric treatment biases exactly the newly lit / newly blocked regions (INF from the GRIS requirement that MIS weights use each domain's own target; Lin et al. 2022, from memory, not re-read).
*Required change:* specify the temporal shift for mover vertices and its Jacobian; or drop PR3 from the unbiased arms, as the design half-suggests.

**B-4 (MINOR). Reference depth.** §6 fixes the arms to REF-ER's depth and cites WL84's max depth 4 (FACT). That makes B internally consistent but defines "multi-bounce" as four events. Comparing a depth-4 B arm with a converged A / C field would attribute truncated energy to the field method (§1).

**B-5 (MINOR). "Bit-identical" needs a condition.** Clean-path identity requires unchanged static acceleration structures (instance transform change only). A rebuilt top-level structure can alter tie-breaking at coincident or near-coincident surfaces (INF; HIST relevance: near-coincident hits were a measured error source in WL44 / WL45 per `AGENTS.md`). The Stage 1 check "clean paths are bit-identical" is the right test.

**B-6 (MINOR, scope of an inference).** F2 states that the measured `ρ` bounds "any selective scheme at path granularity". That is correct only with the qualifier. `ρ` is a property of one camera's sampled paths; a scheme reusing at segment or link granularity keeps the clean part of a path that is dirty at depth 1 only through state at the *new* hit point, which path replay does not have. `ρ` is therefore neither an upper nor a lower bound for A or C (DERIV).

### 3.3 Threat patterns

| Threat | Finding |
|---|---|
| New glossy path absent from support | Not represented; sampled fresh. Correct, no reuse (stated) |
| Changed occlusion, unchanged endpoints | Exact (segment test) |
| Narrow lobes / angular compression | None; exact directions |
| Negative-delta cancellation | B-1 |
| Estimator noise mistaken for change | PR1 none (clean paths identical); PR2 in dirty pixels (B-1) |
| Hidden full path tracing | It is path tracing, declared |
| Speedup from lower fidelity | Only if partial validation or the biased PR3 variant is used; both labelled by the design |

### 3.4 Verdict (Track B)

PR1 and PR2 are **SOUND** estimators of REF-ER at the reference's own depth; PR3 is **CONDITIONALLY SOUND** pending B-3. The track contains no transport representation, so its correctness says nothing about a representation's correctness.

## 4. Track C — Lobe-Partitioned Transport Graph

### 4.1 Verified sound

1. **Lobe partition (DERIV, checked against WL82).** ER is a non-negative sum of lobes; the inner group is multiplied by `s(μ_i) s(μ_o)` with `s ∈ [0, 1]` (FACT, WL82). Any subset of lobes is itself non-negative and reciprocal, and its albedo is bounded by the full albedo ≤ 1. `f = f_R + f_S` is exact, and the induced R / S labelling of path vertices partitions path space without overlap or gap.
2. **Field equations (DERIV, checked).** `I(c, j) = Λ(x_cj, −ω_j)` (radiance invariance), `Φ_R = Σ_j f_R cos θ_j w_j I`, `Λ = direct + Φ_R + Φ_S` with emission excluded from `Λ`. Cosine and measure are placed correctly; the unknown is incident radiance, not irradiance, so no directional collapse is built in.
3. **Segment validity table (DERIV, checked).** Σ → Σ: only `A1` can invalidate. Σ → `A0`: always invalid. A → A in the mover's frame: valid iff no stationary geometry intrudes — this is the correct treatment of exposed same-asset interreflection as C (FACT requirement, `docs/Architecture.md` §2) and correctly refuses object-identity freezing. A → Σ: always invalid. CE-1 is caught by the Σ → Σ test; CE-3 is caught because every shadow segment from the mover is invalid while its A → A links are certified.
4. **F ≡ S (DERIV).** Holds by induction for identical cache points, nodes and random numbers, to floating-point tolerance. This is the only one of the three designs with a same-architecture F/S guarantee (`docs/Architecture.md` §6).
5. **K/C ownership.** Segments in C, vertex events in K, B compiled from ER only. Disjoint.
6. **Sharp lobes bypass the finite basis** in transport and at readout; HIST WL58's obstruction (52–80 % loss at α 0.04 for a 256-bin outgoing projection; text checked) is avoided for S events by construction.

### 4.2 Defects

**C-1 (MAJOR; FATAL for CE-6 / CE-2-type features at the stated K). The incident field is point-sampled; band-limiting the *material* lobes does not band-limit the *field*.**
`I(c, j)` is the radiance along one ray per node. The design calls the field "band-limited ≈ α\*" (§2) and argues that R → R loss "does not compound because every R lobe is itself band-limited" (§5). DERIV: that bounds the bandwidth of `f_R`, not of the incident radiance, which contains directly lit patches, shadow edges and the one-sample S-chain values of the hit point — none prefiltered. `Σ_j B[m, j] I(c, j)` is therefore a fixed-direction K-point quadrature of an integrand that is not band-limited at node spacing: an aliased estimate, with an effective sample count equal to the number of nodes inside the lobe.
- Count (DERIV from the designs' own lobe model; Track A uses `4πα²` sr for the reflected lobe, Track C `πα²`; the factor 4 is the half-vector Jacobian and Track C's figure is the half-vector lobe): at K = 128 a node covers ≈ 0.049 sr; a reflected lobe at α = 0.2 covers ≈ 0.5 sr, i.e. about 10 nodes. HIST WL58: even 256 outgoing bins lose 9.9–17.7 % at α 0.25 (text checked; piecewise-constant projection, a different but related discretisation). INF: at K = 128 an admissible α\* is of order 0.3 or more.
- Counterexample CE-6: the strip subtends less than a node; each cache point either hits it or not; neighbouring cache points share the tangent-frame node set, so the error is spatially coherent (banding), not noise.
- Edit consequence: a small rigid motion makes node rays flip between bright and dark hits on **stationary** receivers. S and F agree with each other and both report a signed change that is aliasing, not transport — "estimator noise mistaken for physical change", in a form the F ≡ S test cannot see.
- Because samples are frozen and deterministic, the error does not average out over frames or iterations; only larger K (memory ∝ K, solve ∝ K²) reduces it.
*Required change:* make `I(c, j)` a solid-angle average over the node's cell (several rays per node, or prefiltered footprint) — a Galerkin projection instead of collocation — and charge that multiplier in §7; or jitter nodes per cache point and accept noise instead of coherent bias. Stage 0(ii) tests B "for band-limited incident fields" only, which cannot detect this; it needs a test with an un-band-limited incident field (a small source).

**C-2 (MAJOR). The solved field is the fixed point of a frozen random operator: biased, and its contraction is not certified.**
Chain weights `f_S cos / p_S` are frozen one-sample estimates and the same realisations are reused in every iteration. DERIV: the solve returns `(I − T̂)⁻¹ ŝ` with `T̂` random; `E[T̂ⁿ] ≠ (E T̂)ⁿ` whenever a sample is revisited, which happens on every cycle of the link graph (any enclosure). The result is consistent as samples → ∞ but **biased** at finite K, with a bias that grows with loop gain — largest on the high-reflectance cluster. The design notes the companion fact that realised `ρ` may exceed 1 (§6) and declares it a stop (§10.6), but treats the output otherwise as unbiased-with-correlated-noise (§10.9). Two further contributors:
- Chain termination is unspecified. A stored chain must end: by Russian roulette (weights `1/q > 1`, raising realised gain) or by truncation (energy loss, a bias with a sign).
- S-restricted mixture sampling: `f_S cos / p_S` for a multi-lobe mixture is not bounded by 1 per sample, and ER's refracted inner lobes have a pdf Jacobian that needed a float32 fix near the horizon (FACT, WL83). Heavy weight tails are plausible (INF), unmeasured.
*Required change:* state the estimator as biased-consistent; add a gate on the bias (two independent link realisations, compare converged fields, WL84 split-half style); specify termination; estimate `ρ` on the realised operator with a valid estimator (the design already cites HIST WL54 for this).

**C-3 (MAJOR until audited; the audit as written checks the wrong sum). Collocated B does not inherit ER's passivity certificate.**
`B[m, j] = f_R(ω_j → ω_m) cos θ_j w_j`. Passivity in the flux norm is `Σ_m B[m, j] cos θ_m w_m / (cos θ_j w_j) = Σ_m f_R(ω_j, ω_m) cos θ_m w_m ≤ 1` for every `j`: a K-node quadrature of the albedo, which can exceed 1 when a node sits on a lobe peak near α\* or at grazing incidence (DERIV). §9 proposes a "passivity audit of B's column sums"; raw column sums of B are not the albedo — the `cos θ w` weights of both indices are needed. Passivity in the sup norm is the row sum `Σ_j B[m, j] ≤ 1`; by the symmetry of `F[m, j] = f_R(ω_j, ω_m)` the two conditions coincide only if nodes and weights are the same set on both sides, which holds here. Reciprocity of the discretisation is therefore preserved (verified), passivity is not guaranteed. With "no clamping" a violation is a stop.
*Required change:* define the audit as the weighted sum above for every `j`, at all grazing nodes; or compile B by Galerkin averaging, for which the certificate carries over up to cell-quadrature error.

**C-4 (MAJOR as a contract matter). The source / receiver mappings are not flux-conserving.**
FACT, `docs/Architecture.md` §3: "Source and receiver mappings conserve area-weighted flux … Near-contact spatial information must survive both sides of the K/C interface." LPTG collocates receivers at cache points and interpolates sender radiance `Φ_R` over the nearest cache points. Neither is area-weighted; closure of the realised C (Σ over receivers of gathered flux versus emitted flux) is not controlled. Counterexample: mover brought into contact with the floor (the design's own edit (d)); cache points inside the occluded footprint are interpolated with lit points just outside, leaking light across the contact line — the irradiance-cache leak, and the same class as WL54's convex sub-port structure loss (HIST, per `AGENTS.md`: errors to 42 % at 1 cm ports; the correction that worked used a conservative sum / area-weight mapping). The geometry-only proximity refinement mitigates the resolution, not the non-conservation.
*Required change:* either visibility-aware interpolation with a closure test in the operator suite, or declare the deviation from §3 for human decision.

**C-5 (MINOR, must be specified). Emission / direct split under MIS.** Direct light is next-event estimation with the full ER; S-restricted sampling is MIS-combined with it (§9). For this to count each light path once, the `f_R` part of next-event estimation must have weight 1 and node links that hit the emitter must contribute no emission, while the `f_S` part is MIS-weighted against S-chain hits on the emitter. A single MIS weight applied to the full `f`, or emission counted on node hits, double-counts or loses `f_R · L_e`. Not stated.

**C-6 (MINOR). Outgoing-node interpolation at link hits.** `Φ_R(x, ω)` at an arbitrary link hit is interpolated over outgoing nodes `m`, while readout uses exact `eval`. Interpolating B rows near grazing, where `f_R cos` varies fastest and the `s(μ)` attenuation has kinks at its knots (FACT, WL82: piecewise linear on 5 μ knots), is a second, unaudited angular discretisation inside the recursion. Exact `eval` at link hits costs the same order as the matrix product.

**C-7 (MINOR, already conceded). Sample-resolution invisibility** (CE-5): the mover is seen by readout chains but casts no indirect shadow in the field. F ≡ S hides it; only the reference comparison shows it.

### 4.3 Counterexample walk-through

- **CE-1.** Correct. Stored Σ → Σ links and chains crossing `A1` are invalidated and regenerated; fresh readout chains see the occluder. If G1, G2 are S-class at the chosen α\*, their mutual transport is carried entirely by chains, i.e. by path tracing from the camera or from a rough receiver's K nodes.
- **CE-2.** Formally represented: node ray W → G, one frozen S sample G → M, one frozen S sample M → light or next-event estimation at M through a sharp lobe. Probability of finding the chain per link is the solid-angle fraction of the light seen through two sharp lobes; K = 128 frozen attempts per cache point, no accumulation. Result: a few order-`1/p` spikes, spatially interpolated into blotches, and a C-2 bias if the chain is in a loop. The design names this its weakest class (§10.2, §14.4); on this axis it is a **consistency-without-convergence** defect: the estimator has no path to the right answer at fixed state size.
- **CE-3.** Correct; cost is the mover's full link set.
- **CE-6.** C-1.

### 4.4 Threat patterns

| Threat | Finding |
|---|---|
| New glossy path absent from support | Regenerated suffixes cover it at K frozen samples; CE-2 is undersampled without remedy |
| Changed occlusion, unchanged endpoints | Exact at link resolution (segment test) |
| Narrow lobes / angular compression | Avoided for S; C-1 for sharp *incident* features on R events |
| Broad multi-bounce propagation | Handled by the re-solve; C-2 bias grows with loop gain |
| Low-rank / sparse loss of energy | C-3, C-4 (closure not certified), chain truncation |
| Estimator noise mistaken for change | C-1 (node flips under small motion) |
| Hidden full path tracing | When the lobe census puts most lobes in S (likely at α\* ≈ 0.3, INF from C-1), field and readout are a path tracer with frozen samples |

### 4.5 Verdict (Track C)

**CONDITIONALLY SOUND.** The continuous formulation, the partition, ownership and the invalidation contract are correct, and F ≡ S is provable. The discretisation is a biased-consistent estimator whose bias terms (C-1, C-2) are largest on GM1's target features and are invisible to the design's own F-versus-S and band-limited Stage 0 tests; passivity and flux conservation of the realised operator are not inherited from ER (C-3, C-4).

## 5. Cross-design comparison (physical correctness only)

| Property | A (DDT-D) | B (PR1 / PR2) | C (LPTG) |
|---|---|---|---|
| Continuous formulation | exact identity | exact (sampled Neumann series) | exact partition and field equations |
| Radiance / measure / cosine | correct | correct (inherits REF-ER) | correct in equations; collocation breaks flux conservation (C-4) |
| Reciprocity | used implicitly (reverse discovery needs it, A-3) | not needed | preserved by B; needed for the sup-norm bound |
| Passivity under composition | inherited from ER (`‖K C‖ ≤ 1`, rate unbounded as ρ → 1) | not applicable | **not inherited**: C-2 (random weights), C-3 (collocated B) |
| Convergence | guaranteed, slow near ρ = 1 | fixed depth of the reference | realised ρ must be measured; ρ ≥ 1 possible |
| Bias versus REF-ER | static error of `b0` + thresholding `/(1 − ρ)` + missed cells | none (PR1, PR2); PR3 conditional | structural: C-1, C-2, interpolation, truncation |
| Same-architecture F ≡ S | **not guaranteed** (A-2) | trivially (same-seed render) | **proved** to float tolerance |
| Changed occlusion, unchanged endpoints (CE-1) | covered if discovered | exact | exact at link resolution |
| New specular chain (CE-2) | undefined without A-3 repair | sampled at cold rate, unbiased | represented, K frozen samples, no convergence path |
| Self-shadow-only edit (CE-3) | correct after A-6 | correct | correct |
| Destroyed highlight (CE-4) | correct by overwrite | PR1 exact; PR2 ghost (B-1) | correct (links regenerated) |
| Noise read as change | A-4 | PR2 dirty pixels only | C-1 node flips |
| K/C double count or gap | none | none | none at segment level; emission / MIS split unspecified (C-5) |

Statements that follow on this axis only:

1. **Only B is unbiased, and only because it holds no transport representation.** Its correctness is REF-ER's correctness (DERIV).
2. **Only C has a provable F ≡ S**, and that proof is what hides its bias: F and S share every aliasing and frozen-sample error (DERIV). F ≡ S must not be read as accuracy (same rule as HIST WL56 / WL59 per `docs/Architecture.md` §8–§9).
3. **A's physically adequate repairs converge on C's stored links** (A-2, A-4) plus a forward chain tracer (A-3). On this axis A-as-repaired and C differ mainly in whether dirtiness is found by signed propagation or by segment tests (INF).
4. **All three reach a newly created sharp chain onto a rough stationary receiver (CE-2) only by sampling it**; none carries it in persistent state. A has no defined route, C has a fixed number of frozen attempts, B samples it at the cold rate without bias. On the brief's headline phenomenon the ordering by physical correctness is B > C > A (INF).
5. **A common reference convention is missing.** WL84's depth-4 REF-ER (FACT) is a truncated series; A and C solve to a residual. A shared depth or roulette rule must be fixed before any arm is compared, or truncated energy will be booked as mechanism error. Track C states this; A and B do not.
6. **None of the defects above is a measured property of the frozen ER baseline.** They concern the global mechanisms and their discretisations. HIST items (WL44–46, 54, 58, 70) are cited as possible failure mechanisms in their original scope only.

## 6. Limits of this review

- No measurement supports any magnitude here; bias and coverage statements are derivations with stated assumptions, and the node-count arithmetic uses the designs' own crude lobe-width model.
- The ER lobe-roughness census of the WL83 / 84 population was not inspected (the only value seen is a descriptor-level α_b 0.059 for one hard WL81 material; HIST, WL81), so the R / S split implied by C-1 is an inference.
- GRIS temporal-reuse details (B-3) and all other literature are from memory; no paper text was re-read in this session.
- Worklogs 51–56, 59, 67 and 69 were not re-read in full.
