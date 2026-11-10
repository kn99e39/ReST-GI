# GM1 design — Track A — Directional Delta Transport

**Status:** design-only review record (2026-11-10), sealed track. Revision `Selective_Recompute` @ `c7b0698`. Nothing was implemented, run or measured. No file other than this one was written.
**Inputs read:** `docs/plans/gm1_architecture_review/00_PROBLEM_BRIEF.md`; `docs/RESEARCH_CENTRIC_TOPIC.md`, `docs/Architecture.md`, `docs/RESEARCH_ROADMAP.md`, `docs/BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md`, `AGENTS.md`; worklogs 84, 58, 46, 77A, 80A, 80B in full; 67 and 70 in part (completion answers and main tables). WL82, 83, 51–56, 59 and 69 are used only through their summaries in the canonical documents and are cited as such. `docs/FNTO_ARCHITECTURE_CONTRACT_V0_1.md` was not opened; the K/C contract is taken from `docs/Architecture.md` §2–§6.
**Labels:** FACT = ESTABLISHED FACT, HIST = HISTORICAL EVIDENCE (original scope only), DERIV = DERIVATION, INF = INFERENCE, HYP = HYPOTHESIS, SPEC = SPECULATION, EST = order-of-magnitude estimate (not a measurement).

## Verdict in brief

**Reject Directional Delta Transport (DDT) as stated; keep a narrowed form as a hypothesis worth one bounded test.**

- The identity `(I − K C1) db = K(de + dC b0)` is exact and is the directional form of incremental radiosity. It defines a *solver ordering*, not a representation, and has no cost content by itself (DERIV, §1).
- Its right-hand side needs `b0(x, ω)` at points and directions that no T0 consumer ever used. A persistent, directionally complete `b0` at sharp-glossy bandwidth is therefore a precondition of DDT. That representation is the open GM1 problem; DDT relocates it and does not solve it (DERIV + INF, §2, §4).
- If `b0` is instead evaluated on demand by T0 path tracing, DDT becomes residual / difference path tracing, which is published (Xu et al. 2024). That form is a retracer and, by the roadmap's GM1 exit clause, does not complete GM1 (FACT for the clause; INF for the mapping).
- Additive deltas are fragile for sharp features (ghost residue, §5). The defensible narrowed mechanism uses the delta only to **discover and rank the transport-dirty set**, and obtains values by **overwriting** dirty state through a selective regather (**DDT-D**, §1).

## 1. Central mechanism

**Setting (FACT, `docs/Architecture.md` §3).** `i = e + C(G) b`, `b = K i`. In GM1, K is the frozen pointwise ER closure applied in the material frame, and C is ray casting in the current geometry. A rigid edit leaves K unchanged in material coordinates (`docs/Architecture.md` §5), so every change is in `e` (direct visibility) and `C`.

**Identity (DERIV).** Subtracting `b0 = K(e0 + C0 b0)` from `b1 = K(e1 + C1 b1)` and writing `C1 b1 − C0 b0 = C1 db + dC b0` gives `(I − K C1) db = K(de + dC b0) =: s`, hence `db = Σ_n (K C1)^n s`. This is defect correction: `s` is the T1 residual of the T0 solution. A T1 iterative solve warm-started from `b0` is the same computation. The only extra content of the delta form is that `s` is evaluated *only where `de` or `dC` is non-zero*, and that is valid only if the stored `b0` is treated as an exact T0 fixed point.

**Narrowed mechanism DDT-D (HYP, falsifiable).** Persist a directional outgoing field `b0` on all surfaces. After a rigid edit:

1. Enumerate the changed-ray set `D` geometrically (rays through the old or new pose of the mover, §3).
2. Evaluate the signed first-order source on `D` from two `b0` lookups per ray.
3. Propagate source magnitude through `K C1` until it falls under a predeclared radiometric tolerance.
4. **Overwrite**, by regathering, exactly the state cells the propagation marks.

**Claim.** At REF-ER-matched quality on affected stationary receivers, the work scales with the radiometric measure of `D` and its significant descendants rather than with the scene. It is then cheaper than a same-architecture full rebuild, and cheaper than a path-traced re-render whenever the output state must be view-independent.

**Falsified if** any of these holds: the directional state cannot be stored at the required bandwidth within the memory cap; the significant dirty set is not small on the GM1 target edits; or the oracle-`b0` version does not beat the controls (§12).

## 2. Persistent state and its information content

| State | Learned? | Content | Note |
|---|---|---|---|
| ER closure per material | learned, frozen | local `eval` / `sample` / `pdf`, lobe parameters | FACT (WL82–84); unchanged |
| Geometry, material frames, acceleration structures for T1 and for the mover's old pose | no | current visibility; the mover's old transform is enough for `C0` on `D` | DERIV |
| **`b0(x, ω_o)`**: outgoing radiance, all surface points, full outgoing hemisphere | open (tabulated or learned) | 4D field, 2 spatial + 2 angular | the binding item |
| `db` / dirty marks | no | signed, sparse in 4D at first order | transient in DDT-D |

**Why `b0` must be directionally complete (DERIV).** For a stationary receiver `x`, `(dC b0)(x, ω_i) = b0(y1, −ω_i) − b0(y0, −ω_i)`, where `y0` and `y1` are the first hits at T0 and T1. Two cases require radiance that was never consumed at T0:

- **Revealed background.** `y1` lies behind the old pose. `b0(y1, −ω_i)` is radiance that left `y1` toward the mover at T0, never toward `x`.
- **Mover at its new pose.** `y1` is a material point `m` on the mover. The lookup is `b0(m, R⁻¹(−ω_i))`, a material-frame direction unrelated to any T0 consumer.

A state populated only along camera- or receiver-driven paths lacks both. A learned `b0` supervised only on consumed directions is exposed to the same mechanism WL46 recorded for K44, which was non-physical at unsupervised (port, direction) pairs (HIST, `docs/worklogs/46_WORKLOG_FNTO_DIRECTIONAL_INTERFACE.md`; a possible failure mechanism, not a defect of ER).

**Angular content (EST).** A GGX lobe of roughness α reflects into roughly `4πα²` sr, so a hemisphere holds about `1/(2α²)` independent lobes: about 300 at α = 0.04. Resolving rather than counting them needs several samples per lobe dimension, i.e. about 10³–10⁴ directions per point. HIST consistent with this: 256 piecewise-constant outgoing bins lose 52–80 % relative L2 at α = 0.04 (`docs/worklogs/58_WORKLOG_PERSISTENT_INTRINSIC_STATE_REUSE.md`). The ER lobe-roughness range was not verified for this review.

**Spatial–angular coupling (INF).** On a near-specular surface `b0(x, ω)` is the reflected environment with parallax, so it is not low-rank as (function of `x`) ⊗ (function of `ω`) when reflected geometry is near. Factorised compression is weakest near contact. This is the directional analogue of the near-contact boundary information loss in WL54 and WL55 (HIST, per `AGENTS.md` summaries).

**Only way around full storage (HYP).** Store `b0` band-limited at a declared angular resolution. Answer sharper queries by continuing a short path in T0 geometry through lobes narrower than the stored resolution, until a lobe wide enough for a lookup is reached. In the limit where nothing is stored, this is on-demand T0 path tracing, i.e. residual path tracing.

**Contract tension (INF; human decision).** A persisted solved `b0` sits in the "incident/outgoing radiance" row of `docs/Architecture.md` §4, whose edit behaviour is "recompute through the current coupled solve". DDT is an incremental way to do that recompute, so there is no ownership conflict. A scene-fitted *learned* `b0` would however be baked radiance, which the same section excludes from runtime rule and state inputs.

## 3. Lifecycle, invalidation and discovery cost

**T0 build (amortised).** Solve `b0` to the state resolution. This is a full directional GI solve and counts as lifecycle cost.

**What a rigid edit invalidates (DERIV).**

- K: nothing, in material coordinates.
- `e`: light samples whose shadow ray crosses the mover's old pose `O0` or new pose `O1`.
- `C`: every ray leaving the mover, and every ray from a stationary point that crosses `O0 ∪ O1`. Per receiver this is the solid angle of `O0 ∪ O1`, minus third-party occlusion.
- `b`: the mover's whole field is invalid, densely, because its incident field changed in every material-frame direction (for a pure translation under distant light only the C-mediated part changes). Stationary receivers are invalid at first order where their lobes overlap `D`, and at higher orders transitively.

**Discovery (DERIV).** First-order `D` is exact and geometric. Sample points and directions on the surfaces of `O0` and `O1`, then trace outward in T1 geometry, and through the old pose for the T0 hit. The number of rays scales with mover area times angular resolution and is independent of scene size. Third-party occlusion is handled by the trace itself. Higher-order invalidation cannot be found geometrically; it is found by propagating magnitude through `K C1`, and that is transport work, not bookkeeping.

**Limit shared with WL70's D6 (HIST, `docs/worklogs/70_WORKLOG_SUN_STATE_REUSE_C1.md`).** Sampled discovery is exact only at the sample resolution.

**Sequences (INF).** Under continuous motion every frame is an edit and nothing is amortised between deltas. Per-frame cost is the full DDT-D update. Accumulated error forces periodic full rebuilds, whose cost belongs to the lifecycle.

## 4. Created and destroyed paths

- **Destroyed** (newly blocked): the negative term `−b0(y0, ·)` on rays now hitting `O1`. It needs the T0 hit, obtained with the mover's old transform.
- **Created** (newly visible): the positive term `b0(y1, ·)` on rays formerly hitting `O0`. This is the revealed-background lookup of §2.
- **New glossy path via the mover:** a lookup on the mover at a never-used direction (§2), followed by a lobe-sampled gather at the receiver.
- **Direct visibility** `de`: the same chord test from the emitters.
- **Specular chains from the light through the mover onto a rough receiver** (caustic-type paths): receiver-side sampling does not find them. They need forward emission of signed radiance from the light or mover, which adds a density-estimation or connection problem at readout (INF; standard SDS difficulty).

**Unaccounted reconstruction.** If any lookup falls back to recomputation, the cost is a T0 path trace per query and must be charged. With a complete stored `b0` there is no hidden reconstruction, but the T0 build was a full directional solve.

## 5. Survival of directional information

- **Transport.** The operator `K C1` is applied with ER `eval` / `sample` and exact ray casting, so no collapse occurs in transport. Collapse can occur only in the stored state: its angular resolution, and whatever represents `db`.
- **Sharp lobes need lobe-sampled gathers** (HIST, WL46: 128 cosine gathers are inadequate at α = 0.04; lobe importance sampling was required).
- **Additive deltas are unsafe for sharp features (DERIV).** `b1 = b0_stored + db` removes an old reflection only if the removal term equals what is baked in `b0_stored` at that cell. Any bandwidth or estimator mismatch leaves an order-one signed ghost at the old silhouette, and can produce negative radiance. Overwriting the dirty cell with a fresh gather `K(e1 + C1 b)` has no such consistency requirement. This is why DDT-D uses deltas for discovery and ranking, not for values.
- **Error propagation is feed-forward (DERIV).** A stored-state error δ enters the result as `δ + (I − K C1)⁻¹ K dC δ`. It appears in the source only, never in the operator, so it cannot destabilise the solve. It is also not corrected: the static error of `b0` is inherited unchanged.

## 6. Coupling and multi-bounce computation

**Convergence (DERIV).** With passive ER (FACT, by construction, WL82–83 per `docs/RESEARCH_ROADMAP.md`) and ray-cast C, `K C1` is a contraction in a flux norm whenever any absorption or escape exists. The WL41–51 divergence mechanisms were learned non-passive kernels (HIST) and do not carry over. Truncating at order N leaves at most `ρ^(N+1) ‖s‖ / (1 − ρ)` with `ρ = ‖K C1‖`; this is a worst-case bound, not a prediction.

**Ordering (DERIV).** Regather the mover first (its full field, from the stale environment). Then regather first-order receivers using the updated mover. Then continue by descending magnitude (Southwell / Gauss–Seidel on the dirty set).

**Signed quantities and variance (DERIV).** On unchanged rays the delta is analytically zero, so there is no difference-of-estimators noise there. This is the real advantage over naive correlated T0/T1 sampling, where `Var(Δ) = Var1 + Var0 − 2 Cov` and the covariance collapses exactly at visibility changes. On changed rays the integrand is of the order of `b` itself, not small. Per-sample variance there is comparable to ordinary path tracing of the same paths. For small motions the positive and negative lobes nearly cancel, and the net signal can fall below the lookup error of the stored state.

## 7. Memory and computation scaling

Notation: `N_x` spatial samples, `N_ω` directions per sample, μ = fraction of state cells whose first-order source is non-zero, φ_τ = fraction marked dirty at tolerance τ after propagation.

- **Memory (DERIV).** Tabulated state: `3 · N_x · N_ω` scalars. EST: 10⁵ points × 5·10³ directions is about 1.5·10⁹ scalars, i.e. several GB before compression. The achievable compression at sharp-glossy bandwidth is unknown.
- **Discovery (DERIV).** Proportional to mover area × angular resolution × 2 traces; independent of scene size.
- **Source (DERIV).** Per changed ray: 2 traces, 2 state lookups, 1 ER `eval`. With path continuation, add the expected T0 path length through sub-resolution lobes.
- **Update (DERIV).** Roughly `φ_τ · N_x · N_ω · g`, with `g` gather samples per cell. A full rebuild costs `N_x · N_ω · g` per sweep times the number of sweeps. The ratio is bounded by `1/φ_τ`; no number is claimed.
- **Per-view image path instead of a state update (DERIV).** Camera-driven delta paths trace the same rays as path tracing unless the changed set is importance-sampled like an emitter. The saving is bounded by the inverse of the affected-pixel fraction, and it must be paid again for every new view.

**Worst cases (DERIV).**

- An occluder that uncovers the dominant emitter: `db ≈ b1`, φ → 1, cost is at least a full rebuild plus discovery and double lookups.
- High-albedo enclosures: after one or two rough bounces the support of `db` is the whole visible scene, and φ_τ is small only through thresholding, which is a bias budget.
- A large or hero mover: its dense regather dominates.
- Mirror chains: the support stays thin in 4D, but only samples can represent it, and order counts are high.

**No a-priori sparsity certificate.** Predicting φ_τ before paying for propagation needs form-factor-like bounds, which is the link/shaft machinery of hierarchical radiosity. Without it the dense case is detected only after the work is done (INF).

## 8. Nonlocal visibility and stationary receivers

- First-order changes on stationary receivers (moved reflection, new or removed shadow and occlusion) are covered exactly by the chord test. This is the transport-dirty ≠ geometry-dirty requirement of `docs/RESEARCH_CENTRIC_TOPIC.md`.
- Receivers that never see the mover but see an affected receiver (second-order glossy) are reached only by propagation. For a sharp lobe the dirty cells form a thin set in (x, ω_o) whose location depends on the first-order result, so cell-level marking at state resolution is needed. Coarser marking over-invalidates; finer marking is not available.
- An undetected first-order ray is unrecoverable downstream (same principle as `docs/BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md`, "a missed candidate is irrecoverable").

## 9. Integration boundary with frozen ER

ER is queried and never modified.

| Query | Where | Purpose |
|---|---|---|
| `eval(ω_i, ω_o)` | stationary receivers on `D`; mover points | explicit connections toward the changed set (signed-source next-event estimation) |
| `sample`, `pdf` | every gather and every continued path vertex | lobe-importance gathers; MIS with changed-set sampling |
| compiled lobe parameters (read-only) | per material | choose stored lookup vs path continuation; choose gather counts |

On the mover, directions are transformed to the material frame, including the tangent frame for anisotropic lobes. ER's local bias (image L1 floor about 0.035–0.04 against the layered reference; FACT, `docs/worklogs/84_WORKLOG_RENDERING_QUALIFICATION_UTILITY_CLOSURE.md`) is inherited. REF-ER isolates it. K/C disjointness is trivial here because K is pointwise. No nonlocal intrinsic K is involved, so GM2 is untouched.

**Extension note (DERIV).** For a non-rigid edit the identity gains a term: `(I − K1 C1) db = K1(de + dC b0) + dK i0`. That requires a stored *incident* field `i0` on the deformed domain as well.

## 10. Principal failure conditions and most dangerous counterexamples

1. **Never-queried directions** (§2). State incomplete, or learned state unsupervised there. The most dangerous, because it attacks the precondition.
2. **Ghost residue under small motion of a sharp reflector** (§5, §6). Large cancelling lobes with a net signal below the state error.
3. **Uncovered window / dominant-occluder edit.** Dense delta, no saving, overhead added.
4. **High-albedo glossy enclosure.** Slow Neumann decay; support becomes dense within two bounces.
5. **Caustic-type chains through the mover.** Not discoverable from receivers.
6. **Edit sequences.** No amortisation, drift, periodic rebuild.
7. **State bandwidth.** If the stored field cannot hold α ≈ 0.04-class lobes within memory, the mechanism reduces to on-demand T0 tracing, which is prior art and a retracer.

## 11. Testable advantage and why it could exist

| Control | Possible advantage | Reason | Limit |
|---|---|---|---|
| Stale state | correct signed change on affected stationary receivers | first-order source is exact on `D` | none in principle; this is the minimum bar |
| Same-architecture full rebuild (cold) | work proportional to φ_τ rather than 1 | unchanged cells are provably untouched at first order | vanishes as φ_τ → 1; a warm-started full solve is the same algorithm without sparsity |
| REF-ER path-traced re-render | only for a **view-independent** state refresh, or many views per edit | a re-render pays per view; the state update pays once | for a single fixed view the proper control is a difference re-render (below), against which no advantage is argued |
| REF-ER difference re-render (correlated T0/T1 sampling; Xu et al. 2024 as the stronger published form) | lookup termination shortens paths; no stored image tied to one camera | stored `b0` replaces T0 recursion | needs the full directional state, whose build and memory are charged |

No comparison with Neural Full Recompute or Native Reuse is proposed or implied. Those are separate roles under `docs/BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md` and are UNMEASURED.

## 12. One bounded decisive falsification experiment (design only; not authorised)

**Question.** With an *oracle* `b0`, does delta-driven selective update beat its controls on GM1-type edits, and does a finite directional state preserve that?

**Scene family.** One scene, frozen ER materials only, taken from the WL83/84 population and including the hard thin / sharp / high-reflectance cluster. A glossy floor and a glossy back wall (stationary), one area light, one rigid ER object.

**Edits (predeclared).**
- E-small: translation plus rotation with overlapping poses.
- E-large: disjoint poses.
- E-adv: the object uncovers the dominant emitter. Adversarial; the expected result is a dense delta.

**Regions (masks from REF-ER T0/T1 and geometry, fixed before the arms run).** Newly visible; newly blocked; moved reflection; second-order (wall cells that see the affected floor and not the mover); unaffected.

**References.** REF-P (layered, T0 and T1) and REF-ER (frozen ER, full path tracing, T0 and T1), each qualified in the WL84 manner with two independent halves and uncertainty at most one quarter of the smallest gated error.

**Arms.**

| Arm | Definition |
|---|---|
| A0 | stale `b0`, no update |
| A1 | REF-ER full re-render at T1, spp ladder; gives the time-to-quality curve |
| A2 | REF-ER difference re-render: converged T0 image plus correlated-seed T1 − T0 |
| A3 | DDT-D with oracle `b0` (on-demand T0 path tracing; unbiased; no stored field) |
| A4 | DDT-D with a tabulated directional `b0` at two predeclared angular resolutions, with the path-continuation rule |
| A5 | same-architecture full rebuild of the A4 state at T1 |

**Measurements.** Per region, against REF-ER (primary) and REF-P (secondary): relative L1; highlight error (WL81 Stage F form); signed-change correlation and slope. Also: energy of `db` by bounce order; μ and φ_τ; rays traced; uncontended GPU kernel time; wall time; peak memory; T0 build cost of the state; ghost and negative-radiance mass.

**Gates.** Numerical thresholds are frozen after the A1/A2 cost ladders exist and before any A3–A5 result is read. No T1 supervision or oracle region mask is available to A3–A5 at runtime.

**Falsifying outcomes.**
- **F1.** On E-small and E-large, A3 does not reach the REF-ER-matched quality level on the affected stationary regions with fewer rays and less GPU time than A1. The identity then has no economic content even with a perfect state. Stop.
- **F2.** A3 passes, but A4 at either predeclared resolution misses the same quality gate, shows ghost mass above the gate, or its state build plus memory exceeds the caps. The representation precondition fails; what remains is A3, which is a difference retracer. DDT is not a GM1 answer.
- **F3.** A4 is not cheaper than A5 at matched quality on E-small and E-large; or on E-adv it costs more than A5 by more than the predeclared overhead allowance without having detected the dense case early.

Passing all three supports DDT-D only on one scene and a rigid edit.

## 13. Closest prior art and honest difference

Four entries were checked by web search in this session: Xu et al. 2024, Günther and Grosch 2015, Dmitriev et al. 2002, Loos et al. 2012. The rest are cited from memory; venues and years are believed correct but page-level details were not verified.

- **Incremental radiosity.** Chen 1990 (SIGGRAPH); George, Sillion and Greenberg 1990 (IEEE CG&A): shooting positive and negative energy after object motion. This is the same identity for diffuse patch radiosity. DDT is its directional restatement; the identity is not new.
- **Dynamic hierarchical radiosity.** Forsyth, Yang and Teo 1994 (EGWR); Shaw 1997 (CGF); Drettakis and Sillion 1997 (SIGGRAPH, line-space hierarchy): shafts find the changed links, which is discovery of `dC` with magnitude bounds. `docs/RESEARCH_CENTRIC_TOPIC.md` already classes the 1997 work as prior art.
- **Directional surface radiance state.** Immel, Cohen and Greenberg 1986 (SIGGRAPH); Sillion et al. 1991 (SIGGRAPH); Christensen et al. 1997 (ACM TOG, glossy clustering). These store exactly the state DDT needs. Their storage growth at sharp lobes is the known obstacle.
- **Selective photon tracing.** Dmitriev, Brabec, Myszkowski and Seidel 2002 (EGWR): pilot photons detect changed regions; corrective photons update them.
- **Progressive difference images.** Günther and Grosch 2015 (CGF 34(4)): stochastic add/remove of the difference image for scene edits, mainly with stochastic progressive photon mapping.
- **Residual path integrals.** Xu, Li, Georgiev, Hedstrom and Ramamoorthi 2024 (CGF / EGSR, arXiv 2406.16302): the frame difference as a correlated path integral, importance sampling from the moving objects, path mappings. **This is the closest work.** Honest difference: it is image-space and fixed-view, evaluates both frames by sampling, and keeps no persistent directional surface state. DDT with on-demand `b0` is that method in other notation.
- **Gradient-domain and control-variate rendering.** Kettunen et al. 2015 (SIGGRAPH); Manzi et al. 2016 (SIGGRAPH Asia, temporal); Rousselle, Jarosz and Novák 2016 (SIGGRAPH Asia): shift mappings and the previous image as a control variate.
- **Delta Radiance Transfer.** Loos, Nowrouzezahrai, Jarosz and Sloan 2012 (I3D): low-rank delta operators for clutter objects; low-frequency.
- **Antiradiance.** Dachsbacher et al. 2007 (SIGGRAPH): signed radiance in place of explicit visibility; documents the cancellation behaviour of signed transport.
- **Caches and temporal reuse.** Gautron, Bouatouch and Pattanaik 2007 (IEEE TVCG, temporal radiance caching); Müller et al. 2021 (SIGGRAPH, neural radiance caching, online adaptation); Hadadan et al. 2021 (SIGGRAPH Asia, Neural Radiosity, trained on the rendering-equation residual); Ouyang et al. 2021 (HPG) and Lin et al. 2022 (SIGGRAPH) for ReSTIR GI / PT temporal path reuse.
- **Operator perturbation view.** Arvo, Torrance and Smits 1994 (SIGGRAPH).

**Novelty position (INF).** Neither the identity, signed propagation, changed-set importance sampling, nor delta-as-residual is new. The only candidate for a contribution is a directionally complete, sharp-glossy-capable persistent state together with a discovery-then-overwrite update whose cost is certified by the dirty measure. The absence of an exact prior match for that combination is not evidence of novelty.

## 14. Self-assessed weaknesses

- Nothing here is measured. Every cost statement is a derivation or an estimate, and the angular-count estimate uses a crude lobe-width model.
- The proposal contains no learned global component. As a GM1 answer it is a classical incremental solver around frozen ER, so it may fail the project's intended contribution even if every gate passes. Whether that is admissible is a human decision, as in the WL77A / WL80B solver-admissibility questions.
- The narrowed DDT-D is close to "transport-dirty discovery plus selective regather", which is the existing Selective_Recompute concept with a specific discovery signal. The delta may add little beyond ranking.
- The falsification experiment's A4 state is a plain table. A failure there does not rule out a better representation; a pass does not show it scales.
- The ER lobe sharpness actually present in the WL83/84 population was not checked, so the severity of the bandwidth problem for those materials is unverified.
- WL82, 83, 51–56, 59 and 69 were used through canonical summaries only; the FNTO v0.1 contract was not re-read.
- Non-rigid edits, nonlocal K, continuous motion and accumulated edits are argued, not analysed in depth.
