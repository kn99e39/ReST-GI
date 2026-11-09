# WL80A final architecture decision record

> WL80A post-sealing record (workflow `wf_9ac78ed1-61a`, resumed once after a session interruption; completed agents replayed from cache). Agent texts are reproduced verbatim (headings added by the Coordinator). The sealed Track A / Track B records were not modified. Design-only: nothing was implemented, trained or measured.

**Status:** design-review outcome only. No architecture is adopted or validated; the recommendation is not implementation authority. Canonical K/C ownership is unchanged.

## A. Track A result


# Track A: chair's answers

## Was the original BGC logically defensible?

**Only as a passive parametrisation of K. It was not defensible as a mechanism for source generalisation.**

**What holds (DERIVATION).**
- Per-input passivity ≤ d.
- Linearity, provided z depends only on geometry.
- Positivity.
- Exact per-pair representability at R ≥ 16.

**Why the central claim fails** (rejection agreed by the chair; confidence HIGH):

1. **No identification (CE-1).** Linear, positive, passive, rest-D4h-equivariant and identity-free constraints leave out-of-orbit boundary and deformed contexts free. The chair adds that this has already happened empirically in this operator family (FACT):
   - WL68 l.295: three equal-fit runs give new-port contrast of .94 / 1.22 / 1.36–1.44;
   - WL78: BND_FAR 0.668 / 0.860 across two seeds.
2. **Wrong target.** The only element new relative to WL57–80 (A3, pair-conditioned angular factors) acts on a factor that WL75 l.308 ranks below the spatial map (FACT). The chair qualifies "non-binding" to "secondary": WL79/80 orbit gaps are 1.15–1.41×, and the earlier angular floors were noise-confounded (WL77A). The conclusion is unchanged.
3. **Already tested.** The remaining parts re-assemble earlier mechanisms: the FIA skeleton, the W+UV-Q / RSK relational inputs, and the rest-D4h maps.
4. **Structurally wrong normalisation.** The independent ρ duplicates ΣK, causes a receiver-uniform edit artefact, excludes reciprocity, and uses the loose bound d instead of ω_c·d.

## Which components survive, which are modified, which are removed, and why

| Component | Disposition | Reason |
|---|---|---|
| **Survive (record-only, modified)** | | |
| C1 budget | Deterministic analytic-chord d = 1−β, shared with B; collision probability dropped as a feature (it equals d) | Removes noise; not a speed gain |
| C2 operator | d·ω_c·χ_D·m·z / max(1,S): one denominator over disjoint local/far supports; projection predeclared and its binding fraction reported | Physically valid by the panel's derivation |
| C3 gauge | w merged into u; v normalised | Removes factor gauge freedom |
| C4 relational hygiene | Clipping, tie-inclusive radius sets, a geometric frame, no rest coordinates, a declared σ_t law | The corridor descriptor is still unspecified |
| C5 symmetry | Rest-D4h used for augmentation and as a predeclared test only | Scope-limited |
| Reciprocity condition | d_i·z_ij symmetric plus a declared 16↔64 map | Corrected by the checker |
| **Removed** | | |
| A3 | Removed as the transfer mechanism | Targets a secondary factor; spans the passive cone at R ≥ 16, so it is an output head, not a prior |
| A6 pair-factor cache | Rationale lost with A3 | — |
| Independent ρ and ε | Removed | Redundant; cause the edit artefact; exclude reciprocity |
| **Unresolved** | Rank per pair class; A6 feasibility; corridor specification; reciprocity construction | — |

## New limitations introduced by the corrections

As recorded by Track A; the chair agrees.
- The non-smooth max(1,·) projection behaves like a clamp where it binds, so it may only be predeclared.
- The corridor adds input dimension and memorisation capacity.
- Clipping may hide a second edge.
- The geometric frame is undefined at symmetric points.
- Orbit over-weighting from augmentation (cf. WL80's 2–18 % cost).
- The per-class rank is a hyperparameter that must be frozen first.
- The chord grid has its own discretisation error.
- The local/far window introduces a second length scale.
- The selective-recompute invalidation set dilates by 13–24 mm.

## Is the revised architecture materially better?

**Better for validity, not for unseen-source transfer.** Track A claims no transfer improvement, and the chair concurs: no correction adds identification for out-of-orbit or deformed contexts.

**Chair corrections to the Track A record** (these do not change its verdict):

1. **R-a's limit is stated with the stale 0.45 ring floor.** The noise-free floors are ring ≈ 0.21–0.22 and FIA-AZ ≈ 0.067–0.083 (FACT, WL77A coordinator table), and WL76 calls the ring restriction a design choice. R-a with Q computed explicitly rather than learned is, in substance, Track B's C1.
2. **Track A did not connect R-a and R-b to WL77A H3 and H2.** R-b (learned symmetric Green's kernel via E_sym) is the same exact sandwich operator Track B derived blind (DERIVATION, checked by the physics reviewer).
3. **The repaired A4 + R have direct lineage** in zonal / exchange-factor radiative transfer and in passive macromodelling. This strengthens "not novel".

## What remains unproven

- Per-pair nonnegative angular rank by pair class, against the replica noise floor. This is Track A's decisive unmeasured quantity.
- A6 cost and accuracy.
- Whether any generator's inductive bias happens to transfer.
- The magnitudes of CE-2 to CE-5.
- Diffusion screening near edges and corners.
- The size of any reciprocity benefit.
- The σ_t law under deformation.


## B. Track B result


# Track B: chair's answers

## Mechanisms proposed independently (blind to the repository)

| Candidate | Name | Mechanism |
|---|---|---|
| **C1** | CP-K | Explicit collision-probability intrinsic operator. Exact first flight F_in (sharing rays with B), a scalar collision-density resolvent on a body-fitted graded mesh, and an exact or complement-defined last flight. Nothing learned. |
| **C2** | SBG | Exact single-flight term, plus chart-wise multi-flight kernels with a learned log-ratio τ ∈ [0,1] relative to exact slab or rest bases, and a geometry-only D2 passivity projection. |
| **C3** | CCA | Context-class atlas: MC-tabulated responses indexed by D4h / translation-equivalent boundary contexts, plus a coarse far term. |
| **C4** | NSM | Collision-order near field plus a spectrally truncated modal tail. |

## Which survived review

- **C1 (amended): selected primary.** The chair agrees on the physics core and on source-independence. Three amendments are required:
  - the passivity falsifier must test ω_c·d;
  - the far field must be positivity-preserving, with any clamp predeclared;
  - the lateral (~1.25 mfp) grading must be addressed and refined separately from depth.
- **C2 (amended): deferred as a conditional variant of C1.** The chair agrees with the ordering.

## Which were rejected, and why (the chair agrees with each)

- **C4.**
  - It is a stored table of C1's columns, with no mechanism of its own.
  - Its signed modal tail violates non-negativity by construction. The physics reviewer confirmed this: only the Perron mode is guaranteed non-negative.
  - Its dent rebuild takes minutes.
- **C3.**
  - At rest it is enumeration, and it degenerates to per-port tabulation on any other shape.
  - The corner composition and the dent interpolator were withdrawn: the interpolator had a hidden variable, and it belongs to the feature → response family that already failed.
  - CCA-rest is rightly kept as an MC-only locality control.
- **Sub-mechanisms rejected within C1 and C2.** All are correctly rejected, and the panel re-derived the decisive counterexamples:
  - C1's original split form (passivity breach of 8–12 % of d);
  - the independent-quadrature partition;
  - the uniform lattice with learned ΔW;
  - truncated Krylov treated as exactly linear;
  - the "plateau-only" falsifier;
  - C2's visibility-on-resolvent;
  - C2's product corner composition (biased low already in the Dirichlet wedge);
  - G∞ as a base;
  - fine-tuning on source packets.

## Points where the chair disagrees with or qualifies the Track B record

1. **The cross-cutting rule "learning must never see the source" is necessary but not sufficient.**
   - WL76 FIA had a source-independent, symmetry-invariant learned rule. It still degraded on development BND_FAR: abs_coarse 0.39–0.42 against about 0.13 on training sources, and contrast drifted from 0.76 to 0.96 (FACT).
   - The correct principle is that the learned function's input domain must be covered independently of source sampling.
2. **The BND_FAR coverage inference is partly wrong.** TBE, TBC, SM and SR each have one supervised source (FACT, WL73). The failures are out-of-orbit positions within single-member classes.
3. **Reciprocity under 16-in / 64-out bins is not necessarily ill-posed.** It is plausibly well-posed on 16×16 aggregated blocks if the fine16 groups nest (INFERENCE; unverified in code).
4. **Track B could not know these, being blind.**
   - C1 is WL77A H3 at θ = 0, whose gate performance WL77A left open, and whose admissibility as an FNTO K is an untaken human decision (FACT).
   - C2 is a revised WL77A H2 RBG.
   - C1's "generalisation becomes a convergence question" is therefore an independent replication of reasoning, not new project evidence.
   - C-g is not novel.
5. **C2's "D2 makes energy totals exact" is NOT SUPPORTED.** D2 gives passivity only, and is weak in blue (DERIVATION).
6. **C1's 1–5 s dent rebuild needs an unbuilt deterministic quadrature.** The project's sampled analogue measured 74.3 s and was noise-limited (FACT, WL76 / WL77A).

## Did any surviving design give a credible explanation of unseen-source generalisation?

**Yes for C1, mechanistically, in this medium only, and conditionally.**

Under isotropic, homogeneous, index-matched transport, all incoming-direction dependence enters through the exact F_in (DERIVATION, derived independently by both tracks and by WL77A). C1 has no source- or context-indexed fitted quantity, so its error is independent of which sources were supervised. Whether its quality is uniform enough on corners, vertical edges and side rims at affordable resolution is the central HYPOTHESIS. That hypothesis is testable without training.

**Limits of the explanation.**
- It explains generalisation by **removing learning**. It does not explain why the learned family failed, because the causal shares of data volume, invariance, optimisation and angular basis are unmeasured.
- It does not extend to anisotropic phase, Fresnel or path-history-dependent BSSRDF media.

**C2** shares the source-side explanation but moves the problem to chart and dent coverage. On the box it is enumeration plus compression.


## C. Comparative architecture verdict


# Comparative verdict (no forced single winner)

| Question | Answer | Basis |
|---|---|---|
| **Most defensible** | **C1 CP-K (amended), as a physics operator for this medium.** Its invariants (linearity, non-negativity, passivity ≤ ω·d, exact D4h, source-independence) hold by derivation, independently of any fit. Defensibility drops sharply when C1 is read as an answer to the brief's *learned*-K question. There it answers by substitution, and the FNTO contract (l.17 / l.86) plus WL77A's open question 3 make that a human contract decision. | DERIVATION; FACT (contract text, WL77A records) |
| **Most physically grounded** | **C1.** The exact Peierls / collision-probability operator under g = 0, homogeneous, index-matched media. C2 shares its exact ends and approximates only the multi-flight resolvent. BGC-0 / BGC-R encode no transport law. BGC-R is physically *valid* (passive ≤ ω_c·d), but validity is not grounding. | DERIVATION |
| **Most learnable** | **C2 SBG, conditionally.** It is the only in-scope learned design with a coherent identifiability argument: dense, source-independent supervision over its own input domain (charts), log-space bounded targets, and no fine-tuning on source packets. C1 is trivially "learnable" because it learns nothing. BGC-0 / BGC-R are the least identified (CE-1, realised empirically in WL68 and WL78). C2's learnability is argued, not measured, and it needs privileged interior tallies plus tens of GPU-h of MC. | DERIVATION / INFERENCE |
| **Best computational outlook** | **Evidence insufficient; regime-dependent.** BGC-0 is cheapest to rebuild (tens of ms) and fits 12 GB at N128 / R8, but has no accuracy mechanism. C1 is plausibly competitive only for batched lighting-only reuse at fixed geometry (ESTIMATE ~0.2–0.8 s per 64-light batch, against a measured A75 0.086 s and FIA 0.349 s). Under deformation, C1's 1–5 s rebuild depends on an unbuilt deterministic quadrature (the sampled analogue measured 74.3 s, noise-limited). C2's non-iterative advantage cancels in exactly the side, edge, corner and dent strata. No candidate has any measured cost. | ESTIMATE; FACT for the anchors |
| **Strongest plausible novelty** | **None is novel as a mechanism.** The narrowest residual is C2's monotone nested-ratio output constraint plus the D2 device (PLAUSIBLY DISTINCT, weak), on top of a mechanism already proposed and rejected internally as WL77A H2. C1 is classical numerics (internally WL77A H3, θ = 0). BGC has no novel element. Any value is empirical and would have to be measured. | Novelty kill searches; FACT (WL77A) |
| **Strongest falsifiable source-transfer mechanism** | **C1.** A new source changes only the exact F_in. "Transfer" becomes uniform quality under discretisation convergence, which is testable without training and against existing references, with predeclared stop rules. C2 shares the source-side argument but moves the problem to chart and dent coverage. BGC has no mechanism: only f_θ's inductive bias reaches out-of-orbit contexts. | DERIVATION |
| **Where evidence is insufficient** | (1) Whether any candidate meets the gates on BND_FAR strata. (2) Whether C1's lateral (~1.25 mfp) flat-source bias and its rest / dent layer-compression bias are material. (3) Far-field positivity and complement consistency. (4) Any cost figure. (5) C2's dent-region compressibility and blend error. (6) Per-pair nonnegative rank (BGC family). (7) Whether the brief's convention (radiance weighting; the "1.8–4 mm" source definition) matches the references. (8) Whether a non-learned K is admissible (contract, not evidence). | — |

**Chair's synthesis.**

The two tracks converge on the same exact object (DERIVATION, checked by the physics reviewer): Track A's E_sym / R-b and Track B's flight–resolvent–flight sandwich are one operator. This convergence is a cross-check of the algebra, not evidence of performance.

The real separation between candidates is therefore not "which representation" but "**what, if anything, should be learned on top of an exactly computable structure**":
- BGC learns the part that physics fixes, and gets no identification for it.
- C1 learns nothing.
- C2 learns only a compression of computable kernels on the box, and adds information only on dent families.

That makes the brief's central question partly a contract question. WL80A does not resolve it.


## D. Recommended candidate


# Panel designation for a future implementation decision

**This is not implementation authority.** Nothing here authorises an implementation, training, measurement, batch, worklog or successor. The user decides whether anything runs. The designation answers the workflow's request for "at most ONE primary candidate" and is a decision-record entry, not advice to proceed.

## Primary: C1 CP-K (amended), with the chair's amendments

It is designated only in a precisely qualified role: **a learning-free, deterministic intrinsic operator under which "can the target K be reproduced at gate accuracy on independent BND_FAR sources?" becomes testable without training.**

It is **not** designated as an answer to the learned-K question, and **not** as a deployable FNTO K. Either reading would require the human contract decision that WL77A left open (FNTO contract l.17 / l.86; WL77A coordinator open question 3; adjudication l.194). The decision record must state that C1 is WL77A H3 at θ = 0.

**Mandatory amendments to Track B's specification** (chair, from the panel):
1. The passivity falsifier tests ΣK > ω_c·d·(1+1e-6), not d.
2. The far field must be positivity-preserving: non-negative interpolation, or explicit / matrix-free ray-telescoped W. Any clamp-and-renormalise must be predeclared in the protocol, and its binding fraction reported (AGENTS.md: no post-hoc energy clamps).
3. Lateral and depth refinement are predeclared and reported as separate axes. A memory-limited level is classified in advance as "affordability-limited", not as a falsification of the factorisation.
4. All discretisation, quadrature, solver and clamp settings are frozen by reference-free criteria before any MC score is read.
5. The step-0 convention gate also resolves the brief's "1.8–4 mm" source definition and the radiance-weighting convention from the code, read-only.
6. External cost statements obey the baseline terminology gate. The path-traced re-render is comparison class 3, not Neural Full Recompute.

## Secondary: C2 SBG, conditional

It becomes relevant only if two things hold:
- (a) C1 passes its convention and contract gates and reaches the quality gates on at least the rest strata;
- (b) the user decides that a physics-sandwiched learned scalar resolvent is an admissible learned K.

Even then, its decisive question is cost and dent amortisation against C1-in-the-loop. Any write-up must disclose that it is a revised WL77A H2 RBG and that its "exact energy totals" claim is not supported.

## Not designated

- **BGC-0 and BGC-R.** Agreed with Track A's rejection. No mechanism identifies out-of-orbit contexts (CE-1, realised empirically in WL68 and WL78); its distinctive element targets a secondary factor; and there is no novel mechanism.
- **C3 and C4.** Agreed with Track B's rejections.

## Explicit statement on the brief's literal question

None of the four candidates is defensible today as a **validated learned K** that generalises to unseen boundary sources. The strongest available mechanism for unseen-source generalisation (C1) contains no learning, and the only learned design with a coherent identifiability argument (C2) depends on C1.


## E. First experiment design (design only, not an executable batch)


# First-experiment DESIGN for the primary (C1 CP-K, amended plus chair amendments)

**Status.** This is a design only, not an executable batch. Nothing here is authorised. Running it requires a user decision, and a protocol committed before any result.

## 1. Core falsification question

Freeze every discretisation setting using reference-free criteria. Under the contract's conventions, does the learning-free collision-probability operator reproduce the Monte Carlo reference K on independent INT, BND_NEAR and BND_FAR strata, at rest and at the existing dented pose? The thresholds are the historical gates:
- abs_coarse ≤ 0.10;
- abs_fine16 ≤ 0.15;
- con_coarse ≤ 0.25;
- dent signed change: relative error ≤ 0.35 and correlation ≥ 0.90.

Every claim must also hold exactly: linearity, non-negativity, passivity ≤ ω_c·d and exact D4h permutation equivariance.

**Possible outcomes and their predeclared meaning.**

| Outcome | Interpretation (INFERENCE) |
|---|---|
| (i) Pass | The target K is deterministically computable at gate accuracy. The learned-family BND_FAR failures concern learning, not the target or the reference. |
| (ii) Miss while still converging at an affordable level | Affordability-limited. This is not a falsification of the factorisation. |
| (iii) Miss with observed order < 0.5 | A contract, convention or reference mismatch, or a lateral-bias floor. Still a fail for C1. |
| (iv) Contract-test failure | The implementation or discretisation is invalid. Stop. |

## 2. Minimal prototype

- **Mesh.** Body-fitted and anisotropically graded.
  - Depth layers 0.05 / 0.05 / 0.1 / 0.1 / 0.2 / 0.2 / 0.4 mm … toward the mid-plane, on every face.
  - Edges graded as a product in the cross-section.
  - Lateral spacing is a **separately declared axis**: L0 = 1.25 mm, L1 = 0.625 mm, L2 if affordable.
  - Dent: the same topology under a vertical stretch map.
- **F_in.** An exact ray integral of each (port, incoming bin) packet. It uses the same rays as B, so Σ_p q = d holds to float precision.
- **Φ_pt and W1.** One shared, deterministic, stratified isotropic ray set per fine first-collision point. Gauss quadrature of e^{−σ_t s} along chords gives uncollided escape to (port, bin) and the next-collision deposit. Two rules apply:
  - the telescoping partition 1ᵀΦ_pt + 1ᵀW1 = 1 holds exactly;
  - chords that leave the medium, including into the dent cavity, terminate there; re-entry belongs to C.
- **W.** A ray-telescoped cell flight kernel. The near field is explicit (about 3 mfp). The far field is positivity-preserving: either non-negative interpolation or a matrix-free explicit kernel. Requirement: 1ᵀW ≤ 1.
- **Φout.** Its total is defined as the complement 1 − 1ᵀW, and its shape over (j, o) comes from quadrature under the reference's radiance-weighting convention.
- **Clamp.** If 1ᵀW > 1 occurs, the clamp policy is predeclared and its binding fraction is reported.
- **Solver.** Converged to 1e-6 relative, or a fixed linear iteration. A coarse correction is accepted only with non-negative restriction/prolongation and a sub-stochastic coarse operator.
- **Scope.** Rest first (template convolutions per face); then the existing dented pose with a local rebuild plus a global re-solve.

## 3. Mathematical contracts (hard tests; each failure is a stop)

1. Linearity, and superposition on random non-negative incident fields (relative ≤ 1e-6).
2. Zero in gives zero out.
3. Entrywise non-negativity (no entry below −1e-9·max).
4. Passivity per column: ΣK ≤ ω_c·d·(1+1e-6). This is the chair amendment; the d test is too loose.
5. Σ F_in = d against B to fp precision.
6. Per-point and per-cell partitions to fp precision.
7. Exact D4h permutation equivariance at rest, when the operator is assembled from orbit representatives.
8. Reported, not gated:
   - the reciprocity defect on 16×16 aggregated blocks, if the fine16 groups nest onto the incoming grid (check this in the bin code first);
   - the clamp binding fraction;
   - per-channel solve iteration counts.

## 4. Synthetic correctness tests (no MC needed except where noted)

| Test | Check |
|---|---|
| Half-space H-function albedo, normal and grazing incidence | ≤ 1 % (Track B step 0(a)) |
| Single scatter: Φ_pt·q | Against analytic slab single scatter. Against an MC single-scatter tally (step 0(b)); this needs new short MC only if the packets carry no collision-order tallies |
| ω = 1 furnace | Total output = Σq = d exactly at convergence (DERIVATION from the identity) |
| ω = 0 | K = 0 |
| Infinite-slab reflection and transmission | Against adding–doubling or an analytic slab solution at interior ports |
| Reference-free self-convergence | Observed order between refinement levels, reported separately for depth and lateral axes, using only operator outputs (no MC) |
| Rest-mesh vs stretched-mesh null test | Apply the dent-stretch machinery with zero displacement; the output must equal rest to fp precision. This isolates rebuild-pipeline errors from physics |

## 5. Physical training data

**None.** Nothing is learned. MC references are used only for validation, and only after settings are frozen.

## 6. Validation data and historical baselines

**Existing references.** Verify availability read-only before protocol freeze. Facts are as reported by the computation reviewer.
- S16-R replicas 4–7, kept as declared held-out. Replicas 0–3 are also admissible, since nothing is fitted, but report them separately.
- The WL77A Part 0 packets, including the BND_FAR strata scored in WL78.
- DEF71 / CONFDEF72 / DEF73 deformation references.

**Prospective independent population.** The WL79/80 19-port population is unreferenced (FACT). New references cost ESTIMATE ≈ 2.4 GPU-h. They would be generated **after** settings are frozen and scored once. This is the cleanest independent confirmation.

**Historical comparators.** Archived numbers only, each labelled by evaluation regime and never compared across regimes without the label.
- A75 (BND4_FAR con_coarse 0.803 / 0.824).
- Q78 (BND_FAR 0.668 / 0.860).
- FIA (WL76).
- The G2 per-source oracle (0.159), as the non-deployable upper reference for the learned family.
- The MC oracle cost (380 s per configuration in WL55; 188.93 s selective update in WL56).

**External.** None in this experiment. The path-traced re-render is a separate reference (class 3), not Neural Full Recompute. No external superiority claim is made. An 8DNA or RNA comparison would require the mandatory baseline gate and is out of scope.

## 7. Independent confirmation strategy

1. **Commit the protocol first.** Mesh levels, quadrature orders, far-field scheme, solver tolerance, clamp policy and stop rules are hash-committed before any MC score is read.
2. **Paired contrast control.** The exact-F_in operator must reproduce the reference con_*, and a bin-averaged-F_in run must collapse it. Neither result is evidential alone.
3. **CCA-rest locality control** (MC-only; Track B's optional control). Context-locality pairs on native, fine16, abs_coarse, con_coarse, the far tail and side exits. This separates a discretisation failure from a contract or reference issue.
4. **Independent quadrature spot-check.** Recompute Φ_pt and W1 for a few hundred points with an independent high-q stochastic estimator (q ≥ 65,536, the WL77A noise-floor regime). This catches deterministic-quadrature bugs.
5. **Prospective population.** Score the never-referenced WL79/80 ports once, after freezing.

## 8. Primary metrics

**Per stratum (INT, BND_NEAR, and each BND_FAR sub-stratum), per channel.**
- Worst-channel upper-95 % jackknife bounds on abs_coarse, abs_fine16 and con_coarse. abs_native is diagnostic.
- The criterion BND_FAR ≤ max(1.25 × INT, INT + 0.03). For C1 this reads as uniform quality, not transfer.
- Dent signed change: relative error and correlation per channel.

**Diagnostics.**
- Observed order per axis, and Richardson estimates (diagnostic only).
- Clamp binding fraction.
- Passivity margin distribution.
- Reciprocity defect.
- Red abs vs con split, to detect lateral re-randomisation bias.

## 9. Runtime and state cost accounting

All measured on the declared main server, with the hardware labelled.
- Per-W application time.
- K application time for one lighting state and for a 64-light batch.
- Inner iterations per channel.
- Dent rebuild time (deterministic quadrature) and influence-set size (including bottom, side and rim sources: D_T ≠ D_G).
- Peak memory and GPU duty.
- Comparison anchors, labelled: A75 / FIA apply (measured), the MC oracle (measured), and the path-traced re-render at matched quality as a separate reference.

Cost does not falsify the physics. It decides only the deployability verdict.

## 10. Stop conditions (predeclared)

| Condition | Action |
|---|---|
| Step-0 convention miss (H-function > 1 %, or single-scatter mismatch beyond the MC CI) | Stop interpretation. No strata scoring |
| Any hard contract test fails | Stop |
| Budget reached (Track B: ≤ 6 GPU-h, peak ≤ 11 GB) | Report at the finest affordable level. If a finer level is infeasible in memory, classify as "affordability-limited". No further levels without a new decision |
| Gate miss at the finest level | C1 falsified for this discretisation. Classify per §1 (ii)/(iii) by observed order. A still-decreasing error is no excuse for the verdict |
| Dent signed-change failure with rest pass | Report as a dent-specific failure. Run the null test and the full-rebuild control before attributing a cause |

## 11. Evidence needed before any full Method study

1. A pass on every stratum at rest and at the dent, on the prospective population.
2. Demonstrated signed-change bias cancellation.
3. A cost measurement in some declared regime, against the path-traced reference and, if external neural claims are wanted, under the full baseline gate (Native Reuse vs Neural Full Recompute, with F/S separate).
4. **The human contract decision** on whether a non-learned or physics-sandwiched K is admissible under FNTO.
5. Only then a learned-component question (for example C2), evaluated against C1 as the learning-free control at equal accuracy.


## F. Lab-meeting summary


# WL80A: summary for the professor

**What this is.** A design-only thought experiment. Nothing was implemented, trained or measured.

**Problem.** ReST-GI's intrinsic operator K maps light entering a material domain to the light that first exits it. It must be reusable under lighting and scene changes, so it must predict responses for boundary source positions it was never supervised on. The test medium is a 100 × 100 × 10 mm homogeneous slab (mean free path ≈ 1 mm, albedo 0.999 / 0.99 / 0.9, isotropic, index-matched), with 3,840 ports × 16 / 64 angular bins.

**Observations (FACT).**
- Every learned operator tried fits its 16 supervised sources but misses the gates on independent boundary-far sources: con_coarse ≈ 0.67–0.95 against a 0.25 gate.
- Equal-fit runs disagree on held-out ports (WL68, WL78).
- Transfer often degrades late in training.
- A per-source oracle with a fixed angular basis reaches ≈ 0.16, so the spatial geometry → transport map, not angular rank, is where transfer fails (WL75).

**Why the current approaches are insufficient (INFERENCE).** In this family, nothing structurally connects out-of-orbit boundary contexts to the supervised ones. Passivity, linearity and symmetry constraints do not identify those contexts.

**What was reviewed.**
- **Track A** critically audited the proposed BGC-Transport (pair-conditioned learned angular-spatial factors). It rejected BGC as a source-generalisation mechanism. Its corrections improve physical validity (a proper ω·d energy bound, no redundant normalisation) but not transfer.
- **Track B**, blind to the repository, proposed four mechanisms. It selected C1 CP-K, an explicit collision-probability operator: exact first flight → scalar collision-density solve → exact last flight, with nothing learned. It deferred C2 SBG, a learned bounded scalar Green's function between exact flights, and rejected C3 (context-atlas tabulation) and C4 (signed modal tail).
- An independent panel (physics, generalisation, computation, three novelty searches) cross-reviewed both tracks.

**Selected mechanism and rationale.**
- C1 is the only candidate whose source generalisation is structural: a new source changes only an exactly computable right-hand side.
- The question "does it transfer to unseen boundary sources?" therefore becomes "does the boundary-layer discretisation converge to the gates?". That can be tested without training and against existing references.
- Its physical invariants hold by derivation.

**Evidence available today.** Derivations only, cross-checked by several independent reviewers:
- the energy identity;
- the source-independence;
- the defect in the original split that was found and repaired;
- the finding that both tracks derived the same exact operator.

No accuracy or cost number exists for any candidate.

**Claims not yet supported.**
- That C1 meets the gates on edge, corner and side strata, or on the dent signed change.
- That C1 is cheaper than path-traced re-rendering. The estimates suggest it probably is not under deformation; lighting-only reuse is the plausible regime.
- That any learned K generalises. C1 contains no learning.
- Novelty. C1 is classical transport numerics, and internally it is the learning-free WL77A H3 operator. C2 is a revised WL77A H2.
- That C1 is admissible as an FNTO K. The contract says learned, so this is a human decision.

**The most important next experimental question.** Can a learning-free collision-probability operator, frozen by reference-free convergence criteria, reproduce the Monte Carlo reference K at gate accuracy on independent boundary-far sources and on the dent signed change?
- A **pass** would show that the target is deterministically computable, and that past failures were failures of learning.
- A **converging miss** would point to a contract or reference mismatch.
- A **plateau** would bound explicit discretisation.

This is a decision input, not a validated method.


## Unresolved disagreements and open questions


# Unresolved disagreements and open questions

## A. Preserved disagreements (arguments on both sides recorded; not settled by agent count)

1. **Learned versus explicit K (contract).**
   - Track B reviewers and the C1 developer agree that C1 substitutes for learning rather than answering the question.
   - The FNTO contract (l.6 / l.17 / l.26 / l.86) requires a *learned* K, and says not to replace learned shading with a limited explicit solver.
   - WL77A left the same question open (coordinator open question 3; adjudication l.194).
   - **This is a human decision.** WL80A does not resolve it.
2. **The value of C2's chart-space learning.**
   - The developer's view: it is the only coherent role left for learning.
   - Track B reviewers: it is compression on a box, and its dent compression is unproven.
   - The generalisation reviewer and the chair: it is the only learned design with defensible identifiability, but only after C1 exists as teacher and control.
3. **Reciprocity well-posedness under 16-in / 64-out bins.**
   - Track B chair: ill-posed.
   - Physics reviewer: well-posed on 16×16 aggregated blocks if the fine16 groups nest.
   - Unverified in the bin code.
4. **Sign of C1's flat-Φout grazing bias.** The magnitude is agreed at a few percent; the sign is not.
5. **Red context-equivalence length.**
   - Slab-mode, leakage-dominated: ≈ 3.5 mm (reviewers, physics reviewer, chair).
   - Infinite-medium: ≈ 18 mm (C3 developer).
   - Diffusion is unreliable within a few mfp of the boundary, so this needs measurement.
6. **Discretisation route for C1.** A uniform FFT lattice or a graded body-fitted mesh. Graded is committed, but far-field / complement consistency on a graded, deformed mesh is unverified.
7. **Whether the angular factor is "non-binding" (Track A) or "secondary" (generalisation reviewer, chair).** This does not change any verdict.

## B. Open technical questions (all unmeasured)

1. Does C1 meet the gates on the BND_FAR strata (corners, vertical edges, side rims) at an affordable graded resolution?
2. Magnitude of C1's lateral flat-source re-randomisation bias at 1.25 mm (estimate: a 0.2–0.4 relative inflation of lateral diffusion; red decay length long by ~9–18 %). Is it visible mainly in red abs metrics and the dent signed change?
3. Does bias cancel in the dent signed change, given that the vertical stretch compresses graded layers to ~40 % under a 6 mm dent?
4. Far-field sign and mass consistency (pFFT vs non-negative interpolation vs matrix-free); clamp binding fraction.
5. Can deterministic grazing-bin flight quadrature reach the noise floor in seconds? The sampled precedent is 74.3 s and noise-limited.
6. Inner iteration counts and DSA robustness at 0.05 mm layers; warm-started joint K∘C cost in multi-domain scenes.
7. Do the second and third graded refinement levels fit in 11 GB? If not, a "fail at the finest affordable level" verdict may reflect affordability rather than the factorisation.
8. C2: the H-matrix rank of G_rest − G_dent at gate accuracy; the chart partition-of-unity blend error; D2 energy error in blue; the supervision count for τ_dent.
9. Per-pair nonnegative rank by pair class, for any learned pair form (Track A's decisive unknown).
10. Causal shares (data volume, invariance, optimisation, angular basis) of the historical BND_FAR failures. No candidate measures these.

## C. Contract and data ambiguities

1. **The "1.8–4 mm" boundary-near source distance.** WL79 l.273 confirms the figure (FACT), but it matches no port-centre distance (1.25 + 2.5k mm). The definition must be fixed from code before any C1 scoring.
2. **Collision-order tallies.** Do the archived MC packets carry them? This decides whether step 0(b) needs new MC.
3. **Radiance-weighting convention** of the port/bin reference. It must be matched exactly by Φout and F_in.
4. **Bin nesting.** Do the fine16 output groups nest onto the 4×4 incoming grid? This decides the reciprocity diagnostic.
5. **Reference population.** The WL79/80 19-port population is unreferenced. Generating references (~2.4 GPU-h ESTIMATE) would provide the cleanest prospective confirmation population.

## D. Literature checks left open

1. CP or interface-current methods with escape resolved at high angular order on fine patches.
2. Directional-exitance zonal or volumetric radiosity variants.
3. Fan et al. §4 (general boundary conditions).
4. Monotone-ratio constraints on learned Green's functions.

## E. Process note

No candidate is validated. No implementation, training, measurement, worklog, successor or batch is authorised by this decision record.

Note on the user's relayed status question ("상태 확인해봐. run이 없고 dead? 만 하나 남았는데?", roughly: "check the status, there's no run and only a 'dead?' left"): this chair step ran to completion. The seal sha256 was re-verified (all ten files match). No sealed or repository file was written or modified.


---

## Independent critic of this decision record (preserved; the record above was not rewritten in response)

# WL80A chair decision record: independent critic's findings

**Scope.** I wrote no file and modified nothing; every repository read was read-only. I ran no web searches. All re-derivations below are mine.

**Your status question** ("run이 없고 dead? 만 하나 남았는데?"). This critic step ran to completion. I did not inspect the workflow runtime, so I cannot tell you why it shows no run and a "dead?" entry. That needs a separate check.

## Verified as correct (FACT, re-checked this session)

- **Seal.** The sha256 of all 10 sealed files matches `docs/plans/wl80a/seal.json`.
- **Worklog anchors** all match their cited lines:
  - WL75 l.308 ("Global angular rank is not the bottleneck… the spatial geometry → T mapping is").
  - WL73 l.147–150: TBE 3439, TBC 3823, SM 408, SR 527.
  - WL79 l.273: BND sources 1.8–4.0 mm from the perpendicular face.
  - WL68 l.295: .94 / 1.22 / 1.36–1.44.
  - WL78 l.275: BND_FAR 0.668 / 0.860.
  - WL76 l.237 and l.230: FIA BND_FAR abs_coarse 0.39–0.42, against TRAIN_KEEP 0.131 / 0.128.
  - WL76 l.263: contrast drift 0.76 → 0.96.
  - WL76 l.252: rebuild 75.0 s, of which quadrature is 74.3 s.
- **WL77A records** match: `coordinator_decision.md` l.13–15 and l.39 (noise-free floors 0.22 / 0.21 and 0.067 / 0.083), and `adjudication_draft.md` l.194.
- **Contract text** matches: FNTO contract l.17.
- **Re-derived algebra** (DERIVATION) agrees with the record:
  - Escape after first collision is ½[1 − μ ln(1+1/μ)], from the Laplace transform of E2. It equals 0.424 at μ = 0.05.
  - The energy identity holds: total output = ω·1ᵀq − (1−ω)·1ᵀψ, where ψ counts collisions after the first.
  - Slab-mode lengths: with L_eff = 10 + 2×0.71 mm and D = 1/3 mm, I get 3.56 / 3.08 / 1.63 mm. The infinite-medium red length is 18.3 mm.
  - Red ρ: (arctan k / k) × 0.999 with k² = 0.0757 gives 0.974.

## Issues, most severe first

### HIGH-1. The lab summary inverts the predeclared outcome rule
- **Problem (FACT, internal to the record).** `first_experiment` §1 says a miss that is still converging means "affordability-limited", and a miss with observed order < 0.5 (a plateau) means a contract, convention, reference or lateral-bias floor. The lab summary says the opposite: "A converging miss would point to a contract or reference mismatch. A plateau would bound explicit discretisation."
- **Correction.** Make the lab summary match §1 and §10. A converging miss is affordability-limited. A plateau or low-order miss indicates a convention, reference or lateral-bias floor.

### HIGH-2. Outcome (i) and the lab summary overclaim what a C1 pass would show
- **Problem (INFERENCE).** §1(i) says a pass means the learned-family failures "concern learning, not the target or the reference". The lab summary says a pass shows "past failures were failures of learning".
- The record's own E-C1 table contradicts this. It lists "MC references share a convention with C1 that differs from physics" and "strata too lenient" as competing explanations for a pass.
- A pass also cannot divide the historical failures among their causes (data volume, invariance, optimisation, angular basis). Unresolved B10 itself says no candidate measures those shares.
- **Correction.** A pass shows that the target K, as defined by the MC reference, is computable from geometry at gate accuracy on the scored strata. It is conditional on the reference convention, and it does not attribute the learned-family failures.

### MEDIUM-HIGH-3. The primary designation is weakly supported against the brief and WL77A
- **The category shift (INFERENCE).** The brief's open problem is "A learned K must predict…", and its central question is about representation. The chair designates C1 only as "a learning-free… operator under which [the question] becomes testable". That is a diagnostic instrument, not a candidate answer. "At most ONE primary" permitted designating none.
- **The unaddressed precedent (FACT).** WL77A rejected H3 for reasons that apply to C1 unchanged (`coordinator_decision.md` l.15):
  - admissibility is an untaken human decision (contract l.17 / l.86);
  - the apply meets the cap only with an unbuilt kernel;
  - it does not extend beyond g = 0, homogeneous, index-matched media.

  The record discloses that C1 is H3 at θ = 0. It never argues why those rejection grounds do not bar designation.
- **Correction.** Either record "no primary learned-K candidate", with C1 listed separately as a diagnostic instrument pending the contract decision, or state explicitly why WL77A's H3 rejection grounds do not apply to the designated role.

### MEDIUM-4. Existing project evidence on C1's lateral bias is omitted
- **Problem.** The record calls the lateral flat-source bias INFERENCE/SPECULATION and "unmeasured" (§2; Unresolved B2). `docs/plans/wl77a/specialist_physics.md` §7, l.299–307, holds a project non-evidence MC emulation of exactly this mechanism: re-emission of the k ≥ 2 collisions from a reduced-resolution density, with single scatter kept exact. Its results (FACT, labelled non-evidence there):

  | Representation | Result |
  |---|---|
  | Isotropic piecewise-constant voxels | Red energy bias +2.1 % at 0.5 mm, +6.0 % at 1.0 mm, +17 % at 2.5 mm |
  | Lateral snap to 2.5 mm port columns, exact depth, interior | Energy 0.0 %, native .096, coarse .065 |
  | Same, edge | .20, energy −11 % |
  | Same, side | .36, energy −23 % |
  | Same, pair contrast | .05–.08 |

- This supports the chair's predicted signature: contrast largely preserved, absolute error affected, with the boundary strata at most risk. It also bounds the magnitude at the 2.5 mm lateral pitch.
- **A related accuracy point.** H3 used 0.5 mm boundary bands and 1.25 mm interior cells (`candidate_H3.md` l.78). "C1 is H3 at θ = 0" is therefore the same mechanism class with a different discretisation: graded 0.05 mm depth and complement-defined escape.
- **Correction.** Cite §7 as existing non-evidence for B2. Restate the identity as "C1 is the H3 θ = 0 operator class, with a revised discretisation".

### MEDIUM-5. Preserved disagreement 5 (red context-equivalence length) is already decidable from the brief
- **Problem (DERIVATION, 2-D diffusion with a K0 lateral profile, source depth ignored).** The brief §3a gives the fraction of interior-source collided energy that falls outside the 128 nearest ports (radius ≈ 13.5–17 mm) as 5.5 / 3.2 / 0.1 % (FACT).
  - The fraction outside radius r is (r/L)·K1(r/L).
  - At r ≈ 15 mm with L = 3.56 / 3.08 / 1.63 mm this gives ≈ 4.5 % / 2.7 % / < 0.1 %, consistent with the measurement.
  - With L = 18 mm it gives ≈ 70 %, inconsistent with it.
- **Correction.** Record the slab-mode length as supported by existing FACT plus this derivation, with the near-boundary diffusion caveat. Drop "measurement is still required" as the reason to keep the disagreement open.

### MEDIUM-6. The passivity bound is stated unconditionally
- **Problem (DERIVATION).** The physics table says "≤ ω·d by identity at convergence, for any h". Track B says "for any h and any W accuracy" (`trackB_selected_candidate.md` l.98), and the chair re-verified that statement.
  - The bound needs 1ᵀψ ≥ 0, which holds if W ≥ 0 entrywise.
  - Non-negative escape needs 1ᵀW ≤ 1.
  - The chair attributes negative far-field entries only to Φout positivity. They can also break the ω·d bound itself.
- **Correction.** State the bound as conditional on W ≥ 0 and 1ᵀW ≤ 1, and link it to amendment 2 and to the E-C1 "passivity breach" row.

### MEDIUM-7. Cost anchors are mislabelled and the estimates hide their assumptions
- **Mislabelled anchor (FACT).** "A75's measured 1–1.4 ms" single-light apply is not a measurement. The WL76 l.248–256 cost table gives A75 only as a 64-light apply, 0.086 s; 0.086/64 = 1.34 ms is amortised (DERIVATION). The only measured single-light anchor in that table is FIA at 0.0039 s, against which C1's 0.07–0.2 s estimate is 18–51×.
- **Ratio arithmetic.** "3–10× of 0.086 s" should read 2.3–9.3×.
- **Hidden iteration assumption (DERIVATION).** At red ρ = 0.974, plain iteration needs ln(1e-6)/ln(0.974) ≈ 524 iterations to reach 1e-6. The estimate implicitly assumes acceleration, but the record lists DSA robustness as unverified (B6).
- **"About 60 GPU-h for a full MC tabulation", labelled FACT.** It is a derivation: 3,840 × 16 × 3.5 s ≈ 59.7 h for one replica.
- **Correction.** Relabel these figures, state the iteration assumption behind the cost estimates, and add FIA's measured single-light figure as an anchor.

### MEDIUM-8. FACT overstatements in the lab summary
- **"Every learned operator tried fits its 16 supervised sources."** The brief says "to the quality gates, or close to them". WL73 classified all three architectures INSUFFICIENT FIT (l.264), and WL76 FIA was INSUFFICIENT FIT (l.168, l.243).
- **"con_coarse ≈ 0.67–0.95".** FIA's development BND_FAR was 0.96–0.99 (WL76 l.237), outside that range.
- **"…so the spatial geometry → transport map … is where transfer fails (WL75)."** This is WL75's interpretation (INFERENCE) placed under FACT. The chair itself qualifies it elsewhere as "secondary", citing WL79/80.
- **Correction.** Qualify or relabel these three statements accordingly.

### LOW-9. "Only candidate whose invariants hold independently of anything fitted"
- **Problem (DERIVATION).** The record's own table shows C2's linearity, positivity and D2 passivity are also structural.
- **Correction.** Narrow it: C1 is the only candidate with no fitted quantity and with exact D4h without Reynolds averaging.

### LOW-10. E-C1 says "Deployable: this fixture: yes"
- **Problem.** This conflates information-wise applicability with deployability. Cost is unmeasured and estimated far above the learned-apply anchors, while the brief makes repeated application inside a GI solve a constraint.
- **Correction.** Relabel as "applicable (inference-time information); deployability unmeasured".

### LOW-11. Lateral-bias wording
- **Problem (DERIVATION).** "Inflates the lateral diffusion length by h²/8 to h²/4" describes the relative inflation of per-flight lateral variance, that is, of the diffusion coefficient. Uniform re-randomisation adds h²/12 to h²/6 against 2/3 mfp² per flight. Length scales as the square root, giving the 9–18 % that B2 states correctly.
- **"Red averages about 38 collisions."** 38 is 1/(1−ρ) for the fundamental mode. It is not established as the mean collision count, so it should be labelled INFERENCE.

### LOW-12. Labels justified by headcount
- **Problem.** Some labels and positions are supported by counting parties rather than by argument:
  - "DERIVATION (checked by at least two independent parties)";
  - "the physics and computation reviewers agree" used as status;
  - the parties listed on each side of disagreement 5.
- I found no selection that rests on a count; the C1 designation is argued.
- **Correction.** Label by the nature of the claim, and record agreement separately.

### LOW-13. Unfair to the Track A record on R-a
- **Problem (FACT).** `trackA_critical_audit.md` l.517 already records the noise-free 0.21–0.22 ring extrapolation. Only `trackA_revised_architecture.md` l.88 cites 0.45.
- **Correction.** Say "the revised-architecture record states R-a's limit with the q = 1024 floor, although the audit noted the noise-free value".

### LOW-14. The reciprocity disagreement talks past itself
- **Problem (FACT, scope).** Track B's "ill-posed" applies to symmetrisation used as a constraint in C3 and C4 (`trackB_selected_candidate.md` l.64). The chair's counter-argument concerns a 16×16 aggregated diagnostic. Both can hold.
- **Correction.** Restate it as a difference of scope, not a contradiction.

### LOW-15. Process
- **Partial reading.** The chair read 4 of the 10 sealed files in full. Claims about the other files rely on panel citations. I spot-checked "Track A did not cite either lineage" and it holds: the only "zonal" hit is the zonal-harmonic sense in LDPRT, at audit l.1007. Each such claim should still disclose its basis.
- **Inconsistent wording on C3 and C4.** §0 says no dimension bears on their rejection, yet the Track B answers restate the rejection reasons with the chair's agreement.
- **Directive phrasing.** The lab summary's "most important next experimental question" reads as a recommended direction. Rephrase it as "the open question the panel identifies as most decisive".
- **Misplaced note.** The closing note answering your status question is embedded in the decision record. It belongs in the workflow log.

## Not found
- No preselection.
- No objection was ignored outright. HIGH-3 is an objection inherited from WL77A that is disclosed but not answered.
- No fabricated measurement.
