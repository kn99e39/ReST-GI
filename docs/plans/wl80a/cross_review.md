# WL80A cross-review

> WL80A post-sealing record (workflow `wf_9ac78ed1-61a`, resumed once after a session interruption; completed agents replayed from cache). Agent texts are reproduced verbatim (headings added by the Coordinator). The sealed Track A / Track B records were not modified. Design-only: nothing was implemented, trained or measured.

## Chair synthesis


# WL80A post-sealing comparison: chair's cross-review

**Chair basis.**
- I re-verified the seal: the sha256 of all ten sealed files matches `seal.json` (FACT, recomputed this session).
- I read these sealed files in full: `neutral_problem_brief.md`, `trackA_original_bgc.md`, `trackA_revised_architecture.md` and `trackB_selected_candidate.md`. For the other sealed files I relied on the three panel reviews, which cite them line by line.
- I re-checked these repository anchors read-only (FACT):
  - WL75 l.308: "Global angular rank is not the bottleneck … the spatial geometry → T mapping is".
  - WL73 l.147–150: the S16-R boundary sources are TBE edge row (port 3439), TBC slab vertex (3823), SM side-mid (408) and SR side-rim (527).
  - WL79 l.273: BND slots sit 1.8–4.0 mm from the nearest perpendicular face.
  - WL80: the WL79/80 19-port population "remains unreferenced and unscored".
  - `docs/plans/wl77a/coordinator_decision.md` l.13–15: H2 RBG and H3 LCR were rejected. Open question 3 asks whether a classical solver plus closure is admissible as a learned FNTO K, and calls it a contract decision.
  - `docs/plans/wl77a/adjudication_draft.md` l.194: "Whether the learning-free θ = 0 CPM operator (H3 Stage 1) meets the WL59 gates … Its contract status is a human decision."
  - The same coordinator table gives the noise-free extrapolated floors: ring 0.22 / 0.21, FIA-AZ 0.067 / 0.083.
  - `docs/FNTO_ARCHITECTURE_CONTRACT_V0_1.md` l.17: "do not replace learned complex shading with a limited explicit radiosity solver".

**Scope of this review.** I implemented, ran, trained and measured nothing, and I ran no web searches. Literature identities come from the panel's novelty searches at the verification levels they state. I weigh arguments, not how many agents agree.

**Candidates.**
- **BGC-0**: original BGC-Transport.
- **BGC-R**: Track A's record-only corrections C1–C5. Track A rejected this as a source-generalisation mechanism and did not adopt it as an architecture.
- **C1 CP-K**: the amended form; Track B's primary.
- **C2 SBG**: the amended form; Track B's deferred learned variant.

C3 CCA and C4 NSM are mentioned only where a dimension bears on their rejection; none does.

---

## 1. Physics

| | BGC-0 | BGC-R | C1 CP-K | C2 SBG |
|---|---|---|---|---|
| Transport law encoded | None; dρZ/(ε+ΣZ) is a parametrisation | None; validity comes from the projection only | Exact Peierls / collision-probability form (g = 0, homogeneous, index-matched); error is discretisation only | Exact flights, single scatter and single flight; the multi-flight resolvent is exact base × learned τ ∈ [0,1] |
| Energy bound | ≤ d: meets the brief, but exceeds the physical ω_c·d by up to 10 % of d in blue | ≤ ω_c·d by projection | = ω·1ᵀq − (1−ω)·1ᵀψ ≤ ω·d by identity at convergence, for any h | ≤ ω·d through the D2 identity; passivity only, not exact totals (see below) |
| Positivity | Yes | Yes | Yes for exact factors; conditional for a precorrected-FFT far field | Yes in the chart-sum form |
| Edit locality | Receiver-uniform (ρ′/ρ)(S/S′) artefact | Local while S ≤ 1 | Physical: emerges from the solve | Physical up to τ_dent accuracy |
| Reciprocity | Impossible with an independent ρ | A stated condition, not built in | Volume-weighted reciprocity traded for an exact partition; the defect must be reported | Symmetric τ broken by per-column D2 scaling |
| Directional correspondence | Learned; frame gauge exposed | Same | Exact by integration in the contract's own frames, including dented frames | Same as C1 at the ends |

**Rationale.**

- **BGC-0 (DERIVATION).** Every physics objection Track A raised was re-derived independently by the physics reviewer:
  - the ρ redundancy;
  - the loose d bound (it should be ω_c·d);
  - the missing measure and partition;
  - the receiver-uniform edit factor.

  BGC-R repairs validity but encodes no transport law. Any z below the bound is admissible.

- **C1 (DERIVATION).**
  - It is the only candidate whose physical invariants hold independently of anything fitted.
  - The original split form breached passivity (R1-1). The panel reproduced this numerically: escape after the first collision is ½[1 − μ ln(1+1/μ)], which gives 0.424 at μ = 0.05 against a flat-cell 0.278. The amended consistent split closes the breach.
  - **Chair amendment to Track B.** Falsifier 3(b) tests ΣK > d·(1+1e-6). The operator's own identity gives ≤ ω_c·d. A test against d therefore leaves (1−ω_c)·d of slack: 10 % of d in blue, 1 % in green. The falsifier should test ΣK > ω_c·d·(1+1e-6) (DERIVATION; physics reviewer).
  - The computation reviewer's point that fp32 rounding cannot trigger the d-test is correct. It is not a reason to keep the looser test.

- **C1 far field (INFERENCE; the physics and computation reviewers agree).**
  - A precorrected FFT with non-positive interpolation stencils, or with near-field subtraction, can produce negative far-field W entries.
  - Escape is defined as the complement 1 − 1ᵀW, so any far-field mass error moves directly into Φout.
  - Track B listed this consistency question as unverified, but not the sign question.
  - "Clamp and renormalise" is a form of energy clamp. AGENTS.md excludes post-hoc energy clamps (FACT). The same rule Track A applied to its own max(1, S) therefore binds C1: the clamp must be predeclared, and its binding fraction reported.

- **C2 (DERIVATION; novelty kill search, checked by the chair).**
  - D2 gives one linear constraint per column: uᵀĜ = (1 − P_esc)ᵀ, with u = (1−ω) + ω·P_esc.
  - The output total is fixed only if Ĝ's split between absorption and escape is also right. The constraint nearly pins the escape total in red, where (1−ω)·1ᵀG ≈ 0.04. It pins little in blue, where it is ≈ 0.8.
  - The C2 developer's claim of "exact energy totals" is therefore NOT SUPPORTED. The Track B chair's "passivity exact via D2" stands.

- **Reciprocity across the 16-in / 64-out bins (preserved disagreement; the chair leans to the physics reviewer, conditionally).**
  - The Track B chair called reciprocity symmetrisation ill-posed. Direction reversal in a port frame is a 180° in-plane rotation of the concentric square-to-disk map, which permutes the concentric bins.
  - If the fine16 groups nest onto the incoming 4×4 grid, reciprocity is well-posed on 16×16 aggregated blocks, though not entrywise at 16×64. This is INFERENCE; nobody inspected the bin code.
  - This matters only as a diagnostic for C1. The brief does not require it.

## 2. Representation

| | BGC-0 | BGC-R | C1 | C2 |
|---|---|---|---|---|
| Source encoding | Moments, source side only; not sufficient (second-order closure errors +10.4 / +21.8 / −17.6 %) | Moments kept, labelled non-sufficient | F_in exact; first continuation W1 point-exact | Same flights, plus exact single and double scatter |
| Boundary context | Endpoint features only: blind to intervening thinning (CE-5, about 0.3× red change with every endpoint fixed) | Corridor descriptor, unspecified | True geometry enters every flight | Slab and semi-slab exact; quarter-slab and dent learned |
| Structural error floor | Identification floor (CE-1) | Same | None as h → 0. At affordable h there is a real floor (lateral grading, below) | τ fit, chart-blend error, dent-family coverage |

**Rationale.**

- **The decisive representation fact.** The learned spatial geometry → T map is where transfer failed (FACT, WL75 l.308; WL68). Both BGC forms keep that map, and BGC-0 adds angular capacity on top of it.
- **Track A's "wrong target" argument, qualified.**
  - WL79/80 show 1.15–1.41× orbit gaps for a frozen or refitted angular basis (FACT).
  - WL77A shows that the earlier angular floors were confounded by noise.
  - So the angular factor is *secondary*, not non-binding. Track A's "a perfect angular family removes at most ≈ 0.159 of ≈ 0.80" is INFERENCE, not a bound, because the errors do not add.
  - The conclusion still holds: A3 adds per-pair capacity to the very map that failed.

- **Chair finding the sealed tracks lack: C1's lateral grading** (physics reviewer). The calculation is DERIVATION; the magnitude is INFERENCE or SPECULATION.
  - Amended C1 grades depth finely (0.05 mm surface layers) but keeps lateral spacing at about 1.25 mm, i.e. 1.25 mfp.
  - The developer's answer to the 1.5 % re-randomisation bias ("falls under the grading") covers depth only.
  - Flat-source re-randomisation inflates the lateral diffusion length by roughly h²/8 to h²/4: about 0.2–0.4 at 1.25 mm.
  - The point-exact first two collision orders are immune. Red, however, averages about 38 collisions, so most red energy is high-order and exposed.
  - The expected signature:
    - con_* is largely cancelled, because the bias is common to all incoming bins;
    - abs_coarse and abs_fine16 are biased for red;
    - the dent signed change is biased, because the decay length differs between rest and dent.
  - A further dent issue (generalisation reviewer, DERIVATION of geometry): the vertical stretch compresses graded layers under a 6 mm dent to about 40 % of their rest thickness. The rest and dent biases are then not the same function of position in the edit region, so first-order cancellation is plausible but not assured.

- **C2.** Its partition-of-unity blend ½(G_k D_k + D_k G_k) is not an exact composition when x and y fall in different charts. "Exact on the box" applies to individual chart kernels only (INFERENCE; magnitude unmeasured).

## 3. Generalisation

| | BGC-0 / BGC-R | C1 | C2 |
|---|---|---|---|
| Unseen sources | Only f_θ's inductive bias reaches out-of-orbit contexts (CE-1) | Structural: a new source changes only the exact right-hand side F_in | Structural on the source side |
| Unseen boundary contexts | Interpolation from one supervised member per class | Geometry of W1, Φ_pt, W and Φout on the true body; accuracy is a discretisation HYPOTHESIS | Enumeration of chart types (box only); "corner without corner data" withdrawn |
| Symmetry | Rest D4h by averaging; dented poses undefined; gauge exposed | Exact as a *consequence* of correct physics; dented-frame and gauge problems do not arise | Exact at rest if τ is Reynolds-averaged; dent family assumed radial |
| Deformation | No mechanism; corridor unspecified; historical family failed DEF | Handled by recomputation; D_T ≠ D_G absorbed by the global solve | Bounded to a radially symmetric dent family, excluding the evaluated pose |

**Chair judgements.**

- **CE-1 is not only a construction; it has already happened in this family (FACT).**
  - WL68 l.295: three all-group fits with near-identical fit scores give new-port contrast of .94 / 1.22 / 1.36–1.44.
  - WL78: BND_FAR is 0.668 / 0.860 across two seeds.

  This strengthens Track A's rejection of BGC.

- **Track B's cross-cutting rule, "learning must never see the source", is necessary but not sufficient.**
  - WL76 FIA had a source-independent learned R_θ with symmetry-invariant features, trained only through source packets. On development BND_FAR its abs_coarse was 0.39–0.42, against about 0.13 on training sources (FACT, as cited by the generalisation reviewer).
  - The WL77A noise re-attribution explains only about 0.07–0.10 of coarse error, not a threefold gap. Causal attribution is INFERENCE.
  - The chair adopts the stronger principle: **the learned function's input domain must be covered by supervision independently of source sampling.** C1 meets it trivially, C2 meets it by design, and BGC does not.

- **Track B's BND_FAR coverage inference is partly wrong.**
  - TBE, TBC, SM and SR each have one supervised member (FACT, WL73).
  - The historical failures are therefore out-of-orbit positions *within* classes that have a single supervised member, not wholly unsupervised classes.
  - Whether a vertical-edge source is supervised is unverified.

- **What C1's transfer argument is, and is not.**
  - It is structural and correct (DERIVATION).
  - It is not new evidence for the project. WL77A had already rated H3's transfer as carried by a classical solver and left the θ = 0 operator's gate performance as an open question (FACT, adjudication l.194).
  - Track B's blind re-derivation independently replicates the *reasoning*. It is not a measurement.

## 4. Learnability

- **BGC-0.** Less identified than ATO. At R ≥ 16 the factors span the passive cone and w, u are identified only as a product. The training family has documented late drift (FACT: WL71A, WL75, WL76). This is a high memorisation risk.
- **BGC-R.** Removes the gauges and the ρ/ε redundancy. The near-field freedom of CE-1 is untouched.
- **C1.** Nothing to learn. Its hidden free choices are grading, quadrature, far-field cutoff, solver tolerance and clamp policy. These must be frozen by **reference-free** convergence criteria before any MC score is read; otherwise they become tuning on held-out data. Track B's falsifier does not say this (procedural INFERENCE; the chair adopts it).
- **C2.**
  - Identifiability is good: dense, source-independent chart supervision, with exact reduced solves for slab and semi-slab and MC point-source fields for edge and dent. This removes WL77A H2's decisive objection that the near-diagonal block was "identified by smoothness only".
  - It needs privileged interior collision-density tallies at training time, which the current packets do not contain.
  - It is the only in-scope learned design with a coherent identifiability argument.
  - Dropping fine-tuning on S16-R packets (option (c)) is correct, because that would re-import the measured drift.

## 5. Computation

All figures below are ESTIMATES unless marked FACT.

| | BGC-0 | BGC-R | C1 | C2 |
|---|---|---|---|---|
| Learned parameters | ~10⁵ | Unspecified | 0 | τ, ~1e3–1e5 |
| State | 3.8 GB (N128, R8) to 30.6 GB (N512, R16) | Dense N128 block 6.0 GB plus far term | Geometric factors ~3–6 GB; ~3.6 MB solver state per lighting state | Flights plus per-channel Ĝ; 4–8 GB with the dent region open |
| 1-light apply | ≥ 4 ms, memory-bound | ≥ 7 ms | ~0.07–0.2 s cold: 50–150× A75's measured 1–1.4 ms | ~3–10 ms on FFT-structured parts; dent and sides not convolutional |
| 64-light apply | 5–10 ms fused; 0.1–1 s in this codebase's style | Similar | ~0.2–0.8 s: 3–10× A75's measured 0.086 s, same order as FIA's measured 0.349 s | Not established |
| Dent rebuild | Tens of ms | Similar | 1–5 s only with an unbuilt deterministic quadrature. The sampled analogue measured 74.3 s and was noise-limited (FACT, WL76 and WL77A q-noise) | Seconds or more; dent region has ~1e10 pairs |
| Scaling with ports | Quadratic | Quadratic | Linear in flights; W independent of ports | As C1 for flights |
| Albedo edit | Retrain | Retrain | Free: W is channel-free | Partial |

**Chair judgements.**

- **The C1 apply gap depends on batch size; it is not a constant factor (DERIVATION).** C1 is uncompetitive for single-state iterative GI and only moderately behind for batched relighting.
- **On the isolated slab fixture there is no outer GI loop** (convex body, no same-object re-entry). A GI solve there is essentially one K application. Track B's 1–10 s figure is the pessimistic multi-domain case (DERIVATION).
- **BGC-0 is computationally the cheapest to rebuild** and fits in 12 GB at N128/R8. Its rejection rests on identification, not cost.
- **C2's "non-iterative" advantage is roughly cancelled** in exactly the target strata (sides, edges, corners, dent), where no translation structure exists (DERIVATION/INFERENCE; this agrees with Track B's deferral).

## 6. Novelty

From the three kill searches, at their stated verification levels; the chair did not re-verify any citation.

- **BGC-0 and BGC-R.** No novel mechanism. Every element maps onto published work or project history. The repaired A4 and R are textbook practice in:
  - zonal and exchange-factor radiative transfer (summation rules, reciprocity, post-fit smoothing);
  - passive multiport macromodelling.

  Track A did not cite either lineage.
- **C1.** The mechanism is ALREADY KNOWN:
  - integral-RTE solvers with FFT/FMM (Fan–An–Ying; Ren–Zhang–Zhong);
  - 3-D ray-traced collision-probability and first-collision-source methods;
  - zonal methods and translucent radiosity for the joint K + C solve;
  - topology-preserving body-fitted SSS meshes.

  Claim C-g ("unseen-source generalisation becomes a discretisation-convergence question") is NOT SUPPORTED as new. It applies to any deterministic linear solver, it is the stated rationale of DeepRTE and Neural Green's Functions, and the project's own MC oracle K already has the property. Internally, C1 is WL77A H3 at θ = 0.
- **C2.** The mechanism is ALREADY KNOWN, and the same combination was proposed and rejected in the project as WL77A H2 RBG (FACT). What remains PLAUSIBLY DISTINCT is narrow: the monotone nested ratio used as a hard output constraint on a learned transport kernel, plus the D2 projection device.

## 7. Practical value

- **FNTO contract (FACT).** l.6 / l.17 / l.26 / l.86 make "learned versus explicit" a contract question, not only a reviewers' disagreement. C1 conflicts with the contract wording. Adopting it as FNTO's K needs the human decision WL77A already left open (open question 3).
- **C1's clearest practical value is not as a deployable dent-update K** (INFERENCE; computation reviewer, adopted by the chair). It is three things:
  1. A deterministic, noise-free candidate replacement for the MC per-configuration oracle (FACT: 380 s per configuration in WL55, 188.93 s selective in WL56, about 60 GPU-h for a full MC tabulation).
  2. A test of whether the target K is deterministically computable at gate accuracy. That separates "the learned family failed at learning" from "the target, contract or reference is inconsistent".
  3. A potential dense, all-source teacher. This is stated as a consequence, not a recommendation.
- **Lighting-only reuse at fixed geometry** is the one regime where C1 could plausibly be cheaper than a path-traced re-render. That is unmeasured.
- **Baseline terminology (FACT).** The "few seconds" path-traced figure is comparison class 3, not Neural Full Recompute (AGENTS.md / `BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md`). No candidate has an external neural comparison specified.

## 8. Preserved disagreements

1. **Learned versus explicit (Track B reviewers vs the C1 developer; contract text).** Unresolved. This is a human contract decision, and WL80A does not resolve it.
2. **Value of C2's chart learning.** Its developer calls it the only coherent role left for learning. Track B reviewers call it compression on a box. The generalisation reviewer says it is the only learned design with a defensible identifiability story. The chair holds that both are true: it is informative as a *learned* experiment, but only after C1 exists as teacher and control.
3. **Reciprocity well-posedness** at 16×16 aggregated blocks: Track B chair vs physics reviewer (§1).
4. **Sign of C1's flat-Φout grazing bias.** Magnitude agreed at a few percent; sign unresolved.
5. **Red context-equivalence length.** Leakage-dominated ≈ 3.5–3.6 mm (reviewers, physics reviewer, chair) vs 18 mm infinite-medium (C3 developer). The chair finds the slab-mode derivation more applicable. Measurement is still required.
6. **Track A's statement of R-a's limit.** It used the stale 0.45 ring floor, whereas the noise-free values are ring 0.21–0.22 and FIA-AZ 0.067–0.083 (FACT, WL77A coordinator table). WL76 itself calls the ring restriction a design choice. The chair sides with the panel: Track A's record understates R-a.
7. **The cost range for C1** spans more than an order of magnitude. None of it is measured.

## 9. Part E: competing explanations for each surviving design

"Surviving" here means C1 (primary) and C2 (deferred, conditional). BGC-R is abbreviated: Track A rejected it, and the chair agrees.

### E-C1: CP-K (amended, plus chair amendments)

| Item | Content | Status |
|---|---|---|
| Historical failure addressed | Learned operators fit supervised sources but fail independent BND_FAR; late drift; seed-dependent held-out behaviour | FACT (brief §3b; WL68, WL71A, WL76, WL78) |
| Preventing mechanism | No source- or context-indexed fitted quantity. The source enters only as the exact F_in. Boundary physics lives in W1, Φ_pt, W and Φout on the true body | DERIVATION |
| Falsifying observation | Step-0 convention miss; any stratum misses a gate at the finest affordable level; ΣK > ω_c·d; dent signed change rel. error > 0.35 or corr < 0.90 | Predeclarable |
| Should succeed | Rest slab INT and face-centre sources, especially blue (low orders dominate) | HYPOTHESIS |
| Should fail | Red grazing-in / grazing-out bins at trihedral corners if cells stay first-order within budget; red abs metrics if lateral re-randomisation bias is material; anything with g ≠ 0, Fresnel or path-history-dependent BSSRDF | HYPOTHESIS (within fixture); DERIVATION (scope) |
| Inference-time information | Mesh, port frames, σ_t, ω_c, g = 0, index matching, B's rays, a solve per lighting state | FACT of spec |
| Deployable | This fixture: yes. As a general ReST-GI material K: no (WL51 Cycles BSSRDF semantics) | DERIVATION |
| Learnable under sparse supervision | Not applicable; MC used only for validation | DERIVATION |
| Principal bottleneck | Per-lighting resolvent at red ρ ≈ 0.974; pFFT / complement consistency; deterministic grazing-bin quadrature | INFERENCE |
| Closest predecessor | WL77A H3 at θ = 0; WL55/56 MC oracle K; classical CP / FCS / integral-RTE | FACT / INFERENCE |

**Competing explanations for C1 outcomes.**

| Possible observation | Explanation 1 | Explanation 2 | Explanation 3 | Discriminating control |
|---|---|---|---|---|
| Pass on all strata | The target K is computable; the learned-family failures concerned learning | MC references share a convention with C1 that differs from physics (both wrong the same way) | Strata too lenient (BND_FAR under-sampled) | Step-0 H-function and single-scatter tally; the prospective, never-referenced WL79/80 population scored after freezing |
| BND_FAR miss, error still decreasing with refinement | Discretisation not converged within budget (affordability) | — | — | Separate lateral and depth refinement; matrix-free W to reach a further level; Richardson diagnostic only |
| Miss with observed order < 0.5 | Contract or convention mismatch (bin weighting, source definition) | Reference bias | Lateral flat-source bias not reduced by depth refinement | Paired exact vs bin-averaged F_in; CCA-rest MC-only locality control; refinement along each axis separately |
| con_* passes, abs_* red fails | Lateral re-randomisation bias (common to bins) | Far-field mass error moved into Φout | — | Lateral-only refinement; clamp binding fraction; 1ᵀW audit |
| Dent signed change fails, rest passes | Layer compression under the stretch map breaks bias correlation | Dent-cavity visibility or chord termination error | Influence set too small (D_T ≠ D_G) | Rest-mesh-on-rest vs stretched-mesh-on-rest null test; full rebuild vs local rebuild |
| Passivity breach | Far-field overshoot or clamp | Truncated Krylov or non-sub-stochastic coarse correction | — | Per-column audit at the ω_c·d bound with a converged plain iteration |

### E-C2: SBG (amended, deferred)

| Item | Content | Status |
|---|---|---|
| Historical failure addressed | Source-transfer failure; WL77A H2 under-identification and sub-mfp projection; FIA's BND_FAR degradation despite a source-blind rule | FACT (history); INFERENCE (FIA attribution) |
| Mechanism | Exact flights carry all source and angle dependence. The learned part is a bounded log-ratio τ supervised densely in chart space. D2 passivity | DERIVATION |
| Falsifier | Learned charts > 1.25× the exact chart pipeline on any stratum; τ_dent misses the signed-change gates on a held-out pose; apply plus rebuild not better than C1-in-loop or the re-render at equal accuracy | Developer's own rule |
| Should succeed / fail | Succeed: rest box, radially symmetric in-range dents. Fail: dent near a side, non-radial dents, bends, out-of-scope media | HYPOTHESIS / DERIVATION |
| Bottleneck | Non-translation-invariant dent kernel (H-matrix rank unproven); tens of GPU-h of training MC with interior tallies | ESTIMATE |
| Predecessor | WL77A H2 RBG; WL76 FIA; Element Learning; Neural Green's Functions; DeepRTE | FACT / INFERENCE |

**Competing explanations for C2 outcomes.**

| Possible observation | Explanation 1 | Explanation 2 | Discriminating control |
|---|---|---|---|
| Matches C1 on the box | Learned compression is lossless | Charts are a tabulation (enumeration), with no generalisation shown | Held-out chart parameters; cost comparison against exact reduced solves |
| Blue energy error after D2 | D2 under-constrains when (1−ω)·1ᵀG dominates | τ fit error | Adjoint w* recalibration vs D2 (novelty search item 3) |
| τ_dent fails a held-out dent | Shape-space coverage (the same failure class as WL68) | Chart-blend error | Dense dent-family sweep vs a local deterministic re-solve |

### E-BGC-R (abbreviated)

- **Falsifier.** Two instances with equal fits differ by O(1) on out-of-orbit boundary contexts. This has already happened in this family (FACT: WL68 l.295; WL78).
- **Competing explanations of any future success.** Generator inductive bias versus near-orbit evaluation contexts. These are discriminated only by a prospectively chosen out-of-orbit population.

## 10. Fact versus untested expectation

**FACT.**
- Seal integrity.
- All cited worklog lines.
- The brief's measured outcomes.
- WL77A's prior rejection of H2/H3 and its open contract question.
- The FNTO contract wording.
- Nothing in WL80A was implemented or measured.

**DERIVATION (checked by at least two independent parties, including the panel).**
- The C1 energy identity.
- The R1-1 breach.
- The equivalence of Track A's E_sym and Track B's flight–resolvent–flight sandwich.
- The D2 single-constraint limitation.
- The C2 product-composition bias.
- Slab-mode lengths of 3.5 / 3.1 / 1.6 mm.

**Untested expectations.** Every accuracy, cost and signed-change statement about any candidate.


---

# Independent panel reviews

## Panel reviewer: physics-representation

# WL80A cross-review: physics and representation

Independent reviewer. Thought experiment only: I implemented, trained, executed and measured nothing, and I wrote no files.

Candidates in scope:
- **BGC-0**: the original BGC-Transport.
- **BGC-R**: the Track A record-only corrections C1–C5. Track A did not adopt them as an architecture.
- **CP-K**: Track B C1, the selected primary.
- **SBG**: Track B C2, the deferred learned variant.

Labels used throughout: FACT, DERIVATION, INFERENCE, HYPOTHESIS, SPECULATION.

## 0. Evidence checked

**Sealed records (read only).**
- Read in full: `neutral_problem_brief.md`, `trackA_original_bgc.md`, `trackA_decision_log.md`, `trackA_revised_architecture.md`, `trackB_candidate_records.md` and `trackB_selected_candidate.md`.
- Read in part:
  - `trackA_critical_audit.md`, lines 1–741 of 1394.
  - `trackB_adversarial_review.md`, lines 1–1006 of 1819: C1 and C2 reviews and responses in full, and part of the first C3 review.
- Not opened: `trackB_specialist_reports.md` and the C4 review section. My C4 remarks rest on the candidate record and the chair's matrix only.

**Repository anchors re-checked (FACT).**
- WL75 l.195: BND4_FAR con_coarse is 0.159 for the G2 oracle and 0.803 / 0.824 for A75.
- WL75 l.308: "Global angular rank is not the bottleneck".
- WL78 l.119: 5.5 / 3.2 / 0.1 % of interior collided energy lies outside the 128-NN set.
- WL78 l.275: Q78 BND_FAR is 0.668 / 0.860.
- WL76 l.31–43: the exact factorisation K = (1−β)·E_{p1}[Q], Q = albedo·(E1+M), Σ K ≤ 1−β.
- WL76 l.98–110 and l.270–305: the FIA design. FIA used 4-depth port-column anchors at 2.5 mm, three ring lobes and q = 1024. Its ring-lobe residual was 0.45 against a noise floor of 0.002. WL76 itself calls the ring-lobe restriction "my design choice … not the physics".
- WL77A l.77–81 and l.114–115: at q = 1024 the ring floor is 0.45. Noise-free it is 0.21–0.22 for ring lobes and 0.07–0.08 for FIA-AZ. A "large, plausibly dominant part" of FIA's failure is q = 1024 single-scatter noise.
- WL77A l.53–56 and l.132: three candidates were rejected.
  - H2 "RBG": exact first-collision / escape projections with a learned scalar resolvent.
  - H3 "LCR": a learned closure followed by an exact collision-probability solve.
- `docs/plans/wl77a/coordinator_decision.md` l.13–15 and l.100.
  - H2 was rejected because projection onto 2.5 mm hats (h > mfp) loses the incidence footprint, and for its near-diagonal identification, cost and novelty.
  - H3 was rejected because "transfer is carried by a published classical CPM … solve; learned closure likely vacuous; admissibility as a learned FNTO K is an untaken human decision; g = 0 only".
- WL79 l.273: the boundary sources sit 1.8–4.0 mm from the nearest perpendicular face, matching the brief.

**Derivations I redid by hand** (all DERIVATION; all agree with the tracks unless stated otherwise):

| Item | Result |
|---|---|
| Collision-density (Peierls) energy identity, original form | ψ = q + ωWψ, with a per-cell partition 1ᵀW + 1ᵀΦ = 1, gives output = 1ᵀq − (1−ω)1ᵀψ ≤ ω·d |
| Same identity, amended CP-K split | Per-point partition for Φ_pt and W1 plus a per-cell complement escape give output = ω1ᵀq − (1−ω)1ᵀψ ≤ ω·d, for any h and any accuracy of W |
| Track A's symmetric form E_sym | Uncollided escape from y into (j,o) is m_j·d(j,ō)·p1(y\|j,ō)/(4πσ_t). Hence m_i K_ij = (ω/4πσ_t)(m d)_i (m d)_j ⟨p1_j, (I−ωT)⁻¹ p1_i⟩ with T symmetric. Correct. **It is the same operator as Track B's flight–resolvent–flight sandwich.** |
| Escape after first collision for a beam with direction cosine μ | ∫(e^{−z/μ}/μ)·E₂(z)/2 dz = ½[1 − μ ln(1+1/μ)]: 0.477 / 0.424 / 0.380 at μ = 0.01 / 0.05 / 0.1. The flat 0.5-mfp-cell average is (½ − E₃(0.5))/(2h) = 0.278. The R1-1 passivity breach of the original split form is confirmed. |
| Second-order moment closure (Track A CE-3) | e^{−r}(1 + r²/2) against 1/(1+r): +10.4 / +21.8 / −17.6 % at r = 1 / 2 / 4 |
| Screening lengths | κ = √((π/H_e)² + σ_a/D) with H_e = 11.33 mm gives 3.54 / 3.06 / 1.63 mm. Infinite-medium lengths are 18.3 / 5.8 / 1.8 mm. Beyond 15 mm, a K₁ estimate of the red slab tail gives about 4 %, consistent with the measured 5.5 % (FACT, WL78). |
| Fundamental slab mode | λ ≈ 0.975, gains 38 / 29 / 8, about 260 Neumann sweeps to 1e-3 for red |
| SBG D2 identity | uᵀ(I−ωW)⁻¹ = 1ᵀ with u = (1−ω) + ωP_esc. Any Ĝ ≥ 0 with Σ_y u(y)Ĝ(y,x) ≤ 1 − P_esc(x) gives column output ≤ ω·d, provided Φout's escape total equals P_esc of the same W. |
| Midway-thinning counterexample | Track A CE-5 and Track B C2-R2 #3 derived it independently. Thinning to 4 mm gives π/5.33 = 0.59 mm⁻¹ against 0.28 mm⁻¹, so red transfer falls by about e^{−1.24} ≈ 0.29× over 4 mm. Track A places the dent on the underside while the brief's dents are on top; by the top/bottom flip this is the same case. |

## 1. Physics dimension

### 1.1 Comparison matrix (rationale follows)

| Property | BGC-0 | BGC-R (C2 operator) | CP-K (amended) | SBG (amended) |
|---|---|---|---|---|
| Exact transport semantics | None. dρZ/(ε+ΣZ) is a parametrisation, not a transport law. | None; z is a free density. | Exact Peierls form for g = 0, homogeneous, index-matched media. Error is discretisation only. | Exact flights, single scatter and single flight. Multi-flight resolvent is an exact base kernel × learned τ ∈ [0,1]. |
| Energy bound | ≤ d·ρ ≤ d: meets the brief, but not the physical ω_c·d | ≤ ω_c·d by projection | = ω·d − (1−ω)1ᵀψ ≤ ω·d by identity at convergence | ≤ ω·d by the D2 identity |
| Positivity | Yes | Yes | Yes for the exact factors. Conditional for the precorrected-FFT far field (see 1.3). | Yes in the chart-sum form |
| Linearity / zero in, zero out | Yes (z depends on geometry only) | Yes | At convergence or a fixed linear iteration; not truncated Krylov | Yes, structurally |
| Edit locality | Violated: receiver-uniform factor (ρ′/ρ)(S/S′) (Track A CE-2) | Local while S ≤ 1; the coupling returns where the projection binds | Physical by construction (δK ∝ ψψ† emerges from the solve) | Physical up to τ_dent accuracy |
| Reciprocity | Impossible with a free ρ | Needs d_i z_ij symmetric plus a 16↔64 map; not built in | Volume-weighted reciprocity of W traded away for an exact partition (conceded) | Symmetric τ, but per-column D2 scaling breaks symmetry |
| Directional correspondence | Learned; frame gauge and dented frames must be learned (CE-7, WL79) | Same | Every bin is an integral in the contract's own frames, so the frame convention, including dented first-triangle frames, is exact | Same as CP-K for the flights |
| Domain ownership (FNTO K/C) | Unspecified | χ_D mask, which is trivial on the single slab | A chord that leaves the medium, including into the dent cavity, terminates; re-entry belongs to C | Same, after the visibility fix (D1) |

**BGC-0 (DERIVATION).**
- The physics objections in Track A all stand under my re-derivation: ρ is redundant with ΣK, the bound d is loose relative to ω·d, the measure and partition are missing, and the edit factor is receiver-uniform.
- One nuance: BGC-0 does satisfy the *brief's* passivity contract (Σ ≤ d). What it violates is the tighter physical bound ω_c·d. For blue (ω = 0.9) that permits up to 11 % excess over physics.

**BGC-R (DERIVATION).**
- The corrected operator is physically valid: linear, positive, passive below ω·d, and dimensionally consistent.
- It encodes no transport law, however. Validity comes entirely from the projection. Below the bound, every K value is whatever the learned z gives.

**CP-K (DERIVATION).**
- It is the only in-scope candidate whose physical invariants hold independently of anything learned or fitted.
- The amended split is necessary. I confirmed the R1-1 breach numerically, and the amended identity closes it.

**SBG (DERIVATION).**
- After its developer's amendments (visibility only inside the single-flight term; a monotone nested ratio; the D2 projection), the physical guarantees match CP-K's.
- The exception is entrywise accuracy: D2 fixes column totals, not the spatial or angular distribution that con_* measures. Track B's review records this, and I agree.

### 1.2 Agreements with the sealed verdicts (physics)

- **Both tracks derived the same exact operator**, independently (Track A's E0/E_sym; Track B's Peierls sandwich). I verified both and they are the same object (DERIVATION). Every incoming-bin dependence enters only through F_in, the first-collision density. Track B's "all of con_* enters through F_in" is therefore correct (DERIVATION).
- **Track A's ω_c·d bound is correct.** CP-K's own identity gives ≤ ω·d. **Disagreement with Track B on a detail:** the CP-K falsifier 3(b) tests ΣK > d(1+1e-6). That is looser than the physics the operator claims to satisfy. Because the identity gives ω_c·d exactly at convergence, the falsifier should test against ω_c·d. As written, a two-grid overshoot of up to (1−ω_c)·d, i.e. 10 % of d for blue, would go undetected (DERIVATION).
- **Track A's checker on the BGC-R reciprocity condition** (d(i,a)z(ia,jo) = d(j,ō)z(jō,iā)) is right; I re-derived it from E_sym.
- **d and B must share one β quadrature.** The two tracks say this independently: Track A checker E-b, and Track B's "same rays as B". I agree.
- **Visibility belongs inside each flight, not on the resolvent.** This is Track B's C2 D1, and it is implicit in Track A's 𝒢 = δ + ωT + (ωT)² + …, where each T carries in-medium visibility. Agreed (DERIVATION).

### 1.3 Disagreements and additions (physics)

1. **Reciprocity across the 16-in / 64-out bin sets: I side with Track A.**
   - Track B's chair rejects reciprocity symmetrisation for C3/C4 as "ill-posed, because the reverse of an outgoing bin is not an incoming bin".
   - Direction reversal in a port frame maps (x, y) on the disk to (−x, −y). That is a 180° in-plane rotation, a D4 element, so it permutes the concentric bins.
   - The brief defines fine16 as a grouping of the 64 outgoing bins into 16. Assuming the groups nest onto the 16 incoming bins (INFERENCE; I did not inspect the bin code), reciprocity is well posed on 16×16 aggregated blocks. It is not well posed entrywise at 16×64.
   - This matters as a *diagnostic* constraint for CP-K, whose stored operator has sacrificed exact reciprocity. It is not a requirement of the brief.
2. **CP-K's positivity is conditional on the far-field scheme (INFERENCE).**
   - Precorrected FFT projects onto and interpolates from a uniform grid, then subtracts the grid's near interactions. With non-positive interpolation stencils, or that subtraction, far-field W entries can come out negative.
   - Track B's chair flagged consistency with the complement-defined escape, but not the sign. "All factors ≥ 0" is structural only with positive projection and interpolation weights, or a positivity-preserving near/far split.
3. **CP-K's "clamp the escape to 0 and renormalise" is a form of energy clamp.** Track A applied the AGENTS.md "predeclare, never post hoc" condition to its own max(1, ·). The same procedural condition should bind CP-K's clamp. Track B says only "report"; I think "predeclare and report the binding fraction" is required (INFERENCE from project rules, FACT that AGENTS.md excludes post hoc energy clamps).
4. **Scope.** Both tracks state, and I confirm, that every physics guarantee of CP-K and SBG is specific to isotropic phase, homogeneous, index-matched media (DERIVATION; FACT that WL76 l.37–41 lists the same exclusions). BGC-0 and BGC-R are not restricted to this scope by construction, but they guarantee nothing physical beyond the projection.

## 2. Representation dimension

### 2.1 Comparison matrix

| Property | BGC-0 | BGC-R | CP-K | SBG |
|---|---|---|---|---|
| Information sufficiency of the source encoding | Insufficient: moments, source side only (CE-3, CE-4) | Moments kept but labelled insufficient; receiver statistics added | Sufficient: F_in exact at points, first continuation W1 point-exact | Sufficient: same flights, plus exact single and double scatter |
| Source-independent behaviour | No; the source enters a learned map | No | Yes by construction: nothing is fitted per source | Yes for sources; learning moves to chart space |
| Boundary-context preservation | Endpoint context only: CE-5 / CE-6 insufficiency | Clipped context plus an unspecified corridor descriptor | Exact: the true geometry enters every flight | Slab and semi-slab exact; quarter-slab and dent learned |
| Geometry conditioning under deformation | No guarantee (Track A Q10) | Partial (corridor) | Body-fitted mesh, ray-integrated flights | Pair-frame dent descriptors (~7-D); weakest link (conceded) |
| Angular expressivity | Pair-conditioned R ≥ 16 spans the full cone | Unspecified | Exact; no angular basis | Exact flights |
| Spatial expressivity | Unbounded but unidentified | Same | Resolution-limited (see 2.3) | Chart-kernel compression plus partition-of-unity blending |
| Structural error floor | Identification floor (CE-1, generic); unmeasured rank floor | Same; no correction adds identification | None as h → 0. A real floor exists at affordable h (2.3). | τ fit error, chart blending, dent family coverage |

**BGC-0 and BGC-R (DERIVATION plus FACT).**
- I agree with Track A's rejection.
- The decisive representation fact is that the learned spatial geometry→T map is where transfer failed (FACT, WL75 l.308; WL68). Both BGC forms keep that map, and BGC-0 adds pair-conditioned angular capacity to it.
- One nuance on Track A's "a perfect angular family can remove at most about 0.159 of the ~0.80 BND4_FAR error". This is INFERENCE, not a strict bound: errors from the angular and spatial parts do not combine additively. The direction of the argument is sound.

**CP-K: the source-generalisation argument is correct (DERIVATION).**
- There is no free function on which Track A's bump construction (CE-1) can act. That construction is generic to finitely supervised *learned* operators, as Track A's own checker states.
- CP-K's error therefore does not depend on which sources were supervised.
- It *does* depend on source context: discretisation error concentrates at edges, corners and grazing bins. "Transfer" thus becomes uniform quality under convergence, which Track B states correctly as a HYPOTHESIS.

**SBG (DERIVATION / INFERENCE).**
- On the rest box, its chart kernels can be computed exactly (D3), so the learned τ adds compression, not information. That is Track B's conclusion, and I agree.
- One addition: the partition-of-unity blend Σ_k ½(G_k D_k + D_k G_k) is not an exact composition where x and y fall in different charts. So "exact on the box" holds only for the individual chart kernels, not for the blended Ĝ (INFERENCE; magnitude unmeasured).
- For dents, the CE-5 midway-thinning case forces pair-frame descriptors. Track B's amendment does this, which turns τ_dent into a learned map over a 7-D-plus family: the same coverage question the brief records failing in source space, now in shape space (INFERENCE).

### 2.2 Cross-track and historical findings that neither sealed track records

1. **CP-K and SBG are, in physics, WL77A H3 and H2 re-derived blind (FACT for the WL77A records; INFERENCE for the mapping).**
   - Amended CP-K is H3's collision-probability solve without the learned closure. Its original optional ΔW was H3's closure, and it is now dropped.
   - Amended SBG is H2's "exact projections × learned scalar resolvent" (Πᵀ Γ Π).
   - H2's decisive WL77A objection was projection onto 2.5 mm hats (h > mfp). SBG's amendments (graded depth, exact single and double scatter) address it directly.
   - H3's objections were "transfer carried by classical CPM", "learned part vacuous", an admissibility decision still open (WL77A coordinator open question 3, FACT), and g = 0 scope. CP-K's developer concedes every one of them.
   - Track B, being blind, could not know this. Track A, which read WL77A, did not connect R-a or R-b to H2/H3.
   - **Consequence:** CP-K adds no new evidence on the WL77A contract question. Whether a non-learned or solver-based K is admissible remains the same untaken human decision.
2. **Track A's dismissal of R-a (exact first-collision anchoring with a source-independent Q) rests on a stale limit (FACT).**
   - Track A's revised record cites R-a's "recorded limit" as the 0.45 ring-lobe floor.
   - WL76 itself calls ring lobes a design choice, not physics. WL77A corrects the floor to 0.21–0.22 for ring lobes and 0.07–0.08 for FIA-AZ once noise is removed.
   - CP-K is R-a with Q *computed*: the exit angular structure comes from the exact escape Φ applied to a resolved collision density. That removes both the ring restriction and the q = 1024 noise by construction (DERIVATION).
   - FIA's other bundled limitation, lateral snapping of the second collision to 2.5 mm port-column anchors, was never attributed (FACT, WL76 l.273, l.305). It is the same class of issue as item 2.3 below.
   - This does not change Track A's verdict on BGC. It means Track A's record understates R-a.
3. **Track B's inference that FIA's "learned angular redistribution" was trying to learn the exact escape geometry is essentially right (INFERENCE, consistent with WL76 l.270–271 and WL77A).** Under isotropic scattering, the outgoing angular structure at a receiver is fixed by Φ applied to the true ψ field.

### 2.3 New representation finding: lateral grading of amended CP-K

In this section the calculation is DERIVATION; magnitudes and gate impact are INFERENCE or HYPOTHESIS.

**The tension.** Amended CP-K grades depth finely (0.05 mm surface layers) but keeps **lateral spacing at about 1.25 mm, i.e. 1.25 mfp**. Under the R1-2 objection, the developer said the 1.5 % re-randomisation bias at 0.5 mm "falls under the grading below". That holds for depth only. Laterally the amended mesh is *coarser* than the withdrawn uniform 0.5 mm option.

**Size of the bias.**
- A flat-source CP re-randomises each emission within its cell.
- Per lateral axis, a flight has ⟨x²⟩ = 2ℓ²/3. Each re-randomisation adds roughly h²/12 to h²/6 of variance, depending on how the landing offset correlates with the flight.
- The relative inflation of the lateral diffusion coefficient is therefore about h²/8 to h²/4: about 0.2–0.4 at h = 1.25 mm, against 0.03 at 0.5 mm.

**Effect.**
- In the leakage-dominated slab, κ_lat² ≈ (D_z/D_lat)·k_z². The red lateral decay length (~3.5 mm, and ~1.7 mm in the 4 mm thinned dent region) would be biased long by roughly 9–18 %.
- Near 15 mm (≈ 4 decay lengths) the red tail would be over-predicted by tens of percent (SPECULATION on the exact factor).
- The bias is nearly common to all incoming bins, so it should largely cancel in con_* (INFERENCE). It would load abs_coarse and abs_fine16 for red, and the dent signed change, because the decay length changes between rest and dent.
- Classical flat-source CP needs optically thin regions. Track B's own review notes this as prior work.

**Affordability of the refinement study (INFERENCE from Track B's own estimates).**
- The base level is about 2–3e5 cells with about 1e8 near-field non-zeros (0.7 GB).
- Halving all sizes multiplies cells by 8. If the explicit near field keeps a fixed physical radius (3 mfp), it multiplies non-zeros by 64. Even a fixed cell-unit stencil multiplies them by 8 per level.
- So the third, and possibly the second, predeclared level is unlikely to fit 11 GB unless W is applied matrix-free.
- The amended falsifier still yields a decisive verdict ("fail at the finest affordable level", no excuse for a still-decreasing error). But a fail may then reflect the affordable resolution rather than the factorisation. The optional CCA-rest control and the paired exact versus bin-averaged F_in control become the main ways to tell the two apart.
- I suggest predeclaring the lateral spacing at each level explicitly and reporting the lateral and depth refinements separately.

### 2.4 Rejected candidates (only where my dimensions bear on them)

**C4 NSM.** Truncating the eigen-expansion of a non-negative kernel leaves only the Perron mode guaranteed non-negative. The rejection on non-negativity is therefore correct (DERIVATION).

**C3 CCA.**
- As a representation it is exact at rest only up to MC noise and the equivalence cutoff.
- That cutoff was measured on one pair, on the near field and fine16 contrast only (FACT, brief §3a).
- My slab-mode calculation supports the reviewers' leakage-dominated red length of about 3.6 mm over the candidate's 18 mm (DERIVATION, diffusion; crude within about 1 mfp of the boundary). That strengthens CCA-rest as a diagnostic.
- Its rejection *as an answer to the central question* stands, because it is enumeration. I see nothing in my dimensions that overturns either rejection.

## 3. Verdict alignment

**Track A, BGC-0 rejected: AGREE.**
- Physics: the ρ redundancy, the d versus ω·d bound and the edit artefact are all re-derived.
- Representation: BGC-0 keeps the learned geometry→T map with no structure that shares information across sources.

**Track A, BGC-R (record only): AGREE that it is valid but not better for transfer.**
- Correction requested: R-a's limit should be stated with the WL77A noise-free values and the WL76 "design choice" qualification.
- The record should also note that R-a with an explicitly computed Q is CP-K.

**Track B, CP-K selected primary: AGREE on physics and source-independence. PARTLY DISAGREE on representation readiness.**
- The physics core, the amended passivity identity and the source-agnostic argument hold (DERIVATION).
- Three amendments are missing:
  - (a) the falsifier should test ω_c·d;
  - (b) positivity of the far field, and predeclaration of the clamp;
  - (c) the lateral 1.25 mfp re-randomisation bias and the affordability of the refinement levels (2.3).
- The selection's novelty and admissibility status is that of WL77A H3. The contract decision WL77A left open is still open; CP-K does not resolve it.

**Track B, SBG deferred: AGREE.**
- Physically valid after the amendments.
- On the box its learned part is compression: at most an amortiser for dent families. There it inherits a shape-space coverage problem of the same kind the brief's learned operators already failed.

## 4. Unmeasured (none of this is established by either track or by this review)

- CP-K's gate accuracy on the BND_FAR strata at affordable grading.
- The size of the lateral and depth flat-source biases.
- The precorrected-FFT sign and closure behaviour.
- Dent signed-change bias cancellation.
- SBG's chart-blend error and τ_dent coverage.
- The per-pair nonnegative rank (relevant only to the BGC family).
- Whether the archived packets carry collision-order tallies, which decides whether the step-0(b) convention gate needs new MC.

No implementation, run or successor follows from this review.

---

## Panel reviewer: generalisation-learnability

# WL80A cross-review: generalisation and learnability, with Part E

Reviewer: independent cross-reviewer. I developed none of the candidates. No file was written. Nothing was implemented, trained, run or measured.

Labels: **FACT** (sealed record or worklog line, re-read this session), **DERIVATION**, **INFERENCE**, **HYPOTHESIS**, **SPECULATION**.

## 0. What I checked

**Seal.**
- The sha256 of all ten sealed files matches `seal.json` (FACT, `sha256sum` this session).
- I did not modify any sealed record.

**Sealed records read.**
- `neutral_problem_brief.md` and `trackA_original_bgc.md`.
- `trackA_revised_architecture.md` in full.
- `trackA_critical_audit.md`: chair synthesis, CE-1 to CE-9 and the objection matrix.
- `trackB_blind_brief.md` and `trackB_selected_candidate.md` in full.
- `trackB_candidate_records.md`: synthesis, C1 and C2.
- `trackB_adversarial_review.md`: C1 and C2 reviews and the developer responses.
- I skimmed the C3/C4 material only through the Track B rejection record.

**Historical anchors checked read-only.**

| Source | What it says (FACT) |
|---|---|
| WL75 l.195 / l.308 | BND4_FAR con_coarse: G2 oracle 0.159, A75 0.803 / 0.824. "Global angular rank is not the bottleneck … the spatial geometry → T mapping is." |
| WL78 l.119–120, l.275 | Interior red energy outside 128-NN is 5.5 %. Q78 BND_FAR is 0.668 / 0.860. |
| WL76 l.98–112 | FIA: SS plus second-collision anchors W, with a learned, source-independent emitter response R_θ. |
| WL76 l.237 | Development BND_FAR con_coarse 0.96–0.99 and **abs_coarse 0.39–0.42**, against roughly 0.13 on training sources. |
| WL76 l.263 | BND_FAR drift 0.76 → 0.96 during training. |
| WL77A worklog and `docs/plans/wl77a/coordinator_decision.md`, `adjudication_draft.md` | H2 RBG = exact first-collision projection plus a learned scalar resolvent. H3 LCR = learned closure plus a collision-probability solve. Both rejected. H3 because "transfer comes from a classical solver" and its "admissibility as a learned K is an untaken human decision". The open item reads: "whether the learning-free θ = 0 CPM operator meets the WL59 gates". The q = 1024 SS-noise re-attribution is in the coordinator's C1/C2 checks. |
| WL77A A-A15 | The first-interaction decomposition was validated in energy and coarse space only. |
| WL73 l.145–150 | S16-R boundary sources are TBE (edge row), TBC (slab vertex), SM (side mid) and SR (side rim). |
| WL79 l.273 | Boundary sources lie 1.8–4.0 mm from the nearest perpendicular face. |
| WL68 l.295 | Three all-group fits with near-identical fit scores differ at new ports: contrast .94 / 1.22 / 1.36–1.44. |
| FNTO contract l.17 | "do not replace learned complex shading with a limited explicit radiosity solver" |
| FNTO contract l.86 | "not an exhaustive explicit per-surface full path tracer hidden under a neural name" |
| `BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md` | Mandatory baseline gate. |

**Literature.** I did no new web search. All literature identities below are as cited in the sealed records or in WL77A. I did not re-verify them.

**History the blind track could not see** (it is material to my dimensions):
- Track B's **C1 is WL77A's H3 with the learned closure removed (θ = 0)**, plus a corrected first continuation. Its proposed falsification run is the question WL77A left open (FACT + DERIVATION from comparing the specs).
- Track B's **C2 is a repaired relative of WL77A's H2** and of **WL76 FIA** (INFERENCE from comparing the specs).

## 1. The six claim levels, kept separate

| Level | Original BGC | Track A revised BGC (record-only, not an architecture) | C1 CP-K (amended) | C2 SBG (amended, deferred) |
|---|---|---|---|---|
| **Exact physical factorisation** | None. Its outer form can represent any passive K, so it is a parametrisation, not a factorisation (DERIVATION, Track A Q1). | None added. C2-corrected passivity, linearity and positivity hold (DERIVATION, Track A checker). | Yes, under g = 0, homogeneous, index-matched: flight–scalar resolvent–flight (DERIVATION, re-derived independently by both tracks and by WL77A). | Same factorisation as C1. The approximation is confined to the multi-flight resolvent (DERIVATION). |
| **Approximate learned representation** | Pair-conditioned (w, u, v) generator from encoders. | Unchanged in kind; A3 removed. | None. | log τ ratios on chart charts, plus τ_dent on a dent family. |
| **Synthetic operator correctness** | Passive only per input, and only with the corrected bound ω·d (DERIVATION). | Passive while S ≤ 1; the max(1, S) projection behaves like a clamp where it binds (DERIVATION). | Passivity identity exact at convergence for any h (DERIVATION, re-verified by the Track B chair). Overshoot and the reciprocity defect still have to be measured. | Passive through the D2 per-column identity, given a C1-type discrete W (DERIVATION). Column scaling breaks reciprocity (DERIVATION). |
| **Supervised fitting** | Expected to fit, because the family always has (FACT, brief §3b). | Same. | Not applicable. | Fits charts. It does not fit source packets; option (c) was dropped. |
| **Independent source transfer** | No mechanism (DERIVATION, CE-1). | No mechanism added (Track A's own verdict). | Structural, conditional on discretisation convergence in boundary layers (DERIVATION + HYPOTHESIS). | Structural on the box if chart coverage is exhaustive. For dents, interpolation within the family (HYPOTHESIS). |
| **Realistic workload viability** | Unresolved: cache/apply roughly 81× ATO per pair (Track A derivation). | Unresolved. | Probably not cheaper than path-traced re-render under deformation (INFERENCE, conceded by its developer). | Unproven; the dent-region compression is open (conceded). |

## 2. Generalisation

### 2.1 Exact symmetry behaviour

**Original and revised BGC.**
- D4h holds only through group averaging at rest (FACT that the rest maps are exact, WL79).
- Undefined on dented poses (FACT, WL79: non-bijective maps).
- Not invariant to the per-port frame gauge (DERIVATION, CE-7).
- Averaging is an L2 projection toward an equivariant truth, so it cannot hurt. But it adds **no information** at out-of-orbit contexts (DERIVATION).
- WL79/80 show that even a refitted basis at rest leaves 1.15–1.27× orbit gaps (FACT). Any learned angular factor inherits the same exposure unless equivariance is structural.

**C1.**
- Exact at rest by permutation assembly from orbit representatives (DERIVATION).
- More important, and under-stated by Track B: C1 **does not need** symmetry to generalise. It computes the contract's bins in whatever per-port frame the contract defines. So the dented-pose frame problem (WL79) and the 45° gauge (CE-7) do not apply to it (DERIVATION).
- Its symmetry behaviour is a *consequence* of correct physics, not a constraint imposed on a learner. This is the strongest generalisation property of any candidate here.

**C2.**
- Exact at rest only if τ is Reynolds-averaged over the stabiliser and the x↔y swap. The developer amended the design to do this.
- On dents it relies on an assumed radial-symmetric dent family (DERIVATION, from its R2-3 amendment).
- Its angular ends are exact, as in C1. Only the scalar τ must respect symmetry, and that is easier than respecting frame-covariant angular factors (INFERENCE).

**Ranking (written rationale, not a score).** C1 is best: symmetry follows from the physics, and no gauge exposure remains. C2 is next: only a scalar has to be symmetrised, and the dent family carries an assumption. BGC (both forms) is worst: rest only, gauge-exposed, and frame-covariant learned factors.

### 2.2 Beyond training orbits, and unseen boundary combinations

**Correction to Track B's coverage inference.**
- Track B inferred that corners, vertical edges and side rims are "plausibly unsupervised classes" (INFERENCE in its record).
- FACT (WL73 l.145–150): S16-R contains one source each of TBE, TBC (slab vertex), SM and SR.
- The historical failures are therefore **out-of-orbit positions within classes that have a single supervised member**. WL77A A-A1 records that every confirmation TBC slot is out of orbit.
- Whether a vertical-edge (side–side) source is supervised I could not verify.
- This does not affect C1. It sharpens what any learned candidate must achieve: within-class interpolation from **one** member per class. The brief's S16 → S16-R outcome ("nearby transfer ~3×, far small/mixed", FACT) is direct evidence that one member per class is not enough for source-space learners.

**BGC (both forms).**
- Nothing structural reaches out-of-orbit contexts except the generator's inductive bias (DERIVATION, CE-1).
- I add an empirical instance of CE-1 that Track A did not cite. WL68 l.295 (FACT): three all-group fits with near-identical fit scores give new-port contrast of .94 / 1.22 / 1.36–1.44.
- WL78's two seeds give BND_FAR 0.668 / 0.860 (FACT).
- That is exactly the "same labels, different held-out operator" freedom that CE-1 constructs. It has already happened twice in this operator family.

**C1.**
- A new source changes only the exact right-hand side F_in (DERIVATION).
- Unseen boundary combinations (corner × grazing in × grazing out) are reached by the geometry of W1, Φ_pt, W and Φout on the true body (DERIVATION).
- Whether accuracy there meets the gates within budget is the central HYPOTHESIS. Two- and three-face boundary layers at grazing outgoing bins converge only at first order while h > μ_out (DERIVATION, developer).

**C2.**
- Generalisation is coverage by enumeration of chart types on a box (DERIVATION; the developer withdrew the "corner without corner data" claim).
- This is legitimate, but it is the same kind of claim that got C3 rejected. On a box, C2's learned part is compression, not generalisation (FACT of the Track B record; I agree).

**A principle the blind track stated too strongly.** Track B's cross-cutting rule is "learning must never see the source". It is **necessary but not sufficient**.
- FACT, WL76 l.98–112 and l.237:
  - FIA's learned R_θ is source-independent, and its features are invariant under the slab symmetries.
  - It was still trained only through source packets.
  - On development BND_FAR its abs_coarse is 0.39–0.42, against about 0.13 on training sources. Contrast drifted from 0.76 to 0.96 over training.
- WL77A's re-attribution assigns much of FIA's *uniform* error to q = 1024 SS noise. But that noise alone gives coarse error of about 0.07–0.10 (FACT, WL77A coordinator check C2), which does not explain a threefold BND_FAR gap in abs_coarse.
- INFERENCE (not causally isolated): a source-blind learned function still fails where its *own input domain* (anchor contexts near edges and corners) is covered only through the second-collision deposits of supervised sources.
- The correct principle: **the learned function's input domain must be covered by supervision independently of source sampling.**
  - C1 meets it trivially (nothing is learned).
  - C2 meets it only through its amended chart-space supervision.
  - BGC does not meet it.

### 2.3 Distance from supervised sources

| Candidate | What error depends on |
|---|---|
| BGC (both) | Distance in context-feature space from the 13 symmetry-distinct S16-R contexts. The brief's FACT (nearby ~3× better, far mixed) shows this distance governs error for this family. Clipping and tie hygiene (revised C4) remove identity routes, but WL78 shows de-identification fixes interior transfer only (FACT: INT 0.12, BND_FAR 0.67 / 0.86). |
| C1 | No supervised sources, hence no dependence on that distance (DERIVATION). Error depends on **location relative to boundaries and on bin grazing-ness**, through discretisation. A BND_FAR/INT gap could still appear, but it would not be a transfer gap. Track B's criterion (d) should be read that way, and Track B says so. |
| C2 | Distance in chart and dent-parameter space from the supervised chart family. Small on the box (exhaustive). Unknown for dents outside the family. |

### 2.4 Deformation response

**BGC (both).**
- Endpoint features cannot see intervening thinning (DERIVATION, CE-5: about 0.3× red change with every endpoint input fixed).
- The corridor descriptor is unspecified (FACT of Track A's record).
- The historical learned models failed DEF, and WL78 was worse than A75 (FACT).
- With the original independent ρ, a remote edit rescales near receivers uniformly (DERIVATION, CE-2). The max(1, S) projection removes this only while S ≤ 1.

**C1.**
- Deformation is handled by recomputation, not prediction.
  - Templates near the dent are re-integrated on the true mesh, and the resolvent is re-solved globally for every lighting state anyway.
  - So D_T ≠ D_G creates no invalidation-set problem for the resolvent itself (DERIVATION). Only the geometry templates have a dilated rebuild set.
- Signed-change accuracy rests on correlated discretisation bias between rest and dent (INFERENCE). One risk not stated in the record:
  - the vertical stretch map compresses the graded depth layers under a 6 mm dent to about 40 % of their rest thickness;
  - the near-surface layer thickness relative to the mfp therefore changes inside the edit region;
  - so the rest and dented biases are *not* the same function of position there.
  
  First-order cancellation is plausible but not assured (DERIVATION of the geometry; magnitude HYPOTHESIS).
- Bending (WL53: global, high-rank ΔK, FACT) is in principle handled identically, by re-meshing (INFERENCE).

**C2.**
- τ_dent uses pair-frame descriptors and the nested ratio G_dent ≤ G_rest (DERIVATION).
- This generalises only within a radially symmetric dent family, excluding the evaluated pose.
- A dent near a side face, a non-radial dent or a bend is outside the family (DERIVATION of scope).
- This geometry-features → transport-quantity mapping is structurally the family that failed in WL68 ("fails on new ports and on deformation", FACT). C2 differs in that it is supervised densely over its input domain rather than at 16 sources. That is the whole bet (INFERENCE).

**Ranking.** C1 is strongest: deformation is exact physics, with discretisation bias as the risk. C2 is bounded to its family. BGC has no mechanism.

### 2.5 Nonlocal transport

**C1.**
- The full resolvent captures red's 5.5 % beyond 128 ports and the slab mode (λ ≈ 0.975) exactly at convergence (DERIVATION).
- Risk: the precorrected-FFT far field may need clamp-and-renormalise when 1ᵀW > 1 (FACT of the Track B record). That touches the complement-defined escape on which exact passivity rests. It must be predeclared, and its binding fraction reported, exactly like Track A's max(1, S) (DERIVATION).

**C2.**
- The infinite-slab kernel is exact and global in-plane. Side and edge charts reach 15–20 mm.
- The dent-region kernel has no translation structure, and its H-matrix compression is an unproven HYPOTHESIS (conceded).

**BGC (both).**
- The local/far split shares one denominator (revised). The far term is unspecified.
- Where the projection binds, the coupling of unrelated receivers returns (DERIVATION, Track A).

### 2.6 Agreement with the sealed verdicts on generalisation

- **Agree, Track A:** BGC has no source-generalisation mechanism. "No correction adds identification" is correct. I add an empirical instance (WL68, WL78 seed spreads).
- **Partly qualify, Track A's "wrong target" argument.**
  - WL75 l.308 is FACT. But WL79/80 show that the angular basis is not closed under exact boundary rotations (1.15–1.41×, FACT), and WL77A shows that the earlier angular floors were noise-confounded.
  - So the angular factor is secondary rather than non-binding.
  - The conclusion still holds, because A3 also adds per-pair capacity in the map that failed (INFERENCE).
- **Agree, Track B:** C1 is the only in-scope design whose transfer argument is structural.
- **Disagree on framing, Track B:**
  1. "Never sees the source" must be strengthened to "input domain covered independently of sources" (§2.2, WL76 evidence).
  2. The class-coverage inference is partly wrong (§2.2).
  3. C1's generalisation is not new evidence for the project. WL77A already rated H3's transfer as "carried by a classical solver" (FACT). Track B's blind re-derivation is an independent replication of the *reasoning*. It is not a measurement.
- **C3/C4 rejections hold in this dimension** (no own transfer mechanism; C3 is enumeration). I agree CCA-rest is a useful MC-only control of the context-locality premise.

## 3. Learnability

### 3.1 Identifiability

**BGC original.**
- At R ≥ 16 the factors span the whole passive cone. w and u are identified only as a product. ρ is redundant with ΣK (all DERIVATION, Track A).
- Out-of-orbit contexts are unidentified (CE-1).
- Reciprocity pins only entries toward supervised positions, mostly far field (DERIVATION, Track A checker correction).

**BGC revised.** Gauges are removed and ρ/ε are gone. The near-field freedom of CE-1 is untouched (DERIVATION).

**C1.** Nothing to identify. The only free choices are discretisation and quadrature settings. These are hidden hyperparameters (grading, ray counts, far-field cutoff) and must be frozen by **reference-free** convergence criteria before any MC score is read. Otherwise they become tuning on held-out data (INFERENCE; the procedural point is not in Track B's falsifier).

**C2.**
- τ is identified pointwise by dense chart supervision: exact reduced solves for slab and semi-slab, MC point-source collision-density fields for edge and dent (DERIVATION).
- This removes WL77A H2's decisive objection, a near-diagonal block "identified by smoothness only" from source packets (FACT, WL77A A-A7) (DERIVATION).

### 3.2 Required source coverage

| Candidate | Requirement |
|---|---|
| BGC | Dense coverage of source *contexts*. Sixteen sources (13 distinct) evidently do not suffice for this family (FACT, brief §3b). The brief notes the high per-packet cost (FACT: about 4 GPU-min per source and geometry). |
| C1 | Zero for construction. MC packets are needed only for validation, and any population can be used without leakage provided settings are frozen first (DERIVATION + INFERENCE). |
| C2 | Zero source packets. Instead roughly 10³–10⁴ MC point-source fields for edge and dent families, tens of GPU-h (ESTIMATE, conceded). This moves the coverage burden to a cheaper and denser space. It does not remove it. |

### 3.3 Privileged supervision

**C1.** It needs exact material parameters (σ_t, ω_c, g = 0) and the index-matched semantics at **inference** (FACT of its scope).
- Deployable for this fixture.
- Not deployable for the project's path-history-dependent Cycles BSSRDF materials (FACT, WL51), for unknown materials, for g ≠ 0 or for Fresnel boundaries (DERIVATION; WL77A A-A9 made the same point for H2/H3).

**C2.**
- Same inference-time requirement as C1.
- At training time it needs **interior collision-density tallies**. These are a privileged intermediate that the reference packet contract does not contain, so new MC instrumentation is required (FACT of the record; deployability unaffected).

**BGC.**
- Training uses only packets.
- Inference uses geometry plus first-flight statistics. Track A amended this so that d shares the analytic β with B (FACT of the record).

### 3.4 Memorisation risk

**BGC.** High. Per-pair capacity adds unknowns. Late-training transfer degradation is FACT for this family (brief §3b; WL71A; WL76 drift). Track A's INFERENCE that A3 amplifies it is sound. Removing A3 removes only the extra capacity, not the family risk.

**C1.** None (DERIVATION).

**C2.**
- Low on the box: low-dimensional regression of a smooth scalar against dense targets (INFERENCE).
- Moderate for dents: the input has about 7+ dimensions, the targets are thousands of noisy MC fields, and the evaluated pose is held out (INFERENCE).
- Dropping fine-tuning on S16-R packets (option (c)) is correct. It would re-import the measured drift regime (FACT, brief §3b).

### 3.5 Optimisation difficulty

**BGC.** A nonconvex generator under a normalising denominator and a non-smooth projection. Two seeds already diverge in held-out behaviour in this family (FACT, WL68, WL78).

**C1.** No optimisation. It does need a linear solve at spectral radius about 0.974 for red, which takes about 260–434 plain Neumann iterations (DERIVATION; WL77A A-A13 gives 434 to 1e-5). Diffusion-synthetic acceleration (DSA) is fragile at h ≈ 0.5 mfp (INFERENCE). Exact linearity requires convergence or a fixed iteration, not truncated Krylov (DERIVATION).

**C2.**
- τ spans decades in red, so it must be learned in log space with an energy-weighted loss (DERIVATION).
- The nested constraints and the D2 projection are geometry-only and linear (DERIVATION).
- Moderate difficulty, and well-posed.

### 3.6 Ambiguity

**BGC.**
- The decoder dichotomy is unresolved: linear means higher-rank ATO, and nonlinear means about 81× apply cost (DERIVATION, Track A).
- The rank per pair class is unmeasured.

**C1.** Two ambiguities:
- the sign of the flat-Φout grazing bias (unresolved between developer and reviewer);
- the brief's "1.8–4 mm" source-distance definition.

On the second: WL79 l.273 confirms the 1.8–4.0 mm figure as FACT, but I could not reconcile it with port-centre distances (1.25 + 2.5k mm). It probably reflects a different reference point. The definition must be fixed before C1's step 0 (FACT that it is unresolved).

**C2.** Product versus supervised corner was resolved to supervised. Whether τ_dent beats a local deterministic re-solve is open; that is its own step-4 falsifier.

### 3.7 Agreement with the sealed verdicts on learnability

- **Agree, Track A:** BGC is less identified than ATO, and "R-c higher-rank ATO" is the linear-decoder limit.
- **Agree, Track B:** C1 has "nothing to learn". That answers brief Q4 by substitution, and the developer concedes it.
- **Add, Track B (blind to WL77A):** this substitution was already posed as a human contract decision.
  - FACT: FNTO contract l.17 ("do not replace learned complex shading with a limited explicit radiosity solver") and l.86.
  - FACT: WL77A's coordinator decision ("admissibility as a learned K is an untaken human decision").
  - Selecting C1 does not resolve that decision; it re-raises it.
- **Disagree mildly, Track B on C2:** "not substantially different from C1" is right about the mechanism. But C2 is the only in-scope design that tests a **learned** component with a defensible identifiability story: dense, source-independent supervision of its own input domain.
  - If the research question stays "learned K", C2's steps 2–3 are the informative learned experiment, and C1 is their prerequisite teacher and control (INFERENCE).
  - C2 also fixes the two decisive WL77A H2 objections: sub-mfp first-collision projection and smoothness-only identification. So "already rejected in WL77A" should not be applied to C2 as amended.
  - WL77A B-B7's novelty objection (Neural Green's Functions / Element Learning / DeepRTE, as cited there) applies unchanged.

## 4. Consolidated agreements and disagreements

| Point | Track A | Track B | This review |
|---|---|---|---|
| BGC as a source-generalisation mechanism | Rejected | (blind) rejected the whole family | **Agree.** Strengthened by FACT WL68 l.295 and WL78 seed spread (CE-1 realised empirically). |
| BGC's A3 targets a non-binding factor | FATAL | — | **Agree in conclusion.** Qualify: the angular factor is secondary, not zero (WL79/80, WL77A). |
| Revised BGC is better for validity, not transfer | Yes | — | **Agree.** C2 corrected passivity is sound (DERIVATION). The max(1, S) clamp-like risk is real. |
| C1 transfers by construction | — | Yes (high confidence) | **Agree, conditional** on boundary-layer convergence. It is not project-new (WL77A H3, θ = 0). |
| "Learning must never see the source" | Implicit (cross-source sharing) | Cross-cutting rule | **Necessary, not sufficient.** WL76 FIA is the counter-evidence (FACT l.237; attribution INFERENCE). |
| BND_FAR classes unsupervised | — | INFERENCE | **Partly wrong.** TBE / TBC / SM / SR are supervised (FACT, WL73). The failure is within-class and out of orbit. |
| C2 deferred as a variant of C1 | — | Yes | **Agree on ordering.** C2 is the only learned design with a coherent coverage argument. Not a repeat of WL77A H2. |
| C3 / C4 rejected | — | Yes | **Agree.** CCA-rest is valid as an MC-only control. |
| First test | — | Convention gate, then the exact-F_in versus averaged-F_in pair | **Agree, and add.** WL76 Stage 1 validated the factorisation only in energy and coarse space (FACT, WL77A A-A15). Native and contrast-level reproduction is genuinely untested. Freeze discretisation settings reference-free before any scoring. |

## 5. Part E

Surviving designs in my dimensions: **C1** (primary) and **C2** (conditional, learned). The revised BGC record is not an architecture, and Track A rejected it. An abbreviated Part E is given for reference only.

### E-C1: CP-K, explicit collision-probability intrinsic operator (amended)

1. **Historical failure addressed.** Learned operators fit supervised sources but fail on independent BND_FAR sources (FACT, brief §3b). This includes late-training drift (WL71A/75/76) and seed-dependent held-out behaviour (WL68, WL78).
2. **Preventing mechanism.** No source- or context-indexed fitted quantity exists. A source enters only as the exact ray integral F_in. Boundary physics lives in the geometric kernels W1, Φ_pt, W and Φout on the true body (DERIVATION).
3. **Falsifying observation.** Any one of these:
   - (a) On INT sources, where observed order matches the predeclared order, the exact-F_in operator fails to reproduce the reference con_coarse. This falsifies the premise that "all incoming dependence enters through F_in" under the contract's conventions.
   - (b) BND_FAR misses a gate at the finest level within budget.
   - (c) Any column has ΣK > d·(1 + 1e-6).
   - (d) Dent signed change has relative error > 0.35 or correlation < 0.90.
4. **Should succeed.** Lighting changes on the rest slab, for interior and face-centre sources, especially blue, where the point-exact low orders dominate (INFERENCE).
5. **Should fail.**
   - Within the fixture: red grazing-in/grazing-out bins at trihedral corners if graded cells stay first-order under budget (HYPOTHESIS).
   - Outside scope, certainly (DERIVATION): any g ≠ 0, Fresnel or path-history-dependent BSSRDF material.
6. **Information at inference.** Mesh and port frames, σ_t, ω_c, g = 0, index matching, B's rays, and an iterative solve per lighting state.
7. **Deployable?** For this homogeneous fixture with known parameters, yes. As a general ReST-GI material K, no; it does not cover the project's other media (FACT, WL51 semantics).
8. **Learnable under sparse supervision?** Not applicable; nothing is learned. Sparse MC is used only for validation, so any population is usable once settings are frozen.
9. **Principal bottleneck.** The per-lighting resolvent solve at red ρ ≈ 0.974 inside the coupled K∘C solve, with coupled acceleration undesigned. Estimated 1–10 s per GI solve plus about 1–5 s dent template rebuild (ESTIMATES in the record).
   - C1 stores no K. "Reuse" is reuse of geometry factors only, so every new lighting state pays a full solve (DERIVATION).
   - Under the baseline gate, C1 is not a neural method. Its external comparator is a path-traced re-render, which must not be labelled Full Recompute (FACT, baseline document).
10. **Closest predecessor.**
    - Within the project: WL77A H3 LCR at θ = 0 (FACT).
    - Externally: collision-probability / first-collision-source methods and the zonal method, as cited in the sealed records and not re-verified by me.

### E-C2: SBG, learned scalar bounded-medium Green's function between exact flights (amended, deferred)

1. **Historical failure addressed.** Two failures:
   - the same source-transfer failure, with angular ends kept exact;
   - specifically, WL77A H2's under-identification and sub-mfp projection, and (INFERENCE) WL76 FIA's BND_FAR degradation despite a source-blind learned rule.
2. **Preventing mechanism.**
   - Exact flights carry all source and angle dependence.
   - The learned part is a bounded scalar ratio, log τ in [0,1] relative to exact slab or rest bases.
   - It is supervised **densely over its own input domain** (charts and a dent family), independently of source sampling.
   - Energy is fixed by the geometry-only D2 identity (DERIVATION).
3. **Falsifying observation.**
   - The learned box charts exceed 1.25× the exact chart-kernel pipeline (step 0) on any stratum.
   - Or τ_dent misses the signed-change gates on a pose excluded from its family.
   - Or apply plus rebuild does not beat C1-in-the-loop, or the path-traced re-render, at equal accuracy. Under the developer's own rule the learned part is then rejected.
4. **Should succeed.** The rest box, where every chart type is supervised. Radially symmetric dents inside the trained depth and radius range, away from sides (HYPOTHESIS).
5. **Should fail.** Dents near a side face (a coupled dent × side family), non-radial dents, bends, any geometry whose chart types are outside the box family, and every out-of-scope medium (DERIVATION of scope).
6. **Information at inference.** As C1, plus chart descriptors in a gauge-free form (stabiliser-averaged τ) and the pair-frame dent parameters.
7. **Deployable?** For parameterised fixtures, yes. As a general material K, no; same physics restriction as C1.
8. **Learnable under sparse supervision?** Not from sparse *source* supervision; it does not try to be. It needs dense chart supervision: exact reduced solves plus roughly 10³–10⁴ MC point-source fields with interior tallies (ESTIMATE, privileged training data). Under that supervision, identifiability is good (DERIVATION).
9. **Principal bottleneck.** The non-translation-invariant dent-region kernel (H-matrix rank unproven) and dense edge blocks (4–8 GB ESTIMATE). Plus training-data generation of tens of GPU-hours.
10. **Closest predecessor.**
    - Within the project: WL77A H2 RBG and WL76 FIA (learned source-independent post-collision response).
    - Externally: Neural Green's Functions, Element Learning and DeepRTE, as cited in WL77A / sealed records and not re-verified by me.

### E-BGC: revised record, abbreviated (does not survive in my dimensions)

1. **Failures addressed.** Non-passivity divergence (WL41/42/50, FACT) and the angular restriction (WL75/76).
2. **Mechanism.** A passive normalised pair generator.
3. **Falsifier.** Two seeds or instances with equal fits differ by O(1) on out-of-orbit boundary contexts. This has already happened in the family (FACT, WL68 l.295; WL78).
4. **Should succeed.** On-orbit and interior contexts.
5. **Should fail.** Out-of-orbit boundary contexts and intervening-corridor deformations (CE-1, CE-5).
6. **Information at inference.** Geometry features plus first-flight moments.
7. **Deployable?** Yes.
8. **Learnable under sparse supervision?** No (DERIVATION).
9. **Principal bottleneck.** Per-pair factor storage and apply cost (crossover R* ≈ 12.6).
10. **Closest predecessor.** Global ATO / W+UV-Q / FIA within the project; DeepONet / hypernetwork heads externally, as cited by Track A.

## 6. Bottom line (INFERENCE)

- **Generalisation:** only the exact flight–resolvent–flight structure (C1, and C2 at its exact ends) has a mechanistic claim to unseen sources. It converts transfer into a discretisation-convergence question, which can be checked without references.
- **Learnability:** the only coherent role left for learning is C2's dense, source-independent supervision of its own input domain. WL76 FIA shows that a source-blind rule trained through source packets is not enough.
- **Open human decision:** neither C1 nor C2 answers the project's learned-K question across materials. Whether a non-learned or physics-sandwiched K is admissible remains the human contract decision first recorded in WL77A. Nothing in WL80A resolves it.

---

## Panel reviewer: computation-practical

# WL80A cross-review: COMPUTATION and PRACTICAL VALUE

Reviewer scope: I am an independent reviewer and developed no candidate. This is a read-only thought experiment. I implemented, trained, ran and measured nothing, and I wrote no files.

Label key: **[F]** FACT, **[D]** DERIVATION, **[I]** INFERENCE, **[H]** HYPOTHESIS, **[S]** SPECULATION. Every timing that is not marked [F] is an estimate.

Candidates in scope:
- **BGC-o**: original BGC-Transport.
- **BGC-r**: Track A revised record. Track A rejected it as a mechanism for source generalisation.
- **C1**: CP-K in its amended form (Track B primary).
- **C2**: SBG (Track B deferred).

## 0. Integrity and assumptions

- [F] The sha256 of all ten sealed files, recomputed this session, matches `seal.json`.

**Assumptions used in every estimate:**
1. [F] Geometry and discretisation follow the brief: P = 3,840 ports, 16 incoming and 64 outgoing bins, RGB, a 100 × 100 × 10 mm body, σ_t = 1 mm⁻¹, albedo ω = 0.999 / 0.99 / 0.9.
2. [F] The main experiment GPU is LabServer63, an RTX 3080 Ti with 12 GB (WL76 cost table). [Assumption] I take public-spec figures of about 0.9 TB/s and 34 TFLOP/s fp32, as Track A did. Sparse kernels reach a fraction of these.
3. [Assumption] The C1 graded mesh has 2–3e5 cells, as in Track B §4. The C2 lattice is 0.5 mm, 8e5 cells, as in the C2 record.
4. [Assumption] A "1-light apply" is one dense lighting state and a "64-light apply" is a batch of 64. This matches the measured convention in WL75/76.

**Physical scales (DERIVATION):**
- [D] The body is 10 mfp thick and 100 mfp wide.
- [D] Diffusion with D = 1/3 mm and H_eff ≈ 11.4 mm gives slab-mode decay lengths of about 3.5 / 3.1 / 1.6 mm (R/G/B). The red infinite-medium length is about 18 mm.
- [D] The per-collision non-leakage of the fundamental slab mode is about 0.974. The resolvent gain is therefore about 37 / 28 / 8 (R/G/B), and plain Neumann needs about 250 iterations to reach 1e-3 in red. This agrees with both tracks.
- [D] A dense K has 61,440 inputs × 245,760 outputs × 3 channels = 4.5e10 entries, about 181 GB in fp32. A 128-nearest-neighbour local dense block is 1.5e9 entries, about 6.0 GB. Both agree with Track A.
- [D] Brute-force Monte Carlo tabulation of K at reference quality costs 16 packets × 3.5 s = 56 s per port per replica:
  - about 60 GPU-h per replica for all 3,840 ports;
  - about 3.7 GPU-h with the rest D4h orbits (about 240 orbit ports);
  - about 16–25 GPU-h for the sources affected by a dent.
  
  Monte Carlo is therefore not a per-edit operator, and any explicit candidate competes against that cost.

**Measured anchors in the repository (FACT):**

| Operator / item | Measured cost | Source |
|---|---|---|
| A75 learned ATO, apply | 1-light 1.07–1.37 ms; 64-light 34–63 ms (RTX 5080) | WL75 |
| A75, 64-light apply | 0.086 s (RTX 3080 Ti) | WL76 |
| FIA, apply | 1-light 3.9 ms; 64-light 0.349 s | WL76 |
| FIA, geometry rebuild | 75 s, of which 74.3 s is the q = 1,024 SS/W quadrature | WL76 |
| q = 1,024 single-scatter quadrature noise | 0.45–0.48 of the M norm; the noise floor is approached only at q ≥ 65,536 | WL77A |
| Geometry + B | about 0.32 of a 0.33 s selective update | WL59 |
| Monte Carlo oracle | 380 s per configuration (2.0 GB) | WL55 |
| Oracle selective dent update | 188.93 s | WL56 |
| CAK compose solve | 0.085 s over 51 iterations | WL70 |
| Cycles at matched contract | 4 spp, 4.67 s | WL67 |
| Cycles 16 spp / 1 spp | 2.40 s / 1.20 s | WL69/70 |

## 1. COMPUTATION

### 1.1 Comparison matrix

| | BGC-o | BGC-r | C1 CP-K (amended) | C2 SBG |
|---|---|---|---|---|
| **Learned parameters** | Generator plus encoders, order 10⁵ (the brief's family is about 270k) [I] | Unresolved; z ≥ 0 with no specified generator | 0 (an optional preconditioner affects only speed) | τ_θ, about 1e3–1e5 |
| **Runtime state** | Pair-factor cache: 243R floats per pair. N128 R8 is 3.8 GB, N128 R16 is 7.6 GB, N512 R16 is 30.6 GB [D, Track A, re-checked] | If inference averaging is used, a dense N128 block of 6.0 GB plus an unspecified far term [D] | No learned state. Sparse geometric factors: W near field about 1e8 nnz (about 0.8 GB with indices); Φout 1e8–3e8 nnz (or convolution templates at rest); F_in about 2.5e7; point templates (small at rest by translation sharing, about 1 GB of dent-region templates [I]); solver vectors about 3.6 MB per lighting state. Total about 3–6 GB for 64 lights [I] | The same flights as C1, plus a resolvent kernel Ĝ for each channel. Where FFT structure is lost (near sides, dent), stored Ĝ is 17–34 GB (adversarial review; the developer revised to 4–8 GB) |
| **Geometry rebuild** | Re-encode and decode in 10–30 ms (N128 R8); ×16 with A5 averaging (0.2–0.5 s) [I, Track A]. Dilation of about 13–24 mm under a local edit | Similar, with a dense averaged block. Analytic-chord d in ms [I] | Rigid: none. Dent: point templates, W, Φout and F_in rows in the dent region. Developer estimate 1–5 s; see 1.2(b) | Features, flights and Ĝ evaluation in the dent region. Dent-region pairs number about 1e9–1e10 (see 1.2(c)), so seconds to minutes [I] |
| **1-light apply** | Memory-bound at ≥ 4 ms (3.8 GB read) [D/I, Track A] | ≥ 7 ms (6 GB read) [D] | 25–40 accelerated W applications × about 1.5–3 ms (SpMV over 0.8 GB), plus the far field and Φout: about 0.07–0.2 s cold [I]. That is 50–150× A75's measured 1–1.4 ms | One pass. About 3–10 ms on FFT-structured parts; the dent and side parts are not convolutional [I] |
| **64-light apply** | 6.1e10 MAC. 5–10 ms with a fused receiver-major kernel; 0.1–1 s in this codebase's style [I, Track A] | Similar order | SpMM with 192 right-hand sides (64 × RGB) at 3.8e10 FLOP per W application, about 8–20 ms each, so 0.2–0.8 s per batch [I]. That is 3–10× A75's measured 0.086 s | Tens of ms by the developer's estimate. Not established |
| **GPU execution pattern** | Dense small GEMMs; tensor-core friendly; atomic scatter is the hazard | Dense streamed blocks | Irregular sparse SpMV/SpMM (memory-bound), precorrected FFT, and ray integration for rebuild. W values need fp32 (red redistribution sensitivity) [I] | FFT plus sparse, or an on-the-fly MLP over about 1e10 pairs |
| **Scaling with port density** (fixed body) | ∝ P·N(P), with N ∝ P inside a fixed physical radius, so **quadratic** [D] | Quadratic, and ∝ A·O in angle | Flights ∝ P (linear). W is independent of ports if cells do not shrink [D] | As C1 for flights. Ĝ is independent of ports |
| **Scaling with angular bins** | ∝ R(A+O) | ∝ A·O | Flights ∝ A and O; **W is independent of bins** [D] | As C1 |
| **Scaling with number of lights** | Linear | Linear | Linear, with arithmetic intensity improving with batching | Linear |
| **Albedo or material edit** | Retrain, or add ω as an input | Retrain | **Free**: W is channel-free and ω is a scalar in the solve [D] | τ depends on ω, so a table per ω or ω as an input |

### 1.2 Derivations that change or sharpen the sealed estimates

**(a) Batching narrows C1's apply gap; it is not a constant factor.** [D/I]
- At 1 light, C1 is bandwidth-bound on W. It must reread about 0.8 GB 25–40 times, while A75 reads about 0.1 GB once. The gap is 50–150×.
- At 64 lights, the matrix read is amortised. The gap falls to about 3–10× the measured A75 time, and to about the same order as FIA's measured 0.349 s.
- Track B gave a single 0.05–0.8 s range per application. The useful conclusion is that C1 is uncompetitive for single-state iterative GI and only moderately behind for batched relighting.

**(b) The C1 rebuild estimate has no measured analogue that supports it.** [F + I]
- WL76 built the closest existing machinery: first-collision forcing, SS escape, and second-collision W from geometry-only fixed-seed quadrature. It measured 74.3 s for the full rest geometry at q = 1,024 [F].
- WL77A later showed that q = 1,024 SS noise was 0.45–0.48 of the M norm, and that the noise floor was approached only at q ≥ 65,536 [F]. Scaled linearly, a sampling-based C1 would need about 80 min for full geometry, or about 10 min for the about 12 % of ports in the dent region [I].
- Track B's 1–5 s dent rebuild is therefore plausible only with **deterministic analytic integration**: Gauss quadrature of e^{−σs} over each (point, port, bin) cell, and analytic chord integrals for W. Track B does name "deterministic, stratified" quadrature, but no such kernel exists in the project.
- My order-of-magnitude count for that route: about 1e6 fine points × 9–25 near ports × 64 bins × 64 nodes ≈ 1e10–1e11 exponential evaluations for the full body. At rest this collapses to templates; under a dent about 12 % of it remains. That suggests sub-second to a few seconds [I].
- Track B was blind and could not see this anchor. Track A cites the 74.3 s correctly but only for BGC's A1.

**(c) C2's "non-iterative" advantage is roughly cancelled by the density of its kernel.** [D/I]
- The support of W that matters is a few mfp. However, e^{−3} ≈ 5 % of next-collision mass lies beyond 3 mfp, so a far field out to about 10 mfp is mandatory (see (d)).
- Ĝ's support is set by the red slab-mode length. A support of about 17 mm (5 × 3.5 mm) gives a lateral partner ratio of about (17/3)² ≈ 30 relative to W's near field. Track B's 25–40 W applications per solve are the same order.
- On FFT-structured parts (the rest interior) C2 wins. Near sides, edges, corners and the dent, which are exactly the BND_FAR and deformation targets, it does not.
- A dent region of radius 30 mm on a 0.5 mm lattice holds about 2e5 cells × about 7e4 partners ≈ 1e10 pairs. That cannot be stored in 12 GB, and evaluating it on the fly with a 1e4-parameter MLP takes about 3e14 FLOP, which is seconds or more per application.
- Ĝ is also needed per channel, whereas C1's W is shared across RGB.
- This agrees with Track B's adversarial "8.6e9 pairs" counterexample and with the deferral of C2.

**(d) The C1 far field is first-order, not an option.** [D]
- Storing W explicitly to 10 mfp means about 5e3 partners per cell, about 1.6e9 nnz and about 13 GB. It does not fit, so the precorrected FFT (pFFT) or a hierarchical scheme is required.
- Because escape is defined as the complement 1 − 1ᵀW, any far-field mass error moves directly into Φout. Track B lists pFFT and complement consistency as "unverified" (unresolved point 3). From a computation standpoint it is the main engineering risk in C1. It is also non-trivial on a deformed, graded mesh with blocked chords in the dent cavity.

**(e) C1's passivity falsifier has real slack against rounding.** [D]
- Output = ω·1ᵀq − (1−ω)·1ᵀψ ≤ d − (1−ω)(d + 1ᵀψ). For red the slack is at least about 1e-3·d and up to a few % with the collision count.
- fp32 rounding near 1e-6 therefore cannot trigger falsifier (b) at 1 + 1e-6. A breach would indicate a real discretisation defect such as a 1ᵀW overshoot from a clamped FFT. The test is computationally sound.
- Total energy is also insensitive to escape errors: red output ≈ leak/(leak + absorb) ≈ 0.96. The binding quantity is spatial and angular redistribution at grazing bins, as Track B says.

**(f) For the immediate test fixture, a GI solve is about one K application.** [D]
- The rest slab is convex, so C has no same-object re-entry. A "GI solve" is a single K application per lighting state, with a small re-entry loop only in the dent cavity.
- In multi-domain scenes (the WL70 floor fixture: 51 outer iterations, 0.085 s, about 1.7 ms each [F]):
  - a cold inner solve at every outer iteration gives about 51 × 0.1 s ≈ 5 s;
  - a warm-started joint iteration with a few W applications per outer step gives about 0.3–0.5 s [I].
- Track B's 1–10 s per solve assumes 10–30 outer iterations each with a full inner solve. That is the pessimistic end.

**(g) Tabulating C1 brings the dirty-set problem back.** [D/I]
- Caching C1's responses as a local K gives ms-scale apply, comparable to CCA-rest.
- Under a dent, the source columns whose response support meets the dent number about 1,000–1,600 ports. That covers top ports within about 25 + 15 mm, the bottom ports beneath (10 mm thick) and nearby side ports, giving 16k–26k columns.
- With local sub-box SpMM in 64-column batches, that is about 10–60 s [I]. Track B said "≈ 7,200 beam solves, minutes": its count is smaller and its per-solve cost larger, so the totals are of similar order.
- This is still about 3–20× cheaper than the 188.93 s Monte Carlo selective update [F], but above the few-second re-render.

### 1.3 Agreement with the sealed verdicts (computation)

- **Agree with Track A on BGC-o:**
  - storage crossover R* ≈ 12.6;
  - about 81× ATO arithmetic per pair;
  - N512 R16 cannot be held resident;
  - A5 averaging makes the blocks rank 16, so dense storage is cheaper.
  
  I add one point [I]: at N128 R8, BGC-o is computationally the **cheapest** in-scope design to rebuild (tens of ms) and fits 12 GB. Cost does not decide its rejection. Identification does (CE-1). This is a scope note, not a disagreement.
- **Agree with Track B on C1:** slower per application than the learned operators, and probably not cheaper than re-rendering under deformation.
  
  **Partial disagreement:**
  1. The gap depends on batch size, per 1.2(a).
  2. The 1–5 s rebuild rests on an unbuilt deterministic quadrature; the project's sampling analogue measured 74.3 s and was noise-limited (1.2(b)).
  3. On the isolated slab fixture, no outer GI loop exists (1.2(f)).
- **Agree with Track B on C2** (cost advantage over C1 "undemonstrated"). I strengthen it: the break-even in 1.2(c) is structural in exactly the target strata.

## 2. PRACTICAL VALUE

### 2.1 Comparison matrix

| | BGC-o | BGC-r | C1 | C2 |
|---|---|---|---|---|
| **Relevance to the FNTO K contract** | Fits the learned-K aim. Adds no identification for out-of-orbit contexts (Track A CE-1) | A validity hygiene record, not an architecture | **Conflicts with the contract wording** (see below); fits the port/bin interface and K/C ownership exactly | A learned τ preserves a learned element, but only as compression on the box |
| **Feasibility in the existing fixture** | High (reuses the W+UV infrastructure). A prototype would re-test the WL72/76/78 family | Low; key parts unspecified (rank, corridor, far term) | Moderate. New code is needed: graded mesh, deterministic flight kernels, sparse W plus pFFT, solver. The existing β/B chord grid, D4h maps and port frames are reusable [I] | Low until C1 exists (teacher, D2 energy projection) |
| **Anisotropic phase** | Structure unchanged (retrain) | Same | State must carry angular moments, P_N or discrete ordinates (×4–9 state); similarity scaling fails exactly in the boundary layer that matters [D/I] | Scalar Ĝ breaks |
| **Heterogeneity** | Structure unchanged; descriptors grow | Same | Peierls form holds for isotropic scattering; the FFT/pFFT is lost; flights need optical-depth marching, so cost rises [D/I] | Domain monotonicity still holds, but G∞ is no longer a closed form |
| **Refractive / Fresnel boundary** | Retrain | Retrain | Flights gain specular/Fresnel reflected chords (images on flat faces, ray tracing on dents). The scalar volume state survives; flight cost rises [D/I] | τ ∈ [0,1] fails, because reflection adds paths absent from ℝ³ [D] |
| **Glossy (rough) interface** | Retrain | Retrain | Needs an angular boundary state coupled to the volume: a different system [I] | Breaks |
| **K/C reuse: lighting change** | Valid (fixed linear map) | Valid | Exact (fixed factors; one solve per state) | Valid |
| **K/C reuse: rigid motion** | Valid in the body frame | Valid | Exact, no rebuild | Valid |
| **K/C reuse: dent** | Local re-encode; generalisation unproven | Same | Local factor rebuild, then a global solve. On the fly, transport-dirty changes (D_T ≠ D_G) are handled by the solve without a dirty-set oracle [D]; tabulation reintroduces the dirty set (1.2(g)) | Dent region is the weakest link |
| **Material albedo edit** | Retrain | Retrain | Free | Partial |

**FNTO contract conflict for C1** [F]:
- `docs/FNTO_ARCHITECTURE_CONTRACT_V0_1.md` l.6: "reusable *learned* source-to-receiver transport response".
- l.17: "do not replace learned complex shading with a limited explicit radiosity solver".
- l.26: K_θa is a "reusable learned intrinsic … operator".

Using C1 as FNTO's K would need a contract decision by the user. Track B recorded this as an open "learned vs explicit" disagreement; the contract text makes it concrete.

### 2.2 Weighing against the recorded cost context

The task wording calls the few-second figure a "direct path-traced re-render". Under the repository's binding terminology (`BASELINE_ROLES…#mandatory-baseline-terminology-and-comparison-gate`) it is comparison class 3 only, not Neural Full Recompute.

- **Lighting-only reuse at fixed geometry.** [I] C1 on the fly takes about 0.07–0.2 s per state, or about 0.2–0.8 s per 64-state batch. That is plausibly below a few-second re-render, if C1 reaches matched accuracy (unmeasured). Learned operators are about 100× cheaper still, but none is accurate on BND_FAR (§3b OUTCOME).
- **Dent.**
  - C1 on the fly: rebuild (sub-second to a few seconds with deterministic quadrature; tens of seconds or more with sampling) plus a solve. That ties the re-render at best [I].
  - C1 tabulated: about 10–60 s [I].
  - C2: seconds or more in the dent region [I].
  - BGC-o and BGC-r: tens of ms, but with no accuracy mechanism.
- **Against the 190 s oracle.** [I] This is C1's clearest practical value. Under the same physical assumptions, it is a deterministic, noise-free candidate replacement for the Monte Carlo per-configuration oracle: the 380 s TR of WL55, 188.93 s in WL56, and about 60 GPU-h for a full reference-quality Monte Carlo tabulation (§0). That is roughly 1–2 orders of magnitude cheaper per configuration, if the boundary layer converges.
  - It could also produce dense all-source supervision, which removes the brief's "population held at 16 sources" limit. I state this as a consequence, not a recommendation.
  - Neither track states this "deterministic oracle / teacher" role explicitly. C2 option (a) uses C1 as a teacher only for chart kernels.

### 2.3 Feasibility details the blind track could not know

- [F] The WL79/80 19-port prospective population is **unreferenced** (WL80 l.244). Track B's falsifier assumes that "never-used INT / BND_NEAR / BND_FAR populations" with references exist. New references would cost about 19 ports × 16 packets × 3.5 s × 8 replicas ≈ 2.4 GPU-h [D].
- [D] C1 has no fitting or selection, provided its mesh and quadrature are predeclared. All existing references are then independent tests for it:
  - all S16-R replicas;
  - the 1,920 WL77A Part 0 packets, which include the BND_FAR strata scored in WL78 [F];
  - the DEF71 / CONFDEF72 / DEF73 deformation references [F, WL79 l.170].
  
  A first verdict therefore needs no new Monte Carlo except step 0(b), the single-scatter tally. This is cheaper than Track B assumed.
- [F] WL76 Stage 1 already verified the first-collision factorisation against references over interior, edge, corner, side and vertical-edge sources (l.279). [F] Single scatter carries 25–36 % of the collided energy (l.268). This supports C1's premise that the low orders are exact, and it ties C1's accuracy to the deterministic single-scatter quadrature, which q = 1,024 sampling demonstrably failed.

### 2.4 Expected value of a bounded prototype

This is a qualitative assessment, not a recommendation. Choosing what follows is the user's decision.

| | Information value | Cost and risk | Assessment [I] |
|---|---|---|---|
| **C1** | High either way. A pass shows that the target K is deterministically computable at gate accuracy on BND_FAR, so the learned-operator failures concern learning, not the target or reference. A fail that keeps converging points to a contract or reference mismatch. A fail that plateaus bounds explicit discretisation | Largest engineering effort (multi-day new numerics). Compute ≤ 6 GPU-h with no training. Main risks: pFFT and complement consistency, deterministic grazing-bin quadrature, convention match | Highest. Does not answer the learned-K question |
| **C2** | Conditional on C1. Measures whether chart-space learning compresses without loss | Needs C1 first; dent compression unproven | Low on its own |
| **BGC-o** | Low. Its distinguishing element (A3) targets a factor the record calls non-binding (WL75 l.195/308, re-checked by Track A) | Moderate (training runs of about 1–5.5 h per seed are typical of this family [F, WL75/76]) | Low |
| **BGC-r** | Low as a model. Some value as validity hygiene for any learned K (single budget, ω·d bound, disjoint local and far terms, leak hygiene) | Unspecified | Record value only |

### 2.5 Agreement with the sealed verdicts (practical value)

- **Agree with Track A:** BGC is rejected as a transfer mechanism. Its revisions improve validity but not transfer. My dimensions do not overturn this.
- **Agree with Track B:** C1 is primary; C2 is deferred and dependent on C1; deployability is "Low"; scope is limited to isotropic, homogeneous, index-matched media.
  
  **Qualifications:**
  1. Practical value lies mainly in the oracle/teacher role and the lighting-only regime, not the dent-update regime.
  2. The fixture needs fewer new references than assumed.
  3. The FNTO contract text makes "learned vs explicit" a contract question, not only a reviewers' disagreement.
- **C3 and C4.** My dimensions do not overturn either rejection.
  - [I] Computationally, "C1 + class tabulation at rest" is CCA-rest with deterministic rather than Monte Carlo columns. CCA's runtime layout (about 164 MB of tables, scatter-bound at about 0.1 s for 64 lights) is the natural fast form of a validated C1 at rest. That is an implementation note, not a reason to revive CCA as a generalisation answer.
  - C4's bandwidth tie with C1 (7 GB per application, about 10–18 ms at about 0.4–0.9 TB/s) and its minutes-long dent rebuild are confirmed by the same arithmetic.

## 3. Open computational points (unmeasured)

1. Whether deterministic grazing-bin flight quadrature reaches the noise floor within seconds. The sampling precedent is 74.3 s and noise-limited.
2. Whether pFFT on a graded, deformed mesh stays consistent with complement escape, so that no 1ᵀW overshoot occurs.
3. Inner iteration counts for red at the graded resolution. The 25–40 figure is a diffusion-condition estimate; DSA performance at 0.05 mm layers is unverified.
4. Warm-started joint K∘C iteration cost in multi-domain scenes.
5. Per-pair nonnegative rank by pair class (Track A's decisive unknown), which still governs any learned pair form.

## Evidence checked

- **Sealed records:** all ten read for the relevant sections; hashes verified.
- **Worklogs:**
  - WL55 (oracle 380 s, 2.0 GB);
  - WL56 (188.93 s selective);
  - WL59 l.161/182 (Geometry + B 0.32 s);
  - WL67 (Cycles 4 spp 4.67 s);
  - WL70 l.27–34 and l.134 (2.09 s, 51 iterations, 0.085 s);
  - WL75 l.216–228 (A75 apply and rebuild costs);
  - WL76 l.102/204/252/268/279/288 (FIA cost, 74.3 s quadrature, Stage 1 coverage);
  - WL77A l.77–115 (q = 1,024 noise);
  - WL78 l.275 (BND_FAR);
  - WL79 l.170/273;
  - WL80 l.25/244 (population unreferenced).
- **Other repository documents:**
  - `docs/BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md`, mandatory gate section;
  - `docs/FNTO_ARCHITECTURE_CONTRACT_V0_1.md` l.6/17/26/86.
- **No web literature was consulted for this review.** Literature identities in the tracks were not re-verified here.
