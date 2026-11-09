# WL80B final verdict on C1

> WL80B record (workflow `wf_9bbd762a-ae2`, 12 agent runs). Agent texts reproduced verbatim; headings added by the Coordinator. Design-only: nothing implemented, executed, trained or measured. The original C1 architect was not resumable; a stand-in reconstructed C1 from the sealed WL80A records (immutable inputs, sha256 in `inputs.json`).

**Conclusion (verdict chair):** C1 PHYSICAL BASELINE ONLY

## Final report

## Implementation Fact
- **Nothing about C1 has been implemented, built or run.** No file was written and no code was executed in WL80B. The material is design records only.
- **The redesign proposes no revised architecture.** Both adversarial reviewers and the architect stand-in agree. What exists is a "corrected specification" of C1, and the architect has since withdrawn it as an executable specification for the dented pose.
- **Code facts I checked myself (read-only):**
  - The archived ballistic operator B is a fixed-seed Monte Carlo quadrature, not an exact integral: `experiments/fnto_rule57/r57_geometry.py:80` `ballistic_matrix(m,V,q=256,seed=57501)`, with weights `exp(-sigma_t t)/q` at l.91 and miss/wrong counts at l.97 [HM, code]. This qualifies the brief's "exactly computed B". It means "same rays as B" (CR l.150) cannot give both Σq = d to floating-point precision and exact D4h.
  - The tally conventions (flux per flux, exit-triangle frame, nested bins) come from Parts B and C, citing `p53_tracer.py` / `p53_geom.py`. I did not re-open them.
- **Records I checked myself:**
  - `coordinator_disposition.md` l.15–25 and l.31–35;
  - `wl77a/coordinator_decision.md` l.11–15 (H3 grounds) and l.32–39 (SS noise floors);
  - FNTO contract l.6, 17, 26, 103 ("learned");
  - `Architecture.md` §2 (physical definition of K; the text never says "learned") and its oracle row;
  - `BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md` l.9–30;
  - `76_WORKLOG_FIRST_INTERACTION_ANCHORED_TRANSPORT.md` l.248–256;
  - `trackB_selected_candidate.md` l.145–182.

## Measurement
- **None for C1.** Every C1 accuracy, convergence, memory and time figure in Parts A–H is a WL80A design claim, a new derivation or an untested hypothesis.
- **Historical anchors (other operators; RTX 3080 Ti, LabServer63)** [HM, `76_*` l.252–255, verified]:

  | Operator | Full rebuild | 64-light apply | 1-light apply | State |
  |---|---|---|---|---|
  | A75 | 0.015 s | 0.086 s | — | — |
  | FIA | 75.0 s (quadrature 74.3 s) | 0.349 s | 0.0039 s | 1.47 GB |

- **WL77A non-evidence single-scatter quadrature floors** [HM, `wl77a/coordinator_decision.md` l.34–39, verified]: the ring floor is 0.453 at q = 1,024 and 0.227 at q = 65,536, extrapolating to about 0.22 noise-free. This is the historical precedent for quadrature noise being mistaken for a representation floor.

## Observation
1. **Original C1's order split was internally inconsistent.**
   - The point-exact first-order escape S1 was combined with a cell-flat continuation. On directly lit single-face cells this double-counts first collisions, with a sign that is systematically Δ ≥ 0 (Chebyshev's integral inequality) [ND, Part B3].
   - So original C1 did not guarantee passivity or energy closure.
   - The amended split (q at points; W1/Φ_pt telescoped per ray; cells from the second collision on; complement escape) partitions every path exactly once. Its identity O = ω·d − (1−ω)1ᵀψ re-derives independently in Parts A and B [ND].
2. **Every BLOCKER found traces to an implementation choice, not to the Peierls/CP factorisation** (Part B0): the precorrected-FFT far field, an unspecified clamp, warm-started or accelerated solves, the archived random B, float32 FFT.
3. **The dented pose has no admissible W** (Review 1, objection 1; conceded).
   - No W construction is simultaneously non-negative, sub-stochastic, visibility-exact, untruncated, within 12 GB and applicable hundreds of times per light.
   - Truncating W under complement escape conserves energy but books the truncated mass as escape at the wrong place and direction.
   - That error is invisible to the closure, passivity and partition tests.
4. **The mechanism is prior art at both component and system level:**
   - flat-source collision probability (CP) by ray tracing;
   - first-collision source;
   - scalar resolvent followed by an angular sweep;
   - response-matrix / interface currents;
   - zonal / exchange factors with channel-free geometry;
   - clustered volume–surface hierarchies with local rebuild and global re-solve (Sillion 1995; Drettakis–Sillion 1997) [PA; Part F, Review 2 objection 2, conceded].
5. **C1 has no learned or fitted quantity.** It is the WL77A H3 θ = 0 class with a revised discretisation (CD l.18, verified). All four H3 rejection grounds carry over: the closure is vacuous by definition, admissibility is an untaken decision, the cost rests on an unbuilt kernel, and the scope is g = 0, homogeneous and index-matched (`wl77a/coordinator_decision.md` l.15, verified).
6. **The one WL80A "reframing contribution" is not a contribution.** SC l.150 (verified) says "unseen-source generalisation becomes a discretisation-convergence question checkable without references". Self-convergence is reference-free; accuracy is not (Part C3). Source independence of a linear operator with exact first flight holds for any deterministic solver and for the WL56 MC oracle.

## Interpretation
- **Physics: sound in principle.** Under g = 0, a homogeneous medium and an index-matched boundary, C1 is an exact-in-the-continuum factorisation [EP]. Its amended discrete form can establish every contract invariant except reciprocity on the convex rest pose [ND]. Reciprocity is not required.
- **Research question: not addressed.** C1 cannot address whether a *learned* intrinsic K transfers.
  - A pass would show only that the MC-defined target K is computable from declared physics on the scored strata (CD HIGH-2, verified).
  - C1 also has privileged information the learned arms lacked: σ_t, ω, g = 0 and index matching (WL77A B4).
- **Weighing.**
  - **Against a method candidate:** no learned part, no new mechanism, inadmissible under FNTO v0.1, and per-frame deformation outside its plausible operating region. The weight of argument is decisive.
  - **For a baseline role:** the strongest remaining case is narrower (Review 2, with the architect's amended wording). It is a competent classical numerical comparator that a future learned-K claim would have to beat at equal accuracy or equal runtime, within its scope.
  - **The baseline role is itself conditional:** a far field has to be specified, and C1's bias and cost against the Cycles-volume/MC reference have to be measured.
- **Meaning of the label "PHYSICAL BASELINE ONLY" here.** It means a physics-based classical numerical comparator. It is not:
  - a physical reference (Cycles volume holds that role: BASELINE_ROLES l.19);
  - an oracle;
  - a label source;
  - an F/S invalidation instrument.

## Unresolved Question
1. **Dent far field.** Does any non-negative, visibility-exact far field fit 12 GB? A cluster hierarchy with refinement of cavity-straddling pairs is [UH]. Until one is specified, the dent stratum and the dent signed change are unexecutable for C1.
2. **Rest-pose accuracy.** On BND_FAR, corners, edges, side middle and blue/low-ω, what is the bias at affordable resolution?
   - The lateral flat-source D inflation is +31 % at 1.25 mm [ND, Part C4.1].
   - The order-2 escape is cell-averaged.
   - Receiver-end angular aliasing is not addressed.
   - All of this is [UH].
3. **Contract.** Must an FNTO K contain learned parameters? Do declared physical constants count as "reusable rule parameters"? Both are human decisions (Part F5).
4. **Comparator row.** Should a "classical deterministic solver comparator" role be added? `BASELINE_ROLES` has no such row; adding one is a human decision.
5. **Cost.** Every C1 time is a near-field-only lower bound. The real cost is unknown, and it is unknown whether C1 is even cheaper than per-state MC (about 2–3.4 s per packet, HM) at matched accuracy.
6. **Bias cancellation.** Do rest and dent biases cancel in the signed change under fixed boundary stacks? [UH]

## Architecture Verdict
**C1 PHYSICAL BASELINE ONLY.**
- **Not a credible method candidate:**
  - learning-free;
  - ALREADY KNOWN at component and system level;
  - inadmissible as the FNTO learned K under contract v0.1;
  - not an answer to the learned-K transfer question;
  - no revision proposed, and none that adds a learned part would remain C1.
- **Its only defensible potential role:** a classical numerical comparator in the g = 0, homogeneous, index-matched scope.
  - Rest pose only, until a dent far field exists.
  - Conditional on its bias against the physical reference and its cost being measured.
- **The status of that role is a human decision.** Nothing is adopted, authorised or measured, and no successor follows from this verdict.
- **Preserved objections:**
  - the dent BLOCKER;
  - the "physical" misnomer;
  - possible non-competitiveness against MC;
  - privileged-information unfairness to learned arms.

## Reproducibility
- **Inputs:** read-only files at the line ranges listed under Implementation Fact, plus Parts A–H and the architect response as supplied in the task.
- **Derivations:** hand derivations only, reproducible from the stated formulas, including:
  - the energy identity;
  - Δ ≥ 0 by Chebyshev's inequality;
  - ρ ≈ 0.974 / 0.966 / 0.878 from B² = 0.0776 mm⁻²;
  - the truncation remainder r_N ≈ 2.6e-8·d at N = 663;
  - the lateral D inflation h²/6 − (h²/π²)Σφ(2πn/h)/n².
- **No execution:** no random numbers, no hardware, no code run. Prior-art citations are as verified by Parts B, C, E, F and Review 2; I did not re-fetch them.

## Answers to the brief's eleven questions

**1. What did the original C1 guarantee?**
- **In the continuum** (g = 0, homogeneous, index-matched) [EP]:
  - an exact Peierls/CP factorisation in which the source enters only through F_in;
  - linearity;
  - collided exits that are disjoint from the ballistic part.
- **As specified (CR l.137–153)** [ND, Part B3/B0], it guaranteed:
  - linearity and zero-in/zero-out (for a fixed linear solve);
  - entrywise non-negativity;
  - first-exit ownership on the convex rest pose;
  - no learned quantity.
- **It did not guarantee passivity or energy closure:**
  - The split ωS1 + ω²ΦoutW(I−ωW)⁻¹F_in, with a point-exact S1 and cell-flat continuation, double-counts a fraction Δ = e_p − ē_cell ≥ 0 of first collisions on lit single-face cells. That is about 0.6–2.4 % of q per bin at 0.5 mm, and about 5 % as μ → 0 even with 0.05 mm layers.
  - "Partition by construction" (CR l.153) was false for the analytic/FFT W actually specified. The developer conceded this (AR l.298).
- **Exact D4h was claimed** but conflicts with pairing F_in to the sampled B (r57_geometry.py:80, q = 256, seed 57501; verified).

**2. Which WL80A claims were too strong?**
- "S1 point-exact", written as Φout∘F_in (underspecified).
- "Partition per cell by construction".
- "Passivity for any h and any W accuracy" (SC l.98). CD l.20 makes it conditional on W ≥ 0 and 1ᵀW ≤ 1. Part B refines this: W ≥ 0 suffices for the identity, and 1ᵀW ≤ 1 is needed for non-negativity and ρ ≤ ω.
- "ΣF_in = d to fp against B" together with "exact D4h" (B is sampled).
- "Unseen-source generalisation becomes a convergence question checkable without references" (SC l.150, verified). Accuracy is not reference-free.
- Cost claims:
  - 0.05–0.8 s per apply, 1–10 s per GI solve, 1–5 s per dent rebuild, all resting on unbuilt kernels and assumed acceleration (CD l.21);
  - the 0.05–1 s dent rebuild over about 450 top ports. Bottom beams also change physically: β rises about 400× under a 6 mm dent [ND].
- "Rigid: nothing to rebuild" and "albedo edits free" are true for the K factors only.
- "Selected primary" (SC l.5), and the chair's "C1 is H3 at θ = 0" and "failures concern learning" (FD). These were superseded by CD l.16–18.
- The partition/reciprocity/non-negativity "trilemma" holds only conditionally (Part B9).
- CD MEDIUM-4's "§7 already bounds the mechanism" does not bound the cumulative W-chain lateral inflation (Part C5; conditional on how the emulation is read).
- CD MEDIUM-7's "A75 single-light not measured": Part E reports a WL75 measurement on an RTX 5080. That is cross-hardware, and I did not re-verify it.

**3. Can C1 be an admissible intrinsic K?**
- **Under FNTO v0.1: no.** l.6, l.26 and l.103 require a learned K and trainable K parameters (verified). l.17 warns against "a limited explicit radiosity solver".
- **Under `Architecture.md`: ambiguous.**
  - §2 defines K physically, without "learned" (verified).
  - The open question it owns is learned-K adequacy.
  - The only non-learned K role it records is the exact selective oracle as a "baseline for avoided work".
- **Physically,** the corrected C1 can meet linearity, non-negativity, passivity, closure (with an r_N tally) and ownership on the convex rest pose [ND].
- **On the dented pose ownership is not established**: there is no admissible W.
- **Admissibility is an untaken human decision** (WL77A H3 ground 2, unrefuted).

**4. What is reused when illumination changes?**
- All geometric state is reused bitwise: mesh, B, d, G1 = W1·F_in, S1 = Φ_pt·F_in, W (near and far) and Φout. All of it is channel-free.
- Only the right-hand side G1·u and a cold fixed-count solve repeat. There is no warm start in contract tests (it breaks zero-in/zero-out).
- Moments m_k = Wᵏ G1 u are channel-free, so one sweep sequence serves RGB. Keeping the per-order exit tallies (about 260 MB per light) also makes a uniform-albedo edit solve-free [ND, Review 2 objection 7, conceded; PA successive orders].
- Cost per lighting state is unknown. Near-field-only lower bound: about 0.86–1.26 s in contract mode [ND estimate].

**5. What is reused when geometry changes?**

| Edit | Reused | Rebuilt |
|---|---|---|
| Rigid, separated by air | All K factors, bitwise in the body frame [EP] | C, U and the GI solve. Optical contact moves the edit into the topology class |
| Dent | Factors farther than r_f ≈ 7–14 mm from moved geometry. Under the amended fixed-thickness boundary stacks, the bottom stack is bitwise unchanged [ND] | 26–42 % of beams (top, bottom, side and rim sources whose chords meet the thinned region), C cavity self-relations, a global cold re-solve. No admissible far-field W is specified, so the dent is not executable |
| Albedo | All factors | The solve, unless per-order tallies are stored |
| σ_t, thickness, topology, contact | Nothing | Full rebuild. Port correspondence is unspecified (`Architecture.md` requires it) |

Exact Woodbury updates exist but cost more than one solve (rank 2e4–6e4). Truncated low-rank updates break W ≥ 0 [ND, Part D].

**6. Largest GPU workload bottleneck.**
- **Main cost: the red multiple-scattering solve.** ρ ≈ 0.974 needs about 175–663 sweeps of W in the invariant-exact cold fixed-count mode. Each sweep reads about 0.8–1.2 GB of near field.
- **Plus a far field that has no feasible form.**
  - Stored to 10 mfp it is about 1.6e9 nnz, about 13 GB, which does not fit.
  - Matrix-free it costs about 1e12 segment deposits per sweep, or else it is ray-effect-limited.
  - The quoted 0.7–1.0 s is therefore a near-field-only lower bound [ND; Reviews 1 and 2, conceded].
- **Second: construction quadrature.** The sampled analogue took 74.3 s of FIA's 75.0 s rebuild [HM, verified]. The deterministic kernel C1 relies on is unbuilt.

**7. What is actually learned?**
Nothing. The optional ΔW closure and the learned preconditioner are removed. σ_t, ω_c, g = 0 and index matching are declared physical constants, not learned. Review 2's point is conceded: a self-refinement-supervised closure would be identifiable. Even so, it would make C1 H3-class, it learns discretisation error rather than transport, and it does not touch the transfer question.

**8. Strongest remaining counterexample.**
- **The dented pose.** No W is simultaneously non-negative, sub-stochastic, visibility-exact, untruncated, within 12 GB and applicable hundreds of times. Truncating W under complement escape conserves energy, yet relocates escape to the emitting cell's ports and bins. Closure, passivity and partition tests cannot see this.
- **At the role level:** whatever C1 measures, it cannot attribute or overturn the learned-family transfer failures (CD HIGH-2).
- **At the rest pose:** a predicted +31 % lateral D inflation at 1.25 mm (+8 to +15 % in length, [UH]). The decisive refinement (sides, corners, dent) is exactly where memory runs out.

**9. Does the revision differ from prior art?**
- No revision is proposed.
- The corrected specification is assembled from known parts:
  - flat-source CP (Marleau–Hébert; DRAGON);
  - first-collision source (Alcouffe; Fan–An–Ying §4);
  - MOC-style or clustered hierarchical far fields (Sillion 1995; Smits–Arvo–Greenberg 1994);
  - Neumann / successive orders;
  - local rebuild with global re-solve (Drettakis–Sillion 1997).
- The 2.5 mm × 64-bin interface is a parameter. It is a better implementation of a known method, not a method, and it is ALREADY KNOWN at system level.

**10. Credible Method candidate, physical baseline, or rejected?**
- **Not a credible Method candidate.**
- At most a physics-based **classical numerical baseline/comparator**:
  - in the g = 0, homogeneous, index-matched scope;
  - rest pose only until a dent far field exists;
  - conditional on measuring its bias against the Cycles-volume/MC reference and its cost.
- It is not a physical reference, oracle, label source or F/S instrument.
- It is rejected as a learned-K answer. Its role is a human decision.

**11. The single falsification experiment for the next human architecture decision.**
- A bounded, rest-pose-only "C1 comparator qualification", detailed under future_falsification_design.
- **Question:** at the finest affordable resolution, does the corrected C1 reproduce the existing MC rest-pose references on the scored strata (INT, BND_NEAR, BND_FAR, corners, side) within the gates, judged by three-level self-convergence plus a separately estimated floor? It also records measured cost against per-state MC and the measured learned anchors.
- **What it decides:** whether a competent classical comparator exists in scope. That is the precondition for the human decision on whether to give C1 any baseline role.
- **If no comparator role is wanted, no experiment is needed.** Admissibility is a contract decision, not a measurement.

## Strongest evidence against the selected conclusion

The strongest evidence is against the word "baseline" in the chosen label, not against rejecting C1 as a method.

1. **"Physical" is a misnomer, and the baseline role itself may be unachievable or not useful.**
   - C1 is a biased deterministic approximation.
     - The predicted lateral flat-source D inflation is +31 % at L0 (1.25 mm) [ND, Part C4.1, crude model].
     - The order-2 escape is cell-averaged.
     - Receiver-end Φout shapes carry angular aliasing that does not refine with h (Review 1, objection 3).
     - The bias is unmeasured.
   - The project's physical reference is Cycles volume (BASELINE_ROLES l.19, verified).
   - The roles table has no row for a classical solver comparator.
   - Calling C1 a "physical baseline" risks the very conflation that the AGENTS.md terminology gate exists to prevent.

2. **On the decisive non-rigid stratum the specification is not executable.** No admissible W fits on the dented pose. Review 1, objection 1, is a BLOCKER and was conceded. A baseline that cannot run on the dent, the stratum where the learned arms' transfer matters (DEF failures in WL71/73/78), is a partial baseline at best. On this ground **C1 UNRESOLVED** is arguable: the cluster-hierarchy far field is [UH].

3. **It may not be a *competent* comparator.**
   - The McGreivy & Hakim argument requires the classical baseline to be competitive, and C1's own cost case is weak:
     - near-field-only lower bound of about 0.86–1.26 s per light in invariant-exact mode;
     - the far field missing or infeasible.
   - Per-state MC costs about 2–3.4 s per packet [HM, WL71–75 per Part E]. A class-3 re-render costs 1.2–2.4 s on a different workload.
   - The measured learned anchors are FIA at 3.9 ms single-light and A75 at 0.086 s per 64 lights (`76_*` l.254–255, verified).
   - If C1 is neither cheaper than MC at equal accuracy nor more accurate at equal cost, it adds nothing as a comparator.

4. **The comparison is not like-for-like.** C1 receives privileged physics (σ_t, ω, g = 0, index matching) that the learned arms lacked (WL77A B4). Beating or losing to it does not show whether the learned family lacked an attainable mechanism (CD HIGH-2, verified).

5. **The case for C1 ARCHITECTURALLY INADEQUATE.**
   - It is inadmissible as the FNTO learned K under v0.1 (l.6, 26, 103, verified).
   - It cannot address the research question.
   - Per-frame deformation is outside its plausible operating region (Review 2 objection 5, conceded).
   - g ≠ 0 and Fresnel boundaries are incompatible with it.
   - A bigger body in contract mode needs about 1e4 or more sweeps.

**Why the label still holds.** None of these shows a defect in the factorisation itself; every BLOCKER is a numerics or specification item (Part B0). "Inadequate" would describe a role mismatch, which "baseline only" already encodes. These objections remain in force as conditions on the label, not as refutations of it.

## Separate evaluation (not aggregated)

Each axis is assessed separately; the axes are not aggregated.

- **Original C1:** CR l.131–285.
- **Revision:** the corrected specification plus the architect's consolidated amendments. No new architecture is proposed.

| Axis | Original C1 | Revision (corrected spec) |
|---|---|---|
| **Physical correctness** | Exact in the continuum under g = 0, homogeneous, index-matched [EP]. Discretely flawed: the split double-counts first collisions (Δ ≥ 0 on lit faces, about 0.6–2.4 % of q per bin at 0.5 mm) [ND, B3]. So passivity and closure are not guaranteed, and the claimed "partition by construction" does not hold for the specified W | Rest pose: all contract invariants establishable [ND]. These are linearity, non-negativity, passivity ≤ ω·d, closure (with an explicit r_N truncation tally), D4h by orbit-representative assembly, rigid invariance, and first-exit ownership with a float64 exact-kernel Toeplitz far field [UH]. Reciprocity is not exact; it is diagnostic only and not required. Dent: ownership NOT ESTABLISHED, because no admissible W exists (BLOCKER). Accuracy everywhere is [UH] |
| **Source generalisation** | Not a generalisation mechanism. The source enters only through F_in, so every column is computed, not inferred. "Checkable without references" (SC l.150) is too strong | Same. Unseen-source accuracy equals context-dependent discretisation accuracy (lateral bias, grading transitions, aliasing). It is structural plus numerical, nothing is learned, and it does not bear on learned transfer |
| **Compression** | Factors about 1–3 GB (U variant) or a sparse graded set, against a 181 GB dense K [WD/ND] | Factors about 2.5–6.8 GB at L0 [ND]. The far field has no fitting form on the dent. L1 explicit needs 13–19 GB. No compression beyond sparsity. A truncated K cache of about 8.9 GB is lossy and cannot be co-resident |
| **Geometry updates** | Rigid: K factors reused. Dent: about 450 ports, 0.05–1 s, claim invalid (bottom, side and rim sources change physically) | Rigid: bitwise in the body frame, plus C, U and GI rebuilt. Dent: 26–42 % of beams rebuilt, plus a global cold solve. Not executable (no far field). Fixed boundary stacks remove spurious bottom-layer change [ND]. Woodbury updates are exact but cost more than a solve. Thickness, topology, contact: full rebuild, no correspondence. Per-frame deformation: outside the plausible operating region |
| **Multi-light utility** | Factor reuse across lights; the joint-unknown cost claim was withdrawn | All factors reused. One channel-free sweep sequence serves RGB, and per-order tallies make uniform-albedo edits solve-free [ND/PA]. Per-light cost unknown, near-field-only lower bound about 1 s. Caching amortises only after about N_cols lighting states: about 4e3 at rest with D4h, about 6e4 otherwise [ND] |
| **Learnability** | An optional ΔW closure (H3-class) | Nothing is learned (ΔW and the learned preconditioner are removed). A refinement-supervised closure would be identifiable (Review 2, conceded) but learns discretisation error, not transport |
| **Novelty** | None as a mechanism (CP + first-collision source + escape) | None. ALREADY KNOWN at system level as a clustered zonal / CP hierarchy with local rebuild and global re-solve [PA] |
| **Scalability** | Plain iteration about 524 red sweeps. Acceleration assumed but unverified | The invariant-exact mode needs about 175–663 red sweeps for this slab (accuracy-set), and about 1e4 or more for bodies of about 100 mfp. Fixed-coefficient acceleration keeps linearity but forfeits guaranteed non-negativity and passivity. The far field is the scalability wall. Cell count about (σ_t·L)³ in optical units |
| **Generality** | g = 0, homogeneous, index-matched | Narrower than first stated: g = 0 only (g ≠ 0 incompatible beyond order 2); ω(x) and cell-aligned piecewise σ_t extensible; continuous or high-contrast σ_t unsupported; Fresnel incompatible as recorded; mesh must be in optical units; offset tiling fails for convex radii below about 1.1 mm |
| **Falsifiability** | Predeclared gates, but the ω·d and two-level rules are weak. The original "still decreasing is no excuse" was later reversed by CD HIGH-1 | Improved. Three-level Richardson per axis (lateral, depth, angular) with a separate floor estimate. INCONCLUSIVE where three levels are unaffordable. Independent β check, reference-side ownership check, convention gate on the scored mesh, closure including r_N. Self-referential checks are replaced. Still unfalsifiable on the dent until a W exists |

---

## Independent critic of this verdict record (preserved; the record above was not rewritten)

**Critic report on the WL80B verdict record "C1 PHYSICAL BASELINE ONLY"**

The record's main finding holds: C1 is not a credible method candidate and does not answer the learned-K question. Four things are overstated: the role label it assigns, the claim that a dent W is impossible, how decisive its falsification design is, and its cost comparisons. No implementation task, learned residual or disguised replacement architecture was found. The record leans on concessions in two places (issue 8) and partly defends C1 in one paragraph (issue 1).

**Citations I checked myself (read-only)**
- `r57_geometry.py:80-98` (fixed-seed B, q=256, seed 57501, miss/wrong counts). Correct.
- `76_*` l.248-256. Correct.
- CD l.15-25 and l.31-35. Correct.
- `wl77a/coordinator_decision.md` l.11-15 and l.32-39. Correct.
- FNTO contract l.6, 17, 26, 103, 180. Correct.
- `Architecture.md` l.17, 58, 87. Correct.
- `BASELINE_ROLES` l.9-30, including l.19. Correct.
- SC l.5, 98, 112, 126-130, 150. Correct.
- `59_*` l.182 and l.219. Correct.
- `56_*` l.30. Correct.
- `candidate_H3.md` l.123. Correct.
- `specialist_physics` l.90-93 and l.299-302. Correct.
- `p53_tracer.py` l.153, 168, 172, 190-191. Correct.
- `p53_geom.py` l.51-72. Correct; the normal direction maps to bin (4,4) as stated.
- `71_*` l.147: 3.37 s per packet. Correct.

**Arithmetic I re-derived**
- The energy identity O = ω·1ᵀq − (1−ω)1ᵀψ and the two-sided bound ω·1ᵀΦ_pt q ≤ ΣK ≤ ωd − ω(1−ω)(d − 1ᵀΦ_pt q) are both correct.
- Δ ≥ 0 by Chebyshev is valid for monotone single-face profiles.
- B² = 0.0776 is correct (10 mm plus 2 × 0.71 mm extrapolation).
- ρ comes out as 0.974 / 0.965 / 0.877. The record's 0.966 and 0.878 are rounding only.
- 525 and 663 sweeps are correct, r_N ≈ 2.6e-8 is correct, and e⁶ ≈ 403 is correct.
- +31 % variance is about +14.5 % in length, consistent with CD LOW-11's 9-18 %.

---

### HIGH

**1. The conclusion label goes beyond what the analysis supports.**
- The record says the comparator role is "a human decision". It also says the role is conditional on bias and cost that have not been measured, unexecutable on the dent, and without a row in BASELINE_ROLES. It then asserts that role as its verdict anyway.
- "PHYSICAL" contradicts its own disclaimer ("not a physical reference").
- The paragraph "Why the label still holds" argues from how sound the continuum factorisation is. That does not bear on whether C1 is useful as a baseline, so the paragraph is a mild default defence.
- **Correction:** split the verdict.
  - Method: NOT A METHOD CANDIDATE / rejected as a learned-K answer. This part is supported.
  - Comparator role: UNRESOLVED and unqualified, pending a human decision.
  - Drop "physical".

**2. "No admissible W on the dented pose" is stated as fact but is really a cost and specification gap.**
- The claim appears in Observation 3, Answers 3, 5 and 8, the evaluation table and the falsification design ("no admissible W exists").
- Unresolved Question 1 treats the same thing as open.
- E4's own chosen construction (matrix-free, ray-telescoped, with visibility) meets every invariant. It fails only on cost.
- The original record (CR l.170) bounds blocked chords longer than 6 mm by e^{-6}/(4πr²) (marked INFERENCE there).
- Under CD HIGH-1, a limit that comes from affordability is not a falsification.
- **Correction:** "No visibility-exact W within the recorded memory and cost caps has been specified, and the size of the escape misplaced by truncation has not been quantified." The dent is UNRESOLVED, not impossible.

**3. The falsification design is predictably non-decisive and contradicts itself.**
- By the record's own numbers, explicit L1 lateral resolution needs 13-19 GB, which exceeds 12 GB.
- So three lateral levels cannot be afforded. The likely result is INCONCLUSIVE by rule, or stop condition 2.
- Getting past this would need a matrix-free near field, which is new design work.
- Answer 6 says the far field "has no feasible form", yet the design specifies a float64 Toeplitz far field at rest, and that far field is [UH] in the evaluation table.
- **Correction:**
  - Reconcile whether a far field is feasible at rest.
  - State the outcome the record's own estimates predict.
  - Name the extra construction needed for three affordable levels.

### MEDIUM

**4. The privileged-information claim is misattributed.**
- WL77A B4 (adjudication_draft l.81) says that **A75** gets none of this physics.
- FIA's SS and W are quadratures of geometry and σ_t, multiplied by albedo, with isotropic re-emission and index matching (`76_*` l.20, l.100-102).
- So "the learned arms lacked σ_t, ω, g=0, index matching" is false for FIA.
- **Correction:** restrict the claim to the A75 / W+UV family and note that FIA shares the first-interaction physics.

**5. Cost comparisons are made at unmatched accuracy, with inconsistent or unsupported figures.**
- "180-260× FIA" and "learned anchors favour 4-260×" ignore that FIA was INSUFFICIENT FIT (CD l.22a) and that A75 fails to transfer. This repeats the mislabelling that CD MEDIUM-7 corrected.
- The C1 figures disagree with each other:
  - 0.7-1.0 s versus 0.86-1.26 s per light;
  - 175-663 sweeps versus 525-663 sweeps ("175" is never explained);
  - 1e9-1e11 versus about 1e12 far-field evaluations per sweep.
- The lower end "4×" is never derived.
- The basis for the estimates (bandwidth, fp32/fp64, GPU) is not stated. The float64 Toeplitz/FFT far field on an RTX 3080 Ti runs at about 1/64 of the FP32 rate, which the estimates do not reflect.
- MC costs 3.37 s per packet (WL71 l.147), which is one column. C1's figures are per light. The units do not match, and "matched accuracy" is never defined.
- **Correction:**
  - Give one consistent derived range with its assumptions.
  - Label every comparison as unmatched in accuracy.
  - Define what matched accuracy means against MC.

**6. Role analysis is incomplete and contradicts itself.**
- BASELINE_ROLES l.19 includes "independent analytic controls" in the physical-reference row.
- Row l.12 and Architecture l.87 hold the exact-oracle role, an MC K builder used as the baseline for avoided work.
- The REDESIGN (§2, E21) gives C1 the role "deterministic analogue of the WL56 MC oracle K". The verdict says C1 is "not an oracle".
- REDESIGN cites CD l.17 for the WL56 analogy, but CD l.17 does not say this.
- **Correction:**
  - Reconcile the two statements.
  - Say why C1 does not qualify as an analytic control (its bias is unmeasured).
  - Fix the CD l.17 citation.

**7. The outcome table has gaps, and the design breaks its own "no new references" rule.**
- Several cases are unclassified:
  - a run converging to a wrong limit (order ≥ 0.5 but floor above the replica spread), which should count as a fail;
  - orders between 0.5 and 2;
  - mixed results across strata;
  - how INCONCLUSIVE strata map to the outcomes.
- "No new physical references" contradicts the S1 check against an MC single-scatter tally at q ≥ 65,536, which is a new MC run (SC l.165).
- The INT / BND_NEAR / BND_FAR rest references are not located with paths.
- Scoring these populations, which SC l.171 calls "never-used", uses up a held-out population a future learned-K test may need. That cost is not stated.

**8. Agreement and concession are used as evidence.**
- Examples: "Both adversarial reviewers and the architect stand-in agree", and "conceded" given as support for the dent BLOCKER, the per-frame deformation limit and the prior-art findings.
- CD LOW-12 requires agreement to be recorded separately from epistemic labels.
- **Correction:** cite the derivation or source behind each claim instead.

**9. Some labels are wrong.**
- These are tagged [HM] but are not measurements:
  - `specialist_physics` l.90-93 (non-evidence at source);
  - the coordinator_decision SS floors (non-evidence checks, l.26);
  - `candidate_H3.md` l.123 "exact reciprocity" (a WL77A design claim).
- **Correction:** relabel them.

### LOW

**10. Original non-negativity is overstated.** Answer 1 says the original guaranteed entrywise non-negativity. CR l.175, option U, uses a masked FFT convolution, which the record's own B12 says can produce negatives. Correct it to "option G only, in exact arithmetic".

**11. The identity conditions are stated loosely.**
- The identity is algebraic. It needs only the complement definitions and a convergent ψ.
- W ≥ 0 gives passivity, through 1ᵀψ ≥ 0.
- 1ᵀW ≤ 1 gives non-negative escape and convergence.
- "W ≥ 0 suffices for the identity" is imprecise.

**12. Generality is overstated.** "g ≠ 0 incompatible" and "Fresnel incompatible" go too far. Collision-probability (CP) and method-of-characteristics (MOC) solvers handle anisotropic scattering and reflective boundaries. Restate as "outside the scalar C1 form; changes state and cost", which is SC l.86's own conceded wording.

**13. Citations are unverified.**
- I could not confirm "Marleau-Hébert NSE 1986". The 3-D ray-tracing CP paper I found is Roy, Hébert & Marleau, NSE 101 (1989) 217-225.
- I did not check Alcouffe 1985 or Khalil 1988.
- The record says it did not re-fetch its citations, so mark them unverified. The prior-art conclusion is unaffected.

**14. One H3 ground is mis-described.** "All four H3 grounds carry over" is a category error for ground 1: C1 has no closure, so "vacuous closure" is moot. Ground 1 should read "transfer carried by a classical solve", which is C1's identity.

**15. "No file was written in WL80B" is inaccurate.** `docs/plans/wl80b/` and the 80B worklog exist as untracked files. Clarify that the verdict author wrote nothing and the records are design files.

**16. The +31 % D inflation is a crude estimate, not established fact.**
- It is used as the strongest evidence, but it is a rough new derivation (ND).
- The nearest existing non-evidence suggests a smaller effect on the gates (`specialist_physics` l.299-302):
  - 1.0 mm voxels: +6 % red energy, coarse 0.043;
  - 2.5 mm lateral snap: coarse 0.065.
- Whether that §7 emulation re-emits once or at every collision order is ambiguous (l.295). That decides whether it bounds the cumulative chain error.
- Present the +31 % as an upper-bound hypothesis.

**What remains supported:** C1 is learning-free and ALREADY KNOWN, it is inadmissible as a learned K under FNTO v0.1, and it does not address learned-K transfer.

Sources:
- [Roy, Hébert, Marleau, NSE 101 (1989), ANS](https://www.ans.org/pubs/journals/nse/article-23610/)
- [Polytechnique Montréal record](https://publications.polymtl.ca/18046/)
