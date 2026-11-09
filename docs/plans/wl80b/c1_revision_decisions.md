# C1 revision decisions (Part G)

> WL80B record (workflow `wf_9bbd762a-ae2`, 12 agent runs). Agent texts reproduced verbatim; headings added by the Coordinator. Design-only: nothing implemented, executed, trained or measured. The original C1 architect was not resumable; a stand-in reconstructed C1 from the sealed WL80A records (immutable inputs, sha256 in `inputs.json`).

**Proposes revision:** NO. I propose no revised C1 architecture. Every defect found in Parts B to F has one of two fixes. Some need a specification correction inside the known collision-probability (CP) / first-collision-source (FCS) / method-of-characteristics (MOC) family: clamp policy (P2*), float64 or no FFT, a cold-start fixed-count iteration, a shared deterministic orbit-symmetric B/F_in ray set, a visibility-exact far field on the dent, a segment-assignment and watertight rule, and composing the point layer into beam operators. The others can only be fixed by spending more resolution or memory (lateral spacing 1.25 mm, matrix-free near field at L1). None of these fixes is architectural. They are standard numerics (PRIOR ART: Marleau-Hebert 3-D CP, Alcouffe FCS, MOC, P3M-style splitting), so the stop rules "no architectural value beyond existing numerical techniques" and "main new mechanism already known" (Part F, F2-F3) apply. The research question is whether a learned intrinsic K transfers to unseen sources and geometry edits. Adding a learned component would only reproduce WL77A H3 (learned CPM closure; its Horn-A 'vacuous closure' ground is unrefuted, coordinator_decision.md l.15) or WL80A C2/WL77A H2 (learned scalar resolvent; covered by Neural Green's Functions / Element Learning / DeepRTE). Either would replace C1's defining property, being learning-free, not refine it. The rules forbid adding a network for preference. So the complete answer is: C1 stays a learning-free deterministic integral-transport operator. Its specification needs the corrections listed in the decision records before it can serve even its qualified role, which is a diagnostic instrument / deterministic analogue of the WL56 MC oracle K (CD l.17). Its admissibility as an FNTO K is a human decision, and nothing here takes it.

**Descends from C1:** No revised architecture is proposed, so there is no descendant to classify. The corrected specification in the decision records keeps C1's identity: the same Peierls/CP factorisation, the same amended order-split (point-exact order-1 escape and order-2 deposit, cell resolvent from order 2), the same complement-defined escape, the same ownership rule and no fitted quantity. It only replaces implementation choices (pFFT, clamp, warm-started or accelerated solve, archived random B, float32, unspecified point placement and tiling) with ones that let C1's own invariants be established. It is therefore C1 with corrected numerics, not a new architecture, and it brings no new mechanism. Any candidate that would add architectural value against the learned-K research question would have to add a learned component, which would make it WL77A H3 (learned closure on the CP generator) or WL80A C2 / WL77A H2 (learned scalar resolvent between exact flights). That would remove C1's defining property, being learning-free and computing every column from declared physics, and would inherit those candidates' recorded rejection or conditional grounds. Such a candidate would be a replacement architecture presented as a refinement, which the rules forbid, so none is proposed.

### E1 Operator equation (amended split: q=F_in u; psi=omega W1 q + omega W psi; Ku = omega Phi_pt q + omega Phiout psi) — **RETAIN**
- **Original mechanism:** Original (i): K = omega S1 + omega^2 Phiout W (I-omega W)^-1 F_in with S1 'point-exact' but notated as Phiout∘F_in (CR l.137-146). Amended (ii): order-1 escape and order-2 collision deposit are point-exact via Phi_pt/W1; cells carry orders >= 2 (SC l.89-92) [WD].
- **Identified defect:** The original split double-counts or loses first collisions. The excess Delta = e_p - e_cell is systematically >= 0 on directly lit single-face cells (Chebyshev integral inequality), about 0.6-2.4 % of q per bin at 0.5 mm and about 5 % at mu->0 even with 0.05 mm layers [ND, Part B3]. The amended split has no counting defect.
- **Evidence or derivation:** Part B3/B5 re-derivation [ND]. Amended energy identity O = omega d - (1-omega) 1^T psi re-derived independently in Parts A, B2 and SC l.98 [WD/ND]. Peierls/CP factorisation exact under g=0, homogeneous, index-matched [EP].
- **Proposed change:** None to the amended form. The original split stays withdrawn.
- **Why the change addresses the defect:** The amended split is the only form in the records that partitions every path by collision order exactly once (Part B5) and carries an exact algebraic energy identity.
- **Preserved invariants:** No multiple counting, energy closure beta + exits + absorbed = 1 (under P1-P3), passivity <= omega d at convergence (under P4/P5), linearity in u.
- **Additional assumptions:** [EP] g=0, homogeneous, index-matched, sigma_t channel-independent. [HM, to be read from archived metadata] dom.g = 0 in the reference walker (Part B1).
- **Computational costs:** No change from the record. Order-2 escape is cell-averaged, unlike H3's exact D2 (Part F1), so accuracy at grazing outgoing bins is first-order in h [WD AR l.393].
- **New failure modes:** Flat-source bias from order >= 2. Its sign depends on the regime (Part B3 table): under-escape for decreasing intra-cell densities, over-escape (about +5.6 % top-cell ratio at mu=0.4, h=0.5) for diffusive profiles [ND].
- **Falsification condition:** Per-column closure d - Sum K - (1-omega)(1^T q + 1^T psi) = 0 to fp precision. A violation means the split or the tiling is wrong (Part B7).
- **Closest prior art:** Peierls / CP (Marleau-Hebert NSE 1986; DRAGON EXCELT). First-collision source (Alcouffe 1985). Fan-An-Ying 2019 (Part F P1-P3) [PA].
- **Novelty implications:** ALREADY KNOWN (Part F2).

### E2 State variables / fine first-collision point layer (q at points p, Phi_pt, W1) — **REVISE**
- **Original mechanism:** Point first-collision density q(p), channel-free. Point factors Phi_pt, W1 from a per-point ray set. Point placement unspecified: 'about 1e3 points per beam' (AR l.408) vs 'about 1e6 global points' (XR l.945) [WD].
- **Identified defect:** (a) Per-beam points stored as W1 at point level need about 1.8e10 nnz (150-220 GB) [ND, Part E0]. (b) A global 1e6-point set (~0.46 mm spacing) projects q onto points and loses part of the 0.3-0.8 mm bin-dependent footprint shift that carries coarse contrast [ND, Part C1 S3]. In that reading 'point-exact first flight' is false.
- **Evidence or derivation:** Part E0 storage count [ND]. Part C1 footprint argument, with footprint magnitudes from wl77a/specialist_physics.md l.90-93 [HM, labelled non-evidence at source].
- **Proposed change:** Specify per-beam point quadrature, composed at build time into beam-level operators G1 = W1·F_in (beam->cell) and S1 = Phi_pt·F_in (beam->(port,bin)). Points never exist at runtime. The per-point direction sets must be rotated per point, or otherwise de-aliased, to suppress ray effects.
- **Why the change addresses the defect:** Only this reading keeps the first flight exact to quadrature and fits memory. It is algebraically identical to the record's intent because all factors are linear.
- **Preserved invariants:** Per-point partition P1 (telescoping per ray, carried into S1 and G1 column sums: 1^T S1 + 1^T G1 = 1^T F_in = d). Channel-free geometry. Linearity.
- **Additional assumptions:** Directional quadrature per point converges with rotated sets [UH]. WL77A non-evidence: one-ray single scatter reached its noise floor only near q = 65,536 (coordinator_decision.md l.38-39) [HM].
- **Computational costs:** Construction about 6e12 exponential/bin evaluations with per-beam points, about 60x the global-point route [ND, Part E5]. At rest, translation templates reduce this to about 3-8 % [ND]. Runtime storage of G1/S1 is about 0.55-1.9 GB [ND, Part E1]. No kernel exists, so all of this is unmeasured.
- **New failure modes:** Ray-effect aliasing that does not refine with h (Lathrop 1968) [PA]. Construction too slow to make dent rebuild useful.
- **Falsification condition:** Independent stochastic spot check of S1 and G1 columns at q >= 65,536 (FD l.286). Disagreement beyond MC noise falsifies the quadrature.
- **Closest prior art:** FCS beam templates. MOC track quadrature. Ray-effect literature (Lathrop NSE 32, 1968) [PA].
- **Novelty implications:** ALREADY KNOWN; this is an implementation choice.

### E3 Graded body-fitted mesh and refinement axes (L0 lateral 1.25 mm; depth 0.05->0.4 mm) — **UNRESOLVED**
- **Original mechanism:** Graded, body-fitted mesh, about 2-3e5 cells: lateral 1.25 mm, depth layers 0.05, 0.05, 0.1, 0.1, 0.2, 0.2, 0.4 mm..., edges as a cross-section product (AR l.338-353). The chair splits lateral and depth refinement into separate axes L0/L1/L2 (FD l.250) [WD].
- **Identified defect:** Flat lateral re-emission inflates D_lat by +31 % at 1.25 mm (+14.5 % in length if D_z is unchanged; net plausibly +8 to +15 %) and by +8.7 % at 0.625 mm. The error is second order in h [ND, Part C4.1]. Side-middle sources see z-cells of about 0.8-1.0 mm (Part C4.4). The 1.8-4 mm BND_NEAR band sits in an unspecified grading transition (Part C4.3). Three-way corner grading is unspecified. L1 requires about 1.6e9 nnz (13-19 GB) in explicit form [ND, Parts C4.1/E1].
- **Evidence or derivation:** Parts C4.1, C4.3, C4.4 and E1 [ND]. wl77a/specialist_physics.md l.299-302 [HM, non-evidence; does not bound the chain error, per Part C5].
- **Proposed change:** None proposed. A finer lateral axis is a resolution/affordability decision, not an architectural one. The mesh still needs specification: tensor vs non-conforming side grading, corner grading, and the BND_NEAR transition.
- **Why the change addresses the defect:** The bias is the generic flat-source error of CP. Refining it is numerics, and the decisive refinement (sides, edges, corners, dent) is exactly where FFT/Toeplitz structure is lost and memory runs out (Part E1).
- **Preserved invariants:** Invariants are independent of h (Part B0). Only accuracy depends on the mesh.
- **Additional assumptions:** The diffusion-scaled K0 exit-profile model used for gate-impact estimates is crude [UH].
- **Computational costs:** Explicit form: L0 2.5-6.8 GB of factors. L1 exceeds 12 GB without a matrix-free near field. All-axis halving 51-77 GB [ND].
- **New failure modes:** Red abs_coarse plausibly 0.04-0.08 at L0 against the 0.10 gate [UH, crude]. INT red error >= edge-row error, the reverse of the learned-family ordering [UH]. Non-monotone errors across BND strata.
- **Falsification condition:** Per CD HIGH-1: a miss with observed order < 0.5 between the two finest affordable levels is a C1 fail. A converging miss is affordability-limited. The predicted order is about 2 for the lateral chain bias.
- **Closest prior art:** Graded CP/MOC meshes; flat-source vs linear-source CP [PA].
- **Novelty implications:** None.

### E4 W far field: precorrected FFT beyond about 3 mfp — **REMOVE**
- **Original mechanism:** Explicit near field within about 3 mfp (about 1e8 nnz) plus a pFFT far field (SC l.112). The chair asks for 'non-negative interpolation or matrix-free' (FD l.256) without choosing.
- **Identified defect:** (a) Precorrection subtracts the grid near interaction, which can put negative entries in W, so W >= 0 is lost [ND, Part E1]. (b) Float32 FFT roundoff of about 1e-7·max breaks the non-negativity test (Part B12). (c) Far-field mass error can push 1^T W > 1 and needs a clamp (Part B2). (d) On the non-convex dented pose the free-space kernel ignores in-medium visibility, so K carries air-crossing paths owned by C (BLOCKER BL-1, Part B6). (e) Exact translation equivariance also needs a grid commensurate with the 2.5 mm port pitch (Part C2).
- **Evidence or derivation:** Parts B2, B6, B12, C2(5) and E1 [ND]. Phillips & White 1997 pFFT construction [PA].
- **Proposed change:** Drop the pFFT. W is defined only by full-chord ray telescoping with visibility, applied matrix-free beyond the explicit near field. An Ewald/P3M-type non-negative kernel split is allowed only on the convex rest pose, in float64, with an explicit test.
- **Why the change addresses the defect:** Ray telescoping is the only W construction in the records that gives W >= 0, 1^T W <= 1 and exact first-exit ownership together on a non-convex body.
- **Preserved invariants:** W >= 0, sub-stochastic columns, ownership (air -> C), P2 without a clamp.
- **Additional assumptions:** A matrix-free ray-telescoped far field costs 1e9-1e11 kernel evaluations per application [UH, Part E1].
- **Computational costs:** Much slower per W application than an FFT. The C1 cost case worsens further against the measured FIA 3.9 ms single-light anchor (76_* l.255) [HM].
- **New failure modes:** The apply becomes compute-bound and exceeds every recorded cost cap.
- **Falsification condition:** Report per-column 1^T W, minimum entry, and the fraction of far-field mass crossing cavity air on the dented pose. Any negative entry or 1^T W > 1 falsifies the construction.
- **Closest prior art:** pFFT (Phillips-White 1997). P3M kernel splitting. MOC matrix-free sweeps [PA].
- **Novelty implications:** None.

### E5 Clamp-and-renormalise of escape when 1^T W > 1 — **REMOVE**
- **Original mechanism:** 'Clamp the escape to 0 and renormalise' (AR l.351). The Track B chair requires complement preservation and reporting (SC l.112), and the cross-review chair requires predeclaration and the binding fraction (FD l.195, l.258).
- **Identified defect:** The policy is unspecified. Zeroing escape without rescaling the W column creates a net gain of omega Sum (1^T W - 1)^+ psi, which breaks passivity (Part B2, BLOCKER BL-2). Interior escape is only about 1e-3 per collision, so a 2 % far-field error makes the clamp bind widely [ND].
- **Evidence or derivation:** Part B2 adversarial example [ND]. E2(5) ~ 1.0e-3 mid-plane escape [ND].
- **Proposed change:** With E4 removed, a ray-telescoped W has 1^T W <= 1 exactly, so no clamp exists. As a guard, predeclare P2* (rescale the column to unit sum, escape 0) and abort if it ever binds above roundoff.
- **Why the change addresses the defect:** A clamp that binds is a post hoc stabiliser; AGENTS.md forbids energy clamps. Removing its cause removes it.
- **Preserved invariants:** P2 complement identity; passivity; non-negativity of escape.
- **Additional assumptions:** Telescoping is computed in float64 so that 1^T W <= 1 holds above roundoff [ND].
- **Computational costs:** None beyond E4.
- **New failure modes:** If the guard binds, the W construction is defective. This is reported, not repaired.
- **Falsification condition:** Binding fraction > 0 above fp roundoff.
- **Closest prior art:** CP conservation normalisation (US 5267276 class) [PA].
- **Novelty implications:** None.

### E6 Ownership rule (in-medium chords only; first exit terminates; re-entry belongs to C; ballistic to B) — **RETAIN**
- **Original mechanism:** Only in-medium chords count in F_in, W1, Phi_pt and W. A chord through the dent cavity is an exit (CR l.151; SC l.95) [WD]. FNTO contract l.44 [HM].
- **Identified defect:** None in the rule itself. The defect was in its implementation via pFFT (moved to E4) and in mesh tiling (moved to E12).
- **Evidence or derivation:** Matches the reference: continue_air=False (p53_tracer.py:153); the WL57 walker has no re-entry branch [HM, Part B6]. WL53: disjoint K/C composes stably and double claim diverges (rho 4.75) [HM, AGENTS.md].
- **Proposed change:** None.
- **Why the change addresses the defect:** This is the FNTO ownership contract and is consistent with the reference.
- **Preserved invariants:** K/C disjointness; first-exit semantics.
- **Additional assumptions:** Index matching. At self-contact the C->K ownership switch is a representational discontinuity of the port/bin contract (Part D2 case 7) [ND].
- **Computational costs:** Visibility tests in every ray integral.
- **New failure modes:** Near-contact geometry, where contract resolution, not C1, sets the error.
- **Falsification condition:** A cavity-crossing-mass audit on the dented pose (see E4).
- **Closest prior art:** FNTO contract; interface-current domain decomposition [PA].
- **Novelty implications:** Not a contribution.

### E7 First flight F_in and its pairing with the archived ballistic B ('same rays as B', Sum F_in = d to fp) — **REVISE**
- **Original mechanism:** F_in is the exact ray integral of sigma_t e^{-sigma_t s}, using the same rays as B so that Sum q = d to fp (CR l.150; SC l.90-94) [WD].
- **Identified defect:** The archived B is a fixed-seed MC quadrature (seed 57501, q = 256 by default; r57_geometry.py:80-98) [HM]. Consequences: (a) missed or wrong-side rays add 0 to beta but cannot deposit, so Sum q = d fails (Part B4); (b) B's sampling error is about 3 % of d at edge-port bins aimed at the side face [ND]; (c) d is not D4h-equivariant, which conflicts with the exact-D4h test (BLOCKER BL-3, Part B10).
- **Evidence or derivation:** Parts B1, B4 and B10 [HM code facts + ND]. Brief l.58 calls B 'exactly computed', which this qualifies.
- **Proposed change:** Rebuild B and F_in together on one dense, deterministic, orbit-symmetric ray set with no nodes on bin edges. Declare the new d. Use a watertight intersector, or a predeclared leak category on both sides. F_in directions are cosine-weighted within the bin (bin_sample convention; p53_geom.py:59-65) [HM].
- **Why the change addresses the defect:** Both complementarity and equivariance must hold at once, and both are hard tests in FD.
- **Preserved invariants:** Ballistic/scattered disjointness, Sum q = d, D4h equivariance of d.
- **Additional assumptions:** The reference's beta convention can be re-derived without changing the reference K (collided exits are independent of how B is tallied) [ND]. The archived q must be read from meta['ballistic']['q'].
- **Computational costs:** One-time deterministic B rebuild. The comparison with the reference must use the reference's own d, or report the d difference.
- **New failure modes:** The new d differs from the reference's d convention, which shifts every column (a convention mismatch, CD HIGH-2).
- **Falsification condition:** Sum F_in vs the new d to fp precision per column, and beta(g·i, g·a) = beta(i, a) to roundoff.
- **Closest prior art:** FCS uncollided solve with inflow boundary source (Fan-An-Ying 2019 §4) [PA].
- **Novelty implications:** None.

### E8 W1 / Phi_pt point partition (one shared ray set per point; exact telescoping) — **RETAIN**
- **Original mechanism:** Each ray from p deposits e^{-s_a} - e^{-s_b} per crossed cell and exits with e^{-L} into exactly one (j, o), so 1^T Phi_pt[p] + 1^T W1[p] = 1 (SC l.95) [WD].
- **Identified defect:** None in the construction. Placement and ray-set design are moved to E2. Edge-node bin assignment breaks equivariance (Part B10, ex. 2), handled in E14.
- **Evidence or derivation:** Part B3 'amended fix verified' [ND]. The chord-segment deposit is analytic in a homogeneous medium (Part A ambiguity 7) [ND].
- **Proposed change:** None (beyond E2 composition).
- **Why the change addresses the defect:** It is the structural fix for the original split's double count.
- **Preserved invariants:** P1 per-point partition, non-negativity, ownership.
- **Additional assumptions:** A watertight intersector (Part B5).
- **Computational costs:** Included in E2.
- **New failure modes:** Aliasing (E2).
- **Falsification condition:** Per-point partition to fp, and a stochastic spot check (E2).
- **Closest prior art:** FCS + CP escape/collision probabilities [PA].
- **Novelty implications:** ALREADY KNOWN.

### E9 Cell-to-cell W (flat-source CP kernel, ray telescoped, partition-first, reciprocity given up) — **RETAIN**
- **Original mechanism:** W[target, source] is the next-collision probability for uniform isotropic emission in the source cell. Built by ray telescoping so that 1^T W <= 1, with exact partition preferred over volume-weighted reciprocity (AR l.300-304; SC l.96) [WD]. Index convention flips between records (Part A ambiguity 3).
- **Identified defect:** Flat-source bias (E3). Reciprocity relation was written transposed (Part A ambiguity 3). The claimed partition/reciprocity/non-negativity trilemma is conditional: volume-weighted symmetrisation plus P2 keeps partition and non-negativity except where the cell's reciprocity defect exceeds its escape probability (about 1e-3, deep cells) [ND, Part B9].
- **Evidence or derivation:** Parts B9 and C4.1 [ND].
- **Proposed change:** Fix the index convention as W[target, source]. Reciprocity is moved to E15 (diagnostic only). The linear-in-cell source stays optional and is not adopted, because it can go negative (AR l.418).
- **Why the change addresses the defect:** Ray-telescoped flat-source W is the only form that meets every contract invariant (Part B0, physics-level conclusion).
- **Preserved invariants:** W >= 0, 1^T W <= 1, rho(omega W) <= omega < 1.
- **Additional assumptions:** Ray-telescoped W is computable to sufficient angular accuracy [UH].
- **Computational costs:** Near field about 1e8 nnz (0.8-1.2 GB) at L0 [WD/ND]. Far field per E4.
- **New failure modes:** Lateral D inflation (E3). Rest vs dent bias differences (E17).
- **Falsification condition:** Column sums and minimum entry per cell, and the refinement order per CD HIGH-1.
- **Closest prior art:** Flat-source CP (DRAGON, Marleau-Hebert) [PA].
- **Novelty implications:** ALREADY KNOWN.

### E10 Phiout: complement-defined cell escape with quadrature shape over (j, o); radiance-weighting convention — **RETAIN**
- **Original mechanism:** 1^T Phiout[y] := 1 - 1^T W[:, y]. Quadrature supplies only the (j, o) shape, tallied 'with the reference's radiance-weighted convention' (SC l.97). Full chords (AR l.438) [WD].
- **Identified defect:** The convention was undefined in WL80A (FD l.458). From code it is a flux-per-flux tally binned in the exit triangle's (t, b, n) frame with no cosine factor (p53_tracer.py:172, 190-191; r57_fastwalk.py:36-37; phys76.py:46-51) [HM, Parts B1/C0]. The shape must be normalised to fp; a float32 error of about 1e-7 lies inside the passivity tolerance (Part B7).
- **Evidence or derivation:** Code conventions [HM]. fine16 = bin_parent(8, 4) nests exactly (p53_geom.py:51-56, 68-72) [HM].
- **Proposed change:** State the convention explicitly: Phi_pt and Phiout are escape probabilities per (j, o) in the exit triangle's current frame, with shapes normalised in float64. The step-0 gate confirms that the walker tally matches phys76 (not opened by Part C).
- **Why the change addresses the defect:** A convention mismatch shifts every column and makes a converging solve converge to the wrong answer (Part C3).
- **Preserved invariants:** P2 complement, non-negativity, consistency with the reference tally.
- **Additional assumptions:** The walker and phys76 tallies agree [UH until checked].
- **Computational costs:** Negligible.
- **New failure modes:** Flat-cell shape bias at grazing outgoing bins (about 2 % at 0.1 mfp layers) [WD AR l.393].
- **Falsification condition:** Convention gate: half-space H-function albedo <= 1 %, and single-scatter Phi_pt·q against an MC single-scatter tally (SC l.163-166).
- **Closest prior art:** CP escape probabilities [PA].
- **Novelty implications:** None.

### E11 Absorption accounting and passivity falsifier — **REVISE**
- **Original mechanism:** Absorption is implicit at (1-omega) per collision. Falsifier Sum K > d(1+1e-6) (SC l.175), corrected to omega_c·d (FD l.194, l.267). Conditional on W >= 0 and 1^T W <= 1 (CD l.20).
- **Identified defect:** (a) The omega·d falsifier cannot detect solver error: a converged solve's error (about 39·1e-6) is far below the absorption margin, which is >= 1.25e-4·d at corners for red [ND, Part B2]. It detects only structural defects larger than that margin. (b) CD l.20 refinement: W >= 0 is sufficient for passivity through 1^T psi >= 0, while 1^T W <= 1 is needed for non-negativity and for the rho <= omega bound, not for the identity [ND].
- **Evidence or derivation:** Part B2 [ND].
- **Proposed change:** Make the per-column closure identity (Part B7) the primary hard test, together with the per-point/per-cell partition tests. Add the two-sided physical bound omega·1^T Phi_pt q <= Sum K <= omega d - omega(1-omega)(d - 1^T Phi_pt q) [ND, exact in the continuum too]. Keep omega·d as a reference-sanity check: a reference column above omega·d beyond noise flags a d-convention defect.
- **Why the change addresses the defect:** These tests detect the defects that can actually occur, whereas the omega·d test is nearly vacuous at convergence.
- **Preserved invariants:** Passivity, energy closure.
- **Additional assumptions:** Exact or monotone solve (E13).
- **Computational costs:** Negligible.
- **New failure modes:** None new.
- **Falsification condition:** Any column violating closure to fp, or either side of the two-sided bound.
- **Closest prior art:** Standard balance checks in CP/S_N codes [PA].
- **Novelty implications:** None.

### E12 Mesh tiling, watertightness and segment assignment (implicit in body-fitted mesh) — **REVISE**
- **Original mechanism:** Body-fitted cells; telescoping assumes each in-medium segment belongs to exactly one cell (implicit) [WD].
- **Identified defect:** Cell tops built from 1.25 mm lateral corners differ from the reference surface (0.625 mm triangles) by sagittas of about 5-26 µm, i.e. 10-50 % of a 0.05 mm layer [ND, Part B6]. A non-watertight intersector can double-assign or drop rays at triangle edges (Part B5).
- **Evidence or derivation:** Parts B5/B6 [ND].
- **Proposed change:** Define cells as the true triangulated medium partitioned by offset layer surfaces, or give an explicit segment-assignment rule. Use a watertight intersector or a single-hit rule, matching the reference's leak accounting (p53_tracer.py:168).
- **Why the change addresses the defect:** P1 and P2 hold exactly only under exact tiling.
- **Preserved invariants:** Partition, closure, no double assignment.
- **Additional assumptions:** Offset surfaces stay non-self-intersecting under the dent (curvature radius about 7.6-36 mm vs layer depth) [ND].
- **Computational costs:** Small.
- **New failure modes:** Offset-surface degeneracy at sharp dent profiles.
- **Falsification condition:** Per-ray segment-length sum equals chord length to fp. Leak count equals the reference convention.
- **Closest prior art:** Standard CP tracking geometry [PA].
- **Novelty implications:** None.

### E13 Solver form (converged 1e-6 relative, warm-started, DSA/Krylov accelerated; or fixed linear iteration) — **REVISE**
- **Original mechanism:** Converged to 1e-6 relative, or a fixed linear iteration. Warm starts (AR l.407). DSA/two-grid optional. Truncated Krylov withdrawn as non-linear (SC l.57).
- **Identified defect:** (a) A 1e-6 residual with condition number about 39 (red) gives a solution error of about 4e-5, which fails the 1e-6 superposition test (BLOCKER BL-4, Part B8). (b) A warm start breaks zero-in-zero-out. (c) A non-monotone accelerated iterate can push tiny K entries below -1e-9·max (BL-5, Part B12). (d) The fp32 residual floor is about 2e-6 for red (Part E2). (e) CG needs symmetry, which was given up. (f) DSA on 25:1 cells is unverified [PA Warsa-Wareing-Morel 2004: DSA degrades in multi-D with discontinuities].
- **Evidence or derivation:** Parts B8, B11, B12, E2 [ND/PA].
- **Proposed change:** Contract mode: cold-start fixed-count Neumann/Jacobi (about 663 red sweeps for kappa·tau <= 1e-6, about 525 for rho^k <= 1e-6) with float64 accumulation. This is exactly linear, monotone, non-negative, passive and D4h-equivariant. An accelerated mode is allowed for speed studies only, labelled 'linear to tolerance', and never used in contract tests. No warm starts in any test.
- **Why the change addresses the defect:** Only the fixed cold-start iteration satisfies all the invariant tests at once (Part B11).
- **Preserved invariants:** Linearity, zero-in-zero-out, non-negativity, passivity (monotone underestimate), equivariance.
- **Additional assumptions:** Boundary sources may converge faster than the fundamental-mode estimate [UH].
- **Computational costs:** About 0.7-1.0 s per single-light RGB application (180-260x the measured FIA 3.9 ms) [ND estimate, Part E3]. This decides deployability negatively for the contract mode.
- **New failure modes:** Contract mode is too slow for any deployable role. Accelerated mode is not invariant-exact.
- **Falsification condition:** Superposition and zero-input tests at 1e-6. Minimum entry >= -1e-9·max.
- **Closest prior art:** Source iteration, DSA (Khalil 1988; Larsen), Krylov-DSA (Warsa et al. 2004) [PA].
- **Novelty implications:** None.

### E14 Exact D4h equivariance via orbit-representative permutation assembly — **REVISE**
- **Original mechanism:** Matrices assembled from orbit representatives by permutation; otherwise about 1e-7 (SC l.114) [WD].
- **Identified defect:** (a) Incompatible with the archived random B (moved to E7). (b) An orbit-symmetric direction set has nodes on bin boundaries (e.g. the normal maps to u = v = 0.5 -> bin (4, 4)), which breaks equivariance (Part B10, ex. 2). (c) Summation order differs between u and g·u, so equivariance holds only to roundoff (Part B10, ex. 3). (d) A far-field grid not commensurate with 2.5 mm breaks translation invariance (Part C2).
- **Evidence or derivation:** Part B10 [ND]; p53_geom.py:35-56 (concentric map equivariant, floor(u·N)) [HM].
- **Proposed change:** Use offset stratification (no node on a bin edge) or split edge-node weights equally. State a test tolerance (about 1e-7 fp32 / 1e-15 fp64). Remove the far-field grid (E4).
- **Why the change addresses the defect:** Exact symmetry is a cheap correctness check of the assembly, not an accuracy claim.
- **Preserved invariants:** D4h equivariance at rest; rigid invariance in the body frame [EP].
- **Additional assumptions:** Dented poses have no symmetry claim (brief l.69).
- **Computational costs:** Negligible.
- **New failure modes:** None beyond roundoff.
- **Falsification condition:** Max |K(g·u) - g·K(u)| above the stated tolerance.
- **Closest prior art:** Symmetric quadrature sets in S_N/MOC [PA]. Not distinctive: any de-identified equivariant model shares the translate/orbit identity (WL77A H1 ground, coordinator_decision.md l.13) [HM].
- **Novelty implications:** None.

### E15 Discrete reciprocity (given up; defect reported) — **REMOVE**
- **Original mechanism:** Exact volume-weighted reciprocity given up for exact partition; defect reported on aggregated 16x16 blocks (AR l.304; FD l.272) [WD].
- **Identified defect:** Three sources of defect: quadrature, F_in vs Phiout asymmetry, and a structural O(h) defect because order >= 2 is exact at the source end and flat-cell at the receiver end (about 2 % at grazing bins) [ND, Part B9]. Reciprocity is not a contract invariant (brief l.70).
- **Evidence or derivation:** Part B9 [ND]. fine16 nesting makes the aggregated test well posed [HM].
- **Proposed change:** Remove any reciprocity claim. Report the aggregated defect only as a last-flight-bias diagnostic.
- **Why the change addresses the defect:** It cannot be restored within C1 without a symmetric last flight, which would be a different (H3-like) design. The contract does not require it.
- **Preserved invariants:** Not applicable.
- **Additional assumptions:** Rest ports have equal area and every 16-level bin has projected solid angle pi/16, so the aggregated K is symmetric for exact transport [ND].
- **Computational costs:** None.
- **New failure modes:** None.
- **Falsification condition:** Not applicable (diagnostic).
- **Closest prior art:** CP reciprocity relations [PA]. H3 had exact reciprocity (candidate_H3.md l.123) [HM], so on this point C1 is a regression.
- **Novelty implications:** None.

### E16 Geometry-dependent state and rigid / albedo edit contract — **REVISE**
- **Original mechanism:** Rigid: nothing to rebuild. Albedo edits free because the factors are channel-free (CR l.181; SC l.118; XR l.932, l.1010) [WD].
- **Identified defect:** True for the K factors only. As an edit contract it omits that C, U, S and GI are invalidated under rigid motion, that optical contact moves the edit into the topology class, and that an albedo edit invalidates every K column and its convergence certification (Part D2 cases 1, 5; D4 items 5, 6) [ND].
- **Evidence or derivation:** Part D2/D4 [ND/EP]. WL59 l.219: rigid frame/state reuse supported for the learned model [HM].
- **Proposed change:** State the full contract per edit class: rigid -> C, U, GI rebuilt, K factors reused bitwise in the body frame; albedo -> factors reused, global re-solve, refinement certification per omega; sigma_t, thickness, topology, contact -> full rebuild and new port correspondence (none specified; Architecture.md l.58 requires it).
- **Why the change addresses the defect:** The record's claims are correct for K but incomplete for the dynamic-GI contract.
- **Preserved invariants:** Intrinsic K invariance in the body frame [EP].
- **Additional assumptions:** The body is separated by air.
- **Computational costs:** Rigid: zero for K factors. Albedo: one global solve per lighting state.
- **New failure modes:** Port-set changes (thickness, topology, contact) have no correspondence semantics.
- **Falsification condition:** Rest vs rigidly moved K bitwise equal in the body frame.
- **Closest prior art:** Precomputed transfer reuse for lighting at fixed geometry (PRT; Hasan et al. 2006) [PA].
- **Novelty implications:** ALREADY KNOWN.

### E17 Dent rebuild: vertical stretch map, influence set, local ray re-integration, global re-solve — **REVISE**
- **Original mechanism:** Same topology under a vertical stretch map (no cut cells). F_in, W1, W and Phiout are ray-integrated on the true mesh. The influence set includes bottom, side and rim sources. Then a global re-solve. 1-5 s on the fly (AR l.322-330, l.408; SC l.113, l.120) [WD].
- **Identified defect:** (a) The full-depth stretch moves every cell in the footprint columns, including the bottom layers, which compress to about 40 %: a representational change wider than the physical one, and a spurious rest-vs-dent bias difference (Parts D2 case 2, D4 item 4; XR l.648-651). (b) Bottom beams change physically: beta rises from e^-10 to e^-4 (about 400x) [ND]. (c) Dirty beams are 26-42 % of all beams, so 'local' means about a third of construction [ND, Part E6]. (d) The 1-5 s figure rests on an unbuilt deterministic quadrature; the sampled analogue took 75.0 s, 74.3 s of it quadrature (76_* l.252) [HM]. (e) Exact local resolvent updates (Woodbury, rank 2e4-6e4) cost more than one global solve, and truncated low-rank updates break W >= 0 (Part D2).
- **Evidence or derivation:** Parts D2, D4 and E6 [ND/HM].
- **Proposed change:** Confine the stretch map to layers above a fixed depth so that unmoved bottom/side layers stay bitwise identical [ND suggestion; effect on signed change UH]. Define the G-rebuild set as cells/beams within r_f (about 7-14 mm) of moved geometry, plus all bottom/side/rim beams whose chords meet the thinned region. Always re-solve globally from cold start. Drop the 1-5 s claim until a kernel is measured.
- **Why the change addresses the defect:** This removes avoidable spurious change and states the true invalidation set. It is still a numerics revision, not an architectural one.
- **Preserved invariants:** Ownership, partition, passivity, D_T != D_G handled by the global solve.
- **Additional assumptions:** Rest/dent bias cancellation in the signed change [UH, SC l.121].
- **Computational costs:** About one third of full construction per dent, plus a full solve [ND]. The WL59-style pipeline geometry/B floor of about 0.32 s is inherited (59_* l.182) [HM].
- **New failure modes:** Signed-change relative error 0.1-0.3 vs the 0.35 gate [UH]. Cavity self-coupling in C adds an outer loop (Part D2).
- **Falsification condition:** Rest-mesh vs stretched-mesh null test (FD l.315). Full-rebuild control. Dent signed change rel. error > 0.35 or corr < 0.90 in any channel.
- **Closest prior art:** Incremental link update + global re-solve (Drettakis-Sillion 1997); CP generalized perturbation theory [PA].
- **Novelty implications:** ALREADY KNOWN.

### E18 Joint K∘C unknowns / GI coupling form — **UNRESOLVED**
- **Original mechanism:** Collision densities may be joint unknowns of the outer K∘C solve (CR l.86-89). The one-W-per-outer-iteration cost claim was withdrawn; nested warm-started inner solves of 1-10 s are proposed (AR l.405-407; SC l.55).
- **Identified defect:** The joint form inherits rho >= 0.974. No coupled acceleration is designed. Warm starts break the linearity tests (E13). It is never stated whether joint unknowns remain permitted (Part A ambiguity 10).
- **Evidence or derivation:** Parts B11, E2 and E4 [ND]. Coupled stability holds: joint columns sum <= omega < 1 (Part B11) [ND].
- **Proposed change:** None proposed. Nested cold inner solves are the only invariant-exact form. A coupled acceleration would be new design work outside the C1 records.
- **Why the change addresses the defect:** There is no evidence base for choosing.
- **Preserved invariants:** Coupled stability (ND).
- **Additional assumptions:** Not applicable.
- **Computational costs:** Nested: 51 x (17-95 ms) ≈ 0.9-4.8 s per GI solve on the WL70-type fixture [ND estimate from XR l.912, not re-verified].
- **New failure modes:** GI cost above the class-3 path-traced re-render (1.2-2.4 s, different workload, XR l.914-915).
- **Falsification condition:** Not applicable until designed.
- **Closest prior art:** Coupled zonal/exchange-factor solves; response-matrix iteration (COMET class) [PA].
- **Novelty implications:** None.

### E19 Learned components (optional ΔW closure; optional learned preconditioner) — **REMOVE**
- **Original mechanism:** Optional ΔW (about 1e3 parameters, symmetric non-negative redistribution, trained on canonical-patch solves) in (i); dropped in (ii) (AR l.331). An optional learned preconditioner affecting speed only (SC l.102).
- **Identified defect:** ΔW reproduces WL77A H3's learned closure, whose 'vacuous closure' (Horn A) ground is unrefuted (coordinator_decision.md l.15) [HM]. A learned preconditioner inside the fixed-iteration contract mode (E13) would make K depend on a learned component (Part C2(3)). At convergence it is irrelevant to the operator.
- **Evidence or derivation:** WL77A coordinator decision [HM]; Part C2(3), Part F1 [ND].
- **Proposed change:** Remove both. No learned component is added to C1.
- **Why the change addresses the defect:** Adding learning only to satisfy a preference is forbidden, and in this physical scope a closure has nothing to learn that exact physics does not already supply (SC l.135, conceded).
- **Preserved invariants:** No fitted quantity (CD LOW-9).
- **Additional assumptions:** Not applicable.
- **Computational costs:** None.
- **New failure modes:** None.
- **Falsification condition:** Not applicable.
- **Closest prior art:** WL77A H3; learned preconditioners in scientific computing [PA].
- **Novelty implications:** Not applicable.

### E20 Tabulated K cache (optional; columns cached for lighting reuse) — **UNRESOLVED**
- **Original mechanism:** About 10 ms per application but minutes per dent (SC l.129; AR l.409; XR l.972) [WD].
- **Identified defect:** Dense K is 181 GB. A truncated 128-NN + tail cache is about 8.9 GB, lossy (128 NN alone drops 5.5 % of interior red energy, brief §3a [HM]), and cannot be co-resident with 2.5-6.8 GB of factors on 12 GB. Break-even is about N_cols lighting states (about 4e3 at rest with D4h storage, about 6e4 otherwise). It never amortises under per-frame deformation (Part E8) [ND].
- **Evidence or derivation:** Parts E1 and E8 [ND].
- **Proposed change:** None. Whether a cache has any role is a deployability decision outside the research question.
- **Why the change addresses the defect:** Caching is the PRT-class reuse already known (Part F P9) and adds no architectural value.
- **Preserved invariants:** A truncated cache loses passivity guarantees only if renormalised improperly [ND].
- **Additional assumptions:** Not applicable.
- **Computational costs:** Tabulation 1-15 min general, 4-56 s at rest [ND estimate].
- **New failure modes:** Stale dirty-column retabulation under the global resolvent.
- **Falsification condition:** Not applicable.
- **Closest prior art:** PRT / direct-to-indirect transfer (Hasan-Pellacini-Bala 2006) [PA].
- **Novelty implications:** ALREADY KNOWN.

### E21 Role and identity claims (selected primary; 'unseen-source generalisation becomes a convergence question checkable without references'; cost estimates) — **REMOVE**
- **Original mechanism:** 'Selected primary' (SC l.5). Reframing claim (SC l.150). Cost estimates of 0.05-0.8 s per application, 1-10 s per GI solve and 1-5 s per dent (SC l.126-130) [WD].
- **Identified defect:** (a) CD l.17 already qualifies the role to a candidate operator / diagnostic instrument. (b) Self-convergence is reference-free, but accuracy is not: a convention mismatch converges to the wrong answer (Part C3). Source-independence is just linearity plus an exact first flight, true of any deterministic linear solver and of the WL56 MC oracle (Part F2, C-g). (c) The cost figures rest on unbuilt kernels and assumed acceleration (CD l.21). Measured anchors favour the learned operators by about 4-260x per single application [ND estimate vs HM anchors, Part E3].
- **Evidence or derivation:** CD l.15-18, l.21 [HM]; Parts C3, E3, F2 [ND].
- **Proposed change:** Replace with: 'learning-free deterministic CP/FCS integral-transport operator; deterministic analogue of the WL56 MC oracle K (56_* l.30); diagnostic instrument only; admissibility as an FNTO K and every value comparator (Neural Full Recompute / Native Reuse, internal F/S, class-3 re-render) are human decisions; no cost is measured'.
- **Why the change addresses the defect:** These were the overclaims identified by the coordinator and by Parts C, E and F.
- **Preserved invariants:** Not applicable.
- **Additional assumptions:** Not applicable.
- **Computational costs:** Not applicable.
- **New failure modes:** Not applicable.
- **Falsification condition:** Not applicable.
- **Closest prior art:** WL77A H3 θ=0 class (CD l.18) [HM].
- **Novelty implications:** Not novel (Part F0).
