# C1 representation and source generalisation (Part C)

> WL80B record (workflow `wf_9bbd762a-ae2`, 12 agent runs). Agent texts reproduced verbatim; headings added by the Coordinator. Design-only: nothing implemented, executed, trained or measured. The original C1 architect was not resumable; a stand-in reconstructed C1 from the sealed WL80A records (immutable inputs, sha256 in `inputs.json`).

# WL80B Part C: representation and source generalisation in C1 (CP-K)

## Summary

C1 has no learned or fitted part and no supervised set of sources. It computes every source column from geometry, so "generalisation" to unseen sources is the wrong frame for it.
- **At the continuum level**, source independence is exact and structural [EP]. The source enters only through the exact first flight.
- **Exact translation and D4h invariance of the discrete scheme at rest** is structural [ND]. It needs a port-aligned mesh, permutation assembly and a far-field grid that lines up with the 2.5 mm port pitch.
- **Accuracy at any particular source context** is a numerical question: discretisation, quadrature and solver convergence, with errors that depend on the context. Its size is unmeasured.
- **Nothing is learned**, unless a learned preconditioner is used without converging or C1 serves as a teacher; both are excluded.

So the right label is "source-uniform accuracy of a classical deterministic solver", conditional on the physics premises, a matching reference convention, and an affordable resolution. It must never be called source-generalisable neural transport. This agrees with CD l.16–18 (HIGH-2, MEDIUM-3/4), which say C1 does not address the learned-K question. It also agrees with WL77A, which rejected H3 because its transfer is carried by classical CPM (`wl77a/coordinator_decision.md` l.15).

A new derivation of the lateral flat-source bias:
- It grows the lateral diffusion coefficient by +31 % at 1.25 mm cells and +8.7 % at 0.625 mm, which is second order as h → 0.
- It mainly threatens the red abs metrics and the dent signed change, not contrast.
- The ordering of strata it predicts differs from the learned-family failure pattern.

I also read the §7 non-evidence emulation differently from CD MEDIUM-4: as I read §7's method, it does not bound C1's cumulative chain error (C5).

## C0. Basis and conventions checked

Read-only. I read:
- CR l.1–290;
- AR l.1–480;
- SC l.85–184;
- XR l.84–110 and l.425–504;
- CD (all);
- the brief (all);
- `wl77a/specialist_physics.md` l.60–418;
- `wl77a/coordinator_decision.md` (all);
- WL76 l.252, 273, 282, 305 (the 75.0 s rebuild of which 74.3 s is quadrature; the snapping of the second collision onto 2.5 mm port columns);
- two code files, for conventions only.

I wrote no files and ran no code. Every number marked [ND] was worked by hand.

**Source convention [HM, code].** A unit-flux cell (i, a) is uniform over the area of port i's triangles. Its directions are uniform in projected solid angle inside bin a, in each triangle's (t, b, n) frame. The bin labels the outward direction the light comes from, and the photon travels along −w (`experiments/phys_ref53/p53_tracer.py:90-101`; `p53_geom.py:59-65`).

**Exit tally [HM, code].** Each exiting photon's weight is added at index port·64 + bin in the triangle frame (`experiments/fnto_fia76/phys76.py:46-51`, "same port/bin mapping as the walker"). So the reference's "radiance-weighted power" is the exiting flux, binned by direction. Φ_pt and Φout must therefore be escape probabilities per (j, o) with no extra cosine factor [ND]. This settles the escape-side convention of FD l.198 / A3-14 only to the extent that the walker matches phys76, which I did not open.

## C1. The exact variables through which the source enters

The packet u = e_(i,a) affects the output of C1 only through these items:

| # | Variable | Source-dependent through | Label |
|---|---|---|---|
| S1 | β(i, a) and d = 1 − β | Port triangles, bin cell in the triangle frame, σ_t, the in-medium chord to the far boundary (e.g. edge ports whose bins point at the side face) | [EP]; exact via B |
| S2 | First-collision density q_(i,a)(r) = ∫ρ_(i,a)(x₀, ω) σ_t e^(−σ_t s) δ(r − x₀ − sω) ds dx₀ dω | The same inputs as S1. The footprint has depth ≈ cos θ mm and a lateral shift ≈ sin θ mm (0.30 / 0.74 / 0.82 mm for inner, outer mid-side and outer corner bins; specialist_physics l.90–93, non-evidence) | [EP] |
| S3 | Discrete q at fine points p, with Σ_p q = d | The placement and weights of the point quadrature for each beam | [WD] SC l.90, l.94. Placement unspecified (A3-6) |
| S4 | Φ_pt q (order-1 escape) and W1 q (cell of the second collision) | Only through q. Φ_pt[·; p] and W1[·; p] are source-free per point | [WD] SC l.95 |
| — | W, Φout, ω_c, the mesh | Not source-dependent | [WD]/[EP] |

**[ND] q is channel-free.** σ_t is the same in every channel, so q is channel-free. ω_c enters only through the solve.

**[ND] Source and receiver ends.** The receiver end (Φout, Φ_pt over (j, o)) and the source end (S1–S3) meet only through the scalar field ψ. No quantity is indexed by (i, a) and fitted.

**Per-beam versus global points.** S3 is the one place where the claim that "the first flight is exact" depends on how the record is read:
- **Per-beam template points** (AR l.294, l.408: about 1e3 points per beam): q is a quadrature of the exact density, accurate to the quadrature budget.
- **One global set** (XR l.945: about 1e6 fine points): q is projected onto that set. With uniform placement over the 1e5 mm³ slab that is about 0.1 mm³ per point, i.e. roughly 0.46 mm spacing.

  The bin-dependent footprint shift is 0.3–0.8 mm, and it carries coarse contrast (specialist_physics l.307, l.315). In the global-point reading, part of the contrast-carrying information is lost in the first flight already, with an error that depends on the source's position relative to the points [ND]. The records do not choose between the two readings.

## C2. Five layers, kept separate

### (1) Continuum source independence: exact and structural

Under g = 0, a homogeneous medium, an index-matched boundary, a σ_t common to all channels and first-exit ownership:

K_c(X → Y) = ω ⟨P_Y, G_c q_X⟩, with G_c = (I − ω_c 𝒮)⁻¹ independent of the source [EP; specialist_physics l.118–129; CR l.14–20].

**[ND]** Any exact linear operator "generalises" to every column, because each column is just the operator applied to that input. No inference from supervised columns takes place. So at this level "source generalisation" says nothing beyond linearity plus an exact first flight. It is not a property any learned family has to reproduce.

**Conditions.** This holds only if:
- the reference realises this physics (H-function agreement ≈ 0.5 %, brief l.39 [HM]);
- the reference uses this source and exit convention (C0).

A mismatch in convention would shift *every* column, including columns that would have been "supervised" in a learned setting.

### (2) Discretisation and quadrature: the real carrier of context-dependent error

1. **The cell chain from the second emission onward.** Each emission is re-randomised uniformly inside its cell. This inflates lateral spread (C4.1). The error depends on context through:
   - the number of flat re-emissions (about 22 weighted collisions for interior red versus 8.6 for the top-edge row; specialist_physics l.82, non-evidence);
   - the local cell size, which is set by the grading (C4.4).
2. **Depth flat-source error in graded layers.** It is small near faces: the developer's estimate is about 2 % at μ_out = 0.05 with 0.1 mfp layers, and smaller with 0.05 mm layers (AR l.393; SC l.155) [WD].
3. **Shape of Φout over (j, o).** It is quadrature-defined, while the cell total is the complement (SC l.97) [WD].
4. **Quadrature of the per-beam templates.**
   - "One shared stratified isotropic ray set per point" (SC l.95), at about 64 × a few rays per point (AR l.408).
   - [PA] If the same direction set is used at every point, it is a discrete-ordinates-style angular quadrature and is exposed to ray effects (Lathrop, NSE 32:357, 1968). The aliasing pattern depends on where the source sits relative to the receivers, so it is source-dependent and is not removed by spatial refinement.
   - [ND] Per beam, about 1e3 points × about 200 rays ≈ 2e5 rays in total.
   - [HM, non-evidence] For comparison, WL77A's one-ray SS at q = 65,536 already sat near its noise-free floor: ring 0.227 vs about 0.22 (`coordinator_decision.md` l.38–39).
   - [UH] The budget is plausible if the direction sets are rotated per point. That is unverified.
5. **The precorrected-FFT far field.** [EP] With an exponential flight law, a fraction e^(−3) ≈ 5.0 % of flights are longer than the 3 mfp near-field radius. [ND] Over about 20 red steps, most histories use the far field at least once. Exact translation invariance on the port lattice then also needs a far-field grid commensurate with 2.5 mm. Otherwise invariance holds only to the far field's interpolation accuracy, which varies with where the source sits relative to the grid. The positivity amendment (FD l.195) is a separate requirement.

### (3) Solver convergence: numerical, mildly source-dependent, not learned

- **Plain sweeps.** ρ_red ≈ 0.974 gives about 524 sweeps to 1e-6 (CD l.21) [WD; ND re-check: ln 1e-6 / ln 0.974 = 524.5].
- **Converged per right-hand side.** The error falls below tolerance for every column, so it carries no source-specific structure.
  - Per-RHS stopping and warm starts make the operator linear only to the tolerance (AR l.442) [WD].
- **Fixed linear iteration.** The truncation remainder follows the column's content in the slow mode: interior red is worst, edge and corner sources are better. It is a one-sided underestimate, so passivity is safe [ND]. Its source dependence is spectral and deterministic.
- **A learned preconditioner** (SC l.102) is harmless only at convergence. Inside a fixed-iteration form it would make the operator depend on a learned component [ND]. It must be excluded or converged.

### (4) Geometry-dependent rebuild: computed, not inferred

- **Rest pose.** Each face's translation templates plus D4h permutation assembly give equal discrete error for all equivalent contexts [ND, conditional on point 5 of layer (2)].
- **Dent.**
  - F_in changes, because the per-triangle frames and the surface move.
  - W1, Φ_pt, W and Φout are ray-integrated on the stretched mesh, and the influence set includes bottom, side and rim sources (SC l.120) [WD].
  - Accuracy for sources near the dent is again a discretisation question.
  - The vertical stretch changes depth grading but not lateral grading (XR l.104). The bias at rest and the bias when dented are then different functions of position (C4.6).
- **Rigid motion.** C1 is exact in the body frame [WD; EP].

### (5) Learned or compressed state

- **Amended C1:** none. ΔW is dropped (AR l.331) [WD].
- **The precorrected FFT** is an approximation of the kernel. It is source-free, but the clamp may bind depending on context.
- **The optional tabulated cache** (AR l.409) compresses computed columns by truncation. It is not trained on a subset of columns.
- **Teacher use** (AR l.479) would move generalisation into the learned student. That is a different architecture and is not part of C1.

## C3. Classification

| Component | Structural | Numerical | Learned |
|---|---|---|---|
| Exact continuum operator; source only through q | yes [EP] | — | — |
| Identical outputs on translate and D4h images (e.g. BND4_FAR s101 / s105 / s107 ≅ s50 / s52 / s53; specialist_physics l.194–204; their references agree within noise per `coordinator_decision.md` l.65–73 [HM, non-evidence]) | yes, if the mesh, far-field grid and assembly are lattice-commensurate [ND] | otherwise to the far-field accuracy | — |
| Accuracy at out-of-orbit contexts (s103 corner offset, corners, vertical edges, side middle, dent) | — | yes: grading, quadrature, far field, solver | — |
| Contrast (con_*) | mostly structural, via the exact q and W1 | order ≥ 3 escape is cell-averaged | — |

**[ND] Verdict:** structural plus numerical, with no learned part.

**Correcting a WL80A claim.** SC l.150 says unseen-source generalisation "can be checked without references". Self-convergence can be checked without references. Accuracy cannot, because a converging solve converges to the wrong answer under a convention mismatch. This is consistent with CD HIGH-1 / HIGH-2.

**[ND] The translate identity is not specific to C1.** It is the property that any de-identified equivariant model also gets. This was WL77A H1's decisive objection (`coordinator_decision.md` l.13), so it does not set C1 apart as a learning representation.

## C4. Counterexamples, worked quantitatively (all [ND]; magnitudes for gate metrics are [UH])

### C4.1 Lateral flat-source bias in the cell chain

**Model.**
- Product cells of lateral size h.
- Each emission is uniform inside its cell, independent of the history.
- An isotropic exponential flight (ℓ = 1 mm). Its 1-D lateral projection f has variance 2/3 and characteristic function φ(k) = arctan(kℓ)/(kℓ) [EP].

**Derivation.**
- For one flight with f = mh + δ, the cell displacement has mean f and variance δ(h − δ). So the extra variance per step is E[δ(h−δ)].
- By Fourier series: E[δ(h−δ)] = h²/6 − (h²/π²) Σₙ φ(2πn/h)/n².
- Steps are i.i.d. laterally in the interior, so this extra variance is a lateral diffusion inflation.
- Decay length L ∝ √D_lat (diffusion scaling, κ² ≈ (D_z/D_lat)k_z², XR l.452).

| h_lat (mm) | Extra variance per step (mm²) | ΔD_lat / D_lat | ΔL / L (if D_z is unchanged) |
|---|---|---|---|
| 0.5 | 0.038 | +5.7 % | +2.8 % |
| 0.625 (chair's L1) | 0.058 | +8.7 % | +4.3 % |
| 1.0 | 0.139 | +20.8 % | +9.9 % |
| 1.25 (amended C1, L0) | 0.208 | +31.1 % | +14.5 % |
| 1.6 | 0.320 | +48 % | +22 % |

- As h → 0 the inflation tends to h²/4, so the order is 2 (observed order from 1.25 to 0.625 ≈ 1.8). This sharpens XR l.449 / CD LOW-11 (h²/8–h²/4; 9–18 % in length) and sits at their upper end.
- **Offset from depth grading.** Graded depth cells near the mid-plane are about 0.8–1.0 mm thick: about 1.5e5 face cells ⇒ about 23 z-layers, with 1.1 mm in the 7 specified layers per side (AR l.342). Inflating D_z partly cancels the lateral bias. The net is plausibly +8 to +15 % in L at L0 [UH].
- **The order-2 rate is a diagnostic.** Under CD HIGH-1, a red miss at L0 that converges at an observed order of about 2 is "affordability-limited", not a plateau.

**Gate impact, red interior [UH, crude model].**
- Assumptions:
  - a lateral exit profile ∝ K₀(ρ/L), with L = 3.57 mm (specialist_physics l.73);
  - the source coarse cell treated as a disk of radius 5.64 mm;
  - about 25 % of red exit taken as cell-exact (order 1 / 2).
- Results:
  - the source cell's multiply-scattered share drops from 0.61 to 0.54 at ΔL/L = 0.145;
  - abs_coarse is roughly 0.04–0.08 at L0, against the 0.10 gate;
  - it falls to roughly 0.01–0.02 at L1.
- The bias is almost common to all bins, so con_coarse is largely unaffected (consistent with XR l.101).
- Blue has fewer re-emissions (about 5) and stays mostly inside the source's coarse cell, so its coarse impact is small.

**Memory at L1 [ND, from WD estimates].** Halving the lateral spacing multiplies cells by 4. With a fixed 3 mfp near-field radius, partners also go up 4×, so the near field grows 16× to about 1.6e9 non-zeros (≈ 13 GB at about 8 B each). That exceeds 11 GB, so L1 is reachable only with a matrix-free near field (cf. XR l.459).

### C4.2 Interior translated source (INT)

- The error is identical for every top-face interior port under port-lattice translation [ND, conditional on C2(2) point 5].
- So INT behaves as a single context class for C1. Its error is dominated by C4.1, and red is the worst channel.
- **[UH] Predicted ordering.** INT red abs ≥ edge-row red abs, because the edge row has fewer re-emissions (8.6 vs 22.3) and a more compact response (99.1 % within 15 mm; specialist_physics l.219–221). That is the reverse of the learned-family pattern (brief l.127–128).
- SD l.177(d) (BND_FAR ≤ max(1.25·INT, INT + 0.03)) would then test how uniform the mesh is, not transfer.

### C4.3 Top-edge row (TBE, s101 type) and BND_NEAR

- **TBE.** About 41 % of collided exit leaves through the side face (specialist_physics l.88, non-evidence).
  - Under a tensor-product reading of "product grading in the cross-section", the edge cross-section is fine in both directions. The along-edge spacing is 1.25 mm, with fewer re-emissions.
  - s101's error then equals s50's exactly.
  - No extrapolation is involved.
- **BND_NEAR (1.8–4 mm from a perpendicular face).** These sources sit where the side-face depth grading changes from about 0.4 to 1.25 mm, and the records do not specify cell sizes there.
  - Their side-exit fraction is decided by cells in that transition zone. The error ordering across strata could therefore be non-monotone [UH].
  - The "1.8 mm" definition does not land on a port centre (1.25 + 2.5k mm; SC l.158) [WD].

### C4.4 Side middle (SM, s105) and side rim (SR, s107)

- [ND] In a tensor-product mesh, the side face's lateral z-spacing *is* the top/bottom depth grading:
  - about 0.05–0.4 mm near the rims;
  - about 0.8–1.0 mm near the mid-plane.
- So SM (z = ±1.25) sees z-cells of about 0.8–1.0 mm, i.e. ΔD of about +13 to +21 % along z, and 1.25 mm along y (+31 %).
- SR (z = ±3.75) sees fine z-cells.
- Over-spreading in z increases leakage to the top and bottom faces, which carry 0.20 / 0.18 / 0.11 of side-source exit (specialist_physics l.88, non-evidence). Size [UH].
- The records also allow a non-conforming mesh with its own side grading (AR l.343: "same grading"). This is unresolved and decides the SM error class.

### C4.5 Corner-offset s103 and true corners

- **s103** is 6.25 / 8.75 mm from two edges (specialist_physics l.199). It is not an orbit image of any S16-R source.
  - For C1 it is a two-face context: its source cells are 1.25 mm, and the edge exits at about 2 L_red pass through graded edge zones.
  - The error equals the discretisation error of that geometry. It is not an extrapolation.
- **True corners.** No record specifies three-way grading (SC l.109–110 covers faces and edges only). The tensor-product reading supplies it implicitly.
- **Grazing incoming bins at corners.** The order-1 escape and the order-2 collision cell are exact (W1). Bin information below the cell scale is lost only from the third emission onward [WD AR l.454–456; ND agrees]. Whether con_coarse ≤ 0.25 holds there is [UH] (SC l.156).

### C4.6 Dent

- The lateral inflation ratio depends only on h/ℓ, so it is about the same at rest and dented.
- The vertical stretch compresses mid-plane layers to about 40 % (XR l.104), so D_z inflation under the dent is smaller (about 0.3–0.4 mm cells, ΔD_z of a few %) than at rest (about 13–21 % at the mid-plane).
- So the net lateral-length bias differs by a few percent of L inside the dent footprint (thinned region L ≈ 1.7 mm).
- If the response change is mainly a change in L, the relative error of the signed change is of order ε_L or larger: plausibly 0.1–0.3 against the 0.35 gate [UH].
- This is geometry-dependent discretisation error, not source generalisation.

### C4.7 Translate in a context the records did not specify

On a translate that does not land on the port lattice (any new geometry, or a dent-adjacent port), the template, far-field and mesh phases differ. C1 then shows its full context-dependent quadrature error, with no symmetry protection [ND].

## C5. The WL77A specialist §7 emulation (non-evidence), read for C1

Source: `docs/plans/wl77a/specialist_physics.md` l.293–308 [HM, labelled non-evidence at its source]. Its rows:
- red energy bias +2.1 / +6.0 / +17 % at 0.5 / 1.0 / 2.5 mm voxels;
- lateral snap at 2.5 mm with exact depth: interior 0.096 native / 0.065 coarse; edge 0.20 (−11 %); side 0.36 (−23 %); pair contrast 0.05–0.08.

**[ND] The emulation is not a conservative random walk.**
- Interior red collided exit is 0.978 (l.80).
- +6 % and +17 % would give 1.037 and 1.14, above the passivity bound (output ≤ d ≤ 1).
- So the represented field was re-emitted in a single stage that does not conserve probability. That fits l.295, which re-emits from the represented density and scores with the collision estimator.
- I could not verify this: the scripts are outside the repository.

**Consequences.**
- (a) The energy-bias column does not predict amended C1. Its escape is defined by the complement, so energy can deviate from ω·d only through (1−ω)·1ᵀψ. For red that is ≤ 0.1 % per collision of count error [ND].
- (b) §7 does not cover the **cumulative** lateral inflation of the W chain (C4.1), which is C1's main lateral mechanism. CD MEDIUM-4's "already bounds the mechanism" is therefore too strong for the chain error. §7 bounds the exit-projection part (the Φout type) at best [ND; interpretation conditional on the reading above].
- (c) The edge and side rows (−11 / −23 %) come from depth misplaced relative to the adjacent face (l.308). Face-relative product grading addresses this [WD]. Whether the tensor reading is what was intended is C4.4's ambiguity.
- (d) The contrast row supports "contrast survives coarsening of k ≥ 2" (l.307). C1 keeps the second-collision cell exact, so its contrast loss should be no larger than in the snap row (0.05–0.08 at 2.5 mm) [UH, extrapolated from one interior pair].

## C6. Corrections and additions to Part A and the WL80A wording

1. **CR l.206 / SC l.150.** "The residual error does not depend on which sources were supervised" is true but vacuous, since nothing is supervised. The error does depend on source *context* (C4).
2. **Part A ambiguity 6** (per-beam versus global points) is not a cost detail. It decides whether the first flight is exact (C1, S3).
3. **Part A ambiguity 7.** The per-point ray set needs per-point rotation or another anti-ray-effect measure. Otherwise angular aliasing is a source-dependent, non-refining error [PA Lathrop 1968; ND].
4. **Part A ambiguity 4 / FD l.256.** The precorrected-FFT grid must be commensurate with 2.5 mm for exact translation equivariance, in addition to positivity [ND].
5. **Part A ambiguity 12.** Where the 1.25 mm row was missing, C4.1 now gives the chain-inflation version (+31 % D_lat; +8 to +15 % L). The §7 table should not be used to fill that row (C5).
6. **Part A is otherwise consistent** with the records I checked: CR l.144–153, l.202–206; AR l.286–353; SC l.90–115; CD l.15–25.

## C7. Unmeasured

Every C1 accuracy number for every stratum is unmeasured, including the abs_coarse estimates in C4.1, which come from a crude diffusion/K₀ model. Also unmeasured:
- the sign and size of the Φout shape bias (A3-11);
- the mesh at the side middle and corners;
- the far-field grid;
- the ray-set design;
- the dent differential bias;
- whether the walker's tally matches phys76's.

The physics premises ([EP], CR l.12–17) limit all of this to g = 0, a homogeneous medium and an index-matched boundary.

Sources:
- [Lathrop, "Ray Effects in Discrete Ordinates Equations", NSE 32 (1968), ANS record](https://www.ans.org/pubs/journals/nse/article-20218/)
- [ANS mirror](https://wx1.ans.org/pubs/journals/nse/a_20218)
