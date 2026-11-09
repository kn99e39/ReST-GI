# C1 physical correctness audit (Part B)

> WL80B record (workflow `wf_9bbd762a-ae2`, 12 agent runs). Agent texts reproduced verbatim; headings added by the Coordinator. Design-only: nothing implemented, executed, trained or measured. The original C1 architect was not resumable; a stand-in reconstructed C1 from the sealed WL80A records (immutable inputs, sha256 in `inputs.json`).

# WL80B Part B: physical correctness audit of the reconstructed C1 (CP-K)

## Scope and labels

This is a read-only design audit. I wrote no files, ran no code and made no measurements. Every number below that is not cited to a file is my own hand arithmetic.

**Files read.**
- WL80A: `trackB_candidate_records.md` l.1–300, `trackB_adversarial_review.md` l.1–480, `trackB_selected_candidate.md` (full), `coordinator_disposition.md` (full), `neutral_problem_brief.md` (full), `final_architecture_decision.md` l.185–520.
- WL77A: `coordinator_decision.md` l.10–20 and `specialist_physics.md` l.285–310.
- Code, for conventions only: `experiments/phys_ref53/p53_tracer.py`, `phys_ref53/p53_geom.py`, `fnto_rule57/r57_geometry.py`, `fnto_rule57/r57_fastwalk.py`, the `fnto_walker71c/engine71c.py` header, `fnto_rank66/ports66.py` and `fnto_fia76/fia76.py`.

**Labels.**

| Tag | Meaning |
|---|---|
| [EP] | Established physics |
| [PA] | Prior art |
| [HM] | Historical measurement or code fact, with file and line |
| [WD] | WL80A design claim |
| [ND] | New derivation (mine) |
| [UH] | Untested hypothesis |

**Short names.** CR, AR, SC, FD and CD are as in Part A. "Brief" is `neutral_problem_brief.md`.

## B0. Verdict summary

The amended C1 is used as recorded (SC l.88–115, plus the chair's FD l.194–199 and CD l.20).

| Invariant | Original C1 (CR) | Amended C1, as recorded | Status of the amended C1 |
|---|---|---|---|
| Ballistic / scattered disjointness | holds | holds | ESTABLISHED, given a rule for missed or wrong-side B rays (B4) |
| No multiple counting of paths | **violated**: first collisions are counted twice or lost | holds | ESTABLISHED (B5) |
| Passivity ΣK ≤ ω_c·d | **violated** (B3) | holds at convergence | CONDITIONAL. It is a **BLOCKER-S** for the far-field-FFT variant until the clamp policy is fixed (B2) |
| Energy closure β + exits + absorbed = 1 | broken by the split | holds | ESTABLISHED under the B2 conditions (B7) |
| Entrywise non-negativity | holds | holds for exact W ≥ 0 and an exact or monotone solve | **BLOCKER-S** as specified: a float32 FFT and a non-monotone accelerated solve at 1e-6 cannot meet the −1e-9·max test (B1) |
| Linearity, zero in gives zero out | — | holds for a fixed linear iteration with a cold start | **BLOCKER-S** as specified: warm starts and a 1e-6 residual tolerance are inconsistent with a 1e-6 superposition test (B8) |
| First-exit ownership at the dent | chord band | holds for full-chord ray-telescoped W | **BLOCKER** for the recorded far-field-FFT route on the non-convex dented pose (B6) |
| Exact D4h equivariance | claimed | claimed | **BLOCKER-S**: incompatible with "Σq = d against the archived B" (B10). Otherwise equivariant to roundoff |
| Discrete reciprocity | claimed | given up | NOT ESTABLISHED. It is a BLOCKER only for any reciprocity claim; the contract does not require it (Brief l.70). It also has a structural part (B9) |
| Solve stability | — | ρ(ωW) ≤ ω < 1 when W ≥ 0 and 1ᵀW ≤ 1 | ESTABLISHED. Iteration counts are large (B11) |

**What BLOCKER-S means.** It is a specification-level blocker: the invariant cannot be established for the design as written in the records. A stated amendment removes it (section B13).

**Physics-level conclusion [ND].** Consider the amended factorisation on the convex rest slab, with a ray-telescoped, entrywise non-negative W and a fixed number of Neumann sweeps from zero. In that configuration every contract invariant can be established exactly, except discrete reciprocity, which the contract does not require. Every BLOCKER above comes from a recorded implementation choice: the precorrected-FFT (pFFT) far field, the clamp, an accelerated or warm-started solve, or the archived B rays. None comes from the factorisation itself.

## B1. Conventions checked in code (needed for every derivation)

**Reference source** [HM, `p53_tracer.py:90–101`].
- Photons are placed uniformly on the port area (area-weighted triangle choice).
- Directions are uniform in projected solid angle inside the incoming bin, and the photon travels along −w.
- Each (port, bin) carries unit flux, with weight 1/n_per per photon.
- The WL57 walker uses the same convention: `r57_fastwalk.py:16, 32–37`.

**Reference exit tally** [HM, `p53_tracer.py:172, 190–191`; `r57_fastwalk.py:36–37`].
- The photon *weight* (flux) is accumulated into (tri_port, local_to_bin in the **exit triangle's** (t, b, n) frame).
- A wrong-side hit or a miss (leak) is dropped from the tally, and these are counted separately (`p53_tracer.py:168, 172`).
- The albedo is applied at each collision (`p53_tracer.py:196`). Russian roulette is unbiased.
- So the brief's "radiance-weighted power" (Brief l.57) is a **flux-per-flux** tally. Φ_pt and Φout must therefore tally escape *probability*, with no extra cosine factor. F_in must sample directions in a cosine-weighted way within each bin, which is what `bin_sample` does (`p53_geom.py:59–65`).

**Bin nesting** [HM, `p53_geom.py:51–56, 68–72`; e.g. `probes74.py:54`].
- Incoming bins (N = 4) and outgoing bins (N = 8) use the same concentric square-to-disk map in the same per-triangle frame. Both label the outward direction (ports66 `bin_semantics`).
- fine16 is `bin_parent(8, 4)`, an exact 2×2 nesting.
- This resolves FD C4: aggregated 16×16 reciprocity is well posed (B9).

**B is not "exactly computed"** [HM, `r57_geometry.py:80–98`; this qualifies Brief l.58].
- B is a fixed-seed (57501) Monte Carlo quadrature with q = 256 by default.
- Each ray carries exp(−σ_t·t)/q, which is analytic along the chord, but positions and directions within a port × bin are random.
- Missed and wrong-side rays add nothing to β, so they are counted inside d.
- I could not check the q actually used for the archived packets. It must be read from `meta['ballistic']['q']`.

**Medium.** g = 0 is a brief FACT (Brief l.36). The walkers sample a Henyey–Greenstein phase function with `dom.g`, so the step-0 gate should read `dom.g` from the archived reference metadata [ND recommendation].

## B2. Passivity and the albedo bound: exact conditions

**Setup** [ND, re-derived].
- Point escape: e_p = 1ᵀΦ_pt[p].
- Cell escape: e_y = 1ᵀΦout[y].
- Index convention: W[target, source], so 1ᵀW[:, y] is the next-collision mass leaving cell y.
- Operator: ψ = ωW1q + ωWψ.

**Derivation.**
- Summing the ψ equation gives 1ᵀψ = ω(1ᵀq − Σ e_p q_p) + ω(1ᵀψ − Σ e_y ψ_y). This step uses (P1) and (P2) below.
- So the total output is O = ω Σ e_p q_p + ω Σ e_y ψ_y = ω·1ᵀq − (1−ω)·1ᵀψ.
- With (P3), O = ω·d − (1−ω)·1ᵀψ.

**Exact conditions for O ≤ ω_c·d.**
- (P1) Per-point partition: e_p + 1ᵀW1[:, p] = 1.
- (P2) Per-cell complement: e_y = 1 − 1ᵀW[:, y]. This is **algebraic, whatever value 1ᵀW takes**.
- (P3) Σ_p q_p ≤ d. Equality is needed for closure, not for the bound.
- (P4) 1ᵀψ ≥ 0 for the ψ actually used.
- (P5) The output is formed from that same ψ.

**Refinement of CD l.20 [ND].**
- W ≥ 0 entrywise, together with W1, q ≥ 0 and an exact solve, is *sufficient* for (P4). It is not necessary.
- 1ᵀW ≤ 1 is **not needed for the energy identity**. It is needed for e_y ≥ 0 (non-negativity) and for the a priori bound ρ(ωW) ≤ ω (B11).

**Clamp adversarial example (the reason for the BLOCKER-S) [ND].** Suppose a far-field FFT error pushes 1ᵀW[:, y] above 1, and the escape is clamped to 0 *without* rescaling column y of W (one reading of AR l.351). Then (P2) fails, and O = ω·d − (1−ω)·1ᵀψ + ω Σ_y (1ᵀW[:, y] − 1)⁺ ψ_y. That is a net gain, so passivity is not guaranteed.

The only policy that keeps the identity is (P2*): rescale column y so that 1ᵀW[:, y] = 1, which sets its escape to exactly 0. The records do not fix the policy (Part A ambiguity 5; FD l.258), so passivity cannot be established for the far-field-FFT variant as written.

**Where the clamp binds [ND, estimate].**
- Mid-plane escape is E₂(5) ≈ e^{−5} − 5·E₁(5) ≈ 0.006738 − 0.005740 ≈ 1.0e-3 per collision.
- The free-space next-collision mass beyond 3 mfp is e^{−3} ≈ 0.050.
- So a far-field mass error of only about 2 % makes the clamp bind at interior cells.
- A trilinear-interpolated exponential kernel on a grid of about 0.5 mm plausibly errs by a few percent [UH]. The binding fraction may therefore be large, and it must be reported.

**Truncated and accelerated solves [ND].**
- Neumann sweeps from zero give ψ_k ≤ ψ. So O_k ≤ O ≤ ω·d: passive at every k.
- For a non-monotone accelerated solve, the breach margin is (1−ω)·1ᵀψ ≥ ω(1−ω)·(d − 1ᵀΦ_pt q).
- The point escape is at most 1/2 on a face, 3/4 at an edge and 7/8 at a corner [EP: the outward solid-angle fraction at a box face, edge or corner]. So the red margin is at least 1.25e-4·d at corners and about 5e-4·d on faces.
- The solution error is about κ·τ with κ ≈ 1/(1−ρ) ≈ 39. A converged solve with τ ≤ 1e-6 therefore cannot breach ω·d (39·1e-6 ≪ 1.25e-4).
- Consequence: the ω·d falsifier (FD l.267) cannot detect solver error. It detects only structural defects, and only those larger than the absorption margin (see B3).

**Tighter bounds that are also physically exact [ND].** Both hold for true transport as well, with exact point escape, and use only quantities that can be computed from geometry:

  ω·1ᵀΦ_pt q ≤ ΣK ≤ ω·d − ω(1−ω)·(d − 1ᵀΦ_pt q)

They make a stronger falsifier than ω·d alone. The decisive structural test, though, is the partition test (FD l.269), not passivity.

**ω·d is the correct physical bound** [EP]. In the continuum, collided exit = d − absorbed, and absorbed = (1−ω)(d + Ψ), where Ψ counts collisions after the first. So exit = ω·d − (1−ω)Ψ. A reference column above ω·d beyond Monte Carlo noise would indicate a reference or d-convention defect. One known source is B's q = 256 sampling error in d (B4).

## B3. The WL80A passivity defect of the original split, and the amended fix

**Defect re-derived [ND; agrees with AR l.24–36 and SC l.13].**
- The original form is K = ωS1 + ω²Φout W(I−ωW)⁻¹F_in, with S1 point-exact.
- A first collision at point p therefore gets the point escape e_p *and* the cell-flat continuation 1 − ē_cell. Their sum exceeds 1 by Δ = e_p − ē_cell.
- That excess is a probability double count: the fraction Δ of first collisions is counted both as escaped and as continuing. Where Δ < 0, mass would be lost instead.

**Values for a top face, cells of 0.5 mfp.**
- Exact beam escape: ½[1 − μ ln(1 + 1/μ)] = 0.477 / 0.424 / 0.380 / 0.321 / 0.249 / 0.153 at μ = 0.01 / 0.05 / 0.1 / 0.2 / 0.4 / 1.
- Flat cell escape: a_k = E₃(0.5k) − E₃(0.5k + 0.5), giving a₀ = 0.278, a₁ = 0.112, a₂ = 0.053 and so on, weighted by the beam's cell masses.
- Resulting Δ ≈ 0.199 / 0.146 / 0.103 / 0.057 / 0.024 / 0.006.

**New result: the sign is systematic [ND].**
- In a directly lit cell on a single face, the first-collision density inside the cell (∝ e^{−z/μ}) and the escape probability (∝ E₂(z)) both decrease with depth.
- By Chebyshev's integral inequality, the density-weighted escape is at least the flat average. So Δ ≥ 0 for every incoming bin and every finite h, not only at near-horizon bins.
- At edges and corners the two functions need not move together, so the sign can be mixed there.

**Per-bin size [ND, hand-integrated, about ±20 %].**
- The concentric map preserves area. Within one of the 8 outer edge bins, sinθ is uniform on [0.5, 1]. Within the 4 outer corner bins, its density is ∝ (sinθ − 0.5).
- Mean excess at h = 0.5 mm: about 1.8 % of q (outer edge bins), about 2.4 % (outer corner bins), about 0.6–0.8 % (the 4 inner bins).
- For red, the breach of ω·d requires ωΔ·d > (1−ω)Ψ′, where Ψ′ is the post-first collision count. The ω·d breach is therefore likely whenever Δ exceeds the absorbed fraction. This is consistent with AR's "1–5 % of d" (INFERENCE there).
- With 0.05 mm layers the μ → 0 excess is still about 0.049 (E₃(0.05) ≈ 0.455, flat average ≈ 0.451). So grading does not cure the original split. The fix has to be structural.

**Amended fix verified [ND].**
- W1 and Φ_pt come from the same rays per point. Each ray from p crosses cells over segments [s_a, s_b], deposits e^{−s_a} − e^{−s_b} into each, and exits with e^{−L} into exactly one (j, o). The total is 1 per ray, hence (P1).
- Every first collision is either escaped or continued, with total probability 1.
- From the second collision onward, cells re-emit uniformly. Both e_y and the continuation of a cell come from the *same* uniform-emission model, tied together by (P2).
- The intra-cell flat-source error is a bias, not a counting error.
- There is no remaining split inconsistency: no path enters both the q layer and the ψ layer.

**Residual flat-source bias after the fix [ND].** Its sign depends on the regime. Re-emitting flat inside a cell in the boundary layer:

| Intra-cell density | Typical regime | Effect of flat re-emission |
|---|---|---|
| Decreasing with depth | Near-source, grazing second collisions | Under-predicts escape; more absorption and lateral spread |
| Increasing with depth (ψ ∝ z + 0.71) | Diffusive | Over-predicts escape, more at grazing outgoing bins |

- Example for the diffusive row: at μ = 0.4, h = 0.5, the weighted mean depth is 0.199 against a centre of 0.25, giving a top-cell ratio of 0.96/0.909 ≈ 1.056.
- This resolves dispute 11 (SC l.225): the developer's sign (AR l.386–391) is correct for the stated ψ ∝ z + 0.71; the reviewer's sign applies to decreasing densities.
- It is consistent with the WL77A non-evidence +2.1 % red energy at 0.5 mm voxels [HM, `wl77a/specialist_physics.md` l.299]. That figure is labelled non-evidence at source.
- This affects accuracy, not the invariants.

**Part A item 9 check.** "The original identity already implied ≤ ω·d" is correct only for the *unsplit*, consistent form, where 1ᵀψ ≥ 1ᵀq = d. The split form has no such identity.

## B4. Ballistic and scattered disjointness

**Claim** [ND]. Each B ray contributes exp(−σt) to β (zero collisions) and 1 − exp(−σt) to F_in (at least one collision). K contains only paths with at least one collision. The reference uses the same tagging, with zero-collision exits counted as B (`engine71c.py` l.5). So the two are disjoint by construction.

**Adversarial example 1: missed and wrong-side rays.** B counts these (`r57_geometry.py:86–87`), but they add 0 to β and are therefore inside d. F_in has no finite chord to deposit their mass along. So Σq = d − (miss + wrong)/q, and the hard test "Σ F_in = d to floating-point precision" (FD l.268) fails. Required: a predeclared rule, either re-cast the ray with a watertight intersector or report a "leak" category on both sides.

**Adversarial example 2: inheriting B's sampling error.** "Same rays as B" carries B's random q = 256 sampling error into C1's collided mass.
- At edge ports, for bins aimed at the side face, β is O(0.5). The per-(i, a) standard error is about √(0.25/256) ≈ 3 % of d. This is a direct energy error exactly in the BND strata [ND, estimate].
- It is the same failure class as the q = 1,024 single-scatter confound [HM, Brief l.138].
- The record's "exact ray integral" (CR l.150) is exact *per ray only*.
- Required: rebuild B and F_in together on one dense, deterministic, orbit-symmetric ray set, and declare the new d.

## B5. No multiple counting of photon paths

The amended C1 splits paths by collision order [ND]:
- order 1: ωΦ_pt q;
- order 2: ωΦout(ωW1 q);
- order k ≥ 3: ωΦout(ωW)^{k−2}(ωW1 q).

Each path with n ≥ 1 collisions appears in exactly one term, and each of its exits is assigned to exactly one (j, o), because a ray exits once.

**Adversarial example: double assignment at a triangle edge.** A ray that hits an edge between two triangles could be tallied twice or not at all by a non-watertight intersector. The reference has the same risk and counts such rays as leaks (`p53_tracer.py:168`). Required: a watertight intersector, or a single-hit rule.

## B6. Finite boundaries: edges, corners, dent cavity, chords leaving the medium

**Rest box (convex) [EP].** The box contains all its chords, so a masked convolution or FFT far field is ownership-exact at rest (CR l.170).

**Edges and corners [ND].**
- Point-level W1 and Φ_pt treat two- and three-face escape exactly, by ray.
- Cell-level Φout at edge cells is flat-source, with mixed-sign bias.
- These are accuracy issues, not invariants.

**Dent: BLOCKER for the recorded far-field-FFT route [ND].**
- The dented medium {z < f(x, y)} is not convex. The chord between two points under opposite rising dent walls passes through cavity air.
- The amended record keeps a far-field pFFT beyond about 3 mfp (SC l.112) and dropped the chord correction (AR l.333). The pFFT uses the free-space kernel for every far pair of medium cells, whatever the visibility.
- Near-surface cells across the cavity therefore receive K-owned next-collision mass for paths that physically exit and re-enter through air, and those paths belong to C (FNTO ownership: CR l.151, SC l.95).
- Under (P2), energy is still conserved. The violated invariant is **ownership**: K carries a fraction ε_cav of air-crossing paths, and C's cavity exits are scaled down by the complement.
- Magnitude depends on the profile, and the dent profile was not in my reading. A cosine dent of depth 6 mm and radius 15 mm has a bottom curvature radius ≈ 2a²/(hπ²) ≈ 7.6 mm. Sagittas are then about 0.15 mm (3 mm chord) and 0.26 mm (4 mm chord), so the top 3–5 graded layers are affected. A spherical cap of the same size has a radius of about 22 mm. AR l.55's 36 mm figure is a different assumption.
- Required: a visibility-exact far field for cells whose far sphere meets the cavity, either matrix-free ray-telescoped W to the full chord (FD l.256 option) or a reinstated explicit chord correction.

**Mesh tiling on the dent [ND].**
- Per-ray telescoping keeps (P1) only if every in-medium segment is assigned to exactly one cell.
- Adversarial example: a body-fitted cell top built from 1.25 mm lateral corners against the reference surface triangulated at 0.625 mm. Their sagitta mismatch is about 1.25²/(8R), roughly 5 µm at R = 36 mm or 26 µm at R = 7.6 mm. That is 10–50 % of a 0.05 mm layer.
- Required: cells defined as the true triangulated medium partitioned by offset layer surfaces, or an explicit segment-assignment rule.

**Chords leaving the medium (F_in, W1, Φ_pt) [HM / ND].** The first exit terminates, and re-entry belongs to C. This matches the reference: `continue_air=False` by default (`p53_tracer.py:153`), and the WL57 walker has no re-entry branch. It is consistent.

## B7. Energy and mass accounting

**Closure [ND].** Incident 1 = β + d.
- d = absorbed + collided exits.
- absorbed = (1−ω)(1ᵀq + 1ᵀψ).
- exits = ω·d − (1−ω)·1ᵀψ.
- Summed: (1−ω)(d + Ψ) + ω·d − (1−ω)Ψ = d. Exact under B2 (P1–P3).

**Recommended hard test.** d − ΣK − (1−ω)(1ᵀq + 1ᵀψ) = 0 to floating-point precision, per column. Report leak and miss categories in the same units as the reference's leak, wrong-side and truncation tallies (`p53_tracer.py:168, 172`).

**Adversarial example: shape normalisation.** Φout = (1 − 1ᵀW)·shape needs the shape to sum to 1 to floating-point precision. A float32 normalisation error of about 1e-7 lies inside the 1e-6 passivity tolerance. Tolerable, but declare it.

## B8. Linearity and zero in, zero out

**Linear configurations [ND].**
- A fixed number of Neumann (Jacobi) sweeps from ψ₀ = 0 is a fixed linear map. It is exactly linear and exactly zero for zero input.
- A converged solve with a relative stopping rule runs a number of iterations that depends on u, so superposition holds only to its error κ·τ.

**Adversarial example 1: tolerance against the test.** At τ = 1e-6 relative residual (SC l.115) and κ ≈ 39 for red, the solution error is up to about 4e-5. That fails the 1e-6 superposition test (FD hard test 1).

**Adversarial example 2: warm starts.** A warm start (AR l.407) makes ψ depend on the previous lighting state. With u = 0 the iterate starts from ψ_prev and reaches zero only to tolerance.

**Requirement.** Use a cold-start fixed linear iteration, or τ ≲ 2.5e-8 for red, giving κτ ≤ 1e-6.

**Exact red sweep counts [ND].** Error after k sweeps ≈ ρ^k/(1−ρ) ≤ 1e-6 needs k ≳ ln(2.6e-8)/ln(0.974) ≈ 663. ρ^k ≤ 1e-6 alone needs about 525–533 (B11).

**Krylov.** Truncated Krylov is not exactly linear [WD, conceded SC l.57]. CG also needs a symmetric operator. Amended C1 gives up exact volume-weighted symmetry (AR l.304), so even the "symmetrised solver" estimate (CR l.225) is not formally available [ND].

## B9. Reciprocity

**Physical relation [EP + ND].** For an index-matched, homogeneous, isotropic medium with K = total − B, and with incoming bins labelled by the outward direction they come from (HM, B1):

  A_i·Ω⊥_a·K[j,o; i,a] = A_j·Ω⊥_o·K[i,a; j,o]

This needs both bins to exist at both resolutions. Aggregate o into its fine16 group g; nesting holds (B1).

**Consequences at rest.**
- Ω⊥ = π/16 for every 16-level bin, and all ports have area 6.25 mm². So the aggregated 16×16 block K̃ is **symmetric**: K̃[j,g; i,a] = K̃[i,a; j,g].
- On the dented pose the relation holds weighted by port area.
- The reference itself satisfies this only within Monte Carlo noise.

**C1 at discrete level: NOT ESTABLISHED.** There are three separate sources.
1. Quadrature: W is partition-first and not symmetrised [WD, AR l.304].
2. F_in is a random q = 256 sample, while Φout is deterministic and normalised to the complement (B4).
3. **Structural [ND].** At order ≥ 2 the forward path is exact at the source end and flat-cell at the receiver end; the reverse path swaps them. So the aggregated reciprocity defect is O(h) at grazing bins, about 2 % at μ_out ≈ 0.05 with 0.05 mm layers (AR l.393 estimate). The defect measures last-flight bias, not just noise.

**Correction to the claimed trilemma [ND].** "Partition, reciprocity and a non-negative complement cannot all hold" (AR l.304; SC l.160) is too strong. The volume-weighted symmetrisation W̃[y←x] = ½(W[y←x] + V_y·W[x←y]/V_x) is non-negative and reciprocal, and (P2) restores the partition. The complement stays non-negative wherever a cell's reciprocity defect is below its escape probability. It therefore fails only where the defect exceeds about 1e-3 (deep cells).

**Status.** Reciprocity is not a contract invariant (Brief l.70). Per the task rule it is a BLOCKER only for any claim that C1 is reciprocal. Report it as a diagnostic.

## B10. Symmetry and angular-bin transformations

**Rest D4h [HM, Brief l.65–68].**
- The 16 operations permute ports and bins exactly.
- The concentric map is equivariant under the square's symmetries, because its sector formulas are symmetric (`p53_geom.py:35–48`), and N is even. So the bin grid maps to itself [ND].

**Adversarial example 1: B's random rays (the BLOCKER-S).**
- B is a fixed-seed random sample, not orbit-symmetric (`r57_geometry.py:80–91`). So β(i, a) ≠ β(g·i, g·a) beyond roundoff, and d is not exactly equivariant.
- FD hard test 5 (Σ F_in = d against the archived B) and hard test 7 (exact D4h) therefore cannot both pass.
- Fix: rebuild B from orbit-representative rays mapped by g, shared with F_in, and declare the new d.

**Adversarial example 2: nodes on bin boundaries [ND].**
- `local_to_bin` floors (u·N) (`p53_geom.py:55`). A quadrature direction on a symmetry plane, for example the normal, maps to u = v = 0.5. That is a grid corner, assigned to bin (4, 4), whose mirror image should be (3, 3).
- An orbit-symmetric direction set naturally contains such nodes, which breaks equivariance.
- Fix: offset stratification so no node lies on a bin edge, or split edge nodes' weight equally.

**Adversarial example 3: roundoff.**
- Permutation-assembled matrices are exactly equivariant as stored. Sparse matrix–vector and FFT summation order still differ between u and g·u.
- So equivariance holds to roundoff: about 1e-7 in float32, about 1e-16 in float64 (AR l.88 says about 1e-7). Hard test 7 needs a stated tolerance.

**In-plane translation invariance at rest [ND].**
- The "per-face template" claim (AR l.294; R2-8) holds to below float32 resolution only where the bin's chords cannot reach a side face within about 17 mfp (e^{−17} ≈ 4e-8).
- Ports near the sides are distinct contexts. This is consistent with "a few hundred boundary contexts" [WD].

**Dented frames [HM, Brief l.69; ND].** There is no symmetry to claim. C1 must bin escapes in the **exit triangle's** current frame (t, b, n), as the reference does (`r57_fastwalk.py:36`), not in a mean port frame. Incoming bins come from `cell_source` in the triangle frames (`p53_tracer.py:95–99`). Rigid motion: K is invariant in the body frame when frames move with the body [EP].

## B11. Stability of the multiple-scattering solve

**A priori bound [ND].** With W ≥ 0 and column sums ≤ 1 − e_y ≤ 1, ‖ωW‖₁ ≤ ω < 1, so ρ(ωW) ≤ ω. Neumann iteration converges in every channel, including the ω = 1 furnace check, because some columns have e_y > 0 and W is irreducible. An unclamped FFT can lose this bound.

**Coupled K∘C [ND].**
- C passes on at most the exited flux (column sum ≤ 1).
- Joint columns sum to at most ω(1 − e_y) + ω·e_y·c_re ≤ ω < 1. The per-domain β + ΣK ≤ 1 − (1−ω)d < 1.
- So dent self-re-entry loops are stable, and joint unknowns stay stable.

**Spectral radius of the slowest mode [ND; agrees with FD's critic, CD l.21].**
- Extrapolated thickness 11.42 mm. B² ≈ (π/11.42)² + 2(π/101.42)² ≈ 0.0776 mm⁻². λ ≈ arctan(k)/k ≈ 0.975.
- ωλ ≈ 0.974 / 0.966 / 0.878 (R / G / B).

| Channel | Plain Neumann sweeps to ρ^k = 1e-6 | Sweeps to 1e-3 |
|---|---|---|
| Red | ≈ 525–533 | ≈ 265 |
| Green | ≈ 396 | ≈ 198 |
| Blue | ≈ 106 | ≈ 53 |

- These are worst-case asymptotic rates [ND]. Boundary sources project less onto the fundamental mode, so they may converge faster [UH].
- Lateral 1.25 mm re-randomisation barely changes ρ, because the lateral buckling is 0.0019 of 0.0776.

**Acceleration [PA / UH].**
- DSA effectiveness is known to degrade in multidimensional S_N calculations with discontinuities, and a Krylov method preconditioned with DSA was the remedy (Warsa, Wareing & Morel, NSE 147, 2004, verified on the web). That is S_N, not collision probability, so it is an analogy only.
- On 25:1 cells (0.05 mm × 1.25 mm) and on non-orthogonal stretched dent cells, a diffusion discretisation may lose the M-matrix property. Corrections could then be negative [ND, UH].
- Any non-monotone accelerated iterate breaks the B1 non-negativity test.
- An exactly invariant-preserving option exists: fixed-count Jacobi/Neumann from zero, which stays linear, non-negative, passive and equivariant. Gauss–Seidel from zero is also monotone and linear, but its sweep ordering breaks exact equivariance [ND].

## B12. Non-negativity in detail (the BLOCKER-S of B0)

**Exact operator [ND].** F_in, W1, Φ_pt ≥ 0 (positive ray weights). With W ≥ 0 and ρ(ωW) < 1, (I − ωW)⁻¹ = Σ(ωW)^k ≥ 0. With 1ᵀW ≤ 1, the escape is ≥ 0. Hence K ≥ 0.

**pFFT with non-negative projection and interpolation [ND].** In exact arithmetic the far part is ≥ 0: the precorrection removes exactly the grid's own near pairs, and the kernel is positive.

**Adversarial example 1: float32 FFT.** Convolution roundoff is about ε·‖·‖ spread over all outputs. Far-tail entries with true values near 1e-10 come out at about ±1e-7·max. That fails "no entry below −1e-9·max" (FD hard test 3). Clipping afterwards would make W nonlinear. Requirement: a float64 FFT, or no FFT.

**Adversarial example 2: accelerated solve.** A non-monotone solve converged to 1e-6 leaves errors of about 4e-5 relative. Tiny far entries of K can become negative well below −1e-9·max. Requirement: a monotone iteration, or report non-negativity "to the solve tolerance" and amend the test.

## B13. Blockers and the amendments that remove them

| Blocker | Required amendment |
|---|---|
| **BL-1** Ownership on the dent with a far-field FFT (B6) | A visibility-exact far field near the cavity (matrix-free full-chord ray-telescoped W, or an explicit chord correction) |
| **BL-2** Passivity and non-negativity under the unspecified clamp (B2) | Predeclare policy (P2*): rescale the W column to unit sum. Forbid zeroing the escape without rescaling. Report the binding fraction, which is expected to be non-trivial at interior cells for any few-percent far-field error |
| **BL-3** D4h exactness against archived B (B10) | Rebuild B and F_in on one shared, deterministic, orbit-symmetric, dense ray set with no nodes on bin edges. Declare the new d. This also removes the q = 256 energy error at edges (B4) |
| **BL-4** Linearity and zero-in-zero-out (B8) | A cold-start fixed linear iteration, or τ ≤ about 2.5e-8 (red), with no warm start in contract tests |
| **BL-5** Non-negativity (B12) | Float64 or no FFT, and a monotone iteration (or amend the test to the solve tolerance, predeclared) |
| **BL-6** Discrete reciprocity (B9) | Not removable within the amended design without a symmetric last flight. Report it as a diagnostic only; make no reciprocity claim |
| Mesh tiling and watertightness (B4, B5, B6) | Must be specified, but each has a clear fix: a segment-assignment rule, a watertight intersector, and handling of missed or wrong-side rays in both B and F_in |

## B14. Corrections to Part A

1. **Ambiguity 11 (sign of the flat-cell bias)** is resolved [ND, B3]. The sign depends on the regime: the developer's sign is correct for ψ ∝ z + 0.71, and the reviewer's sign applies to a decreasing intra-cell density.
2. **Ambiguity 14 (convention source)** is resolved from code [HM, B1]. The tally is flux per flux, with no cosine factor in Φ. F_in directions are cosine-weighted within the bin. Bins are binned in the exit triangle's frame. fine16 nests into the incoming grid, so FD C4 is resolved.
3. **New:** the brief's "exactly computed" B (Brief l.58) is a fixed-seed q = 256 sampled quadrature (`r57_geometry.py:80–98`). "Same rays as B" (CR l.150; SC l.90) therefore conflicts with "exact F_in" and with exact D4h (B4, B10).
4. **CD l.20 refined** [ND, B2]. W ≥ 0 is sufficient for passivity, through 1ᵀψ ≥ 0. 1ᵀW ≤ 1 is needed for non-negativity and for the ρ ≤ ω bound, not for the energy identity.
5. **Original-split defect.** It is systematic in sign at directly lit single-face cells for every incoming bin, not only near the horizon. It is about 0.6–2.4 % of q per bin at 0.5 mm, and about 5 % at μ → 0 even with 0.05 mm layers [ND, B3]. The amended fix removes it exactly [ND].
6. **The partition / reciprocity / non-negativity trilemma** (AR l.304) is conditional, not absolute [ND, B9].

## B15. Not established here (untested or unmeasured)

- Every accuracy statement: the gates, the BND_FAR strata, the dent signed change [UH].
- The absorbed fraction for boundary sources in red. That decides whether the original split also breaches the looser d bound [UH].
- The binding fraction of the far-field clamp, and the error of the FFT interpolation [UH].
- The actual dent profile, so the size of the BL-1 ownership leak [UH].
- The archived B's q and the medium's g, to be read from the archived metadata.
- Accelerated-solver iteration counts on the graded and stretched mesh [UH].

Sources:
- [Warsa, Wareing & Morel 2004, NSE 147 (ANS)](https://ans.org/pubs/journals/nse/article-2431)
- [OSTI 976407 (conference version)](https://www.osti.gov/biblio/976407)
