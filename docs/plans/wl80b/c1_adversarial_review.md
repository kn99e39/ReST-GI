# C1 adversarial / counterfactual review (Part H) and architect response

> WL80B record (workflow `wf_9bbd762a-ae2`, 12 agent runs). Agent texts reproduced verbatim; headings added by the Coordinator. Design-only: nothing implemented, executed, trained or measured. The original C1 architect was not resumable; a stand-in reconstructed C1 from the sealed WL80A records (immutable inputs, sha256 in `inputs.json`).

## Adversarial review 1

# WL80B Part H: adversarial review of the C1 / CP-K redesign (physics and generality)

I read only the allowed inputs, ran nothing and wrote no file. I spot-checked two sources: `neutral_problem_brief.md` l.36–37, 52, 57–58, 69 and `59_WORKLOG_PERSISTENT_INTRINSIC_FS.md` l.47. I re-derived Part B–E statements before relying on them. Where I did not, the claim is cited as "(Part X, not re-verified)".

**Labels.**

| Tag | Meaning |
|---|---|
| [EP] | Established physics |
| [PA] | Prior art |
| [HM] | Historical measurement or record (file:line) |
| [WD] | WL80A design claim |
| [ND] | New derivation (mine) |
| [UH] | Untested hypothesis |

## H0. Verdict

**The no-revision conclusion survives.** I found no argument that a change to the architecture would add value on the learned-K question [ND]. This agrees with Part F.

**The corrected specification in "No revision proposed, §3" does not survive as written:**
- It cannot be executed on the dented pose while meeting its own invariants and the 12 GB budget (H1-1, BLOCKER).
- One of its hard tests contradicts its own solver (H1-2).
- Two of its corrections conflict with each other (H1-3).
- Its claimed physical scope is wider than the derivations support (H2).

None of these findings needs an architectural fix. Each is a specification or scope error.

## H1. Objections

### 1. BLOCKER: no admissible W fits on the dented pose
**Label:** [ND], plus [WD] for XR l.957.

The corrected specification requires three things at once on the dented pose:
- W built only by full-chord, ray-telescoped visibility (E4: the precorrected FFT, pFFT, is removed; a non-negative kernel split is allowed only at rest);
- a cold-start, fixed-count solve of about 525–663 red sweeps (E13);
- complement-defined escape with no clamp (E5).

There are three ways to build W, and each fails:
1. **Store W to about 10 mfp.** That is about 1.6e9 nnz, about 13 GB (XR l.957 [WD]). It does not fit in 11–12 GB, even at L0.
2. **Evaluate the far field matrix-free.** That costs about 1e9–1e11 kernel/ray evaluations per application (Part E1, not re-verified). Over 663 sweeps that is about 7e11–7e13 per lighting state. Infeasible.
3. **Truncate W at the explicit near-field radius (about 3 mfp).** Under complement escape (P2), the truncated next-collision mass is silently booked as escape from the emitting cell. It is placed by that cell's Φout shape, at its local ports and bins.
   - Energy stays exactly conserved.
   - First-exit *location and direction* are wrong. A flight that should collide 3–10 mm away exits at the source cell instead.
   - The free-space fraction of flights longer than 3 mfp is e^{-3} ≈ 5 % [EP]. An interior red history makes about 22–39 collisions, so most histories hit this at least once.
   - The effective ρ drops and the red lateral response shrinks. The size is [UH], plausibly order-unity at coarse red.
   - This breaks the ownership/first-exit invariant in a form that the closure, passivity and partition tests cannot detect, because all three still pass.

**Consequence.** On the dented pose (the decisive stratum), the corrected specification has no W construction that is non-negative, sub-stochastic, visibility-exact, untruncated, fits in 12 GB and can be applied 663 times per light.
- A visibility-aware, non-negative two-level far field (aggregated cells with ray-telescoped aggregate probabilities) would be a numerics fix. It is not in any record. Its positivity and visibility properties are [UH].
- This does not overturn the no-revision conclusion. It does mean the "corrected specification for the qualified diagnostic role" is not yet a specification.

### 2. MAJOR: the per-column closure test contradicts the fixed-count solve
**Label:** [ND].

With ψ_N = Σ_{k<N} (ωW)^k ωG1 u, generation N+2 is deposited but never propagated. Closure then gives:

  d − ΣK − absorbed = r_N = 1ᵀ(ωW)^N ωG1 u ≈ ρ^N·O(d)

For red, ρ = 0.974 and N = 663 give ρ^N ≈ e^{−17.5} ≈ 2.6e-8, so r_N ≈ 2.6e-8·d. That is about seven orders above float64 roundoff.
- The hard test "per-column closure to fp precision" (§3, falsification step 2) therefore fails by construction.
- The alternative is to relax the tolerance after the fact, which AGENTS.md forbids.
- **Fix (specification):** tally r_N as an explicit "truncation" category, as the reference does for its own truncation (`p53_tracer.py:172`, per Part B1 [HM]). Then test closure to fp precision *including* r_N.
- Passivity, non-negativity and linearity are unaffected. A truncated Neumann sum is monotone and stays a sum of non-negative terms.

### 3. MAJOR: per-point rotated direction sets conflict with exact D4h and with translation templates
**Label:** [ND], [PA].

- E2 requires per-point rotated direction sets to suppress ray effects (Lathrop 1968 [PA]).
- E14 requires exact D4h equivariance, and the rest-pose cost model uses in-face translation templates (E5, E16).
- A rotation field R(p) keeps equivariance only if R(g·p) = g∘R(p) for all 16 operations, and translation templates only if R is lattice-periodic.
- A lattice-periodic, D4h-covariant rotation field puts back exactly the coherent, lattice-locked aliasing that rotation was meant to remove [ND]. The remaining freedom is to vary R inside a port cell, which gives partial de-aliasing.
- The records choose neither option.

**Ray effects at the receiver end are not addressed at all.** The Φout (j, o) shape of a near-surface cell spreads over about 25 ports × 64 bins. With 1e3–1e4 rays per cell, as in Part E5, that is roughly 1–10 rays per non-empty bin [ND]. This deterministic angular aliasing does not shrink with spatial refinement. It sets the fine16 and grazing-bin accuracy. E2's correction covers only the source end.

### 4. MAJOR: contract-mode exactness does not carry to near-conservative or larger bodies
**Label:** [ND], diffusion estimate.

The contract mode is the only mode that is invariant-exact (E13). Its sweep count scales as ln(1/(τ(1−ρ)))/(1−ρ), and 1−ρ ∝ B² + σ_a/D.

| Case | ρ | Sweeps (sweep counts) |
|---|---|---|
| Current slab at ω = 1 (furnace) | ≈ 0.975 | ≈ 690 |
| 100 mm cube, red | ≈ 0.998 (B² ≈ 2.9e-3 mm⁻²) | ≈ 1.0e4 |
| 100 mm cube, ω = 1 | ≈ 0.999 | ≈ 2.2e4 |

For any body larger than about 10 mfp in every direction, only accelerated modes are affordable. The redesign itself labels those "linear to tolerance, never used in contract tests".

**Consequence.** Beyond thin bodies, the hard tests (superposition 1e-6, minimum entry ≥ −1e-9·max) are no longer a property of the method. They become a property of a tolerance that must be predeclared per geometry. The invariant section in §3 claims too much generality.

### 5. MAJOR: highly absorbing media and the blue channel fall outside the small-h regime the accuracy argument uses
**Label:** [ND], [HM] for `candidate_H3.md` l.90 as cited by Part F.

- E1 retains a cell-averaged order-2 escape. Part F1 notes H3 had it exact (D2 on a 0.25 mm layer).
- As ω falls, order-2 escape becomes a large fraction of the exit, and the response length shrinks:
  - blue slab-mode length 1.63 mm (Part D1, re-checked);
  - for ω = 0.5 the diffusion length √(D/σ_a) ≈ 0.8 mm, which is only indicative because diffusion is invalid there [EP].
- At L0, h_lat = 1.25 mm, so h/ℓ_resp ≈ 0.77 for blue and above 1 for ω ≲ 0.5. The h²-asymptotic lateral-bias model (Part C4.1) does not apply there. Part C4.1's claim that the blue coarse impact is small is therefore unsupported.
- **Internal inconsistency.** The redesign accepts numerics fixes inside the CP class (E4, E7, E13) but marks E1 RETAIN. A point-exact order-2 escape is the same class (H3 θ = 0). Leaving it out is a choice, not a consequence of the stop rule. This is exactly the regime where the choice binds.

### 6. MAJOR: the order-based HIGH-1 classification cannot separate a floor from convergence with two levels
**Label:** [ND].

Take errors against MC at L0 and L1 with a convention or reference floor of 0.03 (for example the d-convention error of about 3 % at edge bins, Part B4) and a converging part of 0.06 → 0.017:
- e = 0.090 → 0.047, so the observed order is log2(1.91) ≈ 0.94 > 0.5;
- this is classified "converging / affordability-limited", although a non-converging floor of a third of the gate exists.

What is needed:
- three levels and a reference-free Richardson self-convergence estimate;
- the floor estimated separately against the reference.

L1 is not affordable in explicit form (Part E1). For blue, L0 → L1 is not in the asymptotic range (objection 5). As specified, the falsification rule cannot set the method's validity boundary on the BND and dent strata.

### 7. MAJOR: the mesh is specified in millimetres, not mean free paths
**Label:** [ND].

- Grading (0.05–0.4 mm depth, 1.25 mm lateral) is defined in mm (E3).
- For a denser medium, for example mfp = 0.1 mm, the L0 lateral cell is 12.5 mfp. The flat-source lateral inflation term becomes ≈ h²/6 ≈ 26 mfp² against a per-step variance of 2/3 mfp², so the cell chain is meaningless.
- Keeping h/ℓ fixed scales cell count roughly as σ_t³ for a fixed body. W nnz scales further with partners.
- Inside the stated scope (mfp ≈ 1 mm) this is harmless. As a generality claim ("exact physics supplies the whole operator"), it holds only with an optical-unit mesh rule, which no record gives.

### 8. MAJOR: a non-isotropic phase function is incompatible with C1 as an exact operator
**Label:** [EP], [ND].

- For g ≠ 0 the emission direction after a collision depends on the incoming direction.
- A scalar cell density ψ and a scalar per-cell complement escape 1 − 1ᵀW are then no longer exact. Escape probability depends on the emission direction distribution, which depends on history [EP].
- Orders 1 and 2 *could* stay exact, because G1 and S1 are composed per beam and the beam direction is known [ND]. From order 2 onward the cell state needs angular moments (S_N/MOC class [PA]).
- A transport-corrected isotropic chain (σ_s' = σ_s(1−g)) is a model error, not a discretisation error. It would plateau, and under CD HIGH-1 that is a C1 fail.
- The redesign's reason 4 ("exact physics already supplies the whole operator; a network would add no information") holds only for g = 0 (brief l.36 [HM]). It should be stated with that scope.

### 9. MAJOR: heterogeneous media extend only partly
**Label:** [ND], [PA].

| Heterogeneity | What survives | Status |
|---|---|---|
| ω(x) only | W, G1, S1 unchanged; Ω diagonal in the solve; Woodbury for local regions | Supported |
| Piecewise-constant σ_t aligned to cells | Optical-depth telescoping keeps partition, complement and non-negativity (classical lattice CP [PA]) | Plausibly extensible; in-face translation templates are lost, so construction work goes to the full ~6e12 evaluations (Part E5, not re-verified) |
| Channel-dependent σ_t | B, d, q, G1, S1, W and Φout all triple; "q channel-free" (Part C1) fails | — |
| Continuously varying σ_t | Flat-property cells add a second model error | Plateaus at fixed mesh unless σ_t is resolved |
| High-contrast σ_t | Objection 7 applies locally | Infeasible cell count |

### 10. MAJOR: the dent stretch map in E17 is underspecified and conflicts with E12
**Label:** [ND], [HM] for WL59 l.47.

- E12 tiles by offset layer surfaces. E17 uses a vertical stretch "confined to the upper layers". These are different constructions.
- Under the natural reading of E17 (bottom stack fixed, everything above compressed), a 6 mm dent over the 10 mm slab compresses the region above, say, z = 3 mm from 7 mm to 1 mm, a factor of about 0.14.
- The top boundary stack (1.1 mm) is crushed to about 0.16 mm under the dent floor. That changes the flat-source bias exactly at the scored top ports. The spurious contribution to the signed change then *grows* instead of cancelling.
- An invariant-preserving rule exists [ND]: keep both normal-offset boundary stacks at fixed thickness (2 × 1.1 = 2.2 mm < 4 mm remaining under a 6 mm dent; for the measured WL59 dent of 5.75 mm × 22.5 mm, 4.25 mm remains) and compress only the interior.
- Whether rest and dent biases cancel remains [UH] (SC l.121).

### 11. MINOR: offset-layer tiling fails at small curvature radii
**Label:** [ND].

- Offset surfaces into the medium self-intersect wherever the surface's centre of curvature lies inside the medium and its radius is smaller than the offset depth (up to about T/2 = 5 mm).
- Dent rim: for a cosine profile the curvature radius is 2a²/(hπ²): about 7.6 mm at 15 mm × 6 mm, and about 17.8 mm at 22.5 mm × 5.75 mm. Both are safe.
- A 90° bend with inner radius below about 5 mm, or any sharper dent rim, needs general tetrahedral CP cells. E12's assumption is then violated.

### 12. MINOR: three proposed checks are self-referential
**Label:** [ND].

- **"Σ F_in = d_new to fp" (E7).** This is a tautology when B and F_in share rays. It does not test d_new against the true β. Edge-port bins aimed at the side face have β of order 0.5, and they need an independent analytic or high-q β check there.
- **The two-sided bound in E11.** It is built from C1's own 1ᵀΦ_pt q.
- **"Cavity-crossing mass = 0".** Under ray telescoping this is zero by construction. The physically meaningful check is reference-side: the walker also terminates at a cavity exit (`continue_air=False`, per Part B6 [HM]).

### 13. MINOR: the reciprocity reasoning in E15 is inconsistent with the redesign's own criterion
**Label:** [ND].

- E15 says a symmetric last flight "would be a different (H3-like) design". But an adjoint, point-exact last flight is a numerics choice within the same CP/FCS class, by the redesign's own E1/E2 logic.
- The structural O(h) reciprocity defect is a measure of last-flight bias, and the same omission as in objection 5.
- Reporting it as a diagnostic is acceptable. Calling it "not fixable within C1" is not.

### 14. MINOR: the convention gate uses a different geometry
**Label:** [ND].

- The half-space H-function albedo test needs a semi-infinite fixture, not the 10 mm slab mesh. It therefore checks a different discretisation, not the one being scored.
- At ω = 1 the half-space albedo is 1, so the test only bites for ω < 1 cases.

### 15. MINOR: concentrated spatial or angular incidence is limited by the contract, not by C1
**Label:** [EP], [HM] for the brief l.57 bin contract.

- Within one outer incoming bin, θ spans about 30°–90°, so the mean first-collision depth spans μ ≈ 0.87 → 0 mm.
- A collimated beam, or a spot smaller than a port, is averaged by the 16-bin, 2.5 mm contract.
- C1's F_in *could* take exact rays at the source end, but its output stays binned. "Exact first flight" means exact relative to bin- and port-uniform input only.

## H2. Generality matrix

| Case | Classification | Where it stops being valid |
|---|---|---|
| Highly absorbing slab (same σ_t, ω ≪ 1) | **Physics supported by derivation; accuracy unsupported at L0** | When h_lat ≳ ½ℓ_resp (blue already h/ℓ ≈ 0.77; ω ≲ 0.5 above 1), and with a cell-averaged order-2 escape (objection 5). Denser σ_t: objection 7 |
| Near-conservative, high albedo | **Supported by derivation** for this 10 mm slab (ρ ≤ 0.975, convergent even at ω = 1); **unsupported** as invariant-exact for larger bodies | When 1−ρ ≲ 1e-3, i.e. bodies ≳ 50–100 mfp in every dimension (≈ 1e4–2e4 sweeps); invariants then hold only to tolerance (objection 4) |
| Side, edge and corner sources | **Plausibly extensible** | Orders 1–2 are point-exact over multiple faces. Edge/corner grading and the 1.8–4 mm BND_NEAR transition are unspecified (E3 UNRESOLVED). Accuracy [UH]; the order rule cannot classify it (objection 6) |
| Unseen source location | **Supported by derivation** (structural: the source enters only through F_in u; every column is computed) | Accuracy equals discretisation accuracy for that context. Exact equality of translates needs a lattice-locked quadrature, which conflicts with de-aliasing (objection 3). No transfer claim of any kind applies |
| Strongly concentrated angular incidence | **Incompatible** below bin or port scale (contract); **supported** for bin-uniform input; **plausibly extensible** at the source end only | Any sub-bin or sub-port incident structure (objection 15). Grazing bins carry the largest quadrature and flat-source error |
| Local non-rigid edit (dent) | **Plausibly extensible** physics; **specification currently not executable** | No admissible W fits in 12 GB (objection 1). The stretch map may add a spurious signed change (objection 10). The rebuild touches 26–42 % of beams (Part E6, not re-verified) |
| Large deformation (bend, twist) | **Plausibly extensible** physics; **unsupported** by the recorded construction | Offset tiling fails when the inner curvature radius is below the offset depth (objection 11). Templates are lost. Full rebuild plus a matrix-free, visibility-exact W (objection 1). Near-contact gaps below port size: contract error (Part D2, case 7) |
| Non-isotropic phase function | **Incompatible** as an exact operator | Any g ≠ 0 beyond order 2. Transport correction is a model error that plateaus (objection 8) |
| Heterogeneous medium | ω(x): **supported by derivation**. Piecewise σ_t: **plausibly extensible**. Continuous or high-contrast σ_t: **unsupported**. Index mismatch or Fresnel: **incompatible** with the first-exit-only structure as recorded | Cells must align to σ_t and be graded in mfp. RGB σ_t triples all geometric state (objection 9) |

## H3. Invariants of the corrected specification

| Invariant | Holds? | Note |
|---|---|---|
| Ballistic/scattered disjointness | Yes | Needs a leak rule for missed or wrong-side rays [ND] |
| No multiple counting | Yes | Collision-order partition (Part B5, re-derived) |
| Linearity, zero in gives zero out | Yes | Exact for cold fixed-count Neumann; only to tolerance outside thin bodies (objection 4) |
| Non-negativity | Yes, exactly | A sum of non-negative products cannot round below 0 [ND] |
| Passivity ≤ ω·d | Yes | Monotone underestimate; true at every N [ND] |
| Energy closure to fp | **No, as specified** | Truncation remainder r_N ≈ 2.6e-8·d for red (objection 2) |
| First-exit ownership | **Not established on the dented pose** | W truncation under complement escape misplaces exits; no affordable alternative (objection 1) |
| Exact D4h at rest | **Conflicting** | Conflicts with per-point rotated de-aliasing unless the rotation field is lattice-locked and D4h-covariant (objection 3); otherwise holds to roundoff |
| Rigid body-frame invariance | Yes | [EP] |
| Discrete reciprocity | No; not required | Structural last-flight defect; the "H3-like" justification is inconsistent (objection 13) |
| Complement escape | Yes, algebraically | Physically correct only if W is untruncated (objection 1) |

## H4. Bottom line

- **No-revision conclusion:** SUPPORTED. Nothing here gives a reason to add architecture [ND].
- **Corrected specification:** NOT EXECUTABLE as written.
  - BLOCKER: no admissible W fits on the dented pose (objection 1).
  - Five specification-level MAJORs (objections 2, 3, 5, 6, 10), each with a stated numerics fix.
- **Scope:** narrower than stated. Exactness holds for g = 0, ω-only or cell-aligned piecewise-σ_t heterogeneity, an index-matched boundary, a mesh in optical units, and thin bodies in contract mode (objections 4, 7, 8, 9).
- **Measurements:** nothing about C1 has been measured. Every magnitude above is a derivation or [UH].

---

## Adversarial review 2

# WL80B adversarial review of C1 / CP-K and its no-revision redesign: novelty, computation and learnability

**Recommendation: C1 PHYSICAL BASELINE ONLY.** C1 should serve only as a classical-solver comparator. It is not an oracle, not a learned-K answer and not a method contribution.

**Decisive argument.**
- C1 has no learned component, and its mechanism is the textbook collision-probability (CP) / first-collision-source method. The search below adds Sillion 1995 for the volume–surface hierarchy [PA].
- So C1 cannot answer the research question: does a *learned* intrinsic K transfer? (CD HIGH-2, l.16.)
- C1 does have one legitimate value. It is the competent classical numerical method that any learned K must beat at equal accuracy or equal runtime. Omitting such a comparator is the documented "weak baseline" failure in ML-for-PDE work (McGreivy & Hakim, Nature Machine Intelligence 6, 2024) [PA].
- C1 is biased by discretisation and its bias is unmeasured, so it cannot be an "oracle".
- Its "corrected specification" also leaves its own cost unestablished (Objection 1). Even the baseline role therefore stays UNRESOLVED until it is built and measured, and building it is not authorised here.
- I agree with "no revision proposed". I dispute several of the reasons and cost figures behind it, below.

## Objections

### 1. BLOCKER: the corrected specification leaves out the cost of its own far field, and the replacement it chose reintroduces ray effects

**Label:** [ND]; ray effects [PA, Lathrop 1968]; matrix-free cost [UH].

E4 and §3 drop the precorrected FFT (pFFT) and replace it with a "matrix-free, full-chord, ray-telescoped W beyond the explicit near field". But the contract-mode cost in E13 and §3 ("≈ 0.7–1.0 s per single-light RGB application") is the *near-field-only* bandwidth figure: 524 sweeps × 1.3–1.9 ms (Part E3).

**Arithmetic [ND]:**
- A 1.25 × ~1 mm target cell at 10 mm subtends about 1.25e-2 sr. Covering it with about one ray already needs about 1e3 rays per source cell.
- Holding ray-effect error near 10 % at the far range needs about 1e5 rays per source cell.
- That gives 2.5e5 cells × 1e5 rays × ~40 graded segments ≈ 1e12 segment deposits per sweep.
- At 265–663 sweeps this is ≈ 3e14–7e14 deposits per lighting state.
- The cheap alternative, about 1e3 rays, is exactly the source-dependent aliasing that E2 flags and that refinement does not remove.

**The consequence is a dilemma:**
- either the contract-mode far field is computationally infeasible per lighting state;
- or it is ray-effect-limited, which undermines the "accuracy = discretisation" premise of Part C.

The specification must instead say that the far field is constructed once and stored in compressed, non-negative form. Its memory would then have to be shown to fit within 12 GB alongside the 2.4–6.7 GB of factors. This is not in the record.

### 2. MAJOR: the claim that ray telescoping is "the only W construction" with W ≥ 0, sub-stochasticity and ownership (E4, E5) is false; cluster hierarchies are prior art

**Label:** [PA]; applicability [UH].
- Hierarchical radiosity with clustering (Smits–Arvo–Greenberg 1994) can aggregate a far field with non-negative weights, which keeps W ≥ 0 and allows renormalisation against the complement.
- Sillion 1995 (IEEE TVCG 1(3), "A Unified Hierarchical Algorithm for Global Illumination with Scattering Volumes and Object Clusters") extends this to the zonal method for scattering volumes, with visibility handled per cluster pair.

On the dent, cluster pairs that straddle the cavity must be refined to cell level to close BL-1. This is a known refinement criterion, not a new mechanism.

Two consequences:
- E4's "REMOVE pFFT → matrix-free" is one choice among known ones, not a forced one.
- It strengthens Part F: the *whole system*, a volume-plus-surface zonal hierarchy that is rebuilt locally under motion (Drettakis–Sillion 1997), is closer to ALREADY KNOWN than "UNCERTAIN, engineering level". Only the 2.5 mm × 64-bin interface resolution is specific, and that is a parameter.

### 3. MAJOR: "deterministic analogue of the WL56 MC oracle K" is the wrong role label

**Label:** [HM: `BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md` l.12, l.19]; [ND].
- The WL56 oracle is unbiased Monte Carlo with noise. The physical reference role belongs to Cycles volume (l.19).
- C1 is a deterministic *approximation*. Its lateral flat-source bias is predicted to be large: D_lat +31 % at 1.25 mm, with the net length change +8 to +15 % (Part C4.1, [ND]/[UH]).
- Using C1 as an oracle for F/S invalidation or as labels would put an unmeasured bias in place of measured noise.
- There is no row for a "classical deterministic solver comparator" in the baseline roles table. Adding one is a human decision.
- Calling C1 "physical" is also misleading. It is a numerical-solver baseline, not a physical reference.

### 4. MAJOR: "a network would add no information, only a failure mode" (reason 4, E19) is too strong; a learned closure is identifiable here, and its value is computational

**Label:** [PA, Bar-Sinai et al. PNAS 116(31), 2019]; [ND]; [UH].
- The L0 bias is local, source-free and a function only of h/ℓ and the local cell-context class (Part C4.1). That makes it the textbook target of a learned coarse-grid discretisation.
- It is identifiable *without MC noise* by supervising on C1's own L1/L0 difference. The teacher's residual bias is about ¼ of L0's at order 2.
- WL77A's Horn A ("vacuous, below seed spread", `coordinator_decision.md` l.15) was argued for closures supervised by MC references. It does not automatically carry over to self-refinement supervision.
- The value is bounded, though. At best the closure buys L1 accuracy at L0 memory, replacing about 16× the near-field nnz, which is a constant-factor trade.
- A closure on dented or stretched cells faces the same out-of-distribution transfer question as the learned K, just in a lower-dimensional, source-free input.
- A non-learned fix exists too: moment-preserving or linear-source re-emission [PA; positivity UH, AR l.418].

So the correct reason for not adding learning is different: it is H3-class, it does not answer the transfer question, and its benefit is an implementation constant. The reason is not that it is vacuous.

### 5. MAJOR: the dent and per-frame update claims have no cost case at any solver setting

**Label:** [ND] from Part E6 and D2; [HM: `59_*` l.182, `76_*` l.252].
- 26–42 % of beams are dirty, and the vertical stretch dirties every footprint column.
- That is about one third of a deterministic construction that does not exist. Its sampled analogue took 75.0 s, of which 74.3 s was quadrature.
- A global cold re-solve follows, and Objection 1 makes it seconds or more.
- The measured comparators are WL59 S at 0.33 s and A75/Q78 rebuilds at 0.015 / 0.102 s.
- Woodbury updates are exact but cost more than one solve (rank 2e4–6e4). Truncated updates break W ≥ 0.
- So C1 offers no geometry-update advantage under any of the three comparison classes. A Neural Full Recompute comparison is not applicable, because C1 owns no learned state. Internal full/selective rebuild and the class-3 re-render are both unmeasured.
- The redesign is right to drop the 1–5 s figure, but it should state the conclusion: per-frame deformation is outside C1's plausible operating region.

### 6. MINOR: the contract-mode sweep count is internally inconsistent and overstated

**Label:** [ND].
- A cold-start, fixed-count Neumann iteration is *exactly* linear and zero-in-zero-out at any N (Part B8). The 663 figure comes from the κτ ≤ 1e-6 rule for *converged* solves and does not apply here.
- N is an accuracy choice only. Red truncation reaches 1e-3 relative at about 265 sweeps and 1e-2 at about 175 sweeps, both far inside the 0.10 gates.
- E13 cites 663 sweeps but quotes 0.7–1.0 s, which corresponds to 524 sweeps. At 663 sweeps the near-field cost is 0.86–1.26 s.
- This does not change the verdict.

### 7. MINOR: one collision-order sequence serves all of RGB and every uniform albedo

**Label:** [ND]; series form [PA, the successive-orders-of-scattering method].
- The moments m_k = W^k G1 u are channel-free, and ψ_c = Σ_k ω_c^{k+1} m_k.
- One single-RHS sweep sequence therefore serves all three channels. The compute-bound batched cost falls by about 3×, though the bandwidth-bound single-light cost does not change.
- If the per-order exit tallies Φout m_k are kept (≈ 265 × 245,760 floats ≈ 260 MB per light), a uniform-albedo edit needs no re-solve at all.
- This corrects Part D case 5 and E16 ("global re-solve"). It is not novel.

### 8. MINOR: the pFFT negativity rationale (E4 (a)) is wrong; the conclusion survives for other reasons

**Label:** [ND].
- In a precorrected FFT the *correction* matrix has negative entries, but the *assembled* operator has exact near entries and grid-convolved far entries. With non-negative projection and interpolation it is ≥ 0 in exact arithmetic (Part B12 says the same).
- Negative entries come only from higher-order Lagrange stencils or from float32 roundoff in the far tail.
- On the convex rest pose, a layer-pair Toeplitz/FFT built from *exact* ray-telescoped cell kernels in float64 is non-negative, exact and translation-commensurate, and much cheaper.
- E4's REMOVE is justified only by the dent's visibility (BL-1) and the mass error, not by "precorrection".

### 9. MINOR: the cost anchors mix hardware and comparison regimes

**Label:** [HM: `76_*` l.254–255; `75_*` l.221–222; Part E3].
- C1's "4–260×" ratio is a 3080 Ti estimate set against measured anchors. That is legitimate only for the FIA 3.9 ms and A75 64-light 0.086 s anchors (3080 Ti, `76_*` l.254–255).
- The A75 single-light figures of 1.07 / 1.37 ms are RTX 5080 measurements.
- Every C1 number is an estimate. Per Objection 1 it is also a lower bound, because the far field is missing.

## Novelty re-check of the specific revised mechanism

There is no revised mechanism. The corrected specification is:
- flat-source CP [PA: Marleau–Hébert; DRAGON];
- a first-collision source [PA: Alcouffe; Fan–An–Ying §4];
- matrix-free characteristics-style far-field sweeps [PA: the method of characteristics, from memory];
- or a hierarchical cluster far field [PA: Sillion 1995 TVCG; Smits–Arvo–Greenberg 1994];
- a successive-orders (Neumann) solve [PA];
- local rebuild with a global re-solve [PA: Drettakis–Sillion 1997].

Verdict: a better implementation of a known method, not a method.

## What a C1 pass or fail could establish

**A pass** would show only that the MC-defined K is computable from declared physics at L0/L1 on the scored strata. C1 also has privileged information the learned arms lacked: σ_t, ω, g = 0 and index matching (WL77A B4).

**Its one actionable use** is as an equal-accuracy or equal-runtime classical comparator for any future learned-K claim. This applies only in the homogeneous g = 0 scope, and only after its far field and costs are actually specified and measured.

## Sources
- [Sillion 1995, "A Unified Hierarchical Algorithm for Global Illumination with Scattering Volumes and Object Clusters" (core.ac.uk)](https://core.ac.uk/download/50063589.pdf)
- [Sillion, Drettakis & Soler 1995, clustering algorithm (SIGGRAPH history)](https://history.siggraph.org/?p=118730)
- [Pérez, Pueyo & Sillion survey](https://rd.springer.com/chapter/10.1007/978-3-7091-6858-5_28)
- [Bar-Sinai et al. 2019, arXiv 1808.04930](https://arxiv.org/abs/1808.04930v4)
- [Brenner group publication page](https://brennergroup.seas.harvard.edu/publications/learning-data-driven-discretizations-partial-differential-equations)
- [McGreivy & Hakim 2024, arXiv 2407.07218](https://arxiv.org/html/2407.07218v1)
- [McGreivy & Hakim 2024, Princeton record](https://collaborate.princeton.edu/en/publications/weak-baselines-and-reporting-biases-lead-to-overoptimism-in-machi/)
- [Successive order of scattering (Thermopedia)](https://thermopedia.com/content/256/)
- [Ocean Optics Book, single-scattering / SOS](https://oceanopticsbook.info/view/radiative-transfer-theory/level-2/the-single-scattering-approximation)

Repository files I re-read, read-only:
- `C:\Projects\ReST-GI\docs\worklogs\76_WORKLOG_FIRST_INTERACTION_ANCHORED_TRANSPORT.md` l.248–256
- `C:\Projects\ReST-GI\docs\plans\wl77a\coordinator_decision.md` l.10–16
- `C:\Projects\ReST-GI\docs\BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md` l.8–32

---

## Architect response

# WL80B: C1 architect (stand-in) response to adversarial reviews 1 and 2

This is design only. I read a few files to check claims, ran no computation and wrote no file. I re-checked these sources:
- `BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md` l.9–19
- `wl77a/coordinator_decision.md` l.13–15 and l.38–39
- `wl80a/cross_review.md` l.949 and l.957
- `wl80a/neutral_problem_brief.md` l.33–70
- `76_WORKLOG_FIRST_INTERACTION_ANCHORED_TRANSPORT.md` l.248–255
- `experiments/phys_ref53/p53_tracer.py` l.153–195

Every other figure the reviews cite is marked "(not re-verified)".

**Labels:** [EP] established physics · [PA] prior art · [HM] historical measurement or record · [WD] WL80A design claim · [ND] my new derivation · [UH] untested hypothesis.

**Bottom line:**
- I still propose no revised architecture, and both reviews agree.
- The "corrected specification" (redesign §3) is withdrawn as an executable specification for the dented pose.
- The amendments below are numerics inside the CP / first-collision-source (FCS) class. They keep the central mechanism and do not change the question.

---

## Review 1 (physics and generality)

### R1-1 BLOCKER: no admissible W fits on the dented pose — CONCEDE, with a partial AMEND for the rest pose only

**What I concede**
- Storing W out to 10 mfp needs about 1.6e9 nonzeros, about 13 GB, which does not fit [WD, `cross_review.md` l.957, verified].
- A matrix-free far field over 525–663 sweeps is infeasible.
- The decisive point is truncation [ND, I agree]. Under complement escape, 1 − 1ᵀW_trunc silently books the truncated next-collision mass as escape from the source cell. Closure, passivity and partition tests all still pass, while first-exit location and direction are wrong.
- My E10 claim "complement-defined escape" is therefore physically correct only for an untruncated W. E5's "no clamp exists" holds only in the same case.

**Consequences**
1. On the dented pose the specification is NOT EXECUTABLE.
2. E10 gains a mandatory diagnostic: per cell, compute escape both directly by quadrature and as the complement. The difference bounds the truncated mass plus the quadrature error, and must be reported per stratum.
3. Rest-pose AMEND (corrects my E4 rationale, see R2-8). The rest slab is convex, so a float64 layer-pair Toeplitz/FFT far field built from exact ray-telescoped cell-pair kernels needs no visibility. It has no grid projection, so no pFFT mass error.
   - It is non-negative up to float64 FFT roundoff, which is far inside the −1e-9·max test, and needs only layer-pair kernels in memory [ND].
   - Side grading breaks the Toeplitz structure near the four side faces, so those regions need an explicit correction [UH].
   - This keeps the mechanism and applies to the rest pose only.
4. For the dent, the only prior-art route I can see is a non-negative cluster-hierarchy far field with visibility refinement of cavity-straddling cluster pairs (Sillion 1995; Smits–Arvo–Greenberg 1994 [PA, cited by R2, not re-verified]). Its memory, positivity and partition are [UH]; see R2-1.
5. The dent remains UNRESOLVED.

### R1-2 MAJOR: the closure test contradicts the fixed-count solve — CONCEDE + AMEND

- My own derivation agrees: r_N = 1ᵀ(ωW)^N ωG1u. For red, ρ^N = 0.974^663 = e^{−17.5} ≈ 2.6e-8 [ND, checked], which is about seven orders of magnitude above float64 roundoff.
- **AMEND.** Tally r_N as an explicit "truncation" category. The hard test becomes d = ΣK + absorbed + r_N to fp precision.
- The reference has a truncated category too (`p53_tracer.py:160`, `st["truncated"]` at max_steps) [HM]. But the reference uses Russian roulette (`rr_rel=0.05`, l.153), which is unbiased, while C1's r_N is a bias.
- r_N must therefore also enter the accuracy budget (R2-6). It is not just a bookkeeping category.

### R1-3 MAJOR: rotated direction sets conflict with D4h and translation templates — partly REBUT, partly CONCEDE, with an AMEND

**REBUT the D4h half**
- Exact D4h needs only R(g·p) = g∘R(p), meaning covariance [ND].
- Orbit-representative assembly (my E14) supplies this by construction. R(p) is chosen freely, de-aliased and not lattice-periodic, inside one fundamental domain, and its images are defined by permutation.
- So the coherent aliasing comes back only under lattice periodicity, which D4h does not require.

**CONCEDE the template half**
- A non-periodic R loses in-face translation templates.
- The rest construction cost reverts to the untemplated count, about 6e12 evaluations (Part E5, not re-verified).
- Exact equality of translates is lost. It is not a contract invariant: the brief (l.65–69) states only point-group symmetry.

**CONCEDE receiver-end ray effects**
- A Φout (j, o) shape traced with 1e3–1e4 rays per near-surface cell is an aliasing floor that does not refine with h.
- **AMEND (same mechanism).** Compute near-surface cell escape shapes by stratified quadrature per outgoing (j, o) target bin instead of by shooting rays from the source cell. The ray count becomes a third refinement axis with its own self-convergence test (see R1-6). Its cost is [UH].

### R1-4 MAJOR: contract-mode exactness does not carry to larger bodies — partly REBUT, partly CONCEDE

**REBUT the claim as stated**
- A cold-start, fixed-count Neumann iteration is exactly linear, zero-in-zero-out, non-negative, monotone passive and equivariant at any N [ND; R2-6 says the same].
- Superposition at 1e-6 and min entry ≥ −1e-9·max therefore stay method properties for any body. They do not become tolerance properties.

**CONCEDE what actually fails: accuracy**
- For 1 − ρ ≲ 1e-3, r_N stays large at any affordable N. The reviewer's sweep counts (≈1e4 for a 100 mm red cube) are consistent with ln(1/(τ(1−ρ)))/(1−ρ) [ND, checked].
- So "invariant-exact and accurate at affordable cost" is restricted to thin or absorbing bodies. This is a dilemma, not a tolerance relabelling.

**AMEND (option)**
- A fixed-coefficient polynomial acceleration (Chebyshev semi-iteration with predeclared coefficients) or a fixed linear two-grid correction stays exactly linear and zero-in-zero-out. It is also D4h-equivariant if the coarse operator is.
- It forfeits structural non-negativity and passivity, which would then be tested, not guaranteed [ND].
- This is narrower than "accelerated = linear to tolerance". Only adaptive Krylov methods are non-linear.

### R1-5 MAJOR: absorbing media, blue, and cell-averaged order-2 escape — CONCEDE + AMEND

- I concede the inconsistency. A point-exact order-2 escape is inside the CP/FCS class (the H3 θ = 0 analogue). Keeping it cell-averaged was a choice, not a consequence of the stop rule.
- I concede that Part C4.1's "small blue coarse impact" is unsupported: h/ℓ = 1.25/1.63 ≈ 0.77 [ND, checked]. For ω = 0.5 the diffusion length √(D/σ_a) ≈ 0.82 mm, indicative only [EP].
- **AMEND.**
  - Add a second per-beam point generation (order-1 point → order-2 point) and start the cell resolvent at order 3.
  - This roughly multiplies construction cost by the second-generation point count [UH].
  - It binds most where ω is low, because orders 1–2 then carry most of the exit.

### R1-6 MAJOR: two levels cannot separate a floor from convergence — CONCEDE + AMEND

- The reviewer's example checks out: 0.090 → 0.047 gives an observed order of log2(1.915) ≈ 0.94 even though a 0.03 floor is present [ND].
- **AMEND the falsification rule.**
  - Use three levels per refinement axis (h_lat, h_depth, and the angular quadrature count from R1-3).
  - Run Richardson self-convergence on C1 alone, which needs no reference. Check consistency with the predicted order (about 2 for lateral flat-source bias).
  - Estimate the floor separately as (extrapolated C1 − MC), judged against the MC replica spread.
- Where three levels are unaffordable (L1 explicit is 13–19 GB, Part E1, not re-verified; most BND and dent strata), the outcome is INCONCLUSIVE by rule, never "affordability-limited pass/fail".
- This tightens CD HIGH-1. It does not relax it.

### R1-7 MAJOR: mesh specified in mm, not mfp — CONCEDE

- The grading must be stated in optical units (h·σ_t). Cell count scales roughly as (σ_t·L)³ for a fixed body [ND].
- Inside the brief's scope (σ_t = 1000 m⁻¹, l.34 [HM]) nothing changes.
- The generality wording in reason 4 is withdrawn (see R1-8).

### R1-8 MAJOR: a non-isotropic phase function is incompatible — CONCEDE

- For g ≠ 0, a scalar cell density with scalar complement escape is not exact beyond order 2 [EP]. The state would need angular moments, which is the S_N/MOC class [PA] and a different architecture.
- A transport-corrected chain is a model error that plateaus, so under HIGH-1 it is a C1 fail.
- Reason 4 is restated: "exact physics supplies the whole operator" holds only for g = 0, isotropic phase (brief l.36 [HM], verified), homogeneous, index-matched.

### R1-9 MAJOR: heterogeneous media extend only partly — CONCEDE

- I accept the reviewer's table.
- Addition: an index mismatch with Fresnel internal reflection keeps reflected light inside K's ownership. That needs directional boundary currents, i.e. interface-current/MOC state. As recorded it is incompatible, and an extension would be a different state [ND].

### R1-10 MAJOR: the dent stretch map conflicts with offset tiling — CONCEDE + AMEND

- **AMEND (adopting the reviewer's rule; it unifies E12 and E17).**
  - Keep fixed-thickness normal-offset boundary stacks under the top surface (which follows the dent) and over the fixed bottom.
  - Compress only the interior band between them.
- Under the brief's maximum dent depth of 6 mm (l.52 [HM]), 4 mm of material remains, and 2 × 1.1 = 2.2 mm < 4 mm leaves a 1.8 mm interior band [ND].
- The bottom stack stays bitwise unchanged.
- Rest/dent bias cancellation in the signed change remains [UH].

### R1-11 MINOR: offset tiling fails at small curvature radii — partly REBUT, partly CONCEDE

**REBUT the limiting depth**
- Under the R1-10 amendment, offsets go only 1.1 mm deep, not up to T/2 = 5 mm.
- The cosine-profile radius 2a²/(hπ²) is 7.6 mm (15 × 6 mm) and 17.8 mm (22.5 × 5.75 mm) [ND, checked].
- The convex rim is the binding location, and its radius exceeds 1.1 mm by a wide margin. The bowl floor offsets diverge into the medium.

**CONCEDE the scope**
- Any convex feature with radius below about 1.1 mm, and bends outside the brief, need general tetrahedral CP cells.

### R1-12 MINOR: three proposed checks are self-referential — CONCEDE + AMEND

1. "ΣF_in = d_new" is a tautology. **Replace** it with an independent β check: semi-analytic or independent-code high-q integration of e^{−σ_t L} over port area and cosine-weighted bin solid angle, required at least on edge-port bins aimed at side faces.
2. The two-sided bound becomes independent only when 1ᵀΦ_pt q is taken from an MC single-scatter tally.
3. "Cavity-crossing mass = 0" is zero by construction.
   - **Replace** it with a reference-side check: the walker terminates at the first exit (`continue_air=False`, `p53_tracer.py:153`, and the re-entry branch is only taken when `continue_air` is true, l.177 [HM]).
   - **Add** a diagnostic: the mass a visibility-free W would have mis-owned.

### R1-13 MINOR: the reciprocity reasoning in E15 is inconsistent — CONCEDE

- An adjoint, point-exact last flight is within the class. E15's "not fixable within C1" is withdrawn.
- The fix is the same option as R1-5's order-2 point layer, applied at the receiver end.
- Reciprocity remains non-required (brief l.70 [HM]) and is reported only as a diagnostic.

### R1-14 MINOR: the convention gate uses a different geometry — partly REBUT, partly CONCEDE, with an AMEND

- **REBUT on purpose.** The gate tests the tally convention (frame, no cosine factor), not the accuracy of the scored discretisation.
- **CONCEDE** that at ω = 1 it does not bite.
- **AMEND.** Run it on the scored mesh: blue central top ports, where the diffusion length √(D/σ_a) ≈ 1.8 mm is much smaller than 10 mm and 100 mm, so the slab is effectively a half-space [ND]. Add ω < 1 test values. Keep the single-scatter S1 versus MC single-scatter tally, which already uses the scored mesh.

### R1-15 MINOR: concentrated incidence is limited by the contract — CONCEDE

- The E2 wording becomes "first flight exact relative to bin- and port-uniform contract input".

### R1 tables H2 / H3

I accept both, with these corrections from the rebuttals above:

| Item | Correction | Source |
|---|---|---|
| Near-conservative (H2) | The invariants stay exact at any N; accuracy is the limit | R1-4 |
| Exact D4h (H3) | YES by orbit-representative assembly; the cost is lost templates | R1-3 |
| Energy closure (H3) | YES with the r_N tally | R1-2 |
| First-exit ownership (H3) | Rest: plausibly YES with the convex Toeplitz far field [UH]. Dent: NOT ESTABLISHED | R1-1 |

---

## Review 2 (novelty, computation, learnability)

### R2-1 BLOCKER: the far-field cost is omitted, and matrix-free reintroduces ray effects — CONCEDE

- The 0.7–1.0 s contract-mode figure was near-field-only. It is withdrawn as a cost and kept only as a lower bound [ND].
- The dilemma is real: matrix-free telescoping is either infeasible at about 1e12 segment deposits per sweep (arithmetic not re-verified, order of magnitude accepted) or ray-effect-limited.
- **AMEND direction.** Build the far field once and store it compressed:
  - rest pose: as in R1-1;
  - dent: a cluster hierarchy, which must be shown to fit within 12 GB alongside 2.5–6.8 GB of factors [UH].
- A second trade-off appears [ND]. Cell-pair quadrature (point-to-point e^{−τ}/4πr² with visibility) has no source-end ray effects but loses the telescoping guarantee 1ᵀW ≤ 1. It would then need predeclared quadrature normalisation, a standard CP step [PA] that is not a stabiliser, with the magnitude reported.
- The dent remains UNRESOLVED.

### R2-2 MAJOR: "the only construction" is false; cluster hierarchies are prior art — CONCEDE

- E4/E5's "only" is withdrawn.
- Part F tightens. The whole system is a volume-plus-surface zonal hierarchy with a local rebuild and a global re-solve (Sillion 1995; Drettakis–Sillion 1997 [PA]). That makes it ALREADY KNOWN at system level, not "UNCERTAIN, engineering level".
- The 2.5 mm × 64-bin interface is a parameter, not a contribution.

### R2-3 MAJOR: "deterministic analogue of the WL56 MC oracle K" is the wrong role label — CONCEDE

- `BASELINE_ROLES` l.12 (the exact domain-selective oracle) and l.19 (Cycles volume as the physical reference) are verified [HM].
- C1 is a biased deterministic approximation with an unmeasured bias.
- It is not an oracle, not a reference, not a label or F/S invalidation source, and not "physical".
- No role row exists for a classical deterministic solver comparator. Adding one is a human decision.
- The wording I had copied from CD l.17 is withdrawn.

### R2-4 MAJOR: "a network adds no information" is too strong — CONCEDE the reason; E19 REMOVE still stands for different reasons

**CONCEDE**
- A closure supervised by C1's own L1/L0 refinement is identifiable without MC noise (Bar-Sinai et al. 2019 [PA, cited by R2]).
- Horn A (`coordinator_decision.md` l.15, verified) was argued for MC-referenced closures and does not transfer automatically.
- Canonical-patch L1 teachers are affordable even where full-scene L1 is not [ND].

**The corrected reasons for REMOVE**
1. It is H3-class, with H3's remaining grounds (admissibility, unbuilt-kernel cost, scope; l.15).
2. It learns discretisation error, not intrinsic transport, so it does not address the learned-K transfer question.
3. Its benefit is a constant-factor memory/accuracy trade.
4. It faces its own out-of-distribution question on stretched or dented cells.
5. Adding it would change C1's learning-free identity.

A non-learned alternative exists: linear-source CP [PA], whose positivity is [UH].

### R2-5 MAJOR: no cost case for dent or per-frame updates — CONCEDE

- Stated conclusion: per-frame deformation is outside C1's plausible operating region [ND].
- Comparison classes:
  - Neural Full Recompute is NOT APPLICABLE, because C1 owns no learned representation.
  - Internal F/S and a class-3 path-traced re-render are UNMEASURED.
- Measured anchors:
  - A75 full geometry rebuild 0.015 s and FIA 75.0 s (`76_*` l.252 [HM, verified]);
  - WL59 S 0.33 s and Q78 0.102 s (not re-verified).

### R2-6 MINOR: the sweep count is inconsistent — CONCEDE + AMEND

- 663 came from the converged-solve rule and does not apply to fixed-count Neumann. The 524 vs 663 mismatch is acknowledged.
- For red, ρ^N ≤ 1e-3 gives about 262 sweeps and ρ^N ≤ 1e-2 about 175 [ND, checked].
- **Caveat [ND].** r_N bounds the total missing exit mass, but late collision orders dominate the far lateral tail. The per-stratum relative truncation error at distant coarse cells can therefore greatly exceed r_N/d.
- **AMEND.** Set N per stratum from an a-posteriori N versus 2N difference, not from the column-total remainder.
- "Far inside the 0.10 gates" is not established for far strata.

### R2-7 MINOR: one collision-order sequence serves RGB and every uniform albedo — CONCEDE + AMEND

- The moments m_k = Wᵏ G1 u are channel-free (σ_t channel-independent, brief l.34) [ND].
- ψ_c = Σ ω_c^{k+1} m_k, with N set by red.
- Storing the per-order exit tallies Φout m_k costs about 265 × 245,760 floats ≈ 260 MB per light (float32).
- A uniform-albedo edit then needs no re-solve, which corrects E16 and Part D case 5.
- This does not hold for spatially varying ω(x). It is not novel [PA, successive orders of scattering].

### R2-8 MINOR: the pFFT negativity rationale is wrong — CONCEDE

- An assembled pFFT with non-negative low-order projection/interpolation is ≥ 0 in exact arithmetic [ND, agree].
- E4(a) is withdrawn. E4's REMOVE stands on (b) float32 roundoff, (c) mass error, (d) dent visibility and (e) grid commensurability.
- The rest-pose route is now the exact-kernel float64 Toeplitz far field (R1-1).

### R2-9 MINOR: cost anchors mix hardware — CONCEDE

- C1 may be compared only with 3080 Ti anchors: FIA 3.9 ms one-light and A75 0.086 s for 64 lights (`76_*` l.254–255 [HM, verified]).
- The A75 1.07 / 1.37 ms figures are RTX 5080 (not re-verified) and are excluded.
- The "4–260×" ratio is withdrawn. C1's cost is UNKNOWN, and every estimate is a near-field-only lower bound.

### R2 recommendation ("C1 PHYSICAL BASELINE ONLY") — partly CONCEDE, with an AMEND to the wording

I accept the substance:
- C1's only defensible potential use is as an equal-accuracy or equal-runtime classical numerical comparator for a future learned-K claim, within its scope (McGreivy & Hakim 2024 [PA, cited by R2]).

I amend the wording:
- "numerical", not "physical";
- conditional on its bias against the Cycles volume reference being measured, and on a competent far field existing for every stratum it is compared on;
- the role itself is a human decision.

---

## Consolidated amendments

All of these stay inside the CP/FCS class and keep the central mechanism and the question.

1. Tally r_N explicitly; set N per stratum.
2. Rest far field: float64 exact-kernel layer-pair Toeplitz, with side corrections.
3. Dent far field: UNRESOLVED (cluster hierarchy [UH]).
4. Report direct versus complement escape.
5. Orbit-representative assembly with a de-aliased R(p); templates dropped.
6. Stratified per-target Φout shapes; angular count as a refinement axis.
7. Optional order-2 point layer (source end) and adjoint last flight (receiver end).
8. Three-level Richardson rule with a separate floor estimate; INCONCLUSIVE where unaffordable.
9. Mesh stated in optical units.
10. Fixed boundary stacks with a compressed interior for the dent.
11. Independent β check, reference-side ownership check, H-function gate on the scored mesh.
12. Per-order exit tallies for RGB and albedo reuse.
13. Fixed-coefficient acceleration allowed, with non-negativity and passivity tested rather than guaranteed.

Every objection is retained. None is erased.

## Updated classification of C1

- **Identity.** A learning-free classical deterministic integral-transport solver: flat-source CP with an exact first-collision source and complement escape, at port/bin first-exit resolution. It is ALREADY KNOWN at component and system level (CP/FCS/MOC, zonal cluster hierarchy, local rebuild plus global re-solve [PA]). It has no novelty.
- **Scope of exactness (continuum).**
  - g = 0, isotropic phase, index-matched;
  - homogeneous, ω(x) only, or cell-aligned piecewise σ_t;
  - mesh in optical units;
  - accurate at affordable N only for thin or absorbing bodies.
  - Incompatible with g ≠ 0 beyond order 2 and with Fresnel boundaries as recorded.
- **Specification status.**
  - Rest pose: plausibly executable after the amendments; cost UNKNOWN.
  - Dented pose: NOT EXECUTABLE. No admissible, non-negative, visibility-exact, untruncated W within 12 GB has been specified.
  - Per-frame deformation: outside its plausible operating region.
- **Role.**
  - Not an FNTO learned K (contract v0.1).
  - Not an oracle, physical reference, label source or F/S invalidation instrument.
  - Its only defensible potential role is a classical numerical comparator in its scope, after the far field is specified and the bias is measured. That role, admissibility and every comparator are human decisions.
- **Research question.** C1 does not address it: it does not test learned-K transfer.
- **Evidence.** Nothing is measured. Every magnitude above is [ND], [UH] or a cited [HM]. Building or running C1 is not authorised.
- **Disposition.** NO REVISION. C1 is downgraded from "candidate operator / diagnostic instrument" to "known classical solver; specification incomplete (dent); potential numerical comparator only".
