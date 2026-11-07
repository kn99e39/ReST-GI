# WL77A Candidate-Author Responses to Reviews A and B (2026-11-07)

**Role.** Advocate for the candidate authors (H1 CQR, H2 RBG, H3 LCR) in the WL77A architecture-discovery workflow. For every objection in `review_A_physics.md` (A1–A15) and `review_B_novelty_systems.md` (B1–B12), this file gives the strongest honest response for the challenged candidates: a rebuttal where the evidence supports one, and a clear concession where it does not.

**Rules followed.**
- Read-only on the repository except this file. No GPU job, training, server access or Monte Carlo was run.
- WL77 files, historical worklogs and results are untouched. The Gate 4 population (`ports76.json`) and its references were not opened.
- No candidate's central mechanism is changed to escape an objection. The central mechanisms are: H1, the symmetry quotient plus identity-free learned pair blocks; H2, a learned scalar Γ between exact reciprocal projections; H3, a learned local closure inside an exact collision-probability (CPM) solve. Where only a non-central detail would fix an objection (a feature cutoff, a symmetric quadrature pattern, a guard form, a budget statement), the detail is named. The fix is not claimed as already specified, and the original claim is still conceded.

**Labels.**
- **[V]** verified by me in code or results (path:line).
- **[D]** my derivation.
- **[E]** my estimate.
- **[S]** from a specialist, candidate or review file, not rechecked.

**Verification done for this response (read-only).**
- `experiments/fnto_cmp72/rsk72.py:16-63`: RSK features, far field and normalisation.
- `experiments/fnto_rule57/r57_geometry.py:15-35, 52-66, 80-93`: POSES, node features, ballistic seed.
- `experiments/fnto_repr68/eval68.py:1-80`: contrast map L = H·diag(1/d).
- `experiments/fnto_fia77/gate77.py:40-55`: M = ref − SS_q; noise from the S half-split only.
- `experiments/fnto_fia76/fia76.py:65-89`: one-exit-ray SS.
- `experiments/phys_ref53/p53_geom.py:168-173`: map_dent, b = cos²(πr/2R).
- `results/fnto_src73/ports.json`: S16-R ids; s51 = port 3823, face 4, (48.75, 48.75, −5), orbit 8.
- `docs/worklogs/72_*:150-200`, `docs/worklogs/76_*:70-92`.
- `specialist_physics.md:282, 307`; `specialist_systems.md:340-380`.

---

## 0. Summary

| ID | Candidates | Outcome | One line |
|---|---|---|---|
| A1 | H1 H2 H3 | **Concede** (partial defence) | The translate test is a necessary-condition / unit test, not evidence of generalisation. H1's F-a wrongly pools TBC as "covered". |
| A2 | H2 H3 (H1) | **Concede** for H2 and H3; H1 bounded | The unsaturated 30 mm plane features (H2) and the "third-nearest plane" feature (H3) are location keys. H1's residual channel is small, and C1 does not show far-receiver identity. |
| A3 | H1 H2 H3 | **Concede the claims**; **rebut the precedent** | RSK is not an identity-free precedent: it had a whole-slab far field, a global normaliser, a displacement channel and a 50 mm edge key [V]. The no-drift claims are still only predictions out of orbit, and H3's c_max bounds the parameter, not the error. |
| A4 | H1 H2 H3 | **Concede** (the scaling is partly rebutted) | The quoted floors are extrapolations in the wrong metric. The floor stages already use eval68 con_coarse. The 2–3× rescaling holds for amplitude-type errors and not for shift-type errors. |
| A5 | H2 (H3) | **Concede** | H2's SS voxel product of line deposits is biased by aliasing. H3's D2 depth truncation is a real omission. |
| A6 | H2 H3 (H1) | **Concede** | FIA/FIA-AZ with deterministic SS is the missing minimal competitor. |
| A7 | H1 H2 | **Concede** | The source-side information is a few moments per source. The near-diagonal block at intermediate plane distances is identified only by smoothness. |
| A8 | H3 | **Rebut Horn B as stated**; **concede Horn A and the contract question** | The exact coarse correction L* is source-independent and plausibly quasi-local. The within-element remainder is source-dependent, as H3 §9.2 already states. |
| A9 | H3 H2 | **Concede** | H2 must declare S, ω, g = 0, homogeneity and index matching as privileged physics. The scope limits stand. |
| A10 | H1 | **Concede** (minor) | An output ∝ MC d_B carries an unlearnable δd/d. The cost is about 0.04–0.07 con at edge/rim sources, including training sources. |
| A11 | H1 | **Concede**, scoped | The discontinuity exists. It cannot occur at pt0; it is limited to degenerate side↔top projections near dents close to a side face (e.g. te2). |
| A12 | H2 | **Concede** (partial) | Exact representability holds only where the lumping error is below the absorption margin. A symmetric s_X·s_Y guard is an available non-central fix. |
| A13 | H3 | **Concede** | Positivity holds to ε. The Neumann fallback is about 437 iterations, not 260, and the cost depends on θ. |
| A14 | H3 (H2) | **Concede** DEF and the s51 slip | The consequence the reviewer draws is rebutted: top corners are covered by z-flip and are identical at pt0. |
| A15 | H2 H3 | **Concede** | Stage 1 validated only the coarse source-side decomposition. A native Gram-identity check belongs in Stage 0. |
| B1 | H1 H2 H3 | **Concede** | Accept the W+UV-Q control and separate orbit and non-orbit falsifiers. |
| B2 | H1 | **Concede** state and 1-light; **rebut** 64-light | A dense near-block layout gives about 1.5 GB. A batched-bmm 64-light apply plausibly stays within the cap. |
| B3 | H1 | **Concede** | Weighted frames are Dym 2024 and frame averaging is Puny 2022. Add an H1-Reynolds arm. |
| B4 | H2 H3 | **Concede** | The learning-free deployable control must be the primary comparator. F4 needs an "inconclusive" outcome. |
| B5 | H3 | **Concede** the claim and F5 as written; **rebut "only at rest"** | With translation-keyed on-the-fly coefficients the roofline is about 0.3 ms at rest and about 1.5 ms at pt0. A torch-style F5 fails. |
| B6 | H3 H2 | **Concede** (partial rebuttal for H2) | H3's cubature and visibility counts are too low. H2's dent-footprint Π fits the cast cap and its wide normaliser radius affects only cheap learned coefficients, but its full-pose Π and dent SS exceed the caps. |
| B7 | H2 | **Concede** | Cite NGF, Element Learning, DeepRTE, Wang 2005 and Garg 2006. The claim is a combination plus an evaluation. |
| B8 | H3 | **Concede** | Add RTE ML closures. |
| B9 | H1 H2 H3 | **Concede** | Symmetrise H1's 256-point pattern at equal cost. d_B noise breaks image identity in every candidate (and in H2's and H3's guards). |
| B10 | H1 H2 H3 | **Concede** | Use sub-cell-matched ratios, or ratios to each source's own floor. |
| B11 | H1 H2 H3 | **Concede** | Use ≥ 3 seeds or a paired-replica interval on the difference. |
| B12 | H1 H2 H3 | **Concede** | Scope bound only. H2's 18.3 mm is the red diffusion length, which governs ρ in thick media. |

**Net position.** The candidates' invariants (linearity, positivity, zero → zero, passivity, K/C ownership) survive both reviews. No objection is fatal.

What does **not** survive as written:
- the translate test as selection evidence (A1, B1);
- the identity-free claims of H2 and H3 under their current feature sets (A2);
- every quoted representation floor (A4);
- H2's SS cubature (A5);
- several cost statements (B2, B5, B6).

What survives in narrowed form:
- the selection-relevant tests themselves: H1 Stage A, H2 0c and H3 Stage 1 are already eval68 con_coarse floor measurements;
- H3's transfer argument (A7 does not apply to it);
- H3's closure locality, against Horn B (A8).

---

## 1. Responses to Review A

### A-A1. The translate test is a symmetry identity (H1, H2, H3; major) — **Concede, with a partial defence**

**Concession.**
- The reviewer's C1 check shows that the s101/s105/s107 references equal the rigid images of s50/s52/s53 to ≤ 1e-4 relative L2, with energy ratio 1.0000. Any exactly equivariant operator that fits the partners therefore passes, so the test cannot rank H1, H2 and H3 or demonstrate learned generalisation.
- H1's F-a lists "TBE, TBC, SM, SR far" as "contexts covered by the quotient". That is wrong for TBC. s51 is port 3823 at (48.75, 48.75, −5) with `orbit_size` 8 [V, `ports.json`], so all eight (1.25, 1.25) corner ports are its images, and Gate 4 excludes images (`ports76.py:1-5`). Every Gate-4 TBC slot is therefore at a corner offset such as (1.25, 3.75), outside the orbit.
- TBE far slots at e ≠ 1.25 mm are also outside the orbit. TBE slots at e = 1.25 mm whose second edge lies at 10–12 mm are not window-congruent with s50 either, because H1's window is 12 mm.
- F-a must be split into an orbit-equivalent stratum (a unit test: equality with the partner's held-out error) and a non-orbit stratum (the transfer claim). F-c can only be read on the non-orbit stratum.

**Defence (limited).**
- As falsifiers, H2 F-T and H3 F2 are still valid *necessary* conditions: a correct mechanism must pass them, and they can fail through a poor training fit or through non-equivariant residual features. As A2 shows, H2's and H3's features are not exactly translation-equivariant as written, so for those two candidates the test is not automatic. It is non-discriminating, not powerless.
- H1 already labelled Stage C (P1) "development, descriptive, not blind" (`candidate_H1.md` §14 Stage C).
- C1 is an energy-weighted L2 dominated by the near field. It shows aggregate identity. It does not show per-receiver identity at far receivers near the partner's second plane; see A2/H1.

**Accepted change.** Recast every translate prediction as an implementation unit test: the prediction must equal the partner's prediction, up to d_B, to float precision for H1/H3 and up to the declared feature difference for H2. Put all selection weight on non-orbit contexts: TBC offsets, TBE with e ≠ 1.25, SV, SM/SR near a vertical edge, and dents.

### A-A2. Long-range plane features re-create location keys (H2 and H3 major, H1 minor) — **Concede for H2 and H3; H1 bounded**

**H2: concede.**
- §6 gives every face patch within 2ρ_c (30 mm red) the features d and exp(−d/18.3 mm).
- s50's second plane (18.75 mm) is present; s101's (41.25 mm) is absent. The plane-distance tuple is therefore unique to each boundary S16-R source and to each interior source within 30 mm of a side. A shared Γ correction can be keyed on it.
- 18.3 mm is the infinite-medium red diffusion length L_d = 1/√(3σ_aσ_t) = 1/√(3 · 1 · 1000 m⁻²) = 18.26 mm [D]. Green 5.77 mm and blue 1.83 mm match H2's other two scales. In this 10 mm slab, the plane-influence scale is the slab-confined lateral decay of about 3.6 / 3.1 / 1.7 mm [S]. The feature range is therefore physically wrong for this fixture.
- A hard cutoff of the plane set at the slab decay scale (about 3 × 3.6 ≈ 11 mm for red) is a non-central feature change that would remove the key. As specified, H2-C4 does not hold.

**H3: concede.**
- "Sorted distances to the three nearest boundary planes" (§6 item 4) is unsaturated. For interior elements the third plane is the nearest side, 20–50 mm away, which makes it a quasi-absolute coordinate.
- The closure is therefore not translation-equivariant for interior translates, and H3 §7(a)'s "exactly translation-equivariant" is false as written.
- The physics of a one-stencil closure needs no plane information beyond a few mm. A cutoff of about one or two stencil widths would make the feature set match the "local" claim, but that is not what is specified.

**H1: bounded, minor.**
- The source context is hard-windowed at 12 mm, so the source side has no key. The residual channel is the far-zone receiver moments (ℓ = 10 mm, window 12 mm), which mark s50's far receivers near the y = −50 plane.
- Two points limit this:
  1. Those receivers really are within 3.75–12 mm of a plane, so the moment is a physically legitimate feature for the receiver's own exit. C1's ≤ 1e-4 is an energy-weighted aggregate dominated by near receivers, so it does not establish that the far-receiver responses are identical [D].
  2. The far zone carries ≤ 11 % of native L2 [S], and the plane-adjacent subset is a fraction of that.
- Any key-like effect is therefore confined to a small, energy-light block. A shorter receiver-moment scale (≤ 3.6 mm) would be a non-central tightening.

### A-A3. Drift is not removed by deleting identity parameters (H1, H2, H3; major) — **Concede the claims; rebut the precedent**

**The RSK precedent is not identity-free.** I checked `rsk72.py`. RSK retained several location-, identity- or pose-bearing channels:
- A **whole-slab far field.** `phi_far` is evaluated on all 240 coarse cells, using each cell's relative offset from the source (`rsk72.py:57`). The set of offsets to all cells fingerprints absolute position: a near-corner source sees cells on one side only.
- A **global normaliser.** `scale = mass/(Σ w_loc + Σ w_far)` sums over all 240 cells (`:63`). This is a U-like global softmax.
- A **20 mm whole-slab context kernel.** `ctx` is a kernel-weighted average of all cell descriptors (`:34`).
- **Location and pose keys in the descriptor** (`:28-29`): `edge` clamped at 50 mm, which A2 itself calls a location key, and the rest-displacement channel `nd[:,6:9] = d/.006` (`r57_geometry.py:58`).
- **Frozen global CP A/O** (`:48`), and world-axis frames without a D4 quotient (`:16`).

RSK's drift (0.589 → 0.629 and 0.328 → 0.560, `72_*` checkpoint table) is therefore consistent with the location-key mechanism. It is not evidence that a strictly local, identity-free, quotiented model drifts. None of H1–H3 has a whole-slab field or a global normaliser.

**Concessions.**
- (i) H1-C4 (final/@2k ≤ 1.2), H2 §7(4) and H3-C4 are *predictions*. Late training still changes shared MLP weights to reduce residuals on supervised inputs. That moves outputs in unsupervised feature regions (SV, two-plane corners, corner offsets), which only positivity and passivity constrain. H1's argument that "a null-space direction moves every source with the same canonical inputs" protects orbit-equivalent sources only. It does not protect out-of-orbit ones.
- (ii) With A2's location keys present, H2 and H3 also have a drift route as written.
- (iii) **H3's c_max.** I concede that c_max = 0.5 bounds the parameter, not the error.
  - Estimate [D]: an element of about 0.5 × 1.25 × 1.25 mm at a vertical edge with physical non-escape of about 0.6, of which about 0.45 lies in the one-stencil couplings.
  - A 1.5× boost raises that to about 0.45 × 1.5 + 0.15 ≈ 0.83. That is a local escape change of about 40 %: below the reviewer's "to the cap" bound, but O(1).
  - Regularisation toward θ = 0 on weakly supervised classes, or a smaller c_max, would be non-central but must be predeclared.

**What remains defensible.**
- In H1, orbit-equivalent sources cannot drift independently of their partners' training error.
- In H3, the solve keeps transport physically composed (ρ < 1 and the certificate s), so a closure excursion is local and passive. It cannot become a long-range per-source table.

### A-A4. The representation floors use the wrong metric (H1, H2, H3; major) — **Concede the numbers; partly rebut the scaling**

**Concession.**
- The specialist "pair contrast" is a two-bin (b0 − b5) coarse difference at a cell-corner sub-cell, with signal |ΔZ/2|/|Z̄| = 0.12–0.16 (`specialist_physics.md:282`).
- eval68 con_coarse is the 16-bin centred map H·diag(1/d), pooled ΣE/ΣS per channel, worst channel, jackknife hi (`eval68.py:3-4, 56-80`) [V].
- H1-C2 (≤ 0.10), H2-C2 (≤ 0.15) and H3-C2/C8 (≤ 0.15) are therefore unsupported extrapolations.

**H2's first-collision projection: concede.**
- The physics specialist warns explicitly that snapping the first collision "would remove the 0.3–0.8 mm footprint shift" (`specialist_physics.md:307`).
- H2's hats keep the first moment (barycentre), which pure column snapping does not. But the shape is lost at h = 2.5 mm > mfp = 1 mm, before the first S flight, where 66–94 % of exit lands within 3.6 mm [S]. Counterexample (A) is a likely L1 outcome, not a tail risk, as H2 §13(A) itself warns.

**Rebuttal, partial.** The 2–3× rescaling is not universal [D].
- Write the coarse contrast as H Z, with Z_a ≈ Z_0 + ∇Z·δ_a, where δ_a is the incidence-dependent footprint shift. At a centre sub-cell, ∇Z (cross-boundary sensitivity) is small, which is why |HZ|/|Z| ≈ 0.054.
- A **shift-type** model error ε_a gives an error ∇Z·ε_a, so the relative error is ε/δ, independent of the sub-cell. Lateral discretisation of a correctly positioned footprint is mostly of this type.
- An **amplitude-type** error η_a Z_0 (a bin-dependent energy bias) gives a relative error η|Z_0|/(|∇Z||δ|), which does scale up 2–3× at centre sub-cells.
- Concession on the type: for voxelised k ≥ 2 transport the depth-dependent energy bias (+2 % at int b5 [S]) differs between shallow (grazing) and deep (normal) bins. So the amplitude type is plausibly the dominant component for H2 and H3, and the reviewer's figure is the relevant one for them.

**H1 rank-4.**
- The 1.7 % residual is a norm ratio over the whole 16 × 64 block. The bound on the contrast error depends on how much of the block's norm lies in the bin-centred part, which nobody has measured at native level:
  - 1–3 % of *energy* (norm ratio 0.10–0.17) gives a worst case of 0.10–0.17;
  - a norm ratio of 0.03–0.054 gives 0.3–0.6, the reviewer's figure.
- Because rank truncation removes trailing singular directions, and incoming-bin variation lives in singular directions 2–4, the residual is plausibly not orthogonal to the contrast part. So no bound follows either way.

**What is already right.** H1 Stage A (TRAIN con_coarse hi ≤ 0.10), H2 0c (WL59 gates, worst channel, hi) and H3 Stage 1 (WL59 gates on every group) all score eval68 con_coarse. Accepted change: report every floor by sub-cell class, including centre sources such as 2085 [V, S16-R id 0].

### A-A5. The "deterministic" SS cubature is aliased (H2 major, H3 minor) — **Concede**

**H2: concede.** The analysis is correct for H2 as specified (§3.4: 4 × 4 start positions at 0.625 mm, 0.3125 mm voxels, SS = Σ_v P_X P_Y / V_v) [D]:
- At z ≲ 0.3 mm, one 64-bin cell's 16 rays fall in about 16 of the 64 voxel columns under the port, about ¼ coverage.
- Same-port X and Y deposit into the same columns, so the product over-weights them by up to the inverse coverage.
- The bin-dependent lateral offset z·tanθ makes the overlap depend on the bin pair, which is spurious contrast.

H2-C3 ("≤ 0.2 × replica noise in every metric space") is not supported at the stated cost. Stage 0a would detect this, but the cost/accuracy trade-off probably fails. The structurally correct alternative evaluates SS as an entry-chord integral of the exact point-to-bin escape kernel (H3's D1 construction). That is an implementation change of a non-learned term, not of H2's mechanism, but its cost has not been estimated and it is not what H2 specifies.

**H3 (minor): concede.**
- D1's product Gauss rules over bin-clipped, discontinuous integrands converge algebraically.
- D2's fine layer stops at 2 mm depth, so the direct escape of second collisions deeper than 2 mm is dropped from the ω² term. Escape from 2 mm depth is ½E₂(2) ≈ 0.019, against 0.5 at the surface. My estimate is a ≈ 1–3 % omission of D2, about 0.2–0.6 % of collided energy, and it is mostly incidence-independent.
- Either extending 𝔉 or routing the deep remainder into the b̄ term is a non-central fix. H3 Stage 0(i) (rule-doubling ≤ 1e-3) would catch the D1 issue but not the truncation, which needs its own check.

### A-A6. The WL76/77 reinterpretation and the missing FIA control (H2, H3; H1 by motivation; major) — **Concede**

**Verified.** In `gate77.py:45-52`, M = ref − ss, where ss comes from `quad_ports` with q = 1024 and one isotropic exit ray per first collision (`fia76.py:65-89`). The noise is `half` = the S replica half-split only, so SS_q variance is excluded.

**Concession.**
- The magnitude (exact-SS ring 0.22; order ≤ 2 to 0.02–0.03) is a one-cell, two-bin, non-evidence emulation (`specialist_physics.md` §6.4, item 1 of its open list).
- FIA / FIA-AZ with deterministic or high-q SS (and W) is the minimal competitor. It has the same exact first-collision entry and the same source-independent learned emitter, and it is architecturally the closest relative of H2.
- H2-C1 and H3-C8 attribute an expected gain to the exact exit projection, and that is confounded with SS-noise removal.
- H1 §1's "FIA does not even fit" is factually what WL76 measured, but it may reflect quadrature noise rather than the FIA representation.

**What stands.** The candidates' own pass/fail rules (WL59 gates, floors, transfer ratios) are absolute and do not depend on beating FIA. Only the attribution is confounded. The authors support a read-only high-q rerun of the gate77 statistic on archived bundles, and an FIA-detSS control arm before any selection claim that rests on "exact exit vs learned exit".

### A-A7. Out-of-orbit contexts rest on ~4 source configurations (H1 major, H2 major) — **Concede**

**H1.**
- H1 §13 already names this as its strongest counterexample, and H1-C5 is labelled "the claim most likely to fail".
- φ_FC carries first-collision moments only. The edge side-exit fraction (0.25–0.41) and the corner side-exit fraction (0.37–0.56) [S] are multiply-scattered quantities and must be learned to about 1 %.
- The training source-side plane distances are {1.25, 3.75 / 6.25, 8.75, absent} [V, `ports.json`]. Nothing beyond smoothness pins α at e = 3.75–8.75 mm on the top face, at two-plane corners or at SV. This is conceded.

**H2.**
- I concede that "≈ 1.75 × 10⁵ Γ entries constrained per source" overstates the information. A source's data constrain Γ only through Π_Yᵀ Γ Π_X over its 16 rows, and those rows sit on the same ~80–100 footprint nodes with sub-mm shifts. They are nearly collinear: about 3–5 effective source-side vectors.
- A fair count is about (3–5) × (reachable receiver nodes, ~1.75 × 10³) ≈ 10⁴ contracted constraints per source [D].
- Swap symmetry only adds pairs with one end in a supervised footprint. The near-diagonal block with both ends at e ≈ 3.75–8.75 mm is identified by smoothness alone. Under H2's hat basis, s50's footprint (e = 1.25 mm) loads nodes at e ∈ {0, 0.25, 0.5, 1.0, 2.5} and only weakly reaches e = 5 [D].
- That block carries the 32–53 % same-port exit [S]. H2-C7 rests on it, and it is the weakest claim, as H2 already states.

**H3** is correctly noted as not subject to this objection: the boundary physics enters through τ̂ in the solve.

### A-A8. Learning-free solver dilemma (H3; major) — **Rebut Horn B as stated; concede Horn A and the contract question**

**Horn A: concede, as H3 §13 does.**
- If θ = 0 already transfers, the transfer belongs to a classical CPM solve with exact first and second collisions.
- One quibble with "the gap is near noise". eval68 errors are unbiased cross-replica statistics, so a predicted θ = 0 floor of native 0.02–0.06 or coarse ≤ 0.03 is measurable and need not be within the jackknife interval.
- What matters is whether the gap is within the *seed* spread of the closure. That is unknown, and F4 needs a noise-floor "inconclusive" outcome (accepted under B4).

**Horn B: rebut the argument as stated [D].**
- The reviewer's exact correction L* = V(ΦᵀGΦ)⁻¹V is a **fixed matrix that does not depend on the source**. For any source whose second-collision density S e_X is piecewise constant on the elements, b̄_Yᵀ L*⁻¹ b̄_X = b̄_Yᵀ V⁻¹(ΦᵀGΦ)V⁻¹ b̄_X equals the exact ⟨S e_Y, G S e_X⟩. So "useful" and "source-independent" are not in contradiction for this part.
- What is at issue is locality. L* is dense, but ΦᵀGΦ is a well-conditioned SPD matrix (κ ≈ 38 red [S]) whose entries decay exponentially. By decay results for inverses of such matrices (Demko–Moss–Smith / Jaffard type), its inverse also decays exponentially off the diagonal, with a rate set by √κ: roughly (√κ − 1)/(√κ + 1) ≈ 0.72 per kernel bandwidth [D].
- L* − L is therefore quasi-local, and a one-stencil closure can capture its leading part. It cannot capture the tail. How much of the correction the tail holds is unmeasured.

**Horn B: concede the remainder.**
- The genuinely source-dependent part is the within-element shape of S e_X beyond its element integral. H3 §9 item 2 already concedes that a source-independent closure cannot represent it.
- H3 moves most of the steepness into exact D1, D2 and b̄ (only k ≥ 3 is gridded), so this remainder is second order. Whether it dominates the closure's useful signal is exactly what F3 tests.

**Contract.**
- The FNTO lines are verified [V via review]. H3-C9 already says LCR is admissible as a learned K only by human decision, and H3's contract note asks for that decision before Stage 2.
- I concede that a pass with an F4 failure answers the research question with "all transport explicit". That is not evidence for a learned shared operator.

### A-A9. Privileged physics and scope (H3 major, H2 minor) — **Concede**

- Both exact factorisations need a known scalar collision kernel: g = 0, homogeneous, index-matched, with σ_t and ω_c.
- H3 declares this (§4 table, last row; C9). H2 does not list S, ω, g = 0, homogeneity or index matching among its privileged inputs, and it should. The baselines already use part of this physics (W+UV's B uses σ_t and index-matched first-exit geometry; FIA uses ω), so the increment is the known kernel S.
- Both candidates' learned parts carry no angular state. Neither extends to Burley with a path-history-dependent reference (WL51), g ≠ 0 or Fresnel without one; H2 §8 and H3 §8/§13 say so.
- A fixture pass supports only "learned scalar resolvent / closure on top of a known kernel". It does not support "learned K" for unknown media.

### A-A10. Output ∝ MC d_B gives an unlearnable contrast floor (H1; minor) — **Concede**

**Verified.**
- β comes from one generator, seed 57501, consumed sequentially (`r57_geometry.py:80-93`), so each cell gets its own draws.
- eval68 uses L = H·diag(1/d), with the same d for model and reference (`eval68.py:56`).

**The asymmetry [D].**
- H1's L-mapped prediction is m·Π, which is noise-free.
- The reference's L-mapped value is ref/d_B, which carries δd/d to first order.
- H2's and H3's errors are (model − ref)/d_B, so the noise enters only as a second-order factor, except where their guards are active.

**Consequences.**
- No identity-free model can absorb a per-cell noise pattern, so the reviewer's 0.7 % rms gives a con floor of about 0.04–0.07 at edge/rim sources. That includes TRAIN_NEW's edge sources, so it affects H1-C2's floor as well as transfer.
- In quadrature with a 0.10 floor it gives about 0.12.

**Possible fix.** Replace d_B by a deterministic d̂ = 1 − β̂ from φ_FC's QMC chords, with min(d̂, d_B) to keep the B + K^s ≤ 1 budget. This is non-central but not specified, and it adds its own small bias. H1-C1's "identity up to β/d noise" is conceded as non-negligible at edge/rim rows.

### A-A11. The receiver-frame snap is discontinuous (H1; minor) — **Concede, scoped**

The source-side π is continuous; Q⁶⁴ (argmax plus degenerate-projection fallback) is not. I concede that H1-C3's "continuous on dented geometry" covers only the source side.

**Scope [D]** (dent profile b = cos²(πr/2R), linear in z; `p53_geom.py:168-173`):
- **ψ = ±45° switch.** For top↔top pairs, a tilt α of the receiver plane rotates the projected source axis by at most about ½(1 − cos α).
  - pt0's maximum slope 5.75·π/45 ≈ 0.40 gives α ≈ 22° and about 2°.
  - te3 (α ≈ 32°) gives about 4°.
  - The 45° switch is not reached on any listed dent.
- **Degenerate fallback.** The real discontinuity is near-degenerate projections: side-face sources whose canonical t is vertical, against slightly tilted top receivers, and the converse.
  - Dents displace only r < R from their centre, and side faces are untouched by every listed pose (`POSES`, `r57_geometry.py:22-29`).
  - The case therefore needs a side-face source within 15 mm of a tilted top receiver. That cannot happen at pt0: (5, −5) mm with R = 22.5 mm is 22.5 mm from both nearer sides.
  - It can happen at te2: (−17.5, 12.5) mm with R = 22.5 mm reaches x = −40 mm, 10 mm from the side. There, near-rim slopes are small but non-zero, which is exactly the near-degenerate case.

**What follows.** The defect is real but confined to a thin set of side↔dent pairs on poses such as te2. A soft weighted receiver alignment would be a non-central fix; it is not specified.

### A-A12. Ratio-min representability and the guard (H2; minor) — **Concede, partially**

**Representability.**
- t_e = m_e·sigmoid(·) < m_e, so exact representation needs r*_e < m_e.
- The physical margin comes from absorption along the multiple-scatter chain: about 2 % for red [S]. H2's own lumping estimate is about 1 %, and the sign of Q·esc − esc near a face is not controlled.
- So "can represent the physical Γ exactly" (H2-C5) is conceded as "approximately; exact where the lumping error is below the absorption margin". This most likely fails at surface nodes in red.
- Raising t's cap to m_e(1 + κ) would be a non-central fix. The column guard already enforces exact passivity, so it would cost nothing in validity, but it is not specified.

**Reciprocity.**
- H2-C5 already restricts reciprocity to "wherever s = 1", so that part is acknowledged.
- I accept that H3's symmetric form s_X·s_Y would keep reciprocity everywhere and stay passive: Σ_Y s_X s_Y K̃ ≤ s_X Σ_Y K̃ ≤ 1 − β [D]. Adopting it would be a non-central guard change.
- As specified, the source-side guard breaks symmetry where it is active.

### A-A13. Positivity and cost depend on the learned spectral radius (H3; minor) — **Concede**

**Arithmetic [D].**
- From 1 − λ_max ≈ B²/3 with B = π/11.42 mm [S], ρ ≈ 0.999 × (1 − 0.0252) ≈ 0.974.
- The Neumann count is ln(10⁻⁵)/ln(0.974) ≈ 437. H3 §3.3 states 260, which I correct.
- Chebyshev needs about ½√κ · ln(2/ε) ≈ 3.1 × 12.2 ≈ 38 iterations, which matches H3.

**Concession.**
- If the closure pushes columns toward the 1 − 10⁻³ cap, κ approaches 10³, the degree approaches ~190 and the Neumann fallback approaches ~1.15 × 10⁴ iterations.
- The deployed p(L) is positive only to ε relative, so negatives are possible where the true K is ≤ 10⁻⁴ of the near entries (red, beyond ~35 mm).
- Exact positivity therefore costs about 11.5× the Chebyshev apply, not 7×, and the cost depends on θ.

**Non-central fix.** Tie the column cap to the physical non-escape of the element plus a margin, rather than to 1 − 10⁻³.

### A-A14. DEF claims and the s51 slip (H3, H2; minor) — **Concede DEF and the slip; rebut the drawn consequence**

**DEF: concede.**
- DEF scores ΔK. map_dent thins the boundary layers and shears the cells, so the discretisation bias differs between poses and does not cancel in ΔK.
- H3-C6 ("DEF passes at θ = 0 and with the closure, including te3") is an assertion, not a construction. Every historical model fails DEF.
- H2 reports DEF descriptively (H2-C9), which is the appropriate status.

**Slip: concede.** s51 is port 3823 on face 4, z = −5, a **bottom** corner [V]. H3 §7(b) is wrong to call it a top corner.

**Rebuttal of the consequence ("at pt0 top corners are unsupervised").**
- s51's orbit (size 8) includes the z-flip images, i.e. the top corners. H3's closure features are D4h-invariant scalars, so the top-corner closure class is the same input as the bottom corner at rest.
- At pt0 the dent (centre (5, −5) mm, R = 22.5 mm) is ≥ 36 mm from every corner, so every corner element's local geometry is identical to rest.
- Top-corner closure classes are therefore supervised through s51 at both poses. The genuinely unsupervised classes are dent-modified corners (none in the listed poses) and vertical side–side edges away from s51's footprint.

### A-A15. What WL76 Stage 1 validated (H2, H3; minor) — **Concede**

**Verified** (`76_*:75-86`): 36/36 decomposed-vs-reference passes, in energy and coarse space (coarse relative L1 0.13–0.23 %, z² 0.90–1.13). The remaining transport was still MC, and the one failure was the single-scatter quadrature check (rest_s70, z² 1.703, post hoc amendment).

**Concession.**
- Stage 1 did not test the full reciprocal Gram identity with P_Y in per-triangle bin frames on dented ports, or native per-entry accuracy.
- H2 §8 and H3 §8 should say "the source-side first-collision decomposition was validated in coarse space; the full identity is mathematically exact (re-derived by Reviewer A) and its implementation is validated only by Stage 0".
- The packets do span INT, TBE, TBC, SM, SR and SV at rest plus pt0 and te0 [V], so the source-side coverage is broad.

**Accepted change.** Stage 0 must check native-resolution D1 and SS against the walker's single-scatter tally on rest *and* dented ports, including at least one SV and one corner packet. H2's 0a currently covers only 4 cells and needs this extension.

---

## 2. Responses to Review B

### B-B1. The translate argument does not discriminate; W+UV-Q control (H1, H2, H3; major) — **Concede**

**Agreed.**
- The A75 failure at the translates is large: per source 0.40–1.30 against TRAIN_NEW 0.131 [S, the reviewer's `per_matrix.csv` reading].
- A75 carries exactly the channels that break translation equivalence: xyz0, xyz, displacement, face id, a 124-entry global dent grid, world-axis pair offsets (`r57_geometry.py:58-65`) [V], a global U and a free mass (`r57_model.py:61-63`) [S].
- The most economical explanation is "identity / world-axis channels plus no point-group quotient". A translate pass separates neither the candidates from each other nor any of them from a de-identified W+UV.

**Accepted.**
- The **W+UV-Q** control: A75 with identity, absolute, displacement and global-grid channels removed, canonical-frame pair features, exact D4 permutations on the A/O indices, and a localised U. It isolates the remaining distinctive factors: H1's pair-specific angular blocks versus CP pairing, H2's exact ends and H3's solve.
- Separate falsifiers for orbit-equivalent and non-orbit contexts (see A1).

**One clarification.** W+UV-Q is itself a non-trivial new arm: it applies H1's quotient to the CP form. Its result is informative in its own right, because it is the direct test of the WL74A attribution (frames plus identity versus CP pairing).

### B-B2. H1's state and apply are understated (H1; major) — **Concede state and 1-light; rebut 64-light**

**Concession.**
- K^s = d Σ_g π_g m(g) Π(g), and each g has its own α, β and Z. A pair block is a sum of up to |supp π| rank-4 blocks with separate normalisers. §10 counts one frame for the state but about 5 for the rebuild, which is inconsistent.
- Mean |supp π| ≈ 4.8 is right: about 1,850 interior top/bottom ports carry uniform weight over 8 elements [E, reviewer].
- Corrected state: about 1.5 GB as dense near blocks (8.8e4 × 3,072 floats) plus a ×4.8 far zone, or about 2.1 GB as factors. H1-C7's 0.43 GB is wrong.
- A 1-light apply reads ≥ 1.5 GB, which is ≥ 1.6 ms at the 912 GB/s vendor peak. In the A75 torch style it is likely over 5 ms.

**Mitigation that keeps the mechanism.**
- At rest, the uniform-8 interior ports are the translation-invariant flat interior. Their collapsed blocks are identical per lattice-offset class and can be cached.
- On pt0, the dent changes windowed context for top ports within about R + 27 mm and for bottom ports beneath the dent. That is plausibly about half of the ports, giving about 0.6–1.0 GB of explicit dense blocks [E], under the cap but not under the 0.5 GB preference.

**Rebuttal on the 64-light apply.** The ×5 scaling assumes an index_add-style factor apply. With collapsed dense near blocks, the apply is a batched GEMM.
- Near zone: 8.8e4 pairs × 3,072 × 64 lights ≈ 1.7e10 MAC. Far zone: about 7e9 MAC. Total ≈ 1.4 ms at peak compute.
- The binding cost is the scatter of per-pair outputs (~1.1e9 floats). That plausibly puts a torch `bmm` + `index_add` at about 0.03–0.1 s [E], inside the 0.26 s preference.

**Required.** Restate C7 with frame multiplicity. Stage A should measure state, 1-light and 64-light on pt0, not on rest. The 1-light cap remains at risk.

### B-B3. H1's control cannot isolate its mechanism; its frame weights are prior art (H1; major) — **Concede**

**Prior art.**
- Continuous, input-dependent convex frame weights are Dym, Lawrence and Siegel (ICML 2024, arXiv 2402.16077, weighted robust frames). Frame averaging is Puny et al. (ICLR 2022).
- H1 §11 already cited Puny and Kaba as unverified precedent, and §12 called the mechanism "mostly a combination". Item (1) should now be withdrawn as a mechanism-level novelty.
- With |D4| = 8, the full Reynolds average is tractable. The weighted frames add canonicalisation (lower learning burden) and about 1.7× less evaluation, not exactness.

**Identifiability.**
- H1-noQ differs from H1 in three ways at once: exact equivariance, canonicalisation and an up-to-8-fold ensemble. F-c cannot attribute the result.
- Attribution claim (3) also bundles identity removal and CP-pairing removal, and neither is ablated.

**Accepted.**
- Add **H1-Reynolds** (uniform π over 8 everywhere) and **W+UV-Q**.
- Read the arm pairs as follows: H1 vs H1-Reynolds (canonicalisation), H1-Reynolds vs H1-noQ (equivariance plus ensembling), H1 vs W+UV-Q (pair blocks vs CP).
- Use intervals on the differences (B11).

### B-B4. Credit assignment to exact physics (H2, H3; major) — **Concede**

**H3.** θ = 0 is already the primary learning-free control (Stage 1). I accept two additions: F4 needs an explicit "inconclusive" outcome when the 𝔊 → 𝔊_ref gap is below about 2× its jackknife/seed interval, and the admissibility decision must be recorded before Stage 2, as H3 already proposes.

**H2.**
- I concede that H2 has no deployable learning-free Γ comparator. Γ_phys is a privileged per-geometry MC diagnostic that costs minutes per geometry.
- A deployable comparator exists: the Galerkin Schur complement of H3's θ = 0 CPM operator projected onto H2's nodes, or a face-image Green's function of the dipole class. It should be the primary comparator for F-T and F-G.
- If the deterministic Γ transfers as well, H2's learned value reduces to rebuild amortisation, about 0.05 s against minutes. That is a cost claim and a contract question (FNTO lines 17 and 86), which H2 §13(C) already raised.

**Information fairness.**
- The research question is "what must be represented explicitly". Supplying exact physics is the experimental variable, not an unfair advantage in itself.
- I concede, however, that "beats A75" cannot be evidence that the *learned* component transfers. Only "learned beats the learning-free control on unseen groups by more than a paired interval" can be.

### B-B5. H3's 1-light apply is memory-bound (H3; major) — **Concede the claim and F5 as written; rebut "only at rest"**

**Concession.**
- With explicit coefficients, each Chebyshev iteration reads about 3–3.7e7 floats (0.12–0.15 GB). Over 38 iterations that is about 4.6–5.7 GB, or 5–6.3 ms at peak. "Compute-bound" in H3-C7 is wrong for that layout.
- F5 prescribes "A75 torch style". Torch sparse matvecs over 2.5e7 nnz × 38 would take tens of ms, so F5 as written would fail H3.
- The coupled-solve multiplier and the thick-medium κ growth (about 5×) are also conceded (see B12).

**Rebuttal [E]** on "only a declared stencil kernel could meet the cap, and only at rest".
- H3 §10 specifies a structured stencil on a fixed tensor topology, with no index storage and translation-unique coefficients at rest. With coefficients generated per class (z-layer pair × band position × offset, about 10⁵ values, which fit in the 6 MB L2), per-iteration traffic is only the vectors: 103,544 × 3 channels × ~3 accesses × 4 B ≈ 3.7 MB. Over 38 iterations that is about 0.14 GB, about 0.15 ms. Compute is 2.9e9 MAC, about 0.17 ms at peak.
- On pt0, only the dent footprint needs explicit coefficients: about 2.5e4 elements × ~200 near couplings plus the closure ≈ 7e6 floats ≈ 28 MB per iteration. Over 38 iterations that is about 1.1 GB, about 1.2 ms at peak.
- A custom kernel at 20–40 % efficiency could therefore stay under 5 ms on pt0 as well.
- The systems budget itself asks for roofline counts "so that a representation is never rejected for an implementation artefact" (`specialist_systems.md:366`) [V].

**Required.** Restate F5 with a bytes-per-iteration roofline and a declared custom-kernel arm, measured on pt0. Accept that a torch-style measurement over the cap is reported as such.

### B-B6. Nonrigid geometry rebuilds under-counted; no locality certificates (H3, H2; major) — **Concede, with a partial rebuttal for H2**

**H3: concede.**
- About 20 cubature points per weakly singular 6-D self or adjacent pair integral is inconsistent with H3's own Stage 0(i) rule (≤ 1e-3 change on doubling). 10²–10³ points are realistic.
- Visibility is needed per cubature point for pairs straddling the dent surface, not one ray per pair. 0.3–2.5 s is optimistic, and 1–10 s is plausible at about 88 M rays/s [S].

**H2: partial rebuttal.**
- (i) The 6.3e7-cast figure is a full-pose build. After a dent edit, only the dent footprint plus the Π reach is recast: about 7e6 casts by the reviewer's own table, within the 32 M cap. I concede that the initial full-pose build is over the extra-quadrature cap, if that cap applies to full-pose builds as B's 15.7 M implies.
- (ii) The two-hop normaliser radius (~57 mm) affects only the learned coefficients h, t and R. Their *full* rebuild is 0.02–0.1 s (H2 §10), within the preferred 0.1 s, so the wide radius has no cost consequence. The expensive deterministic terms (Π, SS) have the local ~5 mm reach.
- (iii) Concede that dent SS at 1–3 s exceeds the 1 s cap, which H2 had already declared.

**Both.** Concede that the locality certificates are unspecified. A natural one exists: per ray or segment, the set of triangles tested or exited, so invalidation = the rays touching changed triangles. It must be written down, measured on pt0 and te1, and charged against the 1 s cap with B separate.

### B-B7. H2's novelty omits the closest learned prior art (H2; major, framing) — **Concede**

- Neural Green's Functions (NeurIPS 2025, arXiv 2511.01924) already learns a geometry-only, source-agnostic operator decomposition that generalises to unseen sources and geometries, for elliptic PDEs.
- Element Learning, DeepRTE, Wang–Tran–Luebke 2005 and Garg et al. 2006 (reciprocity plus hierarchical low rank to infer unmeasured columns) cover the rest.
- H2 §12 already labels mechanism novelty "weak" and claims only the combination. "Source-independent geometry-conditioned resolvent" should be removed as a mechanism-level distinction.
- What remains: the RTE boundary-to-boundary instance with exact reciprocal angular ends, port/bin correspondence, per-cell passivity, the ratio-min normaliser (a small device near symmetric matrix scaling), and the held-out sparse-source evaluation.

### B-B8. H3's learned-closure precedent is broader (H3; minor) — **Concede**

Add ML moment closures for the RTE (Huang, Cheng, Christlieb, Roberts, JCP 2022) and differentiable-programming closure learning in radiation transport (NeurIPS 2023 workshop). Only the CPM-generator target and the symmetric, M-matrix-preserving parameterisation are specific. H3 §12 already concedes weak mechanism novelty.

### B-B9. Bit-identical image claims depend on symmetric quadratures (H1, H2, H3; minor) — **Concede**

**H1.** A 16 × 16 Sobol pattern is not D4-symmetric, so image features differ by quadrature error at the ~1 % precision H1 needs.
- A D4-symmetric set of the same size (32 base chords × 8 images = 256) keeps the cast count at 15.7 M. It fixes the problem without multiplying casts. This is a non-central change, not specified.

**All candidates.**
- B/β come from one sequential generator (`r57_geometry.py:80-93`) [V], so d differs between images.
- H2's guard s_c = min(1, (1 − β_B)/E^raw) and H3's s_Z = min(1, (1 − β_Z)/colsum) both depend on β, so image identity and reciprocity break where the guards are active.
- H2's and H3's deterministic midpoint/Gauss rules are D4-symmetric if no jitter is used [E].

**Accepted.** Stage 0/A unit tests report feature and K-row discrepancies for image pairs, separated into a quadrature-induced part and a d-induced part, together with guard activity at image pairs.

### B-B10. Sub-cell confound in ratio falsifiers (H1, H2, H3; minor) — **Concede**

- Port-level equivariance equalises predictions, not con_coarse. The coarse contrast signal depends on the sub-cell: |HZ|/|Z| ≈ 0.054 at a centre sub-cell vs 0.14–0.17 at edge/corner [S], and G2 is 0.09/0.09/0.14 by sub-cell (WL75) [S].
- H1 §7(b) argues port-level transfer to any sub-cell; that argument concerns the operator, not the metric.

**Accepted.** Predeclare ratios within hi and within lo sub-cell strata, or ratios to each source's own floor (G2, K_phys or θ = 0). This applies to H1 F-b, H3 F2 and H2 F-T.

### B-B11. Two-seed spread is not an uncertainty (H1, H2, H3; minor) — **Concede**

- A two-seed range is one difference, with no coverage guarantee. WL37 → WL38 (−0.313 dB on one seed, −0.082 dB mean over three) is the project's own precedent.
- **Accepted.** Use ≥ 3 seeds for the decisive mechanism-vs-control comparisons (H1 vs H1-noQ / H1-Reynolds / W+UV-Q; closure vs θ = 0; learned Γ vs a deterministic Γ). Alternatively, use a predeclared paired replica-jackknife interval on the difference, combined with the seed range.

### B-B12. Scalability beyond the fixture (H1, H2, H3; minor) — **Concede**

- **H1.** The near-core state scales as h⁻⁴ (H1 §10 already says about 5.5 GB at 1.25 mm), times the B2 frame factor.
- **H2.** Γ scales as ρ³. The reason is physical: H2's 18.3 mm is the red infinite-medium diffusion length (A2). In a slab thick enough not to truncate it, ρ_red grows toward about 5 × 18 mm, roughly 47× the state.
- **H3.** The state grows with volume, and the iteration count grows as √κ, up to about 5× for thick red.
- **H2 and H3** need a known scalar kernel (A9).

None of this affects the fixture test. It bounds what a pass can claim, and all three §12 statements should carry the bound.

---

## 3. Residual position by candidate (after concessions)

**H1 (CQR).** The invariants are intact.
- Conceded: the translate test is a unit test only; TBC is out of orbit; frame weights are prior art; the state is about 1.5 GB, not 0.43 GB; the d_B floor is about 0.04–0.07 at edge/rim sources; the receiver snap is discontinuous on te2-type pairs; and source-side learnability out of orbit is the binding risk.
- The remaining distinctive test: pair-specific angular blocks under a quotient, against W+UV-Q and H1-Reynolds, on non-orbit contexts with ≥ 3 seeds.

**H2 (RBG).** The invariants are intact, and reciprocity holds off the guard (a symmetric guard would be a non-central fix).
- Conceded: the 30 mm plane features are location keys as written; the L1 floor with first-collision projection is a likely fail; the SS voxel cubature is aliased; there is no deployable learning-free Γ comparator; the novelty is a combination; ratio-min is only approximately exact; and the full-pose Π and dent SS exceed the caps.
- Rebutted: the normaliser invalidation radius has no cost consequence, and the dent-footprint Π fits the cast cap.

**H3 (LCR).** The invariants are intact (positivity to ε), and the transfer argument is not subject to A7.
- Conceded: the "third-nearest plane" feature is a location key; c_max bounds the parameter, not the error; DEF is not by construction; s51 is a bottom corner; Horn A and the contract question stand; the cubature and visibility counts are too low; and the torch-style F5 fails.
- Rebutted: the exact coarse correction is source-independent and plausibly quasi-local, so Horn B does not hold as an impossibility; the stencil-kernel roofline fits the 1-light cap at rest and plausibly on pt0; and top corners are covered by z-flip and are unchanged at pt0.

## 4. Not verified

- The bin-resolved (16 × 64) identity of the translates. Only Reviewer A's angle-integrated check exists.
- The off-diagonal decay of L* (A8) and the share of the closure correction in its tail. This is a derivation, not computed.
- The per-receiver far-field difference near the second plane (A2/H1).
- Every cost figure (B2, B5, B6). These are arithmetic from the candidates' counts and vendor peaks; nothing was timed.
- The sign and size of r*_e − m_e (A12) and of the D2 depth truncation (A5). These are estimates.
- Whether the Gate 4 far TBE/SM/SR slots are along-edge translates. The population was not opened.
