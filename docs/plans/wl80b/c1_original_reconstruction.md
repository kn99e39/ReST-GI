# C1 original reconstruction (Part A)

> WL80B record (workflow `wf_9bbd762a-ae2`, 12 agent runs). Agent texts reproduced verbatim; headings added by the Coordinator. Design-only: nothing implemented, executed, trained or measured. The original C1 architect was not resumable; a stand-in reconstructed C1 from the sealed WL80A records (immutable inputs, sha256 in `inputs.json`).

# WL80B Part A: reconstruction of the original C1 (CP-K) from the sealed WL80A records

**Basis.** This is read-only. I read these files in full or in the C1 sections:
- `docs/plans/wl80a/trackB_candidate_records.md` (C1, l.131–285, and the synthesis);
- `trackB_adversarial_review.md` (C1, l.5–479);
- `trackB_selected_candidate.md`;
- `cross_review.md`;
- `final_architecture_decision.md`;
- `coordinator_disposition.md`;
- `lab_meeting_summary.md`;
- `neutral_problem_brief.md`.

I spot-checked these anchors:
- `docs/worklogs/76_*` l.252–255 and l.268;
- `docs/plans/wl77a/specialist_physics.md` l.299–302;
- `wl77a/coordinator_decision.md` l.15;
- `wl77a/candidate_H3.md` l.78–79 and l.191;
- the FNTO contract l.6, 17, 26 and 86;
- two code files, for bin conventions only.

No file was written.

**Labels.**

| Tag | Meaning |
|---|---|
| [EP] | Established physics |
| [PA] | Prior art |
| [HM] | Historical measurement |
| [WD] | WL80A design claim |
| [ND] | New derivation (mine) |
| [UH] | Untested hypothesis |

**Short names for the source files used in citations.**

| Short name | File |
|---|---|
| CR | `trackB_candidate_records.md` |
| AR | `trackB_adversarial_review.md` |
| SC | `trackB_selected_candidate.md` |
| XR | `cross_review.md` |
| FD | `final_architecture_decision.md` |
| CD | `coordinator_disposition.md` |

## A0. Provenance tiers

| Tier | What it is | Where |
|---|---|---|
| (i) Original sealed C1 | Track B candidate architect's specification | CR l.131–285; optional sub-variants merged at CR l.85–96 |
| (ii) Developer-amended C1 | The developer's response to Review 1 (physics) and Review 2 (learnability/computation) | AR l.246–479. The objections it answers are at AR l.24–117 and l.138–238 |
| (iii-a) Track B synthesis chair (sealed) | Adopts (ii) and adds items: the clamp must preserve the complement and be reported; the dent influence set must include bottom, side and rim sources (D_T ≠ D_G); a paired contrast control; an optional CCA-rest control | SC l.75–182 |
| (iii-b) Cross-review chair (post-sealing) | Six mandatory amendments, a prototype and contracts | FD l.193–199 and l.222–360; XR l.56–66, l.95–104, l.144 |
| (iii-c) Coordinator disposition | Supersedes the cross-review chair's wording | CD l.15–25 and l.31–35 |

## A1. Element-by-element reconstruction

### 1. Operator equation

**(i) Original.** Source: CR l.137–146 [WD].
- q = F_in u is the first-collision density on cells.
- ψ = (I − ω_c W)⁻¹ q, and the output is ω_c Φout ψ.
- Split form: `K_c = ω_c S1 + ω_c² Φout W (I − ω_c W)⁻¹ F_in`, with `S1 = Φout∘F_in`.
- S1 is "evaluated at fine point quadrature, not cell-averaged" (a first-collision-source method). Cells carry only the multiply-scattered remainder.

**(ii) Amended.** Source: AR l.290; restated at SC l.89–92 [WD].
```
q   = F_in u              (first-collision density at fine points p; Σ_p q = d(i,a))
ψ   = ω W1 q + ω W ψ      (cell collision density from the 2nd collision on)
K u = ω Φ_pt q + ω Φout ψ
```
- Cell escape is defined as the complement: 1ᵀΦout[y] := 1 − 1ᵀW[y] (AR l.290, l.301).

**(iii) Chair and coordinator.**
- No change to the equation.
- Coordinator: the bound ΣK ≤ ω_c·d holds at convergence only if W ≥ 0 entrywise and 1ᵀW ≤ 1 (CD l.20) [WD].

**Check of the amended algebra [ND].** I re-derived it:
- Let e = 1 − 1ᵀW.
- From the ψ equation: 1ᵀψ = ω1ᵀW1q + ω(1ᵀψ − eᵀψ).
- With 1ᵀΦ_pt = 1 − 1ᵀW1 per point, the total output is ω1ᵀq − (1−ω)1ᵀψ.
- That equals ω·d − (1−ω)1ᵀψ ≤ ω·d. This agrees with AR l.291, SC l.98 and FD l.503.

### 2. State variables

**(i) Original.** CR l.137–140 and l.149–153 [WD].
- Per channel: the cell collision density ψ, which *includes* first collisions.
- Geometric factors: F_in, W, Φout.

**(ii) Amended.** SC l.90–92 [WD].
- Point first-collision density q(p), which is geometric and channel-free.
- Cell density ψ_c, which *excludes* the first collision.
- Point factors Φ_pt and W1; cell factors W and Φout.
- The ballistic operator B and d are unchanged (SC l.101).

**(iii).** None.

**Notes.**
- The meaning of ψ changes between (i) and (ii). The critic confirms that in (ii) "ψ counts collisions after the first" (FD l.502).
- Per lighting state the solver state is about 3.6 MB (XR l.924, ESTIMATE) [WD].

### 3. Tensor and discretisation dimensions

**Interface** [WD; brief l.43–50].
- Inputs: 3840 × 16 × 3 = 184,320. Outputs: 3840 × 64 × 3 = 737,280 (CR l.214).
- That is 61,440 beams (port × incoming bin) and 245,760 reverse beams (port × outgoing bin) (CR l.216–217).

**(i) Original options.** CR l.174–176 and l.215–227 [WD].

| Option | Cells | W | Other factors | Memory and timing (ESTIMATES) |
|---|---|---|---|---|
| (U) uniform h = 0.5 mm, D4h-symmetric | 200 × 200 × 20 = 8.0e5 | Masked FFT on a 400 × 400 × 40 zero-padded grid (6.4e6 points); analytic self-cell and near field | F_in ≈ 2.5e7 nnz (≈ 0.2 GB); Φout ≈ 1e8 nnz (0.6–1 GB, about 400 cells per reverse beam) | W application 0.3–1 ms per channel; total 1–3 GB |
| (G) graded | 3e4–1.1e5 | Sparse, about 5e7 nnz (0.4 GB) | — | 10–30 ms per application |

Grading for (G): lateral 1.25 mm, 0.25 mm near the side faces, depth layers 0.25 → 1.5 mm. A linear-in-cell source is optional (CR l.177).

**(ii) Amended.** AR l.338–353; SC l.107–112 [WD].
- Option (U) is dropped. The mesh is graded and body-fitted, about 2–3e5 cells.
- Top and bottom faces: lateral 1.25 mm; depth layers 0.05, 0.05, 0.1, 0.1, 0.2, 0.2, 0.4 mm, continuing to the mid-plane. About 1.5e5 cells.
- Side faces: about 3e4 cells.
- Edges: product grading in the cross-section, 1.25 mm along the edge, translation-shared. About 5e4 cells.
- W near field: explicit within about 3 mfp, about 300 partners per cell, about 1e8 nnz, about 0.7 GB.
- W far field: precorrected FFT.
- Reverse beams use full chords, not a 5 mfp truncation (AR l.438).
- The sizes of the fine-point sets for Φ_pt and W1 are not specified (see ambiguity 6).

**(iii) Chair.**
- Lateral spacing becomes a separately declared axis: L0 = 1.25 mm, L1 = 0.625 mm, L2 if affordable (FD l.250).
- The far field must preserve positivity: non-negative interpolation, or a matrix-free explicit ray-telescoped W (FD l.195, l.256).
- The computation reviewer adds that a W stored explicitly to 10 mfp needs about 1.6e9 nnz, about 13 GB, so a far-field scheme is required (XR l.957) [WD].

### 4. Boundary and volume ownership

**(i) Original.** CR l.151 and l.199 [WD].
- Only in-medium chords count in W.
- A chord through the dent cavity is an exit; re-entry belongs to C.
- The first exit terminates the path.

**(ii) Amended.** AR l.289; SC l.95 [WD].
- The same rule applies to Φ_pt and W1.

**(iii).**
- The physics reviewer agrees (XR l.355).
- The ballistic part is owned by B. Collided budget d = 1 − β (brief l.58).

### 5. First-flight treatment

**(i) Original.** CR l.150 [WD].
- `F_in[x; i,a] = ∫_port ∫_bin (contract weighting) ∫ σ_t e^{−σ_t s} 1[r(s)∈x] ds`.
- F_in uses the same rays as B, so Σ_x F_in = d(i,a) to float precision.
- Deposits go into cells.

**(ii) Amended.** SC l.90 and l.94 [WD].
- The same ray integral, but deposited at fine first-collision points p, with Σ_p q = d.
- The first continuation is point-exact. This was the fix for R1-1 (AR l.286–294).

**(iii) Chair.**
- Σ F_in = d against B to fp precision is a hard test (FD l.268).
- The step-0 gate must read the radiance-weighting convention and the "1.8–4 mm" source definition from code (FD l.198).

### 6. Collision transition representation

**(i) Original.** CR l.151 and l.153 [WD].
- W[y,x] is the probability that an isotropic emission, uniform in cell x, has its next collision in cell y. This is a flat-source collision-probability (CP) kernel.
- The record claims a per-cell partition 1ᵀW + 1ᵀΦout = 1ᵀ "by construction".
- An optional learned ΔW closure is part of (i) (CR l.162).

**(ii) Amended.** AR l.288 and l.300–302; SC l.95–96 [WD].
- W1[·; p]: point-to-cell next-collision deposit from one shared, stratified, isotropic ray set per point. 1ᵀΦ_pt[p] + 1ᵀW1[p] = 1 exactly by telescoping.
- W: cell-to-cell, "built by ray telescoping so that 1ᵀW[y] ≤ 1".
- ΔW is dropped (AR l.331).
- Exact volume-weighted reciprocity is given up in favour of an exact partition, and the reciprocity defect is reported (AR l.304).

**(iii) Chair.**
- Per-point and per-cell partitions must hold to fp precision (FD l.269).
- Reciprocity is reported on aggregated 16 × 16 blocks, and only if the bin groups nest (FD l.272).
- Coarse corrections are accepted only if they are non-negative with a sub-stochastic coarse operator (FD l.259).

### 7. Multiple-scattering accounting

**(i) Original.** CR l.144–146 and l.219 [WD].
- Order 1: point-exact (ωS1).
- Order 2 onward: the cell Neumann/resolvent series ω²ΦoutW(I−ωW)⁻¹F_in.
- Solver options: Neumann (about 260 iterations for red to 1e-3), DSA or two-grid (10–20, unverified), or CG. Collision densities can optionally be joint unknowns of the outer K∘C solve (CR l.86–89).

**(ii) Amended.** AR l.416–417; SC l.115 and l.155 [WD].
- Point-exact: the order-1 escape (Φ_pt) and the order-2 collision distribution (W1).
- Cell-averaged: the order-2 escape and all higher orders (W and Φout).
- Solver: converged to 1e-6 relative, or a fixed linear iteration. Truncated Krylov methods are withdrawn as "exactly linear" (SC l.57).
- Making the linear-in-cell source mandatory was declined, because it can go negative (AR l.418).

**(iii) Coordinator.**
- Plain iteration needs about 524 sweeps to reach 1e-6 at red ρ = 0.974. The cost estimates silently assume acceleration (CD l.21) [WD].

### 8. Exit transport

**(i) Original.** CR l.152 [WD].
- Φout[j,o; y] is the probability of uncollided escape from cell y through port j into bin o, tallied with "the reference's radiance-weighted power convention".
- Reverse beams are truncated at about 5 mfp (CR l.217).

**(ii) Amended.** SC l.95 and l.97 [WD].
- Φ_pt: point-exact escape for order 1.
- Φout: per-cell total is *defined* as 1 − 1ᵀW[y]; quadrature supplies only the shape over (j, o).
- Full chords (AR l.438).
- If the far field pushes 1ᵀW above 1, the escape is clamped to 0 and the result renormalised (AR l.351).

**(iii) Chairs.**
- Track B chair: any such adjustment must preserve the complement and be reported (SC l.112).
- Cross-review chair: the clamp must be predeclared and its binding fraction reported (FD l.195, l.258; XR l.66).

### 9. Absorption accounting

**(i) Original.**
- Absorption is implicit: a factor (1−ω) at each collision.
- The record's identity is output = d − (1−ω)1ᵀψ ≤ d (CR l.197) [WD].
- Since ψ ≥ q, (1−ω)1ᵀψ ≥ (1−ω)d, so the original identity already implies output ≤ ω·d. The original's "≤ d" was loose [ND].

**(ii) Amended.**
- Output = ω·d − (1−ω)1ᵀψ (AR l.291) [WD].
- Absorbed total = (1−ω)(1ᵀq + 1ᵀψ), i.e. (1−ω) per collision of every order [ND].

**(iii) Chair and coordinator.**
- The passivity falsifier tests ω_c·d·(1+1e-6), not d (FD l.194, l.267).
- The bound is conditional on W ≥ 0 and 1ᵀW ≤ 1 (CD l.20) [WD].

### 10. Geometry-dependent state

**(i) Original.** CR l.156–158 and l.169–170 [WD].
- The cell mask, port frames and in-medium chord visibility; F_in, W and Φout; ω_c and σ_t; B and d.
- At the dent, a sparse chord correction within about 6 mm.

**(ii) Amended.** SC l.101 and l.104 [WD].
- The body-fitted mesh, port frames and in-medium visibility; F_in, W1, Φ_pt, W and Φout; ω_c, σ_t, B and d.
- The 6 mm chord band is dropped, because ray integration handles chords (AR l.333).
- All geometric factors are channel-free. ω_c enters only in the solve, so albedo edits are free (XR l.932) [WD].

### 11. Illumination application

**(i) Original.** CR l.166 and l.221 [WD].
- The source enters only as the right-hand side F_in[:, i, a] u.
- No learned function sees the source.
- Inside GI, collision densities can be joint unknowns, at about one W application per outer iteration.

**(ii) Amended.**
- The one-W-per-outer-iteration claim is withdrawn: it inherits ρ ≈ 0.974.
- Revised plan: 10–30 outer iterations, each with a warm-started, diffusion-accelerated inner solve of 10–40 W applications; about 1–10 s per GI solve (AR l.405–407; SC l.55) [WD].
- B is applied separately [WD].

**(iii) Computation reviewer.**
- On the isolated convex slab there is no outer loop, so a GI solve is one K application. Batching 64 lights narrows the cost gap (XR l.936–939, l.965–970) [WD].

### 12. Training requirements

**(i) Original.** CR l.160–163 and l.208–211 [WD].
- None for the operator.
- Optional ΔW: about 1e3 parameters; E(3)-invariant inputs (distances to up to three planes, dihedral type, dent curvature); a symmetric, non-negative redistribution with fixed row sums; trained on deterministic canonical-patch solves.
- Optional learned preconditioner.
- MC references, replicas 4–7, are used for validation only.

**(ii) Amended.**
- ΔW is dropped (AR l.331).
- The preconditioner stays optional and affects speed only (SC l.102) [WD].

**(iii) Chair.**
- No training (FD l.290).
- Every discretisation, quadrature, solver and clamp setting is frozen by reference-free criteria before any MC score is read (FD l.196).
- Replicas 0–3 are also admissible but reported separately (FD l.295) [WD].

### 13. State construction and rebuild

**(i) Original.** CR l.180–187 and l.227 [WD].
- Rigid motion: nothing to rebuild.
- Dent:
  - re-mask the cells;
  - recompute the F_in and Φout rows for about 450 ports within about 5 mfp;
  - recompute the chord correction;
  - the FFT spectrum is unchanged.
- Estimated time: 0.05–1 s.

**(ii) Amended.** AR l.328–330, l.408–409 and l.439; SC l.111–114, l.120 and l.128–129 [WD].
- Rest: per-face translation invariance turns F_in, Φout and the point templates into convolutions. Only boundary contexts (a few hundred × 16 bins) need distinct templates.
- Exact D4h holds if the operator is assembled from orbit representatives by permutation; otherwise to about 1e-7.
- Dent: the same topology under a vertical stretch map, so there are no cut cells. F_in, W1, W and Φout are ray-integrated on the true mesh, with no local FFT.
- The influence set includes bottom, side and rim sources.
- Then a global re-solve.
- Estimated rebuild time: 1–5 s computed on the fly; minutes if tabulated.

**(iii) Chair.**
- The 1–5 s figure needs an unbuilt deterministic quadrature (FD l.138) [WD]. The sampled analogue took 75.0 s, of which 74.3 s was quadrature [HM, `76_WORKLOG_FIRST_INTERACTION_ANCHORED_TRANSPORT.md` l.252].
- Added tests:
  - a rest-mesh versus stretched-mesh null test;
  - a full-rebuild control;
  - an independent stochastic spot-check of Φ_pt and W1 at q ≥ 65,536 (FD l.286, l.315, l.352).

## A2. Chair and coordinator amendments in one list

**Track B synthesis chair (sealed).**
- The clamp must preserve the complement definition and be reported (SC l.112).
- The dent influence set includes bottom, side and rim sources (SC l.120).
- Paired exact versus bin-averaged F_in contrast control (SC l.180).
- Optional CCA-rest control (SC l.182).
- Cost does not falsify the physics; it decides deployability (SC l.181).

**Cross-review chair** (FD l.194–199).
1. The passivity falsifier tests ω_c·d.
2. The far field preserves positivity; any clamp is predeclared and its binding fraction reported.
3. Lateral and depth refinement are separate axes. A memory-limited level is "affordability-limited", not a falsification of the factorisation.
4. Settings are frozen by reference-free criteria.
5. The step-0 gate resolves the "1.8–4 mm" source definition and the radiance weighting from code.
6. Baseline terminology: the path-traced re-render is comparison class 3.

**Coordinator.**
- HIGH-1: a converging miss means affordability-limited; a plateau (order < 0.5) is a C1 fail (CD l.15).
- HIGH-2: a pass shows only that the MC-defined target K is computable from geometry on the scored strata. It does not attribute the learned-family failures (CD l.16).
- MEDIUM-3: C1 is a primary candidate *operator / diagnostic instrument*, not a learned-K answer. The WL77A H3 rejection grounds are not refuted (CD l.17).
- MEDIUM-4: C1 is "the WL77A H3 θ = 0 operator class with a revised discretisation" (CD l.18).
- MEDIUM-6: the passivity bound is conditional on W ≥ 0 and 1ᵀW ≤ 1 (CD l.20).
- MEDIUM-7: cost relabels; about 524 plain sweeps; FIA's measured 0.0039 s single-light anchor, against which C1's estimate is 18–51× (CD l.21).
- LOW-10: "applicable; deployability unmeasured" (CD l.24).
- LOW-11: lateral bias is 9–18 % in length, and "38 collisions" is an inference (CD l.25).

## A3. Ambiguities and inconsistencies in the records

1. **S1 notation versus its description.** "`S1 = Φout∘F_in`" (CR l.144), the cell-factor composition, against "S1 is evaluated at fine point quadrature, not cell-averaged" (CR l.146). The original never defined a point-level escape operator, so S1's actual discretisation is underspecified. This mismatch is what R1-1 exploited.

2. **Partition claimed by construction.** "Partition of unity per cell, by construction (each sampled emission …)" (CR l.153) conflicts with the specified build: an analytic/FFT W (CR l.175) and separate reverse beams for Φout (CR l.217). The developer conceded this (AR l.298).

3. **W index convention [ND].**
   - CR l.151 defines W[y,x] as the transition x → y.
   - Reciprocity is then written "V_x W[x,y] = V_y W[y,x]" (CR l.200), and the same form appears at AR l.53.
   - [EP] V_x P(x→y) = V_y P(y→x). Under CR's own convention that is V_x W[y,x] = V_y W[x,y]. The stated relation is transposed. The physics intent is the standard one.
   - The amended record writes "1ᵀW[y]" with y as the *emitting* cell (SC l.96–97). The index roles flip between (i) and (ii).

4. **"Ray-telescoped W" versus a pFFT far field.**
   - SC l.96 says W is ray-telescoped with 1ᵀW[y] ≤ 1. SC l.112 says the far field is a precorrected FFT, which is not ray-telescoped, so the guarantee does not carry over.
   - The chair offers "non-negative interpolation *or* matrix-free" (FD l.256) without choosing.
   - An explicit 10 mfp W does not fit in memory (XR l.957).
   - Unresolved (FD l.438).

5. **The clamp.** "Clamp the escape to 0 and renormalise" (AR l.351) does not say what is renormalised. Renormalising W to a row sum of 1 leaves that cell with zero escape. The chair records require predeclaration and reporting, but no policy is specified (FD l.258).

6. **Fine first-collision points.** Their placement and count are unspecified.
   - "about 1e3 points" per (port, bin) beam (AR l.408).
   - "per-beam first-collision templates" (AR l.294).
   - "about 1e6 fine points" for the whole body (XR l.945, computation reviewer).
   - It is unclear whether points are per beam or a shared global set. That decides the sizes of Φ_pt and W1 and the meaning of q.

7. **Quadrature rule.** The records give three versions:
   - "deterministic, stratified, orbit-symmetrised" (CR l.173);
   - "one shared stratified isotropic ray set per point" (AR l.288);
   - "Gauss quadrature of e^{−σ_t s} along chords" (FD l.253).

   [ND] In a homogeneous medium each chord-segment deposit is analytic (e^{−σs_a} − e^{−σs_b}). Quadrature is needed over directions and over cell or port areas, not along the chord. The "Gauss along chords" wording is ambiguous.

8. **Refinement axes.** "Three grading levels, each halving all graded sizes" (AR l.361; SC l.168) against the chair's separate lateral axis L0/L1/L2 (FD l.197, l.250).

9. **Passivity statement.**
   - Bound: d·(1+1e-6) (AR l.365; SC l.175) against ω_c·d (FD l.194, l.267).
   - Conditions: "for any h and any W accuracy" (SC l.98) against "only with W ≥ 0 and 1ᵀW ≤ 1" (CD l.20).

10. **Solver form.**
    - Joint unknowns were an option in (i) (CR l.88); only the cost claim attached to it was withdrawn (SC l.55). It is never stated whether joint unknowns remain a permitted solve form.
    - "Warm-started" inner solves (AR l.407) make the iterate state-dependent, so the operator is linear only to the solve tolerance (AR l.442). This is consistent with "converged 1e-6", but not with "exact".

11. **Sign of the flat-cell Φout bias.** The reviewer computes −3.3 % at μ = 0.4 (AR l.80); the developer computes +4.6 % (AR l.386–391). Unresolved (SC l.225).

12. **Lateral grading.** The amended mesh is coarser laterally (1.25 mm) than the withdrawn 0.5 mm lattice. The developer's "falls under the grading" covers depth only (XR l.444).
    - Existing non-evidence [HM, labelled non-evidence at source; `docs/plans/wl77a/specialist_physics.md` l.299–302]:
      - piecewise-constant 0.5 mm voxels: red energy +2.1 %;
      - 2.5 mm lateral snap: edge 0.20 (energy −11 %), side 0.36 (energy −23 %), pair contrast 0.05–0.08.
    - That table has no row for 1.25 mm.

13. **Dent influence set and rebuild cost.**
    - Influence set: about 450 top ports (CR l.184) → bottom, side and rim sources required (SC l.120) → about 1,000–1,600 ports for the tabulated variant (XR l.974).
    - Rebuild time: 0.05–1 s (CR l.227) → 1–5 s (AR l.408) → "needs an unbuilt deterministic quadrature" (FD l.138).

14. **Convention source.**
    - "Contract weighting" for F_in (CR l.150) and "radiance-weighted power" for Φout (CR l.152, SC l.97) are not defined in any WL80A record (FD l.458).
    - Φ_pt's weighting is not stated at all.
    - The code check covers bin geometry only: incoming bins are a 4 × 4 concentric square-to-disk partition of equal projected solid angle, in the per-triangle (t, b, n) frame (`experiments/fnto_rank66/ports66.py:47`); outgoing rings are of equal projected solid angle (`experiments/fnto_fia76/fia76.py:15`). [HM, as a code convention.]
    - The radiance-weighting tally itself is unresolved.

15. **Role and identity.**
    - "Selected primary" (SC l.5) → "qualified role, not a learned-K answer, not a deployable FNTO K" (FD l.189–191) → "primary candidate operator / diagnostic instrument" (CD l.17).
    - "C1 is WL77A H3 at θ = 0" (FD l.191) was refined to "the same class with a revised discretisation" (CD l.18).
    - H3 used 0.5 mm boundary bands with 1.25 mm interior cells and gridded only k ≥ 3 (`candidate_H3.md` l.78–79, l.191). Amended C1 grids k ≥ 2 (second collision onward) [HM / ND].
    - H3's rejection grounds (`wl77a/coordinator_decision.md` l.15) remain unrefuted.
    - The FNTO contract requires a *learned* K and warns against a "limited explicit radiosity solver" (FNTO contract l.6, 17, 26) [HM: contract text].

16. **Outcome semantics.**
    - The chair's §1(i), "failures concern learning" (FD l.240), is superseded by HIGH-2.
    - The chair's lab-summary inversion is superseded by HIGH-1.
    - "Falsified" (SC l.174; FD l.351) against "affordability-limited" (FD l.197): the reconciliation in CD l.15 is a fail of C1 at that discretisation, not of the factorisation.

## A4. Status of claims in the reconstruction

- **Measured: nothing about C1.** Every accuracy, cost and convergence statement about C1 is either a WL80A design claim (ESTIMATE) or an untested hypothesis. The anchors in A1 and A3 are [HM] records of other operators.
- **Established physics** (CR l.14–17, re-derived by five Track B specialists): under g = 0, a homogeneous medium and an index-matched boundary, the Peierls collision-probability factorisation is exact, and all incoming-bin dependence enters through F_in [EP].
- **Derived by me** [ND]:
  - the amended energy identity, re-derived (agreeing with SC l.98 and FD l.503);
  - the original identity already implied ≤ ω·d;
  - the transposed reciprocity index (ambiguity 3);
  - the analytic chord-deposit remark (ambiguity 7).
- **Central untested hypothesis** [UH]: that BND_FAR, corner, edge and rim strata, and the dent signed change, meet the gates at affordable graded resolution (SC l.156–157).
