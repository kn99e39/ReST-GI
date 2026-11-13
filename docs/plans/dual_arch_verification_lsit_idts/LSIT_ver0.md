# LSIT physics and contract audit (read-only, commit 4ed4ac3)

**What this audit rests on**
- **Definition used.** LSIT is taken only from `c:\Projects\ReST-GI\docs\plans\DESIGN_PROPOSAL_LOBE_SPLIT_INTRINSIC_TRANSPORT.md`, which I read in full. Citations below as `LSIT §x`.
- **Code re-checked:**
  - the ER closure: `experiments/fnto_compiled83/closure83_dr.py` (eval at lines 99–155, sample at 164–202), plus the headers of `fnto_compiled82/closure82.py` and `fnto_compiled80c/closure80c.py`;
  - stored material states: a numpy read of `experiments/ecro86/data/mats86.npz` (`states` [6,69]);
  - `experiments/ecro86/pt86.py` (bins at 153–156, table readout at 228–233, state rays at 355–376);
  - `experiments/ecro86/ecro86.py` (`dirty_rays`, 57–67).
- **Evidence re-checked:** WL92 component shares and ranks in `docs/worklogs/92_WORKLOG_ECRO_SHARP_REPRESENTABILITY_IDENTIFIABILITY_ATTRIBUTION.md` ("Physical attribution" and "Local operator" tables), and the region definitions at WL86 line 51.
- **Not done:** nothing was run beyond a numpy read and analytic arithmetic. No files were edited.
- **D1 is out of scope as a defect.** D1 is LSIT's conflict with the 2026-11-13 owner decision: explicit events 1–2 versus explicit transport kept as a comparator only. I treat it as an **unresolved contract conflict**, not something to fix here.

## 0. ER closure facts that the conclusions depend on (OBSERVED in code and the stored state)

**Closure form (`closure83_dr.py:121–155`):**

f = Σ_outer λ_k·g_k(ωi,ωo) + s(μi)·s(μo)·[λ0/π + η⁻²·Σ_inner λ_k·g_k(t(ωi), t(ωo))]

- g_k = F·D·G2/(4·cos·cos) with anisotropic GGX and height-correlated Smith masking. I checked the identity 1/(2(co·si + ci·so)) = G2/(4·ci·co).
- Each lobe has its own Schlick f0 on ωi·h.
- Inner lobes are evaluated at the **refracted** directions, so their cosines are interior cosines. They are scaled by 1/η².
- s(μ) is a per-RGB piecewise-linear factor.

**S1 panel (row 1 of `states`):**
- Red: outer k0 λ 0.571 (α 0.044, **f0 0.0074**), outer k1 λ 0.124 (α 0.075, f0 1), inner k4/k5 λ 0.150/0.149 (α 0.031/0.026, f0 1).
- Green: k1 0.265, k2 0.129 (f0 0.0049), inner k4 0.503.
- Blue: k1 0.241, k2 0.204, inner k3 0.103, inner k4 0.444.
- Lambert λ ≤ 0.0021.
- η 2.4955. Anisotropy up to 12 % (k3: α_u 0.0299 vs α_v 0.0266).
- s knots, red: 0.33 / 0.86 / ≈1 / ≈1 / ≈1 at μ = 0 / 0.15 / 0.4 / 0.7 / 1.

**S2 sphere** has the same multi-lobe structure:
- green is dominated by outer k2 (λ 0.65, α 0.113, f0 0.05);
- inner k4 is α 0.127;
- η 2.119.

**Bins (`pt86.py:153–156`).** Four rings uniform in cosθ times 16 azimuth sectors starting at −π. Mirror reflection (x,y,z) → (−x,−y,z) moves the sector by exactly 8 and keeps the ring. **The specular map is an exact permutation of the 64 bins** (SUPPORTED INFERENCE, analytic from code).

## 1. Findings A1 / A2 / A3 / A6 / A8

| Q | Finding | Label | Justification |
|---|---|---|---|
| **A1** — can f_L = F(θd)·D(θh)/(4 cosθi cosθo) reproduce the frozen ER lobes? | **No, not exactly. LSIT's statement that the form "matches ER's own microfacet structure" (§4.2 C1) is contradicted for these materials.** Whether a close enough approximation exists is untested. | **CONTRADICTED** (exact match / structural claim). Approximate adequacy: **PLAUSIBLE HYPOTHESIS, untested** | (i) **Refracted inner lobes have a width that depends on θd.** To first order, interior half-angle θh' ≈ k(θd)·θh with k = cosθ/(η·cosθ'). The same factor holds in-plane and transverse (derivation from the t(ω) map in `closure83_dr.py:91–97`). For S1 (η 2.4955), k = 0.40 / 0.30 / 0.21 / 0.076 at θd = 0 / 45 / 60 / 80°. An inner lobe with α 0.031 therefore has exterior θh width ≈ 0.078 / 0.105 / 0.146 / 0.41 rad. A product F(θd)·D(θh) has one fixed θh width, so it cannot express a 5× width change. (ii) **Each colour mixes lobes with different α and different Fresnel curves.** Red outer k0 (f0 0.0074) carries 0.57 × 0.0074 ≈ 0.004 at normal incidence but 0.57 × F(80°) = 0.57 × 0.39 ≈ 0.22 at 80° (Schlick, 0.8264⁵ = 0.385). So red's dominant lobe changes from α ≈ 0.03–0.075 lobes near the normal to α 0.044 near grazing. Σk Fk(θd)·Dk(θh) with distinct Dk is not rank-1 separable. (iii) **Masking-shadowing.** G2 is omitted from f_L. G2 depends on θi and θo separately (that is, on φd), which a (θh, θd) function cannot carry; G2 → 0 at grazing while 1/(cos·cos) → ∞. (iv) **s(μi)·s(μo)** and the interior-cosine denominator also depend on φd off the peak. (v) **Anisotropy** of up to 12 % on S1 is aligned with the shading frame. (vi) LSIT does not say whether D and F are per RGB channel. A single D shared across channels would fail (ii) outright. |
| **A2** — can non-negative, reciprocal f_L and f_S satisfy f = f_L + f_S with M^S low-rank? | **Obstruction 1 (positivity):** f_S ≥ 0 means f_L ≤ f pointwise. Where the product family cannot follow f's shape (A1 i–v), the best feasible f_L sits under the pointwise envelope, and f_S keeps residuals concentrated at the lobe peak. **Obstruction 2 (rank):** S1 has essentially no broad part (Lambert λ ≤ 0.002), so M^S for S1 is almost entirely this peak residual. A residual that follows the specular map is ≈ Π·diag(d) for an exact bin permutation Π, whose singular values are \|d_b\|. Its **relative** rank does not depend on the residual's amplitude: for d nearly flat over 64 bins, the rank-16 relative residual is about √(48/64) ≈ 0.87, against the measured full-S1 value of 0.46–0.52. **Obstruction 3:** P-T1's criterion "M^S needs rank > 16 for 99 % flux energy" is normalised by M^S's own energy. It can therefore reject even a tiny residual, while "≤ ⅓ of the full S1 value" is ambiguous about its normalisation. **Note:** M^S is used as an exact 64×64 matrix in the explicit L₂ (C3). Low rank matters only for R_θ's rank-16 factors. | Obstructions 1–2: **SUPPORTED INFERENCE** (analytic, from code and stored state). That P1 fails numerically: **PLAUSIBLE HYPOTHESIS** (residual rank not computed). Criterion ill-conditioning: **SUPPORTED INFERENCE** | Positivity plus equality forces f_L ≤ f exactly; nothing in LSIT §4.2 relaxes this. Permutation structure: `pt86.py:153–156`. S1 broad share: `mats86.npz`. Full-S1 rank values: WL92 "Local operator" table. |
| **A3** — does the formulation handle masking-shadowing, layering, normalisation, energy and grazing? | **Energy is automatic** under exact positivity: f_L ≤ f gives a_L(ωi) ≤ a(ωi) ≤ Σλ ≤ 1 (certified by the softmax, `closure82.py` header), and a_S = a − a_L ∈ [0, a]. Energy is **not** guaranteed if positivity is relaxed "within tolerance", or for R_θ: softplus factors are unbounded and there is no passivity constraint (`model86.py` form). Because R_θ is applied once, not iterated, this cannot diverge, but it can gain energy. **Masking:** absent from f_L; pushed into f_S, or into F(θd) along the peak only. **Layering:** the refracted inner lobes and s(μ) are not represented (A1). **Normalisation:** D is a free table, which is fine. **Grazing:** f_L without G diverges as 1/(cosθi·cosθo), so F(θd → 90°) must be driven down to satisfy f_L ≤ f. Ring 0 covers 75.5–90° and also receives directions below the horizon (bin clamp). The lobe estimator weight ρ^L(θl) under D-sampling equals f_L·cos/pdf only approximately: the factor F·(ωo·h)/(cosθi·cosθh) drifts off the peak and at grazing. Valid-hemisphere normalisation on the sphere is not specified. | Energy under positivity: **SUPPORTED INFERENCE**. R_θ energy bound: **UNSUPPORTED** (no mechanism). Masking, layering, grazing representation: **CONTRADICTED** as exact, approximate quality untested | `closure83_dr.py:99–155`; `closure82.py` certificate; LSIT §4.2 C1, C4 and §8 ("linear and non-negative"; no passivity claim for R_θ). |
| **A6** — is neglecting sharp lobes at order ≥ 3 a change of the target problem, and who owns those events? | **It changes the problem.** The reference t (ER path tracing, events 2..5 leaving x2) contains every lobe event. **Ownership is self-contradictory.** C4 says lobe events at order ≥ 3 are "not represented (truncation A2)" and that R_θ "never represents a sharp local lobe", yet R_θ is trained on t − t2 (C4 and Stage 1). That target contains those events, so R_θ is pushed to absorb them, which violates its own contract. That absorption is the WL86A/WL92 "global term compensates blindly" pattern LSIT wants to avoid. **An unlisted, large instance:** the **last** scattering event at a sharp table patch. When x2 is the panel or the sphere (mirror footprint, sphere seeing the panel, panel seeing the sphere), the table's L₃₊ at x2 is the material's lobe applied to the incident field of 2 or more events. By contract R_θ supplies only the K_S-like part there, which is ≈ 0 for S1. So at sharp x2, L₃₊ ≈ dropped. For scale: 3–5 events are 0.46 / 0.47 of the region-C higher-order reading on E2 T1 / T0 (WL92 table). LSIT's §4.7 lists only *chained* sharp events as the failure case, not the readout patch's own lobe. | **CONTRADICTED** (document-internal inconsistency between C4 truncation and the t − t2 target). Magnitude at sharp x2: **PLAUSIBLE HYPOTHESIS** (share not measured). A low-rank absorption for broad inputs is feasible in principle (a permutation restricted to smooth inputs is low-rank), but it is forbidden by the contract | LSIT lines 130–132, 243; fact sheet §2 (t = 2..5 events); `pt86.py:228–233` (table read at x2 in the bin x2 → x1); WL86 line 51 (region C includes panel/sphere receivers and the mirror footprint). |
| **A8** — event correspondence and camera readout | **The correspondence is consistent.** LOW: e1 at x1, then e2 by next-event estimation at x2. The table at x2 holds radiance leaving x2 with ≥ 2 events, which become camera events ≥ 3. L₂ = M^S·E1 + lobe(E1), where E1 is the once-scattered incident field (it excludes direct light and q at x2). So nothing overlaps with LOW, given **f_L + f_S = f exactly**. **Gaps:** (1) lobe events at table order ≥ 3, including x2's own scattering (A6); (2) any split tolerance. If f_L + f_S ≠ f, the table and q^S/E1 use a kernel different from the camera bounce, which uses the full ER f. A deficit becomes missing energy; clipping f_S at 0 instead would add energy (a double count). **Errors that are not gaps:** (3) event 2 at a sharp receiver uses incident-bin centres (bin quantisation); (4) R_θ reads L₂ through binned state-ray flight. **No double counting by R_θ:** its target is t − t2, not t − L₂, so it has no incentive to compensate L₂ errors. | Correspondence: **OBSERVED** (LSIT §4.2 C5, `pt86.py:228–233`). Gaps (1) and (2): **SUPPORTED INFERENCE** | As cited. |

## 2. Contract audit

| Contract | Status | Label | Basis |
|---|---|---|---|
| **Reciprocity** | f_L is symmetric by form. f_S = f − f_L is symmetric because ER is exactly reciprocal (`closure82.py` header). M^S, as a bin double-average of a symmetric f, is symmetric up to Monte Carlo noise. **R_θ** (ECRO form a·(c·flight + P·ΣAQ)) has no reciprocity constraint. The explicit relations are reciprocal only in expectation. | K parts: **SUPPORTED INFERENCE**. R_θ: **UNSUPPORTED** | `model86.py` form; LSIT §4.2 C4 |
| **Non-negativity and energy** | Holds for the K parts if f_L ≤ f is enforced exactly (A3). R_θ is non-negative but has no passivity bound. The architecture's passivity requirement (`Architecture.md` §3) is stated only for K. | Mixed: K parts **SUPPORTED INFERENCE**; R_θ energy **UNSUPPORTED** | — |
| **Source linearity and zero response** | Explicit E1 and L₂ are linear in e and s. R_θ is linear in L₂, with a source-independent generator; zero maps to zero. | **SUPPORTED INFERENCE** (by construction) | LSIT §4.3, §8 |
| **Event-order correctness** | The LOW / table / L₂ / L₃₊ partition is correct. The ownership of lobe events at order ≥ 3 contradicts the training target (A6). | **CONTRADICTED** | LSIT 130–131 vs 243 |
| **No double counting** | Holds only if f_L + f_S = f exactly; otherwise there is a gap or a double count (A8 (2)). | Conditional, **SUPPORTED INFERENCE** | — |
| **Material vs world frame** | **Unspecified for curved patches.** Sphere bins, state rays and the source encoding use the **patch-centre frame** (`pt86.py:179–192`). Normals within one sphere patch differ by tens of degrees. LSIT does not say whether h, θh and θd use the per-point normal. Its second-event lobe is built "around incident-bin centres" defined in the patch-centre frame, and those centres can fall below a point's local horizon. M^S, like M64, is a flat-frame matrix applied to sphere patches with no horizon or valid-solid-angle correction; the effect is unquantified. The panel uses one fixed frame, so it is fine. | Gap: **SUPPORTED INFERENCE**. Magnitude: **UNDETERMINED** | Fact sheet §1, §6; LSIT §4.2 C2(b) |
| **Source / receiver identity of intrinsic scattering** | Identical K_L / K_S at events 1–2 by construction. The camera bounce uses the full ER, so it is identical only if the split is exact. At orders ≥ 3 the learned, geometry-conditioned factors of R_θ are not K, so identity is not enforced there. The H-own pathology (geometry-specialised factors, as in WL92B: 14.5 % geometry change, 51 % between seeds) can recur inside R_θ, especially if R_θ absorbs sharp-x2 scattering (A6). LSIT's claim that the stage conflict "has no place to arise" therefore holds only for events 1–2. | Events 1–2: **SUPPORTED INFERENCE**. Orders ≥ 3: **PLAUSIBLE HYPOTHESIS** of recurrence | LSIT §4.4 |
| **Preservation of sharp directional information** | Exact per ray only for the **sender emission at event 1**. It is binned at: (a) the **receiver's incident field** E1[j,b], before event 2. For a sharp receiver the lobe is then placed at the bin-centre mirror image. The mirror map is an isometry, so the reflected direction is displaced by the full in-bin offset (up to 11.25° of azimuth and 7–21° of polar angle), against S1 lobe radii r50 of 0.05–0.15 rad. This is quantisation, which moves in jumps under edits. (b) The L₂ table read by R_θ's flight. (c) The output table, which is adequate per WL86. (d) M^S, which uses the within-bin uniform incident mean. LSIT §4.4 says "the output is binned only when deposited into the table at the receiver". That contradicts its own C3 (`lobe_relation_bins_j`, "around incident-bin centres"). | Binning sites: **OBSERVED** in LSIT text. "Binned only at deposit": **CONTRADICTED** (internally). Magnitude for sharp→sharp: **PLAUSIBLE HYPOTHESIS** | LSIT lines 109–110, 122, 165–166 |
| **Information sufficiency at the explicit → learned interface** | R_θ sees L₂ as [1270, 64, 3] and propagates it by uniform-in-bin state rays. Two different sharp L₂ fields with equal bin values give the same R_θ output while the truth differs (counterexample below). WL90 measured order-≥ 2 projection as negligible (compounding ≤ 9 %), but on W1/W2, whose L₂ was not sharpened by an exact first gather. | **PLAUSIBLE HYPOTHESIS** (insufficient where L₂ is sharp). Size untested | WL90 ledger 90-3 |
| **Geometry-edit invariance of intrinsic parameters** | f_L and M^S depend on the material only. R_θ weights are fixed and its factors are regenerated per geometry. This is legitimate for C-like content, but not if the factors absorb sharp-x2 lobes (A6). | K: **SUPPORTED INFERENCE** (by construction). R_θ: **PLAUSIBLE HYPOTHESIS** of violation | — |
| **Dynamic visibility of selective lobe-relation updates** | Reusing WL86's conservative test is sound in principle: the segment crosses the old or the new mover OBB with a 1 mm pad, misses are extended to t = 30, and rays whose origin is on a mover are included (`ecro86.py:57–67`). Per-source event-1 lobe rays are re-emitted on every edit. **Arithmetic gap:** the selective rule needs the stored segment (24 B per record, as in WL86). 1,212,416 records × (2+1+4+24) B ≈ 37.6 MB, above LSIT's "≈ 10–20 MB" (§4.6). | Soundness: **SUPPORTED INFERENCE**. Storage estimate: **CONTRADICTED** (if segments are stored) | `ecro86.py`, `pt86.py:362–376` |

## 3. Critical contradictions (only those I can justify precisely)

1. **Ownership of lobe events at order ≥ 3 versus the training target.**
   - C4 (line 130) declares those events "not represented", and §8 says R_θ "never represents a sharp local lobe".
   - C4 (line 131) and Stage 1 (line 243) train R_θ on t − t2, which contains them.
   - So R_θ must either absorb them, violating its contract and repeating the WL86A/WL92 blind compensation, or fit an objective inconsistent with its declared ownership.
2. **The readout patch's own lobe at order ≥ 3 has no owner.**
   - When the camera's x2 is sharp (panel or sphere), the table's L₃₊ at x2 is ER's lobe applied to the incident field of 2 or more events.
   - Under the contract, R_θ provides only K_S-like scattering there, ≈ 0 for S1, whose Lambert λ is ≤ 0.002.
   - This content (3–5 events = 0.46–0.47 of region-C h, WL92) is dropped at sharp x2. It is not the "chained sharp events" case that §4.7 lists.
3. **P-T2 cannot measure the truncation as specified.** "Tag lobe-sampled vertices" names which mixture technique sampled a vertex. ER sampling gives each vertex the weight f_full/pdf_mix whichever lobe was chosen (`closure83_dr.py:164–202`), and next-event estimation also evaluates the full f. In addition, f_L is a fitted half-vector function, not ER's GGX lobe indices. So a technique tag does not split the contribution into f_L and f_S parts.
4. **The Stage 1 "LSIT-oracle" and the P-T3 oracle are not oracles of the LSIT contract.** Both use path-traced L₃₊, which includes the truncated lobe content (contradictions 1–2). The falsification reading "oracle passes, learned fails ⇒ learned ≥ 3 is the limit" therefore mixes up truncation with learnability.
5. **The structural claim "matches ER's own microfacet structure" is contradicted by the closure code (A1).** This covers the refracted inner lobes, whose exterior width changes with θd by a factor k⁻¹ from 2.5 to 13 over 0–80°; per-colour multi-α mixtures with f0 of 0.0049 or 1; G2; and s(μ).
6. **Internal contradiction on binning.** §4.4 says the "output is binned only when deposited". C3 says the second-event lobe is evaluated around incident-bin centres.

## 4. Conceptual counterexamples (analytic)

- **Two facing mirrors.**
  - Event 2 at mirror B uses a bin-centre incident direction, so the reflected beam is displaced by up to half a bin.
  - Every further mirror bounce (B → A → B …) is a lobe event at order ≥ 3, dropped or absorbed.
  - A2 fails by construction. LSIT §4.7 concedes this.
- **Sharp → sharp → diffuse** (light → panel → sphere → floor).
  - E1^L at the sphere is binned per incident bin (0.098 sr) although the panel lobe has Ω50 of 0.0075–0.067 sr.
  - The sphere's event 2 is placed at the bin-centre mirror image, so the floor footprint moves in quantised jumps as the source moves.
- **Narrow directional source (cone or collimated beam).**
  - Event 1 is exact per light, which is an improvement.
  - But two beams that differ within one incident bin at a *sharp* receiver give identical L₂ there. This is the same class as WL86A's identical-encoding beam pairs (patch difference 0.94 vs noise 0.26, post hoc, ledger 86A-4).
- **Curved specular surface.**
  - Across one sphere patch the mirror direction varies by twice the normal variation, which can reach tens of degrees.
  - P = 16 discrete emission points with an S2 lobe radius of about 0.2–0.4 rad risk aliasing.
  - The patch-centre bin centre can lie below a point's local horizon. The flat-frame M^S is not frame-corrected.
- **Identical material, different geometry.**
  - f_L and M^S are invariant by construction.
  - R_θ's factors at panel patches are not, and become geometry-specialised if they absorb sharp-x2 scattering (the WL92B H-own pattern).
- **Different incident fields with similar binned states.**
  - At sharp receivers (incident E1) and at R_θ's input (binned L₂), the outputs are identical while the truth differs.
- **g00010-type encoding ambiguity.**
  - LSIT routes only q^S through the binned s16 encoding. The panel's encoding-driven error therefore scales with the M^S residual share.
  - If positivity forces a large peak residual into f_S (A2), the sharp part of that residual is again encoding-driven and the g00010 ambiguity partly persists (PLAUSIBLE HYPOTHESIS).

## 5. Assumptions to test, and the minimal learning-free Stage-0 check that would falsify each

| Assumption | Minimal Stage-0 check | Falsified if |
|---|---|---|
| A1: the product form F(θd)·D(θh) (per RGB) fits ER | Use the closure82 torch evaluator on a dense (θi, θo, φd) grid. Fit width per θd slice; compare with the analytic k(θd) (exterior width 0.078 → 0.41 rad for S1's inner lobes, 0–80°). Report the flux share of f_L under the constraint f_L ≤ f. | Slice width varies more than the fit tolerance, or f_L captures less than a predeclared share of S1's lobe flux |
| A4 / P1: f_S ≥ 0 and low rank | max(f_L − f) ≤ 0 on the grid. Rank and rank-16 residual of M^S **normalised by the full S1 M**, not by M^S. Report both normalisations. | Positivity violated, or the absolute rank-16 residual is not well below the full-S1 0.46–0.52 |
| A2 truncation share (including x2's own scattering) | Path-traced decomposition with a **deterministic split of the BSDF value** at every vertex, including next-event estimation and x2: weights f_L/pdf and f_S/pdf. Do not use technique tags. Report interior-vertex and last-vertex (sharp x2) shares separately. | Lobe share at table order ≥ 3 > 0.10 of region-C h, either part |
| Last-event ownership at sharp x2 | From stored component targets (t − t2 at panel/sphere patches) and camera throughput: the fraction of region-C h read at sharp x2 | Non-negligible fraction, which means contradiction 2 is material |
| Incident binning at sharp receivers (event 2) | LSIT-L2 with bin-centre incidence vs exact per-ray incidence carried into the second lobe (an EE-style oracle, as in WL90) on region C, E2 / E5 | Difference > noise in region C |
| Sufficiency of L₂ → R_θ | Event-3 field from binned-L₂ state-ray flight vs exact per-ray flight of L₂ | Difference in region C > noise |
| Sphere frame / M^S | M^S flat-frame vs per-point-frame projection (analogous to the W_ray sphere rows) on S2 patches | Per-patch difference above the projection tolerance |
| Lobe estimator weight ρ^L(θl) | Two-patch synthetic: D-sampling with ρ^L vs f_L·cos/pdf with exact weights | Bias above tolerance |
| K_r = 16 variance | 16 independent relation draws of LSIT-L2 region C, scored as WL87/WL91 (pass rate, deployed-draw tail) | Single-draw pass rate below a predeclared rate, as EXP's 75 % was |
| Selective update correctness | Null edit and box move: selective vs full lobe-relation re-trace, bit-identity | Any mismatch |
| Energy of the parts | Directional albedo of f_L and f_S per ωi on the grid; flux row sums of M^S | Any albedo > a(ωi), or a row sum > 1 |

**Overall assessment.** LSIT's event-1 per-ray sender emission is a physically coherent extension of the WL90 first-gather finding. I see three structural defects that are independent of D1:
- the lobe-event ownership at order ≥ 3 (contradictions 1–2), including the readout patch's own lobe;
- the non-separability of the frozen ER into the proposed product lobe (A1/A2);
- an estimator-dependent P-T2 and contract-inconsistent oracles (contradictions 3–4).

D1 remains an unresolved owner-contract conflict.