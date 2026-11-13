# IDTS (Candidate B): representation, interface and runtime review (read-only, commit 4ed4ac3)

**Labels.** OBSERVED means a number I checked in the cited file. SUPPORTED INFERENCE (SUP), PLAUSIBLE HYPOTHESIS (PLAU), UNSUPPORTED and CONTRADICTED are used as briefed. ARITH means analytic arithmetic from code constants; it is never a measurement.

**What I re-checked this session:**
- `results/ecro_attr92/results/local.json`: S1 rank-16 EY residual 0.520 / 0.458; 99 % rank 59 / 46; NMF-16 peak 0.472 / 0.535; mirror-bin share of truth 0.42–0.46; unweighted rank-16 residual 0.19 / 0.37.
- `results/ecro_stage93c/oracle_gate.json`: ORA − T0 whole +0.0015 [−0.0011, +0.0059] / +0.0003 [−0.0011, +0.0024]; single +0.0010 / −0.0010, both CIs include 0.
- `experiments/ecro86/model86.py:91-124`: the a and c heads are generated from tokens that carry centroid, normal, tangent, Ft message passing and a global mean.
- `pt86.py:153-156` (bin_of) and `scene86.py:92-100` (sphere frames).
- The truth-split table in WL86A Amendment 1, and the WL92 / WL92B / WL93B / WL90 text cited below.

**Overall verdict on IDTS as defined.** IDTS is not falsifiable in its current form. Each of B1–B6 is either a requirement statement or a physically trivial principle. None names an operator, so no measurement can confirm or refute IDTS itself. The evidence does constrain any admissible instance, and two of those constraints rule out the obvious ECRO-derived instantiation (see Critical contradictions).

---

## B-4: Can a learned intrinsic operator express near-specular mappings without geometry specialisation?

**Finding 1. In ECRO's operator class this is not possible for S1. Label: CONTRADICTED for that class.**
- ECRO's local operator is `a cᵀ`: 64×64, rank ≤ 16 per patch and colour, softplus-nonnegative.
- In the flux-transfer norm, S1's material matrix M64 has these properties (OBSERVED, `local.json`, WL92 "Local operator" table):
  - rank needed for 99 % energy: 46–59;
  - Eckart–Young rank-16 residual: 0.46–0.52;
  - rank-16 NMF mirror-bin retention: 0.47–0.54.
- So a material-shared rank-16 factor is at best a 0.46–0.52 residual. The in-family R16 construction fails region C at 0.084 / 0.073 (WL92 ladder, OBSERVED).
- Caveat: the norm was fixed before the freeze. In the plain Frobenius norm the residual is 0.19 / 0.37 (OBSERVED). The claim is therefore norm-specific. The flux norm is the one tied to transported power.

**Finding 2. The per-geometry rank-16 fits do not show that rank 16 is enough for an intrinsic operator. Label: SUP.**
- Fixed-geometry free fits reach 0.016–0.043 (WL92, 92-4, OBSERVED). On validation lights at E2 T0 / E1 T1 they generalise worse than the generator (0.052–0.085 vs 0.037 / 0.059, OBSERVED).
- Here is why. Let V_g,i be the span of incident fields that patch i actually receives at geometry g over the light population. Any operator A = M·P_{V_g,i} matches M on that set, and its rank is dim V_g,i. So a low-rank fit can be exact on one geometry because it encodes the realised incident subspace. That is the "geometry-specialised incident information inside the local operator" that IDTS hypothesises.
- Two observations are consistent with this reading:
  - the operator changes under a box move: OWN 0.145 / 0.146, frozen 0.34 / 0.44 (92B-4, OBSERVED);
  - the fits over-fit across lights.
- An intrinsic K must equal M on the whole physically reachable incident set. Its required rank is therefore rank_ε(M), not dim V_g.

**Finding 3. Operators that are intrinsic in the local frame, high-rank and cheap exist. Label: SUP (analytic). Learnability: UNSUPPORTED (not tested).**
- **Mirror permutation, exact in the deployed bins.** Bins are uniform in cosθ (4 rings) × 16 azimuth sectors with φ measured from −π (`pt86.py:153-156`). Specular reflection maps (θ, φ) to (θ, φ+π), which takes bin (iz, ip) to (iz, ip+8 mod 16) exactly.
  - So M ≈ Π_R · B: Π_R is a fixed full-rank permutation with zero parameters, and B is a banded blur over neighbouring bins with θ-dependent Fresnel / G weights.
  - That gives rank up to 64 at O(64 × bandwidth) parameters.
- **Azimuthal Fourier (block-circulant) form for isotropic BSDFs.** This gives 16 frequency blocks of 4×4 per colour: rank up to 64 with about 4·4·9 real parameters.
  - S1 is only near-isotropic (α_v ≈ α_u within about 10 %, fact sheet §7), so this form is approximate.
- **Continuous per-ray evaluation.** This means evaluating a BSDF in half-vector / Rusinkiewicz coordinates (an analytic lobe sum or an MLP) at exact ray directions, as W_ray does in EXP.
  - Rank is unlimited, and the cost scales with the number of direction pairs.
- **The frozen ER itself** (8-lobe GGX closure; passive and reciprocal by construction, fact sheet §7). It is intrinsic and already preserved by the owner decision.
- Analytic consequence (ARITH/SUP): S1's lobe is narrower than a bin.
  - r50 = 0.146 rad (Ω50 ≈ 0.067 sr at normal incidence; sharpest lobe ≈ 0.0075 sr) against 0.098 sr per bin.
  - Rank therefore grows roughly with the number of cells D at finer resolution.
  - At D = 1024 (cell 0.0061 sr), the 99 % rank should be a few hundred (PLAU, not computed). Only structured or per-ray classes stay tractable there.

**Finding 4. What IDTS must specify for B1.**
- (a) The class and basis: binned D, structured Π_R·B, Fourier, or per-ray continuous.
- (b) The resolution D.
- (c) Whether K_phi is learned at all, or is ER / its projection M_D.
  - If K_phi is trained only on ER samples (no geometry in its data), B1's invariance holds by construction. "Learned" then adds nothing over ER.
  - If K_phi is trained on scene targets, specialisation can return (Finding 2).
- (d) The frame on curved patches; see the counterexample below.
- (e) Linearity, nonnegativity, passivity and reciprocity, as required by Architecture §3.

**Counterexample: curved specular surface, and the same material on different geometry. Label: CONTRADICTED for a patch-level shared K in a single frame.**
- Sphere patches use one patch-centre frame (`scene86.py:92-100`; bins and M64 in the flat frame, fact sheet §5–6).
- Normal deviation from the patch centre (ARITH):
  - cap band z ∈ [0.5, 1]: up to about 41° toward the pole, ±19.5° in azimuth;
  - equatorial band: about ±15° polar, ±22° azimuth.
- Mirror outgoing directions therefore spread by twice that, about ±30–80°. S2's lobe is r50 0.21–0.40 rad (12–23°).
- The patch-level kernel is K_patch(wi, wo) = (1/A)∫ f(R_x wi, R_x wo) dA. It depends on the normal distribution inside the patch.
- So "the same intrinsic K shared by every patch of a material" is false at patch level for curved patches. S1 on a flat panel and S1 on a sphere would need different patch operators.
- What stays true: the operator is invariant under rigid motion of the curved object. IDTS must define K_phi at the point level and assign the normal-distribution convolution to the geometry side (C^t or a patch-geometry term).
- Whether the flat-frame M64 biases ORA on the sphere is UNDET (fact sheet §9).

---

## B-6: Does a geometry-conditioned G_psi bring back scene-context entanglement?

**Finding. Yes, unless G_psi's target, inputs and path-class ownership are constrained. Label: SUP on ECRO-family evidence. For IDTS this is UNDETERMINED, because G_psi is undefined.**

Evidence (all OBSERVED):
- **WL86A Amendment 1 truth split.** The single learned flight delivers 0.2–3.6 % against a 12–19 % panel family. The global term adds +0.02…+0.06 in E2 and +0.04 in E1, where the family is only 1.4 %. That is geometry-blind compensation with opposite-sign errors on the two edits.
- **WL92B.** Replacing the global term by truth (X_L3) worsens E2 T1 (f −0.36 / −0.68), so the global term compensates. Exact receiver alone (X_M2) also worsens (−0.26 … −0.59): co-adaptation through tied factors.
  - Panel `a cᵀ` seed-to-seed difference is 0.51 (OWN) and 1.76 (SUM).
- **92B-3.** The LSPOT regression is attributed to the global term's extrapolation (X_L3 accounts for 71–82 %).
- **93B-1/3.** U0 gets the best whole fit through a non-physical decomposition: source stage worse than init, F4 lost.
- **93C-1.** ORA (exact M64 at both uses, with the frozen OWN global co-trained on learned locals) shows no level conflict. But its whole error is not better than T0 (+0.0015 / +0.0003, not significant).

Mechanism (SUP, analytic):
- If G_psi is trained on the SUM target, or on the residual t − (K∘C chain), its loss is minimised by absorbing that chain's geometry-specific errors. Entanglement moves from K into G even when K is frozen and intrinsic.
- Only an exclusive physical component target removes this incentive, for example path-traced events ≥ m that the K∘C chain does not produce. WL92A OWN did exactly that and halved the component errors (single 0.210 → 0.104; global 0.218 → 0.092). On admitted E5, though, OWN was not consistently better (92B-5, OBSERVED).
- So an exclusive target is necessary but not demonstrated to be sufficient.
- **B6 tension.** With weights fixed, G_psi must generalise its geometry conditioning to edited and unseen geometry. WL86 GENERALIZATION is NOT SUPPORTED, and no never-inspected admitted sharp held-out exists (ledger item 19).

**Counterexample: two facing mirrors (ARITH/SUP).**
- Take mirrors at z = 0 and z = h, lit from polar angle θ. The k-th image shifts laterally by k·h·tanθ while the direction alternates w, R(w).
- The order-k response is a geometry-parameterised sequence of narrow spots. Its locations depend on h and on the configuration.
- A G_psi that owns order ≥ 3 must represent this. It can do so only by encoding the geometry (entanglement), or by passing its input through a high-resolution transport, which makes it an explicit chain.
- A smooth or low-rank G (like ECRO's rank-24 Q/P) cannot represent it.
- In the WL86–93C room, panel→panel and sphere→sphere transport are zero: the panel is planar and the sphere convex. Multi-specular chains are therefore nearly absent, and there is no measurement for this case (UNSUPPORTED either way).

---

## B-7: Does the z^t / K_phi / G_psi interface carry enough information for higher-order paths?

**Finding 1. The definition does not say whether z^t is per patch, per point or per relation ray; its resolution; or whether G_psi consumes z^t, K outputs or both. Sufficiency is therefore undecidable. Label: UNDETERMINED.**

What the evidence requires of any choice:

**(i) A 64-bin per-patch state is not a sufficient statistic for the sharp response. Label: OBSERVED / SUP.**
- WL90: the explicit-vs-oracle gap is the angular projection of L1 in the first gather (97–102 % isolated). With 4 sub-bins (256 directions) 9–20 % of J remains; 16 sub-bins (1,024 directions) remove it (+0.0007 / +0.0012 / −0.0031).
- These are single-read lower bounds; the refined recurrence was not run.
- Counterexample, "multiple incident fields → the same binned state":
  - Two delta incidences inside one bin, one at the centre and one near an edge, give identical s (64-bin) values.
  - Π_R maps both into the same mirror bin. But S1's lobe (r50 ≈ 8.4°) is narrower than a sector (22.5°) and than the polar rings (14.5–41.4°). The edge case spills a large share of its flux into neighbouring outgoing bins; the centre case does not.
  - M64 assumes uniform in-bin incidence and keeps only 0.42–0.46 of the flux in the mirror bin (OBSERVED `peak_share_of_truth`). So it must be wrong for at least one of the two.
- Measured instances:
  - g00010: M(enc) 3.77 vs noise 0.66 (93A-2, OBSERVED);
  - identical-encoding beam pairs: 0.94 vs noise 0.26 (86A-4, post hoc, OBSERVED).

**(ii) Within-patch spatial structure.**
- In the point-light WL90 workload, spatial refinement had no effect (OBSERVED).
- For a narrow spot that partially covers a patch, a per-patch angular field cannot localise the reflected spot. LPAT shows the spot smeared in every arm (93B-5, visual only).
- LSPOT89 gained the most from exact per-ray once-scattered reads (−0.0266, ungated, 91-1).
- So z^t's spatial resolution is a separate obligation (PLAU that a spatial component is needed for narrow / spot sources).

**(iii) Sharp events at order ≥ 3 are large in region C, and their ownership is undefined.**
- Events ≥ 3 inside table B are 0.461 / 0.472 of what region C reads (E2 T1 / T0), and 0.622 of E1's panel (WL86A Amendment 1 table, OBSERVED).
- Their sharp-chain share (paths with two or more sharp scatterings) is unmeasured.
- Counterexample, sharp → sharp → diffuse (light → panel → sphere → floor):
  - The caustic position on the floor is set by two near-delta direction maps. At about 2 m from the last specular vertex, one bin's angular width (≈ 0.39 rad) smears this to about 0.8 m.
  - If G_psi owns these paths and consumes a low-dimensional summary (ECRO's global term consumes u = c·s, a rank-16 encoding of the direct field, `model86.py:117-121`), the caustic position is not recoverable.
  - If G_psi consumes the full-resolution state, it is an implicit learned transport over N·D inputs. Its cost and geometry dependence then fall under B-8 and B-6.

**(iv) The final readout does not need high resolution; the intermediate state does.**
- The WL86 oracle patch × 64-bin output table reaches E_H at noise (C 0.0074 / 0.0137, 86-5, OBSERVED).
- So the camera-side 64-bin readout is adequate. The information requirement is on the intermediate state (first gather; possibly later orders in multi-specular scenes, where WL90's compounding of ≤ 9 % and m* = 1 may not hold, PLAU).

**What IDTS must specify for the interface:**
- z^t's carrier: per-patch grid, per-point, or relation-ray-implicit as in X1.
- z^t's resolution, both angular and spatial.
- Per-order indexing: z_1 for direct light, z_2 after one flight, and so on.
- Whether G_psi reads z, K∘z, or the outgoing field after the last explicit order.
- Whether G_psi's output is fed back into C^t or only added to the readout table.
- Which path classes (by order, or by lobe sharpness) go through K∘C and which through G.

---

## B-8: Runtime and storage risks

### Bounds (ARITH unless marked)

Assumptions: N = 1,270 patches; 3 colours; fp32; one summed lighting unless stated; ray-cast throughput taken from WL86. Extrapolations are labelled PLAU.

| Item | 64 bins (deployed) | 256 dirs (WL90: 9–20 % residual) | 1,024 dirs (WL90: gap removed) | ≈ 3,350 (4 cells per sharpest S1 Ω50) |
|---|---|---|---|---|
| z^t per-patch field, all patches | 243,840 values, 0.98 MB | 3.9 MB | 15.6 MB (7.8 MB fp16) | 51 MB |
| ER patches only (330) | 0.25 MB | 1.0 MB | 4.1 MB | 13.3 MB |
| Sharp patches only (panel 42 + sphere 32) | 0.06 MB | 0.23 MB | 0.91 MB | 3.0 MB |
| Per source, if kept unsummed | × L | × L | × L (16 sources: 250 MB) | × L |
| Dense K_phi per material, D²×3 | 12,288 floats (M64 for 6 materials 73,728, OBSERVED) | 0.79 MB | 12.6 MB per material (3 ER materials 38 MB) | 135 MB per material: needs a structured class |
| K application on 330 ER patches, D² MAC × 3 | 4 M MAC | 65 M | 1.04 G MAC (ms-scale on GPU, PLAU) | 11 G MAC |
| Relation rays N·D·κ, κ = 8 | 650,240 (OBSERVED design; 17.6 MB with segments) | 2.6 M | 10.4 M (≥ 42 MB hit + bin; 290 MB with segments) | 34 M |
| Relation rays, κ = 1 | 81k | 325k | 1.3 M (5.2 MB) | 4.3 M |
| Full relation rebuild at WL86 throughput (650k rays in 26–28 ms, OBSERVED 86-9) | 26–28 ms | ≈ 0.1 s (PLAU) | ≈ 0.42 s (PLAU) | ≈ 1.4 s (PLAU) |
| Selective rebuild, 8.7–10.2 % dirty (OBSERVED 86-8) | 21 ms measured (fixed overhead dominates) | — | ≈ 1 M rays, about 40 ms plus overhead (PLAU) | — |

**Measured comparators (OBSERVED):**
- **WL86 ECRO:** per-edit 76–78 ms (state 21, operator 11); factors 43 MB.
- **WL87:** PT 36 ms per edit; EXP 89–95 ms; C3 0.43–0.58 s; TF4096 4.6 s. Source-only: ECRO 13–18 ms vs EXP 51–64 ms.
- **WL88:** R1 relation is 943k rays / 31 MB with a 129 ms update.
- **WL90/91 C3URX1 vs C3UR:**
  - source-only 20.6 vs 36.7 ms;
  - per-edit 68.2 vs 83.6 ms;
  - edit → frame 124.7 ms (PT 250.1);
  - relation 20.8 MB + 12.6 MB buffer;
  - 526,467 probe + 526,467 shadow rays per light;
  - 16 sources 314 ms.
- **92A-8:** state full / selective 33 / 17 ms.

### Risks

**Risk 1. O(N²) communication. Label: SUP.**
- A directional patch-pair operator of size N²·D² is infeasible even at 64 bins: 6.6e9 entries, about 26 GB.
- So C^t must be ray-sparse (O(N·D·κ)), as ECRO and EXP already are.
- O(N²) can return through G_psi:
  - ECRO's generator already uses a dense Ft (1.61 M floats) and dense message passing (`model86.py:83-100`);
  - at N = 10⁴ that is 10⁸ floats (400 MB); at N = 10⁵, 4·10¹⁰.
- Attention-style G scales the same way. WL52's learned C was dense O(N·M) and 8–19× slower than explicit C (AGENTS.md WL52 record).

**Risk 2. Angular storage.**
- A stored per-patch grid at the WL90-sufficient 1,024 directions is modest at this N: 15.6 MB, comparable to ECRO's 17.6 MB state.
- It scales as N·D·L. Keeping per-source fields for source-only updates multiplies it by L.
- X1 shows an alternative: exact first-gather information without a stored grid, by evaluating once-scattered radiance at each relation ray's exact direction (12.6 MB buffer). That is a relation-ray-implicit z.
- IDTS must choose. The choice decides whether storage or per-ray evaluation is the cost driver.

**Risk 3. Update work.**
- Every geometry edit invalidates C^t and every z_k for k ≥ 2.
- Every source change invalidates z_1 and everything downstream. That includes G_psi's input if G reads z, so G must be re-evaluated, and its cost enters the source-only path.
- The source-only path is the one measured learned advantage (WL87: 13–18 vs 51–64 ms). IDTS keeps that advantage only if G_psi is cheap and linear in the source.
- No measured selective-update saving is material at this scale (5–7 ms, ~1 ms, 16 ms; ledger item 14). A larger D·κ makes selectivity matter more (PLAU).

---

## B-10: Missing definitions that block implementation-level verification

1. **K_phi operator class.**
   - Basis: bins D, Π_R·banded, azimuthal Fourier, or continuous per-ray.
   - Parameter count and rank.
   - Point-local vs patch frame, and the treatment of curved patches (normal-distribution convolution: who owns it).
   - Anisotropy frame.
   - Lobe representation (does it reproduce mirror-bin retention ≈ 1 relative to M_D?).
   - Relationship to the frozen ER: is K_phi = ER, M_D, or a learned surrogate, and why is learning needed?
   - Constraints from Architecture §3: linear, nonnegative, passive, reciprocal.
2. **K_phi training target.**
   - Material-only (ER samples, geometry-free, which makes B1 hold by construction) or scene / transport targets (which risks re-specialisation).
3. **Representation of z^t.**
   - Carrier: per-patch grid, per-point / sub-patch, or per relation ray.
   - Angular and spatial resolution.
   - Radiometric quantity: radiance vs cos-weighted irradiance per cell.
   - Per-order index and per-source vs summed storage.
   - Construction of z_1 from sources: replacement of s16 with its 16 random points and 64-bin aggregation, and the g00010 ambiguity.
   - Light-count limit (the current code is hard-limited to 2 lights, `scene86.py:230-237`).
4. **Propagation C^t.**
   - Relation sampling (κ, stratification, importance toward sharp senders: WL88 shows sender placement dominates, +0.050 / +0.053).
   - Source-independent vs per-source relations (C3 / C3UR trade-off).
   - Exact-direction transfer vs binned transfer; dirty-ray rule; closure / conservation.
   - Which orders it propagates.
5. **G_psi.**
   - Inputs: z_1, z_m, K outputs, geometry tokens.
   - Output: table B, per-order fields, readout.
   - Whether its output re-enters C^t.
   - Linearity in the source.
   - Geometry-conditioning form and scaling (avoid O(N²)).
   - Passivity / stability of the composition.
   - Generalisation domain, including unseen source classes (LSPOT).
6. **Event-order / path-class split.**
   - Exactly which camera / table events are owned by camera PT (LOW), by the K∘C chains and by G_psi.
   - Order-based or lobe-based. Sharp chains at order ≥ 3 must have an owner; they are 46–62 % of region-C / panel reads in total order-≥ 3 share (OBSERVED), with the sharp-chain share unmeasured.
   - Its compatibility with the 2026-11-13 owner decision (see contradiction C4).
7. **Training targets for G_psi.**
   - Exclusive physical component (OWN-style events ≥ m) vs residual. The residual invites compensation.
   - Normalisation, and per-geometry diagnostics instead of pooled L2 dominated by g00010 (93A-4, 93C-2).
8. **Consistency constraints (B5).**
   - An operational definition of "physically equivalent incident distributions" at the chosen resolution.
   - A rule for states that are equal in the basis but physically different.
   - A test that the identical K_phi is applied at source and receiver uses (identity by construction vs by tolerance).
9. **Reuse contract (B6).**
   - Which rigid edits are supported, including rotation of the sharp object itself (z is re-expressed in its frame).
   - What is recomputed.
   - Quantitative acceptance: region-C E_H, edit slope / corr thresholds.
   - Comparator roles per `docs/BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md`: Neural Full Recompute vs internal F/S vs PT reference.
10. **Workload.**
    - An admitted, never-inspected held-out with sharp readers and multi-specular chains. None exists now (ledger item 19; E6 never generated).
11. **Cost budgets** for per-edit, per-source and storage, set against the WL86/87/91 measured costs.

---

## Evidence-to-claim matrix

| Claim | Supporting measurement | Contradicting / weakening | Unestablished assumptions | Follows logically? | What would discriminate | Label |
|---|---|---|---|---|---|---|
| **Core**: the principal ECRO failure is geometry-dependent incident information encoded in the nominally intrinsic local operator | Generator heads conditioned on geometry (`model86.py:91-107`, OBSERVED). Panel `a cᵀ` changes with a box move: OWN 0.145 / 0.146, frozen 0.34 / 0.44 (92B-4). Per-geometry rank-16 fits pass (0.016–0.043) while material R16 fails (0.084 / 0.073) (92-4 / 92-3). Untied source / receiver differ 57–68 % (93B-2). ORA region C 0.030 / 0.033 vs T0 0.076 / 0.042, slope ≈ 1.0 (93C-3, inspected E2) | Seed-to-seed variation (0.51) exceeds geometry variation (0.145): non-identifiability fits as well as encoded incident information. Competing measured causes: relation sampling (median 13–17 / 512 rays hit the panel, −20 %, 86A-8); first-gather angular projection (WL90); global extrapolation for LSPOT (92B-3); g00010 encoding (93A-2). ORA ties T0 on pooled single / whole (CIs include 0, `oracle_gate.json`). K/C violation not decidable (WL92B) | That removing geometry conditioning from the local operator is the dominant fix; that ORA's sharp gain transfers to held-out data | Does not follow: "principal" needs a ranking of causes, which no batch produced | Material-only-trained K vs the ECRO-tied operator at the same relation, encoding and G, across ≥ 2 geometries and an admitted held-out; separately vary relation sampling and resolution | PLAUSIBLE HYPOTHESIS (component facts OBSERVED; "principal" UNSUPPORTED) |
| **B1** intrinsic, shared, rigid-invariant, sharp without specialisation | ER is intrinsic, passive and reciprocal (fact sheet §7). M64 at both uses improves sharp renders (93C-3). Mirror-permutation and Fourier structure exist analytically | Rank-16 class: S1 residual 0.46–0.52, retention 0.47–0.54 (`local.json`). Learned retention 0.11–0.21 (92-2, 92A-2). Curved patches: a patch-level kernel depends on the normal spread (analytic, ±30–80° mirror spread on sphere patches) | That a *learned* operator in some class achieves it; what learning adds over ER | Coherent as a requirement. Achievability does not follow from any measurement | Fit the candidate class to M_D (S1 / S2 / B) material-only: residual, retention, passivity. Then compose ORA-style at several geometries | CONTRADICTED for the ECRO rank-16 class; UNSUPPORTED for a learned K in other classes; SUP that non-learned intrinsic operators (ER / M_D) exist |
| **B2** z^t keeps the angular information needed for sharp interreflection | WL90: first-gather angular projection isolated (97–102 %); 1,024 directions remove the gap, 256 leave 9–20 %. g00010 M(enc) 3.77 vs 0.66. 86A-4 beams 0.94 vs 0.26 (post hoc). X1 exact reads −0.016 [−0.020, −0.012] (90-6) | X1's held-out advantage holds in only 1 of 3 conditions; LSPOT89 C_ss harm +0.0025 (91-1). ECRO's s is already 64-bin directional, so the contrast with an "irradiance summary" does not describe ECRO; its compression is u = c·s (rank 16) | Resolution, carrier, spatial component; that m* = 1 holds beyond this room | Requirement direction: yes. Sufficiency of any z: not decidable without a representation | Exact composition at D ∈ {64, 256, 1024} on g00010, beams, LSPOT and a multi-specular case | SUP (as a necessary property); content UNDETERMINED |
| **B3** C^t owns visibility / free space; updates need no retraining of K | State relation is geometry-only: 21 ms selective, 26–28 ms full (86-9); 8.7–10.2 % dirty rays (86-8). C3UR / X1 relations source-independent (90 / 91) | Source-independent relations are sampling-limited for sharp senders: SAM coverage 0.59 → 0.99 (88-4); sender placement dominant (88-3, single draw). A sharp-adequate C^t may need material- / source-aware sampling | That a source-independent C^t at a feasible κ suffices for sharp senders | Yes, if K_phi is geometry-free (by construction) | Sharp-sender coverage vs κ and importance, at fixed K and z | SUP (no-retraining part); OBSERVED feasibility of geometry-only relations; sampling adequacy UNSUPPORTED |
| **B4** G_psi keeps a meaningful learned higher-order responsibility | OWN global field error 0.218 → 0.092 (92A-1). Broad receivers: the learned term supplies about 23 % at about 2 % error (86-10, labelled interpretation) | Geometry-blind compensation (86A Amendment 1); LSPOT extrapolation (92B-3); explicit ≥ learned in every comparison (87-3: 0 better, 3 worse); order ≥ 3 is 46–62 % of sharp-region reads (OBSERVED); learned C slopes 0.36–0.63 (86-6) | Which paths G owns; its inputs; that it can be exclusive yet generalise | "Meaningful" is undefined, so the claim is not testable | Exclusive-target G on the declared path classes vs explicit ≥ m, at an admitted held-out; edit slopes on sharp readers | UNSUPPORTED (sharp / held-out); PLAU (broad / smooth order ≥ 3) |
| **B5** identical intrinsic law at source and receiver uses | Physics: the BSDF is independent of where the light came from. ORA (same M64 at both uses) better on sharp renders, slope 0.995 / 1.000 (93C-3). Untied UQ diverges 57–68 % and U0 is non-physical / fails F4 (93B) | At finite resolution, source-use bins hold delta-like incidence while receiver-use bins hold extended incidence. A uniform-in-bin M_D is biased differently per use; "equal binned states" ≠ "physically equivalent". X_M2 exact receiver alone worsens (co-adaptation, 92B-1) | A definition of equivalence at the chosen resolution | Follows from physics in the continuum; at finite D, only if D resolves the lobe | Same-K both uses at D = 64 vs 1,024 against per-stage physical targets (q, R*) | SUP (principle); at 64 bins, operationally CONTRADICTED by g00010 / beams (binned equivalence classes are not physical ones) |
| **B6** rigid edit: K reused, weights fixed, state updated, GI responds correctly | K part trivially exact if geometry-free. ORA E2 C slope ≈ 1.0 (inspected). D slopes for learned ECRO 0.78–0.97 (86-6) | Learned region-C slopes 0.36–0.63 (86-6); OWN on E2 0.60 / 0.78 (92A-6); WL86 generalisation NOT SUPPORTED; no admitted never-inspected sharp held-out exists | That a fixed-weight G_psi generalises over edited geometry; acceptance thresholds | The K part follows from B1 by construction; the G part does not | Admitted held-out T0 / T1 with sharp readers, comparators per the baseline gate | PLAUSIBLE HYPOTHESIS (K part SUP; G part UNSUPPORTED) |

---

## Minimal specification needed before a Stage-0 falsification means anything

1. **K_phi.**
   - Class and basis; resolution D; point-local frame and curved-patch rule.
   - Its relation to ER (is it ER / M_D, or learned; if learned, a material-only target).
   - The constraint set: linear, ≥ 0, passive, reciprocal.
2. **z^t.**
   - Carrier (grid / sub-patch / relation-ray), angular D and spatial resolution, per-order indexing.
   - Construction of z_1 from lights, replacing or justifying s16.
3. **The ownership split** of camera / table events by order and/or lobe class.
   - In particular, who owns sharp-lobe chains at order ≥ 3.
   - A statement of how it complies with Architecture §1 (2026-11-13).
4. **G_psi.**
   - Input, output, linearity in the source, feedback into C^t or not, scaling class (no dense N²).
   - Exclusive physical target.
5. **Predeclared Stage-0 learning-free checks and thresholds.** Examples:
   - (a) residual / retention / passivity of the K class vs M_D for S1 / S2 / B;
   - (b) exact composition at the chosen D vs WL93C ORA (0.030 / 0.033) and the WL86 oracle (0.0137) on E2 T1, with g00010 and the beam pairs reported per geometry;
   - (c) the order-≥ 3 sharp-chain share in a workload that contains specular chains;
   - (d) storage / update cost vs WL86 / 87 / 91 measured numbers.
6. **An admitted held-out workload** containing sharp readers and multi-specular paths, plus comparator roles per `docs/BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md`.

Without items 1–4, any Stage-0 result tests a particular design choice, not IDTS.

---

## Critical contradictions (precise only)

- **C1. B1 vs ECRO's rank-16 factor class.**
  - S1 flux-norm EY rank-16 residual is 0.46–0.52; the 99 % rank is 46–59 (`local.json`).
  - An IDTS instance that keeps `a cᵀ` (rank 16) as a material-shared K_phi cannot represent S1's response without geometry specialisation at 64 bins.
  - Condition: the flux norm (Frobenius residual 0.19 / 0.37).
- **C2. B1 (sharp) + B5 (same law on physically equivalent fields) vs a 64-bin z^t.**
  - Binned states are not sufficient statistics: g00010 3.77 vs 0.66 noise (93A-2); beam pairs 0.94 vs 0.26 (86A-4, post hoc); WL90 projection gap.
  - So no K_phi on 64-bin states can satisfy both. B2 must fix a resolution beyond 64.
  - In this workload, a single first-gather read needs about 1,024 directions (WL90, a lower bound).
- **C3. A patch-level shared K_phi vs curved specular patches.**
  - Sphere patches use one centre frame. Their normals deviate by up to about 41°, so mirror directions spread over ±30–80°, wider than S2's lobe (12–23°).
  - The patch-level operator is material ⊗ normal distribution: geometry-dependent, though rigid-invariant.
  - "Same intrinsic K for every patch of a material" is false at patch level. It holds only at point level.
- **C4. Contract conflict (conditional, not physical).**
  - If K_phi = ER / M_D and z^t / C^t propagate the orders below G_psi exactly, those orders are explicit transport (the EXP / C3UR class).
  - Architecture §1 (2026-11-13) keeps explicit transport as comparators "rather than the global method to be published".
  - The LSIT proposal hit the same conflict for explicit events 1–2 (`docs/plans/DESIGN_PROPOSAL_LOBE_SPLIT_INTRINSIC_TRANSPORT.md`, lines 174 and 221).
  - IDTS must declare its split before it can be judged admissible.
- **Tension (not a contradiction): B4 + B6 for sharp chains.** No measurement shows a fixed-weight, geometry-conditioned learned higher-order term giving correct sharp-reader edit response. Learned region-C slopes are 0.36–0.63, and only oracles reach ≈ 1.0.

---

## Files (all absolute)

- c:\Projects\ReST-GI\docs\Architecture.md
- c:\Projects\ReST-GI\docs\plans\DESIGN_PROPOSAL_LOBE_SPLIT_INTRINSIC_TRANSPORT.md
- c:\Projects\ReST-GI\docs\worklogs\ (WL86 / 86A / 87 / 88 / 90 / 91 / 92 / 92A / 92B / 93 / 93A / 93B / 93C worklogs)
- c:\Projects\ReST-GI\experiments\ecro86\model86.py
- c:\Projects\ReST-GI\experiments\ecro86\pt86.py
- c:\Projects\ReST-GI\experiments\ecro86\scene86.py
- c:\Projects\ReST-GI\results\ecro_attr92\results\local.json
- c:\Projects\ReST-GI\results\ecro_stage93c\oracle_gate.json
