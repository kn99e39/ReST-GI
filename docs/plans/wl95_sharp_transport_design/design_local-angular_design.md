# WL95 designer report: local-angular track (#local-angular)

**Scope.** This is design only. I edited no repository file and trained or ran nothing. I read the eight evidence files and checked these contracts in code:
- `experiments/ecro86/model86.py:109-124` (apply);
- `experiments/gm1_explicit87/quad87.py:40-110` (M64 cells and fbar);
- `experiments/fnto_compiled83/closure83_dr.py:1-15, 121, 164` (ER closed form, eval_pdf, sample);
- `experiments/ecro86/scene86.py:22-23` (material ids and classes).

**Labels.** [FACT] = verified. [MEAS] = measured value, with its source. [INF] = reasoned inference. [HYP] = untested. No novelty is claimed. Every mechanism below is untested as a fix.

---

## 1 Verified Current Architecture

**Pipeline** [FACT] (auditA §1; read_code §1):
- Source encoding: s[n, 64, 3] is patch-mean direct irradiance per incident bin. It uses 16 fixed points shared by all patches, and the direction inside a bin is discarded.
- Learned local operator: M_n = a cᵀ, rank ≤ 16, non-negative, with c tied between sender and receiver (`model86.py:111-120`).
- Learned once-scattered table: Bq = a(cᵀs), constant per patch × outgoing bin (ΔΩ = 2π/64 = 0.098 sr).
- Flight: E1[n, m] = Σ_{κ=8} Bq[hit, dbin]·z_k ΔΩ/8, over one fixed 512-ray (origin, direction) pattern shared by all patches.
- Receiver: the same c applied to E1.
- Global term: rank 24, driven by the direct activations u, not by E1.
- Readout: hl = thr(x1)·B[patch(x2), bin(x2→x1)], nearest patch, no interpolation.
- Edit: selective state re-trace, bit-equal to a full rebuild, then a full generator recompute. [MEAS] 76–78 ms per edit.

**Material that ECRO compresses** [FACT] (`closure83_dr.py:1-15`). The frozen ER is a closed-form, non-negative mixture:
- a Lambert term plus up to 7 anisotropic GGX lobes;
- 3 outer lobes, and 4 inner lobes refracted through t(w) and modulated by s(μ_i)s(μ_o);
- a 69-float state per material, with exact `eval_pdf` and VNDF `sample`.

So the local angular operator that ECRO's a cᵀ re-learns is already available as an exact continuous function with an importance sampler. The 64-bin M64 is a projection of it: incident bins are averaged uniformly in solid angle over 8×8 cell centres, with no cosine, and outgoing directions are bin-averaged (`quad87.py:48-56, 85-110`).

**Material classes** [FACT]: B floor (Π 4.93), S1 panel (α ≈ 0.05, Π 37.95), S2 sphere (Π 16.13), Lambert walls and boxes (`scene86.py:22`; WL88:188).
- [MEAS] WL86B:106: S1 lobe r50 = 0.146 rad.
- [INF] The 50 %-energy cap is ≈ π·0.146² ≈ 0.067 sr, about 0.7 bin. The bins are anisotropic:
  - ring 0 (z ∈ [0, 0.25]): polar 0.25 rad × azimuth 0.39 rad;
  - ring 3 (z ∈ [0.75, 1]): polar 0.72 rad × azimuth ≤ 0.29 rad.
  So the lobe is not much smaller than a bin, but it is mis-shaped relative to it and moves continuously with ω_i.

## 2 Failure and Evidence Attribution (from the local angular response outward)

**A. Physical contract.** At a patch with incident radiance measure dμ_p(ω_i) = L_i cos θ_i dω_i:

- Local response: L_out(ω_o) = ∫ f_ER(ω_i, ω_o) dμ_p(ω_i).
- Transport: L = L_e + S C L, where S is local scattering (ER) and C is free-space propagation.
- ECRO's table: B ≈ Σ_{j=1..4} (SC)^j q, with q = S e.

The bin pipeline replaces S with three operations:
1. μ_p → s[b], an incident projection (L1, L2);
2. M64 = P_out f̄_in, an incident-cell average plus an outgoing-bin average (L12, L4);
3. a cᵀ ≈ M64, a fixed-basis rank-16 factorisation (L3).

**B. Why a fixed-basis factorisation is the wrong family for sharp lobes.**
- [INF] A GGX lobe is, to first order, a reflection map composed with a narrow kernel, g(∠(ω_o, R_n ω_i)). In any fixed incident × outgoing basis that is a near-permutation. Its numerical rank grows with the number of distinguishable lobe positions, about 2π / 0.067 sr ≈ 90 at the 50 % cap, not with the number of lobe parameters.
- [MEAS] This matches the measured S1 flux rank: 46–59 for 99 % energy even inside the 64-bin projection (WL92 local.json).
- [MEAS] It matches the matched-norm non-negative ladder (auditB Table 1 and ladder):
  - rank 16: val field L1 ≥ 0.36, peak ≤ 0.49;
  - rank 32: val field L1 0.146, peak 0.758;
  - floor B is exact at rank 8–16 (field 0.012–0.017).
- [INF] Rank is therefore the binding quantity only for the peaked materials. It is binding by structure, not by training.

**C. Learned versus bound, in matched norms** [MEAS] (auditB):
- Learned WL94D S1 operators score 0.961–1.044 in the operator L1, where the zero operator scores exactly 1.000. Admissible non-negative rank 16 reaches 0.380 (L1-targeted) to 0.603 (NMF16).
- On held-out panel incident fields, learned field L1 is 0.846–0.892 against 0.36–0.54 for non-negative rank 16.
- So the deployed learned local operator is far from even its own rank ceiling.
- WL93D: S_geo < 1 % for S1. The local error is geometry-common, so the local operator is a material function.
- WL94D T: direct oracle supervision moved energy (0.87→0.99) but not peak (≈ 0.3).
- [INF] Both an optimisation/parameterisation failure (L3) and a structural rank ceiling are present. Neither can be removed by more training of a cᵀ alone.

**D. Even an exact local operator loses the lobe if its output is a bin table.**
- [MEAS] WL90 (GM1, exact q, pooled relation): the explicit-vs-oracle gap is the angular projection of the once-scattered sender in the first gather.
  - Isolated share 97–102 %; spatial share ≈ 0; orders ≥ 2 add ≤ 0.001.
  - 4 sub-bins (256 directions) leave 9–20 % of J; 16 sub-bins (1,024 directions) leave about 0.
- [MEAS] In ECRO, substituting the exact M64 sender lowers the integrated error (C2_srcS1, untextured CK2: 0.047→0.031 / 0.042→0.019; WL93D C2_src E2 T1: 0.018–0.024). It does not reach the oracle (0.0137 on E2 C).
- The CK2 "exact-sender flight floor" (ratio 1.38 / 1.21, panel-face corr −0.33 over 9 patches) is confounded by L1, L2, L4, L5, L6 and L12, and by the `an94.W` 8×8 weighting defect. **Its decomposition is OPEN** (auditA §3.3, C1–C10).
  - [INF] CK2 is grazing: panel incident cos 0.024–0.164 (ring 0 only), and receiver cos 0.22–0.31, which straddles the ring-0/1 boundary.
  - In this regime both the incident-cell average (L12, Fresnel and masking vary strongly over z ∈ [0.016, 0.234]) and the outgoing-bin average (L4) are at their worst.

**E. Propagation sampling is a second, separate floor.**
- [MEAS] WL87/88: sender-side placement at equal rays removes about 90 % of the E2 C deficit (C3 0.024 vs EXP 0.049; C3UR − R0D = +0.050 / +0.053). Receiver-, area- and sender-side constructions estimate the same discrete operator (target identity within noise).
- [MEAS] The deployed 512-ray draw sits in the bad tail.
- [MEAS] Exact per-ray sender reads on receiver-side rays add variance: RAYSEND 0.059; R0X1 +0.003; WL91 C3URX1 deployed W2 draw 0.053. Pooled X1 equals the oracle, so the bias is removable but the variance fails.

**F. Conclusion of the local-angular analysis** [INF]. Sharp-lobe fidelity needs three things, each necessary:
- (i) the incident direction kept at sub-bin resolution (atoms, not bin sums) wherever the incidence is directional;
- (ii) the local response evaluated at continuous outgoing directions, or tabulated at roughly α-matched resolution (≈ 1,024 directions per hemisphere for S1 per WL90);
- (iii) a propagation quadrature placed on the sender side with angular density matched to the lobe, so that (ii) is not destroyed by point-sampling variance.

A better learned factorisation of a cᵀ addresses none of (i)–(iii). These three requirements define the propagation contract that any local representation forces.

## 3 Independent Architecture Hypotheses

### H1: ER-owned continuous lobes with a geometry-only lobe-resolved sender relation (LRS)

**State.**
- (a) The existing 512-ray receiver state for every patch (broad transport).
- (b) **Dense sender relation** for peaked patches. "Peaked" uses WL88's material-only rule, median Π > 8, fixed before WL95, so S1 and S2: 74 patches.
  - Origins y_j are the 16 existing source points of each patch, with their own point frames.
  - Directions are a deterministic per-frame grid {ω_d, Δω_d} of D cells per hemisphere. Default D = 1,024; ladder 256 / 1,024 / 4,096.
  - Each ray stores (hit patch r, receiver incident bin b_r, cos_r).
- (c) Per-material incident-bin-averaged fine response tables F̄_m[b, d′] = (1/ΔΩ)∫_b f_ER(ω, ω_d′) dω at 1,024 outgoing directions. These are geometry-independent.
- (d) Bin fields E1 and I for every patch.
- (e) A resolvent R for events ≥ 3, either learned R_θ or explicit J sweeps.

**Local ownership.** The frozen ER closure owns all local scattering, evaluated at continuous directions. No learned local factor exists: a cᵀ is removed.

**Operator I/O and equations.**
1. Direct atoms are kept unbinned. At y_j: (ω_lj, E_lj = I_l cos/d²·V), which the source pass already computes (`pt86.py:378-402`) before binning.
2. Sharp first scatter: L1_j(ω_d) = Σ_l f_ER(ω_lj, ω_d; frame_j) E_lj.
3. Sender-side flux deposit: Φ = L1_j(ω_d) cos θ_d Δω_d · A_p/16 → E1_N[r, b_r] += Φ / A_r. This is C3-style power deposition.
4. Every other first flight uses the existing state with exact bin M64/ER senders (EXP-like). Exclusion: the existing pass reads Bq = 0 at peaked patches for the q term, so the family (light → peaked → any) has exactly one owner.
5. E1_total = E1_N + E1_B.
6. Events ≥ 3: B3 = R[E1_total]. R is linear and non-negative, driven by the **delivered field E1, not by u** (fixes the L8 wiring). Learned R_θ uses per-term targets t3.
7. Readout:
   - non-peaked x2: unchanged bin lookup;
   - peaked x2: B_p(ω_o) from the fine table Σ_b I_p[b] F̄_m[b, d′], bilinear in the exact ω_o = x2→x1 (removes L10/L4 at the readout).

**Sharp structure.** Exact in (ω_i, ω_o) for directional (atom) incidence. Quantised only by the D-cell quadrature at the sender and the 16-point spatial quadrature.

**Diffuse and nonlocal.** Bins, which are adequate at measured rank (B, Lambert). For orders ≥ 2 at peaked patches the incident field is a bin field; WL90 measured that projection term as negligible, but only in GM1 and only at non-grazing incidence [MEAS / INF].

**Training and supervision.**
- None for the sharp path. ER is already the qualified local layer (WL84).
- R_θ, if learned, is trained on the tallied 3–5-event target given the physical E1 (OWN-style per-term target, no sum-only supervision). Its value over explicit sweeps is the open H-0 question.

**Inference-time physical computation (explicit).** It is ray tracing, stated as such:
- dense relation build: 74 × 16 × D rays (1.21 M at D = 1,024; 4.85 M at 4,096; current state 0.65 M);
- per source: 74 × 16 × D × L ER evaluations plus a scatter-add, with no new rays beyond the existing 16-point shadow rays;
- per frame: readout unchanged except a fine-table bilinear lookup.

**Memory and compute** [INF]: linear extrapolation from WL86 timings; not measured.

| Item | Estimate |
|---|---|
| Relation at D = 1,024 | ≈ 7 MB (≈ 6 B/ray) |
| F̄ tables | 0.79 MB per material |
| Peaked fine tables | 0.9 MB |
| Full build, D = 1,024 | ≈ 50 ms (state build is 26–28 ms per 0.65 M rays) |
| Full build, D = 4,096 | ≈ 200 ms |
| Per-source delivery | ~1 M closure evaluations, ms-scale on GPU |

- Worst case, every patch peaked: 1,270 × 16 × 1,024 = 20.8 M rays, ≈ 125 MB, ≈ 0.8 s build.
- D scales as α⁻² at fixed relative quadrature error. At α = 0.02, D ≈ 6× larger.
- As α → 0 (mirror) the grid cannot resolve the lobe. A per-atom exact mirror ray per source is required (declared fallback, which is per-source tracing).
- The cost-shift risk is real: the sharp family moves from learning into a geometry-only dense tracing precompute.

**Cache invalidation and update.**
- Relation: the existing rule. A ray is dirty if its origin is on a moved object or its segment crosses the old or new OBB with a 1 mm pad. Fixed grid and offsets mean selective equals full bitwise (testable as in `ecro86.py:57-78`).
- A sharp mover dirties all of its own relation rays.
- Source change: no retrace, only atoms, evaluations and deposits.
- ER-state change: F̄ and evaluations change, the relation does not.
- A material becoming peaked: build its relation.
- Non-rigid deformation of a peaked surface: all rays of the deformed patches are dirty, and point frames update exactly.
- R_θ: recomputed per edit, as the generator is now.

**Conservation and reciprocity.**
- ER: furnace 6e-7, reciprocity 1.2e-5 (WL87 Stage 0) [MEAS].
- Flux deposition is linear, non-negative and exactly superposing in atoms. Multiple lights in one bin keep their exact directions; the binned path does not.
- Passivity holds up to grid-quadrature albedo error, reported and not renormalised. The 512-ray S1 quadrature albedo error was median 5.3 % and max 18 % [MEAS WL87]. At D = 1,024 the error must be measured.
- Sender-side and receiver-side estimators target the same C (WL88 target identity) [MEAS].
- The source-independent rule holds: geometry plus ER only. Direct and indirect incidence both go through the same f_ER; atoms versus bins is a quadrature choice of the incident measure, not a rule branch.
  - [INF] An owner may still read this as a direct/indirect branch. Declare it.

**Why it addresses the failure (L-points).**
- L3 removed (no learned local factor).
- L1 and L12 removed for directional incidence (atoms, exact z: critical at CK2 grazing).
- L4 angular reduced to D-grid quadrature. L4 spatial and L2 reduced to 16-point quadrature, tied to the source points.
- L5 and L6 replaced on the sharp family by deterministic sender-side placement with exact receiver cos.
- L8 rewired: the resolvent is driven by the delivered E1.
- L10 removed at peaked x2.
- L11: point frames on the sphere.
- L7 remains for non-Lambert receivers of directional deposits (bin at the receiver).

**Evidence.**
- Supports: WL90 (angular, first gather, 16 sub-bins ≈ exact); WL87/88 (sender-side placement, target identity); WL91 (pooled exact sender equals the oracle); WL93C/93D/94/94D (exact M64 sender is the best render arm, ORA edit slope 0.995–1.0); Stage B (rank ceiling; learned at the zero level); WL86A Amendment 2 (relation carries 74–84 % against learned 1–18 %).
- Weakens: WL91 variance and quality-matched utility failure for exact per-ray reads. H1 claims a deterministic dense grid instead [HYP]. WL86B's H-P and LPTG critiques: the sharp part is physical, "nothing global learned". WL87: explicit ≥ learned everywhere, so R_θ's utility is unproven.
- Inconclusive: CK2 decomposition (open); grazing regime; S1→S1 chains (WL88 workload unqualified); sharp movers (none tested); spatial resolution under the 307× CK2 irradiance range.

**Main risks.**
1. The method's sharp fidelity is physical, not learned. If the owner's objective requires *learned* sharp transport, H1 does not meet it.
2. α⁻² memory/precompute and the mirror fallback.
3. Bins at order ≥ 2 may fail for sharp→sharp and grazing chains.
4. The readout's incident-bin projection at peaked x2 for indirect incidence.
5. The class rule (Π > 8) is a material-only threshold. Its insensitivity must be shown by a Π-threshold sweep, not assumed.

### H2: learned mirror-equivariant lobe kernel (MEK) inside the ECRO-lineage operator

**State.** As H1 (atoms, dense sender relation, bins), but the local operator is learned:

k_θ(ω_i, ω_o | z_m) = λ_0 cos/π + Σ_{j=1..J} A_j,θ(θ_i, z_m) · Ψ_j(Δ_j(ω_o; T_j ω_i), Σ_j,θ(θ_i, z_m))

where:
- T_j is the exact mirror or refraction map (geometric, not learned);
- Ψ_j is a fixed normalised non-negative anisotropic profile;
- A and Σ are softplus smooth functions of incidence elevation and the 69-d material code;
- isotropy about n is built in (azimuthal equivariance).

It is linear in the incident measure, non-negative, and has no fixed-basis rank, because the near-permutation structure is carried by T_j.

**Ownership.** The learned rule owns local response, but trained *directly and alone* against noise-free ER closure samples: per-term, no sum-only supervision, no co-adaptation with G. Propagation, deposits and resolvent are as in H1.

**Why it is distinct.** It keeps a learned local layer and a single learned-operator lineage, in case the owner requires one. It also extends to materials without a closed form, and possibly to patch-aggregated effective lobes for curved or large patches (L11, L2).

**Training.** Unlimited exact samples, so fitting is cheap. A smoke needs no full-scale training.

**Compute and memory.** ≈ J × 3 × 16 knots × 3 channels ≈ 600 floats per material at J = 4. Evaluation cost is of the order of the ER closure. Propagation cost is identical to H1.

**Invalidation.** As H1. A material change re-evaluates A and Σ.

**Why it addresses the failure.** It removes the L3 rank ceiling structurally (equivariance in place of rank), and it removes co-adaptation (L3/L8) by separate exact targets [HYP].

**Evidence.**
- Supports: Stage B rank structure [INF]; S_geo < 1 % (a material-only kernel suffices).
- Weakens:
  - WL94D T: direct supervision did not recover the peak. That was a different parameterisation, so the result is relevant but not decisive.
  - WL93B/C/D: learned local operators drift under joint training.
  - In this scope the target is a known closed form, so learning adds approximation error with no information gain [INF].
- Inconclusive: grazing Fresnel and masking steepness; refracted inner lobes; anisotropy.

**Risks.** Dominated by H1 on fidelity within frozen-ER scope. Its advantages (a learned single framework; non-analytic materials) are outside the measured scope.

### Alternatives considered and rejected

1. **Higher rank (32–48) or more bins for a cᵀ.**
   - Rank 32 reaches field L1 0.146 only *within* the 64-bin space, whose projection is itself the dominant explicit floor (WL90).
   - Uniform refinement multiplies every patch by α⁻², and the learned rank grows with it.
   - It does not touch L1, L5 or L12.
   - The brief bars "more bins always solves it".
2. **Incident first-moment / centroid expansion** (E_b, E_b·ω̄_b).
   - A Taylor expansion over a bin width ≥ the lobe width is invalid.
   - Placing kernels at the ratio-centroid is non-linear and breaks superposition when two sources share a bin.
3. **Exact per-ray sender reads on the existing receiver-side 8-rays-per-bin state (X1/RAYSEND).** Measured variance failure (WL87, WL90, WL91).
4. **Learned beam/lobe propagation (learned C for lobes).** WL52 failed exactly at contact and visibility transitions, and on cost. Narrow beams are dominated by visibility.

## 4 Physical and Computational Trade-offs

| Criterion | Current ECRO | H1 LRS | H2 MEK |
|---|---|---|---|
| 1 Angular representability | Rank-16 tied, ≈ zero level [MEAS] | Exact closure; grid quadrature D | Equivariant fit; error unmeasured [HYP] |
| 2 Propagation fidelity | 8 receiver rays per bin, bad-tail draw [MEAS] | Deterministic sender-side, α-matched [HYP] | Same as H1 |
| 3 Material generalisation | Compressed albedo response [MEAS CK2] | Exact for any ER state [FACT] | Learned, needs a held-out-material test |
| 4 Dynamic geometry reuse | Selective state; full generator | Selective relation, bit-equal; rule fixed | Same |
| 5 Physical correctness | No passivity constraint | ER-passive; quadrature error disclosed | Passivity must be checked per θ_i |
| 6 Training feasibility | Unstable, seed-dependent [MEAS] | None for sharp; R_θ per-term | Cheap closure fit |
| 7 Inference/update complexity | 76–78 ms per edit [MEAS] | +≈ 25–175 ms build (D 1,024–4,096), selective after [INF] | Same |
| 8 Memory scalability | 43 MB factors, O(N²) Ft | +O(N_peaked·16·D), D ∝ α⁻² | Same plus tiny |
| 9 Debuggability | Opaque co-adaptation | Each term physically tallied | Separable local fit |
| 10 Usefulness vs full recompute | 9× is LOW + table structure, not learning [MEAS WL86B/87] | Learned utility only in R_θ (unproven vs explicit) | Same |

**Initial versus edit cost** [INF]:
- H1 and H2 add a geometry-only precompute (dense relation) and a per-source evaluation. An edit retraces dirty rays, and a moved sharp object retraces all of its own rays.
- Worst case: a scene that is mostly peaked turns the relation into a ~20 M-ray precompute. That is several path-tracing spp of rays, though camera-independent.

**Cost-shift statement.** Both hypotheses deliberately put sharp first-scatter transport into explicit ray-traced relations. They do **not** demonstrate learned sharp transport.

## 5 Recommended Hypothesis

**Primary for testing: H1 (LRS).** This is not a selection.

**Mechanism.** Exact continuous ER at atom-resolved directional incidence, with sender-side deterministic α-matched relation deposition for peaked senders, and a resolvent driven by the delivered E1.

**Minimum evidence before any adoption.**
- (a) CK2 and E2 floor decomposition showing that sender representation (bin vs continuous) and propagation placement each matter, in a matched quantity.
- (b) D ≤ 1,024 and P_s ≤ 16 reach the converged-propagation error.
- (c) Cost below path-tracing the same family to the same error.

**Baselines.** C0 (deployed own), C2_srcS1/M64, EXP (H-0 null), C3UR pooled, ORA, REF-ER.

**Smallest workload.** CK2 untextured, E2 T0/T1 region C, E5, plus one prequalified non-grazing sharp case.

**Smallest implementation boundary.** An offline evaluator over the frozen scene and state, using `DrClosure83`, `xport87.qtable` and the Mitsuba scene. No production, checkpoint or ECRO code changes.

**H2** is retained only as the learned-local alternative, to be discriminated in the same smoke by a closure-fit arm.

## 6 Falsification Experiment (future bounded smoke; nothing run)

**Matched quantities.**
- **Q_s (sender):** panel once-scattered outgoing radiance on a common 4,096-direction grid against the exact closure at atoms. Bin senders are evaluated as their piecewise-constant extension. Report flux-weighted L1, energy and peak, using the **true 4×16 W**, with `an94.W` alongside only as a bridge.
- **Q_d (delivery):** per-receiver-patch irradiance (Lambert exitance) of the light → peaked → receiver family against the path-traced e2_p (2 × 4,096 spp halves). Box A is masked correctly; not the 1130:1238 slice.
- **Q_r (integrated):** E_H against REF-ER on CK2 ROI_T and E2 C.

**Oracle hierarchy.** Each level is well defined; no new oracle capability is invented.

| Level | Sender | Propagation | Isolates |
|---|---|---|---|
| O0 | learned Bq (own) | existing 512-ray deployed | current baseline |
| O1 | M64(s) | existing | the existing C2_srcS1 / CK2 floor |
| O1q | exact q (patch × bin) | existing | O1 → O1q: incident projection (L1, L2, L12) |
| O2r | exact continuous at atoms | existing rays (X1-like) | receiver-side read variance |
| O2(D, P_s) | exact continuous | H1 dense relation; D ∈ {64, 256, 1,024, 4,096}, P_s ∈ {1, 4, 16, 64} | the candidate ladder |
| O3 | exact continuous | D = 16,384, P_s = 256, plus pooled receiver relation | converged propagation |
| O3q | exact q | O3 | O3 → O3q: outgoing-sender projection at converged propagation |
| O4 | — | — | e2_p / REF-ER physical reference |

- H2 arm: MEK fits with J ∈ {1, 2, 4}, fitted on training-split closure samples and evaluated at O3.
- Capacity bounds: NMF16 and NMF32 bin senders at O3.

**Controls.**
- Lambert sender: all sender arms must coincide.
- Floor B: bins expected adequate, O3q ≈ O3.
- S2 sphere: frames.
- Incident distribution:
  - single point light;
  - two lights inside one incident bin (superposition: continuous exact, binned not);
  - LSPOT;
  - LMIX.
- Grazing (CK2) versus non-grazing.
- D and P_s ladders.
- Half-split reference noise; effects must exceed 2× the noise bound.
- Report rays, evaluations, ms (uncontended) and MB per rung, at both full and selective update.

**Acceptance (H1 survives).**
- (a) O3 vs O4 within 2× half-noise rel L1 on box A and the panel face, with flux ratio within 0.95–1.05. If not, **STOP**: contract or reference defect.
- (b) O2(1,024, 16) ≤ 1.5× the O3 error on Q_d.
- (c) O3q − O3 exceeds 2× noise, i.e. continuous sender representation matters beyond bin q.
- (d) Q_r with O2-delivered L2 improves on C2_srcS1 by more than the tolerance on both seeds' unchanged remaining terms.
- (e) Build plus delivery cost ≤ path-tracing the e2_p family to the same Q_d error.

**Falsification.**
- If O3q ≈ O3 and O1q → O3q closes the floor, the lever is propagation, not local representation. H1's local premise is falsified, and a bin-sender EXP/C3-type fix suffices.
- If (b) needs D·P_s beyond 4,096 × 64, or (e) fails, H1 is falsified on cost: it merely shifts the work into tracing.

**H2 falsified if:**
- the best J ≤ 4 fit has Q_d error > 1.5× O3, or held-out field L1 > 0.10 at peak < 0.8; or
- the albedo exceeds 1 at any θ_i.

**Prequalification.** The new non-grazing case must pass SNR ≥ 4× before the freeze (lesson from WL92A).

**Decision enabled.** Attribute the residual to local sender representation (O3q vs O3), propagation and flight (O1q vs O3q; O2 ladder), or integration (Q_r), in matched quantities. This also decides whether H1 or H2 is viable at a cost below path-traced recompute.

## 7 Unresolved Questions

1. **Learned sharp transport as an objective.** Is physically traced sharp first-scatter acceptable inside a "learned transport" method? This is an owner decision; the evidence here argues that the *local* operator should not be re-learned in fixed bins.
2. **Learned resolvent vs the null.** Does learned R_θ beat explicit sweeps (H-0)?
3. **Bin adequacy beyond GM1.** Are bins adequate at orders ≥ 2 for grazing and S1→S1 chains? WL90 is GM1 only.
4. **Untested families and incidence.** Sharp movers; the mirror limit; area lights (spread atoms).
5. **Spatial resolution.** What P_s does the CK2 307× irradiance range need?
6. **Metric defects.** The `an94.W` layout defect affects all WL94-series non-Lambert metrics.
7. **Textured variants.** `field_src` uses SMAT, not PMAT (unverified for textured variants).
8. **K/C placement.** ECRO's event-count partition is still unplaced against the K/C contract.
9. **Baseline gate.** The Neural Full Recompute and mandatory baseline gate is not addressed; no comparative claim is made.

## 8 Architecture Verdict

**MULTIPLE_CANDIDATES_REQUIRE_DISCRIMINATION.**
- H1 is the primary candidate for a learning-free discriminating smoke.
- H2 is a conditional learned-local alternative, dominated within frozen-ER scope.
- Both must also be discriminated against the existing nulls (EXP/H-0, C3UR) and the current ECRO.

The CK2 floor decomposition is open, so no single design is justified yet.

## 9 Reproducibility / Sources

**Evidence files read:**
- `C:\Users\dna10\AppData\Local\Temp\claude\462fa2b2dbb71cdd\46d65e15-e083-4057-875f-9282fb68aa5e\scratchpad\wl95\auditA.md`
- `C:\Users\dna10\AppData\Local\Temp\claude\462fa2b2dbb71cdd\46d65e15-e083-4057-875f-9282fb68aa5e\scratchpad\wl95\auditB.md`
- `C:\Users\dna10\AppData\Local\Temp\claude\462fa2b2dbb71cdd\46d65e15-e083-4057-875f-9282fb68aa5e\scratchpad\wl95\read_code.md`
- `C:\Users\dna10\AppData\Local\Temp\claude\462fa2b2dbb71cdd\46d65e15-e083-4057-875f-9282fb68aa5e\scratchpad\wl95\read_canon.md`
- `C:\Users\dna10\AppData\Local\Temp\claude\462fa2b2dbb71cdd\46d65e15-e083-4057-875f-9282fb68aa5e\scratchpad\wl95\read_wl86_88.md`
- `C:\Users\dna10\AppData\Local\Temp\claude\462fa2b2dbb71cdd\46d65e15-e083-4057-875f-9282fb68aa5e\scratchpad\wl95\read_wl89_92.md`
- `C:\Users\dna10\AppData\Local\Temp\claude\462fa2b2dbb71cdd\46d65e15-e083-4057-875f-9282fb68aa5e\scratchpad\wl95\read_wl93.md`
- `C:\Users\dna10\AppData\Local\Temp\claude\462fa2b2dbb71cdd\46d65e15-e083-4057-875f-9282fb68aa5e\scratchpad\wl95\read_wl94.md`

**Code verified (repository files):**
- `c:\Projects\ReST-GI\experiments\ecro86\model86.py:109-124`
- `c:\Projects\ReST-GI\experiments\gm1_explicit87\quad87.py:40-110`
- `c:\Projects\ReST-GI\experiments\fnto_compiled83\closure83_dr.py:1-15, 121, 164`
- `c:\Projects\ReST-GI\experiments\ecro86\scene86.py:22-23`
- `c:\Projects\ReST-GI\experiments\ecro86\protocol\ecro86.json:4-11, 56`

**Worklogs cited:** `c:\Projects\ReST-GI\docs\worklogs\86B_…:106`; `90_…:59, 201`; `88_…:111-115, 188`.

**Derived numbers:** bin-ring extents, lobe-cap solid angle, and ray, memory and time estimates are [INF] arithmetic from these sources. They are not measurements.