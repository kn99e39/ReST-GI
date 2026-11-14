# WL95 independent design report: #state-reuse designer

Design only. I edited no repository file, trained nothing and ran no GPU work. I re-read the cited code lines myself (`experiments/ecro86/model86.py`, `ecro86.py`, `pt86.py`, `experiments/gm1_c3urx1_91/cand91.py`, `experiments/gm1_relation89/rel89.py`, `experiments/ecro_mat94/an94.py`, `experiments/ecro_s94d/s94.py`). Every other number comes from the Stage A/B audits and reader digests listed in §9.

Labels: [FACT] verified in code or document · [MEAS] measured value with its source · [INF] reasoned inference · [HYP] untested.

---

## 1 Verified Current Architecture

**Deployed checkpoints.** The deployed operator is ECRO v0 with the WL92A OWN checkpoints (own_s8601/8602).

**Linear-operator form.** The table is `B = a·(c·E1 + P·Σ_n area_n·Q_n·u)`, with `u = c·s`. [FACT `experiments/ecro86/model86.py:109-124`]
- `E1[n,m] = Σ_{κ<8} 1[hit≥0]·Bq[hit,dbin]·rw_κ`, with `Bq = a·u` (`model86.py:116-119`).
- `rw` is a fixed buffer `z_k·ΔΩ/κ`, drawn from rng 86001 and shared by all patches (`model86.py:40-46`).

**State.**
- 1,270 patches × 64 equal-solid-angle bins (4 cos-rings × 16 sectors). [FACT `pt86.py:153-156`]
- 512 receiver-anchored state rays per patch, 8 per incident bin. They use one fixed (origin, direction) pattern for all patches and store only `(hit patch, dbin)`. [FACT `pt86.py:353-376`]
- Source encoding s = patch-mean direct irradiance per incident bin over 16 fixed points. The incident direction inside the bin is discarded. [FACT `pt86.py:378-402`]

**Generator.** `factors(gs)` runs 3 rounds of message passing over the dense N×N form-factor matrix built from the state rays. It emits per-patch, per-bin non-negative `c`, `a` (R = 16), `Q` and `P` (ρ = 24). [FACT `model86.py:83-107`]

**Readout.** `hl = thr(x1)·B[nearest patch(x2), bin(x2→x1)]`, with no interpolation. [FACT `pt86.py:228-233`] B holds 2–5-event radiance, so hl covers events 3–6. LOW (≤ 2 events) is path-traced.

**Edit path.**
1. `set_geom`.
2. Selective re-trace of dirty rays (segment/OBB test), which is bit-equal to a full rebuild. [FACT `ecro86.py:57-78`]
3. Full source recompute.
4. Full `model.factors(gs)` recompute. Nothing in the learned factors is selective. [FACT `ecro86.py:82-87`]

Measured edit costs: 76–78 ms per edit; selective 21 ms vs full 26–28 ms for the state. [MEAS WL86:214-215]

**Ownership as practised.**
- ER owns path-traced events; geometry owns visibility; the learned operator owns ≥ 3-event paths.
- Inside the operator, a geometry-conditioned per-patch local operator `a cᵀ` is used twice (sender and receiver, tied), plus a scene-global rank-24 term driven by direct `u`, not by E1. [FACT]
- The partition is by event count, not by the K/C segment. [FACT `docs/Architecture.md:13-38`; WL86B:48]

## 2 Failure and Evidence Attribution

**Integrated failure.**
- E2 T1 region C E_H 0.0717 / 0.0634 against gate 0.06; the table oracle gives 0.0137. [MEAS WL86]
- CK2 ROI_T E_H: C0 0.047 / 0.042. With the M64 sender substituted at S1 (C2_srcS1) it is 0.031 / 0.019. Noise is 0.0084. [MEAS WL94:155-160]

**Attribution by loss point (auditA L1–L12).** Every comparison below is within one norm.

1. **L3, local operator: the dominant learned defect.** [MEAS auditB, WL94D operator flux-L1]
   - Learned S1 operators score 0.96–1.04 in that metric. The zero operator scores 1.000.
   - Admissible non-negative rank 16 reaches 0.380 (L1-targeted) or 0.603 (NMF16).
   - On the held-out panel fields (field rel-L1 with W94): learned 0.846–0.892 against rank-16 non-negative 0.36–0.54.
   - The "0.46–0.52 Eckart–Young floor" quoted in WL94D is a Frobenius number. Comparing it with an L1 ≈ 1.0 is unmatched (auditB §1c).
2. **The rank-16 cap is real but secondary.** [MEAS auditB]
   - No rank-16 operator tried gets below about 0.36 field L1 or above a field peak of about 0.49 at energy ≈ 1.
   - Field L1 ≤ 0.15 with peak ≥ 0.7 needs rank about 32.
   - Floor B is exact at rank 8–16; S2 sits in between.
3. **Geometry conditioning of the local operator carries almost no information.** [MEAS WL93D] S1 S_geo is 0.2–0.8 %; swapping operators between geometries (C1 swaps) changes nothing measurable. The S1 error is geometry-common (E_common 0.79–0.81).
4. **L4, first-gather outgoing (angular) projection.** In GM1 it explains 97–102 % of the explicit-vs-oracle gap. Spatial S ≈ 0; orders ≥ 2 contribute ≤ 0.0009; compounding is ≤ 9 %. [MEAS WL90:162-184]
   - Pooled first-gather-exact (X1) equals the oracle in all six WL91 conditions. [MEAS WL91:160-179]
   - Its single-draw variance fails held-out validity in 2 of 3 conditions and the quality-matched utility rule (3.1× slower). [MEAS WL91:151-156, 310-317]
5. **L5, receiver-side relation sampling.**
   - The deployed draw sits in the bad tail. [MEAS WL87:245-262]
   - Sender-side placement at equal rays is the dominant factor (+0.050 / +0.053). [MEAS WL88 Task A]
   - Receiver-anchored exact reads (R0X1) gave no E_H gain. [MEAS WL90]
6. **L8, compensation.**
   - The global term adds a geometry-blind +0.02…+0.06. [MEAS WL86A Amendment 1]
   - Co-adaptation: a single physical substitution moves table and render metrics in opposite directions. [MEAS WL93D]
7. **WL94D CK2 "exact-sender flight floor" (1.378 / 1.211 / −0.334): confounded.**
   - M64(s) carries L1, L2, L12 and L4. Rays are κ = 8 with a fixed pattern. The panel is lit at grazing incidence (cos θ 0.024–0.164; ring 0; 307× irradiance range).
   - The footprint covers 3 of the 9 panel-face patches. The receiver z of 0.22–0.31 straddles the ring-0/1 boundary.
   - Its decomposition is **OPEN**. [FACT/INF auditA §3]
   - G_PHYS on the training items gives a pooled ratio of 0.953. [MEAS] So no generic floor is shown.
8. **Metric defects to fix before any smoke.**
   - `an94.W` uses an 8 × 8 ring/sector layout (`experiments/ecro_mat94/an94.py:19,32`). [FACT]
   - `delivery_box` covers boxes A + B (`s94.py:309`). [FACT]
   - The panel-face correlation is over 9 patches. [FACT]

**Synthesis [INF].** The failing family is the once-scattered radiance from a sharp sender under a delta (point-light) incidence. ECRO stores it in two lossy places:
- as an outgoing patch × bin table, which needs angular resolution ∝ α⁻² per sender;
- inside a learned geometry-conditioned local operator, which has not even reached the zero-operator level.

Its error is then partly cancelled by a geometry-blind term. None of the three is required by physics. Two invariant factors of the true quantity are available at full resolution: the material ER, and exact-direction visibility records.

## 3 Independent Architecture Hypotheses

### 3.0 Dynamic-reuse contract (derivation basis)

Transport: `L = L_e + S∘G L`.
- `(G L)(x,ω) = L(r_g(x,ω), −ω)` is propagation. It depends on the geometry g.
- `(S L_in)(x,ωo) = ∫ f_{m(x)}(ωi,ωo) L_in(x,ωi) cosθi dωi` is local scattering. f is ER in the local frame.
- Once-scattered radiance under point lights: `q(y,v) = Σ_l f_m(ω_l(y), v)·E_l(y)`, with `E_l = I_l cosθ_l/d²·V_g(y,p_l)`.

**Invariance classes** [FACT for the definitions; INF for the consequences]:

| Object | Depends on | Survives | Invalidated by |
|---|---|---|---|
| f_m (ER) | material only | every geometry edit (rigid, deformation, contact) given the correct local frame | material change |
| local frame (t, b, n)(y) | own surface | edits of other objects | motion or deformation of its own surface |
| relation records R = {(y_j, v_k, n_hit, x_hit, ω_rcv)} | g | edits whose volume no segment crosses | exact segment-volume test (template `ecro86.py:57-78`) |
| singular incident records D = {(y_j, l, ω_l, E_l)} | g, e | edits no shadow segment crosses; unchanged lights | change of light l; moved volume crosses a shadow segment; y_j's surface moves |
| regular incident field (bins) of orders ≥ 2 | g, e, all of L_k | nothing (the resolvent has no edit locality; WL86B (ii)) | any change, by recompute |
| B table, gathers | all | nothing | any change |

**Design principle derived from the table [INF].**
- A sharp outgoing field `q(y,·)` at a glossy sender is the product of an invariant sharp factor (f_m) and a source × geometry-dependent singular factor (ω_l, E_l).
- Caching that product as an outgoing table has three costs:
  - (a) it must be invalidated by every light or geometry change;
  - (b) its resolution must scale with the lobe width;
  - (c) it bakes the projection error that WL90 measured.
- Caching the two factors separately and forming the product at the exact query direction (deferred evaluation) has three properties:
  - the sharp factor is never invalidated;
  - the source factor is a short list of deltas;
  - the geometry factor is exactly and selectively updatable.
- A geometry-conditioned learned local operator adds invalidation coupling: all factors are recomputed per edit through the N² generator (`ecro86.py:82-87`). It does so without measured information gain (S_geo < 1 %).

The two hypotheses below follow this principle. They differ in how the deferred product is integrated into receivers: sampling (H1) versus analytic integration (H2).

---

### H1 — DSR: Deferred-Sender Relation (material-exact sender, sender-anchored exact-direction relation)

**State representation.**
- *Invariant:* f_m (frozen learned ER, WL84, evaluable at continuous (ωi, ωo)). Plus the M64-type material tables `f̄_m(bin_i → v)`: incident-bin-averaged, with a fine or continuous outgoing direction. These serve the regular (binned) incident of later orders.
- *Geometry:*
  - P sample points y_j per patch, stratified and fixed in patch parameter space.
  - From each y_j, K_j sender-anchored relation rays. They are cosine-stratified per outgoing bin, as in C3UR (`rel89.py:53-73`), but anchored at y_j rather than at a fresh point per ray.
  - Each ray record holds (j, exact v_k as 2 × fp16, hit patch n, receiver incident bin m, receiver cosine). That is about 5 B per ray more than today's 27 B.
  - Optional material-only densification for peaked senders (WL88's prospective rule Π > 8: S1, S2). This is a sampling density, not an ownership class.
- *Source:* D, the singular incident records at the same y_j. They come from the existing source pass (N·P shadow rays per light), keeping ω_l instead of binning it.
- *Derived:* the regular incident bins per patch for orders ≥ 2, and the B table.

**Operator, inputs to outputs.**
- First gather (deposition estimator, sender-anchored):
  `Ê(n,m) = (1/A_n) Σ_p (A_p/P) Σ_{j∈p} (1/K_j) Σ_k q(y_j,v_k)·cosθ_{y_jk}/p_j(v_k)·1[hit_jk = n, bin_n(−v_k) = m]`,
  where `q(y_j,v_k) = Σ_l f_m(ω_l(y_j), v_k)·E_l(y_j)` is evaluated exactly per ray.
- [FACT, radiometry] `cosθ_y dA_y dω_y = cosθ_x dA_x dω_x`. So Ê is unbiased for the patch-mean bin irradiance `E(n,m) = (1/A_n)∫_{A_n}∫_{Ω_m} L_1(r(x,ω),−ω) cosθ_x dω dx`.
  - [MEAS WL88:252-299] Receiver-, area- and sender-anchored constructions estimate the same operator (R1_INF − R0_INF −0.009; C3U_INF − R0_INF −0.001; noise 0.013).
- Receiver response: `L_2(n,d) = Σ_m Ê(n,m)·M64_{m(n)}[m,d]`. This is exact ρ/π for Lambert receivers.
- Orders ≥ 2:
  - `L_{k+1}` uses the same relation, with sender outgoing evaluated deferred at v_k from the sender's regular incident bins: `Σ_i E_i(y)·f̄_m(bin_i → v_k)`.
  - The outgoing side is therefore never tabulated for transport; only incident binning remains.
  - B = Σ L_k feeds the unchanged readout.
- Output: the same B [N, 64, 3] contract as ECRO, so LOW, readout and regions are unchanged.

**Local scattering ownership.** ER exclusively, for every event and every material. There is no learned per-patch local operator and no tie between sender and receiver.

**Geometry-dependent propagation ownership.** The explicit relation (visibility and exact directions) exclusively. In K/C terms: K = ER on surfaces, C = relation in free space, with exactly one owner per segment [INF; consistent with `docs/Architecture.md:19-25`].

**Sharp angular structure.**
- Lives only in f_m (invariant) and in the exact v_k.
- For singular incidence: no incident projection (removes L1 and L12), no outgoing projection (removes L4), true sample-point frame (removes L11 for the first gather), and each ray's own cosine (removes L6).
- What remains is MC sampling of the lobe by source-independent rays. With cosine-stratified κ_s rays per bin, a lobe of about 0.07 sr (≈ 0.7 bin) receives O(κ_s) rays per sample point [INF].

**Diffuse and nonlocal transport.**
- Lambert senders reduce to ρ/π·E. Deferral is exact and costs nothing, which makes Lambert a built-in positive control.
- Orders ≥ 2 use binned incident with deferred outgoing. WL90 P2plus ≤ 0.0009 supports this in GM1 [MEAS]. Sharp-to-sharp chains (C_ss) are untested.
- No learned global term. A learned tail for orders ≥ 3 would be a separate hypothesis, needing a separately tallied target and a win over explicit sweeps. It is not needed for the identified failure.

**Reuse after edits.**
- f_m: always reused (zero updated learned parameters).
- R: dirty-segment re-trace, bit-exact against a full rebuild (the existing contract).
- D: re-shadow only the records whose shadow segment crosses a moved volume, or whose y_j moved.
- Frames: recomputed for moved or deformed surfaces only.
- Orders and B: recomputed globally (J sweeps).
- Deformation is admissible without new validity certificates, because a surface BRDF has no interior state [INF].

**Training targets and supervision semantics.** None beyond ER, which already exists, is frozen and was qualified separately (WL83/84). The once-scattered quantity uses exact physics given ER. The only "supervision" is validation against e2_p references.

**Inference-time physical computation (explicit ray tracing).**
- Relation rays: geometry-only; cached; re-traced on the dirty fraction.
- Shadow rays: N·P·L, as today.
- ER evaluations: one per (relation ray from a non-Lambert sender, light). No probe rays, unlike X1, which used 526,467 probe intersections plus 526,467 shadow rays per light [MEAS WL91:256].
- Camera: unchanged (LOW path tracing plus one table lookup).
- **H1's propagation is classical ray tracing.** It does not claim a learned transport operator.

**Memory and compute scaling** [INF, current scene].
- Relation: N·K = 650k rays → about 21 MB.
- Peaked densification ×4 on 74 S1/S2 patches: +152k rays.
- ER evaluations per light: about 0.17 M (non-Lambert senders only) to 0.65 M.
- Initial construction: O(N·K + N_pk·P·K_d) rays. On the order of C3UR's relation build [INF; not measured].
- Per edit: O(f_dirty·(N·K)) re-trace (measured dirty fraction 8.7–10.2 %), plus O(dirty shadow), plus O(J·N·K) gathers. The gathers are not edit-local.
- Worst cases:
  - (i) Many or area lights: the ER evaluation count grows ∝ L.
  - (ii) Environment lighting: the incident becomes regular, the sender falls back to incident binning, and L12 returns for sharp senders.
  - (iii) α → 0: rays per lobe ∝ α², so a source-independent relation cannot resolve mirrors. That needs source-dependent specular continuation (ray tracing, which must be declared).
  - (iv) Large N: the sweeps are O(J·N·K), linear but global.

**Cache and invalidation rules.** As listed under "Reuse after edits". Add a selective-equals-full contract for R and D, following the existing `ecro86.py:161-168` pattern.

**Conservation and reciprocity.**
- [MEAS WL87] ER reciprocity 1.2e-5 and furnace 6e-7. S1 albedo ≤ 0.825 [MEAS auditB].
- The estimator is unbiased for the discrete quantity. No learned operator is recursed, so passivity reduces to ER, which is passive [INF].

**Why it addresses the failures.**
- It removes L3, L7 and L8 by construction: no learned local operator, no tie, no compensating term on the L_2 path.
- It removes L4, L1, L12, L6 and L11 for the singular first gather. Measured evidence of the mechanism:
  - WL90: first-gather angular projection explains 97–102 %;
  - WL91: pooled X1 equals the oracle;
  - WL93D / WL94: C2_src and C2_srcS1 improve sharp renders even with M64 projection.
- It addresses L5 by sender anchoring (WL88 dominant factor).

**Support, weakening and open findings.**
- *Supports:* WL87 (explicit ≥ ECRO, 0 wins / 3 losses), WL88, WL90, WL91 (pooled), WL93C/D (exact composition: edit slope 0.995–1.000), WL93D (S_geo < 1 %), WL94/94D (C2_srcS1), and auditB (learned local operator at zero level).
- *Weakens:*
  - WL91: X1 single-draw variance; held-out validity 1 of 3; LSPOT89 C_ss harm +0.0025.
  - WL90: R0X1 shows no gain. Exact reads need adequate sender sampling.
- *Inconclusive:* the CK2 decomposition (auditA §3); sharp-to-sharp (WL88 unqualified); held-out sharp workloads (E6 not admitted); one room family.

**Main risks.**
- (1) Variance. H1 is structurally X1 with shared-origin shadowing. Correlation across the K_j rays of one y_j may add spatial variance on CK2's 307× gradient.
- (2) Cost shifted into ray tracing and ER evaluation. H1 must beat C3UR/C3URX1 and quality-matched PT re-render, not ECRO.
- (3) Learning status. Its only learned object is material-level. Whether a learning-free global solve around ER is a candidate or a baseline is an open owner decision (`docs/plans/gm1_architecture_review/60_coordinator_synthesis.md:94-100`).
- (4) Receiver spatial structure seen through the mirror readout (L10) is not addressed.

**Smallest kill experiment.** See §6. H1 is falsified if either holds:
- On CK2 / E2 the deferred first gather does not reduce the O1→reference gap by ≥ 50 % at the deployed ray budget.
- Even at ≤ 4× peaked-sender density, the per-draw pass rate is < 2/3 of conditions.

---

### H2 — ALP: Analytic Lobe–Polygon first-gather transfer (material-fitted lobe integrals, pair visibility)

**State representation.**
- *Invariant:*
  - f_m (ER).
  - A **learned material-level lobe-integral model** per material: an LTC-type (linearly transformed cosine) or small lobe-mixture fit `F_m(ω_l, ·)`. It is normalised to the ER directional albedo ρ_m(ω_l), so that `∫_Ω f_m(ω_l,v)cosθ dv ≈ ρ_m(ω_l)·I_poly(M_m(ω_l)⁻¹ Ω)`. It is fitted offline from ER evaluations only.
- *Geometry:*
  - For every non-Lambert sender patch p: the candidate front-facing receiver patches n, with their polygons in world space.
  - A pair visibility fraction V̄_pn estimated from m geometry-only rays, with stored segments for dirty tests.
  - Lambert senders keep the relation of H1 or EXP. Both estimate the same quantity, so this is an estimator choice, not an ownership class.
- *Source:* D, as in H1.

**Operator.**
- Flux into receiver n: `Φ_{p→n} = Σ_{j∈p} (A_p/P) Σ_l E_l(y_j)·ρ_m(ω_l)·I_poly(M⁻¹ Ω_n(y_j))·V̄_pn`. Then `Ê(n,·) = Φ/A_n`.
- The receiver incident bin is assigned by the centroid direction, or split by sub-polygon. This is an approximation for glossy receivers.
- Later orders as in H1.

**Ownership.**
- Local: ER for the response, plus a learned material-invariant integral surrogate of ER (never geometry-conditioned).
- Propagation: explicit polygons and visibility.

**Sharp structure.** Integrated analytically over the receiver solid angle, so there is no lobe-sampling variance. The error moves to two places: (a) lobe-fit error, measurable offline per material; (b) the scalar-visibility approximation, which loses the correlation between occlusion and lobe direction near blockers.

**Diffuse and nonlocal.** For Lambert, I_poly is the exact point-to-polygon form factor. Orders ≥ 2 are as in H1.

**Reuse.**
- Material fits are always reused.
- Pairs: those involving moved patches, plus those whose visibility segments cross moved volumes, are re-evaluated.
- D as in H1; sweeps global.

**Training.** Offline per-material fit against dense ER evaluations. The quantity is exact, with no scene data and no co-adaptation path. A new material needs a fit but no scene retraining.

**Inference-time physical computation.**
- m visibility rays per pair (cached).
- Shadow rays N·P·L.
- P·L polygon integrals per pair. These are ALU work, not rays.

**Scaling** [INF].
- Pairs: N_g × N_vis ≈ 330 × ~1,000 ≈ 3.3·10⁵ here.
- Polygon integrals per light: × P = 16 → about 5·10⁶.
- Visibility rays: about 1.3·10⁶ at m = 4.
- **Worst case O(N_g·N), quadratic.** This is the "large explicit transport table" risk.
- Candidate generation must be conservative: a receiver missed by sampled visibility cannot be recovered (AGENTS.md selector rule).

**Conservation and reciprocity.**
- Energy per incidence is fixed by the ρ normalisation.
- LTC fits are not reciprocal. That is admissible only because the surrogate is used once, never recursed [INF].

**Why it addresses the failures.** It shares H1's removal of L1, L3, L4, L6, L7, L8 and L11. In addition it targets the variance failure measured for exact reads (WL91) by replacing sampling with integration.

**Support, weakening and open findings.**
- *Supports:* the same projection evidence as H1, plus WL91's variance mechanism.
- *Weakens:* nothing measured directly. ER (a coated conductor) may not be LTC-shaped, and S1's near-permutation structure is high-rank in fixed bins (auditB). Low rank in lobe-aligned coordinates is [HYP].
- *Inconclusive:* everything H2-specific.

**Main risks.** Fit bias at grazing incidence (CK2 is entirely ring 0); visibility bias; quadratic pair tables; glossy-receiver bin assignment.

**Smallest kill experiment.** Stage 3 of §6. H2 is falsified if either holds:
- ALP-oracle (exact ER integrated by dense quadrature, with scalar V̄) has bias above H1-POOL's by more than noise at matched visibility rays.
- The material fit gives field L1 > 0.15 or peak < 0.7 on S1, on the actual incident distribution and with the correct W.

---

### Alternatives considered and rejected

1. **Raise the rank or bin count of the per-patch learned `a cᵀ` (rank ~32 or more bins).**
   - The learned operator is at the zero-operator level, far above the rank-16 bound. Capacity is therefore not the first binding constraint [MEAS auditB].
   - The output stays outgoing-bin-projected (the WL90 J term) whatever the rank.
   - Outgoing tabulation resolution scales ∝ α⁻² per sender.
   - The brief forbids assuming that more bins solve the problem.
2. **Keep the geometry-conditioned local operator and change the supervision or untie it** (the WL93–94D recipe line).
   - Measured: NO_RELIABLE_IMPROVEMENT, SHARING_SIGNAL only at parity, co-adaptation, seed disagreement.
   - Geometry conditioning carries < 1 % information while forcing a global recompute per edit.
3. **An outgoing sparse-lobe or beam cache per sender** (DDT/LPTG-like).
   - For delta incidence it equals deferred evaluation (H1) plus an extra approximation.
   - For narrow indirect incidence its state scales like DDT's α⁻⁴.
   - Withdrawn precedents: WL84A.
4. *(Considered, not adopted)* **Learned relation proposal density.** It does not remove the projection. Its status as "learned transport" is an open owner decision.

## 4 Physical and Computational Trade-offs

| Criterion | ECRO v0 (status quo) | H1 DSR | H2 ALP |
|---|---|---|---|
| 1 Angular representability | Learned S1 operator at zero level; rank-16 cap ≥ 0.36 field L1; output bin-projected [MEAS] | Exact ER at continuous directions for singular incidence; incident binning only for orders ≥ 2 [FACT design; WL90 MEAS] | Limited by the lobe fit (unmeasured) [HYP] |
| 2 Propagation fidelity | 8 fixed rays per bin; learned flight delivers 1–18 % (E2) [MEAS] | Unbiased; variance-limited (X1 risk) [MEAS-related] | Variance-free lobe; visibility-fraction bias [HYP] |
| 3 Material generalisation | 69→16 MLP; albedo compression (WL94) [MEAS] | Any ER material, with no transport training [FACT] | Needs a per-material offline fit [FACT design] |
| 4 Dynamic reuse | Rule reuse; all factors recomputed per edit; state selective [FACT] | Material always reused; R and D selective; sweeps global | As H1, plus pair re-evaluation |
| 5 Physical correctness | Linear, non-negative, no passivity, compensation [FACT/MEAS] | Exact given ER; ER reciprocal and passive | Energy normalised; non-reciprocal surrogate used once |
| 6 Training feasibility | 1,024 geometries × 256 spp; identifiability failures | None new | Offline ER fit (cheap) |
| 7 Inference / update | Per-frame identical across table arms (WL87); edit 76–78 ms | Est. ≈ EXP/C3URX1 class (68–125 ms measured for relatives) [INF] | Plus pair visibility; unmeasured |
| 8 Memory | 43 MB factors; Ft dense N² | O(N·K) ≈ 20–40 MB; linear | O(N_g·N); quadratic worst case |
| 9 Debuggability | Untallied co-adapting terms | Every term physically defined and per-ray checkable | Two separately checkable approximations |
| 10 vs full recomputation | Not established (economics NOT TESTED) | Not established; binding bar = quality-matched PT | Not established |

**Cost shifting [INF].**
- H1 moves the sharp first gather entirely into explicit ray tracing plus ER evaluation. It must be declared and compared as explicit transport: against C3UR, C3URX1 and quality-matched PT, under the four-role baseline gate.
- H2 moves it into a pair table. Its quadratic worst case is the "large explicit table" failure mode (compare WL52 / WL80A C4).
- Neither hides ray tracing inside a learned name.
- The only measured learned advantage of ECRO is source-only update cost: 3.5–3.9× [MEAS WL87]. For H1, that update is shadow rays at the sample points plus ER evaluations plus gathers. Its relative cost is unmeasured.

**Initial construction versus per-edit update.** Both hypotheses have:
- a negligible material build (already done);
- linear relation construction (H1) or quadratic construction (H2);
- per-edit cost dominated by the global sweeps O(J·N·K), not by state re-trace. In ECRO the measured selective saving was only 5–7 ms [MEAS WL86].

**Shared scaling assumptions.** One room family, N ≈ 1.3k, point lights, α ≈ 0.05.

## 5 Recommended Hypothesis

**No single primary is selected.** H1 and H2 attack the same measured failure location (the singular first gather from sharp senders, through L4, L1/L12 and L3/L8). They differ only in the integration mechanism: sampling versus analytic. The deciding uncertainties are learning-free and measurable on frozen data:
- the CK2 decomposition (open);
- the variance of exact deferred reads at practical density (negative precedent, WL91);
- the fit error and visibility bias for H2.

H1 is the **reference arm** of the discrimination, because its oracle (pooled H1) is well defined and is the parent quantity of H2's oracle. It is not a demonstrated fix.

Two qualifications:
- If the owner rules that a design whose only learned object is material-intrinsic cannot be the method, both become explicit comparators. For a learned global method, selection is then premature.
- A learned-tail ECRO element (global term driven by E1, with target t3) is orthogonal and is not motivated by the sharp evidence.

## 6 Falsification Experiment (future bounded smoke; nothing trained)

**Preconditions (metric fixes, frozen first).**
- Correct the exitance weight to the 4 × 16 layout.
- Restrict box A to its own 54 patches.
- Report the 3 CK2 footprint patches separately.
- Give bootstrap CIs for 9-patch correlations.
- Half-split reference noise.

**Workloads (smallest valid).**
- CK2-none: one grazing light; e2_p reference at 2 × 4,096 spp; half-noise 0.058 (box A) / 0.069 (panel face).
- E2 T1 / T0 region C, with the WL92 "single + true L₃₊" ladder infrastructure.
- Non-grazing control: GM1 W1 T1 R_mirror (WL90 oracle).
- Held-out: W2 T1, LSPOT89, LMIX89. These are evaluated only, never tuned on.

**Oracle hierarchy.** One factor changes per step; all are well defined.

| Arm | Content |
|---|---|
| O0 | Learned sender + deployed propagation (C0) |
| O1 | Bin-exact sender M64(s) + deployed propagation (C2_srcS1, the WL94D floor) |
| O2 | Exact patch × bin q (`quad87` qtable, panel only) + deployed propagation. Removes L1, L2, L12; keeps L4, L5 |
| O2x | Pointwise-exact q(y, v) read by the deployed receiver-anchored rays (R0X1-like). Control: WL90 found no gain in GM1 |
| O3 | H1 at the deployed budget: sender-anchored, P = 16, 512 rays per panel patch |
| O3-POOL | 64 relation realisations. Separates sampling bias from variance |
| O3-L4 | O3 with outgoing sub-bin projection a ∈ {1, 4, 16}. Directly measures L4 on CK2 |
| O4 | Density ladder K_d ∈ {1, 4, 16}× on peaked senders × P ∈ {16, 64}; ≥ 16 draws each |
| ALP-oracle | Dense-quadrature exact-ER polygon integral with scalar V̄ (m ∈ {4, 16}) |
| ALP-fit | Offline LTC / lobe fit |
| REF | Path-traced e2_p per receiver patch. For renders: LOW + candidate L₂ + path-traced L₃₊, then + explicit sweeps |

**Matched quantities.**
- e2_p receiver exitance (correct W) on box A, panel face, footprint and all non-panel receivers: ratio, rel L1, CI'd correlation.
- Rendered E_H in region C / ROI_T. Same masks and reference per family; GM1 and ECRO numbers are never cross-compared.
- Material-level fit metrics in the auditB flux-L1 / field norms, with the correct W.
- Cost: rays (relation, shadow, visibility), ER evaluations, polygon integrals, warm uncontended ms per source update and per edit, memory.

**Controls.**
- Lambert sender: all arms must agree (positive control).
- S2 sphere: frame (L11).
- Sector-offset bin rotation by half a sector: ring/sector aliasing (L6).
- Grazing versus non-grazing incidence (ring shares reported).
- Point versus narrow-cone light.
- Equal ray budgets O1 = O3; capacity (ALP lobes 1 vs 2); reference half-noise.

**Acceptance (sketch; the owner freezes it).**
- Box A rel L1 ≤ max(2 × half-noise, 0.12); ratio 0.9–1.1; footprint correlation CI lower bound ≥ 0.8.
- Region C / ROI E_H ≤ oracle + 2 × noise.
- ≥ 90 % of draws pass in ≥ 2/3 held-out conditions.
- Per-edit cost ≤ 2× EXP.

**Stop and kill conditions.**
- If O2 ≈ REF: incident projection dominates on CK2. H1/H2's deferred outgoing loses its CK2 justification and rests on WL90 GM1 evidence alone.
- If O3-POOL does not approach REF beyond 2 × noise: the cause is not sender projection (receiver spatial structure or reference definition). Both hypotheses are killed for CK2.
- If O4 fails the pass rate at ≤ 4× peaked density: H1 is rejected at practical cost, a repeat of the X1 failure.
- If ALP-oracle bias exceeds O3-POOL beyond noise, or the ALP fit is > 0.15 L1 / < 0.7 peak: H2 is rejected.

**Decision enabled.** H1, H2 or neither. Combined with the owner ruling on learning status, this yields ONE_CANDIDATE_JUSTIFIED or ARCHITECTURE_SELECTION_PREMATURE.

**Smallest implementation boundary.** An offline evaluation module for the 42 S1 panel senders only:
- singular records at the sample points (the existing `pt86.source` geometry, keeping the direction);
- a sender-anchored relation with exact v, following the `rel89.py` template;
- ER evaluation at continuous directions (the existing ER evaluation used by `quad87` / `pt86`);
- deposition into the existing receiver bins and comparison against existing e2_p.

No production ECRO change and no checkpoint change.

## 7 Unresolved Questions

1. Owner decision: is a global solve whose only learned object is ER a candidate or a baseline?
2. CK2 decomposition (C1–C10, auditA §3).
3. Receiver spatial (sub-patch) structure seen through the mirror readout (L10). The table oracle floor is 0.0137 on E2; CK2's table oracle is unmeasured.
4. Sharp-to-sharp chains and moving sharp senders (WL88 unqualified).
5. Area or environment lighting, where incidence on sharp senders is regular and the L12 projection returns.
6. The α → 0 limit, which needs source-dependent specular continuation.
7. [HYP] Whether S1's response is low-rank in reflection-aligned coordinates. This is relevant only if a binned learned material operator is ever needed. It can be tested offline on CPU.
8. Large-N scaling: global sweeps (H1); quadratic pairs (H2).
9. The impact of the W defect on the learned and render-derived WL94-series values (unmeasured).
10. Whether ECRO's source-only update advantage (3.5–3.9×) survives against H1.

## 8 Architecture Verdict

**MULTIPLE_CANDIDATES_REQUIRE_DISCRIMINATION.**
- H1 (DSR, deferred exact sampling) and H2 (ALP, analytic lobe integration) are materially distinct. Both are derived from the dynamic-reuse contract, which says to cache the invariant material and exact visibility and never a sharp outgoing product.
- Both target the measured failure location. Neither is demonstrated.
- The discriminating evidence is learning-free.
- If the owner deems a material-only learned object insufficient for "the method", the verdict for a learned global transport architecture is ARCHITECTURE_SELECTION_PREMATURE.
- No novelty is claimed. Both are assemblies of known techniques: deferred shading, sender-side deposition, LTC.

## 9 Reproducibility / Sources

**Evidence files** (all in `C:\Users\dna10\AppData\Local\Temp\claude\462fa2b2dbb71cdd\46d65e15-e083-4057-875f-9282fb68aa5e\scratchpad\wl95\`):
- `auditA.md`
- `auditB.md`
- `read_code.md`
- `read_canon.md`
- `read_wl86_88.md`
- `read_wl89_92.md`
- `read_wl93.md`
- `read_wl94.md`

**Code I re-read** (under `c:\Projects\ReST-GI`):
- `experiments/ecro86/model86.py` (40-46, 83-124)
- `experiments/ecro86/ecro86.py` (57-87)
- `experiments/ecro86/pt86.py` (153-200, 225-233, 270-285)
- `experiments/gm1_c3urx1_91/cand91.py` (36-110)
- `experiments/gm1_relation89/rel89.py` (33-73)
- `experiments/ecro_mat94/an94.py` (19, 32)
- `experiments/ecro_s94d/s94.py` (309)

**Worklogs:** WL86–WL94D under `docs/worklogs/`, as cited inline.

**Canonical documents:** `docs/Architecture.md`, `docs/RESEARCH_ROADMAP.md`, `docs/BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md`, `docs/FNTO_ARCHITECTURE_CONTRACT_V0_1.md`.

**Estimates.** Every cost and memory estimate marked [INF] is arithmetic on the measured scene sizes (N = 1,270, K = 512, 74 S1/S2 patches, 4 lights). None was measured.