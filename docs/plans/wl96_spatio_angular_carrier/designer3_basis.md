# WL96 Designer D3: Spatio-Angular Transport Carrier Architecture Design

**Scope.** This is design only. I edited no repository file, trained nothing, and ran no GPU or renderer job. I ran CPU-only numpy calculations and wrote them only to `C:\Users\dna10\AppData\Local\Temp\claude\462fa2b2dbb71cdd\46d65e15-e083-4057-875f-9282fb68aa5e\scratchpad\wl96\d3\` (scripts and outputs are listed in §10).

**Labels.**
- [FACT]: verified in code or docs, with a path:line citation.
- [MEAS]: a measured value, with its source. **[MEAS-toy]** means this designer's CPU toy, which is scratch and not pipeline evidence.
- [INF]: an inference.
- [HYP]: an untested hypothesis.

**No novelty is claimed.** Light tracing / photon splatting for caustics, directional (glossy) radiosity tables, image sources and PRT bases are classical.

---

## 0. Bottom line

1. **[INF] The starting point cannot hold sharp transport if its basis is learned.** The starting point was a source-independent reusable transport basis with compact, cheaply re-weighted source coefficients. For near-field point lights on α ≈ 0.03–0.08 senders, the representation that keeps the sharp lobe needs all of the following:
   - ≤ 10 cm sender cells **and** ≥ 1,024 outgoing bins;
   - an exact local operator at that resolution of effective rank ~500 (99 % energy);
   - a sender-anchored or very dense propagation step.

   None of this is low-rank or learnable more cheaply than evaluating ER itself. What remains is either (i) re-learning ER, the class WL95 rejected (MEK), or (ii) learning visibility / reflected form factors, which is learned C (WL52 negative). §3 gives the evidence.
2. **[INF] Two materially distinct carriers remain viable for the sharp part. Both are physically projected and learning-free.**
   - **Candidate A, LSC-R (Lagrangian Sharp Carrier).** Light-side lobe-sampled photons for directly lit sharp senders, plus a geometry-only lobe-continuation relation for indirectly lit sharp senders. Sharp patches have no table rows.
   - **Candidate B, ESC-R (Eulerian Sharp Carrier).** A refined spatio-angular sender table on sharp facets (10 cm × 16×64 bins), with an exact material operator and a source-independent sender-anchored relation. This is the starting point realised physically: per light, only the incident encoding is recomputed; no transport rays are traced.

   In both, the **only learned component is an ECRO-lineage resolvent R_θ for ≥ 3-event transport on broad (non-sharp) patches**, driven by the exact 2-event incident field.
3. **For the sharp first order, both candidates are explicit GI under another name.** A is in the WL95 G1 family (ELCF-type estimator) plus a specified G2-type resolvent. B is a directional-radiosity-type fine table (G1 / LRS family).
   - Whether a physically projected sharp carrier is admissible inside the ECRO lineage is owner **Gate A** (`docs/Architecture.md:13`; `docs/RESEARCH_ROADMAP.md:128`).
   - Whether the learned resolvent has any utility over explicit sweeps is **unmeasured**. Prior evidence predicts accuracy utility of about 0 and cost utility only on source-only updates [MEAS WL87:281, 315, 346].
   - **If Gate A rules physically projected sharp carriers inadmissible, or if test FL1 (§8) fires, my conclusion is that no viable learned sharp-transport candidate exists on current evidence.**

---

## 1. Facts relied on

- [FACT] The ECRO local sender is M_n = a cᵀ: rank ≤ 16, non-negative, patch- and bin-constant (`experiments/ecro86/model86.py:111-124`).
  - Single flight: E1 = Σ_κ Bq[hit, dbin]·rw over 512 fixed state rays per patch (`model86.py:115-120`; `ray_weights` `model86.py:40-46`).
  - The generator does not see the light: s is an input of `apply` only (`model86.py:91-109`).
- [FACT] The source s is the patch-mean direct irradiance per incident bin from 16 fixed points (`experiments/ecro86/pt86.py:378-402`). The state rays are (hit patch, outgoing bin, segment) (`pt86.py:353-376`). Readout is `hl = thr·B[patch(x2), bin(x2→x1)]` (`pt86.py:228-233`).
- [FACT] ER `eval(wi, wo) = f·cosθ_o` with exact `eval`/`pdf`/`sample` (`experiments/fnto_compiled83/closure83_dr.py:1-14, 121-170`). S1 lobes [FACT, `experiments/ecro86/data/mats86.npz`, material 1]:
  - outer GGX lobes au ≈ 0.044 / 0.075 / 0.028;
  - inner lobes au ≈ 0.026–0.041 with η_e = 2.50, so an effective outside width of about 0.065–0.10 [INF];
  - Lambert weight ≤ 0.002;
  - au ≈ av, i.e. effectively isotropic.
- [FACT] WL95A X1 / X1c evaluate exact ER NEE at a relation ray's hit point or at the patch centre (`experiments/ecro_attr95a/a95.py:90-136`). X1 (C3URX1) costs 526,467 probe + 526,467 shadow rays per light (`docs/worklogs/91_…md:256`).
- [FACT] The event-exclusive targets e2_p / e2_o / e3_p / e3_o exist (`experiments/ecro86a/pt86a.py:73-75, 190-199`). `quad87.qtable` exists (`experiments/gm1_explicit87/quad87.py:256-268`).
- [MEAS WL86:214] ECRO per edit: 76–78 ms (geometry 31, state 21, source 5, operator 11, transfer 1.5).
- [MEAS WL87:281] Source-only update: ECRO 13–18 ms; EXP 51–64 ms.
- [FACT] Canon:
  - EXP/C3UR are comparators; X1 findings do not justify converting to explicit GI (`docs/Architecture.md:13`).
  - The next decision must be "one falsifiable ECRO-lineage … hypothesis" (`docs/RESEARCH_ROADMAP.md:128`).
  - "not an exhaustive explicit per-surface full path tracer hidden under a neural name" (`docs/FNTO_ARCHITECTURE_CONTRACT_V0_1.md:86`).
  - Every segment has exactly one owner (`docs/Architecture.md:25`).
- [MEAS] The supervising session's `results/wl96_calc/ring_alloc.json` reproduces locally:
  - CK2 incidence occupies ring-0 bins 12–15 only;
  - M64·P_ring0 has rank 16 per channel, with an exact non-negative rank-16 factorisation;
  - the WL95A bound operators allocate only a 6.4 % row weight to ring 0.

---

## 2. New CPU toy measurements (WL96-D3, scratch)

**Setup.**
- Exactly-2-event delivery, CK2 S1 panel → box A (54 Lambert patches), one point light.
- ER (ER arm) ported to numpy from `experiments/fnto_compiled82/closure82.py:69-147` with the `mats86` state.
- Exact reference: deterministic panel quadrature (5 mm grid, 16 receiver points per patch).
- Box-A occlusion is checked on light→panel segments; box A is convex for panel→box paths.
- Metric: box-A per-patch exitance (albedo 0.5), flux ratio and abs rel L1 against the toy exact reference.
- Toy limits: S1 → box A only, no other occluders, Lambert receivers only. Light 1 has no path-traced reference.

### 2.1 Validation (light 0 = CK2 'none', grazing; incident share ring 0 = 1.00)

| check | result |
|---|---|
| toy exact vs WL95A PT e2_p (`results/ecro_attr95a/deliv.npz` ref_a/ref_b) | ratio **1.015**, rel L1 **0.068**, corr 0.995. Reference half-noise 0.059 |
| quadrature convergence, 5 mm vs 2.5 mm | rel L1 6e-5 |
| WL95A pooled X1 / X1c / q256 / M64 vs toy exact | 1.007/0.098; 1.022/0.231; 1.401/0.410; 1.521/0.522 |
| toy binned table (20 cm × 4×16, exact patch × bin mean, infinite rays) | **1.420 / 0.428**, reproducing WL95A q256 pooled 1.422 / 0.433 |
| toy patch-centre exact direction (X1c-like, deterministic) | 1.068 / 0.269 (WL95A X1c pooled 1.037 / 0.268) |

[INF] The toy is a noise-free oracle for this family that agrees with the PT reference to within its noise. It reproduces the WL95A binned-bias and X1c attributions independently.

### 2.2 Carrier ladder (physical table, deterministic flight = infinite relation rays). Ratio / rel L1 vs exact.

| sender cell | bins | light 0 (grazing) | light 1 (toy-only, ring 1/2 = 0.79 / 0.21) |
|---|---|---|---|
| 20 cm | 4×16 | 1.420 / 0.428 | 0.981 / 0.162 |
| 10 / 5 / 2.5 cm | 4×16 | 1.419 / 0.436; 1.420 / 0.438; 1.383 / 0.404 | 5 cm: 0.989 / 0.221 |
| 20 cm | 16×64 | 1.048 / 0.103 | 0.979 / 0.052 |
| 20 cm | 32×128 | 1.037 / 0.109 | — |
| **10 cm** | **16×64** | **1.012 / 0.027** | **0.999 / 0.008** |
| 5 cm | 16×64 | 1.015 / 0.020 | 1.001 / 0.005 |
| 5 cm | 32×128 | 1.007 / 0.011 | — |

[INF] What the ladder shows:
- The bias is angular-first. Spatial refinement alone does nothing.
- Joint spatio-angular resolution is needed. This is consistent with WL95A's Ra + Rp.
- The bias is strongly incidence-dependent: grazing 0.43 vs ring-1/2 0.16 at 64 bins.

### 2.3 Estimators (sampling variance vs bias)

| estimator | light 0 | light 1 |
|---|---|---|
| X1 receiver-anchored, κ = 8 (512 rays / patch), per draw median / p90 / worst | **0.603** / 1.08 / 3.23 (WL95A: 0.617 / 1.19 / 2.21) | 0.312 / 0.38 / 0.45 |
| X1 κ = 8, pooled 64 draws | 0.948 / 0.159 | 0.981 / 0.053 |
| X1 κ = 64 (4,096 rays / patch), per draw median | 0.319 | 0.106 |
| receiver-side image-source lobe proposal (G1/LCFG-type), K_f = 64 / 256 samples per receiver point, MC median | 0.158 / 0.081 | 0.061 / 0.038 |
| linearised stationary-point proposal, K_f = 256 MC median | 0.195 (worse) | — |
| **light-side lobe-sampled photons (ELCF-type), M = 40k / 160k / 640k** | **0.069 / 0.033 / 0.018** (worst 0.109 / 0.050 / 0.027; ratio range 0.98–1.03) | 0.039 / 0.019 / 0.009 |
| fine table 10 cm × 16×64 + sender-anchored relation, r = 1 / 4 / 16 rays per (cell, bin) = 172k / 688k / 2.75M rays; median over 8 relation draws | 0.135 / **0.070** / 0.035 (worst r=4 0.092) | 0.070 / 0.035 / 0.020 |

- [MEAS-toy] Photons landing on box A: 20 % (light 0), 46 % (light 1).
- [INF] Receiver-anchored exact estimators (X1 and receiver-side lobe proposals) are variance-limited at grazing. The integrand is concentrated jointly by the lobe and by the 1/d³ grazing incident irradiance near the panel edge nearest the light.
- [INF] The sender-anchored light-side estimator has no heavy tail (worst ≈ 1.5× median) and needs about 80k–320k rays per light. Pooled X1 on CK2 needed about 64 × 650k relation rays.

### 2.4 Exact S1 local operator rank (flux-weighted, approximate projection)

| bins | rank for 90 % / 99 % energy | rel Frobenius error at rank 16 / 32 |
|---|---|---|
| 64 | 36–37 / 61–63 | 0.63–0.66 / 0.36–0.39 |
| 256 | 78–89 / 204–208 | 0.75–0.79 / 0.60–0.65 |
| 1,024 | 95–153 / 492–579 | 0.89–0.91 / 0.75–0.81 |

[INF] At the resolution the ladder requires, the light-independent sharp operator is effectively full rank. A learned low-rank operator (ECRO tied a cᵀ) cannot hold it, whatever the training allocation.

---

## 3. The starting point (learned source-independent sharp basis): rejected as a *learned* representation

**Mathematical form.** B(x, v) = Σ_k c_k(ℓ) Φ_k(x, v; G), with Φ_k geometry-dependent and source-independent and c_k = ⟨ψ_k, s_ℓ⟩ compact.

- **For Lambert receivers the output side is trivial.** Rank ≤ N_rcv. All sharp information sits on the input side: K_j(y, ω) = the receiver distribution of light arriving at sender point y from continuous direction ω.
  - [INF] K_j is the **reflected visibility map** of sender j blurred by the lobe: an indicator-like function in ω whose resolution scales as α⁻² per point and h⁻² over the sender surface.
- **Three ways to represent K_j:**
  1. Densely, as a table (B below). It is source-independent but has α⁻²·h⁻² size, and its local factor has rank ~500 at 1,024 bins [MEAS-toy §2.4].
  2. As a learned material-level lobe model, which re-learns ER (WL95 MEK rejection; ER is already exact).
  3. As a learned geometry-conditioned visibility / transfer field. That is learned C: [MEAS WL52] valid but 8–19× slower to update and failing at contact and visibility transitions.
- **Light-space basis (point lights).** The minimal compact source state is nonlinear: per light and per planar sharp facet, the light position, or equivalently its image source.
  - A per-geometry fitted map from light position to receiver irradiance needs a fine 3D light grid × N_rcv. Example: 32³ × 1,270 × 3 floats ≈ 0.5 GB per geometry [INF arithmetic].
  - Learning it per geometry is a refit lifecycle, which conflicts with "no post-edit retraining" (`docs/Architecture.md:48-56`).
- **Verdict [INF]: not viable as learned.** Its physically projected form is Candidate B.

---

## 4. Candidate A: LSC-R (Lagrangian Sharp Carrier + learned broad Resolvent)

### Definitions

- **Lobe partition [HYP].** A material-only prospective rule splits ER lobes into sharp lobes P (α_eff = α, or η·α for inner lobes, ≤ α_c, e.g. 0.12, frozen before results) and a broad remainder.
  - [FACT] ER eval is a sum of per-lobe terms, so f = f_P + f_B and every path family splits exactly (`closure82.py:121-146`).
  - S1: all GGX lobes are P. S2 (α 0.075–0.18): mixed. Floor B (α 0.27–0.45): all broad.
- **Sharp once-scattered radiance at y** from light ℓ:

  L1^P(y, v) = eval_P(v, ω_ℓ(y)) · I_ℓ · V_ℓ(y) / d_ℓ(y)²

- **2-event incident at broad receiver patch i, incident bin m (sharp-mediated):**

  E2^P[i, m] = Σ_{k=1..M} w_k · 1[hit(y_k, v_k) ∈ i] · 1[bin_i(−v_k) = m] / A_i

  - y_k ~ p_A ∝ cos_y·V_ℓ / d²;
  - v_k ~ p(v | ω_ℓ(y_k)), a mixture of sharp-lobe slope densities plus a defensive cosine component (§2.3 toy);
  - w_k = eval_P(v_k, ω_ℓ)·I·cos θ_v / (d²·p_A·p_v·M).
- **Lobe-continuation relation C^P.** It is geometry-only and source-independent.
  - Every ECRO state ray r of a broad patch that lands on a P surface at y_r stores K_c continuations: v_{r,s} = ER P-lobe samples given the view direction −w_r, with fixed QMC u_s, traced to (h_{r,s}, b_{r,s}).
  - Weights β_{r,s} = eval_P(−w_r, v_{r,s}) / (K_c·pdf_P) are cached coefficients that depend only on geometry and ER. Depth is ≤ 2 sharp events.
  - Radiance arriving along r is L(r) = Σ_s β_{r,s}·T_broad[h_{r,s}, b_{r,s}].
- **Composition.** E_inc^(2) = Flight_R0(M64·s at broad senders) + E2^P (the exact 2-event incident field), and

  B_broad = M64[pmat]·E_inc^(2) + R_θ(G')[E_inc^(2)].

  G' is the relation with continuation edges. Sharp patches have **no** table rows. Camera paths whose x2 is sharp continue for one extra ER-sampled bounce plus NEE, then read B at the broad x3.

### A. State

| part | contents | dims (scene86) | dependency |
|---|---|---|---|
| material-only | ER 69 floats per material; lobe partition (bool per lobe); P-lobe proposal parameters (α, weights) | 6 × 69 floats | material |
| geometry-only | ECRO state relation (as now); continuation C^P | 650,240 rays, 27 B/ray [FACT `pt86.py:353-376`; MEAS WL86]; ~10k S1-landing rays per relation [MEAS WL95A B4] × K_c (1–4) ≈ 10–40k segments + β, plus S2 landings [INF similar order] → ≤ ~2–3 MB | geometry |
| geometry-only, learned | R_θ factors for broad patches (c, a, Q, P as ECRO, rows of sharp patches dropped) | ≤ ECRO's ~43 MB [MEAS WL86] | geometry + weights |
| source-dependent | s (broad, as ECRO); E2^P table [N_broad × 64 × 3] built from M photons per light and sharp facet | 20,320 shadow rays per light [FACT PSRC = 16, N = 1270]; photons transient | lights + geometry |

### B. Operator interface

- Inputs: lights (positions, intensity; any samplable emitter), current geometry/BVH, ER states.
- Outputs: B_broad[N_broad, 64, 3] and camera continuation at sharp vertices.
- Sharp queries are continuous in position and direction: (y_k, ω_ℓ(y_k), v_k) at photons, and (y_r, −w_r, v_{r,s}) at continuations.

### C. Ownership

- ER owns every local event, evaluated exactly at continuous directions.
- Geometry owns every segment: light→y, y→receiver, continuation segments, state rays.
- The learned part owns only compressed ≥ 3-event multi-flight among broad patches.
- [FACT `Architecture.md:25`] One owner per segment is satisfied.
- ECRO's event partition (ECRO owns ≥ 3, B excludes e2) is kept for broad patches. For sharp-mediated transport, e2_p is explicit (photons) and e3_p is explicit at the sharp vertex (continuation) but uses learned or explicit broad radiance at the continuation hit.

### D. Learning

- R_θ maps E_inc^(2) (patch × bin, broad patches) to ≥ 3-event broad outgoing.
- Form: ECRO apply with s replaced by E_inc^(2), a learned flight over G' (continuation edges included), and a rank-24 global term driven by c·E_inc^(2), which fixes L8.
- Supervision: exclusive e3_p + e3_o tallies at broad patches (`pt86a.py:190-199`).
- What it can express beyond explicit ER + transport [INF]: **nothing in information content.** At most amortisation of J explicit sweeps (cost and variance).
- The sharp first order is **not learned**.

### E. Sharp lobe preservation (relation to WL95A)

- Photons evaluate ER at the exact (y, ω_ℓ(y), v), which is X1-level information (X1: 1.02 / 0.11 pooled). They are sender-anchored, so the X1 per-draw heavy tail (median 0.60–0.62) is absent [MEAS-toy: 40k photons 0.069, worst 0.109; light 1 0.039].
- No binned outgoing table exists at sharp senders, so M64's 1.42–1.54 bias cannot arise.
- The receiver incident bin is kept only at broad receivers, where WL95 found bins adequate (floor B exact at rank 8–16 [MEAS WL95 §2.1]).

### F. Dynamic reuse

| event | survives | invalidated / recomputed |
|---|---|---|
| broad rigid edit (box) | ER; partition; non-dirty relation and continuation segments; R_θ weights | dirty relation rays **and continuation segments** (segment-OBB on both legs); R_θ factors regenerated (rule reuse, as ECRO); all photons per light (re-shot, O(M)) |
| light change | relation, continuation, factors | s, photons |
| rigid sharp mover | ER, partition, continuation directions in the facet frame (β invariant) | continuation hit targets, relation rays landing on or crossing it, photons, factors |
| non-rigid sharp | ER | the above, plus no correspondence certificate, so conservative full rebuild of that facet's continuations |

### G. Complexity (symbols: N patches, K rays/patch, L lights, F_s sharp facets, M photons, K_c continuations, n_s sharp-landing relation rays)

**Offline.** No new learned sharp component. R_θ training is WL86-scale target generation: 1,024 geometries × 4 lights × 256 spp [MEAS WL86:129].

**Per light.**
- Rays: 20,320 (s) + 2·M·F_s. CK2 at ≤ 0.07 needs M ≈ 40k, i.e. **80k rays per light per facet**; M = 160k (0.033) means 320k.
- vs C3URX1 1.05M [MEAS WL91:256]; vs pooled X1 on CK2 ≈ 64 × 650k = 41.6M relation rays [MEAS WL95A].
- Time [INF, scaled from WL91 11.5 ms per 1.05M rays + ER]: about 1–4 ms per light per facet.

**Per edit.** As ECRO (76–78 ms [MEAS]) minus sharp rows, plus continuation re-trace (≤ 40k rays) and photon re-shoot.

**Asymptotics [INF].**
- Per light: O(F_s·M), independent of α and of N.
- Per-receiver accuracy ∝ √(landing density), so small receivers or near-contact receivers need more M.
- Geometry: O(N·K + n_s·K_c).

**Avoided vs X1.** Repeated receiver-anchored relation draws, and per-landed-ray probe and shadow evaluation.

**Introduced.** Per-light photon shooting and splatting, continuation bookkeeping, and camera continuation at sharp x2 (one ray plus L shadow rays per such pixel path).

### H. Failure modes

- **α → 0.** Photon variance is α-independent and K_c → 1. A true delta needs ER delta handling (ER has α > 0) [INF robust].
- **Close contact.** Small or near receivers need M ∝ 1/area.
- **Many lights.** Cost is linear in L·F_s. Power-proportional photon allocation is the variance trade.
- **Moving sharp objects.** Re-shoot is cheap.
- **Non-rigid.** No intrinsic table, but continuation validity has no certificate.
- **Unseen materials.** Exact for any ER state. Risk at the α_c boundary, where both sides are exact but the cost/variance profile differs.
- **Curved sharp senders (S2).** Photons are unaffected; image-source methods would not be.
- **Indirect incidence on sharp (e3_p, 0.755 of panel-face e3 [MEAS WL94]).** Handled by continuation reading a broad table: [HYP] low variance, because broad radiance is smooth. Untested.

### Representability, capacity, error sources

- **Representability.** No representation floor at the sharp vertex. The capacity knob is M. [INF from toy] M ≈ c/(ε²·f_land), with f_land = 0.20 / 0.46.
- **Error sources:**
  - photon MC noise;
  - continuation lobe sampling (K_c);
  - binning at glossy-broad receivers;
  - R_θ approximation;
  - camera-continuation noise.

---

## 5. Candidate B: ESC-R (Eulerian Sharp Carrier + learned broad Resolvent)

### Definitions

- Sharp facet j is split into cells c (≤ 10 cm) and outgoing bins b (16×64 equal-solid-angle, i.e. 1,024).
- **Direct-light sharp table, per light, physical:**

  T_ℓ[c, b] = mean_{y ∈ c, v ∈ b} eval_P(v, ω_ℓ(y))·I·V_ℓ(y)/d²

  computed with P_c points per cell × D_b directions per bin. No transport rays; P_c shadow rays per cell.
- **Indirect-incidence sharp table** (for e3_p): T_ind[c, ·] = M_P^{64→1024}·E_inc^ind[c, ·].
  - M_P is the material-level exact ER projection: incident 64 bins (incident indirect field, smooth) → outgoing 1,024.
  - Size 64 × 1,024 × 3 = 197k floats; 1,024² × 3 = 12.6 MB if a fine incident resolution is ever needed.
- **Sender-anchored relation Γ_j.** Geometry-only, source-independent: r rays per (c, b), with fixed offsets, landing at (receiver patch i, incident bin m) with weight A_c·Ω_b·cosθ_v/r.
- **Delivery:** E2^P = Γ_j ⊗ (T_ℓ + T_ind). This is a scatter-add, linear and superposable across lights and incidence types.
- **Composition:** as A, with E_inc^(2) = Flight_R0(M64·s) + Γ⊗T_ℓ, plus R_θ for ≥ 3 events. Sharp facets **do** keep rows: the camera reads T at the fine bin, so no per-pixel continuation is needed.

### A. State

| part | contents | dims (S1 at 10 cm / 16×64) | dependency |
|---|---|---|---|
| material-only | ER; M_P (64 → 1,024) | 197k floats per sharp material | material |
| geometry-only | Γ_j | 168 cells × 1,024 bins × r: r = 4 → 688,128 rays (≈ 5.5 MB at 8 B); r = 16 → 2.75M | geometry |
| geometry-only, learned | R_θ factors (broad) | ≤ 43 MB | geometry + weights |
| source-dependent | s; T_ℓ (516,096 floats ≈ 2.06 MB per light per facet, transient) | 2,688 shadow rays per light (16 per cell) + ~2.7–25M ER evals | lights |

### B. Interface

- Inputs: lights, geometry, ER.
- Sharp outputs: T[c, b] at fine bins. Broad outputs as A.
- Queries: cell (sub-patch position) × fine bin. Continuous information is projected to (10 cm, 0.006 sr).

### C. Ownership

- ER is owned through exact projections (T_ℓ, M_P).
- Geometry owns Γ_j, R0 and the source shadow rays.
- The learned part owns only the broad ≥ 3-event resolvent.

### D. Learning

- Same R_θ as A.
- [MEAS-toy §2.4] M_P / T are full-rank at 1,024 bins. No learned compression is admissible, so the sharp part stays physical.
- Beyond explicit ER + transport, it expresses nothing new.

### E. Sharp lobe preservation

[MEAS-toy]:
- Infinite rays: 0.027 (grazing) / 0.008.
- r = 4: 0.070 / 0.035.
- r = 16: 0.035 / 0.020.
- vs M64 0.428 / 0.162.

Errors are deterministic per geometry: a fixed relation pattern, as in ECRO's deployed draw. They are not reduced by more lights or frames.

### F. Dynamic reuse

| event | survives | invalidated / recomputed |
|---|---|---|
| light change | **everything geometric** (Γ_j, M_P, R_θ factors) | only T_ℓ and s. **This is B's niche: no transport rays per light** |
| broad rigid edit | M_P, T_ℓ (if light paths are clear) | Γ_j rays crossing the old or new OBB (selective); R0; R_θ factors |
| rigid sharp mover | M_P; cell/bin definitions in the facet frame | all of Γ_j (0.7–2.8M rays), its T_ℓ |
| non-rigid sharp | ER | cells deform and bin frames rotate per cell: rebuild Γ_j fully; curved cells need finer h |

### G. Complexity

- **Geometry.** O(Σ_j A_j·h⁻²·N_bins(α)·r), with N_bins ∝ α⁻² [INF]. At α = 0.01 that is ~16k bins, ~11–44M rays per facet: infeasible.
- **Per light.** O(cells·P_c) shadow rays + O(cells·P_c·D_b·bins) ER evals + O(|Γ|) scatter. Sub-ms-class [INF].
- **Per edit.** ECRO-class + selective Γ.
- **Memory.** ~5.5–22 MB per facet (Γ) + 2 MB per light (T) + ≤ 12.6 MB per material.
- **Avoided vs X1.** All per-light transport rays.
- **Introduced.** A fine geometry-only relation comparable in size to the whole ECRO state, per sharp facet.

### H. Failure modes

- **α → 0.** α⁻² bins: hard failure. B is admissible only above an α_min to be measured.
- **Grazing incidence.** Largest bias (0.07 at r = 4 vs 0.035).
- **Close contact.** Needs h ≲ distance × angular resolution.
- **Many lights.** Good.
- **Moving or non-rigid sharp.** Full Γ rebuild.
- **Unseen materials with sharper lobes.** Resolution must be chosen per material class.
- **Memory with many sharp facets.** Linear in sharp area.

---

## 6. Shared learned component R_θ and its utility

- [HYP] R_θ = the ECRO generator (`model86.py:91-107`) with three changes:
  1. its input is the exact 2-event field E_inc^(2), not s;
  2. the global term is driven by c·E_inc^(2), not u;
  3. its flight runs over G' (A: continuation edges; B: Γ-coupled sharp rows treated as fixed exact operators).
- Targets: exclusive e3 tallies.
- [MEAS cross-regime] Prior evidence predicts little headroom:
  - E2: exact prefix + learned G 0.030 / 0.033 vs exact prefix + PT L3+ 0.0127, and C3UR 0.026 (WL95 judge §2);
  - WL87: learned fidelity utility 0 / 3 / 36; only the source-only update is 3.5–3.9× cheaper (about 40–50 ms) [WL87:315, 346];
  - sweep vs generator cost: 11.6 vs 12.5 ms (cross-scene).
- [INF] With M64 broad q (not W_ray MIS), the explicit source update may itself become cheap, which would remove that niche.

---

## 7. Comparison

| | ECRO (OWN) | explicit X1 (C3URX1) | H-0 (EXP/C3UR) | **A LSC-R** | **B ESC-R** |
|---|---|---|---|---|---|
| sharp carrier | learned rank-16, patch × 64 (non-passive, albedo 2.3 / 3.1 [MEAS WL95A]) | exact per landed relation ray | patch × 64 (q) | exact continuous, sender-anchored | 10 cm × 1,024 table + fixed sender relation |
| CK2 grazing e2_p delivery, rel L1 | 0.57–0.97 pooled [MEAS WL95A] | per draw median 0.60–0.62; pooled 64 draws 0.10–0.16 | 0.43 (q) [MEAS] | 0.069 (40k) / 0.033 (160k) [toy] | 0.070 (r = 4) / 0.035 (r = 16) [toy] |
| per-light rays | 20k | ~1.05M | 20k + MIS | 20k + 80–320k per facet | 20k + 2.7k |
| per-geometry extra | — | relation | relation | ≤ 40k continuations | 0.7–2.8M per facet |
| α → 0 | fails | variance grows | fails | robust | fails (α⁻²) |
| learned role | everything (fails sharp) | none | none | broad ≥ 3 only | broad ≥ 3 only |
| canon status | base | comparator | comparator | Gate A | Gate A |

---

## 8. Smallest next feasibility experiment (no training; learning-free + oracle)

**Batch class.** The owner decides (AGENTS.md policy). Two readings:
- suffix "WL96A", because it tests WL96's design question;
- regular, because the carriers are new mechanisms for this lineage.

**Location.** LabServer63, with full-scale GPU-utilisation probes per the standing rule.

**Workloads.**
- W-a: CK2 'none' light 0 (grazing). Existing CKF_none references.
- W-b: one non-grazing CK2 light chosen by a frozen geometry-only rule: maximise the ring-1/2 irradiance share on S1 subject to box-A e2_p SNR ≥ 4 in a 256-spp pilot that is not used for scoring. Then build a new 2 × 4,096-spp `field_src(split=True)` reference.
  - Illustration only: the toy light (−0.9, −0.5, 2.4) gives ring 1/2 = 0.79 / 0.21.
- W-c: E2 T1 region C (existing REF-ER, ORA2 0.0127).
- W-d: CK2 rigid T0 / T1 box-A edit, for the signed-change slope.

### Tests

**S0 — contracts.**
- Reproduce WL95A A4 / q256 / X1 to 1e-6.
- Port the deterministic panel quadrature oracle (DQ) to Dr.Jit. It must match CKF e2_p within ratio ± 0.03 and rel L1 ≤ n0 + 0.02. The toy already gives 1.015 / 0.068 vs n0 0.059.

**S1 — test 1 (representability) and test 2 (physical oracle).**
- Arms on W-a and W-b against DQ and PT:
  - A-ph, M ∈ {40k, 160k}, 16 seeds;
  - B-fbc, (10 cm, 16×64), r ∈ {4, 16}, 8 relation draws;
  - B-ladder, cells {20, 10, 5} × bins {64, 256, 1,024};
  - references X1 pooled, q256, M64.
- Metrics:
  - Q_d2 on box A (54 patches) and the panel face (9 patches): flux ratio, abs rel L1, corr;
  - per-seed and per-draw median / p90 / worst.
- Added **A-ct (e3_p).** Continuation K_c ∈ {1, 4} reading an **oracle** broad table (PT e2+e3 outgoing), against PT e3_p on the panel face. This isolates continuation error.

**S2 — test 3 (generalisation).** W-b and W-c with thresholds frozen before results, plus the W-d signed change (slope and corr of ΔE2^P vs PT Δe2_p).

**S3 — test 4 (variance vs bias).** QMC vs MC seeds for A; relation redraws for B; X1 draws as reference.

**S4 — test 5 (cost).** Warm, uncontended, with Dr.Jit compile excluded and reported. Measure:
- per-light update (rays, ER evals, ms);
- per-edit update (dirty rays, ms);
- memory (MB).

Compare against C3URX1 (single and 64-pooled) and ECRO (13–18 ms source-only; 76–78 ms edit).

**S5 — test 6 (learned utility, no training).**
1. Render E_H on CK2 ROI_T and E2 C (2 × 4,096 spp) for:
   - LOW + arm L2 + PT L3+ (floor);
   - LOW + arm L2 + explicit C3UR-recurrence L3+ (explicit resolvent on the arm's L2; for A on G' with continuations).

   Headroom H_L = difference. This is an upper bound on what any learned resolvent could add in accuracy.
2. Time explicit sweeps on G' vs the ECRO generator (`factors`).
3. Optional non-deployable bound: an in-sample least-squares fit of the rank-24 global term on the evaluated geometry. If even this does not beat the explicit sweeps beyond tolerance, learning has no accuracy role.

### Predeclared falsification (freeze before results)

- **FA1.** A-ph at M = 160k: median rel L1 vs DQ ≤ 0.05 and ratio ∈ [0.97, 1.03] on W-a and W-b; vs PT ≤ n0 + 0.03. Otherwise A's direct sharp carrier is falsified.
- **FA2.** A-ph: per-seed p90 ≤ 1.5 × median and worst ≤ 0.10. Otherwise heavy-tail variance.
- **FA3.** A-ct: panel-face e3_p rel L1 ≤ n0 + 0.05 vs PT. Otherwise continuation fails, and e3_p stays unowned.
- **FB1.** B at (10 cm, 1,024 bins, r = 4): rel L1 ≤ 0.08 (W-a) and ≤ 0.05 (W-b), ratio ∈ [0.95, 1.05].
- **FB2.** B: worst relation draw ≤ 0.10.
- **FB3.** B: memory ≤ 32 MB per facet + material.
- **FB4 (α ladder).** Synthetic roughness-scaled S1 variants at α × {2, 1, 0.5}, DQ-only. Report the bins needed for ≤ 0.05. If they grow ≥ 3.5× per halving, B is admissible only above the measured α_min.
- **FC (both, cost).** Per-light update ≤ C3URX1 single-draw cost (1.05M rays) and per-edit ≤ ECRO + 10 ms at equal or better fidelity. Otherwise there is no economic case.
- **FI (integrated).** E_H(arm L2 + PT L3+) ≤ floor + 0.0167 (2 × 4,096 tolerance [MEAS WL95A B3]) on CK2 ROI and E2 C; edit slope ∈ [0.9, 1.1].
- **FL1 (learned role).** If H_L ≤ 0.0167 on every workload **and** explicit sweeps on G' ≤ generator time, the learned resolvent has no measured utility. The candidate then reduces to explicit GI (H-0+), and under canon **no viable learned candidate remains**.
- **Gate A (owner).** If the owner rules physical sharp carriers inadmissible, A and B are withdrawn.

### Compute estimate [INF]

- DQ / photons / tables / relations: about minutes of GPU.
- Renders: ~10 s each [MEAS WL95A B4: 8.6–11.4 s] × roughly 20–30 tables.
- One new 2 × 4,096-spp field reference for W-b.

---

## 9. Open questions

- Is a physically projected sharp carrier inside the ECRO lineage admissible (Gate A)? Both candidates depend on it.
- e3_p via continuation is [HYP]. The broad-table semantics at continuation hits (all orders vs truncated) must be fixed so no event family is double counted.
- The α_c partition boundary: S2 (α 0.075–0.18) and anisotropic lobes are untested.
- Generality is limited to point lights, one planar sharp facet, one curved sharp object, and rigid box edits. Area and environment lights and non-rigid sharp surfaces are untested. Light 1 is toy-only, without a PT reference.
- The toy covers S1 → box A exactly-2-event only, Lambert receivers and no foreign occluders. Its light-0 agreement with PT validates only that case.

---

## 10. Reproducibility (scratch, not repository evidence)

**Directory:** `C:\Users\dna10\AppData\Local\Temp\claude\462fa2b2dbb71cdd\46d65e15-e083-4057-875f-9282fb68aa5e\scratchpad\wl96\d3\`

| script | role | command |
|---|---|---|
| `toy96.py` | ER numpy port, DQ oracle, validation vs `results/ecro_attr95a/deliv.npz` | `python toy96.py validate` → `res_validate.json` |
| `toy96b.py` | exact quadrature, carrier ladder, X1 draws, receiver-side image-source proposal | modes `exact` / `ladder` / `x1` / `portal` with light 0 or 1 |
| `photon.py` | light-side photons | `python photon.py 0\|1` |
| `portal2.py` | linearised proposal | `python portal2.py 0` |
| `fbc.py` | Candidate B flight | `python fbc.py 0\|1` |
| `rankM.py` | operator rank | `python rankM.py` |

**Outputs:** `res_*.json`, `E_exact_L0.npy`, `E_exact_L1.npy`.

**Seeds.** X1 draws: 96100 + d. Photon MC seeds: 100–115. FBC relation draws: 9600 + 10r + d.

**Environment.** Python 3.11.9, numpy 2.4.6, local PC CPU.
