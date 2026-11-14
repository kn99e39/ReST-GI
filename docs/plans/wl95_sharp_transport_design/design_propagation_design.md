# WL95 — Propagation-first architecture design (independent designer #propagation)

**Scope.** This is a design-only report.
- I edited no repository file. I did not train, run GPU work or start experiments. I wrote no scratch files.
- Inputs: the eight Stage A/B and reader files, plus spot-checks of these code and doc lines:
  - `experiments/ecro86/model86.py:109-124`
  - `experiments/ecro86/pt86.py:153-156`
  - `experiments/ecro_mat94/an94.py:19,32`
  - `experiments/ecro_s94d/s94.py:152-160`
  - `experiments/gm1_c3urx1_91/cand91.py:1-110`
  - `experiments/gm1_relation88/rel88.py:5-15`
  - `experiments/gm1_explicit87/quad87.py:85,240,268`
  - `experiments/ecro86a/pt86a.py:73-74,175-199`
  - `docs/worklogs/90_…md:150-206`
  - `docs/worklogs/81_…md:186-250`
  - `docs/worklogs/82_…md:29-40`
  - `docs/Architecture.md:17-38`
- Labels: [FACT] verified, [MEAS] measured (source cited), [INF] reasoned inference, [HYP] untested.
- I make no novelty claim. Every mechanism below (MIS, specular/stationary-point next-event estimation, image sources, two-pass caching) is known technique.
- No mechanism below is presented as a demonstrated fix.

---

## 1 Verified Current Architecture

**ECRO OWN, the deployed lineage** [FACT, `model86.py:109-124`, `pt86.py:153-156,228-233,353-402`; auditA §1]:

- **Bins:** 64 equal-solid-angle bins (4 cos θ rings × 16 sectors, ΔΩ = 2π/64), indexed `bin = iz·16 + iφ`.
- **Patches:** 1,270. The panel S1 is 42 patches of 20 cm.
- **Source:** `s[n,m] = (1/16) Σ_j Σ_l I cos/d² V · 1[bin = m]`. Each patch uses the same 16 points; s is patch-mean direct irradiance per incident bin.
- **Local operator:** `Bq = a·(c·s)`, so M_n = a cᵀ (rank ≤ 16, non-negative, per patch and channel). It is a learned *bin-projected* once-scattered sender table.
- **Flight:** `E1[n,m] = Σ_{κ<8} 1[hit≥0] · Bq[hit, dbin] · z_k ΔΩ/8`.
  - 512 receiver-side rays per patch.
  - One fixed (origin, direction) pattern is shared by all patches.
  - Each ray reads the sender's patch-constant, bin-constant value.
- **Receiver:** `u1 = c·E1`, with the same tied c.
- **Global term:** `y_g = P Σ area Q u`, driven by the direct u, not by E1.
- **Output:** `B = a(u1 + y_g)` = 2–5-event outgoing radiance.
- **Supervision (OWN):** single → t2 (exactly 2 events), global → t − t2 (`own92a.py:172-176`).
- **Readout:** `hl = thr(x1) · B[nearest patch(x2), bin(x2→x1)]`. Events 1–2 at the camera are path-traced (LOW).
- **Edit path:**
  - set_geom (31 ms);
  - selective dirty-ray retrace, bit-equal to a full rebuild (21 vs 26–28 ms);
  - full source recompute (5 ms);
  - full generator recompute (Ft is N², message passing is global; 11 ms);
  - total 76–78 ms per edit [MEAS WL86:214-215].
- **Ownership of transport (functional):**
  - ER owns every physically evaluated event (LOW).
  - Geometry owns visibility (state rays).
  - The learned generator owns the local response used inside the table (a, c) *and* the ≥ 3-event compression (P, Q).
  - ECRO partitions by event count, not by K/C segment. Its relation to the K/C contract is unresolved in canon [FACT, read_canon (B)].
- **Local layer outside ECRO:** frozen learned ER (WL82–84) is an analytic closure of 8 lobes (Lambert + 7 GGX), with incidence-dependent energy s(μ) and coat-refracted inner lobes [FACT, WL82:29-40].
  - It is continuously evaluable and sampleable.
  - ECRO does not use this continuity inside the table. It re-learns a bin-projected rank-16 copy (`a cᵀ`) from geometry and material embeddings [FACT].

---

## 2 Failure and Evidence Attribution (Evidence B and C)

### 2.1 Local sender, matched norms (Stage B)

- [MEAS, auditB] In the WL94D operator metric (flux-weighted L1, zb-weighted, pooled):
  - zero operator = 1.000;
  - learned S1 = 0.961–1.044;
  - SVD16 = 0.891 (signed, inadmissible);
  - NMF16 = 0.603;
  - L1-targeted non-negative rank 16 = 0.380.
- [MEAS, auditB] Field level on the held-out panel fields (W94 weights as coded): learned 0.846–0.892 rel L1, peak 0.22–0.33. Non-negative rank-16 bounds: 0.36–0.54, peak 0.35–0.44.
- [MEAS, auditB] No rank-16 operator tried gets below ≈ 0.36 field L1 or above ≈ 0.49 peak. Field L1 ≤ 0.15 at peak ≥ 0.7 needs rank ≈ 32. Floor B is exact at rank 8–16.
- [MEAS, auditB] Position conditioning over-fits: per-patch rank 16 gives 0.263 train vs 0.451 val.
- [MEAS, WL93D] S1 geometry specialisation S_geo is 0.2–0.8 %. The S1 error is geometry-common (E_common 0.79–0.81).
- [INF] Two separate facts follow:
  1. The learned S1 sender carries **no measurable geometry information** and is at the zero-operator level, so there is no learned content in the sharp first-order path to preserve.
  2. Even a perfectly optimised rank-16 bin operator would cap the sender at ≈ 0.36 field L1. In a fixed bin×bin basis the mirror-like part is near-permutation (high rank). In a reflection-aligned basis it is low-dimensional, and ER already is that basis.

### 2.2 Propagation side

- **WL90** [MEAS, GM1 W1/W2, exact q, pooled relation, E_H R_mirror]:
  - Making the first gather exact (EE) removes 97–102 % of the explicit-vs-oracle gap.
  - Angular-only A = +0.0177…+0.0339; spatial-only S ≈ −0.001…−0.003.
  - P2plus (projection only at orders ≥ 2) ≤ 0.0009; compounding ≤ 9 %.
  - Fixed 4 sub-bins leaves 9–20 % of J; 16 sub-bins leaves ≈ 0.
  - Isolation holds in 3 configurations × 4 regions.
- **WL91** [MEAS] Per-ray exact read (X1) on the C3UR relation:
  - The pooled result sits within −0.0025…+0.0035 of ORACLE in all 6 conditions: the bias is removed.
  - The single-draw sampling term rises to 0.005–0.014, against 0.0005–0.005 for C3UR.
  - Held-out validity 1/3; utility rule (a) fails (3.1× slower at the 0.06 tier).
  - [FACT, `cand91.py:3-6`] C3URX1 relation rays are κ_s per (sender patch, outgoing **bin**) cell: directions are bin-uniform, not lobe-aware for the actual incident light.
- **WL90 R0X1** [MEAS] Receiver-side relation with exact reads: +0.003 [−0.006, +0.012], no gain.
- **WL87 / WL88** [MEAS] Receiver-side relation sampling is heavy-tailed:
  - deployed draw −0.049 vs pooled −0.024;
  - 56–62 % of important light→panel→Lambert relations never hit;
  - SAM (sender-area MIS) coverage 0.59 → 0.993, E_H 0.041 ± 0.017 → 0.030 ± 0.008;
  - sender placement at equal rays removes ≈ 90 % of the deficit (C3UR − R0D +0.050).
- **WL86A Amendment 2** [MEAS] The relation can carry 74–84 % of the family; an important patch sees the panel through 0 / 13 / 249 of 512 rays.
- **CK2 exact-M64(s) floor** [MEAS WL94D; auditA §3]: ratio 1.378 / 1.211 / 1.206; corr 0.713 / 0.716 / −0.334 (the last over 9 patches).
  - The decomposition is **open**: C1 outgoing-bin angular, C2 sender spatial, C3 incident-side L1+L12, C4 16-point source, C5 κ=8 single realisation, C6 ring-0/1 cosine aliasing, C7 receiver, C9 an94.W.
- [INF] Why C1 is plausible on CK2 (auditA §3.2 geometry):
  - Panel incidence is grazing (cos θ 0.024–0.164), so the true lobe exits at z_out ≈ 0.02–0.10.
  - The ring-0 bin spreads that flux uniformly over z ∈ [0, 0.25].
  - Rays from box A reach the panel at panel-frame z ≈ 0.2–1.0, so those in ring 0 read flux that physically never reaches them.
  - This predicts over-delivery and misplacement, i.e. ratio > 1 and negative face correlation.
  - It is a prediction, not a measurement.
- [INF, a geometric-optics estimate used later] Size of the region that actually contributes:
  - For a planar GGX sender of roughness α, light distance d_l and receiver distance d_x, the stationary region has widths σ_⊥ ≈ 2α·d_h and σ_∥ ≈ 2α·d_h / cos θ*, with d_h = d_l d_x/(d_l + d_x).
  - CK2 (α ≈ 0.05, d_l ≈ 0.5 m, d_x ≈ 0.25 m, cos θ* ≈ 0.1): ≈ 1.7 cm × 17 cm, consistent with auditA's "≈10 cm strips of 2 panel patches".
  - From x at 0.25 m this strip subtends ≈ 0.01–0.015 sr. With 8 uniform rays per 0.098 sr bin, the expected hits per receiver point are ≈ 1.
  - The read is therefore a ≈ 1-event Poisson process. This is consistent with X1's variance (WL91) and with R0X1 giving no gain (WL90).

### 2.3 Attribution by loss point (propagation view)

| Loss point | Carrier today | Dominant evidence | Status |
|---|---|---|---|
| L1 / L2 / L12 incident projection at the sender | s, M64 | WL86A ENC ≤ 5 % (E1/E2 point light); WL92B encq vs q 0.36 (unweighted, other norm) | measured elsewhere; open on CK2 (C3, C4) |
| L3 learned rank-16 tied a cᵀ | generator | Stage B: learned = zero level, bound 0.36–0.60 (matched) | measured; capacity vs optimisation open |
| **L4 outgoing bin projection at the sender (first gather)** | Bq / q table | **WL90 97–102 % (GM1)** | established in GM1; inferred for CK2 |
| L5 κ = 8 fixed receiver-side sampling | state rays | WL87 / WL88 / WL86A | measured; open on CK2 |
| L6 ring-0/1 cosine aliasing | rw | none | untested |
| L7 tied receiver c | generator | WL93D C2_rcv; WL94 C2_rcv 0.97–1.00 | measured |
| L8 global compensation | P, Q | WL86A Amendment 1; WL93D | measured |
| L10 nearest-patch readout | readout | WL86A: amplifier | measured |
| L11 sphere frame | binning, M64 | by construction | established |

### 2.4 Effect of WL92–WL94D on a propagation-first design

**Supporting:**
- WL93C ORA and WL93D C2_src: an exact sender + the existing flight gives sharp renders of 0.018–0.033 against the learned 0.040–0.077, and an edit slope of 0.995–1.000 against 0.57–0.75 (E2 region C regime).
- WL94 / 94C / 94D C2_srcS1: CK2 ROI_T 0.031 / 0.019 vs C0 0.047 / 0.042 (noise 0.0084).
- So replacing the learned panel sender by a physical one consistently helps at render level, even through the flawed bin flight.

**Weakening:**
- The CK2 exact-sender floor (1.21–1.38) shows that a physical sender read through the current bin flight is not enough.
- WL91: exact reads without lobe-aware sampling fail held-out and utility.
- WL87: explicit ≥ ECRO, so a learned remainder has no demonstrated accuracy edge.

**Inconclusive:**
- WL94A–94D seed disagreement on the learned sender.
- Whether S→S chains matter outside GM1.
- WL93B LPAT (2-light interface limit).
- The an94.W bias on all WL94-series exitance numbers (auditA §4).

---

## 3 Independent Architecture Hypotheses (Evidence A)

### 3.0 Shared contract: equations and state variables

Notation:
- ER BRDF f(y; ω_i, ω_o), continuous and reciprocal (WL87 Stage 0: reciprocity 1.2e-5, furnace 6e-7 [MEAS]).
- Compact lights l at p_l with intensity I_l.
- G(x,y) = cos θ_x cos θ_y / |x − y|²; V is visibility.

1. Once-scattered radiance (the sharp first-order source):
   L₁(y, ω) = Σ_l f(y; ω_l(y), ω) · I_l cos θ_l(y) V(y, p_l) / |p_l − y|², with ω_l(y) = (p_l − y)/|p_l − y|.
2. First gather (the quantity WL90 isolated):
   E₂(x, ω_i) dω_i ↔ ∫_M L₁(y, ω_{y→x}) G(x,y) V(x,y) dA_y,
   and B₂(x, ω_o) = ∫ f(x; ω_{x→y}, ω_o) L₁(y, ω_{y→x}) G V dA_y.
3. Table quantities stay patch × bin projections (P) of these: B_k[n,d] = (1/(A_n ΔΩ)) ∫_n ∫_d L_k.
4. Higher orders: L_{k+1} = 𝒮_ER 𝒢 L_k. The camera reads LOW + B at x₂.
5. **Matched reference quantity** [FACT, `pt86a.py:73-74,175-199`]:
   - e2_p = exactly 2 events with vertex 2 on material 1, using the same ER, light and visibility. This is exactly the panel term of B₂.
   - Exitance must use the true layout weight W_T[d] = ((d//16) + ½)/4 · ΔΩ, **not** `an94.W` (`an94.py:19,32`).
6. Conservation and passivity:
   - ER albedo ≤ 1 (S1 max 0.825 [MEAS T_ENERGY]).
   - Any recursion (sweeps) is passive if 𝒮_ER is passive and C closure ≤ 1.
   - Any one-shot learned term has no recursion requirement (canon, read_canon D3).

### 3.1 Hypothesis H-A: Lobe-aware Continuous First Gather (LCFG)

**One-sentence mechanism.** The first gather is never read through a stored sender table.
- At each receiver sample point, a fixed source-independent estimator rule integrates the exact once-scattered sender radiance L₁ at exact directions.
- It uses MIS between the existing receiver-uniform state rays and a **sender-lobe stationary-point proposal** constructed from the current light positions and ER's lobe parameters.
- Everything after the first broad event is carried by patch × bin tables.

**State representation.**
- (i) Unchanged geometry state: 512 receiver-uniform rays per patch (hit, dbin, seg) (`pt86.py:353-376`).
- (ii) New geometry-only planar-group table for peaked materials:
  - plane (o, n), aperture polygons, and per-lobe roughness α_k from frozen ER;
  - peaked = the predeclared material rule, e.g. WL88's material-only Π rule (B 4.93, S1 37.95, S2 16.13 [MEAS WL88]).
  - The class rule only chooses *proposals*; it does not assign ownership (see Conservation).
- (iii) Per-source cache, per receiver sample j (16 per patch; a sub-patch grid), light l and peaked plane P: stationary point y*, proposal samples {y_i, pdf_i}, visibility bits V(y_i, p_l) and V(y_i, x_j), and ER values.
- (iv) Receiver tables:
  - B₂ at sub-patch resolution (4×4 per patch);
  - scalar for Lambert receivers, 64 bins for glossy receivers.
- (v) Regular table B_{≥3}[n, d], patch × 64 bins.

**Operator inputs and outputs.**
- First-order operator 𝒜_G,ER: (lights {p_l, I_l}, geometry state, ER) → B₂ (sub-patch × bins). It is linear in I_l for fixed positions and uses no source identity.
- Regular operator ℛ: B₂ (or E₂) → B_{≥3}.
  - Default: explicit same-state sweeps with M64 / ER projection (EXP-like).
  - Optional: a learned geometry-conditioned linear operator R_θ(G), admissible only under §3.1 "Training".

**Estimator** [HYP: the construction is standard, its accuracy here is untested].
- Stationary point for a planar sender: p'_l = p_l − 2((p_l − o)·n) n and y* = x + t(p'_l − x), with t = ((o − x)·n)/((p'_l − x)·n).
- Proposal density p_s: an anisotropic Gaussian on the plane at y*.
  - Its axes and widths follow §2.2 (σ_⊥ ≈ 2α_k d_h, σ_∥ ≈ 2α_k d_h / cos θ*).
  - Equivalently, ER-lobe (VNDF) microfacet samples mapped to the plane with an analytic Jacobian.
  - The pdf is exact in either form.
- Receiver-uniform density converted to area: p_u(y) = (κ_u/ΔΩ) cos θ_y / |x − y|² (WL88 R1 form, `rel88.py:5-15`).
- Balance-heuristic MIS; every sample reads L₁(y_i, ω_{y_i→x_j}) by next-event estimation (NEE) with continuous ER.
- For non-peaked senders only p_u is used; their L₁ may stay a patch × bin q read. WL90 S ≈ 0; Lambert L₁ is isotropic.

**Local scattering ownership.**
- The local angular response is **frozen ER only**, evaluated continuously at the sender (first order) and at the receiver (B₂).
- No learned copy of the local response is used on the sharp path (L3, L7 and L12 are removed for this family).
- The same ER rule serves direct and C-mediated incidence. The response rule does not branch; only the propagation estimator differs.
- This must be checked against Architecture.md:21 ("no direct/indirect response branch"). I read it as compatible for the default explicit ℛ. A *learned* ℛ that responds to indirect incidence with learned factors while ER responds to direct incidence is a response-rule branch, and ECRO v0 already has it (LOW = ER, table = learned). That is an owner decision (§7).

**Geometry-dependent propagation ownership.**
- Free-space segments y→x and y→p_l are owned by the estimator (visibility rays).
- Segments in the regular remainder are owned by the state rays.
- Ownership is exclusive by *path prefix order*:
  - all paths light → any sender → receiver (exactly 2 events at the receiver) belong to 𝒜;
  - all longer paths belong to ℛ.
- This matches OWN's t2 / t3 split, so the tallies sum to the reference by construction.

**Treatment of sharp angular structure.**
- Never stored at a sender; evaluated per query at exact directions.
- After the first gather, sharp structure persists only in what receivers re-emit:
  - Lambert receivers: isotropic, so no angular loss;
  - glossy receivers: a 64-bin projection (P2plus ≤ 0.0009 in GM1 [MEAS]; elsewhere unmeasured).
- Spatial footprint structure (CK2 footprint = 23 % of a face) is kept at sub-patch resolution in B₂, which mitigates L10 for the first order.

**Diffuse and nonlocal transport.**
- Explicit sweeps of P-projected tables on the same state (orders 3–5).
- WL87/89 show these are accurate for broad transport once the first gather is right [MEAS WL90 M₁ within 0.001–0.002 of ORACLE].

**Reuse after geometry edits.**
- ER is invariant; there are no retrained parameters.
- State rays: the existing selective retrace.
- Proposal cache: invalidated by the rules below.
- ℛ: explicit sweeps are global (the resolvent has no edit locality [INF WL86B (ii)]). A learned R_θ recomputes its generator globally, as today.

**Cache invalidation and update rules.**
1. Light l moved → all entries for l. Intensity or colour change only → rescale (linear), no retrace.
2. Peaked plane P moved or deformed → all entries of P (recompute y*).
3. Receiver point x_j moved → entries of j.
4. Any proposal segment (y_i–p_l or y_i–x_j) crossing old ∪ new OBB of a moved object (1 mm pad, the `ecro86.py:57-67` rule) → retrace that visibility bit only.
5. Material change → re-evaluate ER values, no retrace.
6. Non-rigid edit without a validity certificate → conservative retrace of every entry touching the changed geometry (canon Architecture.md:62-64).
7. B₂ rows are recomputed for affected receivers, which captures stationary transport-dirty receivers (D_T ≠ D_G). ℛ then runs globally.

**Training targets and supervision semantics.**
- Sharp first order: none. ER is frozen and already qualified (WL84) on one coated-conductor family.
- Learned ℛ (optional):
  - input = the physical B₂ / E₂ from 𝒜 on the training geometries (input-consistent);
  - target = the separately tallied t_{≥3} (`pt86a.field_src`, k ≥ 3);
  - reader-importance weights; no sum-only supervision;
  - admissible only if it beats explicit sweeps on the same state at matched cost (WL86B criterion).
- There is no learned first-order partner whose deficit ℛ could absorb, so the L8 compensation channel is closed for this family by construction [INF].

**Inference-time physical computation (explicit about ray tracing).**
- Per source configuration, 𝒜 *is* ray tracing: per valid (j, l, P) it traces 2·M rays (shadow to the light, visibility to x_j) and makes M ER evaluations.
- It also needs NEE probes for the uniform state rays that land on peaked senders (cand91-style probe + shadow).
- Per frame: unchanged (LOW + table lookup). No extra rays at render time. The cost lives in the per-edit and per-source path, not the per-frame path.

**Memory and compute** [INF estimates; anchors are MEAS on an RTX 3080 Ti, WL90/91/86].
- Receiver points N_r = 1,270 × 16 = 20,320.
- Analytic culling: N_r · L · P_planes, plus a per-patch tangent-plane test for curved peaked patches. With the sphere's 32 patches: N_r · L · 33 ≈ 2.7 M tests at L = 4, which is trivial.
- Rays, worst case (every point valid, M = 16): 20,320 × 16 × 2 ≈ 0.65 M rays per light for the panel plane.
- X1 reads restricted to peaked landings: ≤ 0.53 M per light, WL91's all-landings count [MEAS], so in practice fewer.
- The anchor is X1 exact evaluation at 11–14 ms for 0.26–2.1 M rays [MEAS WL91:256]. Expected 𝒜 cost is therefore ≈ 10–30 ms per source configuration [INF].
- Memory: proposal cache ≤ 20 MB worst case at L = 4, M = 8. B₂ sub-patch tables ≈ 4 MB (330 glossy patches × 16 × 64 × 3 floats) + 0.2 MB (Lambert scalars). Existing state 17.6 MB [MEAS].
- Per edit: dirty proposal segments, about the 8.7–10.2 % dirty-ray fraction of WL86 [MEAS] unless a peaked sender or a large-footprint receiver moves, plus ℛ (explicit EXP-like sweeps 89–95 ms per edit [MEAS WL87], or learned ≈ 11 ms + apply).
- Worst-case scaling:
  - O(L · N_r · P) culling without hierarchy; O(L · (P log N_r + Σ_P footprint samples)) with a frustum BVH.
  - Ray cost is output-sensitive to total caustic footprint area × M.
  - For a scene dominated by many small curved peaked patches, P → N_peaked and the method becomes all-pairs sharp transport, i.e. a path tracer for that family. This is declared, not hidden.
  - Area or environment lights have no compact virtual source; the stationary point becomes a region and variance returns. The design is for compact emitters (point, spot, small area).

**Conservation and reciprocity.**
- 𝒜 is an unbiased MC estimator of the e2 integral given ER and geometry.
- MIS over the full sender domain keeps every (receiver, sender) contribution owned exactly once, whatever the peaked rule.
- No amplification (ER passive). Reciprocity holds in expectation (ER reciprocal); tables lose exact reciprocity after projection, as now.
- Explicit sweeps are passive.

**Why it addresses the failures.**
- L1, L2, L12 (incident projection) and L4 (outgoing bin and patch projection at the sender) are removed for the first-order family. L4 is the WL90 term, 97–102 % in GM1.
- L3 and L7 (the learned rank-16 tied operator, which is at zero level) are removed from the sharp path.
- L6: the cosine is evaluated at the exact direction.
- L5: replaced by a lobe-aware proposal. This targets the ≈ 1-hit Poisson mechanism of §2.2, which is the specific reason WL91 / R0X1 exact reads failed.
- L8: the compensation channel is closed for this family.
- L10: mitigated by sub-patch B₂.
- Not addressed: L9 and L11 inside ℛ; order ≥ 2 projection at glossy receivers; curved senders (only by fallback).

**Main risks.**
1. **Cost shift**: the sharp first order *is* ray-traced. Its learned content is nil. The learned share of the method shrinks to ℛ, whose advantage over explicit sweeps is unmeasured (WL87: explicit ≥ ECRO). A may therefore be judged the H-0 null plus a better first-gather estimator (owner decision).
2. The variance of the proposal under grazing, anisotropic, coated (refracted inner lobe) ER is untested.
3. Curved peaked senders: Alhazen-type stationary points can be multiple and unstable.
4. Non-compact lights.
5. Contract conflict if ℛ is learned (above).

**Smallest falsifying experiment:** §6. A is killed if H2-pool (exact reads, converged relation) does not match e2_p on CK2 box A, or if H3 does not cut single-draw variance relative to X1 at matched rays.

### 3.2 Hypothesis H-B: Persistent Virtual-Source Atom State (VSA)

**One-sentence mechanism.** The directional state of every patch is a two-basis object: a smooth 64-bin table plus a sparse list of **virtual-source atoms**.
- Atoms are created by an ER-lobe decomposition (analytic, from the 8-lobe closure) whenever a compact or atom incident field meets a sharp lobe.
- Atoms propagate by aperture-tested gathers and chain by reflection (image sources).
- Atoms deposit into smooth tables at broad scatterers.
- One local rule is applied to direct and indirect incidence alike.

**State representation.**
- Per patch n:
  - T_n ∈ ℝ^{64×3}, the smooth part;
  - A_n = {a_k = (p'_k, Φ_k ∈ ℝ³, Σ_k (2×2 angular blur in the tangent frame), aperture = patch / planar group n, parent id, order)}.
- For a planar sender lit by a point light, p'_k is the mirror image of p_l. The ideal-mirror outgoing radiance is then *exactly* a point source seen through the aperture [FACT of geometric optics]. Glossy lobes add the blur Σ_k.
- Second-order atoms are mirror images of first-order atoms across another peaked plane, with clipped apertures.

**Operator inputs and outputs.**
- Local rule Λ_ER: incident atom (direction μ_i at the patch, blur Σ_i, flux Φ_i) or smooth incident bins → per ER lobe k:
  - narrow lobes (α_k below a predeclared threshold) → an outgoing atom: mean = specular map of μ_i (refraction-remapped for coat-inner lobes per WL82 ER [INF]); Σ_o = J Σ_i Jᵀ + Σ_ER,k(μ_i); Φ_o = Φ_i · albedo_k(μ_i) per channel;
  - broad lobes → bin-projected remainder added to T_n.
- The lobe partition is exact: the lobe albedos sum to the ER albedo ≤ 1, so it is passive by construction.
- Propagation: atom a → receiver x gives E(x) = Φ_a cos θ_x / |x − p'_a|² · 𝟙[segment x→p'_a crosses the aperture ⊕ blur] · V(x, y*) V(y*, upstream).
  - For glossy blur, the aperture-local integral uses the same stationary-point proposal as H-A.
- Readout: at a glossy x₂, atoms are evaluated at the exact direction x₂→x₁, which removes the readout-vertex angular projection for sharp content.

**Local scattering ownership.**
- Frozen ER, decomposed analytically per lobe. Each lobe has exactly one owner (atom or table), so the H-P critique "moderate lobes have no owner" does not apply [INF].
- The threshold changes accuracy and cost, not ownership. It is fixed beforehand from a learning-free measurement of bin-projection error vs α.
- There is no learned local compiler. WL81 showed a fixed R = 8 *learned compiled closure* failed representation on thin-coat sharp conductors [MEAS WL81:194-217], so H-B deliberately consumes the frozen ER lobes as given rather than re-fitting them.

**Geometry-dependent propagation ownership.**
- Atom gathers own sender→receiver segments for atom-carried energy.
- The state rays own smooth-table transport.
- Disjointness holds because each lobe's energy lives in one carrier.

**Treatment of sharp angular structure.** It is stored persistently as atoms. The cost is independent of α; only the number of atoms matters.

**Diffuse and nonlocal transport.** Smooth tables with explicit sweeps or a learned ℛ, as in H-A. Atom deposits at broad receivers enter T.

**Reuse after edits.**
- ER is invariant.
- Atoms are source- and geometry-dependent state:
  - a rigid move of a non-peaked object only invalidates gathers whose segments cross its OBBs;
  - moving a peaked plane invalidates its atoms and their descendants (dependency tree via parent id).

**Cache invalidation.**
1. The same segment-OBB rule applies to gather visibility.
2. Atom subtree invalidation follows the parent-id dependency graph.
3. Light change → that light's atom forest.
4. Aperture clipping is recomputed for moved planes.

**Training and supervision.**
- No learned sharp component.
- The optional learned ℛ follows the H-A rules.
- The partition threshold is measured, not learned.

**Inference-time computation.**
- Atom creation is analytic and costs nothing measurable.
- Gathers: per receiver point × atoms whose frustum contains it → M visibility rays each.
- Chains: per atom × peaked planes intersected by its frustum.
- This is still ray tracing: at first order it costs the same as H-A, since the estimator is shared.

**Memory and compute** [INF].
- First order: L × (planar peaked groups + curved peaked patches) = 4 × 33 = 132 atoms (≈ 32 B each).
- Depth 2: ≤ 132 × 33 ≈ 4.4 k before frustum culling.
- Worst case: O(L · P^d) for depth d (image-source explosion) → requires a depth limit (≤ 2) and a flux threshold.
- Gather tests: O(N_r · #atoms) without a frustum BVH.
- Curved peaked patches need per-patch tangent-plane atoms (astigmatic, approximate).

**Conservation and reciprocity.**
- Flux conserving for ideal planar mirrors (aperture flux).
- Glossy blur kernels must be normalised to the lobe albedo.
- Merging atoms, if used, is a nonlinear projection, so H-B forbids merging within a source configuration to keep superposition exact.
- Reciprocity is approximate (blur model).

**Why it addresses the failures.**
- Everything H-A removes at first order, and in addition:
  - order ≥ 2 sharp chains (S→S) and the readout-vertex angular projection (part of L10 / L4 at x₂) for sharp content;
  - the narrow-beam information loss shown post hoc by WL86A (identical-encoding beam pairs differ by 0.94 / 0.62 against noise 0.26 / 0.22 [MEAS]).

**Evidence.**
- **Supporting:** Stage B (the bin basis is high-rank for S1; lobe-aligned is low-dimensional [INF]); WL86A beam pairs.
- **Weakening:** WL90 P2plus ≤ 0.0009 and isolation in all 4 GM1 regions, including the S1→S1 slab configuration [MEAS]. In the only workload with sharp chains, post-first-gather projection was negligible, so H-B's additional machinery has no measured target yet. WL84A withdrew DDT and LPTG over cost and retracing concerns that partly apply here.
- **Inconclusive:** S→S share outside GM1; curved senders.

**Main risks.**
- Complexity and debuggability (frusta, aperture clipping, dependency trees).
- Atom growth.
- Curved and non-rigid peaked geometry.
- Atoms only from compact emitters. A spatially sharp *broad* emitter (a lit spot reflected by the panel) stays in the smooth basis.
- The same cost-shift risk as H-A, with more machinery.

**Smallest falsifying experiment** (after §6 passes):
1. On the GM1 W1 slab configuration, measure the post-A residual split by the number of peaked vertices (≥ 2) from the path-traced reference. If the S→S-attributable residual is within noise (≤ the noise bound) in R_mirror / C_ss and in CK2, H-B's added claims have no target, and H-B is rejected in favour of H-A.
2. Otherwise, test the atom gather at depth 2 against the reference on those regions.

### 3.3 Alternatives considered and rejected

- **R1. Uniform angular / spatial refinement of the tables (for example 4×4 sub-bins everywhere), with or without a larger learned rank.**
  - WL90's 16 sub-bins close the GM1 explicit gap [MEAS], but storage and relation rays scale ≈ α⁻² per patch everywhere, including broad receivers. Fixed 4 sub-bins leave 9–20 % of J [MEAS].
  - The learned factorisation worsens with resolution for near-permutation structure [INF from Stage B: rank ≈ 32 already needed at 64 bins].
  - It still point-samples inside refined cells (L5 unchanged).
  - Kept only as the directional-discretisation control arm (§6).
- **R2. A learned continuous-direction or higher-rank learned sender** (neural L(y, ω) per patch, per-patch position conditioning, rank 32).
  - WL93D: the S1 sender error is geometry-common (S_geo < 1 %), so there is no geometry content for it to own.
  - Stage B: per-patch conditioning over-fits (0.263 → 0.451).
  - Learned senders sit at the zero-operator level against a 0.36 bound.
  - It would duplicate ER with extra co-adaptation risk. No physical justification.
- **R3. Sender-side lobe-sampled deposition (photon-style C3 with ER-lobe directions and kernel density at receivers).**
  - The receiver kernel bandwidth trades bias at 2 cm-wide footprints and near-contact against variance.
  - The tested C3URX1 (bin-uniform directions) failed held-out on variance [MEAS WL91].
  - Kept as a comparator, not a hypothesis.
- **Also rejected:**
  - Per-frame retracing of the H-A estimator: the LPTG critique, which moves cost into every frame.
  - Learned dense C (WL52: cost and locality).
  - Online per-scene neural radiance caches (source-dependent fitting, not reusable transport).

---

## 4 Physical and Computational Trade-offs (Evidence D)

### 4.1 Ten-criteria comparison

Columns:
- **ECRO-OWN:** the current lineage.
- **NULL:** the immutable explicit comparators EXP / C3UR / C3URX1 on the same state.
- **H-A** and **H-B:** this report.

| Criterion | ECRO-OWN | NULL | H-A | H-B |
|---|---|---|---|---|
| Angular representability (sharp) | Rank-16 bins: learned at zero level [MEAS]; bound 0.36 [MEAS] | Bin q: WL90 J; X1 exact reads [MEAS] | Exact at the first order (continuous ER) [INF]; bins after a broad event | Exact for ideal planar mirrors; blur model otherwise [INF] |
| Propagation fidelity | κ = 8 fixed, bin reads; CK2 floor confounded [MEAS] | Pooled ≈ oracle; single draw heavy-tailed [MEAS] | Unbiased MIS; variance reduction **untested** [HYP] | Same first order; chains untested [HYP] |
| Material generalisation | mat_mlp compresses albedo (WL94) [MEAS] | ER direct [FACT] | ER direct, no retraining [INF] | ER lobes direct [INF] |
| Dynamic geometry reuse | Weights reused; generator global [FACT] | Selective = full [MEAS] | ER reused; selective proposal and state updates [INF] | Atom dependency trees [INF] |
| Physical correctness | Not passive by design; compensation [MEAS] | Unbiased tables; EXP passes WL86 gates [MEAS] | Unbiased first order; projection error later [INF] | Blur and curved approximations biased [INF] |
| Training feasibility | Seed-dependent, co-adaptation [MEAS] | None | None for sharp; optional ℛ with per-term target | None for sharp |
| Inference and update complexity | 76–78 ms per edit; 0.33 ms/spp [MEAS] | EXP 89–95 ms; C3URX1 edit-to-frame 125 ms [MEAS] | ECRO-like + ≈ 10–30 ms per source configuration [INF] | ≈ H-A + chains [INF] |
| Memory scalability | ≈ 43 MB factors, Ft N² [MEAS] | State + tables | + ≤ 24 MB [INF] | Small atoms; O(L P^d) risk |
| Maintainability / debuggability | Entangled terms [MEAS] | High | High: each term tallied against e2_p | Lower (frusta, trees) |
| Usefulness vs full recompute (PT) | 5× edit-to-first-frame at 0.03; fails region C at g [MEAS WL87] | Same | Keeps LOW + table per frame; aims at region C at g [HYP] | Same aim, higher complexity |

### 4.2 Cost-shift assessment (explicit)

- Both H-A and H-B move the sharp first-order family *from a learned table that fails* to **explicit ray-traced evaluation**. This is a reassignment of ownership, not learned efficiency. It must be reported as ray cost, and I do not claim it as learned transport.
- The reusable learned elements are only:
  - ER, the local response, already learned and frozen;
  - optionally ℛ.
- Neither design builds large explicit tables: the extra memory is ≤ 24 MB plus atoms.
- Initial construction and edit update:
  - **H-A construction** = ECRO state + 𝒜 (≈ 0.65 M rays per light worst case on CK2-like scenes) + ℛ.
  - **H-A per edit** = dirty state rays + dirty proposal segments + ℛ (global). The edit-local saving is bounded because ℛ is global, as in ECRO today (WL86B (ii): 20.8 vs 28.1 ms).
  - **H-B** adds atom-tree maintenance.
- Worst case, for both: compact-light count L × total caustic footprint. Scenes with many curved sharp patches or area / environment lights degrade toward path tracing of the sharp family.
- Scaling assumptions: compact emitters, planar or near-planar peaked senders, peaked share ≪ N.

---

## 5 Recommended Hypothesis

**Primary hypothesis for the next falsification: H-A (LCFG), with an explicit regular remainder by default.**

- **Mechanism:**
  - sharp first-order transport = an unbiased MIS estimate of e2 with exact-direction ER reads and a stationary-point lobe proposal toward planar peaked senders;
  - cached at receiver sub-patches per source configuration;
  - remainder = same-state sweeps.
- **Minimum evidence already in hand:**
  - WL90: first-gather angular projection is 97–102 % of the gap; ≥ 2 orders ≤ 9 %.
  - WL91: pooled exact reads equal the oracle in 6/6.
  - WL88: coverage and placement mechanisms.
  - WL93C / 93D / 94: an exact sender always helps renders.
  - Stage B: the learned sender holds nothing to preserve.
- **Missing:**
  - whether variance falls under lobe-aware proposals;
  - whether CK2's floor is a propagation effect.
- **Falsification:** §6, conditions S1–S3.
- **Baselines:**
  - C0 (OWN s8601 / s8602);
  - C2_srcS1;
  - EXP, C3UR, C3URX1 (immutable comparators);
  - REF-ER path tracing at matched quality (re-render role, not Neural Full Recompute);
  - STALE (T0 state at T1);
  - same-mechanism full rebuild vs selective (bit-equality).
- External Neural Full Recompute (8DNA / RNA retraining) is **not in scope**. Any external-superiority claim is UNMEASURED (AGENTS mandatory gate).
- **Smallest workload:** CK2 "none" T1, light 0 (existing e2_p reference); E2 T1/T0 region C (existing WL86 REF-ER); GM1 W1 T1 R_mirror (existing WL89 REF-ER, calibration).
- **Smallest implementation boundary:**
  - an isolated offline evaluator, i.e. a planar stationary-point proposal + MIS first gather with ER evaluation and Mitsuba visibility;
  - reused unchanged: `cand91` radiance(), `rel88` pooled relations, `quad87.qtable`;
  - the true-layout W_T weight;
  - no change to ECRO production, no training, no checkpoint modification.
- H-B is **not** recommended unless §6 decision row D4 triggers it.
- If the owner requires the sharp first-order path itself to be learned, then on this evidence I find no learned mechanism justified for it (§2.1, R2).

---

## 6 Falsification Experiment (Evidence E; learning-free)

### Matched quantity

Receiver exitance of the panel-mediated exactly-2-event family:
- Σ_d B₂[n, d] · W_T[d] per patch, and per 4×4 sub-patch where a reference exists;
- compared against e2_p from `CKF_none.npz` (2 × 4,096 spp halves).
- Glossy receivers use the same bins and the same W_T. `an94.W` must not be reused.
- Regimes stay labelled: CK2 ROI_T, ECRO region C and GM1 R_mirror are not cross-comparable.

### Oracle hierarchy

Every level is well defined with existing code.

| Arm | Sender | Flight / relation | What it isolates |
|---|---|---|---|
| H0 | learned Bl (C0) | deployed | local sender representation error |
| H0b | Stage B L1-NMF16 bound operator | deployed | capacity locus at matched norm |
| H1 | exact M64(s) | deployed | WL94D floor, re-measured with W_T |
| H1q | exact q (`quad87.qtable`; L1/L2/L12 removed) | deployed | C3 (incident-side projection) |
| H1q-sa | q at 4×4 sub-bins | sub-bin reads | C1 (angular) |
| H1q-sp | q at 4×4 sub-patches | deployed | C2 (spatial) |
| H1q-pool | q | 64 pooled realisations | C5 (sampling) |
| H2 | exact per-ray read (X1) | deployed relation | L4 fully removed |
| H2-pool | X1 | pooled relation | ≈ the WL90 EE level: "exact sender + higher-fidelity propagation" |
| **H3 (candidate)** | X1 + stationary-point lobe proposal, MIS | single deployed realisation at a declared ray budget, plus 8 independent realisations | the candidate itself |
| REF | path-traced e2_p | — | independent physical reference |

### Three error classes, kept separate

1. **Local sender representation:** H0 / H0b vs H1q, field metrics against q with W_T.
2. **Propagation / flight:** H1q → H1q-sa / sp / pool → H2 → H2-pool vs REF.
3. **Integrated sharp interreflection:** rendered E_H on CK2 ROI_T and E2 region C.
   - Use B = arm first order + path-traced t_{≥3} to isolate the first order (WL92-ladder style).
   - For the full H-A system, use arm + explicit sweeps.
   - Report signed values and edit slopes.

### Controls

- **Reference noise:** half-split for e2_p; noise_bound for E_H.
- **Positive control:** the non-panel 2-event family (e2_o) on box A. Every arm from H1q to H3 must match within noise.
- **Geometry:** CK2 vs E2 vs W1.
- **Material:** S1 planar sender vs S2 sphere (curved; H3 falls back to p_u, reported, not gated).
- **Lighting and incident angular distribution:** CK2 grazing (ring 0) vs one non-grazing CK2 light variant (needs one new 2 × 4,096 spp e2_p reference); ring shares reported.
- **Directional discretisation:** 64 bins vs 4×4 sub-bins.
- **Model capacity:** H0b.
- **Compute:** H2 vs H3 at equal *total* rays (relation + probe + shadow + proposal); every ray and every ms is reported per stage.

### Stop and acceptance conditions

The numeric values below are proposals, to be frozen before results.

- **S0 (validity).** The positive control fails: ratio outside [0.97, 1.03] or rel L1 > 2 × half-noise → stop, implementation invalid.
- **S1 (premise; kills H-A and H-B first order).** H2-pool on CK2 box A:
  - fails if ratio is outside [0.95, 1.05], or rel L1 > 2 × half-noise (≈ 0.116) + 0.02, or corr < corr_halves − 0.05;
  - outcome: the CK2 floor is not first-gather projection or sampling. Attribute it to semantics, ER at grazing or the reference, and stop.
- **S2 (mechanism; kills H-A).** At matched total rays over 8 realisations, H3 fails if:
  - the region-C / ROI_T E_H SD is > 0.5 × H2's, or
  - its mean signed error is outside the noise bound of H2-pool, or
  - the deployed draw is worse than C3UR (the WL91 failure pattern).
- **S3 (utility; kills practical value, not correctness).**
  - It fails if H3's added per-edit GPU time exceeds ECRO's per-edit 76–78 ms, or
  - if H3 is slower than the best immutable explicit comparator at the predeclared E_H tier (WL91-style rule a), or
  - if it does not reach a region-C level that full path tracing reaches faster at matched quality.
- **Acceptance (H-A first-order contract supported, scope-limited):**
  - S0–S2 pass on CK2, E2 T1/T0 and W1 T1;
  - E_H in E2 region C ≤ 0.06 for a single draw;
  - edit slope in C within 0.75–1.25.
  - The scope remains one room family, rigid box edits and compact lights.

### Decisions enabled

| Row | Outcome | Decision |
|---|---|---|
| D1 | The H1 → H2-pool ladder | Decomposes the open CK2 floor (C1–C6) into angular / spatial / sampling / incident-projection shares, resolving auditA §3 |
| D2 | S1 fails | Propagation-first redesign is not the lever; selection reverts to premature |
| D3 | S1 passes, S2 or S3 fails | Correct but impractical; neither H-A nor H-B (they share the estimator) |
| D4 | All pass | H-A first-order contract supported. Then measure the residual attributable to paths with ≥ 2 peaked vertices: within noise → H-B unnecessary; beyond noise → H-B's chain test is admissible |

**Batch class (for the owner):** a new mechanism hypothesis, so by the AGENTS policy priority (1) a regular numbered batch. No successor is started or authorised here.

---

## 7 Unresolved Questions

1. Is a learning-free sharp first-order component admissible inside the published learned method, or is it the H-0 baseline? This is an open owner decision in canon (60_coordinator_synthesis; WL86B).
2. Is a learned ℛ for indirect incidence compatible with Architecture.md:21 ("no direct/indirect response branch") when ER answers direct incidence? ECRO v0 already has this split, and canon has not placed ECRO relative to K/C.
3. Is a learned ℛ worth anything over same-state sweeps (WL87: explicit ≥ ECRO)? A CPU field-adapted bound on smooth E₂ fields at the panel (as in Stage B) could pre-screen this.
4. The variance of stationary-point proposals for grazing incidence, coat-refracted ER lobes and off-specular peaks.
5. Curved peaked senders (S2), non-rigid deformation and near-contact (mm) gaps, where WL54 / WL55 require spatial resolution on both sides.
6. Area and environment lights; textured peaked senders (`field_src` uses SMAT, not PMAT; textured variants are unverified, auditA C8).
7. The S→S family share beyond GM1, and sharp-mover edits (no workload yet; WL86B gap #7).
8. The impact of the `an94.W` defect on the WL94-series learned and render-derived numbers (bounds ≤ 0.02; the rest unmeasured).
9. Economics at the g-level against path tracing, and external Neural Full Recompute (UNMEASURED).

---

## 8 Architecture Verdict

**MULTIPLE_CANDIDATES_REQUIRE_DISCRIMINATION.**
- H-A (LCFG) is the primary hypothesis to falsify next.
- H-B (VSA) is conditional on decision row D4.
- The explicit NULL (EXP / C3UR / C3URX1) remains the comparator.
- [INF] Propagation-first reading of the evidence:
  - The sharp failure is located at the *first gather*: angular projection of the sender plus receiver-side sampling of a ≈ 1-hit stationary region.
  - It is **not** located in learned geometry content.
  - The learned S1 sender should therefore not own the sharp first order.
- Whether exact-direction, lobe-aware propagation closes the CK2 floor at acceptable variance and cost is **untested**. Until §6 discriminates, no architecture is adopted, and no successor batch is started.

---

## 9 Reproducibility / Sources

- **Evidence files** (scratchpad `C:\Users\dna10\AppData\Local\Temp\claude\462fa2b2dbb71cdd\46d65e15-e083-4057-875f-9282fb68aa5e\scratchpad\wl95\`): `auditA.md`, `auditB.md`, `read_code.md`, `read_canon.md`, `read_wl86_88.md`, `read_wl89_92.md`, `read_wl93.md`, `read_wl94.md`.
- **Code** (c:\Projects\ReST-GI):
  - `experiments/ecro86/model86.py:109-124` (apply / flight / global)
  - `experiments/ecro86/pt86.py:153-156` (bins), `:353-402` (state, source)
  - `experiments/ecro_mat94/an94.py:19,32` (W defect)
  - `experiments/ecro_s94d/s94.py:152-160` (CK2 floor)
  - `experiments/gm1_c3urx1_91/cand91.py:1-110` (X1: bin-uniform relation directions, NEE per landed ray)
  - `experiments/gm1_relation88/rel88.py:5-15` (SAM MIS density)
  - `experiments/gm1_explicit87/quad87.py:85` (fbar), `:240` (enc_matrix), `:268` (qtable)
  - `experiments/ecro86a/pt86a.py:73-74,175-199` (e2_p split)
- **Docs:**
  - `docs/Architecture.md:17-38`
  - `docs/worklogs/90_WORKLOG_GM1_INTERMEDIATE_PROJECTION_GAP_ATTRIBUTION.md:150-206`
  - `docs/worklogs/81_WORKLOG_COMPILED_NEURAL_MATERIAL_RULES_UTILITY.md:186-250`
  - `docs/worklogs/82_WORKLOG_INCIDENT_ENERGY_VS_REFRACTION_ANGULAR.md:29-40`
- **New numbers in this report** are inferences, not measurements:
  - stationary-region width (§2.2);
  - expected hits per receiver point;
  - ray, memory and cost estimates (§3, §4).
  - All measured values are quoted with their source and norm.
