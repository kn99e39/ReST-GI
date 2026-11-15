# WL96 design report: spatio-angular transport carrier (design only)

**Scope.** I edited no repository file, trained nothing and ran no GPU or renderer job. The only computation was CPU numpy, and every script and result is in the scratchpad (paths in §9).

**Starting point.** The brief's `YOUR STARTING POINT: {ANGLE}` arrived as an unfilled placeholder. I chose the angle myself: which spatio-angular **carrier** (what is stored, where, and at what resolution) can hold the sharp, position-dependent S1 lobe, and where a learned component can still add something. If the supervising session meant a specific angle, this report may overlap another designer's.

**Labels.** [FACT] verified in code or docs · [MEAS] measured value with its source · [MEAS-96] my new CPU calculation, scratch only, not frozen evidence · [INF] inference · [HYP] untested.

## 0. Bottom line

1. **[MEAS-96] There is now a learning-free, noise-free oracle for e2_p.** It is a deterministic area-form quadrature of exactly-2-event S1 → box A delivery, using a numpy port of ER.
   - Against the path-traced CK2 e2_p it gives ratio 1.015 and rel L1 0.068. The reference half-noise is 0.059.
   - The ER port matches the stored M64 operator to ≤ 0.4 % on incident rings 1–3. The ring-0 rows are converging to it (0.81 → 0.94 as the outgoing quadrature is refined).
   - With no relation sampling at all, it reproduces WL95A's binned-sender over-delivery: M64 binned outgoing gives 1.486 / 0.487, against WL95A pooled M64_spool 1.495 / 0.500.
   - So the binned bias belongs to the carrier, not to sampling.
2. **[MEAS-96] The bias is cheap to remove; the variance is not.**
   - Keep the incident side of a sharp sender as a carrier (Φ = I·V/d² and its direction, interpolated from nodes) and let ER evaluate the outgoing direction exactly. On CK2 the delivery error is then 0.124 / 0.038 / 0.011 rel L1 at 20 / 10 / 5 cm node spacing.
   - The two extra lights (L2, ring-2; L3, ring-1) give ≤ 0.025 even at 20 cm.
   - The variance of a receiver-anchored gather (ECRO / X1: 512 rays, 8 per bin) does **not depend on the carrier**. The predicted per-patch variance is the same for the exact, node and fine-binned carriers.
   - On CK2 the half-energy solid angle of the glint, seen from the top box-A patches, is 0.0056 sr. That means 0.46 expected relation rays per draw. A Gaussian approximation needs about 530 draws to reach SE-L1 0.059; L2/L3 need about 25.
   - The binned M64 table has about 3× lower variance precisely because it smears the lobe.
3. **[MEAS-96] Sender-directed lobe quadrature removes the variance at low ray cost.**
   - Setup: per light, directions drawn from each sender cell's exact outgoing-flux distribution (ideal lobe importance, stratified).
   - CK2, 5 cm cells × 64 directions = 43 k photons per light: per-draw rel L1 median 0.068 (max 0.074), bias 0.020.
   - CK2, 2.5 cm × 64 = 172 k photons: per-draw 0.034 (max 0.040).
   - L2/L3 at 43 k photons: per-draw ≤ 0.028.
   - For comparison, WL95A's X1 has a single-draw median of 0.62, and its pooled 0.111 (vs PT) costs 64 × 9,995 × 2 ≈ 1.28 M rays.
4. **[MEAS-96] A compact, source-independent, material-level *learned linear* sharp operator is not viable on this evidence.**
   - Take the continuous ER S1 kernel: 1,024 incident cells × 16,384 outgoing cells, flux-weighted SVD.
   - It needs rank 148 / 381 / 614 for 90 / 99 / 99.9 % energy.
   - CK2 delivery error at signed rank 64 / 128 / 256 is 0.51 / 0.28 / 0.17, against 0.056 untruncated.
   - The truncations are signed (negative mass 9–34 %), so a non-negative factorisation can only be worse at the same rank.
   - This is consistent with the supervisor's correction: rank 16 is exact *for ring-0-supported inputs*, but a light-independent operator must cover the whole hemisphere.
5. **Consequence [INF].**
   - The only viable carrier I can specify (Candidate 1, SIC-R) owns the sharp first order with ER and geometry. That part is **explicit GI under another name**, in the G1/H-0 class, even though its estimator differs: a cached incident carrier plus sender-directed deterministic quadrature, with no per-relation-ray NEE.
   - Its learned component is confined to the broad ≥ 3-event resolvent Λ_θ, a concrete specification of WL95's unspecified G2 Λ_θ. Whether Λ_θ adds utility over explicit sweeps is open and must be tested.
   - Candidate 2 (E-L+, the ECRO-faithful learned sharp operator) is specified so it can be killed. Its capacity kill condition is already essentially met learning-free (point 4).
   - If the owner rules (Gate A, `docs/Architecture.md:13`; `docs/RESEARCH_ROADMAP.md:128`) that an explicit sharp first order cannot be part of the method, I find **no viable learned carrier** for sharp first-order transport on this evidence.
   - I make no novelty claim. Incident-side caching, sender-side importance "photon" deposition and PRT-style transfer are established graphics ideas.

---

## 1. Verified facts used

- [FACT] ECRO: the source is s = patch-mean direct irradiance per incident bin from 16 fixed points (`experiments/ecro86/pt86.py:378-402`). Then u = c·s, Bq = a·u (rank ≤ 16, patch- and bin-constant). The single flight runs over 512 fixed state rays with rw = z·ΔΩ/8 (`experiments/ecro86/model86.py:40-46, 109-124`). The global term is rank 24 and driven by u (`model86.py:121-123`). Readout uses the nearest patch and bin (`pt86.py:224-230`).
- [FACT] The generator inputs are geometry, ER states and state rays; the light is never an input (`model86.py:91-107`; `experiments/ecro86/protocol/ecro86.json:11`).
- [FACT] ER: `eval(wi, wo) = f·cos θ_o`, Lambert + 7 anisotropic GGX lobes, 4 inner lobes refracted by η_e, exact sample/pdf (`experiments/fnto_compiled83/closure83_dr.py:1-16, 121-155`).
  - S1 state (`experiments/ecro86/data/mats86.npz` row 1): α_u ≈ 0.026–0.075, η_e = 2.50, Lambert weight 8e-4.
  - The lobe weight sits mainly on outer lobe 0 (0.57, f0 0.007) and inner lobes 5–6.
- [FACT] X1 / X1c (`experiments/ecro_attr95a/a95.py:90-135`): ER eval at the exact light direction and the ray's own outgoing direction, shadowed, at the hit point (X1) or at the panel-patch centre (X1c). Each landed ray costs 1 probe intersection + 1 shadow ray per light.
- [FACT] C3URX1 (`experiments/gm1_c3urx1_91/cand91.py:36-110`) uses the same per-ray exact L₁ on a sender-side relation.
  - [MEAS] 526,467 probe + 526,467 shadow rays per light; source stage 11.5 ms (GM1) (`docs/worklogs/91_*.md:256, 363`).
- [MEAS] ECRO costs: edit 76–78 ms; source-only 13–18 ms; EXP source-only 51–64 ms (`docs/worklogs/87_*.md:281`; WL95 §1.1).
- [FACT] Canon:
  - ECRO is the learned-global base; EXP/C3UR are comparators; X1 findings "do not justify converting the project to explicit GI" (`docs/Architecture.md:13`).
  - The next step must be "one falsifiable ECRO-lineage … hypothesis" (`docs/RESEARCH_ROADMAP.md:128`).
  - No "exhaustive explicit per-surface full path tracer hidden under a neural name" (`docs/FNTO_ARCHITECTURE_CONTRACT_V0_1.md:86`).
- [FACT] Geometry:
  - The panel is 1.2 × 1.4 m at x = −1.5, normal +x, 6 × 7 patches of 20 cm (`experiments/ecro86/scene86.py:29-39`; `ecro86.json:17`).
  - CK2 box A sits at c = (−1.08, 0.6, 1.0), half sizes (0.2, 0.45, 0.45), yaw 0.15. The CK2 light is at (−1.45, −0.6, 1.7), 5 cm in front of the panel plane (`experiments/ecro_local93d/addendum_checker/scene_ck2.json`).
- [MEAS] (`results/ecro_attr95a/deliv.npz`) Reference e2_p on box A by face: panel-facing face 70.6 %, −y′ face 12.8 %, top face 16.5 %.

## 2. New CPU measurements [MEAS-96] (scratch; not frozen evidence)

**Method.** Exactly-2-event S1 → box A delivery in area form:
- E(x) = Σ_panel L₁(y, v) cos_y cos_x / r² dA.
- Panel grid 4 mm; 4 × 4 points per receiver patch; box-A exitance = 0.5 Σ_c E_c.
- Visibility: segment–OBB and segment–sphere tests for light → panel; convexity for panel → box A.
- Converged: a 1 cm / 2 × 2 run agrees within 0.003.
- **This is the expectation of every unbiased receiver-gather estimator of the corresponding sender radiance, with no sampling noise.**
- Lights:
  - CK2 (grazing, cos 0.02–0.24);
  - L2 = (−0.9, −0.9, 1.0), ring-2 incidence on the lit area;
  - L3 = (−0.8, −0.3, 2.5), ring 1.

  L2 and L3 have no path-traced reference; the oracle stands in for one, validated on CK2.

**Table 1. Carrier bias.** Box-A rel L1 vs the deterministic REF (ratio in parentheses).

| arm | CK2 | L2 | L3 |
|---|---|---|---|
| REF vs path-traced e2_p | 0.068 (1.015) | — | — |
| M64 binned outgoing (deterministic) | 0.464 (1.463); vs PT 0.487 (1.486) | 0.141 (0.96) | 0.169 (0.97) |
| X1c (patch-centre incidence, exact outgoing) | 0.269 (1.07); WL95A pooled X1c vs PT 0.268 | 0.051 | 0.049 |
| NODE 200 mm (incident carrier at patch corners, exact ER outgoing) | 0.124 (1.12) | 0.025 | 0.014 |
| NODE 100 mm | 0.038 | 0.006 | 0.004 |
| NODE 50 mm | 0.011 | 0.002 | 0.001 |
| BIC 64 (binned incident at bin centroid, exact outgoing; linear in s) | 0.512 | 0.237 | 0.235 |
| BIC 256 / 1024 / 4096 | 0.118 / 0.032 / 0.015 | 0.063 / 0.019 / 0.010 | 0.094 / 0.016 / 0.003 |
| Continuous ER kernel (1024 incident × 16384 outgoing cells), signed SVD rank 64 / 128 / 256 / full | 0.51 / 0.28 / 0.17 / 0.056 | 0.076 / 0.025 / 0.019 / 0.019 | 0.048 / 0.016 / 0.015 / 0.016 |

**Table 2. Propagation variance** (`var96.json`).
- Ω50 / Ω90 seen from the 9 brightest box-A patches: CK2 0.0056 / 0.099 sr; L2 0.025 / 0.155 sr; L3 0.031 / 0.193 sr.
- Expected relation rays per draw inside Ω50: 0.46 (CK2), 2.0 (L2), 2.5 (L3). This settles the WL95 §4.1 "≈ 1 vs 0.16–0.4" dispute at about 0.46 on CK2.
- Gaussian approximation of the per-draw abs rel L1, receiver-anchored 512-ray gather:
  - exact carrier: 1.36 (CK2), 0.31 (L2), 0.28 (L3);
  - M64 binned: 0.46 / 0.16 / 0.17.
- Draws needed for SE-L1 0.059: exact carrier ≈ 530 / 27 / 22; M64 ≈ 61 / 7 / 8.
- The exact, NODE200 and BIC1024 carriers have equal variance within 4 %.
- Caveat: the measured X1 per-draw median of 0.62 is below the Gaussian 1.36 because of heavy tails. The draw counts describe the mean-square, not the median.

**Table 3. Sender-directed lobe quadrature** (`photon2.json`).
- Setup: per light, each panel cell emits N_q directions drawn from its exact outgoing-flux distribution on a 96 × 384 grid; equal-flux photons; 8 seeds.
- Entries are per-draw rel L1 median / max, then the mean-of-8 error (bias).

| cells × N_q (photons / light) | CK2 | L2 | L3 |
|---|---|---|---|
| 5 cm × 16 (10.8 k) | 0.128 / 0.149, 0.057 | 0.041 / 0.047, 0.016 | 0.057 / 0.063, 0.018 |
| 5 cm × 64 (43 k) | 0.068 / 0.074, 0.020 | 0.021 / 0.024, 0.008 | 0.028 / 0.033, 0.010 |
| 2.5 cm × 64 (172 k) | 0.034 / 0.040, 0.017 | 0.009 / 0.013, 0.004 | 0.015 / 0.017, 0.006 |
| 2.5 cm × 256 (688 k) | 0.015 / 0.017, 0.008 | 0.004 / 0.005, 0.003 | 0.006 / 0.008, 0.004 |

Caveats [INF] on Table 3:
- The sampling uses an ideal flux distribution, not ER's VNDF `sample`, whose pdf differs from the flux and adds weight variance.
- Photons are scored on box A only; occluders other than box A are absent here.
- The panel is planar.
- A grid that keeps 99 % of each cell's flux would need 13 M directions per light: 99 % truncation is broad because of the inner refracted lobes. Importance sampling, not truncation, is the viable form.

---

## 3. Candidate 1: SIC-R (Sender-Incident Carrier, sharp Relay, learned broad Resolvent)

**Class honesty [INF].** The sharp first order is explicit, owned by ER and geometry: a G1/H-0-class mechanism with a different estimator. The learned part is a fully specified G2-class Λ_θ. SIC-R is "ECRO-lineage" only in keeping the source-independent learned global operator and the patch × bin interface. Whether that is admissible is owner Gate A.

### A. State representation

**Material-only**
- Frozen ER states (69 floats per material).
- A prospective "peaked sender" flag, from the WL88 rule median Π > 8. That rule makes S1 and S2 peaked [FACT, WL95 §3.1].

**Geometry-dependent**
- (g1) ECRO state rays: 1,270 × 512 = 650,240 rays, 27 B per ray, about 17.6 MB [MEAS WL86].
- (g2) Peaked-sender node grids with their frames:
  - S1 at h_n = 10 cm: 13 × 15 = 195 nodes;
  - S1 at h_n = 5 cm: 25 × 29 = 725 nodes;
  - S2 on its parametric grid.
- (g3) Relay emission cells on peaked senders: 5 cm (672 on S1) or 2.5 cm (2,688).
- (g4) Learned resolvent factors Λ_θ(G) from a generator of the ECRO type.

**Source-dependent (linear in each light)**
- (s1) The ECRO binned incident field s[N, 64, 3] (20,320 shadow rays per light).
- (s2) Node incident carrier at peaked senders. Two variants:
  - **linear**: fine-binned irradiance at 1,024 incident cells per node (≤ 2.4 MB at 195 nodes; sparse for point lights);
  - **moment**: per light atom, Φ = I·V/d² (rgb) plus the unit direction. This is exact for one light per atom and is not linear across lights in one cell.
- (s3) Relay deposits D[N, 64, 3]: incident irradiance per receiver patch and incident bin.

**Derived**
- E2[N, 64, 3]: incident tables after the first flight, from the sharp relay plus the broad first flight.
- E≥3 = Λ_θ[E2].

### B. Operator interface

1. Carrier: C_p(y) = Σ_atoms (Φ_a(y), ω_a(y)), interpolated from the nodes in the sender's material frame.
2. Sharp relay, per light: for each cell, draw N_q directions v from the exact lobe of L₁(y, ·) = Σ_a f_ER(ω_a, v)·Φ_a·cos... using ER `eval`/`sample`. Trace each to its hit (h, bin_h(−v)) and deposit the photon's flux into D[h, bin, c] / area_h.
3. Broad first flight: B_q = M_m·s at non-peaked senders (exact for Lambert; the floor-B binned operator was near-exact at rank 16 in WL95 §2.1), propagated through the state rays as in ECRO. It is added to D, giving E2.
4. Learned resolvent, linear, non-negative and source-independent:

   E≥3[n, m, c] = Σ_r P_θ[n, m, c, r] · Σ_{n'} area_{n'} Σ_{m'} Q_θ[n', m', c, r] · E2[n', m', c]

   with global rank ρ ∈ {24, 64} and P, Q = softplus(generator(G, ER states)).
5. Final scattering and readout are ER-owned:
   - non-peaked patches: B = M_m·(E2 + E≥3);
   - peaked patches: E≥3 at the patch (broad incidence) is relayed again by step 2, using its bin-averaged incident cells with exact outgoing. This produces the e3_p family explicitly.
   - The camera reads B as in ECRO. The higher-order readout at peaked x2 can use ER at the exact camera direction.

### C. Physical ownership (exclusive by event count and sender class)

- ER owns every local scattering evaluation:
  - exact (eval / sample) at peaked senders;
  - M64 bin projection at broad senders, with the L12 caveat;
  - exact final scatter at receivers.
- Geometry and visibility own the source shadow rays, the node visibility, the relay photons and the state rays.
- Learned Λ_θ owns only the composite ≥ 3-event transport between incident fields.
- There is **no learned local operator**: a·cᵀ, which duplicates ER, is removed.
- Path families:
  - e2 (both _p and _o) is explicit;
  - e3_o is learned (Λ output at non-peaked receivers, excluding incidence whose last sender is peaked);
  - e3_p is learned up to the peaked sender, then explicit relay.
- Mapping to K/C: K ↔ ER at each vertex; C ↔ relay and state rays; Λ_θ is a learned multi-flight C∘K composite, the same status as ECRO's global term. The event-count vs K/C-segment mapping gap of WL95 §1.4 is inherited.

### D. Learning

- **Mapping learned.** Λ_θ: E2 → E≥3, approximating Σ_{j≥1}(T_G S_ER)^j minus the explicit final relay at peaked senders.
- **Supervision.** From `pt86a.field_src(split=True)` tallies [FACT, `experiments/ecro86a/pt86a.py:73-200`]:
  - e3_o at receivers, compared through the frozen M_ER of the receiver (exact for Lambert);
  - at peaked-sender patches, the patch's own ≥ 3-event outgoing tables, compared through M64_S1.
  - The inputs E2 come from the SIC-R prefix on each training item.
  - [MEAS] Target noise is 0.10 rel L1 at 256 spp, heavy-tailed (WL86; WL93A).
- **What it can express beyond explicit ER + transport.** Physically, nothing [INF]. Its only possible utilities are:
  - (i) replacing J − 1 explicit sweeps by one low-rank application;
  - (ii) a smooth, deterministic higher-order field instead of the fixed-pattern sweep noise;
  - (iii) keeping ECRO's source-only update advantage for the broad part.
- **Prior evidence that headroom is small [MEAS, cross-regime].**
  - WL90 orders ≥ 2 contributed ≤ 0.0009 (GM1).
  - Sweeps take 11.6 ms (GM1) vs about 12.5 ms for the ECRO generator.
  - The learned source-only advantage is 3.5–3.9× (WL87:315).
- **Minimum meaningful capacity [HYP].** ρ ≈ 24–64, by analogy with ECRO's rank 24. It is meaningful only if its learning-free upper bound (§8 Stage 5) beats the explicit sweeps.

### E. Sharp-lobe preservation (relation to the WL95A ladder)

- WL95A ladder [MEAS]: binned M64 (1.54) → q256 (1.42) → X1c (1.04 / 0.268) → X1 (1.02 / 0.111).
  - The X1c → X1 step is the spatial projection: the light direction and 1/d² vary within a 20 cm patch, strongly near the light (d ≈ 0.2 m at the panel corner).
- SIC-R keeps the incident structure at h_n ≤ 10 cm. NODE100 sits between X1c and X1 at 0.038 bias on CK2 [MEAS-96].
- The outgoing side is never binned at peaked senders: ER is evaluated per photon direction.
- The relay replaces X1's receiver-anchored heavy-tailed gather (Table 2) by lobe-importance emission (Table 3: 0.034–0.068 per draw on CK2 at 43–172 k photons).
- The residual projection is at the **receiver**: deposits are binned per receiver patch × 64 incident bins. This is exact for Lambert receivers and an L1-type error for glossy receivers (§H).

### F. Dynamic reuse

- **Rigid edit of a non-peaked mover (box).**
  - Reused: ER; node grids; Λ_θ weights; clean state rays; the source of unaffected nodes.
  - Invalidated: state rays and relay photons whose segments cross the edit's swept OBB (the existing `ecro86.py:57-67` dirty test, extended to cached relay segments); the visibility of nodes whose light segments are dirty; all Λ_θ factors (the generator is global, as in ECRO).
  - Recomputed: the dirty relay photons per light; the deposits; the generator; E≥3.
- **Light change.** Recompute s (20,320 shadow rays), node visibility (195–725 shadow rays), the relay (cells × N_q photons) and E≥3. The Λ factors are reused.
- **Moving peaked object.** Nodes and cells move with the material frame. The whole carrier and relay are recomputed; the generator runs.
- **Non-rigid peaked sender.** Rebuild the node frames; the per-cell normals carry the curvature. No learned state needs a validity certificate beyond Λ, which is regenerated.
- **Self-contact.** Cells falling inside contact need sub-cell stratification (§H).

### G. Complexity (scene86 numbers; [INF] unless cited)

| item | SIC-R | explicit X1 (WL95A / C3URX1) | ECRO v0 |
|---|---|---|---|
| offline | Λ_θ training (ECRO scale: 1,024 geometries × 4 lights × 256 spp; WL86) | none | same as SIC-R |
| per light: rays | 20,320 (s) + 195–725 (nodes) + **43 k–172 k relay photons** | 9,995 probe + 9,995 shadow per draw; ≥ 64 draws for 0.111 → 1.28 M; Gaussian mean-square needs ≈ 530 draws on CK2. GM1 C3URX1: 0.53 M + 0.53 M | 20,320 |
| per light: compute | deposit scatter + Λ application O(N·64·3·ρ) ≈ 6 M MAC at ρ = 24 | ER eval per landed ray | ≈ 13–18 ms [MEAS] |
| per edit | selective state (21 ms [MEAS]) + dirty relay retrace × L + generator (≈ ECRO; edit 76–78 ms [MEAS]) | relation update + per-light X1 | 76–78 ms [MEAS] |
| memory | ECRO state 17.6 MB + Λ factors ≈ 45–120 MB (ρ 24–64; ECRO 43 MB [MEAS]) + cached relay segments 1.2–4.6 MB per light (27 B per ray) + carrier ≤ 2.4 MB | relation 20.8 + 12.6 MB [MEAS GM1] | ≈ 61 MB |
| scaling | relay ∝ L · Σ_peaked (A/h_c²) · N_q; independent of the receiver count | ∝ L · (relation rays landing on peaked senders) · draws | ∝ N² (dense Ft), L-light application cheap |

- **Avoided vs X1.** Per-light probe intersections and shadow rays on every landed relation ray, and the heavy-tailed variance. Roughly 7–30× fewer rays per light than pooled X1 at a lower per-draw error on CK2 [MEAS-96 + MEAS].
- **Introduced.** Per-light relay tracing (∝ L, unlike ECRO); receiver-side deposit binning; relay segment caches per light; the e3_p second relay.
- **Against ECRO.** SIC-R's per-light cost is 3–9× ECRO's ray count. The source-only advantage measured for learned ECRO (WL87) is **partly lost** for peaked senders by construction.

### H. Failure modes

- **α → 0.** Per-cell photons collapse to a single direction. Deposits then become a caustic-like image of the cell grid at the receivers (spatial aliasing). Cell size must be ≪ receiver patch, or a cell's photons must be jittered over its area. The bias floor at 5 cm was 0.020, mostly cell discretisation (2.5 cm: 0.008–0.017).
- **Glossy or sharp receivers.** 64-bin incident deposits re-introduce incident projection at the receiver (BIC64: 0.24–0.51 for a sharp field). Peaked receivers (sphere S2) need a node incident carrier too. Sharp-sharp chains (panel → sphere → …) need relay-of-relay.
- **Close contact.** Senders closer than the cell size misplace photon origins. The WL54/55 near-contact requirements apply.
- **Many lights.** Relay cost is linear in L. The moment carrier breaks when several lights fall in one atom, so the linear 1,024-cell carrier (CK2 0.032) is the safe default.
- **Area and environment lights.** Not tested. Broad incidence gives broad outgoing, where receiver-anchored gathering is adequate. A prospective switch rule (incident concentration) is needed. The importance distribution must integrate over the incident spread.
- **Moving sharp objects.** Full relay and carrier recompute per light.
- **Non-rigid peaked senders.** Planar node interpolation fails at high curvature; the nodes must resolve the normal field.
- **Unseen materials.** ER generalisation (WL83) holds only within its family. Λ_θ's embedding of material is an extrapolation risk; WL94 measured albedo compression for ECRO's learned terms.
- **Λ_θ.** Compensation risk: the WL86A global-term compensation pattern. It is mitigated because the sharp e2 is exact and Λ_θ sees only E2, not u.

### Mathematical definition, information retained and discarded, error sources

- **Definition.**
  - L₁(y, v) = Σ_a f_ER(y; ω_a(y), v) Φ_a(y), with Φ_a and ω_a obtained by node interpolation of exact light-visibility samples.
  - E2[h, m] = (1/A_h) Σ_photons flux_k · 1[hit_k ∈ h, bin_h(−v_k) = m] + (broad flight).
  - E≥3 = Λ_θ(G)[E2].
- **Retained.** Exact outgoing direction at peaked senders, sub-patch incident structure at h_n, the exact light direction (moment) or a 1,024-cell direction (linear variant), exact visibility of the relay photons.
- **Discarded.** Receiver sub-bin direction, sub-patch receiver position, incident spread within a moment atom, everything Λ_θ compresses at ≥ 3 events.
- **Error sources.**
  1. node interpolation (Table 1);
  2. cell discretisation and photon variance (Table 3);
  3. receiver binning;
  4. Λ_θ bias;
  5. binned incident at peaked senders for ≥ 3-event incidence (e3_p; unmeasured);
  6. sampling with ER's real sampler instead of the ideal distribution.

---

## 4. Candidate 2: E-L+ (learned, source-independent sharp operator with exact-outgoing readout); specified for falsification

**A. State.** Per material: a learned non-negative factorisation f_ER(ω, v) ≈ Σ_{r ≤ R} c_r(ω) a_r(v), with c_r and a_r small continuous networks or fine tables. The operator stays the ECRO u = c·s, but a_r is read at the **exact** outgoing direction of each state ray. It can be geometry-conditioned as in ECRO (WL93D measured S_geo < 1 %, so the material-level form is the relevant bound).

**B. Interface.** It keeps ECRO's interface exactly: linear in s, source-independent. The output L₁(y, v) = Σ_r a_r(v) u_r(patch(y)) is continuous in v and patch-constant in y.

**C. Ownership.** The learned operator owns the local response, duplicating ER. Geometry owns the flight.

**D. Learning.** Supervision is ER itself (unlimited, noise-free) plus e2 tallies. **It cannot express anything ER does not already provide.** Its only purpose would be keeping ECRO's linear source-independent algebra and cheap source-only updates.

**E. Sharp lobe.** It is limited by:
- incident binning of s: the 64-bin centroid error is 0.24–0.51 (BIC64);
- patch-constant u: the X1c-type spatial projection, 0.27 on CK2;
- rank: Table 1 rows 64–256.

**F. Reuse.** As ECRO. **G. Cost.** As ECRO, plus continuous-a evaluation per state ray.

**H. Failure.** Every row of Table 1 applies at once.

**Representability verdict [MEAS-96 + INF].** At material level and over the whole hemisphere, even a *signed* rank-128 kernel gives 0.28 on CK2 delivery, and rank 256 gives 0.17. 1,024 incident cells are needed for ≤ 0.03. Non-negativity and patch-constant u only make this worse.

**Predeclared kill condition E-F1:** non-negative rank-R material-level fit with CK2 delivery rel L1 > 0.15 at R ≤ 64 ⇒ FALSIFIED. Signed SVD already gives 0.51 at R = 64; a non-negative fit cannot be better in that norm under the same constraints [INF]. Only an NMF confirmation is outstanding (cheap, CPU).

### Rejected learned alternatives (not candidates) [INF]

| alternative | reason |
|---|---|
| Learned visibility or footprint (virtual-source transfer field, neural mirror footprint) | Learned C; WL52 measured 8–19× slower updates and failure at visibility transitions. The S1 → box-A footprint is transition-bounded (box edges). |
| Learned residual or denoiser on the binned table | A compensation mechanism (WL86A); non-physical; not a carrier. |
| Light-aware generator allocating rank per light | Would make rank 16 exact (`results/wl96_calc/ring_alloc.json`), but only re-learns ER's mirror placement and breaks source independence. |
| Source-independent receiver relation, dense | The variance requires about 530 draws (Table 2), i.e. about 345 M state rays. |
| Learned sampling density | Not a representation (WL95 §3.5). |

---

## 5. Comparison

| criterion | ECRO v0 / OWN | explicit X1 / C3URX1 | H-0 (EXP / C3UR) | SIC-R | E-L+ |
|---|---|---|---|---|---|
| sharp e2 bias, CK2 box A | learned 0.57–0.97 pooled [MEAS WL95A]; binned M64 0.49 | ≈ 0 (pooled 0.111 incl. noise) | q-binned 0.43 | 0.011–0.038 carrier + ≈ 0.02 cells [MEAS-96] | ≥ 0.17 (rank 256, signed) |
| single-draw spread | fixed pattern (deployed draw favourable) | heavy-tailed, median 0.62 | as M64 | 0.034–0.068 at 43–172 k photons | ECRO-like |
| learned component | local + global | none | none | Λ_θ (≥ 3 events, broad) | local sharp operator |
| source-only update | cheapest [MEAS WL87] | ≈ 1 M rays per light | 51–64 ms | 64–200 k rays per light + Λ | as ECRO |
| canon status | base | comparator | comparator | Gate A (explicit prefix) | ECRO lineage, falsified on capacity |

---

## 6. Predeclared falsification conditions for SIC-R (frozen before results)

- **S-F0 (oracle validity, STOP gate).** The deterministic e2_p oracle vs path-traced field_src must satisfy |ratio − 1| ≤ 0.03 and rel L1 ≤ n0 + 0.02 on every PT-validated light. Scratch value on CK2: 1.015 / 0.068 (n0 0.059), which passes.
- **S-F1 (carrier).** Carrier-only rel L1 vs oracle ≤ 0.05, box A and panel face, at h_n = 10 cm on ≥ 90 % of the held-out lights. NODE200 is not admissible (CK2 0.124).
- **S-F2 (relay).** With ER's **real** sampler at ≤ 50 k photons per light: mean-of-draws error ≤ 0.03 and per-draw p90 ≤ 0.06 on all lights. Otherwise test ≤ 200 k photons once (predeclared); fail beyond that.
- **S-F3 (generalisation).** ≥ 14 of 16 prospective held-out lights pass S-F1 and S-F2, including ≥ 3 grazing (ring 0), the two-lights-same-cell case (linear carrier) and the T0/T1 box edit.
- **S-F4 (cost).** Per-light update ≤ C3URX1 at matched error, and ≤ 3× ECRO source-only. Per edit ≤ 1.3 × ECRO (76–78 ms).
- **S-F5 (learned utility, decisive for "learned").** The learning-free upper bound of the Λ form (best non-negative rank-ρ fit, §8 Stage 5) must beat the explicit J-sweep recurrence on the same prefix:
  - by more than the render tolerance in held-out E_H, **or**
  - by ≥ 2× in cost at equal E_H.

  Otherwise Λ_θ has **NO_UTILITY**, and SIC-R reduces to H-0 with a better sharp estimator (an owner decision, not a method).
- **S-F6 (e3_p).** Relaying ≥ 3-event panel incidence from bin-averaged cells must reach ≤ 0.05 on panel-mediated e3_p (oracle: field_src e3_p). Otherwise the peaked-receiver carrier is insufficient.

---

## 7. Training-target availability

- e2_p / e2_o / e3_p / e3_o tallies come from `pt86a.field_src(split=True)` [FACT]. They exist for the WL92A-style split targets and for CKF_none (2 × 4,096 spp).
- The deterministic e2_p oracle (this report) can serve any planar peaked sender and any point light at negligible cost, for validation (S-F0/S-F1). It is not a training target for anything learned.
- quad87 q tables are binned and therefore unsuitable as the sharp target. M64 is the binned reference arm.

---

## 8. Smallest next feasibility experiment (no full training)

**Batch class.** A new carrier and ownership hypothesis, which by AGENTS.md priority (1) makes it a regular number. The E-L+ NMF confirmation alone would be a WL95A suffix. The owner decides.

- **Stage 0 (CPU).** Freeze. Parity of the numpy ER vs Dr.Jit ER on 10⁵ random pairs (≤ 1e-4). Port the scratch oracle into a diagnostics directory, with scene ray tests instead of OBB-only visibility. Extend it to all receivers, including the panel face.
- **Stage 1 (oracle, tests 1 and 2).**
  - Deterministic oracle on CK2 T1 plus one T0 edit, for CK2 plus 16 prospective lights drawn by a fixed rule: stratified by panel-centre incidence ring, 4 per ring, inside the room, not occluded by the generator rule. Add one two-lights-in-one-cell pair.
  - New path-traced field_src references (2 × 4,096 spp) for 2 non-CK2 lights (one ring 2, one ring 3) as the **independent validation condition** for S-F0.
- **Stage 2 (representability, test 1).** Carriers NODE{200, 100, 50}, BIC{256, 1024} and moment against the oracle (S-F1, S-F3). The E-L+ NMF ladder R ∈ {16, 32, 64, 128} on the continuous kernel (E-F1).
- **Stage 3 (bias vs variance, test 4; GPU, Dr.Jit, scene86).**
  - (a) Receiver-anchored gather of the node carrier at 1 / 8 / 64 draws.
  - (b) Relay with ER VNDF `sample`, cells 5 / 2.5 cm, N_q ∈ {16, 64, 256}, 16 RQMC shifts.
  - (c) M64 binned arm.
  - Report bias against the oracle and per-draw spread (S-F2). The Table 2 prediction is checked against the measured X1 spread.
- **Stage 4 (cost, test 5).** GPU timing with a full-scale utilisation probe, per the standing rule. Measure per-light update, per edit (fraction of dirty relay photons under the box move), and C3URX1-style X1 and ECRO source-only/edit on the same machine (S-F4).
- **Stage 5 (learned utility, test 6; no generator training).**
  - On the WL94A/WL86 training geometries, compute the SIC-R prefix E2 and the explicit J-sweep resolvent.
  - Fit (i) a per-geometry and (ii) a geometry-shared best non-negative rank-ρ global operator (ρ 24 / 64) directly by NMF, as a learning-free upper bound of the Λ form.
  - On held-out geometries compare: explicit sweeps / bound-Λ / ECRO global term, in delivery and in E_H on CK2 ROI_T and the E2 T1 region C, at matched cost (S-F5, S-F6).
- **Outcomes (frozen mapping).**
  - S-F0 fail → STOP.
  - S-F1/F2/F3 pass and S-F5 pass → SIC-R CANDIDATE_FEASIBLE (training authorisation is a separate decision).
  - S-F1–F3 pass and S-F5 fail → SHARP_CARRIER_EXPLICIT_ONLY (the learned part has no utility; Gate A decides).
  - S-F1 or S-F2 fail → CARRIER_INADEQUATE.
  - E-F1 met → E-L+ FALSIFIED.
- **Estimated compute.** CPU stages are minutes (this scratch run: about 2–5 min per calculation on 16 processes). GPU Stages 3–4 are about 10–20 GPU-min. PT references take 2 lights × 2 × 4,096 spp. Stage 5 is NMF on N × 64 × 3 matrices, CPU or GPU minutes.

## 9. Reproducibility (scratch)

Scratch folder: `C:\Users\dna10\AppData\Local\Temp\claude\462fa2b2dbb71cdd\46d65e15-e083-4057-875f-9282fb68aa5e\scratchpad\wl96\`

| file | what it is |
|---|---|
| `C:\Users\dna10\AppData\Local\Temp\claude\462fa2b2dbb71cdd\46d65e15-e083-4057-875f-9282fb68aa5e\scratchpad\wl96\er_np.py` | numpy ER port |
| `C:\Users\dna10\AppData\Local\Temp\claude\462fa2b2dbb71cdd\46d65e15-e083-4057-875f-9282fb68aa5e\scratchpad\wl96\val_m64.py` | M64 parity check |
| `C:\Users\dna10\AppData\Local\Temp\claude\462fa2b2dbb71cdd\46d65e15-e083-4057-875f-9282fb68aa5e\scratchpad\wl96\val2.py` | M64 parity check (ring-0 refinement) |
| `C:\Users\dna10\AppData\Local\Temp\claude\462fa2b2dbb71cdd\46d65e15-e083-4057-875f-9282fb68aa5e\scratchpad\wl96\deliv96.py` | oracle and carrier arms |
| `C:\Users\dna10\AppData\Local\Temp\claude\462fa2b2dbb71cdd\46d65e15-e083-4057-875f-9282fb68aa5e\scratchpad\wl96\deliv96_hp40_nr4.json` | Table 1 data (plus `deliv96_hp100_nr2.json` for convergence) |
| `C:\Users\dna10\AppData\Local\Temp\claude\462fa2b2dbb71cdd\46d65e15-e083-4057-875f-9282fb68aa5e\scratchpad\wl96\rep.py` | report script |
| `C:\Users\dna10\AppData\Local\Temp\claude\462fa2b2dbb71cdd\46d65e15-e083-4057-875f-9282fb68aa5e\scratchpad\wl96\var96.py` | variance prediction (output `var96.json`) |
| `C:\Users\dna10\AppData\Local\Temp\claude\462fa2b2dbb71cdd\46d65e15-e083-4057-875f-9282fb68aa5e\scratchpad\wl96\rank96.py` | continuous-kernel SVD (output `rank96.json`) |
| `C:\Users\dna10\AppData\Local\Temp\claude\462fa2b2dbb71cdd\46d65e15-e083-4057-875f-9282fb68aa5e\scratchpad\wl96\photon96.py` | truncation-grid photon count (partial; stopped once the count was clear) |
| `C:\Users\dna10\AppData\Local\Temp\claude\462fa2b2dbb71cdd\46d65e15-e083-4057-875f-9282fb68aa5e\scratchpad\wl96\photon2.py` | importance photons, Table 3 (output `photon2.json`) |

**Inputs, read-only:**
- `c:\Projects\ReST-GI\experiments\ecro86\data\mats86.npz`
- `c:\Projects\ReST-GI\results\ecro_attr95a\ops95a.npz`
- `c:\Projects\ReST-GI\results\ecro_attr95a\deliv.npz`
- `c:\Projects\ReST-GI\experiments\ecro_local93d\addendum_checker\scene_ck2.json`
- `c:\Projects\ReST-GI\experiments\ecro86\scene86.py` (imported)

**Limits of the scratch numbers.**
- One planar sharp sender and one Lambert receiver box.
- Box A only for L2/L3.
- No path-traced reference for L2/L3.
- Ideal lobe distribution instead of ER's sampler.
- No GPU timing.

None of this is frozen evidence. It is design input for the owner.
