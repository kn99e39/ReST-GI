# Design Proposal — Lobe-Split Intrinsic Transport with a Learned Smooth Higher-Order Response (LSIT)

**Status.** Independent architecture design review, written 2026-11-14 at the owner's request ("Independent Architecture Design Review", non-numbered, design only).

- Not adopted. Not canonical. Not implemented.
- No experiment, training, rendering or GPU job was run for this document.
- The canonical documents (`docs/Architecture.md`, roadmap) are unchanged.
- **Independence.** This proposal was derived from the worklogs and code listed in §12 only. The earlier assistant design panels (`docs/plans/gm1_architecture_review/`, `docs/plans/global_transport_study_86b/`, WL84A, WL86B) were deliberately **not** read before or while writing it.

Every statement is labelled:
- **[M]** a historical measurement, with its source worklog;
- **[I]** an interpretation of measurements;
- **[H]** a hypothesis of this proposal.

Predicted numbers are never presented as measurements.

---

## 1. Research objective and what the architecture must separate

**Objective.** Unchanged (Architecture.md §1): high-fidelity dynamic GI after supported geometry edits, with minimal recomputation and no retraining. A meaningful learned light-transport component and higher-order interreflection are part of the objective.

The information the system must carry, decomposed by **who owns it** and **whether it survives a rigid edit**:

| Information | Physical owner | Changes with a rigid edit? | Angular structure |
|---|---|---|---|
| Local scattering of each material (frozen ER) | intrinsic K (material) | no | broad part: low-rank in bins; **sharp lobe: near-diagonal, high-rank in absolute bins** |
| Incident direct field (lights) | source + current visibility | yes (shadows) / per source | exact direction per light sample; binned only in the deployed encoding |
| Free-space flight / visibility between surfaces | C (geometry) | yes, locally | deterministic ray geometry |
| Sharp two-event family (light → sharp surface → receiver) | K_lobe ∘ C ∘ e | yes | a **geometric map** (mirror reflection of the incident direction) plus a narrow lobe |
| Higher-order (≥ 3 events) transport | composition of K and C | yes, smoothly in most regions | broad / smooth after the first sharp event |
| Source / receiver role of a surface | none — a role is not a material property | — | the same K must serve both |

**Disagreement with a role-based decomposition.** The brief lists "source and receiver roles" as information to represent. This proposal treats them as **two uses of one intrinsic operator**, not two pieces of information.
- WL93B measured the cost of representing them separately [M]: the learned untied panel operators differ by 57–68 % on one material, and the unsupervised untied arm builds a non-physical decomposition.
- The proposal therefore draws the representational boundary by the **angular structure of the intrinsic kernel** (lobe vs smooth) and by **event order**, not by role.

## 2. Evidence ledger (inspected sources only)

| # | Evidence | Label |
|---|---|---|
| E1 | The patch × 64-bin output table is an adequate container: its oracle projection is at measurement noise, including sharp receivers (E2 region C 0.0137 vs noise 0.013). | [M] WL86 |
| E2 | Learned ECRO v0 is accurate on broad transport (stationary E_H 0.004–0.006). It fails on sharp receivers: E2 region C 0.063 / 0.072 vs rule 0.06; panel 0.034–0.056 vs oracle 0.012; C-region edit slope 0.36–0.63. | [M] WL86 |
| E3 | The learned single flight delivers only 1.6–18 % of the light → panel → receiver two-event family; the global term adds a geometry-blind compensation (+0.02…+0.06 of h). | [M] WL86A, WL92 |
| E4 | In frozen-ER flux norm, the panel material S1's 64 × 64 bin operator needs rank 46–59 for 99 % energy; its rank-16 residual is 0.46–0.52; material-level rank-16 NMF keeps only 47–54 % of the mirror bin. The floor needs rank 8; Lambert is rank 1. The learned generator's panel `a cᵀ` keeps 11–14 % of the mirror bin. | [M] WL92 |
| E5 | Per-geometry rank-16 free factors fitted at fixed geometry reach the sharp rule (OWN 0.017–0.041), but generalise over sources worse than the frozen generator. | [M] WL92 |
| E6 | Learning-free explicit transport on the same state is ≥ ECRO everywhere (E2 C 0.049 vs 0.063–0.072). The residual sharp deficit belongs to receiver-side state sampling (56–62 % of the light → panel → Lambert relation unsampled). Sender-side placement removes it at the same budget. | [M] WL87, WL88 |
| E7 | The explicit-vs-oracle fidelity floor is explained by the **64-bin directional projection of the once-scattered sender radiance L₁ in the first gather** (isolated 97–102 %). Spatial patch averaging and order ≥ 2 projections are negligible. Evaluating the first gather exactly per relation ray takes C3UR's R_mirror from 0.040 to 0.024 at 0.56–0.82× the cost. | [M] WL90 |
| E8 | Pooled exact first gather equals the oracle on held-out geometry; single-draw variance fails the held-out rule in 1 of 3. | [M] WL91 |
| E9 | The learned panel operator changes 14.5 % between geometrically different but intrinsically identical configurations. Exact q closes 96 % of the OWN seed gap. Exact receiver-side substitution worsens the result (co-adaptation of tied factors). | [M] WL92B |
| E10 | Source-stage supervision improves the source stage (validation rel L1 0.90 → 0.74) but raises whole-table / single error by 4.6–7.3 %. g00010 dominates pooled L2 statistics; its deployed-encoding discrepancy is genuine, not target noise. | [M] WL93A |
| E11 | Untying keeps the source gain and removes about 80 % of the whole regression (parity with T0, no net gain). The capacity control was inert. Untied without supervision fits whole best through a non-physical decomposition (operators diverge, the cone-source gate is lost). | [M] WL93B |
| E12 | The exact M64 operator at both uses (+ frozen global), i.e. the binned exact composition, is better than TQ but only equals T0 in pooled single / whole (limiting factor LOCAL_OPERATOR). It is better than T0 on 47–49 / 55 geometries; g00010's encoding discrepancy causes the pooled tie. It is far better on rendered sharp readers (E2 T1 region C 0.030 / 0.033 vs T0 0.076 / 0.042; edit slope ≈ 1.0). There is no level-compensation conflict with the frozen global term. Receiver-only supervision moves the receiver; joint supervision loses the receiver gain. | [M] WL93C |
| E13 | The source interface admits at most two lights (`scene86.light_arrays`). | [M] WL93B |

**What the evidence does not show.**
- It does not show that a shared physical operator is unrepresentable. E5 shows the opposite at fixed geometry.
- It does not show that untying is necessary. E11 is parity only, with an inert capacity control.
- It does not show that learning cannot carry higher-order broad transport. E2 shows that it can.

## 3. Competing causal explanations for WL93A–C

| Explanation | Compatible evidence | Incompatible / weakening evidence | Status |
|---|---|---|---|
| H-cap: insufficient capacity (rank 16, width) | E4 (rank 46–59 needed for S1) | E5 (rank 16 suffices at fixed geometry); Lambert / floor are well inside capacity; E11's capacity control inert (uninformative) | **partially compatible**: capacity in **absolute-bin** coordinates is insufficient for a near-diagonal lobe, but not in general |
| H-opt: optimisation / supervision conflict | E10, E12 (joint supervision loses the receiver gain); E9 co-adaptation | E5 (free fits converge for some inits); per-stage targets each learnable alone | compatible; plausibly a **symptom** of H-struct |
| H-enc: input encoding information loss | E10 / E12 (g00010 genuine encoding discrepancy); E3 narrow-beam loss (WL86A) | WL86A: encoding explains ≤ 5 % of the workload failure; WL92 ENCB passes the rule | compatible **locally** (narrow beams, g00010), not as the main cause |
| H-own: geometry-dependent information stored in the "intrinsic" local operator | E9 (14.5 % geometry dependence of the panel operator); E5 (per-geometry factors work, the generator does not); E11 (role-specific operators diverge) | — | **strongly compatible** |
| H-int: insufficient interface between local and higher-order transport | E7 (binned L₁ projection in the first gather); E3 (global term absorbs the missing family) | E12: no level-compensation conflict with the frozen global | compatible for the **first gather**; not supported as a level conflict |

**Proposed structural explanation (H-struct) [H].** One mechanism links H-cap, H-own, H-opt and H-int.

- A near-specular lobe is a **near-permutation** in absolute direction bins. Each incident bin maps to a narrow set of outgoing bins around its mirror image. In that basis it is high-rank (E4), but it is a low-dimensional object in a mirror-aligned or half-vector parameterisation.
- ECRO represents every local use with one rank-16 bin factorisation and produces it with a geometry-conditioned generator. The only way to fit a near-permutation with rank 16 is to specialise the factors to the incident directions that actually occur at that patch in that geometry.
- That specialisation (E5) moves **geometry-dependent** information into the nominally intrinsic local operator (E9), so it cannot be reused across edits.
- The incident-direction distributions differ between the source use (light directions) and the receiver use (inter-surface directions). The two uses therefore pull the shared factors apart: the observed conflict (E10, E12) and the partial relief from untying (E11).
- Independently, routing the sharp once-scattered radiance through a binned intermediate (L₁ table → first gather) loses the remaining sharp information (E7), even with an exact local operator (E12's binned oracle).

This explanation is **not established**. Its two load-bearing, testable premises are:
- P1: the frozen-ER sharp lobes become low-rank and non-negative once reparameterised or split.
- P2: the binned first-gather projection, not the learned global term, carries the residual sharp error once the local operator is exact.

## 4. Primary proposal: Lobe-Split Intrinsic Transport (LSIT)

### 4.1 Core hypothesis [H]

Sharp and broad transport need different representations, and the boundary between them is a property of the **intrinsic material kernel**, not of the source/receiver role. LSIT therefore assigns:

- **Intrinsic lobe kernel K_L.** A per-material, analytic-form kernel in a reciprocal half-vector parameterisation. It is evaluated per ray, never binned. It carries all sharp directional scattering.
- **Intrinsic smooth kernel K_S.** The per-material 64-bin remainder matrix, which is low-rank by premise P1.
- **Explicit geometry-owned transport.** Up to and including the **second** scattering event (the table's L₂ term), using K_L via per-ray lobe relations and K_S via the existing state rays. This is where sharp interreflection lives (E3, E7).
- **A learned, geometry-conditioned, linear response operator R_θ.** It owns **only events ≥ 3** (the table's L₃₊ term). It is driven by the physically complete second-order field, not by the raw source encoding. This is the part where learning was shown to work (E2 broad, OWN global) and where path tracing is most expensive (bounces 3–6; WL86 economics).

### 4.2 Components, inputs and outputs

All fields are RGB per patch i (1,270 patches) and per direction bin b (64 bins), unless stated otherwise.

**C1. Intrinsic kernel split (offline, per material; reusable across geometry and edits).**
- Input: the frozen ER closure f_m(ω_i, ω_o).
- Output, part 1: the lobe kernel f_m^L(ω_i, ω_o) = F_m(θ_d) · D_m(θ_h) / (4 cos θ_i cos θ_o), with h = normalize(ω_i + ω_o) and (θ_h, θ_d) the Rusinkiewicz half / difference angles. It is reciprocal by construction (P0) and non-negative. D_m and F_m are tabulated 1-D functions fitted to ER. For isotropic coated conductors this matches ER's own microfacet structure; this is an assumption to check (A1).
- Output, part 2: the smooth remainder f_m^S = f_m − f_m^L ≥ 0, enforced by a constrained fit. It is projected to a bin matrix M_m^S [64 × 64].
- Contract: f^L + f^S = f within tolerance; both parts non-negative and reciprocal; M^S low-rank (P1).
- The split is owned by K, so it is intrinsic and contains no geometry.

**C2. Geometry state (owned by C; updated on edits).**
- (a) The existing receiver-gather state rays: 512 per patch, hit patch and outgoing bin; selective re-trace exactly as in WL86.
- (b) The new **lobe relation** for each sharp patch (materials whose f^L carries ≥ ε of the albedo: panel S1, sphere S2):
  - per point x of the patch (P stratified points, P = 16 as deployed);
  - per incident direction bin b (for second-event use) or per exact light direction (for first-event use, evaluated per source);
  - K_r rays sampled from D_m around the mirror image of the incident direction, traced through current geometry.
  - Each ray stores its hit patch, the outgoing bin at the hit, and its lobe weight.
  - Bin-indexed part size: N_sharp × P × 64 × K_r. With N_sharp ≈ 74 and K_r = 16 this is ≈ 1.2 M ray records, the same order as the 0.65 M deployed state rays.
  - The per-light part is not stored. It is re-emitted per source (C3URX1-like, E7).

**C3. Explicit first- and second-event fields (runtime, linear in the source).**
- Direct incidence: e(x, ω_l) exact per light sample (16 points / patch, as the deployed source estimator).
- Once-scattered smooth table: q^S = M^S · s, where s is the deployed binned encoding (adequate for broad transport, E2).
- First flight of the smooth part: E1^S = flight_state(q^S), using the state rays.
- First flight of the lobe part: E1^L[j, b] = Σ_{x, l} e(x, ω_l) · ρ^L(θ_l) · (1/K_r) Σ_rays δ(hit = j, bin = b) cos.
  - This is the per-ray exact first gather of E7; there is no binned L₁.
- Second scattering at every receiver j, which gives the table's two-event term:
  - L₂[j, ·] = M^S_{m_j} · (E1^S + E1^L) + lobe_relation_bins_j(E1^S + E1^L);
  - the second term is non-zero only for sharp receivers and uses the bin-indexed lobe relation;
  - this is K at the receiver use: the same intrinsic K_S and K_L as at the source use.

**C4. Learned higher-order response R_θ (owns events ≥ 3).**
- Input: the explicit second-event outgoing field L₂ [1270, 64, 3], or equivalently the incident field it induces. The geometry state is passed to a geometry-conditioned generator, as in ECRO.
- Output: L₃₊[i, b], the outgoing radiance of 3–5 events.
- Form: a linear, non-negative, zero-preserving operator in its input.
  - Candidates: the ECRO generator form B₃₊ = a·(c·flight(L₂) + P·Σ area·Q·u). Its local factors now only need to represent K_S-like smooth scattering after a smooth flight; lobe events at orders ≥ 3 are **not** represented (truncation A2).
  - It is trained OWN-style: target t − t2 (exists: WL86 t, WL92A f2), input L₂ computed explicitly per training item.
- R_θ never represents a sharp local lobe. Its factors may still be geometry-conditioned (C-like visibility and multi-bounce), which is a legitimate geometry dependence, because R_θ is regenerated from the current geometry state after an edit without retraining.

**C5. Camera readout.** Unchanged WL86 LOW: camera x1 with next-event estimation, one ER bounce to x2 with next-event estimation. The table B = L₂ + L₃₊ is read at x2. Nothing is counted twice:
- LOW covers events 1–2 along the camera path;
- the table at x2 holds events ≥ 2 that leave x2;
- this is the WL86 event correspondence.

### 4.3 Operator summary

```text
Intrinsic, reusable:      f_m = f_m^L (half-vector lobe, per-ray) + f_m^S (M_m^S, 64-bin, low rank)
Geometry state (edit):    S_state (receiver gather rays), S_lobe (sharp-patch lobe relation rays)
Per source:               e  -> q^S = M^S s ;  E1 = flight_state(q^S) + lobe_first_gather(e, S_lobe)
Explicit 2nd event:       L2 = M^S . E1 + lobe_second(E1, S_lobe)
Learned >= 3 events:      L3p = R_theta(G_state)[ L2 ]        (linear, >= 0, 0 -> 0)
Camera:                   LOW (physical) + throughput(x1) * (L2 + L3p)[patch(x2), bin]
```

### 4.4 Reuse and update contract after a rigid edit

| Item | Reused unchanged | Recomputed | Cost owner |
|---|---|---|---|
| K split (f^L, M^S) per material | yes | — | offline, once per material |
| R_θ weights (generator) | yes | — | offline, once per scene family |
| Receiver-gather state rays | unaffected rays | rays crossing the old / new mover volume (selective, WL86) | C |
| Lobe relation (bin-indexed part) | unaffected rays | rays crossing the mover volumes, and rays starting on the mover if it is sharp | C |
| Generator outputs (R_θ factors) | — | one forward pass on the current state (WL86: 11 ms wall) | learned, no retraining |
| Explicit E1 / L₂ | — | per edit and per source | explicit |

**Source / receiver consistency.** The same K_L and K_S are used at the first and second scattering and at the camera bounce: identity by construction. No role-specific parameters exist. The learned R_θ never carries a material lobe, so the WL93A–C stage conflict has no place to arise. Whether it reappears inside R_θ for higher orders is a test (§7, P-T5).

**Sharp directional effects.** Carried per ray by K_L:
- around exact light directions at the first event;
- around incident-bin centres, with K_r lobe samples, at the second event.
- The output is binned only when deposited into the table at the receiver, where WL86 / WL90 showed the table is adequate (E1, E7).

**Higher-order interreflection.** Events 3–5 are learned (R_θ). Sharp events at order ≥ 3 are excluded by assumption A2, and their measured share is a rejection criterion.

### 4.5 Learned versus explicit responsibilities

- **Explicit:** the intrinsic kernel split (offline fit, no geometry), visibility and flight (state rays, lobe relations), events 1–2.
- **Learned:** events ≥ 3, conditioned on geometry, linear in the explicit second-order field.
- This is a deliberate reallocation relative to ECRO v0, which learned everything ≥ 2. It **conflicts with the 2026-11-13 owner decision** that keeps explicit transport as comparators only. See §10, decision D1.

### 4.6 Storage and inference complexity (estimates, not measurements)

| Quantity | Estimate | Basis |
|---|---|---|
| State rays | 0.65 M (unchanged) | WL86 |
| Lobe relation | ≈ 1.2 M ray records (N_sharp 74 × P 16 × 64 bins × K_r 16) ≈ 10–20 MB | arithmetic |
| Per-source first-event lobe rays | N_sharp × P × K_r ≈ 19 k rays per light sample set | arithmetic; C3URX1-type path measured at ≈ 10 ms (WL90) |
| Second-event lobe redistribution | gather / scatter over 1.2 M records, O(records) | arithmetic |
| R_θ | ECRO-size generator, ≈ 0.74 M params, 11 ms wall forward per edit | WL86 |
| Per-edit wall | target ≤ WL86's 76–78 ms + lobe relation re-trace (selective) | to be measured (P-T7) |

### 4.7 Expected failure cases

- **Chained sharp events at order ≥ 3** (two facing mirrors, the panel seen via the sphere): A2 violated; R_θ would have to carry a lobe.
- **Anisotropic or multi-lobe ER**: the coated conductor has two lobes (coat + base). The half-vector form must represent both, or the base lobe falls into f^S and raises its rank (A1).
- **Curved sharp surfaces (sphere)**: the mirror direction varies within a patch. The per-point emission handles this, but at P = 16 points per patch the reflected pattern can alias (A3).
- **Narrow incident beams on sharp senders and area lights**: the first event is exact per light sample, but area lights require many samples (cost).
- **Sub-patch textures and contact geometry**: the output table is piecewise constant (E1 holds only for patch-scale content).
- **Positivity of f^S** near the lobe: the constrained fit may leave a residual that breaks f = f^L + f^S (A4).
- **Learned compensation**: R_θ may still absorb second-order errors geometry-blindly, as in E3. Training R_θ on L₃₊ only (OWN-style) is intended to prevent this.

## 5. Alternatives

**Alt-A: Mirror-reparameterised learned local operator (all-learned).**
- Keep ECRO's learned ≥ 2-event structure.
- Re-express the local factors c, a as functions on half-vector / mirror-relative offsets, making them shift-structured (a convolution-like operator per material).
- Make them **intrinsic per material**: no geometry conditioning of the local factors. The geometry-conditioned generator predicts only flight / visibility weights and the global term.
- Strength: fully learned (aligned with the owner decision); intrinsic local operator; low rank in the reparameterised coordinates.
- Weakness: the intermediate L₁ is still a 64-bin table read by receiver-gather rays, so E7's first-gather projection floor and E6's sampling limit remain. Re-binning between the mirror-relative and absolute grids at 64 bins is lossy for a narrow lobe.

**Alt-B: Explicit transport with a learned residual / cache.**
- Use the explicit sender-side recurrence (WL87 EXP + WL90 / WL91 exact first gather) for all orders.
- Add a learned operator R'_θ that predicts the residual between explicit low-sample estimates and converged transport, plus a learned cache for source-only updates.
- Strength: physically safest; WL87 shows explicit ≥ learned on this workload; minimal ownership questions.
- Weakness: the learned component's measured utility is only the source-only update cost (WL87). It weakens the research claim of learned reusable transport, and its cost scales with orders.

| Criterion | LSIT (primary) | Alt-A | Alt-B |
|---|---|---|---|
| Physical correctness | intrinsic K split reciprocal / non-negative by construction; explicit events 1–2 | intrinsic local operator, but a binned intermediate (E7 floor) | highest (explicit); learned only as residual |
| Sharp expressiveness | per-ray lobe at events 1–2; none at ≥ 3 (A2) | limited by 64-bin L₁ and receiver sampling | full, explicit |
| Reuse after rigid edit | K, R_θ reused; state + lobe relation selectively updated | K, generator reused; state updated | explicit state recomputed; learned residual reused |
| Learned responsibility | events ≥ 3 (≈ 46–48 % of the region-C higher-order reading, WL92) | all events ≥ 2 | residual / cache only |
| Cost / memory | WL86-scale + ≈ 1.2 M lobe records + per-source lobe rays | WL86-scale | WL87 explicit-scale (C3 / TF cost measured 0.4–4.6 s per edit in an unoptimised implementation; C3URX1 cheaper) |
| Training difficulty | moderate: R_θ on smooth higher orders only | high: learn intrinsic lobe operators at 64 bins | low |
| Debuggability / falsifiability | high: oracle checks per component (§6) | medium | high |
| Conflict with current owner decision | yes (explicit events 1–2) | no | yes (explicit-first) |

## 6. Minimal falsification experiment (proposal only; nothing authorised)

**Ordering.** Stage 0 is learning-free oracle and contract checks. It runs before any training and can reject LSIT outright. Stage 1 is a small smoke, run only if Stage 0 passes and the owner approves.

**Preserved baseline.** ECRO v0 / WL92A OWN / WL93A T0 checkpoints and evaluation contracts are unchanged. The untextured GM1 room is the quantitative control.

**Stage 0: contract and oracle (no learning; inspected E2 / E5 plus one new held-out edit).**

| Test | What is computed | Rejects LSIT if (thresholds to be frozen before running) |
|---|---|---|
| P-T1 kernel split | Constrained half-vector lobe fit to frozen ER for B, S1, S2; smooth remainder M^S | f^S < 0 anywhere beyond tolerance; reciprocity residual > 1e-3; M^S_{S1} needs rank > 16 for 99 % flux energy, or its rank-16 flux-norm residual is not ≤ ⅓ of the full S1 value (0.46–0.52, WL92) |
| P-T2 truncation share | Path-traced decomposition of the higher-order reading by lobe-event order (tag lobe-sampled vertices), regions C / panel / D on E2 T0 / T1, E5 T0 / T1 and the held-out edit | share of region-C higher-order reading with a lobe event at order ≥ 3 > 0.10 (A2 false) |
| P-T3 explicit second-order oracle | LSIT-L2 (explicit events 1–2 with the split kernels and lobe relation, P = 16, K_r = 16) + path-traced L₃₊ | region-C E_H > 0.06 or not better than the binned exact composition (WL93C ORA: 0.030 / 0.033 on E2 T1) by max(noise, 0.005); C-region edit slope outside [0.75, 1.25] |
| P-T4 encoding vs operator | P-T3 with the first-event lobe driven by (i) exact light directions vs (ii) the deployed binned encoding, plus the full population incl. g00010 | (diagnostic) if (ii) − (i) carries most of the residual, the failure is the source interface, not the operator, and LSIT's first-event claim is confirmed or refuted accordingly |
| P-T5 learned-global compatibility | LSIT-L2 + the frozen WL92A OWN global (trained on a different input) | no rejection; measures whether a learned ≥ 3 term composes with an explicit L₂ without level conflict (WL93C found none for the binned oracle) |
| P-T7 cost | per-edit selective update of state + lobe relation, per-source lobe rays, memory | per-edit wall > 2× WL86's 76–78 ms, or per-source cost > ECRO's source-only update by > 2× |

**Held-out condition.** One new edit of the WL86 family, generated by a predeclared geometry-only rule (WL92B-style admission including the sharp family and Q4 SNR 4). It is qualified before any LSIT output exists and is never used to fix thresholds. E2 / E5 remain inspected conditions.

**Stage 1: learned smoke (owner approval required; WL93A-scale).**
- Train R_θ (ECRO generator family, same width / rank, no tuning) on L₃₊ targets (t − t2, existing) with inputs L₂ from C3.
- 128 geometries, 2,000 steps, seeds 8601 / 8602.
- Arms:
  - LSIT-learned;
  - LSIT-oracle (path-traced L₃₊), as the representability bound;
  - WL93A T0, as the baseline.
- **Favouring outcome** (all on both seeds):
  - region-C E_H on the held-out edit ≤ 0.06 and ≤ T0 − margin;
  - whole-table rel L1 ≤ T0 + margin;
  - stationary / D within 1.25 × T0 + 0.002;
  - C-region edit slope in [0.75, 1.25];
  - R_θ field error on L₃₊ ≤ 1.25× its training-subset error.
- **Falsifying outcomes:**
  - LSIT-oracle passes but LSIT-learned fails: the learned ≥ 3 operator is the limit, not the lobe structure;
  - both fail: the lobe / event split does not explain the sharp failure.

**Patterned diagnostic.**
- Use a **patch-aligned** checker (25 cm cells on the Lambert back wall / facing box face, albedo 0.3 / 0.8).
- It is read through the near-mirror panel. The reflected pattern exposes directional misplacement and blur of the sharp two-event family.
- Patch-aligned cells keep the piecewise-constant table exact for the texture, so changes reflect transport, not table resolution.
- A sub-patch checker would test the table container instead; keep it only as a labelled separate condition.
- Path-traced references are rendered with the same texture.
- The untextured room remains the quantitative control. Texture changes the transport (albedo-weighted interreflection), so textured results are compared only against their own references.

## 7. Evidence and evaluation contract for LSIT

| Property | Test | Discriminates |
|---|---|---|
| Representation feasibility | P-T1 rank / positivity / reciprocity; P-T3 oracle E_H | H-struct (P1, P2) vs H-cap |
| Physical contract correctness | f = f^L + f^S; reciprocity; zero → 0; superposition; energy (albedo ≤ 1 per incident direction); event correspondence with LOW (no double counting, WL86 V1-style test) | implementation errors vs design errors |
| Source / receiver consistency | identical K at both uses (identity test); reciprocity swap test on a two-patch synthetic scene | H-own (role-specific information) |
| Sharp fidelity | region C, panel object, E2 T0 / T1, E5, held-out; signed reading split level / directional (WL86A metric); panel-family delivered fraction | H-int (first gather) vs H-enc |
| Whole fidelity | whole-table and single / L₃₊ field rel L1 with geometry-level distributions, g00010 always included; robust summaries next to pooled | regression vs T0 |
| Edit response | T1 − T0 slope / correlation in A, C, D; STALE control | dynamic validity |
| Encoding vs operator | P-T4 | H-enc |
| Learned feasibility | Stage 1 LSIT-learned vs LSIT-oracle vs T0 | H-opt / learnability of ≥ 3 events |
| Runtime viability | P-T7: per-edit, per-source, per-frame GPU / wall; memory | practicality |
| Held-out | new qualified edit; held-out sources LMIX / LSPOT (two-light interface) | generalisation |

Aggregate PSNR or pooled L1 is never used alone. Every claim pairs a pooled metric with per-geometry distributions and the sharp-region rendered metric (WL93A / WL93C lesson).

## 8. Physical and computational invariants

- Every path segment has exactly one owner:
  - K_L / K_S own local scattering;
  - the state and lobe relations own free-space flight;
  - R_θ owns only path suffixes with ≥ 3 events.
- The table never holds the once-scattered radiance of the readout patch (LOW owns it).
- All operators are linear and non-negative in the source; zero maps to zero.
- No source identity is used.
- The intrinsic kernel split depends on the material only. Any geometry dependence in it is a contract violation (tested by identical-material, different-geometry checks, as in WL92B).
- After a rigid edit: no retraining; selective ray re-trace; a generator forward pass; explicit recomputation of events 1–2.

## 9. Risks and rejection criteria (summary)

- **R1** (P-T1): the ER lobe is not separable into a non-negative reciprocal half-vector lobe plus a low-rank smooth remainder. LSIT's central premise fails.
- **R2** (P-T2): lobe events at order ≥ 3 carry > 10 % of region-C higher-order reading. The learned ≥ 3 operator would need sharp structure, and LSIT reduces to ECRO's problem.
- **R3** (P-T3): explicit LSIT-L2 is not better than the binned exact composition. E7's first-gather mechanism does not transfer to this composition.
- **R4** (P-T7): per-edit or per-source cost exceeds 2× ECRO's. LSIT is not a dynamic-GI improvement.
- **R5** (Stage 1): LSIT-learned misses its own oracle by more than the margin. Learning ≥ 3 events from an explicit L₂ input is not feasible at smoke scale.
- **R6** (owner contract): explicit events 1–2 are judged inadmissible for the method (D1). LSIT is then not an acceptable architecture irrespective of measurements.

**Physically questionable assumptions, flagged as possible rejection conditions:**
- A1: half-vector separability of the frozen learned ER.
- A2: truncation of lobe events at order ≥ 3.
- A3: P = 16 points per sharp patch is enough for curved senders.
- A4: positivity of the smooth remainder.

## 10. Unresolved decisions for the research owner

- **D1.** Are explicit, intrinsic-kernel-driven events 1–2 (lobe relations owned by C, kernels owned by K) admissible inside the published method, with learning confined to events ≥ 3? This contradicts the 2026-11-13 decision to keep explicit transport as comparators only.
- **D2.** Should the sharp / smooth boundary be a fixed per-material split (C1), or allowed to depend on the incident angle (θ_i-dependent lobe albedo)?
- **D3.** Is a new held-out edit (Stage 0) acceptable, with its qualification cost (≈ 20 min per candidate, WL92B)?
- **D4.** Is the patch-aligned textured receiver condition acceptable as an additional qualitative condition next to the untextured control?
- **D5.** Is the two-light source interface limit acceptable for pattern diagnostics, or should a source-interface change be considered separately (out of scope here)?

## 11. Verdict of this review

**NEEDS_CONTRACT_CLARIFICATION.**

- The design is concrete and falsifiable, and Stage 0 needs no learning.
- But its value as *the* architecture depends on D1. LSIT moves events 1–2 out of the learned operator, which the current owner decision does not permit for the published method.
- If D1 is answered yes, Stage 0 is ready for falsification as specified. If no, Alt-A is the admissible fallback hypothesis, and it is expected to inherit the E7 first-gather floor.

## 12. Sources reviewed

**Documents.**
- `AGENTS.md`.
- `docs/Architecture.md` §1–3 (current decisions of 2026-11-10 / 2026-11-13; K / C ownership; boundary interface).

**Worklogs** (all read directly).
- WL86 (`86_WORKLOG_ECRO_GLOBAL_RESPONSE.md`): design, measurements, observation, interpretation.
- WL86A (`86A_WORKLOG_ECRO_SHARP_FAILURE_ATTRIBUTION.md`): observation and causal attribution.
- WL87 (`87_WORKLOG_GM1_EXPLICIT_TRANSPORT_COMPARATOR.md`): observation, interpretation, verdict.
- WL90 (`90_WORKLOG_GM1_INTERMEDIATE_PROJECTION_GAP_ATTRIBUTION.md`): observation and interpretation.
- WL92 (`92_WORKLOG_ECRO_SHARP_REPRESENTABILITY_IDENTIFIABILITY_ATTRIBUTION.md`): physical attribution, local-operator ranks, ladder, free fits, observation, interpretation.
- WL92A / WL92B / WL93 / WL93A / WL93B / WL93C: authored and executed in this session; the numbers cited are from their worklogs and result JSON (`results/ecro_*`).

**Code.**
- `experiments/ecro86/model86.py` (ECRO form), `scene86.py` (patch layout, two-light interface).
- `experiments/gm1_explicit87/quad87.py` (q estimator, M64 dense matrix).
- `experiments/ecro_stage93/model93.py`.

**Not consulted** (independence requirement): `docs/plans/gm1_architecture_review/`, `docs/plans/global_transport_study_86b/`, WL84A, WL86B.
