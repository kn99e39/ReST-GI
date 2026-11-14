# WL95 independent architecture design: ownership track

**Scope.** This is a design-only report. I edited no repository file, trained nothing and ran no GPU or CPU computation. I read the eight evidence files and spot-checked code: `experiments/ecro86/model86.py:108-124`, `experiments/fnto_compiled82/closure82.py:4-20,101,148,200`, `experiments/gm1_explicit87/quad87.py:85`, `experiments/gm1_relation88/rel88.py:47-59`, `docs/Architecture.md:1-60`.

**Labels.** [FACT] = verified in code or docs. [MEAS] = measured value with its source. [INF] = reasoned inference. [HYP] = untested hypothesis.

**Where the brief's evidence sections are.**
- A (contracts): §3.0 and inside each hypothesis.
- B (existing evidence): §2.
- C (representation feasibility) and D (system feasibility): §4.
- E (smoke criteria): §6.

**Metric discipline.**
- E_H values are compared only within the same workload, region and reference.
- Field and operator numbers are the WL94D metrics as Stage B recomputed them (W94 = the azimuth-mis-weighted `an94.W`; operator metric uses the correct ring zb).
- GM1 R_mirror and ECRO region-C values are never compared with each other.

---

## 1 Verified Current Architecture

### 1.1 ECRO as deployed (WL92A OWN checkpoints in the unchanged `model86.ECRO`)

**Two-tier image.** [FACT]
- LOW is path-traced: camera event 1, plus event 2 by NEE at x1.
- Higher-order pixel: `hl = thr(x1)·B[patch(x2), bin(x2→x1)]`, nearest patch, no extra ray (`pt86.py:228-233`).
- B[1270, 64, 3] is the 2–5-event outgoing radiance at x2, so the table supplies camera events 3–6.

**Bins and source.** [FACT]
- 64 equal-solid-angle bins: 4 cos-rings × 16 sectors, ΔΩ = 2π/64 (`pt86.py:153-156`).
- Source encoding s[n,m] = patch-mean direct irradiance per incident bin, from 16 fixed points per patch shared by all patches (`pt86.py:378-402`).

**Apply** (`model86.py:110-124`) [FACT]:
```
u  = Σ_m c[n,m,r] s[n,m]                       (learned incident factor, tied)
Bq = Σ_r a[n,d,r] u                            (learned once-scattered sender; M_n = a cᵀ, rank ≤ 16, ≥ 0)
E1[n,m] = Σ_{κ<8} 1[hit≥0] Bq[hit,dbin]·rw     (flight on 512 fixed rays/patch, sender patch × bin lookup)
B  = a·( c·E1  +  P Σ_n' area Q u )            (receiver uses the SAME c; global rank-24 term driven by u, not E1)
```

**Learned local scattering.** [FACT]
- WL84 learned ER is a closed-form compact closure: one Lambert lobe and seven anisotropic horizon-safe GGX lobes (3 outer, 4 inner with a refracted interface), 69 floats per material.
- It provides `eval/pdf/sample` (VNDF), is passive by certificate and reciprocal (`closure82.py:4-20,101,148,200`; Architecture.md §3).
- M64 is its bin projection: incident 8×8 cells weighted uniformly in solid angle, outgoing bin average, no cosine (`quad87.py:240-242`).

**Two re-representations of the same local physics.** [INF] ECRO represents local physics twice:
1. through ER, which LOW and the readout use;
2. through a geometry-conditioned learned rank-16 bin operator a cᵀ, which the table uses at both the sender and the receiver.

**Edit path.** [FACT] `set_geom` → selective retrace of dirty state rays (bit-equal to a full rebuild) → full recompute of s → full regeneration of the factors. The O(N²) dense form factor Ft and message passing remove any edit locality (`ecro86.py:57-87`).
- [MEAS] 76–78 ms per edit, of which selective retrace saves 5–7 ms (WL86:214).

### 1.2 Loss points carried forward (Stage A, verified)

| # | Loss point |
|---|---|
| L1 | Source incident-bin collapse |
| L2 | 16 fixed points |
| L3 | Rank-16 tied a cᵀ |
| L4 | Outgoing bin and patch projection of the sender table |
| L5 | κ = 8 fixed shared ray pattern, single draw |
| L6 | Ring-0/1 cosine aliasing |
| L7 | Receiver tied c |
| L8 | Global term driven by u, with no flight geometry |
| L9 | 69→16 material embedding |
| L10 | Nearest-patch readout |
| L11 | Flat frame on curved patches |
| L12 | M64 incident-cell averaging |

---

## 2 Failure and Evidence Attribution

### 2.1 The sharp failure, by ownership

**1. The learned S1 local operator is a material-intrinsic object that a geometry-conditioned generator is re-learning, in the wrong basis.**
- [MEAS] WL93D: S1 S_geo is 0.2–0.8 % (S2 0.07–0.28 %, LAM 0.8–1.7 %, floor B 3–5 %). E_phys 0.75–0.82 is geometry-common.
- [MEAS, Stage B] In the WL94D operator L1 metric the zero operator scores 1.000 and the learned S1 operators score 0.96–1.04.
- [MEAS, Stage B] Admissible non-negative rank-16 operators reach 0.38 (L1-targeted) to 0.60 (NMF16). On held-out panel fields they reach field L1 0.36–0.54 at peak 0.35–0.44, against learned 0.85–0.89 at peak 0.22–0.33.
- [MEAS, Stage B] Rank 16 caps field L1 at ≥ ~0.36 and field peak at ≤ ~0.49. Field L1 ≤ 0.15 with peak ≥ 0.7 needs rank ~32; rank 48 gives 0.049 / 0.847.
- [INF] A near-mirror operator is close to a permutation in the bin×bin basis, hence high-rank there. In ER's half-vector/lobe parametrisation it is about 7 lobes × 3 parameters.
- Ownership conclusion [INF]: no learned component should own the local operator of a material that ER already owns exactly. A geometry-conditioned generator buys < 1 % specialisation for S1, at the cost of a rank cap and a training failure that WL93A–WL94D supervision did not move off the zero-operator level.

**2. Replacing the learned sender with an exact ER projection helps rendered sharp regions in every measured case, even though the pooled table gets worse** (co-adaptation).
- [MEAS] WL93D C2_src, E2 T1 region C: 0.018–0.024 vs C0 0.040–0.077.
- [MEAS] WL93D C2_src, E5 T1: 0.013–0.018 vs 0.025–0.033.
- [MEAS] WL93C ORA (M64 at both ends + frozen G), E2 T1: 0.030 / 0.033 vs T0 0.076 / 0.042; edit slope 0.995 / 1.000 vs 0.57 / 0.75.
- [MEAS] WL94 C2_srcS1, CK2 ROI_T untextured: 0.031 / 0.019 vs C0 0.047 / 0.042 (noise 0.0084).
- [MEAS] The pooled table moves the other way: C2_src single-flight Δ +0.005…+0.008; C2_both not better than C0.
- [INF] The learned receiver and global term are tuned to the learned sender's errors (WL93D COADAPTATION; WL86A Amendment 1, global term +0.02…+0.06 geometry-blind).

**3. Even with an exact ER-projected sender, a binned first-flight floor remains.** It is attributable to angular projection of the once-scattered radiance in the first gather, plus relation sampling.
- GM1 [MEAS, WL90]:
  - The explicit-vs-oracle gap is 97–102 % isolated to first-gather angular projection; spatial ≈ 0; compounding ≤ 9 %.
  - Converged relations R0_INF / R1_INF / C3UR_INF all give R_mirror 0.038 vs oracle 0.014 (WL89).
  - X1 (per-ray exact L₁) removes the bias (pooled ≈ oracle) but raises single-draw variance (WL91: 1 / 3 held-out conditions; W2 T1 deployed draw 0.053).
- ECRO scene, E2 T1 region C [MEAS]. Note: ENCB carries true L₃₊; ORA2 is path-traced tables.

  | Arm | E_H |
  |---|---|
  | ENCB (c = I, a = M64, R = 64), single draw | 0.0449 |
  | ENCB pooled | 0.027 |
  | ORA2 | 0.0127 |
  | Sender-side C3 / C3U / C3UR | 0.024 / 0.023 / 0.026 |
  | Receiver-side R0 | 0.049 |
- [MEAS, WL88 Task A] Sender-vs-receiver placement at equal rays and equal quadrature: +0.050 / +0.053. Power allocation and per-order freshness: noise.
- CK2 [MEAS, WL94D; decomposition OPEN per Stage A]:
  - Exact M64(s) S1 sender + exact receiver vs e2_p, box A: ratio 1.21, rel L1 0.53, half-noise 0.058.
  - Panel face: corr −0.33 over 9 patches.
  - The candidate causes (C1 outgoing-bin ring-0 averaging, C2 spatial, C3 incident L1 + L12, C4 L2, C5 L5, C6 L6) are confounded in one number. In CK2 all panel incidence is grazing ring 0 (cos 0.024–0.164) and panel irradiance varies 307× across the panel.
  - The G_PHYS positive control (training, pooled 0.953) argues that the over-delivery is CK2-specific [INF].

**4. The learned global (≥ 3-event) term is a real but secondary error source on sharp-mediated faces.**
- [MEAS] CK2 panel face: e3_p is 0.755 of box e3 (WL94B). Learned G vs e3 rel L1 0.20 / 0.15 (own) vs half-noise 0.027 (WL94C).
- [MEAS] G is not the binding error on E2 once the single flight is exact (WL93C ORA: "no compensation conflict").

### 2.2 Evidence status relevant to ownership

| Finding | Status | Source |
|---|---|---|
| Local S1 operator geometry-invariant (S_geo < 1 %) | supports ER ownership | WL93D `final93d.json` |
| Learned S1 operator ≈ zero operator; rank-16 cap; rank ~32 needed | supports removing the learned local operator; weakens "more training / supervision" | Stage B |
| Exact sender improves sharp renders in all inspected cases | supports | WL93C/93D/94/94C |
| Sum-only and co-adapted learned terms compensate | supports exclusive, separately tallied learned targets; weakens ECRO's single+global split | WL86A, 86B §C, 93D |
| Binned first flight has a floor ~2× oracle on E2 C and GM1 R_mirror, and an undecomposed CK2 over-delivery | inconclusive for whether a binned reusable flight suffices | WL90/91, 92, 94D |
| Sender-side placement > receiver-side at equal rays | supports a sender-side relation | WL88 Task A, WL87 C3 |
| X1 exact per-ray removes bias but raises variance | supports "exactness needs lobe-adapted sampling" | WL90/91 |
| Explicit same-state transport ≥ ECRO everywhere; learned advantage only source-only update (3.5–3.9×) | weakens any claim that learning helps sharp first-flight fidelity; leaves learned-resolvent utility unmeasured | WL87 |
| Learned visibility / C (GLT) too slow and loses contact locality | supports explicit ownership of visibility | WL52 |
| WL94-series exitance/field metrics use the mis-weighted W94; `delivery_box` = boxes A+B | WL94D sender and delivery numbers carry an unquantified bias (≤ 0.02 field L1 on the bound arms) | Stage A §4 |

---

## 3 Independent Architecture Hypotheses

### 3.0 Shared ownership derivation (Evidence A)

**Transport.**
- L(x,ω) = L_e + ∫ f_k(x)(ω′, ω) L(r(x,ω′), −ω′) cos θ′ dω′. Write K = ER at a vertex (per material, geometry- and source-independent) and C_G = free-space ray cast.
- Neumann series L = Σ_j (K C_G)^j L_e.
- For point lights, the once-scattered radiance is q(y,v) = f(y; ω_ℓ(y), v) I_ℓ cos θ_ℓ / d² V(y,ℓ).
- The 2-event table at x2 is:
  B₂(n,d) = (1/A_n)(1/ΔΩ) ∫_{A_n} ∫_{Ω_d} ∫ f(x; ω, v) q(r(x,ω), −ω) cos θ dω dv dA.

**Ownership principles** (derived from evidence, not assumed):
- **O1 — local response belongs to ER at every vertex.** ER is closed-form, passive, reciprocal and material-only; geometry adds < 1 % for S1. Bin-projected use (M64) is admissible where its error is measured. A learned re-representation of K is not admissible.
- **O2 — visibility belongs to explicit ray casting on a geometry-only relation.** Selective retrace is exact. Learned C failed on cost and locality (WL52).
- **O3 — a learned operator may own only a geometry-dependent composite path family that it is separately supervised on, and that no physical term also owns.** The candidate is the multi-bounce resolvent beyond the first flight. Its utility over explicit sweeps is unmeasured (WL87; WL86B: the fill advantage is bounded by J ≈ 3–5 while the generator is O(N²)).
- **O4 — the sharp L→S→D first flight is information-dense in both source direction and geometry.** Per sender point, the outgoing structure scales roughly as α⁻²; a source-independent tabulation adds another ~α⁻² in incident direction. DDT's α⁻⁴ state was withdrawn at ~38 GB (WL84A). Rank-16 and bin-limited compressions measurably lose it.
  - [INF] So the first flight is either carried on a fixed binned reusable relation, accepting a projection floor, or computed per source with lobe-adapted sampling, accepting per-source ray cost.
  - This choice is the axis on which the two hypotheses differ. It is a real physical and computational trade-off, not a cosmetic variant.

**Common learned component (both hypotheses): learned broad resolvent Λ_θ** [HYP]
- **Output:** the incident field Ê≥₂[n,m,c] at patch n from light scattered ≥ 2 times. The local response at the final vertex is ER: B̂₃₊[n,d] = Σ_m Ê≥₂[n,m] M64[k_n][m,d].
- **Driver:** the physically computed twice-scattered table B₂ (or E₁). It is not u = c·s, which addresses L8.
- **Form:** one linear, non-negative, source-independent operator generated from (relation, Ft, ER states) by an ECRO-style message-passing generator with no post-edit retraining. The internal factorisation (rank ρ, Ft-structured vs global latent) is a protocol-freeze decision. A geometry-blind global latent alone is excluded, because it repeats L8.
- **Supervision:** only on the existing tallied e3 table (`pt86a.field_src(split=True)`: k ≥ 3 at x2; t3 = t − t2). Ownership is exclusive by event count: the physical prefix owns e2 and the learned operator owns e3, and the tallies sum to t [FACT `pt86a.py:174-199`]. The learned term cannot compensate the prefix, because the prefix is not learned.
- **Identifiability:** for Lambert receivers only the irradiance is identified, which is all that matters there. For glossy receivers it is identified through M64 [INF].
- **Energy audit (not a guarantee):** flux(B̂₃₊) ≤ ρ_max/(1 − ρ_max)·flux(B₂). Passivity is not structurally required because the operator is applied once, not recursed (dossier note: this applicability is unresolved).

### 3.1 H-A — ELBR: ER-local response, binned reusable sender-side relation, learned resolvent

**State.**
- Geometry-only relation Γ_s, sender-side (C3UR-type): 8 rays per (sender patch, outgoing bin), cos-weighted, deposited at the receiver's incident cell. Fixed offsets; selective retrace. 650,240 rays, ~17.6 MB at 27 B/ray [FACT WL89:65, cost "~R0"].
- ER states (69 floats per material) and M64 per material (6×64×64×3, ~0.3 MB).
- Source state s (existing, or more points on peaked patches; see sharp treatment).
- Learned rule θ (resolvent only).

**Operator.**
```
q̂[n,d] = Σ_m s[n,m] M64[k_n][m,d]          (exact ER bin projection; no learned factor; L3 removed)
E1     = Flight_{Γs}(q̂)                    (sender-side deposition, binned sender radiance)
B2[n,d]= Σ_m E1[n,m] M64[k_n][m,d]         (receiver = ER; L7 removed)
B      = B2 + M64ᵀ · Λ_θ(G)[B2]            (learned owns e3 only)
```

**Ownership.**
- Local scattering: ER (M64) at the sender and the receiver.
- Geometry-dependent propagation: explicit Γ_s for the first flight; the learned Λ_θ (geometry-conditioned) for ≥ 3 events.

**Sharp structure.**
- Exact up to the bin projection: L12 (incident cells), L4 (outgoing bin, patch) and L1/L2 (source) remain.
- Sender-side placement attacks the dominant relation-sampling error (WL88 Task A) source-independently.
- **No sub-bin information** is carried anywhere. This is H-A's accepted floor.

**Diffuse / nonlocal.** Broad senders are exact under bin projection (floor B is essentially exact at rank 8–16; EXP broad fidelity SUPPORTED, WL87). Multi-bounce content goes through Λ_θ.

**Reuse after edits.**
- θ, ER and M64: invariant.
- Γ_s: dirty rays re-traced, using ECRO's existing exact segment/OBB rule.
- s, q̂, E1, B2: recomputed (linear; cheap).
- Generator outputs: fully regenerated (O(N²) Ft; no locality, as in ECRO).

**Training.** Existing data (1,024 geometries × 4 lights, 256-spp e2/e3 tallies). Only Λ_θ is trained. The prefix is computed exactly as at deployment.

**Inference-time physical computation.** H-A adds **no new ray tracing over ECRO**: s (N·16 shadow rays per light, [MEAS] 5 ms), Γ_s retrace on edit ([MEAS] ~14 ms warm C3UR update in GM1, WL89:124), matrix–vector M64 (~15.6 M MAC per light), flight gather, generator and apply.
- [INF] Per edit ≈ ECRO's 76–78 ms minus the c/a heads.
- Per frame: identical to ECRO (LOW + lookup).

**Memory and scaling.**
- Factors: resolvent only, ≈ 27 MB at R = 16 (ĉ + Q + P) vs ECRO's ≈ 43 MB [INF from shapes].
- Dense Ft is O(N²): 1.6 M entries now, 10⁸ at N = 10⁴ [INF]. This risk is inherited from ECRO.
- The fidelity floor does not improve with compute at fixed 64 bins and grows as α → 0 [INF].
- Worst case: grazing incidence on a near-specular sender (CK2-like), mirror cavities (sharp chains go to Λ_θ), curved sharp patches (L11).

**Cache invalidation.**
- ER/M64 only on a material edit.
- Γ_s rays whose origin moved or whose segment crosses the old or new OBB (+1 mm).
- s points whose shadow ray is dirty (recomputing everything is cheaper).
- Factors: always on a geometry edit.

**Conservation / reciprocity.**
- [MEAS] M64 S1 albedo ≤ 0.825 with the correct weight (T_ENERGY).
- M64 is approximately symmetric (same solid-angle averaging on both sides) [INF].
- The flight does not renormalise invalid rays (sphere leak, L11) [FACT].
- Λ_θ has no reciprocity guarantee (energy audit only).

**Why it addresses the failure.**
- It removes L3, L7 and the L8 compensation channel; reduces L9 exposure (final-vertex albedo exact; [MEAS] CK2 M64 receiver ratios exact 0.600 / 1.599); and reduces L5 via sender-side placement.
- It does **not** address L1, L2, L4, L6, L10, L11 or L12.
- Supporting evidence: Stage B, WL93C ORA, WL93D C2_src, WL94 C2_srcS1, WL92 ENCB, WL88 C3UR.
- Weakening evidence: WL89/90 binned floor (0.038 vs 0.014 GM1); CK2 exact-M64 over-delivery 1.21 / 0.53; WL87 (learned utility over explicit not shown).
- Inconclusive: Λ_θ fidelity on sharp-mediated e3 (CK2 panel face e3_p 0.755).

**Main risks.**
1. The binned floor fails sharp gates in grazing / near-contact cases. This is upstream of anything learned and cannot be rescued by training.
2. Λ_θ is no better than explicit sweeps on the same Γ_s, so H-A collapses to the null H-0, a learning-free comparator that is not a publishable learned method (Architecture.md:13).
3. Λ_θ is inaccurate on sharp-mediated ≥ 3-event families.
4. Scientific scope shrinks: learning owns only e3. This needs an owner judgement.
5. The event-count partition is still not a K/C segment partition; Λ_θ is a composite owner of (C K)^{≥1} C segments (an unresolved ECRO issue, read_canon (B)).

**Rigid / deformation / bending / self-contact.**
- Rigid: valid (rule reuse; exact retrace).
- Local deformation and bending: valid for ER (per-vertex). M64 per patch assumes a flat frame within a patch (L11), so bending a sharp panel breaks the projection [INF].
- Self-contact: patch-scale spatial resolution (L10; WL54/55 near-contact findings) is not addressed.

**Smallest falsifier:** §6, ladder levels 1–1c plus stage 2.

### 3.2 H-B — ELCF: ER-local response, light-side continuous first flight for peaked senders, learned resolvent

**State.**
- As H-A, plus a per-light, source-dependent light-side first-flight set Φ_ℓ for peaked senders.
- Peaked class P is a prospectively fixed, material-only rule (WL88 `rel88.peaked`): Π = median_w max_bo fbar(w)[bo] / (a(w)/π) > 8 [FACT `rel88.py:47-59`; MEAS B 4.93, S1 37.95, S2 16.13].
- In scene86, P = panel (42) + sphere (32) patches.

**Operator.**
- Exclusive by vertex-y material: e2 = e2_P + e2_¬P.
  - e2_¬P: as in H-A (binned q̂ on Γ_s, M64 receivers).
  - e2_P (light-side; no binning of sender incidence or outgoing):
    - P_pts points y_j per peaked patch (area weight A/P_pts).
    - Exact incident direction ω_ℓ(y_j) with a shadow ray.
    - m directions ω_o ~ ER VNDF/lobe sampler p(·|ω_ℓ) (`closure82.sample_closure`).
    - Photon flux Φ = I cos θ_ℓ/d² · V · (A/P_pts) · f(ω_ℓ, ω_o) cos θ_o / (m p(ω_o)).
    - Ray cast; at hit x in patch n deposit: B₂(n,d) += Φ · fbar_{k_n}(−ω_o)[d] / A_n, and E₁(n, bin(−ω_o)) += Φ / A_n.
    - fbar is the exact outgoing-bin average at the exact incident direction (`quad87.fbar`, line 85).
  - [INF] This is an unbiased estimator of the patch × outgoing-bin target quantity (the oracle table's definition). It removes L1, L4 (angular and spatial), L5, L6 and L12 for the sharp first flight, and L11 at the sender (per-point frame). It also removes L7/L12 at the receiver for this family.
- Learned Λ_θ as in H-A, driven by B₂ including e2_P.

**Ownership.**
- Local: ER, exact and continuous (sampling and evaluation) on the peaked first flight; M64 elsewhere.
- First-flight propagation: explicit source-dependent rays for P, Γ_s for ¬P.
- ≥ 3 events: learned Λ_θ.
- A misclassified material changes estimator variance and bias magnitude, not ownership. Both estimators target e2, which avoids H-P's "moderate lobes have no owner" defect (WL86B).

**Sharp structure.** Lobe-importance sampling from the exact incident direction is the standard remedy for L-S-D (caustic) paths that gathering estimators sample poorly [INF; no novelty claimed]. It targets exactly the variance that made X1 fail on lobe-agnostic directions (WL90 R0X1 no gain; WL91 variance).
- The light-side placement is consistent with WL88's "placement dominant" finding.
- P_pts > 16 attacks L2 on peaked patches. CK2 panel irradiance varies 307×.

**Diffuse / nonlocal.** As H-A.

**Reuse after edits.**
- Θ, ER, M64, Γ_s: as H-A.
- Φ_ℓ: invalidate photons whose origin moved, whose segment crosses the old or new OBB, or whose sender shadow ray changed. A light move regenerates Φ_ℓ for that light.
- [INF] A full per-light regeneration is likely cheaper than the dirty bookkeeping.

**Training.** As H-A. Λ_θ sees the physically correct B₂ (train and deploy prefix identical).

**Inference-time physical computation: explicit and counted.**
- Per light, per edit or light change: N_P·P_pts·m first-flight rays plus N_P·P_pts shadow rays, ER sample/eval per ray, and deposit atomics.
- Example: 74 × 64 × 64 = 303,104 rays + 4,736 shadow rays per light.
- [INF] By ray-count analogy with the 650,240-ray state build (26–28 ms full, WL86:214): ~12–15 ms per light on an RTX 3080 Ti class GPU, unmeasured. Four lights: ~50–60 ms added per edit.
- Per frame: unchanged (no per-frame retrace).
- **This is ray tracing, not learning;** it is source-dependent. It erases the only learned advantage WL87 measured (cheap source-only update) for the peaked family.

**Memory and scaling.**
- Transient photon buffer ~10 MB per light at 32 B/photon [INF]; no persistent tables beyond H-A's.
- Cost ∝ L × (peaked area fraction) × P_pts × m.
- Worst case:
  - many lights, or lights moving every frame (degenerates toward a per-frame light tracer);
  - area or environment lights (incidence not delta; light sampling must be added, which multiplies cost);
  - large peaked area.
- Fidelity no longer floors in α at the sender. The readout's nearest-patch lookup (L10) caps caustic sharpness below patch scale.

**Conservation / reciprocity.**
- The photon estimator conserves flux in expectation. ER is passive and reciprocal.
- The mixed estimator (light-side for P, gather for ¬P) is exclusive by sender material, so there is no double count [INF]. A test is needed: e2_P + e2_¬P vs e2 tally.

**Why it addresses the failure.** It covers everything H-A covers, plus the binned first-flight floor (WL90 J, ENCB pooled 0.027 vs 0.013, C3UR converged 0.038 vs 0.014 GM1) and CK2 candidates C1–C6 for the peaked family.
- Supporting evidence:
  - [MEAS] source-dependent sender-side C3 E2 T1 C 0.024 vs EXP 0.049;
  - [MEAS] X1 bias removal (pooled ≈ oracle, WL91).
- Weakening evidence:
  - C3UR, which is source-independent, matched C3 on E2 (0.026), so source dependence was not needed there;
  - no measurement of lobe-sampled light-side deposition exists anywhere.

**Main risks.**
1. A cost shift into per-source ray tracing. If the binned floor is acceptable this cost is unjustified; if lights are dynamic, H-B approaches a light tracer.
2. The source-independent rule remains (ER + θ), but part of the state becomes per-source. Admissibility under the "source-independent reusable rule" objective is an owner decision. The contract allows source-dependent incident fields (Architecture.md §4: "different sources are represented by different incident fields"), but H-B extends that by one sharp bounce.
3. Variance at small P_pts·m on small footprints. CK2 box A footprint is fed by 2 panel patches.
4. Sharp chains beyond the first flight (S→S→D) remain in Λ_θ.
5. It is known technique (light/photon tracing, image sources); no method novelty.

**Smallest falsifier:** §6, ladder level 2.

### 3.3 Alternatives considered and rejected

| Alternative | Reason for rejection |
|---|---|
| **R1. Keep a learned local operator but raise rank to 32–48 or bins to 256+, possibly ECRO-recipe fixes (H-E)** | Ownership duplication of an ER-owned, geometry-invariant object (S_geo < 1 %).<br>[MEAS] Training does not reach even the rank-16 bound (zero-operator level after WL92A–94D supervision).<br>The bin floor does not shrink with rank (ENCB, R = 64, still 0.027 pooled on E2 C vs 0.013).<br>More bins inflate the relation (~α⁻² cells) and are explicitly not assumed to solve it. |
| **R2. Learned, source-independent sampling density / guiding for a sender-side relation (to cure C3URX1 variance)** | For point-light caustics the important outgoing lobe is set by the source, so a source-independent density cannot concentrate on it [INF].<br>A learned reweighting cannot recover relations never sampled (56–62 % never hit, WL87).<br>Whether a learned density counts as "learned transport response" is an open owner decision (WL84A).<br>Kept only as a possible variance-reduction control. |
| **R3. Spherical-Gaussian / lobe-atom directional state for all orders with a learned atom-merging closure** | Atom count multiplies per bounce, so it needs a new, unvalidated learned closure.<br>Broad multi-bounce does not need it (B exact at rank 8–16).<br>It introduces a second representation family without measured necessity. |
| Image sources for planar sharp senders | Not a separate architecture; at most an estimator option inside H-B. |

---

## 4 Physical and Computational Trade-offs (Evidence C, D)

### 4.1 Ten criteria

References in the table: ECRO v0 / own (current) and H-0 (EXP / C3UR, learning-free null).

| Criterion | ECRO own | H-A ELBR | H-B ELCF | H-0 null |
|---|---|---|---|---|
| 1 Angular representability | Learned S1 ≈ zero operator; rank-16 cap ≥ 0.36 field L1 [MEAS] | Full-rank M64 at both ends; bin-projection floor kept (E2 C pooled 0.027 vs 0.013; GM1 0.038 vs 0.014) [MEAS] | Continuous on the peaked first flight [HYP]; bins elsewhere | As H-A |
| 2 Propagation fidelity | Learned flight delivers 1–18 % of the E2 panel family [MEAS] | Sender-side binned relation: 0.026 single on E2 T1 C (C3UR) [MEAS]; CK2 unknown | Unmeasured; X1 / C3 evidence suggests bias removal [INF] | As H-A |
| 3 Material generalisation | Albedo compression (CK2 slopes 0.28–0.63) [MEAS] | Final-vertex exact (M64 ratios exact) [MEAS]; Λ_θ embedding still a risk | As H-A | Exact by construction |
| 4 Dynamic geometry reuse | Rule reuse; factors fully regenerated | Same | Same, plus per-light photon regeneration | State reuse; no learned part |
| 5 Physical correctness | No passivity; compensation | ER passive; exclusive tallies; Λ_θ audited only | Plus an unbiased first flight | Passive, exact operator |
| 6 Training feasibility | Hard (seed disagreement, co-adaptation) [MEAS] | Single learned term, separate target; fewer parameters [INF]; WL92A OWN global already reached val 0.092 field rel L1 vs L₃₊ [MEAS] | Same | None |
| 7 Inference / update complexity | 76–78 ms per edit; 0.33 ms/spp | ≈ ECRO [INF] | + ~12–15 ms per light per edit [INF, unmeasured] | EXP 89–95 ms [MEAS] |
| 8 Memory scalability | ~43 MB factors + O(N²) Ft | ~27 MB + O(N²) Ft | + transient per-light buffers | Relation only |
| 9 Maintainability / debuggability | Entangled terms | Every term physically tallied; prefix testable without learning | Same; plus estimator tests | Highest |
| 10 Usefulness vs full recompute | 9× per frame is LOW + table structure, shared by H-0 (WL87); learned-only gain = source update 3.5–3.9× | Inherits the structure; learned gain over H-0 unproven | Loses the source-update gain on the peaked family | Reference for learning-specific value |

### 4.2 Cost shift and hidden-table risks (explicit)

- **H-A:** no added ray tracing. The risk is the opposite: a learned component too small to matter, relabelled H-0.
- **H-B:** the sharp first flight is moved into per-source ray casting. This is a counted cost, not a learned capability.
- **Large tables:** the dense Ft and generator scale as O(N²) for all ECRO-lineage designs. No design here adds an N²·bins table; a source-independent sub-bin table was deliberately not proposed (DDT ~38 GB precedent).

### 4.3 Initial construction vs per-edit cost

- **Initial (offline):**
  - M64 per material (existing, 16,384 MIS draws per cell);
  - Λ_θ training on existing e3 tallies (similar to ECRO OWN, 24k steps; fewer parameters).
  - Nothing else for H-A. H-B adds no offline cost.
- **Per edit:** set_geom 31 ms + relation retrace ~14–21 ms + s 5 ms + generator ~11 ms + apply ~2 ms [MEAS components WL86/89; INF sum]. H-B adds per-light photons.
- **Scaling assumptions:** fixed 64 bins; point lights; patch counts ~10³; rigid edits. None is verified beyond scene86 / GM1.

### 4.4 Comparison gate

Any external comparative claim (8DNA / RNA) requires the mandatory baseline gate:
- Neural Full Recompute only if their learned representation is actually invalidated;
- Native Reuse otherwise;
- a separate path-traced reference.

None is made here; the status is **UNMEASURED**.

---

## 5 Recommended Hypothesis

**Primary hypothesis to falsify: H-A (ELBR).** It is the smallest ownership change that removes the evidenced failure mechanisms (L3, L7, the L8 compensation channel) without adding ray tracing.

It is recommended **for falsification, not adoption.** Selection between H-A and H-B depends on one learning-free quantity, and the learned-resolvent utility over H-0 is unmeasured. The verdict is therefore MULTIPLE_CANDIDATES_REQUIRE_DISCRIMINATION (§8).

**Mechanism** [HYP]. ER owns local response at every vertex. The first flight uses a geometry-only, sender-side, binned reusable relation. A single learned, source-independent, geometry-conditioned linear operator owns only the separately tallied ≥ 3-event incident field.

**Minimum evidence for H-A to be justified:**
1. **E-A1 (learning-free).** The exact binned prefix plus a path-traced e3 meets the sharp-region criteria on predeclared workloads, at the deployed single-draw budget, with pooled draws reported.
2. **E-A2.** The prefix B₂ vs the path-traced e2 table is within bounds (field rel L1 with the corrected W) on mirror-cone receivers.
3. **E-A3 (bounded training).** Λ_θ reaches a predeclared e3 fidelity on held-out geometry, and beats explicit sweeps on the same Γ_s in quality at equal cost, or in cost at equal quality.

**Falsification.**
- E-A1 fails with pooled relations on any predeclared sharp region. H-A is falsified, and no training can rescue it, since the floor is upstream of θ.
- E-A3 fails against explicit sweeps. H-A's learned claim is unsupported (H-A ≡ H-0).

**Baselines:** ECRO own (deployed), C2_both / ORA, EXP (R0), C3UR, ORACLE / ORA2 path-traced tables, REF-ER render, REF-P for g, STALE, same-mechanism full rebuild.

**Smallest valid workload:** existing scene86 family.
- E2 T1/T0 and E5 region C;
- CK2 untextured plus bx38/bx75 (post hoc origin disclosed);
- LMIX and LSPOT for source generalisation;
- one new held-out sharp edit, admitted by a predeclared SHARP_FAMILY rule. WL93D E6 failed admission, so this must be pre-qualified.

**Smallest implementation boundary:** an offline evaluation module composing existing pieces.
- `pt86` source / readout, `mat94.mq_ext` M64, `eval93b.flight`, the C3UR relation ported to scene86 (WL88 Task A already ran it on E2), `pt86a.field_src(split=True)` tallies.
- Λ_θ (stage 2 only) at the WL93 smoke contract.
- No ECRO production change; no checkpoint modification.

---

## 6 Falsification Experiment (Evidence E)

### Stage 1 — learning-free oracle ladder

All levels keep LOW, readout, scene, lights and seeds identical, with common random numbers. Each level changes one ownership element. The tail is the path-traced e3 except where stated, so that prefix error is isolated.

| Level | Sender | First-flight relation | Receiver | Tail | Removes | Available? |
|---|---|---|---|---|---|---|
| 0 | learned a cᵀ | deployed Γ (R0) | learned c | learned G | — | yes |
| 0′ | learned | R0 | learned | path-traced e3 | L8 / compensation | yes (X_L3) |
| 1 | M64(s) | R0, single / pooled 16–64 | M64 | e3 | L3, L7 | yes (ENCB / X_M12 code) |
| 1a | M64(s) | C3UR sender-side, single / pooled | M64 | e3 | + L5 placement | yes (gm1_relation89, ported) |
| 1b | exact q (`qtable`, X_Q) | C3UR | M64 | e3 | + L1, L2, L12 | yes |
| 1c | X1 per-ray exact | C3UR pooled | M64 | e3 | + L4 angular (bias) | yes (`cand91`) |
| 2 (H-B) | ER continuous, light-side, P_pts ∈ {16, 64}, m ∈ {16, 64, 256} | photons | fbar at exact ω_in | e3 | + L4, L5, L6, L12 for P | new code (ER sample + ray cast + deposit) |
| 3 | path-traced e2 (X_L2) | — | — | e3 | prefix-exact = ORACLE | yes |
| REF | REF-ER render | | | | | yes |

Levels 1, 1a, 1b, 1c and 2 vs 3 on CK2 box A and the panel face decompose the open WL94D CK2 floor into C3 (1→1b), C5 (single vs pooled), C1/C6 (1b→1c→2) and C2/C4 (P_pts) [INF].

**Measurements.**
- Region E_H and signed error (E2/E5 C, CK2 ROI_T, panel face, box A, the held-out sharp edit) with the half-split noise bound.
- Edit slope in C (0.75–1.25).
- Field-level B₂ vs e2: patch × bin rel L1 and exitance with the **correct** W. Report W94 alongside only for continuity, and use the box A mask, not slice 1130:1238.
- Per-level cost: rays per light, ms per edit (warm, uncontended), memory.

**Controls.**
- Geometry: E2 / E5 / CK2 / held-out.
- Materials: S1, S2, floor B, Lambert-only; textured vs untextured.
- Lighting: point lights, LSPOT, LMIX; grazing (CK2 ring 0) vs near-normal panel incidence.
- Incident angular distribution: report ring shares.
- Directional discretisation: Levels 1–1c fixed at 64 bins; Level 2 continuous.
- Reference noise: 4,096-spp halves.
- Relation realisations: ≥ 16.
- Compute matched in rays per level, reported.

**Stop and decision rules** (freeze before results). Thresholds are proposals for the owner.
- Level 1a single draw ≤ F2 (0.06) and ≤ oracle + max(2 × noise, 0.01) on all predeclared sharp regions, and slope pass → E-A1 holds and H-A proceeds to stage 2.
- Level 1a fails but Level 2 passes → H-A falsified; H-B becomes the candidate; its per-light cost must then be reported against PT at matched quality.
- Both fail while Level 3 passes → the first flight is not the residual; attribute to the readout (L10) or e3. Both hypotheses are falsified as fixes.
- Level 0′ ≈ Level 1 → the learned local operator was not the binding error (contradicts §2.1). Stop.

### Stage 2 — bounded learned-resolvent smoke (only if Stage 1 admits)

- Train Λ_θ on e3 only: WL93 smoke contract (128 geometries, 2,000 steps, 2 seeds). The driver is the Level 1a prefix.
- Arms: Λ_θ; explicit sweeps on the same Γ_s for orders ≥ 3 (H-0); path-traced e3; frozen own G.
- Rank ladder ρ ∈ {8, 16, 24} as a capacity control.

**Falsification:** Λ_θ is no better than explicit sweeps on held-out e3, in field rel L1 or region E_H, and not faster at matched quality → the learned claim is unsupported.

**Decision enabled:** whether any learned global component in the ECRO lineage earns ownership. Full-scale training is not needed for this decision.

---

## 7 Unresolved Questions

1. Is the binned first-flight floor (~2× oracle on E2 C and GM1 R_mirror) acceptable? This is the H-A vs H-B hinge. Owner decision on the sharp gate: F2 0.06 vs an oracle-relative bound.
2. Decomposition of the WL94D CK2 over-delivery (C1–C6). Open; Stage 1 is designed to resolve it.
3. Learned resolvent utility vs explicit sweeps: no measurement exists. WL87 suggests explicit ≥ learned.
4. Can Λ_θ represent sharp-mediated ≥ 3-event families (CK2 panel face e3_p 0.755; G error 0.15–0.20 vs noise 0.027)?
5. Admissibility of H-B's source-dependent light-side state under the reusable-rule objective, and of an event-count composite owner relative to K/C (ECRO's unresolved placement). Owner decisions.
6. The W94 mis-weighting's impact on WL94-series learned and render-derived numbers is unmeasured. Correct W before using them as baselines.
7. Curved and bent sharp patches (L11), near-contact (L10), and area / environment lights for H-B.
8. O(N²) generator scaling beyond ~10³ patches.
9. Whether peak-targeted rank-16 objectives exceed 0.5 peak. Moot under O1, but untested [HYP].

---

## 8 Architecture Verdict

**MULTIPLE_CANDIDATES_REQUIRE_DISCRIMINATION.**

- **Justified by evidence, as an ownership conclusion rather than a demonstrated fix:** the local angular response, including sharp lobes, should be owned by frozen ER at every vertex, not by a geometry-conditioned learned bin operator.
  - [MEAS] S_geo < 1 %; learned at the zero-operator level; rank-16 cap; exact-sender render gains in every inspected case.
- **Not decidable on current evidence:**
  - whether the sharp first flight can stay on a binned, source-independent, reusable relation (H-A) or must be computed per source with lobe-adapted sampling (H-B);
  - whether the shared learned resolvent beats explicit sweeps at all.
- Both questions are answerable mostly learning-free (§6 Stage 1). H-A is the recommended primary for falsification.
- Nothing is adopted. No successor batch is started. No novelty is claimed.

---

## 9 Reproducibility / Sources

**Evidence files read in full** (in `C:\Users\dna10\AppData\Local\Temp\claude\462fa2b2dbb71cdd\46d65e15-e083-4057-875f-9282fb68aa5e\scratchpad\wl95\`): `auditA.md`, `auditB.md`, `read_code.md`, `read_canon.md`, `read_wl86_88.md`, `read_wl89_92.md`, `read_wl93.md`, `read_wl94.md`.

**Code verified this session** (repository root `c:\Projects\ReST-GI`):
- `experiments\ecro86\model86.py:108-124`: apply; tied c; global term driven by u.
- `experiments\fnto_compiled82\closure82.py:4-20,101,148,200`: ER = Lambert + 7 anisotropic GGX lobes, eval / VNDF sample.
- `experiments\gm1_explicit87\quad87.py:80-90`: `fbar`.
- `experiments\gm1_relation88\rel88.py:47-59`: material-only peaked rule.
- `docs\Architecture.md:1-60`: ER role, ECRO resumption decision, K/C contract, §4 ownership table.

**Measured values:** cited in-line to WL86–WL94D worklogs, the result JSONs named in the evidence files, and the Stage B offline calculation (`~/wl95_calc/*.json` on LabServer63).

**Labelling:** cost figures for H-A/H-B are [INF] from measured components; all mechanism claims for H-A/H-B are [HYP].

Nothing was run, trained or edited in this session.