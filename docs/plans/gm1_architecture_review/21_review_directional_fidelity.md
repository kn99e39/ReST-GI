# GM1 review 21 — Directional fidelity and representational limits

**Status:** design-only adversarial review (2026-11-10), `Selective_Recompute` @ `c7b0698`. Nothing was implemented, run or measured. Only this file was written.
**Reviewed (in full):** `10_design_trackA_directional_delta_transport.md` (DDT / DDT-D), `11_design_trackB_path_space_reuse.md` (ERPS / PR1–PR3), `12_design_trackC_alternative_operator.md` (LPTG). No other review file in this folder was read.
**Originals read for this review:** `00_PROBLEM_BRIEF.md`; `docs/RESEARCH_ROADMAP.md` (GM1 row and current-direction paragraph); WL58 and WL84 in full; WL46, 47, 66, 74, 77, 77A, 79, 80, 81, 82, 83 in the sections cited; `experiments/fnto_compiled81/protocol/compiled81.json`, `experiments/fnto_compiled80c/closure80c.py`, `experiments/fnto_compiled82/closure82.py` for the ER lobe ranges. WL51–56, 59, 67, 69, 70 are used only through `AGENTS.md` summaries and are not load-bearing here.
**Axis:** what angular and spatial information each design stores and propagates, and whether sharp ER lobes survive it over ≥ 2 bounces. Cost is discussed only where it is forced by a directional requirement.
**Labels:** FACT, HIST (original scope only), DERIV, INF, HYP, SPEC, EST (order-of-magnitude estimate, not a measurement).
**Literature:** cited from memory; no web search was run in this session, so venues/years are believed correct but unverified. No paper detail is load-bearing; the quantitative statements are my own derivations.

## 0. Summary of verdicts (this axis only)

| Design | Verdict | One-line reason |
|---|---|---|
| A — DDT / DDT-D | **UNSOUND** as a stored 4D directional state below α ≈ 0.05–0.1; **CONDITIONALLY SOUND** only in its path-continuation hybrid, which is Track C's mechanism or a retracer | state size ∝ α⁻⁴; every bounce passes through the state; the dirty set is not thin |
| B — ERPS | **SOUND for fidelity, EMPTY as a representation** | exact directions, no basis; carries no information off its sample support; reuse radius on a lobe of roughness α is ≈ α·r |
| C — LPTG | **CONDITIONALLY SOUND** for S→S→eye and broad R→R; **UNSOUND as specified** for sharp-sender → rough-receiver transport and small occluders | the partition band-limits the receiver lobe, not the incident field; K = 128 point-samples it |

No design, as written, represents GM1's sharp glossy interreflection onto a *rough* stationary receiver. A and C fail it by aliasing; B (and both references) fail it by sampling rate.

## 1. Common quantitative ground

### 1.1 What "sharp" means for the frozen ER baseline

- **FACT.** The closure parameterisation allows lobe roughness α ∈ [0.004, 1.0] (`experiments/fnto_compiled80c/closure80c.py:24`, reused by `closure82.py`).
- **FACT.** The qualified material family has coat α_i ∈ [0.05, 0.5] and base α_b ∈ [0.05, 0.6], both log-uniform (`experiments/fnto_compiled81/protocol/compiled81.json:37`); the WL83/84 hard cluster is the lowest α_b tercile.
- **FACT.** WL84 qualified ER on 3 images, "direct lighting + limited indirect", static, path depth 4 (`docs/worklogs/84_…`, Architecture Verdict table and Implementation Fact).
- **FACT.** The ER sampler/pdf consistency tests (WL82 T5, reused in WL83) are restricted to sampled lobes with α ≥ 0.1 because the uniform-quadrature test is unreliable for near-delta lobes (`docs/worklogs/82_…` item "T5 test design"; WL81 B6: α 0.004 lobes noise-limited).
- **INF.** Every experiment proposed by the three tracks draws materials from the WL83/84 population, so the sharpest lobe actually exercised is nominally α ≈ 0.05. The census of *fitted* lobe roughness in the frozen closures was not checked by any track or by me. A pass on that family says nothing about α 0.004–0.02, and "sharp glossy / specular" in the GM1 row is wider than ER's qualified scope. This is a scope gap for a human decision, not a defect of ER.
- **INF.** ER's fidelity in ≥ 2-bounce glossy chains is unmeasured. Its single-bounce highlight error against the layered reference is 0.033–0.093 with a gate of 0.10 (FACT, WL84), so gates against REF-P leave almost no room for a global mechanism; directional gates must be against REF-ER.

### 1.2 GGX lobe geometry (DERIV, small-angle, away from grazing)

With D(θ_h) ≈ 1 / (π α² (1 + θ_h²/α²)²):

- Half-vector HWHM = α·√(√2 − 1) = 0.644 α. Reflection doubles in-plane deviation and scales out-of-plane deviation by 2 cos θ_i, so the outgoing lobe has in-plane FWHM ≈ 2.57 α and effective solid angle ≈ 4π α² cos θ_i.
- Encircled energy: the fraction of lobe mass with tan θ_h ≤ kα is k²/(1 + k²). The FWHM core holds 29 %; 50 % lies within k = 1; 90 % within k = 3 (outgoing deviation 6α); 99 % within k ≈ 10 (deviation 20α). GGX is a narrow core on a wide skirt. Any truncation to "the lobe" loses 10 % of energy at 6α.
- Grazing: at θ_i = 80° the out-of-plane width shrinks by cos θ_i = 0.17. A lobe that is rough at normal incidence is sharp in one axis at grazing. WL84's target images use grazing light (FACT).

### 1.3 Angular sample count (DERIV + HIST calibration)

Four samples per in-plane FWHM gives spacing 0.64 α and **N_ω ≈ 2π / (0.64 α)² ≈ 15 / α²** directions per hemisphere. Independent calibration from WL58 (HIST; static analytic GGX, piecewise-constant bins, optimal projection): 256 bins lose 10–18 % relative L2 at α = 0.25 (≈ 32 bins per lobe solid angle) and 52–80 % at α = 0.04 (≈ 0.8 bins per lobe); 32 bins per lobe ≈ 16 / α². Both routes agree.

| α | N_ω ≈ 16/α² | lobe count 1/(2α²) |
|---:|---:|---:|
| 0.25 | 256 | 8 |
| 0.10 | 1.6·10³ | 50 |
| 0.05 | 6.4·10³ | 200 |
| 0.02 | 4·10⁴ | 1.25·10³ |
| 0.004 | 10⁶ | 3.1·10⁴ |

This is for ≈ 10 % L2 with a piecewise-constant basis; tighter gates or grazing incidence need more (WL58: loss rises 0.60 → 0.80 from 0° to 60° at α 0.04).

### 1.4 Spatial sampling is tied to angular sampling (DERIV)

On a glossy surface, outgoing radiance at fixed ω varies in x at the rate at which the reflected ray sweeps the reflected scene: Δx ≲ d · Δω, with d the distance to the reflected geometry; on a curved reflector of radius R, Δx ≲ R · Δω / 2. This is the spatio-angular shear of light-field sampling (Chai et al. 2000, SIGGRAPH; Durand et al. 2005, SIGGRAPH). With Δω = 0.64 α:

| α | d = 1 m | d = 10 cm | d = 1 cm (near contact) |
|---:|---:|---:|---:|
| 0.05 | 32 mm | 3.2 mm | 0.32 mm |
| 0.02 | 13 mm | 1.3 mm | 0.13 mm |
| 0.004 | 2.6 mm | 0.26 mm | 0.026 mm |

So a 4D outgoing-radiance table needs N_x · N_ω ∝ α⁻⁴ / d_min² cells. HIST consistent in kind: WL54/55 found 1 cm ports lose near-contact structure (different operator and materials; mechanism only).

### 1.5 Caustic-type hit rate (DERIV / EST)

Path: light → sharp sender (α) → rough receiver → eye, small light. From a receiver point, the highlight on the sender subtends an effective solid angle Ω_h ≈ 4π α² (d_L / (d_L + d_R))², i.e. π α² to 4π α². A cosine-distributed direction from the receiver lands in it with probability p ≈ α² … 4α².

| α | p | spp for 2.5 % per-pixel noise, ≈ 1/(p·0.025²) |
|---:|---:|---:|
| 0.05 | 2.5·10⁻³ – 10⁻² | 1.6·10⁵ – 6.4·10⁵ |
| 0.02 | 4·10⁻⁴ – 1.6·10⁻³ | 10⁶ – 4·10⁶ |
| 0.004 | 1.6·10⁻⁵ – 6.4·10⁻⁵ | 2.5·10⁷ – 10⁸ |

The same p governs three different things below: the reference's noise, Track B's birth rate, and Track C's node aliasing.

## 2. Track A — Directional Delta Transport

### 2.1 What is stored and propagated

A 4D outgoing field b0(x, ω_o) on all surfaces, tabulated or learned, plus exact ER and ray-cast C. Deltas are used for discovery; values are overwritten by regather.

### 2.2 Verified sound

- The identity and its reading as defect correction (A §1). DERIV checked.
- The requirement that b0 be directionally complete at never-consumed directions (A §2), and that a learned b0 supervised on consumed directions is exposed to the WL46/WL47 mechanism. HIST scope: WL46 K44 at unsupervised (port, direction) pairs; WL47 visible-only supervision gave wrong, non-passive hidden outputs on a linear surrogate in a tiny Cycles cavity. Possible mechanism, correctly labelled by the author.
- The angular estimate "10³–10⁴ directions at α 0.04" agrees with §1.3 (10⁴).
- Overwrite instead of additive delta for sharp features (A §5). DERIV checked: an additive removal term must equal the stored value cell-for-cell.
- The author's own verdict (reject as stated) is consistent with this review.

### 2.3 Defects

**A1 — FATAL for sharp lobes: state size ∝ α⁻⁴.** DERIV from §1.3–1.4. A's sizing (10⁵ points × 5·10³ directions, "several GB") corresponds to α ≈ 0.05, reflected geometry no closer than ≈ 10 cm, and ≈ 1 m² of surface. Counterexample: a 5 m² court at α = 0.05, d_min = 10 cm needs 5·10⁵ × 6.4·10³ = 3.2·10⁹ cells ≈ 38 GB in float32 RGB, above both cards in use (12 / 24 GB). At α = 0.02 the count grows ≈ 40× (TB scale); at α = 0.004 ≈ 2.4·10⁴×. Near-contact (d = 1 cm) multiplies N_x by 100 at any α. The table is therefore borderline at the *bottom* of ER's qualified range and infeasible across the closure's permitted range. The author names this as failure condition 7 but sizes it only for the easiest case.

**A2 — MAJOR: every bounce passes through the state; §5's "no collapse in transport" is not what the algorithm does.** In DDT-D a dirty cell is overwritten with K(e1 + C1 b_state): the gather reads the stored field at each hit. A k-bounce glossy chain has therefore been projected onto the state basis k times. DERIV (Gaussian approximation, linear interpolation at 4 samples per FWHM): each lookup adds variance h²/6 ≈ 0.068 α² to a lobe of variance ≈ 1.19 α², i.e. ≈ 5 % peak loss per angular lookup and a comparable loss from spatial interpolation; a glossy-to-glossy chain loses ≈ 10–20 % of highlight peak over two stored bounces. That alone is at the level of the 0.10 highlight gate. The only escape is the path-continuation rule, which removes the state from exactly the chains GM1 targets.

**A3 — MAJOR: the first-order dirty set is not thin.** A §8 assumes the dirty cells are "a thin set in (x, ω_o)". Two properties of ER contradict this (DERIV):
- Every ER material has a Lambert lobe and rough lobes (FACT, WL82 lobe groups). Through them every ω_o cell at every x that sees O0 ∪ O1 changes, at magnitude ≈ albedo · Ω_m / π.
- GGX skirts (§1.2). The lobe mass falling on a mover of solid angle Ω_m at deviation 2kα is ≈ Ω_m / (4π α² (1 + k²)²). For α = 0.05, Ω_m = 0.1 sr and a 1 % tolerance, k ≤ 4.1: the dirty region in ω_o is the mover's reflected footprint dilated by ≈ 0.4 rad, ≈ 1 sr instead of ≈ 0.1 sr.
Consequence: with an unfactored 4D table φ_τ is close to the fraction of surface points that see the mover, unless τ is loose, in which case the saving is bought with bias ("speedup caused only by lower fidelity"). A lobe-factored state (one scalar field for the Lambert response, separate carriers for sharp lobes) is required; that is Track C's partition.

**A4 — MAJOR: overwrite without common random numbers turns estimator noise into apparent change.** The stored b0 is not an exact fixed point of the projected operator; it carries its own gather noise and bias. A fresh regather of a dirty cell re-estimates the *unchanged* part of the integrand with different error. Result: seams between overwritten and retained cells and a non-physical component in every signed-change metric, largest where lobes are sharp. HIST analogue (mechanism only): WL34 seams from independent per-unit error fields. A §6's "delta analytically zero on unchanged rays" holds for the additive form the design then abandons.

**A5 — MAJOR: caustic-type chains through the mover.** Acknowledged (A §4). Quantified in §1.5: receiver-side gathers find them at rate p ≈ α²–4α². Forward emission needs density estimation, i.e. a second representation with its own bandwidth.

**A6 — MINOR: stability of the projected iteration.** A §5's feed-forward error argument holds for exact operators. The Gauss–Seidel overwrite iterates P·K·C1 with P the state projection. P is non-expansive for averaging or non-negative interpolation; for a learned field it is not guaranteed (HIST mechanism: WL46 ρ 9.4 / 17.1 with a non-passive directional sender).

**A7 — MINOR: low-rank compression is unsupported.** A leaves compression "unknown". HIST, different operator (nonlocal medium K, homogeneous slab): shared low-rank angular bases redistributed error rather than removing it (WL74: rank-8 refit trades contrast for absolute/interior error, restart spread 120 %), had a measurable floor (WL66), and a fixed basis was not equivariant under exact symmetry images (WL79: 1.26–1.41×; WL80: orbit-balanced refit still fails). WL77A showed part of such floors was q = 1024 estimator noise, so these are mechanisms, not bounds. For a near-specular surface the rank of b0 over a patch of size s grows roughly like (s / (d·Δω))² (INF from §1.4), so separable factorisations are weakest exactly where A needs them.

### 2.4 What would have to change

1. Replace the 4D table by a lobe-factored state: band-limited field for Lambert + rough lobes, explicit traced continuation for sharp lobes. This is LPTG; A then has no separate directional content.
2. Regather with the T0 build's random numbers so unchanged integrand parts reproduce exactly.
3. Declare the sharpness scope (α_min, d_min) before sizing; add a grazing case.
4. Experiment A4 must include a two-stored-bounce chain mask, not only first-order regions, and report peak attenuation per bounce.

### 2.5 Verdict

**UNSOUND** as a persistent 4D directional state for α below ≈ 0.05–0.1 or with near-contact geometry. **CONDITIONALLY SOUND** only where the state holds band-limited components and sharp lobes bypass it, at which point it coincides with Track C (if chains are stored) or with a difference retracer (if they are not).

## 3. Track B — Path-space / reservoir reuse

### 3.1 What is stored and propagated

Sampled paths with exact vertices, exact directions and random numbers. No angular or spatial basis. Spatial content: point samples at the T0 sampler's density, tied to pixels or surface cells. Angular content: one direction pair per vertex per sample.

### 3.2 Verified sound

- No compression, so the WL58 / WL47 obstructions cannot occur (B §2). DERIV: PR1 equals the same-seed REF-ER render; PR2 is unbiased in expectation.
- A k-bounce glossy chain is carried at full sharpness; the limit is sample density, identical to REF-ER (B §5).
- The statement that hybrid shifts degenerate to fresh sampling on glossy chains through the mover (B §4), and that a stored suffix is poorly distributed for another outgoing direction at a sharp vertex.
- Segment-level revalidation covers changed occlusion with unchanged endpoints.
- The author's verdict (not a GM1 mechanism; keep as warm comparator and headroom measurement) is consistent with this review.

### 3.3 Defects and limits

**B1 — MAJOR (limit, acknowledged; quantified here): reuse radius on a sharp lobe is ≈ α · r.** DERIV: reconnecting through a vertex with lobe roughness α multiplies the contribution by D(h′)/D(h) = ((1 + θ²/α²)/(1 + θ′²/α²))². It collapses once the direction at that vertex changes by more than ≈ 1.3 α. A vertex displaced by Δ at distance r changes the direction by Δ/r, so reuse survives only for Δ ≲ 1.3 α r: 6.5 cm at α = 0.05, r = 1 m; 5 mm at α = 0.004. Any GM1 edit with "disjoint poses" voids all reuse on sharp chains that touch the mover *and* on stationary sharp vertices whose next vertex is on the mover. The store holds no directional information that transfers.

**B2 — MAJOR: births of caustic-type paths, and the reference itself.** B §4 and §10.1 acknowledge the birth problem. §1.5 gives the rate: p ≈ α²–4α². Two consequences B does not draw:
- A WL84-grade reference (65536 spp) has per-pixel relative noise ≈ 1/√(pN) ≈ 0.04–0.08 in that mask at α = 0.05, above the "≤ ¼ of the smallest gated error" rule for a 0.10 gate. The mask B's F3 is judged on is reference-limited with a unidirectional tracer. Below α ≈ 0.02 it cannot be qualified at all with the existing harness (EST).
- F3 "expected by derivation" then cannot be distinguished from reference noise. HIST precedent for exactly this confusion: WL83's sole ER failure was a reference-qualification artefact (WL84).

**B3 — MINOR: the lobe-width descriptor is direction-dependent.** Reconnection eligibility needs an outside-direction footprint. For ER's refracted inner lobes the outside width is ≈ η α · cos θ′ / cos θ (DERIV from Snell), unbounded toward grazing exit; for any lobe the out-of-plane width scales with cos θ_i (§1.2); anisotropic lobes have two widths. A per-lobe scalar is not sufficient. B flags the refraction part only.

**B4 — MINOR: depth.** B fixes depth to REF-ER's WL84 value (4). With conductor reflectance up to 0.98 (FACT, box) a depth-4 "reference" truncates most of the multi-bounce energy GM1 is about. Both arms share the truncation, so it is a scope defect, not a bias between arms.

**B5 — MINOR: PR3 history in a moved highlight.** A high-confidence T0 reservoir sitting where the highlight used to be is the canonical ghost. B states the condition for unbiasedness; it should be a gated control, not an optional extension.

**B6 — MINOR: sampler evidence.** Replay and shift Jacobians rely on ER `sample`/`pdf` for lobes with α < 0.1, where the WL82/83 consistency tests did not run (§1.1). WL84's highlight results on the hard material are indirect support. A dedicated lobe-space test (not uniform quadrature) should precede any Jacobian-dependent arm.

### 3.4 What would have to change

1. A separate caustic-type mask with its own reference qualification (larger light, a light-tracing or bidirectional estimator, or a declared exclusion). Mitsuba 3's particle tracer may serve; not verified here.
2. Depth by Russian roulette, as Track C proposes.
3. Report the reuse radius of B1 as a measured curve (reuse benefit vs Δ / (α r)).

### 3.5 Verdict

**SOUND for directional fidelity** (it is REF-ER's estimator), **EMPTY as a directional representation**: zero information off the sample support, no transfer across views, and a reuse radius that shrinks linearly with α. On this axis it is the correct control and cannot be the mechanism.

## 4. Track C — Lobe-Partitioned Transport Graph

### 4.1 What is stored and propagated

- Field I(c, j): indirect *incident* radiance at N cache points on K fixed material-frame directions, one ray (plus one sharp chain sample) per node.
- B per material: K × K table of the rough-lobe response.
- Sharp lobes: stored one-sample chains inside links; fresh chains at readout; exact ER NEE at every vertex.

### 4.2 Verified sound

- f_ER = f_R + f_S is an exact partition and maps onto K/C ownership (DERIV checked).
- **S → S → eye survives at full sharpness.** Camera → sharp receiver → sharp mover → light is a fresh chain with exact ER evaluation; no basis is crossed. This is the bypass WL58's candidate lacked (HIST), and it covers the glossy-to-glossy chain of GM1 when the stationary receiver is itself sharp and seen by the camera.
- Light → vertex is always exact, so collimated incidence never enters the finite interface (the WL58 failing input).
- Readout evaluates rough lobes exactly toward the camera; there is no outgoing basis at the eye.
- The K ∝ 1/α*² wall is stated honestly (C §7).

### 4.3 Defects

**C1 — MAJOR (FATAL for the sharp-sender → rough-receiver class at the declared K): the partition band-limits the receiver lobe, not the incident field.** The bandwidth of I(c, ·) is set by what the receiver *sees*: sender lobes and occlusion edges. B's quadrature is valid only if I is band-limited to the node spacing; the design point-samples it with no prefilter. Counterexample (DERIV): rough stationary receiver, sharp sender (α_S = 0.05, e.g. the mover), small light. Link (c, j) hits the sender and takes exact NEE through the sharp lobe at direction −ω_j. The highlight subtends Ω_h ≈ π α²–4π α² = 0.008–0.031 sr from the receiver (§1.5); one node cell is 2π/K = 0.049 sr at K = 128. Expected nodes inside the highlight: 0.16–0.64. Each cache point therefore reports either zero or 1.6–6× the true contribution. Because nodes are fixed in the material frame, all cache points on a planar receiver share directions, so the error is not noise but a coherent pattern: up to K shifted copies of the sender's highlight (the fixed-direction artefact known from instant-radiosity-style gathers; Keller 1997, SIGGRAPH). Resolving it needs ≈ 16 nodes per highlight, K ≈ 3·10³–1.3·10⁴ at α_S = 0.05, the same 1/α² wall the partition was meant to avoid. C lists this as "the weak path class" (§10.2, §14.4); on this derivation it is a breakdown of the field, it occurs in static scenes too, and it is the GM1 headline phenomenon whenever the stationary receiver is rough. C's region list (highlight ROI, changed stationary ROI, unaffected, mover) has no mask that isolates it, so a pooled L1 can hide it.

**C2 — MAJOR: at K = 128 the threshold α* is ≈ 0.17–0.34, and most of the family's specular lobes fall in S.** DERIV: node spacing √(2π/128) = 0.22 rad; 2 to 4 samples per in-plane FWHM (2.57 α) gives α* ≥ 0.17–0.34; the WL58 calibration (256 bins ↔ 10–18 % at α 0.25) gives the same range. From the descriptor box (FACT), P(α_i ≥ 0.25) = 30 % and P(α_i ≥ 0.35) = 15 % under the log-uniform draw. INF (the fitted outer lobes are assumed to track α_i; census unchecked): for 70–85 % of materials the coat highlight is an S lobe with α 0.05–0.3. Those lobes are too broad to trace cheaply and each stored link carries a single frozen sample of them, so the field inherits frozen one-sample noise and the readout is a path tracer of depth L̄. C §10.4 names the mid-roughness band as 0.05–0.15 (SPEC); the derivation puts its upper edge at α*, i.e. two to three times higher. Lowering α* to 0.05 needs K ≈ 6.4·10³: B alone is 3K² ≈ 1.2·10⁸ floats per material and one solve iteration N·K² ≈ 8·10¹² multiply-adds at N = 2·10⁵.

**C3 — MAJOR: small occluders and small senders alias on the nodes.** C §3 states the limit ("an occluder that falls between all K nodes changes nothing") and files it under "thin or small mover". At K = 128 a node cell is a cone of ≈ 14° diameter (DERIV), so a 10 cm box farther than ≈ 45 cm is seen by at most one node. Its indirect shadow and its rough reflection on stationary receivers are then quantised to 0 or one node weight, and they jump as the box moves across node directions. The signed-change slope/correlation on stationary rough receivers (a GM1 measurement) would be dominated by this. F ≡ S holds and is irrelevant: both are wrong together, which is why H-edit cannot detect it.

**C4 — MINOR to MAJOR (depends on K): R → R does compound in amplitude near α*.** C §5 says the loss "does not compound in sharpness". Each R bounce reads outgoing radiance through B's node interpolation. DERIV (Gaussian approximation, tent kernel of width 0.22 rad): added variance 0.008 rad² against lobe variance (1.09 α)², i.e. peak loss ≈ 19 % per bounce at α = 0.17 and ≈ 5.5 % at α = 0.34; two bounces ≈ 34 % and ≈ 11 %. Lobes just above α* are attenuated bounce by bounce.

**C5 — MINOR: R / S classification is direction-dependent.** A fixed per-material mask ignores (i) grazing narrowing by cos θ_i (a lobe with α = 0.25 has out-of-plane FWHM ≈ 0.11 rad at 80°, below node spacing; WL58 HIST: loss at α 0.25 rises 9.9 → 17.7 % from 0° to 60°), (ii) anisotropic lobes (must be classified by min(α_u, α_v)), (iii) refracted inner lobes whose outside width varies with exit angle. C's Stage 0 includes grazing; it should gate on it per lobe.

**C6 — MINOR: fixed nodes are not rotation-equivariant.** Rotating the mover rotates its node set; the same physical configuration gives a different quadrature. HIST mechanism (different operator): WL79, fixed angular basis 1.26–1.41× worse on exact symmetry images; WL80, orbit-balanced refit insufficient. A rotation control (rotate the whole scene rigidly; the result must not change beyond a declared tolerance) belongs in Stage 0, otherwise node error can be read as edit response.

**C7 — MINOR: sampler evidence.** As B6: the S-restricted sampler works exactly in the α < 0.1 range where the existing consistency tests did not run. C's own χ² identity test must use a lobe-space estimator.

### 4.4 What would have to change

1. Break the incident-side aliasing: per-cache-point stratified jitter of link directions with f_R evaluated exactly (K evals, as at readout) instead of a fixed-node B, plus spatial filtering; or prefiltered (cone / footprint) link queries; or explicit connections for sharp-sender highlights (specular NEE / manifold sampling, Zeltner et al. 2020) or a forward light pass. Each changes the cost model and the F ≡ S argument must be restated.
2. Fix α* from §1.3 and report the lobe census *by lobe roughness*, not only the albedo share in R; add a predeclared stop if the S share of specular energy exceeds a threshold.
3. Add two regions to the experiment: an LSDE mask (rough receiver lit via the sharp mover) and a small-occluder mask (mover subtending less than one node cell), each with its own reference qualification (§1.5).
4. Add the all-nodes-shared vs jittered ablation to Stage 2; it separates C1 from cache-spacing error.
5. Stage 0 additions: grazing gate per lobe, rotation-equivariance control, two-bounce R → R peak attenuation.

### 4.5 Verdict

**CONDITIONALLY SOUND** for (a) sharp chains seen by the camera, including glossy-to-glossy ≥ 2 bounces, and (b) broad R → R transport with lobes well above α*. **UNSOUND as specified** for sharp sender → rough receiver and for occluders smaller than a node cell; both are GM1 target phenomena on stationary receivers. The structure can be repaired on paper (§4.4.1), but the repaired design is no longer the one costed in C §7.

## 5. Cross-cutting findings

1. **Scope of sharpness (human decision).** GM1 says "sharp glossy / specular"; ER is qualified at α ≥ 0.05 and its sampler tests at α ≥ 0.1; the closure permits 0.004. Angular requirements scale as α⁻² and 4D requirements as α⁻⁴: from 0.05 to 0.004 these are 156× and 2.4·10⁴×. No design or experiment in the three tracks states which α the claim is for.
2. **Reference qualification in caustic-type masks.** With the existing unidirectional harness the reference is noise-limited there at α = 0.05 even at 65536 spp and unobtainable below α ≈ 0.02 (§1.5, EST). All three experiments say "qualified as in WL84" without this mask.
3. **Signed-change metrics under noise.** Slope and correlation of T1 − T0 are attenuated by noise in the reference difference (regression dilution); in sharp regions the difference noise is √2 times the image noise against a small signal. Each experiment needs a split-half qualification of the *difference* image per mask. WL77A (HIST) is the project's precedent for estimator noise being read as a representation floor.
4. **GGX skirts.** Any rule that treats a lobe as its core (dirty marking in A, reconnection eligibility in B, R/S footprint in C) mis-assigns 10 % of energy at 6α and 1 % at 20α.
5. **Grazing.** Lobes narrow by cos θ_i in one axis; the WL84 target images are grazing-lit; A's and C's sizing use normal-incidence widths.
6. **Depth.** B freezes depth 4; C uses Russian roulette; A is silent. Comparisons across tracks are not on one reference.

## 6. Cross-design comparison (this axis only)

| Question | A — DDT-D | B — ERPS | C — LPTG |
|---|---|---|---|
| Angular information stored | N_ω directions of outgoing radiance per point; needs ≈ 16/α² | exact, per sample only | K incident nodes (band limit ≈ α* ≈ 0.17–0.34 at K = 128) + exact sharp chains |
| Spatial information stored | N_x points; needs Δx ≲ d·0.64α | sample positions only | cache points for the rough field; continuous hits for chains |
| Sharp lobe, 1 bounce, seen by camera | band-limited by state; ≈ 5–10 % peak loss at 4 samples/FWHM | exact | exact |
| Glossy → glossy, ≥ 2 bounces, seen by camera | compounding: ≈ 10–20 % peak loss, or path continuation (= C / retracer) | exact, at REF-ER sampling rate | exact (fresh chain) |
| Sharp sender → rough receiver (caustic type) | not found by receiver gathers (rate α²–4α²) | found at rate α²–4α²; reuse adds nothing | aliased: 0.16–0.64 nodes per highlight at α 0.05, coherent pattern |
| Newly visible glossy path | needs b0 at never-consumed directions | fresh sampling only | regenerated links; exact if receiver is sharp, aliased if rough |
| Changed occlusion, unchanged endpoints | chord test at sample resolution | exact segment test | segment test; occluders < 14° cone quantised |
| Energy lost by truncation / compression | τ threshold on a dense dirty set; low-rank unsupported | none | frozen one-sample chains; R → R peak loss near α* |
| Scaling as α → 0.004 | infeasible (α⁻⁴) | fidelity unchanged; reuse radius → 0; reference unobtainable | S share → all specular energy; field carries Lambert only; C1 worsens as α² |
| Hides a path tracer | yes, in the continuation hybrid | is one, openly | yes, when the lobe census is S-dominant |
| Verdict | UNSOUND (stored 4D state) | SOUND fidelity / EMPTY representation | CONDITIONALLY SOUND; UNSOUND for S → R and small occluders |

**Ranking on directional fidelity:** B (exact, but not a representation) > C (exact where sharp lobes meet the eye, wrong where they meet a rough receiver) > A (band-limited everywhere unless it turns into C or B).

**Common residue.** After removing what is prior-art path tracing, the only persistent directional content any of the three designs can hold at ER's sharpness is a band-limited field with α* ≈ 0.2–0.35 at practical sizes. Whether that field carries enough of the coated-conductor family's energy to matter is the lobe census none of the tracks has done; it is a zero-render measurement and should precede any other GM1 work on these designs. This is a statement about the three reviewed designs, not a recommendation of a direction.

## 7. Limits of this review

- All numbers in §1–§4 are small-angle derivations or estimates with Gaussian / single-lobe approximations; none is a measurement. Factors of 2 are not significant; the α-scaling is.
- The fitted-lobe roughness census of the frozen ER closures was not inspected.
- WL51–56, 59, 67, 69, 70 were not re-read in the original; they are not used for any claim above beyond mechanism analogies marked HIST.
- Literature is cited from memory without verification in this session.
- The historical angular results (WL46, 47, 58, 66, 74, 77, 79, 80) concern other operators, materials or interfaces; they are used as possible failure mechanisms only and are not measured defects of ER or of the three designs.
