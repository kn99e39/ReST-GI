# WL95 adversarial critique (physics, representability, evidence consistency) of the "#state-reuse" design (H1 DSR, H2 ALP)

**Scope.** I edited no repository file, trained nothing and ran no GPU work.

**Labels.** [FACT] verified in code or a document · [MEAS] measured value with its source · [INF] reasoned inference · [HYP] untested.

**Code checked in this pass** (all under `c:\Projects\ReST-GI`):
- `experiments/ecro86/model86.py:40-46`
- `experiments/ecro86/ecro86.py:57-87, 161-168`
- `experiments/gm1_relation89/rel89.py:30-80`
- `experiments/gm1_c3urx1_91/cand91.py:36-110`
- `experiments/gm1_explicit87/xport87.py:118-125`
- `experiments/ecro_mat94/an94.py:17-33`
- `docs/Architecture.md:13`
- `docs/RESEARCH_ROADMAP.md:125-128`

Everything else comes from the eight Stage A/B and reader files in the scratchpad `wl95\` folder.

---

## 0. Cross-cutting findings (these affect both hypotheses and the overall verdict)

### X1. Both hypotheses conflict with a standing human decision, and the design cites a superseded open question

[FACT] `docs/Architecture.md:13` (human, 2026-11-13) says: keep "EXP/C3UR as learning-free explicit transport comparators/diagnostics rather than the global method to be published. **Later X1-style first-gather findings constrain information preservation but do not justify converting the project to explicit GI.**"

[FACT] `docs/RESEARCH_ROADMAP.md:128` adds three constraints:
- the next decision must isolate "one falsifiable **ECRO-lineage** representation/training/ownership hypothesis";
- "Do not … promote an explicit candidate as the project goal";
- explicit EXP/C3UR and the WL87–WL90 diagnostics "remain immutable null/comparator evidence, not a replacement for learned global transport".

How H1 and H2 sit against this:
- [FACT, from the design] Both have no learned transport object. H1's only learned object is the frozen material ER. H2 adds an offline per-material lobe fit.
- [INF] H1 is literally an X1-style explicit first gather plus explicit sweeps, i.e. the C3URX1 class. H2 is the same pipeline with analytic quadrature.

The design's §5/§7 qualification treats "is a learning-free global solve around ER a candidate or a baseline?" as open. It cites `docs/plans/gm1_architecture_review/60_coordinator_synthesis.md:94-100` (the WL84A era). That question was answered on 2026-11-13, two decisions later.

**Consequence.** Presenting H1/H2 as candidate architectures (verdict MULTIPLE_CANDIDATES_REQUIRE_DISCRIMINATION) contradicts current canon. They are admissible only as comparator or oracle arms. This is fatal to the verdict as written, not to the physics.

### X2. H1 and H2 are not materially distinct architectures

[FACT, design text] Shared between H1 and H2:
- the same invariance table and state classes (f_m, frames, relation R, singular records D);
- the same ownership (ER local, explicit relation for propagation);
- the same orders ≥ 2 ("Later orders as in H1");
- the same edit and invalidation rules (H2: "As H1, plus pair re-evaluation");
- the same readout and the same B contract.

They differ only in the first-gather quadrature (MC versus analytic polygon). The brief asks for materially distinct hypotheses on how to represent local angular response and nonlocal transport. These are two estimators of one design.

[INF] Even accepting both, the discrimination asked for is an estimator choice inside an explicit comparator. It is not an architecture decision.

### X3. The decisive learned-lineage experiment is missing, although it is learning-free and its inputs exist

[MEAS auditB] Stage B established three things:
- Learned S1 sits at the zero-operator level in the WL94D operator L1 metric (0.96–1.04 against 1.000).
- Non-negative rank 16 reaches field L1 0.36–0.54 at peak 0.35–0.44.
- Rank about 32 reaches field L1 ≤ 0.15 with peak ≥ 0.7.

auditB lists as **unmeasured**: "sender delivery and E_H for the bound operators substituted at S1". The bound operators are already saved: `~/wl95_calc/S1_ops.npz` on LabServer63 (nmf16_wl92, nmf16_l1irls, fa_nmf16_l1, svd24/32, rrr*).

The design jumps from "the learned recipe does not reach the rank-16 bound" to "remove the learned sender". That skips the question the canon asks for the ECRO lineage: would a sender at the admissible non-negative rank-16 (or rank-32) bound be delivery- and E_H-sufficient?
- If yes, the failure is optimisation or objective, and a learned sender remains viable.
- If no, a representation limit is shown, now in matched quantities (delivery and E_H, not the operator norm).

Its oracle hierarchy has no such arm (O0 learned, O1 M64, O2 qtable, …). Without it the design cannot answer the brief's central question about learned local angular representation.

### X4. The design omits ECRO-workload evidence that weakens "exact per-ray sender radiance"

The design cites the GM1 evidence (WL90 and WL91 pooled X1 = oracle). It omits two measurements made on the ECRO E2 workload itself:
- [MEAS WL87:222-231] **RAYSEND** (exact once-scattered sender radiance in the first gather, E2 T1 C) is 0.059 (signed −0.058), against EXP 0.049. Worse, not better.
- [MEAS WL86A Amendment 2] **RAY vs BIN** (exact against bin-averaged panel radiance on the deployed state):
  - E2 T1: reading 0.144 against 0.153 (truth 0.181); patch RMS 0.50 against 0.38.
  - E2 T0: reading 0.093 against 0.086 (truth 0.116); patch RMS 0.93 against 0.47.
  - So exact per-ray radiance roughly doubled patch-level error.

[MEAS] On E2, the lever that worked was sender placement:
- C3 0.024 and C3UR 0.026 against R0 0.049 (WL87; WL88 Task A +0.050 / +0.053).
- That was with a **binned** sender.
- The remaining projection share there is bounded by pooled EXP / ENCB 0.027 against ORA2 0.0127 (WL92 ladder), i.e. ≤ about 0.014 E_H.

[INF] On the ECRO workload, deferred evaluation can at most attack the about 0.014 pooled-to-oracle residual. Sampling placement (L5) dominates the deployed error. The design's "mechanism evidence" bullet mixes the two.

These belong under "Weakens" for H1. They are receiver-anchored, so they do not falsify the sender-anchored H1. They do show that exactness alone is not the ECRO lever.

### X5. Cost evidence and the economic bar

[MEAS] Comparable numbers that are already measured:
- WL91 (GM1 workload): C3URX1 source update 75.8 ms against C3UR 92.2 ms; edit-to-frame 124.7 ms. Quality-matched utility fails (3.1× slower at the R_mirror 0.06 tier).
- WL87 (ECRO workload): source-only update ECRO 13–18 ms against EXP 51–64 ms; per-edit PT 36 ms, EXP 89–95 ms.

The two workloads differ, so cross-workload ms ratios are **unmatched** and indicative only.

[INF] Even so, H1 is expected to:
- forfeit ECRO's only measured learned advantage, the 3.5–3.9× source-only update;
- inherit the C3URX1 utility failure risk.

The design lists this as open question 10. Existing measurements already point negative, so it belongs in "Weakens".

The design also quotes "68–125 ms measured for relatives". That mixes WL90 per-edit (68.2 ms, GM1) with WL91 edit-to-frame (124.7 ms), which use different timing boundaries.

The §6 acceptance "per-edit cost ≤ 2× EXP" is not the canonical bar:
- [MEAS] EXP is already 2.5–2.6× PT per edit (89–95 against 36 ms). So the rule admits about 5× PT.
- The binding bar is quality-matched edit-to-first-frame against PT re-render under the four-role gate (`BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md:23-32`; roadmap:27-31).
- Source-only update against ECRO must be reported separately.

---

## 1. H1 — DSR (Deferred-Sender Relation)

### Transport and units check (passes)

[FACT, radiometry] The following are correct:
- `G L(x,ω) = L(r_g(x,ω), −ω)`, the point-light `q(y,v) = Σ_l f(ω_l,v)·I cosθ_l/d²·V`, and the throughput-measure identity.
- The deposition estimator `Ê(n,m) = (1/A_n) Σ_p (A_p/P) Σ_j (1/K_j) Σ_k q·cosθ_y/p(v)·1[…]`:
  - q·cos/p has units W/m²·sr/sr, i.e. exitance;
  - × A_p/P gives flux (W);
  - / A_n gives receiver bin irradiance (W/m²).
- `L_2 = Σ_m Ê·M64[m,d]`, which is exact ρ/π for Lambert.
- Sender-anchored deposition needs no receiver cosine, so L6 (ring-cosine aliasing in `rw`, `model86.py:40-46`) genuinely disappears.
- No double counting against LOW: Ê is the 2-event family, matching e2_p and ECRO's single-flight scope.

### Fatal flaw (as a candidate method)

- **F1 (X1).** H1 is the X1-style explicit GI that `Architecture.md:13` excludes as the published method. It is admissible only as a comparator or oracle arm.

### Serious issues

**S1. The variance claim is internally inconsistent with the stated budget.**
- [FACT, design] "Relation: N·K = 650k rays" means K = 512 per patch, and P = 16 sample points per patch. That gives K_j = 32 rays per y_j, fewer than the 64 outgoing bins.
- So "cosine-stratified per outgoing bin … per y_j" is impossible at this budget. The stratification must be across (j, bin), and "O(κ_s) rays per sample point" is false. At the deployed budget it is κ_s = 8 per bin **per patch**, ≤ 0.5 per bin per y_j.
- [INF, arithmetic with `xport87.py:118-125` (z² uniform within a ring; equal count per bin)]:
  - Ring 0 holds 128 rays per patch.
  - The CK2 footprint-feeding exit band z ≈ 0.10–0.16 (auditA §3.2) holds 128·(0.16² − 0.10²)/0.25² ≈ 32 rays per patch over 0.377 sr.
  - A lobe of about 0.07 sr (r50) therefore gets about 6 rays per panel patch.
  - Only 2 panel patches feed the footprint, so about 12 lobe rays carry the half-mass that lands on 3 receiver patches: about 4 per receiver patch.
  - Expected per-draw relative error per footprint patch is about 0.5. That is before the 307× irradiance gradient sampled by only 16 origins.
  - Meeting the design's own box-A rel L1 ≤ 0.12 needs on the order of 16–25× density on the feeding senders.
- [INF] The proposed kill "fails at ≤ 4× peaked density" is therefore nearly predetermined on CK2. This should be computed from geometry alone on CPU before any smoke is frozen.

**S2. Sharing 16 origins per patch is a regression against C3UR/C3URX1, not a variant of it.**
- [FACT] C3UR draws a fresh `patch_point` per ray (`rel89.py:58-63`, PCG32 keyed by ray id). That gives 512 distinct origins per patch, each with its own NEE in X1 (`cand91.py:78-84`).
- H1 collapses these to 16 origins, so 32 rays share one E_l(y_j). This keeps L2 (16 fixed points, measured 1.2–4.6 % elsewhere [MEAS WL86A]; open on CK2, auditA C2/C4) and correlates the bundles.
- The saving is real: about 20k against 526k shadow rays per light, and no probe rays.
- [INF] It is a cost-for-variance trade on a mechanism whose single-draw variance already failed held-out validity (1/3, [MEAS WL91:310-317]).
- WL88's "sender placement dominant" evidence is for fresh-origin C3UR. Transferring it to shared-origin H1 is [HYP].

**S3. "Ê is unbiased" is overstated.**
- Points "stratified and fixed in patch parameter space" are a deterministic quadrature. It is unbiased only over random draws of the pattern.
- If the pattern is shared across patches (the ECRO practice, `model86.py:40-46`), it reproduces the L2/L5 fixed-pattern bias class.
- On the sphere (8×4 parameter patches), parameter-uniform is not area-uniform. Unbiasedness needs area-weighted points.
- The label should be [INF], conditional on the sampling design.

**S4. "The outgoing side is never tabulated" is false or uncosted for orders ≥ 2.**
- `f̄_m(bin_i → v_k)` at continuous v is either:
  - (a) a per-ray, source-independent 64-vector. This is EXP's `W_ray` construction transposed: 0.17 M non-Lambert rays × 64 × 3 fp32 ≈ 130 MB. Or:
  - (b) an online 8×8 incident-cell ER average per (ray, lit incident bin) per sweep. That is up to 64 × 64 ER evaluations per ray per order.
- The cost section counts only first-gather ER evaluations. The orders ≥ 2 material cost and memory are missing.
- Incident-bin averaging uniform in solid angle without the cosine (L12, `quad87.py:48-56`) returns at sharp senders for orders ≥ 2.
- [MEAS WL94B] CK2 panel-face e3_p share is 0.755 and box A 0.166, far above GM1, where WL90 P2plus ≤ 0.0009 was measured. Extrapolating the GM1 P2plus result to CK2 is unsupported.

**S5. L3/L7/L8 are "removed" by deleting the learned component, not by addressing it.**
- This is relabelling with respect to the brief's learned-representation question.
- [MEAS auditB] The learned S1 sits at the zero-operator level only in operator L1. At field level it is 0.85–0.89 with energy ≈ 1 (overlap about 0.4, against NMF16 0.72).
- The design's §2 phrase "has not even reached the zero-operator level" misreads this: learned values are 0.961–1.044, i.e. at the zero level, not above it, and they are informative at field level.

**S6. The α-scaling argument does not discriminate.**
- Rays per lobe in H1 scale as α², exactly as outgoing-table resolution scales as α⁻². The design's worst case (iii) admits this.
- [MEAS WL90:199-204] Outgoing sub-bin refinement a = 16 left −0.003 to +0.0012 of J in GM1 (a single-read lower bound).
- So "Alternative 1 rejected because tabulation ∝ α⁻²" is not supported. A 16× outgoing physical table is an untested competitor of equal standing, with cost projected but not measured.
- What does separate them: per-light table re-fill (table) against per-ray evaluation (H1), and bias (table) against variance (H1).

**S7. The deformation claim is overreaching.**
- "Deformation is admissible without new validity certificates" conflicts in wording with `Architecture.md:62-64` (non-rigid validity cannot be inferred; missing information means conservative invalidation).
- The f_m invariance is plausible [INF]. But:
  - y_j parameterisation, frames and every relation and record from a deformed surface are invalidated;
  - the existing dirty test is rigid-OBB-specific (`ecro86.py:57-67`, `_seg_obb_t` on old and new box OBBs);
  - a deformation needs a swept-volume test that does not exist.
- Label as [HYP] and state the missing test.

**S8. Invalidation cost of geometry conditioning is a weak differentiator.**
- [MEAS WL86:214-215] The ECRO generator is about 11 ms of a 76–78 ms edit.
- H1's global J sweeps also recompute everything per edit (WL86B (ii): no resolvent edit locality).
- The measured cost of "invalidation coupling through the N² generator" is small. Its true weakness is accuracy, not update cost.

### Genuinely supported

- [FACT] The invariance classes are correct, and the selective-equals-full contract exists to template (`ecro86.py:161-168`).
- [FACT] For point lights, deferred evaluation at the first gather removes L1, L12, L4 and L6 (and L11 at the sender) by construction.
- [MEAS] Mechanism-class evidence:
  - WL90: angular first-gather projection explains 97–102 % (GM1).
  - WL91: pooled X1 = oracle in 6/6 conditions.
  - WL88: target identity across receiver, area and sender constructions.
  - WL93D: S1 S_geo 0.2–0.8 % (geometry conditioning of the local operator carries little).
- [FACT] As an oracle or comparator arm, a pooled H1 is well defined and is the natural reference for "exact sender + higher-fidelity propagation" in the brief's oracle hierarchy.

### Changes needed for a valid falsifiable statement

1. Re-scope H1 as a **comparator/oracle arm** (C3URX1 with shared-origin NEE), not a candidate method.
2. Fix the budget arithmetic: K_j, the per-(j, bin) stratification and the pdf. Report predicted lobe rays per footprint receiver patch from geometry before freezing. Keep fresh-origin C3URX1 as the control for shared origins.
3. Cost the orders ≥ 2 deferred evaluation (option a or b in S4). Declare which one is used.
4. State the kill relative to **C3UR on the same sender-anchored state with binned q**, not relative to O1. Sender placement alone can satisfy "≥ 50 % of the O1→REF gap" (WL87/88). Only the binned-to-deferred difference isolates the H1 mechanism.
5. Move WL87 RAYSEND, WL86A RAY/BIN and the WL91 cost/utility results into "Weakens".

### Verdict H1

**NOT_VIABLE as a primary architecture hypothesis** under the current canon: it is X1-style explicit GI (`Architecture.md:13`), and its central mechanism is C3URX1 with fewer distinct origins. Its adoption evidence was INSUFFICIENT (WL91), and its expected single-draw variance at practical density is high on CK2 (S1).

**VIABLE_WITH_CHANGES as a pooled oracle or comparator arm** in a learning-free attribution smoke.

---

## 2. H2 — ALP (Analytic Lobe–Polygon)

### Transport and units check

[FACT/INF] `Φ_{p→n} = Σ_j (A_p/P) Σ_l E_l·ρ_m(ω_l)·I_poly(M⁻¹Ω_n)·V̄` is dimensionally correct (W). The LTC identity ∫_P D = ∫_{M⁻¹P} D_o is the standard form. The integrand must be f·cosθ_v (the sender exit cosine), normalised to the directional albedo; the design writes that.

### Fatal flaws

- **F1.** The same canonical exclusion as H1 (X1), plus an explicit O(N_g·N) pair table. That is the "large explicit transport table" risk the brief names (compare WL52 / WL80A C4).
- **F2. Scalar pair visibility V̄_pn conflicts with the evaluation target.**
  - [FACT] Region C is defined as "mirror footprint enabled or blocked, or panel/sphere receivers whose mirror cone sees a change" (`ecro86.json`; WL86:51).
  - In a sharp interreflection, a partial blocker removes the lobe entirely or not at all, depending on lobe direction. A lobe-unweighted V̄ from m = 4 uniform rays per pair multiplies the whole lobe integral.
  - [INF] The bias is unbounded (up to 100 %) precisely in the visibility-change region the method is gated on. The brief requires "accurate visibility and interreflection changes".
  - The design mentions "loses the correlation between occlusion and lobe direction near blockers" but does not recognise that this is the gated region.

### Serious issues

**S1. The kill threshold is transplanted from an unmatched norm.**
- "Material fit gives field L1 > 0.15 or peak < 0.7 on S1" borrows auditB's rank-ladder numbers.
- Those are M64-referenced: the binned 64×64 operator with M64's uniform incident-cell averaging, field-weighted by the panel incident distribution, with W.
- An LTC evaluated at continuous directions can only be scored in that norm after binning. That re-applies exactly the projections (L4, L12) H2 claims to avoid. It also tests a different quantity from the one H2 uses (the polygon-integrated flux).
- The matched metric is the polygon-flux error ∫_{Ω_n(y)} f cos dv against dense ER quadrature, at the actual (ω_l, receiver polygon) configurations. The design's own ALP-oracle supplies this.

**S2. Two material surrogates for one BRDF.**
- The first gather uses LTC; orders ≥ 2 use ER/f̄. They agree only in total directional albedo.
- So the per-order material is inconsistent. A coated conductor (two lobes: coat and base) is not single-LTC-shaped.
- [MEAS] CK2 incidence is entirely ring 0, cos θ 0.024–0.164 (auditA §3.2). That is the regime where LTC fits are weakest [INF].
- The brief's "no multi-model decomposition without physical or computational justification" is met only on variance grounds. The design must state the justification and the inconsistency.

**S3. The gain over H1 is variance only, and the evidence for that gain is circumstantial.**
- WL91's variance was measured on a different estimator (X1 with 512 fresh origins and receiver deposition). H2 still samples 16 points per sender.
- Its spatial sampling (L2) is unchanged, and the CK2 307× gradient over 2 feeding patches remains under-sampled.

**S4. Candidate pairs from m = 4 visibility rays are a sampled selector.**
- A pair missed by sampling cannot be recovered (AGENTS selector rule).
- A conservative rule (for example a bounding-volume or exact-occlusion test) is needed, and its cost must be counted.

**S5. Assigning the receiver incident bin by centroid** is an uncontrolled approximation for glossy receivers (floor B 256 patches, S2, S1 for C_ss).

### Genuinely supported

- Lambert senders reduce to exact point-to-polygon form factors [FACT].
- The ALP-oracle (dense exact-ER quadrature with V̄) is a well-defined learning-free diagnostic. It separates lobe-fit error from visibility approximation.

### Changes needed

1. Re-scope as a variance-reduction estimator option inside a comparator, not an architecture.
2. Replace V̄ with lobe-aware visibility (rays importance-sampled from the lobe inside the polygon) or restrict to unoccluded pairs. Then report the visibility-changed region C separately.
3. Restate the fit kill in the polygon-flux norm, at the CK2/E2 incident distribution, with a 2-lobe capacity control.
4. Bound pair growth with a measured N_g × N_vis on more than one scene.

### Verdict H2

**NOT_VIABLE**:
- excluded as the method by canon;
- the scalar-visibility approximation fails exactly in the gated sharp-visibility-change region;
- the quadratic explicit table is the named cost-shifting risk;
- its key threshold is an unmatched-norm transplant.

---

## 3. The proposed smoke (§6): validity problems

**1. One factor per step does not hold.**
- O2x → O3 changes several things at once: anchoring (receiver → sender), origin count (512 → 16), incident deferral (L1/L12), outgoing deferral (L4) and receiver-cos removal (L6).
- O3-L4 with a = 1 isolates outgoing deferral only, with incident still continuous.
- Missing arms:
  - O2s: sender-anchored with binned qtable, i.e. C3UR on this state;
  - O3-incbin: incident binned via f̄, outgoing continuous. auditA C3 (incident-side projection at grazing) is open on CK2 and needs this arm;
  - fresh-origin C3URX1: the control for shared origins.

**2. The learned-representation bound arms are missing (X3).**
- Add O1-NMF16, O1-L1NMF16 and O1-rank32, which substitute the auditB bound operators at S1 in the identical flight.
- Score delivery (box A, panel face, footprint) and E_H.
- This is the only arm that speaks to an ECRO-lineage learned sender, and it is CPU-ready.

**3. Statistically degenerate acceptance.**
- "Footprint correlation CI lower bound ≥ 0.8" over 3 patches, or bootstrap CIs over 9, is not estimable.
- Use per-patch signed ratio with half-split reference CIs.
- Sub-patch receiver tallies would require a reference capability that `field_src` does not provide (patch × bin only, `pt86a.py:73-200`). Do not assume it exists.

**4. Workload and implementation boundary mismatch.**
- The held-out conditions (W2 T1, LSPOT89, LMIX89) are GM1 workloads with the S1 slab.
- The smallest implementation boundary is "the 42 S1 panel senders" in the ECRO scene.
- Running both doubles the implementation.
- [FACT] There is no admitted held-out ECRO sharp workload: E6 failed SHARP_FAMILY (WL93D). So the "≥ 2/3 held-out conditions" rule rests entirely on another family. Declare that.

**5. CK2 as kill workload.**
- It is a post hoc owner-requested addendum scene: one grazing light, a 3-patch footprint, decomposition open.
- Killing on CK2 alone is fragile. E2 T1/T0 region C, with its measured ladder (ORA2 0.0127, pooled EXP 0.027, C3 0.024), should carry primary weight.

**6. G_PHYS interpretation.**
- Pooled ratio 0.953 shows no generic *level* bias. But G_PHYS is label-noise-limited (label reliability 0.525, rel L1 0.641 [MEAS 94D]), so it cannot show the absence of a floor either.
- Reword the claim "no generic floor is shown" accordingly.

**7. Economic acceptance.** Replace "≤ 2× EXP" with quality-matched edit-to-first-frame against PT plus source-only update against ECRO (X5).

**8. Numbering.**
- Per AGENTS.md:3-11, a learning-free attribution of WL94D's CK2 floor and the bound-operator delivery test is a dependent evidence-gap follow-up. That makes it a suffix batch.
- A regular number applies only if a genuinely new learned mechanism is proposed.

---

## 4. Evidence-consistency checks on the design's §1–§2 (pass / fail)

**Pass:**
- §1 contract statements match code: `model86.py:40-46, 109-124`; `ecro86.py:57-87`; the shared 86001 pattern.
- auditB numbers quoted in the matched norm: 0.96–1.04 against 1.000; 0.380 / 0.603; field 0.846–0.892 against 0.36–0.54; rank about 32 for ≤ 0.15 / ≥ 0.7.
- The EY16 0.46–0.52 Frobenius unmatched-pair warning.
- The W 8×8 defect (`an94.py:19,32`) and `delivery_box` at 1130:1238.

**Fail or imprecise:**
- (a) "has not even reached the zero-operator level": wording is wrong (S5 above).
- (b) "L3 the dominant learned defect": on CK2 the sender-substitution share is SSR 0.34 / 0.54, and s8601 misses 2× noise by 0.0005 [MEAS WL94:172-174]. "Dominant" holds on E2 (WL93D C2_src 0.018–0.024 against C0 0.040–0.077) and is borderline on CK2.
- (c) "WL93C/D exact composition, edit slope 0.995–1.000": that was with the learned G frozen. And ORA (M64 both sides) E2 T1 0.030 / 0.033 is worse than C2_src 0.018–0.024, so exact physical substitution interacts with learned partners. This is not direct evidence for an all-physical H1 chain.
- (d) The cost "68–125 ms" mixes boundaries and workloads (X5).
- (e) X4 omissions.

---

## 5. Recommended corrections to the design's verdict

- The design's **MULTIPLE_CANDIDATES_REQUIRE_DISCRIMINATION** is not supported:
  - neither hypothesis is an admissible learned candidate under `Architecture.md:13` / roadmap:128;
  - the two are not materially distinct (X2).
- The defensible conclusion from this design is **ARCHITECTURE_SELECTION_PREMATURE** for the learned global representation.

Its §6 hierarchy is the valuable part. Recast as a learning-free attribution smoke with the changes in §3, it would decide:
- (i) whether the CK2 / E2 sharp residual is sampling (L5), incident projection (L1/L12), outgoing projection (L4) or receiver/readout (L10);
- (ii) whether a non-negative rank-16 or rank-32 sender at the auditB bound is delivery- and E_H-sufficient.

Only (ii) bears directly on whether an ECRO-lineage learned sender can be justified.

**Key files:**
- `c:\Projects\ReST-GI\docs\Architecture.md` (line 13)
- `c:\Projects\ReST-GI\docs\RESEARCH_ROADMAP.md` (line 128)
- `c:\Projects\ReST-GI\experiments\gm1_relation89\rel89.py` (lines 58-63, fresh origin per ray)
- `c:\Projects\ReST-GI\experiments\gm1_c3urx1_91\cand91.py` (lines 48-110)
- `c:\Projects\ReST-GI\experiments\gm1_explicit87\xport87.py` (lines 118-125, cos-in-ring sampler)
- `c:\Projects\ReST-GI\experiments\ecro86\ecro86.py` (lines 57-67, rigid-OBB dirty test)
- `c:\Projects\ReST-GI\experiments\ecro86\model86.py` (lines 40-46)
- `c:\Projects\ReST-GI\experiments\ecro_mat94\an94.py` (lines 19, 32)
- Evidence: `C:\Users\dna10\AppData\Local\Temp\claude\462fa2b2dbb71cdd\46d65e15-e083-4057-875f-9282fb68aa5e\scratchpad\wl95\auditA.md`, `auditB.md`, `read_wl86_88.md`, `read_wl89_92.md`, `read_wl93.md`, `read_wl94.md`, `read_code.md`, `read_canon.md`
