# WL95 adversarial critique: the "propagation-first" design (H-A LCFG, H-B VSA)

**Lens:** cost, scaling, reuse and falsifiability.
**Scope:** read-only. I edited no files and ran no GPU work or experiments.
**Inputs:** the eight scratchpad evidence files, plus these code and doc lines, which I re-verified:
- `experiments/gm1_relation88/rel88.py:1-25`
- `experiments/gm1_relation88/protocol/*.json:16` (peaked_rule, `m`, `kappa_u`)
- `experiments/gm1_c3urx1_91/cand91.py:1-110`
- `experiments/ecro86a/pt86a.py:73-200`
- `experiments/ecro_s94d/s94.py:145-160`
- `experiments/ecro_mat94/an94.py:19,32`
- `docs/worklogs/91_…md:256,278,285-303`
- `docs/worklogs/90_…md:225-240`
- `docs/worklogs/87_…md:266,346,349`
- `docs/Architecture.md:21,44,107`

Labels: [FACT] verified, [MEAS] measured value with its source, [INF] inference, [HYP] untested.

---

## 0. Verdicts

| Hypothesis | Verdict | One-line reason |
|---|---|---|
| **H-A (LCFG)** | **VIABLE_WITH_CHANGES**, but only as a *learning-free first-gather estimator hypothesis*, i.e. an upgrade of the explicit comparator. It is **not** a learned-architecture candidate. | Its causal premise is grounded (WL90, WL91 pooled). However, the estimator is under-specified (MIS over inconsistent x-sampling), two mechanisms are bundled, the costs are understated, it has no control isolating its one new ingredient, its acceptance thresholds are already met by existing baselines, and WL91's held-out failure conditions are omitted. |
| **H-B (VSA)** | **NOT_VIABLE** as a WL95 candidate | No measured target (WL90 P2plus ≤ 0.0009). Its exact-direction readout is hidden per-frame ray tracing. Its depth-2 chains need either Gaussianised ER lobes (which conflicts with the frozen ER) or path-space sampling. Atom growth is O(L·P^d). It is nested in H-A, not distinct from it. |
| **Report-level verdict** "MULTIPLE_CANDIDATES_REQUIRE_DISCRIMINATION" | **Not supported by the report's own content** | H-B is conditional on H-A passing (decision row D4), so nothing discriminates between candidates. The experiment discriminates H-A from the NULL. For the brief's learned-architecture question the defensible label is **ARCHITECTURE_SELECTION_PREMATURE**. **ONE_CANDIDATE_JUSTIFIED** applies only if the owner rules that a learning-free first gather is an admissible candidate (open owner decision, read_canon (D) and WL86B). |

---

## 1. Cross-cutting findings (apply to the whole report)

**X1. The Stage C requirement of "≥ 2 materially distinct hypotheses" is not met.**
- [FACT, design §3.2/§6 D3] H-B "costs the same as H-A at first order, since the estimator is shared". D3 kills both together. H-B is only admitted after D4.
- [INF] H-B is therefore H-A plus chains, not an independent alternative. The report gives one hypothesis and one conditional extension.

**X2. Neither hypothesis contains learned transport, and this is relabelled rather than resolved.**
- [FACT] In H-A, sharp first order = ER plus an MC estimator. ℛ is explicit by default, and a learned ℛ is admissible only if it beats explicit sweeps.
- [MEAS, WL87:346,349] The only measured learned advantage over explicit is the source-only update (3.5–3.9×). Fidelity: 0 wins / 3 losses out of 36.
- [INF] H-A is therefore WL86B **H-0** (explicit patch×bin on the same state; "no novelty; candidate vs baseline is an owner decision", read_canon C) with a modified first gather.
- [FACT, Architecture.md:13; RESEARCH_ROADMAP.md:128] EXP/C3UR are "not the global method to be published". The current decision frontier asks for an ECRO-lineage hypothesis that addresses the learned-flight deficit.
- The designer's own disclosure ("the learned share shrinks to ℛ… may be judged H-0 plus a better estimator") is correct. The **verdict and §5 recommendation should carry that classification**, rather than presenting H-A as the WL95 primary architecture.

**X3. The new mechanism inside H-A is small, and its nearest ancestor is omitted as a control.**
- [FACT, rel88.py:5-15] WL88 R1 SAM already is MIS (balance heuristic) between the deployed receiver-uniform state (κ_u = 8 per bin) and a sender-area proposal for every (flat receiver, peaked sender) pair:
  - x ~ uniform on the receiver;
  - y ~ stratified uniform on the peaked patch;
  - m = 4;
  - "peaked" is the same material-only Π > 8 rule (S1 37.95, S2 16.13).
- [FACT, cand91.py:3-9; gap90] X1 = exact once-scattered radiance read per landed ray. R0X1 (receiver-side) also exists.
- [INF] H3 = SAM-proposal **reshaped to a stationary-point lobe** + X1 reads + sub-patch B₂.
- **SAM-X1** (area-uniform proposal with exact reads) at matched rays is the control that isolates "lobe-aware". It is not in the §6 arm table. Without it, a positive H3 result cannot be attributed to the proposal shape.

**X4. Evidence statements that overreach.** The report labels these [INF], but they feed the recommendation.
- "The learned S1 sender holds nothing to preserve / no learned content" (§2.1, §5).
  - [MEAS, auditB] Operator L1 is at zero level (0.96–1.04). But the field L1 is 0.846–0.892, below zero's 1.0, and field energy is 0.83–1.09.
  - S_geo < 1 % says there is no *geometry* content; it does not say there is no content.
  - Correct statement: no geometry content and no measurable operator-level improvement over zero.
- "Replacing the learned panel sender by a physical one consistently helps at render level" (§2.4).
  - [MEAS] True for untextured sharp ROIs.
  - On textured CK2, "no arm is best on both seeds and both variants" (C2_src 0.031 → 0.063).
  - WL93D C2_src *worsens* the pooled single-flight table (+0.0052…+0.0079).

---

## 2. H-A (LCFG): detailed critique

### 2.1 Fatal flaws
- As an *estimator hypothesis*: none, provided §2.2 items S-1 to S-3 are fixed before the freeze.
- As an *answer to "reusable learned transport"*: fatal unless the owner admits a learning-free first gather (X2).

### 2.2 Serious issues

**S-1. The MIS estimator is not coherently specified (correctness).**
- [FACT, design §3.1] The proposals sit at 16 *fixed* receiver points x_j per patch. The uniform partner is the deployed state, whose rays have *distinct* origins uniform over the patch (`pt86.py:359-363`).
- Balance-heuristic MIS needs both strategies to sample the same domain with the same x-marginal. SAM does this correctly by drawing x ~ uniform on i for the area samples too (`rel88.py:8`). H-A's x_j are deltas, so the "MIS" combination as written estimates neither the patch average nor the sub-patch average.
- Moving to 4×4 sub-patch outputs splits the 512 uniform rays to ≈ 32 per sub-patch, i.e. κ ≈ 0.5 per incident bin. The counts per sub-patch are also random under the fixed shared pattern. The broad (non-peaked) part of B₂ at sub-patch resolution would be far noisier than R0.
- Required fix:
  - jitter x uniformly within the output cell for every strategy;
  - weight with the actual per-cell counts;
  - estimate only the "first hit is peaked" term at sub-patch resolution and keep the rest at patch resolution, as in SAM's split.

**S-2. Fixed receiver points reintroduce an L2-type defect at the receiver.**
- [INF] The design removes the 16-point fixed spatial pattern at the *sender* (L2), then places a 16-point fixed grid at the *receiver*.
- The CK2 footprint is 23 % of a face, one column of 3 of 9 patches, with hard edges (auditA §3.2). Its boundaries would be point-sampled at ¼-patch pitch, which is a deterministic bias, not noise.
- It must be randomised per realisation, or included in the matched-quantity test.

**S-3. Two mechanisms are bundled; one of them has no reference.**
- H3 bundles (i) the lobe-aware proposal + exact reads with (ii) sub-patch B₂ readout (the L10 mitigation).
- [FACT] `CKF_none.npz` e2_p exists only at patch × bin (`pt86a.py:73-200`; `s94.py:157`). The design's "per 4×4 sub-patch where a reference exists" refers to none that exists.
- Mechanism (ii) is therefore validated only through E_H renders, where it confounds S2.
- Required fix: test H3 at patch resolution first. Treat sub-patch as a separate factor, or render a sub-patch e2_p reference beforehand.

**S-4. Cost is understated or inconsistent (the hidden-cost criterion).**
- **Ray tracing is declared, not hidden.** The report is honest that 𝒜 *is* ray tracing.
- **Inconsistency in §4.1.** "Inference and update complexity: ECRO-like + ≈ 10–30 ms" conflicts with H-A's own default ℛ = explicit sweeps.
  - [MEAS, WL87:266] EXP per-edit 89–95 ms vs ECRO 69–78 ms vs PT 36 ms.
  - [MEAS, WL91] The closest measured relative, C3URX1, has edit-to-frame 124.7 ms.
  - [INF] A realistic H-A per-edit is ≈ 100–150 ms, not ECRO + 10–30 ms.
- **Light-count scaling is omitted.**
  - [MEAS, WL91:303] C3URX1 source update is 25 / 78 / 314 ms for 1 / 4 / 16 lights.
  - [MEAS, WL91:256] Exact source evaluation is 526,467 probe + 526,467 shadow rays per light.
  - H-A adds ≈ 0.65 M proposal visibility + shadow rays *per light per peaked plane*, and its proposals move with the light (y* depends on p_l).
  - [INF] "≈ 10–30 ms per source configuration" holds for L ≈ 1. At L = 4 expect tens of ms more. The 11–14 ms anchor (WL91:303) is the NEE evaluation of already-landed rays, not the tracing of new visibility rays.
- **The only measured learned advantage is given up.**
  - [MEAS, WL87:346] ECRO source-only 13–18 ms vs EXP 51–64 ms.
  - H-A recomputes 𝒜 + ℛ on every light change. In an animated-light regime H-A is strictly worse than ECRO on update cost. The design must state this, and S3 must include the source-only update.
- **Memory "worst case ≤ 20 MB" is not a worst case.**
  - It assumes one peaked plane. The culling count uses 33 planes, which is up to 33× more.
  - B₂ sub-patch tables are 4 MB *per light* if lights are kept separate, as ECRO's B[l,…] is.
  - Worst case: O(N_r · L · P · M) proposal entries.
- **The proposal cache probably buys nothing.**
  - [MEAS, WL86:214; WL86B (ii)] Selective vs full state update is 21 vs 26–28 ms ("immaterial"), and ℛ is global.
  - [INF] Selective retrace of proposal visibility bits saves at most part of a ≈ 10–30 ms term while adding dependency bookkeeping.
  - Either measure it against the same-mechanism full 𝒜 rebuild, or drop it.
- **Construction cost** (state + 𝒜 + EXP fill) is unmeasured. Dr.Jit compilation must be excluded and reported separately: WL91 showed 0.73–0.96 s compile events and 20 s cold for PT.

**S-5. Reuse holds, but reuse of a learned representation is vacuous.**
- [INF] "Reuse without retraining" holds trivially because only the frozen ER is learned. That is component-level and "not a spatially nonlocal operator" (Architecture.md:11, 42).
- The dynamic cases that matter for sharp interreflection get no reuse:
  - **Sharp mover** (peaked plane moves): full 𝒜 recompute for P plus descendants. This is WL86B gap #7, and no workload exists for it.
  - **Light move:** full 𝒜.
  - **Curved or deformed peaked senders:** fallback to the uniform proposal, i.e. **R0X1**, which [MEAS, WL90] gave **no E_H gain** (+0.003 [−0.006, +0.012]).
- [FACT, RESEARCH_CENTRIC_TOPIC.md:20; Architecture.md:66] Every decision must state its implications for rigid motion, local deformation, bending and self-contact.
- [INF] H-A's sharp-fidelity mechanism exists only for rigid planar peaked senders under compact lights. The report states the fallback but not the fidelity consequence.

**S-6. Physics risks at the regime that motivated the design.**
- [INF] CK2 panel incidence is cos θ 0.024–0.164 (auditA §3.2). There, GGX/coated lobes are off-specular, Fresnel/masking dominated and strongly elongated, so a Gaussian centred at the mirror-image point may be mis-centred. MIS keeps the estimator unbiased, but variance reduction is exactly what S2 must show. The design correctly labels this [HYP].
- "The pdf is exact in either form" is an overclaim for the VNDF-mapped variant. The half-vector → y map for a path light→y→x has no closed-form inverse or Jacobian in general (it needs a Newton solve). Only the Gaussian-on-plane form has a trivially exact pdf.
- [MEAS, WL86:150] g (REF-ER vs REF-P) is 0.059 in E2 region C. Any improvement below ≈ 0.06 there is fidelity relative to REF-ER only. Physical correctness against REF-P cannot be demonstrated in that region.

**S-7. Ownership and K/C placement.**
- [INF] Partitioning by path order (𝒜 = exactly 2 events at the receiver, ℛ = ≥ 3) sums to the reference by construction. It also maps onto K/C more cleanly than ECRO: K = ER, and C = visibility rays owned by 𝒜 or the state.
- [INF] The Architecture.md:21 "no direct/indirect branch" analysis is correct for explicit ℛ, and the learned-ℛ conflict is correctly escalated as an owner decision. This is a point in the design's favour.

### 2.3 The falsification experiment (§6): is it well defined?

**F-1. S1 is a verification check, not a premise test.**
- [FACT, `pt86a.py:73-200`] The reference e2_p samples a uniform patch point and an in-bin direction, traces back to the receiver, BSDF-samples one bounce, and fires NEE at a k = 2 vertex on material 1. This is a *receiver-side sampled estimator with an exact NEE read*.
- H2-pool (64 pooled R0 realisations + X1) is the same estimator structure for the same integral.
- [INF] If S1 passes, it says only that two unbiased estimators converge. If it fails, it signals an implementation or semantic mismatch (for example the 2 mm probe offset vs a light 5 cm from the panel, or valid-sample normalisation). It does not show "the floor is not first-gather".
- Relabel S1 as a **validity and decomposition gate** (decision D1, which genuinely resolves auditA §3's open C1–C6). Do not treat it as the H-A premise kill.

**F-2. S1's tolerance ignores H2-pool's own Monte Carlo noise.**
- [INF] With ≈ 1 hit per patch per realisation on the footprint, 64 realisations give ≈ 64 hits, so relative noise ≈ 0.12 before the sharp-read variance is added.
- That is comparable to the 0.136 band (2 × 0.058 + 0.02). H2-pool could fail S1 on its own noise.
- Required fix: the tolerance must combine the reference half-noise with H2-pool's own split-half noise, or use more realisations.

**F-3. S2's variance criterion uses the wrong statistic and too few draws.**
- E_H is an absolute-value metric that mixes bias and variance (read_wl89_92 caveat), so the SD of E_H across draws is not estimator variance.
- With n = 8, the CI on a SD ratio is roughly ±30–50 %, so a 0.5× threshold cannot be resolved reliably.
- Required fix: use the per-patch (and per-sub-patch if a reference exists) variance of the matched exitance across **≥ 16** realisations, as WL88 and WL91 did, with paired per-draw comparisons.

**F-4. The acceptance thresholds do not discriminate.**
- The acceptance requires "E2 region C ≤ 0.06 single draw" and "edit slope 0.75–1.25". Existing comparators already meet both:
  - [MEAS] EXP 0.049 (WL87); SAM R1 0.030 ± 0.008 (WL88); C3UR 0.026 (WL88);
  - EXP C slope 0.86–0.98.
- A null result would pass the acceptance.
- Required fix: make the criteria relative to the immutable comparators *and* SAM-X1 at matched total rays and matched time-to-tier. Use WL91's rule (a) form (no > 1.25× slowdown vs C3UR at a common tier), and target ORACLE-level error (0.0137 on E2 C, within noise) where g allows.

**F-5. The held-out conditions where X1 already failed are omitted.**
- [MEAS, WL91:151-156, 278, 285] C3URX1 failed held-out validity in 2 of 3 conditions:
  - W2 T1 deployed draw 0.053, 3.1× slower at the 0.06 tier;
  - LMIX89, no gain;
  - LSPOT89 C_ss harm, +0.0025 [+0.0005, +0.0045].
- The proposed workloads are CK2 (a post hoc, owner-requested qualitative addendum scene with one light, single geometry and 15 ROI patches), E2 (inspected many times) and W1 T1 ("calibration").
- [FACT, read_wl93 §5] No independent sharp held-out exists (E6 not admitted).
- Required fix: W2 T1, LMIX89 and LSPOT89 (R_mirror and C_ss) must be in the gated set. Proposal parameters (M, widths, peaked rule) must be frozen before any of them is evaluated.

**F-6. S3 is arbitrary or incomplete.**
- "Added per-edit GPU time > ECRO's 76–78 ms" would permit roughly 2× ECRO, and it is not the canon bar.
- [FACT, RESEARCH_ROADMAP.md:16; Architecture.md:72] The canon bar is quality-matched edit-to-valid-output, plus per-frame, source-update and memory costs, against the explicit comparators and PT.
- [MEAS, WL91:285-303] Explicit methods already beat PT 7.6–31× on sharp tiers in this family, so the binding economic comparison is against **C3UR, C3URX1 and SAM-X1**, not PT.
- Historical caution: [MEAS, WL67/69/70] quality-matched PT re-render beat or matched reuse designs on other workloads (Cycles 4 spp 4.7 s vs 190 s; 2.4 s vs 24 s; 2.09 vs 2.40 s, fragile). Keep PT in the ladder, but the decisive rule is against the explicit comparators.
- Required additions:
  - source-only update cost at L = 1 / 4 / 16;
  - cold construction;
  - compile time excluded and reported;
  - memory as a function of P and L.

**F-7. "Every level well defined with existing code" is overstated.**
- [FACT] gap90's X(a,s) projections, pooled R0 and R0X1, and cand91, are built on the GM1 fixtures (`fx88`/`fx91`). CK2 lives in `ecro_local93d/addendum_checker`. Porting is needed.
- The sub-patch e2_p reference does not exist.
- [FACT, quad87.qtable] q uses 16 points per patch, so H1q removes L1 and L12 but only partly L2. This matters under CK2's 307× panel irradiance range.

**F-8. The panel-face correlation is computed over 9 patches** (auditA §4.3). Do not gate on it. Use box A (54 patches) with bootstrap CIs.

### 2.4 What is genuinely supported (H-A premise)
- [MEAS, WL90] On GM1, the explicit-vs-oracle gap is first-gather angular projection: 97–102 %, isolated in 3 configurations × 4 regions. Compounding is ≤ 9 %, P2plus ≤ 0.0009, and M₁ ≈ ORACLE.
- [MEAS, WL91] Pooled exact reads are within −0.0025…+0.0035 of ORACLE in 6/6 conditions. The failure is single-draw variance (0.005–0.014 vs 0.0005–0.005).
- [MEAS, WL88] Placement and coverage dominate receiver-side deficits: SAM coverage 0.59 → 0.993; C3UR − R0D +0.050.
- [INF, plausible] A lobe-aware proposal is the natural next variable. The ≈ 1-hit Poisson argument is consistent with R0X1's null result. (It conflates per-patch rays with per-point hits; this is minor.)
- Not supported: anything at CK2 or grazing incidence. The CK2 floor decomposition is open (auditA §3).

### 2.5 Changes required for H-A to be a valid, falsifiable hypothesis
1. Reclassify it in §5 and §8 as a learning-free explicit estimator hypothesis (H-0 lineage). Do not present it as the WL95 learned architecture.
2. Specify a coherent MIS: joint (x, y) sampling, x uniform in the output cell for all strategies, actual counts, and the SAM-style "first-hit-peaked" split.
3. Unbundle the mechanisms. Primary arm: H3 at patch resolution. Sub-patch B₂ is a separate factor, gated only if a sub-patch reference is rendered.
4. Add the controls SAM-X1, R0X1 (= H2), C3URX1, C3UR, SAM and R0, all at matched total rays (relation + probe + shadow + proposal).
5. Gate on held-out conditions W2 T1, LMIX89 and LSPOT89 (R_mirror and C_ss), plus E2 T0/T1 and CK2 with one new non-grazing light. Use ≥ 16 realisations and paired per-draw statistics. Matched quantity: per-patch exitance with W_T (not `an94.W`) plus per-patch estimator variance.
6. Kill conditions (unambiguous). Any of:
   - (a) the mean signed error is outside the pooled-oracle noise;
   - (b) H3 is not better than SAM-X1 **and** C3URX1 on paired per-draw E_H in ≥ 2/3 held-out conditions;
   - (c) a > 1.25× time-to-tier slowdown vs C3UR at any common tier;
   - (d) harm beyond noise in C_ss or stationary regions.
7. Cost report:
   - source update at L = 1/4/16;
   - per-edit (sharp mover and non-peaked mover separately);
   - cold construction with compile excluded;
   - memory vs P and L;
   - the proposal cache's selective-vs-full benefit (drop the cache if immaterial).
8. State the deformation / bending / self-contact implications explicitly: fallback to R0X1, no measured gain.

---

## 3. H-B (VSA): detailed critique

### 3.1 Fatal flaws
1. **No measured target.**
   - [MEAS, WL90:172-184] P2plus is ≤ 0.0009, and isolation holds in all 4 GM1 regions, including the S1→S1 slab configuration.
   - [MEAS, WL86B:169-186] 89 % of what E2 region C reads comes from Lambert senders; glossy→glossy chains are ≤ 11 %.
   - The design admits this ("no measured target yet"). A candidate with no evidenced target fails the brief's "no recommendation merely to generate progress" rule.
2. **Hidden per-frame ray tracing.**
   - The readout "atoms evaluated at the exact direction x₂→x₁" needs, per camera sample, atom frustum/aperture tests and V(x₂, y*) visibility.
   - That is per-frame ray work proportional to visible atoms. It is exactly the LPTG failure the designer rejected in §3.3 ("per-frame retracing … moves cost into every frame") and that WL84A cites (read_canon C).
   - Caching it per sub-patch would bring back the angular projection it claims to remove.
3. **The ER contract conflicts for chains.**
   - Propagating a glossy atom through a second peaked plane with Σ_o = JΣ_iJᵀ + Σ_ER replaces GGX/coated lobes (heavy-tailed) by Gaussians. That is an approximation of ER, not the frozen ER.
   - [FACT, Architecture.md:44] "Future modifications to ER require a new, separately justified hypothesis".
   - The exact alternative, ER evaluated at both peaked vertices, is a two-specular-vertex path integral, i.e. a manifold-path sampler, which is a path tracer for the chain family.
4. **Combinatorial and state blow-up.**
   - Atom count grows as O(L·P^d). Frustum gathers are O(N_r · #atoms) without a BVH.
   - Atoms are source-dependent (they hold p′_l), so "persistent state" is a misnomer: every light change rebuilds the forest.
   - Aperture clipping and dependency trees lower debuggability, which the design admits.

### 3.2 Serious issues
- At first order H-B is H-A with more machinery, so it is not materially distinct (X1).
- Its "exact for ideal planar mirrors" [FACT] is true but irrelevant, because no workload material is an ideal mirror (S1 is a coated 7-GGX-lobe closure).
- The partition threshold is "measured, not learned", but no such measurement exists.

### 3.3 Genuinely supported
- [MEAS, WL86A post hoc] The 64-bin encoding loses narrow-beam information: identical-encoding beam pairs differ 0.94 / 0.62 against noise 0.26 / 0.22.
- That is post hoc, and it is not the point-light failure.

### 3.4 What would be required
- First, a measured, noise-exceeding residual attributable to paths with ≥ 2 peaked vertices, after an H-A-level first gather (the design's D4).
- Then: a per-frame cost budget declared at the readout, exact-ER chain evaluation or a declared ER-approximation hypothesis, a depth and flux cap, and an atom-count scaling measurement.
- Until then, retain H-B as an unadopted idea, not a candidate.

---

## 4. Answers to the lens questions

| Question | H-A | H-B |
|---|---|---|
| Ray tracing or large tables hidden inside a "learned" operator? | Ray tracing is declared openly. There is no learned operator to hide it in. Tables are small (MB scale), but the memory "worst case" is understated by up to the factor P, and B₂ by L. | Hidden per-frame atom visibility at readout. Atom tables are small but grow O(L·P^d). |
| Initial construction cost | Unmeasured. ≈ state (26–28 ms) + 𝒜 per light per plane + EXP fill [INF]. Compile excluded. | As H-A plus atom forest construction (unmeasured). |
| Per-edit update | ≈ 100–150 ms [INF; anchored on WL87 EXP 89–95 ms and WL91 C3URX1 124.7 ms], not "ECRO + 10–30 ms". A sharp mover or light move means a full 𝒜 recompute. | As H-A plus subtree rebuilds. |
| Source-only update | Loses ECRO's only measured learned advantage (WL87 13–18 vs 51–64 ms); scales linearly with L (WL91 25/78/314 ms). | Same or worse. |
| Practical vs full recomputation | Explicit methods in this family already beat PT 7.6–31× on sharp tiers (WL91). The real bar is C3UR / C3URX1 / SAM-X1 at matched time-to-tier, where C3URX1 previously failed (3.1×). | Not assessable. Per-frame cost is undeclared. |
| Geometry-edit reuse without retraining | Trivially true, since nothing is learned. Fidelity reuse exists only for rigid planar peaked senders with compact lights. Deformation and curvature fall back to R0X1 (no gain, WL90). | Same, plus fragile dependency trees. |
| Falsification well defined? | Not yet: S1 is a verification check, S2's statistic is wrong and n is too small, the acceptance is already met by baselines, the held-out failures are missing, SAM-X1 is missing, S3 is arbitrary, and the sub-patch reference is missing. Fixable (§2.5). | No: no target, so no kill condition until D4 produces one. |

---

## 5. Recommended disposition for the WL95 synthesis
- **H-A:** VIABLE_WITH_CHANGES, scoped as "learning-free lobe-aware MIS first gather with exact reads". It is a candidate for the explicit-comparator lineage or the owner's H-0 decision. It is not a learned architecture.
- **H-B:** NOT_VIABLE now.
- **Report verdict:** replace MULTIPLE_CANDIDATES_REQUIRE_DISCRIMINATION with ARCHITECTURE_SELECTION_PREMATURE for the learned-transport question. Record H-A's corrected smallest experiment as the decision input for the owner's open question "is a learning-free global solve around ER a candidate or a baseline?" (read_canon (D)).
- Batch class: no successor is started or authorised.

## 6. Sources
**Scratchpad files:** `C:\Users\dna10\AppData\Local\Temp\claude\462fa2b2dbb71cdd\46d65e15-e083-4057-875f-9282fb68aa5e\scratchpad\wl95\` → `auditA.md`, `auditB.md`, `read_code.md`, `read_canon.md`, `read_wl86_88.md`, `read_wl89_92.md`, `read_wl93.md`, `read_wl94.md`.

**Code and docs verified this pass** (all under `c:\Projects\ReST-GI`):
- `experiments\gm1_relation88\rel88.py:1-25` (SAM: x uniform on the receiver, y stratified on the peaked patch, balance heuristic, curved receivers uniform only)
- `experiments\gm1_relation88\protocol\*.json:16` (peaked_rule; m = 4; κ_u = 8)
- `experiments\gm1_c3urx1_91\cand91.py:1-110` (X1 on the sender-side C3UR relation; probe + shadow per landed ray per light; no cache)
- `experiments\ecro86a\pt86a.py:73-200` (e2_p: uniform patch point × in-bin direction, BSDF-sampled bounce, NEE at the k = 2 vertex on mid 1; patch × bin only)
- `experiments\ecro_s94d\s94.py:145-160` (CK2 floor: M64(s) sender, `m0.rw` flight, `AN.W` exitance)
- `experiments\ecro_mat94\an94.py:19,32` (NB, NPHI, NZ = 64, 8, 8 defect)
- `docs\worklogs\91_*.md:256,278,285-303`
- `docs\worklogs\90_*.md:225-240`
- `docs\worklogs\87_*.md:266,346,349`
- `docs\Architecture.md:21,44,107`