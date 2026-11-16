# WL98 — Adversarial critic X2: learnability and evidence

**Date.** 2026-11-16. **Role.** Adversarial critic X2 of the design-only batch WL98. **Lens.** Learnability and evidence:
- Can each candidate's learned part be trained to the required accuracy with a realistic data budget? This covers coverage of rare grazing / sharp configurations, exposure bias between training inputs and runtime-delivered inputs, nonconvex lobe or sparse-support outputs, and seed variability.
- Does each plan separate H1 / H2 / H3 cleanly, with controls that really share data, budget and reference?
- Is the claimed advantage over each candidate's learning-free twin plausible, or decorative (the WL96 / WL96A pattern)?
- Is any "learned" part local BRDF compression in disguise (forbidden by the brief)?

**Documents critiqued** (all read in full): `DESIGN_REVIEW.md` (coordinator draft: primary FLS, alternative ECRO-FC), `D1_physics_measure_first.md` (XQT / LLT), `D2_learning_generalization_first.md` (LIEM-S / LIEM-B), `D3_systems_cost_first.md` (RB-DTO / LER-Δ).

**What this is.** A design critique only. Nothing was implemented, trained or rendered. No existing file was changed and nothing was committed. One numpy scratch script was run locally (§1.3). It is design input, not evidence of record.

**Labels.**
- [MEAS] a measured value, with its source.
- [AUDIT] a WL97 interpretation.
- [CODE] read from frozen code.
- [PRED] my prediction or argument (untested).
- [SCRATCH] my numpy calculation.
- [HUM] a human decision.

---

## 0. Summary

1. **The coordinator's primary candidate FLS does not survive this lens.** In the scene86 / GM1 family every element is planar or convex with pointwise frozen ER [MEAS fixture; CODE]. So the physics of FLS's learned map Λ_θ (incident field of one element → outgoing lobes of that element) is exactly ER applied to the incident deposits. Its three claimed roles each have a learning-free counterpart:
   - (1) "element-local scattering" is ER;
   - (2) "sub-element structure" is the positions of the incident deposits, which pointwise ER does not move;
   - (3) the "closure" is K-lobe mixture reduction.

   Λ_θ is therefore local BRDF compression plus a learned mixture reducer. Its delivery is explicit lobe cubature, i.e. sender-side deposition, which the twin ELT gets for free.

   The plan also puts the decisive learning-headroom test (FLS vs ELT) **after** training, in Stage 2. A learning-free Stage-1 version exists: optimal-fit lobes vs ELT at the same K and Q. The plan does not include it.

   Survives: **NO** (CONDITIONAL only if the owner answers HD-1 "yes" *and* a Stage-1 optimal-fit-vs-ELT headroom gate is added).
2. **ECRO-FC is not a candidate.** Its learned "local" factors are conditioned on current visibility (form factors `Ft`, openness `opn`), on world position / normal / tangent and on a scene-global mean token [CODE `model86.py:83-104`]. That is the brief's stop condition "dynamic visibility implicitly coupled into the representation". Its sender half re-learns ER.

   As an H1 / H2 *diagnostic*, it changes three things at once (carrier, coverage, passivity). It also keeps rank 16, which WL97 showed is an **all-incidence capacity limit** (rank 46–59 for 10 % error; coverage re-allocation is zero-sum) [AUDIT J1 / B-06].

   Survives: **NO** as a candidate; CONDITIONAL as a diagnostic arm, only with the corrections in §4.
3. **No designer's GM1 candidate has a demonstrated learning headroom over its twin.** All three independent designers argue that GM1 offers no rigid-reusable learned object (D1 Proposition I, D2 §3, D3 §2.5). Every quantitative scratch they produced points the same way:
   - D1's toy: twin / oracle error ratio 1.0–1.33;
   - D2's slab: high-frequency exit content only where internal walks are short (2.5–12 events);
   - D3's regime map: utility above 1.25 only in a 20-body regime with placeholder constants and a straw-man twin.

   I find no design argument that defeats that convergence.
4. **Survivors (all conditional on human scope / canon decisions):**
   - **LIEM-S** (D2): the best learnability story. Impulse supervision from a declared measure gives designed coverage and essentially no exposure bias, and the output family is normalised. But the high-frequency content comes from the exact interface (it is not learned). Its geometry edit (rigid motion of the medium) cannot discriminate it from its tabulated twin. The learned-K shape-generalisation prior (WL68, WL71–WL80) is strongly negative. The reference for refractive-boundary media is unqualified.
   - **RB-DTO** (D3): physically meaningful learned content, but a canon conflict (Architecture §2 assigns exposed same-asset paths to C). The shared model must infer self-visibility of unseen meshes, and its main twin is a straw man (uniform-binned GB tables, instead of the compressed precomputed-transfer twins of the PRT literature).
   - **XQT** (D1): trainable (supervised denoising with noise-free targets, no exposure bias), but predicted decorative by D1's own toy. Its Stage-1 oracle is mis-specified in the direction that inflates headroom (§3.2).
5. **Cross-cutting plan defects (§4).**
   - **(a) H1 is mis-posed for the ECRO lineage.** At fixed rank, coverage is a reallocation (J1 / B-06), so "ECRO + coverage" must be read against weighting-matched class floors and a rank axis. Coverage alone is not enough.
   - **(b) "≥ 3 seeds" is badly underpowered for the measured seed spreads.** With the OWN box-A spread, 3 seeds per arm detect only an 11.6× effect; with WL93D-sized drift, a 2.4× effect [SCRATCH §1.3].
   - **(c) Class floors and oracle fits must be fitted on one reference half and scored on the other.**
   - **(d) Grazing admission written as "S1 irradiance at cos_i ≤ 0.15" selects near-field heavy-tailed lightings, not CK2-like delivery** [AUDIT B-04].
   - **(e) DESIGN_REVIEW's WH held-out is drawn from the training generator, so it is IID, not independent.**
   - **(f) DESIGN_REVIEW §10 cites `CRITIQUE.md`, which does not exist.**
6. **Proposed verdict: NO_DEFENSIBLE_CANDIDATE** for the current scope: GM1 workloads, the current ownership canon, and the "learned transport carries information beyond an estimator" reading of the brief. The verdict would become DESIGN_READY_FOR_HUMAN_REVIEW for {LIEM-S primary, RB-DTO independent alternative} only if:
   - the owner accepts GM2-type workloads (D2 HD2) and the certified rigid self-path refinement (D3 D1); and
   - the plan defects of §4 are fixed.

   These are human decisions. This critique cannot make them. Not adoption, not experiment approval.

---

## 1. Lens, evidence anchors and scratch

### 1.1 Learnability facts the designs must respect (verified)

| # | fact | source |
|---|---|---|
| L-a | ECRO generator inputs: world centre ×0.5, world normal, tangent, log area, material, openness `1 − Σ Ft` (current visibility); message passing over the dense `Ft` (N × N current form factors); a global mean token per round. The "local" factors `c`, `a` are functions of all of these | [CODE `experiments/ecro86/model86.py:83-104`] |
| L-b | Rank-16 light-independent operator: exact on the CK2 support (allocation), but under any all-incidence weighting fitting CK2 costs elsewhere (training row L1 0.42 → 0.54; up-weighting to 51 % gives CK2 0.59; CK2-only gives 0.98 elsewhere). Eckart-Young all-incidence capacity: rank 46–59 for 10 % | [AUDIT J1 §2, §5 F0-08 ruling; B-06] |
| L-c | CK2-like lightings: 4 / 10 / 27 of 3,712 OWN lightings reach CK2 / half / tenth of the ring-0 mirror-to-box proxy. Ring-0-dominated training lightings are "mostly near-field heavy-tailed cases rather than CK2-like grazing delivery". The four CK2-lit bins carry 2.0 % of training panel irradiance. The S1 response is supervised only through receivers | [AUDIT B-04] |
| L-d | Seed-signed learned error: box-A delivery 0.52× / 1.63× (two OWN seeds); WL93D seed drift 26–32 % vs geometry drift 4–14 % | [MEAS via AUDIT J1 / J7; WL93D] |
| L-e | Exact per-ray reads: unbiased in expectation, heavy-tailed per draw. Paired X1 − A4 = −0.024 ± 0.055 at the deployed budget (no average gain). About 16× draws are needed | [AUDIT J3] |
| L-f | The CK2 glint delivers 0.46 expected relation rays per draw to the brightest box-A patches | [MEAS WL96 l.48] |
| L-g | Patch-spatial averaging at x2 read through the sharp reflection costs 0.010 rel-L2 in ROI_T; angular bin averaging there costs 0.0001 | [AUDIT J4 via D1 M3] |
| L-h | Learned K shape-generalisation history: geometry → T map fails new ports / deformation (WL68); generalisation stays local to supervision, corners fail (WL71); no improvement on new ports (WL72); new boundary sources at the oracle floor (WL73); angular basis / frame limits (WL74–WL80) | [MEAS, memory index / roadmap] |

### 1.2 Four learnability tests applied to every candidate

- **T-twin.** Name the learning-free twin with the same interface, information and budget, and say what the network can know that the twin cannot. If the answer is "nothing", the learned part is decorative before any training (the WL96A pattern).
- **T-local.** If the learned map's inputs are confined to one element (or one material), and that element's physics is pointwise ER, the learned map is ER compression, whatever wrapper it has.
- **T-cover.** Can the training measure be *designed* over the learned map's input space, so that coverage is a declared number? If coverage can only come from a scene / light sampler, the B-04 pathology (1e-3-level occurrence) returns.
- **T-expose.** Are training inputs drawn from the same process as runtime inputs? If training inputs come from path tracing or oracles while runtime inputs come from the learned / explicit pipeline itself, errors compound over sweeps.

### 1.3 Scratch [SCRATCH, design input only]

Script: `experiments/wl98_calc/X2/learnability_calc.py`. Output: `results/wl98_calc/X2/learnability_calc.json`. numpy only, seed 98200, local CPU.

**(i) Seed power.** Minimum detectable multiplicative effect between two trained arms, two-sided α 0.05, power 0.8, n training seeds per arm, log-normal seed spread:

| seed log-SD (source) | n = 2 | 3 | 5 | 8 | 12 | 20 |
|---|---|---|---|---|---|---|
| 0.81 (OWN box-A 0.52× / 1.63×, L-d) | 75× | 11.6× | 5.1× | 3.4× | 2.6× | 2.1× |
| 0.29 (WL93D seed drift, L-d, approx.) | 4.7× | 2.4× | 1.8× | 1.55× | 1.42× | 1.30× |
| 0.10 (hypothetical well-conditioned training) | 1.7× | 1.35× | 1.22× | 1.16× | 1.13× | 1.10× |

Reading:
- Every design writes "≥ 3 seeds [HUM]". At the measured spreads, 3 seeds cannot resolve the effect sizes the designs care about (tens of percent).
- Seed count is not a free human choice detached from evidence. It is computable from a pilot seed spread, and the plan must contain that pilot.
- Caveat: the 0.81 row is itself estimated from only two seeds (ln(1.63 / 0.52) / √2), and the 0.29 row is a coefficient of variation read as a log-SD. Both are order-of-magnitude anchors, not measured seed SDs on any WL98 gate metric. That is the reason a pilot is needed.

**(ii) Lobe-position basin under a downstream L2 loss.** Normalised 2D angular Gaussian lobe, width w ≈ 2α. The L2 gradient with respect to the lobe mean falls below 1 % of its maximum beyond an offset of:
- 17° at α 0.03 (w 3.4°);
- 58° at α 0.1;
- never within 90° at α 0.3.

At an 80° offset (normal → grazing specular) the gradient at α 0.03 is 4e-58 of its maximum.

Reading: a learned sharp-lobe *position* trained only through a downstream radiance loss has a basin of about ±5w. So it must be initialised or supervised near the analytic specular direction, and the analytic direction comes from ER. This is the FLS / RB-DTO dilemma of §2.3 and §3.4.

**(iii) Coverage arithmetic.** At the natural rate 4 / 3,712 (L-c), 1,000 CK2-like training lightings need 928,000 natural lightings, 250× the OWN population.
- Stratification fixes this cheaply only for maps whose input space is *local* (an element, an impulse).
- For scene-level maps (ECRO lineage, FLS's downstream closure loss, LLT, LER-Δ) stratification means generating scenes, and at fixed capacity the extra grazing mass is taken from elsewhere (L-b).

---

## 2. DESIGN_REVIEW.md (coordinator draft)

### 2.1 Candidate A — continuous directional learned kernel (REJECTED by the draft)

I agree with the rejection, and the reasons are stronger than stated:
- The nonlocal form K_θ(x, ω_o; x', ω_i) must learn occlusion between x and x', which is the visibility stop condition (WL52).
- The local decoder form is T-local: in GM1 it is ER compression.
- T-cover: a scene-level nonlocal kernel cannot be impulse-supervised (O(N × directions) columns per geometry, D2 §6.3).

Survives: **NO**.

### 2.2 Candidate B — sparse multi-resolution angular representation (kept as control)

- I agree that the *learned* version is decorative under T-twin. The refinement location is analytic (specular direction from ER and the incident direction, D3 §3.4). The coefficients are explicit flux integrals.
- Learnability objection, if anyone revives it: variable-support sparse outputs (predicting support plus values) are the hardest output type for a shared network. A wrong support is a hard (non-differentiable) miss, and its failure mode at grazing (a support cell missing the horizon-piled lobe) is exactly the rare-configuration case.
- Its learning-free version (FC 4×16, FC 8×32, adaptive) is the correct H2 carrier control.

Survives: **NO** as learned; **YES** as learning-free control.

### 2.3 Candidate C — FLS, flux-allocated spatio-angular lobe state (PRIMARY in the draft)

**Fatal objections.**

1. **BRDF compression in disguise (T-local).** The invariant T-v forbids Λ_θ any information outside its element. In GM1 the element's transport is pointwise ER on a planar or convex patch [MEAS fixture]. So the true map is fixed: incident deposits (position, direction, flux) → outgoing field L(u, ω) = Σ_j f_ER(ω_j, ω) E_j δ(u − u_j). There is no lateral transport, and no information other than ER and the deposits.
   - Role 1, "element-local scattering of distributed, partially occluded incidence": ER evaluated at the deposits. Partial occlusion is already encoded in which deposits arrived.
   - Role 2, "sub-element spatial–angular structure / emission anchor": the deposits' own positions. The X1c → X1 gain (0.27 → 0.11) is obtained by *emitting at the exact point*, which needs no learning [MEAS WL96 §A; C2].
   - Role 3, "closure (reduction to K lobes)": Gaussian / vMF mixture reduction, a classical learning-free problem (moment-preserving merging, greedy KL / ISE reduction).

   What remains learnable is a better *reduction criterion*, "minimise downstream delivery error". That is a learned compression objective, not light transport. The draft's defence ("ER is not re-learned as a function; the map is from distributed incident transport states") is a change of wrapper. The learned function is still determined entirely by ER and its inputs.

   The brief forbids reducing the learned component to local BRDF compression. HD-1 asks the owner, but the honest answer under the brief's own wording is that FLS's Λ_θ *is* that reduction plus a compressor.
2. **The learned part does not substantially replace explicit transport.** FLS delivery is explicit lobe cubature: K·Q rays per element per sweep, traced in current geometry (§2 "Delivery"). That is sender-side deposition, the same family as C3 / C3UR / photons, which D3 §2.4 notes is already cheap and removes about 90 % of the receiver-placement deficit [MEAS WL87 / WL89]. Every transport *path* in FLS is traced explicitly. The only learned element is per-element lobe fitting. This meets the brief's unsuitability criterion "the learned operator does not substantially replace or reuse explicit transport".

**Serious objections.**

3. **Learning headroom is not gated before training.** The decisive comparison is H-L (FLS vs ELT). It sits in Stage 2, *after* training, which repeats the WL96 / WL96A sequence: train, then discover the twin is as good. A learning-free headroom gate is available in Stage 1. Stage 1 already computes "optimal-fit lobes K ∈ {1, 2, 4, 8} fitted to the exact outgoing field" and ELT. The gap (ELT − optimal-fit) bounds what any Λ_θ can gain over ELT at the same K and Q.
   - The plan should STOP FLS in Stage 1 if that gap is within paired noise on the admitted workloads.
   - The Stage-1 stop as written instead compares optimal-fit lobes with FC 8×32. That tests the carrier (H2), not learning (H-L).
4. **ELT is under-specified.** "Moment-matched lobes from exact ER, same K, Q, C" defines the twin for a *single* incident component. For a multi-component incident set it needs a declared reduction rule (e.g. greedy moment-preserving merge to K). Without one, the twin is whatever the implementer writes after seeing FLS, which is a confound. The twin family must also include **no reduction** (all components × ER lobes, K unbounded). That bounds what closure loses at all.
5. **Lobe learnability (nonconvex).** [SCRATCH §1.3 (ii)] A sharp lobe mean trained through a downstream radiance loss has a basin of about ±17° at α 0.03. Grazing lobes sit about 80° from the normal. So either:
   - μ_k is initialised or supervised from the analytic ER specular direction, in which case the learned part adds only residual shaping and the ELT comparison decides it; or
   - it is learned from the downstream loss, in which case optimisation stalls on plateaus.

   D2 §5.1 adds capacity evidence: generic Gaussian / Student-t mixtures with K ≤ 4 reach only 0.35–0.6 lobe L1 on the grazing S1 lobe and overshoot peak radiance 2–7×. K = 4 t-mixtures were numerically unstable. The exact family for that lobe is ER (7 GGX + Lambert). FLS's anisotropic-vMF family therefore has a *representation* risk exactly where the hypothesis is tested (H2 inside FLS), and Λ_θ inherits it.
6. **Exposure bias is acknowledged (F7) but not designed against.** Λ_θ is "trained against path-traced delivered fields". At runtime its inputs are cubature deposits from upstream Λ_θ lobes:
   - Q nodes per lobe at fixed spherical-design positions, with flux weights;
   - one set per upstream lobe;
   - their structure differs in kind from path-traced point samples.

   This is the teacher-forcing / free-running gap, and it compounds over J sweeps. The plan needs:
   - (a) training on pipeline-delivered inputs, or a declared mixture of both;
   - (b) a measurement of Λ_θ error on path-traced vs pipeline-delivered inputs for the same configurations, per sweep.
7. **Coverage tension (T-cover).** Λ_θ's input space is local, so an impulse-like designed measure is possible. But the closure's loss ("minimise downstream delivery error") is scene-level, which brings the scene-sampler coverage problem back (§1.3 (iii)). FLS must choose:
   - local supervision: cheap coverage, but then it is ER fitting; or
   - downstream supervision: scene-level coverage, B-04 rarity, and gradient flow through traced cubature, whose gradients with respect to lobe parameters miss visibility-edge terms.

   The draft does not say which.
8. **Superposition.** The draft states the non-additivity honestly. From a learnability view it means the training distribution must cover the *set* distribution of incident components (multi-light, multi-bounce mixtures). That is combinatorially larger than single-impulse coverage, and T-s needs a [HUM] tolerance with no evidence base.

**Strengths.**
- The passive π-softmax-with-NULL allocation is a sound energy construction (it fixes the E9 failure mode by construction).
- Clipped-hemisphere normalisation addresses the measure part of E4.
- Deterministic delivery removes single-draw variance.
- Its Stage-0 / Stage-1 structure and statistics section absorb the WL97 lessons.

**Survives: NO.** It hits two brief unsuitability criteria (BRDF compression; the learned part does not replace explicit transport), and its H-L test is placed after training. The draft could keep FLS alive only by (i) the owner accepting HD-1 *and* (ii) adding the Stage-1 headroom gate of objection 3. I predict [PRED] that this gate stops FLS: in GM1, ELT with an explicit merge rule and the same K and Q should come within noise of optimal-fit lobes, because there is no non-ER information to exploit.

### 2.4 ECRO-FC (independent alternative in the draft)

**Fatal objection (as a candidate).**
- The learned per-patch factors depend on current visibility (`Ft`, `opn`), world pose and a global token [CODE L-a].
- After a rigid edit they are regenerated from the new geometry. Validity is therefore "rule reuse with per-edit regeneration", and the "local" response carries dynamic visibility, which is the brief's stop condition.
- The sender half re-learns ER (D2 L1; D3 §3.3).
- D3 §2.3 also shows that the per-edit generator costs more kernel time (4.2 ms) than everything it replaces (3.95 ms) [ARITH on MEAS].
- So ECRO-FC cannot carry H3 for geometry edits.

**Serious objections (as the H1 / H2 diagnostic the draft actually uses it for).**
1. **Three simultaneous changes.** ECRO-FC changes carrier + coverage + passivity together. Only "ECRO + coverage" isolates coverage. ECRO-FC vs ECRO + coverage then mixes carrier and passivity constraint. Passivity can cost fidelity on the fitted rows (J1: non-passive rows coexist with sub-unity total energy). One more arm is needed: ECRO + coverage + FC carrier, without the passivity constraint.
2. **Rank 16 is an all-incidence capacity limit** (L-b). At fixed rank, "coverage" re-allocates, so a CK2 gain is expected to cost elsewhere. An 8×32 carrier quadruples the bins and sharpens the per-incidence response, which raises the rank needed. Rank 16 in 8×32 is therefore likely *more* capacity-limited than in 4×16.

   Without a rank axis (e.g. rank 16 / 48 / 64 at matched coverage) every ECRO-FC outcome is ambiguous between H1 (allocation) and H2 (capacity of the operator class, not of the angular carrier). The draft's H2 is about the angular carrier and does not cover operator-class capacity.
3. **Training targets.** Unless the S1 stage is directly supervised against a physical target, the B-04 / A-06 identifiability problem (sender supervised only through receivers) persists. Coverage then cannot be separated from identifiability. D1's C1 adds a physical sender target, but the draft's ECRO + coverage arm does not.

**Strengths.**
- It is the minimal-change lineage test the brief's H1 / H2 split asks for.
- It shares the generator, budget and reference with frozen ECRO.

**Survives: NO** as a candidate; **CONDITIONAL** as a diagnostic arm (with a rank axis, a passivity-free FC arm and a physical S1 target).

### 2.5 Draft-level defects not tied to one candidate

- §10 points to `CRITIQUE.md` "verbatim" and to "the WL98 worklog". Neither exists in the directory. The draft therefore claims an independent critique that the reader cannot inspect.
- The table "same data / budget / reference as FLS?" marks ECRO + coverage and ECRO-FC "yes". FLS is trained on path-traced *delivered fields per element*, ECRO on *tables*. The geometry / light population can be shared; the data format and supervision cannot. D1 §6.4 states this distinction correctly; the draft does not.
- WG admission example "≥ 50 % of S1 irradiance at cos_i ≤ 0.15" is a criterion on *sender irradiance*. B-04 shows that ring-0-dominated lightings are mostly near-field heavy-tailed, not CK2-like delivery. Admission must be on the delivered sharp share at receivers (a deterministic exact-vs-binned delivery difference, as D1 W-G2 does).
- WH is "sealed configurations from a seeded sampler, seeds disjoint": the same generator as training, i.e. an IID test. The brief asks for an *independent* held-out workload. This needs a distribution shift: another geometry family (D1's V3 partition, D3's unseen body class) or a declared shift axis.

---

## 3. Designer reports

### 3.1 D1 — overall

D1's Proposition I is the most important learnability statement in the batch. In GM1, a learned component can only:
- (a) estimate an integral with fewer queries;
- (b) amortise repeated application;
- (c) infer unqueried visibility.

I accept it. It is the formal reason FLS fails T-local and ECRO-FC fails T-twin.

D1's E0 code finding (world coordinates and the global token, `model86.py:95, 99`) is correct [CODE, verified]. So are its rejections A1, A2, B1, C1 and O1. I add the following:
- A2 fails T-cover (scene-level).
- B1 fails T-twin with a measured null (WL96A).
- C1 is the same objection as FLS objection 1.
- O1 is per-asset training (Native Reuse class).

### 3.2 D1 — XQT, exact-sample quadrature transport (PRIMARY)

**Fatal objections.** None from this lens taken alone. Whether a learned *estimator* counts as the project's learned transport is HD-1, and the brief's criterion "does not substantially replace explicit transport" sits very close: every bit of transport information comes from explicit rays and exact ER, and W only reweights.

**Serious objections.**
1. **Predicted decorative by the author's own scratch.** In the flatland toy, flux-conserving pooling gains 8–50× in ray-equivalent terms, but the learning-free twin captures almost all of it: oracle / twin error is 1.0–1.33 [SCRATCH D1 §9]. The remaining hope is anisotropic / edge-aware kernels.
2. **The S1-c oracle is mis-specified in the direction that inflates headroom.** XQT's W does not take s_r: it is field-independent, which is what keeps it linear. The D1 S1-c oracle chooses per-source-cell kernels *with the exact field*, so it can know where the glint lands. W cannot, because the glint location moves with the light while W is fixed per geometry.

   The correct optimistic bound for any linear XQT is a per-geometry kernel optimised jointly over the declared source distribution (all training lights for that geometry). The field-aware oracle is an upper bound for a *nonlinear* estimator, which XQT is not. With the corrected oracle the oracle / twin margin can only shrink. The stop rule m_req must be applied to the corrected oracle.
3. **Spatial pooling vs the J4 readout floor.** Pooling quanta across neighbouring receivers *is* patch-spatial smoothing. J4 measured that this costs 0.010 rel-L2 in ROI_T for receivers seen through the sharp reflection, about 100× the angular-averaging cost there (L-g). D1's own rule M3 says such receivers need sub-element structure.

   So XQT's variance gain and its readout bias pull in opposite directions exactly in the sharp ROI. The plan should report the bias part of the pooled error at sharp-seen receivers separately.
4. **Information floor.** With 0.46 expected rays per draw into the CK2 glint (L-f), a 9-neighbour pool holds about 4 hits. Pooling cannot create information. The heavy tail (kurtosis 7–27, C2) shrinks but remains. "Fewer relation rays for equal error" is therefore bounded by hit statistics, not by the kernel. This is a point in the twin's favour too, but it caps η_oracle in the sharp ROI.
5. **Regeneration cost.** About 6 M MLP evaluations per rebuild, and dirty receptive fields after an edit. This is the ECRO generator pattern (per-edit regeneration of learned state), and D3 measured that pattern as no cheaper than the explicit work it replaces at N 1,270. The XQT-specific number is unmeasured.

**Strengths.**
- The best-posed *learning problem* in the batch: supervised regression to noise-free deterministic targets.
- Training inputs are the same relation draws that runtime uses, so there is **no exposure bias** (T-expose passes).
- The input space is local and designable (T-cover passes).
- Linear, flux-exact and passive by construction.
- The plan has learning-free stop rules before training.

**Survives: CONDITIONAL** on HD-1 and on a corrected S1-c oracle. Predicted [PRED] to stop at S1-c.

### 3.3 D1 — LLT, learned lobe transfer from sparse probes (ALTERNATIVE)

**Fatal objection.** LLT is logically trapped between two outcomes.
- If the S1-d information test passes (the probes determine delivery), the learning-free twin from the same probes has the same information. For planar receivers, the footprint of a known ER lobe (exact moments from ER, as LLT's own input) is analytic.
- Anything LLT adds beyond that is footprint shape on curved receivers or **inferred visibility between probes**. Inferred visibility is learned C for a dynamic quantity: the brief's stop condition, and WL52's measured failure at transitions.

So LLT is either decorative or inside a stop condition.

**Serious objections.**
- Learnability: the output is a softmax over a variable candidate set plus a mixture density per candidate.
- The informative training cases (an occluder edge inside the footprint, between probes) are a thin set in random-geometry generators, which is the B-04 rarity pattern again.
- Like ECRO-COV, the per-light cost does not amortise.

**Strengths.** The S1-d collision-pair information test is a genuinely good, learning-free kill gate. It should be reused for any candidate that infers anything (including RB-DTO's implicit self-visibility).

**Survives: NO.**

### 3.4 D1 — rejected candidates and control E0

- **A1** local continuous kernel: NO (T-local; ER duplicate). Agree.
- **A2** nonlocal continuous kernel: NO (owns visibility, or is a non-measure-exact XQT). Agree.
- **B1** learned resolvent on the B carrier: NO (measured null ceiling WL96A, T-twin). Agree.
- **C1** sender lobes: NO (T-local). Agree. This is the same objection as FLS objection 1, and D1 reached it independently.
- **E0** frozen ECRO: control only. Agree.
- **O1** asset-level reflectance-field K: NO under current canon. Agree. It is also per-asset training (Native Reuse class).

### 3.5 D1 — plan defects

- **The H1 reading uses the wrong floor.** §6.6 "H1 supported" compares C1 with the "R0 floor of the 64-bin solid-angle representation". R0 is a *per-field direct projection*. ECRO is a *light-independent rank-16 operator*, whose floor is far above the per-field projection (L-b). C1 can never reach R0, so H1 would be read as "optimisation failure" when the cause is operator-class capacity.

  D1 lacks the delivery-optimal operator-class floor (J1 §8, "never computed"). DESIGN_REVIEW and D2 G1 include it.
- **C1 changes the convention.** C1 adds a "physical sender target projected flux-exactly to the same 64 bins", while the table remains in the SA convention. C1 is therefore not "same representation" (it is a partial H2 change). C1f separates this only if C1 uses a target in the SA convention.
- **The S1-c oracle is field-aware for a field-independent W** (§3.2 objection 2).
- **Seeds:** "≥ 3" (§1.3 (i)). The plan should include a pilot seed-spread stage and compute n from it.
- **W-H via a "never-used partition of the WL89 V3 family" is good for independence.** But V3 was inspected during WL89–WL91, so "never inspected" must hold for the partition, not just the family.

### 3.6 D2 — LIEM-S (PRIMARY)

**Fatal objections.** None from this lens alone. The scope and canon objections are human decisions (HD2, HD3, HD7, HD10).

**Serious objections.**
1. **The high-frequency content is not learned.** The sharp exit lobe comes from the exact rough-dielectric closure. The learned internal-direction density is, by D2's own slab scratch, angularly smooth wherever internal walks are long (27–50 events). It has structure only for short walks (2.5–12 events), where Twin-W, the explicit walk, is cheap.

   So in admitted configurations the *learned* part either is smooth (Twin-D or a smooth table suffices) or saves little over Twin-W. That is the brief's central hypothesis ("spatially distributed, high-frequency light transport learned…") tested on the part of the transport that is not high-frequency. The I0.2 admission gate is the right kill gate. I predict [PRED] it is hard to pass with rough interfaces, which broaden exits further than the smooth-face toy.
2. **The geometry edit does not exercise the learned part.**
   - I3(i), a rigid move of the medium object: *every* rigid-invariant K (Twin-T, a per-asset table) is reused exactly. The test cannot discriminate LIEM from its twins on edit utility, only on memory and precompute.
   - I3(ii), an occluder move: K is untouched for every arm.

   So H3 for LIEM reduces to "memory / no per-asset precompute vs Twin-T, and time-to-quality vs Twin-W". These are static properties. The edit is not where the hypothesis bites. The brief asks for geometry-edit utility. This should be stated as the scope of what Track I can show.
3. **Shape generalisation prior (L-h).** Held-out shape classes are exactly where the WL68–WL80 learned-K lineage failed (new ports, corners, deformation, frames).
   - LIEM-S changes the output family (normalised densities, analytic anchors) and the supervision (impulses).
   - But D2 does not argue why its intrinsic features (thickness, edge distances, curvature) suffice where earlier geometry → operator maps did not.
   - A per-shape oracle (I1) passing while the shared model (I2) fails would repeat WL68 / WL71.

   The plan has the right arms (I1 per-query oracle vs I2 shared). It lacks a **per-shape fitted LIEM** arm (D3's B5 analogue). Without one, a held-out failure cannot be split into "shared-model generalisation" vs "family capacity".
4. **Ballistic split and sampling.** LIEM-S is a sampling-based density. In FNTO's deterministic coupled solve, applying K means fixed-sample MC per entry cell (D2 §6.1 Runtime). That reintroduces per-solve variance (the L-e problem) unless enough samples are spent, which eats the Twin-W cost lever. Twin-W must be compared at equal *variance*, not equal sample count.
5. **Reference.** WL53 qualified only index-matched media. Mitsuba float32 skips about 89 µm at interfaces, and Cycles' BSSRDF is path-history dependent (WL51). I0.1 is a real stop risk, which D2 states.
6. **Prior art.** Vicini et al. 2019 (learned shape-adaptive subsurface scattering) is the same mechanism class: a shape-feature-conditioned learned exit measure, trained on random walks. D2 marks this unverified. I note only that it makes LIEM-S's *learnability* plausible (it was learned there) while making novelty hard.

**Strengths.**
- The best T-cover design in the batch: the impulse measure is declared with strata, and coverage is a treatment (μ_D vs μ_S).
- T-expose passes (an impulse query is the runtime query format).
- Normalised output family with energy by construction.
- Exact rigid invariance.
- Three named twins and learning-free kill gates (I0.2, I1) before training.
- Physically meaningful learned content (nonlocal intrinsic).

**Survives: CONDITIONAL** on HD2 / HD3 / HD7 and on I0.2 / I0.1. The per-shape-fit arm should be added.

### 3.7 D2 — LIEM-B (ALTERNATIVE, representation-level)

- **Serious objections.**
  - It does not use the exact interface closure, so interface sharpness (α 0.04) must be resolved by cells: about α⁻² directions.
  - A large categorical output space learned from sparse walk events is an H1 risk, which D2 itself notes.
  - Tabulated per asset, it becomes Twin-T, so learning buys only cross-asset generalisation, the same L-h risk.
- **Strengths.**
  - Multinomial NLL on unbiased counts.
  - Deterministic and linear, so it fits the F / S solve.
- **Not an independent alternative.** It shares ownership, data and failure modes with LIEM-S, as D2 says.

**Survives: CONDITIONAL**, as an H2 control only.

### 3.8 D2 — rejected R-A … R-E

- **R-A** continuous ECRO kernel: NO. Agree. Its ceiling at measured learned cost is ≤ 1.12 (D2 §4).
- **R-B** learned sparse table: NO as learned. Agree.
- **R-C** learned patch lobe: NO. Agree, and D2 §5.1 is the capacity evidence I cite against FLS.
- **R-D** rigid-cluster self-transport: D2 rejects it on canon *and* weak learned value ("twin = cached explicit relations, equally rigid-invariant; learning buys only sweep amortisation"). This directly contradicts D3's primary, RB-DTO. The batch has two designers in disagreement. My view is in §3.10: D2's twin argument is the stronger one *unless* the explicit intra-body work is iterated sharp transport at large J_int, which neither has measured.
- **R-E** learned sharp delivery: NO. Agree (visibility).

### 3.9 D2 — plan defects

- **G2 "H1 supported if ECRO-COV closes ≥ X % of ECRO-R's gap to the class floor … without a non-grazing regression" is the right form.** It must use the class floor at the *same incidence weighting* as ECRO-COV. At fixed rank, the floor itself moves with the weighting (L-b). Otherwise an allocation shift is read as learning.
- **"≥ 3 seeds"** (§1.3 (i)). ECRO-R ("unchanged recipe, fresh seeds") is the right arm for estimating seed spread. Use it as the pilot that sets n before ECRO-COV runs.
- **Track I edits cannot discriminate LIEM from Twin-T on edit utility** (§3.6 objection 2).
- **No per-shape fitted LIEM arm** (§3.6 objection 3).
- **The G0 stop rule is good** (no role ≥ U_min at matched quality → no ECRO-lineage training for architecture purposes).

### 3.10 D3 — RB-DTO, rigid-body directional transfer operator (PRIMARY)

**Fatal objections.** None from this lens alone. The canon conflict (D3 D1) is a human decision.

**Serious objections.**
1. **The shared compiler must learn static self-visibility of unseen meshes.** D3 allows self-visibility to be "cached once per body (or implicit in the learned N_b)".
   - If it is implicit, the shared model infers, from mesh tokens with sparse attention, which exits a multi-bounce glossy chain reaches. That is learned visibility (static, but learned), with WL52's transition failure and WL68 / WL71's "new ports / corners fail" as priors.
   - If it is cached explicitly, the learning-free twin gets the same cache.

   D3 lists blurred self-shadow transitions as a failure mode. It should be a Stage-1 kill gate with an S1-d-style collision-pair test.
2. **Discontinuous target map.** For low-α glossy cavities the exit measure as a function of (x_i, ω_i) changes mode discontinuously where the chain's bounce count changes (V-groove 1 ↔ 2 ↔ 3 bounces). A mixture-density network blurs such boundaries (the WL52 locality failure in another place). §1.3 (ii) adds that lobe *positions* far from initialisation are not learnable through a downstream loss. The sharp channel must therefore be supervised per exit sample (likelihood), which D3 does (good). But the boundaries remain.
3. **The twin is a straw man.**
   - Twin 2 is "explicit precomputed sparse *binned* per-body transfer" at uniform α⁻² resolution (0.17–13.7 GB).
   - The relevant learning-free twin is *compressed* precomputed transfer: all-frequency wavelet PRT (Ng 2003), radiance transfer fields for rigidly moving objects (Pan 2007), shadow fields (Zhou 2005), as D3 itself lists under D9.
   - Twin 1 should use sender-side chain tracing (which D3 §2.6 item 7 says beats any learned map for α ≲ 0.02), not uniform-binned K_int.

   Against those twins, the memory and continuity advantages are not established. Without them the claimed advantage is decorative by construction of the comparison.
4. **The regime map's numbers are placeholders.** t_q 5–20 ns, uniform-binned K_int, which D3 admits inflates the lever. U ≥ 1.25 appears only at 2 × 200 / K 4,096 / J 8 (≈ 1.3) and above. Stage 0 is the right gate. The design rests on a regime (20 glossy concave bodies, J_int ≈ 8–16) that no current workload contains and whose admission values are all [HUM].
5. **Per-asset vs shared.** Per-asset fits (B5) are the realistic path to accuracy, but put the method in the external Native Reuse class (D4). B5 is correctly a control. The plan then needs B4 ≈ B5 on held-out bodies, which runs into the L-h prior.
6. **Nonlinear reduction.** The per-patch output cache is "mixture-reduced (≤ M_p lobes)". Like FLS, that reduction depends on the incident field, so superposition is lost, and the reduction is a closure with its own learning-free twin (a merge rule). This is not discussed.
7. **Energy budget.** B uses E_esc(x_i, ω_i), "computed exactly from the cached self-visibility". Per continuous incident sample this is an ER-lobe × self-visibility integral, i.e. one explicit bounce per sample at runtime, or a 4D precomputed table. Either way it is explicit work the cost model omits.

**Strengths.**
- The learned content is genuinely nonlocal and not BRDF compression (displaced exits; the convex-body N_b ≡ 0 contract test is a good T-local check).
- Supervision per exit sample with path tags avoids sum-only identifiability (the WL92 lesson).
- T-cover is designed (grazing strata, procedural bodies).
- T-expose passes (impulse format).
- Cost-anatomy Stage 0 before training, a B2 / B5 / B3 / B4 attribution ladder, and a certificate-violation control.

**Survives: CONDITIONAL** on D3 D1 (canon), on adding compressed-PRT and chain-tracing twins, and on a Stage-1 transition / visibility-inference gate.

### 3.11 D3 — LER-Δ (ALTERNATIVE)

**Fatal objections.**
1. **Per-environment learned state is per-scene training.** Changing the environment is retraining. It is not "one shared model", and it puts the method in the external Native Reuse class (D3 D4 admits this).
2. **The Δ-iteration applies H_E to signed, out-of-span fields** (−B carries negative flux). That is the measured FNTO failure mode: WL41 / WL44, learned operators wrong or non-passive outside their training span. The simplex budget guarantees passivity only per *non-negative* incident sample. For signed inputs there is no fidelity or convergence guarantee, and clamping is forbidden.

**Serious objections.**
- Its twin is not the infeasible precomputed dense resolvent D3 names. It is explicit re-sweeping (J × N·K), which costs 0.53 ms per order at N 1,270 [MEAS E5]. LER-Δ is therefore decorative wherever sweeps are cheap, which includes every workload in scope.
- Learnability of an all-order sharp environment resolvent: this is the ECRO global-term / WL96A role, with a measured null headroom on CK2.

**Strengths.** The edit enters exactly and explicitly through (M − B), and the path-tag accounting is explicit.

**Survives: NO.**

### 3.12 D3 — rejected roles

- A (ECRO-lineage continuous kernel): NO. Agree. D3 correctly identifies visibility-coupled "local" factors [CODE L-a].
- B (learned sparse table): NO. Agree.
- C at patch level: NO. Agree (T-local).
- Online NRC: NO. Agree (retrains).
- Learned inter-body C: NO. Agree.
- Learned sampling / relation scoring: NO as a transport representation. Agree, but see the missed alternative in §5.
- Resolvent at larger J: NO in scope. Agree.

### 3.13 D3 — plan defects

- **Track B Stage 1 uses a B2 per-body oracle floor, but no learning-free twin floor at matched memory with compression** (§3.10 objection 3).
- **The Stage 2 stop level is [HUM].** It needs a seed-spread pilot (§1.3 (i)).
- **The Track G H1 test (G1 coverage retraining) has no rank axis and no weighting-matched floor** (L-b). D3 also gives it no H3 role, which is consistent.
- **Track B "W-held: a body class never generated in training" is a genuine shift.** This is good, and better than DESIGN_REVIEW's WH.

---

## 4. Cross-cutting defects in the falsification plans

| # | defect | affects | fix |
|---|---|---|---|
| X-1 | **H1 vs H2 dichotomy is ill-posed for a fixed-rank light-independent operator.** Coverage re-allocates a capacity that is too small for all incidences (L-b), so "coverage fixes CK2" and "capacity limits CK2" are both true at different weightings | DR, D1, D2, D3 Track G | three-way attribution: (i) learned arm vs **class floor at the same incidence weighting** = optimisation (H1-opt); (ii) class floor vs weighting = allocation (H1-cov); (iii) class floor under *any* weighting meeting all admitted workloads, with a **rank axis** (16 / 48 / 64) = operator capacity (H2-op). Report the angular-carrier floor (H2-carrier) separately |
| X-2 | **Floors and oracle fits scored on their own fitting data.** A delivery-optimal operator or optimal-fit lobes fitted to a noisy path-traced reference is optimistic by the noise it absorbs | DR Stage 1, D1 R0 / S1-c, D2 G1 / I1, D3 B2 | fit on reference half A, score on half B (or on deterministic quadrature validated against PT) |
| X-3 | **Seed count is not a free [HUM] number.** 3 seeds detect ≥ 2.4× (WL93D-sized spread) or ≥ 11.6× (OWN box-A spread) differences | all | a pilot stage with ≥ 5 seeds of the cheapest learned arm (ECRO-R, D2) to estimate seed SD on the gate metric; then compute n for the declared minimum effect [HUM]; treat seed as a random effect when combining with pixel bootstrap SE |
| X-4 | **Learning headroom gated after training.** The learned-vs-twin comparison is put after training even where a learning-free oracle bound exists (DR FLS vs ELT; D3's RB-DTO compressed twin absent) | DR, D3 | Stage-1 headroom gate: (oracle of the learned family) vs (twin), same information, same budget; stop if within paired noise |
| X-5 | **Oracles with more information than the learned model.** XQT's S1-c oracle sees the field and W does not | D1 | restrict every oracle to the learned model's input information (and linearity) |
| X-6 | **Exposure bias untested.** FLS (path-traced vs cubature inputs) and LIEM-S in a deterministic solve (fixed samples) | DR, D2 | an arm or metric comparing the learned map's error on reference inputs vs pipeline inputs per sweep |
| X-7 | **Grazing admission on sender irradiance** selects near-field heavy-tailed cases (B-04) | DR | admit on delivered sharp share at receivers (exact vs binned delivery difference, D1 W-G2), both glint-core and tail receivers |
| X-8 | **IID "held-out".** DR's WH is drawn from the training sampler | DR | a declared distribution-shift axis (other family, body class, light class), radiance-uninspected |
| X-9 | **Edit tests that cannot discriminate the learned part.** Rigid moves of a rigid-invariant K reuse every arm's K exactly | D2 Track I, D3 Track B (partly) | state that edit utility then reduces to the C update + solve, which is shared; put the learned claim where it is (static memory / time-to-quality) and do not label it geometry-edit utility |
| X-10 | **Missing per-instance fitted arms.** These separate shared-model generalisation from family capacity | D2 (absent), D3 (B5, present), DR (absent: a per-geometry fitted FLS) | add one per candidate |
| X-11 | **CK2 role.** All designs keep CK2 diagnostic. Good. But DR Stage 1's STOP rule names "on WG", and WG includes CK2. Make sure CK2 is excluded from gate arithmetic, not only from approval | DR | define the gate over WG \ {CK2} |
| X-12 | **M64.** No design uses M64 as truth. Good. DR's ECRO + coverage arm, however, has no stated S1 target, and the WL92A OWN lineage used M64-derived component targets. State a physical target explicitly | DR | path-traced or validated-quadrature S1 target |
| X-13 | **Dangling provenance.** DR §10 cites `CRITIQUE.md` and a WL98 worklog, which do not exist | DR | fix before human review |

---

## 5. Missed alternatives

1. **A learned directional *guiding* distribution reused across rigid edits.** A per-element directional lobe distribution in local frames (path-guiding class: Vorba 2014, Müller 2017 / 2019 [LIT, not checked]) used only as the *sampling* density for explicit sender-side or receiver-side relation rays.
   - Errors cost variance, never bias, so it has a trivially defined energy contract.
   - It attacks the measured residual directly (L-e: the exact route is unbiased but heavy-tailed).
   - It is rigid-equivariant in local frames.
   - Weakness: it is a sampler, not a transport representation. WL46 treated sampling "only as a separate conditional variable". Admissibility is [HUM], like XQT's HD-1. It is closer to the measured problem than FLS and needs no closure.
2. **A light-conditioned or higher-rank ECRO operator** as the H2-operator-class arm. Light-conditioning removes the "light-independent all-incidence" capacity limit of L-b. It is the missing control that separates operator-class capacity from angular carrier capacity.
3. **A learning-free mixture-reduction twin** (moment-preserving greedy merge) for every candidate with a K-lobe closure (FLS, RB-DTO's per-patch cache).
4. **Compressed precomputed transfer twins** (wavelet / clustered PRT, radiance transfer fields) for RB-DTO and LIEM-S's Twin-T.
5. **A learned sub-element incident-field super-resolution at receivers seen through sharp reflections**, targeting the measured 0.010 J4 readout floor. Like XQT it is an estimator, but it targets a measured *bias* rather than variance. Brief admissibility is [HUM].

---

## 6. Verdict and rationale

**Proposed verdict: NO_DEFENSIBLE_CANDIDATE** for the stated scope. A verdict is not adoption and not experiment approval.

- **Scope.** GM1 / scene86 workloads, the current K / C ownership canon, and the brief's reading that the learned component must carry transport, not only compress ER or reweight explicit samples.
- **Within that scope.**
  - FLS fails as BRDF compression with explicit delivery.
  - ECRO-FC fails on visibility-coupled factors.
  - LLT is either decorative or infers visibility.
  - LER-Δ is per-scene and fails on out-of-span signed fields.
  - XQT is an estimator whose admissibility is HD-1, and it is predicted decorative.
- **Outside that scope.** LIEM-S (GM2 media) and RB-DTO (certified rigid self-paths) survive conditionally. Their admissibility is a human scope / canon decision (D2 HD2 / HD3 / HD7; D3 D1 / D4). Neither has evidence of learning headroom over a fair twin: LIEM-S's edit does not exercise its learned part, and RB-DTO's twin is a straw man.
- **If the owner accepts those scope / canon decisions,** I would rank the candidates:
  - **LIEM-S primary.** It has the best learnability construction (designed coverage, no exposure bias, normalised output, exact rigid invariance).
  - **RB-DTO independent alternative.** Its learned content and failure modes are independent of LIEM-S.

  The batch verdict would then move to DESIGN_READY_FOR_HUMAN_REVIEW, conditional on the plan fixes X-1 … X-13, with the explicit prediction [PRED] that LIEM-S's I0.2 headroom gate and RB-DTO's Stage 0 are the likely stopping points.
- **Not softened for consensus.** All three independent designers state a scoped "no defensible learned candidate in GM1" reading (D1 §8 under HD-1 "no", D2 §3 / §9, D3 §7). The coordinator's draft answers by putting its primary in exactly the place they rule out.

---

## 7. Provenance

- **Read in full:** `docs/plans/wl98_learned_directional_transport/{DESIGN_REVIEW, D1_physics_measure_first, D2_learning_generalization_first, D3_systems_cost_first}.md`.
- **Verified:**
  - `experiments/ecro86/model86.py:83-120` (factor inputs, `Ft`, `opn`, global token, apply);
  - WL97 `show.py` entries B-04, B-02, B-06, F0-08;
  - `J1` §2 / §5 rulings (allocation vs all-incidence capacity, rank 46–59);
  - `J3` (no average gain at the deployed budget, about 16× draws);
  - `C2` (X1c / X1 0.268 / 0.111, kurtosis);
  - WL96 worklog l.48 (0.46 rays per draw), l.63, l.102.
- **Scratch:** `experiments/wl98_calc/X2/learnability_calc.py` → `results/wl98_calc/X2/learnability_calc.json` (numpy, seed 98200, CPU). Basin entries reported as "0.0" in the JSON mean "threshold not reached within 90°".
- **Not done:** no existing file was modified; nothing was trained, rendered or committed; no server or GPU was used.
