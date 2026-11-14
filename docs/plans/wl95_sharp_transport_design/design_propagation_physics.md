# WL95 adversarial critique: designer "propagation" (H-A LCFG, H-B VSA)

Lens: physics, representability and evidence consistency. Read-only. I edited no files and ran no GPU work. Inputs:
- all eight scratchpad evidence files (`C:\Users\dna10\AppData\Local\Temp\claude\462fa2b2dbb71cdd\46d65e15-e083-4057-875f-9282fb68aa5e\scratchpad\wl95\*.md`);
- code spot-checks in `c:\Projects\ReST-GI`: `experiments/gm1_c3urx1_91/cand91.py:1-40`, `experiments/gm1_relation88/rel88.py:1-30`, `experiments/gm1_explicit87/quad87.py:268-338`, `experiments/ecro86a/pt86a.py:73-200`, `experiments/ecro_s94d/s94.py:148-165`, `experiments/ecro_mat94/an94.py:19,32`;
- greps of WL86B (r50), WL87 (RAYSEND) and WL88 (R0_INF).

Labels: [FACT] verified, [MEAS] measured value with source, [INF] reasoning (including my own arithmetic), [HYP] untested.

## 0. Bottom line

- **H-A: VIABLE_WITH_CHANGES**, but only as a *learning-free physical first-order contract plus decomposition experiment*, not as a "reusable learned transport" architecture.
  - The physics is essentially correct.
  - The estimator's sampling domain is not well defined as written.
  - The key geometric estimate is wrong by about 4–6× in d_x.
  - The falsification protocol has three defects that matter: an underpowered premise test, an acceptance bar the null already passes, and a missing baseline.
- **H-B: NOT_VIABLE as stated.**
  - The atom formula has a unit error.
  - A single atom cannot represent the CK2 regime: incidence-dependent reflectance over a large grazing aperture.
  - It has no measured target.
  - It is not materially distinct from H-A at first order.
- **Brief compliance.** H-A and H-B share their first-order estimator and both remove learning from the sharp path. They are nested, not "materially distinct". §6 tests only H-A, and H-B enters only via D4. So "MULTIPLE_CANDIDATES_REQUIRE_DISCRIMINATION" does not describe what the design actually does.
- **Defensible verdict label: ARCHITECTURE_SELECTION_PREMATURE.**
  - Whether a learning-free sharp path is admissible is an open owner decision (canon: EXP/C3UR are "not the global method to be published"; ECRO was reinstated 2026-11-13).
  - The CK2 floor decomposition (the design's D1) is a prerequisite that is independent of architecture.

## 1. Verified contracts and citation checks

| Design claim | Status |
|---|---|
| ECRO apply/flight/global, bins, 512-ray state, edit path 76–78 ms | [FACT] matches auditA §1 and read_code (`model86.py:109-124`, `pt86.py:153-156,353-402`) |
| `cand91.py:3-6`: C3URX1 relation is κ_s rays per (sender patch, outgoing **bin**), with an exact NEE read per landed ray | [FACT] verified (`cand91.py:3-9`) |
| SAM density form (`rel88.py:5-15`) | [FACT] verified. Also important and not stated by the design: SAM draws **x uniform on the receiver patch** and y stratified-uniform on the peaked sender patch (`rel88.py:7-9`). It is a joint (x, y) density over the whole patch, not a per-point one. |
| e2_p = exactly 2 events, vertex 2 on SMAT == 1, NEE with the same light, ER and visibility | [FACT] `pt86a.py:73-200`: receiver probe, BSDF sample, k = 2 vertex material test |
| `quad87.qtable` removes "L1/L2/L12" | **Partly wrong.** [FACT] `quad87.py:268-338`: `qtable(ctx, lt, P, Mq, seed)` uses P stratified-jittered points per patch (WL87 ran P = 16) and exact per-point light directions with MIS ER sampling, binned to the outgoing patch bin. It removes L1 and L12 and replaces the fixed 16-point pattern with fresh stratified points. **L2 (finite spatial points) stays unless P is raised.** L4 stays by construction. |
| an94.W 8×8 defect, so use W_T | [FACT] `an94.py:19,32`. `s94.py:157` uses AN.W for the CK2 floor. The design's correction is right. |
| Stage B numbers (zero 1.000; learned 0.961–1.044; NMF16 0.603; L1-NMF16 0.380; field 0.36–0.54 / peak 0.35–0.44; rank ≈ 32; per-patch 0.263 / 0.451) | [MEAS] quoted correctly |
| WL90 97–102 %, A / S / P2plus / sub-bin residuals | [MEAS] quoted correctly |
| WL91 pooled ≈ ORACLE, sampling term 0.005–0.014, held-out 1/3, rule (a) 3.1× | [MEAS] quoted correctly |
| "Every oracle level is well defined with existing code" | **Overstated.** [FACT] `cand91` imports `fx91` (GM1 fixtures) and `rel88` imports `fx88`. Neither has been run on CK2, which uses a run-time-patched `pt86.Ctx.trace` and `scene_ck2.json`. Sub-bin q (H1q-sa) has no implementation: `gap90`'s X(a, s) projects path-traced Lx, not q. [INF] These need porting and new code; they are not reuse. |

## 2. H-A (LCFG): findings

### 2.1 Fatal flaws
None in the physics.
- [INF] L₁ (point-light NEE × f), the first-gather area integral with G = cos cos / r², the projection definition B_k = (1/(A ΔΩ))∫∫ L_k, the passivity argument and the prefix-order partition (= OWN t2 / t3) are dimensionally and physically correct.
- The mirror-image stationary-point construction (p′ = p − 2((p − o)·n)n, then the line–plane intersection) is correct for a planar sender.

### 2.2 Serious issues

**S-A1. The MIS estimator's integration domain is undefined, and "unbiased" is misapplied.**
- [FACT] The deployed state rays each have their own origin, uniformly spread over the patch, from one fixed draw (`pt86.py:114-116, 359-363`). No ray starts at a fixed receiver sample x_j.
- The design pairs per-point proposals at 16 fixed sub-patch points with a per-point p_u(y). That does not form a valid balance heuristic: both strategies must sample the same (x, y) domain with known joint densities.
- SAM (`rel88.py:7-9`) does this correctly: x is uniform on the patch for both strategies.
- [INF] At 4×4 sub-patch resolution the uniform strategy has 512/16 = 32 rays per sub-patch, i.e. **0.5 rays per incident bin per sub-patch**. Two consequences:
  - The non-peaked family, which carries 89 % of what region C reads (WL86B, Lambert senders [MEAS]), becomes about 16× noisier per cell unless it stays at patch level. The design's B₂ storage rule does not separate the two.
  - The MIS tails that p_u owns are almost unsampled.
- [INF] A single fixed deployed realisation is a deterministic quadrature. "Unbiased" holds only over realisations. WL87/88 put the deployed draw in the bad tail: −0.049 vs pooled −0.024 vs R0_INF −0.007 [MEAS].
- Required change: define B₂ cells, then sample x uniformly inside each cell for *both* strategies, as SAM does. Keep the non-peaked contributions at patch level, or justify the extra variance. Report per-realisation spread, not "unbiasedness".

**S-A2. The proposal's tail behaviour and the "VNDF equivalent" are unsupported.**
- [INF] GGX D in slope space is a ν = 2 Student-t, with tails ∝ |m|⁻⁴. An anisotropic Gaussian proposal has much lighter tails. In the tails, p_u must carry the integrand, and by S-A1 it is sparse. That is a firefly or variance mechanism exactly where WL91 already failed on variance.
- [INF] "ER-lobe (VNDF) samples mapped to the plane with an analytic Jacobian" is not a receiver-conditioned area density. VNDF sampling is defined at a known y with a known ω_i. Inverting h → y for fixed (x, p_l) means solving a reflection-condition root per sample (a specular-manifold-type Jacobian), not a closed form.
- Required change: use only an explicit area density with heavy (GGX-matched) tails. Drop the VNDF equivalence claim.

**S-A3. The geometry estimate in §2.2 is wrong, and it drives the falsification design.**
- [INF, my arithmetic from auditA §3.2 coordinates]:
  - The footprint senders are the panel corner pair, y ∈ [−0.3, −0.1], z ∈ [1.3, 1.7], on x = −1.5.
  - The light is at (−1.45, −0.6, 1.7), so d_l ≈ 0.45 m. Check: incident cos = 0.05/0.45 ≈ 0.11, inside auditA's 0.10–0.16.
  - The footprint receivers are the +y column of box A's panel face, at about (−1.32 to −1.35, 0.87–1.0, 0.55–1.45). So **d_x ≈ 1.1–1.5 m, not 0.25 m**. The 0.15–0.29 m figure is the *perpendicular* gap. At exit cos ≈ 0.11–0.16, the path length is gap/cos ≈ 1–2 m. Check: outgoing cos ≈ 0.17/1.2 ≈ 0.14, consistent with a mirror.
  - With α ≈ 0.05: d_h ≈ 0.33 m, σ_⊥ ≈ 3.3 cm, σ_∥ ≈ 28 cm. The strip is longer than a 20 cm panel patch.
  - The angular extent seen from x is about 2α·d_l/(d_l + d_x) ≈ 0.026 rad, giving Ω ≈ 2–4·10⁻³ sr.
  - With WL86B's measured panel-lobe r50 = 0.146 rad (`86B…md:106`) in place of 2α: about 0.038 rad, Ω ≈ 4.5·10⁻³ sr.
- The design states 0.01–0.015 sr, which is 3–7× too large. Its own inputs (d_x = 0.25) give about 4.6·10⁻³ sr once cos θ_y is included.
- [INF] Expected hits per receiver patch in the relevant bin, per realisation: 8·Ω/ΔΩ ≈ **0.16–0.4, not ≈ 1**. The "≈ 1-event Poisson" mechanism is qualitatively in the right direction (rare event) but numerically wrong. Its "consistency" with WL91 / R0X1 is not a calibrated match: those are GM1 and sender-side (WL91) measurements, not CK2.
- Consequence: see §6-1 below. S1 is underpowered.

**S-A4. A source-dependent cache gives up the only measured learned advantage and touches the contract.**
- [FACT] 𝒜's cache (proposals, visibility bits, B₂) is per source configuration. A light move recomputes the whole first order of that light by ray tracing (about 0.65 M rays per light plus probes, in the design's own estimate).
- [MEAS] ECRO's only measured advantage over EXP is the source-only update: 13–18 vs 51–64 ms, 3.5–3.9× (WL87:331).
- [FACT] WL86B:163 states that learning was meant to remove the per-light physical fill. WL88's hypothesis required "no source-dependent reconstruction".
- S3 gates per-edit time against ECRO but **never gates source-only update cost**. "Linear in I_l for fixed positions" is not source independence in the canonical sense.
- Required change: add a source-update cost clause against ECRO (13–18 ms) and EXP / C3URX1 (WL91: source 75.8 ms). State openly that H-A trades away source-independent reuse.

**S-A5. Role mismatch with the brief and canon.**
- [FACT] The design itself admits that the learned content shrinks to ER (already frozen, WL84) plus an optional ℛ.
- [MEAS] WL87 found no ℛ advantage over sweeps.
- That makes H-A = WL86B H-0 (the null) plus a better first-gather estimator. Recommending it as the "primary hypothesis" presupposes an owner decision the design itself lists as open (§7.1).
- Required change: present H-A as a physical-contract and comparator hypothesis, or leave the verdict at SELECTION_PREMATURE.

**S-A6. Attribution is generalised across regimes.**
- [MEAS] In the ECRO E2 T1 region C, converged receiver-uniform plus bin-q R0_INF is E_H 0.025 (signed −0.007), deployed R0 is 0.049, and ORACLE is 0.0137 (WL88:266; WL86A).
- So in the Lambert-via-mirror regime that ECRO fails on, receiver sampling (L5) dominates. Bin projection (L4) is a smaller residual. "97–102 %" is a GM1 R_mirror fact with *glossy S1 receivers*.
- The design labels L4 as "inferred for CK2" in §2.3, but the §5 "minimum evidence" and the §8 verdict lean on WL90 as if it carried over.
- [MEAS] WL87 RAYSEND (exact once-scattered read in the first gather, on the deployed ECRO state, E2 T1 C) = **0.059 vs EXP 0.049**. This is the direct ECRO-scene analogue of the design's H2, and it is not cited.

**S-A7. Overstated Stage B readings.**
- "No learned content to preserve": the zero-operator level holds in the input-independent operator L1 metric. At field level, learned is 0.846–0.892 against zero 1.000, with energy 0.83–1.09 [MEAS auditB].
  - It *does* deliver energy.
  - [MEAS] S_geo < 1 % supports "no geometry content". It does not support "no content".
- "Rank 16 caps field L1 at ≈ 0.36": [MEAS auditB] this is the best value *found* among the factorisations tried, at a local optimum, measured against the bin-projected M64 teacher and not physical q. It is an empirical floor, not a proven bound. The near-permutation argument is sound: the ideal mirror in this layout maps (iz, iφ) → (iz, iφ + 8 mod 16), an exact rank-64 permutation [INF].
- "CK2 C1 mechanism": the claim that box A rays reach the panel at panel-frame z ≈ 0.2–1.0 omits the footprint's own paths at z 0.10–0.16 (auditA §3.2). The over-delivery prediction needs a per-patch histogram of ray z, not a range.

### 2.3 What is genuinely supported
- Replacing the learned S1 sender with a physical one helped at render level in every recorded case [MEAS]: WL93D C2_src; WL94 / 94C / 94D C2_srcS1, including the weak s8601 C-arm case (0.084 → 0.076).
- S1's local response is geometry-common (S_geo 0.2–0.8 %, E_common ≈ 0.80) [MEAS]. A physical, material-level owner (ER) is therefore physically justified [INF].
- WL88's SAM shows that sender-area MIS fixes coverage (0.59 → 0.993) [MEAS]. WL90/91 show that exact first-gather reads remove the GM1 projection bias when pooled [MEAS]. H3 is the conjunction of two measured mechanisms plus one new element: a lobe-shaped rather than uniform area proposal.
- The cost shift into ray tracing is declared rather than hidden, which complies with constraint 13 (FNTO contract :86, 88, 150).

### 2.4 Required changes for a valid falsifiable hypothesis
1. Fix S-A1, S-A2 and S-A3 (domain, tails, geometry). Before freezing any threshold, predict per-patch hit counts and variance from computed CK2 geometry.
2. Add the missing arms:
   - **SAM + X1**: uniform sender-area proposal with exact reads, at matched rays. Without it, S2 cannot attribute any gain to lobe-awareness.
   - **RAYSEND (WL87)** re-run on CK2.
3. Replace the acceptance bar. "E2 region C ≤ 0.06 single draw" is already met by EXP (0.049) and C3UR (0.023–0.026) [MEAS]. Acceptance must be relative to the strongest same-state comparator (C3UR / SAM / C3URX1) at matched total rays *and* matched source-update cost, with ORACLE 0.0137 as the table floor.
4. Matching total rays changes κ_u, and so the deployed state contract. Declare this.
5. Count the sphere honestly. The memory budget includes 32 tangent-plane "planes", but §6 says S2 falls back to p_u. Tangent-plane stationary points on a sphere are not the true ones.

## 3. H-B (VSA): findings

### 3.1 Fatal flaws as stated
1. **Unit error.** E(x) = Φ_a cos θ_x / |x − p′|² uses a flux Φ (W) where a radiant intensity I (W/sr) is needed. [FACT, radiometry]
2. **One atom per (light, plane) cannot represent the target regime.**
   - On CK2 the incident cos ranges 0.024–0.164 over the panel and irradiance varies 307× [INF, auditA].
   - The coated ER has incidence-dependent energy s(μ), Fresnel and masking (WL82). The reflected intensity of the image source is therefore a strong function of direction across the aperture.
   - A scalar Φ with a single μ_i is wrong at exactly the grazing, large-aperture configuration that fails.
   - Making I direction-dependent over the aperture turns the atom back into H-A's per-receiver evaluation.
3. **The Gaussian Σ propagation (Σ_o = JΣJᵀ + Σ_ER) is unjustified for GGX at grazing incidence.** The lobes are heavy-tailed, asymmetric and horizon-clipped. "Exact for ideal planar mirrors" does not extend to α ≈ 0.05 at cos ≈ 0.1.

### 3.2 Serious issues
- **No measured target.**
  - [MEAS] WL90 P2plus ≤ 0.0009 (GM1, including S1 → S1).
  - [MEAS] WL86B: ≤ 11 % of what region C reads is glossy → glossy.
  - [MEAS] WL88's S → S workload was STOPPED_WORKLOAD_UNQUALIFIED.
  - [MEAS] ORACLE (readout-vertex projection included) is 0.0137 / 0.006.
- **Mis-cited support.** The WL86A beam pairs (0.94 / 0.62 vs noise 0.26 / 0.22) show *incident-encoding* loss at the first sender (L1). That supports removing s at first order, which H-A already does, and does not support H-B's added persistent order-≥2 state.
- **Internal contradiction.** "Cost independent of α", yet the gather of blurred atoms uses H-A's footprint estimator, whose ray cost scales with footprint area ∝ α²·d_h².
- **The class threshold re-creates the H-P objection.** WL86B's "boundary drawn post hoc at the panel" (r50 0.146 vs 0.15) applies however the threshold is "measured", because the failing material is already known.

### 3.3 Supported
- Image-source geometry for ideal planar mirrors [FACT].
- Exact lobe-wise energy partition, if ER's lobe albedos are separable non-negative terms [INF; not verified in the ER code].

### 3.4 Verdict: NOT_VIABLE
- Reasons: §3.1 (1)–(3), plus no target and no material distinctness.
- To be revisited only after:
  - a qualified S → S or sharp-mover workload shows a beyond-noise residual from paths with ≥ 2 peaked vertices, *after* an exact first gather;
  - the atoms are redefined with direction-dependent intensity over the aperture.

## 4. Defects in the §6 falsification design

1. **S1 is underpowered.**
   - [INF from S-A3] H2-pool with 64 realisations gives about 10–26 hits per footprint patch, i.e. 20–30 % Poisson SD per patch before weight variance. The footprint patches dominate box A's e2_p signal.
   - The threshold is 2 × half-noise + 0.02 ≈ 0.136 rel L1. Variance alone can approach or exceed it.
   - The S1 failure branch ("attribute to semantics, ER at grazing or reference, and stop") would then misattribute variance as bias.
   - "ER at grazing" cannot explain an H2-pool vs e2_p mismatch at all, since both evaluate the same frozen ER. Only code-path or frame mismatch, variance or reference definition can.
   - **Fix:** add a well-defined *deterministic dense area quadrature* over the panel plane: exact L₁ at mm-scale cells, plus shadow and receiver visibility, per box-A patch. For a point light and a planar sender this is the numerically well-defined "exact sender + higher-fidelity propagation" level, and it needs no relation sampling. Size the realisations from the predicted hit counts.
2. **S0's positive control (e2_o on box A, deployed single draw) can fail on sampling.**
   - [MEAS] WL87/88 show the deployed draw in the bad tail.
   - That would declare the implementation invalid wrongly. Use pooled relations, or a tolerance taken from the realisation spread.
3. **The ladder mixes deployed and pooled relations in single-factor steps.**
   - H1q-sa on the deployed state has 0.5 rays per sub-bin, which confounds C1 with C5.
   - [FACT] E_H and rel L1 are absolute-value metrics, so step differences are not additive components (WL91 caveat).
   - **Fix:** a factorial of {M64(s), q(P = 16), q(P ≫ 16), sub-bin q, X1, dense quadrature} × {deployed, pooled}.
   - C4 (16 vs many points) and C6 (ring-0/1 cosine aliasing) have no isolating arm. q with P = 16 does not remove L2 (§1).
4. **S3's anchor is arbitrary.**
   - "Added per-edit time > ECRO's 76–78 ms" permits a total of about 2× ECRO. PT per edit is 36 ms (WL87 [MEAS]).
   - Keep only the quality-matched edit-to-first-frame clause and the comparator clause, and add source-update cost (S-A4).
5. **Regime labels.** CK2 ROI_T, ECRO region C and GM1 R_mirror must stay unpooled. The design does label them. Keep that, and do not carry WL90's GM1 share into CK2 claims.

## 5. Evidence-consistency summary

- **Contradicted or uncited by the design:**
  - WL87 RAYSEND 0.059 > EXP 0.049 (exact reads on the deployed ECRO state are worse);
  - WL88 R0_INF 0.025 / −0.007 (in E2, sampling dominates projection);
  - qtable P = 16 (L2 not removed);
  - CK2 d_x ≈ 1.1–1.5 m.
- **Unmatched metric use:** none found in the design's tables. The W_T correction and the matched-norm Stage B citations are handled correctly. The design does mix one norm family in its sentences: "rank-16 cap" (M64-relative field L1) next to WL90 E_H shares. These are kept in separate sentences, but §8's causal sentence rests on both.
- **Untested mechanisms presented as fixes:** largely avoided ([HYP] tags, "untested"). Exceptions:
  - the claim that L5 is "replaced by a lobe-aware proposal that targets the ≈ 1-hit mechanism", whose quantitative basis is wrong (S-A3);
  - "L8 closed by construction", which holds only with an explicit ℛ.

## 6. Verdicts
- **H-A: VIABLE_WITH_CHANGES** (S-A1 to S-A7 and §4 1–4). Role: a learning-free physical first-order contract and CK2 decomposition. It is not, on current canon, the learned method.
- **H-B: NOT_VIABLE** (§3).
- **Design-level verdict:** replace MULTIPLE_CANDIDATES_REQUIRE_DISCRIMINATION with **ARCHITECTURE_SELECTION_PREMATURE**.
  - The one well-founded next step is the architecture-agnostic CK2 decomposition (D1): dense-quadrature oracle, factorial arms, W_T, sized realisations.
  - Admitting a learning-free sharp first order, and giving up source-independent reuse to get it, is an owner decision that this evidence cannot settle.
