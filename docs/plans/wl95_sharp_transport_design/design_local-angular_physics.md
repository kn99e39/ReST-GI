# WL95 adversarial review: design "local-angular" (physics, representability and evidence-consistency lens)

**Scope.** Read-only review. I edited no repository file and ran nothing except `sed`/`grep`. I read all eight evidence files. The code checks are cited inline.

Labels: [FACT] verified, [MEAS] measured value with its source, [INF] inference, [HYP] untested.

## 0. Verdict summary

| Item | Verdict |
|---|---|
| H1 LRS | **VIABLE_WITH_CHANGES**, but only as a *learning-free comparator / oracle-ladder hypothesis*, not as an answer to the learned-transport question. No fatal physics flaw. Several serious evidence and specification errors (below). |
| H2 MEK | **NOT_VIABLE** as an independent hypothesis. It is not materially distinct from H1, it duplicates the existing learned ER layer, and its built-in isotropy contradicts the anisotropic ER closure. |
| Design-level conclusion | MULTIPLE_CANDIDATES_REQUIRE_DISCRIMINATION is **not supported**. With H2 dominated by construction (the designer says so) and H1 a recombination of designated comparators, the defensible conclusion is **ARCHITECTURE_SELECTION_PREMATURE**. The proposed smoke is valuable as an attribution ladder for the open CK2 floor. |
| Falsification smoke | Acceptance (d) cannot be met by construction on s8602 and on E2. Other gaps are listed in §4. |

## 1. Cross-cutting findings (verified)

**F1. Cosine convention error that would double-count cos θ_o.** [FACT]
- `experiments/fnto_compiled83/closure83_dr.py:7-11` defines `eval(wi, wo) = f cos(theta_o)`, not f. The Lambert term is `lam_0 cos/pi`.
- H1 writes L1_j(ω_d) = Σ_l f_ER E_lj and then deposits Φ = L1 cos θ_d Δω_d · A_p/16.
- If `DrClosure83.eval` is used as "f_ER", cos θ_o is applied twice. That is a systematic under-delivery that grows towards grazing exit, which is exactly the CK2 regime (z_out ≈ 0.02–0.10).
- H2's λ_0 cos/π follows the closure (f·cos_o) convention, while H1's equations use the BRDF convention. The design mixes the two.
- Required: state one convention. With `eval`, the deposit is Φ = eval·E·Δω_d·A_p/P_s, with no extra cos.
- Same for F̄_m: it must be defined on f (as M64/fbar is: no cosine, `quad87.py:85-110`), not on `eval`.

**F2. The closest measured analogue of H1 is mislabelled.**
- [FACT] `experiments/gm1_c3urx1_91/cand91.py:3-9` and WL91:19 define C3URX1 as the WL89 **sender-side**, source-independent, geometry-only relation C3UR (8 rays per sender-patch × outgoing-bin cell, 650,240 rays, bit-identical selective update), with **exact continuous once-scattered radiance at each ray's own origin and direction**. WL88:153 calls C3UR "sender-side ... control, not a candidate".
- The design's §2E and rejected alternative #3 file C3URX1 under "exact per-ray sender reads on *receiver-side* rays". That is wrong.
- [MEAS] The sender-side + exact-continuous-sender combination that H1 builds on has already been measured (WL91:151-179, 310-317):
  - pooled bias is removed (oracle-level);
  - the sampling term is larger than C3UR's in all 6 conditions;
  - the deployed W2 T1 draw scores 0.053, worse than C3UR's 0.038;
  - held-out validity holds in 1 of 3 conditions;
  - quality-matched utility is NOT SUPPORTED (3.1× slower at the 0.06 tier).
- H1's only distinguishing claims are a deterministic dense direction grid (D = 1,024 per origin, against C3UR's 8 random rays per cell) and restriction to peaked senders.
  - [INF] A fixed deterministic grid does not remove sampling error. It freezes it into a geometry-dependent bias.
  - This is the same phenomenon WL91:393 describes for the q table, and the same "one fixed pattern shared by every patch" defect as L5 (the deployed pattern sat in the bad tail, WL87:245-262).
- Required:
  - C3URX1 must be a named baseline arm;
  - H1's distinction from it must be stated as a [HYP];
  - the smoke needs grid-realisation controls (rotated or jittered grids), so that "deterministic" is not a single lucky or unlucky draw.

**F3. Spatial sampling becomes coarser, not finer.** [FACT/INF]
- H1 origins are the 16 existing source points (`pt86.py:378-402`: per-point `e`, `wl`, `bn` exist before the `bincount`, so the atoms are genuinely available) [FACT]. C3URX1 used 512 ray origins per patch.
- On CK2 (auditA §3.2):
  - panel irradiance varies 307×;
  - the box-A footprint is fed by about 10 cm strips of 2 panel patches, i.e. about 16 of H1's fixed, non-stratified points [INF];
  - each point's grazing-incidence lobe is stretched.
- H1's "L2 / L4-spatial reduced to 16-point quadrature" is therefore relabelling, not reduction. L2 is unchanged in count and changes from a patch-mean bias to point-located emission. On the binding workload it may be the dominant term (auditA C2/C4 are open).
- The default P_s = 16 used in every cost estimate is unjustified. P_s = 64 multiplies the rays and the source shadow rays by 4.

**F4. The angular resolution requirement is mis-scaled.**
- D ∝ α⁻² covers only lobe resolution. The delivered flux per receiver patch is a quadrature over the intersection of the lobe with the solid angle the receiver subtends.
- [INF] Back-wall patches (25 cm at about 1.8 m) subtend ≈ 0.019 sr, about 3 cells at D = 1,024, against about 11 cells inside the S1 50 % cap. So per-origin delivery is a 2–3-sample midpoint sum.
- With all 16 origins sharing one grid orientation on a flat panel, the aliasing is coherent across origins.
- The required D scales roughly as max(α⁻², d²/A_r), which has a distance term even for broad lobes. The worst-case and scaling statements must include it.
- WL90's "16 sub-bins ≈ 0" was measured for a sender *table* read through a *pooled (converged) 64-realisation* relation (WL90:42-59, 199-204). It is not evidence that a single deterministic grid of 1,024 rays per origin converges. The design conflates table resolution with ray-quadrature density.

**F5. The H1 sharp path covers e2_p only, but on the CK2 panel face the failing mass is mostly higher-order panel-mediated.**
- [MEAS] e3_p is 0.755 of e3 on the CK2 panel face (WL94B audit). 61 % of B on the panel face is higher-order (WL94:223-234). 69 % of the panel-face loss-normalised target is panel-mediated (WL94B:94-104).
- H1 sends indirect incidence at peaked patches through bins, the existing 512-ray state and R, which are the L4/L5/L7/L12 machinery.
- [INF, partly in H1's favour] Indirect incidence on the panel from nearby Lambert faces is angularly broad, so bins may be adequate there. But that is unmeasured on CK2: WL90's "orders ≥ 2 ≤ 0.001" is GM1, non-grazing.
- Required: a matched tally Q_d3 for e3_p. It already exists in `pt86a.field_src(split=True)`, so this invents no oracle. Without it, panel-face Q_r cannot be attributed.

**F6. Direct/indirect branch in propagation.**
- [FACT] Architecture.md:21: "Direct and C-mediated incidence use the same intrinsic response rule; no source-identity or direct/indirect response branch."
- H1 does more than change the quadrature of the incident measure. Emission from a peaked patch is propagated by the dense sender relation if its incidence was direct, and by the 512-ray receiver state or R otherwise.
- The local rule (f_ER) is shared, but the C approximation differs by event order. That puts a quality discontinuity at the event-count boundary, on top of ECRO's still-unplaced event-count partition (read_canon §B).
- This must be declared as a contract deviation for owner decision, not as "only a quadrature choice". The alternative (dense relation for all orders from peaked senders) requires a non-binned representation of indirect incidence, which is the design's real open problem.

**F7. Objective conflict (designer partly discloses it).**
- [FACT] EXP and C3UR are "not the global method to be published" (Architecture.md:13; ROADMAP:128).
- Prior reviews rejected designs whose sharp part is fully retraced and in which "nothing global is learned":
  - LPTG, WL84A;
  - H-P, WL86B:93-110.
- H1 = H-0/EXP for non-peaked senders + C3UR placement + X1 exact sender, densified, + optional R_θ whose utility over explicit sweeps is the open H-0 question.
- H1 also gives up ECRO's only measured learned advantage, the source-only update (13–18 ms against EXP's 51–64 ms, WL87:331): non-peaked senders need an EXP-style q.
- So H1 answers "what does an explicit method need?", not the brief's learned-representation question. It must be labelled as a comparator / oracle-ladder hypothesis.

**F8. Rank versus optimisation wording is internally inconsistent.**
- §2B says rank is "binding by structure, not by training". §2C and Stage B show:
  - the learned S1 operators sit at the zero-operator level: op L1 0.961–1.044, val field L1 0.846–0.892;
  - admissible non-negative rank 16 reaches 0.36–0.54 field L1 (auditB Table 1).
- So the currently binding gap is optimisation/parameterisation. Rank becomes binding only after that gap closes.
- "Neither can be removed by more training of a cᵀ" is a [HYP]. WL94D ran 1,000 steps, and the loss normalisation (level over lobe) is not isolated.
- Supporting evidence the design omits: WL92's ladder shows that material NMF16 at S1 costs rendered E_H (R16 0.0843 / 0.0726 against ENCB 0.0449 / 0.0524; pooled 0.064 against 0.027) [MEAS, WL92:144-156]. That is the one measured rendering consequence of rank 16.
- The "rank ≈ 90" estimate (2π/0.067 = 94) cannot be confirmed by the 46–59 rank measured inside a 64-dimensional basis, whose rank is capped at 64. It is [INF] order-of-magnitude only.

**F9. One-sided evidence citation.**
- WL93D's C2_src is the best render arm but worsens the pooled single-flight table (+0.0052 to +0.0079) [MEAS, read_wl93 §1]. The table-versus-sharp-reader metric conflict must be carried into H1's acceptance.

**F10. Minor errors.**
1. "Superposition: continuous exact, binned not" is wrong. Binned ECRO is linear and superposition-exact by construction. What binned loses is the within-bin direction of two co-binned lights.
2. Ring-3 azimuth width is ≤ 0.26 rad (0.39 × sin 41.4°), not 0.29.
3. Cost increments are understated. H1 *keeps* the 512-ray state and adds the dense relation, so the extra cost is the full ≈ 50 / 200 ms, not +25–175 ms.
4. Memory is understated. The dirty test needs segment endpoints, as the current state stores at 27 B/ray (read_code §2). That gives ≈ 33 MB at D = 1,024 and ≈ 0.56 GB in the all-peaked worst case, ×6 again at α = 0.02.
5. The Π > 8 threshold was fixed in WL88 with the values already in view (WL88:188: B 4.93, S1 37.95, S2 16.13). It is not prospective in the WL86B sense. Π is itself computed on bin-averaged fbar (WL88:115).

## 2. H1 (LRS)

**Fatal flaws:** none in the transport physics.
- Units check [INF]: E (W/m²) × f (1/sr) = L. L·cosθ·Δω·A/P_s = W. Dividing by A_r gives patch-mean irradiance per receiver bin, consistent with ECRO's E1 = ∫_bin L cos dω.
- The flux deposit is linear, non-negative and superposing.
- The sender-side and receiver-side estimators target the same operator (WL88 target identity).

**Serious issues:** F1 through F7 above, plus the following.
- **Underspecified resolvent and readout types.**
  - R outputs B3, which is outgoing bin radiance.
  - The peaked-x2 readout needs an incident field I_p, which is undefined ("bin fields E1 and I").
  - Only E1 is physical, so the readout either drops orders ≥ 3 at peaked x2 or needs R to output incident fields.
  - F̄_m is a uniform-solid-angle incident-bin average with no cosine, while I_p carries the cosine. So L12 reappears at the readout for indirect incidence.
- **"Exact bin M64/ER senders" for non-peaked patches is ambiguous.** M64(s) carries L1/L2/L12 (ENC: 0.054 on WL87 E2 C). MIS q (EXP: 0.049) does not. Name one.
- **Q_d against e2_p is unmatched for S2 and for non-Lambert receivers.**
  - [FACT] e2_p tallies vertex 2 on shape-material 1 only (`pt86a.py:175-180`). Sphere (S2) mediated delivery sits in e2_o, so the "peaked" family in Q_d must be S1-only or needs a new tally.
  - Irradiance cannot be recovered from e2_p at non-Lambert receivers without the receiver BRDF, which reintroduces L7/L12. Restrict Q_d to Lambert receivers.

**Genuinely supported by evidence:**
- [MEAS] WL90: the first-gather angular projection dominates at converged propagation (GM1).
- [MEAS] WL91: exact continuous sender radiance plus sender-side placement removes the bias when pooled.
- [MEAS] WL88: sender-side placement is the dominant factor at equal rays, on a single draw.
- [MEAS] WL93C/93D/94: the exact M64 sender is the best render arm (ORA edit slope 0.995–1.0).
- [MEAS] Stage B: the deployed learned local operator is far below its own rank bound. ER is passive and reciprocal (furnace 6e-7, reciprocity 1.2e-5).
- [FACT] The atoms already exist in the source pass. The relation is geometry-only, so selective = full bitwise is achievable (`ecro86.py:57-78`).

**Changes required for a valid falsifiable hypothesis:**
1. Reclassify it as a learning-free comparator / oracle ladder.
2. Fix the cos convention (F1).
3. Add C3URX1, C3UR (deployed and pooled), EXP (deployed and pooled), stale and same-mechanism full-rebuild arms.
4. Add grid-realisation controls (F2).
5. Make the D scaling include the receiver's subtended solid angle (F4). Make P_s part of the default cost and use a stratified or denser origin set on CK2 (F3).
6. Add a matched e3_p tally (F5).
7. Declare the direct/indirect propagation branch as a contract deviation (F6).
8. Define R's output type and I_p.
9. Restrict Q_d to S1 senders and Lambert receivers, or add tallies.
10. Use the true 4×16 W everywhere (the design already proposes this).
11. Evaluate Q_r with H1's own explicit pipeline, not inside each seed's co-adapted learned receiver and G (WL93D co-adaptation).

**Verdict: VIABLE_WITH_CHANGES** in the comparator/attribution role. It is not viable as the learned-architecture answer without an explicit owner decision that physically traced sharp first scatter is acceptable.

## 3. H2 (MEK)

**Fatal flaws:**
1. **Built-in azimuthal isotropy contradicts the material it must fit.** [FACT] The ER lobes are "horizon-safe anisotropic GGX", with separate au/av per lobe (`closure83_dr.py:3-4, 11`). An isotropic kernel cannot be exact for an anisotropic state. Whether the ECRO S1 state has au ≠ av was not checked (`mats86.py` copies the WL85 fixture states). Either way, the claim "exact ER for any state" fails in general.
2. **Not materially distinct from H1.**
   - Identical state, propagation, deposits, resolvent and invalidation. Only the local evaluator differs.
   - It is a learned parametric fit of a known parametric closure, which "adds approximation error with no information gain" (the designer's own words).
   - The project already has a learned local layer: WL84 learned ER, the compiled learned material that is the "primary local-scattering development baseline" (Architecture.md:11, 42). H2's "keeps a learned local layer" justification is therefore void.
   - A BSDF surrogate alone does not count as the contribution (RESEARCH_CENTRIC_TOPIC.md:11).

**Serious issues:**
- The λ_0 cos/π convention conflicts with H1 (F1).
- "Held-out field L1 > 0.10 at peak < 0.8" is defined against the M64 bin teacher, which H2 by design does not use. The metric is undefined for a continuous kernel. The thresholds are arbitrary.
- The WL94D T result (energy moved, peak did not) is acknowledged as "not decisive", but no mechanism explains why softplus A/Σ smooth in θ_i would capture grazing Fresnel and masking steepness.

**Supported:** only [INF] from the Stage B rank structure, and S_geo < 1 % (a material-only kernel suffices for S1; floor B is 3–5 %).

**Verdict: NOT_VIABLE.** At most it is a closure-fit control arm in the H1 smoke.

## 4. Falsification experiment

**Acceptance (d) cannot be met by construction.**
- tol = max(2 × noise, 0.005) ≈ 0.017 [MEAS: noise bound 0.0084–0.0086, `s_s860x.json`].
- C2_srcS1 on untextured CK2 is 0.031 / 0.019 [MEAS: WL94:155-160].
- An improvement of more than 0.017 on s8602 needs E_H < 0.002, which is below the noise bound. So "both seeds" fails by construction.
- On E2 T1 C, C2_src is 0.018–0.024 against the oracle 0.0137 (noise ≈ 0.009). The total headroom is 0.004–0.010, which is less than tol. **E2 is non-discriminating for (d).**
- Fix: measure gains against the oracle / REF-ER floor with a noise-scaled effect size, inside H1's own pipeline.

**Acceptance (b) is ill-conditioned.** "≤ 1.5× O3 error", where O3 is near the noise floor once (a) passes, is a ratio of noise-level quantities. Use an absolute or effect-size rule, for example O2 − O3 ≤ max(0.5·(O1q − O3), 2 × noise).

**Acceptance (e) uses the wrong cost comparator.**
- "Path-tracing the e2_p family" is not a canonical baseline.
- The binding bar is quality-matched full-image PT edit-to-frame (WL67/69/70; read_canon D9), plus EXP, C3UR and C3URX1 per-edit and per-source costs.
- The mandatory Neural Full Recompute / Native Reuse gate must name R_θ as the only owned learned state and state whether an edit invalidates it.

**What is good and should be kept:**
- The O1q / O3q / O3 split mirrors WL90's J/A decomposition and is well defined. It directly decides whether the CK2 floor is sender-representation or propagation.
- O2r reuses an existing capability (X1).
- O4 = e2_p with half-split noise.
- Correct box-A mask and the true W.
- SNR ≥ 4 prequalification of the non-grazing case.

**Missing controls:**
- grid-realisation;
- a P_s stratification ladder on CK2 specifically;
- the e3_p family;
- C3URX1;
- stale and full-rebuild arms;
- an S1-only peaked family to match e2_p;
- the table-versus-render metric conflict (F9).

## 5. Answer to the brief's checks

**Loss points:** addressed, relabelled or untouched.

| Status | Loss points |
|---|---|
| Genuinely addressed for first scatter at peaked senders | L3 (by deletion), L1, L12, L4-angular, L6 |
| Relabelled or unchanged | L2 / L4-spatial (F3); L5, replaced by a deterministic pattern with the same fixed-pattern risk (F2) |
| Untouched or reintroduced | L7; L12 at the peaked readout; L10 for indirect incidence; the panel-mediated ≥ 3-event family (F5) |
| Rewired | L8 |

**Unmatched comparisons in the design:**
- the S1 rank-90 estimate against the 64-bin rank 46–59 (F8);
- WL90's table-resolution result reused as a ray-density requirement (F4);
- Q_d against e2_p for S2 and for non-Lambert receivers (§2).

The Stage B matched numbers themselves are quoted correctly.

**Sharp-lobe quantitative plausibility:**
- Angular: D = 1,024 gives about 11 cells per 50 % cap, which is plausible for lobe resolution [INF].
- Spatial: about 16 origins feed the CK2 footprint from a 307× field, which is implausible as adequate at P_s = 16 [INF].
- Far small receivers get only 2–3 cells per origin [INF].
- Mirror limit (α → 0): the grid cannot resolve the lobe, as the designer acknowledges.

**Design-level verdict: ARCHITECTURE_SELECTION_PREMATURE.** The proposed smoke, corrected as above, is a valid bounded attribution of the open CK2 floor. It is not a discrimination between two architecture candidates.

## 6. Sources verified

Code:
- `c:\Projects\ReST-GI\experiments\fnto_compiled83\closure83_dr.py:1-15` (eval = f cos θ_o; anisotropic GGX au/av; 69-float state)
- `c:\Projects\ReST-GI\experiments\ecro86\pt86.py:378-402` (per-point atoms before `bincount`)
- `c:\Projects\ReST-GI\experiments\ecro86\scene86.py:15-25` (material ids)
- `c:\Projects\ReST-GI\experiments\ecro86\mats86.py:1-35` (ER states copied from the WL85 fixture)
- `c:\Projects\ReST-GI\experiments\gm1_c3urx1_91\cand91.py:1-15` (C3URX1 is sender-side, with exact per-ray L1)

Docs:
- `c:\Projects\ReST-GI\docs\Architecture.md:17-38` (no direct/indirect branch; K linear, non-negative, passive)
- `c:\Projects\ReST-GI\docs\worklogs\91_*.md:10, 19, 23` (C3UR relation definition)
- `c:\Projects\ReST-GI\docs\worklogs\88_*.md:111-115, 153, 188` (Π rule; threshold fixed alongside the measured values; C3UR as sender-side control)

Evidence files: `auditA.md`, `auditB.md`, `read_code.md`, `read_canon.md`, `read_wl86_88.md`, `read_wl89_92.md`, `read_wl93.md`, `read_wl94.md`, all under `C:\Users\dna10\AppData\Local\Temp\claude\462fa2b2dbb71cdd\46d65e15-e083-4057-875f-9282fb68aa5e\scratchpad\wl95\`.