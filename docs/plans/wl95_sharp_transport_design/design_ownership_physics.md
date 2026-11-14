# WL95 adversarial critique of the "ownership track" design (physics, representability, evidence consistency)

**Scope.** I edited no files and ran nothing. I read the 8 evidence files in full. I verified the following code:
- `experiments/ecro86/model86.py:108-124` (apply)
- `experiments/gm1_explicit87/quad87.py:1-38, 48-110, 223-242` (`fbar`, dense table, `enc_matrix`)
- `experiments/gm1_relation88/rel88.py:40-62` (peaked rule)
- `experiments/ecro86a/pt86a.py:170-200` (e2/e3 _p/_o split)
- `docs/Architecture.md:17-38`
- `docs/worklogs/89_*.md:40-80, 250-382` (C3UR definition and floor)

Labels: [FACT], [MEAS], [INF], [HYP].

## 0. What the design gets right (verified)

- **Transport algebra is correct.** [FACT/INF]
  - q(y,v) = f·I cosθ_ℓ/d²·V and the B₂ triple integral are dimensionally correct (W m⁻² sr⁻¹).
  - The H-B photon flux Φ = I cosθ_ℓ/d²·V·(A/P_pts)·f cosθ_o/(m p) is in watts.
  - The deposits B₂ += Φ·fbar/A_n and E₁ += Φ/A_n are correct and need no receiver cosine.
  - The Neumann-series form is correct.
- **M64 is approximately symmetric.** M64 averages incident cells uniformly in solid angle (`quad87.py:48-56`) and the outgoing bin uniformly, so it is ≈ (1/ΔΩ²)∬f. With ER reciprocal (`quad87.py:4`), M64 is ≈ symmetric [INF; correct].
- **The e2/e3 tallies are exclusive and sum to t** (`pt86a.py:175-199`) [FACT]. So per-term supervision without sum-only coupling is available.
- **Stage B numbers are quoted in their correct norms** (operator-L1 zero 1.000 vs learned 0.96–1.04; NMF16 0.603; L1-NMF 0.380; field 0.36–0.54).

## 1. Cross-cutting defects (affect H-A and H-B)

**X1. The E2 evidence for "exact sender + learned G is fine" is an error cancellation, and the design cites it as support.** [MEAS, matched: same E2 T1 region C, deployed R0 rays, WL86 REF-ER]
- ENCB (c = I, a = M64, which equals X_M12 to T_FORM 0.018–0.021) with **path-traced L₃₊** scores **0.0449** single draw and 0.027 pooled (WL92).
- The same prefix with the **learned G_own** (ORA = X_M12 + G, WL93C) scores **0.030 / 0.033**. This is consistent with WL92B X_M12 f = 0.74 / 0.25: 0.077 − 0.74·(0.077 − 0.0127) ≈ 0.029.
- So swapping the exact tail for the learned tail *improves* E_H by 0.012–0.015. The learned G offsets the binned prefix's negative signed deficit (EXP signed −0.049).
- §2.1(2) and (4) use ORA and "no compensation conflict" as evidence that G is not binding. That evidence is contaminated by cancellation.
- H-A's Level 1 is **already measured** on E2 T1 C at 0.0449 (single draw). That **fails the design's own stop rule**: oracle + max(2·noise, 0.01) ≈ 0.0127 + ~0.018 ≈ 0.031.

**X2. The table in §2.1(3) compares numbers that are not matched.**
- ENCB has a path-traced L₃₊ tail. C3, C3U, C3UR and R0 (WL87/88) have explicit all-order tails, so the ≥3-event owners differ.
- C3UR reads EXP's **q** (exact light direction per point, 1,024 MIS per point; `quad87.py:10-11`), not M64(s).
- The criterion-2 cell "C3UR 0.026 single" is therefore not a measurement of H-A's prefix (M64(s) on Γ_s). That combination is unmeasured. The "+L1/L12" encoding difference was measured only at +0.004–0.005 on E2 (WL87 ENC 0.054 vs 0.049; WL92 ENCQ vs EXP), not under CK2 grazing incidence.

**X3. The "binned floor ~2× oracle" is established on GM1 S→S, not on the ECRO Lambert-receiver region C.**
- [MEAS] GM1 W1 R_mirror: converged R0_INF = R1_INF = C3UR_INF = 0.038 vs ORACLE 0.014 (WL89:256). This is a true bias floor.
- [MEAS] ECRO E2 T1 C: R0_INF over 64 realisations gives signed −0.007 vs oracle +0.003 (WL88). WL88 reinterprets WL87's pooled −0.024 as slow convergence.
- So the E2 "pooled 0.027 vs 0.013" is not shown to be a floor. It may be residual variance at 16 realisations.
- [INF] The H-A/H-B hinge is receiver-class dependent:
  - Lambert receivers (E2 C, CK2 box A) need only correct irradiance placement.
  - Glossy and sharp receivers (GM1 R_mirror, C_ss) expose sub-bin incidence.
- The proposed smallest workload **omits GM1 W1/W2**, which is the only workload where the floor is established. Without it, Stage 1 cannot discriminate on S→S families.

**X4. O4's dichotomy is false, and its sizing argument mixes per-material and per-geometry costs.**
- O4 claims the choice is either a binned reusable relation with a floor, or a per-source continuous flight. A third option is omitted: **source-independent angular refinement of the sender table and relation on peaked senders only.**
- Quantitative check:
  - [MEAS, WL90:199-204] A fixed sub-bin grid at a = 16 (1,024 directions) leaves +0.0007 / +0.0012 / −0.0031 of a J of 0.017–0.031, i.e. ≥ 90 % closed. At a = 4 the residual is 9–20 % of J.
  - [INF] For S1 at α ≈ 0.05 the reflected lobe is ≈ π(2α)² ≈ 0.03 sr, about 1/3 of a 0.098 sr bin. At a = 16 there are about 5 sub-bins per lobe. This is consistent with the a = 16 result and with the ~2/α² ≈ 800-direction estimate.
  - Cost [INF]: only 74 of 1,270 patches are peaked. A sender-side relation with 8 rays per sub-bin on 74 × 1,024 cells is ≈ 0.61 M rays, comparable to the current 0.65 M.
  - The "extra α⁻² in incident direction" is a **per-material** operator: 1,024² × 3 float32 ≈ 12 MB per material. It is not per-geometry state, so the DDT 38 GB precedent (per-geometry 4D state) does not apply.
  - q at sub-bin resolution can be evaluated per point at the exact light direction, which is cheap.
- R1's rejection lumps this physical refinement with the learned higher-rank op and dismisses it by assertion. That is the mirror image of the brief's "do not assume more bins solves it": it must be measured, not asserted.

**X5. The sharp ≥3-event family is relocated into Λ_θ, not addressed.**
- [FACT `pt86a.py:175-180`] e3_p means k ≥ 3 with vertex 2 (the last sender before the receiver) on material 1. So e3_p is exactly "a later flight from the sharp panel", e.g. L→D→panel→receiver.
- [MEAS, WL94B/94] The panel-mediated share of e3 is 0.755 on the CK2 panel face and 0.166 on box A. The higher-order part is 61 % of the panel-face B. G vs e3 on the panel face is 0.15–0.20 vs half-noise 0.027 (WL94C).
- In both H-A and H-B, this family is owned by Λ_θ, whose form is "a protocol-freeze decision" (rank ρ, Ft-structured or latent).
- [FACT `model86.py:83-100`] Ft is a direction-free patch-pair form factor (L9).
- [INF] Any Λ_θ whose geometry enters only through Ft cannot route the panel's directionally structured B₂ to the right receivers.
- [INF] Any rank-ρ per-patch input projection of B₂ re-imposes the L3 compression, which Stage B measured as severe for S1, on the second panel flight.
- So "removes L3/L7/L8" holds for e2 only. For e3_p the learned sharp-sender flight problem that failed in WL86–94D is renamed, not removed. Leaving the form unspecified also makes E-A3 non-falsifiable as a mechanism.

**X6. Existing evidence already predicts that the learned tail collapses to H-0, and the design does not state this prior.**
- [MEAS, WL90:172-177] The projection error at orders ≥ 2 is P2plus − EE = +0.0009 / +0.0007 / −0.0004. Explicit binned sweeps own e3 at near-oracle bias. Λ_θ's accuracy headroom over H-0 is ≈ 0.
- Cost headroom is bounded by J ≈ 3–5 sweeps (WL86B) against the generator: 11 ms per edit plus O(N²) Ft.
- The only learned advantage ever measured (WL87 source-only update, 3.5–3.9×) came from EXP's pre-fixed MIS q budget (WL87:331). H-0 built on M64(s) does not pay that budget, so the advantage disappears in H-A's own matched null.
- [INF] H-A's learned claim is predicted false by WL87, WL90 and WL86B before any training. A primary whose sole learned element is predicted null conflicts with the brief's "no recommendation merely to generate progress", unless the prediction is stated and the hypothesis is re-scoped.

**X7. The training-feasibility citation uses a norm that is blind to sharp readers.**
- "WL92A OWN global 0.092 field rel L1" is pooled. WL93C and WL93D both record that pooled table metrics and sharp readers move in opposite directions.
- [MEAS] Panel-mediated mass is ≈ 3.4 % of the loss-normalised target. Items at CK2 exposure are 4/96, and reallocation reach is ≤ 1.23–2.84× (WL94B).
- [INF] The Stage-2 smoke (128 geometries, 2,000 steps, 2 seeds) repeats the WL93A–94D contract, which produced NO_SIGNAL, INCONCLUSIVE or seed disagreement five times. It is underpowered for e3_p unless sharp-family metrics, ≥ 3 seeds and exposure are predeclared.

**X8. Overstatements to correct.**
- "Exact sender helps rendered sharp regions in every measured case." False for textured CK2 (WL93D addendum: C2_src 0.031 → 0.063; "no arm best on both seeds and both variants"). In WL94 Stage B, sender substitution leaves amplitude unchanged (±0.02). Restrict the claim to untextured sharp ROIs.
- "Rank 16 caps field L1 ≥ 0.36 / peak ≤ 0.49." These are values reached by tried local optima in the W94-weighted field metric, not a lower bound. The only proven bound is Eckart–Young flux-Frobenius 0.46–0.52. Also, svd32 gives 0.353 field L1, and only the L1 ladder gives 0.146 at rank 32.
- "Supervision did not move it off the zero-operator level." True only in the WL94D operator metric for WL94D models. WL93A TQ moved field metrics (rel L1 0.899 → 0.738, peak 0.31 → 0.48; a different norm, never evaluated in the operator metric).
- "q̂ = exact ER bin projection." q̂ is M64(s), which carries L1, L2 and L12 (Stage A §3.1).
- "Learned re-representation of K is not admissible" (O1). This is a design norm justified by duplication, not by evidence. S_geo < 1 % shows only that the learned operator did not specialise, which is physically trivial for an opaque BRDF.

**X9. The energy audit does not discriminate.** ρ_max/(1 − ρ_max) is 4.7× at S1 0.825 and 9× at the Lambert 0.9 rows (WL94 diversified set). Use the flux of explicit sweeps on the same Γ_s as the reference, computed with the correct W (not W94).

**X10. Stage-1 ladder logic defects.**
- **(a) The relation switches from R0 to C3UR at Level 1a.** The WL94D CK2 floor was measured on R0 (deployed 512 rays), so C1–C6 as defined there are not decomposed. Run a factorial on pooled R0 and on pooled C3UR, one factor at a time:
  - single vs pooled (C5);
  - M64(s) vs q (C3/C4);
  - +X1 (C1, angular);
  - sub-patch (C2);
  - sub-bin a ∈ {4, 16} (C1 resolution).
- **(b) The 1c → 2 step changes at least four things at once:** placement (sender-side relation vs light-side), sampling density (uniform-cos vs lobe), receiver incidence (bin vs exact/dense cell) and P_pts. Its attribution to "C1/C6" is not identifiable.
- **(c) The stop rule "both fail, Level 3 passes → readout (L10) or e3" is logically wrong.** All levels share the readout and the path-traced e3. A Level-2 fail with a Level-3 pass implicates Level 2's own prefix: the ¬P Γ_s family, receiver projection, photon variance or the partition.
- **(d) The admission decision rests on a single draw, while draw distributions are heavy-tailed.** [MEAS] E2 EXP SD 0.018, range 0.024–0.077; WL91 W2 deployed draw 0.053 above the 16-draw maximum. Decide on a draw quantile (e.g. ≥ 90 % of ≥ 16 draws) plus pooled bias.
- **(e) E-A2 has no frozen bound.**
- **(f) The existing `_p` split is material 1 only.** H-B's P = S1 + S2 (sphere) needs a new tally split; that is not an existing capability.

## 2. H-A (ELBR)

**Fatal flaws.** None for the learning-free prefix question. The learned part is not yet a falsifiable hypothesis (X5).

**Serious issues.**
1. The prefix is already measured to fail the design's own oracle-relative bound on E2 T1 C with R0: 0.0449 single, 0.027 pooled (X1). Only the Γ_s variant is untested, and C3UR evidence uses q, not M64(s) (X2).
2. **Internal inconsistency at the receiver.** Γ_s is described as "deposited at the receiver's incident cell", i.e. C3UR's dense 4,096-cell table (WL89:65, 71). The operator then writes E1[n,m] per bin with an M64 receiver, which re-imposes receiver L12 that C3UR avoided. This is irrelevant for Lambert receivers and material for GM1 S→S.
3. Λ_θ is unspecified exactly where the sharp ≥3 family lives (X5), and existing evidence predicts it is null against H-0 (X6).
4. **Resource claims.**
   - "No new ray tracing over ECRO" holds only if R0 is dropped and Ft is derived from Γ_s. The design does not say which.
   - The 27 MB memory figure omits an output-side per-patch basis (N × 64 × 3 × R ≈ 15.6 MB) needed to emit the 64-bin Ê≥₂. The total is ≈ ECRO's 43 MB, not 27 MB.
5. "Sender-side placement attacks the dominant error" is a variance claim, not a bias claim: converged relations share the floor (WL89:379). The design omits that C3UR is worse than R0 on LMIX R_mirror (0.023 vs 0.017, single draw, WL89:333).

**Genuinely supported.**
- [MEAS] The learned S1 sender is far from even non-negative rank-16 performance.
- [MEAS] With all else fixed, an M64 sender improves *untextured* sharp ROIs (WL93D, WL94, WL94C, WL94D).
- [FACT] Exclusive e2/e3 tallies exist.
- [MEAS] Sender-side relations cut variance (WL88/89).
- [MEAS] The GM1 explicit gap is first-gather angular projection (WL90).

**Changes required for a valid falsifiable hypothesis.**
1. Re-read the E2 evidence with matched tails (X1, X2).
2. Make the receiver consistent: dense-cell deposit, or justify bin M64.
3. Specify Λ_θ before freeze. Either:
   - (i) the second flight is explicit on Γ_s and Λ_θ owns ≥ 4 events only; or
   - (ii) a Γ_s-structured directional generator, with an explicit test that its input projection does not compress peaked-sender B₂.

   Forbid an Ft-only or global-latent form for peaked senders.
4. State the WL87/WL90/WL86B prior that Λ_θ ≈ H-0, and make "Λ_θ beats H-0 at matched cost and quality on e3_p" the primary learned falsifier.
5. Add GM1 W1/W2 (R_mirror, C_ss) and a sub-bin Level 1d (a ∈ {4, 16} on P senders, source-independent).
6. Predeclare e3_p sharp-family metrics, ≥ 3 seeds and exposure for Stage 2.

**Verdict: VIABLE_WITH_CHANGES.** This applies as a learning-free first-flight hypothesis. Its learned component is not yet a hypothesis, and existing evidence predicts it is null.

## 3. H-B (ELCF)

**Fatal flaw as written.** The receiver deposit relies on "fbar … exact outgoing-bin average at the exact incident direction (`quad87.fbar`, line 85)".
- [FACT `quad87.py:85-110`] `fbar` is a Monte Carlo histogram estimate with M MIS samples per direction (M64 uses M_dense = 16,384).
- [INF] Per-photon use costs about 3 × 10⁵ photons × 1.6 × 10⁴ samples per light for non-Lambert receivers (floor B, S1, S2). That is infeasible.
- The available capability is the dense 4,096-cell table (`quad87.py:223-231`). With it, receiver L12 is *reduced* to 1/64-bin cells, not removed. The alternative is per-photon low-M Monte Carlo, which adds variance.
- The claims "removes L7/L12 at the receiver" and "unbiased estimator of the oracle-table quantity" must be restated. The brief forbids inventing oracle capabilities.

**Serious issues.**
1. **Not a materially distinct architecture.** The learned content and ownership principles are identical to H-A. The only difference is a classical first-flight estimator: C3-style source-dependent placement + X1-style exact radiance + VNDF lobe sampling. The design itself concedes it is known technique.
2. **Direct/indirect fidelity asymmetry.** Sharp response is resolved only for *direct* incidence on P senders; C-mediated incidence on P (S→S, L→D→S) stays binned or goes to Λ_θ. This conflicts with the spirit of Architecture.md:21: "Direct and C-mediated incidence use the same intrinsic response rule … Sharp specular/glossy response demands sufficient directional information." It is an owner/contract question the design does not raise; it raises only source-dependence. It also leaves e3_p (X5) unaddressed.
3. **"Removes L5" is overstated.** A per-edit photon set is a fixed realisation, i.e. a new sampling error of the L5 class, whose variance is unmeasured. Unbiasedness also needs randomised or stratified y_j per realisation; fixed points reproduce an L2-type fixed bias.
4. **The cost estimate is not anchored to the closest measurements.**
   - [MEAS] WL87 C3 (sender-side, source-dependent, 16 × 650k rays): 0.43–0.58 s per edit.
   - [MEAS] WL91 X1: 526,467 probe + 526,467 shadow rays per light, 11–14 ms evaluation.
   - Per-photon receiver evaluation and dense-cell atomics are also uncounted.
5. **The partition P = S1 + S2 has no existing tally split** (X10f). Sphere photons still meet flat-frame receivers.

**Genuinely supported.**
- [MEAS] WL91: pooled per-ray exact sender radiance removes the first-gather bias, which supports continuous sender radiance.
- [INF] Lobe sampling plausibly addresses X1's variance mechanism.
- [FACT] The peaked rule is prospective and material-only (`rel88.py:47-59`; Π: B 4.93, S1 37.95, S2 16.13), so the P/¬P decomposition has a physical justification and is not post hoc.
- No measurement of lobe-sampled light-side deposition exists.

**Changes required.**
1. Re-scope H-B as a learning-free first-flight estimator arm (Stage-1 Level 2), not a second architecture.
2. Use the dense-cell receiver and restate what it removes.
3. Randomise and stratify y_j and report a draw distribution.
4. Anchor cost to WL87 C3 and WL91 X1.
5. Raise the direct-only sharp asymmetry as an owner decision.
6. Add the S2 tally split, or restrict P to S1.

**Verdict: NOT_VIABLE as an independent architecture hypothesis; VIABLE_WITH_CHANGES as an oracle/estimator level inside Stage 1.**

## 4. Overall verdict check

- "MULTIPLE_CANDIDATES_REQUIRE_DISCRIMINATION" overstates the breadth. The two candidates share one learned component, which is unspecified and predicted null, and differ only in a classical estimator. Under this lens the brief's "≥ 2 materially distinct hypotheses" is not met.
- **The defensible ownership statement:** "ER-owned local response in the prefix is the evidence-favoured direction for untextured sharp ROIs when other terms are fixed." This is not "justified": with an exact tail the R0 prefix is 0.045 on E2 T1 C (X1).
- **What is decidable now, learning-free:**
  - which first-flight estimator (64-bin, sub-bin, X1, light-side) meets matched bounds on GM1 S→S, E2/E5 C and CK2;
  - whether explicit e3 sweeps are already at oracle bias (WL90 suggests yes).
- **Recommended verdict: ARCHITECTURE_SELECTION_PREMATURE for the learned transport representation,** with a learning-free discrimination study. The alternative is to supply a genuinely distinct learned hypothesis, e.g. a Γ_s-structured learned resolvent against adaptive angular resolution, so that MULTIPLE_CANDIDATES has content.

## 5. Sources
- Evidence files: `C:\Users\dna10\AppData\Local\Temp\claude\462fa2b2dbb71cdd\46d65e15-e083-4057-875f-9282fb68aa5e\scratchpad\wl95\{auditA,auditB,read_code,read_canon,read_wl86_88,read_wl89_92,read_wl93,read_wl94}.md`
- Code and docs, under `c:\Projects\ReST-GI`:
  - `experiments\gm1_explicit87\quad87.py:1-38,48-110,223-242`
  - `experiments\ecro86a\pt86a.py:170-200`
  - `experiments\ecro86\model86.py:108-124`
  - `experiments\gm1_relation88\rel88.py:40-62`
  - `docs\Architecture.md:17-38`
  - `docs\worklogs\89_*.md:40-80,250-382`
