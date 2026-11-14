# WL95 adversarial critique of the ownership-track design (H-A ELBR, H-B ELCF)

**Lens:** cost, scaling, reuse and falsifiability.

**What I did.** Design-only review. I edited no repository file and ran no training, GPU or CPU computation. I read all eight evidence files in full and checked the contracts the design relies on in code and worklogs:
- `experiments/ecro86/model86.py:83-124`
- `experiments/ecro86a/pt86a.py:73-75, 160-199`
- `experiments/gm1_explicit87/quad87.py:60-110, 262-300`
- `docs/Architecture.md:13, 21`
- `docs/worklogs/89_WORKLOG_GM1_SENDER_RELATION_QUALIFICATION_AND_DISCRIMINATION.md:60-75, 340-381`
- `docs/worklogs/90_…:58-68, 176`

Labels: [FACT] verified in code or docs · [MEAS] measured value with its source · [INF] reasoned inference · [HYP] untested.

---

## 0. Findings that apply to both hypotheses

### X1. The sharp, panel-mediated ≥3-event family is still owned by the learned Λ_θ. The design moves the learned sharp flight up one order; it does not remove it.
- [FACT] `pt86a.py:175-180` defines e3_p as k ≥ 3 events whose vertex adjacent to the receiver is on material 1. `v2p` is set at k = 2 and reused for every k ≥ 3. So e3_p is exactly the light → … → **panel → receiver** family: the panel is the last sender (broad → narrow → broad).
- [MEAS] This family is a large share of e3 where the sharp failure lives:
  - CK2 panel face: e3_p = 0.755 of box e3, against e2_p = 0.580 of box e2.
  - CK2 box A: e3_p = 0.166 (WL94B).
  - E2 T1 region C reads 0.46 of its value from 3–5 events (WL92).
- [INF] In both H-A and H-B the operator `B̂₃₊ = M64ᵀ·Λ_θ(G)[B₂]` must produce the sharp panel → box flight of the panel's B₂ inside a learned geometry-conditioned operator. This is the same mechanism that delivered 1–18 % of the E2 family in ECRO (WL86A Amendment 1).
- [MEAS] Prior evidence on the learned owner of this family is negative:
  - The learned global term vs e3 on the CK2 panel face is rel L1 0.15–0.20, against half-noise 0.027 (WL94C).
  - The LSPOT regression was attributed to the global term (X_L3 0.71 / 0.82, WL92B).
  - Panel-mediated content is ~3.4 % of the training loss; only 4 / 96 items reach CK2's level (WL94B).
  - Neither item-level resampling (G_EXPOSURE 1.23 < 1.25) nor query-level reweighting (WL94C seed disagreement) moved it reliably.
- So "removes the L8 compensation channel" is true only for e2. For the dominant sharp-mediated e3 the design keeps a learned flight with a known coverage deficit.
- The design lists this only as main risk 3 / inconclusive. It is the central untested claim and must be the primary measured quantity, on the CK2 panel face and the LSPOT region.

### X2. The learned component has essentially no evidenced niche against the learning-free comparator (H-0). Its form is also unspecified, so its learned claim cannot currently be falsified.

**Accuracy headroom.**
- [MEAS] WL90:176: projection at orders ≥ 2 only (P2plus − EE) contributes +0.0009 / +0.0007 / −0.0004 E_H (GM1 R_mirror). Compounding is ≤ 9 % of the gap.
- [INF] Explicit sweeps for e3 on the same binned relation are therefore already near the oracle in the one family where this was measured. A learned Λ_θ has almost no accuracy headroom over explicit sweeps on e3 unless the scene86 / CK2 compounding differs (unmeasured).

**Cost headroom.**
- [MEAS] WL89:350-352: explicit 4-order application on C3UR costs 11.6 ms at N = 1,270, and C3UR apply scales as N^0.38.
- [MEAS] The ECRO generator alone costs 11 ms per edit, plus about 1.5 ms apply (WL86:214).
- [INF] Per edit, Λ_θ regeneration ≈ explicit sweeps. The only remaining cost niche is the source-only update: reused factors (~1.5 ms apply) vs 3 extra explicit sweeps (≈ 8–9 ms, extrapolated from 11.6 ms for 4 orders).
- So the learned advantage is bounded at single-digit ms per source change.
- WL87's 3.5–3.9× source-update advantage was against EXP's MIS q-table (26 ms, WL89:352). H-A replaces that q-table with an M64 mat-vec (q̂ = Σ s·M64), which removes the comparator cost the advantage was measured against [INF].

**Form not specified.**
- [FACT] The design leaves Λ_θ's internal factorisation (ρ; Ft-structured vs global latent) as "a protocol-freeze decision". It excludes "a geometry-blind global latent alone".
- Its memory figure (≈ 27 MB for ĉ + Q + P) nevertheless presupposes the ECRO global-latent form. That is internally inconsistent.
- The underlying dilemma:
  - A global-latent Λ_θ driven by B₂ instead of u still has no flight geometry. Changing the driver does not fix L8, which is "no flight geometry" (auditA L8).
  - An Ft-structured Λ_θ is a learned re-implementation of K·C sweeps on the same relation. That is either H-0 with learned coefficients, or a learned flight again (X1).
- Until this is fixed, E-A3 / Stage 2 cannot be falsified: a failure can always be attributed to the factorisation choice.

### X3. The H-A vs H-B hinge is largely an owner threshold decision, and the chosen workload omits the case where the answer is already known.
- [MEAS] The converged binned floor:
  - GM1 R_mirror: 0.038 against the oracle 0.014 (WL89; WL90 attributes 97–102 % of the gap to first-gather angular projection). The PT floor is 0.024 and C3UR's 0.042 (WL89:364).
  - Scene86 E2 T1 C: ENCB pooled 0.027 against ORA2 0.0127; C3UR single draw 0.026 (WL88 Task A, exact-q sender).
- [INF] Under the proposed E-A1 rule (≤ 0.06 and ≤ oracle + max(2·noise, 0.01)):
  - The scene86 E2 floor probably passes (bound ≈ 0.031 with noise ≈ 0.009).
  - The GM1 R_mirror floor fails (0.038 > ≈ 0.024).
- The smallest workload is scene86-only and excludes GM1 W1 / W2. For those, path-traced oracle tables and a converged-floor measurement already exist. Their exclusion makes the outcome depend on workload choice.
- Required: include GM1 W1 / W2 R_mirror as a known-floor positive control for the kill rule, and freeze the threshold before results. The design marks its thresholds as owner proposals, so as written there is no frozen kill condition.

### X4. Cost was not checked against the practical full-recompute bar.
- [MEAS] The in-scene bar is the frozen-ER path tracer:
  - PT per edit 35–36 ms, 1.80 ms per spp (WL86/87).
  - Binned methods are bias-floored below PT's attainable quality: E2 C bias floor 0.073 vs PT 0.040 at 4,096 spp (WL86); R_mirror floor 0.042 vs 0.024 (WL89).
  - [INF] At quality tiers below the binned floor, full recomputation wins outright.
- Stage 1 and Stage 2 report only "per-level cost". Required: a predeclared quality-matched edit-to-first-frame and sustained-frame comparison against PT / PT_RR on the same tiers (WL86/87/89 ladders). The tier set must include tiers below the binned floor.
- WL67/69/70 Cycles numbers come from different workloads (media slabs; seconds scale) and must not be cited as the bar here. External Neural Full Recompute (8DNA/RNA) remains UNMEASURED and gated by the mandatory baseline gate (AGENTS.md; BASELINE_ROLES §gate). The design states this correctly.

### X5. The O(N²) scaling claim is partly an implementation artefact. The real edit-locality killer is structural.
- [FACT] `model86.py:83-89` scatters the 512 rays per row into a dense N×N tensor. Ft has at most K = 512 non-zeros per row, so it is O(N·K) sparse by construction.
- [FACT] `model86.py:98-99`: every round adds a global mean `glb(t).mean(0)`.
- [INF] Memory and compute as coded: 400 MB at N = 10⁴, 40 GB at N = 10⁵ (infeasible). A sparse Ft would remove that. The global mean term plus 3-hop message passing makes every edit dirty all factors regardless, which is structural.
- [MEAS] Learning-free C3UR build scales as N^1.03 (WL89:368).
- Both hypotheses therefore scale worse in their learned part than their explicit comparator, unless Λ_θ drops the global mean, which would reopen L8.
- The design should replace "inherited O(N²) risk" with:
  - a synthetic learning-free scaling probe: random-weight generator vs explicit sweeps at N ∈ {1.3k, 5k, 10k, 20k};
  - an explicit statement that rule reuse gives no edit locality (WL86: selective vs full state is 20.8 vs 28.1 ms).

### X6. Unmatched pairs in §2.1 point 3.
- The table puts ENCB (M64(s) sender, R0 single draw, 0.0449) next to C3 / C3U / C3UR (0.024 / 0.023 / 0.026). The C3 family used WL87's exact MIS q, not M64(s) (read_wl86_88 §4; quad87 `qtable`).
- So placement and sender type are confounded in that table. WL87's ENC vs EXP difference (0.054 vs 0.049) suggests the sender effect is small on E2, but no M64(s)-on-C3UR number exists.
- The WL92A OWN global term "val 0.092 field rel L1" is a pooled whole-table metric. WL93C/93D show it is insensitive to sharp readers. It is not evidence of feasibility on the sharp-mediated e3 family.

---

## 1. H-A (ELBR): binned sender-side relation, ER at both ends, learned resolvent on e3

### Fatal flaws, as a learned-architecture primary
1. **The learned claim cannot be falsified as specified (X2).** The Λ_θ form is not frozen; the memory and cost figures contradict the stated exclusion; and both admissible forms collapse into either L8 (no flight geometry) or a learned K·C sweep. A Stage 2 failure at smoke scale would be uninterpretable. The WL93–94D smokes all ended NO_SIGNAL / INCONCLUSIVE / seed-disagreement, and E-A3 has no representability bound such as WL92-style per-geometry free-factor fits of the Λ_θ family.
2. **The evidence predicts no headroom (X2).** Explicit e3 sweeps are near the oracle where measured (WL90 P2plus ≤ 0.0009). Per-edit cost is at parity (11.6 vs ~12.5 ms). The source-update advantage shrinks to a few ms once q̂ = M64·s.
   - [INF] H-A is predicted to reduce to H-0, which Architecture.md:13 (human, 2026-11-13) keeps as a comparator, "not the global method to be published".
   - Recommending it as primary on these grounds is close to "recommendation merely to generate progress".

### Serious issues
- **X1.** The panel-mediated e3_p flight stays learned, with a known training-coverage deficit (WL94B 3.4 %; CK2 out-of-distribution, 4 / 96 items).
- **Inconsistency with its own ownership logic.**
  - O3 assigns the learned operator everything beyond the first flight. But the second flight, Flight(B₂ at the panel), is as cheap to compute explicitly as the first (≈ 3 ms per sweep [INF from WL89 11.6 ms / 4]) and is the sharp one.
  - Following O1–O3 consistently pushes the boundary to e4+ (B₃ explicit), shrinking the learned share further. Its size is unmeasured.
  - The design does not justify stopping at e3.
- **Train/deploy driver bias.** Λ_θ is trained to map the binned B₂, which carries the floor bias and a fixed single-draw relation, to the true e3. So it learns geometry-conditioned corrections to the prefix bias inside e3.
  - That is a hidden compensation channel. It is milder than L8 but real, and it is not tested on held-out geometry or on a different relation draw.
  - Required control: evaluate Λ_θ with a re-drawn Γ_s offset seed.
- **The receiver is inconsistent with C3UR.** [FACT] WL89:65, 71: C3UR deposits into the receiver's WL87 **dense incident cell** and scatters with a dense-cell contraction. H-A writes `B2 = Σ_m E1[n,m]·M64[m,d]` on 64-bin E1. That reintroduces L12 at the receiver, so "C3UR-type" does not hold as written.
- **Energy audit is vacuous.** With S1 ρ_max = 0.825, the bound ρ/(1−ρ) ≈ 4.7× B₂ flux would not detect plausible failures. Replace it with a per-family signed flux ratio against tallied e3 at the 4,096-spp CK2 / E2 fields.
- **Known C3UR weaknesses go uncited** (WL89):
  - worse than R0 on LMIX89 R_mirror (0.023 vs 0.017, :333);
  - lower R_mirror edit slope (0.86 vs 1.12, :326);
  - error concentrated at room corners (:381).
  The supporting-evidence list is one-sided.

### What is genuinely supported
- O1 as an ownership conclusion: the S1 local operator should not be re-learned.
  - [MEAS] S_geo < 1 %.
  - [MEAS] Learned operator at the zero-operator level (0.96–1.04) vs non-negative rank-16 0.38–0.60 in the matched norm (auditB).
  - [MEAS] The exact sender improves sharp renders in every inspected case (WL93C ORA, WL93D C2_src, WL94 C2_srcS1), with co-adaptation explaining the pooled-table regressions.
- Explicit ownership of visibility (WL52).
- Sender-side placement reduces relation variance at equal rays (WL88 Task A +0.050 / +0.053; WL89 pass rate 100 % vs 81 %).
- Cost accounting is honest: no hidden ray tracing, Γ_s at 650,240 rays, ~14 ms warm update (WL89:124), 20.8 MB.

### Changes needed for a valid falsifiable hypothesis
1. Freeze one Λ_θ form, with its memory and cost derived from that form.
2. State a learning-unique claim with predicted headroom. Candidates:
   - (a) variance reduction relative to single-draw explicit sweeps;
   - (b) source-update latency;
   - (c) closure of the binned bias in e3.
   Quantify the expected effect from WL89/90 before the run.
3. Either make the e3_p last flight explicit (learned part owns e4+ only), or claim Λ_θ carries it and gate on the CK2 panel face e3_p and LSPOT explicitly.
4. Add a representability bound before training: per-geometry free fit of the Λ_θ family against tallied e3.
5. Compare against explicit sweeps at a **matched ray and wall budget**, including C3UR16 / pooled.
6. Use the dense-cell receiver if C3UR-equivalence is claimed.

### Verdict: NOT_VIABLE as specified (as the primary learned-architecture hypothesis)
- The learning-free part (ER at both ends + sender-side binned relation) is a valid diagnostic and comparator: Stage 1 levels 1 / 1a.
- The learned part is unspecified and predicted, from existing measurements, to have no accuracy or cost niche over H-0.
- It becomes VIABLE_WITH_CHANGES only if changes 1–6 are made.

---

## 2. H-B (ELCF): light-side lobe-sampled first flight for peaked senders, same learned Λ_θ

### Fatal flaws
None, as a learning-free estimator arm. As an independent learned-architecture hypothesis it is not distinct from H-A: it shares Λ_θ and therefore all of X1 / X2. Its distinctive component is explicit photon tracing (known technique; the design discloses this).

### Serious issues
- **Hidden per-photon cost: "exact fbar" at the receiver.**
  - [FACT] `quad87.fbar` (`quad87.py:85-110`) is an M-sample MIS Monte Carlo per direction (EXP used 65,536). M64 used 16,384 per cell.
  - The design deposits `Φ·fbar_{k_n}(−ω_o)[d]` per photon. For non-Lambert receivers (floor B, S2, S1 → S1) that is O(M) closure evaluations per landed photon:
    - 3×10⁵ photons × 10³–10⁴ samples ≈ 3×10⁸–3×10⁹ evaluations per light [INF];
    - none of this is in the 12–15 ms estimate.
  - The feasible alternative is the existing dense-cell table (64 sub-cells per bin, as C3UR uses). It reduces L12 rather than removing it; WL90's fixed-grid 16-sub-bin residual of +0.0007 / +0.0012 suggests that is adequate. The claim "removes L12 at the receiver" must then be withdrawn and the cost counted.
- **Cost shift, quantified.**
  - The ray-count analogy (12–15 ms per light) is plausible: [MEAS] WL91's exact per-ray L₁ evaluation stays at 11–14 ms from 0.26 M to 2.1 M landed rays, which suggests launch-bound cost.
  - [INF] Per edit with 4 lights: ~63–70 ms (H-A sum) + 50–60 ms ≈ 115–130 ms, against PT 36 ms per edit. Any advantage must come from per-frame cost at matched quality. That remains to be shown.
  - [MEAS] The closest precedent, C3URX1 (exact first gather), removed the bias (pooled ≈ oracle) but **failed quality-matched utility** because of single-draw variance: W2 T1 0.74 s vs 0.24 s at the 0.06 tier, 3.1× slower (WL91).
  - H-B's claim that lobe sampling cures that variance is [HYP]. It needs ≥ 16 seeds and a predeclared time-to-quality kill rule against both PT and C3UR. The design has only "report against PT at matched quality", with no threshold.
- **Sampling design at the motivating case.** In CK2 the light is 5 cm from the panel at grazing incidence, and panel irradiance varies 307×.
  - Uniform-area P_pts points per patch puts most samples where irradiance is low. Light-side emission toward the panel, or irradiance-proportional point sampling, would be the variance-appropriate estimator for a delta light.
  - As written, Level 2 can fail on variance for reasons unrelated to the representation question.
- **Reuse and source economics.**
  - The sharp first flight has no reusable representation: it is re-simulated for every light change and geometry edit. "Without retraining" holds trivially because nothing there is learned.
  - It removes the only measured learned advantage (source-only update) for the peaked family. With moving lights it degenerates to a per-frame light tracer (the design discloses this).
  - Photon RNG policy is not declared: fixed per-photon streams keyed by id, for a selective = full equality like WL89:70, or fresh draws, which cause edit-to-edit flicker.
- **Owner-level conflicts beyond the one flagged.**
  - [FACT] Architecture.md:13 (human, 2026-11-13): "Later X1-style first-gather findings constrain information preservation but do not justify converting the project to explicit GI."
  - H-B makes the sharp first flight explicit and per-source, and leaves learning only in e3. This is a direct tension with that decision, beyond the source-dependence question the design raises.
  - Architecture.md:21 (no direct/indirect response branch) is satisfied in letter, because ER is the rule in both estimators. But the estimator branch on directly lit peaked senders is a direct/indirect asymmetry the owner must rule on.
- **e3 owner = H-P's refuted point.** WL86B refuted H-P partly for "broad → narrow → broad links without visibility". In H-B that family (e3_p) goes to Λ_θ, so H-B inherits the refuted point unless Λ_θ has flight geometry (X1, X2).

### What is genuinely supported
- The estimator is well defined and dimensionally correct.
  - [INF, checked] Φ = E_ℓ·(A/P_pts)·f·cosθ_o/(m·p) deposited as Φ·fbar/A_n is an unbiased estimator of the patch × bin-averaged e2 table that `Ctx.field` defines, for flat receivers.
- Exclusive ownership by sender material is testable (e2_P + e2_¬P vs the e2 tally).
- The peaked rule is prospective and material-only (`rel88.peaked`, Π > 8; B 4.93, S1 37.95, S2 16.13). That satisfies WL86B's admissibility criterion of a class rule fixed beforehand.
- Bias removal by exact per-ray first-flight radiance is measured (WL91 pooled within −0.0025…+0.0035 of the oracle).

### Changes needed
1. Replace exact fbar with the dense-cell table (or count the MC cost explicitly) and restate L12 as reduced, not removed.
2. Use irradiance-proportional or light-emission sampling for delta lights.
3. Declare photon RNG keying and the selective = full contract.
4. Predeclare a time-to-quality kill rule against PT and C3UR at matched tiers, ≥ 16 seeds.
5. Treat it as an estimator arm (Level 2), not a separate learned-architecture hypothesis.
6. Obtain the owner ruling on Architecture.md:13 before any adoption path.

### Verdict: VIABLE_WITH_CHANGES, as a learning-free first-flight estimator / diagnostic only
- It is NOT a distinct learned-architecture candidate. Its learned part is H-A's Λ_θ, with the same X1 / X2 defects.

---

## 3. Falsification experiment (§6): defects

1. **The ladder does not isolate C1–C6 as claimed.**
   - 1 → 1a (R0 → C3UR) changes placement (C5) and the cosine-weighting scheme (C6: receiver-side bin-constant read with the ray's own z, vs sender-side cos-weighted emission into dense cells) together. It also switches the receiver from 64-bin to dense-cell if C3UR is used as implemented.
   - 1a → 1b (M64(s) → exact q): [FACT] `quad87.qtable` (`:268-282`) uses P stratified points per patch; WL87 used P = 16. So 1b removes L1 and L12 but **not** L2 unless P ≫ 16, and it changes the point pattern from fixed-shared to stratified. "+ L1, L2, L12" is wrong.
   - 1c → 2 changes estimator class, P_pts and m together.
   - Fix: a factorial or one-factor set:
     - C4: s with 16 vs 1,024 points at fixed relation and sender;
     - C6: receiver-side relation with per-ray exact cosine vs bin-centre cosine;
     - C5: pooled vs single at fixed relation type;
     - C3: M64(s) vs q at P = 16 and at P = 1,024.
2. **Decision-rule logic error.** "Both fail while Level 3 passes → attribute to the readout (L10) or e3."
   - Levels 1–3 share the same nearest-patch readout and the same path-traced e3 tail.
   - A Level-3 pass with a Level-2 failure therefore localises the residual to the **e2 table estimator**, not to the readout or e3.
3. **Single draw vs pooled is not separated in the kill rule.** E-A1 uses the "Level 1a single draw" without saying whether that means the deployed draw or a pass fraction over ≥ 16 draws. WL87 showed the deployed draw in the bad tail.
   - Bias (pooled) and variance (pass fraction) need separate frozen rules. Otherwise a single-draw failure invites a denser-relation rescue, which is the no-rescue issue.
4. **Compute matching is stated and then waived** ("matched in rays per level, reported"). Level 2 (~3×10⁵ photons per light, plus fbar) vs Level 1a (6.5×10⁵ geometry-only rays, shared across lights) is not matched.
   - Required: C3UR16 / pooled arms at Level 2's ray and wall budget. A Level-2 win then cannot be just a larger budget.
5. **No frozen thresholds and no cost kill.** Thresholds are "proposals for the owner", and H-B's cost has no threshold. As written, the experiment has no unambiguous kill condition.
6. **Workload risks.**
   - GM1 W1 / W2 are missing as a known-floor positive control (X3).
   - The held-out sharp edit has no fallback. Precedents: WL93D E6 admitted 0 / 3 candidates; WL88 ended STOPPED_WORKLOAD_UNQUALIFIED.
   - CK2 is post hoc and tuned on probe renders p01–p05. It may serve as a stress case but not as the sole held-out evidence.
   - Path-traced e2 / e3 tables for CK2 come from `field_src(split=True)` (`CKF_none`). An ORA-style CK2 render must be produced and its noise bound reported.
7. **Stage 2 is asymmetric.** At the WL93 smoke contract (128 geometries, 2,000 steps), a pass is informative but a failure is not (precedents WL93A/C, WL94C/D).
   - Required: a frozen Λ_θ form; a free-fit representability bound; a training budget equal to the intended deployment; held-out geometry with a re-drawn relation seed; and gates on the CK2 panel face e3_p, LSPOT and E2 C, not on pooled field rel L1.
8. **Missing measurements.** A PT / PT_RR time-to-quality arm (X4) and a learning-free scaling probe (X5).

What is sound in §6:
- Common random numbers.
- Correct W instead of the an94.W 8×8 layout.
- The box A mask instead of the 1130:1238 slice.
- 4,096-spp half-split noise.
- The Level 0′ (X_L3) stop rule.
- All required oracles exist or are buildable from existing code (ENCB / X_M12, X_Q, cand91, X_L2 / X_L3), with no invented oracle capability.

---

## 4. The design's overall verdict

The design reports **MULTIPLE_CANDIDATES_REQUIRE_DISCRIMINATION**. I find **ARCHITECTURE_SELECTION_PREMATURE** better supported:
- H-A and H-B share the same unspecified learned component. They differ only in a learning-free first-flight estimator, so the proposed discrimination selects an estimator, not a learned representation.
- The hinge is partly normative (X3) and partly already measured. The binned floor is known on GM1 and E2 (WL89/90/92). Bias removal by exact per-ray first-flight radiance is known (WL91). Only the variance and cost of lobe-sampled light-side deposition are new.
- Neither hypothesis states a learning-unique claim with predicted headroom (X2), and the owner decision at Architecture.md:13 bars explicit-GI conversion as the method.

What survives as evidence-backed [MEAS]: the ownership conclusion that ER, not a learned rank-16 bin operator, should own local angular response (S_geo < 1 %; learned at the zero-operator level; matched-norm non-negative rank-16 at 0.36–0.60; rank ~32 needed for field L1 ≤ 0.15 at peak ≥ 0.7). This is a constraint, not a demonstrated fix. The learning-free Stage 1 ladder, corrected per §3, is a valid next diagnostic.

---

## 5. Sources checked this session
- `c:\Projects\ReST-GI\experiments\ecro86\model86.py:83-124`: dense Ft scattered from ≤ 512 rays per row; global mean at :99; global term driven by u at :122-123.
- `c:\Projects\ReST-GI\experiments\ecro86a\pt86a.py:73-75, 175-199`: e3_p = last sender before the receiver on the panel.
- `c:\Projects\ReST-GI\experiments\gm1_explicit87\quad87.py:68-110`: fbar is an M-sample MIS Monte Carlo per direction.
- `c:\Projects\ReST-GI\experiments\gm1_explicit87\quad87.py:268-282`: qtable uses P stratified points.
- `c:\Projects\ReST-GI\docs\Architecture.md:13, 21`.
- `c:\Projects\ReST-GI\docs\worklogs\89_WORKLOG_GM1_SENDER_RELATION_QUALIFICATION_AND_DISCRIMINATION.md:65-71` (C3UR dense-cell deposition), :124, :326, :333, :350-368, :381.
- `c:\Projects\ReST-GI\docs\worklogs\90_WORKLOG_GM1_INTERMEDIATE_PROJECTION_GAP_ATTRIBUTION.md:176` (P2plus).
- Evidence files in `C:\Users\dna10\AppData\Local\Temp\claude\462fa2b2dbb71cdd\46d65e15-e083-4057-875f-9282fb68aa5e\scratchpad\wl95\`: auditA.md, auditB.md, read_code.md, read_canon.md, read_wl86_88.md, read_wl89_92.md, read_wl93.md, read_wl94.md.