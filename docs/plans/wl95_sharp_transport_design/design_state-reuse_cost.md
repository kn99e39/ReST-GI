# WL95 adversarial critique of the "state-reuse" design (H1 DSR, H2 ALP). Lens: cost, scaling, reuse, falsifiability

I edited no repository file, trained nothing and ran no GPU job. I re-read code to check the contracts the design relies on. Labels: [FACT] verified in code or document · [MEAS] measured value with its source · [INF] reasoned inference · [HYP] untested.

## 0. Bottom line

- **The design's diagnosis is mostly sound and honestly labelled.**
  - It identifies L3 (the learned local operator) as the dominant learned defect.
  - It treats the CK2 "flight floor" as confounded.
  - It flags the defects in W, in `delivery_box` and in the 9-patch correlation.
  - It declares, rather than hides, that H1's propagation is classical ray tracing.
- **Both hypotheses fail the brief's central question as stated.** The brief asks for "reusable learned light transport … without requiring full transport recomputation".
  - H1 and H2 remove learning from transport entirely. Their only learned object is the frozen ER, which already exists.
  - Both recompute every order and B globally after every edit and every source change.
  - Canonically, this class (EXP / C3UR / C3URX1, WL86B H-0) is a comparator and "not the global method to be published" (`docs/Architecture.md:13`; `docs/RESEARCH_ROADMAP.md:128`). That owner ruling has not changed: the open decision is at `docs/plans/gm1_architecture_review/60_coordinator_synthesis.md:94-100`.
  - The design's fallback ("if the owner rules … then ARCHITECTURE_SELECTION_PREMATURE") is therefore the verdict that applies now, not a contingency.
- **H1 is not materially new.** By the designer's own description it is C3URX1 (WL91) plus three changes: shared-origin shadowing, deferred outgoing for orders ≥ 2, and optional material-only densification. None of these targets the measured failure of C3URX1, which was single-draw variance and quality-matched utility (WL91:151-156, 310-317).
  - Shared origins and fewer shadow rays are cost changes, and exact evaluation was already not the cost bottleneck (11–14 ms at 0.26–2.1 M rays [MEAS WL91:256, 304]).
  - Deferred outgoing for orders ≥ 2 targets a term WL90 measured at ≤ 0.0009.
  - Only densification addresses variance.
  - Under the AGENTS.md batch policy this is a dependent evidence-gap follow-up of WL91 (suffix class), not a regular architecture hypothesis.
- **H2 is an estimator variant of H1 with a quadratic pair table.** Its scalar pair visibility drops the visibility–lobe correlation at contacts and visibility transitions, the WL52 failure locus. Its per-light polygon integrals make it source-dependent, so it loses the only learned advantage ECRO has measured: source-only update, 3.5–3.9× (WL87:331).
- **The falsification plan (§6) is not yet well defined.** It has:
  - no economic kill condition;
  - no C3UR, C3URX1, EXP or quality-matched PT arm, although these are the binding baselines;
  - an acceptance term (the CK2 table oracle) that does not exist (§7 item 3);
  - an O2 description that wrongly claims removal of L2;
  - an O1 that conflates two different WL94-series quantities;
  - a CK2 primary workload that is one post hoc light, 3–9 patches, never admitted by a qualification rule.
- **Recommended overall verdict for this design: ARCHITECTURE_SELECTION_PREMATURE** for a learned transport architecture. The proposed smoke is worth keeping, reclassified as a learning-free decomposition of the open CK2 floor (auditA §3) with H1 as an explicit-comparator arm.

## 1. Verification of the contracts the design relies on

| Design claim | Check | Status |
|---|---|---|
| C3UR rays "cosine-stratified per outgoing bin" (`rel89.py:53-73`) | `_trace`: one ray per (patch, bin, κ_s). Each ray gets a **fresh PCG32 origin** `patch_point(pid, rng, rng)` and a cosine direction in bin `bn` (`experiments/gm1_relation89/rel89.py:57-73`). The receiver stores a dense 8×8 incident **cell** `Q.cell_of(...)` (`rel89.py:74-75`; `SUB = 8`, `experiments/gm1_explicit87/quad87.py:29`). | [FACT]. C3UR is already sender-anchored with per-ray origins. H1's "anchored at y_j" is a *reduction* to P origins per patch, not a new placement. H1 also keeps only the receiver bin m, which drops C3UR's dense-cell receiver incidence. |
| X1 needs probe + shadow rays per landed ray | `cand91.py:51-63`: probe from `seg_origin + 2 mm·v` towards −v. `radiance()` traces order 1 per landed ray (`:79-80`). `source_eval_rays` = n probes + n·L shadow (`:106`). | [FACT]. H1 avoids the probe by reconstructing the surface point from `patch_point` (as `quad87.qtable` does via `point_frames`, `quad87.py:277-279`). It shares one shadow ray per (y_j, light). The saving is real but small in time ([MEAS] WL91: exact evaluation 11–14 ms independent of ray count). |
| O2 = `quad87.qtable` "removes L1, L2, L12" | `qtable` uses P = 16 stratified-jittered points per patch (`quad87.py:273-277`; protocol `"P": 16`, `experiments/gm1_explicit87/protocol/gm1_explicit87.json:41`). It MIS-samples ER about each point's true light direction, then bins into patch × outgoing bin (`:309-330`). | **Partly wrong.** L1 and L12 are removed. L2 is *changed*, from the fixed shared 16-point pattern (`pt86.py:114-117, 378-402`) to a stratified jittered 16-point patch mean, not removed. L4 (spatial and angular) is kept. |
| Dirty test and selective = full (`ecro86.py:57-78`, `:161-168`) | OBB segment test with origin-on-moved-box, plus the bit-equality test against a full rebuild. | [FACT]. The template transfers to H1's relation **only if the y_j pattern and ray offsets are fixed per patch**. That same fixed pattern is the source of L2 / L5 fixed-pattern bias (§2.2). |
| Source pass "N·P shadow rays per light" | `pt86.source`: N·PSRC points, one shadow ray each per light; the pattern is shared by all patches (`pt86.py:115-117, 378-402`). | [FACT]. If H1 reuses these points as its y_j, H1 inherits L2 unchanged. |
| gap90 sub-bin operator X(a, s) available for O3-L4 on CK2 | `gap90.py` imports `fx90` (GM1 fixtures), `rel88` and `rel89` (`experiments/gm1_gap90/gap90.py:27-33`). Neither gap90 nor cand91 contains a CK2 hook. | [FACT]. O3-L4, O3-POOL and O2x on CK2 need new fixture plumbing. They are not existing oracle capabilities, so the brief's "do not invent unavailable oracle capabilities" applies. It is feasible [INF], but it must be declared as new code. |

## 2. H1, DSR (Deferred-Sender Relation)

### 2.1 Fatal flaws (with respect to the brief)

**F1. It is not a hypothesis about reusable learned transport.**
- The learned content is ER only. ER is frozen, already qualified (WL83/84) and already used by EXP, C3UR and C3URX1.
- "Removes L3, L7, L8 by construction" is true only because learning is deleted. Under RESEARCH_CENTRIC_TOPIC.md:11, a "BSDF surrogate … alone does not count".
- Architecture.md:13 assigns this class to comparators.
- It can be a candidate only after an explicit owner ruling. Until then the hypothesis is a comparator improvement.
- Fix: say so in §5 and §8 and reclassify (see the change list in §2.4).

**F2. Full transport recomputation per edit and per source.**
- "Orders and B: recomputed globally (J sweeps)", with full source evaluation per light change.
- That is exactly the recomputation the central question asks to avoid.
- [INF] The only edit-local part is the relation re-trace. In ECRO that was worth 5–7 ms of a 76–78 ms edit [MEAS WL86:214]; in C3URX1 the edit without render is 69.4 ms [MEAS WL91:229].
- So H1 offers no reuse property that EXP / C3UR / C3URX1 lack.

### 2.2 Serious issues

**S1. The variance mechanism that failed in WL91 is untouched, and probably worse.**
- [MEAS] C3URX1 pooled equals the oracle in 6/6 conditions, but:
  - its sampling term is 0.005–0.014 versus 0.0005–0.005 for C3UR;
  - the W2 T1 deployed draw is 0.053, above its own 16-draw maximum;
  - it is 3.1× slower to the R_mirror tier (WL91:151-156, 173-179, 310-317).
- H1 keeps the same source-independent direction sampling of a sharp lobe.
- [INF] Shared origins *reduce* the number of independent spatial samples of the sender from 512 to P = 16 per patch, and correlate the K_j rays of each y_j. Under CK2's 307× irradiance gradient, with the footprint fed by about 10 cm strips of 2 panel patches (auditA §3.2), this adds spatial variance.
- The design's own risk (1) admits this. Nothing in H1 counteracts it except densification.

**S2. The per-sample-point ray-count claim is arithmetically inconsistent.**
- "A lobe of about 0.07 sr (≈ 0.7 bin) receives O(κ_s) rays per sample point" contradicts the stated budget: 512 rays per patch, P = 16, so K_j = 32 rays per y_j. That cannot even stratify 64 bins (0.5 rays per bin per point).
- [INF] The expected lobe hits are about 0.7 × 8 ≈ 5.6 **per patch**, about 0.35 per sample point.
- [INF] On CK2 roughly 11 lobe rays from 2 panel patches would feed the 3 footprint receivers, about 3–4 energy-carrying rays per receiver patch. Per-draw relative error would be of order 50 %, against an acceptance of box A rel L1 ≤ 0.12.
- Densification ×4 gives about 25 %; ×16 gives about 13 %.
- So the kill condition "≤ 4× peaked density" is very likely to trigger [INF]. The design should state this prior explicitly rather than present ×4 as a plausible pass.
- Rays per lobe scale ∝ α², so the cost of passing scales ∝ α⁻². This is the DDT-class scaling the design rejects elsewhere (WL84A, `60_coordinator_synthesis.md:42`).

**S3. L2 is retained but omitted from the loss accounting.**
- With P = 16 points per patch (fixed, if they come from the existing source pass), H1 keeps the 16-point spatial projection of the source.
- auditA C4 lists this as **open** on CK2 precisely because of the 307× gradient.
- Consequence: an O3-POOL failure could be L2, yet the design's stop rule attributes such a failure to "receiver spatial structure or reference definition" (§6). That inference is invalid unless pooling also redraws y_j and P ∈ {16, 64, 256} is a factor of O3-POOL, not only of O4.

**S4. Deferred outgoing for orders ≥ 2 has no measured benefit and a hidden table.**
- WL90 P2plus is ≤ 0.0009 [MEAS WL90:172-177].
- Evaluating `Σ_i E_i(y)·f̄_m(bin_i → v_k)` at continuous v_k needs one of two things:
  - (a) a per-ray 64 × 3 weight table: 650k × 64 × 3 × 4 B ≈ 0.5 GB, absent from the 20–40 MB estimate;
  - (b) a fine-outgoing material table, which is outgoing tabulation and contradicts "never tabulated for transport";
  - (c) per-ray ER-averaged evaluation over 64 incident bins per order.
- It also replaces C3UR's 8×8 dense receiver cells with a single bin m and a receiver cosine. That is a regression for glossy receivers (C_ss, floor B, S2) relative to the closest baseline.
- Fix: keep C3UR's existing `apply` for orders ≥ 2. Make deferral a first-gather-only change, so H1 differs from C3URX1 in one factor.

**S5. Cost moves into ray tracing, and the baselines are missing.** The design declares this honestly in §4, but §6 contains no:
- C3UR arm;
- C3URX1 arm;
- EXP arm;
- quality-matched frozen-ER PT arm.

The measured reference points the design should use:
- [MEAS WL87:266-290] Per edit: PT 36 ms, EXP 89–95 ms. Edit-to-first-frame at matched 0.03 quality: EXP 4.7–5.2× versus PT.
- [MEAS WL91:228-231] At a fixed 64-spp frame:

  | Lifecycle step | C3URX1 | C3UR | PT |
  |---|---|---|---|
  | Edit to frame | 124.7 ms | 138.2 ms | 250.1 ms |
  | Source update to frame | 75.8 ms | 92.2 ms | 213.1 ms |

- [MEAS WL67/69/70] The binding external bar has been quality-matched path-traced re-render.
- The acceptance sketch's "per-edit cost ≤ 2× EXP" therefore admits roughly 5× PT's per-edit wall time. That is not a usefulness criterion.

**S6. Reuse claims beyond rigid edits are asserted, not derived.**
- "Deformation admissible without certificates because a BRDF has no interior state" covers only the BRDF.
- Under deformation every relation ray from the deformed surface is dirty, as is every ray through the swept volume. Patch areas and frames change, which raises L11 (flat-frame M64 / patch-centre bins). So locality degenerates [INF].
- Self-contact / near-contact needs the spatial resolution that WL54/55 showed explicit C at patch scale lacks.
- Architecture.md:62-66 requires an explicit statement of implications for rigid motion, local deformation, bending and self-contact. The design gives one sentence.

**S7. The readout re-tabulates.**
- B remains a patch × 64-bin table read by `thr·B[nearest patch, bin]` (`experiments/ecro86/pt86.py:228-233`).
- For glossy or sharp receivers (C_ss chains, floor B), outgoing projection (L4) returns at the readout.
- "Outgoing never tabulated" is true for transport only. On E2 the table oracle floor is 0.0137, and it is unmeasured on CK2.

### 2.3 What is genuinely supported by evidence

- [MEAS] First-gather angular projection is the explicit-versus-oracle gap in GM1: 97–102 %, spatial ≈ 0, orders ≥ 2 ≤ 0.0009 (WL90:162-184).
- [MEAS] Exact per-ray first-gather radiance removes that bias when pooled: within −0.0025…+0.0035 of the oracle in 6/6 conditions (WL91:160-179).
- [MEAS] Sender placement is the dominant factor at equal rays: +0.050 / +0.053 (WL88 Task A). C3UR already has it.
- [MEAS] Exact physical substitution at the S1 sender improves sharp renders: C2_srcS1 on CK2 untextured 0.047 → 0.031 and 0.042 → 0.019 (WL94:155-160); C2_src on E2 / E5 (WL93D).
- [MEAS] The learned S1 operator is at the zero-operator level in the matched L1 metric (0.96–1.04 versus 1.000), while non-negative rank-16 reaches 0.38–0.60 (auditB). That supports "do not keep a learned per-patch local operator", but in a matched norm on M64-relative quantities, not physical q.
- [FACT] ER reciprocity 1.2e-5 and furnace 6e-7 (WL87:193-199). No recursed learned operator means no passivity risk beyond ER.

### 2.4 Changes required for H1 to be a valid falsifiable hypothesis

1. **Reclassify.** State H1 as "learning-free explicit comparator: C3URX1 with shared-origin shadowing and material-only peaked densification", a WL91 suffix-class follow-up, unless the owner first rules that an ER-only global solve may be the method.
2. **Change exactly one factor relative to C3URX1** at a time:
   - (a) shared origins at P ∈ {16, 64, 256} versus per-ray origins, at equal rays;
   - (b) densification ×{1, 4, 16} versus an equal-ray uniform increase, so densification is not credited with what simply adding rays would achieve.
   - Drop the deferred outgoing for orders ≥ 2, or add it as a separate factor with its table cost declared.
3. **Variance prior and kill rule in one norm.** Predeclare the expected per-draw error from lobe-hit counts (§2.2 S2). Kill on the WL91 per-draw pass-rate rule and the WL91 rule (a) time-to-tier, against C3UR and C3URX1 on the same draws.
4. **Economic kill condition.** Quality-matched edit-to-first-frame and source-update-to-frame against frozen-ER PT with the same readout, and against C3UR. Kill if H1 is not ≤ C3UR at matched quality on ≥ 2/3 held-out conditions, mirroring WL91's adoption rule. Report ray, shadow-ray, ER-evaluation and memory counts.
5. **Fix the ray-count statement and the memory estimate,** including any outgoing tables.
6. **Rigid / deformation / self-contact statement** with an explicit dirty-set derivation for each.

### 2.5 Verdict

**H1: VIABLE_WITH_CHANGES, as a learning-free comparator / oracle-ladder arm only.**
- As a candidate "reusable learned transport architecture" it is **NOT_VIABLE by definition**. There is no learned transport object; it fully recomputes per edit and per source; and the canonical role of its class is comparator.
- Its physics is sound (unbiased in expectation, passive via ER).
- Its decisive risk (single-draw variance) is the one its nearest relative already failed, and none of its changes is designed to fix that risk at constant rays.

## 3. H2, ALP (Analytic Lobe–Polygon first-gather transfer)

### 3.1 Fatal flaws

**F1. It is not an architecture distinct from H1.** H2 is a different estimator of the same quantity: the explicit first-gather flux from glossy senders.
- It shares H1's F1 (no learned transport; its "learned" object is a material BRDF fit, i.e. a BSDF surrogate) and H1's F2 (global recompute).
- Presenting H1 and H2 as "MULTIPLE_CANDIDATES" overstates the hypothesis space: both are explicit estimator choices inside H-0 (WL86B:114-117).

**F2. Its worst-case structure is the "large explicit transport table" the brief asks to be assessed.**
- Pairs scale as O(N_g·N); the design itself gives 3.3·10⁵ here.
- That is quadratic memory and quadratic per-light work:
  - P·L polygon integrals per pair per source change, about 5·10⁶ per light here;
  - plus visibility rays per pair.
- Because the integrals depend on E_l(y_j), **every source update re-evaluates the entire pair table**. H2 therefore forfeits the only learned advantage ECRO ever measured, the 3.5–3.9× cheaper source-only update (WL87:331).
- Precedents rejected for this cost class: WL52 GLT (dense O(N·M), 8–19× slower update); WL80A C3/C4 (per-port tabulation).

### 3.2 Serious issues

**S1. Scalar pair visibility V̄_pn is wrong exactly where sharp transport is hard.**
- For a narrow lobe, what matters is whether the lobe direction is occluded, not the average pair visibility.
- Partial occlusion near blockers, contacts and visibility transitions is the measured failure locus of explicit and learned coupling: WL52 contacts and transitions; WL54/55 near-contact.
- The proposed kill test (ALP-oracle versus O3-POOL "at matched visibility rays") runs on CK2 / E2. Neither is shown to contain partial panel→box occlusion. With no partial-occlusion control in the workload, this flaw cannot be detected.

**S2. Lobe-fit validity at grazing incidence is unknown, and CK2 is entirely grazing.**
- CK2 is entirely ring-0 (panel cos θ 0.024–0.164, auditA §3.2).
- LTC-type fits are typically weakest at grazing [INF].
- ER is a coated conductor; auditB finds the S1 response high-rank in fixed bins (rank ~32 for field L1 ≤ 0.15). Low rank in lobe-aligned coordinates is [HYP].

**S3. The fit kill threshold is borrowed from an unmatched norm.**
- "Field L1 > 0.15 or peak < 0.7 on S1" comes from auditB's **M64-relative, bin-projected** field metric.
- H2's purpose is to avoid bin projection. The quantity H2 consumes is the lobe integral over a receiver polygon's solid angle.
- The fit must be scored in that quantity (relative L1 of ∫_{Ω_n} f cos dω against dense ER quadrature, over the actual (ω_l, polygon) population), not against Bm. Applying the 0.15 / 0.7 threshold here is exactly the "numbers from different norms" error the brief forbids.

**S4. Glossy-receiver bin assignment by polygon centroid** is an unquantified approximation. It makes H2 inapplicable to C_ss chains (sharp sender to sharp receiver), and the design gives no control for it.

**S5. Light-model scope.**
- Area or environment incidence needs integration over incident directions as well, multiplying the per-pair cost.
- The worst case is not stated as an explicit cost formula.

**S6. Candidate generation is conservative only in name.**
- Pairs come from sampled visibility, so a missed receiver is unrecoverable (AGENTS.md selector rule, which the designer cites).
- No coverage-validation procedure is specified, e.g. an energy-coverage check like WL88 R1's 0.59 → 0.993.

### 3.3 What is genuinely supported

- [MEAS] Only indirectly: the bias that H2 would remove is the WL90 first-gather projection, and the variance it would remove is the WL91 X1 sampling term.
- Nothing H2-specific has been measured. The designer says so.

### 3.4 Changes required

- Drop H2 as an architecture candidate.
- Keep **ALP-oracle** (exact ER, dense quadrature over the receiver polygon) as a **variance-free diagnostic arm** inside the H1 ladder.
  - Restrict it to Lambert receivers.
  - Add an explicit partial-occlusion control geometry.
  - Score the lobe fit only in the polygon-integral norm.
- If H2 is ever revived, its acceptance must include source-update cost against ECRO and C3UR, and a pair-count scaling measurement at two or more N.

### 3.5 Verdict

**H2: NOT_VIABLE as specified.**
- It is an estimator variant of H1.
- It has quadratic, source-dependent pair state, which loses the source-update advantage.
- Its scalar visibility is wrong at the hardest sharp configurations.
- Its kill criterion is in an unmatched norm.
- Its useful remnant is a diagnostic oracle arm.

## 4. The falsification experiment (§6): defects that make it ill-defined

1. **O1 conflates two different quantities.**
   - "C2_srcS1, the WL94D floor": C2_srcS1 is a render arm with the **learned** receiver `ap` and learned G (`experiments/ecro_mat94/mat94.py:238-257, 271`).
   - The WL94D CK2 floor is the M64 sender, the deployed flight and the **exact** receiver M64, compared on e2_p exitance (`experiments/ecro_s94d/s94.py:152-162`).
   - For the e2_p ladder O1 must be the latter. For render E_H, the learned receiver and G make O1 non-comparable with O3, which is H1 with exact receivers. Define two separate ladders: a field ladder on e2_p with exact Lambert receivers, and a render ladder with LOW + candidate L₂ + path-traced L₃₊ (the WL92 ladder).
2. **O2 does not remove L2** (§1). Relabel it "L1 + L12 removed; L2 changed to stratified 16-point; L4 and L5 kept". The kill rule "if O2 ≈ REF" needs a numeric tolerance in the e2_p norm, e.g. ≤ 2× half-noise (0.058 box A).
3. **The acceptance criterion references a non-existent quantity.** "ROI E_H ≤ oracle + 2 × noise" on CK2 needs a CK2 table oracle, which the design itself lists as unmeasured (§7 item 3). It must be produced and frozen as a precondition, with its own noise bound.
4. **No economic or utility kill, and no binding baselines.** Add:
   - C3UR, C3URX1 and EXP arms on the same draws, and quality-matched frozen-ER PT;
   - the WL91 rule (a) time-to-tier;
   - source-update-to-frame against ECRO.
   Without these, the experiment cannot decide "practical usefulness versus full recomputation" (brief criterion 10).
5. **The CK2 workload is too narrow to carry kill decisions.**
   - It is a post hoc, owner-requested, unnumbered addendum scene with one grazing light, 4 excited incident bins, a 3-patch footprint and a 9-patch face correlation.
   - It was never admitted by a qualification rule (WL93D addendum).
   - A kill or pass on it says nothing general. Smallest valid version: a predeclared CK2 family with ≥ 8 light positions (grazing and non-grazing) × ≥ 2 box placements, admitted by a geometry-only rule and a minimum-signal rule (as in WL86 Q-tests). Keep E2 / E5 and the GM1 W2 T1 / LSPOT89 / LMIX89 held-out set evaluation-only.
6. **"≥ 90 % of draws pass" has no named pass criterion** per held-out condition. GM1 conditions use R_mirror / C_ss E_H against WL89 REF-ER; ECRO conditions use region C against WL86 REF-ER. Each needs its own frozen threshold, never cross-compared. The design states the non-cross-comparison but not the thresholds.
7. **The O3-POOL attribution rule is invalid unless y_j is redrawn** and P is a factor (§2.2 S3).
8. **The H1 kill "reduce the O1→reference gap by ≥ 50 % at the deployed ray budget"** is in an unstated norm and on an unstated draw. A single draw is variance-confounded. Specify the norm (e2_p box A rel L1 with the corrected W, or region E_H), the draw statistic (median of 16 draws) and paired CIs.
9. **New code must be declared.** O3-L4, O3-POOL and O2x on CK2 need porting of gap90 / cand91 machinery to the scene86 / CK2 fixtures (§1). This is permitted as a learning-free evaluation module, but the brief forbids presenting it as existing capability.
10. **Good elements to keep:**
    - the metric preconditions: fixing W to 4×16, restricting box A to its own patches, bootstrap CIs, half-split noise;
    - the Lambert-sender positive control;
    - the half-sector rotation and grazing / non-grazing controls;
    - the equal ray budgets.

## 5. Cost, scaling and reuse summary (critic's ledger)

| Item | H1 | H2 |
|---|---|---|
| Hidden ray tracing under a learned name | No; declared explicit [FACT design §3] | No; declared |
| Large explicit table | Relation O(N·K), linear (≈ 21 MB) [INF]. A **possible hidden 0.5 GB** if deferred outgoing for orders ≥ 2 uses per-ray tables (§2.2 S4) | **Yes**: O(N_g·N) pairs, quadratic; per-light pair integrals |
| Initial construction | ≈ C3UR relation build [INF]; unmeasured | Pair enumeration + visibility; quadratic |
| Per-edit update | Dirty re-trace (8.7–10.2 % [MEAS WL86]) + re-shadow + **global** J sweeps; ≈ C3URX1 class (69.4 ms without render, 124.7 ms to frame [MEAS WL91:228-229]) [INF] | H1 + pair re-evaluation; unmeasured |
| Source-only update | Shadow (N·P·L) + ER evaluations + global sweeps. Likely ≥ ECRO's 13–18 ms; C3URX1 was 21.4 ms without render [MEAS WL91:231] [INF] | All pair integrals per light; loses ECRO's source advantage [INF] |
| Worst case | Rays per lobe ∝ α² → densification cost ∝ α⁻²; ER evaluations ∝ L; environment lighting reintroduces L12; α → 0 needs source-dependent continuation | Quadratic pairs × P × L; environment / area lighting multiplies the cost |
| Reuse without retraining | Trivially yes (ER frozen); no transport state survives an edit beyond the clean relation rays | Same + clean pairs |
| Practical usefulness versus full recomputation | Not shown. Nearest relative failed quality-matched utility (WL91 rule (a), 3.1×) | Not shown |

## 6. Verdict on the design document as a whole

- Under the brief and the canonical role assignment, its §8 verdict **MULTIPLE_CANDIDATES_REQUIRE_DISCRIMINATION is not supported**:
  - the two "candidates" are explicit estimator variants of the existing comparator class;
  - H1 differs from the measured C3URX1 mainly in cost-side changes that do not target C3URX1's measured failure.
- The defensible verdict is **ARCHITECTURE_SELECTION_PREMATURE** for a learned reusable transport architecture, with the following recorded:
  - [MEAS] The learned local operator is at the zero-operator level in a matched norm. Rank 16 caps S1 field L1 at ≥ ~0.36. Exact first-gather evaluation removes the GM1 bias but fails per-draw variance and utility.
  - [INF] Any learned successor must beat C3UR / C3URX1 at matched quality and cost, not ECRO v0.
  - A learning-free CK2 decomposition (corrected O1 / O2 / O3-POOL with y_j redraw / O3-L4 / ALP-oracle diagnostic, plus C3UR / C3URX1 / PT baselines and an economic kill) is a legitimate **suffix-class** evidence step that would close auditA §3. It is not an architecture selection.
- The designer's honesty points stand and should be preserved:
  - the explicit declaration of ray-tracing ownership;
  - the confounded-floor treatment;
  - the metric-defect preconditions;
  - "no novelty claimed".

## 7. Sources checked by me

**Code** (under `c:\Projects\ReST-GI`):
- `experiments/gm1_relation89/rel89.py:33-80` (C3UR per-ray origins, cosine-in-bin directions, dense receiver cells)
- `experiments/gm1_c3urx1_91/cand91.py:36-115` (probe + shadow per landed ray, cost fields)
- `experiments/gm1_explicit87/quad87.py:29, 260-338` (SUB = 8; qtable P-point stratified source, patch × bin output)
- `experiments/gm1_explicit87/protocol/gm1_explicit87.json:41` (P = 16)
- `experiments/ecro86/ecro86.py:50-90, 155-170` (dirty_rays, update_state, respond, selective = full test)
- `experiments/ecro86/pt86.py:105-130, 376-402` (fixed shared offsets, source encoding)
- `experiments/gm1_gap90/gap90.py:23-33` (GM1 fixtures only)

**Docs:**
- `docs/worklogs/91_*.md:122-124, 224-253` (lifecycle medians, including PT)

**Evidence digests** (in `C:\Users\dna10\AppData\Local\Temp\claude\462fa2b2dbb71cdd\46d65e15-e083-4057-875f-9282fb68aa5e\scratchpad\wl95\`):
- `C:\Users\dna10\AppData\Local\Temp\claude\462fa2b2dbb71cdd\46d65e15-e083-4057-875f-9282fb68aa5e\scratchpad\wl95\auditA.md`
- `C:\Users\dna10\AppData\Local\Temp\claude\462fa2b2dbb71cdd\46d65e15-e083-4057-875f-9282fb68aa5e\scratchpad\wl95\auditB.md`
- `C:\Users\dna10\AppData\Local\Temp\claude\462fa2b2dbb71cdd\46d65e15-e083-4057-875f-9282fb68aa5e\scratchpad\wl95\read_code.md`
- `C:\Users\dna10\AppData\Local\Temp\claude\462fa2b2dbb71cdd\46d65e15-e083-4057-875f-9282fb68aa5e\scratchpad\wl95\read_canon.md`
- `C:\Users\dna10\AppData\Local\Temp\claude\462fa2b2dbb71cdd\46d65e15-e083-4057-875f-9282fb68aa5e\scratchpad\wl95\read_wl86_88.md`
- `C:\Users\dna10\AppData\Local\Temp\claude\462fa2b2dbb71cdd\46d65e15-e083-4057-875f-9282fb68aa5e\scratchpad\wl95\read_wl89_92.md`
- `C:\Users\dna10\AppData\Local\Temp\claude\462fa2b2dbb71cdd\46d65e15-e083-4057-875f-9282fb68aa5e\scratchpad\wl95\read_wl93.md`
- `C:\Users\dna10\AppData\Local\Temp\claude\462fa2b2dbb71cdd\46d65e15-e083-4057-875f-9282fb68aa5e\scratchpad\wl95\read_wl94.md`

**Numerical derivations.** The per-patch lobe-hit counts, the per-draw error estimates and the 0.5 GB per-ray table figure are [INF] arithmetic on stated sizes (N = 1,270, K = 512, P = 16, 64 bins, κ = 8). None was measured.