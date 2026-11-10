# GM1 architecture review — 23 — Experimental identifiability and evaluation fairness

**Status:** design-only adversarial review (2026-11-10), `Selective_Recompute` @ `c7b0698`. Nothing was implemented, run or measured. Only this file was written.
**Reviewed:** `10_design_trackA_directional_delta_transport.md` (DDT / DDT-D), `11_design_trackB_path_space_reuse.md` (ERPS / PR1–3), `12_design_trackC_alternative_operator.md` (LPTG), each in full. No other review file in this folder was read.
**Originals read for this review:** `00_PROBLEM_BRIEF.md`; `AGENTS.md`; `docs/BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md` (full); `docs/RESEARCH_ROADMAP.md` (frontier and GM0–GM4 table); `docs/worklogs/84_*` (full); `docs/worklogs/70_*` and `59_*` (control and F/S passages only). WL38, 46, 54, 56, 58, 67, 69, 82, 83 are used only through `AGENTS.md` / canonical summaries and are cited as such. No web search was made; the one statistical citation below is from memory.
**Axis:** only whether each proposed falsification experiment can isolate what it claims. Physical validity, novelty and cost plausibility of the mechanisms are other reviewers' axes.
**Labels:** FACT, HIST, DERIV, INF, HYP as in the brief. Severity: FATAL (the experiment cannot decide its question as written), MAJOR (a plausible outcome would be misattributed), MINOR (reporting or hygiene).

## 0. Findings in brief

1. **No design is unsound on reference use.** All three score arms against REF-ER and keep REF-P for the ER gap; none claims anything about Neural Full Recompute (FACT, the three §11/§12 texts). The defects are in what is *matched*, what is *charged*, and which *noise* is controlled.
2. **Six defects are shared by all three experiments** (§1). The most consequential: "matched quality" is undefined between a biased, spatially correlated state estimator and an unbiased, white-noise path tracer; and T0 investment (a converged T0 image, a T0 state build) is given to the proposed arm free and withheld from the path-traced control.
3. **Track A's experiment is the least decisive.** Its first stop rule (F1) infers "no economic content even with a perfect state" from an arm that pays two T0 path traces per lookup, and compares it against a control that receives no T0 information. Both directions of F1 are misattributed. The one advantage Track A argues against path tracing (view independence) is not measured by any arm.
4. **Track B's headroom measurement (Stage 1, ρ) is the only quantity in the three documents that is implementation-independent and noise-free by construction.** Its cost verdict (F1) is not, and its PR2 arm has the T0-investment defect.
5. **Track C's experiment is the best staged** (analytic Stage 0, null and off-path controls, "stale passes ⇒ experiment invalid", "S cheaper than REF-ER but not than F ⇒ drop the selective claim"). Its gap: the highlight gate is satisfied by fresh readout chains, so it does not test the *persistent* state's directional content at all.
6. **A design whose advantage cannot be isolated as written:** Track A (DDT-D). Track C's selective-update advantage is isolable; its *representation* claim (H-rep) is isolable only after the path-class correction in §4.

## 1. Defects common to all three experiments

**X1 — Quality matching between biased and unbiased estimators is undefined. MAJOR.**
- FACT: ER's image L1 flattens at about 0.035–0.043 while the layered reference keeps converging, and crosses it near 1024 spp (WL84). Any stored-state arm (A4/A5, LPTG F/S) has an analogous floor from node / cache / table resolution (Track C §14.5 says so).
- DERIV: pooled relative L1 of a white-noise estimator falls as n^(-1/2) and falls further under any spatial pooling; the L1 of a structured bias does neither. Two images with equal pooled L1 are therefore not equal in quality, and the direction of the mismatch favours the biased arm at pixel scale and the unbiased arm at every coarser scale. Tracks B §7 and C §10.9 name this effect; neither experiment measures it.
- Counterexample: LPTG with low-frequency blotches at pooled L1 0.05 is "matched" to REF-ER at the spp whose noise gives 0.05; after a 4×4 box filter the path tracer's error is about 4× lower and LPTG's is unchanged. A speedup read at that level is a fidelity difference.
- Correction: define matching as a **threshold on every gated metric simultaneously** (per region: L1 at pixel scale *and* at two predeclared pooled scales, highlight error, signed-change slope), and report the bias / variance split from replicate seeds (error of the seed-mean image = bias; remainder = variance).

**X2 — Quality levels are chosen after baseline costs exist, with no anchoring rule. MAJOR.**
- HIST: at loose contracts path-traced re-rendering won at 1–16 spp (WL67, WL69, WL70); at tight ones the state methods missed by a bias floor (WL56, WL59). The level decides the verdict.
- Track A freezes gates "after the A1/A2 cost ladders exist"; Track B "after Stage 0"; Track C fixes "quality levels" from the Stage 1 ladder and falsifies H-econ only if S loses "at every predeclared quality level" (one winning level out of several suffices: an unprotected multiple comparison).
- Correction (DERIV, triangle inequality on a common normalisation): err(arm, REF-P) ≤ err(arm, REF-ER) + err(REF-ER, REF-P). Anchor the levels to the measured in-scene ER gap g = err(REF-ER, REF-P): predeclare the rule now (for example levels g and g/2 per region), compute the numbers after the references exist, and report the whole time-to-quality curve. A global-mechanism error far below g buys nothing against REF-P; one far above g makes the global mechanism the dominant error. A win must hold at the anchored levels, not at any level.

**X3 — Signed-change metrics have no noise model. MAJOR.**
- All three reuse "signed-change slope / correlation" and the WL84 rule "reference uncertainty ≤ 1/4 of the smallest gated error". That rule was qualified for absolute error. The change image Δ = T1 − T0 from independent references has √2 times the per-image noise, and its signal is small exactly in the regions GM1 cares about (second-order receivers, E-small).
- DERIV (errors-in-variables attenuation; standard result, e.g. Fuller 1987, *Measurement Error Models*, cited from memory): regressing an arm's Δ on a noisy reference Δ gives slope × λ with λ = Var(signal) / (Var(signal) + Var(ref noise)) < 1. A correct arm then reads as "under-responding", and correlation is capped at sqrt(λ_ref · λ_arm). Estimator noise is thereby mistaken for a physical response deficit.
- Correction: use the two reference halves as instruments, slope = Cov(arm Δ, ref Δ_A) / Cov(ref Δ_A, ref Δ_B); report λ and the correlation ceiling per region; declare a region INDETERMINATE when the reference Δ signal-to-noise is below a predeclared value (the WL84 "reference-limited" class, FACT WL84 run 2 design).
- Related: Track A derives its region masks "from REF-ER T0/T1". Selecting pixels on a noisy Δ and scoring on the same Δ is selection on the outcome. Masks must come from geometry, or from half A with scoring on half B.

**X4 — No replicate seeds for the arms. MAJOR.**
- HIST: in WL38 the full-recompute seed spread alone (0.23 dB) was comparable to the criterion that WL37 missed; WL84 needed 8 replicates to show that a 0.007 margin was stable. None of the three experiments declares arm replicates; Track C's frozen samples make a single realisation look deterministic.
- Correction: at least 8 independent seeds per stochastic arm (for LPTG: independent node rotations and chain seeds for F; S inherits them), all gated metrics with intervals, and a minimum cost margin predeclared against timing spread (WL84 used a 1.10 factor; WL70's 2.09 vs 2.40 s margin was inside the timing definition and was reported as fragile, HIST).

**X5 — T0 investment is asymmetric. MAJOR.**
- Track A's A2 uses a "converged T0 image"; Track B's PR2 the same; A4 and LPTG S use a T0 state build. The cold REF-ER control gets nothing from T0.
- Counterexample: in the unaffected mask PR2 carries the converged T0 error (near zero). "Matched error in every mask" then forces cold REF-ER to an spp it would never need, and PR2 "wins" on pixels where it did no T1 work, paid for by uncharged T0 samples. The same holds for any state arm whose T0 build exceeded the gate quality.
- Correction: (i) every arm, including the path tracer, must deliver gate quality at **both** T0 and T1; (ii) report edit-to-valid-output *and* lifecycle = T0 cost + T1 cost, with the break-even number of edits or views; (iii) the T0 image or state handed to a reuse arm is built to the gate level, not "converged", unless the excess is charged; (iv) add the missing control **masked cold re-render**: the T0 image kept outside the geometrically discovered dirty pixels, cold path tracing inside. Only the margin over that control is attributable to the mechanism rather than to "skip unchanged pixels".

**X6 — Cost verdicts do not separate mechanism from implementation maturity. MAJOR.**
- FACT (WL84): Python tracing is 85–95 % of wall time and 0.4–1.2 s per launch; cold REF-ER runs as an optimised megakernel while every proposed arm adds custom Dr.Jit bookkeeping, sparse solves or segment tests. A 64-spp ER image costs about 10 ms of kernel time, a scale at which fixed overheads decide.
- Correction: predeclare two cost axes, (a) implementation-independent work counts (scene rays by type, ER `eval` / `sample` calls, segment tests, multiply-adds, bytes touched) and (b) uncontended GPU kernel time. Verdict mapping: both favourable = supported; both unfavourable = mechanism falsified at this workload; (a) favourable and (b) not = IMPLEMENTATION-LIMITED, explicitly not a mechanism verdict. Add a validity rule for the workload: if cold REF-ER reaches the anchored level within a small predeclared multiple of the fixed-overhead floor, record "economically non-discriminating workload" (a real finding about the workload, HIST pattern WL67–70, but not a test of the mechanism).

Two smaller shared points. **X7 (MINOR):** none of the designs has a learned global state, so no experiment here invalidates any learned representation; every F/S result would be evidence about a classical incremental solver around frozen ER, and must not be reported as "selective recompute of learned transport" (roadmap GM1 exit clause; `AGENTS.md` terminology). **X8 (MINOR):** WL84 qualified ER at static depth 4 with limited indirect light (FACT). The REF-ER vs REF-P gap in a closed multi-bounce glossy scene is unmeasured and may exceed 0.035–0.04; all three must report it first, per region and for Δ (slope of REF-ER Δ on REF-P Δ), before any arm is read. Track C states this; A and B only imply it.

## 2. Track A — DDT / DDT-D

**Verified sound.**
- REF-ER primary, REF-P secondary, both at T0 and T1 with WL84-style qualification.
- A0 stale, A5 same-architecture rebuild, and A2 (difference re-render) as the stronger path-traced control are all present; Neural Full Recompute and Native Reuse are explicitly excluded as UNMEASURED.
- An adversarial edit (E-adv) with the expected dense result is predeclared; ghost and negative-radiance mass are measured; no runtime oracle masks.
- The oracle / finite-state split (A3 vs A4) is the right idea for separating "identity has no value" from "state cannot hold it".

**Defects.**

| # | Severity | Defect | Counterexample or derivation |
|---|---|---|---|
| A-1 | **FATAL for F1** | A3 is a quality oracle but a cost pessimist, and F1 uses it as a cost bound. | A3 evaluates each `b0` lookup by an on-demand T0 path trace (two per changed ray). A stored state answers the same lookup in O(1). If A3 loses to A1 on rays and GPU time, nothing follows about DDT with a stored state, so "the identity has no economic content even with a perfect state. Stop." is a non sequitur. Conversely, if A3 wins, the win belongs to residual path tracing (Track A §13 names it prior art), not to DDT-D. |
| A-2 | **MAJOR** | A3 is not the mechanism under test. | DDT-D is defined (§1) as overwrite of marked state cells. A3 has "no stored field": it has no cells to mark or overwrite, and its output object (image? at which camera?) is unspecified. A3 and A4 differ in mechanism, not only in `b0` fidelity, so F2's attribution "representation precondition fails" is confounded with the image-vs-state change. |
| A-3 | **MAJOR** | F1's control is the wrong one. | F1 compares A3 with A1 (cold) on "affected stationary regions" only. A3 starts from T0 information; A1 does not, and A1 cannot restrict itself to a region. The fair controls are A2 and the masked cold re-render (X5). As written, A3 can pass F1 purely through the free T0 prior. |
| A-4 | **MAJOR** | The only advantage argued against path tracing is unmeasured. | §11 limits the claim to "a view-independent state refresh, or many views per edit". §12 declares no camera count, no readout procedure from state to image, and no per-view amortisation. F1–F3 cannot confirm or refute it. |
| A-5 | **MAJOR** | A4 vs A5 is not noise-isolated. | Unlike LPTG, DDT-D with tolerance τ is not equal to a rebuild. With independent seeds, A4 − A5 mixes τ-truncation bias, missed discovery and Monte Carlo noise. |
| A-6 | **MAJOR** | Discovery is not scored against an oracle dirty set. | The strategy document requires "radiometrically weighted false negatives, false-positive work and total discovery cost" and states that a missed candidate is irrecoverable (FACT). §12 measures μ and φ_τ only. Counterexample: a sharp-lobe cell (x, ω_o) of 4D measure ∝ α² that no mover-surface discovery sample maps into stays stale; with independent seeds its error is indistinguishable from state noise. |
| A-7 | **MAJOR** | φ_τ is estimator-dependent. | Thresholding a Monte Carlo magnitude: E|s + n| > |s|, so noise inflates marks (false-positive work), while for E-small the cancelling ± lobes fall under τ (false negatives). φ_τ then depends on sample count and is not a scene property; "dirty set not small" (falsification clause in §1) cannot be separated from "source estimate too noisy". |
| A-8 | MINOR | Depth semantics unspecified. | A state solve includes all bounces; WL84's REF-ER used max depth 4. A truncation mismatch would be booked as mechanism error. |
| A-9 | MINOR | No null-edit or off-path control; stale readout not defined (T1 primary visibility or not); cold vs warm-started A5 not declared (`AGENTS.md` gate: declare the policy). |
| A-10 | MINOR | No analytic admission stage. | A4 failing at "two predeclared resolutions" cannot separate "table too coarse" from "mechanism wrong" (Track A §14 concedes this); the ER lobe roughness census was not checked. |

**Minimal corrections.**
1. Replace F1 by two bounds from the same A3 run: **optimistic** work (changed-set rays and ER calls only, lookups charged at zero) and **pessimistic** work (as run). Stop only if the optimistic bound loses to A2 *and* to the masked cold re-render. A pass on the optimistic bound alone is "headroom exists", nothing more.
2. Define one output object for A3–A5: the view-independent state, read out by one declared procedure at ≥ 2 cameras, one of them held out from every design choice; charge readout; report cost against number of views with the crossover against A1 / A2.
3. Common random numbers per state cell between the T0 build and A5. Then A4 − A5 is noise-free and equals truncation plus missed discovery, and the cells that differ between the T0 build and A5 are the **oracle dirty set** for false-negative / false-positive scoring (A-5, A-6 together).
4. Report φ_τ against source sample count and against the oracle set; add null-edit (φ must be exactly 0) and off-path controls in the WL70 manner (HIST: four rejects, one accept).
5. REF-ER with Russian roulette rather than fixed depth; report its difference from depth 4.
6. A zero-render Stage 0 shared with Track C: lobe census of the frozen closures and the angular sample count they imply, before any table resolution is chosen.
7. Apply X1–X6.

**Verdict on this axis: UNSOUND as written; CONDITIONALLY SOUND after corrections 1–3.** Without them the claimed advantage of DDT-D cannot be isolated: the oracle arm tests a different, published mechanism under the wrong cost model, and the argued advantage has no measurement.

## 3. Track B — ERPS (PR1 / PR2 / PR3)

**Verified sound.**
- PR1 equals the same-seed cold render (DERIV, Track B §1), so its quality matching is automatic and its comparison is pure cost: no fidelity-for-speed trade is possible in that arm.
- Stage 1 (same-seed T0/T1 with per-path logging) measures ρ and the dirty share of radiance exactly, with a zero-false-negative check of geometric discovery. This is noise-free and independent of implementation maturity.
- Births are tested where reuse is expected to fail (F3), stated as an expectation before the run. The biased PR3 variant is a labelled negative control.
- Arms are scored against REF-ER; all PR arms are labelled path-traced re-rendering; depth is fixed and equal across arms; wall time is reported but unused.
- The track asks for no GM1 credit: a pass makes it a mandatory comparator, not a mechanism.

**Defects.**

| # | Severity | Defect | Counterexample or derivation |
|---|---|---|---|
| B-1 | **MAJOR** | PR2's T0 anchor is uncharged and "matched error in every mask" is ill-posed (X5). | See X5's counterexample. With a gate-level T0 image instead, the T0 noise persists into T1 as correlated error and adds to the difference noise in dirty masks, which may reverse the result. The outcome of F1 is decided by this undeclared choice. |
| B-2 | **MAJOR** | F1 is an implementation-sensitive verdict (X6). | Cold REF-ER is a megakernel; store, discovery and suffix replay are new Dr.Jit code with gather / scatter over path state. "Not a competitive comparator" can be produced by bookkeeping overhead alone. |
| B-3 | MAJOR (scope) | ρ is over-read. | ρ depends on the sampler (depth cap, light-sample count, Russian roulette) and on fixture closedness, and it bounds only path-granular replay with that sampler. It is not an upper bound for field-based reuse (LPTG re-solves downstream change without retracing; its relevant quantity is the invalid link fraction f), nor for DDT-D's φ_τ. §12's wording is careful; §0's "reuse headroom" is not. Also ρ is unweighted segment work: a small ρ with a large dirty *radiance* share means reuse saves work only where the image did not change. |
| B-4 | MINOR | No null-edit control for bit-identity. | A rebuilt or refitted acceleration structure can change hit distances at float precision and flag clean paths as dirty. A null edit and a move-and-return edit give the nondeterminism floor. |
| B-5 | MINOR | F3 has no interval. | Birth-mask error at equal rays is a low-count Monte Carlo quantity; without replicates (X4) "not below cold" is undecidable. |
| B-6 | MINOR | Role wording. | "ER reuse is valid Native Reuse" uses a role reserved for an *external* baseline whose representation survives the edit (strategy document, comparison 2). ER is ReST-GI's own frozen local rule; call it "not invalidated". Also time cold REF-ER (no logging) and cold store build (F) separately: they are different roles even when the traced paths coincide. |

**Minimal corrections.**
1. Promote Stage 1 to the primary deliverable and extend its log: per region and per first-dirty depth, (a) segment-work fraction, (b) share of T1 radiance on dirty paths, (c) share of |Δ| on dirty paths. Run it on the common fixture used by the other tracks (§5).
2. For Stage 2, declare the T0 budget rule of X5 and add the masked cold re-render.
3. Report work counts beside kernel time and use the three-way verdict of X6 for F1.
4. Null-edit and move-and-return controls; ≥ 8 seeds for F1 and F3.
5. State ρ's scope as "path-granular replay with this sampler on this fixture".

**Verdict on this axis: SOUND for the headroom measurement (Stage 1); CONDITIONALLY SOUND for the cost comparison (Stage 2, F1) after corrections 2–3.** Its advantage (do not redo unchanged work) is isolable, and is bounded by a quantity it measures exactly.

## 4. Track C — LPTG

**Verified sound.**
- Three separately falsifiable claims (H-rep, H-edit, H-econ) with a table mapping each outcome to a meaning.
- Stage 0 is render-free and can reject before implementation: it separates "wrong for this material family" from everything else.
- Controls: stale (with T1 primary visibility, defined), same-architecture F, REF-ER ladder, REF-P; null edit and off-path move as invalidation controls (WL70 precedent); "stale passes ⇒ experiment invalid, not a result"; "S cheaper than REF-ER but not than F ⇒ the advantage is the cache". These two rules are exactly what the other tracks lack.
- Two ablations with predicted outcomes (all lobes through the field; final-gather readout).
- REF-ER with Russian roulette, matching a residual-terminated solve; ER gap reported first; Neural Full Recompute stated as UNMEASURED; NRC named as an unmeasured competitor rather than silently omitted.
- α\* and K fixed from lobe geometry before Stage 0, no sweep.

**Defects.**

| # | Severity | Defect | Counterexample or derivation |
|---|---|---|---|
| C-1 | **MAJOR** | The highlight gate does not test the persistent state. | At readout, sharp reflections are produced by s *fresh* sharp-lobe chains per pixel (§6). A camera-visible highlight ROI is therefore passed by per-frame path tracing; the stored field contributes only what the chain terminates into. H-rep ("sharp glossy interreflection represented") can pass with a field that holds no sharp information, which is the roadmap's "full retracer does not complete GM1" case for the sharp part. The classes that do exercise stored state are S-vertices *inside* links (R→S→R) and the S→R class the design calls its weakest (§10.2). |
| C-2 | **MAJOR** | F ≡ S hides everything F and S share. | F at T1 uses the same cache points, nodes and random numbers (§3), so S − F ≈ 0 is an implementation test (as in HIST WL59, 1e-7, where F itself failed fidelity). Between-node aliasing (§3 limit, §10.7), frozen-sample variance and cache-spacing bias are identical in both and invisible to H-edit. The experiment's mover is a box, so the thin-occluder failure is never provoked. |
| C-3 | **MAJOR** | H-rep is tied to cost. | "F misses the gate *while REF-ER at equal GPU time meets it*" falsifies H-rep only jointly with an economic condition. If F has a bias floor above the gate and the path tracer is slow, the row does not fire and a representation failure is recorded as nothing. |
| C-4 | **MAJOR** | Structured error is acknowledged and unmeasured (X1). | §10.9 states that frozen-sample error is spatially correlated and under-reported by pooled L1; §12 lists only "error vs REF-ER by region". |
| C-5 | MAJOR | No retracer test. | §7 worst case 3 ("a path tracer with overhead") has no measurement: readout rays per pixel at the matched level are not compared with the path tracer's, and no arm removes the field. |
| C-6 | MINOR | "Not cheaper at every predeclared level" (X2); T0 build not in H-econ (X5); arm replicates absent (X4); kernel time of a sparse solve vs a megakernel (X6). |
| C-7 | MINOR | The S-restricted `sample` / `eval` entry point is new numerics. | The three identity tests in §9 cover it; add that REF-ER must run the *unrestricted* frozen ER so that a restriction bug appears as arm error, not as a shared artefact. |

**Minimal corrections.**
1. **Path-class decomposition of REF-ER.** Label each REF-ER path by the R / S class of its vertices (read-only sampled-lobe label) and store per-class images. Gate the field on the classes it owns (R-terminated indirect, R→S→R, S→R onto rough receivers), separately from camera-visible S chains. Add a predeclared ROI for the moved reflected highlight on a *rough* stationary receiver. H-rep is then about the stored state.
2. **Independent-seed F′** (≥ 8 node rotations / chain seeds). Seed-mean error is the structural bias; spread is frozen-sample variance. Add one thin-mover variant narrower than the node spacing, chosen from geometry alone, so the shared S / F blind spot is provoked and read against REF-ER.
3. Split the H-rep row: (a) F′ seed-mean bias above the gate at the declared α\*, K ⇒ H-rep falsified, with no time condition; (b) equal-time comparisons belong to H-econ only.
4. Multi-scale and bias / variance reporting (X1) in the quality match.
5. Retracer rule: if readout rays per pixel at the anchored level are not below REF-ER's rays per pixel, or a no-field arm (chains terminating in direct light only) meets the same gates, the mechanism is a retracer for this scope.
6. H-econ at the anchored levels (X2), both lifecycle and edit-to-valid-output (X5), with the three-way cost verdict (X6).

**Verdict on this axis: CONDITIONALLY SOUND.** The update claim (H-edit) and the economic claim (H-econ) are isolable with small changes. The representation claim (H-rep) is not isolable as written (C-1, C-3) and becomes so with corrections 1–3.

## 5. Threat patterns against the three experiments

| Threat | A | B | C |
|---|---|---|---|
| Newly visible glossy path absent from stored support | Edits cover it; never-queried `b0` directions are the stated risk, but only A4 sees it and A-6 leaves misses unscored | Tested (F3), needs intervals | Camera-visible case passes by fresh chains (C-1); rough-receiver case has no ROI |
| Changed occlusion, unchanged endpoints | Covered by chord discovery; false negatives unscored (A-6) | Exact per stored segment; verified in Stage 1 | Exact per stored segment; between-node occluder shared by S and F (C-2) |
| Narrow lobes lost by angular compression | A4 two resolutions; no admission stage (A-10) | Not applicable (no basis) | Stage 0 plus all-field ablation: adequate |
| Broad multi-bounce spread after a local edit | E-adv predeclared; φ_τ noise-dependent (A-7) | ρ by depth after correction 1 | f measured; downstream solve cost measured |
| Truncation / sparsity losing energy | τ is a bias budget; needs CRN A5 (A-5) | None (unbiased arms) | Residual-terminated; ρ estimator caveat stated (HIST WL54) |
| Discovery cost | Measured, not scored against oracle | Split out in kernel time | Split out |
| Negative-delta cancellation | Ghost mass measured; E-small present | PR2 fallback weight; decorrelation stated | Not applicable (overwrite) |
| Noise mistaken for physical change | X3, A-7 open | X3 open; PR1 immune | X3 open; C-2 hides shared sample error |
| Hidden full path tracing | A3 is one, declared; its cost reading is wrong (A-1) | Is one, declared | Readout may be one; untested (C-5) |
| Speedup from lower fidelity | X1 open | PR1 immune; PR2 via X5 | X1, C-4 open |

## 6. Cross-design comparison (this axis only)

| Criterion | A (DDT-D) | B (ERPS) | C (LPTG) |
|---|---|---|---|
| REF-P / REF-ER used so ER bias is not booked to the mechanism | Yes | Yes | Yes, and gap reported first |
| Stale, full rebuild, incremental, independent T0/T1 references | Present; stale readout undefined | Present; rebuild ≡ cold render, to be timed separately | Present and best specified |
| Baseline roles kept apart | Yes | Yes, one wording slip (B-6) | Yes |
| Noise vs physical change | Weakest (A-5, A-7, mask circularity) | Strongest for PR1 / Stage 1 | Deterministic F ≡ S hides shared error (C-2) |
| Quality matching fixed before results | Timing of freeze stated; definition missing | Same; plus T0 anchor undefined | Same; plus "any level" loophole |
| Mechanism wrong vs implementation immature | Not separable (A-1, A-2, A-10) | Separable for ρ; not for F1 | Separable for Stage 0 and H-edit; not for H-econ without work counts |
| Advantage isolable as written | **No** | Yes (bounded by 1/ρ) | Selective update: yes. Representation: only after C-1 / C-3 corrections |
| Verdict | UNSOUND as written | SOUND (Stage 1) / CONDITIONALLY SOUND (Stage 2) | CONDITIONALLY SOUND |

**One shared fixture would make the three comparable (INF).** The tracks define three different invalid-fraction measures (φ_τ, ρ, f) on three different scenes; no cross-track statement is possible from them. Track B's corrected Stage 1 log on a single common GM1 fixture (dirty segment work, dirty radiance share and dirty |Δ| share, by region and first-dirty depth) is an exact, learning-free and implementation-independent prior for all three: it bounds PR arms directly, gives the oracle against which A's discovery and C's link invalidation can be scored, and shows before any mechanism is built whether the workload discriminates at all. This is an observation about identifiability, not a recommendation of a next batch; whether any experiment is run is a human decision.

**Residual limits of this review.** No measurement supports any statement here beyond the cited worklogs. The attenuation formula is textbook statistics cited from memory. WL70 and WL59 were read only in their control passages; WL82, WL83 and the 46–58 line only through canonical summaries.
