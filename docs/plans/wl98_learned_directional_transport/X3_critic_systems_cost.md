# WL98 — Adversarial critic X3: systems and cost

**Status.** Final report of critic X3 (lens: systems and cost). Design-only critique of `DESIGN_REVIEW.md`, `D1_physics_measure_first.md`, `D2_learning_generalization_first.md`, `D3_systems_cost_first.md` (all four read in full). Nothing was implemented, trained, rendered or adopted; no existing file was changed; no commit. This report was resumed from an interrupted attempt of the same role; its scratch script (`cost_check.py`) was re-run and its numbers re-verified before use.

**Labels.** [MEAS] measured, with source. [AUDIT] WL97 interpretation. [CODE] read from frozen code. [ARITH] my arithmetic on recorded numbers (scripts below). [PRED] my prediction. [HUM] human decision.

**Scratch (design input only, not evidence of record).**
- `experiments/wl98_calc/X3/cost_check.py` → `results/wl98_calc/X3/cost_check.json` (per-unit rates, WL91 ladder floor / noise decomposition, zero-cost quality-matched bound, candidate counts).
- `experiments/wl98_calc/X3/cost_check2.py` → `results/wl98_calc/X3/cost_check2.json` (kernel-basis ceilings, ECRO generator FLOP split and ECRO-FC, XQT per-flight cost, time-to-tier and its floor sensitivity, draw pooling, RB-DTO with symmetric counts, LIEM break-even).
- Inputs (read-only): `results/gm1_c3urx1_91/results/econ/{ladder91,life91,scale91}.json`, `results/ecro_util96a/cost_s1.0.json`, `results/audit97/E5/e5_tables.txt` §4–6, `results/wl98_calc/D3/cost_model.json`, `experiments/ecro86/model86.py:83-110`, `experiments/ecro86/protocol/ecro86.json`.

---

## 0. Bottom line

1. **In the current scene86 / GM1 / CK2 family no candidate in any of the four designs has a cost lever that belongs to learning.** Every cost or quality-matched lever I could compute is equally available to the candidate's learning-free twin (ELT, analytic-kernel XQT twin, flux-conserving / adaptive binning, sender-side deposition, pooled C3URX1). [ARITH + PRED]
2. **The only large, measured-derived lever in GM1 is the table bias floor seen through the frame**, not the cost of producing the table. On the WL91 ladders, two *learning-free* explicit variants differ by 1.75× in frame samples needed to reach the R_mirror 0.06 tier (C3URX1 161 spp vs C3UR 281 spp, W1 T1), while whole-image (`stationary_all`) tiers are identical across all table arms to within 1–2 %. [ARITH on MEAS WL91] Any learned advantage must therefore be shown as a lower floor than the best learning-free twin at matched table cost, in a pre-declared ROI, with draw-resolved uncertainty. No plan states this mechanism explicitly.
3. **Table-level replacement ceilings on the kernel basis** (zero-cost learned replacement of the q ER-half plus four orders, frame 64 spp): 1.135 at N 1,270 and 1.29 at N 11,430; without the frame 1.60 / 1.39. With an engineered dirty test they rise to 1.16 / 1.53 (frame) and 2.62 / 2.03 (no frame). The learned replacement may cost at most −2.7 ms (impossible) to 15.6 ms depending on convention to reach 1.25. The ceiling is set as much by the explicit implementation's relation maintenance (dirty test 4.4 → 43.5 ms kernel) as by transport work. [ARITH on MEAS E5]
4. **Candidate survival (systems lens).** FLS: NO. ECRO-FC: NO as a candidate (diagnostic arm only, and confounded). XQT: CONDITIONAL, with a predicted Stage-1 stop. LLT: NO. LIEM-S: CONDITIONAL (outside GM1; lever only for long walks with a small network; kill rule too weak). LIEM-B: NO as a candidate. RB-DTO: CONDITIONAL (ownership [HUM]; D3's regime map is inflated by an asymmetric twin; symmetric recount gives U ≈ 0.5–1.5 in its own example). LER-Δ: NO.
5. **Proposed verdict: INSUFFICIENT_EVIDENCE** (for GM1 / CK2 alone, NO_DEFENSIBLE_CANDIDATE, scoped). The two candidates with a possible cost regime (RB-DTO, LIEM-S) live in workloads that do not exist yet, and their regime can only be established by a learning-free Stage-0 anatomy that the plans already contain but specify with timing and comparator defects (§5).

---

## 1. Recorded cost facts used (verified this session)

All RTX 3080 Ti, LabServer63, N = 1,270 unless noted. "kernel" = Dr.Jit exec or torch-profiler GPU time as recorded; "wall" = synchronised wall.

| fact | value | source |
|---|---|---|
| R0 edit → frame 64 (wall / kernel) | 144.1 / 25.5 ms | [MEAS] `cost_s1.0.json` (WL96A) |
| C3URX1 / C3UR / R0 edit wall | 124.7 / 138.2 / 145.4 ms; kernel 22.4 / 24.3 / 24.4 | [MEAS] `life91.json` |
| frame 512² × 64 spp | 58.4 wall / 22.67 kernel; 512² × 1 spp 36.2 wall / 0.47 kernel (≈ 35 ms fixed per render call) | [MEAS] E5 §4 |
| q table (20.8 M ER evals + 20,320 shadow rays) | 24.1 wall / 1.84 kernel → 88.6 ps per ER eval | [MEAS] E5 §4; [ARITH] |
| one binned order (N·K = 650,240 entries) | 1.10 wall / 0.528 kernel → 0.81 ns / entry (0.39 ns at N 11,430) | [MEAS] E5 §4; [ARITH] |
| full relation build (650,240 rays) | 24.1 wall / 0.176 Dr.Jit exec → 0.27 ns / ray (torch part unprofiled) | [MEAS] E5 §4 |
| selective re-trace (49,936 dirty rays) | 0.053 ms exec → 1.06 ns / ray | [MEAS] E5 §4 |
| dirty test over all relation rays (torch) | 4.43 ms (N 1,270) → 43.5 ms (N 11,430) kernel, ≈ 7 ns / entry | [MEAS] E5 §4 |
| XOp (relation coefficients) resident W | 2.0 → 21.8 ms kernel | [MEAS] E5 §4 |
| ECRO generator (factors) | 4.20 → 37.7 ms kernel; fitted exponent over N 1,270–11,430: **0.998** | [MEAS] E5 §4; [ARITH] |
| C3URX1 exact first gather, per light | 526,467 probe intersections + 526,467 shadow rays; 11–14 ms wall for 0.26–2.1 M landed rays (flat → overhead) | [MEAS] `life91.json`, `scale91.json` |
| C3URX1 / C3UR tables vs number of sources | 25 / 78 / 314 ms and 35 / 132 / 517 ms for 1 / 4 / 16 sources (≈ 19 / 32 ms per added source, wall) | [MEAS] `scale91.json` |
| WL87 source-only update lever of a linear learned operator | 3.5–3.9×, survives overhead removal | [MEAS] WL87; [AUDIT] D-05 |
| lifecycle CPU overhead share | 69–81 % | [AUDIT] E5 §3.3 / D-09 |

**Two corrections to design claims.**
- D3 (§0.3, §2.3, §6) states that the ECRO generator "scales as O(N²)". The dense `Ft` is N² in memory (6.5 MB → 0.52 GB → 40 GB at 100 k) [CODE `model86.py:83-89`], but the measured time scales as N^1.0 over 1,270–11,430. By FLOP count the per-(patch, bin) heads (2 × 40,960 MAC per token, N × 64 tokens) are 91 % of the generator at N 1,270 and 54 % at N 11,430; the dense message passing overtakes them only near N ≈ 13,700 [ARITH G]. The quadratic term is a memory wall and a future time wall, not the current cost.
- The DESIGN_REVIEW statement that FLS's 0.53 M rays per update "equals the C3URX1 per-light count" is numerically right but not a like-for-like advantage: C3URX1's later orders gather from a cached relation, while every FLS ray is state-dependent and fresh (§2, law L6).

---

## 2. First-principles cost laws (apply to every candidate)

**L1 — Amdahl ceiling of a table-level learned role (kernel basis) [ARITH F].** U = L / (L − A + c).

| N, dirty test | frame 64 | frame 16 | frame 4 | frame 1 | no frame |
|---|---|---|---|---|---|
| 1,270, recorded | 1.135 | 1.33 | 1.49 | 1.56 | 1.60 |
| 1,270, engineered (0.5 ns / entry) | 1.16 | 1.49 | 2.01 | 2.36 | 2.62 |
| 11,430, recorded | 1.29 | 1.36 | 1.38 | 1.39 | 1.39 |
| 11,430, engineered | 1.53 | 1.83 | 1.97 | 2.01 | 2.03 |

Maximum learned cost c for U ≥ 1.25: impossible at N 1,270 with a 64-spp frame; 0.7–1.9 ms at N 1,270 for ≤ 16 spp or no frame; 2.8–15.6 ms at N 11,430. These are cost-at-equal-quality ceilings; they ignore the quality lever L3.

**L2 — Per-entry cost law [ARITH].** An explicit order costs 0.39–0.81 ns per relation entry (kernel, incl. launch), a ray 0.27–1.06 ns, an ER evaluation 0.09 ns. A learned evaluation per entry or per incident sample costs ≈ 1–20 ns (10⁴–10⁵ FLOP at 10–30 TFLOP/s effective). A learned operator therefore wins on cost only if it compresses (orders × entries) by more than r = t_learned / t_gather ≈ 3–50. Element-level learned work (one evaluation per element per sweep, as FLS's Λ_θ) is cheap, but then it replaces only element-level explicit work, which is also cheap.

**L3 — The quality-matched lever lives in the frame [ARITH I on MEAS WL91].** Fitting err² = f² + c²/spp (spp 64–4096) to the WL91 ladders:
- `stationary_all`: floors 0.007–0.0085 for R0, C3UR and C3URX1; spp to every tier identical within 1–2 %. **Whole-image quality-matched utility of any table improvement is ≈ 1.0 in GM1.**
- `R_mirror` (3,244 / 2,129 px, about 1 % of the stationary pixels): floors R0 0.062, C3UR 0.045, C3URX1 0.030 (W1 T1 deployed draw). Tier 0.06: NOT REACHED / 281 / 161 spp. At 0.354 kernel ms per spp this is 99.5 vs 57 ms of frame, a ≈ 1.6× kernel-basis lifecycle ratio produced by the floor alone, learning-free.
- **Ill-conditioning near the floor.** A ±0.005 floor shift moves the W1 deployed C3URX1 spp at tier 0.06 within 147–183, but at W2 (deployed floor 0.054) within 373–4,939, and tier 0.05 becomes unreachable. Single-draw floors vary 0.025–0.054 across three draws (W2). Time-to-tier near a floor is dominated by draw variability and reference noise, not by the treatment.
- PT does not fit this model in 64–4096 spp (heavy-tailed convergence; observed 0.044 at 4,096, 0.024 at 16,384 spp). PT floors from the fit are not floors.

**L4 — Explicit variance is amortisable over steady-state frames [ARITH J].** One extra pooled C3URX1 draw costs ≈ 0.76 ms kernel (re-trace 0.18 + X1 evaluations 0.06 + gather 0.53), 3.4 % of a 64-spp frame. Between edits the geometry is static, so 16 draws can be pooled over 16 frames at ~3 % overhead. A learned or deterministic "no single-draw variance" advantage (FLS, XQT, LIEM-S) is therefore a **latency** advantage over the first ~m frames after an edit, not a steady-state one. No plan defines utility in this way.

**L5 — Linearity in the source is a cost property [MEAS + PRED].** The only measured learned advantage in the lineage is the source-only update (WL87, 3.5–3.9×), which exists because ECRO is linear in the source encoding. Any closure whose output shape depends on the incident field (FLS lobe closure, RB-DTO mixture reduction) loses it, and multiplies its update cost by the number of lights or forces per-source states (explicit C3URX1 already pays ≈ 19 ms wall per added source).

**L6 — State-dependent rays cannot be cached; stateless re-trace avoids relation maintenance [ARITH].** If ray directions depend on radiance (lobe cubature, lobe-anchored probes), every sweep of every update must re-trace them, and a geometry-only dirty test cannot select them: an edit changes radiance, and hence lobes, everywhere. Conversely, such a design needs no cached relation, dirty test or XOp (6.5 ms kernel at N 1,270; 65.6 ms at 11,430 as recorded). That lever belongs to the stateless delivery strategy and is equally available to an explicit lobe twin.

**L7 — Timing-convention traps [MEAS + AUDIT].** (a) 69–81 % of wall is overhead. (b) Kernel time is recorded on two incompatible bases (Dr.Jit exec vs torch profiler, E5 §4 `nan` columns). An arm implemented in torch and its comparator in Dr.Jit are not comparable on "kernel" without a common timeline tool. (c) Compilation is excluded by convention. Variable-size representations (adaptive cells, variable deposits, sparse W with dirty-dependent nnz) can trigger Dr.Jit kernel-variant recompilation on every update (the WL91 21-s lesson); in deployment that is a real cost. (d) The ≈ 35 ms fixed cost per render call dominates every wall-basis spp ladder below about 64 spp.

---

## 3. Regime map: where a quality-matched advantage over explicit transport is possible

| regime | possible? | why | which candidates could use it |
|---|---|---|---|
| GM1 / CK2, whole image, any spp | **no** | L3: all table arms reach whole-image tiers at the same spp | none |
| GM1 sharp ROI, frame-dominated (≥ 16 spp) | yes, via floor only (≤ ≈ 1.6–1.75× measured between explicit variants) | L3; but learning-free arms (C3URX1 pooled, FC 8×32, ELT, analytic XQT twin) already capture it | only as learned share = floor(learned) − floor(best twin) at matched table cost |
| GM1, table-dominated (≤ 4 spp or edit → valid table), N ≤ 11 k | small (zero-cost ≤ 1.4–1.6 recorded; ≤ 2–2.6 engineered) | L1; learned cost must be ≤ 1.9 ms (N 1,270) | any learned role must cost less than the explicit q-half + 4 orders; per-edit generators fail (§4) |
| first sharp flight between separately moving bodies | **no** for learning | explicit sender-side deposition is ≈ R0 cost [MEAS WL89]; 10⁶ photons ≈ 0.3 ms kernel at 0.27 ns / ray | none (LLT, ECRO sender) |
| many lights, linear operator | yes (source-update lever, L5) | measured 3.5–3.9× | only exactly linear candidates (ECRO lineage, XQT, LIEM, LIEM-B); not FLS, not RB-DTO with mixture reduction |
| deep high-albedo glossy self-transport of rigid bodies | possibly | explicit cost ∝ J_int · N_b · K_int per incident change; learned ∝ incident samples · t_q | RB-DTO; break-even needs J_int · K_int ≳ 800–21,000 per patch (§4.D3) |
| long internal walks in media (≥ 10–30 events per exit) | possibly | walk ≈ events × 0.27 ns per exit; a 2·10⁴-FLOP network ≈ 2.5 ns | LIEM-S; D2's own toy puts angular content at noise there |
| N ≫ 10⁴ with dense learned message passing | **no** | N² memory (0.5 GB at 11 k, 40 GB at 100 k) | rules out ECRO / ECRO-FC generator as written |

---

## 4. Per-design, per-candidate critique

### 4.1 DESIGN_REVIEW (coordinator)

#### FLS — flux-allocated spatio-angular lobe state (primary). **Survives: NO.**

Fatal objections.
- **F-FLS-1. The learned operator replaces nothing expensive.** All of FLS's ray work (lobe cubature, J sweeps), the readout and the frame are explicit. Λ_θ replaces the element-local closure (ER evaluation of up to 32 incident components and their reduction to K lobes), which the explicit twin ELT computes from exact ER at about equal cost. FLS and ELT therefore share runtime cost by construction. A learned cost advantage is impossible. A learned quality advantage is limited to the closure error at fixed K, and ELT can buy that back by raising K: +8 lobes cost N · 8 · Q · J rays = 0.53 M rays ≈ 0.14 ms kernel at the measured 0.27 ns per ray [ARITH]. This is the brief's stop condition "the learned operator does not substantially replace or reuse explicit transport". FLS's delivery also *is* explicit lobe transport, which approaches the second stop condition (the sharp response effectively needs full explicit transport).
- **F-FLS-2. The selective-update claim is false (L6).** "Rays are re-traced for lobes whose cubature rays were dirty" cannot hold. Any edit changes the incident field, and therefore the lobes, everywhere. Lobes from moved and unmoved elements change, so their cubature rays change. Every update is a full re-trace of N · K · Q · J state-dependent rays. There is no reusable state after a rigid edit, only reusable weights.

Serious objections.
- **Linearity lost (L5).** FLS's outgoing lobe shapes depend on the incident distribution (§2 "not exactly additive"). That forfeits the only measured learned lever in the lineage, the source-only update. Mitigation (i), per-source states, multiplies the full update by the number of lights.
- **Under-specified incident reduction.** The set encoder is sized for "≤ 32 incident components", but on average N · K · Q / N = 104 point deposits land per element per sweep (more on panel-facing receivers), plus direct light. An extra nonlinear reduction step is needed, with its own closure error and cost; it is not in the contract.
- **Broad transport aliasing.** Q = 13 cubature nodes per lobe give 104 deterministic deposits per element per sweep, against 512 relation rays in EXP. For broad Lambert-heavy transport (most of GM1's 1,270 elements) this is a structured, frame-invariant error that cannot be averaged away. Equal broad quality may need K · Q ≥ 512, i.e. ≥ 5× the stated rays. That is still cheap to trace, but it removes the stated "equal to C3URX1" ray parity.
- **Readout cost unaccounted.** The camera reads K = 8 anisotropic spatio-angular lobes per sample (16.8 M samples per 64-spp frame) instead of one table fetch. The state (0.12 M floats) fits in L2, so this is compute-bound: about 8 GFLOP, ≲ 1 ms if fused [PRED]. In the Python / Dr.Jit prototype it is an unmeasured overhead on the term that dominates every lifecycle.
- **Exercised on ~3 % of elements.** In GM1 the sharp sender is the 6 × 7 = 42 panel patches out of 1,270 [CODE fixture]. The learned closure matters on very few elements, so any image-level effect sits in the 1 % R_mirror ROI where L3's ill-conditioning applies.
- **Dirac / near-mirror and occlusion-in-footprint (F1, F2).** These are acknowledged. On cost: adaptive Q at occlusion edges multiplies rays only where needed, which ELT can do identically.

Genuine strengths: a flux contract by construction (simplex with NULL); exact rigid equivariance of local frames; deterministic delivery removes single-draw heavy tails at the deployed budget; no relation maintenance (L6) — 6.5 ms kernel at N 1,270 and 65.6 ms at 11,430 avoided. But the ELT twin shares every one of these.

Stop conditions hit: "learned operator does not substantially replace explicit transport" (predicted with high confidence); "learned state cannot be reused after a rigid edit" (state: yes; rule: reused).

#### ECRO-FC (independent alternative). **Survives: NO as a candidate; usable only as a diagnostic H2 arm, and then confounded.**

Fatal objections (as a geometry-edit candidate).
- **Per-edit generator costs more than it replaces.** At 4 × 16 the generator is 4.2 ms kernel versus 3.96 ms of explicit work replaced (D3 already found U 0.91–0.97). At 8 × 32 the per-(patch, bin) heads, 91 % of the FLOP, grow 4×. Predicted generator ≈ 15.7 ms at N 1,270 and ≈ 99 ms at 11,430 [ARITH G]. Kernel edit → frame 64 U versus an explicit FC 8×32 comparator at matched state rays: **0.87 (N 1,270), 0.85 (N 11,430)**.
- **No rigid state reuse.** Inputs include world centre / normal / tangent and a scene-global mean token [CODE `model86.py:95, 99`; D1 finding], with dense message passing over all patches. Every factor of every patch is regenerated on every edit. The design keeps "ownership and visibility unchanged", so it keeps this.

Serious objections.
- **State rays per bin.** With K = 512 at 256 bins there are 2 state rays per bin instead of 8. Either the learned flight's per-bin sampling variance rises 4× (the E5 heavy-tail problem worsens), or K rises to 2,048. The latter is 4× relation, 4× dirty test (17.7 ms kernel recorded) and 4× re-trace. The design does not say which.
- **Internal inconsistency.** Candidate B is rejected because a "variable-support sparse output is awkward for a shared network", yet ECRO-FC's carrier is "adaptive 4×16 → 8×32 per element". Either the refinement is an explicit rule (state it), or the network predicts support (the rejected problem). Variable per-element sizes also risk per-update kernel recompilation (L7c).
- **H2 contrast confounded.** ECRO-FC vs ECRO + coverage changes the carrier, the passivity constraint, output capacity (4× heads) and possibly state rays at once. A carrier-only arm (FC, no passivity constraint, matched state rays per bin, matched compute) is needed to read H2.
- Dense N² `Ft`: 0.52 GB at 11 k; no path to larger scenes.

Strengths: the cleanest lineage-internal test of coverage plus carrier; exactly linear, so it keeps the source-update lever.

#### Rejected A (continuous kernel) and B (learned sparse multi-resolution): I agree with both rejections. For A, cost adds to the stated reasons: a neural decoder per state ray per flight (650 k evaluations at 1–20 ns) versus 650 k table fetches at 0.8 ns, i.e. 1.3–25× an explicit order before any generator (L2).

### 4.2 D1 (physics / measure first)

#### XQT — exact-sample quadrature transport (primary). **Survives: CONDITIONAL** (HD-1 admissibility, and S1-c showing that learned weights beat the analytic twin by a margin that matters in quality-matched cost). Predicted to stop at S1-c.

Serious objections (none fatal by contract, but they bound utility).
- **Per-flight cost is 8–26× an explicit order [ARITH H].**
  - Sparse apply with n_nb = 9 is 5.9 M nnz: ≈ 3.8 ms at the N 1,270 gather rate; 16.6 ms at N 11,430.
  - The indirect exact-ER sum over nonzero incident cells, for flights of order ≥ 2, adds 0.45–3.0 ms (N 1,270) and 4–27 ms (N 11,430), depending on the glossy-landing fraction and the cell count (64 / 256).
  - Total per flight: 4.3–13.7 ms against 0.53 ms (N 1,270); 21–73 ms against 2.3 ms (N 11,430).
- **The cost lever belongs to the twin.** XQT and its analytic-kernel twin have identical runtime cost. XQT's only lever is "fewer draws for equal error". Explicit pooling costs ≈ 0.76 ms kernel per extra draw, so XQT's per-flight overhead of 3.8–13 ms equals 5–17 pooled draws. XQT must therefore reach the quality of ≳ 5–17 explicit draws to break even at N 1,270. D1's own flatland scratch gives the twin 8–50× ray-equivalents, but the oracle kernel only 1.0–1.33× better error than the twin. Any positive cost result is therefore the twin's, and the learned share is ≤ 1.6× ray-equivalent, worth ≈ 0.5 draw ≈ 0.4 ms per update [ARITH, from D1's numbers].
- **Draw pooling over frames (L4).** The explicit comparator can pool its ≥ 16 draws over the steady-state frames after an edit at ≈ 3 % frame overhead. XQT's advantage window is therefore the first ~16 frames.
- **W regeneration and masking cost unquantified.** Dirty receptive fields spread the 7.7 % dirty rays over n_nb neighbours: 23–69 % of W entries are dirty after an edit, 0.7–6.5 ms of MLP evaluation at 20 kFLOP each [ARITH]. The visibility-consistency mask needs, per (ray, neighbour), a search of the neighbour's 512 records. That search has no stated data structure or cost.
- **Memory.** W (value + index) is 38–105 MB at N 1,270 and 0.34–0.95 GB at N 11,430, i.e. 1.8–5× the relation.

Strengths: information-honest (it reads every queried hit and invents none); linear, so it keeps the source lever (L5); flux-exact and passive by construction; rigid-invariant inputs; and the twin is pre-named. The S1-c oracle/twin ceiling is exactly the right cheap gate.

#### LLT — learned lobe transfer from sparse probes (alternative). **Survives: NO.**

Fatal objection: **there is no cost to amortise.** The twin (stratified photons from sharp senders with a fixed splat) traces K_p rays per lobe at 0.27 ns each. Raising its budget to 10⁶ photons per light costs ≈ 0.3 ms kernel. LLT can save only microseconds of tracing while adding |candidates| MLP evaluations per lobe. Its quality advantage must come from inferring unprobed visibility, i.e. lever (c) / implicit visibility, which is the brief's stop condition, and the twin removes the need for it at negligible cost. Its per-light cost is also the same as the twin's.

#### Rejected A1, A2, B1, C1, E0 (control), O1. I agree with all of them. On cost: A2 (learned gather) has the WL52 measurement (8–60× slower than explicit GPU C); O1 is the asset light field with α⁻⁴-type content per asset; B1 is the resolvent with the measured 1.02 / 1.04 ceiling.

### 4.3 D2 (learning / generalisation first)

#### LIEM-S — learned intrinsic exit measure with structured lobes (primary). **Survives: CONDITIONAL** (outside GM1; [HUM] HD2 / HD3 / HD7; the kill rule must be repaired).

Serious objections.
- **The cost lever exists only with long walks and a small network [ARITH L].**
  - The explicit walk costs ≈ events × 0.27 ns per exit.
  - LIEM costs one network evaluation plus ≤ 2 anchor rays.
  - With a 2·10⁴-FLOP network at 10 TFLOP/s: walk / LIEM = 0.27 / 1.07 / 3.2 / 5.0 at 2.5 / 10 / 30 / 47 events.
  - With 2·10⁵ FLOP: 0.03–0.62, so LIEM is slower at every event count in D2's toy.
  - D2's own toy puts the angular high-frequency content at noise exactly where walks are long (27–50 events). In the intermediate regime with structure (10–12 events) the per-sample cost is at parity at best.
  - D2's "10⁴–10⁵ parameters, µs-scale batched" must mean µs per batch. At µs per sample, LIEM is 100–1,000× slower than the walk.
- **Amdahl share is not gated.** I0.2 (b) gates on events per exit, which is a proxy. The gate must be the walk's measured share of the edit → frame lifecycle (an L1-style U0). A medium object occupying a few incident samples has a negligible share even at 50 events.
- **The kill rule is not falsifiable as written.** LIEM "survives" if any of (a) faster than Twin-W, (b) more accurate than Twin-D, (c) better memory than Twin-T without per-asset precompute holds.
  - (c) passes by construction whenever error is matched, since a shared model has no per-asset table and Twin-T is GB-scale.
  - (b) beats an *approximate analytic* rule, not explicit transport.
  - Only (a) is the brief's "advantage over explicit transport", so (a) must be necessary for a utility claim. (b) and (c) are secondary findings.
- **Sampling-based output in a deterministic solve.** LIEM-S emits samples, so it inherits per-sample variance inside FNTO's deterministic F/S solve, and its no-variance advantage (L4) is again a latency one.

Strengths: genuine rigid-invariant nonlocal intrinsic content; linear in the source; passive by construction; the I0.2 headroom gate and the three named twins are the right structure.

#### LIEM-B — hierarchical categorical exit distribution (alternative). **Survives: NO as a candidate** (keep it as the I1 representation-capacity control).

Fatal objection on cost: autoregressive coarse → fine prediction needs L sequential network evaluations per sample, i.e. 8 levels × 2–20 ns = 16–160 ns per sample against a walk of 0.7–13 ns at 2.5–47 events. The deterministic-gather form emits k coefficients per entry; k ~ 10³ at sharp exits [D2 §5.1: 1,024 leaves for lobe L1 ≈ 0.1] is a memory-bandwidth load no smaller than Twin-T's apply. D2 itself flags that it is not mechanism-independent.

#### Rejected R-A … R-E. I agree with all of them. R-D (rigid-cluster self-transport) is the same idea as D3's RB-DTO; D2's rejection reason (the twin is the cached, equally rigid-invariant explicit self-relation, so learning buys only sweep amortisation) is the systems argument I quantify against RB-DTO below.

### 4.4 D3 (systems / cost first)

#### RB-DTO — rigid-body directional transfer operator (primary). **Survives: CONDITIONAL** (ownership [HUM D1]; a regime must be shown by Stage 0 with the best explicit twin and symmetric counts).

Serious objections.
- **The regime map is inflated by an asymmetric comparison [ARITH K].** D3's `body_regimes` charges the explicit twin uniform K_int = 4,096–16,384 per patch per order at 0.81 ns. It charges the learned apply a fixed M_in = 20,000–50,000 incident samples per body. But the C_ext rays landing on a 400-patch body number ≈ 164,000 per outer iteration (N_b × 512 × 0.8), and D3's own §2.4 says the sharp single flight is cheap explicitly by sender-side deposition (≈ R0 cost, K ≈ 512).
  - Recount for D3's 20 bodies × 400 patches, J_int 8: sender-side twin K_int 512, landed incidence, J_ext 1–4, t_q 5–20 ns, t_gather 0.39–0.81 ns. Explicit intra-body 12.8–26.5 ms against learned apply 16–262 ms per update gives **U 0.16–1.21**, not 4.8–6.7.
  - At J_int 25: U 0.25–2.35.
  - Break-even per patch: J_int · K_int ≳ 790–21,000 [ARITH].
  - So a lever needs very deep, high-albedo glossy cavities, or incident aggregation.
- **Incident aggregation reintroduces P2 on the input side.** Breaking even at shallow depth requires aggregating incident samples per patch, e.g. to 64 directions. That is a binned incident field at the body's input, which re-creates the fixed-bin grazing bias of P2 for exactly the grazing internal incidence the design targets.
- **The C_ext delivery of sharp exit lobes reintroduces P3.** "C_ext sender rays … evaluate the cache at the exact continuous (x, ω)" is a receiver-anchored gather from a sharp sender, which is the CK2 X1 heavy-tail situation. Sharp exits need sender-side deposition from the cache, at a per-lobe cost that is not in L_rb.
- **Mixture reduction breaks linearity (L5).** "Reduced into a per-patch output cache: a mixture-reduced lobe set (≤ M_p lobes)" is nonlinear in the incident field. The source-update lever is lost; per-light caches cost ∝ lights.
- **The explicit twin also amortises.** The body's self-relation is rigid-invariant, so Twin 1 caches it for free under rigid motion (D3 says so). The only thing RB-DTO amortises is J_int sweeps per incident change, which is the WL96A resolvent role again, moved inside a body. Its ceiling then depends on J_int, which the workload generator chooses.

Strengths: the only design that puts a learning-free cost anatomy (Stage 0, U0 = L / (L − A_intra) with the best explicit twin) *before* any training, and explicitly forbids the uniform-binning twin. If Stage 0 is run as D3 specifies it, it will catch the inflation above. The certificate and path-tag ownership tests are well specified; convex-body N_b ≡ 0 is a good contract test.

#### LER-Δ — learned environment resolvent with explicit Δ iteration (alternative). **Survives: NO.**

Fatal objections.
- **Per-environment learned state.** Any static-environment edit invalidates H_E, and its remedy is retraining, which violates "without neural retraining" for that edit class. That is also the Native Reuse / 8DNA-RNA class, not one shared model.
- **H_E apply is not sparse.** Each Δ-iteration applies the all-order environment resolvent to a field defined on the whole environment (the edit changes the incident field everywhere). That is a learned evaluation per environment incident sample per iteration, I_Δ times. By L2 it costs more than the J explicit sweeps it replaces whenever the network costs more than r × 0.4–0.8 ns per entry. At GM1 the explicit sweeps cost 4 × 0.53 ms.
- **Negative flux in −B.** Passivity of the composite is not guaranteed and clamps are forbidden, so the fallback rate is an unbounded cost term.

#### Rejected A / B / C(patch) and others. I agree with all of them. D3's arithmetic for A (generator per edit ≥ replaced work) is confirmed (§4.1 ECRO-FC).

---

## 5. Defects in the falsification plans

### 5.1 Cross-cutting (all four designs)

1. **Quality-matching mechanism not stated.** No plan says that, in GM1, quality-matched utility is decided by the table floor seen through the frame (L3), that whole-image tiers are predicted to show no difference, and that ROI tiers near a floor are ill-conditioned. Required additions:
   - predeclared tiers away from every arm's floor, with an explicit NOT-REACHED rule;
   - per-arm spp ladders bracketing each tier;
   - time-to-tier reported as a distribution over ≥ 16 relation draws × ≥ 3 seeds;
   - bootstrap uncertainty that includes reference-half noise in the floor estimate.
2. **No steady-state draw-pooling convention (L4).** Utility must be defined as time-to-tier from the edit, with the explicit arm allowed to re-draw relations progressively across frames. Otherwise "no single-draw variance" advantages (FLS, XQT, LIEM-S) are over-credited.
3. **Kernel basis not unified (L7b).** Plans ask for "wall and GPU-kernel bases", but the recorded kernel numbers mix Dr.Jit exec and torch profiler. A common timeline (e.g. one GPU-busy measure for all arms) must be declared, or every arm must be implemented in the same runtime.
4. **Compilation accounting (L7c).** Report kernel cache misses per update in the deployed path. A recompile that happens on every edit is a cost, not an exclusion. This bites adaptive cells (ECRO-FC, FC-adaptive, LIEM-B), variable deposit sets (FLS) and dirty-dependent nnz (XQT).
5. **No de-overheaded explicit comparator arm.** All designs leave "engineer the explicit comparator first?" to [HUM]. On the recorded prototype the explicit side's relation maintenance (dirty test ≈ 7 ns per entry) dominates the kernel lifecycle at scale. An inefficient explicit comparator inflates every learned ceiling (L1 table: 1.29 recorded vs 1.53 engineered at N 11,430). The plans should include an engineered-dirty-test explicit arm, or at least report ceilings under both.
6. **Multi-light axis missing or optional.** The source-update lever (L5) is the only measured learned lever. Plans should report source → frame utility at 1 / 4 / 16 lights, so that nonlinear candidates (FLS, RB-DTO) pay their per-light cost visibly.

### 5.2 DESIGN_REVIEW

- The Stage-1 stop "optimal-fit lobes … do not beat FC 8×32 in delivery error at ≤ its cost" has no declared cost basis or state-ray count for FC 8×32 (K 512 = 2 rays per bin, or 2,048). With an overhead-dominated prototype the clause is undecidable.
- Stage 3 ("equal error → compare cost; equal cost → compare error") has no tiers, no ladders, no floor conditioning and no draw-pooling rule (5.1 items 1–2).
- The H2 contrast ECRO-FC vs ECRO + coverage is confounded (carrier + passivity + capacity + state rays).
- No scale axis. §6 already predicts no raw speed advantage in GM1, so Stage 4 can only find the L3 floor lever, which the learning-free arms share. Stage 4 should say so in advance.
- F8's "steady-state update cost ~ C3URX1" understates FLS (full state-dependent re-trace per sweep, L6, plus per-light states, L5).

### 5.3 D1

- S1-e defines "zero-cost replacement of its role" for XQT, but XQT's role is an estimator whose cost is not removable. The relevant ceiling is U(1 draw + XQT apply) against U(m pooled draws) at equal quality, with the twin at the same cost. Stated as written, the ceiling credits XQT with the twin's gain.
- The §6.6 decision mapping requires "beats its twin by the declared margin" in error only. It should also require that the learned share of quality-matched utility, U(XQT) / U(twin), exceeds a [HUM] margin.
- W-X (scale axis) is optional. Without it, D1 itself predicts the cost-side H3 is predetermined.
- LLT S1-d is well designed as an information test, but LLT fails on cost before S1-d matters (§4.2).

### 5.4 D2

- The I3 kill rule (§4.3) must make "faster than Twin-W at matched quality" necessary for a utility claim.
- I0.2 needs an Amdahl U0 gate on measured walk share, not just events per exit.
- The G0.3 ceilings use wall numbers from WL96A (§4). They should be recomputed on a unified kernel basis with the engineered dirty-test variant.
- Track I uses Cycles / Mitsuba references whose cost is large. The reference cost is fine as one-time cost, but it must be kept out of any "explicit comparator" timing (the baseline gate's PT-reference vs comparator distinction).

### 5.5 D3

- The prior regime map (§4) should be withdrawn or recomputed with symmetric counts (§4.4). As written it predicts U 4.8–48 from an asymmetric twin. Stage 0 is correct; the map is not.
- Stage 4 interpolates on B0's quality–cost curve, which is good, but has no draw-pooling convention (5.1 item 2) and no near-floor ill-conditioning rule.
- B0 must include sender-side deposition for *both* intra-body sharp transport and C_ext delivery of sharp exits, and RB-DTO's L_rb must include the sharp-exit deposition it needs (§4.4).
- The ECRO "O(N²)" statement should be qualified (§1).

---

## 6. Missed alternatives and missing arms

1. **Engineered explicit comparator (arm, not a candidate).** De-overheaded EXP / C3URX1 (kernel dirty test, fused orders, BVH refit), as the utility reference. Without it every ceiling is convention-bound (§5.1 item 5).
2. **MIS of receiver gather and sender-side deposition (learning-free).** This is the natural explicit competitor to XQT (estimator) and FLS (sender cubature): it combines the unbiased receiver gather with the glint-resolving sender splat. None of the designs lists it. D1's "photons, deterministic quadrature [HUM which]" and D2's E-SPL come closest.
3. **Progressive / temporally pooled explicit relations** (L4). This is the steady-state form of the "≥ 16 draws" remedy, at ≈ 3 % frame overhead.
4. **Learned importance sampling / guiding of the explicit estimator** (D3 rejects it as "not a representation"). It is unbiased by construction, reusable across rigid edits if conditioned on object-local geometry, and its value is directly in variance at fixed rays. It deserves an explicit out-of-scope ruling [HUM] rather than silent omission, because it is the learned component with the cleanest cost story at GM1 scale.
5. **Low-spp, table-dominated operating point** (1–4 spp frames plus denoising). This is the only GM1 operating point where table-cost ceilings exceed 1.25 at zero cost (L1). No plan includes it as a declared convention.

---

## 7. Can each plan detect a quality-matched cost advantage?

| plan | can it detect a learned (not twin) advantage? | blocking defects |
|---|---|---|
| DESIGN_REVIEW | partly: FLS vs ELT is paired (good) | no tiers / ladders / pooling rule; FC cost basis undefined; confounded H2; FLS cost is identical to ELT, so only floor differences can show |
| D1 | yes for XQT vs twin in error; not for utility share | S1-e credits the twin's gain to XQT; scale axis optional; no pooling rule |
| D2 | Track G yes (frontier plus ceilings); Track I partially | kill rule passes on (c) by construction; I0.2 lacks a U0 gate; wall-basis ceilings |
| D3 | yes in principle (Stage 0 before training, quality-cost interpolation) | prior map inflated; sharp-exit deposition missing from L_rb; no pooling rule; no near-floor rule |

---

## 8. Proposed verdict

**INSUFFICIENT_EVIDENCE.** This is not adoption and not experiment approval.

- For the scene86 / GM1 / CK2 family, my systems reading is **NO_DEFENSIBLE_CANDIDATE** (scoped). Every cost or quality-matched lever there (floor through the frame, stateless re-trace, deterministic delivery, draw efficiency) is available to a learning-free twin. Learned roles that regenerate factors per edit cost more than they replace on the kernel basis.
- Two candidates (RB-DTO, LIEM-S) have a *possible* regime outside the family. Symmetric recounting shows the regime is narrow: J_int · K_int ≳ 10³–10⁴ per patch, or ≥ 10–30 internal events with a ≤ 2·10⁴-FLOP network. Their own learning-free Stage-0 / I0 gates could establish or refute it, but those workloads do not exist yet, and the gates need the timing and comparator corrections of §5 before they could detect a learned share of utility.
- The coordinator's primary (FLS) and alternative (ECRO-FC) do not survive the cost lens.

[HUM] Every numerical threshold (U_min, tiers, ROI definitions, pooling horizon, engineered-comparator requirement, light counts) is a human decision. Nothing here derives them.
