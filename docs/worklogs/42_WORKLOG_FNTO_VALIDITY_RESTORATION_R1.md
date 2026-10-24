# Worklog — FNTO Validity Restoration: r1 Candidate, Gates 1–5 (2026-10-24)

**Verdict under protocol `fnto_r1` (predeclared, `r1_verdict.py` written before any result): NOT SUPPORTED**, triggered
by the static-representability rule (S_T0 | hist 24.49 dB vs RNA_T0 33.57 dB → S = −9.08 dB, class "fail"). Every other
decisive property that Batch 6 could not measure was measured this time: the composed solve converged on 64 / 64
held-out views (ρ_T0 = 0.68 for the passivity-constrained kernel K_r1P), the state-only edit update removed 65 % of the
stale error that a T1-retrained RNA removes (R_RNA = 0.65, F = 2.44, kernel parameters bit-identical), and the signed
stale→updated change correlates 0.99 with the reference change on A_changed (slope 0.95, explained fraction 0.97).
The −9 dB gap is the intrinsic kernel's own first-event fidelity (K alone vs first-event references 25.05 dB; the
composition S_T0 | r1 25.00 dB loses nothing on top of it). Gates 1 (two criteria), 2b, 2c and 2d failed as declared and
are carried in the verdict. Nothing in Batch 6 (`results/fnto_lego/`, `results/evaluation/41/`, worklog 41, protocol
`fnto_v1`) was modified. Reviewer artefacts: `results/evaluation/42/` (manifest with sha16 per file).

Authorisation: user brief "ReST-GI — FNTO Validity Restoration, Post-Batch-6 Controlled Implementation & Re-evaluation"
(2026-10-23). Scope: verify the six findings of the independent Batch 6 validity audit against code and evidence, correct
confirmed defects in a distinct candidate (r1) under a distinct predeclared protocol (`fnto_r1`), re-evaluate through
Gates 1–5, report. Not done, by instruction: no other architecture, no SPT-1 change, no T1 tuning, no replacement
recommendation. Start commit 7ba0848; protocol commit cc2827d (before any r1 training or held-out evaluation).

Evaluation regimes used in this worklog (never mixed without the label): **hist** = the worklog-37 held-out references
(Cycles, mean of seeds 1/2, 4096 spp, `sample_clamp_direct` 20 / `sample_clamp_indirect` 5, point-sampled, PC);
**r1** = reference regime r1 (same cameras / suns / pixels, no clamp, point-sampled, 4096 spp, LabServer63);
**first_r1** = intrinsic-all first-event references (free-space paths absorbed, glass internal paths kept, 1024 spp);
**probes** = r1 Stage A held-out probe images (512², 512 spp).

---

## A. AUDIT FINDINGS → VERIFICATION → CORRECTION

The audit document itself (`BATCH6_INDEPENDENT_FNTO_VALIDITY_AUDIT.md`, 23 numbered items grouped into six findings)
cited scripts in an `audit_checks/` directory that was not present; every finding was therefore re-measured on the
unchanged Batch 6 code (commit 7ba0848) with new scripts (`experiments/fnto_lego_r1/r1_audit_checks.py`,
`r1_render_checks.py`; results `results/fnto_lego_r1/audit/audit_checks.json`, `render_checks/render_checks.json`).

| # | Audit finding | Verification on Batch 6 (measured) | Status | r1 correction | Evidence (r1) | Limitation |
|---|---|---|---|---|---|---|
| F1 | Port origins buried / off the surface; incident lighting biased | 28.6 % of origins > 1 mm from a surface; 20.3 % first hit along +n is an inside face; port-vs-pixel sun-visibility agreement 0.871; pooled port/pixel irradiance 0.862 | CONFIRMED | Clusters keyed by (1.2 cm voxel, 3 mm plane bin, material, connected component); origin = medoid surface sample; 4 surface samples per port; pixels and gather hits attached by triangle identity | `gate1/gate1.json`: 99.998 % within 0.2 mm; inside-face 4.93 %; agreement 0.962; irradiance 1.001 | Inside-face 4.93 % is enclosed geometry (surfaces inside another component's closed solid: 4.933 % other component, 0.002 % same); agreement < 0.97 (residual: 6.3 % of lit-pixel interior ports read dark, 2.4 % the reverse; port resolution) |
| F2 | Insufficient independent spatial excitation of K | Training-field effective rank 22 (90 %) / 464 (99 %) in a 38,090-port basis; one-bounce indirect field 52 % outside the full span | CONFIRMED | Probe families: sun × masks 25 %, spot 35 %, broad multi-point 25 %, multisun 15 %; 1,000 train + 120 held-out, independent configurations; light calibration (`r1_calib`) | `gate2/coverage.json`: rank 317 / 580 (common basis; Batch 6 13 / 254); indirect field outside full span 0.22 (common) / 0.24 (r1 coarse basis) vs Batch 6 0.28 | Gate 2c threshold 0.10 NOT met (see D); held-out probe fields median 7.6 % outside span, q90 48 % |
| F3 | Effective kernel support narrower than declared | Band kNN lists full at 1.4 cm (declared 5 cm); mid at 5.0 cm (declared 30 cm) | CONFIRMED | Representatives over the full ball with represented-area weights: band = 1.5 cm aggregates, mid = 8 cm aggregates; amendment 1: K/2 nearest exact + orientation-stratified area-proportional picks, K 96 / 128 | `anchors.json`: reach median 4.996 / 29.99 cm (Gate 2a PASS); `gate2/quadrature*.json` | Gate 2b (quadrature accuracy on the real shadowed sun field) FAILS at a0 (0.62 / 0.45) and a1 (0.27 / 0.20) vs 0.15 — see D |
| F4 | Incomplete internal reflection / directional transport | Batch 6 "first-event" masking dropped glass internal reflection / TIR: glass cube centre 0.020 vs 0.046 with all intrinsic paths; Lego-wide +0.06 % | CONFIRMED | Intrinsic-all masking: camera ray OR backfacing inside glass / refraction materials; max bounces 128 | `render_checks.json` glass; Stage A r1 and first_r1 references rendered with it | Directional bandwidth of K_loc (2 GGX lobes) and the 19-bin band representation are unchanged (declared approximations) |
| F5 | Reference clamping and reconstruction-filter mismatches | Blends carry `sample_clamp_indirect` 5 in addition to the generator's `sample_clamp_direct` 20; clamping removes 6–8 % energy and breaks two-sun superposition (−8.8 % signed; unclamped −0.05 %); Stage A / first-event renders used the blend's Gaussian 1.5 px filter while the held-out references are point-sampled | CONFIRMED | Reference regime r1 (no clamp) and point-sampled probes / references. Amendment 2: in this Cycles build `BOX` ignores `filter_width` (BOX 0.01 = 1 px box); `GAUSSIAN` 0.01 point-samples — the first r1 renders (1 px box) were discarded and re-rendered | `render_checks.json` clamp; `r1_filter_check.py` result in protocol amendment 2; untraced pixels 11.2 % → 0.54 % after the re-render | Historical references stay clamped by design (reported separately as regime hist) |
| F6 | Signed change, solver stability and coupling compression not evaluated; "ties" misdescribed | 593/593 Batch 6 tie pairs map to ports of *different* materials (not physically equivalent); the Batch 6 divergence check could not flag slow divergence (ρ = 1.05 → "unconverged" after 100 iterations); no signed Δ | CONFIRMED | Divergence = median of the last 5 residual ratios ≥ 1 after iteration 8 or non-finite; spectral radius by power iteration (Gate 4c); signed change Δ_pred vs Δ_ref on A_changed and stationary pixels; E_relfull arm measures the tie effect; incremental mismatches re-classified | `tests/solver_detection.json` (ρ 0.8 → converged, 0.999 → unconverged, 1.05 / 1.5 → diverged); `relations.json`: 1,435 mismatches, all equal-distance, 0 share material + component; `eval_K_r1P`: E_upd and E_relfull differ by 8e-6 dB (view mean), identical to 3 decimals in every region | Coupling is a dense diagnostic table (109 M rays), not compressed: no efficiency claim |

---

## B. r1 CANDIDATE — IMPLEMENTATION FACTS

Code: `experiments/fnto_lego_r1/` (new; imports Batch 6 modules read-only). Representation contract unchanged
(`docs/FNTO_ARCHITECTURE_CONTRACT_V0_1.md`; `fnto_model.DomainKernel` unchanged: K_loc diffuse + 2 GGX, K_glob rank 64,
K_band 19 bins, K_mid; linear in radiance, non-negative).

- **Ports:** 852,358 Lego + 1,702 box ports (`r1_anchors.py`, built on LabServer63); 9.22 M surface samples; 1,872
  connected components (682 closed). `anchors.npz` a0 sha16 `0a93199fc3fe49f3`; a1 (amendment 1 representatives only)
  `32aa4032280225b7`.
- **Support:** band sources 267,751 aggregates (1.5 cm), mid sources 8,980 coarse ports (8 cm); per port 96 band and 128
  mid representative slots (amendment 1; a0 had 48 / 64 single uniform picks).
- **Coupling C:** `r1_relations.py`, 128 cosine gathers per port over 4 surface samples, 109.3 M rays per state, hits
  attached by triangle identity, inside faces blocked (−2), misses dark (−1). Full build 92 s per state (PC, Mitsuba);
  incremental T0→T1: 689,172 rays selected (0.63 %), 54,160 rows re-traced, 41.9 s; 1,435 entries differ from a full
  rebuild, all at equal hit distance, none sharing material and component. Learned coupling rule C_φ: empty (declared).
- **Stage A r1 (LabServer63, bpy 3.5.1 CPU):** intrinsic-all paths, no clamp, GAUSSIAN 0.01 px (point sampling), 512²,
  512 spp; 120 held-out (seed 2102) + 1,000 train (seed 1101); families sun × masks / spot / broad / multisun; box random
  or absent. Light radiometry from `r1_calib`: sun S → E = S; point / spot power P → I = P/(4π) W/sr.
- **Preprocessing (PC, Mitsuba):** pixels by triangle identity (0.54 % untraced and dropped), per-light visibility bits,
  per-port per-light visible-sample counts (`port_vis.npy`, 20.5 GB); 62.5 M training pixels, 7.34 M held-out.
- **Training (`r1_train.py`, LabServer63 RTX 3080 Ti):** 24,000 steps, 4 images × 16,384 pixels, L2 on log(1 + L),
  Adam 1e-3 cosine → 1e-5, seed 0, final weights; predict in 16,384-pixel chunks with activation checkpointing (memory
  only). K_r1: 4.35 h, peak 5.77 GiB, 6,531,047 + 202,335 parameters, held-out probe monitor 23.3 → 24.4 (step 16k) →
  23.98 dB final. Conditional variant K_r1P (`--passivity 1.0`, hinge on uniform-light effective albedo > 0.98 over
  32,768 sampled ports per domain and step; otherwise identical) was triggered by ρ_T0(K_r1) = 10.47: 5.35 h, peak
  8.27 GiB, monitor 23.6 → 24.6 (step 16k) → 24.36 dB final. Both kernels are frozen for every evaluation arm.
- **Solver (`r1_solve.py`):** Jacobi on per-port incident samples; converged at relative change < 1e-3; diverged when the
  median of the last 5 successive-residual ratios ≥ 1 after iteration 8 or values are non-finite; unconverged at 100.
- **References r1 (`r1_refs_render.py`, LabServer63):** full GI max_bounces 32, 4096 spp; first_r1 1024 spp; same
  cameras / suns / pixel lists as worklog 37; GAUSSIAN 0.01 (amendment 2); rendered 12:28–19:25 KST 2026-10-24 (full
  ≈ 680 s per view, first ≈ 97 s); alpha agreement with the historical masks 99.91 % (full) / 99.95 % (first).
- **Protocol:** `experiments/fnto_lego_r1/protocol/fnto_r1.json` (cc2827d) with amendments 1 (cc2827d) and 2 (4d5525d),
  both recorded before any r1 training; verdict rules fixed in `r1_verdict.py` before results.

### Amendments (what changed, why, what did not change)

1. **Representative quadrature (after Gate 2b FAIL at a0, before training).** Attribution on synthetic fields
   (`gate2/quadrature_attrib.json`): aggregation 3–7 %, uniform-pick bias ≤ 8 %, single-sample variance 25–40 % (the
   dominant part). Change: K/2 nearest aggregates exact, the rest in orientation-stratified equal-count strata with
   area-proportional picks (unbiased; `tests/test_r1_reps_a1.py` PASS); K chosen as the smallest swept value with
   sampling error ≤ 0.075 on the adverse synthetic field (band 96, mid 128; `gate2/quadrature_ksweep.json`). The
   declared Gate 2b field (real T0 view-6 sun with shadows) was not used to choose K. Sources, radii, bins, kernel code,
   thresholds unchanged.
2. **Film (after the first server render chain, before training).** 11.2 % of held-out pixels were untraceable;
   99.5 % lay on depth edges with AOV positions between two surfaces. `r1_filter_check.py`: BOX 0.01 renders the same
   image as BOX 1.0 (1 px box; fractional alpha 0.46 % of pixels), GAUSSIAN 0.01 point-samples (0.012 %). Also found:
   integrator / film settings changed after a process's first render are not re-synced in this bpy build (all
   production scripts set them before the first render; the per-image Cycles seed is therefore the first image's seed
   for a process). Change: GAUSSIAN 0.01 in `r1_stagea_render.py` and `r1_refs_render.py`; Stage A and all four r1
   reference sets re-rendered with identical configurations and seeds; the 1 px box outputs kept as `stagea_box1px/`,
   `refs_box1px/` and not used. Cost: ~5 h of server CPU.

---

## C. GATES — MEASURED SO FAR (regime labels in brackets)

| Gate | Criterion (predeclared) | r1 result | Batch 6 | Verdict |
|---|---|---|---|---|
| G1 origin | ≥ 99 % of origins within 0.2 mm | 99.998 % | 74.8 % | PASS |
| G1 inside face | < 1 % first hit along +n inside | 4.93 % | 8.75 % | FAIL (enclosed geometry) |
| G1 agreement | port vs pixel sun visibility ≥ 0.97 | 0.962 | 0.872 | FAIL (marginal) |
| G1 irradiance | pooled port/pixel in [0.95, 1.05] | 1.001 | 0.862 | PASS |
| G2a reach | median ≥ 0.98 R | 4.996 / 29.99 cm | 1.4 / 5.0 cm | PASS |
| G2b quadrature | median rel. err ≤ 0.15 (band d 1 cm, mid d 10 cm; real view-6 sun) | a0 0.62 / 0.45 → a1 0.27 / 0.20 (energy-weighted 0.23 / 0.24; estimate exactly 0 for 4.1 % / 0.1 % of receivers) | not applicable (support 1.4 / 5 cm) | FAIL |
| G2c coverage | indirect field ≤ 0.10 outside the training span (r1 coarse basis) | 0.24 (rank 323 / 587 / 795) | 0.52 (B6 basis), 0.28 (common basis) | FAIL |
| G2d kernel fidelity | held-out probes ≥ 28 dB AND K_only vs first_r1 ≥ 28 dB | probes: K_r1 23.98 dB, K_r1P 24.36 dB (sun 24.3 / spot 24.8 / broad 23.0 / multisun 26.5; yellow 21.1, glass 19.7, reflective 29.8, box 18.4); K_only vs first_r1: K_r1P 25.05 dB (T0) / 25.00 dB (T1), K_r1 25.14 / 25.06 | 23.7 dB probes / 22.0 dB first-event | FAIL |
| G3a linearity | two-sun superposition, no clamp, \|signed\| ≤ 1 % | −0.05 % (clamped: −8.8 %) | clamped only | PASS |
| G3b light table | box albedo ratio family medians within ± 3 % | max deviation 0.7 % (sun 0.715, broad 0.722, multisun 0.717, spot 0.714; spot: 272 px, 1 image) | — | PASS |
| G4a incremental | only equal-distance differences; \|E_upd − E_relfull\| ≤ 0.05 dB | 1,435 equal-distance, 0 same material+component; output gap 8e-6 dB | 615 "ties", wrongly called equivalent | PASS |
| G4b detection | solver flags ρ 1.05 | PASS (9 iterations) | FAIL (100 iterations, "unconverged") | PASS |
| G4c passivity | ρ_T0 < 1 (power iteration, 40 it.) | K_r1: ρ_T0 = ρ_T1_inc = **10.47** (uniform-light albedo q50 0.48 / q90 7.5 / q99 18.7, 42 % of Lego ports > 1) → FAIL; K_r1P (`--passivity 1.0`, otherwise identical): ρ_T0 = ρ_T1_inc = **0.682** (albedo q99 0.55, 0 % > 1) → PASS; Gate-5 kernel = K_r1P | ρ ≈ 2.1 (K_glob), explosive (K_mid) | FAIL (K_r1) / PASS (K_r1P) |
| G5 | S, F, R_RNA, corr, slope, all views converged (kernel K_r1P) | S = 24.49 − 33.57 = **−9.08 dB → fail**; F = 2.44 (≥ 1.5); R_RNA = 0.65 (< 0.7, ≥ 0.3); corr 0.987, slope 0.948 (hist, A_changed); 64/64 converged in 6–7 iterations | 64/64 diverged; all unmeasurable | NOT SUPPORTED (static rule) |

**K_r1 end-to-end (descriptive; the non-passive kernel, run for attribution only; `eval_K_r1/eval.json`).** All 64
coupled solves (S_T0, D_stale, E_upd, E_relfull × 16 views) diverged, detected at iteration 6 by the non-finite rule;
the successive-residual ratios were 7.9 → 10.0 → 10.3 → 10.4 before overflow, i.e. the solve grew at the measured
spectral radius (10.47) from the first iteration. The fp16 glossy cache was never clamped (max 40.8). **K only, coupling
off, vs first_r1:** 25.14 dB on held-out T0 and 25.06 dB on T1 (view mean; pooled 24.95 / 24.89; Batch 6: 22.0 dB).
Per material (T0, display PSNR): Brown 32.9, Black 30.2, Light_brown 28.9, Sand 27.6, Yellow 25.2, Gray 23.6,
Red_Glass 21.5, transparent 16.1, box 16.3. Lit-pixel MSE 1.46e-3 vs shadowed 4.71e-3. Eval wall-clock 11.5 min
(solve 5.8 s per diverging view, render 0.35–0.42 s per view), peak 9.3 GiB; kernel parameters bit-identical across
arms. The a2 references agree with the historical alpha masks on 99.91 % (full) / 99.95 % (first) of pixels (the
1 px-box renders: 99.76 %).

**K_r1P end-to-end (Gate 5; `eval_K_r1P/eval.json`, `verdict.json`).** View-mean display PSNR in dB; "hist" and "r1"
are the two reference regimes; historical arms from worklog 37 are read from its stored `eval.json` (hist regime,
all alpha > 0 pixels; FNTO rows on the 99.76 % identity-traced pixels — the declared all-pixel row with nearest-traced
fill differs by 0.06 dB: S_T0 24.43, D_stale 23.65, E_upd 24.31).

| arm (K_r1P frozen in all arms; parameters bit-identical) | hist | r1 | A_changed (hist) | box B (hist) |
|---|---|---|---|---|
| K only, coupling off, vs first_r1 | — | 25.05 (T0) / 25.00 (T1) | 26.13 (T1) | 16.1 / 16.4 |
| S_T0 — held-out T0 | **24.49** | 25.00 | — | 18.8 |
| D_stale — T1 inputs, T0 state (stale) | 23.70 | 24.15 | 21.97 | 16.3 |
| E_upd — T1 inputs, incremental state update (0.63 % of rays) | 24.37 | 24.88 | **25.83** | 18.5 |
| E_relfull — full T1 relation rebuild (consistency check) | 24.37 (Δ 8e-6) | 24.88 | 25.83 | 18.5 |
| RNA_T0 / RNA_T1 (worklog 37, T1 retrained) | 33.57 / 33.50 | — | 32.23 | — |
| Surface-Owned SO_T0-PU (stale) / A_full-PU (T1 retrained) | 30.72 / 33.38 | — | 23.06 / 33.66 | — |

Rule values: S = 24.49 − 33.57 = −9.08 dB (fail); F = E_stale / E_upd on A_changed = 6.36e-3 / 2.61e-3 = 2.44;
R_RNA = (E_stale − E_upd) / (E_stale − E_RNA_T1) = 0.65 (E_RNA_T1 5.98e-4); every one of the 64 coupled solves
converged (6–7 iterations; successive-residual ratios 0.15 → 0.50; ρ_T0 = ρ_T1_inc = 0.682 from the eval's own power
iteration). Regions of E_upd | hist (D_stale in brackets): A_changed 25.83 (21.97), A_weak 23.19 (23.10), A_unchanged
24.59 (24.60), A_uncorresponded 26.59 (24.38), A_vis_changed 26.64 (19.95), B 18.52 (16.33). Materials of S_T0 | hist:
Brown 35.6, Light_brown 31.9, Sand 29.3, Black 27.5, Gray 25.0, Yellow 22.6, box 18.8, transparent 18.2, Red_Glass 16.5;
lit pixels 27.6 dB, shadowed 22.7 dB; per-view S_T0 | hist 22.5–27.2 dB. The composition loses nothing relative to the
kernel alone: S_T0 | r1 25.00 vs K_only | first_r1 25.05.

**Signed change (hist regime; A_changed 286,389 Lego pixels present in both views; stationary 3.34 M).** Reference
noise σ (MAD of the stationary pool) 1.5e-4 display units; 268 k A_changed pixels change by more than 3σ.

| Δ = arm(T1) − S_T0(T0) vs Δ_ref | corr | LS slope | sign agreement (\|Δ_ref\| > 3σ) | explained fraction |
|---|---|---|---|---|
| E_upd, A_changed | **0.987** | **0.948** | 0.94 | 0.973 |
| D_stale, A_changed | 0.948 | 0.838 | 0.42 | 0.894 |
| E_upd, stationary | 0.34 | 0.34 | 0.68 | −0.36 |
| D_stale, stationary | 0.07 | 0.007 | 0.33 | 0.005 |

(r1 regime, A_changed, E_upd: corr 0.982, slope 0.939, explained 0.963.) D_stale already follows the reference in
correlation because the pixel's own T1 sun visibility is an input of every arm (direct shadows and the moved box's own
pixels); what the state update adds is the indirect part: sign agreement 0.42 → 0.94, explained fraction 0.89 → 0.97,
slope 0.84 → 0.95. On stationary pixels the predicted change is weakly related to the reference change, whose
magnitude there is close to the noise (1.19 M of 3.34 M pixels above 3σ).

**Costs (K_r1P, LabServer63 RTX 3080 Ti unless noted).** Kernel training one-off 5.35 h (K_r1 4.35 h), peak 8.3 GiB;
kernel tables incl. 854 k × 128 mid weights: inside the 9.3 GiB eval peak; relation full build 92 s per state (PC,
Mitsuba, 109 M rays); incremental T0→T1 update 41.9 s CPU for 689 k selected rays (0.63 %; dense diagnostic table, no
efficiency claim); port sun visibility per view (eval_vis timings in `eval_vis.npz`); solve 6.0–6.2 s per view
(6–7 iterations ≈ 0.9 s each); pixel render 0.5 s per view (≈ 243 k pixels); eval of 64 solves + 32 renders 11.8 min;
learned parameters 26.9 MB (6.73 M, fp32); dynamic incident state 854 k × 128 × fp16 RGB = 656 MB plus the 656 MB
delta-sun glossy cache.

Gate 2b after amendment 1 on the synthetic fields: band 0.05–0.12, mid 0.04–0.08 (`gate2/quadrature_attrib_a1.json`);
on the real shadowed field the remaining error is dominated by far strata whose single area-proportional pick misses the
few lit sources (q90 of the band error = 1.0). No further quadrature change was made: choosing again on the declared test
field would be tuning on that field. Gate 2c: the r1 families raise the independent excitation by ~25× in effective rank
and reduce the indirect field's out-of-span energy, but not to the declared 0.10; classified as a probe-design (data)
limit, not an implementation defect; the render was not changed.

---

## D. BEFORE / AFTER (Batch 6 → r1)

| Property | Batch 6 (worklog 41 protocol fnto_v1) | r1 (protocol fnto_r1) |
|---|---|---|
| Port origins within 0.2 mm of a surface | 74.8 % | 99.998 % |
| Port origins whose +n first hit is an inside face | 8.75 % (20.3 % by the audit's 1 mm test) | 4.93 % (enclosed geometry) |
| Port vs pixel sun-visibility agreement / irradiance ratio | 0.872 / 0.862 | 0.962 / 1.001 |
| Pixel → port attachment | nearest normal-compatible port | triangle identity (0.54 % untraced, dropped) |
| Realised band / mid support (median) | 1.4 cm / 5.0 cm | 4.996 cm / 29.99 cm |
| Representative quadrature error (real field) | — | band 0.27, mid 0.20 (FAIL vs 0.15) |
| Training incident fields | 1,400, sun × masks only | 1,000 train + 120 held-out; sun / spot / broad / multisun |
| Effective rank 90 % / 99 % (common 8 cm basis) | 13 / 254 | 317 / 580 |
| Indirect one-bounce field outside the training span (common basis) | 0.28 | 0.22 |
| Glass internal reflection / TIR in intrinsic targets | missing (cube 0.020) | present (0.046) |
| Probe / reference clamping | direct 20, indirect 5 (held-out); Stage A clamped | none (r1 regime); hist kept and labelled |
| Probe / reference pixel filter | Gaussian 1.5 px probes vs point-sampled references | GAUSSIAN 0.01 (point) everywhere (amendment 2) |
| Renderer two-sun superposition | −8.8 % signed (clamped) | −0.05 % (unclamped) |
| Incremental-update differences | 615 "ties" called equivalent | 1,435 equal-distance, 0 same material+component; output effect 8e-6 dB (E_upd = E_relfull) |
| Static fidelity S_T0 \| hist (RNA_T0 33.57) | unmeasurable (diverged) | 24.49 dB (S = −9.08) |
| Edit recovery F / R_RNA (A_changed, hist) | unmeasurable | 2.44 / 0.65 |
| Slow-divergence detection (ρ 1.05) | not flagged | flagged at iteration 9 |
| Spectral radius gate | none (audit: ρ ≈ 2.1 / explosive) | ρ_T0 = 10.47 (K_r1) / 0.682 (K_r1P with the declared passivity hinge) |
| Uniform-light effective albedo, Lego domain (q50 / q90 / q99, share > 1) | 26 % of ports > 1 for K_mid, 18 % for K_glob (worklog 41) | K_r1 0.48 / 7.5 / 18.7, 42 % > 1; K_r1P 0.34 / 0.41 / 0.55, 0 % |
| Signed Δ evaluation | none | A_changed: corr 0.987, slope 0.948, sign agreement 0.94, explained 0.97; stationary: corr 0.34 |
| Kernel-only first-event fidelity (vs first-event references) | 22.0 dB | 25.05 dB (K_r1P), 25.14 dB (K_r1) |
| Held-out composed solve | 64/64 diverged (non-finite after 5–6 it.) | K_r1: 64/64 diverged at it. 6 (ρ 10.5); K_r1P: 64/64 converged in 6–7 it. (ρ 0.68) |

---

## E. CORRECTIONS TO WORKLOG 41 (append-only; worklog 41 is not edited)

Worklog 41's verdict (NOT SUPPORTED, 64/64 diverged) stands. The following statements in it are now known to be
inaccurate or incomplete; the corrected statement and its evidence are given.

1. "K_band: 32 ports within 5 cm" / "K_mid: 64 coarse ports within 30 cm" (§B). The nearest-neighbour lists were full
   long before the radius: realised support median 1.4 cm and 5.0 cm (`audit/audit_checks.json` F3_support). The
   declared radii were not the realised radii.
2. "divergence is detected" (§B, state solve). The Batch 6 rule flagged only non-finite values; a slowly diverging
   operator (ρ = 1.05) ran 100 iterations and was reported as "unconverged" (`audit_checks.json` F6_solver_detection).
   The 64/64 "diverged" classification of worklog 41 is unaffected (those solves became non-finite after 5–6 iterations).
3. "equal to a full rebuild except 615 ties (identical hit distances)" (§D) and "incremental update tie-equivalent"
   (§F). Of the 615 differing entries 593 map to ports on both sides, and in 100 % of those the two ports have
   different materials (`audit_checks.json` F6_ties). Equal hit distance does not mean the same surface; the
   incremental update was not shown equivalent in output.
4. "Isolated-domain first-event probe renders" (§B) and "declared first-event decomposition" (§C). The Batch 6
   masking excluded glass internal reflection and total internal reflection, which are intrinsic paths under the K/C
   contract (glass cube centre radiance 0.020 vs 0.046 with all intrinsic paths; Lego-wide +0.06 %;
   `render_checks.json` glass). The Stage A and first-event renders also used the blend's Gaussian 1.5 px pixel filter
   while the held-out references are point-sampled (audit item 9), so probe targets and references had different
   reconstruction footprints.
5. "the renderer itself departs by 8.5 % rel L1 (MC noise and the direct clamp)" (§C, superposition). With the
   generator's clamps the two-sun superposition deviates by −8.8 % in signed energy; with clamping off by −0.05 %
   (`render_checks.json` clamp): the deviation is the clamp, not Monte-Carlo noise. In addition the held-out references
   carry `sample_clamp_indirect` 5 from the blend (removing 6–8 % of energy), which worklog 41 did not state.
6. "Lego relations: hit→port q99 9 mm" (§F) was reported without the port boundary state: 28.6 % of Batch 6 port
   origins lie > 1 mm from any surface and 20.3 % first hit an inside face; port-vs-pixel sun-visibility agreement was
   0.871 with a pooled irradiance ratio of 0.862 (`audit_checks.json` F1, `gate1/gate1.json` batch6). The Batch 6
   incident field was therefore biased at the ports before any kernel was applied.
7. "1,400 sun-coherent fields" (§F, D3) is now quantified: effective rank 22 / 464 (90 / 99 %) of 38,090 coarse
   ports; 52 % of a one-bounce indirect field lies outside their full span (`audit_checks.json` F2). The D3
   interpretation of worklog 41 stands.

---

## F. PROCESS RECORD (for reproducibility and for the user's complaint about delay)

- 2026-10-23 09:30–12:00 KST: audit verification on Batch 6 code (PC). 12:00–18:50: r1 ports, mapping, Gate 1,
  calibration, relations, Stage A renderer, prep / train / solve / eval code; render chain launched 18:57 on LabServer63.
- 19:08: Gate 2b run **on the PC while the server chain was running** — a violation of the standing no-cross-machine-
  parallel rule (recorded in the protocol; the check read no rendered data). 19:50: protocol + amendment 1 committed.
- 2026-10-24 06:33: first render chain done (12 h). 06:47: holdout prep showed 11.2 % untraced pixels → filter check
  → amendment 2; Stage A re-rendered 07:00–11:53 (the ~5 h loss would have been avoided by a two-image check of the film
  before the chain). 11:55–12:28: Gate 2b (a1), prep, Gate 2c, eval prep on the PC (server idle). 12:28: a2 references
  started; 12:38: K_r1 training started; 12:40: server orchestrator `r1_after_train.sh` started (rho → optional K_r1P →
  wait refs → eval), so later steps do not depend on a session.
- 13:00–16:50: a six-lens code review of the never-executed production solver / eval (session workflow; its verifier
  stage was lost to the usage limit and every finding was re-checked by hand) → amendment 3 (`e081bff`): the
  `band_sender` einsum crash, the 12.3 GiB residual-norm transient, fp16 glossy-cache guard, NaN-robust pooling,
  pixel-count recording, MAD noise threshold, partial writes. 16:58:53: training done; the orchestrator's first
  `r1_rho.py` had imported the old module and crashed on the einsum defect at 16:59:24 (fixed file landed 16:59:20);
  relaunched from the rho step 16:59:59. ρ(K_r1) = 10.47 → K_r1P trained 17:01–22:23 (step 0.92 s with the hinge).
  19:25: a2 references done. 22:25–22:36 eval_K_r1, 22:36–22:48 eval_K_r1P (all unattended); 22:5x verdict and
  exports on the PC.
- Commits: cc2827d (protocol, amendment 1), 4d5525d (amendment 2), eaa7e03 (Gate 3b), 69c6e1d (orchestrator, rho),
  b52ee7d (verdict / exports / eval views / draft), e081bff (amendment 3), and the closing commit of this worklog.

---

## G. FINAL REPORT (format required by the brief)

### IMPLEMENTATION FACT
See §B. Representation and K/C ownership unchanged from the contract; what changed is the boundary state (ports),
the realised support and its quadrature, the probe design and radiometry, the reference regime, the solver's
divergence rule and the evaluation (signed change, spectral radius, E_relfull). Coupling remains a dense diagnostic
table; no compressed or learned coupling exists in r1.

### MEASUREMENT
Gates 1–4 as in §C; Gate 5 arms, rule values, signed change and costs as in §C ("K_r1P end-to-end"). Summary (dB,
view mean, hist regime unless stated): S_T0 24.49 (r1 regime 25.00; all-pixel row 24.43); RNA_T0 33.57 → S −9.08;
K only vs first_r1 25.05 / 25.00; D_stale 23.70, E_upd 24.37, E_relfull 24.37 (Δ 8e-6); A_changed: D_stale 21.97 →
E_upd 25.83 (RNA_T1 32.23, SO A_full-PU 33.66) → F 2.44, R_RNA 0.65; signed change on A_changed corr 0.987 / slope
0.948 / sign agreement 0.94 / explained 0.973; 64 / 64 solves converged in 6–7 iterations, ρ_T0 = ρ_T1_inc = 0.682
(K_r1: 10.47, 64 / 64 diverged); held-out probes 24.36 dB (K_r1 23.98); kernel parameters bit-identical across all
arms (digests in `train.json` / `eval.json`); training 5.35 h one-off, eval peak 9.3 GiB, solve 6 s and render 0.5 s
per view, incremental coupling update 41.9 s CPU for 0.63 % of the rays, full rebuild 92 s.

### OBSERVATION
- Boundary state: the on-surface multi-sample ports remove the Batch 6 irradiance bias (0.86 → 1.00) and most of the
  visibility disagreement; what remains is enclosed geometry (4.9 %) and port-resolution disagreement (3.8 %).
- Support: the declared radii are now realised, but one representative per far stratum is a poor quadrature on
  shadow-sparse real fields (Gate 2b FAIL after an unbiased, higher-K amendment).
- Excitation: four probe families raise the independent excitation ~25× in effective rank; the indirect field still
  has 22–24 % of its energy outside the training span (Gate 2c FAIL vs 0.10).
- Reference regime: clamping was the cause of the Batch 6 superposition deviation; the point-sampling film had to be
  set with GAUSSIAN 0.01 in this Cycles build.
- Passivity: with corrected ports, full support and 25× richer excitation, the *unconstrained* kernel K_r1 is still
  strongly non-passive (ρ 10.5; 42 % of Lego ports with uniform-light albedo > 1) and every composed solve diverges, as
  in Batch 6. The declared uniform-field hinge (K_r1P) gives ρ 0.68 and 64 / 64 convergence at no cost in probe
  fidelity (24.36 vs 23.98 dB).
- Static fidelity: the composed image is 24.5 dB (hist) / 25.0 dB (r1), essentially equal to the kernel alone against
  first-event references (25.05 dB); the composition adds no measurable error on top of the kernel's. Worst materials:
  Red_Glass 16.5, transparent 18.2, the red box 18.8, Yellow (SSS) 22.6; best: Brown 35.6, Light_brown 31.9.
- Edit: the state-only update with frozen parameters moves the image in the direction of the reference change
  (corr 0.99, slope 0.95, sign agreement 0.94 on A_changed), the incremental relation update is output-identical to a
  full rebuild, and 65 % of the stale error (relative to a T1-retrained RNA) is removed; the remainder is the kernel's
  own error (E_upd on A_changed 25.83 dB vs RNA_T1 32.23 dB). On stationary pixels the predicted change explains
  little of a reference change that is itself near the noise floor.
- The dense coupling table and 109 M rays are diagnostic infrastructure; nothing here is a runtime or compression
  result.

### INTERPRETATION
By cause, as the brief requires (implementation / data / numerical approximation / capacity / architecture):
- **Implementation.** The six audit defects were real and are removed in r1; their joint effect in Batch 6 was that
  the kernel was trained on biased port fields, with 1.4 / 5 cm effective support, without glass internal paths, against
  clamped and mismatched-filter targets, and the solver could not report slow divergence. After correction the
  composition is finite and converged on every held-out view with the constrained kernel. The Batch 6 verdict stands
  (that implementation diverged), but its attribution is sharpened: the decisive missing property was passivity, and
  passivity did not emerge from the corrected data or boundary state — it had to be imposed (K_r1 vs K_r1P).
- **Data.** Coverage remains below the declared 0.10 (0.24). It did not prevent convergence (the hinge did that) and
  did not add visible error beyond the kernel's own on these views (S_T0 | r1 25.00 ≈ K_only | first_r1 25.05);
  whether it limits accuracy at a higher-fidelity kernel is not testable with this one.
- **Numerical approximation.** Gate 2b (quadrature 0.27 / 0.20) and the 1.5 / 8 cm aggregation sit inside K_band / K_mid
  in both training and evaluation; their share of the 25 dB ceiling was not isolated.
- **Capacity / architecture.** The dominant limit of static fidelity is the intrinsic kernel's own first-event fidelity
  (24.4 dB probes, 25.05 dB on held-out views), i.e. the K side of the contract. The failing materials are exactly the
  ones the contract routes through K's internal paths (glass internal reflection / refraction, Yellow SSS) plus the
  box; reflective opaque materials reach 30–36 dB. Whether the ceiling is capacity (width 128, triplane 512 / 64),
  the operator family (2 GGX lobes, 19 hemispherical bins, isotropic-exit K_glob), the aggregation / quadrature, or
  optimisation (monitor peak at step 16 k then −0.2 dB) cannot be separated by this batch; the brief's rule applies —
  a low PSNR after corrections is a measured result requiring this attribution, not grounds for replacement.
- **The reuse contract (C side).** Descriptively supported on one scene and one diffuse-dominant edit: parameters
  bit-identical, state recomputation from geometry alone, incremental update exact in output, signed change in the
  right direction with slope ≈ 1 on the changed region, R_RNA 0.65. This is the property Batch 6 and SPT-1 could not
  show at a converged solve (SPT-1, worklog 39, its own metrics: F 1.67, R 0.82 at −11.3 dB static).
- Labelled comparison: RNA_T0 33.57 (hist); Surface-Owned A_full-PU 33.38 (T1-retrained); SPT-1 S_T0 22.23; FNTO r1
  (K_r1P) S_T0 24.49 — all held-out hist regime. FNTO r1 is +2.3 dB over SPT-1 and −9.1 dB under RNA.

### UNRESOLVED QUESTION
- What bounds the intrinsic kernel at ~25 dB first-event fidelity — capacity, operator family, the 1.5 / 8 cm
  aggregation and its quadrature (Gate 2b), or optimisation? Not separable from this batch.
- Why the single-material red box scores 16–18.8 dB even for K alone against first-event references, while Gate 3b
  shows its light-table radiometry consistent to 0.7 % on probes: the box kernel, or the held-out box pixel / port
  attachment, is unexplained.
- Is R_RNA = 0.65 limited by the update (E_upd − E_relfull = 0, so not by the incremental rule) or entirely by the
  operator's error on A_changed (25.83 vs 32.23 dB)? The data suggest the latter; a kernel of higher fidelity would be
  needed to test it.
- Does the uniform-field passivity hinge suffice on other geometries, or is ρ 0.68 specific to this graph? Only one
  scene was solved.
- Whether a representative quadrature with ~100 slots can reach 0.15 median error on shadow-sparse fields without
  importance sampling by the *incident* field (which would make C's structure field-dependent) — or whether the band /
  mid terms must be evaluated on all aggregates (cost ×5–10).
- Whether the 22–24 % out-of-span indirect energy is reducible by probe design alone within the isolated-domain
  constraint (no free-space interreflection in training).
- Output effect of the Gate 1 residuals (enclosed geometry 4.9 %, agreement 0.962): not measured.

### ARCHITECTURE VERDICT
**NOT SUPPORTED under protocol `fnto_r1`** (`results/fnto_lego_r1/verdict.json`; rules fixed in `r1_verdict.py` before
any result): the static-representability rule fails (S = −9.08 dB ≤ −2). Conditions met: G3, G4 (incl. 4c with the
declared conditional kernel K_r1P), every view converged, F ≥ 1.5, corr ≥ 0.5, slope in [0.5, 1.5]. Conditions not met:
G2d (24.4 / 25.05 dB < 28), S credible, R_RNA ≥ 0.7 (0.65). Carried as declared: G1 (inside-face 4.93 %, agreement
0.962), G2b (0.27 / 0.20), G2c (0.24). Stated per property of the K/C decomposition: the **C side** (current-geometry
free-space coupling, state recomputation, exact incremental update, correct signed change) is supported descriptively
on this scene; the **K side** is supported for passivity only under an explicit constraint and is **not** supported for
fidelity (first-event 25 dB, glass / SSS / box far below). No later batch is authorised by this worklog; no replacement
architecture is recommended (brief §6).

### REPRODUCIBILITY
- Code: `experiments/fnto_lego_r1/` at the closing commit of this worklog (see §F); Batch 6 code unchanged at 7ba0848.
- Protocol: `experiments/fnto_lego_r1/protocol/fnto_r1.json` (amendments 1–2 inside).
- Data: `results/fnto_lego_r1/` — `anchors.npz` (a1, sha16 32aa4032280225b7; a0 kept as `anchors_a0.npz`,
  0a93199fc3fe49f3), `rel_T0/T1/T1_inc.npz`, `stagea/{holdout,train}` (1,124 files, PC and server hashes identical),
  `stagea/prep_*`, `eval_px/`, `eval_vis.npz`, `refs/` (a2), discarded `stagea_box1px/`, `refs_box1px/`.
- Server: LabServer63 `~/rest_fnto_r1/` (repo copy, `launcher/*.status|*.log`, `r1_after_train.sh`).
- Verdict and exports: `r1_verdict.py` → `verdict.json`; `r1_exports.py` → `results/evaluation/42/` with
  `manifest.json` (sha16 per file).
