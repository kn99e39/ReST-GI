# Worklog — Compiled Closure Correctness and Energy Parameterisation Feasibility (WL80D, 2026-11-09)

Branch `Selective_Recompute`; LabServer63 (RTX 3080 Ti 12 GB; Mitsuba 3.5.1 cuda_ad_rgb, Dr.Jit 0.4.6, torch 2.3.1+cu121, env `~/NeuralShaderDemo/external/8dna26/.venv`). Isolated paths: `experiments/fnto_closure80d/`, `results/fnto_closure80d/`, server `~/rest_closure80d/`. WL80C code (`9f5b388`), results (`d219cf9`), worklog and verdict are unchanged. Production FNTO / C1 untouched. Protocol `experiments/fnto_closure80d/protocol/closure80d.json`, committed with the code in `679657b` before any WL80D evaluation.

## 1. Brief (user, 2026-11-09)

One bounded corrective experiment on the WL80C prototype, keeping three WL80C findings apart: (1) the mixture Sample defect, (2) the possibly over-restrictive sum(lambda) <= 1 energy constraint, (3) Evaluate+PDF not faster than the cheap analytic principled BSDF. Part A fixes the sampler and reruns the sampling contracts with unchanged closure parameters; Part B compares the WL80C construction (control) with ONE candidate `f = g / max(1, M)`, `M = sup_wi int g cos`, compiled once per material, labelled empirically passive unless M is certified; Part C repeats the GPU benchmark with the valid sampler and adds an expensive-reference workload only if one is readily available; Part D separates sampler correctness, representation, physical validity, GPU cost and expensive-response applicability. No compiler training, no rank / basis / population / threshold changes, no renderer.

## 2. Implementation correction (Part A) — recorded separately

Root cause, established before any WL80D evaluation (probes `/tmp/sel80d*.py`, reproduced in the regression tests): in Dr.Jit 0.4.6 `array == value` is a horizontal reduction that returns a Python bool, not an element-wise mask (`dr.select` itself behaves correctly with opaque / literal / width-1 operands). WL80C's `closure80c_dr.py` used `sel == (k + 1)` in `_select_lobe` and `sel == 0` in `sample` / `alt_sample`, so every selection mask was `False`: the selected alpha stayed at its default 1.0 and Lambert lanes received the GGX direction. This also invalidated WL80C's independent NDF sampler (it shares `_select_lobe`), so WL80C's C6 compared two defective samplers. Eval / pdf / eval_pdf and the per-lobe normalisation test did not use the masks.

Correction: `experiments/fnto_closure80d/closure80d_dr.py` is a copy of `closure80c_dr.py` with `dr.eq(sel, k + 1)` and `dr.eq(sel, 0)` (three lines; diff in the commit). WL80C files are not modified; the reused WL80C contract / bench code is pointed at the corrected module.

## 3. Architectural change (Part B) — recorded separately

Candidate energy parameterisation (`energy80d.py`): the same R = 8 basis and raw layout; lambda = softplus (no sum bound); per channel f_c = g_c / max(1, M_c). The mixture albedo is linear in lambda, `a_c(wi) = lambda_0c + sum_r lambda_rc A_r(wi)`, with per-lobe albedo A_r by a VNDF estimator whose integrand F G2 / G1 lies in [0, 1]. During fitting M is a differentiable estimate (12 theta x 4 phi grid, 512 stratified common random numbers); after fitting it is compiled once per material on a 64 theta (dense to 89.9 deg) x 16 phi grid with 32,768 stratified samples, M_c = max(grid albedo + 3 se), and the runtime closure is the WL80C closure with lambda' = lambda / max(1, M_c) — unchanged runtime code and state size. Passivity is empirical (finite grid + MC), checked independently with the corrected sampler at random wi up to 89.95 deg. Pre-evaluation dev probe: the torch VNDF albedo agrees with the corrected Dr.Jit sampler to ~1e-4 for alpha 0.004–1, F0 0.04–1, theta up to 89 deg; fit ~0.1 s / step, compile 1.25 s (8 materials), peak 1.5 GB.

Control: the WL80C oracle-P parameters themselves (fit.npz), fitted under the same oracle protocol (generic init, 3000 Adam steps, lr 0.03 → 3e-4); the candidate uses that protocol with an initial lambda of 1/9 like the control.

## 4. Execution

Chain `~/rest_closure80d/chain80d.sh` (no hard timeout; tests gate main), launched 06:48:13 UTC at commit `679657b`; tests 06:48:13–06:48:37, main 06:48:37–07:04:30 (fit 89 s, evaluation 1 s, independent passivity check 751 s, contract 32 s, benchmark 70 s). No foreign GPU process before / after the benchmark. GPU utilisation during the 8-material fit was about 15–25 % (launch-bound, small problem); the passivity check dominated wall time through host-side numpy reduction of 4.4 G samples — both are efficiency weaknesses of this script, not of the closure. Results: `results/fnto_closure80d/` (tests.json, main.json, stage3.json, candidate.npz, bench_rows.json, lane_*.npz, clean logs, chain.status).

## 5. Results

### 5.1 Part A — sampler correctness (control parameters unchanged)

Regression tests (`tests.json`), all PASS:
- T1 lane selection: 2^20 lanes, 64 random closures, per-lane and uniform: 0 index and 0 alpha mismatches vs a numpy inverse-CDF reference (selected-lobe histogram covers all 8 lobes).
- T2 eval / pdf consistency: Dr.Jit eval vs torch: median rel 1.0e-7, p99.9 1.1e-6, max 2.4e-6; Dr.Jit pdf vs numpy float64: median 8.9e-8, p99.9 8.4e-6; one lane differs by rel 198 — within the rule; consistent with the runtime's cos^2 theta >= 1e-7 clamp in Lambda at theta > 89.98 deg, which the float64 reference does not apply (convention difference at the horizon, not a selection error).
- T3 sampler vs quadrature: every case (Lambert; single GGX alpha 0.3, 0.3 x 0.1, 0.05; Lambert + GGX; GGX 0.05 + GGX 0.3; uniform probabilities with one weighted lobe; theta 10 / 45 / 75) agrees within 4 se + 2e-3 for albedo and valid fraction vs integral of pdf (the WL80C defect gave 0.585 vs 0.956).
- T4 WL80C compatibility: regenerated WL80C sets + stored WL80C parameters reproduce the WL80C oracle-P summary to rel 2.7e-14.

WL80C contract rerun with the corrected sampler (`stage3.json`), WL80C thresholds: control and the 50 WL80C adversarial closures pass C1–C7; C5b chi-square 48 / 48 cases p >= 1e-3 (min p 0.027); C6 VNDF vs independent NDF estimator: 0 % beyond 4 sigma, mean relative difference +0.4 % (t 0.74) for control; sampling utility vs the principled sampler: relative-variance ratio median 0.87, p90 1.13 (PASS). WL80C's C3 (adversarial) / C5b / C6 failures are therefore attributed to the implementation defect; WL80C's recorded failures stay as they were.

**SAMPLER_CORRECTNESS: PASS.**

### 5.2 Part B — energy parameterisation (8 WL80C materials, fresh WL80C evaluation directions)

| metric | control (WL80C, sum lambda <= 1) | candidate (g / max(1, M), compiled) | WL80C oracle-U (unconstrained, quoted) |
|---|---|---|---|
| E_L1 median / p90 / max | 0.053 / 0.173 / 0.205 | **0.044 / 0.089 / 0.158** | 0.025 / 0.065 / 0.148 |
| E_alb median / p95 / max | 0.025 / 0.169 / 0.200 | **0.022 / 0.069 / 0.102** | 0.009 / 0.035 / 0.052 |
| E_L1 median theta 5 / 35 / 60 (band) / 80 / 86 | 0.046 / 0.051 / 0.049 / 0.069 / 0.075 | 0.040 / 0.039 / 0.038 / 0.071 / **0.107** | 0.018 / 0.022 / 0.024 / 0.047 / 0.072 |
| highlight peak error median / p90 (27 pronounced) | 0.008° / 1.72° | 0.009° / 1.72° | — |
| highlight \|log width\| median / p90 | 0.120 / 2.81 | **0.244** / 2.81 | — |
| R1 / R2 / R3 | PASS / FAIL / PASS | PASS / **PASS** / **FAIL** | — |
| fit loss median | 2.10e-4 | 1.41e-4 | — |

Per-material mean E_L1 control → candidate: 0.049→0.051, 0.051→0.075, **0.187→0.049** (WL80C worst, sheen-heavy dielectric), 0.056→0.039, 0.082→0.047, 0.060→0.062, 0.050→0.031, 0.030→0.061. Compiled M_c ranged 0.17–5.43 (scaling active, M > 1, in 9 of 24 channel-materials); the compile M exceeded the in-fit estimate by about 1–5 %, so the compiled closures are slightly darker than the fitted ones. Sum lambda' after scaling 0.25–1.65.

- **materially_addresses: YES** (R2 passes, E_L1 median 0.044 <= 0.053): the albedo tail (p95 0.169 → 0.069) and the WL80C worst material are repaired.
- **REPRESENTATION_FEASIBILITY (candidate): FAIL** under the frozen rule — R3 (median |log width| 0.244 > log 1.15 = 0.140); peak location unchanged. Grazing error increases (86°: 0.075 → 0.107). Both are measured on 8 materials of one family.

### 5.3 Physical contract of the candidate

- WL80C checks: C1, C2, C4, C5a, C6, C7 PASS for candidate and 50 adversarial candidate closures; C5b PASS (shared). C3 reports FAIL for both sets, but only through its analytic sub-condition `max sum lambda <= 1 + 1e-6`, which belongs to the WL80C construction and is violated by the candidate by design (1.654 / 4.51); the sampler-albedo part of C3 had 0 violations (max 0.946 / 0.978). Including that sub-condition for the candidate was a protocol oversight on my part; the frozen result is recorded as FAIL and the reason stated here.
- Independent passivity check (corrected sampler, 1024 random wi per closure up to 89.95°, 65,536 samples): candidate 0 / 8,192 (max 0.997); control 0 / 8,192 (max 0.935); **adversarial candidate 1 / 51,200 violations (estimate 1.090)**.
- **Post hoc diagnostic** (`experiments/fnto_closure80d/diag80d_posthoc.py`, `results/fnto_closure80d/diag_passivity_finegrid.json`; run after the frozen evaluation, labelled post hoc): exact-integrand VNDF albedo of the compiled closures on a 140 theta (to 89.999°) x 91 phi grid with an independent sample set. The maximum of every mixture albedo lies at the grazing limit theta → 90°, outside the frozen compile grid (<= 89.9°). Even the 8 real candidate closures exceed 1 there (8 channel-materials, max 1.011, se ~1e-4); adversarial closures reach 1.388 (149 channel-closures above 1 by > 3 se). The grid supremum is therefore **not an upper bound**: Schlick-Fresnel GGX lobe albedo rises steeply towards grazing, and the sup sits at the horizon limit.
- **PHYSICAL_CONTRACT (candidate): FAIL** — not mathematically passive (no certificate) and not even empirically passive at the grazing limit; the defect is in how M was computed, not in the scaling idea, but any repair (e.g. evaluating the theta → 90° limit) is a protocol change after results and was not made.

### 5.4 Part C — GPU utility (corrected sampler; REF Mitsuba principled C++; 4 materials; kernel-history GPU time; no contention)

| workload (largest size) | eval_pdf | eval | pdf | sample |
|---|---|---|---|---|
| io 16.8 M, REF/CMP speedup (CTRL / CAND) | 0.983 / 0.984 | 0.976 / 0.975 | 0.980 / 0.982 | 0.999 / 0.999 |
| fused16 4.2 M x 16, speedup (CTRL / CAND) | 0.939 / 0.940 | 1.144 / 1.144 | 0.850 / 0.850 | **2.214 / 2.214** |
| fused16 ns / query REF vs CAND | 0.058 vs 0.062 | 0.033 vs 0.029 | 0.015 vs 0.018 | 0.097 vs 0.044 |

The corrected sampler is one kernel (WL80C: 25) and, in the compute-bound workload, takes less than half the GPU time of principled's sample (2.2x); evaluate alone is 1.14x faster, evaluate+pdf 0.94x and pdf 0.85x slower; the io workloads are memory-bandwidth-bound ties (also for the single-lobe roughconductor sanity arm). CTRL and CAND cost the same (identical runtime code; the candidate's M is folded into lambda). Frozen rules: advantage (>= 1.10 for eval_pdf and sample at both workloads) **NOT met**; parity (CMP/REF <= 1.25 for eval_pdf and sample, fused) **met** (1.06x and 0.45x).

Costs: compiled state 53 floats = 212 B per material (45 independent; runtime table 88 floats = 352 B) vs 13 principled parameter floats; candidate adds no runtime state. Candidate compile (M on the frozen grid) 0.16 s per material, 1.25 s for 8 (torch, not optimised); compiler MLP forward (WL80C architecture, untrained, cost only) 0.69 ms for 1 or 1,024 materials (launch-bound); runtime closure binding 1.1 ms.

Expensive reference: none readily available (Mitsuba `measured` plugin present but no RGL / MERL data on LabServer63; nothing downloaded or generated) → utility against expensive material responses **UNRESOLVED**.

## 6. Verdicts

| verdict | result |
|---|---|
| SAMPLER_CORRECTNESS | **PASS** (implementation defect fixed; all sampling contracts pass on the original WL80C parameters) |
| REPRESENTATION_FEASIBILITY | control FAIL (R2 tail); candidate FAIL (R3 highlight width) although it materially fixes the albedo tail and the worst material; both on 8 materials, direct fits |
| PHYSICAL_CONTRACT | control PASS; candidate **FAIL** (grid-computed M misses the grazing-limit supremum; real closures exceed albedo 1 by up to 1.1 %, adversarial by up to 39 %) |
| GPU_UTILITY | vs cheap analytic principled: parity yes, advantage no (eval+pdf 0.94x, pdf 0.85x; sample 2.2x faster, eval 1.14x faster in the compute-bound case; memory-bound tie); vs expensive responses UNRESOLVED |
| LEARNED_RULE | **UNTESTED** (no compiler training; direct fits only) |

Decision rule (`justifies_learned_rule_training` = sampler AND candidate representation AND candidate physical AND parity): **NO** — the candidate fails representation (R3) and physical contract, while sampler and parity pass. Completion question: after the sampler fix and the one energy variant, the fixed R = 8 closure does **not** demonstrate enough combined quality and validity under the frozen rules to justify learned-rule training; the control remains valid but fails its albedo tail, the candidate improves the tail but loses highlight-width fidelity and passivity at grazing. Runtime is at parity with a cheap analytic BSDF (no advantage shown); any utility for expensive material responses is unmeasured.

## 7. Unresolved

- Whether an M computed with the grazing limit (or a certified bound) keeps the candidate's quality gain while restoring passivity.
- Why the candidate widens some highlights (|log width| 0.12 → 0.24) — the fit loss is lower, so the loss weighting versus the half-maximum width metric may disagree; not investigated.
- Learned-rule transfer; expensive-reference cost utility; behaviour beyond 8 materials of one family.

## 8. Reproducibility

Code / protocol `679657b` (`experiments/fnto_closure80d/`, uses `experiments/fnto_compiled80c/` from the same commit); post hoc diagnostic script `experiments/fnto_closure80d/diag80d_posthoc.py`; WL80C inputs `results/fnto_compiled80c/smoke/` (`d219cf9`); seeds as in the protocol (sets 80900, fit-M 80410, compile-M 80420, adversarial 80430, tests 80440, passivity 80450, bench 80370, contract 80350; post hoc 80460); LabServer63 RTX 3080 Ti, Mitsuba 3.5.1 cuda_ad_rgb, Dr.Jit 0.4.6, torch 2.3.1+cu121, `LD_LIBRARY_PATH=$HOME/nvlibs/optix595`. No images produced, so no evaluation folder.

## 9. Visual export (post hoc, user request 2026-11-09; no new metric or verdict)

The user asked for quick renders or the images behind the metrics. Scripts `experiments/fnto_closure80d/viz/` (`44753f9`), run on LabServer63 07:23–07:25 UTC; files, hashes and provenance in `results/evaluation/80D/manifest.json`.

- `highlight_grid_material{0..7}.png`: the slope grids used by R3 (reference / control / candidate; |error| on one shared sqrt scale, max 25.7; log2 ratio fixed ±2). Observation: the candidate (and faintly the control) contains a thin ridge along slope x = 0, i.e. a lobe with very small alpha_u; it is visible as a streak across the sphere in the material-2 render and is a plausible contributor to the candidate's larger half-maximum width error (not tested).
- `albedo_vs_theta.png`, `albedo_grazing_zoom.png`: closure albedo keeps rising beyond the compile grid end (89.9°); the candidate overshoots the reference at grazing for every shown material (consistent with its worse 86° E_L1) and reaches > 1 for material 5 at this azimuth.
- `sampler_fix_histograms.png`: WL80C sampler mass 0.585 with a featureless distribution vs corrected sampler 0.987 matching the pdf() expectation.
- `render_highlight_rot0.png`, `render_highlight_rot40.png` (rigid 40° rotation of the sphere), `render_indirect_rot0.png`: Mitsuba llvm path tracer, 256 spp, identical scene / seed per arm, white-furnace check mean 0.99999. Relative L1 vs the same-spp reference (includes both images' MC noise): control 0.014–0.133 (material 2 worst), candidate 0.011–0.032. These are quick visual checks, not a rendering-utility evaluation.
