# Worklog — Compiled Neural Material Rules: Bounded GPU Feasibility Smoke (WL80C, 2026-11-09)

Branch `Selective_Recompute`; machine LabServer63 (RTX 3080 Ti 12 GB, Mitsuba 3.5.1 / Dr.Jit 0.4.6 / torch 2.3.1+cu121, env `~/NeuralShaderDemo/external/8dna26/.venv`). Isolated paths: `experiments/fnto_compiled80c/`, `results/fnto_compiled80c/`, server `~/rest_compiled80c/`. Production FNTO, C1 and historical code untouched. WL80B results were not read or used.

## 1. Original requested scope (user brief, preserved)

Implement and evaluate one isolated prototype of a *Compiled Neural Material Rules* closure
`f(m, wi, wo) = sum_r lambda_r(m) B_r(wi, wo; phi_r(m))`, R <= 8, produced once per material by a shared neural compiler, with analytic Evaluate / Sample / PDF at runtime. Five separate claims: A representable, B learnable rule, C physically consistent, D useful sampling, E competitive GPU cost; Stages 0 (reference / protocol), 1 (closure), 2 (representation + learned-rule gate, two seeds, held-out materials / directions), 3 (Evaluate / Sample / PDF contract), 4 (matched GPU cost), 5 (small rendering validation, conditional); budget ~60 active GPU minutes; final status one of SUPPORTED / NOT_SUPPORTED / INCONCLUSIVE / STOPPED.

## 2. Work done under the original scope (before the correction)

- **Reference** (Stage 0): Mitsuba 3.5.1 `principled` plugin (Burley 2015 Disney BSDF as implemented by Mitsuba), reflection only (`spec_trans` 0, `specular` 0.5, `spec_tint` 0, `sheen_tint` 0.5); descriptor = base RGB, metallic, roughness [0.15, 0.8], anisotropic [0, 0.9], clearcoat, clearcoat_gloss [0, 0.8], sheen. Per-lane materials through nearest-filtered raw 1 x N bitmap textures.
- **Stage 0a reference-only survey** (`survey80c.py`, 2000 box materials, seed 80300, ~51 s GPU; `results/fnto_compiled80c/stage0/survey.json`): texture-driven = uniform-parameter eval exactly (max rel 0.0, 16 materials); reciprocity of f = eval / cos_o max rel 5.2e-6 (983k pairs); cuda vs llvm eval agree to ~5e-6 (one probe). **The reference is not passive:** 14 % of box materials exceed directional albedo 1 at some wi (4-sigma, 2.9 % of (material, wi) lanes), max 1.32, concentrated at grazing incidence and low metallic (Disney's additive diffuse + specular layering). Consequence fixed before any candidate result: the material population is the box intersected with reference-passive materials (point-estimate albedo <= 1 on a 24-wi grid).
- **Closure (Stage 1, frozen, not tuned):** R = 8 = 1 Lambert + 7 frame-aligned anisotropic GGX lobes with Schlick Fresnel (F0_r) and height-correlated Smith G2; RGB weights from a 9-slot softmax (8 lobes + absorption), so lambda >= 0 and sum_r lambda_rc <= 1; each lobe has albedo <= 1, so the closure is passive by construction; f is symmetric (reciprocal). Sampling: wi-independent lobe probabilities, cosine warp / GGX visible-normal sampling (spherical caps), mixture pdf, weight = eval / pdf. Compiled state: 45 independent floats per material (+8 derived sampling probabilities; 88-float runtime table). Compiler: MLP 9-128-128-128-48 (40,496 parameters). Torch implementation for fitting (`closure80c.py`), Dr.Jit runtime for contract / benchmark (`closure80c_dr.py`), including an independent NDF-proportional sampler used only by the contract tests.
- **Protocol** `experiments/fnto_compiled80c/protocol/compiled80c.json` (splits, angular populations incl. held-out incident band 55–65 deg, grazing >= 80 deg, held-out combination block, training budget, oracle definition, gates, contract tests, benchmark, Stage-5 plan, final-status mapping) written and committed (`9f5b388`) before any candidate code ran.

## 3. SCOPE CORRECTION (user direction, 2026-11-09, during the batch)

The user judged the projected completion time (19:30–21:00 KST) a scope expansion and converted the batch into a minimal feasibility smoke: at most 30 additional wall-clock minutes and 10 additional active GPU minutes, no restart, no discarded work, no tuning of rank / basis / thresholds / populations / loss. Recorded as `protocol/amendment1_scope_correction.json` (before any candidate result).

State at the correction: code and protocol complete and uncommitted; only the Stage-0a reference survey existed; no GPU job was running (nothing to interrupt); no candidate result existed.

**Cancelled (never started):** full material splits / dataset generation (3800 materials), compiler training (both seeds), full Stage-2 evaluation on test_iid / test_block / train_eval, full-population contract runs, 8-material benchmark with compile cost / break-even, Stage 5 renderer integration.

**Executed instead** (`smoke80c.py`, commit `9f5b388`):
- Gate A — representability upper bound: direct per-material fit of the frozen closure (oracle-P definition: generic init, 3000 steps, frozen loss / learning rate) for the first 8 reference-passive materials of the existing Stage-0a survey, in stored order (no selection by error). Fit on 64 wi x 16 wo training-style pairs; evaluated on fresh directions (8 theta incl. 60 = held-out band, 80 / 86 = grazing, 2 phi, 1024 wo each) and the frozen highlight geometry; frozen representation thresholds R1–R3 applied to this reduced population (diagnostic, not the original test population). Interpretation note: the reference plugin is evaluated directly on these few directions (milliseconds); no dataset campaign was run.
- Physical contract: existing checks C1–C7, C5b and sampling utility on the 8 fitted closures + 50 adversarial random closures.
- Gate B: REF Mitsuba principled vs CMP compiled closure on the same Dr.Jit CUDA tracing path, Dr.Jit kernel-history GPU time, 4 materials.
- Gate C (learned compiler): **not run** — it would need dataset generation and training.

## 4. Smoke execution

Commit `9f5b388` deployed as an LF archive to `~/rest_compiled80c/code` (CRLF count 0); GPU free of foreign processes before and after. Launched 06:21:56 UTC inside `timeout 600`. **User instruction during the run** ("하드 타임아웃은 없애고, smoke라고 하더라도 gpu utilization은 좀 열심히 돌리면 좋겠다"): the `timeout` wrapper was killed at 06:24:34 and the Python process kept running (so `smoke.status` shows `END rc=137` for the wrapper, not the job; the job wrote all outputs and exited normally). The Gate-A fit (8 materials x 1024 pairs, 2 x 3000 Adam steps) was launch-bound at about 8–12 % GPU utilisation; that is a smoke-design weakness (too little work per launch) and was not changed mid-run. Active GPU time: smoke 194 s (fit 109 s, metrics + highlight, contract 24 s, bench) + attribution checks about 40 s ≈ 4 min. Results: `results/fnto_compiled80c/smoke/` (smoke.json, stage3.json, bench_rows.json, materials.npz, fit.npz, attribution logs).

Materials (survey indices 0, 1, 2, 3, 4, 5, 7, 8; index 6 rejected by the passivity filter). Descriptors in `smoke.json`.

## 5. Results

### 5.1 REPRESENTATION — direct fit upper bound (not a learned rule)

oracle-P (the frozen passive closure, per-material free fit), fresh directions, 8 materials x 8 theta x 2 phi:

| metric | oracle-P | frozen rule | oracle-U (diagnostic: lambda >= 0 without the sum bound, not passive by construction) |
|---|---|---|---|
| E_L1 median / p90 / max | 0.053 / 0.173 / 0.205 | <= 0.10 / <= 0.25 → R1 PASS | 0.025 / 0.065 / 0.148 |
| E_alb median / p95 / max | 0.025 / 0.169 / 0.200 | <= 0.03 / <= 0.10 → **R2 FAIL (tail)** | 0.009 / 0.035 / 0.052 |
| highlight peak error median / p90 (27 pronounced) | 0.008° / 1.72° | <= 1.0° | — |
| highlight \|log width\| median / p90 | 0.120 / 2.81 | <= log 1.15 = 0.140 → R3 PASS | — |
| E_L1 median by theta 5…86° | 0.046 0.047 0.051 0.053 0.049 (60, held-out band) 0.048 0.069 0.075 | | 0.018 … 0.072 |

Clear-fail rule not triggered (median E_L1 0.053 <= 0.20, median E_alb 0.025 <= 0.06). Gate A on the frozen R1–R3: **not passed (R2 tail)**, not a clear failure.

Worst observed failure: material 2 (base 0.33/0.55/0.39, metallic 0.20, roughness 0.34, anisotropic 0.30, clearcoat 0.44, gloss 0.25, sheen 0.87) with per-material mean E_L1 0.187 under oracle-P vs 0.034 under oracle-U; grazing (80/86°) medians are highest for both arms; a minority of highlights have a badly wrong half-maximum width (p90 |log width| 2.8).

Attribution (descriptive, one population of 8): removing only the sum bound roughly halves every error and repairs material 2, so the dominant limit here is the closure's **sufficient passivity bound** (sum_r lambda_rc <= 1 with lobes whose albedo can approach 1 at grazing incidence), not the basis family; the reference materials themselves are passive by selection. The remaining oracle-U error grows at grazing (0.072 at 86°), consistent with basis expressivity there. Training optimisation is not implicated for oracle-P beyond the usual local-minimum caveat (one generic init only; no compiler init exists).

### 5.2 LEARNED RULE

Not measured. No compiler was trained (Gate C cancelled by the scope correction). The direct fit above says nothing about transfer to unseen materials.

### 5.3 PHYSICAL CORRECTNESS (runtime Dr.Jit closure, 8 fitted + 50 adversarial closures)

| check | fitted | adversarial |
|---|---|---|
| C1 finite / nonnegative / invalid inputs → 0 | PASS | PASS |
| C2 reciprocity (max rel) | PASS 1.9e-5 | PASS 1.4e-5 |
| C4 per-lobe pdf normalisation vs independent NDF integral | PASS (max z 3.2) | PASS |
| C5a pdf / weight identity | PASS | PASS |
| C7 support | PASS | PASS |
| C3 passivity via own sampler | PASS (max 0.84) | **FAIL** (64/400, estimate up to 2.34) |
| C5b chi-square | **FAIL** 24/24 (p ≈ 0) | |
| C6 VNDF vs independent NDF estimator | **FAIL** (46 % z > 4, mean +11 %) | **FAIL** (+18 %) |
| sampling utility vs reference sampler (median / p90 rel.-variance ratio) | 2.08 / 6.93 → FAIL | |

Attribution (`experiments/fnto_compiled80c/diag/`, logs in the results folder): every single-lobe closure (Lambert; isotropic and anisotropic GGX, alpha 0.05–0.3, theta 10–75°) is exact — sampler albedo = quadrature albedo, int pdf = 1 − failure fraction. Any lobe **mixture** keeps a valid-sample fraction of 0.585 regardless of composition (expected 0.956), and a lane trace shows the selected lobe index is correct per lane while the selected alpha collapses to a width-1 default (1.0) and Lambert lanes receive the GGX direction. **Cause: an implementation defect in the mixture sampler's `dr.select` of the width-1 opaque lobe parameters (`closure80c_dr.DrClosure._select_lobe` / `sample`), not a property of the closure or of Evaluate / PDF.** Per the scope correction (stop on physical-validity failure, no rescue) it was not fixed or rerun. Physical correctness of Sample is therefore **FAIL as implemented**; Evaluate / PDF pass every applicable check.

### 5.4 GPU PERFORMANCE (REF = Mitsuba principled C++, CMP = compiled closure; same Dr.Jit CUDA tracing path, kernel-history GPU time, 4 materials, 10 reps; no contention)

| workload | eval_pdf | eval | pdf | sample |
|---|---|---|---|---|
| io 16.8 M lanes, REF/CMP speedup (median) | 0.99 | 0.975 | 0.98 | 0.51 (invalid) |
| io ns/query REF / CMP | 0.054 / 0.055 | 0.048 / 0.049 | 0.034 / 0.035 | 0.068 / 0.132 |
| fused16 4.2 M lanes x 16, speedup | **0.94** | **1.145** | **0.85** | 0.39 (invalid) |
| fused16 ns/query REF / CMP | 0.059 / 0.063 | 0.033 / 0.029 | 0.015 / 0.018 | 0.097 / 0.250 |

The io workload is memory-bound for every arm (16.8 M lanes at ~0.9 ms ≈ 740 GB/s; even the single-lobe roughconductor sanity arm takes 0.66 ms), so it cannot separate the arms. In the compute-bound fused workload the compiled closure is 14 % faster for Evaluate alone but 6 % slower for Evaluate+PDF and 15 % slower for PDF than the analytic principled BSDF; no op reaches the frozen 1.10 rule for eval_pdf. Sample timings are not valid evidence: the defective sampler launches 25 kernels per call instead of 1. No compiler exists, so compile cost and break-even were not measured (the predeclared break-even requires t_cmp < t_ref for eval_pdf, which was not observed).

## 6. Compact table

| item | result |
|---|---|
| Reference material | Mitsuba 3.5.1 principled (Disney 2015), reflection only, 9-dim descriptor; reference non-passive for 14 % of the box → passive subset used |
| Lobe family / rank | 1 Lambert + 7 anisotropic GGX x Schlick, R = 8, RGB softmax weights with absorption slot (sum <= 1) |
| Model / compiled-state size | compiler 40,496 params (162 KB fp32, untrained); 45 floats per material (+8 derived; 88-float runtime table) vs 13 reference floats |
| Representability (direct fit, 8 materials) | median E_L1 0.053 / E_alb 0.025, highlight peak 0.008°; tail fails (E_alb p95 0.169 > 0.10) |
| Worst directional failure | sheen-heavy low-metal dielectric (material 2, E_L1 0.187); grazing 80–86° strata; some half-max widths off by > e^2 |
| Physical contract | Evaluate / PDF: all checks PASS; Sample: FAIL (mixture-selection implementation defect, localised, not fixed) |
| GPU Evaluate | io memory-bound tie (0.975–0.99x); fused eval 1.145x faster, eval_pdf 0.94x, pdf 0.85x |
| GPU Sample / PDF | Sample UNRESOLVED (defective implementation); PDF 0.85x (slower) |
| Learned compiler | not trained / not measured |
| Missing evidence | learned-rule transfer, valid sampling and its cost, compile cost / break-even, rendering, full material populations |

## 7. Interpretation and verdicts

- REPRESENTATION: **plausible in the median, not passed at the frozen tail rule** (8 materials, direct fit). The passive mixture bound is the measured dominant limiter here; it is a property of the chosen passive parameterisation, not of the reference.
- LEARNED RULE: **not measured.**
- PHYSICAL CORRECTNESS: **Evaluate / PDF valid; Sample physically invalid as implemented** (software defect; a corrected sampler is unmeasured).
- GPU PERFORMANCE: **no measured advantage** against the analytic principled BSDF on the same GPU path (Evaluate+PDF 0.94x, PDF 0.85x in the compute-bound workload; Evaluate alone 1.145x; memory-bound workloads tie). Sample cost unresolved. Because the comparator is a cheap analytic BSDF, this says nothing about targets whose direct evaluation is expensive (measured / layered stochastic BSDFs), which were not tested.
- Outcome class (scope correction list): **physically invalid** (Sample, as implemented) — with representation plausible but tail-failing, and GPU performance not promising for Evaluate+PDF against this comparator. Final status under the original mapping: STOPPED_BY_CONTRACT_OR_RESOURCE (physical-contract stop), with the split verdict above. No claim about learned K, FNTO or dynamic GI follows.

## 8. Unresolved

- Whether a corrected mixture sampler passes C3 / C5b / C6 and what it costs (expected one kernel, like REF; unmeasured).
- Whether a shared compiler transfers to held-out materials (not run).
- Whether a less conservative but still guaranteed passivity construction would close the tail gap (would change the frozen closure; not authorised).
- Whether any target whose reference evaluation is expensive gives the compiled closure a cost advantage.

## 9. Reproducibility

Code / protocol `9f5b388` (`experiments/fnto_compiled80c/`, protocol `compiled80c.json` + `amendment1_scope_correction.json`); attribution scripts `experiments/fnto_compiled80c/diag/` (run on the deployed `9f5b388` code); LabServer63 RTX 3080 Ti, driver with `~/nvlibs/optix595`, Mitsuba 3.5.1 / Dr.Jit 0.4.6 cuda_ad_rgb, torch 2.3.1+cu121; seeds survey 80300, smoke 80900, adversarial 80360, contract 80350, bench 80370; server results `~/rest_compiled80c/results/{stage0,smoke}`, local copies `results/fnto_compiled80c/`. Not created: an evaluation folder (no images were produced). Skipped original stages: full splits / datasets, compiler training (2 seeds), full Stage-2 evaluation, full-population contract, full benchmark with compile cost, Stage 5 rendering.
