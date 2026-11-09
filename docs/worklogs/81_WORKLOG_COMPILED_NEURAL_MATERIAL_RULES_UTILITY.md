# Worklog — Compiled Neural Material Rules: Learned Representation and Rendering Utility Validation (2026-11-09)

## Origin

- **Brief.** WL81, "Compiled Neural Material Rules: Learned Representation and Rendering Utility Validation", 2026-11-09.
- **User decisions, 2026-11-09:**
  - Option 2.
  - The server is idle.
  - Reuse the WL80D sampling fix, but confirm it with focused regression.
  - The WL80D energy candidate is **not adopted**, because a grazing-limit passivity violation was confirmed.
  - The new candidate is a post-WL80D revision, fixed by a protocol amendment before its first evaluation and judged on independent data.
  - No learned-compiler training unless the physical contract and representation gate pass.
  - Remove duplicate implementation.
  - No hard time or hardware budget.
- **Where.** Branch `Selective_Recompute` on LabServer63 (main server), RTX 3080 Ti 12 GB. Mitsuba 3.5.1, Dr.Jit 0.4.6, torch 2.3.1.
- **Isolated paths.**
  - `experiments/fnto_compiled81/`
  - `results/fnto_compiled81/`
  - `docs/plans/wl81/`
  - server `~/rest_compiled81` (dev runs: `~/rest_compiled81_dev`)
- **Preserved evidence.** WL80C (`9f5b388`, `d219cf9`) and WL80D (`679657b`, `c3577be`) code and evidence are untouched. Their modules are imported, not copied.

## Pre-freeze work (reference selection and debugging; not evidence for any candidate)

1. **Reference implementation.** `layered81.py` is a coated-conductor layered BSDF:
   - a Dr.Jit port of pbrt-v4 `LayeredBxDF::f` and `Sample_f`;
   - per-lane ports of the Mitsuba 3.5.1 rough dielectric and rough conductor interfaces.
2. **Interface audit (dev, `iface`).**
   - 65,536 random pairs × 18 dielectric settings (IOR / α × radiance / importance × component masks) + 3 conductor settings.
   - eval and sample are bit-identical to the plugins; pdf max rel 8.1e-5.
   - One test-harness correction: Mitsuba leaves `bs.pdf` unmasked on rejected conductor samples, so sample validity = nonzero weight.
3. **Walk audit (dev, `walk`), first run.**
   - Sampled albedo vs eval-integrated albedo disagreed: z up to 19, about 5 %.
   - Reciprocity failed: z up to 79.
   - **Cause found.** pbrt-v4 `LayeredBxDF::f` MIS-weights its two exit strategies with `exitInterface.PDF(-w, wi)`. That is the density of the reverse query, not the density with which the NEE strategy generates the inside direction, so the weights do not sum to one.
   - **Correction.** Use `diel_pdf(wi, -w, transmission)`.
4. **Walk audit, after the correction (`walk_b`).** All |z| ≤ 2.6.
   - max_depth 64 truncates up to 1.2 % for a near-white thick slab, so max_depth is frozen at 1024.
5. **Independent slab (dev, `slab`).** Mitsuba volpath (llvm) on a geometric slab: rough dielectric top, interior homogeneous medium, roughconductor floor, directional light, radiance meter.
   - 6 configurations × 3 channels: max |z| 2.15, max |rel| 1.0 %.
   - llvm was used because LabServer63's cuda film is defective (WL53).
6. **WL80C / WL80D horizon defect (found here).**
   - torch `closure80c.eval_closure`: a single GGX lobe of weight 1 has directional albedo 3.15 at μ = 1e-4, 31.4 at 1e-5 and 315 at 1e-6 (α 0.05 and 0.3, F0 1, 2^22 samples).
   - Cause: Λ clamps cos² θ ≥ 1e-7, but the division by cos θ_i is unclamped.
   - Fix: algebraically identical clamp-free closed forms (`closure81.py`, `closure81_dr.py`), the same for every arm. Stage B6 tests it.
7. **Dev smoke of the chain** (`--dev` populations of 16–64 materials; non-evidence; quality numbers not interpreted).
   - Reference noise floor of the dev gate set at 4096 walks: E_L1 median 0.006.
   - Highlight peak half-vs-half: 0.52° at 256 walks and 0.47° at 1024 walks. That is mainly peak quantisation on the 201-grid, not walk noise. `highlight_grid` walks were set to 1024 before the freeze.
   - Renderer validation: Lambert / point light exact to 6e-7, closure plane 8e-7, layered plane |z| ≤ 3.35.

## Freeze

- **`e4b471e` — protocol v1 and all WL81 code.**
  - Pre-freeze changes prompted by the dev smoke (none of them followed any candidate result):
    - `highlight_grid` walks 256 → 1024;
    - C5b supplement on fixed synthetic closures, because the WL80C C5b eligibility set was empty in dev, so the test could not be evaluated;
    - G81 sampling moved to GPU PCG32 / `block_sum`;
    - kernel-history flag enabled in bench / render;
    - Stage C → D rule: train both arms when one qualifies; only a qualified arm can be primary;
    - Stage E precondition;
    - compile-latency repetitions for 1024 materials.
- **`2ac5aed` — amendment 1 (energy candidate `C81_limit_complete_sup`).** Committed before any candidate evaluation on WL81 data. Provenance: post-WL80D hypothesis revision.
- **Server.** `~/rest_compiled81/code` = `git -c core.autocrlf=false archive` of `2ac5aed`.
- **Chain.** One detached chain, `run81.sh`, started 2026-11-09 09:16 UTC; status in `~/rest_compiled81/chain.status`.
  - Order: A0 audits → B regression → A1 populations and targets → C energy A/B → D learned compiler → E GPU → compression → F rendering.
  - Every gate stops downstream stages by the frozen rules.

## Execution amendments (before any candidate or control result)

- **Amendment 2 (`aca745a`).** Prompted by the user ("제발 병렬로 좀 돌려줘") after run 1 was observed at about 5 % GPU utilisation.
  - Run 1's reference generation was CPU-bound in a single process.
  - Walk half-means are now reduced on the GPU (`dr.block_sum`, float64).
  - Launches are 2^24 lanes.
  - Independent jobs inside a stage run in parallel.
- **Amendment 3 (`4db2941`).** Found while verifying the sharding.
  - Dr.Jit 0.4.6 `PCG32(size, initstate, initseq)` adds `arange(size)` to `initseq` itself, so `initseq = arange + offset` produced the sequence 2j + offset.
  - Consequence: launch chunks of one call overlapped, i.e. lanes of different pairs drew identical random streams. This affected only the aborted run-1 and run-2 A1 data, which were never used.
  - Fix: a scalar global offset. Sharded generation is then bit-identical to unsharded generation (verified in dev).
  - Sharding: eval sets ×3, highlight sets ×6.
  - GPU utilisation went from about 5 % to 100 % (13 concurrent processes).
- **Stages A0 and B** were reused from run 1. Their code is unchanged.
- **Stage A1** ran in 18 min.
- **Aborted partial data** is in `results/data_run1_aborted` and `results/data_run2_aborted` on the server and is not used anywhere.

## Run (frozen chain `4db2941`, LabServer63, 2026-11-09)

| Stage | UTC | Result |
|---|---|---|
| A0 reference qualification | 09:16–09:17 (reused in run 3) | QUALIFIED. Interfaces bit-identical to the Mitsuba plugins; walk consistency max \|z\| 1.98 (albedo) / 2.53 (reciprocity); independent volpath slab max \|z\| 2.15, \|rel\| ≤ 1.0 %; truncation 1024 vs 4096 \|rel\| ≤ 1.5e-3, \|z\| ≤ 1.7 |
| B sampler regression + horizon fix | 09:17–09:18 (reused) | CONTINUE. B1–B5 PASS; B6 PASS |
| A1 populations and targets | 09:56–10:08 | QUALIFIED. 4096 walks sufficed for every eval set |
| C energy A/B (direct fits on the gate set) | 10:08–10:27 | No arm qualified. Both physical contracts PASS; every arm fails the representation gate (clear fail) |
| D / E / compression / F | — | NOT REACHED (frozen rule: no compiler training unless the physical contract and representation gate pass) |
| Post hoc attribution (`f5acf7d`) | 10:28–11:43 | See Observation |

## Final report

### Implementation Fact

**Reference.** `coated_conductor_layered_v1` (`layered81.py`):
- a Dr.Jit port of the pbrt-v4 `LayeredBxDF` position-free random walk;
- per-lane ports of the Mitsuba 3.5.1 rough dielectric and rough conductor interfaces;
- one estimator correction: the MIS partner density of the exit strategies (see pre-freeze item 3).

**Population.** Descriptor, 10 dimensions: coat α, IOR, slab optical thickness, slab albedo (RGB), conductor reflectance (RGB), conductor α.
- Splits: 3000 train, 200 val, 300 gate, 300 test_iid, 300 held-out-block materials.

**Software fixes.**
- **Imported unchanged:** the WL80D sampler (`dr.eq`) correction.
- **Implemented here, identically for both arms:**
  - Horizon-safe closed forms for the GGX lobe and VNDF pdf. The WL80C/80D runtime defect is described in pre-freeze item 6.
  - One harness correction: the C5b supplement on synthetic closures, because the WL80C eligibility set was empty in dev.
- **Data-generation fixes (amendments 2–3):**
  - GPU-side reduction;
  - parallel shards;
  - Dr.Jit PCG32 stream-overlap fix.

**Energy arms.**
- Control: WL80C 9-slot softmax (Σλ ≤ 1).
- Candidate: amendment 1, `C81_limit_complete_sup`, softplus weights with f = g / max(1, M), where M is the supremum over the closed hemisphere including μ = 0. It is labelled **empirically passive, not certified**.
- Diagnostic: unbounded (softplus, no bound).

**Not done.**
- Nothing was trained.
- No compiler, GPU benchmark, compression comparator or rendering run exists for WL81. The Stage D/E/F code was dev-smoked only, as non-evidence.
- Production FNTO, WL80C/80D code and their evidence are unchanged.

### Measurement

**Reference quality (Stage A).**
- Reference noise floor, half-vs-half / 2, 4096 walks:
  - E_L1 median 0.004–0.007, p90 ≤ 0.015;
  - E_alb median ≤ 0.0002, p95 ≤ 0.0008;
  - all at least 4× below the thresholds.
- Highlight peak, half-vs-half: 0.36° / 0.37° (gate / test_iid), so about 0.18° at full N. Width ≤ 0.002.
- Pronounced-highlight cases: 1119 of 1200 (gate).
- Generation throughput: about 2e7 walks/s per process; 13 concurrent processes saturate the RTX 3080 Ti. In the A0 microbenchmark, `layered_eval` ran at 3.2e7 walks/s.

**Stage B (frozen WL80C parameters).**
- B1 lane selection: 0 mismatches on 2^20 lanes.
- B2 eval vs torch: median rel 1.0e-7. pdf vs float64: median rel 8.9e-8.
  - One lane differs by rel 198, at the original horizon clamp; this is the defect that fix 2 addresses.
- B3: sampler albedo and valid fraction agree with quadrature.
- B4: the WL80C control metrics are reproduced to 3.9e-14.
- B5: WL80C C1–C7, C4 and C5b all PASS on the frozen control closures and the WL80D adversarial set. Report-only sampling utility vs the principled sampler: median ratio 0.77.
- B6 horizon fix:
  - away from the horizon, the corrected forms equal the originals to rel 4.3e-7 (eval) and 3.7e-7 (pdf);
  - single-lobe albedo (weight 1, F0 1) at μ = 2e-7 is 1548–2638 with the original numerics and ≤ 1 + 4 se with the corrected ones;
  - for α ≥ 0.3 the corrected albedo is within ±0.003 of 1;
  - sharp lobes (α 0.004, 0.05×0.0125) are noise-limited under this uniform-hemisphere estimator (se 0.87 / 0.075).
- **SAMPLER_CORRECTNESS: PASS.**

**Stage C, gate set (300 materials × 8 θ × 2 φ × 1024 wo), direct fits.**

| Arm | E_L1 median / p90 | E_alb median / p95 | Pronounced-highlight peak err / \|log w\| (median) | Grazing E_L1 | R1 / R2 / R3 |
|---|---|---|---|---|---|
| control (WL80C) | 0.213 / 0.418 | 0.110 / 0.368 | 0.75° / 0.192 | 0.187 | F / F / F, clear fail |
| candidate (C81) | 0.183 / 0.411 | 0.084 / 0.345 | 0.77° / 0.266 | 0.209 | F / F / F, clear fail |
| unbounded (diagnostic) | 0.183 / 0.401 | 0.083 / 0.338 | 0.77° / 0.267 | 0.208 | F / F / F, clear fail |

- Thresholds: E_L1 median ≤ 0.10 and p90 ≤ 0.25; E_alb median ≤ 0.03 and p95 ≤ 0.10; peak ≤ 1°; |log w| ≤ log 1.15 = 0.14.
- **Candidate compile:**
  - 100.9 s for 300 materials (0.34 s per material);
  - M median 0.46, max 8.8;
  - M > 1, i.e. actually scaled, for only 2.3 % of materials;
  - the argmax of the mixture albedo lies at the exact grazing limit μ = 0 for 100 % of materials;
  - refinement margin ≤ 3.1e-6; local-variation margin ≤ 0.019;
  - runtime Σλ' up to 1.74;
  - runtime state: 53 floats, identical to the control.
- **Physical contract (gate fits + 50 adversarial closures): PASS for both arms.**
  - C1, C2 (max rel 1.8e-5 / 1.9e-5), C4, C5a, C5b (60 eligible cases; failure fraction 1.7 % / 3.3 %, candidate min p 2.7e-58 in one case), C6 and C7 all pass.
  - C3: 0 violations.
  - G81 grazing limit: 0 violations in 307,200 + 51,200 random-wi cases per arm. Candidate max estimate 1.009 at fixed grazing angles, within 4 se.
  - Report-only sampling utility vs the reference Sample_f: median variance ratio 0.20 / 0.21.
- **Strata (candidate, per-material mean E_L1 by tercile):**
  - optical thickness τ: thin 0.254, mid 0.216, thick 0.127;
  - conductor α_b: sharp 0.283, mid 0.207, rough 0.141;
  - conductor reflectance: low 0.145, mid 0.177, high 0.247.
- **Held-out incident band vs its neighbours:** 0.169 vs 0.164 (candidate).

**Post hoc attribution** (labelled post hoc, run after the frozen verdict; gate set; same basis):

| Arm | Frozen loss, 12000 steps | Frozen loss, reinit, 12000 steps | Metric-matched objective, 6000 steps |
|---|---|---|---|
| control | 0.186 / 0.088 | 0.187 / 0.088 | 0.129 / 0.065 |
| candidate | 0.173 / 0.080 | 0.183 / 0.084 | 0.129 / 0.072 |
| unbounded | 0.173 / 0.080 | 0.174 / 0.079 | 0.129 / 0.072 |

- Values are E_L1 median / E_alb median.
- R3 passes in every post hoc variant (peak 0.65–0.68°, |log w| 0.063–0.093). R1 and R2 fail in every variant.

### Observation

1. **The fixed R = 8 closure does not reach the frozen representation thresholds on this family under any energy parameterisation.**
   - This holds for direct per-material fits, which are the capacity ceiling for a learned compiler.
   - The failure is a clear fail: E_alb median 0.08–0.11 against a 0.06 clear-fail bound.
2. **The energy constraint is not the binding factor here.**
   - The candidate and the unbounded diagnostic are identical to three decimals, because M ≤ 1 for 97.7 % of materials.
   - The control is about 0.03 E_L1 worse under the frozen loss and equal under the metric-matched objective.
3. **Optimisation under the frozen loss is converged.** 4× more steps and a new random init change E_L1 by ≤ 0.014.
4. **The frozen log1p objective accounts for part of the gap.** The metric-matched objective lowers E_L1 median to 0.129 and p90 to 0.25 for every arm. It still fails R1 (0.10) and R2 (E_alb 0.065–0.072 vs 0.03).
5. **Where the error sits.** It concentrates in thin slabs over sharp, highly reflective conductors.
   - Example: the worst material #139, τ 0.022, α_b 0.059.
   - Its reference albedo falls with incidence, from 0.59 at normal incidence to 0.35 at grazing. Every closure arm stays flat or rises.
   - The fits emulate the refracted base lobe with extreme anisotropic lobes (cross-shaped streaks in `stageC_highlight_grids.png`).
   - Separately, the frozen-loss fits add a very narrow mirror-direction spike that the reference does not contain (median and thick-white examples).
6. **Highlight geometry is not the limiting criterion.** Peak position is within 1° in every arm. Width passes R3 with longer optimisation or the metric objective.
7. **The WL80C/80D runtime is non-passive near the horizon.** The single-lobe albedo reaches about 2.6e3 at μ = 2e-7. This holds for the control too, whose sum bound is a mathematical certificate that the implementation did not honour.
8. **The pbrt-v4 `LayeredBxDF` estimator, as ported here, is biased** by its exit-strategy MIS densities: about 5 %, with reciprocity z up to 79. This is a statement about our port. The pbrt-v4 MIS weight was read from its source but not executed in pbrt itself.

### Interpretation (inference, not a measurement)

**The limit is basis-structural** (rows 2–5 of the attribution):
- the weights do not depend on wi;
- the lobes are Schlick-Fresnel reflection GGX, whose albedo is non-decreasing towards grazing.

Together these cannot express an albedo that falls with incidence, nor a refracted, coat-attenuated base lobe whose shape and Fresnel differ from a reflection GGX. Coated conductors with thin slabs over sharp, highly reflective bases are exactly such responses.

A learned compiler cannot exceed its own direct-fit ceiling. So on this family the learned-rule question cannot become positive under the frozen basis, whatever the energy parameterisation.

### Unresolved Question

- **Generality of the attribution.** It rests on the gate set of one family. Whether the same structural limit appears for other expensive layered or measured families is unmeasured.
- **Learned-compiler generalisation** (the transfer gap beyond the direct-fit ceiling) is untested. It was not trained, per the frozen rule.
- **GPU, compression and rendering utility against the expensive reference are unmeasured.** The dev smoke executed them only as non-evidence.
- **The WL80C/80D results computed with the horizon-defective runtime** (sampler albedo tests at θ ≤ 88°, which do not reach the defect band, and the WL80D grazing diagnostic, which used the bounded VNDF integrand) were not re-evaluated. They stay as recorded.
- **The highlight peak criterion is quantisation-limited** on the 201-point grid at sharp coats (floor about 0.18°).

### Architecture Verdict

| Entry | Verdict | Evidence | Scope | Confidence | Important negative cases |
|---|---|---|---|---|---|
| SAMPLER_CORRECTNESS | **PASS** | B1–B5 on frozen WL80C parameters; B6 horizon fix | WL80C closure family, corrected runtime | High (exact / statistical tests) | Original runtime non-passive above 89.98° (all arms) |
| REPRESENTATION_CAPACITY | **FAIL (clear fail)** | Gate-set direct fits; post hoc ceiling with the metric objective still 0.129 / 0.065–0.072 | Coated conductor, 300 gate materials, local reflection | High (reference noise ≤ 1/4 of the thresholds; three arms; three post hoc variants) | Thin slab + sharp, high-r base (worst E_L1 0.69–0.73) |
| LEARNED_RULE_GENERALIZATION | **NOT TESTED** (stopped before training by the frozen rule) | — | — | — | — |
| PHYSICAL_VALIDITY | **PASS for direct fits** (control certified; candidate empirical, not certified) | WL80C C1–C7, C5b, G81 with 0 violations | Gate fits + adversarial, corrected runtime | Moderate–high | Candidate fixed-grazing estimate 1.009 (within 4 se); one candidate χ² case p 2.7e-58 (within the 5 % rule) |
| COMPRESSION_UTILITY | **NOT REACHED** | — | — | — | — |
| GPU_RUNTIME_UTILITY | **NOT REACHED** | — | — | — | WL80C/80D principled parity finding stays |
| RENDERING_UTILITY | **NOT REACHED** | — | — | — | — |
| FNTO_COMPONENT_VIABILITY | **NOT SUPPORTED** (by the frozen mapping: REPRESENTATION_CAPACITY fails) | Above | One family, local reflection | Moderate (one family) | — |

**Completion question.** Does the fixed Compiled Neural Material Rules architecture offer a physically sound, generalizable and computationally useful way to compress complex local material scattering for ReST-GI?

**Answer: negative on this evidence.**
- The corrected closure is physically sound: the sampler fixes and the horizon fix pass, and passivity holds empirically for the candidate and by certificate for the control.
- But its fixed R = 8 basis cannot represent the qualified expensive family to the declared quality even per material.
- So generalisation and utility were not tested.

No adoption. The canonical FNTO architecture is unchanged. No successor.

### Reproducibility

- **Commits:**
  - protocol v1 + code `e4b471e`;
  - amendment 1 `2ac5aed`;
  - worklog `aeb532d`;
  - amendment 2 `aca745a`;
  - amendment 3 `4db2941` (the chain);
  - post hoc diagnostic `f5acf7d`;
  - visuals `viz81.py` (this commit).
- **Server:** LabServer63 `~/rest_compiled81` (`code/COMMIT`, `chain.status`, `results/`).
- **Local copy:** `results/fnto_compiled81/` (JSONs, lanes, states, logs, materials). The large target npz files are listed by sha256 in `results/data_sha256.txt`.
- **Seeds:** protocol `seeds` (materials 81010, directions 81020, walks 81030, …); post hoc 81900.
- **Environment:** Mitsuba 3.5.1 `cuda_ad_rgb` (llvm for the slab check), Dr.Jit 0.4.6, torch 2.3.1+cu121, RTX 3080 Ti, `LD_LIBRARY_PATH=$HOME/nvlibs/optix595`.
- **Visuals:** `results/evaluation/81/` (manifest inside):
  - in-plane lobe slices;
  - albedo vs θ;
  - highlight grids;
  - E_L1 histograms.

  Every visual is computed on the gate-set fits and the reference targets the metrics used.
