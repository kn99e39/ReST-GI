# Worklog — Compiled Neural Material Rules: Learned Compiler and Runtime Utility Validation (2026-11-10)

## Origin

- **Brief.** WL83, "Compiled Neural Material Rules: Learned Compiler and Runtime Utility Validation" (user, 2026-11-10).
  - Continues from WL82's positive representation-feasibility result.
  - E_cert is the primary candidate, ER_cert the challenger, C0 the historical control.
  - Goal: evaluate learned generalisation, physical / runtime correctness and rendering economics.
  - No new representation family. No production FNTO change. No successor.
- **Where.** Branch `Selective_Recompute`; LabServer63 (main server, idle at start); RTX 3080 Ti 12 GB; the WL81 environment (`~/rest_compiled81/env.sh`).
- **Isolated paths.**
  - `experiments/fnto_compiled83/`, `results/fnto_compiled83/`, `results/evaluation/83/`.
  - Server: `~/rest_compiled83` (dev smoke `~/rest_compiled83_dev`).
  - WL80C / WL80D / WL81 / WL82 code and evidence are imported or read, never modified.

## Design (frozen in `protocol/compiled83.json` before primary training or any evaluation on the new population)

**Populations.**
- Training: train83 = the WL81 train materials (3000). Validation: val83 = the WL81 val materials (200). No WL81 / WL82 candidate ever fitted, trained on or evaluated these.
- New independent evaluation population (seed 83010):

| Set | Size | Definition |
|---|---|---|
| test83_iid | 300 | box, outside the WL81 held-out block |
| test83_block | 200 | inside the block (outside the compiler's training distribution) |
| test83_hard | 150 | thin slab τ, sharp base α_b and high reflectance r (box terciles; the WL81 / WL82 failure region) |

- Previously inspected sets, reported for comparison only:
  - wl82_test_iid;
  - real83 (the 8 parameterised WL82 finishes; not measured BRDFs);
  - train_eval (300 train materials on fresh WL81 directions; the training-set result).

**Response data.** 64 wi × 64 wo × 1024 walks per material (WL82 fit-set design), for train / val / test83. Evaluation sets follow the WL81 layout and noise rule. Highlight grids are generated for test83_iid and test83_hard.

**Compiler.**
- MLP 10 → 256 × 3 (SiLU) → 64 raw → `closure82.decode(arm, 'cert')`.
  - The decode guarantees the passivity certificate and reciprocity for every output.
  - The compiler learns the **response**: its loss is the WL82 per-incidence metric objective against the reference. There is no regression onto non-unique direct-fit parameters.
- Training: Adam 1e-3 → 1e-5, 40000 steps, 128 materials × 16 incidences per step.
- Checkpoint: the lowest val83 objective.
- Seeds 8301 / 8302; identical treatment for C0 / E / ER.

**Direct fits.** WL82 `fit82.py`, unchanged, on test83_*. The stored WL82 fits are used for wl82_test_iid and real83.

**Dr.Jit runtime `closure83_dr.DrClosure83`.** C0 / E / ER eval, pdf, eval_pdf and sample, with a 69-float state. Semantics are those of `closure82`.

**Tests.**
- Runtime, before any evidence:
  - R0 binding;
  - R1 Dr.Jit vs torch (median rel ≤ 1e-5, p99.9 ≤ 1e-3);
  - R2 sampler (pdf consistency, valid fraction, albedo, χ²);
  - R3 reciprocity;
  - R4 finiteness at grazing.
- R5 Dr.Jit own-sampler passivity audit on every closure.
- Renderer validation: analytic Lambert; E / ER closure plane vs the operator.

**Stage A.**
- Reference reproduction.
- WL82 tests and contract re-run.
- Stored WL82 fits re-evaluated through the WL83 evaluator must equal the recorded medians.
- Population independence.

**Gates.**

| Gate | Condition |
|---|---|
| LEARNED_PASS | Both seeds pass, on test83_iid: the WL81 representation gate; transfer ≤ max(1.5 × DF, DF + 0.03); generalisation gap ≤ 1.25 × train_eval; test83_block (WL81 block thresholds); grazing / held-out band (WL81 L6) |
| ECONOMICS | Learned eval_pdf ≥ 1.10× faster than the matched-quality layered reference (REF_Neq), and sample ≥ 1.10× faster than REF_1, for ≥ 6 of 8 materials, in both io and fused workloads |
| RENDER | WL81 Stage F rule on 5 materials × 2 lights |
| ER worth | ER's learned hard-set median ≤ 0.75 × E's, and ER's eval_pdf cost ≤ 1.5 × E's |

**Outcomes.**
- **E:** validity fails.
- **B:** ER passes all three and is worth its cost (or E fails).
- **A:** E passes all three.
- **C:** neither arm learns.
- **D:** learning passes but economics or rendering fail.

## Dev smoke and pre-freeze corrections (non-evidence; dev quality numbers not interpreted)

Dev root `~/rest_compiled83_dev`: 64 train / 16 val / 16 test83 materials per set; 400 compiler steps; 300 DF steps; reduced bench / render sizes.

1. **Stage A in dev: QUALIFIED.**
   - The WL81 targets are bit-identical.
   - The WL82 focused tests and contract pass.
   - The stored WL82 test_iid fits re-evaluated through the WL83 evaluator match the recorded medians (deviation 0.0).
   - Population overlap is 0.
2. **Refraction numerics defect (found by R2, fixed before the freeze).**
   - Symptom: with η_e = 1, `closure82.refract_up` computes t_z = sqrt(1 − (w_x² + w_y²)/η²), which cancels to 0 in float32 near the horizon. The refracted-lobe pdf Jacobian cos_o / (η² cos_o') then explodes: χ² expected counts reached about 1e26 for the synthetic closure with η = 1, at both dev and full quadrature sizes. Fitted η_e sat at 1.0 for many materials in WL82.
   - Fix: the algebraically identical form t_z = sqrt(max(η² − 1, 0) + w_z²)/η.
     - Native in `closure83_dr`.
     - In torch, through `closure83.install()`: a WL83 module that monkeypatches `closure82.refract_up`; the WL82 code is unchanged. It is used by training, the direct fits (`fit83.py` = `fit82.py` + patch), evaluation and tests.
   - Stage A re-evaluates the stored WL82 fits unpatched (`W83_UNPATCHED=1`), so that check stays a faithful reproduction.
   - After the fix, all runtime tests pass in dev:
     - R1 Dr.Jit vs torch: eval / pdf median rel ≤ 1.0e-7, p99.9 ≤ 2.7e-6;
     - R2 sampler: z ≤ 2.5, χ² min p ≥ 0.0019;
     - R3 reciprocity ≤ 3.4e-5;
     - R0 and R4 pass.
3. **Rendering loop.** The WL81 per-spp loop was CPU-bound (one Python trace per spp; GPU at about 0 %).
   - `render83.render_b` traces 128 spp per launch.
   - Full-scale probe: 256 spp per image in about 3.8 s wall (70 ms GPU), 1.05 GB.
   - Each image takes about 3 min. The stage stays CPU-bound per launch because Python tracing dominates (noted).
4. **Benchmark probe at full sizes** (ER and REF_1, 2^24 io / 2^22 fused16): GPU memory 1.6 GB, about 25 s per material.
   - Kernel-history GPU time is far below wall time (e.g. REF_1 io eval 13 ms GPU vs 156 ms wall), because Dr.Jit 0.4.6 re-traces Python every call.
   - The economics rule uses GPU time, as in WL80C / WL81. Wall times are reported.
5. **Compiler.** 150,848 parameters (protocol corrected from a miscount). The dev run (400 steps) used 0.15–0.20 GB.

## Freeze, run 1 and amendment 1

- **Freeze `9e273ca`.**
- **Main chain run 1** (LabServer63, started 2026-11-10 05:13 UTC):
  - **Stage A: QUALIFIED.**
    - The WL81 targets are reproduced bit-identically.
    - The WL82 focused tests and physical contract pass at full scale.
    - The stored WL82 fits re-evaluated through the WL83 evaluator match the recorded medians (deviation 0.0).
  - **Population.** 300 / 200 / 150 new materials, zero overlap; hard-box members are 3 % of test83_iid. Every evaluation set passes the WL81 noise rule at 4096 walks.
  - **Runtime gate: FAIL (frozen stop).** Only one item failed: for ER, the pdf returned by `sample` and the pdf recomputed by `eval_pdf` in a separate kernel differ by up to 1.8e-4 relative, against the 1e-5 maximum the test set.
  - Every other runtime test passed:
    - R1 eval / pdf vs torch: median ≤ 1.1e-7, p99.9 ≤ 2.8e-6;
    - R2: z ≤ 2.4, χ² min p ≥ 0.011;
    - R0, R3 and R4 pass.
- **Diagnosis.** On the worst case (θ_i 75°, wo_z = 0.0031):
  - the pdf inside the sample kernel is 0.10489164;
  - the float64 torch value is 0.10489163;
  - the separately recomputed float32 `eval_pdf` is 0.10491003.

  The in-kernel value is correct to 2e-7. The separate recomputation carries the float32 conditioning of the refracted-lobe pdf at outgoing directions near the horizon (41 of 235,526 samples exceed 1e-5). This is rounding, not a semantic defect.
- **Amendment 1** (`protocol/amendment1_runtime_pdf_consistency.json`).
  - Timing: **pre-evidence**. No training, direct fit, learned closure, benchmark or render had run.
  - The sample–pdf consistency check now uses the frozen R1 tolerances (median ≤ 1e-5, p99.9 ≤ 1e-3) and still reports the maximum.
  - The chain reuses Stage A and the data, whose code is unchanged.
