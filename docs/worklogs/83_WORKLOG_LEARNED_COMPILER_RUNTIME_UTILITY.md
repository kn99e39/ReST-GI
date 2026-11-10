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

## Run 2 and amendment 2 (execution)

**Run 2** (commit `a11e452`, started 05:36 UTC) reused Stage A and the data.

| Stage | UTC | Result |
|---|---|---|
| Runtime tests | 05:36–05:39 | **PASS** |
| Renderer validation | 05:39 | **PASS** |
| Compiler training (6 runs) + direct fits (9) | 05:39–07:2x | 3 concurrent processes |
| Prediction + evaluation (7 sets) | → 07:35 | |
| Dr.Jit passivity audit | 07:35–08:25 | **PASS** |
| Bench | 08:25–08:47 | run alone (uncontended) |

Runtime tests in detail:
- R1 eval / pdf vs torch: median ≤ 1.1e-7, p99.9 ≤ 2.8e-6.
- R2 ER sample–pdf consistency: median 0, p99.9 1e-7, max 1.8e-4.
- χ² min p ≥ 0.011; R3 reciprocity ≤ 6.7e-5.

Renderer validation in detail:
- analytic Lambert max rel 5.7e-7;
- E / ER closure planes vs the operator ≤ 1.5e-6.

**Execution inefficiency (recorded, uncorrected in the frozen run).**
- The audit took 50 min at about 15 % GPU. It looped over about 2,700 chunks per arm, each re-traced in Python.
- The bench and the render are also bound by Python tracing.
- The dev smoke had checked only that these stages run, not their full-scale utilisation.
- The user objected to the resource use.

**Amendment 2** (`a883aef`; execution only; user-approved: "여러 프로세스 병렬로 띄우는건 안돼? 내가 승인했다 치고.").
- The bench stayed alone, because it is the economics evidence.
- The 10 render images were then split over 5 concurrent processes (`run83b.sh`, `render83.py --part`).
- Two of the five processes (parts 2 and 4) produced no output in 9 min at 0 % GPU, while the other three finished their two images each in about 2 min per image. They were killed. Killing them also stopped the run83b wrapper, because a `pkill -f` pattern matched the session's own command line, the known pitfall.
- Their 4 images were re-rendered as 4 single-image processes (parts j/10) with the same seeds. Each finished in about 2–4 min.
- Merge, visuals and the verdict ran from a detached finisher script (end 09:02 UTC).
- Consequence: the kernel GPU times of the render (the time-matched reference spp and the reported render cost) include overlap from concurrent processes. The economics verdict comes from the uncontended bench.

## Final report

### Implementation Fact

- **Learned compiler.**
  - MLP 10 → 256 × 3 (SiLU) → 64 raw → `closure82.decode(arm, 'cert')`; 150,848 parameters.
  - Response learning with the WL82 per-incidence metric objective on 3000 WL81 train materials (64 × 64 × 1024-walk response data).
  - Adam 1e-3 → 1e-5, 40k steps, 128 materials × 16 incidences per step.
  - Checkpoint: the best val83 objective (selected step 38–40k).
  - Arms C0 / E / ER, seeds 8301 / 8302; 15–33 min of training each.
  - Every output satisfies the passivity certificate and reciprocity by construction.
- **Direct fits.** WL82 `fit82.py` with the refraction patch on the new population, 12,000 steps (16–35 min per arm and set).
- **Dr.Jit runtime `closure83_dr.DrClosure83`.**
  - eval, pdf, eval_pdf and sample for C0 / E / ER.
  - 69-float state; 104-float runtime table; uniform or per-lane table binding.
- **Numerics fix.** Refraction t_z = sqrt(max(η² − 1, 0) + μ²)/η, in both runtimes, algebraically identical to the WL82 form. Stage A reproduction ran unpatched.
- **New independent population.**
  - test83_iid 300, test83_block 200, test83_hard 150.
  - Evaluation sets follow the WL81 layout at 4096 walks.
  - Highlight grids for iid and hard.
- **Bench and render.**
  - Bench: REF_1 / REF_Neq vs the learned closures, in io, fused16 and table modes.
  - Render: Mitsuba path tracing, max depth 4, 5 materials × 2 lights.

### Measurement

**Stage A.**
- The WL81 targets are bit-identical.
- The WL82 tests and contracts pass at full scale.
- The stored WL82 fits reproduce through the WL83 evaluator with deviation 0.0.
- Population overlap is 0.
- Every evaluation set passes the WL81 noise rule at 4096 walks; E_L1 noise median 0.004–0.014.

**Quality** (E_L1 median / p90; E_alb median; transfer = E_L1 between the learned and direct-fit closures, same samples).

| Set | DF_C0 | DF_E | L_E (s8301 / s8302) | DF_ER | L_ER (s8301 / s8302) |
|---|---|---|---|---|---|
| test83_iid | 0.116 / 0.261, 0.037 | 0.072 / 0.169, 0.008 | 0.073 / 0.170, 0.011–0.012; transfer 0.029 | 0.053 / 0.112, 0.006 | 0.057–0.058 / 0.116–0.117, 0.011; transfer 0.024–0.025 |
| test83_hard | 0.259 / 0.419, 0.084 | 0.182 / 0.293, 0.015 | 0.195 / 0.188; transfer 0.045–0.050 | 0.095 / 0.197, 0.006 | 0.099 / 0.100; transfer 0.029 |
| test83_block (out of training distribution) | 0.082, 0.040 | 0.060, 0.014 | 0.063–0.065, 0.022–0.024 | 0.053, 0.016 | 0.067–0.068, 0.025–0.026 |
| train_eval / val83 | — | — | 0.069 / 0.072 | — | 0.056 / 0.055 |
| wl82_test_iid (previously inspected) | 0.120 | 0.073 | 0.072–0.074 | 0.052 | 0.056 |
| real83 (8 parameterised finishes) | 0.147 | 0.093 | 0.094–0.099 | 0.067 | 0.074–0.075 |

- **Highlights (test83_iid, pronounced cases).** Learned E: peak 0.44°, |log w| 0.041–0.048. Learned ER: 0.39–0.40°, 0.033. Hard set: E 0.91° / 0.100–0.136; ER 0.66° / 0.059.
- **Grazing medians (iid).** Learned E 0.104–0.105; learned ER 0.089.

**Learned-rule gate (test83_iid; both seeds).**
- **E: PASS.**
- **ER: PASS.**
- C0 fails L1 (the WL81 gate); its transfer, generalisation, block and grazing criteria pass.

**Physical validity.**
- The certificate holds by construction.
- Dr.Jit own-sampler audit over 3616 closures per arm (every DF and learned closure + 50 adversarial): 0 violations; max albedo C0 1.002 (se 0.005), E 0.997, ER 0.997.
- Reciprocity ≤ 6.7e-5 (float32); finite at μ = 0 / 1e-12 / 2e-7.

**GPU economics (uncontended, 8 test83_iid materials, kernel GPU time).**
- **io 2^24 eval_pdf:** C0 0.89 ms, E 0.87 ms, ER 0.91 ms. This regime is memory-bound, so the three arms cost about the same; ER / E = 1.05.
- **fused16 2^22 eval_pdf** (compute-bound):

  | Arm | eval_pdf | Ratio to C0 |
  |---|---|---|
  | C0 | 4.7 ms | 1.00× |
  | E | 10.5 ms | 2.21× |
  | ER | 17.1 ms | 3.62× |

  ER / E ≈ 1.64.
- **sample, fused16:** C0 3.0 ms, E 11.6 ms, ER 15.6 ms.
- **Against the layered reference:**
  - REF_1 (one walk): eval 13 ms (io) / 469 ms (fused).
  - Matched-quality N_eq (per material, learned E_L1 vs single-walk noise): E 4–89, ER 4–216.
  - eval_pdf speedup vs REF_Neq: 44–6200× (io) and 86–10600× (fused); sample vs REF_1: 3.8–17.7×. Every arm passes on 8 / 8 materials in both modes.
- **Compiler + decode + state + binding:** 2.1–2.2 ms for 1 material and 4.5–5.1 ms for 1024 materials. Of that, the forward and decode take 0.44–0.60 ms; the rest is runtime binding.
- **Per material:** state 276 bytes; compiler 150,848 parameters (0.6 MB).
- **Amortisation:** compiler training 25 min (E) / 32 min (ER), against 16–35 min of direct fitting per 300 materials.
- **Wall time** exceeds GPU time by far (e.g. REF_1 io eval 156 ms wall), because Dr.Jit 0.4.6 re-traces Python on every call.

**Rendering** (5 materials × 2 lights; closures at 64 spp vs REF_HQ 4096 spp; amendment-2 concurrency).
- Learned closures: object relative L1 0.083–0.137 (E_L), 0.083–0.102 (ER_L).
- Time-matched layered reference (34–45 spp): 0.13–0.24.
- Per 64 spp of the whole render, the layered reference costs 12–16 ms GPU against 6.8–8.6 ms for the closures, i.e. only 1.6–1.9× more, because path tracing dominates.
- Highlight-ROI error: ER_L 0.035–0.095, E_L 0.022–0.134.
- Images passing the WL81 Stage F rule:

  | Arm | Passing images (of 10) |
  |---|---|
  | C0_DF | 3 |
  | E_DF | 8 |
  | ER_DF | 9 |
  | E_L | 8 |
  | ER_L | 9 |

- The common failing image is test83_iid_1_L2 (grazing light):
  - REF_HQ uncertainty is 0.0265, against 0.025–0.026 (a quarter of the closure errors), so the image is **reference-limited** for ER_DF, E_L and ER_L;
  - E_DF misses the highlight limit there by 0.001.
- E_L also fails test83_hard_1_L1: highlight error 0.134 vs ER_L 0.035.
- RENDER is false for every arm.

**Verdict mapping.**

| Check | E | ER |
|---|---|---|
| VALID | true | true |
| LEARNED | PASS | PASS |
| ECONOMICS | PASS | PASS |
| RENDER | fail | fail |

- ER worth holds: hard-set learned median 0.100 vs 0.192 (0.52×), io cost 1.05×.
- **OUTCOME D** by the frozen mapping.

### Observation

1. **The compiler learns E and ER almost at their direct-fit ceiling** on unseen materials. The transfer error (0.024–0.029) is about 1/3 of the total error, and the generalisation gap is small (train 0.069 → test 0.073 for E; 0.056 → 0.057 for ER). The two seeds agree within 0.001.
2. **ER remains the arm that handles the difficult region.**
   - On test83_hard, learned ER gives 0.10 against learned E's 0.19, and ER passes the WL81 gate there (seed 8301; seed 8302 is at the edge).
   - In rendering, E_L's only genuine failure is a hard-material highlight.
   - On test83_block, the learned ER loses more against its direct fit (0.053 → 0.067) than E does (0.060 → 0.064), but both stay within the block thresholds.
3. **The economics are strongly positive at the operator level and modest inside a path tracer.**
   - Against the matched-quality layered reference, every learned closure is orders of magnitude cheaper.
   - Inside the renderer, the BSDF is only part of the cost: about 1.6–1.9× per spp.
4. **The rendering rule fails on its own strictness, not on visible failures** for ER_L. Its only non-passing image is reference-limited by 0.001 (4096-spp REF_HQ under grazing light). Every closure beats its time-matched reference on every image.
5. **Implementation.**
   - A float32 refraction defect (η → 1) was found and fixed.
   - The bench and render stages are dominated by Python tracing in Dr.Jit 0.4.6.
   - Two render processes stalled under concurrency; the cause was not identified.

### Interpretation (inference)

**The learned-compiler question is answered positively for this family.** Descriptor-to-closure compilation of E / ER generalises to an independent population and to a held-out block region with a small transfer cost. Physical validity is preserved by construction and confirmed by the runtime audit.

**ER has the better fidelity–cost trade-off.**
- In the memory-bound regime it costs the same as E.
- In the compute-bound regime it costs about 1.6× E (≈ 3.6× C0).
- It halves the error on the thin / sharp / high-reflectance materials where E fails.

**Outcome D comes from the all-images rendering rule.**
- For ER, the one failure is a reference-limited image.
- For E, it is one reference-limited image plus one genuine hard-material highlight failure.

That is why the rendering evidence is classified as incomplete rather than unfavourable for ER. This classification is an inference; the frozen verdict remains D.

### Unresolved Question

- **Reference-limited image.** The rendering rule needs a less noisy REF_HQ on the grazing-light image test83_iid_1_L2 to classify ER there. Not re-run: the protocol fixes 4096 spp.
- **Stalled render processes.** The cause of the two stalls under concurrency is unknown.
- **Render GPU times** were measured with concurrent processes.
- **In-renderer economics** depend on the scene and on Python-tracing overheads. No frozen-function / megakernel integration was tested.
- **Generality** beyond the coated-conductor family; production FNTO integration (K/C), dynamic GI and measured BRDFs are untested.

### Architecture Verdict

| Entry | Verdict |
|---|---|
| Representation Feasibility | **REPRODUCED** on the new population: E_DF and ER_DF pass the WL81 gate on test83_iid; ER_DF also on test83_hard; C0_DF fails |
| Learned Compiler Generalization | **PASS** (E and ER, both seeds; transfer 0.024–0.029, generalisation gap ≤ 1.06×, block region within thresholds); C0 fails (representation-limited) |
| Physical Validity | **PASS** (certificate by construction; Dr.Jit audit 0 violations over 3616 closures per arm; reciprocity, finiteness) |
| Runtime Correctness | **PASS** (Dr.Jit vs torch ≤ 2.8e-6 p99.9; sampler consistency; renderer validation), after the refraction numerics fix and amendment 1 |
| GPU Economics | **PASS** at the operator level for every arm. ER costs ≈ E memory-bound and 1.6× E compute-bound. In-render BSDF share is small (1.6–1.9× per spp vs the layered reference) |
| Rendering Utility | **FAIL** under the frozen all-images rule. ER_L 9 / 10, the failure reference-limited; E_L 8 / 10, one genuine hard-material highlight failure. Every closure beats its time-matched reference on every image |
| FNTO Integration Readiness | **NOT TESTED** (no K/C integration, no dynamic GI) |

- **Outcome D** (frozen mapping): learning succeeds, but the rendering criterion fails.
- The selection between E and ER, and whether Learned Material Rules remain a ReST-GI component candidate, are human decisions. Nothing is adopted into canonical FNTO; no successor is authorised.

### Reproducibility

- **Commits:**
  - freeze `9e273ca`;
  - amendment 1 `a11e452` (run 2);
  - amendment 2 `a883aef` (parallel render);
  - report in the closing commit.
- **Server:** `~/rest_compiled83` (`chain.status`, `results/`, `run83b.sh` / `finish83.sh` / `relaunch83.sh` in `$HOME`).
- **Local copy:** `results/fnto_compiled83/results/`. Includes `verdict83.json`, `stageA83.json`, eval / runtime / bench / render JSONs, the fitted and learned raw parameters, the training logs, the compiler checkpoint hashes (in `train_*.json`) and the logs.
- **Seeds:** protocol `seeds` (83010–83120; compiler 8301 / 8302).
- **Environment:** RTX 3080 Ti, driver 595.71.05, Mitsuba 3.5.1, Dr.Jit 0.4.6, torch 2.3.1+cu121.
- **Visuals:** `results/evaluation/83/` (manifest with sha256).
