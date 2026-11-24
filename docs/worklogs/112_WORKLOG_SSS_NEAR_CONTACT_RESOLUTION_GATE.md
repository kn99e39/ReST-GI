# Worklog 112 — SSS near-contact artifact resolution and mandatory correctness gate (2026-11-24)

> Append-only. Written for the research owner and later agents. The batch follows the owner brief "SSS Near-Contact Artifact Resolution and Mandatory Correctness Gate" (2026-11-24).

## Batch class, branch, machine

- **Number.** Regular **WL112**. WL110/WL110A are on this branch. WL106–WL111 exist on `Denoise_Reconstructor`; WL111 is the concurrent external baseline benchmark. WL112 opens a new hypothesis: a camera-independent spatial HIGH representation with a conditional learned candidate. It is not a suffix of WL101I.
- **Branch.** `ecro_lr3_adapter` (base `5709d8d`). No new branch or worktree.
- **Machine.** Local RTX 5080. This has been the default machine since 2026-11-20.
- **Software stack.** WL100's training and reference stack, `external/8dna26/.venv-cu128`: Python 3.11, Mitsuba 3.5.1, Dr.Jit 0.4.6, PyTorch 2.8.0+cu128.
- **No pairing with older references.** The WL101A–I raw arrays live only on LabServer63, which is no longer used, and were rendered with a Mitsuba 3.8.0 stack. Their numbers are cited below as history only. No WL112 render is paired with them.
- **Concurrent work.** No shared production file is edited. WL112 touches only `experiments/ecro_sss112/`, `results/ecro_sss112/`, `evaluation/112/` and this worklog.

## What earlier work established (read before design)

| Worklog | Result carried into WL112 |
|---|---|
| WL100 | SSS ECRO v0, trained with a fixed sphere in floor contact. Physical LOW = events 1–2. Learned HIGH = events ≥ 3, read at event 2. |
| WL101A/C | Near-floor HIGH excess is stable and reproducible in raw 4,096-spp renders (+0.0699 luminance, +32% HIGH). LOW is clean. Low-spp grain is stochastic. |
| WL101B | The SVGF silhouette bias is a separate residual. |
| WL101D | The WL100 supervising sampler mis-assigns first hits by 1 mm proximity (Box B/floor coplanar band, sphere tangency). It counts back-facing samples in the denominator. |
| WL101E | Qualified uniform U table. The excess belongs to the representation/readout side (+0.0702), not to the generator. |
| WL101F | A camera-conditioned projection P removes 74% of the excess. 2×2 spatial subdivision S leaves 5.8%. Angular subdivision does almost nothing. Diagnostic only, camera-dependent. |
| WL101G | Camera-independent étendue measure G designed. Stopped at the construction gates (verdict D). No evaluation ran. |
| WL101H | F6 confirmed: native spawn origins inside the sphere within ≈ 1.16 cm of the tangency give unattenuated NEE. Contact-valid semantics C passes the analytic chord checks. The v1 gates fail harmless events → NOT QUALIFIED. Descriptive V0 values: U +0.0702, G +0.0415, S_G +0.0106. |
| WL101I | Contract v2 still fails, now on a final-scale back-threshold tie, statistical power, and over-strict control predicates → NOT QUALIFIED. |
| WL105B | Fabric: spatial refinement helps the physical table. The learned model recovers only part of it. |

The historical verdicts are preserved. WL101G remains D, and WL101H and WL101I remain NOT QUALIFIED.

## Design (frozen before evidence; protocol `experiments/ecro_sss112/protocol/sss112.json`)

**Contract v3 (Stage A).** The main changes from WL101E–I:

- **Analytic first surface.** The construction pins the first surface interaction analytically from the patch parameters: point, local normal and shape. There is no attachment ray query. Membership is exact by construction, which removes the attachment numerics behind the WL101E/G/I stops.
- **One front predicate.** The tracer's own front test is evaluated once and governs numerator and denominator together.
- **Registered exclusions.** Excluded first vertices are the resting-box footprint and underside and the contact-valid singular set. Their analytic measures are tested against the observed counts.
- **Rare events are bounded, not stop rules.** Rare numerical events (quantizer address mismatches, singular exclusions) keep their actual per-cell radiance sums. Their effect on any image region is bounded through the unit-marker query weights of the evaluation streams themselves (A8: ≤ 10% of the contact band, or ≤ 2.5e-4 of region RMS).
- **Controls.** Tracer identity with pt100 (A1). Native-frame pinning identity (A2). Frame invariance (A3). Synthetic analytic radiance vs dense quadrature (A4). Support and singular measure (A5). F6 origin discrimination (A6). Legacy-sampler positive control (A7). LOW pairing (A9).

**Semantics.** The primary semantics is contact-valid **C**: the WL101H rule, generalised to any sphere position. Native **N** is kept as a separate factor. Under N, the batch renders references at all six F0 views and builds U and S_G tables, so the semantics change is measured apart from projection and representation.

**Stage B.** Table arms, all from the same samples (nested):

- U, V, G at the original 1,270×64 layout.
- S_U, S_V, S_G at 2×2 sub-patches.
- Eight independent replicas at 1,024 proposals per sub-cell.

Diagnostic arms:

- The WL100 learned table (A).
- N-semantics U and S_G.
- Camera-conditioned P_v / S_v per view (descriptive only).

Evaluation uses the six WL101G cameras. Each has two 4,096-spp streams, with a paired physical pass and readout pass. Contact bands are normalised: near ±6.49% and contrast ±5.79% of the reference near HIGH. These reproduce WL101F's ±0.01403 and ±0.01252 at WL101C's 0.21633. Further requirements: off-target non-regression ≤ 1.10 × U + 0.0025 relative RMSE, coverage ≥ 0.995, and precision ≤ 0.75 × band. The outcome rule, in order: G sufficient → PROJECTION; else S_G sufficient → SPATIAL_G; else S_U sufficient → SPATIAL_U; else NONE.

**Stage C (conditional).**

- **Training data.** A new D112 distribution: the sphere moves (x 0.55–1.25, y 0.10–0.75, 40% in contact, else gap 0–12 cm), with boxes and lights as in WL100. 192 × 4 training fields and 16 × 4 validation fields. Targets come from construction v3 under C semantics.
- **Arms.** Three matched arms share data, seed, schedule and objective (24k steps, Adam, masked OWN):
  - **O_U:** original ECRO, coarse U targets.
  - **O_G:** original ECRO, coarse G targets.
  - **S2:** the candidate ECRO-S2. It is model86 with `head_a` conditioned on an 8-d sub-patch embedding, giving a 2×2 output table. Everything else is unchanged, at +1,056 parameters.
- **Pre-training checks.** Readout reproduction, parent/child identity, linearity and non-negativity, support, and a budget probe.

**Stage D.** F0 at six views and F1 at three views. F1 is an independent static configuration: WL100 test configuration 2 boxes and light, sphere at (0.80, 0.75). The candidate must pass at every contact-valid view:

- (i) endpoints inside the bands;
- (ii) every radial band to 0.40 m inside the near band;
- (iii) near stable relative HIGH RMSE ≤ 10%;
- (iv) non-regression vs O_U in silhouette, sphere interior, contact/shadow, floor outside, and full frame.

**Stage E.** Three rigid sequences with the same parameters throughout:

- **SEQ_A (lift):** gap 0, 1, 4, 10, 4, 1, 0, 4 cm.
- **SEQ_B (slide):** contact maintained, crossing a floor patch edge.
- **SEQ_C (independent):** F1 location and light. Never used for any design decision.

Further requirements:

- Stale-table control.
- Fresh-process rebuild equality.
- Identical tables at revisited states.
- Unchanged far region.
- At least one discriminating transition per sequence: the physical change is resolved at 5 SE and the stale table is materially worse.

**Gate.** RESOLVED requires qualified physics, a Stage B representation, and a learned candidate that passes D and E. PARTIALLY_RESOLVED means the learned candidate halves the near excess vs O_U at every contact-valid F0 view but fails a criterion. NOT_RESOLVED and UNQUALIFIED follow the protocol.

## Development before the freeze (throwaway seeds, offset +5,000,000; reduced sizes)

**Tracer checks.**

- `tracer112.py` copies the WL100 physics statements. Under N semantics it reproduces `pt100.Ctx` lane by lane: tables identical, camera and readout lanes max abs 0.
- Analytic pinning with the native frame reproduces the native first-hit suffix to 2.1e-7 relative L1. With `Frame3f(n)` the same RNG gives a different direction realisation. That is a rotation about n, distributionally identical for the Lambert surfaces, and is the subject of control A3.
- The legacy-sampler copy matches WL100's stored test target at noise level: rel L1 0.072 vs the expected ≈ 0.071.

**Throughput (full-scale probe).**

| Unit | Wall | GPU utilisation |
|---|---|---|
| Construction, 333M samples (1,024 proposals per sub-cell), after moving accumulation to device float64 | 11.5 s | 17% before that change |
| Physical 4,096-spp render | 70 s | 68% |
| Readout of 65 tables | 41 s | 36% |

The evaluation stages therefore run as two concurrent processes (≈ 5 GB each).

**Gate-instrument fixes before freezing.** These were found on the development run. No evidence seed was used.

1. **A9.** "LOW max abs 0" failed at ~1e-6, caused by float rounding between different kernel compositions. Changed to relative L1 ≤ 1e-5.
2. **A6.** Origins in the resting-box/floor junction float band were counted together with the F6 class. The two are now separate classes.
3. **A7.** "Every contaminated cell flagged" demanded detection of immaterial contamination. The legacy SE also used the v3N variance as a proxy, which produced heavy-tail false positives. The legacy sampler now stores its own second moment, and the predicate is "no materially contaminated cell passes as clean", with family-wise z 5.2 and 7e-4.
4. **Stage E.** Requiring the stale control to be worse on every resolved transition would have turned a test-sensitivity limit into a candidate failure. It is replaced by the discriminating-transition rule.

**Disclosure.** The development run produced noisy readouts of the Stage B table arms: 2 replicas × 32 proposals, 64 spp. At V0 they read U +0.073, G +0.043, S_G +0.012, the same pattern as WL101H. Learned arms trained for 200 steps on 6 geometries produced meaningless values. Fixes 1–3 were made before the arm readouts were inspected. Fix 4 and the A7 second-moment change came after. Neither touches a band, threshold, arm or outcome rule. No decision rule was changed after seeing development values.

## Execution

(appended below)

### First frozen pass (chain from commit `74b647c`, 2026-11-24 18:06–18:47 KST)

Tests T1–T6 passed. T2 reproduces the WL101H X checks: under C, shadow attenuation agrees with the analytic chord law to within 0.998–1.000 from 1 mm to 80 mm; under N it is overstated 2,190-fold inside the 11.5 mm ring.

Stage A ran 8 C, 4 N, 2 legacy, 2 frame-control and 1 synthetic construction, about 2.5 min in total. Stage B rendered the six F0 views (2 × 4,096 spp, physical C and N passes, 65-table readout) in 38 min on two processes.

**Frozen verdict: NOT QUALIFIED.** A1–A4, A6, A7 and A9 pass, and A8 is qualified for every view and region. A5 fails only on its footprint sub-test: replica 1 has a per-replica binomial z of 5.16 against a limit of 5. The other replicas give 2.1–4.8. The chain therefore stopped before Stage C by its frozen rule. This verdict is preserved.

### Amendment 1 — RNG-correlated proposals make the per-replica binomial footprint test invalid (`protocol/amendment1.json`)

The diagnostic is post hoc and used only Stage A sums and reference images. No Stage B arm metric was opened.

- **Replay.** A geometry-only replay of the frozen proposal stream reproduces every stored buried count exactly. The analytic footprint and the exclusion logic are therefore correct.
- **No bias.** The per-cell excess averaged over the 8 replicas is centred: mean z per cell lies between −1.1 and +1.25, with no fixed-sign cells.
- **Cause.** The shared WL86/WL100 Dr.Jit PCG32 lane seeding (initstate per launch, initseq = lane index) correlates proposals. Adjacent lanes show u1 correlation −0.079 at lag 2, and the same slot shows about −0.004 across the 1,024 batch repeats.
- **Effect.** On partially covered sub-cells, between-replica variance is 3.05× the binomial variance. The same proposal formula with decorrelated lane seeding gives 1.14×.
- **Same effect in camera pixels.** Reference block-mean variance is 1.03–1.30× the iid value on the floor of the six views.

Changes, under the brief's instrumentation-fix clause:

1. The A5 footprint test now uses the 8-replica pooled excess with between-replica dispersion: |t| ≤ t₇ at family-wise 1%. Per-replica z values are kept as description.
2. The endpoint stream SE in Stages B, D and E is multiplied by √κ, where κ is the measured pixel block-mean variance ratio of each view. This is conservative and only widens intervals.

Samples, seeds, tables, streams, thresholds, bands, arms and outcome rules are unchanged, and nothing is re-rendered.

**Successor note.** This RNG property affects every WL86–WL112 estimator. It inflates per-sample variance but does not bias means. Statistics across independent seeds remain valid.

### Second pass under Amendment 1 (chain from commit `890a135`, 2026-11-24 18:52–18:54 KST)

The second pass re-ran the analysis only; no samples or streams were regenerated.

- **Stage A: QUALIFIED.** All of A1–A9 pass.
- **Stage B outcome: NONE.** Stage C was not run, and therefore neither were Stages D/E or the learned cost accounting. The frozen gate classification is **NOT_RESOLVED**.

## Final report

Written for the research owner.

### Implementation fact

Code is under `experiments/ecro_sss112/`, raw data under `results/ecro_sss112/primary/`, reviewer artifacts under `evaluation/112/`.

- **Tracer.** `tracer112.py` is a copy of the WL100 physics with a movable sphere and C/N origin semantics. It adds an analytic pinned first surface, 2×2 sub-patch addresses, event-2 query capture and multi-table readout.
- **Construction contract v3.** `stageA112.py` and `analyzeAB112.py`.
- **Evaluation streams.** `eval112.py`.
- **Not run.** The candidate ECRO-S2 (`model112.py`), the D112 data and training code (`data112.py`, `train112.py`) and the Stage D/E/F code exist and passed a reduced development run. None was run at evidence scale because the frozen Stage B rule stopped the batch.
- **Unchanged.** ECRO v0, ER/SDM, LOW/HIGH ownership, the material, every historical checkpoint and reference, and every earlier worklog.

### Measurement

**Physical semantics qualification.** Stage A is QUALIFIED under Amendment 1. As frozen, it was NOT QUALIFIED; that pass is preserved in `results/ecro_sss112/primary/frozen_pass1/`.

| Gate | Result |
|---|---|
| A1 tracer identity | lanes max abs 0 |
| A2 pinned identity | relative L1 2.1e-7 on 8,065 paired probes |
| A3 frame invariance | max \|z\| 1.02 |
| A4 synthetic projection | 80/80 entries within 4 SE; nested parent 7e-8 |
| A5 support | pooled footprint \|t\| ≤ 3.03 vs critical 5.73; singular counts 9–12 per replica vs 8.93 expected; address mismatch 2.0–2.3e-6 |
| A6 origins | C: zero sphere-interior origins in constructions and references; junction-band rate 1.3e-8 per camera path. N: 41,274 / 1,454,382 sphere-interior origins (F6 detected) |
| A7 positive control | legacy WL100 sampler has a 1.08% address/count inconsistency rate (rejected); 0 of 256 materially different cells pass as clean |
| A8 materiality | numerical-event bounds at most 0.065 (near) and 0.085 (contrast) of their 10%-of-band thresholds; region relative bounds ≤ 8.3e-5 |
| A9 LOW pairing | relative L1 ≤ 1e-10 |

**Semantics factor.** C minus N reference shift on near-floor HIGH is −0.00017 to −0.00026 luminance, at most −0.0006 in the 0–10 cm band. Tables built under N give the same endpoints as under C within their intervals: S_G near at V0 is +0.0106 under N vs +0.0104 under C. The F6 correction is physically required at the tangency but immaterial at every image endpoint, confirming WL101H.

**Projection measure and spatial resolution.** Mean HIGH residual, linear HDR luminance, eight table replicas × two 4,096-spp streams. Half-widths are ≤ 0.0024. Bands: near 0.014–0.038, contrast 0.013–0.034.

| Arm | V0 near / contrast | H1 | H2 | H3 | H4 |
|---|---|---|---|---|---|
| U (1270×64) | +0.072 / +0.068 | +0.078 / +0.075 | +0.076 / +0.074 | +0.216 / +0.229 | +0.205 / +0.222 |
| G | +0.043 / +0.052 | +0.047 / +0.057 | +0.045 / +0.055 | +0.139 / +0.167 | +0.134 / +0.165 |
| S_U (2×2) | +0.022 / +0.018 | +0.023 / +0.020 | +0.021 / +0.018 | +0.071 / +0.077 | +0.069 / +0.077 |
| **S_G (2×2)** | **+0.010 / +0.014** | +0.011 / +0.015 | +0.009 / +0.012 | +0.032 / +0.045 | +0.033 / +0.048 |
| original ECRO (WL100) | +0.072 / +0.071 | +0.076 / +0.075 | +0.076 / +0.078 | +0.217 / +0.232 | +0.207 / +0.227 |
| camera-fitted P_v (diag.) | +0.052 / +0.057 | +0.057 / +0.066 | +0.059 / +0.063 | +0.131 / +0.148 | +0.169 / +0.190 |
| camera-fitted S_v (diag.) | +0.020 / +0.021 | +0.024 / +0.027 | +0.024 / +0.026 | +0.091 / +0.098 | +0.101 / +0.110 |

- **S_G.** Near is inside the band at all five contact-valid views. Contrast is inside only at H2. The cause is a negative flank residual under G: −0.003 to −0.015, against U's +0.004 at the front views.
- **Other arms.** V equals U; the receiver restriction alone changes nothing. No arm regresses off-target: every relative RMSE in silhouette, sphere, contact/shadow, floor and full frame is ≤ U. Coverage is 1.000.
- **Near-region stable relative HIGH RMSE.** U 32–38%, S_U 10–15%, S_G 5–9%.
- **Comparison with WL101.** These are new local values on a different stack, so they are not paired with WL101. U at V0 (+0.0722) matches WL101E's +0.0702 in pattern. G (+0.043) and S_G (+0.010) reproduce WL101H's descriptive +0.0415 and +0.0106.

**Factorial, at V0 near.** Measure at fixed resolution: U→G −0.029; S_U→S_G −0.012. Resolution at fixed measure: U→S_U −0.050; G→S_G −0.032. Interaction +0.017.

**Post-hoc diagnostic (`diag112.py`, commit `07b45a0`).** Descriptive only; it changes no verdict.

- **(a) Where near queries land.** 79–88% of the near-floor HIGH query weight lands on the eight lowest sphere patches (ring 0). Flank weight splits about 45% sphere ring 0 and 30% walls. All projection and resolution effects act through ring-0 cells. At V0 the ring-0 contribution is −0.031 for G→S_G and +0.007 at the flank.
- **(b) One further level, 4×4 sub-patches.** Four replicas, read on the same stream seeds (LOW pairing 2e-8).

| Arm | V0 | H1 | H2 | H3 | H4 |
|---|---|---|---|---|---|
| S4_U near / contrast | +0.004 / −0.000 | +0.003 / −0.001 | +0.003 / −0.002 | −0.006 / −0.010 | −0.006 / −0.008 |
| S4_G near / contrast | −0.003 / −0.002 | −0.005 / −0.004 | −0.006 / −0.005 | −0.035 / −0.032 | −0.031 / −0.028 |

  - S4_U: both endpoints are inside the bands at every view, and near RMSE is 2.3–2.9%.
  - S4_G: over-corrects at the lit-side views; it is barely inside there.

**Learned generator adequacy, static, cross-view and dynamic fidelity, reuse correctness, collateral regression of a learned candidate.** **Not measured.** No candidate was trained, because the frozen Stage B rule stopped the batch.

**Runtime and scalability cost.**

| Representation | Floats | Size (float32) |
|---|---|---|
| Coarse table | 243,840 | 0.93 MiB |
| 2×2 | — | 3.72 MiB |
| 4×4 | — | 14.9 MiB |

- **Physical construction.** About 5.8 s per 333M-sample 2×2 replica and about 21 s per 1.3e9-sample 4×4 replica on the local GPU.
- **Development run only.** ECRO-S2 generator 4.8 ms vs 2.5 ms for ECRO v0. Parameters 736,720 vs 735,664. Training about 0.023 s per step. Generator latency does not depend on the weights.
- **Not measured.** No learned edit-to-frame latency was measured for a candidate.

### Observation

The S_G residual maps keep a narrow positive band directly beneath the contact and a negative region on the lit-side flank (H3/H4). Every arm shows silhouette stripes at sphere patch boundaries, the same in camera-fitted arms and in the original ECRO. They do not change with the projection measure.

### Interpretation

**Supported cause of the artifact.** The near-floor overbrightness is a representation/projection error. It is not caused by generator learning, LOW, the semantics change, sampling or reconstruction. The near floor reads the higher-order radiance of the eight lowest sphere patches through their whole-patch mean. Radiance varies strongly within those patches near the contact. The uniform measure averages brighter parts of the patch than the near floor actually sees.

**Projection measure.** Projection alone (G) removes 41% of the near excess at V0 but cannot reach the bands at any view. Its cos weighting also creates a negative flank bias that grows with resolution.

**Spatial representation adequacy.**
- At the predeclared 2×2 resolution, no camera-independent measure meets both endpoints at every contact-valid view. Under the frozen rule, adequacy is therefore not demonstrated.
- One more level of resolution under the uniform measure (S4_U) meets both endpoints at all five views, with no off-target regression and full coverage.
- That result is post hoc, comes from one fixture (F0) and one level, and is not a frozen decision.
- The required resolution is driven by the near-contact patches of the sphere's lowest ring, which carry 79–88% of the near query weight. It is not a global requirement. Whether this depends on gap, light side or sphere position was not measured, because Stage E did not run.

**Learned candidate status.** Not trained. The 2×2 ECRO-S2 candidate was frozen but not admitted by the Stage B rule.

**Measurement note.** The shared PCG32 lane seeding of every WL86–WL112 estimator correlates samples. Within-replica binomial or iid-pixel standard errors are therefore optimistic: about 3× variance for proposal counts, 1.03–1.30× for pixels. Statistics across independent seeds remain valid.

### Unresolved questions

1. Would a camera-independent 4×4 representation, under the uniform measure or one restricted to the lowest sphere ring, meet the contract on an independently frozen set of views, configurations and dynamic sequences? Would a learned decoder reproduce it?
2. Why does G's cos weighting bias the flank negative at fine resolution? Is the uniform-area Lambertian receiver proxy wrong near a contact, where receivers are dominated by the nearby floor?
3. What causes the sphere-silhouette patch-boundary stripes common to every arm?
4. Static and dynamic learned evidence, reuse, collateral regression and learned cost all remain unmeasured.

### Architecture verdict

**Mandatory SSS Gate: NOT_RESOLVED.** The physics is qualified under Amendment 1. Under the frozen rule, no camera-independent representation at the predeclared resolutions (1270×64, 2×2) is sufficient, so no learned candidate was trained. This is a completed negative stage and does not satisfy the gate.

The blocking mechanism is the spatial resolution of the higher-order outgoing radiance on the sphere's lowest-ring patches at the contact. 2×2 is insufficient. 4×4 is sufficient post hoc on F0 under the uniform measure.

Moving to a 4×4 (or ring-restricted adaptive) output representation would be a new representation hypothesis. Under the brief that is a **decision for the research owner**. No candidate is adopted, and Architecture.md's accepted contract does not change.

### Reproducibility

- **Commits.** Freeze `74b647c`. Figure script `30163d4`. Amendment 1 `890a135`. Post-hoc diagnostic `07b45a0`. The report is the next commit.
- **Protocol and manifests.** `experiments/ecro_sss112/protocol/{sss112.json, amendment1.json, freeze_manifest.json}`. The WL100 checkpoint SHA-256 `2d9cb580…` is verified.
- **Environment.** Local RTX 5080 16 GB; `external/8dna26/.venv-cu128` with Python 3.11, Mitsuba 3.5.1, Dr.Jit 0.4.6, torch 2.8.0+cu128.
- **Seeds.**

| Use | Seeds |
|---|---|
| C constructions | 11201001–8 |
| N constructions | 11202001–4 |
| Legacy | 11203001–2 |
| Frame control | 11204001 |
| Synthetic | 11205001 |
| Identity checks | 11206000/1 |
| Streams | 11210000 + 10·view + j |
| P_v | 11211000 + 10·view + r |
| Diagnostic 4×4 | 11208001–4 |
| Development | offset +5,000,000 / +7,000,000 |

- **Run.** `bash experiments/ecro_sss112/chain112.sh`, detached and resumable. Then `python experiments/ecro_sss112/diag112.py results/ecro_sss112/primary` and `python experiments/ecro_sss112/fig112.py results/ecro_sss112/primary evaluation/112`.
- **Logs.** `results/ecro_sss112/primary/logs/`. The development runs are `results/ecro_sss112/{dev,devrun,probe}`.
- **Reviewer index.** [evaluation/112/index.html](../../evaluation/112/index.html) and `manifest112.json`, which holds the SHA-256 of every figure and JSON.
