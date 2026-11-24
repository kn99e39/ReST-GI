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
