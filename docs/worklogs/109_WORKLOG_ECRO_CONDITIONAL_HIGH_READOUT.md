# Worklog — Frozen ECRO Conditional Transport Readout: Feasibility and Cost-Normalized Variance Reduction (2026-11-22)

WL109, a regular numbered batch designated by the owner's brief, runs on the existing `Denoise_Reconstructor` branch from
base `0e3d9c1` (WL108 closure). No branch or worktree was created and `ecro_lr3_adapter` was not touched. The batch asks one
bounded architecture-feasibility question:

> Does the frozen ECRO higher-order transport table enable a conditional-readout estimator that achieves a reproducible
> reduction in TOTAL sampling MSE at equal GPU cost, while preserving the renderer's expected output and dynamic-update
> correctness?

The candidate changes only the estimation strategy for the frozen HIGH contribution. The ECRO checkpoint, the HIGH table
bytes and their generation, the LOW estimator, the source encoding, the patch/bin discretization, event ownership, geometry,
materials, lights, cameras and the sampling distributions are frozen. Nothing is trained, refitted, smoothed or filtered.
The target is the expected output of the frozen ECRO renderer, not the physical path-traced image.

Code: [experiments/ecro_estimator109](../../experiments/ecro_estimator109/).
Protocol: [protocol109.json](../../experiments/ecro_estimator109/protocol109.json).
Raw output: `LabServer69:~/rest_sss101a/results/ecro_estimator109/`.
Reviewer artifacts: [evaluation/109](../../evaluation/109/).

## Stage A — estimator contract, read from the code

The frame program is `rt101.RT._frame_prog`, which calls `pt100.Ctx.trace` in mode `ecro` with `kmax = 2`. One PCG32 lane
stream per sample (`rt101.pcg_046`) feeds three stages in a fixed order.

| Variable | Draws | What it fixes | Feeds |
|---|---|---|---|
| J | 2 uniforms in `PT.gen_rays` | camera ray, first hit `si1`, `wi1`, material, closure, SSS flag | E1, E2, HIGH |
| P | k = 1 SSS entry-to-first-exit walk (sphere first hits only; variable count, NEE draws none) and the k = 1 closure sample (3 uniforms, always drawn) | second hit, its table address a = (patch, bin of −d), throughput T | E2, HIGH |
| S | k = 2 SSS walk (sphere second hits only) | the k = 2 walk NEE | E2 only |

Per sample, with X = (J, P, S):

- E1 = NEE at the first hit plus the k = 1 walk NEE (P-dependent only on sphere first hits);
- E2 = the ballistic through-sphere term generated at the first hit plus NEE and walk NEE at the second hit;
- H(X) = T(J, P) · B[a(J, P)], read at the second hit before the k = 2 walk;
- TOTAL = E1 + E2 + H; LOW = E1 + E2.

The Monte Carlo measures are the original ones: J uniform over the pixel footprint; P the walk's exponential free-flight
and isotropic phase sampling followed by the closure's lobe-mixture sampling with weight f·cos / pdf; S the k = 2 walk. A
table entry B[p, b] is the learned outgoing higher-order radiance of patch p toward direction bin b. It is generated from
fixed state-ray and source offsets and is not an integral over camera paths. HIGH depends on J and P only, and LOW shares J
and P with HIGH, which is the source of the positive Cov(LOW, HIGH) WL103 measured (all of it through C(E2, HIGH)).

## Stage B — the single mechanism C1

**Retained state.** U = the first camera interaction (camera ray, first hit, `wi1`, closure, SSS flag, active mask), a
function of J.

**Integrated variable.** The first-event continuation P. Its conditional distribution given U is sampled by exactly the
original statements: the SSS walk with its NEE block removed (NEE consumes no random number, so the same stream gives the
same collision sequence, exit point, exit direction and exit throughput), followed by the closure sample.

**Estimator.** H_C1(U) = (1/M) Σ_m T(U, P'_m) B[a(U, P'_m)], with P'_1..P'_M drawn from an independent TEA-seeded PCG32
stream. LOW keeps the main stream untouched, so for a given seed C1 and R0 produce bitwise-identical LOW and differ only in
HIGH.

**Validity.** Given U, the P'_m have the same distribution as P and are independent of (P, S). Hence
E[H_C1 | U] = E[H | U] and E[TOTAL_C1] = E[TOTAL_R0]. H_C1 is a finite unbiased inner Monte Carlo estimate of the
conditional expectation. It is not exact integration. Exact finite integration was rejected because the visible (patch,
bin) partition of the hemisphere at a continuous first-hit position depends on visibility. Throughput, the closure PDF,
visibility, the front-face test and event ownership are those of the original k = 1 → k = 2 path.

**Variance algebra.** At n spp per pixel, Var(H̄_C1) = [Var(E[H|J]) + E[Var(H|J)] / M] / n and
Cov(L̄, H̄_C1) = Cov(E[L|J], E[H|J]) / n. The shared-sample covariance E[Cov(L, H | J)] is removed. When that conditional
covariance is negative, C1 increases TOTAL variance; the synthetic contract includes such a case. The pairing identity
Cov(H̄_R0, H̄_C1) = Var(E[H|J]) / n gives an exact J/P attribution of HIGH variance on the real workload without the
WL103 split-stream wiring.

**Fixed a priori.** Inner samples exclude the shared sample, because WL103 found a positive LOW–HIGH covariance in every
ROI. Inner samples are i.i.d.; no stratification. The closure code is the frozen one; a Lambert-specific fast sampler was
rejected because the frozen R0 cannot use it, which would make the cost comparison unequal. M ∈ {1, 2, 4} is chosen by a
predeclared rule on development cameras.

## Preregistration (frozen before any evidence-seed output)

### Views, references and environment

- Evaluation: V0 and H1–H5. WL105–WL108 inspected these views, so no out-of-distribution claim is made.
- Development: WL105 training cameras T00–T03. They are used for the M rule, the futility screen and the synthetic
  statistics only.
- Frozen-ECRO reference: the WL107 qualified assessment reference, 46 WL107 plus 2 WL105 independent 8192-spp blocks per
  view (393,216 spp). Per-pixel variance is recomputed from the authenticated blocks. The reference manifest hash
  (`5fc7ee75…`) and capture-manifest hash (`164ba353…`) are pinned.
- Host: LabServer69, RTX 3090, driver 550.54.15, Mitsuba 3.8.0, Dr.Jit 1.3.1, Torch 2.3.1, FREEZE = 1. Admission reuses
  `renderer108.frozen_runtime` unchanged. It checks sources, checkpoint, scene, learned-table hash and the archived
  4096-spp V0 replay. WL109 adds a replay of the archived WL105 evaluation frames at 4/16/32 spp.

### Arms

| Arm | Definition |
|---|---|
| R0 | Original frame program |
| R0 grid | Original program at 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 24, 28, 32, 40, 48 and 64 spp with independent seeds. Gives the measured quality-versus-time curve; it also provides R0-SPP at C1's measured cost |
| C1 | The mechanism above with M = M* |
| C1-H | The HIGH channel of C1 (diagnostic, not a candidate) |

32 replicate frames per arm, view and spp. Paired R0/C1 frames share the main seed.

### Synthetic contract (Stage C), gates in the protocol

1. **Exact discrete model** (CPU). Eight cases: constant, piecewise-constant, directional, visibility discontinuity,
   1000× contrast, zero HIGH, positive covariance, negative covariance. The exact expectation and the variance for
   M = 1, 2, 4 are compared against the simulated estimator. In the negative case the exact and empirical C1 variance must
   exceed R0's.
2. **Structural, on the real geometry.** Learned table and seven synthetic tables (zero, constant, piecewise, directional,
   contrast, positive and negative covariance), four development views, 1 and 4 spp. Checks:
   - re-implemented R0 = original frame;
   - C1 LOW = R0 LOW;
   - replay mode (inner stream = copy of the main stream) reproduces R0 HIGH, which checks address, throughput and ownership;
   - zero table gives HIGH ≡ 0;
   - determinism;
   - frozen program = eager.
3. **Replicate independence** under the protocol seeding, with the historical seeding as a power control (see the
   development record).
4. **Statistical**, at 4 spp with 64 replicates on T00, regions all / sphere interior / silhouette / box visibility edge /
   contact / control:
   - paired expectation t-test;
   - spatial bias power Q;
   - M = 1 variance equivalence;
   - M = 2 variance theory;
   - known negative and positive covariance cases.
5. **Fixed-jitter independence.** With pixel-centre camera rays, C1 HIGH must be uncorrelated with LOW and with the
   original HIGH. The LOW–HIGH correlation serves as a power control.
6. **Dense quadrature.** At fixed pixel-centre U on diffuse first hits, the 4096-sample inner-MC mean is compared with an
   independent 256² cosine-grid quadrature that uses closure `eval`, not the sampler.

Any failure stops the batch. Ordinary implementation bugs that keep the declared estimator may be fixed and recorded.

### Decision rules (frozen)

- **P2:** primary-ROI 4-spp equal-time GM MSE ratio C1 / R0(t_C1) ≤ 0.85.
- **P3:** its 95% upper bound < 1.0. Two-level bootstrap over views, then replicates and timing samples; normal interval
  on log GM.
- **P4:** bias.
  - B1: paired relative signed-luminance bias, pooled over 4/16/32 spp, with its 95% CI within ±0.5% for every view ×
    {primary, sss_interior, silhouette, material_boundary, control}.
  - B2: spatial bias power Q, computed from 8 independent replicate groups, upper95 ≤ 0.01 (silhouette 0.02).
  - Precision escalation from 32 to 64 paired replicates is allowed once and uniformly, only for an UNRESOLVED B1.
- **P5:** primary at 16 and 32 spp: GM ratio ≤ 1.02 and upper95 ≤ 1.05.
- **P6:** silhouette and SSS interior at 4/16/32 spp: GM ratio ≤ 1.05 and upper95 ≤ 1.10.
- **Reference precision:** reference RMS noise ≤ 10% of the smaller arm RMSE in every decision ROI.
- **P7 (dynamic):** run only after P1–P6 pass. One frozen geometry edit (config 0 → 2, light 0, V0) through the standard
  lifecycle. Checks:
  - D1: table identity, i.e. selective = full rebuild = fresh construction;
  - D2: replay, LOW and frozen = eager;
  - D3: no persistent C1 state;
  - D4: paired bias;
  - D5: equal-time ratio ≤ 0.85 with upper95 < 1.0, against a fresh 131,072-spp reference and a fresh R0 curve;
  - D6: costs.
- **Verdict mapping:** as in the protocol. SUPPORTED needs P1–P7. STATIC_ONLY_SUPPORTED applies when P1–P6 pass and
  either D5 fails with correctness intact or the dynamic stage cannot run.
- **Report only:** 1 spp, equal-spp ratios, variance components and the J attribution, boundary-distance bands, the
  contact region, signed bias against the reference, relative HDR RMSE, time to equal quality, first-call and memory costs.

## Development record before the freeze (code-path smoke, non-evidence seeds)

Development used the smoke seed family (+50,000,000) and reduced sizes. Development statistics were inspected where
recorded below. No evaluation-view output existed before the freeze.

- **Structural identity.** On T00 at 1 and 4 spp: re-implemented R0 = original, C1 LOW = R0 LOW, replay HIGH = R0 HIGH. All
  were bitwise for every table in the smoke structural pass.
- **Cost (T00, frozen, uncontended).** R0 4 spp: 4.40 ms. C1: 6.52 ms (M = 1) and 8.18 ms (M = 2). R0: 5.40 / 6.26 /
  8.13 ms at 5 / 6 / 8 spp. One inner continuation costs about half a full sample, because it repeats the closure sampling
  and, on the sphere, the walk.
- **Seeding finding.** The historical seed · 1000003 convention produces correlated `pcg_046` replicate streams for
  consecutive seeds. All replicates share the lane increments and differ only by structured initial states. In a 2×2
  diagnostic (main stream consecutive or spaced × inner stream consecutive or spaced), the excess consecutive-pair
  correlation of the original HIGH was 0.0124 ± 0.0009 on the contrast table and 0.0053 ± 0.0007 on the learned table with
  consecutive main seeds. With spaced or splitmix64-hashed main seeds it was within 1 SE of 0. The TEA inner stream showed
  none. Evidence frames therefore pass splitmix64 seed values to the unchanged frame program; archived replays keep the
  historical convention. The renderer and its distributions are unchanged. The finding bears on how independent historical
  consecutive-seed replicates were, for example WL103's replicate variances and its G7 failure. Their means remain unbiased.
  It is reported here as a development observation, not re-litigated.
- **Statistical-test corrections.** These are design fixes made before any evidence; the estimator was not changed.
  - A frame bootstrap misstates the noise of Q, because shared resampling weights do not average over pixels. Q's SE now
    comes from independent replicate groups.
  - The variance-theory tests fail only on deviations that are both material and resolved (> 4 SE), to control
    multiplicity.
  - The quadrature z² limit is set relative to the Student expectation ν / (ν − 2).
  - The discrete model's negative case was rebuilt with constant throughput. The first version shared T between E2 and H,
    which made it positive.
- **Futility threshold.** The smoke development on T00 gave TOTAL variance ratios of 0.765 / 0.715 / 0.689 for
  M = 1 / 2 / 4. This implies a screening ratio of about 1.1 at M = 1. The threshold was therefore raised from 1.0 to 1.5
  before the freeze. The evaluation stage costs under two GPU-hours, so the gate now closes the batch only where no
  plausible failure of the 1/time screening assumption could yield a benefit. The change only adds evidence; no decision
  threshold moved.
- **Code-path coverage.** The smoke chain ran every stage end to end, including the bias escalation and the dynamic stage.
  - Smoke data: 4 replicates, contended timing; its statistics carry no evidential weight.
  - Dynamic smoke: D1 tables bitwise equal (selective = full rebuild = fresh construction); D2 identities bitwise after the
    edit.
  - Bug fixed: constructing a second `rt101.RT` overwrites rt101's module-global resident closure table, which broke frozen
    recording on the original runtime. The dynamic stage now restores it.
  - Also fixed before the freeze: relative output paths in the chain script, NaN-safe JSON writes, degenerate bootstrap
    draws, and the `malloc_watermark` call signature.
- **GPU sharing.** Another project's jobs used the GPU intermittently. Renderer outputs are deterministic in the seed, so
  data stages run without an exclusive GPU. Every timing block waits for an exclusive GPU and is repeated if a foreign
  process appears during it.

## Freeze and execution record

**Freeze.** Commit `86e25cd7667a14326643f00ddc5ea4c2a33175d5`. Protocol SHA-256 `08725b76670c3a41ab93daf9aa2079767a7c3550080a9dfe70960b41fcd7d40b`.
Before launch, the deployed LabServer69 files were checked against the committed blob hashes (`COMMIT_SHA256`,
`DEPLOYED_COMMIT`). The chain was started with `run109.sh evidence 86e25cd…` at 19:14 KST.

**Stage C: synthetic contract, PASS (423 s).** Every gate passed.

- Exact discrete model: all eight cases. In the negative case C1's variance exceeds R0's, exactly and empirically.
- Structural: eight tables, four views, 1 and 4 spp.
- Replicate independence under the protocol seeding: |z| < 1 for LOW, original HIGH, C1 HIGH and their difference. The
  historical seed · 1000003 convention resolved the original-HIGH correlation at z = 25.7 on the contrast table and
  z = 8.0 on the learned table, which confirms the development finding inside the frozen evidence.
- Statistical: seven tables, six regions.
- Fixed-jitter independence, with its power control.
- Dense quadrature: four tables. Violation fraction 0.06–0.15%, |mean z| ≤ 0.04, mean z² 1.12–1.20 against the
  Student expectation 1.154, relative mean gap ≤ 0.018%.

Raw: `results/ecro_estimator109/evidence/synth/synth.json`.

**Development stage: STOPPED at its elapsed-time cap (exit 124, 20:21 KST).** The variance part finished for T00. Its
primary TOTAL-variance ratios C1/R0 were 0.765 / 0.715 / 0.691 for M = 1 / 2 / 4 (bootstrap SE ≤ 0.0005). T00's timing
block then waited for an exclusive GPU. Another project's chain held the GPU for the remaining 54 minutes (TransGI b14,
running since 18:50 KST), so the 3600 s stage cap expired. No M was selected and no evaluation-view frame exists.

This is an execution-resource stop, not a scientific failure. Resuming requires an owner-approved amendment: the
exclusive-GPU wait has to be kept out of the stage cap, or the GPU has to be scheduled. The stopped run's logs are kept
under `results/ecro_estimator109/evidence/`.

## Amendment 1: resume after the foreign GPU chain (owner-approved, 2026-11-22)

The owner directed that a watcher be placed on the TransGI chain and that WL109 resume when that chain ends. The execution
correction:

- [`run109_amend1.sh`](../../experiments/ecro_estimator109/run109_amend1.sh) waits for the watched `run_b14.sh` process to
  exit and then for 120 s of uncontended GPU.
- It moves the stopped development output to `dev_stopped_cap/` and reruns the development stage and every later stage
  with the frozen code, seeds, sizes and thresholds. The frozen synthetic result is reused; it comes from the same commit
  and passed.
- The elapsed-time caps of the GPU stages (development, capture, escalation, dynamic) become 24 h, so that exclusive-GPU
  waiting cannot stop the chain. Analysis caps are unchanged.

No estimator, gate, seed or decision rule changes. The T00 variance values from the stopped attempt are superseded by the
rerun, which uses the same seeds and is therefore bitwise reproducible.

## Amendment 1 execution record: operator stop at capture, foreign OOM contribution (2026-11-23)

**What ran.** The watched TransGI b14 chain ended at 20:47 KST. After 120 s of free GPU the chain started. The development
stage selected M\* = 1 (predicted GM equal-time ratio 0.882; not futile). The capture stage then finished every data part
by 20:59:54:

- Stage 0 archived replay;
- paired R0/C1 frames, 6 views × 4 nominal spp × 32 replicates;
- the R0 grid, 18 levels × 32 replicates.

`run/manifest.json` (written at 20:59) lists 4,992 raw files with hashes and 768 paired-identity records. Raw output is
40 GB. Stage 3 timing then waited for an exclusive GPU, because TransGI b15, b17 and a serial rerun queue kept starting.

**Execution defect (ours).** While it waited, the `run109.py` process kept its CUDA context and the Dr.Jit/Torch memory
caches: 1.74–1.78 GiB of device memory on a 24 GiB RTX 3090. Two TransGI chains (b16 eval and b17) then ran
concurrently, and several of their evaluations died with CUDA OOM. Each OOM message lists our process holding 1.74 GiB.
Without that memory, at least two of the failed allocations would have fit:

| Evaluation | Requested | Free at failure | Free without WL109 |
|---|---:|---:|---:|
| bathroom b17_spec256 | 2.32 GiB | 1.44 GiB | 3.18 GiB |
| living_room_2 b17_spec256 | 2.32 GiB | 1.67 GiB | 3.41 GiB |

The primary cause was the two foreign chains running side by side. Our idle memory hold was still a contributing cause,
and it contradicts the owner's instruction that WL109 must wait without being a cause of foreign OOM. Two design errors
produced it:

1. Amendment 1 watched only the b14 PID, not the foreign chain as a whole.
2. The in-process timing wait (`exclusive_block`) keeps GPU memory allocated while it waits.

**Action.** At 01:50 KST the operator terminated `run109.py` by PID. `chain.state` records `STOP stage=capture exit=143`.
No captured file was modified. The on-disk manifest covers all captured data, but `complete=false` and there is no timing
section. No timing, analysis or verdict exists.

Resuming requires a further owner-approved amendment. It should run Stage 3 alone in a fresh process that is started only
once no foreign GPU work is running or queued, and that frees all device memory whenever it waits.

## Amendment 2: guarded resume at Stage 3 without holding GPU memory (owner-approved, 2026-11-23)

The owner approved resuming the run under a rule: WL109 must hold no GPU memory while foreign work runs or is queued.
This is an execution correction only. The estimator, seeds, replicate counts, timing schedule, thresholds and decision
rules are unchanged, and the captured data and their manifest are reused as they are.

**[`guard109.py`](../../experiments/ecro_estimator109/guard109.py).** Every GPU-touching WL109 command now runs under this
guard: per-view timing, analysis (it uses CUDA tensors), escalation and dynamic. The guard imports no GPU library.

- It starts its command only when the user has no tmux session and nvidia-smi has listed no compute process for 120 s.
  TransGI and the other foreign chains run in tmux; WL109 is launched with `nohup`/`setsid`.
- While the command runs, the guard polls nvidia-smi every 2 s. When a compute process outside the command's process
  tree appears, it terminates the command's process group: SIGTERM, then SIGKILL after 20 s. This releases all WL109
  device memory.
- It then waits for the start condition again and restarts the command from the beginning. Outputs depend only on the
  seeds, so a restart reproduces the same data. Escalation rewrites the same files. Dynamic's partial output directory
  is moved aside before a new attempt.
- Each attempt is capped: 7200 s for timing and analysis, 86400 s for escalation and dynamic.
- Every wait, start, preemption and exit is logged to `guard109.jsonl`.

The remaining window is the ≤ 2 s polling interval between a foreign process appearing and WL109 being killed.

**[`timing109.py`](../../experiments/ecro_estimator109/timing109.py).** Stage 3 runs one evaluation view per fresh
process, after that process's own admission. The block content is the same as `run109.py` Stage 3:

- the same slots and `seed_timing` with the view's `EVAL_VIEWS` index;
- 3 warm-up and 30 rounds in the frozen randomized order;
- the end-to-end composition, first-call and watermark measurements;
- the `exclusive_block` contamination rule.

Each view is written atomically to `run/timing_parts/<view>.json`; views already written are kept. A CPU-only `--merge`
does the rest:

- copies the capture manifest to `run/manifest_capture.json`;
- adds `timing`, the per-view admissions and versions, and `table_sha_after`, asserting that every part's table bytes
  equal the capture admission's;
- records the hashes of the parts and the code under `amendment2`, and sets `complete`.

**[`run109_amend2.sh`](../../experiments/ecro_estimator109/run109_amend2.sh).** The driver checks the synthetic PASS
and the non-futile development result, then runs timing for each view, the merge and the analysis. After that it
follows the frozen escalation and dynamic branches of `run109_amend1.sh`, each under the guard.

**Disclosed differences.**

- First-call (recording + compile) times are now measured in a fresh process after admission, not after the capture
  stages in the same process. They are reported only and enter no gate.
- Frame and end-to-end medians are taken after warm-up, as before.

**Validation before launch.** Local tests with stubbed `nvidia-smi`/`tmux`:

- a normal exit passes the child's code through;
- tmux blocks the start;
- a foreign PID preempts the child and attempt 2 completes;
- the attempt timeout returns 124 and leaves no orphan.

A CPU-only merge test on LabServer69 on synthetic parts checked the merged fields and refused to merge into a complete
manifest.

## Final report (2026-11-23)

### Implementation Fact

**Amendment 2 execution.** The chain ran `run109_amend2.sh` at commit `2752ff1`, with deployed files checked against
`COMMIT_SHA256`.

- The guard waited 01:57–04:16 KST for tmux sessions `rerun`, `b18`, `hole`, `fbv` and then for GPU processes. It
  waited again 04:18–06:15 for `b19`.
- Every guarded command finished on attempt 1: six timing views of 67–114 s each, then analysis of 666 s. Zero
  preemptions are recorded in `guard109.jsonl`.
- Every timing block was clean on its first try (`exclusivity` = 1 clean record per view).
- Table bytes after every view equal the admission's.
- The merge produced the final `run/manifest.json` (sha256 `b8c09600…`). The capture manifest was copied unchanged to
  `run/manifest_capture.json` (`e899f811…`). Analysis verified all 4,992 raw-file hashes.

**Estimator and stages.**

- **Development:** M\* = 1, chosen by the frozen rule. Predicted GM equal-time ratio 0.882 for M = 1, 0.979 for M = 2 and
  1.224 for M = 4; not futile.
- **LOW and original HIGH identity:** 768 / 768 paired frames identical to R0 within the structural tolerance (asserted
  per frame at capture).
- **Not run:** escalation, because no B1 test was UNRESOLVED; and the dynamic stage, because the static stage failed.

### Measurement

Evaluation views V0 and H1–H5. Reference: the qualified WL107 393,216-spp per-view blocks. 32 paired replicates; R0
grid of 18 levels × 32 replicates. Times are synchronized frozen-frame medians on an exclusive RTX 3090.

**Gates.**

| Gate | Result | Value |
|---|---|---|
| P1 synthetic contract | pass | all gates (see execution record) |
| P2 primary 4-spp equal-time GM ratio ≤ 0.85 | pass | **0.792** |
| P3 its upper95 < 1.0 | pass | 95% CI [0.763, 0.822] |
| P4/B1 paired mean bias within ±0.5% | pass | **30 / 30 EQUIVALENT**; largest \|rel\| 4.5e-4; widest CI ±2.1e-3 (H3 material boundary) |
| P4/B2 spatial bias power upper95 ≤ limit | **fail** | 28 / 30 within; failures below |
| P5 primary 16 / 32 spp ratio ≤ 1.02, upper95 ≤ 1.05 | pass | 0.790 [0.761, 0.821] / 0.793 [0.765, 0.822] |
| P6 silhouette and SSS interior, 4/16/32 spp, ratio ≤ 1.05, upper95 ≤ 1.10 | pass | silhouette 0.847–0.853 (upper95 ≤ 0.915); SSS 0.797–0.798 (upper95 ≤ 0.826) |
| Reference precision ≤ 10% | pass | max reference noise / arm RMSE 0.0108 |
| R0 grid brackets C1 times | yes | interpolation flag `ok` everywhere |

**The two B2 failures.** Values are Q point ± SE per bias spp; upper95 uses the 7-df Student t.

| ROI | 4 spp | 16 spp | 32 spp | Max upper95 | Limit |
|---|---|---|---|---|---|
| H3 material_boundary | −0.0049 ± 0.0029 | 0.0012 ± 0.0048 | 0.0061 ± 0.0032 | 0.0136 (32 spp) | 0.01 |
| H4 material_boundary | 0.0042 ± 0.0096 | 0.0136 ± 0.0042 | −0.0021 ± 0.0063 | 0.0268 (4 spp) | 0.01 |

In both ROIs the B1 mean bias is EQUIVALENT: H3 −1.3e-4 with CI [−2.1e-3, +1.8e-3]; H4 −4.5e-4 with CI
[−2.1e-3, +1.2e-3]. One further B1 interval excludes zero (H2 silhouette +3.0e-4, CI [+1.7e-5, +5.8e-4]). It is inside
the ±0.5% equivalence band, so under the frozen rule it is not a failure.

**Per view, primary ROI at 4 spp** (equal-time ratio with its within-view CI; equal-spp ratio; C1 / R0 cost):

| View | Equal-time ratio | Within-view CI | Equal-spp ratio | Cost ratio | C1 / R0 frame time |
|---|---:|---|---:|---:|---|
| V0 | 0.729 | [0.727, 0.731] | 0.644 | 1.125 | 11.74 / 10.43 ms |
| H1 | 0.801 | | 0.730 | 1.092 | |
| H2 | 0.759 | | 0.717 | 1.056 | |
| H3 | 0.829 | | 0.811 | 1.022 | |
| H4 | 0.830 | | 0.800 | 1.038 | |
| H5 | 0.809 | | 0.734 | 1.096 | |

**Report only.**

- Equal-spp GM ratio: 0.737 at 4 spp.
- 1-spp equal-time GM ratio: 0.809.
- Contact region: 0.763 at 4 spp.
- Material boundary: 0.848 at 4 spp, CI [0.767, 0.939].
- Peak Dr.Jit allocation: 34.4 MiB for both arms.
- Reference against independent R0 (z): |z| ≤ 1.6 everywhere except the silhouettes of V0 (2.62) and H5 (2.74). This is a
  reference/R0 comparison that does not involve C1.

**Variance components, primary ROI at 4 spp.**

| View | var H | var H_C1 | 2Cov(L, H) | 2Cov(L, H_C1) | var_T ratio | Ideal conditional ratio |
|---|---:|---:|---:|---:|---:|---:|
| V0 | 0.0534 | 0.0535 | 0.138 | 0.0001 | 0.644 | 0.506 |
| H3 | 0.0619 | 0.0621 | 0.165 | −0.0003 | 0.812 | 0.741 |

Across all views, var_T ratio C1 / R0 lies between 0.644 and 0.812, and the ideal conditional ratio (exact inner
integration) between 0.51 and 0.74. The jitter share of HIGH variance is ≤ 0.25%.

### Observation

- **Where C1's gain comes from.** Var(H_C1) ≈ Var(H) in every view. All of C1's TOTAL-variance reduction at M = 1 comes
  from removing the positive LOW–HIGH covariance: R0 reads HIGH from the same k = 1 walk that produces E2's NEE.
- **Pattern in the V0 paired-mean map.** The V0 4-spp paired-mean C1 − R0 HIGH map (`V0_maps_4spp.png`, 32 replicates)
  shows a faint negative ring along the sphere silhouette. V0's own B1 and B2 tests pass, so this is qualitative only.
- **The failing Q values do not scale like a fixed bias.** A stationary per-pixel bias gives Q ∝ spp, because Q is
  normalized by Var(TOTAL_R0), which falls as 1/spp. H4's 16-spp point (0.0136) would then predict about 0.027 at 32 spp.
  The observed value is −0.002 ± 0.006. H3 rises with spp but stays within about 2 SE of zero at every level. This check
  is post hoc and descriptive.
- **The failures are bound failures.** The B2 failures come from upper bounds; no Q point exceeds its limit. The frozen
  code classifies an upper95 above the limit as a failure, not as unresolved, and escalation exists only for B1. A spatial
  bias bound that is imprecise therefore closes the batch.

### Interpretation

**Cost-normalized variance reduction: supported on this scene.** C1 at M = 1 reduces the equal-time TOTAL MSE by about
21% at every nominal spp. All six views and all regions improve, and the overhead is 2–13%.

**Mechanism.** The gain comes from decorrelating HIGH from LOW. It does not come from a less noisy HIGH. Better inner
integration (the ideal ratio 0.51–0.74) would need M > 1, which costs more than it returns here (development M = 2 and 4).

**Unbiasedness.**

- *Mean level:* supported everywhere at the ±0.5% resolution.
- *Spatial structure:* not demonstrated at material boundaries in H3 and H4. The data do not resolve a bias there, and
  the post hoc scaling check argues against a stationary one. Nor do they exclude a structured boundary bias of
  Q ≈ 0.01–0.03. The V0 silhouette ring is a qualitative hint in the same direction.
- *Construction:* C1 is unbiased only if its NEE-free inner walk and closure sample reproduce the distribution of R0's
  walk exit. The synthetic contract verified that distribution only on the contract configurations.

### Unresolved

- Whether a real structured bias exists at material boundaries (H3, H4) or silhouettes (the V0 ring). If it does, the
  next question is whether it comes from the inner walk's exit distribution near interfaces or from the closure / bin
  assignment.
- Whether the 8-group Student-t bound on Q is calibrated in small, high-variance ROIs. Example: H5 material boundary has
  χ² = 13.6 on 3 df against zero although it passes.
- Dynamic behaviour (P7), which was not run.
- Generality beyond this scene and table. The gain depends on the LOW–HIGH covariance structure of the frozen estimator.

### Verdict

**NOT_SUPPORTED**, by the frozen mapping: P4/B2 failed in two of 30 bias ROIs. P1–P3, P5, P6, B1 and reference
precision passed. The dynamic stage was not run.

The variance and cost question itself was answered positively: equal-time GM ratio 0.792, CI [0.763, 0.822]. The
batch closes on non-demonstration of spatial unbiasedness at material boundaries, not on a resolved bias.

No adoption and no successor batch follow from this report.

### Reproducibility

**Code commits.**

- Frozen protocol and code: `86e25cd`.
- Amendment 1 driver: `4704081`.
- Amendment 2 guard, timing and driver: `2752ff1`.

Hashes: protocol sha256 in every JSON; analysis code sha in `metrics109.json`; `run_manifest_sha256` `b8c09600…`.

**Data locations.**

- Raw per-replicate frames: 40 GB on LabServer69 at `~/rest_sss101a/results/ecro_estimator109/evidence/run/raw`, with
  hashes in `run/manifest.json`.
- Small evidence (logs, JSON, metrics, figures): mirrored to `results/ecro_estimator109/lab69/`, untracked.
- Reviewer artifacts: [`evaluation/109/`](../../evaluation/109/), with provenance hashes in `manifest109.json`.

**Environment.** LabServer69, RTX 3090, Mitsuba 3.8.0, Dr.Jit 1.3.1, Torch 2.3.1. Frozen inputs from
`results/ecro_denoise108/inputs`.

**Rerun.**
1. `run109.sh` up to capture.
2. `run109_amend2.sh evidence <commit>` for timing and analysis.

Outputs are deterministic in the seeds. Timing values are hardware- and contention-dependent; the exclusivity records
are in the timing parts.
