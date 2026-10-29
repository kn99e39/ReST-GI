# Worklog 57 — Learned Geometry-Conditioned Intrinsic K: Rule vs State

Status: **STOPPED_GATE1 (2026-10-29 00:39 KST); supervision complete, data detectability gate failed, no model training or architecture adoption.**

Latest evidence supersedes the preflight GO below: GO validated implementation/resource readiness; the subsequent data gate did not permit training. See the Gate1 stop report at the end.

User-authorised bounded batch on `Selective_Recompute`, starting HEAD `55b6be0fb994cd33d3e12f34779fae2756413379` independently verified. Historical WL53–56 implementations, artifacts and negative physical results are frozen. No WL58 is authorised.

The exact preregistration is [rule57.json](../../experiments/fnto_rule57/protocol/rule57.json); the input/output semantics, single candidate, matched direct-conditioned control and fair cache policies are specified in [ARCHITECTURE.md](../../experiments/fnto_rule57/ARCHITECTURE.md). Both learned models use shared weights across poses. Rule–State separation is the hypothesis, not the adopted architecture.

O is the independently sampled physical first-exit oracle; S is its rest configuration reused after deformation. D conditions the passive spatial/angular decoder directly on observed geometry; R computes a compact graph geometry state for the same decoder family. The incoming/outgoing flux basis is the unchanged uniform 2.5 mm /16-in/64-out material boundary. Analytic zero-collision transport is shared and its geometry construction, storage and application are fully charged to both arms. Scattered response is learned, with nonnegative substochastic columns by construction. No target oracle is used during candidate geometry preparation or inference.

The fixed split is 13 training poses (rest plus 12 dents), two validation poses and four held-out dents with disjoint train/validation/test light fields. Two independent training seeds per arm and independent physical oracle seeds quantify repeatability. Model parameters are D 161,328 and R 162,048; both may cache partial affine features or all structured coefficients under the same 512 MB ceiling. Actual lifecycle timings at 1/4/16/64 distinct-light queries include geometry and state construction.

Gate 1 must establish reference validity and detectable geometry response before training. Final-step checkpoints are evaluated on every held-out geometry/channel, absolute and signed dynamic fields, angular pairs, fine/coarse spatial errors and hard physical invariants. Neither faster inaccurate model can win. Full conditional composition proceeds only for useful, physical models; learned-vs-oracle, oracle-vs-Cycles and learned-vs-Cycles are separate comparisons. WL56's 9/12 primary physical result and transmitted-blue floor are preserved.

Reproduction: commit the complete code/protocol before primary supervision or training; export that exact commit into `LabServer69:~/rest_wl57/repo`; run `experiments/fnto_rule57/chain.sh` in a detached server session. Stage logs, status, source/environment manifest, target hashes, checkpoints, all held-out rows, measured cost and a server report are retained under `~/rest_wl57`. No Blender/CUDA revalidation is performed: the already validated official Blender 3.5.1 CUDA configuration remains the physical reference scope.

Final measurements, interpretations, separated architecture verdicts and four-case compatibility will be appended after the declared gates finish. Rigid material-frame reuse and local dents are within scope; global bending/twisting and self-contact are untested.

## Pre-primary implementation validation and resource recovery

Initial protocol/code commit: `a9fb15b`. Two pre-primary defects were detected and repaired before any supervision/training: metadata access before selecting the Mitsuba variant (`c738e7a`) and missing angular broadcast dimension in the per-cell RGB light modulation (`704f360`). Original failure logs are retained under `~/rest_wl57/preflight_a9fb15b` and `preflight_c738e7a`. The corrected focused tests passed 6/6; unchanged WL53–56 tests and the actual historical WL54 mask check ran with outputs isolated in WL57.

At the user's explicit request, the server chain now has a durable Python supervisor, exclusive lock, completed-stage markers, atomic data/checkpoint writes and a finite three-policy OOM fallback. Allocation failure alone triggers retries: smaller execution chunks and activation checkpointing, then microbatch-one accumulation preserving effective batch two and CPU-resident geometry streaming. Parameters, training steps, samples, loss, split and criteria are unchanged. Weights/AdamW/random-generator state resume from the last complete checkpoint (every100steps). Failed logs/checkpoints and actual elapsed attempt costs are retained. Changed trace chunk partitions are recorded because they can change realised Monte Carlo streams; they do not reduce the photon budget. Unknown failures and exhausted recovery policies halt with FAILED; no shared files are deleted. Fault-injection tests verify recovery, duplicate-stage skipping, unknown-error halt and bounded exhaustion. The current batch is not reported as a success merely because a chain starts.


Supervisor-transition defect (implementation/measurement provenance, no acceptance change): the initial shell had already launched supervision before its parent was paused. Replacing the parent alone left one old WL57 data generator orphaned, overlapping the new supervisor briefly. A PID/SID audit detected it; only that old child (PID1366642, SID1365444) was terminated. Rest/tr00/tr01 and geometry inputs completed under704f360 are inherited with hashes. tr02 was independently generated by both processes with the same seed/budget; the surviving primary artifact is the supervised replacement, and the earlier duplicate raw artifact was overwritten before this audit (its run log is retained; it is not claimed as preserved independent evidence). The old child was on tr03 when terminated; no completed tr03 file from it was used. This interval is excluded from uncontended runtime claims. No fit or held-out result existed. A separate data-writer lock and Linux parent-death signal were added for subsequent launches; duplicate supervisor rejection must leave the live status unchanged. The current controlled chain is the sole remaining WL57 writer.


## Focused preflight A1 — user-requested correction before training

The user supplied a static-review brief against704f360 and explicitly stopped mass supervision/training pending focused correctness review. The active primary chain was stopped at00:09:10KST on2026-10-29; no checkpoint or model training existed. Completed physical caches and incomplete generation logs are retained. A subsequent explicit user instruction authorises automatic server-side primary continuation **only if the final corrected preflight is GO**.

Confirmed implementation defects: v1 summed aliased geometry/state tensors twice, copied geometry to CPU during GPU byte accounting, inherited CUDA peaks across models and retained caches during the opposite policy's lifecycle. A1 counts unique storage allocations including sparse arrays, measures incremental/total cache storage and parameters separately, and starts a fresh process per model/seed/pose/policy with separate peak resets for preparation/query/lifecycle.

Architectural concern: R alone had learned graph context, so an improvement could not be credited to explicit state separation. The smallest bounded control repair keeps R unchanged and provides D fixed raw first-hop, second-hop and global means over the same material graph. D's input is560 channels, hidden48/node74,160458parameters; R remains162048 (<1% difference). This analytically matched control was selected without any model fit or held-out model result. The allowable claim is about this concrete learned nonlinear graph-state mechanism versus direct conditioning at matched context scope and fair caching. It cannot isolate abstract state separation from nonlinear aggregation, inductive bias or parameter allocation. No third family, capacity sweep, threshold/split/reference change or adoption occurs.

A1 adds an actual rigid-operator action test after a world transform and material-frame pullback, checksum receipt/quarantine checks for complete cache names, atomic JSON writes, RNG-resume tests, fresh-process cache memory smoke and paired untrained forward/backward resource smoke (no optimizer steps). Disk bounds conservatively retain one uncoalesced record per photon, all action targets and geometry; CPU/GPU limits include training-resident data and geometry allowances. The exact corrected commit must receive GO before the supervisor accepts primary execution. Measurements and final GO/NO-GO follow below.


## Focused preflight A1 final report — GO (2026-10-29)

### Implementation Fact

Final executable source/protocol: `dc6a5326b3abf0d962c18f3536e7b1cf376fe624` (A1 implementation `44180a3` plus a final parameter-iterator ledger correction). The latter materialises the parameter iterator once so retained totals include parameters; an explicit iterator test prevents regression. The first GO and subsequent brief revalidation pause are retained under `~/rest_wl57/preflight_44180a3_preserved`. Final evidence is [results/fnto_rule57/preflight_v2](../../results/fnto_rule57/preflight_v2/), including decision, raw measurements, environment, CSV and hashes.

R is unchanged; D now receives fixed raw two-hop/global context. Both see current geometry and the same physical/material input, have the same local/tail/angular kernel family and cache opportunities, and remain within1% parameters (D160458/R162048). No model has been trained to select this correction. Git comparison against704f360 confirms **all numerical evaluation gates, training settings, geometry/light splits, reference settings, seeds and sample budgets are unchanged**.

### Measurement

Actual RTX3090 GPU tests on LabServer69: focused7/7, extra controls3/3, recovery fault-injection4/4. Historical regressions previously independently rerun against unchanged source: WL53 9/9, WL54 10/10 (generic runner's mask case explicitly skipped), WL55 10/10, WL56 6/6, plus the separate actual WL54 mask check passed. These historical results are retained in `results/fnto_rule57/preflight/tests`; they are not new physical-quality evidence.

The real rigid test transformed the rest mesh by37degrees about(1,2,3) plus(0.03,-0.02,0.011)m, pulled the current geometry back through the declared supplied material frame, rebuilt geometry/ray intersections/coefficients, and compared three incident fields with rest-K reuse. Relative output L1: **D8.74585e-8, R1.07884e-7** (limit1e-6), equal canonical geometry identity; a dent invalidates identity. This exceeds a coordinate-inverse-only check. It uses an untrained operator and32 within-bin geometry samples, not an assertion about learned held-out fidelity.

Unique cache allocations, including geometry, in decimal MB:

| Arm | Parameters | Compact cache | Full cache | Incremental state | Incremental coefficients | Parameter bytes |
|---|---:|---:|---:|---:|---:|---:|
| D |160458|117.247|176.237|10.076|58.990|641832|
| R |162048|111.594|170.584|4.424|58.990|648192|

Common geometry is107170680bytes. Actual retained full geometry/state/coefficients/parameters: D176878752bytes, R171232632bytes; all four policies meet the same512000000byte ceiling. Synthetic aliases/views add zero duplicate bytes; sparse indices and values are counted. Each arm/policy was measured in a separate fresh PID (D1371271/1371357, R1371557/1371643). Cache-preparation PyTorch-CUDA peaks: D compact220508672/full399991296bytes, R compact172376064/full415318016bytes; query peaks: D399991296/258219008, R415318016/252574208bytes. These are explicitly allocator-scoped PyTorch allocated peaks, not an exact whole-driver peak. CPU RSS and CUDA reserved peaks are separately retained in raw worker JSON. No speed or quality advantage follows from this untrained smoke.

Paired rest/dent forward/backward at effective batch2, **zero optimizer steps**, produced finite gradients. D allocated/reserved peak1.826/1.921GB; R3.081/3.169GB. Activation checkpointing plus microbatch accumulation agrees in gradient relative L2 to1.98e-7 on the focused control. RNG restoration was bit-exact for NumPy/Torch CPU/CUDA. Simulated interrupted partials do not replace a complete file; corrupt or unreceipted completed filenames are quarantined with reasons. Unknown errors stop rather than silently retry. OOM fallback, completed-stage skip and finite exhaustion were tested by fault injection, not a deliberately induced real device failure.

Resource feasibility before resumed mass generation (decimal units):

| Resource | Required, conservative | Available, measured |
|---|---:|---:|
| Disk on `/home/nd/rest_wl57` |97.18GB|1399.80GB|
| CPU RAM |25.77GB|128.16GB|
| GPU |6.15GB|25.16GB free|

Disk reserves up to one uncoalesced20-byte record per photon across24 oracle operators (60.40GB), all action targets13.59GB, geometry2.04GB and extra headroom. GPU is a paired-gradient reserved peak plus conservative resident-geometry allowance and512MiB headroom; **it is an estimate of full-training feasibility, not a measured training peak**. Large extra driver allocations remain a measurement scope limitation, with substantial available headroom. No full supervision suite or optimizer training was performed by these preflight tests.

### Observation

The fixed-geometry operator algebra, material-frame rigid reuse, cache identity and bounded execution controls pass. Unique allocation accounting removes the original alias double count and cross-model peak contamination. Both actual cache modes fit comfortably. Resource smoke does not validate static/dynamic learned fidelity.

### Interpretation

The previous positive-R interpretation was too broad. Even after matching available context scope, the comparison bundles learned nonlinear aggregation, inductive bias and parameter allocation into the specific R mechanism. A positive result can support that implementation at quality-matched measured cost; it **cannot prove that abstract Rule–State separation alone is the cause**. Fair D full caching may eliminate any update/query advantage. This is declared before any model fit; no abstract architecture adoption is authorised.

Reference uncertainty and the existing blue K/C physical floor are unchanged. No reference-quality verdict, TransGI claim, learned-K success or production change follows from preflight GO. The earlier prematurely started partial supervision and wrapper overlap remain documented and preserved; only checksummed complete physical caches may be reused.

### Unresolved Question

Whether either fixed learned model reaches useful held-out absolute and differential fidelity, and whether R has a quality-matched update/query advantage against D's best fair cache, remains unmeasured. Full-training peaks and non-PyTorch driver memory must remain distinguished. Global bending/twisting and self-contact are untested.

### Architecture Verdict

**Preflight: GO** for the explicitly narrowed, context-controlled specific mechanism comparison. Implementation measurement correctness and rigid material-frame control: SUPPORTED in these focused tests. Resource feasibility: CONDITIONALLY SUPPORTED by measured bounded smoke plus conservative estimates. D/R learned viability, R advantage, learned-K readiness and coupled physical GI: INCONCLUSIVE pending primary evidence. Rigid material-frame compatibility is measured; local-deformation identity/gradient operation is checked without fidelity claims; global bending/twisting and self-contact remain INCONCLUSIVE. No major architecture adoption or next research batch is decided.

### Reproducibility and server continuation

Final GO at00:22:46KST2026-10-29 for `dc6a5326b3abf0d962c18f3536e7b1cf376fe624`. Exact commands through `experiments/fnto_rule57/run.sh`: `tests/test_r57.py <out>`, `tests/test_preflight.py <out>`, `tests/test_supervisor.py <out>`, `r57_resource.py`. The last invokes fresh untrained `r57_cost.py worker rest {D,R} 5701 {compact,full} <out> untrained` processes and paired-gradient smoke. Environment is recorded in `environment.json`; split/budget controls are in `protocol/rule57.json`; raw file hashes are in `artifact_manifest.json`.

The user's latest instruction explicitly superseded the preflight-only no-launch request: **on final GO automatically continue**. The detached preflight process exec'd the primary supervisor with the same PID/SID1371033 and PPID1. It independently continues reference completion → data gate → four fixed model/seed fits → held-out evaluation → fresh-process cost → conditional composition → server report, with finite OOM recovery and atomic resume. A failed gate or unrecoverable defect stops with logs; it never reports batch success merely because execution started. Current preflight completion is distinct from completion of WL57's architecture experiment.


## Primary chain outcome — Gate1 stop (2026-10-29)

**Implementation fact.** The detached chain completed physical supervision at00:39:18KST, evaluated Gate1 at00:39:21, wrote its stop report and exited at00:39:23 with `STOPPED_GATE1`. The user's later status request was checked at01:05KST: no WL57 process remained, GPU0%/1MiB. No OOM or lost SSH session caused this stop. Checkpoint directory is empty; no D/R training, held-out model comparison, primary model-cost evaluation or composition occurred. Earlier preflight GO is implementation/resource readiness, not a pass of the subsequently measured reference detectability gate.

**Measurement.** All24 physical target packages were completed (19 base poses plus rest/four tests with second seeds);24/24 package physical checks pass, maximum leak/wrong-side fraction6.28e-7. Stored oracle matrices occupy26GiB and geometry/action data15GiB on server. Final resumed supervision-stage wall time was986.6s, excluding earlier generation/reused caches; this is not a total offline-construction cost claim.

The frozen signal rule requires SNR>=3 in at least4/16 concentrated fields in at least two channels, on at least3/4 test poses. Actual R/G/B counts:

| Held-out pose | R | G | B | Qualifying channels |
|---|---:|---:|---:|---:|
| te0 |1|2|0|0|
| te1 |0|0|0|0|
| te2 |0|1|0|0|
| te3 |2|3|1|0|

The independent-seed action difference relative L2 averages3.07–3.35% over the test distributions. Some individual concentrated fields exceed SNR3 (maximum5.22), but the declared population-level requirement fails on every test pose. This is not evidence that geometry never changes K.

**Independent validation.** A CPU-only NumPy float64 replay used direct nested-cell/angular reshaping and sums, without importing production `aggregate()` or `data_gate()`. All twelve qualifying fractions replay exactly (absolute difference0); the stop is reproduced. A separately saved post hoc replay script is `experiments/fnto_rule57/r57_gate_audit.py`. Original gate output/logs/report/manifests and replay output are retained in [results/fnto_rule57/gate1_stop](../../results/fnto_rule57/gate1_stop/). This audit performs no new oracle generation or model fitting and changes no criterion.

**Interpretation.** The fixed2048-photon-per-input-cell, independent-seed target/action setup did not supply the declared deformation detectability across the fixed held-out light population. The gate's noise definition is the per-operator independent-seed difference divided bysqrt2, while the compared response is a two-seed mean; this conservative, frozen definition is reproduced rather than changed after results. The audit confirms the implemented decision; it does not prove that every uncertainty formulation or higher-budget estimator would fail. No threshold, split, geometry/light distribution, seed budget or architecture was changed to obtain a pass.

**Architecture verdict.** Data readiness for the declared learned dynamic comparison: **NOT SUPPORTED** at this budget/distribution. D viability, R viability, R advantage and learned-K readiness: **INCONCLUSIVE**, because neither model was trained. Coupled physical GI and global-deformation/self-contact claims remain unmeasured; WL56 negative physical findings are preserved. This is a data-gate closure of the attempted batch, not a falsification of either neural architecture. Preflight resource/operator controls remain supported in their limited scope.

**Unresolved question.** Whether a justified different estimator/sample budget can resolve the declared geometry effect is the remaining prerequisite; the present evidence does not select a new estimator, altered population or research batch. Existing stop conditions prevent automatic training after this failure. The chain is intentionally not restarted and no subsequent architecture batch is initiated.

**Reproducibility.** Primary source remains `dc6a5326b3abf0d962c18f3536e7b1cf376fe624`; exact protocol and splits are unchanged. Server artifacts: `~/rest_wl57/out/gate1.json`, `out/verdict.json`, `out/SERVER_REPORT.md`, `logs/data_gate.attempt0.log`, all oracle/target manifests, and status `STOPPED_GATE1`. The original generic server-report footer lists possible downstream filenames; heldout/cost/composition outputs do **not** exist for this stopped run. Raw matrices/actions remain on the server; small committed artifacts contain their identities/hashes. The post hoc replay is evidence verification, not predeclared confirmatory training evidence.


## A2 authorised Gate1 recovery — diagnostic then independent confirmation

User brief: "WL57 Gate1 Recovery and Conditional Primary Continuation" (2026-10-29). Original STOPPED_GATE1/dc6a532 source, thresholds and failed observed population remain unchanged. All original test outputs are now diagnostic; a fresh independent confirmation is preregistered in `experiments/fnto_rule57/protocol/rule57_recovery_a2.json`. This is still WL57, not a successor batch or a new learned architecture. Internal GO automatically continues the fixed D/R experiment; NO-GO stops without another recovery cycle or user confirmation.

Stage A used immutable original actions and raw operators only (`r57_recovery_diag.py`, commit6e5c098). At1cm spatial-pair aggregation, R/G operator-change SNR is3.53–9.06 and relative change0.74–1.91%, corroborating WL55 Stage E's measurable coarse intrinsic change. Full light-action signals are weaker; effective conditional photon count2048/sum(x^2) ranges35,358 to21.6M among focused cases although each full matrix used125.8M photons. Only24–35% median focused input flux lies in the geometry-defined dent footprint; last16 directional pairs have zero median footprint excitation. Geometry-defined local regions carry66–96% of aggregate noise-debiased R/G change (blue16–80%, noisier). Whole-boundary/weakly excited cases sit near the null ratio~0.707. All per-field/channel, face, ring, local/complement and seed-noise quantities are retained, not used to select favorable new fields. These support mixed causes: real localized intrinsic change plus poor/uneven excitation and spatial/angular estimation noise; they do not establish universally negligible change or a neural failure.

ONE recovery strategy: sample prescribed incident-flux fields directly with q(source cell/bin)=mean_RGB x, trace the unchanged physical first-exit walk with unit RGB weights, and apply x/q at exit accumulation. Expectation is exactly sum q E[T|s]x/q=Kx; RR/path decisions are independent of queried weights for fixed q. No CRN across geometries, output renormalization, clipping, air continuation or full new K matrix. Float64 first/second moments preserve the existing radiometry. A small original-rest/old-field pilot checks finite-seed common-proposal superposition, zero, nonnegativity/energy and statistical agreement with the old oracle before confirmation.

Prospective population: four new tuples drawn from Python Random578007 over the original depth/radius/centre ranges (exactcf0..cf3 in protocol), plus rest;64new test fields from seed578103 and the unchanged generator. Original training/validation fields and targets are reused exactly in a dedicated training-only directory. Confirmation targets are never loaded by the optimizer. No D/R class/parameter, optimizer, loss,6000step,5701/5702, final-checkpoint or quality/cost gate changes. The scoped source edits only register the new observable geometry manifest, select a separate training-target directory and route the durable supervisor to the isolated recovery runner.

Fixed confirmation budget: two independent64M-photon replicas per pose/field, each committed as two independent32M halves. First generate only16focused fields on5poses (10.24B photons), then evaluate the frozen population gate. Require global1cm/16out SNR>=3 on>=4/16 fields in>=2channels and>=3/4 new poses; local regions remain diagnostics. Four-half variance estimates the same single64M-replica noise scale (mean-delta SE reported separately). Independent photon-second-moment noise calibration and physical checks must also pass. Only on GO fill the remaining48fields (30.72B more), then run all four frozen training fits, held-out evaluation, fair costs, conditional known-scene composition and full server reporting. If the gate fails there is no adaptive estimator/budget/population retry.

Resource plan (preregistered, not measured speed): at most40.96B confirmation photons, no full new fine matrices,>=35GB disk,>=24GiB available RAM and>=7GB GPU plus pilot-derived conservative peak check. Existing full-operator throughput projects roughly6.4GPUh before differences in dense-action accumulation; pilot records actual times/peaks. This is a bounded numerical sample plan, not a claimed neural speedup or an arbitrary short wall-time cap. Old physical matrix/actions remain read-only; the recovery root is `LabServer69:~/rest_wl57_recovery` with atomic half/checksum receipts, source/config identity, exclusive locks, finite OOM chunk reductions and no shared-file deletion. Primary results and final scientific GO/NO-GO are not yet known at this preregistration point.


### A2 implementation/resource pilot — passed, confirmation gate still pending

Preregistration/executable commit: `e809c15` (before any A2 confirmatory ray). Source/protocol was pushed to `Selective_Recompute`. Detached supervisor PID/SID1381850, PPID1, isolated server root `~/rest_wl57_recovery`. Existing primary root `~/rest_wl57` is stopped and untouched. Stage/source/config completed receipts prevent incompatible resume; per-half data receipts include payload SHA, protocol SHA, commit, pose/field/seed/sample count, physical result and runtime. The original protocol and frozen D/R source have zero diff from b27bd02.

Focused tests7/7 and bounded supervisor controls4/4 passed. The fixed diagnostic pilot used old rest fields0/48, four1M independent runs per field. New-vs-old coarse action error is0.86–1.04times combined norm-SE, below the predeclared2times limit in every channel. Integrated differences are below5combinedSE. All8 physical action checks passed. Same-proposal A/B/A+B superposition relativeL1=6.20e-9; zero response exactly0. These verify the implemented weighted estimator and do not establish the new population's detectability.

Pilot resource evidence: peak PyTorch reserved0.537GB, conservative primary allowance4.074GB versus25.059GB available (before this process's reservations), available CPU127.579GB, scratch1366.054GB. The fixed requirement>=7GB GPU remains conservative. Pilot1M actions took1.93–2.11s each. The first completed confirmation32M halves took~34.4–34.8s each; this is actual oracle cost, not a learned-speed claim. At that initial rate the320-half signal stage is roughly3hours and the full1280-half dataset roughly12hours if GO, with geometry/lighting-dependent runtime variation. The numerical budget is unchanged by these measurements. Source/ray sample counts and actual elapsed stages are recorded; no universal short wall-time cap or efficiency claim is imposed.

At this update only the **estimator/resource pilot** has passed. The independent scientific GO/NO-GO has not yet been computed. The server is generating exactly the first16new fields for the fixed four fresh local deformations plus rest. On failure it writes a clean NO-GO report and stops; on success it completes the other48fields then runs fixed D/R fits, held-out tests, fair per-process cost, eligible historical transfer composition and `out/RECOVERY_REPORT.md` / `out/recovery_final.json`. No user or live ChatGPT session is required for these transitions. Primary architecture verdicts remain INCONCLUSIVE until those stages finish.

Small diagnostic/pilot artifacts are preserved in `results/fnto_rule57/recovery_a2/`; exact full confirmation half payloads remain on the server. The fully machine-readable protocol, seeds and frozen controls are in `protocol/rule57_recovery_a2.json`.


### A2 pre-result measurement/report controls and live handoff

Two report-only omissions were corrected before any model fit/result: noninferiority now checks the frozen0.02 mean-relative-L2 tolerance at BOTH declared primary absolute-action resolutions (coarse and fine16), with the existing hard p90/fine fidelity limits retained; and each future isolated cost worker records external GPU processes before/after measurement. Unknown or contended occupancy cannot support an R speed-advantage claim. CPU tests4/4 cover fine-quality regression and one-way noninferiority (9b7006e); occupancy controls3/3 cover live-peer detection, excluding own PID and unknown-occupancy handling (cffe2b0). These do not alter estimator, sample count, population, models, training or numerical thresholds. Old report/cost source is preserved under `report_versions/`; the primary dataset/training identity remains e809c15. `report_correction_commit.txt`, per-worker cost-source SHA and verdict report-source SHA distinguish auxiliary measurement/report provenance.

A separate pure-stdlib read-only observer (18477c5) writes `~/rest_wl57_recovery/progress.json` every30s: supervisor liveness, completed checksum-committed halves, physical-pass count, median per-half runtime and explicitly approximate data-only remaining time. It never starts/restarts/stops a stage, changes a gate or reads intermediate signal to select fields. At terminal state it formats all fixed model/seed worst-case fidelity, each geometry/seed/query best-cache cost, every failed ruled coupled group, scope limits and a ready-to-append Worklog artifact in `out/FINAL_REVIEW.md` and `out/WORKLOG57_A2_FINAL_APPEND.md`. Full machine-readable outcomes remain `out/recovery_final.json`, with raw held-out/cost/composition records retained. No ChatGPT/SSH lifetime dependency is introduced.

Interpretation precision:2048/sum(x^2) is an excitation-based effective-sample proxy under approximately comparable conditional variance, not an exact count of independent weighted photons or a measured new-estimator variance. The reported local-energy shares are exploratory noise-subtracted accounting with finite-seed uncertainty and a nonnegative global-energy floor; they are not inferential confidence bounds. The independent confirmation's global criterion and photon/half-variance calibration are specified separately and do not depend on these exploratory shares.

Current A2 state at the last observer snapshot (02:01KST2026-10-29): confirm_focus RUNNING, supervisor alive,30/320 focused half packets committed, all30 physical checks passed. These are reference-generation progress measurements, **not a scientific GO or a learned result**. The frozen conditional chain remains responsible for the eventual GO/NO-GO and, on GO, complete fixed primary evaluation and reporting.


## A3 — user-authorised GPU execution optimisation (2026-10-29)

The user identified inefficient GPU use during confirmation and explicitly requested code/batch optimisation. At inspection the2M-photon chunks used~1.28GB GPU memory,~10% sampled GPU utilisation and~102% of one CPU core, with32M halves taking~34s. This was a conservative unprofiled execution choice, not an intrinsic physical-reference cost lower bound. The primary was safely paused;52complete packets were checksum-verified and preserved. Only the incomplete active packet was discarded, with the interrupted attempt logged (`performance_transition.json`). Scientific A2 protocol,64M replica/32M half counts, seeds, poses/lights, physical materials, importance estimator, D/R source and all thresholds are unchanged.

The isolated `r57_fastwalk.py` specialises the immutable WL53 walk for the A2 unit-RGB/no-NEE/no-air-continuation case. It reuses compaction indices, defers flux counters to a single GPU-to-CPU transfer, removes unused exit positions/unused S2 temporaries/single-input concatenations, and skips only a provably false RR condition: unit initial RGB and max albedo>=.998 imply relative weight>.05 for the first1024collisions, even with ample float32 error margin. Later RR and the exact canonical intersection/phase/absorption code are retained. Weak references remove an accumulator/sink cycle retaining CUDA tensors. Historical WL53–56 files remain untouched.

### Actual performance and physical/numerical validation

Same32M photons, original rest diagnostic light0, seed580101; separate process per timing:

| Path | Photons per launch group | Wall seconds | PyTorch reserved peak GB |
|---|---:|---:|---:|
| Original reference |2M|34.419|0.807|
| Specialised |2M|27.892|0.724|
| Specialised |8M|8.459|2.628|
| Specialised |16M|4.608|5.100|
| Specialised |32M|2.971|10.106|

This is **11.6x** data-generation throughput on that matched control, not a learned-K or R-vs-D speed advantage. Larger batching supplies most of the gain; removal of repeated host synchronisation/temporary copies adds another~15–20% at the same batch. The fastest tested32M configuration leaves ample room on the24GiB3090. Bounded allocation fallback is32M→16M→8M→2M→250k, with unchanged total photon count. No configuration was selected by confirmation SNR or model quality.

Reference-versus-specialised controls used old rest/te0 and old fields0/48 at1M and full32M. In all8comparisons: identical final RNG state and photon/collision/exit/leak/wrong-side/truncation/step counts; float32 action response exactly identical; float64 second-moment relativeL1<=2.12e-16. All physical checks passed. At32M a canonical full-batch walk itself takes3.18–3.54s versus specialised2.69–2.95s. These exact same-chunk tests distinguish execution transformation from estimator changes. Changing the chunk size changes RNG assignment but not its iid distribution, unbiased flux estimand or fixed photon budget; existing completed packets remain valid and are never overwritten.

Non-invasive CPU profiling of250kphotons counted `.cpu()` calls331→5, `.any()`1608→0, and `torch.cat`2010→0. Explicit nonzero-call counts increase because previously implicit boolean compactions are made explicit; no nonzero-count reduction is claimed. Torch/CUPTI CUDA profiling caused illegal-address errors in BOTH the unchanged reference and specialised Dr.Jit/OptiX contexts; those diagnostic failures are preserved and are not CUDA kernel-timing evidence. Unprofiled fresh-process controls and subsequent actual confirmation runs pass. No driver, library or lab infrastructure was changed.

### Deployment and continuity

Execution-only amendment: `protocol/rule57_execution_a3.json`; implementation09f5af3, separately identified from scientific preregistration e809c15. The server's immutable `execution_policy.json` contains source hashes, parent scientific protocol/commit, validation receipts, measured memory condition and the52preserved payload hashes. New packets record actual walker/chunk, execution-code/policy hashes and RNG-state hash. Source/config/receipt checks and exclusive locks remain active. Existing scientific stage markers are reused only because their input/model/estimator contracts remain valid; the new execution policy is separately checked before resumption. No second scientific recovery cycle occurs.

Independent supervisor PID/SID1390733 (PPID1) resumed confirm_focus attempt1 at02:37:39KST. Initial live fast packets take~2.94s, agreeing with the benchmark;82fast packets had completed with all physical checks passing by the subsequent audit. This is actual confirmation throughput, not just a synthetic benchmark. The observer uses timings from the current execution policy only, so the earlier~12hour extrapolation is superseded: remaining data construction is approximately one hour at the new initial rate if the signal gate passes, with the signal gate's remaining data approximately10–15minutes after restart. Geometry/field runtime can vary; training/evaluation/cost follow separately. Scientific GO is still pending at this execution update.

Small original-source snapshots, all timing/physical/equivalence data, profiler failure logs, CPU profiles, deployment policy and preserved-packet hashes are in `results/fnto_rule57/recovery_a3_execution/`. The A2 conditional sequence remains: full independent signal gate → fixed primary if GO; else clean NO-GO with evidence, no adaptive budget/threshold change.


## A2/A3 primary completion — measured bounded negative result (2026-10-29)

### Implementation Fact

The detached server chain **finished normally**, rather than failing or losing its session. Server log records COMPLETE at04:01:35KST on2026-10-29 (server clock was several minutes behind the client; use approximately4am for wall-clock communication). It completed1280independent32M-photon half packets, all4fixed6000-step model fits, held-out evaluation, both fair cache policies across all declared query counts, eligibility checking for composition, and final server reports. GPU became idle because no work remained. No OOM, unknown exception, retraining, best-checkpoint selection, threshold change or later batch occurred after A3 resumption.

Scientific source/protocol identity remains e809c15/A2; A3 execution09f5af3 preserves that contract and its52already-completed packets. The original dc6a532 STOPPED_GATE1 remains an unchanged historical failure. Auxiliary reporting/cost correction cffe2b0 and read-only observer/source hashes are recorded separately. Final weights for D/R seeds5701/5702 were copied locally and independently SHA256-verified against the server's recorded checkpoint identities.

### Measurement — independent reference/data gate

The single prospective confirmation passed: all4fresh geometry cases satisfy the original population thresholds under the new unbiased importance estimator. Counts of qualifying focused fields out of16, R/G/B: cf0 **5/5/4**, cf1 **11/9/9**, cf2 **9/9/8**, cf3 **7/7/5**. Physical validity and photon/independent-half noise calibration pass. The completed test reference uses40.96Bphotons in1280half packets and two independent64Mreplica bundles per pose/light. All1280packets pass their physical checks. No original observed test field was relabelled as independent confirmation.

The maximum across poses/channels of the mean coarse relative independent-bundle noise norm is0.356%. This is an empirical noise measure, not a proof of zero bias or a physical image-error bound. The new reference makes the declared deformation effect resolvable; it does not make either neural representation successful.

### Measurement — fixed learned candidates

All4training runs reached exactly6000steps and finite losses. D took329.1/328.9s and R420.7/421.1s of recorded training time; end-to-end stage wall times were335.0/333.5s and425.4/425.7s respectively. Peak PyTorch training GPU allocations: D3.348GB, R4.603GB. Training CPU high-water RSS was not captured; do not substitute the resource-preflight estimate for a measurement. Fresh inference workers report CPU high-water1.397–1.479GB and PyTorch allocated peaks up to0.419GB, excluding non-PyTorch driver allocations.

Worst mean error across all5geometry cases (rest+4held-out) and RGB, plus worst detectable deformation error:

| Model | Coarse mean relL2 max | Fine16 mean relL2 max | Delta relL2 max | Delta correlation min | Dynamic groups passed/12 | Angular pair/channel failures/96 |
|---|---:|---:|---:|---:|---:|---:|
| D5701 |7.041%|11.475%|66.34%|0.7652|7/12|53/96|
| D5702 |7.055%|11.577%|65.79%|0.7535|6/12|60/96|
| R5701 |6.762%|11.249%|62.47%|0.8216|7/12|36/96|
| R5702 |6.784%|11.262%|59.92%|0.8335|7/12|36/96|

**Absolute-action norm subsets pass** at every pose/channel: coarse mean<=10%, coarse p90<=20% (worst11.84%), fine2.5mm/16out mean<=15%. Fine64bin mean errors are reported diagnostically (worst14.99%). **This is not complete static directional fidelity**: the independent equal-irradiance directional-contrast requirement fails on every test geometry for every model. All96pair/channel cases per model are detectable; these are descriptive counts over repeated light pairs/geometry conditions, not96independent statistical trials. Every row is retained in `heldout.json` and the post hoc `directional_all.csv`.

**Deformation fidelity also fails**: the declared delta relativeL2<=35% and correlation>=0.90 contract is not met for all qualified channels/groups. R/G pass oncf0–cf2; cf3 R/G and most blue responses fail. R is somewhat better, particularly in blue and directional contrast, but still misses the complete contract in both seeds. Detection masks are generated by reference noise only, identically for both models. The confirmation gate uses the preregistered four-half variance estimator, while the frozen evaluator uses two independent bundle differences; e.g.cf1blue is9qualified fields at the readiness gate and8in the evaluator. This implementation distinction is documented and was not changed to favour a model.

Zero input, fixed-geometry linearity, nonnegativity, analytical passivity and repeated-cache agreement pass all20model/geometry checks. This supports those mathematical/implementation constraints only; it is not a claim of complete physical transport accuracy or learned reciprocity. Directional/dynamic fidelity remains deficient despite passivity.

**O/S/D/R interpretation.** The stale exact rest operator's largest coarse mean absolute error on the new geometry/light population is1.113%, below the learned models'~7% absolute errors. Its predicted intrinsic delta is exactly0, hence relative delta error1on a nonzero detected change. Learned models reduce that delta error to~0.2–0.66 on qualified groups and thus learn part of the geometry response, while their shared absolute approximation bias is much larger. Partial change prediction must not be confused with better absolute physical accuracy than the exact stale control.

### Measurement — cost, memory and conditional composition

Both legitimate cache policies were measured in fresh per-model/pose/policy processes with unique-storage accounting, CUDA fences and before/after external-process checks. All reported comparisons were uncontended. Including the common geometry build, R/D lifecycle-time ratios across4poses,2seeds and1/4/16/64queries range **0.9925–1.0115**; none demonstrates the preregistered25%benefit. Typical first-query total: D0.340s, R0.343s;64-query totals~0.432–0.438s. The common geometry preparation is~0.329s. It contains mesh/boundary/features/analytic-ballistic work and is common to both; its internal production optimisation was not studied. Even excluding it oncf0/seed5701, measured learned cache-build+query is10.60ms D versus13.76ms R at1query, and103.18ms versus103.85ms at64queries. Thus no hidden R speed win appears in this measured learned portion either.

Example full geometry+coefficient caches oncf0: D177.08MB, R171.43MB, within the same512MBlimit. R saves about5.65MB through compact state, but no useful quality-matched speed advantage is established. Training's R GPU peak is higher; do not generalise compact-state savings to all memory scopes. Raw oracle/data-generation and training costs are separate from online cost. The interrupted original slow52packets, A3 performance validation, both actual confirmation attempts and completed fast data construction are preserved; the final reference stages alone total approximately5013.5s including the deliberate interruption's stage wall record. This is not an end-to-end neural GI speed claim.

**Conditional K/C composition was intentionally skipped**, after testing eligibility: no model passes the complete operator fidelity contract. The logged composition stage is the eligibility-check script, not a coupled GI solve. The existing WL56 oracle-vs-Cycles blue floor is neither repaired nor newly evaluated here. No monolithic-reference GI accuracy is inferred from K-only norm agreement.

### Observation

Intrinsic geometry changes are independently detectable. Both learned models approximate absolute actions and partially track changes while obeying zero/linearity/energy constraints. R modestly improves several errors and has fewer directional failures, but neither candidate preserves the required directional and deformation responses. Their fair cached runtime is almost equal at the measured complete-update boundary, with R slightly slower in most low-query cases.

### Interpretation

This is a **completed negative result for the fixed recipes and acceptance contract**, not another reference-readiness stop. It is stronger than the original Gate1 failure because the independent data gate passed and the full matched learned comparison actually ran. Nevertheless it does **not falsify every possible Rule–State architecture** or establish that direct conditioning is universally preferable. Neither candidate reaches useful complete fidelity, so a quality-matched runtime winner cannot be selected.

The remaining error is much larger than the measured new-reference seed noise. Its precise cause is not isolated: the common finite spatial/angular kernel basis, geometry-only ballistic quadrature, original noisy training supervision and optimisation could contribute. Validation loss improved substantially and both seeds give similar results; this does not prove global optimisation convergence or that capacity alone is responsible. R's nonlinear aggregation/inductive bias and parameter allocation remain bundled by design; no abstract state-separation-only causal claim follows. No after-the-fact capacity/angle-rank/sample-budget/threshold sweep or retraining is started.

### Unresolved Question

Which shared approximation or supervision/optimisation limit prevents sufficiently accurate angular and deformation response is the material unresolved dependency before another representation decision. This batch does not choose a new mechanism to repair it.

### Architecture Verdict (separate scopes)

| Question | Verdict | Scope |
|---|---|---|
| Independent reference/data readiness | SUPPORTED | Fresh A2population and fixed sample contract; old failed gate retained |
| Absolute-action norm thresholds | SUPPORTED | Declared coarse/fine16 aggregate norms only |
| Directional-contrast fidelity | NOT SUPPORTED | Every learned model/geometry has failures |
| Deformation-response fidelity | NOT SUPPORTED | Both seeds/classes miss required groups |
| Hard zero/linear/nonnegative/passive/cache constraints | SUPPORTED | Tested numerical constraints; not complete physical fidelity |
| Direct learned K viability | NOT SUPPORTED | This frozen model/training recipe |
| Rule–State learned K viability | NOT SUPPORTED | This frozen model/training recipe |
| R advantage under declared success rule | NOT SUPPORTED | No useful eligible model and no measured25%cost benefit |
| Quality-matched D/R runtime winner | INCONCLUSIVE | Both miss full quality; raw costs still reported |
| General abstract Rule–State hypothesis | INCONCLUSIVE | Not an exhaustive class/abstraction test |
| Learned-K integration/composition readiness | NOT SUPPORTED | Operator eligibility fails |
| Actual coupled GI accuracy | INCONCLUSIVE | Not run after eligibility failure; historical blue floor retained |
| Rigid material-frame reuse | CONDITIONALLY SUPPORTED | Physical baseline and A1operator/frame controls; no new coupled learned-rigid claim |
| Local deformation | NOT SUPPORTED | Complete learned response contract fails despite partial learning |
| Global bending/twisting; self-contact | INCONCLUSIVE | Not tested |
| TransGI comparison readiness | INCONCLUSIVE | No aligned comparator experiment |

The machine-generated primary verdicts remain unchanged. Post hoc `review/review_summary.json` separates these scopes and does not replace any acceptance score. No candidate is adopted, no canonical K/C ownership changes, and no WL58/new hypothesis is authorised by this result.

### Reproducibility and artifacts

All final evidence is under [results/fnto_rule57/recovery_a2/final](../../results/fnto_rule57/recovery_a2/final/): original gate, full heldout/cost/physics/directional records, training histories,32fresh cost-worker measurements, stage receipts/logs, original server reports, terminal hashes,4verified final checkpoints and a report-only review. [Summary figure](../../results/fnto_rule57/recovery_a2/final/out/review/summary.png) and PDF/CSV counterparts are post hoc visualisation of existing measurements; they execute no model or oracle. Reviewer script `r57_final_summary.py` only reads completed JSON. Large complete actions/half second moments and coarse per-case prediction arrays remain on LabServer69 under `~/rest_wl57_recovery`, with their source/seeds/hash manifests retained.

Executable scientific source e809c15; A3execution09f5af3; pre-result cost/report correction cffe2b0; post hoc formatting source2bbeb47. Protocol files `rule57_recovery_a2.json` and `rule57_execution_a3.json` coexist with all original failed protocols. Physical model, model parameters/counts, data splits, seeds5701/5702,6000steps, precision, losses, checkpoint choice and thresholds stayed fixed. All historical53–56 negative results are unchanged. The authorised conditional batch is now closed with the bounded verdict above; no remaining server job is expected.


## Visual supplement — post hoc report only (2026-10-29)

기존 요약 PNG/PDF를 새 시각화 색인에서 바로 찾을 수 있도록 연결했다. 기존 결과나 판정은 변경하지 않았다. [시각화 산출물 색인](../evaluation/57_59_VISUAL_REVIEW.md).
