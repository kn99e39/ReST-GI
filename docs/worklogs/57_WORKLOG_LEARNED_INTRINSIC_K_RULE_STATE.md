# Worklog 57 — Learned Geometry-Conditioned Intrinsic K: Rule vs State

Status: **PREREGISTERED; no learned result or architecture adoption yet.**

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
