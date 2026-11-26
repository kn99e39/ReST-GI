# Worklog — ECRO Low-cost Spatiotemporal Reconstruction: Quality–Performance Frontier (2026-11-24)

> Append-only execution record. Batch class: **regular WL111**. Owner brief "ECRO Low-cost Spatiotemporal Reconstruction:
> Quality–Performance Frontier Improvement", 2026-11-24. The owner assigned the number ("이 작업은 111로 명명해"), and
> the external-benchmark batch was renumbered WL110 on the owner's instruction. Branch `Denoise_Reconstructor` (no new
> branch or worktree). Independent Denoising Track batch: it does not depend on WL109B, and C1 is not used.

## Authorization and scope

- **Question.** Can a compact LOW/HIGH spatial reconstructor combined with validated temporal history replace the
  expensive R3 spatial path with a better end-to-end quality–performance trade-off? And does it stay useful when
  dynamic transport changes invalidate the history?
- **Preserved.** The following are untouched:
  - the ECRO checkpoint and learned table generation and lookup;
  - physical LOW = E1 + E2, HIGH = HL, TOTAL = LOW + HIGH;
  - native R0 sampling and the geometry-update lifecycle;
  - the R3 checkpoint `d51b6919…` and its code;
  - the WL101B SVGF code and settings;
  - all WL101B and WL105–WL109B results and verdicts.
- **Changed, experiment-locally only.**
  - One candidate reconstructor and its temporal history.
  - Online guide programs.
  - Camera-input copies of the frame and guide programs. These run the same statements and are tested bitwise against
    the original programs.
  - Training and evaluation instrumentation.
- **Not authorised.**
  - ECRO or C1 changes, or transport retraining.
  - A second denoiser architecture, sweeps, or selective GI history reuse.
  - Future frames, or references at runtime.
  - Post-hoc threshold changes, a production merge, or external-framework claims.

## Host (owner directives during preparation)

1. **Main RTX 5080.** Preparation started on Main, where triton-windows was installed only in an isolated
   `C:\Projects\recon111\pylib`. Another session's WL113 then took Main's GPU, so the only GPU process this batch had
   started there, a non-evidence smoke, was stopped at once. It wrote nothing.
2. **LabServer70.** The owner then directed LabServer70 and later "3090 사용해". WL109B was placed in waiting by the
   owner.
   - **Execution host:** the LabServer70 RTX 3090, driver 550.54.15. This is the GPU model of WL108's R3 timing.
   - **Container:** WL109B Amendment 3 container (frozen Mitsuba 3.8.0 / Dr.Jit 1.3.1 / Torch 2.3.1 / Triton 2.3.1),
     limited to 4 CPUs.
   - **Guard:** WL109B guard `guard109c.py`, unchanged, with `--gpu-uuid` of the 3090.
   - **Frozen inputs:** WL108 inputs tree and WL107 capture, already on LabServer70, linked read-only.
   - **Copied in:** archived WL101B static frames (1.2 GB, for the SVGF reproduction) and the WL110 full-depth physical
     references (63 MB, descriptive T_PHYSICAL).
   - **Code:** only `experiments/ecro_recon111` is deployed. No other experiment directory on the host is changed.
   - **Shared CPU:** the host CPU is shared (load average about 14 of 16). Timing blocks record the load, require
     load1 ≤ 16, and interleave the arms.

## Design and protocol (frozen before any evidence)

The protocol is [`experiments/ecro_recon111/protocol/recon111.json`](../../experiments/ecro_recon111/protocol/recon111.json).
Its commit is recorded below, before Stage A evidence.

- **Stage A.**
  - Main recovery checks:
    - G0a–c: identity of the frozen renderer, the learned table and the archived replay.
    - G0d: R3 replayed on the 144 archived WL105 inputs.
    - G0e: R3 quality against the WL107 references.
    - G0f: WL101B SVGF replayed on its archived static sequences, without repair.
  - Equivalence checks: the camera-input frame program against the original, bitwise; the online guide program against
    the archived WL105 aux; the centre guide against the archived WL101B guide.
  - R3 cost attribution by CUDA events:
    - stages: features, two U-Nets, two fused kernel applications;
    - diagnostics: fp16 and unfused variants;
    - analytic MACs, conv-output bytes and logits bytes, plus measured copy bandwidth.
  - Guide costs (8×8, 4×4 and 2×2 sub-pixel, centre ray, visibility), SVGF cost, and a candidate cost grid.
  - **Predeclared design rule (cost only, untrained weights):**
    - sub = 8 if the 8×8 guide costs ≤ 1.0 ms, else 4;
    - width m = the largest of {2, 1.5, 1} whose camera-motion subsystem is ≤ 3.0 ms;
    - cost-futility stop if the m = 1 subsystem is ≥ 0.70 × R3's.
- **Candidate CT-AK.** One design:
  - a compact shared-trunk U-Net, fp16, with its bulk at ½ to ⅛ resolution;
  - it predicts per-component 5×5 softmax kernels for three dilated passes (1, 2, 4), giving 29 px support;
  - LOW and HIGH are filtered separately (normalized, nonnegative) and then summed.
- **Temporal history.**
  - Raw LOW and HIGH are accumulated before spatial filtering, with identical weights.
  - The inverse effective sample count (ESS) is tracked exactly, under a 32-frame cap.
  - **Global reset** on any geometry or light edit.
  - Under camera motion a history sample is accepted only if all of these hold:
    - nearest-pixel reprojection of the centre-ray point;
    - same shape id;
    - normal and position checks;
    - pure sub-pixel coverage now and before;
    - previous-eye visibility;
    - a view gate for the SSS sphere (every other material in the scene is Lambert).
- **Training.**
  - Data: WL105 cameras and 4096-spp targets, plus a new spp ladder (1–2048, 2 replicates).
  - Per-pixel age maps reproduce accumulated inputs.
  - Loss: component and full-frame relative MSE.
  - Schedule: 20k iterations, fp16 autocast.
  - Two realizations, final weights only. A route must pass for both.
  - The extra data relative to R3 is declared.
- **Evaluation.**
  - Views: six sealed views K1–K6 (geometry-only acceptance rule), plus the historical V0/H1–H5 reported as historical
    workload.
  - Static protocol: a 65-frame stream with a forced reset; spp grid 1–32.
  - Sequences:
    - Q2: camera orbit;
    - Q3: abrupt box edit;
    - Q4: continuous box motion;
    - Q5: light change.
  - References: frozen-ECRO references qualified against R3's error; T_PHYSICAL is descriptive.
- **Decision.**
  - Routes, at the primary anchor R3 at 4 spp:
    - Route A: quality ratio upper95 ≤ 1.05 at ≥ 15 % lower E2E;
    - Route B: ≥ 10 % RMSE reduction at matched E2E, with lower95 > 0.
  - The routes are applied to static-warm, cold/reset and continuous-edit regimes.
  - Absolute contract C: |bias| ≤ 1 %, silhouette gradient ratio in [0.8, 1.2], boundary noninferiority.
  - Dynamic safety D1–D6.
  - Verdict mapping as in the brief.

## Preparation record (no evidence)

- CPU unit tests pass locally: seeds, ROIs equal to WL105, constant preservation, normalization, temporal accumulation
  and ESS, reset, causality, reprojection, and the mixed / id / hidden / view gates.
- Corrections made before any evidence:
  - The seed-family bases had been placed 100,000 apart against a 1,000,000 family width. They were moved 1,000,000
    apart.
  - One view-gate test asserted the wrong thing: distant plane points legitimately pass the gate. It was rewritten to
    "no sphere pixel with cos < 0.995 is accepted".
- **LabServer70 code-path smokes.** All used the non-evidence seed offset. Their numbers are not interpreted as evidence;
  they are recorded because they changed the implementation before freezing.
  - **Smoke 1.**
    - Identity checks were exact: the learned table hash is equal (`d7e40954…`); the camera-input frame, guide and centre
      programs are bitwise equal to the originals for V0 and H1; R3 replay is EXACT (max relative L1 0.0); SVGF replay
      is 0.0.
    - Every unit test passes, CPU and GPU (fused passes and temporal step equal their references).
    - Triton kernels defined inside a function did not compile (`tl` not in globals). They were moved to
      `kern_r111.py`.
  - **Chain structure (owner instruction).** The first chain gave every stage its own guard, so the batch waited 120 s for
    its own GPU between stages. It was rebuilt with one guard at chain entry and every stage run back to back in one
    container (`lab70/run_r111.sh` + `inner_r111.sh`).
  - **Stage A smoke: cost-futility condition met.**
    - Candidate camera-form subsystem 15.0–17.0 ms against R3 15.6 ms.
    - R3 itself on the 3090 took 11.3 ms (WL108: 10.87 ms). Two U-Nets account for 81 % of it, kernel application for
      13 %. fp16 autocast was slower.
    - Breakdown of the candidate: launch-bound. The 4×4 and 2×2 guides took the same time; a single elementwise kernel
      took 1.6 ms; each program call made 13 separate camera uploads. The host is CPU-shared (load ≈ 14/16) and the
      container is capped at 4 CPUs.
    - With the owner's approval ("수정 진행하도록 해") the execution was changed. No rule, threshold or mechanism
      changed:
      - guides, centre and visibility now run as one Dr.Jit program;
      - S and ST run as CUDA graphs, one per history mode;
      - R3 also runs as a CUDA graph under the common execution contract. Its output equals its eager forward bitwise;
        eager R3 is reported only.
    - The next smoke measured the candidate at 9.0 ms against R3 at 14.7 ms (ratio 0.61, below the 0.70 futility
      limit).
  - **Stale-camera replay defect, found by the sealed-view acceptance in the smoke.**
    - Passing the camera into a frozen Dr.Jit program as one array (plain or as a DLPack-shared torch buffer) made
      Dr.Jit 1.3.1 replay the recorded camera from the second or third distinct camera on. A toy frozen function did
      not show this; the real ray-tracing programs did.
    - Passing 16 opaque scalars gives correct values for every camera, matching the earlier implementation.
    - Scalars are now the only mode. The equivalence check now cycles the camera V0 → H1 → H2 → H3 → V0 and compares
      every replay with the original programs (frame, guides, centre, visibility).
    - The earlier two-camera equivalence check could not detect this defect. No evidence had been produced.
  - **Last smoke, after the scalar-camera fix.**
    - Every stage, including analysis and reviewer figures, runs end to end. The figures run inside the container,
      because host PIL needs a newer glibc.
    - The camera-cycling equivalence check passes. R3 executed as a CUDA graph equals its eager form bitwise (10.9 ms).
    - The cost-futility ratio was 0.701 (candidate 10.2 ms against R3 14.6 ms). The previous smoke gave 0.61.
    - The spread comes from host-CPU launch latency.
    - **Owner decision ("일단 1로 진행해봐"):** freeze now and let the frozen Stage A decide under the unchanged rule.
      The host-load dependence is reported with the result.
  - Analysis fix: the bootstrap CI of an empty or all-NaN sample now returns NaN instead of raising.

## Freeze

The protocol, code and this section are committed together before any evidence. The commit is recorded in the next
section.


Freeze commit: `24211b7` (protocol sha256 `e160f91c…`).

## Execution fix 1 — guard self-preemption at a stage transition (2026-11-24, before any evidence)

- **What happened.** The first evidence launch (`run_r111.sh evidence 24211b7`) passed the tests stage.
  - Exactly when the Stage A process started, the unchanged WL109B guard reported a "foreign" compute PID (24700) on
    the RTX 3090 and killed the container.
  - The PID did not exist a moment later. No other job was on the GPU; WL109B's guard was waiting at its load gate.
  - Only the tests stage had run. Stage A had produced nothing.
- **Cause.** The WL111 chain runs several processes one after another in one container. The exiting tests process was
  still listed by `nvidia-smi` after it had left the container's process table (`docker top`), so the guard counted it
  as foreign. WL109B ran one process per container and never met this.
- **Fix.** `guard_r111.py` is a copy of `guard109c.py` with one change: during a run, a PID counts as foreign only if it
  is seen in two consecutive polls (2 s apart) and still exists in `/proc`. The start condition (120 s without any
  compute process on the 3090) and every other behaviour are unchanged. Foreign work is still detected within about
  4 s.
- **Scope.** This is a non-scientific execution fix. No protocol rule, threshold, design, seed or analysis changed. The
  chain was relaunched in the same output directory: the completed tests stage is skipped, and Stage A starts from
  scratch.

## Diagnostic and execution fix 2 — fixed per-frame overhead (2026-11-24, owner request; before any candidate evidence)

The owner asked to profile the fixed overhead on LabServer70's RTX 3080 Ti (GPU1), in the same container stack, while
the evidence chain ran on the RTX 3090.

- **Method.** The diagnostics are `dbg_overhead_r111.py` and `dbg_graph_r111.py`; they are not evidence.
  - GPU time was isolated by queueing a device sleep before each measured region, so CPU enqueue latency is excluded.
  - CUPTI (`torch.profiler`) crashed with an illegal address alongside Dr.Jit/OptiX, so it was not used.
- **Findings (3080 Ti, container limited to 4 CPUs, host load1 about 14).**
  - **Guide pack (sub = 4).** GPU 1.41 ms, CPU enqueue 3.2 ms. The roughly 2.3 ms floor seen in Stage A is host-side
    work:
    - 16 opaque camera scalars, 0.6 ms;
    - Dr.Jit replay preparation;
    - about a dozen torch post-processing ops at about 13 µs each.
    - Without the 4-CPU quota (16 CPUs) it was slightly slower, so cgroup throttling is not the cause; the shared host
      CPU is.
  - **R3's 8×8 coverage guides.** GPU 3.2 ms. This is genuine ray-tracing cost (16.8 M rays).
  - **ECRO frame.** 1 spp: GPU 2.38 ms, CPU 1.85 ms. 4 spp: GPU 10.4 ms.
  - **Candidate graph: 5.4 ms of device time, of which 3.3 ms was one layout conversion.** The network ran in
    channels_last, so cuDNN returned the 150-channel fp16 logits in channels_last. The fused passes then made an NCHW
    copy of them with `.contiguous()`.
    - By comparison, the network took 1.78 ms and the three passes on NCHW logits 0.28 ms.
    - Reading channels_last logits in place was correct but slower (5.2 ms), because the reads are uncoalesced.
- **Fix 2 (owner-approved option 1; option 2, moving the pack post-processing into the graph, is on hold).**
  - At inference (batch 1, no gradient), the 1×1 head is computed as one GEMM on the channels_last feature view, which
    yields NCHW logits directly.
  - The fused passes accept NCHW and channels_last logits through their strides.
  - Training keeps the cuDNN conv path.
  - Equivalence checks:
    - GEMM head vs conv head: 8.6e-8 relative to peak in fp32 with TF32 off; 3.5e-4 in fp16 autocast, within fp16
      precision;
    - channels_last vs NCHW passes: bitwise equal;
    - all tests pass.
  - Candidate graph device time: 5.40 → 1.66 ms (3080 Ti).
- **Scope.**
  - Execution only: same mechanism, same weights, same function up to fp16 rounding. No rule, threshold, design rule or
    analysis changed.
  - Stage A was measured with the former layout. Its record stays as measured. Its decisions (sub = 4, m = 1.0, no
    cost futility) are unchanged by a faster candidate. With the fix the camera-form subsystem is still above the
    3.0 ms width threshold, so m stays 1.0.
  - Fix 2 is in place before the evidence stages that use the candidate's fused path (Stage E, static, sequences,
    timing). The chain was in its data stage, which does not import these modules.

## Observations relevant to ReST-GI performance beyond this batch (recorded at the owner's request; not decisions)

These come from the diagnostics above. They will be carried into the final report's unresolved questions.

1. **At low spp the ECRO frame is near the CPU/GPU crossover.** On the 3080 Ti, 1 spp costs 2.38 ms of GPU time and
   1.85 ms of host enqueue time; 4 spp costs 10.4 ms against 1.9 ms. Low-spp frame times therefore depend on the host
   CPU state. Earlier low-spp latencies, such as WL101 and WL110's 0.81 ms at 1 spp on the RTX 5080, are comparable only
   under the same exclusive, uncontended host conditions.
2. **The canonical frame program does not serve a moving camera efficiently.** `rt101.RT.frame` records one frozen
   program per (res, spp) with the camera baked in as constants. A moving camera needs re-recording per camera, or else
   it renders the recorded camera. Every ReST-GI timing so far used a fixed camera.
   - Passing the camera as program input works on Dr.Jit 1.3.1 only as opaque scalars. Array inputs replayed stale
     values (above).
   - Scalar inputs cost about 0.3–0.6 ms of host time per call on this host.
   - Moving-camera real-time budgets for ECRO must include this cost.
3. **Sub-pixel coverage guides are a real per-change cost of R3-class reconstruction.** They cost 3.2–4.4 ms of GPU
   time at 8×8 (16.8 M rays) whenever the camera or geometry changes; static views can cache them.
4. **Possible effect on other batches.** Any other code on the Dr.Jit 1.3.1 stack that passes per-call-varying arrays
   into frozen programs may be affected by the stale-replay behaviour. Other batches' code was not audited here.

## Execution fix 3 — Stage E caught a graph-capture defect (2026-11-25; before any static, sequence or timing evidence)

- **Chain run.** The chain at `7a5993e` completed these stages:
  - **Data.** Sealed views K1–K6 and the 48-view ladder (16 min).
  - **Training, both realizations.** 194,102 parameters each; 1,028 s and 979 s. Validation was logged only; relative
    RMSE on the validation views:

    | Realization | 1 spp, age 1 | 1 spp, age 16 | 4 spp, age 1 | 4 spp, age 16 |
    |---|---|---|---|---|
    | r0 | 2.45 % | 1.24 % | 1.70 % | 0.96 % |
    | r1 | 2.60 % | 1.24 % | 1.76 % | 0.93 % |
- **Stage E then stopped the chain, as designed.** Every contract passed for both realizations except E13 "graph equals
  eager". That contract measured relative L1 7.0e-6 to 9.5e-6 against a 1e-6 limit.
- **Passing contracts:**
  - E1 constant fields (≤ 1.05e-6);
  - E2 composition, E3 HDR;
  - E4 reset = S (0.0), E5 causality;
  - E6 real reprojection acceptance (99.3 %);
  - E7 disocclusion retention (0 of 7,553 disoccluded pixels);
  - E8 mixed coverage, E9 id change and E10 view gate, never accepted;
  - E11 box-edit negative control: 72,252 id/normal-matched floor pixels, 11,684 of them with a resolved radiance
    change (mean 10.9 %); every pixel reset;
  - E12 light edit: every pixel reset;
  - E13 fused = reference (real frames) and fused temporal = reference.
- **Cause.** `GraphCandidate` captured its CUDA graphs outside `torch.no_grad()`. The captured network therefore took
  the training branch: the cuDNN 1×1 head with channels_last logits. That is also the slow layout that execution fix 2
  removed. The eager path takes the inference GEMM head. Both compute the same function, but the fp16 rounding differs.
- **Fix.** The graphs are captured under `torch.no_grad()`.
- **Record kept.** The failed Stage E record is kept as `synth_failed_fix3/` and `synth_failed_fix3.log`.
- **Relaunch.** The chain resumes at Stage E. Data and the trained weights are unchanged and reused.
- **Scope.** This is an ordinary implementation defect, fixed under the protocol's Stage E stop rule before any final
  evidence. No rule, threshold, design or weight changed.
- **Stage A.** Stage A timed the candidate graph with this defect. Its record stays as measured, and its decisions are
  unaffected (the candidate only becomes cheaper).

## Host event — RTX 3090 failure during the static stage (2026-11-25 to 2026-11-26)

- **What happened.** References finished at 2026-11-25 17:14 KST; they took 5 h 40 min. The static stage then ran normally.
  - Its last output was `static/H5_spp24_r4.json` at 17:57:44 KST.
  - After that the chain wrote nothing more for about 24 h. Normally an item takes 5–10 s.
  - The owner found that the RTX 3090 had failed. They rebooted LabServer70 at 2026-11-26 17:53 KST.
  - The reboot killed the container. The launcher logged `error waiting for container: unexpected EOF`.
  - The guard logged no exit event, because it died with the host.
- **Checks after the reboot.**
  - Every saved static JSON and image archive loads, and no partial `.tmp` file remains.
  - The 3090 reports normally: P8, 36 °C.
  - The stalled log is kept as `eval_static_gpu_failure_20261125.log`.
- **Restarts.**
  - The first relaunch at 18:52 waited in the guard: another process appeared on the 3090. The owner then stopped it.
  - The chain resumed at 19:54, after WL109B completed (19:30). The static stage continued from `H5_spp24_r5`.
  - This resumed single-process run is logged in `eval_static_resume_20261126.log`.
- **Effect on results.** Every static item is fully determined by its seeds. Each arm is reset per item, and items are
  written atomically. Resuming from the saved items therefore reproduces the uninterrupted run. No rule, seed or design
  changed.
- **Effect on timing.** Timing evidence is measured after this event, on the same physical 3090. Every timing block
  records the load average. A latency anomaly would be reported and not corrected.

## Execution fix 4 — static and sequence items split over three processes (2026-11-26, owner request)

- **Why.** The owner asked for more efficient GPU use. The single static process kept the 3090 busy only about half the
  time:
  - utilization swung between 14 % and 99 % on a 1 s sample;
  - it used 5 GB of 24 GB;
  - about 1.8 CPUs went to per-frame host work and CPU metrics.
- **Change** (`d6a6525`).
  - `eval_r111.py` reads `RECON111_SHARD=i/n`. In the static and sequence stages, a process keeps only the work items whose
    running index is i mod n.
  - `inner_r111.sh` runs those two stages as three such processes in the same container. Each writes its own log
    (`eval_<stage>_shard<i>.log`) and admission record (`admission_<stage>_shard<i>of3.json`).
  - An unsharded completion pass follows. It finds every item already written, or computes any item a shard left out.
  - The references stage and the timing stage refuse to run sharded. Timing keeps exclusive, single-process use of the
    GPU, unchanged.
  - The container stays capped at 4 CPUs.
- **Effect on results.**
  - Items are independent: arms are reset per item and seeds depend only on (view or sequence, spp, replicate, frame).
    The split therefore changes no evaluated quantity.
  - What can differ is GPU nondeterminism between processes. One example is library algorithm choices, at fp16 rounding
    level. Such differences are of the same kind as those between two runs of one process.
  - This is a non-scientific execution change. No rule, threshold, seed, design or weight changed.
- **Restart.** The chain stopped at 414 of 720 static items and relaunched at `d6a6525`.
- **CPU cap during the sequence stage** (owner approval, 2026-11-26 20:43 KST).
  - With three shards, the sequence stage was CPU-bound at the 4-CPU cap: every CFS period was throttled, and 3090
    utilization was about 25 % on 1 s samples.
  - The running container's cap was raised to 8 CPUs with `docker update`, for the sequence stage only.
  - A host watcher (`results/ecro_recon111/cpu_restore_r111.sh`) restores 4 CPUs when the chain writes `DONE eval_seq`,
    `STOP` or `RUNNING eval_timing`. Timing is therefore measured under the protocol's 4-CPU cap.
  - The first two watchers matched words in my own `chain.state` notes and restored 4 CPUs at once. The third matches
    only the chain's stage lines.
  - The protocol's 4-CPU cap was stated for timing conditions. Sequence quality evidence does not depend on the CPU
    budget.

## WL111 Amendment 1 — no container CPU cap, including timing (owner decision, 2026-11-26 21:05 KST)

- **Decision.** The owner directed that the batch set no CPU cap of its own, because host conditions have changed since
  the cap was chosen.
- **Origin of the cap.**
  - The 4-CPU cap was not an owner number. I carried it over from WL109B Amendment 5, where the owner asked for "a few"
    LabServer70 threads while other work held the host CPU. I chose 4 there.
  - I then froze it into this protocol's `host.cpu` text as a timing condition.
  - The 8-CPU value used during the sequence stage was also my own number.
- **Change.**
  - `ctr_r111.sh` starts the container without `--cpus` unless `WL111_CPUS` is set.
  - Every other timing condition is unchanged:
    - exclusive GPU through the guard;
    - a single process;
    - the load average recorded per block, with load1 ≤ 16;
    - arms interleaved in each block.
- **State when applied.** The sequence stage had finished at 21:02 KST: 72 items, and `DONE eval_seq` was written. The
  timing stage had started under the 4-CPU cap but had written no timing record. The chain was stopped and relaunched,
  so the timing stage runs from the start without a cap. No timing measurement exists under both conditions.
- **Scope.** This changes an execution condition of the timing measurement: Absolute latencies are labelled as measured
  on a shared host without a CPU cap. No rule, threshold, seed, design, weight or quality evidence changed. The
  protocol file stays as frozen; this entry records the deviation from its `host.cpu` text.
