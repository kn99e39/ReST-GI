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

## Timing conditions — RTX 3090 power limit and clock reduction (owner, 2026-11-26)

- **Owner report.** The owner suspects that power caused the 3090 failure on 2026-11-25. After the reboot they set a
  power limit and a clock reduction on the 3090, and they expect it to compute noticeably slower.
- **What `nvidia-smi` shows during the timing stage** (21:13–21:15 KST):
  - Power limit 250 W; the default is 350 W. Draw is 200–223 W.
  - SM clock 795–1470 MHz on 1 s samples; the maximum is 2130 MHz. Memory clock 9501 MHz.
  - Active clock event reason: **SW Thermal Slowdown** (0x20), not SW Power Cap. The GPU is at 84–85 °C with the fan at
    100 %.
  - `nvidia-smi` does not report a locked clock range. The owner's exact clock setting is recorded only as the owner
    states it.
- **Reference point.** At Stage A (2026-11-24, before the failure) the environment snapshot read 1725 MHz at 71 °C.
- **Condition log.**
  - From 21:14:50 KST, a host `nvidia-smi` query samples the 3090 every second until the timing stage ends: SM and memory
    clocks, power draw and limit, temperature, fan, utilization and clock event reasons. The output is
    `evidence/timing_gpu_conditions.csv`.
  - A query creates no compute context, so the guard does not count it as a foreign process.
  - The timing blocks `static` (21:07–21:09) and `camera_Q2` (to 21:13) ran before this log started. They ran under the
    same owner settings.
- **Measured effect so far.** These are descriptive comparisons, not decisions. Stage A and the timing stage time
  slightly different code paths.

  | Quantity (V0) | Stage A, 2026-11-24 | Timing, 2026-11-26 (limited) |
  |---|---|---|
  | 4-spp ECRO frame | 11.45 ms | 10.84 ms (R0 E2E) |
  | 8-spp ECRO frame | 21.24 ms | 20.69 ms (R0 E2E) |
  | R3, 4 spp | 11.42 ms (static form, R3 only) | 21.67 ms (R3 E2E incl. frame, ≈ 10.8 ms above R0) |

  So far no large slowdown is visible. The CPU-capped aborted block (21:02–21:04, same GPU settings) agrees with the
  uncapped block within 1–3 %. The CPU cap therefore did not matter for these latencies either.
- **Effect on interpretation.**
  - Every arm in the timing stage is measured under the same limited GPU, interleaved within each block, and the routes
    use ratios. A clock reduction still need not scale all arms equally: GPU-bound work slows while host-bound work does
    not.
  - The frozen Stage A design choices (sub, m, futility) were made at full settings and stay as made.
  - Absolute latencies from this timing stage carry the label "3090 at a 250 W power limit with owner clock reduction;
    thermal slowdown active".

## Final report (chain completed 2026-11-26 21:22 KST, verdict fixed by the frozen mapping)

The chain at `a89cfed` (evidence code `0c56218` + execution fix 4 `d6a6525` + Amendment 1) finished every stage. The
guard logged `exit rc 0`. `analyze_r111.py` returned **NOT_SUPPORTED**.

### Implementation fact

- **Candidate CT-AK.** One compact network at m = 1.0, sub = 4 (Stage A design rule), 194,102 parameters, two training
  realizations (r0, r1).
  - Spatial S: per-component 5×5 softmax à-trous passes, dilations 1, 2, 4.
  - Temporal ST: S plus one raw-LOW/HIGH history. The history uses nearest reprojection and per-step validity tests,
    with up to 32 frames of accumulation. It is reset globally on any geometry or light revision.
- **Execution.** All arms run under the same execution contract. S/ST run as CUDA graphs and R3 as one CUDA graph. Guide
  cost is included in every E2E.
- **Stage E.** All synthetic contracts E1–E13 pass, including graph == eager.

### Measurement

**References.** All sealed-view references qualify: SE / R3 RMSE ratio 0.039–0.092 against a limit of 0.10.
- Q2 qualifies.
- Q5 qualifies.
- Q3 does not qualify, with either config: total change 4.28 % / 4.29 % against the 5 % rule. Under the protocol it is
  skipped and reported.
- Q4 does not qualify: per-step resolved fraction 0.34 against 0.50. A non-qualifying Q4 cannot support a continuous-edit
  claim.

**Routes at the primary anchor (R3, 4 spp).** Ratios are candidate / R3; the bracket is the 95 % interval.

| Regime | R3 E2E | Route A | Route B | Result |
|---|---|---|---|---|
| static_warm (sealed) | 21.67 ms | 1 spp: RMSE 0.486 [0.41, 0.55], time 0.228 | 6 spp: D 0.71 (lower95 0.67) | PASS, both realizations |
| static_warm_historical | 21.67 ms | 1 spp: 0.553 (r0) / 0.489 (r1), time 0.228 | PASS | PASS |
| cold_reset (S, no history) | 21.67 ms | best: 8 spp, RMSE 0.936, time 1.05; ROI fail | D −0.04 | FAIL |
| continuous_edit (Q4, non-qualifying) | 45.04 ms | 8 spp: 0.992, time 1.02 | D −0.06 | FAIL |
| camera_motion (Q2) | 46.10 ms | no point with upper95 ≤ 1.05 | D −1.72 | FAIL |

- **Static, the other anchors.** At anchor 16, Route A passes in both realizations; at anchor 32 it fails for r0. Route
  B passes at both.
- **Equal spp, sealed views.** Mean relative RMSE:

  | spp | R3 | S0 / S1 | ST0 / ST1 after 64 frames | SVGF after 64 frames | R0 |
  |---|---|---|---|---|---|
  | 1 | 0.0259 | 0.0280 / 0.0262 | 0.0074 / 0.0071 | 0.0153 | 0.472 |
  | 4 | 0.0151 | 0.0180 / 0.0164 | 0.0049 / 0.0047 | 0.0108 | — |

**Dynamic safety.**

| Rule | Result | Values |
|---|---|---|
| D1, no stale history after an edit | PASS | 6,144 frames, max rel L1 0.0 |
| D2, disocclusion | PASS | 48 of 22,306,656 disoccluded pixels retained (2.2e-6) |
| D3, no attenuation (Q2) | FAIL | ST / S temporal-change error 1.17 (r0) / 1.15 (r1), rule ≤ 1.05; slope 0.97 is in range |
| D4, reset regression | FAIL | see the attribution below |
| D5, continuous latency | PASS | |
| D6, camera-motion quality (Q2) | FAIL | see below |

- **D6 detail.** The ST / S RMSE ratio grows with spp, in both realizations:
  - primary 1.59 → 2.66 (1 → 8 spp);
  - silhouette 1.21 → 1.32;
  - sss_interior 2.80 → 6.18.
- **Absolute contract C.** It fails in every regime.
  - static_warm: silhouette bias +1.07 % against ≤ 1 %, gradient ratio 0.87 / 0.93.
  - cold_reset: silhouette bias +3.3 % (r0).
  - continuous_edit: silhouette −1.2 to −1.4 %, material boundary −1.05 % (r0).

**Timing.**
- **Conditions.** Measured 21:07–21:16 KST, 2026-11-26:
  - RTX 3090 at the owner's 250 W power limit and clock reduction;
  - no CPU cap; load1 2–3.
  - During busy 1 s samples, the median SM clock was 833 MHz (375–1455), the GPU sat at 84–85 °C, and SW Thermal
    Slowdown was active in 58 of 60 busy samples.
- **Static frame.** R0 / R3 / S0 / ST0 at 4 spp: 10.84 / 21.67 / 12.72 / 12.83 ms.
- **Camera form.** Candidate: guide pack 2.71 + graph 2.26 ms. R3: 8×8 guides 5.78 + graph 12.01 ms.
- The capped aborted block and the uncapped block agree within 1–3 %.

### Observation

- **D4 is a hardware-fault artifact.** Of the 1,440 forced-reset checks (720 static items × 2 realizations), 1,438 are
  exactly 0.0. The other two are both 64.0, in the same item, `H5_spp24_r4`.
  - That item is the last record written before the RTX 3090 failed: 17:57:44 KST on 2026-11-25. The process stalled
    right after.
  - Frames 1–64 of that item have metrics in the range of the other five replicates. Only frame 65, the forced reset,
    is affected.
  - **Post hoc recheck** (labelled, not evidence for the verdict). The item was recomputed alone on the rebooted 3090 in
    a separate root. All 308 metric values are identical to the original, and both reset checks are 0.0. The file is
    `evaluation/111/posthoc/recheck_H5_spp24_r4.json` (sha256 0734884…).
  - D4 is therefore attributed to the GPU fault, not to the candidate. The evidence file stays as written, and the
    verdict stays as computed.
  - The recheck also shows that resumed and sharded execution reproduces a pre-failure item bitwise.
- **The verdict does not depend on D4.** D3 and D6 fail on their own, so NOT_SUPPORTED holds either way.
- **History under camera motion** (Q2, frame 48; `history_maps_Q2_4spp.png`).
  - The history is accepted almost everywhere; only thin silhouette rims are rejected. The SSS sphere carries the full
    32-frame age.
  - The error grows with spp and is largest in sss_interior.
- **The static frontier is strong.** With a warm history, ST at 1 spp is about half of R3's RMSE at 4 spp, at 23 % of its
  E2E latency. ST after 64 frames is 3–4× more accurate than SVGF's 64-frame history.

### Interpretation

- **Static.** With a warm history, CT-AK moves the static quality–performance frontier well past R3. Both routes pass at
  the primary anchor, for both realizations.
- **Without history** (cold_reset), the spatial network alone does not match R3 at equal or lower cost.
- **Camera motion.** The history mechanism, as specified, degrades quality: ST is worse than S, and 2.7–2.9× R3. The
  dynamic-safety contract exists to catch exactly this.
- **Likely cause** (an inference, not isolated by an experiment). The validity tests compare only consecutive frames.
  They therefore never bound the drift a history accumulates over up to 32 steps:
  - the angular drift of view-dependent SSS radiance (0.995 per step accepts the whole orbit);
  - repeated nearest-pixel resampling.
  Either effect would be hidden at low spp by noise and would dominate as noise falls, which matches the rising D6
  ratios.

### Unresolved

- **Q2 attribution.** Which of the two mechanisms causes the Q2 degradation was not separated: stale view-dependent SSS,
  or nearest-resampling drift on Lambert surfaces.
  - The primary ROI is also 1.6–2.8× worse, so the Lambert part is not zero.
  - This would need a controlled run with the history disabled per material, or with bilinear reprojection. Neither is
    authorized here.
- **Continuous edits.** Q3 and Q4 did not qualify, so continuous-edit quality is unmeasured as a claim.
- **Timing comparability.** Timing ran on a power-limited, thermally throttled GPU. Absolute latencies are not comparable
  to Stage A or WL108. GPU-bound and host-bound arms may scale differently under the clock reduction.
- **ReST-GI performance observations.** The observations in the section "Observations relevant to ReST-GI performance
  beyond this batch" stay open as recorded.

### Architecture verdict

**NOT_SUPPORTED** (frozen mapping: a D rule fails).
- D3 and D6 fail on camera motion for both realizations. D4 fails only through the hardware-fault item.
- Descriptively, the static warm-history frontier gain is real (Routes A and B pass) but does not carry into the dynamic
  regimes.
- The candidate is not adopted, and no successor is authorized.

### Reproducibility

- **Code and protocol.** Code `0c56218`, plus execution fix 4 `d6a6525` and Amendment 1 `a89cfed`. Protocol sha256
  e160f91c…
- **Evidence.** Under `~/rest_sss101a/results/ecro_recon111/evidence` on LabServer70, including 13 GB of ladder data.
  The JSON and logs are mirrored to the gitignored `results/ecro_recon111/evidence/`.
- **Analysis.** `analysis.json` sha256 dcf392d5…
- **Reviewer package.** `evaluation/111/`, made by `package_r111.py` at `5bc399d`. `manifest111.json` hashes 801
  evidence JSON files.
- **GPU condition log.** `evidence/timing_gpu_conditions.csv`.
