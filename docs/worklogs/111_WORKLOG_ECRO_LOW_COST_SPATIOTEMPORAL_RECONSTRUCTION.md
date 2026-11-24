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
