# Worklog 110 — ECRO live demonstrator: frozen checkpoint in a continuous frame loop (2026-11-23 KST)

## Authorization, classification and freeze

- **Brief.** On 2026-11-23 the user handed over "ReST-GI Live Demonstrator — Frozen Checkpoint Runtime Viewer" with "일단 이 작업부터 수행하고 커밋해주라". The task class is demo engineering / bounded integration on `ecro_lr3_adapter`. It asks whether the existing ECRO checkpoint can render GI continuously under scripted rigid motion, and what presented frame rate the complete displayed system actually achieves.
- **Classification.** This is a demonstrator, not a research experiment. No hypothesis, gate or architecture verdict of the research line is opened or changed. The living documents are not updated: the batch changes no durable research state.
- **Number.** The batch takes 110. This branch ends at 105B, but `origin/Denoise_Reconstructor` already uses 106–109A, so 110 avoids a cross-branch collision. The gap is intentional.
- **Repository / branch.** `kn99e39/ReST-GI`, `ecro_lr3_adapter` at `62cfad7` (= origin). No new branch or worktree. Unrelated pending work: the WL104K-B resume runs on the local PC, which is currently powered off. It is not touched here.
- **Machine.** LabServer69 is the WL101 evidence machine: RTX 3090 24 GB, driver 550.54.15, CentOS 8.4. It runs WL101 arm B's stack from `~/rest_dyn101`: Mitsuba 3.8.0 / Dr.Jit 1.3.1 (`external/mi_3.8.0`), torch 2.3.1+cu121 and Triton 2.3.1 from `~/NeuralShaderDemo/external/8dna26/.venv`, plus `libatomic` from `external/mi39_libs` (`experiments/ecro_live110/env110.sh`).
  - Checkpoints and `configs.json` are read through a gitignored symlink, `results/ecro_material100 → ~/rest_dyn101/results/ecro_material100`.
  - Checkpoint sha256: SSS `2d9cb580…`, near `b3550787…`. Both are equal to the WL101 record.
  - The frozen sources in this checkout equal the WL101 workspace copies byte for byte apart from CRLF line endings: `rt101`, `seq101`, `common101`, `pt86`, `ecro86`, `model86`, `scene86`, `pt100`, `common100`, `closure83_dr` and `dyn101.json`.
- **Presentation path.** The server is headless: no `DISPLAY`, no X socket, no `ffmpeg`.
  - The supported path is an X11 window under **Xvfb**, drawn with **Tk**. Tk is from the standard library and Pillow's `ImageTk` writes the photo block; no large GUI framework is added.
  - Screen evidence is read straight from the X server's memory-mapped framebuffer (`Xvfb -fbdir`, XWD) by a separate recorder process. The capture shows what the display holds, overlay included.
  - Limitation: Xvfb has no vsync or scan-out. "Presented" here means the X server has processed the image upload and redraw, confirmed by a client round trip. The run says nothing about a physical monitor's refresh.

### What runs per frame (nothing precomputed)

`anim110.pose(t)` → `rt.set_geom` → `rt.update_state` → `rt.source` → `rt.respond` → DLPack hand-off → `rt.frame` → compose `(e1 + e2 + hl) / spp` → optional WL101B SVGF → display transform → uint8 readback → Tk → X round trip.

- `rt.set_geom` covers the BVH and incremental tables.
- `rt.update_state` re-traces relations selectively, using the frozen dirty rule.
- `rt.respond` runs the ECRO generator and the linear application, giving table B.
- `rt.frame` evaluates physical LOW plus the learned-HIGH lookup with replayed kernels.
- The optional SVGF step resets history on every geometry revision.
- The display transform is the WL100 exposure key with `(x / (1 + x))^(1/2.2)`.

All rt101 / model86 / svgf code is imported unchanged. New code lives only in `experiments/ecro_live110/`: `live110.py` (engine, loop, presenter, instrumentation), `anim110.py`, `record110.py`, `verify110.py`, `run110.sh`, `analyze110.py`, `env110.sh`.

### Animation (fixed before evidence)

- **Primary.** SSS, both boxes, on the WL101 V_SSS_ANIM segment (test cfg 2 → cfg 1, centre and yaw linear): `u(t) = 0.5 − 0.5 cos(2πt / 6 s)`.
  - Fixed camera; fixed light (test cfg 0 light 0, the WL101 rule).
  - Frame clock `t = k / 30` makes the frame→geometry map deterministic.
  - Pacing is a 30 FPS cap. A late frame is counted as missed and the schedule re-anchors without catch-up or skipping. The animation therefore lags wall time instead of speeding up or slowing down silently, and the lag is reported.
- **Secondary.** Near-mirror, cfg 0 → cfg 1 (consecutive V_NEAR_TOUR states).
- **Range.** Both WL100 samplers draw box centres and yaw from independent uniform intervals, which form a convex box. Every interpolated pose is therefore inside the training-geometry support.
  - Only the 17 V_SSS_ANIM poses (k / 16) and the tour end points were used by WL101.
  - Continuous intermediate poses were never quality-validated against physical references.
  - Nothing displayed is evidence of out-of-distribution generalisation.

### Frozen evidence plan (`run110.sh`)

- **Runs.** 30 warm-up + 600 measured frames each, at 512² and 1 spp:
  - SSS raw: 30 FPS cap (r1, r2), uncapped (r1, r2), headless uncapped (r1, r2) and a 60 FPS cap;
  - SSS raw with the screen recorder;
  - SSS SVGF: 30 FPS cap with recorder, and uncapped;
  - near raw: 30 FPS cap with recorder.
- **Baseline.** The same-day WL101 `run_b101.py V_SSS_ANIM B_sel` runs before and after (r1, r2). Historical WL101 timing is preserved separately.
- **Contention control.** Every step waits until foreign compute processes stay below 3 % SM for 10 s, and logs `nvidia-smi pmon` beside the run. Another session's TransGI job shares this GPU at about 7–9 % SM.
- **Verification.** `verify110.py`, checks V1–V9 (see its docstring).
- **Development runs (scratch, not evidence).**
  - Smoke loops gave about 29.6 presented FPS at the 30 cap.
  - Presentation through Tk PPM data cost about 7.7 ms. `ImageTk.paste` costs about 2.2 ms, so the presenter was switched to it.
  - A full development `verify110` pass was clean, except that the V7 load counter counted process-wide. It was corrected to count loads during the loop only.

## Evidence (append below)
