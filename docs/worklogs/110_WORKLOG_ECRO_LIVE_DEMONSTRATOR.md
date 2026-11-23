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

### Classification note (added after the freeze)

- After the freeze, this session first saw the batch-numbering policy in `AGENTS.md` (human decision, 2026-11-13). Under that policy the research owner decides the batch class.
- This demonstrator opens no hypothesis and introduces no mechanism. By that rule it is not a regular research batch: at most it is a dependent follow-up of WL101's runtime, or non-research demo engineering.
- The number 110 was already assigned and pushed (`8db9448`). It is kept so that pushed work is not renumbered. The owner may reclassify.

### Execution record (2026-11-23, LabServer69)

- **Implementation and freeze.** Commit `8db9448`, pushed before any evidence run. All runs record this commit, a clean tracked worktree, and their own file hashes (`summary.json` → `provenance`).
- **First chain** (`results/ecro_live110/evidence/`, 14:21–14:44). The full plan completed.
  - The idle gate passed while another session's TransGI b21 evaluation (`eval_transgi.py`, staircase) was resident. It ran at 1–3 % SM with spikes to 28 %, and its post-processing was also resident.
  - The pmon logs show foreign SM on 7 of the first 11 live steps. Two more steps had other foreign processes.
  - This set is **contended**. It is kept and disclosed (`runs110.json` → `contended_first_set`) and is not used for the headline numbers. Its symptoms were compute p95 of 19.6–53 ms, 12–36 missed deadlines at the 30 cap, and 33.5–38.1 uncapped FPS.
- **Clean chain** (`results/ecro_live110/evidence_clean/`, 14:44–14:54). The TransGI chain had finished at 14:41:46. This identical rerun of the frozen plan, same commit, is the primary set. Every idle gate passed in 10–11 s.
- **Execution defect and fix.**
  - Defect: both chains' SVGF recordings captured an empty screen. The SVGF viewer compiles Triton kernels for about 33.5 s before its first frame, and the recorder's fixed 7 s delay plus 8 s capture ended first.
  - Fix: commit `baee724` adds `record110.py --wait-change` (default off, so frozen chain behaviour is unchanged) and `supp110.sh`. The script repeats only that run (`sss_svgf_cap30_rec2`, 16:51, gate 10 s).
  - Effect: only the recording was missing. The original run's timing data is valid and is reported alongside.
- **Post-hoc attribution, labelled.** A scratch probe (`gc.callbacks`, headless, not committed) showed where the single compute spike comes from. The spike occurs at the same frame in every windowed run, and the probe saw exactly one generation-2 Python GC collection of 39 ms in 450 frames, at the frame of the run's only spike (54.9 ms).

### Measurement — live loop (clean set; 512², 1 spp; 600 measured frames after 30 warm-up; LabServer69 RTX 3090)

"Compute" runs from the edit through to the uint8 display-ready image on the host. It is bracketed by synchronised parts, the same bracketing as WL101. "Present" is the Tk photo update plus the X round trip. "Interval" is present-to-present, so it gives the presented frame rate.

| run | presented FPS | interval median / p95 / max (ms) | compute median / p95 (ms) | present median (ms) | missed deadlines |
|---|---|---|---|---|---|
| SSS raw, 30 cap, r1 | 29.94 | 33.34 / 33.95 / 84.2 | 16.94 / 17.29 | 6.07 | 1 |
| SSS raw, 30 cap, r2 | 29.94 | 33.32 / 34.07 / 80.3 | 16.15 / 16.62 | 5.38 | 1 |
| SSS raw, 30 cap, recorder on | 29.94 | 33.33 / 33.99 / 79.9 | 16.27 / 16.90 | 5.50 | 1 |
| SSS raw, uncapped, r1 | 44.24 | 22.46 / 23.21 / 71.3 | 15.72 / 16.22 | 5.44 | – |
| SSS raw, uncapped, r2 | 44.51 | 22.31 / 23.10 / 67.0 | 15.65 / 16.14 | 5.36 | – |
| SSS raw, 60 cap | 43.93 | 22.61 / 23.47 / 69.6 | 15.91 / 16.38 | 5.37 | 600 / 600 |
| SSS raw, headless (no window), r1 / r2 | 60.88 / 60.68 (loop rate) | 16.31 / 16.72 · 16.37 / 16.81 | 15.34 / 15.71 · 15.39 / 15.71 | – | – |
| SSS SVGF, 30 cap, recorder (chain) | 30.00 | 33.34 / 34.06 / 35.4 | 17.30 / 17.89 | 5.36 | 0 |
| SSS SVGF, 30 cap, recorder (supplementary) | 30.00 | 33.34 / 34.04 / 35.0 | 18.18 / 18.62 | 6.02 | 0 |
| SSS SVGF, uncapped | 41.62 | 23.95 / 24.75 / 26.0 | 17.22 / 17.67 | 5.39 | – |
| near raw, 30 cap, recorder | 29.94 | 33.32 / 34.14 / 82.7 | 14.63 / 15.24 | 5.58 | 1 |

- **Part medians.** Taken from SSS raw, 30 cap, r1; values in ms.

  | part | median (ms) |
  |---|---|
  | geometry | 3.02 |
  | relation | 3.93 |
  | source | 1.00 |
  | ECRO neural response | 5.60 |
  | hand-off | 0.04 |
  | ray render (LOW + HIGH lookup) | 2.50 |
  | compose | 0.29 |
  | tonemap + readback | 0.37 |

  - The sum of the parts is within 0.14 ms of the compute wall time (V8).
  - SVGF adds a 0.65 ms guide pass and a 0.53 ms reconstruction.
  - In near mode the render part is 0.66 ms.
- **Spikes and misses.**
  - Every 30-cap raw run misses exactly one deadline, at frame 180 (compute 62–68 ms). The time is spent outside every timed GPU part, at the same frame index across processes; this is the GC pause above.
  - The headless runs show the same pause at frame 420, inside the relation part (44 ms).
  - The SVGF runs have no spike in their measured window.
- **Real-time tracking.** Over the 20 s measured window, the 30-cap runs track wall time within 0.04 s.
  - The uncapped and headless rows advance the frame-clock animation faster than real time (1.47× and 2.0×). They measure throughput, not a real-time animation.
  - The SVGF runs' first frame takes 33.5 s (Triton compilation). That time is inside warm-up but appears as "lag" on their overlay, which counts from loop start.
- **Recording overhead.**
  - The recorder used about 12 % of one CPU core.
  - Recorder on vs off, 30 cap: compute 16.27 ms vs 16.15–16.94 ms, same interval statistics, same single miss. No measurable effect.
  - The independent screen-side count of distinct framebuffer states agrees with the viewer's own rate (probe 120 Hz):

    | recording | distinct states / s |
    |---|---|
    | SSS raw | 29.90 |
    | near raw | 29.84 |
    | SSS SVGF (supplementary) | 30.00 |

- **Memory.**
  - torch peak: 198 MiB raw, 225 MiB SVGF.
  - Device memory delta: 1.51 GB raw, 1.57 GB SVGF (WL101: 1.57 GB).
  - Kernel recordings per process: state 2, source 1, frame 1. No re-recording occurred during the loop.

### Measurement — comparison with WL101 (all labelled; same machine, same stack)

| quantity | value |
|---|---|
| WL101 historical V_SSS_ANIM B_sel, 2026-11-17, 512² × 1 dynamic frame | median 15.66 / 15.66 ms (r1 / r2), p99 16.17 / 15.85 |
| WL101 same-day rerun (unchanged `run_b101.py`, clean set) | median 15.27 / 15.21 ms, p99 16.15 / 15.73; selective = full 48 / 48 each |
| Live engine, same 48-event V_SSS_ANIM sequence (verify V2), geometry → render | median 14.63 ms, p95 15.05 |
| Live loop compute incl. compose / tonemap / readback (30 cap) | median 16.15–16.94 ms |
| + Tk/X11 presentation | 5.4–6.1 ms → uncapped presented interval 22.3–22.5 ms (44.2–44.5 FPS) |
| Live loop without per-part synchronisation (V8, single end sync) | 14.82 ms vs 15.79 ms with per-part sync |

- The live loop runs the WL101 runtime at WL101's cost.
- What separates the 16–17 ms edit-to-frame latency from the presented rate is presentation: about 5.5 ms of Tk/X11 image upload.
- The per-part synchronisation used for instrumentation costs about 1 ms.

### Measurement — verification (clean set `verify.json`; identical values in the contended set)

| check | result |
|---|---|
| V1 checkpoint identity | SSS `2d9cb580…`, near `b3550787…` = WL101 record and the archived timing JSON |
| V2 archived V_SSS_ANIM replay | state digest equal 52 / 52 events vs both archived WL101 runs; dirty-ray counts 52 / 52 |
| V3 archived WL101 output at live segment end points (u = 0 / 1 = test configs exactly) | SSS table B rel L1 **0** (bitwise), 4096-spp e1 / e2 / hl rel L1 0.6–1.5e-7; near table rel L1 0 / 1.1e-6, images 0.5–1.5e-6 (WL101 tolerances 1e-4) |
| V4 fresh single-frame renderer (new RT built at 6 intermediate live poses, same seed) | table bitwise equal 6 / 6; composed HDR mean rel 5.6–8.6e-9, max abs ≤ 5e-5 (not bitwise: separately recorded frame program) |
| V5 selective = full rebuild | 35 / 35 sampled live frames |
| V6 determinism (two independent engines, 180 frames) | geometry, state checksums, table sums and image sums identical on 180 / 180 frames; the last displayed frame of separate processes is byte-identical (PNG sha256 equal for both 30-cap SSS runs, and for both SVGF runs) |
| V7 loop integrity | 540 / 540 generator calls, 0 checkpoint loads during loops, parameter digest unchanged; every live run: 1 load, response every frame, table changed on every moved frame |
| V8 timing synchronisation | per-part sum within 0.14 ms (max 0.17) of wall |
| V9 SVGF history | reconstructed = same-frame forced-reset control on 119 / 119 edited frames; history accumulates only on held frames (age 2 → 9) |

- **V9 silhouette ratios.** On the sphere's 4 px inner boundary band the reconstruction is 0.91× raw luminance (min 0.906), against 0.96× in the interior and 0.99× on the 12 px outer ring.

### Observation

- The window shows newly computed frames continuously while both boxes move. The per-frame table checksum and |ΔB| on the overlay change every frame, and the dirty-ray count stays at 80–81 k (SSS) or 95–96 k (near) of 650,240.
- Visible changes include:
  - box placement;
  - the box shadows on floor and wall;
  - the SSS sphere's neighbourhood;
  - in near mode, the checkered box in front of the near-mirror panel.
- At 1 spp the raw image is very noisy. WL101 measured that the SSS-tour higher-order change itself does not qualify as a material change (0 / 8 transitions). On the near tour, 7 / 8 transitions qualified, but intermediate near poses were never referenced.
- SVGF mode looks smooth, but every animated frame is an edit, so the history resets each frame and the result is spatial-only. Low-frequency mottling remains on the sphere, and the silhouette band is darkened relative to raw, as WL101B found. No contact artifact or temporal instability was quantified beyond V9.

### Interpretation

- **The frozen ECRO checkpoint does render a continuously changing rigid scene in a live frame loop.**
  - The pipeline executes on every frame, computing geometry, relation state, source encoding, generator, table and LOW + HIGH frame.
  - The frames are the WL101 runtime's own output: bitwise state and table, archived image agreement ≤ 1.5e-6.
  - The checkpoint is loaded once and never changes.
  - The animation is deterministic and runs in sync with geometry.
- **The complete displayed system on LabServer69 (RTX 3090, Xvfb + Tk):**
  - It sustains 30 presented FPS at 512² × 1 spp (29.94–30.00 FPS; interval p95 ≤ 34.1 ms) in raw, SVGF and near modes. The only miss is one GC pause per 600 frames in raw mode.
  - Uncapped it reaches **44.2–44.5 presented FPS** raw and 41.6 FPS with SVGF.
  - **60 FPS is not achieved:** at the 60 cap, 600 / 600 deadlines are missed and the rate is 43.9 FPS.
  - The compute-only loop rate without a window is about 61 Hz. The gap between that rate and the presented rate is presentation cost.

### Unresolved question

- A physical display path (GPU-attached monitor, vsync, a real X server or Windows desktop) was not available and was not measured; Xvfb has no scan-out. A desktop GL/CUDA-interop presenter would probably remove most of the 5.5 ms upload, but that is untested.
- The GC pause could be removed with `gc.freeze()` or by disabling GC in the loop. Not done: the frozen code is unchanged.
- Quality of intermediate poses against physical references, camera motion, and other resolutions / spp were not part of this batch.
- The RTX 5080 PC was not used: it was powered off.

### Architecture verdict

None. This is a demonstrator; it makes no SOTA, quality or architecture claim.
- The displayed motion lies inside the WL100 training-geometry support. It is not evidence of out-of-distribution generalisation.
- SVGF mode remains WL101B's unqualified candidate: its quality gate failed in WL101B, and the silhouette darkening is visible here.

### Reproducibility

- **Code.** Implementation `8db9448`; execution fix `baee724`; analysis and this report in the evidence commit.
- **Environment.** `experiments/ecro_live110/env110.sh`. LabServer69, RTX 3090, driver 550.54.15. Mitsuba 3.8.0 / Dr.Jit 1.3.1 / torch 2.3.1+cu121 / Triton 2.3.1, Python 3.10, Xvfb, Tk.
- **Configuration.** 512², 1 spp; camera WL100 SSS / near; WL100 exposure keys 0.1939 / 0.4835. Frame seed = 11,000,000 + frame index. Animation `anim110.py`: period 6 s, frame clock 30 Hz.
- **Launch.**
  ```
  source experiments/ecro_live110/env110.sh                 # on LabServer69
  Xvfb :110 -screen 0 1024x640x24 & export DISPLAY=:110      # or any real X display
  $LIVE110_PY experiments/ecro_live110/live110.py --case sss --mode raw --fps-cap 30 --frames 0 --out <dir>
  ```
  - `--mode svgf` selects SVGF; `--case near` the near-mirror mode; `--fps-cap 0` uncapped; `--present none` headless.
  - `--frames 0` runs until Escape, window close or SIGINT.
  - To view on a desktop, run it under any X display, for example `ssh -X`, VNC or a local Linux desktop. X forwarding over a network adds its own latency, which was not measured.
- **Full evidence.**
  ```
  setsid -f bash experiments/ecro_live110/run110.sh <root>
  bash experiments/ecro_live110/supp110.sh <root>
  python experiments/ecro_live110/analyze110.py <root> evaluation/110 [<contended root>]
  ```
- **Artifacts.** `evaluation/110/` with manifest `manifest110.json`, which holds sha256 and source paths:
  - screenshots and the primary recording (`recording_sss_raw_cap30_rec.webp`, 10 s, WebP q60);
  - GIF previews for near and SVGF;
  - per-run summaries, `verify.json`, the WL101 rerun JSONs, `runs110.json` and figures 01–03.
  - The full-size near and SVGF WebPs and all `frames.jsonl` files stay in `results/ecro_live110/`; their hashes are in the manifest.
