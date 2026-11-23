# Worklog 110A — ECRO live viewer: portable backend structure toward a macOS Python application (2026-11-23 KST) — DRAFT / WIP

> Status: **work in progress**. The code is committed locally as WIP and not pushed. Measurements below come from development runs on LabServer69, not a frozen evidence set. The remaining measurements run as a local detached chain (`chain110a.sh`).

## Brief and classification

- **Brief.** On 2026-11-23 the user asked, after WL110: "69번 서버에서 바로 돌아가는거 말고, Mac 기반 시스템에서 Python 기반 애플리케이션으로 빌드할 수 있는 구조도 만들어볼래?" In other words, a structure that builds as a Python application on a Mac instead of running only on LabServer69.
- **Number.** 110A, as a suffix of WL110: it ports the WL110 demonstrator under preserved contracts. Per the AGENTS.md policy, the owner decides the class.
- **Scope.**
  - No frozen source is modified (rt101, pt86, model86, pt100, common100, closure83_dr, scene86, live110, anim110).
  - No checkpoint change, no retraining, no architecture claim.

## Constraint

macOS has no CUDA, and Dr.Jit / Mitsuba have no Metal backend, so the WL110 stack cannot run there: Mitsuba `cuda_ad_rgb`, OptiX, torch CUDA and Triton. What remains available:
- Mitsuba's **LLVM (CPU) variant `llvm_ad_rgb`** (Embree ray tracing, LLVM code generation);
- torch on **CPU**, optionally **MPS** for the generator only.

## Implementation (WIP, `experiments/ecro_live110a/`)

### Backend shims (`rest_ecro_live/backend.py`)

The frozen code hard-codes CUDA in three places: `pt86` `layered81.init('cuda_ad_rgb')`, `rt101` `set_variant('cuda_ad_rgb')` + `dev = torch.device('cuda')`, and `torch.cuda.synchronize()`. The LLVM path replaces them with shims only:
- `mitsuba.set_variant` maps `cuda_*` → `llvm_*`;
- `rt101.dev = cpu`;
- `torch.cuda.synchronize` becomes a no-op, or `torch.mps.synchronize` when the generator runs on MPS;
- `DRJIT_LIBLLVM_PATH` is discovered from the bundled copy, Homebrew `/opt/homebrew/opt/llvm/lib/libLLVM.dylib`, `/usr/local/opt/llvm/…`, or Linux system paths.

Patching `torch.device` itself breaks `isinstance` checks in `torch.load`, so the module global `rt101.dev` is reassigned instead.

### Other modules

- **`runtime.py`** locates the runtime tree:
  - search order: `$REST_ECRO_RUNTIME` → PyInstaller `_MEIPASS/runtime` → staged `runtime/` next to the package → the repository;
  - it authenticates the checkpoint hashes (SSS `2d9cb580…`, near `b3550787…`, configs `fd2fa676…`).
- **`runtime_manifest.json`** lists the runtime closure, traced with an audit hook (imports + file opens, sss and near):
  - 11 modules: 9 frozen sources plus live110 and anim110;
  - 5 data files: mats86.npz, ecro86.json, dyn101.json, ecro_material100/protocol.json, ecro_nearmirror93e/scene.json;
  - 3 checkpoint/config files.
- **`app.py`** is a backend-neutral Tk loop over `live110.Engine`. It adds `--backend`, `--generator-device`, `--freeze` and `--relation`, plus `info` / `selfcheck` commands. The MPS generator path copies GeoState to MPS and checks the table once against CPU (rel L1 ≤ 1e-4, else it falls back to CPU). Untested: no MPS here.
- **`selfcheck.py`** replays WL101 V_SSS_ANIM (52 events, seeds 9000 + i, 128²) against a CUDA reference.
  - Predeclared tolerances: table rel L1 ≤ 1e-4, frame mean rel ≤ 1e-4 (the WL101 cross-stack values).
  - The reference (`tools/make_reference.py`, generated on LabServer69 CUDA) reproduces the archived WL101 state digests on 52 / 52 events.
- **`tools/stage_runtime.py`** copies the closure plus checkpoints into a standalone tree with `RUNTIME_MANIFEST.json`: sha256 values and the third-party / stdlib imports, for a PyInstaller hidden-import list.

## Diagnostics so far (development runs, LabServer69: i7-6900K 8C/16T, RTX 3090, Mitsuba 3.8.0 / Dr.Jit 1.3.1 / torch 2.3.1)

1. **The LLVM backend runs the unchanged WL101 runtime through shims.** With Dr.Jit linked to system LLVM 11: the user's LLVM 17 build needs GLIBCXX_3.4.30, which the host's libstdc++ lacks.
2. **Kernel replay (`dr.freeze`) is not reliable on LLVM for this program (Dr.Jit 1.3.1).**
   - Dr.Jit warns "frozen function recorded N times … dry-running the recording failed".
   - Per-ray comparison against CUDA over 44 V_SSS_ANIM events:
     - LLVM with freeze: up to 24,480 / 650,240 differing hit entries and table rel L1 up to 8.4e-2. It is correct on some events and corrupt on others.
     - LLVM eager (`FREEZE=0`): ≤ 2 differing rays (hit ties), dbin ≤ 1, table rel L1 ≤ 2.9e-6.
   - → The LLVM default is eager (`rt101.FREEZE = False`, set after `live110`'s import-time assert). This is WL101's B_eager execution, and WL101 showed replay = eager exactly on CUDA.
3. **Self-check, LLVM eager, selective relation: PASS.**
   - Table rel L1 1.7e-6 / 1.1e-7 / 1.1e-7 on the stored events 4 / 20 / 40; frame mean rel 3.3e-7 / 2.0e-7 / 2.1e-7.
   - Table / image sums ≤ 4.4e-7 on all 52 events.
   - State digest byte-equal on 0 / 52: 1–2 tie rays per event differ between OptiX and Embree. Dirty counts equal on 49 / 52.
   - Compute median 201 ms at 128².
4. **Performance on this CPU.**
   - With freeze (incorrect results, timing only): about 200–240 ms per 512² frame (≈ 4–5 FPS). Generator ≈ 90 ms (torch CPU, 8 threads), relation ≈ 70 ms, render ≈ 50 ms.
   - **Eager with selective relation in the continuous window loop: ≈ 1.35–1.41 s per frame at both 512² and 256² (0.70–0.75 FPS).**
   - Attribution (hypothesis, being measured): without replay, `update_state` does no bucket padding, so the dirty-ray count, and with it the kernel size, changes every frame of a continuous animation. LLVM then compiles a new kernel per frame. The self-check is fast because its ping-pong poses repeat and hit the kernel cache.
   - Candidate fix (implemented, unmeasured): `--relation full`, i.e. `rt101.RT.rebuild_state`, WL101 arm B_full, which equals selective (WL101; WL110 V5). It has a fixed size, and becomes the default when replay is off.
5. **Staged-tree test.** The app runs from a staged runtime outside the repository with a separate package copy: runtime authenticated, LLVM backend, window loop OK. The file closure is complete for the raw sss / near paths.

## Remaining work

- **Running now** (`chain110a.sh`, `results/ecro_live110a/chain/status.txt`):
  - self-check CUDA via the package;
  - self-check LLVM with full and with selective relation;
  - LLVM window loop 512² / 256² with full relation;
  - LLVM selective 256² (kernel-compile attribution);
  - CUDA window loop via the package.
- **Not yet written:**
  - `pyproject.toml` (entry point `rest-ecro-live`, pinned `mitsuba==3.8.0`, `drjit==1.3.1`, torch, numpy, pillow);
  - `build_macos.sh` (venv, `brew install llvm`, stage runtime, PyInstaller) and the PyInstaller spec (collect mitsuba / drjit, bundled `runtime/`, bundled `libLLVM.dylib`, hidden imports from `RUNTIME_MANIFEST.json`);
  - a README (macOS prerequisites: Apple Silicon, python.org / Homebrew Python with Tk, Homebrew LLVM);
  - a Linux onedir PyInstaller smoke test.
- **Must be verified on a real Mac** (impossible here):
  - Mitsuba 3.8.0 / Dr.Jit 1.3.1 arm64 wheels;
  - Dr.Jit finding Homebrew `libLLVM.dylib`;
  - Tk / ImageTk on macOS;
  - the MPS generator path and its CPU cross-check;
  - `rest-ecro-live selfcheck --backend llvm` PASS on the Mac;
  - the presented FPS on Apple Silicon.
- **Open decisions:**
  - default resolution for LLVM (512² vs 256²);
  - whether SVGF gets a torch port (Triton is unavailable on macOS; raw mode only for now);
  - a thin-client streaming option from a CUDA server, not requested, as an alternative for real-time on Mac;
  - committing the 3 MB CUDA reference arrays (`rest_ecro_live/reference/`, included in this WIP commit).
- **Freeze diagnosis follow-up.** The issue could be reported upstream or retested on newer Dr.Jit (≥ 1.3.1). Kernel replay is what makes CUDA fast; on LLVM, eager plus kernel-cache hits may be enough.

## Update (2026-11-23 evening) — local chains and their results

### `chain110a.sh`

Run 17:28–17:36 at commit `de97471` with a clean worktree, through a package copy and the staged runtime (19 files) outside the repository.

- **Self-check, all PASS** on the predeclared tolerances.

  | backend / relation | stored-event result vs CUDA reference | compute median at 128² |
  |---|---|---|
  | CUDA, package path | identical (table / frame 0.0) | – |
  | LLVM eager, full relation | table ≤ 1.7e-6, frame ≤ 3.3e-7 | – |
  | LLVM eager, selective relation | same as full | 201 ms |

- **Window loop (Tk on Xvfb), full relation, uncapped, 150 frames.**

  | run | compute median | presented FPS | notes |
  |---|---|---|---|
  | LLVM 512² | 1006 ms | 0.98 | neural 677, render 122, relation 84, compose 46, tonemap 37 ms |
  | LLVM 256² | 286 ms | 3.42 | |
  | LLVM selective 256² | 287 ms | 3.41 | |
  | CUDA 512² via the package | 18.4 ms | 38.8 | WL110 measured 44.2–44.5 FPS uncapped through `live110.py`; the gap is not yet attributed |

- **Correction of the draft's attribution.** Selective and full are equal at 256² after warm-up, so "a new kernel compile every frame" does not explain the steady state. The earlier 1.35–1.41 s at 256² and 512² most likely includes first-run compilation into an empty Dr.Jit disk kernel cache (`~/.drjit`, 863 MB / 424 entries by now). The animation is deterministic, so later runs hit cached kernels.
  - **Consequence for a Mac:** the first run of each new configuration pays a compilation cost until the cache is warm. This is unmeasured on a Mac.
  - The `--relation full` default for eager LLVM is kept; it is the fixed-size path.

### `chain110a_threads.sh`

Run 17:54–18:01 at commit `8613d55`; headless LLVM, full relation, 60 frames.

| configuration | compute | rate |
|---|---|---|
| 512², torch / Dr.Jit threads 8 / 16 (default) | 193 ms | 5.1 FPS |
| 512², 8 / 8 | 192 ms | 5.15 FPS |
| 512², 4 / 12 | 212 ms | – |
| 512², 12 / 4 | 214 ms | – |
| 512², 16 / 16 | 205 ms | – |
| 512², 6 / 6 | 208 ms | – |

- Neural (torch CPU) takes 83–117 ms and render 52–89 ms.
- **The headless 512² loop is about 5× faster than the windowed 512² run above.** Neural 85–97 ms headless vs 677 ms windowed.
- **The 256² sweep is contaminated.** It ran from 17:57 while another session's CPU-heavy job was running (`dev109b.py`, 17 processes, load average 18–20 on 16 hardware threads). Its results (154–1499 ms, erratic) are not used.
- Whether the windowed 512² slowdown is a Tk interaction or the same CPU contention is **unresolved**. `chain110a.sh` ran at 17:31–17:34; the WL109B job started about 17:56, but other load at 17:31 was not recorded.

### Execution notes

- **Mid-chain edit.** During the thread chain, `app.py` was edited (default `--res` 0 → 512 on CUDA / 256 on LLVM). The chain passes `--res` explicitly, so its runs are unaffected, but those runs used a file newer than the recorded start commit. Later chains import the committed package only.
- **CPU gate.** CPU benchmarks need one, in addition to the GPU gate.
  - `chain110a_cpu.sh` waits until no process uses > 30 % CPU and the 1-min load average is < 3 for 30 s (max 6 h).
  - `ps pcpu` is a lifetime average, so a process that was busy earlier can delay the gate. That is the conservative direction.
  - The chain then repeats the 512² / 256² thread configurations headless and runs window vs headless at both resolutions to attribute the Tk slowdown.
  - Launched 18:01 at commit `1152903`; it waits for the WL109B job to finish.

### Packaging files written (not yet executed anywhere)

- `pyproject.toml`: entry point `rest-ecro-live`; pins `mitsuba==3.8.0`, `drjit==1.3.1`, `torch==2.3.1`, `numpy<2`, `pillow`; `[build]` extra `pyinstaller`.
- `launcher.py`.
- `ecro_live_macos.spec`: collects mitsuba and drjit, bundles `runtime/`, the manifest, the reference and `libLLVM.dylib`; takes hidden imports from `RUNTIME_MANIFEST.json`; `target_arch` arm64.
- `build_macos.sh`: venv → `pip install -e .[build]` → stage the runtime with checkpoint hash checks → **LLVM self-check on the Mac must PASS before packaging** → PyInstaller → `info` from the built app.
- `README.md`.

## Remaining work (updated)

1. Read `chain110a_cpu.sh` (results in `results/ecro_live110a/cpu/status.txt`). Attribute the windowed slowdown and fix it if it is a Tk interaction.
2. Attribute the CUDA package-path gap: 38.8 FPS vs WL110's 44 FPS, possibly overlay text or the `proof` path.
3. Run a Linux onedir PyInstaller smoke test of the spec mechanics, in a separate venv and not during a CPU benchmark.
4. **On a real Mac:** `build_macos.sh`, the self-check, the MPS generator check, the measured FPS and first-run compile time.
5. Finalise this worklog, then commit and push. Push needs explicit approval.
