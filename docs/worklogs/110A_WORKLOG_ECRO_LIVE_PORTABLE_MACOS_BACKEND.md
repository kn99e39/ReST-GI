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
