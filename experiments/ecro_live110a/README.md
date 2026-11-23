# ReST-GI ECRO Live — portable viewer (WL110A, work in progress)

A Python application that runs the **frozen WL100 ECRO checkpoint every frame** through the **unchanged WL101 runtime**
(`rt101.RT`) while two boxes follow a deterministic rigid animation, and shows the result in a Tk window. The per-frame work is
the WL110 engine: geometry, relation, source, ECRO generator, LOW + learned-HIGH frame, display. Nothing is pre-rendered.

This README is for someone building or running the viewer. Measurements and diagnosis are in
`docs/worklogs/110A_WORKLOG_ECRO_LIVE_PORTABLE_MACOS_BACKEND.md`.

## Backends

| backend | where | variant | generator | kernel replay | status |
|---|---|---|---|---|---|
| `cuda` | NVIDIA GPU (LabServer69) | `cuda_ad_rgb` | CUDA | on | validated (WL110: 30 FPS sustained at 512², about 44 FPS uncapped) |
| `llvm` | any CPU, including **macOS** | `llvm_ad_rgb` | CPU (or `mps`, experimental) | **off** | validated on Linux against CUDA; **never run on a Mac** |

- **Why replay is off on LLVM.** Dr.Jit 1.3.1 kernel replay corrupts the relation state on this backend. The eager path matches CUDA to about 1e-6.
- **Expected speed on the LLVM backend.** On an 8-core 2016 Intel CPU it is about 5 FPS at 512². macOS numbers are unmeasured.

## Run from a checkout (any OS)

```
pip install -e experiments/ecro_live110a            # mitsuba==3.8.0, drjit==1.3.1, torch==2.3.1, numpy<2, pillow
# the WL100 checkpoints are not in git: put configs.json, sss/model.pt, near/model.pt under results/ecro_material100/
rest-ecro-live info                                  # which backend, which runtime tree, checkpoint hashes
rest-ecro-live selfcheck --backend llvm              # cross-backend check against the CUDA reference (must print PASS)
rest-ecro-live view                                  # live window; Escape or close to stop
```

Useful options:

| option | effect |
|---|---|
| `--backend cuda\|llvm` | choose the backend explicitly |
| `--res 256` | resolution (default 512 on CUDA, 256 on LLVM) |
| `--case near` | the near-mirror scene instead of SSS |
| `--fps-cap 0` | uncapped |
| `--present none` | headless benchmark |
| `--torch-threads` / `--drjit-threads` | CPU thread budget |
| `--generator-device mps` | experimental: the generator on Apple MPS, checked once against CPU, with fallback |
| `--log run.json` | per-frame log |

## Build the macOS app (Apple Silicon) — untested on a Mac

1. Prerequisites:
   - Xcode command line tools;
   - `brew install llvm`;
   - Python 3.10–3.12 with Tk (python.org installer, or Homebrew `python@3.11` + `python-tk@3.11`).
2. Run `bash experiments/ecro_live110a/build_macos.sh --checkpoints <dir with configs.json, sss/, near/>`. It does the following:
   - creates `build/venv` and installs the package with PyInstaller;
   - stages the runtime tree (`tools/stage_runtime.py`: 11 modules, 5 data files and the checkpoints, all hash-checked);
   - **runs the LLVM self-check on the Mac and refuses to package if it does not PASS**;
   - builds `dist/ReST-GI ECRO Live.app` with PyInstaller (`ecro_live_macos.spec`: Mitsuba and Dr.Jit collected, `libLLVM.dylib` bundled);
   - runs `info` from the built app.
3. Double-clicking the app opens the viewer with the defaults.

The runtime tree keeps the repository layout because the frozen modules find their data relative to their own files.
`rest_ecro_live/runtime.py` searches for it in this order:
1. `$REST_ECRO_RUNTIME`;
2. the app bundle;
3. a staged `runtime/` next to the package;
4. the repository.

## What this is not

- No retraining, no frozen source edits, no new renderer. The LLVM path is the same Dr.Jit program compiled for the CPU.
- The animation stays inside the WL100 training-geometry support. The display is not evidence of out-of-distribution generalisation or of image quality. SVGF reconstruction is CUDA / Triton only and is not offered on the LLVM backend.
