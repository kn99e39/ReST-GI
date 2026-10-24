# 8DNA dynamic-scene baseline (worklog 43, protocol `8dna_dyn_v1`)

Independent baseline investigation: what a **fixed, released 8DNA asset** plus the official Mitsuba path tracer
recovers after a rigid edit of another object, and at what fidelity, transport coverage and cost. Separate from FNTO
(no FNTO code is touched) and from the historical RNA / Surface-Owned / SPT-1 baselines.

The upstream tree `external/8dna26` (commit `4a2157c`) is imported, never modified. Released checkpoints only; no
training anywhere.

## Files

| file | role |
|---|---|
| `protocol/dyn_v1.json` | predeclared protocol: scenes, states, placements, seeds, spp, regions, metrics, classification |
| `dynlib.py` | runtime, scenes, box placement checks, pixel classes, counters, parameter digests, rendering |
| `dyn_stages.py` | render stages: `smoke`, `design`, `static`, `refs`, `neural`, `ladder`, `cost` (one process each) |
| `dyn_eval.py` | CPU metrics and the declared descriptive classification -> `eval/eval.json` |
| `dyn_exports.py` | reviewer images -> `exports/` (+ manifest) |
| `lego_envelope_analysis.py` | geometry-only check of the ReST-GI Lego edit against 8DNA's empty-envelope condition |
| `server/mi_env.sh` | OptiX user-space library fix for LabServer63 (see below) |
| `server/mi_optix_check.py`, `server/edna_smoke.py` | the segfault diagnosis and the first 8DNA server smoke |
| `server/run_dyn_chain.sh` | unattended orchestrator: waits for the WL42 FNTO chain, then all stages; `--dry` = 32x32 code check |

## LabServer63 Mitsuba segfault (cause and fix)

- Kernel module and `libcuda` are driver **595.71.05**; `/lib/x86_64-linux-gnu/libnvoptix.so.1` and `libnvidia-rtcore`
  are still **570.195.03**. OptiX's user-space part must match the kernel driver; Mitsuba crashes in `mi.load_dict` while
  building the OptiX acceleration structure (A/B: system libraries 1/1 crash at scene load; 595 libraries 9/9 render).
- Fix without root: `source server/mi_env.sh` puts `~/nvlibs/optix595` (the 595.71.05 `libnvoptix`, `libnvidia-rtcore`,
  `libnvidia-gpucomp`, `nvoptix.bin` extracted from NVIDIA's driver package, worklog 23) first on `LD_LIBRARY_PATH` and
  refuses to run if that copy does not match the running kernel driver. The permanent fix is an administrator installing
  the matching 595 user-space package.
- Separate exit-time crash: DrJit 0.4.6 `jit_shutdown()` segfaults inside `jitc_log` while printing its variable-leak
  report at interpreter exit (gdb backtrace), independent of OptiX. Every entry point therefore ends with `os._exit(0)`
  after its outputs are written (`dynlib.finish`).
- 8DNA *training* still does not fit the 12 GB GPU (worklog 23); this protocol does no training.

## Run (LabServer63)

```bash
cd ~/rest_8dna_baseline/repo/experiments/8dna_dynamic_baseline
setsid -f nohup bash server/run_dyn_chain.sh --dry > ~/rest_8dna_baseline/launcher/dyn_chain_dry.stdout.log 2>&1 < /dev/null
setsid -f nohup bash server/run_dyn_chain.sh > ~/rest_8dna_baseline/launcher/dyn_chain.stdout.log 2>&1 < /dev/null
```

Status: `~/rest_8dna_baseline/launcher/dyn_chain.status`; outputs: `~/rest_8dna_baseline/out` (`EDNA_OUT`).
