# Worklog — LR3 Adapter Port to Mitsuba 3.6.4 and Port Qualification (2026-11-26)

**Batch class:** suffix batch WL114A. It directly extends the completed WL114 to remove the execution-stack constraint that left WL114's Living Room 3 learned stages (B–E) NOT_RUN. It introduces no new mechanism or hypothesis. The owner instructed "어댑터 변경 진행해" (proceed with the adapter change) after asking why LR3 was limited locally. WL114's results and verdicts are unchanged.

**Code:** `experiments/ecro_lr3_114a/` (`pt114a.py`, `qual114a.py`).
**Results:** `results/ecro_lr3_114a/qual/`.
**Runtime:** the local RTX 5080 with `C:/Projects/TransGI_Reproduction/.venv` (Mitsuba 3.6.4, Dr.Jit 1.0.5, torch 2.8.0+cu128).

## 1. Why the port, and why the old stack was used

**Why the old stack.** The ECRO line was developed and qualified on Mitsuba 3.5.1 / Dr.Jit 0.4.6, the 8DNA venv used from WL81 to WL101. In WL102 the LR3 adapter `pt102` was written to *inherit* `pt86` so that the estimator stayed identical to the qualified WL86 / WL100 tracer; gates A1 / A2 checked equivalence against `pt86`. That choice avoided a port. It ran on LabServer69 (RTX 3090), where the stack works.

**Why the port is needed.** On the local RTX 5080, Dr.Jit 0.4.6's large compress / scan kernels never return (WL114 §3), and Mitsuba 3.5.1 cannot load LR3: the load stalls in `Mesh::build_pmf`. So `pt102` cannot run locally.

## 2. What was changed

`pt114a.py` is `pt102` plus the `pt86` pieces it inherited (point / spot `Lights`, `bin_of`, `dir_in_bin`, `accum`, point-light `source`), copied statement by statement onto Mitsuba 3.6.4.

Only Dr.Jit 1.x spelling changed:
- `dr.eq` → `==`;
- `dr.cuda.ad.Float64` → `mi.Float64`;
- host-side counting.

`pt86` / `layered81` are no longer imported. `scene86` and `mesh102` are numpy-only and unchanged.

Unchanged:
- the estimator (NEE at every vertex, BSDF sampling, no emission at hit points, no roulette);
- the patch / bin / probe / state-ray / source definitions;
- the seeding and accumulation.

Added, with WL114 semantics:
- the deep event-bucketed suffix estimator (`trace_deep`, `render_deep`, `field_deep`; buckets up to camera event 96; host-index compaction).

The ECRO model / training (`model86`, `train102`) are torch-only and unchanged.

## 3. Port qualification (measured)

**A2 re-run.**
- Setup: ported tracer, D = 6, scene area light, WL102 seeds 102001 / 102002, 2 × 1,024 spp, 16 s.
- WL102's frozen A2 criterion vs Ref-7 **passes**: GI / FULL means within 3 %, RMSE / combined noise 1.002 / 1.036.
- Cross-version equivalence with the ORIGINAL `pt102` result recorded in `experiments/ecro_lr3_102/gate_a2.json` (same seeds):
  - **GI mean within 6–8e-5 of the original**, RMSE / noise 1.002 vs the original's 1.002.
  - DI mean is −0.13 % from the original, with RMSE / noise vs Ref-7 at 1.87 (original 1.09).

**The DI difference is a platform difference of the reference renderer, not a port defect.**
- Stock Mitsuba 3.6.4 (`path` max_depth 2, `direct` MIS, NEE-only, BSDF-only), run locally on the 5080, gives the same DI as the ported tracer.
- The authenticated Ref-7 / Ref-65 were rendered on the RTX 3090 with the identical Mitsuba 3.6.4, scene file (LF hash), seeds and spp.
- Comparing WL114 Stage A's local same-seed 65,536-spp re-render with server Ref-65:
  - the DI difference is −0.136 % (whole frame);
  - **99.96 % of it comes from 208 pixels**: a 10 × 22-pixel wall patch at columns 317–326, rows 94–115, where the local direct light is up to 53 % lower.
- Excluding that patch:
  - ported DI vs Ref-7: +2e-5 (the original `pt102` vs Ref-7: +2–3e-5);
  - ported vs local stock DI: +2e-5 over the whole frame.
- GI and FULL agree between local and server renders (GI +1.2e-4, WL114 §5).
- The cause inside the renderer (e.g. OptiX intersection behaviour at a seam near the emitter on a different GPU architecture) was not isolated.

**Deep estimator.**
- Events 1–6 of the deep pass equal the D = 6 render (rel L1 1.6e-8).
- The 96-event GI (2 × 1,024 spp) vs Ref-65 has mean +2.6e-4 and RMSE / noise 1.004. This is consistent with the tail beyond 64 events (1.2e-4) plus the local-vs-server GI offset (1.2e-4).

**Field / probe.**
- The deep field's f2 / fh equal `field_components` (rel L1 3e-8; counts identical).
- With the scene light, events 7–96 add 40 % of the D6 field HIGH (area-weighted), consistent with the camera-level truncation (−33 / −23 / −19 % GI).
- An earlier smoke with a hand-placed point light outside the scene bounds gave 0.4 %. That was a test-light artefact, not a result.

**Verdict.** The ported adapter reproduces the original WL102 tracer within noise on LR3; the only deviation is the platform-specific DI patch, which is shared by the local stock renderer. The ported adapter is **qualified for local LR3 use**.

## 4. Status

- The adapter port is complete and qualified.
- LR3 Stages B–E of WL114 (full-depth oracle tables, matched D6 / full learned arms, evaluation vs Ref-65) have **not been started**. They need an owner decision; the owner expressed doubt about whether LR3 is needed.
- Applying the WL114 K\* rule to LR3 Stage A's ladder gives **K\* = 96**: paint misses 1.2–1.3e-3 at 48 events. This would be fixed in the protocol before any LR3 learned evidence.
