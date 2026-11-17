# Worklog — ECRO on Bitterli Living Room 3: Scene-Adapter Extension and Static TransGI Comparison (2026-11-17)

> Append-only execution record. Batch class: regular **WL102** (research owner's decision, 2026-11-17).
> Plan: `docs/plans/wl102_ecro_lr3_transgi/PLAN.md`. Protocol: `experiments/ecro_lr3_102/protocol/wl102.json`.
> Branch `ecro_lr3_adapter` from `724a38e`. Host: LabServer69 (RTX 3090). Approved scope: stages 1–3 (static); stage 4 (edits) not authorised.

## 2026-11-17 — Brief and decisions (user)

- **Target.** Bitterli Living Room 3. First classify why ECRO cannot run it into input/scene-adapter vs architecture problems.
  - Minimal compatibility extensions that keep ECRO's representation, transport semantics and training contract are allowed.
  - A core architecture change, a new representation, or a large scene-specific redesign is not implemented; it is reported as STOPPED_ARCHITECTURE_EXPANSION_REQUIRED.
  - The ECRO-room comparison is an auxiliary control only. No performance or result-improving tuning.
- **References.**
  - Ref-65 is the final physical fidelity reference.
  - Ref-6 is auxiliary, for ECRO's representation accuracy. In ECRO's event convention it is Mitsuba max_depth 7, so it is named Ref-7 in this batch.
  - Truncation error (Ref-7 vs Ref-65) and reference noise are measured separately.
  - Neither method's depth or transport is changed.
- **Code location.** A new branch of this repository; new folder `experiments/ecro_lr3_102/` only. Original modules are imported, never edited.
- **Comparator.** An independent TransGI reproduction (private repository `TransGI_Reproduction`, recommended configuration D_vis + LeakyReLU, fast frame). Its earlier measurements were on LabServer63 and are not used for timing here.

## 2026-11-17 — Read-only classification (before any implementation)

- **Verdict:** ADAPTER-LEVEL for a static, 6-event evaluation; no class-B blocker. Table in the plan, §2.
- **Spot-checked in code.**
  - The generator inputs are centroid, frame, log-area, material row and openness, with a runtime form-factor graph from state rays (`model86.py:83-99`). There is no patch-id embedding.
  - The source encoding is a per-patch, per-incident-bin direct irradiance computed by a point-light loop (`pt86.py:378-402`).
  - State-ray hit ids are int16 (`pt86.py:374`).
- **LR3 scene facts** (from the TransGI conversion): 25 objects, 310,562 vertices, 507,424 triangles, 176.4 m², one mesh sphere light.
  - The walls mesh has no wall on the camera side, so the room is open there.
  - Wall normals point outward and the scene is y-up.
- **Material contract.** WL100 precedent: a physical routine owns a material that ER cannot represent, and its 69-float row is an encoder feature only. LR3's roughplastic (Beckmann) and roughconductor (complex Fresnel) are dispatched to Mitsuba's BSDFs on the same basis.
- **Noted for interpretation** (WL98 D1): the generator takes absolute world coordinates and a global mean token. LR3's coordinates lie outside the ECRO room's range. A fresh per-scene model is trained, so no transfer from the room checkpoints is assumed.

## 2026-11-17 — Environment and references (TransGI side on LabServer69)

- **TransGI environment on LabServer69** (TransGI repository `scripts/setup_labserver69.sh`):
  - uv Python 3.12; Mitsuba 3.6.4 / Dr.Jit 1.0.5 for data and references; torch 2.6.0+cu124.
  - tiny-cuda-nn built with the conda gcc 11 toolchain, with its libstdc++ on the runtime path.
  - User-space LLVM 17 for Dr.Jit's CPU backend (the system LLVM 11 is too old for Dr.Jit ≥ 1.3).
- **Fast-frame venv: Mitsuba 3.8.0 / Dr.Jit 1.3.1, not the 3.9.1 / 1.5.0 used on LabServer63.** Driver 550.54's OptiX compiler rejects PTX emitted by Dr.Jit 1.5 (`unimplemented PTX intrinsics: copysign.f32`). WL101's ECRO arm B uses the same 3.8.0 on this host.
- **TransGI tests.** Main venv 35 passed / 1 skipped; fast venv 44 of 45. The failure is the optional `instanced=True` mode, which hits Dr.Jit 1.3's frozen-function registry validation; the static frame does not use it.
- **References are re-rendered on LabServer69, not reused.** WL55 recorded that LabServer63's Mitsuba cuda film accumulation is wrong (LabServer69 exact), and the TransGI references were rendered on 63 with `mi.render`. The 63 copy is kept as `static_V0_host63.npz` for a host comparison.
  - Ref-65: `render_reference.py configs/living_room_3.yaml`.
  - Ref-7: `--max-depth 7 --suffix _d7`.
  - Both: 65,536 spp, 2 halves, DI with common random numbers.

## 2026-11-17 — Amendment 1 (execution fix before any LR3 ECRO result): patch rule and gate A1

1. **Patch rule.** The frozen rule binned whole triangles by centroid. On LR3 it gave Floor (2 triangles) 2 patches of 21 m² and Walls (8 triangles) 8 patches. That contradicts the rule's stated 25 cm cell, a defect of how the rule was written, not a tuning choice.
   - **New definition.** A patch is the exact surface region {object, dominant-normal class of the face, 25 cm grid cell}. Faces are clipped at the cell planes, and a hit is mapped by its position.
   - **Unchanged:** the cell size, the classes, the frame rule, the back-side rule.
   - **LR3 layout (gate A3 numbers)** (`mesh102.py`):
     - N = 8,051, of which 3,742 are back-side patches of open two-sided meshes (walls, floor, rug, paint, cushions, vases).
     - Patch area: median 0.0625 m² (one full cell); p10 6.96e-4 m²; minimum 6.8e-9 m² (cell-boundary slivers, kept by rule).
     - Area-weighted mean cos to the patch normal: median 1.0, p01 0.859, minimum 0.685 (small curved objects).
     - The sphere light is a closed mesh with 24 patches.
   - **A3 holds:** N < 32,768, and N is inside WL101's measured range (≤ 11,430). The dense N×N form factors take ≈ 0.26 GB.
2. **Gate A1 operationalised (before running it).** "Identical hits on ≥ 99.9 % of rays" cannot hold by construction: the mesh path samples patch points through a triangle CDF instead of scene86's (a, b) parameterisation. A1 is therefore a statistical equivalence on the ECRO room (WL86 E1_T0, all surfaces Lambert, its point-light set), mesh path vs pt86, with the noise taken from two pt86 runs with different table / field seeds. The criteria (all within 2× that noise):
   - per-patch openness and form-factor row sums;
   - per-patch source totals;
   - field f2 / fh at 256 spp;
   - a D = 6 camera render at 256 spp.

## 2026-11-17 — References on LabServer69 (no ECRO result yet)

| | GI mean (R, G, B) | GI noise RMSE | full spp × halves |
|---|---|---|---|
| Ref-65 (max_depth 65) | 0.5445, 0.3494, 0.2799 | 0.00392 | 65,536 × 2 |
| Ref-7 (max_depth 7 = 6 events) | 0.3632, 0.2684, 0.2276 | 0.00298 | 65,536 × 2 |

- **Truncation error.** Ref-7 vs Ref-65, GI mean −33.3 % / −23.2 % / −18.7 %; GI LDR RMSE 0.0963. LR3's albedo-≈1 white paint keeps long paths important.
  - This exceeds the TransGI paper's total LR3 error (LDR RMSE 0.058) and the TransGI reproduction's LeakyReLU result on LabServer63 (0.077, for orientation only).
  - It bounds what a 6-event ECRO can reach on Ref-65 whatever its representation accuracy.
- **Ref-7 vs Ref-65 DI.** The DI of the two references agrees to 6.7e-6, so they share the same DI (common random numbers).
- **Host check of the TransGI reference (LabServer63 copy vs LabServer69).**
  - GI agrees within noise (RMSE 0.00069 vs noise 0.0039).
  - FULL and DI differ by ≈ 5× their noise (DI 0.0030 vs 0.0006); the errors are common to FULL and DI and cancel in GI.
  - This is consistent with WL55's LabServer63 film-accumulation finding. It concerns the TransGI reproduction's GI+DI metrics at the 0.3 % level, not its GI metrics.

## 2026-11-17 — Gates A1–A3 (adapter validation) and execution fixes

Code: `experiments/ecro_lr3_102/{mesh102.py, pt102.py, gate_a1.py, gate_a2.py}`. Results: `gate_a1.json`, `gate_a2.json` (copied next to the code; raw arrays on LabServer69 `~/rest_wl102/results/wl102/`).

- **A2: PASS.** LR3, physical tracer at D = 6 with the scene's area light and Mitsuba BSDFs, 2 × 1,024 spp, vs Ref-7.
  - GI mean difference ≤ 0.008 % per channel; RMSE / combined noise 1.002.
  - FULL: ≤ 0.007 %, 1.005. DI: ≤ 0.003 %, 1.09.
  - This validates the mesh geometry, camera, area-light NEE, two-sided handling, material dispatch and the event convention against Mitsuba itself.
- **A1: PASS.** ECRO room E1_T0, all-Lambert, its point lights; mesh path vs pt86 on the 1,088 shell patches whose regions coincide. Mesh-vs-pt86 RMSE / pt86 seed-to-seed RMSE:
  - openness 0.70 (means 0.1891 / 0.1890);
  - source totals 0.98;
  - field (2..5 events, 256 spp) 1.03 (0.929 / 0.930);
  - D = 6 render (512², 256 spp) 0.99 (1.768 / 1.768).
- **A3: PASS** (Amendment 1 numbers above): N = 8,051 < 32,768.
- **Execution fixes before passing.** All are adapter or test-fixture defects found by A1; none changes a rule or threshold, and no LR3 ECRO result existed.
  1. The polygon clipping dropped vertices lying exactly on a cutting plane from the upper piece, losing area on cell-aligned diagonals. Fixed so that such a vertex belongs to both sides; a unit check conserves area exactly.
  2. A planar face lying on a cell plane (the ECRO room's walls at x = ±2, floor z = 0) had its normal-axis coordinate jitter across that plane in float32, so its key lookup failed for about half the hits. The lookup now retries the neighbouring cell along the face's dominant-normal axis.
  3. Gate A1's test sphere mesh had inward winding, so it was treated as the inside of a closed mesh.
- **LR3 A2 numbers are identical before and after fix 2.** LR3's walls do not lie on cell planes.

## 2026-11-17 — Reviewer images (references and gate A2)

`evaluation/102/` holds the files listed below; sources and SHA-256 hashes are in `manifest.json`. Tone map: LDR = clip(x, 0, 1)^(1/2.2). Difference maps show the mean over RGB of |LDR a − LDR b|, magma colour scale 0–0.3.
- `references_overview.png` (contact sheet)
- `ref65_{full,gi,di}.png`
- `ref7_{full,gi}.png`
- `truncation_gi_absdiff_ldr.png`
- `physD6_{gi,full}_2x1024spp.png`
- `physD6_vs_ref7_gi_absdiff_ldr.png`

Visible in the truncation map (descriptive): the error left by stopping at 6 events concentrates in the wall niches, along the upper walls and in the niche shelves. These receive light mostly through long paths between albedo-≈1 paint surfaces. The gate-A2 difference map is noise only.

ECRO and TransGI prediction images will be added after the stage-3 chain. That chain started at 08:32 UTC: ECRO dataset 371 s, then training.
