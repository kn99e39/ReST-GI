# WL102 plan — ECRO on Bitterli Living Room 3: scene-adapter extension and static comparison with TransGI (2026-11-17)

**Status.** Pre-registration, written before any LR3 result of either method in this batch. Protocol: `experiments/ecro_lr3_102/protocol/wl102.json` (authoritative for gates and metrics). Batch class: regular WL102 (owner decision, 2026-11-17). Approved scope: stages 1–3 below; stage 4 is not authorised.

## 1. Question

Can the existing ECRO v0 (WL100 checkpoint recipe; architecture, table, loss and schedule unchanged) be run on a real triangle-mesh scene, Bitterli *Living Room 3*, through scene-adapter extensions only? If so, how do its static GI quality (against a full-depth physical reference) and lifecycle cost compare with an independent reproduction of TransGI (arXiv 2506.09909) on the same scene, camera, reference and GPU?

The comparison in the ECRO room (scene86) may be used as an auxiliary control only. It never substitutes for the LR3 result.

## 2. Blocker classification (read-only audit before this plan)

| Requirement | ECRO now (WL101 deployment) | LR3 needs | Class | Extension (preserves table, operator I/O, training contract) |
|---|---|---|---|---|
| Geometry | analytic rectangles / sphere / boxes, hard-coded shape ids (`scene86.py:29-101`, `pt86.py:142-192`) | 25 OBJ meshes, 507k triangles | adapter | triangle → patch table, area-weighted patch sampling (§4) |
| Network vs N | generic: inputs centroid / frame / area / material row / openness, graph from state rays (`model86.py:83-99`); ran at N = 11,430 in WL101 | N ≈ 4–9k | — | none; int16 hit ids cap N < 32,768 |
| Two-sided | back hits discarded (`pt86.py:226,367`) | all BSDFs two-sided; wall normals point outward | adapter | flip the shading frame on back hits; back-side patches for open meshes |
| Materials | ER closure states; WL100 physical routines for non-ER materials, rows as encoder features (`common100.py`, `pt100.py`) | textured diffuse, roughplastic, roughconductor | adapter | physical dispatcher = Mitsuba BSDF; 69-float rows as encoder features only (WL100 precedent) |
| Lights | point / spot only (`pt86.py:245-259,378-402`) | one mesh area light | adapter | emitter-sampled NEE in LOW, field targets and the source encoding s[N,64,3]; operator input space, linearity and source independence unchanged |
| Camera | square, z-up target camera (`pt86.py:63-81`) | 640×360, y-up, fov 50 on x | adapter | W×H rays from the scene's to-world matrix |
| Training data | random boxes in the fixed room × point lights (`scene86.py:171-227`) | per scene | adapter | fixed LR3 geometry × point lights in LR3 free space; WL100 sizes, objective, schedule verbatim |
| Depth | D = 6 (`pt86.py:26`); 6 events = Mitsuba max_depth 7 | compare at 6 events and full depth | — | Ref-7 auxiliary, Ref-65 final; D not changed |

Verdict before implementation: **ADAPTER-LEVEL for a static, 6-event evaluation.** The known risk is fidelity, not compatibility. WL90–92 traced ECRO's sharp-receiver error floor to the 64-bin first gather, and LR3 has Beckmann α = 0.1 paint and vases. If fixing a failure would need finer bins or an exact first gather, the batch reports **STOPPED_ARCHITECTURE_EXPANSION_REQUIRED** and implements nothing further.

## 3. References (user decision)

- **Ref-65** (Mitsuba `path`, max_depth 65) is the final physical GI fidelity reference. Both methods' end-to-end quality and the quality–cost trade-off are judged on it.
- **Ref-7** (max_depth 7 = 6 scattering events in ECRO's convention) is auxiliary, for ECRO's representation accuracy.
- **Truncation error.** Ref-7 vs Ref-65 is measured, so that ECRO's total error splits into a representation part and a truncation part.
- **Common settings.** Same scene file, materials, light, camera and conventions for both references; the reference noise is measured separately.
- **No changes to either method's depth.** TransGI's transport settings are not changed, and ECRO's bounce depth is not increased. Depth and transport-semantics differences are reported explicitly.

## 4. Stages

1. **Pre-registration** (this plan and the protocol, committed before results); living-document authorisation record.
2. **Adapter implementation and validation** in a new folder `experiments/ecro_lr3_102/`. Original modules are imported; no historical file is edited.
   - Gate A1: the mesh path reproduces pt86 on the ECRO room with scene86's own partition.
   - Gate A2: the physical tracer at D = 6 matches Mitsuba max_depth 7 on LR3.
   - Gate A3: the patch layout is feasible.
3. **Static comparison** on LR3.
   - Ref-65 / Ref-7.
   - ECRO dataset, training and evaluation (area light held out; point-light controls).
   - TransGI datagen, training and evaluation on LabServer69.
   - Metrics, decomposition, cost table.
   - Worklog 102 and `evaluation/102/`.
4. *(Not authorised.)* A rigid edit on LR3. This would compare TransGI Native Reuse vs ECRO selective relation update; it needs a separate decision after stage 3.

## 5. Comparison gate (`BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md`)

- **Owned learned state.**
  - TransGI: per-vertex latents + MLP trained on LR3.
  - ECRO: generator weights trained on the LR3 dataset.
  - ECRO's material rows are deterministic encoder features, not learned.
- **Invalidation.** None. The scene is static (T0 only), so no Full Recompute / Native Reuse / Selective Update comparison is made or claimed.
- **Timing.** Declared per method in the protocol, all on LabServer69: one-time data generation, training and preparation; per-frame time. Path-traced references are ground truth only, never a cost comparator for neural rebuilding.
- **Comparability.** Physically aligned for quality (same scene, materials, light, camera and reference) and for timing (same host). Depth and transport semantics differ by design and are reported.

## 6. Not authorised

- ECRO depth increase; changes to the representation, bins, network, loss or schedule.
- Tuning or optimisation of either method.
- Dynamic edits.
- Superiority claims beyond the measured static scene.
- Promotion of the adapter into the ECRO architecture contract.
