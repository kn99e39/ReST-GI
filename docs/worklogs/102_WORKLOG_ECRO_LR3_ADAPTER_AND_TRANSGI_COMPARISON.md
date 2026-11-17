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
