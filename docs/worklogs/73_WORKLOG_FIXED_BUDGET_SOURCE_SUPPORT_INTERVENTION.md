# Worklog — Fixed-Budget Source Support Intervention (2026-11-04)

## Authorisation and scope

On 2026-11-04 the user issued the brief "Worklog 73 — Fixed-Budget Source Support Intervention". It authorises one bounded, controlled source-distribution intervention on the WL59 intrinsic-K family.

**The question.** At a fixed physical-supervision budget, does better training coverage of optical boundary conditions produce transferable intrinsic-K responses at independent source locations? And how do D1, A and B differ in that transfer?

**Two explanations to separate:**
- **H_support.** The representations transfer within boundary or source conditions that were represented in training, and fail only when those conditions are absent.
- **H_mapping.** The learned geometry-to-transport mapping does not transfer, even when the support is appropriate.

**Architectures, all preserved unchanged.** All use the fixed WL68 A/O.
- D1: WL59 W+UV, objective J.
- A: W+UV, objective J + L_sym, with exact symmetry.
- B: RSK, objective J.

**Intervention, the only change.** The training-source composition. S16-R keeps 16 source ports, replaces exactly 4 of the WL71 S16 sources, and allocates the replacements to underrepresented boundary conditions (side face, edge, corner) by geometry-only rules.

**Supporting work:**
- the new physical references the intervention needs;
- isolated training and evaluation adapters;
- a new independent confirmation population with boundary strata;
- a deformation configuration whose footprint is covered by enough source pairs;
- diagnostic and cost instrumentation.

**Preserved:**
- K/C ownership, first-exit semantics and ballistic B;
- the scattered path A → T → O;
- 16 incoming / 64 outgoing bins;
- positivity, linearity and passivity;
- materials, metrics and thresholds;
- parameter counts, objectives, optimiser, schedule, clipping and the 20,000-step budget;
- every historical checkpoint and result;
- the WL72 INCONCLUSIVE verdict.

WL72's confirmation population is now development evidence.

**Not authorised:**
- more than 16 training sources;
- capacity, feature or loss changes;
- tuning;
- held-out-based source or checkpoint selection;
- a new architecture;
- production K/C changes;
- new physics or coupled GI;
- automatic adoption or a successor.

**Walker.** The primary references use the unchanged WL57 walker. The WL72A streaming engine is not substituted.

**Placement.** Isolated `experiments/fnto_src73` paths, branch `Selective_Recompute`, no new branch or worktree, LabServer63 (main). User instruction (2026-11-04): 「원래 TransGI 작업을 기다리는 감시 프로세스가 있었을텐데 그건 신경쓰지 말고 네가 우선적으로 서버 선점해」 — the wrapper does not wait for other jobs; timings may therefore be contended, and `logs/gpu_sampler.log` records every GPU process.

## Stage 0 design (2026-11-04)

The frozen design record is `docs/plans/SOURCE_SUPPORT73_V1.md`; the protocol is `experiments/fnto_src73/protocol/src73_v1.json`.

**Inputs.** All rules are geometry only and ran on the archived rest/pt0 geometries on LabServer63 (`ports73.py propose`).
- The te0 geometry is the archived WL57 file from LabServer69, sha256 `79216410…`, copied through the PC with matching hashes.

**S16 audit.** Every S16 source is interior with edge distance ≥ 21.25 mm. No side, edge or corner condition is supported, even through exact images.
- Slots 2, 6, 8, 9 and 12 are one exact rest-symmetry orbit.

**S16-R.**
- Removed: slots 6, 8, 9, 12 (rule: rest duplicates among flat-interior sources).
- Added: 50 (TBE, port 3439), 51 (TBC vertex, 3823), 52 (SM, 408) and 53 (SR, 527).

**Confirmation population.**
- 23 static rest ports: INT near/far, BND_NEAR, BND_FAR, UNS and REM_NEAR.
- 6 te0 footprint ports.
- Deformation pose: te0, selected by a predeclared rule.

**References.**
- 1,024 new training packets (the S16 replica structure).
- 2,240 confirmation packets.
- 32M photons each, unchanged WL57 walker.

**Pre-freeze review.** Before freezing, an independent review workflow ran: three lenses, two adversarial verifiers per finding, 15 findings. It led to one revision of the design. No physical outcome, model prediction or WL72 error was used:
- BND_NEAR and BND_FAR became separate primary groups, and BND_FAR gained second far ports 90–92. Pooled BND let 5 mm interpolation decide.
- REM_NEAR (slots 76–79) was added, together with the removal cost in the regression test. INT_ALL by construction avoided the removed sites.
- Comparisons became three-valued (YES / NO / OVERLAP), so overlapping seeds are not read as "no improvement".
- A relative fit criterion now sits next to the absolute WL59 gates.
- The removal rationale was corrected: the removed sources are duplicates at rest only, not at pt0, including for A.
- DEF73 is declared a near-training-pose check, because te0 overlaps pt0.
- Crashed fits now quarantine their orphaned checkpoints.
- Stage 1 asserts that the qualified matrix sets equal WL72's.

**Dev smoke during design (not evidence).** A pre-freeze dev chain (`~/rest_src73_dev`) found one test defect:
- The adapter-identity test asserted bitwise target equality, but the GPU `index_add` aggregation is not bitwise deterministic. The test now uses relative 1e-6.
- After the fix, Stage 1 passed in dev:
  - evaluation reproduces WL72's archived bounds within 1.05e-6 (tolerance 1e-4);
  - the replays match the archived initial objective within 1e-7.
- The dev chain was then stopped for the design revision. Its training packets, which are unchanged by the revision, are reused by the resumed dev smoke.
