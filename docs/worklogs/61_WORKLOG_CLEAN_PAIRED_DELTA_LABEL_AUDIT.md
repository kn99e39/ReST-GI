# Worklog61 — Clean paired delta-label audit (2026-10-29)

## Authorisation and fixed scope

The user explicitly authorised one bounded clean-reference follow-up to the completed WL60 attribution. The question is whether the frozen K geometry-associated delta gap persists under independent precise references, rather than being materially explained by historical target error. No architecture, learned parameter, F/S algorithm, loss, training budget, geometry, material or historical verdict changes.

Starting source: `bcf2932`. Branch: `Selective_Recompute`, no new branch/worktree. [Design and interpretation boundary](../plans/CLEAN_PAIRED_DELTA_AUDIT_V1.md); [frozen machine protocol](../../experiments/fnto_delta61/protocol/audit_v1.json). Diagnostic source/protocol committed **before generation** as `edf2da5`.

Exactly **train split fields 8 and 9**, with identical incident arrays at rest, trained dent tr11 and validation dent va1. This is the minimum same-training-light control specified in WL60, not the separately generated validation fields numbered 8/9. Two independent 64M-photon action replicas per geometry/field, each from two 32M halves: **24 halves / 768M photons**. New seeds `610000000 + pose_index*10000 + field_id*100 + replica*10 + half`; no outcome-dependent source selection or photon increase.

Physical entry-to-first-exit reference code, material, 2.5 mm / 16-in / 64-out flux convention and fast execution walker are inherited unchanged from WL59. Independent renderer/analytic validation remains inherited evidence, not a fresh rerun here. All original data, labels and P_5901/P_5902 checkpoints remain read-only. New full half actions/moments live under isolated server root `~/rest_delta61`; small results and reviewer evidence are exported to the new numbered locations only.

## Preresult execution record

LabServer69 was idle before launch (RTX 3090, 1 MiB allocated, no compute peers); `/home` had about 1.3 TB free. A detached server process was started with a new session and audit lock. It runs focused tests, source/input/geometry preflight, exactly the fixed reference generation and CPU-only frozen evaluation; any failed stage stops. No original supervisor or training command is invoked.

All selected geometry/data/checkpoint hashes are pinned to WL59 provenance. Preflight also checks reconstructed material-vertex signatures and parent correspondence, current train fields against their original WL57 train arrays, unchanged reference source hashes, and the exact WL60 replay dependency/baseline hashes.

Uncertainty is recorded separately for single-replica differences and the final four-half mean. First/second moments are aggregated before forming variance. Independent current/rest variance adds for each delta; shared rest cancels in the linear difference of the two geometry deltas and is preserved in coherent replica sensitivity. Two replicas are not a bias proof or independent-pixel confidence interval. Original deformed r0 labels have no compatible independent variance estimate: discrepancy over new-reference SE is not a combined significance test.

Final measurements, eligibility and interpretation are appended after the fixed evidence completes. No inference about a correction or successor follows from launch.

## Pre-reference identity-checker correction (A1)

The initial six tests passed, then preflight stopped **before any reference photon** because it compared current rounded/canonical geometry hashes to legacy saved metadata. Investigation on all three fixed geometries established that the saved signatures exactly equal the reconstructed **raw float64 refined vertex bytes**. Parent maps match exactly, and recomputed xyz/rest_xyz/area/node/pair/prior tensors are bitwise equal to saved tensors for every pose. This is a signature-serialization checker defect, not a geometry or reference mismatch.

A1 checks the exact legacy raw signature, records the current canonical signature separately, and additionally requires bitwise-equal derived geometry features. No threshold, photon count, seed, field, geometry, learned parameter or transport semantics changed. Original source/logs/status and the diagnostic comparison are preserved under `failed_attempts/preflight_edf2da5`; the fixed protocol remains intact with a separate implementation amendment. Final generation remains blocked until corrected preflight passes.
