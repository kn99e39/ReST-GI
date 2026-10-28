# Worklog 57 — Learned Geometry-Conditioned Intrinsic K: Rule vs State

Status: **PREREGISTERED; no learned result or architecture adoption yet.**

User-authorised bounded batch on `Selective_Recompute`, starting HEAD `55b6be0fb994cd33d3e12f34779fae2756413379` independently verified. Historical WL53–56 implementations, artifacts and negative physical results are frozen. No WL58 is authorised.

The exact preregistration is [rule57.json](../../experiments/fnto_rule57/protocol/rule57.json); the input/output semantics, single candidate, matched direct-conditioned control and fair cache policies are specified in [ARCHITECTURE.md](../../experiments/fnto_rule57/ARCHITECTURE.md). Both learned models use shared weights across poses. Rule–State separation is the hypothesis, not the adopted architecture.

O is the independently sampled physical first-exit oracle; S is its rest configuration reused after deformation. D conditions the passive spatial/angular decoder directly on observed geometry; R computes a compact graph geometry state for the same decoder family. The incoming/outgoing flux basis is the unchanged uniform 2.5 mm /16-in/64-out material boundary. Analytic zero-collision transport is shared and its geometry construction, storage and application are fully charged to both arms. Scattered response is learned, with nonnegative substochastic columns by construction. No target oracle is used during candidate geometry preparation or inference.

The fixed split is 13 training poses (rest plus 12 dents), two validation poses and four held-out dents with disjoint train/validation/test light fields. Two independent training seeds per arm and independent physical oracle seeds quantify repeatability. Model parameters are D 161,328 and R 162,048; both may cache partial affine features or all structured coefficients under the same 512 MB ceiling. Actual lifecycle timings at 1/4/16/64 distinct-light queries include geometry and state construction.

Gate 1 must establish reference validity and detectable geometry response before training. Final-step checkpoints are evaluated on every held-out geometry/channel, absolute and signed dynamic fields, angular pairs, fine/coarse spatial errors and hard physical invariants. Neither faster inaccurate model can win. Full conditional composition proceeds only for useful, physical models; learned-vs-oracle, oracle-vs-Cycles and learned-vs-Cycles are separate comparisons. WL56's 9/12 primary physical result and transmitted-blue floor are preserved.

Reproduction: commit the complete code/protocol before primary supervision or training; export that exact commit into `LabServer69:~/rest_wl57/repo`; run `experiments/fnto_rule57/chain.sh` in a detached server session. Stage logs, status, source/environment manifest, target hashes, checkpoints, all held-out rows, measured cost and a server report are retained under `~/rest_wl57`. No Blender/CUDA revalidation is performed: the already validated official Blender 3.5.1 CUDA configuration remains the physical reference scope.

Final measurements, interpretations, separated architecture verdicts and four-case compatibility will be appended after the declared gates finish. Rigid material-frame reuse and local dents are within scope; global bending/twisting and self-contact are untested.
