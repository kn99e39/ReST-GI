# Future falsification design (DESIGN ONLY; not an executable batch)

> WL80B record (workflow `wf_9bbd762a-ae2`, 12 agent runs). Agent texts reproduced verbatim; headings added by the Coordinator. Design-only: nothing implemented, executed, trained or measured. The original C1 architect was not resumable; a stand-in reconstructed C1 from the sealed WL80A records (immutable inputs, sha256 in `inputs.json`).

## Design only: C1 rest-pose comparator qualification

This is a design, not an executable batch and not an authorisation. It is relevant only if the human decides a classical comparator role for C1 is worth qualifying. If the human decides C1 has no role, or the question is purely contractual (must K be learned?), no experiment is needed.

### Question (single, bounded)
On the convex rest slab, can the corrected C1 reproduce the existing MC rest-pose reference K on the scored strata within the predeclared gates, at the finest affordable resolution? If so, what is its measured cost at that accuracy?

Pass or fail decides one thing: whether a competent classical numerical comparator exists in the g = 0, homogeneous, index-matched scope.

The dent is **excluded by design**: no admissible W exists. The dent outcome stays UNRESOLVED and the report must say so.

### Contracts (frozen before any MC score is read)
**Operator.** The amended split:
- q = F_in u;
- ψ = Σ_{k<N} (ωW)^k ωG1 u, cold start, fixed N, float64 accumulation;
- K u = ωS1 u + ωΦout ψ.

**Far field.** Float64 exact-kernel layer-pair Toeplitz from ray-telescoped cell-pair kernels. There is no projection grid, and side-face corrections are explicit.

**Escape.** Complement escape. The P2* guard aborts if it binds above roundoff; it is not used as a clamp.

**Quadrature.**
- Per-beam point quadrature composed into G1 and S1.
- Rotations de-aliased and D4h-covariant.
- No direction node on a bin edge.

**Tallies.** Flux per flux, in the exit-triangle frame, with no cosine factor. A step-0 gate confirms the convention against reference code.

**Hard tests (any failure stops the run):**
- per-column closure d = ΣK + absorbed + r_N to fp precision;
- per-point and per-cell partitions;
- minimum entry ≥ −1e-9·max;
- superposition and zero input within 1e-6;
- D4h within a stated roundoff tolerance;
- an independent high-q or analytic β check on edge-port bins aimed at side faces.

**Convention gate.**
- H-function albedo on the scored mesh at blue central top ports, with ω < 1 test values.
- S1 against an independent MC single-scatter tally at q ≥ 65,536.

### Data
**Existing archived rest-pose references only:**
- S16-R held-out replicas 4–7, scored;
- replicas 0–3, reported separately;
- the INT / BND_NEAR / BND_FAR populations.

**Physical reference role:** the Cycles-volume-validated walker convention.

**No new physical references, and no training.** The source definition ("1.8–4 mm") and the archived B q and seed are read from metadata before scoring.

### Baselines and comparison classes
- **MC replica spread.** The noise floor.
- **Per-state MC cost at matched accuracy.** The decisive competitor for comparator competence.
- **Historical learned anchors on the same hardware class only** (RTX 3080 Ti; `76_*` l.254–255):
  - FIA 3.9 ms per light and 0.349 s per 64 lights;
  - A75 0.086 s per 64 lights.
  
  These are reported as context, not as a superiority claim.
- **Neural Full Recompute: NOT APPLICABLE.** There is no edit and C1 owns no learned state.
- **Internal F/S: not applicable at a fixed pose.**
- **Class-3 path-traced re-render:** not needed (different workload). If it is mentioned, it is labelled class 3.

### Metrics
- Worst-channel upper-95 % abs_coarse (gate 0.10), abs_fine16 (0.15) and con_coarse (0.25), per stratum.
- BND_FAR versus INT.
- Paired exact- versus bin-averaged-F_in contrast control.
- Diagnostics:
  - direct versus complement escape difference per stratum;
  - per-stratum N versus 2N truncation difference;
  - aggregated 16×16 reciprocity defect (diagnostic only).

### Refinement and floor
Three predeclared levels on each axis:
- lateral spacing;
- depth grading;
- angular quadrature count.

The steps are:
- Richardson self-convergence on C1 alone, checked against the predicted order (about 2 for lateral bias).
- The floor is estimated as (extrapolated C1 − MC) against the replica spread.

Where three levels are unaffordable on 12 GB, that stratum is **INCONCLUSIVE by rule**. It is never relabelled as a pass or a fail.

### Cost accounting
Measured, uncontended, same hardware, with timing boundaries declared:
- construction (quadrature, assembly);
- state memory and peak GPU memory;
- 1-light and 64-light apply;
- sweeps used per channel.

A converged tabulation is costed separately from per-light solves.

### Stop conditions
1. Any hard test or the convention gate fails. Stop and report a specification or convention defect; no scoring.
2. The finest affordable level exceeds the 12 GB budget at L0. Stop as affordability-limited for the whole experiment.
3. Construction exceeds a predeclared wall-clock cap. The cap is set from the FIA 75 s precedent multiplied by a declared factor before running.

No threshold is moved after a result, and no clamp, damping or tuning on the scored references is allowed.

### What each outcome would mean for the human decision
| Outcome | Meaning |
|---|---|
| **A. Gates pass on all rest strata, and C1 is cheaper than per-state MC at matched accuracy** | A competent classical comparator exists in scope (rest pose only). The human may decide whether to add a comparator role row. This says nothing about learned-K transfer or the learned-family failures (CD HIGH-2) |
| **B. Self-converging misses (order about 2, floor within replica spread)** | Affordability-limited on 12 GB. No competent comparator at this hardware. The physics is not falsified |
| **C. Plateau (order < 0.5, floor above replica spread) on any stratum** | A convention, reference or model-error floor. C1 fails as a comparator. The reference convention itself may need audit, which also concerns the learned arms' targets |
| **D. Gates pass but C1 is not cheaper than MC at matched accuracy** | C1 is accurate but not a competitive comparator. The baseline role has little value |
| **E. Stop condition 1 triggered** | The corrected specification is still incomplete. The role stays UNRESOLVED |

Every outcome leaves C1's dent behaviour, its admissibility as an FNTO K and the learned-K research question unanswered.
