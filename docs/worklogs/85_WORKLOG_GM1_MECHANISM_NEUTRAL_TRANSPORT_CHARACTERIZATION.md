# Worklog — GM1 Mechanism-Neutral Transport Characterization (2026-11-11)

## Origin and scope

- **Brief.** WL85, "GM1 Mechanism-Neutral Transport Characterization" (user, 2026-11-11).
- **Kind of batch.** A bounded characterisation: the ER lobe census, a qualified rigid-edit glossy-interreflection workload with REF-P and REF-ER, path-class and edit-response accounting, and measured full-rendering economics.
- **No mechanism.** It selects, implements and adopts **no** global transport mechanism. No DDT, LPTG or reservoir candidate is built.
- **Frozen inputs.** ER, its compiler and its closure parameters are unchanged.
- **Where.** Branch `Selective_Recompute`; LabServer63 (idle); `experiments/gm1_char85/`, `results/gm1_char85/`, `results/evaluation/85/`; server `~/rest_gm1char85`.

## Design (frozen in `experiments/gm1_char85/protocol/char85.json`)

**Stage 0 — census** (`census85.py`).
- Each of the 8 fitted lobes of every frozen learned ER closure is evaluated alone with its own sampler, over 4 populations (658 materials), 5 incidences and 2 azimuths.
- Per lobe it records the directional albedo and the solid angle of the flux superlevel set holding 50 % of the lobe's energy, expressed as an angular radius r50. A GGX lobe has r50 ≈ 2α; Lambert has 0.77.
- Classes, declared before any path-class result: S r50 < 0.15 rad; M 0.15–0.45; B ≥ 0.45. Continuous distributions are kept.

**Stage 1 — fixture.**
- Geometry: floor (material B), sharp panel (S1), glossy sphere (S2), Lambert back wall, a floating Lambert baffle, one point light.
- T0: the baffle blocks the light → panel beam.
- T1: the baffle is translated by 1 m. The reflected family light → panel → floor / sphere is newly enabled, and the baffle's direct shadow moves onto another stationary region.
- Materials are chosen by a descriptor-only rule from the WL83 test populations.

**Stage 2 — references** (`pt85.py`).
- One estimator for REF-ER and REF-P: next-event estimation at each of at most 6 scattering events, then BSDF sampling, with no Russian roulette.
- 8 replicas × 8192 spp, with common random numbers across T0 / T1.
- Validation V1–V8 (Mitsuba's own path integrator; reconstruction; null; move-and-return; Dr.Jit vs torch closure; clean-path identity).

**Stage 3 — path classes.**
- Contribution by number of scattering events.
- A receiver-class × sender-class matrix from an exact lobe-class split of the evaluated closure (not the sampled lobe index).
- Geometric dirty flags: a segment intersects the baffle volume of either configuration. Clean paths are identical in T0 and T1, so the signed change equals dirty(T1) − dirty(T0).
- Regions are geometric only.

**Workload qualification Q1–Q5** is evaluated on REF-P before any conclusion.

**Stage 4 — economics** (`econ85.py`, alone on the GPU).
- Cold / warm REF-ER.
- Time-to-quality ladder with 4 replicates.
- Resolution scaling, instrumentation overhead, REF-P cost.
- A second view, repeated edits, and Mitsuba's megakernel for context.
- Costs to reach anchored quality levels are reported as measurements only.

## Pre-freeze probes and corrections (non-evidence)

1. **Renderer validation in dev.**
   - pt85 against Mitsuba's path integrator: region ratios 0.998–1.000.
   - Class / depth reconstruction ≤ 5e-7.
   - Null and move-and-return identical to rounding.
   - Dr.Jit vs torch closure 1e-7.
2. **Kernel recompilation at every launch.** The seed was a literal, giving 20–40 s per launch. The seed was made opaque (cache hit).
3. **Instrumentation cost.** The class split first used three masked closures per vertex. It was replaced by one per-lobe pass with the closure's own methods (checked by the reconstruction test).
4. **Clean-path identity failed per pixel** (up to 0.49 of the mean), although frame sums agreed.
   - Cause: the dirty test was asymmetric between the two placements for grazing segments.
   - Fix: dirty = the segment intersects the 1 mm-inflated baffle volume of **either** placement (the identical test in both configurations).
   - After the fix: per-pixel max 4e-6.
5. **Look render** (256 spp, the actual fixture materials; used only to confirm that the geometry and the phenomenon are in view).
   - With dev-pool materials the coated panel was nearly black. The material pool rule therefore gained a descriptor-only observability condition: slab absorption depth τ(1 − ā) ≤ 0.15.
   - With the actual materials, the newly enabled reflected patch, the sphere highlight change and the moving shadow are visible.
   - No path-class, signed-change or cost number was read from it.
6. **Throughput.**
   - With cached kernels an instrumented ER launch (32 spp at 512²) takes about 0.3 s wall, of which about 0.09 s is GPU.
   - References run as 5 concurrent processes; the timing stage runs alone.
   - In-place transform update is not exposed for the cube shape. The edit path is a scene rebuild (measured).
