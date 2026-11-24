# Worklog 113 — Spatially refined HIGH representation and learned SSS near-contact closure (2026-11-24)

> Append-only. Written for the research owner. This batch implements the owner brief "Spatially Refined HIGH Representation and Learned SSS Near-Contact Closure" (2026-11-24).

## Batch class, branch, machine

- **Number.** This is regular **WL113**, the next number on `ecro_lr3_adapter`, where WL112 is the highest. The Denoise_Reconstructor renumbering note records the owner's instruction to ignore other branches' numbers. The batch tests a new representation hypothesis: camera-independent 4×4 refinement, compact local support and a learned refined decoder. The WL112 post-hoc 4×4 result is used as evidence for this hypothesis, not as a WL112 PASS.
- **Base commit.** `1e31414`. No branch or worktree was created.
- **Machine.** Local RTX 5080 16 GB with `external/8dna26/.venv-cu128` (Python 3.11, Mitsuba 3.5.1, Dr.Jit 0.4.6, torch 2.8.0+cu128). No concurrent GPU job ran on this machine during the batch.
- **Preserved verdicts.** WL101G D. WL101H/I NOT QUALIFIED. WL112 first pass NOT QUALIFIED. WL112 amended Stage A QUALIFIED. WL112 Stage B NONE. Mandatory SSS Gate NOT_RESOLVED.
- **Sources read.**
  - WL100, WL101A–I, WL112 (protocol, Amendment 1, Stage B, post-hoc 4×4) and WL105B.
  - WL110 on Denoise_Reconstructor (formerly WL111): full-depth benchmark, −13 % energy beyond D6 in the near ROI, 3 % proposal.
  - `docs/SSS_NEAR_CONTACT_GATE.md` and `docs/CONTRIBUTION_APPROVAL_GATE.md` on Denoise_Reconstructor. These hold the owner decision of 2026-11-24: the gate is mandatory, τ = 3 % is conditionally approved, and the normalisation in §3.1 applies.
- **WL112 artifact check.** The freeze manifest, Amendment 1 and the reviewer manifest hashes resolve. The WL100 checkpoint hash is `2d9cb580…`.

## Design (frozen before evidence; `experiments/ecro_sss113/protocol/sss113.json`)

### Construction and representation arms

- **Construction.** The WL112 contract v3, extended to 4×4 sub-patches. One construction yields nested 2×2 and coarse tables by exact pooling. The primary measure is uniform U; G was over-corrective at 4×4 in WL112.
- **Arms.**
  - U: coarse.
  - S2_U: historical 2×2.
  - S4_U: dense 4×4, the adequacy reference.
  - L4_U: the compact representation.
  - R0_U: lowest-ring-only, a diagnostic support control. It is never a candidate.
  - Decorrelated-RNG S4 control on F0 and F1.

### Compact contract VHS ∪ MEDIUM

A patch gets 4×4 output if either of two geometry- or material-only rules holds:

1. **VHS (visibility heterogeneity).** The range over its 16 sub-cells of the cos-weighted fraction of hemisphere rays (the 64 bin centres) that hit a surface closer than the patch size √area is at least δ = 0.15.
2. **MEDIUM.** The patch belongs to a participating medium whose scattering mean free path is below the patch size. Here that is the SSS sphere: 1/12 m < 0.22 m.

Neither rule uses a camera, light, radiance, ROI or reference.

- **Lifecycle.** Support is recomputed from the current geometry at every state. Contact formation activates support near the contact; separation beyond the patch size removes it.
- **Budget.** At most 30 % of patches may be supported.

### Calibration (pre-freeze, disclosed)

The VHS range is bimodal on F0 and on four D113 training geometries. The support fraction is 21–22 % for every δ in 0.05–0.30, so δ = 0.15 is the plateau midpoint.

A table-level check on the training geometries compared the support against a debiased within-patch RMS deviation D(p) of the physical 4×4 table:

- The largest D (0.2–0.44) falls on contact-floor and lowest-ring sphere patches. VHS covers these.
- Sphere ring 1–3 patches show D 0.26–0.29 with a VHS range of 0. This is SSS structure inside the medium, which is why the MEDIUM rule exists.
- About 50 % of patches have D ≥ 3 % from smooth point-light gradients on Lambert surfaces. These are deliberately not refined and are judged by non-regression against the coarse table.

### Adequacy, equivalence and selection

- **S4 adequate.** At every contact-valid view of F0–F3, the WL112 historical bands (near 6.49 %, contrast 5.79 % of the D6 near HIGH) hold, with no regression against U, coverage ≥ 0.995 and precision ≤ 0.75 band.
- **L4 equivalent.** Both of these hold:
  - Endpoints and rings are within 0.25 band of S4 and inside the bands. Sphere and contact regions are ≤ 1.10 × S4 + 0.0025.
  - Floor-outside and full-frame regions show no regression against the coarse table, and the support budget holds.
- **Selection.** L4 if S4 is adequate and L4 is equivalent. Otherwise dense S4, if its learned budget holds. Otherwise STOP.

### Learned stage

- **Data.** D113 reuses the D112 moving-sphere distribution with WL113 seeds: 192 × 4 training fields and 16 × 4 validation fields, 32 proposals per sub-cell. Coarse targets are pooled from the same samples, so both arms see 512 proposals per cell.
- **Arms.**
  - **O_U:** model86 unchanged, trained on coarse targets.
  - **CAND (ECRO-L4):** model86 plus a sub-patch head. Its inputs are the patch token, the bin embedding, an 8-d sub-cell index embedding and three geometric sub-cell descriptors (VHS near fraction, mean log hit distance, escape fraction). Its outputs are a_sub ≥ 0 on the supported patches, combined with the same channel activations as the coarse output, so the head is linear and source-independent. It is trained with coarse OWN plus refined OWN.
- **Matching.** Same seed, schedule and data. Shared parameters are initialised identically (test T6).

### Evaluation

- **References.** Every stream is one physical pass to 24 events. Events 1–6 are RNG-identical to the depth-6 estimator (D6) and are paired with every readout. Events 7–24 form the tail, and D6 plus tail is the full-depth reference.
- **Contracts.**
  - Historical WL112 contract on HIGH vs D6.
  - **The approved 3 % contract** (R1, R1L, R2, R3 bias, R4, R5a/b, R6, R6b, R7, R8, R10, R11) against the full-depth reference.
  - The same 3 % quantities against D6, as a diagnostic.
- **Not run.** R3 equal-time frontier and R9 radius/medium. RESOLVED is therefore unreachable in this batch, and the protocol states this.
- **Fixtures.** F0 (development/regression, six cameras), F1–F3 (independent configurations for representation selection) and F4 (Stage D only). All F1–F4 cameras are contact-valid by a geometry-only check. F2/F4 use W2 at azimuth 20° because the 35° camera falls outside the right wall when the support x is ≥ 0.95.
- **Sequences.**
  - SEQ_A: gaps 0, 2 mm, 1 cm, 5 cm, 1 cm, 0 and 5 cm. This covers R7 gaps, formation and removal, and revisits.
  - SEQ_B: slide across the floor-patch edge.
  - SEQ_C: independent F1 lift.
  - Stale and fresh-rebuild controls throughout.
- **Gate classification.**
  - PARTIALLY_RESOLVED: the full-depth 3 % contract is met on every evaluated class, or the historical depth-matched contract is met statically and dynamically.
  - NOT_RESOLVED: no adequate representation, or the learned candidate fails depth-matched.
  - UNQUALIFIED: Stage A is not qualified.

## Development before the freeze (throwaway seeds, offset +7,000,000, reduced sizes)

### Tracer and support checks

- Events 1–6 of the deep pass equal the depth-6 pass (rel L1 3e-8).
- Tail events 7–24 carry 1.8 % of the V0 total. Events 25–40 add < 1e-11.
- The 4×4 parent identity holds to 4e-8.
- Tests T1–T8 pass. On the lift sequence, sphere support is 32 patches at every state; floor support is 71 patches in contact and 65 at a 5 cm gap; revisited states give identical supports.

### Throughput (owner question during development)

The owner noticed sparse GPU utilisation. A full-scale per-stage probe with nvidia-smi logging found these causes:

- **Fused depth-6/24 camera kernels.** Six to eight events with SSS walks were fused into one kernel, which ran slowly and compiled slowly. The tail pass ran at 17–30 % GPU.
- **Per-launch tracing.** Python tracing per launch dominated the small launches.
- **Synchronous compression.**

Fixes, all execution-only with identical values:

- **Per-event evaluation in the physical camera pass.**
  - One deep pass at 4,096 spp to 24 events takes 27.5 s at 81 % GPU and 5.1 GB.
  - This replaces fused D6 (53 s) plus tail (61 s).
  - Events 1–6 are identical to the fused D6 pass to 1.4e-9.
  - For constructions, the same per-event evaluation filled 16 GB and was slower, so constructions keep the fused WL112 trace.
- **Larger launches.** 64 spp per launch, identical for the physical and readout passes, so pairing is preserved.
- **Asynchronous compression of outputs** (zlib releases the GIL).
- **Parallelism.** Constructions and streams run as two processes.
- **Dr.Jit cache release** after each pass.

### Instrument fixes found on the development run

No evidence seed was used.

- **Native-mask expansion.** The expansion crashed; fixed.
- **Q4 zero-dispersion t.** A fully covered sub-cell with identical replica counts gave t = ∞. The test now uses an overdispersed-binomial floor.
- **Q4 singular registration.** The F1 and F3 tangency lies exactly on a patch edge, so the singular disk spans several sub-cells. Registration now covers every sub-cell the disk meets; equal sub-cell areas keep the expectation unchanged.
- **Development data ranges** (chain).

### Disclosure

- Oracle readouts from the reduced development run were written to the development root. Their figures were not opened before the freeze.
- No decision rule was changed after any development value.
- Development used a forced L4 selection to exercise the learned path. That switch exists only in development mode.

## Execution

(appended below)
