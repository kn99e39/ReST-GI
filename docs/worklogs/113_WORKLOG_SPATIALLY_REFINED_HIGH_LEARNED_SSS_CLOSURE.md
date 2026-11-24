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

### Execution fix 1 (2026-11-24 21:16 KST, before any Stage A stream or evidence metric)

The first primary chain started at `c564977` and completed eight F0 and F1 constructions. It then stalled at about 0 % GPU in the decorrelated-RNG control.

The per-launch salt of the decorrelated lane seeding was a Python literal. Every launch therefore recompiled the fused five-event construction kernel: 64 launches at full scale, against only 2 in development.

The fix makes the salt opaque (`dr.opaque`). The hash values are identical (checked), so the RNG streams and results do not change.

The chain was stopped, partial temporary files were removed, and the run resumed. The eight completed constructions were kept.

### First primary pass to the Stage B selection (chain `58162b9`, 21:16–21:46 KST)

- Stage A qualification: **QUALIFIED**.
  - Q1–Q5, Q7 and Q8 pass, and Q6 qualifies every comparison.
  - Q4 footprint |t| ≤ 3.15 against the critical value.
  - Singular counts are 27–42 per replica against an expected 35.7.
- The frozen selection code returned **STOP** with `S4_adequate = False`.

### Execution fix 2 — NaN coverage of an empty region (2026-11-24 21:50 KST, after the Stage A metrics were seen; disclosed)

**Cause.** The only S4 "failure" was the coverage minimum at F0 H5. That camera has no visible near-floor region, so the near-region coverage is NaN (no queries). The decision line `adequate &= cov >= 0.995` evaluates NaN as a failure. Every other S4 criterion passed at every contact-valid view.

**Fix.** The minimum is now taken over defined regions only. This matches the frozen coverage requirement ("native query weight ≥ 0.995 in every region"): an empty region carries no query weight. It also matches WL112's handling of the same view, where NaN did not limit.

**Disclosure.** The fix was made after the Stage A arm metrics had been inspected. Both records are preserved:
- the pre-fix decision (STOP) and gate in `results/ecro_sss113/primary/frozen_pass1/`;
- the post-fix decision in `stageB/decision.json`.

**Unchanged.** No threshold, band, arm or rule changed. No sample or stream was regenerated.

### Stopped by the owner (2026-11-24 21:49 KST)

The owner asked for the run to be stopped. At that moment the chain from `dd5fb68` was re-running `analyze113A` with Execution fix 2. All chain processes were stopped, partial `.tmp` files were removed, and `chain.status` reads `STOPPED`.

**Kept on disk.** Everything under `results/ecro_sss113/primary/`:
- 24 Stage A constructions and the VHS descriptors;
- the 30 Stage A streams (D6 plus full depth);
- the Stage A qualification (QUALIFIED) and metrics;
- the pre-fix decision in `frozen_pass1/`.

**Not run.** The post-fix selection, data, training, and Stages D–G.

**Resume.** Run `bash experiments/ecro_sss113/chain113.sh` (detached). The chain skips completed outputs and restarts at `analyze113A`.

### Resumed run (owner instruction, chain `9290eb7`, 2026-11-25 03:45–06:21 KST)

| Time (KST) | Step |
|---|---|
| 03:46 | Analysis A rerun with Execution fix 2: **S4 (dense) selected**. S4 is adequate; L4 is not equivalent. |
| 03:47–04:19 | D113 data: 208 geometries, 9.45 GB. |
| 04:19 | Precheck passed, including the dense budget. |
| 04:19–04:55 | Both arms trained in parallel: O_U in 17.9 min, CAND in 34.4 min. |
| 04:55–05:09 | Stage D. |
| 05:09–06:17 | Stage E. |
| 06:17–06:21 | Rebuild, analysis, cost and figures. |

No further fixes were needed.

## Final report

Written for the research owner.

### Implementation fact

- **Locations.** Code is in `experiments/ecro_sss113/`. Raw outputs are in `results/ecro_sss113/primary/`. Reviewer artifacts are in `evaluation/113/`: 19 figures, an index and a manifest.
- **Representation.** The WL112 contract v3 construction at 4×4 gives nested 2×2 and coarse tables. Compact support uses the VHS ∪ MEDIUM rule.
- **Learned candidate.** ECRO-L4 is model86 plus a sub-patch head with geometric descriptors. By the frozen selection rule it was **trained in dense mode, on all 1,270 patches**.
- **Matched control.** O_U shares the samples, seed, schedule and initialisation with the candidate.
- **References.** One deep physical camera pass to 24 events gives D6 (paired with every readout) and the full-depth reference.
- **Gate metrics.** The 3 % gate quantities (R1, R1L, R2, R3 bias, R4, R5a/b, R6b, R7) are computed against both the full-depth and the D6 references.
- **Unchanged.** ECRO v0, the ER/SDM status, LOW/HIGH ownership, the material and all historical artifacts.

### Measurement

**Physical construction qualification: QUALIFIED.**

| Check | Result |
|---|---|
| Tracer / deep-pass identity | 0 / 3e-8 |
| Pinning | 2.1e-7 |
| Synthetic 4×4 | 96/96 within 4 SE; parent 5e-8 |
| Support, F0–F3 | footprint \|t\| ≤ 3.15; singular counts 27–42 vs 35.7 expected; mismatch ≤ 3.2e-6 |
| Sphere-interior origins (C) | zero |
| Image materiality | all qualified |
| LOW pairing | ≤ 1e-10 |
| Decorrelated RNG, Q8 | equal within the combined interval at every F0/F1 view: the RNG correlation inflates variance only |

**Dense 4×4 oracle: it generalises.**

On independent F1–F3 and a fresh F0, under the historical depth-matched contract:
- At all 14 contact-valid views, near and contrast lie inside the WL112 bands. Near is −0.0083…+0.0065 for S4 vs +0.037…+0.215 for U.
- Near stable relative HIGH RMSE is 2.1–4.5 % for S4 vs 23–42 % for U.
- No regression; coverage ≥ 0.9999.

Against the D6 reference, under the approved 3 % quantities:
- R1 near is −0.9…+2.1 % and R2 is −1.8…+1.1 %. Both are inside 3 %.
- **R1L fails at every view.** The number of failing windows falls from 5,000–8,600 for U to 65–380 for S4, and max |e| from 0.43–0.74 to 0.05–0.13.
- **The innermost ring (0–10 cm) stays at +1.5…+11 %** (U: +35…+64 %). **R4 fails at 9 of 14 views.**

**Compact local refinement: not sufficient under the frozen equivalence.**
- VHS ∪ MEDIUM refines 23–24 % of patches, which is 4.7× the coarse float count.
- Contact endpoints and rings are equivalent to S4 at every view (|L4 − S4| ≤ 0.0005).
- **Sphere-silhouette RMSE exceeds 1.10 × S4 + 0.0025 at 9 of 15 views**, and sphere interior at 2. The silhouette queries surfaces the compact rule leaves coarse.
- The frozen rule therefore selected dense S4. Budget: 29 min projected, 5.7 GB, 14.9 MiB.

**Learned generator adequacy (target domain, validation).**

| Arm | Coarse rel L1 | Refined rel L1 |
|---|---|---|
| CAND | 0.0247 | 0.0859 |
| O_U | 0.0246 | — |
| Target half-noise | 0.020 | 0.082 |

The coarse output measured against the 4×4 targets gives 0.0943.

**Static and cross-view fidelity (Stage D, raw 4,096 spp).** The learned candidate removes the artifact at every contact-valid view of F0–F4.

| Arm | F0 V0 near | F1 W0 near | F0 H3 near |
|---|---|---|---|
| O_U | +0.072 | +0.101 | +0.229 |
| CAND | −0.0009 | +0.0139 | −0.0151 |

- CAND is inside the near and contrast bands everywhere. Its learned-vs-oracle near RMSE is 1.3–4.8 %.
- **The historical contract fails at 4 of 17 contact-valid views** (F1 W0/W1, F4 W0/W1), all on the innermost radial band: CAND +0.015…+0.029 vs bands of 0.014–0.016. The oracle there is +0.010…+0.022, so part of this miss is representational.
- **3 % depth-matched contract.** R1L fails at every view and R4 at most views. R1 fails at F1 W0/W1 and F4 W0 (+3.4…+4.6 %). R6b spread is ≤ 3 % for F0–F4.
- **3 % full-depth contract.** It fails at every view: R1 is −8…−17 % (Stage F).
- **Off-target.** Silhouette RMSE is below O_U at every view (H5: 0.025 vs 0.043). The R5a/R5b bias parts fail at full depth because of truncation.

**Dynamic fidelity (Stage E, same weights throughout, no per-state fitting).**

- **Reuse mechanics are exact.** A fresh-process rebuild equals the in-sequence table (max rel 0) at all 17 states. Revisits give identical tables and supports. 21 of 24 transitions are discriminating.
- **SEQ_A, F0 location.** All states pass at H2. At V0 the 5 cm state fails the innermost band: CAND +0.019 vs band 0.026, while the oracle itself is +0.028. R7 slope is 1.11 / 1.08 with corr 0.9999; V0 is just outside 1 ± 0.1.
- **SEQ_B, slide across the patch edge.** All states pass.
- **SEQ_C, independent F1 location and light.**
  - Contact states lie inside the near bands (+0.010…+0.014), but the innermost band fails at W0.
  - **Lifted states fail.** At 3 cm, CAND is +0.031 / +0.025 vs bands of 0.021 / 0.022. At 8 cm it is +0.037 / +0.038 vs 0.029 / 0.028. The oracle there is +0.012…+0.023.
  - R7 slope is 1.12 / 1.18, so R7 fails.

**Depth truncation (Stage F, near region, relative to full depth).**
- Truncation contributes −6.8…−17 %.
- CAND: the spatial-plus-learned term is −1.8…+4.0 %, total −8…−17 %.
- O_U: the spatial term is +22…+30 % against truncation of −7…−13 %, total **+10…+17 %**. **For the coarse model, the spatial overbrightness and the truncation cancel in sign.**

**Runtime and storage (local 5080, unoptimised Dr.Jit 0.4.6).**

| | CAND (dense) | O_U |
|---|---|---|
| Parameters | 778,464 | 735,664 |
| Spatial outputs | 20,320 | 1,270 |
| Table | 15.8 MiB (+5 KB index) | 0.93 MiB |
| Generator | 17.6 ms, 2.2 GB peak | 2.6 ms, 0.17 GB |
| Support / descriptors | 3.7 ms | — |
| Edit → table | 167 ms | 108 ms |
| 512² 1-spp frame | 39 ms | 39 ms |
| Training | 34.4 min, 5.3 GB | 17.9 min, 0.6 GB |

Data generation took 32 min on two processes and produced 9.45 GB.

### Observation

- The coarse lobes disappear in every view and state, for both the candidate and the oracle.
- A narrow positive rim remains right under the contact. It is stronger for lifted spheres and at the F1/F4 positions.
- The stale table keeps the previous contact state's pattern after both contact formation and contact removal.
- Patch-boundary silhouette stripes are weaker with 4×4 output (silhouette RMSE about 0.013 vs 0.04).

### Interpretation

- **Spatial aggregation is the supported cause of the depth-matched artifact.** In independent configurations a camera-independent dense 4×4 uniform table removes 85–100 % of it.
- **4×4 is not sufficient for the approved 3 % localisation requirements.** A residual at the 6 cm sub-cell scale, within about 10 cm of the support, fails R1L and R4 even for the physical oracle.
- **The compact VHS ∪ MEDIUM rule keeps the contact fidelity, but not the silhouette.** The silhouette needs refinement on surfaces the rule does not select.
- **The learned dense candidate reproduces the oracle where training covers the configuration** (F0, F2, F3; SEQ_A, SEQ_B). At the F1/F4 positions and in lifted states (SEQ_C) it adds a configuration-dependent learned bias of about +0.01…+0.016. That is about the size of the band.
- **Reuse is mechanically exact.** Depth-matched dynamic fidelity holds in 2 of 3 sequences. It fails in the lifted independent states, and R7 slope is narrowly missed.
- **Every depth-6 representation fails the approved full-depth 3 % contract through truncation (−7…−17 %).** The coarse model's apparent full-depth +10 % is signed cancellation, not accuracy.

### Unresolved questions

1. What resolves the residual within 10 cm of the support, which blocks R1L and R4 even for the oracle? Finer than 4×4, or a different basis?
2. Why is there a learned bias at new positions and in lifted states, when D113 contains lifted spheres? Candidates are sampling density near 3–8 cm gaps, the descriptors, or capacity.
3. How should depth beyond D6 be handled? This is the full-depth blocker; depth was deliberately not changed in this batch.
4. R3 equal-time frontier and R9 radius/medium were not run.

### Architecture verdict

**Mandatory SSS Gate: NOT_RESOLVED** by the frozen classification. The learned candidate fails the historical depth-matched contract at 4 of 17 static views and at the lifted SEQ_C states. The approved 3 % full-depth contract fails everywhere through truncation.

The three parts reported separately:
- **Oracle spatial adequacy.** Dense 4×4 is SUPPORTED for the depth-matched historical contract. It is NOT sufficient for 3 % R1L/R4.
- **Learned spatial resolution.** It reproduces the oracle at contact configurations, with a learned bias at new and lifted ones.
- **Full-depth milestone correctness.** Not met, because of truncation.

Nothing is adopted, and Architecture.md's accepted contract is unchanged.

### Reproducibility

- **Commits.** Freeze `c564977`; Fix 1 `58162b9`; Fix 2 `dd5fb68`; stop record `9290eb7`; the report is the next commit.
- **Seeds.**

| Use | Seed |
|---|---|
| Constructions | 11301000 + 100·f + r |
| Decorrelated control | 11302000 + … |
| Streams | 11310000 + 100·f + 10·v + j |
| Data | train 11320100 / val 11320200 |
| Training | 11330001 |
| F4 oracle | 11341000 |
| Stage E | 11350000 + 1000·s + 10·t + 2·v + j |
| Operating point | 11360000 + … |

- **Environment.** Local RTX 5080 16 GB; Python 3.11, Mitsuba 3.5.1, Dr.Jit 0.4.6, torch 2.8.0+cu128. The WL100 checkpoint is `2d9cb580…`.
- **Checkpoints.** `stageC/{O_U,CAND}/model.pt`; SHA-256 in `training.json`.
- **Run.** `bash experiments/ecro_sss113/chain113.sh` (resumable). Figures: `fig113.py ROOT evaluation/113`.
