# Worklog — Full-Depth Spatial HIGH Representation and SSS Correctness Closure (2026-11-26 – 2026-11-27)

**Batch class.** Regular numbered batch (WL115). It tests a new spatial-representation mechanism for the full-depth HIGH table: within-patch continuous, higher-order and adaptive representations. It is not an evidence-gap follow-up of WL114. Numbered after WL114 on `ecro_lr3_adapter`.

**Owner brief.** "ReST-GI — WL115: Full-Depth Spatial HIGH Representation and SSS Correctness Closure".

**Locations.**
- Code: `experiments/ecro_sr115/`. Protocol `protocol/sr115.json`; Stage B freeze section `stage_B_frozen`.
- Raw results: `results/ecro_sr115/primary/`.
- Reviewer artifacts: `evaluation/115/`.

**Machine.** Local RTX 5080 16 GB; ECRO runtime `external/8dna26/.venv-cu128` (Mitsuba 3.5.1 / Dr.Jit 0.4.6).

**Concurrent batch.** The WL114A session (LR3 learned stages, peer session) yielded the GPU to WL115. Its watcher starts WL114A only after this worklog's final report is committed, so the final report was committed after the last WL115 GPU stage.

## 1. Preserved state

Unchanged by this batch:
- WL112–WL114 evidence and verdicts: WL114 H1 = INCONCLUSIVE; Mandatory SSS Gate = NOT_RESOLVED.
- The qualified full-depth SSS target: camera events 3–24, K* = 24.
- ER / SDM, physical LOW (events 1–2), HIGH ownership from event 3 and source-independent linear response.
- The WL114 B1 / B2 checkpoints. They are used as matched controls only and are not adopted.
- LR3 is not touched.

## 2. Design (protocol committed before any WL115 evidence: `07ebddf`)

**Evidence classes**, fixed before any stream:

| Class | Configurations | Role |
|---|---|---|
| SELECTION | F0–F3 and SEQ_A / SEQ_B / SEQ_C, all WL114 views | Only these are used for attribution, selection and calibration. |
| HELD_SEEN | F4, N1–N4, SEQ_D | Evaluated in WL114; never used for any WL115 decision. |
| FRESH | N5–N8 and SEQ_E (new) | Boxes and light from the D112 rule with seed family 11515000; support points by a stress design (below). Minimum summed distance to any D113/D114 training configuration: 0.19 m. |

FRESH stress design:
- N5: contact on a floor-patch corner.
- N6: contact at a patch centre.
- N7: 8 mm gap.
- N8: 45 mm gap.
- SEQ_E:
  - a contact slide across the floor-patch boundary x = 0.75;
  - gaps of 5, 20 and 70 mm;
  - contact re-formation;
  - a revisit after a large slide.

**Stage A (attribution, SELECTION only).** All inputs are nested: the 8×8 and 4×4 tables come from the same samples.

Constructions:
- An 8×8 deep U construction of all patches (n = 1,024 per 8×8 cell). PC4 and PC8 come from the same samples.
- A 32×32 construction on a *diagnostic* contact neighbourhood: the 8 lowest sphere patches plus floor patches within 0.30 m. This is a ring-restricted diagnostic control only.
- A camera-conditioned (query-weighted) capture at 8×8 on all patches and 32×32 on the contact neighbourhood. This is a diagnostic upper bound only, because it depends on the camera.

Representations read out with the WL114 Stage B seeds, so they are paired with the WL114 full-depth references:

| Representation | Definition |
|---|---|
| PC4 | 4×4, piecewise constant |
| PC8 | 8×8, piecewise constant |
| BL4 / BL8 | Bilinear over each surface's cell lattice, with exact per-neighbour direction rebinning; no new information |
| R16 / R32 / R32on8 | Finer cells on the contact neighbourhood only |
| Q4 / Q8 / Q16R / Q32R | Camera-conditioned tables |
| Half-sample tables | Table-noise estimate |

**Focused tests T1–T8 (all pass; `tests115.json`, rerun after the execution change in `tests2/`):**
- Readout kernel = tracer112 readout (rel 4e-10). LOW is paired.
- Bilinear is exact on constant fields and reproduces linear fields to 2e-7. The sphere wraps periodically and the poles are clamped.
- Nested pooling = the tracer113 tables4 contract.
- construct_k(K = 4) is bit-identical to tracer114.construct_deep.
- Deep capture equals deep_pass, and HIGH is conserved over the cells to 7e-11.
- Address mismatch is ≤ 1.3e-5 at K = 32.
- The readout is linear and non-negative.

## 3. Execution record (facts, including the incidents)

- **GPU utilisation complaint (20:12).** The first Stage A chain was strictly sequential, with about 31 % GPU on the owner's monitor.
  - Execution change (`69f10f8`, no scientific change): grouped readout accumulation (1.8× faster) and independent stages run concurrently.
  - The concurrent version oversubscribed VRAM (con8 + capture > 16 GB). WDDM paging slowed the construction launches about 40× for about 12 min. The capture was stopped before it had written any file.
  - Chain v3 (`70c0ca7`, `96b6607`) caps the GPU at two processes. The capture batch went from 48 to 16; the capture is unpaired, and its seeds are unchanged.
  - A 5 s nvidia-smi log is in `logs/gpu_util.csv`. Mean utilisation 20:31–22:14 was 69 %.
  - The running 8×8 construction process was never interrupted. Its sums are unaffected.
- **Stage A2 (post hoc, labelled, append-only in the protocol, `3a21676`).** It was added after the first Stage A readouts (§5).
- **Reference rim (owner question, 2026-11-26 21:28).** The owner asked about the thin bright line on the lower sphere silhouette of the reference.
  - Diagnosis: the line lies exactly on the lower-silhouette pixels (view·n 0.04–0.08). There the luminance is 3–5× the neighbourhood, carried by camera events 2 and higher.
  - A camera ray that grazes the index-matched sphere (σ_t = 12 m⁻¹) crosses only a 5–6 cm chord (ballistic transmittance about 0.5). It then lands on the lit floor behind the sphere.
  - So the rim is a property of the declared SSS semantics (WL100 / WL53: transparent index-matched boundary). It is not a renderer defect, and it is present identically in every method frame (shared LOW / event-2 readout).
  - At the owner's request a Blender 5.1 scene was exported for direct inspection, not rendered: `evaluation/115/blender/`. It has the matched V0 / H2 cameras and the identical medium. Floor and panel materials are approximated (Principled coated metal). The ReST-GI reference EXRs are included.
- **Training gate stall.** Chain C made training wait for a manual `train_go` file, to be created after a dev training probe. The session ended first, so the GPU sat idle from 23:04 to 11:02.
  - The probe was then run: shared-parameter initialisation hash equal to B1's, 0.107 s/step on the dev set, peak 9.7 GB.
  - Training resumed at 11:02.
  - The later chain (`chainG115.sh`) has no manual gates.
- **Stage D readout thrash.** The two Stage D readout processes built learned tables and kept the torch cache, so together they oversubscribed the 16 GB card. WDDM paging slowed streams from about 11 s to 1,276–3,856 s, from about 13:40 to 15:00.
  - Execution fix (`bac06b6`, `4f2ad13`): release the torch cache before each readout. This brings each process to 4.4 GB, and the two processes then run in parallel.
  - The stream files completed during the slow period are unchanged. Readout values do not depend on timing.

## 4. Stage A — spatial error and reconstruction attribution (SELECTION, 48 contact-valid views, full depth)

Kernel authentication:
- The WL114 S4_F table read through the WL115 kernel reproduces the WL114 hl|S4_F at ≤ 1.4e-8 rel.
- LOW pairing with the WL114 streams is ≤ 3e-9 rel.
- Table noise (half-table window error) is p99 ≤ 0.8 % (PC8) and ≤ 0.4 % (R32).

| Representation | R1L failing windows (total) | Views passing R1L | Median max \|window e\| | Ring 0–10 cm mean | Near bias contact / lifted | Views passing all |
|---|---|---|---|---|---|---|
| WL114 S4_F | 15,310 | 0 | 8.1 % | +5.6 % | +1.3 / +3.6 % | 0 |
| PC4 (nested) | 13,848 | 0 | 7.3 % | +5.5 % | +1.2 / +3.6 % | 0 |
| BL4 (same information) | 16,727 | 0 | 6.9 % | +5.9 % | +2.8 / +4.1 % | 0 |
| PC8 | 6,588 | 0 | 5.8 % | +2.4 % | +2.7 / +2.4 % | 0 |
| BL8 | 5,505 | 5 | 5.3 % | +3.0 % | +2.6 / +2.6 % | 2 |
| R16 (contact 16×16) | 2,692 | 13 | 5.0 % | +3.7 % | +2.4 / +2.2 % | 9 |
| R32 (contact 32×32) | 1,849 | 22 | 4.4 % | +2.6 % | +2.3 / +2.1 % | 8 |
| Q16R / Q32R (camera-conditioned) | 0 / 0 (8 views) | 8 / 8 | 2.1 / 1.8 % | −0.2 % | ≈ 0 / +0.1 % | 8 / 8 |

Findings:
- **Resolution.** Refinement near the contact removes most of the localised error. 87 % of failing windows disappear with R32, against 52 % for global 8×8.
- **Contact gradient.** The full-depth HIGH leaving the sphere's lowest ring varies 2.5× across its 8×8 rows, darkest at the pole. The 4×4 lowest sub-row spans polar angles 0–29° (radius 0.19 m) under the equal-area (φ, z) parameterisation.
- **Reconstruction function.** It is not the limitation: bilinear interpolation of the same values is worse than piecewise constant.
- **Residual near over-brightness.** A broad +2–2.6 % remains for U tables at every spatial resolution. Refining from 4×4 to 8×8 even increases it at contact views (+1.2 → +2.7 %), because the coarse 4×4 spatial error had partly cancelled it.
- **Measure.** The camera-conditioned measure at the same resolution removes it completely. With an exact measure, 16×16 is spatially sufficient (Q16R ≈ Q32R).

## 5. Stage A2 — mechanism of the residual (post hoc, SELECTION capture views)

Setup: a contact neighbourhood at 16×16 spatial resolution, with each of the 64 direction bins split into 4×4 angular sub-bins (nested pooling to 2×2 and 1×1).

| Contact table | Mean R1 | R1L failing | Ring 0 | All pass |
|---|---|---|---|---|
| R16n (64 bins) | +2.44 % | 865 | +3.8 % | 0/8 |
| R16A2 (2×2 sub-bins) | +2.21 % | 171 | +2.9 % | 0/8 |
| R16A4 (4×4 sub-bins) | +0.99 % | 26 | +1.8 % | 3/8 |
| Q16RU (camera-conditioned contact, U elsewhere) | +0.08 % | 0 | −0.2 % | 8/8 |

**The residual after spatial refinement is the within-bin angular measure on the contact patches.** The outgoing HIGH toward nearby floor points at grazing exit directions is lower than the bin average. It converges with angular sub-binning toward the camera-conditioned limit.

Angular refinement is outside this batch's change scope, which is spatial only. It is recorded as the identified remaining mechanism (an owner decision), not implemented.

## 6. Stage B — candidate selection and independent oracle qualification

**Selection (frozen in `1587629` before any HELD_SEEN / FRESH evidence).** Class C2 (geometry-adaptive hierarchical refinement), candidate **CAR-16**:
- Readout: T = (1 − w) T4 + w T16 per 4×4 cell, with 4×4 children (16×16 per patch) on the support.
- Activation: w = clamp((m − 0.002) / 0.008, 0, 1). Here m is the cos-weighted exp(−d / 6.25 cm) proximity to a *different rigid object*, taken from the cell centre over the 64 bin-centre directions.
- The activation is geometry only, recomputed per state, and continuous in the geometry. It uses no ring, camera, ROI or reference.
- Calibration on SELECTION capture views (query-weighted importance): m ≥ 0.002 keeps 99.3 %; m ≥ 0.01 gives full weight to 85.6 %.
- Budget: about 950–1,200 active cells per state (about 5 % of the 4×4 cells, about 90 patches).

Rejected classes:
- C0: by BL4.
- C1: no reconstruction gain.
- C3: 16× the rows.

**Qualification** (HELD_SEEN + FRESH, 49 contact-valid views; dense base = WL114 S4_F or a new 4×4 construction; 16×16 on the support, n = 1,024; `analysis/stageB_decision.json`):

| | Dense 4×4 oracle | CAR-16 oracle | Change |
|---|---|---|---|
| R1L failing windows | 19,297 | 3,452 | −82 % (FRESH −74 %) |
| Median max \|window error\| | 9.0 % | 5.3 % | −42 % |
| Innermost ring mean \|bias\| | 5.5 % | 3.5 % | −37 % |
| Lifted near \|bias\| | 3.5 % | 2.1 % | −39 % (FRESH −38 %) |
| R1 / R2 pass | 43 / 45 | 49 / 49 | |
| R4 pass | **21** | **9** | worse |
| Mean near bias | +2.0 % | +2.4 % | more positive |
| Views passing the full 3 % gate | 0 | 4 | |

- Frozen G1–G5 all pass: **QUALIFIED**.
- The unfavourable rows (R4, mean near bias) are the A2 angular component no longer cancelled by the coarse spatial error. They are not regressions of the spatial representation.


## 7. Stage C — learned ECRO-CAR

**Data.** D115 = D114 (unchanged dense / coarse full-depth targets) plus, for every D114 configuration and light:
- the CAR-16 support;
- 16×16 deep U targets on the support patches (n = 32 per child cell and bin; val two halves);
- child geometric descriptors.

D115 is 11.2 GB and took about 75 min, as a second GPU process alongside Stage B.

**Model.** ECROL4 (unchanged dense head; shared parameters constructed first) plus one refinement head on the active cells. 822,928 parameters (+5.7 % vs B).

**Training.**
- Matched controls: the WL114 B1 / B2 checkpoints.
- The shared initialisation hash of CAR1 / CAR2 **equals** B1 / B2's `init_param_sha256`.
- Recipe, data and steps are as B; only the child OWN loss is added.

| Arm | Train s | Peak | Val refined (dense) rel L1 | Val child rel L1 | Child target half-noise | Dense parent vs child targets |
|---|---|---|---|---|---|---|
| CAR1 | 3,924 | 10.3 GB | 0.0831 | 0.0808 | 0.0734 | 0.0871 |
| CAR2 | 3,046 | 10.3 GB | 0.0832 | 0.0808 | 0.0734 | 0.0870 |
| B1 / B2 (WL114) | 3,673 / 3,490 | 5.3 GB | 0.0832 / 0.0831 | — | — | — |

- The dense part is unchanged by the added head.
- At table level the children are learned to 1.10× their target noise (dense 1.04×).

## 8. Stage D — static, independent views (HELD_SEEN + FRESH, 49 contact-valid views, full depth)

| | PC4_O | CAR_O | B1 | CAR1 | B2 | CAR2 |
|---|---|---|---|---|---|---|
| R1L failing windows | 19,297 | 3,452 | 42,236 | **15,341** | 48,494 | **13,825** |
| Median max \|window e\| | 9.0 % | 5.3 % | 11.5 % | 8.1 % | 12.1 % | 8.4 % |
| Innermost ring mean \|bias\| | 5.5 % | 3.5 % | 5.8 % | **3.2 %** | 6.0 % | **3.3 %** |
| Near bias contact mean [min, max] | +1.6 [0.2, 2.6] | +2.5 [1.9, 2.9] | +1.1 [−2.2, 3.9] | +1.8 [−1.7, 3.4] | +1.7 [−3.5, 5.8] | +1.5 [−3.2, 3.7] |
| Near bias lifted mean [min, max] | +3.5 [2.5, 4.9] | +2.1 [1.6, 2.6] | +4.4 [1.2, 7.0] | **+2.8 [1.1, 4.1]** | +4.2 [1.6, 6.9] | **+2.7 [0.9, 3.8]** |
| R4 pass (views) | 21 | 9 | 20 | 20 | 16 | 19 |
| Views passing the full 3 % gate | 0 | 4 | 0 | 0 | 0 | 0 |

Selection-set regression (48 views): R1L failing windows B1 21,303 → CAR1 7,124; B2 20,287 → CAR2 8,243. CAR2 passes the full gate at 3 selection views.

R6b cross-view spread passes at 13 / 13 fixtures for B1, CAR1 and CAR2.

**Frozen H2 (learned adequacy): INCONCLUSIVE** (same for both seeds).

| Criterion | CAR1 | CAR2 | Rule | Result |
|---|---|---|---|---|
| (a) Learned excess within 1.25 × B's + 0.005 | 89.8 % of views | 89.8 % | ≥ 90 % | missed by one view |
| (b) R1L failing windows vs B | −64 % | −71 % | ≥ 50 % reduction | pass |
| (c) Innermost ring \|bias\| vs B | −45 % | −45 % | ≥ 25 % reduction | pass |
| (d) Lifted near \|bias\| vs B | −36 % | −37 % | ≥ 25 % reduction | pass |
| (e) R5 RMSE ≤ 1.10 × B | 80 % of views | 67 % | every view | fail |

Notes on the two failed criteria:
- **(a)** The failing views are contact / 2 cm states where CAR is darker than its oracle: N3 CAR2 −5.0 % vs B2 −2.4 %; N1, SEQ_E contact −1.7 to −2.2 %.
- **(e)** Every failure is off-target (R5b) RMSE, at 10 / 49 views for CAR1 and 16 / 49 for CAR2. The values are tiny in absolute terms: relative RMSE 0.16 % → 0.20–0.32 %. Median ratio 0.96 / 0.99, maximum 1.33 / 1.71 (SEQ_E contact W0). The source is the refinement head outside the gate's near ROI, e.g. box–floor junctions in the support.

## 9. Stage E — dynamic correctness and reuse (same weights throughout)

- **Rebuild and revisit.** A fresh-process rebuild of every sequence state, and every revisit pair, is **bit-identical**. This covers T4, T16, the activation w and the support patch list. There is no stale refinement state.
- **Discriminating transitions.** Resolved transitions (|Δ FULL near HIGH| ≥ 5 SE): SEQ_A 12/12, SEQ_B 6/8, SEQ_C 8/8, SEQ_D 12/12, SEQ_E 14/14. The stale previous-state table is clearly worse (CAR1 / B1 transitions): SEQ_A 8 / 11, SEQ_B 4 / 4, SEQ_C 8 / 5, SEQ_D 8 / 6, SEQ_E 7 / 7. At SEQ_E 70 mm → contact, the stale CAR1 table errs by +28 % while the current table is +1.2 %.
- **R7 tracking** (slope; corr ≥ 0.9987 throughout; band 1 ± 0.1, corr ≥ 0.98):

  | Sequence | B1 | CAR1 | CAR2 | CAR_O | PC4_O |
  |---|---|---|---|---|---|
  | SEQ_A V0 / H2 | 1.12 / 1.09 | **1.03 / 1.02** | **1.02 / 1.00** | — | 1.12 / 1.10 |
  | SEQ_C W0 / W2 | 1.15 / 1.22 | **1.06 / 1.08** | **1.07 / 1.09** | — | 1.09 / 1.13 |
  | SEQ_D W0 / W1 | 1.12 / 1.14 | **1.07 / 1.07** | **1.07 / 1.08** | 1.03 / 1.03 | 1.07 / 1.08 |
  | SEQ_E W0 / W2 (states 2–5) | 1.17 / 1.15 | **1.09 / 1.08** | **1.09 / 1.08** | 1.04 / 1.04 | 1.10 / 1.09 |

  CAR passes R7 in every sequence and both seeds. B fails, and so does the dense oracle at SEQ_A / SEQ_C.
- **Activation continuity.** For gap-only steps, |Δw| is 0.07 (0 → 5 mm), 0.22 (5 → 20 mm) and 0.69 (20 → 70 mm), continuous in the geometry. Slides move the support with the sphere.
- **The 3 % gate at every state of a sequence fails** for every arm. The static R4 / R1L / R1 failures recur in the sequences.

## 10. Stage F (R9) and R3

**Stage F (R9 generalisation beyond the sphere): NOT_RUN.**
- The frozen condition (static and dynamic 3 % gates pass for both seeds) is not met.
- A further blocker is recorded: the trained ECRO conditions on per-material learned states. A different medium (σ_t, albedo) is an untrained material state, so it would need medium semantics outside the scope. A different sphere radius is out of the training distribution.

**R3 (equal-time frontier comparator): NOT_RUN.** The experiment-local CAR runtime has no optimised operating point to set against the WL110 frontier at equal time.

## 11. Stage G — cost (local RTX 5080, GPU otherwise idle; unoptimised experiment-local Dr.Jit 0.4.6 runtime)

| | OB1 (coarse) | B1 (dense 4×4) | CAR1 |
|---|---|---|---|
| Parameters | 735,664 | 778,464 | 822,928 |
| HIGH table | 0.9 MiB | 15.8 MiB | 26.8 MiB (children 10.9 MiB; 934 active cells, 84 patches; metadata 0.36 MiB) |
| Generator (median) | 2.5 ms | 16.0 ms | 25.4 ms |
| Generator peak memory | 0.17 GB | 2.2 GB | 2.2 GB |
| Descriptors / support | — | 3.1 ms | 3.0 ms (support) + 6.2 ms (child descriptors) + 3.0 ms |
| 512² 1-spp frame | 36.9 ms | 37.1 ms | 37.2 ms |
| Edit → new table | 96 ms | 155 ms | 305 ms |
| Training | 1,133 s | 3,673 s | 3,924 s (peak 10.3 GB) |

Target generation per light (n = 32): dense 4×4 1.09 s, plus CAR children 3.0 s, plus support / descriptors 0.12 s. D115 is 11.2 GB on top of D114's 17.6 GB.

Reading:
- The compact output (children on about 5 % of cells) does not make the generator cheap: the child rows are about 0.7× the dense rows, giving +59 % generator time.
- The edit path roughly doubles, from the support / descriptor ray casts and host-side table assembly.

## 12. Final report

### Implementation Fact
Everything is experiment-local in `experiments/ecro_sr115/`; no shared or production file was edited. Added:
- a K × K deep construction (with parity halves);
- continuous event-2 readout with piecewise-constant / bilinear / angular-sub-bin lookups;
- a camera-conditioned diagnostic capture;
- the CAR-16 support (geometry-only other-object proximity);
- the D115 refinement targets;
- ECRO-CAR (one refinement head; shared ECROL4 unchanged);
- the Stage A–G pipeline;
- a Blender scene export (owner request).

ER / SDM, LOW, the full-depth target, source-linear response and the gate are unchanged.

### Measurement
- **Spatial resolution.** Finer information near the contact removes most of the localised error.
  - Oracle (SELECTION): R32 −87 %, R16 −81 %, global PC8 −52 % R1L windows.
  - Oracle (independent): CAR-16 −82 %.
  - Learned (independent): −64 / −71 %.
- **Reconstruction function.** Bilinear interpolation of the same information is worse than piecewise constant (BL4 vs PC4: R1L 16,727 vs 13,848; R1 pass 25 vs 43).
- **Projection measure.** After spatial refinement a broad +2–2.6 % near over-brightness remains under the uniform measure at any spatial resolution. It is removed by the camera-conditioned measure and converges with angular sub-binning (A2: R1 +2.44 → +0.99 % at 4×4 sub-bins). It is the within-bin angular measure on the contact patches.
- **Learned CAR.**
  - It reproduces most of the oracle gain on independent and fresh views: R1L −64 / −71 %, ring 0 −45 %, lifted −36 / −37 % vs the matched dense B.
  - Frozen H2 is INCONCLUSIVE: the learned-excess rule misses by one view, and the off-target RMSE non-regression fails with tiny absolute RMSE.
- **Dynamic.**
  - Exact rebuild / revisit, discriminating transitions and stale tables clearly wrong.
  - R7 tracking passes in every sequence for both CAR seeds (slopes 1.00–1.09). B (1.09–1.22) and the dense oracle fail.
- **Cost.** Generator +59 %, table +69 %, edit-to-table about 2× vs dense B. Frame unchanged. Target generation +2.8× per light.

### Observation
- The full-depth dense-4×4 residual decomposes into:
  - (i) a localised spatial deficit on the contact neighbourhood, largest at the pole of the sphere's equal-area parameterisation and on the floor under it;
  - (ii) a broad angular-measure over-brightness on the same patches.
- At 4×4 the two partly cancel. Removing (i) exposes (ii): R4 pass counts drop and the mean near bias rises from +2.0 to +2.4 % (oracle).
- The reference shows a bright rim on the lower sphere silhouette. It is a property of the index-matched SSS semantics (ballistic transmission through short chords onto the lit floor) and appears identically in every method frame.

### Interpretation (one sphere radius / medium; one light per configuration; bounded to this SSS family)
- A geometry-adaptive spatial representation is a practical and learnable way to remove the localised near-contact spatial error at about 5 % of the cells.
- It also resolves the dynamic tracking overshoot that WL114 attributed to the spatial representation.
- It cannot meet the approved 3 % contract by itself: the remaining broad error is angular, not spatial. Piecewise-constant 64-bin directional resolution on the contact patches is inadequate, and this representation class (direction bins) is outside this batch's change scope.

### Unresolved Question
- Would an angular refinement on the same geometric support close the gate in learned ECRO, and at what cost? The A2 oracle reaches +0.99 % at 16 sub-bins, still not the camera-conditioned limit. This is an owner decision (new representation hypothesis).
- The learned excess at some contact / 2 cm states (CAR darker than its oracle by up to −5 %), and the small off-target RMSE increase from the refinement head outside the near ROI.
- Generalisation to other radius / medium (R9) and equal-time (R3) are not run.

### Architecture Verdict
| Question | Classification |
|---|---|
| Physical representation adequacy (spatial) | **SUPPORTED for the localisation errors** (CAR-16 QUALIFIED by G1–G5); **NOT adequate for the full 3 % contract**: the angular residual remains |
| Learned representation adequacy | **INCONCLUSIVE** (frozen H2: (a) misses by one view, (e) fails; (b)–(d) pass for both seeds) |
| Static correctness | **NOT met** (0 independent views pass the full gate for CAR) |
| Dynamic correctness | Reuse / rebuild / revisit exact; **R7 passes** for both seeds; per-state 3 % gate fails |
| Practical runtime viability | **Not established** (experiment-local runtime; generator +59 %, edit about 2× vs dense; no optimised path) |

- **Mandatory SSS Gate: NOT_RESOLVED** (R3 and R9 NOT_RUN as well).
- WL114 H1 = INCONCLUSIVE is unchanged.
- CAR-16 / ECRO-CAR is an **experimental representation**, not a qualified or adopted architecture candidate. Nothing is adopted.

### Reproducibility
- Commits:
  - `07ebddf`: Stage A protocol / code before any evidence;
  - `82dfe9f`, `69f10f8`, `70c0ca7`, `96b6607`: execution changes;
  - `3a21676`: A2, post hoc;
  - `1587629`: Stage B freeze;
  - `bf7f7cc`: Stage C–E code;
  - `bac06b6`, `4f2ad13`: Stage D execution fix;
  - `a3bd0be`: cost / figures.
- Seeds: `protocol/sr115.json` and the runner constants. Checkpoint sha256 values are in `stageC/CAR*/training.json`.
- Raw data: `results/ecro_sr115/primary/` (Stage A–D streams, tables, D115, checkpoints, analysis JSONs, `logs/gpu_util.csv`).
- Reviewer figures: `evaluation/115/` (manifest with sha256). Blender scene: `evaluation/115/blender/`.
- Runtime: local RTX 5080, external/8dna26/.venv-cu128 (Mitsuba 3.5.1, Dr.Jit 0.4.6, torch cu128); Blender 5.1 for the scene export only.
