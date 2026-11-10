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

## Run (frozen chain `355a161`, LabServer63, 2026-11-10 15:59–16:27 UTC, 28 min)

| Stage | Wall | Mean GPU | Result |
|---|---|---|---|
| S0 census (658 materials) | 35 s | 81 % | done |
| S2 validation | 156 s | 2 % | **PASS** (V1–V6, V8) |
| References: 32 renders, 8 × 8192 spp × {ER, P} × {T0, T1}, 5 concurrent processes | 18 min | **89 %** | done |
| S3 analysis / workload gate | 21 s | — | **UNSUITABLE by the frozen rule** (one clause of Q3; see below) |
| S4 economics (alone; no foreign GPU process) | 361 s | 10 % | done |

**Validation.**
- V1, pt85 against Mitsuba's path integrator: region ratios 0.9999–1.0010.
- V2 / V3, reconstruction: ≤ 5e-7.
- V4 / V5, null and move-and-return: identical to rounding.
- V8, clean-path identity: per-pixel 2e-6. In the full references it is ≤ 3e-6.
- V6, Dr.Jit vs torch closure: 1e-7.
- Truncation: the 6th scattering event carries 0.02 % of the radiance.

**Post-chain supplements** (`posthoc85.py`; labelled; they change no frozen result):
- a diagnostic of the failed clause;
- repeated edits with an in-place geometry update. The frozen chain rebuilt the scene for each edit, which recompiled the kernel (about 36 s per edit). That is a harness artefact, not a path-tracing cost, and those frozen-chain numbers are not used.

## Final report

### Implementation Fact

- **Stage 0 census.** A per-lobe census of the frozen learned ER closures (658 materials + the fixture) gives the albedo share and the angular support (r50) of each fitted lobe, measured with the lobe's own sampler.
- **Renderer.** One characterisation path tracer (`pt85.py`) serves REF-ER and REF-P: next-event estimation at each of ≤ 6 scattering events, no Russian roulette. Its instrumentation records:
  - contribution by event count;
  - an exact receiver-class × sender-class split;
  - geometric dirty flags under common random numbers.
- **Fixture.** Floor B (test83_iid #115, α 0.43 / 0.33), panel S1 (test83_hard #97, α 0.051 / 0.052), sphere S2 (test83_hard #41, α 0.13 / 0.11), Lambert wall and baffle, point light. The baffle is translated by 1 m.
- **Not built.** No global transport mechanism, cache, reservoir or reuse algorithm was implemented. ER is unchanged.

### Measurement

**ER census (45° incidence; q10 / median / q90 over materials).**

| Population | Sharp share (r50 < 0.15) | Moderate share | Broad share (≥ 0.45) | Lambert-slot share | Materials with Lambert ≥ 10 % | Closure r50 (rad) |
|---|---|---|---|---|---|---|
| test83_iid (300) | 0 / 0.00 / 0.32 | 0 / 0.41 / 0.94 | 0.03 / 0.50 / 0.98 | 0.02 / 0.20 / 0.53 | 69 % | 0.26 / 0.47 / 0.64 |
| test83_hard (150) | 0 / 0.00 / 0.07 | 0.64 / 0.94 / 0.98 | 0.01 / 0.04 / 0.11 | 0.01 / 0.04 / 0.09 | 7 % | 0.20 / 0.29 / 0.40 |
| test83_block (200) | 0.08 / 0.36 / 0.64 | 0.01 / 0.06 / 0.25 | 0.29 / 0.54 / 0.76 | 0.24 / 0.50 / 0.72 | 100 % | 0.19 / 0.44 / 0.66 |
| real83 (8) | 0 / 0.05 / 0.32 | 0 / 0.13 / 0.61 | 0.03 / 0.72 / 0.94 | 0.03 / 0.24 / 0.70 | 75 % | 0.20 / 0.57 / 0.67 |

- At 80° incidence the sharp share rises to a median of 0.51 (iid) and 0.33 (hard); the support narrows towards grazing.
- 2–3 lobes carry ≥ 10 % of the albedo (median).
- Fixture materials:
  - S1: about 43 % sharp and 56 % moderate (its main lobe has r50 0.164); Lambert 0.2 %.
  - S2: about 97 % moderate; Lambert 1.5 %.
  - B: about 99.8 % broad; Lambert 17 %.

**Workload qualification (REF-P, frozen rule).**

| Item | Value | Result |
|---|---|---|
| Q1 enabled reflection | +10.5 % (SNR 4154) | PASS |
| Q2 new shadow | −85.6 % | PASS |
| Q3a ≥ 3 events | 3.2 % of stationary radiance | PASS |
| **Q3b** share of events ≥ 2 in the enabled-region change | **0.465 against 0.50** | **FAIL** |
| Q4 sphere | 29 % | PASS |
| Q5 reference uncertainty | 0.0022 | PASS |

**Frozen status: UNSUITABLE.**

*Post hoc diagnostic (labelled):*
- The frozen region overlaps 3,643 of 28,835 pixels whose direct light is also unblocked by the edit (the baffle's T0 shadow).
- In the 25,192 pixels whose direct segment is clear in both configurations, the change is +3.6 % of T0 (REF-P), of which 99.4 % comes from ≥ 2 events.
- So the phenomenon is present. The failed clause reflects the region definition.

**Reference fidelity (pooled 65,536 spp; g = relative L1 of REF-ER against REF-P at T1).**

| Region | g | Notes |
|---|---|---|
| stationary pixels | 0.017 | uncertainty 0.002 |
| floor | 0.025 | |
| sphere | 0.030 | highlight 0.026 |
| enabled region | 0.056 | signed +5.6 % |
| panel | 0.069 | signed −4.3 % |
| wall, new shadow | INDETERMINATE | gap < 4× uncertainty, i.e. ER ≈ P |

- **Signed change T1 − T0, ER against P:** slope 1.00–1.07, correlation ≥ 0.999 in every region → ADEQUATE by the frozen label.
- **Mean change:** ER overestimates it by 9 % in the enabled region, 17 % on the floor and 11 % on the panel.

**Edit response by number of scattering events (REF-P, share of |change|).**

| Region | 1 event | 2 events | 3 events | 4 events |
|---|---|---|---|---|
| stationary pixels | 70 % | 22 % | 6.5 % | 1.3 % |
| sphere | 0 % | 77 % | 19 % | 2.7 % |
| stationary pixels outside the four change regions | 2 % | 56 % | 36 % | 5 % |

- The first row is dominated by the moving direct shadow.
- In the last row the magnitude is small: |change| = 1.5 % of T0, signed +0.4 %.

**Class accounting (REF-ER, share of |change|; receiver ← sender).**

| Region | Main cells | Any sharp interaction | Frozen label |
|---|---|---|---|
| stationary pixels | direct 0.66; B←B 0.15; B←M 0.06; B←S 0.04; S←B 0.04 | 0.085 | BROAD-DOMINATED (all-broad 0.82) |
| enabled region | direct 0.30; B←M 0.28; B←B 0.22; **B←S 0.20** | 0.20 | MIXED |
| sphere | M←B 0.40; M←M 0.32; **M←S 0.25** | 0.27 | SHARP-INDISPENSABLE |
| panel | **S←B 0.56**; M←B 0.35 | 0.60 | SHARP-INDISPENSABLE |
| stationary rest | B←B 0.60; B←M 0.14; B←S 0.09 | 0.13 | MIXED |

Split-half noise floors of these shares are ≤ 0.014 in every cell (≤ 0.041 for B←B in the stationary rest).

**Dirty accounting.**
- Paths with a dirty segment: 20–21 %.
- Segment work at or after the first dirty segment: 21–23 %.
- Dirty contributions carry 5 % of the T1 radiance over the stationary pixels: created 4.8 %, destroyed 10.2 % of T0.

**Economics (plain REF-ER, 512², RTX 3080 Ti, uncontended).**

| Quantity | Value |
|---|---|
| Kernel GPU time | 0.73 ms per spp (46.9 ms per 64 spp); ∝ pixel count (12.0 / 47.0 / 178 ms at 256² / 512² / 1024²) |
| Wall, 64 spp | 0.25 s, of which Python tracing 0.18 s |
| Wall, 1024 spp | 3.97 s with 32-spp launches (tracing 71 %); 1.53 s with 128-spp launches |
| Error vs pooled REF-ER (stationary), spp 1 / 4 / 16 / 64 / 256 / 1024 / 4096 | 0.229 / 0.145 / 0.081 / 0.045 / 0.0245 / 0.0124 / 0.0064 (replicate spread < 1 %) |
| Enabled region error at 1024 / 4096 spp | 0.040 / 0.021 (3.3× the stationary error) |
| Cost to reach error 0.05 | 37 ms GPU / 0.22 s wall |
| Cost to reach g = 0.017 | **387 ms GPU / 2.06 s wall** |
| Cost to reach 0.01 | 1.17 s GPU / 6.2 s wall |
| Cost to reach g / 2 | 1.61 s GPU / 8.5 s wall |
| Scene build | 0.16 s |
| Edit, in-place vertex update (post-chain) | 2.5 ms, then 256 spp in 0.97 s wall / 185 ms GPU, equal to a rebuilt scene to rounding |
| Kernel compilation when not cached | about 36 s, once |
| Second view | same cost as the first (44 ms per 64 spp) |
| Instrumentation overhead | 4.0× GPU |
| REF-P | 2.9× REF-ER GPU |
| Mitsuba megakernel | 40.4 ms per 64 spp, same error |
| Memory | 0.7–1.1 GB plain; 2.5 GB instrumented |

### Observation

1. **The reference pair is accurate and well resolved on this fixture.** ER tracks the physical signed change closely (slope ≤ 1.07). Its largest absolute gaps sit on the sharp panel and in the reflected patch.
2. **Most of the stationary edit response by magnitude is direct light** (the moving shadow). The indirect response is smaller, but it is where sharp and moderate senders matter: half of the indirect change in the enabled patch has a sharp or moderate sender, and on the sphere a quarter has a sharp sender.
3. **Change reaches almost every stationary pixel** through broad two- and three-event paths, at a magnitude (1.5 % of T0) comparable to the ER gap itself.
4. **The fitted closures are not uniformly broad or sharp.** The hard population is moderate-lobe dominated with almost no Lambert weight. The iid population is about half broad.
5. **Path tracing with frozen ER is fast here.** The ER-gap quality level costs 0.39 s of GPU time. A rigid edit costs milliseconds to apply. In this harness Python tracing is 70 % of wall time.

### Interpretation (inference; formally gated by the UNSUITABLE status)

- **Three directional requirements.** A GM1 mechanism on this workload would have to carry:
  - (a) exact direct visibility change;
  - (b) broad-to-broad multi-bounce change of small magnitude over the whole scene;
  - (c) a sharp / moderate-sender family that is local in space (patch, sphere, panel) but not negligible there.
- **Broad-lobe transitions dominate the frame-level response** (0.82). They do not dominate in the regions that define the GM1 phenomenon.
- **Cost budget.** To be useful at the ER-gap quality level on this fixture, a reuse mechanism would have to beat about 0.4 s of GPU work per frame (2 s wall in this harness) with a millisecond-scale edit. The dirty work fraction of about 21 % bounds what path-granular replay could skip. No mechanism is implied to achieve this.
- **WL84A assumptions:**

| Assumption | Status |
|---|---|
| A1 every material has a meaningful Lambert / broad share | **WEAKENED** (7 % of the hard set) |
| A2 energy split around r50 ≈ 0.45 | **PARTLY SUPPORTED** (half the energy is broad for iid, 4 % for hard) |
| A3 sharp-sender → rough-receiver family is material | **SUPPORTED locally** (0.20 of the enabled-region change), small at frame level |
| A4 dense change support | **SUPPORTED**, with small magnitude |
| A5 dirty fraction "a few to 15 %" | **WEAKENED** (20–21 % measured) |
| A6 cold REF-ER reaches the anchored quality in about a second | **SUPPORTED** (0.39 s GPU) |
| A7 the caustic-type family is reference-limited | **WEAKENED on this fixture** (region uncertainty 0.006 against a change of 0.105) |
| A8 Python tracing dominates wall time | **SUPPORTED** |

### Unresolved Question

- **Qualification.** The frozen qualification failed one clause. Whether the post hoc direct-clear sub-region is accepted as the qualified target is a human decision.
- **Scope of the evidence.** One fixture, one edit, one light, three ER materials, 512², fixed depth 6. Class shares depend on the declared thresholds and on these materials.
- **Per-pixel distribution of the small broad change** (fraction of pixels above a threshold) was not tabulated; `maps.npz` holds the data.
- **Runtime.** A production-runtime wall clock (without Python tracing) is unmeasured. Only kernel time separates path-tracing cost from harness cost.
- **Not measured.** Warm path-traced comparators (residual / reservoir re-rendering), other edit types and continuous motion.

### Architecture Verdict

- **By the frozen rule the workload is UNSUITABLE** (Q3b 0.465 < 0.50). Architecture-relevant conclusions are therefore **formally INDETERMINATE**.
- **The measurements themselves are valid** (validation passed, references qualified, uncontended timing). The post hoc diagnostic shows that the target phenomenon is present and resolved.
- **Sufficiency for a specific future hypothesis:** the census, the path-class accounting and the cost budget exist and are reproducible. Whether they suffice depends on the human decision about the qualification clause. No hypothesis is selected here.
- **Living documents were not changed.** For human review, if the evidence is accepted:
  - a roadmap entry for WL85 under the GM1 row;
  - the sharpness scope and workload-scale statements proposed in the WL84A audit.

### Reproducibility

- **Commits:** freeze `355a161` (chain); post-chain supplement `posthoc85.py` and this report in the closing commit.
- **Server:** `~/rest_gm1char85` (`results/renders` holds the 32 reference renders, 44 MB each; `maps.npz`; `chain.status`; `watch.out`).
- **Local copy:** `results/gm1_char85/results/`: `analysis.json`, `census/`, `tests/`, `econ/econ85.json`, `posthoc85.json`, render metadata, logs.
- **Seeds:** ER replicas 85100–85107; P 85200–85207; ladder 85500+; census 85010.
- **Environment:** RTX 3080 Ti, driver 595.71.05, Mitsuba 3.5.1 `cuda_ad_rgb`, Dr.Jit 0.4.6, torch 2.3.1.
- **Visuals:** `results/evaluation/85/` (manifest with sha256): census, references, signed change, depth shares, class accounting, economics.
