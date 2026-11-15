# Worklog — WL96A: Learned Higher-Order Transport Utility Ceiling (2026-11-16)

## Origin and scope

- **Brief.** "WL96A — Learned Higher-Order Transport Utility Ceiling" (user). A suffix of WL96: the dependent, learning-free feasibility test of the one learned element WL96 left standing (a broad ≥ 3-event resolvent behind a physically correct sharp prefix). It closely follows WL96 §F, restricted to CK2.
- **Question.** Given a fixed, physically accurate transport prefix, is there meaningful representability or lifecycle-cost headroom for a learned higher-order ECRO operator over explicit recurrence? Accuracy and runtime headroom are kept separate. A positive result would not have shown trainability or generalization.
- **Preserved.** ER and its convention; `pt86`, `pt86a`, `ecro86`, `model86`, `quad87`, `xport87`, `mat94`, `an94` (imported unchanged); every checkpoint (OWN s8601 / s8602 read-only); the explicit comparator EXP (deployed realization 86001, J = 4); the CK2 geometry, materials (`none`), light, camera and masks; the CKF_none field reference and the REF_ER_CK_none images; WL86–WL96 verdicts. **WL96's NO_VIABLE_LEARNED_MECHANISM verdict stays as written.** This batch adds evidence on CK2 only.
- **Not done.** No training, tuning, new sharp carrier, rank change, new evaluation light or geometry, renormalisation, production-path change or Architecture change.
- **Code.** [experiments/ecro_util96a/](../../experiments/ecro_util96a/): `w96a.py`, `run96a.sh`, and the post-hoc figure script `figw3_posthoc.py` (figure only).
- **Protocol.** [protocol/ecro_util96a.json](../../experiments/ecro_util96a/protocol/ecro_util96a.json). Freeze commit `a1718f7`; protocol sha256 prefix 0a5936504fb8c345.
- **Where.**
  - LabServer63 `~/rest_x96a`: one detached chain, 2026-11-15 17:05:02–17:09:01 UTC (4 min).
  - Local results: `results/ecro_util96a/`.
  - Figures: `results/evaluation/96A/` (with `manifest.json`).
- **Workload.** CK2 `none` only: one point light grazing the S1 panel; the ROI is the panel reflecting box A. No secondary workload was used. The protocol allowed one only to validate a positive result, and there was none.

## Event families (verified; Stage 0)

`pt86a.field_src` traces from the receiver. k = 1 is the receiver point r (the table patch). k = 2 is the vertex s1 adjacent to r on the light side. NEE at vertex k gives exactly k scattering events. The light-order path is light → … → s2 → s1 → r → out.

| family | predicate |
|---|---|
| prefix e2 = e2_p + e2_o | exactly 2 events (light → s1 → r); path-traced reference |
| tail e3 = e3_p + e3_o | 3–5 events (kmax = `pt86.D` − 1 = 5); the higher-order term |
| e3_p | tail **and s1 is the panel**: the *last sender* into r is sharp. The tag is set at k = 2, the vertex adjacent to the receiver, **not** at the first light-side event |
| e3_o | tail and s1 is not the panel. It still contains paths with the panel at s2, s3, …, and with the moderately glossy floor or sphere at s1; the existing tag cannot separate these |
| receiver class | a separate predicate: r on the panel (S1, 42 sharp-reader patches) vs every other receiver; reported as a region, not a family |

**Declared resolvent domain.** The learned resolvent owns e3_o at every receiver. Its input is the exact e2 field of all patches. e3_p belongs to an explicit sharp relay and is taken from the reference in the domain-isolating arms.

**Non-overlap.** The prefix (k = 2) and the tail (k ≥ 3) are disjoint by event count; e3_p and e3_o are disjoint by the s1 predicate.

**The existing learned global term.** WL92A's OWN G was supervised on L3+ = e3_p + e3_o, with ECRO's own source encoding as input. It is therefore compared with the whole tail and is descriptive only.

## A. Implementation fact

- **Explicit tail.** The reference prefix L2 = e2 is propagated with the unchanged `xport87.XOp`: L3 = T L2, L4 = T L3, L5 = T L4. The last gather is split by the material of its hit patch (= s1): e3_p_exp = T_p(L2 + L3 + L4) and e3_o_exp = T_o(L2 + L3 + L4).
  - Deployed state realization: 86001.
  - Pooled: the mean over 16 independent realizations (state offsets 960000 + d; quadrature seed 87001 + 1000(d + 1)). This is the estimator's expectation (binning and quadrature bias without state-ray variance).
- **Render arms.** Every render table is the reference e2 (mean of halves) plus a tail. Rendering uses `mat94.render`: 4 reps × 4,096 spp per arm, seeds 86600 + r, the same camera random numbers in every arm (paired).

| arm | tail |
|---|---|
| F | reference e3 (= the WL95A oracle-table floor) |
| Fa / Fb | e3 of reference half a / b (reference-tail noise) |
| Z | none (e2 only) |
| XD / XP | explicit e3, deployed / pooled |
| XOD / XOP | reference e3_p + explicit e3_o (only the resolvent domain is explicit) |
| XSD / XSP | explicit e3_p + reference e3_o (sharp last sender; descriptive) |
| G8601 / G8602 | OWN learned global term (descriptive) |

- **Stage 0 (all PASS; `s0.json`).**
  - **T_PMAT.** The material-map digest equals the reference's (3d894e73cef701a4).
  - **T_KMAX.** D − 1 = 5 and J = 4.
  - **T_SPLIT_FIELD.** The installed `field_src` parts sum to the unsplit total at the same seed: rel L1 9.8e-8 (float-atomic floor), all parts ≥ 0.
  - **T_SPLIT_TAG.** A source check confirms that the tag is set once at k == 2 and that e2 parts take only k == 2 contributions and e3 parts only k ≥ 3.
  - **T_REF_RECON.** The reference families sum to the WL95A floor table (2.1e-8).
  - **T_EXP_DEPLOYED.** The explicit state equals `build_state` bitwise.
  - **T_EXP_SPLIT.** T_p + T_o = T and T(L2 + L3 + L4) = L3 + L4 + L5 to 2.2e-7; linearity holds to 2.2e-7; outputs are non-negative.
  - **T_READOUT.** masks94 is identical to WL95A's, and a re-render of the WL95A floor table equals the WL95A file to a maximum per-pixel 1.7e-7 of the image mean.
- **Existing behaviour preserved.** No production source was edited. The diagnostics live only in the WL96A namespace.

## B. Measurement

### B1. Tail size and reference noise

- **Field.** At box A, the reference e3 exitance (e3_p 25.7 + e3_o 133.3) is as large as e2 (151.3); the tail is about 51 % of box A's table.
- **ROI image.** Dropping the tail (arm Z) changes the ROI by 0.178 of the ROI total (E_H 0.175). The higher-order image h is 0.32 of the ROI total. **The higher-order term is large at the image level.**
- **Reference-tail noise.** Image n_img = 0.5 · dI(Fa, Fb) = 0.00146 (ROI_T) and 0.0010 (DIFF). Field: reference half-split n0 for the box A e3_o table is 0.032.

### B2. Integrated ROI (paired vs F; `s1.json`)

| arm | ROI_T dI | signed | paired noise nb | E_H vs REF h (noise bound) | DIFF dI |
|---|---|---|---|---|---|
| F (floor) | 0 | 0 | 0 | 0.0118 (0.0082) | 0 |
| Fa / Fb | 0.0015 | ±0.0002 | 0.00003 | 0.0118 / 0.0120 | 0.0010 |
| Z (no tail) | 0.178 | −0.178 | 0.0003 | 0.175 | 0.171 |
| XD explicit, deployed | 0.0029 | +0.0011 | 0.00004 | 0.0119 | 0.0011 |
| XP explicit, pooled | 0.0031 | +0.0020 | 0.00003 | 0.0123 | 0.0010 |
| **XOD** explicit e3_o, deployed | **0.00065** | +0.0006 | 0.00002 | 0.0120 | 0.0011 |
| XOP explicit e3_o, pooled | 0.0006 | +0.0001 | 0.00002 | 0.0119 | 0.0010 |
| XSD / XSP explicit e3_p | 0.0027 / 0.0031 | +0.0005 / +0.0019 | 0.00003 | 0.0118 / 0.0122 | 0.0001 |
| G8601 learned (descriptive) | 0.025 | −0.019 | 0.0001 | 0.0251 | 0.0019 |
| G8602 learned (descriptive) | 0.013 | −0.009 | 0.0001 | 0.0150 | 0.0031 |

**Stage 1 decision (frozen rule).**
- Inputs: H = dI(XOD) − n_img = −0.00082; U = H + 2 nb = −0.00077; the minimum meaningful effect M_img = 0.005.
- U < M_img, so the outcome is **S1_NO_HEADROOM**.
- The conclusion does not depend on the noise subtraction: the raw dI(XOD) + 2 nb = 0.0007 is already 7× below M_img.
- Realized E_H difference: R = E_H(XOD) − E_H(F) = 0.00015 against tol_E = 0.016.
- Stage 2 (the multi-light free fit) was therefore **not triggered**. No fit targets were created.
- The predeclared extension (more field halves and render reps) was not needed.

### B3. Field level, by family and region (`tails.json`; table = W_T-weighted patch × bin × channel rel L1; n0 = reference half-split)

| arm / family / region | flux | table rel L1 (n0) → excess | exitance rel L1 (n0) → excess | corr |
|---|---|---|---|---|
| XD e3_o box A | 0.993 | 0.036 (0.032) → 0.004 | 0.015 (0.009) → 0.006 | 0.999 |
| XP e3_o box A | 0.999 | 0.035 (0.032) → 0.003, SE 0.0009 | 0.010 (0.009) → 0.0004 | 0.9995 |
| XD e3_o panel face | 1.004 | 0.060 (0.055) → 0.005 | 0.024 (0.018) → 0.006 | 0.999 |
| XD e3_o S1 (sharp readers) | 1.002 | **0.090 (0.047) → 0.043** | 0.013 (0.011) → 0.002 | 0.995 |
| XP e3_o S1 | 1.011 | 0.064 (0.047) → 0.016, SE 0.002 | 0.014 (0.011) → 0.003 | 0.994 |
| XD e3_o all | 0.996 | 0.032 (0.025) → 0.007 | 0.017 (0.005) → 0.012 | 0.998 |
| XP e3_o all | 1.001 | 0.028 (0.025) → 0.003 | 0.009 (0.005) → 0.004 | 0.999 |
| XD e3_p box A | 1.008 | 0.049 (0.035) → 0.014 | 0.030 (0.007) → 0.024 | 0.9996 |
| XP e3_p box A | 1.024 | 0.047 (0.035) → 0.012 | 0.025 (0.007) → 0.019 | 0.9999 |
| XD / XP e3_p all | 0.997 / 1.009 | 0.235 / 0.123 (0.100) | 0.206 / 0.072 (0.023) | 0.995 / 0.999 |
| G8601 e3 box A | 0.973 | 0.083 (0.029) → 0.054 | 0.063 (0.008) → 0.055 | 0.92 |
| G8602 e3 box A | 0.973 | 0.090 (0.029) → 0.061 | 0.071 (0.008) → 0.063 | 0.95 |
| G8601 / G8602 e3 panel face | 0.84 / 0.90 | 0.196 / 0.136 | 0.180 / 0.108 | 0.27 / 0.57 |

- **Cross-half check (deployed explicit).** Computed from prefix half a and scored against tail half b, the box A e3_o table rel L1 is 0.044, while half a vs half b is 0.063 (b → a: 0.044 / 0.063). The explicit tail is closer to an independent reference half than the other reference half is.
- **Per-realization spread** (16 draws; table rel L1, median / p90 / worst):
  - e3_o box A: 0.038 / 0.041 / 0.053.
  - e3_o S1: 0.096 / 0.102 / 0.106.
  - e3_p all: 0.245 / 0.258 / 0.266.
- **Field-level secondary rule** (WL96 §F bound max(2 SE, n0 / 2), reported only):
  - box A e3_o: no field headroom (0.004 deployed vs bound 0.016).
  - S1 e3_o, deployed: flagged (0.043 vs 0.024).
  - S1 e3_o, pooled: not flagged (0.016 vs 0.024).

### B4. Lifecycle cost (Stage 3; `cost_s1.0.json`)

Setup:
- Hardware: RTX 3080 Ti 12 GB, run alone (no foreign process).
- Scene: N = 1,270; frame 512² at 64 spp.
- Events: 6 timing-only rigid edits (box A ± 0.15 m in y) and 16 source updates.
- Timing: medians of warm events; no timed event compiled; parts sum / wall = 1.00 in every arm.

| arm | edit → frame, ms | source → frame, ms | U edit | U source |
|---|---|---|---|---|
| R0 explicit (q + prefix order + 3 tail orders) | 144.1 | 88.5 | 1 | 1 |
| LAM24 (prefix order + rank-24 Λ; edit adds the ECRO-class generator) | 154.6 | 85.1 | **0.93** | **1.040** |
| LAM64 | 156.5 | 85.3 | **0.92** | **1.038** |
| ZERO (Λ and generator free: Amdahl ceiling) | 138.9 | 84.9 | 1.038 | 1.043 |

**R0 breakdown, edit:**
- geometry update 30.2 ms;
- relation update 18.1 ms (49,936 of 650,240 relation rays re-traced);
- relation coefficients 5.0 ms;
- q table 25.7 ms (20,320 shadow rays per light);
- prefix order 2.0 ms;
- **tail orders 3.3 ms (2.3 % of the edit)**;
- hand-over 0.2 ms;
- render 59.5 ms (nominal 67 M rays).

**R0 breakdown, source:** q 24.4, prefix 2.1, tail 3.3 (3.7 %), render 58.5 ms.

**Isolated component medians (20 warm reps):**
- 3 tail orders 3.29 ms; one order 1.14 ms;
- Λ24 0.16 ms; Λ64 0.24 ms;
- generator (OWN `factors`) 5.3 ms; ECRO apply 0.77 ms;
- q table 24.2 ms; frame 57.3 ms.

**Memory and CPU:**
- process GPU memory 1,452 MiB; torch peak 251 / 355 / 350 / 306 MiB (R0 / LAM24 / LAM64 / ZERO).
- Edit CPU time: R0 149 ms vs LAM 580–727 ms. CPU time exceeds wall time there, i.e. the generator path runs multi-threaded CPU work.

**Results:**
- **COST criterion (U ≥ 1.25) not met.**
- The scaling extension (rule: U(ZERO) ≥ 1.15) was not triggered.
- Without the frame (descriptive only; not a user-visible update), the ceilings are 1.05× edit and 1.12× source.

**Verdict:** `final.json`: **LEARNED_RESOLVENT_UTILITY = NO_MEANINGFUL_HEADROOM**.

## C. Observation

- **Where explicit recurrence succeeds.**
  - The tail is large: 0.178 of the ROI.
  - In the resolvent's own domain (e3_o), the explicit binned recurrence from the physical prefix is at the reference noise level. Field excess over n0 at box A is 0.003–0.004 (flux 0.99–1.00, corr ≥ 0.999). The ROI change is 0.0006–0.0007, below the reference tail's own noise (0.0015).
  - Pooling 16 realizations changes almost nothing (0.00065 → 0.0006). The deployed draw is neither noticeably biased nor noisy for the broad tail.
- **Where explicit recurrence is weaker.**
  - **Sharp-last tail (e3_p).** The binned panel table as the last sender carries most of the explicit tail's image change (XSD / XSP 0.0027 / 0.0031 vs XOD 0.0006), and box A exitance excess 0.019–0.024. Its flux stays 1.01–1.02, unlike the 1.5× over-delivery of the grazing e2_p in WL95A: the panel's e2 outgoing field is lit by diffuse room light and is not a grazing point-light lobe.
  - The e3_p table excess over all receivers is large for the deployed draw (0.135) and mostly state-ray variance (pooled 0.023).
  - All of this is outside the resolvent's domain and still below M_img.
  - The panel receivers' own e3_o tables (sharp readers) show a deployed-draw table excess of 0.043, mostly variance (pooled 0.016). Their exitance is accurate (0.002). The ROI_T and DIFF masks do not read those tables visibly (DIFF dI ≤ 0.0011).
- **Where the optimistic learned fit changes the result.** Not measured. Stage 2 was correctly not run: with no headroom in the explicit arm, a perfect free fit could improve the image by at most U < 0.
- **The existing learned global term** (descriptive; different input and ownership) is 9–40× further from the floor than explicit e3_o (ROI dI 0.025 / 0.013, under-delivering, signed −0.019 / −0.009). It also has field errors at box A (excess 0.054 / 0.061) and on the panel-facing face (flux 0.84 / 0.90, corr 0.27 / 0.57).
- **Cost.**
  - The replaceable recurrence is 3.3 ms of a 144 ms edit or an 88.5 ms source update.
  - Λ application is 0.16–0.24 ms. The geometry-conditioned generator alone (5.3 ms isolated, 8.4 ms in-lifecycle) costs more than the recurrence it replaces. In the edit lifecycle the q table also rose from 25.7 to 31–33 ms when the generator ran; this is unattributed. Excluding it, LAM edit would still be ≈ 148–150 ms against R0's 144.
  - The q table (24 ms) and the frame (58 ms) dominate every lifecycle.

## D. Interpretation

**What is supported (CK2, one light, this ROI).**
- **No accuracy headroom.** A perfectly accurate broad-last resolvent fed the same physical prefix could not improve the ROI measurably over three explicit 64-bin sweeps on the deployed state. The ceiling of the improvement is below the reference tail's own noise and 7× below the predeclared minimum meaningful effect.
- **No lifecycle headroom.** A free, perfect replacement would speed the lifecycle by at most 1.04×. Any ECRO-class generator makes edits slower (0.92–0.93×). Source updates gain at most 1.04×.
- These measurements confirm, on CK2 and the same GPU, WL96's cross-workload predictions (accuracy headroom about 0.001; Amdahl ceiling 1.09× / 1.16× from WL91's GM1 timings; CK2's 3.3 ms tail is a smaller share still).

**The four claims kept apart.**
1. **Representability of the optimistic free fit: not tested and not needed.** With no headroom over explicit, no representation can buy anything on this workload.
2. **Trainability:** not addressed.
3. **Runtime competitiveness:** measured. It is absent at N = 1,270. The rule did not authorise a scaling sweep. WL91's GM1 scaling shows the tail orders growing slower than the q table and relation build (C3UR table 11.2 → 18.9 ms from N 1,270 → 5,080), so no scaling hypothesis reaches 1.25×.
4. **Generalization beyond CK2:** not tested.

**Remaining confounds.**
- **Scope.** One workload: one grazing point light, Lambert box A as the ROI reader, a fixed panel. Other lights were not referenced.
- **Prefix.** The prefix is the path-traced oracle, which isolates the tail question by design. A deployable sharp prefix carrier adds its own error upstream; that error belongs to the prefix (WL95A / WL96), not to the resolvent.
- **Untagged predicate.** "Panel at s ≥ 2" sits inside e3_o and cannot be separated with the existing tag. The resolvent domain therefore includes some sharp structure in the middle of the chain. The explicit tail handles it within noise here.
- **Sharp-reader tables.** The panel's own e3_o tables have a deployed-draw variance excess that the measured image regions do not expose. A view that reads the panel's outgoing table directly (camera → diffuse x1 → panel x2) is not covered.
- **Timing-only edit.** The edit (box A ± 0.15 m) was not a reference-validated geometry change. The frame convention (512², 64 spp) sets the lifecycle denominator; even with no frame, the ceilings are 1.05× / 1.12×.
- **Generator CPU time.** The CPU time on the generator path (580–727 ms of CPU per 155 ms edit) is unexplained. It does not change the ordering.

**Why the result does not justify learning.** The function a learned higher-order resolvent would own is already computed at reference-noise accuracy by about 3 ms of explicit sweeps on the same state. There is nothing for learning to recover in accuracy. There is also nothing to amortise in cost, because the sweeps are not the lifecycle's bottleneck, and a geometry-conditioned generator costs more than the sweeps.

## E. Architecture verdict

**LEARNED_RESOLVENT_UTILITY = NO_MEANINGFUL_HEADROOM** (CK2 `none`; frozen mapping; Stage 1 S1_NO_HEADROOM, Stage 2 not triggered, COST criterion not met).

- Headroom was absent, so no viability claim follows either way.
- WL96's verdict (NO_VIABLE_LEARNED_MECHANISM, scoped and provisional) is unchanged. On CK2 its dissent (EVIDENCE_INSUFFICIENT: "the resolvent's utility is unmeasured, image-level relevance unresolved") is now answered *for the broad-last resolvent*: utility measured, none found.
- The sharp-prefix relevance part of WL96 §F (oracle vs binned e2_p at higher spp) was **not** part of this brief and remains open.
- Nothing is adopted. No architecture document changes. No successor is authorised.

## F. Reproducibility

- **Commands** (LabServer63, env `~/rest_x96a/env.sh`, Mitsuba `cuda_ad_rgb`, OptiX 5.9.5 libs):
  - `setsid nohup bash code/experiments/ecro_util96a/run96a.sh ~/rest_x96a ~/rest_x94 ~/rest_x95a ~/rest_x92a > chain.out 2>&1 &`
  - Code deployed by `git -c core.autocrlf=false -c core.eol=lf archive` at `a1718f7` (0 CRLF files).
- **Inputs (sha256 prefix):**
  - `CKF_none.npz` 3a6238a8a42d1db4 (seeds 94301 / 94302, 2 × 4,096 spp);
  - WL95A `T95A.npz` d009c57b018e9f39;
  - `FZ_own_s8601.npz` 44fade056130ec75 and `FZ_own_s8602.npz` 03183dd2e6e013dc;
  - `masks94.npz` b9a7d46b1a23d8c4;
  - `REF_ER_CK_none_CK2_T1_r0..r3` (4 × 8,192 spp; WL94).
- **Seeds:**
  - explicit realizations 86001 (deployed) and 960000 + d (pooled), quadrature seed 87001 (+ 1000(d + 1));
  - renders 86600 + r;
  - Stage 0 field test 96001;
  - lifecycle sources rng 96500; random Λ factors 96600.
- **Outputs** (`results/ecro_util96a/`, sha256 prefix):
  - `s0.json` aabaddad35eee08b;
  - `tails.json` 395f0166deedeb5b; `tails.npz` 04da33a8b57eeb32;
  - `T96A.npz` 8019cb39d03421dd;
  - `s1.json` 92f5e763126d83b6;
  - `cost_s1.0.json` 103b775ed571ed26;
  - `final.json` 1c625581262250e4;
  - `chain.status`, `gpu.txt`.
- **Figures** (`results/evaluation/96A/`, hashes in `manifest.json`):
  - figW1 (ROI tail arms and |arm − F| on one absolute sqrt scale);
  - figW2 (box A tail exitance);
  - figW3 (lifecycle; colour-inconsistent chain version, kept) and figW3_lifecycle_posthoc (one colour per component, figure-only fix).
- **Runtime:**
  - Stage 0 25 s; tails 24 s (17 realizations × ≈ 1.1 s).
  - 48 renders at pool 4: 2.6 min, GPU 35–75 %.
  - Lifecycle 19 s alone, GPU 18–26 % (CPU-bound phases of the pipeline).
  - Total chain 4 min.
- **Full-scale probe (before the freeze; GPU utilisation rule).**
  - Utilisation by stage: tails 7 %, one render 23 %, one extension field half 3 % (CPU-bound accumulation), Stage 2 fit 98 %, lifecycle 15 %.
  - Response: pool sizes raised (renders 4, extension halves 3, Stage 2 references 4).
  - Not optimised: the short stages (≤ 30 s).

## Disclosures and deviations

- **Pre-freeze exposure** (recorded in the protocol before the freeze).
  - A dev chain (`~/rest_x96adev`, `--dev`) and a full-scale timing probe (`~/rest_x96aprobe`) ran before the freeze. Outcome lines were suppressed; only pass flags, walls and GPU utilisation were read.
  - **The dev figure figW1 was opened to check the layout.** The dev tails use the real CKF reference and the real explicit operator; only the render spp was reduced. The figure therefore showed qualitatively that |XOD − F| and |XOP − F| were the smallest tail differences, and that G and Z were larger. No number was read.
  - All criteria were written before any WL96A code ran and were not changed.
- **Post-dev fixes before the freeze:** a module import order, numpy-bool JSON serialisation, and pool sizes.
- **No amendment.**
- **Post-hoc, figure only:** figW3 colours (`figw3_posthoc.py`).
- **The full-run `T_READOUT` record** stores the masks check under a key containing "sha". The pass flag includes it (masks identical).
