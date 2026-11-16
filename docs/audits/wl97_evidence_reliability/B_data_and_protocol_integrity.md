# WL97 Track B - Data and Experimental Protocol Audit (ECRO lineage, WL86 - WL96A)

Independent reviewer report, 2026-11-16. Track B (data and experimental protocol). No other track's report or directory was opened.
This run resumed an interrupted first attempt of the same track; every number of the first attempt that is used below was either
recomputed in this run or spot-checked against an independent source (section 8). Nothing in the repository or on the server was
modified outside `docs/audits/wl97_evidence_reliability/`, `experiments/audit97/B/`, `results/audit97/B/` and `~/rest_x97/B/`.

Labels used below: **[MEAS]** = measured / recomputed by this audit; **[REPLAY]** = read from a stored result file of the audited
batch (not recomputed); **[CODE]** = read from frozen source; **[INF]** = this auditor's inference.

---

## 1. Scope and method

Questions (from the track brief): (1) dataset generation, selection, splits and leakage; (2) material / light distribution and rare
sharp-event coverage; (3) checkpoint and configuration identity; (4) frozen thresholds and stop conditions versus what was executed;
(5) reference / oracle construction and independence; (6) post hoc selection and seed counts. Deliverables: a per-batch
protocol-integrity table (section 5) and a training-data coverage assessment tied to P0-1 and P0-5 (sections 3, 6).

Method. Each assessed claim was traced protocol -> code -> inputs -> raw numbers. Worklog prose was used only to locate things.
Independent work of this track:

| # | Check | How | Output |
|---|---|---|---|
| a | Training population identity | regenerated all WL86 splits from the frozen sampler (`scene86.py`, numpy) and compared digests with the geometry / light JSON stored inside the server `.npz` files | `population_regen.json`, `~/rest_x97/B/population_server.json` |
| b | Held-out disjointness | distance of every later evaluation condition (E1, E2, E3a, E4 = E3b, E3c, E5k995, CK2) to every population that trained a checkpoint (P86 1,024 files; P92A 928 files; P94 96 files) under the WL86 rule (box centre < 0.15 m, light < 0.3 m) | `heldout_disjointness.json` |
| c | Sharp-event coverage | independent numpy re-implementation of panel direct irradiance per incident ring (deployed 16 source points, rng 86001) and an ideal-mirror light -> panel -> box proxy, for all 4,096 training lightings and CK2 | `panel_coverage.json` |
| d | Learned S1 passivity vs coverage | server, read-only: directional albedo of the WL92A `own_s8601 / own_s8602` S1 local operator on the CK2 state and on 48 stored training states; per-incident-bin training irradiance share from the 928 x 4 stored source encodings | `learned_s1_albedo.json` |
| e | WL95A flight verdict sensitivity | recomputed the frozen rule from `deliv.npz` at 16 / 32 / 64 draws and with alternative E0; single-draw vs pooled error per arm | `wl95a_flight_sensitivity.json` + section 4 table |
| f | Rank-16 capacity vs allocation | row-weighted non-negative rank-16 fits of the S1 M64 operator under training / CK2 / mixed incidence weights | `rank16_allocation.json` |
| g | Checkpoint identity | sha256 of every `.pt` in every batch run directory (server), symlink chains | section 5.2 |
| h | Freeze integrity | `git diff --stat <freeze> <completion>` per batch; deployed server code hash vs frozen blob (CRLF-normalised) | section 5.1 |

Not done (by rule): no training, no rendering, no re-run of any production chain.

---

## 2. Summary of conclusions

1. **Data integrity is sound.** The WL86 population is exactly reproducible from the frozen sampler and equals what was stored and
   trained on. No train / validation content overlap. No validation-based model selection exists in any training script.
2. **Held-out status is weaker than the labels suggest, but the leakage direction cannot explain the negative results.**
   CK2 was *not* excluded from the population that trained the WL92A OWN checkpoints (the checkpoints judged on CK2 in WL93D-add.,
   WL94 gate, WL94B-D, WL95A, WL96A); the CK2 rule was applied only to the 96-item WL94 candidates. 17 / 928 files violate the
   single-condition rule, none is jointly close. E3a was likewise not excluded from the frozen WL86 checkpoints. Any leakage would
   favour the learned operator, and the conclusions drawn there are unfavourable to it.
3. **CK2 is a single, hand-designed, edge-of-support workload** (box face at the minimum admissible panel distance, light 5 cm
   from the panel plane, chosen by render iteration in a qualitative addendum) that later carried every quantitative P0 claim.
   CK2-like grazing panel -> box delivery occurs in about 0.1 % of training lightings (4 / 3,712); the CK2 incident bins carry
   2.0 % of training panel irradiance; the panel-mediated share of the training loss mass is 3.4 %.
4. **Coverage is a real confound for "the learned sender fails", not for "rank-16 cannot represent the lobe".** A light-independent
   non-negative rank-16 S1 operator loses the CK2 lobe under any allocation that still serves the rest of the incidence domain
   (this audit's sweep). Conversely the learned operator's non-passivity is *not* a rare-bin artefact: it is present on every
   training state and on incident bins that carry 52 - 62 % of training panel irradiance.
5. **The coverage hypothesis was never actually tested by training.** WL94B stopped before training on a validity gate that the
   protocol itself had noted (before the freeze) to be algebraically tighter than its companion gate; the dev subset had already
   shown the failure. Every later retraining batch is a 1,000 - 2,000-step smoke fine-tune on 96 - 128 geometries from co-adapted
   WL92A checkpoints with two lineage seeds that disagree.
6. **Protocol discipline is generally good and well disclosed** (freeze commits real, deployed code = frozen blobs, no silent
   threshold changes found). The bias risks are of a different kind: stop gates that systematically prevent the positive test
   (WL92 G_rep, WL92A E4, WL93 preflight, WL94B G_EXPOSURE), pre-freeze feasibility revisions (WL94C matching, WL94D G_PHYS),
   label-mapping gaps (WL92 code 1, WL92B outcome C vs D, WL94A seed), and a deployed-draw reference point in WL95A that turns
   out to be an atypically good draw.
7. **P0-4 raw decision is robust** to its disclosed dev-figure exposure and to the M_img choice (margin about 7x; the
   threshold is inherited from four earlier protocols).

---

## 3. Data findings

### B-01 (confirmation, not material) - population identity
- [MEAS] Local regeneration from `experiments/ecro86/scene86.py` (sha256 b75a2d9c...) gives digests train 1,024 `1d33827645edbe3a`,
  val 64 `c45e44babacc5eda`, trainhq 32 `e928f3e3e47b2cb8`; the geometry / light JSON stored in `~/rest_ecro86/data/*/g*.npz`
  gives the identical three digests. Re-running the regeneration and the disjointness script in this run reproduced the first
  attempt's files bit-for-bit.
- [CODE] `train86.py:66-109`: fixed step count, seed from `torch.manual_seed(seed)`, validation is logged only; no early stopping,
  no checkpoint selection. Same in `own92a.py`, `c94.py`, `s94.py`.
- [MEAS] Deployed server code of WL86 equals the frozen blobs after CRLF normalisation (the server copy has CRLF line endings;
  `scene86 b75a2d9c`, `train86 340180b3`, `model86 72a728b8`).
- Verdict: VERIFIED. Confidence high.

### B-02 (dataset_limitation / provenance_gap, material for wording, not for direction) - CK2 not excluded from the OWN checkpoints
- Claim affected: every statement that treats CK2 as "held-out" for `own_s8601 / own_s8602` (WL93D addendum, WL94 gate and
  frozen-own columns, WL94B T_REPRO baseline, WL94C/D "own" reference level, WL95A arm A0 and G, WL96A G8601 / G8602).
- [CODE] `ecro_mat94.json /stage_C/training_population`: the CK2 rule is applied to "the first 96 WL86 train files kept by the
  WL92A exclusion AND the CK2 rule" - i.e. to candidates A / B only. `results/ecro_own92a/results/exclusion.json` (rule text)
  lists only E4 positions. CK2 did not exist when WL92A trained (2026-11-13; CK2 was designed in the WL93D addendum).
- [MEAS] `heldout_disjointness.json`, CK2 vs P92A (928 files): 4 box violations + 13 light violations = **17 files**; min box
  distance 0.095 m, min light distance 0.105 m. All 4 box violations are box **B** near CK2's out-of-view box B (their box A is
  1.9 - 3.1 m from CK2's box A). No file is jointly close: the nearest training geometry in the joint (A, B) sense is 0.94 m
  away; the nearest box-A centre is 0.27 m. The closest analogue is `g01017` light 3: 0.145 m from the CK2 light, box A 0.48 m
  from CK2's, ring-0 mirror-to-box proxy 8.4 vs CK2 17.5.
- For comparison the properly excluded conditions sit exactly at the margins: E1 / E2 vs P86 min box 0.154 / 0.155 m, min light
  0.3005 / 0.3034 m; E4 vs P92A 0.153 / 0.302 m (0 violations each).
- Decision-level impact: none on direction. Leakage would help the learned operator on CK2; the CK2 conclusions are that it is
  poor. Impact on wording: CK2 is "not a training item", not "excluded by the frozen margin rule", for OWN.
- Proposed verdict for the held-out claim: CONDITIONALLY_VALID. Confidence high.

### B-03 (dataset_limitation, not material) - E3 not excluded from the frozen WL86 checkpoints
- [MEAS] E3a (the WL87 "untouched condition") vs P86: 25 files violate (1 box at 0.129 m, 24 lights, min 0.089 m). [CODE]
  `scene86.near_test_geom / sample_light` exclude only the conditions present in the WL86 fixture (E1, E2 and their light sets).
- [MEAS] E4 (= WL87 candidate E3b) vs P86: 96 files violate; WL92A removed exactly these (928 kept, re-derived) before training
  SUM / OWN. E5k995: 0 violations against P86 and P92A.
- Impact: WL87 compared explicit transport against the frozen ECRO on E3 and found explicit >= ECRO; leakage again points the
  other way. "Untouched" in WL87 means "no radiance inspected", not "outside the training margin".
- Verdict: CONDITIONALLY_VALID. Confidence high.

### B-04 (dataset_limitation, MATERIAL for P0-1 / P0-5) - rare sharp-event coverage
Verification of the WL94B / WL95 numbers and their extension to the population that actually trained OWN:

| Quantity | Original | This audit | Source |
|---|---|---|---|
| items reaching CK2's panel-mediated box-A share (PMb_A 0.138) | 4 / 96 | 4 / 96 [REPLAY from rows]: g00032 0.252, g00058 0.142, g00081 0.173, g00084 0.140 | `results/ecro_exp94b/audit.json` |
| panel-mediated share of loss mass | 3.4 % | e2_p 2.11 % + e3_p 1.32 % = 3.43 % [REPLAY] | same |
| ring-0 share of training panel irradiance | 4.3 % | P94 (96 files): 4.32 % [MEAS, geometric]; **P92A (928 files): 4.68 %** [MEAS, geometric] and 4.68 % [MEAS, stored encodings]; P86: 4.75 % | `panel_coverage.json`, `learned_s1_albedo.json` |
| share of training panel irradiance in the 4 CK2-lit incident bins (12-15) | not reported (WL94D: "all 64 bins excited, none outside training") | **2.0 %** [MEAS] | `learned_s1_albedo.json` |
| lightings whose lit panel points are all ring 0 | - | 163 / 3,712 (P92A) [MEAS] | `panel_coverage.json` |
| lightings with ring-0 mirror-to-box proxy >= CK2 / >= half / >= tenth | - | **4 / 10 / 27 of 3,712** (P92A); 0 / 1 / 1 of 384 (P94) [MEAS] | same |
| ring-0 mirror-to-box share of all panel irradiance | - | 0.027 % (P92A), 0.011 % (P94) [MEAS] | same |
| boxes with a face within 0.2 m of the panel plane and inside the panel extent | - | 6 / 928 (box A), 12 / 928 (box B) [MEAS] | first-attempt log, recomputable from `population_regen.json` |

- Notes. (i) The WL95 "4.3 %" was computed on the 96-file WL94 population, not on the 928-file population that trained OWN; the
  difference (4.3 vs 4.7 %) is immaterial. (ii) WL94B's item criterion (PMb_A) is incidence-agnostic; under the grazing-specific
  proxy the WL94 population contains no CK2-level item and one at half level. Spearman between my geometric proxy and WL94B's
  path-traced PMb_A is 0.87, so the two measures agree in ranking. (iii) [CODE] `scene86.light_ok` enforces a 0.15 m clearance
  from the sphere and the boxes but none from the panel or walls: 32 / 4,096 training lights are within 0.10 m of the panel
  rectangle and 111 are behind the panel plane. The ring-0-dominated training lightings are therefore largely near-field /
  heavy-tailed cases (this is also what broke the WL93 identifiability preflight: validation `g00010` has a light 7 mm from the
  panel), not CK2-like grazing illumination that delivers to a nearby box.
- [REPLAY] `audit.json /semantics`: the S1 once-scattered response is supervised only through receivers; no sender term exists in
  the WL86 / WL92A objective.
- Decision-level impact: the learned S1 sender was judged on a transport configuration that the training distribution contains
  at the 10^-3 level and never supervises directly. "The learned sender has large error on CK2" is measured; "the learned
  mechanism cannot learn this" is not tested (see B-09).
- Verdict on the coverage statements of WL94B / WL95: VERIFIED (numbers) with the population caveat. Confidence high for counts,
  medium for the mirror proxy (ignores lobe width and Fresnel).

### B-05 (confirmation + qualification, MATERIAL for P0-1) - learned S1 non-passivity is not a coverage artefact
- [MEAS] Reproduction: with `own_s8601 / own_s8602` loaded read-only from `~/rest_x92a` and the CK2 state built by `mat94.ctx_ck2`,
  the S1 local-operator max directional albedo is **2.263723684963198 / 3.077449703949969** - identical to every digit of
  `results/ecro_attr95a/main.json` (A0_s8601 / A0_s8602 albedo_max). This also proves that the checkpoint files WL95A used are
  the files hashed today (739e7eb5... / cee9a09a...).
- [MEAS] Same convention applied to the physical operator: M64 (S1) max albedo 0.825 (`rank16_allocation.json`; WL94D T_ENERGY
  says 0.825). So the convention is not what produces values above 1.
- [MEAS] Beyond CK2: on 48 stored WL86 training states the pooled max is 2.67 / 3.59 and **48 / 48 states are non-passive** for
  both seeds (per-state max 2.1 - 2.5 / 2.7 - 3.6). Fraction of (patch, bin, channel) entries above 1: 10.7 % / 15.0 %.
- [MEAS] Relation to coverage: incident bins whose max albedo exceeds 1 carry **61.9 % / 52.0 %** of training panel irradiance;
  Spearman(bin irradiance share, bin max albedo) = -0.09 / +0.16. Per-ring fraction of entries above 1: ring 0 17 % / 26 %,
  ring 3 4 % / 6 % - grazing rings are worse, but the violation is present in well-covered bins.
- Seed disagreement at the same operator: CK2 box-A delivered flux ratio of the learned sender is 0.52 (s8601) vs 1.63 (s8602)
  [MEAS from `deliv.npz`, 64 draws, sd 0.04 / 0.11]: the two lineages err in opposite directions.
- What this audit did not verify: the radiometric correctness of the `a95.op_metrics` albedo convention itself (W_T weights and
  the source-encoding normalisation) beyond the M64 consistency check above; and whether the non-passive local factor matters
  in deployment (the ECRO operator is applied once, not iterated).
- Verdict for "the deployed learned operator may violate passivity": VERIFIED as a measurement (local S1 factor, both seeds,
  training and CK2 states). Confidence high for the numbers, medium for the physical reading.

### B-06 (confirmation, MATERIAL for the reading of "rank-16 cannot represent") - capacity vs allocation
- Question: is WL95A's `RANK16_DELIVERY = INADEQUATE` (and the WL92 "material rank-16 loses the panel lobe") a capacity
  statement or a consequence of fitting on an incidence distribution in which the CK2 bins carry 2 %?
- [MEAS] `rank16_allocation.json`: per-channel non-negative rank-16 fits of the S1 M64 flux matrix with incident-row weights w
  (3 restarts, 4,000 multiplicative updates; CK2 field metrics as defined in WL95 / WL96):

| Row weighting | weight on CK2 bins | CK2 field rel L1 | CK2 energy | CK2 peak | training-weighted row rel L1 | albedo max |
|---|---|---|---|---|---|---|
| training incidence (P92A) | 2.0 % | 0.90 | 0.35 | 0.015 | 0.42 | 0.97 |
| uniform over bins | 6.3 % | 0.91 | 0.45 | 0.025 | 0.58 | 0.98 |
| training + 10 % CK2 | 11.8 % | 0.90 | 0.39 | 0.030 | 0.42 | 0.98 |
| training + 25 % CK2 | 26.5 % | 0.83 | 0.56 | 0.38 | 0.43 | 1.00 |
| training + 50 % CK2 | 51.0 % | 0.59 | 0.75 | 0.52 | 0.45 | 1.07 |
| CK2 only | 100 % | 1.5e-6 | 1.00 | 1.00 | 0.98 | 0.82 |
| WL95A A1 (NMF16, WL92) [REPLAY of stored operator] | - | 0.92 | 0.46 | 0.028 | 0.57 | 0.96 |
| WL95A A2 (L1-IRLS) [REPLAY of stored operator] | - | 0.89 | 0.11 | 0.002 | 0.35 | 0.81 |

- Reading [INF]. (i) My independent training-weighted fit reproduces the WL95A bound operators' CK2 behaviour (lobe peak 1.5 %
  vs 2.8 % / 0.2 %), so the stored A1 / A2 are not unlucky fits. (ii) Up-weighting CK2 bins sixfold changes nothing; the lobe
  only starts to appear when four bins receive a quarter to a half of the total weight. (iii) With all weight on CK2 the fit is
  exact (four lit bins need rank 4), but useless elsewhere. So the limit is structural for a *light-independent* rank-16
  bilinear operator in the 64-bin basis (a sharp lobe makes M64 close to a 64-wide sparse map), not a coverage artefact; it is
  not a statement that rank 16 cannot represent any single lighting.
- Limits of this check: Frobenius objective on the flux matrix, multiplicative updates (local minima possible), operator-level
  only; the trained ECRO operator is geometry-conditioned per patch and trained through receivers, which this bound does not
  model. The WL96 authors' own calculation (`results/wl96_calc/ring_alloc.json`) reaches the same reading; I reproduced its A1 /
  A2 CK2 numbers exactly from `ops95a.npz`.
- Caveat on the WL95A criterion [CODE `ecro_attr95a.json /decisions/RANK16_DELIVERY`]: adequacy is judged against arm A4 (the
  64-bin M64 sender), which is itself biased 1.5x against the path-traced reference (B-07). "INADEQUATE" therefore means
  "does not reproduce the binned exact operator", which is the right in-family question but not a statement about distance to
  physical truth.
- Verdict: the rank-16 inadequacy on CK2 is VERIFIED as a property of the tested operator family; "coverage-confounded" is
  **not supported** for this specific claim. Confidence medium-high.

### B-07 (statistical_weakness, MATERIAL for the P0-2 / P0-3 wording) - WL95A flight attribution depends on the deployed draw
[MEAS] from `results/ecro_attr95a/deliv.npz` (64 relation draws per arm, box A, reference = mean of the two 4,096-spp halves,
half-split noise n0 = 0.059):

| Arm | single-draw rel L1 (mean / median) | pooled rel L1, D = 1 / 4 / 16 / 64 | flux ratio (mean +- se) | deployed draw: rel L1 / ratio |
|---|---|---|---|---|
| A4 (M64, deployed source points) | 0.761 / 0.726 | 0.50 / 0.63 / 0.57 / 0.55 | 1.544 +- 0.035 | 0.524 / 1.208 |
| M64_spool (pooled source points) | 0.720 / 0.665 | 0.46 / 0.58 / 0.53 / 0.50 | 1.495 +- 0.035 | 0.483 / 1.158 |
| X1c (exact direction, patch centre) | 0.710 / 0.600 | 0.73 / 0.44 / 0.32 / 0.27 | 1.037 +- 0.063 | 0.388 / 0.719 |
| X1 (exact direction and hit point) | 0.737 / 0.617 | 1.08 / 0.40 / 0.28 / **0.11** | 1.022 +- 0.050 | 0.627 / 0.764 |
| A0 learned s8601 / s8602 | 0.595 / 0.974 | 0.57 / 0.97 (D = 64) | 0.517 / 1.630 | 0.588 / 0.901 |

- (i) The binned sender's bias survives pooling: ratio 1.49 - 1.54, about 14 se above 1, while the exact arms pool to 1.02 - 1.04.
  This part of P0-2 is robust.
- (ii) The deployed relation draw (seed 86001) is atypically good for A4: only 11 of 64 draws have a lower error and only 9 of 64
  a lower flux ratio (1.21 vs a draw mean of 1.54). The frozen rule sets E0 = error of that one draw, so the sampling term
  Rs = E0 - E(pooled) is 0.02 and "does not count"; with E0 = the mean (or median) single-draw error the same rule gives
  Rs = 0.26 (0.23), which counts, and the label becomes **MIXED** instead of ANGULAR_PROJECTION_EVIDENCED
  (`wl95a_flight_sensitivity.json`). Sampling variance is about one third of a typical single-draw error.
- (iii) The 16 -> 64 extension followed the predeclared 2-SE rule ([REPLAY] `N1.log`: `AN EXTEND`) and did not change the label:
  my recomputation gives the same label at D = 16, 32 and 64 with the frozen E0. No optional-stopping effect.
- (iv) P0-3: exact-direction, hit-point evaluation removes the *bias* (pooled error 0.11, i.e. 0.05 above reference noise, vs 0.55
  for A4). At the deployed single-draw budget it does not reduce the realised error: X1 is better than A4 in 44 of 64 draws but
  the paired mean difference is 0.024 +- 0.055. The gain needs roughly 16 - 64x the relation budget (or an equivalent variance
  reduction) to materialise.
- Verdicts: P0-2 core statement VERIFIED on CK2; the exclusive label "angular, not sampling" CONDITIONALLY_VALID (deployed-draw
  dependent). P0-3 CONDITIONALLY_VALID (expected-value statement; not a single-draw accuracy statement). Confidence high
  (recomputed from stored per-draw arrays; the per-draw arrays themselves were not regenerated).

### B-08 (confirmation) - WL96A Stage 1 decision and threshold provenance
- [REPLAY] `results/ecro_util96a/s1.json`: dI(Z) = 0.178 (dropping the whole >= 3-event tail changes the ROI by 17.8 %, so the tail
  is not negligible in the ROI); dI(XOD) = 0.00065; reference-tail noise n_img = 0.00146; H = -0.0008; U = -0.0008 < M_img
  0.005 -> S1_NO_HEADROOM. Fully explicit tail XD: dI 0.0029. Learned global terms G8601 / G8602: dI 0.0255 / 0.0126.
- [REPLAY] independence control `tails.json /field/XD_cross_*`: explicit tail built from e2 half a vs reference tail half b: table
  rel L1 0.044 (box A, e3_o) against half-vs-half 0.063. The explicit tail is not flattered by noise shared with the reference.
- [CODE] M_img = 0.005 is the tolerance floor already written in `ecro_local93d.json:33,36`, `ecro_mat94.json:119,150,151`,
  `ecro_q94c.json:43`, `ecro_s94d.json:39` and `ecro_attr95a.json /decisions/noise`. It was not introduced for WL96A.
- Disclosed exposure [CODE `ecro_util96a.json /pre_freeze_exposure`]: dev figure figW1 was opened before the freeze and showed
  qualitatively that XOD / XOP are closest to F. Because the decision margin is about 7x (U would have to exceed 0.005; the
  measured magnitude is 0.0007, below its own noise) and the threshold is inherited, the exposure cannot have changed the
  outcome. Any M_img >= 0.001 gives the same label.
- Scope restrictions that the label carries [CODE]: one geometry, one light, one ROI, N = 1,270 patches; the prefix e2 and the
  sharp-last part e3_p are the path-traced reference (oracle); Stage 2 (fitting a learned-form resolvent) was not triggered, so
  no learned resolvent was ever fitted - the result is a ceiling argument; Stage 3 cost arms use random factors and a
  timing-only edit; U(ZERO) = 1.038 / 1.043 [REPLAY `final.json`] is an Amdahl statement about this lifecycle composition
  (64-spp 512 x 512 frame, q table, state update).
- Verdict: decision VERIFIED as executed and unbiased by the exposure; claim CONDITIONALLY_VALID (restrictions above). Confidence
  high for protocol integrity; the numbers were replayed, not regenerated.

### B-09 (invalid_inference risk, MATERIAL for P0-1 / P0-5) - the exposure / learnability hypothesis is untested
Chain of the retraining evidence [CODE protocols + REPLAY finals]:

| Batch | What was trained | Scale | Outcome | Why it does not test learnability of the sharp sender |
|---|---|---|---|---|
| WL86 / WL92A | full contract | 24,000 steps, 1,024 / 928 geometries, 2 seeds | F2 fail on E2 C / D_STOPPED | no sender supervision; CK2-like transport ~10^-3 of lightings |
| WL93A-C | T0 / TQ / U0 / UQ / C0 / CQ / TR / TQR from same-seed OWN | 2,000 steps, 128 geometries | NO_SIGNAL / SHARING_SIGNAL / NO_SIGNAL | smoke; pooled table metric insensitive to sharp readers (stated in WL93C) |
| WL94 | A / B from OWN, seed 8601 only | 2,000 steps, 96 geometries | INCONCLUSIVE_SMOKE_FIT | smoke fit failed; seed 8602 not run by rule |
| WL94A | C (50 / 50 mix) | 2,000 steps, 96 geometries | INCONCLUSIVE (seed disagreement) | - |
| **WL94B** | **D (exposure) - NOT TRAINED** | - | STOP_INVALID_INTERVENTION (1.232 / 1.236 vs 1.25) | the only prospective exposure test never ran |
| WL94C | D / E query weighting x8 | 2,000 steps, 96 geometries | INCONCLUSIVE (s8601 back to own level, s8602 worse) | - |
| WL94D | T / R / P sender teacher, init from WL94C D | 1,000 steps | NO_RELIABLE_IMPROVEMENT | teacher = M64 bin-level sender, known before the freeze to mis-deliver on CK2 (ratio 1.38, corr 0.71) |

- [CODE `ecro_exp94b.json /pre_freeze_exposure`]: "with sigma = 0.5 and a top quartile, G_EXPOSURE >= 1.25 requires mean PM(H) /
  mean PM(generic) >= 2.33, so it binds before G_CONTRAST (2.0); both are kept as written", and the 8-item dev subset had already
  failed both gates (1.83 / 1.17). The evidence run measured contrast 2.20 (passes 2.0) and exposure 1.232 / 1.236 (fails 1.25)
  [REPLAY `audit.json /gates`]. The stop was therefore foreseeable at the freeze. This is disclosed and nothing was changed
  afterwards, so it is not a post hoc manipulation; its effect is that hypothesis H-P remained untested while the lineage moved
  on to representation-level conclusions.
- [CODE `ecro_s94d.json /stage_A/G_PHYS`, `/feasibility_result/CK2_flight_floor_diagnostic`]: the first G_PHYS gate (corr >= 0.9)
  failed at 0.641 and was replaced before the freeze by "corr >= the label's own reliability (0.525)". The revision is honest
  about label noise, but it lets the batch proceed with a teacher whose attainable target is, in the protocol's own words, "the
  C2_srcS1 control, not the physical e2_p". A null result of WL94D therefore says little about whether direct, physically
  correct sender supervision would work.
- Two-seed structure: seeds 8601 / 8602 of every fine-tune inherit the same-seed OWN parent, and the two parents differ
  strongly at the S1 sender (B-05). "Both seeds" is n = 2 lineages, not two independent replications of a batch.
- Decision-level impact: "the learned S1 sender has large error" is measured; "avoidable" is supported at operator level only
  on training-distribution metrics (a non-negative rank-16 operator reaches training-weighted row error 0.42 where the learned
  factors sit near the zero-operator level, [REPLAY] `main.json` op_flux_l1_mean 1.02 / 0.99) and *not* on CK2, where the
  in-family bound itself loses the lobe (B-06); "cannot be learned" is an UNTESTED_INFERENCE.

### B-10 (dataset_limitation / post hoc selection, MATERIAL for P0-5) - CK2 provenance
- [CODE / worklog 93D addendum, lines 12-50]: CK2 replaced a withdrawn first scene; the owner instructed to "adjust object placement
  until" the checker is visibly reflected; five render probes (p01-p05). Final placement: box-A corner x = -1.345 against the
  rule's -1.35 (face about 0.22 m from the panel, "about the minimum distance the training box rule allows"), one light at
  (-1.45, -0.6, 1.7), 5 cm in front of the panel plane. The addendum protocol carries "labels: none. No verdict, gate or label".
- The same scene then became the only quantitative workload of WL94, WL94A-D, WL95A and WL96A. It has one light, one geometry,
  no approved edit (WL96A uses a timing-only edit) and no second view.
- After WL92B there is no never-inspected sharp held-out condition: E4 was not admitted (Q4 SNR 3.9 vs 4.0 [REPLAY
  `e4_admission.json`: 0.0047 vs 4 x 0.0012]), E5k995 was admitted and then inspected, three E6 candidates failed the
  sharp-family clause (WL93D), CK2 is hand-designed.
- Impact: all four technical P0 claims are single-workload statements at the edge of the training support; this supports the
  second half of P0-5 (no general impossibility) more strongly than the worklog summaries convey.

---

## 4. Reference and oracle independence (question 5)

| Reference / oracle | Construction | Independent of the judged arms? |
|---|---|---|
| WL86 field targets (`gen86`) | path-traced per geometry, per-file seeds | yes (training targets only) |
| Image references REF-ER / REF-P | 4 x 4,096 spp, seeds 86600 + r | arms rendered with the same camera random numbers (paired by design); absolute E_H noise bound 0.008 in the CK2 ROI, larger than most arm differences - decisions correctly use paired dI |
| CKF_none event-split field (WL94) | 2 x 4,096-spp halves; sha256 recorded in WL94 `pre.json`; reused by WL94B-D, WL95A (`ref_a / ref_b`), WL96A through symlinks `fields -> ~/rest_x94/results/fields` | arms use different random streams (state rays 86001, quad 87001, draws 950000 + d); WL96A adds a cross-half control (B-08) |
| "Oracle" tables (ORA2, exact-q, C2_src / C2_rcv) | path-traced fields or exact ER evaluation inserted into the ECRO pipeline | they share the scene model with the reference by construction (that is what makes them oracles); noise independence holds via separate seeds |
| X1 / X1c (WL95A) | analytic ER evaluation at exact directions | same BSDF as the reference (intended); not an independent implementation of the physics |
| q tables (Mq 1,024) | once-scattered targets; WL93A showed a heavy-tailed geometry dominates pooled L2 | noise-limited on near-panel lights (B-04 iii) |
| WL94B e2_p label at 256 spp | reliability corr 0.525 between realisations [CODE s94d] | too noisy to be a sender target; correctly not used as one |

Not verified here: the event-split tag semantics (`pt86a.field_src(split=True)` was added after the WL86A freeze under amendments
1-2; WL96A patches the tag and checks `T_SPLIT_TAG`), and the radiometric conventions of `an94.W / an94.eh`. These are
implementation questions outside this track's lens.

---

## 5. Protocol integrity

### 5.1 Per-batch table

"Code diff" = `git diff --stat <freeze> <completion> -- experiments/<dir>` [MEAS]. Bias risk is this auditor's classification of
whether the item could have moved the *recorded outcome*; "toward stop" means it prevented a test rather than produced a
negative measurement.

| Batch | Freeze (completion) | Code diff after freeze | Amendments / deviations / exposure | Seeds | Outcome | Bias risk |
|---|---|---|---|---|---|---|
| WL86 | 7786d09 (b97027d) | tests86 +63, analyze86 +3, run86, diag_v3, amendment1 | Amendment 1 (63972c8): V3 null-edit test clause re-defined after run 1; chain resumed | 2 | F2 fail 0.063 / 0.072 vs 0.06 on E2 C | low (test clause, not a fidelity gate); F2 miss is 5 - 20 % of the threshold with 2 seeds - a narrow fail |
| WL86A | 46593f0 (bd81442) | pt86a +19 (split option), d2 / d3 scripts, amendments 1-2 | two amendments add measurements after the first results | 2 | attribution | medium (post-freeze additions to the reference tracer; additive, default path unchanged) |
| WL92 | 5e672ff (9212a55) | none | dev local-operator stage seen (deterministic); **gate wording**: G_rep fails because the OWN-k16 fit is "not converged" although it reached 0.036 / 0.017 (< 0.06 level) [REPLAY `gate.json`]; completion code 1 text says "representation insufficient" - disclosed as "not established" | 2 | STOP_INTERVENTION_G_rep | toward stop: the unconverged in-family fit was below the level; the stop is not evidence of a representation limit |
| WL92A | 7247d30 + c40226d (ce87188) | run92a.sh +3 / -1 | run 1 aborted (OOM from concurrency), data deleted, run 2 with execution fix; no protocol amendment; E4 fails Q4 by 0.0047 vs 0.0048 | 2 | D_STOPPED; NO_REGRESSION fail on one source config, one seed | low for measurements; the OWN checkpoints of a STOPPED batch became the base of everything later |
| WL92B | b29d8ff (95ee276) | none | dev chain rendered real E1 / E2 / E4 with real models at 64 spp (disclosed); **label mapping**: code gives outcome C, the written rule gives D (reconciled in WL93 `recon92b.json`) | 2 | outcome C (official) | medium for the label, none for measurements |
| WL93 | c30c797 (56be753) | analyze93 +2 (Amendment 1, owner approval still pending), post hoc scripts | preflight STOP (val 0.508 vs bound 0.485) before any training; pre-freeze correction of the degenerate tiedcm control | - | STOPPED-INCONCLUSIVE | toward stop; cause later shown (WL93A) to be one heavy-tailed validation geometry |
| WL93A | 888ceb6 (ab68553) | post hoc scripts only | post hoc subgroup analysis labelled; corrects WL93's post hoc reading | 2 | NO_SIGNAL_AT_SMOKE_SCALE | low; pooled L2 metric dominated by g00010 (disclosed design limit) |
| WL93B | 03fbc84 (0da2d11) | none | T0 / TQ already inspected (disclosed); pre-freeze pattern-layout correction | 2 | SHARING_SIGNAL | low |
| WL93C | cab6e18 (943decc) | none | WL93A / B results incl. the exact-local floor inspected before the protocol | 2 | NO_SIGNAL_AT_SMOKE_SCALE | low-medium (design informed by the same validation population it is judged on) |
| WL93D | 4f3fc56 (94e13a0) | none | **disclosed exposure**: dev chain printed Task A functional values on the frozen checkpoints for a dev subset + sharp pairs and a dev label; protocol file predates the dev run; E6: 0 of 3 candidates admitted | 2 | REPRESENTATION_OR_OPTIMIZATION_DOMINANT + COADAPTATION_LIMITED | medium (exposure to values of the quantities under test; thresholds unchanged) |
| WL93D add. | d8d092f (16a8aa6) | - | revision 1 withdrawn; scene iterated on renders; no labels by protocol | 2 | qualitative | high for *downstream use*: CK2 becomes the quantitative workload (B-10) |
| WL94 | e347904 (f3b431f) | posthoc94 +36 | dev figures opened (layout); gate RUN_C anticipated (disclosed); seed 8602 not run by rule | 1 | INCONCLUSIVE_SMOKE_FIT | low for outcome; single seed |
| WL94A | 5b7c0c8 (b34bdf9) | none | forgetting rule did not name a seed; implementation reads s8601 (CONTRADICTED) while s8602 gives SUPPORTED | 2 | INCONCLUSIVE | medium for the sub-label, none for the outcome |
| WL94B | 5c0a6cd (2990a2d) | post hoc scripts only | **gate inconsistency known at freeze** (B-09); dev subset had failed both gates | 2 (audit only) | STOP_INVALID_INTERVENTION, D not trained | toward stop (hypothesis H-P untested) |
| WL94C | 6d8b254 (487f83e) | none | **three pre-freeze feasibility runs**; matching changed (greedy -> Hungarian -> same-half pool) after balance values failed G_MATCH; no candidate output existed; dev label printed, not used | 2 | INCONCLUSIVE (seed disagreement) | low-medium (control construction tuned to pass its own gate; thresholds unchanged) |
| WL94D | a08c9bf (6c721fc) | none | **G_PHYS rule revised before freeze** after failing (0.9 -> label reliability 0.525); CK2 flight-floor diagnostic known before freeze | 2 | NO_RELIABLE_IMPROVEMENT | medium: batch proceeded with a teacher known to be delivery-biased (B-09) |
| WL95A | c152120 (30e9c37) | none | 16 -> 64 draw extension by the predeclared 2-SE rule (executed; label unchanged at 16 / 32 / 64); E0 defined on the deployed draw | 2 (learned arms) / 64 draws | RANK16 INADEQUATE, ANGULAR_PROJECTION_EVIDENCED, NEXT_TARGET UNDETERMINED | low for extension; medium for the E0 definition (B-07) |
| WL96A | a1718f7 (efcf86a) | figw3_posthoc +29 | dev figure figW1 opened (qualitative ordering of arms seen); M_img inherited | - | NO_MEANINGFUL_HEADROOM (CK2) | low (margin 7x, B-08) |

General observations [MEAS / CODE]:
- No batch from WL92 to WL96A changed a frozen evaluation script between freeze and completion except WL93 (2 lines, crash fix on
  the stop path). Post hoc scripts are separate files committed with the results.
- Deployed server code equals the frozen blobs where checked (CRLF-normalised sha256, server copy vs `git show <freeze>:<file>`):
  WL86 `scene86 b75a2d9c` / `train86 340180b3` / `model86 72a728b8`; WL92A `own92a.py 6c372e03` (@c40226d); WL94 `mat94.py
  15268a6a` (@e347904); WL95A `a95.py df3a14c2` (@c152120); WL96A `w96a.py afb80dd3` (@a1718f7). Chain status lines record
  the same commits and protocol hashes (WL94 `99e199f3`, WL95A `fb136ad2`).
- No threshold was found to be changed after an evidence run. Changes cluster *before* freezes, after feasibility or dev runs
  that exposed gate values (WL94C, WL94D), plus label-mapping gaps resolved by disclosure rather than amendment (WL92, WL92B,
  WL94A).
- Four of the batches stop before the informative test (WL92, WL92A's E4, WL93, WL94B). Each stop is individually defensible;
  collectively they mean that the positive branch of the lineage (retraining under a corrected contract at full scale, on an
  admitted untouched sharp condition) was never executed.

### 5.2 Checkpoint identity [MEAS, server sha256 prefixes]

| Checkpoint | sha256 | Produced | Consumers (how) |
|---|---|---|---|
| ecro_s8601 / s8602 | 9d332cd61a71 / 4d993121e41d | WL86 | WL86A-WL92 (path) |
| own_s8601 / s8602 | 739e7eb50b25 / cee9a09a289f | WL92A | WL92B, WL93A-D init / eval, WL94* init and baseline, WL95A A0 / G, WL96A G - always read from `~/rest_x92a`; hashes recorded in WL92B `repro.json`, WL93D `ckpt93d.json`, WL94 `pre.json`, WL94B / C / D `repro.json`; **not recorded** in WL95A / WL96A outputs (identity shown indirectly by B-05) |
| sum_s8601 / s8602 | 3ec1d5910a3e / d911c6b22cfd | WL92A | WL92A / B only |
| T0 / TQ | 0d909222b959, 357e36878985 / 87841401d16b, c108b515034f | WL93A | byte-identical copies in WL93B and WL93C |
| A / B s8601 | f80fd6cd441a / b443682c65a7 | WL94 | WL94A-D by symlink chain 94 -> 94r -> 94b -> 94c -> 94d |
| C, D, E, T, R, P | distinct per batch | WL94A / C / D | later batches by symlink |

No checkpoint substitution or silent retraining was found. "Frozen own" is the same pair of files in every batch.

---

## 6. P0 assessment within this track's lens

| Claim | Proposed verdict | Basis | Restrictions |
|---|---|---|---|
| P0-1 learned S1 sender has large avoidable error; deployed operator may violate passivity | **CONDITIONALLY_VALID** (passivity half VERIFIED) | B-05: albedo 2.26 / 3.08 reproduced to all digits, 48 / 48 training states non-passive, not confined to rarely lit bins. B-04 / B-09: sender never supervised directly, CK2-like transport ~10^-3 of lightings, exposure test never trained | "avoidable" holds against the in-family bound on training-distribution metrics, not on CK2 (B-06); "cannot be learned" is untested; two lineage seeds that err in opposite directions |
| P0-2 M64 sender bias not eliminated by relation sampling | **VERIFIED** on CK2 for the bias; label exclusivity CONDITIONALLY_VALID | B-07: pooled ratio 1.49 - 1.54 +- 0.035 vs 1.02 for exact arms | one workload, box A, one light; with a typical draw as E0 the frozen rule gives MIXED |
| P0-3 exact-direction / hit-point ER evaluation substantially closes the CK2 delivery error | **CONDITIONALLY_VALID** | B-07: pooled error 0.55 -> 0.11 (noise 0.06) | true for the expected value over >= 16 - 64 relation draws; no single-draw gain (0.024 +- 0.055) |
| P0-4 learned broad higher-order resolvent has no meaningful accuracy or lifecycle-cost headroom on CK2 | **CONDITIONALLY_VALID** (decision integrity VERIFIED) | B-08: U = -0.0008 vs 0.005, cross-half control, inherited threshold, exposure immaterial | oracle prefix, one light / geometry / ROI, N = 1,270, no learned resolvent actually fitted, timing-only edit |
| P0-5 negatives justify rejecting the tested ECRO mechanisms but not general impossibility | **CONDITIONALLY_VALID** | second half strongly supported (B-04, B-09, B-10); first half supported for (a) the light-independent rank-16 bilinear S1 operator in the 64-bin basis on sharp grazing transport (B-06) and (b) the 64-bin binned flight (B-07) | "tested mechanisms" must be read narrowly: the WL86 / WL92A training contract at one data scale, smoke fine-tunes, one edge-of-support workload; "learned sender fails" as a statement about learning is UNTESTED_INFERENCE |

---

## 7. Special checks requested in the brief

| Check | Result |
|---|---|
| Was CK2 excluded from the OWN checkpoints' training population? | **No.** Only from the WL94 candidates. 17 / 928 files violate the single-condition rule; no jointly close geometry (B-02) |
| WL94B "4 / 96 items" and "ring-0 4.3 %" | Confirmed (4 / 96; 4.32 % on P94, 4.68 % on P92A). Grazing-specific coverage is lower still: 4 / 3,712 lightings at CK2 level (B-04) |
| "rank-16 cannot represent" confounded by coverage? | Not for the light-independent operator: allocation sweep shows a structural limit (B-06). Confounded only if read as a statement about learnability |
| "learned sender fails" confounded by coverage? | Yes, as a learning statement (B-04, B-09); the passivity violation itself is not a coverage artefact (B-05) |
| Were "frozen" operators identical across batches? | Yes (5.2) |
| WL92 gate wording | G_rep "fail" = designated fit unconverged while already below the level (0.036 / 0.017 vs 0.06); stop is "not established" (5.1) |
| WL93 preflight stop | Executed as frozen; cause = heavy-tailed near-panel validation light (sampler has no panel clearance); Amendment 1 still pending approval |
| WL94B gate inconsistency | Confirmed; known before the freeze; dev subset had already failed (B-09) |
| WL94C matching revisions | Three pre-freeze feasibility runs; matching algorithm changed to pass G_MATCH; thresholds unchanged; no candidate output existed |
| WL94D G_PHYS revision | Pre-freeze, after a failed first draft; revised gate is weak and the teacher was known to be delivery-biased (B-09) |
| WL95A extension to 64 draws | Predeclared, executed, label invariant to it (B-07) |
| WL96A dev-figure exposure, M_img provenance | Exposure real but immaterial; M_img inherited from four earlier protocols (B-08) |
| Validation reuse | The 55 (WL92A) / 24 (WL94) validation geometries are the evidence population of every batch WL93A-WL94D and informed successive designs; not used for model selection inside any training run |
| Single / two-seed conclusions | WL94 (1 seed); WL94A sub-label seed-dependent; WL94C seed disagreement; all fine-tunes share two parent lineages |

---

## 8. What could not be verified, and resumption notes

- Not regenerated: any path-traced field, render, per-draw delivery array or timing. B-07 and B-08 are recomputations / replays of
  stored arrays; B-04's WL94B rows are replays (my geometric proxy agrees in rank, Spearman 0.87).
- Not verified: the radiometric correctness of the albedo and W_T conventions (only the M64 consistency check 0.825), the
  event-split tag semantics, `an94` metric weights, the WL86A amendments' content, E6 candidate disjointness, the WL93A-C
  128-geometry subset against CK2 (those arms were not evaluated on CK2), and the contents of the dev / feasibility / probe
  directories (claims that outputs "were not opened" are not checkable from artifacts).
- The mirror proxy ignores lobe width and Fresnel; the rank-16 sweep is a Frobenius multiplicative-update fit with 3 restarts.
- Resumption: the first attempt's scripts `regen_population.py`, `heldout_disjointness.py` were re-run in this run and reproduced
  their outputs exactly; `learned_s1_albedo.json` was validated through the digit-exact match with WL95A's stored albedo maxima
  and the 4.68 % ring-0 agreement between two independent computations; `wl95a_flight_sensitivity.json` was cross-checked by a
  separate inline recomputation (section 3, B-07 table). No file of the first attempt was found to be wrong. `rank16_allocation`
  is new in this run.

---

## 9. Reproducibility appendix

Working directory `c:/Projects/ReST-GI`, HEAD efcf86a, local python 3.11 + numpy. Server LabServer63, `. ~/rest_x95a/env.sh`.

```
PYTHONDONTWRITEBYTECODE=1 python experiments/audit97/B/regen_population.py results/audit97/B/population_regen.json
python experiments/audit97/B/heldout_disjointness.py results/audit97/B/population_regen.json results/audit97/B/heldout_disjointness.json
python experiments/audit97/B/panel_coverage.py results/audit97/B/population_regen.json results/audit97/B/panel_coverage.json
python experiments/audit97/B/wl95a_flight_sensitivity.py results/ecro_attr95a/deliv.npz results/audit97/B/wl95a_flight_sensitivity.json
python experiments/audit97/B/rank16_allocation.py results/ecro_attr95a/ops95a.npz results/audit97/B/learned_s1_albedo.json results/audit97/B/rank16_allocation.json
# server (one short GPU job, < 1 GB, about 1 min; read-only inputs, writes only under ~/rest_x97/B):
$PY server_digest_population.py ~/rest_ecro86 population_server.json
$PY learned_s1_albedo.py ~/rest_x96a/code/experiments ~/rest_x97/B/own_R ~/rest_x92a ~/rest_ecro86 learned_s1_albedo.json
# checkpoint hashes: sha256sum ~/rest_*/results/models/*.pt (section 5.2)
# freeze integrity: git diff --stat <freeze> <completion> -- experiments/<dir> (commit pairs in section 5.1)
```

Seeds: rank16_allocation rng 97000 + 10 c + restart; no other random numbers are drawn by this audit (the population regeneration
uses the frozen WL86 seeds; panel_coverage uses the deployed table seed 86001).

sha256 prefixes (scripts): heldout_disjointness.py f19496efd89aba17; learned_s1_albedo.py 423417a642852a9e; panel_coverage.py
f84b9f3f2d46b4bd; rank16_allocation.py 6e5de7600b55de74; regen_population.py 4612d046b45f45b8; server_digest_population.py
c66e9ea671fa2645; wl95a_flight_sensitivity.py 676a196231d0536a.

sha256 prefixes (outputs): heldout_disjointness.json 599a8fad143f480a; learned_s1_albedo.json 5a09bf43b5889dae;
panel_coverage.json 69edf0b2b09c7e06; population_regen.json f4fffad8d1dc16b7; rank16_allocation.json 3383b81f0b36a77a;
wl95a_flight_sensitivity.json 796bc54ab0f142d9.

sha256 prefixes (inputs): results/ecro_attr95a/deliv.npz f2324a32effc80ca; ops95a.npz d4d02e5a2f423ce9;
results/ecro_exp94b/audit.json bb718930bf5fea29; results/ecro_own92a/results/exclusion.json 147edfa371031f3f;
results/ecro_mat94/select94.json 63b0c9a05420a7ee; results/ecro_util96a/s1.json 92f5e763126d83b6;
experiments/ecro86/scene86.py b75a2d9cbae1f738; experiments/ecro_local93d/addendum_checker/scene_ck2.json 977ba122ab3232e7;
checkpoints own_s8601 739e7eb50b25, own_s8602 cee9a09a289f, ecro_s8601 9d332cd61a71, ecro_s8602 4d993121e41d.

The file `results/audit97/B/_interrupted_run_log.md` is the coordinator's extract of the first attempt's commands (no conclusions).
