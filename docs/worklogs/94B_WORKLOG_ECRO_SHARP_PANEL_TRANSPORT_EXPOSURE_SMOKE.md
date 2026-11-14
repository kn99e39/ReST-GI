# Worklog — WL94B: ECRO Sharp-Panel Transport Exposure Smoke (2026-11-14)

## Origin and scope

- **Brief.** "WL94B — Sharp-Panel Transport Preservation Smoke" (user, 2026-11-14).
- **Batch class.** This is a suffix of WL94. It is a dependent evidence-gap follow-up of WL94 / WL94A under the preserved ECRO contract, and only the training-example sampling exposure changes. The brief names it WL94B and the owner decides the name. No historical number changes.
- **Hypothesis H-P.** Sharp-panel-mediated transport is insufficiently constrained relative to its importance in the current ECRO training distribution. A controlled increase of its training exposure would preserve its fidelity without changing the architecture.
- **Preserved, unchanged:**
  - the ECRO architecture (735,664 parameters) and the material buffer;
  - the OWN objective, loss weights, optimizer, schedule and 2,000 steps;
  - the frozen own_s8601 / own_s8602;
  - the WL94 / WL94A checkpoints, tables, renders and pop evaluations (read-only links);
  - the WL94A 50/50 item list;
  - the CK2 scene, references and masks;
  - the WL94 / WL94A code (`experiments/ecro_mat94`, `experiments/ecro_mat94r`, imported unchanged) and their labels.
- **Code.** `b94.py` and `run94b.sh` in [experiments/ecro_exp94b/](../../experiments/ecro_exp94b/). Post hoc: `posthoc94b.py` and `figb4_posthoc.py`.
- **Protocol.** [protocol/ecro_exp94b.json](../../experiments/ecro_exp94b/protocol/ecro_exp94b.json). Freeze commit `5c0a6cd`; sha256 prefix 6b5b10c78d239337.
- **Where.**
  - Server: LabServer63 `~/rest_x94b`, one detached chain, 14:15–14:18 UTC (2.7 min).
  - Local results: `results/ecro_exp94b/`.
  - Figures: `results/evaluation/94B/`.

## Result in one line

The audit stopped the batch before training with **STOP_INVALID_INTERVENTION**. The predeclared stratified allocation raises panel-mediated training exposure by only 1.23× / 1.24× against the required ≥ 1.25×. Candidate D was **not trained**. The batch outcome is therefore INCONCLUSIVE (stop), and H-P's training-exposure question is **untested**. The coverage audit itself completed: by the predeclared rule, sharp-panel-mediated transport is **UNDERREPRESENTED** in the training data relative to the CK2 evaluation contract.

## Implementation Fact

**Stage A — what the current pipeline does.**
- Sampling and supervision in `c94.train`:
  - It draws the 96 items uniformly, as rng(seed + 94) permutations, about 20.8 visits per item.
  - Every step supervises all 4 lightings × all 1,270 patches × 64 bins × 3 channels, which is 975,360 queries per step for every item.
  - The loss is the per-(lighting, patch, channel) normalised relative L2 (den = bin mean + 0.05 × item mean) on three terms: the sum vs f, S vs e2 = e2_p + e2_o, and G vs e3 = e3_p + e3_o.
  - There is no path-, receiver- or bin-specific weight.
- Path labels:
  - `pt86a.field_src(split=True)` with the WL94 per-patch material routing partitions each target into e2_p / e2_o / e3_p / e3_o.
  - The `_p` suffix means the vertex next to the receiver lies on the S1 panel (material id 1).
  - "Panel-mediated" means e2_p + e3_p.

**Audit render.**
- Every training item (in its WL94A C assignment) was re-rendered with `split=True` using exactly the seeds, spp and batch of its cached WL94 target (940000000 + 16·idx + light, 256 spp).
- An independent realization (943000000 + …, 64 spp) was rendered to measure the reliability of the stratum.
- Per item, lighting, patch and part, the stored aggregates are the loss-normalised mass Σ X / den (den taken from the stored training target, exactly as in `c94.objective`) and the exitance.

**Candidate D (predeclared, not run).**
- The WL94A C item list is unchanged.
- Within each material half, the 12 items with the largest PM (the stratum score below) form stratum H.
- H receives 500 of the half's 1,000 steps: 41–42 visits per H item (2.0× C) and 13–14 per generic item (0.67× C).
- Original / diversified steps would be exactly 1,000 / 1,000.
- The per-step copies would be passed to the unchanged `c94.train`. Its permutation makes one exact pass; the sequence is replayed and stored as the sampling manifest (`schedule_s860x.json`).

**Tests (all PASS).**

| test | result |
|---|---|
| T_REPRO | own / C CK2 tables recomputed = stored (rel ≤ 2.7e-7, 102 keys each); own file sha256 `739e7eb5…` / `cee9a09a…`; C state sha256 = WL94A; own E_H and box-A amplitudes = WL94 stageB, C E_H and panel-face G = WL94A (abs diff 0.0) |
| T_PATH_ID | the four parts sum to the stored training target f (max rel L1 4.7e-8) and e2_p + e2_o to f2 (5.8e-8); WL94 float-atomic floor 3.7e-8 |
| T_PANEL_ID | panel patches are id 1 in all 96 maps; no other patch is id 1 |
| T_QUERY | 4 lightings, N = 1,270 for every item → 975,360 queries per step for every item, so D's total query count equals C's |
| T_MIX | 48 / 48 items, digests = select94 |
| T_ALLOC | 2,000 steps, 1,000 / 1,000, H share 0.5 per half, min 13 visits, replayed counts = counts |

**Physical correctness of the labels.** The path labels exactly partition the targets the model is trained on: the semantics are consistent and no proxy is involved. The objective uses the labels only in sums (e2, e3), so the panel-sender signal is supervised only jointly with e2_o. The panel's learned once-scattered response is supervised only through receivers. Sampling cannot change this; it is a supervision-semantics limit, separate from the sampling deficiency.

## Measurement (Stage A audit)

**Stratum score.**
- PM is the loss-normalised panel-mediated mass on box receivers (box A + box B) divided by the item's total loss-normalised target mass.
- PMb_A is the panel-mediated share of the box-A target.

| quantity | training items (96) | CK2 (untextured reference, box A) |
|---|---|---|
| panel-mediated share of the whole item target | e2_p 2.1 %, e3_p 1.3 % (e2_o 51.1 %, e3_o 45.4 %) | — |
| PM | mean 0.0048, median 0.0043, p10–p90 0.0022–0.0073, max 0.0137 | PM_A 0.0059 |
| PMb_A | mean 0.036, median 0.023, p90 0.075, max 0.25 | **0.138** |
| items reaching CK2 PMb_A | 4 of 96 (4.2 %) | — |
| e2_p share of box e2 | mean 0.068 | box A 0.106; panel face **0.58** |
| e3_p share of box e3 | mean 0.057 | box A 0.166; panel face **0.76** |
| box share of item target | 0.077 (p10–p90 0.074–0.079) | — |

The CK2 panel-facing face carries 83 % of box A's panel-mediated mass.

**Gates.**

| gate | value | rule | result |
|---|---|---|---|
| G_REL | Spearman 0.9989; top-quartile membership agreement 1.0 / 1.0 | ≥ 0.90, ≥ 0.80 | PASS |
| G_CONTRAST | mean PM(H) / mean PM(generic) = 0.00813 / 0.00370 = 2.20 | ≥ 2 | PASS |
| G_EXPOSURE | D / C panel-mediated loss-mass exposure: s8601 1.232, s8602 1.236 (box-A PMb_A 0.047 vs 0.036) | ≥ 1.25 | **FAIL** |

**Underrepresentation (predeclared rule).** **UNDERREPRESENTED.** Only 4.2 % of items (< 10 %) reach the CK2 box-A panel-mediated share. The C stream mean PMb_A is 0.036, which is below 0.5 × 0.138. The rule uses only coverage, never errors.

**Post hoc (labelled; after the stop; CPU only; descriptive; no threshold, allocation or outcome changed).** Source: `posthoc94b.py`, results/ecro_exp94b/posthoc94b.json, figB4.
- On the CK2 panel-facing face, 69 % of the loss-normalised target is panel-mediated.
- In the training items, the box face with the largest panel-mediated share has a median of 33 % (p90 60 %, max 82 %). Only 4 items reach the CK2 level and 42 reach half of it.
- For box A, the best face has a median of 8 % and 2 items reach the CK2 level.
- Only 4.5 % of all box-face loss mass lies on faces with at least half of CK2's panel-mediated share.
- Arithmetic bounds of item-level reallocation, relative to C:

  | allocation | exposure |
  |---|---|
  | the protocol's D (predeclared) | 1.23× |
  | all 2,000 steps on the top quartile | 1.70× |
  | all steps on the single top item | 2.84× |

## Observation

- The relevant physical strata exist, are exactly identified by path labels, and are measured reliably (independent-seed rank correlation 0.999).
- They are a small and diffuse part of the training signal: about 3.4 % of the loss mass and about 0.5 % on box receivers. The item distribution is only moderately skewed, with a top-quartile contrast of 2.2.
- The CK2 condition that drives the sharp ROI is a box face whose target is about 70 % panel-mediated. That condition is almost absent from the 96 training items.
- Re-weighting whole items cannot concentrate panel-mediated supervision strongly: the predeclared allocation gives 1.23×, and even degenerate allocations stay below 3× while discarding most of the generic data.
- The audit's dev subset already showed the same gate pattern. This exposure is disclosed in the protocol, and nothing was changed after it.

## Interpretation (separating the levels)

- **Physical correctness.** The labels, targets and references are consistent: path identity holds to 5e-8, the panel identity is exact, and T_REPRO reproduces the frozen comparisons exactly. The stop is not caused by a label or reference defect.
- **Observed model behaviour.** No D behaviour was observed, because D was not trained. The WL94 / WL94A measurements (own / A / C) are unchanged.
- **Architecture inference.** None. The stop does **not** show that ECRO cannot represent sharp-panel transport, and it does not show that exposure is insufficient. The predeclared intervention was too weak by its own validity gate (1.23× vs 1.25×; the gate binds before G_CONTRAST for a 2.0–2.33 contrast, which was noted before the freeze and kept as written).
- **What the audit does establish.** Under the current item population and objective, training exposure to the CK2-type panel-dominated receiver condition is small. Item-level resampling of these 96 items has a bounded reach (≤ 2.8× even in the extreme).

## Unresolved Questions

- Whether a stronger exposure change would preserve sharp fidelity is untested. Any such change would require one of the following, none of which is authorised:
  - another allocation (not authorised after results);
  - another item population;
  - query-level selection inside items, which would change the per-step objective weighting.
- Whether the sender error (WL94A: 0.53–1.21 on the panel face, identical for A and C) is a coverage, optimisation or representation effect remains open.
- The joint e2 / e3 supervision masks the panel-sender signal: the panel share of box e2 is 6.8 % on average. Its role is open; this is a semantics question, not a sampling one.

## Architecture Verdict

**INCONCLUSIVE (STOP_INVALID_INTERVENTION at the Stage A gate; candidate D not trained).**
- None of the brief's outcome labels SHARP_FIDELITY_PRESERVED / SENDER_IMPROVED_GLOBAL_LIMITED / GLOBAL_IMPROVED_SENDER_LIMITED / MATERIAL_SHARP_TRADEOFF_REMAINS can be assigned.
- Nothing is adopted and the WL94 / WL94A labels are unchanged.
- No amendment, alternative allocation or successor is started; the decision returns to the owner.

**Answers to the brief's questions.**

| question | answer |
|---|---|
| Was sharp-panel transport underrepresented in training? | **Yes, by the predeclared coverage rule (UNDERREPRESENTED).** 4 / 96 items reach the CK2 box-A panel-mediated share; the uniform-stream mean is 0.036 vs 0.138; the CK2 panel-face condition (≈ 70 % panel-mediated) appears on a strong face in only 4 items. |
| Did targeted sampling improve panel-sender fidelity? | Not measured (D not trained). |
| Did it improve panel-mediated global transport? | Not measured. |
| Did it preserve or improve the original frozen sharp ROI? | Not measured. The anchors are unchanged: frozen own 0.047 / 0.042; C 0.084 / 0.067. |
| Were F2 and MR preserved? | Not measured for D. C passed both on both seeds (WL94A). |
| Were gains consistent across seeds? | Not applicable. The audit gate failed on both seeds (1.232 / 1.236). |
| Does the result support training exposure as a sufficient explanation, or does representation uncertainty remain? | **NOT_SHOWN.** Exposure sufficiency is untested, and representation uncertainty remains. The audit shows that item-level exposure of the relevant condition is small and that re-weighting these items has bounded reach. |

## Reproducibility

- **Commit and protocol.**
  - Freeze `5c0a6cd`; protocol sha256 prefix 6b5b10c78d239337.
  - The post hoc scripts are committed with the results.
- **Dev chain.**
  - Root `~/rest_x94bdev` (first 8 items, 60 steps; not evidence).
  - Its gate flags and audit.json were seen before the freeze, as disclosed in the protocol.
- **Input checkpoints.**
  - own_s8601 `739e7eb5…`, own_s8602 `cee9a09a…`.
  - C_s8601 / C_s8602 state sha256 equal the WL94A final hashes.
- **Targets.**
  - Cached WL94 `train_A` / `train_B` files, used read-only.
  - Audit seeds: 940000000 + 16·idx + light at 256 spp (identical to the targets), and 943000000 + 16·idx + light at 64 spp (independent).
- **Sampling manifests.**
  - `results/ecro_exp94b/alloc.json`: counts and stratum H.
  - `schedule_s8601.json` / `schedule_s8602.json`: the replayed 2,000-step sequence of D (never trained), plus C's realized counts.
- **Training.** None. No steps were run and no queries were spent on D.
- **Validation inputs.** Unchanged and unused.
- **Cost.** 2.7 min wall in total:
  - audit renders 8 jobs, ~2.2 min, pool 6;
  - T_REPRO tables 4 jobs;
  - GPU mean 50 % over 8 samples.
- **Outputs.**
  - `results/ecro_exp94b/`: audit.json, alloc.json, schedules, repro.json, final_b.json, posthoc94b.json, chain.status, gpu.txt.
  - `results/evaluation/94B/`: figB0_audit and figB4_posthoc_face_coverage (post hoc), with a sha256 manifest.
