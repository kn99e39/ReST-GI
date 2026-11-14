# Worklog — WL94D: ECRO Sharp Sender Oracle-Supervised Smoke (2026-11-15)

## Origin and scope

- **Brief.** "WL94D — Sharp Sender Oracle-Supervised Smoke" (user). It is a bounded feasibility-to-smoke continuation of WL94C. The batch is a suffix of WL94 under the preserved ECRO contract and adds one source-specific teacher term. The owner decides the name.
- **Hypothesis H-S.** The existing ECRO sender representation reproduces the sharp-panel response more faithfully when it is trained against a physically matched source-specific oracle.
- **Preserved:**
  - the architecture (735,664 parameters), the 64-bin angular discretisation and the material buffer;
  - the WL94C D objective, query weights and item stream (WL94A 50/50 materials);
  - the frozen own checkpoints and the WL94C D checkpoints;
  - all WL94–WL94C artifacts (read-only links);
  - the CK2 references and masks;
  - the earlier code, imported unchanged;
  - all historical verdicts.
- **Code.** `s94.py` and `run94d.sh` in [experiments/ecro_s94d/](../../experiments/ecro_s94d/).
- **Protocol.** [protocol/ecro_s94d.json](../../experiments/ecro_s94d/protocol/ecro_s94d.json). Freeze commit `a08c9bf`; the committed-file sha256 prefix is e0a389463e172bc7.
- **Where.**
  - Server: LabServer63 `~/rest_x94d`, one detached chain, 16:31–16:45 UTC (14 min).
  - Local results: `results/ecro_s94d/`.
  - Figures: `results/evaluation/94D/`.

## Result in one line

The outcome is **NO_RELIABLE_IMPROVEMENT** by the frozen mapping.
- The direct M64 teacher reduces the learned S1 sender error only modestly on the training fields: field rel L1 0.82 → 0.72 / 0.71 versus the plain continuation P at 0.82 / 0.80.
- On held-out validation geometries and lights the gain is about 3 %. On CK2 the sender error gets worse on s8601 (1.54 vs P 1.29), with an energy overshoot of 1.7×.
- The integrated sharp reflection does not improve: T is worse than P on s8601 (0.062 vs 0.044) and within tolerance on s8602.
- F2, MR and the panel-face global term are kept.
- The matched non-S1 oracle control R equals P.

## Source-contract correctness (Stage A; pre-freeze feasibility, re-checked in the chain)

**Contract.**
- The learned once-scattered sender of the single term is `Bq = a (cᵀ s)` per patch (`model86.apply`). On the S1 panel patches (1088–1129 = rect 5 = PSID 5 = material 1) this is the learned sharp-panel sender `Bl`.
- The exact counterpart in the same discretisation is `Bm = s · M64[S1]`: the bin-averaged exact local response from `quad87.enc_matrix`. It is the quantity that the established C2_srcS1 control substitutes.
- `Bl` and `Bm` share the input field, the outgoing bins, the patch frames and the units. The teacher reads the existing internal `Bq`; no head was added.

**Checks.**

| check | value |
|---|---|
| T_SEM | C2_srcS1 rebuilt from my `Bm` path = stored WL94C D tables (7.6e-8 / 6.0e-8); my `Bl` path through the flight = `model.apply` single term (0.0) |
| T_INDEX | pass |
| T_ENERGY | M64 S1 directional albedo max 0.825, mean 0.635 |
| T_REPRO | WL94B rule: own / C tables, checkpoints, metric replays — pass |
| G_PHYS (revised before the freeze) | training items, exact S1 sender + exact receiver single-flight exitance vs the WL94B e2_p path label: pooled ratio 0.953; corr 0.641 ≥ label reliability 0.525 |
| G_LAM | λ_T 0.0189 / 0.0127, λ_R 0.934 / 1.143 (equal initial gradient norm with the D objective) |

The label reliability in G_PHYS is the correlation between the 256-spp label and an independent 64-spp label.

**Rejected / revised checks (disclosed).**
- The first G_PHYS draft required corr ≥ 0.9 and failed (0.641). A probe without training showed that the sharp-reflector e2_p label at 256 spp is noise-limited: its two independent realizations correlate only 0.525. The exact flight agrees with the label better than the label agrees with itself.
- The rule was therefore replaced before the freeze by the noise-aware form above. The noisy e2_p label is not used as a sender target anywhere.

**CK2 flight-floor diagnostic** (precise 2 × 4,096-spp reference; not a gate).

| receivers | ratio | corr | rel L1 | reference half noise |
|---|---|---|---|---|
| all receivers | 1.38 | 0.71 | 0.94 | 0.13 |
| box A | 1.21 | 0.72 | 0.53 | 0.06 |
| panel face | 1.21 | **−0.33** | 0.60 | 0.07 |

- In the 64-bin flight, even the exact S1 sender misplaces the delivered energy beyond noise (the WL90 bin-projection floor).
- `Bm` itself is exact as the bin-level once-scattered response. The learned and the exact sender share this flight.
- So the attainable target of the learned sender in this representation is the C2_srcS1 control, not the physical e2_p.

**Direction coverage.**
- The training fields excite all 64 incident bins of the panel (validation fields 62, CK2 4; none outside the training set).
- There are therefore no bin-level held-out incident directions. Generalization is measured on held-out geometries and lights (the WL86 validation split, 24 items) and on CK2.

**Physical correctness verdict.** The teacher is the same quantity as the learned sender output, and the contract is available: SOURCE_CONTRACT_UNAVAILABLE does not apply. The physical fidelity of the whole path (sender + 64-bin flight) has the floor shown above.

## Training (candidate evidence)

- All candidates start from WL94C D_s<seed> (init sha256 `141ebff2…` / `e68df71b…`).
- Each runs 1,000 steps on the D objective and stream, order rng(seed + 9400), Adam 1e-4 with warm-up 100 and cosine decay to 1e-5.

| candidate | added term |
|---|---|
| T | + λ_T × teacher on the 42 S1 patches |
| R | + λ_R × teacher on all 1,228 non-S1 patches (matched oracle control) |
| P | plain continuation |

- The teacher is the per-entry normalised squared error of `Bl` vs `Bm` (OWN normalisation form).
- λ was fixed by a rule before the freeze: equal initial gradient norm with the D objective.
- There were no non-finite steps. Peak memory was 659–822 MiB and training took 53–105 s per run.

**T's S1 teacher loss** (training mean of the first and last 50 steps).

| seed | T | P |
|---|---|---|
| s8601 | 6.42 → 4.80 | 6.52 → 6.29 |
| s8602 | 6.29 → 5.57 | 6.45 → 6.75 |

## Standalone sender fit

Measurements are deterministic (no MC). Field values are the flux-weighted rel L1 of `Bl` vs `Bm` on the S1 panel. The energy ratio is Σ`Bl`·w / Σ`Bm`·w. The peak ratio is the learned value at the target's peak bin divided by the target's peak value.

| field rel L1 | own | D | **T** | R | P |
|---|---|---|---|---|---|
| s8601 train / val / CK2 | 0.80 / 0.88 / 1.08 | 0.82 / 0.89 / 1.21 | **0.73 / 0.86 / 1.54** | 0.82 / 0.89 / 1.34 | 0.82 / 0.89 / 1.29 |
| s8602 train / val / CK2 | 0.79 / 0.85 / 1.22 | 0.80 / 0.87 / 1.59 | **0.71 / 0.85 / 1.48** | 0.80 / 0.87 / 1.57 | 0.80 / 0.87 / 1.63 |

| energy ratio (val / CK2) and peak ratio (val / CK2) | D | **T** | P |
|---|---|---|---|
| s8601 | energy 0.87 / 0.87; peak 0.23 / 0.06 | energy 0.99 / **1.75**; peak 0.32 / 0.21 | energy 0.87 / 1.01; peak 0.24 / 0.07 |
| s8602 | energy 0.99 / 1.62; peak 0.27 / 0.15 | energy 1.09 / **1.71**; peak 0.33 / 0.26 | energy 0.99 / 1.67; peak 0.27 / 0.15 |

- **S1 local operator vs M64** over all 64 incident bins (flux-transfer weighted rel L1): 0.96–1.04 for every model (val geometries and CK2), including T. For reference, WL92's best rank-16 approximation of M64 S1 leaves an Eckart–Young residual of 0.46–0.52 in the flux-Frobenius norm and keeps 47–54 % of the mirror-bin flux; the learned operators keep 21–33 % of the peak on the validation fields.
- **CK2 delivery to box A.** This compares the learned and the exact S1 sender through the exact receiver.
  - s8601: T 1.05 vs P 0.62 (delivery ratio 1.28 vs 0.70).
  - s8602: T 1.24 vs P 1.23.

**Predeclared SENDER_IMP** (T ≤ 0.8 × min(R, P) on val AND CK2): **false on both seeds**. The val ratios are 0.97 / 0.98; the CK2 ratios are 1.19 (worse) / 0.94.

## Integrated transport

Untextured ROI_T E_H, arm C0. The anchor is frozen own; tol ≈ 0.017.

| | own | D | T | R | P | D with exact S1 sender |
|---|---|---|---|---|---|---|
| s8601 E_H (signed) | 0.047 (−0.040) | 0.044 (−0.030) | **0.062 (+0.040)** | 0.044 (+0.003) | 0.044 (−0.022) | 0.033 |
| s8602 E_H (signed) | 0.042 (+0.028) | 0.089 (+0.086) | **0.097 (+0.095)** | 0.101 (+0.095) | 0.089 (+0.087) | 0.019 |

- s8601: T is worse than P, R and D by 0.018 (> tol).
- s8602: all differences are within tol.
- With the exact S1 sender (C2_srcS1), T / R / P give 0.034 / 0.030 / 0.035 on s8601 and 0.018 / 0.022 / 0.017 on s8602.
- **Panel-face single flight** (learned S1 sender vs the noisy e2_p; half noise 0.32): T 1.04 (+0.77) / 1.49 (+1.28) vs P 0.52 (−0.19) / 1.29 (+1.02). T overshoots.
- **Panel-face global G** vs e3: T 0.21 (−0.17) / 0.060 (−0.002); P 0.23 / 0.060. The global term is preserved.
- **Textured ROI_T E_H.**
  - s8601: T 0.048–0.064 vs P 0.035–0.052; T is worse on all 5 variants.
  - s8602: T 0.065–0.117 vs P 0.056–0.114.
- **Image texture-response slope.** T 1.09–1.17 / 1.04–1.31; P 0.86–0.94 / 0.92–1.17. T is over-contrasted; contrast is not counted as better GI.

**Retention.**
- F2 holds for T, R and P on both seeds (T 0.0829 / 0.0848 vs limits 0.0856 / 0.0883).
- MR holds for all of them (box-A amplitude 0.95–1.04; V_ck population 1.04 / 1.08).
- The V_ck rel L1 of B is unchanged (0.107–0.119).
- Non-panel global on the other box-A faces: T 0.073 / 0.086 vs P 0.054 / 0.081.

**Predeclared RENDER_IMP** (T better than P and R beyond tol): **false on both seeds**. KEEP_T is true on both.

## Observation

- Direct source-specific supervision does move the learned S1 sender toward the teacher on the fields it is trained on: teacher loss −25 % / −11 %, field error −11 %, mirror-peak capture 0.23 → 0.32. The move is mostly an **energy increase** (validation energy ratio 0.87 → 0.99 / 0.99 → 1.09) rather than a lobe-shape correction (peak still about 0.3).
- On the CK2 incident configuration (4 incident bins at the panel, all covered in training) the same model **over-emits** (1.75 / 1.71). It delivers more panel light to box A than the exact sender (delivery ratio 1.28 / 1.19) and brightens the reflection. On s8601 this flips the sign of the sharp error and makes it worse.
- The full S1 operator stays about 1.0 away from M64 for every model. Its error is far above the rank-16 Eckart–Young floor (0.46–0.52, a different norm) and above WL92's per-geometry free-factor fits. The 1,000-step fit is therefore not near a representation limit; it is optimization- or generation-limited at this budget. The rank-16 bound itself is not tested here.
- R (the same oracle rule on non-S1 sources) behaves like P, so the T changes are specific to S1 supervision.
- The exact-sender control with these continuations stays at 0.017–0.035, close to the frozen-own exact-sender values (0.031 / 0.019). The gap of T to its own exact-sender value is 0.028 (s8601) and 0.079 (s8602).

## Interpretation (separating the levels)

- **Source-contract correctness.** The learned sender is directly supervisable against M64 within the existing representation: identical quantity (T_SEM ≤ 8e-8), no new head. The physical e2_p label is too noisy at training spp to validate per patch. The precise CK2 reference shows that the 64-bin flight of even the exact sender deviates from physics (panel-face corr −0.33), so M64 is the right teacher for the representation but not a guarantee of physical delivery.
- **Standalone sender fit.**
  - Partial on training fields, weak on held-out validation fields, and failing on CK2 (energy overshoot).
  - Representability is not settled: the operator error is far above the known rank-16 floor.
  - Optimisation at this budget and λ achieved a 11–25 % loss reduction.
- **Integrated transport.** The sender changes did not improve the sharp reflection on either seed; on s8601 it got worse. The global term, F2 and MR are preserved.
- **Architecture inference.** None about impossibility. The evidence does **not** support H-S at smoke scale. Source-specific supervision of the learned sender, at an equal-gradient weight and 1,000 steps, does not move the sender toward M64 in a way that generalizes to the evaluation incident field.

## Unresolved questions

- Whether a longer or differently weighted teacher fit (not authorised: no sweep or extension) would reach the rank-16 floor and transfer to CK2. Optimisation versus representation is not separated at this budget.
- Why the S1 fit generalizes by energy rather than lobe shape. The teacher normalisation (bin mean + 0.05 × mean) may favour level over peak; this is not isolated.
- How much of the remaining sharp error is the 64-bin flight floor, which no sender fit can remove: the exact sender itself gives CK2 box-A delivery ratio 1.21 and panel-face corr −0.33.

## Architecture Verdict

**NO_RELIABLE_IMPROVEMENT.**
- SENDER_IMP is false on both seeds, and RENDER_IMP is false on both. KEEP holds on both.
- Nothing is adopted. M64 is not deployed. Earlier verdicts are unchanged. No successor is started; the decision returns to the owner.

**Answers to the brief's questions.**

| question | answer |
|---|---|
| Can the existing sender output be supervised against M64? | **Yes.** The internal `Bq` at the S1 patches is the same quantity as `s · M64` (the C2_srcS1 identity reproduced to 8e-8); no head or architecture change is needed. |
| Can the current operator fit the sharp angular response? | **Not shown.** Within 1,000 steps the training-field error falls only 0.82 → 0.72; the operator stays about 1.0 from M64 and keeps about 30 % of the mirror peak. The rank-16 representation floor (0.46–0.52) is far from being reached, so this is not a representation verdict. |
| Does the fit generalize beyond the training directions? | Bin-level held-out directions do not exist (all 64 bins are excited in training). Across held-out geometries and lights the gain is about 3 %; on CK2 it is worse on s8601 (1.54 vs 1.29) with an energy overshoot of 1.7×. |
| Does direct sender supervision improve both seeds? | **No.** The sender criterion fails on both seeds. |
| Does it improve integrated sharp reflection? | **No.** s8601 is worse (0.062 vs P 0.044); s8602 shows no difference beyond noise. |
| Are global transport, F2 and MR preserved? | **Yes**, on both seeds. On the other box-A faces the global error rises slightly on s8601 (0.073 vs 0.054). |
| What gap remains relative to the exact-sender control? | T vs D with the exact S1 sender: 0.062 vs 0.033 (s8601) and 0.097 vs 0.019 (s8602). The exact-sender path itself has a 64-bin flight floor relative to physics on CK2. |

## Reproducibility

- **Commits and protocol.**
  - Freeze `a08c9bf`; protocol sha256 prefix e0a389463e172bc7 (committed LF file).
  - Results are committed with this worklog.
- **Server roots.**
  - Feasibility: `~/rest_x94dfeas` (2 audit runs + 1 probe; no training).
  - Dev: `~/rest_x94ddev`. Run 1 hit a CUDA OOM with 10 concurrent processes; the chain is now limited to 4.
  - Evidence: `~/rest_x94d`.
- **Checkpoints.**
  - own `739e7eb5…` / `cee9a09a…`.
  - Init D `141ebff2…` / `e68df71b…`.
  - Final T `697b82b7…` / `b3e36c5e…`, R `4e24c322…` / `5d0a761c…`, P `25f30458…` / `9e66deb3…`.
- **Teacher provenance.** M64 = `diag92.local_matrices` (quad87 enc_matrix, E87 dense quadrature) via `mat94.mq_ext`; the incident fields are the stored WL86 item fields and the CK2 `ctx.source`.
- **Seeds and budgets.** Seeds 8601 / 8602; order rng(seed + 9400); 1,000 steps per candidate; λ in `results/ecro_s94d/lam.json`.
- **Masks and references.** WL94 CK2 fields, images and masks; arm renders 86600 + r, 2 × 4,096 spp.
- **Cost.**
  - 14 min wall in total.
  - GPU mean 61 % (77 % of samples > 50 %).
- **Outputs.**
  - `results/ecro_s94d/`: s_s860x.json, final_s.json, audit.json, lam.json, repro.json, sender/*.json, models/*.json, pop/*.json, chain.status, gpu.txt.
  - `results/evaluation/94D/`, with a sha256 manifest:
    - figS1 summary (sender fit train / val / CK2, delivery, operator, E_H, exact-sender control, global bias);
    - figS2 training (incl. the S1 teacher loss);
    - figS3 textured;
    - figS4 images per variant (REF / own / D / T / R / P / D with exact sender, both seeds, shared absolute scales).
