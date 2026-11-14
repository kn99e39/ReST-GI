# Worklog — WL95A: Sharp Sender Representability & Flight Attribution Smoke (2026-11-15)

## Origin and scope

- **Brief.** "WL95A — Sharp Sender Representability & Flight Attribution Smoke" (user). It is a suffix of WL95 and a bounded, learning-free attribution batch.
- **Proposition tested.** "The deployed learned S1 sender is a major avoidable error source, while the residual sharp first-flight error must be separated into angular projection and propagation sampling before choosing a replacement representation."
- **Preserved:** ECRO, `model86.py`, `ecro86.py`, `pt86.py`, `mat94`/`an94` (imported unchanged); every checkpoint (own_s8601 / own_s8602 read-only); the ER closure; the CK2 scene, light and camera; the CK2 field and image references; masks; WL94–WL95 history.
- **Not done:** no training, tuning, renormalisation, rank or bin increase as repair, learned residual, e2/e3 rebalancing, or material/light/geometry change. No candidate is promoted.
- **Code.** `a95.py` and `run95a.sh` in [experiments/ecro_attr95a/](../../experiments/ecro_attr95a/). It also holds verbatim copies of the WL95 calculation modules under `wl95calc/`, with md5 equal to `results/wl95_calc/`.
- **Protocol.** [protocol/ecro_attr95a.json](../../experiments/ecro_attr95a/protocol/ecro_attr95a.json). Freeze commit `c152120`; protocol sha256 prefix fb136ad2da79fdca (committed file).
- **Where.**
  - LabServer63 `~/rest_x95a`, one detached chain, 21:02–21:08 UTC (6 min).
  - Local results: `results/ecro_attr95a/`.
  - Figures: `results/evaluation/95A/`.
- **Workload.** CK2 "none" only: one point light at [−1.45, −0.6, 1.7], rgb 30, grazing on the S1 panel. The E2 secondary workload was not used, because no semantically identical panel-tagged split was verified.

## A. Implementation Fact

**Verified contracts (Stage 0, all PASS; `results/ecro_attr95a/s0.json`).**

| test | result |
|---|---|
| T_BINS | 4 rings × 16 sectors; `pt86.Ctx.bin_of` = iz·16 + iφ on 4,096 random directions (agreement 1.0). Corrected W_T = ((d//16) + ½)/4 · 2π/64 sums to π. `an94.W` = W94 (8 × 8 assumption) confirmed: grazing-ring sectors 0–7 / 8–15 get ×0.5 / ×1.5 |
| T_MASKS | S1 = 1088–1129 (PSID 5, PMAT 1); box A = `T` = 1130–1183 (54 Lambert patches, material 4); panel-facing face = `T_panel_face` = rect 6 = 1130–1138 (9) |
| T_EVENT | CKF_none holds e2_p / e2_o / e3_p / e3_o for both halves; e2_p on the panel is exactly 0 (the panel cannot be its own k = 2 vertex) |
| T_COS | ER eval on a Lambert state = ρ/π · cos θ of the **second** argument (rel ≤ 1e-4); eval(wi = viewer, wo = light) · I/d² is the NEE radiance, as in `pt86a.field_src` |
| T_REPRO94D | WL94D CK2 flight-floor values reproduced to 1.5e-7 under the old convention (W94, deployed state, M64(s16), exact receiver): receivers 1.378 / 0.713 / 0.940; box A 1.211 / 0.716 / 0.528; panel face 1.206 / −0.334 / 0.598 (ratio / corr / rel L1). WL94D CK2 learned-sender rel L1 (W94) reproduced for both seeds |
| T_OPS | A1 = WL95 `nmf16_wl92` and A2 = `nmf16_l1irls` (from `S1_ops.npz`, sha256 8654ae79…). A3 = rank-32 non-negative L1 operator regenerated deterministically with the WL95 ladder code; it reproduces WL95 op L1 0.23306 and val field 0.14582 to 1e-6. All non-negative. The `nmf.npz` M64 equals `mq_ext()` |
| T_DEPLOYED_ID | the reconstructed deployed relation (u_state seed 86001) equals `build_state` bitwise |

**Diagnostic code added (WL95A namespace only).**
- **Sender arms on the S1 rows:**
  - A0: learned OWN a(cᵀs16) for s8601 and s8602;
  - A1, A2, A3 and A4 = M64, each applied to s16 (the deployed 16 fixed source points);
  - M64_spool: M64 applied to the source pooled over 64 source-point draws, i.e. 1,024 random points;
  - q256: the physical once-scattered table from `quad87.qtable`, P = 256, Mq = 1,024, restricted to S1, mean of seeds 95001 and 95002;
  - X1c: exact per-ray once-scattered radiance (ER eval at the exact light and outgoing directions, shadowed), evaluated at the centre of the hit panel patch;
  - X1: the same evaluated at the ray's hit point.
- **Propagation.** The flight code is `model86` / `eval93b` replicated in numpy with identical semantics: 512 rays per patch, 8 per bin, cos-weighted `rw`.
  - Deployed relation.
  - Pooled relations: 64 independent redraws of all state-ray offsets (rng 950000 + d) with matched `rw`, paired across arms. The pilot of 16 draws was extended to 64 by the predeclared boundary rule.
- **Receivers.** Every arm delivers only S1-mediated, exactly-2-event light through the exact receiver M64[pmat]. Box A and the panel face are Lambert, so the receiver step is exact there.
- **Render tables.** Each arm's table is its delivered S1 e2 plus the reference e2_o + e3_p + e3_o, identical across arms.
  - The oracle-table floor is the reference e2_p + e2_o + e3.
  - G_s<seed> (diagnostic) is reference e2 plus the own learned global term.
  - Rendering uses `mat94.render`: CK2 ROI_T higher-order image, 2 × 4,096 spp, seeds 86600 + r.
  - E_H uses the `an94.eh` noise bound.

## B. Measurement

All values are on CK2 "none" with one light, use W_T unless stated, and refer to the reference family CKF_none e2_p (mean of the 2 × 4,096 spp halves).

### B1. Sender (CK2 S1 field, 42 patches; deterministic; `main.json`)

| arm | vs M64(s16): rel L1 / energy / peak | vs physical q256: rel L1 / energy / peak | operator flux-L1 vs M64 | max directional albedo |
|---|---|---|---|---|
| A0 own s8601 (learned) | 1.041 / 0.433 / 0.038 | 1.083 / 0.420 / 0.032 | 1.018 (patch mean) | **2.26 (NOT passive)** |
| A0 own s8602 (learned) | 1.118 / 0.891 / 0.129 | 1.190 / 0.864 / 0.108 | 0.987 | **3.08 (NOT passive)** |
| A1 NMF16 | 0.957 / 0.458 / **0.018** | 1.013 / 0.444 / 0.015 | 0.603 | 0.960 |
| A2 L1-NMF16 | 0.895 / 0.111 / 0.002 | 0.916 / 0.108 / 0.002 | 0.380 | 0.806 |
| A3 rank 32 | 0.367 / 0.669 / 0.428 | 0.536 / 0.648 / 0.358 | 0.233 | 0.846 |
| A4 M64(s16) | 0 / 1 / 1 | 0.239 / 0.970 / 0.830 | 0 | 0.825 |
| M64(s_pool) | 0.055 / 0.973 / 0.975 | 0.221 / 0.943 / 0.810 | — | — |

- **Physical-q MC uncertainty.** Split-half rel L1 0.0070 (qA vs qB 0.014; energy 0.9995). This is far below |q − M64(s_pool)| = 0.221, so q is interpretable.
- **Source-point sampling.** A single 16-point draw is 0.117 rel L1 from the pooled source on average (max 0.28).

### B2. Delivery: exactly-2-event S1-mediated exitance (`an.json`)

Box A has 54 patches; the panel face is reported in `an.json` and figA2. The reference half-split noise n0 is 0.059 on box A and 0.070 on the panel face.

| arm | deployed: ratio / abs rel L1 / corr | pooled 64 draws: ratio / abs rel L1 (±SE) / corr | pooled per-draw abs rel L1: median / p90 / worst |
|---|---|---|---|
| A0 s8601 | 0.495 / 0.588 / 0.76 | 0.517 / 0.569 (0.004) / 0.78 | 0.596 / 0.626 / 0.659 |
| A0 s8602 | 1.534 / 0.901 / 0.75 | 1.630 / 0.965 (0.014) / 0.79 | 0.996 / 1.105 / 1.195 |
| A1 NMF16 | 0.921 / 0.623 / 0.72 | 0.975 / 0.655 (0.008) / 0.72 | 0.665 / 0.735 / 0.805 |
| A2 L1-NMF16 | 0.231 / 0.779 / 0.63 | 0.244 / 0.762 (0.003) / 0.64 | 0.773 / 0.793 / 0.823 |
| A3 rank 32 | 1.048 / 0.480 / 0.74 | 1.127 / 0.386 (0.018) / 0.88 | 0.521 / 0.692 / 0.775 |
| A4 M64(s16) | **1.208 / 0.524** / 0.72 | **1.544 / 0.546 (0.035)** / 0.96 | 0.726 / 1.148 / 1.450 |
| M64(s_pool) | 1.158 / 0.483 / 0.74 | 1.495 / 0.500 (0.034) / 0.96 | 0.665 / 1.113 / 1.394 |
| q256 (physical) | 1.077 / 0.534 / 0.68 | 1.422 / 0.433 (0.033) / 0.94 | 0.657 / 1.074 / 1.296 |
| X1c (exact direction, patch centre) | 0.719 / 0.388 / 0.81 | 1.037 / 0.268 (0.045) / 0.95 | 0.600 / 1.073 / 3.778 |
| **X1 (exact direction and point)** | 0.764 / 0.627 / 0.57 | **1.022 / 0.111 (0.041) / 0.99** | **0.617 / 1.186 / 2.212** |

- **Peak.** The reference peak is patch 5 (panel-face +y column). Pooled X1 places its value there at 0.98× the reference. M64 pooled gives 1.26× there but has its arm peak at patch 8. A0 s8601 peaks at patch 2 with 0.30× at the reference peak.
- **Corrected vs old convention.** On the Lambert box A the W_T and W94 exitances agree to ≤ 4e-3 (1.208 vs 1.211; 0.524 vs 0.528). The WL94D box-A floor therefore stands numerically. What it lacks is the pooled view: the deployed draw is a low draw of a biased estimator.

### B3. Integrated ROI (CK2 ROI_T, 2 × 4,096 spp per arm; noise bound about 0.0083 per arm, so a difference needs > about 0.0167)

| table | E_H | signed |
|---|---|---|
| oracle-table floor | 0.0118 | +0.003 |
| deployed A4 (M64) | 0.0152 | −0.001 |
| deployed A1 / A3 / A2 | 0.0206 / 0.0165 / 0.0400 | −0.002 / −0.005 / −0.040 |
| deployed A0 s8601 / s8602 | 0.0249 / 0.0473 | −0.023 / +0.040 |
| pooled A4 / M64(s_pool) / q256 | 0.0245 / 0.0223 / 0.0186 | +0.022 / +0.019 / +0.013 |
| pooled X1c / **X1** | 0.0128 / **0.0124** | −0.007 / +0.001 |
| pooled A1 / A3 / A2 | 0.0209 / 0.0204 / 0.0395 | −0.001 / +0.010 / −0.039 |
| G_s8601 / G_s8602 (reference e2 + own learned G) | 0.0251 / 0.0150 | −0.016 / −0.006 |

### B4. Compute (RTX 3080 Ti 12 GB, LabServer63)

- The main stage took 74 s in total:
  - learned factors 2.1 s;
  - 64 source draws 0.7 s;
  - q256: 2.3 s for 2 seeds × 11.0 M MIS lanes;
  - deployed relation 1.6 s;
  - 16 pilot draws 17 s;
  - extension to 64 draws 49 s.
- Peak torch memory was 169 MiB.
- The deployed relation sends 9,995 rays to S1 patches.
- The final set has 38 renders at 8.6–11.4 s each (374 s total, pool 3).
- Chain wall time was 6 min; GPU mean was 30 % over 18 samples.
- [FACT] The main stage is CPU-bound: the flights and receivers run in numpy. It was not probed for GPU utilisation at full scale before the freeze; only the dev path ran. This is a disclosed shortfall against the standing utilisation rule, though small in absolute time.

## C. Observation

- **Energy and peak at the sender.** At CK2's grazing (ring-0) incidence, every material-level rank-16 operator loses the S1 lobe almost completely: peak 0.018 (NMF16) and 0.002 (L1-NMF16), with energy 0.46 and 0.11. Rank 32 keeps 0.43 of the peak at energy 0.67. The learned senders keep 0.04 and 0.13 of the peak.
  - The learned per-patch S1 operators are non-passive for some incident bins (directional albedo up to 2.3 / 3.1). The bound operators and M64 are passive (≤ 0.96).
  - The WL95 validation-field bounds (rank-16 field rel L1 0.36–0.54) do **not** carry over to this incidence. Ring 0 holds only 4.3 % of training irradiance.
- **Delivery, rank arms.** NMF16 delivers 0.63× M64's box-A flux (pooled) with a map 0.65 away from M64's. L1-NMF16 delivers 0.16× M64's. Rank 32 delivers 0.73× with map distance 0.27.
  - Against the physical reference, NMF16's under-delivery partly offsets M64's over-delivery (A1 pooled ratio 0.98 vs A4 1.54), but its footprint is wrong (abs rel L1 0.66).
- **Delivery, flight chain.** Pooling 64 independent relation draws does **not** reduce the binned M64 sender's error; it exposes an expected-value over-delivery:
  - ratio 1.21 deployed → 1.54 pooled;
  - rel L1 0.52 → 0.55;
  - per-draw p90 1.9.

  The deployed WL94D draw is a favourable draw of a biased estimator. Further along the chain:
  - pooled source points (1.50) and physical q (1.42; incident projection removed) change little;
  - exact per-ray evaluation at the patch centre brings the ratio to 1.04 (rel L1 0.27);
  - at the exact hit point it reaches 1.02 / 0.111 with corr 0.99. That is at the reference noise (0.059) plus the arm's SE (0.041). Stage 3 is therefore unnecessary.
- **Variance.** The exact per-ray estimator is unbiased but heavy-tailed per draw: single-draw median rel L1 0.62, worst 2.2. The deployed single draw of X1 (0.627) is *worse* than the deployed binned M64 (0.524). Fidelity needs both the removal of the binned bias and many relation draws; 64 draws were used here, i.e. 32,768 relation rays per receiver patch.
- **ROI.** The image-level effects are mostly below the 2 × 4,096 spp render resolution:
  - pooled X1 = floor (0.0124 vs 0.0118);
  - pooled M64 = 0.0245, +0.0127 over floor, below the 0.0167 tolerance;
  - the deployed M64 is near the floor (0.0152) because its draw happens to under-deliver;
  - learned s8602 (0.047) is the only arm clearly beyond tolerance.
  - The higher-order diagnostic G_s<seed> sits 0.013 (s8601) and 0.003 (s8602) above the floor.

## D. Interpretation

- **Physical correctness.** Every contract reconciles. WL94D is reproduced. q is precise (split-half 0.007). The exact per-ray chain converges to the path-traced reference within noise. So the reference, the event family and the receiver semantics are consistent, and the residual of every other arm is a property of that arm.
- **Learned sender (A).** On CK2 it is a large error source in delivery: 0.52× / 1.63× flux and a misplaced footprint. It is seed-signed, as in WL94C/94D, and non-passive. It is far from M64 at the field level, and slightly worse than the rank-16 bounds in rel L1 but not in peak. **Avoidable within rank 16 on this workload?** Not shown: the rank-16 bounds are just as wrong at this incidence.
- **Rank-16 capacity (B).** [MEAS] At grazing incidence, material-level non-negative rank-16 operators fit to the training distribution do not represent the S1 response (peak ≤ 0.02) and do not reproduce M64's delivery. [INF] This is a capacity / fit-distribution limit for grazing S1 rows, consistent with WL95's empirical ceiling. It is measured here with application-level delivery, not only in operator norms. A grazing-weighted rank-16 fit was **not** tried; it would require fitting to an incidence regime selected from the evaluation workload, which is not authorised.
- **Directional projection vs sampling (C vs D).** [MEAS]
  - The binned patch × bin sender table, M64(s) or physical q, carries a **systematic** over-delivery to box A of about +40–50 % in expectation. More relation samples do not remove it.
  - It is removed by evaluating the sender radiance at the exact outgoing direction (Ra = 0.232 ± 0.047 of E0 = 0.464), and further by the exact position (Rp = 0.157 ± 0.060).
  - Incident-side projection is small (0.066 ± 0.005).
  - [INF] The loss is the patch-constant and bin-constant outgoing representation of a sharp, position-dependent lobe (a mirror's radiance depends jointly on position and direction), not the 64-bin incident encoding and not relation sampling. Removing it trades bias for per-draw variance, which is the WL91 C3URX1 pattern.
- **Confounds that remain.**
  - One scene, one light, grazing incidence, one sharp material, rigid static geometry.
  - X1c changes two things relative to q256: binned to exact outgoing direction, and patch-average to patch-centre position. q256 → X1c is "outgoing angular + average-to-centre"; only X1c → X1 isolates spatial position.
  - ROI resolution is limited by 2 × 4,096 spp renders.
  - Rank bounds were fit with WL95's W94-weighted objectives (A3) or flux-Frobenius weights (A1). No grazing-specific bound exists.
- **Evidence for an actual angular representation floor.** On this workload, yes for the **binned outgoing sender table**: its expected delivery is biased independently of sampling, and the exact-direction evaluation closes the gap. This is evidence about the patch × 64-bin carrier for a grazing sharp lobe. It is not a statement about other bin layouts, adaptive bases or other scenes.

## E. Architecture Verdict (predeclared rules)

- **RANK16_DELIVERY = INADEQUATE_ON_TESTED_WORKLOAD.**

  | arm | passive | flux vs A4 | map vs A4 | ROI E_H vs A4 |
  |---|---|---|---|---|
  | A1 | yes | 0.63 ✗ | 0.65 ✗ | 0.021 vs 0.024 ✓ |
  | A2 | yes | 0.16 ✗ | 0.84 ✗ | 0.040 vs 0.024, within tol ✓ |

  Both fail criteria (ii) and (iii) clearly. Rank 32 (A3, reported) also fails, with flux 0.73 and map 0.27. The ROI criterion passes only because the M64 comparator itself over-delivers in expectation, and the render noise is about 0.008.
- **FLIGHT_ATTRIBUTION = ANGULAR_PROJECTION_EVIDENCED.**
  - Ra = 0.232 ± 0.047 is ≥ 0.25 × E0 = 0.116 and above 2 SE.
  - Rs = 0.024 ± 0.034 does not count.
  - E(X1, pooled) = 0.051 ≤ 0.5 × E0.
  - Additionally, not a rule category: sender-side **spatial** projection also counts (Rp = 0.157 ± 0.060 ≥ 0.116, just above 2 SE). The exact estimator's per-draw variance is a co-limit for any practical carrier.
- **NEXT_DESIGN_TARGET = UNDETERMINED.**
  - s8601: no one-at-a-time ROI component exceeds tolerance (dS 0.013, dF 0.003, dG 0.013; tol 0.017).
  - s8602: dS 0.035 > dF/dG + tol, which is LEARNED_SENDER on one seed only.
  - At ROI level the render resolution cannot rank the components; the decisive differences are at the delivery level. This is an evidence classification, not an implementation permission.

**Answer to the completion question.**
1. On CK2, a physically admissible rank-16 sender does **not** provide sufficient application-level fidelity at the delivery level: grazing-incidence peak ≤ 0.02, flux 0.63× or 0.16× of M64, footprint wrong. Its ROI image error is not resolvable from M64's at 2 × 4,096 spp.
2. The remaining first-flight error of an exact-bin sender is caused by **directional (outgoing-angular) and positional projection of the sender table**, not by relation sampling. Removing it requires exact sender evaluation, which in turn requires many relation draws to control variance.

Nothing is adopted. No successor is started. The decision returns to the owner.

## Unresolved questions

- Whether any fixed-basis local carrier (more bins, adaptive bins, a lobe basis) removes the outgoing bias on CK2 without the exact estimator's variance. Untested; not authorised.
- Whether a grazing-inclusive training distribution changes the rank-16 conclusion. The fit distribution has 4.3 % ring-0 irradiance.
- The ROI-level ranking of sender vs higher-order errors at higher render spp.
- Generality: other scenes, lights, area lights, sharp movers and non-rigid edits are untested.

## F. Reproducibility

- **Commits.** Freeze `c152120` (protocol sha256 prefix fb136ad2da79fdca). Results are committed with this worklog.
- **Environment.** LabServer63, RTX 3080 Ti 12 GB, `~/rest_x95a`, Python env `~/NeuralShaderDemo/external/8dna26/.venv`, `cuda_ad_rgb` with the OptiX path from `env.sh`. The dev run is `~/rest_x95adev` (not evidence).
- **Inputs.**
  - own_s8601 / own_s8602 (WL92A, `~/rest_x92a`).
  - CKF_none and REF_ER_CK_none from `~/rest_x94`.
  - WL94D audit and SN_own from `~/rest_x94d`.
  - WL95 `S1_ops.npz` (8654ae79…), `wl95_calc*.json` from `~/wl95_calc`.
  - `ops95a.npz` (d4d02e5a…; A1–A4) is newly saved.
- **Seeds.**
  - Relation draws: rng 950000 + d, d = 0..63.
  - Source draws: rng 951000 + d.
  - q MIS: 95001 / 95002.
  - Renders: 86600 + r, r = 0, 1.
  - Rank-32 regeneration: 9500 + 320 + c.
- **Commands.** `setsid nohup bash code/experiments/ecro_attr95a/run95a.sh ~/rest_x95a ~/rest_x94 ~/rest_x94d ~/wl95_calc ~/rest_x92a`. The chain runs `a95.py stage0 | main | an` and `mat94.py render`. Extension to 64 draws re-runs `main` with `A95_EXTEND=1`, triggered by the predeclared rule after the 16-draw pilot.
- **Outputs.**
  - `results/ecro_attr95a/`: s0.json, main.json, an.json, deliv.npz (per-patch exitance per arm, deployed and per draw), ops95a.npz, chain.status, gpu.txt.
  - `results/evaluation/95A/`, with a sha256 manifest:
    - figA1 senders;
    - figA2 delivery;
    - figA3 box-A footprint maps (shared scale);
    - figA4 ROI renders and signed errors (shared scale, one exposure).
