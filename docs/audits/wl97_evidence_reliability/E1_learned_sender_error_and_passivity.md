# WL97 audit, track E1: independent reproduction of P0-1 (learned S1 sender error; passivity)

Date: 2026-11-16. Auditor: independent agent, track E1 (resumed after an interrupted first attempt of the same track; the first attempt's
scripts and outputs were re-checked, completed and are part of this report). Repository HEAD efcf86a, branch `Selective_Recompute`.
No existing file was modified; nothing was trained or tuned; nothing was committed.

**Claim under audit (P0-1):** "The learned S1 sender has large avoidable error, and the deployed learned operator may violate passivity."

**Short answer.**

| half of P0-1 | proposed verdict | one-line reason |
|---|---|---|
| "the learned S1 sender has large error" (CK2, OWN s8601 / s8602) | **VERIFIED** | every original number reproduces (most to 4 digits) with own code, an own physical Monte Carlo table and the correct 4 x 16 weights; robust to source-point and state redraws; also holds in the measure the deployed relation reads, and (weaker) on the training / validation distribution |
| "... avoidable" | **CONDITIONALLY_VALID** | avoidable only relative to the explicit 64 x 64 bin operator M64 at field level; not shown avoidable within the tested rank-16 learned representation (the worklog says so itself); at delivery level the M64 alternative is not clearly better for s8601; "rank 16 cannot represent it" would be an UNTESTED_INFERENCE (CK2 excites 4 incident bins) |
| "the deployed learned operator may violate passivity" (operator level) | **VERIFIED** (the wording understates it) | max directional albedo 2.264 / 3.077 reproduced; 10 % / 15 % of all S1 (patch, incident bin, channel) entries exceed 1, on all 42 patches, also on Lambert patches; violating bins are excited by the training data |
| passivity as an explanation of the CK2 error, or as a stability problem | **UNTESTED_INFERENCE** if read that way (CONDITIONALLY_VALID as stated) | passivity was never imposed (declared in the WL86 protocol), `apply()` is non-recursive, and on the CK2 input the learned sender emits 0.28 / 0.58 of the irradiance it receives (physical 0.67): the CK2 failure is under-emission plus directional misallocation, not energy creation |

---

## 1. Scope and method

- Target evidence: WL95A section B1 (`docs/worklogs/95A_WORKLOG_SHARP_SENDER_REPRESENTABILITY_AND_FLIGHT_ATTRIBUTION.md:53-63`),
  interpretation D (`:143`), code `experiments/ecro_attr95a/a95.py` (read, not imported), stored outputs
  `~/rest_x95a/results/{main.json, an.json, senders.npz, deliv.npz, ops95a.npz}` (LabServer63; local copies of the json / npz except
  `senders.npz` under `results/ecro_attr95a/`).
- Own code only: `experiments/audit97/E1/e1_audit.py` (stages `conv`, `sender`, `train`, `debug`), `e1_read.py`, `e1_rank.py`.
  It imports, as physical infrastructure only, `mat94.ctx_ck2` (CK2 scene, Mitsuba intersection), `pt86.Ctx.state_rays` / `Ctx.source`
  (each cross-checked against an own re-implementation or regeneration), the frozen ER closure `closure83_dr.DrClosure83`, and the frozen
  checkpoint through `mat94.load_frozen` / `model86.ECRO`. It does not import `a95.py`, `an94.py` or their helpers.
- Frozen code used on the server: `~/rest_x95a/code/experiments` (commit c152120, the WL95A freeze), read-only. Own server root
  `~/rest_x97/E1/` only.
- Trace followed for each number: protocol / worklog statement -> executable code line -> input artifact (hash) -> own recomputation ->
  comparison -> inference.

Quantities (my own derivation from `scene86` and `model86`):

- Direction bins: 4 rings in z = cos(theta) x 16 azimuth sectors, 64 equal-solid-angle bins, DOM = 2 pi / 64; bin d = iz * 16 + iphi.
  Correct cos-dOmega weight of bin d: `W_T[d] = ((d // 16) + 1/2) / 4 * DOM`, ring values 0.012272 / 0.036816 / 0.061359 / 0.085903,
  sum = pi exactly (`conv.json: bins`). My own bin function agrees with the production one on random directions (1.0).
- Source encoding `s[n, m, c]`: irradiance (W m^-2) arriving at patch n from incident bin m, mean over 16 fixed patch points.
- Learned once-scattered sender `Bq[n, d, c] = sum_r a[n, d, c, r] * (sum_m c[n, m, c, r] s[n, m, c])` (`model86.py:112-115`), rank 16 per colour.
- Metrics (as the worklog defines them): rel L1 = sum W_T |X - Y| / sum W_T Y; energy = sum W_T X / sum W_T Y; peak = X at Y's arg-max bin
  per patch and channel, summed, over Y's maxima. A zero sender scores rel L1 = 1.

## 2. What was run

All server runs on LabServer63 with `. ~/rest_x95a/env.sh`, GPU peak (torch) 264 MiB, each stage well under one minute.

| stage | command | output | wall |
|---|---|---|---|
| conventions | `bash run_e1.sh conv` | `results/audit97/E1/conv.json` | < 1 min |
| sender / passivity / delivery / redraws | `bash run_e1.sh sender` | `sender.json`, `sender_e1.npz` | 33 s |
| own estimator cross-check | `bash run_e1.sh debug` | `debug.json` | < 1 min |
| training / validation distribution | `bash run_e1.sh train` | `train.json` | < 1 min |
| relation-read measure | `$PY e1_read.py` (env as in `run_e1.sh`) | `read.json` | < 1 min |
| rank indication (local, numpy) | `python experiments/audit97/E1/e1_rank.py /c/Projects/ReST-GI` | `rank.json` | seconds |
| local spot check of the decisive numbers | inline numpy on `sender_e1.npz` + `results/ecro_attr95a/ops95a.npz` | (console; values in section 3) | seconds |

Seeds: physical q 97011 / 97012 (two halves, 4 launches x 64 x 64 stratified directions per (patch, bin) each); own M64 97021 / 97022;
box-A references 97031 / 97032 (area sampling), 97041 / 97042 (hemisphere sampling); source-point redraws rng 97100 + j (j < 32);
state redraws rng 97200 + j (j < 6). Deployed tables: protocol `table_seed` 86001 (regenerated; equal to the production tables).

Input hashes (sha256 prefix): `senders.npz` 5d365d19d6dffadc, `ops95a.npz` d4d02e5a2f423ce9, `main.json` 28d104991bcad6f7,
`own_s8601.pt` 739e7eb50b25d329, `own_s8602.pt` cee9a09a289fca8c, `CKF_none.npz` 3a6238a8a42d1db4, deployed state digest d97fa96d1cbafe5e.

## 3. Results

### 3.1 Conventions (special checks; `conv.json`)

| check | result |
|---|---|
| ER cosine convention, analytic Lambert (albedo 0.7) | `eval(a, b)` = rho / pi * cos(theta of the SECOND argument), max rel deviation 1.8e-7 (first-argument hypothesis: 72x off). So `eval(viewer, light) * I / d^2` is outgoing radiance |
| S1 closure reciprocity `f(a, b) = f(b, a)` | max rel 1.2e-6 |
| S1 physical directional albedo (own importance + uniform estimators) | R 0.38-0.49, G 0.57-0.85, B 0.55-0.78 over mu = 0.02 ... 0.99: always < 1 |
| S1 material | GGX-type lobes with alpha about 0.026-0.075: sharp |
| patch frame vs local frame at panel hits | 3e-8 |
| CK2 workload | one point light 5 cm above the panel plane (panel x = -1.5, light x = -1.45); cos of incidence at S1 patch centres 0.026 / 0.041 / 0.120 (min / median / max) |
| `an94.W` (8 x 8 assumption) vs W_T | ring-0 sectors 0-7 weighted x0.5, sectors 8-15 x1.5 (see F8) |
| deployed tables / source / state | regenerated `u_state`, `u_src` equal the production arrays; own source encoding equals production to 2e-5 abs (rel L1 1e-6 on S1); 9,995 state rays hit S1 (the worklog's count) |

The CK2 source on S1 is entirely in incident ring 0, bins 12-15 (shares 0.023 / 0.209 / 0.356 / 0.412).

### 3.2 Step 1: reconstruction of the learned sender

`Bq = a (c^T s16)` from `model.factors` on the deployed state, contracted with own einsum: **max abs difference to the stored
`senders.npz` A0 arrays = 0.0 for both seeds** (stored maxima 43.29 / 126.35). The production `model.apply(..., use_global=False)`
single-flight term equals my own chain (own flight of own Bq, own receiver contraction) to 3.9e-8 rel L1, which confirms that the
deployed operator uses the factor pair exactly as assumed. Factors are non-negative (min 2e-13).

### 3.3 Step 2: independent physical table and M64

| comparison | rel L1 / energy / peak | noise |
|---|---|---|
| stored q256 vs own q | 0.026 / 1.001 / 1.002 | own split-half (half difference) 0.0216; stored qA vs qB 0.014 |
| own M64 operator vs stored A4 (flux-L1) | 0.0117 (by incident ring 0.045 / 0.009 / 0.005 / 0.012) | own split-half 0.0047 |
| own M64 (s16) vs stored A4 = M64(s16) | 0.044 / 0.971 / 0.934 | - |
| stored M64(s16) vs own q | 0.241 / 0.971 / 0.832 (worklog vs q256: 0.239 / 0.970 / 0.830) | - |
| stored `ops95a` A4 applied to s16 vs stored A4 sender | 0 / 1 / 1 | - |

The stored physical table is confirmed by an independent Monte Carlo within its noise. The stored M64 is confirmed to about 1 % in flux;
the 4.5 % residual in the grazing incident ring is above my MC noise and is therefore a definitional difference in how the incident-bin
average is taken (mine: uniform in solid angle over both bins). It moves no P0-1 number by more than 0.003 (below).

### 3.4 Step 3: sender metrics with the correct 4 x 16 weights (`sender.json: sender_metrics`)

| sender | reference | original (WL95A B1) | reproduced (W_T) | with `an94.W` as coded (W94) |
|---|---|---|---|---|
| A0 s8601 | M64(s16) stored | 1.041 / 0.433 / 0.038 | **1.0409 / 0.4330 / 0.0378** | 1.0746 / 0.5567 / 0.0378 |
| A0 s8602 | M64(s16) stored | 1.118 / 0.891 / 0.129 | **1.1182 / 0.8912 / 0.1286** | 1.2208 / 1.1230 / 0.1286 |
| A0 s8601 | own M64(s16) | - | 1.0391 / 0.4462 / 0.0405 | - |
| A0 s8602 | own M64(s16) | - | 1.1197 / 0.9182 / 0.1377 | - |
| A0 s8601 | physical q (orig: q256; mine: own q) | 1.083 / 0.420 / 0.032 | **1.0830 / 0.4204 / 0.0316** | 1.1307 / 0.5633 |
| A0 s8602 | physical q | 1.190 / 0.864 / 0.108 | **1.1912 / 0.8652 / 0.1079** | 1.3240 / 1.1364 |
| A1 NMF16 | M64(s16) / q | 0.957 / 0.458 / 0.018; 1.013 / 0.444 / 0.015 | 0.9572 / 0.4584 / 0.0181; 1.0126 / 0.4450 / 0.0151 | - |
| A2 L1-NMF16 | M64(s16) / q | 0.895 / 0.111 / 0.002; 0.916 / 0.108 / 0.002 | 0.8952 / 0.1114 / 0.0020; 0.9162 / 0.1081 / 0.0016 | - |
| A3 rank 32 | M64(s16) / q | 0.367 / 0.669 / 0.428; 0.536 / 0.648 / 0.358 | 0.3671 / 0.6687 / 0.4278; 0.5359 / 0.6492 / 0.3585 | - |

Numerical disagreement with the worklog: none beyond rounding (largest 0.001). The learned sender is at or above the zero-sender level in
rel L1 (1.04-1.19 > 1), keeps 3 % / 11-13 % of the physical peak, and is seed-signed in energy (0.42 vs 0.87).
Outgoing energy by ring (share of the physical total): physical q 0.635 / 0.240 / 0.094 / 0.030; s8601 0.093 / 0.181 / 0.124 / 0.023;
s8602 0.189 / 0.343 / 0.268 / 0.065. The learned senders do not produce the grazing (ring 0) specular lobe and put energy in rings 1-2.

Sensitivity (`sender.json: source_point_redraws, state_redraws`):

- 32 redraws of the 16 source points (own source code; learned factors and relation fixed), rel L1 / energy / peak vs own q, mean +- sd [min, max]:
  s8601 1.079 +- 0.015 [1.052, 1.120] / 0.412 +- 0.028 / 0.031 +- 0.002; s8602 1.182 +- 0.031 [1.119, 1.261] / 0.848 +- 0.052 / 0.107 +- 0.006;
  M64 0.272 +- 0.029 [0.233, 0.348] / 0.956 +- 0.067 / 0.818 +- 0.074.
- 6 redraws of the state rays: the learned sender changes by 0.016-0.033 rel L1; its error vs q stays 1.08 / 1.18-1.19.

The conclusion does not depend on the 16 fixed source points or on the state realization.

### 3.5 Is the comparison like-for-like? Is the sender wrong where it is read? (`read.json`)

`Bq` is an internal intermediate of `model86.apply` (`model86.py:109-124`): the model output is `a (c^T flight(Bq) + P Q u)`; `Bq` itself is
never output. The OWN objective (`own92a.py:167-176`) supervises the single-flight output against the exactly-two-flight target and the
global output against the rest; `Bq` is constrained only through the fixed flight and the tied receiver factors. Entries of `Bq` that no
state ray reads are constrained by nothing, so an all-bin field metric could overstate the functional error. Checked:

- The deployed relation reads 2,185 of the 2,688 S1 (patch, outgoing bin) entries (9,995 rays); they hold 74 % of the physical W_T energy.
- Weighted by the power the relation actually transports (`w = sum of ray weight x receiver area`), vs own q:

| sender | all receivers: rel L1 / energy | box A receivers | other receivers | W_T, read entries only | W_T, unread entries only |
|---|---|---|---|---|---|
| A0 s8601 | 1.062 / 0.386 | 1.005 / 0.419 | 1.088 / 0.372 | 1.080 / 0.495 | 1.091 / 0.212 |
| A0 s8602 | 1.162 / 0.804 | 1.396 / 1.354 | 1.055 / 0.554 | 1.244 / 1.054 | 1.042 / 0.338 |
| M64(s16) | 0.248 / 0.944 | 0.225 / 1.113 | 0.258 / 0.867 | 0.240 / 0.996 | 0.245 / 0.901 |
| own q half A vs half B (noise) | 0.049 | 0.019 | 0.063 | 0.039 | 0.056 |

The error is not an artefact of unread entries: in the relation's own measure the learned sender is still at the zero-sender level.
By receiver surface, the largest physical destination of S1 light (surface id 2, 535 rays, power 3.11) receives 0.19 / 0.51 from the
learned senders (6 % / 16 %), while surface id 1 (physical 0.034) receives 0.30 / 0.22 (9x / 6x).

Like-for-like status of the WL95A B1 comparison: same deployed state, same 16 source points, same bins, same (correct) weights, same event
content (once-scattered only) for A0, A1-A4 and q. The only non-identical element is that q is not restricted to the 16 source points
(it integrates over the patch); M64(s16) vs M64(s_pool) is 0.055 in the worklog and the 32 redraws above bound that effect.

### 3.6 Step 4: passivity

**Definition (first principles).** The per-patch local operator `A[m, d] = sum_r c[m, r] a[d, r]` (per colour) maps irradiance arriving in
incident bin m (W m^-2) to the bin-averaged outgoing radiance of bin d. Unit irradiance in bin m produces exitance
`alpha[m] = sum_d A[m, d] * W_T[d]` (the model's own flight uses the same cos-dOmega bin measure: ray weights sum to 3.1404). Because A is
non-negative and linear, exitance <= irradiance for every non-negative input if and only if `alpha[m] <= 1` for every m. `alpha` is the
directional (per-incident-bin) albedo; this is also what `a95.py:149` computes. Normalisation check that does not depend on the worklog:
the learned operator's MEAN albedo on Lambert patches is 0.709 / 0.720 (material 3, true 0.7) and 0.510 / 0.510 (box A, true 0.5); M64's
albedo is 0.58-0.66 by ring, inside the physical S1 range of section 3.1.

| operator | max directional albedo: original | reproduced |
|---|---|---|
| A0 s8601, 42 S1 patches | 2.26 | **2.2637** (patch 3, incident bin 9, green) |
| A0 s8602, 42 S1 patches | 3.08 | **3.0774** (patch 2, incident bin 19, green) |
| A4 = M64 (stored) | 0.825 | 0.8247; own M64 0.8263 (exact-cos variant 0.8416) |
| A1 / A2 / A3 | 0.960 / 0.806 / 0.846 | 0.9601 / 0.8057 / 0.8456 |

Further statistics the worklog does not give (`sender.json: passivity`):

| | s8601 | s8602 |
|---|---|---|
| albedo median / mean / p90 over the 42 x 64 x 3 S1 entries | 0.506 / 0.561 / 1.013 | 0.456 / 0.552 / 1.163 |
| fraction of entries > 1 | 10.4 % | 15.4 % |
| S1 patches with at least one violation | 42 / 42 | 42 / 42 |
| incident bins with a violation on some patch / channel | 36 / 64 | 28 / 64 |
| fraction > 1 by channel R / G / B | 2.1 / 16.6 / 12.5 % | 4.9 / 23.4 / 17.9 % |
| max by incident ring 0 / 1 / 2 / 3 | 2.26 / 2.12 / 1.77 / 1.44 | 2.98 / 3.08 / 2.15 / 1.31 |
| max albedo over 6 state redraws | 2.20-2.25 | 3.06-3.13 |
| max S1 albedo over 232 training geometries (min / median / max) | 1.99 / 2.26 / 3.23 | 2.64 / 2.84 / 3.82 |
| other materials, max albedo (fraction > 1): floor ER / sphere ER / Lambert 0.7 / box A / box B | 1.82 (0.7 %) / 2.08 (4.0 %) / 1.07 (0.1 %) / 1.69 (0.6 %) / 2.02 (0.6 %) | 1.33 (0.1 %) / 1.92 (4.1 %) / 1.46 (1.5 %) / 2.10 (0.9 %) / 1.26 (1.2 %) |

Non-passivity is therefore a real, robust, scene-wide property of the trained weights, not an outlier on one patch and not specific to S1.

**Are the violating bins ever excited by the training data?** Yes (`train.json`; every 4th kept training geometry = 232 files, 898
lightings with S1 lit; all 55 validation files, 214 lightings). Every one of the 64 incident bins of S1 is excited by direct light in the
subsample (least excited bin: 0.026 % of patch-lightings; 13 bins below 0.1 %). 46 % (s8601) / 52 % (s8602) of the training direct
irradiance on S1 arrives in bins that violate on some patch / channel; 9.7 % / 12.0 % arrives on entries whose own albedo exceeds 1
(validation 9.2 % / 11.2 %). The per-bin violation fraction is uncorrelated with how often the bin is excited (corr with log frequency
0.03 / 0.06). The violations are not confined to unsupervised inputs.

**Does it matter on the input actually applied?**

| once-scattered exitance / irradiance on S1 | s8601 | s8602 | physical q | M64(s16) |
|---|---|---|---|---|
| CK2, total | **0.283** | **0.582** | 0.672 (W_T measure; 0.517 with the exact cosine) | 0.653 |
| CK2, per channel R / G / B | 0.19 / 0.35 / 0.30 | 0.46 / 0.65 / 0.64 | - | - |
| CK2, patches with out > in (per-patch max) | 1 / 42 (1.12) | 19 / 42 (1.44) | 0 | 0 |
| CK2, excited entries with albedo > 1 | 8 / 189 | 78 / 189 | - | - |
| CK2, share of irradiance on entries > 1 | 1.5 % | 10.7 % | - | - |
| CK2, share of learned exitance produced by entries > 1 | 6.8 % | 23.2 % | - | - |
| CK2, 32 source redraws, total out / in | 0.281 +- 0.004 | 0.580 +- 0.013 | - | 0.653 |
| training subsample, pooled | 0.589 | 0.580 | - | 0.645 |
| training subsample, lightings with out > in (max) | 8 / 898 (1.32) | 10 / 898 (1.60) | - | 0 |
| validation, lightings with out > in (max) | 1 / 214 (1.11) | 4 / 214 (1.27) | - | 0 |

On CK2 both learned senders emit LESS than the physical sender in total. For s8601 non-passivity is practically inactive on this input.
For s8602 nearly half of the patches (the weakly lit ones) emit more than they receive, but the total is 0.87 of physical; its 1.5x
box-A over-delivery occurs at below-physical total energy, i.e. it is a directional misallocation (energy sent into rings 1-2, where box A
is), not energy creation. [Inference from the ring shares of section 3.4 and the box-A column of section 3.5; not isolated by a counterfactual.]

Context that limits the meaning of the passivity statement:

- No passivity was imposed by design: `experiments/ecro86/protocol/ecro86.json:12` ("no single-scattering passivity / row-sum constraint on
  the multi-flight response") and `docs/worklogs/86_WORKLOG_ECRO_GLOBAL_RESPONSE.md:41`. It is not a broken contract.
- `model86.apply` is non-recursive (one learned flight plus a low-rank global term), so albedo > 1 cannot make the deployed operator
  diverge. It is a physical-validity defect of the local factor pair, relevant to any recursive use and to interpreting `a c^T` as a
  material operator, not a stability failure of the deployed model.

### 3.7 Step 5: what "avoidable" is relative to

- The proposition WL95A set out to test is "the deployed learned S1 sender is a major avoidable error source" (`95A:6`). Its own
  interpretation is: "Avoidable within rank 16 on this workload? Not shown: the rank-16 bounds are just as wrong at this incidence"
  (`95A:143`). P0-1 as worded in the audit brief is therefore stronger than what the worklog concluded.
- Relative to **M64 at the same state, source points, weights and event content** (like-for-like, section 3.5): the gap is large at field
  level (1.08 / 1.19 vs 0.24 against physical q; 0.03 / 0.11 vs 0.83 peak). "Avoidable" is supported in this sense only: an explicit,
  learning-free 64 x 64 bin operator in place of the learned rank-16 factor pair.
- Relative to **rank-16 non-negative operators fitted to the training distribution** (A1, A2): on the same input these score
  0.957 / 0.895 rel L1 and 0.018 / 0.002 peak; the learned senders are 0.08-0.22 worse in rel L1 and better in peak. Not avoidable in
  this sense, as far as tested.
- **Rank 16 is not, by itself, a capacity limit for the CK2 input** (`rank.json`; local linear algebra on the stored M64, no learning):
  CK2 excites 4 incident bins, and those bins carry 2.0 % of the training direct irradiance on S1. Unconstrained weighted truncated SVD
  of M64, applied to s16 and scored against M64(s16) (rel L1 / energy / peak):

  | incident weighting of the rank-16 fit | rank 16 | rank 32 |
  |---|---|---|
  | flux (ring-centre cosine) | 1.877 / 1.224 / 0.529 | 1.101 / 1.013 / 0.840 |
  | training direct-irradiance shares | 1.327 / 0.920 / 0.724 | 0.524 / 1.009 / 0.945 |
  | flux, ring 0 x 10 | 0.238 / 1.186 / 0.978 | 0.099 / 1.003 / 0.994 |
  | flux, ring 0 x 100 | 0.019 / 1.016 / 0.998 | 0.009 / 1.000 / 1.000 |
  | CK2 incident bins only | 0 / 1 / 1 (already at rank 4) | 0 / 1 / 1 |

  These operators have negative entries (down to -1.9 at rank 16 for the first two rows) and are L2 optima, so they are an indication of
  what rank permits, not admissible operators and not L1 bounds. They show that "rank 16 is inadequate on CK2" is a statement about how a
  rank-16 budget is allocated over all 64 incident bins under training-distribution weighting, not about representing the CK2 response.
  WL95A acknowledges that no grazing-weighted fit was tried (`95A:144`, `:154`).
- **At delivery level the M64 alternative is not clearly better for s8601.** Box-A delivery, ratio / abs rel L1 against the path-traced
  reference: deployed relation s8601 0.495 / 0.588, s8602 1.534 / 0.901, M64(s16) 1.208 / 0.524, physical q table 1.077 / 0.534
  (reproduced, section 3.8); pooled over 64 relation draws (stored `an.json`, not re-run): s8601 0.517 / 0.569, s8602 1.63 / 0.965,
  M64 1.544 / 0.546, exact per-ray X1 1.022 / 0.111. In map rel L1 the s8601 sender (0.57-0.59) is about as wrong as the exact-bin sender
  (0.52-0.55); they differ in sign of the flux bias. Only s8602 is clearly worse than M64. What removes the error is exact per-ray
  evaluation, which is the subject of P0-2 / P0-3, not of P0-1.

### 3.8 Box-A delivery of the learned sender (`sender.json: boxa_reference, delivery_boxA_deployed`)

| quantity | original | reproduced |
|---|---|---|
| deployed ratio, s8601 / s8602, vs stored e2_p | 0.495 / 1.534 | 0.4954 / 1.5342 |
| deployed abs rel L1, s8601 / s8602, vs stored e2_p | 0.588 / 0.901 | 0.5879 / 0.9007 |
| own delivery vs stored `deliv.npz` (max rel difference) | - | 4e-16 (A0 both seeds, A4) |
| the same against my own independent box-A reference | - | ratio 0.488 / 1.511; abs rel L1 0.596 / 0.879 |
| M64(s16) deployed vs stored e2_p; vs own reference | 1.21 / 0.52 | 1.208 / 0.524; 1.190 / 0.507 |
| stored e2_p reference vs own reference | half noise 0.059 | ratio 0.985, abs rel L1 0.066, corr 0.995 |

Own reference: direct area sampling of S1 from box-A patch points, exact light, shadow test at the panel, exact ER evaluation, visibility
between the offset end points; split-half noise 0.0037. It agrees with an independent hemisphere-sampling estimator (ratio 1.006, abs rel
L1 0.023 at a hemisphere noise of 0.030) and, in a constant-radiance test, with a numpy form factor (1.2927 vs 1.2944 / 1.2935 on patch 0).
The stored path-traced e2_p family on box A is thus independently confirmed within its own noise.

### 3.9 The learned sender on the training / validation distribution (`train.json`)

This is not in WL95A B1; it tests whether the CK2 numbers are a distribution-edge artefact. Reference here is M64 applied to the stored
source encoding (not a physical q), W_T weights, S1 patches, lightings with S1 lit.

| | s8601 | s8602 |
|---|---|---|
| train subsample (898 lightings): pooled rel L1 / energy vs M64 | 0.777 / 0.913 | 0.765 / 0.899 |
| validation (214 lightings): pooled rel L1 / energy vs M64 | 0.811 / 0.854 | 0.796 / 0.900 |
| per-lighting rel L1, 10 / 50 / 90 % (train) | 0.64 / 0.80 / 1.03 | 0.65 / 0.78 / 1.01 |
| lightings with ring-0 irradiance share < 0.05 (771): median rel L1 | 0.777 | 0.763 |
| lightings with ring-0 share >= 0.95 (50 = 5.6 %): median rel L1 / energy | 1.130 / 0.629 | 1.209 / 1.216 |

Ring 0 carries 4.3 % of the training direct irradiance on S1 (the worklog's figure; reproduced), 6.5 % in validation. CK2 belongs to a
lighting class that exists in training (5.6 % of lightings) and is its worst class, but the learned S1 sender is already 0.77-0.81 away
from M64 on ordinary lightings with about the right energy (0.90-0.97): the lobe shape is not represented in general. This agrees with
WL94D's "field rel L1 0.82" on training fields (`94D:25`, W94 weights).

## 4. Findings

| id | category | material | claim affected | original vs independent | evidence | decision-level impact | proposed verdict | confidence |
|---|---|---|---|---|---|---|---|---|
| E1-F1 | confirmation | yes | P0-1 (error) | sender vs M64(s16) 1.041 / 0.433 / 0.038 and 1.118 / 0.891 / 0.129 -> 1.0409 / 0.4330 / 0.0378 and 1.1182 / 0.8912 / 0.1286; vs physical q 1.083 / 0.420 / 0.032 and 1.190 / 0.864 / 0.108 -> 1.0830 / 0.4204 / 0.0316 and 1.1912 / 0.8652 / 0.1079 (own q) | `95A:57-58`; `sender.json: sender_metrics`; stored sender reproduced with 0.0 difference | the field-level measurement behind P0-1 stands | VERIFIED | high |
| E1-F2 | confirmation | yes | P0-1 (reference validity) | stored q256 vs own MC 0.026 rel L1 (own noise 0.022), energy 1.001; M64 stored vs own 1.2 % flux (4.5 % in the grazing ring, definitional); stored e2_p on box A vs own reference 0.985 / 0.066 (stored noise 0.059); ER cosine convention analytic to 1.8e-7 | `sender.json: q, m64, boxa_reference`; `conv.json` | the references used for P0-1 are not the source of the error | VERIFIED | high |
| E1-F3 | confirmation | yes | P0-1 (error, robustness) | 32 source-point redraws: rel L1 1.05-1.12 / 1.12-1.26; 6 state redraws: +- 0.03; relation-read measure 1.06 / 1.16 (box A 1.005 / 1.40) | `sender.json: source_point_redraws, state_redraws`; `read.json` | not an artefact of the 16 fixed points, of the state draw, or of entries the relation never reads | VERIFIED | high |
| E1-F4 | confirmation (scope extended) | yes | P0-1 (passivity) | max directional albedo 2.26 / 3.08 -> 2.2637 / 3.0774; M64 0.825 -> 0.8247 (own 0.826); in addition 10.4 % / 15.4 % of entries > 1, 42 / 42 patches, all materials incl. Lambert, stable over states and 232 training geometries, violating bins excited in training | `95A:57-58`; `a95.py:149-152`; `sender.json: passivity`; `train.json` | "may violate" understates: the trained local operator is non-passive scene-wide and robustly | VERIFIED | high |
| E1-F5 | invalid_inference (guard) | yes | P0-1 (passivity), any causal use | no original number claims causation; independent: CK2 total out / in 0.283 / 0.582 vs physical 0.672; 6.8 % / 23 % of learned exitance from entries > 1; passivity not imposed (`ecro86.json:12`); `apply()` non-recursive (`model86.py:109-124`) | section 3.6 | non-passivity must not be cited as the cause of the CK2 delivery error or as an instability of the deployed model; it bears on physical interpretability and on recursive reuse only | CONDITIONALLY_VALID (UNTESTED_INFERENCE if used causally) | high (measurement), medium (causal reading) |
| E1-F6 | invalid_inference | yes | P0-1 ("avoidable"), P0-5 | worklog: "Avoidable within rank 16 ... Not shown" (`95A:143`); bounds A1 / A2 0.957 / 0.895 rel L1 on the same input; independent: CK2 excites 4 incident bins holding 2.0 % of training irradiance, M64's CK2 response is rank <= 4, ring-0-weighted rank-16 SVD 0.24 (x10) / 0.02 (x100) vs 1.33 (training weights) | `rank.json`; `sender.json: source` | "avoidable" holds only relative to an explicit bin operator; neither "better training of the same model would fix it" nor "rank 16 cannot represent a grazing sharp response" is established | CONDITIONALLY_VALID | medium-high |
| E1-F7 | invalid_inference | yes | P0-1 ("avoidable", delivery level) | box A abs rel L1 deployed: s8601 0.588, s8602 0.901, M64 0.524, q table 0.534 (reproduced); pooled (stored only): 0.569 / 0.965 / 0.546, X1 0.111 | `sender.json: delivery_boxA_deployed`; `results/ecro_attr95a/an.json` | at application level replacing the learned sender by M64 does not reduce the map error for s8601; the avoidable part at delivery level is seed-dependent and smaller than the binned-carrier error itself | CONDITIONALLY_VALID | medium |
| E1-F8 | dataset_limitation | yes | P0-1, P0-5 | one scene, one light 5 cm above the panel, cos incidence 0.026-0.12, 100 % of S1 irradiance in ring 0 (4.3 % of training irradiance; 5.6 % of training lightings are of this class); in-distribution sender error 0.77-0.81 rel L1 at energy 0.85-0.91 | `conv.json: workload`; `train.json` | the CK2 magnitudes (rel L1 > 1, peak 0.03-0.13) describe the worst lighting class; the general statement supported is "lobe shape not represented (about 0.8 rel L1) with roughly correct energy" | CONDITIONALLY_VALID | high |
| E1-F9 | evaluation_defect | yes for earlier batches, no for WL95A B1 | WL94D-era sender metrics; P0-1 unaffected | `an94.W` weights ring-0 sectors x0.5 / x1.5; with it s8601 reads 1.075 / 0.557 and s8602 1.221 / 1.123 instead of 1.041 / 0.433 and 1.118 / 0.891 (the energy bias of s8602 changes sign) | `conv.json: bins`; `sender.json: sender_metrics.*.W94_as_coded`; `a95.py:27-28` | WL95A B1 uses the corrected weights, so P0-1 numbers are clean; any earlier energy statement about the CK2 sender made with W94 is unreliable in sign for s8602; the "rel L1 about 1" conclusion is weight-independent | IMPLEMENTATION_DEFECT (earlier metric), corrected in WL95A | high |
| E1-F10 | statistical_weakness | yes | P0-1, P0-5 | two seeds of one training recipe (OWN); energy 0.42 vs 0.87, box-A flux 0.49 vs 1.51; learned side deterministic, reference noise 2 % | section 3.4 | no population statement about "the learned sender" beyond these two checkpoints; the seed-signed spread is itself a result | CONDITIONALLY_VALID | high |
| E1-F11 | evaluation_defect (minor) | no | none | `Bq` is an internal, never-supervised intermediate (`model86.py:109-124`, `own92a.py:167-176`); checked: read-measure error equals all-bin error | `read.json` | the caveat exists but does not change the result | NONE | high |
| E1-F12 | confirmation | no | bound-operator rows of B1 | A1 / A2 / A3 rows and albedo maxima reproduced to 4 digits from the stored operators | section 3.4, 3.6 | the comparison rows used for "avoidable" are computed correctly (their provenance as fits was not re-derived) | VERIFIED (computation only) | high |

## 5. P0 assessment within this lens

- **P0-1, error half: VERIFIED.** The learned OWN S1 sender on CK2 is at the zero-sender level at field level against both M64 and an
  independently computed physical table, with the correct weights, for both seeds, under source-point and state redraws, and in the
  relation-read measure. Restriction: two checkpoints, one scene / light / material at extreme grazing incidence.
- **P0-1, "avoidable": CONDITIONALLY_VALID.** True relative to an explicit 64 x 64 bin operator at field level. Not shown within the
  tested learned representation; not clearly true at delivery level for s8601; and the complementary reading "rank 16 cannot do it" is not
  established either.
- **P0-1, passivity half: VERIFIED at operator level** (stronger than "may"); **no demonstrated consequence** for the CK2 error or for
  stability (non-recursive application, net under-emission on CK2, never a design contract).
- **P0-2 (seen in passing, not audited here):** M64(s16) is 0.24 rel L1 from the physical table at the sender and its deployed box-A
  delivery has ratio 1.19-1.21 and abs rel L1 0.51-0.52 against two independent references; the physical q table delivered through the
  same relation gives 1.06-1.08 / 0.51-0.53. For the physical S1 field on CK2 the bin-measure exitance (bin-mean radiance x ring-mean
  cosine) is 1.30x the exact-cosine exitance (ratio 0.769), a sign-consistent mechanism for over-delivery by a bin-constant sender.
  Observation only; pooled-draw statistics were not re-run.
- **P0-5 (within this lens):** the evidence supports rejecting the tested mechanism "OWN-trained, material-conditioned, tied rank-16
  non-negative factor pair as the S1 sender" on this workload. It does not establish that a learned sender in general, or rank 16 in
  general, cannot represent a grazing sharp response (F6), and it rests on two checkpoints (F10).

## 6. What could not be verified, and why

- Pooled 64-draw delivery statistics (`an.json` pool rows), ROI image errors (B3) and the rule verdicts of section E: not re-run
  (render / many-draw cost; other tracks' subject). Quoted from stored `an.json` where used, marked as such.
- "Operator flux-L1 vs M64" column of B1 (1.018 / 0.987): not recomputed (not needed for P0-1).
- Provenance of A1 / A2 / A3 as fits (WL95 ladder code, `S1_ops.npz`): only their application and albedo were recomputed.
- Physical q on training geometries: the in-distribution comparison uses M64 as reference; the training-set statistics use every 4th
  kept training file (232 of 928) and all 55 validation files.
- Only the OWN checkpoints s8601 / s8602 were examined (not `sum_*`, WL86 originals, or WL93-94 candidates).
- The causal statement "s8602's over-delivery is misallocation, not energy creation" is an inference from energy totals and ring shares;
  no counterfactual (e.g. clipping violating entries) was run, because that would be a modification of the model.
- Earlier evidence (WL92B, WL93D) was not re-traced; WL94D only through the in-distribution rel L1 and the W94 effect.
- The 4.5 % ring-0 difference between my M64 and the stored M64 was not traced to a specific line of `quad87.py`.

## 7. Own-code defect disclosure

The first version of my own box-A area estimator shot its shadow ray from an offset origin along the un-offset direction and lost
visibility near grazing (constant-radiance test: hemisphere / area ratio 1.140). It was fixed (segment between the two offset end points;
ratio 0.9998) before any number in this report was produced; the pre-fix outputs are kept as `sender_v0_buggy_area.json` and
`debug_v0_buggy_area.json` and are not used. The defect affected only my own box-A reference, never the sender or passivity numbers.

## 8. Reproducibility appendix

Scripts (sha256 prefix): `experiments/audit97/E1/e1_audit.py` 170e11c3d905bec3 (identical on the server), `e1_read.py` bba6167dd7dd93de,
`e1_rank.py` 7cd0dc760dc98ab2, `run_e1.sh` a004a5a02b6c9573.

Outputs (`results/audit97/E1/`): `conv.json` 6445f50e4e175a72, `sender.json` cd2731a1eccef2e0, `sender_e1.npz` 98fbb9b48a8703ec,
`debug.json` 83afce020fbd3049, `train.json` a31894f8c6db8ab3, `read.json` 538a13062563e1aa, `rank.json` fafc1f57abf8fe44;
server copies and logs in `~/rest_x97/E1/` (`log_*.txt`, `results/`). `_interrupted_run_log.md` is the action log of the first attempt.

```
# server (LabServer63), from ~/rest_x97/E1 with e1_audit.py, e1_read.py, run_e1.sh copied there (LF endings)
bash run_e1.sh conv      # -> results/conv.json
bash run_e1.sh sender    # -> results/sender.json, results/sender_e1.npz
bash run_e1.sh debug     # -> results/debug.json
bash run_e1.sh train     # -> results/train.json
. ~/rest_x95a/env.sh; export PYTHONDONTWRITEBYTECODE=1 E1_CODE=$HOME/rest_x95a/code/experiments E1_ROOT=$HOME/rest_x97/E1 \
  E1_ORIG=$HOME/rest_x95a/results E1_R92A=$HOME/rest_x92a E1_R94=$HOME/rest_x94 E1_R86=$HOME/rest_ecro86
$PY e1_read.py           # -> results/read.json
# local (numpy only), after copying results/*.json and sender_e1.npz to results/audit97/E1/
python experiments/audit97/E1/e1_rank.py /c/Projects/ReST-GI    # -> results/audit97/E1/rank.json
```

Read-only inputs: `~/rest_x95a/code/experiments` (c152120), `~/rest_x95a/results/{senders.npz, ops95a.npz, deliv.npz}`,
`~/rest_x92a/results/models/own_s860{1,2}.pt`, `~/rest_x92a/results/exclusion.json`, `~/rest_x94/results/fields/CKF_none.npz`,
`~/rest_ecro86/data/{train,val}/g*.npz`. Nothing outside `~/rest_x97/E1/` was written on the server.
