# WL95 read-only evidence digest: WL89–WL92B (GM1 explicit floor → ECRO sharp attribution)

Labels: [FACT] verified in code or doc · [MEAS] measured value as reported (cited) · [INF] my inference · [HYP] untested.
Paths are relative to `c:\Projects\ReST-GI`. Abbreviations: WL90 = `docs/worklogs/90_WORKLOG_GM1_INTERMEDIATE_PROJECTION_GAP_ATTRIBUTION.md`; WL91, WL92, WL92A, WL92B likewise. No files were edited and nothing was run beyond reading files and JSON.

---

## 0. Shared metric definitions (read before comparing any number)

**E_H, the rendered-image metric** [FACT]
- Formula: E_H = Σ_{px∈mask, rgb} |hl_X − h_REF| / Σ_{px∈mask, rgb} tot_REF.
  - h = higher-order pixel radiance of the reference.
  - tot = e1 + e2 + h.
  - Both are averaged over the render halves.
- Code: `experiments/ecro_attr92/analyze92.py:54-66`; identical in `experiments/gm1_relation88/analyze88.py:49-63` and `experiments/gm1_gap90/analyze90.py:28-40`. The WL90 docstring at `analyze90.py:3` restates it.
- Related quantities from the same code:
  - signed = same sum without the absolute value.
  - noise_bound = ½|x_a − x_b| + ½|h_a − h_b|, over the same denominator.
  - capture = E_H / H_share.
- Normaliser: E_H is divided by the region's **total** radiance, not by h. A value therefore depends on the region's H_share [INF].
- Families and regions:
  - GM1 (WL89–91): workload W1 = V3, held-out edit W2 = X2, regions R_mirror / C_ss / stationary / D, reference WL89 REF-ER.
  - ECRO (WL86–92B): conditions E1 / E2 / E4 / E5, region C, reference WL86 REF-ER.
- Same formula across the two families, but different scenes, masks and references. GM1 R_mirror numbers and ECRO region-C numbers are not cross-comparable [INF].
- Shared gate F2: E_H ≤ 0.06 (`ecro_attr92.json` thresholds `EH_region_max`).

**Bins** [FACT]
- 64 equal-solid-angle bins = 4 rings uniform in cos θ × 16 azimuth sectors; dΩ = 2π/64 ≈ 0.098 sr (`experiments/ecro86/protocol/ecro86.json:47`; `scene86.py:15-20`).
- 8 state rays per bin (K = 512).

**Reader-weighted reading** [FACT]
- reads(W, X) = Σ W_R · X (`analyze92.py:87-88`). W_R is the importance map with Σ_pixels hl = Σ W_R · B (`diag92.py:89`).
- Family shares are readings divided by h_read (`analyze92.py:119-130`).
- "Panel family delivered" = reading of the single-flight term restricted to state rays that hit material-1 (panel) patches (`diag92.py:240-252`), divided by the reading of the path-traced e2_p.
- WL92B level / directional split (`experiments/ecro_cf92b/cf92b.py:161-168`):
  - level = Σ (per-receiver-patch flux ratio − 1) · W · T / h;
  - directional = signed − level.

**Field-level relative L1** [FACT]
- Σ|X − T| / ΣT over every patch × bin × rgb, for example single vs L₂ (`analyze92.py:140-143`) and Bq vs q (`analyze92.py:133-139`; `cf92b.py:225-231`).
- This is a different norm from E_H and from the local-operator metrics.

---

## (a) WL90: the explicit-vs-oracle gap is the angular bin projection of L₁ in the first gather

**What "projection" means** [FACT] (WL90:18-25; `gap90.py:1-23`)
- Every WL87–89 relation computes L₁ = q, then L_{k+1} = G(P L_k), then B = L₂ + … + L₅.
- P is implicit in the gather's read L[hit, dbin]: one value per patch in space and one value per 0.098 sr bin in direction. It is applied before all four gathers.
- The oracle table is the receiver patch × bin average of path-traced 2–5-event radiance, with no P at any sender.
- q is an accurate patch × bin projection of the exact 1-event field: relative L1 0.013, flux 1.0007 (WL90:22) [MEAS].
- The loss is therefore not an error in q. It is that the gather reads a bin-constant sender radiance rather than radiance in the ray's exact direction [INF].

**Diagnostic operators** [FACT] (WL90:42-59; `gap90.py:10-23`)
- Run on a pooled R0 relation: 64 fresh realisations, WL87 weights.
- X(a, s)[k] = G(P_{a,s} Lx_k), where Lx_k is the path-traced exact k-event radiance read per state ray (4 samples per read).
  - a = sub-bins per bin; s = sub-patches per patch; 0 = exact; a = g × g, so a = 4 is 2 × 2 (256 directions) and a = 16 is 4 × 4 (1,024 directions).
- EE = X(0,0). J = X(1,1) − EE. A = X(1,0) − EE (angular only). S = X(0,1) − EE (spatial only).
- M_m = first m gathers exact, the rest by the pooled table operator.

**Measured, W1 T1 R_mirror E_H (4,096 spp)** [MEAS]
- M₀ (converged table) 0.0363 vs ORACLE 0.0137: G₀ = 0.0226, CRN noise 0.0001 (WL90:156-159).
- EE − ORACLE = −0.0002, i.e. 100.9 % of the gap removed (97 % W1 T0, 102 % W2 T1). ISOLATED in 3 configurations × 4 regions (WL90:162-165).
- Factors, W1 T1 / W1 T0 / W2 T1 (WL90:172-177):

  | Term | W1 T1 | W1 T0 | W2 T1 |
  |---|---|---|---|
  | J (table projection once) | +0.0207 | +0.0309 | +0.0170 |
  | A (angular only) | +0.0226 | +0.0339 | +0.0177 |
  | S (spatial only) | −0.0009 | −0.0011 | −0.0027 |
  | P1only − EE (order 1 only) | +0.0198 | +0.0311 | +0.0170 |
  | P2plus − EE (orders ≥ 2 only) | +0.0009 | +0.0007 | −0.0004 |
  | Compounding M₀ − X(1,1) | +0.0021 | −0.0003 | −0.0001 |

- Rule result: ANGULAR.
- Order: m* = 1. M₁ = 0.0146 / 0.0159 / 0.0149. First-order share 0.96–1.01; compounding ≤ 9 % of the gap (WL90:183-184).
- Field level: the first divergence is panel order 2 (WL90:185-193). This is a field relative L1, a different norm:

  | Panel order 2 | Field relative L1 |
  |---|---|
  | M₀ | 0.137 |
  | EE | 0.053 |
  | angular only | 0.148 |
  | spatial only | 0.089 |
  | truth noise | 0.043 |

- Fixed grid, residual vs EE (WL90:199-204; single-read lower bound, compounding excluded):
  - angular 4 sub-bins: +0.0038 / +0.0063 / +0.0016 (9–20 % of J);
  - angular 16: +0.0007 / +0.0012 / −0.0031;
  - spatial refinement: no effect.

**What "first-gather-exact" (X1) changes** [FACT] (`gap90.py:189-250`; `experiments/gm1_c3urx1_91/cand91.py:36-110`)
- Only L₂ changes; later orders keep the table operator.
- Each landed relation ray carries the exact once-scattered radiance from its own origin y in its own direction v. This is one next-event estimate at the hit of a probe ray launched from y + 2 mm·v towards −v; it is deterministic for point lights.
- This value replaces L[src, bin]. Receiver deposition and response are unchanged (`cand91.py:82-95`).
- The relation stays source-independent. The q table is no longer needed.

**Measured effect and cost** [MEAS]
- C3URX1 vs C3UR on 16 draws (WL90:214-220):
  - W1 T1 R_mirror 0.024 ± 0.005 vs 0.040 ± 0.005; paired −0.016 [−0.020, −0.012], 16 / 16 draws better.
  - W1 T0: −0.021 [−0.028, −0.014].
- R0X1 − R0 (receiver-side relation): +0.003 [−0.006, +0.012], no E_H gain.
  - [INF] (WL90:253-254, and labelled as inference in the worklog): reading sharp exact radiance through 512 receiver directions raises variance.
- Cost, RTX 3080 Ti, warm (WL90:227-240):
  - per-ray exact L₁ 10.4 ms vs q table 26.1 ms;
  - source-only update 20.6 vs 36.7 ms (0.56×);
  - per-edit 68.2 vs 83.6 ms (0.82×);
  - +12.6 MB probe buffer, 6.3 MB scratch.
- The cost of a 4×–16× table refinement was projected, not measured (WL90:242-244).
- Scope: a single family (S1 slab, point light, α ≈ 0.05) (WL90:263).

---

## (b) WL91: C3URX1 bias removal vs variance

- **Bias removal reproduces** [MEAS] (WL91:160-179).
  - Pooled C3URX1 is within −0.0025 to +0.0035 of ORACLE in all six conditions.
  - The first-gather projection term (C3UR pooled − C3URX1 pooled) is 0.020–0.034 in W1, W2 and LSPOT89, and −0.0009 in LMIX89 (no gap there).
- **Variance** [MEAS] (WL91:173-179, 384).
  - Sampling term (draw mean − pooled): C3URX1 0.005–0.014 vs C3UR 0.0005–0.005. The C3URX1 term is larger in all six conditions.
  - The W2 T1 deployed draw is 0.053 (signed +0.044), worse than C3UR (0.038) and above the 16-draw maximum (0.039).
- **Paired R_mirror differences, C3URX1 − C3UR** [MEAS] (WL91:151-156):

  | Condition | Difference [95 %] |
  |---|---|
  | W2 T1 | −0.0100 [−0.0153, −0.0046] |
  | LMIX89 | +0.0010 [−0.0017, +0.0037] |
  | LSPOT89 (834 px) | −0.0266 [−0.0320, −0.0211] |
  | LSPOT89 C_ss | +0.0025 [+0.0005, +0.0045] (harm) |

  - LMIX89 also shows a positive signed residual: +0.0096 for the draws and for the pooled table (WL91:186).
- **Verdicts** [MEAS] (WL91:310-317):
  - HELD-OUT MECHANISM VALIDITY: NOT SUPPORTED (survives in 1 of 3 conditions).
  - QUALITY-MATCHED UTILITY: NOT SUPPORTED. Rule (a) fails: W2 T1 deployed draw 0.74 s vs 0.24 s at the R_mirror 0.06 tier, a 3.1× slowdown.
  - LIFECYCLE COST: SUPPORTED, 0.57–0.90× C3UR (edit-to-frame 124.7 vs 138.2 ms; source 75.8 vs 92.2 ms).
  - Adoption evidence: INSUFFICIENT.
- **Cost detail** [MEAS] (WL91:256, 304):
  - Per light: 526,467 probe intersections + 526,467 shadow rays vs 20,320 for the q table.
  - The exact evaluation stays at 11–14 ms from 0.26 M to 2.1 M landed rays.
- [INF] (WL91:393, stated there as interpretation) The q table's patch × bin averaging turns per-ray point-sampling variance into a fixed bias. X1 exposes that variance.
- Caveat [INF]: E_H is an absolute-value metric, so the "decomposition" columns are differences of E_H values, not additive error components.

---

## (c) WL92: representability of the local operator and the gates

**Local-operator norm, exactly** [FACT]
- M64 = `quad87.Quad.enc_matrix()`: the frozen-ER BRDF (no cosine), averaged over 8 × 8 sub-cells of the incident bin and over the outgoing bin, per material and colour.
  - Code: `experiments/gm1_explicit87/quad87.py:223-242`, with `fbar` at `:85`.
  - The dense estimator is seeded with realisation 86001 (`diag92.py:133-139`).
- z_b = (ring + 0.5)/4 ∈ {0.125, 0.375, 0.625, 0.875} (`diag92.py:114-119`).
- **Flux-transfer matrix** X[bi, bo] = M64[bi, bo] · z_bi · z_bo (`diag92.py:160`, `:125`).
  - Caution [FACT from code]: rows of X sum to z_bi · albedo(bi) / dΩ, because the code computes albedo = X.sum(1) · dΩ / z_bi (`diag92.py:172`).
  - The protocol and comment wording "rows sum to the directional albedo" therefore omits a z_bi / dΩ factor. Incident bins are weighted by their ring cosine (grazing ring 7× lower than the near-normal ring) [INF].
- **`mat_metrics(Mh, Mt, zb)`** (`diag92.py:122-130`). Both matrices are converted to X first.
  - `fro` = ‖X̂ − X‖_F / ‖X‖_F, the relative Frobenius norm on the 64 × 64 matrix of one colour.
  - `l1` = Σ|X̂ − X| / ΣX.
  - `peak` ("mirror-bin retention") = Σ_bi X̂[bi, argmax_bo X[bi, ·]] / Σ_bi X[bi, argmax]: an **aggregate ratio** at the true per-row peak bin. It can exceed 1 (Lambert 1.0–1.1).
  - `mass` = ΣX̂ / ΣX.
  - `peak_share_of_truth` = Σ peak / ΣX (panel: 0.42–0.46).
- **Eckart–Young bound** = sqrt(Σ_{i>16} σ_i² / Σ σ_i²) from the SVD of X, per colour (`diag92.py:161-168`). It equals `svd16.fro`.
- **Rank for energy** = smallest k with cumulative σ²/Σσ² ≥ q, for q = 0.9 / 0.99 / 0.999 (`:169`).
- **NMF16**: multiplicative-update Frobenius NMF **on X** (4,000 iterations, 16 restarts, seed 92100) (`diag92.py:97-111`, `:165-166`).
  - Factors are mapped back as c = W / z_bi, a = Hᵀ / z_bo (non-negative rank-16 factors of M64).
  - The NMF therefore minimises the flux-Frobenius norm, not peak retention or E_H [FACT]. A rank-16 factorisation optimised for another objective was not tested [HYP].
- **Unweighted Frobenius** of plain M64 is also stored (`:173`, "not used"). Panel EY16 there is 0.19–0.37, so it is **not comparable** to the flux-norm numbers (`local.json`) [MEAS].
- **Learned rows** (`diag92.py:177-195`):
  - M_n = Σ_r a[n, bo, c, r] c[n, bi, c, r] from the frozen generators **on the E2 T1 geometry**.
  - `mat_metrics` is computed per (patch, colour) against M64 of the patch material, then averaged.
  - Sphere patches are compared with the flat-frame M64 [FACT] (WL92B:30).
  - The learned `participation_rank_first_patch` (`:193`) uses the SVD of the **unweighted** Ml (one patch, green), while the material participation rank (`:170`) uses the flux-weighted X. They are not matched norms [FACT]. Learned panel ≈ 5.5 vs material 37–44 (`local.json`).

**Values from `results/ecro_attr92/results/local.json` (flux norm), R / G / B** [MEAS]

| Material | EY16 fro | Rank for 0.9 / 0.99 / 0.999 | SVD16 l1 / peak | NMF16 fro / l1 / peak / mass |
|---|---|---|---|---|
| floor B | 0.030 / 0.028 / 0.040 | 3 / 8 / 16–20 | 0.03–0.04 / 0.96–0.97 | 0.029–0.042 / 0.03–0.04 / 0.959–0.968 / ≈1.00 |
| panel S1 | 0.520 / 0.458 / 0.461 | 25–33 / 46–59 / 61–64 | 0.87–0.93 / 0.49–0.56 | 0.544 / 0.485 / 0.488; l1 0.59–0.65; peak 0.472 / 0.535 / 0.528; mass 1.02–1.04 |
| sphere S2 | 0.204 / 0.335 / 0.234 | 10–18 / 31–42 / 53–58 | 0.27–0.47 / 0.62–0.73 | 0.22–0.36 / 0.22–0.34 / 0.60–0.71 / 1.01–1.04 |

- Learned `a cᵀ`, frozen s8601 / s8602, E2 T1 geometry [MEAS]:

  | Material | fro | peak | mass | l1 |
  |---|---|---|---|---|
  | panel | 0.877 / 0.885 | 0.135 / 0.112 | 0.757 / 0.682 | 1.05–1.07 |
  | floor | 0.33–0.34 | 0.80–0.84 | 1.08 | — |
  | sphere | 0.63–0.64 | 0.32–0.34 | 1.15–1.17 | — |
  | Lambert | 0.13–0.32 | 1.0–1.12 | — | — |

  WL92:135-140 matches.

**Ladder: E2 T1 / E2 T0 region C, with path-traced L₃₊ added** [MEAS] (WL92:144-156; `gate.json`)

| Variant | E2 T1 | E2 T0 |
|---|---|---|
| ORA2 | 0.0127 | 0.0099 |
| RS | 0.048 | 0.043 |
| EXP | 0.041 | 0.048 |
| ENCQ | 0.045 | 0.053 |
| **ENCB** (c = I, a = M64, R = 64; `diag92.py:204-210`) | **0.0449** | **0.0524** |
| **R16** (material NMF16; `:213-229`) | **0.0843** | **0.0726** |
| Pooled 16 realisations: EXP / ENCB / R16 | 0.027 / 0.027 / 0.064 | 0.022 / 0.023 / 0.056 |
| Learned single + true L₃₊ | 0.124 / 0.128 | 0.083 / 0.098 |

- ENCB vs the bin-quadrature explicit form: T_FORM 0.018–0.021 relative L1 [MEAS].

**Free-factor fits** [FACT] (`diag92.py:373-509`; protocol `fits`)
- Setup:
  - fixed geometry; c, a, Q, P all free (softplus);
  - Adam, lr 0.01, cosine schedule, 8,000 steps, 16 of 192 fit lights per step, 32 validation lights.
- Objectives:
  - den = t.mean over bins + 0.05 · t.mean;
  - SUM = mean(((B_s + B_g − t)/den)²);
  - OWN = SUM + mean(((B_s − L₂)/den)²) + mean(((B_g − L₃₊)/den)²) (`:414-417`, `:481`).
- Convergence = |mean of the last 500 steps − mean of the previous 500| ≤ 2 % (`:490-494`).

**Fit results** [MEAS] (WL92:164-172). Values are single-flight + true L₃₊ / whole table.

| Fit | E2 T1 | E2 T0 |
|---|---|---|
| OWN-f8601 | 0.029 / 0.027, converged | 0.020 / 0.024, not converged (2.2 %) |
| **OWN-k16** (gate fit) | 0.036 / 0.036, **not converged** (6.1 %) | 0.017 / 0.013, **not converged** (11.6 %) |
| OWN-r1 | 0.041 / 0.044, converged (1.9 %) | 0.024 / 0.021, converged (1.7 %) |
| SUM fits | single + L₃₊ 0.074–0.111; whole 0.016–0.043 | — |

- Component flux (single / L₂): SUM frozen-init 0.82–0.87, SUM untrained-init 1.36–1.37, OWN 0.996–1.002.
- The k16 initialisation (NMF16 + frozen global) has a validation SUM objective of 0.60 (E2 T1) / 1.18 (E2 T0), vs 0.016 / 0.037 for the frozen generator (`final92.json` identifiability) [MEAS].

**Gates** [MEAS] (`results/ecro_attr92/results/gate.json`)
- G_id holds.
- G_info holds: ENCB 0.0449 / 0.0524 ≤ 0.06.
- G_rep **fails**: best owned in-family point = R16, 0.0843 / 0.0726. OWN-k16 (0.036 / 0.017) is excluded because it did not converge (`analyze92.py:194-203`).
- Gate result STOP_INTERVENTION_G_rep. Verdicts: A INCONCLUSIVE, B SUPPORTED, C INCONCLUSIVE, completion code 1.
- WL92:197 discloses that code 1 means "G_rep not established", not a demonstrated representation limit.
- Identifiability rule (`analyze92.py:182`): objective spread ≤ 1.10 **and** allocation ratio ≥ 2.
  - Spread 1.26–1.67, allocation 2.6–28× → "IDENTIFIABLE BY THE RULE".
  - The spread comes mainly from the unconverged untrained-init fits.

**Other measured context** [MEAS]
- E2 T1 region-C truth shares: 2 events via panel 0.18 / 2 events other 0.36 / 3–5 events 0.46.
- Learned single − L₂ −0.168 / −0.175; global − L₃₊ +0.076 / +0.096; panel family delivered 14 % / 18 % (WL92:125).
- Learned Bq on panel patches vs exact q: relative L1 0.90–1.11, flux 0.16–0.72 (WL92:131).

---

## (d) WL92A / WL92B: OWN supervision and the source-side once-scattered panel response

**WL92A** (`experiments/ecro_own92a/own92a.py:172-176`) [FACT]
- OWN uses the WL92 OWN objective, weights 1 / 1 / 1; t₂ is the regenerated exactly-2-event target and t₃ = t − t₂.
- Training: 24,000 steps on 928 geometries (96 excluded for E4).

**WL92A measured** [MEAS]
- Validation field relative L1, OWN vs SUM (WL92A:136-141):
  - single vs L₂: 0.104 / 0.105 vs 0.210 / 0.196;
  - global vs L₃₊: 0.092 / 0.092 vs 0.218 / 0.200;
  - whole table unchanged, 0.077–0.078;
  - debiased RMS of own_s8602 worse: 0.157.
- Panel `a cᵀ` peak retention, E4 T1 geometry, same `mat_metrics` (`own92a.py:345-380`) (WL92A:145-151):
  - OWN 0.207 / 0.212; SUM 0.028 / 0.078; frozen 0.11 / 0.08;
  - fro 0.80–0.81; mass 0.80–0.84;
  - reference points: NMF16 0.47–0.54, frozen ER 1.0.
- Region-C E_H (WL92A:158-161):
  - E4 T1: OWN 0.013 / 0.012 vs SUM 0.025 / 0.024. E4 is not admitted (Q4 SNR 3.9), so this is descriptive.
  - E2 T1: OWN 0.077 / 0.040.
- Outcome: D_STOPPED. NO_REGRESSION fails on E1 LSPOT own_s8602 (0.036).

**WL92B counterfactuals** [FACT] (`cf92b.py:140-213`). Each replaces one stage of the single-flight term; the global term G_ is kept unless stated.
- X_M1: learned_local(f, flight(encq)). encq = exact M64 applied to the deployed 16-point encoding.
- X_Q: learned_local(f, flight(q)), with q the exact once-scattered table **on all patches**, not only the panel.
- X_M2: exact M64 on the model's own E1.
- X_M12: exact M64 on flight(encq).
- X_L2 / X_L3: replace the single-flight / global term by its path-traced target.
- Accounted fraction f = (E_orig − E_X) / max(E_orig − E_floor, 1e-4) (`analyze92b.py:51-57`). E_floor = E_H of the path-traced table.
- Geometry variation (`cf92b.py:247-288`, `rel` at `:272`) = mean over patches of ‖X_b − X_a‖_F / ‖X_a‖_F.
  - The Frobenius norm is taken over the [bi, bo, rgb] flux-weighted tensor per patch, all colours jointly.
  - This differs from the per-colour `mat_metrics.fro`.

**WL92B measured, OWN 8601 / 8602, region C** [MEAS] (WL92B:93-110, 126-131)

| Counterfactual | f, E2 T1 | f, E2 T0 |
|---|---|---|
| X_L2 | 0.86 / 0.77 | 0.89 / 0.88 |
| X_Q | 0.87 / 0.63 | 0.74 / 0.62 |
| X_M1 | 0.91 / 0.59 | 0.64 / 0.49 |
| X_M2 | −0.26 / −0.59 | −0.41 / −0.46 |
| X_M12 | 0.74 / 0.25 | 0.13 / −0.06 |
| X_L3 | −0.36 / −0.68 | 0.11 / 0.11 |
| X_SHQ | ≈ 0 | ≈ 0 |

- The OWN seed gap on E2 T1 (0.037) is closed 0.96 by X_Q.
- Reading split, E2 T1: single-term level −0.25 / −0.15 of h; directional +0.009 / +0.018.
- Learned panel Bq vs exact q, E2 T1, from `results/ecro_cf92b/results/tables/CF_E2_T1.json`:

  | Model | Relative L1 | Flux ratio |
  |---|---|---|
  | OWN | 1.14 / 1.07 | 0.665 / 1.069 |
  | SUM | 1.14 / 1.06 | 0.238 / 0.198 |
  | frozen | 1.11 / 1.01 | 0.242 / 0.341 |

  - encq vs q on panel patches: 0.36.
  - Correction: WL92B:110's parenthesis "SUM 0.20–0.24; frozen 0.24–0.34" refers to **flux** ratios.
- Panel `a cᵀ` change E2 T0 → T1: OWN 0.145 / 0.146, SUM 0.30 / 0.28, frozen 0.34 / 0.44. Seed-to-seed: OWN 0.51, SUM 1.76.
- Outcome C: H_interact SUPPORTED, H_geometry SUPPORTED, H_local INCONCLUSIVE, H_source NOT SUPPORTED. The LSPOT regression is attributed to X_L3: 0.71 / 0.82.
- E5 (admitted, Q4 SNR 13), region C:
  - OWN T1 0.033 / 0.027 vs SUM 0.028 / 0.026;
  - OWN T0 0.034 / 0.017 vs SUM 0.028 / 0.030;
  - OWN_CONSISTENT_ADVANTAGE fails; NO_REGRESSION fails (LSPOT own_s8602 0.026 vs SUM 0.016);
  - ORACLE 0.006, X_EXP 0.030.

**Interpretation of "source-side once-scattered panel response"** [INF]
- Measured facts:
  - Under OWN, the panel Bq has roughly correct flux for one seed but relative L1 > 1. Its directional placement is wrong at field level.
  - At sharp receivers this shows up as a level deficit (light leaves the panel in directions that miss region C), not as a receiver-side directional error.
  - The exact receiver-side local response alone (X_M2) worsens E_H. That is consistent with the tied `a cᵀ` partly compensating on the receiver side.
- Not established:
  - X_Q replaces q on all patches, so "panel" is an attribution from the Bq panel field errors, not a panel-only counterfactual.
  - That tied factors are the cause is not isolated; it would need training [HYP].

---

## Matched-norm comparability table (for later batches)

| Quantity family | Norm / definition | Comparable with |
|---|---|---|
| E_H region values | `analyze92.py:54-66` | Same family, region, mask and reference only. GM1 R_mirror ≠ ECRO region C. |
| Local `fro` / `peak` / `mass` / `l1` | flux X = M · z_bi · z_bo, per colour (and per patch for learned), `diag92.py:122-130` | WL92 local, WL92A local, WL92B `metrics` (all call `DG.mat_metrics`). Not: unweighted M64 numbers, the learned participation rank (unweighted), or WL92B `geometry_variation` / `seed_variation` (joint-rgb Frobenius per patch). |
| Eckart–Young rank-16 bound | flux-X Frobenius; a lower bound only in that norm | NMF16 / learned `fro` |
| Field relative L1 | Σ\|X − T\| / ΣT over patch × bin × rgb | Other field relative L1 only; not E_H, not WL90's per-order field errors on other workloads |
| Counterfactual f | rendered E_H differences over (E_orig − floor) | Within one model and case only |

Notes [FACT]:
- The learned local-operator rows are geometry-dependent: WL92 used the E2 T1 geometry, WL92A E4 T1 and E2 T1, WL92B five configurations.
- The material M64 rows are geometry-independent.

Main sources:
- WL90:154-244
- WL91:145-317
- WL92:133-197
- WL92A:132-225
- WL92B:89-168
- `experiments/ecro_attr92/diag92.py`
- `experiments/ecro_attr92/analyze92.py`
- `experiments/ecro_cf92b/cf92b.py`
- `experiments/ecro_own92a/own92a.py`
- `experiments/gm1_gap90/gap90.py`
- `experiments/gm1_c3urx1_91/cand91.py`
- `results/ecro_attr92/results/local.json`
- `results/ecro_attr92/results/gate.json`
- `results/ecro_attr92/results/final92.json`
- `results/ecro_cf92b/results/tables/CF_*.json`
