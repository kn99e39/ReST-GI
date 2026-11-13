# ECRO / GM1 historical evidence ledger (WL86–WL93C), checked against the sources

**Method.** I read every listed worklog in full in `docs/worklogs/`, plus `docs/Architecture.md` §§1–3 and §§8–9. I re-opened the JSON results to confirm the key numbers; rows marked **[J]** match JSON to the stated rounding. Every row is an **OBSERVED** measurement reported in the named worklog. Rows without [J] are taken from the worklog text and were not re-derived. WL86B and WL89 were not in scope and I did not read them. WL90 and WL91 use the WL89 workloads (W1 = V3, W2 = X2); their definitions are second-hand here. I made no file edits and ran no jobs.

**Scope that applies to every row unless stated.** One semi-enclosed room family. Rigid box edits. Three frozen WL84 ER materials: floor B, sharp panel S1 (α ≈ 0.05), sphere S2. Point lights, plus cone lights (LSPOT) and two coloured lights (LMIX) as held-out source tests. Resolution 512². A patch × 64-bin table with 1,270 patches. 512 state rays per patch. A 16-point source encoding. Two training seeds (8601 / 8602). E_H means "additional higher-order error / total radiance". Region C means sharp readers (panel / sphere receivers or the mirror footprint).

**Workload admission status:**
- E1 is a development configuration (inspected).
- E2 is "held-out", but its aggregate dev numbers were seen once at 128 spp before the WL86 freeze (disclosed).
- E3 (WL87) was untouched and admitted, but its region C is only the floor footprint, with no panel mirror-cone receivers.
- E4 (WL92A) was NOT admitted: Q4 SNR 3.9 against 4.0.
- E5 / k995 (WL92B) was independently admitted (Q4 SNR 13). It is inspected from WL93 onward.
- E6 (WL93) was never generated.

---

## 1. Owner decision of 2026-11-13, verbatim (`docs/Architecture.md` §1)

> **Updated global implementation decision (human, 2026-11-13):** Resume the **ECRO learned-global-transport lineage** as the central implementation base for GM1. This is a development-baseline choice, **not** acceptance of the WL86 frozen ECRO v0 result: its sharp/held-out fidelity failures, learned-flight under-transfer and unconstrained component-compensation concerns remain unresolved. The specific improved global representation, ownership partition, supervision and dynamic-update rule remain open research decisions. Preserve frozen learned ER for local scattering, and keep EXP/C3UR as learning-free explicit transport comparators/diagnostics rather than the global method to be published. Later X1-style first-gather findings constrain information preservation but do not justify converting the project to explicit GI. Physical K/C ownership and the learned-dynamic-reuse scientific objective are unchanged. No successor experiment is authorised solely by this decision.

Related text in the same file:
- **§1, 2026-11-10 decision.** Learned ER is the primary local development baseline, "not adoption as final FNTO K, global transport architecture, or production integration".
- **§3.** At fixed geometry, K must be linear, nonnegative and passive. Composition must stay stable. The text adds: "These are requirements, not a claim that the existing learned kernels satisfy them."
- **§3, ER foundation.** Distinguish (a) the physical reference, (b) frozen ER with full path tracing, and (c) the global mechanism.
- **§9, GM1.** "One mechanism must be explicitly chosen before its experiment; none is selected by this baseline decision."
- **§8 table.** It has no ECRO row.

## 2. Frozen official verdict labels

| WL | Official label (frozen mapping) | Notes |
|---|---|---|
| 86 | `STOPPED_GENERALIZATION_NOT_SUPPORTED` [J `final.json` STATUS]. Item verdicts: REPRESENTABILITY SUPPORTED; GENERALIZATION NOT SUPPORTED; PHYSICAL FIDELITY NOT SUPPORTED; DYNAMIC EDIT RESPONSE CONDITIONALLY SUPPORTED; ECONOMICS NOT TESTED; GM1 VIABILITY NOT SUPPORTED under frozen gates | Amendment 1 (V3 test tolerance) is post hoc and labelled |
| 86A | Diagnostic. LEARNED MAPPING ADEQUACY NOT SUPPORTED; SOURCE INPUT SUFFICIENCY and GEOMETRY STATE SUFFICIENCY CONDITIONALLY SUPPORTED; SUPERVISION/OPTIMIZATION INDETERMINATE; INTERVENTION NOT TESTED; ECRO v0 viability NOT SUPPORTED (unchanged). Completion outcome 4 with an element of 3 | Amendments 1 and 2 are post hoc. The Stage C validation rule FAILED as written |
| 87 | Completion outcome 1 (EXPLICIT SUFFICIENCY SUPPORTED, within this workload) with an element of 4. LEARNED_OPERATOR_ADDITIONAL_UTILITY SUPPORTED on one basis only (source-only update cost); SHARP TRANSPORT REPRESENTABILITY INDETERMINATE; FROZEN STATE ADEQUACY CONDITIONALLY SUPPORTED [J] | No amendment. TF16 wall time invalid |
| 88 | `STOPPED_WORKLOAD_UNQUALIFIED`. Every hypothesis item NOT TESTED; PHYSICAL CORRECTNESS SUPPORTED [J] | E2 Task A is descriptive only. The user declined to amend Q5 |
| 90 | PRIMARY HYPOTHESIS SUPPORTED; INTERMEDIATE PROJECTION ISOLATED; DOMINANT FACTOR ANGULAR; FIRST DIVERGENCE at the first gather; X1 MATERIAL and ECONOMICALLY PLAUSIBLE [J] | X1 is a diagnostic probe, not a candidate |
| 91 | BASELINE ADOPTION EVIDENCE INSUFFICIENT; HELD-OUT MECHANISM VALIDITY NOT SUPPORTED; QUALITY-MATCHED UTILITY NOT SUPPORTED; LIFECYCLE COST SUPPORTED; SOURCE/EDIT VALIDITY SUPPORTED [J] | C3URX1 not adopted; C3UR remains the explicit baseline |
| 92 | `STOP_INTERVENTION_G_rep`, completion code 1. A Representation INCONCLUSIVE; B Information SUPPORTED; C Learning INCONCLUSIVE; D component targets valid, SUM non-identifiability INCONCLUSIVE; E/F NOT TESTED | **Disclosed wording defect:** code 1 must be read as "G_rep not established", not as a demonstrated representation limit |
| 92A | `D_STOPPED` [J]. E4 not admitted (Q4 SNR 3.9). Counterfactual (not a verdict): it would have been B because NO_REGRESSION fails | OWNERSHIP_IMPROVED and PANEL_LOCAL_IMPROVED PASS; HELDOUT_SHARP not passed |
| 92B | Outcome **C** (official) [J `outcome.label`]. WL93 found that under the *written* H_interact rule the label is INCONCLUSIVE and the outcome is D by fall-through [J `recon92b.json`]; WL92B files are unchanged | JSON quirk: `final92b.json outcome.supported` lists only H_geometry, while `hypotheses.H_interact.label` = SUPPORTED |
| 93 | `STOPPED-INCONCLUSIVE` (STOP_TARGET_NOT_IDENTIFIABLE) [J `ident.json` PASS false] | Amendment 1 (stop-path crash fix) is **pending owner approval** |
| 93A | `NO_SIGNAL_AT_SMOKE_SCALE`; H_TARGET NOT SUPPORTED; TARGET_RELIABLE RELIABLE [J] | Smoke scale; feasibility assessment only |
| 93B | `SHARING_SIGNAL` [J], with the note "smoke-scale feasibility evidence only; no architecture validation" | The capacity control was inert |
| 93C | `NO_SIGNAL_AT_SMOKE_SCALE`. Task A ORACLE_FEASIBLE, limiting factor LOCAL_OPERATOR [J] | Smoke scale |

## 3. Ledger

Category keys: a sharp/region C · b broad/whole · c source/receiver stage · d local rank / mirror retention · e first-gather projection / relation exactness · f geometry dependence / co-adaptation · g edit slopes · h encoding-sensitive (g00010, narrow beams, LSPOT) · i explicit comparisons / costs.

### WL86 — `86_WORKLOG_ECRO_GLOBAL_RESPONSE.md` ("Measurements")

| id | cat | Exact values | Condition | Caveats stated there |
|---|---|---|---|---|
| 86-1 | a | E2 T1 region C E_H **0.0717 / 0.0634** (s8601/8602), hl/h 0.90–0.91; oracle 0.0137; rule 0.06 [J 0.0717/0.0634, hl/h 0.901/0.914]. E2 T0 C 0.060 / 0.059 | 4,096 spp, T1 | F2 fails for both seeds → chain stop. E2 dev aggregates were seen once pre-freeze |
| 86-2 | a | E1 T1 C 0.0132 / 0.0110. Panel object: E1 0.040 / 0.034, E2 0.056 / 0.052, oracle 0.012 | E1 is the dev config | Error is 3–5× the oracle: "the learned operator's, not the representation's" |
| 86-3 | b | Stationary pixels: E1 0.0056 / 0.0043, E2 0.0048 / 0.0040 (oracle 0.0032 / 0.0025). D: 0.0049 / 0.0036 and 0.0043 / 0.0034. Capture 0.018–0.024 | T1 | F1 passes |
| 86-4 | b | Field relative L1: validation 0.079 / 0.078, training subset 0.070 / 0.070, target noise 0.048–0.054. Ablations: without global term 0.49–0.50, without single flight 0.51–0.52. Rendered ablation E_H 0.11 / 0.12–0.13 | 1,024-spp targets | Validation ≈ train + 10 %; "does not show memorisation" |
| 86-5 | a/e | Oracle patch×bin table: E_H at noise (stationary 0.0032 / 0.0025; C 0.0074 / 0.0137; panel 0.0121 / 0.0115). Oracle change slope 0.94–1.00, corr ≥ 0.98 | Stage 1, 4,096 spp | This is representability of the *output container*, not of the learned operator |
| 86-6 | g | Higher-order change, slope / corr. D: E1 0.95, 0.96 / 0.90, 0.96; E2 0.80, 0.97 / 0.78, 0.96. **C: E1 0.38, 0.52 / 0.36, 0.56; E2 0.63, 0.98 / 0.47, 0.92.** STALE D: 0.62–0.65 (E1), 0.46 (E2) | T1 − T0 | C has no gate |
| 86-7 | h | LMIX stationary 0.0061 / 0.0051. **LSPOT 0.026 / 0.035, hl/h 1.07 / 1.11, panel 0.056 / 0.118.** Narrow-spot contract field error 0.78–0.89 | E1 T1 geometry | LSPOT is an unseen source class; F4 (0.04) passes |
| 86-8 | f | Dirty state rays 8.7–10.2 %. STALE / ECRO E_H in A 2.6–3.9×; E1 panel 0.27–0.29 vs 0.034–0.040 (7–8×); elsewhere 1.0–1.6× | | "The update matters only locally" |
| 86-9 | i | GPU per spp: PT 1.80 ms, ECRO 0.33 ms. Per-edit wall: PT 35 ms, ECRO 76–78 ms (geometry 31, state 21 [full rebuild 26–28], source 5, operator 11, transfer 1.5). Per-frame GPU advantage 8.8–9.6×. Edit-to-first-frame wall 1.8–5.6× at 0.10 / 0.05 / 0.03. g not reached by either method. **E2 C bias floor 0.073 vs PT 0.040 at 4,096 spp.** Parameters 735,664; factors 43 MB, state 17.6 MB, table 1 MB, torch peak 210 MB. Offline ≈ 15 min supervision + 22 min training | RTX 3080 Ti, uncontended, model 8601 | Verdict NOT TESTED (gated); costs are descriptive. Selective-update saving of 5–7 ms is "immaterial" |
| 86-10 | b/i | LOW (path-traced e1 + e2) supplies 76 % of radiance and the sharp edit response; the learned term supplies ~23 % at ~2 % relative error on broad receivers | Interpretation section | Labelled as inference |

### WL86A — `86A_WORKLOG_ECRO_SHARP_FAILURE_ATTRIBUTION.md`

| id | cat | Exact values | Condition | Caveats |
|---|---|---|---|---|
| 86A-1 | a | Signed learned error: E2 T1 C −0.070 / −0.061; E2 T0 C −0.060 / −0.056; **E1 T1 panel +0.039 / +0.033 (opposite sign)**. Failure is 6–8× noise | Stage A, stored WL86 data | |
| 86A-2 | c | E2 T1 C level part −0.069 / −0.058 vs directional part −0.002 / −0.004. Whole-patch tables 13–20 % low (E2), 6–22 % high (E1). Back-wall field relative L1 0.044–0.045 vs oracle half-split 0.043 | Stage B, exact field → render identity | Camera readout is an amplifier, not a defect |
| 86A-3 | h | ENC_ALL explains −0.2 / −2.6 / −4.9 % of the learned error. Encoding-equivalent rendered E_H 0.025 vs learned 0.063 / 0.072. 16-point vs 1,024-point encoding changes the operator's error by 1.2 / 0.3 / 4.6 % | Point-light workload | The Stage C validation rule FAILED as written (noise proxy). H_A falsified |
| 86A-4 | h | Identical-encoding beam pairs on the panel: patch-level difference 0.94 (noise 0.26) azimuthal, 0.62 (0.22) polar. Region C reading 24 / 30 vs 78 / 77 | Narrow collimated beams | **Post hoc statistic**; the written counterexample rule did not fire |
| 86A-5 | c | Single-flight error E2 T1 C −0.123 / −0.127 (panel senders −0.116 / −0.110); global +0.050 / +0.064. Panel 2-event family is 19 % (T1) / 12 % (T0) of truth; the operator delivers 0.2–3.6 % | Amendment 1 | Post hoc |
| 86A-6 | f | Panel form factor ≥ 0.10: training subset 0.096 / 0.084 vs validation 0.174 / 0.111. Panel own table debiased RMS 0.28 / 0.31 (seen geometries too) | | coverage_generalisation INDETERMINATE |
| 86A-7 | f | Free-factor fit on one geometry: residual / frozen 0.64 / 0.68 (free global tables), 0.62 / 0.49 (all tables free). Held-out-light field error 0.072 / 0.070 vs frozen 0.060 / 0.059 | 64 lights, E2 T1; non-deployable | INDETERMINATE; over-fits |
| 86A-8 | e | State-relation adequacy, RAY (BIN) vs truth: E2 T1 −20 % (−16 %); E2 T0 −20 % (−26 %). Patch RMS 0.50 / 0.93. Median 13–17 of 512 rays hit the panel (minimum 0) | Amendment 2 | Post hoc; E2 T1 is marginal for SAMPLING-LIMITED |

### WL87 — `87_WORKLOG_GM1_EXPLICIT_TRANSPORT_COMPARATOR.md` (§C)

| id | cat | Exact values | Condition | Caveats |
|---|---|---|---|---|
| 87-1 | a/i | **E2 T1 C: EXP 0.0491 [J 0.04906] vs ECRO 0.0717 / 0.0634 vs oracle 0.0137.** E1 panel EXP 0.0185 vs 0.040 / 0.034. E1 C 0.0113 vs 0.0132 / 0.0110 | Same state, encoding, readout; common random numbers | E1 / E2 inspected |
| 87-2 | a | E3 C: EXP 0.0052, ECRO 0.0051 / 0.0063, oracle 0.0050 | Untouched, admitted | E3 C has no panel mirror-cone receivers: "sharp-receiver validation is weak" |
| 87-3 | i | 36 region-class comparisons: ECRO better (both seeds, beyond noise) **0**, worse **3** (E1 T1 panel, E2 T1 C, E2 T1 panel) [J] | 3 edits × T0/T1 × 6 classes | |
| 87-4 | b | EXP stationary 0.0038–0.0041. Field relative L1: EXP 0.043–0.051 vs ECRO 0.045–0.056 (oracle noise 0.020–0.026). Panel table: EXP 0.08–0.10 vs ECRO 0.20–0.23 | T1 | |
| 87-5 | h | LMIX: EXP 0.0039. **LSPOT: EXP 0.0069 vs ECRO 0.026 / 0.035** | | |
| 87-6 | g | C slope: EXP 0.86 / 0.98 / 0.96 (E1 / E2 / E3) [J 0.862 / 0.977 / 0.955] vs ECRO 0.38 / 0.63 / 0.96. D: EXP 0.97–1.01 | | |
| 87-7 | e | E2 T1 C E_H (signed): EXP 0.049 (−0.049), BINQ 0.046, ENC 0.054, RAYSEND 0.059, **C3 0.024 (−0.005)**. E2 T0: EXP 0.055, C3 0.021 | | C3 is not identical-state: it uses a relation per order and per source |
| 87-8 | e | 16 realisations, E2 T1 C: EXP mean 0.044, SD 0.018, range 0.024–0.077, pass rate 75 % [J 0.0444, 0.75]; signed −0.024 [−0.042, −0.006]. Deployed −0.049 (bad tail). Pooled 8,192 rays E_H 0.030 [J 0.0304]. ECRO on the same state −0.058. Important relations never hit: 56–62 % | | Pooled-state residual cause not separated |
| 87-9 | i | Per-edit wall: PT 36, ECRO 69–78, EXP 89–95 ms; C3 0.43–0.58 s; TF64 152 ms; TF4096 4.6 s. ECRO/EXP edit-to-first-frame ratio 0.85–0.98 at 0.10–0.03 and **1.11–1.40 at g**. **Source-only update: ECRO 13–18 ms vs EXP 51–64 ms (ratio 0.25–0.28 [J E1 0.285])**. V = 16 frames: PT 21–40 s, ECRO 3.3–6.5 s, EXP 3.3–6.3 s. Cold first frame: PT 21 s vs table 0.7–0.9 s. Memory: state 17.6 MB, W 3.9 MB, ECRO factors 42.9 MB, table 1 MB, process 3.9–4.9 GB | Alone on the GPU | TF16 wall invalid (compilation). EXP q cost is set by a pre-fixed MIS budget. "≤ 15 % difference … below the 1.25× criterion" |
| 87-10 | e | 512-ray quadrature directional-albedo error, median: panel 5.3 % (max 18 %), B 1.0 %, S2 2.0 % | | Reporting only |

### WL88 — `88_WORKLOG_GM1_DIRECTIONAL_RELATION_FEASIBILITY.md` (historical E2 Task A, descriptive)

| id | cat | Exact values | Condition | Caveats |
|---|---|---|---|---|
| 88-1 | — | No candidate admitted. Common failure is Q5: D higher-order change 3.76× floor vs 4× (Wi) | Qualification partition 4 × 8,192 spp | The Q5 floor used 4 replicas against an 8-replica final: a protocol inconsistency the user chose not to amend |
| 88-2 | e | E2 T1 C E_H (signed): R0 0.049 (−0.049); R0_POOL16 0.030 (−0.023); **R0_INF 0.025 (−0.007)**; R1 0.044; R1_INF 0.027 (−0.017); C3 / C3U / C3UR 0.024 / 0.023 / 0.026; ORACLE 0.017. Target identity holds (R1_INF − R0_INF −0.009 within noise 0.013) | | The deployed deficit is sampling, not a different target |
| 88-3 | e | Factors (T1 / T0): sender vs receiver placement **+0.050 / +0.053** (dominant); power allocation −0.003 / −0.008; fresh relation per order −0.004 / +0.004; quadrature +0.0002 | | Single draw; R0F confounded |
| 88-4 | e | R1 (SAM) coverage 0.59 → 0.993. E_H 0.041 ± 0.017 → 0.030 ± 0.008; paired −0.011 [−0.016, −0.005]. R1M (matched rays) 0.062 | 16 realisations | Descriptive. RS1 control invalid |
| 88-5 | g | C slope: R0 0.98, R1 0.86, R1M 0.78 | | |
| 88-6 | i | R1 relation: 73,282 pairs, 943 k rays, 31 MB. Dev only: R1 update 129 ms vs full 138 ms; R0 20 ms | | Stage 3 economics not run |

### WL90 — `90_WORKLOG_GM1_INTERMEDIATE_PROJECTION_GAP_ATTRIBUTION.md` (WL89 W1 / W2 workload)

| id | cat | Exact values | Condition | Caveats |
|---|---|---|---|---|
| 90-1 | e | M₀ (pooled table recurrence) 0.0363 vs ORACLE 0.0137 → G₀ 0.0226; arm noise 0.0001 | W1 T1 R_mirror, 64 realisations | |
| 90-2 | e | EE (no intermediate projection) − ORACLE −0.0002 (100.9 % of the gap removed); 97 % (W1 T0) and 102 % (W2 T1). ISOLATED in all 3 configurations × 4 regions [J] | | |
| 90-3 | e | Angular-only A +0.0226 / +0.0339 / +0.0177 ≈ J +0.0207 / +0.0309 / +0.0170. Spatial −0.0009 / −0.0011 / −0.0027. First-order share 0.96 / 1.01 / 1.00 [J 0.957]. Compounding 9 % / −1 % / 0 % [J 0.092]. m* = 1 (M₁ within 0.001–0.002 of ORACLE) | | |
| 90-4 | e | 4 sub-bins (256 directions): 9–20 % of J remains. 16 sub-bins: +0.0007 / +0.0012 / −0.0031. Spatial refinement: no effect | | Single-read lower bound; refined recurrence not run |
| 90-5 | e | Panel order-2 field relative L1: M₀ 0.137, EE 0.053, angular-only 0.148, spatial-only 0.089, truth noise 0.043 | | |
| 90-6 | e/i | C3URX1 0.024 ± 0.005 vs C3UR 0.040 ± 0.005; paired −0.016 [−0.020, −0.012] (16/16). R0X1 − R0 +0.003 [−0.006, +0.012]. Source-only 20.6 vs 36.7 ms (0.56×); per-edit 68.2 vs 83.6 ms (0.82×) [J 0.561 / 0.816]. Extra buffer 12.6 MB | 16 realisations | X1 is a diagnostic probe; held-out conditions not evaluated |

### WL91 — `91_WORKLOG_GM1_C3URX1_HELDOUT_VALIDATION_AND_ECONOMICS.md`

| id | cat | Exact values | Condition | Caveats |
|---|---|---|---|---|
| 91-1 | e | Paired C3URX1 − C3UR, R_mirror: W1 T1 −0.0160; **W2 T1 −0.0100 [−0.0153, −0.0046]**; LMIX89 +0.0010 [−0.0017, +0.0037]; LSPOT89 −0.0266 (834 px, ungated). LSPOT89 C_ss **+0.0025 [+0.0005, +0.0045]** [J] | 16 draws | Advantage survives in 1 of 3 held-out conditions; harm criterion violated |
| 91-2 | e | Pooled C3URX1 within −0.0025 … +0.0035 of ORACLE in all 6 conditions. Sampling term: C3URX1 0.005–0.014 vs C3UR 0.0005–0.005. W2 T1 deployed draw 0.053 (signed +0.044), above all 16 realisations | | Outlier ray cause not isolated |
| 91-3 | g | W2 C_ss slope / corr: C3URX1 0.65 / 0.83 FAIL; C3UR 0.79 / 0.93; R0 0.67 / 0.88 FAIL. W2 R_mirror INDETERMINATE for all arms | Deployed draw | |
| 91-4 | h | LMIX89 R_mirror signed error +0.0096 [+0.0046, +0.0146] (C3UR, ORACLE ≈ 0) | | Unattributed |
| 91-5 | i | Edit → frame: 124.7 / 138.2 / 145.4 / 250.1 ms (C3URX1 / C3UR / R0 / PT). Without render: 69.4 / 83.0 / 89.8 / 30.9. Source → frame: 75.8 / 92.2 / 85.4 / 213.1. Source without render: 21.4 / 37.4 / 31.0 / 0. Rays per light: 526,467 probe + 526,467 shadow vs 20,320. State: relation 20.8 MB + buffer 12.6 MB. Process memory 3.33 / 3.41 / 3.27 / 3.16 GB. Cold to first frame 2.38 / 2.28 / 2.90 / 20.94 s. At s = 2: table 27.8 ms, update 24.7 ms, 133.8 MB. 16 sources: 314 vs 517 ms | Warm, compilation excluded | |
| 91-6 | i | Utility rule (a) violated: W2 T1 deployed draw, R_mirror 0.06 tier, 0.74 vs 0.24 s (3.1×). Versus PT: 1.8–5.5× on broad tiers, 7.6–31× on sharp tiers | | |

### WL92 — `92_WORKLOG_ECRO_SHARP_REPRESENTABILITY_IDENTIFIABILITY_ATTRIBUTION.md`

| id | cat | Exact values | Condition | Caveats |
|---|---|---|---|---|
| 92-1 | c | E2 T1 C (share of h): single − L₂ −0.168 / −0.175; global − L₃₊ +0.076 / +0.096; panel family delivered 14 / 18 % (E2 T0 1.6 / 3.0 %). E1 panel global +0.050 / +0.047 | Frozen WL86 generators | The code does not enforce ownership; SUM-only supervision |
| 92-2 | d | **Panel S1 Eckart–Young rank-16 residual 0.52 / 0.46 / 0.46; rank for 99 % energy 46–59; rank-16 NMF mirror retention 0.47–0.54. Learned `a cᵀ`: residual 0.88–0.89, retention 0.11–0.14, mass 0.68–0.76.** Floor B: rank-16 residual 0.03–0.04, rank 8 for 99 % [J: 0.520 / 0.458 / 0.461; rank 59 / 46 / 47; NMF peak 0.472 / 0.535 / 0.528; learned S1 peak 0.135] | Flux-transfer norm | The norm choice was fixed pre-freeze (plain Frobenius is dominated by the grazing ring) |
| 92-3 | c/e | Region C ladder (E2 T1 / E2 T0): ORA2 0.013 / 0.010; RS 0.048 / 0.044; EXP 0.041 / 0.048; ENCQ 0.045 / 0.053; **ENCB (R64, exact local) 0.045 / 0.052; R16 (material NMF) 0.084 / 0.073**; pooled ENCB 0.027 / 0.023; pooled R16 0.064 / 0.056; learned single + true L₃₊ 0.124 / 0.128 and 0.083 / 0.098; true L₂ + learned global 0.052 / 0.064 and 0.028 / 0.041 | 2 × 4,096 spp | About half of ENCB's error is deployed-draw sampling |
| 92-4 | f | Fixed-geometry fits (single + true L₃₊): OWN E2 T1 0.029–0.041; OWN-k16 0.036 / 0.017 **not converged** (6.1 / 11.6 %); SUM whole 0.016–0.043 but single + true L₃₊ 0.074–0.111. Validation-light SUM objective worse than the generator on E2 T0 / E1 T1 (0.052–0.085 vs 0.037 / 0.059) | Non-deployable, one geometry | Over-fitting across sources |
| 92-5 | c | Learned Bq on panel patches: relative L1 0.90–1.11, flux 0.16–0.72 of exact q | | |
| 92-6 | f | SUM allocation ratio 2.4–28×; objective spread 1.26–1.67× → non-identifiability criterion not met | | Spread comes mainly from unconverged fits |

### WL92A — `92A_WORKLOG_ECRO_PHYSICAL_COMPONENT_SUPERVISION.md` (E4 descriptive only)

| id | cat | Exact values | Condition | Caveats |
|---|---|---|---|---|
| 92A-1 | c/b | Validation: single vs L₂ SUM 0.210 / 0.196 → OWN 0.104 / 0.105; global 0.218 / 0.200 → 0.092 / 0.092 [J ratios 0.496 / 0.537, 0.422 / 0.457]. Whole relative L1 0.080 / 0.078 → 0.077 / 0.078. **Whole debiased RMS own_s8602 0.157 vs 0.102** | 55 × 4 validation, 24,000 steps, 928 geometries | Population is 9.4 % smaller than WL86 (disclosed) |
| 92A-2 | d | Panel mirror-bin retention: frozen 0.11 / 0.08, SUM 0.028 / 0.078, **OWN 0.207 / 0.212**. OWN Bq panel flux 1.10 / 1.12 [J] | E4 T1 geometry | Still below rank-16 NMF (0.47–0.54) and ER (1) |
| 92A-3 | a | E4 T1 C: frozen 0.018 / 0.019, SUM 0.025 / 0.024, **OWN 0.013 / 0.012**. E4 T0: OWN 0.020 / 0.017 vs SUM 0.048 / 0.041. ORACLE 0.0096; X_EXP 0.027 (T1) / 0.014 (T0) | | **E4 not admitted** → no held-out verdict. Panel-family share in E4 C is only 0.03 |
| 92A-4 | a | E2 T1 C: OWN 0.077 / **0.040** vs SUM 0.065 / 0.065 | Inspected | Seed-dependent; cause not separated |
| 92A-5 | c | E4 T1 C reading: SUM single −0.098 / −0.092, global +0.129 / +0.114 → OWN −0.008 / −0.007, +0.006 / +0.004. E2 T1 OWN single −0.130 / −0.069 | | |
| 92A-6 | g | C slope / corr: E4 OWN 1.00, 0.99 / 0.98, 0.99 vs SUM 0.95 / 0.93. E2 OWN 0.60 (FAIL) / 0.78. E1 OWN 0.68 / 0.60 vs SUM 0.23 / 0.34 | | E4 not admitted |
| 92A-7 | h | E1 LSPOT stationary: OWN 0.029 / **0.036** vs SUM 0.026 / 0.022 → NO_REGRESSION FAIL. E4 LSPOT4: OWN 0.012 / 0.014 vs SUM 0.017 / 0.020 | | F4 still holds |
| 92A-8 | f/i | STALE E4 T1 C: OWN 0.089 / 0.094 vs SUM 0.059 / 0.064. Runtime: state full / selective 33 / 17 ms; source 4.6 ms; factors 7.2 ms; application 1.1 ms; transfer 2.3 ms; checkpoint 2.97 MB. Training 2,288–2,297 s per arm. Component targets 16 min (one-time) | | Quality ladder not run (conditional on outcome A) |

### WL92B — `92B_WORKLOG_ECRO_OWN_SHARP_AND_SOURCE_ATTRIBUTION.md`

| id | cat | Exact values | Condition | Caveats |
|---|---|---|---|---|
| 92B-1 | c | Accounted fraction f, OWN, E2 T1 / E2 T0: X_L2 0.86 / 0.77, 0.89 / 0.88; **X_Q 0.87 / 0.63, 0.74 / 0.62**; X_M1 0.91 / 0.59, 0.64 / 0.49; **X_M2 −0.26 / −0.59, −0.41 / −0.46**; X_M12 0.74 / 0.25, 0.13 / −0.06; X_L3 −0.36 / −0.68, 0.11 / 0.11; X_SHQ ≈ 0 [J] | Oracle substitutions, non-deployable | H_local INCONCLUSIVE; the tied factor confounds |
| 92B-2 | c | OWN seed gap on E2 T1 C (0.037): X_Q closes 0.96. Panel Bq flux 0.66 vs 1.07 × exact q; relative L1 1.14 / 1.07. Single level error −0.25 / −0.15 of h; directional +0.009 / +0.018 | | |
| 92B-3 | h | E1 LSPOT OWN stationary: X_L3 accounts 0.71 / 0.82; X_SHQ 0.09 / 0.07; encoding effect 0.05 / 0.04 → H_source NOT SUPPORTED. Encoding through the exact matrix vs q on panel 0.36 (point and LSPOT); on all patches 0.03 (point) vs 0.15 (LSPOT) | | |
| 92B-4 | f | **Panel `a cᵀ` change E2 T0 → T1 (panel intrinsically unchanged): frozen 0.34 / 0.44, SUM 0.30 / 0.28, OWN 0.145 / 0.146** [J]. E4 0.03–0.07; floor 0.03–0.05; walls ≤ 0.008. **Seed-to-seed panel difference: frozen 0.54, SUM 1.76, OWN 0.51** [J] | | H_geometry SUPPORTED. A K/C violation cannot be decided because of the tied factor |
| 92B-5 | a/g/h | **E5 (admitted, SNR 13)** region C: T1 SUM 0.028 / 0.026 vs OWN 0.033 / 0.027; T0 SUM 0.028 / 0.030 vs OWN 0.034 / 0.017. LSPOT stationary OWN 0.023 / 0.026 vs SUM 0.022 / 0.016. ORACLE 0.006 / 0.010; X_EXP 0.030 / 0.017; STALE 0.05–0.06. C slope OWN 0.88 / 0.75, SUM 0.76 / 0.89, frozen 1.00 / 0.84. All F1–F4 pass. OWN single − L₂ −0.056 / −0.044; panel family delivered 15 / 51 % | Independently qualified | OWN_CONSISTENT_ADVANTAGE FAIL; NO_REGRESSION FAIL |

### WL93 — `93_WORKLOG_ECRO_SOURCE_STAGE_IDENTIFIABILITY_AND_FACTOR_SHARING.md`

| id | cat | Exact values | Condition | Caveats |
|---|---|---|---|---|
| 93-1 | h | Panel relative L1 vs q (Mq 1,024): exact ER through the deployed encoding 0.508 (validation) vs WL92A OWN learned 0.970 / 0.966 → bound 0.485 / 0.483 FAIL. E2 0.362 passes [J 0.5079 / 0.9704 / 0.9658] | Preflight; nothing trained | STOPPED. Amendment 1 pending |
| 93-2 | h | Post hoc: per-geometry median 0.358; 52 / 55 geometries meet the ratio. g00010 holds 19.7 % of panel q mass, irreducible term 0.99, q-vs-q 0.91; correlation with target noise r = 0.92; flux ratio 1.005 | Post hoc, labelled | **Its "target-noise" reading is corrected by WL93A** |

### WL93A — `93A_WORKLOG_ECRO_SOURCE_STAGE_TARGET_RELIABILITY_AND_SUPERVISION_SMOKE.md`

| id | cat | Exact values | Condition | Caveats |
|---|---|---|---|---|
| 93A-1 | h | Target noise N_b, train-representative: B0 0.095 → B2 0.028 → B5 0.0077 (spatial P dominates; Mq ×4 gives only 1.1×). g00010: 2.67 at B0, **0.66 at B5**; other geometries ≤ 0.06 | 4 realisations | |
| 93A-2 | h/d | M(enc) validation 2.79 [0.36, 3.51], high-confidence 0.56, sharp 0.37; M(OWN) 0.86 / 1.12. **g00010 M(enc) 3.77 vs its noise 0.66** (98 % of the pooled enc numerator). Peak retention: enc 0.95–1.06, OWN 0.16–0.35. H_TARGET NOT SUPPORTED (0.508 → 0.495) | vs B5 reference | g00010 is a genuine encoding/operator discrepancy, not mainly target noise |
| 93A-3 | c | dM = M(TQ) − M(T0): +0.051 / +0.125 (thresholds −0.120 / −0.211) [J]. Whole +4.6–5.4 %, single +7.1–7.3 % (NOREG fails). Secondary: relative L1 0.899 → 0.738 and 0.927 → 0.790; peak 0.315 → 0.484; panel mirror retention 0.237 → 0.311 / 0.321; sharp E2 M 0.831 → 0.586 / 0.758 → 0.612 | 128 geometries, 2,000 steps, tied | Smoke scale; no fit to target noise detected |
| 93A-4 | c | Post hoc: TQ M lower on 54 / 55 and 51 / 55 geometries; g00010 is 78 / 88 % of the pooled change; high-confidence dM −0.053 / −0.069 | Post hoc | Would still fail NOREG |
| 93A-5 | a | Renders, region C T0 → TQ: E2 T1 0.076 → 0.046, 0.042 → 0.035; E5 T0 0.034 → 0.021; E1 LSPOT stationary 0.026 → 0.021, 0.037 → 0.031 | Diagnostic | No superiority claim |

### WL93B — `93B_WORKLOG_ECRO_SOURCE_RECEIVER_FACTOR_SEPARATION_SMOKE.md`

| id | cat | Exact values | Condition | Caveats |
|---|---|---|---|---|
| 93B-1 | b/c | Validation WHOLE: T0 0.0762 / 0.0771, TQ 0.0803 / 0.0806, **UQ 0.0770 / 0.0775**, CQ 0.0802 / 0.0806, **U0 0.0735 / 0.0737**. SRC: T0 0.899 / 0.927, TQ 0.738 / 0.790, UQ 0.737 / 0.808, U0 1.067 / 1.092. RECV (floor 0.0998): UQ 0.1046 / 0.1055, U0 0.1192 / 0.1768. UQ − TQ WHOLE −0.0032 / −0.0031 (79 / 87 % of the regression recovered) [J] | Smoke | UQ reaches parity with T0, not a gain. CQ − TQ within 1e-4: **the capacity control is inert** |
| 93B-2 | f | Panel source vs receiver operator difference: U0 0.72 / 0.68, **UQ 0.57 / 0.68**. Mirror retention source / receiver: UQ 0.31–0.32 / 0.29; U0 0.16 / 0.25–0.29 | E2 T1 | "Weakens the reading of a single shared intrinsic material response" |
| 93B-3 | h | g00010 whole 0.24–0.35 for every arm; UQ source stage there worse than TQ and T0. **E1 LSPOT: U0 0.044 / 0.047 (F4 FAIL)**, UQ 0.020 / 0.033 | | |
| 93B-4 | a/g | Renders, region C: E2 T1 T0 0.076 / 0.042, TQ 0.046 / 0.035, UQ 0.056 / 0.041; E5 T1 UQ 0.042 / 0.028 vs T0 0.032 / 0.025. D slope E2: UQ 0.81 / 0.86 vs T0 0.87 / 0.91 | Diagnostic | Mixed |
| 93B-5 | h | LPAT 2 × 2 cone checker: region C E_H 0.48–0.61 for all arms; stationary 0.34–0.45; reflected spot smeared into a strip in every arm | Visual only | Unseen cone class; the 2-light interface limit forced a 2 × 2 pattern |

### WL93C — `93C_WORKLOG_ECRO_DUAL_STAGE_PHYSICAL_SUPERVISION_SMOKE.md`

| id | cat | Exact values | Condition | Caveats |
|---|---|---|---|---|
| 93C-1 | a/c | ORA (exact M64 at both uses + frozen global): single 0.1033, whole 0.0776 / 0.0774. ORA − TQ: whole −0.0027 / −0.0032, single −0.0064 / −0.0083. **ORA − T0 single +0.0010 / −0.0010** [J]. ORA_PHYS single 0.0998. ORA_RCV single 0.0997 / 0.1033, whole 0.0756 / 0.0772. ORA_SRC whole 0.0796 / 0.0795 | Non-deployable oracle | LOCAL_OPERATOR limiting factor |
| 93C-2 | h | g00010: ORA single 0.464 vs T0 0.289 / 0.393. Per geometry, ORA beats T0 on single for 49 / 55 and 47 / 55 | | The pooled tie comes from g00010 |
| 93C-3 | a/g | ORA renders region C: E2 T1 **0.030 / 0.033**; E5 T0 0.015 / 0.014; E5 T1 0.031 / 0.031. **E2 C slope 0.995 / 1.000** vs T0 0.57 / 0.75, TQ 0.72 / 0.92. Signed single level −1.2 / −0.7 % → +0.15 % | Diagnostic (inspected E2 / E5) | No level-compensation conflict with the global term |
| 93C-4 | c | TR: receiver −0.0023 / −0.0030, whole +0.0009 / +0.0013. TQR: keeps 85 / 91 % of TQ's source gain; receiver back to T0 level (+0.0003 / +0.0005); whole vs TQ −0.0004 / −0.0006 (not significant) | 128 geometries, 2,000 steps | Criterion r fails for both seeds |
| 93C-5 | a | Renders, region C: TR E2 T1 0.075 / 0.062 (worse than T0 for seed 8602); TQR 0.042 / 0.035 | | |
| 93C-6 | c | R* = M64(flight(q)) vs path-traced t2: relative L1 0.064 (E2 T1) | Target property | |

## 4. Cross-batch readings (labelled)

- **SUPPORTED INFERENCE.** Same-state explicit transport is at least as accurate as the learned ECRO table wherever the two were compared (87-3, 87-1, 87-5, 87-6). The learned operator's one measured edge is the source-only update cost (87-9; also 92A-8 ≈ 13 ms of operator work).
- **SUPPORTED INFERENCE.** The per-frame rendering gain comes from the LOW + lookup table structure, not from learning: EXP, TF64 and TF256 share it (WL87 §E).
- **SUPPORTED INFERENCE.** Pooled whole-table relative L1 moves by about 0.003 between arms, while rendered region C moves by about 0.03. The two metrics have different sensitivity (WL93C interpretation; 93B-1 vs 93B-4).
- **SUPPORTED INFERENCE.** The learned panel local operator is geometry- and seed-dependent: it changes 14.5 % with a box move and 51 % between seeds (92B-4). In untied models the source and receiver operators diverge by 57–68 % (93B-2).
- **PLAUSIBLE HYPOTHESIS, untested in the sources.** The rank-16 tied factor cannot serve two incident-field distributions at once (WL93C unresolved list).
- **CONTRADICTED.** WL93's post hoc claim that the gate failure was mainly target noise is contradicted by WL93A (93A-2).

## 5. Claims about this evidence that the sources do not support

1. **"The smoke negatives (WL93A, WL93C NO_SIGNAL) show that source / receiver supervision cannot work, or that the operator cannot represent the response."** UNSUPPORTED. These are feasibility labels at 128 geometries, 2,000 steps and 2 seeds. The pooled metric is dominated by g00010 (93A-4, 93C-2), and per-geometry results mostly favour TQ / ORA. Neither worklog makes a representability claim.
2. **"WL92 completion code 1 proves an ECRO representation limit."** CONTRADICTED by WL92's own disclosure: G_rep was "not established" because the material rank-16 construction failed and the gate fit did not converge. Representation is INCONCLUSIVE, and per-geometry fits fell below 0.06 (92-4).
3. **"Oracle feasibility (WL86 oracle table, WL92 ENCB, WL92B X_Q / X_M12, WL93C ORA) shows the response is learnable."** UNSUPPORTED. All are non-deployable. No learned tied arm reached ORA (93C-4), per-geometry free fits over-fit across sources (92-4, 86A-7), and WL92 Learning (C) is INCONCLUSIVE.
4. **"WL86 REPRESENTABILITY SUPPORTED means the ECRO operator family represents sharp transport."** UNSUPPORTED. It covers only the patch × 64-bin output table (86-5). The operator family with material rank 16 fails (92-3 R16), and learned panel retention is 0.11–0.21 (92-2, 92A-2).
5. **"ECRO's 9× per-frame and 3–6× edit-to-frame advantages are a learning benefit."** CONTRADICTED by WL87: table arms without learning show the same gains. At g, ECRO is 1.11–1.40× slower than EXP (87-9). WL86 itself labels economics NOT TESTED.
6. **"Learned ECRO beats explicit transport."** CONTRADICTED: 0 better, 3 worse out of 36 (87-3). The only edge is the source-only update (3.5–3.9×, about 40–50 ms), and the edit-to-first-frame difference is ≤ 15 % (below the 1.25× criterion).
7. **"Explicit transport (EXP / C3UR / C3URX1) is sufficient for sharp readers and ready to adopt."** UNSUPPORTED:
   - EXP passes only 75 % of E2 realisations (87-8).
   - The angular first-gather floor is 0.036 vs oracle 0.014 (90-1).
   - C3URX1 held-out validity and utility are NOT SUPPORTED; adoption evidence is INSUFFICIENT (WL91).
   - The owner keeps explicit transport only as a comparator (§1 above).
8. **"WL92A shows OWN generalises to held-out sharp scenes."** UNSUPPORTED. E4 was not admitted (D_STOPPED). On the admitted E5, OWN_CONSISTENT_ADVANTAGE FAILS and own_s8601 is worse than SUM (92B-5).
9. **"WL93B shows untying is necessary or improves global transport."** UNSUPPORTED:
   - UQ only removes TQ's regression and reaches parity with T0.
   - The capacity control was inert, so capacity is excluded only for that control form.
   - The best whole fit (U0) is non-physical and fails F4.
   - The label says "no architecture validation" (93B-1/2/3).
10. **"The source encoding causes the sharp failure."** UNSUPPORTED for the point-light workload: weakened in WL86A (86A-3) and H_source NOT SUPPORTED in WL92B (92B-3). Encoding information loss is shown only for narrow beams (post hoc, 86A-4), g00010-type configurations (93A-2), and the visual-only LSPOT / LPAT cases (86-7, 93B-5).
11. **"The LSPOT regression is an encoding problem."** UNSUPPORTED. It is attributed to the learned global term's extrapolation to an unseen source class (X_L3 71–82 %, 92B-3), reproduced on E5.
12. **"The learned local operators are intrinsic, geometry-independent material responses (K-like)."** UNSUPPORTED. The panel operator changes with box position, between seeds, and between source and receiver roles (92B-4, 93B-2). WL92B states that a K/C violation cannot be decided because of the tied factors.
13. **"The ECRO learned term carries the dynamic and sharp edit response."** UNSUPPORTED:
   - The sharp edit response and 76 % of radiance come from physical LOW (86-10).
   - Learned region C slopes are 0.36–0.63 (86-6).
   - OWN slopes on E2 are mixed (92A-6).
   - Only oracles reach a slope of about 1.0 (93C-3).
14. **"Selective state update gives a cost benefit."** UNSUPPORTED. The savings are 5–7 ms (86-9), about 1 ms (WL87 §E) and 16 ms (92A-8, descriptive), all immaterial at this scale.
15. **"WL88 tested (and supported or rejected) a source-independent sharp relation."** UNSUPPORTED. Every hypothesis item is NOT TESTED (workload unqualified). The E2 numbers are descriptive and the C3 sender-placement attribution rests on a single draw (88-3).
16. **"WL93's pooled gate failure was target noise."** CONTRADICTED by WL93A (93A-2).
17. **"Supervision noise explains the WL86 sharp failure."** WEAKENED in WL86A: patch-level target noise is 5–10× below the error.
18. **"WL92B outcome C establishes interacting causes."** QUALIFIED by WL93's reconciliation: under the written rule it is INCONCLUSIVE / D, though no measurement changes.
19. **Held-out caveat on any "held-out" sharp claim.** E2 aggregates were seen pre-freeze. E3 has no mirror-cone receivers. E4 was not admitted. E5 has been inspected since WL93. E6 does not exist. No never-inspected admitted sharp-reader held-out exists for any trained arm after WL92B.
20. **"Neural Full Recompute superiority or inferiority was measured."** UNSUPPORTED. Every worklog states that no external neural baseline was run.

Files relevant to this ledger:
- c:\Projects\ReST-GI\docs\Architecture.md
- c:\Projects\ReST-GI\docs\worklogs\{86,86A,87,88,90,91,92,92A,92B,93,93A,93B,93C}_WORKLOG_*.md
- JSON checked:
  - c:\Projects\ReST-GI\results\ecro86\results\final.json
  - c:\Projects\ReST-GI\results\ecro_attr92\results\local.json
  - c:\Projects\ReST-GI\results\ecro_own92a\results\{local92a,final92a}.json
  - c:\Projects\ReST-GI\results\ecro_cf92b\results\{local92b,final92b}.json
  - c:\Projects\ReST-GI\results\ecro_stage93\{ident,recon92b}.json
  - c:\Projects\ReST-GI\results\ecro_stage93a\final93a.json
  - c:\Projects\ReST-GI\results\ecro_stage93b\final93b.json
  - c:\Projects\ReST-GI\results\ecro_stage93c\oracle_gate.json
  - c:\Projects\ReST-GI\results\gm1_explicit87\results\final87.json
  - c:\Projects\ReST-GI\results\gm1_relation88\results\final88.json
  - c:\Projects\ReST-GI\results\gm1_gap90\results\final90.json
  - c:\Projects\ReST-GI\results\gm1_c3urx1_91\results\final91.json