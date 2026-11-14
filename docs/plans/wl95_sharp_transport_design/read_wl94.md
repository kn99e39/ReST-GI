# WL94–WL94D evidence extraction (read-only)

**Labels used below.** [FACT] means verified in code or a doc. [MEAS] means a value reported in a worklog or a result JSON. [INF] is an inference. [HYP] is untested.

**Paths.**
- Worklogs: `docs/worklogs/94_WORKLOG_ECRO_MATERIAL_GENERALIZATION_VS_SHARP_ANGULAR.md` (WL94), `94A_WORKLOG_ECRO_MATERIAL_FIDELITY_RETENTION_SMOKE.md` (94A), `94B_WORKLOG_ECRO_SHARP_PANEL_TRANSPORT_EXPOSURE_SMOKE.md` (94B), `94C_WORKLOG_ECRO_PANEL_MEDIATED_QUERY_SUPERVISION_SMOKE.md` (94C), `94D_WORKLOG_ECRO_SHARP_SENDER_ORACLE_SUPERVISED_SMOKE.md` (94D).
- Spot-checked JSONs: `results/ecro_mat94/final.json`, `results/ecro_exp94b/audit.json`, `results/ecro_exp94b/final_b.json`, `results/ecro_q94c/final_q.json`, `results/ecro_s94d/{audit,lam,final_s,s_s8601,s_s8602}.json`, `results/ecro_s94d/sender/SN_*.json`.

**Common definitions.**
- E_H is the error in the untextured or textured CK2 ROI_T, arm C0 unless stated otherwise.
- tol = max(2 × noise bound, 0.005) ≈ 0.017; the noise bound is ≈ 0.0084–0.0086 [MEAS `s_s860x.json` criteria].
- "own" means the frozen own_s8601 / own_s8602 checkpoints.

---

## WL94 — material generalization (H-M) vs sharp angular limit (H-A)

**Question.** Is the frozen ECRO's under-response to reflected albedo caused by material–object entanglement in training (H-M), by an angular limit (H-A), or by both? [FACT] 94:7

**Implementation facts.**
- [FACT] Training materials are fixed per object. The only material input is the 69-d `mat_states[PMAT]`, so the network saw exactly one albedo per object (94:27-31).
- [FACT] The reflected amplitude is set by the learned B table at the Lambert box patches. The panel lobe enters only through the panel's learned sender and the global term (94:33-36).
- [FACT] All 13 Stage A tests pass (94:43-49).

**Intervention.**
- Stage B: six frozen counterfactual material arrangements on box A (none, U07, U03, bx75, bx38, bx38i) (94:67-76).
- New arm C2_srcS1: M64 substituted only at the S1 panel sender (94:81-83).
- Stage C, paired smoke: from own_s8601, 2,000 steps, 96 geometries. A keeps the original materials; B uses per-rectangle albedo from {0.2, 0.4, 0.5, 0.6, 0.7, 0.9} (94:96-101).

**[MEAS] Stage B, frozen own, box-A log-response amplitude (s8601 / s8602)** (94:128-133).
- Learned S: U07 0.74 / 0.73; U03 0.01 / 0.37; bx38 0.25 / 0.46.
- Exact receiver (C2_rcv): 0.97–1.00.
- Learned G: 0.16–0.63.
- Sender substitution leaves the amplitude unchanged (±0.02) (94:136).

**[MEAS] Sender split, untextured** (94:146-149).
- The panel-sender single flight (e2_p) is 10 % of S over box A and 59 % on the panel face.
- Its error is 0.59 / 0.87 with the learned receiver and 0.59 / 0.91 with the exact receiver.
- Non-panel senders have errors of 0.05 / 0.06.

**[MEAS] Images, untextured E_H** (94:155-160).

| | s8601 | s8602 |
|---|---|---|
| C0 | 0.047 | 0.042 |
| C2_srcS1 | 0.031 | 0.019 |
| noise bound | 0.0084 | 0.0084 |
| textured C0 range | 0.062–0.106 | 0.030–0.122 |

**[MEAS] SSR_frozen** (94:172-174). `final.json` gives `SSR_frozen [false, true]`.
- s8601: diff 0.0162, share 0.34, against 2 × noise 0.0167. Fails by 0.0005.
- s8602: diff 0.0226, share 0.54. Passes.

**[MEAS] Stage C, s8601 only** (94:180-217).

| | A (fixed) | B (diverse) |
|---|---|---|
| V_orig (init 0.0778, limit 0.0856) | 0.0768 | 0.0879 (F2 FAIL) |
| V_ck (init 0.305) | 0.315 | 0.106 |
| CK2 bx38 amplitude | 0.29 | 1.06 |
| CK2 U03 amplitude | 0.13 | 0.99 |
| untextured E_H | 0.039 | 0.087 (signed −0.086) |
| bx38 E_H | 0.066 | 0.095 |
| SSR_C | — | diff 0.0098, share 0.11 |

**[MEAS] Post hoc** (94:223-234).
- The error is a face effect, not a bin effect. On the panel face, panel-facing and other bins have equal error: own s8601 0.251 vs 0.267, B 0.409 vs 0.374.
- On the panel face, 61 % of B is the higher-order part.
- Panel-face G bias: own −0.17 / −0.11, A −0.02, B −0.46.
- C2_srcS1 moves the panel-face S bias from −0.34 to +0.13, but its rel L1 stays at 0.40.

**Status.**
- Labels [FACT `final.json`]: H-M INCONCLUSIVE (INCONCLUSIVE_SMOKE_FIT) and H-A INCONCLUSIVE.
- Established [MEAS/INF]:
  - The compression is generic across S, G, faces, bins and readers (ROI U07 0.36–0.47 vs DIFF 0.41–0.42).
  - Diversification restores the material response.
  - A sharp-sender residual exists with in-distribution materials.
- Inconclusive:
  - whether F2 and sharp fit can be retained together;
  - representation class (not tested) (94:271-274).

## WL94A — 50/50 material mixture (H-R)

**Question.** Does a 50/50 mixture of original and diversified examples keep F2 and still recover the material response? [FACT] 94A:7

**Intervention.** Candidate C: even ranks use original materials, odd ranks use diversified ones (48 / 48). Code `c94.train` is unchanged: 2,000 steps, same init. A_s8602 was added as a control (94A:23-31, 45).

**[MEAS] Primary criteria, C0** (94A:69-77).

| | s8601 A | s8601 B | s8601 C | s8602 A | s8602 C |
|---|---|---|---|---|---|
| V_orig | 0.0768 | 0.0879 ✗ | 0.0808 ✓ | 0.0798 | 0.0841 ✓ (limit 0.0883) |
| CK2 amplitude (U07 / U03 / bx75 / bx38 / bx38i) | — | — | 1.03 / 0.94 / 1.04 / 1.02 / 1.02 | 0.61 / 0.34 / 0.65 / 0.45 / 0.45 | 1.02 / 0.98 / 1.04 / 1.04 / 1.04 |
| V_ck population | 0.40 | 1.05 | 1.05 | 0.50 | 1.08 |
| untextured E_H (signed) | 0.039 (+0.003) | 0.087 (−0.086) | 0.084 (−0.082) | 0.093 (+0.088) | 0.067 (+0.058) |

- The RET limit is A + 0.017.

**[MEAS] Panel face** (94A:88-92).
- Learned G: s8601 A −0.038, B −0.442, C −0.416; s8602 A −0.057, C −0.071.
- Panel-sender single flight vs e2_p:
  - s8601: C 0.534 (−0.48), A 0.534 (−0.17);
  - s8602: C 1.02 (+0.67), A 1.21 (+0.86).

**[MEAS] Textured E_H** (U07 / U03 / bx75 / bx38 / bx38i) (94A:95-98).

| seed | A | C |
|---|---|---|
| s8601 | 0.056 / 0.163 / 0.041 / 0.066 / 0.049 | 0.086 / 0.059 / 0.081 / 0.082 / 0.067 |
| s8602 | 0.044 / 0.218 / 0.055 / 0.075 / 0.050 | 0.051 / 0.078 / 0.057 / 0.066 / 0.062 |

**[MEAS] ROI texture slope** (94A:104-107): s8601 A 0.24–0.57 vs C 0.61–0.82; s8602 A 0.41–0.64 vs C 0.93–1.21.

**[MEAS] SSR share** (94A:112-117).

| | A | C |
|---|---|---|
| s8601 | 0.30 | 0.10 |
| s8602 | 0.72 | 0.66 |

**[MEAS] Determinism.** Ar reproduced A to about 1e-11 (94A:43).

**Status.**
- Outcome: INCONCLUSIVE. s8601 gives MATERIAL_GAIN_WITH_BASELINE_LOSS and s8602 gives COEXISTENCE_OBSERVED.
- The forgetting answer by the frozen code is CONTRADICTED. It reads s8601 only, a wording gap that is disclosed (94A:79-83, 155).
- Established: F2 and MR hold together on both seeds.
- Inconclusive:
  - Sharp baseline retention is seed-dependent.
  - A_s8602 itself drifted from 0.042 to 0.093, so the RET baseline is unstable (94A:131, 154).

## WL94B — sharp-panel training exposure (H-P), stopped at audit

**Question.** Is sharp-panel-mediated transport insufficiently constrained relative to its importance? [FACT] 94B:7

**Intervention (not trained).** Candidate D: per material half, the top-12 PM items (stratum H) get 500 of 1,000 steps (94B:45-50).

**[FACT] Supervision semantics** (94B:31-38, 63).
- Every step supervises 4 lightings × 1,270 patches × 64 bins × 3 channels = 975,360 queries.
- The loss is per-(lighting, patch, channel) normalised rel L2 (den = bin mean + 0.05 × item mean) on three pairs: sum vs f, S vs e2 = e2_p + e2_o, and G vs e3.
- There is no path-specific weight. The panel sender is supervised only jointly through receivers.

**[MEAS] Coverage** (`audit.json`; 94B:71-81).
- Item-mean shares of the loss-normalised target: e2_p 0.0211 and e3_p 0.0132, so **panel-mediated ≈ 3.4 %**; e2_o 0.511 and e3_o 0.454.
- PM on box receivers: mean 0.0048 (≈ 0.5 %), median 0.0043, p10–p90 0.0022–0.0073, max 0.0137.
- PMb_A: mean 0.0362, median 0.023, p90 0.075, max 0.252. CK2 PMb_A is 0.1377 and PM_A 0.0059.
- Items reaching the CK2 PMb_A level: 4 / 96 (4.2 %).
- e2_p share of box e2: training mean 0.068; CK2 box A 0.106; CK2 panel face 0.580.
- e3_p share of box e3: training mean 0.057; CK2 box A 0.166; CK2 panel face 0.755.
- The panel face holds 0.828 of box A's panel-mediated mass.
- Box share of the item target: 0.077.

**[MEAS] Gates.**
- G_REL: Spearman 0.9989. Pass.
- G_CONTRAST: 0.00813 / 0.00370 = 2.20. Pass.
- G_EXPOSURE: 1.232 / 1.236 against ≥ 1.25. **FAIL**, giving STOP_INVALID_INTERVENTION [FACT `final_b.json`].

**[MEAS] Post hoc** (94B:94-104).
- 69 % of the CK2 panel-face loss-normalised target is panel-mediated.
- Best training face: median 33 %, p90 60 %.
- 4.5 % of box-face mass lies on faces with at least half of CK2's share.
- Reallocation bounds: 1.23× (protocol D), 1.70× (top quartile only), 2.84× (single top item).

**Status.**
- Established: UNDERREPRESENTED by the coverage rule, and item-level resampling has bounded reach.
- Untested: H-P, because D was not trained.

## WL94C — panel-mediated query-level emphasis (H-Q)

**Question.** Does weighting physically identified panel-mediated box queries more heavily preserve sharp transport, beyond the effect of matched non-panel emphasis? [FACT] 94C:7

**Intervention** (94C:44-50).
- D: κ = 8 on box queries with pm_q ≥ 0.5; the other queries of the same item are renormalised so the item mean weight is 1.
- E: a matched control of box queries with pm_q < 0.10, chosen by Hungarian matching on log err0, log exitance and box A / B.
- 2,000 steps from the C init.

**[MEAS] Feasibility** (94C:62-71; `final_q.json`).
- 974 D queries, 2.3 % of box queries.
- Box panel-mediated exposure 2.91× / 2.90×; D query weight share 8.0×; all panel-mediated mass 1.25×.
- E box panel-mediated exposure 1.03×.
- SMD 0.054–0.066.

**[MEAS] Untextured E_H (signed)** (94C:85-90).

| | own | C | D | E |
|---|---|---|---|---|
| s8601 | 0.047 (−0.040) | 0.084 (−0.082) | 0.044 (−0.030) | 0.086 (−0.083) |
| s8602 | 0.042 (+0.028) | 0.067 (+0.058) | 0.089 (+0.086) | 0.066 (+0.057) |

- With the exact S1 sender (C2_srcS1):
  - s8601: own 0.031, C 0.076, D 0.033, E 0.077;
  - s8602: own 0.019, C 0.023, D 0.019, E 0.022.

**[MEAS] Panel face, rect 6** (94C:101-113).

| | own | C | D | E |
|---|---|---|---|---|
| Sender vs e2_p, s8601 | 0.61 (−0.58) | 0.53 (−0.48) | 0.50 (−0.31) | 0.53 (−0.48) |
| Sender vs e2_p, s8602 | 0.76 (+0.32) | 1.02 (+0.67) | 1.25 (+0.96) | 1.02 (+0.67) |
| G vs e3, s8601 | 0.20 (−0.19) | 0.42 (−0.42) | 0.22 (−0.19) | 0.43 (−0.43) |
| G vs e3, s8602 | 0.15 (−0.13) | 0.092 (−0.071) | 0.067 (−0.011) | 0.090 (−0.066) |

- Reference half noise: 0.32 for the sender, 0.027 for G.

**[MEAS] Textured E_H.**
- s8601: D 0.034–0.051 vs C 0.059–0.086.
- s8602: D 0.055–0.114 vs C 0.051–0.078.

**[MEAS] Retention.** F2 for D: 0.0810 / 0.0843. MR passes for D and E.

**Status.**
- Outcome: INCONCLUSIVE (seed disagreement).
- Established, on both seeds:
  - E = C, so there is no generic supervision effect;
  - D with the exact sender reaches own;
  - D's sender bias shifts upward (+0.17 / +0.30).
- Inconclusive: the integrated sign, and the sender changes, which lie below the 0.32 sender half noise.

## WL94D — S1 sender oracle (M64) supervision (H-S)

**Question.** Does the existing sender reproduce the sharp-panel response better when trained against a source-specific oracle? [FACT] 94D:6

**Intervention** (94D:75-85; s94.py:216-260).
- Continuations from WL94C D: 1,000 steps, Adam 1e-4 with warm-up 100 and cosine decay to 1e-5, order rng(seed + 9400).
- T adds λ_T × teacher on the 42 S1 patches (1088–1129).
- R adds λ_R × teacher on the 1,228 non-S1 patches.
- P is a plain continuation.
- [MEAS `lam.json`] λ_T = 0.0189 / 0.0127 and λ_R = 0.934 / 1.143. Gradients: own 0.171 / 0.176, T 9.07 / 13.85. Initial S1 teacher loss 7.59 / 7.43.

**[FACT] Contract** (s94.py:3-5, 47-54).
- Learned sender: `Bl = a·(cᵀ s)`. Exact sender: `Bm = s·M64[mat]`.
- They share the incident field, the outgoing bins, the frames and the units.

**[MEAS] Audit** (`audit.json`; 94D:42-47).
- T_SEM: C2_srcS1 vs stored is 1.51e-7 (s8601) and 6.0e-8 (s8602); the worklog prints "7.6e-8" for s8601. The threshold is 1e-5 (s94.py:125). Bl path vs `model.apply` is 0.0.
- T_ENERGY: M64 S1 albedo max 0.825, mean 0.635.
- G_PHYS on training items:
  - pooled ratio 0.953;
  - corr exact vs label 0.641;
  - label reliability (256 vs 64 spp) 0.525;
  - corr exact vs the independent label 0.446;
  - rel L1 0.641.
  - The first-draft rule (corr ≥ 0.9) failed and was revised before the freeze.

**[MEAS] CK2 flight-floor diagnostic.** Exact S1 sender plus exact receiver against the e2_p reference (mean of halves, 2 × 4,096 spp):

| receivers | ratio | corr | corr_halves | rel L1 | half-noise rel L1 |
|---|---|---|---|---|---|
| all non-panel | 1.378 | 0.713 | 0.930 | 0.940 | 0.127 |
| box A | 1.211 | 0.716 | 0.983 | 0.528 | 0.058 |
| panel face | 1.206 | **−0.334** | 0.867 | 0.598 | 0.069 |

**[MEAS] Coverage.**
- Training excites all **64 / 64** incident bins.
- Validation excites 62 (0 outside training); CK2 excites 4 (0 outside training; irradiance share in trained bins 1.0).
- Held-out bins therefore do not exist, and the operator `heldout_bins` value is `None` in every SN_*.json.

**[MEAS] Sender metrics from `sender/SN_*.json`.** Values are train / val / CK2.

| seed, model | rel_l1_flux | energy_ratio | peak_ratio | delivery_ratio | delivery_box_rel_l1 | operator rel L1 (val / CK2) |
|---|---|---|---|---|---|---|
| 8601 own | .796 / .878 / 1.075 | .927 / .828 / .557 | .287 / .224 / .038 | .895 / .789 / .382 | .343 / .494 / .634 | .975 / 1.018 |
| 8601 D | .817 / .892 / 1.212 | .974 / .866 / .867 | .298 / .234 / .057 | .946 / .829 / .601 | .335 / .486 / .571 | .994 / 1.019 |
| 8601 T | .725 / .855 / 1.544 | 1.082 / .992 / 1.749 | .409 / .316 / .215 | 1.077 / .980 / 1.279 | .304 / .448 / 1.052 | 1.002 / 1.044 |
| 8601 R | .819 / .886 / 1.340 | .961 / .855 / 1.261 | .306 / .244 / .113 | .938 / .825 / .890 | .341 / .484 / .687 | .990 / 1.027 |
| 8601 P | .816 / .886 / 1.294 | .986 / .871 / 1.010 | .303 / .238 / .065 | .959 / .838 / .702 | .330 / .474 / .618 | .993 / 1.035 |
| 8602 own | .789 / .850 / 1.221 | .911 / .932 / 1.123 | .302 / .261 / .129 | .893 / .922 / .764 | .363 / .600 / .783 | .964 / .987 |
| 8602 D | .799 / .874 / 1.587 | .945 / .992 / 1.618 | .313 / .272 / .150 | .929 / .984 / 1.092 | .349 / .652 / 1.201 | .973 / .993 |
| 8602 T | .712 / .846 / 1.484 | 1.007 / 1.090 / 1.709 | .385 / .327 / .258 | .992 / 1.088 / 1.185 | .325 / .648 / 1.241 | .965 / .987 |
| 8602 R | .797 / .866 / 1.573 | .912 / .963 / 1.653 | .321 / .287 / .220 | .893 / .950 / 1.146 | .360 / .663 / 1.297 | .961 / .990 |
| 8602 P | .799 / .870 / 1.629 | .951 / .994 / 1.665 | .316 / .274 / .153 | .936 / .988 / 1.125 | .345 / .648 / 1.231 | .972 / .996 |

- The operator rel L1 on the training items is 0.964–1.007 for every model and seed.

**[MEAS] S1 teacher loss** (first 50 → last 50 steps) (94D:92-93).
- T: 6.42 → 4.80 (s8601), 6.29 → 5.57 (s8602).
- P: 6.52 → 6.29 (s8601), 6.45 → 6.75 (s8602).

**[MEAS] Integrated E_H, untextured** (`s_s860x.json`).

| | D | T | R | P |
|---|---|---|---|---|
| s8601 E_H (signed) | 0.0439 (−0.030) | 0.0619 (+0.040) | 0.0438 (+0.003) | 0.0443 (−0.022) |
| s8601 with exact sender | 0.0328 | 0.0340 | 0.0302 | 0.0346 |
| s8602 E_H (signed) | 0.0889 (+0.086) | 0.0965 (+0.095) | 0.1010 (+0.095) | 0.0890 (+0.087) |
| s8602 with exact sender | 0.0192 | 0.0179 | 0.0224 | 0.0174 |

- s8601: T is worse than P by 0.0176 against tol 0.0170, a **0.0006 margin**. T is also worse than R (0.0181) and D (0.0180).
- s8602: all differences are within tol.
- V_orig for T: 0.0829 / 0.0848 (limits 0.0856 / 0.0883). MR passes.
- Panel-face sender vs e2_p: T 1.04 (+0.77) / 1.49 (+1.28); P 0.52 (−0.19) / 1.29 (+1.02).
- Panel-face G: T 0.21 / 0.060; P 0.23 / 0.060.
- Other box-A faces, G: T 0.073 / 0.086; P 0.054 / 0.081 (94D:128-139).
- Textured E_H: s8601 T 0.048–0.064 vs P 0.035–0.052; s8602 T 0.065–0.117 vs P 0.056–0.114.
- Image texture slope: T 1.09–1.17 / 1.04–1.31; P 0.86–0.94 / 0.92–1.17.

**Status.**
- Outcome [FACT `final_s.json`]: **NO_RELIABLE_IMPROVEMENT**. SENDER_IMP and RENDER_IMP are false on both seeds; KEEP_T is true on both.
- SENDER_IMP rule: T ≤ 0.8 × min(R, P) on val and on CK2 (s94.py:386). Measured ratios: val 0.97 / 0.98, CK2 1.19 / 0.94.
- Established:
  - The sender is directly supervisable (same quantity).
  - The T gain is S1-specific (R ≈ P).
  - The gain is mostly energy, not lobe shape (peak ≈ 0.3).
  - CK2 over-emits (energy 1.75 / 1.71).
  - The 64-bin flight floor exists even for the exact sender.
- Inconclusive:
  - optimisation vs representation (the operator stays ≈ 1.0, far above WL92's rank-16 Eckart–Young floor of 0.46–0.52, which uses a different norm) (94D:109, 147, 163);
  - the teacher normalisation favouring level over peak [HYP] (94D:164).

---

## Metric definitions in `experiments/ecro_s94d/s94.py`

**[FACT] Outgoing weight W** (an94.py:18; used at s94.py:287).
- W[d] = ((d // NPHI + 0.5) / NZ) · 2π/NB, with NB 64, NPHI 8, NZ 8.
- This is the bin's mid-cosθ times dΩ.

**[FACT] Sender field rel L1, flux-weighted** (s94.py:297-300, 293-294).
- Bl and Bm are taken on the panel patches only.
- `rel_l1_flux` = Σ|Bl − Bm|·W / Σ Bm·W.
- `energy_ratio` = Σ Bl·W / Σ Bm·W.
- Sums run over lightings, panel patches, outgoing bins and channels, accumulated over all items of the split before the ratio is taken (pooled, not a per-item mean).
- No patch-area weight is applied.

**[FACT] Peak ratio** (s94.py:301-302).
- pk = argmax over outgoing bins of Bm, per (lighting, patch, channel).
- `peak_ratio` = Σ Bl[pk] / Σ Bm[pk], pooled.

**[FACT] Delivery metrics** (s94.py:303-312).
- The panel-only Bl or Bm field is flown with `EB.flight(model.rw, …)` and then passed through the exact receiver M64[mat].
- Exitance = Σ_bins,ch (·)·W.
- `delivery_rel_l1` and `delivery_ratio` use all non-panel receivers.
- `delivery_box_rel_l1` uses patches 1130:1238, which are hard-coded. That these are box A is [INF], from the worklog's "box A" wording and rect 6 being the panel-facing face; not verified against masks.
- [FACT] `rw` is a fixed buffer, `ray_weights()` = cosθ dΩ / κ (experiments/ecro86/model86.py:40-46, 68). It is not learned.

**[FACT] Operator flux-transfer weighted L1** (s94.py:314-323).
- A[n,m,d,c] = Σ_r c[n,m,c,r] · a[n,d,c,r] over the S1 panel patches, compared against M1 = M64[material 1].
- X = |A − M1| · zb_m · zb_d, with zb = (k // NPHI + 0.5) / NZ.
- rel = Σ X·mask / Σ M1·zb_m·zb_d·mask, split by training-excited vs never-excited incident bins.
- The operator is input-independent: incident bins are weighted uniformly, not by the field.
- Sums are accumulated over the items of the split, then divided (s94.py:333-335).

**[FACT] Teacher loss** (s94.py:57-60).
- mean(((Bl − Bm) / den)²), with den = Bm.mean(over outgoing bins) + 0.05 × Bm.mean(over patch, bin, channel per lighting) + 1e-12.

**[FACT] CK2 flight-floor diagnostic** (s94.py:152-162).
- Sx = flight(exact Bm on the panel only) → exact receiver.
- Exitance with W is compared against er = ½(e2_p_a + e2_p_b) from `CKF_none.npz`.
- ratio = Σexm / Σer; corr; corr_halves = corr(ea, eb); rel L1 = Σ|exm − er| / Σer; half-noise rel L1 = 0.5·Σ|ea − eb| / Σer.
- Masks: receivers (non-panel), box A (`Mk['T']`) and panel face (`Mk['T_panel_face']`).

**[FACT] G_PHYS** (s94.py:139-165).
- Training items; non-panel receivers; exact-sender exitance X against the 256-spp label Yt and the 64-spp label Yi.
- Pass condition: 0.8 ≤ ΣX / ΣYt ≤ 1.25 and corr(X, Yt) ≥ corr(Yt, Yi).

**[FACT] T_ENERGY** (s94.py:179-182).
- albedo per incident bin = Σ_d M1·zb_d · 2π/NB; pass if ≤ 1.0 + 1e-3.

**[FACT] Coverage.**
- `bins_tr` = OR over training items of `s[:, PANEL].sum > 0` (s94.py:131-133).
- Validation and CK2 coverage are computed the same way (s94.py:169-176).

**Discrepancies noted** [MEAS vs doc].
- T_SEM s8601 is 1.51e-7 in `audit.json` but 7.6e-8 in 94D:42. Both are far below the 1e-5 threshold.
- The s8601 "T worse than P" judgment rests on a 0.0006 margin over tol.