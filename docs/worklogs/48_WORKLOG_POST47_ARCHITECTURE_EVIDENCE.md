# Worklog — Post-47 Architecture Evidence: Directional Validity, Nonlocal Capacity and Composition Attribution (2026-10-27)

Protocol `fnto_diag48` + amendments 1–2 (`experiments/fnto_diag48/protocol/diag48.json`). Authorisation: the user brief
"ReST-GI — Worklog 48: Post-47 Architecture Evidence Pipeline" (2026-10-27).

Frozen throughout:
- K_k44P (sha256 `30d55da3…`, equal to the worklog-45 freeze, digest identical after every GPU step);
- relations, anchors, references;
- the code of worklogs 41–47 (imported or subclassed only).

Nothing was trained, and no new render was made. All evidence comes from frozen-model queries, the worklog-45 Burley
oracle, the worklog-46 MS scene and its references, and the worklog-46 known-field (KF) oracle.

**Answer to the completion question.**

| Failure | Attribution |
|---|---|
| **K44's directional-sender failure** (worklog 46) | Not mainly a hidden-supervision effect. K44's Yellow nonlocal output is almost isotropic at every queried direction except a 5–9× deficit toward the port normal. Its first-event energy (R 4.3–4.6× the absorbed sun power) matches what the reference semantics imply. The worklog-45 Burley oracle, scaled to reproduce the references, gives 4.8× (post hoc). Physical passivity and reference fidelity are therefore in conflict for Yellow under Cycles' Burley on this object. |
| **Yellow nonlocal capacity** | K44's band / mid representative aggregation is a demonstrable bound: 0.35 / 0.26 / 0.25 shadow error (R / G / B) even with ideal weights. The 30 cm support alone is not binding once a global rank-64 term covers the tail (0.008). K44's actual error (0.52 R, hemispherical mean, post hoc) is 1.5× this structural bound. |
| **Worklog 46 MS over-brightness and edit degradation** | Not the 19-bin basis. Exact per-ray reads are −0.2 dB vs basis reads, with equal edit correlation. Both effects coincide with the Yellow reflected contribution, which grows 4× once senders are read off-normal. K-side cause inferred, not isolated. |
| **Worklog 46 κ 8 C residual** | Mostly angular quadrature of the 128 gathers. Receiver port position is second. Sender port aggregation is negligible. |

---

## STAGE 0 — WORKLOG 47 INTAKE

Verified: commits 3b1b70d / 976778c / 53f8d31, protocol `fnto_smoke47` + amendment 1, tests ALL_PASS, results in
`results/fnto_smoke47/`.

| Question | Worklog 47 evidence |
|---|---|
| Meaningful angular-sensitive nonlocal response in a physical reference? | Yes for the constructed internal-mirror cavity: D 0.98–1.11 at δE ≤ 5e-6. No for Lego Yellow Burley: R ≤ 1.0 %, G ≤ 1.9 % within bounds; B inconclusive beyond 6 cm. |
| Renderer experiment valid? | Yes: E matched, superposition to noise. |
| Directional input improves held-out prediction? | Yes, with full-boundary supervision only (dense linear surrogate). |
| Independent supervision benefit? | No: it appears only together with directional input. |
| Regime relevant to the FNTO contract? | MIR is a constructed K-intrinsic internal-reflection path (glass-like) that was not measured on the Lego. SSS is the Lego Yellow material under the reference's own Burley semantics (displacement removed). |

**Use in this batch.**
- The optional isolated learning control of Probe A was skipped. Worklog 47 Stage C already separates supervision from
  representation in a fixed linear family.
- Worklog 47's three-camera Yellow SSS profiles serve as the physical reference for Yellow nonlocal exit isotropy:
  max / min over the three cameras is 1.0005–1.008 per band and channel.

**Factual correction to worklog 47** (recorded here; worklog 47 is not edited). Worklog 47's INTERPRETATION says
"K_band takes a 7-bin incident partition". The code has `N_BINS_H = 19` (`experiments/fnto_lego/fnto_common.py`). The
sentence was descriptive context, and no worklog-47 measurement or verdict depends on it.

**Factual qualification of worklog 45.** Worklog 45 called K44's Yellow first-bounce power (1.8× the sun power in red,
normal-direction semantics) "beyond any physical bound". That statement is physically correct. Post-hoc P1 below shows
that the reference semantics extrapolated by worklog 45's own Burley oracle exceed the bound as well, so the excess is not
by itself evidence of a K defect.

---

## IMPLEMENTATION FACT

| File (`experiments/fnto_diag48/`) | Role |
|---|---|
| `probe_a48.py` | Frozen-K44 sender queries per port group (normal / training-camera / C-requested / 19 bins / 64 cosine directions) and first-event energy accounting |
| `probe_b48.py` | Burley-oracle capacity bounds: support-limited, K44-structure quadrature, rank-r, combinations; field responses; K44 comparison |
| `probe_c48.py` | `MSStateX`, a subclass of `ms46_eval.MSState` that adds exact per-ray plane reads. Also C0 recompute, C2b, per-ray basis-vs-exact comparison, per-class plane contributions and edit metrics. |
| `kf48.py` | Known-field κ 0 / 2 / 8 decomposition, arms K1–K4 |
| `diag48_posthoc.py` | Post-hoc diagnostics P1 / P2 (amendment 2) |
| `panels48.py` | Numeric panels |
| `tests/test_diag48.py` | Focused tests |

**Tests** (before evidence): ALL_PASS.

| Test | Result |
|---|---|
| K44 sha256 vs worklog-45 freeze | equal |
| Direction conventions | training px_cam front-facing 1.000; C-requested −d front-facing 0.989; bin centres in own bin |
| Cosine quadrature | constant → π; cosine-weighted 2.0948 vs 2π/3 = 2.0944 |
| Nonlocal query linearity | 5e-7 |
| Support tables | Σ nbrA / area(5 cm) 1.002, Σ midA / area(30 cm) 1.006 |
| Burley oracle | self-exclusion; plane normalisation 0.988 |
| Exact query = basis value at bin centres | 2e-8 |
| Paired T0 / T1 plane pixels | same camera ray (48–70 k pairs per view) |

**Amendments.**
1. Committed before any result. Probe A releases the converged state before building the first-event state; the first
   attempt hit a GPU OOM before printing anything.
2. Committed after the probe results and labelled post hoc. P1 measures the reference-semantics energy of the Burley
   oracle; P2 repeats Probe B's K44 comparison with the hemispherical mean in place of the normal. Amendment 2 also
   records three defects:
   - Probe A's `F_cosmean_over_normal_median` is invalid (an [n] / [n, 1] broadcast); `F_pooled` is valid, and no
     criterion uses F.
   - Probe C's `basis_table_s` timing key is overwritten by later views.
   - Probe B's blue 10–30 cm bins divide by a zero reference response (~1e28): undefined.

**Not done:**
- K training or tuning;
- rank / support changes;
- any production change;
- glass transmission;
- new renders, including a known-field MS control (not affordable without new references);
- a direct Cycles measurement of Yellow total exitance.

---

## MEASUREMENT

### Probe A — directional K validity

Views 6 / 15. State: converged production state for distributions, worklog-45 first-event field for energy.

**Coverage.** Ports observed in training are 33.9 % of Yellow ports (35.7 % of the area) and 31.4 % of opaque ports.

**Group statistics** (Yellow nonlocal = glob + band + mid; view 6 / view 15).

| Yellow group | AI median (19 bins) | C-request / normal (median) | C-request / cos-mean (pooled) | dev supervised / unsupervised (C-req) | cos-mean level | nonlocal share of flux |
|---|---|---|---|---|---|---|
| observed lit | 5.46 / 6.83 | 5.2 / 6.6 | 0.995 / 0.996 | 0.029 / 0.028; 0.030 / 0.027 | 0.67 / 0.53 | 0.85 / 0.81 |
| observed shadow | 5.90 / 6.49 | 5.2 / 5.6 | 1.001 / 0.998 | 0.036 / 0.046; 0.040 / 0.048 | 0.57 / 0.38 | 0.99 / 0.98 |
| never lit | 7.38 / 9.92 | 6.8 / 9.2 | 1.004 / 1.003 | — | 0.71 / 0.55 | 0.91 / 0.89 |
| never shadow | 9.58 / 10.1 | 6.5 / 6.8 | 1.009 / 1.008 | — | 0.65 / 0.36 | 0.99 / 0.99 |

- Opaque groups: AI 1.00, nonlocal share 0.3–7 %, request / normal 1.00. Their nonlocal output is near zero, as
  expected.
- Every query is finite.
- Extremes: Yellow AI max is up to 1.5e5, and the q99 of request / normal is 47–1817. These come from a small tail of
  ports.
- dev is the median |log(L(dir) / L_cosmean)|. Camera directions also give 0.029–0.042.

**Energy on the first-event field** (outgoing flux / absorbed Lego sun power, RGB).

| | view 6 | view 15 |
|---|---|---|
| K44 hemispherical, all parts | 4.30 / 1.64 / 0.20 | 4.57 / 1.83 / 0.18 |
| … nonlocal part | 4.08 / 1.41 / 0.06 | 4.41 / 1.63 / 0.07 |
| … from Yellow ports | 4.12 / 1.47 / 0.07 | 4.45 / 1.71 / 0.08 |
| K44 production (normal semantics) | 1.96 / 0.58 / 0.15 | 1.96 / 0.61 / 0.15 |
| **Post hoc P1:** Burley oracle with WL45 scales s_c (Yellow exitance) | **4.81 / 1.32 / 0.05** | **4.78 / 1.48 / 0.06** |

**Predeclared criteria.**

| Criterion | view 6 | view 15 |
|---|---|---|
| output_invalid | **true** | **true** |
| port_supervision_association (never / observed AI ≥ 1.5) | false (8.48 / 5.68 = 1.49) | true (10.02 / 6.66 = 1.50) |
| direction_supervision_association (≥ 1.5) | false (1.15) | false (1.07) |

The rule requires both views, so classification is **"representation-or-other"**: the output is invalid, and neither
association holds in both views.

### Probe B — nonlocal spatial capacity

Oracle: Yellow Burley, 2000 receivers × 852,358 object-scope sources, 16 held-out view sun fields. The table gives the
shadow-receiver field error (relative L2), R / G / B, with the HS error for R.

| Condition | R | G | B | R HS |
|---|---|---|---|---|
| S5 (exact < 5 cm) | 0.861 | 0.352 | 0.020 | 0.486 |
| S30 (exact < 30 cm) | **0.233** | 0.000 | 0.000 | 0.055 |
| Q (K44 band 96 reps + mid 128 reps, ideal weights) | **0.353** | **0.258** | **0.253** | 2.38 |
| Q + G64 (rank-64 of the residual, HS-optimal) | **0.339** | 0.252 | 0.246 | 2.08 |
| S30 + G64 | **0.008** | 0.000 | 0.000 | 0.021 |
| R64 (rank only) | 0.209 | 0.899 | 0.990 | 0.670 |
| R256 | 0.109 | 0.681 | 0.909 | 0.470 |

- Red, by distance of the receiver to the nearest lit port (0–1 / 1–3 / 3–10 / 10–30 cm):
  - S30: 0.14 / 0.18 / 0.29 / 0.57;
  - Q + G64: 0.24 / 0.31 / 0.41 / 0.56;
  - S30 + G64: 0.006 / 0.007 / 0.009 / 0.018.
- Shadow response share by distance (R): 17 / 39 / 37 / 7 %.
- Support tables: Σ nbrA median 0.076 m², Σ midA median 7.36 m²; 96 / 128 valid entries.
- Storage per receiver (floats): band 5472, mid 384, global 384, versus a dense object-scope row of 852,358.

**K44 vs s_c-scaled oracle, shadow receivers.**

| Readout | R | G | B |
|---|---|---|---|
| Predeclared, normal: error / ratio / corr | 0.712 / 0.38 / 0.20 | 0.827 / 0.33 / 0.47 | 0.763 / 0.74 / 0.57 |
| **Post hoc P2, hemispherical mean:** error / ratio / corr | **0.517 / 0.87 / 0.57** | **0.535 / 1.21 / 0.80** | 0.824 / 1.70 / 0.65 |
| Lit receivers, hemispherical ratio | 0.89 | 0.82 | 0.73 |

**Predeclared criteria** (red): support_loss **true** (0.233), aggregation_loss **true** (0.353 − 0.233 = 0.12),
rank_binding **true** (0.339), fitting_accounts **true** (normal: 0.712 ≥ 2 × 0.339). With the post-hoc hemispherical
mean, 0.517 < 0.678, so fitting_accounts would not hold. Both are reported.

### Probe C — directional C reads and edit attribution

MS scene, views 0 / 1 / 2, T0 / T1.

| Image | C0 PSNR / lum | C1 (basis) PSNR / lum | C2 (exact) PSNR / lum | C2b PSNR / lum | Yellow share of plane / ref: C0 → C2 |
|---|---|---|---|---|---|
| T0 v0 | 20.61 / 0.51 | 26.98 / 1.12 | 26.78 / 1.15 | 26.92 / 1.12 | 0.22 → 0.86 |
| T0 v1 | 21.01 / 0.46 | 26.91 / 1.25 | 26.74 / 1.26 | 26.85 / 1.26 | 0.23 → 1.04 |
| T0 v2 | 22.27 / 0.46 | 26.21 / 1.27 | 25.99 / 1.30 | 26.16 / 1.28 | 0.27 → 1.10 |
| T1 v0 | 20.60 / 0.47 | 26.86 / 1.12 | 26.65 / 1.16 | 26.80 / 1.13 | 0.23 → 0.92 |
| T1 v1 | 21.01 / 0.42 | 26.82 / 1.26 | 26.64 / 1.27 | 26.76 / 1.26 | 0.24 → 1.08 |
| T1 v2 | 22.29 / 0.45 | 26.14 / 1.28 | 25.91 / 1.30 | 26.09 / 1.28 | 0.27 → 1.12 |

- Opaque share: 0.16–0.27, unchanged across conditions. Box share 0.001–0.038.
- Baseline: C0 recomputed equals the stored worklog-46 R6a to 2e-7.

**Edit, plane pixels** (corr / slope / changed-pixel relative error; σ = view-0 replicate 0.0047).

| View | C0 | C1 | C2 | C2b |
|---|---|---|---|---|
| v0 | 0.951 / 1.02 / 0.19 | 0.818 / 1.01 / 0.20 | 0.812 / 1.01 / 0.20 | 0.817 / 1.01 / 0.20 |
| v1 | 0.927 / 1.10 / 0.29 | 0.700 / 1.10 / 0.32 | 0.698 / 1.10 / 0.32 | 0.698 / 1.10 / 0.32 |
| v2 | 0.682 / 0.93 / 0.39 | 0.361 / 0.99 / 0.48 | 0.355 / 1.00 / 0.49 | 0.359 / 0.99 / 0.48 |

**Per ray** (same state; plane IS rays hitting K, 6 images).

| Hit class | Basis / exact (pooled) | Rel L2 | Normal / exact | Exact q99 / median |
|---|---|---|---|---|
| Yellow | 0.97–0.99 | 0.11–0.18 | **0.19–0.20** | 2.8–3.2 |
| Opaque | 0.99–1.00 | 0.02–0.06 | 0.87–0.94 | — |
| Glass | 0.96–1.03 | 0.05–0.17 | 0.46–0.63 | — |
| Box | 0.99–1.03 | 0.04–0.07 | 0.94–1.16 | — |

**Predeclared criteria.**
- basis_major_overbright: **false**. |C1 − 1| − |C2 − 1| = −0.01 to −0.03; exact reads are slightly brighter.
- exact_improves_static: **false** (−0.18 to −0.23 dB).
- exact_improves_edit: **false** (Δcorr −0.001 to −0.005).

**Cost** (RTX 3080 Ti).

| Item | Time |
|---|---|
| C0r solve | 8.6–8.7 s |
| C2 solve (exact plane reads) | 10.6 s (worklog 46 R6br basis-read solve: 191 s) |
| Pixel reads, iso | 0.27–0.37 s per view |
| Pixel reads, exact | 4.4–6.2 s per view |
| Basis table | not recorded (bookkeeping defect; worklog 46: 28 s per table) |
| Peak memory | 9.9 GB |

### Probe C secondary — known-field transfer decomposition

Clean pixels; the "all" pixel set agrees within 0.01. Columns are pooled ratio / median |rel| / corr.

| κ | K1 (port gathers, port field = WL46 R2) | K2 (exact hit field) | K3 (pixel origin, 128) | K4 (pixel origin, 2048) | Oracle replicate |
|---|---|---|---|---|---|
| 0 | 1.041 / 0.074 / 0.950 | 1.042 / 0.075 / 0.950 | 1.143 / 0.037 / 0.814 | 1.143 / 0.013 / 0.815 | — |
| 2 | 1.027 / 0.091 / 0.939 | 1.040 / 0.088 / 0.940 | 1.122 / 0.057 / 0.803 | 1.122 / 0.020 / 0.806 | — |
| 8 | 1.022 / 0.139 / 0.912 | 1.035 / 0.131 / 0.917 | 1.127 / 0.099 / 0.794 | 1.128 / **0.034** / 0.802 | 1.000 / **0.032** / 0.996 |

- K1 reproduces worklog 46 R2 (κ 8: 1.019 / 0.140 / 0.903 on "all").
- Predeclared contributors at κ 8:
  - K1 → K2: **no** (Δcorr +0.005, Δmed −0.008);
  - K2 → K3: **yes** (Δmed −0.033, with Δcorr −0.12 to −0.14);
  - K3 → K4: **yes** (Δmed −0.065, Δcorr +0.008).

### Resources

| Item | Amount |
|---|---|
| GPU (RTX 3080 Ti) | ≈ 874 s ≈ 14.6 min in total: tests 110 s, failed Probe A attempt 31 s, Probe A 97 s, B 54 s, C 368 s, KF (Mitsuba CUDA) 105 s, post hoc 109 s |
| CPU rendering | 0 (no new render) |
| Experimental wall time | ≈ 23 min of execution, in two windows: 06:41–06:56 UTC (tests, failed attempt, chain) and 07:33–07:40 UTC (post hoc, panels). Between them the session was interrupted and nothing ran. |

All of this is within the 30 / 30 / 90 min limits.

---

## OBSERVATION

- **Probe A** (`results/evaluation/48/01_probeA_energy_anisotropy.png`).
  - K44's Yellow nonlocal output is flat across camera, C-requested and cosine directions (within 3–5 %). The near-normal
    direction is the exception: its value is 5–9× lower, which produces the large bin anisotropy.
  - Never-observed ports are somewhat more anisotropic (AI 7.4–10.1 vs 5.5–6.8). Their output level is the same as
    observed ports.
  - Shadowed ports emit at about the same level as lit ports in the hemispherical mean (0.36–0.65 vs 0.53–0.71).
  - The opaque control is well behaved.
- **Probe B** (`02_probeB_capacity.png`).
  - Error from the representative aggregation grows with distance from the light: red 0.24 → 0.56 for 1 → 30 cm.
  - The beyond-30 cm tail is almost exactly rank-64.
  - Rank alone is poor for G / B (0.90 / 0.99 at rank 64) because their response is short-range and full-rank.
- **Probe C** (`03_probeC_kf.png`).
  - Basis and exact reads are visually and numerically interchangeable.
  - The 4× jump in the Yellow reflected contribution between normal-direction (C0) and off-normal reads (C1 / C2)
    accounts for both the brightness gain and the brightness excess.
  - Edit correlation drops identically for C1, C2 and C2b. View 2 changes little (reference change RMS 0.0084), so its
    correlation is the most sensitive.
- **KF.**
  - Port-origin arms keep corr 0.91–0.95.
  - Pixel-origin arms lower the typical error to the oracle noise (κ 8, 2048 rays) but add a bias and outlier population
    (pooled +12–14 %, corr 0.80–0.82). This population is present at every κ and is unaffected by the ray count.
- **Failures and defects:**
  - Probe A F-median metric (invalid);
  - Probe C basis-timing key;
  - Probe B blue far bins (undefined);
  - two Probe A OOM attempts and one post-hoc OOM, all before any output.

## INTERPRETATION

**Probe A.**
- **Direct evidence.**
  - K44's Yellow nonlocal sender is invalid in two separable ways:
    1. a direction artefact concentrated at the port normal: Cycles' Yellow exit is isotropic to ≤ 0.8 %, while K44 is
       5–9× low at the normal;
    2. an energy level above physical passivity (R 4.3–4.6×).
  - The supervision associations are absent (direction) or borderline (port).
- **Reasonable inference.**
  - The normal artefact matters: production reads Yellow at the normal, which is exactly the worklog-45 "0.35–0.41×
    visible sender" and the worklog-46 C0 under-brightness.
  - The energy level is not a K44 fitting defect relative to the reference. P1 (post hoc) implies the reference semantics
    carry ≈ 4.8× in red: Cycles' object-scoped Burley evaluated on the Lego's thin, multi-layer geometry with the scale
    that reproduces the references. A K that is both passive and faithful to this reference is then not possible for
    Yellow.
- **Confounds.**
  - P1 is an oracle extrapolation (the scale is fitted on visible pixels) and not a direct Cycles exitance measurement.
  - The normal artefact's origin (band / mid w_o conditioning) was not localised further.
- **Unsupported:** that hidden-direction supervision is the dominant cause of K44's directional failure.

**Probe B.**
- **Direct evidence.**
  - With ideal weights, K44's band / mid representative aggregation leaves ≥ 0.25 shadow error in every channel.
  - The 30 cm support plus a rank-64 global covers the far field (0.8 %).
- **Inference.** The capacity limit lies in the in-support quadrature (1.5 cm aggregates and 96 representatives for a
  1 / r kernel with d_B 0.39 cm, and 8 cm cells for 5–30 cm). It does not lie in the support radius or the global rank.
- **Confounds.**
  - The rank-64 term is HS-optimal; HS is dominated by near-field quadrature error, so Q + G64 is an upper bound on the
    structural minimum, not the minimum.
  - "Ideal weights" are cell-average kernels, not a least-squares optimum.
  - Self-port transport is excluded (it belongs to K_loc).
  - The oracle is a fitted Burley approximation, not Cycles.
- **K44 vs bound.** In the hemispherical-mean readout, K44's red error (0.52) is 1.5× the structural bound (0.34);
  roughly 0.39 in quadrature lies beyond structure. The predeclared normal-readout rule (TRUE) is confounded by the Probe A
  normal artefact.

**Probe C.**
- **Direct evidence.** The 19-bin basis is not the cause of the over-brightness, the PSNR level or the edit degradation:
  C1 ≈ C2 ≈ C2b throughout. Basis vs exact per ray is 0.97–1.03 pooled.
- **Inference.**
  - Both effects follow from reading K44's Yellow sender off-normal, where it is 5× its normal value. This raises Yellow's
    plane contribution from 0.22–0.27 to 0.86–1.12 of the reference.
  - The same Yellow sender carries K44's weak spatial fidelity (shadow corr 0.57) and worklog 45's weak Yellow change
    response (slope 0.17).
- **Confounds.** No known-field MS control exists, so the reference's Yellow share is unknown, and IS / analytic-plane
  semantics are not separated from K.
- **Unsupported:** that exact queries would fix the edit tracking.

**KF.**
- **Direct evidence.** At a fixed 128-ray budget, the typical κ 8 residual is dominated by angular quadrature. Receiver
  position (port vs pixel) is second, and sender port aggregation is negligible.
- **Confound.** The pixel-origin arms introduce an unexplained bias / outlier population (+12–14 %, corr −0.12), so their
  corr / pooled values are not improvements. This is consistent with worklog 46 axis D, now quantified for a known field.

## UNRESOLVED QUESTION (consequential for the next decision)

1. Is the Cycles Yellow first-event exitance on the Lego really ≈ 4.8× the absorbed power in red?
   - P1 is an oracle extrapolation; a direct renderer measurement (total Yellow exitance under one sun) was not made.
   - The answer decides whether "passive K" and "reference-faithful K" are compatible for Yellow.
2. Where does K44's normal-direction deficit come from?
   - Candidates are the band / mid w_o inputs, never supervised at the normal, or something else. It is not localised.
3. What share of the MS over-brightness and edit degradation is K-side?
   - Answering this needs a known-field MS reference (new renders).
4. How much of Probe B's in-support quadrature bound could a least-squares-optimal (field-weighted) structure recover?
   - The ideal-weight and HS-rank bounds are upper bounds.
5. Where does the KF pixel-origin outlier population come from?

## ARCHITECTURE VERDICT

| Item | Status | Evidence scope | New vs worklog 47 |
|---|---|---|---|
| A. Directional K validity | **NOT SUPPORTED** for Yellow (5–9× normal-direction deficit vs isotropic Cycles exit); **SUPPORTED** for the opaque control | frozen K44, 2 views, 8 port groups | new |
| B. Hidden sender supervision attribution | **NOT SUPPORTED** as the dominant cause (direction association 1.07–1.15×; port association 1.49 / 1.50, rule not met; equal output levels) | same | new. Worklog 47's surrogate result (hidden outputs need supervision) is unchanged |
| C. Physical / numerical passivity | Physical: **NOT SUPPORTED** (K44 R 4.3–4.6×). Reference fidelity of that energy: **CONDITIONALLY SUPPORTED** (P1, post hoc: oracle-implied reference 4.8×). Numerical: production converges, directional in-solve diverges (worklog 46) | first-event field; P1 is an oracle extrapolation | new |
| D. Spatial support adequacy | Support radius: **CONDITIONALLY SUPPORTED** (binding alone at 0.23 R, not binding with a rank-64 tail: 0.008). In-support aggregation / representatives: **NOT SUPPORTED** (≥ 0.25 shadow error with ideal weights) | Burley oracle, 2000 receivers, 16 fields | new |
| E. Low-rank approximation adequacy | **CONDITIONALLY SUPPORTED** as a far-field term; **NOT SUPPORTED** standalone (R64: 0.21 / 0.90 / 0.99) | same; HS-optimal | new (worklog 45 gave energy only) |
| F. Directional C query accuracy | **SUPPORTED**: the 19-bin basis ≈ exact (−0.2 dB, Δcorr ≤ 0.005); sender aggregation negligible in KF | MS 6 images; KF 40 k px | new |
| G. Sampling adequacy | **NOT SUPPORTED** for 128 cosine gathers at κ 8 (quadrature dominates; 2048 reaches oracle noise in median) | KF known field | new quantification of worklog 46 D |
| H. Geometry-edit response attribution | **INCONCLUSIVE**: the C-side read approximation is excluded; a K-side Yellow sender cause is inferred, not isolated | MS 3 views | new |
| I. Computational cost viability | **CONDITIONALLY SUPPORTED**: exact sparse reads cost less than per-port bases (solve 10.6 s vs 191 s); pixel reads 15–20× iso; cost is driven by query volume | MS on RTX 3080 Ti | new |

No FNTO viability claim follows from these probes, and no architecture is selected.

**Questions with sufficient evidence for a human decision:**
- basis vs exact directional reads, which are equivalent in accuracy, with exact reads cheaper for sparse readers;
- that 128 cosine gathers do not suffice for κ ≥ 8 or glossy receivers;
- that Yellow incoming angular information is unnecessary (worklog 47);
- that K44's band / mid representative aggregation is a measurable Yellow capacity bound, while support + rank-64 tail
  is not;
- that the reference-semantics energy of Yellow exceeds passivity (as an oracle extrapolation), which puts physical
  passivity and reference fidelity in tension.

**Questions that need a larger / production-level test or a new representation class:**
- a direct Cycles exitance measurement (unresolved question 1);
- a known-field MS reference (question 3);
- a K representation with finer in-support quadrature or an LS-optimal structure, which is a new representation class
  and a human decision;
- whether a reference-faithful Yellow K can be stable in the coupled solve.

## REPRODUCIBILITY

- **Commits:**
  - 7ddc874 — protocol, code and tests, before evidence
  - ba32794 — amendment 1
  - bb12fb0 — amendment 2 (post hoc)
  - a post-hoc memory fix (no logic change)
  - this worklog's commit
- **Frozen baselines:** K_k44P sha256 `30d55da373a01ab8…` (= worklog-45 freeze); worklog-47 source commit 53f8d31;
  worklog-46 MS / KF data in `results/fnto_diag46/`.
- **Environment:** LabServer63.
  - RNA venv: torch 2.8.0+cu128, bpy 3.5.1 (unused).
  - Mitsuba 3.5.1 (8dna26 venv, `mi_env.sh`).
  - RTX 3080 Ti, 32 CPU threads.
- **Seeds:**

  | Item | Seed |
  |---|---|
  | Probe A groups | 4801 |
  | Probe B receivers | 4802 |
  | Probe C ray sample | 4803 |
  | KF pixel rays | 4804 (pixel selection 4602 = worklog 46) |
  | P1 subsample | 4805 |
- **Budgets:**
  - Probe A: ≤ 1500 ports per cell, ≤ 64 camera / 16 request directions per port, 200k-port energy subsample.
  - Probe B: 2000 × 852,358 operator.
  - KF: 128 / 2048 rays.
- **Commands:**
  - `~/rest_fnto_r1/d48_chain.sh` (probe_a48, probe_b48, probe_c48, kf48)
  - `diag48_posthoc.py <a1a_fit> <a3_oracle> <out>`
  - `panels48.py results/fnto_diag48 results/evaluation/48`
- **Outputs:**
  - `results/fnto_diag48/` — probe_a, probe_b, probe_c (+ pred npz, kept on the server and PC, not committed), kf,
    posthoc, tests, runtime.json
  - `results/evaluation/48/` — 3 numeric panels, JSON copies and a manifest; no image difference maps

No later batch or worklog 49 is started.
