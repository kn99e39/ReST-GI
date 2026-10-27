# Worklog — FNTO Directional K/C Interface: Component Attribution, Controlled Composition and Real-scene Validation (2026-10-26 / 27)

Protocol `fnto_dir46` + amendments 1–2 (`experiments/fnto_dir46/protocol/dir46.json`). Authorisation: the user brief
"Worklog 46 — Directional K/C Interface" (2026-10-26). Frozen throughout: K_k44P parameters (digest identical after every
run), the four-component K, relation tables, sun fractions, references and every historical baseline and worklog.

**Answer to the completion question.**

- **Preserving directional boundary transport is correct as an interface.**
  - In the synthetic contracts the directional sender is exact. With lobe-importance sampling, one- and two-bounce
    glossy transport is within 1 % of the truth.
  - On a known field rendered by Blender, it removes the 2–6× bias of the normal-direction sender.
- **The interface does produce a real-scene specular gain.** In a Cycles glossy-plane scene the glossy-reflection ROI
  improves from 18.25 to 26.65 dB (+8.4 dB, 6 / 6 images). The gain has two parts:
  - directional sender reads: +2.6 dB;
  - lobe sampling at the glossy receiver: +3.1 dB;
  - their interaction: +2.75 dB (synergy).
- **The frozen learned K44 cannot be the directional sender inside the solve.** Its nonlocal outputs at unsupervised
  directions are non-passive (ρ 9.4 / 17.1, solves diverge). One-step, they push the Yellow indirect from 1.41× to
  4.07× the reference.
- **The incoming axis is negligible for K44 on this scene.** Its only directional-incoming component (K_loc gloss,
  κ ≈ 2 %) gives +0.002 dB.

---

## IMPLEMENTATION FACT

**What changed.** A directional interface on top of the unchanged r1 / k44 machinery
(`experiments/fnto_dir46/dir46_core.py`, `dir46_solve.py`). Its switches are listed below.

| Switch | Values |
|---|---|
| Sender nonlocal part (glob + band + mid) | `n` = production, evaluated at the port normal; `basis` = per-port 19-bin piecewise-constant angular basis (the K_band incident partition) read at the bin of −d; `exact` = per-ray query |
| Sender K_loc incoming | `E` = production collapse ρ/π E_tot + Σκ/π E_ind; `dir` = the learned BSDF on the port's 128 directional gather samples |
| Sender K_loc outgoing direction | normal (R4) or the ray direction (R5) |

- The sun's GGX lobe stays the production per-ray term in every mode, so it is counted once (test T10).
- Every query is linear in the incident field.
- The disabled path reproduces production: one step agrees to 1.7e-4 (fp16 storage), and a converged view agrees with
  the frozen eval to 1e-5 dB.

**What K44 actually supports (stage 1 audit, smoke on view 6).**

| Component | Directional capability |
|---|---|
| K_loc | Full learned BSDF f(w_i, w_o) (κ median 0.019, α 0.09–0.13 on opaque) |
| K_band / K_mid | Take w_o as an input, but were supervised only at camera directions of visible pixels |
| K_glob | Isotropic by design |
| Production sender | Evaluates band / mid at the port normal |

Worklog 45's "visible Yellow ports send 0.41× of the reference" is entirely this normal-direction evaluation:
- the same port tables queried toward the camera give 0.99×;
- the pixel path gives 0.98×.

At unsupervised (port, direction) pairs K44's Yellow nonlocal output is non-physical: the cosine-weighted mean is 46×
the normal value, and max / min over the 19 bin centres has a median of 238. Glass gives 8.5×; the box is sane (0.93).

**Other components.**
- **Synthetic contracts** (`syn46.py`, analytic geometry / BSDF, production ggx / loc_response).
- **Known-field oracle** (`emit46_render.py`, `kf46_eval.py`). Blender emission f_s(x) g_κ(|n·w|) with
  g_κ = (κ+2)/2 c^κ, so the flux equals that of the isotropic field. Mitsuba re-traces hits for the triangle-normal
  diagnostic R2t.
- **Specular scene MS** (`ms46_render.py`, `ms46_prep.py`, `ms46_eval.py`).
  - Geometry: official Lego + red box with the worklog-37 T0 / T1 box move, plus a glossy plane (Cycles Glossy BSDF,
    GGX, roughness 0.2 → α 0.04, colour 0.9) at x = 0.85 facing −x.
  - Light: one sun with no caustic on the plane.
  - Views: 3, at 512².
  - References: 4096 spp, r1 settings, plus a replicate of view 0.
  - The plane is an **analytic control domain** (ρ 0, one GGX lobe). The Lego and box use frozen K44.
- **Lego suite** (`lg46_run.py`, `lg46_onestep.py`, `lg46_metrics.py`).

**Not done.** Not authorised, and not done:
- K training or redesign;
- rank / support / mixture / GGX / passivity / threshold tuning;
- glass transmission;
- learned C;
- Yellow heuristics.

Glass is measured as a limitation only.

**Tests** (`tests/test_dir46.py`, final code): ALL_PASS.

| Test | Result |
|---|---|
| Frames | bin centres in own bin; orthonormality 4e-7 |
| Radiance / irradiance normalisation | E_const = π; diffuse ρc exact |
| Zero field | exactly 0 in every mode |
| Known reflection peak | 0.04° from the mirror direction (α 0.01) |
| Linearity | 2.5–2.9e-4 relative (fp16 storage) |
| Finite outputs | yes |
| Frozen digest | identical |
| Baseline-off identity | see above |
| Sun lobe counted once | 1.7e-4 |

Synthetic S0 / S6 exact.

**Amendments** (each committed before the run it changed).
1. R3 / R5 diverge, so the Lego evidence for them is one-step diagnostics. The MS scene gets a "directional reads only"
   sender (R3mr / R6br): the Lego / box coupling is solved with production semantics and only the reads into the plane
   are directional. The S2 post-hoc diagnostic is added.
2. A bug fix in `ms46_eval`'s residual bookkeeping. Plane-hit rays were counted twice, which put a constant floor under
   the residual and flagged converged solves as diverged. The first MS run therefore produced no metrics; it is kept as
   `ms46_eval_amend1_bookkeeping_bug.json`. The residual history is now stored.

---

## MEASUREMENT (condition matrix)

All PSNRs are display-space (ldr3), as in worklogs 37–45.

| Axis | Condition vs control | Scene / regime | Baseline | Candidate | Signed effect | Uncertainty / validity |
|---|---|---|---|---|---|---|
| A known field | R2 vs R1 | KF emit46, views 6 / 15, 40 k non-Yellow px | R1 pooled 2.17 (κ 2), 5.58 (κ 8) | R2 pooled 1.025 / 1.019 | bias removed | oracle replicate med. 3.2 % (κ 8) |
| C transfer | R2 at κ 0 / 2 / 8 | same | — | pooled 1.042 / 1.025 / 1.019; corr 0.942 / 0.930 / 0.903; med. 0.077 / 0.092 / 0.140 | corr and med. worsen with κ | criterion met at κ 2, **not at κ 8** (corr −0.039, med. +0.063) |
| C diagnostic | R2t (triangle normal) vs R2 | same | R2 | corr 0.932 / 0.908 | +0.002 / +0.005 corr | port normal is not the residual error |
| A synthetic | S3 dir vs iso | analytic | iso +10.7 % (κ 2), +44 % (κ 8) | dir +0.05 % / −0.0004 % | exact | deterministic, 2^16 rays |
| B synthetic | S2 equal E, different angular distribution | analytic, α 0.04 / 0.09 | iso −99 % / −97 % | dir+IS −4.2 % / −8.7 % (32 seeds) | — | **criterion FAIL** (2 %); 2,048-seed diagnostic −0.08 % / −0.12 % (z −0.29 / −0.18): MC noise, no bias |
| D synthetic | S4 one glossy bounce | analytic | iso −97 %; dir+cos128 −16 % | dir+IS −0.4 % | — | 32 seeds |
| D synthetic | S5 two glossy bounces | analytic | iso−cos −88 %; dir−cos −21 % | dir−IS −0.9 % | — | seed CV 0.11 |
| A frozen K44 | R3 (sender basis toward −d) | Lego rel_T0 | ρ 0.527 | **ρ 9.36, view-6 solve diverges** | not production-capable | power iteration |
| A+B frozen K44 | R5 | Lego | ρ 0.527 | **ρ 17.1, diverges** | not production-capable | — |
| A one-step | R3 vs R0, first gather | Lego 16 held-out T0 views | 26.39 dB | 23.09 dB | −3.30 dB, CI [−3.60, −3.00], 0 / 16 | Yellow 24.58 → 20.06; Yellow 1-bounce indirect 1.41× → 4.07×; box 1.00 → 1.41×; opaque 0.68 → 0.99× (corr 0.85 unchanged) |
| A basis error | R3 basis vs exact per ray | view 6, one step | exact | rel. L2 18 %, lum 0.965 | — | one view |
| B frozen K44 | R4 vs R0 (full suite) | Lego 16 T0 + 16 T1 (×2 arms), r1 | T0 25.821 | 25.823 | **+0.002 dB**, CI [+0.001, +0.003], 14 / 16 | all 96 solves converge; Yellow +0.003, opaque −0.0005, box −0.006 |
| D real scene | R6a vs R0m (lobe IS) | MS plane ROI, 6 images | 18.25 | 21.30 | **+3.05 dB**, 6 / 6 | replicate floor 32.9 dB |
| A real scene | R3mr vs R0m (directional reads) | MS plane ROI | 18.25 | 20.85 | **+2.60 dB**, 6 / 6 | lum ratio 0.42–0.51 → 1.08–1.26 |
| A+D real scene | R6br vs R0m | MS plane ROI | 18.25 | 26.65 | **+8.41 dB**, 6 / 6 | lum ratio 1.12–1.28 (over-bright) |
| A×D | interaction R6br − R3mr − R6a + R0m | MS plane ROI | — | — | **+2.75 dB synergy** | 6 images |
| A full | R3m / R6b (K44 directional inside the solve) | MS | — | **diverged** (NaN at iteration 6) | not production-capable | residual history stored |
| Regression | R6br vs R0m | MS Lego / Yellow pixels | 23.57 / 20.86 | 23.54 / 20.84 | −0.02 dB | 6 images |
| Edit, plane | T0 → T1 change on plane pixels | MS, 3 views | R0m corr 0.63 / 0.50 / 0.50 | R6a 0.95 / 0.93 / 0.68; R6br 0.82 / 0.70 / 0.36 | IS helps, directional reads reduce tracking vs R6a | slopes R6a 1.02 / 1.10 / 0.93, R6br 1.01 / 1.10 / 0.99 |
| Edit, Lego | R4 vs R0 | Lego A_changed / stationary | total corr 0.984, slope 0.961 | identical (±0.001) | none | — |

**Cost** (LabServer63, RTX 3080 Ti; equal ray budget unless stated).

| Item | R0 production | Directional |
|---|---|---|
| Sender per iteration | outgoing() 0.79 s | basis table 28 s per iteration (19 × N nonlocal queries at 0.62 M/s, ≈ 35×) |
| Per-ray directional K_loc | — | 45 s per iteration (1.4 M queries/s) |
| Lego solve | 7.6 s | R4 8.7 s |
| MS solve | iso 8.5 s | directional reads 191 s (≈ 22×) |
| Memory | — | +195 MB basis, +109 MB per-ray bin index; peak 9.8 GB (unchanged) |
| Plane-pixel IS | — | 128 extra rays per glossy pixel (≈ 6 M rays per view), traced once in prep |

The relation update for the T0 → T1 edit is unchanged (worklog 42: 0.63 % of rays).

---

## OBSERVATION

- **MS plane ROI** (`results/evaluation/46/01_MS_*`).
  - R0m is blocky: the plane port's 128 cosine gathers at 2.5 cm port resolution. It is also half as bright as the
    reference (lum 0.42–0.51), because the iso / normal sender under-reports what the Lego sends toward the plane.
  - R6a is smooth but colour-washed and dark.
  - R3mr restores brightness but stays blocky.
  - R6br reproduces the reflected Lego and box. It is 12–28 % too bright, and the residual error sits where the
    reflection shows Yellow surfaces.
- **MS edit maps** (`02_MS_edit_*`): IS-based conditions follow the box's reflected motion. R6br adds spatially varying
  errors that change with the edit (corr 0.36 on view 2).
- **Lego**: R4 is visually identical to R0 (`04_LG_view6.png`; the R4 − R0 map is ≈ 0). R3 was never a converged image;
  its one-step brightening is concentrated on Yellow.
- **KF scatter** (`03_KF_scatter.png`): R1 scales with g_κ(1) (2× / 5×). R2 lies on the diagonal with a widening spread
  at κ 8.
- **Representative failures**:
  - K44 directional senders inside the solve diverge (Lego and MS);
  - the S2 / κ-8 predeclared criteria were missed;
  - R6br over-brightness.

## INTERPRETATION

- **Angular information loss: confirmed and separable.** The normal-direction sender and the isotropic collapse lose the
  directional information:
  - known field: 2.2× / 5.6× bias;
  - synthetic: −88 % to −99 %;
  - MS: half-bright reflections.

  Restoring it is correct on known fields (A, C).
- **Sampling insufficiency: confirmed.** 128 cosine gathers cannot integrate an α 0.04 lobe (S4 −16 %, S5 −21 %; MS
  +3.1 dB from IS alone). It is separable from the representation axis, and the two are synergistic (+2.75 dB).
- **Intrinsic K representation deficiency: the binding limit for the learned candidate.** K44's band / mid outputs are
  meaningful only at supervised camera directions; elsewhere they are non-passive (ρ 9.4). As a result, the directional
  interface cannot be used with K44 inside the coupled solve. This is the same unsupervised-sender issue as worklog 45,
  now along the angular axis: it is amplified, not repaired. Yellow is hit hardest (1-bounce indirect 4.07×).
- **Hidden sender inaccuracies: interact with A.** At visible ports the directional query is correct (0.99×), but the
  aggregate effect is dominated by non-visible ports and directions.
- **C transfer deficiency: small at moderate angular frequency.** It grows with κ (corr 0.94 → 0.90). It is not due to
  the port normal (R2t ≈ R2). It is consistent with 128-ray quadrature and port position. Enclosed receivers (0.64)
  and near-hit receivers (1.03–1.12) remain contract limits.
- **Incoming axis: real in principle, negligible for K44 here.** The analytic control shows it in principle (S2, S4,
  S5). For K44 on Lego it is negligible because the only consumer is the 2 % gloss lobe. Whether a K with stronger
  directional entry would benefit cannot be answered with K44.
- **Reference / supervision confounds.**
  - The MS plane is an analytic domain, so its exact BSDF helps the MS gain. The gain therefore measures interface plus
    sampling, not learned glossy K.
  - The MS reference floor is 32.9 dB; R6br at 26.7 dB is well above it.
  - In the edit comparison only view 0 has a replicate σ.

## UNRESOLVED QUESTION

- Whether a K that is passive at all (port, direction) pairs it is queried at would make the in-solve directional
  sender stable and beneficial. With frozen K44 this cannot be measured: no converged R3 / R5 image exists.
- Why R6br is 12–28 % too bright on the plane. The two candidates are K44's directional over-response toward the plane
  and the 19-bin basis error (18 % rel. L2 vs exact on view 6). An exact-query MS run was not made.
- Why directional reads reduce edit tracking on the plane relative to IS alone (corr 0.95 → 0.82, 0.93 → 0.70,
  0.68 → 0.36).
- The A × B interaction on Lego: unresolved. The one-step diagnostic cannot see B (no indirect incident field at the
  first gather), and R3 / R5 do not converge.
- κ-8-level angular frequency: whether the C residual is quadrature or port position (no budget sweep was run;
  R6-style sampling was applied only to the analytic receiver).

## ARCHITECTURE VERDICT

| Item | Status | Basis |
|---|---|---|
| A. Outgoing directional representation | **CONDITIONALLY SUPPORTED** | Exact on synthetic and known fields; +2.6 dB real-scene reads; unusable with K44 inside the solve |
| B. Incoming directional representation | **CONDITIONALLY SUPPORTED** (analytic control) / **not measurable for K44** (+0.002 dB) | S2 criterion missed by MC noise only |
| C. Directional free-space transfer | **CONDITIONALLY SUPPORTED** | Bias-free at κ ≤ 8; predeclared criterion met at κ 2, not κ 8 |
| D. Sampling adequacy of 128 cosine gathers | **NOT SUPPORTED** for glossy α 0.04 | Lobe IS required (+3.1 dB alone, synergistic with A) |
| E. End-to-end composition | **NOT SUPPORTED** with frozen K44 as in-solve sender (diverges); **CONDITIONALLY SUPPORTED** as a hybrid | Production solve + directional reads + IS into an analytic glossy domain: +8.4 dB ROI, +4.35 dB whole image, Lego unchanged |
| F. Existing Yellow and opaque regression | R4: no regression; R3 one-step: **regression** | Yellow −4.5 dB, indirect 4.07× |
| G. Geometry-edit response | **CONDITIONALLY SUPPORTED** | Lego unchanged; plane change tracking improved by IS, degraded by directional K44 reads relative to IS alone |
| H. Computational viability | **CONDITIONALLY SUPPORTED** | Bounded and query-based (no dense tensor), but the 19-bin nonlocal basis costs ≈ 35× per iteration and the MS directional solve ≈ 22×; not yet compatible with the interactive update target without compression |

**Overall.**
- **Retained as a viable FNTO interface direction, conditionally.** The directional K/C interface is correct and gives
  a large, attributable specular gain when the sender is passive and correct in the queried directions and the glossy
  receiver samples its lobe.
- **Not retained with the frozen K44 as the directional sender.** K44 was not supervised for that role and is
  non-passive there.

This is a verdict on the interface, not on FNTO or ReST-GI viability.

## REPRODUCIBILITY

- **Commits:**
  - 12e086a — protocol and stage-1 core, before evidence runs
  - 2be531f — implementation
  - 28ef37d — amendment 1
  - 6f5f3d7 — amendment 2
  - this worklog's commit — metrics / exports
- **Environment:** LabServer63, RTX 3080 Ti, driver 595.71.
  - RNA venv: torch 2.8.0+cu128, bpy 3.5.1 (Cycles CPU).
  - Mitsuba geometry: 8dna26 venv (Mitsuba 3.5.1 cuda_ad_rgb, scipy 1.14.1 installed this batch) with
    `mi_env.sh` (OptiX workaround).
  - GPU steps were serialised by one flock.
- **Seeds:**

  | Item | Seed |
  |---|---|
  | KF renders | 4600 + v (replicate + 100) |
  | KF pixels | 4602 |
  | MS references | 4610 + v (replicate + 100) |
  | MS plane-port rotations | 4647 |
  | MS IS rays | 4646 + 10 v + (T1) |
  | Synthetic | 100 / 200 / 300 / 400 / 500 + seed, diagnostic 10000 + seed |
  | Bootstrap | 46 |
- **Server scripts:** `~/rest_fnto_r1/d46_{kf_chain2,ms_render_chain,gpu1_chain,gpu2_chain,ms_rerun}.sh`.
- **Outputs:** `results/fnto_diag46/{tests,syn,kf,lg,lg_onestep,lg_smoke,ms}/`.
- **Reviewer exports:** `results/evaluation/46/` — 11 panels and the JSONs, with a manifest. One absolute error / diff
  scale is used across all panels: **M = 0.565 display units**, sqrt mapping.
- **Not run:**
  - the full stage-1 audit (3,000 ports, 2 views); only the view-6 smoke is reported;
  - an exact-query MS run;
  - a sampling-budget sweep.

## NEXT DECISION BOUNDARY (decision input only; no worklog 47 is started)

The evidence separates three things a reviewer would decide on.

1. Whether the directional interface becomes part of the FNTO contract. Supported as interface semantics.
2. What K would have to satisfy to serve as a directional sender: passivity and correctness at the (port, direction)
   pairs C queries, beyond camera directions. K44 does not.
3. Whether lobe-adaptive receiver sampling is part of C's contract for glossy receivers. Required by D.

No next architecture or batch is selected here.
