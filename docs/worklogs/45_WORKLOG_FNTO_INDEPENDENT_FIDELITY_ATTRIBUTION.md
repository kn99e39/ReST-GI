# Worklog — FNTO Independent Fidelity Attribution, Parallel Diagnostic Batch (2026-10-26)

**Diagnostic batch, protocol `fnto_diag45` + amendments 1–6. Nothing was trained or corrected.** Two tracks ran on the
frozen K_k44P baseline (commit 63e2741): Track A (intrinsic K / Yellow SSS) and Track B (free-space C / K–C
composition). Headline:
- **Yellow first-event error** is a K-intrinsic representation limit. A physics oracle with no learned parameter, given
  the same verified input, is tied with K overall and better in shadow, 3–30 cm from lit ports.
- **C transports a known field correctly** (pooled 1.05×, corr 0.93 vs an independent Blender oracle).
- **The Yellow indirect over-prediction (1.74×)** does not come from C or from the solver. It comes from what K emits
  into C at Yellow ports that no camera sees. Summed over Yellow, that first-bounce power is 1.8× the whole Lego's sun
  power in red. The receiver's nonlocal response then picks it up.

Authorisation: user brief "FNTO Independent Fidelity Attribution — Parallel Diagnostic Batch" (2026-10-26). Worklogs 41,
42 and 44, every frozen input and every historical baseline are unchanged. Baseline provenance: 18 inputs hashed on PC
and server, all identical (`results/fnto_diag45/baseline_freeze_{pc,server}.json`).

**Regimes:**
- **first_r1**, **full_r1** — as worklogs 42 / 44.
- **emit45** (new) — Blender renders of held-out T0 views in which surfaces emit a known analytic field f(x) for
  non-camera rays and are diffuse white for camera rays. The oracle irradiance is E_in = π (P_emit − P_emitonly).
  Yellow pixels keep their real material, so they show Yellow's response to that field.
- **emit45-emitall** — the same, with Yellow also emitting.
- **emit45-bump** — a localized field 10·exp(−|x−c|²/2σ²), σ = 2 cm, all surfaces emitting.
- **a1a-plane** — a flat Yellow plane with a sharp shadow edge.

All PSNRs are display-space (ldr3) as in worklogs 37–44. Ratios are luminance sums, linear radiance.

---

## IMPLEMENTATION FACT

| Item | Fact |
|---|---|
| Frozen inputs | K_k44P `30d55da3…`, rel_T0 `1da7e922…`, rel_T1_inc `41ed71cf…`, anchors `32aa4032…`, eval_vis_k44 `3735fa69…`, refs, eval views — sha256 PC = server |
| Code | `experiments/fnto_diag45/` (track_a: a1a_plane_render, a1a_fit, a2_localize, a3_burley_oracle, a4_split; track_b: emit45_render, b2_b3_emit_eval, b3b_iter, b3c_passes_render, b3d_interface, b3e_coverage, b3f_bump, b3g_eind_split, b3h_lit_coverage, b4_edit; common: freeze_baseline, d45_tests, d45_exports) |
| Commit order | e924a6f protocol + freeze before any measurement. **B4 and the first B2 / B3a ran on code that was not yet committed**; it was committed in dc5a2b4 together with amendment 1, which records this. Amendments 2–6 (b664832, 9cf728b, 607b933, e6790c2, 2c67f9a) were each committed before the run they introduced |
| Amendments | 1: B2 hit-point NaN fix, oracle-noise replicate, emitall control, source split. 2: K→C interface check. 3: supervision coverage of sender ports. 4: localized-field control. 5: split of the composed indirect by the ports holding the gathered field. 6: training lit coverage. All are descriptive; no threshold was introduced |
| Protocol tests | Emission calibration: pixel = 1.000. emitonly / f = 1.0000 (q01–q99 0.998–1.002). Burley plane integral 1.0000 per channel. C: zero field → 0; constant field = π c × hit fraction (max dev 7e-16). Tests 3–4 ran after A3 / B2 (analytic identities) |
| Not done | No K / C / solver / passivity / visibility / reference change; no training; 0.3 mm rule not propagated into C; no correction implemented |
| Machine | LabServer63 (renders CPU Cycles 3.5.1, K/C GPU serialised — never two GPU jobs at once); A2 / B4 / exports CPU |

### B1 — C contract audit (code + relation table)
Contract, from the code:
- **Gather.** 128 cosine-distributed rays per port, spread over its 4 surface samples, starting 0.1 mm off the surface.
  E = π/128 Σ L(hit).
- **Hit mapping.** A hit maps to a port by surface identity. Misses carry 0. Inside faces of closed solids (−2) carry 0.
- **Carried value.** The hit port's isotropic outgoing radiance I, plus the per-ray sun-GGX specular of the hit port.
- **Sender evaluation.** I is evaluated by port-level tables with the exit direction fixed to the port normal.
- **Glass.** C has no transmission path: glass ports are opaque isotropic senders.

Table statistics (rel_T0):

| Statistic | Value |
|---|---|
| Miss | 8.25 % |
| Enclosed (inside faces) | 4.95 % |
| Hits closer than 0.3 mm | 3.12 % |
| Self-port hits | 0.25 % |

Yellow ports:

| Statistic | Value |
|---|---|
| Gather rays that hit Yellow again | 78.7 % |
| Median hit distance | 1.2 cm |
| Hits within 1 cm | 38 % |
| Hits closer than 0.3 mm | 2.5 % |

Box ports: miss 56 %, median hit distance 16 cm.

---

## TRACK A MEASUREMENT & OBSERVATION (Yellow, intrinsic K)

**A1a — Burley scale (a1a-plane, 4096 spp).**

| Channel | Fitted d | Cycles 3.x formula (0.25/π·radius/(1.9−A+3.5(A−0.8)²)) | Mass < 5 cm | Mass < 30 cm |
|---|---|---|---|---|
| R | 6.1 cm | 7.2 cm | 32 % | 85 % |
| G | 1.2 cm | 0.93 cm | 81 % | 100 % |
| B | 0.39 cm | 0.20 cm | 99 % | 100 % |

Red has 99.7 % of its mass within 1 m. The lit-plateau albedo matches the material colour once the white specular
peak is removed (R−B 0.71 vs colour 0.77; G−B 0.46 vs 0.47), so Cycles' Burley conserves energy on a plane.
**Correction to worklog 44:** "Yellow SSS (Burley, red radius 1 m) … beyond K_band 5 cm / K_mid 30 cm" overstated
the range. Red's mean free path is ~6 cm and 15 % of its mass lies beyond 30 cm.

**A2 — localisation** (K_k44P K-only vs first_r1, all 16 held-out T0 views, 1.61 M Yellow pixels).

Yellow carries **79 %** of the Lego K-only squared error (others: Black 5.7, Sand 5.4, Gray 5.1 %). Inside Yellow:

| Group | Share of Yellow pixels | Share of Yellow SE | PSNR | K/ref |
|---|---|---|---|---|
| Lit | 45 % | 22 % | 28.07 dB | 0.99 |
| Shadow | 55 % | 78 % | 23.45 dB | 0.95 |

Shadow pixels by distance to the nearest sun-lit port:

| Distance | Share of Yellow SE | K/ref | PSNR |
|---|---|---|---|
| < 0.3 cm | 2 % | 0.87 | 25.0 dB |
| 0.3–1 cm | 12 % | 0.90 | 25.6 dB |
| 1–3 cm | 20 % | 0.97 | 24.8 dB |
| 3–10 cm | 31 % | 0.98 | 22.3 dB |
| 10–30 cm | 13 % | 1.36 | 19.6 dB |

Other axes:
- Channels: R 49 %, G 44 %, B 7 % of the SE.
- Views: each view carries 3–9 % of the SE, with no dominant view and no monotone dependence on sun elevation.
- Lit pixels at grazing incidence (n·w < 0.25): K/ref 0.88–0.95.

**A3 — input vs operator** (physics oracle L_c = s_c Σ R_c(|x−x'|) E_sun(x') A(x') over ports).
- E_sun comes from the same verified port field K_k44P uses (eval_vis_k44).
- Three scales s_c were fitted on views 0–7 and scored on views 8–15 vs first_r1, 12,000 Yellow pixels per view.

| Arm (test views) | all | lit | shadow | 95 % CI (all) |
|---|---|---|---|---|
| K_k44P | 24.75 | 27.68 | 23.38 | [24.10, 25.56] |
| Oracle, object-scoped sources | 23.92 | 22.57 | 25.31 | [21.39, 26.28] |
| Oracle, Yellow-only sources | 23.00 | 22.28 | 23.63 | [20.94, 24.72] |
| Oracle, connected component only | 16.13 | 20.13 | 14.50 | [15.41, 16.85] |

**Primary comparison A:** the CIs overlap overall.
- Shadow: the oracle is +1.9 dB.
- Lit: K is +5.1 dB. The oracle has no sub-2 mm term, no Fresnel and no per-pixel normal.

By distance to lit ports:

| Distance | Oracle | K | K/ref | Oracle/ref |
|---|---|---|---|---|
| < 1 cm | 23.4 | 27.0 | | |
| 1–3 cm | 25.5 | 24.6 | | |
| 3–10 cm | 24.1 | 22.1 | | |
| 10–30 cm | 24.2 | 19.9 | 1.28 | 0.78 |

Per channel: K R 22.9 / G 23.7 / B 31.9 dB; oracle R 25.0 / G 25.3 / B 22.2 dB.

**A1b — entry semantics.** The connected-component restriction collapses to 16 dB. Cycles' Burley is object-scoped:
light crosses gaps between separate bricks of the one Lego object, and the reference contains that.

**A4 — representation.**
- Rank-64 energy of the Yellow-receiver × coarse-port Burley operator: R 0.83, G 0.82, B 0.92 (rank 256: 0.98).
- Source distance in the oracle, shadow pixels:

  | Sources | All channels | Red |
  |---|---|---|
  | < 5 cm | 43 % | 18 % |
  | 5–30 cm | 48 % | 63 % |
  | > 30 cm | 9 % | 19 % |

- K's shadow output is 71 % band+mid and 29 % glob; band and mid are stored together and are not separable here.

---

## TRACK B MEASUREMENT & OBSERVATION (C and composition)

**B2 — C single flight vs Blender (emit45; views 6, 15, 2, 10; 80,000 non-Yellow pixels).**

| Group | E_C / E_in (pooled) | median abs. rel. error | corr |
|---|---|---|---|
| all | 1.053 | 0.121 | 0.932 |
| box | 1.005 | 0.048 | 0.990 |
| transparent | 1.002 | 0.053 | 0.960 |
| Sand / Brown / Light_brown | 1.026–1.040 | 0.12–0.15 | 0.945–0.956 |
| Gray / Black | 1.10 | 0.14 | 0.86 / 0.90 |
| Red_Glass | 1.03 | 0.50 | 0.92 |
| receivers with > 5 % near-hit rays | 1.159 | 0.152 | 0.837 |
| receivers with > 10 % inside-face rays | 0.647 | 0.375 | 0.661 |

- Evaluating f at the exact hit point instead of the hit port's origin changes nothing (1.053 vs 1.053), so output
  projection is not an error source.
- The oracle's own Monte-Carlo spread (second seed, view 6) is median |rel| 1.3 %, corr 0.9997. The ~12 % per-pixel
  scatter is therefore C's own: port aggregation over 4 samples, 128-ray quadrature, port vs pixel location.

**B3a — K's Yellow response to C-gathered verified fields (one K/C step, sun off).**

| Field | pred/ref | per channel R / G / B |
|---|---|---|
| Smooth, non-Yellow emitters (views 6, 15, 2, 10) | 0.557 | 0.38 / 0.80 / 1.02 |
| Smooth, all emit (emitall; views 6 / 15) | 0.88 / 0.77 | — |
| Yellow-sourced increment (emitall − emit) | 0.97 / 0.88 | R 0.70–0.78, G ~1.0, B ~1.1 |
| Localized bump (views 6 + 15) | 1.16 overall | see distance table below |

Bump field by distance:

| Distance | pred/ref |
|---|---|
| 0–2 cm | 1.17 |
| 2–5 cm | 1.37 |
| 5–10 cm | 1.32 |
| 10–20 cm | 1.04 |
| 20–40 cm | 0.78 (R 0.62) |

The bump oracle's seed-to-seed difference is ≤ 1.6 %. Under these fields 92–98 % of K's response is band+mid.

**B3b — production solve per iteration** (views 6 / 15). Both converge in 6 iterations with no growth.

| View | Yellow indirect pred/ref (iter 1 → 6) | Full PSNR vs full_r1 (iter 1 → 6) | Opaque | Glass | Box |
|---|---|---|---|---|---|
| 6 | 1.93 → 2.51 | 25.57 → 24.32 dB | 0.93 | 0.04 | 1.69 |
| 15 | 0.76 → 0.96 | 26.25 → 26.50 dB | 0.75 | 0.18 | 0.79 |

Source split of the last gather (closure 1.0000):
- At Yellow pixel ports, 96 % / 88 % of the gathered irradiance comes from rays hitting Yellow ports.
- Yellow's indirect: 2.40 of 2.51 (view 6) and 0.90 of 0.96 (view 15) come from Yellow sources.

**B3c — opaque receivers** (Cycles diffuse-indirect pass, 2048 spp).
- Receiver response is correct: the counterfactual ρ_K E_ref/π vs the reference diffuse indirect is 0.99 / 0.99, corr
  0.9996 / 0.994.
- ρ_K / DiffCol is 0.99 for opaque materials. Black is the exception: 0.27–0.30, with counterfactual corr ≈ 0.1
  (unresolved).
- The converged gathered E at the pixel's port vs π·DiffInd: 0.89 / 0.76, corr 0.72 / 0.64, median |rel| 0.36 / 0.39.

  | Material | View 6 | View 15 |
  |---|---|---|
  | Sand | 1.08 | 0.91 |
  | Gray | 0.65 | 0.80 |
  | Black | 0.51 | 0.45 |
  | box | 1.46 | 0.77 |

**B3d — K→C interface** (first-bounce emission I = outgoing(0, h_sun), views 6 / 15).

| Location | I/ref | Notes |
|---|---|---|
| Visible opaque ports | 0.96 / 0.90 | |
| Visible box ports | 1.00 / 0.99 | |
| Visible Yellow ports | **0.41 / 0.35** | K's own pixel prediction at the same pixels: 0.98 / 0.92 of ref; I/K_pixel 0.41 / 0.38 |

Summed over all Yellow ports, the area-weighted power sent into C (π I A) is:
- 4.24 / 3.35 (R), 1.03 / 0.88 (G), 0.073 / 0.066 (B) × Yellow's own sun power;
- **1.81 / 1.83 (R), 0.44 / 0.48 (G), 0.031 / 0.036 (B) × the whole Lego's sun power.**

The physical bound (Yellow sun power × albedo 0.80 / 0.50 / 0.03) is exceeded about 5× in red and 2× in green and
blue. Of the sent power, 77 % / 76 % comes from shadowed, non-enclosed Yellow ports, which are 87 % of Yellow's area.

**B3e / B3h — coverage of the sender ports.**
- 64 % of Yellow area was never seen by any training-probe pixel (prep2_train), and 89 % of that area was never
  directly lit in any of the 1,000 training images.
- Observed-in-training Yellow ports: 5 % of their area was never lit (median 218 lit images).
- Never-observed ports send 53 % / 50 % of Yellow's power. They carry only 4.4 % / 2.8 % of the Yellow-sourced
  irradiance that the ports of visible Yellow receivers gather directly.

**B3f — see B3a (localized field).**

**B3g — which ports hold the gathered irradiance behind the Yellow indirect** (linear split, sun off, closure 1.000).

| View | Total pred/ref | Ports visible in view | Observed in training, not visible | Never observed | Receiver-local term |
|---|---|---|---|---|---|
| 6 | 2.51 | 0.37 | 0.95 | 1.10 | 0.08 |
| 15 | 0.96 | 0.14 | 0.37 | 0.41 | 0.04 |

So 82 % / 81 % of the predicted Yellow indirect is the nonlocal response to irradiance held at ports no camera of that
view sees. The mean gathered E per class is similar (visible 0.19, observed 0.22, never 0.20 on view 6), but the
non-visible classes are 95 % of Yellow's area.

**B4 — edit dynamics** (T0 → T1, matched Lego pixels, 16 views).

| Change | Region | sign agreement | corr | slope | Other |
|---|---|---|---|---|---|
| direct / intrinsic | A_changed | | 0.990 | 0.988 | explained 0.98 |
| indirect, updated | A_changed | 0.94 | 0.767 | 0.551 | explained 0.58; CI slope [0.31, 0.91], corr [0.64, 0.93] |
| indirect, updated | A_changed opaque | | 0.923 | 0.884 | |
| indirect, updated | A_changed Yellow | | 0.686 | **0.166** | magnitude 0.25 |
| indirect, updated | stationary opaque | | 0.777 | 0.880 | |
| indirect, updated | stationary Yellow | | 0.20 | | |
| indirect, stale | A_changed | | −0.52 | | |
| total | A_changed opaque | | 0.995 | | |
| total | A_changed Yellow | | | 0.66 | |

**B5 — indirect by material and distance** (all 16 views).
- Yellow is 99 % of the linear indirect squared error. Its indirect pred/ref is 1.74 (lit 1.48, shadow 2.02).
- By distance to lit ports: 1.37–1.59 within 1 cm, 1.84 at 1–3 cm, 2.94 at 3–10 cm, 4.20 at 10–30 cm.
- 95 % of the Yellow indirect prediction is band+mid at the receiver (ind 0.05, glob 0.03 of ref).
- Opaque indirect: 0.93 / 0.75; glass: 0.04 / 0.18; box: 1.69 / 0.79 (views 6 / 15).

---

## CROSS-TRACK ATTRIBUTION

Machine-readable form: `results/fnto_diag45/integration/integration.json`.

| Error | Class | Main evidence |
|---|---|---|
| Yellow first-event (K-only) error | **K intrinsic** (operator representation of 5–30+ cm object-scoped transport) | same verified input, the oracle is tied overall and better in shadow / 3–30 cm; object scope needed; 15 % red mass beyond 30 cm; rank-64 energy 0.83; noise floor 29.3–29.7 dB not limiting |
| Yellow composed indirect over-prediction | **K–C interface** (sender emission at non-visible Yellow ports), with a bounded **K intrinsic** receiver error | C verified; K's response to verified fields 0.56–1.37× cannot reach 1.74 / 4.2; sent power above any physical bound; 82 % of the prediction from irradiance held at non-visible ports |
| Opaque indirect under-prediction | composition source side + C contract limits; partly **unresolved** | receiver response exact; gathered E 0.76–0.89; enclosed receivers 0.65; glass transmission absent from C |
| Glass indirect missing | K–C interface / contract gap, **unresolved** between glass receiver response and absent transmission path | C to glass receivers correct; glass indirect tiny and almost all receiver-local |
| Yellow edit-change under-prediction | inherits the Yellow chain; C update correct | B4 |
| Solver | no independent error | converged, closures 1.0000 |
| Data / reference | not limiting | oracle noise 1.3 %; protocol tests |

**Consistency between the tracks:**
- Both see long-range red under-representation. Track A: the oracle needs 5–30+ cm transport (rank-64 energy 0.83).
  Track B: red response 0.38 to a non-Yellow field and 0.62 at 20–40 cm from a bump.
- Track A's K-only metric is near-unbiased at visible pixels (0.975). Track B finds the over-prediction in the sender
  path at non-visible ports, which a camera-pixel metric cannot see by construction. The two do not conflict.
- **Not consistent:** K over-predicts first-event light far from lit ports (A2, 1.36× at 10–30 cm) but under-responds
  far from a localized field (B3f, 0.78× at 20–40 cm). This stays open.

---

## INTERPRETATION

- **C's free-flight transfer is verified** against an independent renderer on a known field. Its limits are where the
  contract says so:
  - enclosed receivers get no light from inside faces (0.65);
  - near-coincident hits read high (1.16);
  - there is no transmission through glass;
  - the sender is isotropic.

  The 0.3 mm rule was not moved into C.
- **The composed Yellow indirect is not a C error and not a solver error.** In the composition, K is used in two
  roles:
  - as a receiver, mapping gathered irradiance to pixel radiance;
  - as a sender, giving every port's outgoing radiance through port-level tables with the exit direction fixed to the
    normal.

  K was supervised only as a receiver at camera-visible pixels.
  - In the receiver role, its response to verified fields is within 0.56–1.37×.
  - In the sender role, on Yellow, it emits too little where cameras look (0.35–0.41×) and, in total, more power than
    the object receives (red 1.8× the whole Lego's sun power). Most of that comes from shadowed ports that are mostly
    never observed and never lit in training.

  The receiver's band/mid aggregation (5 cm / 30 cm support) then turns irradiance held at those hidden ports into
  visible radiance. This is why the over-prediction grows away from lit regions.
- **The Yellow first-event error is the kernel's own operator limit.** It is in shadow, 3–30 cm from lit ports, in red
  and green. A no-learning Burley operator with three scalars and object-scoped sources is +1.9 dB better there on
  identical inputs.
- These readings rest on one scene, one sun family and two solve views for B3b–B3g. The attribution of the sender
  excess assumes Cycles' Burley conserves energy on the Lego geometry; this was verified on a plane only. No per-port
  reference exists for ports a camera never sees.

## UNRESOLVED QUESTION

- Whether Cycles' object-scoped Burley conserves energy on the Lego's folded geometry (plane: yes). If it does not,
  the sender energy bound is weaker than stated.
- Why K's first-event output over-predicts 10–30 cm from lit ports (A2) while its receiver response to a localized
  field under-predicts at 20–40 cm (B3f).
- Within the K-intrinsic Yellow limit: support (15 % red beyond 30 cm), rank (glob rank 64; operator rank-64 energy
  0.83), or the fit itself. band and mid are stored together and were not separated.
- Opaque gathered-field deficit (0.76–0.89): how much is missing glass transmission, sender error on other materials,
  or port-vs-pixel location. B3c compares the port average with the pixel point.
- Black: K's diffuse albedo is 0.27–0.30 of Cycles' DiffCol.
- Glass indirect: the receiver-response vs transmission-path split.

## ARCHITECTURE VERDICT (evidence status, not a batch verdict; no gate is applied)

- **Concept (K/C separation, FNTO contract):** not contradicted. C's transport of a known field is correct; the
  composed errors trace to how K is used at the interface and to K's own operator.
- **K:**
  - Yellow first-event fidelity is limited by the operator's representation of object-scoped 5–30+ cm transport.
  - The receiver response to verified free-space fields is within 0.56–1.37×.
  - The sender role is unconstrained at ports outside supervision, and on Yellow it violates energy bounds.
  - Not supported for Yellow; supported for opaque materials and the box at visible ports.
- **C:** free flight supported descriptively (one scene, four views, one known field family). Contract limits: enclosed
  receivers, near-coincident hits, no transmission path, isotropic sender.
- **Composition:** solver stable (6 iterations, ρ 0.55). It propagates the sender excess but adds none of its own.
- **Viability:** not assessed beyond this scene and edit. No claim relative to RNA, Surface-Owned or 8DNA.

## REPRODUCIBILITY

- **Commits:** e924a6f, dc5a2b4, b664832, 9cf728b, 607b933, e6790c2, 2c67f9a, plus this worklog's commit. Protocol
  file: `experiments/fnto_diag45/protocol/diag45.json`.
- **Server orchestration** (LabServer63, `~/rest_fnto_r1/`), with statuses in `launcher/d45_*.status`:
  - `d45_emit_chain.sh` — emit45 views 6, 15, 2, 10; emitonly 64 spp, emit 1024 spp; seed 4500+v.
  - `d45_a1a_chain.sh` — A1a, fit, A3.
  - `d45_b3c_chain.sh` — passes, 2048 spp, seed 4510+v.
  - `d45_b2fix_chain.sh` — B2 re-score and oracle replicate (seed offset 100).
  - `d45_b3_chain.sh` — B3b / B3c, emitall, B3a-emitall.
  - `d45_a4_chain.sh`.
  - `d45_bump_chain.sh` — seeds +0 / +100.
  - Also run: b3d, b3e, b3g, b3h.
- **Outputs:** `results/fnto_diag45/{track_a,track_b,common,integration}/`.
- **Reviewer exports:** `results/evaluation/45/` with manifest and sha256.
  - One absolute error / difference scale for every map, **M = 0.570 display units**, sqrt mapping, real-unit
    colorbars.
  - 01 Track A Yellow K-only error (views 10, 15); 02 B3a Yellow responses (emit, emitall, bump); 03 B4 indirect
    change (views 6, 15); 04 B2 scatter.
- **Incidents:**
  - b3b and b3d ran out of GPU memory on the second view (11.6 GB GPU) and were re-run after freeing tensors; there was
    no logic change.
  - The first B2 hit-point variant was NaN (miss rays t = ∞) and was fixed in amendment 1.
  - The production C numbers were unaffected by either.

## NEXT DECISION BOUNDARY (decision input only — no batch is started, no direction recommended)

**Smallest intervention target the evidence supports.** The radiance K emits into C at Yellow ports: the sender path,
port tables evaluated at the normal, at ports never visible, never observed or never lit in supervision. That emission
carries energy beyond any physical bound and dominates the composed Yellow indirect.

**What a reviewer would need to decide** before any intervention:
- whether the sender role must be supervised or constrained at all, or the composition must stop drawing on unobserved
  ports;
- whether the energy bound holds on the Lego geometry (one Cycles test);
- whether the remaining K-intrinsic Yellow first-event limit — support / rank / fit — is in scope at the same time.

**What would change this attribution:**
- evidence that Cycles' Burley creates energy on folded geometry;
- a per-port reference for hidden ports showing their exit radiance is in fact as high as K sends.

No successor batch is authorised; this batch stops here.
