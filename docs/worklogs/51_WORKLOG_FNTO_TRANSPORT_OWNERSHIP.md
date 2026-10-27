# Worklog — FNTO Transport Path Ownership and Coupled Instability Attribution (2026-10-27)

**Protocol.** `fnto_owner51` (`experiments/fnto_owner51/protocol/owner51.json`), committed with the code in a066742 before
any primary evidence. It has one design revision made before that commit, which is disclosed in the protocol (see
Gate 2).

**Authorisation.** The user brief "ReST-GI — Worklog 51: FNTO Transport Path Ownership and Coupled Instability
Attribution" (2026-10-27).

**Hardware.** Every run was on LabServer69: an RTX 3090 for the GPU stages, and 16 CPU cores running Cycles 3.5.1
inside docker `restgi/bpy-runtime:22.04`.

**Frozen.** The following were not changed:
- K_k44P (sha256 `30d55da3…`);
- the WL50 arm checkpoints (`ckpt_L0/L1_seed5011/5012`, sha256 `0dfc7793` / `6041f0b4` / `3b40a940` / `1aa75e89`);
- C (`rel_T0` hit / t), the r1 solver and the render;
- the WL50 composition semantics (`c50_common`);
- every reference, material and piece of geometry;
- worklogs 1–50.

No production module was modified. Edge masks and the SEM override exist only on `State` instances created inside the
diagnostic processes; the tests check this, including that `none` equals production.

**Answer to the completion question.**
- **The divergence does not come from a duplicated K/C path in the renderer's path space.**
  - The K training target contains no free-space transport. The probe renders mask every non-camera opaque hit to black.
  - So C does not recompute anything the target already holds (CASE A NOT SUPPORTED in renderer path space).
- **The dominant mechanism is high-gain recurrence of a vertex-specific reference operator** (CASE B, with the
  vertex-semantics hypothesis SUPPORTED):
  - **What Cycles does:** it applies the object-scoped Burley BSSRDF only at path vertices with no diffuse ancestor
    (the camera vertex, or a vertex reached through glossy/singular bounces).
  - **How Yellow behaves at all other vertices:** every vertex reached after a diffuse or SSS-exit bounce sees Yellow
    as a plain Lambertian with albedo 0.98 · colour = 0.783 / 0.493 / 0.029.
  - **Evidence:**
    - the source code;
    - a controlled render, which gives exactly 0 versus a Burley prediction of 2.0 after a diffuse ancestor, and 0.99–1.00 after a mirror.
  - **What the target captures:** only the camera-vertex operator. Its first-event gain is 8.1× its own incident
    power in red (WL49 bake).
  - **What the composition does with it:** it applies that operator at every bounce. The arms' gain on the
    C-returned field is 8.1 / 10.5 (L0C / L1C, red), and C returns 85 % of the Yellow flux to Yellow.
    Hence ρ = 6.9 / 9.0.
  - **Diagnostic counterfactual:** replacing only the Yellow interior-vertex sender with the renderer's Lambert
    operator (SEM) gives ρ 0.79 for H / L0C / L1C.
    - All 48 SEM solves converge.
    - The coupled Yellow indirect becomes 0.74–0.75× the reference with correlation 0.81–0.83.
      Production H gives 1.74× with correlation 0.25.
- **A secondary, rule-triggered normalization finding** (CASE C rule): port-area reciprocity between two Yellow
  halves is 1.14 / 0.87.
  - The predeclared mapping therefore reads **CASE D** (B dominant, C minor).
  - The asymmetry sits on a 3.5–4 % cross term and cannot produce ρ 6.9–9.0. The class-level reduction g × M
    reproduces ρ exactly (6.888 vs 6.888).
- **Physical-contract question (separate from renderer path space).**
  - The camera-vertex Burley is object-scoped. By a geometric proxy, 57–59 % of its red response at Yellow receivers
    comes from source points whose chord leaves the surface into air.
  - That is transport which, under the intended FNTO contract, belongs to C.
  - Its exact magnitude is unmeasured (no inside test). It is reported, not resolved.
- **Correction of worklog 50's interpretation** (append-only; worklog 50 is not edited):
  - WL50 concluded from its plane refcheck that "indirect rays see the camera-seen Burley radiance, so the failure is
    not a camera / indirect reference mismatch", and inferred a K/C path overlap.
  - On a uniformly lit plane, Burley and Lambert exitance nearly coincide, so that check could not discriminate.
  - This worklog shows that the reference does differ by path vertex (diffuse ancestor → Lambert), and that this
    difference, not a duplicated path, accounts for the instability under the measured conditions.

---

## GATE 0 — target provenance and mathematical contract

### Probe and target facts (code)

**Probe renders.** `r1_stagea_render.Scene` + `mask_intrinsic`:
- Every node material gets `Mix(Fac = max(IsCameraRay, Backfacing if glass else 0), black emission, original)`.
- Settings: max_bounces 128, black world, no clamping, GAUSSIAN 0.01 point sampling, configuration lights only.

**WL50 Yellow target.** `r = Cycles probe col − K44 K_loc − K44 K_glob` at Yellow pixels.

### Cycles 3.5.1 implementation facts

These are read from the kernel source shipped with bpy 3.5.1 (`bpy/3.5/scripts/addons/cycles/source/kernel`):
- **`svm/closure.h:903–926`** — for CLOSURE_BSSRDF_BURLEY, `if (path_flag & PATH_RAY_DIFFUSE_ANCESTOR) param1 = 0`,
  so the radius becomes 0. The source comment: "disable in case of diffuse ancestor, can't see it well then and adds
  considerably noise".
- **`closure/bssrdf.h:300–333`** — a channel with radius < `BSSRDF_MIN_RADIUS` becomes a Lambertian `DiffuseBsdf`
  with the closure weight.
- **`integrator/subsurface.h:49, 98`** — the subsurface bounce clears `PATH_RAY_CAMERA`. The exit point gets a
  `DiffuseBsdf` (weight 1) instead of a shader evaluation.
- **`integrator/path_state.h:183–184`** — a `LABEL_DIFFUSE` bounce sets `PATH_RAY_DIFFUSE | PATH_RAY_DIFFUSE_ANCESTOR`.
- **The blend.** The Lego is **one object** with 8 material slots. Yellow = Mix(Fac 0.02, SSS Burley [colour
  (0.7991, 0.5029, 0.0296), scale 1, radius (1, 0.2, 0.1), IOR 1.4], Glossy 0.3), and Yellow is the only SSS material.

### Component table of the K target (Yellow)

| Component | Status | Evidence |
|---|---|---|
| Direct illumination | PRESENT | NEE at BSSRDF entry points and the 2 % glossy |
| Intrinsic SSS | PRESENT, camera-vertex Burley only, object-scoped (entry points on any Lego material within the probe radius, incl. across air) | source facts; Gate 2 object-scope test |
| Internal interreflection | ABSENT as explicit paths (folded into the Burley profile) | Burley is a profile model |
| External free-space interreflection | ABSENT | `mask_intrinsic`: non-camera opaque hits are black emission |
| Multiple transport events | ABSENT, except the single camera-vertex BSSRDF event | as above |
| Existing indirect lighting | ABSENT | black world |

### Transport equations and path-family ownership

The protocol records these before the counterfactuals were run.
- **Intended FNTO contract.** `L_out = K(E_tot)` at every boundary vertex, and `E_ind = C(L_out)`. K is path-history
  independent.
- **Reference (Cycles), camera vertex.** `L_cam(x_B) = ∫ S_Burley,obj(x_B ← x_A) E_tot(x_A) dA + glossy`.
- **Reference (Cycles), interior Yellow vertex.** `L(y) = 0.98 c / π · E_tot(y) + glossy`.
- **WL50 composition.** The camera-vertex operator is used at every vertex.

The residual subtraction is algebraic; it does not establish path-disjoint ownership. By provenance, the residual is an
intrinsic camera-vertex response plus K44 fit error, and it contains no composed transport. It is **not**
path-identifiable as a vertex-independent intrinsic operator, because the reference itself is vertex-dependent.

### Air-chord proxy

This uses the WL45 object-scoped Burley oracle at 1000 held-out Yellow receivers, under view fields 6 / 15. It is a
geometric **proxy** with no inside test.

| | R | G | B |
|---|---|---|---|
| response from chords leaving into air | 0.589 / 0.593 | 0.502 / 0.512 | 0.406 / 0.418 |
| … in-support (< 30 cm) | 0.572 / 0.590 | 0.502 / 0.512 | 0.406 / 0.418 |
| response entering through non-Yellow Lego materials | 0.369 / 0.308 | 0.148 / 0.111 | 0.046 / 0.028 |

---

## IMPLEMENTATION FACT

All code is in `experiments/fnto_owner51/`:
- `o51_ops.py` — stages g0 / g3 / g1 / views / smoke. It holds the edge masks, the per-channel power iteration (each
  channel renormalised separately, which fixes the WL50 diag NaN), the mode statistics, the SEM override, the C
  closure / reciprocity audit and the air-chord proxy.
- `o51_slab_render.py` — the Gate 2 Cycles control.
- `o51_report.py` — the predeclared rules.
- `o51_figures.py`
- `tests/test_o51.py`
- `o51_chain.sh` — the server chain.

**SEM (diagnostic only).**
- **What changes:** the value a Yellow port sends into C gathers becomes `0.98 c/π · E_tot + K44 kap/π · E_ind`. This
  is the renderer's diffuse-ancestor operator plus K44's own glossy term; the sun glossy lobe is unchanged.
- **What stays:** every receiver and render term (K_loc, K_glob, arm or band + mid, γ, N_arm) is unchanged. The one
  causal variable is therefore the operator applied at interior Yellow vertices.
- **Consequence:** the sender no longer depends on the arm, so ρ and the solved field are identical for H / L0C / L1C;
  the arms differ only at camera-vertex Yellow pixels.

**Tests.** 9 / 9 pass (`results/fnto_ownership51/tests/test_o51_pre.json`):
- baseline identity;
- class classification vs anchors (Y 338,864, O 511,408, G 2,086, X 1,702);
- port-area sums (Lego 40.835, Yellow 15.875 m²);
- mask `none` = production;
- exact edge-mask sets (0 mismatches over 3000 ports × 128 gathers × 8 masks);
- channel-diagonal and homogeneous map;
- SEM override scope (non-Yellow ≤ 1e-6 relative, which is CUDA `index_add_` non-determinism; Yellow = formula);
- uniform-radiance unit consistency (E = π × valid fraction, exact).

---

## MEASUREMENT

### Gate 1 — frozen K/C loop attribution

**Per-channel spectral radius (R / G / B)**, from 40 power iterations with init seed 0:

| Edge set | H | L0C 5011 | L1C 5011 |
|---|---|---|---|
| none (production) | 0.549 / 0.435 / 0.400 | **6.888** / 4.288 / 0.399 | **8.982** / 5.138 / 0.399 |
| no Y←Y | 0.548 | **0.854** / 0.660 / 0.400 | **0.988** / 0.676 / 0.400 |
| no other←Y | 0.548 | 6.427 | 8.398 |
| no Y←other | 0.549 | 6.877 | 8.971 |
| no other←other | 0.576 | 6.882 | 8.977 |
| only Y←Y | 0.576 | **6.427** / 4.100 | **8.398** / 4.959 |
| no Y←Y with t < 3 mm | 0.548 | 6.121 | 7.998 |
| no Y←Y with t < 3 cm | 0.548 | 1.515 / 0.930 | 1.921 / 1.013 |
| no Lego↔box | 0.528 | 6.888 | 8.982 |
| **SEM** | **0.791 / 0.508 / 0.400** | **0.791 / 0.508 / 0.400** | **0.791 / 0.508 / 0.400** |

Replicates and caveats:
- Seed 5012 unmasked: L0C 7.247, L1C 8.971.
- Init-seed reproducibility: relative differences 5e-7 / 2e-8 / 2e-8. The growth rate is constant to the third
  decimal over the last 10 iterations.
- K44's operator is not elementwise non-negative (it has learned weights), so removing edges need not lower ρ. H's
  "no other←other" and "only Y←Y" give 0.576 against 0.549.

**Rules (predeclared).**
- **Y←Y necessary:** YES for L0C and L1C. No other group is necessary.
- **Y←Y sufficient:** YES.
- **Near-coincident (< 3 mm) necessary:** NO.
- **Reproducible:** YES.

Per the protocol, removing Y←Y edges proves only that they carry the unstable feedback, not that they were double
counted.

**Dominant mode (red, seed 5011).**

| | H | L0C | L1C | SEM (all) |
|---|---|---|---|---|
| Yellow share of outgoing flux | 0.07 | 0.997 | 0.998 | 0.996 |
| arm share of the Yellow sender | — | 0.990 | 0.993 | — |
| Yellow gain g_Y (out / in flux) | 0.65 | **8.08** | **10.47** | **0.798** |
| return fraction M_YY (Yellow → Yellow) | 0.77 | 0.85 | 0.86 | 0.99 |
| class-reduced ρ (g × M) | 0.532 | 6.888 | 8.982 | 0.790 |
| Y←Y gathered flux at t < 3 cm | 0.82 | 0.86 | 0.86 | 0.90 |
| participation ratio / top-1 % port share | 0.023 / 0.32 | 0.13 / 0.16 | 0.11 / 0.19 | 0.10 / 0.19 |

Notes on the mode:
- The mode is spread over Yellow, not concentrated in a few ports. Its largest connected components are 289 / 290.
- In the unstable mode, the loop lives at short range: 86 % of the Y←Y gathered flux comes from hits within 3 cm.

**Views (held-out T0, 16 views).** Indirect component vs the reference (full_r1 − first_r1), pooled luminance ratio
and correlation:

| | Yellow | opaque | glass | box |
|---|---|---|---|---|
| H prod one-step | 1.36 / 0.24 | 0.67 / 0.87 | 0.08 | 0.90 / 0.92 |
| L0C / L1C prod one-step | **4.00 / 4.24** (corr 0.47 / 0.39) | 1.07 / 1.10 | 0.16 | 1.13 / 1.14 |
| SEM one-step (H / L0C / L1C) | 0.42 / 0.45 / 0.44 (corr 0.75–0.81) | 0.59 | 0.07 | 0.77 |
| H prod coupled solve | **1.74 / 0.25** | 0.89 / 0.88 | 0.10 | 1.14 / 0.94 |
| SEM coupled solve (H / L0C / L1C) | **0.72 / 0.75 / 0.74** (corr 0.77 / 0.81 / 0.83) | 0.84 / 0.90 | 0.08 | 1.03 / 0.94 |

Solve convergence and rules:
- SEM solves converge on 16 / 16 views for each condition, in 12–14 iterations; H production converges in 6–7.
- **SEM stabilises:** PASS (all ρ < 1, all solves converged).
- **SEM matches the reference indirect:** PASS (L0C 0.753, L1C 0.737, within [0.67, 1.5]).

**Display PSNR vs full_r1 (view mean; descriptive only, not adoption evidence).**

| | Yellow | opaque | glass | box | all |
|---|---|---|---|---|---|
| H prod solve | 23.56 | 29.69 | 16.68 | 37.11 | 25.82 |
| H SEM solve | 26.39 | 30.04 | 16.77 | 38.64 | 27.81 |
| L0C SEM solve | 27.64 | 30.04 | 16.77 | 38.64 | 28.47 |
| L1C SEM solve | 28.75 | 30.04 | 16.77 | 38.64 | 28.99 |

The SEM rows are one seed and a diagnostic operator. They do not show that any production change would work.

### Gate 2 — path-controlled renderer case (`results/fnto_ownership51/gate2_renderer/slab.json`)

**Setup.**
- **Geometry:** two closed Yellow slabs, 0.4 × 0.4 m, 4 mm thick, with a 4 mm air gap.
- **Material:** the blend's Yellow, displacement link removed, otherwise unchanged.
- **Lighting:** sun at angle 0.
- **Sampling:** 1024 spp for the camera views, 4096 for the patches; seeds 5101 / 5102.
- **Cost:** 70 s of CPU in total.

| Config / measurement | R | G | B | Seed spread |
|---|---|---|---|---|
| **L_above** (Lambert, same geometry) patch E_meas / E_pred — chain control | 1.000 | 1.000 | 1.000 | 1e-4 |
| **J_below** (one object, lit from below) top face, camera vertex | 0.845 | 0.685 | 0.033 | 0.001 |
| J_below via **diffuse** patch (diffuse ancestor): E_meas / E_pred | **0.000** | **0.000** | **0.000** | exact 0, both seeds |
| J_below via **mirror** patch (singular ancestor): L_mirror / L_top | 0.993 | 0.996 | 1.001 | 0.01 |
| **S_below** (two objects, identical vertices) top face, camera vertex | 3.6e-10 | 2.6e-11 | 0 | — |
| S_below bottom slab's top face (camera vertex in the gap) | 0.853 | 0.760 | 0.043 | 0.003 |
| J_above (Yellow, lit from above at 45°) via diffuse patch, descriptive | 1.55 | 1.07 | 1.54 | 0.004 |

What each row shows:
- **Diffuse-ancestor rule CONFIRMED.** The chain is valid, and J_below gives exactly 0 (the predicted value was 2.03
  R). After a diffuse bounce, Yellow is Lambert: the top face is in shadow, so it has no exitance.
- **Singular ancestors keep Burley.** The mirror ratio is 0.99–1.00.
- **Object scope.** The top face's camera-vertex radiance across the air gap is 0.845 when the slabs are one object and
  ≈ 0 when they are separate objects. This is a changed reference condition, not a C ablation.
- **The free-space return path in Cycles.** In S_below, the bottom slab's gap face has camera-vertex exitance 0.853.
  Yet the separate top slab receives exactly nothing, because the gap face is Lambert at a diffuse-ancestor vertex and
  in shadow.
  - A path-history-independent K would send 0.853 into C there.
- **Lambert vs Burley on a uniformly lit slab (J_above, descriptive).** The Lambert interior operator is 1.55× (R) the
  camera-vertex Burley exitance of this 4 mm slab. Burley loses red through the thin slab; B is confounded by the
  45° glossy lobe.

**Pre-commit design revision.** The protocol first used J_above as the chain control.
- The uncommitted profiling smoke (16 / 32 spp) gave 1.54 / 1.07 / 1.52 for it, showing that this Yellow
  configuration cannot validate the measurement chain.
- The control was moved to L_above (plain Lambert) before the commit.
- The smoke had also already shown J_below = 0 and S_below ≈ 0. Those rules were not changed. This is recorded in the
  protocol under `design_revisions_before_commit`.

### Gate 3 — energy and normalization audit (`gate3_energy/g3_closure.json`)

**Units.**
- E is in W m⁻², L in W m⁻² sr⁻¹.
- `E_ind = π/K Σ L_k` is exact for uniform radiance; this is tested.
- Isotropic flux is π L A_port.

**C closure.** For uniform unit radiance per sender class, the captured fraction is ≤ 1 for every class:
- Y_hi 0.865, Y_lo 0.869, O 0.832, G 0.644, X 0.439.
- Yellow → same Yellow half: 0.731. Yellow → Yellow total: 0.77.

**Reciprocity Q[r, s].**
- Within [0.97, 1.03] for every pair except **Y_hi ↔ Y_lo: 1.144 / 0.874**. The predeclared CASE C rule is
  triggered.
- That pair carries 3.5–4 % of the Yellow emission; the 73 % same-half return cannot be checked by reciprocity.

**Port vs mesh areas.**

| | Mesh (m²) | Ports (m²) | Port / mesh |
|---|---|---|---|
| Lego | 27.0 | 40.8 | 1.51 |
| Yellow | 10.95 | 15.87 | 1.45 |
| Gray | 3.34 | — | — |
| Sand | 8.66 | — | — |
| Black | 2.54 | — | — |
| Brown | 0.93 | — | — |
| Light_brown | 0.52 | — | — |
| transparent | 0.033 | — | — |
| Red_Glass | 0.033 | — | — |
| box | 0.235 | 0.235 | 1.0 |

Port areas enter the numerator and denominator of every flux gain and of C's return fraction alike, and the arms were
trained and applied on the same measure. A uniform 1.45–1.5× inflation cannot change ρ.

**Gains.** The four quantities are kept separate:
- **Renderer-fidelity measurement.**
  - Camera-vertex Yellow exit / own incident = 8.10 / 2.86 / 0.115 (WL49 bake, mesh areas, view 6).
  - Cycles interior-vertex albedo = 0.783 / 0.493 / 0.029 (source code plus Gate 2).
- **Effective learned-operator gain.** Yellow first-event out / in flux (port areas), views 6 / 15:

  | | R | G | B |
  |---|---|---|---|
  | H | 4.24 / 3.35 | 1.03 / 0.88 | 0.07 |
  | L0C | 9.71 / 8.07 | 3.81 / 3.33 | 0.15 / 0.14 |
  | L1C | 11.10 / 8.95 | 4.08 / 3.51 | 0.15 / 0.14 |

  The arms reproduce the reference camera-vertex gain they were trained on. Their gain on the C-returned mode field
  (8.1 / 10.5) is of the same size.
- **Physically closed flux accounting.** C captures ≤ 0.87 of emitted flux, with no violation.
- **Coupled spectral radius.** ρ ≈ g_Y × M_YY: 8.08 × 0.85 ≈ 6.9 and 10.47 × 0.86 ≈ 9.0. Under SEM it is
  0.798 × 0.99 ≈ 0.79.

A global first-event ratio alone does not explain ρ. Here ρ is explained by the gain applied at every recurrence times
the return fraction. A finite Cycles image is consistent with this, because Cycles never applies Burley recursively.

---

## OBSERVATION

- **The reference is not generated by one path-history-independent Yellow operator.**
  - At camera vertices, Yellow is a non-passive, object-scoped Burley response (gain 8.1 red on this geometry).
  - At every diffuse-ancestor vertex, it is a passive local Lambert (0.78 red).
- **K44, L0C and L1C are all trained only on the camera-vertex response.**
  - K44 converges because its normal-direction sender is 5–9× low (WL48).
  - The reference-faithful arms diverge.
- **The unstable loop:**
  - is Yellow → Yellow;
  - is 86 % within 3 cm;
  - is unchanged by removing near-coincident (< 3 mm) or cross-domain edges.
- **SEM behaviour.** With the renderer's interior operator substituted at Yellow senders, the composition converges for
  every K variant. The Yellow indirect is then 0.72–0.75× the reference with correlation 0.77–0.83.
  - The remaining shortfall (≈ 0.75×) is not attributed in this batch. Candidates:
    - Yellow seen after glossy/singular bounces is Burley in Cycles but Lambert in SEM;
    - C's own gathered-field deficit (WL45: 0.76–0.89×).
- **Glass** stays at 0.07–0.10× of its reference indirect in every condition. Transmission is out of scope.

## INTERPRETATION

- **Proven path overlap (K vs C, renderer path space): NOT SUPPORTED.**
  - The target has no free-space paths.
  - The Cycles full render contains both the camera-vertex teleport paths and Lambert-emitted free flights.
  - SEM reproduces that path set without duplication.
- **High-gain feedback through disjoint paths: SUPPORTED.**
  - The paths are disjoint, but the composition recurses an operator the reference only ever applies once per path, at
    the observed end.
  - The instability is a property of the operator (ρ reproducible to 1e-7, growth constant), not a numerical artefact.
- **Training-target contamination: NOT SUPPORTED in renderer path space.**
  - The target is a clean camera-vertex response.
  - It is, however, not the operator the composition needs at interior vertices.
  - Relative to the physical intended contract, the object-scoped Burley teleports light across air (proxy 57–59 % of
    the red response). That is a renderer path-semantics feature of the reference, not something C recomputes in
    Cycles' own path space. Its effect on a physically disjoint K/C split is unquantified.
- **Interface / accounting error: minor.**
  - Units, closure and inter-class reciprocity hold.
  - The Yellow-halves reciprocity (1.14 / 0.87) is a real port-area weighting inconsistency of modest size and does not
    drive the instability.
- **Characterised contract mismatch** (no correction implemented and none selected):
  - The FNTO contract assumes one intrinsic K per optical domain, applied identically at every boundary vertex and
    trained from first-event references.
  - Under Cycles 3.5.1, a reference-consistent decomposition would have to distinguish the operator at an observed,
    no-diffuse-ancestor vertex from the operator at interior vertices.
  - Alternatively, the reference itself would have to apply one operator at all vertices; that would be a reference
    change.
  - Either choice changes the research contract and is a human decision.

## UNRESOLVED QUESTION (consequential)

- **Glossy/singular ancestors.** How much of the reference Yellow indirect comes via glossy/singular ancestors, where
  Cycles keeps Burley? This would set how far a two-operator (observed / interior) decomposition can match the
  reference.
- **Air-crossing share.** How much of the camera-vertex Burley response crosses air, measured with an inside test or
  object-split renders on the Lego? This determines whether a physically disjoint K/C split is compatible with this
  reference.

## ARCHITECTURE VERDICT (measured conditions only: one scene, Lego Yellow under Cycles 3.5.1 Burley, held-out T0, seed 5011 for modes / views)

| Item | Verdict |
|---|---|
| A. Intrinsic target validity | **CONDITIONALLY SUPPORTED.** Valid as the camera-vertex (no-diffuse-ancestor) response. Not valid as a path-history-independent operator for this reference. Contains object-scoped cross-air transport (proxy) |
| B. Same-domain coupling contribution | **SUPPORTED.** Y←Y necessary and sufficient for ρ > 1; 86 % of the loop flux within 3 cm |
| C. Proven transport path overlap (K vs C) | **NOT SUPPORTED** in renderer path space. Physical-contract overlap via object-scoped teleport is INCONCLUSIVE (proxy only) |
| D. Reference energy accounting | **SUPPORTED.** Camera-vertex Burley non-passive (8.1×); interior operator passive (0.78); finite Cycles images consistent |
| E. Numerical coupling stability (as the cause) | **NOT SUPPORTED.** ρ is a reproducible operator property |
| F. K/C normalization consistency | **CONDITIONALLY SUPPORTED.** Units, closure and class reciprocity hold; Yellow-halves reciprocity 1.14 / 0.87 |
| G. Need for ownership-contract modification | **SUPPORTED.** The single vertex-independent K contract cannot match this reference with a convergent composition; correction characterised above, not implemented |
| H. Sufficiency of evidence for the human decision | **SUPPORTED** for the mechanism. The two unresolved shares above remain open |

Predeclared CASE mapping: **CASE D** (B dominant; C rule-triggered and quantitatively minor). The vertex-semantics
hypothesis is SUPPORTED. K44, DASQ and FNTO as a whole are not judged beyond these conditions.

## REPRODUCIBILITY

**Commits.**
- a066742 — protocol, code and tests, before evidence.
- This worklog commit — results, figures and the report.
- Server repo files were synced by tar; the hashes match local (`o51_ops.py` d4a5d24b…, `o51_slab_render.py`
  5c013c1e…).

**Chain.** `experiments/fnto_owner51/o51_chain.sh` (setsid; status in `~/rest_fnto_r1/launcher/o51_chain.status`).

**Stages and wall time.**

| Stage | Time | Peak GPU memory |
|---|---|---|
| tests | 32 s | — |
| g0 | 4 s | 0.45 GB |
| g3 | 17 s | 9.3 GB |
| g1 | 1460 s | 20.3 GB |
| views | 1381 s | 19.6 GB |
| slab (CPU, parallel) | 70 s | — |

- Total ≈ 48 GPU-min, plus about 5 min of smoke runs. Within the 1.5 GPU-h budget.

**Seeds.** Power-iteration init 0 / 1; arm checkpoints 5011 (5012 for ρ only); render 5101 / 5102.

**Outputs.**
- `results/fnto_ownership51/`:
  - gate0_provenance/g0_chord.json
  - gate1_operator/{g1, views}.json
  - gate2_renderer/slab.json + npy
  - gate3_energy/g3_closure.json
  - tests/
  - report.json
  - times.log
- `results/evaluation/51/`:
  - 01_slab_path_semantics.png — shared absolute sqrt scale;
  - 02_rho_by_mask.png;
  - manifest.json.

**Operational notes.**
- The first test run hit an out-of-memory error from full-size fp32 copies; the comparisons were moved to a port
  subset.
- The SEM override's closure created a reference cycle that kept States alive; `gc.collect()` was added.
- Neither affects results.

No later batch is started or recommended.
