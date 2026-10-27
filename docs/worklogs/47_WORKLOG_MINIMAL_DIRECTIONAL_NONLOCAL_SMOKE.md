# Worklog — Minimal Directional Nonlocal Transport Representation Smoke (2026-10-27)

Protocol `fnto_smoke47` + amendment 1 (`experiments/fnto_smoke47/protocol/smoke47.json`). Authorisation: the user brief
"ReST-GI — Worklog 47: Minimal Directional Nonlocal Transport Representation Smoke" (2026-10-27). A smoke experiment on
representation sufficiency. No production architecture was built, no K was trained, and nothing in FNTO production
code, K_k44P, the K/C contract, references, protocols or earlier worklogs was changed.

**Answer to the completion question.**

- **Collapsing incoming radiance to irradiance removes necessary information when the intrinsic path is
  direction-selective.** This holds in the synthetic positive control (S1) and in a Cycles internal-mirror cavity (MIR).
  - Equal-irradiance input pairs (irradiance matched to 3e-7 in Cycles) give nonlocal outputs that differ by
    0.98–1.11 (relative L2), while the reference noise is 2e-5.
  - No irradiance-only model can separate such pairs. The best linear E-only map, fitted on the test set itself, still
    has 0.87–0.90 relative error.
- **A small direction-aware linear operator recovers the missing relationship, but only with full-boundary
  supervision.** B1 reaches 1e-5 (S1) and 3e-6 (MIR, below the 6.5e-5 reference noise floor).
  - With visible-only supervision (B0), the hidden outputs are wrong (1.03–1.06) and non-passive (S1 column albedo up
    to 1.77).
- **The regime that dominates the K44 error does not show the effect.** For the Lego Yellow material (Cycles Burley SSS
  plus 2 % glossy), fields at 0° and 60° with equal irradiance give the same nonlocal response at 3–30 cm, within the
  measured uncertainty:
  - red: every |difference| ≤ 1.0 % (bounds ≤ 2.3 %);
  - green: ≤ 1.9 %.
  - Blue is inconclusive beyond 6 cm because the signal there is zero.

---

## IMPLEMENTATION FACT

**What was created** (`experiments/fnto_smoke47/`, numpy + bpy only):

| File | Content |
|---|---|
| `smoke47_core.py` | 12-bin angular partition (projected solid angles sum to π), S0 / S1 oracles, renderer-domain conventions, field and equal-irradiance pair sampler, closed-form linear fit, metrics |
| `stage_ac47.py` | Stage A oracle evidence; Stage C A0 / A1 / B0 / B1 on S0, S1, MIR |
| `render47.py` | Cycles MIR cavity (basis / field / irradiance renders) and Yellow SSS plane |
| `eval_b47.py` | Stage B metrics and classes, MIR transfer matrices, superposition check |
| `diag47.py` | Post-hoc metric repair (see below) |
| `panels47.py` | Reviewer panels and manifest |
| `tests/test_smoke47.py` | Focused tests |

**Synthetic oracles (Stage A).** The domain has 4 strip ports (centres 0 / 1.5 / 3 / 4.5, width 1) × 12 incident or
outgoing bins: 3 rings (0–30–60–90°) × 4 azimuth sectors, giving 48 boundary states.

- **S0, angular-insensitive control.** Lambertian entry (t 0.9), port-to-port redistribution exp(−|Δx| / 2) with
  column albedo ≤ 0.7, then Lambertian exit. By construction T = G · E-map (rank 4).
- **S1, angular-sensitive control.** An internal rough mirror at depth 1. A beam entering port p in bin j exits at
  x_p − 2h tanθ cosφ (box overlap with the ports), in the mirrored-azimuth bin (0.7) and its two neighbours (0.1 each),
  with reflectance 0.8. Rank 40; maximum flux albedo 0.72.

  S1 is an illustration of an internal reflection path. It is not a claim about real materials.

**Renderer references (Stage B).** Blender / Cycles 3.5.1, CPU.

- **MIR (positive control).**
  - Domain: a closed black box (1.1 × 0.4 × 0.2 m) whose floor is a rough mirror (GGX, roughness 0.25, colour 0.9),
    with 4 ceiling windows (ports, x-width 0.1).
  - Inputs: windows P0 and P1, each with 8 sun directions (signed tan −1.5 … 1.875, angle 0.5°).
  - Port isolation: shadow-only masks.
  - Outputs: port-mean exiting radiance at all 4 windows, seen by 8 orthographic cameras along the matching outgoing
    directions (64 × 64 px; point-sampled GAUSSIAN 0.01).
  - Irradiance is measured with ρ = 1 Lambertian planes placed in the open windows.
  - The interior path (window → mirror → window) is enclosed inside one fixed domain, so it is intrinsic under the K/C
    contract. It stands in for an internal-reflection path, not for SSS.
- **SSS (relevance control, predeclared with its causal justification).** Worklogs 44–45 attribute most of the K44
  error to Yellow.
  - Material: the blend's own `Yellow` (Burley, radius (1, 0.2, 0.1), scale 1; mix 0.98 SSS + 0.02 Glossy 0.3), with
    its displacement link removed so that bump normals cannot make irradiance angle-dependent.
  - Geometry: a 3 × 1 m plane under a shadow-only half-plane occluder (edge at x = 0).
  - Lights: suns in the y–z plane, so the shadow edge does not move with polar angle. Field A is at 0°, strength 5;
    field B is at 60°, strength 10. Both give E = 5.
  - Cameras: top, +y 45°, −y 45° (64 × 8 px over x ∈ [−10, 30] cm).
  - Nonlocal bands: 3–6, 6–12, 12–30 cm.

**Learned operators (Stage C).** A dense linear map from the input representation to the full boundary output state,
solved in closed form:

  Σ_visible ‖W z − y‖² + λ tr(Wᵀ Lap W) + ε ‖W‖², with λ 1e-6 and ε 1e-9 (relative).

- Lap is the graph Laplacian over the output bins of the same port. With visible-only supervision, the hidden outputs
  are therefore the harmonic extension of the visible ones, which adds no hyperparameter.
- There is no optimiser, so no optimisation failure is possible. A model's error is either an information limit or
  the predeclared hidden-output rule.
- Inputs:
  - A: E per input port.
  - B: per-bin radiance for synthetic data, per-direction irradiance share for the renderer.
- Visible outputs:
  - synthetic: rings 0–1 × sectors 0–1, 4 of 12 bins at every port;
  - renderer: t ∈ {−0.75, 0, 0.375}, 3 of 8 directions.
- Data: 128 training configurations × 2 seeds, and 64 held-out configurations (32 random plus 16 equal-irradiance
  pairs). Train / test separation is asserted.
- MIR targets are T_r·L, where T_r comes from 16 basis renders (set 1). An independent set 2 gives the reference noise
  floor. Superposition was verified against direct renders.

**Unchanged.** K_k44P, the four-component K, production C / solver, every reference, every protocol, and worklogs
1–46. No GPU was used.

**Tests** (`results/fnto_smoke47/tests/`): ALL_PASS, both locally and on the server, before evidence and after
amendment 1.

| Test | Covers |
|---|---|
| Zero field | zero input gives zero output |
| Linearity / superposition | max abs < 1e-12 |
| Equal-irradiance pairing | E difference < 1e-12 |
| S1 known mapping | 0.8 · 0.7 · 0.5 exact |
| S1 equal-E / different output | — |
| S0 equal-E / equal output | — |
| Direction conventions | synthetic specular bin = geometric reflection; renderer travel / exit / camera |
| Units | constant radiance gives E = π; Lambertian exit |
| Passive flux | S0 0.63, S1 0.72 |
| Fit recovery | full supervision recovers T1 to 2.4e-5; hidden rows are harmonic to ε/λ |
| Train / test separation | asserted |
| Deterministic evaluation | — |

**Amendment 1** (committed before any Stage C metric existed). The first Stage C attempt stopped on the train / test
assertion: MIR has only 2 input ports, so all-inactive (zero) configurations occur in both sets. The Stage C sampler now
redraws all-zero configurations; the Stage A sampler is unchanged.

**Pre-evidence note.** After the first local test run, a Stage C crash check was run in the scratchpad. Its output was
discarded unread (it was redirected to /dev/null and deleted). Stage C was then rerun only after amendment 1.

**Post-hoc metric repair** (`diag_posthoc.json`; no new data and no fitting choice). Two predeclared ratios have a zero
true denominator in some cases:

1. Stage A nonlocal pair difference. The denominator ‖Y_A,nl‖ is 0 for some pairs. The repair uses max(‖Y_A,nl‖,
   ‖Y_B,nl‖).
2. Stage C pair discrimination on S0, where ΔY ≡ 0. The repair reports the spurious pair difference ‖ΔŶ‖ / ‖Y‖.

The S0 "directional" error is likewise undefined, because S0's exit is isotropic.

---

## MEASUREMENT

### Stage A — synthetic oracles (32 equal-irradiance pairs, seed 4791)

| | S0 (control) | S1 (positive control) |
|---|---|---|
| Per-port E match (max rel) | 2.8e-16 | 2.8e-16 |
| Angular input difference (mean rel) | 1.30 | 1.30 |
| Output difference, mean / median / max | 1.3e-16 / 1.1e-16 / 2.3e-16 | **1.068 / 1.076 / 1.383** |
| Nonlocal output difference, mean (repaired) | 1.2e-16 | **1.015** |
| Max column flux albedo / min entry | 0.63 / 0.0019 | 0.72 / 0 |
| Nonlocal share of output | 0.51 | 0.20 |
| Predeclared criterion | S0_insensitive **true** | S1_sensitive **true** |

### Stage B — renderer, paired fields with equal irradiance

**MIR.**
- Basis transfer T_r is 32 × 16; replicate noise per set is 7.8e-5 (rel).
- Pair R1: P0 open; A: t 0.75, B: t 1.5.
- Pair R2: P0 and P1 open; A: {0.375: 0.5, 1.125: 0.5}, B: {0.75: 0.6, −0.75: 0.4}.

| Pair | E_A / E_B at open ports (measured) | δE | D_rel all outputs | D_rel nonlocal ports | noise_D | Class | Superposition (direct vs basis) |
|---|---|---|---|---|---|---|---|
| R1 | 1.000 / 1.000 | 3.0e-7 | **1.108** | **1.108** (P1–P3) | 1.7e-5 | SENSITIVE | A 2.5e-5 (noise 2.5e-5), B 4.6e-5 (2.0e-5): pass |
| R2 | 1.000, 1.000 / 1.000, 1.000 | 5.1e-6 | **0.984** | **0.986** (P2, P3) | 2.8e-5 | SENSITIVE | A 2.0e-4 (2.1e-4), B 3.6e-5 (3.3e-5): pass |

Spatial / directional totals (radiance, summed):

| Pair | Field | Sum over directions, P0 / P1 / P2 / P3 | Main output direction |
|---|---|---|---|
| R1 | A | 0.003 / 29.6 / 0.10 / 0.0001 | t 0.75 |
| R1 | B | 0.002 / 0.02 / 71.5 / 0.0003 | t 1.5 |
| R2 | A | 0.03 / 0.57 / 1.95 / 3.08 | grazing, t 1.5–1.875 |
| R2 | B | 11.4 / 17.8 / 17.3 / 0.09 | t 0.75 and −0.75 |

**SSS (Lego Yellow, Burley).**
- Measured lit-side irradiance: E_A = E_B, δE 1.3e-6.
- Spatial output: |B − A| / max over each nonlocal band.
- Bound = 3 noise_D + δE.

| Camera | Band | R: D / bound | G: D / bound | B: D / bound |
|---|---|---|---|---|
| top | 3–6 cm | 0.19 % / 0.96 % | 0.05 % / 3.1 % | 0.41 % / 4.6 % |
| top | 6–12 cm | 0.64 % / 2.0 % | 0.07 % / 6.2 % (INC) | 1.2 % / 11 % (INC) |
| top | 12–30 cm | 0.33 % / 1.0 % | 1.9 % / 3.8 % | 13 % / 21 % (INC; signal ≈ 0 beyond 12 cm) |
| +y 45° | 3–6 cm | 0.59 % / 0.20 % (INC: resolved but < 10 %) | 1.2 % / 2.5 % | 2.0 % / 3.3 % |
| +y 45° | 6–12 cm | 0.60 % / 1.9 % | 0.33 % / 3.0 % | 1.4 % / 1.3 % (INC) |
| +y 45° | 12–30 cm | 1.0 % / 2.3 % | 0.08 % / 1.8 % | 56 % / 121 % (INC) |
| −y 45° | 3–6 cm | 0.05 % / 0.89 % | 0.05 % / 3.1 % | 0.41 % / 4.6 % |
| −y 45° | 6–12 cm | 0.68 % / 2.1 % | 0.07 % / 6.2 % (INC) | 1.2 % / 11 % (INC) |
| −y 45° | 12–30 cm | 0.32 % / 1.0 % | 1.9 % / 3.8 % | 16 % / 21 % (INC) |

- Classes over the 9 nonlocal band × camera cells:
  - R: 8 INSENSITIVE, 1 INCONCLUSIVE;
  - G: 7 INSENSITIVE, 2 INCONCLUSIVE;
  - B: 3 INSENSITIVE, 6 INCONCLUSIVE.
  - No cell is SENSITIVE.
- Lit side, local, not evidence: the top and −y cameras differ by 20–94 % (the glossy 2 % highlight at the specular
  direction). The +y camera, which sees no highlight, agrees to 0.2–0.3 % in R / G.

### Stage C — A0 / A1 / B0 / B1

Held-out evaluation uses 64 configurations and all outputs. Values are seed 4701 / seed 4702; one value means both
seeds agree to the shown precision.

| Regime | Cond. | Params | Train | Held-out | Hidden | Visible | Directional | Nonlocal | Pair discrim. | Pair-diff err | Neg. coef. (min / max) | Max flux albedo | Fit s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S0 | A0 | 192 | 0 | 0.0012 | 0.0015 | 0 | n/a | 0.0012 | n/a (ΔY = 0; spurious 1e-16) | n/a | 0 | 0.63 | 0.003 |
| S0 | A1 | 192 | 0 | 0 | 0 | 0 | n/a | 0 | n/a (1e-16) | n/a | 0 | 0.63 | 0.002 |
| S0 | B0 | 2304 | 0 | 0.0012 | 0.0015 | 0 | n/a | 0.0012 | n/a (1.2e-8) | n/a | 0 | 0.63 | 0.77 |
| S0 | B1 | 2304 | 0 | 0 | 0 | 0 | n/a | 0 | n/a (3.6e-9) | n/a | 0 | 0.63 | 0.69 |
| S1 | A0 | 192 | 0.91 / 0.88 | 0.948 / 0.950 | 0.956 | 0.934 | 1.00 | 1.13 | **0** | 1.00 | −0.46 / −0.50 | 0.70 / 0.80 | 0.003 |
| S1 | A1 | 192 | 0.91 / 0.90 | 0.954 / 0.939 | 0.964 / 0.939 | 0.934 | 1.01 | 1.25 | **0** | 1.00 | −0.56 / −0.49 | 0.51 / 0.58 | 0.003 |
| S1 | B0 | 2304 | 5e-6 | 0.856 | **1.057** | 0 | 0.845 | 0.865 | 0.658 | 0.902 | −3e-6 (numerical) | **1.77 (8 cols > 1)** | 0.70 |
| S1 | B1 | 2304 | 6e-6 | **1.4e-5** | 1.5e-5 / 9e-6 | 0 | 0 | 0 | **1.000** | 0 | −5e-6 (numerical) | 0.72 | 0.64 |
| MIR | A0 | 64 | 0.91 / 0.90 | 0.988 / 0.994 | 1.003 / 1.008 | 0.898 | 0.99 | 1.00 | **0** | 1.00 | −0.19 / −0.01 | — | 0.0005 |
| MIR | A1 | 64 | 0.90 | 0.892 / 0.887 | 0.891 / 0.884 | 0.898 | 0.90 | 0.90 | **0** | 1.00 | −0.26 / −0.12 | — | 0.0004 |
| MIR | B0 | 512 | 3e-6 | 0.954 | **1.033** | 0 | 0.954 | 0.999 | 0.513 | 0.913 | ≈ 0 | — | 0.02 |
| MIR | B1 | 512 | 3e-6 | **3e-6** (vs replicate 9.2e-5) | 3e-6 | 0 | 0 | 0 | **1.000** | 0 | ≈ 0 | — | 0.014 |

**Bounds** (test set):

| Regime | E-only linear floor | E-only pair floor (any function of E) | Reference noise floor |
|---|---|---|---|
| S0 | 6e-16 | 5e-17 | — |
| S1 | **0.896** | **0.681** | — |
| MIR | **0.871** | **0.591** | 6.5e-5 |

Linearity residual is 0 for all models. The MIR flux albedo is not measurable, because 8 cameras along one arc do not
integrate the outgoing hemisphere.

**Effects** (signed change of held-out / hidden relative L2; seed 4701, seed 4702 in parentheses where different).

| Regime | A0→A1 (supervision, E-only) | A1→B1 (angular input, full sup.) | B0→B1 (supervision, directional) | A0→B0 (angular input, visible sup.) | Interaction |
|---|---|---|---|---|---|
| S0 | −0.0012 / −0.0015 | 0 / 0 | −0.0012 / −0.0015 | 0 / 0 | 0 |
| S1 | +0.006 (−0.011) / +0.008 (−0.017) | **−0.954 (−0.939) / −0.964 (−0.939)** | −0.856 / **−1.057** | −0.092 / **+0.101** | −0.86 / −1.07 |
| MIR | −0.096 (−0.107) / −0.112 (−0.125) | **−0.892 (−0.887) / −0.891 (−0.884)** | −0.954 / **−1.033** | −0.034 (−0.040) / **+0.030 (+0.025)** | −0.86 / −0.92 |

**Predeclared criteria** (both seeds identical):

| Criterion | S0 | S1 | MIR |
|---|---|---|---|
| directional_benefit | false (nothing to gain) | **true** | **true** |
| learned (B1) | true | **true** | **true** |
| supervision_benefit_A | true (ratio; absolute 0.0015, degenerate near 0) | false | false |
| supervision_benefit_B | true (same caveat) | **true** | **true** |
| irradiance_sufficient | **true** | false | false |

**Runtime** (`runtime.json`; LabServer63, 32 CPU threads; no GPU).

| Item | Time |
|---|---|
| CPU rendering in total | 34 s (MIR basis 2 × 6–7 s, fields 7 s, SSS 14 s) |
| Whole evidence chain (tests, Stage A, renders) | 41 s |
| Stage C | 5.7 s wall (fits 5.2 s) |
| Post-hoc diagnostics, panels | a few seconds |
| GPU | 0 min |

Render counts: 256 basis + 64 field + 4 irradiance renders (MIR); 12 + 2 (SSS). This is far inside the 30 / 30 / 90 min
budget.

## OBSERVATION

- **MIR** (`results/evaluation/47/02_MIR_pairs.png`).
  - In R1, field A's light leaves only through P1 at t 0.75. Field B's equal irradiance leaves only through P2 at t 1.5,
    with a weaker t 1.875 tail. The outputs do not overlap.
  - In R2, A sends a little light to grazing directions at P1–P3. B sends strong light at t ±0.75 through P0–P2.
  - The images show the exit bands at different places. The difference maps share one scale (M = 59.2).
- **SSS** (`03_SSS_profiles.png`). In the shadow (x > 0), the A and B profiles lie on top of each other in R and G for
  every camera, down to 6e-2 (R) and 1e-5 (G). Blue falls to exactly 0 beyond ≈ 12 cm in both fields; G does the same
  beyond ≈ 25 cm. On the lit side, the top and −y cameras show the 2 % glossy highlight of the field whose mirror
  direction they face. The +y camera, which faces neither, shows identical lit profiles. The local SSS entry is
  therefore also angle-independent here.
- **Synthetic** (`01_stageA_oracles_pair.png`). S0 outputs under A and B coincide bin for bin. Under S1, A and B light
  different (port, bin) outputs.
- **Learned** (`04_stageC_errors.png`).
  - E-only models stay at or near the E-only floor whatever the supervision. A1 even fits with large negative
    coefficients (min / max up to −0.56), which is non-physical.
  - B0 fits the visible outputs exactly. Its hidden outputs are wrong and, on S1, non-passive.
  - B1 is exact up to reference noise.
- **Failure cases:**
  - every E-only condition on S1 and MIR;
  - B0's hidden outputs;
  - SSS blue beyond 6 cm (no signal);
  - one SSS red cell where a 0.6 % difference is resolved (+y 45°, 3–6 cm), still far below the 10 % sensitivity
    threshold.

## INTERPRETATION

- **Information loss (A).** Where intrinsic transport is direction-selective (S1, MIR), irradiance collapse is an
  information limit, not a fitting limit.
  - Equal-E pairs have identical E-only inputs, so any E-only predictor has pair error ≥ 0.59–0.68.
  - The best E-only linear map, fitted on the test set itself, reaches only 0.87–0.90.
  - The trained A1 models sit at 0.89–0.95, close to that floor. More capacity cannot remove the floor.

  Where transport enters diffusely (S0 by construction; Cycles Burley on the Lego Yellow, measured), irradiance carries
  the needed information within the measured uncertainty.
- **Supervision deficiency (B).**
  - Without angular input, full-boundary supervision helps little (A0→A1: S1 ±0.01, MIR −0.10). The E-only floor caps
    it.
  - With angular input, the supervision axis is decisive: B0→B1 changes the hidden error by −1.03 to −1.06.
  - Visible-only supervision combined with directional input gives:
    - exact visible outputs;
    - hidden outputs worse than E-only (A0→B0, hidden: +0.10 on S1, +0.03 on MIR);
    - non-passive hidden extrapolation (S1 albedo 1.77).

  The two factors are strongly synergistic: the interaction is −0.86 to −1.07. Neither factor is sufficient on its own.
- **Learning capacity / optimisation (C).** Not the limiting factor in this surrogate.
  - The fits are closed form, and B1 recovers the operator to 1e-5 (S1) or below reference noise (MIR) with 128
    configurations.
  - B uses 12× (S1) / 8× (MIR) more parameters than A. With 128 configurations ≥ 48 / 16 input dimensions, B1 is fully
    identified, and A's deficit is bounded by the information floor, not by its parameter count.
  - This says nothing about K44's capacity. K44 is a shared nonlinear-in-geometry network, not a dense per-state
    matrix.
- **Renderer uncertainty (D).**
  - **MIR:** irradiance matched to ≤ 5e-6; output differences 4–5 orders of magnitude above replicate noise;
    superposition holds to noise. The measurement is valid.
  - **SSS:** irradiance matched to 1.3e-6.
    - Red / green conclusions hold within ≤ 2.3 % / ≤ 6.2 % bounds; blue is noise-limited where its signal ends.
    - The reference is Cycles' Burley implementation, which is also what FNTO's references use. The result is a
      statement about that reference semantics, with the displacement removed. It is not a statement about physical
      SSS in general, about random-walk SSS, or about bump-mapped surfaces.
- **Relation to K44 (surrogate ≠ K44).**
  - K44's K_mid / K_glob take irradiance, and K_band takes a 7-bin incident partition.
  - For Yellow, the Stage B result says that the incoming collapse is not the missing information within the measured
    bound, which is consistent with worklog 45's attribution of the Yellow first-event error to K's intrinsic profile
    and of the indirect error to the sender side.
  - For direction-selective internal paths (for example glass internal reflection), the collapse would be an
    information limit. The size of that share in the Lego scene is not measured here.

## UNRESOLVED QUESTION

- How much of the Lego scene's intrinsic transport is direction-selective.
  - Red_Glass internal reflection / transmission is the candidate.
  - This smoke contains only a constructed cavity, so the share is unknown. It decides whether incoming collapse matters
    for the real scene beyond Yellow.
- Whether a structured, shared operator (such as K44's networks, or a unified K) extrapolates to hidden (port,
  direction) outputs better than the harmonic-extension rule used here. B0's hidden failure and non-passivity are a
  property of this surrogate's predeclared rule, not a test of K44 or of a unified K.
- Whether the Yellow insensitivity survives the bump normals that were removed here, or random-walk SSS. Both are
  untested.
- The resolved 0.6 % red difference in one SSS cell (+y 45°, 3–6 cm) is not attributed.

## ARCHITECTURE VERDICT

| # | Question | Status | Evidence scope |
|---|---|---|---|
| 1 | Is irradiance-only information sufficient on S0? | **SUPPORTED** | Synthetic, by construction: pair difference 2e-16; A1 exact; E-only floor 6e-16 |
| 2 | Is irradiance-only information sufficient on S1? | **NOT SUPPORTED** | Synthetic: E-only floor 0.896, pair floor 0.681, pair discrimination 0; information-theoretic |
| 3 | Is the same limitation observed in the renderer case? | **SUPPORTED** for the Cycles internal-mirror cavity (MIR: D 0.98–1.11 at δE ≤ 5e-6, noise 2e-5; E-only floor 0.871); **NOT SUPPORTED** for Lego Yellow Burley SSS (no SENSITIVE cell; R ≤ 1.0 %, G ≤ 1.9 % within bounds; B inconclusive beyond 6 cm) | Two constructed / single-material renderer regimes; 0° vs 60° only; displacement removed |
| 4 | Does directional input provide held-out benefit? | **SUPPORTED** with full-boundary supervision (A1→B1 −0.89 to −0.95, pair discrimination 1.000); **NOT SUPPORTED** with visible-only supervision (hidden +0.03 to +0.10, non-passive) | S1, MIR surrogate (dense linear, closed form) |
| 5 | Does boundary supervision provide independent benefit? | **NOT SUPPORTED** as an independent factor (A0→A1 ≤ 0.11, capped by the E-only floor); **SUPPORTED** conditional on directional input (B0→B1 −0.86 / −1.06, interaction −0.86 to −1.07) | S1, MIR surrogate |
| 6 | Does the result justify a more expensive FNTO experiment? | **INCONCLUSIVE** | For the Yellow SSS regime that carries most of the K44 error, the evidence does not support incoming collapse as a cause, so it gives no basis for an experiment premised on that. For direction-selective internal paths the information need is established, but their share of the real scene is unmeasured. |
| 7 | What is still unknown about four-component vs unified K? | — | (a) Whether K44's or a unified K's shared structure extrapolates passively to hidden (port, direction) outputs; the surrogate only shows that camera-only supervision plus a generic extension rule does not. (b) The direction-selective share of real intrinsic transport (glass). (c) Whether separating local / band / mid / global parts helps or hurts angular-input use. None of these was tested here, and nothing here tests the four-component K's fitting ability. |

**Case classification** (per renderer regime, protocol rule):
- **MIR → CASE 1.** The reference differs under matched irradiance, and the directional operator learns it.
- **Yellow Burley SSS → CASE 4.** The reference is approximately angle-insensitive within the measured uncertainty (R
  and G; B inconclusive).

No claim extends to SSS in general, to all mirror paths, or to other materials.

## REPRODUCIBILITY

- **Commits:**
  - 3b1b70d — protocol and code, before evidence
  - 976778c — amendment 1, before Stage C results
  - this worklog's commit — evaluation, post-hoc diagnostics, panels, worklog, living-document entries
- **Environment:** LabServer63.
  - Blender / Cycles 3.5.1 as the bpy module in the RNA venv (Python 3.10, numpy 2.2.6, matplotlib 3.10.9), CPU,
    32 threads.
  - Local test run: Windows Python 3.11 + numpy.
  - No GPU.
- **Seeds:**

  | Item | Seed |
  |---|---|
  | Stage A pairs | 4791 |
  | Stage C train | 4701 / 4702 |
  | Stage C test | 4790 |
  | MIR basis | 4710 + b (replicate + 100) |
  | MIR fields | 4730 + i (replicate + 100) |
  | MIR irradiance | 4750 + i |
  | SSS | A 4760, B 4761 (replicate + 100); irradiance 4770 / 4771 |
- **Data split:** 128 training configurations per seed; 64 held-out (32 random plus 16 equal-E pairs). No training
  configuration equals a test configuration (asserted).
- **Dimensions:**

  | Regime | Input | Output |
  |---|---|---|
  | Synthetic | 48 (B) / 4 (A) | 48 |
  | MIR | 16 (B) / 2 (A) | 32 |
- **Commands** (server, `P=<RNA venv python>`):
  - `~/rest_fnto_r1/s47_chain.sh` (tests, `stage_ac47.py a`, `render47.py -- mir_basis 0|1`, `-- mir_fields`, `-- sss`)
  - `eval_b47.py <results>`
  - `stage_ac47.py c <results>/stage_c <results>/mir/mir_T.npz`
  - `diag47.py`
  - `panels47.py <results> results/evaluation/47`
- **Outputs:**
  - `results/fnto_smoke47/` — stage_a, mir, sss, stage_b.json, stage_c, diag_posthoc.json, runtime.json, tests
  - `results/evaluation/47/` — 4 panels, JSON copies and a manifest; one absolute difference scale M = 59.2 radiance
    units, sqrt mapping
- **Not done:**
  - K44 evaluation or retraining;
  - any production change;
  - glass / transmission references;
  - bump-mapped or random-walk SSS;
  - a structured (shared-parameter) learned operator.

No later batch or worklog 48 is started.
