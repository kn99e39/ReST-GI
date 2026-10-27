# Worklog — DASQ Composable K: Controlled Integration, Stability, Fidelity and Geometry-Edit Validation (2026-10-27)

Protocol `fnto_comp50` + amendments 1, 2, 2b (`experiments/fnto_comp50/protocol/comp50.json`). Authorisation: the user
brief "ReST-GI — Worklog 50: DASQ Composable K" (2026-10-27). All runs were on LabServer69 (RTX 3090 24 GB).

**Frozen.** K_k44P (sha256 `30d55da3…`), C relations / rays, the r1 solver, the incident field, every reference, the
worklog-49 checkpoints and verdicts, and worklogs 1–49. The production code was not modified: the arms enter through an
isolated subclass. The H baseline reproduces worklog 44 / 46 exactly on 69: S_T0 | hist 25.146 dB, S_T0 | r1 25.821 dB,
ρ 0.549.

**Answer to the completion question.**
- **DASQ's spatial advantage can be put into a complete FNTO K.** The complete-K first event improves over the matched
  existing aggregation:
  - Yellow +0.51 dB [0.43, 0.61] (seed 5011, 16 / 16 views) and +0.29 dB [0.18, 0.41] (seed 5012, 14 / 16);
  - over historical K44: +0.37–0.44 dB on Yellow, +2.1–2.3 dB on never-observed receivers;
  - no regression anywhere else.
- **The advantage does not survive repeated GI composition.** Every coupled solve diverges, for both arms, both seeds,
  T0 and T1: 64 / 64 per arm, ρ 6.9–9.0 against H 0.55.
  - The divergence is a red-channel Yellow ↔ Yellow loop: the arm carries 99 % of the dominant mode, and Yellow
    receivers gather 99.8 % of their energy from Yellow senders.
  - It is not DASQ-specific: L0C diverges too. It appears as soon as the Yellow in-support term is a learned response
    faithful to the Cycles first event, which is non-passive (worklog 49 bake 3.9×; here the arms' Yellow sender is
    4.1–4.9× the Lego sun power).
  - K44 converges only because its own band / mid sender is 5–9× low at the port normal (worklog 48), with K_glob
    carrying the Yellow red output.
- **Completion outcome:** 2 (first-event improvement that does not survive coupled composition), with an element of 3:
  under the current reference-fidelity contract, the decomposition cannot hold a reference-faithful Yellow response
  without a larger contract decision.

---

## GATE 0 — learned evidence and integration contract

**Worklog 49 L1 fluctuation.** The stored validation curves (last four evaluations, steps 17k–20k) are:

| Arm / seed | Last four validation losses |
|---|---|
| L1 4911 | 0.01432 / 0.01738 / 0.01491 / 0.01494 |
| L1 4912 | 0.02217 / 0.01451 / 0.01660 / 0.01663 |
| L0 | flat at 0.186–0.188 |

- The final two evaluations differ by < 0.3 %. The 14–15 % "change" came from comparing step 20k with an excursion at
  step 18k, during the decaying-learning-rate phase.
- Classification: **stochastic variation, not non-convergence**. The final checkpoints are deterministic and stable.
- Worklog 49's predeclared verdict (INCONCLUSIVE) is unchanged. Worklog 50 trains new matched arms on a different
  target with a robust fair-state rule.

**Decomposition** (Yellow receiver ports = 338,864 domain-A ports with material 0):

| Term | Ownership |
|---|---|
| K_loc | K44, unchanged |
| K_glob | K44, unchanged |
| K44 band + mid | **Replaced** by N_arm (isotropic; linear in E_tot per channel) at Yellow receivers only; unchanged for every other receiver |
| C, solver, direct sun, indirect gathers, port sender | Production semantics |

- **No-overlap rule.** The arms are trained on the shared residual r = Cycles probe colour − K44 K_glob − K44 K_loc at
  Yellow probe pixels (aggregated per port), so double counting with K_loc / K_glob is impossible by construction.
  - The decomposition test reproduces k44_train.predict to 5e-8.
  - No oracle, s_c or reference value is used at inference.
- **Fields:** K44's own band / mid input irradiance per probe configuration, with the worklog-49 splits.
- **Target statistics** (test split). The residual is 71 / 78 / 62 % of Cycles colour (RGB); K_loc is 4 / 13 / 33 %;
  K_glob is 25 / 9 / 5 %. K44's own band + mid fits the residual at rel L2 0.874 / 0.624 / 0.663.

## IMPLEMENTATION FACT

| File (`experiments/fnto_comp50/`) | Role |
|---|---|
| `c50_common.py` | ArmOp (L0 / L1 element tables + matched network weights for all Yellow ports); ArmState (production State.outgoing with a keep mask and the arm); arm_render_pixels; activate / deactivate patching of r1_solve |
| `c50_targets.py` | Residual targets and fields |
| `c50_train.py` | Matched training |
| `c50_eval.py` | The unchanged r1_eval protocol per condition: K_only, S_T0, D_stale, E_upd, E_relfull, spectral radius, signed change |
| `c50_onestep.py` | Gate 2 |
| `c50_diag.py` | Amendment 1: divergence attribution |
| `c50_refcheck.py` | Amendments 2 / 2b: Cycles camera-vs-indirect SSS check |
| `c50_report.py` | Report |
| `tests/test_c50.py` | Focused tests |

**Tests** (ALL_PASS before evidence, 3×).

| Test | Result |
|---|---|
| Frozen K44 and worklog-49 checkpoint hashes | equal |
| Candidate-off equivalence | outgoing 6e-8 = production-vs-production CUDA non-determinism; residual history 6e-7 |
| Arm ownership | non-Yellow ports / pixels unchanged to 1e-7; keep mask exact |
| Arm linearity / zero field / finite | 2e-7 |
| Residual decomposition | 5e-8 |

**Amendments.**
1. Divergence attribution diagnostic.
2. Plane test of whether indirect rays see the camera-seen Yellow SSS radiance.
- 2b. Pure-Burley rerun; the first refcheck run was confounded by the 2 % glossy highlight seen in the mirror direction.

**Engineering incidents** (no evidence affected):
- **A 43-minute CPU-bound training setup.** The npz arrays were re-read per row: GPU 0 %, no output. Fixed with
  arrays loaded once.
- **The stuck process survived pkill.** It kept running next to the chain-2 training until 20:46. It wrote nothing (the
  checkpoints are timestamped 20:19–20:45 from chain 2), but it shared CPU and 2.3 GB GPU memory.
- **Monitoring change.** The watcher was rebuilt to exit on stage changes, low GPU utilisation or stale logs
  (`C:\Users\dna10\restgi_relay\watch_chain.sh`).

## MEASUREMENT

### Matched learning (Gate 0 / 1 contract)

Training: 30k steps, batch 8 configurations × 64 rows, cosine schedule; 511–525 s per run.

| Arm / seed | Val rel L2 R / G / B | Test rel L2 R / G / B | Fair state (last-5 range) |
|---|---|---|---|
| L0 5011 | 0.834 / 0.566 / 0.540 | 0.863 / 0.609 / 0.638 | 0.65 % ✓ |
| L0 5012 | 0.834 / 0.565 / 0.541 | 0.863 / 0.608 / 0.638 | 1.36 % ✓ |
| L1 5011 | 0.825 / 0.554 / 0.533 | 0.855 / 0.599 / 0.633 | 0.58 % ✓ |
| L1 5012 | 0.825 / 0.555 / 0.528 | 0.855 / 0.599 / 0.627 | 0.77 % ✓ |
| K44 own band + mid (reference) | — | 0.874 / 0.624 / 0.663 | — |

- **The learned comparison is interpretable.**
- On the Cycles residual, L1's gain over L0 is only 1–2 % relative, whereas on the noise-free oracle target (worklog
  49) it was 65–86 %.
- Most of the probe residual is not explained by any arm, including K44's own band / mid. Probe-pixel noise and
  non-nonlocal residue are the likely causes; this split is not separated.

### Gate 1 — complete-K first event (K_only vs first_r1, held-out views, 16 per state)

| T0 Yellow PSNR (dB) | All | Lit | Shadow | 3–30 cm | Never-observed | R / G / B | View 14 |
|---|---|---|---|---|---|---|---|
| H (K44) | 24.95 | 28.07 | 23.45 | **21.62** | 20.07 | 23.27 / 23.75 / 31.79 | 27.80 |
| L0C 5011 / 5012 | 24.87 / 25.07 | 27.88 / 27.87 | 23.40 / 23.66 | 20.98 / 21.38 | 21.63 / 21.76 | 23.37 / 23.40 / 32.21 (5011) | 27.81 / 27.94 |
| **L1C 5011 / 5012** | **25.39 / 25.32** | **28.20 / 28.22** | **23.97 / 23.88** | 21.47 / 21.33 | **22.32 / 22.17** | 23.81 / 24.00 / 32.74 (5011) | 28.21 / 28.18 |

- T1 results agree within 0.05 dB.
- Non-Yellow pixels are identical to H (31.96 dB T0) by construction.
- **L1C − L0C, per-view Yellow PSNR** (predeclared: ≥ +0.3 dB, CI > 0, both seeds):

  | Seed | T0 | T1 |
  |---|---|---|
  | 5011 | +0.51 [0.43, 0.61], 16 / 16 | +0.50 [0.41, 0.59] |
  | 5012 | **+0.29** [0.18, 0.41], 14 / 16 | +0.27 [0.17, 0.39] |

  The criterion is **not met** by 0.015 dB on seed 5012. Both CIs exclude 0.
- **Integration valid:** Yellow-lit ≥ H − 1 dB for every arm; L1C lit is even +0.13–0.15 dB over H.
- **First-event edit** (K_only T1 − T0 vs first_r1 T1 − T0), corr / slope:

  | | All Lego pixels | Yellow |
  |---|---|---|
  | H | 0.988 / 0.986 | 0.871 / 0.811 |
  | L0C | 0.990 / 0.985 | 0.895 / 0.808–0.810 |
  | **L1C** | 0.991 / 0.988 | **0.911–0.912 / 0.855–0.866** |

### Gate 2 — single flight (one production gather after the first event; seed 5011)

| | Yellow indirect / ref (full − first) | corr | Opaque ratio | Box ratio | Glass ratio | One-step PSNR vs full (Yellow) | First-event Yellow sender / Lego sun power, R (views 6 / 15) |
|---|---|---|---|---|---|---|---|
| H | 1.36 | 0.24 | 0.67 | 0.90 | 0.08 | 26.39 (24.58) | 1.81 / 1.83 |
| L0C | **4.00** | 0.47 | 1.07 | 1.13 | 0.16 | 22.42 (19.31) | 4.15 / 4.39 |
| L1C | **4.24** | 0.39 | 1.10 | 1.14 | 0.16 | 22.28 (19.17) | 4.74 / 4.88 |

- The reference indirect contains all bounces, so a single flight that exceeds it 4× cannot be accurate.
- **Cycles reference check** (amendment 2b, pure Burley plane, white patch at 0.3 / 0.6 m, replicated): indirect
  irradiance / camera-radiance prediction = R 1.16–1.22, G 1.01–1.03, B 1.005–1.013.
  - Indirect rays see the camera-seen SSS radiance; the hypothesis that Cycles weakens SSS on indirect paths is **not
    supported**.
  - The plane itself is passive (R exit 1.18 vs E/π 1.59).
- The glossy-confounded first run (ratios 0.69 / 0.52 / 0.09) is preserved and is not evidence.

### Gate 3 — coupled solve (unchanged solver; no damping or clamping)

| Condition | ρ (power iteration) | S_T0 / D_stale / E_upd / E_relfull | Solve median |
|---|---|---|---|
| H | 0.549 / 0.553 | 64 / 64 converged | 5.8 s |
| L0C 5011 / 5012 | 6.89 / 7.25 | 64 / 64 **diverged** (each) | 6.8 s to the stop |
| L1C 5011 / 5012 | 8.99 / 8.97 | 64 / 64 **diverged** (each) | 9.6 s |

**Attribution** (amendment 1, homogeneous power iteration, seed 5011):

| | ρ | Dominant-mode power | Yellow output terms (R) | Yellow receivers' energy from Yellow senders |
|---|---|---|---|---|
| H | 0.549 (R 0.55, G 0.44, B 0.40) | opaque 86 %, Yellow 7 %, box 7 % | K_glob 86 %, K_loc 12 % | 94 % |
| L0C | 6.89 (R; G / B decayed below fp16) | Yellow 99.7 % | arm 99.0 % | 99.8 % |
| L1C | 8.98 (R) | Yellow 99.8 % | arm 99.3 % | 99.8 % |

### Gate 4 — real scene and edit after coupling

**NOT APPLICABLE.** No converged L0C / L1C solve exists (stop rule); r1_eval excluded every non-finite view. Only the
first-event edit (Gate 1) is measured.

### Gate 5 — cost (RTX 3090, identical code path)

| Item | H | L0C | L1C |
|---|---|---|---|
| Arm table build (338,864 ports) | — | 1.3 s | 6.0 s |
| Non-zero arm weights × 3 channels | — | 75.9 M | 157.1 M (2.07×) |
| Arm apply per solver iteration | — | 0.016 s | 0.36 s (23×; ≈ +45 % of an H iteration ≈ 0.8 s) |
| r1_eval wall | 649 s | 702–705 s | 909–922 s |
| Peak memory | 10.0 GB | 12.1 GB | 17.7 GB |
| Training per run | — | 511–513 s | 524–525 s |

- T1 update: the relation update is unchanged (incremental). The arm tables depend only on Lego geometry, so they need
  no rebuild for a box edit; only the per-iteration arm apply is repeated.
- Resources: GPU ≈ 1.9 h of compute (plus 43 min of a stuck CPU-bound process holding 2.3 GB); CPU renders 12 s
  (refcheck); wall 19:20–22:05 KST. All within the declared 4 h / 7 h.

## OBSERVATION

- **First event** (`results/evaluation/50/01_first_event_view{8,14}.png`, one absolute error scale M = 4.62).
  - L1C reduces Yellow shadow / hidden-receiver error and keeps K44's lit fidelity.
  - View 14 is no longer an outlier: worklog 49's oracle-only arms were at about 20 dB there, but the complete K keeps
    K_loc.
- **Single flight / solve.** Replacing K44's Yellow band / mid by any reference-faithful isotropic response doubles
  the Yellow sender (1.8× → 4.1–4.9× the Lego sun power) and makes the Yellow ↔ Yellow loop explosive in red.
- **Failures** (all preserved): every L0C / L1C coupled solve; the glossy-confounded refcheck; the stuck training setup.

## INTERPRETATION

- **Spatial representation improvement.** It transfers to the complete K at first event: L1C > L0C in both seeds
  (CIs > 0), and L1C > H on Yellow, shadow and hidden receivers. It is smaller than in oracle space, and the
  predeclared both-seed +0.3 dB threshold is missed by 0.015 dB.
- **Learned approximation.** Interpretable and matched (fair state met). On the noisy Cycles residual the arms are close
  (1–2 %), so supervision noise limits how much spatial advantage the target can reveal.
- **Component composition.** The composition is the binding failure, and it is shared by both aggregations. A Yellow
  response that is faithful to Cycles' first event, combined with C's Yellow-to-Yellow gathers, has loop gain ≈ 7–9.
  - Cycles' own full render is finite, so the reference's effective Yellow-to-Yellow multi-bounce gain is much smaller
    than first event × C implies.
  - Direct evidence excludes a camera / indirect SSS difference (refcheck ≈ 1).
  - The most consistent remaining explanation (inference, not isolated): object-scoped Burley already transports light
    between separate surfaces of the same object (worklog 41), and C transports the same exposed Yellow-to-Yellow paths
    again. This is an ownership overlap between K's reference semantics and C's same-asset interreflection, amplified
    by the first event's non-passivity (3.9×).
- **Numerical stability.** Measured outcome: "physically non-passive reference semantics → numerically unstable
  repeated composition". It is neither an inaccurate stable fixed point nor a solver defect: H is stable with the
  identical solver.
- **Edit response.** First-event Yellow change tracking improves with DASQ (slope 0.86 vs 0.81). Coupled edit response
  is unmeasurable.
- **Overhead.** 2× weights, 23× arm-apply time and 1.5× memory against L0C. Bounded, but not free.

## UNRESOLVED QUESTION (consequential)

1. **What is the double-counting mechanism?** Is the loop gain caused by object-scoped Burley cross-surface transport
   overlapping C's same-asset Yellow-to-Yellow paths, and how large is each share? This is a direct reference /
   ownership question, and the answer decides whether K or C must own those paths.
2. **What is the passivity contract?** A reference-faithful Yellow K (first-event 3.9×) and a convergent composition
   with C appear incompatible under current ownership. This is a human contract decision: reference semantics vs
   physical passivity vs ownership change.
3. **Does the probe-residual noise cap the measurable spatial benefit?** The arms differ by 1–2 % on Cycles residuals
   but by 65–86 % on the oracle.

## ARCHITECTURE VERDICT

| Item | Status | Basis |
|---|---|---|
| A. Learned comparison validity | **SUPPORTED** | Matched arms, fair state met in 4 / 4 runs; worklog 49's own verdict unchanged (its fluctuation = stochastic) |
| B. Complete-K first-event fidelity | **CONDITIONALLY SUPPORTED** | L1C − L0C +0.51 / +0.29 dB (CIs > 0); predeclared +0.3 both-seed criterion missed by 0.015 dB; L1C > H Yellow +0.37–0.44, hidden receivers +2.1–2.3 |
| C. K/C interface compatibility | **NOT SUPPORTED** under current ownership | Reference-faithful Yellow sender + C Yellow→Yellow loop; same for L0C |
| D. Single-flight transfer | **NOT SUPPORTED** for Yellow | One flight 4.0–4.2× the all-bounce reference indirect (H 1.36); opaque 1.07–1.10 (H 0.67) |
| E. Coupled-solve convergence | **NOT SUPPORTED** | 64 / 64 diverged per arm and seed; ρ 6.9–9.0 (H 0.55); red Yellow↔Yellow, arm 99 % |
| F. Coupled-solve fidelity | **NOT APPLICABLE** | No converged solve |
| G. Geometry-edit response | First event **CONDITIONALLY SUPPORTED**; coupled **NOT APPLICABLE** | Yellow change corr 0.91 / slope 0.86 (L0C 0.90 / 0.81, H 0.87 / 0.81) |
| H. Local / opaque / box regressions | **SUPPORTED** (none) | Non-Yellow identical; Yellow lit +0.13–0.15 vs H (first event) |
| I. Computational viability | **CONDITIONALLY SUPPORTED** | 2.07× weights, 0.36 s per iteration (23× L0C, ≈ +45 % per iteration), 17.7 GB |
| J. DASQ development viability | **INCONCLUSIVE** | DASQ's first-event advantage is real; the composition failure is not DASQ-specific and is blocked by an unresolved passivity / ownership contract, so further development is gated on that human decision |

The completion outcome is 2, with an element of 3. DASQ is not adopted, and no next step is selected here.

## REPRODUCIBILITY

- **Commits:**

  | Commit | Content |
  |---|---|
  | 8b681c7 | protocol + code, before evidence |
  | 30d56e6 | engineering fix (npz load + smoke switch) |
  | 5779507 | amendment 1 |
  | 20cbf26 | amendment 2 |
  | cf4f43d | amendment 2b |
  | 15c95a5 | report |
  | this worklog's commit | — |
- **Hashes:** K_k44P `30d55da3…`; worklog-49 checkpoints `e4b73423 / 480a198a / 052479fa / 082d48c8`; worklog-50
  checkpoints in `results/fnto_comp50/train/`.
- **Seeds:**

  | Item | Seed |
  |---|---|
  | Training | 5011 / 5012 |
  | Validation-curve subset | 5010 |
  | Splits | worklog 49 (4901–4903) |
  | Refcheck | 5021 / 5121 |
  | Bootstrap | 50 |
- **Environment:** LabServer69 RTX 3090, torch 2.8.0+cu128; Cycles 3.5.1 in docker (refcheck).
- **Server scripts:** `~/rest_fnto_r1/c50_chain2.sh` (tests, train, five evals, onestep), `c50_chain3.sh` (diag),
  `c50_refcheck.sh`; times in `launcher/c50_times.log`.
- **Outputs:**
  - `results/fnto_comp50/` — gate0_contract, train, eval/{H, L0_5011, L1_5011, L0_5012, L1_5012}, gate2_single_flight
    (+ refcheck), gate3_solver, report.json, tests;
  - `results/evaluation/50/` — first-event panels, JSON copies, manifest.

No worklog 51 is started.
