# Worklog 104K-A — ER-v1-SDM optimization closure and scene-conditioned fidelity (2026-11-22 KST)

## Authorization, classification and freeze

The owner's brief authorizes a diagnostic follow-up of WL104K, named WL104K-A. It asks why ER-v1-SDM improves held-out angular error while worsening the hard_1 highlight, and whether its scene advantage is stable enough to justify further integration work. No new architecture is implemented.

- **Batch class.** Under the AGENTS.md policy this is a suffix batch: a dependent evidence-gap follow-up of WL104K.
- **Preserved unchanged.** WL104K's frozen verdict and its appended post hoc section. ER-v1-SDM is not adopted.
- **Repository and branch.** `ecro_lr3_adapter`, started at `4bc5510`.
- **Host.** Local RTX 5080, with no other GPU job during the batch.
- **Freeze.** The protocol and all diagnostic code were committed at `4201954` before any evidence: [protocol](../../experiments/ecro_wl84_104ka/protocol.json), [code](../../experiments/ecro_wl84_104ka/).
- **Pre-freeze development (disclosed in the protocol).**
  - The protocol text was written before any code ran.
  - Stage A, a hard_1 lane smoke, and one dev chain with a 2,100-step extension were run on scratch copies.
  - Stage C on the frozen 12k states is deterministic. Its 12k numbers and labels for all three scenes were therefore seen during development; for example, the hard_1 highlight core was MIXED.
  - No rule, region, bin, threshold or seed was changed after any development output.
- **Additional descriptive analysis (not predeclared).** [`descriptive104ka.py`](../../experiments/ecro_wl84_104ka/descriptive104ka.py) was written after the frozen decision and changes no label. It covers the squared-error decomposition of the rendered first bounce and the per-start peak heights on the existing peak grid. It is labelled where it appears.

## Implementation Fact

- **Stage A — reproduction.**
  - The archived WL104K states, eval files and decision were authenticated against `evaluation/104K/manifest.json`.
  - The post hoc scene outputs, which had no frozen manifest, were hashed before any new evidence.
  - Held-out angular el1 of all 12 states (A / T / P × 4 starts), recomputed from the archived states, reproduces `evaluation/104K/eval.json` exactly (0.0 relative).
  - Scene replay identity holds in all three scenes.
- **Event ownership (code and numbers).**
  - **E1:** camera vertex on an ER object (sphere or cube) × next-event estimate of the single spot light. Local wi is the view side and wo the light side; the weight is rgb / d² times visibility.
  - **E2:** one R0-proposal bounce from the ER vertex to another surface × that surface's native direct light, weight 1 / pdf. The proposal is common to all arms, and F / p is recomputed per arm.
  - **Non-ER surfaces:** non-ER first vertices use the native BSDF in every arm.
  - **TOTAL:** E1 + fixed E2 + the archived R0 HIGH (events ≥ 3 of the R0 render), identical for all arms. TOTAL differences therefore come only from E1 and E2.
- **Stage B — matched extension.**
  - From every 12k state, 12,000 further steps of Adam at constant lr 3e-4 (the WL82 terminal rate), with gate projection.
  - The 12k Adam moments were not archived, so the moments restart from zero. Exact continuation was impossible (disclosed restart semantics).
  - Same budget for all arms. The WL104K convergence rule (a) is unchanged.
  - Stationarity: full-batch KKT-projected gradient and active gate constraints.
  - The extended states passed the WL104K reciprocity and passivity contracts (`tests104k --fitted`).
- **Stage C — scene-conditioned attribution.**
  - For each scene, an experiment-local Dr.Jit pass with the frozen camera, geometry and light traced 16 jittered samples per pixel on a new independent stream. It recorded every visible first-bounce lane on the ER objects: 81,405 / 105,398 / 85,611 lanes for iid_1 / hard_1 / iid_159.
  - At exactly those (wi, wo) pairs it evaluated:
    - the physical layered response (WL104D evaluator, depth 1024, 8,192 fresh walks per lane in two halves, new sequence range);
    - A / T / P for every start, 12k and extended.
  - **Correspondence check, passed in every region of every scene.**
    - Lane-predicted E1 region sums vs the rendered fixed-pass E1: models ≤ 2.0 % (hard_1 and iid_159 ≤ 0.3 %).
    - Physical response vs the archived layered reference E1: within 1.9 % or 3 combined SE.
  - The models were not refitted or selected with these data.
- **Stage D — independent validation.**
  - No independent material / lighting scene exists under the existing qualified scene contracts. The fixed-proposal inputs exist only for the three post hoc scenes, and the remaining WL104D scene (iid_0_L2) failed the WL104D scene-precision gate. No replacement scene was constructed.
  - The new independent evidence is Stage C's camera and walk streams.
  - The extended-state scene replay reuses the archived seeds (common random numbers) and is development evidence.

## Measurement

### Optimization (Stage B)

[Figure 1](../../evaluation/104K-A/fig01_optimization_closure_curves_and_stationarity.png); [extension_fit.json](../../evaluation/104K-A/extension_fit.json).

| | A | T | P |
|---|---:|---:|---:|
| Rule (a) at 12k (WL104K) | 100 % | 100 % | 53 % |
| Rule (a) after the extension | 100 % | 100 % | **83 %** (FAIL; ≥ 90 % required) |
| Max last-2,000-step change after the extension | 0.04 % | 0.15 % | 1.41 % |
| Median training-objective change 12k → extended | −0.09 % | −0.12 % | −0.91 % |
| KKT-projected gradient norm (start → 12k → extended) | 3.39 → 0.21 → 0.19 | 11.7 → 0.47 → 0.51 | 6.07 → 0.45 → 0.56 |
| Gate knots at 0 / at 1 (12k → extended) | — | 0 / 55 % → 0 / 55 % | 10 / 40 % → 11 / 41 % |
| Extension wall time (4 starts × 9 materials) | 278 s | 340 s | 550 s |

- P's remaining rule-(a) failures are two held-out materials: block_186 (0.8–1.4 %) and iid_191 (0.6–0.65 %). The other seven materials are ≤ 0.31 %.
- P's projected gradient does not decrease further. This is consistent with Adam / projection noise at a shallow plateau.
- **Optimization qualification: FAIL** (rule unchanged).

### Angular stability (Stage B)

[ext_angular.json](../../evaluation/104K-A/ext_angular.json).

| Contrast | Development 12k → extended | Held out 12k → extended |
|---|---|---|
| T − A | −14.71 → −14.69 % | −23.02 → −23.04 % |
| P − T | −44.04 → −44.15 % | −47.08 → −47.24 % |
| P − A | −52.29 → −52.40 % | −59.20 → −59.36 % |

- **Angular stability: STABLE.** Every change is ≤ 0.16 pp, with the same sign.
- **Pooled el1, 12k → extended:** A .07673 → .07665, T .06154 → .06147, P .03330 → .03315.
- **Response change on the evaluation lanes:** ≤ 0.14 % RMS for every arm and start.
- **WL104K practical-gain criteria re-applied to the extended states** (descriptive; does not change WL104K's frozen verdict): all met except the optimization rule.

### Scene utility across starts

[Figure 2](../../evaluation/104K-A/fig02_per_start_angular_and_scene_quality.png); [decision.json](../../evaluation/104K-A/decision.json).

P − A objects stable RMS per start (12-replicate jackknife 95 % CI), extended states. The 12k states agree within 1 pp everywhere.

| Scene | Event | Start 1 (generic) | Start 2 | Start 3 | Start 4 | Label |
|---|---|---|---|---|---|---|
| iid_1 | first bounce | −94.4 % | −94.8 % | −94.4 % | −93.8 % | STABLE_IMPROVEMENT |
| iid_1 | total | −96.9 % | −98.7 % | −96.7 % | −96.6 % | STABLE_IMPROVEMENT |
| hard_1 | first bounce | +3.5 % [0.0, 7.1] | **+16.5 % [12.4, 20.6]** | −4.1 % [−6.6, −1.5] | −2.7 % [−4.9, −0.5] | **MIXED** |
| hard_1 | second bounce | −32.4 % | −27.5 % | −27.2 % | −28.0 % | STABLE_IMPROVEMENT |
| hard_1 | total | +3.0 % [−0.6, 6.5] | **+15.8 % [11.8, 19.9]** | −4.4 % [−6.9, −1.9] | −3.1 % [−5.3, −0.9] | **MIXED** |
| iid_159 | first bounce | −34.1 % | −34.5 % | −31.7 % | −29.8 % | STABLE_IMPROVEMENT |
| iid_159 | total | −34.7 % | −34.7 % | −32.5 % | −30.7 % | STABLE_IMPROVEMENT |

iid_1's −94 % to −100 % values come from the noise-debiased estimator: P's error is at or below the replicate-noise estimate, which clips at −100 %.

### Scene-conditioned attribution

At the scenes' own first-bounce directions, using the 12k states. [Figures 3–5](../../evaluation/104K-A/index.html); `lane_attribution_*.json`.

Signed excess of the model over the physical response, Σ w (F_model − F_phys) / Σ w F_phys, per region. The walk-noise SE is ≤ 0.22 % of the region sum.

| Scene, region | A (4 starts) | P (4 starts) | Label (P) |
|---|---|---|---|
| hard_1 highlight core | −1.9 to −2.1 % | −1.8 to −2.1 % | MIXED |
| hard_1 highlight shoulder | +12.3 to +13.7 % | +9.9 to +11.2 % | ANGULAR_RESPONSE_OVERSHOOT |
| hard_1 grazing | −18.7 to −21.0 % | +6.3 to +9.5 % | ANGULAR_RESPONSE_OVERSHOOT |
| hard_1 lower sphere | −12.0 to −13.6 % | 0.0 to +2.7 % | ANGULAR_RESPONSE_OVERSHOOT |
| hard_1 cube | −5.3 to −6.5 % | **+13.7 to +18.8 %** | ANGULAR_RESPONSE_OVERSHOOT |
| iid_1 highlight core | −2.2 to −2.7 % | −0.1 to +0.2 % | ANGULAR_RESPONSE_OVERSHOOT (by rule; +0.04 to +0.19 %) |
| iid_159 highlight core | −4.6 to −5.1 % | −1.5 to −1.9 % | MIXED |

**Region L1 against the physical response** (start-mean):

| Scene | A | P | Walk-noise floor |
|---|---:|---:|---:|
| iid_1, all object pixels | 6.0 % | 0.74 % | 0.29 % |
| iid_159, all object pixels | 5.0 % | 2.8 % | 0.89 % |
| hard_1, all object pixels | 5.1 % | 4.9 % | 1.3 % |
| hard_1, highlight core | 3.2 % | 3.3 % | 0.8 % |

**hard_1 highlight core by mirror offset** (share of the region's physical contribution, 12k):
- About 11 % of the core's light comes from within 4° of the mirror direction; the median mirror offset is 10.6°.
- At 0.5–4°, P is too bright in starts 1–2 (bins +0.03 to +0.19 % of the region total) and nearly neutral in starts 3–4. A is too dark there.
- At 5–20°, both are too dark (−0.4 to −1.2 % per bin).
- The core's view polar spans 20–36° (median 28°). 30 % of its contribution lies within ±2.5° of an angular-evaluation probe polar (probes at 20° and 35°).

**Additional descriptive analysis (not predeclared)**, [figure 9](../../evaluation/104K-A/fig09_hard1_peak_pixels_and_peak_height_per_start.png), [descriptive.json](../../evaluation/104K-A/descriptive.json):

- **hard_1 squared-error change of the rendered first bounce, P vs A** (12k), as a share of A's squared error:

  | | Start 1 | Start 2 | Start 3 | Start 4 |
  |---|---:|---:|---:|---:|
  | All object pixels | +23 % | +36 % | −16 % | −12 % |
  | The 20 pixels with the largest increase | +28 % | +40 % | +5 % | +5 % |
  | The highlight core | +26 % | +39 % | −14 % | −9 % |

  The cube contributes +0.1 %.
- **Peak grid, hard_1, 20–35° probes, P vs physical.** Starts 1–2 overshoot the peak height by +1.4 to +6.3 %. Starts 3–4 are within −0.4 to +1.6 %. A undershoots at 5–20° (−5 to −7 %).
- **iid_1 and iid_159.** Squared error falls 100 % and 62–72 % in every start.

### Noise and variability

[Figure 7](../../evaluation/104K-A/fig07_noise_and_variability_accounting.png); [stageA.json](../../evaluation/104K-A/stageA.json).

| Source | Size |
|---|---|
| Angular reference half-split noise | 0.4 % / 0.5 % of A's el1 |
| P − A across-start range (angular) | 1.8 pp development, 0.6 pp held out |
| P − A material-bootstrap CI width | 60 pp development, 31 pp held out (3 / 6 materials) |
| Scene render-replicate jackknife SE | 0.5–1.8 % of the stable RMS (A, T, P), except iid_1 P, whose stable RMS is at the noise level, so a relative SE is undefined |
| Lane walk-noise SE of region sums | ≤ 0.22 % |

On hard_1, the across-start spread of P − A in the scene (−4 % to +16 %) exceeds the render noise by an order of magnitude.

## Observation

- **Optimization.** The extension leaves the angular result unchanged to 0.16 pp. P still drifts by 0.6–1.4 % per 2,000 steps on two held-out materials, so the unchanged rule (a) fails at 83 %.
- **hard_1's image regression is concentrated in a handful of highlight-peak pixels.**
  - It appears in starts 1–2; starts 3–4 improve the image.
  - The same starts overshoot the mirror-peak height on the independent angular peak grid at the incidences the highlight uses (20–35°).
  - In energy terms the overshoot is tiny (≤ 0.2 % of the highlight's light). The squared-error image metric weights the brightest pixels heavily, so it dominates there.
- **Where P helps in hard_1.** Energetically, P corrects A's darkening at grazing and on the lower sphere and its shoulder excess. It brightens the dark cube by +14–19 % in every start. The cube is too dark to matter in the squared-error metric (+0.1 %).
- **iid_1 and iid_159.** P removes most of A's scene-conditioned bias and L1 in every start: iid_1 L1 6.0 → 0.7 %, iid_159 5.0 → 2.8 %.

## Interpretation

- **Why held-out angular error falls while the hard_1 highlight worsens.** Mostly because the two metrics weight errors differently:
  - The angular el1 is energy-weighted L1 over many directions. P lowers it on hard_1 (−26 %).
  - The image metric is squared error over pixels, dominated by the few brightest highlight-peak pixels.
  - In those pixels P overshoots the mirror peak by a few percent, but only in starts 1–2. The independent peak grid shows the same start-dependent peak overshoot.
  - The attribution is therefore an angular-response error (peak height near the mirror direction, 0.5–4°) that is **start-dependent**, not Monte Carlo noise and not camera weighting. Lane correspondence ≤ 2 % and walk noise ≤ 0.22 % rule those out.
- **Contributing factor: incidence coverage.** The highlight's incidence (20–36°) lies between two probe polars (20°, 35°). The hard_1 highlight-core rule label is MIXED because the core's energy-weighted net excess is the same for A and P (−2 %). The image difference comes from the peak pixels, not from the net excess.
- **Start dependence.** P's fit is not unique at the mirror peak. Fits that are equally good on the training objective (all starts within 1.8 pp angularly) differ by about 5 pp in peak height. The scene-level outcome of hard_1 therefore depends on the fitting start.
- **Scope.** Neither the angular nor the scene evidence shows that a converged P would remove the start dependence, and the optimization rule remains unresolved.

## Unresolved Question

- Whether additional optimization would bring P to rule-(a) convergence (block_186, iid_191), and whether the peak-height start dependence then shrinks. Not run beyond the matched extension.
- Whether a peak-sensitive criterion would select a stable P fit. Excluded: no new loss or start selection by scene evaluation was allowed.
- Scene utility on an independent material / lighting case. No qualified case exists under the current scene contracts.
- E2 attribution at the level of individual directions. Only E1 was attributed; E2 improves in every start of every scene.
- Sampling and pdf for the gated response, which remains a precondition for any native integration.

## Architecture Verdict

| Item | Result |
|---|---|
| Optimization qualification | **FAIL** (P rule (a) 83 % after a matched 12k-step extension; A, T 100 %) |
| Angular fidelity stability | **STABLE** (all contrasts within 0.16 pp; P − A −52.4 % / −59.4 %) |
| Scene-conditioned attribution | Lane correspondence PASS in all scenes. hard_1 highlight core **MIXED** by the frozen rule. Descriptively, the image regression is a start-dependent mirror-peak overshoot (starts 1–2). |
| Scene utility across starts (extended) | iid_1 **STABLE_IMPROVEMENT**; iid_159 **STABLE_IMPROVEMENT**; hard_1 **MIXED** (E1 and total; E2 STABLE_IMPROVEMENT) |
| Remaining representation limitations | Start-dependent mirror-peak height; residual grazing error; weakly determined gates on low-energy components |
| Integration work justified (frozen rule) | **UNRESOLVED** (optimization FAIL and hard_1 MIXED; no STABLE_REGRESSION) |
| Candidate adoption readiness | **NOT READY** (gated-response sampler / pdf not designed; experiment-local evaluator only) |

This answers the completion question. ER-v1-SDM improves held-out angular error while worsening the hard_1 highlight because:
- the energy-weighted angular metric hardly registers a few-percent mirror-peak overshoot;
- the squared-error image metric is dominated by exactly those peak pixels;
- the overshoot occurs only in two of four equally good fits.

The scene advantage is stable in two scenes and start-dependent in hard_1. Under the frozen rule, this does not establish that further integration work is justified. WL104K's verdict is unchanged, and ER-v1-SDM is not adopted. Any further step (more optimization, a peak-sensitive criterion, sampler design or rejection) is an owner decision. No new architecture hypothesis is started.

## Reproducibility

- **Commits.** Freeze `4201954`. The report commit adds `descriptive104ka.py`, `evaluation/104K-A/`, this worklog and the living-document note.
- **Environment.** Local RTX 5080, `external/8dna26/.venv-cu128` (Python 3.11.9, Torch 2.8.0+cu128, Mitsuba 3.5.1, Dr.Jit 0.4.6).
- **Timing.** Chain 17:36–18:12 KST: stage A 3 s; extension 19.5 min; contracts and evaluation 0.5 min; lanes 12 s; extended scene replay 16 min.
- **Seeds.**
  - Camera lanes 1041200 + scene index.
  - Lane walks 1041210 + scene index, pair0 2^29 + 4M × scene index (8,192 walks per lane).
  - Bootstrap 1041140.
  - Scene replay seeds as in WL104I.
  - Optimizer: restart semantics as above.
- **Raw outputs.** `results/ecro_wl84_104ka/primary/`: stageA.json, ext (fit.json, states.npz), tests_fitted_ext.json, ext_angular.json, lanes/<scene> (attribution.json, lanes.npz, maps.npz), scenes_ext/<scene>, decision.json, logs, status.txt.
- **Reviewer package.** [evaluation/104K-A/index.html](../../evaluation/104K-A/index.html): figures 1–8 (MEASURED) and figure 9 (additional descriptive), review copies of the JSON outputs, `manifest.json` and `figures_manifest.json`.

```bash
cd experiments/ecro_wl84_104ka
bash chain.sh ../../results/ecro_wl84_104ka/primary
PY=../../external/8dna26/.venv-cu128/Scripts/python.exe
PYTHONUTF8=1 $PY -B figures104ka.py ../../results/ecro_wl84_104ka/primary ../../evaluation/104K-A
PYTHONUTF8=1 $PY -B descriptive104ka.py ../../results/ecro_wl84_104ka/primary ../../evaluation/104K-A
```
