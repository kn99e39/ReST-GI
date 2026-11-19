# Worklog 104B — Learned-ER material-fidelity error attribution (2026-11-18)

## Authorization and freeze, 2026-11-18 KST

The user's brief requests a bounded, diagnostic-only attribution of the physical-fidelity error of the frozen learned-ER material representation. The error was measured in WL104 (object relative L1 0.035 / 0.027 / 0.030 against the layered BSDF) and in WL104A (material TOTAL object energy −3.39 / +0.87 / +0.15 %, mostly in LOW). The question is where the error originates:
- transport event;
- angular distribution vs energy magnitude;
- learned approximation vs the representation or fit;
- compiler / runtime;
- rendering or reference semantics.

Nothing is retrained and no correction is implemented. The batch identity is a suffix of WL104 (AGENTS.md: a dependent evidence-gap follow-up of WL104 / WL104A); no WL104B exists on any branch. It runs on branch `ecro_wl84_scene`, on the local RTX 5080 by the owner's answer. The three image workers run as parallel processes, as planned before freeze, after the owner's resource-use remarks in WL104 / WL104A.

**Sources and provenance.**
- **States.** The frozen WL83 states are ER_L (compiler seed 8301) and ER_DF (WL83 per-material direct fit, 12,000 steps, cert metric). Their compiled states equal WL84's audit hashes.
- **Response data.** The WL83 high-precision layered eval set (`eval83_<set>`), fit targets (`fit82_<set>`) and per-lane metrics (`lanes83_<set>`) were copied read-only from LabServer63. They are 640 MB, and their sha256 values were spot-checked against the server.
- **Code.** All WL81–WL84 code is imported unchanged: closure82 with the closure83 robust-refraction patch (as WL83 evaluated), closure83_dr, layered81 / data81.ref_eval, eval82.group_metrics, the render83 plugins, and the WL102 / WL104 tracer and scene.

**Contracts and arms.**
- **One tracer.** All image arms use the same tracer (pt102, D = 6, NEE at every vertex). Only the scene BSDF differs, so the events (E1, E2, HIGH = hits 3–6, TOTAL) are identical by construction.
- **One response convention.** Every evaluator uses eval = f cos θ_o, with wi on the camera side, in the shading frame.
- **ER_DF's role.** ER_DF separates the compiler's generalization from the ER closure family together with its fitting procedure.

[Protocol](../../experiments/ecro_wl84_104b/protocol/wl104b.json) — measurements:
- **Images.** 32 paired replicates × 4,096 spp for each arm.
- **Event decomposition.** Signed event energies, which are exactly additive, plus a stable mean-square decomposition with cross terms. The cross-replicate U-statistic makes the identity exact.
- **Response metrics.** Energy (albedo ratio) and shape (WL82 eshape = albedo-weighted total variation of normalized responses) on the WL83 eval set, 8 θ × 2 φ per material.
- **Angular atlases.** Polar mirror-direction grids (peak, centroid, RMS radius) and hemisphere grids.
- **Correspondence.** Centre-ray prediction of the E1 image error, split into an energy part and a shape part, compared with the rendered E1 error.
- **Pipeline contracts:**
  - C0: state hashes;
  - C1: the render contract;
  - C2: event sums vs Mitsuba depth 7;
  - C3: Dr.Jit vs torch closure, and ER reciprocity;
  - C5: WL83 per-lane reproduction and the WL84 object error;
  - S_target: WL83 fit targets vs a new layered estimate.

The classification rules are fixed in the protocol: physical-reference limitation, rendering-contract error, fit/training vs qualified representation limitation, mixed, inconclusive.

### Development checks before freeze

- **Tests.** `tests104b.py` passes 15 checks:
  - solid angles and Lambert integration;
  - polar-cap geometry and the mirror convention;
  - Dr.Jit ≡ torch closure (≤ 2.6e-7) and ER reciprocity;
  - layered isotropy under azimuthal rotation;
  - the U-statistic decomposition identity, unbiasedness and agreement with WL104A's estimator;
  - energy additivity and the energy / shape split algebra.
- **Pipeline smoke** (non-evidence, reduced counts). In the smoke, the WL83 per-lane metrics reproduce to about 1e-8.
- **Finding 1: walk depth.** The in-scene layered plugin (`render81.Lay81`) calls `layered_eval` / `layered_sample` with layered81's default walk depth of **64**. The WL83 fitting / evaluation reference (`data81.ref_eval`) uses **1024**.
- **Finding 2: hard_1 E1 offset.** In two independent smokes, the in-scene layered E1 for hard_1 sat about 4 % below the walk-average prediction (z ≈ −3, 8 passes).
- **Diagnostic arm added before freeze.** I added `LAY1024`: the same plugin code with walk depth 1024, as an experiment-local subclass; shared code is untouched. With it come:
  - depth-64 controls on the WL83 eval set and at centre rays;
  - per-plugin render contracts against the reference of each plugin's own depth;
  - the comparison pairs ER vs LAY1024 (primary attribution), ER vs LAY (WL84 / WL104 semantics) and LAY vs LAY1024 (the reference's own truncation);
  - a frozen physical-reference-limitation rule.

  In the second smoke the LAY1024 plugin matched ref_eval (ratio 1.001). The depth-64 plugin was again about 4 % low. Yet ref_eval itself changed by only 0.5 % between depths 64 and 1024, so the cause is not settled; the evidence run measures it.
- **Smoke disclosure.** Both smokes printed a classification string (non-evidence, tiny counts). The rules were already in the protocol and were not changed in response to it.

## Evidence, 2026-11-18 / 19 KST

### Implementation fact

The chain at freeze commit `ea2da2e` (clean experiment directory) ran three workers in parallel from 21:47 to 23:09 KST on the local RTX 5080, and the analysis completed. Every image produced all 32 replicates of all four arms. Nothing in the ER, ECRO or WL81–84 code was changed. The only additions are experiment-local instrumentation, the diagnostic depth-1024 plugin subclass and the reference renders.

| Contract | Result |
|---|---|
| C0 | ER_L and ER_DF state hashes equal the WL84 audit in all 3 materials |
| C1, ER render contract | tracer E1 = closure prediction to ≤ 2e-7 relative |
| C1, layered render contracts | object E1 plugin / own-depth reference ratio 0.993–1.001, \|z\| ≤ 1.03 |
| C2 | tracer TOTAL vs Mitsuba depth 7 for LAY, ER_L and ER_DF: object mean ratios 0.9997–1.0013, rel L1 below every limit |
| C3 | Dr.Jit ≡ torch closure; ER reciprocity holds |
| C5a | WL83 per-lane el1 / ealb / eshape reproduced to ≤ 1.2e-8 |
| C5b | WL84 object material error reproduced: 0.0346 / 0.0266 / 0.0293 vs 0.035 / 0.027 / 0.030 |
| S_target | WL83 fit targets unbiased: energy ratio 1.000 / 0.9999 / 0.9989 (SE ≤ 0.0009); rel L1 at the noise expectation |

**The smoke's walk-depth signal was noise.**
- Centre-ray E1 ratio, depth 64 / depth 1024: 1.0001 / 0.9999 / 0.9997 (\|z\| ≤ 1.1).
- WL83-set albedo ratio LAY64 / LAY1024: within ±0.15 %.
- LAY vs LAY1024 in the images: 0.00 % in every event and ROI.

So the in-scene layered reference of WL84 / WL104 is **not** limited by its walk depth on these materials. The ±4 % in the 8-pass smoke was small-sample noise. The physical-reference-limitation rule is not triggered.

### Measurement

**Event table** (ER_L vs the depth-1024 layered reference, objects; ER_DF in parentheses). Energy is signed, as % of the layered TOTAL. MS shares are fractions of the stable mean square of the TOTAL difference; the cross terms are reported in the metrics and the identity residual is about 0.

| Image | E1 energy | E2 energy | HIGH energy | TOTAL energy | TOTAL rel RMS | MS share E1 / E2 / HIGH | LOW share of MS |
|---|---:|---:|---:|---:|---:|---|---:|
| iid_1_L2 | −1.13 (−2.04) | −1.94 (−0.45) | −0.32 (−0.22) | **−3.38** (−2.71) | 3.89 % (4.14 %) | 0.59 / 0.24 / 0.01 | 0.95 |
| iid_0_L2 | +0.56 (+0.46) | +0.35 (+0.37) | −0.06 (−0.06) | **+0.86** (+0.77) | 3.15 % (2.47 %) | 0.75 / 0.10 / 0.00 | 1.00 |
| hard_1_L1 | −0.13 (−0.36) | +0.16 (+0.14) | +0.12 (+0.07) | **+0.15** (−0.15) | 2.70 % (3.05 %) | 0.93 / 0.03 / 0.01 | 0.97 |

Notes on the table:
- **Precision.** All energy intervals are ≤ ±0.02 points.
- **Reference choice.** ER_L vs the WL84 in-scene reference (LAY, depth 64) gives the same numbers to two decimals.
- **ROI sign changes.** Within hard_1 the object total hides opposite-sign ROIs:
  - highlight core: E1 −1.72 %;
  - highlight ring: E1 **+3.80 %**, TOTAL +4.31 %, RMS 5.69 %.
- **Grazing ROI** (all images): E2 and HIGH dominate.
  - Energy: iid_1 −3.35 / −1.75 %, iid_0 −2.41 / −1.28 %, hard_1 −1.88 / −0.95 %.
  - LOW share of MS: 0.45–0.57.

**Response level** (WL83 eval set, 8 θ × 2 φ per material, against the WL83 high-precision layered reference):

| Material | Probes: energy / shape / both / none (ER_L) | Mean el1 ER_L / ER_DF | Mean eshape ER_L | Albedo ratio range (φ 0) | eshape noise |
|---|---|---:|---:|---|---:|
| iid_1 | 0 / 4 / 12 / 0 | 0.085 / 0.087 | 0.077 | 0.950–1.000 (lowest at θ 60–70°) | ≤ 0.003 |
| iid_0 | 0 / 16 / 0 / 0 | 0.062 / 0.059 | 0.061 (0.032 → 0.116 toward 86°) | 0.993–1.004 | ≤ 0.008 |
| hard_1 | 0 / 8 / 8 / 0 | 0.115 / 0.116 | 0.115 (0.043 → 0.283 toward 86°) | 0.964–1.011 | ≤ 0.015 |

No probe in any material has an energy-only error. Every probe has a resolved shape error.

**Angular grids** (around the mirror direction):
- **Peak position:** off by ≤ 0.4° (hard_1), up to 2.4° (iid_0 at 86°).
- **Lobe width (RMS radius):** within ±6 %.
- **Peak and cone energy:**
  - iid_1 at 60°: cone 0.87×, peak 0.87×.
  - iid_0 at 86°: peak 1.38×.
  - hard_1 at 60°: peak 1.29×; at 86°: cone 0.77×.
- **Hemisphere TV against layered:** 0.02–0.17, against a layered noise floor of 0.001–0.016.

**ER anisotropy** (φ 0 vs 90°, albedo): ≤ 0.1 % for ER_L. ER_DF reaches 2.7–3.4 % at 80–86° on hard_1.

**Correspondence** (centre-ray response prediction vs the rendered E1 difference, objects eroded):

| Image | r | Slope | Shape part of predicted MS | Energy part of predicted MS |
|---|---:|---:|---:|---:|
| iid_1 | 0.996 | 0.961 | 0.83 | 0.08 |
| iid_0 | 0.982 | 0.960 | 1.03 | 0.02 |
| hard_1 | 0.950 | 0.886 | 0.95 | 0.07 |

- **Established.** Correspondence is established in all three images for ER_L and for ER_DF.
- **Energy part in sums.** It can still carry a share of a signed ROI sum: iid_1 objects −10.3 of −17.3 predicted.
- **Rendered vs predicted ring sums.** iid_1 −1.32 vs −1.32; iid_0 0.85 vs 0.86; hard_1 13.52 vs 13.45.

**Fit vs learned:**
- **Response level.** ER_DF / ER_L mean el1: 1.03 / 0.96 / 1.01.
- **Image level.** Object TOTAL RMS: 1.06 / 0.78 / 1.13.
- **Rule outcome.** The DF is not clearly better in any material.
- **Convergence.** The direct fits converged: set median loss changed < 0.1 % over the last 2,500 of 12,000 steps (iid 0.06950, hard 0.12612).

### Observation

- **Atlases.** [atlas_test83_hard_1_L1](../../evaluation/104B/atlas_test83_hard_1_L1.png) shows the same signed angular pattern for ER_L and ER_DF against layered:
  - at θi 20°, a deficit at the lobe centre with an excess around it;
  - at θi 60°, a too-strong, too-narrow core along the mirror direction with deficits in the shoulders;
  - at θi 80°, an excess in the broad off-specular region and a deficit near the grazing core.
- **E1 correspondence.** [Correspondence](../../evaluation/104B/correspondence_E1.png): the rendered E1 difference maps (sphere highlight centre and ring, cube gradient) are reproduced almost exactly by the centre-ray prediction. The shape part carries the spatial structure; the energy part is weak and broad.
- **Event maps.** [Event maps](../../evaluation/104B/events_test83_hard_1_L1.png):
  - the E1 differences carry the highlight ring and core;
  - E2 differences sit on the grazing cube faces and in the floor reflection;
  - HIGH differences are small.

### Interpretation

Direct evidence:
- **WL104A's LOW-dominant interpretation is valid, and LOW splits into E1 and E2 by region.** E1 carries most of the object error structure: 59–93 % of the stable MS of the TOTAL difference. E2 adds a comparable energy share on iid_1 and dominates at grazing pixels together with HIGH.
- **The error is an angular-distribution (shape) error, not an energy-magnitude error.**
  - Lobe position and width are nearly right.
  - The errors are in lobe profile: core height vs shoulders and off-specular tails.
  - Albedo errors are secondary: up to −5 % at 60–70° on iid_1 and −4.8 % at 86° on hard_1, always together with shape errors.
- **The response-level error predicts the rendered E1 error** in sign, location and scale (r ≥ 0.95, slope 0.89–0.96).
- **The earliest stage carrying the discrepancy is the ER closure family with its frozen direct fit (ER_DF).**
  - It is not the compiler: ER_L ≈ ER_DF at both the response and image levels.
  - It is not the compiled runtime or the renderer (C1 and C3 are exact).
  - It is not the fit targets: they are unbiased.
  - It is not the physical reference: it is consistent across walk depths and with Mitsuba.

Inference: the converged per-material direct fit reproduces the same lobe-profile mismatch. This suggests that the ER closure family, under the WL83 fitting objective and direction weighting, cannot represent the layered lobe profile of these materials. It is not proof of capacity: another objective, initialization or weighting of the same family was not tested.

### Unresolved question

- **Representation vs fitting objective.** Whether the mismatch is a hard limit of the family (8 GGX / Lambert lobes, refraction-aware inner group, attenuation knots) or of the fitting objective cannot be separated. The objective's direction weighting does not emphasize the grazing / shoulder regions that dominate here.
- **E2 at grazing.** The E2 / HIGH energy deficit at grazing (−2 to −3.4 %) is consistent with the grazing-direction albedo deficits. Its per-path attribution is not separately measured.
- **Coverage.** Three materials, one light each.

### Architecture verdict

- **REPRESENTATION_LIMITATION_SUPPORTED, qualified.** It holds for the frozen WL83 direct-fit procedure and is not a capacity proof. The evidence is ER_DF ≈ ER_L in all three materials, with unbiased targets and all contracts passing.
- **Ruled out:**
  - RENDERING_CONTRACT_ERROR;
  - a physical-reference walk-depth limitation;
  - FIT_OR_TRAINING_LIMITATION of the learned compiler.
- **Next investigation.** The evidence is sufficient to say what kind it would be: a representation or fitting-objective investigation of the ER closure's lobe profile, at grazing and near-specular regions. A compiler / training or implementation correction is not indicated. No correction is authorized or implemented.

### Reproducibility

- **Commits.** Branch `ecro_wl84_scene`, freeze `ea2da2e`; the report commit adds this section, `evaluation/104B` and the living-document lines.
- **Environment.** Local RTX 5080, `external/8dna26/.venv-cu128` (Python 3.11.9, torch 2.8.0+cu128, Mitsuba 3.5.1 / Dr.Jit 0.4.6). Three worker processes in parallel.
- **Seeds:**
  - images: 1049000 + 10000 i + r;
  - grids: 1043000 + 1000 i + …;
  - centre: 1044000–1048000 + i;
  - targets: 1042010 + i;
  - depth controls: 1042500 / 1044500 + i;
  - Mitsuba ER_DF depth 7: 104990 + 10 i + h.
  - Full budgets are in the protocol.
- **Commands**, from `experiments/ecro_wl84_104b`:
  - `<venv> tests104b.py`
  - `<venv> chain104b.py results/ecro_wl84_104b/evidence evaluation/104B`
- **Raw data** (gitignored), under `results/ecro_wl84_104b/evidence/<image>/`: `contracts.json`, `response_wl83set.json`, `response_grids.npz`, `centre.npz`, `rep/r###.npz` (LAY, LAY1024, ER_L, ER_DF: e1, e2, h), `mitsuba_ERDF7.npz`; plus `logs/` and `chain.json`.
- **Inputs.** Copies are in `results/ecro_wl84_104b/inputs63/`.
- **Reviewer artefacts.** `evaluation/104B/`: metrics, manifest, gallery.
