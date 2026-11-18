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
