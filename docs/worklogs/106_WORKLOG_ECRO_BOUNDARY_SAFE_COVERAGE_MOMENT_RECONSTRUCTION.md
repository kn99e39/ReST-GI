# Worklog — ECRO Boundary-Safe Reconstruction: Coverage-Moment-Constrained Kernels (2026-11-19)

Batch WL106, Denoising Track, branch `Denoise_Reconstructor` (existing development line; no new branch or worktree).
Regular batch: it introduces a new reconstruction mechanism for the WL105 boundary failure.
The frozen baseline is WL105 (commit a3c4069); its code, data, checkpoints and worklog are not modified.
The protocol is `experiments/ecro_denoise106/protocol/denoise106.json`.

## Scope (user brief, 2026-11-19)

Investigate, and conditionally implement, ONE boundary-safe reconstruction mechanism for the ECRO single-frame
Denoising Track. Hypothesis to falsify: *systematic silhouette bias in the current ECRO reconstructors is caused
substantially by geometry-inconsistent spatial radiance mixing, and can be reduced by constraining the reconstruction
operator using the existing subpixel geometry information.*

The batch has five parts:

- A: qualify the WL105 failure on frozen inputs.
- B: synthetic boundary contracts with known targets and a gate.
- C: select one architecture-preserving mechanism, or stop with INPUT_REPRESENTATION_INSUFFICIENT.
- D: R4 (R3-derived) trained with the same recipe and evaluated on the WL105 held-out views.
- E: qualify the evaluation targets with more independent frozen-ECRO samples. The 10% rule is not relaxed.

The following are not changed or added:

- ECRO, LOW/HIGH estimation and the HIGH table;
- the contact-region work;
- temporal history, new material classes, larger kernels or capacity, post-hoc energy compensation;
- evaluation thresholds;
- branch merges and automatic adoption.

## Pre-registration 1 (Parts A and B, before any diagnostic or synthetic result)

**Mechanism candidate.** The candidate is fixed before the qualification evidence, from the structure of the
WL105 operator. A softmax kernel applied to pixel means has expectation sum_q w_q E[x_q]. Under uniform pixel
sampling, E[x_q] is the area-weighted mixture of the surfaces in pixel q. For per-surface-constant radiance, that
expectation equals E[x_p] for every radiance assignment only if the kernel's coverage moment matches the pixel:
sum_q w_q c_q = c_p.

The candidate operator (`tilt106.py`) replaces each predicted kernel by its minimum-KL modification satisfying this
moment over the 10 coverage channels: the 9 existing 8 x 8 shape coverages plus the implied miss fraction.

- Exact part: neighbours holding a surface absent from p get zero weight, as implied by w >= 0.
- Tilted part: an exponential tilt applies on pixels covered by >= 2 surfaces. It is solved by Newton iterations.
- Differentiable part: one final Newton step gives the implicit derivative.

No parameter is added. In the fused Triton inference path (`kpn_triton106.py`), the operator equals the
reference/training path.

**Operator tests (`tests106.py`, all pass on LabServer63)**:

- tap order equals the unfold order;
- moment residual <= 1e-5 on generic synthetic coverage; <= 2e-3 at extreme-point pixels of a 0.6 px strip, where the
  exact solution needs lambda -> infinity and is approached asymptotically;
- windows with one coverage vector are bit-identical to the unconstrained operator;
- constant field exact;
- Triton solver equals the torch solver (weights within 3.6e-5);
- fused equals reference (1.2e-6 relative);
- autograd gradient equals finite differences (7e-4 relative);
- R4 state_dict, initialisation and parameter count (1,055,624) identical to R3;
- deterministic; inputs unmodified; HDR finite.

Two development fixes were made before any evidence:

- Newton step cap: without it the dual Newton step diverged when the kernel mass sat on a single pixel.
- Exact support restriction: it replaced an eps-regularised tilt that left 3e-4 residual at vertex pixels.

`proto106.py` is a development timing prototype on validation views; its numbers are not evidence.

**Part A plan.** Hashes and stored-output reproduction are checked first. The exact decomposition is
y - T = (sum w T - T) [mixing] + sum w (x - T) [weight-noise]. It is reported per ROI, per sphere-coverage bin and
per kernel coverage moment, for frozen R1/R2/R3 on the six evaluation views. The diagnostic constraint on frozen R3
logits is applied on the eight validation views only.

**Part B plan and gate (SG1-SG3)** are recorded in the protocol before `synth106.py` is run.

## Run (LabServer63, 2026-11-19; code at commit 10892c0, chain `chain106a.sh`)

`tests106.py` passed (20/20). `qualify106.py` and `synth106.py` both exited 0. The synthetic gate block written by
`synth106.gate` reads `SG1 true, SG2 false, SG3 true, pass false`.
Evidence and figures: `evaluation/106/` (manifest `manifest106.json`, hashes equal to the server copies).

### Part A — qualification of the WL105 failure (frozen inputs and checkpoints)

**Implementation fact.**

- Weights hashes equal WL105 (R1 cfa468c7…, R2 3e420609…, R3 d51b6919…).
- The data manifest hash matches.
- Every eval/val noisy, target and aux file hash equals `run_manifest` (0 mismatches).
- Recomputed R1–R3 outputs equal the stored WL105 evaluation outputs to 5.7e-8 relative.

**Code facts (semantics).**

- The frame program jitters each sample uniformly in the pixel square: `pt86.gen_rays`,
  x = (j + U) · 2/res − 1.
- The guide counts shape hits on the 8 × 8 midpoint grid of the same square: `render105.aux_buffers`.
- Guide and radiance are therefore aligned, a box-filter estimate of the same area fractions.
- The kernel is a softmax over 225 taps with reflect padding and unfold tap order.
- Nothing in the WL105 application restricts which surfaces a tap may come from. The operator can mix radiance
  across distinct surfaces at partially covered pixels, and it does: the mean coverage-moment mismatch on the
  silhouette is L1 0.20–0.35 for every arm.

**Measurement (evaluation views, WL105 two-half target, pooled over views and replicates, at 4 / 16 / 32 spp).**
The silhouette signed bias splits exactly into a mixing term and a weight-noise term:

| arm | total | mixing term sum w T − T | weight-noise term sum w (x − T) |
|---|---|---|---|
| R1 | +5.71 / +5.33 / +4.75% | +5.79 / +5.35 / +4.75% | −0.08 / −0.02 / 0.00% |
| R2 | +3.71 / +3.38 / +3.14% | +3.85 / +3.40 / +3.15% | −0.14 / −0.02 / −0.01% |
| R3 | +2.23 / +1.90 / +1.65% | +2.52 / +1.98 / +1.68% | −0.29 / −0.08 / −0.03% |

R3 material boundary: total +0.74 / +0.60 / +0.47%; mixing +1.12 / +0.74 / +0.63%.

**Observation 1.** The bias is the kernel applied to the expectation, i.e. wrong support or weights. It is not
noise-correlated weighting and not a learned bias of the expected value.

**Observation 2.** The direction does not fit plain cross-surface import:

- The R3 kernels' signed sphere-coverage moment is negative from coverage ≈ 0.3 upward (−0.04 to −0.22 at 4 spp).
  The kernels import non-sphere surfaces into the sphere's partial pixels, yet the bias is positive.
- The bias is positive in every coverage bin (R3, 4 spp: +1.6 to +3.2%).

**Validation-view diagnostic (8 views × 4 replicates; frozen R3 logits passed through the constraint; never on
evaluation views).** The coverage moment becomes exact (L1 0.000):

- Silhouette bias at 4 / 16 / 32 spp **rises**: R3 +1.22 / +0.88 / +0.85% → constrained +3.62 / +2.78 / +2.63%.
  Mixing term +3.79 / +2.98 / +2.59%.
- Material-boundary bias **falls**: +1.32 / +0.99 / +0.83% → +0.07 / +0.19 / +0.18%.
- Primary bias stays below 0.06% either way.

So on silhouette pixels, a kernel that imports exactly the pixel's own surface mixture from its neighbours still
over-predicts the target by 2.6–3.8%. The partially covered silhouette pixel is darker than the coverage-weighted
combination of nearby full-sphere and full-background radiance. Its sphere portion is the grazing rim, consistent
with the thin dark rim line seen in the WL105 post hoc crops. That darkening is intra-surface radiance variation
between neighbour and pixel, which the coverage moment cannot see. At material boundaries between planar surfaces
the error is genuinely cross-surface and the constraint removes it.

### Part B — synthetic boundary contracts

**SG1** (constant field) passes: max error 1.8e-6.

**SG3** (exact-area truth, generic scenes, piecewise-constant fields) passes. Band signed energy, unconstrained →
constrained:

| scene | K1 box | K3 random |
|---|---|---|
| disk | −3.5% → −0.07% | −3.0% → −0.06% |
| disk over empty | −5.8% → −0.12% | — |
| edge 17° | +0.14% → 0.00% | +2.4% → +0.01% |
| corner | +0.9% → +0.01% | +2.3% → +0.02% |

Near-boundary full pixels are within 1e-4, and the coverage-moment mismatch goes from 0.50–0.57 to 0.000.

**SG2** (guide-consistent truth, max |relative bias| ≤ 1e-3 "in every scene") **fails**:

- It fails only in the report-only thin-strip probe: 1.5–3.5e-3 at extreme-point pixels, moment residual 2.5e-3.
  There the exact solution requires λ → ∞ and is approached asymptotically.
- Every gated scene is ≤ 5.0e-4, and the edge-0 and wide-strip probes are ≤ 1e-4.
- The operator tests had pre-registered ≤ 5e-3 for exactly this extreme-point case, but the gate text did not
  exclude report-only scenes. This inconsistency is mine (pre-registration). The gate is applied as frozen.

**Identifiability probes (report only).**

- Grid-aligned edge at offset 0.3 px: the 8 × 8 guide says 0.25 and the true area is 0.30. The constrained band
  bias is −8.9%; unconstrained is +27%. This is a guide-quantization limit.
- Rim darkening of A within 6 px: the constraint does not help. Band energy goes disk +28% → +33%,
  edge +34% → +35%, corner +22% → +22% (box kernel); near-full pixels +2.7 to +3.8%. This is the same signature as
  the real-data validation diagnostic.
- K4, the frozen R3 LOW network on Monte-Carlo inputs (out of distribution):
  - per-surface constants: band −23% → −1.0% at 4 spp, −14% → −0.1% at 32 spp;
  - rim field: −4.4% → +16% at 4 spp, −3.1% → +12% at 32 spp. The unconstrained network partly cancelled rim
    darkening with cross-surface import, and the constraint removes that cancellation.

## Stop (frozen rule) and final report

**Implementation Fact.**

- Parts A and B ran on frozen WL105 inputs and checkpoints. Nothing in WL105 was modified.
- R4 was **not trained**. No new targets were rendered: the Part E plan (48 halves) was not executed.
- No evaluation view was read by any constrained operator.
- The pre-registered synthetic gate failed on SG2, and the protocol stops the batch before R4 training.

**Measurement.** See Parts A and B above.

**Observation.**

- WL105 silhouette bias is ≥ 100% explained by the kernel's action on the expectation.
- The R3 kernels mix surfaces, but importing background lowers rather than raises their silhouette values.
- With exact coverage moments the silhouette bias grows on validation views, while the material-boundary bias
  nearly vanishes.

**Interpretation.**

- The hypothesis splits:
  - "systematic silhouette bias is caused by spatial mixing of the expected radiance": supported descriptively
    (mixing term ≈ total);
  - "it is geometry-inconsistent cross-surface mixing correctable with the existing coverage guide": contradicted by
    the frozen-kernel diagnostic for silhouettes, but holds for planar material boundaries.
- The silhouette residual is intra-surface (rim) radiance variation inside partially covered pixels.
- A trained R4 could in principle still place its weight on same-coverage pixels along the rim, where the bias would
  be small. That learned adaptation is the untested part.

**Unresolved Question.**

- Does training under the constraint (R4) learn rim-tangential kernels that remove the silhouette bias? Or must the
  input representation change, for example per-surface or rim-aware subpixel radiance information? The latter
  would be an owner decision on the rendering contract.
- Target qualification (Part E) remains planned but unexecuted.

**Architecture Verdict.** **INCONCLUSIVE**: R4 was not trained, so the trained hypothesis was not tested.

- The stop is the frozen synthetic-gate rule (SG2). The SG2 failure lies in a report-only identifiability probe and
  is not a defect of the operator on generic geometry.
- Diagnostic evidence points away from cross-surface mixing as the silhouette cause. It does not establish
  INPUT_REPRESENTATION_INSUFFICIENT, because a trained R4 is not excluded from solving it.
- R4 is not adopted. No other mechanism is started.

Continuing would need an owner-approved amendment. The candidate amendment would restrict SG2 to the gated scenes,
consistent with the operator-test threshold for extreme-point pixels, and leave every other rule unchanged.

**Reproducibility.**

- Code: `experiments/ecro_denoise106/` (pre-registered at 10892c0; `report106.py` adds descriptive figures only).
- Server: `LabServer63:~/rest_sss101a/results/ecro_denoise106/` (tests.log, qualify/, synth/, review/).
- Wrapper `run106.sh` = `run105.sh` with the output directory renamed.
- Evaluation: `evaluation/106/` with `manifest106.json`.

## Closure (owner decision, 2026-11-19)

The research owner chose to close WL106 with the frozen **INCONCLUSIVE** verdict.

- SG2 is not amended. R4 is not trained. The stopped evaluation (Part E targets, R4 training, comparison) is not
  resumed.
- Every positive and negative finding above is preserved as recorded, including:
  - the mixing-term attribution;
  - the material-boundary correction by the constraint;
  - the silhouette increase under the constraint on validation views;
  - the SG2 pre-registration inconsistency;
  - the identifiability probes.
- The prepared but unrun scripts (`train106.py`, `target106.py`, `eval106.py`, `analyze106.py`) remain in the
  repository as unexecuted code. They carry no evidence.

**Next scientific question (owner direction; not an implementation authorisation).** The next batch will
investigate whether the remaining grazing-rim reconstruction error originates from:

- insufficient input information; or
- limitations of the current nonnegative spatial-kernel operator.

Constraints on that batch:

- It must not assume that new input channels are necessary.
- It must first determine whether the existing coverage, normal, depth and noisy component information can
  represent the required radiance variation.
- No successor architecture is implemented until the owner approves its separate contract.
