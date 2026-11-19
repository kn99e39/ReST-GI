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
