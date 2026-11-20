# Worklog 104G — Certified ER fidelity recovery (2026-11-20 KST)

## Implementation Fact

Preflight found a clean worktree in `C:\Projects\ReST-GI`, branch `ecro_lr3_adapter`, HEAD `a0966dcd00daff22ff07bf8107e99a9173f25808`. GitHub resolved the old configured origin `kn99e39/NeuralShaderDemo` and requested `kn99e39/ReST-GI` to the same repository ID `1379024934`. Both URLs advertised the same target branch SHA. The frozen WL104B–F ancestors and the WL83 direct-fit contract supplied the baseline evidence. WL104G is the owner-authorized suffix follow-up to WL104F.

Protocol commit `05f806f` froze the settings for all four stages before fresh evidence. Stage A checks the one disputed near-normal incidence with actual returned PDF values, 32 independent blocks of 65,536 samples per arm and eight separately scrambled 262,144-point uniform-solid-angle Sobol integrals. The independent integral compares acceptance mass and raw first direction moments against the sampler, including the zero contribution of rejected draws. The other check estimates four known cosine-weighted integrals. Each decision uses the absolute discrepancy plus four standard errors, with a 0.003 margin on raw PDF moments and a 0.006 margin on the known integrals. The original WL104F gate remains historical evidence. [Protocol](../../experiments/ecro_wl84_104g/protocol.json) · [WL104F](104F_WORKLOG_CERTIFIED_ER_ROOT_CAUSE.md)

R0 is frozen WL83 ER_DF. R1 uses the original WL82 per-incidence `el1 + ealb` objective. R2 uses the exact WL104F F² objective. For each incident group it sums squared RGB errors in the native cosine-weighted response over the original outgoing lanes with weights `use/q`, divides by the likewise weighted sum of squared layered responses, and averages groups uniformly. Both arms use the same frozen A/C logits as matched starts, 12,000 float32 Adam steps per start, the WL104F cosine schedule from 0.006 to 0.00006 with 50 warm-up steps, and choose the lower terminal original-data score per arm and material. The nine rows are the original WL104D panel, with the same development and held-out assignments. Each held-out material has its own historical fit lanes. The independent angular and scene targets are held out from fitting and selection. This tests refitting on separate responses and scenes for those materials. It does not test zero-shot descriptor generalization.

The first fit invocation stopped during path preflight before any update because WL104's local mirror lacks block-material checkpoints. Commit `722d986` resolved the checkpoints in the existing local WL83 archive and checked them against WL104D's hashes before fitting. The original 64×64 fit arrays, exact compiled R0, A/C starts, descriptor table and amended layered reference all verified. The experiment uses the same certified 64 parameters and exact 69-float runtime packing, or 276 bytes per material. The renderer and production contracts were preserved. [Fit runner](../../experiments/ecro_wl84_104g/fit104g.py) · [Independent angular archive provenance](../../evaluation/104D/amendment1/provenance.json)

## Measurement

### Operator and fitting

Stage A passed. The fresh C $y$-integral deviation was +0.00057647 with SE 0.00083903, giving an absolute four-SE bound of 0.00393258. The historical deviation was +0.00809941 with SE 0.00200248. The largest fresh bound on any of the eight exact integral estimates is 0.005710. The largest bound on any independent sampled-vs-Sobol raw PDF moment is 0.001080. C's acceptance probability was 0.92925835 versus integrated PDF mass 0.92923476. Sample/recomputed PDF and weight identity tests passed on both arms. This is consistent with a statistical fluctuation in the historical check and supports the current operator contract within the measured margins. [Operator audit](../../evaluation/104G/operator.json)

| Arm | Training wall time | Torch peak allocated | Max last-2,000-step own-loss change | Median own-loss gain versus R0 |
|---|---:|---:|---:|---:|
| R1 | 99.610 s | 106.2 MiB | 0.00002117 | 0.00025737 |
| R2 | 95.844 s | 106.2 MiB | 0.00000177 | 0.00348652 |

Both starts for all nine materials passed the 0.002 convergence rule and the two-step deterministic replay. R1 lowered the original training score on 9/9 materials. The largest gain, 0.01571, came from `test83_block_186`. R2 lowered its own training score on 9/9. The two objectives have different units and their training gains are reported separately. WL104G made no new fit targets. [Fit metrics and both start trajectories](../../evaluation/104G/fit.json) · [Fit states](../../evaluation/104G/fit_states.npz)

### Independent layered response

All nine angular references qualified. Maximum half-split error was 0.014442 in the 55–65° shape band, 0.015980 for grazing shape, and 0.000724 for albedo. All 27 compiled material/arm pairs passed physical checks. Worst Torch/Dr.Jit relative peak disagreement was `2.05e-6`, reciprocity residual `2.01e-14`, and passivity certificate upper bound 1.000000028 with the historical `1e-6` numerical tolerance. Exact archived R0 reproduced the WL104D angular metrics. Current area-integrated albedo error can rise even for a closure whose independent passivity certificate holds.

| Arm | Primary material eshape 20° / 50° / 60° / grazing | Primary mean el1 / ealb | Held-out median / worst el1 | Held-out median / worst ealb |
|---|---:|---:|---:|---:|
| R0 | .10263 / .07200 / .09772 / .05011 | .09219 / .02192 | .07072 / .11145 | .01543 / .04800 |
| R1 | .10329 / .07241 / .09765 / .04968 | .09216 / .02186 | .06520 / .11119 | .01546 / .03684 |
| R2 | .10246 / .13194 / .14222 / .20871 | .13004 / .08718 | .07847 / .14023 | .04043 / .06878 |

R1's mean `el1` gains reproduce across both reference halves. They average 0.000249 on development materials and 0.003769 on held-out materials. Both are below 0.005. Its largest independent `el1` gain came on `test83_block_186`, 0.07255 to 0.05091, which also reduced its `ealb` from 0.04800 to 0.02292. R2 worsened `el1` on both halves on 8/9 materials against R0 and all nine against R1. R2's `el1` increase versus R1 averaged 0.02151 and 0.01539 for development and held-out. The worst held-out `el1` is 0.14023 on `test83_hard_53`. On the primary material R2 shifts mean signed directional albedo bias from −1.50% to +6.44%, with worst absolute signed bias increasing from 3.28% to 14.29%. These albedo values are relative to the independent layered reference for the same direction. The corresponding absolute RGB error, shape distributions, both-azimuth probes, and core/shoulder/tail energy are stored per material. [Angular measurements](../../evaluation/104G/response.json) · [Angle and cohort figure](../../evaluation/104G/angular_fidelity.png)

On 262,144 resident direction lanes, seven synchronized evaluations after two warm-ups gave median wall times for eval / PDF / sample of 2.316 / 0.929 / 3.185 ms for R0, 2.306 / 0.926 / 3.178 ms for R1, and 2.329 / 0.927 / 3.173 ms for R2. These are launch-inclusive timings on the primary material. [Runtime fields in response audit](../../evaluation/104G/response.json)

### Paired scene fidelity

Development comparisons replay the frozen camera, lights, geometry, masks, R0 and depth-cap-1024 layered reference at 12×1,024 spp. The predeclared first held-out scene `test83_iid_159_L2` used the WL104D held-out camera at 12×4,096 spp and a separate 2,048-spp reference-only mask. R0/reference precision qualified before any candidate scene render. The fallback was unused. The WL104A floor-near and lower-sphere masks extend the review for interaction regions. The original object, cube, sphere, highlight core/ring and grazing ROIs were preserved. [Held-out reference qualification](../../evaluation/104G/test83_iid_159_L2_reference_readiness.json)

Native renders evaluate each fit at every event. The fixed-proposal comparison uses the same R0 proposal, native rejection and terminal response for all first-event weights. Recombined images from that comparison use each arm's direct E1, its fixed-proposal E2, and R0 HIGH. Native HIGH is the higher-order path-traced contribution. Stable RGB RMS uses the paired layered TOTAL denominator. Positive paired gain means lower error and intervals are 95% delete-one jackknife.

On the primary object ROI, R2 improves fixed-proposal E2 by **0.593 pp** [0.386, 0.801] relative to R0. Native R2 E2 is 0.550%, down from 1.135%. The primary TOTAL gain of **0.159 pp** [0.097, 0.221] misses the 0.5-pp practical threshold and its **0.536-pp E1 regression** exceeds the 0.5-pp regression margin. The uncancelled joint E1/E2 RMS worsens by **0.400 pp** [0.356, 0.444]. The twice E1×E2 error cross term shifts from +1.974 to −1.901 in squared percentage units. In signed energy, E1 shifts from −2.036% to −3.228% while E2 shifts from −0.463% to +0.608%. This separates the real E2 improvement from the cancellation that permits a lower TOTAL RMS despite increased joint error. R1's primary native TOTAL worsens by 0.0052 pp [0.0016, 0.0089]. [Primary native and fixed-proposal metrics](../../evaluation/104G/test83_iid_1_L2_metrics.json) · [Primary error maps](../../evaluation/104G/test83_iid_1_L2_native_errors.png)

The primary E2 gain is concentrated on the cube, where the native E2 RMS falls from 1.187% to 0.436%. Sphere E2 moves from 0.847% to 0.913% and floor-near TOTAL from 0.245% to 0.772%. On the companion scene, R2 regresses on E1, E2, HIGH and TOTAL in the objects ROI. Its TOTAL is worse by 3.097 pp [3.040, 3.154] versus R0. Fixed-proposal E2 is also worse by 0.793 pp [0.785, 0.802]. The companion cube TOTAL increases from 1.476% to 7.230% and floor-near TOTAL from 0.216% to 1.063%. Signed TOTAL energy moves from −0.143% to −6.143% on objects. [Companion metrics](../../evaluation/104G/test83_hard_1_L1_metrics.json) · [Companion error maps](../../evaluation/104G/test83_hard_1_L1_native_errors.png)

| Scene | Native arm | E1 RMS % | E2 RMS % | HIGH RMS % | TOTAL RMS % | Fixed-proposal E2 RMS % |
|---|---|---:|---:|---:|---:|---:|
| `test83_iid_1_L2` | R0 | 3.656 | 1.135 | 0.262 | 4.126 | 1.135 |
| | R1 | 3.660 | 1.136 | 0.258 | 4.131 | 1.129 |
| | R2 | 4.192 | 0.550 | 0.265 | 3.967 | 0.541 |
| `test83_hard_1_L1` | R0 | 2.971 | 0.293 | 0.131 | 3.003 | 0.293 |
| | R1 | 3.065 | 0.291 | 0.123 | 3.095 | 0.292 |
| | R2 | 5.814 | 1.087 | 0.938 | 6.099 | 1.086 |
| `test83_iid_159_L2` | R0 | 6.093 | 0.560 | 0.029 | 5.813 | 0.560 |
| | R1 | 6.056 | 0.538 | 0.025 | 5.760 | 0.556 |
| | R2 | 5.764 | 0.431 | 0.034 | 5.705 | 0.440 |

The held-out R2 native joint E1/E2 RMS improves by 0.339 pp [0.337, 0.341]. TOTAL improves by **0.108 pp** [0.102, 0.114], below the 0.3-pp held-out threshold. Signed TOTAL moves from −3.068% to −4.574%, a 1.506-pp absolute bias regression against the 1-pp margin. Fixed-proposal E2 improves by 0.121 pp [0.106, 0.136]. Native R1 TOTAL improves by 0.053 pp [0.046, 0.059] on this scene and worsens by 0.093 pp [0.082, 0.103] on the companion. All native and fixed-proposal scene precision checks passed. [Held-out metrics](../../evaluation/104G/test83_iid_159_L2_metrics.json) · [Held-out error maps](../../evaluation/104G/test83_iid_159_L2_native_errors.png)

The two development R0 native replays agree with archived event buffers within `2.20e-7` relative L1. A pair of repeated 4-spp crosses reproduces within `1.22e-10`, and R0 crossed with its own proposal and terminal response matches the native E2 across all 36 paired packets within `1.99e-7`. The event residual mean-square decomposition reproduces TOTAL to floating-point precision. The exported native and fixed-proposal mean buffers reproduce TOTAL by summing the three events within `9.32e-8` relative to peak. The per-event mean images, fixed masks, paired buffers and per-ROI signed errors are retained in the raw run. [Output identities](../../evaluation/104G/manifest.json)

## Observation

The same-objective control improves its own fit score and independent angular `el1` on all nine materials. Most gains are small. R2 also converges under its own objective, but it raises primary angular shape error at 50°, 60° and grazing and increases the absolute directional albedo error by 0.06526. Its best effects show up as lower E2 in the primary and held-out scenes. The same sign appears when R0 supplies the proposal and terminal evaluation, while the companion's adverse E2 result also persists under R0's proposal. The two developments therefore show different consequences from the same material-local fitting recipe.

## Interpretation

- **Fidelity recovery, R2−R0.** The tested F² recipe trades angular shape and energy accuracy for localized E2 gains. Those gains coexist with worse E1 and joint error on the primary, a clear companion regression, and an undersized, signed-bias-regressing held-out TOTAL gain.
- **Objective effect, R2−R1.** Matched A/C starts, original fit data, capacity and Adam schedule establish an adverse objective effect on the independent angular panel. The same comparison shows a 0.164-pp primary TOTAL gain alongside a 0.396-pp joint E1/E2 regression and a 3.004-pp companion TOTAL regression. This tests the complete F² recipe with its original normalization and without a separate albedo term. It leaves individual effects of squared residual weighting and removal of the albedo term confounded.
- **Additional optimization, R1−R0.** Original-objective refitting has measurable headroom on the isolated `block_186` angular response. Changing starts and repeating optimization are both part of this control relative to historical R0. The converged R1 fit permits neither a general fidelity claim nor an explanation of R2's different transport effects by extra optimization alone.
- **Independent validation.** The six held-out materials qualify as unseen evaluation targets for the fit. The first independently specified held-out scene qualified. Its positive R2 component results fall short of the predeclared scene margin and regress in absolute signed TOTAL bias. The required generalization result is unsupported.

## Unresolved Question

The 64-parameter family's best achievable layered-response accuracy remains unknown. The tested F² recipe and matched original-objective refit both used the original angular samples, including the WL104C support gap at 55–65°. The two converged fits do not identify an approximation lower bound or separate that support limitation from every possible fitting objective. The evidence also covers only one held-out scene and this coated-conductor panel.

## Architecture Verdict

**NO_DEMONSTRATED_FIDELITY_IMPROVEMENT.** The operator, fit convergence, physical reference and scene precision checks qualify the negative comparison. R2 versus R0, R2 versus R1 and R1 versus R0 all fail the prospective recovery criterion. The observed component and material benefits, the broader R2 regressions and the R1 optimization headroom are retained as bounded evidence. Certified ER remains the project baseline under the owner's existing decision. Decisions about retention, modification or replacement belong to the owner. [Machine-readable decision](../../evaluation/104G/decision.json)

## Reproducibility

- **Commits.** Verified initial HEAD `a0966dcd00daff22ff07bf8107e99a9173f25808`. Protocol and operator `05f806f1f77021484d20ad00401fed526e7388f3`, matched fit `722d986e784230ff9fc5783eacac349e98c97d55`, independent angular and scene evaluation `4a3af34fc949167f6eca8735948b4dbad7213c29`. The archived WL104D freeze and amendment are `68f803d` and `c46015c`. WL104E closure is `2efd8f8` and WL104F closure is `a0966dc`.
- **Environment.** Local Windows build 26200, RTX 5080 16,303 MiB, driver 596.49, Python 3.11.9, Torch 2.8.0+cu128, Mitsuba 3.5.1 `cuda_ad_rgb`, Dr.Jit 0.4.6, NumPy 1.26.4. Stage A took 1.297 s with 9.64 MiB Dr.Jit device allocator peak. Matched fitting took 195.454 s combined. Angular comparison took 1.063 s including operator timing. Stage D took 842.0 s including held-out reference preflight, checks, renders, output and analysis. Dr.Jit peaked at 1.35 GiB device allocations and 14.59 MiB pinned host allocations. Context and non-Dr.Jit GPU allocations are outside those peaks. Same-run native R1/R2 times were 23.58/23.64 s for the primary, 37.76/37.83 s for the companion and 206.63/206.06 s for the held-out. Held-out layered/reference-mask and native R0 preflight timings were 35.56/2.58 s and 33.70 s. Preflight and main stage have different renderer context and tracing overhead. These timings are accounting data.
- **Seeds and files.** Scene seed bases were 1044900, 1046900 and 1048900 with replicate indices 0–11 and 512 spp per launch. The held-out mask uses 1049890. Samples for operator qualification use PCG64 seeds 10470011/10470012 and Sobol seeds 10470101–10470108. Original fit arrays SHA-256 is `13035b009c9695af06bb4f88c548015403d62e5dd86977aabf2e53fcfbffc149`. All raw outputs live under `results/ecro_wl84_104g/`. The manifest records exact raw/compiled checkpoints, angular inputs, meshes, historical event buffers, masks, executed code and reviewed artifacts. [Manifest](../../evaluation/104G/manifest.json)

Reproduce from the repo root with the verified frozen inputs and fresh destinations. The original stage commits are recorded in each output JSON. A report-time provenance fix allows a reproduction at a descendant commit when Git verifies that both the frozen qualification source and protocol match. The original fit and its qualification metadata are retained.

```powershell
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$python = 'external/8dna26/.venv-cu128/Scripts/python.exe'
$code = 'experiments/ecro_wl84_104g'
$out = 'results/ecro_wl84_104g_reproduction'
& $python -B "$code/qualify104g.py" "$out/operator"
& $python -B "$code/fit104g.py" "$out/operator/operator.json" "$out/fit"
& $python -B "$code/response104g.py" "$out/fit" "$out/response"
& $python -B "$code/scenes104g.py" "$out/fit" "$out/response" "$out/scenes"
& $python -B "$code/package104g.py" $out 'results/ecro_wl84_104g_review_reproduction'
```
