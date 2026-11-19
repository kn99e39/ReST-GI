# Worklog 104F — Certified ER response and proposal separation (2026-11-20)

## Implementation Fact

The experiment crosses the frozen WL104D Amendment 1 evaluation functions and actual native proposals at the first continuation event. In the arm labels `AA`, `AC`, `CA`, and `CC`, the first letter selects the evaluation and the second selects the proposal. Geometry, cameras, lights, masks, material states, random seeds and the C terminal evaluation at the next hit come from the WL104E intervention. The independent image reference is the archived layered random-walk tracer at depth cap 1024. The angular reference is the independent stochastic layered measurement from WL104D Amendment 1. [Frozen protocol](../../experiments/ecro_wl84_104f/protocol.json) · [Cross implementation](../../experiments/ecro_wl84_104f/cross104f.py)

Let $F_X=f_X\cos\theta_o$ denote the cosine-weighted BRDF returned by the renderer. Each native proposal is a fixed mixture of one Lambert cosine lobe and seven GGX VNDF lobes. Inner lobes sample in refracted coordinates and include $\cos\theta_o/(\eta^2\cos\theta_o')$ when expressed as an external solid-angle density. The same angular validity cuts apply to A and C. Their strictly positive Lambert mixture term supplies common support on the valid domain. The directional density is a subdensity because invalid reflections and non-exiting inner draws return a zero-contribution atom. The native sampler computes $w_X=F_X/p_X$ on accepted directions. The intervention draws from $p_Y$, keeps its rejected draws at zero, and uses $F_X/p_Y$ on the accepted draws. E2 applies this weight to the fixed, visibility-tested terminal response. E1 evaluates $F_X$ at the direction to the visible point light with its geometric light factor and uses no ER proposal draw. [Native eval, PDF and sample code](../../experiments/fnto_compiled83/closure83_dr.py) · [WL104E tracer](../../experiments/ecro_wl84_104e/trace104e.py)

## Measurement

The object metrics use stable across-replicate RGB RMS with layered TOTAL as the denominator. Signed energy differences use integrated layered TOTAL luminance as the denominator. Intervals are paired 95% jackknife intervals with 12 replicates. The recombined TOTAL uses E1 from the evaluator arm, E2 from the cross and the archived C path-traced HIGH in every arm.

| Scene | E2 cross | E2 RMS % [95% CI] | Signed E2 error % | Recombined TOTAL RMS % |
|---|---|---:|---:|---:|
| `test83_iid_1_L2` | `AA` | 1.146 [0.964, 1.327] | −0.488 | 4.048 |
| | `AC` | 1.150 [0.979, 1.320] | −0.490 | 4.052 |
| | `CA` | 4.605 [4.493, 4.716] | +2.696 | 9.238 |
| | `CC` | 4.608 [4.495, 4.720] | +2.694 | 9.240 |
| `test83_hard_1_L1` | `AA` | 0.290 [0.280, 0.299] | +0.129 | 3.050 |
| | `AC` | 0.290 [0.280, 0.300] | +0.128 | 3.050 |
| | `CA` | 0.287 [0.277, 0.296] | +0.131 | 3.193 |
| | `CC` | 0.287 [0.277, 0.297] | +0.130 | 3.193 |

**Fixed $f$, varying $p$.** The primary signed differences for changing $p_A$ to $p_C$ were −0.0020% [−0.0094%, +0.0054%] at $f_A$ and −0.0019% [−0.0093%, +0.0054%] at $f_C$. The primary spatial RMS upper bounds were 0.071% and 0.050%. Both clear the frozen ±0.25% signed and 1.0% spatial equivalence margins. Per-pixel variance across replicates, averaged over the objects ROI, was 4.31% and 5.05% lower with $p_C$ at $f_A$ and $f_C$, respectively. The companion signed intervals also contain zero and its measured variance differences are about 0.4%. The clipped stable RMS estimator is at zero for both companion fixed-response differences. The signed intervals and measured variance give the resolution of that comparison.

**Fixed $p$, varying $f$.** The primary swap $f_A$ to $f_C$ changes the E2 spatial map by 5.054% RMS under either proposal. The 95% lower bounds both exceed 5.051%. The projection of the isolated response swap onto the original `CC−AA` E2 change is 99.972% [99.788%, 100.156%] with $p_A$ and 99.976% [99.809%, 100.143%] with $p_C$. Thus the response-only change clears the prospective 75% criterion and exceeds the fixed-response proposal effect bound. The primary cube response-only map changes by 5.531% RMS and the sphere by 1.288%. The companion response-only map change is small at 0.092% RMS under either proposal. The projections describe the C−A change. [Primary scene and ROI statistics](../../evaluation/104F/test83_iid_1_L2_metrics.json) · [Companion scene and ROI statistics](../../evaluation/104F/test83_hard_1_L1_metrics.json) · [Spatial maps](../../evaluation/104F/cross_E2_residuals.png)

**Native baselines and direct E1.** Each RMS entry below is a percent of reference TOTAL RMS. A and C are full native paths. The LC row is the frozen WL104E layered S1 intervention with C direct and terminal terms. ER_DF is the frozen WL104D direct-fit comparator. Its archived TOTAL and component RMS, calculated from the saved replicate buffers, use the same paired reference. For E1 on the primary, the paired 95% intervals are [3.662%, 3.667%] for A and [8.414%, 8.433%] for C. The direct responses' signed errors are −1.995% and −2.395%. C's positive E2 error partly cancels the negative E1 error when energy is summed.

| Scene | Arm | E1 RMS % | E2 RMS % | HIGH RMS % | TOTAL RMS % |
|---|---|---:|---:|---:|---:|
| `test83_iid_1_L2` | A | 3.665 | 1.146 | 0.259 | 4.138 |
| | C | 8.424 | 4.608 | 0.140 | 9.240 |
| | ER_DF | 3.656 | 1.135 | 0.262 | 4.126 |
| | LC | 8.424 | 0.018 | 0.034 | 8.418 |
| `test83_hard_1_L1` | A | 3.020 | 0.290 | 0.122 | 3.050 |
| | C | 3.161 | 0.287 | 0.129 | 3.193 |
| | ER_DF | 2.971 | 0.293 | 0.131 | 3.003 |
| | LC | 3.161 | 0.049 | 0.039 | 3.162 |

The native TOTALs and component RMS use their respective archived formulas and paired buffers. The RMS of the sum includes covariance and cannot be recovered by summing the component RMS. [Frozen scene decomposition](../../evaluation/104E/descriptive_event_residuals.json) · [Frozen layered S1 and terminal measurements](../../evaluation/104E/cases.json)

**Operator acceptance.** The complementary PDF audit uses three materials, six incident polar angles and two azimuths, with eight blocks of 16,384 draws under each of A, C and cosine proposals per probe. It checks the density on accepted draws, integrates that density using the balance mixture including zero lanes for rejected proposals, and compares its mass to the observed acceptance probability. It also estimates the exact cosine-weighted integrals of (1,1+x/2,1+y/2,1+z/2) and compares fixed-evaluator RGB directional albedo across the two native proposals. The known exact integrals are (1,1,1,4/3). This audit adds a check of the sampled distribution against the reported density to the existing (w\,p=F) identity.

All 36 probes pass support, both proposal mass checks and both fixed-response albedo checks. One of 288 scalar known-integral checks fails. At `test83_iid_1` and incidence 5° / azimuth 22.5°, C's estimate of the (y) moment exceeds 1 by **0.00809941**, with SE **0.00200248** and a frozen limit of **0.00800994**. Its acceptance is 0.92951202 versus integrated PDF mass 0.92950894. The three other exact-integral checks for that draw set pass. The failure is reproducible with the fixed draws and provides an unresolved sampler qualification result. The audit alone cannot distinguish a statistical fluctuation from a small distributional inconsistency. [Operator audit](../../evaluation/104F/operator_audit.json) · [Frozen gate result](../../evaluation/104F/decision.json)

## Observation

A and C have independent physical shape residuals at matched directions. On the archived `test83_iid_1` angular measurements, the C fit raises shape error from 0.10312 to 0.16157 at 20° and from 0.07212 to 0.08107 at 50°, while lowering it from 0.09765 to 0.06829 at 60°. The probe-averaged directional albedo error improves from 0.02193 to 0.01665, while the averaged `el1` rises from 0.09214 to 0.09523. These response comparisons use the stochastic reference, shared lanes and reference precision checks. WL104E measured about 66% of the primary object's ER-first-hit E2 energy below 50° and about 1.51% in 55–65°. The mismatch across angle bands persists even when the proposal is held fixed. Current data do not isolate the contribution of a particular angular subregion. [Independent response](../../evaluation/104D/amendment1/response.json) · [Frozen propagation evidence](../../evaluation/104E/direction_diagnostics.json)

## Interpretation

**H1, evaluated angular response, is the dominant measured cause of the primary C−A E2 regression.** The isolated response swap recovers essentially the whole spatial change under each proposal and reproduces the positive E2 bias. A proposal-only explanation for that scene-scale regression is rejected within the frozen equivalence margins. The independent E1 residual also follows the response-evaluation path because a direct point-light query never samples ER. The scene experiment establishes this shared evaluation mechanism and leaves the individual angle contributions unresolved. It does not attribute every absolute E2 residual or all of TOTAL to the C−A swap.

**H2, sampling and PDF, has a bounded scene effect and an open operator qualification.** Common support, proposal mass including the rejection atom, native throughput identity and the scene equivalence checks pass. The one exact-integral check prevents full sampler qualification. It does not identify an evaluator/PDF repair.

## Unresolved Question

**H3 capacity and H4 optimization remain confounded with objective mismatch.** WL104D demonstrates within-family movement in angular error at the cost of other angles and scene fidelity. That establishes a trade-off for the tested fits. The frozen (F)-squared correction and the matched original-objective reoptimization control require a passing operator audit. That condition failed. The original 64×64 material-local fit data are available and hash verified. Neither fit was run. An independent test of a better fit at the same capacity and a matched original-loss recovery attempt would have distinguished the tested objective and optimization explanations. This batch has neither that comparison nor a valid lower bound on family approximation error.

## Architecture Verdict

**INCONCLUSIVE for correction or replacement.** Evaluation is the established proximal mechanism for the primary regression, and proposal changes alone are too small to explain it. The full PDF consistency gate failed. No response correction was implemented or trained. Corrected E1, E2, TOTAL, angular generalization and held-out scene fidelity were therefore unmeasured. `test83_iid_159` contributed operator qualification probes only. The existing learned ER remains the bounded local-scattering development baseline. This closes the dedicated investigation without adopting a candidate or a replacement contract.

## Reproducibility

- **Freeze and replay.** Primary implementation and protocol committed at `9a84a48dcd6c2726f0d99821be03badc357d8144` before final evidence. The first execution passed its focused identities and wrote the operator audit, then stopped on a backslash key in the archived Windows manifest before writing any full-scale scene replicate. Commit `ae6b713214c1ed589a8af370a6856abc5f0074c8` normalizes path separators, and the replay uses a fresh output directory and the same seeds. Both saved operator audits are byte-identical with SHA-256 `11785b4c1d999b7b4b242bc090926084cc30ca0f4f4330a022dedda32fa4750b`. The aborted execution is retained under `results/ecro_wl84_104f/cross/`.
- **Budget.** 256×256, 12 paired replicates per scene, 1,024 spp per replicate and two 512-spp launches per proposal. Seed bases are `1044900` and `1046900`, plus replicate index. The operator seed starts at `1046001` with fixed material and angle offsets in the protocol. The focused native and deterministic replay checks use 4 spp. Scene precision gates pass in both cases. All 24 archived AA/CC E2 replays match within (5.20\times10^{-8}) relative L1. The worst focused repeat discrepancy is (2.52\times10^{-10}). Native A/C/ER_DF event accounting gives E1+E2+HIGH exactly in the archived scene metrics. The preserved primary WL104E layered focused identity is (1.14\times10^{-12}) and maximum native weight/PDF inconsistency is (7.95\times10^{-8}) of peak.
- **Host and cost.** Local RTX 5080, 16,303 MiB capacity, driver 596.49, Windows build 26200, Python 3.11.9, Mitsuba 3.5.1 `cuda_ad_rgb`, Dr.Jit 0.4.6 and NumPy 1.26.4. Part A took 25.265 s including audit, focused checks, rendering and analysis. A cost-only replay took 25.719 s and reached 943.7 MiB of Dr.Jit device allocations and 9.77 MiB of pinned host allocations. These are allocator peaks and exclude device allocations outside Dr.Jit. The cost-only replay agrees within `1.07e-8` relative L1 across all four E2 fields in all 24 packets. Its operator audit and frozen decision are byte-identical. The state size is the frozen 64 parameters and 69 runtime floats per material. There was no training or new data collection. Server 63 was used to read the archived fit inputs only. Their SHA-256 is `13035b009c9695af06bb4f88c548015403d62e5dd86977aabf2e53fcfbffc149`, matching the WL104D manifest. [Resource replay](../../evaluation/104F/resources.json)
- **Inputs and outputs.** The runner validates frozen descriptor, mesh, material-state, mask and replicate hashes. `results/ecro_wl84_104f/cross_replay/` contains the 24 raw E2 cross packets and full audit. Its provenance and the hashes of the reviewed files are in the evaluation manifest. [Run status](../../evaluation/104F/status.json) · [Verified input and output hashes](../../evaluation/104F/manifest.json)

Use the repo root and fresh output directories. The frozen parent artifacts must be present at the paths listed in the manifest.

```powershell
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$python = 'external/8dna26/.venv-cu128/Scripts/python.exe'
& $python -B experiments/ecro_wl84_104f/cross104f.py results/ecro_wl84_104f/reproduction
& $python -B experiments/ecro_wl84_104f/package104f.py results/ecro_wl84_104f/reproduction results/ecro_wl84_104f/review_reproduction
```

To reproduce the allocator report, use the same environment and a new output path.

```powershell
$profile = 'import sys; sys.path.insert(0,"experiments/ecro_wl84_104f"); import cross104f as x; x.main(); print(x.dr.whos_str())'
& $python -B -c $profile results/ecro_wl84_104f/profile_reproduction
```
