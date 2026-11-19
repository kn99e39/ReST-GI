# Worklog 104E — Certified ER error propagation through scene transport

## Authorization and freeze · 2026-11-19 KST

This suffix batch follows WL104D Amendment 1 on ecro_lr3_adapter. It tests which application of the frozen closure produces the E2 spatial regression and whether layered substitution also recovers TOTAL. The owner directed all rendering to the local machine and authorized read-only file transfer. Both completed, precision-qualified development cases were copied from LabServer63 and rendered on the local RTX 5080. [WL104D history](104D_WORKLOG_CERTIFIED_ER_ANGULAR_COVERAGE_OBJECTIVE.md).

Freeze commit 5cca29b638569475505e3224f30f1b058f0ef00b preceded evidence. A four-spp development smoke tested identities and native sampling weights. The frozen run used 12 paired replicates at 1,024 spp, 512 spp per launch, and the original seed bases 1044900 and 1046900. A separate two-spp replay of each first packet records realized path features. [Prospective protocol](../../experiments/ecro_wl84_104e/protocol.json).

## Implementation Fact

Point-light next-event evaluation at hit 1 produces E1. First continuation (S1) draws a direction and multiplies throughput by that sampler's matched weight. At hit 2, point-light BSDF evaluation (N2), visibility and geometric attenuation multiply that throughput to produce E2. Contributions at hits 3–6 make up HIGH in the WL104D physical camera tracer. ECRO's learned-HIGH mode uses a separate table lookup at hit 2. This diagnosis replays the camera tracer in which the regression was observed and preserves the E2 estimator used in both modes.

The dispatcher substitutes the sphere/cube operators. At S1, A and C each retain their native proposal and matching eval/pdf weight. L retains the layered random walk, its augmented-path weight and its validity sentinel. The point-light terminal estimator uses no BSDF-PDF MIS. At N2, variants under a common S1 reuse identical directions, geometry and light visibility. The two letters in a hybrid name specify S1 and N2. All hybrids use the same archived C E1. HIGH follows the chosen first continuation and uses C from hit 2 onward. With unchanged C continuation, it reuses archived C HIGH. The original full A and C, ER_DF and the layered reference serve as archived controls. No shared renderer or fitted state changed.

## Measurement

**Reproduction and estimator checks.** The archived A/C object TOTAL RMS and signed E2 recomputed exactly. Across 24 paired full-budget replicates, freshly replayed AA/CC/LL E2 agreed with their archived native versions to at most 1.94e−7 relative L1, against a 2e−5 limit. The reduced native test maximum was 8.24e−10, paired repeat maximum 1.16e−9, and sampled w × pdf versus eval residual maximum 7.95e−8 relative to peak. Accepted sampled PDFs were positive and finite. The synthetic interaction identity passed. Both scenes passed the original signed-energy half-width limit 0.0025 and object RMS half-width limit 0.005. Largest half-widths were 0.001986 and 0.001008 for signed energy, and 0.001812 and 0.000867 for E2/TOTAL RMS. [Scene metrics](../../evaluation/104E/cases.json).

**Counterfactual spatial RMS.** Percent of layered reference TOTAL RGB RMS on the combined sphere/cube ROI. A and C are the all-stage comparators.

| Arm | iid_1_L2 E2 | iid_1_L2 TOTAL | hard_1_L1 E2 | hard_1_L1 TOTAL |
|---|---:|---:|---:|---:|
| A | 1.146 | 4.138 | 0.290 | 3.050 |
| C | 4.608 | 9.240 | 0.287 | 3.193 |
| AC | 1.146 | 8.672 | 0.290 | 3.191 |
| CA | 4.608 | 9.240 | 0.287 | 3.193 |
| LC | 0.018 | 8.418 | 0.049 | 3.162 |
| CL | 4.608 | 9.240 | 0.286 | 3.193 |
| LL | ≈0 | 8.418 | ≈0 | 3.162 |
| ER_DF | 1.135 | 4.126 | 0.286 | 3.003 |

LL shares the E2 estimator and seeds of the layered reference and reproduces that same-seed E2. The reference half-split map on the primary objects had E2 RMS 1.71% and TOTAL RMS 1.79% of reference TOTAL. The spatial statistic uses cross-replicate pairs to remove mean-square Monte Carlo bias. The maps show signed residuals and the reference half-split noise on a common scale per event. [Primary maps](../../evaluation/104E/maps_test83_iid_1_L2.png), [companion maps](../../evaluation/104E/maps_test83_hard_1_L1.png).

**Signed energy on iid_1_L2.** Percent of layered TOTAL energy in the objects ROI.

| Arm | E1 | E2 | HIGH | TOTAL |
|---|---:|---:|---:|---:|
| A | −1.995 | −0.487 | −0.212 | −2.693 |
| C | −2.395 | +2.694 | ≈0 | +0.298 |
| AC | −2.395 | −0.488 | −0.191 | −3.074 |
| CA | −2.395 | +2.695 | ≈0 | +0.300 |
| LC | −2.395 | +0.001 | +0.013 | −2.381 |
| CL | −2.395 | +2.693 | ≈0 | +0.298 |
| LL | −2.395 | ≈0 | +0.013 | −2.382 |

**Causal E2 attribution.** Split CC−AA into CA−AA (S1), AC−AA (N2), and CC−CA−AC+AA (interaction). Cross-replicate projection onto the original E2 change assigns S1 99.9986% on the primary objects (95% CI 99.9982–99.9991%), 99.9986% on cube and 99.9998% on sphere. N2 contributes 0.0014% and the interaction contributes approximately −0.00007% on objects. The companion S1 share is 99.89% of its smaller E2 map change. The companion cube share slightly exceeds 100% because N2 partly cancels it. These signed projections add to one and describe contributions to the spatial change. [Attribution figure](../../evaluation/104E/attribution.png).

LC improves primary E2 RMS by 4.590 points (95% CI 4.477–4.702) and clears the 1.731-point margin without a companion regression. Its TOTAL gain is 0.822 points (0.749–0.895) versus 2.551 required. LL gives the same outcome. CL clears neither criterion. The primary cube's E2 RMS is 5.008% in C and 0.020% in LC, versus 1.199% in A. [Frozen decision output](../../evaluation/104E/decision.json).

## Observation

In the fixed diagnostic packets, 99.89% of the sampled primary object E2 energy with C at S1 terminates on a Lambertian floor or wall. The cube-to-floor route dominates the sampled cube E2 energy. Its accepted path counts per diagnostic packet are similar under A and C (3,512 versus 3,499), while propagated brightness rises about 7.8%. This comparison includes changes in direction and weights. With C, about 66.0% of ER first-hit E2 energy is below 50° and 1.51% is in 55–65°. The corresponding cube band share is about 0.005%. About 53.6% of the cube ER first-hit E2 energy is in the 3–12° mirror shoulder. These descriptions use a fixed 24-spp diagnostic subset across replicates. [Path diagnostics](../../evaluation/104E/direction_diagnostics.json).

On the frozen independent iid_1 probes, C improves eshape at 60° from 0.09765 to 0.06829 and worsens it at 50° from 0.07212 to 0.08107 and at 20° from 0.10312 to 0.16157. The measured transport weights emphasize directions outside the improved band. A particular angle or sampling probability remains unisolated. [WL104D independent response](../../evaluation/104D/amendment1/response.json).

## Interpretation

The E2 regression is traced to S1 with its native sampling and path weight. The two-step transport product propagates that error through geometry and a mainly Lambertian terminal response. The primary regression has little dependence on a second approximate ER application. Layered S1 restores E2 on cube and sphere, and the terminal-only intervention leaves the spatial regression in place.

A separate direct E1 residual remains. A post-hoc projection of existing event buffers gives E1 RMS 8.424% for C versus 3.665% for A. After LC, the unchanged C E1 accounts for essentially all of the remaining TOTAL residual. For the original C−A TOTAL map change, E1/E2/HIGH projection shares are 55.19%/44.13%/0.68%. The original C signed TOTAL bias is small because the signed E1/E2 contributions cancel spatially integrated energy. The E1 arithmetic is descriptive, and the prospective interventions held E1 at C. [Descriptive event decomposition](../../evaluation/104E/descriptive_event_residuals.json).

## Unresolved Question

The native S1 substitution identifies the combined effect of direction selection and matched weights. Their individual contributions and their behavior beyond the development scenes remain unmeasured. The learned HIGH table was outside this diagnosis. The measurements leave ER's capacity and a production correction unresolved.

## Architecture Verdict

**First-continuation attribution SUPPORTED. E2 recovery SUPPORTED. Joint E2/TOTAL scene recovery NOT SUPPORTED under the frozen margin.** The result establishes a local error-propagation mechanism on the reported primary case and a separate direct E1 floor for TOTAL.

## Reproducibility

- **Host and costs.** Local RTX 5080, driver 596.49, Windows build 26200, Python 3.11.9, Torch 2.8.0+cu128, Mitsuba 3.5.1 cuda_ad_rgb, Dr.Jit 0.4.6. Total driver runtime was 135.77 s, including verification, focused checks, rendering and analysis. Accumulated renders across three S1 choices were 33.95 s for iid_1_L2 and 64.62 s for hard_1_L1.
- **Inputs.** Frozen compiled states, material IDs, descriptors, geometry assets, masks and replicate buffers matched their saved hashes. Copied raw inputs are in results/ecro_wl84_104e/parent/scenes/. Primary outputs and per-replicate diagnostics are in results/ecro_wl84_104e/primary/. [Execution and raw hashes](../../evaluation/104E/provenance.json), [review manifest](../../evaluation/104E/review_manifest.json).
- **Commands.** Use the local WL104 Python environment from the repo root, with OMP_NUM_THREADS, MKL_NUM_THREADS and OPENBLAS_NUM_THREADS set to 1. The primary and reviewer output directories are created by the programs and must be unused for a fresh reproduction.

~~~powershell
$python = 'external/8dna26/.venv-cu128/Scripts/python.exe'
$commit = '5cca29b638569475505e3224f30f1b058f0ef00b'
& $python -B experiments/ecro_wl84_104e/run104e.py results/ecro_wl84_104e/parent results/ecro_wl84_104e/primary --commit $commit
& $python -B experiments/ecro_wl84_104e/review104e.py results/ecro_wl84_104e/primary results/ecro_wl84_104e/parent evaluation/104E
~~~
