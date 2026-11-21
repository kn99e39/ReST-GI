# Worklog 105A — Fabric HIGH attribution (2026-11-21 KST)

## Scope

The owner requested a cause analysis of the WL105 Fabric result. The comparison uses the frozen WL105 checkpoint, native geometry and hair BSDF, exact two-event LOW, and existing six-event camera reference. The sole intervention replaces the HIGH table while holding the lookup addresses and physical camera paths fixed. A camera/light crossover tests the source of the held-out luminance bias. The run contains no fitting of ECRO weights. [Frozen protocol](../../experiments/ecro_fabric105/attribution105a.json) · [Parent result](105_WORKLOG_ECRO_NATIVE_FABRIC.md)

## Controls

- **Learned.** Replay the archived WL105 HIGH. Its table contains one RGB value for each outgoing bin in an 18 cm cell and hemisphere.
- **Area-averaged physical.** Replace the learned table with native paths averaged over the exact same WL105 patch pool and outgoing bins. Each view uses two independently seeded tables at 4,096 spp per bin per table. This removes the learned table prediction discrepancy while retaining the area-sampled target and frozen readout.
- **Camera-weighted physical.** Fit one RGB constant per existing address from the camera's physical second-hit HIGH labels. Each channel uses squared first-scattering throughput as the fit weight. The two 512-spp fit streams are independent of the 2×512-spp evaluation streams. The paired area table fills addresses with no fit samples. The fit covers 98.4–98.5% of addresses, including many that have little camera weight. This privileged diagnostic tests the effect of replacing the area average with a value fitted to the camera's path distribution within the same interface.
- **Physical camera.** Retain the native per-ray six-event reference. Physical HIGH at the camera starts with a second scattering from the second hit, which is the third scattering event counted from the camera. The diagnostic's split-camera HIGH reproduces the physical HIGH on dedicated replay seeds within `1.42e-8` relative L1. LOW arrays match exactly in both oracle comparisons.

## Measurement

The 512² paired cameras reuse the parent mask and samples. Signed HIGH energy is relative to the physical HIGH within the fabric mask. Pixel relative HDR L1 below compares the complete LOW+HIGH image against the paired physical scene. The last two columns compare an 8×8 high-pass residual of HIGH with the learned error on the same view. The mean-square estimate multiplies independent half-residuals to suppress independent path noise.

| View | HIGH table | Signed HIGH energy error | Pixel TOTAL relative L1 | HIGH pixel MSE / learned | HIGH detail MSE / learned |
|---|---|---:|---:|---:|---:|
| Public | Learned | −0.66% | 43.52% | 100% | 100% |
| | Area-averaged physical | +17.33% | 46.24% | 98.3% | 99.6% |
| | Camera-weighted physical | +0.66% | 35.49% | 77.1% | 90.0% |
| Held-out | Learned | +66.73% | 57.90% | 100% | 100% |
| | Area-averaged physical | +79.43% | 62.64% | 105.0% | 99.4% |
| | Camera-weighted physical | +0.67% | 36.30% | 71.0% | 88.4% |

The learned and area tables differ by 43.24% and 42.83% relative L1 in the two views, versus 12.04% and 12.08% between area-reference halves. In paired image space, those table discrepancies partly offset the error from area pooling and lookup. For held-out full-resolution HIGH, mean-square terms expressed relative to the cross-half physical HIGH second moment are 0.9939 for area target and lookup error, 0.0345 for learned-minus-area table error, and −0.0815 for their cross term. Their sum is 0.9469. In the public view the respective terms are 0.8079, 0.0245 and −0.0108. The area term combines the original sampling pool's distribution with the grouping and readout. [Numeric results](../../evaluation/105A/summary.json) · [Paired HIGH images](../../evaluation/105A/high_attribution.png)

The light-view crossover uses 256² and 2×256 spp per configuration. The cells show signed fabric TOTAL luminance error with the native LOW held fixed within each comparison.

| Camera | Original environment | Rotated environment |
|---|---:|---:|
| Public camera | −0.02% | +13.01% |
| Held-out camera | +2.62% | +18.56% |

Changing the environment raises TOTAL bias by 13.03 or 15.93 percentage points, depending on camera. Changing just the camera shifts it by 2.64 or 5.55 points. In HIGH alone, the rotated environment creates +46.67% error with the public camera and +68.80% with the held-out camera. The two independent halves agree on these comparisons. [Crossover and half results](../../evaluation/105A/crossover.json)

## Interpretation

**The loss of yarn contrast is predominantly a consequence of the shared patch/bin representation and readout.** Replacing the learned values with physical area means leaves more than 99% of detail MSE in both views. Reweighting the physical labels to the camera path distribution restores the mean brightness and improves coarse shape, but 88–90% of the detail MSE still remains. A shared constant in each cell/bin averages different native fiber positions, tangents, and directions. The paired images show sharp stripes in per-ray HIGH and cell boundaries after either table substitution.

**Overbrightness on the held-out image is primarily associated with the changed lighting.** The crossover shows substantial excess with the original camera under the rotated environment. At the held-out setting the physical area table is brighter than the learned table. The camera-weighted table removes most of the integrated bias with unchanged address resolution. The evidence supports a lighting-dependent mismatch between area-average supervision and the path-weighted values used in camera readout. Learned table fitting has a separate error at its own target and partly offsets that mismatch in these images.

These tests do not isolate the contributions of cell size, bin width or surface-pool acceptance. The camera-weighted table minimizes a per-ray objective under one camera and light. Its two half estimates differ by 46–55% across all addresses, and its shared readout is tested on separate paths. Its remaining error is an empirical result for this control, not a formal lower bound on all image-space fits and not a deployable correction. The experiment neither proposes an architecture replacement nor rules out different fitting behavior.

## Reproducibility

- Original oracle execution `bf98012` produced and saved both view pairs. Resume recomputed their metrics exactly from raw buffers and verified the archived masks/LOW/HIGH against the hashed WL105 inputs. The first crossover replay aborted inside CUDA after one configuration. Commit `2d963a5` ran each configuration in a fresh process with the frozen geometry, camera, light, spp and seeds. The complete result contains all four configurations.
- The two oracle views took 72.34 s and 72.13 s. The resumed crossover and validation took 18.17 s. Hardware and renderer match WL105, local RTX 5080 with Mitsuba 3.5.1 and Dr.Jit 0.4.6. [Input and output proof](../../evaluation/105A/manifest.json)
- Raw 512² per-half HIGH and table pairs, camera masks, and 256² crossover means are stored under `results/ecro_fabric105a/`. The protocol and artifact hashes identify these files.

```powershell
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$py = 'external/8dna26/.venv-cu128/Scripts/python.exe'
$code = 'experiments/ecro_fabric105'
& $py -B "$code/attribute105a.py" results/ecro_fabric105 results/ecro_fabric105a_reproduction
& $py -B "$code/review105a.py" results/ecro_fabric105a_reproduction results/ecro_fabric105a_review
```
