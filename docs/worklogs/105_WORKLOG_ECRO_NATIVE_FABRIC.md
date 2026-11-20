# Worklog 105 — ECRO training on native 8DNA Fabric (2026-11-20 KST)

## Implementation

The user authorized a fresh local ECRO fit on the public 8DNA Fabric dataset, including the necessary adapter. The clean `ecro_lr3_adapter` checkout began at `29faf44`. The public Fabric asset and original image configuration come from `external/8dna26` at `4a2157ca24e506c5ac0831f27d656ecc50a64f07`. This is a standalone static workload using the official `get_scene` camera, native B-spline yarn geometry, hair material and HDR environment. The held-out row uses the same geometry with a separate camera and environment rotation. [Protocol](../../experiments/ecro_fabric105/protocol.json) · [Mitsuba hair material](https://mitsuba.readthedocs.io/en/v3.5.1/src/generated/plugins_bsdfs.html#hair-material-hair)

The original geometry contains 1,213,440 spline segments. Uniform-segment, uniform-azimuth surface proposals estimate weights from the centerline derivative and tube radius. Each proposal is kept only if a native short ray hits the originating primitive at the expected distance. The retained native hits provide the positions, shading frames and BSDFs. The patch quadrature uses these accepted locations and approximate tube surface weights. It represents 92 occupied 18 cm cells with two outgoing hemispheres per cell, for 184 patches and 64 bins per patch. The two directions partition the angular domain and share the cell area between their states. The 69-float material input carries hair descriptors for the ECRO network. Scattering itself remains native Mitsuba eval/PDF/sample, including both sides of the anisotropic transmissive BSDF. [Curve adapter and event tracer](../../experiments/ecro_fabric105/fabric105.py)

Camera transport accounts for visible environment (E0), the first two scattering events (E1 and E2), and the remaining four (HIGH). Environment next-event estimation and BSDF escape paths contribute to the same event using power-heuristic MIS. ECRO uses the physical E0–E2 and a learned HIGH lookup at the second surface hit. The field labels use the same physical sampler from the patch pool. Reference images use the independent native Mitsuba path integrator with six scattering events and Russian roulette outside that horizon.

## Qualification and fit

Native geometry probes accepted 213,081 of 262,144 draws (81.28%, 81.23% of estimated area). State rays hit geometry 92.86% of the time and all their hits mapped to exact occupied cells. Test camera lookup fallback was zero. In the 128² transport check at 2×96 spp, adapter vs native fabric-region signed luminance differed by +0.53%. At 8×8 block averages, relative HDR L1 was 5.17% against 7.85% between the native halves. A zero table preserved physical LOW within `2.15e-8` relative L1. This qualifies the local transport integration within the prospective margins and sampling resolution. [Preflight](../../evaluation/105/preflight.json)

The original 735,664-parameter ECRO v0 was initialized and trained locally with the WL102 OWN loss and Adam schedule. Seeds and budgets were fixed before training. Eight groups of four rotated environments supplied 32 training illuminations at 24 samples per patch/bin. Two groups of four independently selected rotations supplied validation labels at 2×48. The public and held-out eval rotations were excluded from this training and validation schedule. The exact native camera images were reserved for evaluation. Data generation took 4.14 s for training and 2.09 s for validation. The 12,000-step fit took **74.33 s**, with peak Torch allocation of 117.44 MiB. The final checkpoint owns the validation result. Relative patch-field L1 was 0.564 at step 4,000, 0.601 at 8,000, and 0.617 at 12,000. The fixed primary recipe selected the terminal step. [Fit trace](../../evaluation/105/training.json)

## Render precision

The original 512², 2×64-spp results are retained separately. Their native reference halves differed by 44.76% and 51.01% pixelwise relative HDR L1 within the two fabric masks, motivating a rendering precision replay. Commit `96caab2` froze a fresh 2×512 spp comparison on the same views and checkpoint, with a separate 2×256-spp patch-field reference. The replay preserved all model weights, geometry, cameras and environment settings. [128-spp comparison](../../evaluation/105/fabric_comparison.png) · [Precision replay protocol](../../experiments/ecro_fabric105/precision105.json)

## Result

Both images and the trained checkpoint completed. The table uses relative HDR L1 within the fabric mask. The 8×8 result is the full image after spatial averaging. Signed luminance is relative to native reference energy in the fabric mask.

| View | Pixel L1, LOW / ECRO / explicit six events | 8×8 L1, LOW / ECRO / explicit six events | Signed luminance, LOW / ECRO |
|---|---:|---:|---:|
| Public camera and original environment | 40.22% / 44.88% / 12.35% | 39.41% / 34.84% / 2.04% | −25.60% / +0.27% |
| Held-out camera and environment rotation | 41.73% / 59.67% / 14.39% | 40.58% / 48.41% / 2.43% | −26.57% / +18.26% |

Native reference halves differ by 17.24% and 19.99% in pixel L1 and 2.77% and 3.19% after block averaging. Explicit E0/E1/E2 are identical between paired physical and learned modes. Patch fallback affects fewer than six in 100 million second-hit lookups. Physical HIGH accounts for 39.75% and 40.92% of fabric RGB radiance. Learned HIGH relative L1 against the paired physical HIGH is 109.48% and 141.51%, and its signed luminance error is −0.66% and +66.73%. The public result recovers integrated energy and some coarse variation but softens the yarn contrast and increases pixel error. The held-out result is overbright and increases both error measures. [1,024-spp comparison](../../evaluation/105/high_spp/fabric_comparison.png) · [Numeric results](../../evaluation/105/high_spp/metrics.json)

**Adapter viability supported. Learned HIGH fidelity is insufficient on this run.** The result establishes local operation of the existing network and a physically qualified adapter on the original curve asset. The learned term does not preserve the fiber-scale illumination pattern. At patch level, total relative L1 is 0.478 and 0.480 against 0.464 and 0.464 between the reference halves. Sampling noise and coarse pooling limit conclusions about the capacity of the underlying network. This study covers one static Fabric geometry, two camera/light settings, one fit seed and a six-event reference horizon. No native 8DNA neural retraining cost or comparative lifecycle was measured.

## Reproducibility

- **Runtime.** Local RTX 5080 on Windows, Python 3.11.9, Torch 2.8.0+cu128, Mitsuba 3.5.1 `cuda_ad_rgb`, Dr.Jit 0.4.6. Both high-spp views and the patch-field controls took 74.81 s, peaking at 1.834 GiB in the Dr.Jit device allocator. The native reference camera traces took 2.70 s and 3.22 s. ECRO camera traces took 12.95 s and 12.94 s under the Python adapter's dispatch and reduction. The explicit Python path timings were 19.99 s and 19.91 s. These costs are specific to the current implementation, hardware and budgets.
- **Files.** Execution commits are `0cf45cb` for the native adapter and fit and `96caab2` for the precision replay. `results/ecro_fabric105/fit/model.pt` has SHA-256 `64ad6d25bc8f297cdfbf60dd6d0f95229da76fc1696fe6999ff29ed7c2b91920`. Raw HDR, masks, geometry, fit labels and render halves are under `results/ecro_fabric105/`. [Raw and review file hashes](../../evaluation/105/high_spp/manifest.json)

From the repository root, use a fresh output path for a new fit. The precision replay requires the exact saved checkpoint named in its protocol.

```powershell
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$python = 'external/8dna26/.venv-cu128/Scripts/python.exe'
$code = 'experiments/ecro_fabric105'
$out = 'results/ecro_fabric105_reproduction'
foreach ($stage in @('preflight','generate','train','evaluate')) {
    & $python -B "$code/run105.py" $stage $out
    if ($LASTEXITCODE -ne 0) { break }
}
# Run once on the recorded inputs, with no existing renders_1024 directory.
& $python -B "$code/run105.py" refine 'results/ecro_fabric105'
& $python -B "$code/review105.py" 'results/ecro_fabric105' 'results/ecro_fabric105_review' --render-dir renders_1024
```
