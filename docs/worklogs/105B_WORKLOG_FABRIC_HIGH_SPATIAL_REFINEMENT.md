# Worklog 105B — Fabric HIGH spatial refinement (2026-11-21 KST)

## Question and implementation

The owner authorized one adapter improvement following WL105A. This trial tests whether finer spatial addresses can preserve more fabric detail while using the existing ECRO v0 model and physically evaluated LOW. The initial HEAD was `41df700` on `ecro_lr3_adapter`. The preregistration and execution code were committed at `e385c80`. [Protocol](../../experiments/ecro_fabric105/refinement105b.json)

On the public B-spline asset, split each 18 cm world cell along all three spatial axes into 6 cm cells. Keep 64 direction bins per hemisphere, both hemispheres, area weighting, the native surface probe stream, same hair BSDF, and native per-ray event accounting. The original area weights recombine to the coarse cells to `7.24e-17` relative L1. Occupied cells increase from 92 to 875, or 184 to 1,750 patch-side addresses. The immutable accepted probes stay at 213,081. Their median count per occupied cell falls from 2,478 to 226. The smallest fine cell has one probe. The fixed-size pools in small cells can therefore reuse the same native point many times. The adapter stores the cell size with new geometries and uses the original 18 cm default for old files. [Adapter](../../experiments/ecro_fabric105/fabric105.py) · [Geometry evidence](../../evaluation/105B/refinement.json)

The native transport preflight passed on the unchanged 2×96 camera test. The refined state maps 99.9944% of hit state rays to an occupied 6 cm cell. Camera lookup fallback was 0.0291% in focused validation. Native transport signed energy disagreement was +0.65% and block-averaged relative HDR L1 was 5.20%, versus 7.92% between native reference halves. A zero HIGH table preserved native LOW within `2.11e-8` relative L1.

## Controls

- For a representation-level comparison, insert native physical area averages into both adapters. The coarse films are the hashed WL105A area oracle. The fine films use independent 4,096-spp per-bin half tables in the 6 cm grid, tested on the same two archived camera seeds. Neither area table uses camera values as training targets.
- For the learned comparison, train a fresh ECRO v0 fit with the WL105 seed (10501), Adam schedule, OWN loss, train/validation rotations, per-patch source budget and per-bin transport spp. Use 12,000 steps and the final checkpoint on both. The 6 cm adapter increases total training samples, graph size and the number of lookups. The equal-step comparison includes this cost difference.
- Reuse the archived 512², 2×512 spp native LOW and physical HIGH, masks and per-view cameras/lights. Measure pixel relative HDR L1 and signed fabric energy on the full image, 8×8 block averages, and cross-half MSE of the 8×8 HIGH residual. Joint learned recovery requires at least 2 percentage points lower pixel L1 and 3% relative lower HIGH detail MSE in both views, gain in each pixel half and no more than 3 points absolute luminance-bias regression. This leaves the angular bin width and camera-independent averaging fixed.

## Physical area-table result

| View | TOTAL pixel L1, 18 → 6 cm | TOTAL 8×8 L1, 18 → 6 cm | HIGH detail MSE reduction | Signed TOTAL luminance, 18 → 6 cm |
|---|---:|---:|---:|---:|
| Public | 46.24% → 45.24% | 36.69% → 35.02% | 7.34% | +4.50% → +6.55% |
| Held-out | 62.64% → 57.49% | 52.42% → 46.27% | 7.40% | +21.29% → +20.12% |

The fine physical table improves detail and pixel errors in both original views. This identifies a spatial contribution at fixed angular resolution and area-mean semantics. The area-table brightness difference persists and rises in the public view. Its held-out bias remains +75.07% for HIGH alone. The finer cells thus give limited detail recovery with the original directional bins and source averaging. Physical LOW matches the archived buffers exactly. Fallback accounts for at most 0.036% of driven second-hit lookups. The reference table half-to-half relative L1 is 9.66% and 9.55% for the two fine area pools.

## Learned result

| View | TOTAL pixel L1, 18 → 6 cm | TOTAL 8×8 L1, 18 → 6 cm | HIGH detail MSE reduction | Signed TOTAL luminance, 18 → 6 cm |
|---|---:|---:|---:|---:|
| Public | 43.52% → 43.17% | 34.65% → 33.43% | 5.93% | −0.17% → +2.22% |
| Held-out | 57.90% → 57.82% | 48.08% → 46.47% | 8.26% | +17.89% → +19.97% |

The learned result recovers part of the representation-level detail gain. Its native LOW agrees exactly with the 18 cm camera buffers on both halves. Both halves show lower TOTAL pixel L1. The mean gain is 0.34 pp in public and 0.08 pp held out, below the preregistered 2 pp in both views. Even though the held-out physical area oracle gains 5.15 pp in TOTAL pixel L1, the 6 cm learned image retains 0.08 pp of improvement over the 18 cm learned image. The oracle and learned comparisons have different starting errors and the contrast between their gains is not a literal additive attribution. Learned HIGH held-out energy bias rises from +66.73% to +74.50%. The spent spatial resolution gives better coarse contrast, smaller cells in the residual, and a brighter held-out image. The learned fine table differs from its own area reference by 38.85% and 29.96% relative L1 across the two lighting settings. These are within the 6 cm representation and use its own physical-pool targets. [Full metrics](../../evaluation/105B/summary.json) · [HIGH comparison figure](../../evaluation/105B/high_refinement.png)

## Verdict

**Spatial contribution supported. Joint learned fidelity recovery not supported.** Moving from 18 cm to 6 cm at fixed direction bins reduces HIGH detail error on physical and trained tables. It yields small native pixel gains and worse luminance bias at substantially greater cost. The test isolates spatial refinement under the inherited area-sampling and readout policy. It leaves the relative importance of finer direction bins and changed supervision/readout weighting open. This result is limited to the public Fabric geometry, two camera and environment conditions, and one matched fit seed.

## Cost and reproducibility

- The same 735,664-parameter ECRO v0 and 12,000-step schedule took 74.33 s at 18 cm and 229.72 s at 6 cm on the local RTX 5080. Torch peak allocation rose from 117.44 to 899.54 MiB. The fine trial's training and validation data generation took 25.34 and 24.64 s. The 6 cm physical oracle took 269.30 and 279.78 s for the two views, including camera readout. The end-to-end learned evaluation took 20.61 and 21.48 s. The train comparison matches per-bin budgets and step count and allows total work to grow with the grid. [Training trace](../../evaluation/105B/training.json)
- Frozen runner `e385c80`. The completion marker records HEAD `edc926e`, an unrelated WL104I figure commit made during this execution. The WL105B, Fabric and model files are identical between those two commits. Outcome files are in `results/ecro_fabric105b/`. The candidate checkpoint SHA-256 is `b2c34a39b878c623a6012301830d70c6660013781bd38f13f0eaf0d924cc945d`. [Raw and review hashes](../../evaluation/105B/manifest.json)

```powershell
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$py = 'external/8dna26/.venv-cu128/Scripts/python.exe'
$code = 'experiments/ecro_fabric105'
$out = 'results/ecro_fabric105b_reproduction'
& $py -B "$code/run105b.py" results/ecro_fabric105 results/ecro_fabric105a $out
& $py -B "$code/review105b.py" $out results/ecro_fabric105b_review
```
