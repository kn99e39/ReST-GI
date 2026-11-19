# Worklog 104C — Certified ER angular fidelity under an objective intervention

## Authorization and freeze · 2026-11-19 KST

WL104C is a suffix follow-up to WL104B on `ecro_lr3_adapter`. The user brief authorizes a matched comparison of the frozen WL83 ER_DF, a same-objective optimization control, and a shape-aware objective on the same certified family and fit data. The owner selected LabServer63. The brief requires a stop when the original data cannot support the comparison. [Preflight protocol](../../experiments/ecro_wl84_104c/protocol/wl104c.json).

The baseline starts at `add00a667aaaaec4b9a0b84f0517df914e0138a7`. Protocol and replay choices were frozen in `15cbbc22a2b4b13cba8046abeda1f830ae85db93`. The run first exited while reading the mesh hash keys before response or image evaluation. Commit `d5f46982743f898f8619360cad08af8722958b30` reads those keys from the existing WL104 protocol's scene section. It changes one lookup and leaves the input bytes, metrics, budget and gates as frozen. The first log and the completed retry are retained on the server.

The preflight replays all 16 response lanes for each of WL104B's three development materials, plus the first complete 4,096 spp image replicate for both ER_DF and the layered depth-1024 comparison. A same-seed image replay checks the pipeline across machines. The 32-replicate statistical estimates remain the original WL104B evidence. The angular support gate uses a 5° cone about the mirror direction at the problematic 60° probe. For incidence at 60°, any direction inside the cone has a polar angle in [55°, 65°]. The gate checks both endpoints so that exchanging incident and outgoing directions under reciprocity cannot manufacture support.

## Implementation fact

The experiment-local driver reused the WL83 ER decode and Dr.Jit runtime and the WL104B tracer and layered plugin. It read the archived WL83 fits and reference samples and replayed E1, E2, HIGH and their sum. It then counted positive-weight lanes in all three original WL83 direct-fit sets. The certified family has 64 fit parameters and 69 compiled state floats per material. The passivity check on the frozen states uses the existing analytic bound on the sum of the nonnegative per-channel mixture coefficients.

The synthetic coverage check passed for an excluded middle angle, a directly sampled mirror pair, its reciprocal counterpart and a grazing group. A separate check on the server used direction cosines directly and confirmed the missing incidence band and outgoing counts across all 2,662,400 lanes. The new objective, matched optimization, fitting determinism and held-out quality tests did not become eligible.

## Measurement

**Baseline reproduction passed.** All three compiled ER_DF states matched the WL84 audit hashes. Across 48 response lanes, the maximum absolute discrepancy from the archived WL83 `el1`, `ealb`, or `eshape` was `1.47e-8`. The maximum Torch–Dr.Jit deviation was `9.23e-7` relative to peak. The maximum measured reciprocity residual was `2.57e-6` relative to peak on the eligible lanes. The largest analytic albedo upper bound was `0.999984` rounded upward. In the six fresh same-seed renders, the maximum object RGB relative L1 discrepancy for E1, E2, HIGH or TOTAL versus the archived same-arm result was `1.14e-7`. The largest absolute signed discrepancy was `7.77e-9` of reference TOTAL energy. Paired render times were 19.0, 30.9 and 38.3 s. The complete preflight took 111.8 s, including response replay, input hashing and the data audit. [Results](../../evaluation/104C/preflight.json).

| Frozen data | Materials | Original fit lanes | Positive-weight lanes | Incident groups, 55–65° | Positive-weight lanes with outgoing angle in band | Both endpoints in band and within 5° of mirror |
|---|---:|---:|---:|---:|---:|---:|
| `test83_iid` | 300 | 1,228,800 | 1,189,223 | 0 | 173,417 | 0 |
| `test83_block` | 200 | 819,200 | 781,936 | 0 | 107,577 | 0 |
| `test83_hard` | 150 | 614,400 | 603,739 | 0 | 86,051 | 0 |
| **Total** | **650** | **2,662,400** | **2,574,898** | **0** | **367,045** | **0** |

## Observation

The archived arrays reproduce the deliberate 55–65° incident-angle exclusion in WL81's `train_style`, inherited by WL83. All materials have positive-weight groups at incidence angles ≥80°, averaging 10.66–11.13 groups per material across the three sets. The detected failure is specific to the missing incidence band. The grazing counts establish occupancy only. [Incident-angle histogram](../../evaluation/104C/angular_support.png).

WL104B found peak ratios of 0.87 for `iid_1` and 1.29 for `hard_1` at 60°. WL82's historical shape-only diagnostic already reduced `test_iid` median `eshape` from 0.037 to 0.032 when comparing unbounded ER metric and shape fits. That experiment dropped energy fidelity from the shape-only objective and used an independent albedo oracle as a reference diagnostic. It provides evidence that an objective can move shape metrics under those conditions. It does not supply a certified energy-preserving objective comparison or an independent WL104B-scene gain. [WL104B](104B_WORKLOG_LEARNED_ER_ERROR_ATTRIBUTION.md), [WL82](82_WORKLOG_INCIDENT_ENERGY_VS_REFRACTION_ANGULAR.md).

## Interpretation

Every fit pair with an outgoing direction in the excluded band has its incident direction outside that band. The data therefore contain no reciprocal query pair inside the 60° near-mirror neighborhood. An objective can give more weight to supported pairs and may change interpolation into the band through shared ER parameters. A finite-budget failure would leave that interpolation limitation confounded with the loss and the family. The established WL104B lobe-profile finding and baseline reproduction hold under the original fitting procedure.

## Unresolved questions

Whether an objective-only intervention can improve certified ER at the excluded lobe probe and in independent rendered scenes remains open. The influence of optimization, the shape–energy trade-off, grazing accuracy, worst-case held-out materials and independently held-out lighting or views remains unmeasured for an intervention. Fitting costs and candidate states are unavailable because no new fit was run.

## Architecture verdict

**INCONCLUSIVE.** The angular-support preflight returned **DATA_UNQUALIFIED**, triggering the user-defined stop before candidate implementation and fitting. This outcome establishes a limitation of the frozen data for the proposed comparison. The response capacity of the certified ER family remains undetermined by WL104C.

## Reproducibility

- **Runtime.** LabServer63 (`63`), RTX 3080 Ti 12 GiB, NVIDIA 595.71.05, Python 3.10.21, NumPy 1.26.4, Torch 2.3.1+cu121, Mitsuba 3.5.1 `cuda_ad_rgb`, Dr.Jit 0.4.6. Scene seeds were `1049000`, `1059000`, and `1069000` in WL104B image order.
- **Server inputs.** `~/rest_wl104c/` contains the Git archive of the baseline code and the preflight overlay. The archived WL104B inputs are the mesh files in `results/ecro_wl84_104/scene/` and `rep/r000.npz` per scene under `results/ecro_wl84_104b/evidence/`. Both `inputs63/rest_compiled83` paths in those result directories point to `~/rest_compiled83`. Reproduce with the executed commit above using `git archive <commit> experiments evaluation/104B/metrics.json` and the same input layout.
- **Hashes.** The complete metrics JSON records the input, code, protocol, replay and archived-replicate SHA-256 values. The baseline code archive is `d9edfbee4950e07c8a4c6fd92c67876b550d8873d79a51ae4ba35491d1586b9d`, the corrected overlay is `c1a53e9e5590870eca61347798e1c37257818f9b0703f5280f160833fd2046b4` and the input archive is `09553bfe774da403e2ca2d1737bc8673feee8b564597452b4a6ae4f508f7aaf4`. Metrics JSON `642522cc21f738b589549a554cbbdd542f60827f9dfd0239931755642a3f95ed` and incident-angle figure `545ec34712c2866803068a77e5827457f3f1343deba398fe4aed334486361783`. The figure uses the same hashed inputs and plot script `98fb73dfe2f7e7808d51f70d14cfc796129ef607e7014ff6c873ccd4c58766ad`.
- **Exact run outputs.** `~/rest_wl104c/primary_retry.log` and `~/rest_wl104c/results/ecro_wl84_104c/primary_retry/`. The three `*_replay.npz` files contain the fresh per-event RGB buffers.

```bash
cd ~/rest_wl104c
source ~/rest_compiled84/env.sh
commit=d5f46982743f898f8619360cad08af8722958b30
out=results/ecro_wl84_104c/reproduction
"$PY" -B experiments/ecro_wl84_104c/preflight104c.py \
  results/ecro_wl84_104b/evidence "$out" --commit "$commit"
"$PY" -B experiments/ecro_wl84_104c/plot_coverage104c.py \
  ~/rest_compiled83/results/data "$out/preflight.json" "$out/angular_support.png"
```
