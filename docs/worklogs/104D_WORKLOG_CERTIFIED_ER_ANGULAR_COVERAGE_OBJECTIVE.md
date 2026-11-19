# Worklog 104D — Certified ER angular coverage × fitting objective

## Authorization and freeze · 2026-11-19 KST

The owner authorized one bounded follow-up to WL104C on `ecro_lr3_adapter`. WL104C's missing-band result is the starting constraint. The question is whether coverage, fitting objective, or their interaction improves angular and rendered fidelity while using the same certified ER family. This continues the owner's prior server assignment on LabServer63.

The branch started at `660cf98d6ae7535726aed61d4f65cf569d296988`. Execution code and the prospective protocol were frozen at `68f803d82a3a289261437e344ebafe7135777c73`. The protocol SHA-256 is `ddf5f37da7a391d3db33e3244095b1ade4a6cf25157e95dc069030333cee2f45`. The experiment uses the original layered BSDF, 64-parameter certified ER, 69-float runtime state and WL104B's six-event path tracer. [Protocol](../../experiments/ecro_wl84_104d/protocol.json), [WL104C](104C_WORKLOG_CERTIFIED_ER_ANGULAR_OBJECTIVE.md).

| Arm | Direct-fit data | Objective |
|---|---|---|
| A | Original WL83 | `el1 + ealb` |
| B | Original WL83 | `eshape + ealb` |
| C | Reallocated angular support | `el1 + ealb` |
| D | Reallocated angular support | `eshape + ealb` |

The WL82 shape-only result provides a historical diagnostic. Under unbounded ER on its IID population, the metric and shape-only median shape errors were 0.037 and 0.032. WL104D keeps the directional energy penalty and the certified mode. Each arm uses the same generic and deterministically perturbed starts, WL82 Adam schedule and 12,000 updates per start. The lower terminal training objective selects the state per arm and material. Both starts must qualify under the same last-2,000-step rule. Historical WL83 ER_DF and ER_L are retained as external-to-factorial material comparators for the same descriptors. [WL82 fit evidence](82_WORKLOG_INCIDENT_ENERGY_VS_REFRACTION_ANGULAR.md).

The nine descriptors comprise the three WL104B development materials and two seed-selected, uninspected-by-WL104B descriptors from each WL83 IID, block and hard population. The held-out designation is relative to WL104D intervention decisions. The population had appeared in WL83 aggregate evaluations. Independent angular measurement comes from fresh proposals and walk seeds. Scene evaluation uses the original development camera and lights. The six held-out material scenes use an independently specified camera with an unchanged WL84 light selected by a fixed alternating rule.

## Stage A · Baseline and angular access

The focused tests passed for proposal geometry, quota and PDF checks, objective differentiation, agreement with the established metric, bitwise deterministic short fits, the passivity certificate, and event statistics. The full WL104C replay reproduced its lane/coverage results and audit hashes. Across its six fresh 4,096 spp, same-seed images, the largest object RGB relative L1 discrepancy was `1.14e-7`.

The original and reallocated design use 4,096 lanes and 4,194,304 reference walks per material. Arms C and D retain 48 original incidence groups per material byte-for-byte and replace eight on each side of 55–65°. Their 16 new groups jointly stratify incident polar angle and azimuth. Each uses 32 original-proposal draws, 16 uniform-solid-angle core draws within 3° of mirror, and 16 shoulder draws in the 3–12° annulus. The balance-heuristic mixture PDF preserves the solid-angle integral and WL83's zero-weight rejected lanes. The 9,437,184 new fit walks replace a like number of original slots. The separate reciprocity audit uses 9,437,184 fresh walks and supplies no fit targets.

All nine materials passed Stage A. Each supplied 16 occupied incident cells, at least 16 active draws per region and group, and unchanged grazing group counts. Minimum reference-contribution ESS across material/polar strata was 63.87 in the core and 48.83 in the shoulder. Maximum half-to-half discrepancies were 1.00% and 1.04%. Maximum forward/reversed weighted reference mismatch after dividing out the outgoing cosine was 0.301%.

The fixed-budget design removes support on the flanks of the filled band. An independent array check verified that all 432 retained groups were bit-identical. Over all nine materials, incident-group counts went from 30 to 0 at 50–55°, from 0 to 144 at 55–65°, from 55 to 3 at 65–70°, and from 109 to 109 at 80° and above. Positive-weight lanes went from 35,709 to 35,904 because the new proposal rejects fewer draws. The coverage factor is therefore a redistribution of angular and proposal budget with new core and shoulder support, and conclusions apply to that entire redistribution. The response evaluation includes flanking and grazing probes. Evaluation at 55–65° retains independent direction pairs and walks, and that band is inside the training distribution for C and D. [Incident coverage](../../evaluation/104D/angular_coverage.png).

## Evidence and closeout · 2026-11-19 KST

### Implementation Fact

On LabServer63 the frozen execution advanced from Stage A through the matched fit and into independent reference qualification. It used the original WL82 `group_terms`, WL83 robust refraction path, `cert` decoder and the same layered random-walk evaluator for every new reference. Shared representation and renderer source files were preserved. While the first run was reproducing the baseline, automated review rejected an interruption proposed to revise the flank-group replacement rule. The frozen run continued on its original protocol.

### Measurement

- **Optimization.** All 72 fits (four arms × two starts × nine materials) completed 12,000 updates and passed convergence. Maximum absolute own-objective change over the final 2,000 updates was `9.00e-5` / `2.16e-4` / `1.45e-4` / `5.27e-5` for A/B/C/D against the common `0.005` limit. Joint optimization took 447.24 s and peaked at 396,998,144 Torch-allocated bytes. Both starts and final selected states are present in the fit archive. [Fit metrics](../../evaluation/104D/fit.json), [fitted states](../../evaluation/104D/fit_states.npz).
- **Independent reference.** The fresh angular set used 1,024 directional lanes at each of ten polar angles and two azimuths, with 4,096 layered walks per active lane split into independent halves. The separate peak grid used its own directions and walk stream. The 55–65° mean half-split shape noise exceeded the `0.015` limit for `test83_hard_76`, at **0.0183944**. The other eight materials passed the band check. All nine passed the grazing-shape limit (maximum 0.0159802 versus 0.025) and albedo limit (maximum 0.0007523 versus 0.008). [Precision metrics](../../evaluation/104D/response.json), [precision figure](../../evaluation/104D/reference_precision.png).
- **Independent audit.** Recomputing from the raw reference halves reproduced 0.0183944. The six in-band lane-group values for the failing material ranged from 0.01692 to 0.02104, across the three polar probes and two azimuths. The audit also checked all primary file hashes and confirmed 72 finite terminal states.
- **Stages after the gate.** The completed reference arrays and fit states were archived before exit. The decision gate prevented independent candidate response scoring, candidate runtime and physical-contract measurements, and the factorial scene renders. The WL104C reproduction images remain baseline-only evidence. Total execution time was 583.05 s. [Recorded stop](../../evaluation/104D/status.json), [baseline replay](../../evaluation/104D/baseline_preflight.json).

### Observation

The new training set met separate angular-access, contribution-ESS, reference-half and reciprocal-direction criteria on each of nine materials. The failure occurred in the independent reference's angular shape, after all fits passed their own convergence checks. Its magnitude was 22.6% above the threshold for one preselected held-out hard material, and both azimuths at all three target angles showed the same precision problem. Albedo precision remained within its own limit.

### Interpretation

The budget redistribution resolved WL104C's missing local training support at 60° and thinned neighboring angular support. Adequate fit support and precision of the independent pointwise shape comparison are separate requirements. The latter is unresolved at the frozen walking budget for `hard_76`. Convergence reflects minimization of each arm's own objective on its training distribution. The terminal objective values do not answer whether coverage or loss improved independent angular or rendered fidelity.

### Unresolved Question

The effects of data coverage, objective and their interaction on independent responses and scene fidelity remain unresolved. Peak, core, shoulder, tail, grazing and image-quality effects of individual candidates are unmeasured. The fixed-budget flank trade-off also limits which causal interpretation a future qualified comparison of these exact datasets could support.

### Architecture Verdict

**INCONCLUSIVE · REFERENCE_UNQUALIFIED.** Stage A resolved the targeted data-support gate and Stage B produced matched, converged fits. Stage C stopped at the frozen reference-precision gate. WL104D establishes the conditions of the stop. Evidence of a coverage effect, objective effect, interaction or hard capacity limit is unavailable.

### Reproducibility

- **Raw evidence.** `~/rest_wl104d/results/ecro_wl84_104d/primary/` and `~/rest_wl104d/primary.log`. The raw directory holds original and reallocated training arrays, exact fits, independent reference halves and peak-grid arrays, baseline replays and the stopped status.
- **Runtime.** Host `63`, RTX 3080 Ti 12 GiB, NVIDIA 595.71.05, Python 3.10.21, NumPy 1.26.4, Torch 2.3.1+cu121, CUDA 12.1, Mitsuba 3.5.1 `cuda_ad_rgb` and Dr.Jit 0.4.6.
- **Inputs.** `code.tar` was made with `git archive <frozen commit> experiments evaluation/104B/metrics.json evaluation/104C/preflight.json` on the Windows checkout with `core.autocrlf=true` and hashes to `392b59d1b508c1daba16d316a4f5aa3d20a7a3d772803e120c6790ca3d0eeb8d`. The executed hashes use the archive's CRLF bytes. `masks.tar` contains the six archived WL104 layered image halves reserved for development ROI construction and hashes to `4332f06ed0c5e5c5abdb14a0360b760f19e572c56893c440dd7b5e6f44214052`. Existing WL104B and WL104C baseline inputs are linked read-only from the WL104C deployment and frozen WL83 inputs under `~/rest_compiled83/results`.
- **Output identities.** Source inputs, code, settings, result files and all seeds are listed in [provenance](../../evaluation/104D/provenance.json). The figure script is post-processing with hash `eed9a492e77b39ab698748c7191d1cacc517fbc4d2904b16ef627e9b8f39f3c7`.

```bash
cd ~/rest_wl104d
source ~/rest_compiled84/env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
commit=68f803d82a3a289261437e344ebafe7135777c73
out=results/ecro_wl84_104d/reproduction
"$PY" -B experiments/ecro_wl84_104d/run104d.py ~/rest_compiled83/results "$out" --commit "$commit"
"$PY" -B experiments/ecro_wl84_104d/figures104d.py "$out" reviewer_reproduction
```

## Amendment 1 · Owner-approved reference precision extension

The owner approved one continuation on `ecro_lr3_adapter` after the frozen reference-noise stop. This allocation was post hoc to the reference-only failure in `test83_hard_76` and preceded candidate scoring. The original execution, its protocol and its **REFERENCE_UNQUALIFIED** stop remain the historical primary result. Amendment 1 code and sampling allocation were frozen at `c46015c99d6729a3354efd7184342c0748934e32` before generating any new walks. The amendment protocol at execution hashes to `08de92c851a1fa8bfbd11182a3544f0e92a341bf65315fc2fa73051764fde911`. [Amendment 1 protocol](../../experiments/ecro_wl84_104d/amendment1/protocol.json).

### Implementation Fact

The amendment verified the original protocol, executed-code and data hashes, all original result hashes, the replay inputs, and the 72 fitted states and their selected indices. It reproduced the frozen precision calculation exactly. The six target groups are indices 148–153 in the existing direction array, with polar angles 57.5°, 60° and 62.5° at azimuths 22.5° and 112.5°. The original layered evaluator sampled exactly 12,288 new walks per saved direction pair, with an independent seed and disjoint per-walk PCG stream sequences. The two original 2,048-walk half means were retained separately and combined with their respective 6,144-walk additions, weighted 1:3. Each active lane now has two independent 8,192-walk halves.

The nominal increment was 75,497,472 walks. Of these, 74,858,496 contribute to the estimator. The 638,976 draws on 52 rejected, zero-weight proposal lanes were stored with the added samples, and their original combined placeholder values were preserved. The focus checks verified sample and half counts, stream intervals, half reconstruction and the preceding gate result. All reference fields outside the active target lanes, including the entire peak-grid reference and the original PDFs and masks, were checked for bitwise equality. The original states supply A/B/C/D through the parent selection rule. There was no new fit.

### Measurement · Independent reference and physical contracts

| Polar | Azimuth | Original half-split shape noise | Amended half-split shape noise |
|---:|---:|---:|---:|
| 57.5° | 22.5° | 0.016917 | 0.008685 |
| 57.5° | 112.5° | 0.017165 | 0.009209 |
| 60° | 22.5° | 0.018154 | 0.009211 |
| 60° | 112.5° | 0.019145 | 0.010002 |
| 62.5° | 22.5° | 0.017944 | 0.009560 |
| 62.5° | 112.5° | 0.021042 | 0.010395 |
| **Material mean** | | **0.0183944** | **0.0095104** |

The original material-mean limit of 0.015 passes, and the original evaluator reproduced the amended precision rows exactly. For `hard_76`, grazing shape noise stayed at 0.0159802 and aggregate albedo noise moved from 0.0007523 to 0.0005438. The reference increase took 2.73 s on LabServer63. Every independent physical check passed across the four intervention arms and two historical comparators (54 compiled material records). The maximum normalized Torch–Dr.Jit residual was `1.05e-6`, maximum normalized reciprocity residual was `2.03e-14`, and the analytic albedo upper bound was at most 0.999999979. The original 64 parameters and 69 compiled floats per material were verified. Per-arm median Dr.Jit evaluation time was 9.95–10.38 ms for 20,480 evaluations, including data transfer. The 447.24 s joint fit cost belongs to the original primary run.

### Measurement · Independent angular fidelity

The table reports medians of paired per-material reductions in `eshape` at 55–65°. Positive values favor the first arm. The development and held-out medians are separate, as in the original decision rule.

| Contrast | Development | Held out |
|---|---:|---:|
| C − A, redistribution with metric objective | +0.00208 | +0.01463 |
| B − A, shape objective with original data | +0.00046 | +0.00081 |
| D − A, both interventions | +0.00186 | +0.01522 |
| D − B, redistribution with shape objective | +0.00204 | +0.01220 |
| D − C, shape objective with redistributed data | +0.00020 | −0.00122 |
| Average redistribution effect across the two objectives | +0.00174 | +0.01614 |
| Average objective effect across the two distributions | +0.00012 | +0.00031 |
| Factorial interaction in gains | +0.00008 | −0.00151 |

All nine materials had lower in-band `eshape` in C and D than in A. For both reference halves, C met the 0.01 absolute minimum on the held-out median (0.01441 and 0.01447), and the development median gain stayed near 0.002. D followed the same split. `iid_1` had gains of 0.02910 in C and 0.03137 in D, while `hard_1` remained a worst case (band `eshape` 0.11425 in A, 0.11308 in C and 0.11288 in D). `hard_76` improved by 0.00888 in C and 0.00999 in D.

For A versus C, the median in-band peak relative absolute error decreased from 13.59% to 4.09% in development and from 14.21% to 5.38% held out. In the same order, median normalized core/shoulder/tail mass errors for C were 0.0055/0.0074/0.0129 and 0.0032/0.0146/0.0189, compared with A's 0.0134/0.0207/0.0380 and 0.0050/0.0230/0.0339. The peak grid was independent of fitting and kept its original walks. The corresponding raw files give peak offsets, per-probe `el1`, `ealb` and `eshape`, both reference halves, and signed region energy.

C regressed the development median shape at 50° by 0.00297 and improved it at 70° by 0.00153. Held-out median gains were 0.00469 at 50° and 0.00622 at 70°. The C held-out median grazing shape error increased by 0.00582, with a maximum per-material increase of 0.01496 inside the frozen 0.02 limit. Relative to A, B worsened aggregate albedo error by 0.00850 for `block_161`, exceeding the 0.008 regression limit. Applying the shape objective to the redistributed data (D versus C) worsened aggregate albedo on two held-out block materials by as much as 0.03297. The data factor includes the lost flank support as well as the new target band and directional proposal.
