# Worklog60 — Frozen learned-K fidelity attribution (2026-10-29)

## Authorisation, scope and evidence inventory

The user authorised a bounded evidence-first diagnosis of the learned-K fidelity failure shared by WL59 Full Rebuild F and Selective Update S. This is **not** a replacement architecture, new training, angular sweep or update-efficiency campaign. [Diagnostic scope](../plans/FROZEN_K_ATTRIBUTION_V1.md) was committed with the analysis source before deriving causal interpretations. Starting source: `f0035fbbc69a405c17f13204e9653ddd75608d1b`, branch `Selective_Recompute`; no worktree/branch change.

Read the current research topic, architecture, roadmap/evidence roles, WL57–59, WL59 protocol/checkpoints/final evidence and the WL57 operator/data/geometry/training implementations needed to interpret them. Earlier reference findings are inherited through these sources, not independently rerun. No broad repository/literature review was conducted.

Available locally: both WL59 checkpoints, complete held-out score arrays/CSV, two training histories, original protocol, data/checkpoint provenance and post hoc independent-reference sensitivity. Available read-only on LabServer69: both complete stored-action files, original geometry/B, all incident fields, original training/validation targets, independent old rest target replicas and fresh primary reference replicas. Checkpoint parameters remain **157,024** per seed. No optimizer or model parameter was changed.

Large inputs remain in their original server locations. New code is isolated under `experiments/fnto_attr60`, derived measurements under [results/fnto_attr60](../../results/fnto_attr60/), and reviewer tables under [results/evaluation/60](../../results/evaluation/60/). No historical Worklog, original result, protocol or production code was edited. Living documents remain unchanged because no corrective architecture decision is justified.

## Implementation Fact

Implemented CPU-only stored-output, subspace, label-precision and small frozen-checkpoint replay diagnostics. No CUDA execution, reference generation, learning, fitting of a replacement operator, cache update or C/solve integration occurred. SVD/projection is an analytic relaxation for a lower bound, not trained model output.

Four focused tests validate flux aggregation, rank/projection bounds and the component-error identity. A standalone CPU transcription of the frozen checkpoint algebra was independently checked against immutable primary predictions (fields 8/9/62/63): relative L2 **1.0334e-7 / 1.0149e-7** for seeds 5901/5902. This replay validation precedes interpretation of its train/validation outputs.

Main analysis time: **33.14 s CPU**, peak RSS **3,558,848 KiB** on Linux. Compatible old-label/coverage analysis: **12.55 s**. Native-split frozen replay: **16.29 s**, retained separately; crossed replay: **19.06 s**. These include diagnostic loading/computation/hashing and are not neural runtime benchmarks. The local system Python lacks Torch; analysis used the existing server environment with `CUDA_VISIBLE_DEVICES=""`, four Torch threads and one OpenBLAS thread. The first local bytecode-cache write was sandbox-denied; syntax was checked in memory instead. No scientific computation failed or changed budget.

## Measurement — primary failures before secondary diagnosis

These scores reproduce the original WL59 definitions and populations. Primary pose is `pt0`; rest/pt0 use the same fixed 64 test fields. Signed-delta masks are reference-only, restricted to the first 16 fields. Directional pairs use fields 48+2k and 49+2k. No mask or threshold was changed.

| Seed | Absolute coarse mean RGB, pt0 | Primary delta error RGB | Qualified delta fields RGB | Pair 7 contrast error RGB | Directional cells failing / 24 |
|---|---|---|---|---|---:|
| 5901 | .061526 / .062000 / .066766 | .190814 / .189734 / **.410085** | 11 / 10 / 8 | **.507076 / .490890 / .421280** | 5 |
| 5902 | .061809 / .062704 / .066364 | .209915 / .200201 / **.408700** | 11 / 10 / 8 | **.487344 / .473408 / .434919** | 4 |

All primary directional cells qualify. The original delta error bound is .35, with correlation >= .90; directional error bound .25. Blue delta correlations remain .9134/.9146: the failure is not absence of correlation alone. Absolute coarse/fine16 norm bounds and algebraic physical constraints pass, as originally reported. F/S discrepancy remains at approximately 1e-7; the same failures occur in F, so S-specific numerical inconsistency does not explain them.

Cross-seed **residual-vector cosine**, not merely score similarity: blue primary delta **.98979**; pair 7 R/G/B **.98721/.96645/.99761**. The common residual is highly reproducible across these two training seeds, but shared labels, architecture, objective and geometry all remain confounded.

The existing matched-population reference review gives mean-reference delta noise norms **.03058/.02897/.04398** relative to the same qualified delta signals. Pair 7 mean-reference contrast noise is **.02332/.02182/.01636**. Replica-wise blue delta errors remain .410–.412 and worst directional errors about .420–.509. Choosing the other existing reference seed does not remove the failures. These are descriptive two-replica estimates, not confidence intervals or proof of zero systematic bias.

## Mathematical bounds — what the output family actually excludes

The frozen model has `Kx = Bx + O h(x,G)`, with eight outgoing angular mixtures shared across material cells for each channel. At the evaluated 1 cm / 16-out interface, aggregate the original 64-bin O by exactly the original flux-sum map. Each frozen O has rank eight. Because O is geometry-independent, learned scattered **deltas and directional contrasts** also lie in its span.

Define the **effective residual target** `r = reference - Bx` using the existing approximate geometry-only B. This is NOT an independently measured collided-light component. With orthogonal projector P onto the frozen O span, `||(I-P)r|| / ||reference||` is an optimistic output-span lower bound: it permits arbitrary signed, independent latent values at each receiver, ignoring incoming encoding, positivity, spatial coupling and geometry conditioning. For contrasts/deltas, subtract the corresponding B contrast/delta and normalize by that total reference contrast/delta.

The stored learned residual lies in the computed span to <= **2.41e-6** relative error, validating this decomposition against actual predictions.

| Primary quantity | Seed | Observed error RGB | Frozen outgoing-span floor RGB |
|---|---:|---|---|
| Signed delta | 5901 | .190814 / .189734 / .410085 | .049038 / .032455 / .058018 |
| Signed delta | 5902 | .209915 / .200201 / .408700 | .048266 / .035615 / .074463 |
| Pair 7 contrast | 5901 | .507076 / .490890 / .421280 | .077743 / .077555 / .088511 |
| Pair 7 contrast | 5902 | .487344 / .473408 / .434919 | .077629 / .069510 / .091415 |

By orthogonal error decomposition, only **2.16–4.42% of pair 7 squared error** and **2.00–3.32% of blue-delta squared error** lies outside the frozen outgoing span. Thus the exclusion from that span alone is not the dominant observed failure. This does **not** show that the remaining error is attainable by the actual constrained learned model or that increasing rank would have no interaction benefit.

An even looser SVD control allows a freely chosen rank-eight output basis **separately for each pair/channel**, with arbitrary signed spatial factors and fixed B. Pair 7 residual floors are **.01214/.01095/.00785**; the maximum over all primary pairs/channels is **.01633**. This relaxation is intentionally weaker than one globally shared positive kernel; its small floor does not prove whole-family adequacy or select a new basis. It supplies no information-theoretic rejection of the volume interface.

## Diagnostic observation — incoming information and B/scatter boundary

The 16-bin input API preserves the declared pairs as distinct fields; their incoming bins differ. Under the frozen A projection of the remaining `(1-beta)x` budget, **none** of the 24 primary pair/channel fields collides exactly for either checkpoint (minimum encoded contrast norms .001400/.001682). A maps 16 bins to eight mixtures and necessarily has a nullspace, but no physical oracle response to a constructed nullspace field is available. No input-information impossibility for the actual failed cases is established. WL58's sharp-GGX result is not evidence for a volume information limit here.

B is available as a computed component; a true independent physical B/scattered decomposition is not stored in the original action/half accumulators. Algebraically subtracting the same B from prediction and target leaves their total error unchanged and cannot assign that error to learned scattering.

For primary delta, computed B-delta norm / total target-delta norm is **.04754/.05331/.13138** RGB. For pair 7 it is **.90959/.91867/.98685**. These magnitudes are not component error estimates: components can cancel in a signed contrast. A fixed-B positivity floor on individual absolute actions, `||positive(Bx-reference)||/||reference||`, is tiny (maximum <= **.000807**, with primary <= .000749). This rules out a large **forced absolute-action error from B already exceeding total reference**, but does not validate B's directional quadrature or isolate its physical error. No B optimization or resolution sweep is warranted from these measurements alone.

## Measurement — supervision precision, population and provenance

WL59 reused unchanged WL57 training-only targets. Source lineage copies the exact 96 train and 32 validation fields into A2, while generating a different test population (seed 578103). Training targets are original first-exit sparse-operator actions with 2048 photons per input cell, one r0 training realization. Final primary evaluation uses two independent 64M action bundles per field, with importance sampling and independent-half provenance. Do not quantitatively transfer noise estimates between those estimators/populations.

A compatible old-label control **is** available at rest: original r0/r1 files evaluated the same train/validation fields. Use `(r0-r1)/sqrt(2)` as a descriptive single-label noise estimate, normalized by their midpoint signal, not by the primary test signal.

| Old rest population | Mean coarse absolute single-label noise RGB | Directional-contrast single-label noise range, all pair/channels |
|---|---|---|
| 96 train fields / last 8 pairs | .01367 / .01356 / .01479 | **.05359–.45672** |
| 32 validation fields / last 8 pairs | .02090 / .02055 / .02169 | **.05912–.18045** |

This establishes that small contrasts in some original supervised fields are much less precise than their absolute actions. It does not establish that label noise caused the primary pair 7 or blue-delta failure. Only rest has this compatible replica control; independent deformed train/validation target replicas needed to quantify their signed-delta uncertainty are unavailable (e.g. tr02_r1 is absent). Shared sparse-operator samples also correlate field errors; pair cells are not independent statistical trials.

All splits contain **16 one-hot incoming fields**, or eight directional pairs. Therefore “training had no directional excitation” is contradicted. Train generator widths differ from test widths and non-pair smooth angular populations differ, but pair replacement overwrites angular fields: stale `kappa`/direction metadata is not the final pair input. Actual failed test pair 7 uses bins **8 and 13**, which are not the same-bin sub-bin collision of WL58. Its spatial effective support is **168/163/163** material cells RGB; train directional-field red supports range **14.7–923.3**. Pair 7 is not outside that scalar support range, although its joint spatial/angular pattern is different (nearest full training-field cosine .24945). These descriptors do not prove coverage adequacy or a causal distribution-shift mechanism.

## Measurement — fitting histories and crossed frozen replay

Validation in the frozen training code evaluates **absolute action only**, not signed deformation or directional-contrast error. Training samples paired rest/current actions with `loss = action + .25*delta`; the delta denominator also includes `.0001*||rest action||²`. No explicit directional-pair difference loss is used, although directional fields enter action training. These are objective facts, not proof of the best loss or of causal imbalance.

| Seed | Initial validation action loss | Final / best logged validation loss | Last five validation-loss range | Median weighted-delta fraction of last five logged train losses |
|---|---:|---|---|---:|
| 5901 | .054381 | .010202 / .010165 | .010165–.010282 | **.9510** |
| 5902 | .054613 | .010262 / .010228 | .010228–.010349 | **.9502** |

The recent weighted delta term dominates the logged loss; an explanation that it was simply absent or numerically negligible is contradicted. Logged gradient norms before clipping are about .0147/.0153 median, below the clip of 1. These are sparse two-field snapshots, not full training risk, a gradient convergence proof or evidence that all optimization directions are resolved. A validation plateau of this static objective does not certify the failed delta/directional objectives. Test static objective replay from existing scores is about .0090–.00915, so the scalar action evidence does not indicate a gross static held-out collapse; the populations still differ.

A small prospective replay subset crosses **the same training and validation light subsets** over rest, trained dent tr11 and held-out dent va1. tr11 is chosen from geometry parameters as the closest training dent to pt0; va1 is the deeper validation dent. Train fields are 8/9/94/95, validation fields 8/9/30/31. No field/pose was selected by favorable residuals. The first two fields estimate focused signed response; the last two form one directional pair. The native-split result was retained before adding the crossed control.

| Geometry / lights | Focused delta error RGB, seed 5901 | Seed 5902 |
|---|---|---|
| tr11 / train | .1910 / .1653 / .2299 | .1991 / .1707 / .2281 |
| tr11 / validation | .3022 / .2974 / .4554 | .3068 / .3023 / .4523 |
| va1 / train | .6188 / .6617 / .9329 | .6177 / .6607 / .9326 |
| va1 / validation | .7698 / .7949 / .9545 | .7671 / .7949 / .9541 |

These comparisons use original noisy r0 labels and **have no independent delta-SNR qualification**. They are diagnostics, not new pass/fail tests. The geometry-associated degradation persists when the same training lights are used, making a pure light-split-only account inadequate for this subset. Actual physical generalization failure remains only weakly indicated because different delta magnitudes, deformation-label noise, limited sample breadth and training exposure remain confounded. The sampled directional-pair errors range .131–.234 across these conditions, smaller than the primary worst pair but not a full train/validation directional evaluation.

## Secondary diagnostic geometries

Only after primary analysis, retain the WL59 historical cf0–cf3 populations as secondary. Coarse absolute errors stay near .061–.067. Blue signed-delta errors span **.2967–.5799**; cf2 is the relatively better case, while other blue cases remain poor. Directional failures persist (4–5 of 24 per pose/seed). These geometries were historically inspected in WL57 and are not independent primary replications. [Complete primary/secondary table](../../results/evaluation/60/all_primary_secondary_failures.csv) includes every qualified and unqualified recorded group without replacing primary failures.

## Interpretation — compact attribution table

Verdicts below apply to the stated causal claim, not the entire topic named in the first column.

| Hypothesized cause | Supporting evidence | Contradicting evidence | Remaining confound | Verdict |
|---|---|---|---|---|
| S-specific numerical/cache inconsistency explains shared fidelity failure | None in tested outputs | F and S agree ~1e-7 and F already fails | No universal algorithm claim | **CONTRADICTED** for measured shared failure |
| Independent primary test-reference seed noise alone explains failure | Reference noise is nonzero | Same-population errors persist against both replicas; noise much smaller | Systematic bias is not excluded | **CONTRADICTED** as sole explanation |
| Exclusion from the frozen outgoing span dominates the primary residual | Nonzero computed span floors | Only 2.16–4.42% of worst-pair squared error and 2.00–3.32% of blue-delta squared error is outside it | In-span response may be unattainable under other constraints | **CONTRADICTED** as dominant explanation |
| Shared spatial/angular kernel or input compression is fundamentally inadequate | Finite shared positive family; rank-reducing input map | Loose rank-eight floors are small; actual primary encoded pairs do not collide | Joint positivity, input, spatial and geometry constraints untested for attainability | **INCONCLUSIVE** |
| No directional training / no effective delta objective | None | All splits have eight one-hot pairs; weighted delta is ~95% of recent logged train loss | Objective may still not fit the evaluated contrasts | **CONTRADICTED** in that literal form |
| Noisy original supervision contributes materially | Compatible rest train directional noise up to .457; inherited noisy delta labels | No controlled clean-label intervention; rest absolute label noise is much smaller than learned bias | Deformed-label precision, estimator bias and optimization | **WEAKLY INDICATED** causally; noisy-label existence **SUPPORTED** |
| Geometry-conditioned response generalizes poorly | Crossed replay degrades at va1 on the same lights; two seeds agree | No independently qualified clean delta for this subset; pt0 is less poor | Delta amplitude, label noise, only two focus fields | **WEAKLY INDICATED** |
| Optimization/fitting or objective mismatch causes the failure | Residuals persist within output span; validation does not assess failed differences | Static validation improves and plateaus; no optimization intervention | Capacity, noisy gradients, constrained model family, full train risk | **INCONCLUSIVE** |
| Approximate B forces large unavoidable absolute error | B is approximate and important in pair 7 | Fixed-B positivity floor <= .000807 | No independent B/scattered oracle; cancellation and remaining-budget normalization | **INCONCLUSIVE** for directional error; large positivity obstruction **CONTRADICTED** |

The main supported new facts are reproducible common residual structure, weak old supervision for some contrasts, and the limited explanatory power of output-span exclusion alone. None isolates a corrective mechanism. Do not infer an information-theoretic failure from neural prediction error, and do not infer that learning longer, changing rank, altering conditioning or cleaning labels must fix it.

## Unresolved Question — smallest distinguishing next control

**Conclusion B: insufficient causal evidence to select one corrective hypothesis.** Stop diagnosis before new labels, optimization or representation changes.

The smallest next control for the newly exposed confound is a **clean paired delta-label audit on the same two existing training fields 8/9**, at **rest, trained tr11 and validation va1**, with the frozen checkpoints unchanged. It asks: is the crossed geometry gap a physical model error or predominantly uncertainty/bias in the original signed supervision? Original r0 labels and frozen predictions already exist; the missing information is independent, sufficiently precise reference actions on these same field/geometry combinations.

Minimum fixed data: 3 geometries × 2 fields × 2 independent 64M-photon action replicas (each replica split into two 32M halves), same WL59 material/boundary conventions. This is **768M photons**, approximately **62 s of reference execution** extrapolated from WL59's 8.192B / 660.5 s on the same hardware, plus setup, validation and I/O. This is a cost estimate, not a new timing promise. Check uncertainty of the **delta**, not merely absolute-action precision; if it remains unresolved at this fixed budget, report that rather than launching a budget sweep. No new Cycles validation or systematic-bias proof is implied by using the inherited estimator.

Interpretation predeclared for a future separately authorised control:

- If the original labels disagree materially with precise actions while frozen predictions track the old labels, label contamination becomes supported for that subset; a clean-label correction is still a separate decision.
- If precise actions confirm the va1/tr11 gap on the identical fields, a geometry-dependent physical generalization defect is supported on that subset. This does not distinguish a restrictive geometry/kernel family from how it was fitted.
- If both trained and validation predictions fail precise responses, the problem already exists in fitting/representing training conditions. Existing label precision alone is not a sufficient explanation.
- If neither a physical gap nor material label disagreement remains, the noisy small replay subset was not diagnostic of the primary failure; return to the untouched primary populations.

The directional-primary question remains separate: existing pair-7 replicas already resolve total contrast but do not expose true ballistic/scattered components or constrained-model attainability. If component attribution becomes the next question, the minimum extra datum is an independently tagged zero-collision action for the two fixed pair-7 fields at pt0, rather than an angular sweep or optimized B. That is not required or executed here and is not sufficient by itself to resolve blue-delta supervision.

No single passive read-only comparison can, in general, distinguish insufficient model family from failure to optimize that family. If clean-label checks eliminate supervision as the main explanation, a later explicitly bounded feasibility/fitting intervention would still need its own decision. This batch does not implement or authorise it.

## Architecture Verdict

No architecture adoption or rejection beyond the existing WL57–59 verdicts. Physical K/C ownership and F/S contracts remain unchanged. The tested recipe remains physically unqualified; abstract persistent-state reuse remains INCONCLUSIVE. WL58 sharp-GGX failure is not transferred to the volume model. WL56's complete image-level blue floor is neither attributed nor repaired by these K-only diagnostics; no coupled solve ran here.

## Reproducibility and closure

Code was committed **before** each derived measurement/interpretation:

- `f051cec`: scope, stored-output bounds, four tests and local score/history summary.
- `b423de4`: compatible old-label precision and input-coverage/positivity controls.
- `f988b06`: native-split frozen CPU replay, preserved as `replay_native.json`.
- `8cb7f6f`: matched geometry-by-light replay, `replay.json`.
- `05f1b20`: complete reviewer-table exporter.

Original WL59 scientific source, checkpoint/data versions and inherited reference history remain in its original [provenance](../../results/fnto_persist59/final/provenance.json). Checkpoints used here:

| Checkpoint | SHA-256 |
|---|---|
| P_5901.pt | `59685b214bdf1f81079dcd56fe7a61173b90640b3c5c9fb8b3f14094729aaad6` |
| P_5902.pt | `c92e4fda4bbe9cf5569d55c45a2881d95eaa58df3896748088d060b55f1aab3f` |

Every diagnostic JSON records source commit/script SHA and exact input file hashes, including the large action files, resolved inherited data paths, original target files and geometries used. The local summary records exact original metric/protocol/training-source hashes. The [review manifest](../../results/evaluation/60/manifest.json) binds all derived tables to those diagnostic inputs; [batch provenance](../../results/fnto_attr60/provenance.json) records full commit identities and artifact hashes.

```sh
# CPU only; original completed experiment remains read-only.
CUDA_VISIBLE_DEVICES="" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 \
  ~/NeuralShaderDemo/external/8dna26/.venv/bin/python -m unittest -v test_analysis
# Run each committed script from an isolated source directory, stdout to a fresh diagnostic artifact:
python analyze_stored.py --root ~/rest_persist_v1 --source-commit f051cec
python supervision.py --root ~/rest_persist_v1 --source-commit b423de4
python replay.py --root ~/rest_persist_v1 --source-commit 8cb7f6f
# Local JSON summary/reviewer exports:
python3 experiments/fnto_attr60/summarize.py
python3 experiments/fnto_attr60/export_tables.py
```

These commands do not run WL59's supervisor. Use the recorded source versions for exact native/crossed reproduction and fresh output paths; do not overwrite completed evidence. No replacement model or next hypothesis has been started. This diagnostic batch is complete at the existing-evidence boundary.

Concurrent work note: commit `c8057e4` added append-only visual supplements to WL57–59 and their visual gallery during this diagnosis. Those changes were made by the separate visual-review session, not this batch. Verification compares scientific inputs with the starting source and those three Worklogs with that visual-supplement commit; both comparisons are unchanged.
