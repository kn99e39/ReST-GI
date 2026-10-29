# Worklog62 — Frozen-K primary fidelity evidence reconciliation (2026-10-30)

## Authorisation and bounded scope

The renewed user brief repeats the bounded evidence-first attribution request already executed as WL60. Reuse the completed investigation and its WL61 follow-up rather than repeating reference construction or training. This Worklog records the current primary-failure attribution after reconciling that additional evidence.

Starting source: `2570510`, branch `Selective_Recompute`. [Scope and arithmetic controls](../plans/FROZEN_K_EVIDENCE_RECONCILIATION_V1.md); diagnostic code committed **before use** as `f0a0229`. No new hypothesis, architecture, training, inference, reference generation, expensive GPU execution, B optimization, global-normalization change or angular sweep. No server was contacted. All computations use existing local JSON and the original unchanged checkpoints' provenance. No historical Worklog is edited.

Read current research/architecture/roadmap contracts, WL57–59 and the original WL59 protocol/final artifacts, then WL60–61 to account for completed controls. Earlier physical reference findings remain inherited evidence; they were not rerun.

## Implementation Fact

A small standard-library-only script verifies population identities, joins the existing primary errors/reference sensitivity with WL60 bounds, and exports **all 54 primary diagnostic rows**: two seeds × (three signed-delta channel groups + eight directional pairs × three channels). No row is selected by its outcome. Five focused tests cover arithmetic, undefined/negative estimates and population alias rejection.

The additional arithmetic is explicitly a **post hoc moment-style sensitivity**, using already recorded primary mean-reference noise norm ratios:

```text
sensitivity_relative_squared = (raw_error^2 - noise_ratio^2) / (1 - noise_ratio^2)
```

This is not a new physical measurement, unbiased error ratio, confidence interval, bias proof or revised acceptance rule. The two-replica estimate is descriptive. Negative squared values are retained; an undefined denominator is not converted into success. Raw WL59 errors, qualification and verdicts remain authoritative.

## Measurement — primary population first

WL59's primary population is **pt0 versus rest, A2 test fields**. Signed-delta evaluation uses the original reference-qualified first-16-field masks; blue qualifies eight fields. Directional pair 7 is **test fields 62/63 at pt0**. All 24 primary directional cells qualify for each seed. Neither policy changes the following errors: F/S differences are approximately 1e-7.

| Quantity | Seed | Raw primary error | Mean-reference noise / same signal | Moment-style sensitivity error |
|---|---:|---:|---:|---:|
| Blue signed delta | 5901 | .410085 | .043976 | .408115 |
| Blue signed delta | 5902 | .408700 | .043976 | .406721 |
| Pair 7 R | 5901 | .507076 | .023322 | .506678 |
| Pair 7 G | 5901 | .490890 | .021823 | .490522 |
| Pair 7 B | 5901 | .421280 | .016356 | .421019 |
| Pair 7 R | 5902 | .487344 | .023322 | .486918 |
| Pair 7 G | 5902 | .473408 | .021823 | .473018 |
| Pair 7 B | 5902 | .434919 | .016356 | .434669 |

Original limits are .35 delta relative L2 with correlation >=.90, and .25 directional relative L2. The original blue correlations are .9134/.9146; error magnitude, rather than lack of correlation alone, causes its failure. Errors against each existing primary reference replica independently remain about .410–.412 for blue delta and .420–.509 for pair 7. Primary independent seed noise alone does not explain these failures. The arithmetic sensitivity changes none of those conclusions; it is not an error-bar calculation or new gate.

Absolute response norms still pass (primary coarse mean RGB about .0615–.0668, fine16 about .102–.1115). Hard zero/linearity/nonnegative/passivity constraints pass. Their success is compatible with incorrect directional and deformation responses. Those original results are not rerun or broadened here.

Both seeds have nearly the same residual pattern: WL60's primary blue residual cosine is .98979; pair-7 RGB residual cosines are .98721/.96645/.99761. Reproducibility identifies a common issue, not whether the common cause is representation, training targets, optimization or a shared geometry-only approximation.

## Mathematical and representation evidence retained from WL60

The **frozen outgoing-span** floors are about .058/.074 for the two primary blue-delta predictions and .070–.091 for pair 7. Only 2.00–3.32% of blue squared error and 2.16–4.42% of pair-7 squared error is outside that span. Exclusion from the frozen outgoing span alone is therefore not the dominant residual. The much looser per-pair free rank-eight floors are <=.01633.

Those controls permit arbitrary signed receiver latents and relax the actual positive spatial/input/geometry constraints. Small lower bounds do not prove that the constrained kernel family can realize the response, and do not establish that changing rank would be useless. They do not select a new model.

No primary pair collides exactly under the frozen incoming projection. Nevertheless, the 16-to-8 incoming mixture map has a nullspace, and no measured physical response to a controlled nullspace collision is available. Input information sufficiency remains INCONCLUSIVE. WL58's sharp-GGX histogram impossibility is a different, prospectively declared surface probe; it supplies no homogeneous-volume impossibility theorem.

The two-hop dependency-local encoder, pair/prior features, dense tail and global normalization have different dependencies. Computational selective-update validity does not show that the geometry features are sufficient statistics for physical K. Merely noticing local encoder support is not a controlled geometry-information collision, because pair distances and global reductions can still change responses.

## Observation — ballistic/scattered attribution remains unavailable

The existing computed B and learned scattered outputs are known; an **independent physical zero-collision versus collided decomposition** of the primary reference is not recorded. Subtracting the same computed B from prediction and total reference leaves total error unchanged. It creates an effective residual, not an oracle scattering component.

For pair 7, computed B-contrast norm / total target-contrast norm is .910/.919/.987 RGB. This large magnitude is not B's error. Signed components can cancel, and a small positivity floor does not rule out directional B error. WL60's fixed-B positivity floor on absolute actions is <=.000807; that contradicts a large forced absolute error from B already exceeding total reference, not every possible B-side defect. No error is assigned to B or scattering without independent component data.

## Measurement — supervision, fitting and secondary populations

The original WL57 training-only supervision remains unchanged in WL59: 96 train fields, 32 validation fields, one r0 deformed-label realization, same 6000-step recipe and final-checkpoint rule. Rest r0/r1 controls show that original **train directional-contrast** label noise can be .054–.457 despite only about 1.4% mean absolute-action noise. This establishes weak contrast precision in some supervision, not causation of the primary failure. Noise quantities are never transferred between different fields/estimators.

Training histories show static validation loss falling from about .054 to .0102 for both seeds. Logged validation evaluates absolute actions, not the failed delta/directional tests. The weighted delta term is about 95% of the last five logged training losses, so it was not absent or numerically negligible. Sparse two-field snapshots and a plateau of the static metric do not prove convergence of the relevant objectives. Absence of a dedicated directional-pair loss is a possible objective mismatch, not proof of a remedy.

Historical cf0–cf3 remain secondary, with blue signed errors about .297–.580 and four/five directional failures per 24 cells. Their existence reinforces bounded repeatability but does not replace the independent primary test. Complete arrays remain in WL59/60 evidence.

### What WL61 establishes, and what it does not

The reconciliation verifies **byte-disjoint train/test field arrays** and confirms that WL61 fields 8/9 exactly match the original **train** bytes. Their integer indices do not identify the primary test fields numbered 8/9. WL61 used rest/tr11/va1, not pt0; only rest overlaps geometrically.

WL61's qualified train-field R/G replay errors at va1 fall from about .62/.66 against old r0 labels to about .20/.19 against new references. The apparent geometry gap is 89–94% smaller. Independent-replica squared-error diagnostics leave no convincing severe R/G physical geometry gap on that small subset. **Historical evaluation-label error materially inflated that secondary replay conclusion.** This is a supported, population-specific finding.

WL61 va1 blue has SNR **2.131 < 3**, with final-mean delta SE about .319 of the signal, so it remains unqualified. In contrast, original primary blue is qualified and its mean-reference noise ratio is about .044. Neither WL61's R/G correction nor its unqualified blue can explain away the original primary negative.

WL61 also resolves nonzero trained-geometry approximation error (.134–.175 on the two tr11 fields), below the old .35/.90 limits on those fields. It does not prove optimizer failure or model-family inadequacy, nor that noisy training labels caused the primary bias. The frozen model can be closer to an independent physical action than to noisy old evaluation labels without implying a successful general K.

## Interpretation — compact attribution table

Each verdict applies to the exact stated claim, preserving the required separation of measurements, bounds and causal interpretation.

| Hypothesized cause | Supporting evidence | Contradicting evidence | Remaining confound | Verdict |
|---|---|---|---|---|
| S-specific numerical inconsistency causes shared primary failure | None in tested output | F/S ~1e-7; F itself fails | Tested algorithm scope only | **CONTRADICTED** |
| Primary independent-reference seed noise alone explains failure | Nonzero noise | Primary replica-wise errors and sensitivity remain far beyond bounds | Systematic estimator bias not excluded | **CONTRADICTED** as sole explanation |
| Frozen output-span exclusion dominates | Nonzero span floors | Only ~2–4.4% of the important squared residual is outside it | Actual constrained in-span attainability | **CONTRADICTED** as dominant explanation |
| Full spatial/angular/kernel representation loses required information | Finite constrained family and rank-reducing incoming map | Loose bounds small; no actual primary encoded collision measured | Positivity, spatial structure, geometry conditioning and fit | **INCONCLUSIVE** |
| Historical label error inflated the WL60 R/G replay gap | WL61 matched clean reference control | Does not test primary pt0/test population | Noise versus systematic historical label bias | **SUPPORTED**, secondary population only |
| Training-label limitations contribute to primary failure | Weak old contrast precision; shared supervision | No controlled clean-label training intervention; WL61 is an evaluation audit | Actual fit, geometry/input coverage, physical B | **WEAKLY INDICATED**, contribution not isolated |
| Geometry conditioning is the demonstrated primary cause | Primary errors depend on deformation | Severe secondary R/G gap mostly disappears with qualified targets | Different physical signal sizes and unknown kernel sufficiency | **INCONCLUSIVE** |
| Optimization / objective mismatch explains primary residual | Validation does not test the failing differences; residual remains in relaxed output span | Static fitting improves/plateaus; no optimization intervention | Capacity/constraints and noisy targets | **INCONCLUSIVE** |
| Physical B approximation explains directional failure | B contrast is large and shared | No independent component error measured; absolute positivity floor tiny | Coupled remaining-budget split, cancellation, learned scattered error | **INCONCLUSIVE** |

The current evidence supports a real, repeatable **primary total-K approximation failure** for the frozen recipe. It does not support selecting label cleanup, longer training, a new basis, a geometry-encoder replacement or a B change as the corrective mechanism. No general architecture-class rejection follows.

## Unresolved Question — smallest missing distinguishing control

**Conclusion B: insufficient evidence to select one bounded corrective hypothesis.** The earlier proposed two-training-field audit has already been executed as WL61; do not propose it as unperformed or rerun it automatically.

The most direct remaining small component question is:

> Does the worst primary directional contrast come mainly from the geometry-only zero-collision approximation B, or from the learned collided response after accounting for B and their cancellation?

Existing evidence gap: primary reference packets store total first-exit actions and total moments, not path-history-tagged components. Scalar collision counters cannot recover the spatial/angular zero-collision action or its covariance with the total.

**Minimum proposed data, not executed:** original **pt0 / test fields 62 and 63 only**, two independent 64M replicas each (two 32M halves per replica), adding **zero-collision versus collided output tags** to the same physical estimator. This is 256M photons/eight halves; no rest, other geometry, angular sweep or training population is needed for this directional question. Reuse the original field bytes, physical conventions and frozen checkpoints. Preserve all original evidence; new tagged outputs must have separate provenance.

The future instrumentation must not change paths, RNG calls, material semantics or weighting. Record total/B/scattered first and second moments plus the needed cross moments or paired accumulators, because components from the same photon are correlated. First require tagged B+scattered to reproduce total transport and its uncertainty; a failed partition check invalidates attribution.

Planning cost: roughly **3–4 minutes of reference execution at the conservative 2M chunk policy**, extrapolated from WL61's 24.4–29.3 s per 32M half, plus setup/I/O and focused validation. The new primary fields may have different path lengths; this is not a timing guarantee. A faster chunk policy would be a separately recorded execution choice, not a scientific advantage. No job or source modification for this control has been started.

For the same directional contrast define `E_total = E_B + E_scattered` and report both norms and `2<E_B,E_scattered>` using a common normalization. Outcomes:

- Small independently measured B error compared with the persistent total error localizes the main defect to the learned collided path/interface on this pair, without separating capacity from fitting.
- Large B error aligned with total residual supports a B-side contribution. It still does not establish that optimizing B alone restores all quality.
- Large opposing component errors expose cancellation; do not attribute by norm ratios alone.
- Insufficient component precision or failed total closure leaves the question INCONCLUSIVE, with no automatic photon increase.

This control would distinguish two still-plausible component explanations; it would **not** resolve every representation/optimization question or the whole primary blue-delta population. Any constrained fitting intervention or broader clean-label campaign would require another explicit decision. No such intervention is designed as an adopted correction or run here.

## Architecture Verdict

No change to learned K, K/C ownership, Rule/State execution contracts, parameters, thresholds, historical baselines or scientific verdicts. WL57/58/59 negatives, WL60 bounds and WL61 partial qualification all remain intact; WL56's coupled physical-GI floor is unresolved. The abstract reuse hypothesis remains INCONCLUSIVE. Living architecture/research documents remain unchanged because no major new accepted design decision is justified.

## Reproducibility

- Committed analysis source: `f0a0229`, before the new 54-row arithmetic/population check. Five focused tests pass locally with the standard library; no Torch, GPU, model inference or network access is used.
- Inputs and exact SHA256 values are recorded in [reconciliation.json](../../results/fnto_attr62/reconciliation.json): original WL59 protocol/scores/reference sensitivity, WL60 bounds/population/history evidence, WL61 qualified analysis and identity/noise review. The script verifies the population byte identities instead of assuming equal field indices mean equal fields.
- Original checkpoints remain P_5901 SHA `59685b214bdf1f81079dcd56fe7a61173b90640b3c5c9fb8b3f14094729aaad6` and P_5902 SHA `c92e4fda4bbe9cf5569d55c45a2881d95eaa58df3896748088d060b55f1aab3f`; 157,024 parameters each, unchanged final 6000-step fits. Full historical source/reference versions remain in [WL59 provenance](../../results/fnto_persist59/final/provenance.json), [WL60 provenance](../../results/fnto_attr60/provenance.json), and [WL61 provenance](../../results/fnto_delta61/provenance.json).
- All per-pair/channel and signed-channel sensitivity rows are in [review CSV](../../results/evaluation/62/all_primary_noise_sensitivity.csv), including non-failing cells. [Batch provenance](../../results/fnto_attr62/provenance.json) binds code, inputs and derived artifacts. Existing detailed evidence is linked rather than regenerated.

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s experiments/fnto_attr62 -p 'test_*.py' -v
PYTHONDONTWRITEBYTECODE=1 python3 experiments/fnto_attr62/reconcile.py \
  --out <new-output-directory> --review <new-review-directory>
```

**Bounded completion:** primary failure remains measured and unexplained at the component/learning-cause level; secondary label contamination is real but does not transfer populations. Conclusion B, with an explicitly unexecuted minimum component control. No successor architecture or corrective training is authorised or started.
