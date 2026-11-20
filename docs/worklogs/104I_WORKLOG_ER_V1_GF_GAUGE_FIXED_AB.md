# Worklog 104I — ER-v1-GF: gauge-fixed parameterization and controlled A/B (2026-11-20 KST)

## Authorization, classification and freeze

The owner's brief authorizes one bounded architecture experiment. It develops a minimally revised certified ER parameterization, ER-v1-GF, that removes the six exact gauges found in WL104H, and compares it with the original under matched fitting. ER-v1-GF is a candidate only. Nothing is adopted, and the canonical ER is unchanged.

- **Repository and branch.** Branch `ecro_lr3_adapter` at `9b1c9e5`, equal to the `kn99e39/ReST-GI` remote. The untracked `experiments/ecro_fabric105/review105a.py` belongs to another session and was not touched.
- **Host.** Local RTX 5080, as in WL104E–H. This placement was not re-confirmed for this batch.
- **Design before implementation.** [ER-v1-GF design and limitations](../plans/ER_V1_GF_GAUGE_FIXED_PARAMETERIZATION.md).
- **Freeze.** Commit `7f710a9` froze the candidate, [protocol](../../experiments/ecro_wl84_104i/protocol.json), stage code and decision rules before evidence.
- **Pre-freeze development change.** On 20-step smoke states, the native-audit requalification re-tested both proposals even when only one had failed. It now re-tests only the failing proposal or proposals, as the protocol intends. The change is recorded in the protocol and was made before freeze.
- **Post-freeze execution fixes, commit `6cfbbc4`.** Neither changes a hypothesis, threshold, data set or comparison. The failed attempts and their logs are preserved as `*_attempt1`.
  1. The frozen WL104F `operator_case` raises on a common-support failure. It is now recorded as a failed probe, requalified per protocol, and supplemented by a descriptive support census.
  2. The post-fit identifiability diagnostic iterated over the stacked start/terminal arrays. It now iterates over the per-start states.
- **Post-hoc diagnostics** are labelled where they appear: `posthoc104i_rank.py` and the support census.

## Implementation Fact

**Mathematical facts** (derivation in the design document):

- **Original parameterization (64).** Per channel, softmax over 9 slots: Lambert, 3 outer, 4 inner and an absorption slot. 5 sigmoid knots; the lobe shapes; η.
- **Exact gauges.** One softmax shift per channel. One attenuation-scale gauge per channel: s → κ s and λ_{L,I} → λ_{L,I} / κ², with absorption compensating.
- **ER-v1-GF (58).** The absorption slot is removed. The outer amplitudes use additive log-ratio coordinates against the remaining block budget B = 1 − Σ λ_outer. The Lambert/inner block is pinned to B and split by a 5-way additive log-ratio. The knots carry the absorbed scale. Lobe shapes and η are unchanged.
- **Equivalence.** The map original → candidate (a = log λ_k / B, b = log λ_{4+m} / λ_0, s' = s √(T / B)) is gauge-invariant and preserves every composed term exactly. The inverse section (κ ∈ (max s', 1)) returns an interior original state. The two parameterizations therefore reach exactly the same set of physical responses.
- **Compiled-state boundary.** The canonical compiled state has Σλ = 1, the closure of the original raw-reachable set (Σλ < 1). The runtime, the 69-float layout and the certificate are unchanged.
- **Native proposal.** The native proposal of an equivalent state differs, because the compiled sampling probabilities are computed from λ.
- **Smoothness.** No max, tie or non-smooth point exists in the forward map.

**Verified implementation facts.** The candidate is isolated in [`experiments/ecro_wl84_104i/gf104i.py`](../../experiments/ecro_wl84_104i/gf104i.py) and reuses the unchanged closure82 evaluator, closure83_dr runtime and sampler. No production file changed.

**Training data and objective.** The WL83 original 64 × 64 fit lanes per material, hash-checked against the WL104D provenance, under the WL82 el1 + ealb objective. This is the original direct-fit recipe that produced R0.

## Measurement

### Q1 — response-family equivalence (Stage B)

There are 144 states:
- historical: R0, A, C and ER_L × 9 materials;
- training starts: generic + 3 seeds × 9;
- synthetic: 48 random states and 12 boundary states (Lambert/inner block → 0, outer → all mass, absorption dominant, knots ±12, η ±12, F0 ±12, roughness ±12, one dominant inner or outer lobe).

They were evaluated on 4,000 test directions, and the historical states also on all reference lanes ([equivalence.json](../../evaluation/104I/equivalence.json)):

| Check | Worst | Limit |
|---|---:|---:|
| Float64 response, original vs candidate | 1.1e-15 of peak | 1e-11 |
| Float64 composed terms (O_k, S·L, S·I_k) | 6.3e-16 | 1e-11 |
| Inverse section response / raw round trip | 7.1e-16 / 9.4e-11 | 1e-11 / 1e-8 |
| Compiled Dr.Jit runtime (float32) | 5.5e-7 | 2e-6 |
| Angular WL82 metrics on the reference (36 historical states) | within 2e-6 | 2e-6 |
| Reciprocity / FD / JVP | 1.7e-14 / 1.2e-7 / 9.2e-15 | 1e-10 / 1e-6 / 1e-10 |
| Validity (Σλ ≤ 1 + 1e-6, knots in [0,1], η, proposal) | all 144 valid | — |
| Gauge-orbit invariance: infinitesimal (exact JVP) / finite moves | 6.6e-16 / 6.9e-11 | 1e-9 |

### Q2 — exact gauge removal (Stage C), with post-hoc reading

Frozen-rule measurement over the R0 / A / C states of 9 materials ([ident.json](../../evaluation/104I/ident.json), [spectra](../../evaluation/104I/jacobian_spectra.png)):
- **Candidate.** 0 exact nulls (σ / σ_max < 1e-15) in all 27 states.
- **Original.** 6 exact nulls in 26 of 27 states. At R0/iid_159 one of the six measured 1.04e-15, so the count was 5.
- **Historical gauge vectors in the original Jacobian.** Their response is ≤ 2.2e-16 relative in every state.
- **Tangent spaces.** The frozen principal-cosine test with a 1e-12 rank cut gave a minimum cosine of 0.99968, below the required 1 − 1e-9. The worst states are the hard materials.

**Post hoc, after the frozen decision** ([posthoc_rank.json](../../evaluation/104I/posthoc_rank.json)):

- **Where the original's structural nulls sit.** In this unrotated computation they lie at the float64 floor: 3 at about 1e-20 (attenuation scale) and 3 at 1.5e-16 to 1.0e-15 (softmax shift). The 1e-15 cutoff, taken from WL104H's rotated computation, sits on that floor.
- **What the candidate's smallest directions are.** They reach 3.5e-15 (hard_53 C). They are dominated by Fresnel coordinates of saturated or near-empty lobes, and the original has the same directions.
- **Exact tangent identity.** The chain-rule identities J_original = J_candidate · ∂y/∂z and J_candidate = J_original(section) · ∂z/∂y hold to ≤ 3.7e-16 relative in all 27 states. The tangent spaces are therefore identical. The truncated principal-cosine test is coordinate-dependent near the cutoff, which is why it failed.
- **Fitted states (descriptive).** At some fitted hard_53 states, 1 candidate direction falls below 1e-15, also Fresnel-saturation. The original shows 5–7.

### Remaining approximate redundancy (candidate, R0 states)

- **Group couplings.** The largest principal cosine is ≥ 0.99 in 8–9 of 9 materials for split–Fresnel, split–roughness, split–attenuation, outer amplitude–Fresnel and Fresnel–attenuation.
- **Weak directions.** Counts with σ / σ_max < 1e-3 are 23–41 per state, the same as the original's non-structural counts.
- **Near-duplicate lobes.** The WL104H near-duplicate lobes are physical and are unchanged.

### Q3 — fidelity after matched retraining (Stages D, E)

**Fits.**
- The A1 (original) and B1 (candidate) starts are the same 4 physical starts per material: generic plus 3 seeds, generic + 0.5 N(0,1) in original raw space mapped exactly. The start objective gap is 8.6e-8.
- Both arms used Adam, lr 0.03 → 3e-4 cosine, 12,000 steps, float32, deterministic (two-step replay gap 0).
- A1 from the generic start reproduces historical R0's training objective (difference 0 to the printed precision) and its angular el1 (≤ 2e-5). This is the reproduction control.
- [Histories](../../evaluation/104I/training_histories.png), [fit.json](../../evaluation/104I/fit.json).

**Angular results.** Primary seeds; B1 vs A1 relative el1 change with a paired bootstrap 95% CI ([angular.json](../../evaluation/104I/angular.json), [bands](../../evaluation/104I/angular_el1_bands.png), [residual maps](../../evaluation/104I/angular_residual_maps.png)).

| Cohort, band | A0 el1 | A1 | B1 | (B1 − A1) / A1 |
|---|---:|---:|---:|---:|
| Development, all probes | .0877 | .0876 | .0875 | −0.10% [−0.28, +0.06] |
| Held out, all probes | .0774 | .0761 | .0751 | −1.39% [−3.34, −0.08] |
| Development, 5–20° / 55–65° / 80–86° | — | — | — | +0.65% [−1.7, +3.3] / −0.47% [−1.9, +0.5] / −0.23% [−0.8, +0.3] |
| Held out, 5–20° / 55–65° / 80–86° | — | — | — | −1.58% [−4.3, +0.6] / −3.30% [−7.6, −0.2] / +1.06% [−0.3, +3.5] |

- No band has a lower bound above +5%, so no LOCAL_REGRESSION flag was raised.
- The mirror core, shoulder and tail signed-energy errors are within ±1.6% (relative) between arms.
- Reference half-split el1 noise is 3.6e-4.

**Scenes, fixed R0 proposal (primary).** Objects stable RMS, seed-mean (B1 − A1) / A1, replicate-jackknife 95% CI ([scene figure](../../evaluation/104I/scene_relative_rms.png), [error maps](../../evaluation/104I/scene_error_maps.png)):

| Scene | E1 | E2 (fixed) | TOTAL (fixed, R0 HIGH) | Per-seed E1 |
|---|---:|---:|---:|---|
| iid_1 (dev) | +1.02% [+1.01, +1.02] | −7.7% [−9.5, −5.9] | −0.15% [−0.28, −0.01] | +0.03 / +3.5 / −0.44% |
| hard_1 (dev) | +0.39% [+0.32, +0.46] | −0.47% [−0.50, −0.44] | +0.38% [+0.31, +0.45] | +1.1 / +0.17 / −0.07% |
| iid_159 (held out) | −2.1% [−2.13, −2.11] | −6.4% [−7.1, −5.8] | −1.8% [−1.78, −1.74] | −0.1 / −4.5 / −1.6% |

The fixed-proposal pass reproduced the archived R0 E1/E2 within 2.0e-7 relative L1. The lane F/p identity was ≤ 1.1e-7 of peak, and the R0 Lambert proposal probability is > 0, so the common support holds. Baseline scene precision passed.

### Q4 — practical benefit

| Criterion | Measured (B1 / A1) | Predeclared benefit |
|---|---|---|
| Across-seed response spread | 0.18 development, 0.92 held out | ≤ 0.8 in both cohorts |
| Final training objective | median 0.9999; 5 of 9 materials lower (ratio < 1); block_186 0.92 | ≤ 0.99 with ≥ 6 of 9 materials lower |
| Fit wall time (uncontended) | 108.3 / 104.8 s = 1.03 | ≤ 0.9 |
| Peak memory | 193 MiB each | — |
| Steps to within 1% of the final objective | median 4,000–6,000 both | — |

No criterion is met and no regression criterion is triggered. The runtime is the same kernel and the same 69 floats, so runtime and memory are identical by construction.

### Native proposals (secondary)

**Native-scene contrasts** (unqualified, see the audit below):

| Scene | TOTAL | E2 | Per-pixel E2 variance B1 / A1 |
|---|---:|---:|---:|
| iid_1 | −0.16% | −7.6% | 1.000 |
| hard_1 | +0.38% | −0.38% | 1.001 |
| iid_159 | −1.75% | −4.6% | 0.96 |

**Audit: FAIL** ([native_audit.json](../../evaluation/104I/native_audit.json)). 5 of 108 probes failed directly:
- **3 common-support failures** (seeds 104911 and 104913, iid_1). The runtime sampler returned an accepted draw with wo.z = 0. Requalification failed where re-tested.
- **2 known-integral failures of A1** (original parameterization; seed 104913, iid_159, 50° and 86°). The requalification bounds were 0.0072 and 0.0061 against the 0.006 limit.

The descriptive support census, added after the first execution, counts 0–3 such invalid accepted draws per state in about 12M draws per material. It finds them in the **historical R0 (A0)** too: 3 on iid_1. The support defect is pre-existing in the runtime sampler and not specific to the candidate. The WL104F historical exception (one known-integral failure, requalified by WL104G as a fluctuation) remains recorded.

## Observation

- The two parameterizations are the same response family numerically, from float64 down to the float32 runtime, including the boundary states.
- With identical physical starts, the trajectories coincide initially and then separate on some seeds and materials. Examples are block_186 seed 104912 and iid_159 (training-history figure). The final objectives agree to a median ratio of 0.9999.
- The candidate's scene differences are dominated by single seeds. For example, iid_1 E2 is −19% on seed 104912 and −0.0% / −1.1% on the others. The replicate intervals are narrow because they condition on the realized seeds.

## Interpretation

**Mathematical facts.** The six exact gauges are structural redundancies of the original coordinates. A smooth 58-parameter chart without them exists and reaches exactly the original response family. Its compiled representative lies at Σλ = 1, which is not reachable by the original raw coordinates; the response is identical to interior states.

**Verified implementation facts.**
- Equivalence holds at the declared precision for 144 states.
- The candidate has no exact null direction at any tested state.
- The original's gauge directions are exact to machine precision.
- The local tangent spaces are identical.

**Measured outcomes.**
- Under the frozen rules, fidelity after matched retraining is preserved. Every primary upper bound is ≤ +1.02%, against the +5% margin.
- No practical benefit was demonstrated in stability, objective or cost.
- The two frozen identifiability rules failed, and the native audit failed.

**Causal interpretation.**
- The frozen FAIL for exact gauge removal and structural identifiability comes from the numerical implementation of the rules, not from a gauge remaining in the candidate:
  - an original-side structural null at 1.04e-15, on the float64 floor, against a 1e-15 cutoff;
  - a truncated principal-cosine test that is not coordinate-invariant.
- Removing the exact gauges changes neither the tangent space nor the approximate near-degeneracies. The fitting differences between A1 and B1 are optimizer-path effects of different coordinates (Adam is coordinate dependent), seen as seed-specific divergence. They are not a capacity or response-family effect.
- The native-audit failures hit the original A1 as well as the candidate, and the support defect is present in historical R0. The native comparison is unqualified for both arms.

**Remaining speculation (not tested).** The development-cohort spread reduction (0.18) and the held-out improvements (−1.4% el1, −1.8% scene TOTAL) may reflect a systematic optimizer benefit of the gauge-free chart, or chance across three seeds. Three seeds and nine materials cannot separate the two.

## Unresolved Question

- **Optimizer-path effects.** Whether the gauge-free chart systematically changes them would need more seeds or an optimizer that is invariant to the coordinate change. Neither was run.
- **Support defect mechanism.** The sampler's invalid accepted draws (a valid flag with wo.z = 0, rate about 2e-7, observed on iid_1) were not isolated. The working hypothesis, an inconsistent recomputation near total internal reflection across kernels, is unverified.
- **Original-arm known-integral failures.** Whether they are statistical or a PDF inconsistency of that fitted state is unresolved.
- **Approximate degeneracies.** Near-duplicate lobes, amplitude–Fresnel and split–attenuation overlap remain in both parameterizations.
- **Coverage.** One coated-conductor family, 9 materials, 3 scenes.

## Architecture Verdict

**Decision table** (frozen rules):

| Item | Label |
|---|---|
| Response-family equivalence | **PASS** |
| Exact gauge removal | **FAIL** (frozen numerical rule; original-side count 5 at R0/iid_159; post hoc: the candidate has no exact null and is gauge-invariant by construction and test) |
| Training fidelity preservation | **PASS** |
| Structural identifiability improvement | **FAIL** (frozen truncated-tangent test; post hoc: tangent spaces identical by chain rule, ≤ 3.7e-16) |
| Training stability or practical benefit | **FAIL** (no demonstrated benefit) |
| Native proposal qualification | **FAIL** (affects both arms; pre-existing support defect in R0) |
| ER-v1-GF candidate viability | **NOT SUPPORTED** under the frozen rule |

The NOT SUPPORTED verdict follows mechanically from the exact-gauge-removal row. That row failed because of an original-side numerical rank count, while all direct evidence shows the candidate is gauge-free and fidelity-preserving. Whether the frozen rule's outcome or the post-hoc reading should govern is a human decision; no label was changed. ER-v1-GF is not adopted, the canonical ER is unchanged, and the ER family is not rejected. [Decision](../../evaluation/104I/decision.json).

## Reproducibility

- **Commits.** Freeze `7f710a9`. Execution fixes `6cfbbc4`. The report commit adds `posthoc104i_rank.py`, the packager update, `evaluation/104I/`, this worklog and the living-document note.
- **Environment.** Local RTX 5080 16,303 MiB, `external/8dna26/.venv-cu128` (Python 3.11.9, Torch 2.8.0+cu128, Mitsuba 3.5.1, Dr.Jit 0.4.6).
- **Timing.** Chain 16:59–17:09 KST. Equivalence 30 s, identifiability 14 s. Fits A1 104.8 s / B1 108.3 s, run uncontended. Angular 3 s. Native audit 24 s. Scenes: fixed pass about 4 s per development replicate and 9 s per held-out replicate; native renders 10–18 s per state.
- **Seeds.**
  - Starts 104911–104913 plus the generic start; bootstrap 104931; synthetic states 104921; test directions 104801.
  - Scenes 1044900 / 1046900 / 1048900 + r (12 replicates; 1,024 spp development, 4,096 spp held out, archived 512-spp launches in 128-spp sub-chunks).
  - Audit 1049001 + 100k + 10j + 2t + p, requalification 1049501 +, census 1049901.
- **Inputs.** Authenticated through the WL104H / WL104G manifests and the WL104D provenance (train lanes). Hashes are in [manifest](../../evaluation/104I/manifest.json).
- **Raw outputs.** `results/ecro_wl84_104i/primary/`: equivalence, ident, ident_fit, fit, angular, native, scenes/<scene>/{fixed_r###, native_*_r###, analysis, maps}, logs, and the preserved `*_attempt1` directories. Post-hoc outputs are in `results/ecro_wl84_104i/posthoc/`.

```bash
cd experiments/ecro_wl84_104i
export PYTHONUTF8=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
bash ../../results/ecro_wl84_104i/primary/chain.sh          # all stages (writes results/ecro_wl84_104i/primary)
PY=../../external/8dna26/.venv-cu128/Scripts/python.exe
$PY -B posthoc104i_rank.py ../../results/ecro_wl84_104i/posthoc/rank.json        # post hoc
$PY -B package104i.py ../../results/ecro_wl84_104i/primary ../../evaluation/104I
```
