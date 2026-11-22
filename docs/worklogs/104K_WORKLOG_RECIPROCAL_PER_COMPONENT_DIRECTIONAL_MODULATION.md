# Worklog 104K — Reciprocal per-component directional modulation (ER-v1-SDM candidate) (2026-11-22 KST)

## Authorization, classification and freeze

The owner's brief authorizes one bounded architecture hypothesis experiment. It develops a minimal ER variant with reciprocal, per-component directional modulation and tests whether this variant recovers part of WL104J's incident-conditioned angular headroom while staying physically valid. The owner named the batch WL104K.

ER-v1-SDM is a candidate only. It is not adopted, and the canonical ER and ER-v1-GF are unchanged.

- **Repository and branch.** `kn99e39/ReST-GI`, branch `ecro_lr3_adapter`, started at `7d471be`. No new branch or worktree. The untracked `evaluation/104J.zip` belongs to the owner and was not touched.
- **Host.** Local RTX 5080. The GPU was idle before launch, and no other project's job ran during the batch.
- **Freeze.** The protocol and all stage code were committed at `711c6a3` before any evidence: [protocol](../../experiments/ecro_wl84_104k/protocol.json), [code](../../experiments/ecro_wl84_104k/).
- **Pre-freeze development**, disclosed in the protocol:
  - **Contract tests** were developed and run. One tolerance was decided from development numerics: Dr.Jit parity. On random directions down to cos 1e-3, the *unmodified* production runtime itself differs from torch float64 by up to 5.2e-6 (float32), while the experiment-local evaluator at identity gates matches that runtime to 6e-8. The rule therefore became "≤ max(2e-6, 1.5 × the base runtime's parity)".
  - **Timing probe.** A 300-step full-scale probe.
  - **Dev chain.** A 150-step end-to-end chain whose non-converged numbers were printed.
  - **Scene smoke.** A scene-replay smoke run.
  - **Changes after the dev chain:**
    - the transfer label now also requires qualification (stricter);
    - a chunk-size flag was added for the scene replay;
    - figure layout was adjusted.

  No threshold was loosened.

## Implementation Fact

### Representation, verified in code

Every component is computed with the unchanged closure82 / closure83 functions (`lobe_f`, `refract_up`, `s_interp`); no production file changed. The candidate response is:

F_SDM(wi, wo) = Σ_j q_j(μi) q_j(μo) C_j(wi, wo),  j ∈ {Outer 1–3, Lambert, Inner 1–4}

- **Components.** C_j are the certified ER components, cosine-weighted. Lambert and inner include s(μi) s(μo); inner lobes keep the refracted coordinates and 1/η².
- **Gates.** q_j is piecewise linear through the 5 attenuation knot positions (μ = 0, .15, .4, .7, 1), per RGB channel, with values in [0, 1].
- **Gate parameterization.** The gates are parameterized by their knot values and projected to [0, 1] after every Adam step. Identity (all knots = 1) is a finite state that reproduces A.
- **Gradient at identity.** It is generically non-zero (∂F/∂q = C_j (φ(μi) q(μo) + q(μi) φ(μo))), so identity is not a zero-gradient trap. This was verified by finite differences and by a 200-step trainability test.

### Mathematical contract

- **Reciprocity.** Each C_j / cos θo is symmetric in (wi, wo):
  - D(h) is symmetric;
  - Fresnel is evaluated at wi·h = wo·h;
  - the horizon-safe denominator is symmetric;
  - the refraction map is applied to both directions;
  - s(μi) s(μo) is symmetric.

  The gate factor q_j(μi) q_j(μo) is symmetric, so f_SDM is reciprocal.
- **Passivity.**
  - C_j ≥ 0 and 0 ≤ q_j ≤ 1 give 0 ≤ F_SDM ≤ F_base pointwise.
  - Hence every channel's directional albedo is bounded by the base ER's, which the WL82 certificate bounds by Σλ ≤ 1 for every incident direction. ER-v1-GF compiled states have Σλ = 1, inside the certificate.
  - The argument uses only pointwise nonnegativity and the base certificate, not symmetry.
- **Baseline inclusion.** q ≡ 1 gives A exactly. T is P with all eight gates equal.

### Arms

| Arm | Gates | Trainable / material | Runtime floats / material |
|---|---|---:|---:|
| A shared ER-v1-GF | none (closure82 path, as in WL104J) | 58 | 69 |
| T shared symmetric gate | one RGB gate on all 8 components | 73 | 84 |
| P per-component gates (ER-v1-SDM) | one RGB gate per component | 178 | 189 |

**Degrees of freedom.** Nominal counts overstate the free degrees of freedom. The numerical Jacobian rank (σ/σmax > 1e-10, 1,280 reference lanes) at the fitted best states is:
- A: 50–57 of 58;
- T: 62–69 of 73;
- P: 135–150 of 178.

With σ/σmax > 1e-3, the counts are 10–29, 24–38 and 40–89. A constant gate acts like an amplitude, the Lambert and inner gates overlap the shared attenuation, and gates on near-empty components are undetermined.

### Fitting

| Item | Setting |
|---|---|
| Data | WL104J fit set (authenticated against `evaluation/104J/manifest.json`) |
| Objective | WL82 el1 + ealb |
| Optimizer | WL82 Adam, 12,000 steps, float32 |
| Starts | the 4 matched physical starts (generic + seeds 104911–13) |
| Gate initialization | T and P start from the same base state with identity gates (the identical physical response) |
| Training | base and gates optimized jointly; no arm is warm-started from another arm's fit (WL104J lesson) |
| Optimizer asymmetry (disclosed) | gate knots are projected, base coordinates are not |
| Warm-start diagnostic | not run |

**Reproduction of WL104J's A.**
- Same data and objective. The pooled held-out el1 is 0.07673 here against 0.0767 in WL104J, and the median final objective matches to 1e-5 relative.
- Terminal raw states are not bitwise identical (max raw difference 0.33). The gradient reduction order differs between the WL104J index-gather path and the expand path used here. The divergence lies along weakly determined directions.

## Measurement

### Contracts and controls

Gate stage passed; [tests.json](../../evaluation/104K/tests.json), [tests_fitted.json](../../evaluation/104K/tests_fitted.json).

**Baseline identity.**
- float64 identity-gate T / P vs A: ≤ 1e-12 on historical (R0, WL104D A / C, ER_L), random and extreme states.
- float32 training path: 1.3e-7.

**Reciprocity.**
- |f(wi,wo) − f(wo,wi)| / max f, interior pairs: ≤ 3e-15 for random gates including 0 / 1 knots.
- With grazing pairs included: ≤ 1.7e-14 for random gates.
- Every fitted T and P state: ≤ 6e-15, grazing included.
- cos_i F(wi,wo) = cos_o F(wo,wi): same precision.

**Passivity.**
- F_SDM − F_base ≤ 2.6e-16 of peak (float64 rounding) on all test lanes; min F ≥ 0.
- MC directional albedo at μi ∈ {0.01 … 1}: ≤ 1 within 4 SE for extreme states and all fitted states.
- Σλ ≤ 1.

**Gradients and trainability.**
- Autograd vs finite differences ≤ 3e-7 relative, at identity and interior gates.
- Every arm moved a gate of every material within 200 steps.

**Dr.Jit parity.**
- SDM evaluator vs torch: 3.7e-6. The base runtime vs torch on the same lanes is 5.2e-6.
- Identity SDM vs runtime: 6e-8.

**Synthetic controls** (joint fit from truth + 0.1 N, identity gates; held-out el1 against the noise-free truth):

| Control | A | T | P | Rule | Result |
|---|---:|---:|---:|---|---|
| S0 null (no gates) | .00021 | .00023 | .00026 | \|T−A\|, \|P−A\| ≤ 1e-3 | PASS |
| ST known shared gate | .0201 | .00040 | .00049 | T and P ≤ 50 % of A | PASS |
| SP known per-component gates | .0159 | .0089 | .00060 | P ≤ 50 % of T | PASS |

The joint fitting procedure recovers both mechanisms to the noise level, and a shared gate cannot imitate the per-component target.

### Held-out angular contrasts

Independent WL104D evaluation reference. Median of 4 starts, with a 95 % cluster bootstrap over materials. [eval.json](../../evaluation/104K/eval.json).

| Contrast | Development | Held out | Half a / b (dev; held out) | Starts negative |
|---|---|---|---|---|
| **T − A** shared gate | −14.7 % [−39.7, −0.2] | −23.0 % [−50.5, −6.2] | −14.6 / −14.6; −22.3 / −22.3 | 4/4, 4/4 |
| **P − T** per-component beyond shared | **−44.0 %** [−75.9, −25.6] | **−47.1 %** [−57.9, −42.1] | −42.7 / −43.0; −44.6 / −44.6 | 4/4, 4/4 |
| **P − A** total | **−52.3 %** [−85.5, −25.6] | **−59.2 %** [−77.7, −47.0] | −51.1 / −51.3; −56.9 / −56.9 | 4/4, 4/4 |

- **Pooled el1** (mean per probe, median start): A 0.0767, T 0.0615, P 0.0333. By cohort: development .0866 / .0739 / .0412; held out .0718 / .0554 / .0293.
- **Normalized shape (eshape), P − A:** −52.4 % / −59.5 %.
- **Directional albedo (ealb), P − A:** −52.4 % / −75.1 %.
- **Training vs held-out reduction:** P − A 53.4 / 60.6 % training vs 52.3 / 59.2 % held out. No overfitting.
- **Transfer, P − A:**
  - nearby polars: −57.2 % [−88.8, −28.0] / −62.7 % [−80.0, −50.9];
  - new azimuths: −52.9 % / −60.7 %;
  - every interval excludes zero.
- **Concealed regression.** None. Every material improves (−26 % to −85 %), and every incident-polar group improves in both cohorts (−17 % to −76 %).
- **Resolution.** The reference half-split noise is 0.42 % / 0.50 % of A's el1.

**Per material** (all probes, median start):

| Material | Cohort | A | T | P | T − A | P − T | P − A |
|---|---|---:|---:|---:|---:|---:|---:|
| iid_1 | development | .0912 | .0550 | .0132 | −40 % | −76 % | −85 % |
| iid_0 | development | .0545 | .0525 | .0258 | −4 % | −51 % | −53 % |
| hard_1 | development | .1140 | .1141 | .0846 | 0 % | −26 % | −26 % |
| iid_191 | held out | .0487 | .0174 | .0085 | −64 % | −51 % | −83 % |
| iid_159 | held out | .0625 | .0535 | .0286 | −14 % | −47 % | −54 % |
| block_161 | held out | .0642 | .0398 | .0120 | −38 % | −70 % | −81 % |
| block_186 | held out | .0468 | .0131 | .0068 | −72 % | −48 % | −85 % |
| hard_76 | held out | .1108 | .1104 | .0614 | 0 % | −44 % | −45 % |
| hard_53 | held out | .0980 | .0980 | .0587 | 0 % | −40 % | −40 % |

### Regions and incidence

Region L1 divided by the total reference energy; pooled; best start.

| | Core | Shoulder | Tail | 5–20° | 35–50° | 55–65° | 70° | 80–86° |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A, development | .0095 | .0281 | .0571 | .062 | .058 | .095 | .112 | .153 |
| P, development | .0066 | .0157 | .0371 | .029 | .034 | .052 | .066 | .120 |
| A, held out | .0072 | .0208 | .0490 | .054 | .057 | .070 | .075 | .116 |
| P, held out | .0039 | .0073 | .0245 | .020 | .026 | .026 | .029 | .067 |

- **Signed region energy** (mean |signed|), A → P: core .082 → .034, shoulder .063 → .015, tail .037 → .012.
- **Grazing (80–86°)** improves least: −22 % development, −43 % held out.

### Gates

[Figure 5](../../evaluation/104K/fig05_learned_gate_curves_per_component.png).

- **T.** On hard_1, hard_76 and hard_53, T's shared gate stays at identity (T − A = 0 %). Over all materials, 55 % of T's knots are at 1.
- **P.** 40 % of knots are at 1 and 10 % at 0. The gates on the energetically dominant components follow smooth curves that differ from component to component, for example:
  - Lambert gated down toward grazing on block_161;
  - Outer 3 gated down toward normal incidence on iid_191.

  Gates on components with a small energy share are erratic and swing between 0 and 1.
- **Mean energy share of P's components:** Lambert 21 %, Inner 2 21 %, Outer 3 16 %, Outer 2 14 %, Inner 3 11 %, Inner 1 11 %, Outer 1 3 %, Inner 4 3 %.

### Convergence

| Rule | A | T | P | Required |
|---|---|---|---|---|
| (a) units with last-2,000-step change ≤ 0.5 % | 100 % | 100 % | **53 %** (max 2.95 %, iid_191 / block_186 / iid_0) | ≥ 90 % |
| (b) extended 24k-step change of the training contrast | T − A 0.16 pp | P − T 1.48 pp | P − A 1.25 pp | ≤ 3 pp |

- **Extended generic-start held-out el1** (descriptive): A .0766, T .0614, P .0326. At 12k the generic start gave .0770 / .0617 / .0337.
- **Outcome.** P was still improving at 12k, so rule (a) fails for P.

### Compute and storage

| | A | T | P |
|---|---:|---:|---:|
| Added storage per material | 0 B | 60 B | 480 B |
| Torch float32 eval, 2^22 lanes | 55 ms | 38 ms | 52 ms |
| Dr.Jit eval, 2^22 lanes | 2.6 ms (production runtime) | 3.8 ms | 7.9 ms |
| Fit, 4 starts × 9 materials × 12k steps | 487 s | 590 s | 828 s |
| Peak memory | 0.95 GB | 1.26 GB | 2.28 GB |

- The torch timings are not comparable between arms: A goes through closure82, T and P through the component path.
- The T / P Dr.Jit evaluator is experiment-local and unoptimized: it is component-split and re-traces the gates.
- The fit runs shared the GPU with the concurrent synthetic runs.

### Scene stage

**Not run.** The frozen scene condition requires the angular gain label PASS. That label is INCONCLUSIVE because optimization is unresolved.

## Observation

- **Effect size.** A reciprocal, passive per-component gate removes about half (development) to 60 % (held out) of A's held-out angular error.
  - About three quarters of that comes from per-component freedom beyond one shared gate: P − T −44 / −47 %.
  - The effect is consistent across all 9 materials, all 4 starts, both reference halves, held-out outgoing directions, nearby polars and new azimuths.
  - It appears in albedo, normalized shape, core, shoulder and tail.
- **Hard materials.** The shared gate does nothing on the three hard materials, while per-component gates still give −26 % to −45 % there.
- **Remaining error.** The largest remaining errors are at grazing incidence and on the hard materials (hard_1 keeps 74 % of its error).
- **Historical context only, not a comparator.** WL104J's non-reciprocal oracles B (amplitude-local) and C (full-local) reached pooled el1 0.0489 and 0.0415 on the same reference. P reaches 0.0333 while remaining reciprocal and passive. The fitting contracts differ: WL104J chained warm starts with frozen shared parameters, while P here is fitted jointly. The comparison is descriptive.

## Interpretation

- **Mathematical correctness: established.** The candidate is reciprocal, nonnegative and passive by a pointwise-envelope argument that holds for the implemented closure. It contains A exactly. Every contract and the representation tests pass, and the synthetic controls show that joint fitting can exploit both shared and per-component gates.
- **Observed approximation gain: large and reproducible at 12,000 steps.**
  - The measurement satisfies every element of the frozen practical-gain rule except optimization resolution: −10 % tier, intervals, halves, starts, no concealed regression, transfer.
  - The unresolved part is P still decreasing its training objective. The extended run suggests further gain, not a reversal: held-out .0337 → .0326 for the generic start.
  - Under the frozen rule the gain, per-component benefit and transfer labels are INCONCLUSIVE, not PASS.
- **Causal interpretation: limited.**
  - The gain shows that the shared-parameter ER family under-represents the angular response of these materials in a way that symmetric, per-component directional reweighting can partly correct.
  - It does not identify which physical mechanism WL104J's error came from. Gates on weak components are undetermined.
  - Distinct gate curves do not establish physically independent scattering paths.
  - P has about 2–3× A's numerically free directions, so the comparison is not capacity-matched.
- **Real-scene utility: unmeasured.** The scene stage was not run.
- **Production feasibility: partially measured.**
  - 480 B per material, and a 3× slower unoptimized Dr.Jit evaluation.
  - Sampling for P is not designed: the production sampler and pdf assume the ungated mixture. Importance sampling of the gated response is an open requirement.

## Unresolved Question

- **Optimization.** Whether the angular gain label would be PASS with a converged P. The rule failed only on P's last-2,000-step criterion, and no further budget was run, per the brief.
- **Scene utility.** Whether the angular gains survive scene transport. The fixed-proposal scene replay is implemented, its replay identity is verified in the smoke test, and it costs about 10–15 min. It was not run, by rule.
- **Sampling.** How a gated response should be sampled. The current sampler targets the ungated lobes, so native rendering with P would need a pdf / proposal design. This was not part of this batch.
- **Weak gates.** Gates on low-energy components are undetermined, and the knot count was fixed (no sweep). Whether regularization or fewer gates would keep the gain is untested, and a regularizer was excluded by the brief.
- **Grazing.** The remaining grazing and hard-material error is not addressed by this factorization.
- **Coverage.** One coated-conductor family, 9 materials, 10 incident polars.

## Architecture Verdict

| Item | Label |
|---|---|
| Baseline identity | **PASS** |
| Reciprocity | **PASS** |
| Passivity | **PASS** |
| Synthetic controls | **PASS** |
| Angular fidelity gain | **INCONCLUSIVE** (measured P − A −52 % / −59 %; optimization rule (a) failed for P) |
| Per-component benefit beyond shared gating | **INCONCLUSIVE** (measured P − T −44 % / −47 %; same reason) |
| Held-out directional transfer | **INCONCLUSIVE** (measured −53 % to −63 %, all intervals below 0; same reason) |
| Scene utility | **NOT RUN** (scene condition requires angular gain PASS) |
| Runtime feasibility | **MEASURED** |
| Candidate architecture viability | **INCONCLUSIVE** |

[Decision](../../evaluation/104K/decision.json).

The INCONCLUSIVE labels come from one frozen optimization criterion, not from a failed or reversed effect. ER-v1-SDM is not adopted, and the canonical ER and ER-v1-GF are unchanged. Adoption, modification, a converged re-fit, scene validation or rejection are owner decisions, and no further architecture experiment is started.

## Reproducibility

- **Commits.** Freeze `711c6a3`. The report commit adds `evaluation/104K/`, this worklog and the living-document note.
- **Environment.** Local RTX 5080 16,303 MiB, `external/8dna26/.venv-cu128`: Python 3.11.9, Torch 2.8.0+cu128, Mitsuba 3.5.1, Dr.Jit 0.4.6.
- **Timing.** Chain 14:46–15:43 KST: contracts 49 s, fits 55 min (primary, synthetic and extended concurrent), fitted contracts 10 s, evaluation 34 s.
- **Seeds.** Starts generic + 104911–104913; synthetic noise 1041130, synthetic start 1041131, bootstrap 1041140, tests 1041150. Data and reference seeds are those of WL104J and WL104D.
- **Inputs.** Authenticated through the WL104G manifest, the WL104D Amendment 1 provenance, materials83 hashes and `evaluation/104J/manifest.json`. Hashes are in the stage JSON files.
- **Raw outputs.** `results/ecro_wl84_104k/primary/`: tests.json, tests_fitted.json, fit_primary, fit_extended, synth_S0 / ST / SP, eval (eval.json, lanes.npz), decision.json, logs, status.txt.
- **Reviewer package.** [evaluation/104K/index.html](../../evaluation/104K/index.html): figures fig01–fig08, each tagged MEASURED or CONCEPTUAL with source data; the per-material × incidence table and CSV; `manifest.json` and `figures_manifest.json` (figure-data checks all pass).

```bash
cd experiments/ecro_wl84_104k
bash chain.sh ../../results/ecro_wl84_104k/primary 12000
PYTHONUTF8=1 ../../external/8dna26/.venv-cu128/Scripts/python.exe -B figures104k.py ../../results/ecro_wl84_104k/primary ../../evaluation/104K
```

## Post-hoc qualitative scene evaluation (2026-11-22, owner request; appended)

The owner asked for qualitative material, including renders, showing how the reported improvement appears on the existing scene data set. This section is **post hoc**. The frozen decision table above, including "Scene utility: NOT RUN", is unchanged, and nothing here is a frozen test.

- **What was run.** The frozen, previously unexecuted scene replay [`scenes104k.py`](../../experiments/ecro_wl84_104k/scenes104k.py), with code unchanged, launched by [`qual_chain.sh`](../../experiments/ecro_wl84_104k/qual_chain.sh) (commit `9fb0eb6`).
  - **Scenes and budget.** Scenes iid_1 and hard_1 (development materials) and iid_159 (held-out material and camera), 12 replicates each.
  - **Replay.** Common R0 proposal with F/p recomputed per arm, so every arm is evaluated at the same sampled directions.
  - **States.** A0 (R0) plus A / T / P × 4 starts.
  - **Total term.** "Total" adds the archived R0 higher-order term to every model.
  - **Replay identity.** ≤ 2.0e-7 relative L1; lane F/p ≤ 1.1e-7.
  - **Run time.** 16:40–16:56 KST, local RTX 5080, about 32–45 s per replicate.
- **Objects error of the shown renders** (start 1 = generic, 12-replicate mean, RMS / mean reference luminance):

  | Scene | First bounce A / T / P | First + second | Total |
  |---|---|---|---|
  | iid_1 | 22.6 / 7.7 / 1.4 % | 11.8 / 5.0 / 3.5 % | 11.1 / 5.0 / 3.4 % |
  | hard_1 | 21.8 / 21.5 / **24.2** % | 10.1 / 9.9 / **11.1** % | 8.2 / 8.1 / **9.1** % |
  | iid_159 (held out) | 10.6 / 8.4 / 5.6 % | 5.7 / 4.7 / 3.5 % | 5.5 / 4.5 / 3.4 % |

- **Descriptive contrast P − A** (objects stable RMS, the noise-debiased WL104E estimator; mean of 4 starts; replicate-jackknife 95 % CI):

  | Scene | First bounce | Second bounce | Total |
  |---|---|---|---|
  | iid_1 | −94 % | −100 % [−112, −88] | −97 % [−113, −81] |
  | hard_1 | **+3.6 % [+0.5, +6.7]** | −29 % | **+3.0 % [0.0, +6.1]** |
  | iid_159 | −32 % | −34 % | −33 % [−33.3, −32.5] |

  −100 % means P's error is below the replicate-noise estimate; the estimator clips at 0. T − A on the total is −69 % (iid_1), −1 % (hard_1) and −10 % (iid_159).
- **Observation.**
  - On iid_1 and the held-out iid_159, the angular gain carries into the rendered first and second bounce.
  - On hard_1, P lowers the second-bounce error but raises the first-bounce error, and the total is slightly worse than A. The zoomed highlight shows that P overshoots the centre of the mirror highlight, while A undershoots it.
  - So hard_1's angular el1 gain of −26 % does not translate into a better direct-highlight image.
  - This is consistent with grazing incidence and the peak being the least improved angular regions. It is descriptive and was not tested.
- **Package.** [evaluation/104K/qualitative/qualitative.html](../../evaluation/104K/qualitative/qualitative.html) contains:
  - click-to-switch reference / A / T / P renders and absolute-error images per scene and event (same exposure);
  - full-frame comparisons with error and error-reduction maps (one absolute luminance error scale for all maps, 0–0.84, 4th-root display);
  - zoomed highlight, grazing and lower-sphere regions;
  - the scene error summary.

  Script: [`qual104k.py`](../../experiments/ecro_wl84_104k/qual104k.py). Hashes are in `qualitative_manifest.json`. Raw outputs are in `results/ecro_wl84_104k/primary/posthoc_scenes/`.
