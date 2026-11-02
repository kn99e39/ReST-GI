# Worklog — W+UV Component Attribution of the WL71A Drift, WL71B (2026-11-03)

**Numbering.** This is worklog 71B under the user's letter-suffix rule of 2026-11-03.

**Authorisation.** User brief of 2026-11-03: one bounded, read-only mechanistic attribution of the WL71A source-generalization drift on the unchanged WL59 K and the existing WL71A checkpoints, with no training. The user also instructed: "이거도 서버에서 돌려. worklog 작성 및 push까지 한 뒤에 이 pc를 shutdown 명령어로 종료해줘" (run this on the server too; after writing the worklog and pushing, shut this PC down).

**Status of the analysis.** Development data only. The analysis is post hoc by nature: WL71A's results were known when it was designed.

**Files.**

| Item | Location |
|---|---|
| Protocol | `experiments/fnto_attrib71b/attrib_protocol.json`, committed before any output (0b0a81e) |
| Code | `experiments/fnto_attrib71b/attrib.py`, `chain_attrib.sh`, `figure71b.py` |
| Results | `results/fnto_attrib71b/` |
| Reviewer figure | `results/evaluation/71B/` |

## Implementation Fact

**Run.**

- LabServer63, torch 2.3.1, one server chain (`~/rest_attrib71b`): tests → Stage A → Stage C → label.
- 2026-11-02 16:18–16:37 UTC.
- The code snapshot is the committed repository.
- WL71A checkpoints were read from `~/rest_drift_smoke/out`. All 12 sha256 hashes match `results/fnto_drift71a/checkpoints.sha256`.
- WL68 references were used as in WL71A.

**One failed attempt, preserved in `attempt1/`.** A test compared Σ_j T with the mass at 1e-6. The float32 sums over 3,840 receivers differ by 3.6e-6, so the test failed. The check was changed to a float64 comparison with relative tolerance 1e-5 (commit after 0b0a81e) before any stage output existed. This repeats the identifiability smoke's lesson that tolerances must be set for storage precision.

**Algebra decomposed, unchanged.** For a source i:

```text
W_raw = softplus(edge MLP) · prior        (128 nearest edges)
U     = softmax_j(node head)              (global destination basis, rank 16 per mixture and channel)
V_raw = softplus(node head of i)
mass  = sigmoid(node head of i)
scale = mass / (ΣW_raw + ΣV_raw)
T     = T_W + T_UV
      = scatter(W_raw · scale) + U (V_raw · scale)
Σ_j T = mass
```

**Tests (`results/fnto_attrib71b/tests.json`): all passed.**

- **Checkpoint identity:** 12/12.
- **A/O:** identical across all checkpoints and equal to net_all_5902.
- **Decomposition closure:** T_W + T_UV against `family71.network_columns`, max 0.0 (bitwise).
- **Counterfactual closure:** EEEE and LLLL reproduce Early and Late exactly (0.0).
- **Passivity of every combination:**
  - T ≥ 0;
  - Σ_j T equals the mass to 3.6e-6 relative;
  - max column sum 0.99999;
  - max response β + (1−β) Σ_r A·mass = 0.989.
- **Source correspondence:** holds.
- **Evaluation identity:** the EEEE / LLLL scores reproduce WL71A's evaluation of R/N@2000 and R/N@20000 to ≤ 1.4e-7.

**Counterfactual design.**

- Early = R@2000, Late = R@20000; the N pair is a consistency control.
- Full 2^4 factorial over W (W_raw), U, V (V_raw) and M (mass). The scale is recomputed with the original algebra.
- The hybrids are diagnostic counterfactuals, not reachable networks.

## Measurement

### Stage A — components at rest, arm R, Early (2,000) → Late (20,000)

Source mass and W share:

| Source | Change in V_raw | Mass | W share of flux | Flux within 1 cm (total) | Flux within 1 cm (W part) |
|---|---|---|---|---|---|
| fit 0–3 | V small, T_W changes 5–8% | stable (.77–.80) | ≥ .97 | .91–.92 | |
| unseen 4 | small | | | .91 | |
| unseen 5 (1025) | .03 → .05 | **.78 → .62** | .98 → .95 | .92 → .83 | |
| unseen 6 (2707, corner) | **.03 → 23.6** | **.80 → .39** | **.97 → .46** | **.91 → .30** | .94 → .66 |
| unseen 7 | .06 → .95 | .78 → .76 | .95 → .91 | .90 → .86 | |

**Global U** (per geometry):

| Measure | 0 | 500 | 2,000 | 6,000 | 10,000 | 20,000 |
|---|---|---|---|---|---|---|
| Relative change from init | 0 | .79 | 3.1 | | | 7.9 |
| Column mass within 1 cm of the fit ports | .06 | | .30 | .43 | .55 | .56 |
| Mean column entropy (bits) | 11.7 | | | | | 6.6 |

**Where the UV flux lands.** The fraction of UV flux landing within 1 cm of a fit port, at both fit and unseen sources: .05–.07 at init, .28–.39 at 2,000, .37–.68 at 20,000.

**Consistency.** pt0 and cf3 give the same values. The N arm is the same, with slot-6 V_raw going .03 → 33.4.

### Stage C — 2^4 counterfactual (arm R)

**Values.** The four rows below are, in order: Early, Early with W from Late, Early with V from Late, Early with mass from Late. Group shares are defined in the protocol: degrade, recover, and their mean as a fraction of the drift.

| Metric | Early (EEEE) | Late (LLLL) | +W only (LEEE) | +V only (EELE) | +mass only (EEEL) |
|---|---|---|---|---|---|
| H1 coarse, pooled worst | .095 | .511 | .303 | .375 | .377 |
| H1 contrast | .327 | .990 | 1.176 | .680 | .760 |
| slot 5 coarse (rest) | .067 | .524 | .24 | .10 | .40 |
| slot 6 coarse (rest) | .180 | 1.074 | .60 | .83 | .72 |
| slot 6 flux < 1 cm | .915 | .304 | .636 | .561 | .935 |

U alone (ELEE) changes nothing relative to Early: H1 coarse .095, slot 6 flux .915.

**Shares** (degrade / recover, as fractions of the drift):

| Metric | W | U | V | U+V | mass |
|---|---|---|---|---|---|
| H1 coarse | .50 / .21 | .00 / .11 | .67 / .24 | .81 / .24 | .68 / .10 |
| H1 contrast | 1.28 / .09 | .00 / .18 | .53 / −.44 | .62 / −.41 | .65 / .01 |
| slot 5 coarse | .38 / .28 | 0 / 0 | .07 / .00 | .11 / .00 | **.73 / .59** |
| slot 6 coarse | .47 / .18 | .00 / .12 | .73 / .40 | .87 / .40 | .60 / .07 |
| slot 6 flux | .46 / .21 | 0 / 0 | **.58 / .57** | .58 / .57 | −.03 / .14 |

**Factorial main effects** (mean L−E, H1 coarse): W .089, U .022, V .140, M .112. The largest interactions are VM −.080 and WVM +.052.

**Deformation** (cf3 − rest, slots 4–5; error / minimum correlation):

| Combination | Error / correlation |
|---|---|
| Early | .85 / .53 |
| Late W only | 1.77 / **.08** |
| Late V only | .87 / .49 |
| Late mass only | .79 / .63 |
| Late | 1.87 / .20 |

At the pt0 fitting ports, Late V alone drops the correlation from .98 to .81, while Late W keeps it at .96.

**N pair.** The same labels and similar shares:

- H1 coarse U+V .59, W .28, mass .29;
- slot 6 flux V .70;
- slot 5 mass .66.

**Label: INTERACTING / MULTIPLE MECHANISMS.** It holds for R and for N, for both the coarse and the contrast metric.

## Observation

1. **The pooled unseen-source drift is not carried by one factor.** For H1 coarse, each single Late group (W, V, mass) on its own reproduces 50–68% of the degradation. Restoring any single group from Early recovers only 10–24%. The failure pathways are largely redundant: the Late model stays degraded unless several factors are restored together.
2. **Different unseen sources fail through different factors.**
   - Slot 6, in the bottom corner with no receivers shared with the fitting sources, loses its near-source concentration mainly through V. Its source mixture into the global basis grows about 800×, the UV share of its flux rises from 3% to 54%, and V moves the 1 cm flux both ways (degrade .58, recover .57 of the drift). W contributes about a third.
   - Slot 5 degrades mainly through its total mass (.78 → .62), which is a normalization effect (degrade .73, recover .59).
   - Directional contrast degrades most when the Late W is inserted (1.28× the drift), but restoring W alone does not recover it (.09).
3. **U alone has no effect.** U changes strongly: it concentrates toward the fit-port neighbourhoods, from .06 to .56 of its column mass, with entropy 11.7 → 6.6 bits. Inserting it alone changes nothing, because Early V is too small to weight it. Its influence appears only through V.
4. **Signed geometry change at unseen ports is lost mainly through W; deformation at the fit ports is harmed mainly by V.** Late W alone collapses the cf3 change correlation (.53 → .08).
5. **Fit-port quality barely moves across the 16 combinations** (worst .134–.138). The counterfactual changes act on unsupervised ports.

## Interpretation

These are post hoc, development-only inferences.

- **The primary hypothesis is supported in part.** Training does redistribute transport at unsupervised source ports. At the corner source it moves response from local W into a global UV path whose basis has become concentrated on the fit-port neighbourhoods; this is the slot-6 mechanism. That is not the whole drift:
  - an independent total-mass reduction at slot 5;
  - changes in W itself (contrast, unseen deformation);
  - mutual redundancy among these.

  Together these mean no single component accounts for the deterioration.
- **On what the label permits.** INTERACTING / MULTIPLE MECHANISMS does not license an intervention aimed at one factor alone. The counterfactuals show that restoring any one factor (only W, only U/V, only the mass) recovers at most about a quarter of the pooled drift. A single-factor intervention would therefore not be expected to remove the failure on this evidence. The source-specific attributions are post hoc and per source:
  - V dominates the near-source flux collapse at the uncovered corner source;
  - mass dominates at slot 5;
  - W dominates unseen deformation.
- **What the other outcomes would have permitted.**
  - LOCAL-MECHANISM DOMINANT would have pointed at the edge MLP W.
  - GLOBAL-FACTOR DOMINANT would have pointed at the UV path.
  - NORMALIZATION DOMINANT would have pointed at the mass/scale algebra.

  None was supported.

## Unresolved Question

1. **Why does the slot-6 V grow ~800× while the fit sources' V stays small?** It could be an extrapolation of the node head at uncovered positions or an Adam low-gradient effect. This is not resolved by read-only analysis.
2. **Are the factor contributions stable across Early/Late pairs** (e.g. 6,000 vs 20,000), and across seeds? Only one pair per arm was analysed.
3. **How do these mechanisms behave under S16 coverage** (WL71 D1), where the corner sources were partly covered?

## Architecture Verdict

**INTERACTING / MULTIPLE MECHANISMS** (R and N; coarse and contrast).

**Completion question.** *Which components of the current W+UV spatial factorization account for the deterioration of unseen-source transport during continued physical fine-tuning?*

Several, interacting:

- **V** (source mixture into a fit-concentrated global U) drives the near-source flux collapse at the uncovered corner source.
- **The per-source mass** drives slot 5.
- **W** drives contrast and unseen deformation degradation.
- The pathways are redundant: no single factor's restoration recovers more than ~24% of the pooled drift.
- **U** matters only through V.

No successor is proposed or implemented. No architecture or living-document change follows.

## Reproducibility

- **Commits:** 0b0a81e (protocol and code before any output), the test-tolerance fix (next commit), and this worklog's commit.
- **Inputs:**
  - WL71A checkpoints, 12 files, hashes in `experiments/fnto_attrib71b/checkpoints71a.sha256`;
  - WL71A `eval.json` (`eval71a.json`) for the identity check;
  - WL68 inputs and bundles.
- **Outputs** (`results/fnto_attrib71b/`):
  - `tests.json`, `stageA.json`, `stageC.json` (32 combinations, R and N), `label.json`;
  - `logs/`, including the GPU sampler;
  - `attempt1/`;
  - `frozen_hashes_before_run.txt`, `source_commit.txt`.
- **Runtime.** Chain total about 19 min:
  - tests about 25 s;
  - Stage A about 25 s;
  - Stage C 18 min, about 30 s per combination. This stage is CPU-bound float64 jackknife scoring with low GPU use, as the watcher noted.
- **Environment.** LabServer63, torch 2.3.1, no random draws.
