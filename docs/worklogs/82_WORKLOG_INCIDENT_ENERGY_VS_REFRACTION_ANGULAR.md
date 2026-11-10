# Worklog — Incident-dependent Energy vs. Refraction-conditioned Angular Representation (2026-11-10)

## Origin

- **Brief.** WL82, "Incident-dependent Energy vs. Refraction-conditioned Angular Representation" (user, 2026-11-10). A bounded representation-mechanism experiment that attributes WL81's representation failure to H1 (incident-dependent energy) or H2 (refraction-conditioned angular shape). Not a production FNTO implementation; no learned compiler; no successor authorised.
- **Where.** Branch `Selective_Recompute` (no new branch / worktree), LabServer63 (main server, idle at start), RTX 3080 Ti 12 GB, Mitsuba 3.5.1 / Dr.Jit 0.4.6 / torch 2.3.1 (WL81 environment, `~/rest_compiled81/env.sh`).
- **Isolated paths.** `experiments/fnto_compiled82/`, `results/fnto_compiled82/`, `results/evaluation/82/`, `docs/plans/wl82/`; server `~/rest_compiled82` (dev smoke `~/rest_compiled82_dev`). WL80C / WL80D / WL81 code, data and evidence are imported or read, never modified.

## Sources read before design

WL80C / WL80D / WL81 worklogs; WL81 protocol v1 + amendments 1–3; `layered81.py`, `closure81.py`, `energy81.py`, `fit81.py`, `eval81.py`, `data81.py`, `contract81.py`, `diag81_posthoc.py`, `viz81.py`; WL81 Stage C results and post hoc JSONs; `results/evaluation/81/`.

## Holdout status (verified before the protocol)

- WL81 generated `train / val / gate / test_iid / test_block / train_eval` in Stage A1. `chain.status` shows no stage after C (D/E/F not reached).
- The post hoc diagnostic and the visuals read only the gate set.
- On LabServer63 the atime of `eval_test_iid.npz` (10:02:34 UTC) is the WL81 highlight generation reading its `phis`. The atimes of `fit_test_iid`, `fit_test_block`, `eval_test_block` and `hl_test_iid` (10:43:11–14 UTC) coincide with the creation of `data_sha256.txt` (10:43:17 UTC), i.e. hashing.
- No fit, candidate, control or visual ever read the holdouts. **test_iid and test_block are the independent evaluation sets of WL82; the gate set is development data** (it was inspected in WL81 and is used here for development and mechanism diagnosis only).

## Design (frozen in `protocol/compiled82.json` before any candidate evaluation)

**Stage A decomposition.** For every incident direction, A(wi) = ∫ f cos dwo and p(wo|wi) = f cos / A.
- E_shape = energy-weighted total variation of p between closure and reference (in [0, 2]). A closure channel with zero albedo counts as TV 1 (no division, no exclusion).
- Oracle (diagnostic only; uses reference albedo at evaluation time, may violate reciprocity): f_or = f_R8 · A_ref^ind / A_R8^ind. The ratio comes from an **independent** albedo set: 4096 fresh wo × 256 fresh walks at every evaluation incidence, with the closure albedo computed on the same independent samples.
- Algebraic identity, checked for every arm: the same-sample oracle has E_L1 = E_shape exactly and E_alb = 0.
- Items:
  - A: the WL81 frozen direct fits (log1p);
  - B: A + oracle;
  - C: shape-focused R = 8 fit (normalised-shape objective, energy free per incidence);
  - D: C + oracle;
  - additionally the WL82-protocol C0 refit + oracle.

**Arms.** All arms keep the WL81 corrected R = 8 lobes. Fixed lobe groups: INNER = Lambert + GGX 4–7, OUTER = GGX 1–3.

| Arm | Definition | Params | State |
|---|---|---|---|
| C0 | WL81 closure | 48 | 53 |
| E | outer lobes + s(μ_i) s(μ_o) · inner lobes; s = per-channel piecewise-linear knots at μ ∈ {0, .15, .4, .7, 1}, values in [0, 1] | 63 | 68 |
| Einc | s(μ_i) · C0 — incident-only, **non-reciprocal diagnostic** (deployable analogue of the oracle) | 63 | 68 |
| ER | E with the inner GGX lobes evaluated at the coat-refracted directions t(w) = (w_x/η, w_y/η, √(1 − (1 − w_z²)/η²)) with the Jacobian η⁻², one fitted η_e ∈ [1, 2.5]; ER(η = 1) ≡ E | 64 | 69 |

- **Energy modes:**
  - cert = WL80C softmax (Σλ ≤ 1);
  - unb = softplus (diagnostic, isolates the energy-constraint confound).
- **Certificate (cert, C0 / E / ER).** Every component albedo is ≤ 1:
  - for the refracted lobe this follows from the solid-angle identity cos θ_o dω_o = η² cos θ_o' dω_o' (the outside albedo is the inside integral over the refraction cone);
  - hence a(wi) ≤ Σλ ≤ 1.
- **Reciprocity** is exact by construction for C0 / E / ER.
- **Grazing.** Horizon-safe forms (WL81) for every lobe; refracted inner directions never graze.
- **H2 specificity.** ER differs from E only by the deterministic refraction remap (one scalar per material). The arms are not exactly capacity-matched: 64 vs 63 parameters and a higher evaluation cost (reported).

**Fitting.**
- New fit sets: 64 wi × 64 wo × 1024 walks per material. This is a deviation from WL81's 64 × 16: per-incidence albedo and shape need many wo per wi.
- Objectives:
  - metric = per-incidence relative L1 + directional-albedo error, equal weights, fixed before any WL82 result from the WL81 post hoc finding about the log1p loss;
  - shape = per-incidence normalised TV.
- Optimiser: the WL81 direct-fit Adam (0.03 → 3e-4 cosine, warm-up 50).
- Steps:
  - 12000 primary;
  - 6000 and a perturbed init on the gate set;
  - predeclared convergence rule |Δ median| ≤ 0.005, with a single escalation to 24000 steps.

**Decision (test_iid).**
- E (invalid): any validity condition fails (reproduction, focused tests, cert contracts, E_shape noise rule, split).
- A: E_cert passes the WL81 gate.
- C: otherwise, if the oracle-corrected R = 8 (C0 or shape fit) reaches R1.
- B: otherwise, if ER_cert passes the gate with ≥ 25 % lower median E_L1 and E_shape than E_cert.
- D: otherwise.
- test_block is a region check (R1 / R2; it has no highlight set).

**Real materials.** 8 predeclared named finishes (gold, copper, chrome, aluminium car paint, candy red over silver, satin brass, milky silver, rough iron) with published metal F0 values. They are fitted and evaluated like the other sets, plus sphere shading images (direct lighting at the surface, no GI).

## Implementation (pre-freeze)

- `closure82.py` (arms, decode, torch sampler / pdf)
- `data82.py` (reproduction, fit sets, independent albedo sets, real set)
- `fit82.py`
- `eval82.py` (decomposition, oracle, highlight, strata)
- `contract82.py` (focused tests T0–T10; physical contract T1–T4)
- `gate82.py` (convergence rule, verdict)
- `viz82.py`
- `run82.sh` (one detached chain, job pool)

WL81 modules are imported unchanged.

## Dev smoke and pre-freeze corrections (non-evidence; dev quality numbers not interpreted)

Dev root `~/rest_compiled82_dev`: 16 materials per WL81 set, 8 real materials, 200 / 400 / 800 steps.

**Run 1–3 results.**
- Stage R reproduced the WL81 `eval_gate` and `fit_gate` prefixes **bit-identically**. Every later stage executed, including the escalation path, the contract, the visuals and the verdict mapping.

**Corrections made before the freeze** (also recorded in `pre_freeze_corrections` of the protocol):
1. **T10 test bug.** The same-sample oracle has E_alb = 0 only where every closure channel has albedo > 0. The synthetic zero-channel group is now excluded from that check.
2. **T5 test design.** The random test closures included near-delta lobes. The uniform-quadrature standard error is unreliable there: the identical sampler code gave C0 z 3.6 against Einc z 12.
   - Random closures are now restricted to sampled lobes with α ≥ 0.1 (the WL80C chi-square eligibility principle).
   - The chi-square threshold is now 1e-4 for 72 tests (multiplicity).
   - Two seeds afterwards: max z ≤ 2.9, chi-square min p ≥ 0.001.
3. **Torch test-sampler pdf.** The dev own-sampler audit showed albedo means up to 1e10 for every arm, including C0, while still passing the formal 4-se rule because se grows with the tail.
   - Cause: `vndf_pdf_wo` required wi·h > 0. That holds mathematically for reflection, but at near-mirror grazing pairs float32 cancellation in wi + wo made it fail. The pdf became 0 where eval was non-zero, and the sample weight reached 3e9.
   - Fix: validity now equals eval's (cos_i, cos_o > 1e-7, |wi + wo| > 1e-6), as in the WL81 Dr.Jit runtime. The eval path was never affected.
   - Afterwards, at full audit sample counts: cert arms max albedo 1.004 with 0 violations; unbounded arms up to 23, mostly adversarial λ (report only).
4. **numpy-bool JSON serialisation** in the test harness.

**Full-scale probe** (300 gate materials, 64 × 64 fit set):
- One fit saturates the GPU: ER 58–63 ms/step, C0 26–32 ms/step, peak 1.6 GB. Four concurrent processes bring no throughput gain, so the chain uses MAXJ = 2.
- `torch.compile` was probed and **rejected**. It is 2.2–2.5× faster, but its forward deviates from eager by about 3 % on the s(μ) arms, and compiled reruns are not bit-identical. All fits are eager.
- Expected main-run wall clock: about 6.5–7 h.

## Freeze and run

- **Freeze.** Commit `705db09`: protocol v1 and all WL82 code. Deployed with `git -c core.autocrlf=false archive` to `~/rest_compiled82/code`.
- **Chain.** One detached chain, `run82.sh`, on LabServer63 (main server, idle), started 2026-11-09 15:54 UTC and finished 2026-11-10 00:17 UTC (about 8.4 h).

| Stage | UTC | Result |
|---|---|---|
| R reference reproduction | 15:54 | REPRODUCED: WL81 `eval_gate` / `fit_gate` prefixes bit-identical |
| D data | 15:54–15:56 | fit sets 4 × (64 wi × 64 wo × 1024 walks); real evaluation set passes the WL81 noise rule at 4096 walks; independent albedo sets: relative SE of A_ref 0.32–0.36 % (median) |
| T focused tests | 15:56–15:57 | PASS: T0 nesting ≤ 4.9e-12, T5 sampler / pdf max z 3.2 and χ² min p 0.014, T6 Jacobian max z 0.34, T8 bit-identical, T9 no overlaps / seed clashes, T10 identity |
| FG gate fits (30) + EG | 15:57–20:15 | |
| CONV | 20:15 | every arm converged (largest Δ 0.0039, E_cert); no escalation; perturbed-init spread ≤ 0.0019 |
| FH holdout and real fits (30) + EH | 20:15–00:10 | |
| K physical contract | 00:10–00:15 | cert arms PASS |
| V visuals, X verdict | 00:15–00:17 | **OUTCOME A** |

**Post-chain additions** (do not change any verdict):
- `bench82_posthoc.py` runs the evaluation-throughput measurement that the protocol declares in `evaluation.metrics` but the frozen chain code omitted.
- The independent review checks below.

## Final report

### Implementation Fact

- **Arms.** All keep the WL81 corrected R = 8 lobes.
  - **C0** (48 parameters / 53 state floats).
  - **E**: inner-group (Lambert + GGX 4–7) symmetric attenuation s(μ_i)s(μ_o); s is piecewise linear on 5 μ knots per channel (63 / 68).
  - **Einc**: incident-only s(μ_i) on the whole closure; non-reciprocal diagnostic (63 / 68).
  - **ER**: E plus the inner GGX lobes evaluated at coat-refracted directions with the η⁻² solid-angle Jacobian and one fitted η_e ∈ [1, 2.5] (64 / 69).
  - **Energy modes:** cert (softmax Σλ ≤ 1, a mathematical sufficient condition for passivity, with the solid-angle identity for refracted lobes) and unb (softplus, diagnostic).
- **Fits.** Per-material direct fits on new fit sets of 64 wi × 64 wo.
  - Objectives: metric (per-incidence relative L1 + albedo error) or shape (per-incidence normalised TV).
  - WL81 Adam schedule, 12000 steps.
  - Gate set additionally: 6000-step and perturbed-init runs.
- **Evaluation.** WL81 evaluation sets and WL81 metrics (E_L1, E_alb, highlight) and thresholds, plus:
  - E_shape;
  - an oracle albedo correction from independent albedo samples;
  - the algebraic identity check;
  - strata, the failure cluster and named materials;
  - 8 real-material finishes with sphere images.
- **Not done.**
  - No learned compiler; no Dr.Jit runtime for E / ER; no GPU runtime, compression or rendering utility.
  - No FNTO or production change.
  - WL80C / WL80D / WL81 code and evidence are unchanged.

### Measurement

**Reference validity.**
- WL81 targets reproduced bit-identically. The WL81 Stage C numbers are reproduced through the WL82 evaluator: control 0.21273, candidate and unbounded 0.18248, peak errors identical.
- E_shape noise medians are 0.007 (gate / test_iid), 0.004 (test_block) and 0.011 (real). Every value is ≤ 1/4 of the threshold.
- Oracle identity (same-sample oracle E_L1 vs E_shape): max deviation ≤ 4.4e-16 for every arm and set. No zero-albedo channels occurred.

**Optimisation.**
- 12000 vs 6000 steps: |Δ median| ≤ 0.0039 for every arm, so every arm counts as converged.
- Perturbed-init spread ≤ 0.0019.

**test_iid (primary, untouched holdout), primary fits.**

| Arm | E_L1 median / p90 / max | E_alb median / p95 | E_shape median / p90 | Oracle E_L1 median | Grazing E_L1 | Peak ° / \|log w\| | WL81 gate |
|---|---|---|---|---|---|---|---|
| C0_cert (control) | 0.120 / 0.261 / 0.76 | 0.037 / 0.152 | 0.105 / 0.236 | 0.105 | 0.189 | 0.74 / 0.090 | FAIL (R1, R2) |
| C0_unb | 0.119 / 0.260 / 0.76 | 0.037 / 0.154 | 0.105 / 0.236 | 0.104 | 0.190 | 0.77 / 0.090 | FAIL |
| **E_cert** | **0.073 / 0.173 / 0.39** | **0.007 / 0.035** | 0.072 / 0.174 | 0.072 | 0.109 | 0.57 / 0.055 | **PASS** |
| E_unb | 0.043 / 0.099 / 0.27 | 0.004 / 0.022 | 0.043 / 0.099 | 0.043 | 0.048 | 0.51 / 0.031 | PASS |
| Einc_unb (diagnostic) | 0.084 / 0.166 / 0.40 | 0.006 / 0.025 | 0.084 / 0.166 | 0.084 | 0.090 | 0.61 / 0.049 | PASS |
| **ER_cert** | **0.052 / 0.113 / 0.32** | **0.006 / 0.029** | 0.052 / 0.111 | 0.052 | 0.083 | 0.54 / 0.044 | **PASS** |
| ER_unb | 0.038 / 0.073 / 0.17 | 0.005 / 0.022 | 0.037 / 0.073 | 0.037 | 0.050 | 0.57 / 0.022 | PASS |
| C0_unb_shape (D2) | (energy free) | — | 0.083 / 0.165 | 0.084 / 0.165 | — | 0.65 / 0.053 | oracle PASS |
| E_unb_shape | (energy free) | — | 0.040 / 0.090 | 0.041 | — | 0.57 / 0.027 | oracle PASS |
| ER_unb_shape | (energy free) | — | 0.032 / 0.068 | 0.032 | — | 0.63 / 0.021 | oracle PASS |

**Gate set (development; WL81-inspected), Stage A items.**
- A: WL81 frozen fits, control / candidate E_L1 0.213 / 0.182.
- B: A + oracle, 0.193 / 0.168.
- C0 metric refit 0.119, + oracle 0.106.
- C: shape-focused R8, E_shape 0.084.
- D: C + oracle, E_L1 0.084.
- E_cert 0.073, Einc 0.085, ER_cert 0.055. The gate numbers agree with test_iid to within 0.003.

**test_block** (region check, R1 / R2 only; no highlight set):

| Arm | E_L1 median | E_alb median | R1 / R2 |
|---|---|---|---|
| C0_cert | 0.082 | 0.039 | R1 pass, R2 fail |
| E_cert | 0.059 | 0.014 | both pass |
| ER_cert | 0.053 | 0.016 | both pass |

**Real materials** (8 finishes, full WL81 gate):

| Arm | E_L1 median / p90 | E_alb median | Gate |
|---|---|---|---|
| C0_cert | 0.147 / 0.297 | 0.030 | FAIL |
| E_cert | 0.093 / 0.227 | 0.012 | PASS (\|log w\| 0.113 against 0.14) |
| ER_cert | 0.067 / 0.152 | 0.008 | PASS |

Per-material mean E_L1, C0 → E → ER:

| Finish | C0 | E | ER |
|---|---|---|---|
| polished clear-coated gold | 0.265 | 0.236 | 0.140 |
| thin-varnish chrome | 0.214 | 0.158 | 0.089 |
| satin brass | 0.183 | 0.115 | 0.074 |
| copper | 0.201 | 0.139 | 0.138 |
| milky silver | 0.107 | 0.041 | 0.041 |

**Failure cluster** (predeclared: lowest-tercile τ, lowest-tercile α_b, highest-tercile reflectance; test_iid n = 10, gate n = 12; not gated). Medians of per-material mean E_L1 on test_iid:

| Arm | E_L1 | E_shape | E_alb |
|---|---|---|---|
| C0_cert | 0.255 | 0.231 | 0.089 |
| Einc | 0.180 | 0.181 | — |
| R8 shape-focused (energy free) | — | 0.181 | — |
| E_cert | 0.184 | 0.185 | 0.017 |
| E_unb | 0.094 | — | — |
| E shape-focused | — | 0.084 | — |
| ER_cert | 0.109 | 0.108 | 0.009 |
| ER_unb | 0.067 | — | — |
| ER shape-focused | — | 0.069 | — |

**Named gate materials** (mean E_L1 / E_shape):

| Material | C0 | Einc | E_cert | ER_cert | R8 shape fit (E_shape) | ER shape fit (E_shape) |
|---|---|---|---|---|---|---|
| #139 | 0.394 / 0.357 | 0.272 | 0.286 | 0.151 | 0.268 | 0.103 |
| #222 | 0.259 | 0.185 | 0.193 | 0.121 | 0.184 | 0.071 |

**Physical contract** (fitted closures of every set + 50 adversarial per arm; 958 closures, 45,984 audit rows).
- **Cert arms C0 / E / ER: PASS.**
  - Reciprocity max rel ≤ 1.0e-13.
  - Non-negative and finite, including μ = 0 and 1e-12.
  - Σλ ≤ 1, s ∈ [2e-5, 1], η_e ∈ [1, 2.5].
  - Own-sampler audit: 0 violations; max albedo 1.004 (C0, within se), 0.977 (E), 0.972 (ER).
- **Unbounded arms** (report only):
  - violations only in about 5.2 % of closures, i.e. the 50 adversarial closures (plus one fitted C0_unb closure);
  - fitted E_unb / ER_unb closures had max albedo ≤ 1 at every audited direction.
- **Einc** is non-reciprocal, as designed (max rel 1.0).

**Fitted mechanism parameters.**
- E's s profile (median over materials, μ = 0 / .15 / .4 / .7 / 1): 0.57 / 0.74 / 0.90 / 0.95 / 0.99 on test_iid, i.e. falling towards grazing.
- ER's η_e does not track the true coat IOR:
  - correlation −0.20 to +0.14 across arms and sets;
  - median |η_e − η| 0.35–0.67;
  - 10th / 90th percentiles at the range limits 1.0 / 2.5 on gate and test_iid;
  - η_e ≈ 1 (that is, E) on test_block.

**Cost.**
- Evaluation throughput (post-chain, torch eager, 2^22 lanes, relative to C0 at 4.7 ns/eval): Einc 1.12×, E 1.63×, ER 1.72×.
- Fit step at 300 materials: C0 26–32 ms, ER 58–63 ms.
- Runtime state: 53 / 68 / 69 floats (C0 / E / ER).

### Observation

1. **The WL81 representation failure is removed by incident-dependent energy modulation at the population level.**
   - The physically admissible E passes every WL81 representation criterion on the untouched test_iid, on test_block (R1 / R2) and on the real set.
   - The WL82 C0 control under the same protocol fails R1 and R2.
2. **Part of WL81's failure was the fitting protocol, not the basis.**
   - The same C0 basis drops from 0.18 (WL81 log1p objective, 64 × 16 data) to 0.12 under the per-incidence metric objective and 64 × 64 data. That is still a fail, with E_alb 0.037.
3. **Rescaling energy after the fact is not enough; refitting with free energy is.**
   - The oracle applied to the C0 fit lowers E_L1 only 0.120 → 0.105 (energy share 13 %), because the C0 fit already compromised its shape to match energy with incidence-independent weights.
   - When energy is free during fitting, R8 reaches 0.084: the non-reciprocal Einc and the shape-focused R8 fit plus the oracle give the same number. That passes R1.
   - The reciprocal E reaches 0.073, i.e. better than pure energy (its outgoing attenuation also reshapes the distribution: E_shape 0.072 < 0.084).
4. **Refraction adds further accuracy everywhere and matters most in the failure cases, but it is not needed for the population target.**
   - ER_cert vs E_cert: 0.052 vs 0.073 overall; 0.109 vs 0.184 in the failure cluster; 0.151 vs 0.286 on #139.
   - The shape-focused angular ceiling falls from R8 0.083 to E 0.040 to ER 0.032 overall, and from 0.181 to 0.084 to 0.069 in the cluster.
5. **The certificate is conservative for E / ER.**
   - The unbounded E / ER fits are clearly better (0.043 / 0.038) and were empirically passive at every audited direction.
   - So the cert gap is the price of the Σλ ≤ 1 sufficient condition. Consistent with the design-time expectation: a coat lobe rising to Fresnel 1 at grazing and a base attenuated by transmission sum to ≤ 1 physically but need Σλ > 1.
6. **Visuals.**
   - C0's directional albedo is flat with θ_i where the reference falls (#139, #202). E and ER follow the reference.
   - C0 fits contain a spurious narrow mirror spike (normalised slices). The ER slices follow the reference tails most closely.

### Interpretation (inference)

**H1 is supported in the tested scope.** WL81's failure was mainly that an R = 8 closure with incidence-independent weights cannot express a directional albedo that falls with incidence; a compact, reciprocal, certified incident-dependent attenuation suffices for the WL81 representation target.

**H2 (necessity) is not supported for that target.** But the refraction-aware angular mechanism is not empty:
- in the predeclared thin-slab / sharp / high-reflectance cluster, the R8 angular basis has an E_shape ceiling of about 0.18 even with perfect energy;
- the outgoing attenuation of E halves it;
- refraction lowers it further.

The fitted η_e does not identify the physical coat index. ER's gain should be read as a useful refraction-shaped angular degree of freedom, not as evidence that the representation learned the coat's refraction.

### Unresolved Question

- **Cluster-level adequacy.** Failure-cluster medians stay above 0.10 for E_cert (0.18) and ER_cert (0.11). Whether a cluster-level criterion would require ER, or something beyond it, was not predeclared and is not decided here.
- **η_e identifiability.** Fitted values hit the range limits. Whether a descriptor-tied η, or a wider range, would change ER is untested.
- **Certificate tightness.** A passivity certificate that allows the coat-Fresnel / base-transmission complementarity (Σλ > 1 with a proof) was not attempted.
- **Learned compiler generalisation** of E / ER, sampling efficiency with the new mechanisms, Dr.Jit runtime, GPU / rendering utility: all untested.
- **Generality** beyond this coated-conductor family.

### Architecture Verdict

| Entry | Verdict | Evidence / scope |
|---|---|---|
| Implementation Readiness | READY | T0–T10 pass; reproduction bit-identical |
| Reference Validity | QUALIFIED | bit-identical WL81 targets; E_shape noise ≤ 0.011; A_ref^ind SE 0.33 % |
| Optimisation Sufficiency | SUFFICIENT | Δ ≤ 0.0039, reinit ≤ 0.0019; no escalation |
| Energy Representation | SUPPORTED (E_cert passes R2: E_alb 0.007 / 0.035) | the certificate costs fidelity (cert 0.073 vs unb 0.043) |
| Angular Representation | R8 sufficient for the population target once energy is free (0.084); insufficient in the failure cluster (0.18) | ER is the best tested angular family (0.032 / 0.069 shape ceiling); η_e not physically identified |
| Physical Validity | PASS for C0 / E / ER cert (certificate + T1–T4) | unb / Einc report only |
| Runtime Feasibility | measured descriptively only (eval 1.63× / 1.72× C0) | no runtime-utility claim |

**Completion answer.**
- Correcting incident-dependent energy explains WL81's representation failure at the level of the WL81 target. **Outcome A.**
- A refraction-aware angular mechanism is not required to reach that target. In the thin-slab, sharp, high-reflectance cases it is the mechanism that further reduces the remaining angular error.
- A failed direct fit is not a capacity proof, and an improved metric is not a validated production architecture.
- Nothing is adopted into canonical FNTO; no successor is authorised.

### Reproducibility

- **Commits:** protocol + code `705db09`; post-chain throughput script and this report in the closing commit.
- **Server:** `~/rest_compiled82` (`code/COMMIT`, `chain.status`, `results/`).
- **Local copy:** `results/fnto_compiled82/results/`: JSONs, lane npz, fitted parameters, logs, contract, tests, `verdict82.json`, `bench82_posthoc.json`.
- **WL81 inputs:** sha256 in `results/fnto_compiled81/results/data_sha256.txt`.
- **Seeds:** protocol `seeds` (82010–82120); throughput 82200.
- **Environment:** Mitsuba 3.5.1 `cuda_ad_rgb`, Dr.Jit 0.4.6, torch 2.3.1+cu121 (eager), RTX 3080 Ti, LabServer63.
- **Visuals:** `results/evaluation/82/` (manifest with sha256 and scales). Every figure is computed from the fitted parameters and the reference walks of the evaluated sets; materials are selected by the C0 control ranking.
