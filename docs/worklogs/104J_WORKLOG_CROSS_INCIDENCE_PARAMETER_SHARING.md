# Worklog 104J — Cross-incidence parameter sharing ablation of the certified ER (2026-11-21 KST)

## Authorization, classification and freeze

The owner's brief authorizes one bounded diagnostic architecture experiment. It asks whether the remaining angular response error of the certified ER comes partly from requiring one set of GGX lobe parameters to serve every incident angle. The owner named the batch WL104J. Under the AGENTS.md numbering policy it is a dependent follow-up of WL104H / WL104I: it closes their evidence gap, and the relaxed arms are diagnostic oracles, not a new production mechanism. Nothing is adopted.

- **Repository and branch.** `kn99e39/ReST-GI`, branch `ecro_lr3_adapter`, started at `6a5c7b9`. No new branch or worktree.
- **Host.** Local RTX 5080 (the default placement since 2026-11-20).
  - From 19:00 to about 20:15 KST another project's GPU job (TransGI_Reproduction data generation, 6 processes) shared the GPU. It covered development and most of the evidence fits; the owner moved that job to LabServer69 during the run.
  - Fit wall times are therefore contended. They are a compute budget only, not a cost comparison between arms.
- **Freeze.** The prospective protocol and all stage code were committed at `deefea9` before any evidence: [protocol](../../experiments/ecro_wl84_104j/protocol.json), [code](../../experiments/ecro_wl84_104j/).
- **Pre-freeze development (disclosed in the protocol).** Code was debugged on scratch copies only:
  - a smoke data set (64 lanes, 512 walks);
  - a full-size data set made from the frozen seeds, used for a 300-step timing probe and a 150-step end-to-end dev chain.

  The 150-step states were far from converged. Their printed contrasts and control outcomes are not evidence, and no threshold, rule, arm or data choice was changed after they were seen. After the dev chain, the only code changes were a guard for runs shorter than 2,000 steps and figure layout. The evidence run regenerated all data from the seeds after the freeze, and its fit set is identical to the dev copy.
- **Execution amendment 1 (`0350dae`, scheduling only).** With the GPU shared, the frozen chain ran its secondary fits one after another and would have taken several more hours. After the primary fits and S0a were finished, the chain shell was stopped. The already-running S0b process was left alive, and [`chain_resume.sh`](../../experiments/ecro_wl84_104j/chain_resume.sh) ran S1, S2 and the extended check concurrently, followed by the unchanged evaluation and decision. The commands, arguments, seeds and steps are identical, and no fit was restarted or discarded. The status file records the hand-over.
- **Post-hoc diagnostic.** [`posthoc104j_s1.py`](../../experiments/ecro_wl84_104j/posthoc104j_s1.py) was written after the frozen decision and is labelled where it appears. It changes no label.

## Implementation Fact

**Response family (confirmed in code).** Every arm uses the unchanged closure82 ER evaluator with the closure83 robust refraction:

F(wi, wo) = O0 + O1 + O2 + s(μi) s(μo) (L + I0 + I1 + I2 + I3)

- 3 outer GGX lobes, 4 inner GGX lobes at refracted coordinates (scaled 1/η²), and a Lambert term.
- Per lobe: shape (log-α_u, log-α_v logits) and an F0 shared by the three channels.
- Per channel: the amplitudes and 5 attenuation knots.
- One η.

No production file changed.

**Incident groups.** 10 groups, one per incident polar angle: 5, 20, 35, 50, 57.5, 60, 62.5, 70, 80 and 86°, each at azimuths 22.5° and 112.5°. These are exactly the incident directions of the WL104D Amendment 1 independent reference, so the 55–65° region has three explicit groups.

**Arms.** All arms use the same fit set, the WL82 objective (el1 + ealb per probe, mean over probes) and the WL82 Adam schedule (lr 0.03 → 3e-4 cosine, 12,000 steps, float32).

| Arm | Shared across incidences | Local per incident group | Trainable / material |
|---|---|---|---:|
| **A** shared ER-v1-GF | everything | — | 58 |
| **B** amplitude-only local oracle | lobe shapes, F0, attenuation, η (frozen from A) | 8 amplitudes per channel, budget Σλ ≤ 1 (log-ratio against an absorption slot) | 240 |
| **C** full local lobe oracle | attenuation, η (frozen from A) | B's amplitudes + roughness, anisotropy and F0 of all 7 lobes | 450 |
| **C without Fresnel** (secondary) | attenuation, η, F0 | amplitudes + roughness / anisotropy | 380 |

- **Frozen attenuation.** The attenuation knots are frozen at their maximal admissible scale (κ_c = max knot of A's ER-v1-GF state). This reproduces A exactly and gives the local amplitudes the original certified budget.
- **Initialization and starts.** B starts from A and C from B, in each of 4 starts: generic plus seeds 104911–13, the WL104I start set. The C fallback to an A-compatible start was never triggered.
- **Validity.** Each group's local state is a valid certified ER state, but the incident-conditioned response is a diagnostic oracle. It is not reciprocal, as measured below, and was never rendered.

**Data.** All layered references use the WL104D proposal: half WL83 mixture, a quarter uniform inside the 3° core, a quarter in the 3–12° shoulder. Walks are depth 1024, with two independent halves.

| Set | Role | Directions | Walks |
|---|---|---|---|
| Fit set (new) | targets for every arm | the 20 group directions × 1,024 fresh outgoing directions (seed 1041010) | 8,192 / lane, seed 1041011, sequence range from 2^40 |
| Nearby-polar transfer set (new) | transfer diagnostic only | midpoints 12.5 … 83° at both azimuths | 8,192 / lane, 2^41 range |
| New-azimuth transfer set (new) | transfer diagnostic only | the 10 group polars at azimuths 67.5° / 157.5° | 8,192 / lane |
| Evaluation reference (WL104D Amendment 1, archived) | evaluation only | same 20 incident directions, independent outgoing directions and walks, plus the peak grid | 4,096 / lane (16,384 in the hard_76 band) |

**Qualification (Stage 0, before fitting) passed.**
- Every frozen input was authenticated by hash: WL104G manifest, WL104D provenance, the materials83 descriptors, and the WL104I fit states.
- The evaluation reference passed its frozen WL104D precision rule.
- The new sets passed the same rule. Worst cases: 55–65° shape noise 0.0127 / 0.0137 / 0.0128 against 0.015; grazing 0.0118–0.0120 against 0.025; albedo ≤ 0.0006 against 0.008.
- Support: ≥ 192 active shoulder lanes and ≥ 256 core lanes per probe.
- Isolation: 0 active outgoing directions shared between fit and evaluation lanes; disjoint walk-sequence ranges; the evaluation-reference hash was unchanged before and after.
- Data generation took 31 s. [Data precision figure](../../evaluation/104J/fig08_fit_and_transfer_data_precision.png).

**Synthetic and numerical controls (Stage 2, gate) passed** ([tests.json](../../evaluation/104J/tests.json)).

| Test | Result |
|---|---|
| A vs original 64-parameter ER (float64) | 1.6e-16 |
| A vs compiled Dr.Jit runtime | 9.1e-7 |
| B section reproduces A | ≤ 4.9e-16 on R0, WL104D A/C, ER_L, WL104I B1, random and extreme states |
| C with B's state equals B | 0 |
| Group locality | changing groups 4 and 7 changes exactly probes 8, 9, 14, 15 |
| Terms ≥ 0, budget, term-sum identity | min term 5.6e-17; budget ≤ 0.99999986; term sum 3.8e-16 |
| Lambert cosine convention | exact |
| Deterministic replay, frozen-parameter masks, C-without-Fresnel F0 mask | bitwise 0 |
| Autograd vs finite differences | ≤ 2.6e-7 relative |
| Synthetic targets | S1 representable by B and S2 by C; both differ from any shared state |

## Measurement

el1 is the WL82 energy-weighted relative L1 of f·cos θo against the mean of the two independent evaluation halves. "Pooled" means the sum over (material, probe) in a cohort: development = iid_1, iid_0, hard_1; held out = iid_191, iid_159, block_161, block_186, hard_76, hard_53. Contrasts are the median over 4 starts with a 95 % cluster bootstrap over materials. All states are terminal; nothing was selected on evaluation data. [eval.json](../../evaluation/104J/eval.json), [decision.json](../../evaluation/104J/decision.json).

### Held-out angular contrasts

| Contrast | Development | Held out | Half a / half b (dev; held out) | Starts in the same direction | Label |
|---|---|---|---|---|---|
| **B − A** (amplitude sharing relaxed) | **−35.8 %** [−49.9, −24.2] | **−36.6 %** [−51.9, −26.8] | −35.3 / −35.3; −35.8 / −35.6 | 4/4, 4/4 | SUBSTANTIAL |
| **C − B** (lobe shape / F0 sharing relaxed) | **−14.4 %** [−21.7, −11.2] | **−15.6 %** [−23.4, −9.6] | −14.1 / −14.1; −15.1 / −15.0 | 4/4, 4/4 | MEASURABLE |
| **C − A** (total) | **−45.2 %** [−55.6, −38.4] | **−46.6 %** [−59.1, −35.6] | −44.6 / −44.5; −45.4 / −45.3 | 4/4, 4/4 | SUBSTANTIAL |
| C without Fresnel − B (secondary) | −13.5 % | −14.2 % | | | MEASURABLE |
| C − C without Fresnel (secondary) | −0.9 % | −1.5 % | | | NEGLIGIBLE |

- **Pooled el1, median start** (mean per probe, all 9 materials): A 0.0767, B 0.0489, C 0.0415, C without Fresnel 0.0422.
- **External controls.** Historical R0 and the WL104I A1 / B1 generic refits score 0.0809 on the same lanes, so the newly matched A (0.0767) is in the same range.
- **Directional albedo error (ealb).** B − A −57.5 % / −69.5 %; C − A −67 % / −77 %.
- **Normalized shape error (eshape).** B − A −34.6 % / −36.6 %; C − B −14.4 % / −15.6 %; C − A −44.2 % / −46.5 %.

### Per material

All 20 probes, median start. The last column is C − A over the grazing probes (≥ 80°) only. The full table over every material and incidence is in [table_el1_by_material_and_incidence.csv](../../evaluation/104J/table_el1_by_material_and_incidence.csv) and in the [reviewer page](../../evaluation/104J/index.html).

| Material | Cohort | A el1 | B el1 | C el1 | B − A | C − B | C − A | C − A at ≥ 80° |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| iid_1 | development | .0912 | .0460 | .0405 | −50 % | −12 % | −56 % | −60 % |
| iid_0 | development | .0545 | .0413 | .0324 | −24 % | −22 % | −41 % | −54 % |
| hard_1 | development | .1140 | .0799 | .0702 | −30 % | −12 % | −38 % | −46 % |
| iid_191 | held out | .0487 | .0254 | .0228 | −48 % | −10 % | −53 % | −65 % |
| iid_159 | held out | .0625 | .0508 | .0469 | −19 % | −8 % | −25 % | −37 % |
| block_161 | held out | .0642 | .0223 | .0191 | −65 % | −14 % | −70 % | −72 % |
| block_186 | held out | .0468 | .0244 | .0215 | −48 % | −12 % | −54 % | −84 % |
| hard_76 | held out | .1108 | .0828 | .0732 | −25 % | −12 % | −34 % | −52 % |
| hard_53 | held out | .0979 | .0672 | .0470 | −31 % | −30 % | −52 % | −64 % |

Every material improves in both contrasts. B − A ranges from −19 % to −65 % and C − B from −8 % to −30 %.

### Regions, incidence bands and signed residuals

Region L1 is the quadrature-weighted |model − reference| luminance in the region, divided by the total reference energy of the probe. It is pooled, best training-objective start.

| | Core < 3° | Shoulder 3–12° | Tail ≥ 12° | 5–20° | 35–50° | 55–65° | 70° | 80–86° |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A, development | .0095 | .0281 | .0571 | .062 | .058 | .095 | .112 | .153 |
| B, development | .0031 | .0096 | .0479 | .031 | .039 | .060 | .084 | .099 |
| C, development | .0020 | .0094 | .0423 | .028 | .038 | .058 | .078 | .078 |
| A, held out | .0072 | .0208 | .0490 | .054 | .057 | .070 | .075 | .116 |
| B, held out | .0043 | .0082 | .0338 | .025 | .038 | .046 | .053 | .066 |
| C, held out | .0015 | .0062 | .0316 | .023 | .036 | .046 | .050 | .044 |

- **Signed region energy** (mean |signed error| over materials and probes): core .082 / .026 / .012, shoulder .063 / .011 / .007, tail .037 / .007 / .006 for A / B / C.
- **Peak height** (independent peak grid, [figure 4](../../evaluation/104J/fig04_core_shoulder_tail_and_peak_error_by_material_and_incidence.png)): A's sign-changing peak errors (for example +22 % near normal and −8 % at 60° on iid_1, +29 % at 70° on hard_1) are mostly removed by B. At 86°, B leaves the peak overshoot on iid_0, hard_1, iid_159 and hard_76, and C reduces part of it.

### Training vs held-out, transfer and reciprocity

- **Overfitting.** Training-lane el1 reductions match the held-out reductions:
  - B − A: 37.3 / 37.1 % training vs 35.8 / 36.6 % held out;
  - C − B: 15.0 / 16.8 % vs 14.4 / 15.6 %.

  No contrast is overfit-dominated.
- **Nearby-polar transfer.** Local parameters evaluated at independently held-out midpoint polars with the neighbouring groups' parameters:
  - B − A −29.7 / −29.5 %;
  - C − B −6.1 / −7.9 %;
  - C − A −33.8 / −35.1 %.

  The lower- and upper-neighbour errors are similar (C 0.047 / 0.049, A 0.073).
- **New-azimuth transfer** (same polar, azimuths 67.5° / 157.5°): B − A −36.9 / −37.3 %, C − B −14.7 / −16.1 %.
- All transfer labels are TRANSFER_POSITIVE.
- **Reciprocity of the model.** Over 80 directions and all ordered pairs:

  | Arm | Σ\|f(wi,wo) − f(wo,wi)\| / Σ mean | 95th-percentile pair difference |
  |---|---|---|
  | A | 1.4e-15 | — |
  | B | median 2.0 % (max 4.6 %) | 29 % |
  | C | median 2.7 % (max 10 %) | 27 % |

  A is reciprocal by construction; B and C are not.

### Lobe parameters and overlap

[Figure 5a](../../evaluation/104J/fig05a_local_lobe_parameters_vs_incident_angle.png), [figure 5b](../../evaluation/104J/fig05b_lobe_overlap_and_energy_share_vs_incident_angle.png).

- **Term pairs with angular-response cosine ≥ 0.95** (per probe, of 28): mean 1.59 (A), 1.34 (B), 1.03 (C).
  - C has fewer such pairs than A at 46 % of probes and more at 17 %.
  - The change is irregular across incidence: pairs disappear in one group and reappear in the next.
- **C's local parameters.**
  - Several lobes jump between neighbouring groups, mostly lobes with small amplitude. Examples: roughness of a low-amplitude inner lobe changing by an order of magnitude between 57.5° and 60°; F0 values jumping between 0 and 1.
  - The well-populated lobes vary smoothly or stay near A's value.
- **B's local amplitudes** move strongly with incidence. On several materials, energy is handed between near-duplicate lobes, for example Outer 2 ↔ Outer 3 on iid_1 above 60°.

### Controls, convergence and resolution

| Item | Measured | Frozen rule |
|---|---|---|
| S0a null target, near-truth start | held-out el1 A .00020, B .00028, C .00043 | PASS (\|Δ\| ≤ 1e-3) |
| S0b null target, generic start (slack calibration) | A .00077, B .00067, C .00061 | descriptive |
| **S1 known amplitude variation** | A .0970, **B .0793 (82 % of A)**, C .0553 | **FAIL** (B ≤ 50 % of A required) |
| S2 known shape variation | A .0759, B .0410, C .0079 (19 % of B) | PASS (C ≤ 50 % of B) |
| Physical gain vs slack | B − A .031 / .026 vs null gain ≤ 1e-4 | separable |
| Last-2,000-step change ≤ 0.5 % | A 100 %, B 100 %, **C 81 %** (max 0.66 %), C without Fresnel 94 % of units | **FAIL for C** (≥ 90 % required) |
| Nesting (B ≤ A, C ≤ B in training objective) | 0 violations of 36 units | pass |
| Extended 24k-step check | arm medians change: A 0.44 %, **B 1.9 %**, C 0.33 %; training contrasts change 0.03 % (B − A) and 2.9 % (C − B) | **FAIL for B** (≤ 1 % required); contrasts pass |
| Extended held-out el1 (descriptive) | A .0766, B .0486, C .0413 (12k: .0767 / .0489 / .0415) | — |
| Reference resolution | half-split el1 noise / A el1 = 0.42 % (dev), 0.50 % (held out) | resolves (≤ 1.25 %) |

**Reference noise floor of el1** (el1 between the two evaluation halves / 2, per material): from 0.002 (iid_1) to 0.015 (hard_76, hard_1). Assigning the whole floor to the residual, C's pooled development el1 would still be at least 46 % of A's.

**Post hoc, S1 attribution** ([s1_truth_frozen_B.json](../../evaluation/104J/s1_truth_frozen_B.json)).
- **Setup.** Same noisy S1 target, optimizer and steps as the frozen run. B with the truth's shared parameters frozen (shapes, attenuation, η) and amplitudes started unmodulated.
- **Result.** It reaches held-out el1 0.00026: 0.27 % of the frozen A, at the S0a noise level of about 0.0002, starting from 0.18.
- **Meaning.** The S1 failure is not a lack of expressivity of the B parameterization. It arises because B inherits shared parameters from an A fit that compensated the amplitude variation. This is a property of the frozen A → B → C chain, and it limits the chain's amplitude/shape attribution.

**Compute.**
- Fit wall time for 4 starts × 9 materials × 12,000 steps (contended GPU): A 539 s, B 204 s, C 397 s, C without Fresnel 1,632 s. The last was the most contended.
- Peak allocated memory: 0.95 / 0.46 / 0.77 / 0.77 GB.
- Trainable parameters per material: 58 / 240 / 450 / 380. B and C use 4× and 8× A's parameters per material; the improvement is not capacity-neutral.
- Evaluation took about 30 s.

## Observation

- **Effect size and consistency.** Relaxing only the amplitudes removes about a third of the held-out angular error. It removes most of the core, shoulder and signed-energy error at low and middle incidences, and most of the directional albedo error.
  - Adding lobe-shape freedom removes a further 14–16 % of B's error. Almost all of it comes from roughness and anisotropy; F0 adds about 1 %.
  - Both effects are consistent across all 9 materials, all 4 starts and both reference halves.
  - Both persist on held-out outgoing directions and transfer to nearby held-out incidences, the polar transfer more weakly.
- **What remains after C.** More than half of A's error remains (C/A = 0.55 development, 0.53 held out). It is dominated by the tail (≥ 12° from mirror: about 0.042 of C's 0.054 total region L1 in development). There the signed energy error is already small (0.006), so the remainder is an angular-distribution error with near-zero net energy. Grazing incidence (80–86°) keeps the largest band error for B. C reduces the grazing error more than at other incidences.
- **Controls.** The synthetic controls show cross-mimicry. On a pure amplitude variation (S1), B recovered only 18 % of A's error while C recovered 43 %. On a pure shape variation (S2), B recovered 46 % of A's error.

## Interpretation

Each type of claim below is kept separate.

- **Observed angular fidelity improvement (diagnostic oracles only).**
  - Measured on the independent reference, at the tested 10 incidences and 9 materials: B − A −36 / −37 %, C − B −14 / −16 %, C − A −45 / −47 %.
  - It is reproducible across starts and halves, separable from optimizer slack, not overfit-dominated, and transfers to nearby held-out incidences.
- **Diagnostic approximation headroom.** Removing cross-incidence sharing of amplitudes and lobe parameters within the same seven-GGX structure exposes about 45 % of A's held-out el1. This is empirical headroom of non-reciprocal, incidence-indexed oracles with 4–8× the parameters. It is not a lower bound on what a valid shared BSDF could reach, and not a measured capacity limit.
- **Architectural cause: not established.**
  - The frozen attribution rule labels the split AMPLITUDE_PRIMARY (amplitude share 0.79 / 0.79). However, the S1 control failed: when the truth is a pure amplitude variation, the A → B → C chain credits most of the recoverable error to C, not B. In S2, B absorbs part of a pure shape variation.
  - The chained diagnostic therefore does not separate amplitude-sharing from shape-sharing mechanisms reliably. The post-hoc check supports the reason: with the true shared parameters, B recovers S1 to the noise level. The S1 failure therefore comes from the frozen shared parameters handed to B, which come from an A fit that has already compensated. The post-hoc check cannot change the label.
  - The amplitude/shape attribution of the physical headroom is not qualified.
- **Valid physical representation: not addressed.** B and C break reciprocity (2–3 % summed, 27–29 % at the 95th pair percentile). Their gains say nothing about a reciprocal, passive production BSDF.
- **Residual.** Even with both relaxations, the remaining error (more than half of A's, mostly tail-shape error) is not explained by cross-incidence sharing within this structure, at the tested granularity.

## Unresolved Question

- Whether the physical headroom is primarily amplitude redistribution or lobe-parameter freedom. The frozen chain does not identify it (S1 failed). The post-hoc check shows why, but on physical data no "true" shared parameters exist to freeze, so the split stays order- and path-dependent.
- Whether a reciprocal, passive representation can reach any part of the headroom. For example, a dependence on both μi and μo, or on the half vector. Untested; no such model was built.
- What limits the tail-shape residual after C: the GGX lobe forms, the attenuation / η sharing that stayed frozen, or the 10-group granularity. Untested.
- Optimization detail under the frozen rule: C 81 % of units within 0.5 % over the last 2,000 steps; B's extended-run change 1.9 %. Descriptively the held-out contrasts are unchanged at 24k steps (A / B / C .0766 / .0486 / .0413 vs .0767 / .0489 / .0415), but the frozen rule fails.
- Coverage: one coated-conductor family, 9 materials, 10 incident polars at 2–4 azimuths; no scene effect measured.

## Architecture Verdict

**Decision table** (frozen rules, [decision.json](../../evaluation/104J/decision.json)):

| Item | Label |
|---|---|
| Baseline / data / reference qualification | PASS |
| Numerical identities, nesting, masks, gradients | PASS |
| Synthetic controls | S0a PASS, **S1 FAIL**, S2 PASS |
| B − A amplitude sharing | SUBSTANTIAL (held out, both cohorts) |
| C − B lobe-parameter sharing | MEASURABLE |
| C − A total | SUBSTANTIAL |
| Overfit-dominated / slack-inseparable / start-unstable | none |
| Optimization resolved | **FAIL** (C last-2,000 rule; B extended-run rule) |
| Transfer | TRANSFER_POSITIVE for every contrast |
| Residual after C | RESIDUAL_LARGE (0.55 / 0.53 of A) |
| Attribution | AMPLITUDE_PRIMARY by the rule, **not qualified** (S1 failed) |
| **Hypothesis** | **INCONCLUSIVE** (CONTROL_FAILED, OPTIMIZATION_UNRESOLVED) |

**Decision-oriented summary.**
- **Baseline and controls qualified?** Data, reference and numerical controls yes. Synthetic attribution controls no: S1 failed.
- **Does amplitude-only relaxation explain the residual?** It explains about 36 % of A's held-out error, and most of the core, shoulder, albedo and signed-energy error. It does not explain the majority of the residual.
- **Does additional lobe-parameter freedom explain it?** It removes a further 14–16 %, mainly through roughness and anisotropy.
- **Generalization to held-out directions?** Yes for both: held-out outgoing directions equal the training reductions, nearby incidences transfer, and the new azimuths transfer.
- **Distinguishable from reference noise?** Yes: the noise ratio is 0.4–0.5 % of A's el1, and the result replicates on both halves.
- **Overfitting dominant?** No.
- **Lobe redundancy.** Not resolved. Overlap counts fall on average, but irregularly across incidence. Low-amplitude lobes take erratic local parameters.
- **Hypothesis status.**
  - Diagnostic headroom from relaxing cross-incidence sharing is observed and large.
  - Whether it is caused by amplitude or by lobe-parameter sharing is unresolved.
  - The frozen hypothesis verdict is INCONCLUSIVE.
  - More than half of the residual is not explained by either relaxation.

No downstream ER redesign is authorized or recommended here. The incident-conditioned oracles are not adopted, and the canonical ER and ER-v1-GF are unchanged. The owner retains the architecture decision.

## Reproducibility

- **Commits.** Freeze `deefea9`; execution amendment 1 `0350dae`. The report commit adds `posthoc104j_s1.py`, `evaluation/104J/`, this worklog and the living-document note.
- **Environment.** Local RTX 5080 16,303 MiB, `external/8dna26/.venv-cu128`: Python 3.11.9, Torch 2.8.0+cu128, Mitsuba 3.5.1, Dr.Jit 0.4.6. GPU shared with another project's job from 19:00 to about 20:15 KST.
- **Timing.** Chain 19:20–20:34 KST: data 31 s, tests 7 s, primary fits 46 min, secondary fits concurrent, evaluation about 20 s.
- **Seeds.**
  - Fit directions 1041010, fit walks 1041011 (pair0 2^27).
  - Transfer directions 1041020 / 1041022, transfer walks 1041021 (pair0 2^28).
  - Synthetic noise 1041030, synthetic start 1041031, bootstrap 1041040, tests 1041050.
  - Starts generic + 104911–104913.
- **Inputs.** Authenticated through the WL104G manifest, the WL104D Amendment 1 provenance, the materials83 descriptor hashes and the WL104I manifest. Hashes are in each stage JSON.
- **Raw outputs.** `results/ecro_wl84_104j/primary/`: data, tests.json, fit_primary, fit_extended, synth_S0a / S0b / S1 / S2, eval (eval.json, decision.json, lanes.npz), logs, status.txt. Post hoc: `results/ecro_wl84_104j/posthoc/`.
- **Reviewer package.** [evaluation/104J/index.html](../../evaluation/104J/index.html), figures fig01–fig08 (MEASURED / CONCEPTUAL tags), the comparison table and `figures_manifest.json` with hashes and figure-data checks (all pass).

```bash
cd experiments/ecro_wl84_104j
bash chain.sh ../../results/ecro_wl84_104j/primary 12000        # frozen chain (amendment 1: chain_resume.sh after the primary fits)
PY=../../external/8dna26/.venv-cu128/Scripts/python.exe
PYTHONUTF8=1 $PY -B posthoc104j_s1.py ../../results/ecro_wl84_104j/primary ../../evaluation/104J/s1_truth_frozen_B.json
PYTHONUTF8=1 $PY -B figures104j.py ../../results/ecro_wl84_104j/primary ../../evaluation/104J
```
