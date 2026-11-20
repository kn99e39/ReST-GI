# Worklog 104H — Certified ER component-level causal ablation (2026-11-20 KST)

## Authorization, classification and freeze

The owner's brief asks for a diagnostic, hierarchical component ablation of the frozen certified ER response. Its purpose is error localization, not fidelity improvement. The batch is a suffix of the WL104 line under the AGENTS.md policy: it closes the evidence gap left by WL104B–G (which internal response terms carry the lobe-profile error) without a new mechanism, training or representation. The owner named it WL104H.

- **Repository and branch.** The configured origin is `kn99e39/NeuralShaderDemo`. `git ls-remote https://github.com/kn99e39/ReST-GI.git` advertised `ecro_lr3_adapter` at `bf98012`, equal to the local HEAD, so both names resolve to the same repository. The only untracked file, `experiments/ecro_fabric105/review105a.py`, belongs to another session and was not touched.
- **Host.** Local RTX 5080. The owner directed WL104E–G rendering of this line to the local machine, and every input of this batch is local. This placement was not re-confirmed separately for WL104H.
- **Freeze.** Protocol and instrumentation were committed at `fbdef22` before any evidence. [Protocol](../../experiments/ecro_wl84_104h/protocol.json).
- **Pre-freeze development changes.** Two tolerances were set from development numerics before evidence. Neither depends on the residual or on any comparison.
  - The Torch–Dr.Jit per-term parity limit is 1e-4 of the term peak instead of 2e-5. The worst development case (2–4e-5) is a refracted inner lobe at a grazing pair (cos 0.008 / 0.02) with η ≈ 1.0002. There the float32 runtime's η² − 1 cancels.
  - The Jacobian finite-difference check uses h = 1e-4 with a 1e-4 column-norm floor. At h = 1e-5 the error was roundoff on columns of about 1e-7 of the largest norm.
- **Development smokes** (non-evidence; they printed only identities and timings).
  - The sub-chunked replay matched a native render to 3.4e-8 (E1) and 6.1e-8 (E2) relative L1.
  - Deterministic replay was bitwise identical.
  - The held-out camera reproduced the archived ROI masks with 0 mismatched pixels.
  - One full 512-spp launch took 11.2 s at a 4.0 GiB Dr.Jit peak.

## Implementation Fact

**Instrumentation.** All of it is experiment-local in [`experiments/ecro_wl84_104h/`](../../experiments/ecro_wl84_104h/). No shared closure, renderer, state, proposal, geometry, camera, light, mask or reference changed.

- `comp104h.py` is the exact term graph, in Torch float64 and as a Dr.Jit subclass of `closure83_dr.DrClosure83` that exports each term with the native operations and order.
- `response104h.py` is the response-space phase.
- `params104h.py` is the parameter phase.
- `scenes104h.py` is the scene phase.
- `tests104h.py` holds the focused tests.
- `package104h.py` builds the reviewer artifacts.
- `posthoc104h_coupling.py` is post hoc and labelled as such.

**Verified decomposition.** The decomposition is cosine-weighted, F = f cos θ_o:

F = O0 + O1 + O2 + S · (L + I0 + I1 + I2 + I3), with S = s(μ_i) s(μ_o).

| Term | Definition | Learned raw parameters (64) |
|---|---|---|
| O_k, k = 0..2 | λ_{c,1+k} g_k(wi, wo) cos θ_o, outer anisotropic GGX with Schlick F0_k | log α_u / log α_v / F0 logits z[3k..3k+2]; softmax slot 1+k of z[21 + 9c + r] |
| L | λ_{c,0} cos θ_o / π | softmax slot 0 |
| I_k, k = 0..3 | λ_{c,4+k} g_{3+k}(t(wi), t(wo)) cos θ_o / η², refracted inner GGX | lobe 3+k logits; slot 4+k; η logit z[63] |
| s(μ) | piecewise-linear interpolation of 5 knots per channel | z[48..62] (sigmoid) |
| absorption | 9th softmax slot; sets the remaining energy | slot 8 |

The 8 sampling probabilities are compiled from λ and F0. They belong to the proposal and were never perturbed. Torch and Dr.Jit share this ownership exactly. They differ only in precision (float64 vs float32 compiled state) and summation order. The Dr.Jit native evaluation equals the term sum in native order bitwise.

**Interventions.** There are 16, each a function of the exact terms ([definitions](../../experiments/ecro_wl84_104h/comp104h.py)):

| Level | Interventions | Validity |
|---|---|---|
| Group | LAMBERT_OFF, OUTER_OFF, INNER_OFF, ATTEN_OFF (S = 1), ETA_OFF (η = 1: no refracted mapping, no 1/η²) | VALID_FAMILY (λ = 0, knots = 1 or η = 1 inside the certified family) |
| Factor | ATTEN_IN_OFF (s(μ_i) → 1), ATTEN_OUT_OFF (s(μ_o) → 1) | DIAGNOSTIC_ONLY (non-reciprocal) |
| Joint | ETA_ATTEN_OFF | VALID_FAMILY |
| Lobe | O0–O2_OFF, I0–I3_OFF | VALID_FAMILY |

Each VALID_FAMILY intervention was cross-checked against the native evaluator with the equivalently modified 69-float state. Every albedo-matched variant is DIAGNOSTIC_ONLY. Two interaction diagnostics are reported:
- the attenuation in × out factorization, (1 − s_i)(1 − s_o)(L + ΣI);
- the η × attenuation coupling of the inner group, (1 − S)(ΣU − ΣI).

Additive removal interactions are algebraically zero and are not reported as findings.

**Response phase.** It used all 9 WL104D panel materials × 20 probes × 1,024 lanes of the WL104D Amendment 1 independent layered reference, which has two independent walk halves and a 257-direction peak grid.
- **Metric.** Per-probe inner products normalised by the WL82 el1 albedo normaliser.
- **Noise-unbiased residual energy.** ⟨a − F, b − F⟩.
- **Uncertainty.** A 16-block lane jackknife.
- **Albedo-matched variants.** Scaled per probe and channel to R0's directional albedo, and qualified only if a_X ≥ 0.01 a_R0 in every channel.

**Parameter phase.** It used exact float64 forward-mode Jacobians at the R0 raw state. No optimizer and no step were used. It reports:
- group column-space projections of the cross-half residual;
- singular spectra;
- principal angles between groups;
- integrated gradients for C − A.

**Scene phase.** The three precision-qualified scenes were replayed from the archived WL104G evidence with the same seeds and the same 512-spp launch layout. Each launch was processed in sub-chunks through PCG32 `initseq` offsets, so every lane reproduced its archived random stream. All term images were accumulated per event.
- **E1.** Every term was evaluated at the fixed point-light direction.
- **E2.** The native R0 proposal was kept at the first continuation. Every term was divided by the same actual R0 pdf and multiplied by the fixed R0 terminal response.
- **Unchanged.** Non-ER first hits and archived R0 HIGH.
- **C − A.** WL104D A and C were evaluated at the same directions.

**Untouched contracts:** the frozen states (64 raw / 69 compiled), the response convention, A / C / ER_DF and other baselines, proposal PDFs, geometry, cameras, lights, masks, references, event definitions, historical worklogs and thresholds.

## Measurement

### A. Functional decomposition and contracts

Results of the frozen run at `fbdef22` ([tests](../../evaluation/104H/tests.json)):

| Check | Result |
|---|---:|
| Torch float64 term sum vs native (R0 / A / C, 9 materials, 4,000 directions incl. grazing and near-mirror) | ≤ 2.0e-16 of peak |
| Dr.Jit native vs term sum in native order | 0 (bitwise) |
| Dr.Jit native vs float64 sum of float32 terms (response lanes) | ≤ 1.4e-7 |
| Torch vs Dr.Jit per term | ≤ 4.0e-5 of term peak (limit 1e-4) |
| VALID_FAMILY intervention vs native eval of the modified state | ≤ 1.6e-7 |
| ETA_OFF vs closure82 arm E | 1.5e-17 |
| Interaction algebra (both) | ≤ 2.8e-13 |
| Reciprocity of VALID_FAMILY / non-reciprocity of DIAGNOSTIC_ONLY | ≤ 1.7e-14 / ≥ 6.8e-5 |
| Invalid directions; grazing (cos ≥ 3e-7) identity, non-negativity | exactly 0; 9.4e-8, all terms ≥ 0 |
| η → 1 limit, I_k vs U_k | 3.1e-7 |
| Jacobian vs central differences | ≤ 1.3e-8 |
| Exact C − A decomposition closure | ≤ 2.5e-14 (response), ≤ 1.9e-17 (scenes) |

Scene identities:
- **Replay.** The replayed term images summed to the archived R0 E1 and E2 within 1.9e-7 relative L1 in every replicate of all three scenes.
- **A / C.** The A and C sums equalled archived native A / C E1 within 1.9e-7 on the development scenes.
- **Lane level.** Native-minus-terms was exactly 0 at E1 and E2. w · pdf − eval was ≤ 1.1e-7 of peak, and the sampled pdf equalled the evaluated pdf exactly.
- **C − A at the common R0 proposal.** The fixed-R0-proposal C − A E2 change projects on the archived native CC − AA change with share 0.9997 (iid_1) / 1.0005 (hard_1) and cos 1.000.

**Reference precision.** The WL104D A1 rule passed for all 9 materials. The noise-unbiased residual energy is:

| Cohort | Residual energy | Reference noise energy | Ratio | Ratio per band |
|---|---:|---:|---:|---:|
| Development | 1.52e-3 | 1.9e-6 | ≈ 800× | ≥ 66× |
| Held out | 3.11e-3 | 2.8e-6 | ≈ 1,100× | ≥ 66× |

### B. Component contributions (R0)

Share of R0 directional luma albedo, all probes ([figure](../../evaluation/104H/term_albedo_shares.png), [peak shares](../../evaluation/104H/term_shares_at_peak.png)):

| Cohort | O0 | O1 | O2 | S·L | S·I0 | S·I1 | S·I2 | S·I3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Development | .045 | .099 | .244 | .034 | .119 | .366 | .090 | .003 |
| Held out | .085 | .065 | .321 | .133 | .062 | .219 | .102 | .013 |

The outer group's share rises toward grazing: O2 is 0.43 / 0.52 at 80–86°. The inner group dominates near normal (S·I1 0.45 / 0.31).

### C. Response-space hierarchical ablation

Pooled over all probes and lanes ([heat maps](../../evaluation/104H/response_rho_all.png), [core](../../evaluation/104H/response_rho_core.png), [shoulder](../../evaluation/104H/response_rho_shoulder.png), [tail](../../evaluation/104H/response_rho_tail.png), [summary](../../evaluation/104H/summary.json)). In the tables:
- ρ is the alignment of ΔF_X with the residual r = ref − R0. A negative ρ means the reference exceeds R0 along that term's shape.
- "Rel. change" is |ΔF_X| / |F_R0|.
- "Shape fraction" is |matched change| / |raw change|.

| Intervention | Dev ρ [95%] | Held ρ [95%] | Rel. change dev / held | Shape fraction dev / held |
|---|---:|---:|---:|---:|
| LAMBERT_OFF | −0.009 [−0.010, −0.007] | −0.010 [−0.011, −0.010] | 0.017 / 0.049 | 5.3 / 6.9 |
| OUTER_OFF | −0.219 [−0.241, −0.196] | −0.222 [−0.243, −0.200] | 0.94 / 0.96 | 0.98 / 0.99 |
| INNER_OFF | +0.071 [+0.060, +0.082] | +0.036 [+0.033, +0.039] | 0.24 / 0.17 | 1.37 / 1.05 |
| ATTEN_OFF | +0.026 [+0.013, +0.040] | −0.017 [−0.021, −0.013] | 0.018 / 0.043 | 4.6 / 1.7 |
| ETA_OFF | +0.071 [+0.043, +0.099] | −0.013 [−0.027, +0.001] | 1.21 / 0.74 | 0.70 / 0.64 |
| O0_OFF / O1_OFF / O2_OFF | −0.23 / −0.24 / −0.11 | −0.21 / −0.05 / −0.17 | 0.34, 0.25, 0.62 / 0.43, 0.18, 0.68 | 0.49, 0.69, 0.58 / 0.56, 0.81, 0.81 |
| I0_OFF / I1_OFF / I2_OFF / I3_OFF | +0.05 / +0.08 / −0.04 / −0.04 | +0.01 / +0.03 / +0.04 / −0.04 | 0.14, 0.16, 0.03, 0.01 / 0.08, 0.10, 0.09, 0.006 | — |

- **Pooled level.** No intervention reaches |ρ| > 0.24. Every full-strength removal increases the pooled residual energy (error change −0.002 to −184 times the residual energy).
- **Labels over the 720 response cells** (2 cohorts × 6 bands × 4 regions × 15 interventions):

  | Variant | DETECTABLE | NO_DETECTABLE | CONDITIONAL_ERROR_LOCALIZATION | CONFOUNDED |
  |---|---:|---:|---:|---:|
  | Raw | 595 | 106 | 19 | 0 |
  | Albedo-matched (DIAGNOSTIC_ONLY) | 585 | 83 | 52 | 0 |

  All 180 albedo normalisations qualified.

**Outer-group sign pattern.** OUTER_OFF ρ in the mirror core (< 3°), with labels:

| Cohort | 5–20° | 35–50° | 55–65° | 70° | 80–86° |
|---|---:|---:|---:|---:|---:|
| Development | +0.93 (D) | −0.09 | −0.74 (L) | −0.74 (L) | +0.01 |
| Held out | +0.91 (D) | −0.42 | −0.94 (D) | −0.94 (L) | +0.04 |

The shoulder (3–12°) shows the same sign change: +0.74 / +0.83 near normal, −0.52 / −0.77 at 70°. The tail stays within ±0.2.

**Response cells with any CONDITIONAL_ERROR_LOCALIZATION.** There are 8 such cells:

| Cell | Interventions with L (ρ) | Families |
|---|---|---|
| dev, 35–50°, core | ETA_OFF (−0.53), ETA_ATTEN_OFF (−0.53) | one |
| dev, 70°, all lanes | OUTER_OFF (−0.64), O1 (−0.59) | one |
| dev, 80–86°, shoulder | ETA_OFF (+0.60), ETA_ATTEN_OFF (+0.62) | one |
| held, 5–20°, core | O2 (+0.73) | one |
| held, 70°, core | OUTER_OFF (−0.94), O0 (−0.68), O2 (−0.87) | one |
| held, 70°, shoulder | OUTER_OFF (−0.77), O2 (−0.67) | one |
| dev, 55–65°, core | OUTER_OFF (−0.74), O2 (−0.53), I2 (−0.59) | two: outer + inner |
| dev, 70°, core | OUTER_OFF (−0.75), O1 (−0.68), O2 (−0.71), I2 (−0.60) | two: outer + inner |

In the two-family cells, the frozen |cos| ≥ 0.9 confounding rule did not fire, because the cosines were below 0.9.

**Interaction diagnostics.**
- **Attenuation in × out.** The interaction is ≤ 0.7% of |F| in response space.
- **η × attenuation coupling of the inner group.** It is 0.25 (dev) and 0.41 (held) of |F|, and 0.29 / 0.43 at grazing.

### D. Parameter-level sensitivity (local, no optimisation)

**Residual fraction explainable to first order.** R²_all, cross-half, all probes, 95% jackknife ([figure](../../evaluation/104H/parameter_r2.png), [params.json](../../evaluation/104H/params.json)):

| Material | iid_1 | iid_0 | hard_1 | iid_191 | iid_159 | block_161 | block_186 | hard_76 | hard_53 |
|---|---|---|---|---|---|---|---|---|---|
| R²_all | 0.483 [.472, .493] | 0.926 | 0.971 | 0.721 | 0.756 | 0.652 | 0.453 | 0.860 | 0.993 |
| unique amplitude | 0.098 | 0.199 | 0.401 | 0.094 | 0.520 | 0.063 | 0.103 | 0.409 | 0.239 |

**Unique shares of the other groups.** These are roughness, anisotropy, Fresnel, attenuation and η.
- They are ≤ 0.07 in every material except iid_159.
- In iid_159 the unique shares are Fresnel 0.445 and attenuation 0.337, while amplitude alone has R² 0.144. The groups are complementary there.

**Principal cosines between groups** ([coupling figure](../../evaluation/104H/parameter_coupling.png)):
- amplitude–Fresnel and amplitude–attenuation: 1.000 in 9/9 materials;
- amplitude–roughness: ≥ 0.993 in 9/9;
- amplitude–anisotropy: ≥ 0.99 in 6/9.

**Singular spectra and null directions.**
- There are 11–26 numerically null directions per material (σ / σ_max < 1e-6) and 12–24 weak ones (< 1e-3).
- The null directions are mostly amplitude / Fresnel mixtures. Some are roughness / anisotropy.
- The weak directions are dominated by attenuation knots and amplitude.
- The central-difference check passed at ≤ 1.3e-8.

**Integrated gradients, C − A.** Completeness is ≤ 2e-12. The group shares cancel strongly:
- iid_1: amplitude −1.37, Fresnel +2.24;
- block_186: amplitude −12.5, roughness +7.8, Fresnel +5.95.

**Post hoc, after evidence (descriptive, no rule)** ([coupling](../../evaluation/104H/posthoc_coupling.json), [term collinearity](../../evaluation/104H/posthoc_term_collinearity.json)):
- **Exact structural gauge.** Scaling every attenuation knot of a channel by (1 + ε) changes F identically to scaling that channel's Lambert and inner λ by (1 + ε)², with the absorption slot compensating. The exact-JVP relative difference is 0.0 for all 9 materials × 3 channels.
- **Structural nulls.** Six exact null directions per material are structural: 3 softmax shifts (σ ≈ 1e-17) and 3 attenuation-scale gauges (σ ≈ 1e-19–1e-21, loadings about 0.9 attenuation / 0.1 amplitude).
- **Knot saturation.** No knot is saturated. The largest knots sit near 0.9999, with sigmoid slope about 5e-5.
- **Amplitude–F0 coupling within a lobe.** It reaches ≥ 0.99 in 4 lobes: iid_1 O1 0.9997, iid_191 O1 0.9994, iid_191 I1 0.993, iid_159 I0 0.995.
- **Near-duplicate lobes.** These are term pairs whose responses have median per-probe cosine ≥ 0.95:

  | Material | Near-duplicate pairs (cosine) |
  |---|---|
  | iid_1 | O1~O2 0.976, I0~I1 0.999 |
  | block_186 | I0~I1 0.995, I0~I2 0.992, I1~I2 0.978 |
  | hard_76 | I0~I1 0.996 |
  | iid_191 | O0~O1 0.962 |

  For example, the fitted iid_1 outer α are 0.0555–0.0596 and its inner α 0.457–0.487.

### E. The WL104D C − A change (separate from the absolute error)

This is an exact decomposition: the shares sum to 1 ([figure](../../evaluation/104H/ca_decomposition.png)).

| Where | O0 | O1 | O2 | Outer sum | Attenuation | Lambert content | Inner content (largest) |
|---|---:|---:|---:|---:|---:|---:|---|
| Response, development (all) | .27 | −1.61 | +2.35 | 1.01 | .00 | −.004 | ≤ .012 |
| Response, held out (all) | .07 | −.50 | −1.46 | −1.89 | −.07 | .00 | I3 +2.94 |
| Scene iid_1 E1 (objects) | .21 | −1.67 | +2.46 | 1.00 | −.002 | .02 | ≤ .03 |
| Scene iid_1 E2 (objects) | .27 | −2.29 | +3.10 | 1.08 | −.007 | −.07 | ≤ .05 |
| Scene hard_1 E1 | .00 | +1.62 | .00 | 1.63 | −.53 | .00 | I1 −.56, I0 +.35 |
| Scene iid_159 E1 | .01 | −.01 | .00 | .01 | +1.14 | +.10 | I0 −6.54, I1 +6.49 |

On iid_1, the OUTER_OFF change is nearly antiparallel to C − A: cos −0.925 at E1 and −0.967 at E2.

### F. Scene-conditioned attribution

Objects ROI, 12 replicates, delete-one jackknife ([ρ map](../../evaluation/104H/scene_rho.png); group and lobe maps for [iid_1](../../evaluation/104H/scene_groups_test83_iid_1_L2.png) / [lobes](../../evaluation/104H/scene_lobes_test83_iid_1_L2.png), [hard_1](../../evaluation/104H/scene_groups_test83_hard_1_L1.png) / [lobes](../../evaluation/104H/scene_lobes_test83_hard_1_L1.png), [iid_159](../../evaluation/104H/scene_groups_test83_iid_159_L2.png) / [lobes](../../evaluation/104H/scene_lobes_test83_iid_159_L2.png)). L = CONDITIONAL_ERROR_LOCALIZATION, C = CONFOUNDED_OR_NON_IDENTIFIABLE.

| Scene, event | Residual RMS (of ref TOTAL) | L / C interventions (ρ) |
|---|---:|---|
| iid_1 E1 (dev) | 0.0366 | L: LAMBERT_OFF −0.55, OUTER_OFF +0.57, O1 +0.55, O2 +0.56 |
| iid_1 E2 (dev) | 0.0113 | L: O0 −0.63, I0 −0.60, I1 −0.61, I3 −0.66 |
| hard_1 E1 (dev) | 0.0297 | C: OUTER −0.75, INNER −0.76, ATTEN +0.75, ATTEN_IN +0.72, ATTEN_OUT +0.73, O2 −0.66, I1 −0.69, I2 −0.67 |
| hard_1 E2 (dev) | 0.0029 | none |
| iid_159 E1 (held out) | 0.0609 | C: INNER −0.90, ATTEN +0.74, ATTEN_IN +0.79, O2 −0.84; L: I0 −0.90, OUTER −0.68, O1 −0.67, ATTEN_OUT +0.68, ETA −0.53 |
| iid_159 E2 (held out) | 0.0056 | L: ETA −0.67, ETA_ATTEN −0.67 |

Effect sizes in iid_1 E2:
- **Negligible energy despite a strong label.** I0_OFF and I1_OFF have error change ≈ 0 and energy change ≈ 0. Their L label reflects spatial alignment of a negligible-energy term.
- **Single ablations that lower a scene ROI's error.** Some removals reduce one ROI's error. This is reported as sensitivity, not as a fault:

  | Intervention | Scene, event | RMS before → after |
  |---|---|---|
  | O2_OFF | iid_1 E1 | 0.0366 → 0.0308 |
  | ATTEN_OFF | iid_1 E2 | 0.0113 → 0.0097 |
  | ATTEN_OFF | hard_1 E1 | 0.0297 → 0.0250 |
  | ATTEN_IN_OFF (DIAGNOSTIC_ONLY) | iid_159 E1 | 0.0609 → 0.0466 |

  The valid ATTEN_OFF raises the iid_159 E1 error to 0.1128.

**Label counts over 945 scene cells** (3 scenes × 7 ROIs × 3 events × 15 interventions):

| DETECTABLE | CONFOUNDED | CONDITIONAL_ERROR_LOCALIZATION | NO_DETECTABLE |
|---:|---:|---:|---:|
| 621 | 203 | 112 | 9 |

Of the 45 scene × ROI × event cells with any L or C, 38 contain interventions from two or more families. The 7 single-family cells are:
- iid_1 sphere E1 and E1+E2: the outer group, +0.87 to +0.90;
- iid_159 objects / cube / highlight-core E2: the η mapping;
- iid_159 highlight ring E1+E2: the inner group / I0;
- iid_159 grazing E1: I0.

**Response–scene agreement.** Spearman correlation over the 15 interventions, between the material's all-probe response ρ and the objects-ROI scene ρ:

| Scene | E1 | E2 |
|---|---:|---:|
| iid_1 | −0.29 | +0.66 |
| hard_1 | −0.05 | −0.44 |
| iid_159 | −0.69 | +0.59 |

## Observation

- **Response space.** The residual has a sign-changing angular structure that is aligned with the outer group's lobe shape. Near normal, the reference is lower than R0 in the mirror core where the outer lobes sit. At 55–70°, it is higher. Pooled over all angles, this structure cancels and no term aligns with the residual beyond |ρ| = 0.24.
- **η mapping.** It aligns at development mid-angle cores and grazing shoulders, with opposite signs.
- **Interaction size.** The η × attenuation coupling of the inner group is large (25–41% of |F|). The in × out attenuation factorization is small (≤ 0.7%).
- **Scene co-location.** Positive response terms co-locate spatially on the objects, so scene alignments are broad. hard_1 E1 and iid_159 E1 show one shared direction: inner-group removal, outer-group removal and attenuation neutralization have |ρ| 0.72–0.90 and mutual |cos| ≥ 0.9.
- **C − A on iid_1.** The change is an outer-group change. O1 and O2, whose responses are nearly collinear at each probe (median per-probe cosine 0.976), carry large opposite shares that sum to the group's 1.00 / 1.08.
- **Response vs scene.** The interventions' rankings in the two spaces do not agree consistently: Spearman correlations range from −0.69 to +0.66.

## Interpretation

1. **Component contribution.** Every group and every lobe has a detectable effect, defined as a change ≥ 3σ of reference noise with a resolved error change. The exceptions at the pooled response level are ATTEN_OUT_OFF (development) and I3 (held out). Contribution size does not track error alignment. Lambert carries 3–13% of luma albedo, and its pooled alignment is −0.01.
2. **Error sensitivity.** No full-strength intervention reduces the pooled response residual. Some ablations reduce individual scene-ROI errors (section F). These are sensitivity results under fixed conditions, not evidence that the removed term is faulty.
3. **Conditional attribution within the frozen decomposition.**
   - *C − A change (separate diagnostic).* It is attributable at group level to the outer GGX group, on the iid_1 scene (E1 1.00, E2 1.08) and in development response space (1.01). The common R0 proposal reproduces the native CC − AA change exactly (share 0.9997), consistent with WL104F. The lobe-level split within the group is not identifiable, because O1 and O2 are near-duplicates. On held-out materials the C − A change is carried by compensating outer and inner shares (I3 +2.94 vs O2 −1.46).
   - *Absolute R0 residual, response space.* CONDITIONAL_ERROR_LOCALIZATION holds only in stratum-specific cells. The single-family cells are the outer group at 70° (and near-normal core O2, held out) and the η mapping in two development strata. At 55–65° and 70° development cores, the outer group and inner lobe I2 localize together. This is an ambiguity that the frozen cosine rule does not flag.
   - *Absolute residual, scenes.* Most localizing cells are multi-family, and inner versus attenuation is confounded (|cos| ≥ 0.9) on hard_1 and iid_159 E1. Single-family scene localizations are the outer group on the iid_1 sphere at E1 and the η mapping at iid_159 E2.
   - *Parameters.* 45–99% of the noise-unbiased residual energy lies in the local 64-dimensional tangent space. The unique group shares are mostly small, and the amplitude span overlaps every other group (principal cosine ≥ 0.99). No single parameter group carries the residual uniquely.
4. **Unique architectural root cause.** None is identifiable from this evidence. Five reasons:
   - (a) The family has exact gauge directions. The softmax shift and the attenuation-scale ≡ Lambert+inner amplitude equivalence make the amplitude, attenuation and absorption parts of any attribution non-unique by construction.
   - (b) The fitted states contain near-duplicate lobes. Lobe-level ablations within such pairs measure an arbitrary split of a shared shape.
   - (c) Inner-group and attenuation changes are indistinguishable in two scenes.
   - (d) Response-space and scene-space rankings disagree.
   - (e) Several families satisfy the localization rule in the same cells.

   Evidence against specific explanations:
   - The Lambert term and the attenuation in × out factorization do not explain the pooled angular residual: alignment ≤ 0.03 and change ≤ 5%.
   - A pure proposal mechanism for the C − A E2 change is again excluded.

## Unresolved Question

- **Lobe-level attribution within near-duplicate pairs.** It was not tested under a gauge-fixed or symmetry-reduced parameterization, which would itself change the analysis frame.
- **Local vs global.** The parameter analysis is first-order at R0. Nonlinear reachability, the family's best achievable error and a capacity bound remain undetermined.
- **Scene-space parameter sensitivities.** They were not measured. Scenes received component ablations only.
- **Albedo-matched counterfactuals.** They were evaluated in response space only.
- **Event scope.** E2 counterfactuals act at the first continuation with the terminal and HIGH fixed. N2 and higher events were not intervened.
- **Response–scene disagreement.** Which lane directions dominate each scene ROI was not isolated.
- **55–65° training gap.** R0 lacks training support in the 55–65° band (WL104C). Band-stratum results are conditional on that gap; the reference qualified there.
- **Coverage.** Three scenes and nine materials of one coated-conductor family.

## Architecture Verdict

**Diagnose only. No adoption, modification, replacement or abandonment decision.**

Was the internal error localized?
- **Partially, and only conditionally.** The WL104D C − A regression is attributable to the outer GGX group as a whole. The absolute R0 residual aligns with the outer group in stratum-specific cells, with a sign change from near-normal (excess) to 55–70° (deficit) in the mirror core. It aligns with the η mapping in a few cells.
- **No unique component, lobe or parameter-group root cause is identifiable.** The overall causal finding is CONFOUNDED_OR_NON_IDENTIFIABLE for a unique attribution, and DETECTABLE_COMPONENT_EFFECT for every component.

No certified-ER architecture contract changed. [Machine-readable decision lists](../../evaluation/104H/decision.json).

## Reproducibility

- **Commits.** Branch `ecro_lr3_adapter`. Parent `bf98012`. Freeze `fbdef22` (protocol, instrumentation, tests). The report commit adds `package104h.py`, `posthoc104h_coupling.py`, `evaluation/104H/`, this worklog and the living-document note.
- **Protocol hash.** The executed working-tree protocol is sha256 `0a37b053…bb7820c`, CRLF bytes after an in-place edit on Windows. The committed blob is `713195cc…dead746` (LF). `git diff` shows identical content. Output JSONs record the executed bytes.
- **Environment.** Local RTX 5080 16,303 MiB; `external/8dna26/.venv-cu128`: Python 3.11.9, Torch 2.8.0+cu128, Mitsuba 3.5.1 `cuda_ad_rgb`, Dr.Jit 0.4.6, NumPy 1.26.4.
- **Wall times.** Run 2026-11-20 16:17–16:24 KST.

  | Stage | Wall time |
  |---|---|
  | Tests | 5.6 s |
  | Scene renders (three processes in parallel; GPU about 85%, 14.1 GB) | 70.8 s / 40.4 s / 190.1 s (held-out figure summed over its two parallel halves of about 95 s each) |
  | Response | 3.9 s |
  | Parameters | 260 s |
  | Scene analyses | about 4 s each |

- **Budgets and seeds.**
  - Response: all 1,024 lanes × 20 probes × 9 materials, 16 lane blocks.
  - Scenes: 12 replicates at 1,024 spp (development, seeds 1044900 / 1046900 + r) and at 4,096 spp (held out, 1048900 + r); 512 spp per archived launch in 64-spp sub-chunks.
  - Tests: seed 104801.
- **Inputs.** Authenticated against `evaluation/104G/manifest.json`: the amended reference, compiled / raw states, WL83 direct fits, materials and the WL104G / WL104E scene packets and masks. Hashes are in [manifest](../../evaluation/104H/manifest.json).
- **Raw outputs (gitignored).** `results/ecro_wl84_104h/primary/`, holding `tests.json`, `response/`, `params/`, `scenes/<scene>/r###.npz`, term images, `analysis.json`, `maps.npz` and `logs/`; and `results/ecro_wl84_104h/posthoc/`.
- **Display.** Shared absolute scales per unit system:
  - response maps: ±0.050 of the probe albedo normaliser, signed square root;
  - scene maps: ±76.5 luma radiance, signed fourth root.

```bash
cd experiments/ecro_wl84_104h
export PYTHONUTF8=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=../../external/8dna26/.venv-cu128/Scripts/python.exe
O=../../results/ecro_wl84_104h_reproduction/primary   # the packager reads post hoc files from $O/../posthoc
mkdir -p $O ../../results/ecro_wl84_104h_reproduction/posthoc
$PY -B tests104h.py $O/tests.json
$PY -B scenes104h.py render test83_iid_1_L2 $O/scenes/test83_iid_1_L2
$PY -B scenes104h.py render test83_hard_1_L1 $O/scenes/test83_hard_1_L1
$PY -B scenes104h.py render test83_iid_159_L2 $O/scenes/test83_iid_159_L2 --reps 0:6
$PY -B scenes104h.py render test83_iid_159_L2 $O/scenes/test83_iid_159_L2 --reps 6:12
$PY -B response104h.py $O/response
$PY -B params104h.py $O/params
for s in test83_iid_1_L2 test83_hard_1_L1 test83_iid_159_L2; do $PY -B scenes104h.py analyze $s $O/scenes/$s; done
$PY -B posthoc104h_coupling.py $O/../posthoc/coupling.json          # post hoc
$PY -B posthoc104h_coupling.py terms $O/response/lanes.npz $O/../posthoc/term_collinearity.json
$PY -B package104h.py $O ../../evaluation/104H_reproduction
```
