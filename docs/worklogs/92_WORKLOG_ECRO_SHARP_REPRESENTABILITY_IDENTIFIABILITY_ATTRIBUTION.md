# Worklog — ECRO Sharp Interreflection: Representability, Identifiability and Learning-Failure Attribution (2026-11-13)

## Origin and scope

- **Brief.** "ReST-GI — ECRO Learned Global Transport: Sharp Interreflection Representability, Identifiability and Learning Failure Attribution" (user, 2026-11-13). One bounded hypothesis-discrimination batch.
- **Context.** The human decision of 2026-11-13 restores ECRO as the central learned-global-transport development base. WL86's frozen negative verdict stands.
- **Question.** Does the existing ECRO operator have the representational and informational capacity to reproduce sharp directional interreflection, and, if so, does physically identifiable supervision improve learned generalisation without changing the architecture?
- **Primary hypothesis (prospective).** The sharp-response failure is caused at least partly by non-identifiable learned component responsibilities, not solely by an insufficient operator representation. Alternatives: H_rep, H_info, H_learn, H_id, H_interact.
- **Preserved.** WL84 ER; the WL86 code, checkpoints, data, references, masks, thresholds and verdicts; WL86A and WL87–WL91 (imported unchanged). No in-family arm changes the architecture, the ranks (R 16, global 24), the 1,270 × 64 table, the 512 state rays or the 16-point source encoding.
- **Not done.** No second operator family, no capacity change, no explicit residual, no GM2, no adoption.
- **Where.** Branch `Selective_Recompute`, LabServer63. `experiments/ecro_attr92/`, `results/ecro_attr92/`, `results/evaluation/92/`. Server `~/rest_x92`, reading `~/rest_ecro86`.
- **Terminology.** Explicit transport and path tracing are physical comparators and diagnostics. No external neural baseline is run; no Neural Full Recompute claim is made.

## ECRO v0 as recovered from the implementation

Source: `experiments/ecro86/model86.py`, `train86.py`, `gen86.py`, `pt86.py` (read, not reconstructed from worklogs).

**Operator.**
- `u = c·s`, `Bq = a·u`, `E1` = flight of `Bq` through the 512 state rays with weight cos dΩ / κ, `B = a·(c·E1) + a·(P·Σ_j area_j Q_j·u_j)`.
- `c`, `a` are per patch, per colour, 64 bins × 16 channels, non-negative. `Q`, `P` are per patch, rank 24, non-negative.
- The local matrix `M_n = a_n c_nᵀ` (incident bin → outgoing bin, non-negative rank ≤ 16) is used with **tied factors** in three places: source → once-scattered radiance `Bq`; first-flight irradiance `E1` → outgoing radiance of the single-flight term; encoder and decoder of the global term.

**Physical correspondence of the terms.**

| Model quantity | Path family it can represent | Path-traced counterpart |
|---|---|---|
| `Bq` | once-scattered outgoing radiance | L₁ (the table q) |
| single-flight term `a·(c·E1)` | exactly 2 scattering events, all senders | L₂ |
| global term | 3–5 scattering events | L₃ + L₄ + L₅ |
| `B` | the WL86 target | L₂ + … + L₅ |

- The mapping is exclusive: every path has exactly one event count, so the two additive terms have disjoint physical targets.
- The implementation does not enforce it. `train86.py` supervises only the sum `B` (`model86.loss_fn`). Nothing prevents the global term from carrying two-event light or the single-flight term from carrying less than L₂.

**Training contract.** 1,024 geometries × 4 point lights, 256-spp path-traced field targets; 24,000 steps of Adam, one geometry per step; loss = squared error relative to the target's patch mean over bins plus 5 % of the field mean; seeds 8601 / 8602.

## Design of this batch (`experiments/ecro_attr92/protocol/ecro_attr92.json`)

**Stages.**
- **S0 Baseline and provenance.** Hashes of checkpoints, data, references. The frozen checkpoints must reproduce the tables stored with the WL86 renders, and the re-rendered tables the WL86 E_H values.
- **S1 Truth by family.** Path-traced tables of exactly-2-event radiance (last sender on the panel / elsewhere) and 3–5-event radiance (WL86A tracer, unchanged), two halves of 2,048 spp. Reader weights W_R (Σ_pixels hl = Σ W_R·B).
- **S2 Sharp deficit.** Truth contribution, each learned term's contribution, residual, compensation sign, on E2 T1, E2 T0, E1 T1.
- **S3 Ladder of the 2-event term.** Tables = (variant of the 2-event term) + (path-traced L₃₊). Only the 2-event term changes.

| Variant | Meaning | What it isolates |
|---|---|---|
| ORA2 | path-traced L₂ | must equal the oracle table |
| RS | exact once-scattered radiance per deployed ray | the deployed rays without any bin projection (non-deployable for ECRO) |
| EXP | exact 64-bin q through the deployed rays | sender bin projection |
| ENCQ | q from the deployed encoding through M64 | source encoding |
| ENCB | the ECRO single-flight form with exact full-rank local matrices (R = 64) | everything the deployed interface allows, without the rank limit (non-deployable capacity control) |
| R16 | the same form with rank-16 non-negative factors of M64 | an in-family point with exclusive ownership |
| LRN | the frozen generators' single-flight term | what was learned |

  - Pooled variants (16 state realisations) are a non-deployable adequate-sampling control.
- **Local operator.** M64 = bin-averaged frozen-ER response per material. Singular spectrum, Eckart–Young rank-16 bound, rank-16 NMF, and the learned `a cᵀ` of the frozen generators, all in the flux-transfer norm.
- **S4 Fixed-geometry free-factor fits** (non-deployable). All four factor tables free on one geometry; 192 fit lights, 32 validation lights; objectives SUM (the WL86 objective) and OWN (SUM + component ownership); initialisations: frozen generators, untrained generators, the rank-16 construction; 8,000 steps; a convergence rule.
- **S5 Identifiability.** Across the SUM fits and the frozen generators: same objective value with different allocations?
- **Gate, then at most one intervention.**

**Gate (level = the frozen WL86 rule F2: E_H ≤ 0.06 in region C).**
- G_id: component targets valid (tests).
- G_info: ENCB + L₃₊ meets the level in both failing cases (E2 T1, E2 T0), deployed state draw.
- G_rep: the better of R16 + L₃₊ and the single-flight term of the converged OWN-k16 fit + L₃₊ meets the level in both failing cases.
- The intervention runs only if all three hold. Otherwise the intervention path stops; S0–S5 are the evidence.

**Conditional intervention.**
- Control: the WL86 contract (frozen checkpoints; a fresh rerun for the E4 comparison).
- Intervention: objective OWN with the path-traced 2-event target regenerated from the WL86 generator's own seeds and samples; f₃ = f − f₂. Same architecture, data, steps, optimiser, seeds.
- Evaluation: WL86 gates F1–F4 and the component attribution on E1 / E2 (inspected), a new condition E4 and held-out sources.
- E4: the WL87 candidates E3b / E3c, defined on 2026-11-11 and never rendered, chosen by a geometry-only rule that requires a sharp-reader class; admitted by the unchanged WL86 stage-0 criteria.

**Interpretation rules (prospective).**
- *Failure to fit*: a converged fit stays above a constructed in-family point.
- *Failure to optimise*: a converged fit ends above the objective value of a constructed in-family point (a better point exists and was not found).
- *Failure to generalise*: fixed-geometry factors reach a level the generator does not.
- *Information limit*: ENCB + L₃₊ misses the level although ORA2 meets it.
- A failed optimiser is not treated as a proof of insufficient capacity: the Eckart–Young bound and the constructed points carry the capacity statements.

## Prior evidence used as constraints (not re-measured unless stated)

- WL86: oracle table E_H 0.014 in E2 T1 region C; frozen ECRO 0.063 / 0.072.
- WL86A: the single flight from panel senders delivers 1–18 % of its family; the global term compensates; free-factor fit indeterminate (64 lights, over-fitting).
- WL87: explicit transport on ECRO's own state and encoding: EXP 0.049, ENC 0.054, RAYSEND 0.059, sender-side C3 0.024 in E2 T1 region C.
- WL90 / WL91: the first-gather angular projection of the once-scattered radiance is the explicit floor on the moving-sharp workload; exact first gather removes it in expectation.
- These bound what any operator on the deployed interface can reach. They do not by themselves explain the learned failure.

## Pre-freeze development (non-evidence)

- **Design refinement from the local-operator dev run.** The plain Frobenius norm of M64 is dominated by the grazing ring (f ∝ 1 / (4 cos cos)): a rank-16 fit in that norm spends every channel on 16 grazing incident bins. The analysis and the rank-16 construction therefore use the flux-transfer matrix X[bi, bo] = M64 z_bi z_bo, whose rows sum to the directional albedo. This concerns frozen-ER physics only and was fixed before the freeze.
- **Dev chain** (`~/rest_x92dev`; 64 spp, 6 fit lights, 1,000 fit steps, 48 training geometries, 300 training steps; the intervention branch forced so that every command runs): completed end to end, every stage rc = 0, tests pass. All dev numbers are noise-dominated and are not interpreted. The dev local-operator stage (deterministic frozen-ER quantities, re-measured in the evidence run) was seen.
- **Full-scale probes** (`probe92.py`, LabServer63, alone):
  - Component-target regeneration: the WL86A split tracer with the WL86 generator seeds reproduces `pt86.Ctx.field` to 1e-7 and the stored float16 training target to 1.7e-4 (two lightings of `train/g00000`). The planned T_F2_SUM identity test (≤ 1e-3) is therefore attainable; the dev failure came from the reduced dev sample count only.
  - Fixed-geometry fit at the full batch (16 lights, all four tables free, OWN objective): 19.7 ms per step, 680 MiB peak; 8,000 steps ≈ 2.6 min per fit, 21 fits in a pool of 4.
  - Field tracer: about 2.9 s per lighting at 256 spp; fit data ≈ 15 min per geometry (three in parallel).
  - Expected chain wall: about 1 h without the intervention, about 2 h with it.
