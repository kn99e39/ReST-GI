# Worklog — GM1 Intermediate Transport Representation Fidelity-Gap Attribution and Feasibility (2026-11-12)

## Origin and scope

- **Brief.** "ReST-GI — GM1 Intermediate Transport Representation Fidelity-Gap Attribution and Feasibility" (user, 2026-11-12). One bounded diagnostic batch.
- **Question.** Is the common sharp fidelity floor of the converged explicit relations (R0 / R1 / C3UR, W1 R_mirror E_H ≈ 0.038 against the oracle table's ≈ 0.014) caused primarily by intermediate spatial / angular projection? What evidence bears on keeping or reconsidering the present explicit representation?
- **Primary hypothesis (prospective).** The gap is information discarded when the sender radiance is projected onto the patch × 64-bin table before each gather, not relation sampling variance.
- **Preserved.**
  - WL84 ER and the WL86–WL89 code, protocols, results and verdicts (imported, never edited).
  - R0 = WL87 EXP; C3UR = WL89, kept as the explicit relation development baseline.
  - The WL89 workloads (W1 = V3, W2 = X2), references, masks, oracle tables and order fields.
  - WL89's H1 verdict is not reinterpreted.
- **Not done.** No learned operator, no new architecture, no GM2, no adoption. The diagnostics are not deployable representations.
- **Where.** Branch `Selective_Recompute`, LabServer63. `experiments/gm1_gap90/`, `results/gm1_gap90/`, `results/evaluation/90/`. Server `~/rest_gap90`, reading `~/rest_rel89`.

## A. What the code computes (verified against the sources)

| Path | Code | What it does |
|---|---|---|
| Table recurrence (every WL87–WL89 relation) | `xport87.recurrence`, `rel88.recurrence` | L₁ = q, L_{k+1} = G(P L_k), B = L₂ + … + L₅ |
| Projection P at the sender | implicit in the table read `L[hit, dbin]` (`xport87.XOp.gather`, `rel89.SendRelR.apply`) | the gather reads the hit patch's table entry: patch-constant in space, constant over a 0.098 sr direction bin. Applied before **every** gather, so it compounds through the orders |
| Source projection q | `quad87.qtable` | once-scattered radiance at 16 points per patch, bin-averaged by 1,024 MIS samples. Against the path-traced order-1 field: relative L1 0.013, flux 1.0007 (WL89). q is therefore an accurate patch × bin *projection* of the exact 1-event field |
| Gather G at the receiver | per-ray frozen-ER weights W (`quad87.Quad`), exact local direction, sphere rows per band | writes the receiver's own patch × bin entry |
| Camera readout | `pt86` mode `ecro` | LOW (e1, e2 path-traced) + B[patch(x₂), bin(x₂ → x₁)] |
| Oracle table | `pt86.Ctx.field` | receiver patch × bin average of the path-traced 2–5-event radiance; **no P at any sender** |

- Table and oracle carry the same event family (2–5 events at the receiver; no next-event at the probe vertex) and use the same readout.
- With relation sampling converged, the differences that remain are:
  - (i) P at the senders, at each of the four gathers;
  - (ii) the receiver quadrature (W, receiver point sampling);
  - (iii) the q estimator.
- WL89's attribution of the gap to "relation discretisation" was an inference from converged − oracle. It did not separate (i) from (ii) / (iii).

## B. Prior evidence that shaped the design (disclosed)

- WL89, already inspected: converged R0 / R1 / C3UR 0.038 against ORACLE 0.014.
- Deployed R0 with the exact first-gather radiance (repaired RAYSEND, RS0) was 0.023, against R0 0.060.
- Every diagnostic below is new and was fixed before being computed.

## C. Diagnostics (frozen in `protocol/gm1_gap90.json`)

**Exact-read operators.** They run on a pooled R0 relation: 64 fresh realisations, WL87 weights.
- For every state ray, the gather reads a path-traced k-event radiance instead of the table entry. The tracer is the pt86 estimator split by event count.
- The radiance is optionally projected at the read:

  X(a, s)[k] = G(P_{a,s} Lx_k), k = 1…4

- a = sub-bins per direction bin (0 = exact direction); s = sub-patches per patch (0 = exact point).
- 4 samples per read, averaged over the valid ones, as the table average.

| Operator | Meaning |
|---|---|
| X(0,0) = EE | no intermediate projection |
| X(1,0) | angular projection only (exact point, bin-averaged direction) |
| X(0,1) | spatial projection only (patch-averaged point, exact direction) |
| X(1,1) | the table's projection applied **once** to the exact field |
| M_m (m = 0…4) | first m gathers exact, the rest by the pooled table operator. M₀ = table recurrence; M₄ = EE |
| P1only, P2plus, EA1, SE1 | projection at order 1 only / at orders ≥ 2 only; angular / spatial projection at order 1 only |
| Fixed counterfactual grid (no sweep) | angular 4 / 16 sub-bins, spatial 4 / 16 sub-patches, joint (4,4), (16,16). Single-read projections: a lower bound of the loss of a refined table recurrence, since compounding is excluded |

**Rules** (primary: W1 T1, R_mirror; E_H with common random numbers across arms).

| Rule | Definition |
|---|---|
| Gap | G₀ = E_H(M₀) − E_H(ORACLE). It must exceed 2 × its noise, else stop |
| Isolation | E_H(EE) − E_H(ORACLE) ≤ 0.25 G₀ → ISOLATED; ≤ 0.5 G₀ → PARTIAL; else NOT ISOLATED (another shared approximation) |
| Factor | with J = X(1,1) − EE, A = X(1,0) − EE, S = X(0,1) − EE: ANGULAR if A ≥ ⅔ J and S ≤ ⅓ J; SPATIAL in the mirrored case; otherwise BOTH / INTERACTING |
| Order | m* = smallest m with M_m within 0.25 G₀ of the oracle; first-order share = (P1only − EE) / J; compounding = (M₀ − X(1,1)) / G₀ |
| Field level | first table order whose error exceeds 2 × truth noise on the sharp sender classes |
| Ladder | smallest grid point within 0.25 J of EE |

**Feasibility probe.** Interpreted only if isolation holds and the first gather carries the loss (m* = 1, or first-order share ≥ 0.5).
- R0X1 / C3URX1: the first gather reads the exact once-scattered radiance per relation ray. That is one deterministic next-event evaluation per ray per source; the relation is unchanged and source-independent. Later orders use the table operator.
- Evaluated on the deployed draw and the 16 WL89 realisations.
- X1 is MATERIAL if the paired C3URX1 − C3UR interval in R_mirror is < 0.
- X1 is ECONOMICALLY PLAUSIBLE if its measured per-edit cost is ≤ 1.25× and its source-only update ≤ 2× C3UR's. Costs are measured warm and alone, with compilation excluded.
- The probe is a diagnostic proof of feasibility of one mechanism. It is not a candidate and is not adopted.

**Tests** (gate; any failure stops):
- per-order tracer = pt86 estimator on camera rays;
- sub-bin / sub-patch samples stay inside the read's cell;
- pooled exact gather vs path-traced order fields (flux within 3 %);
- single-read table projection of the 1-event field = pooled table gather of q (flux within 2 %);
- X1 plumbing ≤ 1e-5;
- relation digests unchanged by the exact reads;
- replay ≤ 1e-5.

## Pre-freeze development (non-evidence; dev geometry Wk via `~/rest_rel89dev`)

- **Dev tests: all pass first run.**
  - Tracer vs pt86: 1e-7.
  - Exact gather flux per order: 1.004 / 1.003 / 1.001 / 0.994.
  - Projection-once vs table gather of q: 1.001.
  - Plumbing: 4e-9 / 0.
- **Scale probe of the pooled exact-read stage.** 3.3 s per realisation for the 10 variants (about 22 M probe paths), after a 20 s first-call compilation. At 64 realisations that is about 4 min per configuration; three configurations run concurrently.
- **Dev chain end to end.** See the evidence section for the freeze.
- **Dev chain end to end** (`WL90_DEV=1`, dev geometry Wk through the WL89 dev root, 64 spp, 4 pooled realisations). Every stage completed in 2 min: tests, order field, pooled exact-read tables, feasibility tables, renders, costs, analysis, visuals. Dev values are noise-dominated and are not evidence.
- **Implementation optimisation of the probe-ray buffers** (same operator; tests and plumbing re-verified).
  - Buffers are now built on the GPU from the resident relation segments (torch → Dr.Jit directly), and the per-ray radiance is returned as a torch tensor.
  - Buffer rebuild on the dev geometry: 70–90 ms → 1.1 ms (C3UR) / 7 ms (R0).
  - Exact once-scattered radiance per source: about 10 ms for about 0.6 M rays; the q table takes 25 ms.
