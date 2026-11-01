# Constrained K Representability v1 — Track A (WL68)

User direction dated 2026-11-01. The chronological record is in Worklog68.

This is an existence/attainability probe on **experimental copies** of the unchanged WL59 P intrinsic K family. Nothing below is adopted or changed: the representation, its rank/width/parameter count (157,024), K/C ownership, F/S semantics, the production checkpoints and the original thresholds. No coupled GI is run and no successor is proposed. The diagnostic fitting objective is not a training recipe.

**Central question.** Can the actual positive/passive WL59 K family constructively realize qualified collided directional transport on controlled physical probes? And what evidence separates representation limits from fitting or generalization failure?

## Implemented family (from code, not prose)

For a source port i, incoming bin a and channel c, the response is:

```text
scatter[a,j,o,c] = d[i,a] Σ_r A[r,a,c] T[j,r,c] O[r,o,c]
d = 1-β_model                                    geometry-only q=256 zero-collision quadrature
A = softmax(in_logits, dim r)                    global; every incoming-bin column is a probability vector over 8 mixtures
O = softmax(out_logits, dim o)                   global; every mixture is a distribution over 64 outgoing bins
T[j,r,c] = Σ_{e:src=i,dst=j} W[e,r,c] + Σ_k U[j,k,r,c] V[i,k,r,c]
```

The spatial coefficients have this structure:

- **W** is `softplus(edge MLP) · prior_c(dist)` on the 128 nearest rest-coordinate neighbours of i. The prior decay lengths are 18.1, 5.8 and 1.8 mm per channel.
- **U** is a softmax over all destinations: rank 16 per mixture and channel, per geometry.
- **V** is a softplus. W and V are rescaled so that `Σ_j T = mass = sigmoid(·) ∈ (0,1)`.
- **Geometry conditioning.** All of W, U, V and mass come from 16 local node features through two-hop message passing and pair features. A and O do not depend on geometry.
- **Passivity** holds for every parameter value: `β + (1-β) Σ_r A·mass < 1`.

So each probe row is a convex combination of 8 nonnegative fields. Each field is separable in space and outgoing direction, its outgoing lobe is global, and its spatial mass is below 1.

The WL66 relaxations admit responses the family forbids:

| WL66 floor | What it permits that the family forbids |
|---|---|
| Relaxed rank-8 | Signed, unstructured responses on a free shared subspace |
| Frozen A | Arbitrary outgoing structure |
| Frozen A+O | Signed spatial coefficients |

None of the three includes the geometry-to-coefficient network.

On the probe population (4 sources per geometry, rank 16 ≥ 4), the closure of the network's response set lies in the **coefficient class FCoef**: global simplex A/O and per-source `T ≥ 0` with `Σ_j T < 1`. FCoef differs from the network only by geometry conditioning and parameter sharing.

## Design

| Element | Fixed choice |
|---|---|
| Fitting data | WL66 probes: rest/pt0 × 4 ports × 16 bins. Fit on replicas 0–3; score on replicas 4–7 only. These probes were previously inspected, so they are diagnostic data and never fresh confirmation. |
| Objective J | Per matrix and channel: relative squared error of the collided response, averaged over native/fine16/coarse, plus the coarse directional-contrast error (WL66 `H D⁻¹`) |
| Network fits | Parameter groups A, O, AO, net (all but A/O) and all. Inits: P_5901 and P_5902, plus one from-scratch init for the all group. Adam, 20,000 steps, cosine schedule 3e-3→3e-5, clip 1. |
| Coefficient class | FCoef with 4 restarts (two WL59 inits, two random). Adam, 20,000 steps, 1e-2→1e-4. |
| Convex floors | Frozen WL59 A,O with admissible `T ≥ 0, Σ ≤ 1` under J. Admissible counterparts of WL66's PAOc and F_A (coarse contrast). FISTA in float64. |
| Teacher controls | T1/T1r: the other WL59 seed. T2: P_5902 network with the angular bases of the best physical FCoef restart. Tcoef: random A/O with WL59 spatial coefficients. The student starts from a different member with the same optimizer. |
| Held-out | New ports by a geometry-only rule: slots 4–7, ≥1 cm from the fitting ports, `1−β ≥ 0.5` at all geometries. Held-out geometry cf3 is the WL59 diagnostic pose farthest from pt0. Groups: H1 (rest/pt0 new ports), H2 (cf3 fitting ports), H3 (cf3 new ports). Independent seeds on LabServer63. K fixed by a preflight budget rule. |
| Other evidence | Signed deformation (pt0−rest, cf3−rest); energy; passivity; the original WL59 extended-field metrics (historical fields); basis floors for every fitted A/O; a labelled held-out angular-transfer oracle |
| Identity gates | Input hashes; WL59 sources; geometry features bitwise; frozen replay ≤2e-5; factorisation ≤1e-5; WL66 statistics reproduced ≤1e-4; LabServer63-vs-69 reference identity on 4 WL66 packets with their original seeds |

## Decision

Quality uses the WL59 original thresholds on upper 95% bounds, in all channels:

- coarse collided relative L2 ≤ 0.10;
- fine16 ≤ 0.15;
- coarse contrast ≤ 0.25.

The **primary candidate** is the physical all-parameter fit with the lowest final training J.

The definitions below feed the outcome rule:

- **S_fit**: quality holds on fit-population rest and pt0, scored on replicas 4–7.
- **S_gen**: quality holds on H1-rest, H1-pt0, H2 and H3.
- **Optimizer adequate**: every informative net teacher control passes. A teacher control passes at contrast ≤ 0.10, coarse ≤ 0.04 and fine16 ≤ 0.06.
- **Converged**: the change in J over the last decile is below 1%.

| Outcome | Rule | Family status |
|---|---|---|
| A | S_fit and S_gen | Viable on the tested population |
| B | S_fit only | Needs a specific additional falsification: generalization |
| C | Not S_fit, with an adequate optimizer and convergence | Unsupported for the tested requirements; impossibility not established |
| D | Otherwise | Representation vs optimization/reference not separated |

The attribution across convex floors, parameter groups, FCoef versus network, Tcoef and the oracle is descriptive only.

**Protocol:** `experiments/fnto_repr68/protocol/repr68_v1.json`.

**Code:**

- `family68.py`: exact family, objective, FCoef, convex solver;
- `fit68.py`, `probes68.py`, `ports68.py`, `eval68.py`, `export68.py`, `chain68.py`;
- `extract68.py`: input staging on LabServer69;
- `test68.py`: tests.
