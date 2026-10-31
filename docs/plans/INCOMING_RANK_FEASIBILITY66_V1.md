# Incoming Rank Feasibility v1 — WL59 P shared 16→8 incoming angular compression

User direction dated 2026-11-01. This is a bounded architectural feasibility experiment with chronological record in Worklog66. It uses isolated physical probe generation and representation-space analysis only. Nothing below is changed or adopted: the production candidate, the frozen checkpoints, the mixture count/rank, the loss, the thresholds, K/C ownership, F/S caching and coupled GI. There is no successor.

**Central question.** Under qualified, independently tagged collided first-exit references for spatially localized directional incidence, does the rank-8 shared incoming angular interface of the frozen WL59 P model necessarily discard physically relevant collided-transport information? If it does not, can an inadequate *learned* incoming basis be distinguished from other unverified K limitations?

## Implemented algebra (not the conceptual FNTO equations)

`r57_model.Transport.apply` with the WL59 runtime state:

```text
q[i,r,c] = Σ_a A[r,a,c] (1-β[i,a]) x[i,a,c]          A = softmax(in_logits, dim 0): 8×16 per channel, global
y[j,r,c] = Σ_{e:dst=j} W[e,r,c] q[src_e,r,c] + Σ_k U[j,k,r,c] Σ_i V[i,k,r,c] q[i,r,c]
scatter[j,o,c] = Σ_r O[r,o,c] y[j,r,c]               O = softmax(out_logits, dim 1): 8×64 per channel, global
total = scatter + B x                                 β, B: geometry-only zero-collision quadrature (q=256)
```

Fix a geometry, a source port i and a channel. The scattered response to unit flux in incoming bin a is then row a of `D_i Aᵀ G_i`, with `D_i = diag(1-β_i)` and `G_i[r,(j,o)] = T_r[j,i] O[r,o]`. In normalised coordinates `Z_i = D_i⁻¹ R_i`, the model is exactly `Ẑ_i = Aᵀ G_i`.

Softmax over mixtures makes every column of A sum to one. So `Aᵀ1₈ = 1₁₆`, and **1 lies in range(Aᵀ) for every parameter value**. With `H = I − 11ᵀ/16` and `C_i = H Z_i`, every architecture member has an incoming column space `span(1) ⊕ T`, where T is at most 7-dimensional, lies in 1⊥, and is shared by all ports and geometries of a channel. The span(1) component is free, so the absolute and the contrast Z-errors share the numerator `‖(I−P_T)C_i‖²`. Raw errors equal the Z errors with each row scaled by `(1-β_ia)²`. All selected ports have `1−β ≥ 0.9899`.

Any fixed output aggregation (fine16, 1 cm/16-out) is linear and applied after the incoming map, so the bounds hold in every space. They are bounds for the Frobenius norm with equal weight per unit-flux probe. They are not claims about other norms or source populations.

## Nested floors (contrast metric; absolute metric reported)

| Floor | Relaxes | Estimator |
|---|---|---|
| Relaxed shared rank-8 (PRIMARY) | Everything except the shared, 1-containing 8-dim incoming space (one T per channel across the 8 port/geometry matrices) | `tail₇` of the pooled unbiased cross-Gram, which is low-biased (concavity). The upper estimate is the held-out residual of a subspace fitted on the other replica half, which is high-biased. |
| Per-port relaxed | As above, with one T per matrix | Same construction |
| Frozen A | Everything except the checkpoint's A | Fixed projector; unbiased |
| Frozen A + O | Signed spatial coefficients `t_rj` only | Per-destination block projection; unbiased |
| Frozen O / outgoing rank-8 | Secondary outgoing-side control | Fixed projector / bracket |
| Frozen model | — | Unbiased cross-replica error |

All cross terms are U-statistics over independent replicas, so they carry no noise bias. Uncertainty is a delete-one-replica jackknife with t(K−1). Packet moment variances are cross-checked against the across-replica scatter. Eigenvalues are called resolved only when they exceed 3 SE.

## Probes, budget and gates

- **Geometry and ports.** Rest and pt0. Four geometry-only ports, chosen by rule from rest material coordinates: top dent centre, top dent flank (centre + r/2), top far control, and bottom below the dent. Ties are broken by (distance at 1e-9 m, index). Each port must satisfy `1−β ≥ 0.5` and lie ≥1 cm from the sides. The selection is 2085 / 2441 / 1123 / 2069, recorded in `ports.json` with the frames, β rows and parents for both geometries.
- **Probes.** One probe per bin: unit flux in all channels, uniform on the port, uniform in projected solid angle inside the bin (the r57 `cell_source` semantics). That gives 128 probes. Each probe is traced with the unchanged first-exit importance estimator plus the WL63 observational tags.
- **Budget rule.** One 32M packet is timed per geometry. K = 8 replicas if the predicted walk time is ≤ 4 h, else K = 4, else stop. The numeric budget is committed before generation. The seeds are `680000000 + g·100000 + slot·10000 + bin·100 + replica`.
- **Stops:**
  - any packet fails the physical, one-hot, unit-flux or 1e-12 tag-closure gates;
  - frozen replay exceeds 2e-5;
  - factorisation or span exceeds 1e-5;
  - a geometry feature mismatch;
  - precision fails: the primary per-matrix contrast signal S_C/SE < 10, or the pooled SE/S_C > 0.05.

## Decision (primary space: 1 cm / 16-out, pooled, per channel)

The rho ratios below are `√(energy / physical contrast energy)`, read at 95% jackknife bounds.

- **A**: relaxed lower bound ≥ 0.10.
- **Else, if the relaxed upper bound < 0.05:**
  - **B**: frozen-A lower bound ≥ 0.10 for both WL59 seeds.
  - **C**: for both seeds, the frozen-A upper bound energy ≤ 0.25 × the model contrast-error lower bound, and frozen-A upper < 0.10.
  - Otherwise nondecisive.
- **Overall.** A precision or identity stop gives **D**. Otherwise the precedence is: any channel A → A; any channel B → B; all channels C → C; otherwise D.
- **Descriptive attribution, not a verdict.** If the frozen A+O upper bound is ≤ 0.25 × the model lower bound, ≥ 75% of the probe model contrast error energy is attainable inside the frozen angular bases with free signed spatial coefficients.
- **Secondary, descriptive only.** WL64-B (contrast objective) and WL65-B (partial clean labels) checkpoints.
- **Historically inspected, labelled.** Pair7's bins (8, 13) in the per-bin-pair table.

## Scope

The verdict applies only to this probe population (homogeneous slab, 2 geometries, 4 ports, 16 bins) and these norms. A high model error on localized probes does not establish the cause of the original Pair7 error. A small relaxed floor does not show that the remaining error is attainable by the constrained, positive, passive learned family. No rank change, retraining or successor follows automatically.

- **Protocol:** `experiments/fnto_rank66/protocol/rank66_v1.json`.
- **Code:** `ports66.py`, `probes66.py`, `model66.py`, `estimators66.py`, `analyze66.py`, `export66.py`, `chain66.py`.
- **Tests:** `test66.py`.
- **Non-evidence smoke harness:** `smoke66.py`.
