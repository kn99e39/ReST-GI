# Worklog66 — Incoming angular rank feasibility of the frozen WL59 P intrinsic K (2026-11-01)

## Authorisation and scope

The user's brief of 2026-11-01 authorised one bounded architectural feasibility experiment. It asks whether the frozen WL59 P model's shared 16-incoming-bin → 8-mixture angular compression has a physically significant information bottleneck under qualified collided first-exit references. The probes are spatially localized directional incidences.

It was motivated by three earlier results:

- WL63: the qualified Pair7 failure is dominated by learned collided transport.
- WL64: a fixed directional-contrast supervision intervention was rejected.
- WL65: a qualified partial clean-label A/B gave no meaningful held-out benefit.
- WL60 had bounded only the outgoing span, not the incoming compression under independent localized angular excitation.

The batch runs on branch `Selective_Recompute` from start `2731a93`. There is no new branch or worktree.

Allowed work:

- isolated probe generation;
- representation-space analysis;
- precision accounting;
- tests and reports.

Prohibited:

- any change to network, checkpoint, rank or mixture count, loss or thresholds;
- retraining;
- clean-label interventions;
- K/C ownership or F/S changes;
- coupled GI;
- any successor.

WL59–65 and every historical protocol and checkpoint are preserved unchanged.

Read before design:

- the current contracts in the `AGENTS.md` summary;
- the WL59 runtime and model algebra (`r57_model.Transport`, `fnto_persist59/runtime.py`, `train.py`, `evaluate.py`);
- WL60's frozen bounds and replay;
- WL63's tagging (`tag_reference.py`) and its result;
- WL64/65 results;
- the r57/p53 first-exit oracle and its `cell_source` semantics.

## Prospective design (committed before any evidence)

[Design](../plans/INCOMING_RANK_FEASIBILITY66_V1.md) and [protocol](../../experiments/fnto_rank66/protocol/rank66_v1.json) were written against the **implemented** algebra.

**Incoming factorisation.** For a fixed geometry, port and channel, the scattered response to unit flux in bin a is row a of `D_i Aᵀ G_i`. A is one global column-stochastic 8×16 softmax per channel. In budget-normalised coordinates `Z = D⁻¹R`, the model is exactly `AᵀG`.

**Consequence of column-stochastic A.** Because `Aᵀ1 = 1` for every parameter value, every member of the architecture represents the bin-independent component exactly. Its contrast space `H·range(Aᵀ)` is at most 7-dimensional and is shared by all ports and geometries.

**Valid bounds that follow.** These relaxed lower bounds hold for any parameters and any training:

- the per-port relaxed bound;
- the pooled shared relaxed bound;
- the frozen-A bound;
- the frozen-A+O bound (free signed spatial coefficients);
- outgoing controls.

**Estimators.** Replica cross-products are U-statistics. The relaxed floor is bracketed: a low-biased estimator from cross-Gram tail concavity, and a high-biased one from held-out split subspaces. Fixed-projector floors and model errors are unbiased. Uncertainty is a delete-one-replica jackknife.

**Verdict.** The verdict is fixed on the primary 1 cm/16-out contrast metric, with thresholds 0.10 / 0.05 / 25% and A > B > C > D precedence. The rationale is recorded in the protocol before evidence: the absolute metric's denominator is dominated by a component that the architecture cannot lose.

Port selection is geometry-only and deterministic: 2085 (top dent centre), 2441 (top dent flank), 1123 (top far control), 2069 (bottom below dent). It was computed before any reference. `1−β ≥ 0.9899` at both geometries.

**Budget rule.** K=8 replicas of 32M per probe if predicted walk time ≤4 h. The resource preflight fixes the numeric budget, which is committed before generation.

**Checkpoints.**

- Primary: WL59 P_5901 / P_5902.
- Secondary, descriptive: WL64-B and WL65-B.

### Non-evidence smoke run and one pre-commit correction (disclosed)

Before the protocol commit, the whole pipeline ran in a scratch root (`~/rank66_smoke`) with 2M-photon packets and K=4 replicas. That is 1/16 of the evidence budget per packet, and half the replicas.

- **Seeds.** The smoke used the evidence seed formula, so its photon streams are not independent of the later evidence packets. The smoke is never pooled with evidence or used in the verdict.
- **Gates.** All 512 smoke packets passed the physical and tag gates. Replay reproduced WL59 at 8.0e-8 / 7.5e-8. Factorisation and model-span residuals were ≤6.4e-8. Packet-moment noise matched the across-replica scatter (median ratio 1.013).
- **Outcome seen.** The smoke analysis printed per-channel categories (B in every channel at smoke precision) before the protocol commit. **No decision rule, threshold, metric, population, port, budget or seed was changed after that output.**

The smoke exposed one code-correctness defect, which was corrected before the commit:

- **Defect.** The frozen contrast basis `H·range(Aᵀ)` was built with a 1e-10 relative rank tolerance. That retained a float32-softmax noise direction (singular value ~4e-8), giving rank 8 instead of the architecture-exact ≤7. The extra direction enlarged the frozen-A projection space and biased F_A **low**.
- **Fix.** The basis now uses relative tolerance 1e-5 with an explicit cap of 7, as the protocol's algebra states.
- **Verification.** A new test confirms rank 7 and equality with the absolute `range(Aᵀ)` residual for random float32 column-stochastic A. Rerun on the smoke: rank 7 in all 18 checkpoint/channel cases, and F_A equals the absolute residual to ≤5e-7 relative.

One test tolerance was also changed. The check `1 ∈ range(Aᵀ)` used 1e-8, which is below float32 softmax rounding (observed 7.4e-8); it now uses 1e-6. This change was made before any smoke analysis.

The smoke outputs, including the defective first analysis, are preserved under `~/rank66_smoke/smoke_v1_bugged/` and `~/rank66_smoke/`.
