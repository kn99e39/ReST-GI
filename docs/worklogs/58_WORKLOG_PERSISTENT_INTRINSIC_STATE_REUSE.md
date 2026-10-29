# Worklog58 — Source-independent rule and persistent intrinsic transport state

Batch identity: **intrinsic_state_reuse_v1**. Worklog numbering is chronological only.

## Authorisation and scope

The user explicitly authorised the new bounded batch after the completed WL57 negative result on2026-10-29. One candidate only; Selective_Recompute only, no new branch/worktree, no production/material/reference changes or automatic successor. A fundamental validity-gate failure closes this candidate before training. The separate GPU-training execution improvement proposal is deferred at the user's request.

Candidate design: [Rule/State and dependency contract](../plans/INTRINSIC_STATE_REUSE_V1.md). Executable protocol: `experiments/fnto_reuse58/protocol/reuse_v1.json`. Code: `experiments/fnto_reuse58`. Original WL53–57 code, thresholds and failed evidence are preserved.

## Source grounding

Inspected the current architecture/research-topic/roadmap, FNTO K/C physical ownership contract, WL53–57 physical and learned-reference findings, WL56 complete-domain selective recomputation, WL57 model/geometry/loss implementation and completed result manifests. Starting source29fd693 onSelective_Recompute. This is not an independent rerun of historical physical measurements: those are inherited evidence, explicitly labelled as such.

WL57 established neither complete dynamic/angular quality nor a useful R advantage. Its reusable weights and per-pose rebuilt state do not establish persistent computed-state reuse. WL56's full/selective physical packages agree while the complete physical target still fails in blue. Neither fact is promoted to a new-candidate success.

## Prospective preflight

The first gate isolates finite angular-interface sufficiency before fitting any new weights. The candidate increases the old16-in/64-out interface to64-in/256-out and defines exact dependency-based persistent state. One compact static GGX surface probe is prospectively included, alongside the unchanged homogeneous medium as the intended main fixture. Its inclusion and strict gate are choices for this candidate, not a claim that WL53 volume references support glossy surface physics.

Independent analytic float64 GGX reference,512/1024/2048 projected-angle grids and independent polar quadrature; fixed alpha/angles/source-pair population, optimal output projection and indistinguishable-input lower bounds. Thresholds and stop actions are fixed in the committed protocol. No learned training, oracle expansion or final primary angular measurement ran before protocol/code commit. Local five focused tests passed; these are implementation tests, not physical-fidelity success.

Reports, measurements and final verdicts will be recorded here and in machine-readable result files. Per user instruction, `docs/Architecture.md` contains central design only; no batch report will be appended there.


## Final report — STOPPED_ANGULAR_REPRESENTATION

### Implementation Fact

The implemented and executed portion is the weight-independent **angular admission gate**, its streaming quadrature, source-pair information bound, five focused analytic tests, durable server-side chain, provenance and machine-readable verdicts. The persistent graph-state architecture and conservative invalidation contract were specified, but the encoder/decoder, F/S cache updater, new supervision, neural training and K/C integration were **not implemented or executed** after this fundamental gate failed. No claim of F/S equivalence, retained-state correctness or selective savings follows from the design alone.

The candidate deliberately had a prospective glossy-capable finite interface:64incoming/256outgoing bins, no continuous-direction/specular bypass. The additional static GGX probe was optional in the user brief and was explicitly included as a mandatory candidate gate before measurement. This negative result applies to that choice. It is not a discovery that every homogeneous-volume experiment requires glossy physics, nor an independent explanation of WL57's in-basis dynamic failures.

### Measurement

All20reference cases passed the predeclared1%quadrature control. The largest fine-grid comparison discrepancy was0.007761% (binned output); energy discrepancy0.003351%, L2-norm discrepancy0.005572%. Independent polar versus square quadrature differed by at most0.002604% in energy and0.001856% in L2 norm. Reference output/input flux ranged0.857259–0.998304, below1.001. Reciprocity relative error2.07e-16; the independent normal-incidence peak check error5.73e-14. These are deterministic integration/bias checks, **not Monte Carlo seed variance or statistical confidence intervals**.

For alpha0.04, the optimal256-bin piecewise-constant outgoing projection alone loses52.36–80.00%relativeL2 across the declared cases. Even unlimited model capacity could not remove that loss within this fixed output basis. The normal/30°/60°primary-angle subset loses60.28–80.00%. At alpha0.25 the loss is9.90–17.71%; four of20totalcases pass the fixed10%projection criterion. Every case is retained below.

| Case | alpha | Output/input flux | Optimal output relativeL2 | 10% criterion |
|---|---:|---:|---:|---|
| a0.04_theta0_phi17 | 0.04 | 0.998304 | 0.602767 | FAIL |
| a0.04_theta0_phi61 | 0.04 | 0.998304 | 0.602767 | FAIL |
| a0.04_theta30_phi17 | 0.04 | 0.997984 | 0.627433 | FAIL |
| a0.04_theta30_phi61 | 0.04 | 0.997984 | 0.646110 | FAIL |
| a0.04_theta60_phi17 | 0.04 | 0.995431 | 0.793281 | FAIL |
| a0.04_theta60_phi61 | 0.04 | 0.995431 | 0.799986 | FAIL |
| a0.25_theta0_phi17 | 0.25 | 0.915808 | 0.098978 | PASS |
| a0.25_theta0_phi61 | 0.25 | 0.915808 | 0.098978 | PASS |
| a0.25_theta30_phi17 | 0.25 | 0.904142 | 0.108296 | FAIL |
| a0.25_theta30_phi61 | 0.25 | 0.904142 | 0.110522 | FAIL |
| a0.25_theta60_phi17 | 0.25 | 0.857259 | 0.167201 | FAIL |
| a0.25_theta60_phi61 | 0.25 | 0.857259 | 0.177087 | FAIL |
| a0.04_cell18_X | 0.04 | 0.998057 | 0.561582 | FAIL |
| a0.04_cell18_Y | 0.04 | 0.998141 | 0.624560 | FAIL |
| a0.04_cell27_X | 0.04 | 0.998253 | 0.523563 | FAIL |
| a0.04_cell27_Y | 0.04 | 0.998278 | 0.578607 | FAIL |
| a0.25_cell18_X | 0.25 | 0.906730 | 0.108493 | FAIL |
| a0.25_cell18_Y | 0.25 | 0.909813 | 0.104585 | FAIL |
| a0.25_cell27_X | 0.25 | 0.914229 | 0.099826 | PASS |
| a0.25_cell27_Y | 0.25 | 0.915202 | 0.099214 | PASS |

The source-swap lower bound uses the actual256-bin output vectors, independently of continuous-output projection error. X and Y are distinct unit-flux collimated incident fields at unchanged geometry/material. They fall within one64-bin input cell, so the candidate's input arrays are exactly equal. For any common deterministic prediction z, triangle inequality implies that if BOTH relative errors were<=epsilon, then ||yX-yY||<=(||yX||+||yY||)epsilon. Therefore the tabulated lower bound must not exceed the prospective25%criterion.

| alpha | Input cell | Input difference | Necessary worst-pair error lower bound | 25% criterion |
|---:|---:|---:|---:|---|
| 0.04 | 18 | 0.0 | 0.568655 | FAIL |
| 0.04 | 27 | 0.0 | 0.559097 | FAIL |
| 0.25 | 18 | 0.0 | 0.157134 | PASS |
| 0.25 | 27 | 0.0 | 0.144761 | PASS |

The alpha0.04 lower bounds55.91–56.87% exceed25% with a large margin relative to quadrature discrepancies. This input-information failure persists even if evaluation is restricted to the actual output bins; it is not solely a demanding continuous readout metric. The probe is achromatic: RGB repeat the same operator, not three independent experiments.

Executed analytic gate time: **13.177s**; peak CPU RSS **59,666,432bytes**. Environment: Python3.10.21, NumPy1.26.4, LabServer69 CPU. GPU was not needed for this small bounded analytic gate; no GPU training or GPU speedup was measured. Full candidate state memory, training resource feasibility, invalidation fraction, update/first-query/repeated-query/C/solve/readout lifecycle costs remain unmeasured. Five tests passed independently on local and server environments; they are five test definitions, not ten distinct physical validations.

### Observation

The selected histogram interface discards sub-bin incident direction needed for sharp local reflection. Increasing the coefficient counts4x on both sides relative to WL57 is still insufficient for this particular prospective high-frequency probe. The smoother control reduces the source-alias bound, but does not make every output-projection group meet10%. No favorable control replaces a failed group, and no post hoc resolution/rank sweep was performed.

### Interpretation

This is a **representation admission failure** before learning, not an optimiser failure, a measured neural-model failure, a volume-reference failure or a state-update implementation failure. It is independent of learned capacity because both proofs allow the best possible output within the selected basis. The new candidate cannot enter the claimed glossy-capable evaluation at its frozen interface. A useful local-deformation F/S comparison in the homogeneous-volume subset was not reached and is not adjudicated by this result.

The user brief's stop condition applies: the proposed rule interface cannot represent the declared required directional response. Continuing training or adding a continuous/specular path after this evidence would change the frozen representation. No second candidate, material change, threshold relaxation, retraining campaign or successor is started.

### Unresolved Question

Which directional interface can satisfy the intended physical scope while allowing a fair persistent-state F/S comparison is unresolved. The abstract benefit of preserving valid intrinsic computed state, its geometry-dependent support and measured cost advantage remains unresolved; this gate supplies no evidence for or against selective-update efficiency.

### Architecture Verdict

| Question | Verdict | Measured scope |
|---|---|---|
| Analytic reference readiness | SUPPORTED |20static GGX cases; not a new Cycles/volume validation |
| Source-independent response correctness | INCONCLUSIVE | No learned rule fit; selected input representation has a proven alias obstruction |
| Directional representation adequacy | NOT SUPPORTED | Frozen64-in/256-out interface on the prospectively included sharp GGX probe |
| Full-state learned fidelity F | INCONCLUSIVE | Not trained/evaluated |
| Selective-state fidelity S | INCONCLUSIVE | Not implemented/evaluated |
| State invalidation correctness | INCONCLUSIVE | Contract specified, no executed updater |
| Actual update/query cost advantage | INCONCLUSIVE | No F/S lifecycle measurement |
| K/C composition eligibility | NOT SUPPORTED | Candidate fails admission; no coupled GI solve |
| Rigid/local-deformation scope | INCONCLUSIVE | No new candidate geometry execution |
| Global bending/self-contact | INCONCLUSIVE | Untested |
| Long-term architecture compatibility | INCONCLUSIVE | Ownership specified only; directional adequacy blocks implementation eligibility |
| Abstract reusable-rule/persistent-state hypothesis | INCONCLUSIVE | This finite-interface stop is not a universal falsification |

No central architecture is adopted or replaced. The batch closes as a bounded **negative preflight result**; the full research question is unanswered by this candidate. The historical WL57 learned failure and WL56 blue physical floor remain intact.

### Reproducibility

Source/protocol fixed before primary measurement: **af600ec7c4b3a6d7682f69882819de2902c93525**. Document-role cleanup is separate commit73542fa; the scientific starting HEAD remains29fd693. Exact source/protocol hashes and artifact SHA256 values are in `results/fnto_reuse58/preflight/provenance.json`. Server snapshot: `~/rest_reuse_v1/repo`; root status `STOPPED_ANGULAR_REPRESENTATION`. The detached process had PPID1 and its own session; the completed-status file and raw launch/completion timestamps are preserved. Server clock was about3minutes behind the client; raw stamps are not silently corrected.

```sh
python -m unittest discover -s experiments/fnto_reuse58 -p 'test_*.py' -v
python experiments/fnto_reuse58/angular_preflight.py --out <new-empty-output-directory>
# Exact detached server snapshot used:
# REUSE_ROOT=$HOME/rest_reuse_v1 bash $HOME/rest_reuse_v1/repo/experiments/fnto_reuse58/chain.sh
```

The gate rejects overwriting completed evidence.512/1024/2048 grids, alpha/angles, input cells/offsets, thresholds and float64 arithmetic are in the committed protocol. This deterministic gate has no photon seed or neural checkpoint. Seed580101 is used only for the fixed reciprocity implementation check. All case-level outputs, controls, source-pair bounds, logs, tests and hardware/runtime records are in [the original evidence](../../results/fnto_reuse58/preflight/). GGX/Smith equations follow the linked primary PBRT4 definition in the design; performance and failure bounds are measured here, not inferred from that source. Historical results are inherited and were not independently rerun in this batch.
