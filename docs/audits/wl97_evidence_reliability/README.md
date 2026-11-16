# WL97 evidence reliability audit — index

**Worklog.** [docs/worklogs/97_WORKLOG_HISTORICAL_EVIDENCE_RELIABILITY_AUDIT.md](../../worklogs/97_WORKLOG_HISTORICAL_EVIDENCE_RELIABILITY_AUDIT.md)

**Start here.** [COORD_synthesis_and_p0_ledger.md](COORD_synthesis_and_p0_ledger.md) holds the executive verdict, the P0 matrix, retractions, defects, special-check coverage and the dependency impact. The coordinating session wrote it and is not independent.

**Proposals (not adopted).** [PROPOSED_LIVING_DOCUMENT_ANNOTATIONS.md](PROPOSED_LIVING_DOCUMENT_ANNOTATIONS.md)

**Reviewer reports.** Each is kept verbatim, as its agent wrote it.

| file | track | phase |
|---|---|---|
| MAP_evidence_dependency_and_provenance.md | evidence dependency map, executed revisions, artefact identities | 1 |
| A_math_physics_contract.md | implemented transport equations, conventions, event accounting, passivity | 1 |
| B_data_and_protocol_integrity.md | data, splits, coverage, freeze / amendment / exposure integrity | 1 |
| C1_measurement_statistics_wl92_wl94d.md | metrics and statistics WL92–WL94D | 1 |
| C2_measurement_statistics.md | metrics and statistics WL87–WL91, WL95–WL96A | 1 |
| D_systems_performance_cost.md | timing boundaries, Amdahl ceilings, cost claims (static) | 1 |
| F0_inference_chain.md | inference-chain review (formed before other findings) | 1 |
| E1_learned_sender_error_and_passivity.md | independent reproduction of P0-1 | 1 |
| E2_relation_sampling_reproduction.md | independent reproduction of P0-2 / P0-3 (relation sampling) | 1 |
| E3_deterministic_delivery_oracle.md | independent reproduction of P0-2 / P0-3 (deterministic quadrature) | 1 |
| E4_resolvent_accuracy_headroom.md | independent reproduction of P0-4 (accuracy) | 1 |
| E5_lifecycle_cost_reproduction.md | independent re-timing of P0-4 (cost) | 1 |
| J1_p01_learned_sender_adjudication.md | adversarial adjudication, P0-1 | 2 |
| J2_p02_binned_sender_bias_adjudication.md | adversarial adjudication, P0-2 | 2 |
| J3_p03_exact_evaluation_adjudication.md | adversarial adjudication, P0-3 and FLIGHT_ATTRIBUTION | 2 |
| J4_p04_accuracy_headroom_adjudication.md | adversarial adjudication, P0-4 accuracy | 2 |
| J5_p04_cost_and_wl96_cost_critic.md | adversarial adjudication, P0-4 cost and WL96 cost critic | 2 |
| J6_metric_evaluation_defects_adjudication.md | adversarial adjudication, metric defects WL92–WL95A | 2 |
| J7_data_protocol_statistics_adjudication.md | adversarial adjudication, data, protocol and founding failure | 2 |

**Code, outputs and logs.**
- Scripts: `experiments/audit97/<track>/`. The helper `experiments/audit97/_tools/show.py` prints structured findings.
- Outputs: `results/audit97/<track>/`, plus `phase1_results.json` and `phase2_results.json` (the structured returns of all agents).
- Server working directories: LabServer63 `~/rest_x97/<track>/`.
- Interrupted first attempts left auto-extracted action logs in `results/audit97/<track>/_interrupted_run_log.md`.
