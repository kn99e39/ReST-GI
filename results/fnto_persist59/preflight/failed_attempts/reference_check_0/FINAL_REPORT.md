# Persistent Intrinsic F/S — server completion report

Status: FAILED_OR_STOPPED_PRIMARY_GATE

## Implementation Fact
One dependency-local positive graph rule; same checkpoint for F and actual resident S. Nonrigid B and global normalization rebuilt and charged. No source/target lookup in state construction.

## Measurement

## Observation
Global state effects are explicitly reduced; preserved raw computation is not confused with unchanged radiance. All diagnostic poses/channels remain in heldout.json.

## Interpretation
Physical learned fidelity and F/S equivalence are separate verdicts. If F misses required fidelity, matching F does not establish physical validity or a quality-matched speedup.

## Unresolved Question
Any remaining representation/supervision/optimization bias is not causally isolated by this single candidate. No successor selected.

## Architecture Verdict
- rule_input_generalization: INCONCLUSIVE
- primary_directional_adequacy: INCONCLUSIVE
- full_state_learned_fidelity: INCONCLUSIVE
- FS_equivalence: INCONCLUSIVE
- invalidation_correctness: INCONCLUSIVE
- selective_update_quality_matched_benefit: INCONCLUSIVE
- actual_coupled_GI_eligibility: INCONCLUSIVE
- sharp_glossy_global_deformation_self_contact: INCONCLUSIVE

## Reproducibility
Source d7256dc
Protocol SHA 1e84a085c0a1916ecefe1a2607d1df30fa22fd2fbe38a2bf02a3bd0b4cc0134b
Seeds5901/5902,6000steps, final checkpoints; physical reference unchanged. Complete stage/attempt logs, data manifests, environment, model and artifact receipts retained under this server root.
Sharp glossy, global bends and self-contact are outside scope. WL58 evidence unchanged. No new hypothesis or production adoption.
