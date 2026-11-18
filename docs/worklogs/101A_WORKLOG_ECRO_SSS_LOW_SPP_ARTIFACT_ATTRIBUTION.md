# Worklog 101A — ECRO SSS low-SPP artifact attribution

## Authorization and freeze, 2026-11-18

The user authorized a controlled attribution follow-up to WL100 and WL101 on WL101 SSS cfg0/test-0. This uses the available suffix **WL101A** because the scientific question directly follows the observed artifact under the existing architecture. An isolated branch starts at completed WL101 so the active WL102 checkout and ongoing long-path experiments can continue independently.

The protocol fixes the scene, camera, checkpoint, reference identities, renderer stack, seed schedule, regions, error estimators, display scales and dominance rule before evidence execution. The sole optimized renderer is the original WL101 arm B on LabServer69. Its original published arm A and B arrays, gbuffer and two six-event reference halves are read-only inputs. No new physical reference renders are planned. Hashes distinguish the original server's CRLF source bytes from the equivalent LF git content.

The learned table and image-space HIGH are different objects. The table is conditioned on fixed geometry and source. The readout samples camera and first-event paths, which can change both lookup addresses and weights between seeds. The study measures both table identity and image-space HIGH variance. Matching the physical reference seeds at 4096 spp supports an event-by-event LOW control and an independent-pair estimate of stable HIGH error against the custom physical reference. WL100's native-volpath qualification remains a separate bound on physical interpretation.

Protocol and tools are under [experiments/ecro_sss101a](../../experiments/ecro_sss101a/). Raw artifacts go to `results/ecro_sss101a/` and reviewer artifacts to [evaluation/101A](../../evaluation/101A/).

### Pre-execution checksum erratum

The local input check caught two omitted characters in the initial frozen string for the archived arm-B 4096-spp prediction. The corrected string matches the original server file and the independently copied local bytes. The file itself, the reference halves, the sampling protocol and the decision criteria were already fixed. No WL101A renders had been launched.

### Archived input verification

The five copied WL101 cfg0 arrays match their server SHA-256 values. The published arm A and arm B 4096-spp images use seed 101500. Their full-image relative L1 discrepancy is 2.75e-7. The corresponding physical LOW discrepancy is 1.76e-7 and the learned HIGH discrepancy is 7.77e-7. These are historical estimator-equivalence checks on the exact scene. The new run adds the fresh replay and paired physical LOW controls.

The fixed center-ray masks contain 262,144 full-frame pixels, 49,660 sphere pixels, 45,692 interior pixels, 3,968 inner-boundary pixels, 72,657 floor pixels and 38,071 floor contact/shadow pixels. The partition and subset checks passed. Synthetic known-input tests validated component variance and covariance, finite-replicate bias correction, the inverse-SPP slope and the HDR display transfer.

Against the WL101 average reference, the archived independent arm-B prediction's sphere-relative RMSE is 1.858% full, 1.487% LOW and 3.651% HIGH. The corresponding reference-mean noise estimates are 1.031%, 0.868% and 1.910%. The unpaired LOW discrepancy includes independent prediction and reference sampling. These archived values alone cannot resolve systematic LOW or quantify the 32-spp artifact.
