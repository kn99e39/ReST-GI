# Worklog 101A — ECRO SSS low-SPP artifact attribution

## Authorization and freeze, 2026-11-18

The user authorized a controlled attribution follow-up to WL100 and WL101 on WL101 SSS cfg0/test-0. This uses the available suffix **WL101A** because the scientific question directly follows the observed artifact under the existing architecture. An isolated branch starts at completed WL101 so the active WL102 checkout and ongoing long-path experiments can continue independently.

The protocol fixes the scene, camera, checkpoint, reference identities, renderer stack, seed schedule, regions, error estimators, display scales and dominance rule before evidence execution. The sole optimized renderer is the original WL101 arm B on LabServer69. Its original published arm A and B arrays, gbuffer and two six-event reference halves are read-only inputs. No new physical reference renders are planned. Hashes distinguish the original server's CRLF source bytes from the equivalent LF git content.

The learned table and image-space HIGH are different objects. The table is conditioned on fixed geometry and source. The readout samples camera and first-event paths, which can change both lookup addresses and weights between seeds. The study measures both table identity and image-space HIGH variance. Matching the physical reference seeds at 4096 spp supports an event-by-event LOW control and an independent-pair estimate of stable HIGH error against the custom physical reference. WL100's native-volpath qualification remains a separate bound on physical interpretation.

Protocol and tools are under [experiments/ecro_sss101a](../../experiments/ecro_sss101a/). Raw artifacts go to `results/ecro_sss101a/` and reviewer artifacts to [evaluation/101A](../../evaluation/101A/).
