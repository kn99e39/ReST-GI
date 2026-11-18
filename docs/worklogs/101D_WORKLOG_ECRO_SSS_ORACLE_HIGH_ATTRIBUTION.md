# Worklog 101D — ECRO SSS Oracle HIGH attribution

## Authorization and contract audit, 2026-11-18 KST

WL101D is the owner-authorized bounded successor to WL101C on the existing `ecro_lr3_adapter` checkout. The scope is substitution of the fixed-scene HIGH table. The original checkpoint, representation, renderer, estimator, six-event physical references and WL101C masks govern the comparison.

**Quantity.** WL100 trains the physical boundary suffix `f2+fh` through its unchanged OWN loss. `f2` owns exactly E2, and `fh` owns E3–E5 in the boundary probe. `B_theta[p,b,c]` is the resulting outgoing E2–E5 radiance per patch, outgoing bin and color after applying the generator factors to the existing incident irradiance encoding. The model's dynamic source and learned-flight approximations are part of the generated-table side of the diagnostic.

**Index and normalization.** WL100 samples positions uniformly within the existing planar/equal-area sphere patch, and directions uniformly in patch-frame z and azimuth. Each valid first-surface sample contributes radiance. The original probe divides by its valid first-hit count per slot. There is no multiplication by patch area, cosine or bin solid angle. The positive-going physical direction at the attached point is traced in reverse, with ray origin displaced 2 mm and a 1 mm first-hit position check. The camera queries `patch_bin(si, sid, -ray.d)` at interaction 2 and reads RGB using `(pid*64+bin)*3+c`, multiplied by the physical first interaction's path throughput. Calibration uses the existing patch-frame and clamping rules. A uniform physical slot average is the training target. Differences between that average and what a particular camera ray needs belong to the downstream representation/readout term under this oracle.

**Event ownership.** The camera's first physical event prepends one to E2–E5 of the suffix, yielding camera E3–E6. A complete index-matched SSS walk counts as one optical-domain interaction. Ballistic light connection through the sphere belongs to the next optical-domain event, and therefore contributes at the same order in probe and camera modes. Air flight adds no event. A probe reuses the original five-event trace, including Beer and the same SSS walk. The probe excludes direct E1. No physical image enters the table estimate.

Preflight checks the slot assignment and reachable support explicitly. Results and exact stop rules are recorded by the [protocol](../../experiments/ecro_sss101d/protocol/oracle.json).
