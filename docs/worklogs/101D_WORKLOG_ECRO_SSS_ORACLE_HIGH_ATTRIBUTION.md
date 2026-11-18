# Worklog 101D — ECRO SSS Oracle HIGH attribution

## Authorization and contract audit, 2026-11-18 KST

WL101D is the owner-authorized bounded successor to WL101C on the existing `ecro_lr3_adapter` checkout. The scope is substitution of the fixed-scene HIGH table. The original checkpoint, representation, renderer, estimator, six-event physical references and WL101C masks govern the comparison.

**Quantity.** WL100 trains the physical boundary suffix `f2+fh` through its unchanged OWN loss. `f2` owns exactly E2, and `fh` owns E3–E5 in the boundary probe. `B_theta[p,b,c]` is the resulting outgoing E2–E5 radiance per patch, outgoing bin and color after applying the generator factors to the existing incident irradiance encoding. The model's dynamic source and learned-flight approximations are part of the generated-table side of the diagnostic.

**Index and normalization.** WL100 samples positions uniformly within the existing planar/equal-area sphere patch, and directions uniformly in patch-frame z and azimuth. Each valid first-surface sample contributes radiance. The original probe divides by its valid first-hit count per slot. There is no multiplication by patch area, cosine or bin solid angle. The positive-going physical direction at the attached point is traced in reverse, with ray origin displaced 2 mm and a 1 mm first-hit position check. The camera queries `patch_bin(si, sid, -ray.d)` at interaction 2 and reads RGB using `(pid*64+bin)*3+c`, multiplied by the physical first interaction's path throughput. Calibration uses the existing patch-frame and clamping rules. A uniform physical slot average is the training target. Differences between that average and what a particular camera ray needs belong to the downstream representation/readout term under this oracle.

**Event ownership.** The camera's first physical event prepends one to E2–E5 of the suffix, yielding camera E3–E6. A complete index-matched SSS walk counts as one optical-domain interaction. Ballistic light connection through the sphere belongs to the next optical-domain event, and therefore contributes at the same order in probe and camera modes. Air flight adds no event. A probe reuses the original five-event trace, including Beer and the same SSS walk. The probe excludes direct E1. No physical image enters the table estimate.

Preflight checks the slot assignment and reachable support explicitly. Results and exact stop rules are recorded by the [protocol](../../experiments/ecro_sss101d/protocol/oracle.json).

## Stage 0 result — INCONCLUSIVE, 2026-11-18 KST

### Implementation fact

The test used the original `pt100.Ctx.field_components` inherited by WL101's `RT`. An instance-local observation hook retained the original valid denominator and first-hit address. It delegated every path to the original tracer. The same-seed E2 and E3–E5 suffix comparison against camera mode at depth 5 was **exact to 0** on accepted probes. The original synthetic camera test covered 10,761 nonzero channel lookups at 1 spp. It checked the learned tensor stride with an RGB address-coded table, an all-ones table and a zero table. Maximum address and superposition discrepancies were **1.10e-7** and **2.38e-7**. Synthetic table substitution left both LOW event families bit-identical.

The learned table exactly matches the archived WL101 arm B and WL101A table. SSS cfg0 geometry, illumination, camera, material and checkpoint identities passed the stored-input gates. The learned table is retained in the new raw folder. The production tracer, weights and physical references were not written.

### Measurement — original probe support

A 32-spp membership audit at seed **10190901** covered all 81,280 nominal addresses, with **2,592,482** accepted samples out of 2,600,960 trials. The unchanged generator recorded **9,073** accepted samples in a nominal slot different from the actual first hit's patch/bin, **0.3500%** across all accepted samples and **1,295 slots**. This includes:

| Observed category | Accepted samples |
|---|---:|
| Different geometric shape | 9,063 |
| Same shape, other patch | 5 |
| Same patch, other bin | 5 |
| Accepted before the traced front-face test, with a back-facing hit | 9,053 |

The first three rows partition the 9,073 misroutes. The last row is a separate count. The direct first-hit replay reproduced the recorded counts and every misrouted slot exactly. The largest marked slot is floor patch 206 / bin 54, with **26 of 32** accepted hits at another address. Its patch center is (1.625, 1.125, 0) m. The off-shape first hits are all within the original 1 mm acceptance window. [Membership matrix](../../evaluation/101D/shape_membership.png)

### Measurement — use by the unchanged camera

A unit-valued synthetic table marked the 1,295 suspect slots and set all other entries to zero. The only table operation was the original camera HIGH gather and path-throughput multiplication. Both original 4,096-spp camera streams at seeds **101100 / 101101** gave positive marked-slot weight in every pixel of the frozen near ROI. The unit marker maps identify support usage. Their values are neither physical radiance nor a proportion of the HIGH bias.

| Original region | Pixels | Nonzero marker pixels, seed 101100 / 101101 | Mean RGB unit-marker weight, 101100 / 101101 |
|---|---:|---:|---:|
| Full | 262,144 | 237,150 / 237,118 | 0.010224 / 0.010226 |
| Broad contact/shadow | 38,071 | 26,184 / 26,206 | 0.017468 / 0.017464 |
| Near floor | 1,318 | 1,318 / 1,318 | 0.084857 / 0.084795 |
| Flank | 6,042 | 6,024 / 6,027 | 0.038296 / 0.038320 |

The maximum near-floor marker weight across RGB was **0.3563 / 0.3512**. The two marker frames took **10.627 / 10.599 s** on LabServer63. The direct first-hit replay took **0.358 s** after setup. [Camera support comparison](../../evaluation/101D/camera_support.png) · [Audit metrics](../../evaluation/101D/metrics.json)

### Observation

The dominant shape swap is between the floor and Box B's coplanar underside. Accepted first hits run floor→Box B **2,362** times and Box B→floor **6,115** times. The sphere and floor also exchange first hits at the tangency. Counts are **42** floor→sphere and **41** sphere→floor. The camera marker combines all flagged addresses. It does not isolate the contribution of either geometric contact to the WL101C hotspot.

### Interpretation

The original WL100 probe accepts a hit using position proximity. Its bin numerator and count are accumulated under the **requested** address. An accepted hit can belong to a different shape. Front-facing activity is checked **after** copying `okp` for the count, so a back-facing wrong-surface hit can contribute zero E2–E5 radiance while still incrementing the denominator. These are geometric membership and normalization violations for the intended outgoing radiance cell. The unchanged sampler preserves those effects in the nominal training target. Rejecting or relabelling those samples changes the effective WL100 target and would require a separate decision under this brief. A longer nominal probe run would estimate the same contaminated target more precisely.

### Unresolved question

The measured support mismatch establishes the first blocker to a physical Oracle comparison. Its sign and magnitude in the WL101C learned HIGH bias are undetermined. The original contact excess **+0.06991** remains the historical WL101C result. Neither a generator fraction nor a representation/readout fraction is measured. The planned Oracle halves and physical Oracle image arms were held at the admission gate. This is a semantic stop before uncertainty qualification.

### Architecture verdict

**INCONCLUSIVE — first-hit membership admission failed.** The same-domain physical quantity required for substitution is not established on camera-reachable table slots under the exact supervising sampler. The boundary probe's first-hit ownership and count normalization must be resolved to make this particular generator/readout split interpretable. The evidence gives no attribution to the learned generator, angular discretization, or camera readout alone.

### Reproducibility

- Protocol and support admission commit `6ddf8fcbdc6d219a35c5656bb0a4298ebe231b26`. Exact first-hit replay commit `89be56901040ba68cb156f981b0704650e183b50`. Early stop placement and offline report commit `5951cbe05d91fae4a6550e2194388ae832bad01b`. These are experiment-specific tools. The initial stage-0 probe was development evidence and its frozen-seed reproduction is recorded separately below.
- LabServer63 RTX 3080 Ti, driver/OptiX 595.71.05, Mitsuba 3.8.0, Dr.Jit 1.3.1, Torch 2.3.1+cu121, Python 3.10.21 and NumPy 1.26.4. No foreign compute process was observed at the marker-readout boundaries.
- Scene hash `a18c76efd49245dfb3f50d6d215e50d1b70e87854bac943126b60b09a50b71cb`. Checkpoint SHA-256 `2d9cb5805ecff8a6aae6639ddca7a3bcd0c84b0eaa9b2a3282fe127df9284f42`. Learned response bytes SHA-256 `d7e40954304bfc2b03f679af0ef6a39876f21ffc21d3a07fe3dfc3de7bd18543`.
- Project raw paths `results/ecro_sss101d/{preflight,support,membership}/` contain the original B theta copy, 32-spp probe packet, both synthetic marker images, per-slot membership tables, run metadata and logs. The probe packet is retained as failed-admission evidence. The reviewer panels deliberately label the archived learned/physical pair and the unit marker. [Gallery](../../evaluation/101D/index.html)

#### Committed stage-0 reproduction

Replaying seed 10190901 from `5951cbe05d91fae4a6550e2194388ae832bad01b` exactly reproduces every synthetic check, every valid count, and all 9,073 first-hit mismatches. The raw 32-spp table differs by at most 2.38e-7 due to floating-point summation order. The replay process completed in 4.862 s with warm renderer caches. Its raw path is `results/ecro_sss101d/preflight_replay/`. This is a repeat of a failed membership gate, with no production Oracle estimation. [Checksums](../../evaluation/101D/manifest101d.json)
