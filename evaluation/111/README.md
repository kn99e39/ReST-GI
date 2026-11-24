# WL111 reviewer index — External Baseline Benchmark and Contribution Approval Gate

Worklog: [docs/worklogs/111_WORKLOG_EXTERNAL_BASELINE_BENCHMARK_AND_APPROVAL_GATE.md](../../docs/worklogs/111_WORKLOG_EXTERNAL_BASELINE_BENCHMARK_AND_APPROVAL_GATE.md). Hashes and sources are in `manifest111.json`.

**Display transform (all images):** LDR = clip(x, 0, 1)^(1/2.2) at exposure 1. **Error maps:** signed (x − ref) luminance / mean reference luminance. Scales: ±0.5 for the image grids, ±0.2 for the contact crop, ±1 for LR3.

| Deliverable (brief) | Where |
|---|---|
| A. Baseline registry | [docs/EXTERNAL_BASELINE_REGISTRY.md](../../docs/EXTERNAL_BASELINE_REGISTRY.md) |
| B. Workload compatibility matrix | same document, "Workload compatibility matrix" |
| C. Reproduction / comparability report | registry "Executed baseline runs" + worklog |
| D. Executed manifests | `manifest111.json`; raw run manifests on Main `C:\Projects\bench111\{evidence,lr3\out}` and local `results/bench111/` (untracked) |
| E. Raw measurements | `summary111.json` (all arms, spp, replicates' aggregates, timing), `tiers111.json`, `sss_gate111.json` |
| F. Quality–performance frontiers | `frontier_fixtures.png`, `frontier_S0_V0.png`, `frontier_S0_H1.png`, `frontier_S0_H2.png`, `frontier_S1_V0.png`, `frontier_SL_V0.png`, `frontier_N0_V0.png` (unqualified reference), `frontier_LR3.png` |
| G. Shared-scale reference and prediction renders | `images_S0_V0.png`, `images_S0_H1.png`, `images_S0_H2.png`, `images_S1_V0.png`, `images_SL_V0.png`, `images_N0_V0.png`, `images_LR3.png`, `contact_crop_S0_V0.png` |
| H. Dynamic edit / lifecycle | `dynamic_edit_latency.png`; [gate §5–§6](../../docs/CONTRIBUTION_APPROVAL_GATE.md) |
| I. Claim-to-evidence matrix | [docs/CLAIM_EVIDENCE_MATRIX.md](../../docs/CLAIM_EVIDENCE_MATRIX.md) |
| J. Numerical approval gate proposal | [docs/CONTRIBUTION_APPROVAL_GATE.md](../../docs/CONTRIBUTION_APPROVAL_GATE.md) §1–§4, §7 |
| K. Current ReST-GI gap analysis | gate §4 table and the structural requirement below it |
| L. Mandatory SSS near-contact gate | [docs/SSS_NEAR_CONTACT_GATE.md](../../docs/SSS_NEAR_CONTACT_GATE.md); `sss_radial_profile.png`, `contact_crop_S0_V0.png` |
| Benchmark contract | [docs/BENCHMARK_PROTOCOL.md](../../docs/BENCHMARK_PROTOCOL.md) |

**Figure captions.**
- `frontier_*.png`: median frame time on the RTX 5080 (log) vs relative debiased HDR RMSE against the full-depth reference (log). Numbers are spp; error bars are 95 % bootstrap over replicates; the dotted line is ECRO's converged bias floor; vertical lines mark the 10 / 15 / 20 ms budgets.
- `frontier_LR3.png`: GI LDR RMSE against Ref-65. TransGI points use the fast-frame time; the "adopted" point's quality options are not in the timed frame. ECRO's point is historical (WL102, RTX 3090).
- `sss_radial_profile.png`: signed relative floor-luminance error per ring from the sphere support point.
- `dynamic_edit_latency.png`: edit → first 512² × 1 spp frame, bars = median, ticks = p95.
