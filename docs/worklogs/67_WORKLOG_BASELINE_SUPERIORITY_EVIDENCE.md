# Worklog 67 — Baseline Superiority Evidence: Quality-Matched Full Recompute, Neural Baselines and ReST-GI (2026-11-01)

**Authorisation:** the user's 2026-11-01 brief "Baseline Superiority Evidence", run in parallel with the separately approved WL66 representation study. Branch `Selective_Recompute`; no branch or worktree. Isolated adapters, common-scene/reference preparation, instrumentation, tests and new outputs only. No ReST-GI K, baseline hyper-parameter, schedule, threshold, mask or material change. No WL66 result is read or used. **Protocol:** `experiments/baseline_sup67/protocol/bsup67.json` (`baseline_sup67_v1`), committed `847eb8a` before any comparative outcome.

## Completion answer

> On which physically aligned workloads, if any, does ReST-GI provide a measured quality-matched advantage over Full Recompute or relevant neural GI baselines, and what prevents that advantage where it has not been established?

**On none.** The only aligned workload on which ReST-GI runs at all is W1, the WL56 homogeneous two-slab medium with its dent and rigid edits. There, quality-matched physical Full Recompute (Cycles volume path tracing, the reference renderer itself) meets the complete 3% contract at **4 spp in 4.67 s** for the four face-on views. The ReST-GI exact-domain oracle Selective update takes **189.8 s** (Full 371.9 s) and still fails that contract (**9/12**, max group bias **10.4%**). That is **class C**: no demonstrated advantage. Full Recompute is about 40× faster and more accurate.

What prevents an advantage on W1:
- **Physical fidelity.** The 2.5 mm oracle K/C has a blue/transmission floor. Neither F nor S reaches the contract at any cost.
- **Cost.** Changed-domain TR construction alone is about 175 s.
- **The learned K is ineligible.** The WL59 learned rule, which would remove TR, fails its own directional/deformation fidelity contract and has never been run in a coupled solve.

**Two scoped observations do not reach A:**
- **Stationary-receiver signed change.** The oracle S predicts the per-port signed change on the stationary receiver more precisely than single-realization Cycles even at 1024 spp. S error is 0.068/0.030/0.045 against Cycles' 0.116–0.122/0.053–0.055/0.075–0.091 R/G/B. But S is slower (189.8 s vs 127.4 s), and the metric is partly limited by reference noise.
- **Rigid resident update.** The GPU-resident rigid S update (0.69 s) is faster than the cheapest Cycles point (2.6 s at 1 spp), but it is less accurate in group bias and fails the contract (10/12). This is **class B**: a scoped computational advantage with insufficient physical fidelity.

On **W2**, the WL22 teaset under common light (T0→T3 rigid relation change), the neural baselines are physically aligned with Full Recompute, and Full Recompute dominates them. Mitsuba path tracing at **64 spp (0.52 s on the RTX 3090)** has a lower interaction-ROI error (0.076/0.077) than every frozen or refitted RNA/8DNA output (0.103–0.124). The refits took 1.76 h and 3.24 h (WL27, RTX 5080). **ReST-GI cannot run on W2 (class D):** no K exists for rough conductors. TransGI is **blocked**: no released implementation.

## Implementation Fact

### Baseline availability (executable status)

| Arm | Status | Basis |
|---|---|---|
| A. Quality-matched Full Recompute | EXECUTED on W1 (Cycles 3.5.1 CUDA, reference renderer/settings) and W2 (Mitsuba 3.5.1 `path`, reference integrator) | Same renderer, build and settings as each workload's reference; new disjoint seeds |
| ReST-GI oracle F/S/T | EXECUTED on W1 (unchanged `fnto_sel56`, isolated root `~/rest_bsup67/w1oracle`) | Re-executed r0 equals WL56 r0 to relative L1 ≤1.5e-8 (T 3.6e-7); WL56 r1 inherited as the second realization |
| ReST-GI learned (WL59 P) | NOT COMPARABLE to W1 | Different geometry (single-domain held-out dent), intrinsic-only cost, no coupled solve; fails its own contract. Running it inside the WL56 solve would be a new composition and was not done |
| RNA | Stored WL22 outputs re-scored on W2 (reproduce WL22 to 0.0 / 6e-8); BLOCKED on W1 | See blocker evidence |
| 8DNA | Stored WL22 outputs re-scored on W2; inference re-execution BLOCKED on LabServer69; BLOCKED on W1 | Host GCC 8.4 cannot build its CUDA extension (torch needs GCC ≥9); WL27 inference timing (RTX 5080) inherited per protocol |
| TransGI | BLOCKED | arXiv 2506.09909 (v1, 2025-06-11): no code/project link on the abstract page; no official repository found by GitHub repository search on 2026-11-01. Paper scope: rigid motion only, glossy surfaces, test scenes with transparent objects removed, scene-specific training (~30 min sampling + ~2 h on an RTX 5000). No surrogate was implemented |

### Workloads and contract (fixed before outcomes)

- **W1 — WL56 two-medium slab** (rest/dent/rigid, 20 cm unit-radiance emitter).
  - Reference: WL56 Cycles 1024 spp × 4 seeds (LabServer69, CUDA), read-only.
  - Query workload: the four face-on views top/gapdown/gapup/bot at 220² and 0.5 mm.
  - Scoring: WL56 amended masks and the unchanged `s56_analyze.group_stats`, applied as the historical rule |bias| ≤ 3% + 3SE.
  - Quality metrics: M1 = passes among the 12 dent face × channel tests (single-realization form: SE = reference SE). M2 = max |bias|. M3 = per-port relative L2. M4 = stationary-receiver (lower top) per-port signed change error.
  - QM_contract: lowest spp at which both Cycles realizations reach 12/12. QM_FNTO: lowest spp at which both Cycles realizations are ≥ the ReST-GI arm on all of M1–M4.
- **W2 — WL22 teaset common light** (T0, T3).
  - Reference: GT mean of 2 × 32768 spp (RTX 5080).
  - Scoring: the unchanged `cross_backbone_eval.refit_block` (interaction-ROI display MAE, gain, WL22 recovery rule).
  - QM: lowest PT spp at which both realizations have T3 ROI error ≤ each neural arm's.
- **Ladders:** spp {1, 4, 16, 64, 256, 1024}, two independent realizations per level, seeds disjoint from every reference and oracle seed (tested).

### Execution and corrections (all before the affected evidence)

1. `847eb8a`: protocol and adapters. `4fc6f9c`: GPU-idle gating (three empty checks) and a 15 s GPU-process sampler.
2. Focused tests **13/13 PASS**. They cover:
   - seed isolation (W1, W2);
   - W1 sources identical to the WL56 server snapshot and environment hashes;
   - 60 reference images present;
   - WL56 dent signatures and dirty sets;
   - fixture determinism;
   - W2 input SHA256;
   - lighting/states equal to the WL22 lock;
   - frozen-directory GT equal to the design GT;
   - released 8DNA checkpoint SHA256.
3. W1 oracle (02:33–02:44 KST) and W1 Cycles ladder plus RNA-native bounce check (02:46–03:05) ran on an idle GPU.
4. **W2 attempt 0 failed at import** (03:07) and rendered nothing. `ednalib.init_upstream()` imports the 8DNA integrators, whose CUDA extension cannot be JIT-built with GCC 8.4. The chain's ERR trap did not fire inside a function, so the chain stayed `running` for about 11.7 h with the GPU idle. This was a chain defect.
   - Fix `354675c`: an identical Mitsuba initialisation without the neural-integrator import (variant, LoopRecord and VCallRecord flags unchanged); an 8DNA blocker record (the protocol's fallback); `set -E`.
   - The failure log is preserved as `w2_pt_attempt0_import_failure.log`.
5. The analysis stage failed in its CSV export (indexing bug, report-only). The JSON was already written. Fixed in `57c8dae` and re-run; the computed metrics are unchanged, and the first output is preserved as `analysis_attempt0`.
6. **Contention:** the sampler found foreign GPU processes only at 01:19–01:29 (WL66 smoke, before any evidence stage). There were none during any stage (`results/evaluation/67/contention.json`). The pre-evidence 4-spp Cycles smoke briefly overlapped WL66's smoke process; it is not evidence.

## Measurement

### W1 — physical quality and latency (dent; LabServer69 RTX 3090, driver 550.54.15)

Single-realization M1 and M2 (worst of two realizations); M3 and M4 per channel R/G/B (worst realization). Latency is edit-to-output for the four face-on views, warm process, median of two.

| Arm | M1 /12 | M2 max \|bias\| | M3 per-port rel L2 | M4 stationary-change rel L2 err | Latency (s) |
|---|---|---|---|---|---|
| Cycles 1 spp | 11 | 3.38% | 0.251/0.206/0.204 | 3.96/1.77/3.47 | 2.68 |
| **Cycles 4 spp** | **12** | 2.39% | 0.119/0.096/0.081 | 1.90/0.93/1.91 | **4.67** |
| Cycles 16 spp | 12 | 0.91% | 0.067/0.052/0.039 | 0.85/0.43/0.86 | 6.32 |
| Cycles 64 spp | 12 | 0.41% | 0.028/0.023/0.016 | 0.43/0.20/0.38 | 12.11 |
| Cycles 256 spp | 12 | 0.34% | 0.016/0.012/0.010 | 0.25/0.11/0.20 | 34.18 |
| Cycles 1024 spp | 12 | 0.11% | 0.009/0.007/0.004 | 0.122/0.055/0.091 | 127.44 |
| ReST-GI oracle **S** (scattered + ballistic) | **9** | **10.37%** | 0.034/0.049/0.105 | **0.077/0.032/0.047** | **189.81** |
| ReST-GI oracle F | 9 | 10.37% | identical to S | identical to S | 371.90 |
| Stale T (bin, negative control) | 7 | 33.9% | 0.083/0.146/0.473 | 0.80/0.92/0.99 | 11.80 |

- **QM_contract = 4 spp, 4.67 s.** QM_FNTO is **not reached within the predeclared ladder**: Cycles matches or beats S on M1–M3 from 64 spp, but not on M4 up to 1024 spp.
- **Reading M4.** The reference's own change noise is about half of Cycles-1024's single-realization change noise, so M4 is partly reference-limited for every arm. Post hoc, labelled extrapolation from the measured 1/√spp trend: Cycles would need roughly 3,000–6,000 spp (about 6–13 min) to match S's M4. This is not measured and not used for a verdict.
- **Signed change.** F/S slopes are 1.03–1.04 / 0.99 / 0.97, corr ≥0.9975. T slopes are 0.20 / 0.08 / 0.01. Stale K misses the amplitude, as in WL56.
- **Reproduction.** The WL56 two-replicate dent S biases are reproduced exactly (max |Δ| 0.0).
- **Rest and rigid.**
  - Cycles: 12/12 from 4 spp (rest and rigid).
  - S: rest 9/12; rigid 10/12 (max bias 8.6%).
  - GPU-resident rigid S, three updates: 1.06 / 0.69 / 0.68 s, excluding the 8.2 s one-time cache load and CSR build. Cycles rigid at 1 spp: 2.58 s, 9–10/12, max bias 4.1–4.6%, M3 about 0.22; at 4 spp: 4.33 s, 12/12.
- **Cost components.**
  - Cycles: reset + build 0.11 s. Each camera render costs about 0.6 s at 1 spp, which is fixed overhead (sync/BVH/kernel and EXR film I/O). At 1024 spp a render takes 17–46 s.
  - S (dent): package rebuild 188.73 s (TR); shared update 1.07 s (boundary 0.11, CSR 0.30, C 0.03, source field 0.15, K 0.01, solve 0.20, readout 0.19, other 0.10).
  - The oracle readout covers three fixed direction keys for every port. Each new view direction needs its own camera KC construction in the TR build, while each extra Cycles view costs one more render. Neither scaling was measured beyond the fixed four views.
- **Memory.** Oracle GPU allocated peaks were 4.10–4.14 GB (reserved ≤5.06 GB), plus 1.68 GB of K per domain on disk (WL56). Cycles VRAM was not logged. The W2 PT process reached 4.66 GB used (nvidia-smi). No logged allocation reached 20 GB.

### W2 — teaset T0→T3 (interaction ROI vs GT mean, WL22 metric)

| Arm | T3 ROI display MAE | Gain | WL22 recovery | Whole-image T3 PSNR | Update latency |
|---|---|---|---|---|---|
| RNA frozen | 0.1243 | 0.002 | no | 17.7 dB | 0 s (invalid output) |
| RNA refit | 0.1094 | 0.559 | yes (borderline) | 17.9 dB | 11,648 s rebuild (WL27, RTX 5080) + 6.8 s inference |
| 8DNA frozen | 0.1144 | −0.001 | no | 19.6 dB | 0 s (invalid output) |
| 8DNA refit | 0.1033 | 0.578 | no (1.281) | 21.4 dB | 6,354 s schedule (WL27, RTX 5080) + 100.7 s at 1024 spp |
| PT 16 spp | 0.130–0.131 | 0.95–1.00 | yes | 18.7 dB | 0.21–0.24 s |
| **PT 64 spp** | **0.076–0.077** | 0.97–1.00 | yes | 23.5 dB | **0.52 s** (load 5 ms) |
| PT 1024 spp | 0.020 | 1.00 | yes | 34.8 dB | 8.47 s |

- **QM:** 64 spp for every neural arm.
- **Cross-machine check passes.** The 4096-spp LabServer69 render vs the RTX 5080 GT has ROI display MAE 0.0101 (bound 0.0157) and ROI mean ratio 1.00001. The W2 quality comparison is therefore physically aligned.
- **Timing is only partially aligned.** Neural rebuild and inference timings are inherited from the RTX 5080 (WL27), because 8DNA inference cannot be re-executed on LabServer69. PT timing is local (3090), excludes the one-time Mitsuba JIT warm-up, and uses the offline Mitsuba renderer.
- The re-scored neural numbers reproduce the WL22 record exactly.

### Interoperability blockers on W1 (executed evidence)

- **RNA, training-view coverage.** RNA's native generator samples cameras on the upper hemisphere (`training_dataset/blender_render_dataset.py`, `'hemisphere'`: z in [0, 0.9999)), so the bottom-face query (−z) is never supervised. Gap faces (lower top, upper bottom) can be seen from outside only through the 8 mm side aperture. The maximum exterior viewing elevation is 10.1–58.0° (median 28.1°), so the face-on query is 32–80° away (median 62°) from every observable direction, and 0/100 ports come within 30°. The stationary receiver carrying the edit's signed change is therefore unobservable under RNA's data protocol.
- **RNA, native transport settings.** The generator uses `max_bounces 32` and `sample_clamp_direct 20`. Rendering the dent with only these two settings changed (same seed, paired) lowers the group means by **−15.5 / −9.1 / −0.2%** on the upper top and **−83.3 / −65.4 / −1.5%** on the lower bottom (R/G/B). The control render with reference settings and the same seed is within 0.05% of the reference. RNA's native targets would be strongly biased for this medium.
- **RNA, lighting.** RNA has distant (sun) light conditioning only, while W1's source is a near-field emitter (code citation).
- **Not executed, no verdict use:** the Transparent-BSDF G-buffer pass-alpha concern.
- **8DNA:** supervision is Mitsuba path sampling, and WL53 measured Mitsuba 3.5.1 float32 volpath skipping about 89 µm at null interfaces (−12% at σ_t 1000 m⁻¹). This is inherited evidence, not re-measured. The extension also cannot be built on LabServer69.

## Comparison table

| Baseline (source) | Scene / regime | Edit | Quality before → after edit | Reference qualification | Total update latency | Inference / readout | Offline cost | Memory | Comparability | ReST-GI verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| Cycles Full Recompute (Blender 3.5.1 CUDA, reference settings) vs ReST-GI oracle S/F (WL56 code, re-executed) | W1 two-slab medium, Cycles volume | Local dent, stationary receiver | Cycles 4 spp: rest 12/12 → dent 12/12. S: rest 9/12 → dent 9/12 (blue +3.9/+3.6/+10.4%) | 4×1024 spp, split-half floor 0.05–0.34%, WL53-validated | Cycles 4.67 s (4 spp) / 127 s (1024); S 189.8 s; F 371.9 s | Cycles per view 1.0–1.2 s at 4 spp; S readout 0.19 s for 3 keys | Oracle initial rest K 356–366 s per domain (WL56); Cycles none | S GPU 4.1 GB + 1.68 GB/domain K | 1 PHYSICALLY ALIGNED | **C: disadvantage** (Full Recompute faster and meets contract). Scoped: S better on M4 per-port change precision vs Cycles ≤1024 spp, while slower |
| Same | W1 | Rigid translation | Cycles 1 spp 9–10/12; S resident 10/12 (max bias 8.6%) | as above | S resident 0.69 s vs Cycles 2.58 s (1 spp) / 4.33 s (4 spp, 12/12) | — | S one-time 8.2 s cache + CSR | as above | 1 (rigid QM not predeclared beyond M1–M3) | **B: scoped computational advantage with insufficient fidelity** |
| Stale T (negative control) | W1 | Dent | 7/12 bin; change slope 0.01–0.20 | as above | 11.8 s | — | — | — | 1 | Control behaves as expected; not a method |
| ReST-GI learned P (WL59, inherited) | Single-domain medium (not W1) | Held-out dent | Fails directional/blue-delta contract | WL59 replicas | Intrinsic-only S 0.33 s (dent), no C/solve/readout | — | 420 s fit per seed (WL57 supervision inherited) | 231 MB state | 3 NOT COMPARABLE | D (physically ineligible, no coupled solve) |
| RNA (66b5b09) | W1 | — | — | — | — | — | — | — | 3 | D: blocked (coverage, native bounce/clamp bias, distant light) |
| 8DNA (released teaset) | W1 | — | — | — | — | — | — | — | 3 | D: blocked (Mitsuba volume reference, extension build) |
| RNA frozen / refit (WL22 outputs) vs Mitsuba PT | W2 teaset common light | T0→T3 rigid relation change | RNA 0.090 (T0) → frozen 0.124 / refit 0.109; PT 64 spp 0.070 → 0.077 | GT 2×32768 spp; cross-machine check passed | PT 0.52 s (3090); RNA refit 3.24 h (5080) | RNA 6.8 s full call (5080) | RNA dataset 2.35 h + training (WL27) | — | 1 quality / 2 timing | Neural baseline disadvantaged vs Full Recompute; ReST-GI D |
| 8DNA frozen / refit (WL22 outputs) vs Mitsuba PT | W2 | T0→T3 | 8DNA 0.072 (T0) → frozen 0.114 / refit 0.103 | as above | PT 0.52 s; 8DNA schedule 1.76 h (5080) | 100.7 s at 1024 spp (5080) | — | — | 1 quality / 2 timing | Neural baseline disadvantaged vs Full Recompute; ReST-GI D |
| TransGI (arXiv 2506.09909) | W1/W2 | — | — | — | — | — | — | — | 3 | D: no released implementation; no volume/transmission scope |

## Observation

- On W1 the reference renderer is cheap. The scene is small, so a few Cycles samples per pixel already average the 100-port groups below the 3% contract. The oracle's 2.5 mm K/C interface carries a systematic blue/transmission bias that no extra cost removes.
- The oracle's strength is low per-port noise after the solve, on the changed receiver in particular. Its weakness is systematic group bias plus TR construction cost.
- Both neural baselines are worse on W2 than 64-spp path tracing at 0.5 s, even before counting their hours-long refits. Their frozen outputs do not respond to the edit (gain ≈ 0).

## Interpretation

- **A (quality-matched measured superiority): none.**
- **B (scoped computational advantage, insufficient fidelity):** the W1 rigid resident update (0.69 s vs ≥2.6 s), with failed contract and higher group bias. The M4 precision of S over single-realization Cycles ≤1024 spp is a quality dimension, not a cost advantage: S is slower there.
- **C (no demonstrated advantage):** W1 dent against quality-matched Full Recompute, which meets the contract at about 1/40 of S's latency.
- **D (blocked or incomparable):** ReST-GI on W2 (no rough-conductor K); the WL59 learned K on W1 (ineligible, never composed); RNA and 8DNA on W1; TransGI everywhere.
- **Strongest valid ReST-GI advantage:** stationary-receiver per-port signed-change error 0.077/0.032/0.047 vs Cycles-1024's 0.122/0.055/0.091 (W1 dent, aligned), at 1.5× the latency.
- **Strongest disadvantage:** at the required contract, Cycles 4 spp reaches 12/12 in 4.67 s, while S reaches 9/12 in 189.8 s.
- This does not falsify the reuse hypothesis. It shows that W1 cannot support a cost claim: physically exact path tracing of this small fixture is cheaper than its oracle K, so a learned K would have to beat a few-second path-traced update at contract quality. Scene scale, view count and update frequency were not varied.

## Unresolved Question

- How do the costs scale with scene size, view count and update frequency, where path-tracing cost grows per view and per pixel while a solved boundary state could be read out repeatedly? This batch measured only four fixed views on one small fixture.
- Is there a configuration where a ReST-GI K exists and an external neural baseline runs under its native semantics? W2 has no ReST-GI K, and W1 is outside RNA's and 8DNA's semantics.
- Could TransGI run if an implementation were released? Its paper excludes transparent objects and deformation.
- How much of the M4 advantage is the oracle's own accuracy rather than reference-noise cancellation? A denser reference would resolve it.

## Architecture Verdict

No architecture is adopted or changed. **ReST-GI quality-matched superiority: NOT SUPPORTED on any executed workload** (W1 class C, rigid scoped B; W2 D). External neural baselines (RNA, 8DNA) are dominated by quality-matched Full Recompute on W2. TransGI comparison: BLOCKED. The direction is the user's decision.

## Reproducibility

- **Commits:** protocol/adapters `847eb8a`; chain gating `4fc6f9c`; W2 init/blocker/errtrace fix `354675c`; CSV fix and report exporter `57c8dae`.
- **Server:** LabServer69 `~/rest_bsup67` (repo at each commit; `w1oracle` root with read-only links to `~/rest_wl56/cache` and `out/cycles`). Driver 550.54.15; torch 2.3.1+cu121, Mitsuba 3.5.1, Dr.Jit 0.4.6 (8dna26 venv); Blender 3.5.1 e1ccd9d4a1d3 CUDA.
- **Seeds:** W1 ladder `670000 + 1000·state + 10·level + r`; bounce check 679001; W2 `6700000 + 100000·state + 1000·level + 100·r` (chunk seeds following); cross-machine check 6790000.
- **Artifacts:**
  - `results/baseline_sup67/`: analysis JSON/CSV, blockers, tests, raw timings, launcher logs incl. failed attempts and the GPU sampler log.
  - `results/evaluation/67/`: reviewer figure `01_quality_vs_latency.png` (the stale-T point at 33.9% sits under the legend; its value is in the table), contention screen, SHA256 manifest.
  - Raw images, oracle packages (~5 GB) and per-run NPZ remain on the server.

```sh
# on LabServer69, fresh root: clone repo at 57c8dae into ~/rest_bsup67/repo, create w1oracle links and w2in copies (sha256.json), then
setsid -f bash repo/experiments/baseline_sup67/chain.sh > launcher/chain.log 2>&1 < /dev/null
```

No later batch is authorised or recommended.
