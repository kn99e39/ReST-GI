# Worklog — Exact Selective K Recompute and Dynamic Update Cost Headroom (2026-10-28)

**Authorisation:** the user's WL56 brief. Work directly on `Selective_Recompute`; no branch/worktree, production change, new learned representation, training, selector, or WL57. **Protocol:** `experiments/fnto_sel56/protocol/sel56.json` (`fnto_sel56_v1`).

**Completion answer.** Exact optical-domain selective recomputation reproduces full recomputation on the tested local dent, including the changed radiance of an unchanged receiving domain. Same-sample T/KC operator content is bit-identical (serialized metadata differs); the maximum F/S relative L1 output difference is **1.55e-8**. Independent clean-domain samples agree within the predeclared uncertainty rule. Actual complete dent updates take **371.40 s full versus 188.93 s selective**, a **1.966× / 49.13%** saving for this physical Monte Carlo oracle.

This is **not full physical-GI success**: both F and S pass **9/12** primary face-on channel tests, with blue residuals of **+3.87%, +3.57%, +10.27%** on the lower top, upper bottom, and lower bottom. The speed comparison is at the same approximation and measured accuracy, but that approximation fails the complete 3% physical-quality target. Do not describe it as a demonstrated speedup at the required physical fidelity.

After selection, the changed domain still costs **175.50 s of TR construction**, **92.89%** of the selective update; its complete new package including setup, serialization and checks costs **183.71 s**. This establishes oracle cost headroom, not a Rule–State method or a learned-GI advantage. No representation is adopted and no next batch is started.

## IMPLEMENTATION FACT

### Sources and provenance

- Starting HEAD verified as `11c3ac6ed100fc409692e2801e9abc6d43461807`, clean, on `Selective_Recompute`.
- During setup, four user-directed central-document commits arrived: `9e393db`, `d381b76`, `2b3baa2`, `495ce97`. Their diffs were reviewed. They adopt dynamic transport validity/reuse as the objective, keep Rule–State unverified, and specify Full/Selective/Stale and a future matched TransGI comparison. They do not change WL53–55 implementations. The WL56 preregistration commit is a descendant of these changes.
- WL50–55, the FNTO architecture contract and baseline-role documents were inspected. WL55 source, original operators and Cycles references exist locally/on LabServer69. The frozen `p53_tracer.py` and `k55_ops.py` hashes match the original server copies exactly. Full dependency hashes are in `out/environment.json`; original WL55 operator file SHA256 values are in `out/initial_cache.json`.
- No WL53–55 implementation or historical worklog/result was edited. No K44, DASQ, learned C, production FNTO, or neural parameters were trained or changed. No Git branch/worktree was created, switched or deleted.

### Fixture and representation, fixed before results

Two 10×10×1 cm medium slabs, centred at z = −9/+9 mm (8 mm air gap). Material is unchanged from WL53: index-matched interfaces, homogeneous sigma_t = 1000 m^-1, isotropic phase, RGB single-scattering albedo (0.999, 0.99, 0.9).

- **Rest:** both slabs flat.
- **Dent:** upper slab's top is displaced by the existing WL53 cosine-squared dent, depth 5 mm, radius 20 mm, centre (0,0); its bottom is fixed. Lower geometry is bit-identical to rest.
- **Rigid control:** upper slab translated (+3.7, −2.1, +2.0) mm; neither intrinsic domain changes. This directly tests a translation, not a new arbitrary-rotation experiment.
- **Source:** fixed downward, unit-radiance, one-sided absorbing Lambertian emitter, 20×20 cm at z = 54 mm, 2.5 mm tessellation. No sun.
- **Why this source:** WL55's separately traced current-geometry exact sun term already contains intrinsic transport. Giving that term to the stale arm would partly update the very response meant to stay stale. Here **all direct and indirect medium input passes through the arm's K**. External illumination consists only of source rays intersecting the current boundary, not target images.

All arms use uniform **2.5 mm** medium K/C cells, **64 outgoing / 16 incoming** equal-projected-angle bins, **2048 photons per input cell** (125,829,120 photons per domain construction). No CAK rule is used. Camera KC has three direction keys (`top`, `bot`, `otop`) in every package. Cached WL55 rest packages were restricted to these same three keys; every freshly built package uses the same keys. K is entry→first exit; every subsequent air segment, including same-domain re-entry at a dent, belongs to C.

The source has 1 cm ownership ports with 2.5 mm geometry tessellation; it is an analytic emitter, not a third optical-domain K. Both medium grids and directional bases are identical across arms.

### Arms and exact update boundary

| Arm | Intrinsic work after edit | Current work shared in definition, independently timed |
|---|---|---|
| F | Actually reconstruct both complete domain T/KC packages | Boundary geometry, CSR transfer, C, incident source field, global coupled solve, camera readout |
| S | Reconstruct upper package only; reuse lower cache | Same current state updates; no old solved radiance reused |
| T | Both rest packages retained | Current geometry/C/source/solve; outgoing-bin readout for a fair stale comparison |

The dirty set is supplied by exact intrinsic geometry/material signatures: dent `{upper}`, rigid `{}`. It never reads reference images. A stationary receiver can change radiance while its intrinsic K remains valid; its incoming/outgoing state is solved again globally.

F/S same-sample packages use matching domain seeds. The full baseline actually rebuilds the unchanged domain instead of multiplying a single-domain time. Independent controls swap only the clean-domain seed while retaining current dirty K. The expensive primary order is F0, S0, S1, F1; rigid F reconstructs both domains once. Every run records actual reconstruction flags and hashes.

**Readout:** F/S physical primary is scattered KC + current-geometry ballistic KB. For stale comparison, use **bin readout for all F/S/T**: a rest camera KC is not valid on a changed local frame. The stale `nee+KB` output is retained as a diagnostic, not used as evidence of accurate stale shading. Bin readout includes ballistic transport already; KB is not added a second time.

## MEASUREMENT

### Tests, stability and reuse identity

| Check | Result |
|---|---|
| WL56 focused tests | 6/6: ownership/correspondence, emitter area/orientation, shared global fixed point and zero/superposition, conservative refinement, current C/KB, deterministic masks |
| Historical WL53 | 9/9 |
| Historical WL55 | 10/10 |
| Historical WL54 | 9 performed tests passed; runner printed 10/10 including one skipped mask check. The actual mask check was then run separately and passed (1/1). Do not count the skip as performed evidence |
| Same-sample F/S | All 3 comparisons pass (two dent seeds, one rigid seed); max relative L1 1.55e-8 across radiance and complete flux-state fields |
| Package identities | Recomputed clean F equals cached clean S; reconstructed dirty F equals reconstructed dirty S, exact content hashes in both seeds |
| Independent clean K | Every reported group agrees under the registered 3SE + 1e-4 criterion; largest primary face-on difference 0.81 percentage points (all-camera maximum 2.01 points) |
| Stability | All stored solves converge, 45–51 iterations; rho 0.413–0.779, max fixed-point relative residual 8.98e-8 |
| Energy/ownership | max K column sum 0.997681; max leak + wrong-side rate 6.28e-7; C column + escape closure exact, zero back hits; no damping/clamping or overlapping path family |

The initial failed focused run (5/6) is preserved. Its source mesh orientation was corrected without changing physical source extent, location or radiance.

### Independent reference and detectability

Official Blender **3.5.1**, build **e1ccd9d4a1d3**, **CUDA**, WL53 volume material/integrator settings: 1024 spp, 1024 bounces, no adaptive sampling, denoising or clamping, `GAUSSIAN 0.01` point sampling; 0.5 mm pixels, 220² images. Four seeds per state, 60 images total. The user's CUDA validation is inherited; no GPU smoke revalidation was added. Every render has a build/device manifest.

Primary face-on cameras are top, gapdown, gapup, bottom. Both original interior masks and amended centre-ray/≥90% coverage masks are reported. Historical rule remains `|group bias| <= 3% + 3SE`, >=5 ports and >=2% of the selected camera signal. Reference SE is estimated from independent **group seed means**, not assumed independent pixels/ports. K SE uses two independent samples. Intervals are descriptive ±3SE, not guaranteed 95% intervals from two K seeds.

The stationary lower top passes the preregistered reference detectability gate in every channel:

| Receiver region | Reference dent−rest change R/G/B | L2 signal / reference noise R/G/B |
|---|---|---|
| Whole lower top (100 ports) | +2.228 / +4.056 / +2.186% | 72 / 150 / 189 |
| Fixed radius-25 mm disc (16 ports) | +6.548 / +16.892 / +88.658% | 91 / 171 / 202 |

The disc was fixed before results; it does not replace the whole-face primary population. No spp escalation, source change or deformation enlargement was needed.

### Physical accuracy: full and selective agree, but the target fails

**Dent, amended masks, bias against Cycles (%).** F and S are equal to the printed precision. All groups below have 100 ports.

| Camera / medium face | F/S scattered + ballistic, R/G/B | F/S bin, R/G/B | T bin, R/G/B |
|---|---|---|---|
| top / upper top | −0.534 / −0.248 / +0.898 | −0.561 / −0.188 / +1.175 | −0.429 / −0.156 / +1.196 |
| gapdown / lower top | −1.370 / −0.216 / **+3.869** | −1.320 / −0.181 / +3.585 | −3.171 / −3.825 / +1.480 |
| gapup / upper bottom | −1.585 / −0.815 / **+3.574** | −1.443 / −0.483 / +9.353 | −3.528 / −4.932 / −0.155 |
| bottom / lower bottom | −1.108 / +1.663 / **+10.272** | −0.377 / +4.556 / +33.758 | −2.087 / +1.651 / +33.520 |

The three blue misses exceed uncertainty: F/S K SE is 0.0001 / 0.0140 / 0.0989 points on those faces; reference SE 0.0092 / 0.0387 / 0.0278 points. They are not rescued by source or C seed noise (below). The known disc passes all three channels (NEE+KB −2.019 / −1.524 / +1.341%); that local result does not erase full-face failures.

**All ruled face-on channel tests, after A2 emitter readout correction:**

| State / arm / readout | Amended mask | Original interior mask |
|---|---|---|
| Rest S / scattered+ballistic | 9/12 | 15/17 |
| Dent F or S / scattered+ballistic | **9/12** | 15/17 |
| Dent F or S / bin | 8/12 | 14/17 |
| Dent T / bin | 8/12 | 14/17 |
| Rigid F or S / scattered+ballistic | 10/12 | 16/22 |

These are **group×channel counts**, not counts of independent scenes or groups. The original unit admits partially covered source ports and differs in signal/ruling population; it includes trivial direct-emitter checks. The masks must not be compared as if they selected identical medium regions. Every camera/group/channel, including unruled/oblique and fixed-disc diagnostics, absolute error, relative L2, p99 error and SE is in `final/all_groups.csv` and `final/analysis_readout_v2.json`.

### Stationary receiver recovery and stale K

Fair comparison uses the same current-frame bin readout in all arms. On the full stationary lower top, predicted dent−rest change has:

| Arm | Slope R/G/B | Correlation R/G/B | Predicted mean change R/G/B |
|---|---|---|---|
| F or S | 1.040 / 0.996 / 0.975 | 0.9988 / 0.9994 / 0.9992 | +2.367 / +4.123 / +2.173% |
| T | 0.199 / 0.077 / 0.0098 | 0.9974 / 0.9981 / 0.9966 | +0.475 / +0.331 / +0.022% |

S passes the predeclared slope/correlation criterion. In the fixed disc its slope is 1.019 / 0.985 / 0.972, correlation >=0.9969. Stale K largely misses the amplitude even though correlation is high. Its relative L2 change error is 0.80 / 0.92 / 0.99 versus S 0.064 / 0.029 / 0.042 on the whole face.

**Absolute ranking is not uniform:** T can be closer to Cycles in blue through cancellation of the existing positive interface/readout bias (upper bottom bin −0.16% T versus +9.35% F/S). This is not valid stale transport. Distinguish detected K invalidation, correct signed-change prediction, and absolute physical accuracy.

### Sampling and readout attribution

- With fixed K, a second external-field seed changes primary NEE+KB group biases by at most **0.269 points**; a second C seed changes them by at most **0.022 points**. These cannot explain the 3.6–10.3% blue residual. These are separate diagnostics, never extra tolerance for the original gate.
- Changing **only** camera evaluation from current KC+KB to 64-bin output raises lower-bottom blue error from +10.27% to +33.76%. Binned readout is therefore a substantial additional limitation for absolute transmitted-blue radiance. It remains the consistent stale comparison.
- The remaining KC+KB blue floor is unresolved between the fixed spatial/directional boundary approximation and intrinsic/readout estimation bias. This batch does **not** establish a unique cause or introduce a new representation to repair it. WL55's exact-current-sun path did not test this entire broad-incident-field-through-K regime.
- Diagnostic `otop` is an additional fixture limitation: its close orthographic plane yields **1200/48400** medium first-hit backfaces in rest/dent and **1000/48400** in rigid, indicating rays beginning inside a medium. Its outgoing-boundary port comparison is not applicable for those pixels. Every primary face-on camera has **zero** such hits. The poor oblique rows, including −100% upper-bottom values, remain visible in the CSV and are not interpreted as a new K failure. No unregistered camera replacement was selected.

### Actual cost and memory

RTX 3090 24 GB on LabServer69, driver 550.54.15; torch 2.3.1+cu121, Mitsuba 3.5.1 intersector, Dr.Jit 0.4.6, numpy 1.26.4. GPU process list was empty before every primary update. All GPU timings are synchronized; each F/S update is a separate process. Process/CUDA startup is excluded from warm update timing. Original v1 diagnostics (rho, validation) are separately timed and excluded. Component sums include other bookkeeping/state transfers.

| Warm update | Measured times (s) | Median (s) |
|---|---|---|
| Dent F, reconstruct two complete packages | 370.691, 372.103 | **371.397** |
| Dent S, reconstruct upper only | 188.719, 189.148 | **188.933** |
| Dent T, raw rest cache + current state | 9.054, 9.055 | 9.054 |
| Rigid F, actually reconstruct both | 373.818 (one sample) | 373.818 |
| Rigid S, cache load + current state | 9.192, 9.242 | 9.217 |
| Rigid S, already GPU-resident K, 3 complete current-state updates | 1.049, 0.685, 0.683 | **0.685** |

The dent S/F ratio is 0.509 and meets the preregistered <=0.75 efficiency criterion **for the tested fixed oracle**. Rigid disk-cache speedup is 40.56×, dominated by removing two unnecessary oracle constructions. The GPU-resident experiment is an actual rigid update, not an inferred dent timing.

**Dent component medians (s), independently measured:**

| Work | F | S |
|---|---|---|
| Complete package access/rebuild/checks | 370.328 | 187.847 |
| Current boundary | 0.111 | 0.112 |
| CSR conversion / host→GPU | 0.295 | 0.297 |
| Explicit C | 0.026 | 0.035 |
| 64M source rays / current incident field | 0.146 | 0.146 |
| K applied to external field | 0.007 | 0.009 |
| Coupled solve | 0.202 | 0.201 |
| Camera NEE/bin/ballistic readout | 0.186 | 0.187 |
| Other bookkeeping / state transfer | 0.097 | 0.099 |

- Each rebuilt domain TR takes **175.3–178.9 s**; this is not a speedup comparison with WL55's 380 s: WL55 used five KC directions and partly contended runs, whereas WL56 uses three and idle GPU start conditions. Every WL56 arm is matched. GPU occupancy was sampled before updates; unrelated host CPU/IO activity was not continuously instrumented.
- K file size is **1.683 GB/domain**, including about 0.531 GB of camera KC. Two active domain packages cost about **3.37 GB**. Retaining both old and current upper K adds another 1.68 GB per sample. Experimental duplicate F/S checkpoints and seed replicates are evidence storage, not required runtime cache.
- Source-cache loads were 0.415/0.423 s, conversion writes 0.991/0.989 s; content checks are separate CPU work. These are warm filesystem/page-cache measurements, not a forced cold-disk test.
- Initial rest packages came from WL55; actual rigid F reconstructed the same canonical rest operators independently: **356.48 s TR total**, **366.05 s build/setup/write total**, then identity/state-update work. This measures initial intrinsic construction cost; a wholly cold process/import/kernel-startup scene pipeline was not benchmarked.
- An unchanged 1.68 GB cache load takes about 0.34 s, and content rehash about 3.3 s. Hashing both packages accounts for about 6.6 s in a no-TR update. This conservative bookkeeping is measurable overhead, not an intrinsic transport requirement. GPU-resident initial CSR setup costs 0.337 s and initial cache/checks are charged separately.
- Peak allocated GPU across recorded runs is <=**4.45 GB**, CPU RSS <=**5.82 GB**. Reserved-GPU peaks, actual file sizes, hashes, per-domain TR stats, allocation/transfer and all individual times are recorded in the JSON. Persistent and peak memory are distinct.
- A2 direct-emitter assignment adds roughly **20–22 microseconds** of CPU array assignment in post-processing; no medium/flux field changes. Whole expensive update timings were not repeated after this camera-only fix. The quoted primary end-to-end costs are preserved v1 measurements, with this small readout omission explicitly qualified.

## OBSERVATION AND INTERPRETATION

**Gain A is demonstrated as an oracle baseline:** lower intrinsic K is valid and bit-identical while the lower receiver's GI changes. Selecting whole invalid optical domains preserves full-recompute output. It avoids one unnecessary complete physical construction, not a learned reconstruction campaign. Same-shaped rest slabs permit other sharing optimizations; this is the requested full-per-domain baseline, not a claim of an optimal all-shape library.

**Gain B is not demonstrated:** even after exact selection, the changed upper domain spends 175.50 s in TR (92.89% of S). Eliminating only that computation, with all other measured work unchanged, would leave **13.44 s**; this is an analytic upper-bound exercise, not a measured fast method. Full new-package work including checks is 183.71 s (97.24% of S), but some of it is avoidable serialization/hashing rather than transport structure.

**Break-even (post hoc A1 accounting clarification, original curve retained):** the original illustrative curve double-counted dirty identity hashing. The corrected disjoint fixed-raw-cache model is

    T_F = 2k + 1.1433; T_S = k + 5.2185 seconds,

where k is a hypothetical *complete new package* cost. A >=20% saving requires k >=**7.17 s**. Keeping a 1.68 GB cache/identity pipeline fixed is not a prediction for a smaller learned K, and millisecond creation is incompatible with those unchanged buffers.

For a hypothetical compact resident representation, with common state cost H and selection overhead O,

    T_F = 2k + H; T_S = k + H + O;
    >=20% saving iff k >= (0.25H + 1.25O)/0.75.

Using the **rigid** measured H=0.6848 s only as an illustration, and assuming O=0, gives k>=0.2283 s. At k=1/10 ms, the model yields only 1.0015×/1.0144×. Neither H nor O has been measured for a compact learned dent model. These curves show why expensive-TR skip ratios cannot establish neural-GI speedup. If K becomes cheap, incident-field generation, C, solve, readout and cache policy matter; C alone is not currently the dominant shared term.

**Scaling:** measured only N=2, dirty counts 0 and 1. N=4/8 and other invalidated fractions are labelled algebraic projections with unknown H(N), not constructed-scene timing. No asymptotic, real-time, heterogeneous-asset, or learned-operator scalability claim follows.

## Future four-case compatibility (decision evidence, no architecture selected)

| Case | Contract implication | Evidence here |
|---|---|---|
| Rigid | Material-coordinate K stays reusable; current geometry, source, C and radiance must update | Translation tested; rotation frame controls inherited, not newly broadened |
| Local deformation | Complete changed-domain K invalidates; stationary receivers can reuse intrinsic K but must update radiance | Directly tested, same-sample equality and physical differential detected |
| Global bend/twist | A shared rule would need global geometry-dependent transport; no assumption of local/sparse/low-rank changes | Not tested; WL53–55 dense/global bend evidence preserved |
| Self-contact/contact changes | Must distinguish exposed air re-entry in C from changed medium connectivity/interfaces in K; domain merging/repartition may be required | Not tested; no frozen-topology or fixed-support generality claim |

A future model's target must be statistically resolvable aggregated transport/current-scene response, not individual noisy fine-cell entries. Preserve linearity at fixed geometry, positivity, passivity, disjoint K/C ownership, material correspondence and conservative mapping. The present broad-field blue floor and precise camera readout remain part of its physical-quality obligation.

## ARCHITECTURE VERDICT

| Item | Verdict under this fixture/protocol |
|---|---|
| Exact selective recomputation correctness | **SUPPORTED:** F/S package identity and global output equivalence; independent-seed control passes |
| Physical GI accuracy | **NOT SUPPORTED** at the complete primary 3% contract: 9/12 dent channel tests, blue residuals persist in F and S |
| Rigid K reuse | **CONDITIONALLY SUPPORTED:** exact canonical-coordinate translation control and solved-state reuse; no new broad rotation/numerical-pose claim |
| Local non-rigid invalidation | **SUPPORTED** on the upper dent: stale K misses response amplitude; no claim that all invalidations are sparse/local |
| Stationary receiver recovery | **SUPPORTED for the primary signed-change criterion**; full-face absolute blue GI remains NOT SUPPORTED |
| End-to-end update cost benefit | **CONDITIONALLY SUPPORTED:** 1.966× at identical oracle approximation; not demonstrated at the failed complete physical target or for a neural method |
| Memory/caching cost | **CONDITIONALLY SUPPORTED as measured accounting:** ~1.68 GB/domain; peak 4.45 GB GPU. Deployable scalability not shown |
| Additional Rule–State headroom | **CONDITIONALLY SUPPORTED as oracle cost headroom only:** 175.5 s dirty TR remains; Gain B feasibility/accuracy/speed are **INCONCLUSIVE** |
| Applicability beyond tested geometry | **INCONCLUSIVE:** one homogeneous two-slab dent and translation, no heterogeneity/bend/twist/contact validation |
| TransGI comparison readiness | **NOT SUPPORTED:** no compatible external benchmark run; required alignment below |
| Learned-K integration readiness | **INCONCLUSIVE:** nothing learned, baseline physical blue floor unresolved |

### Unresolved questions material to the next decision

1. Can a future within-domain rule/state mechanism reproduce a physically credible broad-field response, including the blue and camera-direction limits, before claiming the measured oracle headroom as a realizable benefit?
2. For a cheaper K and its actual storage/transfer contract, what remains of domain-selection savings versus current C, source, solve and readout cost? The current 1.68 GB oracle-cache threshold cannot answer this for a neural model.
3. How should ownership/correspondence change under medium self-contact/connectivity changes? No fixed-domain answer was selected here.

A TransGI comparison requires compatible medium/material support, identical scenes and edits, independent quality/temporal reference, hardware, resolution, offline supervision/training versus online update/inference, solve/readout, memory, initialization and cache boundaries. No comparison or novelty claim was made.

## REPRODUCIBILITY AND FAILURES

**Primary code:** `3b3ce4c1801ceca5aeac016bc45536c0c41aa1dd`; initial protocol/implementation `2b47b4c`. Document-only parent decisions are listed above. Source remains in `experiments/fnto_sel56/`; inherited directories are unchanged.

**Pre-primary corrections:** `d8aa305` emitter triangle-order fix; `6616d00` Blender import-path/exit-code wrapper; `f3a1fe5` inherited renderer OUT/LOG setup. The first two renderer attempts produced no images and remain in the logs. `ffe55c4` group analysis; `374390d` fail-fast ordered primary launcher; `671440a` actual historical mask test and server waiter; `60f8bf9` resident timing; `2f09c48` complete wall-time accounting; `552726a` A0 variance separation; `3b3ce4c` occupancy/software recording. All precede primary K runs. Eight complete fresh primary operators (about 13.47 GB total) were actually constructed; historical source/cache files are separate.

**Post hoc:** A1 (`0fa11d8`) fixes only sensitivity accounting and report/export; A2 (`c1471e7`, `4f165c6`) fixes analytic emitter camera readout, preserves original medium/flux fields exactly, audits the invalid diagnostic camera, and repeats affected tables. Primary `analysis.json`, original runs and their timing remain unchanged. `analysis_readout_v2.json`/`readout_v2/` are distinct. A1/A2 are not new confirmatory experiments or relaxed gates.

**Seeds:** rest lower 5561/5562, rest upper reversed; dent upper 5565/5566. Primary external field 5661 (fixed across K samples); source sensitivity 5662. C 5681 (diagnostic 5682). KB 5700 + 13·domain + direction index. Cycles rest/dent 5601–5604, rigid 5611–5614. Exact counts/identities in protocol and manifests.

**Commands on LabServer69** (`~/rest_wl56/repo/experiments/fnto_sel56/`):

```sh
run.sh tests/test_s56.py ~/rest_wl56/out/tests/focused.json
bash launch_reference.sh
bash launch_when_ready.sh
# The server waiter requires reference.status=done and reference_gate.proceed=true.
# launch_primary.sh: audit, masks, prepare, rest, actual F/S/T, verify,
# independent clean-seed controls, actual rigid F/S, replay, resident, noise, analyze.
run.sh s56_readout_audit.py
WL56_PRED_ROOT=$HOME/rest_wl56/out/readout_v2/runs \
WL56_ANALYSIS_OUT=analysis_readout_v2.json run.sh s56_analyze.py
run.sh s56_report.py report
run.sh s56_report.py export
```

Server chains use **server-side setsid/nohup, redirected stdin/stdout/stderr and status/failure files**, independently of this chat/SSH. References finished 18:34 KST; the complete primary chain reached analysis at 19:03 KST despite the chat being inactive. They were observed `done`, no remaining WL56 primary job. Scratch/output exclusively `~/rest_wl56`; lab-managed infrastructure and other users' files untouched.

**Artifacts:** `results/fnto_sel56/out/` and `launcher/` contain original measured JSON/logs; `results/fnto_sel56/final/` contains all-group CSV, original/corrected analyses, compact predictions, per-seed physical boundary references, raw SHA256 manifest, readout audit, cost summary and figures. `results/evaluation/56/` contains reviewer figures plus provenance manifest. Full operators, source images, complete flux states and corrected copies remain on LabServer69. Do not confuse original evidence, post hoc camera correction and algebraic cost projections.

**Independently measured here:** F/S reconstruction/equivalence, physical references and signed edit response, actual domain cost/cache/occupancy/memory, source/C noise controls, tests rerun as specified. **Inherited:** WL53 volume-reference physical validation (H-function/furnace/history/reciprocity), CUDA-versus-fork equivalence approved by the user, and previous WL55 architecture evidence. No learned generalization or neural reconstruction result is inherited as evidence for this oracle batch.

The bounded batch is complete. No later batch is authorised or recommended automatically.
