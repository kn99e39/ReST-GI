# Worklog — 8DNA Dynamic-Scene Reuse and Transport-Update Baseline Investigation (2026-10-24)

**Status: DRAFT.** Sections A, C (code/paper columns), E, F and H are written from code, paper and geometry evidence.
The measured columns of B, C and D are **pending** the server chain (protocol `8dna_dyn_v1`, LabServer63, after the WL42
FNTO chain). Nothing in this draft is a measured 8DNA edit result yet.

User brief: "Independent Baseline Investigation — 8DNA Dynamic Scene Reuse & Transport Update" (2026-10-24). It runs beside
FNTO (WL42); FNTO, its contract, the historical RNA / Surface-Owned / SPT-1 baselines and every earlier worklog are untouched.
Branch `baseline/8dna_dynamic_reuse` (worktree), started from `b52ee7d`.

Labels used throughout: **[PAPER]** claim of arXiv 2604.25129v1 (section/figure/table); **[IMPL]** fact of the official code
`github.com/lwwu2/8dna26` @ `4a2157c` (file:line); **[REPO]** earlier ReST-GI evidence; **[MEAS]** measured in this session;
**[INF]** inference, with the reason.

---

## 0. Session record

- **Placement decisions (user, 2026-10-24).** Asked whether 8DNA GPU work may run on the PC while the WL42 FNTO chain runs on
  LabServer63 (standing rule: no PC/server parallel), the user answered "서버 측에서 segfault가 나는 원인을 분석하고 그것을
  먼저 해결해봐" (diagnose and fix the server's Mitsuba segfault first). After the fix, asked where and when to run the main
  8DNA renders, the user chose **after the FNTO chain, on the server**.
- **Server segfault: diagnosed and fixed** (§H.2). Root cause: OptiX user-space libraries of driver 570.195.03 under a
  595.71.05 kernel driver. Separate exit-time crash: DrJit 0.4.6 `jit_shutdown()` leak-report logging.
- **Study.** A read-only workflow (6 readers, 2 adversarial verifiers, 1 completeness critic) covered the paper, the code,
  the repo's 8DNA history (WL19–30), the Lego edit protocol (WL35–42), the RPI paper and the living documents. Its reports
  are session artefacts; every load-bearing claim below was re-checked against code or paper text.
- **Lego geometry check** [MEAS, numpy on `lego_mesh.npz`, server CPU] (§E.1).
- **Protocol `8dna_dyn_v1`** written before any render of its scenes (`experiments/8dna_dynamic_baseline/protocol/dyn_v1.json`).
- **Code review before any render** (3 lenses: API, protocol consistency, metrics/orchestration; adversarial verification):
  18 findings confirmed and fixed, 9 rejected. Two would have stopped the chain at the design stage (the seal OBJ contains a
  closed cuboid pedestal filling the AABB footprint up to y = 0.1218, so the in-envelope box now sits on the pedestal top
  beside the figure; the first E2 T0 spot lay on the cloth's draped edge). Others: the E3 bunny's albedo grid did not move
  with the instance (Mitsuba media ignore instance transforms); receivers now exclude box-filter footprints touching other
  classes (1-pixel erosion); paired-seed noise estimators; SSIM defined without scikit-image; parameter checks that stop the
  chain; declared 5-update median. The protocol file records this revision. Commits `c1bcc81`, `56112b4`.
- **Synthetic CPU test of the evaluation** (`test_eval_synthetic.py`, server CPU): fabricated inputs with known answers are
  classified TRACKED / NOT_TRACKED as constructed; it caught a classifier bug (a fully stale, zero-variance change was
  labelled NOISE_LIMITED instead of NOT_TRACKED).
- **No GPU dry run was possible before FNTO finishes**: beside the 11 GB FNTO training only ~1.2 GB is free; dry renders
  ran out of memory at DrJit initialisation and texture allocation, and a geometry-only design pass stalled on DrJit cache
  flushes. The chain therefore runs a full dry pass (32², ≤ 16 spp) right after FNTO and starts the real stages only if
  every dry stage passes.

---

## A. Implementation facts

### A.1 What is learned and what stays explicit

| item | learned / explicit | evidence |
|---|---|---|
| per-asset model `EightDNA`: albedo(xi,wi) + spline flows p(xo\|xi,wi), p(wo\|xo,xi,wi); triplane 3×8×64², two cubemaps 6×32²×8; **669,321 parameters (2.68 MB)**, identical layout in all 11 released checkpoints | learned, persistent | [IMPL `models/eight_dna.py:16-88`, `configs/default.yaml`; checkpoint scan] |
| conditioning inputs: entry point xi and direction wi in the asset frame only; no light, pose, part or scene input | — | [IMPL `eight_dna.py:57-68, 204-206`] |
| the real asset (mesh, BSDF, medium) is still in the render scene: primary hits, first-vertex BSDF, self-visibility ray, shadowing of other objects | explicit | [IMPL `integrator.py:134, 154-163, 182`; `eight_dna.py:199-200`] |
| first-vertex ("direct lobe") scattering | explicit, real BSDF, current visibility | [IMPL `integrator.py:154-163`; PAPER §3.1 "separating the direct scattering"] |
| everything with ≥2 events inside the asset (SSS, transmission, internal reflection, part-to-part reflection, self-occlusion) | learned (`bounce: 2`) | [IMPL `path_sampling_dataset.py:126-127`; PAPER §2 "all scattering events within the asset"] |
| exit of the learned lobe | a sampled direction from the AABB centre, **projected onto the asset AABB**; emitter sampling and path continuation start from that AABB point in the current scene | [IMPL `eight_dna.py:229-241, 309-326`] (the paper's App. A.3 projects back to the asset surface; code ≠ paper here) |
| everything outside the asset | ordinary Mitsuba path tracing of the current scene | [IMPL `integrator.py:103-215`] |

Training [IMPL `utils/dataset/path_sampling_dataset.py`, `utils/light_transport.py`; PAPER §3.3, Tab. 3]: the isolated asset
only (one `instance0`, no emitters evaluated); rays line-sampled from the unit sphere; 128⁴ = 2.68×10⁸ candidates per epoch,
30 epochs × 8,192 steps × 32,768. Lighting never enters training, so relighting needs no re-bake. Paper Tab. 3: 1.78 h
(surface) / 2.23 h (volumetric) pre-bake, training GPU not named; [REPO WL27] teaset re-bake 1.76 h on the RTX 5080.

### A.2 Questions of brief §3A

| question | answer | label |
|---|---|---|
| physical paths baked into an asset | all ≥2-event paths whose segments stay inside the asset's envelope, including same-asset exposed interreflection (teaset part-to-part) and SSS | IMPL; PAPER §2, App. A.1 (exit recorded "when leaving the entire asset") |
| incident illumination | any spatially and directionally varying radiance arriving from **outside** the hull/AABB, sampled at render time; no light inside the envelope; no MIS over the incident location (variance under small/delta emitters) | PAPER Fig. 5, §4.4; IMPL `eight_dna.py:316-326` |
| external geometry | outside the AABB: path traced with current visibility (first-vertex events, learned NEE from the AABB exit, continuation). **Inside the AABB** (not touching the asset): seen by camera rays and first-vertex events only; learned internal paths, learned NEE and learned continuation ignore it (leak), and where it blocks a re-hit direction the BSDF branch adds its light while the learned weight still holds the unblocked energy (double count) | IMPL (code audit case b, verified); PAPER §4.4, Fig. 14 ("no other objects can intersect with the asset's convex hull") |
| effective forbidden region | **the AABB** in the code (learned exits live on it), the convex hull in the paper | IMPL vs PAPER |
| multiple learned assets | supported by the integrator (`asset_<id>` per instance, per-asset loops); **not exercised by any released scene** (only `asset_instance0`, identity `to_world`); paper Fig. 16 is qualitative | IMPL; PAPER App. C |
| after a rigid transform of the asset | translation exact; 90° rotations exact; general rotation approximate (exit re-projected onto the larger world AABB; up to about one half-extent displacement at 45°); scale wrong (`(x−t)@A` uses Aᵀ as the inverse) | IMPL `eight_dna.py:205-206, 223-233`; verified arithmetic |
| after an edit of another object | nothing to update: `prepare_scene` caches only (m2w, AABB, ShapePtr) per asset; the next frame re-traces everything | IMPL `integrator.py:81-100` |
| what needs new ray tracing / path sampling | every frame: camera rays, first-vertex NEE and BSDF rays, one self-visibility ray (traced full width), learned NEE shadow rays from the AABB exit, continuation rays; ≈0.94 M MACs of network per asset hit (two query calls) | IMPL `integrator.py`, `eight_dna.py`; counts measured in §D (pending) |
| costs reported | paper Tab. 2 (RTX 3090, 800², 128 spp): 0.52–0.93 min per frame for 8DNA vs 0.82–15.6 min PT; Tab. 3 pre-bake hours; Tab. 7 inference memory 3.7–8.8 GB | PAPER |
| absent from a full scene-edit pipeline | incremental or partial re-render; radiance/transport caching; dirty tracking; any check that external geometry stays outside the AABB; convex decomposition (suggested, not implemented); per-part ownership; scaled instances; external media; counters; retraining other than from scratch | IMPL `models/`, `train.py`; PAPER §4.4 |
| pitfalls found | reusing an integrator across `mi.load_dict` scenes leaves stale ShapePtrs (silent plain-PT fallback or aliasing a new scene's shape); a missing asset id silently mis-pairs models and shapes; `ray_intersect_preliminary(ray, active)` binds the mask to `coherent` | IMPL (verified) |

Paper vs demonstration [PAPER, verified]: the paper shows no quantitative test of an external occluder or reflector **just
outside** an asset's hull; Fig. 14's "outside convex hull ✓" panel is a schematic; the only near-light quantitative test
(Fig. 20 / Tab. 6) has no background geometry; only the Cat scene has an external mirror (the Teaset's interreflections are
its own, baked).

---

## B. Reproduced results

[MEAS] Stage `static`, LabServer63 RTX 3080 Ti, **official environment** (python 3.10, torch 2.3.1+cu121, Mitsuba 3.5.1,
DrJit 0.4.6, demo-notebook JIT flags LoopRecord = VCallRecord = False), released checkpoints, upstream code unmodified,
commit `56112b4`. Display transform of the notebook; each machine against its own references (never pooled).

| official demonstration | 8DNA vs own reference (PSNR / SSIM) | 8DNA seed repeat | reference A vs B | neural time | reference time (one) | PC RTX 5080, compatibility env [REPO WL21–22] |
|---|---|---|---|---|---|---|
| seal `scene2`, 256², `neuralpath`, 256 spp; reference C++ `volpath` 2 × 32,768 spp | 26.56 dB / 0.775 | 25.34 dB | 29.75 dB | 21.0 s | 1,791 s | 26.61 dB / 0.773, repeat 25.34 dB |
| teaset `get_scene`, 512², 256 spp; reference C++ `path` 2 × 2,048 spp | 29.20 dB / 0.895 | 34.66 dB | 44.05 dB | 36.9 s | 39.3 s | 29.20 dB / 0.892, repeat 34.66 dB |

- The released models reproduce their PC numbers on the official stack to within 0.05 dB; the seal's error is at the level
  of the neural render's own seed noise at 256 spp, the teaset's is not (it sits in part-to-part reflections [REPO WL21]).
- Reference integrators: stage 0 found C++ `volpath`/`path` equivalent to the scenes' own `prbvolpath`/`prb` (128², 2,048 spp
  each: MSE 0.0107 vs 0.0109 between two `prbvolpath` seeds, mean ratio 0.998; `path` vs `prb` 4.30e-5 vs 4.15e-5, ratio
  0.9997). VCallRecord = False renders at launch size (0.91 s per 4 spp), so the notebook value is used.
- Learned parameters bit-identical before/after; no non-finite pixel. Peak torch memory 0.51 GB (seal) / 2.47 GB (teaset);
  process GPU memory up to 10.5 GiB for teaset at 512² (DrJit caching).

---

## C. Dynamic transport coverage matrix

Families per brief §4. "Integration" = what this baseline adds (scene composition, an explicit box, a second learned asset,
counters); it adds no renderer feature.

| family | original 8DNA (paper) | public implementation | added by integration | unsupported / unverified | measured here |
|---|---|---|---|---|---|
| asset-internal (≥2 events inside one asset) | baked into F | baked (`bounce` 2), learned lobe forced on a same-instance re-hit | — | any internal change needs a full re-bake | static fidelity B, E1–E3 (pending) |
| cross-asset free space, asset ↔ explicit geometry outside the AABB | qualitative only (background geometry, Fig. 9) | path traced in the current scene | red box (E1, E2 `T1_out`) | — | E1/E2 `T1_out` (pending) |
| cross-asset free space, learned ↔ learned | qualitative only (Fig. 16) | path traced; multi-asset loops exist but are unexercised upstream | second asset (E3 seal + bunny), m2w translation check | overlapping or nested envelopes | E3 (pending) |
| same-asset exposed free space (inside the envelope) | baked; valid only while nothing enters the hull | baked and frozen | box inside the AABB (E1/E2 `T1_in`) | an object inside the envelope (leak + double count) | E1/E2 `T1_in` (pending) |
| multi-bounce (scene level) | path tracing with MIS and RR | path tracing; each asset visit is one depth step (`max_depth` 10) | — | multi-bounce through geometry inside the AABB | via E1–E3 families (pending) |
| direct visibility changes | NEE at the boundary and at the learned exit | first-vertex NEE exact; learned NEE tested from the AABB point, not the surface | — | occluders between surface and AABB | occlusion family, E2 direct/indirect split (pending) |
| current radiance-state changes | lighting-independent F; L_i re-estimated every frame | no transport or radiance state persists; nothing incremental | — | reuse of any earlier frame's work | cost stage (pending) |

---

## D. Measured fidelity and computational cost

**Pending** (stages `refs`, `neural`, `ladder`, `cost`; classification rules of `dyn_v1.json`).

---

## E. Missing capabilities and confounds

### E.1 The ReST-GI Lego + red box edit violates 8DNA's empty-envelope condition [MEAS, geometry]

`lego_envelope_analysis.py` on `results/rna_surface_ownership_lego/mesh/lego_mesh.npz` (sha256 `08972022…`) and
`B_box_T0.npz` (`33ea7a46…`), T1 = T0 + 0.44 x:

- The box lies **inside the whole-Lego AABB at T0 and at T1** (and inside its convex hull, per the study's witness-vertex
  test). A whole-Lego 8DNA asset with the box as explicit geometry is therefore code case (b) in both states: the box is seen
  by first-vertex events only; the Lego's learned ≥2-event paths, learned NEE and continuation ignore it.
- **No single connected component's AABB contains either box position** (0 of 1,872).
- Grouping components along the box's face planes gives 6 groups whose AABBs avoid both box positions (plate below
  66 components 8.49 m²; above the box top 515 / 8.22 m²; behind 1,280 / 9.81 m²; three small groups). But up to **23.5 %
  of the Lego's surface lies inside another group's AABB** (5.1 % inside the plate group's), so inter-group transport would
  again hit case (b).
- The **finest partition of intact components with pairwise disjoint AABBs has 2 clusters**: one bottom component
  (2.88 m²) and one cluster of the other 1,871 components whose AABB spans the whole Lego and **contains the box at T0 and
  T1**. So no decomposition into intact connected components satisfies the condition for this edit; it would need cutting
  meshes (new artificial boundaries), which neither the paper nor the code provides.

[INF] With the reference's object-wide Burley SSS crossing disconnected components [REPO WL41 §A], cutting or grouping
would also remove cross-group subsurface coupling.

### E.1b How much of the Lego edit's change is first-event transport [MEAS, existing Cycles data, no render]

`lego_change_share.py` on the B4 held-out targets ("hist": 2×2048 spp, clamp d20/i5) and the WL41 first-event references
(one intrinsic event, 1024 spp, clamp d20), both PC Cycles, 16 views, stationary Lego pixels (Lego in both states,
uncorresponded excluded). share = ⟨dG_first, dG_full⟩ / ⟨dG_full, dG_full⟩ in linear RGB, pooled:

| set | pixels | first-event share | fraction of A_changed change energy |
|---|---|---|---|
| all stationary | 3,632,691 | 0.80 | — |
| A_changed | 286,947 | 0.81 | 1.00 |
| A_vis_changed (direct-visibility flag differs) | 57,882 | 0.83 | 0.94 of A_changed's energy lies in these pixels |
| A_changed, Sand | 207,114 | 0.83 | 0.87 |
| A_changed, Yellow | 28,455 | 0.62 | 0.10 |
| A_changed, lit / shadowed at T1 | 67,981 / 218,966 | 0.84 / 0.79 | 0.46 / 0.54 |

Reading [INF]: the edit's stationary-receiver signal is dominated by the box's direct shadow on the Sand baseplate. For a
non-SSS material the Cycles first event is a single surface vertex, which 8DNA evaluates with the real BSDF and current
visibility even when the box is inside the AABB; Yellow's first event includes Burley SSS of direct light, which 8DNA would
learn and connect to the sun from its AABB exit, where it cannot see the box. So roughly three quarters of this edit's change
energy is of a kind any method with current direct visibility follows, and the remainder (≈ 20 % multi-event plus Yellow's
SSS-of-direct) is the part that separates in-envelope transport handling. Energy-weighted shares, not a measurement of 8DNA.

### E.2 Material and regime confounds for any Lego 8DNA arm

- Yellow (41.4 % / 41.6 % of held-out T0/T1 pixels) is Cycles Christensen-Burley SSS; Mitsuba 3.5.1 has no BSSRDF plugin;
  the closest is a dielectric boundary with a homogeneous medium, which needs closed components (682 of 1,872 are closed)
  [REPO WL40, WL42; study L6]. A Mitsuba Lego is a different physical scene; its numbers would form a new regime.
- The Lego regime's only light is a delta sun; the paper expects high 8DNA variance for small/delta emitters and has no
  MIS over the incident location [PAPER §4.4]; [REPO WL22] a delta light on teaset gave 8DNA seed noise of 60–93 % of the
  interaction signal.
- The historical "hist" reference uses Cycles sample clamping (direct 20, indirect 5), which stock Mitsuba cannot reproduce;
  only the clamp-free r1 regime is a like-for-like target [REPO WL42 F5].
- Training on the server does not fit (12 GB; 2 GiB GPU `randperm` OOM) [REPO WL23]; on the PC an 8DNA re-bake peaked at
  15.6 GiB [REPO WL27]. A 256² triplane, which the paper uses for its own Lego/Flowers (App. C), adds 1.47 M parameters.
- No Lego 8DNA checkpoint exists; the paper's own Lego (credited to the Mitsuba 3 gallery, qualitative Fig. 17) was not
  released, and whether it is the same asset as RNA's `Lego_rough_aovs_normalized.blend` is unverified.

### E.3 Other confounds (pending measurement where marked)

- Display transform: this protocol uses the 8DNA notebook transform clamp(x^(1/2.2)); the Lego protocols use
  (L/(1+L))^(1/2.2). Values in dB are not comparable across these regimes.
- The 8DNA experiments here use official assets and scenes; they are not the Lego and none of their dB values may be compared
  with Lego / Cycles numbers.

---

## F. Fair comparison protocol against FNTO

*(Specification only. Choosing whether to run any of it is the user's decision.)*

**F.1 Which FNTO claims an 8DNA + path-tracing baseline can test** (claims C1–C10 as listed in the living docs; mapping is
[INF] from §A and §E):

| FNTO claim | what "released or retrained 8DNA + conventional path tracing" tests | condition |
|---|---|---|
| C4/C5 zero reusable-parameter updates after a rigid edit | 8DNA meets it **by construction** for edits outside every asset AABB: nothing learned is updated, the change comes from path tracing | box outside all envelopes |
| C3 path partition (exposed same-asset free-space paths in C, not K) | 8DNA bakes them; an edit inside the envelope measures the error FNTO's partition is meant to remove | box inside an envelope (E1/E2 `T1_in`; the Lego edit itself) |
| C7 nonlocal influence without a dense path-tracing rebuild | 8DNA's coupling *is* dense Monte-Carlo path tracing; it is the reference mechanism FNTO's learned C must match in quality and beat in cost | equal-quality cost accounting |
| C8 efficiency vs quality-matched full recompute | 8DNA + PT gives a per-frame cost with no update phase and no learned-state invalidation; FNTO must report update + render cost at matched error | same scene, same reference, same error metric |
| C1/C6 intrinsic transport completeness and static fidelity | 8DNA is the closest learned asset-level operator of the same type (entry point/direction → exit distribution, incl. SSS media) | requires the same scene and material semantics |
| C10 conditional research claim | the surviving claim must hold against this baseline at equal quality | — |

**F.2 Two admissible designs** (each must be run entirely in one renderer regime):

1. *Mitsuba regime on an asset both methods can be trained on.* An official 8DNA asset (or a Mitsuba-converted asset with
   declared materials), a moved box both outside and inside the envelope, references by the scene's own path tracer, FNTO's
   K/C trained from probes in the same regime. Measures C3/C4/C5/C7/C8 like for like. FNTO's current pipeline renders
   Stage A probes with Cycles, so this requires a Mitsuba probe path for FNTO; that is FNTO work and is not specified here.
2. *Cycles Lego regime.* Needs an 8DNA Lego asset trained from Cycles-equivalent path samples. Cycles exposes no path-vertex
   records, so this needs either a custom path sampler implementing Cycles' Burley SSS or a declared Mitsuba material
   conversion (a different scene, E.2) — and in both cases a cut decomposition of the Lego (E.1). This is the "major
   renderer development" stop condition of the brief.

**F.3 Arms and accounting for design 1** (matching FNTO's required accounting): (a) 8DNA + PT, box outside the envelope;
(b) the same, box inside; (c) a per-state 8DNA re-bake as the full-recompute anchor (offline cost [REPO WL27] 1.76 h);
(d) FNTO stale / updated / oracle per its contract. Report per material and lit/shadow, signed change on stationary
receivers, R and F on the changed region, parameter digests, rays and network queries per frame, update and render time,
equal-error and equal-time ratios against brute-force path tracing, learned and state bytes, offline supervision cost.

---

## G. Implications for ReST-GI evaluation

**Pending** the measured sections. Already fixed by code and geometry [INF from §A, §E]:

- What 8DNA already achieves: reusable learned asset-internal transport under arbitrary near-field lighting from outside its
  envelope, and correct dynamic response to edits **outside** its envelope with zero parameter updates — because the
  conventional path tracer re-traces all free-space transport every frame.
- What the additional conventional renderer provides: all cross-asset and asset↔scene free-space transport, current
  visibility, multi-bounce between assets, at full per-frame Monte-Carlo cost; no incremental update exists.
- What FNTO would need to demonstrate beyond that baseline: correct response to edits **inside** an asset's envelope (the
  official Lego edit is one), and/or a lower update-plus-render cost than per-frame path tracing at matched quality.
- What remains unknown: the measured size of 8DNA's in-envelope error, its static fidelity on SSS-heavy assets under a
  nearby occluder, and the equal-quality cost ratio (all pending), and every Lego-specific 8DNA quantity (no Lego asset).

Proposed factual corrections to living documents (not applied; listed for the user):
1. FNTO contract §8 prior-art list omits 8DNA, which WL40 judged the closest precedent of K_a.
2. `RESEARCH_CENTRIC_TOPIC.md:107` ("frozen 8DNA … do not track it") is scoped to a part moving inside one teaset asset;
   read alone it can be over-generalised to all 8DNA geometry edits, while edits outside the envelope are path traced.

---

## H. Reproducibility information

### H.1 Sources and environment
- Official code `https://github.com/lwwu2/8dna26` @ `4a2157ca24e506c5ac0831f27d656ecc50a64f07` (no newer upstream commit on
  2026-10-24), unmodified, at `external/8dna26` (PC) and `~/NeuralShaderDemo/external/8dna26` (server).
- Released assets/weights: `scenes.zip` sha256 `e6065389…d1d19`, `weights.zip` `ef134a50…bb13c` [REPO WL19/21]; per-checkpoint
  sha256 recorded by every stage.
- Server environment: python 3.10.21, torch 2.3.1+cu121, Mitsuba 3.5.1, DrJit 0.4.6, `cuda_ad_rgb`, RTX 3080 Ti 12 GB,
  driver 595.71.05; LoopRecord off; VCallRecord per stage 0 (`stage0/flags.json`).

### H.2 LabServer63 Mitsuba segfault [MEAS]
- Kernel module and `libcuda.so` 595.71.05; `/lib/x86_64-linux-gnu/libnvoptix.so.1 -> libnvoptix.so.570.195.03`,
  `libnvidia-rtcore.so.570.195.03` (system, root-owned).
- `server/mi_optix_check.py` (64², 4 spp cube scene): system libraries → SIGSEGV inside `mi.load_dict` (faulthandler: the
  scene-load line; last completed stage `variant_set`), 1/1. With `LD_LIBRARY_PATH=~/nvlibs/optix595` (the 595.71.05
  user-space files extracted in WL23) → scene loads, render 0.8 s, finite, the 595 libraries mapped (`/proc/self/maps`), 9/9.
- Exit-time crash with the fix in place: SIGSEGV after outputs are written, 2/2 default, 2/2 after explicit `del`/`gc`,
  3/3 with the DrJit log muted; 0/2 with `os._exit(0)`; no crash without a scene (DrJit arrays only). gdb (user-space copy
  of Ubuntu's package): `jit_shutdown()` → `jitc_log()` → libc during the variable-leak report in `Py_FinalizeEx`.
- First 8DNA server smoke (`server/edna_smoke.py`, released seal, `scene2`, 64², 16 spp): exit 0, finite, warm render 3.3 s,
  torch peak 41 MB.
- Fix: `server/mi_env.sh` (version-guarded `LD_LIBRARY_PATH`) + `os._exit` at the end of every entry point.

### H.3 Commands
See `experiments/8dna_dynamic_baseline/README.md`. Outputs: `results/8dna_dynamic_baseline/` (copied from the server),
reviewer exports and manifest in `results/evaluation/43/`.
