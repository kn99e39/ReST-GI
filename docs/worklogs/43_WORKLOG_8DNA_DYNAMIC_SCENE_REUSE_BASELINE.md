# Worklog — 8DNA Dynamic-Scene Reuse and Transport-Update Baseline Investigation (2026-10-24)

**Status: complete (2026-10-25).** Protocol `8dna_dyn_v1` ran on LabServer63 after the WL42 FNTO chain (2026-10-24 22:59
→ 2026-10-25 17:10 KST, including a 4 h hang, §0). All labels below are the protocol's declared rules applied without
change; every number was reproduced by an independent re-implementation (§H.4). Scope: official 8DNA assets in their own
Mitsuba scenes, one box and one view per state. **No number here is comparable with Lego / Cycles / FNTO / RNA numbers.**

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
- **Run.** FNTO finished 2026-10-24 22:48 KST; dry pass 22:59–23:03 (all 9 stages rc 0); real stages from 23:03.
  The user chose to keep the declared spp when told the run would take ~18 h.
- **Hang (2026-10-25 07:29–11:26 KST, my fault).** After all E1 and four E2 references, the first E2 direct-only reference
  used a 512² × 64-spp launch (16.7 M lanes, 4× every other launch). It exhausted the 12 GB GPU; DrJit flushed its cache and
  the process then spun on one CPU core with the GPU idle for ~4 h. The user noticed ("vram만 차지하고 로드율이 안올라가서
  일을 안하고 있는데"). My PC watcher woke only on status changes, not on a stalled log. Fix `41e655f`: direct-only chunk 16,
  a 4.2 M-lane cap per launch, a DrJit cache flush before each render, progress lines every 60 s, a stall watcher, and
  `--resume`. The run resumed from the refs stage at 11:28; the 18 finished references were reused after their sidecars
  matched; no measured value had been looked at. Recorded as `revision_during_run` in the protocol.
- **Result checks before writing** (workflow, 4 agents): an independent numpy re-implementation (no shared code), a visual
  review of all exports, an attribution of the in-envelope failure, and an adversarial critic of every claim (§H.4).

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
| what needs new ray tracing / path sampling | every frame: camera rays, first-vertex NEE and BSDF rays, one self-visibility ray (traced full width), learned NEE shadow rays from the AABB exit, continuation rays; ≈0.94 M MACs of network per asset hit (two query calls; code-derived, not measured) | IMPL `integrator.py`, `eight_dna.py`; learned-query counts measured in §D.4 |
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

- The released models reproduce their PC numbers on the official stack to within 0.05 dB.
- Error composition [MEAS, MSE(N,G) − neural noise − reference noise]: for the seal, Monte-Carlo noise is 78 % of the display
  MSE at 256 spp, with an spp-independent non-noise excess of about 0.00046 display MSE (≈ 33.4 dB) that is also present at
  1024 spp; for the teaset, noise is 18 % of the full-frame MSE and 5 % on asset pixels, i.e. its error is bias (located in
  part-to-part reflections [REPO WL21], not re-measured here).
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
| asset-internal (≥2 events inside one asset) | baked into F | baked (`bounce` 2), learned lobe forced on a same-instance re-hit | — | any internal change needs a full re-bake | static, unchanged by out-of-envelope edits (E1 24.44–24.45 dB, E2 26.71 dB asset pixels); E2 bias-limited, E1 noise-dominated |
| cross-asset free space, asset ↔ explicit geometry outside the AABB | qualitative only (background geometry, Fig. 9); no quantitative external-occluder test | path traced in the current scene | red box (E1, E2 `T1_out`) | — | **E2 TRACKED** (gain 1.004, corr 0.996, R 0.995; occlusion incl. its indirect part and reflection both TRACKED); **E1 NOISE_LIMITED** (descriptively consistent with tracking) |
| cross-asset free space, learned ↔ learned | qualitative only (Fig. 16) | path traced; multi-asset loops exist but were unexercised upstream | second asset (E3 seal + bunny), m2w translation check | overlapping or nested envelopes | both learned assets active and the m2w path verified; the edit itself **NOISE_LIMITED** (not tested) |
| same-asset exposed free space (inside the envelope) | baked; valid only while nothing enters the hull | baked and frozen | box inside the AABB (E1/E2 `T1_in`) | an object inside the envelope (leak + double count) | **E2 NOT_TRACKED** (gain 0.085, corr 0.12, R −0.19); **E1 PARTIAL** (gain 0.57, corr 0.73, R 0.78) |
| multi-bounce (scene level) | path tracing with MIS and RR | path tracing; each asset visit is one depth step (`max_depth` 10) | — | multi-bounce through geometry inside the AABB | out of envelope: E2 indirect occlusion TRACKED (gain 1.002); in envelope: E2 indirect occlusion NOT_TRACKED (gain −0.003) |
| direct visibility changes | NEE at the boundary and at the learned exit | first-vertex NEE exact; learned NEE tested from the AABB point, not the surface | — | occluders between surface and AABB | E2: 8DNA's occlusion change equals the reference direct change (LS 1.008) in both placements; in envelope the direct part is only 0.6 % of the occlusion |
| current radiance-state changes | lighting-independent F; L_i re-estimated every frame | no transport or radiance state persists; nothing incremental | — | reuse of any earlier frame's work | 0 parameters updated, 0 state bytes; every edit = full re-render (78 s / 144 s) after a ~2 ms scene update |

---

## D. Measured fidelity and computational cost

[MEAS] LabServer63, official stack, released checkpoints, VCallRecord False, references C++ `volpath` (E1, E3) and `path`
(E2) at 2 × 32,768 / 2 × 8,192 spp, 8DNA `neuralpath` 1,024 spp at paired seeds. Display clamp(x^(1/2.2)). Receivers = the
stationary asset's pixels in both states and the box-absent scene, eroded by one pixel. Placements (design stage): E1 box
s = 0.18, T1_out 0.040 outside the seal AABB on the floor, T1_in on the pedestal top 10 mm inside the AABB beside the figure;
E2 box s = 0.12, T1_out 0.030 outside the teaset AABB on the table, T1_in on the tray in the cell between milk pot, tea pot
and tin. Both T1_in boxes are also inside (E2) or partly inside (E1) the asset's convex hull, so these runs do not separate
the code's AABB rule from the paper's hull rule.

### D.1 Edit tracking on stationary learned-asset receivers (declared labels)

| edit | receivers | rms dG | signal / ref noise | signal / neural noise | gain | corr | R (A_changed px) | F | label |
|---|---|---|---|---|---|---|---|---|---|
| E2 teaset, box **outside** AABB | 50,019 | 0.0090 | 44.6× | 11.0× | 1.004 | 0.996 | 0.995 (1,398) | 23.4 | **TRACKED** |
| E2 teaset, box **inside** AABB | 48,819 | 0.0215 | 41× | 9× | 0.085 | 0.117 | **−0.19** (4,741) | 0.87 | **NOT_TRACKED** |
| E1 seal, box **outside** AABB | 20,744 | 0.0155 | **2.97×** | **2.22×** | 0.87 | 0.85 | 0.96 (2,924) | 1.62 | **NOISE_LIMITED** |
| E1 seal, box **inside** AABB | 20,699 | 0.0045 | 3.13× | 3.94× | 0.57 | 0.73 | 0.78 (715) | 1.11 | **PARTIAL** |
| E3 seal ↔ moved bunny (both learned) | 5,649 | 0.0150 | 2.30× | **0.99×** | 0.86 | 0.60 | 0.67 (1,236) | — | **NOISE_LIMITED** |

The gate needs > 3× both noises. Learned parameters were bit-identical (sha256) before and after every rendering process.

### D.2 Path families

| edit | occlusion (black − absent) | reflection (red − black) | E2 direct share of occlusion | E2 indirect (≥2-event) occlusion |
|---|---|---|---|---|
| E2 outside | TRACKED (1.003 / 0.998) | TRACKED (0.996 / 0.994) | 0.26 | TRACKED (gain 1.002) |
| E2 inside | **NOT_TRACKED (0.004 / 0.03)** | TRACKED (0.914 / 0.960) | 0.006 | **NOT_TRACKED (gain −0.003)** |
| E1 outside | NOISE_LIMITED (0.86 / 0.84) | NOISE_LIMITED (1.00 / 0.92) | — | — |
| E1 inside | NOISE_LIMITED (0.02 / 0.27) | TRACKED (0.90 / 0.94) | — | — |

(gain / corr.) Attribution [MEAS unless marked]:
- **E2 inside, leak.** On the 5,021 receivers the box darkens strongly (selected on reference A, measured on B), the
  reference loses 0.050 luminance (−30 % locally); 8DNA changes by −0.0003, i.e. 0.6 % is reproduced and 8DNA is too bright
  by 99.4 % of the change. A joint least-squares fit gives dN_occ = 1.008·dG_direct − 0.003·dG_indirect: 8DNA reproduces the
  direct (first-event) part of the occlusion and none of the ≥2-event part, which is 99.6 % of it here (mostly the tray
  mirroring the pots and the pots mirroring the tray).
- **E2 inside, double count and R < 0.** The box's red reflection is added (gain 0.91; 0.87–0.97 near the box, 0.45 at
  25–50 px where the reflection needs a second asset event) while the light it blocks is not removed. On the changed region
  the reference's red channel nearly cancels (occlusion −0.073, reflection +0.067), 8DNA's does not (+0.053), so the updated
  render is further from T1 than the stale T0 render (red-channel display MSE 0.0056 → 0.0096; green/blue unchanged). Adding
  the missed indirect occlusion alone would close 81 % of the gap. [INF from code `eight_dna.py:198-256`] Where the BSDF ray
  now hits the box, the self-visibility switch turns the forced learned lobe into a mixture of the unchanged learned lobe and
  the real BSDF lobe that sees the box; the learned NEE and continuation start on the AABB and cannot see it. Neural seed noise
  in the shadowed pixels rises 17 % with unchanged mean, the signature of that lobe switch.
- **E1 inside.** The same leak, smaller: 8DNA reproduces about 1 % of the strong darkening (8DNA dN exactly zero on 88 % of
  strongly darkened pixels), but the occlusion family stays under the gate (2.35× reference noise). R = 0.78 comes from the
  red reflection streak on the figure (R 1.06 on brightening pixels, 0.10 on darkening pixels).
- **E2 outside, control.** Direct and indirect occlusion LS coefficients 1.010 / 1.001, reflection 0.996, R 0.995.
- **E1 outside.** Under the declared gate this edit is not classified. Descriptively the values equal what perfect tracking
  would give under the measured noise (expected gain 0.89, corr 0.85; noise-corrected gain 0.98, corr 0.99); the strongly
  darkened pixels show −0.064 (reference) vs −0.062 (8DNA). Classifying it needs both spp raised ≥ 2× (sensitivity only).
- **E3.** The 8DNA bunny is very noisy under the area light (seed repeat 18.6/19.1 dB on its pixels) and 4.8–5.0 % darker
  than the reference (cause not determined); dN noise equals the signal, so no label is possible even with 4× spp.

### D.3 Static fidelity in the edit scenes (asset pixels, eroded)

| scene | absent / T0 / T1_out | T1_in red / black | neural repeat | ref A vs B |
|---|---|---|---|---|
| E1 seal | 24.45 / 24.45 / 24.44 dB | 24.45 / 24.43 dB | 25.1 dB | 25.4 dB |
| E2 teaset | 26.71 / 26.71 / 26.71 dB | **25.00 / 23.45 dB** | 37.2 dB | 46.4 dB |
| E3 seal / bunny | seal 25.14 → 24.51, bunny 20.1 → 20.3 dB | — | 26.2 / 18.6 dB | 25.4 / 24.1 dB |

An out-of-envelope box leaves the asset's static error unchanged; an in-envelope box lowers the teaset's by 1.7 dB (red) /
3.3 dB (black) and turns its mean bias from −0.0014 to +0.0035/+0.0039 (too bright). Full frame E2: 33.23 → 31.70 / 30.23 dB.

### D.4 Cost

| item | E1 seal 256² | E2 teaset 512² | E3 256² |
|---|---|---|---|
| scene update after moving the box (`params.update()`, median of 5) | 1.82 ms | 2.57 ms | — |
| learned parameters updated / transport-state bytes | 0 / 0 | 0 / 0 | 0 / 0 |
| re-render after the edit, 1,024 spp | 77.7 s | 143.9 s | ≈ 99 s |
| in-place edit vs freshly loaded scene (same seeds) | max 5.0e-3, mean 9e-8 | max 5.6e-4, mean 4e-9 | — |
| learned queries per pixel per spp (NEE = sampling) | 0.53–0.55 | 0.25–0.26 | 0.30 |
| learned bytes | 2.69 MB | 2.69 MB | 2 × 2.69 MB |
| peak torch / process GPU memory | 0.56–0.77 GB / ≤ 1.8 GB | 1.3–2.6 GB / ≤ 4.9 GB | 2.0 GB / 4.4 GB |
| path-traced reference, one image | 30 min (32,768 spp) | 3.1 min (8,192 spp) | 56 min (32,768 spp) |
| offline pre-bake (not run) | 2.23 h volumetric [PAPER Tab. 3] | 1.78 h surface [PAPER Tab. 3]; 1.76 h re-bake [REPO WL27] | — |

The scene update is the Mitsuba/OptiX update of the moved mesh, not a transport update: every frame re-traces everything.
Ray "widths" counted in the Python integrator are launched lanes × calls (wavefront), not active rays, and are not reported
as ray counts.

**Equal-quality comparison against brute-force path tracing** (state T1_out, red; error vs G, floor-corrected):

| scene | 8DNA 1,024 spp | PT time to the same error | equal-error time ratio PT / 8DNA | note |
|---|---|---|---|---|
| E1 seal | 84 s, display MSE 9.7e-4 | ≈ 760 s (extrapolated past 4,096 spp, MSE ∝ 1/t) | **≈ 9×** (receivers 8.8×, relMSE 9.4×; 13.5× with the measured last-rung slope; 7.7× without floor correction) | 8DNA is ~7× more accurate at equal time; specific to this error level (8DNA keeps a non-noise excess ≈ 4.4e-4) |
| E2 teaset | 156 s, display MSE 4.7e-4 (bias-limited) | ≈ 3.9 s (interpolated, 16–64 spp) | **≈ 0.025×** (PT ~40× cheaper; receivers ~155×) | 8DNA's error flattens at its bias; PT is faster at every rung |

Implementation mix: 8DNA runs a Python wavefront integrator with torch calls; PT runs C++ `volpath`/`path`. The two scenes
differ in asset type, light, resolution, box and placement at once, so they do not establish a volumetric-vs-surface rule.

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

### E.3 Confounds and limits of the measured experiments

- **Regime.** Official assets and scenes in Mitsuba, display clamp(x^(1/2.2)); the Lego protocols use Cycles and
  (L/(1+L))^(1/2.2). No dB, R, F or gain here may be compared with Lego, FNTO, RNA or Batch 1–6 values.
- **Breadth.** One scene per asset type, one box size, one placement per state, one camera, one reference seed pair. E1 and
  E2 differ at once in asset type, integrator, light (area vs envmap), resolution, box size, placement and reference noise.
- **Power.** For the seal, the 2 × 32,768-spp references (asset A vs B 25.4 dB) and 1,024-spp 8DNA renders cannot classify
  changes below ~3× their noise (E1 T1_out, the E1 occlusion families, E3). NOISE_LIMITED is a statement about the
  measurement, not about 8DNA.
- **Energy weighting.** gain and corr are dominated by shadow cores and highlights (the top 1 % of pixels carry 55–97 % of
  ⟨dG,dG⟩); broad low-energy responses are not visible in the labels.
- **Envelope rule.** Both in-envelope boxes are also inside or across the convex hull; these runs do not show whether the AABB
  (code) or the hull (paper) is the operative boundary.
- **Direct/indirect split (E2 only)** assumes 8DNA's direct term equals the max_depth-2 reference (consistent with the
  measured LS coefficient 1.008); "indirect" includes explicit first-vertex continuations as well as the learned lobe.
- **No delta emitter** in these scenes (area light, envmap); nothing here transfers to the Lego's delta-sun regime.
- **Unexplained:** the E3 bunny is 4.8–5.0 % darker in 8DNA than in its reference; the seal keeps a non-noise excess of
  ≈ 4.5e-4 display MSE in every scene.

---

## F. Fair comparison protocol against FNTO

*(Specification only. Choosing whether to run any of it is the user's decision.)*

**F.1 Which FNTO claims an 8DNA + path-tracing baseline can test** (claims C1–C10 as listed in the living docs; mapping is
[INF] from §A and §E):

| FNTO claim | what "released or retrained 8DNA + conventional path tracing" tests | condition |
|---|---|---|
| C4/C5 zero reusable-parameter updates after a rigid edit | 8DNA meets it **by construction** for edits outside every asset AABB: nothing learned is updated, the change comes from path tracing (measured: E2 TRACKED; E1, E3 noise-limited) | box outside all envelopes |
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

Stated as facts with their evidence level; which of them matters for the research direction is the user's decision.

- **What a fixed 8DNA asset already achieves** [MEAS E2; IMPL]: reuse of learned asset-internal transport under new scene
  configurations with zero parameter updates, and a correct response to an edit **outside** its envelope, including the
  indirect (≥2-event) occlusion and the reflection that edit causes (E2: TRACKED, R 0.995). On the seal the same edit is
  descriptively consistent with this but noise-limited.
- **What the conventional renderer contributes** [IMPL; MEAS cost]: every free-space segment outside the envelope, current
  visibility, and all coupling between assets and scene, at full per-frame Monte-Carlo cost (78–144 s per 1,024-spp frame
  here). No transport state exists, so nothing is updated and nothing from earlier frames is reused; the only edit-specific
  work is a ~2 ms scene update.
- **What this baseline does not cover** [MEAS E2/E1; IMPL]: an edit **inside** an asset's envelope. There the occlusion of
  same-asset (part-to-part) transport is not reproduced (E2: 0.6 %), the edit's reflection is added on top (double count),
  and the update can end further from the truth than not updating (E2: R −0.19, static −1.7/−3.3 dB). The official Lego edit
  is of this kind (§E.1): its box lies inside the Lego's AABB and hull in both states, and no decomposition into intact
  components with disjoint AABBs keeps it out. About 80 % of that edit's stationary change is first-event transport (§E.1b),
  which a method with current direct visibility follows; the rest is where the in-envelope behaviour matters.
- **Which FNTO claims this baseline can test** (mapping of §F.1, not a result about FNTO): C4/C5 zero-update response to
  out-of-envelope edits (8DNA + PT meets it on E2); C3 path partition (the in-envelope failure above is the failure mode the
  partition targets); C7/C8 cost (8DNA + PT is the per-frame path-tracing reference; its equal-error time against brute-force
  PT depends on the asset and the error level: ≈ 9× less time on the seal at its 1,024-spp error, while on the teaset PT
  reaches 8DNA's bias-limited error in ~1/40 of the time).
- **What remains experimentally unknown**: AABB vs hull as the operative boundary; how the in-envelope error scales with
  distance and box size; whether learned-to-learned coupling (E3) is tracked (noise-limited); 8DNA under a delta sun; every
  Lego-specific 8DNA quantity (no Lego asset exists; its construction would need a material conversion, a cut decomposition
  and a new reference regime, §E.2, §F.2).

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
  driver 595.71.05; LoopRecord off; VCallRecord off (the notebook value; stage 0 rendered the launch-size probe in 0.91 s per
  4 spp), recorded in `stage0/flags.json`.

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

### H.3 Commits, data and commands
- Branch `baseline/8dna_dynamic_reuse` (worktree `.claude/worktrees/8dna-baseline`, from `b52ee7d`):
  `c1bcc81` protocol and stages; `56112b4` E2 T0 placement (stages smoke → refs start); `41e655f` hang fix (refs resumed →
  exports); worklog commits `f3d7182`, `c277da6` and this one. Server checkout `~/rest_8dna_baseline/repo` from git bundles.
- Seeds: references A 1,000,000 + chunk, B 2,000,000 + chunk (direct-only +500,000); 8DNA main 0 + chunk, repeat 100,000 +
  chunk; ladder 500,000 / 600,000; static B as WL21 (neural 0 / 100,000, references 200,000 / 300,000).
- Outputs: server `~/rest_8dna_baseline/out` → `results/8dna_dynamic_baseline/out` (200 MB: refs, neural, static, ladder,
  cost, design, stage0, eval, exports) and `results/8dna_dynamic_baseline/launcher` (stage logs). Lego analyses:
  `results/8dna_dynamic_baseline/lego/{lego_envelope,lego_change_share}.json`.
- Reviewer artefacts: `results/evaluation/43/` (37 images and 4 JSON records, sha256 in `manifest.json`).
- Run: `experiments/8dna_dynamic_baseline/README.md`; `server/run_dyn_chain.sh [--dry | --resume STAGE]`.

### H.4 Result checks
- Independent re-implementation (numpy, no shared code) of every metric: 273 of 290 fields agree to 1e-6; all 19 gate
  booleans and 15 labels identical. The 17 others are A_changed counts and their recovery terms, off by ≤ 0.15 % of pixels
  because `dyn_eval.box5` uses a float32 summed-area table near the 0.01 threshold; R moves by ≤ 2e-4 and no label changes.
- Visual review of all exports: placements match the design, boxes rest on their supports, reference and 8DNA align to the
  pixel, no non-finite pixels, no excess fireflies, the E3 bunny's marble pattern follows the instance. The E2 T0 box is
  almost hidden behind the teaset (14 px), which affects only how the images read.
- Attribution and critic reports are session artefacts; their numbers are quoted in §D with the critic's wording limits.
