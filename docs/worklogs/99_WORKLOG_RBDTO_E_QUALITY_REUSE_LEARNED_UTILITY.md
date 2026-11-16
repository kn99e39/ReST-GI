# Worklog — WL99: RB-DTO-E Quality, Reuse and Learned Utility Validation (2026-11-16)

> Append-only. Sections are added as gates complete; earlier sections are never rewritten.

## Origin and scope

**Brief.** The user's brief "WL99 — RB-DTO-E Quality, Reuse, and Learned Utility Validation".
- The user authorises adding non-convex rigid-body workloads to the experimental scope.
- The architecture, novelty, problem definition and research direction are unchanged.
- RB-DTO-E (WL98 J2 reformulation of D3's RB-DTO) is investigated as one candidate. It is not adopted.

**Batch class.** A regular number: a bounded experimental batch on a new candidate representation.

**Hypotheses, kept separate.**
- H1, representability: can the learned body-local operator represent high-frequency multi-bounce self-interreflection?
- H2, reusability: can the body-local state stay unchanged under qualified rigid edits while external transport updates?
- H3, learned utility: does learning beat strong non-learned precomputed transport at matched quality and resources? Credit is split into U_pre and U_learn.

**Stages (protocol `rbdto99_v1`, `experiments/rbdto99/protocol/rbdto99.json`).**
- Stage 0: qualification of non-convex rigid-body workloads with path-tagged references.
- Stage 1: learning-free comparators and headroom gates (E0–E4, U0, U_pre).
- Stages 2–3: candidate implementation and quality / reuse / utility evaluation, only if Stage 1 passes.

**Where.**
- Code: `experiments/rbdto99/`.
- Server: LabServer63, `~/rest_x99/` (development in `~/rest_x99dev/`).
- Reviewer images: `evaluation/99/`.

## Fixture and instruments (built this session)

- **Renderer.** `pt99.py`: the WL86 Dr.Jit / Mitsuba 3.5.1 ER path tracer re-cut for one rigid body.
  - Next-event estimation to point lights at every vertex plus ER sampling, no Russian roulette, depth 8.
  - Frozen ER states S1 / S2 / B and Lambert from `experiments/ecro86/data/mats86.npz`; no new material.
  - Fixed shape slots (5 room rectangles, 3 body plates, 2 Lambert boxes) moved by vertex updates, with an opaque camera and lights. Kernels therefore compile once per process (about 65 s) rather than once per configuration.
- **Path tags** per NEE contribution:
  - B1: first body event.
  - S2, S3: self-transport of direct light after one, or two or more, self-hits. Sd = S2 + S3.
  - SX: self-transport of external-indirect input.
  - X: external indirect on the body.
  - ES: body self-transport exits delivered to external receivers.
  - E: other.
  - The tag sum is the estimator.
- **Exact-direction probes** (rays aimed at a given surface point from a given direction) and an `oracle_self` frame mode. In that mode, body-pixel paths whose second vertex is on the body terminate, so the cost and variance of everything except N_b are measured.
- **Body family** (`scene99.py`):
  - V-groove (development geometry);
  - trihedral corner and U-channel (held-out geometries);
  - convex box (null control).
  - The certificate is a separating-axis test between the body's convex hull and every foreign box.

## Development before the freeze (disclosed; not evidence)

Three development look sets (`dev_a`, `dev_b`, `dev_c`, 64–256 spp) were rendered to choose the fixture.

- **In the GM1 room the glossy body mostly reflects lit walls.** X was 0.96 of body-pixel radiance and Sd ≈ 0. The fixture therefore uses a 'dark' room: the floor (ER B) is the only external reflector, and walls and ceiling have albedo 0.
- **A light outside the 90° groove's ±45° cross-section acceptance gives no double reflection** (dev_a). The groove is pitched 50° towards the camera, and the light mirrors the camera in x (retro geometry).
- **The retro-reflection lights the whole aperture.** Thin Lambert bars were added as occluders, to create shadow structure whose reflection carries spatial detail (dev_c).
- **The reflected detail is soft at the S1 lobe scale.** S1 has GGX α 0.026–0.075; reflected shadow edges are cm-scale over 0.3 m paths. S1 is the sharpest frozen ER state in project use, and no sharper material was introduced. This bounds what "sharp" means in this batch.
- **Focused tests (development run, 128 px):**
  - null body tags exactly zero;
  - tag partition exact;
  - certificate rejects an intruder in the groove;
  - probe / pixel Sd ratio median 1.008 (IQR 0.979–1.018);
  - co-moved scene with the same seed differs by 1.4e-4 relative, against 0.10 between independent seeds;
  - oracle_self vs cam for B1, X and E agree to 1.5e-7 (float re-association between two kernels).
  - The test T3 criterion was changed before evidence from "bitwise" to "≤ 1e-5 × channel max".
- **A Stage 0 dry run at 64 spp** exercised the whole qualification pipeline. Noise dominated, as expected; no admission decision was taken from it.

## Protocol freeze

Protocol `rbdto99_v1`, workloads `workloads99.json` (7 candidates: A1 groove S1 retro, A2 groove S1 grazing, B1 groove S2, C box null, A3 corner, A4 channel, A5 groove S1 oblique held-out illumination) and the Stage 0 code are committed before any qualification evidence. Rules set by the agent (the brief asks for pre-registration and gives no numbers), all fixed before the evidence they govern:
- reference halves 16384 spp, T0 / T1 extended ×2 up to 65536 when the per-pixel noise rule (≤ 0.025 relative on ROI_self) fails;
- admission Q1 (Sd ≥ 0.20 of body-pixel radiance), Q3 (spatial edge width ≥ 2 px and contrast ≥ 0.30 in the sharpest 10 % of the self region; angular 25 %-change angle ≥ 2 pixel pitches by exact-direction probes), Q4 (co-moved invariance, certificates), Q5 (T1 vs frozen-T0 exit field ≥ 0.30), Q6 (intrusion detected and ≥ 0.10 effect);
- Stage 1 gates: U0(τ = 0.05) ≥ 1.25 on an admitted sharp workload (G1a), and learning-specific headroom U0 / U_pre ≥ 1.25 at ≤ 64 MB per body or a preparation-cost route (G1b); otherwise STOPPED_NO_LEARNABLE_HEADROOM.

## Stage 0 execution log (2026-11-16, LabServer63, `~/rest_x99/`)

**Focused tests** (`tests99.json`, evidence run): all pass.
- T1 null-body tags are exactly 0.
- T2 tag partition is exact.
- T3 oracle_self vs cam differ at float re-association level only.
- T4 certificate: T0 / T1 pass and the intruder fails.
- T5 probe / pixel Sd ratio median 1.008.
- T6 co-motion with the same seed differs by 1.4e-4 relative, against 0.10 between independent seeds.

**Execution fix (commit `720c135`).**
- At 10:43 UTC the renders were found running at about 31 % GPU. Each 16-spp launch spent about 1 s in Python kernel tracing, against about 40 ms of GPU work.
- Fix: per-launch scatter into persistent float64 GPU buffers, and 128 spp per launch.
- Effect: 1024 spp at 384² went from 80.8 s to 3.9 s; GPU about 75–100 % during renders.
- The estimator is unchanged. Only the per-launch seed partition differs.
- References completed before the fix (15 files, b = 16) are reused alongside new ones.

**Round 1 (7 candidates): NOT_QUALIFIED_WITHIN_FIRST_CANDIDATES.**
- Admitted: A5 (groove S1, oblique light, held-out illumination) and C (null; S2 + S3 + SX + ES exactly 0).
- Not admitted:
  - A1 (retro groove S1): Q3 T1 edge width 1.67 px < 2, and Q4 co-moved difference 0.0217 > 0.02. The Q4 limit uses only the T0 reference noise (T0 was extended to 65536 spp per half, the co-moved reference stayed at 16384), so this rule under-counts the comparison noise. That is a protocol-design weakness; it is recorded here and was not changed.
  - A2 (grazing): Q_ref T1 0.063 > 0.025 even at 65536 per half, and Q3.
  - B1 (S2): Q3 T1 contrast 0.28.
  - A3 (corner): Q3 contrast 0.19–0.26.
  - A4 (channel): Q3 edge width 1.84–1.92 px.
- Diagnostic (`diag_q3.py`, not a gate): only 7–22 % of the sharp-ROI pixels lie within 2 px of a body silhouette, and interior-only statistics are nearly identical. The Q3 failures are therefore not a silhouette artefact. They have three causes:
  - noise-limited T1 edges (A1, A2): noise / signal gradient 0.37–0.52;
  - details at or below the pixel scale (A4): noise ratio 0.04–0.07;
  - smooth self-reflection with little detail (A3, B1 T1).

**Round 2 (4 more candidates, 11 of 16; commit `4c47c54`; rules unchanged): still not qualified.**
- A6: groove S1, fov 12°, smaller T1 motion.
- B2: S2, same changes.
- A8: channel, fov 10°.
- A9: corner with two bars, fov 12°.
- All four fail only Q3: contrast 0.13–0.25, or T1 edges 1.70–1.97 px.
- A6's Q4 passes (0.008); its Q6 passes as well (intrusion detected, effect 0.15).

**Round 3 (4 more candidates, 15 of 16; commit `88c1dca`).**
- Modelled on the admitted A5 design (oblique light, bar shadow carried by the self-reflection): A10 groove S1 mirrored, B3 groove S2, A11 channel, A12 corner.
- Started at 12:47 UTC and stalled from 12:57.
- Cause: the WL21A public render (`render.py`, another session's batch) took 11.3 of the 12 GB of GPU memory from 12:54. Dr.Jit reported memory exhaustion and made no GPU progress for 75 min.
- The agent stopped its own chain at 14:12. Completed references are kept for resumption.
- Pending: the user's decision on scheduling or machine.
