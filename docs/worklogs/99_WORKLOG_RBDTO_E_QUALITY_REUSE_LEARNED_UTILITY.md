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
