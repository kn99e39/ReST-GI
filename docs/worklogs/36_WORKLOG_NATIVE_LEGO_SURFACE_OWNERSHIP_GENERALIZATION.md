# Worklog — Architecture Batch 3: Surface-Owned Units on the Native RNA Lego (2026-10-18)

## SESSION QUESTION

User prompt "Architecture Batch 3 — Native RNA Surface Ownership Generalization" (2026-10-18):

> Does the existing independent Surface-Owned Neural Transport architecture retain credible static
> radiance fidelity and independently reconstructible learned-state ownership when evaluated on a
> native RNA asset whose original model reproduces paper-comparable quality?

**Short answer: yes, within the predeclared screening (Case A).**

Setup:
- Asset: the official Lego, with the worklog-35 native pipeline.
- Model: 16 independently owned complete RNA units, matched capacity, trained with the unchanged
  worklog-32 trainer.
- Evaluation: 16 held-out views against 4096-spp references.

Results:
- **Fidelity cost is small.** The units score 0.31–0.32 dB below native RNA HQ, with logL1 1.05×
  and high-frequency error 1.06×. Off-band (interior) error is unchanged (0.996×).
- **Seams.** The worklog-34 PU composition removes 91% of the excess seam discontinuity.
- **Reconstruction.** Full re-initialisation and retraining of one unit leaves the other 15 units
  bit-identical. It changes composed outputs exactly on the predicted set (owned pixels plus 1.1%
  neighbour band), and recovers the unit's quality.
- **Difference from Teaset.** The interior per-unit fidelity loss seen there (worklog 34) does
  **not** appear on Lego. The remaining cost is boundary-local and high-frequency.

## RELATION TO EARLIER WORK

- **Worklogs 32–34:** Teaset ownership, baseline audit and boundary attribution.
- **Worklog 35:** native RNA qualified on Lego.

None of these is reinterpreted, and their code and artifacts are unchanged.
- **Reused unchanged:**
  - `rso_model` (OwnedTriplaneSet, matched-capacity rule);
  - `rso_train` (trainer);
  - `b2_compose.pu_weight` (worklog-34 weight).
- **New code:** `experiments/rna_surface_ownership_lego/`. **Outputs:**
  `results/rna_surface_ownership_lego/`. **Exports:** `results/evaluation/36/`.
- **Protocol:** `protocol/b3_v1.json`, committed in `cd5a98f` before any candidate was trained or
  evaluated. Code commits: `6c7398e` and this session's final commit.

## ARCHITECTURE (final report)

**Partition L16** (`b3_partition.py`, `b3_model.LegoOwner`):
- **Surface identity.** A rendered point is placed on the mesh triangle it lies on. The 4.2 M-
  triangle official mesh has 8 materials and 1872 connected components.
  - Candidates: 8 nearest triangle centroids plus the triangles of the 8 nearest of 8 M
    auxiliary on-surface points.
  - Selection: exact point-triangle distance, restricted to normal-compatible triangles
    (|n_t·n| ≥ 0.3).
- **Unit.** The k-means cell (K = 16, area-weighted, seed 0) of that on-surface point.
- **Component.** The triangle's connected component.
- **Resulting units:** compact (extents 0.38–0.59 on a 0.89 × 1.60 × 0.94 asset). 259 of 1832
  bricks are cut by unit boundaries.
- **Rejected variants** (both before any training):
  1. Whole triangles assigned by centroid. Large planar triangles (up to 1.76 m) inflated unit
     AABBs to the full asset length.
  2. Centroid-only candidate search. It missed large triangles (0.1% of on-mesh points up to 13 mm
     off). Auxiliary points fixed this: 99.9% within 0.55 mm.

**Arm A:** native RNA HQ, the worklog-35 reproduced checkpoint (`tog-lego/version_2/last.ckpt`,
final weights; 7.35 M parameters). The released `lego.ckpt` is the secondary control.

**Arm B (independent L16):**
- Model: 16 complete `NeuralSurfaceTriPlane` units, each framed on its own unit AABB. Triplane
  resolution 125–196 (A's texel size) and MLP width 128 (= 512/√16).
- Size: **11.35 M parameters (1.54× A)**, 0.44–0.99 M per unit.
- Sharing: no trainable state is shared. Fixed buffers (AABBs) are shared.
- Training: hard owner selection.
- Contract, otherwise identical to A: same data (worklog-35 native set), loss, optimiser, schedule
  (250 epochs), seed 0 and data placement (host).

**Arm C:** B's parameters unchanged, composed with the worklog-34 partition-of-unity rule:
- w = clip(0.5 − s/2h, 0, 0.5), with h = 2 × the 2.25 mm dev-view pixel footprint = 4.5 mm.
- Blending only across an ownership cut inside one brick (continuous surface).
- **Declared transfer adaptation:** s is the exact distance to the Voronoi bisector of the two
  cells. Worklog 34's dense-sample approximation cannot resolve h at Lego's ~5 mm sample spacing.
- Composition data: centres plus the per-component unit table. No learned state is added.

**Ownership contract** (Θ_i = unit i's triplane + MLP):
- **Parameters:** updating unit i changes only Θ_i.
- **Output influence:**
  - under B: S_i;
  - under C: S_i ∪ B_i, where B_i is the band inside i's same-brick neighbours.

## EVIDENCE

### Contracts (`tests/test_b3.py`, 4/4 PASS)

1. **Owner on 20 k fresh on-mesh points:**
   - cell agreement 100%;
   - surface distance ≤ 0.55 mm at q99.9;
   - component agreement 98.4%. The rest are coincident hidden contact faces, which geometry
     cannot separate. This is reported, not required.
   - **Deterministic.**
   - **Tolerance change:** first set to 1e-4 m, then set to half a pixel footprint after measuring
     0.55 mm. This is a test tolerance, not a quality gate.
2. **Corner disambiguation.** A point nearer the wall by position, rendered with the floor normal,
   resolves to the floor (and vice versa).
3. **Known functions and components:**
   - the hard step of 0.60 becomes 0.0006 under PU;
   - output outside the band is exact;
   - different components never blend.
4. **Unit re-initialisation and retraining:**
   - the other units are bit-identical;
   - changed outputs = S_1 ∪ B_1 exactly.

### Static fidelity

**Held-out (primary):** 16 views, 4096 spp, seeded independently of any candidate. View-mean
PSNR uses the code tonemap on alpha > 0 pixels. |e| values are display-space LDR errors.

| arm | view-mean PSNR | Δ vs A | pooled | logL1 (×A) | hf \|e\| (×A) | seam J (interior 0.0069) | seam \|e\| | band \|e\| | off-band \|e\| (×A) |
|---|---|---|---|---|---|---|---|---|---|
| A reproduced | 33.44 | — | 33.31 | 0.0084 | 0.0162 | 0.0068 | 0.0138 | 0.0147 | 0.0139 |
| A released | 33.69 | +0.26 | 33.56 | 0.97 | 0.98 | 0.0069 | 0.0131 | 0.0141 | 0.96 |
| **B** | **33.12** | **−0.32** | 33.01 | **1.048** | **1.066** | **0.0111** | 0.0155 | 0.0161 | **0.996** |
| **C = B + PU** | **33.13** | −0.31 | 33.02 | 1.047 | 1.064 | **0.0072** | 0.0142 | 0.0154 | 0.996 |
| REINIT / REINIT + PU | 33.13 / 33.14 | −0.31 / −0.30 | 33.03 / 33.04 | 1.046 / 1.045 | 1.063 / 1.062 | 0.0112 / 0.0072 | 0.0155 / 0.0142 | 0.0161 / 0.0154 | 0.994 |

**Other sets, labelled:**
- **Dev** (8 views, 4096 spp, diagnostics): B −0.50, C −0.50 dB; logL1 1.07×; seam J 0.0115 → 0.0071
  (A 0.0067).
- **Native 40-view val** (1024 spp, noisy): A 31.99, B 31.60 (−0.39), C 31.60. Seam J
  0.0138 → 0.0117 (A 0.0113).

**Reference points:**
- The two equally valid A models (reproduced vs released) differ by 0.26 dB, so B's −0.31 dB is
  about one A-to-A spread.
- B has 1.54× A's parameters. The comparison is capacity-confounded upward: B is larger, not
  smaller.

**Per material** (held-out, B/A display |e|):
- Yellow 1.06;
- Gray 1.15;
- transparent 1.13 (9 k px);
- Black 0.98;
- Red glass 0.89 (17 k px);
- Light brown 0.92;
- Sand 0.88;
- Brown 0.87.

**Per unit** (B/A): 0.77–1.21. The worst are units 14 (1.21) and 9 (1.14).

**Screening (predeclared): Case A.**
- C: −0.31 dB ≥ A − 0.5 and logL1 1.047 ≤ 1.10.
- Not Case C: C removes 91% of B's seam excess, but its off-band error is 0.996× A, not > 1.10×.
- Not Case B: B is −0.32 dB, not < −1.0, and logL1 1.048 ≤ 1.20.

### Independent reconstruction (u* = unit 9, most training pixels, rule fixed before training)

| property | result |
|---|---|
| parameter preservation | units_changed = [9]; the 15 other units are bit-identical (digests) |
| output preservation, B (hard) | 0 changed pixels outside S_9 on all 64 evaluated views |
| output influence, C | 1 587 332 changed pixels on the 3 sets = exactly the predicted S_9 ∪ B_9; 0 outside. 1 569 301 owned, **18 031 (1.1%) in neighbours' bands** |
| recovery | unit-9 held-out \|e\| 0.0120 vs B 0.0124 (A 0.0109); image metrics equal to B |
| radiometric validity | not tested: static intervention; no geometry changed |

### Cost (RTX 5080, WSL; data in host RAM)

| | A (worklog 35) | B | REINIT (1 unit) |
|---|---|---|---|
| training | 6.0 h (75 s/epoch, 16 GB memory pressure) | **2.58 h**: median 32.5 s/epoch; 45 epochs > 40 s while the user was gaming (02:30–03:30) | **1.54 h** (21 s/epoch, 0.6× B) |
| peak torch memory | ~15 GB (with spill) | 1.94 GiB | 0.38 GiB |
| load | 0.8 s | 0.36 s | 0.34 s |
| inference / view, PyTorch, 1024² | 0.04–0.08 s | — | C: 0.24 s (per-unit loop + band; prototype) |

## OBSERVATION (`results/evaluation/36/`)

- **`01_heldout_view*`:** B and C are visually indistinguishable from A and the reference at full
  frame. The error maps share A's pattern, which is concentrated on the dark tread and the
  technic parts.
- **`02_crops`:**
  - **Seam crops:** no visible step on the studded base plate (view 0). In view 8 the dark tread
    region has errors of similar size in A and B. These are baseline inaccuracies, not partition
    errors.
  - **High-gradient crops** (gear, studs): B/C errors are slightly larger than A's along fine edges,
    consistent with hf 1.06×.
  - **C versus B:** C lowers error near the red seam contours without smoothing interior detail.
- **`03_reinit_influence`:** the unit-9 change is confined to the predicted set (27 601 px in view
  0, none outside).

## ATTRIBUTION

**What succeeded:**
- Ownership on a native, paper-comparable RNA asset: exact parameter and output preservation and
  full-state unit reconstruction.
- Static fidelity within the screening.
- Seam removal by the worklog-34 composition, transferred with one declared distance adaptation.

**Generalises from Teaset:**
- Exact ownership and reconstruction semantics.
- Seams are exposure of independent per-unit fits, removable by PU composition: 91% here vs 86%
  on Teaset C2.

**Contradicts the Teaset expectation:**
- The **interior** per-unit fidelity loss (Teaset C2 P16-PU: −0.20 dB, logL1 1.18×, off-band
  elevated) does not appear: Lego off-band error is 0.996×.
- The Teaset interior loss is therefore not an inherent property of independent units. It was
  likely tied to:
  - that asset's material regime (conductors, where shared angular learning matters more);
  - its noisier training targets (128 spp);

  Neither was isolated, so the cause remains a hypothesis.

**Remaining costs (measured):**
- −0.3 dB overall, larger near unit boundaries (band error 1.05× even under PU) and on fine
  high-frequency detail (1.06×).
- Material dependence: gray and transparent are worse, sand and brown better.
- 1.54× parameters.

**Unresolved:**
- Which mechanism produces the residual hf and band cost:
  - per-unit triplane resolution (125–196 vs A's 512 over a larger frame);
  - edge-of-domain supervision;
  - decoder width 128.
- Whether a capacity-equal B would hold.
- Seed variance of B (one seed).
- Radiometric recovery after a real geometry edit.

## NEXT DECISION (recommendation; not implemented)

Static ownership feasibility now holds on a credible native asset. The next architecture question
should therefore be **physical geometry-edit reconstruction**:
- Move a Lego part.
- Generate current-geometry native targets.
- Fully recompute the transport-dirty units (oracle dirty set) vs all units vs object level, with
  valid units preserved.

This answers radiometric validity and the sparsity of D_T at unit granularity, which every static
batch so far has left open. The residual boundary/hf cost (a representation-fidelity refinement)
is secondary and can be revisited if the edit experiment shows that boundary units dominate the
error.

Architecture Batch 4 was not started.
