# Worklog — FIA Post-Collision Angular Expressivity Falsification (2026-11-07)

## Origin and authorisation

- **Brief.** The user pasted the WL77 brief "FIA Post-Collision Angular Expressivity Falsification". It authorises ONE bounded change: replace WL76's ring-uniform outgoing lobes with an azimuth-resolving, D4-equivariant outgoing response.
- **Unchanged.** The first-interaction / second-collision factorization, the anchors, the quadrature and the spatial support stay as they were.
- **Placement and paths.** LabServer63 (main). Code: `experiments/fnto_fia77/`. Protocol: `protocol/fia77_v1.json`. Evidence root: `~/rest_fia77`.
- **Preserved verdicts.** WL76's Stage 1 FAIL (with its post hoc amendment) and its INSUFFICIENT FIT verdict are preserved.
- **Not authorised:**
  - changes to the anchors, the quadrature or SS / W;
  - a second architecture;
  - per-port corrections;
  - threshold or training tuning;
  - checkpoint selection;
  - adoption, or a successor batch.

## Design (frozen before any WL77 candidate evidence)

**Removed degrees of freedom.** WL76 ring lobes are uniform inside each of the 4 polar rings of the 8x8 concentric grid. Every azimuthal variation inside a ring is therefore removed: 60 of the 64 per-receiver directions. By the WL76 post hoc diagnostic, 0.45 of the reference multiple-scatter native signal lies in that variation.

**New local outgoing distribution.**

O(o) = Σ_{m<3} α_m L_m(o) + α_az Λ(o)

Λ(o) = ν(o) [1 + s·a1·cos(φ_o − φ_r) + s·a2·cos 2(φ_o − φ_r)] / Z

- α is a softmax over the four components (the three ring lobes and Λ) per pair and channel.
- ν is a learned ring profile.
- φ_o is the azimuth of bin o's centre in the receiver frame (t, b).
- φ_r is the azimuth of the tangential part of the anchor → receiver vector in the same frame. It is geometric, not learned.
- s = |r_t| / (|r_t| + 1 mm).
- |a| ≤ 1/2.
- Z is the exact per-pair normaliser.

**Properties.**
- Nonnegative and normalised, so positivity, linearity and passivity are unchanged.
- Invariant under in-plane rotation or reflection of the frame. It is therefore exactly D4-equivariant wherever bins permute.
- The runtime uses an exact 8-function basis: 3 ring lobes plus ν × {1, cos, sin, cos2, sin2}. A shared int32 CSR index stores it.

**Limitation.** The coarse tail keeps the ring lobes, and its share is reported.

**Initialisation.** WL76 constructors and seeds 7601 / 7602. The azimuthal-lobe weight starts at about 0.04 with a = 0. There is no warm start.

**Unchanged machinery.** The geometry, SS, W, the quadrature (seeds, q = 1024, rescaling) and the objective are imported from `fnto_fia76/fia76.py`, not copied.

**Gates.**

| Stage | Gate | Stop outcome |
|---|---|---|
| 1 | The FIA-AZ relaxation's projection floor of the reference multiple-scatter part (training replicas) must be ≤ 0.5 × the ring floor | STOP |
| 2 | Full-runtime state ≤ 4 GiB and 64-light apply ≤ 10 × A75 | STOP |
| 3 | WL76 fit gate, unchanged | Attribution (brief Stage 4) and STOP |

**If Stage 3 passes.** Reuse the unused WL76 Stage 4 population, after hash and independence checks. Run the transfer test against AR, A75 and the preserved F76.

**Local tests: 11/11 pass.** They cover:
- the WL76 physics, CSR, source-independence and objective tests;
- bin azimuths;
- per-pair normalisation and positivity;
- a synthetic azimuthal target, fitted with residual < 0.05 by the lobe form against > 0.3 by ring lobes;
- full native D4 × z-flip equivariance with exact bin permutations.

## Execution (LabServer63, 2026-11-07)

1. **Freeze.** Commit 7b4e7a0. `run77.sh` started the dev smoke at 02:47 UTC.
2. **Dev smoke, non-evidence:**
   - Tests: 11/11 on torch 2.3.1.
   - Stage 1 expressivity: computed on the same training references as the evidence run (after the freeze). It failed the gate; with `R77_DEV_SKIP_GATES` the dev path continued.
   - Stage 2 cost: passed.
   - The dev fits had started when I stopped the dev chain at about 02:55 UTC. The Stage 1 outcome is fixed by the training references, and later stages would not run in the evidence chain, so there was nothing left for the dev chain to test.
   - Stopping the dev processes with `pkill -f` also killed my ssh session, because the pattern matched it. The remaining processes were then killed by PID.
3. **Evidence chain** (`ev77.sh`, root `~/rest_fia77`):
   - Tests: 11/11.
   - Stage 1 expressivity: **FAIL, STOP** (frozen rule), recorded in `stage1/expressivity.json` (sha256 3d20ec0a…).
   - Per protocol, Stage 2, training, Stage 3 and the transfer test were not run.

## Final report (2026-11-07)

### Implementation Fact

- **FIA-AZ** is WL76 FIA with an azimuth-resolving, D4-equivariant outgoing lobe ν·[1 + a1·cos(φ_o − φ_r) + a2·cos 2(φ_o − φ_r)]/Z, mixed with the ring lobes. The geometry, SS, W, quadrature, anchors and objective are imported unchanged from `fia76.py`.
- **Physical contracts hold.** All 11 focused tests pass:
  - positivity, zero, linearity, column sums ≤ 1, rebuild identity, CSR = reference scatter;
  - per-pair normalisation;
  - full native D4 × z-flip equivariance with exact bin permutations (r90, mx + z-flip, within 2 × quadrature-seed noise);
  - on a synthetic azimuthal target, the lobe form fits with residual < 0.05 against > 0.3 for ring lobes.
- **Dev cost** (non-evidence; structural, independent of trained weights):

  | Measure | FIA-AZ | Note |
  |---|---|---|
  | State | 1.63 GB | 4 GiB gate |
  | Rebuild | 71–72 s | 70.5 s quadrature, unchanged by design |
  | 64-light apply | 0.495 s | A75: 0.086 s → 5.75 ×, inside the 10 × gate |
  | 1-light apply | 4.6 ms | |
  | Physics | passes | |

### Measurement

**Stage 1** (16 S16-R rest sources, archived training bundles). The table gives the per (incoming bin, receiver, channel) least-squares residual of the reference multiple-scatter part M = mean(S) − albedo·SS, as relative L2.

| Basis | Residual (mean) | Range | Replica-noise floor |
|---|---|---|---|
| Ring indicators (4; WL76 restriction) | **0.453** | 0.437–0.479 | 0.0019 |
| FIA-AZ relaxation, ring × {1, cos(φ_o − φ_r), cos 2(φ_o − φ_r)} (12) | **0.385** | 0.366–0.417 | 0.0017 |
| Free phase, ring × {1, cos, sin, cos2, sin2}(φ_o) (20) | **0.352** | 0.334–0.378 | 0.0016 |

- The exact single-scatter share of the reference was 0.267.
- **Gate:** FIA-AZ / ring = 0.85, against a required ≤ 0.5. **FAIL, STOP.**

### Observation

- The WL76 ring restriction is real: 0.45 of the multiple-scatter native signal is in-ring structure, far above noise.
- Receiver-relative first and second azimuthal harmonics recover only about 15 % of it.
- With free orientation, harmonics up to order 2 still leave 0.35. Inside a ring, bins differ only in azimuth, so the remaining in-ring structure is azimuthal at order ≥ 3, or sharp features the low-order expansion cannot form.
- The pattern is the same for interior and boundary training sources: 0.37–0.42 at the four new boundary sources.
- The source of that higher-order structure was not attributed. Candidates: the bin-grid geometry near grazing, or footprint-wide superposition of many anchor directions.

### Interpretation (the brief's questions)

1. **Does azimuth-resolving R_θ remove the WL76 native floor?** No. This representation reduces the floor only from 0.45 to at most about 0.39, as an upper bound from the relaxation. It does not substantially relax the restriction.
2. **Can it fit the supervised response?** Not tested. Stage 1 stopped the batch before training, as the protocol requires.
3. **Does coarse contrast remain inaccurate after angular expressivity improves?** Not reached: the expressivity did not materially improve.
4. **Is second-collision anchoring still a plausible limitation?** Yes, still untested. WL76's contrast error remains unattributed.
5. **Does any gain transfer?** Not measured.
6. **FIA vs rank-8 / rank-16 learned ATO at matched source budgets?** Only WL76 evidence exists: FIA fits far worse on supervised sources. WL77 adds no learned comparison.
7. **Rebuild and inference cost?** Dev measurement: rebuild about 71 s against 0.015 s for W+UV, dominated by the unchanged quadrature; 64-light apply 5.75 × A75; state 1.63 GB.
8. **Does the evidence justify continued investment?** This batch shows only that low-order receiver-relative azimuthal modelling is insufficient. The remaining multiple-scatter angular structure needs higher-order directional resolution, which this bounded batch was not authorised to test. Whether that is worth pursuing is the user's decision.

### Unresolved Question

- What carries the remaining order ≥ 3 in-ring structure (0.35 of the multiple-scatter native signal).
- Whether a representation that can express it, for example a per-pair full 64-bin outgoing distribution or a higher-order harmonic or learned-basis expansion, would fit. Its cost in state and apply is untested.
- The WL76 contrast error and anchor-snapping questions remain open.

### Architecture Verdict

**STOP at Stage 1: the angular restriction is not relaxed** by the predeclared bounded representation (0.385 vs 0.453; ≤ 0.226 required).
- **Classification:** none of ANGULAR LIMIT SUPPORTED, FIT RECOVERED or TRANSFER SUPPORTED can be assigned, because no candidate was trained. Whether the WL76 failure is primarily caused by outgoing angular expressivity is **INCONCLUSIVE**. The ring restriction is a large, real share of the native signal, and low-order harmonics do not remove it.
- **Status:** FIA is not adopted. The preserved WL73 / WL75 / WL76 results and verdicts are unchanged. The unused WL76 Stage 4 population remains unscored.

### Reproducibility

- **Code:** `experiments/fnto_fia77/` at 7b4e7a0, protocol `protocol/fia77_v1.json`. Geometry and quadrature come from `fnto_fia76/fia76.py` at the same commit.
- **Evidence:** `LabServer63:~/rest_fia77` (`stage1/expressivity.json`, `tests/tests.json`, logs).
- **Dev:** `~/rest_fia77_dev` (Stage 1 / 2 records, stopped fits; `STOPPED.txt`).
- **Repository copies:** `results/fnto_fia77/`: `stage1_expressivity.json`, `dev_stage2_cost_probe_nonevidence.json`, `tests.json`.
- **References:** archived WL68 / WL71 / WL73 training bundles (S16-R rest; K replicas).
- **Environment:** RTX 3080 Ti, torch 2.3.1, float64 projections.
