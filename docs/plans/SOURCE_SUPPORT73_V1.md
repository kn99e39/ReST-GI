# WL73 Stage 0 design record — fixed-budget source-support intervention (src73_v1)

**Status:** frozen design for one bounded diagnostic batch (Worklog 73). It is not an architecture proposal and adopts nothing. All three spatial representations (D1, A, B) and their objectives stay unchanged; only the training-source composition changes.

All port selections below are **geometry only**. The rules are applied by `experiments/fnto_src73/ports73.py` to the archived rest/pt0/te0 geometries. No response, model output or WL71/WL72 error value is read. The exact lists are frozen in `protocol/src73_v1.json` (`frozen`), and the server chain recomputes them and stops on any difference.

## 1. S16 coverage audit (geometry only)

The slab is 100 × 100 × 10 mm with 3,840 ports of 2.5 mm. Boundary contexts are defined on the rest geometry:

| Context | Definition | Ports | S16 sources | S16 exact-image union (136 ports) |
|---|---|---|---|---|
| INT | top/bottom, both in-plane edge distances ≥ 10 mm | 2048 | 16 | 136 |
| TBE | top/bottom edge band (nearest edge < 10 mm, farther ≥ 10 mm) | 1024 | 0 | 0 |
| TBC | top/bottom corner zone (both < 10 mm) | 128 | 0 | 0 |
| SM | side, \|z\| = 1.25 mm row, along-edge ≥ 10 mm | 256 | 0 | 0 |
| SR | side, \|z\| = 3.75 mm row (next to the top/bottom edge), along-edge ≥ 10 mm | 256 | 0 | 0 |
| SV | side, < 10 mm from a vertical slab edge | 128 | 0 | 0 |

**Coverage facts:**
- Every S16 source is interior, and its edge distance is ≥ 21.25 mm.
- Edge, corner and side conditions are absent from S16, including through its exact symmetry images.
- Local thickness is 10 mm at rest for all sources. At pt0 it is 4.42–10 mm (dent).

**Redundancy facts:**
- Slots 2, 6, 8, 9 and 12, at (±28.75, ±28.75) on the top or bottom face, form a single exact D4 × z-flip orbit.
- All five lie at least 11 mm outside the pt0 dent (rest-xy distance from the dent centre ≥ 33.6 mm against R = 22.5 mm), so their 1 cm neighbourhood is undeformed in both training geometries.
- They are exact duplicates of one supervised condition **at rest only**. At pt0 they sit at different distances from the dent, and the response covers every receiver, so their pt0 supervision is distinct. This holds for A too: A's pt0 symmetry images go to the pt0 variant geometries, and only the asw image fixes pt0.
- The other S16 sources are unique by this criterion. Slots 0/3, 1/7 and 10/13 are z-flip pairs at rest but differ at pt0.

The per-slot audit (face, context, edge distance, rest/pt0 thickness, pt0 displacement, distance to the pt0 centre, nearest other source and nearest other source image, orbit) is written to `ports.json` → `audit`.

## 2. S16-R (exactly 16 sources, 4 replaced)

**Removal rule** (predeclared, geometry only):
- Among flat-interior S16 sources (INT context, rest-xy distance from the pt0 dent centre ≥ R + 10 mm), remove every exact rest-symmetry duplicate of another flat-interior source, keeping the lowest slot of each orbit.
- The rule must remove exactly 4 sources, or the design is infeasible.
- **Result:** slots 6, 8, 9 and 12 are removed (ports 2707, 2895, 2732, 944). Slot 2 is kept.

**Replacements:** one source in each unsupported boundary context.
- Each is the nearest admissible port to a target 31.25 mm from the face centre along the edge.
- Admissible means:
  - not an exact image of any previously referenced port (WL66/68/71 slots 0–15 and 20–27, WL72 slots 40–47) or of an earlier replacement;
  - min-bin (1 − β) ≥ 0.5 at rest and pt0.
- The 31.25 mm position leaves a mid-edge region about 30 mm from every image of the new sources (the "far" stratum), while a 5 mm "near" position still exists.

| Slot | Context | Port | Rest position (mm) | min (1−β) |
|---|---|---|---|---|
| 50 | TBE (1.25 mm edge row, top) | 3439 | (48.75, −31.25, 5) | 0.725 |
| 51 | TBC (slab vertex, bottom) | 3823 | (48.75, 48.75, −5) | 0.627 |
| 52 | SM (side mid row) | 408 | (−50, 31.25, 1.25) | 0.990 |
| 53 | SR (side rim row) | 527 | (−31.25, −50, 3.75) | 0.719 |

**Budget:**
- Every S16-R source has rest and pt0 references with 8 replicas: fit replicas 0–3, evaluation replicas 4–7.
- Each is 32M photons per (geometry, bin, replica), identical to S16.
- The 12 retained sources reuse their archived bundles bitwise.
- The new references cost 1,024 packets.
- SV is deliberately left unsupported, and serves as a control context.

## 3. Independent confirmation (geometry only)

**Exclusions:**
- No confirmation port equals an exact image of any previously referenced port, any S16 or S16-R source, or another confirmation port.
- Static ports are at least 10 mm apart, to decorrelate them.

**Distances:**
- dA is the 3-D distance to the nearest *actual* S16-R training port. It is relevant to the position-dependent W+UV models (D1, A without the symmetry term).
- dI is the distance to the nearest *exact image* of an S16-R port. It is relevant to A's symmetry supervision and to the position-free RSK.
- "Near" minimises |dA − 5 mm|, so a near port is near for every architecture.
- "Far" maximises dI, so a far port is far from every image and therefore from every actual port.

| Group | Slots | Contexts | dA / dI (mm) |
|---|---|---|---|
| INT_DED | 60–63 | interior, edge ≥ 21.25 mm (S16 support range); near/far × top/bottom | near 5 / 2.5–3.5; far 16 / 10.3 |
| BND_NEAR (primary) | 64, 66, 68, 70 | TBE, TBC, SM, SR, 5 mm from an actual new source | 5 / 5 |
| BND_FAR (primary) | 65, 67, 69, 71, 90–92 | TBE, TBC, SM, SR farthest from every S16-R image; second far port for TBE, SM, SR | 16–33 / 10.6–21.7 |
| REM_NEAR (primary, removal cost) | 76–79 | interior, 5–5.6 mm from each actual removed S16 source | 17.7–22.9 / 5–5.6 (d to the removed source 5–5.6) |
| UNS | 72–75 | SV (unsupported by both populations) | 8–21 / 5.3–8.5 |
| FP (rest and te0) | 80–85 | interior footprint of te0 | 5–10 / 2.5–5.6 |

All static confirmation is on rest with K = 4 replicas. FP has rest and te0 references, K = 4.

**Inference limits, stated before results:**
- TBC "far" (slot 67, 8.75 mm from both edges, ballistic β ≈ 0) is confounded with edge distance. The corner zone is only 10 × 10 mm.
- SV near/far differ only modestly in distance, because every vertical edge lies within about 10 mm of a TBC source image. UNS is therefore descriptive: no source lies in that context, but it is adjacent to the images of the new TBC source. It is not a support-specificity control.
- The TBC corner zone is too small for a second far port.
- Within INT, the maximum attainable dI is 10.3 mm, because S16's images cover the interior densely.

**Deformation pose:**
- Rule: among in-family WL57 poses never used for a port-probe reference and not WL59 training poses (te0, te1, te3, va0, va1, cf0, cf1), choose the one whose dent centre is farthest from every D4 image of the probed dented poses' centres (pt0, cf2, cf3, te2); ties go to the larger radius.
- Result: **te0** (centred, depth 5 mm, R 20 mm; 7.07 mm from the nearest pt0 image).
- The archived WL57 te0 geometry (sha256 `79216410…`) is used, identity-gated against a fresh LabServer63 build.

**Footprint ports:**
- Six top and bottom ports within 0.6 R of the te0 centre: top dz −4.8, −3.8 and −2.0 mm; the bottom ports sit under the thinned region.
- WL72 found that only ports in or under the dent qualified (signal/SE 185–3,533, against ≤ 2.8 outside). Six such ports make the ≥ 4 requirement feasible.
- No in-family dent reaches a side, edge or corner port, so deformation is measurable only for interior sources. Static and deformation endpoints are reported separately.
- **Confound:** te0 overlaps the pt0 training dent. The centre offset is 7.07 mm, R is 20 vs 22.5 mm and depth is 5 vs 5.75 mm. DEF73 is therefore a check of interpolation near the training pose. It does not test transfer to new deformations, and S16 vs S16-R differences in DEF73 are not attributed to boundary support.

## 4. Stage 1 reproduction (before any new reference)

1. **Evaluation identity.** eval73 must reproduce WL72's archived pooled bounds for the six historical models (WL71 D1, WL72 A/B) on S16, FIN71, H3, CONF72 and DEF71, within a relative 1e-4.
2. **Training-path identity.**
   - Problem73 on S16 must equal `fit71.Problem`: targets and objective within 1e-6, because GPU index_add aggregation is not bitwise deterministic.
   - 500-step replays of D1_5901, A_5901 and B_7201 through the WL73 path must reproduce:
     - the archived initial objective within 1e-5;
     - the archived logged J trajectory within 1 %.
   - Replays are controls, never candidates.

Any failure stops the batch.

## 5. Fits and decision (summary; full text in the protocol)

**Runs:**
- Six fits: D1R_5901/5902, AR_5901/5902 and BR_7201/7202. Each uses the family's unchanged constructor, initialisation or seed, objective and optimiser (Adam, cosine 3e-3 → 3e-5, 20,000 steps, clip 1.0).
- The final iterate is primary. Checkpoints @2000/6000/10000 are descriptive only.

**Comparisons.** All comparisons are three-valued over the two seeds of each condition, using worst-channel upper 95 % bounds w of con_coarse and abs_coarse.
- *better(G)*:
  - YES if, for both metrics, max over new seeds ≤ 0.8 × min over old seeds;
  - NO if, for some metric, min over new seeds > 0.8 × max over old seeds;
  - otherwise OVERLAP.
- *worse(G)* is defined the same way with factor 1.25.

**Per architecture** (S16 seeds vs S16-R seeds):
- **fit_ok:** either the WL59 gates pass on TRAIN_NEW (rest and pt0), or w(TRAIN_NEW) ≤ 1.25 × w(TRAIN_KEEP) for the same model. The deciding basis is reported.
- **no deficit:** every S16 seed already passes the gates on BND_FAR.
- **far:** better(BND_FAR).
- **near:** better(BND_NEAR).
- **accurate:** the gates pass on BND_FAR.
- **regression:** worse(INT_ALL) or worse(REM_NEAR).

| Outcome | Class |
|---|---|
| a gate fails or a primary group is unmeasured | INCONCLUSIVE |
| fit_ok fails | INSUFFICIENT FIT |
| no deficit | NO SUPPORT DEFICIT |
| far = OVERLAP | INCONCLUSIVE (seed variance) |
| far = YES, with regression | TRADE-OFF / REDISTRIBUTION |
| far = YES, accurate | SUPPORT-SENSITIVE GENERALIZATION |
| far = YES, not accurate | partial support sensitivity and SUPPORT INSUFFICIENT (accuracy) |
| far = NO, near = YES | SUPPORT-SENSITIVE (local only) and SUPPORT INSUFFICIENT (transfer) |
| far = NO, near not YES | SUPPORT INSUFFICIENT |

A regression adds TRADE-OFF in the last two rows. A regression OVERLAP is reported as a possible regression.

**Reporting:**
- DEF73 (te0 − rest) is reported separately and is UNMEASURED if fewer than 4 matrices qualify.
- The cross-architecture ranking at S16-R uses better(BND_FAR) and is descriptive only.

The design was revised once before freezing, after an independent pre-freeze review. The review found that pooled BND let 5 mm interpolation decide, that the degradation test could not see the removal cost, that overlapping seeds were read as "no improvement", and that the duplicate rationale was wrong at pt0. The revision added BND_NEAR/BND_FAR as separate primary groups, REM_NEAR, the second far ports, the three-valued comparisons, the relative fit criterion and the te0 confound.

## 6. Known confounds (not removable within this batch)

- Source location and boundary context cannot be manipulated independently: near/far is crossed with context, but at different attainable distances.
- Removing the four duplicates removes actual supervision at those positions for D1 and B at rest, and for all three architectures at pt0. Only A keeps them at rest, through slot 2's exact images. This is the fixed-budget trade-off.
  - It enters the decision through REM_NEAR.
  - TRAIN_REMOVED is descriptive, and in-sample for the S16 models.
  - D1-vs-A differences are not pure support-transfer differences.
- B differs from D1/A in initialisation and parameter count. A's symmetry term changes effective supervision, not source count.
- A better fit to new support does not by itself establish transfer, and a failure to transfer does not establish representational impossibility.
