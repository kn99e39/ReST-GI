# Worklog 101I — Construction qualification contract v2 (2026-11-18)

## Scope

WL101I is a suffix batch of [WL101H](101H_WORKLOG_ECRO_SSS_CONTACT_SEMANTICS_QUALIFICATION.md). It recovers and requalifies the physical-oracle construction contract. The target is **contract v2, not ECRO v2.** Nothing below changes: ECRO, ER, LOW/HIGH, learned parameters, the event contract, native semantics, native references, WL101H's contact-valid candidate semantics (h_min 2.124e-7 m), or any frozen protocol or result. WL101G stays stopped (D) and WL101H stays NOT QUALIFIED. No projection primary is run, and the candidate is not promoted. Contract v2 remains a proposal until the owner adopts it.

## v1 failure mechanisms (development evidence on WL101H seeds)

`dev101i.py` ran read-only replays on WL101H's seeds only.

- **D4 (offset sensitivity).** At seed 10204101 there were 112 flips among 163,840 samples in 40 flagged slots. Every flip is a visibility graze at an occluder silhouette. The native offset m along the source normal shifts where the ray crosses the occluder's plane by about m·tanθ.
  - Flips by occluder: the panel's free edges (y = −0.3/0.9, z = 0.3) 67, the sphere silhouette 17, Box A 13, the open-face edge 5.
  - The left wall dominates because it sees the thin panel's four free edges from 0.5 m.
  - Native visibility therefore deviates from its offset → 0 limit in bands proportional to m. This is a real native-vs-continuous deviation, not a low-support artefact (WL101H flagged slots have a median of 1,791 visible samples).
- **D3.** Two events, two mechanisms.
  - A sphere-cap sample at z = 9.3e-8 m was already excluded as singular, but was labelled `miss_other` because labels were assigned before exclusion. Label only.
  - A Box B side-face sample at z = 1.67e-7 m: the origin offset is horizontal, so the origin sits in the floor-plane float band and its ray misses the floor. The sample is included with vis = 0, so this is the audit's F3 class. v1 caught junctions only when the origin was on or behind a plane, not in the band above it.
- **S5 (subcell moves).** At seed 10203101, 119 samples sit in a different subcell in WL101G than in WL101H: 118 across the azimuth sub-bin border d = 0.5, 1 across a spatial border. Within one process the two kernel compositions give identical addresses (0 moves). In the moved slots, 129 samples have |d − 0.5| < 1e-5. Float32 d differs from a float64 recomputation by up to 1.33e-6 (atan2), and 59 of these samples fall on opposite sides of 0.5. Sampled azimuths within the atan2 error band of the midline are assigned by compilation-dependent rounding. Slot totals are unaffected.
- **S3 / S4.** v1 tested a single sampled singular event against 1e-3 per subcell. It bounded band exclusions by 2 × the maximum subcell deviation and counted junction samples by frequency. Each of these choices made harmless events fail.

## Contract v2 (frozen before any final-seed evidence)

The [protocol](../../experiments/ecro_sss101i/protocol/contract_v2.json), the gate mathematics [gates101i.py](../../experiments/ecro_sss101i/gates101i.py) and the analysis are frozen. Thresholds keep v1's values: θ_abs = 7.0e-4 and θ_rel = 2.5e-4 × max(|T|, 1e-3). This batch adds their derivation.

The prefix throughput to an event-2 query is at most 1, so a per-slot table bias b moves any pixel or region mean by at most max|b|. θ_abs is therefore 5% of the WL101F near-contact band, and θ_rel is 10% of the WL101G additive RMSE margin. Neither depends on the view or on any V0 result.

- **S3 — support accounting, no free threshold.**
  - Registered exclusions get analytic measures. The candidate's singular floor disk has r_s = √(4R·h_min − 4h_min²) = 0.583 mm, covering 6.83e-5 of subcell 3 of floor patch 155. The singular sphere cap has height 2h_min.
  - Lattice bands get upper fractions: 6.4e-5 + 2b_m(1/l₁ + 1/l₂).
  - Per replica, three tests: observed singular counts against Poisson expectations; zero singular samples in unregistered cells; no slot whose tie and boundary-miss count is significantly above the lattice band.
  - A single event is never treated as an area.
- **S4 — per-class bias accounting for U, V, G per slot and S_G per (slot, s).** Each class has a measured bias b with SE, kept separate from a conservative bound B.
  - *Singular:* continuity proxy from the WL101H candidate replay, plus a model allowance.
  - *Band:* radiance of the near-border shell, weighted as the table.
  - *Junction:* bound only, from the analytic band measure or the Poisson upper bound on the observed count.
  - *Silhouette:* paired visibility at m/10 and m/100. The bias relative to the offset → 0 limit is b = −Δ₁₀/0.9, the SE comes from per-sample contributions, and the linearity residual is the model allowance.
  - Decision per cell, Bonferroni z over 7 × 81,280 decisions: PASS if B ≤ θ; FAIL if |b| − z·SE > θ; otherwise UNRESOLVED.
- **S5.**
  - Native v2 construction at WL101G seeds must reproduce the stored slot totals: n and v identical, reals within 1e-6, and every S_G change ≤ θ.
  - Candidate instruments are re-measured, not reused: camera independence, split-path consistency and LOW invariance at all six cameras, and cos normalization.
- **Diagnostics never stop.** Labels are assigned after exclusion. D3 is split into semantic events (inside-box, support conflict) and label-only events.
- **Controls:**
  - *Native discrimination:* native semantics must fail S2.
  - *Positive control:* h_min = 1.0e-4 gives a singular disk of about 1.26 cm, roughly the F6 ring. Every contact-subcell S_G cell must FAIL S4.
  - *Unit tests* (`tests101i.py`) cover: singular geometry; S3 consistency, unregistered events and band excess; S4 silhouette FAIL, PASS and UNRESOLVED with exact recovery; the S4 singular positive control; and S5 moves.
- **Verdict.**
  - NOT QUALIFIED if any global gate, the discrimination check or the positive control fails, or a semantic D3 event occurs.
  - PARTIALLY QUALIFIED if some S4 cells FAIL or are UNRESOLVED; those cells are listed as support restrictions.
  - QUALIFIED otherwise.
- **Seeds:** final 10205101–4, rate 10205105, positive control 10205201, historical regression 10203101–3. Development used only WL101H seeds.

**Singular reconciliation.** The candidate's singular set has non-zero continuous measure: about 1.07e-6 m² on the floor and about 1.07e-6 m² on the sphere cap. Its radiance is about 0.17 by continuity, not zero. Excluded samples leave both numerator and denominator. Tables therefore estimate means over the support minus this set, and the S4 singular term bounds how far they can sit from full-support means.

**Development checks before the freeze.**
- The SE of the paired silhouette difference was first a Cauchy–Schwarz sum. It overstated the SE about 3.5× because it ignores the cancellation between correlated terms. The per-sample formula, which needs the cross term Σdg²·L, replaced it; this is an estimator correction, not a threshold change.
- The r_s formula was rewritten without float64 cancellation.
- A reduced smoke run on development seeds (64 proposals per slot) exercised every runner and analysis path. Its numbers are not interpreted.

## Final evidence (freeze commit `a4709da51b1edfb599a0595ff449c9ec8e58757f`)

The LabServer63 chain ran four steps, each with exit 0 and no OptiX warning:

1. candidate at seeds 10205101–5;
2. native at the same seeds;
3. positive control at seed 10205201 (h_min = 1.0e-4);
4. native historical regression at seeds 10203101–3 with the WL101G Amendment-1 oracle.

The frozen analysis then ran on the server. z_family = 5.35. The singular continuity proxy is L = 0.1742, with model allowance 0.0168.

### Gates that passed

- **S2.** Candidate ring 0 and inside-box 0 in all five replicas. Native ring 1,700–1,890 in every replica, so native F6 is detected.
- **S3, candidate.** Observed singular counts per replica are 7 / 7 / 3 / 5 / 9 against an analytic expectation of about 8.0. The two-sided p ranges from 0.09 to 0.91. There are no singular samples in unregistered cells and no band excess. The registered measures are a floor disk of 1.068e-6 m² (r_s = 0.583 mm, floor patch 155, subcell 3) and a sphere cap of 1.068e-6 m².
- **S5 regression.** n and v slot totals are identical to WL101G, and real slot totals agree within 8e-10. Moved samples: 119, 118 and 102. All moves are directional, none crosses a spatial subcell border, and the largest S_G change is 1.4e-9.
- **Instruments.** Candidate camera independence is exact (real difference 0.0). Split-path consistency at all six cameras is 4.4e-8 to 4.9e-8 relative L1. LOW invariance is exact. Cos normalization passes.
- **Unit tests** pass.

### What failed, and why

**S1 (candidate and native).** One invalid membership sample at seed 10205102, launch 87, slot 75845: a Box B face (patch 1185), 2.3 cm inside the patch boundary. Its direction lies in ring 0 with u2 = 3.9e-6. The exact patch-frame cosine is therefore u2/4 = 9.8e-7, just below WL101E's back threshold of 1e-6, while the recorded world-space cosine is 1.013e-6, just above it. The attachment was not evaluated even though the sample was classified as a front candidate.

The inferred mechanism: the face-specific attachment kernel re-evaluates the back predicate with different rounding from the classification kernel. This is the same compilation-dependent family as the S5 moves, and it is not proven at the instruction level. There is no wrong-surface admission, and the band is measure-zero. v2 registered no back-threshold band, so the frozen S1 correctly fails. This is a new numerical class that first appeared at final scale. The development seeds did not contain it.

**Discrimination.** The frozen predicate required native S1 to pass, and native fails S1 because of the same shared proposal. Native S2 does flag F6 in every replica.

**Positive control.** 63 of the 64 contact-subcell S_G cells FAIL; the median measured bias is −4.8e-3. Bin 4 is UNRESOLVED: its pooled weight is low (98); a singular term of −4.7e-3 is partly cancelled by a silhouette term of +1.3e-3; SE is 2.5e-3. No contaminated cell passes, but the frozen predicate required all 64 to fail.

**Frozen verdict: NOT QUALIFIED.** It rests on S1, the coupled discrimination predicate and the strict positive-control predicate.

### S4 accounting (candidate, pooled over four replicas; the batch did not reach PARTIAL because the global gates failed)

| Table | Active cells | PASS | FAIL | UNRESOLVED | Max measured \|bias\| | UNRESOLVED dominated by |
|---|---|---|---|---|---|---|
| U | 80,704 | 76,947 | 0 | 3,757 | 9.2e-6 | junction bound (all 3,757) |
| V | 73,945 | 64,558 | 0 | 9,387 | 0.23 (low-support slot) | silhouette 5,697 · junction 3,425 · band 265 |
| G | 73,945 | 64,511 | 0 | 9,434 | 0.23 (low-support slot) | silhouette 5,706 · junction 3,419 · band 309 |
| S_G | 293,602 | 273,375 | 0 | 20,227 | 1.98 (low-support cell) | silhouette 12,206 · band 4,244 · junction 3,777 |

- **No cell fails.** Every UNRESOLVED cell has a stated cause.
- **U.** All 3,757 UNRESOLVED slots contain at least one observed junction sample. The 95% Poisson upper bound (4.74/N) × (|L_j − T| + |T|) exceeds θ_rel, although the measured U bias never exceeds 9.2e-6.
- **G, silhouette.** The measured silhouette bias is median 0, p99 7.2e-4 absolute and 9.1e-4 relative, and 2,211 slots have a measured value above θ. The effect is real (native offset visibility at occluder silhouettes) but is not resolved cell by cell under the family-wise z. Unresolved G slots by surface: left wall 2,737, back wall 1,917, floor 1,680, ceiling 1,530.
- **Contact patch 155 (candidate).** All bins PASS: b ≈ −1.5e-6, B ≈ 2e-5. In native semantics the same cells read U 0.266 against 0.256 for the candidate.
- **WL101H v1 worst slots.** 43016, 80198 and 80774 all PASS, so those v1 stops were false.
- **D4 left-wall slots.** 47918 and 50807 have measured silhouette biases of −5.3e-2 and −2.5e-3 and are UNRESOLVED.

### Diagnostics

- **Candidate labels:** front 269,110,850; back_panel 5,395,010; back_sphere 1 (native 1,839); miss_opening 53,874,955; miss_other 9 (native 1,276, the F5 cap misses); excluded 7.
- **Candidate D3:** zero semantic events; 66 junction-label events (registered); 2 unregistered label events. Both are open-face corner grazes at y = −2 with z ≈ 3 and with x ≈ −2. The v2 label set omitted a seam label; this is label only.
- **Junction class:** 938–1,259 samples per replica.

## Final report

**Implementation fact.**
- Contract v2 was implemented as `construct101i.py`, `gates101i.py`, `run101i.py` and `analyze101i.py`, with `tests101i.py`. Development replays are `dev101i.py` and `diag_invalid101i.py`.
- Thresholds are unchanged from v1, and their derivation is now recorded.
- WL101H's candidate semantics, ECRO, the event contract and the native references are untouched. No projection evaluation was run.

**Measurement.** As above. The positive control and the native run used the same v2 gates.

**Observation.**
- v2 resolved the v1 false stops:
  - S3 now matches its analytic model;
  - S4's former worst slots and the contact patch pass;
  - S5 slot totals reproduce WL101G exactly;
  - the D3 events are explained.
- The candidate shows no measured material bias in U at any slot.
- A new unregistered membership event appeared only at final scale (the back-threshold tie).
- Native silhouette visibility produces a measurable but individually unresolved deviation of about 0.1% in G/V across about 6k slots, concentrated where thin occluders' free edges are seen at short range.

**Interpretation.**
- v2's per-class accounting does separate harmless events from contamination: F6 is caught in S2, and a 1.26 cm singular exclusion is caught in S4 in 63 of 64 cells.
- It is not yet sufficient evidence for qualification, for three reasons:
  1. **Incomplete numerical register.** The membership register is incomplete for direction-threshold ties. One real final-scale event stops S1, and running the full-scale geometry pass on the development seeds did not reveal it.
  2. **Insufficient statistical power.** About 13% of G slots and 7% of S_G cells are UNRESOLVED. The causes are conservative junction bounds in slots with an observed lattice sample, and a real silhouette deviation at the θ scale.
  3. **Over-strict control predicates.** The positive-control and discrimination predicates were stricter than their purpose: no contaminated cell passed, yet they report failure.
- A comparison restricted to PASS cells would be defensible only after an owner decision on each of these.

**Unresolved questions.**
- Whether to register the back-threshold band (and other direction-coordinate threshold bands) in membership. Validate by a targeted replay.
- Whether native silhouette visibility, a semantics shared with the native/candidate camera reference, belongs to the declared measure or is a correction target.
- Whether junction bounds should use the analytic band measure only, instead of observed-count Poisson bounds.
- Whether positive controls should require "no PASS" rather than "all FAIL".

These are v3 design questions for the owner. They were not tuned here.

**Architecture verdict.**
- No architecture change follows. The candidate stays an experimental semantics and is not promoted.
- The contract is **NOT QUALIFIED** under v2 as frozen, for the reasons above.
- **Completion answer:** v2 does **not yet** provide sufficient, independently reproducible evidence to qualify the contact-valid estimator for a new camera-independent U/G comparison. It does show that the remaining obstacles are an incomplete numerical registry, statistical power, and two control-predicate formulations, not a material semantic defect of the candidate.

**Reproducibility.**
- Commit: freeze `a4709da`; this report is the next commit. Protocol: `experiments/ecro_sss101i/protocol/contract_v2.json`.
- Environment: LabServer63 RTX 3080 Ti, OptiX 595.71.05, Mitsuba 3.8.0, Dr.Jit 1.3.1, Python 3.10.21 (`~/NeuralShaderDemo/external/8dna26/.venv`), via `run101i.sh` / `chain101i.sh`.
- Seeds: final 10205101–4, rate 10205105, positive control 10205201, regression 10203101–3, independence 10205001, smoke 10205300+.
- Executed-code hashes are in each run manifest.
- Analysis: `PYTHONPATH=…/experiments/ecro86 python analyze101i.py results/ecro_sss101i/final results/ecro_sss101g/primary_a2 results/ecro_sss101h/review_server/metrics.json OUT`.
- Raw results are under `results/ecro_sss101i/`; review artifacts and hashes are in `evaluation/101I/`.
