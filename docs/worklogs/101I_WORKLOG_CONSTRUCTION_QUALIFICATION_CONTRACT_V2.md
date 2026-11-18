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
