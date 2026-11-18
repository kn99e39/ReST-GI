# PROPOSAL — Construction qualification contract for physical-oracle table construction (post-WL101G audit, 2026-11-18)

**Status: proposal only. It requires human approval.**
- It does **not** amend or replace the frozen WL101G protocol (`experiments/ecro_sss101g/protocol/generalization.json` and Amendments 1–2).
- It does **not** authorise a relaunch, a successor batch, a sampler change, a tolerance change, reclassification of historical samples, or any change to the physical target.
- It does **not** recommend continuation. It states what would have to be true **before** another primary run of a WL101G-type construction could be trusted.

The evidence for each requirement is in [WL101G_FORENSIC_AUDIT.md](WL101G_FORENSIC_AUDIT.md). Finding IDs (F1–F7) refer to that report.

---

## 1. Principles

1. **Qualify at full scale, before the freeze.** Construction geometry is deterministic and cheap: about 21 s per replica, radiance included, in WL101G. All geometry and validity gates are therefore run on the **exact frozen seeds and proposal counts** before the protocol is frozen. A frozen construction then cannot meet an unqualified rare event (F1, F3).
2. **Every gate has an owner.**
   - A **semantic** gate protects a table value or the physical support, and may stop a run.
   - A **diagnostic** gate assigns labels or attribution. It reports and never stops.
3. **Tolerances are metric and derived.** Boundary bands come from a float32 error model (coordinate ulp × conditioning, plus native offsets), not from parametric fractions or observed counts (F1).
4. **Bounds are per-slot and measure × radiance, not global frequency.** A class is negligible only if its per-slot measure fraction times a measured radiance difference falls below a declared fraction of the decision margins. Event frequency alone is never accepted as proof.
5. **Separate correctness from numerics.** Deviations of native semantics from continuous physics (F5, F6, the F2 grazing band) go in a different ledger from float-rounding ties (F1, F3, F4). Only the owner decides how a correctness deviation is treated.

---

## 2. Stage Q0 — Contact and seam register (a document, written before any code)

Enumerate every geometric feature with a separation smaller than **3 × the largest native offset**. For each, record its kind, an analytic degeneracy region, a metric float band, the expected per-replica count from geometric measure, a proposed owner, and a treatment.

**Native offsets to cover.**
- `spawn_ray` and `spawn_ray_to`: (1 + max|p|)·1500·2^-24 along n. Verified to within 0.04% on four replayed rays.
- The SSS exit continuation: 1e-4 m along the exit direction.
- The attachment probe: 2 mm start, 3 mm reach.

**Feature kinds.** Tangent contacts, 90° junctions, coplanar overlaps, convex free edges, open-face edges, two-sided sheets, address seams.

**Analytic degeneracy region.** The set of sampled points whose native origins fall inside or behind another solid. For example: the sphere/floor ring r* = √(2R·mag) = 1.17 cm, and the cap h = mag = 0.17 mm.

**Metric float band.** On the 2^-23 lattice of each sampled coordinate.

**Treatment.** Keep, flag, or exclude. Any finite-area degeneracy region needs an explicit owner decision.

For the WL101 SSS cfg0 scene, the audit's §3 inventory (8 feature classes) is the starting register. It must be checked against the frozen geometry, not copied.

---

## 3. Stage Q1 — Targeted boundary stress tests

These are deterministic and use only geometry and native transport: no camera, no reference image, no held-out data.

**Sampling, for each register entry:**
- Lattice offsets u = k·2^-23 and 1 − k·2^-23 for k = 0 to 64.
- Metric offsets {0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3} m on both sides of the feature.
- All 64 bins, plus forced grazing directions (n·w ∈ {1e-3, 1e-2, 0.1}).

**Records, per sample:**
- Attachment outcome: shape, face, normal, re-address. Include a graze onto the adjacent face at convex edges (unresolved Q4).
- Visibility outcome, and the receiver's shape and face.
- **Origin validity:** analytic point-in-solid tests for every native origin type (sphere and boxes analytic; room interior).
- **NEE first-hit sense:** entry or exit. This tests the F6 assumption in `pt100.direct` that a sphere hit is an entry.
- Continuation termination.

**Pass criteria:**
- Every outcome maps to its register entry.
- The metric tie band equals the observed maximum excursion × a declared safety factor, recorded with its derivation.
- Every label is checked against an exact analytic classification on synthetic cases that include offset origins. This is the test that would have caught F2.

---

## 4. Stage Q2 — Whole-domain construction-scale preflight

1. Run the complete construction **geometry** on the exact frozen seeds and counts: membership, visibility, origin validity and class labels. Run it once more on at least one independent seed set to estimate rates.
2. Persist **per (slot, s, t) counts per class**, and **class-split radiance sums** (ΣL and ΣL² per class), inside the construction packet. Each class's table effect can then be computed without a GPU replay. The audit could detect F6 only because a single visible/non-visible split happened to be stored.
3. Optionally, with owner approval, include construction radiance on the frozen seeds. This is construction-side and touches no held-out data. It measures the per-slot radiance effect of each finite-area class, as the audit did for F6.

---

## 5. Gate ownership and form

**Semantic gates (may stop):**

| ID | Gate | Proposed form |
|---|---|---|
| S1 | Support identity | No admission of a wrong shape, face or address outside the registered metric band. |
| S2 | Native-origin validity | Zero samples whose native origins lie inside or behind another solid, except in a registered region the owner has approved. |
| S3 | Per-slot excluded measure | For every class excluded from numerator and denominator: the maximum over slots of the excluded fraction stays ≤ ε_ex, unless the class is registered with a radiance bound. |
| S4 | Per-slot radiance-effect bound | For every class that changes L or a weight: the measured \|Δtable\| per slot (from the class-split sums) stays ≤ θ. Proposed: θ ≤ 0.1 × the 0.0025 additive RMSE margin and ≤ 0.05 × the contact bands. The values are for the owner to set before the freeze. |
| S5 | Instrument consistency | Keep the existing Stage 0 event/query, LOW invariance, lookup, cos normalization and camera-independence gates. They performed as designed. |

**Diagnostic outputs (never stop, always reported):**

| ID | Output |
|---|---|
| D1 | Miss labels, assigned by **mechanism** (origin and feature from the register), not by exit location. |
| D2 | `back_receiver` attribution by cause. |
| D3 | Saved packets for any unregistered event. Because Q2 runs on the frozen seeds, D3 should be empty for construction. In evaluation streams it is reported for owner review. |

---

## 6. Ledgers

| Ledger | Contents |
|---|---|
| L1 Correctness (native vs continuous physics) | Per register entry: the deviation, its measure, its measured radiance effect, and the owner's decision. Initial entries: F6 floor ring (material), F5 sphere cap, F2-type grazing band, SSS exit escapes below the floor. |
| L2 Numerical | Every float tolerance, with its derivation, the observed maximum and the margin. This covers the membership band (replacing 4·eps and 8e-6), the direction/coordinate tolerances, and the camera-independence floor. |
| L3 Labels | Diagnostic counts per class and replica. |

**Rule.** Every negligibility claim cites a measure bound and a radiance bound from L1 or L2. A frequency alone is not accepted.

---

## 7. Amendment governance

1. Before an amendment is proposed, the triggering samples are replayed **on the traced ray**, and their mechanism is mapped to a register entry (new or existing).
2. Any amendment re-runs Q1 and Q2 in full, not only the replica that failed.
3. The owner explicitly approves how each of their restrictions is operationalised before the rerun. For example, WL101G Amendment 1 read "same kind of miss recurs" as more than 1e-6 of eligible proposals, about 330 events per replica.
4. A tolerance or gate value widened during development is re-derived whenever a defect fix changes the data it was calibrated on. Example: WL101G's seam gate, 2e-5, and its 1e-3 m edge band were calibrated on counts contaminated by the F2 defect.
5. The run manifest records the hashes of the **executed experiment code** (sampler, construction, gates), not only the renderer sources.

---

## 8. Decisions requested from the research owner

These are not recommendations to continue.

1. **Floor-ring semantics (F6).** For any WL101-line comparison of table measures at the contact, choose one:
   - (a) keep native semantics and report the ring as an explicit confounder of the U-vs-G difference at the contact slot;
   - (b) flag native-degenerate support and report tables both with and without it, descriptively;
   - (c) another option.

   Changing the reference or the physical target is outside this audit's authority.
2. **Whether to authorise a read-only GPU replay** confirming F6 per sample (unresolved Q1), and, separately, an image-level attribution of F6 on existing WL101E/F arms (unresolved Q2).
3. **Living-document annotation** (proposed text only; not applied by the audit), for the WL101G entry in `docs/RESEARCH_ROADMAP.md`:
   > *Independent audit ([docs/audits/wl101g_forensic_audit](audits/wl101g_forensic_audit/WL101G_FORENSIC_AUDIT.md)): the three stop triggers had no table effect. The audit found a gate-invisible native contact-ring defect: native origins lie inside the sphere within 1.17 cm of the tangency. It raises the uniform tables at the contact floor slot by +2.3–2.7% (+0.006–0.007 luminance) and is excluded by V/G. U-vs-G comparisons at the contact require an owner decision on this region before interpretation. Verdict D unchanged.*
4. **Batch classification and naming** of any follow-up, under the numbering policy in AGENTS.md. The owner decides this.
