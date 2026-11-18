# Physical-oracle construction qualification contract v1 (WL101H; proposal, not adopted)

**Status.** A design for qualifying physical-oracle table construction before any projection primary. It was written in WL101H after the [WL101G forensic audit](../audits/wl101g_forensic_audit/WL101G_FORENSIC_AUDIT.md) and its [proposed contract](../audits/wl101g_forensic_audit/PROPOSED_CONSTRUCTION_QUALIFICATION_CONTRACT.md). It records what WL101H implemented and measured. **Adoption is the research owner's decision.** It does not authorise a projection primary. The measured qualification outcome is in [WL101H](../worklogs/101H_WORKLOG_ECRO_SSS_CONTACT_SEMANTICS_QUALIFICATION.md).

## 1. Two owners, two kinds of gate

| Owner | Protects | May stop? | Examples in WL101H |
|---|---|---|---|
| **Semantic** | A table value, the physical support, or the evaluation instrument | Yes | S1 support identity · S2 origin validity · S3 per-slot excluded measure · S4 per-slot bias bound · S5 instrument consistency |
| **Diagnostic** | A label or an attribution | Never; always reported | D1 mechanism labels · D2 back-receiver attribution · D3 unregistered-event packets · D4 offset sensitivity |

A label cannot change a table, so it never stops a run (WL101G's two containment stops had zero table effect). A defect that can change a table must have a semantic gate, even when it produces no "miss" (audit F6).

## 2. Ordered stages

1. **Q0 register.** List every geometric feature closer than about 3 × the largest native offset: tangent contacts, junctions, coplanar overlaps, convex edges, open-face edges, two-sided sheets and address seams. For each, record:
   - its native-origin degeneracy region, computed analytically;
   - its float band;
   - its class;
   - its treatment.

   A finite-area degeneracy needs a decision on its semantics (§3).
2. **Q1 targeted stress.** Run deterministic, geometry-only probes of each register entry at metric and lattice offsets. Every tolerance is derived from these probes or from a float error model with a declared rule and safety factor. Frequencies are never used to set a tolerance. WL101H derived:
   - **h_min**, the contact-valid origin floor: 4 × the largest failing half-gap = 2.12e-7 m;
   - **b_m**, the metric membership band: 16 ulp(3 m) = 3.8e-6 m, checked against an edge stress that missed only at insets ≤ 1e-7 m.
3. **Q2 full-scale preflight on the exact frozen seeds** (plus one independent rate seed), before freezing. It runs:
   - membership, point identity, origin validity for every spawn the estimator actually traces, mechanism classes and labels;
   - **class-split radiance sums per (slot, s, t)**, so every class's table effect can be computed offline.
4. **Freeze**, and only then any evaluation stream.

## 3. Semantics ledger (native vs continuous physics)

Each register entry with a native deviation is recorded with:

- the deviation and its measure;
- its measured radiance effect;
- the chosen treatment.

There are three possible treatments:

- **Keep native** and report the entry as a confounder.
- **Candidate semantics** with a matched reference. The same tracer serves the construction suffix and the camera reference, and it is never compared with the native reference as if they were the same quantity.
- **Support redefinition** with explicit per-slot excluded-measure accounting.

WL101H's candidate for the sphere/floor contact keeps native rays wherever the native origin is in free space. Only where the native offset reaches the other solid does it move the origin to the free-gap midpoint. Where no representable free-space origin exists (half-gap < h_min) it removes the event or sample from the support.

## 4. Gate forms

- **S1 support identity:**
  - no invalid proposal;
  - no valid point inside another solid by more than b_m;
  - an attachment miss counts as a boundary miss only within the metric band b_m of a patch boundary.
- **S2 origin validity:**
  - no included sample has a traced origin inside a solid;
  - an NEE origin is checked only where the NEE ray is actually used: light on the front side, inside the cone, non-SSS vertex;
  - origins on a junction plane are a registered, S4-bounded class.
- **S3 excluded measure, per slot and per subcell:** at most 1e-3. Bounds are per slot, never global frequencies.
- **S4 bias bound, per slot:** measure × measured radiance deviation, at most 7.0e-4 luminance absolute and at most 2.5e-4 × |G| relative. Three sources are bounded:
  - the analytic band measure of ties and boundary misses;
  - the measured singular fraction;
  - the junction fraction × the class radiance difference.
- **S5 instrument consistency, all of:**
  - split query vs full trace at every view camera;
  - LOW invariance;
  - planar cos normalization;
  - camera independence of construction (integer columns identical, real columns at the reduction-order floor);
  - reproduction of earlier stored packets where the code path is unchanged.
- **Discrimination check:** the same gates run on the known-defective native semantics must flag the defect, while its harmless labels pass. Otherwise the gates cannot separate semantic contamination from diagnostics, and the batch stops.

## 5. Governance

- An amendment requires a replay of the triggering samples on the traced ray and a mapping to a register entry. It is followed by a full Q1 + Q2 rerun.
- The owner approves how each of their restrictions is operationalised.
- A tolerance calibrated on data that a later fix changes must be re-derived.
- Every run manifest records the hashes of the **executed experiment code**, not only the renderer sources. WL101H's `run101h.py` does this.
- A Dr.Jit/OptiX "Disabling this ray tracing operation" or a PTX compile error invalidates the run (exit 99).

## 6. Outcome of v1 in WL101H — not validated

**What worked.**
- The ownership split and the discrimination check work: S2 caught the native contact-ring defect in every replica, and the native S1 passed.
- Instrument checks S5 passed at all six cameras: split consistency, LOW invariance, cos normalization and camera independence.
- The candidate semantics passed every correctness-oriented test.

**What did not.** v1 stopped on harmless events. A v2 needs four changes, none implemented or approved:

1. **Measure, not frequency.** S3 and S4 must use an analytic measure (singular disk or cap area, band widths) with a statistical bound. A single sample in about 1,000 must not decide them.
2. **Bounded deviation.** The S4 deviation bound (2 × the maximum subcell deviation) is too conservative in bright, high-variance cells. Replace it with a bound tied to the decision margins it protects.
3. **Junctions by metric distance.** Classify a junction by the requested point's metric distance to it (the float band on either side), not by which side of the plane the origin lies. The above-plane Box B/floor lattice sample (audit F3) was unregistered under v1.
4. **Slot-level reproduction.** Require exact slot totals, and bound subcell-border flips caused by float address rounding (about 1e-7 of samples).

**D4.** The native grazing visibility deviation measured by D4 (relative 8e-4 in about 100 well-supported slots) needs an owner decision: keep it as a native confounder, or define a threshold from image-level materiality.
