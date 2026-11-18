# WL101G — Independent forensic audit of the stopped construction and validation pipeline (2026-11-18)

**Status.** Read-only audit. No experiment was resumed, no sampler, tolerance, gate or historical sample was changed, and WL101G was not relaunched. The WL101G protocol, amendments, worklog, outputs and verdict D are preserved unchanged. The proposed corrected contract is in [PROPOSED_CONSTRUCTION_QUALIFICATION_CONTRACT.md](PROPOSED_CONSTRUCTION_QUALIFICATION_CONTRACT.md) and **requires human approval**. It authorises nothing.

**Independence.** The auditor is a fresh assistant session that had no access to the WL101G session's context. It is the same model family as the implementing agent, so its independence is procedural, not institutional. The WL101G worklog's "account for the audit" was treated as a set of claims. Each claim below was checked against code, preserved artifacts, or a focused read-only reproduction.

**Evidence labels used throughout.**
- **[CODE]**: read directly from the source at the frozen commits.
- **[REPLAY]**: a preserved log, replay packet or stored construction packet, read without re-execution.
- **[REPRO]**: a new non-mutating numpy computation, CPU only, no renderer. Scripts and outputs are in [repro/](repro/).
- **[INF]**: inference that follows from the above but was not measured per sample.

---

## 0. Answer to the completion question

**Why did WL101G need repeated amendments?** Three gates were frozen without having been run at the scale they would face. The gates had zero tolerance, and the construction faces about 1.3e9 deterministic proposals. Each stop came from a rare event class that a 5e6-sample smoke run (1/65 of one replica) could not expose.

Construction is cheap: about 21 s per replica at full scale. It is also deterministic [REPLAY]:
- replica 0's valid, buried and back counts are identical in all three executions;
- its front, back-receiver and tangency counts are identical in the two executions that reached the containment stage. A full-scale geometry pass on the frozen seeds before freezing would have exposed all three stops, at no risk to held-out data.

Behind the scale gap are four design faults:
1. **A dimensionally wrong tolerance.** The inherited WL101E edge-miss rule is a parametric fraction of the patch, not a metric distance. On Box B's shortest patch axis it is narrower than one float32 ulp of the coordinates. It was set in WL101E from a single observed event.
2. **Symptom-level classification.** Containment classes were defined by where a miss exits the room, not by the mechanism that produced it. A classifier defect (exit computed from the surface point, not the traced origin) was absorbed during development by widening tolerances.
3. **No contact register.** Contacts and seams were not enumerated geometrically before the categories were written.
4. **Misallocated stop power.** Gates that can only change a diagnostic label could stop the batch. A gate-invisible defect that does change table values had no gate at all.

**Which errors could have affected the physical estimator?** The three triggers that stopped WL101G did not change any table value. Amendment 1 excluded one sample. Amendment 2 and the final stop changed only labels, because visibility counts only front-facing hits [CODE].

The audit did, however, find an **estimator-relevant defect that no WL101G gate can see**. The floor ring of radius **1.17 cm** around the sphere/floor tangency is admitted as valid physical support. In that ring every native spawned origin (offset 1.70e-4 m) lies **inside the sphere** [CODE + REPRO]. As a result:
- the WL101E/G radiance estimator records **unattenuated** direct light as an "E2 through-sphere ballistic" event. The physical attenuation would be about 4.6e-4.
- the visibility ray is labelled `back_receiver`, which no gate checks.

The stored packets show the predicted signature quantitatively [REPLAY]:
- In the contact subcell, the non-visible sample fraction has a per-bin median of 2.73–2.97%, against a predicted 2.73%.
- Those samples are 4.6–4.7× brighter than the visible samples.
- The uniform tables (Un, and by the identity test U_E) are therefore **+2.3–2.7% higher at the contact floor slot** (+0.0061 to +0.0070 luminance). The affected spatial subcell is about **+10%** higher.
- V and G exclude the ring. Part of any U→G difference at that slot therefore comes from this native defect, not from the étendue measure.

**What prequalified contract is needed before another primary can be trusted?** In outline (details in the proposal):
1. A geometric contact and seam register, with an analytic native-origin degeneracy region for each entry.
2. Targeted boundary stress tests, with metric tolerances derived from a float error model.
3. A whole-domain, construction-scale geometry preflight on the exact frozen seeds.
4. Explicit semantic versus diagnostic gate ownership. Diagnostic labels report but never stop. Semantic gates bound the per-slot excluded or anomalous measure and its radiance effect, not the global frequency.
5. A separate correctness ledger (native vs continuous physics) and numerical-error ledger.
6. An owner decision on the floor-ring semantics before any U-vs-G comparison is interpreted.

**Is the current estimator qualified?** **No.** The radiance pipeline reproduces native semantics, but those semantics are physically wrong in a registered-but-unrecognised contact region. Its V/G weighting near contacts has never been tested against an independent visibility model. This audit does not declare the estimator qualified.

---

## 1. Implementation fact

Data flow per construction proposal, traced at commits `2f9f575` / `0d6f60a` / `7298e9b` [CODE]. The inherited files `oracle_field.py`, `pt100.py`, `pt86.py`, `query_readout.py` and `projections.py` are byte-identical across all three commits and HEAD.

| Step | What happens | Code |
|---|---|---|
| 1. Domain coordinates | Slot `= pid·64 + bin`. PCG32 `(seed·1000003 + launch)` draws `u0,u1` (area) and `u2,u3` (direction in bin). Values lie on the 2^-23 float lattice. | [oracle_field.py:76-91](../../../experiments/ecro_sss101e/oracle_field.py#L76-L91) |
| 2. Requested point | Planar: `PO + u0·PE1 + u1·PE2` in float32. Sphere: analytic. Patch frames come from the patch tables. | [pt86.py:183-193](../../../experiments/ecro86/pt86.py#L183-L193) |
| 3. Burial | Analytic only, for three cases: floor strictly inside Box B's footprint, Box B underside, and sphere `p.z ≤ 0`. `back` is `w·n ≤ 1e-6`. | [oracle_field.py:92-101](../../../experiments/ecro_sss101e/oracle_field.py#L92-L101) |
| 4. Native attachment | Ray from `p + 0.002·n` along `−n`, `maxt` 3 mm, against a **single-shape scene**. It checks shape, distance ≤ 4e-6 m, normal ≥ 0.999999, re-address and native front. It cannot see other shapes. | [oracle_field.py:103-119](../../../experiments/ecro_sss101e/oracle_field.py#L103-L119) |
| 5. Tie / invalid | `tie` = attached-but-mismatched within 8e-6 of any coordinate boundary, **or** a planar miss within **4·eps** (WL101E). Amendment 1 adds planar misses within 8e-6 of the area boundary. Anything else non-valid is `invalid`, which stops the run. | [oracle_field.py:120-129](../../../experiments/ecro_sss101e/oracle_field.py#L120-L129), [gpu101g.py:13-34](../../../experiments/ecro_sss101g/gpu101g.py#L13-L34) |
| 6. Radiance L | `FirstSurface` injects the native hit. `pt100.trace` in probe mode adds the k=1 *ballistic* term as E2, then NEE and BSDF continuation. Continuation and NEE origins use Mitsuba `spawn_ray` / `spawn_ray_to`. The SSS exit uses `ep + ew·1e-4`. | [oracle_field.py:148-157](../../../experiments/ecro_sss101e/oracle_field.py#L148-L157), [pt100.py:96-137](../../../experiments/ecro_material100/pt100.py#L96-L137), [pt100.py:67-94](../../../experiments/ecro_material100/pt100.py#L67-L94) |
| 7. Visibility (V/G only) | `spawned = hit.spawn_ray(w)` is traced in the full scene. `front` is a hit whose `to_local(−w).z > 1e-6`. `vis = front`, `g = vis·cos_x`. | [gpu101g.py:81-98](../../../experiments/ecro_sss101g/gpu101g.py#L81-L98) |
| 8. Containment labels | A miss is labelled `opening` if it exits through y = −2, `seam` if the exit lies within 1e-3 m of two room planes, `tangency` if the sample is a sphere point with `p.z < 1e-3` and `w.z < 0`, and `unexplained` otherwise (stop). Before A2 the exit was computed from `hit.p`; after A2 from `spawned.o`. **No label enters any column.** | [gpu101g.py:37-53](../../../experiments/ecro_sss101g/gpu101g.py#L37-L53), [gpu101g.py:84-96](../../../experiments/ecro_sss101g/gpu101g.py#L84-L96) |
| 9. Accumulation | Per (slot, s, t): `n, c, v, g, L, LL, vL, gL`. | [gpu101g.py:97-103](../../../experiments/ecro_sss101g/gpu101g.py#L97-L103) |
| 10. Tables | Ratio estimators: `Un = ΣL/Σn`, `V = ΣvL/Σv`, `G = ΣgL/Σg`. A zero denominator falls back to Un (V, G) or to the parent (subcells). Numerator and denominator always exclude the same samples. | [tables101g.py:14-56](../../../experiments/ecro_sss101g/tables101g.py#L14-L56) |
| 11. Validation | Membership, tie, Amendment-1 recurrence, excursion, containment and seam/tangency gates per replica. Then cos normalization, uniform identity and camera independence. | [run_generalization.py:66-77](../../../experiments/ecro_sss101g/run_generalization.py#L66-L77), [run_generalization.py:190-251](../../../experiments/ecro_sss101g/run_generalization.py#L190-L251) |

**Native spawn offset.** The spawn offset is not in repo code; it is Mitsuba's `offset_p`. The audit modelled it as `|Δo| = (1 + max_i|p_i|)·1500·2^-24` along the geometric normal. This matches all four replayed rays to within 0.04% [REPLAY + REPRO]:

| Replayed ray | Measured offset (m) | Predicted (m) |
|---|---|---|
| A2 trigger, ray 1 | 2.12481e-4 | 2.12407e-4 |
| A2 trigger, ray 2 | 2.13086e-4 | 2.13109e-4 |
| Final stop, ray 1 | 2.28284e-4 | 2.28262e-4 |
| Final stop, ray 2 | 2.17688e-4 | 2.17673e-4 |

At the sphere/floor contact (`max|p| ≈ 0.9`) the offset is **1.70e-4 m**.

**Consistency of origins between estimator and classifier.**
- The radiance suffix and the visibility ray both start from the same native hit, with the same `spawn_ray` offset rule. They are consistent with each other and with forward native semantics.
- The pre-A2 classifier was not consistent with the traced ray; A2 fixed this.
- Two inconsistencies remain:
  1. **Membership qualifies the point, not the origins.** The attachment test runs against a single shape. It never checks that the origins the estimator actually uses (`spawn_ray`, `spawn_ray_to`, SSS `ep + 1e-4·ew`) lie in free space on the requested side. This gap admits the floor ring (§2, F6).
  2. **The tangency predicate is broader than its comment.** The code comment says "whose spawned ray starts below the floor plane", but the code tests only `hit.p.z < 1e-3` and `w.z < 0`. It never tests `spawned.o.z < 0`. The category therefore spans 1 mm, while the mechanism spans 0.17 mm.

---

## 2. Verified failure mechanisms

### 2.1 Failure matrix

| ID | Trigger / location | Requested physical event | Actual native event | Root cause (verified) | Code | Tolerance involved | Inclusion & normalization | Radiance effect | Evidence |
|---|---|---|---|---|---|---|---|---|---|
| **F1** (A1 stop, `primary/`) | membership `invalid=1`; replica 0, seed 10203101, launch 57, slot 77555, **Box B +y′ face, cell (0,0)** (pid 1211), bin 51 | Native attachment of a point on the +y′ face, 5.96e-8 m from its convex vertical edge with the −x′ face | Attach ray hits nothing | float32 rounding of a yawed point within half an ulp (ulp 1.19e-7 m) of a **convex mesh edge**. WL101E's miss-tie band of **4·eps parametric is 4.77e-8 m on Box B's 0.1 m axis**, less than one coordinate ulp. The rule cannot cover this class at any sample count. | oracle_field.py:125-128; A1 gpu101g.py:13-34 | 4·eps → 8e-6 (= 8e-7 m on this axis, ≈ 6.7 ulp) | 1 sample moved from invalid to tie. Excluded from numerator and denominator alike, so the ratio is unaffected. | None measurable. Excluded measure ≤ the boundary band (≤ 2×8e-6 of the u-range per axis). L is smooth at the edge. | primary.log; primary_membership_replay.log; amendment1_verification.log; [REPRO r1] reconstructs the point to float precision |
| **F2** (A2 stop, `primary_a1/`) | containment `unexplained=2`; replica 0, launches 20/55, slots 76036/76037, **Box B −x′ face centre cell** (pid 1188), bins 4/5; n·w = 0.089/0.069 | Continuous segment from the surface point crosses x = 2 at y = −1.99896, **1.04 mm inside the right wall** (a front receiver) | The spawned origin is 2.125e-4 m along n, so the ray crosses 7/42 µm **outside** the wall's front edge and escapes through the opening. The classifier computed the exit from `hit.p`, which is neither opening nor near an edge, so it was labelled unexplained. | Origin mismatch between classifier and traced ray. Confirmed. | gpu101g.py:88 (pre-A2 `hit.p`) | Edge band 1e-3 m (widened from 1e-4 before the freeze to absorb P3) | None. `vis = 0` with either label. | None on tables. Native forward visibility differs from continuous visibility in a grazing band of width ~ offset/\|cos\|. [INF] negligible, not measured. | primary_a1.log; containment replay; ray retrace; spawn-origin check: A1 → A2 replica 0 counts opening +524, seam −522, unexplained −2, all others identical [REPLAY] |
| **F3** (final stop, `primary_a2/`) | containment `unexplained=2`; replica 3, seed 10203104, launches 56/117, **Box B −y′ face bottom row** (pid 1203 cell (1,0), pid 1202 cell (0,0)), bins 54/51 | Segment from the side face downward. The floor's front face is at distance 0 and 9.7e-8 m, so it is visible. | Horizontal spawn offset leaves the origin at p.z = **0 and 5.56e-8 m** (u1 = 0 and 2·2^-23 on the 0.233 m z-axis). The floor hit at t ≈ 0 is missed, the ray leaves below the floor, and it is labelled unexplained. | A float32 **u-lattice sample on a measure-zero 90° junction**. This is the same mechanism as the `seam` class at room junctions. It falls outside `seam` only because labels are assigned by exit location. It is **not** equivalent to the sphere tangency (contrary to the WL101G worklog). | gpu101g.py:84-96; oracle_field.py:96-99 | None. The seam band applies only near room edges; tangency is sphere-only. | None. `vis = 0`; the sample remains in Un. | None measurable. Physical vis would be 1. The affected measure is ≈ (K+1)·2^-23 of the bottom-row u-range [INF]. | primary_a2.log; primary_a2_containment_replay.log; [REPRO r1]; lattice model: expected ≈ 0.56/replica vs observed 2 in 4 replicas [INF] |
| **F4** (`seam`, recurring, gated) | 15/15/18/7 per replica after A2. Before A2: 540 in replica 0, of which 522 were opening escapes. | Junction segments that physically reach the adjacent surface | Missed at t ≈ 0 near room junction edges | The same lattice-junction mechanism as F3 [INF, order-of-magnitude ≈ 11/replica expected]. None of the post-A2 seam events was ever replayed. | gpu101g.py:50-53 | Gate ≤ 2e-5 of valid. It was set before the freeze from counts contaminated by the F2 defect and is now about 360× looser than the observed rate. | None | None measurable | primary_a1_containment_spawn_origin_check.log; progress.json |
| **F5** (`tangency`, recurring, gated) | 1315/1270/1281/1312 per replica, in the 8 sphere ring-0 patches | Sphere-cap point heading downward. The floor front is at sub-mm distance, so it is visible. | The spawned origin along the sphere normal is **below the floor plane** for p.z < 1.70e-4 m (cap radius 1.17 cm). The ray misses and leaves below the floor. | **Deterministic geometry of the spawn offset at a tangent contact, not a numerical event.** The model predicts **1237 ± 35** per replica, against 1270–1315 observed (+3 to +6%, unexplained). | gpu101g.py:91; Mitsuba offset_p | Gate ≤ 2e-5 of all valid. This is global; per-slot it is 7.5e-4 of the ring-0 sphere samples. | Excluded from V/G for every direction from the cap; included in Un | L is unaffected (the SSS walk starts inside the sphere). V/G bias per affected slot ≤ 7.5e-4·\|L_cap − L_slot\| [INF, measure-derived] | [REPRO r2]; progress.json |
| **F6** (**no gate**; found by this audit) | Floor patch 155 (x 0.75–1.0, y 0.25–0.5), spatial subcell 3, **r < 1.17 cm around the tangency** (0.9, 0.4, 0): 0.683% of the patch, 2.73% of the subcell, ≈ 1,790 valid samples per replica | Outgoing E2–E5 radiance of a positive-gap floor point under the sphere. It is shadowed: the physical ballistic attenuation along the 0.641 m chord is exp(−12·0.641) = 4.6e-4. | Every spawned origin at z = 1.70e-4 m lies **inside the sphere**. (a) The NEE shadow ray's first hit is the sphere's far boundary, an *exit*. `pt100.direct` treats it as an entry, so `de ≈ 0` and the light passes **unattenuated** into the "through" term, recorded as E2. (b) The BSDF continuation hits the sphere interior and terminates, so E3+ = 0. (c) The visibility ray hits the sphere from inside and is labelled `back_receiver`, so vis = 0. | Membership qualifies the point but not the native origins. The burial rules know only `p.z ≤ 0` on the sphere. `direct()` assumes the first sphere hit is an entry. | pt100.py:78-89; oracle_field.py:96-99, 103-116; gpu101g.py:83-86 | None. `back_receiver` is ungated. | **Valid** in membership: counted in Un/U_E, excluded from V/G | **Measured signature** in stored packets. Subcell 3: non-visible mean luminance 1.12–1.16 vs visible 0.31–0.32. Non-visible fraction per bin: median 2.73–2.97% (predicted 2.73%); subcell 0: median 0. **Un − V at slot level +0.0061/+0.0061/+0.0070 (+2.3–2.7%); subcell 3 +0.032–0.034 (≈ +10%).** Implied ring-sample mean luminance 1.47–1.56. In other subcells and neighbouring patches Un ≈ V within noise. | [REPRO r3, r4]; [CODE]. Per-sample mechanism is [INF], not GPU-replayed. |
| **F7** (`back_receiver`, ungated) | 1.643% of valid samples per replica | n/a | A mixture: the panel's back side (claimed dominant), the F6 ring (≈ 1,790/replica), and sphere-cap rays going up through the floor from below (≈ 10/replica [REPRO r2]) | Never classified by cause at full scale | gpu101g.py:93 | none | Excluded from V/G | Unknown beyond F6 | progress.json; development diag only |

### 2.2 Verification of the WL101G worklog's three root-cause claims

| Worklog claim | Audit ruling | Reason |
|---|---|---|
| **(1) Amendment 1:** "inadequate requalification of a numerical membership rule" after a 4× scale-up; the event was likely at about 3e-9 per proposal. | **Partially confirmed; the root cause is deeper.** | The event, location and mechanism are confirmed. The rule was not merely under-qualified at scale: it is in the wrong units. A parametric band means different metric widths per patch axis (4.8e-8 to 1.6e-7 m). On Box B's 0.1 m axis it is below one float32 ulp of the coordinates. This is the **second** frozen-run occurrence of the class: WL101E's first frozen failure was a ceiling edge at 1·eps. The 4·eps rule was set from that one event with an ad-hoc margin. The worklog's "no wrong-surface hit" is vacuous, because the attachment scene contains only the requested shape [CODE]. |
| **(2) Amendment 2:** "classifier used the surface point while the ray was traced from the spawned origin; rays were genuine escapes; labels only." | **Confirmed, with one qualification.** | Code diff and replay agree, and the counts move exactly as stated. "Genuine escape" is true of the **native traced** ray only. The continuous segment from the surface point hits the wall 1.04 mm inside the edge. The label fix is correct; the native-vs-continuous visibility deviation remains (small, unmeasured). "No table value can change" is confirmed [CODE]. |
| **(3) Final stop:** "box–floor equivalent of the zero-gap sphere tangency; the tangency rule was written for the sphere only." | **Partially confirmed; the mechanism equivalence is incorrect.** | Incomplete contact enumeration is confirmed. But the sphere tangency is a **finite-area spawn-offset** effect (a cap 0.17 mm high and 1.17 cm in radius). The box–floor events are **float-lattice samples exactly on a 90° junction** (u = 0 and 2·2^-23), the same mechanism as the `seam` class. The worklog's rate "once per 1.6e8" was taken from one replica. The lattice model gives about 1 per 6e8, consistent with 2 events in 4 replicas. |
| "Tangency misses (≈ 4e-6) … numerical events at the 1e-9 level" | **Not confirmed.** | Tangency is a deterministic geometric consequence of the spawn offset, predicted to within about 5% [REPRO r2]. It is not numerical. |
| "about 1.6% back-facing, mostly the panel's back face" | **Unverified at full scale.** | The class also contains the F6 ring, which carries a material radiance effect. |
| "The containment check … cannot change any table value" (common cause 3) | **Confirmed.** | It also cannot detect the class that does change table values (F6). |
| Roadmap: "None of them reflects on the projection being tested." | **Correct for the stop triggers; incomplete as a statement about the estimator.** | F6 bears directly on the U→G comparison at the contact slot. The audit does not edit the roadmap (scope). The annotation it proposes is in the proposal, §8. |

### 2.3 Why the pre-freeze tests missed each failure

- **F1.** The smoke run had 5.2e6 proposals, so the expected number of events was about 0.015. The rule had never been derived from a float error model, and the WL101E precedent of the same class was not treated as a warning.
- **F2.** The development diagnostic `diag_containment.py` used the same defective `room_exit(hit.p, …)`. The one corner graze it found (P3) was absorbed by widening the edge band from 1e-4 to 1e-3 m and the seam gate from 1e-6 to 2e-5. No self-test checked `room_exit` against a ray-traced ground truth (selftest101g covers only tables and estimators). After A2 the widened values stayed in place.
- **F3/F4.** Categories were written from observed development events, not from a register of contacts and junctions. The lattice probability of a sample landing exactly on an edge (≈ 2^-23 per draw per lattice step) predicts about 0.5 box–floor events and about 10 junction events per replica, and about 0.01 in the smoke run.
- **F5.** The rate was observed in development (15, then 26), but the event was labelled numerical rather than modelled, so its per-slot concentration and meaning were never assessed.
- **F6/F7.** No test asks whether the estimator's native origins lie in free space. `back_receiver` has no gate, and the uniform-identity gate compares Un with U_E, which share the same defect. The "camera-independent" and "cos normalization" gates are blind to it by construction.

---

## 3. Contact and boundary inventory (complete for this scene)

The scene is the WL101A SSS cfg0 geometry: room x, y ∈ [−2, 2], z ∈ [0, 3], open at y = −2; panel at x = −1.5; sphere with centre (0.9, 0.4, 0.4) and r = 0.4; Box A floating (z 0.25–1.25); Box B resting on the floor (z 0–0.70, yaw 0.292). Gaps were checked from the frozen geometry: sphere–Box B 0.27 m; Box B to the right and back walls 0.28 m and 0.72 m; Box A to the floor 0.25 m [REPRO r1 and geometry in §1].

| # | Contact or boundary | Kind | Native-origin degeneracy | Lattice/float band | WL101G handling | Table-relevant? |
|---|---|---|---|---|---|---|
| 1 | Sphere–floor | Tangent point (0° wedge) | **Finite area.** Floor ring r ≤ 1.17 cm, where origins are inside the sphere (F6). Sphere cap h ≤ 0.17 mm, where origins are below the floor (F5). | — | Cap: `tangency` label. Ring: **unrecognised** (`back_receiver`). Burial only at `p.z ≤ 0`. | **Yes (F6)**; F5 small |
| 2 | Box B–floor | 4 convex-on-plane edges plus a coplanar overlap (underside = floor footprint) | None (vertical faces offset horizontally; floor offset vertically) | Bottom-row u-lattice (F3); footprint boundary `|u| < half` in float32 | Footprint and underside buried; edges unregistered, hence the stop | Negligible by measure |
| 3 | Room junctions: floor–3 walls, ceiling–3 walls, 2 vertical wall–wall corners | 90° concave | None | Lattice bands on both sides (F4) | Labelled `seam` only because the exit lies near two room planes | Negligible |
| 4 | Room open-face edges: floor, ceiling and both side walls at y = −2 | Free edge at the escape boundary | Grazing displacement of order offset/\|cos\| (F2 type) | Attach-miss band (WL101E's original event) | `opening` / `seam` | Small, unmeasured [INF] |
| 5 | Convex box edges: Box A 12, Box B 12 | Convex mesh edges | — | Attach-miss band (F1); possible graze onto the adjacent face (normal mismatch, so `invalid`; unregistered, rate unknown) | A1 for misses only | Negligible |
| 6 | Panel | Thin two-sided sheet with 4 free edges | Back side receives `back_receiver` | Attach-miss band | `back_receiver` | Excluded by definition |
| 7 | Sphere azimuth seam and poles | Address seam (not geometric) | — | Address tie within 8e-6 | `tie` | Negligible |
| 8 | SSS exit continuation `ep + 1e-4·ew` near the floor | A separate offset rule | Exits within about 1e-4 m above the floor heading down escape below it | — | Unregistered | Native semantics; small [INF] |

---

## 4. Gate design assessment

| Gate | Ownership (what it protects) | Strictness vs actual risk |
|---|---|---|
| Membership `invalid == 0` | **Semantic** (support identity) | Right purpose, wrong boundary definition. It merges harmless rounding at edges (measure zero) with wrong-surface admission (harmful). The boundary band is parametric, not metric. At about 1e9 proposals, a stop is inevitable without an error-model-derived band. |
| Tie fraction ≤ 5e-5 of eligible | Semantic (excluded measure) | A **global frequency** cap. It does not bound per-slot bias: ties could concentrate in one slot. It should be a per-slot excluded-measure bound. |
| Amendment-1 recurrence > 1e-6 of eligible | Governance | Operationalised by the agent as about 330 events per replica, roughly 100× the incident rate. The owner's text read "stop … if the same kind of miss recurs." Whether the owner approved this reading is not recorded separately (§7). |
| Address excursion, re-address | Semantic | Appropriate. The thresholds were set before the freeze from a conditioning argument. |
| Containment `unexplained == 0` | **Diagnostic only** (labels; [CODE]: no column uses them) | **Over-strict.** It stopped the batch twice with zero table effect. It is also **blind** to non-miss defects (F6). |
| Seam / tangency ≤ 2e-5 of valid | Diagnostic | Global frequency. Calibrated on defect-contaminated development counts. It does not express per-slot concentration. Tangency is deterministic geometry and needs a measure bound, not a rate. |
| `back_receiver` | — (**no gate**) | **Under-strict.** It carries the only class found with a material table effect (F6). |
| Cos normalization | Semantic (sampler) | Good; planar only. |
| Uniform identity (Un vs U_E) | Reproducibility | Good for reproducibility, but it cannot detect shared semantic defects (F6 is in both). |
| Camera independence | Semantic for the "camera-independent" claim | Well designed (integer columns exact, real sums within the measured reduction-order floor). |
| Lookup and Stage 0 consistency | Semantic (evaluation instrument) | Good. Passed in all three executions. |

**Summary.** Stop power sat on diagnostic labels with no table effect (two of the three stops). The table-relevant semantic question was unasked: do the estimator's actual native origins and events match the requested physical event within the registered support? Every rare-event gate was frequency-based and global, while the risk is per-slot and measure-based.

---

## 5. Estimator validity assessment

1. **Radiance L per accepted sample.** It is unchanged from WL101E, and WL101E showed probe = camera-mode suffix to 1e-6. It is therefore **native-correct**. It is **not physically correct** in the F6 floor ring. There the native renderer's `direct()` treats an inside-sphere origin's first sphere hit as an entry and books unattenuated light as E2 [CODE]. The stored packets show the predicted signature at the predicted rate [REPLAY + REPRO]. The physical reference renderer contains the same native defect, but reaches it only at native camera-query density. The tables average it over the 25 cm patch at area density (Un/U_E) or remove it (V/G).
2. **Measure and weights.**
   - Un is correct as the WL101E uniform measure, defect included.
   - V/G use forward native visibility from the spawned origin. This deviates from the continuous "receiver population" measure that G claims to estimate in four places:
     - (a) lattice junction bands (negligible by measure);
     - (b) the tangency cap (0.075% of ring-0 sphere samples);
     - (c) the F6 ring (physically visible from the sphere underside, natively vis = 0);
     - (d) grazing bands at visibility boundaries, of width ~ offset/\|cos\|.
   - No test compares native forward visibility with an independent continuous or reciprocal visibility. The replay's "reverse from wall" probe re-hit Box B 2.4 mm from the source. It was not a clean reciprocity test.
3. **Normalization.** Correct ratio estimators. Excluded samples leave numerator and denominator together, and fallbacks are explicit [CODE]. Self-tests reproduce the nested sums.
4. **Amendments.** A1 changes the inclusion of one sample. A2 changes no table value. Neither alters the physical semantics.
5. **Conclusion.** The estimator is **not qualified** for interpreting U → G or S_G changes at the contact. At the contact floor slot the comparison is confounded by about 2.3–2.7% (slot) and about 10% (subcell). This comes from the native ring defect, not from the étendue measure. Its effect on image-level endpoints (contact, sphere, silhouette) is **unmeasured**. It is not offered as an explanation of the WL101C/E 0.10–0.14 m excess lobes, which lie outside the 1.17 cm ring. Away from contacts, nothing found here impugns the estimator.

---

## 6. Measurement versus inference

| Claim | Basis |
|---|---|
| All 52 manifest files present with matching SHA-256; protocol and amendment hashes match their commits | [REPLAY], verified here |
| Executions ran the recorded renderer sources (11 repo files identical after LF→CRLF; server copies use CRLF) | [REPLAY] + hash check |
| Construction geometry is deterministic across executions (replica 0 valid/buried/back identical in all three runs; front/back_receiver/tangency identical in `primary_a1` and `primary_a2`) | [REPLAY] |
| Spawn-offset formula | [REPLAY] 4/4 rays to within 0.04% |
| F1 point is on the +y′/−x′ convex edge at 5.96e-8 m; 4·eps band = 4.77e-8 m there | [REPRO r1] |
| F2 mechanism and label-only effect | [CODE] + [REPLAY] |
| F3 samples at p.z = 0 / 5.56e-8 on Box B's bottom row | [REPLAY] + [REPRO r1] |
| F3 and F4 share the lattice-junction mechanism; expected rates | [INF] (order-of-magnitude model; no seam event replayed) |
| F5 tangency is deterministic spawn-offset geometry | [REPRO r2] (1237 ± 35 vs 1270–1315) |
| F6 mechanism (origin inside sphere, unattenuated E2, vis = 0) | [CODE] + [REPRO] geometry; **per-sample [INF]** |
| F6 signature and magnitude in tables | [REPLAY] stored packets via [REPRO r3, r4] |
| F6 effect on image endpoints | **Unmeasured** |
| A2 rays would hit the wall under continuous geometry | [REPLAY] (exit computed from `hit.p`) |

---

## 7. Unresolved questions

1. **Per-sample confirmation of F6.** A GPU replay of the patch-155 subcell-3 samples is needed to confirm per sample: origin inside the sphere, `cross` set with de ≈ 0, E3+ = 0. This would be a read-only focused reproduction; it was not run (local machine has no CUDA stack).
2. **Image-level effect of F6** on the WL101E/F/G arms and the physical reference, including the native camera-query density inside the ring. Not authorised here.
3. **Identity of the post-A2 seam events** (7–18 per replica). Never replayed.
4. **Convex-edge grazing attachments** onto the adjacent face (normal mismatch, so `invalid` without a tie) remain an unregistered potential stop class. Rate unknown.
5. **The 3–6% excess** of observed tangency counts over the spawn-offset model.
6. **`pt100.direct`, latent.** If Mitsuba's `spawn_ray_to` returns an unnormalised direction (as in Mitsuba 3), then `sh.t` is parametric. The code `sr.maxt = dist − sh.t − de − 2e-4` mixes units, which over-extends the residual shadow segment. For this light and geometry the overshoot (≤ about 0.2 m past the light) meets no geometry, so no effect is expected here. Needs confirmation against the Mitsuba 3.8 source.
7. **Governance.** Whether the owner approved the agent's operationalisation of "same kind of miss recurs" as > 1e-6 of eligible (about 330 per replica).
8. **Executed experiment code identity.** The run manifest hashes the renderer sources but not the WL101E/F/G experiment code (`oracle_field.py`, `gpu101g.py`, …). The commit is a string supplied by the operator. The outputs' hashes and deterministic counts are consistent with the stated commits, but the code identity itself is not machine-recorded.

---

## 8. Architecture implications

- **No change to ECRO, the trained model, B_theta, LOW or the event contract follows from this audit.** WL101G stays at **D (evidence insufficient)**. Nothing about cross-view fidelity or spatial resolution was measured, and this audit measures nothing about it either.
- **Physical-oracle tables near tangent contacts.** The WL101C–G line compares tables against a **native** reference whose semantics contain a contact-scale defect (F6). Any piecewise-constant table spreads the defect over the 25 cm patch, with a weight that depends on the measure (uniform includes it, visibility-weighted excludes it, camera-projected samples it at camera density). A difference between such tables at the contact slot can therefore reflect measure choice applied to a defect, not representational fidelity. Before WL101F/G-style measure comparisons are interpreted at the contact, the owner must decide whether the native ring semantics are the target, or whether the support contract must exclude or flag native-degenerate regions. Redefining the target is outside this audit's authority.
- **G's camera-independence claim is supported** (reader independence was measured). **Its measure semantics near contacts are not.**

---

## 9. Reproducibility

**Inspected source identities (git blobs).**
- WL101G code at `2f9f575` (freeze), `0d6f60a` (A1), `7298e9b` (A2) and `c0daf39` (stop/worklog). `gpu101g.py` blobs differ by commit as expected (`ab9c9319…` → `371e4e3e…` → `7c444190…`).
- Inherited code unchanged across all four commits and HEAD:
  - `ecro_sss101e/oracle_field.py` `adb9a627…`
  - `ecro_material100/pt100.py` `c8b201da…`
  - `ecro86/pt86.py` `912e68ac…`
  - `ecro_sss101f/query_readout.py` `38e7899e…`
  - `ecro_sss101f/projections.py` `aa974195…`
- WL101E first-miss history: commit `fc5fa68` and worklog 101E, line 21.

**Protocol identities.**

| File | SHA-256 |
|---|---|
| `generalization.json` | `bdd77b70…a5c6` |
| `amendment1.json` | `0e0d0af3…b962` |
| `amendment2.json` | `19d5f547…bd1` |

**Artifacts.**
- All files indexed in [evaluation/101G/manifest101g.json](../../../evaluation/101G/manifest101g.json): 52/52 present and hash-matched.
- Construction packets used: `primary_a2/construction_{0,1,2}.npz` = `a431ef7b…`, `1baa2490…`, `8c6ea054…` (equal to progress.json).
- WL101E `recovery2/oracle_0.npz` = `1c71d1f2…`.
- Seeds: replicas 10203101–10203104 and camera-independence seed 10203001, as in the protocol.

**Focused reproductions.** These ran on CPU with numpy 2.3.5, read only, and wrote nothing outside [repro/](repro/). Hashes are in [repro/SHA256SUMS.txt](repro/SHA256SUMS.txt).

| Script | What it checks |
|---|---|
| `r1_edge_geometry.py` | Reconstruction of the failing Box B samples; metric tie bands |
| `r2_tangency_model.py` | Spawn-offset model of the sphere/floor contact |
| `r3_contact_ring_subcells.py` | Per-subcell visible/non-visible split, floor patch 155 and neighbours |
| `r4_contact_ring_bins.py` | Per-bin non-visible fraction and brightness ratio |

Run them from the repository root, for example `python docs/audits/wl101g_forensic_audit/repro/r3_contact_ring_subcells.py`.

**Not run.** No GPU replay, no rendering, no evaluation stream, and no full regression.
