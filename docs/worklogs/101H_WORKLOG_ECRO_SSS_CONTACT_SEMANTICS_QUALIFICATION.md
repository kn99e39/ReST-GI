# Worklog 101H — Contact estimator semantics recovery and construction qualification (2026-11-18)

## Scope

WL101H follows the independent forensic audit of WL101G ([audit](../audits/wl101g_forensic_audit/WL101G_FORENSIC_AUDIT.md)). It is a suffix batch: it closes WL101G's estimator-semantics evidence gap and adds no learned mechanism. WL101G stays stopped with verdict D. Amendment 3 is not introduced, and neither WL101G's primary nor a replacement projection primary is run.

The batch asks whether a physically and numerically qualified estimator/reference contract can be established for a new camera-independent projection comparison, or why such a comparison remains invalid.

**Unchanged:** ECRO, its checkpoint, B_theta, LOW/HIGH, the event contract, the scene, the native tracer, the native reference, the audit, and every historical worklog, protocol, log, seed and result. **New:** an isolated candidate-semantics path, its qualification instruments, and this record. The batch name follows the AGENTS.md suffix rule; the owner may rename it.

Three conditions are kept separate throughout:

- **N:** the historical native estimator and native reference.
- **C:** the candidate estimator with its own matched reference.
- **X:** independent analytic and geometric checks.

A C reference is never compared with an N reference as if they were the same quantity.

## Stage A — per-sample GPU reproduction of F6 (read-only)

`f6_replay.py` re-ran the exact WL101G replica-0 proposal launches (seed 10203101) and kept every sample on floor patch 155, the patch that contains the tangency. The re-trace reproduces the stored WL101G packet: counts are identical and luminance sums agree to 2.7e-8. Each sample records:

- the requested point and normal;
- native NEE, continuation and visibility origins, with a float64 inside-sphere test;
- the NEE first-hit sense, `de` and the through term (a line-by-line replica of `pt100.direct`, equal in branch decisions and within 2.2e-6 of the original through value);
- E2 and E3+, and the visibility receiver.

| Floor samples of patch 155 | Ring, r < r*(p) (1,745) | Rest of subcell 3 (63,718) |
|---|---|---|
| Native origins inside the sphere (NEE, continuation, visibility) | 100% | 0% |
| NEE first sphere hit is an exit (`dot(n, wl) > 0`) | 100% | 0% |
| `de` (median) | 0 | 0.590 m |
| Through term / (f I / d²) | exactly 1.0 (physical exp(−σ·chord) = 4.57e-4) | — |
| E3+ = 0 | 100% | 4.9% |
| Visibility: sphere seen from inside (`back_receiver`) | 100% | 0% |
| Mean outgoing E2–E5 luminance | 1.485 | 0.314 |

Here r*(p) = √(2R·m(p)), with the native offset m(p) = (1 + max|p|)·1500·2⁻²⁴; its median is 1.158 cm. Outside the ring, the native `de` agrees with an independent float64 chord to 1.8e-4 m, so shadow attenuation is correct wherever the origin is valid.

A designed radial grid (16 radii × 8 azimuths × 16 directions × 256 repeats) shows a step between 11.5 mm and 12 mm: L falls from 1.485 to 0.192. Native attenuation is 1.0 inside the step and agrees with the analytic value within 0.2% outside it.

**F6 is confirmed per sample, exactly as the audit inferred.** One audit open item is also closed: in this Mitsuba build `spawn_ray_to` returns a unit direction (|d| = 1 on every replayed ray), so `pt100.direct`'s residual `maxt` is metric. Audit Q6 does not apply here.

*Execution note.* The first replay attempt mixed a single-shape attachment scene with full-scene tracing in one Dr.Jit kernel. OptiX disabled a ray operation and PTX compilation failed. That output is kept only in `results/ecro_sss101h/discarded/` and was not used. The fix materialises the attachment first. Every WL101H job now fails (exit 99) if its log contains "Disabling this ray tracing operation" or "COMPILE ERROR".

## Candidate semantics and its derivations (frozen before stages D and F)

**Contact-valid origins** (`semantics101h.py`, installed on one runtime instance; the originals are untouched). For register entry 1, the sphere/floor contact, the candidate computes the analytic free distance g along the native offset side:

- If the native offset m < g/2, the native ray is kept bit for bit.
- Otherwise the origin moves to the gap midpoint.
- The SSS exit step becomes min(1e-4, g/2) toward the floor.
- If g/2 < h_min, no float32-representable free-space origin exists. The event is singular and contributes nothing. In construction, singular samples leave the support of every measure and are counted per subcell.

Only three source lines change: the NEE spawn, the continuation spawn and the SSS exit. The event contract is unchanged.

- **h_min = 2.12e-7 m** comes from the Q1 stress test (`stress101h.py`). With the candidate at h_min = 0, the largest half-gap at which any check fails is 5.3e-8 m; the declared rule multiplies it by 4. The checks are: origin in free space, NEE entry sense, chord within 1e-3, continuation and visibility front hits, and SSS exit hitting the floor. Native semantics fail up to 6.3e-5 m (floor) and 7.0e-5 m (cap). The singular region shrinks from the native 1.17 cm ring to about 0.58 mm.
- **Membership band b_m = 3.8e-6 m** (16 ulp of 3 m), replacing WL101E's parametric 4·eps band, which is narrower than one ulp on Box B (audit F1). The Q1b edge stress (`stress_edges101h.py`, 396,160 points on all 1,238 planar patches) finds attachment misses only at insets ≤ 1e-7 m from true face edges. That is well inside b_m/2.

The [protocol](../../experiments/ecro_sss101h/protocol/qualification.json) fixes the following:

- **Contact register:** the audit's §3 inventory, checked against code and replay.
- **Semantic gates S1–S5:** support identity; origin validity; per-slot excluded measure; a per-slot bias bound of measure × measured radiance deviation (≤ 7.0e-4 absolute and ≤ 2.5e-4 relative); and instrument consistency, including reproduction of the WL101G packets.
- **Diagnostic outputs D1–D4:** mechanism labels, back-receiver attribution, unregistered-event packets, and offset sensitivity. These never stop a run.
- **Discrimination check:** the native semantics must fail S2.
- **Materiality thresholds** and the three F6 questions: the contact slot, the U/G comparison, and the 10–14 cm lobes.
- **Seeds:** stage D reuses WL101G's 10203101–4. The intended frozen seeds are 10204101–4, with 10204105 as a rate seed.
- **Image scope:** image evaluation uses V0 archived streams only.
- **Verdict mapping:** QUALIFIED, PARTIALLY QUALIFIED or NOT QUALIFIED.

*Development fixes before the freeze.* A reduced smoke run exposed three problems in my new code, all fixed:

1. A slot-indexing error in the per-slot membership sums.
2. A key-classification error in the integer columns of the independence check.
3. NEE-origin validity checked on samples where `pt100.direct` never uses the NEE ray: light behind the face, or the SSS sphere. This had produced about 2.5e5 spurious junction and inside-box labels.

After the fixes, the smoke run shows native ring 34, cap 18 and junction 1, and candidate ring 0, cap 0, junction 1 and inside-box 0, at 64 proposals per slot. No threshold depends on any stage D or F result.

## Frozen execution (commit `9089f09fe292ecda0a31e6b03b3a31ae1112c55e`)

The seven jobs ran in this order on LabServer63 with exit 0 and no OptiX warning:

1. candidate replay;
2. stage-D native construction and 3. stage-D candidate construction at WL101G seeds 10203101–4;
4. native and 5. candidate V0 camera passes (101100/101101, 4,096 spp);
6. native and 7. candidate stage-F preflight at intended seeds 10204101–4 plus rate seed 10204105.

The frozen analysis ran on the server, so the 3.1 GB mirror was not needed first.

### X checks — candidate transport against independent analytic and physical expectations

| Radius from the tangency | Native L | Candidate L | Candidate attenuation / exp(−σ·chord) |
|---|---|---|---|
| 0.3 mm (singular, r < 0.58 mm) | 1.485 | 0 (excluded) | — |
| 0.6–11.5 mm | 1.485 | 0.173–0.181 | 0.9991–1.0000 |
| 12 mm | 0.192 | 0.178 | 0.9990 |
| 20 / 40 / 80 mm | 0.188 / 0.199 / 0.255 | 0.182 / 0.197 / 0.254 | 0.9982 |

- **Attenuation.** Inside the old ring, the candidate's shadow attenuation equals the float64 chord law; native semantics overstate it 2,190-fold.
- **Continuity.** Candidate radiance is continuous across the old ring boundary.
- **Higher orders.** Native radiance just outside the ring is also inflated: E3+ is 0.179 against 0.165 at 12 mm, and the excess decays with radius. Paths that leave a floor point, scatter through the sphere and return to a ring vertex pick up its unattenuated NEE.
- **Rest of patch 155.** The audit did not predict this. On the WL101G replica-0 samples of patch 155, the non-ring mean changes 0.2618 → 0.2610, and 98.8% of non-ring samples are bit-identical.

### Stage D — effect of F6

**Reproduction.** Native stage-D construction reproduces the stored WL101G packets at slot level: n and v are identical and real sums agree to ≤ 8e-10 relative. Within slots, 102–119 samples per replica (of 3.3e8) fall on the other side of a subcell border at coordinate 0.5. This is float32 address rounding between two kernel compositions. The frozen S5 rule demanded per-subcell identity and therefore fails (see Stage F).

**Packet differences, candidate vs native.** 1,364 slots change by more than 1e-5 relative:
- 393 of them contain contact classes;
- 971 contain none and change only through higher-order paths, by at most 1.1%.

Counts are identical outside the contact.

**Tables.** Native vs candidate, four replicas. *Material* means larger than both 3 SE and 7.0e-4.

| Patch | U (native → candidate) | V | G | Material bins U / V / G |
|---|---|---|---|---|
| **155 (contact)** | 0.2704 → 0.2606 (−3.6%) | 0.2665 → 0.2650 | 0.2661 → 0.2646 | 64 / 54 / 56 of 64 |
| 154, 156, 139, 171 | ≤ 1e-4 change | ≤ 1e-4 | ≤ 1e-4 | 0 |

- **U − G at the contact patch changes sign:** +0.0043 under native semantics, −0.0040 under candidate semantics. The spatial subcell S_G(3) changes by −0.0055.
- **Comparison with the audit.** The audit expected +2.3–2.7% on U, from the ring alone, and V/G unaffected. Both are refined: higher-order contamination raises U by 3.6% and V/G by 0.6%.

**Images at V0.** Each condition is compared with its own reference.

*Reference shift, C − N, on luminance* (both streams agree to within 2e-5; the native near-floor reference is 0.2163):

| Region | Shift |
|---|---|
| near | −2.6e-4 |
| bands 0–10 cm / 10–14 cm / 14–18 cm | −5.6e-4 / −3.2e-4 / −1.7e-4 |
| flank | −2.3e-5 |
| sphere boundary | −4.5e-5 |
| far control (≥ 1 m from contact) | −1e-6 |

*Effect on the comparisons:*
- **U arm, C − N change of the residual.** near +6e-5, contrast +1.2e-4, 10–14 cm band +1.3e-4, silhouette −2.7e-4. None is material.
- **U − G.** near +9e-5, contrast +7e-5, contact/shadow +1e-5, sphere −6e-5, silhouette −2.5e-4. None is material.
- **Converged RMSE.** No region changes by more than 0.0006.

*Descriptive only: development view V0, unqualified contract, not a projection result.* The near residuals are U_E +0.0702, G +0.0415 and S_G +0.0106 under native semantics, and +0.0704, +0.0415 and +0.0105 under candidate semantics.

**Answers to the frozen questions.**

| Question | Answer |
|---|---|
| Contact floor slot | **Material** for U, V and G. |
| Existing U/G comparison | **Material at the table slot** (the sign flips), **not material at any V0 image endpoint**. The contact slot is hidden beneath the sphere from V0 and carries little query weight. |
| 10–14 cm lobes | **Not explained by F6.** The reference shift and the U-residual change are both below materiality; U_E's band excess stays at +0.078. |

### Stage F — construction-scale preflight and gates

| Gate | Native (discrimination run) | Candidate | Notes |
|---|---|---|---|
| S1 support identity | pass | pass | invalid 0 and support conflict 0 on all 5 seeds |
| S2 origin validity | **fail**, ring 1,760–1,915 per seed | pass, ring 0 and inside-box 0 | F6 caught by a semantic gate |
| S3 excluded measure | pass | **fail** | per-subcell singular fraction 1.0e-3–5.3e-3 = **one** singular sample in a subcell of ~1,000 (or 189) samples. The analytic singular measure is about 7e-5. |
| S4 bias bound | fail (108–162 slots) | **fail** (113–171 slots; max bound 7.7e-4–2.7e-3) | dominated by 2·max subcell deviation in bright, high-variance slots, and by single junction samples counted by frequency |
| S5 instrument | — | **fail** (reproduction only) | split consistency at all six cameras, LOW invariance, cos normalization and camera independence pass. Reproduction fails per-subcell identity although it is exact per slot (above). |
| D3 unregistered | 2 | 2 | Box B/floor junction lattice samples at z ≈ 5.6e-8 m (audit F3 class). My junction rule only caught origins *on or behind* a plane. |
| D4 offset sensitivity | 200 slots | 98 slots (80 on the left wall) | well-supported slots (median 1,791 visible samples). A few visibility flips per slot; median relative \|ΔG\|/G 7.9e-4; net −472 of 1.08e9 visible samples. Probably the open-face grazing band (register 4); edge identity not replayed. |

The discrimination check holds: S2 flags native F6 while native S1 passes.

**Frozen verdict: NOT QUALIFIED.** S3, S4 and S5 fail and D3 is non-empty.

## Final report

**Implementation fact.**
- An isolated candidate path, `semantics101h.py`, changes three source lines of the copied `pt100` trace/direct for register entry 1; the originals are untouched.
- Mechanism-class construction (`construct101h.py`) adds a metric membership band, origin validity for the rays actually traced, class-split sums and offset sensitivity. Runners record executed-code hashes.
- Stress tests Q1 and Q1b fixed h_min and b_m. A per-sample replay instrument and a frozen analysis complete the path.
- Nothing in ECRO, B_theta, LOW, the event contract or the historical native reference changed.

**Measurement.** Every number above.

**Observation.**
- **The audit was right on F6.** The mechanism, location and table signature are confirmed sample by sample.
- **The audit's magnitude is refined.** F6 also contaminates higher-order radiance near the contact, so the slot effect is larger than estimated and V/G are not immune.
- **Audit Q6 does not apply.** spawn_ray_to returns unit directions.
- **The audit's F3 mechanism is confirmed by recurrence.**
- **Audit F5's 3–6% count excess is unresolved.** Native cap counts were 1,255–1,325 per replica.

**Interpretation.**
- **The candidate semantics are internally consistent and physically better at the contact.** Estimator and reference share one tracer, the split query matches the full trace at all six cameras, attenuation and continuity agree with the independent checks, and every non-contact table is unchanged.
- **The qualification contract I implemented is not fit for purpose.** Its excluded-measure and bias gates use single-sample frequencies or very conservative deviation bounds, so they fail on harmless events. The junction classifier misses the above-plane float band. The reproduction rule ignores float subcell flips. Its relative D4 threshold flags a 0.08% native visibility deviation.
- **It does discriminate:** it catches F6 in native semantics. But it cannot yet separate harmless events from semantic contamination, which is a declared stop condition. A new camera-independent projection comparison therefore remains invalid until a corrected gate set is designed, frozen and requalified at full scale.

**Unresolved questions.**
1. A v2 gate design:
   - analytic singular and band measure with statistical bounds instead of single-sample frequencies;
   - junction classes by the requested point's metric distance to the junction;
   - slot-level reproduction with bounded subcell flips;
   - a D4 threshold tied to image-level materiality, with an owner decision on the native grazing band.
2. Ray-level identity of the D4 flips.
3. Audit F5's count excess.
4. Whether to adopt contact-valid semantics as a regime for future contact comparisons. This is an owner decision; the candidate is not promoted.

**Architecture verdict.**
- No architecture change follows. WL101G stays D.
- The candidate is **NOT QUALIFIED** for a new projection comparison, by the frozen rule and with the precise reasons above.
- F6 matters for any interpretation at the contact table slot. It does not explain the V0 contact excess or the 10–14 cm lobes.

**Reproducibility.**
- Commits: protocol/code freeze `9089f09`; this report is the next commit.
- Environment: LabServer63 RTX 3080 Ti, OptiX 595.71.05, Mitsuba 3.8.0, Dr.Jit 1.3.1, Python 3.10.21, `~/NeuralShaderDemo/external/8dna26/.venv`, launched via `run101h.sh` / `chain101h.sh`.
- Seeds:

  | Use | Seeds |
  |---|---|
  | Replay | 10203101 (WL101G), 10204001 |
  | Stage D | 10203101–4 |
  | Stage F | 10204101–5 |
  | Independence | 10204001 |
  | Camera | 101100 / 101101 |
  | Smoke | 10204300+ |

- Analysis: `python experiments/ecro_sss101h/analyze101h.py RESULTS WL101G_PRIMARY_A2 WL101_REFERENCES OUT`, with PYTHONPATH including `experiments/ecro86`.
- Raw results are under `results/ecro_sss101h/`. Reviewer artifacts and hashes are in `evaluation/101H/`.
