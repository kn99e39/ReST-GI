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
