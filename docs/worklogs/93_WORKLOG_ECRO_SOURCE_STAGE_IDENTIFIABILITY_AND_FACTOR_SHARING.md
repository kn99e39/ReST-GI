# Worklog — ECRO Source-Stage Identifiability and Local Factor Sharing (WL93) (2026-11-13)

## Origin and scope

- **Brief.** "ReST-GI — WL93 ECRO Source-Stage Identifiability and Local Factor Sharing" (user, 2026-11-13).
- **Batch class.** Regular numbered batch under the AGENTS.md numbering policy (2026-11-13). It introduces a new mechanism (source-stage supervision and untied source / receiver local factors) and a new architecture decision question. It is not a WL92B evidence-gap suffix.
- **Question.** Can ECRO achieve reproducible sharp interreflection fidelity with a physically shared local response operator once its source-stage response is explicitly supervised? Or does it need source / receiver factor separation beyond supervision and capacity effects?
- **Hypotheses.**
  - H_IDENTIFIABILITY: stronger source-stage supervision improves the learned local response and sharp interreflection without untying.
  - H_SHARING: tied factors prevent independently accurate source and receiver responses even with identifiable source-stage supervision.
  - H_INTERACTION: neither alone suffices; their combination gives a reproducible improvement.
- **Fixed.**
  - ECRO global branch: frozen. The global prediction is identical across arms for the same seed.
  - Source encoding, ranks / widths, ray budget, ER materials and WL86 thresholds are unchanged.
  - No explicit GI, no source-class handling, no adoption, no follow-up batch.
- **Where.** Branch `Selective_Recompute`, LabServer63 (main). Code `experiments/ecro_stage93/`; results `results/ecro_stage93/`, `results/evaluation/93/`. Server root `~/rest_x93`, reading `~/rest_ecro86`, `~/rest_x92`, `~/rest_x92a`, `~/rest_x92b`.

## WL92B rule reconciliation (required before WL93 evidence; WL92B files unchanged)

**Written rule.** WL92B's frozen protocol (`attribution_rules.H_interact`) reads: "no dominant counterfactual, but at least two with f ≥ 0.25 for both OWN seeds in a case".

**Implemented rule.** `analyze92b.py` (line 110) implements the "no dominant" clause as `not (H_local SUPPORTED or H_global SUPPORTED)`. That checks only X_M12 and X_L3 for dominance.

**Under the written rule.**
- On region C of both E2 cases, X_L2 (single-flight term) and X_Q (exact once-scattered table) are DOMINANT (f ≥ 0.5 for both OWN seeds).
- H_interact is therefore INCONCLUSIVE, not SUPPORTED.
- With H_interact not SUPPORTED, only H_geometry is SUPPORTED. H_geometry is not a dominance hypothesis, so the frozen outcome mapping has no matching clause. The code falls through to **D**; the official record is **C**.

**Re-derivation.** `prep93.py recon92b` re-derives this from `~/rest_x92b/results/final92b.json` into `results/ecro_stage93/recon92b.json`. The dev chain re-derived: official H_interact SUPPORTED / outcome C; written rule INCONCLUSIVE / D (fall-through).

**Materiality for WL93.** Not material:
- No measurement, counterfactual value, E5 admission or E5 criterion changes.
- The mechanism finding that WL93 builds on is unaffected: the OWN error sits in the single-flight term (X_L2 dominant) and within it in the source-side once-scattered stage (X_Q dominant).
- WL93 thresholds and criteria do not depend on WL92B's outcome label.

**Interpretation going forward.** WL92B's "outcome C" should be read as "the single-flight / source-stage counterfactuals dominate, with co-adapted tied factors and a geometry-dependent panel operator" (outcome D under its written mapping). WL92B's worklog, protocol and results are preserved unchanged.

## Design (`experiments/ecro_stage93/protocol/ecro_stage93.json`)

**Architecture (model93.py).**

The table is B = S_θ(s) + G_frozen(s).

- **G_frozen.** The global term (3–5 events) of WL92A `own_s<seed>`, evaluated with that model's own frozen factors (c, a, P, Q). It is identical for every arm of a seed (test T_GLOBAL_IDENTITY) and is never trained.
- **S_θ.** The single-flight term a_r · (c_r · flight(a_s · (c_s · s))) from a trainable copy of the OWN generator (trunk and local heads). The copy's global heads are frozen and unused.
- **Disclosed change in every arm.** The single-flight local factors are decoupled from the global term's factors. This is needed to hold the global prediction identical across arms. "Tied" refers to source / receiver tying inside the single-flight term.
- **Modes.**
  - **tied:** c_s = c_r, a_s = a_r (the WL86 / WL92A operator).
  - **untied:** the source uses (head_c, head_a); the receiver uses duplicated heads (head_c2, head_a2); the trunk is shared.
  - **tiedcm:** capacity-matched tied control. One shared (c, a) = softplus(head(tb) + head2(tb)), where head2 has the same architecture as the local head. head2's hidden layers are freshly initialised from `torch.Generator` (seed = run seed + 9300) with the nn.Linear default bound; its output layer is zero. The result is a 2×-wide tied head with exactly the untied arm's trainable parameter count.
- **Pre-freeze correction (dev chain).**
  - The first tiedcm design averaged the pre-activations of two copy-initialised heads.
  - That design is symmetric: both copies receive identical Adam updates, so the pair stays identical forever. In the dev chain it reproduced the tied arm bit-for-bit (validation errors of C0 / CQ equal to T0 / TQ to every printed digit, with the same E_H), making it a degenerate control.
  - It was replaced before any evidence run.
  - The extra parameters of tiedcm (fresh hidden layers, zero output) are necessarily initialised differently from untied's copies. This is disclosed.
  - A post-training test now requires tiedcm and untied to differ from tied.
- **Dev chain bug fix.** The first dev training had zero trainable parameters: the OWN base was frozen before being deep-copied. The copy is now made trainable explicitly. The tests require n_trainable > 0, untied = tiedcm > tied, and a frozen base and frozen global heads.
- **Initialisation.** Every arm equals WL92A OWN at step 0 (test T_INIT_EQUIV). For untied the duplicated heads are copies; for tiedcm the added head outputs zero.

**Arms (2 seeds each, 8601 / 8602, initialised from the same-seed WL92A OWN).**

| Arm | Factors | Objective |
|---|---|---|
| T0 | tied | OWN (WL92A objective) |
| TQ | tied | OWN + source-stage |
| U0 | untied | OWN |
| UQ | untied | OWN + source-stage |
| C0 | tiedcm (capacity control for U0) | OWN |
| CQ | tiedcm (capacity control for UQ) | OWN + source-stage |

**Objective.**
- OWN = mean(((S + G − t)/den)²) + mean(((S − t2)/den)²) + mean(((G − (t − t2))/den)²), with den = t.mean(bins) + 0.05 · t.mean(all). The last term is constant because G is frozen.
- Source stage: + w · mean(((Bq − q)/den_q)²), w = 1 (fixed). Bq = a_s · (c_s · s) is the single-flight term's source-side once-scattered table.
- q is the offline physical once-scattered table: `quad87.qtable` (P 16, Mq 1,024), seed 93000 + 4·idx + lighting, same geometry and light. It is never used at runtime.

**Training.**
- Population: the WL92A population (WL86 train / val minus the WL92A E4 exclusion; 928 / 55 geometries × 4 lights).
- 8,000 steps; fresh Adam; lr 3e-4 → 1e-5 cosine after a 200-step warmup; clip 10; one geometry (4 lightings) per step.
- Data order from `default_rng(seed + 93)`: identical for every arm of a seed.
- No extension and no seed replacement.

**Identifiability preflight (STOP_TARGET_NOT_IDENTIFIABLE).**
- Computed on the 55 validation geometries and on E2 T0 / T1.
- The irreducible discrepancy of the exact frozen-ER operator applied through the deployed 16-sample encoding (panel-patch relative L1 of `enc_q(s16)` vs q) must be ≤ 0.5 × WL92A OWN's learned discrepancy (Bq vs q on panel patches), for both seeds.
- Rationale: if the deployed encoding cannot carry the target, supervising toward it cannot be distinguished from fitting noise.

**Held-out E6 (qualified before any training).**
- **Candidates.** Ordered from `default_rng([93550, k])`, using WL92B's E5 generation rule.
- **Lights.** WL86 test lights L1 / L2 by parity.
- **Disjointness.** Every box centre is ≥ 0.15 m from all 1,120 WL86 data files.
- **Geometry rule.** A, C, D, E and C_footprint ≥ 1,000 px.
- **References.** REF-ER / REF-P 8 × 8,192 spp (seeds 93710 / 93730 + r + 100k) and the diag92 truth split, for the first three geometry-admissible candidates in order.
- **Admission.** WL86 stage-0 Q1–Q4 (Q4 at SNR 4.0), Q6, Q7, plus the sharp family: the light → panel → receiver 2-event share of the region-C higher-order reading ≥ 0.08 with SNR ≥ 4.
- **No admission.** If no candidate is admitted: STOPPED_WORKLOAD_UNQUALIFIED, no training, no held-out claim.
- **E5.** WL92B candidate 995 is re-evaluated as an inspected (not independent) workload.

**Evaluation.**
- Tags: E2 T0 / T1, E1 T1 LSPOT, E5k995 T0 / T1 / LMIX / LSPOT, E6 T0 / T1 / LMIX / LSPOT.
- Models: 12 arm models, plus WL92A own / sum for context and reproduction.
- Renders: 2 × 4,096 spp (seeds 86600 / 86601, common random numbers).
- Controls on E6: an ORACLE table and the explicit X_EXP.

**Criteria (fixed now).**
- **ADV(X, Y).** On E6 region C at T1 and at T0, for both seeds: E_H(X) ≤ E_H(Y) − max(noise_X, noise_Y, 0.005).
- **STAGE(X, Y).** Panel-patch Bq-vs-q relative L1 on E6 T1 and E2 T1: X ≤ 0.9 × Y, for both seeds.
- **NOREG(X).** Against the same-seed T0:
  - no WL86 gate passed by T0 is lost (F1; F2 in A and D; F4);
  - stationary and D E_H ≤ 1.25 × T0 + 0.002 on E2 / E5 / E6 T0 and T1;
  - the same bound on stationary E_H for every source configuration.
- **DYN(X).** F3 (region D) is not lost against T0 on E2, E5 and E6.
- **PASS_sup(X, Y)** = ADV ∧ STAGE ∧ NOREG ∧ DYN. **PASS_share(X, Y)** = ADV ∧ NOREG ∧ DYN.

**Labels.**
- INTERACTION_SUPPORTED: PASS_sup(UQ, T0) ∧ PASS_share(UQ, CQ) ∧ ¬PASS_sup(TQ, T0) ∧ ¬PASS_share(U0, C0).
- SHARING_SUPPORTED: PASS_share(UQ, CQ) ∧ PASS_share(UQ, TQ), unless INTERACTION applies.
- IDENTIFIABILITY_SUPPORTED: PASS_sup(TQ, T0) ∧ ¬PASS_share(UQ, CQ).
- NOT_SUPPORTED: none of PASS_sup(TQ, T0), PASS_sup(UQ, U0), PASS_share(U0, C0), PASS_share(UQ, CQ), PASS_sup(UQ, T0) holds.
- INCONCLUSIVE: any other combination.
- STOPPED-INCONCLUSIVE: a test, control or reproduction fails, the target is not identifiable, or E6 is not admitted.
- Labels do not authorise adoption.

**Tests.**
- T_INIT_EQUIV: every arm at step 0 = WL92A own on E2 T1 (≤ 1e-5).
- T_GLOBAL_IDENTITY: the global term is identical.
- T_TIED: tensor identity for tied / tiedcm; for untied, a source-only gradient leaves the receiver heads at zero gradient.
- T_LINEAR: zero, scaling, superposition and non-negativity.
- T_QTARGET: the stored E2 T1 q is reproduced.
- T_DET.
- T_REPRO: WL92A own re-rendered on E2 T0 / T1 equals WL92A's own renders in region-C E_H (≤ 1e-9).

**Chain (run93.sh, one detached setsid chain).**
1. Inputs and the WL92B reconciliation.
2. Validation q, then the identifiability gate.
3. Training q in parallel with E6 generation.
4. E6 references and admission, in candidate order.
5. E6 source references and oracle fields.
6. Pre-training tests.
7. 12 trainings.
8. Post-training tests.
9. Tables, local matrices, timing.
10. Renders.
11. Analysis, visuals.

## Dev chain and full-scale probes (before the freeze; not evidence)

**Dev chain** (`~/rest_x93dev`, `--dev`: Mq 64, 48 training geometries, 200 steps, 64 spp, E6 forced to the first geometry-admissible candidate).
- Every stage ran end to end: reconciliation, identifiability gate, q generation, E6 generation / references / admission, tests, training, tables, renders, analysis, visuals.
- Two defects were found and fixed before any evidence run:
  - **Zero trainable parameters.** The base was frozen before being copied.
  - **Degenerate capacity control.** The averaged copy pair gave C0 ≡ T0 bit-for-bit (see the design section).
- After the fixes, the tests pass:
  - init equivalence 1.7e-7;
  - global identity exact;
  - untied receiver-head gradient 0 under a source-only loss;
  - n_trainable: tied 405,424, untied = tiedcm 487,952;
  - trained C0 vs T0 single-term difference 2.0e-3 / 2.2e-3, U0 vs T0 0.05 / 0.15 (non-degenerate).

**Full-scale probes** (`~/rest_x93probe`, RTX 3080 Ti 12 GB, 125 GB RAM, 32 cores).

| Stage | Measurement | Chosen setting |
|---|---|---|
| q generation (Mq 1,024) | ≈ 0.35 s per geometry × 4 lights; 6 concurrent shards: GPU 31 %, 4.4 GB (CPU / IO-bound, short) | 8 concurrent shards of 128 (≈ 3 min total) |
| training (300-step probe, full step cost) | 1 process: 46 ms/step, GPU 33 %, 1.2 GB; 4 processes: GPU 82 %, 5.6 GB, 8.4 s per run-equivalent; 6 processes: GPU 89 %, 8.7 GB, 8.7 s (no throughput gain) | 4 concurrent (≈ 15 min per 8,000-step run; 12 runs ≈ 45–50 min); RSS 2.2 GB with 93 items, so ≈ 13 GB per process at full population (52 GB for 4, within 121 GB available) |
| E6 references / split / sources / oracle field | identical code and concurrency as WL92B (pool 6: 32 REF renders at 8,192 spp ≈ 19.5 min per candidate; split ≈ 30 s; sources and oracle fields ≈ 5 min) | pool 6, up to three candidates |
| renders | identical path to WL92B (pool 5); single render 8 s at 4,096 spp | pool 5 (≈ 360 renders) |
| T_REPRO | WL92A own_s8601 E2 T1 table via the WL93 path, rendered at 4,096 spp seed 86600, vs WL92A's render: max \|hl\| difference 6.0e-8 (float32 ULP), 1,893 region-C pixels differ, region-C single-half E_H difference 4.2e-9 | tolerance set to 1e-6 (the pre-written 1e-9 would fail on ULP-level table differences; 1e-6 is three orders below the 0.005 decision margin) |

