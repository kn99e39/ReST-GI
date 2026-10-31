# Clean Supervision v2 — fixed top-up, conditional A/B, independent evaluation

User direction dated 2026-10-31 continues the WL65 partial clean-supervision hypothesis after Phase A stopped on precision (STOP_INSUFFICIENT_PRECISION: 107/576 deformation-loss-scaled cells, all in fields 84–87). It is a prospectively declared **extension** of the completed Phase A, recorded append-only in Worklog65. `clean_v1.json`, the 832 Phase-A packets, labels, analysis and the STOP verdict are unchanged. No new architecture hypothesis, no repeated top-up, no successor.

**Central question.** Under the unchanged fixed K representation, training objective and population, does the prospectively qualified partial clean-supervision intervention improve independent physical directional fidelity?

## Gate 1 — fixed reference top-up

Fields 84, 85, 86, 87 × rest + 12 original training dents = 52 actions. Each action gets 384M additional photons as 12 × 32M packets, 6 per original replica group (balanced). Total 624 packets, 19.968B photons. Seeds `660000000 + pose_index*100000 + field*100 + replica*10 + k` (k=0..5), disjoint from Phase A, WL64 and WL59 streams. The estimator is Phase A's, unchanged: WL59 first-exit action with the fast walker, observed through `moments65.observed_action`; one field per call with its own proposal; identical frozen source hashes and physical packet checks. Before any new photon, all 832 Phase-A packets and 26 bank files are rehashed against their receipts and the full v1 identity manifest is recomputed and must equal Phase A's.

Combination: packet-count weights `w_i = N_i / ΣN`; mean `Σ w_i m_i`, variance of the mean `Σ w_i² v_i`, where `v_i` comes from aggregated first/second moments (exclusive bins aggregated before variance). Replica group r = Phase-A replica r + top-up replica r (8 packets, 256M per group). Delta variance adds current and rest variance; the shared rest covariance cancels between dents. Unchanged fields must reproduce the Phase-A bank bitwise and every Phase-A analysis row to 1e-9 relative.

Prospective projection (not a decision variable): assuming SE halves and the weak-delta norm shrinks with it, the worst cell (tr03 field 85 B) is expected at ≈0.249 against the 0.25 gate. The budget is the user's fixed choice; a realised failure is a STOP, not a reason to add photons.

## Gate 2 — precision qualification

All original Phase-A gates on the complete 13×16 combined bank, unchanged: coarse absolute ≤1%, fine16 absolute ≤5%, coarse directional SNR ≥3 and SE/contrast ≤10%, every delta SE/√(‖δ‖²+1e-4‖rest‖²) ≤0.25, rest pooled contrast SE ≤0.5 × the original single-label noise per RGB. Validity additionally requires identity/physical checks and a gross estimator-consistency check between the independent Phase-A and top-up means of the 52 actions (median normalised squared difference in [0.67, 1.5]). Statuses as v1. Any precision failure ⇒ STOP (outcome C), no training, no further photons.

## Gate 3 — controlled A/B (only if READY)

The frozen WL59 `train.py`/`run.sh` are executed unmodified (resource policy 0) in two isolated roots. A: original r0 for all 96 fields. B: for rest + 12 training dents, `train[80:96]` replaced by the combined bank; `train[0:80]`, validation tensors, validation geometries and every evaluation input byte-identical. Seeds 5901/5902, same initialisation, same field schedule (`default_rng(574001)`), geometry cycle, AdamW/cosine/clip, action + 0.25 deformation loss, 6,000 steps, final checkpoint. Order alternated across seeds; no historical checkpoint reused. Before training, a new uninspected population U65 (`light_fields(rest,64,6508103,'test')[48:64]` on pt0, 2.048B tagged photons, all 24 total-contrast cells must reach SNR ≥3 for full confirmation) is generated and qualified; it never feeds training, labels or selection.

Controls (failure ⇒ outcome D): label isolation reloaded from disk; final numpy RNG state identical across arms and equal to an independent reconstruction; identical optimizer hyperparameters; Arm A reproduces the preserved WL59 evaluation within 2e-3 absolute on every directional, static and dynamic metric with identical eligibility flags and pt0 failure count; physics/F-S checks pass.

## Gate 4 — evaluation and adjudication

Unchanged WL59 `evaluate.py` and `cost.py` per arm; trained rigid/no-op checks; Pair7 (WL63 tags) and the WL64 population labelled historically inspected; U65 uninspected. Primary: pt0 F-policy 8 pairs × RGB, cells detectable in both arms. Improvement per seed: B/A mean ≤0.90 and no more qualified failures. Outcome A requires primary **and** U65 improvement in both seeds; otherwise B (sub-label "regression" if B/A ≥1.10 on primary or U65 in any seed). Complete K remains every original WL59 gate for both seeds and policies; a directional gain alone is not complete K and does not license coupled GI.

The intervention is partial: 16 of 96 fields on 13 training geometries; 80 fields keep their original noise. A negative result does not falsify full-clean supervision; a positive result does not establish representation adequacy or physical correctness beyond the evaluated populations.

Protocol: `experiments/fnto_clean65/protocol/topup_ab_v2.json`; code `topup65.py`, `qualify65.py`, `fresh65.py`, `arms65.py`, `compare65.py`, `invariants65.py`, `export65.py`, `chain65.py`; focused tests `test65b.py`; non-evidence smoke harness `smoke65.py`.
