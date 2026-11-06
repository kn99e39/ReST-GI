# Rank-16 Learned ATO Transfer Falsification — design record (WL75, 2026-11-06)

This is an unadopted bounded-experiment design. The machine-readable contract is
`experiments/fnto_rank75/protocol/rank75_v1.json`. The chronological record is `docs/worklogs/75_WORKLOG_RANK16_LEARNED_ATO_TRANSFER.md`.

## Question

WL74 showed that a globally shared rank-16 A/O basis (G2), used as a labelled source-specific oracle, roughly halves the boundary contrast error. It also met the WL59 gates on every development group. No learned K was trained with it.

WL75 asks one thing. Does the learned W+UV spatial operator, trained exactly like the WL73 AR controls but with mixture rank 16 and the frozen G2 basis, get more accurate at independent source positions? And is the measured cost of doing so acceptable?

## The single intervention

- **Rank 8 → 16.** The global mixture rank changes from 8 to 16, using the G2_r0 basis. That basis was selected by the WL74 fitting objective alone (lowest J_final among the three G2 restarts). It was not selected by any held-out or development score.
- **Rank-dependent output channels.** These are the only shape changes:
  - `edge[-1]`: M·C outputs.
  - `node_head[-1]`: U, V and mass for M mixtures.
  - The in/out logits.
- **Capacity.** The change raises learned parameters from 155,104 to 273,040 and angular logits from 1,920 to 3,840. It also doubles the resident W/U/V state.
- **Everything else is unchanged:**
  - geometry features and trunk;
  - node rank 16;
  - bins, ballistic B, d = 1 − β, and first-exit semantics;
  - the passivity normalisation;
  - S16-R data and its replica split;
  - objective J + 1.0·L_sym;
  - Adam with cosine 3e-3 → 3e-5;
  - 20,000 steps, clip 1.0, checkpoints;
  - the final iterate is the result.
- **What this does not separate.** The comparison tests the complete rank-16 design. It does not separate angular-basis quality from the added spatial channels.

## Initialisation (predeclared)

- The trunk is P_590x bitwise.
- Each rank-dependent final-layer row for mixture r copies P_590x's row for mixture r mod 8. This is the same duplication from which G2_r0 was started in WL74.
- A/O are G2_r0, frozen.
- No channel is initialised randomly.

## Stage 0 correctness

- **Rank-generic class at M = 8.** It must equal the historical runtime class in columns, coefficients, apply, column sums and resident rebuild, and reproduce the historical training objective.
- **Synthetic rank-16 embedding.** Mixture r ← r mod 8 with repeated logits must reproduce the rank-8 physical response. This is a correctness test only.
- **Physics checks at rank 16.** Positivity, zero input, linearity, passivity and cache identity.
- **Provenance and isolation.** Basis provenance and the selection rule are checked, as is Stage 3 isolation.
- **Decision logic.** Unit-tested on synthetic records.

## Stage 2 gate (before any independent reference exists)

The data are the S16-R training sources, scored on held-out replicas 4–7. The metric is the worst-channel upper 95 % bound. Each rank-16 seed must meet all three conditions:

1. **Capacity:** con_coarse ≤ 0.8 × the better AR seed, on both TRAIN_NEW groups. AR sits at the fixed-basis oracle floor (0.270) there, so meeting this requires going below the rank-8 G0 floor.
2. **No regression:** absolute errors (native / fine16 / coarse), and interior contrast on TRAIN_KEEP, ≤ 1.25 × the worse AR seed.
3. **Validity:** passivity holds everywhere, and A/O stay bitwise equal to G2.

Two further conditions apply:
- The recomputed AR and oracle values must reproduce the archived WL73/WL74 values within 1e-3.
- If the gate fails, the outcome is INSUFFICIENT FIT and the batch stops before reference generation.

## Stage 3 independent confirmation

- **Population.** The frozen WL74 Stage 4 population:
  - 10 static sources in TBE / TBC / SM / SR (near and far) and INT;
  - 6 footprint sources at rest and cf1;
  - 1,408 packets.

  WL74 never generated references for it. The WL74 dev smoke made 0.5M-photon dev packets that no one read; WL75 does not use them.
- **Reference method.** WL74's own probe code is used unchanged: WL57 walker, gates, 32M photons, K = 4, predeclared seeds.
- **Primary comparison.** BND4_FAR con_coarse. The result is YES only if both rank-16 seeds are ≤ 0.8 × the better AR seed and also below the AR seeds' lower 95 % bound. It is NO if both are above 0.8 × the worse AR seed. Anything else is OVERLAP.
- **Regression checks.** Three-valued ×1.25 checks on absolute error on BND4_FAR/NEAR, INT4 and FP4_cf1, and on INT4 contrast.
- **Reported separately:**
  - the WL59 absolute gates;
  - DEF4 (cf1 is a near-training-pose check);
  - the native and fine16 metrics.

## Stage 4 / 5

- **Stage 4 (descriptive only):**
  - learned versus source-specific oracle gap;
  - coarse-cell sub-position;
  - label-free orientation invariants under the 15 exact slab symmetries;
  - error versus distance to supervised sources;
  - checkpoint drift.
- **Stage 5:** the measured rank-16 / rank-8 ratios of rebuild, apply, state, memory and training time, plus the G2 pretraining time. "Cost dominates" is attached only to a transferable benefit with a ratio above 2.

## Limits known in advance

- Rank 16 adds both angular and spatial capacity.
- The global sharing of A/O stays in place.
- The frozen population has no previously unrepresented (SV) source; that stratum exists only in the WL73 development group.
- cf1 lies 4.4 mm from the pt0 training dent.
- Oracle T is per-source latent data, not a transferable model.
