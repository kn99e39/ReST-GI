# Worklog — Spatial-Map Coverage and Cross-Source Generalization, WL68 Follow-up (2026-11-02)

## Authorisation and scope

On 2026-11-02 the user issued a pasted brief: one bounded follow-up to WL68, on the cross-source and deformation generalization of the existing geometry-conditioned spatial map of the WL59 P intrinsic-K family.

**Central question.** Does the unchanged WL59 geometry-conditioned spatial map generalize physically valid collided transport when source coverage is controlled? Or does the failure persist in a way that justifies reconsidering the mapping architecture?

**The batch has four stages:**

1. Spatial-map identifiability diagnosis.
2. A cross-source synthetic teacher control.
3. A controlled physical coverage experiment with these arms:
   - A: frozen WL59;
   - B: WL68 fitted;
   - C: sparse coverage;
   - D: expanded coverage;
   - E: coefficient oracle.
4. An independent physical evaluation on previously unscored ports and replicas.

**The batch may change only:**

- training populations;
- supervision-coverage conditions;
- diagnostic fitting and evaluation infrastructure;
- synthetic teacher-generalization controls;
- necessary qualified references.

**The batch preserves:**

- the WL59 family and its parameterisation;
- positivity and passivity;
- A/O and spatial semantics;
- geometry features;
- K/C ownership;
- the ballistic/collided split;
- normalisation;
- historical checkpoints;
- the original eligibility thresholds.

**Not authorised:**

- rank, width or depth changes;
- feature or architecture changes;
- threshold changes or production-loss changes;
- tuning historical checkpoints;
- a new learned mechanism or spatial heuristic;
- hyperparameter sweeps;
- production K/C changes;
- coupled GI;
- a successor.

**Coordination.** A concurrent session (rest-gi-d2, WL69 follow-up) claimed Worklog 70, LabServer69 and `experiments/sunreuse70`. By agreement this batch is **Worklog 71**. It runs on LabServer63 under `~/rest_cover71`, with code in `experiments/fnto_cover71/` and results in `results/fnto_cover71/`. It does not use LabServer69's GPU or touch the other batch's paths.

- **Starting HEAD:** 7a646f4.
- **Population status:** the WL66/WL68 populations (WL66 slots 0–3, WL68 slots 4–7, cf3) and every model scored on them are development evidence here, never fresh confirmation.
