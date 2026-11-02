# Spatial-Map Coverage and Cross-Source Generalization v1 (WL71)

User direction dated 2026-11-02. The chronological record is in Worklog71; the frozen contract is `experiments/fnto_cover71/protocol/cover71_v1.json`.

This is a bounded follow-up to WL68 on **experimental copies** of the unchanged WL59 P intrinsic-K family. It does not change, adopt or propose any of the following:

- the representation, rank or width;
- geometry features;
- A/O or spatial semantics;
- K/C ownership;
- thresholds;
- production checkpoints.

## Question

Can the unchanged geometry-conditioned WL59 spatial map learn a source-independent rule that predicts qualified collided responses at unseen source ports and an unseen geometry state? And does the WL68 failure come from insufficient source supervision or from a limitation of the map?

## Stage 1: identifiability on development evidence

Stage 1 analysed the WL68 fits and references (`stage1_71.py`):

- On rest, every top-face port has the same non-position local features; a source is identified only by its absolute position.
- Physical collided responses are translation-similar across the slab: about 90% of the flux falls within 1 cm at every tested port.
- The fitted WL68 networks reproduce this at fitting ports but not at new ports. At port 2707, 34% of the flux falls within 1 cm.
- Their T changed by 80–95% almost everywhere, not only near the fitting sources.

Whether more sources constrain this position-specific solution is the hypothesis tested here.

## Design

| Element | Fixed choice |
|---|---|
| Spatial-only arms | Train everything except `in_logits`/`out_logits` (the WL68 `net` group). The A/O are fixed to the WL68 `net_all_5902` bases, which passed the WL68 held-out oracle. |
| Coverage | S4: WL66 slots 0–3. S16: S4 plus WL68 slots 4–7 plus 8 new ports (2 inside the pt0 dent on the top face, 1 below it, 5 flat), chosen by a geometry-only rule. Training geometries are rest and pt0. |
| Arms | **A** frozen WL59. **B** WL68 fits. **C** S4 at 4 replicas per source. **D1** S16 at 4 replicas per source (fixed per-source precision). **D2** S16 at 1 replica per source (fixed total budget, equal to C). **J** joint A/O on S16 (separately controlled). **E** labelled oracle with fixed A/O. Inits are P_5901 and P_5902, each with the fixed A/O. |
| Teacher control | TA (WL68 `net_all_5902`) and TB (P_5902 spatial map plus fixed A/O). Students start from P_5901 with fixed A/O, are trained on S4 or S16 teacher responses, and are scored noise-free at the final ports (rest, cf2), at 64 random ports and on the cf2−rest signed change. |
| Final held-out | 8 previously unscored ports on rest and on cf2. cf2 is a WL57 confirmation pose never used for probe references and the farthest dent from pt0. 4 replicas each, new seeds. Includes the signed deformation cf2 − rest. |
| Development | cf3 (WL68 references), labelled as development. |
| Optimizer / objective | WL68 `net` optimizer and objective J (20,000 steps). No sweep. |
| Criteria | WL59/WL68 quality thresholds on the upper 95% bound (0.10 / 0.15 / 0.25). WL59 dynamic thresholds for deformation: error ≤ 0.35 and corr ≥ 0.90. WL68 teacher thresholds. |

## Decision

The **primary** candidate is the D1 run with the lower final J.

**Outcome rules:**

| Outcome | Condition |
|---|---|
| A | S_fit, S_cross (FIN_rest), S_state (FIN_cf2) and S_def (DEF_cf2) all pass. |
| B | Not A, and the primary improves on the same-init C run by at least 20% in the worst-channel contrast and collided error on both final groups. |
| C | Not A or B, and controls are adequate. Labelled by whether the S16 teacher students generalize. |
| D | Otherwise (confounded). |

**Adequate controls** means all three hold:

- the primary converged;
- both S16 teacher students recover their teachers at the training sources;
- the final groups are precision-qualified.

**Descriptive only:**

- D2 against D1;
- J against the WL68 sparse joint fits;
- the oracle;
- extended fields and passivity;
- Stage 1;
- the development groups.

## Code

Code is in `experiments/fnto_cover71/`:

- `ports71.py`
- `probes71.py`
- `fit71.py`
- `eval71.py`
- `export71.py`
- `chain71.py`
- `family71.py`: a batched restriction of the network, certified against `family68`
- `test71.py`
- `stage1_71.py`
