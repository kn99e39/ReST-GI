# Worklog — WL94A (brief label WL94-R): ECRO Material Fidelity Retention Smoke (2026-11-14)

## Origin and scope

- **Brief.** "WL94-R — Material Fidelity Retention Smoke, Non-numbered follow-up to WL94" (user, 2026-11-14).
- **Batch class.** This is a suffix of WL94: a dependent evidence-gap follow-up under the same architecture contract. The brief calls it WL94-R; under the repository's letter convention the worklog is **94A**. The owner decides the name; no historical number changes.
- **Hypothesis H-R.** WL94 B lost baseline fit because fine-tuning removed the original material–object distribution. A 50/50 mixture would keep baseline fidelity and still recover material generalization.
- **Preserved, unchanged:**
  - the WL94 labels (H-M INCONCLUSIVE, H-A INCONCLUSIVE);
  - the WL94 A_s8601 / B_s8601 checkpoints, tables, renders and measurements, reused read-only through links;
  - the frozen own checkpoints, the CK2 scene, its references and the WL94 code.
- **Code.**
  - `r94.py` and `run94r.sh` in [experiments/ecro_mat94r/](../../experiments/ecro_mat94r/).
  - It imports `experiments/ecro_mat94` (mat94, c94, an94, viz94) unchanged.
- **Protocol.** [protocol/ecro_mat94r.json](../../experiments/ecro_mat94r/protocol/ecro_mat94r.json). Freeze commit `5b7c0c8`.
- **Where.**
  - Server: LabServer63 `~/rest_x94r`, one detached chain, 12:19–12:30 UTC (11 min).
  - Local results: `results/ecro_mat94r/`.
  - Figures: `results/evaluation/94A/`.

## Implementation Fact

**Candidate C.**
- It uses the same 96 WL94 training geometries in the same order.
- The geometry at sorted rank *k* uses:
  - the original materials and the cached WL94 `train_A` targets if *k* is even;
  - the diversified materials and the cached `train_B` targets if *k* is odd.
- This gives exactly 48 / 48 examples, 96 in total, the same as A and B. No examples were added.
- Training is the **unchanged** WL94 `c94.train`: OWN objective, 2,000 steps, Adam 3e-4 → 1e-5, the same init and the same data order rng(seed + 94). Only the item list differs.

**Why rank parity.** The brief fixes the ratio at 50/50. Rank parity uses every geometry exactly once and is deterministic. It also keeps the WL94 order, so at every step C trains on the same geometry that A and B did. The resulting step split is 1,002 / 998 (s8601) and 1,005 / 995 (s8602), precomputed in the protocol.

**Tests (all PASS).**
- T_REPRO:
  - A / B state sha256 equal the WL94 final hashes;
  - CK2 tables recomputed from the checkpoints match the stored ones to 1.8e-7;
  - WL94 Stage C metrics are reproduced to 1e-6.
- T_MIX: 48 / 48 examples; per-item material digests match select94; the step split is as precomputed.
- T_LEAK: no 0.3 / 0.8 and no within-rectangle variation in any training map.
- T_INIT: C, Ar and A_s8602 / C_s8602 start from the same init as A for that seed (s8601 `a85ddb42…`).
- T_PROPAGATION: every loaded item's model input map equals its target map.

**Training determinism (Ar).** A_s8601 was retrained with the same code and data as a control. It reproduced A to about 1e-11 in every metric (untextured ROI E_H 0.039031646 in both). The variability that matters here is therefore the seed, not run-to-run noise.

**Seed 8602.** It was run because C_s8601 passed F2 and MR (brief Stage D). It comprises C_s8602 and a compute-matched fixed-material control A_s8602, both started from own_s8602.

## Predeclared criteria (protocol, before any C output)

- **F2:** V_orig rel L1 ≤ 1.10 × initial (WL94 threshold).
- **MR:** WL94 material criteria at the table level (C0, B, box A):
  - amplitude in [0.85, 1.15] for every textured variant;
  - corr ≥ 0.90 for the checker variants;
  - the V_ck Lambert population amplitude in [0.85, 1.15] with corr ≥ 0.90.
- **RET:** untextured CK2 ROI_T E_H(C) ≤ E_H(A, same seed) + max(2 × noise bound, 0.005).
- **Outcome**, which must hold for every run seed:
  - COEXISTENCE_OBSERVED: F2 ∧ MR ∧ RET.
  - MATERIAL_GAIN_WITH_BASELINE_LOSS: MR ∧ ¬(F2 ∧ RET).
  - BASELINE_RETAINED_MATERIAL_GAIN_LOST: F2 ∧ RET ∧ ¬MR.
  - INCONCLUSIVE: anything else.
- **Forgetting answer:**
  - SUPPORTED if C passes F2 and RET with a panel-face G bias within 0.05 of A's;
  - CONTRADICTED if C fails F2 or RET with a panel-face G bias ≥ 0.5 × B's;
  - INCONCLUSIVE otherwise.

## Measurement

**Primary criteria.** Every entry is the C0 arm.

| | s8601 A (WL94) | s8601 B (WL94) | **s8601 C** | s8602 A | **s8602 C** |
|---|---|---|---|---|---|
| V_orig rel L1 (init 0.0778 / 0.0803; limit 0.0856 / 0.0883) | 0.0768 | 0.0879 ✗ | **0.0808 ✓** | 0.0798 | **0.0841 ✓** |
| CK2 B amplitude U07 / U03 / bx75 / bx38 / bx38i | 0.64 / 0.13 / 0.67 / 0.29 / 0.34 | 1.05 / 0.99 / 1.08 / 1.06 / 1.06 | **1.03 / 0.94 / 1.04 / 1.02 / 1.02** | 0.61 / 0.34 / 0.65 / 0.45 / 0.45 | **1.02 / 0.98 / 1.04 / 1.04 / 1.04** |
| checker corr (bx75 / bx38 / bx38i) | 0.97 / 0.91 / 0.93 | 0.99 / 0.99 / 1.00 | 0.98 / 0.99 / 0.99 | 0.88 / 0.94 / 0.94 | 0.97 / 0.99 / 0.99 |
| V_ck Lambert amplitude (corr) | 0.40 (0.91) | 1.05 (1.00) | **1.05 (0.99)** | 0.50 (0.94) | **1.08 (1.00)** |
| MR | ✗ | ✓ | **✓** | ✗ | **✓** |
| untextured ROI_T E_H (signed) | 0.039 (+0.003) | 0.087 (−0.086) | **0.084 (−0.082)** | 0.093 (+0.088) | **0.067 (+0.058)** |
| RET (limit = A + 0.017) | — | ✗ | **✗** | — | **✓** |

**Per-seed outcome.**
- s8601: MATERIAL_GAIN_WITH_BASELINE_LOSS.
- s8602: COEXISTENCE_OBSERVED.
- Across seeds: **INCONCLUSIVE** (seed disagreement).
- Forgetting answer by the frozen code: **CONTRADICTED**. The implemented rule reads seed 8601 only; the protocol text does not name a seed. On s8602 the same rule would give SUPPORTED. Disclosed below.

**Secondary measurements.**
- **Frozen own, for reference.** Untextured ROI E_H is 0.047 (s8601) and 0.042 (s8602). Fine-tuning A on the 96 original-material geometries moves it to 0.039 and **0.093**. On s8602 the fixed-material control itself brightens the panel reflection (+0.088), with the panel-sender single flight on the panel face overshooting at +0.86.
- **Panel-facing box face, untextured, signed error.**
  - Learned G:
    - s8601: A −0.038, B −0.442, **C −0.416**.
    - s8602: A −0.057, **C −0.071**.
  - B total: s8601 C −0.382; s8602 C +0.103 (A +0.188).
  - Panel-sender single flight (vs e2_p): s8601 C 0.534 rel (−0.48), A 0.534 (−0.17); s8602 C 1.02 (+0.67), A 1.21 (+0.86).
- **Textured ROI_T E_H (C0)** for U07 / U03 / bx75 / bx38 / bx38i.

  | seed | A | C | B (s8601 only) |
  |---|---|---|---|
  | s8601 | 0.056 / 0.163 / 0.041 / 0.066 / 0.049 | 0.086 / 0.059 / 0.081 / 0.082 / 0.067 | 0.101 / 0.066 / 0.091 / 0.095 / 0.081 |
  | s8602 | 0.044 / 0.218 / 0.055 / 0.075 / 0.050 | 0.051 / 0.078 / 0.057 / 0.066 / 0.062 | — |

- **ROI texture-response slope**, U07 / U03 / bx75 / bx38 / bx38i.

  | | U07 | U03 | bx75 | bx38 | bx38i |
  |---|---|---|---|---|---|
  | s8601 A | 0.49 | 0.24 | 0.57 | 0.36 | 0.44 |
  | s8601 C | 0.72 | 0.61 | 0.80 | 0.77 | 0.82 |
  | s8602 A | 0.46 | 0.64 | 0.51 | 0.50 | 0.41 |
  | s8602 C | 0.93 | 1.11 | 1.02 | 1.21 | 1.20 |

- **DIFF (diffuse reader) slope**, U07 / U03 / bx75: A ≈ 0.42; C 0.84–0.86 (s8601) and 0.91–0.97 (s8602).
- **SSR, untextured E_H(C0) − E_H(C2_srcS1), share of E_H(C0).**

  | | A | C |
  |---|---|---|
  | s8601 | 0.30 | 0.10 |
  | s8602 | 0.72 | 0.66 |

  SSR passes for A_s8602 and C_s8602 and fails for A_s8601 and C_s8601.
- **V_ck whole rel L1.** Init 0.305 / 0.276. A 0.315 / 0.280. C 0.110 / 0.118 (s8601 / s8602). B (s8601) 0.106.
- **Cost.**
  - The whole chain took 11 min, including two extra trainings (Ar, A_s8602).
  - Each training took 107–110 s at 656 MB.
  - Targets and references were reused; no new physical reference was rendered.

## Observation

1. **Mixed training preserved original-material fidelity under F2 on both seeds.** V_orig was 0.0808 and 0.0841, inside the limits but above A (0.0768, 0.0798).
2. **It retained the material-response generalization on both seeds.** CK2 amplitudes were 0.94–1.04, checker corr ≥ 0.97 and population amplitude 1.05 / 1.08, essentially at B's level.
3. **The untextured sharp baseline split by seed.**
   - On s8601, C reproduces B's failure: the reflection is dark (−0.082), and the panel-face G collapses (−0.416, against B's −0.442 and A's −0.038).
   - On s8602, C keeps the panel-face G (−0.071 against A's −0.057) and is *better* than A (0.067 vs 0.093).
   - However, A_s8602 itself degraded from frozen own (0.042 → 0.093), so "retention relative to A" is measured against an unstable control.
   - Relative to frozen own, both C runs are worse in the untextured reflection: 0.084 vs 0.047, and 0.067 vs 0.042.
4. **Textured sharp ROI.** On s8601, C is worse than A on 4 of 5 variants (all but U03) and better than B. On s8602 the result is mixed (better on 2 of 5: U03 and bx38). The reflected checker contrast improved strongly (slope 0.36 → 0.77 on s8601; 0.50 → 1.21 on s8602, an overshoot). Better contrast is not better absolute transport.
5. **The panel-sender error is independent of material generalization.** The panel-sender single flight on the panel-facing face is wrong by 0.53 (both A and C, s8601) and by 1.02–1.21 (s8602), with opposite sign per seed. Exact panel sender removes 30–72 % of the untextured reflection error where the baseline is bright (A s8601, s8602 runs). Its share is small where the global term has collapsed (C s8601: 10 %).

## Interpretation

- **Physical correctness.** This is inherited and re-verified: the WL94 tables and metrics reproduce, the material routing is unchanged, and the references are reused.
- **Model behaviour.**
  - A 50/50 mixture is enough to keep the WL94 material-response gain and to stay within the original-material fit limit, on two seeds. Material generalization and the F2 fit criterion therefore coexist at this budget.
  - The sharp-panel baseline is not governed by the material mixture alone.
  - Whether the panel-face global term collapses depends on the seed: it did for s8601 (as in B) and did not for s8602.
  - Even fixed-material fine-tuning on 96 geometries moves the sharp baseline substantially for s8602.
- **Forgetting as the main reason for B's regression.**
  - It is **not established**. The frozen-code answer is CONTRADICTED, from s8601: with half the examples original, C still shows B's panel-face collapse.
  - s8602 points the other way.
  - The evidence supports this much: the panel-face / sharp baseline is fragile under smoke fine-tuning in a seed-dependent way, whatever the material mixture.
- **Sharp-specific residual, independent of material generalization.** It is the panel sender's once-scattered delivery to the panel-facing box face (53–121 % error, sign seed-dependent), together with the panel-mediated global term on that face. Neither is fixed by the material distribution. This is consistent with WL93D (C2_src) and WL94.
- **Architecture inference (bounded).** At smoke scale, the existing ECRO represents material-general responses without losing original-material fit when original examples are retained. Its sharp-panel transport is not stabilised by the material distribution. This short-run result is not a representation-class statement.

## Unresolved Question

- Is the seed-dependent panel-face global collapse (s8601 C / B) an optimisation-path effect or a capacity trade-off? Two seeds are not enough to decide; no further runs are authorised.
- What is the right baseline for RET? A degraded on s8602 relative to frozen own, so A is not a stable "original baseline". Against frozen own, neither C run retains the untextured reflection.
- The forgetting-answer rule did not name a seed. The implemented rule reads s8601. This is a protocol wording gap; it is disclosed and nothing was remapped.
- DIFF bx38 / bx38i remain noise-limited, as in WL94.

## Architecture Verdict

- **Outcome: INCONCLUSIVE.**
  - s8601 maps to MATERIAL_GAIN_WITH_BASELINE_LOSS (RET fails).
  - s8602 maps to COEXISTENCE_OBSERVED.
  - The predeclared mapping requires agreement across run seeds.
- **Answers to the brief.**
  - Original-material fidelity: preserved under F2 on both seeds.
  - Material generalization: retained on both seeds.
  - Untextured sharp baseline: lost on s8601, kept relative to A on s8602, but worse than frozen own on both.
  - Textured sharp ROI: worse than A on s8601, mixed on s8602; contrast improved.
  - Panel-face global term: collapsed on s8601 (−0.42), kept on s8602 (−0.07).
  - Forgetting as the main cause: not established (seed-dependent).
  - Sharp-specific residual: the panel-sender delivery to the panel-lit face plus the panel-mediated global term, independent of material generalization.
- No adoption, no full training and no successor. WL94's H-M / H-A labels are unchanged. The decision returns to the owner and the supervising assistant.

## Reproducibility

- **Commit and protocol.** Freeze `5b7c0c8`, protocol sha256 prefix d0fd618cb4e1cc9e.
- **Server.** `~/rest_x94r`, which links read-only to `~/rest_x94` for data, fields, references, WL94 tables, models and pop. The dev root is `~/rest_x94rdev` (not evidence; no values were printed and no dev figures were opened).
- **Checkpoints.**
  - own_s8601 `739e7eb5…`, own_s8602 `cee9a09a…`.
  - A / B s8601: as in WL94; state hashes re-verified in `results/ecro_mat94r/repro.json`.
  - C / Ar / A_s8602 / C_s8602: initial and final state hashes in `results/ecro_mat94r/models/*.json`.
- **Assignment manifest.** Rank parity over `select94.json` (A digests at even ranks, B digests at odd ranks); the B albedo histogram is as in WL94.
- **Seeds.** Data order rng(8695) / rng(8696). Renders use 86600 + r. References are reused from WL94.
- **Budget.** 2,000 steps, 96 examples per candidate (48 / 48 for C).
- **Cost.** 11 min wall in total; 107–110 s per training.
- **Figures** (`results/evaluation/94A/`, manifest with sha256):
  - `figR1_summary_s<seed>`: amplitudes against the MR band, untextured ROI with the RET limit, per-variant ROI, panel-face signed B / S / G;
  - `figR2_training_s<seed>`;
  - `figR3_images_<variant>`: REF / frozen own / A / B / C higher-order close-ups, signed error and texture response on shared absolute scales and one exposure.

## Final report (Korean, for the owner)

**1. Implementation Fact (구현 사실)**
- 후보 C는 WL94와 같은 96개 기하와 순서를 씁니다. 짝수 순위 48개는 원래 재질(A 타깃), 홀수 순위 48개는 다양화 재질(B 타깃)입니다.
- 학습 코드, 초기값, 2,000 step, 목적함수는 WL94와 같습니다. 바뀐 것은 항목 목록뿐입니다.
- 재현 검사, 50/50 계약, 누수 검사, 초기값 동일성을 모두 통과했습니다.
- A를 재학습하면 원래 A와 1e-11 수준으로 같습니다. 즉 학습이 결정적이라, 변동 요인은 시드뿐입니다.

**2. Measurement (측정)**
- **원래 재질 적합(F2):** C는 s8601 0.0808, s8602 0.0841로 두 시드 모두 통과했습니다.
- **재질 응답(MR):** C는 진폭 0.94–1.04, 모집단 1.05 / 1.08로 두 시드 모두 통과했습니다.
- **미텍스처 sharp ROI:**
  - s8601은 C 0.084 대 A 0.039로 실패입니다. 반사가 어둡고, 패널 면 전역 항은 −0.42로 B(−0.44)와 같습니다.
  - s8602는 C 0.067 대 A 0.093으로 통과했고, 패널 면 전역 항도 −0.07로 유지됐습니다.
  - 단, A_s8602 자체가 frozen own 0.042에서 0.093으로 악화돼 있었습니다.

**3. Observation (관찰)**
- 원래 예제를 절반 섞으면 재질 일반화와 원래 재질 적합(F2)이 두 시드 모두에서 함께 성립합니다.
- sharp 패널 기준선은 시드에 따라 갈렸습니다. 원래 재질만으로 미세조정한 대조군도 s8602에서는 크게 흔들렸습니다.
- 패널 sender 오차는 A와 C가 같습니다. 즉 재질 분포와 무관합니다.

**4. Interpretation (해석)**
- 재질 응답 일반화는 이 아키텍처에서 원래 적합을 크게 잃지 않고 얻을 수 있습니다.
- 반면 sharp 패널 전송(패널 sender 공급과 패널 매개 전역 항)은 재질 분포로 안정되지 않고, smoke 미세조정 자체에 시드 의존적으로 취약합니다.
- 망각이 B 퇴행의 주원인인지는 확립되지 않았습니다. 동결 코드 판정은 s8601 기준 CONTRADICTED이고, s8602는 반대 방향입니다.

**5. Unresolved Question (미해결 질문)**
- 패널 면 전역 항이 붕괴하는 것이 최적화 경로 때문인지 용량 때문인지 가르지 못했습니다.
- RET 기준선이 불안정합니다. frozen own 대비로는 두 C 모두 미텍스처 반사가 나빠졌습니다.
- 망각 판정 규칙에 시드 지정이 빠져 있었습니다(공개).

**6. Architecture Verdict (아키텍처 판정)**
- **INCONCLUSIVE.** s8601은 MATERIAL_GAIN_WITH_BASELINE_LOSS, s8602는 COEXISTENCE_OBSERVED로 시드 간 결과가 엇갈렸습니다.
- 채택, 전체 학습, 후속 배치는 없습니다. WL94 판정은 그대로입니다.

**7. Reproducibility (재현성)**
- freeze `5b7c0c8`, LabServer63 `~/rest_x94r`, 11분 소요, 그림은 `results/evaluation/94A/`.
