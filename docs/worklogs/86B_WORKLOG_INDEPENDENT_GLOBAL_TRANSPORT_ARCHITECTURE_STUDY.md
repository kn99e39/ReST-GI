# Worklog — Independent Global Transport Architecture Study, Post-WL86A / Pre-WL87 (2026-11-11)

## Origin and scope

- **Brief.** "ReST-GI — Independent Global Transport Architecture Study, Post-WL86A / Pre-WL87" (user, 2026-11-11). The brief asks for the final report in Korean with English technical terms; the report below is kept in that form.
- **Scope.** Design and research study only. No implementation, experiment, training, rendering or GPU use. No code, checkpoint, protocol, earlier worklog or canonical architecture document was changed. No architecture is selected, nothing is authorised, no successor batch or WL87 prompt is produced.
- **Numbering.** 86B = letter numbering after WL86A for unnumbered design work (saved as a worklog at the user's request after the report was delivered).
- **Revision.** `Selective_Recompute` @ `bd81442` at the time of the study.
- **Method.** Three multi-agent workflows: an evidence dossier over WL41–86A and the canonical documents (9 scopes), a prior-art survey (3 angles), and a design panel (4 independent lenses, 4 adversarial reviews, 1 completeness critic). The report is the coordinator's synthesis; supporting material is in `docs/plans/global_transport_study_86b/` (readable digests) and `results/arch_study86b/` (raw structured outputs).
- **Evidence labels.** **[F]** 코드·프로토콜에서 확인한 구현 사실, **[M]** 측정 증거, **[W]** worklog 해석, **[I]** 작성자의 추론, **[H]** 미검증 가설.

---

## A. CURRENT ASSESSMENT

**현재 증거는 어떤 learned global transport architecture도 선택하게 해주지 않는다.** 방어 가능한 입장은 세 개가 조건부로 남고, 이들을 가르는 결정적 증거는 전부 learning-free 측정인데 아직 하나도 없다.

- **반증된 형태** [M]
  - dense learned coupling: 비용에서 탈락(WL52).
  - non-passive·path-history-dependent target에 fit한 뒤 재귀 합성하는 learned K(WL41–51).
  - 합(sum)에만 감독된 다항 learned decomposition(four-part K, ECRO의 single-flight + global).
- **반증되지 않았지만 필요성도 보이지 않은 형태**: ECRO 유형의 learned higher-order response.
  - broad transport에서는 unseen pose에서 E_H 0.003–0.006 [M].
  - 같은 state 위의 explicit 계산과 비교된 적이 없다.
- **ECRO 경제성의 귀속**: 측정된 9×는 "샘플당 5.45× 저렴 × 1.6–1.7× 적은 샘플"이다 [M, `results/ecro86/results/econ/econ86.json`]. 이는 two-tier estimator의 성질이라 물리적으로 채운 table도 똑같이 얻는다 [I].
- **WL86A 실패의 크기**: 0.063 / 0.072 vs gate 0.06, held-out edit 하나의 region class 하나. sharp receiver에서 ER 자체의 물리 reference 대비 gap(0.059–0.065)이 gate와 같은 크기다 [M].
- **초기 추론의 수정**: "sharp 실패는 receiver-side gather의 구조적 문제"라는 작성자의 초기 추론은 약화되었다. 배치된 state ray에 정확한 panel radiance를 넣으면 해당 family의 74–84 %가 전달되는데 learned flight는 1–18 %만 전달한다 [M, WL86A amend. 2]. 실패의 큰 부분은 capacity / loss / optimisation / coverage 사이에서 분리되지 않은 학습 실패다.

---

## B. ARCHITECTURAL PROBLEM

b = K(e + C_G b)에서 ER을 쓰면 T_G = S_ER C_G, q = S_ER e, b = q + T_G b, H_G[q] = Σ_{k≥2} T_G^k q.

| 학습 대상 | rigid edit 의존성 | 물리 계산 비용 | 기록 |
|---|---|---|---|
| K (intrinsic scattering) | 불변 [M, WL53] | pointwise는 closure 평가, nonlocal medium은 slab당 175–380 s [M, WL55/56] | ER만 gate 통과(local, 재질 family 하나). learned nonlocal K는 6–7 % 오차 vs stale exact 0.5–1.2 % [M, WL57–66] |
| C_G (free-space) | mover를 지나는 link만 변함 | ray casting, 5–11 ms @10k–40k ports [M, WL52] | learned C는 유효·안정하지만 8–60× 느림 |
| H_G (higher-order response) | 전역적으로 변함 | 경로 길이 × (ray + closure) | ECRO: broad 통과, sharp class 실패 |
| 압축된 합성계 G → low-rank operator | state 전체 재생성 | 위와 동일 | ECRO가 이 형태 |

구조적 관찰 [I]:

1. **Reuse에는 두 뜻이 있다.** State reuse는 K와 유효한 link를 그대로 두는 것이고, rule reuse는 고정된 생성 규칙이 state를 통째로 다시 만드는 것이다. ECRO는 후자다.
2. **Resolvent에는 edit locality가 없다.** C_G의 변화는 국소적이지만 (I − T_G)^-1의 변화는 dense다. 따라서 higher-order 항은 selective update의 대상이 될 수 없고, 측정으로도 selective state update는 경제적 내용이 없었다(20.8 vs 28.1 ms wall) [M].
3. **Sharp directional relation은 두 곳에만 있다.** S의 narrow lobe와 C의 visibility. broad S를 거칠 때마다 각도 구조가 평활되므로, 압축 가능성의 유일한 물리적 근원은 broad scattering의 반복이다.
4. **GM1에서는 모든 transport 단계가 explicit하게 싸다.** 비싼 것은 분산(spp)과 경로 길이뿐이다. 학습만이 줄일 수 있는 큰 물리 비용으로 측정된 것은 nonlocal K 구성뿐인데, 이는 GM1 범위 밖이다.
5. **ECRO는 event count로 분할하고 K/C segment로 분할하지 않는다** [F]. 정본 문서 어디에도 ECRO와 K/C contract의 관계가 적혀 있지 않다.

---

## C. HISTORICAL FAILURE SYNTHESIS

단일 원인으로 묶이지 않는다.

| 계열 | 실패 내용 [M] | 직접 원인 |
|---|---|---|
| FNTO four-part K (WL41–45) | 64/64 발산(ρ 10.47), hinge 후 static −9 dB | nested support 4항을 하나의 target에 joint fit했고 partition이 없음 [F] |
| Directional interface (WL46–48) | off-normal ρ 9.4–17.1 | 감독되지 않은 (port, direction)에서 non-passive |
| DASQ 합성 (WL50–51) | ρ 6.9–9.0 | Cycles Burley가 diffuse ancestor 없는 vertex에만 적용됨. camera-vertex operator(gain 8×)를 매 bounce 재귀 |
| Learned C (WL52) | 유효·안정(ρ 0.51)이나 8–60× 느림 | explicit이 이미 쌈. patch 평균이 contact / visibility transition을 잃음 |
| Explicit C / CAK (WL53–56) | 1 cm port에서 3–8 % 오차 | K, C 양쪽의 boundary information 부족 |
| Learned intrinsic K (WL57–80) | transfer·deformation gate 전부 실패 | geometry → T map이 감독 source 5–18 mm 안에서만 transfer |
| ECRO (WL86/86A) | E2 region C 0.063 / 0.072 | 몇 개 Lambert patch의 level error(13–20 % 낮음). single flight 미전달을 global 항이 geometry 없이 보상 |

**Four-part K와 ECRO의 공통점**

| 요구 성질 | Four-part K | ECRO |
|---|---|---|
| Exclusive path ownership | 외곽 K/C는 배타적, 내부 4항은 아님 | 외곽 ≤2 events / table은 배타적, 내부 2항은 아님 |
| Directional support | band 19 bins, 나머지는 irradiance | 64 bins, R = 16 채널 |
| Identifiable targets | 없음(K_glob이 Yellow red mode의 86 % 흡수) | 없음(global 항 +0.017…+0.064 과보상) |
| Stable repeated application | 실패 | 해당 없음(one-shot, flux 0.980–0.998) |
| Principled partition | distance band(heuristic) | 외곽 event count는 원리적, 내부 분할은 아님 |

공통 패턴 [I]: 외곽 partition은 배타적이었지만 내부의 additive learned 항들은 합에만 감독되었고, geometry 정보가 빈약한 항이 geometry-specific 항의 부족분을 흡수했다. 두 번째 공통점은 aggregate 지표(field error, flux, whole-image)가 다른 reader 아래의 거동을 예측하지 못했다는 것이다.

공통이 아닌 것: 재귀 불안정(K 전용, reference semantics가 원인), 경제성 실패(learned C, oracle K), 보간 범위 실패(WL57–80).

---

## D. CANDIDATE HYPOTHESES

### D.0 Sharp transport 여섯 질문

1. **유한 ray sampling이 양립 불가능한가?** 아니다. 문제는 "receiver 쪽에서, source를 알기 전에 고정된, sender lobe와 무관한" relation이다. 중요한 patch가 panel을 보는 ray는 512개 중 median 13–17개다 [M]. 다만 그 relation도 정확한 입력이면 74–84 %를 전달하므로 단독 원인은 아니다.
2. **Exhaustive ray 없이 support 찾기**: sender lobe는 ER에서 closed-form으로 sample 가능하다 [F]. Sender-side 추적은 가능하지만 프로젝트 안에 측정이 없다. WL46의 +3.1 dB는 receiver 쪽 lobe sampling이었다 [F, WL46 line 16].
3. **재질 각도 구조의 역할**: lobe 합이 exact하므로 고정 per-lobe 할당의 lobe-class partition은 exact·reciprocal하다 [F, `experiments/fnto_compiled82/closure82.py`]. Incidence 의존 할당은 reciprocity를 깬다.
4. **Support 보존 vs aggregate**: reader가 적분하면 aggregate가 허용된다. 그러나 reader-independent state는 가장 sharp한 reader 기준으로 설계되어야 한다. Back wall은 aggregate로 정확했는데도 mirror reader 아래 7–10 % 결손이 났다 [M, E2 T1].
5. **Representability vs identifiability**: container는 입증되었다(oracle 0.0025–0.0137 [M], 단 sphere와 E1 class E는 noise 수준이 아님). Operator family representability는 미입증이다(free-factor fit이 49–68 %를 남김, INDETERMINATE). Identifiability는 실패했다.
6. **Global term compensation 회피** [I]: 항별로 따로 tally한 target을 주거나, 물리적으로 싸게 계산되는 항은 학습하지 않거나, free additive remainder를 없애야 한다. 어느 것도 재학습으로 시험되지 않았다.

### D.1 H-P — Lobe-class exclusive partition + broad core (가장 강한 정식화, 선택 아님)

- **Operator**: S = S_B + S_N(고정 per-lobe 할당), P_N = (I − C S_N)^-1.
  - broad field: β = S_B P_N e + S_B (P_N C) β.
  - 최종: b = β + S_N P_N (e + C β).
- **물리 계산**: light-side narrow prefix(sender-side), eye-side narrow suffix(path tracing), broad first flight(state rays).
- **학습**: m회 explicit 적용 뒤의 multiply-scattered broad tail을 만드는 generator뿐.
- **Runtime 입출력**: geometry state + source encoding → patch × bin table. Edit 시 weight만 유지되고 state는 전부 재생성된다.
- **Ownership**: lobe-word로 배타적이고, class별 tally가 reference에 합산된다.
- **압축 근원** [H]: broad bounce 반복 후 개별 link 의존성이 희석되어 low rank가 된다.
- **실패 양식**
  - tail 식에 broad → narrow → broad link가 visibility 없이 들어 있다(WL86A에서 geometry-blind였던 형태).
  - moderate lobe(hard population에서 albedo median 94 %)의 owner가 없다.
  - class 경계가 실패한 panel(r50 0.146 rad vs 경계 0.15) 주위에 사후에 그어진다.
  - sharp는 전부 재추적이라 로드맵의 "retracer는 GM1을 완성하지 않는다"에 걸린다.
- **반증 예측**: 재학습 없이 sender-side 항만 더하면 E2 region C가 family별 oracle 수준으로 회복되고, 끄면 WL86A 결손이 재현된다. 회복되지 않으면 partition 설명이 틀린 것이다.
- **학습 전 필요 증거**: 64-outgoing-bin interface의 합성 오차 vs lobe 폭·입사각(실제 closure로 계산), class-swap control.
- **패널 판정**: WEAKENED 2 / REFUTED 1.

### D.2 H-0 — Null with a construction

- **내용**: H-P에서 learned tail을 뺀 것. 같은 state 위의 explicit patch × bin transport이고 학습은 K(ER)에만 있다.
- **지위**: 유일하게 반박되지 않은 null이자 빠진 comparator다. Architecture로서는 learned global object가 없고 고전적 two-pass 구성이라 novelty가 없다. Candidate인지 baseline인지는 소유자 결정이다(WL80B, WL84A X7 선례).
- **불리한 측정**: 세 변형 중 둘(binned sender, exact first flight)은 이미 sharp family를 patch 수준에서 놓친다(−16…−26 %, patch RMS 0.38–0.93) [M]. 셋째 변형(sender-side deposit)은 저장소에 없다.
- **결정적 실패**: ECRO가 통과한 broad / near-contact region에서 explicit table이 편향되면 null이 무너지고 learned closure가 살아 있는 가설이 된다.

### D.3 H-E — ECRO operator 형태 유지, recipe가 원인

- **내용**: b̂ = q + Tq + T·R_θ[e]는 그대로 두고, 항별 target, sender radiance Bq 감독, importance-weighted loss, 더 조밀한 state, 더 큰 R 중 무엇이 실패를 만들었는지가 가설이다.
- **근거** [M]: 정확한 입력의 relation은 74–84 %를 전달하고 learned flight는 1–18 %(E1 panel에서는 31–60 %)만 전달한다. 다섯 후보 원인이 한 번도 하나씩 변경되지 않았다.
- **제약**: 동결 gate 판정은 NOT SUPPORTED이고, 재개는 "no rescue after a result" 규칙에 걸린다. 허용 여부는 소유자 결정이다.
- **결정적 실패**: 항별 target을 준 뒤에도 global 항이 보상하거나, bin-projected panel 행렬의 non-negative rank가 R = 16을 크게 넘으면 형태 자체가 한계다.

### D.4 검토 후 탈락

| 가설 | 탈락 사유 |
|---|---|
| Enclosure-bounded intrinsic operator | 새로운 영역(sharp interior)에서는 compact 표현이 없고, compact한 영역에서는 기존 PRT |
| Learned prolongation (sub-patch closure) | explicit refinement가 1.00–1.04× [M, WL54]. sharp family를 자체 논리로 제외 |
| Control-variate table | 자체 산술로 약 1.2× |
| Defect correction / approximate resolvent | 정확한 explicit operator를 전제하므로 그것이 있으면 불필요. ECRO가 이미 learned table 뒤에 물리 flight 1회를 둔 형태 [F] |
| Learned generator of nonlocal K | GM1 범위 밖. WL57–80 transfer 기록이 일관되게 부정적 |

패널이 검토하지 않은 class: learned 양을 sampling density로만 쓰는 것(틀린 값이 bias가 아니라 variance가 됨). "learned transport response" invariant가 이를 배제하는지는 문서에 없다.

---

## E. CRITICAL COMPARISON

기호: N patches, B bins, R state rays/patch, L lights, P pixels, s spp, D̄ 평균 경로 길이, J sweeps, r rank, δ dirty fraction, c = ray + closure 단가.

| 메커니즘 | Edit당 | Light당 fill | Frame당 |
|---|---|---|---|
| Path tracing (frozen ER) | BVH refit | — | P·s·D̄·c |
| Path-traced table | 동일 | N·B·s_t·D̄·c | P·s·2c |
| Explicit patch × bin | δ·N·R·c_ray | J·(N·R + N·B²) | P·s·2c |
| ECRO 형태 learned | N·R·c_ray + O(N²) generator | N·R + N·B·r | P·s·2c |

- **학습 고유의 절감** [I]: readout 절감은 고유하지 않다. Fill에서 learned와 explicit의 차이는 light당 J회 sweep 대 1회 적용이므로 상한이 J(3–5 추정)다. 반면 generator는 O(N²)다. 학습에만 있을 수 있는 이득은 coarse basis에서의 정확도(sub-patch / sub-bin closure)뿐이고 측정되지 않았다.
- **Sharpness scaling**: 저장 방식은 α^-4 / d_min² [H, WL84A 유도]로 커진다. Sampling 방식은 폭이 아니라 narrow 표면적·light 수·chain 길이에 비례한다.

| 비교 대상 | 측정된 것 [M] | 측정되지 않은 것 |
|---|---|---|
| 물리적으로 매칭된 explicit 계산 | 없음 | 같은 state 위 정확도·kernel time 전부. "ms 규모"는 연산 수 계산 |
| Path-traced table | 256 spp에서 0.157–1.24 s(경합 wall), 4,096 spp에서 6–24 s | rendered error vs kernel time ladder |
| Frozen ER + full transport | 1.806 ms/spp(6-vertex, Russian roulette 없음). WL85에서 cold PT가 g에 0.39 s | roulette PT의 g 수준 비용 |
| ECRO v0 | operator 11.3–11.8 ms, edit 76–78 ms vs PT 35–36 ms | generator kernel time, 10–100× patch scaling |
| 역사적 K/C | explicit C 5–11 ms. path-traced 재렌더가 oracle update를 10–40× 이김(WL67/69). WL70은 근소 우위(취약) | — |
| Prior work | 조사된 precomputed / learned operator 중 rigid-edit 유효성과 sharp glossy interreflection을 함께 갖춘 것은 없음 | 일부 1차 문헌은 abstract 수준만 확인 |

Prior art의 추가 사항: 모든 runtime cache는 footprint / roughness guard로 read를 막고, near-mirror 두 번째 vertex lookup은 그 guard를 위반한다. Physical cache는 frame당 1.5–3.5 ms지만 edit 후 lag가 있다. Learned operator가 고유하게 줄 수 있는 것은 source independence와 zero-lag 유효성이다.

---

## F. STRONGEST COUNTERARGUMENT

**H-P에 대해**
- 실패의 큰 부분은 partition이 아니라 학습 실패라는 것이 WL86A 자체의 수치다(74–84 % vs 1–18 %).
- 64-bin core가 소유할 수 있는 것은 near-Lambert뿐이다. WL58의 10–18 % 손실은 256 outgoing bins 기준이고 64-bin 수치는 없다. Lambert slot share는 median 0.20 / 0.04.
- Sharp glossy interreflection을 persistent state가 하나도 담지 않는다. 같은 이유로 WL84A Track C가 제안자 스스로에 의해 철회되었다.
- 판별 기준 목록이 FNTO·ECRO 실패에서 사후에 도출되었고, ECRO도 자기 사전 선언 contract는 통과했다.

**"Learned global 불필요"로 읽는 것에 대해**
- 부재는 대칭이다. Explicit comparator가 측정된 적이 없어 학습이 불필요하다고도, 필요하다고도 보이지 않았다.
- ECRO의 양의 결과는 실재하며 어떤 관점도 설명하지 못했다.
  - region D change slope 0.78–0.95 vs stale 0.46–0.65.
  - stale은 region A에서 2.6–3.9×, E1 panel에서 7–8× 나쁘다.
- "geometry-conditioned 학습은 실패한다"는 귀납은 방 하나, slab 하나, 합성 family 하나에 근거한다. "학습은 K에서 된다"는 성공 하나(ER)에 근거하며, WL57–80이 K였다는 점을 빠뜨린다.

**증거 기반 자체에 대해**
- WL86 room에서 region C가 읽는 것의 89 %는 Lambert sender다. Sharp reader가 moderate / sharp sender를 보는 glossy-to-glossy higher-order chain은 최대 11 %다. Milestone 이름이 가리키는 family를 어떤 fixture도 시험하지 않았다.
- Sharp 물체가 mover인 edit는 한 번도 없었다.
- Light가 point라서 sharp-sender family가 어려운 것이며, GM1 row는 light class를 고정하지 않는다.
- Fidelity gate는 수렴 품질(4,096 spp 대비)에서, economics ladder는 noise 수준 품질에서 판정되었다. 후자에서는 PT 자체의 region C 오차(0.12–0.32)가 ECRO bias(약 0.07)보다 크다.

---

## G. EVIDENCE GAPS

가설을 가르는 측정을 판별력 순으로 적는다. 1–5는 learning-free다. 이 목록은 증거 공백의 기록이며 어떤 batch도 제안하거나 승인하지 않는다.

| # | 종류 | 측정 | 바뀌는 판단 |
|---|---|---|---|
| 1 | operator representability + economics | 동결 WL86 state 위 explicit patch × bin transport 세 변형을 동일 readout으로. region·path family별 점수, 복수 ray realisation, 비경합 kernel time | H-0이 서는지, learned closure가 살아나는지, partition 설명이 틀렸는지 |
| 2 | end-to-end economics | path-traced table(16–4,096 spp)과 Russian-roulette PT의 rendered error vs kernel time ladder, region C 포함 | two-tier 계열 전체에 경제적 여지가 있는지 |
| 3 | information sufficiency | WL86A RAY / BIN을 독립 ray set 8개 이상과 미열람 edit에서 반복 | receiver-side relation이 구조적 편향인지 예산 부족인지(H-P vs H-E) |
| 4 | operator representability | panel / sphere / floor의 64×64 bin-projected ER 행렬 rank·non-negative rank, 64-bin 합성 오차 vs lobe 폭·입사각 | R = 16 capacity, class 경계 위치와 core 소유 에너지 |
| 5 | information + dynamic | lobe-word tagged radiance 및 T1 − T0 share. region C slope(0.36–0.63)의 family 분해 | broad tail이 ER gap 위의 무엇을 소유하는지, 동적 실패가 정적 실패와 같은 family인지 |
| 6 | learned generalisation | ECRO를 한 요인씩 바꿔 재학습 | H-E의 성립 여부(no-rescue 규칙상 허용 여부는 소유자 결정) |
| 7 | operator representability | sharp reader가 ER sender를 주로 보거나 sharp 표면이 움직이는 구성에서의 oracle table | patch × 64-bin table이 GM1 대상 family에 적합한지 |
| 8 | dynamic + economics | stale control이 broad class에서 실패하고 higher-order share가 큰 workload의 g·cold PT 비용 | geometry-conditioned response가 시험 가능한 workload가 존재하는지 |

**증거가 아닌 소유자 결정**
- frozen ER 주위의 learning-free global solve가 candidate인지 baseline인지.
- multi-view / multi-light / multi-frame이 GM1 안인지, 경제적 목표가 edit 후 첫 유효 frame인지 steady state인지.
- "selective"의 두 정의(AGENTS.md의 complete owned unit vs Architecture §6의 F/S) 중 무엇으로 보고하는지.
- learned sampling density가 "learned transport response" invariant 안인지.

---

## H. ARCHITECTURE ASSESSMENT

**완료 질문에 대한 답**: 물리적으로 근거 있는 learned global transport 가설 중 방어 가능한 것은 조건부로 둘이다.

- **H-P의 learned broad tail**: 재명세가 필요하고, coarse-basis projection error의 closure로서만 방어 가능하다.
- **H-E**: ECRO 형태 유지, recipe가 원인.

둘 다 learned 부분의 역할이 입증되지 않았고, 현재 증거가 지지하는 유일한 문장은 null(H-0)이다. 유효 결론 유형으로는 "결정적 실험이 빠진 여러 가설"에 해당하며, 그 실험이 learning-free라는 점이 이번 연구의 주된 결과다.

**Heuristic decomposition과 구별시키는 증거**
1. Class 규칙이 실패 사례가 아니라 측정량(실제 table 해상도의 합성 오차)에서 사전에 고정되고, 미열람 edit와 두 번째 재질 population에 그대로 적용될 것.
2. Class tally가 reference에 합산되고, narrow lobe를 흡수체로 바꾸면 broad-class 출력이 불변일 것.
3. 기존 reference에서 실패할 수 있는 전방 예측(D.1의 sender-side 항 on/off)이 있을 것.
4. 학습 전에 같은 family의 oracle로 representability를 보이고, learned 항이 같은 state의 explicit sweep과 물리 low-rank tail을 정확도 또는 비용에서 이길 것.
5. Learned 항마다 따로 tally된 target이 있고, reader-importance-weighted family별 점수와 함께 g가 보고될 것.

Production architecture는 선택하지 않았고, 후속 batch를 제안하지 않는다.

---

## I. REPRODUCIBILITY

- **저장소**: branch `Selective_Recompute`, 연구 시점 HEAD `bd81442`. 연구 중 저장소 변경 없음(이 worklog와 보조 자료만 사후에 추가).
- **Workflow run** (세션 내 journal, 저장소 밖):
  - evidence dossier `wf_e801fbe6-3ce`: 9 scope, WL41–86A와 정본 문서.
  - prior-art survey `wf_6867cebb-72e`: 3 angle.
  - design panel `wf_337c43ee-816`: 4 관점 설계 + 4 adversarial review + completeness critic.
- **Critic이 HEAD에서 직접 확인한 파일**: `experiments/ecro86/protocol/ecro86.json`, `results/ecro86/results/econ/econ86.json`, `experiments/ecro86/pt86.py`, `experiments/fnto_compiled82/closure82.py`, `experiments/gm1_char85/pt85.py`, WL46 line 16, 로드맵 GM1 row, Architecture §2 / 3 / 6. 나머지 수치는 dossier 대조다.
- **보조 자료**
  - `docs/plans/global_transport_study_86b/dossier.md` — WL41–86A 사실·측정·해석 dossier.
  - `docs/plans/global_transport_study_86b/prior_art.md` — prior-art survey.
  - `docs/plans/global_transport_study_86b/panel_designs_and_reviews.md` — 네 관점의 가설 명세와 리뷰 전문 요약.
  - `docs/plans/global_transport_study_86b/panel_reviews.md`, `panel_critic.md` — adversarial review와 completeness critic digest.
  - `results/arch_study86b/dossier.json`, `prior_art.json`, `panel.json` — workflow의 구조화 원출력.
- **패널 진술 중 교정된 것**
  - state ray는 "cosine-stratified"가 아니라 64개 equal-solid-angle bin 안에서 균일(cos은 weight) [F]. 보조 자료의 dossier / panel 본문에는 교정 전 표현이 남아 있다.
  - "64 bins에서 10–18 % 손실"은 256 outgoing bins 수치.
  - "256-spp table 0.16 s GPU"는 8-shard 경합 wall의 평균.
  - "109 M rays 0.105 s"는 throughput 외삽.
  - "1–18 % 전달"은 E2 한정.
- **미검증 가정**
  - explicit recurrence의 비용과 정확도.
  - sender-side deposit의 patch 수준 정확도.
  - −20 % RAY 결손이 편향인지 noise인지(ray realisation 하나).
  - panel 행렬의 rank.
  - α^-4 / d_min² scaling.
  - broad core의 low-rank 성질.
- **Post hoc 지위**: WL86A amendment 1·2, identical-encoding beam 통계(저장소에 script 없음), WL85의 architecture 관련 해석(Q3b로 형식상 INDETERMINATE).
- **문서 불일치(수정하지 않음)**
  - AGENTS.md가 Architecture / Roadmap보다 오래됨.
  - 로드맵의 "edit-response gates passed"는 region D만 해당.
  - WL86 worklog의 "no memorisation"은 aggregate 비율 기준(box patch는 unseen에서 1.6–2.7× 나쁨).
  - WL86 worklog의 "both terms needed"는 재학습 ablation이 아니라 항 제거.
- **1차 문헌 한계**: Wallace 1987, Sillion & Puech 1989, Xu et al. 2014, Drettakis & Sillion 1997 등은 abstract 수준으로만 확인했다.
- **보조 자료의 지위**: subagent 출력이며 worklog 본문이 교정한 오류를 포함한다. 본문과 다르면 본문이 우선한다. 시각 자료는 없다(설계 전용 연구, 렌더·측정 없음).
