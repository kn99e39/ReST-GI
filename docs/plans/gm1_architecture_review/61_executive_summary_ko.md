# GM1 Global Transport Architecture Review — 최종 보고 (2026-11-10)

설계 전용 리뷰입니다. 구현·학습·렌더·벤치마크는 하지 않았고, 채택하거나 승인한 것도 없습니다. 기준 리비전은 `Selective_Recompute` @ `c7b0698`입니다. 아래 비용 수치는 모두 추정(EST) 또는 유도(DERIV)이며 측정값이 아닙니다. 에이전트들의 의견 일치는 물리적 정확성이나 novelty의 증거가 아닙니다.

**결론:** GM1 global mechanism으로 방어 가능한 후보가 지금은 없습니다. 세 트랙 모두 교차 리뷰 뒤에 자기 트랙의 mechanism 후보 자격을 철회했습니다. architecture를 고르기 전에 측정되지 않은 수치 두 개를 먼저 확보해야 합니다(권고 유형 C + D).

## A. Current Research State

**WL84가 실제로 확립한 것**
- 학습한 ER closure가 local material 개발 baseline으로 자격을 얻었습니다.
- 그 범위는 coated-conductor 재질군 하나, 정적 장면, 이미지 3장입니다.
- 물리·런타임 계약을 통과했습니다.
- GPU 시간은 layered reference보다 spp당 약 1.5배 적게 듭니다.
- 이미지 오차는 약 0.035–0.04 아래로 내려가지 않는 bias floor가 있습니다.
- Python 추적 하네스에서는 wall-clock 이점이 없습니다.

**GM1에서 아직 검증되지 않은 것**
- ER을 쓴 다중 반사 interreflection의 합성 정확도
- 편집 후 정지 수신면의 변화
- sharp directional transport를 persistent state에 담는 방법
- 품질을 맞춘 편집 후 비용
- global 쪽에 학습된 상태를 둘지 여부 자체

## B. Independent Architecture Candidates

| 트랙 | 핵심 mechanism | 가능성의 근거 | 실패 이유(리뷰 후) |
|---|---|---|---|
| A. Directional Delta Transport | `(I − K C1)Δb = K(Δe + ΔC b0)`. 편집이 만든 변화만 전파 | 1차 변화의 지지집합은 기하적으로 정확히 찾을 수 있고, 수동적인 ER에서는 풀이가 안정적입니다 | 이 항등식은 incremental radiosity를 방향 버전으로 다시 쓴 것이라 비용 이점을 담고 있지 않습니다. `b0`를 방향별로 저장하는 일이 곧 미해결 문제입니다(4D 상태가 α⁻⁴로 커져 α=0.05에서 약 38 GB). 손익분기에 필요한 dirty 비율이 기하적 하한보다 작습니다. **설계자 본인이 기각했습니다.** |
| B. Path-space / reservoir 재사용 | T0 경로를 저장하고, 세그먼트 단위로 무효화한 뒤 replay 또는 GRIS shift | REF-ER의 불편 추정기이고, 정지 수신면의 변화를 정확히 찾아냅니다 | 표현을 갖지 않는 retracer이며, ReSTIR PT / GRIS·difference re-rendering·ray segment tree와 구별되지 않습니다. 새로 생기는 sharp 경로에는 재사용할 샘플이 없습니다. **스스로 통제군 역할로 한정했습니다.** |
| C. LPTG (Lobe-Partitioned Transport Graph) | ER lobe를 rough/sharp로 나눕니다. rough는 영속 incident field로 풀고, sharp는 저장된 ray chain을 replay합니다. 세그먼트 인증으로 무효화합니다 | 구축 비용이 고품질 프레임 1–3장 수준으로 유일하게 감당 가능하고, F≡S가 증명됩니다 | 카메라에 보이는 sharp highlight는 매 프레임 새 chain이 만들므로 sharp 부분은 retracer입니다. field가 점 샘플이라 sharp sender → rough receiver에서 aliasing이 생깁니다. 편집마다 dense solve가 필요해 S≈F입니다. global에 학습 요소가 없습니다. **GM1 후보에서 철회했습니다.** |

## C. Adversarial Findings

- **Physical correctness**
  - 세 설계 모두 K/C 이중계산이나 소유권 공백은 없습니다.
  - A는 tolerance norm이 정의되지 않았습니다(flux가 작다고 radiance가 작은 것은 아닙니다. 1/(1−ρ)만큼 증폭됩니다).
  - C는 biased-consistent 추정기이고, collocation 행렬이 ER의 수동성을 물려받지 않습니다.
- **Directional fidelity**
  - 실용적인 크기의 영속 상태는 α\* ≈ 0.2–0.35 근처에서 band-limited됩니다.
  - 그보다 날카로운 성분은 어느 설계에서도 재추적합니다.
- **Path coverage**
  - GM1의 핵심 현상은 새로 생긴 sharp chain이 rough한 정지 수신면에 닿는 경우입니다.
  - 이 현상을 영속 상태로 담는 설계는 없습니다.
  - 그 영역에서는 65536-spp 참조도 노이즈 한계로 추정됩니다.
- **Update economics**
  - broad lobe를 한 번 거치면 dirty가 퍼져 solve가 dense해집니다.
  - 그래서 선택적 갱신은 지배 비용이 아닙니다.
  - 이는 WL56/59, WL67/69/70의 과거 패턴과 같습니다. 다만 그 기록은 ER baseline의 결함을 측정한 것이 아니라 가능한 실패 mechanism입니다.
- **Scalability**
  - WL84 규모에서는 어떤 비용 주장도 검증할 수 없습니다(64 spp에 약 10 ms, 호출당 추적 0.4–1.2 s).
  - 저장 상태 방식이 유리한 구간은 megapixel 출력에 편집당 여러 프레임이 필요한 영역에서만 나타납니다.

## D. Novelty Kill-search

- **PLAUSIBLY DISTINCT로 분류된 mechanism은 없습니다.** 모든 핵심 mechanism이 ALREADY KNOWN입니다.
- **가장 가까운 선행연구**
  - incremental radiosity (Chen 1990, Drettakis–Sillion 1997)
  - Granier–Drettakis 2001: glossy 증분 갱신. LPTG와 구조가 사실상 같습니다.
  - Bala et al. 1999: ray segment tree
  - Xu et al. 2024: residual path integrals. rigid edit 재렌더에 대해 공개된 가장 강한 비교 대상입니다.
  - ReSTIR PT / GRIS
  - two-pass 방법들
- **남는 것**: application novelty(학습 closure의 lobe 분할을 쓰는 점)와 아직 측정되지 않은 utility뿐입니다.
- **NOT SUPPORTED**: "1/ρ가 모든 선택적 방식의 상한"이라는 주장, "재사용은 경로 탄생에 도움이 안 된다"는 일반형 주장, 그리고 모든 경제성 주장.

## E. Comparative Architecture Judgment

- **강점**
  - B는 정확합니다.
  - C는 감당 가능한 상태와 가장 나은 실험 위생을 갖습니다.
  - A에서 살아남은 통찰은 변화가 바뀐 link 위에만 지지된다는 점과 overwrite-only 원칙입니다.
- **가장 큰 미지수**(모든 결론이 여기에 달려 있습니다)
  1. 동결한 ER closure의 albedo 가중 lobe roughness 분포. Lambert 가중치를 포함합니다.
  2. GM1 fixture에서 cold REF-ER의 time-to-quality.
- **불확실성**
  - α\* 추정이 리뷰어마다 약 3배 차이 납니다.
  - 비용 상수는 3배 오차만으로도 결론이 뒤집힐 수 있습니다.
  - "모든 ER 재질에 Lambert lobe가 있다"는 parameterization에 대한 사실일 뿐 fit된 가중치에 대한 사실이 아닙니다(감사에서 지적).

## F. Coordinator Recommendation

- **권고**
  - 지금은 어떤 mechanism도 채택하거나 시험하지 않을 것을 권합니다.
  - 처음 제안된 DDT도, hybrid도 권하지 않습니다.
- **이유**
  - 각 트랙의 옹호자 자신이 감사를 통과한 논거로 후보 자격을 철회했습니다.
  - 그나마 덜 나쁜 것을 고르면, admission 조건을 측정하지 않은 채 mechanism을 시험하게 됩니다.
- **가장 먼저 시험할 만한 것**(소유자가 원할 경우)
  - mechanism이 아니라 **mechanism 중립 characterisation**입니다.
  - 세 옹호자와 리뷰어 둘이 독립적으로 같은 결론에 도달했습니다.
  - 다만 현재 로드맵 문구에는 이를 넣을 자리가 없습니다.
- **구현 전에 필요한 결정**은 H 절에 있습니다.
- **제 판단의 한계**
  - 이 리뷰는 세 트랙의 설계 공간만 탐색했습니다. 방어 가능한 mechanism이 존재하지 않음을 보인 것은 아닙니다.
  - 특히 기하에 조건화된 *학습된* global 상태는 아무도 제안하지 않았습니다. 제안이 없었다는 사실은 그에 대한 반증이 아닙니다.

## G. Proposed GM1 Validation Contract (설계안, 미승인)

mechanism 시험이 아니라 admission 측정입니다.

- **Stage 0 (렌더 없음)**
  - 동결 closure의 fit된 lobe roughness 분포를 albedo 가중으로 집계합니다.
  - 기하적으로 dirty 확률을 예측합니다.
- **Stage 1**
  - 공통 GM1 fixture 하나에서 REF-P와 REF-ER를 T0/T1에 대해 split-half로 렌더합니다.
  - ER gap g를 먼저 보고합니다.
  - cold REF-ER time-to-quality ladder를 측정합니다.
  - 같은 시드의 T0/T1 REF-ER 경로 로그를 남깁니다. R/S lobe class 라벨을 붙이고 시드는 8개 이상 씁니다.
- **통제**
  - null edit (dirty가 0이어야 합니다)
  - move-and-return
  - stale
  - 기하 마스크
  - 참조 SNR이 미달인 영역은 INDETERMINATE로 처리합니다.
- **측정 항목**
  - path class별 T1 radiance 비중과 |Δ| 비중
  - path-dirty와 pixel-dirty 비율
  - sharp→rough class의 크기
  - cold REF-ER가 g에 고정한 품질에 도달하는 kernel 시간
- **이 판단을 약화시키거나 반증하는 결과**
  - |Δ|의 대부분이 broad-lobe class에 있거나 에너지의 대부분이 α\* 이상의 lobe에 있다면, C형 상태를 명세할 가치가 생깁니다.
  - cold REF-ER가 약 1초 이내에 anchored 품질에 도달하면, 그 workload에는 어떤 재사용 방식도 경제적 여지가 없습니다. 그때는 mechanism보다 workload와 gate가 먼저 풀 문제입니다.
- **이 측정이 가능하게 하는 결정**
  - 저장 상태 계열을 GM1 후보나 baseline으로 명세할지
  - workload 규모
  - sharpness 범위

## H. Reproducibility and Provenance

- **산출물**: `docs/plans/gm1_architecture_review/` (README에 색인)
  - 설계 3편, 리뷰 4편, novelty 3편, 교차 리뷰 3편, 감사 1편, prior-art matrix, synthesis
- **워크플로**: 에이전트 14개, 약 33분, 오류 없음. 봉인된 설계 사이에 누출 징후는 없습니다(감사 §6).
- **감사 결과**
  - 저장소 인용 62건 중 53건 CONFIRMED, 8건 PARTLY, 1건 INACCURATE입니다.
  - INACCURATE 1건은 WL67을 "matched quality"로 표현한 것입니다. 정확한 표현은 "Cycles는 4 spp에서 계약을 충족했다"입니다.
  - 문헌 23건 중 20건 VERIFIED, 2건 PARTLY, 1건 UNVERIFIED입니다.
- **소유자가 결정해야 할 것**(`50_` §5, `60_` §4)
  1. mechanism 중립 측정을 GM1 mechanism 결정보다 먼저 해도 되는지
  2. GM1 global mechanism이 학습 요소를 반드시 포함해야 하는지
  3. GM1의 sharpness 범위(α 하한)
  4. multi-view / 편집당 다중 프레임이 GM1에 포함되는지
  5. workload 규모
  6. warm path-traced re-render 비교군의 명칭과 회계
  7. 공통 path termination 규약
- **미해결 이견**: 12건 (`50_` §7)
