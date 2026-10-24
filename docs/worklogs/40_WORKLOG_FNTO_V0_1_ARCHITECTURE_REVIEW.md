# Worklog — FNTO v0.1 Architecture Review: Utility and Success Probability (2026-10-23)

## SESSION QUESTION

사용자 질문 (2026-10-23):

> ReST-GI의 다음 방향성으로서 FNTO(Factorized Neural Transport Operator, v0.1) 아키텍처가 얼마나 효용성이 높을지,
> 실제 성공 확률이 어떻게 되는지 검토해 달라.

**검토 대상은 두 가지다.**
1. 사용자가 붙여 넣은 FNTO v0.1 원안. Object-local neural transport kernel K_a(sharp kernel + 저랭크 internal kernel)와 학습 inter-object coupling C_φ로 구성되고, `b_a = K_a[e_a + Σ_{b≠a} C_ab(R_ab) b_b]`, 선형성 계약을 둔다.
2. 같은 날 08:50–08:53에 커밋된 저장소 계약 `docs/FNTO_ARCHITECTURE_CONTRACT_V0_1.md`와 living doc 등록(5cf1680..d054082). 원안에서 바뀐 점이 있다. intrinsic optical domain을 쓰고, 노출된 동일 asset 자기 상호반사를 C로 옮기며(a=b 허용), Stage A/B 학습 계약을 둔다.

**성격:** 분석·검토 세션이다.
- 실험, 학습, 렌더를 하지 않았다.
- living document를 수정하지 않았다.
- 사용자 결정이나 구현 승인을 기록하지 않았다.
- 아래 확률은 측정값이 아니라 검토 에이전트의 보정된 주관 추정치다.

## A. 방법

다중 에이전트 workflow 두 개를 실행했다(읽기 전용, 총 18 에이전트, 오류 0).

| run | 구성 | 에이전트 |
|---|---|---|
| `wf_b02d0a32-ad3` (원안 검토) | 독립 검토 5관점: 물리·수학 계약 / 학습 가능성·데이터 / 선행 연구·신규성 / 프로젝트 정합성·주장 감사 / 비용·일정·평가. 관점마다 반박 검증자 1명. 이후 낙관(steelman)·회의(skeptic) 확률 추정자 2명 | 12 |
| `wf_03a463a2-cd8` (저장소 계약 차이 검토) | 3관점: 도메인 분할 물리 / 학습·결합 식별 가능성 / 문서 정합성·거버넌스. 관점마다 반박 검증자 1명 | 6 |

- 반박 검증자는 저장소 파일을 다시 확인했다. 문헌은 웹 검색으로 대조했다.
- 판정은 upheld / weakened(교정된 주장으로 대체) / refuted(제외) 셋 중 하나다. 확률 추정자에게는 refuted를 뺀 발견 사항만 전달했다.
- 일부 에이전트는 `.blend`와 메시 파일을 headless로 직접 읽었다. Yellow SSS falloff, 부품 수, 벽 두께 등이 그렇게 확인됐다.
- 원본 출력과 스크립트: `results/fnto_review/`(gitignore 대상, 로컬 보관). 해시는 §H 참조.

**한계:**
- 확률 수치는 단일 추정자의 판단이다. 소수 둘째 자리의 정밀도는 의미가 없다. 낙관–회의 사이를 대략적인 불확실성 범위로 읽는다.
- 일부 비용 수치는 근거가 약한 외삽이다. 예: Stage A 10–300 GPU-h, 5–8주.
- 원안 검토의 일부 에이전트는 이미 커밋된 저장소 계약도 함께 읽었다. 따라서 원안과 계약의 차이는 두 workflow 모두에서 다뤄졌다.

## B. 결론 요약

- **효용:** 성공하면 프로젝트에서 가장 강한 결과다.
  - 고정 학습 K가 SSS, 유리, 광택을 RNA급으로 표현한다.
  - 강체 편집이 파라미터 갱신 0, T1 감독 없이 몇 초 만에 흡수된다. Surface-Owned 재학습은 약 1.9 h가 걸렸다(WL37).
- **성공 확률:** 낮다(누적, §C).
  - Lego 정적 게이트(S2) 통과는 0.12–0.21이다.
  - 재학습 없는 T1 편집 복구(S3)까지는 0.06–0.12다.
  - SIGGRAPH 2027 제출 가능 수준(S5)까지는 약 0.01이다.
- **핵심 병목은 아이디어 자체가 아니다.**
  - (i) 이 Lego 장면의 노란 SSS 렌더러 의미론(§D-2)
  - (ii) Stage A 지도 데이터 경로 부재(§D-4)
  - (iii) 선행 연구 대비 차별화(§E)
- **연구 질문 재정의:** FNTO가 성공하면 강체 편집에서 학습 파라미터의 dirty 집합이 구조적으로 비어 있게 된다. "무효화된 unit의 완전한 학습 상태 재계산"이 "결합·상태의 증분 전파"로 바뀐다. 이것은 현재 연구 정의(결정 3)의 극한 경우이고, Surface-Owned 메커니즘의 한 사례가 아니다. 이 재정의의 수용 여부는 사용자 결정 사항이며 아직 기록된 결정이 없다(§G).

## C. 단계별 성공 확률

단계 정의(두 추정자 공통). 각 단계는 앞 단계 통과를 전제로 한다.
- **S1 계약 구현 가능:** FNTO-v0를 구현할 수 있고 합성 계약 테스트를 통과한다. 구현 요건은 커널 파라미터화, 학습 데이터, 결합, 선형+비음수+에너지 제한이다. 테스트는 선형성, 해석적 SSS slab, 얇은 껍질 투과, 결합 수렴이다.
- **S2 정적 표현력:** native Lego 홀드아웃 T0에서 RNA_T0(33.57 dB) 대비 −2 dB 이내. SPT-1이 −11.3 dB로 탈락한 바로 그 게이트다. 노란 SSS와 유리가 더 이상 지배적 오차원이 아니어야 한다.
- **S3 편집:** 홀드아웃 T1 박스 편집에서 파라미터 갱신 0, T1 감독 없이 changed-region R ≥ 0.8(자기 재학습 대비), 그리고 전체 이미지가 자기 T1 재학습 대비 약 0.5 dB 이내.
- **S4 일반화:** 두 번째 장면에서 상호반사·광택 지배 편집 또는 회전 편집에서도 재사용이 성립한다. RNA 자체가 실패하는 near-mirror 도체는 제외한다.
- **S5 제출:** 검증된 선행 연구 대비 신규성과 증거 폭을 갖춘 SIGGRAPH 2027 제출이 마감 내에 가능하다.

단위는 확률(0–1)이다. "조건부"는 앞 단계 통과를 전제로 한 값이고, "누적"은 S1부터 해당 단계까지 전부 통과할 확률(조건부의 곱)이다.

| 단계 | 낙관 조건부 | 낙관 누적 | 회의 조건부 | 회의 누적 |
|---|---|---|---|---|
| S1 계약 구현 가능 | 0.70 | 0.70 | 0.60 | 0.60 |
| S2 Lego 정적 −2 dB 이내 | 0.30 | 0.21 | 0.20 | 0.12 |
| S3 재학습 없는 T1 편집 복구 | 0.55 | 0.12 | 0.50 | 0.06 |
| S4 두 번째 장면 일반화 | 0.45 | 0.05 | 0.35 | 0.02 |
| S5 SIGGRAPH 2027 제출 가능 | 0.25 | 0.013 | 0.30 | 0.006 |

**근거 요약:**
- **S1:** SPT-1의 합성 테스트 스위트, relation과 visibility 코드, Jacobi 기반을 재사용할 수 있다(`experiments/spt_lego/`). escape-path(adjoint) 추정기는 자극 없이 커널 지도 신호를 만들 수 있다. 감점 요인은 세 가지다. 양수성과 수동성이 미지정이고, 학습된 multi-bounce solve를 통과하는 gradient의 메모리가 미예산이며, 원안과 계약의 K/C 분할이 불일치한다.
- **S2:** 가장 어려운 관문이다. 노란 재질 하나가 결정적이다(§D-2, §D-3). 노란 재질만 RNA 수준으로 고쳐도 pooled 기준 약 −3.3 dB가 남는다.
- **S3:** SPT-1은 명시적 relation 재구성으로 같은 편집에서 R = 0.82, W = −0.19 dB를 냈다(WL39, 기술적·seed 1). S2가 통과되면 이 편집은 주로 e와 C를 시험한다. 박스는 평행이동만 하므로 K_Lego는 변환되지 않는다. 감점 요인은 두 가지다. 학습 C의 근접(3.6/12 mm)·제3자 가림 영역이 학습 분포 밖이고, seed 편차가 0.23 dB다(WL38).
- **S4:** 새 장면에는 새 Stage A/B 데이터, 참조, RNA 상한이 필요하다. 프로젝트 이력상 새 영역마다 새 한계가 드러났다(WL28–29, WL33).
- **S5:** 공식 마감은 미공개다. 2026년 패턴(1월 23일)을 따르면 2026-10-23 기준 약 13주가 남는다. 첫 Lego 결과까지 5–8주로 추정되지만 근거가 약하다. 서버 실행은 12 GB GPU 하나에서 순차로 해야 한다. 따라서 사전 선언된 반복은 1–3회 정도다.

**확률을 움직이는 설계 변형(민감도; 추정자 산출, 권고가 아님):**

| 변형 | 효과(추정) |
|---|---|
| 학습 C_φ 대신 정확한 광선 기반 결합 | S3 약 0.5–0.55 → 0.65–0.70. S5 신규성은 약 0.05–0.1 하락("도메인별 8DNA + 경로추적 결합"에 가까워짐). 계약 §6의 학습 결합 질문은 답할 수 없게 됨 |
| 계약의 a=b 경로 분할 대 원안의 Σ_{b≠a} | 원안을 쓰면 S3가 약 0.1–0.2 하락. 하락폭은 박스가 가리는 Lego→Lego 성분의 크기에 달려 있고, 이 크기는 아직 측정되지 않음 |
| 3-스케일 커널(점-국소 sharp + banded 근거리 SSS + 비음수 저랭크 원거리) | S2 약 +0.08–0.12. 단일 전역 P·Q만 쓰면 S2 약 0.2 이하 |
| delta-sun 직접광을 픽셀 단위로 정확히 분리 | S2 약 +0.08–0.10 |
| Cycles 의미론에 맞춘 escape-path Stage A | S1 약 +0.08, S2 약 ×1.3. Cycles와의 정합 확인 없이 Mitsuba로 근사하면 S2 약 0.1까지 하락 |
| 비음수 인자 + 채널별 스펙트럴 반경 제약 | S1 약 +0.05. 수렴이 안정됨 |
| K로 도메인별 8DNA형 커널 재사용 | S1·S2 상승(S2 +0.05–0.1), 일정 단축. 신규성은 C와 도메인 분할로만 좁혀짐 |
| 사전 측정 게이트(SSS falloff, 가림 성분 분리, rank spectrum 등) | 성공 확률은 거의 변화 없음(±0.03). 실패 비용이 서버 수일에서 수시간으로 줄고 실패 귀속이 가능해짐 |
| 마감 대상을 SIGGRAPH Asia 2027 / TOG로 | S5 약 2배 |

## D. 성공을 좌우하는 기술적 쟁점 (검증된 것)

**D-1. 원안의 K/C 경계가 기존 편집과 모순된다 (critical, upheld).**
- 원안은 self-interreflection과 internal multi-bounce를 K에 넣고 결합을 Σ_{b≠a}로 둔다.
- WL37 편집에서 박스는 들어 올린 버킷 아래 자유공간에 있다. T0에서 판 위 12 mm, T1에서 3.6 mm다.
- 그래서 버킷 하면↔바닥판 경로가 "재사용 가능한" K_Lego에 고정된다. K가 배치 의존적이 되고, 정지 수신부의 stale 오차를 그대로 재현한다.
- 저장소 계약(2f483a3)은 이 경로를 C로 옮겨(a=b) 문제를 고쳤다. 그러나 이 변경이 원안 및 WL39 §F("possibly near-field self-interreflection")와 다르다는 점은 어디에도 기록되지 않았다.
- **새 문제:** "외부 편집이 침투할 수 없는 보호 영역"이 어떤 편집군 기준인지 규칙이 없다(최소 가림체·틈 크기 P(E) 미정). 그래서 K는 암묵적인 박스 크기 편집군에 대해서만 intrinsic하다. 더 작은 가림체를 허용하면 재분할과 Stage A 재학습이 필요하다.
- Lego의 Displacement는 전부 BUMP 전용이다. 따라서 "보호된 미세형상"은 이 asset에서 실체가 없다.
- 분리 측정 방법: 수정된 falsifier를 쓴다. Cycles에서 박스를 카메라에만 보이지 않게 하면 직접광 그림자만 막히고 간접광 차폐는 분리되지 않는다. 따라서 경로 태깅이나 맞춤 적분기가 필요하다.

**D-2. 노란 플라스틱의 렌더러 의미론 (critical, weakened → 교정 주장으로 유지).**
- 확인된 사실(`lego_b4_T0_s0.blend` headless, 파일 버전 3.5.10; `official_blend_inspect.json`):
  - Yellow는 98% SSS + 2% Glossy다.
  - SSS falloff = **Christensen-Burley**, Radius (1.0, 0.2, 0.1), Scale 1.0, IOR 1.4.
  - Lego는 4.2 M 삼각형, 1,872개 연결 부품을 가진 **단일 Blender mesh object**다(크기 0.89 × 1.60 × 0.94).
  - 큰 노란 부품은 2V/A ≈ 7–8 mm인 속 빈 껍질이다.
- Cycles의 Burley 디스크 투영은 같은 object 안에서만 샘플링한다. 따라서 참조 이미지의 "내부" 노란 SSS는 부품 사이 공기 틈을 건너뛰고, 박스로 가려지지 않는다.
- 이 전달은 FNTO 분할에서 들어갈 곳이 없다. 한 연결 매질 안에 갇힌 K도 아니고, 자유공간 C도 아니다.
- 도메인을 분리해 렌더하는 Stage A(Lego를 객체로 쪼개거나 이웃 제거)는 Burley 결과 자체를 바꾼다. 그러면 Stage A의 K 지도 신호가 T0/T1 참조와 어긋난다.
- 노란 재질은 홀드아웃 T0 픽셀의 41.4%를 차지하고, SPT-1 pooled 제곱오차의 91%를 차지한다(MSE 70.5×, `results/spt_lego/eval/diag_material.json`). 그래서 이 쟁점 하나가 S2를 거의 결정한다.
- **미측정:** 그림자 쪽 노란 radiance 중 부품 간 Burley hop에 의존하는 비율.

**D-3. 정적 게이트 산술과 커널 형태.**
- 노란 재질만 RNA 수준으로 고치면 pooled 약 30.2 dB, 즉 −3.3 dB다.
- −2 dB를 통과하려면 원인 미규명인 반사·박스 재질의 1.4–3.8× 격차도 함께 줄여야 한다. 이 격차의 일부는 빠진 SSS 출사광의 하류 효과일 수 있어서, 이 산술은 비관적 경계다.
- **커널 형태:** 원안의 두 갈래(점-국소 sharp + 전역 저랭크 P·Q)는 노란 SSS 커널에 효율적으로 맞지 않는다. 노란 SSS는 적·녹 채널에서 넓고, 얇은 벽을 통과하는 국소 피크가 있다. 전역 저랭크는 banded 커널에 비효율적이다(rank가 면적/커널 지지 면적에 비례).
- 투명·Red_Glass의 굴절 투과는 날카로우면서 비국소적이라 어느 갈래에도 맞지 않는다. 다만 이 장면에서 픽셀의 0.64%, 오차의 약 0.6%에 불과하다.
- 저장소 계약에서는 굴절이 K^sharp("same-interface")와 K^internal("transmission/refraction") 양쪽에 걸쳐 있다. 경로 분할은 구현에 위임되어 있다. 또 AGENTS.md("non-overlapping")와 계약("disjoint or explicitly weighted")이 서로 다르다.

**D-4. Stage A 지도 데이터 경로가 없다 (critical, weakened).**
- RNA 생성기(`blender_render_dataset.py` 209–235행)는 단일 sun, `device = "CPU"`("GPU not supported"), `use_light_tree = False` 구성이다. 공간적으로 국소화된 프로브를 만들 수 없다.
- **대안 경로**와 문제점:
  - (a) 패턴 렌더 다수: 비용 미측정
  - (b) 커스텀 Cycles fork 재빌드
  - (c) 맞춤 escape-path 추적기: Burley SSS, 거친 유리, bump/scratch 노드 그룹을 재현해야 한다
  - (d) Mitsuba volpath 근사: 의미론 불일치가 K의 실패로 잘못 귀속될 수 있다
- **escape-path(adjoint) 추정기**(upheld)의 원리: 출력점 (x_o, ω_o)에서 물체 안으로 추적해 빠져나가는 (x_i, ω_i)를 커널 샘플로 쓴다. 학습 시점에 임의 입사장 L_i에 대해 weight × L_i(x_i, ω_i)로 타깃을 만든다. 이 방식은 정확하고 자극 없이 동작한다.
  - 단서: 매끄러운 유전체를 통과하는 delta 광원, 손실 선택, 추적기 의미론 일치.
- 식별 가능성: 장면 단위 T0 RNA 데이터셋(픽셀별 sun 방향)만으로는 K를 결합과 분리할 수 없다. 선형성은 자극된 입력 span 밖으로 K를 확장해 주지 않는다. 계약 §5도 이를 인정한다.
- Stage A는 엄밀히 말해 식별 문제라기보다 압축 충실도 문제다. Cycles 재질 그래프와 기하가 K를 원리적으로 결정하기 때문이다. 따라서 실패 시 재질 의미론에서 직접 샘플한 커널과 비교해 "표현 한계"와 "자극 부족"을 구분할 수 있다.

**D-5. 학습 C_φ 대 정확한 결합 (major, weakened).**
- 자유공간 비행은 광선 투사로 정확히 계산 가능한 기하 적분이다(visibility, Jacobian, frame 변환).
- 쌍(pair) 가시성은 제3자와 오목한 Lego의 자기 가림에 의존한다. 따라서 결국 ray/visibility 질의가 필요하다. 이는 계약 §6의 "implicit dense ray/path tracing rebuild 금지"와 긴장 관계에 있는데, 허용되는 질의 예산이 정의되지 않았다.
- 학습 C의 학습 분포 밖 위험 영역: 근접 영역(form factor가 거의 특이), 제3자 가림.
- **추가 결함:** C의 지도 신호는 학습된 U/V basis에 의존한다. 따라서 K나 U/V를 재학습하면 C도 무효가 된다.
- 학습 C는 같은 basis, 같은 광선 예산의 MC 광선 결합보다 홀드아웃 T1에서 나을 때만 정당화된다.
- SPT-1의 R = 0.82는 생각보다 약한 증거다. 자기 재학습 대비 값이고, 절대 changed-A MSE는 0.00404다. 이는 Surface-Owned A_full의 9.4×, RNA_T1의 6.7×에 해당한다.

**D-6. 수학적 허용성 (major, upheld).**
- 선형성만으로는 부족하다. 비음수성, 수동성(채널별 ρ(KC) < 1), 상반성이 보장되지 않는다.
- 부호 있는 저랭크 P·Q는 음수 응답을 낼 수 있다.
- 계약의 선형성은 비음수 α, β에 대해서만 쓰여 있다. 이는 원뿔(cone) 선형성이다. relu/softplus 출력단도 이 조건을 통과할 수 있다.
- SPT-1의 연산자는 실제로 relu(Σ t⊙g)였다(`spt_model.py` 79행). 따라서 가법 선형이 아니었고, FNTO의 선형성은 실제로 새로운 변화다.
- 반사율이 높은 공동(투명 유리 0.973, glossy 0.8, Yellow 적 채널)에서는 물리적 ρ가 이미 1에 가깝다. a=b 자기 루프는 대각 이득을 더한다.
- 참조는 max_bounces 128을 쓴다. 따라서 잔차로 종료하는 반복은 고차 참조 대비 오차를 보고해야 한다.

**D-7. delta sun (major, upheld).**
- B4 렌더는 반경 0인 sun을 쓴다. 날카로운 그림자 경계를 port 계수로 투영하면 경계가 뭉개진다.
- 픽셀 단위 정확한 직접광 항(RNA의 visibility interface와 동형)과, e_a의 투영을 받는 비국소 항을 명시적으로 분리해야 한다.
- 계약은 SPT-1식 "수신 픽셀 직접광 visibility로 내부 응답 전체를 gating"하는 방식을 금지한다(그림자 쪽 SSS 소실 방지). 그러나 직접광을 어떻게 평가하는지는 명시하지 않는다.

**D-8. Lego가 시험하는 것과 시험하지 못하는 것.**
- **교정:** 2026-10-23 채팅 보고는 "Lego는 K^sharp를 시험하지 못한다"고 했다. 이 주장은 반박 검증에서 **refuted**되었다.
  - Glossy roughness 0.3은 GGX α ≈ 0.09다. 이는 Phong 지수 약 245에 해당하며, SH2가 97% 실패한 Phong-200 사례와 같은 날카로움이다.
  - 혼합 가중치는 2–5%지만 하이라이트 방향의 피크는 Gray/Sand에서 확산 lobe의 2–3배다. Black은 광택 지배적이다(확산색 0.0028).
  - FNTO에서는 delta-sun 하이라이트가 K^sharp를 통과한다. 그래서 Lego의 모든 lit 광택 픽셀이 K^sharp를 시험한다.
- **교정된 주장:**
  - Lego는 직접광 하이라이트에 대해 K^sharp를 시험한다.
  - near-mirror 반사, 날카로운 투과(유리 α ≈ 0.64), 에너지가 작은 물체 간 정반사는 시험하지 못한다.
- T1은 박스의 순수 +0.44 m 평행이동이고, T1b도 Lego 평행이동이다. 회전이 없다. 따라서 K의 강체 회전 공변성과 SSS·유리 도메인의 이동은 첫 실장면 편집에서 시험되지 않는다.
- 학습 방향 사전(dictionary)은 SH와 달리 회전에 닫혀 있지 않다. 회전하면 aliasing이 생긴다.

**D-9. 비용과 메모리.**
- 상태로서의 adaptive lobe는, port별 사전을 고정하면 선형성과 양립한다. 저장소 계약 §3.1/§3.2가 사실상 이렇게 정한다.
- 대신 방향 상태 메모리가 구속 조건이 된다.
  - SPT-1의 multibounce cache는 [Na, 512, 3] fp16, 약 568 MB였다.
  - D = 64 계수로 광원 bin마다 상태를 캐시하면 광원당 약 142 MB, 512 bin이면 약 73 GB다. 12/16 GB GPU에서는 불가능하다.
- SPT-1 학습은 약 4.9 h/run, 약 8.3 GB였다. 동시 실행에서는 OOM이 났다(amendment 1). 따라서 FNTO 전체 실행은 순차로만 가능하다.
- 편집당 갱신 비용: 명시적 gather를 쓰면 SPT-1의 약 7 s(대부분 CPU KD-tree mapping)보다 커질 가능성이 높다. 측정된 값은 아니다.

## E. 선행 연구와 신규성

검토자가 웹으로 확인한 문헌만 적는다.

- **8DNA** (Wu, Lu, Xu, Hašan, Ramamoorthi; SIGGRAPH 2026, arXiv 2604.25129)
  - 근거리 8D 전달을 학습한다. 입사점·방향에서 출사점·방향으로 가며, SSS와 광택 상호반사를 포함한다. 런타임 경로추적으로 asset 사이를 결합한다.
  - FNTO의 K_a[L_i]에 가장 가까운 선행 연구다. 이 프로젝트가 WL19–30에서 baseline으로 썼지만 FNTO 계약 §8에는 없다.
  - WL25 기록상 8DNA 논문은 asset 내부 부품 간 전달의 해법으로 convex decomposition을 언급한다. 이는 FNTO 도메인 분할의 부분적 선례다.
- **b = K[e + ΣCb] 구조의 선례:** 결정 8의 우선 시기(2007–2013) 전후. 어느 것도 저장소 문서에 인용되지 않았다.
  - Precomputed Shadow Fields (Zhou et al., SIGGRAPH 2005)
  - Precomputed Radiance Transfer Field for Rendering Interreflections in Dynamic Scenes (Pan et al., CGF 26(3), EG 2007). 배치에 독립적인 물체별 RTF, diffuse+glossy, CPCA.
  - Iwasaki et al. (EGSR 2007). 강체를 2차 광원으로 다룬다.
  - Modular Radiance Transfer (Loos et al., TOG 30(6), SIGGRAPH Asia 2011, DOI 10.1145/2024156.2024212). 모듈별 사전계산 전달을 interface로 런타임 결합한다. 다만 diffuse·저주파만 다룬다.
  - Delta Radiance Transfer (Loos et al., I3D 2012)
  - Light Field Transfer (Cossairt, Nayar, Ramamoorthi, TOG 27(3) 2008). interface 4D light field를 반복한다.
- **저랭크+희소 전달 행렬:** symmetric photography (Garg et al. 2006), Kernel Nyström (Wang et al. 2009), Direct-to-Indirect Transfer (Hašan et al. 2006), PRT (Sloan et al. 2002). Adaptive lobe의 선례로 spherical Gaussians (Wang et al. 2009).
- **신경 객체 합성:** OSF (Guo et al.; "objects or lights move, without retraining"), Yu et al. TMLR 2023, NeLT (TOG 2023), Neural PRT (Rainer et al., CGF 2022), Neural SSS (CGF 2024).
- **RNA 서술 검증:**
  - 원안의 "RNA 논문이 SSS와 복잡한 multiple scattering 학습을 핵심 기여로 제시"는 대체로 맞다. 논문은 subsurface scattering을 목표 shader에 포함하고 §6.2에서 Subsurface Lego를 평가한다.
  - 다만 RNA도 shadow/indirect 광선으로 asset 간 MC 전달을 이미 합성한다. 따라서 "RNA를 unit으로 나눈 것이 아니다"라는 주장은 RNA 자체의 "asset별 + MC" 배치 모델에 대해 논증해야 한다.
- **저장소 계약 §8의 인용 5건:** 모두 실재하고 귀속도 정확하다(RNA DOI 10.1145/3695866, CGF 10.1111/cgf.15158, CGF 10.1111/cgf.70348, arXiv 2509.18497 = TOG 10.1145/3763305, NLT). 문제는 가장 가까운 선례가 빠졌다는 점이다.

**살아남을 수 있는 신규성:** 조합이다.
- (a) RNA급 재질 의미론(SSS·유리)을 갖춘 intrinsic 커널
- (b) relation 조건부 학습 결합
- (c) 편집 도달성 기반 도메인 분할(동일 asset 자유공간 경로 a=b)
- (d) 선택적 무효화 회계

(b)와 (c)에는 부분 선례가 있다(TransGI, RTF, Iwasaki, 8DNA convex decomposition). 리뷰어를 설득하려면 "도메인별 8DNA/RNA + 경로추적 결합" baseline을 동일 품질에서 비용으로 이겨야 한다. 이 baseline도 재학습이 필요 없고, 아직 저장소에 없다.

## F. 원안의 SPT-1 서술 감사 (WL39 대조)

| 원안 주장 | 판정 | 근거 |
|---|---|---|
| SSS·transmission 채널 부재가 실패 원인 | 정확 | WL39 §C: Yellow 70×, transparent 3.9×, Red_Glass 4.5×, 그림자 픽셀 17.7 vs 32.4 dB |
| geometry-dependent incident state 분리에 "성공" | 과장 | WL39가 "operational"이라 했으나 기술적 결과, seed 1, 자기 재학습(약 22 dB) 대비 R이다. 전체 view-mean은 C_upd 21.98 dB가 D_stale 22.06 dB보다 약간 나빴다. 출력 영향도 국소화되지 않았다 |
| self-interreflection 처리 부족 | 미확인 | WL39 §F는 "possibly"로만 언급했다. SPT-1 relation은 동일 object 내 hit도 포함했다(`spt_relations.py`, hit_obj ∈ {0,1}) |
| 중요한 전달을 명시적 ray gathering으로 옮기고 나머지를 저주파로 근사 | 부분적 | gather는 기하만 운반했다. 직접광 head d(φ,n,ω_o,ω_L)는 full-frequency이고 dataset visibility로 gating됐다. SH2는 간접 상태에만 적용됐다 |
| SPT-1 + SSS branch가 아니다 | 부분적으로 수사적 | FNTO는 SPT-1의 골격(persistent field·operator + geometry 재계산 relation + 반복 bounce)을 유지한다. 대표 추가 요소는 WL39 §F 열린 질문 그대로다. 금지된 "rescue"는 아니다(WL39 판정 불변, 새 프로토콜 사전 선언 시). 다만 Batch 6 해당 여부에 사용자 결정이 필요하다. 동시에 바뀌는 요인이 약 7개라서 단일 요인 귀속이 불가능하다 |

## G. 커밋된 living doc의 정합성 (5cf1680..d054082)

- **권한 표기 불일치.** AGENTS.md 신규 블록은 "the user-approved specified architecture"라고 적었다. 계약 문서, Architecture §0, RCT, Roadmap은 "assistant-authored at user request"라고 적었다. RCT §2 "User Decisions in Force"와 Roadmap의 날짜별 결정 목록에는 FNTO 항목이 없다. 그래서 결정 3(Surface-Owned가 active hypothesis)이 형식상 그대로다. 커밋 author "NaDan"은 사용자와 같은 GitHub 계정(noreply)이므로 작성자 상이의 증거는 아니다.
- **연구 정의의 이중화.** RCT §1/§9, Architecture §1/§9, BASELINE §1은 여전히 complete-unit 재계산을 정의로 둔다. FNTO의 "학습 파라미터 갱신 0" 질문이 이를 대체하는지, 일반화하는지, 분기하는지 아무 문서도 말하지 않는다. 결정 6("≥ 5 batch 후 viability 판단")의 계산 방식도 미정이다. 상태 줄도 오래됐다(Architecture "Batches 1–3 of ≥ 5", RCT "Batches 1–4 of ≥ 5").
- **Batch 번호와 게이트.** AGENTS.md에 "Batch 6 is not authorised"가 남아 있고, FNTO 첫 구현이 Batch 6인지 정해지지 않았다. 계약 §9는 K 게이트, C 연결, 홀드아웃 편집을 "한 조사"로 묶고 세분화를 피하라고 한다. 이는 중단 게이트와 자동 진행 금지 규칙과 긴장 관계에 있다.
- **비정량 게이트.** admissibility 게이트는 "information path", "meaningful response"로 쓰여 있고 지표·임계값이 없다. 정적 목표도 "approximately 1 dB on a qualified regime"이며 "Suggested targets, not evidence"로 표기돼 있다. SPT-1은 합성 계약 테스트 5/5를 통과한 뒤 −11.3 dB로 탈락했다.
- **소유 의미론 미조정.** Architecture §2 Unit Contract는 "mandatory for every candidate"다. 그런데 FNTO에는 파라미터 소유 표가 없다. 공유 C_φ가 "frozen forever"인지 coupling인지, 부적합할 때의 의미론이 무엇인지도 없다. Stage B의 "train or calibrate C_φ" 때문에, 사후 C 조정이 "zero-update" 주장의 실패가 아니라 "calibration"으로 보고될 여지가 있다.
- **비목표와의 충돌.** FNTO §3.3/§4는 알고리즘 후보 인덱스와 의존성 유효 상태 재사용을 포함한다. 그런데 Architecture §8, §3a, AGENTS는 이를 별도 승인 전 비목표로 둔다. 첫 batch가 어떤 relation 구성 방식을 쓰는지(전수 재구성 대 희소 후보) 명시되지 않았다.
- **SPT-1 재사용 경계.** AGENTS는 "SPT-1+SSS patch"를 금지하고, 계약 §6은 SPT-1을 "diagnostic only, NOT starting implementation"으로 둔다. 그러나 relation builder, 전파, canonical field 중 무엇을 재사용해도 patch가 아닌지 정의되지 않았다.
- **잔존 용어.**
  - 계약 §7 "unchanged transport-closed domains"는 §2와 모순된다.
  - §3.1 "An object domain", §5 "object-local operator supervision"
  - Architecture §0 "C_ab is cross-domain free flight"는 바로 앞의 a=b와 모순된다.
  - Roadmap §3b가 §3a 앞에 있다.
  - RCT §7과 Architecture §8 frontier 절에 FNTO 반영이 없다.
- **커밋 형태.** d0ecc2d..d054082 다섯 개가 같은 메시지로 3초 간격, 파일 하나씩 바꿨다. 커밋 12개 모두 본문이 비어 있다. 그래서 self-interreflection을 K에서 C로 옮긴 근거가 기록에 없다. 중간 상태도 서로 모순된다(333e5b5에서는 전 문서가 K 포함, 2f483a3에서는 계약만 반대).
- **평가 영역.** Stage A는 새 참조 영역(고립 도메인 공간 프로브)이다. RNA_T0 33.57 dB(spt_v1 영역, train_T0 400장 × 1024 spp)와의 비교는 학습 신호 차이와 표현 한계가 섞인다. T0 장면 RGB로 C나 전역 스케일을 보정해도 되는지 정해지지 않았다. 보정을 허용하면 누설 대조 arm이 필요하다.
- **기존 평가 스택 재사용 가능 범위**(그대로 쓰면 영역 비교 가능):
  - 홀드아웃 T0/T1 16 views(1024², Cycles seeds 1/2 평균, 4096 spp)
  - WL37 영역(`eval/views.npz`)
  - view-mean/pooled PSNR, HDR logL1, hf
  - spt_v1 판정 형태(credible ≥ −1.0, weak [−2, −1), fail < −2.0)
  - F, R 정의
  - RNA_T0/T1, Surface-Owned, SPT-1 arm
- **비교 arm의 귀속력 부족.** 계약 §6 arm 4–6(stale/current relation, 오라클 입사장)만으로는 K, C, e, 투영, solver 오차를 분리할 수 없다. 오라클-C(같은 basis 위 정확한 광선 결합), 오라클-e, 고정 고반복 대조가 없다.

## H. 미확인 사항 (확률을 크게 움직이는 지점)

1. 그림자 쪽 노란 radiance 중 부품 간 Burley hop에 의존하는 비율, 그리고 도메인 분리 렌더가 이를 얼마나 바꾸는지
2. WL37 정지 수신부 변화 중 박스가 가린 Lego→Lego 반사광 성분의 크기
3. 반사·박스 재질 1.4–3.8× 격차가 독립적인지, 빠진 SSS 출사광의 하류 효과인지
4. 노란 도메인 연산자의 유효 rank·banding (커널 용량, 메모리, Stage A 비용 결정)
5. Stage A/B 데이터 비용(GPU-h, 저장 용량)과 Cycles 의미론 정합 가능성
6. 학습 C_φ의 홀드아웃 근접·가림 배치 일반화, 그리고 필요한 Stage B 학습 배치 수
7. 도메인별 8DNA/RNA + MC 결합이 비슷한 비용으로 FNTO 품질에 도달하는지(신규성 붕괴 여부)
8. SIGGRAPH 2027 공식 마감(미공개)과 FNTO 실행의 seed 편차

## I. 산출물과 출처

원본(로컬, `results/`는 gitignore 대상):

| 파일 | 내용 | sha256 |
|---|---|---|
| `results/fnto_review/wf1_proposal_review.json` | 원안 검토 workflow 전체 반환값(digest, 원 검토·검증, 두 추정) | `da73138789d41e5a38f913767c34c22b2bc48b283fc4e97cafb709e0d5ae4812` |
| `results/fnto_review/wf2_repo_contract_delta_review.json` | 저장소 계약 차이 검토 전체 반환값 | `aeab1589a4922453994a4987255dd1147066b7535bdf1311b43961a46ef98312` |
| `results/fnto_review/wf1_script.js` | 원안 검토 workflow 스크립트 | `140d7f5ff535247fb3bdfa37181103da8630f2887c3a555dd362b37cc54bee9e` |
| `results/fnto_review/wf2_script.js` | 차이 검토 workflow 스크립트 | `2ded95a5f15b61cd5b1c4fe9a40c2962b256de3f9e2e7f55c1facd0c556222fe` |

- 검토 시점 HEAD: d054082 (Selective_Recompute).
- 이 세션에서 living document는 수정하지 않았다.
- `results/evaluation/40/`은 만들지 않았다. 이미지 등 리뷰용 시각 산출물이 없기 때문이다.

## J. 열린 질문

이 검토는 아무 batch도 승인하거나 진행하지 않는다. 다음은 사용자 결정이 필요한 열린 질문이며, 권고가 아니다.

- 원안(K에 self-interreflection, Σ_{b≠a})과 저장소 계약(a=b) 중 어느 것이 규범인가.
- FNTO의 "학습 파라미터 갱신 0" 질문이 Selective_Recompute 연구 정의(결정 3)를 대체하는가, 일반화하는가, 별도 track인가.
- FNTO 첫 구현이 Batch 6인가. K 게이트와 C·편집 시험을 한 batch로 묶는가.
- §G의 문서 불일치(권한 표기, 잔존 용어, 누락된 선행 연구, frontier 절)를 정정할 것인가.

## APPENDIX A. 발견 사항 목록과 반박 검증 판정

각 항목의 주장 원문, 증거, 검증 사유는 §I의 JSON에 있다. refuted 항목은 확률 추정에서 제외됐다(wf1 cost-timeline-eval F3, 교정 내용은 §D-8).

### A.1 Workflow 1 (원안 검토): 관점별 발견 사항과 반박 검증 판정

**physics-math**

| id | severity | verdict | title |
|---|---|---|---|
| F1 | critical | upheld | The verbatim K/C split puts edit-blocked same-asset free-space paths into the 'reusable' K |
| F2 | major | weakened | Pairwise R_ab cannot express third-party occlusion, and free flight is not something that needs learning |
| F3 | major | weakened | Linearity is necessary but not enough: positivity, passivity (ρ(KC)<1) and reciprocity are unspecified, and signed low-rank P·Q breaks positivity |
| F4 | major | upheld | A delta sun with hard cast shadows needs an explicit split of e_a between an exact local term and a projected nonlocal term |
| F5 | major | upheld | The Lego is one mesh, and the reference's SSS sampling semantics decide what 'internal' means |
| F6 | major | weakened | Rigid-motion covariance holds only if all kernel features are object-frame, and canonical-frame incident directions must be excited in training |
| F7 | major | weakened | If FNTO works, learned-parameter dirtiness collapses to internal deformation, and D_T moves into dynamic state and coupling; this redefines the selective-recompute question |
| F8 | minor | weakened | The block-operator equation and the factored SSS kernel have direct 2005–2011 precedents that define the novelty boundary and offer tested design answers |

**learnability-data**

| id | severity | verdict | title |
|---|---|---|---|
| F1 | critical | weakened | The dominant Lego transport mode (yellow SSS) is a short-range banded kernel that fits neither proposed branch efficiently |
| F2 | critical | weakened | K_a is not identifiable from the scene-level T0 RNA dataset, and linearity does not extend it beyond the excited input span |
| F3 | major | upheld | The adjoint 'escape-path' estimator is correct and stimulus-free, with three specific caveats: delta lights through smooth dielectrics, loss choice, and tracer semantics |
| F4 | major | weakened | Refractive transmission through transparent / Red_Glass parts is sharp and nonlocal, so it fits neither branch, and whole-image PSNR cannot see it |
| F5 | major | weakened | Static-fidelity budget: fixing SSS alone leaves about −3 dB, and 'within ~1 dB of RNA' requires every material near RNA under a harder generalisation task |
| F6 | major | weakened | Probe-based Stage A data cost scales with effective operator rank unless probes exploit locality; the 8D concern is better framed per sub-block |
| F7 | major | weakened | Boundary-port coverage: SPT-1 anchors exist only where T0 cameras saw the surface; SSS entry faces and inner wall faces need full-mesh coverage |
| F8 | minor | upheld | The linearity contract is sound, but 'adaptive lobes' must adapt only to geometry and material, never to the incident field |
| F9 | minor | weakened | Prior art covers learned 8D BSSRDFs, port-based modular transfer and linear transfer operators; the novelty margin is the editable K/C path partition at RNA-class fidelity |

**prior-art**

| id | severity | verdict | title |
|---|---|---|---|
| F1 | critical | upheld | 8DNA already realises K_a[L_i] (near-field 8D per-asset transport incl. SSS) and is missing from the FNTO prior-art list |
| F2 | critical | upheld | The composition equation b_a = K_a[e_a + Σ C_ab b_b] is classic dynamic-scene precomputed transfer from the project's priority 2005-2013 era |
| F3 | major | weakened | The proposal and the repo contract disagree on where self-interreflection lives, which decides K validity for the Lego box edit |
| F4 | major | weakened | The low-rank P·Q internal kernel plus a sparse sharp part repeats 2003-2010 transport-matrix structure |
| F5 | major | weakened | Proposal's characterisation of RNA is partly overstated; RNA already composes assets with MC inter-asset transport |
| F6 | major | weakened | What novelty could survive: learned compressed coupling plus edit-reachable domain partition, judged by selective-invalidation accounting |
| F7 | major | weakened | Identifiability and data cost of K: precedent shows spatially resolved transport needs many structured excitations |
| F8 | minor | weakened | Sharp-directional kernel with adaptive lobes: not novel, and repo evidence warns about near-mirror bandwidth |
| F9 | minor | upheld | Neural object-centric composition is crowded prior art; 'per-object neural kernels + dynamic coupling' alone will not pass |

**project-fit-claims**

| id | severity | verdict | title |
|---|---|---|---|
| F1 | critical | weakened | Proposal's equation (self-interreflection inside K_a, coupling only for b != a) is already contradicted by the existing Lego edit; the living docs silently changed it |
| F2 | major | weakened | Governance: living docs already adopt FNTO as 'primary', with provenance and authorisation wording that conflicts with the user decision record |
| F3 | major | weakened | Audit of the proposal's factual statements about SPT-1 |
| F4 | major | weakened | FNTO reframes the user-confirmed Selective_Recompute thesis: learned state is reused and only non-learned state is recomputed |
| F5 | major | upheld | FNTO is a successor in SPT-1's operator-state family, not a new family; it bundles many simultaneous changes |
| F6 | major | weakened | Unit granularity: nonlocal internal kernels set a coarse ownership floor, conflicting with decision 2's finer units; sub-object edits have no update path |
| F7 | major | upheld | Regime comparability: which existing baselines and protocols carry over unchanged, and which FNTO choices break comparability |
| F8 | major | upheld | Prior-art gap: the b = K[e + sum C b] modular-transfer structure is well established in the 2007-2013 priority window but is absent from the docs |
| F9 | minor | upheld | Identifiability and offline cost are first-order unknowns; the claim that FNTO 'preserves RNA's strengths' is untested |

**cost-timeline-eval**

| id | severity | verdict | title |
|---|---|---|---|
| F1 | critical | weakened | Stage-A operator supervision is the dominant cost and schedule risk: RNA's per-pixel stimulus path is CPU-only and directional-only |
| F2 | critical | weakened | Adaptive lobes as state conflict with the linearity contract; a fixed per-port dictionary makes directional state memory the binding constraint |
| F3 | major | refuted | The Lego scene cannot test K^sharp; its reflective and glass materials are rough |
| F4 | major | weakened | Yellow SSS is nearly whole-object for red, and the single-mesh Lego makes domain isolation semantically risky |
| F5 | major | weakened | A decisive minimal evaluation can reuse the WL37/39 stack, but R needs an affordable reference and the gates must be ordered |
| F6 | major | weakened | Per-edit update cost: explicit gathering scales with D and iteration count, and learned C_phi does not remove the visibility rays but adds multi-configuration data |
| F7 | major | weakened | Training cost and the 12 GB server limit allow only serial full runs |
| F8 | major | weakened | Engineering estimate: about 5-8 weeks to a first full Lego result, with clear reuse boundaries |
| F9 | major | weakened | Timeline: about 1-3 full design iterations fit before the assumed deadline, with single-scene, single-edit evidence |
| F10 | minor | upheld | Missing 2007-2013 prior art with direct design and cost relevance: Modular Radiance Transfer and locally low-dimensional transport |

### A.2 Workflow 2 (저장소 계약 차이 검토): 관점별 발견 사항과 반박 검증 판정

**domain-partition-physics**

| id | severity | verdict | title |
|---|---|---|---|
| F1 | critical | weakened | K's content depends on the allowed edit family, and the partition rule names the held-out edit's occluder |
| F2 | critical | weakened | Yellow plastic is thin-walled and optically thin, so much of the 41.6% yellow appearance moves from K into C |
| F3 | major | weakened | Exposed same-asset interreflection in C is the channel type SPT-1 already had, and its reflective-material gap is unexplained |
| F4 | major | weakened | The K^sharp / K^internal / C path partition is incomplete, and responsibility for it is pushed to the implementation |
| F5 | major | upheld | Positivity, passivity and convergence of the coupled iteration are unconstrained, and the a=b self-loops add diagonal gain |
| F6 | major | upheld | Rigid-motion covariance of K is not exercised by the planned first real-scene edit |
| F7 | minor | weakened | Domain and port count on the Lego is unspecified and likely large |

**learning-coupling**

| id | severity | verdict | title |
|---|---|---|---|
| F1 | critical | weakened | Stage A has no defined stimulus family, probe budget or renderer; forward probes cannot enforce the K/C path partition, while an escape-path estimator could |
| F2 | major | weakened | Stage B: the learned C_φ duplicates an exact ray-cast integral, the 'dense rebuild' ban is unquantified, and SPT-1's R = 0.82 is weaker evidence than it looks |
| F3 | major | upheld | The admissibility gate and the static target are not quantitative or predeclarable as written |
| F4 | major | weakened | Whether T0 scene RGB may train K or C is unresolved, and that choice decides both leakage and the meaning of the RNA comparison |
| F5 | major | upheld | The comparison arms cannot separate K error from C, e, projection and solver error, and the reused edit barely exercises K's SSS and glass response |
| F6 | minor | weakened | The linearity/superposition check is tautological for a bilinear K, conflicts with positivity enforcement, and port-excitation coverage has no metric |
| F7 | minor | weakened | K is fully determined by known material graphs, so Stage A tests compression fidelity rather than identifiability; the closest 2011 precedent is not in §8 |

**doc-consistency-governance**

| id | severity | verdict | title |
|---|---|---|---|
| F1 | critical | weakened | Authority label 'user-approved' is unsupported, and the decision is recorded as an assistant act with no user-decision entry |
| F2 | major | weakened | AGENTS.md now holds contradictory directives on architecture switching, negative-result rescue and the 'SPT-1 patch' boundary |
| F3 | major | weakened | The paper-level research question changed silently: the living docs' definitions still say complete-state recompute of owned units, FNTO targets zero learned updates |
| F4 | major | weakened | FNTO's ownership semantics are not reconciled with the 'mandatory for every candidate' Unit Contract (shared C_phi, possibly shared K^sharp, RNA-semantics confound) |
| F5 | major | weakened | FNTO's deployable update needs candidate discovery and dependency tracking, which Architecture §8 and AGENTS still forbid without a separate authorisation |
| F6 | minor | weakened | The rapid 2f483a3 reframing left contradictory domain terminology and an edit-relative domain definition |
| F7 | minor | upheld | Mechanical commit problems: identical-message per-file commits, inconsistent intermediate states, no rationale, and a WL39 attribution overstated |
| F8 | minor | upheld | §8 references verify, but the closest component-wise transport prior art (decision-8 window) is absent |

