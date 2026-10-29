# WL57 학습 실행 효율 개선안

2026-10-29. 완료된 WL57 코드와 측정값을 바탕으로 작성한 후속 실행 설계다. 이 문서는 새 학습 실험 실행, 모델 변경, WL58 시작 또는 기존 결과 교체를 뜻하지 않는다. 아래 속도 효과는 아직 측정하지 않았다.

## 확인한 현재 상태

- 소스: `r57_train.py`, `r57_model.py`, `r57_data.py`, `r57_geometry.py`. 기준 HEAD `29fd693`.
- 환경: LabServer69, RTX 3090 24 GiB, PyTorch 2.3.1+cu121. FP32, TF32 비활성화.
- 완료된 6,000-step 학습: D 약329초/seed, R 약421초/seed. 검증과 저장을 포함한 평균 약55/70 ms per step.
- Torch peak allocated: D 3,348,082,176 bytes, R 4,603,326,464 bytes. 이는 전체 장치 메모리 점유나 reserved 메모리와 다른 값이다.
- 그래프의 뒤쪽 지속적인 높은 사용률 구간이 실제 네 모델 학습에 해당한다. 앞쪽에는 oracle 생성 등 다른 작업이 섞여 있다. 약04시 이후 idle은 chain 완료와 부합한다. GPU 사용률 그래프 하나로 학습 병목이나 SM 효율을 확정할 수는 없다.
- 기존 A3의 약11.6배 개선은 oracle 생성 측정이다. 학습 속도 개선치로 전용할 수 없다.

## 1순위: 동기화와 반복 전송 제거

`aggregate()`는 coarse 출력 크기를 정하려고 `int(parent.max())`를 실행한다. 현재 loss 구성에는 정상 microbatch=2 기준 step당 coarse 호출이 7회 있다. GPU scalar를 Python 정수로 읽으므로 CPU가 GPU 결과를 기다린다. coarse port 수는 geometry 생성 때 CPU metadata로 고정한다.

같은 함수가 호출될 때마다 동일한 64→16 bin index를 생성하고 GPU로 옮긴다. step당 aggregate 호출은 11회다. bin mapping을 한 번 GPU에 보관하고, prediction의 fine16 합을 한 번 구해 coarse 합과 여러 loss 항에서 재사용한다. uniform parent 구조가 확인된 경로만 reshape/reduction 후보로 검증하고, 일반 correspondence에는 보존적인 index mapping을 유지한다.

`float(L)`, `float(delta)`, `float(loss)`, `float(clip_grad_norm_(...))`도 매 step 동기화한다. 실제 기록은 250 step 간격이므로, 기록할 step에서만 detached scalar를 묶어 CPU로 읽는다. 비유한 loss를 optimizer 이전에 차단하는 안전 검사는 유지한다. gradient clipping 연산 자체도 유지한다.

입력 lights와 targets는 CPU에 남아 step마다 세 번 `.to('cuda')`를 실행한다. batch=2에서 x/current target/rest target 합계 약13.27 MB/step, 모델당6,000 step에 약79.6 GB의 논리적 입력 전송이다. geometry는 기본 경로에서 이미 GPU에 상주하므로 geometry 상주를 새 개선으로 계산하지 않는다.

현재 active train/validation target은 1,312 fields다. 원본64-bin FP32 target 전체가 약3.87 GB, loss가 사용하는16-bin 합은 약0.967 GB다. train/validation lights 전체는 약94 MB다. 먼저 GPU-resident target/input indexing으로 반복 H2D를 제거하고, 검증된16-bin target 캐시로 저장량도 줄인다. held-out geometry/light/target은 optimizer 캐시에 넣지 않는다. 캐시 identity에 데이터·geometry·mapping 버전을 포함한다.

## 2순위: 여유 VRAM으로 연산 묶음 확대

geometry당 3,840 ports ×128 neighbors =491,520 edges를 기본65,536개씩 8회 나눠 처리한다. coefficient MLP와 적용 scatter가 각각 Python loop를 돌며, current/rest geometry에 반복된다.

131,072 /262,144 /전체491,520 edge의 세 후보만 짧게 비교한다. 더 큰 GEMM과 적은 kernel launch를 기대할 수 있지만, 개선 여부는 실측한다. chunking해도 backward를 위한 전체 edge activation은 유지되므로, chunk 확대가 전체 activation을 단순8배 늘리는 것은 아니다. 대신 순간 temporary/작업공간 피크를 측정해야 한다.

source-major 128-edge 구조가 검증되면 coefficient column normalization의 source `index_add`를 `(P,128,M,RGB).sum(1)`으로 바꾸고 scale도 broadcast한다. 모든 edge와 물리적 column mass를 그대로 보존한다. outgoing destination scatter는 중복 receiver가 있어 별도 문제다. 이것이 주요 병목으로 측정될 때만 정렬된 receiver reduction을 검토한다.

OOM fallback은 먼저 검증된 작은 edge chunk로 돌아간다. 기존 batch=2/microbatch=1 gradient accumulation fallback은 마지막 수단으로 보존한다. fallback 선택과 피크 메모리, 수치 차이를 기록하며 OOM 이후 부분 결과를 정상 결과로 취급하지 않는다. 처음부터 메모리를 절약하려고 activation checkpointing을 켜지는 않는다.

## 3순위: 학습 중 변하지 않는 계산 캐시

- D의 raw geometry 1-hop/2-hop/global mean context는 geometry만으로 정해진다. pose별 한 번 계산한다. learned affine projection은 매 optimizer step 다시 계산한다.
- R의 learned state와 D/R learned coefficients는 weight가 바뀌므로 step 사이에 재사용할 수 없다.
- 고정 geometry와 고정 training light의 ballistic `B x`에는 학습 가능한 parameter가 없다. 학습용으로 미리 계산할 수 있다. 새 light에 대한 inference에서는 여전히 현재 B 적용이 필요하다.
- loss는64→16 outgoing 합을 사용한다. softmax로 정규화한64-bin O를16-bin으로 합쳐 learned 출력을 직접16-bin으로 만들면 큰64-bin 출력의 생성과 재집계를 생략할 수 있다.64 logits와 normalization은 유지하고 full64 평가 경로는 유지한다.
- ballistic16과 target16을 residual target으로 합칠 수도 있다. 이때 loss 분모는 기존 TOTAL target norm을 사용해야 한다. delta 항 역시 total target difference와 rest norm의 원래 분모를 유지한다. residual norm으로 바꾸면 다른 학습 목적이 된다.

이 캐시는 학습용 고정 입력 최적화다. production geometry-only K cache나 임의 light inference 성능 개선으로 주장하지 않는다. cache construction 시간과 memory를 별도 계상한다.

## 4순위: geometry batch와 launch overhead

current/rest 두 geometry를 geometry 축으로 묶으면 같은 light batch=2와 paired objective를 유지하면서 연산을 키울 수 있다. global mean, softmax, column normalization은 geometry별로 분리해야 한다. 두 도메인을 섞어 정규화하면 다른 모델이다.

먼저 dense edge/node 부분에 국소 `torch.compile`을 검토한다. scalar extraction과 sparse 경로 문제를 해결한 뒤에도 CPU launch overhead가 크면 고정 buffer의 CUDA Graph를 검토한다. pinned 2.3.1 환경에서 실제 지원과 compile 비용을 측정하며 라이브러리를 임의 업그레이드하지 않는다. graph capture용 geometry별 buffer를 무제한 복제하지 않는다. cosine LR, clipping, nonfinite guard, checkpoint resume 의미를 유지한다.

독립 seed 학습 동시 실행은 단일 학습을 최적화한 뒤 전체 네 학습의 총 wall time이 줄어드는 경우에만 선택한다. 이미 GPU가 지속적으로 바쁘면 경합으로 느려질 수 있다. 공정한 모델별 inference cost 측정은 기존처럼 단독 실행한다.

## 별도 학습·정밀도 판단이 필요한 항목

effective light batch를2→8/16으로 늘리면 gradient와 sample/update schedule이 달라진다. 이를 기존6,000-step 실험과 동일한 실행 최적화로 취급하지 않는다. AMP/TF32도 수치 계약이 바뀌므로 첫 적용 대상에서 제외한다. 숨은 폭, rank, neighbor 수를 성능 명목으로 바꾸지 않는다.

## 구현 전에 고정할 짧은 검증 절차

1. D/R의 고정 초기 weight와 선언된 train/validation 입력만 사용한다. 기존 batch2, loss 가중치·분모, FP32, TF32 off, clipping, optimizer, seed/RNG 순서를 유지한다.
2. 최적화 하나씩 forward/loss/전체 parameter gradient/한 optimizer update를 비교한다. 제안 시작 gate는 상대 loss 차이1e-6, gradient relative L2 차이1e-4이며 near-zero 성분의 absolute 차이도 함께 기록한다. 이는 계획상의 수치 검증 기준이며 기존 과학적 acceptance threshold를 대체하지 않는다. reduction 순서가 바뀌므로 bitwise 동일함을 가정하지 않는다.
3. zero, linearity, nonnegative, passivity, boundary mapping 회귀를 기존 threshold로 통과시킨다. NaN/Inf 및 OOM fallback, resume도 별도로 확인한다.
4. 20 warm-up +100–200 timed steps의 제한된 진단으로 각 후보를 비교한다. validation/save 포함 비용과 순수 step 비용을 둘 다 잰다. compile/cache 초기 비용도 포함한6,000-step 예상 총비용은 측정치 기반 추정으로 표시한다.
5. CPU wall time과 CUDA Events를 이용해 H2D, geometry/coefficient, apply, loss, backward, optimizer, validation, save 시간을 분리한다. CUDA timing은 적절한 경계 동기화를 사용하고 매 연산마다 강제 synchronize하지 않는다. 혼합 DrJit/OptiX 프로세스의 기존 CUPTI profiler crash를 재사용하지 말고 PyTorch-only 진단부터 시작한다.
6. steps/s, 전체 작업 wall time, allocated/reserved VRAM, 장치 전체 점유, CPU RSS를 보고한다. VRAM을 채우는 것 자체나 NVML 사용률 상승을 성공 기준으로 삼지 않는다. 측정 전 학습 속도 배수를 약속하지 않는다.

PyTorch2.3.1의 asynchronous execution, 정확한 CUDA timing, CUDA Graph 제약은 해당 버전의 [공식 CUDA 문서 소스](https://github.com/pytorch/pytorch/blob/v2.3.1/docs/source/notes/cuda.rst)를 기준으로 확인한다.

우선 구현 묶음은 **고정 metadata/index 상주 + 불필요 scalar read 제거 + target/input GPU 상주 + edge chunk 확대**다. 이후 profiling이 남은 비용을 보여주는 순서에 따라 immutable context, ballistic/16-bin 학습 경로, compile/graph를 진행한다. 새로운 전체 학습 campaign은 이 설계 문서만으로 시작하지 않는다.
