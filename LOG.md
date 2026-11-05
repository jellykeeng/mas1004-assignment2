# Log

One entry for each recipe, selection or Track B model you train, newest at
the bottom. The numbers come from `experiments.csv`. "Why" and "what it means"
are in your own words: your agent asks you for them and does not make them up.
End the log with your plan for what to try next.

<!-- The form of an entry:

## <date> <tag> (Track A or B)
- changed: what is different from the run before
- why: your reason
- validation: 0.000 (before: 0.000)
- what it means: what you think the result says
-->

## 2026-11-05 baseline (Track A, seed 0)

- 목적: 변경 전 스타터 레시피의 전체 학습 결과를 비교 기준으로 기록한다.
- 조건: 스타터 레시피 그대로, 전체 훈련 사진 44,257장, ResNet18, 입력 224 x 224,
  batch size 64, seed 0, CUDA. 3,906 steps로 249,984회 사진을 제시했다.
- 코드: commit `a23f9e6`, recipe SHA-256
  `17ac485223fa61741f2c63aeb1c91bc283c88c6b7aaa8f02fdb3e86924ff29c9`.
  결과 파일의 recipe hash가 `track-a-1`에 게시한 레시피와 일치한다.
- 검증: 사진 12,077장에 대한 도시별 정확도의 평균 **0.5044**.

| 도시 | 검증 정확도 |
| --- | ---: |
| Seoul | 0.5774 |
| Busan | 0.3942 |
| Tokyo | 0.6385 |
| Osaka | 0.3604 |
| Taipei | 0.5515 |

- 학습 시간: 2,238.2초 (약 37.3분). 데이터 준비, 검증, 내보내기 시간은 별도다.
- Colab의 ONNX 검사 출력: 44.7 MB, 11.17 million parameters,
  1.81 billion multiply-adds at 224 x 224; `format and limits: OK`.
- 수치 관찰: 오사카의 정확도가 가장 낮다. 혼동행렬에서 실제 오사카 2,128장 중
  855장을 도쿄로 예측했다 (약 40.18%, 결과 JSON에서 계산).
- 해석의 범위: seed 0 한 번의 검증 결과다. 리더보드 점수와는 다른 데이터에서
  측정한 값이며, 아직 개선 실험의 결과나 seed 간 변동 폭은 없다.
- 학생이 사진을 직접 보고 내린 판단: 아직 기록되지 않았다.
- 원본: 2026-11-05에 받은 `from_colab.zip`의
  `results/baseline_seed0/result.json`, `results/colab_experiments.csv`,
  `results/colab_logs/1105_151903_official_baseline_seed0.txt`,
  `results/colab_logs/1105_155816_check_submission.txt`.
  Track A 모델은 ZIP에 포함하지 않는 노트북의 기본 동작에 따라 Colab Drive에 남아 있다.

### 다음 계획

1. 레시피와 훈련 사진은 유지하고 seed 1로 같은 기준 실험을 실행한다.
2. 두 실행의 점수 차이를 확인한 후, 오답 사진을 직접 살펴본다.
3. 사진에서 관찰한 내용을 바탕으로 변경할 항목을 하나 정하고 그 효과를 측정한다.

## 2026-11-05 baseline (Track A, seed 1)

- 변경: seed 0에서 seed 1로만 변경했다. 레시피, 전체 훈련 사진, batch size,
  steps, 예산, 모델과 입력 크기는 첫 기준 실험과 동일하다.
- 목적: 같은 학습 방법에서 난수 선택에 따른 점수 차이를 관찰한다.
- 코드: commit `0dd327b`, recipe hash `17ac485223fa` (seed 0과 동일).
- 검증: 사진 12,077장에 대한 도시별 정확도의 평균 **0.4984**.

| 도시 | seed 0 | seed 1 |
| --- | ---: | ---: |
| Seoul | 0.5774 | 0.5626 |
| Busan | 0.3942 | 0.3842 |
| Tokyo | 0.6385 | 0.6140 |
| Osaka | 0.3604 | 0.3891 |
| Taipei | 0.5515 | 0.5421 |
| 도시별 정확도의 평균 | 0.5044 | 0.4984 |

- 두 실행의 검증 점수 평균: **0.5014**. 절대 차이: **0.0060 (0.60%p)**.
  `result.json` 두 개의 `val_score`에서 계산했다.
- 해석의 범위: 방법을 바꾸지 않아도 이번 두 실행에서 0.60%p 차이가 났다.
  작은 점수 상승 한 번만으로 개선 효과를 단정하지 않는다. 두 번의 실행으로
  전체 변동 범위나 통계적 유의성을 추정한 것은 아니다.
- 학습 시간: 2,179.5초 (약 36.3분). 준비, 검증, 내보내기 시간은 별도다.
- Colab의 ONNX 검사 출력: 44.7 MB, 11.17 million parameters,
  1.81 billion multiply-adds at 224 x 224; `format and limits: OK`.
- 학생이 사진을 직접 보고 내린 판단: 아직 기록되지 않았다.
- 원본: `from_colab-2.zip`의 `results/baseline_seed1/result.json`,
  `results/colab_experiments.csv`,
  `results/colab_logs/1105_161517_official_baseline_seed1.txt`,
  `results/colab_logs/1105_165326_check_submission.txt`.

### 다음 계획

1. Colab에서 seed 0 모델이 검증 사진을 확신하면서 틀린 사례를 도시별 48장씩
   모은다 (`curate.py mistakes`, `results/curate/`). 이 모음은 전체 사진의
   무작위 표본이 아니라 확신이 높은 오답을 골라 본 자료다.
2. 학생이 사진 모음을 직접 보고 공통점을 적는다.
3. 학생의 관찰을 바탕으로 변경할 항목을 하나 정한 뒤, 같은 seed와 예산으로
   변경 전후를 비교한다. 아직 사진 제외나 레시피 변경은 하지 않았다.

## 2026-11-05 baseline 오답 사진 모음

- 실행: `python src/curate.py mistakes results/baseline_seed0/model.onnx --n 48`
  (commit `7e8ccc8`).
- 결과: 검증 사진 12,077장에 seed 0 ONNX 모델을 적용한 후, 도시별로 확신이 높은
  오답 48장씩을 골라 총 240장의 사진을 5개 모음으로 저장했다.
- 파일: `results/curate/mistakes_baseline_seed0_<city>_00.png`와 같은 이름의 CSV.
  `<city>`는 seoul, busan, tokyo, osaka, taipei다.
- 모음의 도시 이름은 정답 라벨이고, 사진 아래 화살표 뒤의 도시는 모델의 예측이다.
  숫자는 softmax 출력 확률이다. 번호는 같은 이름의 CSV에 있는 사진 경로와 대응한다.
- 이 모음은 확신이 높은 오답을 살펴보는 자료이며, 전체 검증 사진의 무작위 표본은
  아니다. 기존의 기준 실험 점수는 `official.py`의 `result.json` 값을 유지한다.
- 원본: `from_colab-3.zip`과
  `results/colab_logs/1105_165739_curate_mistakes.txt`.
- 이 단계는 사진 검토 준비이므로 `experiments.csv`에 새 학습 행을 추가하지 않았다.
- 학생의 사진 관찰과 개선 결정: 응답 대기.

### 다음 계획

1. 학생이 도시별 사진 모음을 직접 보고 공통적인 특징 두 가지를 설명한다.
2. 관찰에서 나온 개선 가설을 하나 정하고, 훈련 사진 선택 또는 학습 방법 중
   해당 항목 하나를 변경해 기준 실험과 비교한다.

## 2026-11-05 학생의 사진 관찰: 오사카와 부산

학생이 오답 사진 모음을 직접 보고 남긴 말:

> 오사카 - 신사, 주택, 주택가 같은 경우 인도가 없음
>
> 부산 - 좀 더 어지럽고 왁자직껄, 부산스러움, 8번 같은 경우 주택가인데 차가 세워져 있음 (일본은 주택가에 도로에 차가 안 세워져 있는듯)

- 검토한 모음: `mistakes_baseline_seed0_osaka_00`와
  `mistakes_baseline_seed0_busan_00`. 부산의 구체적인 사례로 8번을 지목했다.
- 일본 주택가의 노상 주차에 대한 말은 학생이 이번 사진에서 세운 비교 가설이다.
  도시 전체에 적용되는 사실이나 모델이 실제로 사용한 단서로 확정하지 않는다.
- 다음 단계에 대한 학생의 선택: **다른 도시의 오답 사진도 먼저 보기**.
  서울, 도쿄, 타이베이 모음을 추가로 검토한 다음 첫 개선 실험을 정한다.

### 다음 계획

1. 서울, 도쿄, 타이베이 오답 사진 모음을 학생에게 보여준다.
2. 학생의 추가 관찰을 기록하고, 학생이 선택한 변경 항목 하나의 효과를 측정한다.

## 2026-11-05 학생의 추가 관찰: 부산과 서울

학생의 말 (발췌):

> 부산은 바다가 있는 것을 제외하고는 많은 부분이 서울이랑 비슷한 것 같아 그래서 오답이 서울이 많은 것 같아 부산에

- 학생의 가설: 서울과 부산에 공통적인 대도시 풍경이 있어서 모델이 둘을 혼동할 수 있다.
- 확인한 오답 패턴: 부산 오답 모음 48장의 CSV에서 34장이 서울 예측이다.
  이 모음은 확신이 높은 오답만 골라 놓은 자료다.
- 전체 검증 결과의 혼동행렬로도 확인했다. 아래 값은 각 실행의 `result.json`에서
  실제 부산 행을 읽어 계산한 값이다.

| 실행 | 부산 검증 사진 | 부산 오답 수 | 서울로 예측한 수 | 부산 오답 중 서울 예측 비율 |
| --- | ---: | ---: | ---: | ---: |
| seed 0 | 1390 | 842 | 442 | 52.49% |
| seed 1 | 1390 | 856 | 463 | 54.09% |

- 두 기준 실험 모두 부산 오답의 절반 이상이 서울 예측이다. 학생이 지적한 혼동
  패턴은 수치에서도 나타난다. 풍경의 유사성이 그 원인인지는 아직 검증할 가설이다.

### 다음 계획

서울, 도쿄, 타이베이 사진을 더 살펴본 뒤 학생이 첫 개선 실험의 항목을 선택한다.
