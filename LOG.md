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
