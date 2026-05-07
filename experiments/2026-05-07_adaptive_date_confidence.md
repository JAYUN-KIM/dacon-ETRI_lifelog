# 2026-05-07 실험 정리: Calibration과 adaptive date-confidence prior

## 핵심 결과

- 전날 최고점 `0.5892681038`에서 시작했다.
- 첫 제출 `sub_currentbest_meanalign_g040_20260507.csv`가 `0.5892271317`을 기록했다.
- 두 번째 제출 `sub_currentbest_meanalign_g060_20260507.csv`가 `0.5892082651`로 소폭 추가 개선되었다.
- 세 번째 제출 `sub_adaptive_dateconf_dateprior_b03_s10_20260507.csv`가 `0.5886910305`를 기록하며 오늘 최고점을 갱신했다.

## 오늘의 해석

오늘은 기존 best를 단순히 더 강하게 섞는 대신, 문제를 다시 한 번 `subject별 날짜 미관측 라벨 보간 문제`로 보고 접근했다. 먼저 현재 best의 target 평균이 train target mean과 살짝 어긋나 있을 수 있다고 보고 logit mean-align 보정을 적용했다. `g040`, `g060` 모두 실제 Public LB에서 작게 개선되어, 현재 확률 평균 보정 방향이 유효함을 확인했다.

이후 uniform date interpolation blend에서 한 단계 더 나아가, test 날짜가 같은 subject의 train label 날짜와 얼마나 가까운지에 따라 date prior를 다르게 믿는 adaptive blend를 설계했다. 가까운 train label이 있는 날짜에서는 date prior를 더 많이 섞고, 멀거나 한쪽 날짜만 있는 경우에는 현재 best를 더 유지하는 방식이다. 이 adaptive date-confidence prior가 `0.5886910305`로 가장 큰 추가 개선을 만들었다.

## 생성한 주요 코드

- `scripts/make_calendar_bracket_priors_20260507.py`
  - subject별 앞/뒤 train label bracket prior와 다른 subject의 calendar prior를 실험했다.
- `scripts/make_current_best_calibration_20260507.py`
  - 현재 best 기준 target mean-align, temperature, correlation alignment 후보를 생성했다.
- `scripts/make_adaptive_date_prior_blends_20260507.py`
  - train label 날짜와 test 날짜의 거리 기반 confidence를 계산하고, confidence에 따라 date prior blend 강도를 다르게 적용했다.

## 주요 후보와 결과

| 후보 | 역할 | Public LB |
|---|---|---:|
| `sub_currentbest_meanalign_g040_20260507.csv` | 현재 best의 target mean을 train mean 방향으로 약하게 보정 | `0.5892271317` |
| `sub_currentbest_meanalign_g060_20260507.csv` | mean-align을 조금 더 강화 | `0.5892082651` |
| `sub_adaptive_dateconf_dateprior_b03_s10_20260507.csv` | 날짜 거리 confidence 기반 adaptive date prior blend | `0.5886910305` |

## 실패/보류 후보

- `sub_hybrid_bracket_calendar_anchor_w07_20260507.csv`
  - bracket/calendar prior를 균일하게 섞는 후보였지만, adaptive date-confidence가 더 설득력 있어 제출하지 않았다.
- `sub_currentbest_corralign_g040_20260507.csv`
  - target correlation을 train correlation에 맞추는 후보였지만, mean-align보다 변화폭과 리스크가 커 제출하지 않았다.
- `sub_currentbest_temp_t0970_20260507.csv`
  - confidence temperature 조정 후보였지만, 오늘은 mean-align과 adaptive prior가 더 우선이었다.

## 다음 방향

1. 현재 최고는 `0.5886910305`로 갱신한다.
2. 단순 mean-align은 `g060` 근처에서 개선폭이 작아졌으므로 다음에는 `g080` 같은 미세조정보다 새 축 탐색을 우선한다.
3. adaptive date-confidence prior가 강하게 먹혔으므로, 다음 실험은 confidence 함수를 target별로 다르게 두는 방향이 유망하다.
4. 특히 Q 계열과 S 계열의 날짜 보간 신뢰도가 다를 수 있으므로 `Q/S별 confidence scale`, `S3 약화`, `subject별 prior 신뢰도`를 따로 설계할 가치가 있다.
5. 센서 feature 확장은 계속 후순위로 둔다. 현재까지는 작은 데이터에서 센서 feature보다 target state/date structure가 더 강한 leaderboard 신호로 작동했다.
