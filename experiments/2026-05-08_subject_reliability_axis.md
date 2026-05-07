# 2026-05-08 실험 정리: Subject reliability 기반 date prior와 streak prior

## 핵심 결과

- 전날 최고점 `0.5886910305`에서 시작했다.
- `sub_0508_currentbest_meanalign_g035.csv`는 `0.5886634521`로 소폭 개선되었다.
- `sub_0508_subjrel_dateprior_adaptive.csv`는 `0.5884463893`으로 오늘 최고점을 갱신했다.
- `sub_0508_streak_reversion_w055.csv`는 `0.589895946`으로 악화되었다.

## 오늘의 해석

첫 번째 제출은 전날 mean-align 보정이 아직 아주 약하게 남아 있는지 확인하는 안전 카드였다. 개선폭은 작았지만 `0.5886910305`에서 `0.5886634521`로 내려가며, 현재 best의 target mean 방향 보정이 완전히 끝난 것은 아니라는 점을 확인했다.

두 번째 제출은 새 축이었다. 전날 adaptive date-confidence prior는 test 날짜가 train label 날짜와 가까울수록 date prior를 더 믿는 구조였다. 오늘은 여기에 subject별 prior 신뢰도를 추가했다. subject별 과거 라벨만으로 다음 라벨을 얼마나 잘 맞히는지를 간단히 추정하고, 이 신뢰도가 높은 subject에서는 date prior를 조금 더 강하게 섞었다. 이 방식이 `0.5884463893`으로 가장 큰 개선을 만들었다.

세 번째 제출은 streak/reversion prior였다. 각 target의 최근 연속 상태가 계속될지 또는 반전될지를 prior로 만들었지만, Public LB에서는 크게 악화되었다. 이는 현재 test 구간에서는 단순 streak 지속/반전보다 날짜 근접성과 subject별 보간 신뢰도가 더 강한 신호이며, 확률을 재귀적으로 history에 넣는 streak 방식이 노이즈를 증폭했을 가능성이 있다.

## 생성한 주요 코드

- `scripts/make_0508_new_axis_candidates.py`
  - 현재 best 위에서 mean-align 후보, subject reliability 기반 adaptive date prior 후보, streak/reversion prior 후보를 생성했다.

## 주요 후보와 결과

| 후보 | 역할 | Public LB |
|---|---|---:|
| `sub_0508_currentbest_meanalign_g035.csv` | 현재 best의 target mean을 train mean 방향으로 약하게 추가 보정 | `0.5886634521` |
| `sub_0508_subjrel_dateprior_adaptive.csv` | subject별 prior 신뢰도를 반영한 adaptive date prior blend | `0.5884463893` |
| `sub_0508_streak_reversion_w055.csv` | 최근 target streak 지속/반전 prior를 약하게 반영 | `0.589895946` |

## 다음 방향

1. 현재 최고는 `0.5884463893`으로 갱신한다.
2. subject별 date prior 신뢰도 축은 계속 유지할 가치가 있다.
3. streak/reversion prior는 현재 형태로는 보류한다. 재사용한다면 recursive update를 줄이거나, target별로 Q2/Q3 등 일부에만 매우 약하게 적용해야 한다.
4. 다음 실험은 `subject별 reliability 산식 개선`, `target별 reliability scale`, `date confidence와 subject reliability의 곱/상한 조정`이 우선이다.
5. mean-align은 개선폭이 작아졌으므로 보조 보정으로만 사용하고, 제출 3장을 모두 여기에 쓰지는 않는다.
