# 2026-05-09 실험 정리: Target별 reliability scale 정교화

## 핵심 결과

- 전날 최고점 `0.5884463893`에서 시작했다.
- `sub_0509_best_meanalign_g030.csv`는 `0.5884235486`으로 소폭 개선되었다.
- `sub_0509_afterg030_targetscale_soft.csv`는 `0.5883722159`로 오늘 최고점을 갱신했다.
- `sub_0509_afterg030_mixed_date_bracket.csv`는 `0.5884111821`로 개선되었지만 최고점에는 미치지 못했다.

## 오늘의 해석

오늘 실험은 전날 먹혔던 `subject reliability + adaptive date prior` 축을 더 정교하게 다듬는 방향이었다. 첫 번째 제출은 현재 best 위에 아주 약한 mean-align을 한 번 더 적용한 안전 카드였다. 개선폭은 작았지만 여전히 target 평균 보정이 약하게 유효함을 확인했다.

두 번째 제출은 g030을 새 anchor로 놓고, target별 reliability scale을 다시 조정한 후보였다. 특히 S 계열은 보수적으로 두고, S3는 계속 약하게 유지했다. 이 후보가 `0.5883722159`로 오늘 최고점을 갱신하면서, 현재 가장 유망한 축은 `bracket 추가`보다 `target별 reliability/date-prior 강도 최적화`임을 확인했다.

세 번째 제출은 date prior에 bracket prior를 confidence가 높은 구간에서 일부 섞은 후보였다. `0.5884111821`로 기존 최고보다는 개선되었지만 targetscale soft보다 낮았다. 따라서 bracket prior는 보조 후보로는 가치가 있으나 현재 주력 축은 아니다.

## 생성한 주요 코드

- `scripts/make_0509_reliability_refine_candidates.py`
  - 전날 best 기준으로 mean-align, target별 reliability scale, date+bracket mixed prior 후보를 생성했다.
- `scripts/make_0509_after_g030_candidates.py`
  - 첫 제출 `g030`을 새 anchor로 다시 잡고, 남은 제출 후보를 중복 보정 없이 재생성했다.

## 주요 후보와 결과

| 후보 | 역할 | Public LB |
|---|---|---:|
| `sub_0509_best_meanalign_g030.csv` | 현재 best에 약한 mean-align 추가 | `0.5884235486` |
| `sub_0509_afterg030_targetscale_soft.csv` | g030 anchor 기준 target별 reliability/date-prior scale 재조정 | `0.5883722159` |
| `sub_0509_afterg030_mixed_date_bracket.csv` | g030 anchor 기준 date prior와 bracket prior를 confidence 기반 혼합 | `0.5884111821` |

## 다음 방향

1. 현재 최고는 `0.5883722159`로 갱신한다.
2. 다음 실험은 `sub_0509_afterg030_targetscale_soft.csv`를 새 anchor로 두고 진행한다.
3. `targetscale_soft`가 가장 유효했으므로 Q/S별 scale을 더 세분화한다.
4. S3는 계속 약하게 유지한다. 최근 실험에서 S3 또는 streak류 보정이 강해지면 악화되는 경향이 있었다.
5. bracket prior는 단독 주력으로 밀기보다, confidence가 매우 높은 subset에서만 약하게 쓰는 후보로 남긴다.
6. mean-align은 개선폭이 작아졌으므로 하루 첫 제출의 안전 보정 카드 또는 최종 후보 보조 보정 정도로만 사용한다.
