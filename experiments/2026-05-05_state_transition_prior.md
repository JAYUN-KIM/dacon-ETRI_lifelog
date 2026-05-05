# 2026-05-05 실험 정리: State-transition prior 축

## 핵심 결과

- 기존 최고 축인 `sub_seed3_routing_q6040_s2080_alpha098.csv`를 anchor로 유지했다.
- 센서 feature를 더 늘리는 대신, subject별 최근 타겟 상태가 test 구간으로 이어진다는 가설을 별도 prior로 만들었다.
- `sub_anchor_q6040_statepast_q14_s07_20260505.csv` 제출 결과 `0.5922690071`로 큰 폭 개선되었다.
- 이후 target별로 state prior 강도를 다르게 둔 `sub_anchor_q6040_statepast_tw_b_20260505.csv`가 `0.5919692903`으로 추가 최고점을 갱신했다.

## 오늘의 해석

작은 데이터셋에서는 센서 feature expansion보다 개인별 타겟 상태 전이가 더 강한 일반화 신호로 작동했다. 특히 Q2/Q3 계열은 최근 피로도/스트레스 상태가 이어지는 경향이 있어 state prior를 더 강하게 섞는 것이 유효했고, S1/S3는 state prior가 노이즈일 수 있어 약하게 두는 편이 안정적이었다.

## 생성한 주요 코드

- `scripts/run_etri_state_transition_candidates_20260505.py`
- `scripts/make_state_transition_blend_grid_20260505.py`
- `scripts/make_state_transition_targetwise_refine_20260505.py`

## 주요 후보와 역할

| 후보 | 역할 | 상태 |
|---|---|---|
| `sub_anchor_q6040_statepast_q14_s07_20260505.csv` | past-only state prior를 Q 14%, S 7% 섞은 첫 개선 후보 | 제출 완료, `0.5922690071` |
| `sub_anchor_q6040_statepast_tw_b_20260505.csv` | Q2/Q3 강화, S1/S3 약화 target-wise 후보 | 제출 완료, `0.5919692903` |
| `sub_anchor_q6040_statepast_tw_qboost1_20260505.csv` | 내일 1순위 후보, Q2/Q3를 한 단계 더 강화 | 제출 대기 |
| `sub_anchor_q6040_statepast_tw_b_s3zero_20260505.csv` | S3 prior 제거형 안전 미세조정 | 제출 대기 |
| `sub_anchor_q6040_statepast_tw_qaggr_20260505.csv` | Q2/Q3 공격 강화 후보 | `qboost1` 결과 확인 후 검토 |

## 다음 액션

1. `sub_anchor_q6040_statepast_tw_qboost1_20260505.csv`를 먼저 제출한다.
2. 개선되면 `tw_qaggr`로 Q2/Q3 강화 폭을 더 탐색한다.
3. 악화되면 `tw_b_s3zero`로 S3 노이즈 제거형 미세조정을 확인한다.

