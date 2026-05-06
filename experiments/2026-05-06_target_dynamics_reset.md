# 2026-05-06 실험 정리: Target-dynamics reset 축

## 핵심 결과

- 기존 센서/seed ensemble 축을 더 미세조정하지 않고, ETRI 문제를 `subject별 7개 binary state의 시간 전이 예측`으로 다시 해석했다.
- `sub_reset_targetdyn_revert_anchor_w10_20260506.csv` 제출 결과 `0.5902811814`를 기록했다.
- 같은 방향을 조금 더 강하게 섞은 `sub_reset_targetdyn_revert_anchor_w14_grid_20260506.csv`가 `0.5898630289`로 추가 개선되었다.
- subject-date interpolation prior를 추가로 섞은 `sub_dateinterp_smooth_tau10_anchor_w07_20260506.csv`가 `0.5892681038`을 기록하며 오늘 최고점을 갱신했다.

## 오늘의 해석

이번 개선은 센서 feature 확장이 아니라, `개인별 상태가 시간적으로 이어진다`는 구조적 가설에서 나왔다. 특히 train/test 날짜를 다시 확인해보니 test가 subject별 train 마지막 이후로만 이어지는 단순 미래 예측이 아니라, train 라벨 날짜 사이에 끼어 있는 형태도 많았다. 그래서 단순 recursive future prior뿐 아니라 가까운 날짜의 train label을 거리 가중으로 보는 interpolation prior도 유효했다.

다만 interpolation prior는 현재 best 대비 작은 비율로 섞었을 때만 안전했다. 순수 interpolation prior는 기존 best와 예측 차이가 커서 public/private 안정성 측면에서 위험하다. 따라서 다음 단계에서는 이 축을 더 세게 미는 것보다 target별/subject별로 어디에서만 믿을지 조절하는 방향이 맞다.

## 생성한 주요 코드

- `scripts/run_etri_reset_target_dynamics_20260506.py`
- `scripts/make_reset_targetdyn_blend_grid_20260506.py`
- `scripts/make_subject_date_interpolation_prior_20260506.py`
- `scripts/validate_submission.py`: Windows/WSL 경로 모두에서 동작하도록 root path를 script 기준으로 수정했다.

## 주요 후보와 역할

| 후보 | 역할 | 상태 |
|---|---|---|
| `sub_reset_targetdyn_revert_anchor_w10_20260506.csv` | target-dynamics reset prior를 기존 best에 10% 섞은 첫 개선 후보 | 제출 완료, `0.5902811814` |
| `sub_reset_targetdyn_revert_anchor_w14_grid_20260506.csv` | 같은 축을 14%로 조금 더 강하게 섞은 후보 | 제출 완료, `0.5898630289` |
| `sub_dateinterp_smooth_tau10_anchor_w07_20260506.csv` | 가까운 subject-date label interpolation prior를 현재 best에 7% 섞은 후보 | 제출 완료, `0.5892681038` |
| `sub_dateinterp_smooth_tau10_anchor_w10_20260506.csv` | interpolation prior를 더 강하게 섞은 후보 | 미제출, 다음 검토 후보 |
| `sub_dateinterp_smooth_tau10_anchor_tw_q12_s08_20260506.csv` | Q/S별로 interpolation prior 강도를 다르게 둔 후보 | 미제출, 리스크 있는 후보 |

## 다음 방향

1. 현재 최고는 `0.5892681038`로 기록한다.
2. uniform blend를 더 세게 미는 것은 한계가 있을 수 있으므로, target별/subject별로 dynamics/interpolation prior의 신뢰도를 다르게 주는 쪽을 우선한다.
3. 센서 feature를 대량 추가하는 방향은 뒤로 미루고, `개인별 상태 전이`, `가까운 날짜 interpolation`, `target 공동분포`를 중심축으로 계속 확장한다.
4. metric이 Average Log-Loss이므로 점수는 낮을수록 좋다. 오늘의 `0.5892681038`은 기존 `0.5919692903` 대비 의미 있는 개선이다.
