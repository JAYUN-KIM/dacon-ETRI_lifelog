# DACON ETRI Lifelog AI Competition

DACON 제5회 ETRI 휴먼이해 인공지능 논문경진대회 실험 기록 저장소입니다.  
라이프로그 센서 데이터를 활용해 수면, 피로도, 스트레스 관련 7개 이진 타겟을 예측하는 문제를 다뤘습니다.

이 저장소는 대회 종료 이후 논문 작성과 포트폴리오 정리를 위해, 실험 코드와 핵심 결과를 재현 가능한 형태로 남기는 것을 목표로 합니다.

## 프로젝트 개요

- 대회: DACON 제5회 ETRI 휴먼이해 인공지능 논문경진대회
- 목표: subject별 lifelog date 기준 Q1~Q3, S1~S4 총 7개 binary target 예측
- 평가 지표: Average Log-Loss
- 데이터 규모: train 약 450행, test 250행
- 입력 데이터: 일별 라이프로그 메타 정보와 센서 parquet 데이터
- 최종 실험 상태: 2026-05-18 기준 실험 종료, 논문 작성 단계로 전환

## 현재 성과

<!-- AUTO:PROJECT_STATUS:START -->
- 최고 Public LB: **0.5877431660**
- 최신 최고점 갱신일: **2026-05-15**
- 핵심 개선 축: 개인별 날짜/상태 prior, target-wise calibration, 보수적 앵커 블렌딩
- 최근 연구 축: 원천 센서 기반 raw-context feature와 공동 타겟 패턴 보정
- 상세 실험 기록: `experiments/log.json`
<!-- AUTO:PROJECT_STATUS:END -->

최종 최고 기록은 `sub_0515_metricproxy_s_only.csv` 계열에서 나왔습니다.  
수면 시간창 재정렬, 개인별 날짜 prior, target별 보정, S계열 수면 지표 proxy를 얇게 결합하는 방향이 가장 안정적이었습니다.

## 예측 타겟

| 타겟 | 의미 |
|---|---|
| Q1 | 수면의 질 |
| Q2 | 취침 전 피로도 |
| Q3 | 스트레스 |
| S1 | 총 수면시간 권장 여부 |
| S2 | 수면 효율 권장 여부 |
| S3 | 수면 지연시간 권장 여부 |
| S4 | 각성시간 권장 여부 |

## 핵심 접근

1. 안정적인 일별 센서 집계  
   작은 데이터셋 특성상 복잡한 피처 확장보다 HR, pedometer, activity, light, screen, charging 등 비교적 안정적인 일별 집계가 더 강건했습니다.

2. 보수적인 확률 보정  
   log-loss는 과신한 확률에 민감하므로 probability shrink, clipping, target별 blend를 통해 예측 확률을 안정화했습니다.

3. target-wise routing  
   Q계열과 S계열의 모델 선호가 달라 전체 타겟에 같은 blend를 적용하면 손해가 났습니다. Q계열은 LGB 비중을 높이고, S계열은 CatBoost-heavy 구성이 더 유효했습니다.

4. 개인별 날짜/상태 prior  
   subject별 과거 라벨 흐름과 가까운 날짜의 상태가 test 구간으로 이어진다는 가설이 가장 큰 개선 축이었습니다.

5. 수면 시간창 재정렬  
   lifelog date 단순 집계 대신, 수면 관련 타겟에 맞춰 저녁부터 다음날 아침까지의 센서 구간을 재구성한 실험이 성능을 추가로 개선했습니다.

6. 수면 지표 proxy 보정  
   센서 기반으로 수면 시간, 효율, 지연, 각성 proxy를 만들고 S계열에 약하게 반영한 후보가 최종 최고점을 만들었습니다.

## 주요 실험 흐름

| 단계 | 핵심 내용 | 결과 |
|---|---:|---|
| 안정 베이스라인 | daily aggregation + LGB/Cat ensemble | 0.599대 진입 |
| target-wise blend | Q/S별 LGB-CatBoost 비율 분리 | 0.5969444527 |
| seed ensemble | 3-seed ensemble로 분산 완화 | 0.5957800203 |
| state/date prior | subject별 최근 상태와 날짜 prior 도입 | 0.5919692903 |
| interpolation/date confidence | 가까운 날짜 라벨 흐름을 보수적으로 반영 | 0.5886910305 |
| sleep-window | 수면 문제 정의에 맞춘 시간창 재정렬 | 0.5878818802 |
| sleep metric proxy | S계열 수면 지표 proxy 보정 | 0.5877431660 |

## 주요 인사이트

- 데이터가 작기 때문에 “많은 피처”보다 “안전한 피처와 보수적 보정”이 더 중요했습니다.
- public LB 개선은 모델 구조보다 개인별 prior와 시간축 해석을 바꿀 때 크게 나타났습니다.
- Q계열과 S계열은 같은 문제로 묶기보다 서로 다른 라우팅/보정 정책이 필요했습니다.
- 센서 raw-context를 크게 확장하는 실험은 단독으로는 불안정했고, 검증된 앵커에 얇게 섞는 방식이 더 안전했습니다.
- 최종 단계에서는 개선폭이 매우 작아져, leaderboard 최적화보다 논문용 해석과 실험 재현성 정리가 더 중요해졌습니다.

## 주요 코드

| 파일 | 역할 |
|---|---|
| `scripts/run_etri_target_history_seed3_routing_q5050_s2080_alpha098.py` | seed ensemble + target-wise routing 초기 강한 앵커 |
| `scripts/run_etri_state_transition_candidates_20260505.py` | subject별 상태 전이 prior 후보 생성 |
| `scripts/make_state_transition_blend_grid_20260505.py` | state prior blend 강도 탐색 |
| `scripts/make_subject_date_interpolation_prior_20260506.py` | subject-date interpolation prior 생성 |
| `scripts/make_0509_after_g030_candidates.py` | target별 reliability/scale 보정 후보 생성 |
| `scripts/make_0510_reset_raw_context_candidates.py` | 원천 센서 기반 raw-context 후보 생성 |
| `scripts/make_0511_sleep_window_candidates.py` | 수면 시간창 재정렬 후보 생성 |
| `scripts/make_0515_sleep_metric_proxy_candidates.py` | 수면 지표 proxy 기반 S계열 보정 후보 생성 |
| `scripts/make_0517_joint_correlation_relax_candidates.py` | 공동 타겟 상관 완화 후보 생성 |
| `scripts/validate_submission.py` | 제출 파일 shape/null/range 검증 |
| `scripts/auto_push.py` | 실험 로그 기록, README 갱신, GitHub 업로드 |

## 프로젝트 구조

```text
etri-lifelog/
├── data/                  # 원본/가공 데이터와 제출 파일, gitignore
├── experiments/           # 실험 로그와 주요 실험 정리
├── notebooks/             # EDA 및 모델링 노트북
├── paper/                 # 논문 작성 자료
├── scripts/               # 실험/후보 생성/검증 스크립트
├── src/                   # 공통 모듈
└── README.md
```

## 상세 기록

- 실험 로그: [`experiments/log.json`](experiments/log.json)
- 최종 회고: [`experiments/2026-05-18_final_wrapup.md`](experiments/2026-05-18_final_wrapup.md)
- 주요 스크립트: [`scripts/`](scripts/)

## 비고

원본 데이터셋과 제출 파일은 용량 및 대회 규정 관리를 위해 GitHub에 포함하지 않았습니다.  
이 저장소에는 실험 코드, 실험 로그, 연구 방향 메모, 논문 작성에 활용할 수 있는 결과 해석을 중심으로 정리했습니다.
