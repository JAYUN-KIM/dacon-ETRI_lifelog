# DACON ETRI Lifelog AI Competition

라이프로그 센서 데이터를 활용해 수면, 피로도, 스트레스 관련 7개 이진 타겟을 예측하는 프로젝트입니다.  
작은 데이터셋에서 과적합을 줄이고, 개인별 상태 전이 신호를 활용해 Average Log-Loss를 낮추는 것을 목표로 합니다.

## 프로젝트 개요

- 대회: 제5회 ETRI 휴먼이해 인공지능 논문경진대회
- 플랫폼: DACON
- 평가 지표: Average Log-Loss
- 데이터 규모: train 약 450행, test 250행
- 입력 데이터: subject별 일자 단위 라이프로그 및 센서 parquet
- 목표: Q1~Q3, S1~S4 총 7개 binary target 예측

## 현재 성과

<!-- AUTO:PROJECT_STATUS:START -->
- 최고 Public LB: **0.5892681038**
- 최신 최고점 갱신일: **2026-05-06**
- 핵심 개선 축: subject별 최근 타겟 상태 전이를 활용한 state-transition prior
- 상세 실험 기록은 `experiments/` 디렉토리에 분리 보관
<!-- AUTO:PROJECT_STATUS:END -->

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

## 핵심 접근법

1. 안정적인 일별 센서 집계
   - 활동량, 조도, 화면 사용, 충전 상태, 심박, 걸음 수 등 안정적인 센서 중심으로 일자 단위 feature를 구성했습니다.

2. 보수적인 확률 보정
   - 작은 데이터셋에서 확률 과신이 쉽게 발생해 probability shrink와 clipping을 적용했습니다.

3. CatBoost-heavy ensemble과 target-wise routing
   - Q 계열과 S 계열의 모델 선호도가 달라, target 그룹별로 LightGBM/CatBoost blend 비율을 다르게 적용했습니다.

4. Seed ensemble
   - 3개 seed ensemble로 작은 데이터셋에서 발생하는 모델 분산을 줄였습니다.

5. State-transition prior
   - 센서 feature 확장만으로는 한계가 있어, subject별 최근 Q/S 상태가 test 구간으로 이어진다는 가설을 별도 prior로 구성했습니다.
   - 이 축에서 `0.59553`대에서 `0.5919692903`까지 큰 폭 개선을 확인했습니다.

## 주요 인사이트

- feature를 무작정 늘리는 것보다 안정적인 집계와 확률 보정이 더 강했습니다.
- Q 계열은 LightGBM 비중을 높였을 때, S 계열은 CatBoost-heavy 구성이 더 안정적이었습니다.
- 5-seed ensemble보다 3-seed ensemble이 더 좋았고, seed 수를 늘리는 것이 항상 개선으로 이어지지는 않았습니다.
- 개인별 최근 타겟 상태 전이 신호가 public leaderboard에서 가장 큰 개선을 만들었습니다.
- S1/S3는 state prior가 노이즈가 될 수 있어, target별로 prior 강도를 다르게 주는 방식이 효과적이었습니다.

## 대표 실험 코드

| 파일 | 역할 |
|---|---|
| `scripts/run_etri_seed3_routing_q6040_s2080_alpha098.py` | 기존 best 계열 anchor 모델 |
| `scripts/run_etri_state_transition_candidates_20260505.py` | state-transition prior 후보 생성 |
| `scripts/make_state_transition_blend_grid_20260505.py` | state prior 강도 grid 실험 |
| `scripts/make_state_transition_targetwise_refine_20260505.py` | target-wise state prior 세부 조정 |
| `scripts/validate_submission.py` | 제출 파일 shape/null/range 검증 |

## 프로젝트 구조

```text
etri-lifelog/
├── data/                  # 원본/가공 데이터, 제출 파일은 gitignore
├── experiments/           # 실험 로그와 주요 실험 정리
├── notebooks/             # EDA 및 모델링 노트북
├── paper/                 # 논문 작성 자료
├── scripts/               # 실험/후보 생성/검증 스크립트
├── src/                   # 공통 모듈
└── README.md
```

## 재현 흐름

```bash
conda activate etri
cd /mnt/c/etri-lifelog

# 후보 생성 예시
python scripts/run_etri_state_transition_candidates_20260505.py

# 제출 파일 검증 예시
python scripts/validate_submission.py sub_anchor_q6040_statepast_tw_b_20260505.csv
```

## 상세 기록

- [2026-05-05 state-transition prior 정리](experiments/2026-05-05_state_transition_prior.md)
- [실험 로그 JSON](experiments/log.json)

## 비고

원본 데이터와 제출 파일은 용량 및 대회 규정 관리를 위해 GitHub에 포함하지 않습니다.  
이 저장소는 실험 코드, 핵심 결과, 논문 작성에 활용할 수 있는 연구 기록을 중심으로 정리합니다.
