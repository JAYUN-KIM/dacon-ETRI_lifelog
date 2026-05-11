# DACON ETRI Lifelog AI Competition

라이프로그 센서 데이터를 활용해 수면, 피로도, 스트레스 관련 7개 이진 타겟을 예측하는 DACON 논문경진대회 프로젝트입니다.  
목표는 단순한 점수 개선뿐 아니라, 작은 데이터셋에서 과적합을 줄이고 개인별 상태 변화와 센서 기반 생활 맥락을 안정적으로 반영하는 실험 흐름을 남기는 것입니다.

## 프로젝트 개요

- 대회: 제5회 ETRI 휴먼이해 인공지능 논문경진대회
- 플랫폼: DACON
- 평가 지표: Average Log-Loss
- 데이터 규모: train 450행, test 250행
- 입력 데이터: subject별 일자 단위 라이프로그와 센서 parquet
- 예측 대상: Q1~Q3, S1~S4 총 7개 binary target

## 현재 성과

<!-- AUTO:PROJECT_STATUS:START -->
- 최고 Public LB: **0.5878818802**
- 최신 최고점 갱신일: **2026-05-11**
- 핵심 개선 축: 개인별 날짜/상태 prior, target-wise calibration, 보수적 앵커 블렌딩
- 최근 연구 축: 원천 센서 기반 raw-context feature와 공동 타겟 패턴 보정
- 상세 실험 기록: `experiments/log.json`
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

## 핵심 접근

1. 안정적인 일별 센서 집계  
   활동량, 조도, 화면 사용, 충전 상태, 심박, 보행 등 비교적 안정적인 센서를 일자 단위로 집계했습니다.

2. 보수적인 확률 보정  
   작은 데이터셋에서 log-loss가 확률 과신에 민감하므로 probability shrink와 clipping을 적용했습니다.

3. CatBoost-heavy ensemble과 target-wise routing  
   Q 계열과 S 계열의 모델 선호가 달라 LightGBM/CatBoost blend 비율을 타겟 그룹별로 다르게 적용했습니다.

4. 개인별 날짜 및 상태 prior  
   subject별 최근 타겟 상태가 test 구간으로 이어진다는 가설을 별도 prior로 구성했고, public leaderboard에서 큰 개선을 확인했습니다.

5. 원천 센서 기반 reset 실험  
   기존 feature table을 넘어 mAmbience, mGps, mUsageStats, mWifi, mBle, wLight, wPedo 세부 정보를 다시 집계해 raw-context 후보를 생성했습니다.

## 주요 인사이트

- feature를 무작정 늘리는 것보다 안정적인 prior와 보수적 보정이 더 강했습니다.
- Q 타겟과 S 타겟은 같은 방식으로 블렌딩하면 손해가 있었고, target-wise routing이 유효했습니다.
- seed ensemble은 작은 데이터셋에서 모델 분산을 줄였지만, seed 수를 무작정 늘리면 오히려 public 성능이 희석될 수 있었습니다.
- 개인별 날짜/상태 prior가 가장 큰 개선 축이었고, 이후에는 target별 보정 강도 조절이 중요했습니다.
- 원천 센서 기반 raw-context 모델은 새 축으로 검토했지만, 단독 모델보다는 앵커에 얇게 섞는 방식이 더 안전했습니다.

## 주요 코드

| 파일 | 역할 |
|---|---|
| `scripts/run_etri_target_history_seed3_routing_q5050_s2080_alpha098.py` | seed ensemble + target-wise routing 계열 anchor |
| `scripts/run_etri_state_transition_candidates_20260505.py` | state-transition prior 후보 생성 |
| `scripts/make_state_transition_blend_grid_20260505.py` | state prior blend 강도 탐색 |
| `scripts/make_subject_date_interpolation_prior_20260506.py` | subject-date interpolation prior 생성 |
| `scripts/make_0509_after_g030_candidates.py` | target별 reliability/scale 보정 후보 생성 |
| `scripts/make_0510_new_axis_candidates.py` | 공동 타겟 패턴 및 센서 KNN 후보 생성 |
| `scripts/make_0510_reset_raw_context_candidates.py` | 원천 센서 기반 raw-context reset 후보 생성 |
| `scripts/validate_submission.py` | 제출 파일 shape/null/range 검증 |

## 프로젝트 구조

```text
etri-lifelog/
├── data/                  # 원본/가공 데이터 및 제출 파일, gitignore
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
python scripts/make_0510_reset_raw_context_candidates.py

# 제출 파일 검증 예시
python scripts/validate_submission.py sub_0510_reset_rawctx_w010.csv
```

## 상세 기록

- 실험 로그: [`experiments/log.json`](experiments/log.json)
- 주요 스크립트: [`scripts/`](scripts/)

## 비고

원본 데이터셋과 제출 파일은 용량 및 대회 규정 관리를 위해 GitHub에 포함하지 않습니다.  
이 저장소는 실험 코드, 핵심 결과, 논문 작성에 활용할 수 있는 연구 로그 중심으로 정리합니다.
