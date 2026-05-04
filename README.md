# 제5회 ETRI 휴먼이해 AI 논문경진대회

> **라이프로그 데이터를 활용한 수면 · 감정 · 스트레스 예측**  
> **DACON | ICTC 2026 | Average Log-Loss**

---

## 프로젝트 개요

이 프로젝트는 라이프로그 데이터를 기반으로 사용자의 **수면 상태**, **피로도**, **스트레스**를 예측하는  
**제5회 ETRI 휴먼이해 AI 논문경진대회** 참가 기록 및 실험 저장소입니다.

- **대회명**: 제5회 ETRI 휴먼이해 AI 논문경진대회
- **플랫폼**: DACON
- **목표**: 라이프로그 기반 다중 타겟 예측
- **평가 지표**: Average Log-Loss
- **최종 산출물**
  - 대회 제출용 예측 결과
  - 실험 로그 및 재현 가능한 코드
  - ICTC 2026 논문 초안

---

## 예측 태스크

| 타겟 | 문제 유형 | 설명 |
|------|-----------|------|
| Q1 | 이진 분류 | 취침 후 수면의 질 |
| Q2 | 이진 분류 | 취침 전 피로도 |
| Q3 | 이진 분류 | 스트레스 |
| S1 | 회귀 기반 이진화 | 총 수면시간 |
| S2 | 회귀 기반 이진화 | 수면 효율 |
| S3 | 회귀 기반 이진화 | 수면 지연시간 |
| S4 | 회귀 기반 이진화 | 수면 중 각성 시간 |

> **평가 메트릭:** Average Log-Loss  
> 값이 낮을수록 성능이 좋습니다.

---

## 프로젝트 구조

```bash
etri-lifelog/
├── data/
│   ├── raw/                 # 원본 데이터 (gitignore)
│   └── processed/           # 전처리 데이터 (gitignore)
├── notebooks/               # EDA 및 모델링 실험 노트북
├── src/
│   ├── features/            # 피처 엔지니어링 코드
│   ├── models/              # 모델 정의 및 학습 코드
│   └── utils/               # 공통 유틸리티
├── experiments/             # 실험 로그 및 기록
├── submissions/             # 제출 파일 (gitignore)
├── paper/                   # 논문 초안 및 정리 자료
└── scripts/
    ├── auto_push.py         # 실험 기록 + 커밋 + 푸시 자동화
    └── update_readme.py     # README 자동 업데이트
```

---

## 개발 환경

- **OS**: Windows 11 + WSL2 (Ubuntu)
- **Python**: 3.11
- **Conda 환경명**: `etri`
- **IDE**: Cursor (VSCode 기반)
- **작업 방식**: `.ipynb` 노트북 중심 실험 + Python 스크립트 자동화

---

## 환경 설정

```bash
conda env create -f environment.yml
conda activate etri
```

---

## 실험 기록 및 GitHub 푸시

실험이 끝난 뒤 아래 명령어로  
**실험 로그 저장 + README 업데이트 + Git commit + Git push**를 한 번에 수행할 수 있습니다.

### 기본 사용

```bash
python scripts/auto_push.py --msg "LightGBM baseline"
```

### 리더보드 점수 포함

```bash
python scripts/auto_push.py --msg "Q1 피처 추가 실험" --score 0.4823
```

### alias 사용 시

```bash
etri-push --msg "stable + shrink 0.8" --score 0.6080524417
```

---

## 실험 로그 관리

실험 기록은 아래 파일에 누적 저장됩니다.

```bash
experiments/log.json
```

README의 일부 영역은 `auto_push.py` 및 `update_readme.py`에 의해 자동으로 갱신됩니다.

---

## Leaderboard Summary
<!-- AUTO:SUMMARY:START -->
- 총 실험 수: **34**
- 오늘 업로드 수: **2**
- 최고 LB: **0.1000000000**  (2026-05-04, test)
- 최신 기록: **2026-05-04** / test
<!-- AUTO:SUMMARY:END -->

---

## Recent Experiments
<!-- AUTO:EXPERIMENTS:START -->
| 날짜 | 실험 내용 | LB | 태그 | 커밋 |
|---|---|---:|---|---|
| 2026-05-04 | test | 0.1000000000 | exp | 588f8ea |
| 2026-05-04 | ETRI 실험 자동화 정비 및 seed3 Q 0.6:0.4 후보 생성, 제출 전 검증 스크립트를 추가하고 CatBoost 임시 로그가 자동 커밋되지 않도록 auto_push를 보강 | - | exp | dce30e5 |
| 2026-05-02 | 새로운 피처 축 실험 3종 수행, rich mUsageStats는 0.59726345, minimal mUsageStats는 0.5967032976, target history prior는 0.596075678을 기록했으나 현재 최고인 3-seed ensemble + 타겟별 라우팅 0.5957800203은 넘지 못함, 내일부터는 기존 구조를 벗어난 완전 신규 접근으로 전환 예정 | 0.5957800203 | exp | 8ca0cfc |
| 2026-05-02 | 새로운 피처 축 실험, mUsageStats를 최소 count/ratio feature만 남겨 3-seed ensemble과 타겟별 블렌드 라우팅 구조에 추가하여 rich usage feature의 과적합 가능성 확인 | 0.5967032976 | exp | e2f5259 |
| 2026-05-02 | 새로운 피처 축 실험으로 mUsageStats 앱 사용 패턴 feature를 풍부하게 추가했으나 0.59726345로 악화, 앱 사용 정보는 과도하게 넣을 경우 노이즈와 과적합 가능성이 있음을 확인 | 0.5957800203 | exp | 06e6d36 |
| 2026-05-01 | seed ensemble 구조 실험 정리, 3-seed ensemble은 0.5957800203으로 큰 폭 개선되었으나 5-seed ensemble은 0.5963145855로 소폭 악화되어 현재는 3-seed + 타겟별 블렌드 라우팅 구성이 최적임을 확인 | 0.5957800203 | exp | 5256c8f |
| 2026-05-01 | 구조 실험에서 seed 3개 앙상블과 타겟별 블렌드 라우팅을 결합하여 0.5957800203 기록, 단일 seed 대비 큰 폭으로 개선되며 예측 안정화 효과 확인 | 0.5957800203 | exp | dc6e14a |
| 2026-05-01 | 타겟별 블렌드 라우팅 안정형 실험에서 Q 타겟 0.5:0.5, S 타겟 0.2:0.8, alpha 0.98 설정으로 0.5969444527를 기록하며 최고점 갱신, 다만 개선폭이 작아 다음 단계는 seed ensemble 등 구조적 실험으로 전환 | 0.5969444527 | exp | e984d4d |
| 2026-04-30 | 타겟별 블렌드 라우팅 실험에서 Q 타겟은 0.6:0.4, S 타겟은 0.2:0.8, alpha 0.98 설정으로 0.597019454를 기록하며 오늘 최고점 갱신, Q 계열은 더 LGB 중심이고 S 계열은 기존 CAT 중심 구성이 유효함을 확인 | 0.5970194540 | exp | 44bfcd1 |
| 2026-04-30 | 타겟별 블렌드 라우팅 실험에서 Q 타겟 0.4:0.6, S 타겟 0.2:0.8, alpha 0.98 설정으로 0.5970621139를 기록하며 이전 Q 0.3:0.7 설정보다 추가 개선됨을 확인 | 0.5970621139 | exp | b696eed |
<!-- AUTO:EXPERIMENTS:END -->

---

## Daily Upload Status
<!-- AUTO:DAILY:START -->
### unknown
- 업로드 수: **2**
- 최근 실험:
  - LightGBM 베이스라인 - 단순 센서 7개 피처 35개
  - LightGBM v3 - 시간대+개인화+hr상세 90피처, val split 개선 중

### 2026-05-04
- 업로드 수: **2**
- 당일 최고 LB: **0.1000000000**
- 최근 실험:
  - ETRI 실험 자동화 정비 및 seed3 Q 0.6:0.4 후보 생성, 제출 전 검증 스크립트를 추가하고 CatBoost 임시 로그가 자동 커밋되지 않도록 auto_push를 보강
  - test

### 2026-05-02
- 업로드 수: **3**
- 당일 최고 LB: **0.5957800203**
- 최근 실험:
  - 새로운 피처 축 실험으로 mUsageStats 앱 사용 패턴 feature를 풍부하게 추가했으나 0.59726345로 악화, 앱 사용 정보는 과도하게 넣을 경우 노이즈와 과적합 가능성이 있음을 확인
  - 새로운 피처 축 실험, mUsageStats를 최소 count/ratio feature만 남겨 3-seed ensemble과 타겟별 블렌드 라우팅 구조에 추가하여 rich usage feature의 과적합 가능성 확인
  - 새로운 피처 축 실험 3종 수행, rich mUsageStats는 0.59726345, minimal mUsageStats는 0.5967032976, target history prior는 0.596075678을 기록했으나 현재 최고인 3-seed ensemble + 타겟별 라우팅 0.5957800203은 넘지 못함, 내일부터는 기존 구조를 벗어난 완전 신규 접근으로 전환 예정

### 2026-05-01
- 업로드 수: **3**
- 당일 최고 LB: **0.5957800203**
- 최근 실험:
  - 타겟별 블렌드 라우팅 안정형 실험에서 Q 타겟 0.5:0.5, S 타겟 0.2:0.8, alpha 0.98 설정으로 0.5969444527를 기록하며 최고점 갱신, 다만 개선폭이 작아 다음 단계는 seed ensemble 등 구조적 실험으로 전환
  - 구조 실험에서 seed 3개 앙상블과 타겟별 블렌드 라우팅을 결합하여 0.5957800203 기록, 단일 seed 대비 큰 폭으로 개선되며 예측 안정화 효과 확인
  - seed ensemble 구조 실험 정리, 3-seed ensemble은 0.5957800203으로 큰 폭 개선되었으나 5-seed ensemble은 0.5963145855로 소폭 악화되어 현재는 3-seed + 타겟별 블렌드 라우팅 구성이 최적임을 확인

### 2026-04-30
- 업로드 수: **3**
- 당일 최고 LB: **0.5970194540**
- 최근 실험:
  - 타겟별 블렌드 라우팅 실험에서 Q 타겟은 0.3:0.7, S 타겟은 0.2:0.8, alpha 0.98 설정으로 0.5973702612를 기록하며 기존 공통 블렌드보다 개선됨을 확인
  - 타겟별 블렌드 라우팅 실험에서 Q 타겟 0.4:0.6, S 타겟 0.2:0.8, alpha 0.98 설정으로 0.5970621139를 기록하며 이전 Q 0.3:0.7 설정보다 추가 개선됨을 확인
  - 타겟별 블렌드 라우팅 실험에서 Q 타겟은 0.6:0.4, S 타겟은 0.2:0.8, alpha 0.98 설정으로 0.597019454를 기록하며 오늘 최고점 갱신, Q 계열은 더 LGB 중심이고 S 계열은 기존 CAT 중심 구성이 유효함을 확인

### 2026-04-29
- 업로드 수: **4**
- 당일 최고 LB: **0.5978689256**
- 최근 실험:
  - structure experiment + target-wise alpha split 결과 0.598044066으로 소폭 악화, 공통 alpha 0.98 + blend 0.2:0.8이 더 안정적임을 확인
  - structure experiment + small recent/change feature set added to stable baseline resulted in strong degradation to 0.606743338, confirming that the current task favors stable conservative setup over change-heavy features
  - 4월 29일 구조 실험 3종 수행: target-wise alpha split(0.598044066), small recent/change features(0.606743338), logit-space blending(0.5979976852) 모두 기존 최고를 넘지 못했고, stable baseline + alpha 0.98 + blend 0.2:0.8 + weighted probability average가 최적 구성으로 재확인됨

### 2026-04-27
- 업로드 수: **2**
- 당일 최고 LB: **0.5978689256**
- 최근 실험:
  - alpha 0.98 + blend 0.15:0.85는 0.5979019616으로 소폭 악화, 현재 최고 구성은 stable baseline + alpha 0.98 + blend 0.2:0.8 with 0.5978689256로 확정하고 실험을 마감, 이후 논문 작성 단계로 전환
  - 논문 초안 작성 시작, 연구 방향과 핵심 메시지 정리 및 실험 결과 기반 본문 구조 설계
<!-- AUTO:DAILY:END -->

---

## 날짜별 진행 기록
<!-- AUTO:TIMELINE:START -->
### unknown
- 진행 내용:
  - LightGBM 베이스라인 - 단순 센서 7개 피처 35개
  - LightGBM v3 - 시간대+개인화+hr상세 90피처, val split 개선 중
- 다음 이어서 할 일:
  - `LightGBM v3 - 시간대+개인화+hr상세 90피처, val split 개선 중` 기준으로 다음 실험 이어가기

### 2026-05-04
- 그날 최고 점수: **0.1000000000**
- 진행 내용:
  - ETRI 실험 자동화 정비 및 seed3 Q 0.6:0.4 후보 생성, 제출 전 검증 스크립트를 추가하고 CatBoost 임시 로그가 자동 커밋되지 않도록 auto_push를 보강
  - test (LB: 0.1000000000)
- 다음 이어서 할 일:
  - `test` 기준으로 다음 실험 이어가기

### 2026-05-02
- 그날 최고 점수: **0.5957800203**
- 진행 내용:
  - 새로운 피처 축 실험으로 mUsageStats 앱 사용 패턴 feature를 풍부하게 추가했으나 0.59726345로 악화, 앱 사용 정보는 과도하게 넣을 경우 노이즈와 과적합 가능성이 있음을 확인 (LB: 0.5957800203)
  - 새로운 피처 축 실험, mUsageStats를 최소 count/ratio feature만 남겨 3-seed ensemble과 타겟별 블렌드 라우팅 구조에 추가하여 rich usage feature의 과적합 가능성 확인 (LB: 0.5967032976)
  - 새로운 피처 축 실험 3종 수행, rich mUsageStats는 0.59726345, minimal mUsageStats는 0.5967032976, target history prior는 0.596075678을 기록했으나 현재 최고인 3-seed ensemble + 타겟별 라우팅 0.5957800203은 넘지 못함, 내일부터는 기존 구조를 벗어난 완전 신규 접근으로 전환 예정 (LB: 0.5957800203)
- 다음 이어서 할 일:
  - `새로운 피처 축 실험 3종 수행, rich mUsageStats는 0.59726345, minimal mUsageStats는 0.5967032976, target history prior는 0.596075678을 기록했으나 현재 최고인 3-seed ensemble + 타겟별 라우팅 0.5957800203은 넘지 못함, 내일부터는 기존 구조를 벗어난 완전 신규 접근으로 전환 예정` 기준으로 다음 실험 이어가기

### 2026-05-01
- 그날 최고 점수: **0.5957800203**
- 진행 내용:
  - 타겟별 블렌드 라우팅 안정형 실험에서 Q 타겟 0.5:0.5, S 타겟 0.2:0.8, alpha 0.98 설정으로 0.5969444527를 기록하며 최고점 갱신, 다만 개선폭이 작아 다음 단계는 seed ensemble 등 구조적 실험으로 전환 (LB: 0.5969444527)
  - 구조 실험에서 seed 3개 앙상블과 타겟별 블렌드 라우팅을 결합하여 0.5957800203 기록, 단일 seed 대비 큰 폭으로 개선되며 예측 안정화 효과 확인 (LB: 0.5957800203)
  - seed ensemble 구조 실험 정리, 3-seed ensemble은 0.5957800203으로 큰 폭 개선되었으나 5-seed ensemble은 0.5963145855로 소폭 악화되어 현재는 3-seed + 타겟별 블렌드 라우팅 구성이 최적임을 확인 (LB: 0.5957800203)
- 다음 이어서 할 일:
  - `seed ensemble 구조 실험 정리, 3-seed ensemble은 0.5957800203으로 큰 폭 개선되었으나 5-seed ensemble은 0.5963145855로 소폭 악화되어 현재는 3-seed + 타겟별 블렌드 라우팅 구성이 최적임을 확인` 기준으로 다음 실험 이어가기

### 2026-04-30
- 그날 최고 점수: **0.5970194540**
- 진행 내용:
  - 타겟별 블렌드 라우팅 실험에서 Q 타겟은 0.3:0.7, S 타겟은 0.2:0.8, alpha 0.98 설정으로 0.5973702612를 기록하며 기존 공통 블렌드보다 개선됨을 확인 (LB: 0.5973702612)
  - 타겟별 블렌드 라우팅 실험에서 Q 타겟 0.4:0.6, S 타겟 0.2:0.8, alpha 0.98 설정으로 0.5970621139를 기록하며 이전 Q 0.3:0.7 설정보다 추가 개선됨을 확인 (LB: 0.5970621139)
  - 타겟별 블렌드 라우팅 실험에서 Q 타겟은 0.6:0.4, S 타겟은 0.2:0.8, alpha 0.98 설정으로 0.597019454를 기록하며 오늘 최고점 갱신, Q 계열은 더 LGB 중심이고 S 계열은 기존 CAT 중심 구성이 유효함을 확인 (LB: 0.5970194540)
- 다음 이어서 할 일:
  - `타겟별 블렌드 라우팅 실험에서 Q 타겟은 0.6:0.4, S 타겟은 0.2:0.8, alpha 0.98 설정으로 0.597019454를 기록하며 오늘 최고점 갱신, Q 계열은 더 LGB 중심이고 S 계열은 기존 CAT 중심 구성이 유효함을 확인` 기준으로 다음 실험 이어가기

### 2026-04-29
- 그날 최고 점수: **0.5978689256**
- 진행 내용:
  - 5월 초까지 실험 중심으로 운영하기로 결정, 상위권 유지 및 top 60 목표 하에 핵심 실험 변화와 최고점 갱신 시 연구 로그를 함께 기록하는 방식으로 진행 (LB: 0.5978689256)
  - structure experiment + target-wise alpha split 결과 0.598044066으로 소폭 악화, 공통 alpha 0.98 + blend 0.2:0.8이 더 안정적임을 확인 (LB: 0.5978689256)
  - structure experiment + small recent/change feature set added to stable baseline resulted in strong degradation to 0.606743338, confirming that the current task favors stable conservative setup over change-heavy features (LB: 0.5978689256)
  - 4월 29일 구조 실험 3종 수행: target-wise alpha split(0.598044066), small recent/change features(0.606743338), logit-space blending(0.5979976852) 모두 기존 최고를 넘지 못했고, stable baseline + alpha 0.98 + blend 0.2:0.8 + weighted probability average가 최적 구성으로 재확인됨 (LB: 0.5978689256)
- 다음 이어서 할 일:
  - `4월 29일 구조 실험 3종 수행: target-wise alpha split(0.598044066), small recent/change features(0.606743338), logit-space blending(0.5979976852) 모두 기존 최고를 넘지 못했고, stable baseline + alpha 0.98 + blend 0.2:0.8 + weighted probability average가 최적 구성으로 재확인됨` 기준으로 다음 실험 이어가기

### 2026-04-27
- 그날 최고 점수: **0.5978689256**
- 진행 내용:
  - alpha 0.98 + blend 0.15:0.85는 0.5979019616으로 소폭 악화, 현재 최고 구성은 stable baseline + alpha 0.98 + blend 0.2:0.8 with 0.5978689256로 확정하고 실험을 마감, 이후 논문 작성 단계로 전환 (LB: 0.5978689256)
  - 논문 초안 작성 시작, 연구 방향과 핵심 메시지 정리 및 실험 결과 기반 본문 구조 설계 (LB: 0.5978689256)
- 다음 이어서 할 일:
  - `논문 초안 작성 시작, 연구 방향과 핵심 메시지 정리 및 실험 결과 기반 본문 구조 설계` 기준으로 다음 실험 이어가기

### 2026-04-26
- 그날 최고 점수: **0.5978689256**
- 진행 내용:
  - probe experiment + fixed alpha 0.98 + fixed blend 0.2:0.8 (LB: 0.5978689256)
  - final probe result 확인, alpha 0.98 + blend 0.1:0.9는 0.5980970493로 소폭 악화, 현재 최고 구성은 alpha 0.98 + blend 0.2:0.8 with 0.5978689256로 확정 (LB: 0.5978689256)
- 다음 이어서 할 일:
  - `final probe result 확인, alpha 0.98 + blend 0.1:0.9는 0.5980970493로 소폭 악화, 현재 최고 구성은 alpha 0.98 + blend 0.2:0.8 with 0.5978689256로 확정` 기준으로 다음 실험 이어가기

### 2026-04-25
- 그날 최고 점수: **0.5979394488**
- 진행 내용:
  - stable baseline + fixed alpha 0.92 + fixed blend 0.3:0.7 (LB: 0.5987808466)
  - probe experiment + fixed alpha 0.96 + fixed blend 0.2:0.8 (LB: 0.5979394488)
  - stable baseline 기반 conservative shrink와 CatBoost-heavy ensemble 탐색, alpha 0.96 + blend 0.2:0.8에서 0.5979394488로 최고점 갱신, 실험 종료 후 논문 초안 작성 단계로 전환 (LB: 0.5979394488)
- 다음 이어서 할 일:
  - `stable baseline 기반 conservative shrink와 CatBoost-heavy ensemble 탐색, alpha 0.96 + blend 0.2:0.8에서 0.5979394488로 최고점 갱신, 실험 종료 후 논문 초안 작성 단계로 전환` 기준으로 다음 실험 이어가기

### 2026-04-24
- 그날 최고 점수: **0.5991758604**
- 진행 내용:
  - stable baseline + fixed alpha 0.90 (LB: 0.6059629619)
  - stable baseline에 shrink alpha 0.90과 fixed blend 0.4:0.6 적용, 0.6000012518로 크게 개선되며 11등 진입 (LB: 0.6000012518)
  - stable baseline 유지, shrink alpha 0.90과 fixed blend 비율 조정 실험 진행, 0.3:0.7에서 0.5991758604로 최고점 갱신 (LB: 0.5991758604)
- 다음 이어서 할 일:
  - `stable baseline 유지, shrink alpha 0.90과 fixed blend 비율 조정 실험 진행, 0.3:0.7에서 0.5991758604로 최고점 갱신` 기준으로 다음 실험 이어가기

### 2026-04-23
- 그날 최고 점수: **0.6064374291**
- 진행 내용:
  - stable baseline + fixed alpha 0.80 (LB: 0.6076386989)
  - stable baseline + fixed alpha 0.78 (LB: 0.6082991747)
  - stable baseline에서 shrink alpha 미세조정 실험 진행, alpha 0.78은 악화, 0.80은 개선, 0.85에서 0.6064374291로 최고점 갱신 (LB: 0.6064374291)
- 다음 이어서 할 일:
  - `stable baseline에서 shrink alpha 미세조정 실험 진행, alpha 0.78은 악화, 0.80은 개선, 0.85에서 0.6064374291로 최고점 갱신` 기준으로 다음 실험 이어가기

### 2026-04-22
- 그날 최고 점수: **0.6080524417**
- 진행 내용:
  - 첫 자동화 테스트 (LB: 0.6080524417)
  - stable baseline 0.608 유지, strong ensemble 0.617 비교, 안정적 구조가 더 우세함 확인 및 README 자동화 정리 (LB: 0.6080524417)
  - stable baseline 0.608 유지, strong ensemble 0.617 비교, 복잡한 구조보다 안정적 구조가 더 우세함 확인 (LB: 0.6080524417)
  - stable baseline 0.608 유지, strong ensemble 0.617 비교, README 날짜별 기록 자동화 정리 (LB: 0.6080524417)
- 다음 이어서 할 일:
  - `stable baseline 0.608 유지, strong ensemble 0.617 비교, README 날짜별 기록 자동화 정리` 기준으로 다음 실험 이어가기
<!-- AUTO:TIMELINE:END -->

---

## 논문 작성

- **제출처**: ICTC 2026 EDAS
- **트랙**: IWETRIAI
- **형식**: IEEE 6-page Full Paper
- **마감일**: 2026-06-26

이 저장소는 대회 성능 향상뿐 아니라,  
최종적으로 **논문 제출용 결과 정리 및 재현 가능한 실험 관리**를 목표로 합니다.

---

## 비고

- 원본 데이터 및 제출 파일은 `.gitignore`로 관리합니다.
- 실험 결과는 가능한 한 재현 가능하도록 기록합니다.
- README는 실험 진행 상황에 따라 자동 갱신됩니다.
