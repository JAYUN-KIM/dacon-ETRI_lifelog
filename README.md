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
- 총 실험 수: **8**
- 오늘 업로드 수: **2**
- 최고 LB: **0.6076386989**  (2026-04-23, stable baseline + fixed alpha 0.80)
- 최신 기록: **2026-04-23** / stable baseline + fixed alpha 0.78
<!-- AUTO:SUMMARY:END -->

---

## Recent Experiments
<!-- AUTO:EXPERIMENTS:START -->
| 날짜 | 실험 내용 | LB | 태그 | 커밋 |
|---|---|---:|---|---|
| 2026-04-23 | stable baseline + fixed alpha 0.78 | 0.6082991747 | exp | 87dee83 |
| 2026-04-23 | stable baseline + fixed alpha 0.80 | 0.6076386989 | exp | cbdec5e |
| 2026-04-22 | stable baseline 0.608 유지, strong ensemble 0.617 비교, README 날짜별 기록 자동화 정리 | 0.6080524417 | exp | 49e3ce0 |
| 2026-04-22 | stable baseline 0.608 유지, strong ensemble 0.617 비교, 복잡한 구조보다 안정적 구조가 더 우세함 확인 | 0.6080524417 | exp | 4cf6cf8 |
| 2026-04-22 | stable baseline 0.608 유지, strong ensemble 0.617 비교, 안정적 구조가 더 우세함 확인 및 README 자동화 정리 | 0.6080524417 | exp | 39d7741 |
| 2026-04-22 | 첫 자동화 테스트 | 0.6080524417 | best | 39d7741 |
| - | LightGBM v3 - 시간대+개인화+hr상세 90피처, val split 개선 중 | - | - | - |
| - | LightGBM 베이스라인 - 단순 센서 7개 피처 35개 | - | - | - |
<!-- AUTO:EXPERIMENTS:END -->

---

## Daily Upload Status
<!-- AUTO:DAILY:START -->
### unknown
- 업로드 수: **2**
- 최근 실험:
  - LightGBM 베이스라인 - 단순 센서 7개 피처 35개
  - LightGBM v3 - 시간대+개인화+hr상세 90피처, val split 개선 중

### 2026-04-23
- 업로드 수: **2**
- 당일 최고 LB: **0.6076386989**
- 최근 실험:
  - stable baseline + fixed alpha 0.80
  - stable baseline + fixed alpha 0.78

### 2026-04-22
- 업로드 수: **4**
- 당일 최고 LB: **0.6080524417**
- 최근 실험:
  - stable baseline 0.608 유지, strong ensemble 0.617 비교, 안정적 구조가 더 우세함 확인 및 README 자동화 정리
  - stable baseline 0.608 유지, strong ensemble 0.617 비교, 복잡한 구조보다 안정적 구조가 더 우세함 확인
  - stable baseline 0.608 유지, strong ensemble 0.617 비교, README 날짜별 기록 자동화 정리
<!-- AUTO:DAILY:END -->

---

## 날짜별 진행 기록
<!-- AUTO:TIMELINE:START -->
### unknown 어디까지 했는지
- 진행 내용:
  - LightGBM 베이스라인 - 단순 센서 7개 피처 35개
  - LightGBM v3 - 시간대+개인화+hr상세 90피처, val split 개선 중
- 다음 이어서 할 일:
  - `LightGBM v3 - 시간대+개인화+hr상세 90피처, val split 개선 중` 기준으로 다음 실험 이어가기

### 2026-04-23 어디까지 했는지
- 그날 최고 점수: **0.6076386989**
- 진행 내용:
  - stable baseline + fixed alpha 0.80 (LB: 0.6076386989)
  - stable baseline + fixed alpha 0.78 (LB: 0.6082991747)
- 다음 이어서 할 일:
  - `stable baseline + fixed alpha 0.78` 기준으로 다음 실험 이어가기

### 2026-04-22 어디까지 했는지
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
