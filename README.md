# 제5회 ETRI 휴먼이해 AI 논문경진대회

> 라이프로그 데이터를 활용한 수면·감정·스트레스 예측  
> DACON | ICTC 2026 | Average Log-Loss

## 태스크

| 타겟 | 유형 | 설명 |
|------|------|------|
| Q1 | 분류 | 취침 후 수면의 질 |
| Q2 | 분류 | 취침 전 피로도 |
| Q3 | 분류 | 스트레스 |
| S1 | 회귀→분류 | 총 수면시간 |
| S2 | 회귀→분류 | 수면 효율 |
| S3 | 회귀→분류 | 수면 지연시간 |
| S4 | 회귀→분류 | 수면 중 각성 시간 |

**평가 메트릭**: Average Log-Loss (낮을수록 좋음)

## 디렉토리 구조

```
etri-lifelog/
├── data/
│   ├── raw/          # 원본 데이터 (gitignore)
│   └── processed/    # 전처리 데이터 (gitignore)
├── notebooks/        # EDA, 실험 노트북
├── src/
│   ├── features/     # 피처 엔지니어링
│   ├── models/       # 모델 정의
│   └── utils/        # 공통 유틸
├── experiments/      # 실험 결과 로그
├── submissions/      # 제출 파일 (gitignore)
├── paper/            # 논문 초안
└── scripts/
    └── auto_push.py  # 깃허브 자동 푸시
```

## 환경 세팅

```bash
conda env create -f environment.yml
conda activate etri
```

## 실험 푸시

```bash
# 기본
python scripts/auto_push.py --msg "LightGBM baseline"

# 리더보드 점수 포함
python scripts/auto_push.py --msg "Q1 피처 추가 실험" --score 0.4823
```

## 실험 로그

<!-- 아래는 auto_push.py가 자동 업데이트 -->
`experiments/log.json` 참고

## 논문

- 제출처: ICTC 2026 EDAS (IWETRIAI 부문)
- 형식: IEEE 6-page full paper
- 마감: 2026.06.26
