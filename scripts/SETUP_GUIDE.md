# 초기 세팅 가이드 (WSL2에서 한 번만)

## 1. 이 폴더를 WSL2 홈에 복사

Windows에서 받은 파일을 WSL2로 이동:
```bash
# Windows 경로 예시: C:\Users\자윤\etri-lifelog
# WSL2에서:
cp -r /mnt/c/Users/<이름>/etri-lifelog ~/etri-lifelog
cd ~/etri-lifelog
```

또는 GitHub에서 바로 clone:
```bash
git clone https://github.com/<username>/etri-lifelog.git ~/etri-lifelog
cd ~/etri-lifelog
```

---

## 2. Conda 환경 생성

```bash
conda env create -f environment.yml
conda activate etri

# 설치 확인
python -c "import lightgbm, torch, optuna; print('OK')"
```

---

## 3. GitHub 레포 최초 연결

GitHub에서 빈 레포 `etri-lifelog` 생성 후:

```bash
cd ~/etri-lifelog
git init
git remote add origin https://github.com/<username>/etri-lifelog.git
git branch -M main

# 첫 푸시
git add -A
git commit -m "init: project structure"
git push -u origin main
```

---

## 4. GitHub 인증 (토큰 방식, 한 번만)

```
GitHub → Settings → Developer settings
→ Personal access tokens → Tokens (classic)
→ Generate new token
→ 권한: repo 전체 체크
→ 토큰 복사 (한 번만 보임!)
```

```bash
# WSL2에서 자격증명 저장
git config --global credential.helper store

# 첫 push 시 묻는 username/password에서:
# username: GitHub 아이디
# password: 위에서 복사한 토큰
# → 이후 자동으로 저장됨
```

---

## 5. 매일 사용법

```bash
conda activate etri

# 실험 후 → 이것만 치면 끝
python scripts/auto_push.py --msg "LightGBM Q1 피처 추가"

# 리더보드 제출 후 점수 나오면
python scripts/auto_push.py --msg "LightGBM baseline 제출" --score 0.4823
```

---

## 6. 데이터 위치

```bash
# 대회에서 받은 데이터 → data/raw/ 에 넣기
# (git에 올라가지 않음 - .gitignore 처리됨)
cp /mnt/c/Downloads/*.csv ~/etri-lifelog/data/raw/
```

---

## 디렉토리 설명

```
etri-lifelog/
├── data/
│   ├── raw/          ← 원본 CSV (gitignore)
│   └── processed/    ← 전처리 결과 (gitignore)
├── notebooks/        ← EDA, 실험 .ipynb
├── src/
│   ├── features/     ← 피처 엔지니어링 코드
│   ├── models/       ← 모델 클래스
│   └── utils/        ← 공통 함수
├── experiments/
│   └── log.json      ← 실험 기록 (자동 생성)
├── submissions/      ← 제출 CSV (gitignore)
├── paper/            ← 논문 초안
└── scripts/
    ├── auto_push.py  ← 깃허브 자동 푸시
    └── SETUP_GUIDE.md
```
