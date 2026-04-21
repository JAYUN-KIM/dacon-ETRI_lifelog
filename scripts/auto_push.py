#!/usr/bin/env python3
"""
ETRI 대회 실험 결과 자동 깃허브 푸시 스크립트

사용법:
    python scripts/auto_push.py --msg "LightGBM baseline"
    python scripts/auto_push.py --msg "Q1 피처 추가" --score 0.4823
"""
import subprocess
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(cmd: str, check: bool = True) -> str:
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, cwd=ROOT
    )
    if check and result.returncode != 0:
        print(f"[ERROR] cmd: {cmd}")
        print(result.stderr)
        sys.exit(1)
    return result.stdout.strip()


def update_experiment_log(msg: str, score: float | None) -> None:
    log_path = ROOT / "experiments" / "log.json"

    logs: list = []
    if log_path.exists():
        with open(log_path, encoding="utf-8") as f:
            logs = json.load(f)

    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "message": msg,
        "score": score,
        "git_branch": run("git rev-parse --abbrev-ref HEAD", check=False),
        "git_hash": run("git rev-parse --short HEAD", check=False) or "init",
    }
    logs.append(entry)

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)

    print(f"[LOG] 실험 기록 추가 → experiments/log.json")


def auto_push(msg: str, score: float | None) -> None:
    # 1. 실험 로그 업데이트
    update_experiment_log(msg, score)

    # 2. 커밋 메시지
    date_str = datetime.now().strftime("%Y-%m-%d")
    score_str = f" | LB {score:.4f}" if score is not None else ""
    commit_msg = f"[{date_str}] {msg}{score_str}"

    # 3. git add (data/, submissions/ 는 .gitignore로 자동 제외)
    run("git add -A")

    # 4. 변경사항 체크
    status = run("git status --porcelain", check=False)
    if not status:
        print("[INFO] 변경사항 없음 — 푸시를 스킵합니다.")
        return

    print(f"[GIT] 변경 파일:\n{status}\n")

    # 5. commit & push
    run(f'git commit -m "{commit_msg}"')
    run("git push origin HEAD")

    print(f"\n✅ 푸시 완료!")
    print(f"   커밋: {commit_msg}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ETRI 실험 자동 깃허브 푸시")
    parser.add_argument("--msg", required=True, help="실험 설명")
    parser.add_argument("--score", type=float, default=None, help="LB 점수 (낮을수록 좋음)")
    args = parser.parse_args()

    auto_push(args.msg, args.score)
