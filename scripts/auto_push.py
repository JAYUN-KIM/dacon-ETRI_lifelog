#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime

ROOT = Path("/mnt/c/etri-lifelog")
LOG_PATH = ROOT / "experiments" / "log.json"
README_PATH = ROOT / "README.md"

DEFAULT_GITIGNORE_BLOCK = """
# ETRI competition local data
data/raw/data/
submissions/
""".strip()


def run(cmd, cwd=ROOT, capture=False, check=True):
    print("$", " ".join(cmd))
    if capture:
        return subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=check)
    return subprocess.run(cmd, cwd=cwd, check=check)


def ensure_git_repo():
    if not (ROOT / ".git").exists():
        raise RuntimeError(f"Git repository not found: {ROOT}")


def ensure_experiments_dir():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def ensure_log_file():
    ensure_experiments_dir()
    if not LOG_PATH.exists():
        LOG_PATH.write_text("[]\n", encoding="utf-8")


def load_logs():
    ensure_log_file()
    text = LOG_PATH.read_text(encoding="utf-8").strip()
    if not text:
        return []
    data = json.loads(text)
    if isinstance(data, dict):
        data = data.get("experiments", [])
    if not isinstance(data, list):
        raise ValueError("experiments/log.json must be a JSON list or {'experiments': [...]} format")
    return data


def save_logs(logs):
    LOG_PATH.write_text(json.dumps(logs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def ensure_gitignore():
    gitignore = ROOT / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text(DEFAULT_GITIGNORE_BLOCK + "\n", encoding="utf-8")
        return

    text = gitignore.read_text(encoding="utf-8")
    changed = False
    for rule in ["data/raw/data/", "submissions/"]:
        if rule not in text:
            text = text.rstrip() + "\n" + rule + "\n"
            changed = True
    if changed:
        gitignore.write_text(text, encoding="utf-8")


def get_head_commit_short():
    try:
        res = run(["git", "rev-parse", "--short", "HEAD"], capture=True, check=True)
        return res.stdout.strip()
    except Exception:
        return None


def detect_tag(score):
    if score in [None, ""]:
        return "exp"
    try:
        score = float(score)
    except Exception:
        return "exp"

    logs = load_logs()
    scored = [x for x in logs if x.get("score") not in [None, ""]]
    if not scored:
        return "best"
    best_so_far = min(float(x["score"]) for x in scored)
    return "best" if score < best_so_far else "exp"


def append_experiment_log(message, score=None, tag=None):
    logs = load_logs()
    now = datetime.now()

    if tag is None or tag == "":
        tag = detect_tag(score)

    entry = {
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
        "date": now.strftime("%Y-%m-%d"),
        "message": message,
        "score": None if score in [None, ""] else float(score),
        "tag": tag,
        "commit": get_head_commit_short() or "-"
    }
    logs.append(entry)
    save_logs(logs)
    print(f"log appended -> {LOG_PATH}")
    return entry


def update_readme():
    script = ROOT / "scripts" / "update_readme.py"
    if not script.exists():
        raise FileNotFoundError(f"update_readme.py not found: {script}")
    run([sys.executable, str(script)], cwd=ROOT)


def git_add_commit_push(message, score=None):
    date_prefix = datetime.now().strftime("%Y-%m-%d")
    commit_msg = f"[{date_prefix}] {message}"
    if score not in [None, ""]:
        commit_msg += f" | LB {float(score):.10f}"

    run(["git", "add", "-A"], cwd=ROOT)

    diff_check = run(["git", "diff", "--cached", "--quiet"], cwd=ROOT, check=False)
    if diff_check.returncode == 0:
        print("No staged changes to commit.")
        return

    run(["git", "commit", "-m", commit_msg], cwd=ROOT)
    run(["git", "push", "origin", "HEAD"], cwd=ROOT)


def parse_args():
    parser = argparse.ArgumentParser(description="Auto log + README update + git push for ETRI experiments")
    parser.add_argument("--msg", required=True, help="실험 설명")
    parser.add_argument("--score", default=None, help="리더보드 점수 예: 0.6080524417")
    parser.add_argument("--tag", default=None, help="safe / exp / fail / best 등 수동 지정")
    parser.add_argument("--no-push", action="store_true", help="git push 없이 log/README까지만")
    parser.add_argument("--no-readme", action="store_true", help="README 자동 갱신 생략")
    return parser.parse_args()


def main():
    args = parse_args()

    ensure_git_repo()
    ensure_gitignore()

    append_experiment_log(args.msg, score=args.score, tag=args.tag)

    if not args.no_readme:
        update_readme()

    if args.no_push:
        print("Done without git push (--no-push).")
        return

    git_add_commit_push(args.msg, score=args.score)
    print("All done.")


if __name__ == "__main__":
    main()
