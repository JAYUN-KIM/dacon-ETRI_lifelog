#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

ROOT = Path("/mnt/c/etri-lifelog")
LOG_PATH = ROOT / "experiments" / "log.json"
README_PATH = ROOT / "README.md"

SUMMARY_START = "<!-- AUTO:SUMMARY:START -->"
SUMMARY_END = "<!-- AUTO:SUMMARY:END -->"

EXP_START = "<!-- AUTO:EXPERIMENTS:START -->"
EXP_END = "<!-- AUTO:EXPERIMENTS:END -->"

DAILY_START = "<!-- AUTO:DAILY:START -->"
DAILY_END = "<!-- AUTO:DAILY:END -->"


def load_logs():
    if not LOG_PATH.exists():
        return []

    text = LOG_PATH.read_text(encoding="utf-8").strip()
    if not text:
        return []

    data = json.loads(text)
    if isinstance(data, dict):
        data = data.get("experiments", [])
    if not isinstance(data, list):
        return []
    return data


def fmt_score(x):
    if x is None or x == "":
        return "-"
    try:
        return f"{float(x):.10f}"
    except Exception:
        return str(x)


def build_summary(logs):
    if not logs:
        return "아직 기록된 실험이 없습니다.\n"

    scored = [x for x in logs if x.get("score") not in [None, ""]]
    best = min(scored, key=lambda x: float(x["score"])) if scored else None
    latest = logs[-1]

    today = datetime.now().strftime("%Y-%m-%d")
    today_logs = [x for x in logs if x.get("date") == today]

    lines = [
        f"- Total Experiments: **{len(logs)}**",
        f"- Today Uploads: **{len(today_logs)}**",
    ]
    if best:
        lines.append(
            f"- Best LB: **{fmt_score(best.get('score'))}**  "
            f"({best.get('date', '-')}, {best.get('message', '-')})"
        )
    lines.append(
        f"- Latest Run: **{latest.get('date', '-')}** / {latest.get('message', '-')}"
    )
    return "\n".join(lines) + "\n"


def build_recent_experiments(logs, limit=10):
    if not logs:
        return "최근 실험 기록이 없습니다.\n"

    rows = []
    rows.append("| Date | Message | LB | Tag | Commit |")
    rows.append("|---|---|---:|---|---|")

    for x in reversed(logs[-limit:]):
        commit = x.get("commit", "-")
        if commit and commit != "-":
            commit = commit[:7]
        rows.append(
            f"| {x.get('date', '-')} | "
            f"{x.get('message', '-')} | "
            f"{fmt_score(x.get('score'))} | "
            f"{x.get('tag', '-')} | "
            f"{commit} |"
        )
    return "\n".join(rows) + "\n"


def build_daily_section(logs, limit=7):
    if not logs:
        return "일일 기록이 없습니다.\n"

    grouped = defaultdict(list)
    for x in logs:
        grouped[x.get("date", "unknown")].append(x)

    days = sorted(grouped.keys(), reverse=True)[:limit]
    lines = []

    for d in days:
        items = grouped[d]
        scored = [x for x in items if x.get("score") not in [None, ""]]
        best_score = min(float(x["score"]) for x in scored) if scored else None

        lines.append(f"### {d}")
        lines.append(f"- Uploads: **{len(items)}**")
        if best_score is not None:
            lines.append(f"- Best of Day: **{best_score:.10f}**")

        for x in items[-3:]:
            msg = x.get("message", "-")
            score = fmt_score(x.get("score"))
            tag = x.get("tag", "-")
            lines.append(f"  - [{tag}] {msg} / LB: {score}")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def replace_block(text, start_marker, end_marker, new_content):
    start_idx = text.find(start_marker)
    end_idx = text.find(end_marker)

    if start_idx == -1 or end_idx == -1 or end_idx < start_idx:
        return text

    start_idx += len(start_marker)
    return text[:start_idx] + "\n" + new_content + text[end_idx:]


def ensure_readme():
    if README_PATH.exists():
        return

    template = f"""# Dacon ETRI Lifelog

실험 코드와 기록을 관리하는 저장소입니다.

## Leaderboard Summary
{SUMMARY_START}
{SUMMARY_END}

## Recent Experiments
{EXP_START}
{EXP_END}

## Daily Upload Status
{DAILY_START}
{DAILY_END}
"""
    README_PATH.write_text(template, encoding="utf-8")


def main():
    ensure_readme()
    logs = load_logs()

    text = README_PATH.read_text(encoding="utf-8")
    text = replace_block(text, SUMMARY_START, SUMMARY_END, build_summary(logs))
    text = replace_block(text, EXP_START, EXP_END, build_recent_experiments(logs, limit=10))
    text = replace_block(text, DAILY_START, DAILY_END, build_daily_section(logs, limit=7))

    README_PATH.write_text(text, encoding="utf-8")
    print(f"README updated: {README_PATH}")


if __name__ == "__main__":
    main()
