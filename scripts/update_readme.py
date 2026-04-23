import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

ROOT = Path('/mnt/c/etri-lifelog')
LOG_PATH = ROOT / 'experiments' / 'log.json'
README_PATH = ROOT / 'README.md'

SUMMARY_START = '<!-- AUTO:SUMMARY:START -->'
SUMMARY_END = '<!-- AUTO:SUMMARY:END -->'
EXP_START = '<!-- AUTO:EXPERIMENTS:START -->'
EXP_END = '<!-- AUTO:EXPERIMENTS:END -->'
DAILY_START = '<!-- AUTO:DAILY:START -->'
DAILY_END = '<!-- AUTO:DAILY:END -->'
TIMELINE_START = '<!-- AUTO:TIMELINE:START -->'
TIMELINE_END = '<!-- AUTO:TIMELINE:END -->'


def load_logs():
    if not LOG_PATH.exists():
        return []
    try:
        with open(LOG_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return []

    if isinstance(data, dict):
        data = data.get('experiments', [])
    if not isinstance(data, list):
        return []
    return data


def fmt_score(x):
    if x is None or x == '':
        return '-'
    try:
        return f"{float(x):.10f}"
    except Exception:
        return str(x)


def replace_block(text, start_marker, end_marker, new_content):
    start_idx = text.find(start_marker)
    end_idx = text.find(end_marker)
    if start_idx == -1 or end_idx == -1 or start_idx > end_idx:
        return text
    start_idx += len(start_marker)
    return text[:start_idx] + '\n' + new_content.rstrip() + '\n' + text[end_idx:]


def ensure_markers(text):
    additions = []
    if SUMMARY_START not in text or SUMMARY_END not in text:
        additions.append(f"## Leaderboard Summary\n{SUMMARY_START}\n{SUMMARY_END}\n")
    if EXP_START not in text or EXP_END not in text:
        additions.append(f"## Recent Experiments\n{EXP_START}\n{EXP_END}\n")
    if DAILY_START not in text or DAILY_END not in text:
        additions.append(f"## Daily Upload Status\n{DAILY_START}\n{DAILY_END}\n")
    if TIMELINE_START not in text or TIMELINE_END not in text:
        additions.append(f"## 날짜별 진행 기록\n{TIMELINE_START}\n{TIMELINE_END}\n")
    if additions:
        text = text.rstrip() + '\n\n' + '\n'.join(additions)
    return text


def build_summary(logs):
    if not logs:
        return '아직 기록된 실험이 없습니다.'

    scored = [x for x in logs if x.get('score') is not None]
    best = min(scored, key=lambda x: float(x['score'])) if scored else None
    latest = logs[-1]
    today = datetime.now().strftime('%Y-%m-%d')
    today_logs = [x for x in logs if x.get('date') == today]

    lines = [
        f"- 총 실험 수: **{len(logs)}**",
        f"- 오늘 업로드 수: **{len(today_logs)}**",
    ]
    if best:
        lines.append(
            f"- 최고 LB: **{fmt_score(best.get('score'))}**  ({best.get('date', '-')}, {best.get('message', '-')})"
        )
    lines.append(
        f"- 최신 기록: **{latest.get('date', '-')}** / {latest.get('message', '-')}"
    )
    return '\n'.join(lines)


def build_recent_experiments(logs, limit=10):
    if not logs:
        return '최근 실험 기록이 없습니다.'

    rows = [
        '| 날짜 | 실험 내용 | LB | 태그 | 커밋 |',
        '|---|---|---:|---|---|'
    ]
    for x in reversed(logs[-limit:]):
        commit = x.get('commit', '-')
        commit = commit[:7] if commit and commit != '-' else '-'
        rows.append(
            f"| {x.get('date', '-')} | {x.get('message', '-')} | {fmt_score(x.get('score'))} | {x.get('tag', '-')} | {commit} |"
        )
    return '\n'.join(rows)


def build_daily_status(logs, limit=7):
    if not logs:
        return '일일 기록이 없습니다.'

    grouped = defaultdict(list)
    for x in logs:
        grouped[x.get('date', 'unknown')].append(x)

    days = sorted(grouped.keys(), reverse=True)[:limit]
    lines = []
    for d in days:
        items = grouped[d]
        scored = [x for x in items if x.get('score') is not None]
        best_score = min(float(x['score']) for x in scored) if scored else None
        lines.append(f"### {d}")
        lines.append(f"- 업로드 수: **{len(items)}**")
        if best_score is not None:
            lines.append(f"- 당일 최고 LB: **{best_score:.10f}**")
        last_msgs = items[-3:]
        if last_msgs:
            lines.append('- 최근 실험:')
            for entry in last_msgs:
                lines.append(f"  - {entry.get('message', '-')}")
        lines.append('')
    return '\n'.join(lines).strip()


def build_timeline(logs, limit=14):
    if not logs:
        return '날짜별 진행 기록이 없습니다.'

    grouped = defaultdict(list)
    for x in logs:
        grouped[x.get('date', 'unknown')].append(x)

    days = sorted(grouped.keys(), reverse=True)[:limit]
    blocks = []
    for d in days:
        items = grouped[d]
        scored = [x for x in items if x.get('score') is not None]
        best = min(scored, key=lambda x: float(x['score'])) if scored else None

        blocks.append(f"### {d}")
        if best:
            blocks.append(f"- 그날 최고 점수: **{fmt_score(best.get('score'))}**")
        blocks.append('- 진행 내용:')
        for entry in items:
            msg = entry.get('message', '-')
            score = entry.get('score')
            if score is not None:
                blocks.append(f"  - {msg} (LB: {fmt_score(score)})")
            else:
                blocks.append(f"  - {msg}")

        latest_msg = items[-1].get('message', '기록 없음')
        blocks.append('- 다음 이어서 할 일:')
        blocks.append(f"  - `{latest_msg}` 기준으로 다음 실험 이어가기")
        blocks.append('')
    return '\n'.join(blocks).strip()


def main():
    logs = load_logs()

    if README_PATH.exists():
        text = README_PATH.read_text(encoding='utf-8')
    else:
        text = '# Dacon ETRI Lifelog\n\n'

    text = ensure_markers(text)
    text = replace_block(text, SUMMARY_START, SUMMARY_END, build_summary(logs))
    text = replace_block(text, EXP_START, EXP_END, build_recent_experiments(logs, limit=10))
    text = replace_block(text, DAILY_START, DAILY_END, build_daily_status(logs, limit=7))
    text = replace_block(text, TIMELINE_START, TIMELINE_END, build_timeline(logs, limit=14))

    README_PATH.write_text(text, encoding='utf-8')
    print(f'README updated: {README_PATH}')


if __name__ == '__main__':
    main()
