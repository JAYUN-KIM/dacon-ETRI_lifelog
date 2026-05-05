import json
from pathlib import Path


ROOT = Path("/mnt/c/etri-lifelog")
LOG_PATH = ROOT / "experiments" / "log.json"
README_PATH = ROOT / "README.md"

STATUS_START = "<!-- AUTO:PROJECT_STATUS:START -->"
STATUS_END = "<!-- AUTO:PROJECT_STATUS:END -->"


def load_logs():
    if not LOG_PATH.exists():
        return []
    try:
        data = json.loads(LOG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(data, dict):
        data = data.get("experiments", [])
    return data if isinstance(data, list) else []


def fmt_score(value):
    if value is None or value == "":
        return "-"
    try:
        return f"{float(value):.10f}"
    except Exception:
        return str(value)


def replace_block(text, start_marker, end_marker, content):
    start = text.find(start_marker)
    end = text.find(end_marker)
    if start == -1 or end == -1 or start > end:
        return text
    start += len(start_marker)
    return text[:start] + "\n" + content.rstrip() + "\n" + text[end:]


def build_status(logs):
    scored = [entry for entry in logs if entry.get("score") is not None]
    best = min(scored, key=lambda entry: float(entry["score"])) if scored else None

    lines = []
    if best:
        lines.append(f"- 최고 Public LB: **{fmt_score(best.get('score'))}**")
        lines.append(f"- 최신 최고점 갱신일: **{best.get('date', '-')}**")
    else:
        lines.append("- 최고 Public LB: **기록 전**")

    lines.append("- 핵심 개선 축: subject별 최근 타겟 상태 전이를 활용한 state-transition prior")
    lines.append("- 상세 실험 기록은 `experiments/` 디렉토리에 분리 보관")
    return "\n".join(lines)


def main():
    if README_PATH.exists():
        text = README_PATH.read_text(encoding="utf-8")
    else:
        text = "# DACON ETRI Lifelog AI Competition\n\n"

    if STATUS_START in text and STATUS_END in text:
        text = replace_block(text, STATUS_START, STATUS_END, build_status(load_logs()))

    README_PATH.write_text(text, encoding="utf-8")
    print(f"README updated: {README_PATH}")


if __name__ == "__main__":
    main()
