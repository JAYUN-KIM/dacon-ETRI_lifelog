import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path('/mnt/c/etri-lifelog')
LOG_PATH = ROOT / 'experiments' / 'log.json'
README_UPDATER = ROOT / 'scripts' / 'update_readme.py'
GITIGNORE = ROOT / '.gitignore'


def run(cmd, cwd=ROOT, check=True):
    print('>', ' '.join(cmd))
    return subprocess.run(cmd, cwd=str(cwd), check=check)


def ensure_dirs():
    (ROOT / 'experiments').mkdir(parents=True, exist_ok=True)
    (ROOT / 'scripts').mkdir(parents=True, exist_ok=True)


def ensure_gitignore():
    wanted = ['data/raw/data/', 'submissions/']
    existing = ''
    if GITIGNORE.exists():
        existing = GITIGNORE.read_text(encoding='utf-8')
    lines = existing.splitlines()
    changed = False
    for item in wanted:
        if item not in lines:
            lines.append(item)
            changed = True
    if changed:
        GITIGNORE.write_text('\n'.join([x for x in lines if x.strip()]) + '\n', encoding='utf-8')


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
    return data if isinstance(data, list) else []


def save_logs(logs):
    LOG_PATH.write_text(json.dumps(logs, ensure_ascii=False, indent=2), encoding='utf-8')


def get_head_commit_short():
    try:
        out = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=str(ROOT), text=True).strip()
        return out
    except Exception:
        return '-'


def append_log(message, score=None, tag='exp'):
    logs = load_logs()
    now = datetime.now()
    logs.append({
        'timestamp': now.strftime('%Y-%m-%d %H:%M:%S'),
        'date': now.strftime('%Y-%m-%d'),
        'message': message,
        'score': score,
        'tag': tag,
        'commit': get_head_commit_short(),
    })
    save_logs(logs)


def update_readme():
    if README_UPDATER.exists():
        run([sys.executable, str(README_UPDATER)])


def git_commit_and_push(message, score=None, no_push=False):
    ensure_gitignore()
    run(['git', 'add', '-A'])
    # CatBoost writes volatile training logs; keep experiment commits focused on code/docs.
    run(['git', 'reset', '--', 'catboost_info', 'scripts/catboost_info'], check=False)

    date_prefix = datetime.now().strftime('%Y-%m-%d')
    if score is None:
        commit_msg = f'[{date_prefix}] {message}'
    else:
        commit_msg = f'[{date_prefix}] {message} | LB {float(score):.10f}'

    staged = subprocess.run(['git', 'diff', '--cached', '--quiet'], cwd=str(ROOT)).returncode != 0
    if not staged:
        print('변경 사항이 없어 commit/push를 건너뜁니다.')
        return

    run(['git', 'commit', '-m', commit_msg])
    if not no_push:
        run(['git', 'push', 'origin', 'HEAD'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--msg', required=True, help='실험 설명')
    parser.add_argument('--score', type=float, default=None, help='LB 점수')
    parser.add_argument('--tag', default='exp', help='태그 예: safe, exp, best')
    parser.add_argument('--no-push', action='store_true', help='git push는 하지 않음')
    args = parser.parse_args()

    ensure_dirs()
    append_log(args.msg, args.score, args.tag)
    update_readme()
    git_commit_and_push(args.msg, args.score, no_push=args.no_push)
    print('완료되었습니다.')


if __name__ == '__main__':
    main()
