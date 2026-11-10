"""Live games as a measurement, not a judge (agent/FRAMEWORK.md, rule 4): each knowledge version's places with a 95%
interval, and how many games per version it takes to tell two versions apart.

    python agent/live_eval.py            prints the table; writes agent/dashboard/live_eval.json

A version is the memory the games were played with (games.jsonl "memory" hash). The spread of one place (SD) is
pooled over all games; telling a difference d apart at 5% / 80% power takes 2 * (1.96 + 0.84)^2 * SD^2 / d^2 games
per version (two-sample comparison of means).
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GAMES = ROOT / 'agent' / 'games.jsonl'
OUT = ROOT / 'agent' / 'dashboard' / 'live_eval.json'


def games():
    return [json.loads(l) for l in GAMES.read_text(encoding='utf-8').splitlines() if l.strip()] if GAMES.exists() else []


def needed(sd, d):
    return math.ceil(2 * (1.96 + 0.84) ** 2 * sd ** 2 / d ** 2)


def table(gs):
    places = [g['placement'] for g in gs if g.get('placement')]
    if len(places) < 2:
        return {'games': len(places), 'versions': []}
    mean = sum(places) / len(places)
    sd = math.sqrt(sum((p - mean) ** 2 for p in places) / (len(places) - 1))
    order, by = [], {}
    for g in gs:
        if not g.get('placement'):
            continue
        k = g.get('memory') or '?'
        if k not in by:
            order.append(k)
        by.setdefault(k, []).append(g)
    rows = []
    for k in order:
        ps = [g['placement'] for g in by[k]]
        m = sum(ps) / len(ps)
        half = 1.96 * sd / math.sqrt(len(ps))
        rows.append({'memory': k, 'games': [g['n'] for g in by[k]], 'places': ps, 'mean': round(m, 2),
                     'ci95': [round(max(1, m - half), 2), round(min(8, m + half), 2)], 'top2': sum(p <= 2 for p in ps),
                     'top4': sum(p <= 4 for p in ps)})
    return {'games': len(places), 'mean': round(mean, 2), 'sd': round(sd, 2),
            'games_per_version_to_detect': {'1 place': needed(sd, 1), '0.5 place': needed(sd, 0.5)},
            'versions': rows}


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    t = table(games())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(t, ensure_ascii=False, indent=1), encoding='utf-8')
    if not t.get('versions'):
        print('live evaluation: not enough games')
        return
    print(f"live evaluation: {t['games']} games, mean place {t['mean']}, SD {t['sd']}; to tell versions apart: "
          f"{t['games_per_version_to_detect']['1 place']} games each for 1 place, "
          f"{t['games_per_version_to_detect']['0.5 place']} for 0.5")
    for r in t['versions']:
        print(f"  memory {r['memory']}: {len(r['places'])} games, mean {r['mean']} (95% {r['ci95'][0]}-{r['ci95'][1]}), "
              f"top-2 {r['top2']}, top-4 {r['top4']}")


if __name__ == '__main__':
    main()
