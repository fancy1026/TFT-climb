"""The dashboard's data: agent/dashboard/games.json (the agent's games) and earlier.json (this account's games before
the agent), flat rows for the TFT Climb Agent dashboard (claude.ai artifact 5UM9qvc1tFyefb6QewwkB2).

    python agent/dashboard.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AGENT = ROOT / 'agent'
OUT = AGENT / 'dashboard'
TIER_NAMES = {'IRON': 'Iron', 'BRONZE': 'Bronze', 'SILVER': 'Silver', 'GOLD': 'Gold', 'PLATINUM': 'Platinum',
              'EMERALD': 'Emerald', 'DIAMOND': 'Diamond', 'MASTER': 'Master'}


def rank_label(r):
    if not r or not r.get('tier'):
        return None
    t = TIER_NAMES.get(r['tier'], r['tier'].title())
    return f"{t}{' ' + r['division'] if r.get('division') and r['tier'] not in ('MASTER',) else ''} {r.get('lp', 0)} LP"


def game_rows():
    rows = []
    f = AGENT / 'games.jsonl'
    for line in (f.read_text(encoding='utf-8').splitlines() if f.exists() else []):
        if not line.strip():
            continue
        g = json.loads(line)
        n = g.get('numbers') or {}
        hp, lv, gold, items = (n.get(k) or {} for k in ('hp', 'level', 'gold', 'items_on_board'))
        c = g.get('coach') or {}
        rows.append({
            'game': g.get('n'), 'run': g.get('run'), 'when': g.get('when'), 'comp': g.get('comp'),
            'place': g.get('placement'), 'top4': (1 if g.get('placement') and g['placement'] <= 4 else 0)
            if g.get('placement') else None,
            'lp_change': g.get('lp_change'), 'ladder': g.get('ladder_after'), 'rank': rank_label(g.get('rank_after')),
            'hp_3_1': hp.get('3-1'), 'hp_4_1': hp.get('4-1'), 'hp_5_1': hp.get('5-1'),
            'level_3_2': lv.get('3-2'), 'level_4_1': lv.get('4-1'), 'level_4_5': lv.get('4-5'),
            'gold_4_1': gold.get('4-1'), 'items_3_1': items.get('3-1'), 'items_4_1': items.get('4-1'),
            'carry_stars': n.get('carry_stars_final'), 'comp_done': n.get('comp_final'), 'last_stage': n.get('last_stage'),
            'failed_steps': n.get('failed_steps'), 'late_answers': n.get('late'),
            'review': c.get('summary'), 'changes': ' | '.join(c.get('changes') or []),
            'checks': ' | '.join(c.get('checks_next_game') or []), 'coach_minutes': g.get('coach_minutes'),
            'code': g.get('code'), 'gate': g.get('check_passed')})
    return rows


def earlier_rows():
    src = ROOT / 'agent' / 'history' / 'benchmark-metrics.json'
    if not src.exists():
        return []
    rows = []
    for g in json.loads(src.read_text(encoding='utf-8')):
        R = {r['stage']: r for r in g.get('rounds', [])}
        get = lambda s, k: (R.get(s) or {}).get(k)
        rows.append({'run': g['run'], 'model': g['model'].split(':')[-1], 'effort': g.get('effort'),
                     'comp': g.get('comp'), 'queue': g.get('queue'), 'place': g.get('placement'),
                     'hp_3_1': get('3-1', 'hp'), 'hp_4_1': get('4-1', 'hp'), 'level_4_1': get('4-1', 'level')})
    return rows


def lesson_rows():
    """The playbook's items (agent/memory/playbook.md): id, the lesson, when it applies, status, evidence."""
    import re
    f = AGENT / 'memory' / 'playbook.md'
    text = f.read_text(encoding='utf-8') if f.exists() else ''
    rows = []
    for m in re.finditer(r'^\[(P\d+)\]\s*(.*?)(?=^\[P\d+\]|\Z)', text, re.S | re.M):
        body = ' '.join(m.group(2).split())
        part = lambda key, nxt: (re.search(rf'{key}:\s*(.*?)(?=\s(?:{nxt}):|\Z)', body) or [None, None])[1]
        lesson = re.split(r'\sWhen:', body)[0].strip()
        rows.append({'id': m.group(1), 'lesson': lesson, 'when': part('When', 'Evidence|Status'),
                     'evidence': part('Evidence', 'Status'), 'status': part('Status', 'Evidence|When')})
    return rows


def learning_rows():
    """agent/learning.jsonl: one row per thing the agent learned or fixed - what, because of what, what changed."""
    f = AGENT / 'learning.jsonl'
    rows = [json.loads(l) for l in (f.read_text(encoding='utf-8').splitlines() if f.exists() else []) if l.strip()]
    for i, r in enumerate(rows):
        r['key'] = f"{r.get('game')}-{i + 1}"
    return rows


def live_rows():
    """Every game's live coach reviews (runs/<run>/live_coach.jsonl): what it saw each round and its notes."""
    rows = []
    f = AGENT / 'games.jsonl'
    for line in (f.read_text(encoding='utf-8').splitlines() if f.exists() else []):
        if not line.strip():
            continue
        g = json.loads(line)
        lf = ROOT / 'runs' / str(g.get('run')) / 'live_coach.jsonl'
        if not lf.exists():
            continue
        for i, l in enumerate(lf.read_text(encoding='utf-8').splitlines()):
            if l.strip():
                r = json.loads(l)
                rows.append({'key': f"{g.get('n')}-{i + 1}", 'game': g.get('n'), 'stage': r.get('stage'),
                             'seen': r.get('seen'), 'notes': r.get('notes'), 'seconds': r.get('seconds')})
    return rows


NAMES = {'none': 'no memory', '2af886c': 'memory before the demonstration'}


def judge_rows():
    """One row per verdict (agent/verdicts): the two versions' points on the held-out test cases, paired wins and
    losses, and whether it shipped. Ids and points only, as in the verdicts themselves."""
    rows = []
    for f in sorted((ROOT / 'agent' / 'verdicts').glob('*.json')):
        v = json.loads(f.read_text(encoding='utf-8'))
        base, cand = v['score']['base']['total'], v['score']['candidate']['total']
        prop = v.get('proposal')
        rows.append({'key': v['id'], 'when': v['when'][:16].replace('T', ' '),
                     'base': NAMES.get(v['base'], 'memory at ' + v['base'][:7] if v['base'] != 'none' else 'no memory'),
                     'candidate': ('proposal ' + prop) if prop else NAMES.get(v['candidate'], 'memory at ' + v['candidate'][:7]),
                     'base_points': base[0], 'candidate_points': cand[0], 'of': base[1], 'cases': v['cases'],
                     'samples': v.get('samples', 1), 'better': v['paired']['better'], 'worse': v['paired']['worse'],
                     'sign_p': v['paired']['sign_p'], 'kind': 'proposal' if prop else 'comparison',
                     # a comparison of two versions is a measurement; only a proposal can ship
                     'shipped': bool(prop and v.get('shipped', v['accept'])), 'provisional': v.get('provisional', False),
                     'reason': v['reason'] if prop else 'a measurement, not a proposal'})
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    jr = judge_rows()
    (OUT / 'judge.json').write_text(json.dumps(jr, ensure_ascii=False, indent=0), encoding='utf-8')
    (OUT / 'live.json').write_text(json.dumps(live_rows(), ensure_ascii=False, indent=0), encoding='utf-8')
    lr = learning_rows()
    (OUT / 'learning.json').write_text(json.dumps(lr, ensure_ascii=False, indent=0), encoding='utf-8')
    g, e, l = game_rows(), earlier_rows(), lesson_rows()
    (OUT / 'games.json').write_text(json.dumps(g, ensure_ascii=False, indent=0), encoding='utf-8')
    (OUT / 'earlier.json').write_text(json.dumps(e, ensure_ascii=False, indent=0), encoding='utf-8')
    (OUT / 'lessons.json').write_text(json.dumps(l, ensure_ascii=False, indent=0), encoding='utf-8')
    print(f'dashboard data: {len(g)} agent games, {len(e)} earlier games, {len(l)} lessons -> {OUT}')


if __name__ == '__main__':
    main()
