"""The control room's data (agent/FRAMEWORK.md): every agent's session, the lead's timeline, and the board of problems
and improvements - what the owner reads to see the agents work and talk.

    python agent/console.py            -> agent/console/data.json (published with agent/console/index.html)

Sources, all files the agents already write:
- agent/reviews/<tag>-<role>.jsonl: each model session's stream (what it said, the tools it used, its final answer),
  with <tag>-<role>.brief.txt, the brief the coordinator or the lead gave it;
- agent/climb.log: the lead's timeline (roles started and finished, gates, guards, judge verdicts);
- agent/verdicts/, agent/proposals/: the judge's results and what was proposed;
- agent/coordinator-log.jsonl: the coordinator's own entries - problems found, decisions, owner's guidance.
"""
import json
import re
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AGENT = ROOT / 'agent'
REVIEWS = AGENT / 'reviews'
OUT = AGENT / 'console'
ROLES = {'analyst': 'Analyst', 'proposer': 'Researcher', 'harness_fixer': 'Coder', 'data': 'Data', 'reviewer': 'Reviewer', 'curator': 'Curator', 'reasoner': 'Reasoner',
         'checker': 'Checker',
         'coach': 'Coach (first design)'}
MAX_STEPS = 160


def short(text, n):
    text = re.sub(r'\s+', ' ', str(text or '')).strip()
    return text if len(text) <= n else text[:n - 1] + '…'


def tool_line(block):
    inp = block.get('input') or {}
    what = inp.get('description') or inp.get('file_path') or inp.get('pattern') or inp.get('command') or inp.get('url') or ''
    return f"{block.get('name')}: {short(what, 180)}"


def session(path):
    """One model session from its stream-json log."""
    name = path.name[:-len('.jsonl')]
    role = next((r for r in ROLES if name.endswith('-' + r)), None)
    if not role:
        return None
    tag = name[:-len(role) - 1]
    steps, final, model = [], None, None
    for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
        if not line.startswith('{'):
            continue
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get('type') == 'system' and e.get('subtype') == 'init':
            model = e.get('model')
        elif e.get('type') == 'assistant':
            for b in e.get('message', {}).get('content', []):
                if b.get('type') == 'text' and b.get('text', '').strip():
                    steps.append({'kind': 'say', 'text': short(b['text'], 1500)})
                elif b.get('type') == 'tool_use':
                    steps.append({'kind': 'tool', 'text': tool_line(b)})
        elif e.get('type') == 'result':
            final = e
    if len(steps) > MAX_STEPS:
        steps = steps[:MAX_STEPS // 2] + [{'kind': 'gap', 'text': f'{len(steps) - MAX_STEPS} steps left out'}] + steps[-MAX_STEPS // 2:]
    brief_file = path.with_suffix('.brief.txt')
    end = datetime.fromtimestamp(path.stat().st_mtime)
    minutes = round((final or {}).get('duration_ms', 0) / 60000, 1)
    answer = (final or {}).get('result') or ''
    return {'id': name, 'role': role, 'agent': ROLES[role], 'tag': tag, 'model': model,
            'ended': end.strftime('%Y-%m-%d %H:%M'), 'minutes': minutes,
            'cost': round((final or {}).get('total_cost_usd') or 0, 2),
            'brief': brief_file.read_text(encoding='utf-8') if brief_file.exists() else '',
            'answer': short(answer, 4000), 'steps': steps, 'finished': final is not None}


def timeline():
    """The lead's log lines that say what happened between agents."""
    log = AGENT / 'climb.log'
    if not log.exists():
        return []
    keep = re.compile(r'(game \d+:|game \d+ recorded|analyst:|proposer:|harness_fixer:|coach:|judge:|guard:|check:|'
                      r'agent/STOP|no game|Owner)')
    out = []
    for line in log.read_text(encoding='utf-8', errors='replace').splitlines():
        m = re.match(r'(\d\d:\d\d:\d\d) (.*)', line)
        if m and keep.search(m.group(2)) and 'roll-down' not in m.group(2):
            out.append({'time': m.group(1), 'text': short(m.group(2), 400)})
    return out[-400:]


def board():
    """Problems and improvements: each with who found it, its status and the evidence."""
    items = []
    for f in sorted((AGENT / 'verdicts').glob('*.json')):
        v = json.loads(f.read_text(encoding='utf-8'))
        if not v.get('proposal'):
            continue
        b, c = v['score']['base']['total'], v['score']['candidate']['total']
        status = ('shipped' if v.get('shipped', v['accept']) else 'waiting for the owner' if v.get('awaiting') == 'owner'
                  else 'provisional' if v.get('provisional') else 'rejected')
        prop_file = AGENT / 'proposals' / f"{v['proposal']}.json"
        prop = json.loads(prop_file.read_text(encoding='utf-8')) if prop_file.exists() else {}
        items.append({'title': short(prop.get('why') or v['proposal'], 260), 'found_by': {'analyst': 'Analyst',
                      'proposer': 'Proposer', 'owner': 'Owner'}.get(prop.get('author'), prop.get('author', '?')),
                      'kind': 'knowledge', 'status': status,
                      'detail': f"Judge: {b[0]} -> {c[0]} of {b[1]} points, {v['paired']['better']} better / "
                                f"{v['paired']['worse']} worse ({v.get('samples', 1)} samples). {v['reason']}",
                      'evidence': f"agent/verdicts/{f.name}", 'when': v['when'][:16].replace('T', ' ')})
    for f in sorted(REVIEWS.glob('*-harness_fixer.jsonl')):
        s = session(f)
        try:
            ans = json.loads(re.findall(r'\{.*\}', s['answer'])[-1])
        except (IndexError, ValueError, TypeError):
            continue
        for x in ans.get('fixed', []):
            items.append({'title': short(x, 260), 'found_by': 'Coder', 'kind': 'harness fault', 'status': 'fixed',
                          'detail': f"game run {s['tag']}", 'evidence': short(x.split()[0], 12), 'when': s['ended']})
        for x in ans.get('left', []):
            items.append({'title': short(x, 260), 'found_by': 'Coder', 'kind': 'harness fault', 'status': 'open',
                          'detail': f"game run {s['tag']}", 'evidence': f"agent/reviews/{s['tag']}-faults.md", 'when': s['ended']})
        for x in ans.get('strategy_notes', []):
            items.append({'title': short(x, 260), 'found_by': 'Coder', 'kind': 'note for the proposer',
                          'status': 'open', 'detail': f"game run {s['tag']}", 'evidence': '', 'when': s['ended']})
    log = AGENT / 'coordinator-log.jsonl'
    if log.exists():
        for line in log.read_text(encoding='utf-8').splitlines():
            if line.strip():
                e = json.loads(line)
                items.append({'title': e['title'], 'found_by': e.get('by', 'Coordinator'), 'kind': e.get('kind', 'problem'),
                              'status': e.get('status', 'open'), 'detail': e.get('detail', ''),
                              'evidence': e.get('evidence', ''), 'when': e.get('when', '')})
    return sorted(items, key=lambda i: i['when'], reverse=True)


def agents(sessions):
    roster = [('Coordinator', 'This chat: starts the agents, reads their work, reports to the owner.', None),
              ('Data', 'Code: reads stream videos and games into round-by-round state and decision cases.', None),
              ('Analyst', "Strong players' games (train only): decision types, tempo, patterns.", 'analyst'),
              ('Researcher', 'Why players did it, in what context; one change for the Judge, or none.', 'proposer'),
              ('Judge', 'Code: held-out decisions, paired comparison, ships or not.', None),
              ('Coder', 'After each game: harness fixes with regression tests.', 'harness_fixer'),
              ('Player', 'Opus 5.5 in the game, on a pinned knowledge version.', None),
              ('Coach (first design)', 'Games 1-21: reviewed and changed everything itself (retired).', 'coach')]
    out = []
    for name, job, role in roster:
        runs = [s for s in sessions if s['role'] == role] if role else []
        out.append({'name': name, 'job': job, 'runs': len(runs), 'last': runs[-1]['ended'] if runs else '',
                    'cost': round(sum(s['cost'] for s in runs), 2)})
    return out


def main():
    sessions = [s for s in (session(p) for p in sorted(REVIEWS.glob('*.jsonl'), key=lambda p: p.stat().st_mtime)) if s]
    import sys
    sys.path.insert(0, str(AGENT))
    from channel import read as channel
    data = {'generated': datetime.now().strftime('%Y-%m-%d %H:%M'), 'agents': agents(sessions), 'channel': channel(),
            'sessions': list(reversed(sessions)), 'timeline': timeline(), 'board': board(),
            'queue': json.loads((AGENT / 'queue.json').read_text(encoding='utf-8')) if (AGENT / 'queue.json').exists() else []}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'data.json').write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
    print(f"console: {len(sessions)} sessions, {len(data['timeline'])} timeline lines, {len(data['board'])} board items "
          f"-> {OUT / 'data.json'} ({(OUT / 'data.json').stat().st_size // 1024} kB)")


if __name__ == '__main__':
    main()
