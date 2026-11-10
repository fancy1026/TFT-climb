"""The domain expert's feedback: a section of the framework of its own (the owner, 2026-10-30: "have another section in
the framework, which is from domain expert feedback ... Sometimes I would watch some rounds of combat and provide some
feedback, which is also another way of improving").

It goes to the Coordinator, as changes to the framework from outside it - never to the player (the owner, 2026-10-30:
"my feedback should be communicated to the coordinator as external changes to the framework, not the player"). No role
may change it. The Coordinator reads each point and changes the framework so that it finds and fixes that kind of fault
itself - a role's criteria, a packet's or the state's data, a metric, a check, the harness - and records the change
against the point.

It comes first (the owner, 2026-11-02: "my feedback should override the reviewer as domain expert found truth"; "this
should be prioritized, the other agents could disagree, or point out the problems but need to state the reasons"):
- the owner's rules for a comp (agent/expert/rules/<comp>.json, written by the Coordinator from the owner's points) are
  the first section of the comp notes the player reads (comp_coach.render);
- the owner's grades of a game's decisions (a point's "grades") replace the Reviewer's in that game's audits wherever
  the audits are read (apply: the stage scores, the diff reports, the report page), the Reviewer's kept beside them;
- a role that disagrees with one, or sees a problem in it, says so with its reasons in its final JSON
  ("owner_disagreements"); code writes them to agent/expert/disagreements.jsonl for the Coordinator to put to the owner.
  The point stays in force until the owner changes it.

agent/expert/feedback.jsonl: one entry per piece of feedback - the owner's words ("text"), the game and rounds watched,
and its points (kind harness | strategy | framework, the point in the owner's words, the evidence the Coordinator found,
"grades": [{"round", "area", "grade"}] when the point grades a decision, and "changes": what the framework changed for
it, with the commit).
`python agent/expert.py` lists the points, what was changed for each and the roles' disagreements; `add <entry.json>`
writes a new entry; `change <point id> "<what changed (commit)>"` records a change.
"""
import json
import re
import sys
from datetime import datetime
from pathlib import Path

AGENT = Path(__file__).resolve().parent
FEEDBACK = AGENT / 'expert' / 'feedback.jsonl'
RULES = AGENT / 'expert' / 'rules'
DISAGREEMENTS = AGENT / 'expert' / 'disagreements.jsonl'
KINDS = ('harness', 'strategy', 'framework')
GRADES = ('very good', 'good', 'ok', 'bad', 'very bad')


def rows():
    if not FEEDBACK.exists():
        return []
    return [json.loads(l) for l in FEEDBACK.read_text(encoding='utf-8').splitlines() if l.strip()]


def save(entries):
    FEEDBACK.parent.mkdir(parents=True, exist_ok=True)
    FEEDBACK.write_text(''.join(json.dumps(e, ensure_ascii=False) + '\n' for e in entries), encoding='utf-8')


def add(entry):
    """A new piece of feedback -> its id (x<n>); its points get ids x<n>.<m>."""
    entries = rows()
    xid = f'x{len(entries) + 1}'
    pts = []
    for m, p in enumerate(entry.get('points') or [], 1):
        if p.get('kind') not in KINDS:
            raise ValueError(f"point {m}: kind must be one of {KINDS}, not {p.get('kind')!r}")
        pts.append({'id': f'{xid}.{m}', **p, 'changes': list(p.get('changes') or [])})
    entries.append({'id': xid, 'when': entry.get('when') or datetime.now().strftime('%Y-%m-%d %H:%M'),
                    **{k: v for k, v in entry.items() if k not in ('id', 'points', 'when')}, 'points': pts})
    save(entries)
    return xid


MEMORY_FEEDBACK = AGENT / 'memory' / 'feedback'


def point_comp(e, p, games):
    """The comp a strategy point is about: its own "comp", else the comp its game played (agent/games.jsonl: a flex
    game's committed comp), else every comp."""
    if p.get('comp'):
        return p['comp']
    g = games.get(('run', e.get('run'))) or games.get(('n', e.get('game'))) or {}
    return g.get('comp_played') or (g.get('comp') if g.get('comp') not in (None, 'flex') else None) or GENERAL


def render_feedback():
    """The owner's feedback the player reads (the owner, 2026-11-07: "for each comp, there should accumulate a file,
    records the feedbacks, and then the model would see it, as another way to avoid the mistakes"): every strategy
    point, by game and round, in agent/memory/feedback/<comp>.md for the comp it is about - Every_comp.md for the
    points on every comp. Written before every game (agent/climb.py play); nothing is deleted. -> the files written."""
    games = {}
    try:
        for line in (AGENT / 'games.jsonl').read_text(encoding='utf-8').splitlines():
            g = json.loads(line) if line.strip() else {}
            games[('run', g.get('run'))] = games[('n', g.get('n'))] = g
    except (OSError, ValueError):
        pass
    by = {}
    for e in rows():
        for p in e.get('points') or []:
            if p.get('kind') == 'strategy' and p.get('text'):
                rounds = ', '.join(p.get('rounds') or [])
                by.setdefault(point_comp(e, p, games), []).append(
                    (f"- game {e['game']}" if e.get('game') else '- a test game') + (f" ({rounds})" if rounds else '')
                    + f": {p['text']}")
    MEMORY_FEEDBACK.mkdir(parents=True, exist_ok=True)
    written = []
    for comp, lines in by.items():
        f = MEMORY_FEEDBACK / f'{slug(comp)}.md'
        head = ("The owner's feedback on our earlier games, for every comp - mistakes found, by game and round; do not "
                "repeat them:" if comp == GENERAL else
                f"The owner's feedback on our earlier games with {comp} - mistakes found, by game and round; do not "
                f"repeat them:")
        f.write_text(head + '\n' + '\n'.join(lines) + '\n', encoding='utf-8')
        written.append(f)
    return written


def change(pid, what):
    """The Coordinator changed the framework for point `pid` -> the point."""
    entries = rows()
    for e in entries:
        for p in e.get('points') or []:
            if p['id'] == pid:
                p.setdefault('changes', []).append({'when': datetime.now().strftime('%Y-%m-%d %H:%M'), 'what': what})
                save(entries)
                return p
    raise KeyError(pid)


def open_points():
    """Points no framework change has been recorded for yet -> [(entry, point)]."""
    return [(e, p) for e in rows() for p in e.get('points') or [] if not p.get('changes')]


def where(e, p):
    rounds = p.get('rounds') or e.get('rounds') or []
    return (f"game {e.get('game')} (run {e.get('run')}" + (f", {', '.join(rounds)}" if rounds else '') + ')'
            if e.get('run') else 'no game named')


def slug(comp):
    return re.sub(r'[^A-Za-z0-9]+', '_', comp or '').strip('_')


GENERAL = 'Every comp'               # the owner's rules for every comp (x19.2: a comp's own rules stay in its file)


def rules(comp, general=True):
    """The owner's rules for a comp -> {'rules': [{'id', 'rule', 'why', 'from'}], 'banned_augments': [names]} - its
    own rules, then (general) the rules for every comp (agent/expert/rules/Every_comp.json)."""
    def read(name):
        try:
            return json.loads((RULES / f'{slug(name)}.json').read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return {}
    d = read(comp)
    own = [r for r in d.get('rules') or [] if r.get('rule')]
    every = [r for r in read(GENERAL).get('rules') or [] if r.get('rule')] if general and comp != GENERAL else []
    return {'rules': own + every, 'banned_augments': list(d.get('banned_augments') or [])}


def grades(run):
    """The owner's grades of one game's decisions -> [{'round', 'area', 'grade', 'why', 'point'}]; and the owner's
    words on whole stages -> {stage: words}."""
    out, stages = [], {}
    for e in rows():
        if e.get('run') != run:
            continue
        for s, words in (e.get('stages') or {}).items():
            stages[str(s)] = words
        for p in e.get('points') or []:
            for g in p.get('grades') or []:
                if g.get('round') and g.get('area') and g.get('grade') in GRADES:
                    out.append({'round': g['round'], 'area': g['area'], 'grade': g['grade'],
                                'why': g.get('why') or p['text'], 'point': p['id']})
    return out, stages


def apply(run, rounds):
    """A game's audited rounds with the owner's grades in place of the Reviewer's: the Reviewer's grade and reasons
    stay in the decision as "reviewer_grade"/"reviewer_why"; an owner grade for an area the Reviewer did not grade is
    added; a round the audits lack is added. -> the rounds (new list, the input unchanged)."""
    owner, _ = grades(run)
    if not owner:
        return rounds
    out = [dict(x, decisions=[dict(d) for d in x.get('decisions') or [] if isinstance(d, dict)])
           for x in rounds if isinstance(x, dict)]
    def names(text):
        return set(re.findall(r"[A-Z][a-z']{3,}", text or ''))
    for key in dict.fromkeys((g['round'], g['area']) for g in owner):
        rnd = next((x for x in out if x.get('round') == key[0]), None)
        if rnd is None:
            rnd = {'round': key[0], 'decisions': []}
            out.append(rnd)
        mine = [g for g in owner if (g['round'], g['area']) == key]
        same = [d for d in rnd['decisions'] if str(d.get('area') or '').lower() == key[1]]
        # each grade replaces the Reviewer's decision it is about: pairs sharing most names (Ionic, Archangel...) first
        pairs = sorted(((len(names(g['why']) & names(f"{d.get('why')} {d.get('better')}")), -i, -j)
                        for i, g in enumerate(mine) for j, d in enumerate(same)), reverse=True)
        took, used = {}, set()
        for _, i, j in pairs:
            if -i not in took and -j not in used:
                took[-i] = -j
                used.add(-j)
        for i, g in enumerate(mine):
            new = {'area': g['area'], 'grade': g['grade'], 'by': 'owner',
                   'why': f"The owner (domain expert; this grade comes first): {g['why']} [{g['point']}]"}
            if i in took:
                old = same[took[i]]
                for k in ('grade', 'why', 'better', 'cause', 'logical', 'logical_why'):
                    if k in old:
                        old[f'reviewer_{k}'] = old.pop(k)
                old.update(new)
            else:
                rnd['decisions'].append(new)
    return out


def note_disagreements(role, tag, summary):
    """A role's "owner_disagreements" (its final JSON) -> agent/expert/disagreements.jsonl. -> how many."""
    items = (summary or {}).get('owner_disagreements') if isinstance(summary, dict) else None
    items = [d for d in items or [] if isinstance(d, dict) and str(d.get('why') or '').strip()]
    if items:
        DISAGREEMENTS.parent.mkdir(parents=True, exist_ok=True)
        with open(DISAGREEMENTS, 'a', encoding='utf-8') as f:
            for d in items:
                f.write(json.dumps({'when': datetime.now().strftime('%Y-%m-%d %H:%M'), 'role': role, 'session': tag,
                                    'point': d.get('point'), 'why': d['why'], 'status': 'open'},
                                   ensure_ascii=False) + '\n')
    return len(items)


def disagreements(status='open'):
    if not DISAGREEMENTS.exists():
        return []
    rs = [json.loads(l) for l in DISAGREEMENTS.read_text(encoding='utf-8').splitlines() if l.strip()]
    return [r for r in rs if status is None or r.get('status') == status]


if __name__ == '__main__':
    if len(sys.argv) >= 3 and sys.argv[1] == 'add':
        print(add(json.loads(Path(sys.argv[2]).read_text(encoding='utf-8'))))
    elif len(sys.argv) >= 4 and sys.argv[1] == 'change':
        print(change(sys.argv[2], sys.argv[3])['id'])
    else:
        for e in rows():
            print(f"{e['id']} {e.get('when')} {where(e, {})}")
            for p in e.get('points') or []:
                ch = p.get('changes') or []
                print(f"  {p['id']} {p['kind']:<9} {'changed' if ch else 'OPEN   '} {p['text'][:100]}")
                for c in ch:
                    print(f"      - {c['what'][:140]}")
        for d in disagreements():
            print(f"disagreement ({d['role']}, {d['when']}) on {d.get('point')}: {d['why'][:200]}")
